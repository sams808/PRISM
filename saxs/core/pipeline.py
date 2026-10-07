"""Headless SAXS project pipeline: one JSON configuration, seven steps, a manifest.

    inventory -> reduce -> describe -> fit -> robustness -> figures -> tables

Each step reads the previous step's files from the output folder, so any step can be rerun alone
(``run(config, steps=["tables"])``). Two ways in:

* raw 2D frames of a beamstopless instrument: ``frames_roots`` + pairing rules; ``inventory`` and ``reduce`` write the
  reduced curves (sector reduction, loading classes, per-mm normalisation);
* curves reduced elsewhere: a ``curves`` block (``{"dir": ..., "pattern": "*.dat", "name_regex": "^(.*)_merged$"}``,
  named-column tables with q, I, sigma and optionally dQ, statistical sigma and smearing flags); start at ``describe``
  or ``fit``.

``fit`` runs the physical fitting workflow (saxs.core.fit_workflow) with the references, contrasts and tags defined in
the ``fit`` block; ``robustness`` refits the retained models on perturbed data; ``figures`` and ``tables`` write the
publication outputs. ``manifest.json`` records the code version, the options, the inputs (sizes, mtimes), the outputs
and the timings of every step run. Sample-agnostic: everything sample-specific lives in the configuration and its
tables.

Layout of the output folder::

    inventory.csv, pairs.csv, unpaired.csv                (inventory)
    curves/<sample><suffix>.dat, reduction_summary.csv    (reduce)
    curves_sasview/<sample>__sasview.dat                  (reduce: three-column files for SasView / external fit engines)
    descriptors.csv                                       (describe)
    fits.json, fits_summary.csv, fits/<curve>_fit.csv     (fit; robustness updates them and adds fits_robustness.csv)
    figures/<name>.pdf|svg|png                            (figures)
    tables/<base>.xlsx|csv|md                             (tables)
    manifest.json
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STEPS = ("inventory", "reduce", "describe", "fit", "robustness", "figures", "tables")
Progress = Optional[Callable[[int, int, str], None]]
Log = Callable[[str], None]
FIT_KEYS = ("options", "references", "contrast", "tags", "robustness_variants")


# ------------------------------------------------------------------------------------ config
@dataclass
class ProjectConfig:
    name: str = "saxs_project"
    frames_roots: List[str] = field(default_factory=list)
    empty_pattern: str = r"(?i)empty"
    calibrant_pattern: str = r"(?i)agbeh"
    exclude_pattern: Optional[str] = None
    name_from: str = "stem"
    default_wavelength_A: Optional[float] = None
    pairing: Dict[str, object] = field(default_factory=lambda: {"rule": "same_folder", "explicit": None,
                                                                "fallback_to_session": True, "default_capillary_mm": 1.0})
    curves: Optional[Dict[str, object]] = None
    samples_table: Optional[str] = None
    sample_key_regex: Optional[str] = None
    energy_ev: Optional[float] = None
    output_dir: str = "prism_saxs_output"
    reduction: Dict[str, object] = field(default_factory=dict)
    descriptors: Dict[str, object] = field(default_factory=dict)
    fit: Dict[str, object] = field(default_factory=dict)
    series_table: Optional[str] = None
    figures: Dict[str, object] = field(default_factory=dict)
    tables: Dict[str, object] = field(default_factory=dict)
    base_dir: str = ""                      # folder the relative paths are resolved against (the config's)

    # -- io ----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("base_dir", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object], base_dir: str = "") -> "ProjectConfig":
        d = {k: v for k, v in dict(d).items() if not k.startswith("_")}
        unknown = [k for k in d if k not in cls.__dataclass_fields__]
        if unknown:
            raise KeyError(f"unknown configuration keys {unknown}")
        cfg = cls(**d)
        cfg.base_dir = base_dir
        pairing = {"rule": "same_folder", "explicit": None, "fallback_to_session": True, "default_capillary_mm": 1.0}
        pairing.update(cfg.pairing or {})
        cfg.pairing = pairing
        bad_fit = [k for k in (cfg.fit or {}) if k not in FIT_KEYS]
        if bad_fit:
            raise KeyError(f"unknown 'fit' keys {bad_fit}; known: {list(FIT_KEYS)} (workflow options go under fit.options)")
        return cfg

    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d, base_dir=os.path.dirname(os.path.abspath(path)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"_comment": TEMPLATE_COMMENT, **self.to_dict()}, fh, indent=2)

    @classmethod
    def template(cls, frames_roots: Optional[Sequence[str]] = None, output_dir: str = "prism_saxs_output",
                 curves_dir: Optional[str] = None) -> "ProjectConfig":
        """A template to edit: from raw frames (``frames_roots``) or from reduced curves (``curves_dir``)."""
        curves = None
        if curves_dir is not None:
            curves = {"dir": curves_dir, "pattern": "*.dat", "name_regex": None}
        return cls(frames_roots=[] if curves_dir is not None else list(frames_roots or ["path/to/session_1", "path/to/session_2"]),
                   curves=curves, output_dir=output_dir, samples_table="samples.csv", series_table="series.csv",
                   fit={"options": {"floor": 0.03, "dbic_min": 10.0},
                        "references": {"pattern": "^(?P<family>.*)_with_minority$", "replace": r"\g<family>_reference"},
                        "contrast": None, "tags": {}, "robustness_variants": ["stat", "qmax", "noedge", "nosmear", "offset"]},
                   figures={"formats": ["pdf", "svg", "png"], "dpi": 600, "style": "Publication",
                            "grid": {"name": "Fig_curves", "shift_decades": 1.6, "x_range": [4e-3, 0.62], "q_max": 0.22, "per_decade": 60,
                                     "ylabel": "Intensity (normalized, shifted)"},
                            "fit_panels": [{"name": "Fig_fits_{group}", "group": "*", "ncol": 3, "title": "{group}", "iq4": False}],
                            "trends": {"name": "Fig_trends", "x_label": "x"},
                            "frame": None},
                   tables={"base_name": "tables", "extra_columns": {}})

    # -- helpers -------------------------------------------------------------------------
    def resolve(self, path: Optional[str]) -> Optional[str]:
        if path is None or path == "":
            return None
        p = os.path.expanduser(str(path))
        if os.path.isabs(p):
            return p
        return os.path.abspath(os.path.join(self.base_dir or os.getcwd(), p))

    @property
    def out(self) -> str:
        return self.resolve(self.output_dir)


TEMPLATE_COMMENT = [
    "PRISM SAXS project. Paths are relative to this file. Steps: inventory, reduce, describe, fit, robustness, figures, tables.",
    "From raw 2D frames: frames_roots + pairing. From curves reduced elsewhere: curves = {'dir', 'pattern' (glob), 'name_regex' "
    "(group 1 = curve name), optional q_col / i_col / sigma_col / dq_col}; then run from 'describe' or 'fit'.",
    "samples_table: csv with 'sample' and either 'mu_mm' or 'composition' (e.g. 50.6(SiO2)13.6(B2O3)...) + 'density_g_cm3'.",
    "fit.options: WorkflowOptions overrides (floor, grain_p_bounds, oz_xi_bounds, gate_window, norm_qmax, gate_span, gate_null_max, "
    "families, second_population, dbic_min, visibility_fraction, visibility_qmax, report_q, robust_qmax, offset_variant ...).",
    "fit.references: {'explicit': {curve: reference}} and/or {'pattern': regex, 'replace': replacement} (re.sub on the full "
    "name) and/or {'column': samples-table column}; a reference must be one of the curves.",
    "fit.contrast: {'phase': formula, 'density_g_cm3': rho, 'alt_density_g_cm3': rho2, 'composition_column', 'density_column'} "
    "(particle contrast and stoichiometric ceiling from each sample's composition) or {'column': samples-table column, 1e10 cm^-2}.",
    "fit.tags: {tag: 'header:<regex with one group>' or 'samples:<column>'} (used by the 'offset' robustness variant).",
    "series_table: csv with sample, panel_row, panel_col, stack_index, label (+ colour, marker, linestyle, x_value, family_label, "
    "col_title, quantified, note_symbol, group, extra_panels, fit_order); needed for the grid and trends figures.",
    "figures.fit_panels: entries with 'samples' (a list) or 'group' ('*' = every group of the series table), 'only': "
    "'structural' | 'all', 'iq4': true for I q^4 panels; '{group}' in names and titles.",
    "tables.extra_columns: {'column title': 'series:<column>[:<decimals>]' or 'samples:<column>[:<decimals>]'}.",
]


# ------------------------------------------------------------------------------------ manifest
def _file_info(path: str) -> Dict[str, object]:
    try:
        st = os.stat(path)
        return {"path": path, "size": int(st.st_size), "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
    except OSError:
        return {"path": path, "size": None, "mtime": None}


def code_version() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        from core.qt_help import APP_VERSION
        info["prism"] = APP_VERSION
    except Exception:
        info["prism"] = "unknown"
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        info["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        info["git"] = ""
    return info


class Manifest:
    def __init__(self, out_dir: str):
        self.path = os.path.join(out_dir, "manifest.json")
        self.data: Dict[str, object] = {"format": "prism-saxs-manifest", "version": 1, "steps": {}}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                pass

    def record(self, step: str, config: ProjectConfig, options: Dict[str, object], inputs: Sequence[str], outputs: Sequence[str],
               seconds: float, notes: Optional[Dict[str, object]] = None) -> None:
        self.data["code"] = code_version()
        self.data["config"] = config.to_dict()
        self.data["config_dir"] = config.base_dir
        self.data.setdefault("steps", {})[step] = {
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(float(seconds), 2), "options": options,
            "inputs": [_file_info(p) for p in inputs], "outputs": [_file_info(p) for p in outputs], "notes": notes or {}}
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=1, default=str)


# ------------------------------------------------------------------------------------ helpers
def _samples_df(cfg: ProjectConfig) -> Optional[pd.DataFrame]:
    p = cfg.resolve(cfg.samples_table)
    return pd.read_csv(p, encoding="utf-8-sig") if p else None


def _sample_key(cfg: ProjectConfig, name: str) -> str:
    """The samples-table key of a curve name (``sample_key_regex`` group 1, else the name itself)."""
    if cfg.sample_key_regex:
        m = re.search(cfg.sample_key_regex, name)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return name


def _mu_source(cfg: ProjectConfig):
    from .sector_batch import MuSource

    df = _samples_df(cfg)
    if df is None:
        return None
    return MuSource.from_samples_table(df, key_regex=cfg.sample_key_regex, energy_ev=cfg.energy_ev)


def _curves_dir(cfg: ProjectConfig) -> str:
    return os.path.join(cfg.out, "curves")


def _load_reduced(cfg: ProjectConfig):
    from .sector_batch import load_reduced_folder

    folder = _curves_dir(cfg)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder} does not exist: run the 'reduce' step first (or define a 'curves' block)")
    reduced = load_reduced_folder(folder, pattern=r".*\.dat$")
    if not reduced:
        raise FileNotFoundError(f"no reduced curve in {folder}: run the 'reduce' step first")
    return {r.name: r for r in reduced}


def _curve_set(cfg: ProjectConfig, log: Log = print) -> Tuple[Dict[str, object], List[str]]:
    """``{name: Curve}`` and the input paths: the external ``curves`` block when defined, else the reduce output."""
    from .loader import load_tabular_curve

    if cfg.curves:
        spec = dict(cfg.curves)
        folder = cfg.resolve(spec.get("dir"))
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError(f"curves.dir {folder!r} does not exist")
        pattern = str(spec.get("pattern", "*.dat"))
        name_re = spec.get("name_regex")
        exclude = spec.get("exclude_regex")
        curves, paths = {}, []
        for fn in sorted(os.listdir(folder)):
            if not fnmatch.fnmatch(fn, pattern):
                continue
            if exclude and re.search(str(exclude), fn):
                continue
            stem = os.path.splitext(fn)[0]
            name = stem
            if name_re:
                m = re.search(str(name_re), stem)
                if not m:
                    continue
                name = m.group(1) if m.groups() else m.group(0)
            path = os.path.join(folder, fn)
            c = load_tabular_curve(path, q_col=spec.get("q_col"), i_col=spec.get("i_col"), sigma_col=spec.get("sigma_col"),
                                   dq_col=spec.get("dq_col"))
            c.name = name
            if c.metadata.get("n_dropped"):
                log(f"  {name}: {c.metadata['n_dropped']} row(s) dropped (non-finite or q <= 0)")
            curves[name] = c
            paths.append(path)
        if not curves:
            raise FileNotFoundError(f"no file matching {pattern!r} (name_regex {name_re!r}) in {folder}")
        return curves, paths
    reduced = _load_reduced(cfg)
    return {n: r.usable_curve() for n, r in reduced.items()}, [r.dat_path for r in reduced.values()]


def _series_table(cfg: ProjectConfig, required: bool = True):
    from .series_figures import SeriesTable

    p = cfg.resolve(cfg.series_table)
    if not p or not os.path.exists(p):
        if required:
            raise ValueError("the configuration has no series_table (or the file is missing)")
        return None
    return SeriesTable.from_csv(p)


def _fits_path(cfg: ProjectConfig) -> str:
    return os.path.join(cfg.out, "fits.json")


def _load_fits(cfg: ProjectConfig):
    from .fit_workflow import WorkflowResult

    p = _fits_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} does not exist: run the 'fit' step first")
    return WorkflowResult.load(p)


def _read_pairs(cfg: ProjectConfig):
    from .frames import FramePair, frame_record

    p = os.path.join(cfg.out, "pairs.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} does not exist: run the 'inventory' step first")
    df = pd.read_csv(p, encoding="utf-8-sig")
    cache = {}
    pairs = []
    roots = [cfg.resolve(r) for r in cfg.frames_roots]

    def rec(path, root):
        if path not in cache:
            cache[path] = frame_record(path, root, cfg.empty_pattern, cfg.calibrant_pattern, cfg.name_from, diagnostics=False,
                                       default_wavelength_A=cfg.default_wavelength_A)
        return cache[path]

    for _, r in df.iterrows():
        root = next((x for x in roots if os.path.abspath(str(r["sample_path"])).startswith(os.path.abspath(x))), None)
        s = rec(str(r["sample_path"]), root)
        e = rec(str(r["empty_path"]), root)
        s.name = str(r["sample"])
        pairs.append(FramePair(sample=s, empty=e))
    return pairs


# ------------------------------------------------------------------------------------ fit inputs
def references_for(cfg: ProjectConfig, names: Sequence[str], log: Log = print) -> Dict[str, Optional[str]]:
    """Reference curve of every curve from ``fit.references`` (explicit pairs win over the regex rule, which wins over
    a samples-table column). A reference equal to the curve itself, or not among the curves, is dropped (logged)."""
    spec = dict((cfg.fit or {}).get("references") or {})
    out: Dict[str, Optional[str]] = {n: None for n in names}
    if spec.get("column"):
        df = _samples_df(cfg)
        if df is None or spec["column"] not in df.columns:
            raise KeyError(f"fit.references.column {spec['column']!r} not in the samples table")
        col = {str(k): v for k, v in zip(df["sample"], df[spec["column"]])}
        for n in names:
            v = col.get(_sample_key(cfg, n))
            out[n] = None if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "") else str(v)
    if spec.get("pattern"):
        pat = re.compile(str(spec["pattern"]))
        for n in names:
            if pat.fullmatch(n):
                out[n] = pat.sub(str(spec.get("replace", "")), n)
    for k, v in dict(spec.get("explicit") or {}).items():
        if k in out:
            out[k] = v
    for n, r in list(out.items()):
        if r == n:
            out[n] = None
        elif r is not None and r not in out:
            log(f"  {n}: reference {r!r} is not among the curves -> no reference")
            out[n] = None
    return out


def contrasts_for(cfg: ProjectConfig, names: Sequence[str], log: Log = print) -> Dict[str, Dict[str, Optional[float]]]:
    """Particle contrast (and an alternative, and the stoichiometric ceiling) per curve from ``fit.contrast``."""
    spec = (cfg.fit or {}).get("contrast")
    out = {n: {"contrast": None, "contrast_alt": None, "phi_max": None} for n in names}
    if not spec:
        return out
    df = _samples_df(cfg)
    if df is None:
        raise ValueError("fit.contrast needs a samples_table")
    rows = {str(k): r for k, r in df.set_index("sample", drop=False).iterrows()}
    if spec.get("column"):
        for n in names:
            r = rows.get(_sample_key(cfg, n))
            if r is not None and pd.notna(r.get(spec["column"])):
                out[n]["contrast"] = float(r[spec["column"]])
        return out
    from .contrast import phase_contrast_1e10cm2, stoichiometric_ceiling

    phase, rho_p = spec["phase"], float(spec["density_g_cm3"])
    rho_alt = spec.get("alt_density_g_cm3")
    ccol, dcol = spec.get("composition_column", "composition"), spec.get("density_column", "density_g_cm3")
    for n in names:
        r = rows.get(_sample_key(cfg, n))
        if r is None or pd.isna(r.get(ccol)) or pd.isna(r.get(dcol)):
            log(f"  {n}: no composition/density in the samples table -> spheres on a relative scale")
            continue
        comp, rho_s = str(r[ccol]), float(r[dcol])
        out[n]["contrast"] = phase_contrast_1e10cm2(phase, rho_p, comp, rho_s)
        if rho_alt:
            out[n]["contrast_alt"] = phase_contrast_1e10cm2(phase, float(rho_alt), comp, rho_s)
        if spec.get("ceiling", True):
            out[n]["phi_max"] = stoichiometric_ceiling(phase, rho_p, comp, rho_s)
    return out


def tags_for(cfg: ProjectConfig, curves: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """``fit.tags``: {tag: 'header:<regex with one group>' | 'samples:<column>'}."""
    spec = dict((cfg.fit or {}).get("tags") or {})
    out = {n: {} for n in curves}
    df = _samples_df(cfg) if any(str(v).startswith("samples:") for v in spec.values()) else None
    rows = {str(k): r for k, r in df.set_index("sample", drop=False).iterrows()} if df is not None else {}
    for tag, src in spec.items():
        kind, _, arg = str(src).partition(":")
        for n, c in curves.items():
            if kind == "header":
                for h in getattr(c, "header_lines", []) or []:
                    m = re.search(arg, h)
                    if m:
                        out[n][tag] = m.group(1) if m.groups() else m.group(0)
                        break
            elif kind == "samples":
                r = rows.get(_sample_key(cfg, n))
                if r is not None and arg in r and pd.notna(r[arg]):
                    out[n][tag] = str(r[arg])
    return out


def fit_inputs(cfg: ProjectConfig, log: Log = print):
    """The CurveInputs of the project (curves, references, contrasts, tags) and the input paths."""
    from .fit_workflow import CurveInput

    curves, paths = _curve_set(cfg, log)
    refs = references_for(cfg, list(curves), log)
    con = contrasts_for(cfg, list(curves), log)
    tags = tags_for(cfg, curves)
    inputs = [CurveInput.from_curve(c, name=n, reference=refs[n], tags=tags[n], **con[n]) for n, c in curves.items()]
    return inputs, paths


# ------------------------------------------------------------------------------------ steps
def step_inventory(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .frames import pair_frames, scan_frames, write_inventory

    out = cfg.out
    os.makedirs(out, exist_ok=True)
    roots = [cfg.resolve(r) for r in cfg.frames_roots]
    errors = []
    records = scan_frames(roots, cfg.empty_pattern, cfg.calibrant_pattern, cfg.exclude_pattern, cfg.name_from, diagnostics=True,
                          progress=progress, on_error=lambda p, e: errors.append(f"{p}: {type(e).__name__}: {e}"),
                          default_wavelength_A=cfg.default_wavelength_A)
    for e in errors:
        log(f"  skipped: {e}")
    inv = os.path.join(out, "inventory.csv")
    write_inventory(records, inv)
    pr = cfg.pairing
    explicit = pr.get("explicit")
    if isinstance(explicit, str):
        explicit = cfg.resolve(explicit)
    pairing = pair_frames(records, rule=pr.get("rule", "same_folder"), explicit=explicit,
                          fallback_to_session=bool(pr.get("fallback_to_session", True)),
                          default_capillary_mm=float(pr.get("default_capillary_mm", 1.0)))
    t = pairing.table()
    t["sample_path"] = [p.sample.path for p in pairing.pairs]
    t["empty_path"] = [p.empty.path for p in pairing.pairs]
    pairs_csv = os.path.join(out, "pairs.csv")
    t.to_csv(pairs_csv, index=False, encoding="utf-8-sig")
    unp = os.path.join(out, "unpaired.csv")
    pd.DataFrame([{"sample": r.name, "frame": r.path} for r in pairing.unpaired], columns=["sample", "frame"]).to_csv(unp, index=False)
    dups = pairing.duplicate_names()
    for name, rels in dups.items():
        log(f"  !! duplicate sample name {name!r}: {rels} (exclude one with exclude_pattern, or their outputs get a __2 suffix)")
    log(f"  {len(records)} frames, {len(pairing.pairs)} pairs, {len(pairing.unpaired)} unpaired -> {inv}")
    return {"outputs": [inv, pairs_csv, unp], "inputs": [], "notes": {"n_frames": len(records), "n_pairs": len(pairing.pairs),
                                                                     "n_unpaired": len(pairing.unpaired), "skipped": errors,
                                                                     "duplicate_names": dups}}


def step_reduce(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .sector_batch import SASVIEW_SUFFIX, ReductionOptions, reduce_pairs, summary_table

    pairs = _read_pairs(cfg)
    opts = ReductionOptions.from_dict(cfg.reduction)
    mu = _mu_source(cfg)
    errors = []
    sasview_dir = os.path.join(cfg.out, "curves_sasview")
    results = reduce_pairs(pairs, opts, mu, out_dir=_curves_dir(cfg), progress=progress,
                           on_error=lambda p, e: errors.append(f"{p.sample.name}: {type(e).__name__}: {e}"), sasview_dir=sasview_dir)
    for e in errors:
        log(f"  failed: {e}")
    summary = os.path.join(cfg.out, "reduction_summary.csv")
    summary_table(results).to_csv(summary, index=False, encoding="utf-8-sig")
    for r in results:
        log(f"  {r.name:20s} T={r.transmission.T:.3g} mu t={r.mu_t:.2f} {r.loading_class:10s} usable [{r.q_min_use}, {r.q_max_use}] centre: {r.centre_source}")
    inputs = [os.path.join(cfg.out, "pairs.csv")] + ([cfg.resolve(cfg.samples_table)] if cfg.samples_table else [])
    sasview_files = [os.path.join(sasview_dir, f"{r.name}{SASVIEW_SUFFIX}.dat") for r in results]
    log(f"  {len(sasview_files)} SasView-ready three-column curve(s) -> {sasview_dir}")
    return {"outputs": [r.dat_path for r in results] + sasview_files + [summary], "inputs": inputs, "options": opts.to_dict(),
            "notes": {"n_curves": len(results), "failed": errors, "classes": {c: int(sum(r.loading_class == c for r in results))
                                                                                for c in ("measurable", "marginal", "opaque", "unreliable")}}}


def step_describe(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .descriptors import DescriptorOptions, describe_batch

    curves, paths = _curve_set(cfg, log)
    opts = DescriptorOptions.from_dict(cfg.descriptors)
    mu = _mu_source(cfg)
    rho = (lambda n: mu.rho_e_for(n)) if mu is not None else None
    _results, df = describe_batch(list(curves.values()), opts, rho_e=rho, progress=progress)
    path = os.path.join(cfg.out, "descriptors.csv")
    os.makedirs(cfg.out, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  {len(df)} curves described -> {path}")
    return {"outputs": [path], "inputs": paths, "options": opts.to_dict(),
            "notes": {"n_with_maximum": int(df["has_maximum"].sum()) if "has_maximum" in df else 0}}


def _write_curve_fits(result, folder: str) -> List[str]:
    from .fit_workflow import curve_table
    from .tables import transliterate_frame

    os.makedirs(folder, exist_ok=True)
    out = []
    for name, cr in result.curves.items():
        p = os.path.join(folder, f"{name}_fit.csv")
        transliterate_frame(curve_table(cr)).to_csv(p, index=False, encoding="utf-8-sig")
        out.append(p)
    return out


def step_fit(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .fit_workflow import WorkflowOptions, run_workflow, summary_table

    inputs, paths = fit_inputs(cfg, log)
    opts = WorkflowOptions.from_dict((cfg.fit or {}).get("options"))
    os.makedirs(cfg.out, exist_ok=True)
    result = run_workflow(inputs, opts, progress=progress)
    jpath, cpath = _fits_path(cfg), os.path.join(cfg.out, "fits_summary.csv")
    result.save(jpath)
    summary_table(result).to_csv(cpath, index=False, encoding="utf-8-sig")
    curve_files = _write_curve_fits(result, os.path.join(cfg.out, "fits"))
    if np.isfinite(result.null_max):
        log(f"  reference/reference null maximum {result.null_max:.3g} ({result.null_pairs} pairs)")
    for name, cr in result.curves.items():
        f = cr.retained_fit
        g = cr.gate
        log(f"  {name:20s} gate {g.method}{' PASS' if g.passed else ''}{f' (peak {g.peak:.3g})' if np.isfinite(g.peak) else ''}"
            f" -> {cr.retained:12s} chi2_r {f.redchi:5.2f}")
    return {"outputs": [jpath, cpath] + curve_files, "inputs": paths, "options": opts.to_dict(),
            "notes": {"n_curves": len(result.curves), "n_structural": int(sum(c.structural for c in result.curves.values())),
                      "null_max": result.null_max}}


def step_robustness(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .fit_workflow import robust_flags, robustness, summary_table

    result = _load_fits(cfg)
    inputs, paths = fit_inputs(cfg, log)
    variants = tuple((cfg.fit or {}).get("robustness_variants") or ("stat", "qmax", "noedge", "nosmear", "offset"))
    df = robustness(result, inputs, variants, progress=progress)
    jpath, rpath, cpath = _fits_path(cfg), os.path.join(cfg.out, "fits_robustness.csv"), os.path.join(cfg.out, "fits_summary.csv")
    result.save(jpath)
    df.to_csv(rpath, index=False, encoding="utf-8-sig")
    summary_table(result).to_csv(cpath, index=False, encoding="utf-8-sig")
    for name, cr in result.curves.items():
        if cr.structural:
            fl = robust_flags(df, name)
            log(f"  {name:20s} {cr.retained:12s} primary {'robust' if fl.get('robust_primary') else 'NOT robust'}"
                + (f", second {'robust' if fl.get('robust_second') else 'NOT robust'}" if fl.get("robust_second") is not None else "")
                + (f"  [{fl.get('robustness_failures')}]" if fl.get("robustness_failures") else ""))
    return {"outputs": [jpath, rpath, cpath], "inputs": [jpath] + paths, "options": {"variants": list(variants)},
            "notes": {"n_rows": int(len(df))}}


def _fmt(text: str, **kw) -> str:
    try:
        return str(text).format(**kw)
    except (KeyError, IndexError):
        return str(text)


def step_figures(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from . import series_figures as sf

    spec = dict(cfg.figures or {})
    table = _series_table(cfg, required=False)
    curves, paths = _curve_set(cfg, log)
    fig_dir = os.path.join(cfg.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    formats = tuple(spec.get("formats", ("pdf", "svg", "png")))
    dpi = int(spec.get("dpi", 600))
    style = spec.get("style", "Publication")
    outputs: List[str] = []
    try:
        result = _load_fits(cfg)
    except FileNotFoundError as exc:
        result = None
        log(f"  fit figures skipped: {exc}")
    jobs: List[Tuple[str, object]] = []
    grid = spec.get("grid", {"name": "Fig_curves"})
    if grid and table is not None:
        jobs.append(("grid", grid))
    elif grid:
        log("  grid figure skipped: it needs a series_table")
    if result is not None:
        for fs in spec.get("fit_panels", [{"name": "Fig_fits_{group}", "group": "*", "ncol": 3, "title": "{group}"}]):
            jobs.append(("fit_panels", fs))
    trends = spec.get("trends", {"name": "Fig_trends"})
    if trends and table is not None and (result is not None or os.path.exists(os.path.join(cfg.out, "descriptors.csv"))):
        jobs.append(("trends", trends))
    if spec.get("frame"):
        jobs.append(("frame", spec["frame"]))
    n_jobs = len(jobs)
    for k, (kind, js) in enumerate(jobs):
        if progress is not None:
            progress(k, n_jobs, kind)
        if kind == "grid":
            g = dict(js)
            name = g.pop("name", "Fig_curves")
            if "x_range" in g:
                g["x_range"] = tuple(g["x_range"])
            if g.get("figsize"):
                g["figsize"] = tuple(g["figsize"])
            fig = sf.grid_figure(curves, table, style=style, **g)
            outputs += sf.export_figure(fig, os.path.join(fig_dir, name), formats, dpi)
        elif kind == "fit_panels":
            outputs += _fit_panel_figures(dict(js), result, table, fig_dir, formats, dpi, style, log)
        elif kind == "trends":
            desc_path = os.path.join(cfg.out, "descriptors.csv")
            desc = pd.read_csv(desc_path, encoding="utf-8-sig") if os.path.exists(desc_path) else None
            fits_df = None
            if result is not None:
                from .fit_workflow import summary_table
                fits_df = summary_table(result)
            panels = None
            if js.get("panels"):
                panels = [sf.TrendPanel(**{**p, "ylim": tuple(p["ylim"]) if p.get("ylim") else None,
                                           "require": tuple(p["require"]) if p.get("require") else None,
                                           "hlines": tuple(p.get("hlines", ()))}) for p in js["panels"]]
            fig = sf.trends_figure(table, desc, fits_df, panels=panels, x_label=js.get("x_label", "x"), style=style,
                                   figsize=tuple(js["figsize"]) if js.get("figsize") else None)
            outputs += sf.export_figure(fig, os.path.join(fig_dir, js.get("name", "Fig_trends")), formats, dpi)
        elif kind == "frame":
            frame = js
            sample_frame, empty_frame = frame.get("sample_frame"), frame.get("empty_frame")
            if frame.get("sample") and not (sample_frame and empty_frame):
                pairs = pd.read_csv(os.path.join(cfg.out, "pairs.csv"), encoding="utf-8-sig").set_index("sample")
                row = pairs.loc[frame["sample"]]
                sample_frame, empty_frame = row["sample_path"], row["empty_path"]
            fig = sf.frame_figure(cfg.resolve(sample_frame), cfg.resolve(empty_frame), crop_px=int(frame.get("crop_px", 45)),
                                  q_marker=frame.get("q_marker"), titles=tuple(frame.get("titles", ("sample", "empty"))), style=style,
                                  default_wavelength_A=cfg.default_wavelength_A)
            outputs += sf.export_figure(fig, os.path.join(fig_dir, frame.get("name", "Fig_frames")), formats, dpi)
    if progress is not None:
        progress(n_jobs, n_jobs, "done")
    log(f"  {len(outputs)} figure files -> {fig_dir}")
    return {"outputs": outputs, "inputs": ([cfg.resolve(cfg.series_table)] if table is not None else []) + paths, "options": spec}


def _fit_panel_figures(fs: Dict[str, object], result, table, fig_dir: str, formats, dpi: int, style: str, log: Log) -> List[str]:
    from . import series_figures as sf

    name = str(fs.pop("name", "Fig_fits_{group}"))
    only = fs.get("only", "all")
    per_figure = int(fs.get("per_figure", 12))
    pick = (lambda names: [n for n in names if n in result.curves and (only != "structural" or result.curves[n].structural)])
    batches: List[Tuple[str, List[str]]] = []
    if fs.get("samples"):
        batches.append((_fmt(name, group=""), pick(list(fs["samples"]))))
    elif table is not None and "group" in fs:
        g = fs.get("group")
        for grp in (table.groups() if g in ("*", None) else [g]):
            batches.append((_fmt(name, group=grp or ""), pick(table.samples_in_group(grp))))
    else:
        names = pick(list(result.curves))
        for i in range(0, len(names), per_figure):
            batches.append((_fmt(name, group=str(i // per_figure + 1)), names[i:i + per_figure]))
    out = []
    for fname, samples in batches:
        if not samples:
            continue
        title = fs.get("title")
        fig = sf.fit_panels_figure(result, samples, table, ncol=int(fs.get("ncol", 3)), letters=fs.get("letters", True),
                                   title=(_fmt(title, group=fname) if title else None), style=style, iq4=bool(fs.get("iq4", False)),
                                   x_range=tuple(fs["x_range"]) if fs.get("x_range") else None)
        out += sf.export_figure(fig, os.path.join(fig_dir, fname), formats, dpi)
    return out


def _extra_columns(cfg: ProjectConfig, table, samples_df: Optional[pd.DataFrame], names: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """``{curve: {column title: value}}`` from ``tables.extra_columns`` =
    ``{"column title": "series:<column>[:<decimals>]" | "samples:<column>[:<decimals>]"}``."""
    out: Dict[str, Dict[str, object]] = {n: {} for n in names}
    for title, source in (cfg.tables or {}).get("extra_columns", {}).items():
        parts = str(source).split(":")
        kind, col = parts[0], parts[1] if len(parts) > 1 else ""
        nd = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None

        def fmt(v):
            if nd is None or v is None:
                return v
            try:
                return round(float(v), nd) if np.isfinite(float(v)) else v
            except (TypeError, ValueError):
                return v

        for n in names:
            if kind == "series" and table is not None and col in table.df.columns and table.has(n):
                out[n][title] = fmt(table.get(n, col))
            elif kind == "samples" and samples_df is not None and col in samples_df.columns:
                key = _sample_key(cfg, n)
                vals = samples_df.loc[samples_df["sample"].astype(str) == key, col]
                if len(vals):
                    out[n][title] = fmt(vals.iloc[0])
    return out


def step_tables(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from . import tables as tb

    spec = dict(cfg.tables or {})
    table = _series_table(cfg, required=False)
    result = _load_fits(cfg)
    if table is not None:                       # series-table order first, then the rest
        order = [n for n in table.samples if n in result.curves] + [n for n in result.curves if not table.has(n)]
        result.curves = {n: result.curves[n] for n in order}
    desc_path = os.path.join(cfg.out, "descriptors.csv")
    desc = pd.read_csv(desc_path, encoding="utf-8-sig") if os.path.exists(desc_path) else None
    extra = _extra_columns(cfg, table, _samples_df(cfg), list(result.curves))
    tdir = os.path.join(cfg.out, "tables")
    paths = tb.write_workflow_tables(result, os.path.join(tdir, spec.get("base_name", "tables")), extra=extra, descriptors=desc,
                                     formats=tuple(spec.get("formats", ("xlsx", "csv", "md"))))
    log(f"  tables -> {tdir}: {sorted(paths)}")
    return {"outputs": list(paths.values()), "inputs": [_fits_path(cfg), desc_path], "options": spec}


STEP_FUNCTIONS = {"inventory": step_inventory, "reduce": step_reduce, "describe": step_describe, "fit": step_fit,
                  "robustness": step_robustness, "figures": step_figures, "tables": step_tables}


def run(config: ProjectConfig, steps: Sequence[str] = STEPS, progress: Progress = None, log: Log = print) -> Manifest:
    """Run the requested steps in pipeline order and record them in the manifest. With a ``curves`` block the
    inventory and reduce steps are skipped (the curves are already reduced)."""
    bad = [s for s in steps if s not in STEP_FUNCTIONS]
    if bad:
        raise ValueError(f"unknown steps {bad}; known: {list(STEPS)}")
    os.makedirs(config.out, exist_ok=True)
    manifest = Manifest(config.out)
    for step in [s for s in STEPS if s in steps]:
        if config.curves and step in ("inventory", "reduce"):
            log(f"[{step}] skipped: the project reads reduced curves (curves block)")
            continue
        log(f"[{step}]")
        t0 = time.time()
        info = STEP_FUNCTIONS[step](config, progress, log)
        manifest.record(step, config, info.get("options", {}), info.get("inputs", []), info.get("outputs", []), time.time() - t0,
                        info.get("notes"))
        log(f"  done in {time.time() - t0:.1f} s")
    return manifest
