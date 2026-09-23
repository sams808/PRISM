"""Headless SAXS 2.0 project pipeline: one JSON configuration, seven steps, a manifest.

    inventory -> reduce -> describe -> fit -> robustness -> figures -> tables

Each step reads the previous step's files from the output folder, so any step can be rerun alone
(``run(config, steps=["tables"])``). The configuration (:class:`ProjectConfig`) holds the frame
roots and patterns, the pairing rule, the samples table (mu or compositions), the option overrides
of every module, the series table and the figure / table specifications. ``manifest.json`` records
the code version, the options, the inputs (sizes, mtimes), the outputs and the timings of every step
run. Sample-agnostic: everything sample-specific lives in the configuration and its tables.

Layout of the output folder::

    inventory.csv, pairs.csv, unpaired.csv          (inventory)
    curves/<sample><suffix>.dat, reduction_summary.csv   (reduce)
    descriptors.csv                                 (describe)
    fits.json, fits.csv                             (fit; robustness updates them and adds fits_relaxed.csv)
    figures/<name>.pdf|svg|png                      (figures)
    tables/<base>.xlsx|csv|md                       (tables)
    manifest.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

STEPS = ("inventory", "reduce", "describe", "fit", "robustness", "figures", "tables")
Progress = Optional[Callable[[int, int, str], None]]
Log = Callable[[str], None]


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
    def template(cls, frames_roots: Optional[Sequence[str]] = None, output_dir: str = "prism_saxs_output") -> "ProjectConfig":
        return cls(frames_roots=list(frames_roots or ["path/to/session_1", "path/to/session_2"]), output_dir=output_dir,
                   samples_table="samples.csv", series_table="series.csv",
                   figures={"formats": ["pdf", "svg", "png"], "dpi": 600, "style": "Publication",
                            "grid": {"name": "Fig_curves", "shift_decades": 1.6, "x_range": [4e-3, 0.62], "q_max": 0.22, "per_decade": 60,
                                     "ylabel": "Intensity (normalized, shifted)"},
                            "fit_panels": [{"name": "Fig_fits_{group}", "group": "*", "ncol": 3, "title": "{group}"}],
                            "trends": {"name": "Fig_trends", "x_label": "x"},
                            "frame": None, "level_names": {"L1": "level 1", "L2": "level 2"}},
                   tables={"base_name": "tables", "amplitude_unit": "", "extra_columns": {}})

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
    "PRISM SAXS 2.0 project. Paths are relative to this file. Steps: inventory, reduce, describe, fit, robustness, figures, tables.",
    "samples_table: csv with 'sample' and either 'mu_mm' or 'composition' (e.g. 50.6(SiO2)13.6(B2O3)...) + 'density_g_cm3'; optional 'rho_e'.",
    "series_table: csv with sample, panel_row, panel_col, stack_index, label (+ colour, marker, linestyle, x_value, family_label, col_title, "
    "quantified, note_symbol, group, extra_panels, fit_order).",
    "reduction / descriptors / fit: option overrides (see ReductionOptions, DescriptorOptions, UnifiedOptions).",
    "figures.fit_panels: entries with 'samples' (a list) or 'group' ('*' = every group of the series table); '{group}' in names and titles.",
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


def _mu_source(cfg: ProjectConfig):
    from .sector_batch import MuSource

    df = _samples_df(cfg)
    if df is None:
        return None
    return MuSource.from_samples_table(df, key_regex=cfg.sample_key_regex, energy_ev=cfg.energy_ev)


def _curves_dir(cfg: ProjectConfig) -> str:
    return os.path.join(cfg.out, "curves")


def _load_curves(cfg: ProjectConfig):
    from .sector_batch import load_reduced_folder

    folder = _curves_dir(cfg)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder} does not exist: run the 'reduce' step first")
    reduced = load_reduced_folder(folder, pattern=r".*\.dat$")
    if not reduced:
        raise FileNotFoundError(f"no reduced curve in {folder}: run the 'reduce' step first")
    return {r.name: r for r in reduced}


def _series_table(cfg: ProjectConfig):
    from .series_figures import SeriesTable

    p = cfg.resolve(cfg.series_table)
    if not p:
        raise ValueError("the configuration has no series_table: figures and tables need one")
    return SeriesTable.from_csv(p)


def _load_fits(cfg: ProjectConfig):
    from .unified_levels import load_results_json

    p = os.path.join(cfg.out, "fits.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} does not exist: run the 'fit' step first")
    return load_results_json(p)


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
    from .sector_batch import ReductionOptions, reduce_pairs, summary_table

    pairs = _read_pairs(cfg)
    opts = ReductionOptions.from_dict(cfg.reduction)
    mu = _mu_source(cfg)
    errors = []
    results = reduce_pairs(pairs, opts, mu, out_dir=_curves_dir(cfg), progress=progress,
                           on_error=lambda p, e: errors.append(f"{p.sample.name}: {type(e).__name__}: {e}"))
    for e in errors:
        log(f"  failed: {e}")
    summary = os.path.join(cfg.out, "reduction_summary.csv")
    summary_table(results).to_csv(summary, index=False, encoding="utf-8-sig")
    for r in results:
        log(f"  {r.name:20s} T={r.transmission.T:.3g} mu t={r.mu_t:.2f} {r.loading_class:10s} usable [{r.q_min_use}, {r.q_max_use}] centre: {r.centre_source}")
    inputs = [os.path.join(cfg.out, "pairs.csv")] + ([cfg.resolve(cfg.samples_table)] if cfg.samples_table else [])
    return {"outputs": [r.dat_path for r in results] + [summary], "inputs": inputs, "options": opts.to_dict(),
            "notes": {"n_curves": len(results), "failed": errors, "classes": {c: int(sum(r.loading_class == c for r in results))
                                                                                for c in ("measurable", "marginal", "opaque", "unreliable")}}}


def step_describe(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .descriptors import DescriptorOptions, describe_batch

    reduced = _load_curves(cfg)
    opts = DescriptorOptions.from_dict(cfg.descriptors)
    mu = _mu_source(cfg)
    rho = (lambda n: mu.rho_e_for(n)) if mu is not None else None
    curves = [r.usable_curve() for r in reduced.values()]
    _results, df = describe_batch(curves, opts, rho_e=rho, progress=progress)
    path = os.path.join(cfg.out, "descriptors.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  {len(df)} curves described -> {path}")
    return {"outputs": [path], "inputs": [r.dat_path for r in reduced.values()], "options": opts.to_dict(),
            "notes": {"n_with_maximum": int(df["has_maximum"].sum()) if "has_maximum" in df else 0}}


def step_fit(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .unified_levels import UnifiedOptions, batch_fit, save_results_json

    reduced = _load_curves(cfg)
    opts = UnifiedOptions.from_dict(cfg.fit)
    curves = [r.usable_curve() for r in reduced.values()]
    results, df = batch_fit(curves, opts, robustness=False, progress=progress)
    jpath, cpath = os.path.join(cfg.out, "fits.json"), os.path.join(cfg.out, "fits.csv")
    save_results_json(results, jpath)
    df.to_csv(cpath, index=False, encoding="utf-8-sig")
    for r in results:
        log(f"  {r.sample:20s} " + (f"{r.name:8s} par={r.n_params} rms={r.rms_log:.3f}" + (f" Rg2={r.derived['Rg2_nm']:.1f} nm" if r.l2 else "")
                                    if r.fitted else f"skipped: {r.skip_reason}"))
    return {"outputs": [jpath, cpath], "inputs": [r.dat_path for r in reduced.values()], "options": opts.to_dict(),
            "notes": {"n_fitted": int(sum(r.fitted for r in results)), "n_l2": int(sum(r.l2 for r in results)),
                      "n_tail": int(sum(r.tail for r in results))}}


def step_robustness(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from .tables import robustness_sheet
    from .unified_levels import UnifiedOptions, results_table, robustness_pass, save_results_json

    reduced = _load_curves(cfg)
    results = _load_fits(cfg)
    opts = UnifiedOptions.from_dict(cfg.fit)
    n = len(results)
    for i, r in enumerate(results):
        if progress is not None:
            progress(i, n, r.sample)
        if r.fitted and r.sample in reduced:
            robustness_pass(reduced[r.sample].usable_curve(), r, opts)
    if progress is not None:
        progress(n, n, "done")
    jpath, cpath, rpath = os.path.join(cfg.out, "fits.json"), os.path.join(cfg.out, "fits.csv"), os.path.join(cfg.out, "fits_relaxed.csv")
    save_results_json(results, jpath)
    results_table(results).to_csv(cpath, index=False, encoding="utf-8-sig")
    robustness_sheet(results).to_csv(rpath, index=False, encoding="utf-8-sig")
    for r in results:
        if r.fitted and r.l2:
            log(f"  {r.sample:20s} Rg2 {r.derived['Rg2_nm']:.1f} nm, relaxed {r.relaxed.derived.get('Rg2_nm', float('nan')):.1f} nm -> "
                f"{'resolved' if r.resolved else 'not resolved'}")
    return {"outputs": [jpath, cpath, rpath], "inputs": [jpath], "options": opts.relaxed().to_dict(),
            "notes": {"n_resolved": int(sum(bool(r.resolved) for r in results))}}


def _fmt(text: str, **kw) -> str:
    try:
        return str(text).format(**kw)
    except (KeyError, IndexError):
        return str(text)


def step_figures(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from . import series_figures as sf

    spec = dict(cfg.figures or {})
    table = _series_table(cfg)
    reduced = _load_curves(cfg)
    curves = {n: r.usable_curve() for n, r in reduced.items()}
    fig_dir = os.path.join(cfg.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    formats = tuple(spec.get("formats", ("pdf", "svg", "png")))
    dpi = int(spec.get("dpi", 600))
    style = spec.get("style", "Publication")
    outputs: List[str] = []
    jobs: List[str] = []
    grid = spec.get("grid", {"name": "Fig_curves"})
    if grid:
        jobs.append("grid")
    fit_specs = spec.get("fit_panels", [{"name": "Fig_fits_{group}", "group": "*", "ncol": 3, "title": "{group}"}])
    jobs += ["fit_panels"] * len(fit_specs)
    if spec.get("trends", {"name": "Fig_trends"}):
        jobs.append("trends")
    if spec.get("frame"):
        jobs.append("frame")
    n_jobs, k = len(jobs), 0

    def tick(msg):
        nonlocal k
        if progress is not None:
            progress(k, n_jobs, msg)
        k += 1

    if grid:
        tick(grid.get("name", "Fig_curves"))
        g = dict(grid)
        name = g.pop("name", "Fig_curves")
        if "x_range" in g:
            g["x_range"] = tuple(g["x_range"])
        if g.get("figsize"):
            g["figsize"] = tuple(g["figsize"])
        fig = sf.grid_figure(curves, table, style=style, **g)
        outputs += sf.export_figure(fig, os.path.join(fig_dir, name), formats, dpi)
    fits = None
    if fit_specs or spec.get("trends", True):
        try:
            fits = {r.sample: r for r in _load_fits(cfg)}
        except FileNotFoundError as exc:
            log(f"  fit figures skipped: {exc}")
    level_names = spec.get("level_names")
    class_notes = spec.get("class_notes")
    if fits:
        for fs in fit_specs:
            fs = dict(fs)
            name = fs.pop("name", "Fig_fits_{group}")
            groups = [None]
            if "group" in fs:
                g = fs.pop("group")
                groups = table.groups() if g in ("*", None) else [g]
            for grp in groups:
                samples = fs.get("samples") or (table.samples_in_group(grp) if grp is not None else table.samples)
                title = fs.get("title")
                tick(_fmt(name, group=grp or ""))
                fig = sf.fit_panels_figure(fits, samples, table, ncol=int(fs.get("ncol", 3)), letters=fs.get("letters", True),
                                           title=(_fmt(title, group=grp or "") if title else None), style=style, level_names=level_names,
                                           class_notes=class_notes, ylabel=fs.get("ylabel", spec.get("ylabel_fits", "Intensity")),
                                           x_range=tuple(fs.get("x_range", (4.5e-3, 0.22))))
                outputs += sf.export_figure(fig, os.path.join(fig_dir, _fmt(name, group=grp or "")), formats, dpi)
    trends = spec.get("trends", {"name": "Fig_trends"})
    if trends:
        tick(trends.get("name", "Fig_trends"))
        desc_path = os.path.join(cfg.out, "descriptors.csv")
        desc = pd.read_csv(desc_path, encoding="utf-8-sig") if os.path.exists(desc_path) else None
        fits_df = pd.read_csv(os.path.join(cfg.out, "fits.csv"), encoding="utf-8-sig") if os.path.exists(os.path.join(cfg.out, "fits.csv")) else None
        panels = None
        if trends.get("panels"):
            panels = [sf.TrendPanel(**{**p, "ylim": tuple(p["ylim"]) if p.get("ylim") else None,
                                       "require": tuple(p["require"]) if p.get("require") else None,
                                       "hlines": tuple(p.get("hlines", ()))}) for p in trends["panels"]]
        fig = sf.trends_figure(table, desc, fits_df, panels=panels, x_label=trends.get("x_label", "x"), style=style,
                               figsize=tuple(trends["figsize"]) if trends.get("figsize") else None)
        outputs += sf.export_figure(fig, os.path.join(fig_dir, trends.get("name", "Fig_trends")), formats, dpi)
    frame = spec.get("frame")
    if frame:
        tick(frame.get("name", "Fig_frames"))
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
    return {"outputs": outputs, "inputs": [cfg.resolve(cfg.series_table)] + [r.dat_path for r in reduced.values()], "options": spec}


def _extra_columns(cfg: ProjectConfig, table, samples_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, object]]:
    """``{"column title": "series:<column>[:<decimals>]" | "samples:<column>[:<decimals>]"}``."""
    out: Dict[str, Dict[str, object]] = {}
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

        if kind == "series" and col in table.df.columns:
            out[title] = {s: fmt(table.get(s, col)) for s in table.samples}
        elif kind == "samples" and samples_df is not None and col in samples_df.columns:
            out[title] = {str(s): fmt(v) for s, v in zip(samples_df["sample"], samples_df[col])}
    return out


def step_tables(cfg: ProjectConfig, progress: Progress = None, log: Log = print) -> Dict[str, object]:
    from . import tables as tb

    spec = dict(cfg.tables or {})
    table = _series_table(cfg)
    results = _load_fits(cfg)
    by = {r.sample: r for r in results}
    ordered = [by[n] for n in table.samples if n in by] + [r for r in results if r.sample not in table.df.index]
    summary_path = os.path.join(cfg.out, "reduction_summary.csv")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig") if os.path.exists(summary_path) else None
    desc_path = os.path.join(cfg.out, "descriptors.csv")
    desc = pd.read_csv(desc_path, encoding="utf-8-sig") if os.path.exists(desc_path) else None
    extra = _extra_columns(cfg, table, _samples_df(cfg))
    sheets = {spec.get("fits_sheet", "fits"): tb.fit_sheet(ordered, summary, extra=extra, amplitude_unit=spec.get("amplitude_unit", ""))}
    if desc is not None:
        d = desc.set_index("sample", drop=False)
        d = d.loc[[n for n in table.samples if n in d.index] + [n for n in d.index if n not in table.df.index]]
        sheets[spec.get("modelfree_sheet", "model-free")] = tb.modelfree_sheet(d.reset_index(drop=True), extra=extra)
    sheets[spec.get("robustness_sheet", "robustness")] = tb.robustness_sheet(ordered)
    from .unified_levels import UnifiedOptions
    notes = spec.get("notes") or tb.default_notes(UnifiedOptions.from_dict(cfg.fit).to_dict())
    tdir = os.path.join(cfg.out, "tables")
    paths = tb.write_tables(sheets, os.path.join(tdir, spec.get("base_name", "tables")), notes=notes,
                            formats=tuple(spec.get("formats", ("xlsx", "csv", "md"))))
    log(f"  tables -> {tdir}: {sorted(paths)}")
    return {"outputs": list(paths.values()), "inputs": [os.path.join(cfg.out, "fits.json"), summary_path, desc_path], "options": spec}


STEP_FUNCTIONS = {"inventory": step_inventory, "reduce": step_reduce, "describe": step_describe, "fit": step_fit,
                  "robustness": step_robustness, "figures": step_figures, "tables": step_tables}


def run(config: ProjectConfig, steps: Sequence[str] = STEPS, progress: Progress = None, log: Log = print) -> Manifest:
    """Run the requested steps in pipeline order and record them in the manifest."""
    bad = [s for s in steps if s not in STEP_FUNCTIONS]
    if bad:
        raise ValueError(f"unknown steps {bad}; known: {list(STEPS)}")
    os.makedirs(config.out, exist_ok=True)
    manifest = Manifest(config.out)
    for step in [s for s in STEPS if s in steps]:
        log(f"[{step}]")
        t0 = time.time()
        info = STEP_FUNCTIONS[step](config, progress, log)
        manifest.record(step, config, info.get("options", {}), info.get("inputs", []), info.get("outputs", []), time.time() - t0,
                        info.get("notes"))
        log(f"  done in {time.time() - t0:.1f} s")
    return manifest
