"""Batch sector reduction of paired 2D frames with loading classes and per-mm normalisation.

Builds on :mod:`saxs.core.edf_sectors` (one pair) and :mod:`saxs.core.frames` (inventory and
pairing): reduces every (sample, empty) pair with one :class:`ReductionOptions`, attaches the
absorption coefficient of the analysed material (:class:`MuSource`: a table of mu per sample or a
composition + density evaluated with xraydb at the frame's energy), classifies the loading from the
measured transmission and its uniformity, writes one ``.dat`` per sample with the full provenance in
its header, and summarises the campaign in a table.

Units: ``curve`` is the corrected intensity in counts/s/pixel in empty-transmission units,
(S/t_s - T E/t_e)/T; ``curve_per_mm`` divides it by the effective glass thickness
t_eff = -ln T / mu, the quantity comparable between loadings.

Everything is sample-agnostic: the sample identity is a string, mu comes from the caller.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import edf_sectors as es
from .curve import Curve
from .frames import FramePair

Progress = Optional[Callable[[int, int, str], None]]
CLASSES = ("measurable", "marginal", "opaque", "unreliable")
DAT_COLUMNS = ["q_A^-1", "I_corr", "sigma_I", "I_per_mm", "sigma_per_mm", "sample_cps_px", "empty_cps_px",
               "ratio_S_over_TE", "n_pix_sample", "usable"]
CODE_VERSION = "PRISM saxs.core.sector_batch v1 (sector-aware, beam centre refined from the direct beam)"
SASVIEW_SUFFIX = "__sasview"


# ------------------------------------------------------------------------------------ options
@dataclass
class ReductionOptions:
    """Every threshold of the reduction, with the rethink's values as defaults."""
    axis_exclusion_deg: float = 20.0
    bar_rect_px: Tuple[float, float] = (40.0, 14.0)
    rmax_px: int = 650
    footprint_r_px: float = 16.0
    vertical_half_deg: float = 30.0
    refine_centre: bool = True
    max_shift_px: float = 4.0
    bright_fraction: float = 0.01
    mu_t_measurable: float = 3.0
    mu_t_marginal: float = 6.0
    heterogeneity_range: Tuple[float, float] = (0.3, 3.0)
    min_ratio: float = 1.25
    min_ratio_footprint: float = 5.0
    r_footprint_px: float = 30.0
    nsigma: float = 3.0
    min_run: int = 8
    flatfield_method: str = "single_photon"     # Poisson sigma: one photon = the single-photon value ("legacy": pre-2026-10)
    plateau_bands_px: Tuple[Tuple[int, int], ...] = ((6, 12), (12, 20))
    per_mm_probes: Tuple[Tuple[float, float, float], ...] = ((0.02, 0.019, 0.021), (0.05, 0.048, 0.052), (0.15, 0.14, 0.16))
    dat_suffix: str = "__sectors"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> "ReductionOptions":
        d = dict(d or {})
        out = cls()
        for k, v in d.items():
            if not hasattr(out, k):
                raise KeyError(f"unknown reduction option {k!r}")
            cur = getattr(out, k)
            if isinstance(cur, tuple):
                v = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in v)
            setattr(out, k, v)
        return out


def classify_loading(mu_t: float, heterogeneity: float, options: Optional[ReductionOptions] = None) -> str:
    """measurable (mu t below ``mu_t_measurable``), marginal (below ``mu_t_marginal``), opaque (above), or
    unreliable when the central-disc / footprint transmission ratio leaves ``heterogeneity_range``
    (light beside the powder column or a hole in it). The uniformity rule is applied first."""
    o = options or ReductionOptions()
    lo, hi = o.heterogeneity_range
    if not (np.isfinite(heterogeneity) and lo <= heterogeneity <= hi):
        return "unreliable"
    if not np.isfinite(mu_t):
        return "opaque"
    if mu_t < o.mu_t_measurable:
        return "measurable"
    if mu_t < o.mu_t_marginal:
        return "marginal"
    return "opaque"


# ------------------------------------------------------------------------------------ mu source
def _guess_composition_mode(text: str) -> str:
    t = text.strip()
    if re.search(r"\d\s*\(", t) or any(ch in t for ch in ":=+;|"):
        return "mixture"
    return "formula"


class MuSource:
    """Linear attenuation coefficient (1/mm) and electron density per sample name.

    Values come from a table (``name -> mu_mm``) and/or from a composition text with a density,
    evaluated with :func:`saxs.core.chemistry.calculate_absorption` at the requested energy (the
    frame's own energy unless ``energy_ev`` overrides it). ``key_regex`` (one capturing group) maps a
    frame name to the table key, e.g. ``r"^(.*?)(?:_rep\\d+)?$"`` to share one composition between
    repeats of a sample."""

    def __init__(self, table: Optional[Dict[str, float]] = None,
                 compositions: Optional[Dict[str, Tuple[str, float]]] = None,
                 key_regex: Optional[str] = None, energy_ev: Optional[float] = None,
                 rho_e: Optional[Dict[str, float]] = None, composition_basis: str = "molar"):
        self.table = {str(k): float(v) for k, v in (table or {}).items() if v is not None and np.isfinite(float(v))}
        self.compositions = {str(k): (str(t), float(d)) for k, (t, d) in (compositions or {}).items()}
        self.rho_e_table = {str(k): float(v) for k, v in (rho_e or {}).items() if v is not None and np.isfinite(float(v))}
        self.key_regex = key_regex
        self.energy_ev = energy_ev
        self.composition_basis = composition_basis
        self._cache: Dict[Tuple[str, float], Tuple[float, float]] = {}

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_table(cls, mapping: Dict[str, float], key_regex: Optional[str] = None) -> "MuSource":
        return cls(table=mapping, key_regex=key_regex)

    @classmethod
    def from_samples_table(cls, table: Union[str, pd.DataFrame], key_regex: Optional[str] = None,
                           energy_ev: Optional[float] = None, sample_column: str = "sample", mu_column: str = "mu_mm",
                           composition_column: str = "composition", density_column: str = "density_g_cm3",
                           rho_e_column: str = "rho_e", composition_basis: str = "molar") -> "MuSource":
        df = pd.read_csv(table) if isinstance(table, (str, os.PathLike)) else table
        if sample_column not in df.columns:
            raise ValueError(f"samples table needs a {sample_column!r} column")
        mu, comp, rho = {}, {}, {}
        for _, r in df.iterrows():
            key = str(r[sample_column])
            if mu_column in df.columns and pd.notna(r[mu_column]):
                mu[key] = float(r[mu_column])
            if composition_column in df.columns and density_column in df.columns and pd.notna(r[composition_column]) \
                    and pd.notna(r[density_column]):
                comp[key] = (str(r[composition_column]), float(r[density_column]))
            if rho_e_column in df.columns and pd.notna(r[rho_e_column]):
                rho[key] = float(r[rho_e_column])
        return cls(table=mu, compositions=comp, key_regex=key_regex, energy_ev=energy_ev, rho_e=rho,
                   composition_basis=composition_basis)

    # -- lookup ------------------------------------------------------------------------
    def key(self, name: str) -> str:
        if self.key_regex:
            m = re.match(self.key_regex, name)
            if m and m.groups():
                return m.group(1)
        return name

    def has(self, name: str) -> bool:
        k = self.key(name)
        return k in self.table or k in self.compositions

    def _from_composition(self, key: str, energy_ev: float) -> Tuple[float, float]:
        ck = (key, round(float(energy_ev), 3))
        if ck not in self._cache:
            from .chemistry import calculate_absorption

            text, density = self.compositions[key]
            _comp, ab = calculate_absorption(text, density, energy_ev=float(energy_ev),
                                             mode=_guess_composition_mode(text), basis=self.composition_basis)
            self._cache[ck] = (float(ab.mu_linear_mm_inv), float(ab.electron_density_cm3) * 1e-24)
        return self._cache[ck]

    def mu_for(self, name: str, energy_ev: Optional[float] = None) -> Optional[float]:
        k = self.key(name)
        if k in self.table:
            return self.table[k]
        if k in self.compositions:
            e = self.energy_ev if self.energy_ev is not None else energy_ev
            if e is None:
                raise ValueError("an energy is required to evaluate mu from a composition")
            return self._from_composition(k, e)[0]
        return None

    def rho_e_for(self, name: str, energy_ev: Optional[float] = None) -> Optional[float]:
        """Electrons per A^3 (needed for the K4 / rho_e^2 normalisation check)."""
        k = self.key(name)
        if k in self.rho_e_table:
            return self.rho_e_table[k]
        if k in self.compositions:
            e = self.energy_ev if self.energy_ev is not None else (energy_ev or 8041.0)
            return self._from_composition(k, e)[1]
        return None


# ------------------------------------------------------------------------------------ result
@dataclass
class ReducedSample:
    name: str
    curve: Curve
    transmission: es.Transmission
    loading_class: str
    centre_source: str
    q_min_use: Optional[float]
    q_max_use: Optional[float]
    n_use: int
    mu_mm: Optional[float] = None
    t_eff_mm: Optional[float] = None
    session: str = ""
    sample_path: str = ""
    empty_path: str = ""
    sample_exposure_s: float = float("nan")
    empty_exposure_s: float = float("nan")
    distance_m: float = float("nan")
    wavelength_A: float = float("nan")
    pixel_m: float = float("nan")
    centre_px: Tuple[float, float] = (float("nan"), float("nan"))
    centre_header_px: Tuple[float, float] = (float("nan"), float("nan"))
    centre_offset_px: Tuple[float, float] = (float("nan"), float("nan"))
    empty_centre_px: Tuple[float, float] = (float("nan"), float("nan"))
    empty_centre_offset_px: Tuple[float, float] = (float("nan"), float("nan"))
    ratio_plateaus: Dict[str, float] = field(default_factory=dict)
    n_hot_sample: int = 0
    n_hot_empty: int = 0
    ff_sample: float = 1.0
    ff_empty: float = 1.0
    per_mm_probe: Dict[float, float] = field(default_factory=dict)
    dat_path: Optional[str] = None
    options: Dict[str, object] = field(default_factory=dict)
    reduction: Optional[es.SectorReduction] = None      # kept in memory for diagnostics, never serialised

    @property
    def mu_t(self) -> float:
        T = self.transmission.T
        return float(-math.log(T)) if T is not None and T > 0 else float("nan")

    @property
    def heterogeneity(self) -> float:
        return self.transmission.heterogeneity

    @property
    def columns(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.curve.metadata.get("columns", {}).items()}

    def curve_per_mm(self) -> Optional[Curve]:
        """I_corr / t_eff. A file read back supplies its own ``I_per_mm`` / ``sigma_per_mm`` columns
        (the values as written, so a fit from a file sees exactly the file)."""
        if self.t_eff_mm is None or not np.isfinite(self.t_eff_mm) or self.t_eff_mm <= 0:
            return None
        cols = self.columns
        if "I_per_mm" in cols and len(cols["I_per_mm"]) == len(self.curve.q) and np.isfinite(cols["I_per_mm"]).any():
            I, sig = cols["I_per_mm"], cols.get("sigma_per_mm")
        else:
            I = self.curve.intensity / self.t_eff_mm
            sig = None if self.curve.sigma is None else self.curve.sigma / self.t_eff_mm
        c = self.curve.copy_with(intensity=I, sigma=sig, name=self.name, step="per_mm", t_eff_mm=self.t_eff_mm)
        c.metadata["units"] = "counts/s/pixel per mm of material: I_corr / t_eff"
        c.metadata["per_mm"] = True
        return c

    def usable_curve(self, per_mm: bool = True) -> Curve:
        """The curve on its usable range, per mm when possible (``metadata['per_mm']`` says which),
        with the per-point ratio / empty columns kept in ``metadata['columns']``: the input of the
        descriptors and of the physical fits."""
        base = self.curve_per_mm() if per_mm else None
        if base is None:
            base = self.curve.copy_with(name=self.name)
            base.metadata["per_mm"] = False
        cols = self.columns
        u = cols.get("usable", np.ones(len(base.q), bool)).astype(bool)
        out = base.copy_with(q=base.q[u], intensity=base.intensity[u], sigma=None if base.sigma is None else base.sigma[u],
                             name=self.name, step="usable_range", n_use=int(u.sum()))
        out.metadata["columns"] = {k: v[u].tolist() for k, v in cols.items()}
        out.metadata["class_mu_t"] = self.loading_class
        out.metadata["sample"] = self.name
        return out

    def header_dict(self) -> Dict[str, object]:
        """Ordered ``key -> value`` provenance written to the ``.dat`` header and read back."""
        tr = self.transmission
        fmt = lambda t: f"{t[0]:.4f}, {t[1]:.4f}"     # noqa: E731
        d: Dict[str, object] = {
            "code": CODE_VERSION, "sample": self.name, "session": self.session, "sample_frame": self.sample_path,
            "empty_frame": self.empty_path, "sample_exposure_s": self.sample_exposure_s, "empty_exposure_s": self.empty_exposure_s,
            "distance_m": self.distance_m, "wavelength_A": self.wavelength_A, "pixel_m": self.pixel_m,
            "centre_px": fmt(self.centre_px), "centre_source": self.centre_source, "centre_header_px": fmt(self.centre_header_px),
            "centre_offset_from_header_px": fmt(self.centre_offset_px), "empty_centre_px": fmt(self.empty_centre_px),
            "empty_centre_offset_from_header_px": fmt(self.empty_centre_offset_px),
            "sector": self.curve.metadata.get("sector", ""),
            "transmission_T": tr.T, "transmission_dT": tr.dT, "transmission_T_disc4": tr.T_disc4, "transmission_T_disc8": tr.T_disc8,
            "transmission_footprint_r_px": tr.footprint_r_px, "transmission_vertical_half_deg": tr.vertical_half_deg,
            "heterogeneity_T4_over_T": tr.heterogeneity,
        }
        for k, v in self.ratio_plateaus.items():
            d[f"ratio_plateau_r{k}px"] = v
        d.update({"flatfield_factor_sample": self.ff_sample, "flatfield_factor_empty": self.ff_empty,
                  "n_hot_sample": self.n_hot_sample, "n_hot_empty": self.n_hot_empty,
                  "mu_mm": self.mu_mm, "t_eff_mm": self.t_eff_mm, "mu_t_eff": self.mu_t, "class_mu_t": self.loading_class,
                  "q_min_use": self.q_min_use, "q_max_use": self.q_max_use, "n_use": self.n_use,
                  "options": json.dumps(self.options, sort_keys=True)})
        return d


# ------------------------------------------------------------------------------------ reduction
def _read_frame_cached(path: str, cache: Optional[Dict[str, tuple]], keep: bool) -> tuple:
    if cache is not None and path in cache:
        return cache[path]
    item = es.read_frame(path)
    if cache is not None and keep:
        cache[path] = item
    return item


def _plateaus(ratio: np.ndarray, bands: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    rr = np.arange(len(ratio))
    out = {}
    for a, b in bands:
        sel = (rr >= a) & (rr <= b)
        with np.errstate(invalid="ignore"):
            out[f"{a}-{b}"] = float(np.nanmedian(ratio[sel])) if sel.any() and np.isfinite(ratio[sel]).any() else float("nan")
    return out


def _from_reduction(red: es.SectorReduction, name: str, options: ReductionOptions, mu: Optional[float],
                    session: str = "", sample_path: str = "", empty_path: str = "") -> ReducedSample:
    md = red.curve.metadata
    tr = red.transmission
    mu_t = -math.log(tr.T) if tr.T and tr.T > 0 else float("nan")
    cls = classify_loading(mu_t, tr.heterogeneity, options)
    t_eff = red.t_eff_mm
    cols = {"sample_cps_px": np.asarray(md["_sample_cps_px"]), "empty_cps_px": np.asarray(md["_empty_cps_px"]),
            "ratio": np.asarray(md["_ratio"]), "n_pix": np.asarray(md["_n_pix"]), "usable": np.asarray(md["_usable"]).astype(int)}
    if t_eff is not None and t_eff > 0:
        cols["I_per_mm"] = np.asarray(red.curve.intensity) / t_eff
        cols["sigma_per_mm"] = np.asarray(red.curve.sigma) / t_eff
    else:
        cols["I_per_mm"] = np.full(len(red.curve.q), np.nan)
        cols["sigma_per_mm"] = np.full(len(red.curve.q), np.nan)
    scalar_md = {k: v for k, v in md.items() if not k.startswith("_")}
    curve = Curve(q=red.curve.q, intensity=red.curve.intensity, sigma=red.curve.sigma, name=name, path=sample_path,
                  transmission=tr.T, file_role="corrected", metadata=dict(scalar_md, columns={k: v.tolist() for k, v in cols.items()},
                                                                        class_mu_t=cls, sample=name, session=session),
                  provenance=[dict(p) for p in red.curve.provenance])
    probes: Dict[float, float] = {}
    if t_eff is not None and t_eff > 0:
        for q0, lo, hi in options.per_mm_probes:
            sel = (curve.q > lo) & (curve.q < hi)
            probes[q0] = float(np.nanmean(curve.intensity[sel]) / t_eff) if sel.any() else float("nan")
    else:
        probes = {q0: float("nan") for q0, _, _ in options.per_mm_probes}
    return ReducedSample(
        name=name, curve=curve, transmission=tr, loading_class=cls, centre_source=str(md["centre_source"]),
        q_min_use=red.q_min_use, q_max_use=red.q_max_use, n_use=int(np.sum(cols["usable"])), mu_mm=mu, t_eff_mm=t_eff,
        session=session, sample_path=sample_path or str(md.get("sample_frame", "")), empty_path=empty_path or str(md.get("empty_frame", "")),
        sample_exposure_s=float(md["sample_exposure_s"]), empty_exposure_s=float(md["empty_exposure_s"]),
        distance_m=float(md["distance_m"]), wavelength_A=float(md["wavelength_A"]), pixel_m=float(md["pixel_m"]),
        centre_px=tuple(md["centre_px"]), centre_header_px=tuple(md["centre_header_px"]), centre_offset_px=tuple(md["centre_offset_from_header_px"]),
        empty_centre_px=tuple(md["empty_centre_px"]), empty_centre_offset_px=tuple(md.get("empty_centre_offset_from_header_px", (float("nan"), float("nan")))),
        ratio_plateaus=_plateaus(red.ratio, options.plateau_bands_px), n_hot_sample=int(md.get("n_hot_sample", 0)),
        n_hot_empty=int(md.get("n_hot_empty", 0)), ff_sample=float(md["flatfield_factor_sample"]), ff_empty=float(md["flatfield_factor_empty"]),
        per_mm_probe=probes, options=options.to_dict(), reduction=red)


def reduce_arrays_with_options(sample: np.ndarray, gs: es.FrameGeometry, empty: np.ndarray, ge: es.FrameGeometry,
                               options: ReductionOptions, mu_mm: Optional[float] = None, name: str = "sample",
                               sample_path: str = "", empty_path: str = "", session: str = "") -> ReducedSample:
    """Reduce two frames in memory with every threshold taken from ``options``."""
    o = options
    red = es.reduce_arrays(sample, gs, empty, ge, mu_mm=mu_mm, axis_exclusion_deg=o.axis_exclusion_deg,
                           bar_rect_px=tuple(o.bar_rect_px), rmax_px=int(o.rmax_px), footprint_r_px=o.footprint_r_px,
                           vertical_half_deg=o.vertical_half_deg, name=name, sample_path=sample_path, empty_path=empty_path,
                           refine_centre=o.refine_centre, max_shift_px=o.max_shift_px, bright_fraction=o.bright_fraction,
                           min_ratio=o.min_ratio, min_ratio_footprint=o.min_ratio_footprint, r_footprint_px=o.r_footprint_px,
                           nsigma=o.nsigma, min_run=int(o.min_run), flatfield_method=o.flatfield_method)
    return _from_reduction(red, name, o, mu_mm, session, sample_path, empty_path)


def reduce_pair_records(pair: FramePair, options: Optional[ReductionOptions] = None, mu_source: Optional[MuSource] = None,
                        cache: Optional[Dict[str, tuple]] = None) -> ReducedSample:
    """Reduce one :class:`~saxs.core.frames.FramePair` (frames read with fabio; empties cached)."""
    o = options or ReductionOptions()
    s_data, _hs, gs = _read_frame_cached(pair.sample.path, cache, keep=False)
    e_data, _he, ge = _read_frame_cached(pair.empty.path, cache, keep=True)
    mu = mu_source.mu_for(pair.sample.name, pair.sample.energy_ev) if mu_source is not None else None
    return reduce_arrays_with_options(s_data, gs, e_data, ge, o, mu_mm=mu, name=pair.sample.name,
                                      sample_path=pair.sample.path, empty_path=pair.empty.path, session=pair.sample.session)


def reduce_pairs(pairs: Iterable[FramePair], options: Optional[ReductionOptions] = None, mu_source: Optional[MuSource] = None,
                 out_dir: Optional[str] = None, progress: Progress = None,
                 on_error: Optional[Callable[[FramePair, Exception], None]] = None,
                 sasview_dir: Optional[str] = None) -> List[ReducedSample]:
    """Reduce every pair; with ``out_dir`` each result is written as ``<name><dat_suffix>.dat`` (the
    full provenance layout) and with ``sasview_dir`` also as a SasView-ready three-column file."""
    o = options or ReductionOptions()
    pairs = list(pairs)
    cache: Dict[str, tuple] = {}
    results: List[ReducedSample] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n = len(pairs)
    used: Dict[str, int] = {}
    for i, pair in enumerate(pairs):
        if progress is not None:
            progress(i, n, pair.sample.name)
        try:
            res = reduce_pair_records(pair, o, mu_source, cache)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(pair, exc)
            continue
        used[res.name] = used.get(res.name, 0) + 1
        if used[res.name] > 1:                      # the same sample name twice: keep both, distinguishable
            res.name = f"{res.name}__{used[res.name]}"
            res.curve.name = res.name
            res.curve.metadata["sample"] = res.name
        if out_dir:
            res.dat_path = os.path.join(out_dir, f"{res.name}{o.dat_suffix}.dat")
            write_reduced_dat(res, res.dat_path)
        if sasview_dir:
            write_sasview_dat(res, os.path.join(sasview_dir, f"{res.name}{SASVIEW_SUFFIX}.dat"))
        results.append(res)
    if progress is not None:
        progress(n, n, "done")
    return results


# ------------------------------------------------------------------------------------ files
def write_reduced_dat(result: ReducedSample, path: str) -> None:
    """Tab-separated curve with the reduced_v2 column layout and a ``# key: value`` header."""
    c = result.curve
    cols = result.columns
    te = result.t_eff_mm if (result.t_eff_mm is not None and np.isfinite(result.t_eff_mm) and result.t_eff_mm > 0) else None
    lines = [f"# {k}: {v}" for k, v in result.header_dict().items()]
    lines.append("# columns: " + "  ".join(DAT_COLUMNS))
    lines.append("# I_corr = (S/t_s - T E/t_e)/T in counts/s/pixel (empty-transmission units); I_per_mm = I_corr / t_eff_mm")
    lines.append("\t".join(DAT_COLUMNS))
    sig = c.sigma if c.sigma is not None else np.full(len(c.q), np.nan)
    ipm_col = cols.get("I_per_mm")
    spm_col = cols.get("sigma_per_mm")
    for i in range(len(c.q)):
        if ipm_col is not None and len(ipm_col) == len(c.q):
            ipm, spm = ipm_col[i], (spm_col[i] if spm_col is not None else float("nan"))
        else:
            ipm = c.intensity[i] / te if te else float("nan")
            spm = sig[i] / te if te else float("nan")
        lines.append(f"{c.q[i]:.6e}\t{c.intensity[i]:.6e}\t{sig[i]:.6e}\t{ipm:.6e}\t{spm:.6e}\t{cols['sample_cps_px'][i]:.6e}\t"
                     f"{cols['empty_cps_px'][i]:.6e}\t{cols['ratio'][i]:.5e}\t{int(cols['n_pix'][i])}\t{int(cols['usable'][i])}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_sasview_dat(result: ReducedSample, path: str, per_mm: bool = True, usable_only: bool = True, sep: str = " ") -> str:
    """A file SasView's ASCII reader (sasdata) opens as a 1D data set: three numeric columns Q (1/A),
    I, dI separated by ``sep``, comment lines starting with ``#`` before the data, no column-name
    line after them, no dQ column (a pinhole instrument: SasView would read a 4th column as the q
    resolution), at least five rows. SasView rejects the ten-column provenance layout of
    :func:`write_reduced_dat` as "2D ASCII data", so this is the file to give an external fit engine.
    The intensity is per mm of material when the reduction knows mu (``per_mm``), else I_corr; with
    ``usable_only`` only the usable range is written (the input of PRISM's own fits)."""
    src = result.curve_per_mm() if per_mm else None
    units = "counts/s/pixel per mm of material (I_corr / t_eff)"
    if src is None:
        src = result.curve
        units = "counts/s/pixel, empty-transmission units: (S/t_s - T E/t_e)/T"
    q = np.asarray(src.q, float)
    I = np.asarray(src.intensity, float)
    sig = np.asarray(src.sigma, float) if src.sigma is not None else np.full(len(q), np.nan)
    keep = np.isfinite(q) & np.isfinite(I) & (q > 0)
    cols = result.columns
    if usable_only and "usable" in cols and len(cols["usable"]) == len(q):
        keep &= cols["usable"].astype(bool)
    has_sigma = bool(np.isfinite(sig[keep]).all()) and keep.any()
    tr = result.transmission
    lines = ["# PRISM SAXS 2.0 sector reduction: SasView-ready 1D export (columns Q, I, dI; no dQ column)",
             f"# sample: {result.name}", f"# intensity: {units}",
             "# scale: not absolute (SasView labels I as 1/cm by default; treat the scale factor as free)",
             f"# range: {'usable range only' if usable_only else 'all rings'}",
             f"# points: {int(keep.sum())}", f"# transmission_T: {tr.T:.6g}", f"# heterogeneity_T4_over_T: {tr.heterogeneity:.3g}",
             f"# class_mu_t: {result.loading_class}", f"# mu_mm: {result.mu_mm}", f"# t_eff_mm: {result.t_eff_mm}",
             f"# q_min_use: {result.q_min_use}", f"# q_max_use: {result.q_max_use}",
             f"# sample_frame: {result.sample_path}", f"# empty_frame: {result.empty_path}",
             f"# centre_source: {result.centre_source}", f"# full provenance: {os.path.basename(result.dat_path) if result.dat_path else 'see the reduction .dat'}",
             "# Q (1/A)" + sep + "I" + sep + "dI" if has_sigma else "# Q (1/A)" + sep + "I"]
    for i in np.flatnonzero(keep):
        row = f"{q[i]:.10e}{sep}{I[i]:.10e}"
        if has_sigma:
            row += f"{sep}{sig[i]:.10e}"
        lines.append(row)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def export_sasview(results: Iterable[ReducedSample], folder: str, per_mm: bool = True, usable_only: bool = True,
                   suffix: str = SASVIEW_SUFFIX) -> List[str]:
    """One SasView-ready file per reduced sample in ``folder`` (``<name><suffix>.dat``)."""
    os.makedirs(folder, exist_ok=True)
    return [write_sasview_dat(r, os.path.join(folder, f"{r.name}{suffix}.dat"), per_mm, usable_only) for r in results]


def read_dat_header(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s.startswith("#"):
                break
            if ":" in s:
                k, v = s[1:].split(":", 1)
                meta.setdefault(k.strip(), v.strip())
    return meta


def _pair_of(text: str) -> Tuple[float, float]:
    try:
        a, b = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)[:2]]
        return a, b
    except ValueError:
        return float("nan"), float("nan")


def _float_of(meta: Dict[str, str], key: str, default: float = float("nan")) -> float:
    v = meta.get(key)
    if v in (None, "", "None", "nan"):
        return default
    try:
        return float(str(v).split("+-")[0].split()[0])
    except ValueError:
        return default


def load_reduced_dat(path: str) -> ReducedSample:
    """Read a file written by :func:`write_reduced_dat` (or the reduced_v2 layout) back into a
    :class:`ReducedSample`. Older headers with slightly different key names are mapped."""
    meta = read_dat_header(path)
    arr = _load_numeric(path)
    q, I, sig, ipm, spm, S, E, ratio, npix, usable = (arr[:, i] for i in range(10))
    name = meta.get("sample") or os.path.splitext(os.path.basename(path))[0].split("__")[0]
    T = _float_of(meta, "transmission_T", _float_of(meta, "transmission_T_used (footprint r<16px, vertical +-30 deg)"))
    tr = es.Transmission(T=T, dT=_float_of(meta, "transmission_dT", float("nan")),
                         T_disc4=_float_of(meta, "transmission_T_disc4", _float_of(meta, "transmission_T_disc_r<4px")),
                         T_disc8=_float_of(meta, "transmission_T_disc8", _float_of(meta, "transmission_T_disc_r<8px")),
                         footprint_r_px=_float_of(meta, "transmission_footprint_r_px", 16.0),
                         vertical_half_deg=_float_of(meta, "transmission_vertical_half_deg", 30.0))
    mu = _float_of(meta, "mu_mm")
    t_eff = _float_of(meta, "t_eff_mm")
    try:
        options = json.loads(meta.get("options", "{}"))
    except json.JSONDecodeError:
        options = {}
    cols = {"sample_cps_px": S.tolist(), "empty_cps_px": E.tolist(), "ratio": ratio.tolist(), "n_pix": npix.astype(int).tolist(),
            "usable": usable.astype(int).tolist(), "I_per_mm": ipm.tolist(), "sigma_per_mm": spm.tolist()}
    cls = meta.get("class_mu_t", "")
    curve = Curve(q=q, intensity=I, sigma=sig, name=name, path=os.path.abspath(path), transmission=T, file_role="corrected",
                  metadata=dict(meta, columns=cols, class_mu_t=cls, sample=name, sector=meta.get("sector", "")))
    curve.record("load_reduced_dat", path=str(path))
    plateaus = {}
    for k, v in meta.items():
        m = re.match(r"ratio_plateau_r(\d+-\d+)px", k)
        if m:
            plateaus[m.group(1)] = _float_of(meta, k)
    res = ReducedSample(
        name=name, curve=curve, transmission=tr, loading_class=cls, centre_source=meta.get("centre_source", ""),
        q_min_use=_float_of(meta, "q_min_use", None), q_max_use=_float_of(meta, "q_max_use", None), n_use=int(usable.sum()),
        mu_mm=(mu if np.isfinite(mu) else None), t_eff_mm=(t_eff if np.isfinite(t_eff) else None), session=meta.get("session", ""),
        sample_path=meta.get("sample_frame", ""), empty_path=meta.get("empty_frame", ""),
        sample_exposure_s=_float_of(meta, "sample_exposure_s"), empty_exposure_s=_float_of(meta, "empty_exposure_s"),
        distance_m=_float_of(meta, "distance_m"), wavelength_A=_float_of(meta, "wavelength_A"), pixel_m=_float_of(meta, "pixel_m"),
        centre_px=_pair_of(meta.get("centre_px", "")), centre_header_px=_pair_of(meta.get("centre_header_px", "")),
        centre_offset_px=_pair_of(meta.get("centre_offset_from_header_px", "")), empty_centre_px=_pair_of(meta.get("empty_centre_px", "")),
        empty_centre_offset_px=_pair_of(meta.get("empty_centre_offset_from_header_px", "")), ratio_plateaus=plateaus,
        n_hot_sample=int(_float_of(meta, "n_hot_sample", _float_of(meta, "n_clipped_sample", 0))),
        n_hot_empty=int(_float_of(meta, "n_hot_empty", _float_of(meta, "n_clipped_empty", 0))),
        ff_sample=_float_of(meta, "flatfield_factor_sample", 1.0), ff_empty=_float_of(meta, "flatfield_factor_empty", 1.0),
        dat_path=os.path.abspath(path), options=options)
    if res.t_eff_mm:
        o = ReductionOptions.from_dict({k: v for k, v in options.items() if k == "per_mm_probes"}) if "per_mm_probes" in options else ReductionOptions()
        for q0, lo, hi in o.per_mm_probes:
            sel = (q > lo) & (q < hi)
            res.per_mm_probe[q0] = float(np.nanmean(I[sel]) / res.t_eff_mm) if sel.any() else float("nan")
    return res


def _load_numeric(path: str) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#") or s[0].isalpha():
                continue
            rows.append([float(x) for x in s.split()])
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"{path}: expected the 10-column reduced layout, got shape {arr.shape}")
    return arr


def load_reduced_folder(folder: str, pattern: str = r".*\.dat$", exclude_pattern: Optional[str] = None) -> List[ReducedSample]:
    out = []
    for f in sorted(os.listdir(folder)):
        if not re.match(pattern, f) or (exclude_pattern and re.search(exclude_pattern, f)):
            continue
        out.append(load_reduced_dat(os.path.join(folder, f)))
    return out


# ------------------------------------------------------------------------------------ summary
SUMMARY_COLUMNS = ["sample", "session", "sample_frame", "empty_frame", "exposure_s", "T_used", "dT_rel", "T_disc4", "T_disc8",
                   "heterogeneity_T4_over_T", "ratio_plateau_6_12", "ratio_plateau_12_20", "mu_mm", "mu_t_eff", "t_eff_mm",
                   "class_mu_t", "n_clipped_sample", "n_clipped_empty", "centre_source", "centre_dx_px", "centre_dy_px",
                   "empty_centre_dx_px", "ff_sample", "ff_empty", "q_min_use", "q_max_use", "n_use",
                   "I_per_mm_at_0p02", "I_per_mm_at_0p05", "I_per_mm_at_0p15", "dat_path"]


def summary_table(results: Iterable[ReducedSample]) -> pd.DataFrame:
    """One row per sample with the columns of the rethink's ``phase1_summary.csv`` (plus ``dat_path``)."""
    rows = []
    for r in results:
        tr = r.transmission
        row = dict(sample=r.name, session=r.session, sample_frame=r.sample_path, empty_frame=r.empty_path,
                   exposure_s=r.sample_exposure_s, T_used=tr.T, dT_rel=(tr.dT / tr.T if tr.T else float("nan")),
                   T_disc4=tr.T_disc4, T_disc8=tr.T_disc8, heterogeneity_T4_over_T=tr.heterogeneity,
                   mu_mm=r.mu_mm, mu_t_eff=r.mu_t, t_eff_mm=r.t_eff_mm, class_mu_t=r.loading_class,
                   n_clipped_sample=r.n_hot_sample, n_clipped_empty=r.n_hot_empty, centre_source=r.centre_source,
                   centre_dx_px=r.centre_offset_px[0], centre_dy_px=r.centre_offset_px[1], empty_centre_dx_px=r.empty_centre_offset_px[0],
                   ff_sample=r.ff_sample, ff_empty=r.ff_empty, q_min_use=r.q_min_use, q_max_use=r.q_max_use, n_use=r.n_use,
                   dat_path=r.dat_path or "")
        for k, v in r.ratio_plateaus.items():
            row[f"ratio_plateau_{k.replace('-', '_')}"] = v
        for q0, v in r.per_mm_probe.items():
            row[f"I_per_mm_at_{str(q0).replace('.', 'p')}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = [c for c in SUMMARY_COLUMNS if c in df.columns] + [c for c in df.columns if c not in SUMMARY_COLUMNS]
    return df[ordered] if len(df) else pd.DataFrame(columns=SUMMARY_COLUMNS)


def usable_per_mm_curve(result: ReducedSample) -> Curve:
    """Convenience alias of :meth:`ReducedSample.usable_curve` (per mm when mu is known)."""
    return result.usable_curve(per_mm=True)
