"""Physical fitting workflow for a series of absolute SAXS curves (one engine: saxs.core.physical_fit).

For every curve:

1. **Base fits.** B0 = grain power law (exponent bounded, default 3.7-4.7) + flat level C >= 0;
   B1 = grain power law + Ornstein-Zernike term A / (1 + q^2 xi^2) with xi capped (default 1-5 Å: sub-nanometre
   concentration fluctuations; a larger xi would absorb nanometre structure). B1 is preferred only when it beats B0
   by dBIC > ``dbic_min`` with no parameter at a bound.
2. **Gate: is there a structure to fit?** Model-free: the ratio of the curve to its *reference* (the same material
   without the minority component, e.g. a parent composition), each with its fitted high-q term subtracted and
   normalised at q <= ``norm_qmax``, must exceed 1 by more than its uncertainty over a contiguous run spanning a factor
   >= ``gate_span`` in q inside ``gate_window``, with a peak above ``gate_null_max`` - by default the largest
   reference/reference ratio in the same window (the scatter between references themselves, e.g. batch-to-batch
   differences of the grain law). A curve used as a reference gets no structural term. A residual run against the
   base is reported (``fit_gate_span``) as a diagnostic only: it fires on clean featureless curves whose grain law is
   not a single power law at the 2-3 sigma level, so it cannot decide.
3. **Structural candidates** (on both bases): polydisperse spheres SPH (absolute when the particle contrast is known),
   spheres with a Percus-Yevick S(q) SPHPY, Teubner-Strey TS, Debye-Anderson-Brumberger DAB, one unified level with a
   tied Porod prefactor UNI; each also with a second, smaller unified population (+UNI2). Candidate keys:
   ``"TS"`` (on the preferred base), ``"TS@B1"`` (on the other base), ``"TS+UNI2"``, ``"TS+UNI2@B1"``.
4. **Selection.** A structural model is retained only if the gate passed, it converged, no free parameter is at a
   bound, it beats its base by dBIC > ``dbic_min``, its structural component carries >= ``visibility_fraction`` of the
   model over a span >= ``gate_span`` in q (q <= ``visibility_qmax``), and, for spheres, the ensemble Rg lies inside
   the window (1/q_max..1/q_min; otherwise it is a q^-4 law, i.e. grain-like). SPHPY must beat SPH by dBIC >
   ``dbic_min``. A second population must beat the retained single structure by dBIC > ``dbic_min`` and be visible
   itself. Every rejection is recorded with its reason.
5. **Robustness** (optional): the retained model refitted from its solution on perturbed data (statistical weights
   only, q <= ``robust_qmax``, last point removed, unsmeared without the flagged low-q points, an optional offset of
   part of the curve); a parameter is robust if it stays within max(20 %, 2 sigma) and off its bounds.

Nothing here knows a sample: names, references, contrasts and ceilings come from the caller (see saxs.core.pipeline
for the JSON project interface). Units: q in 1/Å, I in 1/cm, lengths in Å unless a name says otherwise.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from saxs.core.contrast import invariant, min_contrast_1e10cm2, two_phase_product_cm4, volume_fraction_from_invariant
from saxs.core.models import COMPONENTS, SphereSchulz, TeubnerStrey
from saxs.core.physical_fit import (PhysFitResult, RatioResult, Run, TermSpec, fit_physical, reference_ratio,
                                    residual_runs)

Progress = Optional[Callable[[int, int, str], None]]

FAMILIES = {"sphere": "SPH", "sphere_py": "SPHPY", "teubner_strey": "TS", "debye": "DAB", "unified": "UNI"}
FAMILY_TITLES = {"SPH": "polydisperse spheres", "SPHPY": "spheres + Percus-Yevick S(q)", "TS": "Teubner-Strey",
                 "DAB": "Debye (DAB)", "UNI": "unified level", "UNI2": "second (small) unified population",
                 "B0": "grain law + flat level", "B1": "grain law + Ornstein-Zernike term"}
KEY = {"SPH": "sph", "SPHPY": "sph", "TS": "ts", "DAB": "dab", "UNI": "uni"}
BASES = ("B0", "B1")
_Q_REF = 0.0026            # the start grids are written for a window starting near this q (1/Å)


# ------------------------------------------------------------------------------------------- options and inputs
@dataclass
class WorkflowOptions:
    floor: float = 0.03                        # relative floor of the weights
    n_nodes: int = 12                          # Gauss-Hermite nodes of the resolution smearing
    smear: bool = True                         # use the dQ column when a curve has one
    grain_p_bounds: Tuple[float, float] = (3.7, 4.7)
    highq: Tuple[str, ...] = ("flat", "oz")    # high-q terms offered to the base
    oz_xi_bounds: Tuple[float, float] = (1.0, 5.0)
    highq_seed_qmin: float = 0.3               # start of the high-q term = median I above this q (else last 3 points)
    gate: str = "reference"                    # "reference" | "none" (structures always offered, flagged)
    gate_window: Tuple[float, float] = (0.006, 0.1)
    norm_qmax: float = 0.006
    gate_span: float = 2.0
    gate_sigma: float = 1.0
    gate_null_max: Optional[float] = None      # None: the largest reference/reference ratio in gate_window
    unreferenced: str = "offer"                # curves without a reference: "offer" (flagged) | "skip"
    families: Tuple[str, ...] = ("sphere", "sphere_py", "teubner_strey", "debye", "unified")
    second_population: bool = True
    second_rg_bounds: Tuple[float, float] = (5.0, 100.0)
    dbic_min: float = 10.0
    visibility_fraction: float = 0.10
    visibility_qmax: float = 0.2
    prescreen: int = 10                        # structural fits: optimise the 10 best starts of the grid
    max_nfev: int = 3000
    polish_rounds: int = 3                     # continue a best start that ran out of evaluations (0 = off)
    report_q: float = 0.45                     # the high-q term is also reported at this q
    robust_qmax: float = 0.2
    robust_tol: float = 0.20
    nosmear_qmin: Optional[float] = None       # unsmeared variant: q above this (else the curve's smeared flags)
    offset_variant: Optional[Dict[str, object]] = None   # {"factor": 1.06, "q_lo": 0.02, "q_hi": 0.035, "tag": "holder", "value": "H1"}

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, object]]) -> "WorkflowOptions":
        o = cls()
        unknown = []
        for k, v in dict(d or {}).items():
            if not hasattr(o, k):
                unknown.append(k)
                continue
            cur = getattr(o, k)
            setattr(o, k, tuple(v) if isinstance(cur, tuple) and isinstance(v, (list, tuple)) else v)
        if unknown:
            raise KeyError(f"unknown fit option(s): {', '.join(sorted(unknown))}")
        bad = [f for f in o.families if f not in FAMILIES]
        if bad:
            raise KeyError(f"unknown model famil(ies) {bad}; known: {sorted(FAMILIES)}")
        return o


@dataclass
class CurveInput:
    name: str
    q: np.ndarray
    I: np.ndarray
    sigma: np.ndarray                          # total uncertainty (weights)
    sigma_stat: Optional[np.ndarray] = None    # counting statistics only (robustness variant)
    dq: Optional[np.ndarray] = None            # Gaussian sigma of the q resolution
    smeared_flag: Optional[np.ndarray] = None  # True where the Gaussian dQ under-represents the beam
    reference: Optional[str] = None            # name of the reference curve
    contrast: Optional[float] = None           # particle contrast, 1e10 cm^-2 (spheres absolute; invariant -> phi)
    contrast_alt: Optional[float] = None       # an alternative contrast (e.g. another density of the same phase)
    phi_max: Optional[float] = None            # largest volume fraction the composition allows (checks only)
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.q = np.asarray(self.q, float)
        self.I = np.asarray(self.I, float)
        self.sigma = np.asarray(self.sigma, float) if self.sigma is not None else 0.03 * np.abs(self.I)
        for k in ("sigma_stat", "dq"):
            v = getattr(self, k)
            setattr(self, k, None if v is None else np.asarray(v, float))
        if self.smeared_flag is not None:
            self.smeared_flag = np.asarray(self.smeared_flag, bool)

    @classmethod
    def from_curve(cls, curve, reference: Optional[str] = None, **kw) -> "CurveInput":
        """From a saxs.core.curve.Curve: sigma = curve.sigma (3 % of I when missing), dQ / statistical sigma / smeared
        flags from the columns kept by saxs.core.loader.load_tabular_curve when present."""
        md = getattr(curve, "metadata", {}) or {}
        cols = md.get("columns") or {}

        def col(*patterns):
            for c, v in cols.items():
                if any(re.fullmatch(p, str(c), flags=re.I) for p in patterns) and not isinstance(v, list):
                    return np.asarray(v, float)
            return None

        smeared = col(r"smeared.*", r".*smear.*flag.*")
        return cls(name=kw.pop("name", curve.name), q=curve.q, I=curve.intensity,
                   sigma=curve.sigma if curve.sigma is not None else None,
                   sigma_stat=kw.pop("sigma_stat", col(r"sigma_stat.*", r"d?i_stat.*")),
                   dq=kw.pop("dq", md.get("dq")), smeared_flag=None if smeared is None else smeared > 0.5,
                   reference=reference, **kw)


# ------------------------------------------------------------------------------------------- results
@dataclass
class GateResult:
    method: str                                # "reference" | "none" | "is reference" | "no reference"
    passed: bool
    peak: float = float("nan")
    q_peak: float = float("nan")
    run: Optional[Run] = None
    null_max: float = float("nan")
    note: str = ""
    ratio_q: Optional[np.ndarray] = None
    ratio: Optional[np.ndarray] = None
    ratio_err: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, object]:
        arr = lambda a: None if a is None else [None if not np.isfinite(x) else float(x) for x in a]   # noqa: E731
        return {"method": self.method, "passed": bool(self.passed), "peak": _f(self.peak), "q_peak": _f(self.q_peak),
                "run": None if self.run is None else self.run.to_dict(), "null_max": _f(self.null_max), "note": self.note,
                "ratio_q": arr(self.ratio_q), "ratio": arr(self.ratio), "ratio_err": arr(self.ratio_err)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "GateResult":
        arr = lambda a: None if a is None else np.array([np.nan if x is None else x for x in a], float)   # noqa: E731
        return cls(method=str(d["method"]), passed=bool(d["passed"]), peak=_nan(d.get("peak")),
                   q_peak=_nan(d.get("q_peak")), run=Run.from_dict(d.get("run")), null_max=_nan(d.get("null_max")),
                   note=str(d.get("note", "")), ratio_q=arr(d.get("ratio_q")), ratio=arr(d.get("ratio")),
                   ratio_err=arr(d.get("ratio_err")))


def _f(v) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else float(v)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


@dataclass
class CurveResult:
    name: str
    reference: Optional[str]
    fits: Dict[str, PhysFitResult]
    base: str
    gate: GateResult
    fit_gate_span: float
    retained: str
    reasons: Dict[str, List[str]] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    contrast: Optional[float] = None
    contrast_alt: Optional[float] = None
    phi_max: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def retained_fit(self) -> PhysFitResult:
        return self.fits[self.retained]

    @property
    def structural(self) -> bool:
        return self.retained not in BASES

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "reference": self.reference, "fits": {k: f.to_dict() for k, f in self.fits.items()},
                "base": self.base, "gate": self.gate.to_dict(), "fit_gate_span": _f(self.fit_gate_span),
                "retained": self.retained, "reasons": self.reasons,
                "derived": {k: _f(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for k, v in self.derived.items()},
                "contrast": self.contrast, "contrast_alt": self.contrast_alt, "phi_max": self.phi_max, "tags": self.tags}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CurveResult":
        return cls(name=str(d["name"]), reference=d.get("reference"),
                   fits={k: PhysFitResult.from_dict(v) for k, v in dict(d["fits"]).items()}, base=str(d["base"]),
                   gate=GateResult.from_dict(d["gate"]), fit_gate_span=_nan(d.get("fit_gate_span")),
                   retained=str(d["retained"]), reasons={k: list(v) for k, v in dict(d.get("reasons", {})).items()},
                   derived={k: (np.nan if v is None else v) for k, v in dict(d.get("derived", {})).items()},
                   contrast=d.get("contrast"), contrast_alt=d.get("contrast_alt"), phi_max=d.get("phi_max"),
                   tags=dict(d.get("tags", {})))


@dataclass
class WorkflowResult:
    options: WorkflowOptions
    curves: Dict[str, CurveResult]
    null_max: float = float("nan")
    null_pairs: int = 0
    created: str = ""
    robustness: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, object]:
        return {"format": "prism-saxs-fit-workflow", "version": 1, "created": self.created,
                "options": self.options.to_dict(), "null_max": _f(self.null_max), "null_pairs": self.null_pairs,
                "curves": {k: c.to_dict() for k, c in self.curves.items()},
                "robustness": None if self.robustness is None else self.robustness.to_dict(orient="records")}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "WorkflowResult":
        if d.get("format") != "prism-saxs-fit-workflow":
            raise ValueError("not a PRISM SAXS fit-workflow file")
        rob = d.get("robustness")
        return cls(options=WorkflowOptions.from_dict(d.get("options")), created=str(d.get("created", "")),
                   curves={k: CurveResult.from_dict(v) for k, v in dict(d["curves"]).items()},
                   null_max=_nan(d.get("null_max")), null_pairs=int(d.get("null_pairs", 0)),
                   robustness=None if rob is None else pd.DataFrame(rob))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, allow_nan=True)

    @classmethod
    def load(cls, path: str) -> "WorkflowResult":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ------------------------------------------------------------------------------------------- seeds and terms
def _scale(q: np.ndarray) -> float:
    """Length scale of the start grids: 1 for windows starting near q = 0.0026 1/Å (within +-25 %), else q_ref/q_min."""
    s = _Q_REF / float(np.min(q))
    return 1.0 if 0.8 <= s <= 1.25 else s


def _grain_term(q: np.ndarray, I: np.ndarray, o: WorkflowOptions) -> TermSpec:
    lo = q < float(np.min(q)) * 10 ** 0.36                      # the lowest ~0.36 decade: grain-dominated
    p0 = 4.2
    B0 = float(np.median(I[lo] * q[lo] ** p0)) if lo.any() else 1e-6
    return TermSpec("power_law", "grain_", values={"B": B0, "p": p0}, bounds={"p": tuple(o.grain_p_bounds), "B": (0.0, np.inf)})


def _highq_level(q: np.ndarray, I: np.ndarray, o: WorkflowOptions) -> float:
    hi = q > o.highq_seed_qmin
    c0 = float(np.median(I[hi])) if hi.sum() > 2 else float(I[-3:].mean())
    return max(c0, 1e-4)


def _flat_term(q, I, o) -> TermSpec:
    return TermSpec("flat_background", "flat_", values={"C": _highq_level(q, I, o)}, bounds={"C": (0.0, np.inf)})


def _oz_term(q, I, o) -> TermSpec:
    return TermSpec("lorentz_oz", "oz_", values={"A": _highq_level(q, I, o) * 1.5, "xiL": 3.0},
                    bounds={"A": (0.0, np.inf), "xiL": tuple(o.oz_xi_bounds)})


def base_terms(q: np.ndarray, I: np.ndarray, o: WorkflowOptions) -> Dict[str, List[TermSpec]]:
    out = {}
    if "flat" in o.highq:
        out["B0"] = [_grain_term(q, I, o), _flat_term(q, I, o)]
    if "oz" in o.highq:
        out["B1"] = [_grain_term(q, I, o), _oz_term(q, I, o)]
    if not out:
        raise ValueError("options.highq must contain 'flat' and/or 'oz'")
    return out


def grain_seed(q: np.ndarray, I: np.ndarray, o: WorkflowOptions) -> Dict[str, float]:
    """Grain-law start from the lowest-q points alone (grain-dominated), never from a base fit that may have bent the
    grain law to absorb a structure."""
    m = (q <= float(np.min(q)) * 10 ** 0.28) & (I > 0)
    if m.sum() < 2:
        m = np.argsort(q)[:3]
    slope, _ = np.polyfit(np.log(q[m]), np.log(I[m]), 1)
    lo, hi = o.grain_p_bounds
    p = float(np.clip(-slope, lo + 0.05, hi - 0.05))
    return {"grain_p": p, "grain_B": float(np.exp(np.mean(np.log(I[m]) + p * np.log(q[m]))))}


def structural_terms(kind: str, q: np.ndarray, ci: CurveInput, o: WorkflowOptions) -> List[TermSpec]:
    qmin, qmax = float(np.min(q)), float(np.max(q))
    if kind in ("SPH", "SPHPY"):
        absolute = ci.contrast is not None and np.isfinite(ci.contrast) and ci.contrast != 0
        drho = float(ci.contrast) if absolute else 1.0
        bounds = {"R": (10.0, 4.0 / qmin), "sig": (0.05, 0.7), "phi": (0.0, 0.74 if absolute else np.inf)}
        if kind == "SPH":
            return [TermSpec("sphere_schulz", "sph_", values={"phi": 0.02, "R": 120.0, "sig": 0.3}, bounds=bounds,
                             fixed={"drho": drho, "phi_hs": 0.0})]
        return [TermSpec("sphere_schulz", "sph_", values={"phi": 0.02, "R": 120.0, "sig": 0.3, "phi_hs": 0.1},
                         bounds=dict(bounds, phi_hs=(0.0, 0.5)), fixed={"drho": drho}, free=("phi_hs",))]
    if kind == "TS":
        return [TermSpec("teubner_strey", "ts_", values={"S": 100.0, "d": 400.0, "xi": 150.0},
                         bounds={"S": (0.0, np.inf), "d": (2 * math.pi / qmax, 2 * math.pi / qmin), "xi": (5.0, 2 * math.pi / qmin)})]
    if kind == "DAB":
        return [TermSpec("dab", "dab_", values={"A": 1e3, "xi": 80.0}, bounds={"A": (0.0, np.inf), "xi": (2.0, 1.0 / qmin)})]
    if kind == "UNI":
        return [TermSpec("beaucage_tied", "uni_", values={"G": 1e3, "Rg": 100.0, "d": 3.6, "pf": 0.5},
                         bounds={"G": (0.0, np.inf), "Rg": (5.0, 1.0 / qmin), "d": (2.0, 4.5), "pf": (0.1, 1.5)})]
    raise KeyError(kind)


def second_term(o: WorkflowOptions) -> TermSpec:
    return TermSpec("beaucage_tied", "uni2_", values={"G": 30.0, "Rg": 30.0, "d": 3.5, "pf": 0.5},
                    bounds={"G": (0.0, np.inf), "Rg": tuple(o.second_rg_bounds), "d": (2.0, 4.5), "pf": (0.1, 1.5)})


def _phi_scale(ci: CurveInput) -> float:
    """Sphere amplitude starts are volume fractions for a contrast ~ 8e10 cm^-2; without a contrast the amplitude is
    phi drho^2 and the starts are scaled accordingly."""
    return 1.0 if (ci.contrast is not None and np.isfinite(ci.contrast) and ci.contrast != 0) else 64.0


def start_grid(kind: str, I_exc: float, s: float, ci: CurveInput) -> List[Dict[str, float]]:
    """Starting points (prefixed names) for one structural family; lengths scale with s, amplitudes with I_exc."""
    e = max(I_exc, 1.0)
    f = _phi_scale(ci)
    if kind in ("SPH", "SPHPY"):
        return [{"sph_R": R * s, "sph_sig": sg, "sph_phi": ph * f} for R in (30.0, 70.0, 130.0, 250.0)
                for sg in (0.2, 0.5) for ph in (0.003, 0.03, 0.15)]
    if kind == "TS":
        return [{"ts_d": d * s, "ts_xi": xi * s, "ts_S": S} for d in (150.0, 400.0, 800.0) for xi in (30.0, 100.0, 300.0)
                for S in (e, e * 10)]
    if kind == "DAB":
        return [{"dab_xi": xi * s, "dab_A": A} for xi in (10.0, 30.0, 80.0, 200.0) for A in (e * 3, e * 100)]
    if kind == "UNI":
        return [{"uni_Rg": rg * s, "uni_G": G, "uni_d": d} for rg in (20.0, 50.0, 100.0, 200.0)
                for G in (e * 2, e * 30) for d in (3.0, 3.8)]
    return []


def second_grid(kind: str, s: float, seed: Dict[str, float], ci: CurveInput) -> List[Dict[str, float]]:
    f = _phi_scale(ci)
    grid = [dict(uni2_Rg=rg * s, uni2_G=G, uni2_d=d) for rg in (15.0, 30.0, 60.0) for G in (3.0, 30.0, 300.0) for d in (3.0, 3.8)]
    st = [dict(seed, **g) for g in grid]
    if kind == "SPH":
        st += [dict(seed, sph_R=R * s, sph_sig=sg, sph_phi=ph * f, **g) for R in (100.0, 150.0, 220.0) for sg in (0.15, 0.35)
               for ph in (0.01, 0.05) for g in grid[::4]]
    elif kind == "TS":
        st += [dict(seed, ts_d=d * s, ts_xi=xi * s, **g) for d in (400.0, 600.0, 900.0) for xi in (80.0, 200.0) for g in grid[::4]]
    elif kind == "SPHPY":
        st += [dict(seed, sph_R=R * s, sph_sig=sg, sph_phi=ph * f, sph_phi_hs=h, **g) for R in (100.0, 180.0, 260.0)
               for sg in (0.15, 0.3) for ph in (0.01, 0.05) for h in (0.1, 0.3) for g in grid[::6]]
    elif kind == "DAB":
        st += [dict(seed, dab_xi=xi * s, **g) for xi in (50.0, 100.0, 200.0) for g in grid[::4]]
    return st


def excess_level(base: PhysFitResult, o: WorkflowOptions) -> float:
    """Largest excess of the data over the base between just above the normalisation window and mid gate window."""
    m = (base.q > o.gate_window[0] * 4 / 3) & (base.q < o.gate_window[1] * 0.5)
    d = base.I[m] - base.model[m]
    return float(np.max(d)) if m.any() and np.max(d) > 0 else 10.0


def highq_curve(fit: PhysFitResult) -> np.ndarray:
    """The fitted high-q term (flat or OZ) on the fit's own q grid."""
    return fit.components["flat"] if "flat" in fit.components else fit.components["oz"]


# ------------------------------------------------------------------------------------------- steps
def _fit(ci: CurveInput, key: str, terms, o: WorkflowOptions, starts=None, prescreen=None,
         q_range=None, sigma=None, I=None, smear: Optional[bool] = None, max_nfev: Optional[int] = None) -> PhysFitResult:
    use_dq = (o.smear if smear is None else smear) and ci.dq is not None
    return fit_physical(f"{ci.name}:{key}", ci.q, ci.I if I is None else I, ci.sigma if sigma is None else sigma, terms,
                        dq=ci.dq if use_dq else None, floor=o.floor, q_range=q_range, starts=starts,
                        n_nodes=o.n_nodes, max_nfev=max_nfev or o.max_nfev, prescreen=prescreen, polish_rounds=o.polish_rounds)


def fit_bases(ci: CurveInput, o: WorkflowOptions) -> Tuple[Dict[str, PhysFitResult], str]:
    """B0 and/or B1 (as allowed by options.highq) and the preferred one."""
    fits = {}
    for k, terms in base_terms(ci.q, ci.I, o).items():
        fits[k] = _fit(ci, k, terms, o, starts=[{"grain_p": p} for p in (3.9, 4.4)], max_nfev=max(o.max_nfev, 4000))
    if "B0" in fits and "B1" in fits:
        b0, b1 = fits["B0"], fits["B1"]
        base = "B1" if (b1.bic < b0.bic - o.dbic_min and not b1.any_at_bound()) else "B0"
    else:
        base = next(iter(fits))
    return fits, base


def fit_gate_span(base: PhysFitResult, o: WorkflowOptions) -> float:
    """Diagnostic: the longest one-signed run of base residuals beyond 1 (span in q, q <= visibility_qmax)."""
    m = base.q <= o.visibility_qmax
    runs = residual_runs(base.q[m], base.residual[m], 1.0)
    return runs[0].span if runs else 0.0


def ratio_to_reference(ci: CurveInput, base: PhysFitResult, ref: CurveInput, ref_base: PhysFitResult,
                       o: WorkflowOptions) -> RatioResult:
    sig = dict(zip(ci.q, ci.sigma))
    sig_r = dict(zip(ref.q, ref.sigma))
    return reference_ratio(base.q, base.I, np.array([sig[x] for x in base.q]), ref_base.q, ref_base.I,
                           np.array([sig_r[x] for x in ref_base.q]), flat=highq_curve(base), flat_ref=highq_curve(ref_base),
                           norm_window=(0.0, o.norm_qmax), threshold=o.gate_sigma)


def _window(rr: RatioResult, o: WorkflowOptions) -> np.ndarray:
    return (rr.q >= o.gate_window[0]) & (rr.q <= o.gate_window[1]) & np.isfinite(rr.ratio)


def null_maximum(refs: Sequence[str], inputs: Mapping[str, CurveInput], bases: Mapping[str, PhysFitResult],
                 o: WorkflowOptions) -> Tuple[float, int]:
    """Largest ratio between any two reference curves over the gate window (all ordered pairs)."""
    vals = []
    for a in refs:
        for b in refs:
            if a == b:
                continue
            rr = ratio_to_reference(inputs[a], bases[a], inputs[b], bases[b], o)
            m = _window(rr, o)
            if m.any():
                vals.append(float(np.max(rr.ratio[m])))
    return (max(vals) if vals else float("nan")), len(vals)


def reference_gate(ci: CurveInput, base: PhysFitResult, ref: CurveInput, ref_base: PhysFitResult, null_max: float,
                   o: WorkflowOptions) -> GateResult:
    rr = ratio_to_reference(ci, base, ref, ref_base, o)
    m = _window(rr, o)
    z = np.where(np.isfinite(rr.ratio), (rr.ratio - 1) / rr.ratio_err, 0.0)
    runs = [u for u in residual_runs(rr.q[m], z[m], o.gate_sigma) if u.sign > 0]
    run = runs[0] if runs else None
    peak = float(np.max(rr.ratio[m])) if m.any() else float("nan")
    q_peak = float(rr.q[m][np.argmax(rr.ratio[m])]) if m.any() else float("nan")
    thr = null_max if np.isfinite(null_max) else 1.0
    passed = bool(run is not None and run.span >= o.gate_span and np.isfinite(peak) and peak > thr)
    note = "" if np.isfinite(null_max) else "no reference/reference null available: peak threshold 1"
    return GateResult("reference", passed, peak, q_peak, run, null_max, note, rr.q, rr.ratio, rr.ratio_err)


def fit_structures(ci: CurveInput, fits: Dict[str, PhysFitResult], base: str, o: WorkflowOptions,
                   progress_note: Callable[[str], None] = lambda s: None) -> None:
    """All structural candidates on every available base (preferred base untagged, the other '@Bx'), in place."""
    s = _scale(ci.q)
    terms_by_base = base_terms(ci.q, ci.I, o)
    kinds = [FAMILIES[f] for f in ("sphere", "teubner_strey", "debye", "unified", "sphere_py") if f in o.families]
    for bk in [base] + [k for k in terms_by_base if k != base]:
        tag = "" if bk == base else f"@{bk}"
        bfit = fits[bk]
        I_exc = excess_level(fits[base], o)
        seed_base = {k: v.value for k, v in bfit.params.items() if v.vary}
        seed_low = dict(seed_base, **grain_seed(ci.q, ci.I, o))
        for kind in kinds:
            if kind == "SPHPY":
                sph = fits.get("SPH" + tag)
                if not isinstance(sph, PhysFitResult):
                    continue
                st = [dict(seed_base, **{k: v.value for k, v in sph.params.items() if k.startswith("sph_") and v.vary},
                           sph_phi_hs=h) for h in (0.05, 0.15, 0.3)]
            else:
                grid = start_grid(kind, I_exc, s, ci)
                st = [dict(seed_low, **x) for x in grid] + [dict(seed_base, **x) for x in grid[:4]]
            progress_note(f"{ci.name}: {kind}{tag}")
            try:
                fits[kind + tag] = _fit(ci, kind + tag, list(terms_by_base[bk]) + structural_terms(kind, ci.q, ci, o), o,
                                        starts=st, prescreen=o.prescreen)
            except Exception as exc:                                              # recorded, never hidden
                fits[kind + tag] = f"failed: {exc}"                               # type: ignore[assignment]
        if not o.second_population:
            continue
        for kind in kinds:
            one = fits.get(kind + tag)
            if not isinstance(one, PhysFitResult):
                continue
            seed = dict({k: v.value for k, v in one.params.items() if v.vary}, **grain_seed(ci.q, ci.I, o))
            key = f"{kind}+UNI2{tag}"
            progress_note(f"{ci.name}: {key}")
            try:
                fits[key] = _fit(ci, key, list(terms_by_base[bk]) + structural_terms(kind, ci.q, ci, o) + [second_term(o)],
                                 o, starts=second_grid(kind, s, seed, ci), prescreen=o.prescreen)
            except Exception as exc:
                fits[key] = f"failed: {exc}"                                      # type: ignore[assignment]


def component_visible(f: PhysFitResult, key: str, o: WorkflowOptions) -> bool:
    """A component is visible if it carries >= visibility_fraction of the model over a contiguous q span >= gate_span
    (q <= visibility_qmax)."""
    m = f.q <= o.visibility_qmax
    share = f.components[key][m] / f.model[m]
    runs = residual_runs(f.q[m], share, o.visibility_fraction)
    return bool(runs and runs[0].sign > 0 and runs[0].span >= o.gate_span)


def sphere_in_window(f: PhysFitResult) -> bool:
    """Spheres must have their Guinier knee inside the window: 1/q_max <= Rg(ensemble) <= 1/q_min."""
    if "sph_R" not in f.params:
        return True
    rg = SphereSchulz().derived(R=f.value("sph_R"), sig=f.value("sph_sig"))["Rg_ensemble"]
    return 1.0 / f.q.max() <= rg <= 1.0 / f.q.min()


def parse_key(k: str, base: str) -> Tuple[str, bool, str]:
    """'TS+UNI2@B1' -> ('TS', True, 'B1'); untagged keys sit on the preferred base."""
    body, _, btag = k.partition("@")
    parts = body.split("+")
    return parts[0], ("UNI2" in parts), (btag or base)


def select(cr: CurveResult, o: WorkflowOptions) -> CurveResult:
    """Apply the retention rules (module docstring, step 4); records a reason list for every candidate."""
    offered = cr.gate.passed or cr.gate.method in ("none", "no reference")
    reasons: Dict[str, List[str]] = {}
    ok_singles = []
    for k, f in cr.fits.items():
        if k in BASES:
            continue
        if not isinstance(f, PhysFitResult):
            reasons[k] = [str(f)]
            continue
        first, two, bk = parse_key(k, cr.base)
        if two:
            continue
        base = cr.fits[bk]
        why = []
        if not offered:
            why.append("gate")
        if not f.success:
            why.append("no convergence")
        if f.any_at_bound():
            why.append("at bound: " + ",".join(f.any_at_bound()))
        if base.bic - f.bic <= o.dbic_min:
            why.append(f"dBIC vs {bk} {base.bic - f.bic:.1f}")
        if not component_visible(f, KEY[first], o):
            why.append("component not visible")
        if not sphere_in_window(f):
            why.append("sphere Rg outside window")
        if first == "SPHPY":
            sph = cr.fits.get(k.replace("SPHPY", "SPH"))
            if isinstance(sph, PhysFitResult) and f.bic >= sph.bic - o.dbic_min:
                why.append("S(q) not needed (dBIC vs SPH <= dbic_min)")
        reasons[k] = why or ["ok"]
        if not why:
            ok_singles.append((f.bic, k))
    retained = min(ok_singles)[1] if ok_singles else cr.base
    if retained not in BASES:
        single = cr.fits[retained]
        ok2 = []
        for k, f in cr.fits.items():
            if k in BASES or not isinstance(f, PhysFitResult):
                continue
            first, two, bk = parse_key(k, cr.base)
            if not two:
                continue
            why = []
            if not f.success:
                why.append("no convergence")
            if f.any_at_bound():
                why.append("at bound: " + ",".join(f.any_at_bound()))
            if single.bic - f.bic <= o.dbic_min:
                why.append(f"dBIC vs {retained} {single.bic - f.bic:.1f}")
            if not component_visible(f, "uni2", o):
                why.append("uni2 not visible")
            if not component_visible(f, KEY[first], o):
                why.append(f"{KEY[first]} not visible")
            if not sphere_in_window(f):
                why.append("sphere Rg outside window")
            reasons[k] = why or ["ok"]
            if not why:
                ok2.append((f.bic, k))
        if ok2:
            retained = min(ok2)[1]
    cr.retained = retained
    cr.reasons = reasons
    cr.derived = derive(cr, o)
    return cr


# ------------------------------------------------------------------------------------------- physics of a result
def derive(cr: CurveResult, o: WorkflowOptions) -> Dict[str, float]:
    """Physical quantities of the retained model: grain exponent, high-q term, and for every structural term its
    sizes, the invariant (when it converges), the volume fraction for the given contrast and the smallest contrast
    able to produce the invariant. Lengths in nm."""
    f = cr.retained_fit
    out: Dict[str, float] = {"grain_p": f.value("grain_p"), "grain_p_err": f.params["grain_p"].stderr or np.nan}
    hq = highq_curve(f)
    out["highq_at_report_q"] = float(np.interp(np.log(o.report_q), np.log(f.q), hq)) if f.q.max() >= o.report_q else np.nan
    if "flat_C" in f.params:
        out.update(highq_C=f.value("flat_C"), highq_C_err=f.params["flat_C"].stderr or np.nan, oz_xi_A=np.nan)
    else:
        out.update(highq_C=f.value("oz_A"), highq_C_err=f.params["oz_A"].stderr or np.nan, oz_xi_A=f.value("oz_xiL"))
    for t in f.terms:
        key = t.key
        if key not in ("sph", "ts", "dab", "uni", "uni2"):
            continue
        comp = COMPONENTS[t.component]()
        kw = {p.name: f.value(t.prefix + p.name) for p in comp.params()}
        err = lambda name: (f.params[t.prefix + name].stderr or np.nan) if (t.prefix + name) in f.params else np.nan   # noqa: E731
        if key == "ts":
            d = TeubnerStrey().derived(**kw)
            out.update(ts_d_nm=kw["d"] / 10, ts_d_err_nm=err("d") / 10, ts_xi_nm=kw["xi"] / 10, ts_xi_err_nm=err("xi") / 10,
                       ts_d_over_xi=kw["d"] / kw["xi"], ts_fa=d["fa"])
        elif key == "sph":
            d = comp.derived(**kw)
            out.update(sph_amplitude=kw["phi"], sph_absolute=bool(cr.contrast), sph_R_nm=kw["R"] / 10,
                       sph_R_err_nm=err("R") / 10, sph_Dn_nm=d["D_number"] / 10, sph_Dvol_nm=d["D_vol_weighted"] / 10,
                       sph_Lvol_nm=d["L_vol_scherrer"] / 10, sph_sig=kw["sig"], sph_Rg_nm=d["Rg_ensemble"] / 10,
                       sph_phi_hs=kw.get("phi_hs", 0.0))
        elif key == "dab":
            out.update(dab_xi_nm=kw["xi"] / 10, dab_xi_err_nm=err("xi") / 10)
        else:
            out.update({f"{key}_Rg_nm": kw["Rg"] / 10, f"{key}_Rg_err_nm": err("Rg") / 10, f"{key}_d": kw["d"],
                        f"{key}_pf": kw["pf"]})
        if key in ("ts", "dab", "sph") or (key in ("uni", "uni2") and kw["d"] >= 3.5):
            fn = lambda x, comp=comp, kw=kw: comp.eval(x, **kw)      # noqa: E731
            Q = invariant(fn)
            out[f"{key}_Q_cm_invA3"] = Q
            out[f"{key}_phi1phi_drho2_1e20cm4"] = two_phase_product_cm4(Q) / 1e20
            out[f"{key}_drho_min_1e10cm2"] = min_contrast_1e10cm2(Q) if key != "sph" or cr.contrast else np.nan
            if cr.contrast:
                out[f"{key}_phi"] = kw["phi"] if key == "sph" else volume_fraction_from_invariant(Q, cr.contrast)
                if cr.contrast_alt:
                    out[f"{key}_phi_alt"] = (kw["phi"] * (cr.contrast / cr.contrast_alt) ** 2 if key == "sph"
                                             else volume_fraction_from_invariant(Q, cr.contrast_alt))
    return out


# ------------------------------------------------------------------------------------------- driver
def run_workflow(inputs: Sequence[CurveInput], options: Optional[WorkflowOptions] = None,
                 progress: Progress = None) -> WorkflowResult:
    """Fit every curve: base fits, gate (reference ratio calibrated on the references), structural candidates for
    the gated curves, selection. Progress: progress(i, n, message)."""
    o = options or WorkflowOptions()
    by_name = {c.name: c for c in inputs}
    if len(by_name) != len(inputs):
        raise ValueError("curve names must be unique")
    missing = sorted({c.reference for c in inputs if c.reference and c.reference not in by_name})
    if missing:
        raise KeyError(f"reference curve(s) not among the inputs: {', '.join(missing)}")
    refs = sorted({c.reference for c in inputs if c.reference})
    n = len(inputs)
    bases: Dict[str, Tuple[Dict[str, PhysFitResult], str]] = {}
    for i, c in enumerate(inputs):
        if progress:
            progress(i, 2 * n, f"base fits: {c.name}")
        bases[c.name] = fit_bases(c, o)
    base_fit = {k: v[0][v[1]] for k, v in bases.items()}
    null_max, n_pairs = (o.gate_null_max, 0) if o.gate_null_max is not None else null_maximum(refs, by_name, base_fit, o)
    curves: Dict[str, CurveResult] = {}
    for i, c in enumerate(inputs):
        if progress:
            progress(n + i, 2 * n, f"structures: {c.name}")
        fits, base = bases[c.name]
        fits = dict(fits)
        if o.gate == "none":
            gate = GateResult("none", True, note="gate disabled")
        elif c.name in refs:
            gate = GateResult("is reference", False, note="reference curve: no structural term offered")
        elif not c.reference:
            gate = (GateResult("no reference", True, note="no reference: structure offered, not corroborated")
                    if o.unreferenced == "offer" else GateResult("no reference", False, note="no reference: skipped"))
        else:
            gate = reference_gate(c, fits[base], by_name[c.reference], base_fit[c.reference], float(null_max), o)
        if gate.passed:
            fit_structures(c, fits, base, o, progress_note=(lambda s, i=i: progress(n + i, 2 * n, s)) if progress else (lambda s: None))
        cr = CurveResult(name=c.name, reference=c.reference, fits=fits, base=base, gate=gate,
                         fit_gate_span=fit_gate_span(fits[base], o), retained=base, contrast=c.contrast,
                         contrast_alt=c.contrast_alt, phi_max=c.phi_max, tags=dict(c.tags))
        curves[c.name] = select(cr, o)
    if progress:
        progress(2 * n, 2 * n, "done")
    return WorkflowResult(options=o, curves=curves, null_max=float(null_max) if null_max is not None else np.nan,
                          null_pairs=n_pairs, created=time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------------------------------- robustness
def _variant_data(ci: CurveInput, v: str, o: WorkflowOptions):
    q, I, s = ci.q, ci.I.copy(), ci.sigma.copy()
    q_range, smear = None, None
    if v == "stat":
        if ci.sigma_stat is None:
            return None
        s = ci.sigma_stat
    elif v == "qmax":
        q_range = (0.0, o.robust_qmax)
    elif v == "noedge":
        q_range = (0.0, float(np.sort(q)[-2]) * 1.0001)
    elif v == "nosmear":
        if o.nosmear_qmin is not None:
            q_range = (o.nosmear_qmin, np.inf)
        elif ci.smeared_flag is not None and ci.smeared_flag.any():
            q_range = (float(q[ci.smeared_flag].max()) * 1.0001, np.inf)
        smear = False
    elif v == "offset":
        ov = o.offset_variant
        if not ov or (ov.get("tag") and ci.tags.get(str(ov["tag"])) != str(ov.get("value"))):
            return None
        f = np.interp(np.log(q), np.log([float(ov["q_lo"]), float(ov["q_hi"])]), [float(ov["factor"]), 1.0])
        I, s = I * f, s * f
    return I, s, q_range, smear


def robustness(result: WorkflowResult, inputs: Sequence[CurveInput],
               variants: Sequence[str] = ("stat", "qmax", "noedge", "nosmear", "offset"),
               progress: Progress = None) -> pd.DataFrame:
    """Refit the retained structural model of every curve on perturbed data, starting from its solution. One row per
    (curve, variant, parameter); ``within_tol`` = within max(robust_tol x |main|, 2 sigma_main)."""
    o = result.options
    by_name = {c.name: c for c in inputs}
    todo = [cr for cr in result.curves.values() if cr.structural and cr.name in by_name]
    rows = []
    for i, cr in enumerate(todo):
        if progress:
            progress(i, len(todo), f"robustness: {cr.name}")
        f0 = cr.retained_fit
        ci = by_name[cr.name]
        start = {k: v.value for k, v in f0.params.items() if v.vary}
        for v in variants:
            d = _variant_data(ci, v, o)
            if d is None:
                continue
            I, s, q_range, smear = d
            try:
                f = _fit(ci, f"{cr.retained}:{v}", f0.terms, o, starts=[start], q_range=q_range, sigma=s, I=I, smear=smear)
            except Exception as exc:
                rows.append(dict(curve=cr.name, model=cr.retained, variant=v, param="(fit)", note=f"failed: {exc}"))
                continue
            for k, pv in f.params.items():
                if not pv.vary:
                    continue
                m = f0.params[k]
                tol = max(o.robust_tol * abs(m.value), 2 * (m.stderr or 0.0))
                rows.append(dict(curve=cr.name, model=cr.retained, variant=v, param=k, main=m.value, main_stderr=m.stderr,
                                 value=pv.value, stderr=pv.stderr, rel_change=(pv.value / m.value - 1) if m.value else np.nan,
                                 within_tol=bool(abs(pv.value - m.value) <= tol), at_bound=bool(pv.at_bound),
                                 n_points=f.n_points, n_excluded=f.n_excluded, excluded=f.excluded_reason,
                                 chi2_red=f.redchi, chi2_red_main=f0.redchi, note=""))
    if progress:
        progress(len(todo), len(todo), "robustness done")
    df = pd.DataFrame(rows)
    result.robustness = df
    return df


STRUCT_PREFIXES = ("sph_", "ts_", "dab_", "uni_", "uni2_")


def robust_flags(df: Optional[pd.DataFrame], curve: str) -> Dict[str, object]:
    """Per curve: robust_primary (all parameters of the primary structure), robust_second (the second population),
    the failing (parameter:variant) pairs and the variants run."""
    if df is None or not len(df) or "curve" not in df:
        return {}
    rr = df[(df.curve == curve) & (df.param != "(fit)")]
    if not len(rr):
        return {}
    bad = rr[(~rr.within_tol.astype(bool)) | (rr.at_bound.astype(bool))]
    prim = tuple(p for p in STRUCT_PREFIXES if p != "uni2_")
    has2 = rr.param.str.startswith("uni2_").any()
    return {"robust_primary": bool(len(bad[bad.param.str.startswith(prim)]) == 0),
            "robust_second": (bool(len(bad[bad.param.str.startswith("uni2_")]) == 0) if has2 else None),
            "robustness_failures": ";".join(sorted(set(f"{a}:{b}" for a, b in zip(bad.param, bad.variant)))),
            "robust_variants": ";".join(sorted(set(rr.variant)))}


def param_robust(df: Optional[pd.DataFrame], curve: str, param: str) -> bool:
    """True when the parameter passed every robustness variant (or none was run)."""
    if df is None or not len(df) or "curve" not in df:
        return True
    rr = df[(df.curve == curve) & (df.param == param)]
    return bool(len(rr) == 0 or (rr.within_tol.astype(bool).all() and not rr.at_bound.astype(bool).any()))


# ------------------------------------------------------------------------------------------- tables
def summary_table(result: WorkflowResult, extra: Optional[Mapping[str, Mapping[str, object]]] = None) -> pd.DataFrame:
    """One row per curve: gate, base, retained model, fit quality, BIC margins, physical quantities, robustness.
    ``extra``: {curve: {column: value}} prepended (compositions, x-values...)."""
    rows = []
    for name, cr in result.curves.items():
        base, ret = cr.fits[cr.base], cr.retained_fit
        alts = [(f.bic, k) for k, f in cr.fits.items() if isinstance(f, PhysFitResult) and k != cr.retained
                and k not in BASES and cr.reasons.get(k) == ["ok"]]
        alt = min(alts) if alts else None
        g = cr.gate
        row = dict(extra.get(name, {}) if extra else {})
        row.update(curve=name, reference=cr.reference or "", gate=g.method, gate_passed=g.passed, gate_peak=g.peak,
                   gate_q_peak=g.q_peak, gate_run_span=(g.run.span if g.run else np.nan),
                   gate_run_q=(f"{g.run.q_lo:.4g}-{g.run.q_hi:.4g}" if g.run else ""), gate_null_max=g.null_max,
                   fit_gate_span=cr.fit_gate_span, base=cr.base, retained=cr.retained,
                   retained_title=_model_title(cr.retained), n_points=ret.n_points, n_excluded=ret.n_excluded,
                   chi2_red=ret.redchi, rms_log10=ret.rms_log, dBIC_retained_vs_base=base.bic - ret.bic,
                   next_best=alt[1] if alt else "", dBIC_next_minus_retained=(alt[0] - ret.bic) if alt else np.nan,
                   at_bound=";".join(ret.any_at_bound()), contrast_1e10cm2=cr.contrast, phi_max=cr.phi_max)
        row.update(cr.derived)
        row.update(robust_flags(result.robustness, name))
        rows.append(row)
    df = pd.DataFrame(rows)
    first = [c for c in dict.fromkeys(k for v in (extra or {}).values() for k in v) if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _model_title(key: str) -> str:
    first, two, bk = parse_key(key, "")
    t = FAMILY_TITLES.get(first, first)
    if two:
        t += " + small population"
    if bk and first not in BASES:
        t += f" (on {bk})"
    return t


def comparison_table(result: WorkflowResult) -> pd.DataFrame:
    """Every model of every curve: chi2, k, BIC, dBIC vs the preferred base, parameters at bound, selection verdict."""
    rows = []
    for name, cr in result.curves.items():
        base = cr.fits[cr.base]
        for k, f in cr.fits.items():
            if not isinstance(f, PhysFitResult):
                rows.append(dict(curve=name, model=k, status=str(f), retained=False))
                continue
            rows.append(dict(curve=name, model=k, title=_model_title(k), status="ok" if f.success else f"no convergence: {f.message}",
                             n_points=f.n_points, n_free=f.n_free, chi2=f.chi2, chi2_red=f.redchi, rms_log10=f.rms_log,
                             BIC=f.bic, dBIC_vs_base=base.bic - f.bic, at_bound=";".join(f.any_at_bound()),
                             verdict="; ".join(cr.reasons.get(k, ["base"] if k in BASES else [])), retained=(k == cr.retained)))
    return pd.DataFrame(rows)


def parameters_table(result: WorkflowResult) -> pd.DataFrame:
    """Every parameter of every model: value, standard error, free/fixed, bounds, at-bound flag, unit."""
    rows = []
    for name, cr in result.curves.items():
        for k, f in cr.fits.items():
            if not isinstance(f, PhysFitResult):
                continue
            units = {}
            for t in f.terms:
                for p in COMPONENTS[t.component]().params():
                    units[t.prefix + p.name] = p.unit
            for pk, pv in f.params.items():
                rows.append(dict(curve=name, model=k, retained=(k == cr.retained), param=pk, value=pv.value, stderr=pv.stderr,
                                 free=pv.vary, lower=pv.lo if pv.vary else np.nan, upper=pv.hi if pv.vary else np.nan,
                                 at_bound=pv.at_bound, unit=units.get(pk, "")))
    return pd.DataFrame(rows)


def curve_table(cr: CurveResult, key: Optional[str] = None) -> pd.DataFrame:
    """q, I, model, components and normalised residual of one fit (default: the retained one)."""
    f = cr.fits[key or cr.retained]
    df = pd.DataFrame({"q_invA": f.q, "I": f.I, "model": f.model, "residual_norm": f.residual})
    for ck, cv in f.components.items():
        df[f"comp_{ck}"] = cv
    return df


def method_notes(result: WorkflowResult) -> List[str]:
    o = result.options
    nm = result.null_max
    return [
        "Fits in ln I with weights 1/sqrt((sigma/I)^2 + %g^2)%s; BIC = chi2 + k ln n." % (o.floor, ", Gaussian pinhole smearing with dQ" if o.smear else ""),
        "Base: grain power law (exponent %g-%g) + flat level, or + Ornstein-Zernike term (xi %g-%g A), preferred by dBIC > %g." % (
            o.grain_p_bounds[0], o.grain_p_bounds[1], o.oz_xi_bounds[0], o.oz_xi_bounds[1], o.dbic_min),
        "Gate: ratio to the reference curve (high-q terms subtracted, normalised at q <= %g 1/A) above 1 by more than %g sigma "
        "over a run spanning >= x%g inside q = %g-%g 1/A, peak > %s." % (o.norm_qmax, o.gate_sigma, o.gate_span, o.gate_window[0],
                                                                         o.gate_window[1], f"{nm:.3g} (largest reference/reference ratio, {result.null_pairs} pairs)" if np.isfinite(nm) else "1"),
        "Retention: gate passed, converged, no free parameter at a bound, dBIC > %g over the base, component >= %g %% of the model "
        "over >= x%g in q (q <= %g 1/A), spheres with Rg inside 1/q_max..1/q_min; a second population needs dBIC > %g over the "
        "retained single structure." % (o.dbic_min, 100 * o.visibility_fraction, o.gate_span, o.visibility_qmax, o.dbic_min),
        "Model keys: SPH spheres, SPHPY spheres + Percus-Yevick S(q), TS Teubner-Strey, DAB Debye, UNI unified level (tied B), "
        "+UNI2 a second small unified population, @B0/@B1 the non-preferred base.",
    ]
