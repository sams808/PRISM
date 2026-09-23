"""Two-level unified (Beaucage) fits with a restricted parameter budget, BIC forward selection and a
robustness pass: the model retained by the September 2026 rethink (``phase3e_beaucage_bic.py``).

    I(q) = L1 + [L2] + C [+ T q^2]
    Lk   = Gk exp(-q^2 Rgk^2 / 3) + Bk [erf(q Rgk / sqrt 6)]^(3 dk) q^-dk       (Beaucage 1995; Tomita 2024 Eq. 10)

* L1: the powder grains (Rg1 25-150 nm, d1 3-4.5, B1 free: the Porod term of the coarser grains
  folds into it);
* L2 (optional): a nanoscale level (Rg2 4-20 nm, d2 2-4.5) whose B2 is tied to G2 by the Beaucage
  (1996) relation times a polydispersity factor pf2 in 0.1-1.5, so a level is a population with a
  Guinier knee or it is not kept: it cannot degenerate into a power law;
* C: flat background; T q^2 (optional): the rising flank of the amorphous halo.

L2 and the tail are added by forward selection only when they lower the BIC by more than
``dbic_min`` (10). The fit is of log10 I with the propagated sigma and a systematic floor (3 %),
on the usable range rebinned to 40 points per decade, q <= 0.20 1/A, two starting radii per level.
Robustness: the fit is repeated with relaxed L1 bounds (d1 <= 5, Rg1 <= 300 nm); a level is
"resolved" only when both keep it with Rg2 agreeing within 20 %. Every parameter at a bound is
flagged; a level is never kept with G2 below ``amplitude_floor`` of the curve maximum.

The formulas are evaluated through :mod:`saxs.core.composite_models` (``beaucage_unified``,
``beaucage_tied``, ``flat_background``, ``rising_tail``), so both engines share one implementation.
Amplitudes are parametrised in log10 (G, B, C, T, pf). Pure numpy / lmfit; sample identities are
strings carried in the curve metadata.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .composite_models import BeaucageTied, BeaucageUnified, FlatBackground, RisingTail, beaucage_polydisperse_prefactor
from .curve import Curve
from .descriptors import rebin_log

Progress = Optional[Callable[[int, int, str], None]]
LN10 = math.log(10.0)
_L1, _L2, _BG, _TAIL = BeaucageUnified(), BeaucageTied(), FlatBackground(), RisingTail()


# ------------------------------------------------------------------------------------ options
@dataclass
class UnifiedOptions:
    q_max: float = 0.20
    per_decade: int = 40
    sigma_floor: float = 0.03
    dbic_min: float = 10.0
    rg1_bounds: Tuple[float, float] = (250.0, 1500.0)
    d1_bounds: Tuple[float, float] = (3.0, 4.5)
    rg2_bounds: Tuple[float, float] = (40.0, 200.0)
    d2_bounds: Tuple[float, float] = (2.0, 4.5)
    logpf2_bounds: Tuple[float, float] = (-1.0, 0.18)
    logC_bounds: Tuple[float, float] = (-9.0, 5.0)
    logG1_bounds: Tuple[float, float] = (-9.0, 12.0)
    logB1_bounds: Tuple[float, float] = (-30.0, 10.0)
    logG2_bounds: Tuple[float, float] = (-9.0, 9.0)
    logT_bounds: Tuple[float, float] = (-12.0, 8.0)
    rg1_starts: Tuple[float, ...] = (400.0, 700.0)
    rg2_starts: Tuple[float, ...] = (80.0, 150.0)
    d1_start: float = 4.0
    d2_start: float = 3.5
    logpf2_start: float = -0.3
    tail_exponent: float = 2.0
    max_nfev: int = 4000
    min_points: int = 30
    q_min_required: float = 0.01
    amplitude_floor: float = 1e-6
    bound_tol: float = 0.01
    relaxed_d1_max: float = 5.0
    relaxed_rg1_max: float = 3000.0
    robust_tol: float = 0.20
    use_usable: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> "UnifiedOptions":
        out = cls()
        for k, v in dict(d or {}).items():
            if not hasattr(out, k):
                raise KeyError(f"unknown unified-fit option {k!r}")
            if isinstance(getattr(out, k), tuple):
                v = tuple(v)
            setattr(out, k, v)
        return out

    def relaxed(self) -> "UnifiedOptions":
        """The robustness variant: relaxed upper bounds of the grain level."""
        return replace(self, d1_bounds=(self.d1_bounds[0], self.relaxed_d1_max),
                       rg1_bounds=(self.rg1_bounds[0], self.relaxed_rg1_max))


def candidate_name(l2: bool, tail: bool) -> str:
    return "L1" + ("_L2" if l2 else "") + ("_T" if tail else "")


CANDIDATES: Tuple[Tuple[bool, bool], ...] = ((False, False), (False, True), (True, False), (True, True))


# ------------------------------------------------------------------------------------ model
def components(values: Dict[str, float], q: np.ndarray, l2: bool, tail: bool, tail_exponent: float = 2.0) -> Dict[str, np.ndarray]:
    """Model components from a ``{name: value}`` dict (log10 amplitudes), through the shared
    composite components."""
    q = np.asarray(q, float)
    v = values
    c = {"L1": _L1.eval(q, G=10 ** v["logG1"], Rg=v["Rg1"], B=10 ** v["logB1"], p=v["d1"])}
    if l2:
        c["L2"] = _L2.eval(q, G=10 ** v["logG2"], Rg=v["Rg2"], d=v["d2"], pf=10 ** v["logpf2"])
    bg = _BG.eval(q, C=10 ** v["logC"])
    if tail:
        bg = bg + _TAIL.eval(q, T=10 ** v["logT"], m=tail_exponent)
    c["background"] = bg
    return c


def model(values: Dict[str, float], q: np.ndarray, l2: bool, tail: bool, tail_exponent: float = 2.0) -> np.ndarray:
    return sum(components(values, q, l2, tail, tail_exponent).values())


def _values(params) -> Dict[str, float]:
    return {k: float(p.value) for k, p in params.items()}


def _residual(params, q, y, w, l2, tail, tail_exponent):
    return (np.log10(np.maximum(model(_values(params), q, l2, tail, tail_exponent), 1e-300)) - y) * w


def log_weights(I: np.ndarray, sigma: np.ndarray, sigma_floor: float) -> np.ndarray:
    """1 / sigma_log10 with the propagated sigma and a relative floor added in quadrature."""
    return 1.0 / np.sqrt((sigma / (I * LN10)) ** 2 + (sigma_floor / LN10) ** 2)


def prepare_curve(curve: Curve, options: Optional[UnifiedOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """(q, I, sigma) of the fit input: usable points (when the curve carries a ``usable`` column and
    ``use_usable``), q <= q_max, I > 0, rebinned to ``per_decade``. The fourth element is a reason
    when the curve cannot be fitted (too few points, or not usable below ``q_min_required``)."""
    o = options or UnifiedOptions()
    q = np.asarray(curve.q, float)
    I = np.asarray(curve.intensity, float)
    sig = np.asarray(curve.sigma, float) if curve.sigma is not None else np.zeros_like(I)
    keep = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0) & (q <= o.q_max)
    cols = (curve.metadata or {}).get("columns") or {}
    if o.use_usable and "usable" in cols and len(cols["usable"]) == len(q):
        keep &= np.asarray(cols["usable"], float).astype(bool)
    q, I, sig = q[keep], I[keep], sig[keep]
    if len(q) < o.min_points:
        return q, I, sig, f"only {len(q)} usable points (need {o.min_points})"
    if q.min() > o.q_min_required:
        return q, I, sig, f"usable only above q = {q.min():.3g} (need below {o.q_min_required:g})"
    qb, Ib, sb = rebin_log(q, I, sig, o.per_decade)
    ok = np.isfinite(Ib) & (Ib > 0)
    return qb[ok], Ib[ok], sb[ok], None


def make_parameters(q: np.ndarray, I: np.ndarray, l2: bool, tail: bool, rg1_0: float, rg2_0: Optional[float],
                    options: UnifiedOptions):
    """Starting values and bounds of one candidate (the rethink's seeding rules)."""
    from lmfit import Parameters

    o = options
    hi = q >= 0.12
    C0 = float(np.median(I[hi])) if hi.sum() >= 3 else float(I[-1])
    pr = Parameters()
    pr.add("logC", value=np.log10(max(C0 * 0.7, 1e-9)), min=o.logC_bounds[0], max=o.logC_bounds[1])
    G1 = max(np.interp(min(1.0 / rg1_0, q.max()), q, I), 1e-8) * np.exp(1.0 / 3)
    pr.add("logG1", value=np.log10(G1), min=o.logG1_bounds[0], max=o.logG1_bounds[1])
    pr.add("Rg1", value=rg1_0, min=o.rg1_bounds[0], max=o.rg1_bounds[1])
    pr.add("d1", value=o.d1_start, min=o.d1_bounds[0], max=o.d1_bounds[1])
    pr.add("logB1", value=np.log10(3 * 1.62 * G1 / rg1_0 ** 4), min=o.logB1_bounds[0], max=o.logB1_bounds[1])
    if l2:
        G2 = max(np.interp(min(1.0 / rg2_0, q.max()), q, I) * 0.3, 1e-8)
        pr.add("logG2", value=np.log10(G2), min=o.logG2_bounds[0], max=o.logG2_bounds[1])
        pr.add("Rg2", value=rg2_0, min=o.rg2_bounds[0], max=o.rg2_bounds[1])
        pr.add("d2", value=o.d2_start, min=o.d2_bounds[0], max=o.d2_bounds[1])
        pr.add("logpf2", value=o.logpf2_start, min=o.logpf2_bounds[0], max=o.logpf2_bounds[1])
    if tail:
        pr.add("logT", value=np.log10(max(C0 * 0.3 / 0.15 ** 2, 1e-9)), min=o.logT_bounds[0], max=o.logT_bounds[1])
    return pr


def fit_candidate(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, l2: bool, tail: bool, options: UnifiedOptions):
    """Best of the multi-start least-squares fits of one candidate (lmfit MinimizerResult), None
    when every start failed."""
    from lmfit import Minimizer

    y = np.log10(I)
    w = log_weights(I, sigma, options.sigma_floor)
    best = None
    for rg1_0 in options.rg1_starts:
        for rg2_0 in (options.rg2_starts if l2 else (None,)):
            pr = make_parameters(q, I, l2, tail, rg1_0, rg2_0, options)
            try:
                r = Minimizer(_residual, pr, fcn_args=(q, y, w, l2, tail, options.tail_exponent)).minimize(
                    method="least_squares", max_nfev=options.max_nfev)
            except Exception:
                continue
            if best is None or r.chisqr < best.chisqr:
                best = r
    return best


def forward_select(bics: Dict[Tuple[bool, bool], float], g2: Dict[Tuple[bool, bool], Optional[float]], I_max: float,
                   options: UnifiedOptions) -> Tuple[Tuple[bool, bool], List[Tuple[Tuple[bool, bool], float]]]:
    """Forward selection from (L1) over the available candidates: at each step the candidate with one
    more term and the lowest BIC is adopted when it lowers the BIC by more than ``dbic_min``. An L2
    candidate whose G2 is below ``amplitude_floor`` x I_max is never eligible (a fake level).
    Returns (chosen, path of (candidate, BIC))."""
    chosen = (False, False)
    path = [(chosen, bics[chosen])]
    for _ in range(2):
        options_ = [k for k in bics if k != chosen and sum(k) == sum(chosen) + 1 and all(a >= b for a, b in zip(k, chosen))]
        options_ = [k for k in options_ if not (k[0] and g2.get(k) is not None and g2[k] < options.amplitude_floor * I_max)]
        if not options_:
            break
        best = min(options_, key=lambda k: bics[k])
        if bics[chosen] - bics[best] > options.dbic_min:
            chosen = best
            path.append((chosen, bics[chosen]))
        else:
            break
    return chosen, path


def is_at_bound(value: float, lo: float, hi: float, tol: float = 0.01) -> bool:
    """The rethink's rule for positive bounds (value within ``tol`` relative of a bound); for bounds
    at or below zero the same tolerance is taken on the bound span."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    if lo > 0:
        return bool(value <= lo * (1 + tol) or value >= hi * (1 - tol))
    span = hi - lo
    return bool(value - lo <= tol * span or hi - value <= tol * span)


# ------------------------------------------------------------------------------------ results
@dataclass
class ParamValue:
    value: float
    stderr: Optional[float]
    min: float
    max: float
    at_bound: bool
    vary: bool = True


@dataclass
class UnifiedFitResult:
    sample: str
    cls: str = ""
    fitted: bool = True
    skip_reason: Optional[str] = None
    l2: bool = False
    tail: bool = False
    n_pts: int = 0
    q_min: float = float("nan")
    q_max: float = float("nan")
    n_params: int = 0
    chi2r: float = float("nan")
    rms_log: float = float("nan")
    chi2r_base: float = float("nan")
    rms_base: float = float("nan")
    bic: float = float("nan")
    dBIC_vs_base: float = float("nan")
    dBIC_L2: float = float("nan")
    dBIC_tail: float = float("nan")
    params: Dict[str, ParamValue] = field(default_factory=dict)
    derived: Dict[str, float] = field(default_factory=dict)
    candidates: Dict[str, Dict[str, object]] = field(default_factory=dict)
    selection_path: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    covariance_ok: bool = False
    per_mm: Optional[bool] = None
    q: np.ndarray = field(default_factory=lambda: np.array([]))
    I: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    options: Dict[str, object] = field(default_factory=dict)
    relaxed: Optional["UnifiedFitResult"] = None
    resolved: Optional[bool] = None

    # -- evaluation -----------------------------------------------------------------
    @property
    def values(self) -> Dict[str, float]:
        return {k: p.value for k, p in self.params.items()}

    @property
    def name(self) -> str:
        return candidate_name(self.l2, self.tail)

    def model(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        q = self.q if q is None else np.asarray(q, float)
        return model(self.values, q, self.l2, self.tail, float(self.options.get("tail_exponent", 2.0)))

    def components(self, q: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        q = self.q if q is None else np.asarray(q, float)
        return components(self.values, q, self.l2, self.tail, float(self.options.get("tail_exponent", 2.0)))

    def residual_log(self) -> np.ndarray:
        return np.log10(self.I) - np.log10(np.maximum(self.model(), 1e-300))

    def at_bound_names(self) -> List[str]:
        return [k for k, p in self.params.items() if p.at_bound]

    # -- serialisation --------------------------------------------------------------
    def to_json(self) -> Dict[str, object]:
        d = {k: v for k, v in asdict(self).items() if k not in ("q", "I", "sigma", "relaxed")}
        d["q"], d["I"], d["sigma"] = self.q.tolist(), self.I.tolist(), self.sigma.tolist()
        d["relaxed"] = self.relaxed.to_json() if self.relaxed is not None else None
        return d

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "UnifiedFitResult":
        d = dict(d)
        params = {k: ParamValue(**v) for k, v in (d.pop("params") or {}).items()}
        relaxed = d.pop("relaxed", None)
        q, I, s = (np.asarray(d.pop(k, []), float) for k in ("q", "I", "sigma"))
        out = cls(params=params, q=q, I=I, sigma=s, **d)
        out.relaxed = cls.from_json(relaxed) if relaxed else None
        return out


# ------------------------------------------------------------------------------------ fitting
def _param_values(r, options: UnifiedOptions) -> Dict[str, ParamValue]:
    out = {}
    for k, p in r.params.items():
        lo, hi = float(p.min), float(p.max)
        out[k] = ParamValue(value=float(p.value), stderr=(float(p.stderr) if p.stderr is not None and np.isfinite(p.stderr) else None),
                            min=lo, max=hi, at_bound=is_at_bound(float(p.value), lo, hi, options.bound_tol), vary=bool(p.vary))
    return out


def _derived(values: Dict[str, float], l2: bool, tail: bool) -> Dict[str, float]:
    d = {"Rg1_nm": values["Rg1"] / 10, "D1_eq_nm": 2 * values["Rg1"] * math.sqrt(5 / 3) / 10, "d1": values["d1"],
         "G1": 10 ** values["logG1"], "B1": 10 ** values["logB1"], "C": 10 ** values["logC"]}
    if tail:
        d["T"] = 10 ** values["logT"]
    if l2:
        G2 = 10 ** values["logG2"]
        d.update({"Rg2_nm": values["Rg2"] / 10, "D2_eq_nm": 2 * values["Rg2"] * math.sqrt(5 / 3) / 10, "d2": values["d2"],
                  "G2": G2, "pf2": 10 ** values["logpf2"],
                  "B2": 10 ** values["logpf2"] * beaucage_polydisperse_prefactor(G2, values["Rg2"], values["d2"])})
    return d


def _rms_log(r, q, y, l2, tail, tail_exponent) -> float:
    return float(np.sqrt(np.mean((np.log10(np.maximum(model(_values(r.params), q, l2, tail, tail_exponent), 1e-300)) - y) ** 2)))


def fit_unified_levels(curve: Curve, options: Optional[UnifiedOptions] = None) -> UnifiedFitResult:
    """Fit the four candidates (L1, L1+T, L1+L2, L1+L2+T) and select forward on the BIC."""
    o = options or UnifiedOptions()
    md = curve.metadata or {}
    name = str(md.get("sample") or curve.name)
    res = UnifiedFitResult(sample=name, cls=str(md.get("class_mu_t", "")), options=o.to_dict(), per_mm=md.get("per_mm"))
    q, I, sig, reason = prepare_curve(curve, o)
    res.q, res.I, res.sigma = q, I, sig
    if reason is not None:
        res.fitted, res.skip_reason = False, reason
        return res
    res.n_pts, res.q_min, res.q_max = int(len(q)), float(q.min()), float(q.max())
    y = np.log10(I)
    fits = {}
    for l2, tail in CANDIDATES:
        r = fit_candidate(q, I, sig, l2, tail, o)
        if r is not None:
            fits[(l2, tail)] = r
    if (False, False) not in fits:
        res.fitted, res.skip_reason = False, "the base fit (L1 + C) failed"
        return res
    bics = {k: float(r.bic) for k, r in fits.items()}
    g2 = {k: (10 ** float(r.params["logG2"].value) if k[0] else None) for k, r in fits.items()}
    I_max = float(np.max(I))
    for k, r in fits.items():
        res.candidates[candidate_name(*k)] = {"bic": float(r.bic), "chisqr": float(r.chisqr), "chi2r": float(r.redchi),
                                              "rms_log": _rms_log(r, q, y, k[0], k[1], o.tail_exponent), "n_params": int(r.nvarys),
                                              "success": bool(r.success), "G2": g2[k],
                                              "fake_level": bool(k[0] and g2[k] is not None and g2[k] < o.amplitude_floor * I_max)}
    chosen, path = forward_select(bics, g2, I_max, o)
    res.selection_path = [candidate_name(*k) for k, _ in path]
    r = fits[chosen]
    res.l2, res.tail = chosen
    res.n_params = int(r.nvarys)
    res.chi2r, res.bic = float(r.redchi), float(r.bic)
    res.rms_log = _rms_log(r, q, y, res.l2, res.tail, o.tail_exponent)
    base = fits[(False, False)]
    res.chi2r_base = float(base.redchi)
    res.rms_base = _rms_log(base, q, y, False, False, o.tail_exponent)
    res.dBIC_vs_base = float(base.bic - r.bic)
    if (True, res.tail) in fits and (False, res.tail) in fits:
        res.dBIC_L2 = float(fits[(False, res.tail)].bic - fits[(True, res.tail)].bic)
    if (res.l2, True) in fits and (res.l2, False) in fits:
        res.dBIC_tail = float(fits[(res.l2, False)].bic - fits[(res.l2, True)].bic)
    res.params = _param_values(r, o)
    res.derived = _derived(res.values, res.l2, res.tail)
    res.covariance_ok = bool(getattr(r, "errorbars", False)) and all(p.stderr is not None for p in res.params.values())
    if not res.covariance_ok:
        res.warnings.append("covariance unavailable (a parameter sits at a bound or the problem is degenerate)")
    for k in res.at_bound_names():
        res.warnings.append(f"{k} at a bound")
    for k, c in res.candidates.items():
        if c["fake_level"]:
            res.warnings.append(f"candidate {k} rejected: G2 below {o.amplitude_floor:g} of the curve maximum")
    return res


def robustness_pass(curve: Curve, result: UnifiedFitResult, options: Optional[UnifiedOptions] = None) -> UnifiedFitResult:
    """Refit with the relaxed grain-level bounds; ``result.relaxed`` holds the variant and
    ``result.resolved`` says whether level 2 is kept by both with Rg2 within ``robust_tol``
    (None when the retained fit has no level 2)."""
    o = UnifiedOptions.from_dict(result.options) if options is None else options
    result.relaxed = fit_unified_levels(curve, o.relaxed())
    result.resolved = resolved_flag(result, result.relaxed, o.robust_tol)
    return result


def resolved_flag(result: UnifiedFitResult, relaxed: Optional[UnifiedFitResult], tol: float = 0.20) -> Optional[bool]:
    if not result.fitted or not result.l2:
        return None
    if relaxed is None or not relaxed.fitted or not relaxed.l2:
        return False
    a, b = result.params["Rg2"].value, relaxed.params["Rg2"].value
    return bool(abs(b - a) <= tol * a)


def fit_with_robustness(curve: Curve, options: Optional[UnifiedOptions] = None) -> UnifiedFitResult:
    o = options or UnifiedOptions()
    res = fit_unified_levels(curve, o)
    if res.fitted:
        robustness_pass(curve, res, o)
    return res


# ------------------------------------------------------------------------------------ batch / tables
TABLE_COLUMNS = ["sample", "cls", "fitted", "skip_reason", "n_pts", "q_min", "n_params", "l2", "tail", "chi2r", "rms_log", "chi2r_base",
                 "rms_base", "dBIC_vs_base", "dBIC_L2", "dBIC_tail", "logC", "logT", "Rg1_A", "Rg1_err", "d1", "d1_err", "logG1",
                 "logB1", "Rg1_at_bound", "d1_at_bound", "Rg2_A", "Rg2_err", "d2", "d2_err", "logG2", "logpf2", "logB2",
                 "Rg2_at_bound", "d2_at_bound", "pf2_at_bound", "D2_eq_nm", "covariance_ok", "at_bounds", "relaxed_l2",
                 "relaxed_Rg2_A", "relaxed_d1", "relaxed_Rg1_A", "relaxed_rms_log", "resolved", "warnings"]


def result_row(res: UnifiedFitResult) -> Dict[str, object]:
    nan = float("nan")
    p = res.params

    def val(k):
        return p[k].value if k in p else nan

    def err(k):
        return (p[k].stderr if (k in p and p[k].stderr is not None) else nan)

    row = {"sample": res.sample, "cls": res.cls, "fitted": res.fitted, "skip_reason": res.skip_reason or "", "n_pts": res.n_pts,
           "q_min": res.q_min, "n_params": res.n_params, "l2": res.l2, "tail": res.tail, "chi2r": res.chi2r, "rms_log": res.rms_log,
           "chi2r_base": res.chi2r_base, "rms_base": res.rms_base, "dBIC_vs_base": res.dBIC_vs_base, "dBIC_L2": res.dBIC_L2,
           "dBIC_tail": res.dBIC_tail, "logC": val("logC"), "logT": val("logT"), "Rg1_A": val("Rg1"), "Rg1_err": err("Rg1"),
           "d1": val("d1"), "d1_err": err("d1"), "logG1": val("logG1"), "logB1": val("logB1"),
           "Rg1_at_bound": bool(p["Rg1"].at_bound) if "Rg1" in p else False, "d1_at_bound": bool(p["d1"].at_bound) if "d1" in p else False,
           "Rg2_A": val("Rg2"), "Rg2_err": err("Rg2"), "d2": val("d2"), "d2_err": err("d2"), "logG2": val("logG2"), "logpf2": val("logpf2"),
           "logB2": (math.log10(res.derived["B2"]) if "B2" in res.derived and res.derived["B2"] > 0 else nan),
           "Rg2_at_bound": bool(p["Rg2"].at_bound) if "Rg2" in p else False, "d2_at_bound": bool(p["d2"].at_bound) if "d2" in p else False,
           "pf2_at_bound": bool(p["logpf2"].at_bound) if "logpf2" in p else False, "D2_eq_nm": res.derived.get("D2_eq_nm", nan),
           "covariance_ok": res.covariance_ok, "at_bounds": ";".join(res.at_bound_names()),
           "relaxed_l2": (res.relaxed.l2 if res.relaxed is not None else None),
           "relaxed_Rg2_A": (res.relaxed.params["Rg2"].value if (res.relaxed is not None and "Rg2" in res.relaxed.params) else nan),
           "relaxed_d1": (res.relaxed.params["d1"].value if (res.relaxed is not None and "d1" in res.relaxed.params) else nan),
           "relaxed_Rg1_A": (res.relaxed.params["Rg1"].value if (res.relaxed is not None and "Rg1" in res.relaxed.params) else nan),
           "relaxed_rms_log": (res.relaxed.rms_log if res.relaxed is not None else nan),
           "resolved": res.resolved, "warnings": "; ".join(res.warnings)}
    return row


def results_table(results: Iterable[UnifiedFitResult]) -> pd.DataFrame:
    rows = [result_row(r) for r in results]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS) if rows else pd.DataFrame(columns=TABLE_COLUMNS)


def batch_fit(curves: Iterable[Curve], options: Optional[UnifiedOptions] = None, robustness: bool = True,
              progress: Progress = None) -> Tuple[List[UnifiedFitResult], pd.DataFrame]:
    """Fit every curve (with the robustness pass by default) -> (results, table)."""
    o = options or UnifiedOptions()
    curves = list(curves)
    results: List[UnifiedFitResult] = []
    n = len(curves)
    for i, c in enumerate(curves):
        name = str((c.metadata or {}).get("sample") or c.name)
        if progress is not None:
            progress(i, n, name)
        results.append(fit_with_robustness(c, o) if robustness else fit_unified_levels(c, o))
    if progress is not None:
        progress(n, n, "done")
    return results, results_table(results)


def save_results_json(results: Sequence[UnifiedFitResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"format": "prism-unified-levels", "version": 1, "results": [r.to_json() for r in results]}, fh, indent=1)


def load_results_json(path: str) -> List[UnifiedFitResult]:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("format") != "prism-unified-levels":
        raise ValueError(f"{path} is not a unified-levels result file")
    return [UnifiedFitResult.from_json(r) for r in d["results"]]
