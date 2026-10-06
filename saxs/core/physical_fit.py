"""Physically constrained fits of absolute 1D SAXS curves: sums of composite_models components, optionally smeared by
the per-point Gaussian resolution, fitted in log I with sigma weights and a relative floor.

Sample-agnostic building blocks:
  - `TermSpec` / `fit_physical`: one model = an ordered list of components (with per-term seeds, bounds, fixed
    parameters, and a q window outside which a term is zero is NOT offered: terms are global), fitted by
    lmfit/least_squares on r_i = (ln M_i - ln I_i) / sqrt((sigma_i/I_i)^2 + floor^2), M = resolution-smeared model.
    Several starting points can be given; the lowest chi2 is kept. Result: values, standard errors, at-bound flags,
    chi2, BIC = chi2 + k ln n, AIC, per-component smeared curves, normalised residuals.
  - `residual_runs`: longest run of consecutive normalised residuals of one sign beyond a threshold and its q span
    (a visibility test: a structure is "seen" if a base model leaves such a run over a factor >= 2 in q).
  - `reference_ratio`: model-free ratio of a curve to a reference curve (e.g. a parent composition), after
    subtracting each curve's flat level and normalising in a low-q window; excess runs with uncertainties.
  - contrast and invariant helpers: electron density of a composition, X-ray SLD, invariant of a fitted component,
    two-phase volume fraction from the invariant.

Units: q in 1/Å, I in 1/cm, lengths in Å unless a name says otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from saxs.core.composite_models import COMPONENTS
from saxs.core.resolution import smear_points

R_E_CM = 2.8179403262e-13      # classical electron radius, CODATA 2018
N_A = 6.02214076e23            # Avogadro constant, CODATA 2018 (exact)


# =============================================================================
# Model specification and fit
# =============================================================================

@dataclass
class TermSpec:
    component: str                                   # key of composite_models.COMPONENTS
    prefix: str                                      # parameter prefix, e.g. "grain_"
    values: Dict[str, float] = field(default_factory=dict)          # starting values (unprefixed names)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed: Dict[str, float] = field(default_factory=dict)           # parameters held fixed at these values
    free: Sequence[str] = ()                                         # parameters to free that are fixed by default


@dataclass
class ParamResult:
    value: float
    stderr: Optional[float]
    vary: bool
    lo: float
    hi: float
    at_bound: bool


@dataclass
class PhysFitResult:
    name: str
    terms: List[TermSpec]
    params: Dict[str, ParamResult]
    n_points: int
    n_free: int
    chi2: float
    bic: float
    aic: float
    success: bool
    message: str
    covariance_ok: bool
    q: np.ndarray
    I: np.ndarray
    model: np.ndarray
    components: Dict[str, np.ndarray]
    residual: np.ndarray                              # normalised residual r_i
    floor: float
    smeared: bool
    n_excluded: int
    excluded_reason: str

    @property
    def redchi(self) -> float:
        return self.chi2 / max(self.n_points - self.n_free, 1)

    @property
    def rms_log(self) -> float:
        return float(np.sqrt(np.mean((np.log10(self.model) - np.log10(self.I)) ** 2)))

    def any_at_bound(self) -> List[str]:
        return [k for k, p in self.params.items() if p.vary and p.at_bound]

    def value(self, key: str) -> float:
        return self.params[key].value


def _is_at_bound(v: float, lo: float, hi: float, rtol: float = 0.01) -> bool:
    """True if v sits at (within rtol of the range from) a bound. A lower bound of exactly 0 (amplitudes, volume
    fractions) is only "reached" at v <= 1e-4 x the upper bound (or v <= 0 when unbounded above): a small but non-zero
    amplitude is a result, not a bound."""
    if not np.isfinite(v):
        return True
    if lo == 0:
        if v <= (1e-4 * hi if np.isfinite(hi) else 0.0):
            return True
        if np.isfinite(hi) and (hi - v) < rtol * hi:
            return True
        return False
    span_ok = np.isfinite(lo) and np.isfinite(hi) and hi > lo
    if span_ok and lo > 0 and hi / lo > 50:                          # log-scaled parameter
        lv, ll, lh = math.log(max(v, 1e-300)), math.log(lo), math.log(hi)
        return (lv - ll) < rtol * (lh - ll) or (lh - lv) < rtol * (lh - ll)
    if span_ok:
        return (v - lo) < rtol * (hi - lo) or (hi - v) < rtol * (hi - lo)
    if np.isfinite(lo) and lo != 0 and abs(v - lo) < 1e-9 * max(abs(lo), 1.0):
        return True
    if np.isfinite(lo) and lo == 0 and v <= 0:
        return True
    return False


def build_parameters(terms: Sequence[TermSpec]):
    import lmfit
    params = lmfit.Parameters()
    for t in terms:
        comp = COMPONENTS[t.component]()
        for p in comp.params():
            name = t.prefix + p.name
            lo, hi = t.bounds.get(p.name, (p.min, p.max))
            if p.name in t.fixed:
                params.add(name, value=float(t.fixed[p.name]), vary=False, min=-np.inf, max=np.inf)
                continue
            vary = p.vary or (p.name in t.free)
            v = float(t.values.get(p.name, p.value))
            v = min(max(v, lo), hi)
            params.add(name, value=v, min=lo, max=hi, vary=vary)
    return params


def model_components(terms: Sequence[TermSpec], values: Dict[str, float], q: np.ndarray,
                     dq: Optional[np.ndarray] = None, n_nodes: int = 20) -> Dict[str, np.ndarray]:
    """Per-term curves at q (resolution-smeared when dq is given), keyed by prefix without the underscore."""
    out = {}
    for t in terms:
        comp = COMPONENTS[t.component]()
        kw = {p.name: float(values[t.prefix + p.name]) for p in comp.params()}
        f = lambda qq, comp=comp, kw=kw: comp.eval(qq, **kw)
        out[t.prefix.rstrip("_")] = smear_points(f, q, dq, n_nodes) if dq is not None else f(q)
    return out


def fit_physical(name: str, q: np.ndarray, I: np.ndarray, sigma: np.ndarray, terms: Sequence[TermSpec],
                 dq: Optional[np.ndarray] = None, floor: float = 0.03, q_range: Optional[Tuple[float, float]] = None,
                 starts: Optional[Sequence[Dict[str, float]]] = None, n_nodes: int = 20,
                 max_nfev: int = 4000, prescreen: Optional[int] = None) -> PhysFitResult:
    """Fit ln I with weights 1/sqrt((sigma/I)^2 + floor^2). `starts`: extra starting points as {prefixed name: value}
    applied on top of the TermSpec values; the solution with the lowest chi2 is returned. `prescreen=N`: evaluate the
    objective at every starting point first and optimise only the N best (plus the TermSpec default). Points with
    I <= 0, non-finite values, or outside q_range are excluded and counted (n_excluded, excluded_reason)."""
    import lmfit
    q = np.asarray(q, float); I = np.asarray(I, float); sigma = np.asarray(sigma, float)
    dqa = None if dq is None else np.asarray(dq, float)
    ok = np.isfinite(q) & np.isfinite(I) & np.isfinite(sigma) & (I > 0) & (sigma > 0)
    reasons = []
    if (~ok).sum():
        reasons.append(f"{int((~ok).sum())} non-positive or non-finite")
    if q_range is not None:
        inr = (q >= q_range[0]) & (q <= q_range[1])
        if (ok & ~inr).sum():
            reasons.append(f"{int((ok & ~inr).sum())} outside q {q_range[0]:g}-{q_range[1]:g}")
        ok &= inr
    qf, If, sf = q[ok], I[ok], sigma[ok]
    dqf = None if dqa is None else dqa[ok]
    w = 1.0 / np.sqrt((sf / If) ** 2 + floor ** 2)
    lnI = np.log(If)

    def total(p):
        vals = p.valuesdict()
        comps = model_components(terms, vals, qf, dqf, n_nodes)
        return np.sum(list(comps.values()), axis=0)

    def resid(p):
        m = total(p)
        return (np.log(np.clip(m, 1e-300, None)) - lnI) * w

    def start_params(st):
        params = build_parameters(terms)
        if st:
            for k, v in st.items():
                if k in params and params[k].vary:
                    params[k].set(value=min(max(float(v), params[k].min), params[k].max))
        return params

    start_list = list(starts or [])
    if prescreen is not None and len(start_list) > prescreen:
        costs = []
        for st in start_list:
            with np.errstate(all="ignore"):
                rr = resid(start_params(st))
            costs.append(float(np.sum(rr ** 2)) if np.all(np.isfinite(rr)) else np.inf)
        order = np.argsort(costs)[:prescreen]
        start_list = [start_list[i] for i in order]

    best = None
    for st in [None] + start_list:
        params = build_parameters(terms)
        if st:
            for k, v in st.items():
                if k in params and params[k].vary:
                    params[k].set(value=min(max(float(v), params[k].min), params[k].max))
        try:
            r = lmfit.Minimizer(resid, params).minimize(method="least_squares", max_nfev=max_nfev)
        except Exception as exc:                                            # pragma: no cover - reported, not hidden
            if best is None:
                best = ("failed", str(exc), params)
            continue
        chi2 = float(np.sum(r.residual ** 2))
        if best is None or isinstance(best, tuple) or chi2 < best[0]:
            best = (chi2, r)
    if isinstance(best, tuple) and best[0] == "failed":
        raise RuntimeError(f"{name}: every start failed ({best[1]})")
    chi2, r = best
    vals = r.params.valuesdict()
    pres = {}
    for k, p in r.params.items():
        pres[k] = ParamResult(float(p.value), None if p.stderr is None else float(p.stderr), bool(p.vary),
                              float(p.min), float(p.max), bool(p.vary and _is_at_bound(p.value, p.min, p.max)))
    k_free = int(sum(1 for p in r.params.values() if p.vary))
    n = int(qf.size)
    comps = model_components(terms, vals, qf, dqf, n_nodes)
    model = np.sum(list(comps.values()), axis=0)
    return PhysFitResult(
        name=name, terms=list(terms), params=pres, n_points=n, n_free=k_free, chi2=chi2,
        bic=chi2 + k_free * math.log(n), aic=chi2 + 2 * k_free, success=bool(r.success), message=str(r.message),
        covariance_ok=bool(r.covar is not None), q=qf, I=If, model=model, components=comps,
        residual=(np.log(model) - lnI) * w, floor=floor, smeared=dqf is not None,
        n_excluded=int((~ok).sum()), excluded_reason="; ".join(reasons) or "none")


# =============================================================================
# Visibility and model-free evidence
# =============================================================================

@dataclass
class Run:
    sign: int
    i0: int
    i1: int
    q_lo: float
    q_hi: float
    span: float              # q_hi / q_lo
    n: int
    max_abs: float


def residual_runs(q: np.ndarray, r: np.ndarray, threshold: float = 1.0) -> List[Run]:
    """All runs of consecutive points with r > threshold (sign +1) or r < -threshold (sign -1), longest span first."""
    q = np.asarray(q, float); r = np.asarray(r, float)
    s = np.where(r > threshold, 1, np.where(r < -threshold, -1, 0))
    runs, i = [], 0
    while i < len(s):
        if s[i] == 0:
            i += 1; continue
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        runs.append(Run(int(s[i]), i, j, float(q[i]), float(q[j]), float(q[j] / q[i]), j - i + 1,
                        float(np.max(np.abs(r[i:j + 1])))))
        i = j + 1
    return sorted(runs, key=lambda u: -u.span)


@dataclass
class RatioResult:
    q: np.ndarray
    ratio: np.ndarray
    ratio_err: np.ndarray
    norm: float
    runs: List[Run]


def reference_ratio(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, q_ref: np.ndarray, I_ref: np.ndarray,
                    sigma_ref: np.ndarray, flat=0.0, flat_ref=0.0,
                    norm_window: Tuple[float, float] = (0.0, np.inf), rel_spread: float = 0.0,
                    threshold: float = 1.0) -> RatioResult:
    """R(q) = (I - flat) / (I_ref - flat_ref) on the sample grid (reference interpolated in log-log inside its own
    range only), divided by its median in `norm_window`. `flat` / `flat_ref` are scalars or arrays on the sample /
    reference grids (e.g. a fitted q-dependent high-q term). Uncertainty: both sigmas in quadrature plus `rel_spread`
    (e.g. a packing spread). Runs of (R - 1)/err beyond `threshold` are returned (longest first)."""
    q = np.asarray(q, float); I = np.asarray(I, float); sigma = np.asarray(sigma, float)
    qr = np.asarray(q_ref, float); Ir = np.asarray(I_ref, float); sr = np.asarray(sigma_ref, float)
    flat = np.broadcast_to(np.asarray(flat, float), q.shape)
    flat_ref = np.broadcast_to(np.asarray(flat_ref, float), qr.shape)
    inside = (q >= qr.min()) & (q <= qr.max())
    q, I, sigma, flat = q[inside], I[inside], sigma[inside], flat[inside]
    flat_ref = np.interp(np.log(q), np.log(qr), flat_ref)
    lr = np.interp(np.log(q), np.log(qr), np.log(np.clip(Ir, 1e-300, None)))
    ref = np.exp(lr)
    ref_err = np.interp(np.log(q), np.log(qr), sr / np.clip(Ir, 1e-300, None)) * ref
    num, den = I - flat, ref - flat_ref
    good = (num > 0) & (den > 0)
    R = np.where(good, num / np.where(good, den, 1), np.nan)
    rel = np.sqrt((sigma / np.where(good, num, 1)) ** 2 + (ref_err / np.where(good, den, 1)) ** 2 + rel_spread ** 2)
    m = good & (q >= norm_window[0]) & (q <= norm_window[1])
    norm = float(np.nanmedian(R[m])) if m.any() else 1.0
    R = R / norm
    err = R * rel
    z = np.where(good, (R - 1) / err, 0.0)
    return RatioResult(q=q, ratio=R, ratio_err=err, norm=norm, runs=residual_runs(q, z, threshold))


# =============================================================================
# Contrast, invariant, volume fraction
# =============================================================================

def electron_density_cm3(composition: Dict[str, float], density_g_cm3: float) -> float:
    """Electrons per cm^3 of a material given as {formula: molar amount} (e.g. oxides in mol %, or a single formula
    with amount 1) and its mass density. Z and atomic masses from xraydb."""
    import xraydb
    z_tot, m_tot = 0.0, 0.0
    for formula, amount in composition.items():
        if amount == 0:
            continue
        for el, n in xraydb.chemparse(formula).items():
            z_tot += amount * n * xraydb.atomic_number(el)
            m_tot += amount * n * xraydb.atomic_mass(el)
    return float(density_g_cm3) * N_A * z_tot / m_tot


def xray_sld_1e10cm2(rho_e_cm3: float) -> float:
    """X-ray scattering-length density r_e rho_e in units of 1e10 cm^-2 (forward scattering, no anomalous terms)."""
    return R_E_CM * rho_e_cm3 / 1e10


def invariant(func: Callable[[np.ndarray], np.ndarray], q_lo: float = 1e-6, q_hi: float = 100.0, n: int = 20000) -> float:
    """Q = integral q^2 I(q) dq (I in cm^-1, q in 1/Å -> Q in cm^-1 Å^-3), on a log grid; the caller must pass a
    component whose integrand converges inside [q_lo, q_hi] (any term decaying faster than q^-3)."""
    from core.np_compat import trapezoid
    x = np.logspace(math.log10(q_lo), math.log10(q_hi), n)
    return float(trapezoid(x ** 3 * func(x), np.log(x)))


def two_phase_product_cm4(Q_cm_invA3: float) -> float:
    """phi (1 - phi) drho^2 in cm^-4 from the invariant: Q = 2 pi^2 phi (1 - phi) drho^2 (Porod)."""
    return float(Q_cm_invA3) * 1e24 / (2 * math.pi ** 2)


def volume_fraction_from_invariant(Q_cm_invA3: float, drho_1e10cm2: float) -> float:
    """Smaller root phi of phi (1 - phi) = X, X = Q / (2 pi^2 drho^2); NaN if X > 1/4 (contrast too small)."""
    X = two_phase_product_cm4(Q_cm_invA3) / (float(drho_1e10cm2) * 1e10) ** 2
    if not np.isfinite(X) or X > 0.25 or X < 0:
        return float("nan")
    return float((1 - math.sqrt(1 - 4 * X)) / 2)
