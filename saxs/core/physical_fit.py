"""The fit engine: one model = an ordered sum of components from saxs.core.models, optionally smeared by the
per-point Gaussian resolution, fitted in log I.

Objective: r_i = (ln M_i - ln I_i) / sqrt((sigma_i / I_i)^2 + floor^2), M = the resolution-smeared model, minimised
by lmfit / scipy least_squares (trust-region reflective, native bounds). Several starting points can be given; with
``prescreen=N`` the objective is evaluated at every start first and only the N best are optimised; the lowest chi2
wins. A result carries values, standard errors, at-bound flags, chi2, BIC = chi2 + k ln n, AIC, the smeared
component curves and the normalised residuals, and round-trips through JSON (``to_dict`` / ``from_dict``).

Model-free evidence used by the workflow (saxs.core.fit_workflow):
  ``residual_runs``    runs of consecutive normalised residuals of one sign beyond a threshold, with their q span;
  ``reference_ratio``  the ratio of a curve to a reference curve after subtracting each one's high-q term,
                       normalised in a low-q window, with its uncertainty and excess runs.

Units: q in 1/Å, I in 1/cm (or any consistent unit), lengths in Å.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saxs.core.models import COMPONENTS
from saxs.core.resolution import smear_points


# ------------------------------------------------------------------------------------------- model specification
@dataclass
class TermSpec:
    component: str                                   # key of saxs.core.models.COMPONENTS
    prefix: str                                      # parameter prefix, e.g. "grain_"
    values: Dict[str, float] = field(default_factory=dict)          # starting values (unprefixed names)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed: Dict[str, float] = field(default_factory=dict)           # parameters held fixed at these values
    free: Sequence[str] = ()                                         # parameters to free that are fixed by default

    @property
    def key(self) -> str:
        return self.prefix.rstrip("_")

    def to_dict(self) -> Dict[str, object]:
        return {"component": self.component, "prefix": self.prefix, "values": dict(self.values),
                "bounds": {k: [float(a), float(b)] for k, (a, b) in self.bounds.items()},
                "fixed": dict(self.fixed), "free": list(self.free)}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TermSpec":
        return cls(component=str(d["component"]), prefix=str(d["prefix"]), values=dict(d.get("values", {})),
                   bounds={k: (float(v[0]), float(v[1])) for k, v in dict(d.get("bounds", {})).items()},
                   fixed=dict(d.get("fixed", {})), free=tuple(d.get("free", ())))


@dataclass
class ParamResult:
    value: float
    stderr: Optional[float]
    vary: bool
    lo: float
    hi: float
    at_bound: bool


def _num(v) -> Optional[float]:
    """JSON-safe float (inf / nan -> string markers handled by json.dumps with allow_nan; keep None)."""
    return None if v is None else float(v)


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

    def values(self) -> Dict[str, float]:
        return {k: p.value for k, p in self.params.items()}

    def eval(self, q: np.ndarray, dq: Optional[np.ndarray] = None, n_nodes: int = 12) -> Dict[str, np.ndarray]:
        """The fitted components (and their sum under "total") on any q grid, smeared when dq is given."""
        comps = model_components(self.terms, self.values(), np.asarray(q, float), dq, n_nodes)
        comps["total"] = np.sum([v for v in comps.values()], axis=0)
        return comps

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "terms": [t.to_dict() for t in self.terms],
            "params": {k: {"value": float(p.value), "stderr": _num(p.stderr), "vary": bool(p.vary), "lo": float(p.lo),
                           "hi": float(p.hi), "at_bound": bool(p.at_bound)} for k, p in self.params.items()},
            "n_points": int(self.n_points), "n_free": int(self.n_free), "chi2": float(self.chi2), "bic": float(self.bic),
            "aic": float(self.aic), "success": bool(self.success), "message": self.message,
            "covariance_ok": bool(self.covariance_ok), "q": self.q.tolist(), "I": self.I.tolist(),
            "model": self.model.tolist(), "components": {k: v.tolist() for k, v in self.components.items()},
            "residual": self.residual.tolist(), "floor": float(self.floor), "smeared": bool(self.smeared),
            "n_excluded": int(self.n_excluded), "excluded_reason": self.excluded_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PhysFitResult":
        params = {k: ParamResult(float(v["value"]), None if v.get("stderr") is None else float(v["stderr"]),
                                 bool(v["vary"]), float(v["lo"]), float(v["hi"]), bool(v["at_bound"]))
                  for k, v in dict(d["params"]).items()}
        arr = lambda key: np.asarray(d[key], float)      # noqa: E731
        return cls(name=str(d["name"]), terms=[TermSpec.from_dict(t) for t in d["terms"]], params=params,
                   n_points=int(d["n_points"]), n_free=int(d["n_free"]), chi2=float(d["chi2"]), bic=float(d["bic"]),
                   aic=float(d["aic"]), success=bool(d["success"]), message=str(d["message"]),
                   covariance_ok=bool(d["covariance_ok"]), q=arr("q"), I=arr("I"), model=arr("model"),
                   components={k: np.asarray(v, float) for k, v in dict(d["components"]).items()},
                   residual=arr("residual"), floor=float(d["floor"]), smeared=bool(d["smeared"]),
                   n_excluded=int(d["n_excluded"]), excluded_reason=str(d["excluded_reason"]))


def is_at_bound(v: float, lo: float, hi: float, rtol: float = 0.01) -> bool:
    """True if v sits at (within rtol of the range from) a bound; log scale for bounds spanning > x50. A lower bound
    of exactly 0 (amplitudes, volume fractions) is only reached at v <= 1e-4 x the upper bound (or v <= 0 when
    unbounded above): a small but non-zero amplitude is a result, not a bound."""
    if not np.isfinite(v):
        return True
    if lo == 0:
        if v <= (1e-4 * hi if np.isfinite(hi) else 0.0):
            return True
        if np.isfinite(hi) and (hi - v) < rtol * hi:
            return True
        return False
    span_ok = np.isfinite(lo) and np.isfinite(hi) and hi > lo
    if span_ok and lo > 0 and hi / lo > 50:
        lv, ll, lh = math.log(max(v, 1e-300)), math.log(lo), math.log(hi)
        return (lv - ll) < rtol * (lh - ll) or (lh - lv) < rtol * (lh - ll)
    if span_ok:
        return (v - lo) < rtol * (hi - lo) or (hi - v) < rtol * (hi - lo)
    if np.isfinite(lo) and abs(v - lo) < 1e-9 * max(abs(lo), 1.0):
        return True
    return False


def build_parameters(terms: Sequence[TermSpec], start: Optional[Dict[str, float]] = None):
    """lmfit Parameters for the terms (component defaults, then TermSpec values / bounds / fixed / free), with an
    optional starting point ``{prefixed name: value}`` clipped into the bounds of the free parameters."""
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
            params.add(name, value=min(max(v, lo), hi), min=lo, max=hi, vary=vary)
    for k, v in (start or {}).items():
        if k in params and params[k].vary:
            params[k].set(value=min(max(float(v), params[k].min), params[k].max))
    return params


def model_components(terms: Sequence[TermSpec], values: Dict[str, float], q: np.ndarray,
                     dq: Optional[np.ndarray] = None, n_nodes: int = 12) -> Dict[str, np.ndarray]:
    """Per-term curves at q (resolution-smeared when dq is given), keyed by prefix without the underscore."""
    out = {}
    for t in terms:
        comp = COMPONENTS[t.component]()
        kw = {p.name: float(values[t.prefix + p.name]) for p in comp.params()}
        f = lambda qq, comp=comp, kw=kw: comp.eval(qq, **kw)      # noqa: E731
        out[t.key] = smear_points(f, q, dq, n_nodes) if dq is not None else f(q)
    return out


def fit_physical(name: str, q: np.ndarray, I: np.ndarray, sigma: np.ndarray, terms: Sequence[TermSpec],
                 dq: Optional[np.ndarray] = None, floor: float = 0.03, q_range: Optional[Tuple[float, float]] = None,
                 starts: Optional[Sequence[Dict[str, float]]] = None, n_nodes: int = 12,
                 max_nfev: int = 4000, prescreen: Optional[int] = None, polish_rounds: int = 3) -> PhysFitResult:
    """Fit ln I with weights 1/sqrt((sigma/I)^2 + floor^2). ``starts``: extra starting points (prefixed names) applied
    on top of the TermSpec values; the TermSpec default start is always tried first. When the best start stops on
    the evaluation budget (lmfit counts the finite-difference Jacobian too), it is continued from its own solution
    with a fresh budget, up to ``polish_rounds`` times, so that ``success`` means converged rather than out of
    budget. Points with I <= 0, sigma <= 0, non-finite values or outside ``q_range`` are excluded and counted
    (``n_excluded``, ``excluded_reason``)."""
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
    if ok.sum() < 4:
        raise ValueError(f"{name}: fewer than 4 usable points")
    qf, If, sf = q[ok], I[ok], sigma[ok]
    dqf = None if dqa is None else dqa[ok]
    w = 1.0 / np.sqrt((sf / If) ** 2 + floor ** 2)
    lnI = np.log(If)

    def resid(p):
        m = np.sum(list(model_components(terms, p.valuesdict(), qf, dqf, n_nodes).values()), axis=0)
        return (np.log(np.clip(m, 1e-300, None)) - lnI) * w

    start_list = list(starts or [])
    if prescreen is not None and len(start_list) > prescreen:
        costs = []
        for st in start_list:
            with np.errstate(all="ignore"):
                rr = resid(build_parameters(terms, st))
            costs.append(float(np.sum(rr ** 2)) if np.all(np.isfinite(rr)) else np.inf)
        start_list = [start_list[i] for i in np.argsort(costs)[:prescreen]]

    best, failure = None, None
    for st in [None] + start_list:
        try:
            r = lmfit.Minimizer(resid, build_parameters(terms, st)).minimize(method="least_squares", max_nfev=max_nfev)
        except Exception as exc:                                            # pragma: no cover - reported, not hidden
            failure = str(exc)
            continue
        chi2 = float(np.sum(r.residual ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, r)
    if best is None:
        raise RuntimeError(f"{name}: every start failed ({failure})")
    chi2, r = best
    for _ in range(max(int(polish_rounds), 0)):
        if r.success or not (getattr(r, "aborted", False) or "function evaluations" in str(r.message)):
            break
        start = {k: p.value for k, p in r.params.items() if p.vary}
        try:
            r2 = lmfit.Minimizer(resid, build_parameters(terms, start)).minimize(method="least_squares", max_nfev=max_nfev)
        except Exception:                                                   # pragma: no cover
            break
        c2 = float(np.sum(r2.residual ** 2))
        if c2 > chi2 * (1 + 1e-9):
            break
        chi2, r = c2, r2
    vals = r.params.valuesdict()
    pres = {k: ParamResult(float(p.value), None if p.stderr is None else float(p.stderr), bool(p.vary),
                           float(p.min), float(p.max), bool(p.vary and is_at_bound(p.value, p.min, p.max)))
            for k, p in r.params.items()}
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


# ------------------------------------------------------------------------------------------- model-free evidence
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

    def to_dict(self) -> Dict[str, object]:
        return {"sign": self.sign, "i0": self.i0, "i1": self.i1, "q_lo": self.q_lo, "q_hi": self.q_hi,
                "span": self.span, "n": self.n, "max_abs": self.max_abs}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, object]]) -> Optional["Run"]:
        return None if d is None else cls(**{k: d[k] for k in ("sign", "i0", "i1", "q_lo", "q_hi", "span", "n", "max_abs")})


def residual_runs(q: np.ndarray, r: np.ndarray, threshold: float = 1.0) -> List[Run]:
    """All runs of consecutive points with r > threshold (sign +1) or r < -threshold (sign -1), longest span first."""
    q = np.asarray(q, float); r = np.asarray(r, float)
    s = np.where(r > threshold, 1, np.where(r < -threshold, -1, 0))
    runs, i = [], 0
    while i < len(s):
        if s[i] == 0:
            i += 1
            continue
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
    range only), divided by its median in ``norm_window``. ``flat`` / ``flat_ref`` are scalars or arrays on the sample /
    reference grids (e.g. a fitted q-dependent high-q term). Uncertainty: both sigmas in quadrature plus ``rel_spread``.
    Runs of (R - 1)/err beyond ``threshold`` are returned (longest first)."""
    q = np.asarray(q, float); I = np.asarray(I, float); sigma = np.asarray(sigma, float)
    qr = np.asarray(q_ref, float); Ir = np.asarray(I_ref, float); sr = np.asarray(sigma_ref, float)
    flat = np.broadcast_to(np.asarray(flat, float), q.shape)
    flat_ref = np.broadcast_to(np.asarray(flat_ref, float), qr.shape)
    inside = (q >= qr.min()) & (q <= qr.max())
    q, I, sigma, flat = q[inside], I[inside], sigma[inside], flat[inside]
    flat_ref = np.interp(np.log(q), np.log(qr), flat_ref)
    ref = np.exp(np.interp(np.log(q), np.log(qr), np.log(np.clip(Ir, 1e-300, None))))
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
