"""
dta_kinetics.py — framework-agnostic rate-dependence kinetics for DTA/DSC:
Lasocka (Tg vs ln q), Moynihan / Kissinger / Flynn-Wall-Ozawa / Augis-Bennett
activation energies, with uncertainties that stay honest at n = 2-5.

Pure numpy + stdlib. No Qt, no pandas, no dta_program, no provenance: a
RatePoint is (rate, temperature) and nothing else, which is what lets the
known-Ea round trip be tested with no data file at all. Everything about
WHERE a point came from — which file, which segment, which specimen, how
many prior heats it had seen — lives one layer up in dta_rate_series.py,
and so do the provenance warnings (mixed_window, thermal_history,
single_specimen_cycles, model_quantity_mismatch, ...). This module emits
only the warnings a fit can see for itself.

Design contract (the user's standing instruction, "still make this possible
to process such data"): NOTHING here refuses to compute for a physical
reason. A negative activation energy, a negative Lasocka B, a positive
Kissinger slope, two points, a rate span of 1.2x — every one of those comes
back as the number PLUS a SeriesWarning, with ok=True. The only conditions
that stop arithmetic are arithmetic (fewer than two usable points, zero
x-spread, a logarithm of a non-positive number), and even those are
REPORTED on the result (ok=False, message=...) rather than raised. The one
deliberate exception is ols_line(), a low-level numeric primitive that does
raise ValueError — every public fit catches it and turns it into a result.

Unit law, in two halves, both pinned by tests:
  * A heating rate in °C/min and in K/min is the SAME NUMBER. A rate is a
    temperature INTERVAL per unit time, so the kelvin offset cancels in the
    subtraction. Rates are therefore NEVER converted anywhere in this module
    — the same rule dta_science's temperature-shift helpers already follow
    for spreads. Only the TIME base matters: q in K/s = q in K/min / 60,
    which moves every log-intercept by ln 60 and leaves every slope alone.
  * Temperatures ARE converted, because 1/T and ln T are not offset
    invariant and an Ea computed from °C in a 1/T is wrong by a factor of
    roughly three. The conversion happens in exactly ONE function,
    to_kelvin(), and test_kelvin_offset_appears_twice_in_the_module reads
    this source file to prove there is no second one.
Lasocka's B is offset-invariant: fitted in °C and in K it agrees to machine
precision (bit-identical only when the points are exactly collinear — adding
K0 to every y does not shift ybar by exactly K0 in floating point, so with
real scatter the last bit of y - ybar differs and B differs by ~1e-15
relative). Lasocka's A is not offset-invariant, so both A_C and A_K are
reported side by side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 0) Constants
# =============================================================================

R_GAS = 8.314462618       # J mol^-1 K^-1 (CODATA, exact) — same literal as
                          # figures_science's Arrhenius model, deliberately.
K0 = 273.15               # °C -> K offset; see to_kelvin() for the unit law.
DOYLE_C = 1.052           # Doyle's correction to the Ozawa slope (FWO form)

QUANTITY_KINDS = ("Tg", "Tp")

#: Severity vocabulary, worst first. There is deliberately NO level that
#: reads as "refused" — this module never refuses, so a word like "error"
#: would be a lie the UI could act on.
SEVERITIES = ("critical", "warn", "info")

#: Sanity envelope for a characteristic temperature in °C. The low edge is
#: absolute zero expressed with K0 rather than a second copy of the literal,
#: which is what keeps the kelvin offset to a single occurrence in the code.
PLAUSIBLE_T_C = (-K0, 3000.0)

#: Sanity envelope for |q| in °C/min (== K/min). Used by the layer above for
#: the `unit_suspect` check; exported here so there is one definition.
PLAUSIBLE_Q = (1e-3, 200.0)

#: Default reference rate at which every fit reports a back-predicted
#: temperature, so two series measured over different rate ranges can still
#: be compared at one agreed rate.
DEFAULT_Q_REF = 10.0

#: R² below this raises `low_r2`. Only meaningful at dof >= 1; the layer
#: above may pass its own value through the fit's `r2_min` keyword.
R2_MIN_DEFAULT = 0.90

LN10 = 2.302585092994046   # for the fragility index m = Ea/(ln10·R·Tg)

TG_EA_CAVEAT = (
    "Ea(Tg) is the apparent activation energy of structural relaxation at Tg "
    "(Moynihan form: d ln q / d(1/Tg) = -Ea/R). The glass transition is a "
    "kinetic freezing of relaxation, not a nucleated transformation, so this "
    "is not an activation energy in the Kissinger sense: the underlying "
    "relaxation is non-Arrhenius (VFT) and Ea(Tg) is only the local Arrhenius "
    "tangent at Tg. Use it to compare glasses measured the same way, and to "
    "get the fragility m = Ea/(2.303·R·Tg). Do not extrapolate it to other "
    "temperatures and do not compare it with a crystallization Ea."
)

LASOCKA_B_NOTE = (
    "Lasocka's B = dTg/d(ln q) is the rate sensitivity of the glass "
    "transition, in K per e-fold of heating rate. B must be > 0: a faster "
    "ramp freezes the structure at a higher temperature, so Tg can only rise "
    "with q. For an Arrhenius-like relaxation near Tg, d ln q / d(1/Tg) = "
    "-Ea_g/R gives B ≈ R·Tg²/Ea_g, which is why B is reported together with "
    "the Ea_g it implies — the two are the same measurement seen twice. B is "
    "invariant to the temperature unit (the offset cancels in a slope); A is "
    "not, and it also moves with the time base of q, so A is reported in both "
    "°C and K and the reference rate is always stated."
)


def to_kelvin(t_C):
    """Celsius -> Kelvin, the ONLY temperature conversion in this module.

    Together with K0's definition this is one of exactly two lines here that
    mention 273.15, which test_kelvin_offset_appears_twice_in_the_module pins
    by reading the source. The rule exists because a second conversion
    creeping in elsewhere (a `value_unit` field, a display-unit toggle
    reaching the fit, a to_kelvin call inside a rate resolver) produces an
    offset that still looks like a plausible Tg while making Ea wrong by
    roughly a factor of three — a failure no plot inspection catches.

    Accepts a scalar or an array and returns the same shape. Heating rates
    must never be passed through here: see the module docstring's unit law.
    """
    if isinstance(t_C, np.ndarray):
        return t_C.astype(float) + K0
    if isinstance(t_C, (list, tuple)):
        return np.asarray(t_C, dtype=float) + K0
    return float(t_C) + K0


# =============================================================================
# 1) Warnings — the only channel by which this module says "no"
# =============================================================================

@dataclass(frozen=True)
class SeriesWarning:
    """One objection to a fit, in four renderings that cannot be separated.

    `code` is a stable lower_snake key and is the ONLY thing tests and UI
    logic ever branch on; prose is free to improve without breaking anything.
    The same object renders as a banner line, as a short plot annotation, and
    as two CSV columns, which is what makes a warning impossible to lose
    between the screen and the exported file.

    row_uids holds RateRow uids, never table indices: a table can be sorted
    or filtered between the fit and the render, and an index would then point
    at the wrong row. An empty tuple means the warning is about the whole
    series rather than particular rows.
    """

    code: str
    severity: str
    message: str
    row_uids: Tuple[str, ...] = ()
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def glyph(self) -> str:
        """'⚠' for anything actionable, 'ℹ' for information only."""
        return "ℹ" if self.severity == "info" else "⚠"

    def _bare(self) -> str:
        """The message without a leading glyph.

        Defensive: callers one layer up compose messages from a table that
        already shows the glyph, and '⚠ ⚠ ...' in a plot corner looks like a
        bug in the tool rather than a warning about the data.
        """
        text = self.message.strip()
        for g in ("⚠", "ℹ", "!"):
            if text.startswith(g):
                text = text[len(g):].lstrip()
        return text

    def line(self) -> str:
        """Banner / CSV form: glyph + the full sentence."""
        return f"{self.glyph} {self._bare()}"

    def short(self, limit: int = 60) -> str:
        """<= `limit` characters, for a crowded plot annotation.

        Truncates on a word boundary when one is available in the second half
        of the budget, so the fragment still reads as a sentence opening
        rather than a cut-off word.
        """
        text = self._bare()
        if len(text) <= limit:
            return text
        cut = text[: max(1, limit - 1)]
        space = cut.rfind(" ")
        if space >= limit // 2:
            cut = cut[:space]
        return cut.rstrip(" ,.;:—-") + "…"


def max_severity(warnings: Sequence[SeriesWarning]) -> str:
    """Worst severity present, '' when there is none.

    Drives the banner's objectName in the Qt layer and a CSV column, so a
    reviewer can filter a batch export on `critical` in one click.
    """
    for sev in SEVERITIES:
        for w in warnings:
            if w.severity == sev:
                return sev
    return ""


def sort_warnings(warnings: Sequence[SeriesWarning]) -> List[SeriesWarning]:
    """critical -> warn -> info, then alphabetically by code.

    Every rendering of a warning list is length-limited somewhere (a plot
    corner, a status line, a table cell). Sorting by severity here means the
    truncation always keeps the worst problem rather than the first-coded one.
    """
    order = {sev: i for i, sev in enumerate(SEVERITIES)}
    return sorted(warnings, key=lambda w: (order.get(w.severity, len(SEVERITIES)), w.code))


def warning_codes(warnings: Sequence[SeriesWarning],
                  row_uid: Optional[str] = None) -> List[str]:
    """Sorted unique codes; with `row_uid`, only those implicating that row.

    Series-wide warnings (empty row_uids) are ALWAYS included when a row is
    named. A single-row extract from an export must never look innocent
    because the series-level objection lived on a different line.
    """
    codes = set()
    for w in warnings:
        if row_uid is None or not w.row_uids or row_uid in w.row_uids:
            codes.add(w.code)
    return sorted(codes)


def _split_by_severity(warnings: Sequence[SeriesWarning]
                       ) -> Tuple[List[SeriesWarning], List[SeriesWarning]]:
    """(critical/warn, info) — the rule behind KineticFitResult.warnings/notes.

    One classification rule, applied in one place, so "is dof_one a warning
    or a note?" has a mechanical answer instead of a per-call-site opinion.
    """
    warns = [w for w in warnings if w.severity != "info"]
    notes = [w for w in warnings if w.severity == "info"]
    return sort_warnings(warns), sort_warnings(notes)


# =============================================================================
# 2) Input
# =============================================================================

@dataclass
class RatePoint:
    """The minimal fit input: one temperature measured at one heating rate.

    Deliberately provenance-free. `uid` is a back-reference to the RateRow it
    came from when it came from one, and `label` is display text — neither is
    interpreted here. Keeping this dataclass down to (rate, temperature) is
    what lets the headline known-Ea round-trip test run with no data file,
    no fixture and no import from the rest of PRISM.
    """

    q_C_per_min: float
    T_C: float
    quantity_kind: str = "Tg"        # "Tg" | "Tp"
    label: str = ""                  # legend / residual-table text
    uid: str = ""                    # the RateRow this came from, if any
    T0_C: float = float("nan")       # ramp start; Augis-Bennett only

    @property
    def T_K(self) -> float:
        """The temperature in kelvin — the only form any 1/T or ln T uses."""
        return to_kelvin(float(self.T_C))

    @property
    def q(self) -> float:
        """|heating rate|, in K/min == °C/min (see the module unit law).

        The absolute value is taken because a cooling segment carries a
        negative rate and its magnitude is still a perfectly good lever arm;
        the fact that it was a cooling run is a provenance warning
        (`cooling_row`, emitted one layer up), not a reason to drop it.
        """
        return abs(float(self.q_C_per_min))


# =============================================================================
# 3) Honest least squares
# =============================================================================

@dataclass
class LineFit:
    """An unweighted straight-line fit with textbook standard errors.

    Every uncertainty field is nan and r2 is None when dof == 0, because at
    n = 2 the line is exact by construction: SSE is 0, s² = 0/0 is UNDEFINED
    and not zero, and an R² of 1 describes the arithmetic rather than the
    data.
    """

    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    cov_slope_intercept: float
    n: int
    dof: int
    r2: Optional[float]
    s_resid: float
    residuals: np.ndarray


def ols_line(x, y) -> LineFit:
    """Unweighted OLS y = m·x + b, hand-rolled rather than delegated.

    Three measured reasons, all re-verified on this machine with numpy 2.4.6
    and scipy 1.15.1 and pinned by tests:
      1. np.polyfit(x, y, 1, cov=True) RAISES at n = 2 ("the number of data
         points must exceed order to scale the covariance matrix"). n = 2 is
         a case this feature must REPORT, not crash on.
      2. scipy.stats.linregress at n = 2 returns stderr = 0.0, r² = 1.0 and
         intercept_stderr = 0.0 — a fake zero uncertainty sitting directly
         under an activation energy. Handing a user ±0 kJ/mol is the worst
         possible failure mode for this module, so it is structurally
         impossible here: dof == 0 produces nan and None, never a number.
      3. numpy's covariance scaling has changed historically (SSE/(n-order-2)
         -> SSE/(n-order-1), numpy gh-11196/11197), which would make the
         reported SE depend on which numpy the user happens to have. cov(m,b),
         the residual vector, s and dof are all needed here anyway.
    At n >= 3 this agrees with both libraries to machine precision (measured
    max abs difference 5.3e-16 across slope, intercept, both SEs and R²).

        Sxx = Σ(x-x̄)²            Sxy = Σ(x-x̄)(y-ȳ)
        m   = Sxy/Sxx            b = ȳ - m·x̄
        s²  = SSE/(n-2)          SE(m) = √(s²/Sxx)
        SE(b) = √(s²·(1/n + x̄²/Sxx))      cov(m,b) = -s²·x̄/Sxx

    Non-finite (x, y) pairs are dropped, so `residuals` is aligned with the
    FINITE input pairs; every caller in this module pre-filters, which keeps
    residuals aligned one-to-one with the fitted points.

    Raises ValueError on fewer than two usable points or zero x-spread — the
    two algebraic impossibilities. This is the single raising function in the
    module; every public fit catches it and returns ok=False instead.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length; got {x.size} and {y.size}.")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n < 2:
        raise ValueError(
            f"A slope needs at least 2 points with finite x and y; got {n}. "
            f"Add a measurement at another heating rate.")
    xbar = float(x.mean())
    ybar = float(y.mean())
    Sxx = float(((x - xbar) ** 2).sum())
    if Sxx <= 0.0:
        raise ValueError(
            "All x values are identical, so the slope is undefined. Add a point "
            "at a different heating rate.")
    Sxy = float(((x - xbar) * (y - ybar)).sum())
    Syy = float(((y - ybar) ** 2).sum())
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    residuals = y - (slope * x + intercept)
    sse = float((residuals ** 2).sum())
    dof = n - 2
    if dof > 0:
        s2 = sse / dof
        slope_se = float(np.sqrt(s2 / Sxx))
        intercept_se = float(np.sqrt(s2 * (1.0 / n + xbar ** 2 / Sxx)))
        cov_mb = float(-s2 * xbar / Sxx)
        r2: Optional[float] = float(1.0 - sse / Syy) if Syy > 0.0 else None
        s_resid = float(np.sqrt(s2))
    else:
        # n == 2: exact by construction. See the class docstring.
        slope_se = intercept_se = cov_mb = float("nan")
        r2 = None
        s_resid = float("nan")
    return LineFit(slope=slope, intercept=intercept, slope_se=slope_se,
                   intercept_se=intercept_se, cov_slope_intercept=cov_mb,
                   n=n, dof=dof, r2=r2, s_resid=s_resid, residuals=residuals)


#: Two-sided 95 % Student-t quantiles for dof 1..30. Used only when scipy is
#: unimportable (a frozen build that dropped it); verified against
#: scipy.stats.t.ppf(0.975, dof) to a maximum absolute error of 4.71e-4.
_T95_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

_Z975 = 1.9599639845400545     # standard-normal 97.5 % quantile


def _t95_expansion(dof: float) -> float:
    """Cornish-Fisher expansion of the t quantile, for dof > 30 only.

    Measured against scipy over dof 31..2000: maximum absolute error 2.63e-8,
    which is far below any uncertainty a five-point DTA series can carry. It
    exists so that losing scipy degrades the confidence interval's last
    digits rather than removing the interval altogether.
    """
    v = float(dof)
    z = _Z975
    return (z
            + (z ** 3 + z) / (4.0 * v)
            + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * v ** 2)
            + (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z) / (384.0 * v ** 3)
            + (79.0 * z ** 9 + 776.0 * z ** 7 + 1482.0 * z ** 5
               - 1920.0 * z ** 3 - 945.0 * z) / (92160.0 * v ** 4))


def t_crit_95(dof: int) -> Optional[float]:
    """Two-sided 95 % Student-t quantile, or None when there is no such thing.

    scipy is imported lazily inside the function, per PRISM convention: this
    module is otherwise pure numpy and must stay importable (and fast to
    import) in a frozen build.

    Returns None for dof <= 0 rather than a number. That None is load-bearing
    — it is what forces every caller to print "no uncertainty" instead of
    quietly formatting a confidence interval that does not exist.
    """
    try:
        dof = int(dof)
    except (TypeError, ValueError):
        return None
    if dof < 1:
        return None
    try:
        from scipy.stats import t as _t      # noqa: PLC0415 — lazy by convention
        value = float(_t.ppf(0.975, dof))
        if np.isfinite(value):
            return value
    except Exception:
        pass
    if dof in _T95_TABLE:
        return float(_T95_TABLE[dof])
    return float(_t95_expansion(dof))


def _pairwise_slope_range(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """min/max two-point slope over every pair, or None below n = 3.

    At n = 3-5 this says more than R² does: it answers "how far does the
    answer move if I believe one pair of points instead of all of them?",
    which is the question a reviewer of a three-rate Kissinger plot actually
    has. R² at dof = 1 answers nothing.
    """
    n = int(np.size(x))
    if n < 3:
        return None
    slopes = [float((y[j] - y[i]) / (x[j] - x[i]))
              for i in range(n) for j in range(i + 1, n)
              if x[j] != x[i] and np.isfinite(y[j] - y[i])]
    if not slopes:
        return None
    return (min(slopes), max(slopes))


def _safe_exp(v: float) -> float:
    """exp() that returns inf instead of emitting a RuntimeWarning.

    A pathological fit can produce an intercept of several thousand; numpy
    would warn, and the rate-series tests run under filterwarnings('error')
    to prove no numpy warning escapes a fit. Overflow to inf is the honest
    answer and it renders as 'inf', which reads as "look at this fit".
    """
    if not np.isfinite(v):
        return float("nan") if np.isnan(v) else float(v)
    if v > 709.0:
        return float("inf")
    if v < -745.0:
        return 0.0
    return float(np.exp(v))


# =============================================================================
# 4) Result
# =============================================================================

@dataclass
class KineticFitResult:
    """Everything the Qt layer, the CSV writer and the plot need about one fit.

    Carries its own axis labels, equation string and temperature basis so no
    presentation layer ever re-derives them — a second derivation is a second
    place for "ln q" and "ln(q/Tp²)" to drift apart between the axis and the
    arithmetic.

    `ok` is False ONLY when the arithmetic could not be done at all (fewer
    than two usable points, zero x-spread). It is never False because the
    answer is physically wrong: a negative Ea comes back as ok=True with a
    `negative_ea` warning attached, because the user asked to be able to
    process such data and to be told what is wrong with it.
    """

    method: str
    method_label: str
    equation: str
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    n: int
    dof: int
    r2: Optional[float]
    s_resid: float
    x: np.ndarray
    y: np.ndarray
    residuals: np.ndarray
    labels: List[str]
    x_label: str
    y_label: str
    temp_basis: str                                    # "°C" | "K (converted)"
    Ea_kJ_mol: Optional[float] = None
    Ea_se_kJ_mol: Optional[float] = None
    Ea_ci95_kJ_mol: Optional[Tuple[float, float]] = None
    t_crit_95: Optional[float] = None                  # None at dof 0
    pairwise_slope: Optional[Tuple[float, float]] = None
    pairwise_Ea_kJ_mol: Optional[Tuple[float, float]] = None
    lasocka_A_C: Optional[float] = None
    lasocka_A_se: Optional[float] = None
    lasocka_A_K: Optional[float] = None
    lasocka_B: Optional[float] = None
    lasocka_B_se: Optional[float] = None
    q_ref_C_per_min: float = float("nan")
    T_at_q_ref_C: float = float("nan")
    fragility_m: Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)
    rows_used: List[str] = field(default_factory=list)
    n_total: int = 0
    n_included: int = 0
    n_fitted: int = 0
    warnings: List[SeriesWarning] = field(default_factory=list)
    notes: List[SeriesWarning] = field(default_factory=list)
    ok: bool = True
    message: str = ""

    # ---- derived views -----------------------------------------------------

    @property
    def all_warnings(self) -> List[SeriesWarning]:
        """warnings + notes, worst first. The single list every renderer uses."""
        return sort_warnings(list(self.warnings) + list(self.notes))

    @property
    def max_severity(self) -> str:
        """Worst severity across warnings and notes, '' when clean."""
        return max_severity(self.all_warnings)

    @property
    def codes(self) -> List[str]:
        """Sorted unique warning codes — what tests and UI logic branch on."""
        return warning_codes(self.all_warnings)

    def _uncertainty_phrase(self) -> str:
        """The stats line, which must say 'no uncertainty' whenever dof == 0.

        Centralised because that phrase is the user-visible contract for
        n = 2: whatever the method, whatever the units, a two-point fit must
        say so in the same words in the panel, in the plot and in the CSV.
        """
        if self.dof <= 0:
            return (f"n = {self.n}, dof = 0 — no uncertainty: with two points the "
                    f"line is exact, so σ and R² are undefined, not zero")
        r2_txt = "R² n/a" if self.r2 is None else f"R² = {self.r2:.4f}"
        s_txt = "s = n/a" if not np.isfinite(self.s_resid) else f"s = {self.s_resid:.4g}"
        return f"n = {self.n}, dof = {self.dof}, {r2_txt}, {s_txt}"

    def _headline(self) -> List[str]:
        """The two or three numeric lines both renderers share."""
        lines: List[str] = []
        if self.Ea_kJ_mol is not None and np.isfinite(self.Ea_kJ_mol):
            if self.Ea_se_kJ_mol is not None and np.isfinite(self.Ea_se_kJ_mol):
                lines.append(f"Ea = {self.Ea_kJ_mol:.1f} ± {self.Ea_se_kJ_mol:.1f} kJ/mol (1σ)")
                if self.Ea_ci95_kJ_mol is not None:
                    lo, hi = self.Ea_ci95_kJ_mol
                    lines.append(f"95 % CI [{lo:.0f}, {hi:.0f}] kJ/mol (t, dof = {self.dof})")
            else:
                lines.append(f"Ea = {self.Ea_kJ_mol:.1f} kJ/mol")
        if self.lasocka_B is not None and np.isfinite(self.lasocka_B):
            se = ("" if self.lasocka_B_se is None or not np.isfinite(self.lasocka_B_se)
                  else f" ± {self.lasocka_B_se:.4g}")
            lines.append(f"B = {self.lasocka_B:.4g}{se} K per ln(q)")
            if self.lasocka_A_C is not None and np.isfinite(self.lasocka_A_C):
                a_se = ("" if self.lasocka_A_se is None or not np.isfinite(self.lasocka_A_se)
                        else f" ± {self.lasocka_A_se:.4g}")
                lines.append(f"A = {self.lasocka_A_C:.4g}{a_se} °C "
                             f"({self.lasocka_A_K:.4g} K) at q = 1 K/min")
        if not lines:
            se = ("" if not np.isfinite(self.slope_se) else f" ± {self.slope_se:.4g}")
            lines.append(f"slope = {self.slope:.4g}{se}")
        return lines

    def report(self) -> str:
        """Full panel text: the fit, the residual statistics, then EVERY
        warning and note as its own `.line()`.

        Nothing is elided here. `annotation()` is the abbreviated rendering;
        this one is the record, and a user who exports the panel text must
        get every caveat that was attached to the number.
        """
        head = f"{self.method_label}   [{self.equation}]"
        if not self.ok:
            out = [head, f"No fit: {self.message}"]
            for w in self.all_warnings:
                out.append(w.line())
            return "\n".join(out)
        lines = [head]
        lines.extend(self._headline())
        lines.append(self._uncertainty_phrase())
        if self.n_fitted != self.n_total:
            lines.append(f"{self.n_fitted} of {self.n_total} points fitted "
                         f"({self.n_total - self.n_fitted} unusable or without T0)")
        if self.pairwise_Ea_kJ_mol is not None:
            lo, hi = self.pairwise_Ea_kJ_mol
            lines.append(f"pairwise Ea range: {lo:.0f} – {hi:.0f} kJ/mol")
        elif self.pairwise_slope is not None:
            lo, hi = self.pairwise_slope
            lines.append(f"pairwise slope range: {lo:.4g} – {hi:.4g}")
        if np.isfinite(self.T_at_q_ref_C) and np.isfinite(self.q_ref_C_per_min):
            lines.append(f"predicted {self._quantity_name()} at "
                         f"q = {self.q_ref_C_per_min:g} °C/min: {self.T_at_q_ref_C:.1f} °C")
        if self.fragility_m is not None and np.isfinite(self.fragility_m):
            lines.append(f"fragility m = {self.fragility_m:.1f}")
        for key in sorted(self.extras):
            lines.append(f"{key} = {_fmt_number(self.extras[key])}")
        if self.labels:
            lines.append("residuals: " + ", ".join(
                f"{lab}: {res:+.4g}" for lab, res in zip(self.labels, self.residuals)))
        for w in self.all_warnings:
            lines.append(w.line())
        return "\n".join(lines)

    def annotation(self, max_warnings: int = 4) -> str:
        """Plot-corner text: headline numbers, then the worst warnings short.

        Drawn INTO the figure rather than beside it, so a PNG that leaves this
        application still carries its caveats. `max_warnings` exists because
        the corner of an axes is finite; `sort_warnings` guarantees the ones
        kept are the worst ones, and the count of those dropped is stated.
        """
        if not self.ok:
            lines = [self.method_label, f"No fit: {self.message}"]
        else:
            lines = [self.method_label]
            lines.extend(self._headline())
            lines.append(self._uncertainty_phrase())
        shown = self.all_warnings
        extra = 0
        if max_warnings is not None and max_warnings >= 0 and len(shown) > max_warnings:
            extra = len(shown) - max_warnings
            shown = shown[:max_warnings]
        for w in shown:
            lines.append(f"{w.glyph} {w.short()}")
        if extra:
            lines.append(f"… {extra} more warning(s) — see the panel")
        return "\n".join(lines)

    def _quantity_name(self) -> str:
        """'Tg' or 'Tp' — whichever this method's x axis is built from."""
        meta = _METHOD_META.get(self.method, {})
        needs = meta.get("needs", ("Tg",))
        return needs[0] if needs else "T"

    # ---- CSV -------------------------------------------------------------

    def csv_fit_row(self) -> Dict[str, Any]:
        """The fit-level CSV record this module owns.

        Model-irrelevant fields are the empty string, never 0 and never the
        text 'nan' — a blank cell reads as "not applicable to this model",
        whereas a 0 reads as a measurement. dta_rate_series prepends the
        series-level columns (series_id, specimen_keys, ...) to this dict.
        """
        allw = self.all_warnings
        return {
            "method": self.method,
            "method_label": self.method_label,
            "equation": self.equation,
            "temp_basis": self.temp_basis,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "n_rows_total": self.n_total,
            "n_rows_included": self.n_included,
            "n_rows_fitted": self.n_fitted,
            "rows_used": ";".join(self.rows_used),
            "slope": _csv_num(self.slope),
            "slope_se": _csv_num(self.slope_se),
            "intercept": _csv_num(self.intercept),
            "intercept_se": _csv_num(self.intercept_se),
            "r_squared": _csv_num(self.r2),
            "s_resid": _csv_num(self.s_resid),
            "dof": self.dof,
            "t_crit_95": _csv_num(self.t_crit_95),
            "Ea_kJ_mol": _csv_num(self.Ea_kJ_mol),
            "Ea_se_kJ_mol": _csv_num(self.Ea_se_kJ_mol),
            "Ea_ci95_lo_kJ_mol": _csv_num(None if self.Ea_ci95_kJ_mol is None
                                          else self.Ea_ci95_kJ_mol[0]),
            "Ea_ci95_hi_kJ_mol": _csv_num(None if self.Ea_ci95_kJ_mol is None
                                          else self.Ea_ci95_kJ_mol[1]),
            "pairwise_Ea_lo_kJ_mol": _csv_num(None if self.pairwise_Ea_kJ_mol is None
                                              else self.pairwise_Ea_kJ_mol[0]),
            "pairwise_Ea_hi_kJ_mol": _csv_num(None if self.pairwise_Ea_kJ_mol is None
                                              else self.pairwise_Ea_kJ_mol[1]),
            "pairwise_slope_lo": _csv_num(None if self.pairwise_slope is None
                                          else self.pairwise_slope[0]),
            "pairwise_slope_hi": _csv_num(None if self.pairwise_slope is None
                                          else self.pairwise_slope[1]),
            "lasocka_A_C": _csv_num(self.lasocka_A_C),
            "lasocka_A_K": _csv_num(self.lasocka_A_K),
            "lasocka_A_se": _csv_num(self.lasocka_A_se),
            "lasocka_B": _csv_num(self.lasocka_B),
            "lasocka_B_se": _csv_num(self.lasocka_B_se),
            "fragility_m": _csv_num(self.fragility_m),
            "q_ref_C_per_min": _csv_num(self.q_ref_C_per_min),
            "T_at_q_ref_C": _csv_num(self.T_at_q_ref_C),
            "residuals": ";".join(f"{r:.6g}" for r in self.residuals),
            "extras": ";".join(f"{k}={_fmt_number(self.extras[k])}"
                               for k in sorted(self.extras)),
            "fit_ok": self.ok,
            "fit_message": self.message,
            "max_severity": self.max_severity,
            "n_warnings": len(allw),
            "warning_codes": ";".join(warning_codes(allw)),
            "warning_messages": " | ".join(w.line() for w in allw),
        }

    def csv_point_rows(self) -> List[Dict[str, Any]]:
        """One record per FITTED point, with the warnings repeated on each.

        The repetition is deliberate and is the point of the method: a user
        who filters the export down to a single interesting point must still
        see every objection that applied to the fit it came from.
        """
        allw = self.all_warnings
        all_codes = ";".join(warning_codes(allw))
        all_msgs = " | ".join(w.line() for w in allw)
        rows: List[Dict[str, Any]] = []
        for i in range(int(self.n)):
            uid = self.rows_used[i] if i < len(self.rows_used) else ""
            rows.append({
                "point_index": i + 1,
                "uid": uid,
                "label": self.labels[i] if i < len(self.labels) else "",
                "method": self.method,
                "method_label": self.method_label,
                "equation": self.equation,
                "x_label": self.x_label,
                "y_label": self.y_label,
                "x": float(self.x[i]),
                "y": float(self.y[i]),
                "residual": float(self.residuals[i]),
                "slope": _csv_num(self.slope),
                "intercept": _csv_num(self.intercept),
                "Ea_kJ_mol": _csv_num(self.Ea_kJ_mol),
                "max_severity": self.max_severity,
                "warning_codes": ";".join(warning_codes(allw, uid)) if uid else all_codes,
                "warning_messages": all_msgs,
            })
        return rows


def _fmt_number(v: Any) -> str:
    """Compact human formatting used in panel text and the `extras` column."""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "nan" if np.isnan(f) else ("inf" if f > 0 else "-inf")
    return f"{f:.6g}"


def _csv_num(v: Any) -> Any:
    """CSV cell for a possibly-absent number: '' for None/nan, else the float.

    Absent must be blank rather than 0 or the string 'nan'. A 0 in an Ea
    column is a claim; a blank is the absence of one.
    """
    if v is None:
        return ""
    if isinstance(v, bool):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(f):
        return ""
    return f


# =============================================================================
# 5) Shared fit machinery
# =============================================================================

#: Static metadata for every method. Separated from KINETIC_METHODS (which
#: adds the callables at the bottom of the file) so that a failed fit can
#: still report the right labels and equation without the registry having
#: been built yet, and so labels exist in exactly one place.
_METHOD_META: Dict[str, Dict[str, Any]] = {
    "lasocka": {
        "label": "Lasocka  Tg = A + B·ln q",
        "equation": "Tg = A + B·ln q",
        "needs": ("Tg",),
        "needs_T0": False,
        "default_for": ("Tg",),
        "x_label": "ln q   [q in K/min]",
        "y_label": "Tg   [°C]",
        "temp_basis": "°C",
    },
    "moynihan": {
        "label": "Moynihan  ln q = C − Ea/(R·Tg)",
        "equation": "ln q = C − Ea/(R·Tg)",
        "needs": ("Tg",),
        "needs_T0": False,
        "default_for": (),
        "x_label": "1/Tg   [K⁻¹]",
        "y_label": "ln q   [q in K/min]",
        "temp_basis": "K (converted)",
    },
    "kissinger_tg": {
        "label": "Kissinger form on Tg  ln(q/Tg²) vs 1/Tg",
        "equation": "ln(q/Tg²) = −Ea/(R·Tg) + C",
        "needs": ("Tg",),
        "needs_T0": False,
        "default_for": (),
        "x_label": "1/Tg   [K⁻¹]",
        "y_label": "ln(q/Tg²)   [q in K/min]",
        "temp_basis": "K (converted)",
    },
    "kissinger": {
        "label": "Kissinger  ln(q/Tp²) vs 1/Tp",
        "equation": "ln(q/Tp²) = −Ea/(R·Tp) + ln(A·R/Ea)",
        "needs": ("Tp",),
        "needs_T0": False,
        "default_for": ("Tp",),
        "x_label": "1/Tp   [K⁻¹]",
        "y_label": "ln(q/Tp²)   [q in K/min]",
        "temp_basis": "K (converted)",
    },
    "fwo": {
        "label": "Flynn-Wall-Ozawa  ln q vs 1/Tp",
        "equation": f"ln q = C − {DOYLE_C:g}·Ea/(R·Tp)",
        "needs": ("Tp",),
        "needs_T0": False,
        "default_for": (),
        "x_label": "1/Tp   [K⁻¹]",
        "y_label": "ln q   [q in K/min]",
        "temp_basis": "K (converted)",
    },
    "augis_bennett": {
        "label": "Augis-Bennett  ln(q/(Tp−T0)) vs 1/Tp",
        "equation": "ln(q/(Tp−T0)) = −Ea/(R·Tp) + ln(ν₀)",
        "needs": ("Tp",),
        "needs_T0": True,
        "default_for": (),
        "x_label": "1/Tp   [K⁻¹]",
        "y_label": "ln(q/(Tp−T0))   [min⁻¹]",
        "temp_basis": "K (converted)",
    },
}


def _label_of(p: RatePoint, i: int) -> str:
    """Display text for a point: its own label, else its rate, else its index."""
    if p.label:
        return p.label
    if np.isfinite(p.q_C_per_min):
        return f"{p.q_C_per_min:g} °C/min"
    return f"point {i + 1}"


def _usable_points(points: Sequence[RatePoint]
                   ) -> Tuple[List[RatePoint], List[RatePoint]]:
    """Split into (arithmetically usable, unusable).

    Unusable means the ARITHMETIC cannot be done on that point, never that
    the physics is doubtful: a non-finite rate or temperature, a rate of
    exactly zero (ln 0), or a temperature at or below absolute zero (1/T and
    ln T undefined). Those points are dropped from the fit and counted, so
    n_fitted < n_total makes the drop visible; the provenance layer above
    emits the `no_value` / `no_rate` warnings that name them.
    """
    good: List[RatePoint] = []
    bad: List[RatePoint] = []
    for p in points:
        try:
            q = p.q
            t_K = p.T_K
        except (TypeError, ValueError):
            bad.append(p)
            continue
        if np.isfinite(q) and q > 0.0 and np.isfinite(t_K) and t_K > 0.0:
            good.append(p)
        else:
            bad.append(p)
    return good, bad


def _dropped_point_warning(method_key: str, bad: Sequence[RatePoint],
                           n_total: int) -> List[SeriesWarning]:
    """`dropped_points` — name every point the arithmetic could not take.

    Without it the drop is visible only as `n_fitted < n_total`, a number a
    reader has to notice and then explain to themselves. It matters most
    where the two layers disagree: `RateRow.fittable` accepts any finite
    value, while this module additionally needs T in kelvin above zero, so a
    value at or below absolute zero is fittable to the table and unusable
    here. For Lasocka that gap is real rather than arithmetic — Lasocka fits T_C
    directly and never touches T_K — so the point is a physical objection
    being dropped by an arithmetic rule, and M0 says an objection must travel
    with the result instead of happening quietly.
    """
    if not bad:
        return []
    names = ", ".join(_label_of(p, i) for i, p in enumerate(bad))
    return [SeriesWarning(
        code="dropped_points", severity="warn",
        message=(f"{len(bad)} of {n_total} point(s) were left out of the {method_key} "
                 f"fit because the arithmetic has no answer for them — a rate of "
                 f"zero, or a temperature at or below absolute zero: {names}. The "
                 f"remaining points were fitted anyway."),
        row_uids=tuple(p.uid for p in bad if p.uid),
        detail={"n_dropped": len(bad), "n_total": int(n_total),
                "method": method_key})]


def _solve_T_at_q(residual: Callable[[float], float],
                  lo: float = 20.0, hi: float = 5000.0,
                  samples: int = 2000) -> float:
    """Invert a fitted model for T (in K) at one reference rate.

    A scan for a sign change followed by bisection, rather than algebra,
    because the Kissinger and Augis-Bennett forms are transcendental in T and
    one numeric path for all six models is less code and fewer places to get
    a sign wrong than six inversions. Returns nan when no root is bracketed,
    which is the honest answer for a fit whose slope has the wrong sign.
    """
    grid = np.geomspace(lo, hi, int(samples))
    prev_T = float(grid[0])
    try:
        prev_f = float(residual(prev_T))
    except Exception:
        return float("nan")
    for t in grid[1:]:
        t = float(t)
        try:
            f = float(residual(t))
        except Exception:
            return float("nan")
        if not np.isfinite(f):
            prev_T, prev_f = t, f
            continue
        if f == 0.0:
            return t
        if np.isfinite(prev_f) and (f > 0.0) != (prev_f > 0.0):
            a, fa, b = prev_T, prev_f, t
            for _ in range(200):
                mid = 0.5 * (a + b)
                fm = float(residual(mid))
                if fm == 0.0:
                    return mid
                if (fm > 0.0) == (fa > 0.0):
                    a, fa = mid, fm
                else:
                    b = mid
            return 0.5 * (a + b)
        prev_T, prev_f = t, f
    return float("nan")


def _common_warnings(fit: LineFit, r2_min: float) -> List[SeriesWarning]:
    """The objections any straight-line fit can raise about itself.

    Ordered by the table in the rate-series spec, all non-gating:
    `two_points_only` (warn), `dof_one` (info), `low_r2` (warn).
    """
    out: List[SeriesWarning] = []
    if fit.n == 2:
        out.append(SeriesWarning(
            code="two_points_only", severity="warn",
            message=("Two points: the line is exact by construction, so no "
                     "uncertainty can be estimated (σ and R² are undefined, not "
                     "zero). Fitted anyway."),
            detail={"n": 2}))
    if fit.dof == 1:
        out.append(SeriesWarning(
            code="dof_one", severity="info",
            message=("dof = 1: R² is not evidence here (a line through 3 points "
                     "leaves 1 residual); read the ± and the pairwise range."),
            detail={"dof": 1}))
    if fit.dof >= 1 and fit.r2 is not None and fit.r2 < r2_min:
        out.append(SeriesWarning(
            code="low_r2", severity="warn",
            message=(f"R² = {fit.r2:.3f} — the points do not lie on a line. Note "
                     f"that R² is no protection at dof 1; read the ± and the "
                     f"pairwise range instead."),
            detail={"r2": float(fit.r2), "r2_min": float(r2_min)}))
    return out


def _kelvin_note(method_key: str) -> SeriesWarning:
    """The standing reminder of the unit law, attached to every K-based fit."""
    label = _METHOD_META[method_key]["label"]
    return SeriesWarning(
        code="kelvin_basis", severity="info",
        message=(f"{label} uses absolute temperature: values are converted to "
                 f"kelvin for the fit, regardless of the °C/K display switch. A "
                 f"heating rate is the same number in °C/min and K/min and is "
                 f"never converted."),
        detail={"method": method_key})


def _finalize(*, method_key: str, fit: LineFit, x: np.ndarray, y: np.ndarray,
              used: Sequence[RatePoint], n_total: int, n_included: int,
              q_ref_C_per_min: float, T_at_q_ref_C: float,
              r2_min: float = R2_MIN_DEFAULT,
              ea_per_slope_J: Optional[float] = None,
              extras: Optional[Dict[str, float]] = None,
              warnings: Optional[Sequence[SeriesWarning]] = None,
              fragility_m: Optional[float] = None,
              lasocka: Optional[Dict[str, Optional[float]]] = None,
              ) -> KineticFitResult:
    """Assemble a KineticFitResult and attach the model-level warnings.

    `ea_per_slope_J` is the factor f with Ea = f·slope in J/mol (−R for the
    Kissinger family, −R/1.052 for Flynn-Wall-Ozawa). Passing the factor
    rather than the Ea keeps the standard error, the confidence interval and
    the pairwise range consistent with the slope automatically — the three
    of them can never drift apart because they are all derived here, once.
    """
    meta = _METHOD_META[method_key]
    t_crit = t_crit_95(fit.dof)
    pairwise = _pairwise_slope_range(np.asarray(x, float), np.asarray(y, float))

    Ea = Ea_se = None
    ci: Optional[Tuple[float, float]] = None
    pairwise_Ea: Optional[Tuple[float, float]] = None
    if ea_per_slope_J is not None:
        Ea = float(fit.slope * ea_per_slope_J / 1000.0)
        if np.isfinite(fit.slope_se):
            Ea_se = float(abs(fit.slope_se * ea_per_slope_J) / 1000.0)
            if t_crit is not None:
                ci = (Ea - t_crit * Ea_se, Ea + t_crit * Ea_se)
        else:
            Ea_se = float("nan")
        if pairwise is not None:
            lo, hi = sorted(s * ea_per_slope_J / 1000.0 for s in pairwise)
            pairwise_Ea = (float(lo), float(hi))

    wlist: List[SeriesWarning] = list(warnings or [])
    wlist.extend(_common_warnings(fit, float(r2_min)))
    if meta["temp_basis"] != "°C":
        wlist.append(_kelvin_note(method_key))
    if Ea is not None and np.isfinite(Ea) and Ea < 0.0:
        wlist.append(SeriesWarning(
            code="negative_ea", severity="critical",
            message=(f"Ea = {Ea:.1f} kJ/mol is NEGATIVE: the temperature falls as "
                     f"the rate rises, which no relaxation- or nucleation-"
                     f"controlled transition does. The number is shown so you can "
                     f"see it; do not quote it."),
            row_uids=tuple(p.uid for p in used if p.uid),
            detail={"Ea_kJ_mol": Ea}))

    warns, notes = _split_by_severity(wlist)
    return KineticFitResult(
        method=method_key,
        method_label=meta["label"],
        equation=meta["equation"],
        slope=float(fit.slope), intercept=float(fit.intercept),
        slope_se=float(fit.slope_se), intercept_se=float(fit.intercept_se),
        n=int(fit.n), dof=int(fit.dof), r2=fit.r2, s_resid=float(fit.s_resid),
        x=np.asarray(x, float), y=np.asarray(y, float),
        residuals=np.asarray(fit.residuals, float),
        labels=[_label_of(p, i) for i, p in enumerate(used)],
        x_label=meta["x_label"], y_label=meta["y_label"],
        temp_basis=meta["temp_basis"],
        Ea_kJ_mol=Ea, Ea_se_kJ_mol=Ea_se, Ea_ci95_kJ_mol=ci, t_crit_95=t_crit,
        pairwise_slope=pairwise, pairwise_Ea_kJ_mol=pairwise_Ea,
        lasocka_A_C=None if lasocka is None else lasocka.get("A_C"),
        lasocka_A_se=None if lasocka is None else lasocka.get("A_se"),
        lasocka_A_K=None if lasocka is None else lasocka.get("A_K"),
        lasocka_B=None if lasocka is None else lasocka.get("B"),
        lasocka_B_se=None if lasocka is None else lasocka.get("B_se"),
        q_ref_C_per_min=float(q_ref_C_per_min),
        T_at_q_ref_C=float(T_at_q_ref_C),
        fragility_m=fragility_m,
        extras=dict(extras or {}),
        rows_used=[p.uid for p in used],
        n_total=int(n_total), n_included=int(n_included), n_fitted=int(fit.n),
        warnings=warns, notes=notes, ok=True, message="",
    )


def _impossible(method_key: str, points: Sequence[RatePoint], message: str,
                *, n_included: Optional[int] = None,
                warnings: Optional[Sequence[SeriesWarning]] = None
                ) -> KineticFitResult:
    """A well-formed result saying the arithmetic could not be done.

    Returned, never raised: the Qt layer must be able to show the reason in
    the same panel that would have shown the fit, and a batch export must get
    a row for this series rather than an exception that loses the other
    series in the batch. `ok=False` here always means arithmetic, never
    physics — see the KineticFitResult docstring.
    """
    meta = _METHOD_META.get(method_key, {
        "label": method_key, "equation": "", "x_label": "", "y_label": "",
        "temp_basis": "°C", "needs": ("Tg",),
    })
    pts = list(points)
    warns, notes = _split_by_severity(list(warnings or []))
    empty = np.empty(0, dtype=float)
    return KineticFitResult(
        method=method_key, method_label=meta["label"], equation=meta["equation"],
        slope=float("nan"), intercept=float("nan"),
        slope_se=float("nan"), intercept_se=float("nan"),
        n=0, dof=0, r2=None, s_resid=float("nan"),
        x=empty, y=empty.copy(), residuals=empty.copy(),
        labels=[], x_label=meta["x_label"], y_label=meta["y_label"],
        temp_basis=meta["temp_basis"],
        n_total=len(pts),
        n_included=len(pts) if n_included is None else int(n_included),
        n_fitted=0,
        warnings=warns, notes=notes, ok=False, message=message,
    )


def _guard(core: Callable[..., KineticFitResult], report_as: str,
           points: Sequence[RatePoint], **kw) -> KineticFitResult:
    """Run a fit core, turning ANY exception into an ok=False result.

    The never-raises contract is enforced here rather than trusted to each
    core, so an unforeseen numpy edge case in one model cannot take down a
    batch of six. ValueError from ols_line arrives with a message that
    already says what the user must change, so it is passed through verbatim.
    """
    pts = list(points)
    try:
        return core(pts, **kw)
    except ValueError as exc:
        return _impossible(report_as, pts, str(exc))
    except Exception as exc:                                  # pragma: no cover
        return _impossible(report_as, pts,
                           f"{type(exc).__name__}: {exc}")


def _require_two(method_key: str, used: Sequence[RatePoint],
                 n_total: int, what: str) -> None:
    """Raise the one-sentence ValueError _guard turns into a reported result."""
    if len(used) < 2:
        raise ValueError(
            f"No fit — {len(used)} of {n_total} point(s) usable (need 2). {what}")


# =============================================================================
# 6) The fits
# =============================================================================

def _lasocka_core(points: List[RatePoint], *, q_ref_C_per_min: float,
                  r2_min: float, temperature_in_K: bool) -> KineticFitResult:
    """Body of fit_lasocka; see that function for the physics.

    Separated so _guard() can wrap it and turn any arithmetic impossibility
    into a reported result instead of an exception.
    """
    used, dropped_unusable = _usable_points(points)
    _require_two("lasocka", used, len(points),
                 "Compute a Tg for more rows, or type values in by hand.")
    q = np.array([p.q for p in used], dtype=float)
    T_fit = np.array([p.T_K if temperature_in_K else float(p.T_C) for p in used],
                     dtype=float)
    T_K = np.array([p.T_K for p in used], dtype=float)
    x = np.log(q)
    fit = ols_line(x, T_fit)

    B = float(fit.slope)
    A_fit = float(fit.intercept)
    A_C = A_fit - K0 if temperature_in_K else A_fit
    A_K = A_fit if temperature_in_K else to_kelvin(A_fit)

    extras: Dict[str, float] = {}
    if B > 0.0:
        # The B <-> Ea_g cross-check, and the fragility it implies. Guarded on
        # B > 0 because R·Tg²/B at B <= 0 is either a division by zero or a
        # negative "activation energy" — an artefact of the broken fit, not a
        # measurement, and printing it would dignify it.
        Tg_mean_K = float(T_K.mean())
        Ea_g = R_GAS * Tg_mean_K ** 2 / B
        extras["Ea_g_from_B_kJ_mol"] = Ea_g / 1000.0
        extras["fragility_m_from_B"] = Ea_g / (LN10 * R_GAS * Tg_mean_K)

    warns: List[SeriesWarning] = _dropped_point_warning(
        "lasocka", dropped_unusable, len(points))
    if B < 0.0:
        warns.append(SeriesWarning(
            code="negative_lasocka_b", severity="critical",
            message=(f"Lasocka slope B = {B:.2f} K is NEGATIVE. B must be > 0 (Tg "
                     f"rises with rate); a negative B means the fit is not "
                     f"measuring a rate dependence. Do not quote B or Ea_g from "
                     f"it. Fitted anyway."),
            row_uids=tuple(p.uid for p in used if p.uid),
            detail={"B": B}))

    T_at_ref = float("nan")
    if np.isfinite(q_ref_C_per_min) and q_ref_C_per_min > 0.0:
        T_at_ref = A_C + B * float(np.log(abs(q_ref_C_per_min)))

    return _finalize(
        method_key="lasocka", fit=fit, x=x, y=T_fit, used=used,
        n_total=len(points), n_included=len(used),
        q_ref_C_per_min=q_ref_C_per_min, T_at_q_ref_C=T_at_ref, r2_min=r2_min,
        extras=extras, warnings=warns,
        lasocka={"A_C": A_C, "A_K": A_K, "A_se": float(fit.intercept_se),
                 "B": B, "B_se": float(fit.slope_se)},
    )


def fit_lasocka(points: Sequence[RatePoint], *,
                q_ref_C_per_min: float = DEFAULT_Q_REF,
                r2_min: float = R2_MIN_DEFAULT,
                temperature_in_K: bool = False) -> KineticFitResult:
    """LASOCKA   Tg = A + B·ln q,  q in K/min (== °C/min, numerically).

    B = dTg/d(ln q) is the rate sensitivity of the glass transition, in K per
    e-fold of heating rate, and it is what the Lasocka plot is FOR. B is
    invariant to the temperature unit — the kelvin offset cancels in a slope,
    verified identical to machine precision fitted in °C and in K
    (bit-identical on exactly collinear points; ~1e-15 relative apart once
    there is scatter, because ybar is not shifted by exactly K0 in floating
    point) — while A, the extrapolated
    Tg at q = 1 K/min, moves with both the temperature unit and the time base
    of q. Both A_C and A_K are therefore reported and the reference rate is
    always stated. See LASOCKA_B_NOTE for why B > 0 is required and for the
    B ≈ R·Tg²/Ea_g cross-check that is reported in `extras` when B > 0.

    `temperature_in_K` fits the same points with Tg expressed in kelvin. It
    exists to make the offset-invariance of B testable, not because the Qt
    layer should ever use it; the plotted and exported A is A_C either way.
    """
    return _guard(_lasocka_core, "lasocka", points,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min,
                  temperature_in_K=temperature_in_K)


def _one_over_T_core(points: List[RatePoint], *, method_key: str,
                     y_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     ea_per_slope_J: float, q_ref_C_per_min: float,
                     r2_min: float,
                     extras_of: Optional[Callable[[LineFit, np.ndarray, np.ndarray],
                                                  Dict[str, float]]] = None,
                     fragility: bool = False,
                     hint: str = "") -> KineticFitResult:
    """The shared body of every Arrhenius-family fit: y = f(q, T) vs 1/T.

    Written once because the four models differ ONLY in the y expression and
    in the Ea-per-slope factor. Four copies of the x = 1/to_kelvin(T) line
    would be four chances for one of them to forget the conversion, and that
    mistake produces a plausible-looking Ea that is wrong by ~3x.
    """
    used, dropped_unusable = _usable_points(points)
    _require_two(method_key, used, len(points), hint)
    q = np.array([p.q for p in used], dtype=float)
    T_K = np.array([p.T_K for p in used], dtype=float)
    x = 1.0 / T_K
    y = y_of(q, T_K)
    fit = ols_line(x, y)

    warns: List[SeriesWarning] = _dropped_point_warning(
        method_key, dropped_unusable, len(points))
    if fit.slope >= 0.0:
        warns.append(SeriesWarning(
            code="positive_slope_kissinger", severity="critical",
            message=(f"Slope is {fit.slope:+.1f} K, not negative → a non-positive "
                     f"activation energy. Check the peak assignment and each "
                     f"row's heating rate. Fitted anyway."),
            row_uids=tuple(p.uid for p in used if p.uid),
            detail={"slope": float(fit.slope)}))

    extras: Dict[str, float] = {}
    if extras_of is not None:
        extras.update(extras_of(fit, q, T_K))

    frag = None
    if fragility:
        Ea_J = fit.slope * ea_per_slope_J
        Tg_mean_K = float(T_K.mean())
        if np.isfinite(Ea_J) and Tg_mean_K > 0.0:
            frag = float(Ea_J / (LN10 * R_GAS * Tg_mean_K))

    T_at_ref = float("nan")
    if np.isfinite(q_ref_C_per_min) and q_ref_C_per_min > 0.0:
        qr = float(abs(q_ref_C_per_min))
        slope, intercept = float(fit.slope), float(fit.intercept)

        def residual(t_K: float) -> float:
            """y(q_ref, T) minus the fitted line at the same T; zero at the root."""
            yy = float(y_of(np.array([qr]), np.array([t_K]))[0])
            return yy - (intercept + slope / t_K)

        T_at_ref_K = _solve_T_at_q(residual)
        T_at_ref = float(T_at_ref_K - K0) if np.isfinite(T_at_ref_K) else float("nan")

    return _finalize(
        method_key=method_key, fit=fit, x=x, y=y, used=used,
        n_total=len(points), n_included=len(used),
        q_ref_C_per_min=q_ref_C_per_min, T_at_q_ref_C=T_at_ref, r2_min=r2_min,
        ea_per_slope_J=ea_per_slope_J, extras=extras, warnings=warns,
        fragility_m=frag,
    )


def fit_moynihan(points: Sequence[RatePoint], *,
                 q_ref_C_per_min: float = DEFAULT_Q_REF,
                 r2_min: float = R2_MIN_DEFAULT) -> KineticFitResult:
    """MOYNIHAN   ln q = C − Ea/(R·Tg);  x = 1/Tg [K⁻¹], y = ln q.

    This is the form Moynihan derived for the glass transition and the one to
    quote for an Ea(Tg): the enthalpy-relaxation argument gives
    d ln q / d(1/Tg) = −Ea/R with no Tg² term. Reports the fragility index
    m = Ea/(ln10·R·T̄g) alongside, because m is what Ea(Tg) is usually wanted
    for. Read TG_EA_CAVEAT before comparing this number with a
    crystallization activation energy — they are not the same kind of
    quantity.
    """
    return _guard(_one_over_T_core, "moynihan", points,
                  method_key="moynihan",
                  y_of=lambda q, T: np.log(q),
                  ea_per_slope_J=-R_GAS,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min,
                  extras_of=lambda fit, q, T: {"Tg_mean_K": float(T.mean())},
                  fragility=True,
                  hint="Compute a Tg for more rows, or type values in by hand.")


def fit_kissinger_tg(points: Sequence[RatePoint], *,
                     q_ref_C_per_min: float = DEFAULT_Q_REF,
                     r2_min: float = R2_MIN_DEFAULT) -> KineticFitResult:
    """KISSINGER FORM ON Tg   ln(q/Tg²) = −Ea/(R·Tg) + C.

    Offered because users ask for "the Kissinger plot" by name even when the
    quantity is a glass transition. It differs from Moynihan by exactly the
    Tg² term, worth −2·R·T̄g: on the synthetic five-rate set used in the tests
    the measured difference is −13.19 kJ/mol against a predicted −13.20. That
    is a real systematic offset, not noise, which is why both forms exist
    here rather than one being silently substituted for the other.
    """
    return _guard(_one_over_T_core, "kissinger_tg", points,
                  method_key="kissinger_tg",
                  y_of=lambda q, T: np.log(q / T ** 2),
                  ea_per_slope_J=-R_GAS,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min,
                  extras_of=lambda fit, q, T: {"Tg_mean_K": float(T.mean())},
                  fragility=True,
                  hint="Compute a Tg for more rows, or type values in by hand.")


def fit_kissinger(points: Sequence[RatePoint], *,
                  q_ref_C_per_min: float = DEFAULT_Q_REF,
                  r2_min: float = R2_MIN_DEFAULT) -> KineticFitResult:
    """KISSINGER   ln(q/Tp²) = −Ea/(R·Tp) + ln(A·R/Ea).

    The default for a peak temperature. x = 1/Tp in KELVIN (mandatory: a 1/T
    in °C is a physics error, not a units nicety), y = ln(q/Tp²) with q in
    K/min. slope = −Ea/R, and the intercept carries the pre-exponential:
    A = exp(intercept)·Ea/R, reported as extras["A_per_min"] because its time
    base is the time base of q. Changing q from K/min to K/s moves the
    intercept by ln 60 and leaves Ea untouched.

    This relation is the module's self-test: peak temperatures generated from
    it with Ea = 250 kJ/mol and A = 1e15 /min come back as 250.000000 kJ/mol
    and 1.0000000000e15 /min.
    """
    def _extras(fit: LineFit, q: np.ndarray, T: np.ndarray) -> Dict[str, float]:
        """Recover the pre-exponential A from the Kissinger intercept.

        A = exp(intercept)·Ea/R, and its time base is the time base of q, so
        A is reported as A_per_min because q is in K/min throughout. Only
        computed when Ea > 0: A from a negative Ea is not a frequency factor,
        it is the reciprocal of one with the sign lost.
        """
        Ea_J = -R_GAS * fit.slope
        out = {"ln_A_R_over_Ea": float(fit.intercept),
               "Tp_mean_K": float(T.mean())}
        if Ea_J > 0.0:
            out["A_per_min"] = _safe_exp(float(fit.intercept)) * Ea_J / R_GAS
        return out

    return _guard(_one_over_T_core, "kissinger", points,
                  method_key="kissinger",
                  y_of=lambda q, T: np.log(q / T ** 2),
                  ea_per_slope_J=-R_GAS,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min,
                  extras_of=_extras,
                  hint="Compute a peak temperature for more rows, or type values "
                       "in by hand.")


def fit_fwo(points: Sequence[RatePoint], *, doyle_correction: bool = True,
            q_ref_C_per_min: float = DEFAULT_Q_REF,
            r2_min: float = R2_MIN_DEFAULT) -> KineticFitResult:
    """FLYNN-WALL-OZAWA   ln q = C − c·Ea/(R·Tp),  c = 1.052 (Doyle).

    Ozawa's form drops the Tp² term that Kissinger keeps, which biases Ea
    upward by very nearly 2·R·T̄p. Doyle's c = 1.052 is the standard partial
    correction for that, and "Flynn-Wall-Ozawa" means the corrected form in
    the literature, so it is the default here.

    extras["Ea_c1_kJ_mol"] always carries the UNCORRECTED (c = 1) value. That
    is deliberate: on peak temperatures generated exactly from Kissinger with
    Ea = 250 kJ/mol, the uncorrected value comes back as 264.19 kJ/mol
    (+5.68 %, and 264.19 − 250 = 14.19 ≈ 2·R·T̄p = 14.24) while the Doyle form
    gives 251.13 (+0.45 %). Shipping only one number would hide a 5.7 %
    systematic; shipping both makes it a visible property of the method
    rather than a surprise in someone's paper.
    """
    c = DOYLE_C if doyle_correction else 1.0
    other_c = 1.0 if doyle_correction else DOYLE_C

    def _extras(fit: LineFit, q: np.ndarray, T: np.ndarray) -> Dict[str, float]:
        """Both Ea values and the 2·R·T̄p bias that separates them.

        Ea_c1_kJ_mol is always the uncorrected one whichever form was asked
        for, so the systematic is visible in the export rather than implied
        by which checkbox happened to be ticked.
        """
        Tp_mean = float(T.mean())
        return {
            "doyle_c": float(c),
            "Tp_mean_K": Tp_mean,
            "Ea_c1_kJ_mol": float(-R_GAS * fit.slope / 1000.0),
            "Ea_c1052_kJ_mol": float(-R_GAS * fit.slope / DOYLE_C / 1000.0),
            "ozawa_bias_2RTp_kJ_mol": float(2.0 * R_GAS * Tp_mean / 1000.0),
            "Ea_other_c_kJ_mol": float(-R_GAS * fit.slope / other_c / 1000.0),
        }

    return _guard(_one_over_T_core, "fwo", points,
                  method_key="fwo",
                  y_of=lambda q, T: np.log(q),
                  ea_per_slope_J=-R_GAS / c,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min,
                  extras_of=_extras,
                  hint="Compute a peak temperature for more rows, or type values "
                       "in by hand.")


def _augis_bennett_core(points: List[RatePoint], *, q_ref_C_per_min: float,
                        r2_min: float) -> KineticFitResult:
    """Body of fit_augis_bennett; see that function for the physics.

    Kept separate from _one_over_T_core because Augis-Bennett is the one
    model with a per-point extra input (T0), and that input drives a drop
    rule the other four models must not inherit.
    """
    n_total = len(points)
    included, dropped_unusable = _usable_points(points)
    used: List[RatePoint] = []
    dropped: List[RatePoint] = []
    for p in included:
        t0 = float(p.T0_C)
        if np.isfinite(t0) and (float(p.T_C) - t0) > 0.0:
            used.append(p)
        else:
            dropped.append(p)
    _require_two("augis_bennett", used, n_total,
                 "Enter T0 (the ramp start) for more rows, or use Kissinger, "
                 "which needs no T0.")

    q = np.array([p.q for p in used], dtype=float)
    T_K = np.array([p.T_K for p in used], dtype=float)
    dT = np.array([float(p.T_C) - float(p.T0_C) for p in used], dtype=float)
    x = 1.0 / T_K
    y = np.log(q / dT)
    fit = ols_line(x, y)

    warns: List[SeriesWarning] = _dropped_point_warning(
        "augis_bennett", dropped_unusable, n_total)
    if dropped:
        names = ", ".join(_label_of(p, i) for i, p in enumerate(dropped))
        warns.append(SeriesWarning(
            code="no_t0", severity="warn",
            message=(f"{len(dropped)} of {len(included)} points have no usable T0 "
                     f"(missing, or not below the peak) and were excluded from "
                     f"Augis-Bennett: {names}. Enter T0 (the ramp start) for them, "
                     f"or use Kissinger, which needs no T0."),
            row_uids=tuple(p.uid for p in dropped if p.uid),
            detail={"n_dropped": len(dropped), "n_available": len(included)}))
    if fit.slope >= 0.0:
        warns.append(SeriesWarning(
            code="positive_slope_kissinger", severity="critical",
            message=(f"Slope is {fit.slope:+.1f} K, not negative → a non-positive "
                     f"activation energy. Check the peak assignment and each "
                     f"row's heating rate. Fitted anyway."),
            row_uids=tuple(p.uid for p in used if p.uid),
            detail={"slope": float(fit.slope)}))

    extras: Dict[str, float] = {
        "ln_nu0": float(fit.intercept),
        "nu0_per_min": _safe_exp(float(fit.intercept)),
        "T0_mean_C": float(np.mean([float(p.T0_C) for p in used])),
        "Tp_mean_K": float(T_K.mean()),
    }

    T_at_ref = float("nan")
    if np.isfinite(q_ref_C_per_min) and q_ref_C_per_min > 0.0:
        qr = float(abs(q_ref_C_per_min))
        T0_mean_K = to_kelvin(extras["T0_mean_C"])
        slope, intercept = float(fit.slope), float(fit.intercept)

        def residual(t_K: float) -> float:
            """As above, with the mean T0 standing in for a per-point one.

            A back-prediction at a rate nobody measured has no T0 of its own,
            and the mean of the fitted rows' ramp starts is the only defensible
            stand-in; nan below T0 keeps the bracket search out of ln(<=0).
            """
            if t_K <= T0_mean_K:
                return float("nan")
            return float(np.log(qr / (t_K - T0_mean_K))) - (intercept + slope / t_K)

        T_at_ref_K = _solve_T_at_q(residual, lo=max(20.0, T0_mean_K + 1e-6))
        T_at_ref = float(T_at_ref_K - K0) if np.isfinite(T_at_ref_K) else float("nan")

    return _finalize(
        method_key="augis_bennett", fit=fit, x=x, y=y, used=used,
        n_total=n_total, n_included=len(included),
        q_ref_C_per_min=q_ref_C_per_min, T_at_q_ref_C=T_at_ref, r2_min=r2_min,
        ea_per_slope_J=-R_GAS, extras=extras, warnings=warns,
    )


def fit_augis_bennett(points: Sequence[RatePoint], *,
                      q_ref_C_per_min: float = DEFAULT_Q_REF,
                      r2_min: float = R2_MIN_DEFAULT) -> KineticFitResult:
    """AUGIS-BENNETT   ln(q/(Tp − T0)) = −Ea/(R·Tp) + ln(ν₀).

    T0 is the temperature at which the RAMP STARTS, so that (Tp − T0)/q is
    the elapsed heating time to the peak — that is what the derivation
    requires, and it is exactly the segmentation layer's segment T-span
    start. It is NOT the crystallization onset Tx, though many papers
    substitute it, so the input field must be labelled "T0 (ramp start)" and
    default to the segment start rather than be silently invented.

    Its advantage over Kissinger is that the intercept is ln(ν₀) directly,
    with ν₀ in min⁻¹ when q is in K/min. Its cost is that it needs a T0 per
    point: points without a usable one are dropped from THIS fit only — never
    from the series, never from any other method — and named in a `no_t0`
    warning that points at Kissinger as the T0-free alternative.
    """
    return _guard(_augis_bennett_core, "augis_bennett", points,
                  q_ref_C_per_min=q_ref_C_per_min, r2_min=r2_min)


# =============================================================================
# 7) Registry
# =============================================================================

#: The method registry, mirroring figures_science.FIT_MODELS so the Qt combo
#: box is BUILT from this dict rather than duplicating the list of methods.
#: `needs` gates enablement ("this series holds Tg, so Kissinger-on-Tp is
#: offered but flagged") and `needs_T0` tells the UI when to show the T0
#: column, so the UI never re-decides which method suits which quantity.
KINETIC_METHODS: Dict[str, Dict[str, Any]] = {
    key: dict(meta, fn=fn)
    for key, meta, fn in (
        ("lasocka", _METHOD_META["lasocka"], fit_lasocka),
        ("moynihan", _METHOD_META["moynihan"], fit_moynihan),
        ("kissinger_tg", _METHOD_META["kissinger_tg"], fit_kissinger_tg),
        ("kissinger", _METHOD_META["kissinger"], fit_kissinger),
        ("fwo", _METHOD_META["fwo"], fit_fwo),
        ("augis_bennett", _METHOD_META["augis_bennett"], fit_augis_bennett),
    )
}


def default_method_for(quantity_kind: str) -> str:
    """"Tg" -> "lasocka";  "Tp" -> "kissinger";  anything else -> "lasocka".

    Lasocka is the Tg default rather than Moynihan because it is the plot the
    user asked for by name and because its B is a direct measurement, whereas
    Ea(Tg) needs the caveat in TG_EA_CAVEAT before it means anything. Both
    are one combo-box entry apart.

    Never raises on an unknown kind: an unrecognised quantity gets the Tg
    default and the layer above flags the mismatch.
    """
    kind = str(quantity_kind or "").strip().lower()
    for key, meta in KINETIC_METHODS.items():
        if any(kind == d.lower() for d in meta.get("default_for", ())):
            return key
    return "lasocka"


def fit_points(points: Sequence[RatePoint], method: str, **kw) -> KineticFitResult:
    """Registry dispatch: run `method` over `points`.

    Accepts the registry key ("kissinger") or the display label, so a combo
    box can pass either. An unknown method or an unexpected keyword comes
    back as an ok=False result naming the valid keys — not an exception,
    because this is the entry point a batch loop calls once per series and
    one bad entry must not lose the rest of the batch.
    """
    key = str(method or "").strip()
    entry = KINETIC_METHODS.get(key)
    if entry is None:
        lowered = key.lower()
        for k, meta in KINETIC_METHODS.items():
            if k.lower() == lowered or str(meta.get("label", "")).lower() == lowered:
                key, entry = k, meta
                break
    if entry is None:
        return _impossible(
            key or "unknown", points,
            f"Unknown method {method!r}. Valid methods: "
            f"{', '.join(sorted(KINETIC_METHODS))}.")
    try:
        return entry["fn"](points, **kw)
    except TypeError as exc:
        return _impossible(key, points,
                           f"{key} does not accept those options: {exc}")
