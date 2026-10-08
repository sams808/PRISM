"""
dta_rate_series.py — assembling a multi-rate DTA/DSC series and telling the
truth about it.

Rows come from four places and all four produce the same `RateRow`:

  (a) the heating segments of ONE cyclic run   -> `rows_from_segments`
  (b) SEPARATE single-rate files, the textbook
      one-specimen-per-rate design             -> `rows_from_file`
  (c) records already imported into the library -> `rows_from_record`
  (d) typed by hand (literature, another tool) -> `manual_row`

Mixing them is allowed, and the mixture is flagged rather than forbidden.

DESIGN CONTRACT — the user's standing instruction was "ok but still make this
possible to process such data", and this module is where that instruction is
kept:

  * `check_series()` NEVER raises, NEVER filters a row and NEVER gates a fit.
    It returns a list of `SeriesWarning`, sorted worst first.
  * `fit_series()` NEVER raises. An algebraic impossibility (fewer than two
    fittable rows, one single rate, a zero rate span) comes back as
    `ok=False` with a message. A PHYSICAL objection — a negative activation
    energy, a negative Lasocka B, cooling-segment kinetics, four successive
    cycles of one specimen, 1 % mass loss along the series — never reaches
    that branch: the number is computed, returned, and carries the warning.
  * Warning prose therefore never contains "cannot", "invalid" or "refusing",
    and every message that a reader might mistake for a refusal ends
    "Fitted anyway."

LAYERING (MERGED_DECISIONS M2). `dta_kinetics` knows only (rate, temperature)
and owns the arithmetic and the warnings a fit can raise about itself.
Everything about WHERE a point came from — which file, which segment, which
specimen, how many prior heating ramps it had seen, how much mass it had lost
— lives here, and so do the provenance warnings. The dependency is one-way:

    dta_program -> dta_segments -> dta_rate_series -> qt_dta
    dta_kinetics --------------->  dta_rate_series

TWO ADAPTER RULES, both load-bearing (MERGED_DECISIONS M3 and M4):

  * `_seg_fields()` is the ONLY place `dta_segments` attribute names appear.
    If that module renames a field, this one function is the only edit.
  * Array preparation is NEVER re-implemented here. `SegmentedRun.xy()` /
    `dta_segments.prepare_segment_xy()` is the single place that slices,
    masks, sorts and de-duplicates. A whole-file `np.argsort` on a cyclic
    record interleaves four heating ramps and turns a 620 °C glass transition
    into 432 °C; the rate series must never build its own arrays and risk
    reproducing that.

Units: temperatures are stored in CELSIUS, always, with the unit in the field
name (`value_C`) so a display-unit switch can never corrupt a stored
measurement. Heating rates are stored in °C/min, which is numerically K/min
(a rate is a temperature INTERVAL per unit time and the +273.15 offset
cancels), so a rate is never converted anywhere.
"""
from __future__ import annotations

import ntpath
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dta import dta_kinetics as dk
from dta import dta_segments as ds
from dta.dta_kinetics import (KINETIC_METHODS, K0, PLAUSIBLE_Q, PLAUSIBLE_T_C,
                              KineticFitResult, RatePoint, SeriesWarning,
                              default_method_for, max_severity, sort_warnings,
                              warning_codes)

NAN = float("nan")

# =============================================================================
# 0) Vocabularies
# =============================================================================

#: HOW a row was added. Never confused with WHERE its number came from
#: (`VALUE_SOURCES`) or WHICH rate candidate the fit used (`RATE_SOURCES`) —
#: three orthogonal questions that a single "source" string would conflate.
ROW_SOURCES: Tuple[str, ...] = ("segment", "file", "library", "manual")

#: Where a row's measured value came from. "—" is the deliberate default:
#: adding a row does NOT compute it, because the settings that decide the
#: number are stale by construction at add time (SPEC_rate_series §1
#: conflict 4). "failed" is a stored outcome with a verbatim message, never a
#: swallowed exception and never a row that quietly disappears from the fit.
VALUE_SOURCES: Tuple[str, ...] = ("—", "computed", "manual", "failed")

#: Which rate candidate to use. "auto" walks the ladder in `resolve_rate`.
RATE_SOURCES: Tuple[str, ...] = ("auto", "measured", "programmed", "filename", "manual")

#: How the characteristic temperature was constructed. Kept ORTHOGONAL to
#: `quantity_kind` ("Tg" | "Tp"): conflating them into one string makes
#: `mixed_quantity` (a critical problem — a Tg and a Tp are different
#: temperatures) indistinguishable from `mixed_method` (a warn — method
#: scatter entering the rate slope), and those need different fixes.
METHODS: Tuple[str, ...] = ("parallel", "double", "deriv", "peak_max", "peak_min", "manual")

#: Bumped inside each stored series record, so a field can be added without
#: touching `core.project_io.PROJECT_VERSION`.
SERIES_SCHEMA_VERSION = 1

#: Identifies the writer in every exported fit CSV.
GENERATED_BY = "PRISM dta_rate_series v1"

#: Tier-B warning codes NOT implemented in this module, and why. Exported so
#: a UI (or a reviewer) can see the omission rather than assume the check ran
#: and passed. `hidden_excluded` is the only genuinely absent one: it depends
#: on a "Show excluded" toggle that exists in the Qt layer and has no field
#: in this model, so the Qt layer owns it.
STUBBED_WARNING_CODES: Tuple[str, ...] = ("hidden_excluded",)

#: How far outside the rest of the series a de-converted value may land and
#: still count as the same series. A row converted to kelvin twice, minus
#: 273.15, IS one of the series' own measurements, so it lands inside their
#: range — but it may be the extreme member, so a few degrees of slack is
#: needed. Used only by the `unit_suspect` kelvin fingerprint.
UNIT_TWIN_PAD_C: float = 5.0


def _f(value: Any, default: float = NAN) -> float:
    """float() that answers `default` instead of raising. Used on every value
    that reaches this module from a CSV cell, a JSON round trip or a widget."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite(value: Any) -> bool:
    """True when `value` is a real, finite number."""
    v = _f(value)
    return bool(np.isfinite(v))


def _stem(path: str) -> str:
    """Basename without extension, using ntpath so a Windows path is split
    correctly even when this runs on Linux (the same reason
    `htxrd_science.ramp_value_from_template` gives)."""
    base = ntpath.basename(str(path or ""))
    dot = base.rfind(".")
    return base[:dot] if dot > 0 else base


# =============================================================================
# 1) The row
# =============================================================================

@dataclass
class RateRow:
    """One (rate, temperature) measurement, with everything known about where
    it came from.

    Six fields and only six are USER-EDITABLE: `sample`, `include`, `note`,
    `q_manual_C_per_min`, `rate_source` and `value_C`. Everything else is
    provenance written once by whichever builder created the row. Typing a
    value sets `value_source="manual"` and `method="manual"` and appends the
    computed value to `note`, so a hand-entered number is never mistaken for
    a measured one and "Compute all" never silently overwrites it.

    `value_C` is CELSIUS, always — the unit is in the field name and there is
    no `value_unit` field. A °C/K display switch relabels a column header and
    renders `value_C + 273.15` fresh; it can never convert a stored number,
    which deletes an entire class of bug (a value converted to kelvin twice
    still looks like a plausible Tg and makes Ea wrong by roughly a factor of
    three).
    """

    # ---- identity -------------------------------------------------------
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_kind: str = "manual"                # ROW_SOURCES — HOW it was added
    sample: str = ""                           # USER-EDITABLE: the plot label
    specimen_key: str = ""                     # identifies the PHYSICAL specimen
    file: str = ""
    record_id: str = ""                        # Spectrum.id when source_kind == "library"

    # ---- segment provenance ---------------------------------------------
    segment_index: int = -1
    segment_kind: str = ""                     # dta_segments Segment.kind
    segment_label: str = ""                    # "heat 2 (cycle 2)"
    role_in_cycle: str = ""                    # dta_segments Segment.role
    cycle_index: int = 0
    row_start: int = -1
    """First ORIGINAL dataframe row of this segment, from
    `SegmentedRun.row_indices(seg)[0]`. NEVER `Segment.row_start`, which
    indexes the CLEANED arrays: the two differ by exactly the rows the
    cleaning dropped, which is zero on every file shipped here and therefore
    a confusion that is invisible in testing and wrong in the field."""
    row_stop: int = -1                         # `row_indices(seg)[-1] + 1`, exclusive
    n_heating_segments_in_source: int = 1      # 1 = clean one-ramp file; >1 = cyclic run

    # ---- thermal history -------------------------------------------------
    state: str = "unknown"                     # "virgin" | "after-cycle-N" | "unknown"
    prior_heats: int = 0                       # complete heating ramps BEFORE this one
    prior_max_T_C: float = NAN
    cum_mass_loss_pct: float = NAN
    """Mass lost by the end of THIS heating ramp, as a percentage of the
    as-loaded specimen mass.

    WHY it is read only across HEATING segments (MERGED_DECISIONS M1): every
    cooling segment of the real cyclic run shows an apparent mass GAIN of
    +0.084 to +0.130 mg. Nothing is condensing — that is buoyancy. The
    specimen plus pan displaces roughly 0.11 cm³ of purge gas whose density
    rises as the furnace cools, so the balance reads heavier on the way down
    and lighter on the way up. Summing per-segment deltas would therefore
    accumulate one buoyancy offset per cycle and the error would GROW with
    cycle count. This field instead takes a single difference between two
    readings, and every row's second reading is at the top of its own heating
    ramp — 1292 to 1295 °C on all four cycles of the real run — so the
    DIFFERENCES BETWEEN ROWS are at matched temperature and buoyancy-free.
    That is what makes `mass_loss_spread` a trustworthy warning rather than a
    new wrong number. The absolute value still carries one constant offset
    (the two readings are 19 °C and 1292 °C apart), which is why
    `mass_loss_high` is a warn and not a measurement anyone should quote.

    On a cooling or hold row the value is carried forward from the most
    recent heating ramp, because a cooling-segment delta-m is meaningless and
    a stale honest reading beats a fresh wrong one."""
    sample_mass_mg: float = NAN

    # ---- rate: every candidate kept, the used value derived --------------
    q_measured_C_per_min: float = NAN          # Segment.rate_mode (mode-refined)
    q_programmed_C_per_min: float = NAN        # Segment.rate_nominal_C_min; NaN unaligned
    q_filename_C_per_min: float = NAN
    q_manual_C_per_min: float = NAN            # USER-EDITABLE
    rate_source: str = "auto"                  # USER-EDITABLE: pins a candidate

    # ---- the measurement --------------------------------------------------
    quantity_kind: str = "Tg"                  # "Tg" | "Tp"
    method: str = "parallel"                   # METHODS
    signal: str = ""                           # the Y column the number was read off
    window_C: Tuple[float, float] = (NAN, NAN)
    value_C: float = NAN
    value_source: str = "—"                    # VALUE_SOURCES
    error: str = ""                            # verbatim exception text; SEPARATE from note
    extras: Dict[str, float] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    T0_C: float = NAN                          # segment T_start, for Augis-Bennett

    # ---- flags / user controls -------------------------------------------
    is_partial: bool = False
    setpoint_reached: Optional[bool] = None    # None = unknown (no alignment available)
    include: bool = True                       # USER-EDITABLE
    note: str = ""                             # USER-EDITABLE

    # ---- derived ----------------------------------------------------------
    @staticmethod
    def new_uid() -> str:
        """A fresh row identity, matching `core.qt_models.Spectrum.new_id()`."""
        return uuid.uuid4().hex

    @property
    def q_C_per_min(self) -> float:
        """The SIGNED rate the fit will see — `resolve_rate(self)[0]`.

        Derived, never stored, so the table, the fit and the export cannot
        disagree about which candidate was used. The sign is kept: a cooling
        segment carries a negative rate, `cooling_row` says so, and the fit
        takes the magnitude.
        """
        return resolve_rate(self)[0]

    @property
    def q_used_source(self) -> str:
        """Which candidate `resolve_rate` ACTUALLY used, not which was asked for."""
        return resolve_rate(self)[1]

    @property
    def value_K(self) -> float:
        """Display/export only. Nothing in this module fits on this value."""
        return dk.to_kelvin(self.value_C) if _finite(self.value_C) else NAN

    @property
    def fittable(self) -> bool:
        """Included, with a finite non-zero rate and a finite value.

        A rate of exactly zero is excluded even though SPEC_rate_series §4.1
        lists only finiteness: `ln q` is undefined at q = 0, so
        `dta_kinetics._usable_points` would drop such a point anyway and
        `n_fitted` would then disagree with the fit's own n.

        The two counts agree on everything EXCEPT a value at or below
        absolute zero. That one stays fittable here — it is a real, typed
        number and the table must show it and warn about it (`unit_suspect`
        does) rather than make it vanish — while `_usable_points` drops it,
        because 1/T and ln T have no answer there. The fit reports the drop
        itself: `n_fitted < n_total` on the result, plus a `dropped_points`
        warning naming the point. So the two counts can differ by exactly
        that, and when they do, something says so.
        """
        q = self.q_C_per_min
        return bool(self.include and np.isfinite(q) and q != 0.0
                    and _finite(self.value_C))

    @property
    def virgin(self) -> bool:
        """True when no complete heating ramp preceded this measurement."""
        return int(self.prior_heats) == 0

    @property
    def label(self) -> str:
        """Plot-legend text: 'BM-0-cycles · heat 1 (cycle 1) @ 20.0 °C/min'."""
        base = (self.sample or _stem(self.file) or self.record_id or "row").strip()
        bits = [base] if base else []
        if self.segment_label:
            bits.append(self.segment_label)
        text = " · ".join(bits) if bits else "row"
        q = self.q_C_per_min
        if np.isfinite(q):
            text += f" @ {abs(q):.1f} °C/min"
        return text

    @property
    def flags(self) -> str:
        """';'-joined short flags for the table's Flags column and the CSV."""
        out: List[str] = []
        if self.is_partial:
            out.append("partial")
        if self.setpoint_reached is False:
            out.append("setpoint not reached")
        q = self.q_C_per_min
        if (self.segment_kind == ds.KIND_COOL or self.role_in_cycle == ds.ROLE_COOL
                or (np.isfinite(q) and q < 0.0)):
            out.append("cooling")
        if self.q_used_source in ("manual", "filename"):
            out.append("rate " + self.q_used_source)
        if self.value_source == "manual":
            out.append("value typed")
        if self.value_source == "failed":
            out.append("failed")
        if not self.include:
            out.append("excluded")
        return ";".join(out)

    def to_point(self) -> RatePoint:
        """The provenance-free (rate, temperature) pair `dta_kinetics` fits."""
        return RatePoint(q_C_per_min=self.q_C_per_min, T_C=_f(self.value_C),
                         quantity_kind=self.quantity_kind, label=self.label,
                         uid=self.uid, T0_C=_f(self.T0_C))

    def set_manual_value(self, value_C: float) -> None:
        """Store a hand-typed value, recording what it replaced.

        The note is appended rather than replaced because the computed number
        is evidence: a reviewer looking at a series months later must be able
        to see that row 2's 620.8 was overridden by hand with 618.0, and by
        how much.
        """
        old = _f(self.value_C)
        self.value_C = _f(value_C)
        self.value_source = "manual"
        self.method = "manual"
        if np.isfinite(old):
            addition = f"manual override; computed was {old:.1f} °C"
            self.note = f"{self.note}; {addition}" if self.note else addition

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict. NaN becomes None so `json.dumps` stays valid, and
        `window_C` becomes a list because JSON has no tuple.

        `extras` gets the same treatment as every other float field. It did
        not, and a row whose parallel tangent failed therefore wrote a bare
        `NaN` token into `dta_rate_series.json` -- which PRISM reads back fine
        and a strict JSON reader refuses. Two of the eight rows the committed
        cyclic fixture produces hit that case, so it was the normal outcome
        rather than an edge one.
        """
        out: Dict[str, Any] = {}
        for name in ("uid", "source_kind", "sample", "specimen_key", "file",
                     "record_id", "segment_index", "segment_kind", "segment_label",
                     "role_in_cycle", "cycle_index", "row_start", "row_stop",
                     "n_heating_segments_in_source", "state", "prior_heats",
                     "rate_source", "quantity_kind", "method", "signal",
                     "value_source", "error", "is_partial", "setpoint_reached",
                     "include", "note"):
            out[name] = getattr(self, name)
        for name in ("prior_max_T_C", "cum_mass_loss_pct", "sample_mass_mg",
                     "q_measured_C_per_min", "q_programmed_C_per_min",
                     "q_filename_C_per_min", "q_manual_C_per_min", "value_C",
                     "T0_C"):
            v = _f(getattr(self, name))
            out[name] = None if not np.isfinite(v) else v
        lo, hi = self.window_C
        out["window_C"] = [None if not _finite(lo) else _f(lo),
                           None if not _finite(hi) else _f(hi)]
        out["extras"] = {str(k): (None if not _finite(v) else _f(v))
                         for k, v in dict(self.extras).items()}
        out["settings"] = dict(self.settings)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RateRow":
        """Inverse of `to_dict`. Unknown keys are ignored rather than fatal:
        a project saved by a newer PRISM must still open in an older one."""
        d = dict(d or {})
        row = cls(uid=str(d.get("uid") or cls.new_uid()))
        for name in ("source_kind", "sample", "specimen_key", "file", "record_id",
                     "segment_kind", "segment_label", "role_in_cycle", "state",
                     "rate_source", "quantity_kind", "method", "signal",
                     "value_source", "error", "note"):
            if d.get(name) is not None:
                setattr(row, name, str(d[name]))
        for name in ("segment_index", "cycle_index", "row_start", "row_stop",
                     "n_heating_segments_in_source", "prior_heats"):
            if d.get(name) is not None:
                try:
                    setattr(row, name, int(d[name]))
                except (TypeError, ValueError):
                    pass
        for name in ("prior_max_T_C", "cum_mass_loss_pct", "sample_mass_mg",
                     "q_measured_C_per_min", "q_programmed_C_per_min",
                     "q_filename_C_per_min", "q_manual_C_per_min", "value_C",
                     "T0_C"):
            setattr(row, name, _f(d.get(name)))
        win = d.get("window_C") or (NAN, NAN)
        try:
            row.window_C = (_f(win[0]), _f(win[1]))
        except (TypeError, IndexError, KeyError):
            row.window_C = (NAN, NAN)
        row.extras = {str(k): _f(v) for k, v in dict(d.get("extras") or {}).items()}
        row.settings = dict(d.get("settings") or {})
        row.is_partial = bool(d.get("is_partial", False))
        row.include = bool(d.get("include", True))
        sp = d.get("setpoint_reached", None)
        row.setpoint_reached = None if sp is None else bool(sp)
        return row


# =============================================================================
# 2) resolve_rate — the one place four candidates become one number
# =============================================================================

#: The "auto" ladder. MEASURED first, deliberately reversing the order one of
#: the source designs proposed: the measured rate is what the specimen saw,
#: the programmed rate is only the instruction. On the real run they agree to
#: 0.25 %, which moves ln q by 0.002, so nothing is lost by preferring the
#: honest one — and when they DISAGREE, preferring the instruction would put a
#: rate on the x-axis that the furnace never delivered.
_RATE_LADDER: Tuple[Tuple[str, str], ...] = (
    ("measured", "q_measured_C_per_min"),
    ("programmed", "q_programmed_C_per_min"),
    ("filename", "q_filename_C_per_min"),
    ("manual", "q_manual_C_per_min"),
)
_RATE_FIELD = dict(_RATE_LADDER)


def resolve_rate(row: RateRow) -> Tuple[float, str]:
    """(rate, source actually used) for one row — the single rate authority.

    `rate_source="auto"` walks measured -> programmed -> filename -> manual
    and stops at the first finite candidate. An explicit `rate_source` PINS a
    candidate, but when that candidate is non-finite the ladder is walked
    anyway and the SOURCE ACTUALLY USED is returned, never the one requested.

    That last rule is the same honesty rule
    `TgParallelTangentResult.low_mode` already follows, and it is the reason
    this function exists at all: the table, the fit and the CSV export all
    call it, so "the plot fitted a different rate than the CSV printed" is
    structurally impossible rather than merely unlikely.

    Returns `(nan, "none")` when no candidate is finite. The sign is
    preserved — a cooling segment's negative rate is a real measurement and
    `cooling_row` reports it; taking the magnitude is the fit's job.
    """
    want = str(getattr(row, "rate_source", "auto") or "auto").strip().lower()
    if want not in RATE_SOURCES:
        want = "auto"
    if want != "auto":
        v = _f(getattr(row, _RATE_FIELD[want], NAN))
        if np.isfinite(v):
            return v, want
    for name, attr in _RATE_LADDER:
        v = _f(getattr(row, attr, NAN))
        if np.isfinite(v):
            return v, name
    return NAN, "none"


# =============================================================================
# 3) The series
# =============================================================================

@dataclass(frozen=True)
class SeriesTolerances:
    """Every threshold a check can trip, in one frozen place.

    Frozen because a tolerance that a check mutates while running is a
    tolerance nobody can reproduce from the exported CSV; `detail` on each
    warning carries the value that fired alongside the threshold that fired
    it, so a reviewer can tell a tripped default from a tripped physical
    limit.
    """

    rate_tol_pct: float = 2.0
    """Two rates count as "the same rate" below this relative difference."""
    rate_mismatch_tol_pct: float = 5.0
    """Programmed vs measured disagreement above this raises `rate_mismatch`.
    On the real run the four heating ramps sit at 0.23 / 0.19 / 0.06 / 0.12 %,
    so a 5 % gate is the furnace failing to follow the program, not noise."""
    span_critical: float = 2.0
    span_warn: float = 4.0
    """q_max/q_min. Under 4x the lever arm is shorter than one e-fold of
    useful range; under 2x there is essentially none."""
    mass_loss_spread_pct: float = 0.5
    mass_loss_abs_pct: float = 1.0
    window_tol_C: float = 1.0
    r2_min: float = 0.90
    kelvin_fingerprint_C: float = 1.0
    """How far outside the rest of the series a de-converted value may land
    and still be called a kelvin twin, widening `UNIT_TWIN_PAD_C`.

    It used to mean "two rows differing by 273.15 +- this", which is how
    SPEC_rate_series wrote the check -- and that test missed the bug it was
    named for (a doubly-converted row sits 273.15 above ITS OWN value, not
    above another row's) while false-firing on any series spanning 273 C.
    See `_chk_units` for the rule that replaced it."""


@dataclass
class RateSeries:
    """An ordered set of `RateRow`s and the tolerances its checks use.

    `rows` is in INSERTION ORDER and is never sorted in place: the table
    shows it, "Row 3" in a warning means the third line on screen, and
    `fitted_rows()` hands out a sorted COPY for plotting. Sorting the live
    list would renumber every warning the user is reading.
    """

    series_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Rate series"
    quantity_kind: str = "Tg"
    rows: List[RateRow] = field(default_factory=list)
    tolerances: SeriesTolerances = SeriesTolerances()
    notes: str = ""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- structural integrity -------------------------------------------
    def validate(self) -> None:
        """Raise ValueError ONLY on a programming bug.

        Duplicate uids and unknown enum values are bugs in the code that
        built the series, and they corrupt every uid-keyed lookup (a warning
        would implicate the wrong row, a table edit would write to the wrong
        row). Nothing PHYSICAL is checked here and nothing physical ever
        raises — that is `check_series`'s job and it returns warnings.
        """
        seen = set()
        for i, row in enumerate(self.rows, 1):
            if row.uid in seen:
                raise ValueError(
                    f"duplicate row uid {row.uid!r} at position {i}: every uid-keyed "
                    f"lookup (warnings, table edits, exports) would target the wrong row")
            seen.add(row.uid)
            if row.source_kind not in ROW_SOURCES:
                raise ValueError(f"row {i}: source_kind {row.source_kind!r} "
                                 f"is not one of {ROW_SOURCES}")
            if row.value_source not in VALUE_SOURCES:
                raise ValueError(f"row {i}: value_source {row.value_source!r} "
                                 f"is not one of {VALUE_SOURCES}")
            if row.rate_source not in RATE_SOURCES:
                raise ValueError(f"row {i}: rate_source {row.rate_source!r} "
                                 f"is not one of {RATE_SOURCES}")

    # ---- membership -------------------------------------------------------
    def add(self, row: RateRow) -> None:
        """Append one row, keeping insertion order."""
        self.rows.append(row)

    def extend(self, rows: Iterable[RateRow]) -> None:
        """Append many rows, keeping insertion order."""
        self.rows.extend(rows)

    def remove(self, uid: str) -> bool:
        """Remove by uid, never by position. True when a row went."""
        for i, row in enumerate(self.rows):
            if row.uid == uid:
                del self.rows[i]
                return True
        return False

    def get(self, uid: str) -> Optional[RateRow]:
        """The row with this uid, or None."""
        for row in self.rows:
            if row.uid == uid:
                return row
        return None

    def index_of(self, uid: str) -> int:
        """1-BASED position, matching what the table shows and what every
        warning message means by "Row 2". 0 when the uid is not present."""
        for i, row in enumerate(self.rows, 1):
            if row.uid == uid:
                return i
        return 0

    def clear(self) -> None:
        """Drop every row, keeping the series identity and tolerances."""
        self.rows.clear()

    # ---- views (all return copies) ----------------------------------------
    def included_rows(self) -> List[RateRow]:
        """Rows the user has not switched off, in table order."""
        return [r for r in self.rows if r.include]

    def fitted_rows(self) -> List[RateRow]:
        """The rows that will be fitted, as a COPY sorted by |rate| ascending.

        Ascending |q| is plot order for every model here (x is ln q or 1/T,
        both monotone in q), so the residual vector a user reads off the
        result lines up with the points left-to-right on the figure.
        """
        return sorted((r for r in self.rows if r.fittable),
                      key=lambda r: abs(r.q_C_per_min))

    def failed_rows(self) -> List[RateRow]:
        """Rows whose measurement was attempted and did not produce a number."""
        return [r for r in self.rows if r.value_source == "failed"]

    def distinct_rates(self, rows: Optional[Sequence[RateRow]] = None) -> List[float]:
        """|rates| present, clustered by `rate_tol_pct`, ascending.

        Defaults to ALL rows with a finite rate, not the fitted ones: a file
        that produced four heating ramps at three distinct rates has three
        distinct rates the moment it is loaded, before any value has been
        computed. `check_series` passes `fitted_rows()` explicitly wherever a
        check is about the FIT rather than about the series.
        """
        pool = self.rows if rows is None else rows
        return cluster_rates((r.q_C_per_min for r in pool),
                             self.tolerances.rate_tol_pct)

    def rate_span_ratio(self, rows: Optional[Sequence[RateRow]] = None) -> float:
        """q_max / q_min over the distinct rates; nan below two of them."""
        rates = self.distinct_rates(rows)
        if len(rates) < 2 or rates[0] <= 0.0:
            return NAN
        return float(rates[-1] / rates[0])

    def specimen_keys(self, rows: Optional[Sequence[RateRow]] = None) -> List[str]:
        """Distinct non-empty specimen keys, in first-seen order.

        A blank key means UNKNOWN and is excluded, never treated as a shared
        identity: three hand-typed literature values with no specimen key are
        not evidence that they came from one specimen, and reporting
        `single_specimen_cycles` on them would be an alarm about an unknown
        fact.
        """
        pool = self.rows if rows is None else rows
        out: List[str] = []
        for r in pool:
            key = (r.specimen_key or "").strip()
            if key and key not in out:
                out.append(key)
        return out

    def points(self, *, require_T0: bool = False) -> List[RatePoint]:
        """RatePoints for the fitted rows, in plot order."""
        pts = [r.to_point() for r in self.fitted_rows()]
        if require_T0:
            pts = [p for p in pts if np.isfinite(p.T0_C)]
        return pts

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Raw dict for `core.project_io`, which must never import `dta.*`."""
        return {
            "schema_version": SERIES_SCHEMA_VERSION,
            "series_id": self.series_id,
            "name": self.name,
            "quantity_kind": self.quantity_kind,
            "created_at": self.created_at,
            "notes": self.notes,
            "tolerances": _tolerances_dict(self.tolerances),
            "rows": [r.to_dict() for r in self.rows],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RateSeries":
        """Inverse of `to_dict`; unknown keys ignored, missing keys defaulted."""
        d = dict(d or {})
        tol = SeriesTolerances(**{k: _f(v) for k, v in dict(d.get("tolerances") or {}).items()
                                  if k in _tolerances_dict(SeriesTolerances())})
        series = cls(
            series_id=str(d.get("series_id") or uuid.uuid4().hex),
            name=str(d.get("name") or "Rate series"),
            quantity_kind=str(d.get("quantity_kind") or "Tg"),
            rows=[RateRow.from_dict(r) for r in (d.get("rows") or [])],
            tolerances=tol,
            notes=str(d.get("notes") or ""),
            created_at=str(d.get("created_at") or ""),
            meta=dict(d.get("meta") or {}),
        )
        return series


def _tolerances_dict(tol: SeriesTolerances) -> Dict[str, float]:
    """Field-name -> value for a frozen dataclass, without `dataclasses.asdict`
    (which deep-copies and would choke on a future non-dataclass field)."""
    return {f: getattr(tol, f) for f in tol.__dataclass_fields__}


# =============================================================================
# 4) Row builders
# =============================================================================

def specimen_key_from_header(header: Mapping[str, Any], path: str = "") -> str:
    """'<Sample or stem>|<Size mg>|<Run date>' — the identity of one PHYSICAL
    specimen.

    Deliberately NOT the file path. Two exports of one run (a re-export, a
    trimmed copy, the same data reached through the library and through the
    file dialog) are ONE specimen, and `single_specimen_cycles` /
    `mixed_design` only mean anything if the key says so. Conversely two
    different specimens of the same composition measured on the same day get
    different keys, because the mass differs.

    Never raises: a header missing every field yields the file stem alone.
    """
    header = dict(header or {})

    def _first(*keys: str) -> str:
        """First non-empty header value among `keys`, case-insensitively."""
        for key in keys:
            for have in header:
                if str(have).strip().lower() == key.lower():
                    val = str(header[have]).strip()
                    if val:
                        return val
        return ""

    sample = _first("Sample") or _stem(path)
    size_raw = _first("Size")
    size = ""
    if size_raw:
        # TA writes "Size\t97.8930\tmg", which the header parser joins back
        # into "97.8930\tmg"; take the first numeric token so the key is the
        # same whether the unit rode along or not.
        token = size_raw.replace("\t", " ").split()
        if token:
            v = _f(token[0])
            size = f"{v:.4f}" if np.isfinite(v) else token[0]
    date = _first("Date", "RunDate")
    return "|".join((sample, size, date))


def rate_from_filename(filename: str, template: str) -> Optional[float]:
    """Jana-style '?'-wildcard template -> the rate encoded in a file name.

    Delegates to `xrd.htxrd_science.ramp_value_from_template` rather than
    re-implementing it: that function already handles the Windows-path
    basename rule, case insensitivity and the comma decimal separator, and a
    second copy of a filename parser is a second place for them to drift.
    Imported lazily so this module stays importable without the XRD package.
    Returns None (never raises) when the name does not fit the template.
    """
    if not filename or not template:
        return None
    try:
        from xrd.htxrd_science import ramp_value_from_template
    except Exception:
        return None
    try:
        return ramp_value_from_template(str(filename), str(template))
    except Exception:
        return None


def _seg_fields(segment: "ds.Segment", *, run: Optional["ds.SegmentedRun"] = None,
                match: Optional["ds.StepMatch"] = None) -> Dict[str, Any]:
    """THE ONLY place `dta_segments` attribute names appear (MERGED_DECISIONS
    M3). If that module renames a field, this function is the whole edit.

    The three mappings that are easy to get wrong, and why:

      * `q_measured_C_per_min` <- `Segment.rate_mode`, falling back to
        `rate_mean`. `rate_mode` is the MODE-refined measured rate: on the
        real run's first cooling ramp the mean is -14.155 and the median
        -18.006 against a programmed -20, and only the mode recovers -19.960,
        because the decaying passive tail wrecks the mean and its length
        wrecks the median.
      * `q_programmed_C_per_min` <- `Segment.rate_nominal_C_min`, which is
        NaN unless a `ProgramAlignment` was attached. It must stay NaN in
        that case: filling it from a measured estimator would put the wrong
        number on every Arrhenius x-axis AND permanently silence
        `rate_mismatch` and `setpoint_not_reached`, because they would then
        be comparing a value with itself.
      * `row_start` / `row_stop` <- `SegmentedRun.row_indices(seg)[0]` and
        `[-1] + 1`. `Segment.row_start` indexes the CLEANED arrays and must
        NEVER slice a DataFrame; the two coincide on every file shipped here,
        which is exactly why the distinction has to be enforced by code.

    `setpoint_reached` is None (unknown) when `target_gap_C` is NaN, False
    when the furnace ended more than 5 °C short of the programmed target.
    None and False are different claims and the UI must be able to tell them
    apart.

    Scope of the "only place" claim, stated precisely so the guard test can
    prove it: this is the only place the CONFUSABLE names appear —
    `rate_mode`, `rate_mean`, `rate_nominal_C_min`, `programmed_rate`,
    `target_gap_C` and the row indexing. Elsewhere this module compares
    `Segment.kind` against `dta_segments.KIND_*` and reads `T_max` / `T_start`
    while walking the run in time order; those are public constants and
    self-describing scalars, not names a rename could silently invert.
    """
    gap = _f(getattr(segment, "target_gap_C", NAN))
    if not np.isfinite(gap) and match is not None:
        gap = _f(getattr(match, "target_gap_C", NAN))
    nominal = _f(getattr(segment, "rate_nominal_C_min", NAN))
    if not np.isfinite(nominal) and match is not None:
        nominal = _f(getattr(match, "programmed_rate", NAN))

    measured = _f(getattr(segment, "rate_mode", NAN))
    if not np.isfinite(measured):
        measured = _f(getattr(segment, "rate_mean", NAN))

    if run is not None:
        rows = run.row_indices(segment)
        row_start = int(rows[0]) if rows.size else -1
        row_stop = int(rows[-1]) + 1 if rows.size else -1
    else:
        row_start = row_stop = -1

    return {
        "segment_index": int(getattr(segment, "index", -1)),
        "segment_kind": str(getattr(segment, "kind", "")),
        "role_in_cycle": str(getattr(segment, "role", "")),
        "cycle_index": int(getattr(segment, "cycle_index", 0)),
        "row_start": row_start,
        "row_stop": row_stop,
        "q_measured_C_per_min": measured,
        "q_programmed_C_per_min": nominal,
        "setpoint_reached": None if not np.isfinite(gap) else bool(gap <= 5.0),
        "is_partial": bool(getattr(segment, "is_partial", False)),
        "T0_C": _f(getattr(segment, "T_start", NAN)),
    }


def _kind_ordinal_labels(segments: Sequence["ds.Segment"]) -> Dict[int, str]:
    """{segment.index: 'heat 2 (cycle 2)'} — the short human name of a segment.

    Numbered within its own KIND, in time order, because "heat 2" is what a
    user says and `Segment.index` (which counts holds too) would call the
    second heating ramp #4.
    """
    counts: Dict[str, int] = {}
    out: Dict[int, str] = {}
    for seg in segments:
        counts[seg.kind] = counts.get(seg.kind, 0) + 1
        label = f"{seg.kind} {counts[seg.kind]}"
        if getattr(seg, "cycle_index", 0):
            label += f" (cycle {int(seg.cycle_index)})"
        out[int(seg.index)] = label
    return out


def _pick_columns(df, signal: str = "") -> Dict[str, str]:
    """{'time','temp','mass','signal'} -> column names of a TA SDT export.

    Matched by name prefix, not by position, because the Sig order differs
    between instrument configurations; anything not found is absent from the
    dict and its consumer degrades (no mass column -> `cum_mass_loss_pct`
    stays NaN, which is the honest answer, not 0).
    """
    cols = [str(c) for c in getattr(df, "columns", [])]
    low = {c: c.lower() for c in cols}
    out: Dict[str, str] = {}
    for c in cols:
        name = low[c]
        if "time" not in out and name.startswith("time"):
            out["time"] = c
        elif "temp" not in out and name.startswith("temperature (") and "difference" not in name:
            out["temp"] = c
        elif "mass" not in out and (name.startswith("weight") or name.startswith("mass (")):
            out["mass"] = c
    if signal and signal in cols:
        out["signal"] = signal
    else:
        for c in cols:
            name = low[c]
            if name.startswith("temperature difference (") and "v)" not in name:
                out["signal"] = c
                break
        else:
            for c in cols:
                if low[c].startswith("heat flow"):
                    out["signal"] = c
                    break
    if "time" not in out and cols:
        out["time"] = cols[0]
    if "temp" not in out and len(cols) > 1:
        out["temp"] = cols[1]
    return out


def _column(df, name: Optional[str]) -> Optional[np.ndarray]:
    """One dataframe column as a float array, or None when absent/unparseable."""
    if not name:
        return None
    try:
        import pandas as pd
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    except Exception:
        try:
            return np.asarray(df[name], dtype=float)
        except Exception:
            return None


def rows_from_segments(run: "ds.SegmentedRun", df, *, header: Optional[Mapping[str, Any]] = None,
                       source_kind: str = "segment", file: str = "", record_id: str = "",
                       sample: str = "", specimen_key: str = "",
                       alignment: Optional["ds.ProgramAlignment"] = None,
                       roles: Sequence[str] = (ds.ROLE_HEAT,),
                       quantity_kind: str = "Tg", method: str = "parallel",
                       signal: str = "", window_C: Tuple[float, float] = (350.0, 700.0),
                       measure: Optional[Callable[..., Tuple[float, Dict[str, float]]]] = None,
                       filename_template: str = "",
                       settings: Optional[Mapping[str, Any]] = None,
                       tolerances: Optional["SeriesTolerances"] = None,
                       rates_were_asked_for: Optional[bool] = None,
                       ) -> Tuple[List[RateRow], List[SeriesWarning]]:
    """One `RateRow` per segment whose kind or role is in `roles`.

    Heating only by default. A cooling segment is ADDED AND FLAGGED when
    asked for (`roles=("cool",)`), never silently dropped and never silently
    kept: `cooling_row` travels with it.

    The whole run is walked in TIME order even when only a few segments are
    wanted, because `prior_heats`, `prior_max_T_C` and `cum_mass_loss_pct`
    are properties of everything that happened BEFORE a segment, not of the
    segment. That walk is what makes heat 1 and heat 3 of one specimen
    distinguishable as two different materials.

    `measure` is an optional callable `(x, y, window, row) -> (value_C,
    extras)` (extra arguments are passed only if it accepts them). Its
    default of None leaves `value_C` NaN and `value_source` "—": adding rows
    must not compute them, because the Tg-tab settings that decide the number
    are stale by construction at add time, and a compute-on-add would draw a
    healthy-looking line through three filled rows and one blank before the
    user has inspected a single baseline. A `measure` that RAISES produces a
    row with `value_source="failed"` and the verbatim message — never an
    exception, never a row that vanishes from the table.

    Note the first parameter is a `SegmentedRun`, not the bare segment list
    SPEC_rate_series §4.3 wrote: only the run carries `keep_index`, and
    without it there is no legitimate way to turn a segment into DataFrame
    rows (MERGED_DECISIONS M3).

    Returns `(rows, warnings)`. Never raises on data.
    """
    warnings: List[SeriesWarning] = []
    rows: List[RateRow] = []
    if run is None:
        return rows, warnings

    segments = list(run.segments)
    labels = _kind_ordinal_labels(segments)
    align = alignment if alignment is not None else getattr(run, "alignment", None)

    cols = _pick_columns(df, signal) if df is not None else {}
    clock_warn = _clock_warning(cols.get("time"), file)
    mass = _column(df, cols.get("mass")) if df is not None else None
    y_values = _column(df, cols.get("signal")) if df is not None else None
    signal_name = cols.get("signal", "") if df is not None else signal

    # The run and the dataframe must describe the SAME record. They can stop
    # doing so without anyone noticing, because the Qt layer caches one
    # `SegmentedRun` per record id and a re-import or a crop replaces the
    # dataframe under it; `run.row_indices()` then returns positions past the
    # end of the new columns. `prepare_segment_xy` already refuses that case
    # with a named diagnostic, but the mass walk below indexes the raw column
    # directly and used to raise a bare IndexError out of a function whose
    # docstring promises it never raises on data. Say so once, loudly, and
    # carry on with the per-row failures the mismatch will produce.
    n_df = -1
    if df is not None:
        try:
            n_df = int(len(df))
        except (TypeError, ValueError):
            n_df = -1
    scope_ok = not (n_df >= 0 and int(run.n_rows) != n_df)
    if not scope_ok:
        warnings.append(SeriesWarning(
            code="segment_scope", severity="critical",
            message=(f"The stored segmentation describes {int(run.n_rows)} rows but "
                     f"the data now has {n_df}. The rows are kept and every value "
                     f"they could not read is marked failed; re-detect the segments "
                     f"for this record to clear it."),
            detail={"n_rows_segmented": int(run.n_rows), "n_rows_now": n_df}))

    hdr = _header_mapping(header)
    if not specimen_key:
        specimen_key = specimen_key_from_header(hdr, file)
    if not sample:
        sample = str(hdr.get("Sample") or "").strip() or _stem(file)

    m_first = NAN
    if mass is not None and mass.size:
        finite = mass[np.isfinite(mass)]
        if finite.size:
            m_first = float(finite[0])
    sample_mass = _f(_header_size_mg(hdr), m_first)

    q_from_name = rate_from_filename(file, filename_template) if filename_template else None

    heat_segments = [s for s in segments if s.kind == ds.KIND_HEAT]
    n_heats = len(heat_segments)

    wanted = {str(r).strip().lower() for r in (roles or ())}
    # `ROLE_HOLD` is exported in `SEGMENT_ROLES`, so a scope combo will offer
    # it, but `assign_roles` only ever emits it for a hold that no ramp
    # preceded — which on a real run never happens, since every hold follows
    # something. Asking for "hold" therefore matched nothing and produced a
    # `no_heating_segment` placeholder on a file containing four of them.
    # Treat it as the umbrella word a user means by it.
    if ds.ROLE_HOLD in wanted:
        wanted |= {ds.ROLE_HOLD_HOT, ds.ROLE_HOLD_COLD, ds.KIND_ISOTHERMAL}

    # --- the time-order walk that fills thermal history -------------------
    heats_done = 0
    max_T_seen = NAN
    last_cum = NAN
    for seg in segments:
        prior_heats = heats_done
        prior_max = max_T_seen

        cum_here = last_cum
        if seg.kind == ds.KIND_HEAT and mass is not None and np.isfinite(m_first) and m_first > 0:
            idx = run.row_indices(seg)
            if idx.size and int(idx[-1]) < int(mass.size):
                m_end = _f(mass[int(idx[-1])])
                if np.isfinite(m_end):
                    cum_here = 100.0 * (m_first - m_end) / m_first
                    last_cum = cum_here

        if (seg.kind in wanted) or (str(seg.role).lower() in wanted):
            row = _row_for_segment(
                seg, run=run, df=df, align=align, labels=labels,
                source_kind=source_kind, file=file, record_id=record_id,
                sample=sample, specimen_key=specimen_key,
                quantity_kind=quantity_kind, method=method,
                signal_name=signal_name, window_C=window_C,
                prior_heats=prior_heats, prior_max=prior_max,
                cum_mass_loss=cum_here, sample_mass=sample_mass,
                n_heats=n_heats, q_from_name=q_from_name,
                settings=settings)
            if measure is not None and y_values is not None:
                _apply_measure(row, run, seg, y_values, window_C, measure)
            rows.append(row)

        if seg.kind == ds.KIND_HEAT:
            heats_done += 1
        seg_max = _f(getattr(seg, "T_max", NAN))
        if np.isfinite(seg_max):
            max_T_seen = seg_max if not np.isfinite(max_T_seen) else max(max_T_seen, seg_max)

    name = _stem(file) or sample or "this record"
    if not rows:
        # Nothing in this record matched the requested roles. The honest
        # answer is a row that says so, not silence: a user who added five
        # files and sees four rows has to work out which one is missing,
        # whereas a fifth row with an empty rate names it and can be
        # completed by typing.
        placeholder = RateRow(source_kind=source_kind, sample=sample or name,
                              specimen_key=specimen_key, file=str(file or ""),
                              record_id=str(record_id or ""),
                              quantity_kind=quantity_kind, method=method,
                              signal=signal_name,
                              window_C=(_f(window_C[0]), _f(window_C[1])),
                              sample_mass_mg=_f(sample_mass),
                              n_heating_segments_in_source=int(n_heats))
        if q_from_name is not None:
            placeholder.q_filename_C_per_min = _f(q_from_name)
        if settings:
            placeholder.settings = dict(settings)
        rows.append(placeholder)
    # Default by source: picking the segments of the run on screen IS asking for
    # its several rates, while a file that turns out to hold several is a guess
    # the engine will not make. A caller may state it outright either way.
    asked = (source_kind == "segment" if rates_were_asked_for is None
             else bool(rates_were_asked_for))
    warnings.extend(_source_warnings(rows, name, tolerances,
                                     rates_were_asked_for=asked))
    if clock_warn is not None:
        warnings.append(replace_warning_rows(clock_warn, rows))
    return rows, warnings


def replace_warning_rows(warn: SeriesWarning,
                         rows: Sequence[RateRow]) -> SeriesWarning:
    """A copy of `warn` naming these rows. `SeriesWarning` is frozen-by-habit
    rather than by decorator, so this is the one place a row list is attached
    after the fact -- the Qt banner filters add-time warnings by row liveness
    and a warning with no uids would outlive the rows it describes."""
    return SeriesWarning(code=warn.code, severity=warn.severity,
                         message=warn.message,
                         row_uids=tuple(r.uid for r in rows),
                         detail=dict(warn.detail))


def _header_mapping(header: Any) -> Dict[str, Any]:
    """`header` as a plain dict, whatever it actually is. Never raises.

    `dict(header or {})` raised `ValueError` on a string and `TypeError` on an
    int, inside a function whose docstring promises "Never raises on data".
    That is reachable: `io_universal.parse_saxs_edf_ascii` stores
    `meta["raw_header"]` as a single joined STRING, and `rows_from_record`
    forwards `meta["raw_header"]` unchecked, so a scripted batch over a mixed
    library crashed where the GUI only showed a failed row with
    "dictionary update sequence element #0 has length 1; 2 is required" as its
    stated reason. A header that is not a mapping carries no Sample and no
    Size, which is exactly what an empty dict means.
    """
    if header is None:
        return {}
    if isinstance(header, Mapping):
        return dict(header)
    try:
        return dict(header)                      # a TAHeader, or pairs
    except (TypeError, ValueError):
        return {}


def _clock_warning(time_col: Any, name: str = "") -> Optional[SeriesWarning]:
    """`clock_unit` when the clock column was not already in minutes.

    Informational and never a refusal: the conversion has already been applied
    (`_segment_dataframe`), so the rate on the row is right. The sentence
    exists because a 60x rate correction the user cannot see is indistinguish-
    able from a 60x rate error they cannot see.
    """
    scale, unit = ds.clock_scale_to_min(time_col)
    if scale == 1.0:
        return None
    stem = _stem(str(name or "")) or "This source"
    return SeriesWarning(
        code="clock_unit", severity="warn",
        message=(f"{stem} writes its clock as \u201c{time_col}\u201d, not in minutes. "
                 f"It was converted (x {scale:g}) before the rate was measured, so "
                 f"the rate shown is per MINUTE. Check it against the method: read "
                 f"as minutes it would have been {1.0 / scale:g}x wrong."),
        detail={"column": str(time_col), "scale": float(scale), "unit": unit})


def _header_size_mg(header: Mapping[str, Any]) -> float:
    """The `Size` header value in mg, NaN when absent or unparseable."""
    for key in header:
        if str(key).strip().lower() == "size":
            token = str(header[key]).replace("\t", " ").split()
            if token:
                return _f(token[0])
    return NAN


def _row_for_segment(seg, *, run, df, align, labels, source_kind, file, record_id,
                     sample, specimen_key, quantity_kind, method, signal_name,
                     window_C, prior_heats, prior_max, cum_mass_loss, sample_mass,
                     n_heats, q_from_name, settings) -> RateRow:
    """Build one row from one segment. All dta_segments reads go through
    `_seg_fields`; everything else here is provenance this module owns."""
    match = align.match_for(int(seg.index)) if align is not None else None
    fields = _seg_fields(seg, run=run, match=match)
    row = RateRow(source_kind=source_kind, sample=sample, specimen_key=specimen_key,
                  file=str(file or ""), record_id=str(record_id or ""),
                  quantity_kind=quantity_kind, method=method,
                  signal=signal_name, window_C=(_f(window_C[0]), _f(window_C[1])))
    for key, value in fields.items():
        setattr(row, key, value)
    row.segment_label = labels.get(int(seg.index), f"segment {seg.index}")
    row.n_heating_segments_in_source = int(n_heats) if n_heats else 1
    row.prior_heats = int(prior_heats)
    row.prior_max_T_C = _f(prior_max)
    row.cum_mass_loss_pct = _f(cum_mass_loss)
    row.sample_mass_mg = _f(sample_mass)
    row.state = "virgin" if prior_heats == 0 else f"after-cycle-{int(prior_heats)}"
    if q_from_name is not None:
        row.q_filename_C_per_min = _f(q_from_name)
    if settings:
        row.settings = dict(settings)
    if getattr(seg, "note", ""):
        row.note = str(seg.note)
    return row


def _apply_measure(row: RateRow, run, seg, y_values, window_C, measure) -> None:
    """Run the caller's measurement on ONE segment's prepared arrays.

    The arrays come from `SegmentedRun.xy()` and from nowhere else
    (MERGED_DECISIONS M4): that is the single function which slices, masks,
    sorts per segment and collapses duplicates. Any failure — a bad window, a
    baseline with too few points, a segment whose temperature is not a usable
    abscissa — is stored on the row as `value_source="failed"` plus the
    verbatim message, because a row that fails loudly is analysable and a row
    that silently disappears from a Lasocka fit is not.
    """
    try:
        prep = run.xy(seg, y_values, x_kind=ds.X_TEMPERATURE)
    except Exception as exc:
        row.value_source = "failed"
        row.error = str(exc)
        return
    try:
        out = _call_measure(measure, prep.x, prep.y, window_C, row)
    except Exception as exc:
        row.value_C = NAN
        row.value_source = "failed"
        row.error = str(exc)
        return
    value, extras = _unpack_measure(out)
    if value is None or not np.isfinite(_f(value)):
        row.value_C = NAN
        row.value_source = "failed"
        if not row.error:
            row.error = "The measurement returned no value."
    else:
        row.value_C = _f(value)
        row.value_source = "computed"
        row.error = ""
    if extras:
        row.extras = {str(k): _f(v) for k, v in dict(extras).items()}


def _positional_arity(measure) -> Optional[int]:
    """How many positional arguments `measure` accepts, None when unreadable.

    `*args` reports 4 (enough), because "enough" is the only question here.
    A callable whose signature cannot be introspected — most C functions —
    returns None and the caller falls back to probing.
    """
    import inspect
    try:
        sig = inspect.signature(measure)
    except (TypeError, ValueError):
        return None
    count = 0
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return 4
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
            count += 1
    return count


def _call_measure(measure, x, y, window, row):
    """Call `measure` with as many arguments as it accepts.

    A three-argument measurer `(x, y, window)` is the documented contract; a
    four-argument one also receives the row, which is what a caller needs to
    look at `segment_kind` or `prior_heats` before choosing a window.

    The arity is decided BEFORE the call wherever a signature can be read,
    because a bare `try: measure(x, y, window, row) except TypeError:` cannot
    tell a wrong number of arguments from a TypeError raised inside the
    measurer's own body. Those two have opposite correct answers: the first
    means "retry with three", the second means "this is the diagnostic the
    user has to see". Conflating them replaces a real fault
    ("unsupported operand type(s) for +: 'float' and 'NoneType'") with a
    fictional one ("missing 1 required positional argument: 'row'") in the
    row's `error` field, which is the one place BM_FINDINGS §8 requires a
    failed row to name its cause — and it runs a `*args` measurer's body
    twice, side effects and all.

    When the signature is unreadable the probe is kept, narrowed by the
    traceback: a binding failure is raised by the interpreter at this call
    site and carries no inner frame, while a TypeError from inside the
    measurer carries the measurer's frame in `tb_next` and is re-raised.
    """
    n_args = _positional_arity(measure)
    if n_args is not None:
        return measure(x, y, window, row) if n_args >= 4 else measure(x, y, window)
    try:
        return measure(x, y, window, row)
    except TypeError as exc:
        if exc.__traceback__ is not None and exc.__traceback__.tb_next is not None:
            raise
        return measure(x, y, window)


def _unpack_measure(out) -> Tuple[Optional[float], Dict[str, float]]:
    """Accept `value`, `(value, extras)` or `(value,)` from a measurer."""
    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], dict(out[1] or {})
        if len(out) == 1:
            return out[0], {}
        return None, {}
    return out, {}


def cluster_rates(values: Iterable[float], rate_tol_pct: float = 2.0) -> List[float]:
    """Distinct |rates|, ascending, merging neighbours within `rate_tol_pct`.

    ONE clustering rule for the whole module. It used to be two: the builder
    binned with `round(abs(q), 2)` while `RateSeries.distinct_rates` used the
    tolerance, and a real 20 °C/min ramp measured four times comes back as
    19.99 / 20.03 / 20.10 — three bins to the builder, one rate to the series.
    The disagreement was not cosmetic: the builder switched every row off as
    "several different rates" while `single_rate` (the check that explains
    there is no lever arm) never fired, because the rows it looks at were off.
    A replicate-cycles run at one programmed rate is an ordinary experiment
    and must load with its rows ON.
    """
    vals = sorted(abs(_f(v)) for v in values
                  if np.isfinite(_f(v)) and _f(v) != 0.0)
    tol = max(float(rate_tol_pct), 0.0) / 100.0
    out: List[float] = []
    for v in vals:
        if out and abs(v - out[-1]) <= tol * max(abs(out[-1]), 1e-12):
            continue
        out.append(float(v))
    return out


def _source_warnings(rows: Sequence[RateRow], name: str,
                     tolerances: Optional[SeriesTolerances] = None,
                     *, rates_were_asked_for: bool = False
                     ) -> List[SeriesWarning]:
    """`no_heating_segment` / `ambiguous_file_rate` for ONE source, at add time.

    The auto-detection rule, in one place and identical for all four sources:

      | matched segments        | result                                     |
      |-------------------------|--------------------------------------------|
      | 0                       | one placeholder row, no rate, CRITICAL     |
      | 1                       | one row, rate filled in, nothing to type   |
      | >1, rates the same      | N rows, all on — one rate measured N times |
      | >1, rates different     | see `rates_were_asked_for`                 |

    `rates_were_asked_for` decides the last line, and getting it wrong breaks
    the feature this module exists for. When a separate FILE turns out to hold
    several ramps at different rates, "what rate is this file?" is a genuine
    guess, so every row is added switched OFF with an INFO naming them: the
    engine declines to guess, it does not decline to load, and switching rows
    off rather than dropping them keeps the choice visible and one click away.

    But when the user pressed "Add segments from current file" on a cyclic run,
    the several different rates ARE what they asked for -- that program was
    written to measure one specimen at 20, 20, 15 and 10 °C/min precisely so the
    rates could be compared. Switching those rows off left a series of four good
    rows with nothing fitted and `too_few_points` on screen, i.e. the Lasocka and
    Arrhenius feature produced nothing on the exact dataset it was built for.
    So an explicit segment pick arrives switched ON, and the real objection --
    that successive cycles of one specimen confound rate with thermal history --
    is raised by `check_series` as `single_specimen_cycles` and
    `thermal_history`, which describe the problem without suppressing the fit.

    "The same rate" means the same to `cluster_rates` as it does to
    `RateSeries.distinct_rates`, i.e. within `rate_tol_pct`. It has to: four
    replicate cycles of a 20 °C/min ramp measure 19.99-20.10, and binning
    them at 0.01 °C/min made the third row of that table unreachable on real
    data — every replicate run loaded with all its rows switched off.

    Both conditions are re-derived by `check_series` from the stored row
    state, so a series reopened from a project or a CSV carries them too;
    emitting them here as well is what lets the UI show the problem the
    instant a file is added rather than after the first fit.
    """
    out: List[SeriesWarning] = []
    tol = tolerances or SeriesTolerances()
    measured = [r.q_C_per_min for r in rows if np.isfinite(r.q_C_per_min)]
    # Clustered on the raw values, ROUNDED only for display and for the
    # exported `detail`: a message reading "19.9544 °C/min" is noise, and the
    # stored rate the fit uses is on the row, not in this warning.
    rates = [round(v, 2) for v in cluster_rates(measured, tol.rate_tol_pct)]
    if not measured:
        out.append(SeriesWarning(
            code="no_heating_segment", severity="critical",
            message=(f"{name} contains no ramp of the kind asked for, so the rate "
                     f"is unknown. The row is kept — type a rate into the Rate "
                     f"column, or pick a segment by hand."),
            row_uids=tuple(r.uid for r in rows),
            detail={"source": name}))
        return out
    if len(rows) > 1 and len(rates) > 1:
        listed = ", ".join(f"{v:g}" for v in rates)
        kinds = sorted({r.segment_kind for r in rows if r.segment_kind})
        word = "heating ramps" if kinds == [ds.KIND_HEAT] else "ramps"
        if rates_were_asked_for:
            out.append(SeriesWarning(
                code="multi_rate_source", severity="info",
                message=(f"{name} holds {len(rows)} {word} at {listed} °C/min, "
                         f"added switched on — that spread of rates is what a rate "
                         f"series is for. They are successive cycles of ONE "
                         f"specimen, so read the thermal-history warning before "
                         f"quoting a slope."),
                row_uids=tuple(r.uid for r in rows),
                detail={"n_segments": len(rows), "rates": list(rates)}))
        else:
            out.append(SeriesWarning(
                code="ambiguous_file_rate", severity="info",
                message=(f"{name} holds {len(rows)} {word} at {listed} °C/min "
                         f"— pick which to include. All of them were added, "
                         f"switched off, so nothing was guessed for you."),
                row_uids=tuple(r.uid for r in rows),
                detail={"n_segments": len(rows), "rates": list(rates)}))
            for r in rows:
                r.include = False
    return out


def rows_from_file(path, *, segmenter: Optional[Callable[..., "ds.SegmentedRun"]] = None,
                   program=None, use_program: bool = True,
                   **kwargs) -> Tuple[List[RateRow], List[SeriesWarning]]:
    """Source (b): one separate file, usually at one rate.

    Deliberately NOT separate logic. The file is parsed with
    `core.io_universal.parse_ta_sdt_txt`, its OrgMethod program is read,
    `dta_segments.segment_run` finds the segments, and `rows_from_segments`
    does the rest — so a clean single-ramp file yields exactly ONE row with
    its rate already filled in and nothing to type, and a file that turns out
    to hold several heating ramps yields several rows, because that IS the
    cyclic case and the only honest answer to "what rate is this file?" is
    "it has more than one".

    A parse or segmentation failure yields ONE row with `value_source =
    "failed"`, the verbatim message in `error` and `include=False` — never an
    exception (a batch of twenty files must not be lost to one bad file), and
    never a silently skipped file (a series that is quietly one row short is
    worse than one that says so).
    """
    from pathlib import Path

    path_str = str(path)
    try:
        import core.io_universal as iou
        header, _signals, df = iou.parse_ta_sdt_txt(Path(path_str))
    except Exception as exc:
        return _failed_source_row(path_str, f"{exc}", source_kind="file")

    prog = program
    if prog is None and use_program:
        try:
            from dta import dta_program as dp
            prog = dp.program_from_header(header, header_lines=getattr(header, "lines", None))
            if not prog:
                prog = None
        except Exception:
            prog = None

    try:
        run = _segment_dataframe(df, program=prog, segmenter=segmenter)
    except Exception as exc:
        return _failed_source_row(path_str, f"{exc}", source_kind="file")

    kwargs.setdefault("source_kind", "file")
    kwargs.setdefault("file", path_str)
    kwargs.setdefault("header", header)
    return rows_from_segments(run, df, **kwargs)


def rows_from_record(rec: Mapping[str, Any], *, segmenter=None, program=None,
                     use_program: bool = True, **kwargs
                     ) -> Tuple[List[RateRow], List[SeriesWarning]]:
    """Source (c): a record already imported into the library.

    Uses the in-memory dataframe when the record carries one, so a file the
    user already opened is not read from disk a second time (and still works
    when the network share it came from is offline). Falls back to the path.
    A record with neither yields one failed row, the same shape every other
    source failure takes.
    """
    rec = dict(rec or {})
    df = rec.get("df")
    path = str(rec.get("path") or "")
    record_id = str(rec.get("id") or "")
    title = str(rec.get("title") or "")
    meta = dict(rec.get("meta") or {})
    header = meta.get("raw_header") or {}

    if df is None:
        if not path:
            return _failed_source_row(title or "library record",
                                      "This library record carries no data and no path.",
                                      source_kind="library", record_id=record_id)
        kwargs.setdefault("record_id", record_id)
        kwargs.setdefault("sample", title)
        kwargs["source_kind"] = "library"
        return rows_from_file(path, segmenter=segmenter, program=program,
                              use_program=use_program, **kwargs)

    prog = program
    if prog is None and use_program and header:
        try:
            from dta import dta_program as dp
            # `meta["header_lines"]` is the ONLY surviving copy of the 19
            # OrgMethod lines once a record has been through a .prism save:
            # `core.project_io._json_safe` flattens the `TAHeader` subclass to
            # a plain dict, so `header.lines` is gone and the header's single
            # collapsed "OrgMethod" key is the string "1: Select gas 1".
            # Without it a reopened cyclic run re-derives a one-step program,
            # every `q_programmed_C_per_min` comes back NaN, every
            # `setpoint_reached` None, and `rate_mismatch` /
            # `setpoint_not_reached` are permanently silenced — the opposite
            # of MERGED_DECISIONS M6, which says a reopened project re-derives
            # its segments. `core.io_universal` writes the field for exactly
            # this; this is its consumer.
            prog = dp.program_from_header(
                header,
                header_lines=(getattr(header, "lines", None)
                              or meta.get("header_lines")))
            if not prog:
                prog = None
        except Exception:
            prog = None
    try:
        run = _segment_dataframe(df, program=prog, segmenter=segmenter)
    except Exception as exc:
        return _failed_source_row(title or path or "library record", f"{exc}",
                                  source_kind="library", record_id=record_id)

    kwargs.setdefault("source_kind", "library")
    kwargs.setdefault("file", path)
    kwargs.setdefault("record_id", record_id)
    kwargs.setdefault("header", header)
    kwargs.setdefault("sample", title)
    return rows_from_segments(run, df, **kwargs)


def _segment_dataframe(df, *, program=None, segmenter=None) -> "ds.SegmentedRun":
    """Segment a TA dataframe. `segmenter` overrides the default for tests.

    The clock is converted to MINUTES from the unit in its own column label
    (`ds.clock_scale_to_min`). Without that a `Time (s)` export segmented as
    though its clock were minutes and reported 0.333 C/min for a 20 C/min ramp
    -- a 60x error with no warning anywhere, which on a two-file mixed-unit
    series came out as a Lasocka B wrong by 4.9x and of the wrong sign.
    `_clock_warning` says so on the row.
    """
    cols = _pick_columns(df)
    t = _column(df, cols.get("time"))
    T = _column(df, cols.get("temp"))
    if t is None or T is None:
        raise ValueError("This file has no time and temperature columns to segment.")
    scale, _unit = ds.clock_scale_to_min(cols.get("time"))
    if scale != 1.0:
        t = t * scale
    if segmenter is not None:
        return segmenter(t, T, program=program)
    return ds.segment_run(t, T, program=program)


def _failed_source_row(name: str, message: str, *, source_kind: str,
                       record_id: str = "") -> Tuple[List[RateRow], List[SeriesWarning]]:
    """One row standing for a source that could not be read.

    The row exists on purpose: a user who added five files and sees four rows
    has to work out which one is missing, whereas a fifth row reading
    "failed: <message>" names it. `include=False` keeps it out of the fit
    without keeping it out of sight.
    """
    row = RateRow(source_kind=source_kind, sample=_stem(name) or name,
                  file=name if source_kind != "manual" else "",
                  record_id=record_id, value_source="failed", error=message,
                  include=False)
    warn = SeriesWarning(
        code="no_heating_segment", severity="critical",
        message=(f"{_stem(name) or name} could not be read, so it has no heating "
                 f"ramp and no rate: {message} The row is kept so the gap is "
                 f"visible; fix the file or type the numbers in by hand."),
        row_uids=(row.uid,), detail={"source": name, "error": message})
    return [row], [warn]


def manual_row(*, sample: str = "", q_C_per_min: float = NAN, value_C: float = NAN,
               quantity_kind: str = "Tg", specimen_key: str = "",
               state: str = "unknown", prior_heats: int = 0,
               cum_mass_loss_pct: float = NAN, window_C: Tuple[float, float] = (NAN, NAN),
               method: str = "manual", signal: str = "", note: str = "",
               error: str = "", T0_C: float = NAN, include: bool = True,
               ) -> RateRow:
    """Source (d): a row typed by hand.

    Literature values, a number from another program, a specimen measured on
    another instrument — and, just as importantly, what makes every fit,
    warning and export in this module testable with no data file at all.

    A value of NaN with a non-empty `error` is the shape of a FAILED
    measurement carried by hand, which is how the real four-row BM series is
    reproduced in tests without the 5 MB original.
    """
    row = RateRow(source_kind="manual", sample=sample, specimen_key=specimen_key,
                  quantity_kind=quantity_kind, method=method, signal=signal,
                  window_C=(_f(window_C[0]), _f(window_C[1])),
                  state=state, prior_heats=int(prior_heats),
                  cum_mass_loss_pct=_f(cum_mass_loss_pct),
                  q_manual_C_per_min=_f(q_C_per_min), T0_C=_f(T0_C),
                  note=note, error=error, include=include)
    if _finite(value_C):
        row.value_C = _f(value_C)
        row.value_source = "manual"
    elif error:
        row.value_source = "failed"
    return row


# =============================================================================
# 5) Checks — warnings only, never a gate
# =============================================================================

def _short(row: RateRow, index: int) -> str:
    """'row 3 (BM-0-cycles · heat 3 @ 15.0 °C/min)' for a warning message."""
    return f"row {index} ({row.label})"


def _join(labels: Sequence[str], limit: int = 4) -> str:
    """Comma-join with an honest '+N more' tail, so a 30-row series does not
    produce a warning nobody reads."""
    labels = list(labels)
    if len(labels) <= limit:
        return ", ".join(labels)
    return ", ".join(labels[:limit]) + f", +{len(labels) - limit} more"


def check_series(series: RateSeries, *, model: str = "",
                 verify_paths: bool = True) -> List[SeriesWarning]:
    """Every provenance and design objection to this series, as warnings.

    NEVER raises, NEVER filters a row, NEVER gates a fit. This is the
    function that carries the user's instruction "ok but still make this
    possible to process such data": cyclic data, confounded thermal history,
    too few rates, duplicated rates, mass loss and cooling-segment kinetics
    all produce warnings here and a number in `fit_series`.

    SCOPE RULE, decided once so no individual check re-decides it: every
    check evaluates the FITTED set, EXCEPT `no_value`, `no_rate`,
    `rows_not_fitted`, `excluded_rows`, `manual_values`, `missing_source_file`,
    `no_heating_segment` and `ambiguous_file_rate`, which evaluate ALL rows.

    That rule is why `duplicate_rate` does NOT fire on the real BM series:
    the second 20 °C/min row is the one whose baseline failed, so it is not
    in the fit, and claiming a duplicated rate would be false. Fix that row's
    window and `duplicate_rate` starts firing — correctly.

    Each check runs inside its own guard. A bug in one check therefore costs
    that one warning, never the whole list and never an exception in front of
    a user who is trying to look at their data.
    """
    out: List[SeriesWarning] = []
    if series is None:
        return out
    for check in _CHECKS:
        try:
            out.extend(check(series, model, verify_paths) or ())
        except Exception:
            # A failing check must not take the other twenty-eight with it,
            # and must never surface as a traceback over a data view.
            continue
    return sort_warnings(out)


# -- individual checks --------------------------------------------------------
# Every one has the signature (series, model, verify_paths) -> list, and every
# one is registered in _CHECKS at the bottom of this section.

def _chk_counts(series, model, verify_paths):
    """`too_few_points`, `rows_not_fitted`, `excluded_rows`, `manual_values`."""
    out = []
    rows = series.rows
    fitted = series.fitted_rows()
    n_tot, n_fit = len(rows), len(fitted)
    if n_fit < 2:
        out.append(SeriesWarning(
            code="too_few_points", severity="critical",
            message=(f"No fit yet — {n_fit} of {n_tot} row(s) has a usable rate and "
                     f"value (2 are needed). Press \"Compute all values\", or type "
                     f"values into the Value column."),
            detail={"n_fitted": n_fit, "n_total": n_tot}))
    not_fitted = [r for r in rows if not r.fittable]
    if not_fitted:
        excluded = [r for r in not_fitted if not r.include]
        no_val = [r for r in not_fitted if r.include and not _finite(r.value_C)]
        out.append(SeriesWarning(
            code="rows_not_fitted", severity="info",
            message=(f"{len(not_fitted)} of {n_tot} rows are not in the fit "
                     f"({len(excluded)} switched off, {len(no_val)} with no "
                     f"value). They stay in the table."),
            row_uids=tuple(r.uid for r in not_fitted),
            detail={"n_not_fitted": len(not_fitted), "n_excluded": len(excluded),
                    "n_no_value": len(no_val)}))
    if excluded_rows := [r for r in rows if not r.include]:
        # NOT "by hand": `_source_warnings` switches rows off itself whenever
        # one file yielded several distinct rates (SPEC_rate_series §7.5), so
        # on the flagship cyclic file all four rows arrive off before the user
        # has touched anything. Telling them they did it is a false statement
        # about their own actions and sends them looking for a click they
        # never made, so the sentence names both possible authors.
        #
        # It no longer names `ambiguous_file_rate` as THE explanation: the
        # segment door raises `multi_rate_source` instead, so on a cyclic file
        # this line used to point the reader at a code that was not on the
        # banner. The two emitters are named together or not at all.
        out.append(SeriesWarning(
            code="excluded_rows", severity="info",
            message=(f"{len(excluded_rows)} row(s) are switched off and greyed in the "
                     f"plot — by you, or automatically because their file holds "
                     f"several rates (see ambiguous_file_rate / multi_rate_source): "
                     f"{_join([_short(r, series.index_of(r.uid)) for r in excluded_rows])}."),
            row_uids=tuple(r.uid for r in excluded_rows),
            detail={"n_excluded": len(excluded_rows)}))
    if manual := [r for r in rows if r.value_source == "manual"]:
        out.append(SeriesWarning(
            code="manual_values", severity="info",
            message=(f"{len(manual)} value(s) typed by hand, not computed: "
                     f"{_join([_short(r, series.index_of(r.uid)) for r in manual])}."),
            row_uids=tuple(r.uid for r in manual),
            detail={"n_manual": len(manual)}))
    return out


def _chk_missing_values(series, model, verify_paths):
    """`no_value` and `no_rate` — evaluated over ALL rows, per the scope rule."""
    out = []
    n_tot = len(series.rows)
    n_fit = len(series.fitted_rows())
    blank = [r for r in series.rows if not _finite(r.value_C)]
    if blank:
        first_error = next((r.error for r in blank if r.error), "")
        tail = f" {first_error}" if first_error else ""
        out.append(SeriesWarning(
            code="no_value", severity="critical",
            message=(f"{len(blank)} row(s) have no value and are not in the fit: "
                     f"{_join([_short(r, series.index_of(r.uid)) for r in blank])}.{tail} "
                     f"Set a value by hand or change the window, or the fit stands on "
                     f"{n_fit} of {n_tot} rows."),
            row_uids=tuple(r.uid for r in blank),
            detail={"n_no_value": len(blank), "first_error": first_error}))
    # SPEC_rate_series' table says "non-finite or <= 0". Measured against the
    # rest of the table, "<= 0" is wrong: a cooling segment carries a
    # NEGATIVE rate, `cooling_row` already reports it, and the fit uses the
    # magnitude — firing `no_rate` on it too would contradict the cooling
    # rule and would fire on every cooling series this feature is meant to
    # accept. Only a rate that is absent or exactly zero (ln 0 is undefined)
    # stops a row being fittable.
    rateless = [r for r in series.rows
                if not np.isfinite(r.q_C_per_min) or r.q_C_per_min == 0.0]
    if rateless:
        out.append(SeriesWarning(
            code="no_rate", severity="critical",
            message=(f"{len(rateless)} row(s) have no heating rate: "
                     f"{_join([_short(r, series.index_of(r.uid)) for r in rateless])}. "
                     f"Type one into the Rate column, or pick a segment; the rows "
                     f"that do have one are fitted anyway."),
            row_uids=tuple(r.uid for r in rateless),
            detail={"n_no_rate": len(rateless)}))
    return out


def _chk_rate_design(series, model, verify_paths):
    """`single_rate`, `few_rates`, `duplicate_rate`, `narrow_rate_span`."""
    out = []
    fitted = series.fitted_rows()
    if len(fitted) < 2:
        return out
    rates = series.distinct_rates(fitted)
    tol = series.tolerances

    if len(rates) == 1:
        out.append(SeriesWarning(
            code="single_rate", severity="critical",
            message=(f"All {len(fitted)} fitted rows are at {rates[0]:g} °C/min — "
                     f"there is no rate lever arm, so there is no slope to measure. "
                     f"Add a row at a different rate."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"q": rates[0], "n_fitted": len(fitted)}))
    elif len(rates) == 2:
        out.append(SeriesWarning(
            code="few_rates", severity="warn",
            message=(f"Only 2 distinct rates ({rates[0]:g} and {rates[1]:g} °C/min) — "
                     f"the slope has no residual degrees of freedom, so R² and σ are "
                     f"not evidence about it. Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"rates": rates}))

    # duplicates, clustered the same way distinct_rates clusters
    groups: Dict[float, List[RateRow]] = {}
    for r in fitted:
        q = abs(r.q_C_per_min)
        for key in groups:
            if abs(q - key) <= (tol.rate_tol_pct / 100.0) * max(abs(key), 1e-12):
                groups[key].append(r)
                break
        else:
            groups[q] = [r]
    for q, members in sorted(groups.items()):
        if len(members) > 1:
            out.append(SeriesWarning(
                code="duplicate_rate", severity="warn",
                message=(f"Duplicated rate: {q:g} °C/min appears on {len(members)} rows "
                         f"({_join([_short(m, series.index_of(m.uid)) for m in members])}) "
                         f"— weight without leverage. {len(rates)} distinct rates are in "
                         f"the fit. Both rows are kept, unweighted. Fitted anyway."),
                row_uids=tuple(m.uid for m in members),
                detail={"q": float(q), "n_rows": len(members),
                        "rate_tol_pct": tol.rate_tol_pct}))

    ratio = series.rate_span_ratio(fitted)
    if np.isfinite(ratio) and ratio < tol.span_warn:
        severity = "critical" if ratio < tol.span_critical else "warn"
        out.append(SeriesWarning(
            code="narrow_rate_span", severity=severity,
            message=(f"Rate span is {ratio:.1f}× ({rates[0]:g}–{rates[-1]:g} °C/min) — "
                     f"well under one decade, so the lever arm is short and the slope "
                     f"is sensitive to every point; {tol.span_warn:g}× is the usual "
                     f"minimum. Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"span_ratio": float(ratio), "span_warn": tol.span_warn,
                    "span_critical": tol.span_critical,
                    "q_min": rates[0], "q_max": rates[-1]}))
    return out


def _chk_monotonic(series, model, verify_paths):
    """`non_monotonic` — the quantity must rise with |rate|, or something other
    than the rate is moving it."""
    fitted = series.fitted_rows()
    if len(fitted) < 3:
        return []
    tol = series.tolerances.rate_tol_pct / 100.0
    ups, downs = [], []
    for a, b in zip(fitted, fitted[1:]):
        qa, qb = abs(a.q_C_per_min), abs(b.q_C_per_min)
        if abs(qb - qa) <= tol * max(qa, 1e-12):
            continue                      # same rate: no direction to compare
        d = _f(b.value_C) - _f(a.value_C)
        if d > 0:
            ups.append((a, b, d))
        elif d < 0:
            downs.append((a, b, d))
    if not (ups and downs):
        return []
    q = series.quantity_kind or "Tg"
    pairs = "; ".join(
        f"{abs(a.q_C_per_min):g}→{abs(b.q_C_per_min):g} °C/min gives "
        f"{_f(a.value_C):.1f}→{_f(b.value_C):.1f} °C" for a, b, _d in downs)
    return [SeriesWarning(
        code="non_monotonic", severity="critical",
        message=(f"{q} is NOT monotonic in heating rate ({pairs}). A faster ramp can "
                 f"only push {q} up, so this ordering is not a rate effect — it is "
                 f"thermal history, baseline choice, or a specimen that changed. "
                 f"Fitted anyway."),
        # Every fitted row is named, not only the two either side of a
        # descending step: the ordering is a property of the whole series, and
        # a single-row extract of the row that happens to sit on an ascending
        # step must not look innocent. The offending pairs are in the message
        # and counted in `detail`.
        row_uids=tuple(r.uid for r in fitted),
        detail={"n_descending_pairs": len(downs), "n_ascending_pairs": len(ups),
                "descending_pairs": [
                    [abs(a.q_C_per_min), abs(b.q_C_per_min)] for a, b, _d in downs]})]


def _chk_thermal_history(series, model, verify_paths):
    """`thermal_history` and `virgin_mixed` — the two ways one specimen's
    successive cycles stop being one material."""
    out = []
    fitted = series.fitted_rows()
    if len(fitted) < 2:
        return out
    priors = sorted({int(r.prior_heats) for r in fitted})
    states = sorted({(r.state or "unknown") for r in fitted})
    if len(priors) > 1 or len(states) > 1:
        # Pair each prior-heat count with the state that goes with it before
        # rendering. Two independently sorted lists printed side by side read
        # as pairs and are not: "0, 1, 2, 3, 4 (after-cycle-1, after-cycle-2,
        # after-cycle-3, after-cycle-4, virgin)" says a virgin specimen had
        # seen four prior ramps, which is the exact confound this warning
        # exists to describe.
        by_prior: Dict[int, List[str]] = {}
        for r in fitted:
            by_prior.setdefault(int(r.prior_heats), [])
            state = (r.state or "unknown")
            if state not in by_prior[int(r.prior_heats)]:
                by_prior[int(r.prior_heats)].append(state)
        listed = ", ".join(f"{p} ({'/'.join(by_prior[p])})" for p in priors)
        out.append(SeriesWarning(
            code="thermal_history", severity="critical",
            message=(f"Mixed thermal history: the fitted rows had seen {listed} prior "
                     f"heating ramps. Each is a differently "
                     f"annealed material, and kinetic analysis assumes one material. "
                     f"Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"prior_heats": priors, "states": states}))
    virgins = [r for r in fitted if r.virgin]
    if len(virgins) == 1 and len(fitted) >= 2 and len(priors) > 1:
        r = virgins[0]
        out.append(SeriesWarning(
            code="virgin_mixed", severity="warn",
            message=(f"{_short(r, series.index_of(r.uid))} is the only virgin "
                     f"(as-quenched) measurement; the first heat of a quenched glass "
                     f"routinely behaves differently from every later one. Consider "
                     f"switching it off, or reading the fit without it. Fitted anyway."),
            row_uids=(r.uid,), detail={"n_virgin": 1, "n_fitted": len(fitted)}))
    return out


def _chk_design(series, model, verify_paths):
    """`single_specimen_cycles`, `mixed_design`, `separate_specimens`.

    There is deliberately NO `mixed_sources` code. Mixing a single-ramp file
    with a segment of a cyclic run is harmless in itself, and alarming on the
    COUNT of `source_kind` values would fire on this feature's own intended
    use. What matters is the CONFOUND — whether one specimen supplies more
    than one fitted row — and that is what `mixed_design` keys on.
    """
    out = []
    fitted = series.fitted_rows()
    if len(fitted) < 2:
        return out
    by_key: Dict[str, List[RateRow]] = {}
    for r in fitted:
        key = (r.specimen_key or "").strip()
        if key:
            by_key.setdefault(key, []).append(r)
    if not by_key:
        return out
    multi = {k: v for k, v in by_key.items() if len(v) >= 2}
    n_known = sum(len(v) for v in by_key.values())

    if len(by_key) == 1 and n_known >= 2:
        key, members = next(iter(by_key.items()))
        name = members[0].sample or key
        out.append(SeriesWarning(
            code="single_specimen_cycles", severity="warn",
            message=(f"All {len(members)} fitted rows are successive cycles of one "
                     f"specimen (\"{name}\") — each heat anneals the material the next "
                     f"heat measures, so heating rate and thermal history are "
                     f"confounded and the slope is unable to separate them. "
                     f"Fitted anyway."),
            row_uids=tuple(m.uid for m in members),
            detail={"specimen_key": key, "n_rows": len(members)}))
    elif multi and len(by_key) > 1:
        key, members = sorted(multi.items(), key=lambda kv: -len(kv[1]))[0]
        others = sum(len(v) for k, v in by_key.items() if k != key)
        name = members[0].sample or key
        out.append(SeriesWarning(
            code="mixed_design", severity="warn",
            message=(f"{len(members)} rows are successive cycles of \"{name}\" and "
                     f"{others} are separate specimens — the cycle effect and the "
                     f"specimen-to-specimen effect are confounded in one slope. "
                     f"Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"n_cycle_rows": len(members), "n_other_rows": others,
                    "specimen_key": key}))
    elif len(by_key) >= 2 and not multi:
        out.append(SeriesWarning(
            code="separate_specimens", severity="info",
            message=(f"{len(by_key)} separate specimens, one per rate — the intended "
                     f"design. The fit assumes identical composition and identical "
                     f"quench history across them; confirm that."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"n_specimens": len(by_key)}))
    return out


def _chk_mass(series, model, verify_paths):
    """`mass_loss_spread` and `mass_loss_high`.

    Both read `cum_mass_loss_pct`, which is measured only between matched
    temperatures on heating ramps (see the field docstring); the SPREAD is
    the buoyancy-free half of that number and is therefore the one that gets
    to be critical.
    """
    out = []
    fitted = series.fitted_rows()
    tol = series.tolerances
    vals = [(r, _f(r.cum_mass_loss_pct)) for r in fitted if _finite(r.cum_mass_loss_pct)]
    if len(vals) >= 2:
        lo = min(v for _r, v in vals)
        hi = max(v for _r, v in vals)
        spread = hi - lo
        if spread > tol.mass_loss_spread_pct:
            severity = "critical" if spread > 2.0 * tol.mass_loss_spread_pct else "warn"
            out.append(SeriesWarning(
                code="mass_loss_spread", severity=severity,
                message=(f"Mass loss along the series: {lo:.2f} % to {hi:.2f} % "
                         f"cumulative ({spread:.2f} pp apart) — the rows are not "
                         f"mass-comparable, so part of the trend is volatilization "
                         f"rather than rate. Fitted anyway."),
                row_uids=tuple(r.uid for r, _v in vals),
                detail={"lo_pct": lo, "hi_pct": hi, "spread_pp": spread,
                        "mass_loss_spread_pct": tol.mass_loss_spread_pct}))
    high = [(r, v) for r, v in vals if v > tol.mass_loss_abs_pct]
    if high:
        worst = max(high, key=lambda rv: rv[1])
        out.append(SeriesWarning(
            code="mass_loss_high", severity="warn",
            message=(f"{len(high)} fitted row(s) have lost more than "
                     f"{tol.mass_loss_abs_pct:g} % of their mass (worst: "
                     f"{_short(worst[0], series.index_of(worst[0].uid))} at "
                     f"{worst[1]:.2f} %); volatilization has changed the composition. "
                     f"Fitted anyway."),
            row_uids=tuple(r.uid for r, _v in high),
            detail={"n_rows": len(high), "worst_pct": worst[1],
                    "mass_loss_abs_pct": tol.mass_loss_abs_pct}))
    return out


def _chk_consistency(series, model, verify_paths):
    """`mixed_quantity`, `mixed_method`, `mixed_window`, `mixed_signal`."""
    out = []
    fitted = series.fitted_rows()
    if len(fitted) < 2:
        return out
    kinds = sorted({(r.quantity_kind or "").strip() for r in fitted})
    series_kind = (series.quantity_kind or "").strip()
    if len(kinds) > 1 or (series_kind and kinds and kinds != [series_kind]):
        out.append(SeriesWarning(
            code="mixed_quantity", severity="critical",
            message=(f"Mixed quantities in one series ({', '.join(kinds)}; the series "
                     f"is labelled {series_kind or '—'}) — a Tg and a Tp are different "
                     f"temperatures, so one slope across them has no physical meaning. "
                     f"Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"quantity_kinds": kinds, "series_quantity_kind": series_kind}))
    methods = sorted({(r.method or "").strip() for r in fitted if (r.method or "").strip()})
    if len(methods) > 1:
        out.append(SeriesWarning(
            code="mixed_method", severity="warn",
            message=(f"Rows mix '{methods[0]}' and '{methods[1]}' constructions of "
                     f"{series_kind or 'the quantity'}. The methods differ "
                     f"systematically, so method scatter enters the rate slope. "
                     f"Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"methods": methods}))
    windows = [(r, (_f(r.window_C[0]), _f(r.window_C[1]))) for r in fitted
               if _finite(r.window_C[0]) and _finite(r.window_C[1])]
    if len(windows) >= 2:
        los = [w[0] for _r, w in windows]
        his = [w[1] for _r, w in windows]
        if (max(los) - min(los) > series.tolerances.window_tol_C
                or max(his) - min(his) > series.tolerances.window_tol_C):
            listed = _join([f"{w[0]:g}–{w[1]:g}" for _r, w in windows])
            out.append(SeriesWarning(
                code="mixed_window", severity="warn",
                message=(f"Rows were analysed with DIFFERENT windows ({listed} °C). On "
                         f"this kind of data the window moves Tg by 10–20 K, LARGER "
                         f"than the whole rate-induced shift over a 2× rate range "
                         f"(~5 K) — a series with inconsistent windows measures the "
                         f"window, not the rate. Fitted anyway."),
                row_uids=tuple(r.uid for r, _w in windows),
                detail={"window_tol_C": series.tolerances.window_tol_C}))
    signals = sorted({(r.signal or "").strip() for r in fitted if (r.signal or "").strip()})
    if len(signals) > 1:
        out.append(SeriesWarning(
            code="mixed_signal", severity="warn",
            message=(f"Rows were measured on different signals ({', '.join(signals)}) — "
                     f"a heat-flow onset and a temperature-difference onset are not the "
                     f"same construction. Fitted anyway."),
            row_uids=tuple(r.uid for r in fitted),
            detail={"signals": signals}))
    return out


def _chk_rate_provenance(series, model, verify_paths):
    """`rate_mismatch`, `setpoint_not_reached`, `partial_segment`, `cooling_row`,
    `rate_unverified`, `rate_overridden`."""
    out = []
    fitted = series.fitted_rows()
    tol = series.tolerances
    mism = []
    for r in fitted:
        m, p = _f(r.q_measured_C_per_min), _f(r.q_programmed_C_per_min)
        if np.isfinite(m) and np.isfinite(p) and p != 0.0:
            pct = abs(abs(m) - abs(p)) / abs(p) * 100.0
            if pct > tol.rate_mismatch_tol_pct:
                mism.append((r, m, p, pct))
    if mism:
        r, m, p, pct = mism[0]
        out.append(SeriesWarning(
            code="rate_mismatch", severity="warn",
            message=(f"{_short(r, series.index_of(r.uid))}: programmed {p:+.2f} but "
                     f"measured {m:+.2f} °C/min ({pct:.0f} % apart) — the furnace did "
                     f"not follow the program. The measured rate is what the fit uses."
                     + (f" {len(mism) - 1} other row(s) too." if len(mism) > 1 else "")
                     + " Fitted anyway."),
            row_uids=tuple(x[0].uid for x in mism),
            detail={"n_rows": len(mism), "worst_pct": pct,
                    "rate_mismatch_tol_pct": tol.rate_mismatch_tol_pct}))
    unreached = [r for r in fitted if r.setpoint_reached is False]
    if unreached:
        # WHY this says less than it used to. SPEC_rate_series §4.5 wrote the
        # sentence "the measured X °C/min is the real rate; the programmed
        # Y °C/min is not what the specimen saw" — and on real data that is
        # false. MERGED_DECISIONS M3 sets the flag at `target_gap_C <= 5.0`,
        # and BM_FINDINGS §3 measured why that fires on every heating ramp
        # there is: a heat's setpoint arrives during the SOAK that follows it
        # (the four ramps end 5.1-8.3 °C short and the 1300 °C hold then tops
        # out at 1298.6), so the gap on a heat is nonzero by construction.
        # Meanwhile measured and programmed agree to 0.06-0.26 % on those same
        # ramps, which the module's own `rate_mismatch` tolerance (5 %) calls
        # a match. A warn-level sentence that contradicts the number beside it
        # is exactly the dilution M0 is trying to avoid, so this states the
        # gap it measured and leaves the reading to the person — the wording
        # `dta_segments.attach_alignment` already uses for the same quantity.
        heats = [x for x in unreached if x.segment_kind == ds.KIND_HEAT]
        soak_note = (" On a heating ramp the setpoint is usually reached during the "
                     "hold that follows, not inside the ramp." if heats else "")
        r = unreached[0]
        out.append(SeriesWarning(
            code="setpoint_not_reached", severity="warn",
            message=(f"{len(unreached)} row(s) ended short of the programmed target "
                     f"({_join([_short(x, series.index_of(x.uid)) for x in unreached])}); "
                     f"the first stopped with the furnace still travelling, at a "
                     f"measured {_f(r.q_measured_C_per_min):+.2f} against a programmed "
                     f"{_f(r.q_programmed_C_per_min):+.2f} °C/min."
                     + soak_note +
                     " The measured rate is what the fit uses. Fitted anyway."),
            row_uids=tuple(x.uid for x in unreached),
            detail={"n_rows": len(unreached), "n_heating_rows": len(heats)}))
    partial = [r for r in fitted if r.is_partial]
    if partial:
        out.append(SeriesWarning(
            code="partial_segment", severity="warn",
            message=(f"{len(partial)} row(s) come from a truncated segment "
                     f"({_join([_short(x, series.index_of(x.uid)) for x in partial])}) "
                     f"— acquisition stopped inside the ramp, so the transition may sit "
                     f"outside the recorded span. Fitted anyway."),
            row_uids=tuple(x.uid for x in partial),
            detail={"n_rows": len(partial)}))
    cooling = [r for r in fitted
               if r.segment_kind == ds.KIND_COOL or r.role_in_cycle == ds.ROLE_COOL
               or (np.isfinite(r.q_C_per_min) and r.q_C_per_min < 0.0)]
    if cooling:
        out.append(SeriesWarning(
            code="cooling_row", severity="warn",
            message=(f"{len(cooling)} row(s) come from a COOLING segment "
                     f"({_join([_short(x, series.index_of(x.uid)) for x in cooling])}) "
                     f"— a cooling-rate Tg is not interchangeable with a heating-rate "
                     f"Tg, and a cooling exotherm gives a negative Ea. The rate "
                     f"magnitude is what the fit uses. Fitted anyway."),
            row_uids=tuple(x.uid for x in cooling),
            detail={"n_rows": len(cooling)}))
    typed = [r for r in fitted if r.q_used_source in ("manual", "filename")]
    if typed:
        out.append(SeriesWarning(
            code="rate_unverified", severity="info",
            message=(f"{len(typed)} row(s) carry a rate that was typed or read from a "
                     f"file name rather than measured from the instrument record: "
                     f"{_join([_short(x, series.index_of(x.uid)) for x in typed])}."),
            row_uids=tuple(x.uid for x in typed),
            detail={"n_rows": len(typed)}))
    overridden = []
    for r in fitted:
        man, meas = _f(r.q_manual_C_per_min), _f(r.q_measured_C_per_min)
        if np.isfinite(man) and np.isfinite(meas) and r.q_used_source == "manual":
            if abs(abs(man) - abs(meas)) > (tol.rate_tol_pct / 100.0) * max(abs(meas), 1e-12):
                overridden.append(r)
    if overridden:
        out.append(SeriesWarning(
            code="rate_overridden", severity="info",
            message=(f"{len(overridden)} row(s) use a hand-typed rate that differs from "
                     f"the measured one: "
                     f"{_join([_short(x, series.index_of(x.uid)) for x in overridden])}."),
            row_uids=tuple(x.uid for x in overridden),
            detail={"n_rows": len(overridden)}))
    return out


def _chk_units(series, model, verify_paths):
    """`unit_suspect` and `value_outside_window`.

    The 273.15 fingerprint is the runtime net under the structural rule that
    a temperature is stored in Celsius and converted in exactly one place: a
    value converted twice still looks like a plausible Tg on a plot.

    The test is against the SERIES' OWN TREND, not against other rows
    pairwise, and that is the whole point. SPEC_rate_series §4.5 wrote it as
    "two rows differing by 273.15 ± tol", but a doubly-converted row sits
    273.15 above ITS OWN correct value, and in a Lasocka series every row has
    a different correct value by construction — so the difference to any
    other row is 273.15 ± their Tg gap and the pairwise test misses the exact
    bug it is named for (measured: [500, 506, 784.15, 517] °C, where 784.15
    is 511.0 converted twice, did not fire). It fired only when two rows
    happened to share a temperature, and it false-fired `critical` on any
    legitimate series spanning 273.15 ± 1 °C.

    So: a row is suspect when it sits more than half a kelvin offset above
    every other row AND subtracting K0 lands it back inside their range.
    Both halves are needed — the first alone flags any high outlier, the
    second alone flags any series that happens to span 273 °C (measured:
    [200, 210, 473.15, 480] °C, an ordinary spread, used to raise a
    `critical`). What this deliberately does NOT catch is every row being
    converted twice at once: that shifts the whole series together, leaves B
    and Ea untouched, and is caught by the PLAUSIBLE_T_C bound above when it
    puts the values somewhere impossible.
    """
    out = []
    tol = series.tolerances
    bad = []
    for r in series.rows:
        v, q = _f(r.value_C), abs(_f(r.q_C_per_min))
        if _finite(v) and not (PLAUSIBLE_T_C[0] <= v <= PLAUSIBLE_T_C[1]):
            bad.append((r, f"value {v:.1f} °C is outside "
                           f"{PLAUSIBLE_T_C[0]:.0f}–{PLAUSIBLE_T_C[1]:.0f} °C"))
        if np.isfinite(q) and q > 0 and not (PLAUSIBLE_Q[0] <= q <= PLAUSIBLE_Q[1]):
            bad.append((r, f"rate {q:g} °C/min is outside "
                           f"{PLAUSIBLE_Q[0]:g}–{PLAUSIBLE_Q[1]:g} °C/min"))
    finite_rows = [r for r in series.rows if _finite(r.value_C)]
    for a in finite_rows:
        others = [_f(b.value_C) for b in finite_rows if b is not a]
        if len(others) < 2:
            continue
        lo, hi = min(others), max(others)
        v = _f(a.value_C)
        pad = max(_f(tol.kelvin_fingerprint_C, 0.0), UNIT_TWIN_PAD_C)
        if (v - hi) > 0.5 * K0 and (lo - pad <= v - K0 <= hi + pad):
            bad.append((a, f"sits {v - hi:.0f} °C above every other row "
                           f"({lo:.1f}–{hi:.1f} °C) and lands back inside them when "
                           f"273.15 is subtracted ({v - K0:.1f} °C) — the signature "
                           f"of one value having been converted to kelvin twice"))
    if bad:
        reasons = [f"{_short(r, series.index_of(r.uid))}: {why}" for r, why in bad if why]
        out.append(SeriesWarning(
            code="unit_suspect", severity="critical",
            message=("Suspicious units — " + _join(reasons, 3) +
                     ". Check the Value and Rate columns; the fit uses them as they "
                     "stand. Fitted anyway."),
            row_uids=tuple({r.uid for r, _w in bad}),
            detail={"n_rows": len({r.uid for r, _w in bad}),
                    "kelvin_fingerprint_C": tol.kelvin_fingerprint_C,
                    "unit_twin_pad_C": max(_f(tol.kelvin_fingerprint_C, 0.0),
                                           UNIT_TWIN_PAD_C),
                    "plausible_T_C": list(PLAUSIBLE_T_C),
                    "plausible_q": list(PLAUSIBLE_Q)}))
    outside = []
    for r in series.fitted_rows():
        lo, hi = _f(r.window_C[0]), _f(r.window_C[1])
        v = _f(r.value_C)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(v) and not (lo <= v <= hi):
            outside.append((r, v, lo, hi))
    if outside:
        r, v, lo, hi = outside[0]
        out.append(SeriesWarning(
            code="value_outside_window", severity="warn",
            message=(f"{_short(r, series.index_of(r.uid))}'s value ({v:.0f} °C) lies "
                     f"outside its {lo:g}–{hi:g} °C window — an extrapolated tangent "
                     f"intersection is arithmetic, not a measurement. Fitted anyway."),
            row_uids=tuple(x[0].uid for x in outside),
            detail={"n_rows": len(outside)}))
    return out


def _chk_sources(series, model, verify_paths):
    """`no_heating_segment`, `ambiguous_file_rate`, `missing_source_file`.

    Re-derived here from the stored row state so that a series reopened from
    a project or read back from a CSV carries the same objections the builder
    emitted when the file was first added.
    """
    out = []
    orphan = [r for r in series.rows
              if r.source_kind in ("segment", "file", "library")
              and r.segment_index < 0 and not np.isfinite(r.q_C_per_min)]
    if orphan:
        out.append(SeriesWarning(
            code="no_heating_segment", severity="critical",
            message=(f"{len(orphan)} source(s) produced no heating ramp, so the rate is "
                     f"unknown: {_join([_short(r, series.index_of(r.uid)) for r in orphan])}. "
                     f"The rows are kept — type a rate, or pick a segment."),
            row_uids=tuple(r.uid for r in orphan),
            detail={"n_rows": len(orphan)}))

    by_file: Dict[str, List[RateRow]] = {}
    for r in series.rows:
        if r.file and r.n_heating_segments_in_source > 1:
            by_file.setdefault(r.file, []).append(r)
    for path, members in by_file.items():
        # Count and rates must come from the SAME set of rows. This used to
        # print `n_heating_segments_in_source` (heats only) beside rates
        # gathered from every member, so the moment cooling rows joined the
        # series — which MERGED_DECISIONS M1 promotes to MUST — the sentence
        # claimed "4 heating ramps at 7 rates", three of which the furnace
        # never ran as a heating rate. Mirrors `_source_warnings`, which was
        # already self-consistent, so the two emitters of this one code now
        # say the same thing.
        kinds = sorted({m.segment_kind for m in members if m.segment_kind})
        word = "heating ramps" if kinds == [ds.KIND_HEAT] else "ramps"
        rates = [round(v, 2) for v in
                 cluster_rates((m.q_C_per_min for m in members
                                if np.isfinite(m.q_C_per_min)),
                               series.tolerances.rate_tol_pct)]
        if len(rates) > 1:
            listed = ", ".join(f"{v:g}" for v in rates)
            # Same split as `_source_warnings`: rows the user picked as segments
            # of the run in front of them were MEANT to span several rates, and
            # telling them "pick which to include" about a deliberate choice is
            # both wrong and the thing that used to leave the series unfittable.
            if all(m.source_kind == "segment" for m in members):
                out.append(SeriesWarning(
                    code="multi_rate_source", severity="info",
                    message=(f"{_stem(path)} contributes {len(members)} {word} at "
                             f"{listed} °C/min — that spread is what a rate series "
                             f"is for. They are successive cycles of ONE specimen, "
                             f"so read the thermal-history warning before quoting "
                             f"a slope."),
                    row_uids=tuple(m.uid for m in members),
                    detail={"file": path, "rates": rates, "n_rows": len(members)}))
            else:
                out.append(SeriesWarning(
                    code="ambiguous_file_rate", severity="info",
                    message=(f"{_stem(path)} holds {len(members)} {word} at "
                             f"{listed} °C/min "
                             f"— pick which to include; nothing was guessed for you."),
                    row_uids=tuple(m.uid for m in members),
                    detail={"file": path, "rates": rates, "n_rows": len(members)}))

    if verify_paths:
        missing = []
        for r in series.rows:
            if r.file and not os.path.exists(r.file):
                missing.append(r)
        if missing:
            r = missing[0]
            out.append(SeriesWarning(
                code="missing_source_file", severity="warn",
                message=(f"{len(missing)} row(s) point at a source file that is no "
                         f"longer there (first: {r.file}). The stored values still fit "
                         f"— a stored Tg is a measurement, not a cache — but "
                         f"re-measuring needs the file: put it back where it was, or "
                         f"add it again from its new place and remove the stale row."),
                row_uids=tuple(x.uid for x in missing),
                detail={"n_rows": len(missing), "first_path": r.file}))
    return out


def _chk_duplicate_rows(series, model, verify_paths):
    """`duplicate_source_row` — the same segment of the same file, twice.

    "Add segments from current file" appends; the button stays enabled; and
    nothing anywhere compared a new row's provenance against the rows already
    in the series. Pressing it twice therefore doubled every row, and the fit
    that came out was the WORST kind of wrong: the slope is bit-identical, so
    nothing looks different, while sigma falls by sqrt(2), dof goes 1 -> 4 and
    the `dof_one` critical ("R^2 is not evidence here") disappears. The fit
    improves in exactly the two places a reader checks. Only `duplicate_rate`
    fired, and its wording ("weight without leverage... Both rows are kept,
    unweighted") reads as a legitimate replicate design.

    Identity is `(source_kind, file, segment_index)` and NOT the uid: uids are
    minted fresh on every add, which is also why `RateSeries.validate()` could
    never have caught this. Rows are KEPT -- a genuine duplicate is sometimes
    what the user wants, and M0.1 forbids refusing one -- and the warning is
    critical because the number it changes is an uncertainty.
    """
    out = []
    groups: Dict[Tuple[str, str, int], List[RateRow]] = {}
    for r in series.rows:
        if r.source_kind not in ("segment", "file", "library"):
            continue        # a typed row has no provenance to repeat
        if not str(r.file or "").strip() or int(r.segment_index) < 0:
            continue
        key = (str(r.source_kind), str(r.file), int(r.segment_index))
        groups.setdefault(key, []).append(r)
    repeats = [rows for rows in groups.values() if len(rows) > 1]
    if not repeats:
        return out
    dups = [r for rows in repeats for r in rows]
    worst = max(len(rows) for rows in repeats)
    out.append(SeriesWarning(
        code="duplicate_source_row", severity="critical",
        message=(f"{len(dups)} row(s) are the SAME segment of the same file added "
                 f"more than once ({len(repeats)} segment(s), up to {worst} copies): "
                 f"{_join([_short(r, series.index_of(r.uid)) for r in dups])}. "
                 f"Copies add no information but they do divide the fitted "
                 f"uncertainty by sqrt(n) and raise the degrees of freedom, so the "
                 f"fit reads better than the data supports. Remove the copies "
                 f"unless you meant to weight that point."),
        row_uids=tuple(r.uid for r in dups),
        detail={"n_duplicate_rows": len(dups), "n_segments": len(repeats),
                "max_copies": worst}))
    return out


def _chk_abscissa(series, model, verify_paths):
    """`coarse_abscissa` — a value measured off a heavily de-duplicated curve.

    `prepare_segment_xy` collapses exact-duplicate temperatures by averaging
    and records the share it collapsed; `_rate_arrays_for` stores that share on
    the row as `extras["duplicate_fraction"]`. A column written to one or two
    decimals, or a ramp slow enough that the rounding step exceeds dT per
    sample, can collapse most of the curve: measured 16.7 % on a noiseless
    1 C/min 2-decimal ramp and 58.3 % at 0.5 C/min.

    This used to be an exception, which made such a file unable to yield a Tg
    at all (M0.1's forbidden case). It is a warning now, which is the whole
    point: the number is real, the curve behind it has fewer independent
    points than the row count suggests, and the user is the one who gets to
    decide whether that matters.
    """
    out = []
    bad = []
    worst = 0.0
    for r in series.rows:
        frac = _f(dict(r.extras or {}).get("duplicate_fraction"))
        if np.isfinite(frac) and frac > 0.01:
            bad.append(r)
            worst = max(worst, float(frac))
    if not bad:
        return out
    out.append(SeriesWarning(
        code="coarse_abscissa", severity="warn",
        message=(f"{len(bad)} value(s) were read off a curve whose temperature "
                 f"column repeats itself: up to {100.0 * worst:.1f} % of the points "
                 f"shared a temperature and were averaged together "
                 f"({_join([_short(r, series.index_of(r.uid)) for r in bad])}). "
                 f"The values stand — the duplicates are averaged, not dropped — "
                 f"but the curve holds fewer independent points than the row "
                 f"count suggests. It usually means a coarsely rounded "
                 f"temperature column or a scan slow enough that the rounding "
                 f"step exceeds the temperature change per sample."),
        row_uids=tuple(r.uid for r in bad),
        detail={"n_rows": len(bad), "worst_fraction": float(worst)}))
    return out


def _chk_model(series, model, verify_paths):
    """`model_quantity_mismatch` — the chosen model wants a quantity this
    series does not hold."""
    if not model:
        return []
    meta = KINETIC_METHODS.get(str(model).strip())
    if meta is None:
        return []
    needs = tuple(meta.get("needs", ()))
    kind = (series.quantity_kind or "").strip()
    if not needs or not kind or kind in needs:
        return []
    return [SeriesWarning(
        code="model_quantity_mismatch", severity="warn",
        message=(f"{meta.get('label', model)} is defined for {needs[0]}; this series "
                 f"holds {kind} — the number returned is arithmetic on the right shape "
                 f"of equation, not the activation energy that model names. "
                 f"Fitted anyway."),
        detail={"method": model, "needs": list(needs), "quantity_kind": kind})]


_CHECKS: Tuple[Callable[..., List[SeriesWarning]], ...] = (
    _chk_counts,
    _chk_missing_values,
    _chk_rate_design,
    _chk_monotonic,
    _chk_thermal_history,
    _chk_design,
    _chk_mass,
    _chk_consistency,
    _chk_rate_provenance,
    _chk_units,
    _chk_sources,
    _chk_duplicate_rows,
    _chk_abscissa,
    _chk_model,
)


# =============================================================================
# 6) Fits
# =============================================================================

def _merge_warnings(result: KineticFitResult,
                    extra: Sequence[SeriesWarning]) -> KineticFitResult:
    """Fold provenance warnings into a fit result, de-duplicated and sorted.

    De-duplication is on (code, message, row_uids) rather than on `code`
    alone: two `duplicate_rate` warnings about two different rates are two
    different facts, while the same warning arriving from both layers is one.
    """
    seen = set()
    warns: List[SeriesWarning] = []
    notes: List[SeriesWarning] = []
    for w in list(result.warnings) + list(result.notes) + list(extra):
        key = (w.code, w.message, tuple(w.row_uids))
        if key in seen:
            continue
        seen.add(key)
        (notes if w.severity == "info" else warns).append(w)
    result.warnings = sort_warnings(warns)
    result.notes = sort_warnings(notes)
    return result


def fit_series(series: RateSeries, method: str = "", *,
               q_ref_C_per_min: float = dk.DEFAULT_Q_REF,
               warnings_in: Optional[Sequence[SeriesWarning]] = None,
               verify_paths: bool = False) -> KineticFitResult:
    """THE entry point the Qt layer calls. NEVER raises.

    Builds RatePoints from `fitted_rows()` (plot order, ascending |q|),
    dispatches through `KINETIC_METHODS`, and merges `check_series()` into
    the result so a fit is never seen without the objections that apply to
    it.

    `ok=False` means the ARITHMETIC could not be done — fewer than two
    fittable rows, or a single rate giving zero x-spread — and it arrives as
    a message on a well-formed result, not as an exception. A physically
    wrong answer is never `ok=False`: a negative Lasocka B, a negative Ea,
    four cycles of one specimen and a 2× rate span all come back as the
    number plus their warnings, because the user asked to be able to process
    exactly this data and to be told what is wrong with it.

    `warnings_in` lets `fit_all` share ONE `check_series` pass across six
    models instead of recomputing it six times. `verify_paths` defaults to
    False here (and True in `check_series`) because a fit runs on every
    keystroke of a debounced UI and stat()-ing a network share on each one
    would be the slowest thing in the loop.
    """
    if series is None:
        series = RateSeries()
    key = (str(method).strip() or default_method_for(series.quantity_kind))
    checks = list(warnings_in) if warnings_in is not None else \
        check_series(series, model=key, verify_paths=verify_paths)

    rows = series.fitted_rows()
    points = [r.to_point() for r in rows]
    try:
        result = dk.fit_points(points, key, q_ref_C_per_min=q_ref_C_per_min,
                               r2_min=series.tolerances.r2_min)
    except Exception as exc:                                  # pragma: no cover
        # dta_kinetics already guarantees never-raises; this is the belt to
        # its braces, because the Qt layer has no second line of defence.
        result = dk.fit_points([], key)
        result.ok = False
        result.message = f"{type(exc).__name__}: {exc}"

    # n_total / n_included describe the SERIES; dta_kinetics could only see
    # the points it was handed. n_fitted is left exactly as the fit reported
    # it (fit.n), so "3 of 4 rows" always means the three the arithmetic
    # actually used rather than the three this layer expected it to use.
    result.n_total = len(series.rows)
    result.n_included = len(series.included_rows())
    return _merge_warnings(result, checks)


def evaluable_methods(series: RateSeries) -> List[str]:
    """Registry keys whose `needs` the present rows satisfy, in registry order.

    A method is offered when the series holds the quantity it is defined for,
    and — for Augis-Bennett — when at least two rows carry a finite T0, since
    a fit over fewer than two points is not a fit. Anything the series does
    not satisfy is left out of `fit_all` rather than fitted and marked; the
    user can still ask for it explicitly and get the number with
    `model_quantity_mismatch` attached.
    """
    rows = series.fitted_rows()
    kinds = {(r.quantity_kind or "").strip() for r in rows} or {series.quantity_kind}
    n_t0 = sum(1 for r in rows if _finite(r.T0_C))
    out = []
    for key, meta in KINETIC_METHODS.items():
        needs = set(meta.get("needs", ()))
        if needs and not (needs & kinds):
            continue
        if meta.get("needs_T0") and n_t0 < 2:
            continue
        out.append(key)
    return out


def fit_all(series: RateSeries, *, q_ref_C_per_min: float = dk.DEFAULT_Q_REF,
            methods: Optional[Sequence[str]] = None,
            verify_paths: bool = False) -> List[KineticFitResult]:
    """Every applicable model, sharing ONE `check_series` pass.

    Sharing the pass is not only cheaper: it means the Lasocka row and the
    Moynihan row of the exported fit CSV carry the SAME warning objects, so a
    reviewer comparing them is comparing models rather than comparing two
    independent recomputations of the same objections.
    """
    if series is None:
        return []
    keys = list(methods) if methods is not None else evaluable_methods(series)
    checks = check_series(series, verify_paths=verify_paths)
    out = []
    for key in keys:
        model_checks = list(checks) + list(_chk_model(series, key, verify_paths))
        out.append(fit_series(series, key, q_ref_C_per_min=q_ref_C_per_min,
                              warnings_in=model_checks))
    return out


# =============================================================================
# 7) Tables and CSV
# =============================================================================

#: Tg-tab settings snapshotted onto each row, and the CSV column each becomes.
#: Snapshotting rather than pointing at a shared store is what makes an
#: exported series self-describing: the file says which baseline mode and
#: which window produced each number, months after the widgets have moved.
SETTINGS_CSV_KEYS: Tuple[str, ...] = (
    "tg_method", "smooth_derivative", "y_col", "dy_source", "invert_y",
    "low_mode", "low_range", "low_point_x", "high_mode", "high_range",
    "high_point_x", "tp_signal", "tp_window_min_C", "tp_window_max_C",
)

#: A rows CSV must carry at least one of these to be a rows CSV at all. See
#: the schema floor in `read_series_csv`.
ROWS_CSV_REQUIRED_ANY: Tuple[str, ...] = (
    "uid", "value_C", "value_K", "value_source", "q_C_per_min", "quantity_kind",
    "q_measured_C_per_min", "segment_index",
)

ROWS_CSV_COLUMNS: Tuple[str, ...] = (
    "row_index", "uid", "include", "fitted", "source_kind", "sample", "specimen_key",
    "state", "file", "record_id", "segment_index", "segment_kind", "segment_label",
    "role_in_cycle", "cycle_index", "row_start", "row_stop",
    "n_heating_segments_in_source",
    "q_C_per_min", "q_used_source", "rate_source", "q_measured_C_per_min",
    "q_programmed_C_per_min", "q_filename_C_per_min", "q_manual_C_per_min",
    "quantity_kind", "method", "signal", "window_min_C", "window_max_C",
    "value_C", "value_K", "value_source", "error",
    "T0_C", "extras", "prior_heats", "prior_max_T_C", "cum_mass_loss_pct",
    "sample_mass_mg", "is_partial", "setpoint_reached", "flags", "note",
) + SETTINGS_CSV_KEYS + ("max_severity", "warning_codes", "warning_messages")

FIT_CSV_LEAD: Tuple[str, ...] = (
    "series_id", "series_name", "quantity_kind", "method", "method_label", "equation",
    "temp_basis", "x_label", "y_label",
    "n_rows_total", "n_rows_included", "n_rows_fitted", "n_excluded", "n_no_value",
    "rows_used", "rates_used_C_per_min", "n_distinct_rates", "rate_span_ratio",
    "slope", "slope_se", "intercept", "intercept_se", "r_squared", "s_resid", "dof",
    "t_crit_95", "Ea_kJ_mol", "Ea_se_kJ_mol", "Ea_ci95_lo_kJ_mol", "Ea_ci95_hi_kJ_mol",
    "pairwise_Ea_lo_kJ_mol", "pairwise_Ea_hi_kJ_mol", "pairwise_slope_lo",
    "pairwise_slope_hi", "lasocka_A_C", "lasocka_A_K", "lasocka_A_se", "lasocka_B",
    "lasocka_B_se", "fragility_m", "q_ref_C_per_min", "T_at_q_ref_C", "residuals",
    "extras", "monotonic", "fit_ok", "fit_message",
    "specimen_keys", "prior_heats_used", "mass_loss_range_pct",
    "max_severity", "n_warnings", "warning_codes", "warning_messages",
    "generated_by", "generated_at",
)

WARNINGS_CSV_COLUMNS: Tuple[str, ...] = (
    "series_id", "code", "severity", "message", "row_indices", "row_uids", "detail_json",
)


def _cell(value: Any) -> Any:
    """CSV cell: '' for absent, never 0 and never the string 'nan'.

    A blank means "not applicable / not measured"; a 0 in a mass-loss or Ea
    column is a claim. The existing DTA export writes the text 'nan', which
    is neither, and this schema deliberately breaks with it.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return "" if not np.isfinite(value) else value
    return value


def rows_table(series: RateSeries,
               warnings: Optional[Sequence[SeriesWarning]] = None) -> List[Dict[str, Any]]:
    """One dict per row, in TABLE order (never rate-sorted), pure stdlib.

    `warning_codes` on a row is that row's own codes PLUS every series-wide
    code, so a single-row extract from the export is never innocent-looking.
    """
    warns = list(warnings if warnings is not None else check_series(series))
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(series.rows, 1):
        mine = [w for w in warns if not w.row_uids or row.uid in w.row_uids]
        q, q_src = resolve_rate(row)
        rec: Dict[str, Any] = {
            "row_index": i,
            "uid": row.uid,
            "include": bool(row.include),
            "fitted": bool(row.fittable),
            "source_kind": row.source_kind,
            "sample": row.sample,
            "specimen_key": row.specimen_key,
            "state": row.state,
            "file": row.file,
            "record_id": row.record_id,
            "segment_index": row.segment_index if row.segment_index >= 0 else "",
            "segment_kind": row.segment_kind,
            "segment_label": row.segment_label,
            "role_in_cycle": row.role_in_cycle,
            "cycle_index": row.cycle_index or "",
            "row_start": row.row_start if row.row_start >= 0 else "",
            "row_stop": row.row_stop if row.row_stop >= 0 else "",
            "n_heating_segments_in_source": row.n_heating_segments_in_source,
            "q_C_per_min": _cell(q),
            "q_used_source": q_src,
            "rate_source": row.rate_source,
            "q_measured_C_per_min": _cell(_f(row.q_measured_C_per_min)),
            "q_programmed_C_per_min": _cell(_f(row.q_programmed_C_per_min)),
            "q_filename_C_per_min": _cell(_f(row.q_filename_C_per_min)),
            "q_manual_C_per_min": _cell(_f(row.q_manual_C_per_min)),
            "quantity_kind": row.quantity_kind,
            "method": row.method,
            "signal": row.signal,
            "window_min_C": _cell(_f(row.window_C[0])),
            "window_max_C": _cell(_f(row.window_C[1])),
            "value_C": _cell(_f(row.value_C)),
            "value_K": _cell(row.value_K),
            "value_source": row.value_source,
            "error": row.error,
            "T0_C": _cell(_f(row.T0_C)),
            "extras": ";".join(f"{k}={row.extras[k]:g}" for k in sorted(row.extras)
                               if _finite(row.extras[k])),
            "prior_heats": row.prior_heats,
            "prior_max_T_C": _cell(_f(row.prior_max_T_C)),
            "cum_mass_loss_pct": _cell(_f(row.cum_mass_loss_pct)),
            "sample_mass_mg": _cell(_f(row.sample_mass_mg)),
            "is_partial": bool(row.is_partial),
            "setpoint_reached": "" if row.setpoint_reached is None else bool(row.setpoint_reached),
            "flags": row.flags,
            "note": row.note,
        }
        for key in SETTINGS_CSV_KEYS:
            rec[key] = _cell(row.settings.get(key, ""))
        rec["max_severity"] = max_severity(mine)
        rec["warning_codes"] = ";".join(warning_codes(warns, row.uid))
        rec["warning_messages"] = " | ".join(w.line() for w in sort_warnings(mine))
        out.append(rec)
    return out


def _is_monotonic_increasing(series: RateSeries,
                             fitted: Sequence[RateRow]) -> bool:
    """Does the quantity rise with |rate| across the fitted rows?

    True with fewer than two rows (there is no ordering to violate) and True
    on ties: two rows at the same rate, within `rate_tol_pct`, are one rate
    measured twice and say nothing about direction — the same exemption
    `_chk_monotonic` makes, so the column and the warning agree about what a
    step is.
    """
    rows = [r for r in fitted if _finite(r.value_C) and np.isfinite(r.q_C_per_min)]
    if len(rows) < 2:
        return True
    rows = sorted(rows, key=lambda r: abs(r.q_C_per_min))
    tol = max(float(series.tolerances.rate_tol_pct), 0.0) / 100.0
    for a, b in zip(rows, rows[1:]):
        qa, qb = abs(a.q_C_per_min), abs(b.q_C_per_min)
        if abs(qb - qa) <= tol * max(qa, 1e-12):
            continue
        if _f(b.value_C) < _f(a.value_C):
            return False
    return True


def fit_table(series: RateSeries, fits: Sequence[KineticFitResult],
              warnings: Optional[Sequence[SeriesWarning]] = None) -> List[Dict[str, Any]]:
    """One dict per evaluated model.

    Model-irrelevant columns are blank (a Lasocka row leaves every `Ea_*`
    empty and a Kissinger row leaves `lasocka_*` empty) — one file, two
    families, no NaN soup. `specimen_keys`, `prior_heats_used` and
    `mass_loss_range_pct` carry the confound NUMERICALLY, so it survives
    someone deleting the message columns.
    """
    fitted = series.fitted_rows()
    rates = [abs(r.q_C_per_min) for r in fitted]
    priors = sorted({int(r.prior_heats) for r in fitted})
    losses = [_f(r.cum_mass_loss_pct) for r in fitted if _finite(r.cum_mass_loss_pct)]
    mass_range = f"{min(losses):.3f}–{max(losses):.3f}" if losses else ""
    n_excluded = sum(1 for r in series.rows if not r.include)
    n_no_value = sum(1 for r in series.rows if not _finite(r.value_C))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    out = []
    for fit in fits:
        allw = list(fit.all_warnings)
        if warnings:
            seen = {(w.code, w.message, tuple(w.row_uids)) for w in allw}
            allw += [w for w in warnings
                     if (w.code, w.message, tuple(w.row_uids)) not in seen]
            allw = sort_warnings(allw)
        base = fit.csv_fit_row()
        rec: Dict[str, Any] = {
            "series_id": series.series_id,
            "series_name": series.name,
            "quantity_kind": series.quantity_kind,
            "n_excluded": n_excluded,
            "n_no_value": n_no_value,
            "rates_used_C_per_min": ";".join(f"{v:g}" for v in rates),
            "n_distinct_rates": len(series.distinct_rates(fitted)),
            "rate_span_ratio": _cell(series.rate_span_ratio(fitted)),
            # Measured from the fitted points, NOT from the absence of the
            # `non_monotonic` warning. That warning needs BOTH an ascending
            # and a descending step to fire, so a series that falls all the
            # way — the one ordering a rate effect cannot produce — left this
            # column reading True. The column a reviewer scans must answer the
            # question its name asks: does the quantity rise with |rate|.
            "monotonic": _is_monotonic_increasing(series, fitted),
            "specimen_keys": ";".join(series.specimen_keys(fitted)),
            "prior_heats_used": ";".join(str(p) for p in priors),
            "mass_loss_range_pct": mass_range,
            "generated_by": GENERATED_BY,
            "generated_at": stamp,
        }
        rec.update(base)
        rec["max_severity"] = max_severity(allw)
        rec["n_warnings"] = len(allw)
        rec["warning_codes"] = ";".join(warning_codes(allw))
        rec["warning_messages"] = " | ".join(w.line() for w in allw)
        out.append({k: rec.get(k, "") for k in FIT_CSV_LEAD})
    return out


def warnings_table(series: RateSeries,
                   warnings: Sequence[SeriesWarning]) -> List[Dict[str, Any]]:
    """One dict per warning, worst first.

    `detail_json` carries the numbers AND the threshold that fired, so a
    reviewer can tell a tripped default apart from a tripped physical limit
    without reading this source file.
    """
    import json
    out = []
    for w in sort_warnings(list(warnings)):
        idxs = [series.index_of(u) for u in w.row_uids]
        out.append({
            "series_id": series.series_id,
            "code": w.code,
            "severity": w.severity,
            "message": w.message,
            "row_indices": ";".join(str(i) for i in idxs if i),
            "row_uids": ";".join(w.row_uids),
            "detail_json": json.dumps(w.detail, default=str, sort_keys=True),
        })
    return out


def _write_csv(records: Sequence[Mapping[str, Any]], columns: Sequence[str],
               path) -> str:
    """Write a table to UTF-8-BOM CSV. pandas is imported HERE, not at module
    import, so a headless science import of this module costs nothing."""
    import pandas as pd
    frame = pd.DataFrame(list(records), columns=list(columns))
    out = str(path)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_series_csv(series: RateSeries, warnings: Optional[Sequence[SeriesWarning]],
                     path) -> str:
    """`<stem>_rows.csv` — one line per row, in table order. Returns the path."""
    return _write_csv(rows_table(series, warnings), ROWS_CSV_COLUMNS, path)


def write_fit_csv(series: RateSeries, fits: Sequence[KineticFitResult],
                  warnings: Optional[Sequence[SeriesWarning]], path) -> str:
    """`<stem>_fit.csv` — one line per evaluated model. Returns the path."""
    return _write_csv(fit_table(series, fits, warnings), FIT_CSV_LEAD, path)


def write_warnings_csv(series: RateSeries, warnings: Sequence[SeriesWarning],
                       path) -> str:
    """`<stem>_warnings.csv` — the optional detail channel. Returns the path."""
    return _write_csv(warnings_table(series, warnings), WARNINGS_CSV_COLUMNS, path)


def _cell_str(rec: Mapping[str, Any], key: str) -> str:
    """One CSV cell as a stripped string, '' when absent."""
    v = rec.get(key, "")
    return "" if v is None else str(v).strip()


def _cell_bool(rec: Mapping[str, Any], key: str, default: bool) -> bool:
    """CSV truthiness: pandas may hand back 'True', 'TRUE', '1' or a real bool."""
    v = _cell_str(rec, key)
    if v == "":
        return default
    return v.lower() in ("true", "1", "yes", "y", "t")


def read_series_csv(path) -> Tuple[RateSeries, List[SeriesWarning]]:
    """Read a `<stem>_rows.csv` back into a series. Never raises on content.

    Rules, each with a reason:
      * Unknown columns are ignored, so a file written by a newer PRISM opens.
      * FRESH uids are minted. A uid identifies a row inside one live series;
        loading the same CSV twice must not produce two rows that every
        uid-keyed lookup confuses.
      * A `value_K` column is converted to Celsius on read and the unit is
        dropped. There is no `temp_unit` column in this schema and no
        `value_unit` field in the model, so kelvin can enter only here and
        leaves immediately.
      * A referenced file that no longer exists keeps its value and gets
        `missing_source_file`: a stored Tg is a MEASUREMENT, not a cache.
    """
    import pandas as pd
    p = str(path)
    try:
        frame = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        series = RateSeries(name=_stem(p))
        return series, [SeriesWarning(
            code="loaded_from_csv", severity="critical",
            message=f"{_stem(p)} could not be read back: {exc} Nothing was loaded.",
            detail={"path": p, "error": str(exc)})]

    records = frame.to_dict(orient="records")
    # The schema FLOOR. "Unknown columns are ignored, so a file written by a
    # newer PRISM opens" is the right rule and it needs a bottom: without one,
    # `item,qty / milk,2 / eggs,12` minted two included rows with a blank
    # sample, no rate and no value, at INFO severity, and the status line said
    # "values are as saved". One recognised column is enough to say this file
    # came from a rate series; none of them means it did not.
    present = {str(c) for c in getattr(frame, "columns", [])}
    if records and not (present & set(ROWS_CSV_REQUIRED_ANY)):
        offered = ", ".join(sorted(ROWS_CSV_REQUIRED_ANY))
        return RateSeries(name=_stem(p)), [SeriesWarning(
            code="loaded_from_csv", severity="critical",
            message=(f"{_stem(p)} does not look like a rate-series rows CSV: none of "
                     f"its {len(present)} column(s) is one of {offered}. Nothing was "
                     f"loaded and the series on screen is unchanged."),
            detail={"path": p, "columns": sorted(present)})]
    series = RateSeries(name=_stem(p) or "Rate series")
    for rec in records:
        row = RateRow()
        for name in ("source_kind", "sample", "specimen_key", "state", "file",
                     "record_id", "segment_kind", "segment_label", "role_in_cycle",
                     "rate_source", "quantity_kind", "method", "signal",
                     "value_source", "error", "note"):
            v = _cell_str(rec, name)
            if v:
                setattr(row, name, v)
        for name in ("segment_index", "cycle_index", "row_start", "row_stop",
                     "n_heating_segments_in_source", "prior_heats"):
            v = _cell_str(rec, name)
            if v:
                try:
                    setattr(row, name, int(float(v)))
                except ValueError:
                    pass
        for name in ("q_measured_C_per_min", "q_programmed_C_per_min",
                     "q_filename_C_per_min", "q_manual_C_per_min", "T0_C",
                     "prior_max_T_C", "cum_mass_loss_pct", "sample_mass_mg"):
            setattr(row, name, _f(_cell_str(rec, name) or None))
        row.window_C = (_f(_cell_str(rec, "window_min_C") or None),
                        _f(_cell_str(rec, "window_max_C") or None))

        value_C = _f(_cell_str(rec, "value_C") or None)
        if not np.isfinite(value_C):
            value_K = _f(_cell_str(rec, "value_K") or None)
            if np.isfinite(value_K):
                value_C = value_K - K0
        row.value_C = value_C
        # The stored value_source is honoured when it is one this version
        # knows; anything else (a foreign writer, a hand-edited cell) is
        # replaced by what the row actually shows. A row that carries a
        # number is "computed" rather than "—", because "—" means "nobody has
        # measured this yet" and that would be a false claim about a file
        # that plainly holds the measurement.
        if row.value_source not in VALUE_SOURCES or (
                row.value_source == "—" and np.isfinite(value_C)):
            row.value_source = "computed" if np.isfinite(value_C) else "—"
        row.include = _cell_bool(rec, "include", True)
        row.is_partial = _cell_bool(rec, "is_partial", False)
        sp = _cell_str(rec, "setpoint_reached")
        row.setpoint_reached = None if sp == "" else sp.lower() in ("true", "1", "yes")
        if row.rate_source not in RATE_SOURCES:
            row.rate_source = "auto"
        if row.source_kind not in ROW_SOURCES:
            row.source_kind = "manual"
        extras = {}
        for part in _cell_str(rec, "extras").split(";"):
            if "=" in part:
                k, _sep, v = part.partition("=")
                extras[k.strip()] = _f(v)
        row.extras = extras
        settings = {}
        for key in SETTINGS_CSV_KEYS:
            v = _cell_str(rec, key)
            if v:
                settings[key] = v
        row.settings = settings
        # A row whose rate only survived as the resolved q_C_per_min column
        # (hand-edited export, foreign writer) keeps that rate as a manual
        # candidate rather than losing it.
        if not np.isfinite(row.q_C_per_min):
            q = _f(_cell_str(rec, "q_C_per_min") or None)
            if np.isfinite(q):
                row.q_manual_C_per_min = q
        series.add(row)

    warns: List[SeriesWarning] = [SeriesWarning(
        code="loaded_from_csv", severity="info",
        message=(f"{len(series.rows)} rows loaded from {_stem(p)}; the values are as "
                 f"saved. Press \"Compute all values\" to recompute them with the "
                 f"current settings."),
        detail={"path": p, "n_rows": len(series.rows)})]
    warns.extend(w for w in _chk_sources(series, "", True)
                 if w.code == "missing_source_file")
    return series, sort_warnings(warns)


def relocate(series: RateSeries, old_prefix: str, new_prefix: str) -> int:
    """Repoint every row's `file` from one directory prefix to another.

    Touches `file` and nothing else — never a value, never an include flag —
    because a moved share does not change a measurement. Matching is
    case-insensitive and normalises the separator, since PRISM is a Windows
    tool and `\\\\Imr-research\\ta\\...` and `//imr-research/ta/...` name the
    same place. Returns how many rows changed.
    """
    old = str(old_prefix or "")
    new = str(new_prefix or "")
    if not old:
        return 0

    def _norm(text: str) -> str:
        """Comparable form of a Windows path: one separator, one case."""
        return text.replace("/", "\\").lower()

    key = _norm(old)
    n = 0
    for row in series.rows:
        if not row.file:
            continue
        if _norm(row.file).startswith(key):
            row.file = new + row.file[len(old):]
            n += 1
    return n
