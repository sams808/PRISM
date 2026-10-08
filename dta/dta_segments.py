"""Segment a run into physically homogeneous ramps and holds, align them to the
programmed method, and hand out analysis-ready arrays.

The whole module exists because a global ``argsort`` by temperature is
meaningless on a cyclic program: on BM-0-cycles.txt it interleaves 8 ramps over
19-1299 C four times, leaving 719 duplicate-temperature pairs and 1002
non-finite derivative values, and collapsing all three Tg methods onto one
number (431.758 C, spread 4.0e-5) while the app congratulates the user on the
agreement.

Four design rules, each of which was measured rather than assumed, and each of
which a future edit can silently undo:

1. **Data-derived segmentation is primary; the program is a labelling layer.**
   Only the measured temperature record decides where a segment starts and
   stops. The programmed method is mapped ONTO those segments afterwards. The
   reason is the real file: its steps `Ramp to 50 C` and the `Isothermal 20 min`
   that follows are ONE continuous passive cool in the data (the furnace never
   reached 50 C), so slicing by the controller clock would produce a "hold"
   running 211 -> 102 C. A Tg, an integral or a derivative on that is a wrong
   number, not a rough one. With no program at all the segmentation is
   unchanged: same 12 segments, same kinds, same 4 cycles.

2. **The rate estimator is conditioned, not just fast.** The obvious O(n)
   prefix-sum formulation of a centred least-squares slope is numerically
   wrong in three separate ways on real records -- 95 % error on a time column
   with a large offset, 2.1 C/min on a long one, and +99.6 C/min on a
   perfectly smooth -10 C/min ramp whose 0.5 min window happens to hold one
   point. All three are fixed in `estimate_rate`, whose docstring names them
   and the measurement for each. `rate_min_points` sits on top of that and
   buys a factor of two in noise, not correctness; the specification credited
   it with the correctness as well, and the measurement says otherwise.

3. **A hold is identified by a FLAT CORE, never by a rate threshold.** The
   measured |dT/dt| inside the four real holds spans 0.000-2.28 C/min, which is
   a strict SUBSET of the range every ramp visits on its way through a
   turnaround or down a passive tail. No threshold, with or without hysteresis,
   can separate them. A stretch is a hold iff it contains ``min_hold_min``
   minutes of temperature spanning at most ``hold_flat_span_C`` -- measured
   13-15 min on the four real holds and 0.3-0.7 min on everything else.

4. **Two row-index spaces exist and they are not interchangeable.**
   ``Segment.row_start`` / ``row_stop`` index the CLEANED arrays (non-finite and
   backwards-in-time rows removed). ``SegmentedRun.row_indices(segment)``
   returns ORIGINAL DataFrame positions and is the only API a DataFrame may be
   sliced with. On all three real files the cleaning drops nothing, so an
   off-by-N here is invisible until the day a file has one NaN in it.

Pure numpy + stdlib. No pandas, no Qt, no scipy. The only project import is
``dta.dta_program``, one way, for the optional programmed-method layer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dta.dta_program import (
    MethodProgram,
    MethodStep,
    STEP_EQUILIBRATE,
    STEP_ISOTHERMAL,
    STEP_JUMP,
    STEP_RAMP,
)

__all__ = [
    "KIND_HEAT", "KIND_COOL", "KIND_ISOTHERMAL", "KIND_EQUILIBRATE",
    "SEGMENT_KINDS",
    "ROLE_EQUILIBRATE", "ROLE_HEAT", "ROLE_COOL", "ROLE_HOLD_HOT",
    "ROLE_HOLD_COLD", "ROLE_HOLD", "SEGMENT_ROLES",
    "X_TEMPERATURE", "X_TIME", "X_KINDS",
    "SegmentScopeError",
    "SegmentationConfig", "Segment", "StepMatch", "ProgramAlignment",
    "PreparedXY", "SegmentedRun", "MAX_SPAN_PER_WINDOW",
    "estimate_rate", "clean_time_temperature", "segment_temperature_program",
    "group_cycles", "align_segments_to_program", "attach_alignment",
    "segment_run", "prepare_segment_xy",
    "segment_label", "describe_segment", "segment_table",
    "format_segment_table", "SEGMENT_TABLE_COLUMNS",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Four-character kind tokens: they line up in a combo box, in the exported CSV
# and in format_segment_table() without padding tricks, which is the whole
# reason "heat"/"cool" beat "heating"/"cooling".
KIND_HEAT = "heat"
KIND_COOL = "cool"
KIND_ISOTHERMAL = "isothermal"
KIND_EQUILIBRATE = "equilibrate"
SEGMENT_KINDS: Tuple[str, ...] = (KIND_EQUILIBRATE, KIND_HEAT, KIND_COOL, KIND_ISOTHERMAL)

# `kind` is what the temperature did; `role` is what the segment MEANS inside
# its cycle. They differ exactly where it matters: two segments can both be
# `isothermal` while one is the 1300 C soak that ends a heat and the other is
# the 50 C wait that ends a cool, and a cycle-to-cycle comparison must never
# confuse them.
ROLE_EQUILIBRATE = "equilibrate"
ROLE_HEAT = "heat"
ROLE_COOL = "cool"
ROLE_HOLD_HOT = "hold_hot"
ROLE_HOLD_COLD = "hold_cold"
ROLE_HOLD = "hold"
SEGMENT_ROLES: Tuple[str, ...] = (ROLE_EQUILIBRATE, ROLE_HEAT, ROLE_HOLD_HOT,
                                  ROLE_COOL, ROLE_HOLD_COLD, ROLE_HOLD)

X_TEMPERATURE = "T"
X_TIME = "t"
X_KINDS: Tuple[str, ...] = (X_TEMPERATURE, X_TIME)

NAN = float("nan")


class SegmentScopeError(ValueError):
    """Asked to produce one sorted curve from something that is not a single
    monotone scan.

    This is the structural half of the guardrail. The other half is a UI that
    disables the temperature abscissa on a hold; but a UI check can be
    forgotten by the next call site, so the science layer refuses as well. A
    caller that ignores the scope gets a loud exception naming the segment
    instead of a silent reproduction of the global-argsort bug.

    Subclasses ValueError so existing ``except ValueError`` handlers in the Qt
    layer keep working and show the message in the status line.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentationConfig:
    """Every knob the segmenter has, with the measured reason for each default.

    Deliberately NOT a knob: a pointwise |dT/dt| tolerance for "is this a
    hold". See `min_hold_min` / `hold_flat_span_C` and the module docstring --
    exposing an `iso_tol` would invite the user to tune a parameter that
    provably cannot do the job, and a value of 2.0 C/min invents two spurious
    hold segments inside the cooling tails of the real file.
    """

    # --- rate estimate
    rate_window_min: float = 0.5
    """Centred least-squares window, in MINUTES not points. One export here is
    0.5 s/pt and another 2.8 s/pt; a point-based window would mean a different
    physical bandwidth on each."""

    rate_min_points: int = 5
    """Minimum samples inside the rate window, enforced by widening in INDEX
    space. Values below 3 disable the guard entirely and restore the two-point
    secant defect -- kept reachable on purpose so a test can demonstrate it."""

    # --- ramp / flat decision
    rate_threshold_C_min: Optional[float] = None
    """Enter-threshold on |dT/dt|. None derives one (see rate_threshold_frac).
    Segmentation of the real run is constant for any value in 1.0-8.0 C/min, so
    this is a safety valve, not a tuning parameter."""

    rate_threshold_frac: float = 0.25
    """Derived threshold = this fraction of the slowest PROGRAMMED ramp rate,
    or of p95(|dT/dt|) when there is no program."""

    rate_threshold_floor_C_min: float = 0.20
    """Never trust a threshold below this: on a file that is one long hold,
    p95(|dT/dt|) is noise and a fraction of noise is still noise."""

    hysteresis_frac: float = 0.60
    """Schmitt-trigger exit threshold = this * enter threshold. Stops a ramp
    from being chopped in two every time the PID breathes."""

    # --- what counts as a real hold
    min_hold_min: float = 2.0
    hold_flat_span_C: float = 1.0
    """The flat-core test: a stretch is a hold iff some window of at least
    `min_hold_min` minutes inside it has a peak-to-peak temperature of at most
    `hold_flat_span_C`. Measured separation on the real run: 13.2-15.2 min of
    flat core on the four true holds against 0.33-0.70 min on all eight ramps
    -- a 19x margin across an empty gap, where the |dT/dt| ranges overlap
    completely."""

    # --- what counts as a real ramp
    min_ramp_min: float = 1.0
    min_ramp_span_C: float = 5.0
    """A ramp must satisfy BOTH (unlike a hold, which needs one flat core).
    Requiring both is what keeps a pure-noise record from fragmenting: noise
    crosses the rate threshold constantly, but a noise excursion never both
    lasts a minute and moves 5 C."""

    # --- corner refinement (extrapolated onset)
    refine_boundaries: bool = True
    refine_ramp_to_hold: bool = True
    """A ramp->hold corner is ROUNDED by the PID settling curve, so the
    threshold crossing lands late and by a variable amount. Intersecting the
    ramp's tangent with the plateau level fixes that: the four 20 min holds of
    the real run come out 19.308/19.267/19.358/19.408 min, spread 0.141 min,
    against spread 1.546 min unrefined."""

    refine_hold_to_ramp: bool = False
    """A hold->ramp corner is a STEP change, and the raw threshold crossing is
    already within 0.03 min of it. Refining it would add a model where there is
    no rounding to model."""

    refine_fit_min: float = 10.0
    refine_guard_min: float = 1.0
    refine_max_shift_min: float = 5.0
    refine_level_quantile: float = 0.05
    """Plateau level = the 95th percentile (after a heat) or 5th (after a cool)
    of the hold's temperature, NOT its median: on a short hold the median still
    sits on the settling curve and drags the corner the wrong way."""

    # --- reporting
    mode_bin_C_min: float = 0.25
    """Histogram bin for `rate_mode`. The mode exists because on cooling
    segment #3 the mean is -14.155 and the median -18.006 against a programmed
    -20; only the mode (-19.960) recovers it. The mean is wrecked by the
    decaying tail and the median by its length."""

    control_tol_frac: float = 0.90
    """|dT/dt| below this fraction of the programmed rate means the furnace is
    no longer following the ramp. Reported as `control_lost_below_C` so nobody
    believes "-20 C/min" applies at 200 C."""

    control_min_tail_frac: float = 0.05
    """A loss of control is only reported when the uncontrolled tail is at
    least this fraction of the segment. EVERY ramp decelerates into the hold
    that follows it -- that is the setpoint arriving, not control being lost --
    and on the real run those corners are 0.5-1.2 % of the ramp while the four
    genuine cooling tails are 25.7-49.4 %. A 5 % cut sits in the middle of a
    20x empty gap, the same kind of margin the flat-core hold test has."""

    def validated(self, dt_median_min: float) -> "SegmentationConfig":
        """Return self, or a copy whose `rate_window_min` is wide enough to hold
        `rate_min_points` samples at this sampling interval.

        WHY this is a separate, opt-in helper and NOT called by
        `segment_temperature_program`: the actual coarse-sampling guard lives in
        `estimate_rate`, which widens the window in INDEX space. Widening in
        TIME space instead would change the estimator's bandwidth wherever the
        sampling is merely uneven, and would therefore change the answer on
        files that never had the problem. This method exists so a UI can show
        the user an honest effective window (a "0.5 min" spin box is a lie on a
        2.8 s/pt file), and so a caller that deliberately disables the index
        guard can still get a usable estimate.
        """
        try:
            dt = float(dt_median_min)
        except (TypeError, ValueError):
            return self
        if not np.isfinite(dt) or dt <= 0.0:
            return self
        need = max(int(self.rate_min_points), 2) - 1
        wanted = need * dt
        if wanted <= float(self.rate_window_min):
            return self
        return replace(self, rate_window_min=float(wanted))


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """One physically homogeneous stretch of the record.

    `row_start` / `row_stop` index the CLEANED arrays, the ones
    `clean_time_temperature` returned. They are NOT DataFrame positions. Use
    `SegmentedRun.row_indices(segment)` for anything that touches the original
    table; on every real file here the two happen to coincide, which is exactly
    why the distinction has to be enforced by the API instead of by care.
    """

    index: int                    # 1-based, TIME order -- never rate or temperature order
    kind: str                     # KIND_*
    role: str                     # ROLE_*
    cycle_index: int              # 1-based
    row_start: int                # [row_start, row_stop) into the CLEANED arrays
    row_stop: int
    n_points: int
    t_start: float
    t_end: float
    T_start: float
    T_end: float                  # traversal order, so T_end < T_start on a cool
    T_min: float
    T_max: float
    rate_mean: float              # (T_end - T_start) / (t_end - t_start), C/min
    rate_median: float            # median of the pointwise estimate
    rate_mode: float              # MODE-refined -- the measured rate to report
    rate_fit_r2: float            # OLS T~t quality: did it actually hold the rate
    control_lost_below_C: float   # NaN when never lost; needs an alignment to mean anything
    monotone_fraction: float      # 1.0 on every real ramp
    n_reversals: int              # rows a stable sort by T would have to move
    n_duplicate_T: int            # points sharing a temperature with another point
    is_partial: bool              # acquisition stopped INSIDE this segment
    step_indices: Tuple[int, ...] = ()        # written step numbers, from the alignment
    step_text: str = ""                       # verbatim programmed text of the block
    rate_nominal_C_min: float = NAN           # PROGRAMMED rate, NaN if unaligned
    target_gap_C: float = NAN                 # C still missing from the programmed target
    note: str = ""                            # "target not reached (94 C)" / "absorbed hold"

    @property
    def duration_min(self) -> float:
        """Wall-clock length of the segment, in minutes."""
        return float(self.t_end) - float(self.t_start)

    @property
    def span_C(self) -> float:
        """Signed temperature traversed: negative on a cool."""
        return float(self.T_end) - float(self.T_start)

    @property
    def slice(self) -> slice:
        """Slice into the CLEANED arrays only. Never slice a DataFrame with it."""
        return slice(int(self.row_start), int(self.row_stop))

    @property
    def is_ramp(self) -> bool:
        """True where temperature is a usable abscissa."""
        return self.kind in (KIND_HEAT, KIND_COOL)

    @property
    def is_hold(self) -> bool:
        """True where temperature is NOT an abscissa and time is the only one."""
        return self.kind in (KIND_ISOTHERMAL, KIND_EQUILIBRATE)

    @property
    def direction(self) -> int:
        """+1 heating, -1 cooling, 0 flat -- the traversal sense of the segment."""
        if self.kind == KIND_HEAT:
            return 1
        if self.kind == KIND_COOL:
            return -1
        return 0

    def to_dict(self) -> dict:
        """JSON-safe dict, NaN rendered as None so json.dumps stays valid."""
        out: Dict[str, Any] = {}
        for name in ("index", "kind", "role", "cycle_index", "row_start", "row_stop",
                     "n_points", "n_reversals", "n_duplicate_T", "is_partial",
                     "step_text", "note"):
            out[name] = getattr(self, name)
        for name in ("t_start", "t_end", "T_start", "T_end", "T_min", "T_max",
                     "rate_mean", "rate_median", "rate_mode", "rate_fit_r2",
                     "control_lost_below_C", "monotone_fraction",
                     "rate_nominal_C_min", "target_gap_C"):
            value = float(getattr(self, name))
            out[name] = None if not np.isfinite(value) else value
        out["step_indices"] = list(self.step_indices)
        return out


# ---------------------------------------------------------------------------
# Alignment result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepMatch:
    """One detected segment paired with the block of programmed steps it ran."""

    segment_index: int                 # 1-based, matches Segment.index
    step_indices: Tuple[int, ...]
    """WRITTEN step indices, as the operator numbered them in the method."""
    programmed_kind: str               # "ramp" | "isothermal" | "ramp+isothermal"
    programmed_rate: float
    """SIGNED by the direction the ramp actually had to travel. TA writes a
    cooling ramp as a POSITIVE rate with a lower target, so the written sign
    carries no information; signing it here is what makes `programmed_rate` and
    `measured_rate` comparable without a second convention."""
    programmed_target: float
    programmed_duration: float
    measured_rate: float
    measured_duration: float
    target_gap_C: float
    absorbed_step_indices: Tuple[int, ...] = ()
    """Steps that ran but left no separate signature in the data -- the four
    `Isothermal 20 min at 50 C` of the real file, which the furnace spent still
    passively cooling. ABSORBED is not the same claim as "not executed"."""
    cost: float = 0.0
    step_text: str = ""                # verbatim programmed lines of the block
    step_positions: Tuple[int, ...] = ()
    """Positions of this block in the PHYSICAL (repeats unrolled) step list.

    Needed alongside `step_indices` because `Repeat segment 1 for 3 times`
    makes four physical passes all carry written index 1: a coverage test that
    de-duplicates by written index reports pass 1 as covering passes 2-4 as
    well, so a run that stopped after one cycle would be declared complete.
    Positions are unique by construction, so they are what coverage is
    measured on; written indices remain what the user is shown."""
    repeat_passes: Tuple[int, ...] = ()
    """`MethodStep.repeat_pass` for each step of the block (0 on pass one)."""


@dataclass(frozen=True)
class ProgramAlignment:
    """Result of mapping the programmed method onto the detected segments."""

    matches: Tuple[StepMatch, ...] = ()
    unmatched_steps: Tuple[int, ...] = ()       # programmed, never observed
    unmatched_segments: Tuple[int, ...] = ()    # observed, not in the program
    total_cost: float = 0.0
    method: str = ""                            # "dp" | "dp-failed" | "no-program"
    programmed_total_min: float = NAN
    measured_total_min: float = NAN
    notes: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the DP ran and every thermal step is accounted for.

        "Every step" counts each PASS of a repeated block separately, so a
        four-cycle method written as one cycle plus `Repeat segment 1 for 3
        times` is only ok when all four passes left a signature.
        """
        return self.method == "dp" and not self.unmatched_steps

    def match_for(self, segment_index: int) -> Optional[StepMatch]:
        """The match for a 1-based segment index, or None when unaligned."""
        for m in self.matches:
            if m.segment_index == segment_index:
                return m
        return None

    @property
    def absorbed_steps(self) -> Tuple[int, ...]:
        """Every programmed step that ran but left no separate signature."""
        out: List[int] = []
        for m in self.matches:
            out.extend(m.absorbed_step_indices)
        return tuple(out)


@dataclass(frozen=True)
class PreparedXY:
    """Analysis-ready arrays for exactly one segment, and how they were made."""

    x: np.ndarray
    y: np.ndarray
    y_alt: Optional[np.ndarray]          # the separate dY source column, when given
    source_index: np.ndarray
    """ORIGINAL row positions, one per returned point. Where duplicates were
    collapsed it names the FIRST row of the group -- y is their mean, and no
    single row can stand for two, so the traceable answer is the one the
    instrument wrote first."""
    x_kind: str                          # X_TEMPERATURE | X_TIME
    was_sorted: bool
    """True when the ascending-temperature rule was applied, INCLUDING when the
    permutation turned out to be the identity. It answers "is x guaranteed
    ascending", not "were any rows moved" -- `Segment.n_reversals` answers the
    second question and does it before the masking."""
    n_duplicates_collapsed: int

    def __len__(self) -> int:
        """Number of prepared points (after masking and duplicate collapse)."""
        return int(self.x.size)

    @property
    def duplicate_fraction(self) -> float:
        """Share of the masked points that shared an x value with another.

        The number the old 1 % guardrail refused on. It is REPORTED now rather
        than raised (see `prepare_segment_xy`), so every consumer that prints a
        value off these arrays can say how coarse the abscissa was. 0.0 when
        nothing collapsed, and 0.0 on an empty curve rather than a NaN that
        would have to be special-cased by every caller.
        """
        total = int(self.n_duplicates_collapsed) + int(self.x.size)
        if total <= 0:
            return 0.0
        return float(self.n_duplicates_collapsed) / float(total)


# ---------------------------------------------------------------------------
# estimate_rate
# ---------------------------------------------------------------------------
MAX_SPAN_PER_WINDOW = 500.0
"""Longest record, in units of the rate window, that `estimate_rate` fits in
one pass before splitting into overlapping chunks. See `_ls_slope_one_pass`."""


def _ls_slope_one_pass(t: np.ndarray, T: np.ndarray, hw: float,
                       min_points: int) -> np.ndarray:
    """Centred windowed least-squares slope over the WHOLE of t, in O(n).

    Assumes `t` is finite and strictly increasing. Prefix sums make it O(n);
    the accuracy of that trick is what the chunking in `estimate_rate` exists
    to protect, and the two comments below are the whole reason this is not
    three lines of `np.cumsum`.
    """
    n = int(t.size)
    lo = np.searchsorted(t, t - hw, side="left")
    hi = np.searchsorted(t, t + hw, side="right")
    if min_points >= 3:
        # Widen in INDEX space so a window that is nearly empty still holds a
        # line's worth of points. Never in time space: that would change the
        # estimator's bandwidth on files that never had the problem.
        k = min_points // 2
        idx = np.arange(n)
        lo = np.minimum(lo, np.maximum(idx - k, 0))
        hi = np.maximum(hi, np.minimum(idx + k + 1, n))

    # Centre and scale BOTH axes before the prefix sums. The slope is invariant
    # to a shift in t or T and scales by 1/half_span, but the arithmetic is
    # not: `den = N*Sxx - Sx**2` is a difference of two nearly equal numbers
    # whose size is set by t itself while the answer's size is set by the
    # variance INSIDE the window. Measured on a perfect 20 C/min ramp summing
    # raw t: 4.4e-4 C/min error over 890 min, but 18.9 C/min -- a 95 % error --
    # as soon as the time column merely STARTS at 1e5 min instead of 0. The
    # scratchpad prototype this module descends from summed raw t, and all
    # three real files here start near zero and run under 900 min, which is
    # the only reason that was invisible.
    t_mid = 0.5 * (float(t[0]) + float(t[-1]))
    t_half = 0.5 * (float(t[-1]) - float(t[0]))
    if not np.isfinite(t_half) or t_half <= 0.0:
        t_half = 1.0
    u = (t - t_mid) / t_half
    T_mid = 0.5 * (float(np.min(T)) + float(np.max(T)))
    if not np.isfinite(T_mid):
        T_mid = 0.0
    v = T - T_mid

    zero = np.zeros(1, dtype=float)
    c_x = np.concatenate((zero, np.cumsum(u)))
    c_xx = np.concatenate((zero, np.cumsum(u * u)))
    c_y = np.concatenate((zero, np.cumsum(v)))
    c_xy = np.concatenate((zero, np.cumsum(u * v)))

    N = (hi - lo).astype(float)
    Sx = c_x[hi] - c_x[lo]
    Sxx = c_xx[hi] - c_xx[lo]
    Sy = c_y[hi] - c_y[lo]
    Sxy = c_xy[hi] - c_xy[lo]
    den = N * Sxx - Sx * Sx

    out = np.asarray(np.gradient(T, t), dtype=float)
    # A RELATIVE degeneracy test, plus the arithmetic fact that a line needs
    # two points. The absolute `|den| > 1e-12` the prototype used is
    # meaningless here: den is the residue of a cancellation between two terms
    # of size N*Sxx, so the only honest question is whether it rose above that
    # cancellation's own rounding noise -- and for N == 1 its true value is
    # exactly zero, so whatever survives is pure roundoff. Measured on a
    # perfectly smooth -10 C/min cooling ramp sampled at 0.28 min/pt, where
    # every 0.5 min window holds one point: the absolute test let that
    # roundoff through as +99.6 C/min. With these two tests the same point
    # falls through to the plain difference and reads -9.969.
    good = (N >= 2.0) & (np.abs(den) > 1e-11 * np.maximum(N * np.abs(Sxx), 1e-300))
    out[good] = ((N[good] * Sxy[good] - Sx[good] * Sy[good])
                 / (den[good] * t_half))
    return out


def estimate_rate(time_min, temp_C, *, window_min: float = 0.5,
                  min_points: int = 5) -> np.ndarray:
    """dT/dt (C/min) by centred least squares over a fixed TIME window, widened
    in index space to hold at least `min_points` samples. O(n) via prefix sums.

    The window is in MINUTES because one export here is 0.5 s/pt and another is
    2.8 s/pt, and the same setting must mean the same physical smoothing on
    both.

    WHAT THIS FUNCTION IS ACTUALLY GUARDING AGAINST, measured rather than
    inherited. The specification this module was built to said that coarse
    sampling shatters the 12-segment run into 72+ pieces with rates to
    -357 C/min, and prescribed `min_points` as the cure. The symptom is real
    and reproduces; the diagnosis was wrong. The cause is NUMERICAL, and there
    are three separate faults in the prefix-sum formulation the scratchpad
    prototype used:

      1. It summed RAW t. `den = N*Sxx - Sx**2` is a difference of two nearly
         equal numbers whose size is set by t while the answer's size is set
         by the variance INSIDE the window. On a perfect 20 C/min ramp that
         costs 4.4e-4 C/min over 890 min starting at t = 0, but 18.9 C/min --
         a 95 % error -- as soon as the time column merely STARTS at 1e5 min.
         Fixed by centring and scaling both axes (`_ls_slope_one_pass`).
      2. One global prefix sum over a long record. The stored values reach the
         whole-record total while a window holds a few dozen samples, so the
         difference is already 5 digits short before the least squares begins:
         2.1 C/min of error on 400 000 points over 20 000 min. Fixed by
         fitting long records in overlapping chunks, which is exact because
         the estimator is local.
      3. An ABSOLUTE degeneracy floor, `|den| > 1e-12`. When a window holds one
         point den's true value is exactly zero, so everything above that floor
         was roundoff. That is where the spectacular numbers came from: on a
         perfectly smooth -10 C/min cooling ramp sampled at 0.28 min/pt the
         prototype's formulation reports +99.6 C/min. Fixed by requiring two
         points and testing den against its own cancellation noise.

    With all three fixed the estimator is accurate to 4e-5 C/min from 2 000 to
    1 000 000 points, at any time offset -- measured worst case 3.7e-5 C/min, on
    a million points at 0.001 min/pt, which is 1.9 ppm of a 20 C/min ramp and
    2e-4 of the smallest threshold the segmenter will use. The residual is
    prefix-sum cancellation and it grows with the point count and with 1/dt: at
    the 0.046-2.8 min/pt sampling of every real export here it is below 1e-8.
    (An earlier draft of this docstring claimed 4e-6; that bound holds only
    below ~1e5 points and an adversarial pass measured 4.6e-6 above it.)
    The segmentation of the real run
    is 12 correct segments at EVERY decimation from 0.046 to 3.0 min/pt with
    `min_points` at 5 or at 0.

    So what does `min_points` still buy? Noise, which is the honest answer and
    a smaller one. Whenever the time window holds fewer than `min_points`
    samples the fit is widened in INDEX space to reach them, and on a 20 C/min
    ramp carrying +/-0.3 C of noise that halves the scatter of the estimate:
    pointwise standard deviation 0.53 / 0.26 / 0.13 C/min with the floor
    against 1.22 / 0.54 / 0.31 without it at dt = 0.19 / 0.38 / 0.75 min, and
    worst-case error 2.0-2.4x smaller. Where the window is already full it is
    bit-for-bit a no-op.

    The widening is in INDEX space, not time space, on purpose: growing the
    time window would change the estimator's bandwidth wherever sampling is
    merely uneven, and so would change the answer on files that never had the
    problem. Here it only ever ADDS samples to a window too empty to fit a
    line. Its one limit is that a widened window must stay narrower than the
    shortest feature: at dt >= 4.5 min the +/-2 sample widening spans 18 min,
    which is wider than this run's 19 min holds and starts to smear them. That
    is 100x coarser than any TA SDT export and 24x coarser than the committed
    fixture.

    `min_points` below 3 disables the widening. That door is left open so a
    test can measure what it is worth instead of asserting it.

    Savitzky-Golay is rejected for this job: a fixed `delta` on non-uniform
    sampling biases the rate by exactly mean(dt)/median(dt) - 1, which is
    +2.12 % on the real file (and +122 % on a deliberately bimodal sampling
    grid, where this estimator is still exact to 1e-6). Smoothing first and
    then differencing is rejected too: many times the noise of the windowed
    fit, for the same bandwidth.
    """
    t = _as_float_array(time_min)
    T = _as_float_array(temp_C)
    n = int(t.size)
    if n != int(T.size):
        # Never raise: a mismatched pair is a caller bug we report as "no rate
        # information", so a batch loop does not lose every other record. The
        # same promise is why the coercion above tolerates an object-dtype
        # column instead of throwing on it.
        return np.full(max(n, int(T.size)), NAN)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)

    finite = np.isfinite(t) & np.isfinite(T)
    needs_clean = (not bool(finite.all())
                   or not bool(np.all(np.diff(t[finite]) > 0.0)))
    if needs_clean:
        tc, Tc, keep = clean_time_temperature(t, T)
        out = np.full(n, NAN)
        if keep.size:
            out[keep] = estimate_rate(tc, Tc, window_min=window_min,
                                      min_points=min_points)
        return out

    try:
        hw = max(float(window_min), 0.0) / 2.0
    except (TypeError, ValueError):
        hw = 0.25
    if not np.isfinite(hw):
        hw = 0.25

    if hw <= 0.0:
        # Explicit "no window" request: plain finite differences.
        return np.asarray(np.gradient(T, t), dtype=float)

    try:
        mp = int(min_points)
    except (TypeError, ValueError):
        mp = 5
    k_pad = mp // 2 if mp >= 3 else 0

    span = float(t[-1] - t[0])
    chunk_span = MAX_SPAN_PER_WINDOW * 2.0 * hw
    if not np.isfinite(span) or span <= chunk_span:
        return _ls_slope_one_pass(t, T, hw, mp)

    # A long record is fitted in overlapping chunks. The estimator is LOCAL --
    # no point's window reaches further than hw minutes or k_pad samples -- so
    # a chunk carrying that much halo on each side reproduces the single-pass
    # answer for every point it owns, while keeping the prefix sums short
    # enough to stay accurate. Measured on a perfect 20 C/min ramp of 400 000
    # points over 20 000 min with a 0.5 min window: 2.1 C/min error in one
    # pass, 1e-6 chunked. Records that long do not occur in a TA SDT export,
    # but `estimate_rate` is public and an estimator that is silently wrong on
    # a big file is worse than a slow one.
    out = np.empty(n, dtype=float)
    start = 0
    while start < n:
        stop = int(np.searchsorted(t, t[start] + chunk_span, side="right"))
        stop = min(max(stop, start + 1), n)
        # The halo is at least one sample wide even when the time window and
        # the index widening would both allow zero, because the np.gradient
        # fallback inside the core uses a one-sided formula at an array edge
        # and a central one elsewhere; without the extra sample a chunk's
        # first owned point could silently get the wrong one of the two.
        pad = max(k_pad, 1)
        first = int(np.searchsorted(t, t[start] - hw, side="left"))
        first = max(0, min(first, start - pad))
        last = int(np.searchsorted(t, t[stop - 1] + hw, side="right"))
        last = min(n, max(last, stop + pad))
        piece = _ls_slope_one_pass(t[first:last], T[first:last], hw, mp)
        out[start:stop] = piece[start - first:stop - first]
        start = stop
    return out


# ---------------------------------------------------------------------------
# clean_time_temperature
# ---------------------------------------------------------------------------
def _as_float_array(values) -> np.ndarray:
    """A flat float array, with unparseable entries as NaN instead of a raise.

    `np.asarray(x, dtype=float)` throws on an object-dtype column, and an
    object-dtype time or temperature column is not exotic: a run that was
    restarted and appended a second header block, or an export written with a
    decimal comma, arrives exactly that way. Everything downstream already
    drops non-finite rows and reports them through `keep_index`, so coercing
    the junk to NaN puts the bad rows in the channel that already exists --
    whereas raising here breaks `segment_run`'s documented "never raises on
    data" promise before its own guards are even reached.
    """
    try:
        return np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        pass
    try:
        raw = np.asarray(values, dtype=object).ravel()
    except Exception:
        return np.zeros(0, dtype=float)
    out = np.full(int(raw.size), NAN, dtype=float)
    for i, item in enumerate(raw):
        try:
            out[i] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def clean_time_temperature(time_min, temp_C
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite, strictly-increasing-in-time view of a run, plus `keep_index`.

    Returns ``(t, T, keep_index)`` where `keep_index` maps every cleaned row back
    to its ORIGINAL position. Rows with a non-finite time or temperature are
    dropped; if the remaining timestamps are not strictly increasing they are
    stably sorted and duplicate/backwards timestamps are dropped (a repeated
    timestamp makes every derivative infinite, and a stable sort keeps the first
    occurrence, which is the one the instrument wrote).

    On all three real TA exports here `keep_index` is ``arange(n)``. That is
    precisely why it is returned rather than assumed: every segment boundary,
    every DataFrame slice and every exported row index shifts by the number of
    dropped rows, and with an identity mapping on the test files an off-by-N
    would never show up.
    """
    t = _as_float_array(time_min)
    T = _as_float_array(temp_C)
    n = min(int(t.size), int(T.size))
    if int(t.size) != int(T.size):
        t, T = t[:n], T[:n]
    if n == 0:
        empty_i = np.zeros(0, dtype=np.intp)
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), empty_i

    idx = np.flatnonzero(np.isfinite(t) & np.isfinite(T)).astype(np.intp)
    if idx.size > 1 and not bool(np.all(np.diff(t[idx]) > 0.0)):
        idx = idx[np.argsort(t[idx], kind="mergesort")]
        idx = idx[np.concatenate(([True], np.diff(t[idx]) > 0.0))]
    return t[idx], T[idx], idx


#: Multiplier that turns one tick of a clock column into MINUTES, keyed by the
#: unit written in the column's own label. Every rate in this module is per
#: MINUTE, so a `Time (s)` column read as minutes puts a 60x error straight
#: into `rate_mode` and from there into every Lasocka and Arrhenius slope --
#: measured at 4.9x in B, with the wrong sign, on a two-file mixed-unit series.
#: The unit is in the label the instrument wrote, so the check is available and
#: there is no reason not to make it.
CLOCK_UNIT_SCALES: Tuple[Tuple[Tuple[str, ...], float, str], ...] = (
    (("ms", "msec", "millisecond", "milliseconds"), 1.0 / 60000.0, "ms"),
    (("s", "sec", "secs", "second", "seconds"), 1.0 / 60.0, "s"),
    (("min", "mins", "minute", "minutes"), 1.0, "min"),
    (("h", "hr", "hrs", "hour", "hours"), 60.0, "h"),
)


def clock_scale_to_min(col_name: Any) -> Tuple[float, str]:
    """`(scale, unit)` turning this clock column's values into MINUTES.

    Reads the unit out of the parenthesis of the column label -- `Time (min)`,
    `Time (sec)`, `Time (s)`, `time [h]` -- and returns `(1.0, "")` when there
    is no unit to read, which is the only honest answer and preserves the
    historic "assume minutes" behaviour for a bare `Time` column.

    A scale of exactly 1.0 with a named unit ("min") and a scale of 1.0 with no
    unit are deliberately distinguishable by the second element: a caller that
    wants to WARN about a converted clock must not warn about a column that
    was already in minutes.
    """
    text = str(col_name or "").lower()
    inner = ""
    for opener, closer in (("(", ")"), ("[", "]")):
        a = text.rfind(opener)
        b = text.rfind(closer)
        if 0 <= a < b:
            inner = text[a + 1:b].strip()
            break
    if not inner:
        return 1.0, ""
    token = inner.replace(".", "").strip()
    for names, scale, unit in CLOCK_UNIT_SCALES:
        if token in names:
            return float(scale), unit
    return 1.0, ""


def clock_to_minutes(values, col_name: Any) -> Tuple["np.ndarray", float, str]:
    """`(minutes, scale, unit)` -- the clock column in MINUTES. See `clock_scale_to_min`."""
    scale, unit = clock_scale_to_min(col_name)
    arr = _as_float_array(values)
    return (arr if scale == 1.0 else arr * scale), scale, unit


# ---------------------------------------------------------------------------
# Internals: threshold, state machine, run surgery
# ---------------------------------------------------------------------------
def _resolve_threshold(rate: np.ndarray, cfg: SegmentationConfig,
                       program: Optional[MethodProgram]) -> float:
    """Enter-threshold on |dT/dt|, in C/min.

    Preference order: an explicit config value, then a quarter of the SLOWEST
    programmed ramp rate, then a quarter of p95(|dT/dt|) in the data. Deriving
    from the slowest programmed rate is what makes one default work on a 1 C/min
    creep experiment and on a 50 C/min quench; the p95 fallback is what makes it
    work with no program at all, and on the real file both land inside the
    1.0-8.0 C/min plateau over which the segmentation does not change at all.

    **The floor applies to the p95 path ONLY**, and that is the whole point of
    it: its stated reason is "on a file that is one long hold, p95(|dT/dt|) is
    noise and a fraction of noise is still noise". A PROGRAMMED rate is not
    noise -- it is the file's own statement of what it was asked to do. Letting
    a 0.20 C/min floor override it classified a 0.10 C/min ramp whose header
    reads `Ramp 0.10 C/min to 75.00 C` as a HOLD: no rate, no temperature
    abscissa, no Tg, and a banner asserting the file "contains no ramp" while
    `program.ramp_rates == (0.1,)` sat in the same object. None of the three
    knobs the UI exposes could reach it. Since 0.1-1 K/min is the standard scan
    range for enthalpy-relaxation and fictive-temperature work, that closed off
    the exact domain the Lasocka feature exists for (MERGED_DECISIONS M0.1).
    The programmed path keeps only a positivity guard, so a method claiming a
    zero or negative rate still falls back to something usable.
    """
    if cfg.rate_threshold_C_min is not None:
        try:
            explicit = float(cfg.rate_threshold_C_min)
        except (TypeError, ValueError):
            explicit = NAN
        if np.isfinite(explicit) and explicit > 0.0:
            return explicit
    scale: Optional[float] = None
    if program is not None:
        try:
            rates = tuple(r for r in program.ramp_rates if np.isfinite(r) and r > 0.0)
        except Exception:
            rates = ()
        if rates:
            scale = float(min(rates))
    if scale is not None:
        derived = float(cfg.rate_threshold_frac) * scale
        return derived if derived > 0.0 else float(cfg.rate_threshold_floor_C_min)
    a = np.abs(rate[np.isfinite(rate)])
    scale = float(np.percentile(a, 95)) if a.size else 0.0
    return float(max(cfg.rate_threshold_floor_C_min, cfg.rate_threshold_frac * scale))


def _hysteretic_state(rate: np.ndarray, r_on: float, r_off: float) -> np.ndarray:
    """+1 heating / -1 cooling / 0 flat, with Schmitt-trigger hysteresis.

    A point is hard-assigned when |rate| >= r_on. Otherwise it keeps the last
    hard state as long as the sign still matches and |rate| >= r_off. Without
    the hysteresis a single PID breath below the threshold cuts a 60 min ramp
    into three segments. Fully vectorised: the "last hard state" is a running
    maximum over positions, not a Python loop.
    """
    n = int(rate.size)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    hard = np.where(rate >= r_on, 1, np.where(rate <= -r_on, -1, 0)).astype(np.int8)
    pos = np.where(hard != 0, np.arange(n), -1)
    last = np.maximum.accumulate(pos)
    last_hard = np.where(last >= 0, hard[np.maximum(last, 0)], 0).astype(np.int8)
    keep = (last_hard != 0) & (np.sign(rate) == last_hard) & (np.abs(rate) >= r_off)
    return np.where(hard != 0, hard, np.where(keep, last_hard, 0)).astype(np.int8)


def _runs(state: np.ndarray) -> List[List[int]]:
    """Contiguous constant-state runs as mutable ``[start, stop, state]`` lists."""
    if state.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(state)) + 1
    bounds = np.concatenate(([0], cuts, [state.size]))
    return [[int(bounds[k]), int(bounds[k + 1]), int(state[bounds[k]])]
            for k in range(len(bounds) - 1)]


def _longest_flat_core_min(t: np.ndarray, T: np.ndarray, span_C: float) -> float:
    """Longest time window whose peak-to-peak temperature is <= `span_C`.

    Two-pointer sweep with monotonic deques, O(n). This is the measurement the
    hold test is built on, so it must be exact rather than a sampled
    approximation: the separation it produces (13-15 min on a real hold against
    0.3-0.7 min on a ramp turnaround) is the only thing standing between a
    20 min soak and the tail of a passively cooling furnace.
    """
    n = int(t.size)
    if n == 0:
        return 0.0
    dmax: deque = deque()
    dmin: deque = deque()
    best = 0.0
    lo = 0
    for hi in range(n):
        while dmax and T[dmax[-1]] <= T[hi]:
            dmax.pop()
        dmax.append(hi)
        while dmin and T[dmin[-1]] >= T[hi]:
            dmin.pop()
        dmin.append(hi)
        while T[dmax[0]] - T[dmin[0]] > span_C and lo < hi:
            lo += 1
            if dmax[0] < lo:
                dmax.popleft()
            if dmin[0] < lo:
                dmin.popleft()
        if T[dmax[0]] - T[dmin[0]] <= span_C:
            best = max(best, float(t[hi] - t[lo]))
    return float(best)


def _is_real_hold(t: np.ndarray, T: np.ndarray, i0: int, i1: int,
                  cfg: SegmentationConfig) -> bool:
    """A hold is a stretch where the temperature was actually HELD.

    It must contain a window of at least `min_hold_min` minutes whose
    peak-to-peak temperature is at most `hold_flat_span_C`. This single test
    replaces the usual duration/span heuristics, and it is the only thing that
    separates a genuine 20 min plateau from the slow tail of a passively cooling
    furnace: their pointwise |dT/dt| ranges overlap completely (holds 0.000-2.28
    C/min, contained inside every ramp's range), their flat cores do not
    (13-15 min against 0.3-0.7 min).
    """
    if t[i1 - 1] - t[i0] < cfg.min_hold_min:
        return False
    return _longest_flat_core_min(t[i0:i1], T[i0:i1],
                                  cfg.hold_flat_span_C) >= cfg.min_hold_min


def _is_real_ramp(t: np.ndarray, T: np.ndarray, i0: int, i1: int,
                  cfg: SegmentationConfig, state: int = 0) -> bool:
    """A ramp must last `min_ramp_min` AND move `min_ramp_span_C`, in the
    direction it claims.

    All three, not any one. A noise record crosses any rate threshold
    constantly and produces runs that are long OR large, but never both and
    rarely in a consistent direction; requiring all three is what keeps
    `segment_temperature_program` from fragmenting a flat noisy record into
    dozens of pseudo-ramps. Real ramps clear the first two bars by two orders
    of magnitude (60+ min, 1000+ C) and the third by construction.
    """
    long_enough = (t[i1 - 1] - t[i0]) >= cfg.min_ramp_min
    delta = float(T[i1 - 1] - T[i0])
    big_enough = abs(delta) >= cfg.min_ramp_span_C
    right_way = (state == 0) or (delta * float(state) > 0.0)
    return bool(long_enough and big_enough and right_way)


def _dissolve(runs: List[List[int]], t: np.ndarray, T: np.ndarray,
              cfg: SegmentationConfig) -> List[List[int]]:
    """Remove runs that are neither a real hold nor a real ramp by giving their
    points to a neighbour, then merge neighbours of equal state.

    Iterated to a fixed point because dissolving one run can make the merged
    result fail its own test. A run between two ramps of OPPOSITE sign is a
    turnaround: it is split at the extremum so each half joins the ramp it
    belongs to, rather than being dumped wholesale into one of them. Runs only
    ever disappear, so the loop terminates.
    """
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k in range(len(runs)):
            i0, i1, s = runs[k]
            if s == 0:
                if _is_real_hold(t, T, i0, i1, cfg):
                    continue
            else:
                if _is_real_ramp(t, T, i0, i1, cfg, s):
                    continue
            left = runs[k - 1] if k > 0 else None
            right = runs[k + 1] if k + 1 < len(runs) else None
            if left is None and right is None:
                continue
            sub = T[i0:i1]
            dT = float(sub[-1] - sub[0])
            if left is not None and right is not None and left[2] * right[2] == -1:
                j = int(np.argmin(sub)) if left[2] < 0 else int(np.argmax(sub))
                cut = int(np.clip(i0 + j + 1, left[0] + 1, right[1] - 1))
                left[1] = cut
                right[0] = cut
            elif left is not None and right is not None:
                same = [nb for nb in (left, right)
                        if nb[2] != 0 and nb[2] == int(np.sign(dT))]
                if len(same) == 1:
                    target = same[0]
                else:
                    left_dur = t[left[1] - 1] - t[left[0]]
                    right_dur = t[right[1] - 1] - t[right[0]]
                    target = left if left_dur >= right_dur else right
                if target is left:
                    left[1] = i1
                else:
                    right[0] = i0
            elif left is not None:
                left[1] = i1
            else:
                right[0] = i0
            runs.pop(k)
            changed = True
            break
        if changed:
            continue
        for k in range(len(runs) - 1):
            if runs[k][2] == runs[k + 1][2]:
                runs[k][1] = runs[k + 1][1]
                runs.pop(k + 1)
                changed = True
                break

    # A run can survive `_is_real_ramp` failing simply because it had nowhere
    # to be dissolved into -- a pure-noise record collapses to exactly one run,
    # and whichever sign the state machine happened to leave on it would then
    # be published as "the whole file is a cooling ramp". Anything that is not
    # a real ramp is demoted to flat, which is the honest residual class:
    # `isothermal` here means "no ramp was established", not "the temperature
    # was constant".
    for run in runs:
        if run[2] != 0 and not _is_real_ramp(t, T, run[0], run[1], cfg, run[2]):
            run[2] = 0
    k = 0
    while k < len(runs) - 1:
        if runs[k][2] == runs[k + 1][2]:
            runs[k][1] = runs[k + 1][1]
            runs.pop(k + 1)
        else:
            k += 1
    return runs


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares ``y = m x + b`` without numpy's polyfit warnings."""
    n = float(x.size)
    if n < 2:
        return 0.0, float(y[0]) if y.size else 0.0
    Sx = float(x.sum())
    Sy = float(y.sum())
    Sxx = float((x * x).sum())
    Sxy = float((x * y).sum())
    den = n * Sxx - Sx * Sx
    if abs(den) < 1e-12:
        return 0.0, float(Sy / n)
    m = (n * Sxy - Sx * Sy) / den
    return float(m), float((Sy - m * Sx) / n)


def _fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination of a straight line through (x, y).

    Reported per segment as `rate_fit_r2`: it answers "did the furnace actually
    HOLD this rate", which neither the mean nor the mode can. A cooling segment
    that lost control halfway shows a high |rate_mode| and a visibly degraded
    R-squared, and that pair is the honest description.
    """
    if x.size < 3:
        return NAN
    m, b = _fit_line(x, y)
    resid = y - (m * x + b)
    ss_res = float(np.sum(resid * resid))
    centred = y - float(np.mean(y))
    ss_tot = float(np.sum(centred * centred))
    if not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return NAN
    return float(1.0 - ss_res / ss_tot)


def _clamped_boundary(new_cut: int, a: List[int], b: List[int]) -> Optional[int]:
    """Where to put the boundary between runs `a` and `b`, or None to leave it.

    The refinement must never eat a whole run: each side keeps at least three
    samples, so the accepted window is ``[a[0] + 3, b[1] - 3]``. On a coarsely
    sampled record that window can be EMPTY (two adjacent runs spanning fewer
    than six samples between them), and a bare ``np.clip`` with a_min > a_max
    silently returns a_max -- which here is BELOW ``a[0]`` and would give `a` a
    negative length, i.e. a segment whose `n_points` is negative and whose
    `_make_segment` indexes an empty slice. There is no sensible boundary to
    move to in that case, so the honest answer is to keep the one the state
    machine found.

    None also means "no change": `_refine_ramp_hold_corner` returns the
    ORIGINAL cut when the construction is untrustworthy, and answering None
    there makes "the refinement declined" and "the window is unusable" one
    case for the caller instead of two. Whenever the window IS usable and the
    cut moves, both sides are left with at least three samples, so the tiling
    stays valid by construction rather than by luck.
    """
    lo, hi = a[0] + 3, b[1] - 3
    if hi < lo:
        return None
    cut = int(min(max(int(new_cut), lo), hi))
    return None if cut == a[1] else cut


def _refine_ramp_hold_corner(t: np.ndarray, T: np.ndarray, ramp: List[int],
                             hold: List[int], cfg: SegmentationConfig,
                             ramp_is_left: bool) -> int:
    """Extrapolated-onset corner: intersect the ramp's tangent with the hold's
    level.

    Both pieces are fitted AWAY from the corner (`refine_guard_min`) so the PID
    settling curve biases neither. This is the standard thermal-analysis onset
    construction, and it is what makes the four programmed 20 min holds come out
    within 0.141 min of each other instead of 1.546 min. Returns the original
    cut unchanged whenever the construction is not trustworthy -- too few
    points, a flat "ramp", or a shift larger than `refine_max_shift_min`.
    """
    ri0, ri1 = ramp[0], ramp[1]
    hi0, hi1 = hold[0], hold[1]
    cut = ramp[1] if ramp_is_left else hold[1]
    tc = float(t[cut]) if cut < t.size else float(t[-1])
    if ramp_is_left:
        b_hi = np.searchsorted(t, tc - cfg.refine_guard_min, side="right")
        b_lo = np.searchsorted(t, tc - cfg.refine_guard_min - cfg.refine_fit_min,
                               side="left")
    else:
        b_lo = np.searchsorted(t, tc + cfg.refine_guard_min, side="left")
        b_hi = np.searchsorted(t, tc + cfg.refine_guard_min + cfg.refine_fit_min,
                               side="right")
    b_lo, b_hi = max(int(b_lo), ri0), min(int(b_hi), ri1)
    if b_hi - b_lo < 5 or hi1 - hi0 < 5:
        return cut
    m, b = _fit_line(t[b_lo:b_hi], T[b_lo:b_hi])
    if abs(m) < 1e-9:
        return cut
    # Plateau level = the value the furnace actually settled to, taken as a
    # robust extreme in the ramp's direction. NOT the median: on a short hold
    # the median still sits on the settling curve and drags the corner the
    # wrong way, which is the bias the whole construction exists to remove.
    q = 1.0 - cfg.refine_level_quantile if m > 0 else cfg.refine_level_quantile
    level = float(np.quantile(T[hi0:hi1], q))
    tstar = (level - b) / m
    if not np.isfinite(tstar) or abs(tstar - tc) > cfg.refine_max_shift_min:
        return cut
    return int(np.clip(np.searchsorted(t, tstar, side="left"), 1, t.size - 1))


def _mode_rate(r: np.ndarray, bin_w: float) -> float:
    """Most frequent rate (in `bin_w` bins) refined by the median inside that bin
    plus one bin either side.

    The measured rate to report. On cooling segment #3 of the real file the mean
    is -14.155 C/min and the median -18.006 against a programmed -20; the mode
    is -19.960. The mean is destroyed by the decaying tail of a furnace that has
    stopped following the setpoint, the median merely wounded by its length --
    but the controlled part is still the single most POPULATED rate, which is
    exactly what a mode finds.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return NAN
    lo, hi = float(r.min()), float(r.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return NAN
    width = float(bin_w) if np.isfinite(bin_w) and bin_w > 0 else 0.25
    if hi - lo < width:
        return float(np.median(r))
    edges = np.arange(lo, hi + width, width)
    if edges.size < 2:
        return float(np.median(r))
    counts, _ = np.histogram(r, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centres[int(np.argmax(counts))])
    sel = r[np.abs(r - mode) <= width]
    return float(np.median(sel)) if sel.size else mode


def _reversals_and_duplicates(T: np.ndarray, direction: int) -> Tuple[int, int]:
    """(rows a stable sort in the segment's OWN direction would move, points
    sharing a temperature with another point).

    `n_reversals` is deliberately defined as the number of positions a stable
    argsort changes, not as the number of sign flips in dT: it answers the
    question the caller actually has, which is "how far is this segment from
    being usable with temperature as the abscissa".

    The sort is in the segment's own traversal direction -- DESCENDING on a
    cool. A perfectly monotone cooling ramp is a perfectly good abscissa (you
    read it backwards), so scoring it against an ascending order would report
    every one of its rows as out of place and make the number useless exactly
    where it matters. Measured on the fixture: 0-2 on all eight ramps, 4-13 on
    the four holds, whose temperature order has nothing to do with time.
    """
    n = int(T.size)
    if n == 0:
        return 0, 0
    if direction == 0:
        direction = 1 if float(T[-1]) >= float(T[0]) else -1
    key = T if direction > 0 else -T
    order = np.argsort(key, kind="stable")
    n_rev = int(np.count_nonzero(order != np.arange(n)))
    n_dup = n - int(np.unique(T).size)
    return n_rev, n_dup


def _monotone_fraction(T: np.ndarray, direction: int) -> float:
    """Fraction of consecutive steps that move in the segment's own direction.

    1.0 on every real ramp of the real file, which is the property that makes a
    per-segment sort a no-op and the global sort a bug. On a hold it is
    meaningless by construction and lands near 0.5, which is the honest answer:
    a hold has no direction.
    """
    n = int(T.size)
    if n < 2:
        return 1.0
    d = np.diff(T)
    if direction > 0:
        return float(np.count_nonzero(d > 0) / d.size)
    if direction < 0:
        return float(np.count_nonzero(d < 0) / d.size)
    span = float(T[-1] - T[0])
    if span >= 0:
        return float(np.count_nonzero(d > 0) / d.size)
    return float(np.count_nonzero(d < 0) / d.size)


def _make_segment(index: int, kind: str, i0: int, i1: int, t: np.ndarray,
                  T: np.ndarray, rate: np.ndarray, n_total: int,
                  cfg: SegmentationConfig) -> Segment:
    """Build one Segment from cleaned-array bounds ``[i0, i1)``."""
    sub_T = T[i0:i1]
    sub_t = t[i0:i1]
    sub_r = rate[i0:i1]
    duration = float(sub_t[-1] - sub_t[0]) if sub_t.size else 0.0
    direction = 1 if kind == KIND_HEAT else (-1 if kind == KIND_COOL else 0)
    n_rev, n_dup = _reversals_and_duplicates(sub_T, direction)
    return Segment(
        index=int(index),
        kind=kind,
        role="",
        cycle_index=0,
        row_start=int(i0),
        row_stop=int(i1),
        n_points=int(i1 - i0),
        t_start=float(sub_t[0]),
        t_end=float(sub_t[-1]),
        T_start=float(sub_T[0]),
        T_end=float(sub_T[-1]),
        T_min=float(sub_T.min()),
        T_max=float(sub_T.max()),
        rate_mean=float((sub_T[-1] - sub_T[0]) / duration) if duration > 0 else 0.0,
        rate_median=float(np.median(sub_r)) if sub_r.size else NAN,
        rate_mode=_mode_rate(sub_r, cfg.mode_bin_C_min),
        rate_fit_r2=_fit_r2(sub_t, sub_T),
        control_lost_below_C=NAN,
        monotone_fraction=_monotone_fraction(sub_T, direction),
        n_reversals=n_rev,
        n_duplicate_T=n_dup,
        is_partial=bool(i1 == n_total),
    )


# ---------------------------------------------------------------------------
# segment_temperature_program
# ---------------------------------------------------------------------------
def segment_temperature_program(time_min, temp_C, *,
                                config: Optional[SegmentationConfig] = None,
                                program: Optional[MethodProgram] = None,
                                return_rate: bool = False):
    """Tile a run with physically homogeneous segments.

    A Schmitt trigger on |dT/dt| finds the coarse runs; then a FLAT-CORE test
    decides every flat one: it is a hold iff it contains at least
    `min_hold_min` minutes of temperature spanning at most `hold_flat_span_C`.
    No |dT/dt| threshold can make that call -- the measured hold range
    (0.000-2.28 C/min) is a strict subset of every ramp's range.

    The optional `program` is used for ONE thing here: deriving a rate threshold
    from the slowest programmed ramp. Segmentation without it gives the same
    segments on the real file, which is the guarantee that makes the program a
    refinement rather than a requirement.

    Returns a list of Segments indexed into the CLEANED arrays, or
    ``(segments, rate)`` with `return_rate=True`, where `rate` is the pointwise
    estimate aligned with the cleaned rows.

    Guarantees, all asserted by tests: ``segments[0].row_start == 0``,
    ``segments[-1].row_stop == n_cleaned``, adjacent segments abut exactly,
    ``sum(n_points) == n_cleaned``. Any run with at least one finite row yields
    at least one segment; only an empty array yields ``[]``.
    """
    cfg = config or SegmentationConfig()
    t, T, _keep = clean_time_temperature(time_min, temp_C)
    n = int(t.size)
    if n == 0:
        return ([], np.zeros(0, dtype=float)) if return_rate else []

    if n < 5:
        # Too short for any rate estimate to mean anything. One segment, called
        # what it is: a stretch of temperature we cannot characterise.
        rate = np.zeros(n, dtype=float)
        seg = _make_segment(1, KIND_ISOTHERMAL, 0, n, t, T, rate, n, cfg)
        seg = replace(seg, is_partial=True)
        segments = group_cycles([seg])
        return (segments, rate) if return_rate else segments

    rate = estimate_rate(t, T, window_min=cfg.rate_window_min,
                         min_points=cfg.rate_min_points)
    r_on = _resolve_threshold(rate, cfg, program)
    r_off = cfg.hysteresis_frac * r_on
    runs = _dissolve(_runs(_hysteretic_state(rate, r_on, r_off)), t, T, cfg)
    if not runs:
        runs = [[0, n, 0]]

    if cfg.refine_boundaries and len(runs) > 1:
        for k in range(len(runs) - 1):
            a, b = runs[k], runs[k + 1]
            if (a[2] != 0) == (b[2] != 0):
                continue                     # ramp/ramp turnaround is already exact
            if a[2] != 0:
                if not cfg.refine_ramp_to_hold:
                    continue
                new_cut = _refine_ramp_hold_corner(t, T, a, b, cfg, True)
            else:
                if not cfg.refine_hold_to_ramp:
                    continue
                new_cut = _refine_ramp_hold_corner(t, T, b, a, cfg, False)
            cut = _clamped_boundary(new_cut, a, b)
            if cut is not None:
                a[1] = cut
                b[0] = cut

    segments: List[Segment] = []
    for k, (i0, i1, state) in enumerate(runs):
        kind = KIND_HEAT if state > 0 else (KIND_COOL if state < 0 else KIND_ISOTHERMAL)
        segments.append(_make_segment(k + 1, kind, i0, i1, t, T, rate, n, cfg))

    # A flat stretch at the very START of a record that later ramps is the
    # instrument equilibrating, not a programmed soak. Calling it out matters
    # because an equilibrate carries no kinetics and must not be offered as a
    # hold to compare across cycles.
    if len(segments) > 1 and segments[0].kind == KIND_ISOTHERMAL and \
            any(s.is_ramp for s in segments[1:]):
        segments[0] = replace(segments[0], kind=KIND_EQUILIBRATE)

    segments = group_cycles(segments)
    return (segments, rate) if return_rate else segments


# ---------------------------------------------------------------------------
# group_cycles
# ---------------------------------------------------------------------------
def group_cycles(segments: Sequence[Segment]) -> List[Segment]:
    """Fill `cycle_index` and `role` on every segment.

    Convention: a new cycle begins at every `heat` segment that FOLLOWS a
    `cool`, and the record starts in cycle 1. Heating-first, because a hot
    plateau is produced by the ramp before it: reading the record the other way
    would put each 1300 C soak in the cycle that has not happened yet, and every
    cycle-to-cycle comparison of the soak would be off by one.

    Roles are `equilibrate`, `heat`, `cool`, and for a flat stretch `hold_hot`
    or `hold_cold` depending on which ramp preceded it (`hold` when nothing
    did). A hold's role, not its kind, is what says whether it is the 1300 C
    soak or the 50 C wait.
    """
    out: List[Segment] = []
    cycle = 1
    seen_cool = False
    last_ramp = 0
    for seg in segments:
        if seg.kind == KIND_HEAT and seen_cool:
            cycle += 1
            seen_cool = False
        if seg.kind == KIND_HEAT:
            role, last_ramp = ROLE_HEAT, +1
        elif seg.kind == KIND_COOL:
            role, last_ramp, seen_cool = ROLE_COOL, -1, True
        elif seg.kind == KIND_EQUILIBRATE:
            role = ROLE_EQUILIBRATE
        else:
            role = (ROLE_HOLD_HOT if last_ramp > 0
                    else (ROLE_HOLD_COLD if last_ramp < 0 else ROLE_HOLD))
        out.append(replace(seg, cycle_index=cycle, role=role))
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _thermal_physical_steps(program: MethodProgram) -> Tuple[MethodStep, ...]:
    """Thermal steps of the program AS EXECUTED (repeats unrolled).

    `.steps` is what the operator WROTE; with a `Repeat segment N M times` in
    the method the written list and the executed list differ, and it is the
    executed one the data can be aligned against.
    """
    try:
        physical, _notes = program.physical_steps()
    except Exception:
        physical = tuple(getattr(program, "steps", ()) or ())
    return tuple(s for s in physical if getattr(s, "is_thermal", False))


def _written_indices(steps: Sequence[MethodStep]) -> Tuple[int, ...]:
    """Written step numbers of `steps`, each listed once, in first-seen order.

    The physical list repeats a written index once per pass of a
    `Repeat segment` block. `unmatched_steps` is a list of step NUMBERS the
    user can look up in their method, so the same number four times would be
    noise; the pass count belongs in the note, not in this tuple.
    """
    seen: set = set()
    out: List[int] = []
    for st in steps:
        ix = int(getattr(st, "index", 0))
        if ix not in seen:
            seen.add(ix)
            out.append(ix)
    return tuple(out)


def _block_info(segment: Segment, block: Sequence[MethodStep]
                ) -> Tuple[float, float, float, float, float]:
    """(cost, signed programmed rate, target, programmed duration, target gap).

    An infinite cost means "this block cannot be what this segment is" and the
    DP will never choose it: a ramp segment must consume exactly one ramp step,
    travelling the right way; a hold segment must consume no ramp step at all.
    Everything else is a soft penalty, so a furnace that missed its target still
    aligns -- it just costs more.
    """
    ramps = [s for s in block if s.kind == STEP_RAMP]
    isos = [s for s in block if s.kind == STEP_ISOTHERMAL]
    settle = [s for s in block if s.kind in (STEP_EQUILIBRATE, STEP_JUMP)]
    prog_dur = float(sum(s.duration_min for s in isos
                         if np.isfinite(s.duration_min)))
    gap = 0.0
    if segment.is_ramp:
        if len(ramps) != 1:
            return np.inf, NAN, NAN, NAN, NAN
        r = ramps[0]
        want = +1 if float(r.target_C) > segment.T_start else -1
        got = +1 if segment.kind == KIND_HEAT else -1
        if want != got:
            return np.inf, NAN, NAN, NAN, NAN
        prate = abs(float(r.rate_C_min))
        if not np.isfinite(prate) or prate <= 0.0:
            return np.inf, NAN, NAN, NAN, NAN
        cost = abs(abs(segment.rate_mode) - prate) / max(prate, 1e-6)
        prog_dur += abs(float(r.target_C) - segment.T_start) / max(prate, 1e-6)
        gap = (max(float(r.target_C) - segment.T_end, 0.0) if got > 0
               else max(segment.T_end - float(r.target_C), 0.0))
        # A missed target is a real mismatch but a mild one: 0.002/C makes the
        # 44 C the real furnace missed by worth 0.09, well below the 0.2 cost
        # of leaving a step unmatched, so the step is still claimed.
        cost += 0.002 * gap
        prate_out = float(want) * prate
        ptarget = float(r.target_C)
    else:
        if ramps or not (isos or settle):
            return np.inf, NAN, NAN, NAN, NAN
        ptarget, prate_out, cost = NAN, 0.0, 0.0
    if prog_dur > 0:
        d = abs(segment.duration_min - prog_dur) / prog_dur
        # A partial segment was cut short by the operator, not by the furnace:
        # half weight, or the last segment of every early-stopped run misaligns.
        cost += 0.5 * d if segment.is_partial else d
    cost += 0.05 * (len(block) - 1)
    return float(cost), float(prate_out), float(ptarget), float(prog_dur), float(gap)


def _absorbed(segment: Segment, block: Sequence[MethodStep]) -> Tuple[int, ...]:
    """Steps of the block that ran but left no separate signature in the data.

    For a ramp segment that is every non-ramp step in the block -- the case that
    matters: `Ramp to 50 C` + `Isothermal 20 min at 50 C` is ONE passive cool in
    the record because the furnace never reached 50 C, so the hold is absorbed.
    "Absorbed" and "not executed" are different claims and only the first is
    true.
    """
    if not block:
        return ()
    if segment.is_ramp:
        return tuple(int(s.index) for s in block if s.kind != STEP_RAMP)
    return tuple(int(s.index) for s in block[1:])


def _extend_ramp_gaps_through_soak(matches: Sequence["StepMatch"],
                                   seg_list: Sequence[Segment],
                                   ) -> List["StepMatch"]:
    """Re-measure each ramp's target gap at the end of the soak that follows it.

    A Q600 ramp step hands control to the next step slightly short of its
    setpoint, and the isothermal after it finishes the approach. Measured at
    the ramp's own last point, every heating ramp of BM-0-cycles looks like a
    missed setpoint -- 7.83 / 8.26 / 6.75 / 5.10 C below 1300 C. Measured at
    the end of the hold that follows, the same four ramps are 1.39 / 1.36 /
    1.36 / 1.37 C short: they reached it.

    A ramp that genuinely never arrived is untouched, because there is no
    following hold segment to extend into -- its programmed soak was ABSORBED
    into the same cooling segment (the furnace was still drifting down through
    it), so the four real cools stay 44.36 / 27.32 / 9.88 / 11.81 C short.

    This is a reporting fix, deliberately applied AFTER the DP: the alignment's
    own cost term keeps using the ramp-end gap, so no assignment changes. It
    matters because `target_gap_C <= 5 C` is what the rate series turns into
    `setpoint_reached`, and measured in the wrong place that rule flags 100 %
    of real heating ramps -- a warning that fires on everything says nothing.
    """
    by_pos = {s.index: i for i, s in enumerate(seg_list)}
    out: List["StepMatch"] = []
    for m in matches:
        seg_pos = by_pos.get(m.segment_index)
        if (seg_pos is None or not np.isfinite(m.target_gap_C)
                or not np.isfinite(m.programmed_target)):
            out.append(m)
            continue
        seg = seg_list[seg_pos]
        if not seg.is_ramp or m.target_gap_C <= 0.0:
            out.append(m)
            continue
        # Only a hold the alignment matched to the very next programmed step(s)
        # counts: an unaligned hold, or one belonging to a later block, says
        # nothing about THIS ramp's setpoint.
        end_pos = m.step_positions[-1] + 1 if m.step_positions else None
        extreme = seg.T_max if seg.kind == KIND_HEAT else seg.T_min
        k = seg_pos + 1
        while k < len(seg_list) and seg_list[k].kind == KIND_ISOTHERMAL:
            nxt = next((x for x in matches
                        if x.segment_index == seg_list[k].index), None)
            if (nxt is None or end_pos is None or not nxt.step_positions
                    or nxt.step_positions[0] != end_pos):
                break
            extreme = (max(extreme, seg_list[k].T_max) if seg.kind == KIND_HEAT
                       else min(extreme, seg_list[k].T_min))
            end_pos = nxt.step_positions[-1] + 1
            k += 1
        if k == seg_pos + 1:
            out.append(m)
            continue
        gap = (max(float(m.programmed_target) - extreme, 0.0)
               if seg.kind == KIND_HEAT
               else max(extreme - float(m.programmed_target), 0.0))
        out.append(replace(m, target_gap_C=float(gap)))
    return out


def _block_text(block: Sequence[MethodStep]) -> str:
    """Verbatim programmed lines of a block, numbered as the instrument prints."""
    parts = []
    for s in block:
        text = s.text or (s.render() if hasattr(s, "render") else "")
        parts.append(f"{s.index}: {text}")
    return "; ".join(parts)


def align_segments_to_program(segments: Sequence[Segment], program: MethodProgram,
                              *, max_block: int = 3, skip_cost: float = 0.2,
                              segment_skip_cost: float = 1.0) -> ProgramAlignment:
    """Monotone MANY-TO-ONE dynamic-programming alignment of segments to steps.

    Each detected segment consumes 1..`max_block` CONSECUTIVE thermal steps, in
    order. Many-to-one is mandatory, not a convenience: a programmed hold the
    furnace never settled into is ABSORBED into the adjacent ramp segment (steps
    5+6 of the real file become one cooling segment), and a one-to-one aligner
    is forced to report those four holds as "not observed", which is a different
    and false claim -- they ran, the furnace was simply still coasting through
    them.

    Leading and trailing steps may stay unmatched (an early-stopped run), at
    `skip_cost` each. A segment may consume NOTHING at `segment_skip_cost`,
    which is what keeps the alignment finite when the data contains more
    segments than the program has steps -- the alternative is a total failure
    that throws away the matches it did find. The default makes that move dearer
    than any real match, so it is only ever taken when nothing else is possible.

    Never raises. A program with no thermal steps yields ``method="no-program"``;
    an infeasible one yields ``method="dp-failed"`` with the reason in `notes`.
    """
    seg_list = list(segments)
    notes: List[str] = []
    measured_total = float(seg_list[-1].t_end) if seg_list else NAN

    steps = _thermal_physical_steps(program) if program is not None else ()
    n_seg, n_step = len(seg_list), len(steps)
    if n_seg == 0 or n_step == 0:
        if program is None:
            notes.append("No programmed method was supplied; segments are data-derived only.")
        elif n_step == 0:
            notes.append("The method contains no thermal steps, so there is nothing to align to.")
        else:
            notes.append("No segments were detected, so there is nothing to align.")
        return ProgramAlignment(
            method="no-program",
            unmatched_steps=_written_indices(steps),
            unmatched_segments=tuple(s.index for s in seg_list),
            measured_total_min=measured_total,
            programmed_total_min=_programmed_total(program, seg_list),
            notes=tuple(notes),
        )

    try:
        max_block = max(1, int(max_block))
    except (TypeError, ValueError):
        max_block = 3

    inf = float("inf")
    D = np.full((n_seg + 1, n_step + 1), inf)
    D[0, 0] = 0.0
    for j in range(1, n_step + 1):
        D[0, j] = skip_cost * j
    back: Dict[Tuple[int, int], Optional[int]] = {}
    for i in range(1, n_seg + 1):
        for j in range(0, n_step + 1):
            best, best_jp = inf, None
            lo = max(0, j - max_block)
            for jp in range(lo, j + 1):
                if not np.isfinite(D[i - 1, jp]):
                    continue
                if jp == j:
                    c = float(segment_skip_cost)
                else:
                    c = _block_info(seg_list[i - 1], steps[jp:j])[0]
                if not np.isfinite(c):
                    continue
                v = D[i - 1, jp] + c
                if v < best:
                    best, best_jp = v, jp
            D[i, j] = best
            back[(i, j)] = best_jp

    j_end = int(np.argmin(D[n_seg, :]))
    if not np.isfinite(D[n_seg, j_end]):
        notes.append("No monotone assignment of these segments to these steps exists; "
                     "the segments stand on their own.")
        return ProgramAlignment(
            method="dp-failed",
            unmatched_steps=_written_indices(steps),
            unmatched_segments=tuple(s.index for s in seg_list),
            measured_total_min=measured_total,
            programmed_total_min=_programmed_total(program, seg_list),
            notes=tuple(notes),
        )

    path: List[Tuple[int, int, int]] = []
    i, j = n_seg, j_end
    while i > 0:
        jp = back[(i, j)]
        if jp is None:
            break
        path.append((i - 1, jp, j))
        i, j = i - 1, jp
    path.reverse()

    matches: List[StepMatch] = []
    unmatched_segments: List[int] = []
    for si, jp, j in path:
        seg = seg_list[si]
        block = steps[jp:j]
        if not block:
            unmatched_segments.append(seg.index)
            continue
        cost, prate, ptarget, pdur, gap = _block_info(seg, block)
        matches.append(StepMatch(
            segment_index=seg.index,
            step_indices=tuple(int(s.index) for s in block),
            step_positions=tuple(range(jp, j)),
            repeat_passes=tuple(int(getattr(s, "repeat_pass", 0)) for s in block),
            programmed_kind="+".join(s.kind for s in block),
            programmed_rate=prate,
            programmed_target=ptarget,
            programmed_duration=pdur,
            measured_rate=seg.rate_mode,
            measured_duration=seg.duration_min,
            target_gap_C=gap,
            absorbed_step_indices=_absorbed(seg, block),
            cost=cost,
            step_text=_block_text(block),
        ))

    # A ramp's setpoint is only fairly judged after the soak that follows it
    # completes the approach; see _extend_ramp_gaps_through_soak.
    matches = _extend_ramp_gaps_through_soak(matches, seg_list)

    # Coverage is counted on POSITIONS in the physical (repeats unrolled) list,
    # never on written step indices: with `Repeat segment 1 for 3 times` the
    # four passes of step 1 all carry index 1, so a set of written indices
    # would let pass 1 stand in for passes 2-4 and a run stopped after one
    # cycle would report `unmatched_steps == ()` and `ok is True`. The tuple
    # still reports WRITTEN indices, once each, because that is the number the
    # operator can find in their method; the note names the passes.
    used_pos = {p for m in matches for p in m.step_positions}
    missing = [(pos, steps[pos]) for pos in range(n_step) if pos not in used_pos]
    seen_written: set = set()
    unmatched_list: List[int] = []
    for _pos, st in missing:
        ix = int(st.index)
        if ix not in seen_written:
            seen_written.add(ix)
            unmatched_list.append(ix)
    unmatched_steps = tuple(unmatched_list)
    if unmatched_steps:
        repeated = [st for _pos, st in missing if int(getattr(st, "repeat_pass", 0)) > 0]
        extra = ""
        if repeated:
            extra = (f" ({len(missing)} unexecuted step slots in all, including "
                     f"{len(repeated)} on a repeated pass of the method)")
        notes.append("Programmed step(s) " + ", ".join(str(i) for i in unmatched_steps)
                     + extra
                     + " left no signature in the record and were not matched.")
    for m in matches:
        if m.absorbed_step_indices:
            notes.append(
                "Step(s) " + ", ".join(str(i) for i in m.absorbed_step_indices)
                + f" ran but were absorbed into segment #{m.segment_index}: the furnace "
                  "was still travelling through them, so they left no separate signature.")
    if unmatched_segments:
        notes.append("Segment(s) " + ", ".join(f"#{i}" for i in unmatched_segments)
                     + " have no counterpart in the programmed method.")

    return ProgramAlignment(
        matches=tuple(matches),
        unmatched_steps=unmatched_steps,
        unmatched_segments=tuple(unmatched_segments),
        total_cost=float(D[n_seg, j_end]),
        method="dp",
        programmed_total_min=_programmed_total(program, seg_list),
        measured_total_min=measured_total,
        notes=tuple(notes),
    )


def _programmed_total(program: Optional[MethodProgram],
                      segments: Sequence[Segment]) -> float:
    """Programmed length of the whole method on the run's own clock.

    Anchored at the FIRST segment's ``t_start`` and ``T_start`` because the
    length of the opening ramp is ``|target - T0| / rate``: the specimen's
    starting temperature is part of the answer, and using a nominal 25 C instead
    of the measured 19.014 C moves the total by minutes.
    """
    if program is None:
        return NAN
    try:
        t0 = float(segments[0].t_start) if segments else 0.0
        T0 = float(segments[0].T_start) if segments else 25.0
        entries = program.schedule(t0=t0, T0=T0)
    except Exception:
        return NAN
    return float(entries[-1].t_end) if entries else NAN


def _control_lost_below(time_min: np.ndarray, temp_C: np.ndarray, rate: np.ndarray,
                        segment: Segment, nominal: float,
                        cfg: SegmentationConfig) -> float:
    """Temperature below which the furnace stopped following the programmed rate.

    Defined as the temperature at the LAST sample whose |dT/dt| still reached
    `control_tol_frac` of the nominal rate; NaN when the rate was held all the
    way to the end of the segment. Only meaningful with an alignment, because
    "the programmed rate" is exactly the thing the data alone does not know.

    A tail shorter than `control_min_tail_frac` of the segment (or than one
    rate window) is NOT reported. Every ramp decelerates into the hold that
    follows it; that is the setpoint arriving, not control being lost, and
    reporting it would put a "control lost below 1284 C" on a ramp that held
    20.0 C/min for 64 of its 64.7 minutes. Measured separation on the real run:
    0.53-1.24 % of the segment on the four heats against 25.7-49.4 % on the
    four cools.

    This is a one-line field that stops a real misreading: on the real run,
    cooling control is lost below roughly 448 / 346 / 228 / 228 C, and a user
    reading "-20 C/min" off the segment table would otherwise apply it at 200 C
    where the furnace was coasting at -3.
    """
    if not segment.is_ramp:
        return NAN
    if not np.isfinite(nominal) or abs(nominal) <= 0.0:
        return NAN
    sl = slice(int(segment.row_start), int(segment.row_stop))
    r = np.abs(np.asarray(rate, dtype=float)[sl])
    T = np.asarray(temp_C, dtype=float)[sl]
    t = np.asarray(time_min, dtype=float)[sl]
    if r.size == 0 or r.size != T.size or t.size != r.size:
        return NAN
    held = np.isfinite(r) & (r >= cfg.control_tol_frac * abs(nominal))
    if not held.any():
        # Control was never established at all; the first sample is the honest
        # boundary, and the segment's own note will say the rate was never met.
        return float(T[0])
    last = int(np.flatnonzero(held)[-1])
    if last >= r.size - 1:
        return NAN
    tail_min = float(t[-1] - t[last])
    duration = float(t[-1] - t[0])
    if tail_min <= float(cfg.rate_window_min):
        return NAN
    if duration > 0.0 and (tail_min / duration) < float(cfg.control_min_tail_frac):
        return NAN
    return float(T[last])


def attach_alignment(segments: Sequence[Segment], alignment: ProgramAlignment,
                     *, config: Optional[SegmentationConfig] = None,
                     time_min=None, temp_C=None, rate_C_min=None) -> List[Segment]:
    """Copy the alignment's findings onto each segment and build its `note`.

    Fills `step_indices`, `step_text`, `rate_nominal_C_min`, `target_gap_C` and
    `note`; fills `control_lost_below_C` as well when the cleaned `time_min`,
    `temp_C` and `rate_C_min` arrays are supplied, because that number needs
    BOTH the programmed rate (which only the alignment knows) and the pointwise
    measured rate (which only the arrays carry). `segment_run` always supplies
    them; a caller who does not simply gets NaN there rather than a wrong
    number.
    """
    cfg = config or SegmentationConfig()
    by_index = {m.segment_index: m for m in alignment.matches}
    out: List[Segment] = []
    for seg in segments:
        m = by_index.get(seg.index)
        if m is None:
            out.append(seg)
            continue
        note_bits: List[str] = []
        if np.isfinite(m.target_gap_C) and m.target_gap_C > 0.5:
            # Word it as what was measured, not as a verdict. A heating ramp
            # that stops 7.8 C short of 1300 did not fail -- the setpoint
            # arrived during the soak that follows it, and the furnace reached
            # 1298.6 there. A cooling ramp that stops 44.4 C short of 50 never
            # got there at all. The same field describes both, so the text has
            # to state the gap and the target and leave the reading to the
            # person who knows which one they are looking at.
            target = (f" of its {m.programmed_target:.0f} °C target"
                      if np.isfinite(m.programmed_target) else "")
            note_bits.append(f"ramp ended at {seg.T_end:.0f} °C, "
                             f"{m.target_gap_C:.1f} °C short{target}")
        if m.absorbed_step_indices:
            note_bits.append("absorbed step(s) "
                             + ", ".join(str(i) for i in m.absorbed_step_indices))
        if seg.is_partial:
            note_bits.append("acquisition stopped inside this segment")
        lost = NAN
        if time_min is not None and temp_C is not None and rate_C_min is not None:
            lost = _control_lost_below(time_min, temp_C, rate_C_min, seg,
                                       m.programmed_rate, cfg)
            if np.isfinite(lost):
                note_bits.append(f"rate control lost below {lost:.0f} °C")
        out.append(replace(
            seg,
            step_indices=m.step_indices,
            step_text=m.step_text,
            rate_nominal_C_min=float(m.programmed_rate),
            target_gap_C=float(m.target_gap_C),
            control_lost_below_C=lost,
            note="; ".join(note_bits),
        ))
    return out


# ---------------------------------------------------------------------------
# SegmentedRun
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentedRun:
    """Everything one record's segmentation produced, in one object.

    `time_min`, `temp_C` and `rate_C_min` are the CLEANED arrays; `keep_index`
    maps them back to the ORIGINAL rows and `n_rows` is the original row count.
    Segment bounds index the cleaned arrays; `row_indices()` is the only
    translation to original positions, and the only thing a DataFrame may be
    sliced with.
    """

    segments: Tuple[Segment, ...]
    alignment: ProgramAlignment
    program: Optional[MethodProgram]
    config: SegmentationConfig
    time_min: np.ndarray
    temp_C: np.ndarray
    rate_C_min: np.ndarray
    keep_index: np.ndarray
    n_rows: int

    def __len__(self) -> int:
        """How many segments the record was tiled with."""
        return len(self.segments)

    def __iter__(self) -> Iterator[Segment]:
        """Segments in TIME order, which is the only order they ever come in."""
        return iter(self.segments)

    def __bool__(self) -> bool:
        """False when nothing usable was found, so `if run:` reads correctly."""
        return bool(self.segments)

    @property
    def n_dropped_rows(self) -> int:
        """Original rows the cleaning removed (non-finite or backwards in time)."""
        return int(self.n_rows) - int(self.keep_index.size)

    def by_kind(self, kind: str) -> Tuple[Segment, ...]:
        """Segments of one KIND_*, in time order."""
        return tuple(s for s in self.segments if s.kind == kind)

    def by_role(self, role: str) -> Tuple[Segment, ...]:
        """Segments of one ROLE_*, in time order."""
        return tuple(s for s in self.segments if s.role == role)

    def cycles(self) -> Dict[int, Tuple[Segment, ...]]:
        """{cycle_index: segments of that cycle}, in time order within a cycle."""
        out: Dict[int, List[Segment]] = {}
        for s in self.segments:
            out.setdefault(int(s.cycle_index), []).append(s)
        return {k: tuple(v) for k, v in sorted(out.items())}

    @property
    def n_cycles(self) -> int:
        """How many thermal cycles the record contains."""
        return len(self.cycles())

    def segment_by_index(self, index: int) -> Optional[Segment]:
        """The segment with this 1-based index, or None."""
        for s in self.segments:
            if s.index == index:
                return s
        return None

    def row_indices(self, segment: Segment) -> np.ndarray:
        """ORIGINAL row positions of this segment.

        The ONLY way to slice a DataFrame. `Segment.row_start` / `row_stop`
        index the cleaned arrays, and the two differ by exactly the rows the
        cleaning dropped -- which on every file shipped here is zero, so a
        confusion between them is invisible in testing and catastrophic in the
        field.
        """
        return np.asarray(self.keep_index, dtype=np.intp)[segment.slice]

    def xy(self, segment: Segment, y, *, y_alt=None,
           x_kind: str = X_TEMPERATURE, dedupe: bool = True,
           invert: bool = False,
           max_duplicate_frac: Optional[float] = 0.01) -> PreparedXY:
        """Analysis-ready arrays for one segment; see `prepare_segment_xy`.

        `y` (and `y_alt`) may be ORIGINAL-length, which is the normal case for a
        DataFrame column, or already cleaned-length. Both are accepted and the
        choice is made by size, because the caller that has a DataFrame and the
        caller that has this object's own arrays are different callers.
        """
        rows = self.row_indices(segment)
        x_src = self.temp_C if x_kind == X_TEMPERATURE else self.time_min
        x = np.asarray(x_src, dtype=float)[segment.slice]
        y_seg = _slice_like(y, rows, segment, int(self.n_rows), int(self.keep_index.size))
        alt_seg = (None if y_alt is None else
                   _slice_like(y_alt, rows, segment, int(self.n_rows),
                               int(self.keep_index.size)))
        return _prepare_arrays(x, y_seg, alt_seg, rows, segment=segment,
                               x_kind=x_kind, dedupe=dedupe, invert=invert,
                               max_duplicate_frac=max_duplicate_frac)

    def default_window(self, segment: Segment, *, central_frac: float = 0.60
                       ) -> Tuple[float, float]:
        """Central 60 % of this segment's own temperature range, ascending.

        A window in the segment's own range, never a global default: the historic
        350-700 C box is meaningless on a cooling branch that never went above
        300, and a window that falls outside the data is what produces "Baseline
        range has too few points" three dialogs later.
        """
        lo, hi = float(segment.T_min), float(segment.T_max)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (NAN, NAN)
        mid = 0.5 * (lo + hi)
        try:
            frac = float(central_frac)
        except (TypeError, ValueError):
            frac = 0.60
        frac = min(max(frac, 0.0), 1.0)
        half = 0.5 * frac * (hi - lo)
        return (mid - half, mid + half)

    def table(self) -> List[dict]:
        """One dict per segment, keys exactly SEGMENT_TABLE_COLUMNS."""
        return segment_table(self.segments)

    def format_table(self) -> str:
        """Fixed-width rendering of `table()` for a log or a status pane."""
        return format_segment_table(self.segments)

    def describe(self) -> str:
        """One line naming the segments, cycles and clock of the whole run."""
        if not self.segments:
            return "no segments"
        kinds = "".join(s.kind[0].upper() for s in self.segments)
        span = self.segments[-1].t_end - self.segments[0].t_start
        return (f"{len(self.segments)} segments ({kinds}), {self.n_cycles} cycle(s), "
                f"{span:.1f} min, alignment={self.alignment.method}")


def _slice_like(values, rows: np.ndarray, segment: Segment, n_rows: int,
                n_clean: int) -> np.ndarray:
    """Take a segment's worth of `values`, in original or cleaned indexing.

    Sized by the array itself rather than by a flag: a DataFrame column has
    `n_rows` entries and this object's own arrays have `n_clean`. When the two
    are equal (every real file here) the answer is the same either way.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == n_rows and n_rows != n_clean:
        return arr[rows]
    if arr.size == n_clean:
        return arr[segment.slice]
    if arr.size == n_rows:
        return arr[rows]
    raise SegmentScopeError(
        f"y has {arr.size} points but this run has {n_rows} original rows "
        f"({n_clean} after cleaning); it belongs to a different record.")


# ---------------------------------------------------------------------------
# segment_run
# ---------------------------------------------------------------------------
def segment_run(time_min, temp_C, *, program: Optional[MethodProgram] = None,
                config: Optional[SegmentationConfig] = None) -> SegmentedRun:
    """clean -> segment -> group_cycles -> align -> attach.

    The one entry point the Qt layer calls; its result is cached per record.
    Never raises on data: an empty or unusable record yields a SegmentedRun with
    no segments and an alignment that says why, because a workspace that throws
    while you are choosing a file is worse than one that tells you the file has
    nothing in it.
    """
    cfg = config or SegmentationConfig()
    n_rows = int(_as_float_array(time_min).size)
    t, T, keep = clean_time_temperature(time_min, temp_C)

    segments, rate = segment_temperature_program(t, T, config=cfg,
                                                 program=program,
                                                 return_rate=True)
    segments = list(segments)

    if program is None:
        alignment = ProgramAlignment(
            method="no-program",
            unmatched_segments=tuple(s.index for s in segments),
            measured_total_min=float(segments[-1].t_end) if segments else NAN,
            notes=("No programmed method was supplied; segments are data-derived only.",),
        )
    else:
        alignment = align_segments_to_program(segments, program)
        segments = attach_alignment(segments, alignment, config=cfg,
                                    time_min=t, temp_C=T, rate_C_min=rate)

    return SegmentedRun(
        segments=tuple(segments),
        alignment=alignment,
        program=program,
        config=cfg,
        time_min=t,
        temp_C=T,
        rate_C_min=np.asarray(rate, dtype=float),
        keep_index=np.asarray(keep, dtype=np.intp),
        n_rows=n_rows,
    )


# ---------------------------------------------------------------------------
# prepare_segment_xy -- the single array-preparation rule
# ---------------------------------------------------------------------------
def _prepare_arrays(x: np.ndarray, y: np.ndarray, y_alt: Optional[np.ndarray],
                    source_index: np.ndarray, *, segment: Segment, x_kind: str,
                    dedupe: bool, invert: bool,
                    max_duplicate_frac: Optional[float]) -> PreparedXY:
    """Shared body of `prepare_segment_xy` and `SegmentedRun.xy`.

    The order of operations here is load-bearing and is documented on the public
    function; every step is commented with what breaks if it moves.
    """
    if x_kind not in X_KINDS:
        raise ValueError(f"x_kind must be one of {X_KINDS!r}, got {x_kind!r}")

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    alt = None if y_alt is None else np.asarray(y_alt, dtype=float).ravel()
    src = np.asarray(source_index, dtype=np.intp).ravel()
    if y.size != x.size or (alt is not None and alt.size != x.size) or src.size != x.size:
        raise SegmentScopeError(
            f"segment #{segment.index}: x/y/source arrays have different lengths "
            f"({x.size}/{y.size}/{src.size}) and cannot describe one curve.")

    # 2. finite mask across EVERY array at once. Masking y separately from x is
    #    how the pre-rework code could return arrays of different lengths.
    mask = np.isfinite(x) & np.isfinite(y)
    if alt is not None:
        mask &= np.isfinite(alt)
    x, y, src = x[mask], y[mask], src[mask]
    if alt is not None:
        alt = alt[mask]

    # 3. the scope rule.
    was_sorted = False
    if x_kind == X_TEMPERATURE:
        if segment.is_hold:
            raise SegmentScopeError(
                f"segment #{segment.index} is a {segment.kind} ({segment.role}); "
                "temperature is not an abscissa on a hold. After a per-segment sort "
                "the measured holds still carry duplicate temperatures and produce "
                "infinite derivatives. Use x_kind='t'.")
        if not segment.is_ramp:
            raise SegmentScopeError(
                f"segment #{segment.index} has kind {segment.kind!r}, which is not a "
                "single monotone scan; temperature cannot be its abscissa.")
        # `was_sorted` reports that the temperature rule RAN, not that the
        # permutation was non-trivial: the caller's question is "is x
        # guaranteed ascending", and on a monotone heating ramp the answer is
        # yes precisely because the sort was the identity. That identity is
        # what makes the reworked path bit-identical to the old global one on a
        # single-ramp file.
        was_sorted = True
        if x.size > 1 and not bool(np.all(np.diff(x) > 0)):
            # STABLE sort: equal temperatures keep their acquisition order, so a
            # duplicate pair collapses to the mean of the two values actually
            # measured first, not to an arbitrary one of them.
            order = np.argsort(x, kind="stable")
            x, y, src = x[order], y[order], src[order]
            if alt is not None:
                alt = alt[order]
    else:
        if x.size > 1 and not bool(np.all(np.diff(x) > 0)):
            raise SegmentScopeError(
                f"segment #{segment.index}: time is not strictly increasing over this "
                "segment, so it cannot be an abscissa either. The record needs "
                "clean_time_temperature() first.")

    # 4. collapse exact-duplicate x, AVERAGING y. Unconditional, because the
    #    pre-rework code only de-duplicated inside one branch of one function and
    #    kept the first y, which silently discards half of a doubled point.
    n_collapsed = 0
    if dedupe and x.size > 1:
        first = np.concatenate(([True], np.diff(x) != 0.0))
        if not bool(first.all()):
            group = np.cumsum(first) - 1
            n_groups = int(group[-1]) + 1
            counts = np.bincount(group, minlength=n_groups).astype(float)
            y = np.bincount(group, weights=y, minlength=n_groups) / counts
            if alt is not None:
                alt = np.bincount(group, weights=alt, minlength=n_groups) / counts
            n_collapsed = int(x.size - n_groups)
            # y is the group MEAN but source_index can only name one row, so it
            # names the first of the group. Documented on PreparedXY.
            src = src[first]
            x = x[first]

    if n_collapsed and x.size and max_duplicate_frac is not None:
        frac = n_collapsed / float(n_collapsed + x.size)
        if frac > float(max_duplicate_frac):
            raise SegmentScopeError(
                f"segment #{segment.index}: {n_collapsed} of {n_collapsed + x.size} "
                f"points ({100.0 * frac:.1f} %) share an x value. That is not a "
                "single monotone scan -- it is the signature of the wrong abscissa "
                "or the wrong scope.")

    if dedupe and x.size > 1 and not bool(np.all(np.diff(x) > 0)):
        raise SegmentScopeError(
            f"segment #{segment.index}: a non-positive x step survived duplicate "
            "collapse, so the prepared curve is not single-valued.")

    # 5. invert LAST, so the sort and the collapse saw the measured signal.
    if invert:
        y = -y
        if alt is not None:
            alt = -alt

    return PreparedXY(x=x, y=y, y_alt=alt, source_index=src, x_kind=x_kind,
                      was_sorted=was_sorted, n_duplicates_collapsed=n_collapsed)


def prepare_segment_xy(time_min, temp_C, y, *, segment: Segment,
                       keep_index=None, y_alt=None,
                       x_kind: str = X_TEMPERATURE, dedupe: bool = True,
                       invert: bool = False,
                       max_duplicate_frac: Optional[float] = 0.01) -> PreparedXY:
    """Analysis-ready arrays for ONE segment. The only place slicing, masking,
    sorting and duplicate collapse happen.

    `time_min`, `temp_C`, `y` and `y_alt` must all be the same length. With
    `keep_index` they are the ORIGINAL rows and the segment's cleaned bounds are
    translated through it; without it they are assumed already cleaned.

    Order of operations, which is load-bearing:

      1. rows = this segment's rows (and nothing else -- a global operation on a
         cyclic record is the bug this module exists to remove);
      2. AND the finite masks of x, y and y_alt together, so the outputs cannot
         end up different lengths;
      3. abscissa rule:
         * ``x_kind='T'`` on a ramp -> STABLE ascending argsort (a no-op on a
           monotone ramp, which is why the reworked path is bit-identical to the
           old one on a single-ramp file);
         * ``x_kind='T'`` on a hold -> ``SegmentScopeError``. Temperature is not
           an abscissa on a hold: after a per-segment sort the four measured
           holds still carry 2-15 duplicate temperatures and produce 4-24
           non-finite derivative values, `inf` among them;
         * ``x_kind='t'`` -> no sort, and time must already be strictly
           increasing;
      4. collapse exact-duplicate x, AVERAGING y (and y_alt);
      5. invert.

    `max_duplicate_frac` is the structural guardrail for a call site that
    forgets to check the scope: when step 4 collapses more than that share of
    the points, `SegmentScopeError` is raised instead of quietly reproducing
    the global-argsort result. **Pass `None` to report instead of raising**,
    and read `PreparedXY.duplicate_fraction`.

    `None` is what every interactive path passes, because a duplicate flood is
    not always a wrong scope: a temperature column written to one or two
    decimals, or a ramp slow enough that the rounding step exceeds dT per
    sample, produces one legitimately. Measured: a NOISELESS 1 C/min ramp with
    a 2-decimal column is 16.7 % duplicate and a 0.5 C/min one is 58.3 %, and
    refusing those made the file unable to yield a Tg at all -- a hard block
    for a data-quality reason, which MERGED_DECISIONS M0.1 forbids, under a
    message ("the wrong abscissa or the wrong scope") that was false on exactly
    that data. The collapse has already AVERAGED those duplicates one line
    earlier, so the curve being thrown away was analysable: on the 1 C/min case
    it returns a Tg within 0.9 C of the planted answer.

    A non-positive x step surviving the collapse still raises unconditionally.
    That one is not a judgement about data quality, it is the invariant this
    function exists to provide.
    """
    if x_kind not in X_KINDS:
        raise ValueError(f"x_kind must be one of {X_KINDS!r}, got {x_kind!r}")
    t = np.asarray(time_min, dtype=float).ravel()
    T = np.asarray(temp_C, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    alt = None if y_alt is None else np.asarray(y_alt, dtype=float).ravel()

    if keep_index is None:
        rows = np.arange(int(segment.row_start), int(segment.row_stop), dtype=np.intp)
    else:
        rows = np.asarray(keep_index, dtype=np.intp).ravel()[segment.slice]

    n_needed = int(rows.max()) + 1 if rows.size else 0
    for name, arr in (("time_min", t), ("temp_C", T), ("y", yy)):
        if arr.size < n_needed:
            raise SegmentScopeError(
                f"segment #{segment.index} needs row {n_needed - 1} but {name} has "
                f"only {arr.size} points; the segment belongs to a different record.")
    if alt is not None and alt.size < n_needed:
        raise SegmentScopeError(
            f"segment #{segment.index} needs row {n_needed - 1} but y_alt has only "
            f"{alt.size} points; the segment belongs to a different record.")

    x = T[rows] if x_kind == X_TEMPERATURE else t[rows]
    return _prepare_arrays(x, yy[rows], None if alt is None else alt[rows], rows,
                           segment=segment, x_kind=x_kind, dedupe=dedupe,
                           invert=invert, max_duplicate_frac=max_duplicate_frac)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SEGMENT_TABLE_COLUMNS: Tuple[str, ...] = (
    "index", "kind", "role", "cycle", "rows", "n",
    "t_start", "t_end", "dur_min",
    "T_start", "T_end", "T_min", "T_max",
    "rate_mean", "rate_median", "rate_mode", "rate_nominal",
    "fit_r2", "monotone", "reversals", "dup_T",
    "control_lost_below", "target_gap", "partial", "steps", "note",
)


def _round(value: float, nd: int = 3):
    """Round for display, leaving NaN as NaN rather than inventing a 0."""
    v = float(value)
    return v if not np.isfinite(v) else round(v, nd)


def segment_label(segment: Segment, *, unit: str = "°C",
                  show_cycle: bool = True, show_step: bool = True) -> str:
    """One-line label for a combo box, a legend or a figure watermark.

    Example: ``'#3  cool  -20.0 °C/min  1299 -> 94 °C  cycle 1  -> step 5'``.
    The rate carries its SIGN so a cooling segment can never be mistaken for a
    heating one in a list, and the step reference is what lets a user connect
    the row to the method they wrote.
    """
    bits: List[str] = [f"#{segment.index}", segment.kind]
    if segment.is_ramp:
        rate = segment.rate_mode
        if not np.isfinite(rate):
            rate = segment.rate_mean
        bits.append(f"{rate:+.1f} {unit}/min")
        bits.append(f"{segment.T_start:.0f} -> {segment.T_end:.0f} {unit}")
    else:
        bits.append(f"{0.5 * (segment.T_start + segment.T_end):.0f} {unit}")
        bits.append(f"{segment.duration_min:.1f} min")
    if show_cycle and segment.cycle_index:
        bits.append(f"cycle {segment.cycle_index}")
    if show_step and segment.step_indices:
        word = "step" if len(segment.step_indices) == 1 else "steps"
        bits.append("-> " + word + " " + ", ".join(str(i) for i in segment.step_indices))
    if segment.is_partial:
        bits.append("(partial)")
    return "  ".join(bits)


def describe_segment(segment: Segment, *, unit: str = "°C") -> str:
    """Full sentence describing one segment, warnings included.

    Longer than `segment_label` on purpose: this is what goes in a tooltip, a
    status line or an export comment, where the three rate estimators, the
    quality of the fit and every reason to distrust the segment have to travel
    WITH the number instead of living in a manual.
    """
    head = (f"Segment #{segment.index} of cycle {segment.cycle_index}: "
            f"{segment.kind} ({segment.role}) from {segment.T_start:.1f} to "
            f"{segment.T_end:.1f} {unit} in {segment.duration_min:.1f} min, "
            f"{segment.n_points} points")
    bits: List[str] = [head]
    if segment.is_ramp:
        bits.append(f"measured rate {segment.rate_mode:+.3f} {unit}/min (mode; mean "
                    f"{segment.rate_mean:+.3f}, median {segment.rate_median:+.3f})")
        if np.isfinite(segment.rate_fit_r2):
            bits.append(f"linearity R2 = {segment.rate_fit_r2:.4f}")
    else:
        bits.append(f"held near {0.5 * (segment.T_min + segment.T_max):.1f} {unit} "
                    f"(span {segment.T_max - segment.T_min:.2f} {unit})")
    if segment.step_indices:
        word = "step" if len(segment.step_indices) == 1 else "steps"
        prog = f"programmed {word} " + ", ".join(str(i) for i in segment.step_indices)
        if np.isfinite(segment.rate_nominal_C_min) and segment.rate_nominal_C_min:
            prog += f" at {segment.rate_nominal_C_min:+.2f} {unit}/min"
        bits.append(prog)
    else:
        bits.append("no programmed step matched")
    if segment.note:
        bits.append(segment.note)
    return "; ".join(bits) + "."


def segment_table(segments: Sequence[Segment], *, unit: str = "°C") -> List[dict]:
    """One dict per segment, keys exactly `SEGMENT_TABLE_COLUMNS`.

    `rows` is rendered as ``"start:stop"`` and refers to the CLEANED arrays, the
    same convention as `Segment.row_start`. `unit` re-renders the degree symbol
    inside the free-text `note` column so a caller working in kelvin does not
    end up with a table that mixes the two.
    """
    out: List[dict] = []
    for s in segments:
        note = s.note.replace("°C", unit) if unit != "°C" else s.note
        out.append({
            "index": s.index,
            "kind": s.kind,
            "role": s.role,
            "cycle": s.cycle_index,
            "rows": f"{s.row_start}:{s.row_stop}",
            "n": s.n_points,
            "t_start": _round(s.t_start),
            "t_end": _round(s.t_end),
            "dur_min": _round(s.duration_min),
            "T_start": _round(s.T_start),
            "T_end": _round(s.T_end),
            "T_min": _round(s.T_min),
            "T_max": _round(s.T_max),
            "rate_mean": _round(s.rate_mean),
            "rate_median": _round(s.rate_median),
            "rate_mode": _round(s.rate_mode),
            "rate_nominal": _round(s.rate_nominal_C_min),
            "fit_r2": _round(s.rate_fit_r2, 4),
            "monotone": _round(s.monotone_fraction, 4),
            "reversals": s.n_reversals,
            "dup_T": s.n_duplicate_T,
            "control_lost_below": _round(s.control_lost_below_C, 1),
            "target_gap": _round(s.target_gap_C, 2),
            "partial": s.is_partial,
            "steps": ",".join(str(i) for i in s.step_indices),
            "note": note,
        })
    return out


def format_segment_table(segments: Sequence[Segment]) -> str:
    """Fixed-width rendering of `segment_table` for a log, a pane or a docstring."""
    rows = segment_table(segments)
    if not rows:
        return "(no segments)"
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows])
              for c in SEGMENT_TABLE_COLUMNS}
    lines = [" ".join(c.rjust(widths[c]) for c in SEGMENT_TABLE_COLUMNS)]
    for r in rows:
        lines.append(" ".join(str(r[c]).rjust(widths[c]) for c in SEGMENT_TABLE_COLUMNS))
    return "\n".join(lines)
