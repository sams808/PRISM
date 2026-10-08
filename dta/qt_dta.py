"""
qt_dta.py — DTA/DSC/TGA Tg tool, ported from ui_dta_processing.py's TgGuiApp.

All math delegates to dta_science.py (extracted and pytest-covered in M2) —
this file is purely the Qt presentation layer. Reuses the shared PlotWidget
(qt_widgets.py) instead of embedding matplotlib independently, and the shared
PerItemSettingsStore (qt_settings_store.py) for per-record settings instead
of re-solving persistence ad hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.np_compat import trapezoid
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QProgressDialog, QPushButton,
    QRadioButton, QScrollArea, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from dta.dta_science import (
    BaselineParams, TgDoubleTangentResult, TgParallelTangentResult,
    compute_derivative, compute_tg_derivative, compute_tg_double_tangent,
    compute_tg_parallel_improved, moving_average_10, resolve_baseline_params,
    _line_y,
)
from dta.dta_program import (
    PROGRAM_PARTIAL, SOURCE_BOTH, SOURCE_MTH, SOURCE_ORGMETHOD, SOURCE_USER,
    STEP_ISOTHERMAL, STEP_RAMP,
    MethodProgram, MethodStep, ProgramComparison,
    compare_programs, merge_programs, parse_mth_file, program_from_header,
)
from dta.dta_segments import (
    clean_time_temperature, clock_scale_to_min,
    KIND_COOL, KIND_HEAT, ROLE_COOL, ROLE_HEAT, SEGMENT_TABLE_COLUMNS, X_TEMPERATURE, X_TIME,
    PreparedXY, Segment, SegmentationConfig, SegmentedRun, SegmentScopeError,
    describe_segment, prepare_segment_xy, segment_label, segment_run, segment_table,
)
from dta import dta_rate_series as drs
from dta.dta_kinetics import (
    DEFAULT_Q_REF, KINETIC_METHODS, KineticFitResult, SeriesWarning,
    default_method_for, max_severity, sort_warnings, to_kelvin,
)
from dta.dta_rate_series import RateRow, RateSeries
from core.io_universal import parse_ta_sdt_txt
from core.qt_widgets import CheckComboBox, PlotWidget
from core.qt_widgets import to_float as _to_float

NAN = float("nan")

# ---------------------------------------------------------------------------
# Scope vocabulary
# ---------------------------------------------------------------------------
# What the Tg / Calculs / export machinery is pointed at. A single segment is
# the only scope a Tg can be computed in -- the other three are views. They are
# string constants rather than an Enum to match the rest of PRISM (KIND_*,
# ROLE_*, X_* in dta_segments) and so a scope can be written straight into a
# CSV cell or a project file without a converter.
SCOPE_SEGMENT = "segment"
SCOPE_ALL_TIME = "all_time"
SCOPE_ALL_HEATING = "all_heating"
SCOPE_ALL_COOLING = "all_cooling"

# A scope SELECTION is stored as one short string -- an aggregate constant, or
# "segment:<index>". One string rather than a (kind, index) pair because the
# same value has to survive three round trips it cannot control: a QComboBox
# item's userData, the per-record scope memory, and `preserve_zoom`'s content
# key, which compares with `==`.
_SEGMENT_KEY_PREFIX = "segment:"

AGGREGATE_SCOPE_LABELS: Dict[str, str] = {
    SCOPE_ALL_TIME: "All data (vs time)",
    SCOPE_ALL_HEATING: "All heating segments",
    SCOPE_ALL_COOLING: "All cooling segments",
}

# ---------------------------------------------------------------------------
# Program tab vocabulary
# ---------------------------------------------------------------------------
METHOD_TABLE_COLUMNS: Tuple[str, ...] = (
    "#", "Step", "Kind", "Rate (°C/min)", "Target (°C)", "Duration (min)",
    "Source", "Matched",
)
"""Columns of the method table, in order.

Every quantity here is in the unit the OPERATOR wrote the method in, which on
a TA export is °C and minutes, and the headers say so literally. The table
deliberately does NOT follow the workspace's °C/K switch: the `Step` column is
the instrument's own line, quoted verbatim, and re-expressing the numeric
columns beside it would put "Ramp 20.00 °C/min to 1300.00 °C" in one cell and
"1573.15" in the next. A correctly labelled °C table next to a kelvin axis is
honest; a half-converted one is not. `_PROGRAM_UNIT_NOTE` says so on screen.

`Duration (min)` is fed from `MethodStep.duration_min` alone and never from
`MethodStep.value`. They were one field in the prototype, and a "Mass flow
100.00 mL/min" step then read as a 100-minute hold (MERGED_DECISIONS M9/D1).
"""

PROGRAM_ROW_DETECTED = "detected"
PROGRAM_ROW_ABSORBED = "absorbed"
PROGRAM_ROW_NOT_EXECUTED = "not executed"
"""The three claims a row of the segment table can make, kept apart because
two of them are routinely confused and the confusion is a false statement
about somebody's experiment.

  * `detected`     the record holds a distinguishable stretch for this step;
  * `absorbed`     the step RAN and left no separate signature -- the four
                   `Isothermal for 20.00 min` holds at 50 °C of the cyclic
                   file, which the furnace spent still coasting downwards past
                   211 / 165 / 117 / 117 °C. They happened; the measurement
                   simply cannot separate them from the cool they happened in;
  * `not executed` the step was programmed and the data never shows it, which
                   is what an aborted or early-stopped run looks like.

Collapsing the last two into one "missing" row would tell a user that four
holds written in their own method never ran.
"""

PROGRAM_TABLE_KEYS: Tuple[str, ...] = ("status",) + SEGMENT_TABLE_COLUMNS
"""Keys of the segment table, in column order: `dta_segments`' own columns with
a leading `status`. The status is a separate column rather than a value smuggled
into `kind`, so that "absorbed" can never be read as a kind of segment."""

PROGRAM_TABLE_HEADERS: Dict[str, str] = {
    "status": "Status", "index": "#", "kind": "Kind", "role": "Role",
    "cycle": "Cycle", "rows": "Rows", "n": "N",
    "t_start": "t start (min)", "t_end": "t end (min)", "dur_min": "Duration (min)",
    "T_start": "T start (°C)", "T_end": "T end (°C)",
    "T_min": "T min (°C)", "T_max": "T max (°C)",
    "rate_mean": "Rate mean (°C/min)", "rate_median": "Rate median (°C/min)",
    "rate_mode": "Rate measured (°C/min)", "rate_nominal": "Rate programmed (°C/min)",
    "fit_r2": "Linearity R²", "monotone": "Monotone frac",
    "reversals": "Reversals", "dup_T": "Duplicate T",
    "control_lost_below": "Control lost below (°C)", "target_gap": "Target gap (°C)",
    "partial": "Partial", "steps": "Steps", "note": "Note",
}


def program_table_column(key: str) -> int:
    """Column index of a `PROGRAM_TABLE_KEYS` name, or -1.

    Exists so a caller -- and a test -- addresses a cell by what it MEANS
    rather than by a literal index that moves the day a column is inserted.
    """
    try:
        return PROGRAM_TABLE_KEYS.index(str(key))
    except ValueError:
        return -1


TARGET_GAP_WARN_C = 5.0
"""A programmed target missed by more than this many °C is called out.

The same number the rate series uses for `setpoint_reached`, deliberately: two
parts of the app must not disagree about whether the furnace arrived. Below it
sit the four heating ramps of the cyclic file, which stop 1.36-1.39 °C short of
1300 °C and then reach the setpoint during the soak that follows -- ordinary
behaviour, not a warning. Above it sit its four cooling ramps, 9.9 to 44.4 °C
short of a 50 °C target they never reached at all.
"""

PROGRAM_TABLE_HEADER_TIPS: Dict[str, str] = {
    "status": ("detected — the record holds its own stretch for this step.\n"
               "absorbed — the step ran and left no separate signature.\n"
               "not executed — the data never shows the step at all.\n\n"
               "The last two are different claims and only one of them says "
               "the step did not happen."),
    "rows": "Row range in the CLEANED arrays, after non-finite rows were dropped.",
    "rate_mode": ("Measured rate, taken as the MODE of the pointwise estimate. "
                  "On a cooling ramp with a passive tail the mean reads "
                  "-14.2 and the median -18.0 against a programmed -20; only "
                  "the mode recovers -19.96."),
    "rate_nominal": ("Programmed rate, signed by the direction the furnace "
                     "actually had to travel. A hold reads 0 because it has "
                     "no programmed rate, not because a rate of 0 was asked for."),
    "target_gap": ("How far short of the programmed target the ramp stopped. "
                   "A hold reads 0 because it has no target temperature at "
                   "all — read it together with the Kind column."),
    "control_lost_below": ("Temperature below which the measured rate fell "
                           "under 90 % of the programmed one: the furnace was "
                           "coasting from there down, so the programmed rate "
                           "describes the top of the ramp and not the bottom. "
                           "Blank means control was never lost."),
    "monotone": "Fraction of the segment over which temperature moves one way.",
    "dup_T": "Points sharing a temperature with another point in the segment.",
    "reversals": "Rows a stable sort by temperature would have to move.",
    "partial": "yes when acquisition stopped inside this segment.",
    "steps": "Written step numbers of the method this segment ran.",
}
"""Tooltips for the segment table's header cells.

Only the columns whose plain reading is WRONG without them. `Target gap` and
`Rate programmed` both read 0 on a hold -- correct in the science layer, where
a hold has no target and no rate -- and "target gap 0" otherwise reads as "hit
it exactly", which is the opposite of what a number that does not exist means.
"""

_PROGRAM_UNIT_NOTE = (
    "Both tables are in °C and minutes — the units the instrument wrote the "
    "method in. They do not follow the °C/K switch on the Tg tab: the Step "
    "column quotes the operator's own line and the Note column quotes the "
    "science layer's own sentence, and neither can be re-expressed without "
    "rewriting text that is shown verbatim."
)

_WARN_FG = "#b4503a"
"""Foreground for the Program tab's warning label. Inline rather than a theme
class because the shared stylesheet has no warning role yet, and a warning the
user cannot tell apart from a caption is a warning that goes unread."""

_STEP_SOURCE_LABELS: Dict[str, str] = {
    SOURCE_ORGMETHOD: "OrgMethod",
    SOURCE_MTH: ".mth",
    SOURCE_BOTH: "both",
    SOURCE_USER: "user",
}


def _join_ints(values: Sequence[Any]) -> str:
    """'6, 10, 14 and 18' — a list of step numbers as a person would read it."""
    items = [str(int(v)) for v in values]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _plural(count: int, noun: str) -> str:
    """'1 step' / '19 steps'. The Program tab's summary is read by a person."""
    n = int(count)
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _fmt_method_number(value: float) -> str:
    """A programmed quantity in TA's own two-decimal format, or blank.

    Two decimals rather than `%g`, so the `Rate` and `Target` columns read the
    same as the verbatim `Step` text beside them ("20.00", not "20"). Anything
    else invites the reader to wonder which of the two cells was rounded.
    """
    v = float(value)
    return "" if not np.isfinite(v) else f"{v:.2f}"


def _fmt_cell(value: Any) -> str:
    """One table cell. Missing is BLANK, never 0 and never 'nan'.

    A NaN rendered as "nan" reads as a broken number; a NaN rendered as 0.0
    reads as a measurement. Blank is the only rendering of "this quantity does
    not exist for this row" that cannot be mistaken for either.
    """
    if value is None:
        return ""
    if isinstance(value, bool):          # before int: bool IS an int
        return "yes" if value else "no"
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:g}"
    return str(value)


def _nan_float(value: Any) -> float:
    """`value` as a float, or NaN — including for None and for junk text.

    NaN rather than 0.0 or an exception: a missing rate and a rate of zero are
    different claims (one is unknown, the other says the furnace was holding),
    and every consumer here already branches on `np.isfinite`.
    """
    try:
        if value is None:
            return NAN
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _baseline_modes(bp: BaselineParams) -> Tuple[str, str]:
    """('point' | 'range' | 'auto') for the LOW and the HIGH baseline.

    One rule in one place: the per-segment export, the batch and the rate
    series all report the baseline mode, and three copies of this expression is
    how one of them ends up saying "auto" about a manual point.
    """
    def mode(point: Any, rng: Any) -> str:
        if point is not None:
            return "point"
        return "range" if rng is not None else "auto"

    return mode(bp.low_point, bp.low_range), mode(bp.high_point, bp.high_range)


# ---------------------------------------------------------------------------
# Result degeneracy
# ---------------------------------------------------------------------------
DEGENERATE_SPREAD = 1e-2
"""Spread below which the three Tg methods have COLLAPSED rather than agreed.

In the abscissa's own unit, so the comparison is always against numbers the
user can see in the result box.

The threshold is measured, not chosen (SPEC_cycles C12/R4, pinned in
`tests/dta/test_dta_segments.py::test_degenerate_tg_collapse_needs_a_1e_2_threshold`):

  * a global sort of the cyclic fixture makes all three methods return
    455.951 °C with a spread of **1.22e-4 °C**; the full 18996-row file does
    the same at 431.758 °C with **4.0e-5 °C**;
  * the healthy single-ramp control `EXAMPLES/DTA_example.txt` spreads
    **3.15 °C** through the widget and 3.43 °C through the science layer.

So 1e-2 sits four orders of magnitude above the collapse and two below the
healthy case. The 1e-6 that was also proposed waves BOTH collapses through,
and the failure it lets ship is the app printing "✓ Methods agree (spread
0.00 °C)" over a meaningless number -- strictly worse than an error, because
it is an endorsement.

Three independent constructions -- two tangent geometries and an extremum of a
derivative -- cannot land on the same number to four decimals on real data.
When they do, they are all reading one degenerate index, and the right report
is a problem, not agreement.
"""


def finite_spread(values: Sequence[Any]) -> float:
    """max - min over the finite entries; NaN when fewer than two are finite.

    NaN rather than 0.0 for the under-determined case, deliberately: a spread
    of 0 is exactly what the degeneracy guard looks for, so returning it when
    only ONE method produced a number would report a collapse that never
    happened. `dta_science.tg_agreement` makes the same distinction (`agree`
    is None below two methods), and the two must not disagree.
    """
    finite = [float(v) for v in values
              if v is not None and np.isfinite(float(v))]
    if len(finite) < 2:
        return NAN
    return float(max(finite) - min(finite))


def degenerate_result_line(spread: float, unit: str) -> str:
    """The sentence shown when the methods collapse. One wording, one place.

    Used by the result box, the figure annotation and (through
    `tg_degenerate`) the CSV, so a reader cannot be told three different
    things about the same number.
    """
    suffix = f" {unit}" if unit else ""
    return (f"⚠ All three methods returned the SAME value (spread "
            f"{spread:.2e}{suffix}). That is not agreement: two tangent "
            "constructions and a derivative extremum cannot coincide to four "
            "decimals on real data, so they have collapsed onto one degenerate "
            "index. Check that the window holds a real transition and that the "
            "scope is a single monotone scan.")


# ---------------------------------------------------------------------------
# Result CSV schema
# ---------------------------------------------------------------------------
LEGACY_RESULT_COLUMNS: Tuple[str, ...] = (
    "file", "sample", "x_col", "y_col", "dy_source", "temp_unit",
    "window_min", "window_max", "smooth_derivative",
    "low_mode", "low_range", "low_point_x",
    "high_mode", "high_range", "high_point_x",
    "Tg_double", "Tg_parallel", "Tg_derivative",
)
"""The 18 columns the result CSV carried before segments existed, in order.

Frozen. Somebody's spreadsheet, script or notebook already reads these names
and this order; the segment columns are APPENDED so that consumer keeps
working unchanged. `test_export_schema_is_backward_compatible` pins the
prefix, which is the contract.

These columns are in the DISPLAY unit, named by `temp_unit`, because that is
what they have always been -- switching the workspace to K and exporting has
always produced kelvin here.
"""

SEGMENT_RESULT_COLUMNS: Tuple[str, ...] = LEGACY_RESULT_COLUMNS + (
    "segment_index", "segment_kind", "segment_role", "cycle_index",
    "segment_label", "program_step",
    "row_start", "row_stop", "n_points",
    "t_start_min", "t_end_min", "T_start_C", "T_end_C",
    "rate_measured_C_min", "rate_programmed_C_min",
    "target_gap_C", "control_lost_below_C", "is_partial",
    "tg_spread", "tg_degenerate", "warnings", "error",
)
"""Full schema: the 18 legacy columns, then WHERE the number came from.

Every appended temperature or rate column carries its unit IN ITS NAME and is
always °C, whatever the workspace's °C/K switch says. That is not an
oversight and it is not inconsistency with the legacy block: a `Segment` is
stored canonically in °C (it is measured once and displayed many times), and a
column called `T_start` whose unit depended on a radio button at export time
would be unreadable six months later. The legacy block keeps its historic
behaviour because changing it would break the consumer this schema exists to
protect; `temp_unit` says which unit those 18 are in.

SPEC_cycles §3 step 10 called this "the 32-column schema". It is 40. The eight
extra are `segment_label`, `target_gap_C`, `control_lost_below_C`,
`is_partial`, `tg_spread`, `tg_degenerate`, `warnings` and `error` -- the
schema was counted before MERGED_DECISIONS M0.1 required every warning to
travel into the CSV, and a warning that stays in the UI while the number
leaves in a file is exactly the failure M0.1 names. A column count is not a
contract; the first 18 names are.

`row_start` / `row_stop` are ORIGINAL DataFrame positions
(`SegmentedRun.row_indices`), never `Segment.row_start`, which indexes the
cleaned arrays -- the same rule and the same column names as
`dta_rate_series`, so the two CSVs can be joined.
"""

BATCH_SCOPE_FILE = "file"
BATCH_SCOPE_EVERY = "every_segment"
BATCH_SCOPE_MATCHING = "matching_segments"

BATCH_SCOPE_LABELS: Dict[str, str] = {
    BATCH_SCOPE_FILE: "Whole file (one row per file)",
    BATCH_SCOPE_EVERY: "Every segment",
    BATCH_SCOPE_MATCHING: "Segments matching the current one",
}

MATCH_RATE_TOL_C_MIN = 0.5
"""How close two measured rates must be to count as "the same rate", in °C/min.

Measured on the real cyclic run: its four heating ramps run at +19.949,
+19.962, +14.991 and +9.988 °C/min. The two nominally-20 ramps differ by
0.013, and the nearest DIFFERENT rate is 5.0 away. A 0.5 °C/min window is
therefore 38x wider than the repeat scatter and 10x narrower than the smallest
real difference -- it cannot split a matched pair and it cannot merge two
programmed rates.

Kind must match too, so a -20 cool never matches a +20 heat: the sign is part
of the identity of a rate, not a detail of formatting.
"""

# ---------------------------------------------------------------------------
# Rate series (Arrhenius / Lasocka) — vocabulary
# ---------------------------------------------------------------------------
# SPEC_rate_series §6 and MERGED_DECISIONS M5. The feature the user asked for
# by name: "add a feature to load data taken at different rates to make the
# arrhenius and lasocka plots" — from separate FILES as well as from the
# segments of one cyclic run.

RATE_TAB_TITLE = "Rate series"
"""Tab caption, and the identity the tab is found by in tests. The tab sits
between "Calculs" and "Export / Batch" (M5); nothing in this file addresses a
tab by number, so inserting it moved no existing code."""

PLOT_PAGE_MAIN = 0
PLOT_PAGE_RATE = 1
"""Indices into `plot_stack`. Page 0 is `self.plot` — the SAME object the Tg
tab has always drawn on, reparented into the stack and nothing else, because
the pinned Qt tests reach `widget.plot.ax` / `.figure` and a rate series must
not cost them a rename. Page 1 is `self.rate_plot`, a second canvas rather
than a second use of the first: a Tg overlay and an Arrhenius fit share no
axes, and clearing one to draw the other is how a user loses the figure they
were reading."""

RATE_QUANTITY_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    ("Tg · parallel tangent", "Tg", "parallel"),
    ("Tg · double tangent", "Tg", "double"),
    ("Tg · |dY| max", "Tg", "deriv"),
    ("Tp · maximum", "Tp", "peak_max"),
    ("Tp · minimum", "Tp", "peak_min"),
)
"""(label, `RateRow.quantity_kind`, `RateRow.method`) for the Quantity combo.

One combo for two orthogonal fields, deliberately: a user picks "the number I
am collecting", which is a quantity AND a construction, and two combos would
allow "Tp · parallel tangent" — a pairing no method in `dta_science` can
produce. `dta_rate_series` keeps the fields separate anyway (`mixed_quantity`
is critical, `mixed_method` is a warn), so nothing downstream loses the
distinction."""

RATE_DEFAULT_WINDOW_C: Tuple[float, float] = (350.0, 700.0)
"""The window a row records when the Tg tab's own window is not a temperature.

Not a guess: it is the same pair `_apply_dataframe` writes into the Tg window
fields on every load, so it is the app's standing default rather than a number
invented here. It is used only when the abscissa in force is the clock or a
signal column, where the Tg fields hold minutes or milliwatts and copying them
into a `window_min_C` cell would be a unit error in an exported file."""

RATE_TABLE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("use", "Use"),
    ("sample", "Sample"),
    ("source", "Source"),
    ("segment", "Segment"),
    ("rate", "q (°C/min)"),
    ("quantity", "Quantity"),
    ("value", "Value"),
    ("from", "From"),
    ("history", "History"),
    ("mass", "Δm (%)"),
    ("flags", "Flags"),
    ("note", "Note / error"),
)
"""(key, header) of the row table, in order. SPEC_rate_series §6.

"Value" carries no unit in this tuple because its header follows the °C/K
switch and is written by `_rate_refresh_table`; every other header is fixed."""

RATE_COL_USE = 0
RATE_COL_SAMPLE = 1
RATE_COL_SOURCE = 2
RATE_COL_SEGMENT = 3
RATE_COL_RATE = 4
RATE_COL_QUANTITY = 5
RATE_COL_VALUE = 6
RATE_COL_FROM = 7
RATE_COL_HISTORY = 8
RATE_COL_MASS = 9
RATE_COL_FLAGS = 10
RATE_COL_NOTE = 11

RATE_EDITABLE_COLUMNS: Tuple[int, ...] = (RATE_COL_SAMPLE, RATE_COL_RATE,
                                          RATE_COL_VALUE, RATE_COL_NOTE)
"""The four typable columns, plus the `Use` checkbox which is checkable rather
than editable. Everything else is provenance: a measured rate, a segment label,
a cycle number and a mass loss are facts about the source, and an editable copy
of a measured number is a number nobody can trace. `rate_source` is the fifth
user-editable field and it is set from the row's context menu instead of a
column, because it is a choice among four named candidates rather than free
text — see `_on_rate_table_menu`."""

_ROW_WARN_FG = _WARN_FG
"""Table/figure warning tone. Deliberately the same value as the Program tab's
`_WARN_FG`: one warning colour in the workspace, under the name
SPEC_rate_series gives it, so the rug ticks and the failed cells match the
warning text a user has already learnt to read."""

RATE_TG_METHOD_LABELS: Dict[str, str] = {
    "parallel": "Parallel", "double": "Double", "deriv": "|dY| max",
}
"""`RateRow.method` -> the prefix `_measure_tg` writes in front of that
construction's failure message. Used to pull ONE method's sentence out of the
joined error string; see `rate_method_error`."""

RATE_PEAK_METHODS: Tuple[str, ...] = ("peak_max", "peak_min")
"""The two `RateRow.method` values that mean "the temperature of an extremum"
rather than a tangent construction."""

RATE_ANNOTATION_MAX_WARNINGS = 4
"""How many warnings the figure annotation carries before it says "… n more".

The corner of an axes is finite, `sort_warnings` guarantees the ones kept are
the worst, and the number dropped is stated — while the banner and the Fit
panel both carry the complete list, so nothing is only in the elided place."""

RATE_CURVE_POINTS = 240
"""Samples in the drawn fit curve. The fitted line is straight in the MODEL's
own space, which for the two Kissinger forms and Augis-Bennett is not straight
against |q|, so the curve is sampled rather than drawn end to end."""

RATE_CURVE_Q_PAD = 10.0
"""How far outside the measured rate range the drawn curve may stray, as a
factor. A fit whose slope is nearly zero sends `q(T)` to 1e30 for a
temperature one degree outside the data; clipping keeps a log axis readable
instead of compressing every measured point into one pixel."""

RATE_MODEL_Q_OF_T: Dict[str, Any] = {
    "lasocka":       lambda T_C, m, b, T0_C: np.exp((T_C - b) / m),
    "moynihan":      lambda T_C, m, b, T0_C: np.exp(b + m / to_kelvin(T_C)),
    "fwo":           lambda T_C, m, b, T0_C: np.exp(b + m / to_kelvin(T_C)),
    "kissinger_tg":  lambda T_C, m, b, T0_C: (to_kelvin(T_C) ** 2
                                              * np.exp(b + m / to_kelvin(T_C))),
    "kissinger":     lambda T_C, m, b, T0_C: (to_kelvin(T_C) ** 2
                                              * np.exp(b + m / to_kelvin(T_C))),
    "augis_bennett": lambda T_C, m, b, T0_C: ((T_C - T0_C)
                                              * np.exp(b + m / to_kelvin(T_C))),
}
"""`q(T)` for each fitted model: the rate at which this fit says the quantity
lands at temperature `T` (°C). `m` and `b` are the fit's own slope and
intercept, in the model's own space.

Parameterised by TEMPERATURE and not by rate, which is the entire reason this
table is six readable lines. Rearranging `y = b + m·x` for `q` is closed form
for every one of the six models — the exponential simply falls out — whereas
three of them (both Kissinger forms and Augis-Bennett) are TRANSCENDENTAL in
T and would each need a numeric inversion. `dta_kinetics` already owns exactly
one such inversion (`_solve_T_at_q`, for the back-prediction at q_ref); a
second copy in the presentation layer would be a second place to get a sign
wrong, and it would be the copy that draws the line the user reads. Verified
against `dta_kinetics` on data lying exactly on each model's own line: the
curve reproduces the fitted rates to 7e-15 relative.

The curve is generated by sweeping T across the span of the fitted values and
asking each model which rate it assigns, so it cannot drift from the numbers
in the annotation: both come from the same `slope` and `intercept`.

Rates come out in °C/min with no conversion, because a rate in K/min and in
°C/min is the same number (`dta_kinetics.to_kelvin`'s unit law: the +273.15
offset cancels in a temperature interval per unit time). Temperatures are
converted, in the one place each model needs it, through `to_kelvin` — never
through a literal here."""

# Prefix marking a record key that came from the library's stable Spectrum.id.
# Anything else is a display title, which is a WEAKER identity (see
# DtaWorkspace._record_key) and is therefore never used as a cache key.
_RECORD_ID_PREFIX = "id:"
_RECORD_TITLE_PREFIX = "title:"

_CURRENT_RECORD = object()
"""Sentinel default for `_prepare_xy(run=...)`.

`None` is a MEANINGFUL value there -- it says "this record could not be
segmented, prepare it as one whole record" -- so it cannot double as "argument
not supplied". Without the distinction the batch loop would silently fall back
to the workspace's own run and prepare one file's columns against another
file's segment boundaries.
"""


def _unit_from_col(col: str) -> str:
    """The temperature unit a column's own name claims, or "" if it is not one.

    A CLOCK column is rejected before anything else, because the °C test below
    ends with `"c)" in cl` and `Time (sec)` contains `c)`. That made a seconds
    clock answer "°C": with the display switch on K the axis got +273.15 added
    to it, and the misclassification also made `_is_temperature_col` true, so
    selecting a hold segment refused the clock -- the one abscissa that IS
    valid on a hold. Call sites additionally ask `_looks_like_time` first where
    they must scale a clock, but that ordering must not be what keeps this
    correct: a future call site that asked this function alone would reproduce
    the bug, so the guard belongs here, once.
    """
    cl = col.lower()
    if _looks_like_time(col):
        return ""
    # Kelvin markers are checked FIRST and returned immediately: the older
    # ordering checked "temp"/"temperature" first, which matches a column
    # literally named "Temperature (K)" too and always misclassified it as
    # °C before the Kelvin branch ever ran. Real DTA exports are almost
    # always °C (TA Instruments' own convention, see io_universal.py's
    # canonical "T_C" key), so °C stays the fallback for an ambiguous
    # generic "Temp"/"Temperature" name with no explicit unit marker.
    if "kelvin" in cl or "(k)" in cl or cl.strip().endswith(" k"):
        return "K"
    if "°c" in cl or "temperature" in cl or "temp" in cl or "c)" in cl:
        return "°C"
    return ""


def _shift_temperature(value, old_unit: str, new_unit: str):
    """Convert a scalar float or ndarray from old_unit to new_unit ('°C'
    or 'K'). A no-op when the units already match or either is blank/
    unrecognized. Kelvin and Celsius differ only by a constant +273.15
    offset, so this is the one place that offset is applied -- everything
    else (axis data, window/baseline fields, exported numbers) routes
    through this so a switch can never apply it inconsistently in two
    places. Deliberately NOT used for differences/spreads (e.g. the Tg
    methods' agreement threshold): a difference in °C already equals the
    same difference in K, since the constant offset cancels out."""
    if not old_unit or not new_unit or old_unit == new_unit:
        return value
    if old_unit == "°C" and new_unit == "K":
        return value + 273.15
    if old_unit == "K" and new_unit == "°C":
        return value - 273.15
    return value


def _is_temperature_col(col: str) -> bool:
    """True when this column carries a temperature, in either unit.

    Used to decide whether the "temperature is not an abscissa on a hold" rule
    applies to the user's chosen X. It is a rule about TEMPERATURE, not about
    every column somebody might put on the X axis, so a signal column plotted
    against a hold must not inherit the refusal.
    """
    return _unit_from_col(col) in ("°C", "K")


def _looks_like_time(col: str) -> bool:
    """True for the acquisition clock column (`Time (min)` in every TA export)."""
    return "time" in str(col).lower()


def _looks_like_temperature(col: str) -> bool:
    """True for the furnace temperature column, false for a temperature DIFFERENCE.

    `Temperature Difference (°C)` is the DTA signal, not the abscissa, and it is
    a temperature by unit; segmenting a run on it would produce nonsense. The
    canonical map is consulted first wherever one exists (see
    `_time_temp_columns`); this keyword rule is the fallback for the paths that
    call `parse_ta_sdt_txt` directly and never see a parser `meta`.
    """
    cl = str(col).lower()
    if "difference" in cl or "delta" in cl:
        return False
    return "temp" in cl and _is_temperature_col(col)


def _time_temp_columns(colnames: Sequence[str],
                       canonical: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """`(time column, temperature column)`, either one `""` when absent.

    Prefers `meta["canonical_map"]`, which io_universal already resolved from
    the `Sig<n>` header lines, and falls back to keyword matching for
    `load_file` / the batch loop, which call `parse_ta_sdt_txt` directly and so
    have no parser meta to read.
    """
    names = [str(c) for c in colnames]
    canon = canonical or {}
    t_col = str(canon.get("time_min") or "")
    T_col = str(canon.get("T_C") or "")
    if t_col not in names:
        t_col = next((c for c in names if _looks_like_time(c)), "")
    if T_col not in names:
        T_col = next((c for c in names if _looks_like_temperature(c)), "")
    return t_col, T_col


def _header_lines_of(header: Any) -> List[List[str]]:
    """`[[key, value], ...]` in file order from a `TAHeader`, `[]` from a plain dict.

    A plain dict keeps one value per repeated key, so a 19-step cyclic method
    collapses to its first `OrgMethod` line. `TAHeader.lines` is the only place
    the other 18 survive, and this is the adapter that carries them into the
    `meta` dict the workspace stores.
    """
    out: List[List[str]] = []
    for item in (getattr(header, "lines", ()) or ()):
        try:
            key, value = item
        except (TypeError, ValueError):
            continue
        out.append([str(key), str(value)])
    return out


def _normalize_header_lines(lines: Any, header: Any) -> List[List[str]]:
    """Header lines from a parser `meta`, falling back to the header object.

    Both forms reach the workspace: a library record carries
    `meta["header_lines"]` (plain lists, because `_json_safe` flattens a
    `TAHeader` on a project save), while `load_file` holds the `TAHeader`
    itself. Malformed entries are skipped rather than raised on -- this runs on
    the file-loading path, where a thinner program is a far better outcome than
    a failed import.
    """
    out: List[List[str]] = []
    for item in (lines or ()):
        if isinstance(item, str):
            continue
        try:
            key, value = item
        except (TypeError, ValueError):
            continue
        out.append([str(key), str(value)])
    return out or _header_lines_of(header)


def _resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """`name` as this frame spells it, or None.

    Exact match first, then case-insensitive. A batch is pointed at files
    exported on different days and TA's own column casing is not stable across
    module versions ("Temperature (C)" / "temperature (C)"), so an exact-only
    match refuses files that hold exactly the right column.
    """
    if not name:
        return None
    if name in df.columns:
        return name
    lowered = str(name).lower()
    for col in df.columns:
        if str(col).lower() == lowered:
            return str(col)
    return None


def _interp_ascending(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    """`np.interp` that survives a descending abscissa. NaN when it cannot.

    `np.interp` REQUIRES ascending `xp` and does not check: handed a
    descending array it returns the endpoint value silently, so a Tg marker on
    a cooling view would be pinned to the end of the curve and look like a
    real measurement. Segment preparation sorts ascending, but this overlay is
    also reached from views that plot traversal order, and a marker in the
    wrong place is worse than no marker.

    Sorting only when needed keeps the common, already-ascending case free.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size == 0 or xa.size != ya.size:
        return NAN
    if xa.size > 1 and not bool(np.all(np.diff(xa) >= 0)):
        order = np.argsort(xa, kind="stable")
        xa, ya = xa[order], ya[order]
    return float(np.interp(float(xq), xa, ya))


# ---------------------------------------------------------------------------
# Rate series — cell rendering
# ---------------------------------------------------------------------------
# Module-level and widget-free, so the wording of every cell can be pinned by a
# test without building a workspace, and so one rule renders a row in the table
# and (step 6) in the figure legend.


def rate_quantity_label(row: "RateRow") -> str:
    """'Tg · parallel' — the Quantity cell.

    Reads the row's own two fields rather than the combo: a series assembled
    over two sittings can hold rows of both kinds, and `mixed_quantity` is the
    critical warning that says so. A cell that showed the CURRENT combo would
    hide exactly that.
    """
    kind = str(row.quantity_kind or "").strip() or "?"
    method = str(row.method or "").strip()
    return f"{kind} · {method}" if method else kind


def rate_history_text(row: "RateRow") -> str:
    """'virgin' / 'cycle 3 · 2 prior heats' / '' when nothing is known.

    Blank rather than "virgin" when the row carries no thermal history at all
    (a hand-typed literature value, `state == "unknown"`): claiming a specimen
    was virgin because nobody said otherwise is an assertion about an unknown
    fact, and this column is read as evidence when a Lasocka fit comes out
    non-monotonic.
    """
    prior = int(row.prior_heats or 0)
    state = str(row.state or "").strip().lower()
    if prior <= 0:
        return "virgin" if state == "virgin" else ""
    bits = []
    if int(row.cycle_index or 0) > 0:
        bits.append(f"cycle {int(row.cycle_index)}")
    bits.append(f"{prior} prior {'heat' if prior == 1 else 'heats'}")
    return " · ".join(bits)


def rate_note_display(row: "RateRow") -> str:
    """The "Note / error" cell: the verbatim failure first, then the note.

    `error` and `note` are SEPARATE fields on the row and this is the only
    place they are shown together, joined by ' | '. Editing the cell writes the
    note half only (`_rate_note_from_cell`), so a user who annotates a failed
    row cannot delete the reason it failed — which is the one sentence that
    tells the next reader whether the failure was the data or the settings.
    """
    err = str(row.error or "").strip()
    note = str(row.note or "").strip()
    if err and note:
        return f"{err} | {note}"
    return err or note


def rate_method_error(joined_error: str, method: str) -> str:
    """The sentence the CHOSEN Tg construction produced, out of `_measure_tg`'s
    joined error string.

    `_measure_tg` runs all three constructions and joins their failures with
    " | " — right for the Tg tab, which shows all three side by side, and wrong
    for a rate row, which was measured with exactly one of them. A row measured
    as "parallel tangent" whose error cell quoted the double tangent's
    complaint would send the user to adjust the wrong baseline, and on this
    dataset that is not hypothetical: heat 2 fails on the HIGH baseline of the
    parallel construction while the double tangent fails for its own reason.

    The whole string is the fallback when no prefix matches, because dropping a
    failure message is worse than showing one sentence too many.
    """
    text = str(joined_error or "").strip()
    if not text:
        return ""
    prefix = RATE_TG_METHOD_LABELS.get(str(method or ""), "")
    if prefix:
        for part in text.split(" | "):
            part = part.strip()
            if part.startswith(prefix + ":"):
                return part[len(prefix) + 1:].strip() or part
    return text


def _rate_note_from_cell(text: str, row: "RateRow") -> str:
    """Inverse of `rate_note_display`: the note half of an edited cell.

    A user editing a failed row's cell is handed "<error> | <note>" to type
    into. Stripping a leading copy of the error means the round trip is stable
    however they edit it — append to the end, or clear the lot and type one
    word — instead of accumulating the error text into the note on every edit.
    """
    typed = str(text or "").strip()
    err = str(row.error or "").strip()
    if err and typed.startswith(err):
        typed = typed[len(err):].lstrip()
        if typed.startswith("|"):
            typed = typed[1:].lstrip()
    return typed


@dataclass(frozen=True)
class DtaScope:
    """What the workspace is currently looking at, and what that permits.

    `computable` is the single source of truth for "may a Tg / integral / result
    row be produced here". It is False for every aggregate view, because the
    whole reason this module grew a scope is that a Tg computed across four
    heating ramps and four cooling ramps of one cyclic run is not a rough
    number, it is a meaningless one (all three methods collapse onto 431.758 °C
    with a 4e-5 spread while the app reports agreement).

    `x_kind` is a FORCED abscissa (`X_TIME` on an aggregate view, `X_TIME` on a
    hold) or `""` when the user's choice stands. It is enforced by which column
    the X combo offers, not inside the array adapter: the adapter is handed an
    array and reads the kind off the column that produced it, so a forced kind
    can never be applied to the wrong numbers.
    """

    kind: str                       # SCOPE_*
    segments: Tuple[Segment, ...]
    label: str
    computable: bool
    x_kind: str = ""

    @property
    def segment(self) -> Optional[Segment]:
        """The one segment this scope resolves to, or None for an aggregate."""
        return self.segments[0] if len(self.segments) == 1 else None

    @property
    def is_single(self) -> bool:
        """True when exactly one segment is in scope (the computable case)."""
        return len(self.segments) == 1


@dataclass
class DtaSegmentResult:
    """Everything one segment's Tg computation produced, kept per segment.

    Results are held per segment rather than per widget so that arrowing
    through a cyclic run does not silently show segment 1's Tg over segment 7's
    curve. `DtaWorkspace.res_double` / `res_parallel` / `tg_deriv` are
    PROPERTIES onto the current segment's entry, because five of the pinned Qt
    tests read them directly (counted, not assumed: grep
    `.res_double|.res_parallel|.tg_deriv` in tests/dta/test_qt_dta.py) and one
    storage with two names cannot drift apart the way two stores would.

    Everything after `degenerate` is the SETTING the three numbers were
    produced under, not decoration. A stored result has to be re-displayable
    and exportable later, and a Tg without the window, the signal and the unit
    it was measured with is a number nobody can check. `unit` in particular is
    what lets the °C/K switch tell which stored results it has just invalidated.
    """

    res_double: Optional[TgDoubleTangentResult] = None
    res_parallel: Optional[TgParallelTangentResult] = None
    tg_deriv: Optional[float] = None
    window: Tuple[float, float] = (NAN, NAN)
    """Tg window, in the ABSCISSA's own unit -- °C, K, or minutes on a hold.

    Deliberately not named `window_C` (as SPEC_cycles §2.5 sketched it):
    `dta_rate_series.RateRow.window_C` really is °C because the science layer
    is canonically °C, whereas this one is whatever the user is looking at. A
    `_C` suffix over a kelvin value is the exact class of mislabelling the
    canonical storage exists to prevent.
    """
    baseline: Optional[BaselineParams] = None
    degenerate: bool = False
    spread: float = NAN
    """max - min of the finite Tg values, in the same unit as `window`."""
    unit: str = ""
    """`_result_unit_label` at compute time: "°C", "K", or "" off a
    non-temperature abscissa. Empty is meaningful -- it says the numbers are
    not temperatures at all and a °C/K switch must leave them alone."""
    x_col: str = ""
    y_col: str = ""
    dy_col: str = "(same)"
    smooth_derivative: bool = False
    error: str = ""
    """Why nothing was produced, verbatim. Carried rather than raised so a
    batch of forty segments is not lost to one that has too few baseline
    points; `SEGMENT_RESULT_COLUMNS` exports it."""
    warnings: Tuple[str, ...] = ()
    """Measured reasons to distrust these numbers -- a window outside the
    segment, a missed setpoint, partial acquisition. Never a reason to refuse
    (MERGED_DECISIONS M0.1): they travel into the result box, the figure and
    the CSV, and gate nothing."""

    @property
    def tg_values(self) -> Tuple[float, float, float]:
        """`(double, parallel, |dY| max)` as plain floats, NaN where missing."""
        td = self.res_double.tg if self.res_double is not None else NAN
        tp = self.res_parallel.tg if self.res_parallel is not None else NAN
        tx = self.tg_deriv if self.tg_deriv is not None else NAN
        return (float(td), float(tp), float(tx))

    @property
    def has_values(self) -> bool:
        """True when at least one method produced a number."""
        return any(np.isfinite(v) for v in self.tg_values)


def _whole_record_segment(n_rows: int) -> Segment:
    """A stand-in `Segment` spanning the whole record, index 0.

    Used ONLY as the carrier of the array-preparation rule when a record could
    not be segmented at all -- no clock column, no temperature column, or a
    parse that produced neither. It exists so that `prepare_segment_xy` stays
    the single owner of masking, sorting and duplicate collapse (C7): the
    alternative is a second copy of the rule inlined here, which is exactly the
    three-copies-of-argsort problem this rework removes.

    `index = 0` marks it as not-a-real-segment (real segments are 1-based), and
    `kind = heat` selects the "sortable abscissa" rule -- ascending stable sort
    plus duplicate collapse -- which is precisely what the deleted global
    `argsort` did. The measured fields are NaN because nothing measured them;
    nothing may report this segment to the user.
    """
    n = max(int(n_rows), 0)
    return Segment(
        index=0, kind=KIND_HEAT, role=ROLE_HEAT, cycle_index=1,
        row_start=0, row_stop=n, n_points=n,
        t_start=NAN, t_end=NAN, T_start=NAN, T_end=NAN, T_min=NAN, T_max=NAN,
        rate_mean=NAN, rate_median=NAN, rate_mode=NAN, rate_fit_r2=NAN,
        control_lost_below_C=NAN, monotone_fraction=NAN,
        n_reversals=0, n_duplicate_T=0, is_partial=False,
        note="whole record; this file was not segmented",
    )


def segment_scope_key(index: int) -> str:
    """The scope key naming one segment by its 1-based `Segment.index`."""
    return f"{_SEGMENT_KEY_PREFIX}{int(index)}"


def scope_key_segment_index(key: Any) -> Optional[int]:
    """The segment index a scope key names, or None for an aggregate key."""
    text = str(key or "")
    if not text.startswith(_SEGMENT_KEY_PREFIX):
        return None
    try:
        return int(text[len(_SEGMENT_KEY_PREFIX):])
    except ValueError:
        return None


def _seg_rate(seg: Segment) -> float:
    """Measured rate to display: the mode, falling back to the mean.

    Same precedence as `segment_label`, so a rate never reads differently in
    the combo than it does in the note beneath it.
    """
    rate = float(seg.rate_mode)
    return rate if np.isfinite(rate) else float(seg.rate_mean)


def _single_segment_scope(seg: Segment) -> DtaScope:
    """Scope covering exactly one segment -- the only computable shape.

    A HOLD forces the time abscissa. That is not a preference: after a
    per-segment sort the four measured holds of the real cyclic file still
    carry 2-15 duplicate temperatures and produce 4-24 non-finite derivative
    values, `inf` among them. `prepare_segment_xy` refuses the combination
    outright; forcing the X combo is what stops the user reaching that refusal
    instead of a plot.
    """
    return DtaScope(kind=SCOPE_SEGMENT, segments=(seg,),
                    label=segment_label(seg), computable=True,
                    x_kind=(X_TIME if seg.is_hold else ""))


def default_scope_key(run: Optional[SegmentedRun]) -> str:
    """The scope a record opens in when nothing is remembered for it.

    A multi-segment record opens on the whole run rather than on segment #1,
    because the first thing a user needs to see on a cyclic file is that it IS
    cyclic -- opening straight into one ramp hides eleven others and invites
    exactly the "why is my Tg 431.758" question this rework exists to answer.
    """
    if run is None or not run.segments:
        return SCOPE_ALL_TIME
    if len(run.segments) == 1:
        return segment_scope_key(run.segments[0].index)
    return SCOPE_ALL_TIME


def scope_from_key(key: Any, run: Optional[SegmentedRun]) -> DtaScope:
    """Resolve a stored scope key against THIS run. Never raises.

    Always re-validated rather than trusted: a key is remembered per record and
    the record behind it can be replaced by one with fewer segments, so an
    unresolvable key falls back to the record's own default. The alternative --
    an IndexError, or worse a segment belonging to another file -- is the
    failure mode `_run_cache`'s id rule exists to prevent, and it must not
    creep back in through the selection.

    Four shapes come out:

      * no segmentation (no clock or no temperature column) -> the whole
        record, COMPUTABLE, because refusing to analyse a file the app has
        always analysed would be a regression, not a safeguard;
      * exactly one segment -> that segment. Every single-ramp file, including
        `EXAMPLES/DTA_example.txt`, lands here whatever the key says, and the
        segment covers all its rows, so the prepared arrays are bit-identical
        to the pre-rework ones and the pinned Tg 354.4666531002641 stays put;
      * one segment of a multi-segment run -> that segment, computable;
      * anything covering two or more segments -> not computable. This is an
        ARITHMETIC limit, not a physical objection: a tangent construction
        needs one monotone scan, and eight passes over 19-1299 C are not one.
        Nothing is blocked from being *processed* -- the view still draws, the
        rate series still reads every segment -- only the tangent maths that
        has no meaning here is switched off, and the reason is written in
        `scope_note_label` rather than popped in a dialog.
    """
    if run is None or not run.segments:
        return DtaScope(kind=SCOPE_ALL_TIME, segments=(),
                        label="whole record (not segmented)",
                        computable=True, x_kind="")
    segments = tuple(run.segments)
    if len(segments) == 1:
        return _single_segment_scope(segments[0])

    index = scope_key_segment_index(key)
    if index is not None:
        seg = run.segment_by_index(index)
        if seg is not None:
            return _single_segment_scope(seg)

    if key in (SCOPE_ALL_HEATING, SCOPE_ALL_COOLING):
        kind = KIND_HEAT if key == SCOPE_ALL_HEATING else KIND_COOL
        chosen = run.by_kind(kind)
        if len(chosen) == 1:
            # One ramp of that kind IS a single monotone scan. Presenting it as
            # an uncomputable aggregate would be a false refusal. The combo no
            # longer OFFERS a singleton aggregate (see `_available_scope_keys`),
            # so this is reached only by a remembered key or a direct call --
            # both of which must still resolve to the honest answer.
            return _single_segment_scope(chosen[0])
        if chosen:
            rates = ", ".join(f"{_seg_rate(s):+.1f}" for s in chosen)
            word = "heating" if kind == KIND_HEAT else "cooling"
            return DtaScope(
                kind=str(key), segments=chosen,
                label=(f"all {word} segments — {len(chosen)} of {len(segments)}, "
                       f"{rates} °C/min"),
                computable=False, x_kind="")
        # Nothing of that kind in this run: fall through to the whole run. An
        # EMPTY scope must never be produced -- `_rule_segment` reads "no
        # segments" as "this record was not segmented at all" and would
        # silently prepare the whole record.

    span = segments[-1].t_end - segments[0].t_start
    return DtaScope(
        kind=SCOPE_ALL_TIME, segments=segments,
        label=(f"whole run — {len(segments)} segments, "
               f"{run.n_cycles} cycle(s), {span:.1f} min"),
        computable=False, x_kind=X_TIME,
    )


def _scope_for_run(run: Optional[SegmentedRun]) -> DtaScope:
    """The scope a record opens in, derived from its segmentation alone.

    Kept as the no-selection entry point: the batch loop and `_prepare_xy`
    handed a foreign run have no stored key to resolve.
    """
    return scope_from_key(default_scope_key(run), run)


class DtaWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, records: Optional[List[Dict[str, Any]]] = None):
        super().__init__(parent)
        self.records: List[Dict[str, Any]] = records or []
        self._record_lookup: Dict[str, Dict[str, Any]] = {}
        for i, rec in enumerate(self.records):
            key = rec.get("title") or f"Record {i + 1}"
            base, suffix = key, 2
            while key in self._record_lookup:
                key = f"{base} ({suffix})"
                suffix += 1
            self._record_lookup[key] = rec

        self.path: Optional[Path] = None
        self.header: Dict[str, str] = {}
        self.colnames: List[str] = []
        self.df: Optional[pd.DataFrame] = None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

        # --- segmentation / programmed method -------------------------------
        self.run: Optional[SegmentedRun] = None
        """Segmentation of the loaded record; None when it could not be made."""
        self.program: Optional[MethodProgram] = None
        """Method read from the header's OrgMethod lines -- the authoritative
        one, because it is what the instrument wrote next to the data. Never
        None once a record is loaded: `program_from_header` reports failure
        through `status`/`error` instead of raising or returning nothing."""
        self.mth_program: Optional[MethodProgram] = None
        """A `.mth` the user loaded by hand, kept only to cross-check the
        header program. It never overrides it."""
        self.mth_path: Optional[Path] = None
        self.mth_comparison: Optional[ProgramComparison] = None
        """`compare_programs(self.program, self.mth_program)`, recomputed every
        time either side changes -- including on a record switch, because a
        `.mth` that matched the previous record may well not match this one and
        a stale verdict is worse than no verdict."""
        self._seg_param_note: str = ""
        """Why a typed detection knob was refused, if one was. Shown under the
        Re-detect button; never a dialog, because a rejected keystroke is not
        an event worth interrupting anyone for."""
        self.seg_results: Dict[int, DtaSegmentResult] = {}
        """Tg results per `Segment.index`, so one segment's numbers can never
        be shown over another segment's curve."""
        self.header_lines: List[List[str]] = []
        """Every header line in file order. A plain dict keeps only the first
        value of a repeated key, which collapses a 19-step cyclic method to
        `1: Select gas 1`; this is what `program_from_header` needs instead."""
        self._record_id: Optional[str] = None
        self._last_prepared: Optional[PreparedXY] = None
        """The arrays `_prepare_xy` produced most recently, and HOW it made
        them. Read by `_result_warnings` and by `_rate_arrays_for`, both of
        which run immediately after their own `_prepare_xy` call and need the
        duplicate-collapse count that `_measure_tg`'s (x, y) arrays no longer
        carry. Coupled to `_prepare_xy` being the ONE place arrays are made,
        which is the invariant that makes "most recently" unambiguous."""
        self._rate_prep_extras: Dict[str, float] = {}
        """Preparation facts for the row `_rate_compute_one` is measuring:
        how much of its curve was duplicate-collapsed, and whether its stored
        row span could be reproduced. Merged into `RateRow.extras`, so they
        travel into the rows CSV and the project file with the value."""
        self._run_cache: Dict[str, SegmentedRun] = {}
        self._scope_memory: Dict[str, Any] = {}
        """Scope last chosen for a record, keyed as `_record_key`. Only
        id-backed keys belong in here, for the same reason as `_run_cache`:
        one file's choice applied to another file's data is a wrong number
        with no visible cause."""
        self._scope_key: str = SCOPE_ALL_TIME
        """Which scope the user picked, as a `segment_scope_key` / SCOPE_*
        string. Validated against the run in hand on every read, never
        trusted -- see `scope_from_key`."""
        self._x_forced_time: bool = False
        """True while the X axis sits on the clock because a scope PUT it
        there, not because the user chose it. Without the distinction the
        forced time axis is a one-way door: a cyclic file opens on the whole
        run (time), and every ramp segment selected afterwards would be
        plotted against time, which is not the abscissa a Tg is read on."""
        self._suspend_scope: bool = False
        """True while `_refresh_segment_combo` is rebuilding the list.
        `clear()` and `addItem()` each emit `currentIndexChanged`, and the
        handler resets the window, re-runs the guardrails and redraws; without
        this flag, loading a 12-segment file would run that chain a dozen
        times before the user has touched anything (R5)."""
        self._window_user_edited: bool = False
        """Set by the user typing in the Tg window fields. The segment-derived
        default window must never overwrite a value the user chose, and it must
        never fire on a single-segment file at all -- the 350/700 defaults were
        authored for exactly that case and the pinned Tg depends on them."""
        self._seg_config = SegmentationConfig()
        self._progress_dialog: Optional[QProgressDialog] = None
        """The live `QProgressDialog` of a running export or batch, so the
        wrapper that opened it can always take it down -- including from a
        `finally` after the callable half raised."""
        self._canonical_map: Dict[str, Any] = {}
        """`meta["canonical_map"]` of the loaded record: how the clock and the
        furnace-temperature columns are found without guessing from names."""

        # `res_double` / `res_parallel` / `tg_deriv` are PROPERTIES onto
        # `seg_results[<current segment>]` -- see the property block below
        # `_compute`. There is deliberately no instance attribute of those
        # names: two stores for one number is how segment 1's Tg ends up on
        # screen over segment 7's curve.
        self._prev_temp_unit = "°C"  # tracked so the toggle handler always
        # knows the OLD unit to convert range fields from, regardless of
        # which of the two buttons Qt happens to signal.

        # --- rate series (Arrhenius / Lasocka) ------------------------------
        # Built BEFORE _build_ui, because building the Rate series tab writes
        # its notes, and a note that counts the rows needs the series to exist.
        self.rate_series: RateSeries = RateSeries(quantity_kind="Tg")
        """The multi-rate series: rows in INSERTION order, never sorted in
        place. "Row 3" in a warning is the third line on screen."""
        self._rate_add_warnings: List[SeriesWarning] = []
        """Warnings raised by the four add paths (`no_heating_segment`,
        `ambiguous_file_rate`), kept because they explain state the user can
        see — chiefly rows that arrived switched OFF. Filtered against the live
        rows on every render, so removing a row removes its warning."""
        self.rate_warnings: List[SeriesWarning] = []
        """What the banner is currently showing, worst first. Public: a test
        reads it instead of parsing the label."""
        self._rate_frames: Dict[str, pd.DataFrame] = {}
        """Source frame per ROW SOURCE (`_rate_frame_key`), not per row: four
        segments of one cyclic file share one frame, and caching it per row
        would hold four copies of a 5 MB export."""
        self._rate_canonical: Dict[str, Dict[str, Any]] = {}
        """`meta["canonical_map"]` per row source, alongside `_rate_frames`.
        Parallel to it rather than tupled into it because three tests and the
        clear path already treat `_rate_frames` as {key: DataFrame}. A source
        read from disk at compute time has no parser meta and simply has no
        entry here; `_time_temp_columns` then falls back to its keyword rule,
        which is the same rule `dta_rate_series` used when the rows were
        built."""
        self.rate_last_fit: Optional[KineticFitResult] = None
        """The most recent `fit_series` result, or None before anything could be
        fitted. Public, so a test and the export step read the result itself
        instead of parsing a label."""
        self.rate_annotation_text: str = ""
        """The exact block drawn INTO the rate figure. Stored before it is
        drawn and asserted to be among `rate_plot.ax.texts`, so what a saved
        PNG carries and what this attribute reports cannot drift apart."""
        self._rate_notice: str = ""
        """One line about the last add that is not a property of any row —
        "2 of 3 checked records are no longer in the library". Shown in the
        rows note; never a dialog."""
        self._rate_table_updating: bool = False
        """True while `_rate_refresh_table` writes cells. `itemChanged` is
        blocked as well; this flag is the second half of the R3 guard, for any
        path that writes a cell with signals live."""

        self._build_ui()
        if self.records:
            self.record_combo.setCurrentIndex(0)
            self._load_selected_record()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(440)
        left_layout = QVBoxLayout(left)

        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        # Tab order is  Tg | Program | Calculs | Rate series | Export / Batch
        # (MERGED_DECISIONS M5). The Program tab is built after the Tg tab
        # because it drives `segment_combo`, which the Tg tab owns. Nothing in
        # this file addresses a tab by number, so inserting "Rate series"
        # fourth moved no existing code.
        self.tabs.addTab(self._build_tg_tab(), "Tg")
        self.tabs.addTab(self._build_program_tab(), "Program")
        self.tabs.addTab(self._build_calc_tab(), "Calculs")
        self._rate_tab = self._build_rate_tab()
        """Kept by IDENTITY rather than by index: `_on_tab_changed` asks which
        widget is showing, so a later tab insertion cannot silently point the
        plot stack at the wrong page."""
        self.tabs.addTab(self._rate_tab, RATE_TAB_TITLE)
        self.tabs.addTab(self._build_batch_tab(), "Export / Batch")

        self.status_label = QLabel("")
        self.status_label.setObjectName("SectionNote")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        # Two canvases, one slot. `self.plot` keeps its name, its identity and
        # its type -- it is REPARENTED into the stack and nothing else, because
        # the pinned Qt tests reach `widget.plot.ax` and `widget.plot.figure`
        # and matplotlib artists, figures and savefig are untouched by
        # reparenting. `rate_plot` is a second PlotWidget rather than a second
        # use of the first: a Tg overlay and an Arrhenius fit share no axes,
        # and clearing one to draw the other loses the figure the user was
        # reading (SPEC_rate_series C2, MERGED_DECISIONS M5).
        self.plot_stack = QStackedWidget()
        self.plot = PlotWidget(figsize=(7.8, 5.6))
        self.ax2 = None
        self.rate_plot = PlotWidget(figsize=(7.2, 5.6))
        self.plot_stack.addWidget(self.plot)        # PLOT_PAGE_MAIN
        self.plot_stack.addWidget(self.rate_plot)   # PLOT_PAGE_RATE
        right_layout.addWidget(self.plot_stack)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        # Last: the guardrail pass needs every widget it enables to exist, and
        # with no data loaded it settles the workspace into the "nothing to
        # compute yet" state instead of offering live buttons over no arrays.
        self._refresh_segment_combo()
        self._update_scope_guardrails()
        self._refresh_detection_fields()
        self._refresh_program_tab()
        # `_record_lookup` is filled by __init__ BEFORE the UI is built, and a
        # workspace constructed with records never goes through `set_records`,
        # so the rate tab's check-list has to be filled here as well or it is
        # empty exactly for the user who passed records in.
        self._rate_refresh_library()

        # Connected LAST, after `plot_stack` exists. `addTab` emits
        # currentChanged for the first tab, and a handler that ran during
        # construction would reach for a stack that had not been built
        # (SPEC_rate_series §10 hazard i). The handler also guards on
        # `hasattr`, because two defences against a construction-order crash
        # cost one line and finding this one cost a test run.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._set_status(
            "Parallel (improved): each baseline is either a range (linear fit) or a point "
            "(a line parallel to the other baseline passing through that point)."
        )

    def _build_tg_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        data_box = QGroupBox("Data")
        data_form = QVBoxLayout(data_box)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        data_form.addWidget(self.file_label)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("Imported"))
        self.record_combo = QComboBox()
        self.record_combo.addItems(list(self._record_lookup.keys()))
        self.record_combo.currentIndexChanged.connect(self._load_selected_record)
        rec_row.addWidget(self.record_combo, 1)
        data_form.addLayout(rec_row)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Scope"))
        self.segment_combo = QComboBox()
        self.segment_combo.setToolTip(
            "Which part of the record every tab works on.\n\n"
            "A multi-step or cyclic run is split into segments (heat / cool / "
            "hold) and each one is analysed on its own. Tg, Calculs and the "
            "result CSV need a single segment: a tangent construction is "
            "defined on one monotone scan, and eight passes over the same "
            "temperatures are not one. The aggregate views are for looking, "
            "not for measuring."
        )
        self.segment_combo.setEnabled(False)
        self.segment_combo.currentIndexChanged.connect(self._on_segment_changed)
        seg_row.addWidget(self.segment_combo, 1)
        data_form.addLayout(seg_row)

        self.scope_note_label = QLabel("")
        self.scope_note_label.setObjectName("SectionNote")
        self.scope_note_label.setWordWrap(True)
        data_form.addWidget(self.scope_note_label)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._on_open)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._on_reload)
        btn_row.addWidget(open_btn)
        btn_row.addWidget(reload_btn)
        data_form.addLayout(btn_row)
        layout.addWidget(data_box)

        plot_box = QGroupBox("Plot")
        plot_form = QFormLayout(plot_box)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.dy_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._on_x_change)
        self.y_combo.currentTextChanged.connect(self._on_y_change)
        self.dy_combo.currentTextChanged.connect(self._refresh_plot)
        plot_form.addRow("X", self.x_combo)

        unit_row = QHBoxLayout()
        self.temp_c_btn = QPushButton("°C")
        self.temp_k_btn = QPushButton("K")
        switch_tip = (
            "Switch the temperature axis, Tg results, and window/baseline "
            "ranges between °C and K (applies workspace-wide, including the "
            "Calculs tab). Existing range values and any computed Tg are "
            "converted in place, not reinterpreted. Disabled when the "
            "current X column isn't a recognized temperature axis."
        )
        for btn, obj_name in ((self.temp_c_btn, "UnitSwitchLeft"), (self.temp_k_btn, "UnitSwitchRight")):
            btn.setCheckable(True)
            btn.setObjectName(obj_name)
            btn.setToolTip(switch_tip)
            btn.setEnabled(False)  # enabled once a recognized temperature column is loaded
        self.temp_c_btn.setChecked(True)
        self.temp_unit_group = QButtonGroup(self)
        self.temp_unit_group.setExclusive(True)
        self.temp_unit_group.addButton(self.temp_c_btn, 0)
        self.temp_unit_group.addButton(self.temp_k_btn, 1)
        self.temp_c_btn.toggled.connect(self._on_temp_unit_toggled)
        self.temp_k_btn.toggled.connect(self._on_temp_unit_toggled)
        unit_row.addWidget(self.temp_c_btn)
        unit_row.addWidget(self.temp_k_btn)
        unit_row.addStretch(1)
        plot_form.addRow("Temp. unit", unit_row)

        plot_form.addRow("Y", self.y_combo)
        plot_form.addRow("dY source", self.dy_combo)

        opt_row = QHBoxLayout()
        self.invert_y_check = QCheckBox("Invert Y")
        self.show_deriv_check = QCheckBox("dY overlay")
        self.explicit_check = QCheckBox("Explicit")
        self.explicit_check.setChecked(True)
        for cb in (self.invert_y_check, self.show_deriv_check, self.explicit_check):
            cb.stateChanged.connect(self._refresh_plot)
            opt_row.addWidget(cb)
        plot_form.addRow(opt_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth dY (MA10)"))
        self.smooth_group = QButtonGroup(self)
        smooth_off = QRadioButton("Off")
        smooth_on = QRadioButton("On")
        smooth_off.setChecked(True)
        self.smooth_group.addButton(smooth_off, 0)
        self.smooth_group.addButton(smooth_on, 1)
        smooth_off.toggled.connect(self._refresh_plot)
        smooth_row.addWidget(smooth_off)
        smooth_row.addWidget(smooth_on)
        plot_form.addRow(smooth_row)
        layout.addWidget(plot_box)

        tg_box = QGroupBox("Tg")
        tg_layout = QVBoxLayout(tg_box)

        method_row = QHBoxLayout()
        self.method_group = QButtonGroup(self)
        self.method_double = QRadioButton("Double")
        self.method_parallel = QRadioButton("Parallel (improved)")
        self.method_deriv = QRadioButton("|dY| max")
        self.method_parallel.setChecked(True)
        for i, btn in enumerate((self.method_double, self.method_parallel, self.method_deriv)):
            self.method_group.addButton(btn, i)
            btn.toggled.connect(self._refresh_plot)
            method_row.addWidget(btn)
        tg_layout.addLayout(method_row)

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Window"))
        self.xmin_edit = QLineEdit("350")
        self.xmax_edit = QLineEdit("700")
        for edit in (self.xmin_edit, self.xmax_edit):
            # textEdited, NOT textChanged: setText() (a record load, a °C/K
            # conversion, a segment-derived default) must not be mistaken for
            # the user choosing a window.
            edit.textEdited.connect(self._on_window_edited)
        window_row.addWidget(self.xmin_edit)
        window_row.addWidget(QLabel("→"))
        window_row.addWidget(self.xmax_edit)
        tg_layout.addLayout(window_row)

        ranges_box = QGroupBox("Manual ranges (shared)")
        ranges_layout = QVBoxLayout(ranges_box)

        self.low_min_edit, self.low_max_edit, self.low_point_edit, self.low_use_point_check = self._baseline_row(
            ranges_layout, "LOW"
        )
        slope_row = QHBoxLayout()
        slope_row.addWidget(QLabel("SLOPE:"))
        self.slope_min_edit = QLineEdit()
        self.slope_max_edit = QLineEdit()
        slope_row.addWidget(self.slope_min_edit)
        slope_row.addWidget(QLabel("→"))
        slope_row.addWidget(self.slope_max_edit)
        slope_row.addWidget(QLabel("(Double Tangent only)"))
        ranges_layout.addLayout(slope_row)

        self.high_min_edit, self.high_max_edit, self.high_point_edit, self.high_use_point_check = self._baseline_row(
            ranges_layout, "HIGH"
        )
        tg_layout.addWidget(ranges_box)

        action_row = QHBoxLayout()
        self.manual_compute_check = QCheckBox("Manual")
        action_row.addWidget(self.manual_compute_check)
        compute_btn = QPushButton("Compute")
        compute_btn.setObjectName("Primary")
        compute_btn.clicked.connect(self._compute)
        self.compute_btn = compute_btn
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_results)
        redraw_btn = QPushButton("Redraw")
        redraw_btn.clicked.connect(self._refresh_plot)
        action_row.addWidget(compute_btn)
        action_row.addWidget(clear_btn)
        action_row.addWidget(redraw_btn)
        tg_layout.addLayout(action_row)

        self.result_label = QLabel("Tg: —")
        self.result_label.setWordWrap(True)
        tg_layout.addWidget(self.result_label)
        layout.addWidget(tg_box)
        layout.addStretch(1)
        return tab

    def _baseline_row(self, parent_layout: QVBoxLayout, label: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))
        vmin = QLineEdit()
        vmax = QLineEdit()
        row.addWidget(vmin)
        row.addWidget(QLabel("→"))
        row.addWidget(vmax)
        use_point = QCheckBox("Use point")
        row.addWidget(use_point)
        row.addWidget(QLabel("Point:"))
        vpt = QLineEdit()
        row.addWidget(vpt)
        parent_layout.addLayout(row)

        def _toggle(checked: bool):
            vmin.setEnabled(not checked)
            vmax.setEnabled(not checked)
            vpt.setEnabled(checked)
            self._refresh_plot()

        use_point.toggled.connect(_toggle)
        _toggle(False)
        return vmin, vmax, vpt, use_point

    # ------------------------------------------------------------------
    # Program tab — construction
    # ------------------------------------------------------------------

    def _build_program_tab(self) -> QWidget:
        """What was asked of the furnace, what it did, and the gap between.

        Read-only except for the three detection knobs. Everything bad it has
        to report -- a `.mth` that belongs to another run, a step that never
        executed, a cooling ramp that stopped holding its programmed rate at
        448 °C -- is said in the warnings label and in the tables, and never in
        a modal dialog. A warning that must be clicked away is a warning that
        gets clicked away, and each of these is standing information about the
        record on screen rather than an event.

        Wrapped in a `QScrollArea` for the same reason `processing/
        qt_fit_params.py` is: the left panel is capped at 440 px and this tab
        carries two tables and four groups.
        """
        tab = QWidget()
        outer = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self.program_summary_label = QLabel("")
        self.program_summary_label.setObjectName("SectionNote")
        self.program_summary_label.setWordWrap(True)
        layout.addWidget(self.program_summary_label)

        self.program_warning_label = QLabel("")
        self.program_warning_label.setWordWrap(True)
        self.program_warning_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.program_warning_label)

        # --- programmed method -------------------------------------------
        method_box = QGroupBox("Programmed method")
        method_layout = QVBoxLayout(method_box)
        self.method_table = self._make_program_table(METHOD_TABLE_COLUMNS)
        self.method_table.cellClicked.connect(self._on_method_row_clicked)
        self.method_table.setToolTip(
            "Every step of the method, as the instrument recorded it next to "
            "the data. 'Matched' says which detected segment ran the step; "
            "click a row to select that segment."
        )
        method_layout.addWidget(self.method_table)

        mth_row = QHBoxLayout()
        self.load_mth_btn = QPushButton("Load .mth…")
        self.load_mth_btn.setToolTip(
            "Cross-check a saved TA method file against the method recorded in "
            "this data file. The data file's own record stays authoritative: a "
            ".mth can confirm a step or disagree with it, and never relabels "
            "anything."
        )
        self.load_mth_btn.clicked.connect(self._on_load_mth)
        self.clear_mth_btn = QPushButton("Clear")
        self.clear_mth_btn.setToolTip("Forget the loaded .mth.")
        self.clear_mth_btn.clicked.connect(self._on_clear_mth)
        mth_row.addWidget(self.load_mth_btn)
        mth_row.addWidget(self.clear_mth_btn)
        mth_row.addStretch(1)
        method_layout.addLayout(mth_row)

        self.mth_note_label = QLabel("")
        self.mth_note_label.setObjectName("SectionNote")
        self.mth_note_label.setWordWrap(True)
        method_layout.addWidget(self.mth_note_label)
        layout.addWidget(method_box)

        # --- detected segments --------------------------------------------
        seg_box = QGroupBox("Detected segments vs programme")
        seg_layout = QVBoxLayout(seg_box)
        self.program_table = self._make_program_table(
            tuple(PROGRAM_TABLE_HEADERS[k] for k in PROGRAM_TABLE_KEYS))
        for col, key in enumerate(PROGRAM_TABLE_KEYS):
            tip = PROGRAM_TABLE_HEADER_TIPS.get(key)
            header_item = self.program_table.horizontalHeaderItem(col)
            if tip and header_item is not None:
                header_item.setToolTip(tip)
        self.program_table.cellClicked.connect(self._on_program_row_clicked)
        self.program_table.setToolTip(
            "One row per detected segment, plus a row for every programmed "
            "step that ran without leaving its own signature ('absorbed') and "
            "every step the data never shows ('not executed'). Click a row to "
            "select that segment."
        )
        seg_layout.addWidget(self.program_table)

        self.program_selection_label = QLabel("")
        self.program_selection_label.setObjectName("SectionNote")
        self.program_selection_label.setWordWrap(True)
        seg_layout.addWidget(self.program_selection_label)

        self.program_table_note = QLabel(_PROGRAM_UNIT_NOTE)
        self.program_table_note.setObjectName("SectionNote")
        self.program_table_note.setWordWrap(True)
        seg_layout.addWidget(self.program_table_note)
        layout.addWidget(seg_box)

        # --- detection knobs -----------------------------------------------
        detect_box = QGroupBox("Detection")
        detect_layout = QVBoxLayout(detect_box)
        form = QFormLayout()
        self.seg_window_edit = QLineEdit()
        self.seg_window_edit.setToolTip(
            "Centred least-squares window the rate estimate is measured over, "
            "in MINUTES. Minutes and not points, because one export here is "
            "0.5 s per point and another 2.8 s: a point-based window would mean "
            "a different physical bandwidth on every file."
        )
        self.seg_hold_edit = QLineEdit()
        self.seg_hold_edit.setToolTip(
            "Shortest stretch that can count as a hold, in minutes."
        )
        self.seg_span_edit = QLineEdit()
        self.seg_span_edit.setToolTip(
            "How far the temperature may wander inside that stretch, in °C, "
            "for it to count as flat."
        )
        form.addRow("Rate window (min)", self.seg_window_edit)
        form.addRow("Min hold (min)", self.seg_hold_edit)
        form.addRow("Flat span (°C)", self.seg_span_edit)
        detect_layout.addLayout(form)

        knob_note = QLabel(
            "Min hold and Flat span are one test: a stretch is a hold when some "
            "window of at least Min hold minutes inside it varies by no more "
            "than Flat span.\n\n"
            "There is deliberately no |dT/dt| tolerance. It was measured and it "
            "cannot work: on a real cyclic run the pointwise rate of a true hold "
            "and of the tail of a cooling ramp overlap completely, and a "
            "2 °C/min tolerance invents two hold segments inside the cooling "
            "tails. The flat-core test above separates the same two cases by a "
            "factor of 19. Offering the knob would only invite tuning something "
            "with no working value."
        )
        knob_note.setObjectName("SectionNote")
        knob_note.setWordWrap(True)
        detect_layout.addWidget(knob_note)

        btn_row = QHBoxLayout()
        self.redetect_btn = QPushButton("Re-detect")
        self.redetect_btn.setObjectName("Primary")
        self.redetect_btn.setToolTip(
            "Segment every loaded record again with these settings."
        )
        self.redetect_btn.clicked.connect(self._on_redetect)
        self.seg_reset_btn = QPushButton("Reset")
        self.seg_reset_btn.setToolTip("Back to the measured defaults.")
        self.seg_reset_btn.clicked.connect(self._on_seg_params_reset)
        btn_row.addWidget(self.redetect_btn)
        btn_row.addWidget(self.seg_reset_btn)
        btn_row.addStretch(1)
        detect_layout.addLayout(btn_row)

        self.seg_params_note = QLabel("")
        self.seg_params_note.setObjectName("SectionNote")
        self.seg_params_note.setWordWrap(True)
        detect_layout.addWidget(self.seg_params_note)
        layout.addWidget(detect_box)

        layout.addStretch(1)
        return tab

    @staticmethod
    def _fit_program_columns(table: QTableWidget, max_width: int = 260) -> None:
        """Size columns to their contents, then cap the wide ones.

        Both tables carry one free-text column -- the verbatim step line, and
        the science layer's own note sentence -- that is several hundred
        characters long. Sized to its contents inside a 440 px panel it pushes
        every numeric column past the right edge, where the reader has to
        scroll to find out whether a rate was 20 or -20. The full text stays
        reachable: it is in the cell and in its tooltip, and the column can be
        dragged wider.
        """
        table.resizeColumnsToContents()
        for col in range(table.columnCount()):
            if table.columnWidth(col) > max_width:
                table.setColumnWidth(col, max_width)

    @staticmethod
    def _make_program_table(headers: Sequence[str]) -> QTableWidget:
        """A read-only, row-selectable table with `headers`.

        Read-only is a correctness rule, not a convenience: every cell here is
        either the instrument's own record or a measurement derived from it,
        and an editable copy of a measured number is a number nobody can trace.
        """
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSortingEnabled(False)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(160)
        return table

    def _build_calc_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        box = QGroupBox("Calculs")
        form = QFormLayout(box)

        self.calc_y_combo = QComboBox()
        form.addRow("Y", self.calc_y_combo)

        deriv_row = QHBoxLayout()
        self.calc_use_deriv_check = QCheckBox("Use derivative")
        self.calc_use_deriv_check.toggled.connect(self._calc_clear)
        deriv_row.addWidget(self.calc_use_deriv_check)
        deriv_row.addWidget(QLabel("d/dX:"))
        self.calc_deriv_x_combo = QComboBox()
        deriv_row.addWidget(self.calc_deriv_x_combo, 1)
        form.addRow(deriv_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Xmin"))
        self.calc_xmin_edit = QLineEdit()
        range_row.addWidget(self.calc_xmin_edit)
        range_row.addWidget(QLabel("Xmax"))
        self.calc_xmax_edit = QLineEdit()
        range_row.addWidget(self.calc_xmax_edit)
        form.addRow(range_row)

        btn_row = QHBoxLayout()
        integrate_btn = QPushButton("Integrate")
        integrate_btn.clicked.connect(self._calc_integrate)
        max_btn = QPushButton("Find Max")
        max_btn.clicked.connect(self._calc_find_max)
        min_btn = QPushButton("Find Min")
        min_btn.clicked.connect(self._calc_find_min)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._calc_clear)
        for b in (integrate_btn, max_btn, min_btn, clear_btn):
            btn_row.addWidget(b)
        form.addRow(btn_row)
        # Named so the scope guardrail can reach them. Clear is deliberately
        # NOT among them: clearing a stale artifact must work in every scope.
        self.integrate_btn = integrate_btn
        self.calc_max_btn = max_btn
        self.calc_min_btn = min_btn

        self.calc_result_label = QLabel("")
        self.calc_result_label.setWordWrap(True)
        form.addRow(self.calc_result_label)
        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _build_rate_tab(self) -> QWidget:
        """The "Rate series" tab: rows in, one multi-rate series out.

        SPEC_rate_series §6. This is the feature the user asked for by name —
        "add a feature to load data taken at different rates to make the
        arrhenius and lasocka plots" — and the four add buttons are the four
        answers to "where does a rate come from": the segments of the cyclic
        run on screen, separate files (one specimen per rate, the textbook
        design), records already in the library, and a typed row for a value
        from the literature or another tool.

        Scrolled, as `processing/qt_fit_params.py` is, because the row table
        wants every pixel it can get inside a 440 px panel and the groups below
        it must stay reachable on a laptop screen.

        Built with NO data loaded — `test_dta_workspace_constructs_standalone`
        pins that a bare `DtaWorkspace()` constructs — so nothing here reads
        `self.df` or `self.colnames`. The one column-dependent widget (the Tp
        signal combo) is filled by `_rate_refresh_columns`, which runs on every
        tab switch and at the start of every add path (§10 hazard ii).
        """
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # The banner is the FIRST widget and is never hidden, so the layout
        # cannot jump as warnings come and go and so a warning cannot be below
        # the fold. It is a label, not a dialog: this whole feature has no modal
        # warning anywhere (MERGED_DECISIONS M0.1 — a dialog is the surface a
        # user learns to dismiss without reading).
        self.rate_warn_banner = QLabel("")
        self.rate_warn_banner.setObjectName("WarningBannerOk")
        self.rate_warn_banner.setWordWrap(True)
        layout.addWidget(self.rate_warn_banner)

        rows_box = QGroupBox("Series rows")
        rows_layout = QVBoxLayout(rows_box)
        self.rate_table = self._make_rate_table()
        rows_layout.addWidget(self.rate_table, 1)

        add_row = QHBoxLayout()
        self.rate_add_seg_btn = QPushButton("Add segments from current file")
        self.rate_add_seg_btn.setToolTip(
            "One row per HEATING ramp of the record on screen, each with its "
            "measured and programmed rate already filled in. Cooling ramps are "
            "not added by default — their kinetics are a different experiment, "
            "and a row added from one arrives flagged as cooling.")
        self.rate_add_seg_btn.clicked.connect(self._on_rate_add_segments)
        self.rate_add_files_btn = QPushButton("Add files…")
        self.rate_add_files_btn.setToolTip(
            "Pick several files measured at different rates. Each file's rate "
            "is read from its own programmed method and its own data — nothing "
            "to type.")
        self.rate_add_files_btn.clicked.connect(self._on_rate_add_files)
        add_row.addWidget(self.rate_add_seg_btn)
        add_row.addWidget(self.rate_add_files_btn)
        rows_layout.addLayout(add_row)

        lib_row = QHBoxLayout()
        self.rate_library_combo = CheckComboBox("imported")
        self.rate_add_lib_btn = QPushButton("Add checked")
        self.rate_add_lib_btn.clicked.connect(self._on_rate_add_library)
        lib_row.addWidget(self.rate_library_combo, 1)
        lib_row.addWidget(self.rate_add_lib_btn)
        rows_layout.addLayout(lib_row)

        edit_row = QHBoxLayout()
        self.rate_add_blank_btn = QPushButton("Add blank row")
        self.rate_add_blank_btn.setToolTip(
            "An empty row to type a rate and a value into — a literature "
            "value, or a specimen measured on another instrument.")
        self.rate_add_blank_btn.clicked.connect(self._on_rate_add_blank)
        self.rate_remove_btn = QPushButton("Remove selected")
        self.rate_remove_btn.clicked.connect(self._rate_remove_selected)
        self.rate_clear_btn = QPushButton("Clear all")
        self.rate_clear_btn.clicked.connect(self._rate_clear)
        for btn in (self.rate_add_blank_btn, self.rate_remove_btn, self.rate_clear_btn):
            edit_row.addWidget(btn)
        rows_layout.addLayout(edit_row)

        self.rate_rows_note = QLabel("")
        self.rate_rows_note.setObjectName("SectionNote")
        self.rate_rows_note.setWordWrap(True)
        rows_layout.addWidget(self.rate_rows_note)
        layout.addWidget(rows_box, 1)

        quantity_box = QGroupBox("Quantity")
        quantity_form = QFormLayout(quantity_box)
        self.rate_quantity_combo = QComboBox()
        for label, kind, method in RATE_QUANTITY_CHOICES:
            self.rate_quantity_combo.addItem(label, (kind, method))
        self.rate_quantity_combo.currentIndexChanged.connect(self._on_rate_quantity_changed)
        quantity_form.addRow("Quantity", self.rate_quantity_combo)

        self.rate_tp_signal_combo = QComboBox()
        self.rate_tp_signal_combo.setToolTip(
            "The signal a peak temperature is read off. Ignored for Tg, which "
            "uses the Y column of the Tg tab so a rate-series value and a Tg "
            "tab compute on the same segment are the same number.")
        quantity_form.addRow("Tp signal", self.rate_tp_signal_combo)

        tp_row = QHBoxLayout()
        self.rate_tp_min_edit = QLineEdit("350")
        self.rate_tp_max_edit = QLineEdit("700")
        tp_row.addWidget(self.rate_tp_min_edit)
        tp_row.addWidget(QLabel("→"))
        tp_row.addWidget(self.rate_tp_max_edit)
        # Labelled °C and stored °C, whatever the display switch shows. The
        # alternative -- a field that follows the °C/K toggle -- is R4: a second
        # Kelvin conversion site, and a window converted twice still looks like
        # a plausible temperature while making Ea wrong by about a factor of 3.
        quantity_form.addRow("Tp window (°C)", tp_row)

        self.rate_settings_note = QLabel("")
        self.rate_settings_note.setObjectName("SectionNote")
        self.rate_settings_note.setWordWrap(True)
        quantity_form.addRow(self.rate_settings_note)

        compute_row = QHBoxLayout()
        self.rate_compute_btn = QPushButton("Compute all values")
        self.rate_compute_btn.setObjectName("Primary")
        self.rate_compute_btn.setToolTip(
            "Measure every row with the settings above — ONE window, ONE method, "
            "one set of baselines for all of them, because the comparison between "
            "rates IS the measurement. A row that cannot be measured keeps its "
            "place in the table and says why. A value you typed yourself is never "
            "overwritten.")
        self.rate_compute_btn.clicked.connect(self._on_rate_compute)
        self.rate_recompute_failed_btn = QPushButton("Retry failed")
        self.rate_recompute_failed_btn.setToolTip(
            "Re-measure only the rows that failed. Use it after widening the "
            "window or fixing a baseline: rows that already carry a number are "
            "left exactly as they were, so a series that is three-quarters "
            "measured is never re-measured wholesale under settings chosen for "
            "its last quarter.")
        self.rate_recompute_failed_btn.clicked.connect(self._on_rate_retry_failed)
        compute_row.addWidget(self.rate_compute_btn)
        compute_row.addWidget(self.rate_recompute_failed_btn)
        quantity_form.addRow(compute_row)
        layout.addWidget(quantity_box)

        fit_box = QGroupBox("Fit")
        fit_layout = QVBoxLayout(fit_box)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.rate_model_combo = QComboBox()
        # Built FROM the registry, in registry order, label and key together.
        # A hand-written list here would be a second opinion about which models
        # exist, and `needs` gating below is a second opinion about which suit
        # which quantity -- both of which are `dta_kinetics`' to hold.
        for _key, _meta in KINETIC_METHODS.items():
            self.rate_model_combo.addItem(str(_meta.get("label") or _key), _key)
        self._rate_select_default_model()
        self.rate_model_combo.currentIndexChanged.connect(self._on_rate_model_changed)
        model_row.addWidget(self.rate_model_combo, 1)
        fit_layout.addLayout(model_row)

        opts_row = QHBoxLayout()
        self.rate_show_labels_check = QCheckBox("Point labels")
        self.rate_show_labels_check.setChecked(True)
        self.rate_show_labels_check.setToolTip(
            "Name each point on the figure. Off for a crowded series; the legend "
            "and the table still identify every row.")
        self.rate_show_labels_check.toggled.connect(self._on_rate_view_changed)
        self.rate_show_excluded_check = QCheckBox("Show excluded")
        self.rate_show_excluded_check.setChecked(True)
        self.rate_show_excluded_check.setToolTip(
            "Excluded rows are drawn greyed with an ×. Unticking hides them from "
            "the FIGURE only — they stay in the table and in the export, and the "
            "banner then says the figure is not showing everything.")
        self.rate_show_excluded_check.toggled.connect(self._on_rate_view_changed)
        opts_row.addWidget(self.rate_show_labels_check)
        opts_row.addWidget(self.rate_show_excluded_check)
        opts_row.addWidget(QLabel("q ref"))
        self.rate_qref_edit = QLineEdit(f"{DEFAULT_Q_REF:g}")
        self.rate_qref_edit.setToolTip(
            "The rate the fit is asked to back-predict the quantity at, in "
            "°C/min. It changes no fitted parameter — only the predicted value "
            "reported beside them.")
        self.rate_qref_edit.setMaximumWidth(60)
        self.rate_qref_edit.editingFinished.connect(self._on_rate_view_changed)
        opts_row.addWidget(self.rate_qref_edit)
        fit_layout.addLayout(opts_row)

        self.rate_fit_label = QLabel("")
        self.rate_fit_label.setWordWrap(True)
        self.rate_fit_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fit_layout.addWidget(self.rate_fit_label)
        layout.addWidget(fit_box)

        # Export / reload. Every button here is a thin dialog wrapper over a
        # method that takes a path, so a test never has to monkeypatch
        # QFileDialog to exercise the writing itself.
        io_box = QGroupBox("Export / reload series")
        io_layout = QVBoxLayout(io_box)
        self.rate_export_rows_btn = QPushButton("Export rows CSV…")
        self.rate_export_rows_btn.setObjectName("Primary")
        self.rate_export_rows_btn.setToolTip(
            "One line per row, in table order, with every warning code that "
            "applies to it AND every series-wide one — so a single-row extract "
            "is never innocent-looking.")
        self.rate_export_rows_btn.clicked.connect(self._on_rate_export_rows)
        self.rate_export_fit_btn = QPushButton("Export fit CSV…")
        self.rate_export_fit_btn.setToolTip(
            "One line per model that the present rows can evaluate — Lasocka, "
            "Moynihan, Kissinger side by side, each with its own warnings.")
        self.rate_export_fit_btn.clicked.connect(self._on_rate_export_fit)
        self.rate_save_png_btn = QPushButton("Save plot PNG…")
        self.rate_save_png_btn.setToolTip(
            "The annotation is drawn into the axes, so the saved figure carries "
            "its own caveats.")
        self.rate_save_png_btn.clicked.connect(self._on_rate_save_png)
        self.rate_load_csv_btn = QPushButton("Load rows CSV…")
        self.rate_load_csv_btn.setToolTip(
            "Reload a series exported earlier. Stored values are kept as saved — "
            "a measured Tg is a measurement, not a cache — so press "
            "\"Compute all values\" to re-measure them with the current settings.")
        self.rate_load_csv_btn.clicked.connect(self._on_rate_load_csv)
        for btn in (self.rate_export_rows_btn, self.rate_export_fit_btn,
                    self.rate_save_png_btn, self.rate_load_csv_btn):
            io_layout.addWidget(btn)
        self.rate_io_note = QLabel("")
        self.rate_io_note.setObjectName("SectionNote")
        self.rate_io_note.setWordWrap(True)
        io_layout.addWidget(self.rate_io_note)
        layout.addWidget(io_box)

        layout.addStretch(1)
        self._rate_update_quantity_guardrails()
        self._rate_refresh_table()
        self._rate_refit()
        self._rate_refresh_notes()
        return tab

    def _make_rate_table(self) -> QTableWidget:
        """The row table: a pure VIEW of `rate_series.rows`, keyed by uid.

        Three rules, each of them a bug that was designed out rather than
        fixed:

          * `setSortingEnabled(False)`. A sorted table renumbers every row
            while the user is reading a warning that says "row 3", and Qt's
            sort moves items between rows — so an `itemChanged` arriving after
            a sort carries a row index that means nothing.
          * the uid lives in `Qt.UserRole` of column 0 and EVERY mutation is
            resolved through it (`_rate_uid_at_row` -> `_rate_row_by_uid`),
            never through `currentRow()`.
          * the vertical header stays VISIBLE, because it numbers the rows from
            1 and that is exactly what a warning means by "row 3".
        """
        table = QTableWidget(0, len(RATE_TABLE_COLUMNS))
        table.setHorizontalHeaderLabels([h for _k, h in RATE_TABLE_COLUMNS])
        table.setSortingEnabled(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.DoubleClicked
                              | QAbstractItemView.EditKeyPressed)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(180)
        table.itemChanged.connect(self._rate_on_item_changed)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_rate_table_menu)
        return table

    def _build_batch_tab(self) -> QWidget:
        """Three groups: this scan, every scan of this file, many files.

        They are separate boxes because they answer three different questions
        and a cyclic record makes the difference matter. "Export result CSV"
        writes the ONE segment on screen and is therefore switched off in an
        aggregate scope (G1); "Export all segments" writes every segment of the
        loaded record and must stay ON in exactly that scope, because the whole
        run is where a user of a twelve-segment file sits. Putting both under
        one heading would make the second look like it had been disabled too.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        export_box = QGroupBox("Export current")
        export_layout = QVBoxLayout(export_box)
        export_csv_btn = QPushButton("Export result CSV…")
        export_csv_btn.setObjectName("Primary")
        export_csv_btn.clicked.connect(self._export_current_csv)
        self.export_csv_btn = export_csv_btn
        save_png_btn = QPushButton("Save figure PNG…")
        save_png_btn.clicked.connect(self._save_figure_png)
        export_layout.addWidget(export_csv_btn)
        export_layout.addWidget(save_png_btn)
        layout.addWidget(export_box)

        seg_box = QGroupBox("Export every segment of this file")
        seg_layout = QVBoxLayout(seg_box)
        self.skip_holds_check = QCheckBox("Skip hold segments")
        self.skip_holds_check.setChecked(True)
        self.skip_holds_check.setToolTip(
            "A Tg is a step in a temperature scan and a hold has no "
            "temperature scan — its measured temperatures repeat. Unchecking "
            "this exports the holds too, measured against the clock, with the "
            "substitution written into each row's warnings cell.")
        # The checkbox changes what BOTH per-segment sweeps will do, and both
        # notes say how many rows that is, so the text has to follow the box.
        self.skip_holds_check.toggled.connect(lambda _c: self._refresh_batch_notes())
        seg_layout.addWidget(self.skip_holds_check)
        export_segments_btn = QPushButton("Export all segments…")
        export_segments_btn.clicked.connect(self._export_all_segments_csv)
        self.export_segments_btn = export_segments_btn
        seg_layout.addWidget(export_segments_btn)
        self.export_segments_note = QLabel("")
        self.export_segments_note.setObjectName("SectionNote")
        self.export_segments_note.setWordWrap(True)
        seg_layout.addWidget(self.export_segments_note)
        layout.addWidget(seg_box)

        batch_box = QGroupBox("Batch over files")
        batch_layout = QVBoxLayout(batch_box)
        note = QLabel("Batch uses CURRENT settings (window, manual ranges, smoothing).")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        batch_layout.addWidget(note)

        # One row per file is what this batch has always produced, so it stays
        # the default: a folder of single-ramp exports must behave exactly as
        # it did before segments existed.
        self.batch_scope_group = QButtonGroup(self)
        self.batch_scope_buttons: Dict[str, QRadioButton] = {}
        for key in (BATCH_SCOPE_FILE, BATCH_SCOPE_EVERY, BATCH_SCOPE_MATCHING):
            btn = QRadioButton(BATCH_SCOPE_LABELS[key])
            self.batch_scope_group.addButton(btn)
            self.batch_scope_buttons[key] = btn
            batch_layout.addWidget(btn)
        self.batch_scope_buttons[BATCH_SCOPE_FILE].setChecked(True)
        self.batch_scope_group.buttonToggled.connect(self._on_batch_scope_changed)

        self.batch_scope_note = QLabel("")
        self.batch_scope_note.setObjectName("SectionNote")
        self.batch_scope_note.setWordWrap(True)
        batch_layout.addWidget(self.batch_scope_note)

        snap_row = QHBoxLayout()
        self.batch_snapshot_group = QButtonGroup(self)
        self.batch_snapshot_no = QRadioButton("No snapshots")
        self.batch_snapshot_png = QRadioButton("PNG snapshots")
        self.batch_snapshot_no.setChecked(True)
        self.batch_snapshot_group.addButton(self.batch_snapshot_no, 0)
        self.batch_snapshot_group.addButton(self.batch_snapshot_png, 1)
        snap_row.addWidget(self.batch_snapshot_no)
        snap_row.addWidget(self.batch_snapshot_png)
        batch_layout.addLayout(snap_row)

        run_btn = QPushButton("Run batch…")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self._batch_run)
        batch_layout.addWidget(run_btn)
        layout.addWidget(batch_box)
        layout.addStretch(1)
        return tab

    # ------------------------------------------------------------------
    # Status / data loading
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _load_selected_record(self) -> None:
        title = self.record_combo.currentText().strip()
        if not title or title not in self._record_lookup:
            return
        rec = self._record_lookup[title]
        df = rec.get("df")
        meta = rec.get("meta") or {}
        path = rec.get("path")

        if df is None and path:
            try:
                hdr, cols, df = parse_ta_sdt_txt(Path(path))
                # header_lines travels with raw_header everywhere, because the
                # header itself loses its repeats the moment it is copied into
                # a plain dict or written to a project file.
                meta = {**meta, "raw_header": hdr, "signals": cols,
                        "header_lines": _header_lines_of(hdr)}
            except Exception as exc:
                QMessageBox.critical(self, "Load error", f"Could not reload {path}:\n{exc}")
                return
        if df is None:
            QMessageBox.critical(self, "Load error", "Selected record has no data attached.")
            return

        display_title = rec.get("title") or (Path(path).stem if path else "DTA data")
        self._apply_dataframe(df, meta=meta, path=path, display_title=display_title,
                              record_id=self._record_key(rec, title))

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open DTA file", "", "Text files (*.txt *.dat *.asc);;All files (*.*)")
        if path:
            self.load_file(Path(path))

    def _on_reload(self) -> None:
        if self.path:
            self.load_file(self.path)

    def load_file(self, path: Path) -> None:
        self.path = Path(path)
        self.header, self.colnames, self.df = parse_ta_sdt_txt(self.path)
        self._apply_dataframe(
            self.df,
            meta={"raw_header": self.header,
                  "header_lines": _header_lines_of(self.header)},
            path=str(self.path), display_title=self.path.name,
        )

    def _suggest_default_columns(self, meta: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        canonical = (meta or {}).get("canonical_map") or {}

        def pick_from_canon(keys):
            for key in keys:
                col = canonical.get(key)
                if col in self.colnames:
                    return col
            return None

        def pick_by_keyword(keywords):
            for kw in keywords:
                for c in self.colnames:
                    if kw.lower() in str(c).lower():
                        return c
            return None

        xdef = pick_from_canon(["T_C", "time_min", "X"]) or pick_by_keyword(["temperature", "temp"]) or (self.colnames[0] if self.colnames else "")
        ydef = pick_from_canon(["DSC_mW_mg", "HF_mW", "TG_pct", "mass_mg", "Y"]) or pick_by_keyword(
            ["heat flow", "dsc", "dta", "heat", "signal", "tg"]
        ) or (self.colnames[1] if len(self.colnames) > 1 else (self.colnames[0] if self.colnames else ""))
        return xdef or "", ydef or ""

    def _apply_dataframe(self, df: pd.DataFrame, *, meta: Optional[Dict[str, Any]],
                         path: Optional[str], display_title: str,
                         record_id: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        raw_header = (meta or {}).get("raw_header", {}) or {}
        # A Mapping or nothing. `io_universal.parse_saxs_edf_ascii` stores
        # `raw_header` as a single joined STRING, and every `self.header.get(...)`
        # below would then raise AttributeError out of record selection.
        # `qt_shell._dta_records_from_library` filters to DTA kinds today, so
        # no such record can reach here — this is the isinstance that keeps that
        # true if the filter ever widens.
        self.header = raw_header if isinstance(raw_header, Mapping) else {}
        self.header_lines = _normalize_header_lines((meta or {}).get("header_lines"), self.header)
        self._canonical_map = (meta or {}).get("canonical_map") or {}
        self._record_id = record_id or f"{_RECORD_TITLE_PREFIX}{display_title}"
        self._window_user_edited = False
        self._x_forced_time = False
        # The previous record's segmentation has to go BEFORE the new frame
        # lands. Repopulating the column combos below emits currentTextChanged,
        # which redraws, and a redraw that slices this record with the last
        # record's segment boundaries draws a wrong curve -- or, when the new
        # file is shorter, raises. Detection re-runs at the end of this method.
        self.run = None
        self.program = None
        self.seg_results = {}
        self.colnames = list(df.columns)
        self.df = df.copy()

        for combo in (self.x_combo, self.y_combo, self.calc_y_combo, self.calc_deriv_x_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.colnames)
            combo.blockSignals(False)
        self.dy_combo.blockSignals(True)
        self.dy_combo.clear()
        self.dy_combo.addItems(["(same)"] + self.colnames)
        self.dy_combo.setCurrentText("(same)")
        self.dy_combo.blockSignals(False)

        xdef, ydef = self._suggest_default_columns(meta)
        self.x_combo.setCurrentText(xdef)
        self.y_combo.setCurrentText(ydef)
        self.calc_y_combo.setCurrentText(ydef)
        self.calc_deriv_x_combo.setCurrentText(xdef)

        self.xmin_edit.setText(self._format_temp_default(350.0))
        self.xmax_edit.setText(self._format_temp_default(700.0))
        self.calc_xmin_edit.setText(self._format_temp_default(350.0))
        self.calc_xmax_edit.setText(self._format_temp_default(700.0))

        sample = self.header.get("Sample", self.path.stem if self.path else display_title)
        suffix = self.path.name if self.path else display_title
        self.file_label.setText(f"{suffix}\nSample: {sample}")

        # Last, because every array the workspace hands out is scoped by it and
        # _clear_results() below already triggers a redraw.
        self._detect_run()
        self._scope_key = self._remembered_scope_key()
        self._refresh_segment_combo()
        self._update_scope_guardrails()
        self._apply_default_window()
        self._refresh_detection_fields()
        self._refresh_program_tab()
        # The rate tab describes the record in hand — how many heating ramps it
        # holds and at what rates, and which columns a Tp can be read off — so
        # it has to follow a record switch. Its ROWS deliberately do not: a rate
        # series is assembled ACROSS records, and dropping it when the user
        # opens the next file would make the whole feature unusable.
        self._rate_refresh_columns()
        self._rate_refresh_notes()
        self._clear_results()
        self._refresh_plot()

    def set_records(self, records: List[Dict[str, Any]]) -> None:
        """Refresh the "Imported" dropdown from the library's current DTA-kind
        spectra — called by the shell whenever the DTA workspace becomes
        active, so newly imported files show up without rebuilding the whole
        widget."""
        self.records = records or []
        self._record_lookup = {}
        for i, rec in enumerate(self.records):
            key = rec.get("title") or f"Record {i + 1}"
            base, suffix = key, 2
            while key in self._record_lookup:
                key = f"{base} ({suffix})"
                suffix += 1
            self._record_lookup[key] = rec

        current = self.record_combo.currentText()
        self.record_combo.blockSignals(True)
        self.record_combo.clear()
        self.record_combo.addItems(list(self._record_lookup.keys()))
        self.record_combo.blockSignals(False)
        # The Rate series tab's "imported" check-list is the same list under
        # another name. Refreshed here as well as on a tab switch, because the
        # shell calls this whenever the DTA page becomes active -- including
        # when the Rate series tab is already the one showing, where no tab
        # change will ever fire and the list would sit stale.
        self._rate_refresh_library()
        if current in self._record_lookup:
            self.record_combo.setCurrentText(current)
        elif self._record_lookup and self.df is None:
            self.record_combo.setCurrentIndex(0)
            self._load_selected_record()

    def _on_y_change(self) -> None:
        if not self.calc_y_combo.currentText():
            self.calc_y_combo.setCurrentText(self.y_combo.currentText())
        self._refresh_plot()

    def _on_window_edited(self, _text: str = "") -> None:
        """Remember that the Tg window is the user's, not a default."""
        self._window_user_edited = True

    # ------------------------------------------------------------------
    # Segmentation and scope
    # ------------------------------------------------------------------

    def _record_key(self, rec: Optional[Dict[str, Any]], display_title: str) -> str:
        """Identity of a record, for the segmentation cache and scope memory.

        Prefers the library's stable `Spectrum.id`. Records built by hand -- the
        tests do this, and so does `load_file` for a bare path -- carry no
        `"id"`, and those fall back to the display title.

        A title is a WEAKER identity than an id: importing the same file twice
        produces two spectra with the same title, which `set_records` only tells
        apart with a cosmetic " (2)" suffix that can be re-assigned in a
        different order on the next refresh. So a title-keyed record is treated
        as NOT CACHEABLE (`_key_is_stable`): its key is still used to look a run
        up, it is simply never written, and the detection re-runs. That costs
        milliseconds on a 2221-row file and it removes the one failure mode that
        would be invisible in testing and wrong in the field -- handing one
        file's segments to another file's data.
        """
        rid = ""
        if isinstance(rec, dict):
            rid = str(rec.get("id") or "").strip()
        return f"{_RECORD_ID_PREFIX}{rid}" if rid else f"{_RECORD_TITLE_PREFIX}{display_title}"

    @staticmethod
    def _key_is_stable(key: Optional[str]) -> bool:
        """True only for an id-backed key, the only kind safe to cache under."""
        return bool(key) and str(key).startswith(_RECORD_ID_PREFIX)

    def _run_for_dataframe(self, df: pd.DataFrame, *,
                           program: Optional[MethodProgram] = None,
                           canonical: Optional[Dict[str, Any]] = None
                           ) -> Optional[SegmentedRun]:
        """Segment one DataFrame, or return None. Never raises.

        Both a clock and a furnace-temperature column are required: a rate is
        °C per MINUTE, and substituting the row number for the clock would
        silently change what every window in `SegmentationConfig` means. A
        record without them is not segmented, and the whole-record fallback in
        `_prepare_xy` keeps it working exactly as it did before this rework.

        A Kelvin temperature column is converted to °C first, because segment
        bounds are stored canonically in °C and only converted for display.

        The CLOCK is converted to minutes from the unit in its own column label
        for the same reason: every rate here is per minute, every window in
        `SegmentationConfig` is in minutes, and a `Time (s)` export read as
        minutes reported 0.333 °C/min for a 20 °C/min ramp. The unit is written
        on the column, so the check costs nothing and not making it put a 60x
        error into the rate with no warning anywhere. The conversion is
        monotone, so it changes no ordering, no `keep_index` and no segment
        boundary row — only the numbers that are in minutes.
        """
        t_col, T_col = _time_temp_columns(list(df.columns), canonical)
        if not t_col or not T_col:
            return None
        try:
            t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
            t = t * clock_scale_to_min(t_col)[0]
            T = pd.to_numeric(df[T_col], errors="coerce").to_numpy(dtype=float)
            T = _shift_temperature(T, _unit_from_col(T_col), "°C")
            return segment_run(t, T, program=(program if program else None),
                               config=self._seg_config)
        except Exception:
            # segment_run does not raise on data; this catches a column that
            # is not numeric at all. A file we cannot segment must still open.
            return None

    def _detect_run(self) -> None:
        """Read the programmed method and segment the loaded record. Never raises.

        A no-op when no data is loaded, which is what keeps a bare
        `DtaWorkspace()` constructible with an empty combo.
        """
        self.run = None
        self.program = None
        self.seg_results = {}
        if self.df is None:
            self._recompare_mth()
            return

        try:
            self.program = program_from_header(
                self.header, self.header_lines,
                name=(self.path.name if self.path else ""),
                source_path=(str(self.path) if self.path else None),
            )
        except Exception:
            self.program = None
        # A .mth loaded against the PREVIOUS record says nothing about this
        # one. It is kept (silently dropping it would be its own surprise) and
        # re-judged here, so a file that no longer matches announces itself
        # instead of leaving a stale "identical" on screen.
        self._recompare_mth()

        key = self._record_id
        cached = self._run_cache.get(key) if key else None
        if cached is not None and int(cached.n_rows) == int(len(self.df)):
            self.run = cached
            return

        run = self._run_for_dataframe(self.df, program=self.program,
                                      canonical=self._canonical_map)
        self.run = run
        if run is not None and self._key_is_stable(key):
            self._run_cache[str(key)] = run

    def _scope(self) -> DtaScope:
        """What the workspace is currently pointed at. See `scope_from_key`.

        Resolved on every call rather than cached: the run underneath can be
        replaced (a new record, a re-detection) while a key from the old one is
        still stored, and a stale cached scope would carry another file's
        segment object into `prepare_segment_xy`.
        """
        return scope_from_key(self._scope_key, self.run)

    def _current_segment(self) -> Optional[Segment]:
        """The single segment in scope, or None on an aggregate view."""
        return self._scope().segment

    def _rule_segment(self, scope: DtaScope, run: Optional[SegmentedRun], *,
                      n_rows: int, x_col: str, x_kind: str
                      ) -> Tuple[Segment, Optional[np.ndarray]]:
        """`(segment, keep_index)` for `prepare_segment_xy`, or raise.

        Three jobs, each of which is a rule rather than plumbing:

        * **G2, the structural guardrail.** An aggregate scope has no single
          curve to prepare, so it raises `SegmentScopeError` naming the
          segments. The UI disables the Tg buttons outside a single segment,
          but a UI check can be forgotten by the next call site, so the refusal
          lives here as well.
        * **The unsegmented fallback.** With no run, the whole record stands in
          as one segment so that `prepare_segment_xy` remains the only copy of
          the array rule.
        * **The hold rule is about temperature, not about X.** A hold refuses
          `x_kind='T'` because its measured temperatures are duplicated and
          non-monotone; that is a fact about the FURNACE TEMPERATURE. When the
          user has put some other column on X, the refusal does not apply, and
          the segment's kind is coerced for the purposes of the rule only. The
          protection does not go away: the unconditional duplicate collapse and
          the 1 % flood guard inside `prepare_segment_xy` still bite.
        """
        if scope.segments and not scope.is_single:
            names = ", ".join(f"#{s.index} {s.kind}" for s in scope.segments)
            raise SegmentScopeError(
                f"the current scope covers {len(scope.segments)} segments ({names}); "
                "one sorted curve cannot describe them. Select a single segment.")

        seg = scope.segment
        if seg is None:
            return _whole_record_segment(n_rows), None

        keep = None if run is None else run.keep_index
        if x_kind == X_TEMPERATURE and seg.is_hold and not _is_temperature_col(x_col):
            seg = replace(seg, kind=KIND_HEAT, role=ROLE_HEAT)
        return seg, keep

    def _prepare_xy(self, x: np.ndarray, y: np.ndarray, *, x_col: str,
                    y_alt: Optional[np.ndarray] = None, invert: bool = False,
                    dedupe: bool = True,
                    run: Any = _CURRENT_RECORD,
                    scope: Optional[DtaScope] = None,
                    keep_index: Optional[np.ndarray] = None) -> PreparedXY:
        """Thin DataFrame->ndarray adapter onto `dta_segments.prepare_segment_xy`.

        This is the ONLY place in the Qt layer that turns columns into
        analysis-ready arrays. It replaced three independent copies of

            mask = isfinite(x) & isfinite(y); order = np.argsort(x)

        whose global sort is meaningless the moment a record contains more than
        one ramp: on the real cyclic file it interleaves eight passes over
        19-1299 °C and collapses all three Tg methods onto one number.

        `run` / `scope` default to the loaded record's; the batch loop passes
        another file's, and passes an explicit `run=None` for a file that could
        not be segmented -- which is why the default is a sentinel and not
        `None`. A `scope` on its own (the loaded run, one named segment) is how
        `_draw_multi_segment` prepares each curve of an overlay: the scope says
        WHICH segment, the run supplies the `keep_index` that turns its cleaned
        bounds into DataFrame rows, and the two must come from the same record.
        `x` must be ORIGINAL-length (one value per DataFrame row) and
        already in display units -- the °C/K conversion is monotone, so it
        changes no ordering and is applied before this, exactly as before.

        The abscissa kind is read off `x_col`, never off the scope: this method
        is handed an array, and the only honest way to know what that array is
        is to ask which column produced it. A scope that forces the time axis
        does so by forcing the X combo, so the array arrives already correct.

        `keep_index` is for a caller that has no `SegmentedRun` but DOES know
        which original rows its segment's cleaned bounds refer to — the rate
        series, which re-derives the mapping from the row's own frame. It is
        used only when the run did not supply one; the two can never disagree
        because a run that exists always wins.

        **`max_duplicate_frac=None`: a duplicate flood is reported here, never
        refused.** `prepare_segment_xy`'s 1 % guardrail is for a call site that
        forgot to check the scope, and in this layer that check has two owners
        already — `_rule_segment` refuses an aggregate scope outright (G2), and
        `_rate_segment_rows` re-derives a rate row's rows instead of trusting a
        bounding interval. What the guardrail actually caught in the field was a
        coarsely rounded temperature column: a NOISELESS 1 °C/min ramp written
        to two decimals is 16.7 % duplicate and a 0.5 °C/min one is 58.3 %, and
        refusing those meant the file could not yield a Tg at all, behind a
        modal dialog whose text blamed "the wrong abscissa or the wrong scope"
        when both were right. That is MERGED_DECISIONS M0.1's forbidden case.
        The duplicates are AVERAGED and the share is carried on
        `PreparedXY.duplicate_fraction` into `_result_warnings` and into
        `RateRow.extras`, where `coarse_abscissa` reports it.
        """
        if run is _CURRENT_RECORD:
            run = self.run
            if scope is None:
                scope = self._scope()
        if scope is None:
            scope = _scope_for_run(run)

        x_arr = np.asarray(x, dtype=float).ravel()
        x_kind = X_TIME if _looks_like_time(x_col) else X_TEMPERATURE
        seg, keep = self._rule_segment(scope, run, n_rows=int(x_arr.size),
                                       x_col=x_col, x_kind=x_kind)
        if keep is None and keep_index is not None:
            keep = np.asarray(keep_index, dtype=np.intp)
        # prepare_segment_xy reads exactly ONE of its two abscissa arguments,
        # the one `x_kind` names. The user's column is handed to both slots so
        # the unused one can never be read by accident and the length checks
        # still apply to the array that matters.
        prepared = prepare_segment_xy(x_arr, x_arr, y, segment=seg, keep_index=keep,
                                      y_alt=y_alt, x_kind=x_kind, dedupe=dedupe,
                                      invert=invert, max_duplicate_frac=None)
        self._last_prepared = prepared
        return prepared

    # ------------------------------------------------------------------
    # Scope selector, memory and guardrails
    # ------------------------------------------------------------------

    def _segment_for_display(self, seg: Segment) -> Segment:
        """A copy of `seg` with its temperatures in the CURRENT display unit.

        Segment bounds are stored canonically in °C and converted only for
        display (the same rule the axis and the window fields follow). Passing
        a stored segment to `segment_label(unit="K")` would print °C numbers
        under a K suffix, which is the exact class of error the canonical
        storage exists to prevent.

        Rates are NOT converted: °C/min and K/min are the same number, because
        the +273.15 offset cancels in a difference.

        NEVER give the result to anything that slices arrays -- it carries
        display numbers, and `row_start`/`row_stop` are only meaningful next to
        the record they were measured on.
        """
        unit = self._temp_unit()
        if unit == "°C":
            return seg
        return replace(
            seg,
            T_start=_shift_temperature(seg.T_start, "°C", unit),
            T_end=_shift_temperature(seg.T_end, "°C", unit),
            T_min=_shift_temperature(seg.T_min, "°C", unit),
            T_max=_shift_temperature(seg.T_max, "°C", unit),
        )

    def _available_scope_keys(self) -> List[Tuple[str, str]]:
        """`[(scope key, combo text), ...]` for the current segmentation.

        An aggregate is offered only when the run actually holds segments of
        that kind, so the list can never present a scope that resolves to
        nothing. A single-segment record gets one entry and no aggregates:
        there is nothing to choose, and offering "All heating segments" as a
        synonym for "the only segment" is noise.
        """
        run = self.run
        if run is None or not run.segments:
            return []
        keys: List[Tuple[str, str]] = []
        if len(run.segments) > 1:
            keys.append((SCOPE_ALL_TIME,
                         f"{AGGREGATE_SCOPE_LABELS[SCOPE_ALL_TIME]} — "
                         f"{len(run.segments)} segments"))
            for key, kind in ((SCOPE_ALL_HEATING, KIND_HEAT),
                              (SCOPE_ALL_COOLING, KIND_COOL)):
                n = len(run.by_kind(kind))
                # Two or more, not one: an aggregate over a single segment is
                # that segment under a second name, and `BM-0.txt` (one heat
                # plus a trailing hold) would otherwise list "All heating
                # segments (1)" directly above "#1 heat".
                if n > 1:
                    keys.append((key, f"{AGGREGATE_SCOPE_LABELS[key]} ({n})"))
        for seg in run.segments:
            keys.append((segment_scope_key(seg.index),
                         segment_label(self._segment_for_display(seg),
                                       unit=self._temp_unit())))
        return keys

    def _resolved_scope_key(self, key: Any = None) -> str:
        """`key` if this run can honour it, else the run's own default.

        The counterpart of `scope_from_key`: that one resolves a key to a
        scope, this one resolves it to the key actually in force, which is what
        the combo, the scope memory and the zoom key all have to agree on.

        Validated against `_available_scope_keys` rather than against a second
        copy of the "which scopes exist" rule, deliberately: the combo and this
        check must never be able to disagree about whether a key is offered,
        and building twelve labels to answer the question is cheaper than the
        class of bug where they do.
        """
        run = self.run
        if run is None or not run.segments:
            return SCOPE_ALL_TIME
        if len(run.segments) == 1:
            return segment_scope_key(run.segments[0].index)
        wanted = str(self._scope_key if key is None else key)
        for available, _text in self._available_scope_keys():
            if available == wanted:
                return wanted
        return default_scope_key(run)

    def _remembered_scope_key(self) -> str:
        """The scope this record was last left on, or its default.

        Unlike `_run_cache`, the memory accepts title-keyed records too. The
        risk is bounded in a way the run cache's is not: a remembered SCOPE is
        re-validated against the run in hand, is shown in the combo and in the
        note, and the worst outcome is the wrong segment being pre-selected in
        plain sight. A wrongly cached `SegmentedRun`, by contrast, would slice
        one file's data on another file's boundaries with nothing on screen to
        say so.
        """
        key = self._scope_memory.get(self._record_id) if self._record_id else None
        return self._resolved_scope_key(key if key is not None
                                        else default_scope_key(self.run))

    def _refresh_segment_combo(self) -> None:
        """Rebuild the scope list from the current segmentation.

        Repopulated with signals blocked AND `_suspend_scope` set, belt and
        braces, because the handler this guards does far more than redraw: it
        rewrites the Tg window, re-runs the guardrails and can move the X
        column. `clear()` alone emits `currentIndexChanged`.
        """
        combo = self.segment_combo
        keys = self._available_scope_keys()
        self._scope_key = self._resolved_scope_key()

        self._suspend_scope = True
        combo.blockSignals(True)
        try:
            combo.clear()
            for key, text in keys:
                combo.addItem(text, key)
            if keys:
                index = next((i for i, (k, _t) in enumerate(keys)
                              if k == self._scope_key), 0)
                combo.setCurrentIndex(index)
                self._scope_key = keys[index][0]
            # One entry is a statement, not a choice: a single-ramp file has
            # exactly one scope and the combo is there to name it.
            combo.setEnabled(len(keys) > 1)
        finally:
            combo.blockSignals(False)
            self._suspend_scope = False

    def _on_segment_changed(self, _index: int = -1) -> None:
        """The user picked a scope: remember it, re-guard, and queue a redraw.

        The redraw goes through `plot.request_redraw` rather than being drawn
        here, so holding the arrow key down through a 15-entry list coalesces
        into one render of a 19k-point file instead of fifteen (R5). Everything
        that is NOT drawing -- the window, the guardrails, the note -- is
        applied immediately, because those are what the user reads while the
        debounce is still pending.
        """
        if self._suspend_scope:
            return
        data = self.segment_combo.currentData()
        if data is None:
            return          # nothing is selected: an empty or half-built list
        key = self._resolved_scope_key(data)
        if key == self._scope_key:
            return          # re-selecting the same scope is not a change
        self._scope_key = key
        if self._record_id:
            self._scope_memory[self._record_id] = self._scope_key
        # A Tg measured on segment 1 must not stay on screen over segment 4's
        # curve. It does not have to be thrown away either: results are keyed
        # by segment, so this shows the new scope whatever IT has measured,
        # which is nothing until the user computes there.
        self._show_results_for_scope()
        # Guardrails FIRST: they can move the X column, and the default window
        # is expressed in whatever the abscissa is. Reversed, a segment
        # selected off the whole-run view would get a window in minutes
        # written into a temperature axis.
        self._update_scope_guardrails()
        self._apply_default_window()
        # Cheap, and it is the only feedback a user reading the Program tab
        # gets from a row click: the scope combo they just drove lives on the
        # Tg tab. Rebuilding the two tables here would not be cheap, so this
        # updates the sentence and the row highlight only.
        self._refresh_program_selection_label()
        if self.df is not None:
            self.plot.request_redraw(self._render)

    def _on_x_change(self, _text: str = "") -> None:
        """The X column changed: re-guard, re-default the window, redraw.

        The window fields mean whatever the abscissa means, so moving from a
        temperature axis to the clock turns 350-700 °C into 350-700 min, which
        on a 890-minute run is off the end of the data. The segment-derived
        default is re-applied in the new abscissa's units -- but only on a
        multi-segment record and only when the user has not typed a window, so
        the single-ramp file's pinned 350/700 cannot move.
        """
        # The user owns the abscissa from here on: a later scope change may
        # still force the clock where the maths requires it, but it will not
        # quietly hand the axis back afterwards.
        self._x_forced_time = False
        self._update_scope_guardrails()
        self._apply_default_window()
        self._refresh_plot()

    def _apply_default_window(self) -> None:
        """Put the Tg window inside the current segment's own range.

        Deliberately a no-op on a single-segment record. The historic 350/700
        box was authored for exactly that case and the pinned Tg
        354.4666531002641 depends on it; the auto window would clamp 700 to
        388.71 and move the number. It is also a no-op once the user has typed
        in the field -- `_window_user_edited` is set from `textEdited`, so a
        programmatic `setText` here can never be mistaken for the user's own
        choice.

        On a multi-segment record the window follows the ABSCISSA: the central
        60 % of the segment's temperature range on a temperature axis, of its
        time span on the clock. A window that falls outside the data is what
        produces "Baseline range has too few points" three dialogs later.
        """
        run = self.run
        if run is None or len(run.segments) <= 1 or self._window_user_edited:
            return
        seg = self._scope().segment
        if seg is None:
            return
        x_col = self.x_combo.currentText()
        if _looks_like_time(x_col):
            lo, hi = float(seg.t_start), float(seg.t_end)
            mid, half = 0.5 * (lo + hi), 0.30 * (hi - lo)
            lo, hi = mid - half, mid + half
        elif _is_temperature_col(x_col):
            lo, hi = run.default_window(seg)
            unit = self._temp_unit()
            lo = _shift_temperature(lo, "°C", unit)
            hi = _shift_temperature(hi, "°C", unit)
        else:
            # The abscissa is a signal column. A segment knows its time span
            # and its temperature span and nothing else, so there is no
            # default to offer -- and writing a temperature range into a
            # milliwatt axis would be a number that looks deliberate and is
            # not. The user's window stands.
            return
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return
        self.xmin_edit.setText(f"{lo:.6g}")
        self.xmax_edit.setText(f"{hi:.6g}")

    def _apply_x_axis_rule(self, scope: DtaScope) -> None:
        """Enforce `scope.x_kind` on the X combo. Never raises.

        Two halves. The ENTRIES are disabled so a forbidden abscissa cannot be
        chosen, each with a tooltip saying why; and the current selection is
        moved to the clock when the scope forces it. The move is made with
        signals blocked -- the caller redraws once, and an unblocked
        `setCurrentText` here would re-enter the whole scope chain through
        `_on_x_change` (R5).

        Nothing is disabled when `x_kind` is empty, which is every single-ramp
        file: the pinned unit-switch test moves X to `Time (min)` and back.
        """
        force_time = scope.x_kind == X_TIME
        tip = ("Temperature is not an abscissa here. "
               + ("A hold has no temperature scan -- its measured temperatures "
                  "repeat and its derivative is infinite where they do."
                  if scope.is_single else
                  "This view spans several segments that pass over the same "
                  "temperatures, so one temperature does not name one point.")
               + " Select a single ramp segment to plot against temperature.")
        model = self.x_combo.model()
        for i in range(self.x_combo.count()):
            item = model.item(i) if hasattr(model, "item") else None
            if item is None:
                continue
            blocked = force_time and _is_temperature_col(self.x_combo.itemText(i))
            item.setEnabled(not blocked)
            item.setData(tip if blocked else None, Qt.ToolTipRole)

        t_col, T_col = _time_temp_columns(self.colnames, self._canonical_map)
        current = self.x_combo.currentText()
        if force_time:
            if t_col and current != t_col:
                self._set_x_column(t_col)
                self._x_forced_time = True
        elif self._x_forced_time:
            # The scope no longer forces the clock, so hand the abscissa back.
            # Only when the clock is still where WE put it: `_on_x_change`
            # clears the flag the moment the user picks a column, and a
            # choice of theirs must survive every scope change after it.
            self._x_forced_time = False
            if T_col and t_col and current == t_col:
                self._set_x_column(T_col)

    def _set_x_column(self, col: str) -> None:
        """Move the X combo programmatically, without re-entering the chain.

        Signals stay blocked: the caller is mid-guardrail and redraws once
        afterwards, whereas an unblocked `setCurrentText` re-enters
        `_on_x_change` -> `_update_scope_guardrails` -> here (R5).
        """
        self.x_combo.blockSignals(True)
        try:
            self.x_combo.setCurrentText(col)
        finally:
            self.x_combo.blockSignals(False)
        self._update_temp_switch_enabled(col)

    def _update_scope_guardrails(self) -> None:
        """G1: switch off what this scope cannot answer, and say why.

        The rule is arithmetic, not physical. A tangent construction, an
        integral and a "find max" all need ONE monotone scan with a
        single-valued abscissa; an aggregate view is several, and on the real
        cyclic file a whole-file Tg collapses all three methods onto 431.758 °C
        with a 4e-5 spread while the app cheerfully reports agreement. So the
        actions are disabled and `scope_note_label` carries the reason.

        Never a dialog: a warning that has to be clicked away is a warning that
        gets clicked away, and this one is standing information about what the
        user is looking at, not an event.
        """
        scope = self._scope()
        live = self.df is not None and scope.computable
        reason = "" if live else self._scope_block_reason(scope)
        for btn in (self.compute_btn, self.integrate_btn, self.calc_max_btn,
                    self.calc_min_btn, self.export_csv_btn):
            btn.setEnabled(live)
            btn.setToolTip(reason)
        self._apply_x_axis_rule(scope)
        self.scope_note_label.setText(self._scope_note_text(scope))
        # Deliberately NOT in the loop above. "Export all segments" measures
        # each segment on its own, so the aggregate scope that switches the
        # single-scan actions off is precisely the scope it is most wanted in;
        # gating it on `computable` would disable the one export that answers a
        # cyclic file.
        self._refresh_batch_notes()

    def _scope_block_reason(self, scope: DtaScope) -> str:
        """Why the Tg / Calculs actions are off, in one sentence."""
        if self.df is None:
            return "Load a record first."
        return (f"{len(scope.segments)} segments are in scope. A tangent, an "
                "integral or an extremum is defined on one monotone scan — "
                "pick a single segment above.")

    def _scope_note_text(self, scope: DtaScope) -> str:
        """The standing note under the scope combo. Plain language, no jargon.

        Carries three things at once: what is being shown, what cannot be
        computed on it and why, and every reason to distrust the segment in
        hand (partial acquisition, a setpoint never reached, control lost on
        the way down). Those last ones are measured facts about the data and
        they must be in front of the user before a number is, not in a manual.
        """
        if self.df is None:
            return ""
        run = self.run
        unit = self._temp_unit()
        if run is None or not run.segments:
            return ("Whole record — no clock/temperature pair was found, so it "
                    "was not split into segments and is analysed as one scan.")

        seg = scope.segment
        if seg is None:
            bits = [scope.label + "."]
            if scope.kind == SCOPE_ALL_TIME:
                bits.append("Temperature is plotted against time and every "
                            "segment is shaded.")
            else:
                bits.append("Each segment is drawn as its own curve.")
            bits.append("Tg, Calculs and the result CSV need a single segment: "
                        "select one above.")
            return " ".join(bits)

        disp = self._segment_for_display(seg)
        bits = [segment_label(disp, unit=unit) + "."]
        if len(run.segments) > 1:
            bits.append(f"Segment {seg.index} of {len(run.segments)}.")
        if seg.is_hold:
            bits.append("Temperature is not an abscissa on a hold "
                        f"(it repeats within {disp.T_max - disp.T_min:.2f} {unit}), "
                        "so the X axis is fixed to time.")
        warning = self._segment_warning_text(seg)
        if warning:
            bits.append("⚠ " + warning)
        return " ".join(bits)

    def _scope_attribution(self) -> str:
        """`'seg #3 cool, cycle 1'` — which segment a number came from.

        Empty on a record with one segment or none, where there is nothing to
        attribute. On a cyclic run it is the difference between "Max = 35.136
        at x = 434.328" and the same figure with the fact, invisible today,
        that it was read off a COOLING branch of the first cycle. A number
        that outlives its context (a screenshot, a pasted line in a notebook)
        must carry the context with it.
        """
        run = self.run
        seg = self._scope().segment
        if seg is None or run is None or len(run.segments) <= 1:
            return ""
        bits = [f"seg #{seg.index} {seg.kind}"]
        if seg.cycle_index:
            bits.append(f"cycle {seg.cycle_index}")
        return ", ".join(bits)

    def _with_attribution(self, text: str) -> str:
        """`text` with the segment attribution appended, when there is one."""
        attribution = self._scope_attribution()
        return f"{text}  [{attribution}]" if attribution else text

    def _segment_warning_text(self, seg: Segment) -> str:
        """Every measured reason to distrust this segment, in one sentence.

        `Segment.note` is preferred WHOLE when it exists. It is the science
        layer's own sentence, assembled from the alignment, and it already
        names the missed target, the absorbed steps, the partial acquisition
        and the temperature below which rate control was lost. Re-deriving
        those from the fields next to it produced the same facts twice.

        The note is rendered verbatim, in °C, even when the workspace is in
        kelvin. Its numbers are °C and only a text substitution is available
        here (`segment_table` does exactly that and thereby prints a °C number
        under a K symbol); a correctly-labelled °C sentence beside a converted
        label is honest, a mislabelled one is not.

        The field-by-field fallback below is for a record with no programmed
        method, where there is no alignment and therefore no note.
        """
        if seg.note:
            return seg.note if seg.note.endswith(".") else seg.note + "."
        bits: List[str] = []
        if seg.is_partial:
            bits.append("acquisition stopped inside this segment")
        if np.isfinite(seg.target_gap_C) and seg.target_gap_C > 5.0:
            bits.append(f"the programmed target was missed by {seg.target_gap_C:.0f} °C")
        if np.isfinite(seg.control_lost_below_C):
            bits.append("rate control was lost below "
                        f"{seg.control_lost_below_C:.0f} °C — the quoted rate "
                        "does not apply under that")
        return ("; ".join(bits) + ".") if bits else ""

    # ------------------------------------------------------------------
    # Program tab — the .mth cross-check
    # ------------------------------------------------------------------

    def load_mth(self, path) -> ProgramComparison:
        """Load a TA `.mth` and cross-check it against this record's OrgMethod.

        Directly callable with a path so the whole cross-check can be exercised
        without monkeypatching a file dialog; `_on_load_mth` is the four-line
        wrapper that only chooses the path.

        **The loaded file never drives anything.** `self.program` -- what the
        segmentation, the segment labels, the programmed rates and every
        Matched cell are built from -- stays the OrgMethod the instrument wrote
        next to the data. A `.mth` on disk is only what somebody saved under
        that name; the header is what the instrument says it actually ran. So
        the whole authority of a loaded `.mth` is to confirm a step (Source
        "both") or to disagree with it, loudly, in the warnings area.

        Never raises and never opens a modal dialog, not even on a file that is
        not a method file at all: `parse_mth_file` reports failure through
        `status` / `error`, and that failure belongs on screen next to the data
        rather than in a box to be clicked away.
        """
        p = Path(path)
        self.mth_program = parse_mth_file(p)
        self.mth_path = p
        self._recompare_mth()
        self._refresh_program_tab()
        # The verdict sentence already names the file, so it stands alone.
        self._set_status(self._mth_verdict_text())
        return self.mth_comparison

    def clear_mth(self) -> None:
        """Forget the loaded `.mth`. The record's own method is untouched."""
        self.mth_program = None
        self.mth_path = None
        self.mth_comparison = None
        self._refresh_program_tab()
        self._set_status("Cleared the loaded .mth. The method shown is the one "
                         "recorded in the data file.")

    def _on_load_mth(self) -> None:
        """Dialog wrapper over `load_mth`."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TA method file", "",
            "TA method files (*.mth);;All files (*.*)")
        if not path:
            return
        self.load_mth(Path(path))

    def _on_clear_mth(self) -> None:
        """Button wrapper over `clear_mth`."""
        self.clear_mth()

    def _recompare_mth(self) -> None:
        """Re-run the comparison. Called whenever EITHER side can have changed.

        A record switch counts: the `.mth` a user loaded to check record A says
        nothing about record B, and a verdict left over from A is worse than no
        verdict because it looks current. `compare_programs` treats an
        unreadable side as "incomparable" rather than as agreement, so a
        missing header cannot be talked into confirming anything.
        """
        self.mth_comparison = (None if self.mth_program is None else
                               compare_programs(self.program, self.mth_program))

    def _method_view(self) -> Optional[MethodProgram]:
        """The program to DISPLAY: OrgMethod, tagged "both" where a `.mth` agrees.

        The merge happens only when the two are the SAME METHOD. A mismatching
        `.mth` describes a different run, so letting it tag rows here would put
        the word "both" against steps two unrelated methods happen to share and
        would append its own extra steps to a table that is supposed to be the
        record of what ran. On a mismatch the table shows the OrgMethod alone
        and the disagreement goes to the warnings area, itemised.
        """
        prog = self.program
        if prog is None or not prog:
            return prog
        mth = self.mth_program
        cmp_ = self.mth_comparison
        if mth is None or not mth or cmp_ is None or not cmp_.is_same_method:
            return prog
        return merge_programs(prog, mth)

    def _mth_verdict_text(self) -> str:
        """One sentence on the loaded `.mth`, or the "none loaded" statement."""
        mth, cmp_ = self.mth_program, self.mth_comparison
        if mth is None:
            return ("No .mth loaded. The method shown is the one recorded in "
                    "the data file's own header, which is the authoritative one.")
        name = self.mth_path.name if self.mth_path else "the method file"
        if not mth:
            return (f"{name} could not be read as a TA method file"
                    + (f" ({mth.error})" if mth.error else "")
                    + ". Nothing from it is shown.")
        if cmp_ is None:
            return f"{name}: loaded, not yet compared."
        pct = f"{cmp_.score * 100:.0f}%"
        if cmp_.verdict == "identical":
            # len(cmp_.rows), not len(mth.thermal_steps): the comparison is
            # made on the PHYSICAL steps, so a method written as one cycle
            # plus `Repeat segment 1 for 3 times` compared four passes and
            # should say four passes' worth.
            return (f"{name} is identical to the method recorded in the data "
                    f"file: all {len(cmp_.rows)} thermal steps match.")
        if cmp_.verdict == "equivalent":
            return (f"{name} is equivalent to the method recorded in the data "
                    f"file ({pct} of thermal steps match exactly).")
        if cmp_.verdict == "incomparable":
            return (f"{name} could not be compared with this record: "
                    + ("; ".join(cmp_.reasons) or "one side has no thermal steps")
                    + ".")
        return (f"{name} DOES NOT MATCH the method recorded in the data file "
                f"({pct} of thermal steps agree). It is shown for comparison "
                "only and relabels nothing.")

    # ------------------------------------------------------------------
    # Program tab — step / segment correspondence
    # ------------------------------------------------------------------

    def _step_matches(self) -> Dict[int, Tuple[str, Optional[int], str]]:
        """`{written step number: (cell text, segment to select, tooltip)}`.

        Four distinct answers, and they are four because the last three get
        collapsed into "no" by anything that only asks "was this step found":

          * a thermal step with its own detected segment -> ``"#3 cool"``;
          * a thermal step the furnace was still travelling through ->
            ``"#3 (absorbed)"``: it RAN, it left no separate signature;
          * a thermal step the data never shows -> ``"not executed"``;
          * a NON-thermal step (select gas, mass flow) -> ``"—"``, because it
            moves no setpoint and therefore cannot leave a signature in a
            temperature record. Calling that one "not executed" would accuse
            the instrument of skipping an instruction it carried out.

        Keyed by WRITTEN step number, and a number can own several segments: a
        `Repeat segment` block makes every pass carry the index the operator
        wrote once.
        """
        out: Dict[int, Tuple[str, Optional[int], str]] = {}
        program = self.program
        if program is None or not program:
            return out

        thermal = {int(s.index) for s in program.steps if s.is_thermal}
        for st in program.steps:
            if int(st.index) not in thermal:
                out[int(st.index)] = (
                    "—", None,
                    "Not a thermal step: it moves no setpoint, so it leaves no "
                    "signature in the temperature record.")

        run = self.run
        if run is None or not run.segments:
            for ix in sorted(thermal):
                out[ix] = ("—", None,
                           "This record was not split into segments, so no step "
                           "could be matched to it.")
            return out

        owners: Dict[int, List[Segment]] = {}
        absorbed: Dict[int, List[Segment]] = {}
        for seg in run.segments:
            match = run.alignment.match_for(seg.index)
            absorbed_here = set(match.absorbed_step_indices) if match else set()
            for ix in seg.step_indices:
                bucket = absorbed if int(ix) in absorbed_here else owners
                bucket.setdefault(int(ix), []).append(seg)

        absorbed_tip = ("This step ran, but left no separate signature: the "
                        "furnace was still travelling through it, so the "
                        "record cannot separate it from the ramp it happened "
                        "during. Absorbed is not 'not executed'.")
        unmatched = {int(i) for i in run.alignment.unmatched_steps}
        for ix in sorted(thermal):
            owning, soaked = owners.get(ix, []), absorbed.get(ix, [])
            if owning or soaked:
                # BOTH lists, not whichever is non-empty first. A `Repeat
                # segment` block runs one written step several times and the
                # furnace need not behave the same way on every pass: the hold
                # can settle on pass 1 and be absorbed into the cooling tail on
                # pass 4. Reporting only the pass that produced a segment would
                # quietly show the flattering half of the run.
                parts = ([(s, f"#{s.index} {s.kind}") for s in owning]
                         + [(s, f"#{s.index} (absorbed)") for s in soaked])
                parts.sort(key=lambda pair: int(pair[0].index))
                tips = [self._segment_tooltip(s) for s in owning]
                if soaked:
                    tips.append(absorbed_tip)
                out[ix] = (", ".join(label for _seg, label in parts),
                           int(parts[0][0].index), " ".join(tips))
            elif ix in unmatched:
                out[ix] = ("not executed", None,
                           "Nothing in the data matches this step. As far as "
                           "the measurement shows it never ran.")
            else:
                out[ix] = ("—", None,
                           "No alignment was made between this step and the "
                           "data.")
        return out

    def _segment_tooltip(self, seg: Segment) -> str:
        """Full description of one segment, in °C, for a cell tooltip.

        `describe_segment`'s default unit, never the display unit: its
        sentence carries the three rate estimators and the segment's own
        warnings as free text, and `unit=` there substitutes the degree symbol
        without converting a single number. A °C sentence correctly labelled
        °C is honest next to a kelvin axis; a relabelled one is not.
        """
        return describe_segment(seg)

    def _step_text_summary(self, indices: Sequence[int]) -> str:
        """The distinct step texts behind `indices`, e.g. `Isothermal for 20.00 min`."""
        program = self.program
        if program is None or not program:
            return ""
        by_index = {int(s.index): s for s in program.steps}
        seen: List[str] = []
        for ix in indices:
            st = by_index.get(int(ix))
            if st is None:
                continue
            text = (st.text or st.render()).strip()
            if text and text not in seen:
                seen.append(text)
        return "; ".join(seen)

    # ------------------------------------------------------------------
    # Program tab — refresh
    # ------------------------------------------------------------------

    def _refresh_program_tab(self) -> None:
        """Rebuild the whole tab from current state. Never raises.

        Deliberately does NOT rewrite the three detection fields: this runs on
        a `.mth` load too, and silently reverting a half-typed knob because the
        user picked a file is the kind of small theft that makes an app feel
        haunted. `_refresh_detection_fields` is called where the settings
        actually change instead.
        """
        self._refresh_method_table()
        self._refresh_program_table()
        self._refresh_program_warnings()
        self._refresh_program_selection_label()
        self.mth_note_label.setText(self._mth_verdict_text())

    def _refresh_method_table(self) -> None:
        """One row per programmed step, in written order."""
        table = self.method_table
        program = self._method_view()
        steps: Tuple[MethodStep, ...] = tuple(program.steps) if program else ()
        matches = self._step_matches()

        table.setRowCount(len(steps))
        for row, st in enumerate(steps):
            matched_text, seg_index, matched_tip = matches.get(
                int(st.index), ("—", None, ""))
            cells = (
                str(int(st.index)),
                (st.text or st.render()),
                str(st.kind),
                _fmt_method_number(st.rate_C_min),
                _fmt_method_number(st.target_C),
                _fmt_method_number(st.duration_min),
                _STEP_SOURCE_LABELS.get(str(st.source), str(st.source)),
                matched_text,
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 0:
                    # The click target lives on column 0 of every row, so a
                    # click anywhere in the row resolves the same way.
                    item.setData(Qt.UserRole, seg_index)
                if col == len(cells) - 1 and matched_tip:
                    item.setToolTip(matched_tip)
                if col == 1:
                    item.setToolTip(st.render())
                if not st.is_recognized:
                    item.setToolTip(
                        "This line of the method was not understood. It is "
                        "shown verbatim and kept in place, so every step "
                        "number after it still matches the method as written.")
                table.setItem(row, col, item)
        self._fit_program_columns(table)

    def _refresh_program_table(self) -> None:
        """Detected segments, plus the absorbed and never-executed steps.

        An absorbed row is placed immediately AFTER the segment that absorbed
        it, because that adjacency is the fact being reported: the hold is
        inside that cool. Never-executed rows go at the end, where nothing is
        adjacent to them, which is also the truth.

        Measured columns of a non-segment row are left BLANK rather than zero:
        there is no measurement, and a 0.0 in a duration column is a claim.
        Its `Kind` is the PROGRAMMED kind of the step ("isothermal"), which is
        unambiguous only because `Status` is a column of its own -- that is the
        whole reason the status is not smuggled into `kind`.
        """
        table = self.program_table
        run = self.run
        rows: List[Tuple[Dict[str, Any], Optional[int], str]] = []

        if run is not None and run.segments:
            measured = segment_table(run.segments)
            for seg, data in zip(run.segments, measured):
                row = dict(data)
                row["status"] = PROGRAM_ROW_DETECTED
                rows.append((row, int(seg.index), self._segment_tooltip(seg)))

                match = run.alignment.match_for(seg.index)
                for ix in (match.absorbed_step_indices if match else ()):
                    rows.append((
                        {"status": PROGRAM_ROW_ABSORBED,
                         "kind": self._step_kind(ix),
                         "steps": str(int(ix)),
                         "note": (f"step {int(ix)} ({self._step_text_summary([ix])}) "
                                  f"ran but left no separate signature: the furnace "
                                  f"was still travelling through it, so it is "
                                  f"absorbed into segment #{seg.index}")},
                        int(seg.index),
                        "Absorbed — it ran. That is a different claim from "
                        "'not executed'."))

            for ix in run.alignment.unmatched_steps:
                rows.append((
                    {"status": PROGRAM_ROW_NOT_EXECUTED,
                     "kind": self._step_kind(ix),
                     "steps": str(int(ix)),
                     "note": (f"step {int(ix)} ({self._step_text_summary([ix])}) "
                              f"was programmed but never appears in the data")},
                    None,
                    "Not executed — nothing in the record matches this step."))

        table.setRowCount(len(rows))
        for r, (data, seg_index, tip) in enumerate(rows):
            for col, key in enumerate(PROGRAM_TABLE_KEYS):
                item = QTableWidgetItem(_fmt_cell(data.get(key, "")))
                if col == 0:
                    item.setData(Qt.UserRole, seg_index)
                    if tip:
                        item.setToolTip(tip)
                table.setItem(r, col, item)
        self._fit_program_columns(table)
        self._highlight_program_rows()

    def _step_kind(self, index: int) -> str:
        """`MethodStep.kind` of a written step number, or ''."""
        program = self.program
        if program is None or not program:
            return ""
        for st in program.steps:
            if int(st.index) == int(index):
                return str(st.kind)
        return ""

    def _refresh_program_warnings(self) -> None:
        """Put `program_warning_lines()` on screen, visibly, or say all is well."""
        lines = self.program_warning_lines()
        label = self.program_warning_label
        if lines:
            label.setText("\n\n".join("⚠ " + line for line in lines))
            label.setStyleSheet(f"color: {_WARN_FG}; font-weight: 600;")
        elif self.df is None:
            label.setText("")
            label.setStyleSheet("")
        else:
            label.setText("No discrepancy between the programmed method and "
                          "the data.")
            label.setStyleSheet("")

    def program_warning_lines(self) -> List[str]:
        """Everything the Program tab has to report, worst first, in English.

        Public and argument-free on purpose: a test, and later an export, must
        be able to read the same sentences the user reads rather than
        re-deriving them from the widgets and drifting.

        Ordered by how badly each can mislead. A `.mth` from another run comes
        first because it can make every other row on the tab wrong. Nothing
        here refuses anything -- the user asked that unusual runs still be
        processable, so all of this is information carried alongside the
        computation, never a gate in front of it.
        """
        lines: List[str] = []
        if self.df is None:
            return lines

        prog = self.program
        cmp_ = self.mth_comparison
        if self.mth_program is not None and (cmp_ is None or not cmp_.is_same_method):
            lines.append(self._mth_verdict_text()
                         + " The segment labels, the programmed rates and every "
                           "match on this tab come from the method recorded in "
                           "the data file itself.")
            if cmp_ is not None and cmp_.reasons:
                lines.append("Where they differ: " + "; ".join(cmp_.reasons[:5]) + ".")

        if prog is None or not prog:
            why = (prog.error if prog is not None and prog.error
                   else "no OrgMethod lines were found in the file header")
            lines.append(f"No programmed method could be read from this file ({why}). "
                         "Segments are derived from the data alone, so there is "
                         "nothing to match them against and no programmed rate "
                         "to compare a measured one with.")
        else:
            if prog.status == PROGRAM_PARTIAL:
                lines.append(
                    f"{prog.n_unrecognized} step(s) of the method were not "
                    "understood. They are shown verbatim and kept in place, so "
                    "the step numbers here still match the method as written — "
                    "but they take no part in the alignment.")
            for warning in prog.warnings:
                lines.append(str(warning))

        run = self.run
        if run is None or not run.segments:
            lines.append(
                "This record was not split into segments: both a clock column "
                "and a furnace-temperature column are needed and one of them "
                "was not found. It is analysed as a single scan.")
            return lines

        align = run.alignment
        # Both "unmatched" findings are claims about a COMPARISON, so they are
        # only meaningful once the comparison actually ran. With no method at
        # all every segment is trivially unmatched, and saying "the record
        # contains more thermal activity than the method accounts for" about a
        # method that does not exist is simply false -- the "no programmed
        # method" line above has already said the only true thing there is.
        aligned = align.method == "dp"
        if not aligned and align.method != "no-program":
            # The alignment was attempted and failed. Its own notes say why,
            # and without them the tab would show a method and a set of
            # segments with nothing to say the two were never connected.
            for note in align.notes:
                lines.append(str(note))
        if aligned and align.unmatched_steps:
            lines.append(
                "Programmed step(s) " + _join_ints(align.unmatched_steps)
                + " never appear in the data. That is NOT the same claim as the "
                  "absorbed steps below: as far as the measurement shows, these "
                  "did not run.")
        if aligned and align.unmatched_segments:
            lines.append(
                "Segment(s) " + _join_ints(align.unmatched_segments)
                + " are in the data but match no programmed step: the record "
                  "contains more thermal activity than the method accounts for.")

        absorbed = tuple(dict.fromkeys(align.absorbed_steps))
        if absorbed:
            text = self._step_text_summary(absorbed)
            lines.append(
                "Programmed step(s) " + _join_ints(absorbed)
                + (f" ({text})" if text else "")
                + " ran but left no separate signature: the furnace was still "
                  "travelling through them. They are listed as \"absorbed\", "
                  "which is a different claim from \"not executed\" — they "
                  "happened, and the record simply cannot separate them from "
                  "the ramp they happened during.")

        missed = [s for s in run.segments
                  if np.isfinite(s.target_gap_C) and s.target_gap_C > TARGET_GAP_WARN_C]
        if missed:
            bits = [f"#{s.index} {s.kind} stopped at {s.T_end:.0f} °C, "
                    f"{s.target_gap_C:.1f} °C short" for s in missed]
            lines.append(
                "Programmed targets that were not reached: " + "; ".join(bits)
                + ". On a heating ramp the setpoint usually arrives during the "
                  "hold that follows, so a small gap there is ordinary; a "
                  "cooling ramp that stops short never got there at all, and "
                  "any hold programmed after it was spent still coasting.")

        lost = [s for s in run.segments if np.isfinite(s.control_lost_below_C)]
        if lost:
            bits = []
            for s in lost:
                rate = (f"{s.rate_nominal_C_min:+.0f} °C/min programmed"
                        if np.isfinite(s.rate_nominal_C_min) else "programmed rate")
                bits.append(f"#{s.index} ({rate}) down to "
                            f"{s.control_lost_below_C:.0f} °C")
            first = lost[0]
            example = ""
            if np.isfinite(first.rate_nominal_C_min):
                example = (f" For segment #{first.index}, "
                           f"\"{first.rate_nominal_C_min:+.0f} °C/min\" is true at "
                           f"{first.T_start:.0f} °C and is no longer true below "
                           f"{first.control_lost_below_C:.0f} °C.")
            lines.append(
                "The furnace held the programmed rate only part of the way: "
                + "; ".join(bits)
                + ". Below those temperatures it was coasting, not following "
                  "the programme, so the programmed rate describes the top of "
                  "each ramp and not the bottom." + example
                + " Any rate-dependent quantity read below that point carries a "
                  "rate the specimen never saw.")

        partial = [s for s in run.segments if s.is_partial]
        if partial:
            one = len(partial) == 1
            lines.append(
                "Acquisition stopped inside segment(s) "
                + _join_ints([s.index for s in partial])
                + (": its duration and its end temperature are"
                   if one else ": their durations and end temperatures are")
                + " where the recording stopped, not where the programme did.")

        p_tot, m_tot = align.programmed_total_min, align.measured_total_min
        if aligned and np.isfinite(p_tot) and np.isfinite(m_tot) and p_tot > 0:
            diff = m_tot - p_tot
            # 2 % or one minute, whichever is larger. The real cyclic file
            # agrees to 0.5 min in 890 (0.06 %), so anything this catches is a
            # genuine departure rather than the arithmetic of a rounded method.
            if abs(diff) > max(1.0, 0.02 * p_tot):
                lines.append(
                    f"The run lasted {m_tot:.1f} min against {p_tot:.1f} min "
                    f"programmed ({diff:+.1f} min). The programmed clock assumes "
                    "every ramp takes |target − start| / rate minutes, so a large "
                    "difference means the furnace was not tracking the setpoint, "
                    "or the run was stopped early.")
        return lines

    def _program_summary_text(self) -> str:
        """The standing one-paragraph statement of what this tab is showing."""
        if self.df is None:
            return "No record loaded."
        prog = self.program
        bits: List[str] = []
        if prog is not None and prog:
            n_ramp = sum(1 for s in prog.steps if s.kind == STEP_RAMP)
            n_hold = sum(1 for s in prog.steps if s.kind == STEP_ISOTHERMAL)
            bits.append(f"Method: {_plural(len(prog.steps), 'step')} "
                        f"({_plural(n_ramp, 'ramp')}, {_plural(n_hold, 'hold')}), "
                        "read from the data file's own header.")
        else:
            bits.append("Method: none could be read from the file header.")
        run = self.run
        if run is not None and run.segments:
            span = run.segments[-1].t_end - run.segments[0].t_start
            bits.append(f"Data: {_plural(len(run.segments), 'segment')} in "
                        f"{_plural(run.n_cycles, 'cycle')} over {span:.1f} min.")
            align = run.alignment
            if align.method == "no-program":
                bits.append("No alignment: segments are data-derived only.")
            elif align.ok:
                bits.append("Every programmed step is accounted for.")
        else:
            bits.append("Data: not segmented.")
        return " ".join(bits)

    def _refresh_program_selection_label(self) -> None:
        """Say which segment the workspace is pointed at, on this tab.

        The scope combo lives on the Tg tab, so a click here would otherwise
        have no visible effect for a user reading the Program tab. Saying it in
        words is the alternative to yanking them to another tab, which is rude
        and loses their place in a 19-row method.
        """
        if getattr(self, "program_summary_label", None) is None:
            return                      # called before the tab was built
        self.program_summary_label.setText(self._program_summary_text())
        if self.df is None:
            self.program_selection_label.setText("")
            return
        scope = self._scope()
        seg = scope.segment
        if seg is None:
            self.program_selection_label.setText(
                f"Scope: {scope.label}. Click a row to select a single segment.")
        else:
            self.program_selection_label.setText(
                "Scope: " + segment_label(self._segment_for_display(seg),
                                          unit=self._temp_unit()) + ".")
        self._highlight_program_rows()

    def _highlight_program_rows(self) -> None:
        """Select the table row for the segment currently in scope, if any."""
        table = getattr(self, "program_table", None)
        if table is None:
            return
        seg = self._scope().segment
        if seg is None:
            table.clearSelection()
            return
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            # Column 0 IS the status column, so this also skips the
            # absorbed row that carries the same segment index.
            if (item is not None and item.data(Qt.UserRole) == int(seg.index)
                    and item.text() == PROGRAM_ROW_DETECTED):
                table.selectRow(row)
                return
        table.clearSelection()

    # ------------------------------------------------------------------
    # Program tab — interaction
    # ------------------------------------------------------------------

    def select_segment(self, index: Optional[int]) -> bool:
        """Point the whole workspace at segment `index`. True when it moved.

        Drives the scope COMBO rather than writing `_scope_key` directly. The
        combo is the single place a scope is chosen, and the scope memory, the
        guardrails, the default window, the forgetting of stale results and the
        redraw all hang off its `currentIndexChanged`; a second path that set
        the key by hand would have to reproduce all five and would reproduce
        them slightly differently. Signals are deliberately NOT blocked here
        for exactly that reason.
        """
        if index is None:
            return False
        key = segment_scope_key(int(index))
        combo = self.segment_combo
        for i in range(combo.count()):
            if combo.itemData(i) == key:
                moved = combo.currentIndex() != i
                if moved:
                    combo.setCurrentIndex(i)
                self._refresh_program_selection_label()
                return moved
        return False

    def _on_program_row_clicked(self, row: int, _column: int = 0) -> None:
        """A segment-table row was clicked: select its segment."""
        item = self.program_table.item(int(row), 0)
        self.select_segment(None if item is None else item.data(Qt.UserRole))

    def _on_method_row_clicked(self, row: int, _column: int = 0) -> None:
        """A method-table row was clicked: select the segment that ran it."""
        item = self.method_table.item(int(row), 0)
        self.select_segment(None if item is None else item.data(Qt.UserRole))

    # ------------------------------------------------------------------
    # Program tab — detection settings
    # ------------------------------------------------------------------

    def _refresh_detection_fields(self) -> None:
        """Write the config in force into the three knobs, and explain it."""
        cfg = self._seg_config
        self.seg_window_edit.setText(f"{cfg.rate_window_min:g}")
        self.seg_hold_edit.setText(f"{cfg.min_hold_min:g}")
        self.seg_span_edit.setText(f"{cfg.hold_flat_span_C:g}")
        note = self._detection_effective_note()
        if self._seg_param_note:
            note = (self._seg_param_note + (" " + note if note else ""))
        self.seg_params_note.setText(note)

    def _detection_effective_note(self) -> str:
        """What the typed rate window actually becomes on THIS file's sampling.

        A "0.5 min" spin box is a lie on a 2.8 s-per-point export, and the
        estimator's index-space minimum-points guard is invisible from the UI.
        `SegmentationConfig.validated` exists to answer exactly this and its
        docstring says so.
        """
        run = self.run
        if run is None or np.asarray(run.time_min).size < 3:
            return ""
        dt = float(np.median(np.diff(np.asarray(run.time_min, dtype=float))))
        if not np.isfinite(dt) or dt <= 0.0:
            return ""
        cfg = self._seg_config
        n_points = int(round(cfg.rate_window_min / dt)) + 1
        effective = cfg.validated(dt)
        if effective.rate_window_min <= cfg.rate_window_min:
            return (f"Sampling here is {dt:.4g} min/point, so the "
                    f"{cfg.rate_window_min:g} min window holds about {n_points} "
                    f"points.")
        return (f"Sampling here is {dt:.4g} min/point: a {cfg.rate_window_min:g} min "
                f"window would hold only about {n_points} points, so the estimator "
                f"widens it to {cfg.rate_min_points} points "
                f"(~{effective.rate_window_min:.4g} min). Without that guard a "
                f"coarsely sampled export fragments into dozens of spurious "
                f"segments.")

    def apply_detection_settings(self, *, rate_window_min: Any = None,
                                 min_hold_min: Any = None,
                                 hold_flat_span_C: Any = None
                                 ) -> SegmentationConfig:
        """Re-segment with these knobs and return the config in force.

        Directly callable, so re-detection is testable without going through
        three line edits. Never raises and never opens a dialog.

        A value that is missing, unparseable or not strictly positive is
        REFUSED and the previous one kept, with the reason written under the
        buttons. Not a silent fallback to a default the user did not ask for,
        and not a dialog: a rejected keystroke is not an event worth
        interrupting anyone for. A zero-length rate window or a zero-minute
        minimum hold is not a strict setting, it is an unusable one.

        `_run_cache` is emptied WHOLESALE, not just for the record on screen.
        The cache is keyed by record and carries no memory of the configuration
        a run was detected under, so leaving another record's run in it would
        show two files segmented by two different rules with nothing on screen
        to say which was which.
        """
        cfg = self._seg_config
        refused: List[str] = []
        updates: Dict[str, float] = {}
        for name, label, value in (
                ("rate_window_min", "Rate window", rate_window_min),
                ("min_hold_min", "Min hold", min_hold_min),
                ("hold_flat_span_C", "Flat span", hold_flat_span_C)):
            try:
                number = float(value)              # None raises TypeError
            except (TypeError, ValueError):
                refused.append(f"{label} (not a number)")
                continue
            if not np.isfinite(number) or number <= 0.0:
                refused.append(f"{label} (must be greater than 0)")
                continue
            updates[name] = number
        if updates:
            cfg = replace(cfg, **updates)
        self._seg_config = cfg
        self._seg_param_note = (
            "Kept the previous value for " + ", ".join(refused) + "."
            if refused else "")
        self._run_cache.clear()
        self._redetect_now()
        return cfg

    def _redetect_now(self) -> None:
        """Re-run detection on the loaded record and resettle every view.

        The order matters and is the same as `_apply_dataframe`'s: detect,
        re-resolve the scope, rebuild the combo, forget results, then
        guardrails BEFORE the default window, because the guardrails can move
        the X column and the window is expressed in whatever the abscissa is.

        Results are forgotten rather than redrawn. A Tg measured on the old
        boundaries is not a measurement on the new ones: the curve, the window
        and the baselines have all moved underneath it.
        """
        self._refresh_detection_fields()
        if self.df is None:
            self._refresh_program_tab()
            return
        self._detect_run()
        self._scope_key = self._remembered_scope_key()
        self._refresh_segment_combo()
        self._forget_all_results()
        self._update_scope_guardrails()
        self._apply_default_window()
        self._refresh_detection_fields()
        self._refresh_program_tab()
        self._refresh_plot()
        run = self.run
        self._set_status(
            "Re-detected: " + (run.describe() if run is not None and run
                               else "this record could not be segmented."))

    def _on_redetect(self) -> None:
        """Button wrapper: read the three fields, then `apply_detection_settings`."""
        self.apply_detection_settings(
            rate_window_min=_to_float(self.seg_window_edit.text()),
            min_hold_min=_to_float(self.seg_hold_edit.text()),
            hold_flat_span_C=_to_float(self.seg_span_edit.text()),
        )

    def _on_seg_params_reset(self) -> None:
        """Back to the measured defaults, and re-detect with them."""
        defaults = SegmentationConfig()
        self.apply_detection_settings(
            rate_window_min=defaults.rate_window_min,
            min_hold_min=defaults.min_hold_min,
            hold_flat_span_C=defaults.hold_flat_span_C,
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _smooth_derivative_enabled(self) -> bool:
        return self.smooth_group.checkedId() == 1

    # ---- Temperature unit switch --------------------------------------

    def _temp_unit(self) -> str:
        return "K" if self.temp_unit_group.checkedId() == 1 else "°C"

    def _convert_x_to_display(self, x: np.ndarray, x_col: str) -> np.ndarray:
        """Convert x (as read straight from the dataframe, in whatever unit
        the column's own header claims) to the display unit.

        Temperature follows the °C/K switch. A CLOCK column is converted to
        MINUTES, from the unit written in its own label, because that is what
        the segmenter now does (`_run_for_dataframe`) and the two must agree:
        `Segment.t_start` / `t_end` are minutes whatever the column says, and
        `_apply_default_window` writes them straight into the window fields. On
        a `Time (sec)` export with more than one segment, a minutes window on a
        seconds axis would have been 60x out — introduced by fixing the rate,
        which is why it is fixed in the same breath. Every other column is left
        alone, so switching °C/K never touches a milliwatt axis.
        """
        # The clock FIRST. `_unit_from_col` ends with `"c)" in cl -> "°C"`,
        # which matches `Time (sec)`; with the switch on K that added 273.15
        # to the clock, and it shadowed the conversion below.
        if _looks_like_time(x_col):
            scale = clock_scale_to_min(x_col)[0]
            return x if scale == 1.0 else np.asarray(x, dtype=float) * scale
        native = _unit_from_col(x_col)
        if native:
            return _shift_temperature(x, native, self._temp_unit())
        return x

    def _result_unit_label(self, x_col: str) -> str:
        """Unit suffix for displayed/exported numbers: the switch's current
        setting for a recognized temperature column, otherwise ''.

        A CLOCK column is tested first and answers '' deliberately. This is
        what the per-segment export writes into `temp_unit`, and a number read
        off the clock has no temperature unit -- the hold rows of
        `export_all_segments_csv` leave that cell empty on purpose. The test
        comes first because `_unit_from_col` ends with `"c)" in cl -> "degC"`,
        which matches `Time (sec)`: such a column used to be labelled degC
        here, and with the display switch on K `_convert_x_to_display` even
        added 273.15 to the clock."""
        if _looks_like_time(x_col):
            return ""
        native = _unit_from_col(x_col)
        return self._temp_unit() if native else native

    def _display_x_label(self, x_col: str) -> str:
        """Axis label for the x column, in the unit the axis is actually in."""
        if _looks_like_time(x_col):
            # `_convert_x_to_display` converted it; say so rather than keeping
            # a label the numbers under it no longer match.
            return "Time (min)" if clock_scale_to_min(x_col)[0] != 1.0 else x_col
        if not _unit_from_col(x_col):
            return x_col
        return f"Temperature ({self._temp_unit()})"

    def _format_temp_default(self, celsius_value: float) -> str:
        """Tg-window/Calculs-range defaults are authored in °C (the
        canonical DTA unit); convert to whatever unit is currently active
        so re-loading a file while working in Kelvin doesn't reset the
        window to a physically wrong value."""
        v = _shift_temperature(celsius_value, "°C", self._temp_unit())
        return f"{v:.6g}"

    def _update_temp_switch_enabled(self, x_col: str) -> None:
        enabled = bool(_unit_from_col(x_col))
        self.temp_c_btn.setEnabled(enabled)
        self.temp_k_btn.setEnabled(enabled)

    def _convert_range_fields(self, old_unit: str, new_unit: str) -> None:
        """Keep every typed range/point field physically meaningful across
        a unit switch, converting in place rather than leaving the same
        typed number to mean something different. Only touches a group of
        fields when the X column that group actually windows is currently
        recognized as temperature -- so switching units while X is, say,
        Time never reinterprets a time window as degrees."""
        def _shift_field(edit: QLineEdit) -> None:
            v = _to_float(edit.text())
            if v is not None:
                edit.setText(f"{_shift_temperature(v, old_unit, new_unit):.6g}")

        if _unit_from_col(self.x_combo.currentText()):
            for edit in (
                self.xmin_edit, self.xmax_edit,
                self.low_min_edit, self.low_max_edit, self.low_point_edit,
                self.slope_min_edit, self.slope_max_edit,
                self.high_min_edit, self.high_max_edit, self.high_point_edit,
            ):
                _shift_field(edit)

        calc_x_col = self.calc_deriv_x_combo.currentText() if self.calc_use_deriv_check.isChecked() else self.x_combo.currentText()
        if _unit_from_col(calc_x_col):
            _shift_field(self.calc_xmin_edit)
            _shift_field(self.calc_xmax_edit)

    def _on_temp_unit_toggled(self, checked: bool) -> None:
        if not checked:
            return  # exclusive group fires toggled(False) on the outgoing button too; act once
        new_unit = self._temp_unit()
        if new_unit == self._prev_temp_unit:
            return
        old_unit = self._prev_temp_unit
        self._prev_temp_unit = new_unit
        self._convert_range_fields(old_unit, new_unit)
        # The rate series shows temperatures too -- a Value column, a y axis, a
        # fitted curve and its point labels -- and it can hold rows when no
        # record is loaded at all (a typed literature value), so this runs BEFORE
        # the `df is None` return below. NOTHING is converted here: `value_C`
        # stays exactly as it is and the table and the figure are re-rendered
        # from it, which is the whole of R4. The `hasattr` is construction order:
        # the Tg tab, which owns these two buttons, is built before the rate tab.
        if hasattr(self, "rate_table"):
            self._rate_refresh_table()
            self._rate_refit()
        if self.df is None:
            return
        # Asked BEFORE anything is dropped: the pinned round-trip test switches
        # to K with a result on screen and requires it to be recomputed, and
        # `_drop_results_in_other_units` is about to remove the very entry that
        # answers the question.
        had_results = self._current_result() is not None
        self._drop_results_in_other_units(new_unit)
        # Segment bounds are stored in °C and rendered in the active unit, so
        # the labels are stale the moment the unit changes. Rebuilt, not
        # relabelled: the combo text is produced by `segment_label`, and two
        # copies of that formatting would drift.
        self._refresh_segment_combo()
        self._update_scope_guardrails()
        if had_results:
            self._compute()
        else:
            self._refresh_plot()

    def _get_window(self) -> Tuple[float, float]:
        a = _to_float(self.xmin_edit.text())
        b = _to_float(self.xmax_edit.text())
        if a is None or b is None:
            raise ValueError("Provide a numeric Tg window.")
        return (min(a, b), max(a, b))

    def _get_xy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        if self.df is None:
            raise ValueError("No data loaded.")
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        dy_sel = self.dy_combo.currentText()
        if not x_col or not y_col:
            raise ValueError("Pick X and Y columns first.")

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        x = self._convert_x_to_display(x, x_col)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        if dy_sel and dy_sel != "(same)" and dy_sel in self.df.columns:
            y_alt = pd.to_numeric(self.df[dy_sel], errors="coerce").to_numpy(dtype=float)
            dy_col = dy_sel
        else:
            y_alt = None
            dy_col = "(same)"

        prepared = self._prepare_xy(x, y, x_col=x_col, y_alt=y_alt,
                                    invert=self.invert_y_check.isChecked())
        # A separate array even when the dY source is the Y column itself, as
        # before: the two travel into different dta_science calls and must not
        # alias.
        y_dy = prepared.y_alt if prepared.y_alt is not None else prepared.y.copy()
        return prepared.x, prepared.y, y_dy, x_col, y_col, dy_col

    def _raw_baseline_inputs(self) -> Dict[str, Any]:
        return dict(
            low_use_point=self.low_use_point_check.isChecked(),
            low_point_x=_to_float(self.low_point_edit.text()),
            low_min=_to_float(self.low_min_edit.text()),
            low_max=_to_float(self.low_max_edit.text()),
            high_use_point=self.high_use_point_check.isChecked(),
            high_point_x=_to_float(self.high_point_edit.text()),
            high_min=_to_float(self.high_min_edit.text()),
            high_max=_to_float(self.high_max_edit.text()),
            slope_min=_to_float(self.slope_min_edit.text()),
            slope_max=_to_float(self.slope_max_edit.text()),
        )

    def _resolved_baseline_params(self) -> BaselineParams:
        return resolve_baseline_params(manual_enabled=self.manual_compute_check.isChecked(), **self._raw_baseline_inputs())

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _refuse_uncomputable_scope(self, label: QLabel) -> bool:
        """True when this scope cannot carry a number; says so in `label`.

        The buttons are already disabled (G1), so reaching here means a
        programmatic call. It still must not be a dialog: the answer is
        standing information about what is selected, and a modal box over a
        redraw is how a user learns to dismiss warnings without reading them.
        """
        scope = self._scope()
        if scope.computable:
            return False
        reason = self._scope_block_reason(scope)
        label.setText(f"⚠ {reason}")
        self._set_status(reason)
        return True

    def _measure_tg(self, x: np.ndarray, y: np.ndarray, y_dy: np.ndarray, *,
                    window: Tuple[float, float], smooth_d: bool,
                    bp: BaselineParams, unit: str, x_col: str, y_col: str,
                    dy_col: str, seg: Optional[Segment] = None
                    ) -> DtaSegmentResult:
        """Run the three Tg methods on ONE prepared curve. Never raises.

        The single measurement path: the Tg tab, the per-segment export and the
        multi-file batch all come through here, so a number computed in one of
        them is the same number in the other two. Before this existed, the
        batch carried its own copy of the three calls and its own copy of the
        mode/range formatting, which is how a batch drifts from what the user
        saw on screen.

        A failing method is RECORDED, not raised (MERGED_DECISIONS M0.1). The
        measured case is real and it is the one that matters most: on the real
        cyclic run the second heating ramp has no glass transition left inside
        350-700 °C, the derivative peak lands on the 700 °C window edge and the
        HIGH baseline gets one point, so `compute_tg_parallel_improved` raises
        "Baseline range has too few points." A row that says that is useful; a
        modal dialog in the middle of a twelve-segment export is not, and a row
        that silently disappears from a later Lasocka fit is the worst of the
        three.

        `unit` is passed in rather than read from the widget because the batch
        measures another file with the workspace's settings, and a helper that
        reached back into the widget would report whatever the widget happened
        to show at the time.
        """
        xmin, xmax = float(window[0]), float(window[1])
        res = DtaSegmentResult(window=(xmin, xmax), baseline=bp, unit=unit,
                               x_col=x_col, y_col=y_col, dy_col=dy_col,
                               smooth_derivative=bool(smooth_d))
        errors: List[str] = []

        try:
            res.tg_deriv = compute_tg_derivative(
                x, y_dy, xmin, xmax, smooth_derivative=smooth_d,
                restrict_range=None)
        except Exception as exc:
            res.tg_deriv = None
            errors.append(f"|dY| max: {exc}")

        try:
            res.res_double = compute_tg_double_tangent(
                x, y, xmin, xmax, threshold=0.20, guard_frac=0.0,
                smooth_derivative=smooth_d, manual_low=bp.low_range,
                manual_slope=bp.manual_slope,
            )
        except Exception as exc:
            res.res_double = None
            errors.append(f"Double: {exc}")

        try:
            res.res_parallel = compute_tg_parallel_improved(
                x, y, xmin, xmax, smooth_derivative=smooth_d,
                manual_low_range=bp.low_range, manual_high_range=bp.high_range,
                manual_low_point=bp.low_point, manual_high_point=bp.high_point,
            )
        except Exception as exc:
            res.res_parallel = None
            errors.append(f"Parallel: {exc}")

        res.error = " | ".join(errors)
        res.spread = finite_spread(res.tg_values)
        res.degenerate = bool(np.isfinite(res.spread)
                              and res.spread < DEGENERATE_SPREAD)
        res.warnings = self._result_warnings(res, seg=seg, x=x)
        return res

    def _result_warnings(self, res: DtaSegmentResult, *,
                         seg: Optional[Segment], x: np.ndarray
                         ) -> Tuple[str, ...]:
        """Measured reasons to distrust `res`. Never a reason to refuse it.

        Three kinds, all of them facts about THIS measurement rather than
        judgements about the experiment:

        * the collapse, which is the one the app used to celebrate;
        * a window that does not sit inside the curve. It is worth its own
          sentence because the exception it eventually produces ("Baseline
          range has too few points") names the symptom and not the cause: on
          the cyclic file the 350-700 °C default falls entirely outside the
          62-212 °C tail of a cooling ramp, and the user needs to be told
          about the window, not about the baseline;
        * the segment's own findings (partial acquisition, a setpoint never
          reached, rate control lost on the way down), which belong with every
          number read off that segment and not only in the Program tab;
        * a duplicate-collapsed abscissa. `prepare_segment_xy` averages points
          that share a temperature and reports the share; this used to be an
          exception and is now a sentence, because a 1 °C/min scan written to
          two decimals is legitimately 16.7 % duplicate and refusing it left
          the file with no Tg at all. Read off `_last_prepared`, which is the
          preparation that produced the arrays `res` was measured on — the
          coupling is `_prepare_xy` being the one place arrays are made.
        """
        bits: List[str] = []
        prepared = self._last_prepared
        if prepared is not None and prepared.duplicate_fraction > 0.01:
            bits.append(
                f"{prepared.n_duplicates_collapsed} of "
                f"{prepared.n_duplicates_collapsed + len(prepared)} points "
                f"({100.0 * prepared.duplicate_fraction:.1f} %) shared an x value "
                f"and were averaged together — a coarsely rounded abscissa, or a "
                f"scan slow enough that the rounding step exceeds the change per "
                f"sample. The curve holds fewer independent points than the row "
                f"count suggests")
        if res.degenerate:
            bits.append(f"methods collapsed onto one value (spread "
                        f"{res.spread:.2e})")
        lo, hi = res.window
        finite_x = np.asarray(x, dtype=float)
        finite_x = finite_x[np.isfinite(finite_x)]
        if finite_x.size and np.isfinite(lo) and np.isfinite(hi):
            x_lo, x_hi = float(finite_x.min()), float(finite_x.max())
            if hi < x_lo or lo > x_hi:
                bits.append(f"the window {lo:.6g}..{hi:.6g} lies entirely "
                            f"outside this curve ({x_lo:.6g}..{x_hi:.6g})")
            elif lo < x_lo or hi > x_hi:
                bits.append(f"the window {lo:.6g}..{hi:.6g} is only partly "
                            f"inside this curve ({x_lo:.6g}..{x_hi:.6g})")
        if seg is not None and seg.index:
            warning = self._segment_warning_text(seg)
            if warning:
                bits.append(warning.rstrip("."))
        return tuple(bits)

    def _result_lines(self, res: DtaSegmentResult, *,
                      attribution: str = "") -> List[str]:
        """The result box's text for one stored result. One formatter.

        Shared by `_compute` and the scope change that re-displays a segment's
        earlier numbers, so a result cannot read one way when it is computed
        and another way when it is shown again.
        """
        unit = res.unit
        td, tp, tx = res.tg_values
        lines = [
            f"Y: {res.y_col}" + (f"  [{attribution}]" if attribution else ""),
            f"Double: {td:.2f} {unit}" if np.isfinite(td) else "Double: —",
            f"Parallel: {tp:.2f} {unit}" if np.isfinite(tp) else "Parallel: —",
            f"|dY| max: {tx:.2f} {unit}" if np.isfinite(tx) else "|dY| max: —",
        ]
        if res.dy_col != "(same)":
            lines.append(f"dY src: {res.dy_col}")
        if res.res_parallel is not None:
            lp = res.res_parallel.low_used
            lines.append(f"LOW: point x={lp[0]:.6g}" if res.res_parallel.low_mode == "point" else f"LOW: {lp[0]:.6g}..{lp[1]:.6g}")
            hp = res.res_parallel.high_used
            lines.append(f"HIGH: point x={hp[0]:.6g}" if res.res_parallel.high_mode == "point" else f"HIGH: {hp[0]:.6g}..{hp[1]:.6g}")

        # THE degeneracy guard (R4). `tg_agreement` is not asked the question
        # at all when the answer would be a collapse: it has no way to tell
        # "three methods agree to 0.01 °C" from "three methods returned one
        # number", and the app printing "✓ Methods agree (spread 0.00 °C)" over
        # a meaningless value is an endorsement, not a diagnostic.
        if res.degenerate:
            lines.append(degenerate_result_line(res.spread, unit))
        else:
            from dta.dta_science import tg_agreement
            agreement = tg_agreement({"Double": td, "Parallel": tp,
                                      "|dY| max": tx}, threshold=5.0)
            if agreement["agree"] is True:
                lines.append(f"✓ Methods agree (spread {agreement['spread']:.2f} {unit})")
            elif agreement["agree"] is False:
                lo, hi = agreement["extremes"]
                lines.append(f"⚠ Methods disagree: spread {agreement['spread']:.2f} {unit} ({lo} vs {hi}) — inspect the baselines")

        for warning in res.warnings:
            if res.degenerate and warning.startswith("methods collapsed"):
                continue        # already said, in full, above
            lines.append("⚠ " + warning)
        if res.error:
            lines.append("⚠ " + res.error)
        return lines

    def _compute(self) -> None:
        if self._refuse_uncomputable_scope(self.result_label):
            return
        try:
            x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
            window = self._get_window()
        except Exception as exc:
            # Only the INPUTS can reach this branch now: an unreadable column
            # pair or a non-numeric window. Those are typing errors and a
            # dialog is the right answer to them. A method that fails on the
            # data is recorded in the result instead (`_measure_tg`).
            QMessageBox.critical(self, "Tg computation failed", str(exc))
            return

        res = self._measure_tg(
            x, y, y_dy, window=window,
            smooth_d=self._smooth_derivative_enabled(),
            bp=self._resolved_baseline_params(),
            unit=self._result_unit_label(x_col),
            x_col=x_col, y_col=y_col, dy_col=dy_col,
            seg=self._current_segment(),
        )
        self._store_result(res)
        self.result_label.setText(
            "\n".join(self._result_lines(res,
                                         attribution=self._scope_attribution())))
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Per-segment results, and the three legacy aliases onto them
    # ------------------------------------------------------------------

    def _result_key(self) -> Optional[int]:
        """Key in `seg_results` the current scope's numbers belong to.

        `Segment.index` for a real segment, `0` for a record that could not be
        segmented at all (the same index `_whole_record_segment` uses, and real
        segments are 1-based so the two can never collide), and None for an
        aggregate scope, where nothing is computable and so nothing may be
        stored.
        """
        scope = self._scope()
        seg = scope.segment
        if seg is not None:
            return int(seg.index)
        if not scope.segments:
            return 0
        return None

    def _current_result(self) -> Optional[DtaSegmentResult]:
        """The stored result for the current scope, or None."""
        key = self._result_key()
        return None if key is None else self.seg_results.get(key)

    def _store_result(self, res: Optional[DtaSegmentResult]) -> None:
        """Keep `res` as the current scope's result, or drop what is stored.

        Keyed by segment so that arrowing through a cyclic run shows each
        segment its own numbers instead of the last segment's numbers over the
        new segment's curve -- and so the per-segment export has something to
        report that the user has actually seen.
        """
        key = self._result_key()
        if key is None:
            return
        if res is None:
            self.seg_results.pop(key, None)
        else:
            self.seg_results[key] = res

    def _alias(self, field: str) -> Any:
        """Read one field of the current scope's result. See the properties."""
        res = self._current_result()
        return None if res is None else getattr(res, field)

    def _set_alias(self, field: str, value: Any) -> None:
        """Write one field of the current scope's result.

        Setting None on a scope that has no stored result is a no-op rather
        than an empty entry: `_forget_results` and `__init__` both assign None,
        and a dict full of blank `DtaSegmentResult`s would make
        `seg_results[i]` stop meaning "segment i has been measured".
        """
        key = self._result_key()
        if key is None:
            return
        res = self.seg_results.get(key)
        if res is None:
            if value is None:
                return
            res = DtaSegmentResult()
            self.seg_results[key] = res
        setattr(res, field, value)

    @property
    def res_double(self) -> Optional[TgDoubleTangentResult]:
        """Double-tangent result for the CURRENT segment.

        A property, not an attribute: `seg_results` is the only storage, and
        six of the pinned Qt tests plus `_draw_overlays`, `_current_result_row`
        and the °C/K switch read this name. Two stores for one number is how a
        per-segment result and a "current" result drift apart, and the symptom
        would be segment 1's Tg drawn over segment 7's curve -- exactly the
        confusion this whole rework exists to remove.
        """
        return self._alias("res_double")

    @res_double.setter
    def res_double(self, value: Optional[TgDoubleTangentResult]) -> None:
        self._set_alias("res_double", value)

    @property
    def res_parallel(self) -> Optional[TgParallelTangentResult]:
        """Parallel-tangent result for the current segment. See `res_double`."""
        return self._alias("res_parallel")

    @res_parallel.setter
    def res_parallel(self, value: Optional[TgParallelTangentResult]) -> None:
        self._set_alias("res_parallel", value)

    @property
    def tg_deriv(self) -> Optional[float]:
        """|dY| max Tg for the current segment. See `res_double`."""
        return self._alias("tg_deriv")

    @tg_deriv.setter
    def tg_deriv(self, value: Optional[float]) -> None:
        self._set_alias("tg_deriv", value)

    def _forget_results(self) -> None:
        """Drop the CURRENT scope's Tg results without redrawing.

        Split out of `_clear_results` for the scope change, which must not
        draw: it queues a debounced render instead, and a synchronous redraw
        here would defeat the coalescing that keeps a held-down arrow key from
        re-rendering a 19k-point file once per list entry (R5).

        Only the current scope, because the whole point of keying results by
        segment is that measuring segment 4 does not throw away the number
        already read off segment 1. `_forget_all_results` is the other one.
        """
        self._store_result(None)
        self.result_label.setText("Tg: —")

    def _forget_all_results(self) -> None:
        """Drop every segment's results. For a re-detection or a unit change.

        A re-detection moves the boundaries, so a Tg measured on the old ones
        is not a measurement on the new ones -- for any segment, not just the
        one on screen.
        """
        self.seg_results = {}
        self.result_label.setText("Tg: —")

    def _show_results_for_scope(self) -> None:
        """Re-display whatever the newly selected scope has already measured.

        The alternative -- forgetting on every scope change -- was step 7's
        conservative placeholder, and it makes a twelve-segment file
        unworkable: the user computes segment 1, looks at segment 3, comes back
        and the number is gone. Nothing is recomputed here; a stored result is
        only ever shown next to the curve it was measured on, because it is
        stored under that segment's index.
        """
        res = self._current_result()
        if res is None:
            self.result_label.setText("Tg: —")
            return
        self.result_label.setText(
            "\n".join(self._result_lines(res,
                                         attribution=self._scope_attribution())))

    def _drop_results_in_other_units(self, unit: str) -> None:
        """Forget every stored result whose numbers are in another unit.

        A stored Tg is a number in the unit it was computed in. Converting nine
        frozen `dta_science` result objects field by field -- tangent slopes,
        intercepts, baseline endpoints -- would be a dozen chances to convert
        one field twice or not at all, and the numbers are cheap to recompute.
        So the °C/K switch recomputes the segment on screen (which is what the
        pinned round-trip test asserts) and forgets the rest rather than keep
        °C numbers under a K label.

        A result whose `unit` is empty was not read off a temperature axis at
        all (minutes, on a hold) and is left alone: the switch does not touch
        those numbers either.
        """
        for key in [k for k, res in self.seg_results.items()
                    if res.unit and res.unit != unit]:
            self.seg_results.pop(key, None)

    def _clear_results(self) -> None:
        self._forget_results()
        self._set_status("Cleared results.")
        if self.df is not None:
            self._refresh_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        """Draw now.

        A direct caller -- a checkbox, a column change, a finished compute --
        supersedes any queued segment redraw, so the pending one is dropped
        first. Without that, a render queued 100 ms ago by the scope combo
        fires after this one and paints the previous scope over it
        (`PlotWidget.cancel_pending`'s own docstring).
        """
        if self.df is None:
            return
        self.plot.cancel_pending()
        self._render()

    def _teardown_ax2(self) -> None:
        """Drop the secondary axis before a redraw. Never raises.

        Matplotlib keeps a removed axes alive in the figure's child list if the
        removal throws, and a second `twinx()` on top of a stale one silently
        stacks axes that never go away.
        """
        if self.ax2 is None:
            return
        try:
            self.ax2.remove()
        except Exception:
            pass
        self.ax2 = None

    def _render(self) -> None:
        """Draw the current scope. The single entry point for every redraw.

        Three modes, because three genuinely different things are being shown
        and pretending otherwise is what produced the whole-file collapse:

          * one segment (or an unsegmented record) -> the analysis view, the
            one that existed before this rework, unchanged in substance;
          * the whole multi-segment run -> the NAVIGATION view: temperature
            against time with every segment shaded, because the first thing to
            know about a cyclic file is its shape;
          * all heating / all cooling -> one prepared curve per segment,
            overlaid for comparison, each sorted within itself and none across.

        Zoom is preserved per (record, scope): panning around a peak and then
        re-rendering for any reason must not throw the view away, but moving to
        another segment must, since the axes now mean something else.
        """
        if self.df is None:
            return
        scope = self._scope()
        self.plot.preserve_zoom((self._record_id, self._scope_key))
        self._teardown_ax2()
        ax = self.plot.ax
        ax.clear()
        try:
            if scope.is_single or not scope.segments:
                self._draw_single_segment(scope)
            elif scope.kind == SCOPE_ALL_TIME:
                self._draw_whole_run(scope)
            else:
                self._draw_multi_segment(scope)
        except SegmentScopeError as exc:
            # The structural guardrail reached the renderer. That is a bug in
            # whichever path chose this scope, but the user gets the reason on
            # the axes rather than an empty window or a traceback.
            self._draw_scope_message(str(exc))
        except Exception as exc:                       # pragma: no cover - defensive
            self._draw_scope_message(f"Could not draw this record: {exc}")
        self.plot.figure.tight_layout()
        self.plot.restore_zoom(ax)
        self.plot.canvas.draw_idle()

    def _draw_scope_message(self, message: str) -> None:
        """Put a plain-language reason on empty axes instead of nothing."""
        self._x = self._y = None
        ax = self.plot.ax
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                va="center", fontsize=9, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_single_segment(self, scope: DtaScope) -> None:
        """The analysis view: the user's Y against the user's X, one curve.

        Identical in substance to the pre-rework `_refresh_plot` — the arrays
        come from `_get_xy`, which is where the segment scoping now lives, so
        on a single-ramp file this draws exactly what it always drew.
        """
        try:
            x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        except Exception as exc:
            # Drop the cached arrays rather than keep the previous record's:
            # `_draw_overlays` and the Calculs artifacts read them, and arrays
            # that no longer describe what is on screen are worse than none.
            # The reason goes on the axes for the same reason -- an empty
            # canvas with no explanation reads as "the data is bad".
            self._draw_scope_message(str(exc))
            return
        self._x, self._y = x, y
        self._update_temp_switch_enabled(x_col)

        ax = self.plot.ax
        ax.plot(x, y, linewidth=2.0)
        ax.set_xlabel(self._display_x_label(x_col))
        ax.set_ylabel(y_col + (" (inv)" if self.invert_y_check.isChecked() else ""))
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        ax.set_title(sample)
        ax.grid(True, alpha=0.25)

        try:
            xmin, xmax = self._get_window()
            ax.axvspan(xmin, xmax, alpha=0.04)
        except Exception:
            pass

        if self.show_deriv_check.isChecked():
            try:
                xmin, xmax = self._get_window()
                w = (x >= xmin) & (x <= xmax)
                if w.sum() >= 20:
                    dy = compute_derivative(y_dy[w], x[w])
                    if self._smooth_derivative_enabled():
                        dy = moving_average_10(dy)
                    self.ax2 = ax.twinx()
                    self.ax2.plot(x[w], dy, linewidth=1.2, alpha=0.85)
                    self.ax2.set_ylabel(f"dY/d{x_col}")
                    self.ax2.grid(False)
            except Exception:
                pass

        self._draw_overlays(dy_col=dy_col)
        self._draw_segment_watermark(scope)

    def _draw_segment_watermark(self, scope: DtaScope) -> None:
        """Stamp WHICH segment this is onto the figure itself.

        Only on a multi-segment record, where the question can arise. A saved
        PNG of segment #3 of a four-cycle run is indistinguishable from a PNG
        of the whole file unless the figure says so, and a figure that has left
        the app is exactly where a number loses its provenance. The segment's
        own warnings ride along for the same reason: they must travel with the
        picture, not stay behind in the UI.
        """
        seg = scope.segment
        run = self.run
        if seg is None or run is None or len(run.segments) <= 1:
            return
        bits = [f"#{seg.index} {seg.kind}"]
        if seg.cycle_index:
            bits.append(f"cycle {seg.cycle_index}")
        rate = _seg_rate(seg)
        if seg.is_ramp and np.isfinite(rate):
            bits.append(f"{rate:+.1f} {self._temp_unit()}/min")
        text = "  ".join(bits)
        warnings = []
        if seg.is_partial:
            warnings.append("partial")
        if np.isfinite(seg.target_gap_C) and seg.target_gap_C > 5.0:
            warnings.append(f"target missed by {seg.target_gap_C:.0f} °C")
        if warnings:
            text += "\n⚠ " + "; ".join(warnings)
        self.plot.ax.text(
            0.99, 0.02, text, transform=self.plot.ax.transAxes, ha="right",
            va="bottom", fontsize=8, alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.12),
        )

    def _draw_whole_run(self, scope: DtaScope) -> None:
        """Navigation view: furnace temperature against time, segments shaded.

        This is the answer to "what did this file actually do". The signal is
        drawn behind it on a secondary axis so the record is still recognisable
        as data, but the primary axis is the thermal program, because the
        decision the user makes from this view is which segment to analyse.

        No `self._x`/`self._y` are published: there is no single prepared curve
        here, and leaving the previous scope's arrays behind would let
        `_draw_overlays` mark a Tg from another segment onto this picture.
        """
        run = self.run
        if run is None:
            return
        self._x = self._y = None
        ax = self.plot.ax
        unit = self._temp_unit()
        t = np.asarray(run.time_min, dtype=float)
        T = _shift_temperature(np.asarray(run.temp_C, dtype=float), "°C", unit)

        for seg in run.segments:
            # Alternating tones by kind, so the H/C/H/C rhythm of a cyclic run
            # is visible at a glance and a hold reads as a pause, not a gap.
            alpha = {KIND_HEAT: 0.10, KIND_COOL: 0.05}.get(seg.kind, 0.16)
            ax.axvspan(float(seg.t_start), float(seg.t_end), alpha=alpha, zorder=0)
            ax.text(0.5 * (float(seg.t_start) + float(seg.t_end)), 0.99,
                    f"#{seg.index}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, alpha=0.8)

        ax.plot(t, T, linewidth=1.6)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"Temperature ({unit})")
        ax.grid(True, alpha=0.25)
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        ax.set_title(sample)

        y_col = self.y_combo.currentText()
        if y_col and self.df is not None and y_col in self.df.columns:
            try:
                signal = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
                rows = np.asarray(run.keep_index, dtype=np.intp)
                if signal.size >= int(run.n_rows):
                    signal = signal[rows]
                if signal.size == t.size:
                    if self.invert_y_check.isChecked():
                        signal = -signal
                    self.ax2 = ax.twinx()
                    self.ax2.plot(t, signal, linewidth=0.9, alpha=0.45)
                    self.ax2.set_ylabel(y_col)
                    self.ax2.grid(False)
            except Exception:
                self._teardown_ax2()

        ax.text(0.01, 0.98, scope.label, transform=ax.transAxes, va="top",
                ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.12))

    def _draw_multi_segment(self, scope: DtaScope) -> None:
        """Comparison view: one prepared curve per segment, overlaid.

        Each segment goes through `prepare_segment_xy` on its OWN scope, so
        every curve is sorted within itself and none is sorted across — which
        is the whole difference between this picture and the global `argsort`
        that used to collapse four heating ramps into one meaningless line.

        A segment that cannot be prepared on the current abscissa (a hold under
        a temperature X, say) is skipped and NAMED under the axes rather than
        silently dropped: a missing curve with no explanation is how a user
        concludes the data is bad.
        """
        run = self.run
        if run is None:
            return
        self._x = self._y = None
        ax = self.plot.ax
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        if not x_col or not y_col or self.df is None:
            self._draw_scope_message("Pick X and Y columns first.")
            return

        x_all = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        x_all = self._convert_x_to_display(x_all, x_col)
        y_all = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        invert = self.invert_y_check.isChecked()

        drawn, skipped = 0, []
        for seg in scope.segments:
            try:
                prepared = self._prepare_xy(x_all, y_all, x_col=x_col, invert=invert,
                                            scope=_single_segment_scope(seg))
            except SegmentScopeError as exc:
                skipped.append(f"#{seg.index}: {exc}")
                continue
            rate = _seg_rate(seg)
            label = f"#{seg.index} {seg.kind}"
            if np.isfinite(rate):
                label += f" {rate:+.1f}"
            if seg.cycle_index:
                label += f" (cyc {seg.cycle_index})"
            ax.plot(prepared.x, prepared.y, linewidth=1.4, label=label)
            drawn += 1

        ax.set_xlabel(self._display_x_label(x_col))
        ax.set_ylabel(y_col + (" (inv)" if invert else ""))
        ax.grid(True, alpha=0.25)
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        ax.set_title(sample)
        if drawn:
            ax.legend(fontsize=7, loc="best")
        note = scope.label
        if skipped:
            note += "\n" + "\n".join(f"not drawn — {s}" for s in skipped[:3])
        ax.text(0.01, 0.98, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", alpha=0.12))
        if not drawn:
            self._draw_scope_message(
                "No segment in this view can be drawn against "
                f"{x_col!r}.\n" + ("\n".join(skipped[:3]) if skipped else ""))

    def _draw_overlays(self, dy_col: str) -> None:
        if self._x is None or self._y is None:
            return
        x, y = self._x, self._y
        ax = self.plot.ax
        explicit = self.explicit_check.isChecked()
        unit = self._result_unit_label(self.x_combo.currentText())
        method = ("double", "parallel", "deriv")[self.method_group.checkedId()] if self.method_group.checkedId() >= 0 else "parallel"

        td = self.res_double.tg if self.res_double is not None else float("nan")
        tp = self.res_parallel.tg if self.res_parallel is not None else float("nan")
        tx = self.tg_deriv if self.tg_deriv is not None else float("nan")
        info_lines = [
            f"Double: {td:.2f}{unit}" if np.isfinite(td) else "Double: —",
            f"Parallel: {tp:.2f}{unit}" if np.isfinite(tp) else "Parallel: —",
            f"|dY| max: {tx:.2f}{unit}" if np.isfinite(tx) else "|dY| max: —",
        ]
        if dy_col != "(same)":
            info_lines.append(f"dY src: {dy_col}")
        res = self._current_result()
        if res is not None and res.degenerate:
            # Into the FIGURE, not only the result box: a PNG is what leaves
            # the building, and three identical numbers in a corner of it look
            # like a result that agrees with itself unless the picture says
            # otherwise (MERGED_DECISIONS M0.1 -- the warning travels with the
            # number).
            info_lines.append(f"⚠ collapsed, not agreed (spread {res.spread:.1e})")

        def draw_tg_marker(tg: float):
            if not np.isfinite(tg):
                return
            ax.axvline(tg, linestyle="--", linewidth=2.0)
            yy = _interp_ascending(tg, x, y)
            if np.isfinite(yy):
                ax.plot([tg], [yy], marker="o")

        try:
            xmin, xmax = self._get_window()
        except Exception:
            # nanmin/nanmax, not x[0]/x[-1]: the view is descending whenever
            # the abscissa runs backwards, and first/last would then hand
            # axvspan and the tangent lines an INVERTED span. A bug this
            # rework exposes rather than one it adds -- before segmenting, the
            # global argsort made every view ascending by accident.
            #
            # With nothing finite to span there is also nothing to annotate,
            # so the whole overlay is skipped. `float(x[0])` raised IndexError
            # on an empty array here and took the redraw down with it.
            if x.size == 0 or not np.isfinite(x).any():
                return
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        xx = np.array([xmin, xmax], dtype=float)

        if method == "double" and self.res_double is not None:
            r = self.res_double
            if explicit:
                ax.plot(xx, _line_y(r.m_low, r.b_low, xx), linestyle="--", linewidth=1.3)
                ax.plot(xx, _line_y(r.m_slope, r.b_slope, xx), linestyle="-.", linewidth=1.3)
                ax.axvline(r.x_left, alpha=0.15, linestyle=":")
                ax.axvline(r.x_right, alpha=0.15, linestyle=":")
            draw_tg_marker(r.tg)
        elif method == "parallel" and self.res_parallel is not None:
            r = self.res_parallel
            if explicit:
                ax.plot(xx, _line_y(r.m_par, r.b_low, xx), linestyle="--", linewidth=1.2)
                ax.plot(xx, _line_y(r.m_par, r.b_high, xx), linestyle="--", linewidth=1.2)
                ax.plot(xx, _line_y(r.m_par, r.b_mid, xx), linestyle="-", linewidth=1.3)
                ax.axvline(r.x_left, alpha=0.15, linestyle=":")
                ax.axvline(r.x_right, alpha=0.15, linestyle=":")
            draw_tg_marker(r.tg)
        elif method == "deriv" and np.isfinite(tx):
            draw_tg_marker(tx)

        ax.text(
            0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
        )

    # ------------------------------------------------------------------
    # Calculs
    # ------------------------------------------------------------------

    def _get_calc_range(self) -> Tuple[float, float]:
        a = _to_float(self.calc_xmin_edit.text())
        b = _to_float(self.calc_xmax_edit.text())
        if a is None or b is None:
            raise ValueError("Provide numeric Xmin / Xmax for Calculs.")
        return (min(a, b), max(a, b))

    def _get_calc_y(self) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.df is None:
            raise ValueError("No data loaded.")
        y_col = self.calc_y_combo.currentText() or self.y_combo.currentText()
        if not y_col:
            raise ValueError("Select a Y for Calculs.")
        use_deriv = self.calc_use_deriv_check.isChecked()
        x_col = self.calc_deriv_x_combo.currentText() if use_deriv else self.x_combo.currentText()
        x_col = x_col or self.x_combo.currentText()

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        x = self._convert_x_to_display(x, x_col)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        prepared = self._prepare_xy(x, y, x_col=x_col,
                                    invert=self.invert_y_check.isChecked())
        x, y = prepared.x, prepared.y

        if use_deriv:
            if x.size < 3:
                raise ValueError("Not enough points for derivative.")
            # No dedup here any more: `prepare_segment_xy` collapses duplicate
            # x unconditionally and guarantees a strictly increasing abscissa,
            # so np.gradient cannot divide by a zero step. The old dedup lived
            # in this branch alone and kept the FIRST y of a duplicated pair,
            # which silently discarded half of a doubled measurement.
            return x, np.gradient(y, x), f"d({y_col})/d({x_col})"
        return x, y, y_col

    def _calc_x_col(self) -> str:
        """The column the Calculs abscissa is read from. Mirrors `_get_calc_y`."""
        use_deriv = self.calc_use_deriv_check.isChecked()
        x_col = (self.calc_deriv_x_combo.currentText() if use_deriv
                 else self.x_combo.currentText())
        return x_col or self.x_combo.currentText()

    def _calc_x_unit(self) -> str:
        """Unit of the Calculs limits, as a suffix to print beside them.

        The limits used to be printed bare, and the segment scope silently
        switches this axis between °C and MINUTES: a hold forces the time
        abscissa (`_single_segment_scope`) while the Xmin/Xmax boxes keep
        whatever numbers were in them. So "from 350 to 700 = 314.304" was a
        mW·°C integral on a ramp and a mW·min one on a hold, in the same
        sentence, and six of the twelve segments of the real cyclic file have
        time spans overlapping 350-700 — which is how a hold returned 79.9358
        with no complaint at all.
        """
        x_col = self._calc_x_col()
        if _looks_like_time(x_col):
            # Always minutes: `_convert_x_to_display` has converted the column.
            return "min"
        if _unit_from_col(x_col):
            return self._temp_unit()
        return ""

    def _calc_range_note(self, x, xmin: float, xmax: float, n_inside: int) -> str:
        """Why this window holds too few points to work with, in one sentence.

        The Tg tab answers this condition inline, with the curve's own range in
        it ("the window 350..700 lies entirely outside this curve
        (64.7167..84.025)"). The Calculs tab raised `ValueError("Range too
        small for integration.")` into `QMessageBox.critical` — a modal dialog
        about a fact concerning the data in the window, naming neither the
        curve's range nor the fact that the axis had become minutes. That is
        MERGED_DECISIONS M0.1's warn-never-block rule, and the dialog was
        reachable from three of the fifteen scope entries of the real cyclic
        file with the default 350-700 boxes untouched.
        """
        unit = self._calc_x_unit()
        suffix = f" {unit}" if unit else ""
        finite = np.asarray(x, dtype=float)
        finite = finite[np.isfinite(finite)]
        where = ""
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if xmax < lo or xmin > hi:
                where = (f"; it lies entirely outside this curve "
                         f"({lo:.6g}..{hi:.6g}{suffix})")
            else:
                where = f"; this curve covers {lo:.6g}..{hi:.6g}{suffix}"
        return (f"⚠ {xmin:.6g}..{xmax:.6g}{suffix} holds {n_inside} point(s) of "
                f"this scope, and 2 are needed{where}. "
                f"The abscissa here is {self._calc_x_col()}.")

    def _calc_axis(self):
        """Secondary axis when Calculs Y differs from Tg Y, or derivative is used."""
        use_deriv = self.calc_use_deriv_check.isChecked()
        tg_y = self.y_combo.currentText()
        calc_y = self.calc_y_combo.currentText()
        is_alt = bool(calc_y and tg_y and calc_y != tg_y)
        if not use_deriv and not is_alt:
            return self.plot.ax
        if self.ax2 is None:
            self.ax2 = self.plot.ax.twinx()
        if is_alt:
            self.ax2.spines["right"].set_color("red")
            self.ax2.tick_params(axis="y", colors="red")
        self.ax2.set_ylabel("Calculs" if is_alt else "Derivative")
        return self.ax2

    def _calc_clear(self) -> None:
        self._refresh_plot()
        self.calc_result_label.setText("")
        self._set_status("Calculs cleared.")

    def _calc_integrate(self) -> None:
        # Cleared FIRST, always. The label is the only place this tab's number
        # appears, nothing else ever reset it, and the segment scope can change
        # under it: after a refused integral on segment #2 the panel still read
        # "= 314.304  [seg #1 heat, cycle 1]" while the combo and the plot
        # showed #2. A stale number with another segment's attribution beside
        # it is the one failure shape a user cannot detect.
        self.calc_result_label.setText("")
        if self._refuse_uncomputable_scope(self.calc_result_label):
            return
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if int(w.sum()) < 2:
                self.calc_result_label.setText(self._with_attribution(
                    self._calc_range_note(x, xmin, xmax, int(w.sum()))))
                self._set_status("Calculs: the range holds too few points.")
                return
            area = float(trapezoid(y[w], x[w]))
            unit = self._calc_x_unit()
            span = (f"from {xmin:.4g} to {xmax:.4g} {unit}" if unit
                    else f"from {xmin:.4g} to {xmax:.4g}")
            self.calc_result_label.setText(self._with_attribution(
                f"Integrate ({y_col}) {span} = {area:.6g}"))
            ax = self._calc_axis()
            self.plot.ax.axvspan(xmin, xmax, alpha=0.08)
            ax.fill_between(x[w], y[w], 0.0, alpha=0.12)
            self.plot.ax.text(0.01, 0.01, f"Intg = {area:.6g}", transform=self.plot.ax.transAxes, va="bottom", ha="left")
            self.plot.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, "Integrate failed", str(exc))

    def _calc_find_extremum(self, mode: str) -> None:
        self.calc_result_label.setText("")       # see `_calc_integrate`
        if self._refuse_uncomputable_scope(self.calc_result_label):
            return
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if int(w.sum()) < 2:
                self.calc_result_label.setText(self._with_attribution(
                    self._calc_range_note(x, xmin, xmax, int(w.sum()))))
                self._set_status("Calculs: the range holds too few points.")
                return
            xw, yw = x[w], y[w]
            j = int(np.nanargmax(yw)) if mode == "max" else int(np.nanargmin(yw))
            xv, yv = float(xw[j]), float(yw[j])
            label = "Max" if mode == "max" else "Min"
            unit = self._calc_x_unit()
            suffix = f" {unit}" if unit else ""
            self.calc_result_label.setText(self._with_attribution(
                f"{label} ({y_col}) in [{xmin:.4g}, {xmax:.4g}]{suffix} = {yv:.6g} "
                f"at x={xv:.6g}{suffix}"))
            ax = self._calc_axis()
            self.plot.ax.axvspan(xmin, xmax, alpha=0.06)
            ax.plot([xv], [yv], marker="o")
            ax.axvline(xv, linestyle="--", linewidth=1.5)
            self.plot.ax.text(0.01, 0.01, f"{label} = {yv:.6g} @ x={xv:.6g}", transform=self.plot.ax.transAxes, va="bottom", ha="left")
            self.plot.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, f"Find {mode.title()} failed", str(exc))

    def _calc_find_max(self) -> None:
        self._calc_find_extremum("max")

    def _calc_find_min(self) -> None:
        self._calc_find_extremum("min")

    # ------------------------------------------------------------------
    # Export / batch
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_range(rng: Optional[Tuple[float, float]]) -> str:
        """`'340..348'` for a baseline range, blank when there is none."""
        return "" if rng is None else f"{rng[0]:.6g}..{rng[1]:.6g}"

    @staticmethod
    def _fmt_export(value: Any) -> Any:
        """A cell for one of the APPENDED columns. Missing is blank.

        Never `nan` and never 0: the existing export writes the string `nan`
        into an empty Tg cell, which a spreadsheet shows as text and a reader
        takes for a broken number, and a 0 in a `target_gap_C` column reads as
        "hit the target exactly". The legacy 18 keep their historic float NaN,
        because changing them is what backward compatibility forbids.
        """
        if value is None:
            return ""
        if isinstance(value, bool):                # before int: bool IS an int
            return "yes" if value else "no"
        if isinstance(value, float):
            return "" if not np.isfinite(value) else value
        return value

    def _segment_result_row(self, seg: Optional[Segment],
                            res: Optional[DtaSegmentResult], *,
                            run: Optional[SegmentedRun] = None,
                            path: Optional[Path] = None,
                            sample: str = "") -> Dict[str, Any]:
        """One CSV row: the settings, the three Tg values, and where they came from.

        Keys are `SEGMENT_RESULT_COLUMNS` in order -- the 18 pre-segment columns
        in their original positions, then the provenance. A consumer that reads
        by name keeps working; a consumer that reads by position keeps working
        for the columns it knows about.

        `run` is needed for the two row indices and nothing else: `row_start` /
        `row_stop` are ORIGINAL DataFrame positions via
        `SegmentedRun.row_indices`, never `Segment.row_start`, which indexes the
        cleaned arrays. On every file shipped here the two coincide (no
        non-finite rows), which is exactly why the rule has to be in the code
        and not in the reader's memory (SPEC_cycles R2).

        `res` may be None -- a segment nobody has measured yet -- and then the
        three Tg cells are blank rather than 0, and every setting cell is blank
        too, because no settings produced it.
        """
        bp = res.baseline if (res is not None and res.baseline is not None) else None
        td, tp, tx = res.tg_values if res is not None else (NAN, NAN, NAN)
        if res is not None and res.res_parallel is not None:
            low_mode = res.res_parallel.low_mode
            high_mode = res.res_parallel.high_mode
        elif bp is not None:
            low_mode, high_mode = _baseline_modes(bp)
        else:
            low_mode = high_mode = ""

        window = res.window if res is not None else (NAN, NAN)
        row: Dict[str, Any] = dict(
            file=str(path) if path else "",
            sample=sample,
            x_col=(res.x_col if res is not None else ""),
            y_col=(res.y_col if res is not None else ""),
            dy_source=("" if res is None else
                       (res.y_col if res.dy_col == "(same)" else res.dy_col)),
            temp_unit=(res.unit if res is not None else ""),
            window_min=float(window[0]), window_max=float(window[1]),
            smooth_derivative=("" if res is None else int(res.smooth_derivative)),
            low_mode=low_mode,
            low_range=self._fmt_range(bp.low_range if bp is not None else None),
            low_point_x=("" if bp is None or bp.low_point is None else float(bp.low_point)),
            high_mode=high_mode,
            high_range=self._fmt_range(bp.high_range if bp is not None else None),
            high_point_x=("" if bp is None or bp.high_point is None else float(bp.high_point)),
            Tg_double=td, Tg_parallel=tp, Tg_derivative=tx,
        )

        row_start = row_stop = None
        if seg is not None and run is not None:
            try:
                rows = run.row_indices(seg)
                if rows.size:
                    row_start, row_stop = int(rows[0]), int(rows[-1]) + 1
            except Exception:
                row_start = row_stop = None

        # A stand-in whole-record segment carries index 0 and no measurements;
        # reporting its NaN rates as a segment's would invent a segment the
        # record does not have.
        real = seg is not None and int(seg.index) > 0
        row.update({
            "segment_index": self._fmt_export(int(seg.index) if real else None),
            "segment_kind": (seg.kind if real else ""),
            "segment_role": (seg.role if real else ""),
            "cycle_index": self._fmt_export(int(seg.cycle_index) if real else None),
            "segment_label": (segment_label(seg) if real else ""),
            "program_step": (";".join(str(i) for i in seg.step_indices) if real else ""),
            "row_start": self._fmt_export(row_start),
            "row_stop": self._fmt_export(row_stop),
            "n_points": self._fmt_export(int(seg.n_points) if real else None),
            "t_start_min": self._fmt_export(float(seg.t_start) if real else None),
            "t_end_min": self._fmt_export(float(seg.t_end) if real else None),
            "T_start_C": self._fmt_export(float(seg.T_start) if real else None),
            "T_end_C": self._fmt_export(float(seg.T_end) if real else None),
            "rate_measured_C_min": self._fmt_export(_seg_rate(seg) if real else None),
            "rate_programmed_C_min": self._fmt_export(
                float(seg.rate_nominal_C_min) if real else None),
            "target_gap_C": self._fmt_export(float(seg.target_gap_C) if real else None),
            "control_lost_below_C": self._fmt_export(
                float(seg.control_lost_below_C) if real else None),
            "is_partial": self._fmt_export(bool(seg.is_partial) if real else None),
            "tg_spread": self._fmt_export(res.spread if res is not None else None),
            "tg_degenerate": self._fmt_export(bool(res.degenerate) if res is not None else None),
            "warnings": (" | ".join(res.warnings) if res is not None else ""),
            "error": (res.error if res is not None else ""),
        })
        return row

    def _current_result_row(self) -> Dict[str, Any]:
        """The row for what is on screen, in the full schema.

        Measures now rather than exporting `seg_results`, and that is the whole
        point: the user may have typed a new window since pressing Compute, and
        a row whose `window_min`/`window_max` cells describe one window while
        its `Tg_*` cells were read under another is internally inconsistent --
        wrong in a file that outlives the session, and impossible to notice.
        Measuring here makes the row self-describing.

        It goes through `_measure_tg`, the same call the Compute button makes,
        so the export can never use a different construction from the screen.
        It deliberately does not WRITE to `seg_results`: an export is a read.
        """
        x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        seg = self._current_segment()
        res = self._measure_tg(
            x, y, y_dy, window=self._get_window(),
            smooth_d=self._smooth_derivative_enabled(),
            bp=self._resolved_baseline_params(),
            unit=self._result_unit_label(x_col),
            x_col=x_col, y_col=y_col, dy_col=dy_col, seg=seg,
        )
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        return self._segment_result_row(seg, res, run=self.run, path=self.path,
                                        sample=sample)

    def _export_current_csv(self) -> None:
        """Dialog wrapper over `export_current_csv`."""
        if self.df is None:
            QMessageBox.critical(self, "Export", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result as CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.export_current_csv(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Export", str(exc))
            return
        QMessageBox.information(self, "Export", f"Saved:\n{path}")

    def export_current_csv(self, path: Path) -> Dict[str, Any]:
        """Write the current result row to `path` and return it.

        Separated from the dialog so the export can be exercised without one,
        and so the slot has somewhere to catch from: `_current_result_row`
        calls `_get_xy`, which now refuses an aggregate scope, and an exception
        escaping a Qt slot is a crash rather than a message.
        """
        row = self._current_result_row()
        self._write_rows_csv([row], path)
        return row

    @staticmethod
    def _write_rows_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
        """Write `rows` to `path` in `SEGMENT_RESULT_COLUMNS` order.

        Two rules, and both matter for a file somebody else reads:

        * the ORDER is the schema's, not the order keys happened to appear in.
          `pd.DataFrame(list_of_dicts)` orders columns by first appearance, so a
          batch whose first row is a short error row would emit a differently
          shaped CSV from the same batch with the files in another order;
        * a column no row carries is NOT invented. An all-errors batch stays
          narrow instead of emitting 40 headers over nothing, and an empty
          `Tg_parallel` cell therefore always means "measured, no value" rather
          than "never attempted". Any unexpected key is appended rather than
          dropped -- losing data to protect a column order would be the wrong
          trade.
        """
        frame = pd.DataFrame(list(rows))
        ordered = ([c for c in SEGMENT_RESULT_COLUMNS if c in frame.columns]
                   + [c for c in frame.columns if c not in SEGMENT_RESULT_COLUMNS])
        frame.reindex(columns=ordered).to_csv(path, index=False)

    def _error_row(self, *, path: Optional[Path], sample: str, error: str,
                   seg: Optional[Segment] = None) -> Dict[str, Any]:
        """A deliberately SPARSE row: no Tg columns at all.

        The distinction is the whole value of the row. An empty `Tg_parallel`
        CELL means "measured, and there is no value"; an absent `Tg_parallel`
        KEY means "never measured". The whole-file refusal on a cyclic run is
        the second kind, and `test_batch_whole_file_refuses_a_cyclic_file` pins
        it by asserting the key is not there -- a blank cell would read as a
        failed measurement of a number that was never attempted.

        The segment is named when one is known, because twelve error rows with
        no way to tell which segment each belongs to are twelve unusable rows.
        """
        row: Dict[str, Any] = {
            "file": str(path) if path else "", "sample": sample, "error": error,
        }
        if seg is not None and int(seg.index) > 0:
            row["segment_index"] = int(seg.index)
            row["segment_kind"] = seg.kind
            row["segment_label"] = segment_label(seg)
        return row

    def _x_col_for_segment(self, df: pd.DataFrame, seg: Optional[Segment],
                           x_col: str) -> Tuple[str, str]:
        """`(column to use as X, why it was changed)` for one segment.

        Two substitutions, one in each direction, and the second is what makes
        a per-segment export of a cyclic file worth running at all.

        **A hold gets the clock.** A HOLD has no temperature scan: its measured
        temperatures repeat, so `prepare_segment_xy` refuses temperature as an
        abscissa there and the UI forces the clock on screen for exactly that
        reason. An export must do the same thing rather than raise, because
        MERGED_DECISIONS M0.1 says a hold row is still produced when the user
        asks for one -- with the substitution stated in the row, not silently.

        **A ramp gets the temperature back.** A cyclic record opens on the
        whole-run view, whose only honest abscissa is time, so `x_combo` sits on
        the clock -- not because the user chose it but because an aggregate view
        has no single scan (`_x_forced_time`). Every segment being exported IS a
        single scan, and a Tg is a step in a TEMPERATURE scan: measured with the
        clock on X and the default 350-700 window, all eight ramps of the real
        cyclic file report "Tg window too small" because 350-700 MINUTES misses
        a 64-minute ramp entirely. The substitution is refused the moment the
        user picks a column themselves, which is precisely what
        `_x_forced_time` records.

        The window is NOT adjusted to match. One window for every row is what
        makes the rows comparable, and a row whose window does not fit its curve
        says so in its `warnings` cell (`_result_warnings`) instead of being
        quietly measured under different settings from its neighbours.
        """
        if seg is None:
            return x_col, ""
        t_col, T_col = _time_temp_columns(list(df.columns), self._canonical_map)
        if seg.is_hold and _is_temperature_col(x_col):
            if not t_col or t_col == x_col:
                return x_col, ""
            return t_col, (f"a hold has no temperature scan, so X is {t_col} "
                           f"here and not {x_col}")
        if (seg.is_ramp and self._x_forced_time and _looks_like_time(x_col)
                and T_col and T_col != x_col):
            return T_col, (f"the clock was on X only because the whole-run view "
                           f"has no single scan; this segment is one, so X is "
                           f"{T_col} here and not {x_col}")
        return x_col, ""

    def _row_for_segment(self, *, df: pd.DataFrame,
                         run: Optional[SegmentedRun], seg: Optional[Segment],
                         path: Optional[Path], sample: str,
                         x_col: str, y_col: str, dy_sel: str,
                         window: Tuple[float, float], smooth_d: bool,
                         invert: bool, bp: BaselineParams) -> Dict[str, Any]:
        """Measure ONE curve of ONE file and return its row. Never raises.

        `seg=None` means "this file as one curve", which is only ever reached
        for a record that holds a single segment or none -- the multi-segment
        refusal happens in `_batch_segments_for_file`, where it can be worded
        for a file that is not on screen.

        Arrays come from `_prepare_xy` with an EXPLICIT scope, the same way
        `_draw_multi_segment` prepares each overlay: the scope names the segment
        and the run supplies the `keep_index` that turns its cleaned bounds into
        DataFrame rows. Passing one file's run with another file's frame is the
        mistake this keyword-only signature makes hard to write.
        """
        scope = (_single_segment_scope(seg) if seg is not None
                 else _scope_for_run(run))
        x_used, x_note = self._x_col_for_segment(df, seg, x_col)
        try:
            x = pd.to_numeric(df[x_used], errors="coerce").to_numpy(dtype=float)
            x = self._convert_x_to_display(x, x_used)
            y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
            if dy_sel and dy_sel != "(same)":
                dy_col_resolved = _resolve_column(df, dy_sel)
            else:
                dy_col_resolved = None
            if dy_col_resolved:
                y_alt = pd.to_numeric(df[dy_col_resolved],
                                      errors="coerce").to_numpy(dtype=float)
                dy_col = dy_col_resolved
            else:
                y_alt = None
                dy_col = "(same)"
            prepared = self._prepare_xy(x, y, x_col=x_used, y_alt=y_alt,
                                        invert=invert, run=run, scope=scope)
        except Exception as exc:
            return self._error_row(path=path, sample=sample, error=str(exc),
                                   seg=(seg if seg is not None else scope.segment))

        y_dy = prepared.y_alt if prepared.y_alt is not None else prepared.y.copy()
        reported = seg if seg is not None else scope.segment
        res = self._measure_tg(
            prepared.x, prepared.y, y_dy, window=window, smooth_d=smooth_d,
            bp=bp, unit=self._result_unit_label(x_used), x_col=x_used,
            y_col=y_col, dy_col=dy_col, seg=reported)
        if x_note:
            res.warnings = (x_note,) + tuple(res.warnings)
        return self._segment_result_row(reported, res, run=run, path=path,
                                        sample=sample)

    # ---- progress reporting -------------------------------------------

    def _progress_ticker(self, label: str, total: int):
        """A `progress(done, total, text) -> bool` callback over a QProgressDialog.

        The callable halves of the export and the batch know nothing about Qt:
        they call this function and stop when it returns False. That is what
        lets a test drive cancellation with a two-line lambda instead of
        simulating a click on a dialog.

        `setValue` on a window-modal `QProgressDialog` processes events itself,
        which is what repaints the bar and lets Cancel be clicked during a
        synchronous loop. The modality is not decoration: without it the
        processed events would also deliver clicks to the combos underneath,
        and a scope change halfway through a batch would change the settings
        the remaining files are measured with.
        """
        dialog = QProgressDialog(label, "Cancel", 0, max(int(total), 1), self)
        dialog.setWindowTitle("PRISM")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(400)   # a two-file batch never flashes a box
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        self._progress_dialog = dialog

        def tick(done: int, total_now: int, text: str = "") -> bool:
            dialog.setMaximum(max(int(total_now), 1))
            dialog.setLabelText(f"{label}\n{text}" if text else label)
            dialog.setValue(int(done))
            return not dialog.wasCanceled()

        return tick

    def _close_progress(self) -> None:
        """Take down the progress dialog, whatever happened. Never raises."""
        dialog = self._progress_dialog
        self._progress_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass

    # ---- batch scope --------------------------------------------------

    def batch_scope(self) -> str:
        """Which `BATCH_SCOPE_*` the radio group is on."""
        for key, btn in self.batch_scope_buttons.items():
            if btn.isChecked():
                return key
        return BATCH_SCOPE_FILE

    def set_batch_scope(self, scope: str) -> None:
        """Choose the batch scope by key. Unknown keys are ignored.

        Public and dialog-free for the same reason every file-dialog method has
        a callable twin: a test that has to click a radio button to reach the
        batch logic is testing Qt, not the batch.
        """
        btn = self.batch_scope_buttons.get(str(scope))
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)          # exclusive group; unchecks the others
        self._refresh_batch_notes()

    def _on_batch_scope_changed(self, _button=None, _checked: bool = False) -> None:
        """Keep the standing note in step with the radio group.

        `buttonToggled` fires twice per change (off, then on), and both fire the
        same idempotent refresh -- there is nothing to debounce here because the
        note is three lines of text, not a redraw of a 19k-point record.
        """
        self._refresh_batch_notes()

    def _skip_holds(self) -> bool:
        """Whether a per-segment sweep leaves the holds out. One checkbox.

        Governs BOTH "Export all segments" and the batch's "Every segment", and
        the batch note says so on screen. Two checkboxes asking the same
        question is how the two sweeps end up disagreeing about what a file
        contains.
        """
        return bool(self.skip_holds_check.isChecked())

    def _refresh_batch_notes(self) -> None:
        """Rewrite the two standing notes for the current state. Never raises.

        Standing text, not a dialog, and not deferred to the moment the user
        presses the button: the whole-file refusal on a cyclic run is
        information about what the selected option will do, and a user who
        learns it from a message box after choosing a save path has already
        been made to work for nothing.
        """
        note = getattr(self, "export_segments_note", None)
        # `batch_scope_note` is the LAST widget this method touches and the last
        # one `_build_batch_tab` creates, so testing for it is the one check
        # that cannot pass over a half-built tab.
        if note is None or getattr(self, "batch_scope_note", None) is None:
            return
        run = self.run
        if self.df is None:
            self.export_segments_btn.setEnabled(False)
            note.setText("Load a record first.")
        elif run is None or not run.segments:
            self.export_segments_btn.setEnabled(False)
            note.setText("This record was not split into segments — no clock "
                         "and temperature pair was found. Use “Export result "
                         "CSV” above for the whole record.")
        else:
            targets, refusal = self._export_segment_targets()
            self.export_segments_btn.setEnabled(bool(targets))
            if refusal:
                note.setText(refusal)
            else:
                note.setText(
                    f"{_plural(len(targets), 'row')} — one per segment, each "
                    f"measured on its own scan with the current settings, of "
                    f"{_plural(len(run.segments), 'segment')} in this record. "
                    "A segment whose measurement fails still gets a row, with "
                    "the reason in its `error` cell.")
        self.batch_scope_note.setText(self._batch_scope_note_text())

    def _batch_scope_note_text(self) -> str:
        """Why the selected batch scope will do what it does, in plain words."""
        scope = self.batch_scope()
        if scope == BATCH_SCOPE_FILE:
            return ("One row per file, from the whole file as a single scan. A "
                    "file that holds more than one segment is REFUSED a number "
                    "and gets a row naming its segments instead: a tangent, an "
                    "integral and a derivative extremum are each defined on one "
                    "monotone scan, and one global sort over eight interleaved "
                    "ramps is not a curve. That is an arithmetic limit, not a "
                    "physical objection — the segments are all still "
                    "measurable, one at a time, with the other two options.")
        if scope == BATCH_SCOPE_EVERY:
            holds = ("Hold segments are left out (“Skip hold segments” above)."
                     if self._skip_holds() else
                     "Hold segments are included, measured against the clock.")
            return ("One row per segment of every file, each segment measured "
                    "on its own scan. A single-ramp file still yields exactly "
                    f"one row. {holds}")
        seg = self._current_segment()
        if seg is None:
            return ("Needs a single segment selected in the Tg tab: the match "
                    "is made against the scan on screen. Files are skipped "
                    "with a reason until one is selected.")
        return (f"Only segments of the same kind as {segment_label(seg)} whose "
                f"measured rate is within {MATCH_RATE_TOL_C_MIN:g} °C/min of "
                f"{_seg_rate(seg):+.2f} °C/min. The sign counts, so a cooling "
                "ramp never matches a heating one. This is the scope that "
                "builds a same-rate series across files.")

    def _segment_matches_reference(self, seg: Segment, ref: Segment) -> bool:
        """True when `seg` is "the same scan" as `ref` for batch purposes.

        Kind first, and the kind is not negotiable: a −20 °C/min cool and a
        +20 °C/min heat have the same |rate| and nothing else in common, so the
        sign is part of the identity of a rate rather than a detail of
        formatting.

        Then the measured rate, within `MATCH_RATE_TOL_C_MIN`. Measured rather
        than programmed, because a segment of a file with no readable method has
        no programmed rate at all and would silently match nothing.

        Two segments whose rate could not be estimated at all (both NaN) match
        on kind alone. That is the honest answer: refusing them would drop rows
        because of a missing diagnostic rather than because of the data, and
        `NaN <= tol` is False, so without this branch the comparison would look
        like a decision when it was an accident of IEEE arithmetic.
        """
        if seg.kind != ref.kind:
            return False
        a, b = _seg_rate(seg), _seg_rate(ref)
        if not np.isfinite(a) and not np.isfinite(b):
            return True
        return bool(abs(a - b) <= MATCH_RATE_TOL_C_MIN)

    def _batch_segments_for_file(self, run: Optional[SegmentedRun]
                                 ) -> Tuple[Tuple[Optional[Segment], ...], str]:
        """`(curves to measure, refusal)` for ONE batch file. Never raises.

        A `None` entry means "this file as one curve", which is what every
        single-ramp export has always been and must stay. An empty tuple with a
        non-empty refusal means this file contributes one error row instead of
        a measurement -- and the wording is chosen for a file that is NOT on
        screen, because the user is looking at a different record while forty
        others are being read.

        The whole-file refusal on a multi-segment file is the only refusal here
        that is about arithmetic rather than about a setting, and it is the one
        MERGED_DECISIONS M0.1 could be mistaken for forbidding. It does not:
        M0.1 forbids refusing to PROCESS data for a physical reason. There is no
        number to refuse here -- a single-valued abscissa is a precondition of
        the tangent construction, not a preference about it -- and the same file
        is fully processed by the other two scopes, which is what the refusal
        text points at. Silently returning the collapsed 431.758 °C would be
        the real failure to serve the user.
        """
        scope = self.batch_scope()
        segments = tuple(run.segments) if run is not None else ()

        if scope == BATCH_SCOPE_FILE:
            if len(segments) > 1:
                names = ", ".join(f"#{s.index} {s.kind}" for s in segments)
                return (), (
                    f"this file holds {len(segments)} segments ({names}) and "
                    "the batch scope is one row per file. A tangent, an "
                    "integral and a derivative extremum are each defined on "
                    "ONE monotone scan, and one global sort over interleaved "
                    "passes through the same temperatures is not a curve — so "
                    "there is no whole-file number to report, not merely an "
                    "unreliable one. Choose "
                    f"“{BATCH_SCOPE_LABELS[BATCH_SCOPE_EVERY]}” or "
                    f"“{BATCH_SCOPE_LABELS[BATCH_SCOPE_MATCHING]}” to measure "
                    "every segment of it separately.")
            return (None,), ""

        if scope == BATCH_SCOPE_EVERY:
            if not segments:
                # Not segmentable is not the same as multi-segment: a file with
                # no clock column is one curve and always has been.
                return (None,), ""
            kept = (tuple(s for s in segments if s.is_ramp)
                    if self._skip_holds() else segments)
            if not kept:
                return (), ("every segment of this file is a hold and “Skip "
                            "hold segments” is on; uncheck it to measure them "
                            "against the clock.")
            return kept, ""

        ref = self._current_segment()
        if ref is None:
            return (), ("“" + BATCH_SCOPE_LABELS[BATCH_SCOPE_MATCHING] + "” "
                        "needs a single segment selected in the Tg tab — the "
                        "match is made against the scan on screen, and there "
                        "is none.")
        if not segments:
            return (), (f"this file was not split into segments, so nothing in "
                        f"it can be matched against {segment_label(ref)}; it "
                        "has no clock and temperature pair to segment on.")
        # "Skip hold segments" is NOT applied here. The reference segment is an
        # explicit choice, so a user pointing at a hold is asking for holds, and
        # silently dropping them would answer a question nobody asked.
        matched = tuple(s for s in segments
                        if self._segment_matches_reference(s, ref))
        if not matched:
            found = ", ".join(f"#{s.index} {s.kind} {_seg_rate(s):+.2f}"
                              for s in segments)
            return (), (
                f"no segment of this file matches {segment_label(ref)} "
                f"(same kind, rate within {MATCH_RATE_TOL_C_MIN:g} °C/min). "
                f"It holds: {found} °C/min.")
        return matched, ""

    # ---- every segment of the current file ----------------------------

    def _export_segment_targets(self) -> Tuple[Tuple[Segment, ...], str]:
        """The segments "Export all segments" will measure, and the note for it.

        Holds are excluded when the checkbox says so, which is the default: a
        Tg is a step in a temperature scan and a hold has no temperature scan
        (its measured temperatures repeat, and `prepare_segment_xy` refuses
        temperature as an abscissa there). Unchecking it does NOT hard-block --
        M0.1 -- it exports the hold rows too, against the clock, with the
        reason in each row's `warnings` cell.
        """
        run = self.run
        if run is None or not run.segments:
            return ((), "This record was not split into segments; "
                        "use “Export result CSV” for the whole record.")
        segments = tuple(run.segments)
        if self._skip_holds():
            kept = tuple(s for s in segments if s.is_ramp)
            if not kept:
                return ((), "Every segment of this record is a hold, and "
                            "“Skip hold segments” is on. Uncheck it to export "
                            "them against the clock.")
            return kept, ""
        return segments, ""

    def export_all_segments_csv(self, path: Path, *,
                                progress: Optional[Any] = None
                                ) -> List[Dict[str, Any]]:
        """Measure every segment of the LOADED record and write one CSV.

        Dialog-free and side-effect-free: it neither moves the scope selector
        nor writes `seg_results`. Each segment is measured with an explicit
        single-segment scope, exactly the way `_draw_multi_segment` draws one,
        so measuring segment 7 cannot disturb what segment 1 shows.

        Every segment gets a row even when its measurement failed -- the row
        then carries the reason in `error`. That is the point of the file: on
        the real cyclic run the second heating ramp genuinely has no glass
        transition left inside 350-700 °C, and a row saying so is the finding.
        A row that quietly vanished would leave a four-rate Lasocka series
        looking like a three-rate one with no explanation.

        `progress(done, total, label) -> bool` is an optional callback; return
        False from it to stop early. The dialog wrapper passes a
        `QProgressDialog` through it, and a test passes a plain function --
        which is why cancellation is a return value and not a widget.
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        targets, note = self._export_segment_targets()
        if not targets:
            raise ValueError(note or "Nothing to export.")

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        dy_sel = self.dy_combo.currentText()
        window = self._get_window()
        smooth_d = self._smooth_derivative_enabled()
        invert = self.invert_y_check.isChecked()
        bp = self._resolved_baseline_params()
        sample = self.header.get("Sample", self.path.stem if self.path else "")

        rows: List[Dict[str, Any]] = []
        total = len(targets)
        for done, seg in enumerate(targets):
            if progress is not None and not progress(done, total,
                                                     segment_label(seg)):
                break
            rows.append(self._row_for_segment(
                df=self.df, run=self.run, seg=seg, path=self.path,
                sample=sample, x_col=x_col, y_col=y_col, dy_sel=dy_sel,
                window=window, smooth_d=smooth_d, invert=invert, bp=bp))
        if progress is not None:
            progress(len(rows), total, "")
        self._write_rows_csv(rows, path)
        return rows

    def _export_all_segments_csv(self) -> None:
        """Dialog wrapper over `export_all_segments_csv`."""
        if self.df is None:
            QMessageBox.critical(self, "Export segments", "No data loaded.")
            return
        targets, note = self._export_segment_targets()
        if not targets:
            QMessageBox.critical(self, "Export segments", note)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save one row per segment as CSV", "", "CSV (*.csv)")
        if not path:
            return
        tick = self._progress_ticker("Measuring each segment…", len(targets))
        try:
            rows = self.export_all_segments_csv(Path(path), progress=tick)
        except Exception as exc:
            QMessageBox.critical(self, "Export segments", str(exc))
            return
        finally:
            self._close_progress()
        failed = sum(1 for r in rows if r.get("error"))
        self._set_status(f"Exported {_plural(len(rows), 'segment')} to {path}"
                         + (f"; {failed} carried an error." if failed else "."))
        QMessageBox.information(self, "Export segments",
                                f"{_plural(len(rows), 'row')} saved:\n{path}")

    def _save_figure_png(self) -> None:
        """Dialog wrapper over `save_figure_png`."""
        if self.df is None:
            QMessageBox.critical(self, "Save figure", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save figure as PNG", "", "PNG (*.png)")
        if not path:
            return
        try:
            self.save_figure_png(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save figure", str(exc))
            return
        QMessageBox.information(self, "Save figure", f"Saved:\n{path}")

    def save_figure_png(self, path: Path) -> Path:
        """Write the current figure to `path` and return it.

        Flushes any queued redraw FIRST. A render queued by the scope combo up
        to 120 ms ago has not run yet, so without this the PNG can show the
        previous segment while the app on screen shows the new one -- and the
        file, unlike the screen, is what leaves the building.
        """
        self.plot.cancel_pending()
        self._render()
        self.plot.figure.savefig(path, dpi=200, bbox_inches="tight")
        return Path(path)

    def _batch_run(self) -> None:
        """Dialog wrapper. All of the logic is in `batch_tg_to_csv`."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select files for batch Tg", "", "Text files (*.txt *.dat *.asc);;All files (*.*)")
        if not files:
            return
        out_csv, _ = QFileDialog.getSaveFileName(self, "Save batch results as CSV", "", "CSV (*.csv)")
        if not out_csv:
            return
        tick = self._progress_ticker("Reading and measuring…", len(files))
        try:
            rows = self.batch_tg_to_csv([Path(f) for f in files], Path(out_csv),
                                        progress=tick)
        except Exception as exc:
            QMessageBox.critical(self, "Batch failed", str(exc))
            return
        finally:
            self._close_progress()
        failed = sum(1 for r in rows if r.get("error"))
        self._set_status(
            f"Batch: {_plural(len(rows), 'row')} from "
            f"{_plural(len(files), 'file')} to {out_csv}"
            + (f"; {failed} carried an error." if failed else "."))
        QMessageBox.information(self, "Batch complete", f"Saved batch CSV:\n{out_csv}")

    def batch_tg_to_csv(self, files: Sequence[Path], out_csv: Path, *,
                        progress: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Run the batch over `files`, write `out_csv`, return the rows.

        Split from `_batch_run` so the batch can be exercised without
        monkeypatching a file dialog: a dialog is a way of choosing paths, not
        a place for logic to live.

        Writes through `_write_rows_csv`, so a batch CSV and a per-segment CSV
        have the same columns in the same order and can be concatenated.
        """
        rows = self.batch_tg_rows(files, progress=progress)
        self._write_rows_csv(rows, out_csv)
        return rows

    def batch_tg_rows(self, files: Sequence[Path], *,
                      progress: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Result rows for `files`, using the workspace's current settings.

        One row per file under `BATCH_SCOPE_FILE` -- which is what this batch
        has always produced and stays the default -- and one row per segment
        under the other two scopes, so a folder of cyclic runs is measurable
        without opening each file by hand.

        Never raises for a single bad file: that file contributes a row whose
        `error` column says what happened, so a batch of forty is not lost to
        one unreadable export. The three refusals `_batch_segments_for_file`
        can return land in the same place and read the same way.

        Every measurement goes through `_row_for_segment` -> `_measure_tg`, the
        same path the Compute button and the per-segment export use. Before
        that, this loop carried its own copy of the three `dta_science` calls
        and its own copy of the mode/range formatting, which is how a batch
        number drifts from the one the user saw on screen -- there is no second
        copy left to drift.

        `progress(done, total, label) -> bool` is optional; returning False from
        it stops the batch after the current file, keeping the rows already
        measured. `total` counts FILES, not rows: the row count is not known
        until every file has been segmented.
        """
        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        dy_sel = self.dy_combo.currentText()
        window = self._get_window()
        smooth_d = self._smooth_derivative_enabled()
        invert = self.invert_y_check.isChecked()
        bp = self._resolved_baseline_params()

        rows: List[Dict[str, Any]] = []
        total = len(files)
        for done, f in enumerate(files):
            p = Path(f)
            if progress is not None and not progress(done, total, p.name):
                break
            sample = ""
            try:
                header, _cols, df = parse_ta_sdt_txt(p)

                def _find(name):
                    """Case-insensitive column lookup, for a sibling export
                    whose header spells the same signal differently."""
                    for c in df.columns:
                        if c.lower() == str(name).lower():
                            return c
                    return None

                x_col = x_name if x_name in df.columns else (_find(x_name) or x_name)
                y_col = y_name if y_name in df.columns else (_find(y_name) or y_name)
                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError(f"Missing columns: {x_name} or {y_name}")
                sample = header.get("Sample", p.stem)

                # Each batch file is segmented on its own terms: the workspace's
                # run belongs to the record on screen and says nothing about
                # this file. A single-ramp file -- every file this tab has ever
                # been pointed at -- yields one segment covering all its rows,
                # so the numbers are unchanged.
                file_program = program_from_header(
                    header, _header_lines_of(header), name=p.name, source_path=str(p))
                file_run = self._run_for_dataframe(df, program=file_program)
            except Exception as exc:
                rows.append(self._error_row(path=p, sample=sample, error=str(exc)))
                continue

            curves, refusal = self._batch_segments_for_file(file_run)
            if refusal:
                rows.append(self._error_row(path=p, sample=sample, error=refusal))
                continue
            for seg in curves:
                rows.append(self._row_for_segment(
                    df=df, run=file_run, seg=seg, path=p, sample=sample,
                    x_col=x_col, y_col=y_col, dy_sel=dy_sel, window=window,
                    smooth_d=smooth_d, invert=invert, bp=bp))

        if progress is not None:
            progress(total, total, "")
        return rows

    # ==================================================================
    # Rate series (Arrhenius / Lasocka) — the tab the user asked for
    # ==================================================================
    # SPEC_rate_series §6, MERGED_DECISIONS M0.2. Design rule, and it is the
    # rule that makes this feature testable: EVERY add / mutate entry point
    # takes explicit arguments and opens no dialog. `QFileDialog` appears only
    # inside an `_on_rate_*` wrapper whose whole body is "get the paths ->
    # return if empty -> call the public method". No test ever monkeypatches a
    # dialog to exercise logic, and no logic ever lives where a test cannot
    # reach it.

    @property
    def rate_rows(self) -> List[RateRow]:
        """The series' rows, in insertion order. Read-only view.

        A property rather than a second list, for the same reason
        `res_parallel` is a property onto `seg_results`: two containers holding
        the same rows is how the table and the fit end up disagreeing about
        what is in the series.
        """
        return self.rate_series.rows

    # ---- which page of the plot stack is showing ----------------------
    def _on_tab_changed(self, index: int) -> None:
        """Point the plot stack at the canvas the visible tab owns.

        Identity, not index: `self.tabs.widget(index) is self._rate_tab`. A tab
        inserted later cannot silently make this show the wrong canvas.

        Guarded on `hasattr`: `addTab` emits `currentChanged` and this must
        survive being called before `plot_stack` exists (SPEC_rate_series §10
        hazard i). The connection is also made after the stack is built, so in
        practice the guard never fires — two defences, because a construction
        crash here takes the whole workspace with it.

        Entering the tab refreshes the view and re-runs the FIT, but never
        computes a VALUE: a straight line through at most twenty points is free
        and must always match the rows on screen, whereas measuring a Tg reads
        the Tg tab's window and baselines and is a thing the user asks for by
        pressing a button.
        """
        if not hasattr(self, "plot_stack"):
            return
        on_rate = self.tabs.widget(int(index)) is getattr(self, "_rate_tab", None)
        self.plot_stack.setCurrentIndex(PLOT_PAGE_RATE if on_rate else PLOT_PAGE_MAIN)
        if not on_rate:
            return
        self._rate_refresh_columns()
        self._rate_refresh_library()
        self._rate_refresh_table()
        self._rate_refit()
        self._rate_refresh_notes()

    # ---- settings the add paths record -------------------------------
    def _rate_quantity(self) -> str:
        """"Tg" or "Tp" — the quantity NEW rows are added as."""
        data = self.rate_quantity_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[0])
        return "Tg"

    def _rate_measure_method(self) -> str:
        """"parallel" / "double" / "deriv" / "peak_max" / "peak_min".

        The construction a value will be read with. Named apart from the FIT
        model (`KINETIC_METHODS`, the Lasocka/Kissinger choice) because the two
        are different decisions about different things, and one word for both
        is how a Kissinger fit ends up labelled "parallel".
        """
        data = self.rate_quantity_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[1])
        return "parallel"

    def _rate_signal(self) -> str:
        """The Y column a new row's value will be read off.

        For Tg it is the Tg tab's own Y column, deliberately: a rate-series
        value and a Tg-tab compute on the same segment must be the same number,
        and two independent signal choices is the shortest route to two
        different numbers with no visible cause. For Tp it is the Tp signal
        combo, which is a different signal on purpose — a crystallization
        exotherm is usually read on heat flow while a Tg step is read on the
        DTA difference.
        """
        if self._rate_quantity() == "Tp":
            return (self.rate_tp_signal_combo.currentText().strip()
                    or self.y_combo.currentText())
        return self.y_combo.currentText()

    def _rate_tp_window_C(self) -> Tuple[float, float]:
        """The Tp window from its two fields, in °C, ordered.

        The fields are °C ALWAYS — their label says so — and never follow the
        °C/K switch. That is R4: a temperature field that converts is a second
        Kelvin conversion site, and a window converted twice still looks like a
        plausible temperature while making Ea wrong by about a factor of three.
        """
        lo = _to_float(self.rate_tp_min_edit.text())
        hi = _to_float(self.rate_tp_max_edit.text())
        if lo is None or hi is None:
            return RATE_DEFAULT_WINDOW_C
        return (min(float(lo), float(hi)), max(float(lo), float(hi)))

    def _rate_window_C(self) -> Tuple[float, float]:
        """The window a new row records, in CELSIUS.

        Three cases, and the third is the one that matters:

          * quantity Tp -> the Tp fields, already °C;
          * the abscissa is a temperature -> the Tg tab's window, converted
            from the DISPLAY unit through `_shift_temperature` (the single
            offset site in this file);
          * the abscissa is the clock or a signal column -> the Tg fields hold
            MINUTES or milliwatts, and copying them into a `window_min_C` cell
            would be a unit error inside an exported file.
            `RATE_DEFAULT_WINDOW_C` is recorded instead and the settings note
            says so.

        The third case is the NORMAL one on a cyclic record: the whole-run view
        forces the clock onto X (`_x_forced_time`), so the fields read 350-700
        MINUTES while the user is looking at an 890-minute run.
        """
        if self._rate_quantity() == "Tp":
            return self._rate_tp_window_C()
        if not _is_temperature_col(self.x_combo.currentText()):
            return RATE_DEFAULT_WINDOW_C
        try:
            lo, hi = self._get_window()
        except ValueError:
            return RATE_DEFAULT_WINDOW_C
        unit = self._temp_unit()
        return (float(_shift_temperature(lo, unit, "°C")),
                float(_shift_temperature(hi, unit, "°C")))

    def _rate_window_is_default(self) -> bool:
        """True when `_rate_window_C` fell back to the app default.

        Reported in the settings note rather than left silent: a row whose
        window cells came from a default the user never typed must say so.
        """
        return (self._rate_quantity() != "Tp"
                and not _is_temperature_col(self.x_combo.currentText()))

    def _rate_settings_snapshot(self) -> Dict[str, Any]:
        """The Tg/Tp settings in force, under `SETTINGS_CSV_KEYS`' own names.

        Snapshotted onto each row rather than pointed at a shared store, which
        is what makes an exported series self-describing: the file says which
        baseline mode and which window produced each number, months after the
        widgets have moved. The key names are `dta_rate_series`' so a row's
        settings become CSV columns with no translation layer to drift.
        """
        bp = self._resolved_baseline_params()
        low_mode, high_mode = _baseline_modes(bp)
        dy_sel = self.dy_combo.currentText()
        tp_lo, tp_hi = self._rate_tp_window_C()
        return {
            "tg_method": self._rate_measure_method(),
            "smooth_derivative": int(self._smooth_derivative_enabled()),
            "y_col": self.y_combo.currentText(),
            "dy_source": (self.y_combo.currentText()
                          if (not dy_sel or dy_sel == "(same)") else dy_sel),
            "invert_y": int(self.invert_y_check.isChecked()),
            "low_mode": low_mode,
            "low_range": self._fmt_range(bp.low_range),
            "low_point_x": ("" if bp.low_point is None else float(bp.low_point)),
            "high_mode": high_mode,
            "high_range": self._fmt_range(bp.high_range),
            "high_point_x": ("" if bp.high_point is None else float(bp.high_point)),
            "tp_signal": self.rate_tp_signal_combo.currentText(),
            "tp_window_min_C": float(tp_lo),
            "tp_window_max_C": float(tp_hi),
        }

    def _rate_row_kwargs(self) -> Dict[str, Any]:
        """The measurement description every add path stamps onto its rows.

        One place, so a row added from a file and a row added from a segment
        carry the same quantity, method, signal, window and tolerances. Four
        add paths with four copies of this dict is how a series ends up with
        rows nobody can compare.
        """
        return dict(quantity_kind=self._rate_quantity(),
                    method=self._rate_measure_method(),
                    signal=self._rate_signal(),
                    window_C=self._rate_window_C(),
                    settings=self._rate_settings_snapshot(),
                    tolerances=self.rate_series.tolerances)

    def _rate_record_id(self) -> str:
        """The loaded record's raw `Spectrum.id`, or "".

        `self._record_id` is a PREFIXED cache key (`id:` / `title:`); a
        `RateRow.record_id` is the bare id, and a title-keyed record has none —
        blank, not the title, because a title is a weaker identity that two
        imports of one file share (`_record_key`).
        """
        key = str(self._record_id or "")
        if key.startswith(_RECORD_ID_PREFIX):
            return key[len(_RECORD_ID_PREFIX):]
        return ""

    @staticmethod
    def _rate_frame_key(row: RateRow) -> str:
        """Which cached DataFrame a row reads: its file path, else its record.

        Keyed per SOURCE, never per row: the four heating ramps of one cyclic
        file share one frame, and a per-row cache would hold four copies of a
        5 MB export.
        """
        path = str(getattr(row, "file", "") or "").strip()
        if path:
            return path
        rid = str(getattr(row, "record_id", "") or "").strip()
        return f"{_RECORD_ID_PREFIX}{rid}" if rid else ""

    # ---- column / library combos -------------------------------------
    def _rate_refresh_columns(self) -> None:
        """Fill the Tp signal combo from the loaded record's columns.

        Not done in the constructor: a bare `DtaWorkspace()` has no columns and
        `test_dta_workspace_constructs_standalone` pins that it still builds
        (SPEC_rate_series §10 hazard ii). Called on every tab switch and at the
        start of every add path, so a row is never stamped with a column the
        record does not have.

        A selection the user made survives a refresh; signals stay blocked
        because nothing downstream needs to react to a list being rebuilt into
        the same state.
        """
        combo = self.rate_tp_signal_combo
        current = combo.currentText()
        names = list(self.colnames)
        if [combo.itemText(i) for i in range(combo.count())] == names:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            elif names:
                # Default to the Tg tab's Y column when it is one of these: a
                # peak and a step are often read off the same export, and it is
                # the only column we know the user has already chosen.
                y_col = self.y_combo.currentText()
                combo.setCurrentText(y_col if y_col in names else names[0])
        finally:
            combo.blockSignals(False)

    def _rate_refresh_library(self) -> None:
        """Repopulate the "imported" check-list from the library records.

        `CheckComboBox.set_items` preserves the ticks of surviving entries, so
        a refresh triggered by the shell cannot silently untick a record the
        user was about to add.
        """
        self.rate_library_combo.set_items(list(self._record_lookup.keys()))

    def _rate_lookup_record(self, key: Any) -> Optional[Dict[str, Any]]:
        """A library record by display key or by raw `Spectrum.id`.

        The combo hands back display keys; a test, a project file and a script
        all naturally hand back ids. Accepting both means neither caller has to
        know which the other uses.
        """
        text = str(key or "").strip()
        if not text:
            return None
        if text in self._record_lookup:
            return self._record_lookup[text]
        for rec in self.records:
            if str(rec.get("id") or "").strip() == text:
                return rec
        return None

    # ---- the four add paths ------------------------------------------
    def rate_add_segments(self, *, roles: Sequence[str] = (ROLE_HEAT,)) -> int:
        """Add one row per matching segment of the record ON SCREEN.

        Heating ramps by default. A cooling ramp is available
        (`roles=(ROLE_COOL,)`) and arrives FLAGGED — `cooling` in its Flags
        cell, `cooling_row` in its warnings — never silently mixed in with the
        heats and never refused: on this dataset the cooling exotherms give a
        physically impossible negative activation energy, and
        MERGED_DECISIONS M1 turns that from a refusal into a warning.

        Returns the number of rows added. 0 with a reason in the status line
        and in the rows note when there is nothing to add, never a dialog: a
        button press that finds no data is not an event worth interrupting
        anyone for.
        """
        self._rate_refresh_columns()
        if self.df is None:
            self._rate_notice = ("No record is loaded, so there are no segments to add. "
                                 "“Add files…” reads files straight from disk.")
            self._rate_after_change("Rate series: no record loaded.")
            return 0
        run = self.run
        if run is None or not run.segments:
            self._rate_notice = (
                "This record has no detected segments — segmentation needs both a "
                "clock and a furnace-temperature column. Add a blank row and type "
                "the rate, or add the file from disk.")
            self._rate_after_change("Rate series: this record could not be segmented.")
            return 0
        rows, warns = drs.rows_from_segments(
            run, self.df, header=self.header, source_kind="segment",
            file=str(self.path or ""), record_id=self._rate_record_id(),
            alignment=run.alignment, roles=tuple(roles), **self._rate_row_kwargs())
        self._rate_notice = ""
        n = self._rate_absorb(rows, warns, frame=self.df,
                              canonical=self._canonical_map)
        self._set_status(
            f"Rate series: added {_plural(n, 'row')} from "
            f"{self.path.name if self.path else 'this record'}. "
            f"Values stay blank until you compute them.")
        return n

    def rate_add_files(self, paths: Sequence[Any]) -> int:
        """Add rows from separate FILES — the headline of this feature.

        Five files measured at 5, 10, 15, 20 and 30 °C/min become five rows
        with their rates read from each file's own programmed method and its
        own data, and NOTHING to type. That is the textbook one-specimen-per-
        rate design and it is what the user asked for by name.

        A file holding several heating ramps yields several rows, because that
        IS the cyclic case and the only honest answer to "what rate is this
        file?" is "it has more than one" — they arrive switched off with an
        `ambiguous_file_rate` note rather than the engine guessing which ramp
        was meant.

        A file that cannot be read becomes ONE visible failed row carrying the
        verbatim message: never an exception (a batch of twenty files must not
        be lost to one bad file) and never a silently skipped file (a series
        that is quietly one row short is worse than one that says so).
        """
        self._rate_refresh_columns()
        items = [Path(str(p)) for p in (paths or []) if str(p).strip()]
        if not items:
            return 0
        kwargs = self._rate_row_kwargs()
        rows: List[RateRow] = []
        warns: List[SeriesWarning] = []
        for path in items:
            try:
                got, got_warns = drs.rows_from_file(path, **kwargs)
            except Exception as exc:                      # pragma: no cover - defensive
                # `rows_from_file` promises never to raise on data, so reaching
                # here is a bug in the builder rather than a bad file. Even
                # then the user gets a row: nineteen good files must not be
                # lost to one unexpected traceback.
                got, got_warns = self._rate_failed_row(
                    str(path), f"{exc}", source_kind="file")
            rows.extend(got)
            warns.extend(got_warns)
        self._rate_notice = ""
        n = self._rate_absorb(rows, warns)
        failed = sum(1 for r in rows if r.value_source == "failed")
        msg = (f"Rate series: added {_plural(n, 'row')} from "
               f"{_plural(len(items), 'file')}.")
        if failed:
            msg += (f" {_plural(failed, 'row')} could not be read — the reason is in "
                    f"its “Note / error” cell.")
        self._set_status(msg)
        return n

    def rate_add_library_records(self, keys: Sequence[Any]) -> int:
        """Add rows from records already imported into the library.

        Uses each record's in-memory frame when it has one, so a file the user
        already opened is not read from disk a second time and still works when
        the share it came from is offline.

        A key that no longer resolves adds nothing and is reported in the rows
        note and the status line — visible, but not as a row, because there is
        no source to make a row about.
        """
        self._rate_refresh_columns()
        wanted = [k for k in (keys or []) if str(k or "").strip()]
        if not wanted:
            return 0
        kwargs = self._rate_row_kwargs()
        rows: List[RateRow] = []
        warns: List[SeriesWarning] = []
        missing: List[str] = []
        frames: List[Tuple[RateRow, Any, Dict[str, Any]]] = []
        for key in wanted:
            rec = self._rate_lookup_record(key)
            if rec is None:
                missing.append(str(key))
                continue
            try:
                got, got_warns = drs.rows_from_record(rec, **kwargs)
            except Exception as exc:                      # pragma: no cover - defensive
                got, got_warns = self._rate_failed_row(
                    str(rec.get("title") or key), f"{exc}", source_kind="library",
                    record_id=str(rec.get("id") or ""))
            rows.extend(got)
            warns.extend(got_warns)
            frame = rec.get("df")
            if frame is not None:
                canon = dict((rec.get("meta") or {}).get("canonical_map") or {})
                frames.extend((r, frame, canon) for r in got)
        self._rate_notice = ""
        if missing:
            self._rate_notice = (
                f"{_plural(len(missing), 'checked record')} could not be found in the "
                f"library any more ({', '.join(missing[:3])}) — reopen the DTA page to "
                f"refresh the list.")
        n = self._rate_absorb(rows, warns)
        for row, frame, canon in frames:
            key = self._rate_frame_key(row)
            if key:
                self._rate_frames[key] = frame
                if canon:
                    self._rate_canonical[key] = canon
        self._set_status(
            f"Rate series: added {_plural(n, 'row')} from "
            f"{_plural(len(wanted) - len(missing), 'library record')}."
            + (" " + self._rate_notice if self._rate_notice else ""))
        return n

    def rate_add_manual_row(self, *, sample: str = "", q_C_per_min: float = NAN,
                            value_C: float = NAN, note: str = "",
                            quantity_kind: str = "", specimen_key: str = "",
                            state: str = "unknown", prior_heats: int = 0,
                            cum_mass_loss_pct: float = NAN, T0_C: float = NAN,
                            error: str = "", include: bool = True) -> str:
        """Add a typed row — literature, another instrument, another program.

        Returns its uid, which is the handle every mutation uses.

        The window is left UNKNOWN (NaN) rather than stamped with the current
        one: nobody knows what window a literature Tg was read under, and a row
        claiming this session's window would make the consistency checks report
        a series that agrees with itself when it does not.
        """
        row = drs.manual_row(
            sample=sample, q_C_per_min=_nan_float(q_C_per_min),
            value_C=_nan_float(value_C),
            quantity_kind=(quantity_kind or self._rate_quantity()),
            specimen_key=specimen_key, state=state, prior_heats=int(prior_heats),
            cum_mass_loss_pct=_nan_float(cum_mass_loss_pct),
            signal=self._rate_signal(), note=note, error=error,
            T0_C=_nan_float(T0_C), include=include)
        self._rate_absorb([row], [])
        self._set_status("Rate series: added a typed row — fill in its rate and value.")
        return row.uid

    def _rate_failed_row(self, name: str, message: str, *, source_kind: str,
                         record_id: str = "") -> Tuple[List[RateRow], List[SeriesWarning]]:
        """One visible row standing for a source that could not be read.

        The same shape `dta_rate_series` produces for a bad file, rebuilt here
        for the defence-in-depth branch above (a builder that raised despite
        promising not to). `include=False` keeps it out of the fit without
        keeping it out of sight.
        """
        stem = Path(str(name)).stem or str(name)
        row = RateRow(source_kind=source_kind, sample=stem,
                      file=str(name), record_id=str(record_id or ""),
                      value_source="failed", error=str(message), include=False)
        warn = SeriesWarning(
            code="no_heating_segment", severity="critical",
            message=(f"{stem} could not be read, so it has no heating ramp and no "
                     f"rate: {message} The row is kept so the gap is visible."),
            row_uids=(row.uid,), detail={"source": str(name), "error": str(message)})
        return [row], [warn]

    def _rate_absorb(self, rows: Sequence[RateRow],
                     warnings: Sequence[SeriesWarning], *, frame=None,
                     canonical: Optional[Dict[str, Any]] = None) -> int:
        """Append built rows to the series and put their warnings on screen.

        The single place rows enter the series, so the four add paths cannot
        drift in what they do afterwards. Values are NOT computed here and no
        add path passes a `measure` callable: the settings that decide a number
        are stale by construction at add time, and computing on add would draw
        three filled rows plus a blank one — a healthy-looking line — before
        the user had inspected a single baseline (SPEC_rate_series §1
        conflict 4).
        """
        rows = list(rows or [])
        was_empty = not self.rate_series.rows
        if rows:
            self.rate_series.extend(rows)
            self._rate_add_warnings.extend(list(warnings or []))
            if was_empty:
                # An empty series takes its quantity from what was just added;
                # a series with rows in it keeps its own, because relabelling a
                # Tg series as Tp would change which models are offered for
                # numbers that are still Tg values.
                self.rate_series.quantity_kind = (rows[0].quantity_kind
                                                  or self.rate_series.quantity_kind)
            if frame is not None:
                for row in rows:
                    key = self._rate_frame_key(row)
                    if key:
                        self._rate_frames[key] = frame
                        if canonical:
                            self._rate_canonical[key] = dict(canonical)
        self._rate_after_change()
        return len(rows)

    def _rate_after_change(self, status: str = "") -> None:
        """Re-render everything the series drives, and re-fit it. Never MEASURES.

        Called after every add, removal and cell edit. Re-fitting here is what
        makes the figure and the banner impossible to leave stale: a removed row
        changes the fit, and a fit that still showed it would be a published
        number describing data that is no longer in the series. Measuring is a
        different thing entirely and happens only in `rate_compute_all`.

        The order matters only in that the table is rebuilt before the notes
        count its rows.
        """
        self._rate_refresh_table()
        self._rate_refit()
        self._rate_refresh_notes()
        if status:
            self._set_status(status)

    # ---- dialog wrappers: get path(s), call the public method ---------
    def _on_rate_add_segments(self) -> None:
        self.rate_add_segments()

    def _on_rate_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add DTA files measured at different rates", "",
            "Text files (*.txt *.dat *.asc);;All files (*.*)")
        if not paths:
            return
        self.rate_add_files(paths)

    def _on_rate_add_library(self) -> None:
        keys = self.rate_library_combo.checked()
        if not keys:
            self._set_status("Rate series: tick one or more imported records first.")
            return
        self.rate_add_library_records(keys)

    def _on_rate_add_blank(self) -> None:
        self.rate_add_manual_row()

    def _on_rate_quantity_changed(self, _index: int = -1) -> None:
        """The quantity combo moved: re-describe what a NEW row will record.

        Existing rows are untouched — their own `quantity_kind` and `method`
        are what they were measured as, and rewriting them would relabel
        measurements nobody re-made. A series that ends up holding both kinds
        says so through `mixed_quantity`, which is critical.
        """
        if not self.rate_series.rows:
            self.rate_series.quantity_kind = self._rate_quantity()
            # An empty series has no measurement to protect, so the model may
            # follow the quantity. With rows in it the model is left alone: the
            # user's model choice is a choice, and quietly replacing it would
            # change the number on screen without anybody asking.
            self._rate_select_default_model()
        self._rate_update_quantity_guardrails()
        self._rate_refit()
        self._rate_refresh_notes()

    def _rate_update_quantity_guardrails(self) -> None:
        """Enable the Tp widgets only while Tp is the quantity.

        Disabled rather than hidden: a hidden setting is one a user cannot find
        again, and a greyed field with a value in it still says what would be
        used if they switched.
        """
        is_tp = self._rate_quantity() == "Tp"
        for w in (self.rate_tp_signal_combo, self.rate_tp_min_edit,
                  self.rate_tp_max_edit):
            w.setEnabled(is_tp)

    # ---- mutations: always by uid, never by row position --------------
    def _rate_row_by_uid(self, uid: Any) -> Optional[RateRow]:
        """The row with this uid, or None. The only way a mutation finds a row."""
        return self.rate_series.get(str(uid or ""))

    def _rate_uid_at_row(self, table_row: int) -> str:
        """The uid stored in `Qt.UserRole` of column 0, or "".

        Column 0 is the checkbox column and carries the identity precisely
        because it is the one column that always exists and is never blank.
        """
        item = self.rate_table.item(int(table_row), RATE_COL_USE)
        return "" if item is None else str(item.data(Qt.UserRole) or "")

    def _rate_selected_uids(self) -> List[str]:
        """uids of the selected rows, in table order.

        Through the selection MODEL rather than `currentRow()`: the table is
        multi-select, and "remove selected" on a three-row selection that acted
        on `currentRow()` would remove one row and look broken.
        """
        model = self.rate_table.selectionModel()
        if model is None:
            return []
        out: List[str] = []
        for index in sorted(model.selectedRows(), key=lambda i: i.row()):
            uid = self._rate_uid_at_row(index.row())
            if uid:
                out.append(uid)
        return out

    def rate_set_include(self, uid: str, flag: bool) -> bool:
        """Switch a row in or out of the fit. True when a row was found."""
        row = self._rate_row_by_uid(uid)
        if row is None:
            return False
        row.include = bool(flag)
        return True

    def rate_set_sample(self, uid: str, text: str) -> bool:
        """Rename a row — the label the plot legend will carry."""
        row = self._rate_row_by_uid(uid)
        if row is None:
            return False
        row.sample = str(text or "").strip()
        return True

    def rate_set_note(self, uid: str, text: str) -> bool:
        """Set a row's note. Never touches `error`.

        Two fields, so annotating a failed row cannot delete the sentence that
        says why it failed — which is the one thing the next reader needs in
        order to tell a bad specimen from a bad window.
        """
        row = self._rate_row_by_uid(uid)
        if row is None:
            return False
        row.note = str(text or "").strip()
        return True

    def rate_set_value(self, uid: str, value_C: float) -> bool:
        """Store a hand-typed value, in CELSIUS. Non-finite CLEARS the value.

        Clearing restores `value_source = "—"` ("nothing measured yet") rather
        than leaving `"manual"` behind on an empty cell: "—" and a blank number
        mean two different things everywhere else in this feature and they must
        not become the same thing here.
        """
        row = self._rate_row_by_uid(uid)
        if row is None:
            return False
        v = _nan_float(value_C)
        if not np.isfinite(v):
            row.value_C = NAN
            row.value_source = "—"
            # Back to the construction this ROW was added under -- from its own
            # settings snapshot, never from the combo. The combo may have moved
            # to Tp since, and stamping "parallel" onto a Tp row would produce
            # a "Tp · parallel tangent" pairing that no method in dta_science
            # can make, purely because someone cleared a cell.
            row.method = str(row.settings.get("tg_method") or row.method or "manual")
            return True
        row.set_manual_value(float(v))
        return True

    def rate_set_rate(self, uid: str, q_C_per_min: float,
                      source: str = "manual") -> bool:
        """Type a rate. Non-finite clears the override and returns to "auto".

        The typed number goes into `q_manual_C_per_min` and the measured and
        programmed candidates are LEFT ALONE, so the tooltip and the CSV still
        show what the instrument reported next to what the user preferred —
        overwriting the measured field would erase the evidence that they
        differed.
        """
        row = self._rate_row_by_uid(uid)
        if row is None:
            return False
        q = _nan_float(q_C_per_min)
        if not np.isfinite(q):
            row.q_manual_C_per_min = NAN
            row.rate_source = "auto"
            return True
        row.q_manual_C_per_min = float(q)
        row.rate_source = (source if source in drs.RATE_SOURCES else "manual")
        return True

    def rate_set_rate_source(self, uid: str, source: str) -> bool:
        """Pin which rate candidate a row uses ("auto" walks the ladder).

        A pinned candidate that turns out non-finite does not silently become
        zero: `resolve_rate` walks the ladder anyway and reports the source it
        ACTUALLY used, which is what the tooltip and the CSV then show.
        """
        row = self._rate_row_by_uid(uid)
        if row is None or source not in drs.RATE_SOURCES:
            return False
        row.rate_source = str(source)
        return True

    def rate_remove(self, uids: Sequence[str]) -> int:
        """Remove rows by uid. Returns how many went.

        By uid and never by position: a row index is stale the moment anything
        else is removed, and "remove rows 2 and 3" executed positionally
        removes row 2 and then whatever moved up into slot 3.
        """
        gone = 0
        for uid in list(uids or []):
            if self.rate_series.remove(str(uid)):
                gone += 1
        if gone:
            self._rate_drop_orphan_frames()
            self._rate_after_change(f"Rate series: removed {_plural(gone, 'row')}.")
        return gone

    def _rate_drop_orphan_frames(self) -> None:
        """Forget cached frames no surviving row points at.

        `_rate_frames` is keyed per SOURCE, so removing one of four segments of
        a cyclic file must not drop the frame the other three still need —
        which is why this is a reachability sweep and not a pop beside the
        removal. Only `rate_clear()` used to free anything, so a session
        working through fifty files held every frame it had ever measured
        (~0.85 MiB per 2 MB export) until the tab was cleared.
        """
        live = {key for key in (self._rate_frame_key(r)
                                for r in self.rate_series.rows) if key}
        for key in [k for k in self._rate_frames if k not in live]:
            self._rate_frames.pop(key, None)
            self._rate_canonical.pop(key, None)

    def rate_clear(self) -> int:
        """Empty the series, keeping its identity, name and tolerances."""
        n = len(self.rate_series.rows)
        self.rate_series.clear()
        self._rate_add_warnings = []
        self._rate_frames = {}
        self._rate_canonical = {}
        self._rate_notice = ""
        self.rate_last_fit = None
        self._rate_after_change(f"Rate series: cleared {_plural(n, 'row')}.")
        return n

    def rate_apply_row_action(self, uids: Sequence[str], action: str) -> int:
        """Apply one context-menu action to several rows. Dialog-free.

        `action` is a `RATE_SOURCES` key or "include" / "exclude" / "remove".
        The menu wrapper does nothing but map a chosen QAction onto this call,
        so every menu entry is reachable from a test without a menu.
        """
        uids = [str(u) for u in (uids or []) if str(u or "").strip()]
        if not uids:
            return 0
        act = str(action or "").strip()
        if act == "remove":
            return self.rate_remove(uids)
        done = 0
        for uid in uids:
            if act in ("include", "exclude"):
                done += int(self.rate_set_include(uid, act == "include"))
            elif act in drs.RATE_SOURCES:
                done += int(self.rate_set_rate_source(uid, act))
        if done:
            self._rate_after_change()
        return done

    def _rate_remove_selected(self) -> int:
        """Remove the selected rows; say so when nothing is selected."""
        uids = self._rate_selected_uids()
        if not uids:
            self._set_status("Rate series: select one or more rows to remove.")
            return 0
        return self.rate_remove(uids)

    def _rate_clear(self) -> int:
        """Clear-all button. No confirmation dialog: every row is rebuildable
        from its source in one click, and a confirmation on a cheap, visible
        action is what teaches people to click through dialogs."""
        return self.rate_clear()

    def _on_rate_table_menu(self, pos) -> None:
        """Right-click menu on the row table: rate source, include, remove.

        `rate_source` is the fifth user-editable field and it lives here rather
        than in a column because it is a choice among four NAMED candidates,
        not free text — and each entry can then show the number it would use
        and be switched off when that candidate does not exist.
        """
        uids = self._rate_selected_uids()
        item = self.rate_table.itemAt(pos)
        if not uids and item is not None:
            uid = self._rate_uid_at_row(item.row())
            uids = [uid] if uid else []
        if not uids:
            return
        row = self._rate_row_by_uid(uids[0]) if len(uids) == 1 else None
        menu = QMenu(self.rate_table)
        source_menu = menu.addMenu("Rate from")
        for key in drs.RATE_SOURCES:
            act = source_menu.addAction(self._rate_source_action_label(key, row))
            act.setData(key)
            if row is not None and key != "auto":
                candidate = _nan_float(getattr(row, drs._RATE_FIELD[key], NAN))
                act.setEnabled(bool(np.isfinite(candidate)))
                if not act.isEnabled():
                    act.setToolTip("This row has no such rate. Type one into the "
                                   "q column to use “manual”.")
            if row is not None and row.rate_source == key:
                act.setCheckable(True)
                act.setChecked(True)
        menu.addSeparator()
        for label, key in (("Include in the fit", "include"),
                           ("Exclude from the fit", "exclude"),
                           ("Remove", "remove")):
            act = menu.addAction(label)
            act.setData(key)
        chosen = menu.exec(self.rate_table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        self.rate_apply_row_action(uids, str(chosen.data() or ""))

    def _rate_source_action_label(self, key: str, row: Optional[RateRow]) -> str:
        """Menu text for one rate candidate, with its value when there is one."""
        if key == "auto":
            return "auto (measured → programmed → filename → typed)"
        if row is None:
            return key
        value = _nan_float(getattr(row, drs._RATE_FIELD[key], NAN))
        if np.isfinite(value):
            return f"{key} — {value:g} °C/min"
        return f"{key} — none recorded"

    # ---- the table: a pure view of the series -------------------------
    def _rate_value_display(self, row: RateRow) -> str:
        """`row.value_C` rendered in the DISPLAY unit, or "".

        One of exactly two places this feature converts a temperature — the
        other is `_rate_value_from_cell`, its inverse — and both go through
        `_shift_temperature`, the single offset site in this file. `value_C` is
        never rewritten by a unit switch: the switch relabels a header and
        re-renders a cell, which is what makes a °C→K→°C round trip
        bit-identical (R4).
        """
        v = _nan_float(row.value_C)
        if not np.isfinite(v):
            return ""
        shown = _shift_temperature(float(v), "°C", self._temp_unit())
        return f"{float(shown):.6g}"

    def _rate_value_from_cell(self, text: str) -> float:
        """A typed Value cell in °C. NaN when it is blank or not a number.

        The inverse of `_rate_value_display`: a number typed while the
        workspace shows kelvin is stored as celsius, so what is stored never
        depends on what was on screen.
        """
        v = _to_float(str(text or ""))
        if v is None:
            return NAN
        return float(_shift_temperature(float(v), self._temp_unit(), "°C"))

    def _rate_rate_tooltip(self, row: RateRow) -> str:
        """"measured 19.95 · programmed 20 · used: measured" for the q cell.

        Every candidate, always, plus the one `resolve_rate` actually used —
        which is not necessarily the one the row asked for, and that difference
        is exactly what a user needs to see when a fit uses a rate they did not
        expect.
        """
        bits = []
        for label, attr in (("measured", "q_measured_C_per_min"),
                            ("programmed", "q_programmed_C_per_min"),
                            ("filename", "q_filename_C_per_min"),
                            ("typed", "q_manual_C_per_min")):
            value = _nan_float(getattr(row, attr, NAN))
            if np.isfinite(value):
                bits.append(f"{label} {value:g}")
        if not bits:
            return "No rate recorded — type one into this cell."
        bits.append(f"asked for: {row.rate_source}")
        bits.append(f"used: {row.q_used_source}")
        return " · ".join(bits)

    def _rate_item(self, table_row: int, col: int) -> QTableWidgetItem:
        """The item at (row, col), created once and REUSED afterwards.

        Reuse is a correctness rule, not an optimisation: `setItem` deletes the
        item it replaces, and the cell being filled may be the very item whose
        `itemChanged` signal is still being delivered. Deleting it under Qt's
        feet is a crash; writing into it is not.
        """
        item = self.rate_table.item(int(table_row), int(col))
        if item is None:
            item = QTableWidgetItem("")
            self.rate_table.setItem(int(table_row), int(col), item)
        return item

    def _rate_fill_row(self, table_row: int, row: RateRow) -> None:
        """Write one series row into one table row. Signals must be blocked.

        Styling is set on every cell every time, including the cleared cases:
        items are REUSED, so a row that stops failing has to stop being red and
        a row that is switched back on has to stop being italic. "Set it when
        it applies and leave it otherwise" leaves the previous row's styling
        behind.
        """
        unit = self._temp_unit()
        q = _nan_float(row.q_C_per_min)
        value = _nan_float(row.value_C)
        mass = _nan_float(row.cum_mass_loss_pct)
        texts = {
            RATE_COL_USE: "",
            RATE_COL_SAMPLE: str(row.sample or ""),
            RATE_COL_SOURCE: str(row.source_kind or ""),
            RATE_COL_SEGMENT: str(row.segment_label or "—"),
            RATE_COL_RATE: (f"{q:g}" if np.isfinite(q) else ""),
            RATE_COL_QUANTITY: rate_quantity_label(row),
            RATE_COL_VALUE: self._rate_value_display(row),
            RATE_COL_FROM: str(row.value_source or "—"),
            RATE_COL_HISTORY: rate_history_text(row),
            RATE_COL_MASS: (f"{mass:.3f}" if np.isfinite(mass) else ""),
            RATE_COL_FLAGS: "; ".join(f for f in row.flags.split(";") if f),
            RATE_COL_NOTE: rate_note_display(row),
        }
        italic = not row.include
        failed = row.value_source == "failed"
        for col in range(len(RATE_TABLE_COLUMNS)):
            item = self._rate_item(table_row, col)
            item.setText(texts[col])
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if col == RATE_COL_USE:
                flags |= Qt.ItemIsUserCheckable
                item.setData(Qt.UserRole, row.uid)
                item.setCheckState(Qt.Checked if row.include else Qt.Unchecked)
                item.setToolTip("Untick to leave this row out of the fit. It stays in "
                                "the table and in the export, greyed on the plot.")
            elif col in RATE_EDITABLE_COLUMNS:
                flags |= Qt.ItemIsEditable
            item.setFlags(flags)
            if col == RATE_COL_RATE:
                item.setToolTip(self._rate_rate_tooltip(row))
            elif col == RATE_COL_NOTE:
                item.setToolTip(row.error or "")
            elif col == RATE_COL_VALUE:
                item.setToolTip(f"Stored as {value:.6g} °C; shown in {unit}."
                                if np.isfinite(value)
                                else "No value yet — compute it, or type one.")
            else:
                item.setToolTip("")
            font = item.font()
            font.setItalic(bool(italic))
            font.setBold(bool(failed and col in (RATE_COL_VALUE, RATE_COL_FROM)))
            item.setFont(font)
            if failed and col in (RATE_COL_VALUE, RATE_COL_FROM):
                item.setForeground(QBrush(QColor(_ROW_WARN_FG)))
            else:
                item.setData(Qt.ForegroundRole, None)

    def _rate_refresh_table(self) -> None:
        """Rebuild the whole table from the series. The table holds NO state.

        Wrapped in `blockSignals` AND `_rate_table_updating`, which is R3: every
        `setText` here would otherwise fire `itemChanged`, whose handler parses
        the cell it has just been handed back into the model — silently
        rewriting rates and values, and on a stale uid writing into the wrong
        row entirely.
        """
        table = self.rate_table
        rows = list(self.rate_series.rows)
        table.blockSignals(True)
        self._rate_table_updating = True
        try:
            header = table.horizontalHeaderItem(RATE_COL_VALUE)
            if header is not None:
                header.setText(f"Value ({self._temp_unit()})")
            if table.rowCount() != len(rows):
                table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                self._rate_fill_row(i, row)
        finally:
            self._rate_table_updating = False
            table.blockSignals(False)
        self._fit_program_columns(table, max_width=200)

    def _rate_refresh_row(self, table_row: int) -> None:
        """Re-render ONE table row, reusing its items.

        Used from `_rate_on_item_changed`, where a full rebuild is not
        available: `setRowCount` would delete the item whose signal is still
        being delivered.
        """
        uid = self._rate_uid_at_row(table_row)
        row = self._rate_row_by_uid(uid)
        if row is None:
            return
        table = self.rate_table
        table.blockSignals(True)
        self._rate_table_updating = True
        try:
            self._rate_fill_row(int(table_row), row)
        finally:
            self._rate_table_updating = False
            table.blockSignals(False)

    def _rate_on_item_changed(self, item) -> None:
        """One edited cell -> one model change -> that row re-rendered.

        Returns immediately during a rebuild (R3). Resolves the row by uid, so
        an edit can never land on a different row than the one that was typed
        into, and re-renders only that row: a full rebuild from inside a signal
        handler would delete the emitting item.
        """
        if item is None or self._rate_table_updating:
            return
        col = int(item.column())
        uid = self._rate_uid_at_row(item.row())
        row = self._rate_row_by_uid(uid)
        if row is None:
            return
        if col == RATE_COL_USE:
            self.rate_set_include(uid, item.checkState() == Qt.Checked)
        elif col == RATE_COL_SAMPLE:
            self.rate_set_sample(uid, item.text())
        elif col == RATE_COL_RATE:
            self.rate_set_rate(uid, _nan_float(_to_float(item.text())))
        elif col == RATE_COL_VALUE:
            self.rate_set_value(uid, self._rate_value_from_cell(item.text()))
        elif col == RATE_COL_NOTE:
            self.rate_set_note(uid, _rate_note_from_cell(item.text(), row))
        else:
            return
        self._rate_refresh_row(item.row())
        self._rate_refit()
        self._rate_refresh_notes()

    # ---- warnings, notes, figure -------------------------------------
    def _rate_live_warnings(self) -> List[SeriesWarning]:
        """Add-time warnings that still implicate a row in the series.

        Filtered rather than accumulated: removing the file that produced
        `ambiguous_file_rate` must remove the sentence about it, or the banner
        becomes a log of everything that ever happened. De-duplicated on
        (code, rows) because adding the same file twice is a real thing to do
        and it is one fact, not two.
        """
        live = {r.uid for r in self.rate_series.rows}
        out: List[SeriesWarning] = []
        seen = set()
        for w in self._rate_add_warnings:
            if w.row_uids and not (set(w.row_uids) & live):
                continue
            key = (w.code, tuple(sorted(w.row_uids)))
            if key in seen:
                continue
            seen.add(key)
            out.append(w)
        return sort_warnings(out)

    def _rate_loaded_live_warnings(self) -> List[SeriesWarning]:
        """Warnings the CSV reader produced, filtered like the add-time ones.

        `_rate_loaded_warnings` was written in two places and read in none, so
        `missing_source_file` — which `read_series_csv` is the ONLY caller that
        computes, because it is the only one passing `verify_paths=True` —
        could not reach the user by any route in the whole application. A
        series reloaded from a share that had been renamed said nothing at all,
        and the sentence about re-measuring needing the file was unreachable
        text. The suite looked green because the test asserts the warning is
        RETURNED by `read_series_csv`, never that a renderer sees it.

        `verify_paths` stays False on the fit path, as documented there: a fit
        runs on every edit and stat()-ing a network share on each one would be
        the slowest thing in the loop. The load is a once-per-file action and
        is where the check belongs.
        """
        warns = list(getattr(self, "_rate_loaded_warnings", ()) or ())
        if not warns:
            return []
        live = {r.uid for r in self.rate_series.rows}
        out = [w for w in warns if not w.row_uids or (set(w.row_uids) & live)]
        return sort_warnings(out)

    def _rate_view_warnings(self) -> List[SeriesWarning]:
        """Warnings about the FIGURE rather than about the data.

        `hidden_excluded` is the one code `dta_rate_series` deliberately does
        not implement (`STUBBED_WARNING_CODES`): it depends on a "Show excluded"
        toggle that exists only here, and the model has no field for it. So the
        Qt layer owns it, and it must exist — a figure that is silently not
        showing two of six rows is the figure that gets published.
        """
        if self.rate_show_excluded_check.isChecked():
            return []
        hidden = [r for r in self.rate_series.rows if not r.include]
        if not hidden:
            return []
        return [SeriesWarning(
            code="hidden_excluded", severity="warn",
            message=(f"“Show excluded” is off, so {_plural(len(hidden), 'row')} "
                     f"in the table is not on the figure. The rows and the export "
                     f"are unaffected. Fitted anyway."),
            row_uids=tuple(r.uid for r in hidden),
            detail={"n_hidden": len(hidden)})]

    def _rate_has_values(self) -> bool:
        """True once any row carries a number — measured or typed.

        The switch between "this series has not been measured yet" and "this
        series has a result", used by the banner and by the annotation.
        """
        return any(np.isfinite(_nan_float(r.value_C))
                   for r in self.rate_series.rows)

    def _rate_banner_warnings(self) -> List[SeriesWarning]:
        """What the banner shows: the add-time and view warnings, plus every
        objection to the FIT once there is a result to object to.

        The "once a value exists" rule is deliberate, and it is the line between
        information and an alarm. With nothing measured yet the only objections
        `check_series` can raise are `too_few_points` and `no_value` — which say
        exactly what the summary line already says ("4 rows, 3 distinct rates,
        0 values"), in the warning tone, about the perfectly normal state two
        clicks after adding files and before pressing Compute. Spending the
        colour there teaches a user that amber means "you have not finished
        yet", after which it no longer means "something is wrong with your
        data", which is the one job it has.

        Nothing is hidden by the rule: the moment ANY row carries a number
        every objection to the fit is on the banner, the `From` column reads
        "—" on every unmeasured row throughout, and the rows note says in words
        that adding never computes.

        De-duplicated on (code, ROWS) and deliberately NOT on the message as
        well. `_merge_warnings` keeps the message in its key because two
        `duplicate_rate` warnings about two different rates are two facts — but
        two warnings differing only in rates differ in `row_uids` too, so the
        rows alone separate them. Including the message instead prints the SAME
        observation twice whenever the add path and `check_series` both make it
        in their own words: adding the cyclic file raises `ambiguous_file_rate`
        naming the file and its three rates, and `check_series` raises it again
        about the same four rows. The add-time wording wins because it is the
        more specific of the two, which is why the live list is walked first.
        """
        out = (self._rate_live_warnings() + self._rate_loaded_live_warnings()
               + self._rate_view_warnings())
        fit = self.rate_last_fit
        if fit is not None and self._rate_has_values():
            seen = {(w.code, tuple(sorted(w.row_uids))) for w in out}
            for w in fit.all_warnings:
                key = (w.code, tuple(sorted(w.row_uids)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(w)
        return sort_warnings(out)

    def _rate_summary_text(self) -> str:
        """The clean-state banner line: what the series holds, with no alarm.

        Deliberately factual and deliberately not a warning tone — no alarms on
        unknown facts. A series with four rows and no values yet is a perfectly
        normal state two clicks after adding files.
        """
        rows = list(self.rate_series.rows)
        if not rows:
            return ("No rows yet — add the heating segments of this record, or add "
                    "files measured at different rates.")
        n_rates = len(self.rate_series.distinct_rates())
        n_values = sum(1 for r in rows if np.isfinite(_nan_float(r.value_C)))
        return (f"✓ No series warnings — {_plural(len(rows), 'row')}, "
                f"{_plural(n_rates, 'distinct rate')}, {_plural(n_values, 'value')}.")

    def _rate_update_warnings(self) -> None:
        """Render the warnings into the banner. Gates nothing.

        The banner is the first widget in the tab and is never hidden, so the
        layout cannot jump and a warning can never be below the fold. The
        objectName carries the severity to the stylesheet; Qt does not
        re-evaluate a stylesheet when an objectName changes, hence the
        unpolish/polish pair.

        An INFO-only list is shown in the quiet tone, not the amber one. Its
        lines still appear, each behind its own "ℹ" — but "this file holds four
        heating ramps, pick which to include" is information about the data, not
        a fault in it, and painting it amber spends the colour a user has learnt
        to read as "something is wrong here".

        It is a RENDERER and nothing else. `_rate_refit` calls it after the fit
        and never before, and neither the fit nor `rate_compute_all` ever reads
        `self.rate_warnings` back — a warning describes a result, it never
        decides whether to produce one (MERGED_DECISIONS M0.1).
        """
        self.rate_warnings = self._rate_banner_warnings()
        banner = self.rate_warn_banner
        if self.rate_warnings:
            banner.setText("\n".join(w.line() for w in self.rate_warnings))
            severity = max_severity(self.rate_warnings)
            name = {"critical": "WarningBannerCritical",
                    "warn": "WarningBanner"}.get(severity, "WarningBannerOk")
        else:
            banner.setText(self._rate_summary_text())
            name = "WarningBannerOk"
        banner.setObjectName(name)
        style = banner.style()
        if style is not None:
            style.unpolish(banner)
            style.polish(banner)

    def _rate_addable_note(self) -> str:
        """What "Add segments from current file" would add, right now."""
        if self.df is None:
            return "No record loaded — “Add files…” reads files straight from disk."
        run = self.run
        name = self.path.name if self.path else "this record"
        if run is None or not run.segments:
            return (f"{name} has no detected segments (segmentation needs a clock and a "
                    f"furnace-temperature column), so there is nothing to add from it.")
        heats = [s for s in run.segments if s.kind == KIND_HEAT]
        if not heats:
            return (f"{name} contains no heating ramp — its "
                    f"{_plural(len(run.segments), 'segment')} are holds or cools.")
        rates = ", ".join(f"{abs(_seg_rate(s)):.4g}" for s in heats)
        return f"{name}: {_plural(len(heats), 'heating ramp')} at {rates} °C/min."

    def _rate_refresh_notes(self) -> None:
        """The two SectionNote labels: what would be added, and under what.

        The second sentence of the rows note is the one that matters, and it is
        on screen permanently rather than in a dialog: adding a row NEVER
        computes its value. On this dataset the second heating ramp has no
        glass transition left inside 350-700 °C, so a compute-on-add would draw
        three filled rows, one blank, and a healthy-looking line through them
        before anyone had looked at a baseline.
        """
        bits = [self._rate_addable_note(),
                "Adding never computes a value: the window and baseline settings that "
                "decide the number are read when you compute, not when you add."]
        if self._rate_notice:
            bits.append("⚠ " + self._rate_notice)
        self.rate_rows_note.setText(" ".join(bits))

        lo, hi = self._rate_window_C()
        settings = [f"New rows record {self.rate_quantity_combo.currentText()} on "
                    f"“{self._rate_signal() or '—'}”, window {lo:g}–{hi:g} °C."]
        if self._rate_window_is_default():
            settings.append(
                f"The Tg tab's window is not a temperature on this view (X is "
                f"“{self.x_combo.currentText()}”), so the app default "
                f"{RATE_DEFAULT_WINDOW_C[0]:g}–{RATE_DEFAULT_WINDOW_C[1]:g} °C is "
                f"recorded instead of a number in the wrong unit.")
        settings.append("Values are stored in °C; the Value column follows the °C/K "
                        "switch and the stored number never moves.")
        self.rate_settings_note.setText(" ".join(settings))

    # ---- the fit model: built from the registry, gated by `needs` ------
    def _rate_select_default_model(self) -> None:
        """Point the model combo at the default for the series' quantity.

        `dta_kinetics.default_method_for` decides, not this layer: "Tg means
        Lasocka, Tp means Kissinger" is a statement about which model is defined
        for which quantity, and the registry is where that lives.
        """
        want = default_method_for(self.rate_series.quantity_kind)
        combo = self.rate_model_combo
        for i in range(combo.count()):
            if str(combo.itemData(i) or "") == want:
                if combo.currentIndex() != i:
                    combo.blockSignals(True)
                    try:
                        combo.setCurrentIndex(i)
                    finally:
                        combo.blockSignals(False)
                return

    def _rate_model(self) -> str:
        """The fit model key in force, always a valid `KINETIC_METHODS` key."""
        key = str(self.rate_model_combo.currentData() or "")
        if key in KINETIC_METHODS:
            return key
        return default_method_for(self.rate_series.quantity_kind)

    def rate_set_model(self, key: str) -> bool:
        """Select a fit model by registry key. Dialog-free; True when it exists.

        A model whose `needs` this series does not satisfy is still selectable
        from here, on purpose: `evaluable_methods` leaves it out of `fit_all` and
        the combo greys it, but a user (or a script) who asks for it BY NAME gets
        the number with `model_quantity_mismatch` attached rather than a refusal.
        Nothing in this feature hard-blocks for a physical reason
        (MERGED_DECISIONS M0.1), and "Kissinger on Tg data" is a judgement about
        the experiment, not an arithmetic impossibility.
        """
        combo = self.rate_model_combo
        for i in range(combo.count()):
            if str(combo.itemData(i) or "") == str(key):
                if combo.currentIndex() == i:
                    self._rate_refit()
                else:
                    combo.setCurrentIndex(i)      # fires _on_rate_model_changed
                return True
        return False

    def _rate_q_ref(self) -> float:
        """The back-prediction rate, °C/min. Falls back to the registry default.

        A blank or nonsensical field is not an error worth a dialog: q_ref
        changes no fitted parameter, only the "predicted Tg at q" line, so the
        honest response to junk is the documented default and a fit that still
        reports its slope.
        """
        v = _to_float(self.rate_qref_edit.text())
        if v is None or not np.isfinite(float(v)) or float(v) <= 0.0:
            return float(DEFAULT_Q_REF)
        return float(v)

    def _rate_refresh_model_combo(self) -> None:
        """Grey the models this series cannot be fitted with, and say why.

        The decision is `dta_rate_series.evaluable_methods`', never this layer's:
        `KINETIC_METHODS[key]["needs"]` says which quantity a model is defined
        for and `needs_T0` says which one needs a ramp start. A UI that
        re-decided that would be a second opinion about physics living in a
        widget, and the two opinions would diverge the first time a model was
        added.

        Greyed, never removed, and the SELECTION IS NEVER CHANGED: a model that
        silently swapped itself would change the number on screen without anybody
        asking for it. A selection that is not evaluable stays selected, is
        fitted, and arrives carrying `model_quantity_mismatch` — the honest
        version of the same information.
        """
        combo = self.rate_model_combo
        allowed = set(drs.evaluable_methods(self.rate_series))
        kinds = sorted({str(r.quantity_kind or "").strip()
                        for r in self.rate_series.fitted_rows()
                        if str(r.quantity_kind or "").strip()})
        if not kinds:
            kinds = [str(self.rate_series.quantity_kind or "Tg")]
        model = combo.model()
        for i in range(combo.count()):
            key = str(combo.itemData(i) or "")
            meta = KINETIC_METHODS.get(key, {})
            ok = key in allowed
            item = None if model is None else model.item(i)
            if item is not None:
                item.setEnabled(bool(ok))
            if ok:
                tip = str(meta.get("equation") or "")
            elif not (set(meta.get("needs", ())) & set(kinds)):
                needs = ", ".join(meta.get("needs", ())) or "another quantity"
                tip = (f"Defined for {needs}; this series holds "
                       f"{', '.join(kinds)}. Change the Quantity and re-measure, "
                       f"or add rows of that kind.")
            else:
                tip = ("Needs a ramp-start temperature (T0) on at least two rows. "
                       "Rows added from a segment carry one; typed rows do not.")
            combo.setItemData(i, tip, Qt.ToolTipRole)

    def _on_rate_model_changed(self, _index: int = -1) -> None:
        """A different model: re-fit, never re-measure.

        The VALUES do not depend on the model — a Tg is a Tg whether it is
        plotted against ln q or 1/T — so changing the model must not touch a
        single row. Anything else would mean the number in the table depended on
        which fit happened to be on screen when it was computed.
        """
        self._rate_refit()

    def _on_rate_view_changed(self, *_args) -> None:
        """A view option moved (labels, excluded rows, q_ref): re-fit and redraw.

        `q_ref` is here rather than in a measurement path because it changes only
        the back-prediction the fit reports. "Show excluded" is here because
        hiding rows from the figure raises `hidden_excluded`, which is a warning
        about the figure and therefore has to reach the banner.
        """
        self._rate_refit()

    # ---- measuring the rows -------------------------------------------
    def _rate_shared_settings(self) -> Dict[str, Any]:
        """Every setting a Compute-all run will use, read ONCE.

        One snapshot for the whole loop, never one read per row. The window, the
        baselines, the smoothing flag and the column choices are WIDGETS, and a
        redraw or a stray keystroke between row 1 and row 4 would measure the two
        halves of a series under different settings — the one thing a rate series
        cannot survive, because the comparison between its rows IS the
        measurement. It is also what each row's own `settings` snapshot then
        records, so the exported file can say what produced each number.
        """
        return dict(window=self._rate_window_C(),
                    bp=self._resolved_baseline_params(),
                    smooth_d=self._smooth_derivative_enabled(),
                    quantity_kind=self._rate_quantity(),
                    method=self._rate_measure_method(),
                    signal=self._rate_signal(),
                    dy_sel=self.dy_combo.currentText(),
                    invert=bool(self.invert_y_check.isChecked()),
                    settings=self._rate_settings_snapshot())

    def rate_compute_all(self, only_failed: bool = False) -> Tuple[int, int]:
        """Measure every row with ONE shared set of settings. `(measured, failed)`.

        Dialog-free and never raising, so a test drives it directly. Nothing is
        skipped silently and nothing is removed:

          * a row that CANNOT be measured keeps its place in the table with
            `value_source="failed"` and the verbatim message. On the real cyclic
            file heat 2 raises "Baseline range has too few points." because after
            cycle 1 the specimen is largely crystallized, no glass transition is
            left inside 350-700 °C, and the derivative peak lands on the 700 °C
            window edge with one point above it (BM_FINDINGS §8). That row is then
            visible in FOUR places — the table, the banner, the annotation's
            "Not fitted" line and a rug tick at its own rate on the x axis —
            because a row that silently vanishes from a 4-point Lasocka fit leaves
            the user believing they fitted four points when they fitted three, and
            that is the failure this feature exists not to have;
          * a row whose value was TYPED is kept exactly as it is. A manual
            override wins and survives every later Compute all: the user overrode
            the computation on purpose, and a button labelled "compute" quietly
            undoing that is what makes people keep their numbers in a spreadsheet
            instead;
          * a typed row with no source to measure becomes a failed row saying so,
            rather than staying blank and looking like something nobody had tried.

        `only_failed=True` is `rate_retry_failed`'s path and touches nothing else.
        Neither path consults `self.rate_warnings`: a warning describes a result,
        it never decides whether to produce one (M0.1).
        """
        self._rate_refresh_columns()
        shared = self._rate_shared_settings()
        measured = failed = kept = 0
        for row in list(self.rate_series.rows):
            if only_failed and row.value_source != "failed":
                continue
            if row.value_source == "manual":
                kept += 1
                continue
            self._rate_compute_one(row, **shared)
            if row.value_source == "computed":
                measured += 1
            else:
                failed += 1
        self._rate_refresh_table()
        self._rate_refit()
        self._rate_refresh_notes()

        what = "Retried" if only_failed else "Computed"
        if not measured and not failed:
            tail = (" — every row already carries a typed value." if kept
                    else " — nothing to measure.")
            self._set_status(f"Rate series: {what.lower()} nothing{tail}")
            return measured, failed
        bits = [f"Rate series: {what} {_plural(measured, 'value')}"]
        if failed:
            bits.append(f"{_plural(failed, 'row')} could not be measured — the "
                        f"reason is in its “Note / error” cell, in the banner and "
                        f"on the figure")
        if kept:
            bits.append(f"{_plural(kept, 'typed value')} left untouched")
        self._set_status(". ".join(bits) + ".")
        return measured, failed

    def rate_retry_failed(self) -> Tuple[int, int]:
        """Re-measure ONLY the rows that failed, with the settings as they stand.

        The button for "I widened the window, try that one again". It cannot
        disturb a row that already carries a number, so a series that is
        three-quarters measured is never re-measured wholesale under settings
        chosen to rescue its last quarter — which would quietly change three
        numbers the user had already read.
        """
        return self.rate_compute_all(only_failed=True)

    def _on_rate_compute(self) -> None:
        self.rate_compute_all()

    def _on_rate_retry_failed(self) -> None:
        self.rate_retry_failed()

    def _rate_frame_for(self, row: RateRow) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """`(frame, canonical_map)` for one row's source. Reads disk at most once.

        The frame is the row's OWN source, resolved through `_rate_frame_key`:
        five files at five rates are five different frames, and measuring any of
        them against whichever frame happens to be on screen is the whole class of
        bug this lookup exists to make impossible.

        `rate_add_files` deliberately leaves files on disk — a user may add twenty
        and compute two — so the first compute reads and caches; `rate_add_segments`
        and `rate_add_library_records` have already cached the frame the user is
        looking at, so a compute never re-reads a file that is open.

        Raises ValueError with a sentence the user can act on, never returning a
        silent None: the caller turns it into a failed row, and a failed row with
        no reason is the one thing worse than a failed row.
        """
        key = self._rate_frame_key(row)
        if not key:
            raise ValueError(
                "This row was typed in, so it has no data to measure — type its "
                "value as well, or add the file it came from.")
        frame = self._rate_frames.get(key)
        if frame is not None:
            return frame, dict(self._rate_canonical.get(key) or {})
        path = str(row.file or "").strip()
        if not path:
            raise ValueError(
                "The library record this row came from is no longer in memory and "
                "it carried no file path, so there is nothing left to measure. "
                "Re-import it and add the row again.")
        _header, _signals, df = parse_ta_sdt_txt(Path(path))
        self._rate_frames[key] = df
        # No parser meta on this path, so no canonical map: `_time_temp_columns`
        # falls back to its keyword rule, which is the same rule
        # `dta_rate_series._pick_columns` used when this row's span was recorded.
        return df, dict(self._rate_canonical.get(key) or {})

    @staticmethod
    def _rate_segment_for(row: RateRow, *, n_rows: int,
                          bounds: Optional[Tuple[int, int]] = None) -> Segment:
        """A `Segment` carrying THIS row's slice of its own frame.

        `row.row_start` / `row.row_stop` are ORIGINAL DataFrame positions —
        `SegmentedRun.row_indices(seg)[0]` and `[-1] + 1`, per MERGED_DECISIONS
        M3. With `bounds` they are overridden by CLEANED bounds, which is what
        `_rate_segment_rows` computes and the only form `prepare_segment_xy`
        can combine with a `keep_index`.

        Rebuilding the whole `SegmentedRun` to look one segment up would be wrong
        as well as wasteful: a detection knob can have moved since the row was
        added, and a re-derived segment #4 could then cover a different stretch of
        data from the one whose rate is printed in the row.

        A row with no recorded span (a typed row, or one whose builder failed)
        gets the whole frame, which is the honest reading of "no segment was
        recorded": as far as anything knows there is one curve here. The kind is
        carried over, so a HOLD row still meets `prepare_segment_xy`'s refusal of
        temperature as an abscissa instead of being quietly sorted.
        """
        n = max(int(n_rows), 0)
        whole = _whole_record_segment(n)
        if bounds is None:
            start = int(getattr(row, "row_start", -1))
            stop = int(getattr(row, "row_stop", -1))
        else:
            start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop <= start or start >= n:
            return whole
        stop = min(stop, n)
        return replace(
            whole,
            index=max(int(getattr(row, "segment_index", 0) or 0), 0),
            kind=(str(row.segment_kind or "") or whole.kind),
            role=(str(row.role_in_cycle or "") or whole.role),
            cycle_index=max(int(getattr(row, "cycle_index", 0) or 0), 1),
            row_start=start, row_stop=stop, n_points=stop - start,
            is_partial=bool(row.is_partial),
            note="span recorded when the row was added",
        )

    def _rate_segment_rows(self, row: RateRow, *, time_min: np.ndarray,
                           temp_C: np.ndarray
                           ) -> Tuple[Segment, Optional[np.ndarray], str]:
        """`(segment, keep_index, problem)` naming the EXACT rows of this row.

        THE defect this exists to fix. `row.row_start` / `row.row_stop` are a
        bounding INTERVAL over a row SET (MERGED_DECISIONS M3 stores
        `row_indices(seg)[0]` and `[-1] + 1`), and an interval cannot carry a
        set. Rebuilding `np.arange(row_start, row_stop)` from it is equivalent
        only while `keep_index` is the identity — i.e. only while the cleaning
        dropped nothing and reordered nothing. Nothing checked that, so the
        rate path did not drift from the authoritative one, it **ignored
        `keep_index` entirely** and reproduced the pristine-file answer no
        matter how many rows had been rejected.

        Measured on the committed fixture, by stalling the clock inside heating
        segment 1 (a repeated timestamp, which `clean_time_temperature` drops):
        the Tg tab moved 622.79 -> 619.93 °C while the rate row stayed pinned
        at 622.7934813374297 at every count, with no warning code naming it and
        `rows.csv` printing a span the value had not been read from. On two
        concatenated exports the two paths differed by **227 °C**, and a
        cooling segment whose span came out reversed fell into the "no span
        recorded" branch and was measured over the WHOLE record under the note
        "this file was not segmented" — the global-scope curve the rework
        exists to prevent.

        The fix re-derives the mapping instead of storing it, which is sound
        because `clean_time_temperature` is a pure function of the clock and
        the temperature: given the same frame it returns the same
        `keep_index`, so finding `row_start` and `row_stop - 1` in it recovers
        the cleaned bounds exactly, and `keep_index[lo:hi]` is the original row
        set. A segment is always a contiguous range of the CLEANED arrays, so
        two integers are enough once they index the right array. The reversed
        span resolves itself: `row_stop - 1` is looked up, not compared.

        When the frame no longer produces a `keep_index` containing both ends —
        the file was re-written under the row — the stored span is used as
        before and `problem` says so, which `_rate_compute_one` records in
        `extras` as `row_span_not_reproducible`. It reports; it does not refuse
        (M0.1).
        """
        T = np.asarray(temp_C, dtype=float).ravel()
        n = int(T.size)
        start = int(getattr(row, "row_start", -1))
        stop = int(getattr(row, "row_stop", -1))
        if start < 0 or stop < 0:
            # No span recorded at all: a typed row, or a builder that failed.
            return self._rate_segment_for(row, n_rows=n), None, ""
        t = (T if time_min is None else np.asarray(time_min, dtype=float).ravel())
        try:
            _tc, _Tc, keep = clean_time_temperature(t, T)
        except Exception:
            return self._rate_segment_for(row, n_rows=n), None, ""
        keep = np.asarray(keep, dtype=np.intp)
        if int(keep.size) == n and bool(np.array_equal(keep, np.arange(n, dtype=np.intp))):
            # The identity case, which is every real export shipped here: the
            # stored interval IS the row set, and this stays bit-identical to
            # the behaviour the pinned numbers were measured under.
            return self._rate_segment_for(row, n_rows=n), None, ""
        lo_hits = np.flatnonzero(keep == start)
        hi_hits = np.flatnonzero(keep == stop - 1)
        if lo_hits.size and hi_hits.size and int(hi_hits[0]) >= int(lo_hits[0]):
            lo, hi = int(lo_hits[0]), int(hi_hits[0]) + 1
            return (self._rate_segment_for(row, n_rows=n, bounds=(lo, hi)),
                    keep, "")
        return (self._rate_segment_for(row, n_rows=n), None,
                f"the stored row span {start}..{stop} could not be found in this "
                f"file as it is now, so the value was read off those row "
                f"positions directly")

    def _rate_arrays_for(self, row: RateRow, *, signal: str = "",
                         dy_sel: str = "", invert: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """`(x, y, y_dy, x_col)` for one row — its own segment, in CELSIUS.

        Three rules, each of them a wrong number if broken:

          * **the frame is the row's own** (`_rate_frame_for`), not the one on
            screen;
          * **the arrays come from `dta_segments.prepare_segment_xy`** through
            `_prepare_xy`, and from nothing else (MERGED_DECISIONS M4). NEVER from
            `_get_xy`: that method's abscissa is the workspace's current X column
            and its scope is the workspace's current scope, so on a cyclic record
            it hands back four heating ramps interleaved under a global `argsort`,
            and a Tg read off that is 432 °C instead of 620.8 (Risk R2).
            `_prepare_xy` is given `run=None` and an explicit single-segment
            scope, exactly as the batch loop does for a foreign file;
          * **the abscissa is the furnace temperature in °C, never the display
            unit.** `row.window_C` is °C and `row.value_C` is °C, so a compute
            that saw kelvin would need a second conversion to store its answer —
            which is R4, and the reason a Kissinger Ea here is bit-identical in
            both display units. `_convert_x_to_display` is deliberately NOT
            called.

        The dY source follows the Tg tab's own rule: the Y column unless a
        different one is selected, and a separate array even when it is the same
        column, because the two travel into different `dta_science` calls and must
        not alias.
        """
        df, canonical = self._rate_frame_for(row)
        _t_col, T_col = _time_temp_columns(list(df.columns), canonical)
        if not T_col:
            raise ValueError(
                f"{Path(str(row.file or '')).name or 'This source'} has no "
                f"furnace-temperature column, so a temperature cannot be read off "
                f"it.")
        y_name = _resolve_column(df, str(signal or row.signal or ""))
        if not y_name:
            raise ValueError(
                f"This source has no column “{signal or row.signal}” to read the "
                f"value off. Pick a signal this file actually carries.")
        x = pd.to_numeric(df[T_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype=float)
        y_alt = None
        if dy_sel and dy_sel != "(same)":
            alt_name = _resolve_column(df, dy_sel)
            if alt_name:
                y_alt = pd.to_numeric(df[alt_name],
                                      errors="coerce").to_numpy(dtype=float)
        t_raw = (None if not _t_col else
                 pd.to_numeric(df[_t_col], errors="coerce").to_numpy(dtype=float))
        seg, keep, problem = self._rate_segment_rows(row, time_min=t_raw, temp_C=x)
        prepared = self._prepare_xy(x, y, x_col=T_col, y_alt=y_alt, invert=invert,
                                    run=None, scope=_single_segment_scope(seg),
                                    keep_index=keep)
        # Provenance of the number about to be produced, merged into
        # `RateRow.extras` by `_rate_compute_one`. Written only when there is
        # something to say, so a row measured off a clean curve carries exactly
        # the extras it always did.
        facts: Dict[str, float] = {}
        if prepared.duplicate_fraction > 0.0:
            facts["duplicate_fraction"] = float(prepared.duplicate_fraction)
        if problem:
            facts["row_span_not_reproducible"] = 1.0
        self._rate_prep_extras = facts
        y_dy = prepared.y_alt if prepared.y_alt is not None else prepared.y.copy()
        return prepared.x, prepared.y, y_dy, T_col

    def _rate_peak_value(self, x: np.ndarray, y: np.ndarray, *,
                         window: Tuple[float, float], method: str
                         ) -> Tuple[float, Dict[str, float]]:
        """The TEMPERATURE of an extremum inside the window, plus its diagnostics.

        Tp is the x of the peak and not its height. The mistake is easy to make
        and it yields a "temperature" in milliwatts that every downstream
        Arrhenius fit will happily consume.
        """
        if method not in RATE_PEAK_METHODS:
            raise ValueError(f"{method!r} is not a peak construction; expected one "
                             f"of {', '.join(RATE_PEAK_METHODS)}.")
        lo, hi = float(window[0]), float(window[1])
        inside = (x >= lo) & (x <= hi) & np.isfinite(y)
        n = int(np.count_nonzero(inside))
        if n < 2:
            raise ValueError(
                f"Only {n} usable point(s) of this segment lie inside {lo:g}-{hi:g} "
                f"°C, so there is no peak to find there. Widen the Tp window, or "
                f"check that it matches this segment's temperature range.")
        xw, yw = x[inside], y[inside]
        j = int(np.nanargmax(yw)) if method == "peak_max" else int(np.nanargmin(yw))
        extras = {"peak_y": float(yw[j]), "n_window_points": float(n)}
        if j in (0, int(xw.size) - 1):
            # Recorded, not refused. BM_FINDINGS §8 is exactly this failure on the
            # Tg side, and the right product behaviour is a number that says what
            # is wrong with it rather than no number at all.
            extras["peak_at_window_edge"] = 1.0
        return float(xw[j]), extras

    def _rate_tg_value(self, x: np.ndarray, y: np.ndarray, y_dy: np.ndarray, *,
                       window: Tuple[float, float], bp: BaselineParams,
                       smooth_d: bool, method: str, signal: str, dy_col: str,
                       x_col: str) -> Tuple[float, Dict[str, float], str]:
        """`(value, extras, error)` from the ONE shared Tg measurement path.

        Goes through `_measure_tg`, which the Tg tab, the per-segment export and
        the multi-file batch also use — so a rate-series value and a Tg-tab
        `_compute()` on the same segment under the same settings are the same
        number by construction rather than by coincidence.

        All three constructions run even though one is wanted, and the other two
        land in `extras` with their spread. That is nearly free at these sizes and
        it is provenance a reader of the exported CSV needs: three methods
        agreeing to 0.01 °C have COLLAPSED rather than agreed
        (`DEGENERATE_SPREAD`), and without the spread in the file there is no way
        to tell that from a well-behaved measurement.
        """
        res = self._measure_tg(x, y, y_dy, window=window, smooth_d=smooth_d,
                               bp=bp, unit="°C", x_col=x_col, y_col=signal,
                               dy_col=dy_col, seg=None)
        picked = {"parallel": (None if res.res_parallel is None
                               else res.res_parallel.tg),
                  "double": (None if res.res_double is None else res.res_double.tg),
                  "deriv": res.tg_deriv}.get(str(method))
        td, tp, tx = res.tg_values
        extras: Dict[str, float] = {"tg_double_C": float(td),
                                    "tg_parallel_C": float(tp),
                                    "tg_deriv_C": float(tx),
                                    "spread_C": float(res.spread)}
        error = rate_method_error(res.error, method)
        if picked is None or not np.isfinite(_nan_float(picked)):
            if not error:
                error = (f"The {RATE_TG_METHOD_LABELS.get(method, method)} "
                         f"construction returned no value.")
            # The measured reasons travel WITH the failure, because the exception
            # names the symptom and not the cause: "Baseline range has too few
            # points" is what the user reads, while "the window 350..700 is only
            # partly inside this curve (62..212)" is what they have to fix.
            if res.warnings:
                error = f"{error} ({'; '.join(res.warnings)})"
            return NAN, extras, error
        return float(picked), extras, ""

    def _rate_compute_one(self, row: RateRow, *, window: Tuple[float, float],
                          bp: BaselineParams, smooth_d: bool,
                          quantity_kind: str, method: str, signal: str,
                          dy_sel: str, invert: bool,
                          settings: Dict[str, Any]) -> None:
        """Measure ONE row in place. Never raises; records the failure instead.

        The description is stamped on the row BEFORE the measurement is
        attempted, so a row that fails still says what was tried and under which
        window — the difference between a failure a user can fix and one they can
        only wonder about.

        The row's quantity, method, signal and window are overwritten with the
        shared ones on purpose. "Compute all" means every row measured the same
        way; leaving each row on its own stale snapshot is how a series ends up
        holding numbers nobody can compare, and `mixed_quantity` / `mixed_method`
        would then fire on rows that were never re-measured — a typed value, for
        instance — instead of on an actual accident.
        """
        row.quantity_kind = str(quantity_kind or row.quantity_kind)
        row.method = str(method or row.method)
        row.signal = str(signal or row.signal)
        row.window_C = (float(window[0]), float(window[1]))
        row.settings = dict(settings)
        row.extras = {}
        self._rate_prep_extras = {}
        try:
            x, y, y_dy, x_col = self._rate_arrays_for(
                row, signal=row.signal, dy_sel=dy_sel, invert=invert)
            if str(quantity_kind) == "Tp":
                value, extras = self._rate_peak_value(x, y, window=window,
                                                      method=row.method)
                error = ""
            else:
                dy_col = (dy_sel if (dy_sel and dy_sel != "(same)") else "(same)")
                value, extras, error = self._rate_tg_value(
                    x, y, y_dy, window=window, bp=bp, smooth_d=smooth_d,
                    method=row.method, signal=row.signal, dy_col=dy_col,
                    x_col=x_col)
        except Exception as exc:
            row.value_C = NAN
            row.value_source = "failed"
            row.error = str(exc)
            return
        row.extras = {str(k): float(v) for k, v in dict(extras).items()}
        row.extras.update(self._rate_prep_extras)
        if np.isfinite(_nan_float(value)):
            row.value_C = float(value)
            row.value_source = "computed"
            row.error = ""
        else:
            row.value_C = NAN
            row.value_source = "failed"
            row.error = error or "The measurement returned no value."

    # ---- the fit, the panel text and the annotation --------------------
    def _rate_refit(self) -> Optional[KineticFitResult]:
        """Fit the series as it stands and re-render everything that shows it.

        Fully synchronous. A straight line through at most twenty points needs no
        debounce, and synchronous means a test never waits for the number it just
        asked for.

        It NEVER consults `self.rate_warnings`, and neither does
        `rate_compute_all`. That is the invariant carrying the user's instruction
        "still make this possible to process such data": `fit_series` is handed
        the rows and produces a number plus its objections, and nothing on this
        side reads the objections back to decide whether to fit. Pinned by a test
        that makes `check_series` return thirty critical warnings and requires the
        fitted slope to be unchanged.

        Order is load-bearing at the end: `rate_annotation_text` is stored BEFORE
        `_rate_draw` puts it on the axes, so the attribute and the figure carry
        the same string by construction rather than by two formatting calls that
        agree today.
        """
        if not hasattr(self, "rate_model_combo"):
            # Two defences against a construction-order crash, as
            # `_on_tab_changed` has: the trailing refresh of `_build_rate_tab`
            # runs after the Fit group exists, and a crash here would take the
            # whole workspace down.
            return None
        series = self.rate_series
        fit = None
        if series.rows:
            fit = drs.fit_series(series, self._rate_model(),
                                 q_ref_C_per_min=self._rate_q_ref())
        self.rate_last_fit = fit
        self._rate_refresh_model_combo()
        self._rate_update_warnings()
        self.rate_annotation_text = self._rate_annotation_text(fit)
        self.rate_fit_label.setText(self._rate_report_text(fit))
        self._rate_draw()
        return fit

    def _rate_row_base_label(self, row: RateRow) -> str:
        """The undisambiguated short name — see `_rate_row_label`.

        `sample` is tried before the file stem because it is the name the
        operator gave the specimen on the instrument ("NaBiO3-400C-N2"), which
        says more than whatever the export happened to be filed as; the stem is
        the fallback for a record with no Sample header.
        """
        seg = str(row.segment_label or "").strip()
        if int(getattr(row, "n_heating_segments_in_source", 1) or 1) > 1 and seg:
            return seg
        for text in (row.sample, Path(str(row.file or "")).stem, row.record_id):
            text = str(text or "").strip()
            if text:
                return text
        return seg

    def _rate_row_label(self, row: RateRow) -> str:
        """The short name a row goes by on the figure and in the annotation.

        `RateRow.label` is the long form ("sample · heat 2 (cycle 2) @ 20.0
        °C/min"), which is right for a legend and too long for the corner of an
        axes. This rule picks whichever half actually distinguishes this row from
        its neighbours: inside one cyclic file every row shares the file name and
        differs by segment, while across five separate files every row is segment
        "heat 1" and differs by file.

        Picking the distinguishing half is not the same as BEING distinct, and
        the difference is not hypothetical: re-running one specimen name in two
        exports, or two heats pulled from files whose Sample headers match, both
        give several rows one label. An annotation that names two rows
        identically is worse than a clumsy one, because the reader cannot tell
        which point the warning is about. So a label that collides with another
        row's is extended by whatever separates them -- its segment, else its
        1-based table position -- and only rows that actually collide pay for it.
        """
        base = self._rate_row_base_label(row)
        others = [r for r in self.rate_series.rows if r.uid != row.uid]
        if base and not any(self._rate_row_base_label(r) == base for r in others):
            return base
        n = self.rate_series.index_of(row.uid)
        seg = str(row.segment_label or "").strip()
        if base and seg and seg != base:
            candidate = f"{base} · {seg}"
            if not any(self._rate_row_base_label(r) == base
                       and str(r.segment_label or "").strip() == seg
                       for r in others):
                return candidate
        if base:
            return f"{base} (row {n})" if n else base
        if seg:
            return f"{seg} (row {n})" if n else seg
        return f"row {n}" if n else "row"

    def _rate_not_fitted_rows(self, fit: Optional[KineticFitResult]) -> List[RateRow]:
        """Included rows that did NOT make it into the fit, in table order.

        Computed by DIFFERENCE against `fit.rows_used` rather than by guessing at
        reasons, so it catches every cause at once: no value yet, a failed
        measurement, a missing rate, and a point the model itself dropped (a row
        with no T0 under Augis-Bennett, or a temperature at or below absolute
        zero). A reason this layer had to enumerate is a reason it could forget,
        and the forgotten one is the row that disappears in silence.
        """
        used = set(fit.rows_used) if fit is not None else set()
        return [r for r in self.rate_series.rows if r.include and r.uid not in used]

    #: How many unfitted rows the accounting line names before it gives up and
    #: counts the rest. Four left the fifth anonymous behind "+1 more" on the
    #: real cyclic file with cooling rows added, which is the one row a reader
    #: cannot go and look at.
    RATE_ACCOUNTING_MAX_NAMES = 8

    def _rate_accounting_line(self, fit: Optional[KineticFitResult]) -> str:
        """"n = 3 of 8 rows   (2 switched off, 3 no value, 0 dropped …)".

        The line the whole failure-visibility design rests on. A user who reads
        "n = 3 of 8 rows" cannot believe they fitted eight points, and the names
        make it a specific row rather than an arithmetic complaint.

        **The counts PARTITION the rows**, and they did not. The old line mixed
        `fit.n` (model level) with counts taken over `r.include` (series level)
        and its parenthetical knew only two causes, so on real data it printed
        sums that were not the row total: `3 + 0 + 3 = 6` of 8 rows under
        Augis-Bennett, where five markers were drawn and the line went through
        three, and `4 + 0 + 3 = 7` of 8 with a rate blanked. A reader checking
        the arithmetic of a figure found it wrong, which is worse than not
        printing it.

        Every row now lands in exactly one bucket:

          * **fitted** — `fit.n`, the points the model used;
          * **no value** — no finite value, counted over ALL rows exactly as
            `check_series`' `no_value` counts them. That alignment is the
            second fix: `n_no_value` used to skip excluded rows, so switching a
            FAILED row off moved it into "excluded" and the figure printed
            "0 no value" three lines above a banner reading "3 row(s) have no
            value". Both statements were true of their own definitions and the
            pair was a contradiction to anyone reading the figure;
          * **switched off** — excluded AND carrying a value, so it cannot
            double-count with the above;
          * **dropped by the model** — the remainder: included, has a value,
            and still not in the fit. A row with no `T0` under Augis-Bennett, a
            row whose rate is zero, a temperature at or below absolute zero.
            Computed by difference, so an unforeseen cause lands here instead
            of unbalancing the sum.
        """
        rows = list(self.rate_series.rows)
        n_fitted = int(fit.n) if (fit is not None and fit.ok) else 0
        missing = self._rate_not_fitted_rows(fit)
        n_no_value = sum(1 for r in rows if not np.isfinite(_nan_float(r.value_C)))
        n_off = sum(1 for r in rows
                    if not r.include and np.isfinite(_nan_float(r.value_C)))
        n_dropped = max(len(rows) - n_fitted - n_no_value - n_off, 0)
        text = (f"n = {n_fitted} of {_plural(len(rows), 'row')}   "
                f"({n_off} switched off, {n_no_value} no value, "
                f"{n_dropped} dropped by the model)")
        if missing:
            limit = self.RATE_ACCOUNTING_MAX_NAMES
            names = ", ".join(self._rate_row_label(r) for r in missing[:limit])
            if len(missing) > limit:
                names += f", +{len(missing) - limit} more"
            text += f"      Not fitted: {names}"
        if rows and not any(r.include for r in rows):
            # The remedy the fit itself prints ("Compute a Tg for more rows, or
            # type values in by hand") comes from `dta_kinetics`, which has no
            # `include` concept, and it is the one thing that cannot help when
            # every row is switched off — which is the normal state right after
            # "Add files…" on a file holding several rates.
            text += "      Every row is switched off: tick the ones to include."
        return text

    def _rate_quantity_line(self) -> str:
        """"quantity: Tg · parallel · window 350-700 °C", read off the ROWS.

        Off the rows and not off the combo, deliberately: the combo says what the
        NEXT measurement would be, and an annotation beside a fitted line has to
        say what the numbers in it actually are. A series holding two kinds says
        "mixed", which is the same fact `mixed_quantity` reports as critical.
        """
        rows = [r for r in self.rate_series.rows
                if np.isfinite(_nan_float(r.value_C))]
        if not rows:
            return (f"quantity: {self.rate_quantity_combo.currentText()} "
                    f"(nothing measured yet)")
        kinds = sorted({rate_quantity_label(r) for r in rows})
        windows = sorted({(float(r.window_C[0]), float(r.window_C[1])) for r in rows
                          if np.isfinite(_nan_float(r.window_C[0]))
                          and np.isfinite(_nan_float(r.window_C[1]))})
        what = kinds[0] if len(kinds) == 1 else f"mixed ({', '.join(kinds)})"
        if len(windows) == 1:
            win = f"window {windows[0][0]:g}-{windows[0][1]:g} °C"
        elif not windows:
            win = "window not recorded"
        else:
            win = f"{len(windows)} different windows"
        return f"quantity: {what} · {win}"

    def _rate_annotation_text(self, fit: Optional[KineticFitResult]) -> str:
        """The exact block drawn INTO the rate figure.

        Assembled from PUBLIC `KineticFitResult` API only — `annotation()` for the
        numbers, `all_warnings` with each warning's own `glyph` and `short()` for
        the tail — so the headline here can never drift from `report()`'s. The
        numbers are asked for with `max_warnings=0` and the warnings re-attached
        below, purely to slide the two provenance lines in between them, which is
        the order SPEC_rate_series §6 spells out: the numbers, then what was
        fitted and what was not, then the objections.

        Drawn into the axes rather than beside them because a PNG that leaves this
        application has to carry its caveats with it. A figure saying
        "Ea = -36.7 kJ/mol" whose reasons live only in a panel the reader never
        saw is worse than no figure.
        """
        if not self.rate_series.rows:
            return ""
        if not self._rate_has_values():
            return ("No values yet.\nEvery row is on the axis at its own rate; "
                    "press “Compute all values” to measure them.\n"
                    + self._rate_quantity_line())
        if fit is None:
            return self._rate_quantity_line()
        head = [ln for ln in fit.annotation(max_warnings=0).split("\n")
                if not ln.startswith("…")]
        lines = head + [self._rate_accounting_line(fit), self._rate_quantity_line()]
        shown = fit.all_warnings
        if shown:
            lines.append("─" * 30)
            for w in shown[:RATE_ANNOTATION_MAX_WARNINGS]:
                lines.append(f"{w.glyph} {w.short()}")
            extra = len(shown) - RATE_ANNOTATION_MAX_WARNINGS
            if extra > 0:
                lines.append(f"… {_plural(extra, 'more warning')} — in full in the "
                             f"banner and the Fit panel")
        return "\n".join(lines)

    def _rate_report_text(self, fit: Optional[KineticFitResult]) -> str:
        """The Fit panel's text: the complete record, nothing elided.

        `KineticFitResult.report()`'s own docstring draws the division: the
        annotation is the abbreviated rendering for a finite corner of an axes,
        this one is the record. Every warning appears here in full even when the
        figure shows four of twelve, which is what makes eliding on the figure
        acceptable at all.
        """
        if not self.rate_series.rows:
            return ("No rows yet. Add the heating segments of this record, or add "
                    "files measured at different rates, then compute their values.")
        if not self._rate_has_values():
            return ("No values yet — press “Compute all values”. Adding rows never "
                    "measures them: the window and baseline settings that decide "
                    "the number are read when you compute, not when you add.")
        if fit is None:
            return self._rate_quantity_line()
        return "\n".join([fit.report(), self._rate_accounting_line(fit),
                          self._rate_quantity_line()])

    def _rate_fit_curve(self, fit: Optional[KineticFitResult]
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """`(|q|, quantity in °C)` along the fitted line, or None.

        Generated by sweeping TEMPERATURE and asking `RATE_MODEL_Q_OF_T` which
        rate this fit assigns to it — see that table for why that direction and
        not the other. The curve and the annotation's numbers come from the same
        `slope` and `intercept`, so the line a user reads and the parameters
        printed beside it cannot disagree.

        None whenever there is nothing honest to draw: no fit, a model with no
        entry in the table, fewer than two fitted rows, or a fit so flat that
        every sampled rate lands outside `RATE_CURVE_Q_PAD` of the measured range.
        A missing line is a visible absence; a line drawn off the end of the axis
        is a lie about a fit that exists.
        """
        if fit is None or not fit.ok:
            return None
        q_of_T = RATE_MODEL_Q_OF_T.get(str(fit.method))
        if q_of_T is None:
            return None
        rows = [r for r in (self.rate_series.get(u) for u in fit.rows_used)
                if r is not None]
        values = [float(r.value_C) for r in rows
                  if np.isfinite(_nan_float(r.value_C))]
        rates = [abs(float(r.q_C_per_min)) for r in rows
                 if np.isfinite(_nan_float(r.q_C_per_min))
                 and float(r.q_C_per_min) != 0.0]
        if len(values) < 2 or len(rates) < 2:
            return None
        t_lo, t_hi = min(values), max(values)
        pad = max(0.08 * (t_hi - t_lo), 1.0)
        grid = np.linspace(t_lo - pad, t_hi + pad, RATE_CURVE_POINTS)
        T0 = _nan_float(fit.extras.get("T0_mean_C"))
        with np.errstate(all="ignore"):
            try:
                q = np.asarray(q_of_T(grid, float(fit.slope), float(fit.intercept),
                                      T0), dtype=float)
            except Exception:                            # pragma: no cover
                return None
        q_lo = min(rates) / RATE_CURVE_Q_PAD
        q_hi = max(rates) * RATE_CURVE_Q_PAD
        keep = np.isfinite(q) & (q > 0.0) & (q >= q_lo) & (q <= q_hi)
        if int(np.count_nonzero(keep)) < 2:
            return None
        qs, ts = q[keep], grid[keep]
        order = np.argsort(qs)
        return qs[order], ts[order]

    def _rate_draw(self) -> None:
        """Draw the series, the fitted line and the annotation, in that order.

        Every row is on the figure the moment it exists, which is the point: five
        files at 5/10/15/20/30 °C/min show their lever arm before a single value
        has been measured, and a row with no value is a RUG TICK at its own rate
        rather than an absence. A row that cannot be seen is a row that gets
        forgotten in a fit it was never in — so a failed row gets a tick in the
        warning tone, an excluded row with a value is GREYED and never hidden
        (unless the user unticks "Show excluded", which itself raises
        `hidden_excluded`), and the annotation names every row not in the fit.

        |q| on the abscissa: the fit uses the magnitude, a cooling row's negative
        sign is carried by its `cooling` flag and its warning, and a log axis
        could not show it anyway. It is also the one coordinate EVERY row has,
        with or without a value, which is what lets the unmeasured rows be drawn
        at all — a model's own 1/T abscissa exists only for a row that already
        carries a number, and drawing in model space would therefore make exactly
        the rows that need watching invisible.

        `cancel_pending()` first, always (R6): `PlotWidget`'s own docstring says a
        queued render fires up to 120 ms later and would overwrite this one.
        """
        if not hasattr(self, "rate_plot"):
            # The tab is built before the right-hand panel exists, so the first
            # refresh runs without a canvas. Nothing is lost: entering the tab
            # draws it.
            return
        self.rate_plot.cancel_pending()
        ax = self.rate_plot.ax
        ax.clear()
        ax.grid(alpha=0.25)
        rows = list(self.rate_series.rows)
        unit = self._temp_unit()
        kind = self.rate_series.quantity_kind or "Tg"
        ax.set_xlabel("|heating rate|   (°C/min)")
        ax.set_ylabel(f"{kind}   ({unit})")
        if not rows:
            ax.text(0.5, 0.5,
                    "No rows yet.\nAdd the heating segments of this record, or add "
                    "files measured at different rates.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            self.rate_plot.figure.tight_layout()
            self.rate_plot.canvas.draw_idle()
            return

        show_excluded = self.rate_show_excluded_check.isChecked()
        show_labels = self.rate_show_labels_check.isChecked()
        used_x: List[float] = []
        included: Tuple[List[float], List[float]] = ([], [])
        excluded: Tuple[List[float], List[float]] = ([], [])
        no_value: List[float] = []
        failed: List[float] = []
        labelled: List[Tuple[float, float, str]] = []
        no_rate = 0
        for row in rows:
            q = _nan_float(row.q_C_per_min)
            if not np.isfinite(q) or q == 0.0:
                no_rate += 1
                continue
            x = abs(float(q))
            used_x.append(x)
            v = _nan_float(row.value_C)
            if np.isfinite(v):
                y = float(_shift_temperature(float(v), "°C", unit))
                if row.include:
                    included[0].append(x)
                    included[1].append(y)
                    labelled.append((x, y, self._rate_row_label(row)))
                elif show_excluded:
                    excluded[0].append(x)
                    excluded[1].append(y)
                    labelled.append((x, y, self._rate_row_label(row)))
            elif row.value_source == "failed":
                failed.append(x)
            else:
                no_value.append(x)

        if included[0]:
            ax.plot(included[0], included[1], linestyle="none", marker="o",
                    label=f"{kind} value")
        if excluded[0]:
            ax.plot(excluded[0], excluded[1], linestyle="none", marker="x",
                    color="0.55", alpha=0.6, label="excluded")
        for i, x in enumerate(no_value):
            ax.plot([x], [0], marker="|", markersize=12, color="0.55",
                    transform=ax.get_xaxis_transform(), clip_on=False,
                    label=("no value yet" if i == 0 else None))
        for i, x in enumerate(failed):
            ax.plot([x], [0], marker="|", markersize=12, color=_ROW_WARN_FG,
                    transform=ax.get_xaxis_transform(), clip_on=False,
                    label=("failed" if i == 0 else None))

        curve = self._rate_fit_curve(self.rate_last_fit)
        rising = bool(len(included[1]) >= 2 and included[1][-1] >= included[1][0])
        if curve is not None:
            fit = self.rate_last_fit
            qs, ts = curve
            ys = np.asarray([_shift_temperature(float(t), "°C", unit) for t in ts],
                            dtype=float)
            name = str(fit.method_label or fit.method).split("  ")[0]
            if fit.r2 is not None and np.isfinite(fit.r2):
                name += f"  (R² = {fit.r2:.3f})"
            ax.plot(qs, ys, linestyle="-", linewidth=1.4, color="#c0392b",
                    label=name)
            rising = bool(ys.size >= 2 and ys[-1] >= ys[0])

        if show_labels:
            for x, y, text in labelled:
                ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color="0.35")

        if used_x and min(used_x) > 0:
            ax.set_xscale("log")

        # The annotation takes the corner the data leaves empty and the "no rate
        # yet" note takes the opposite one, so on a half-measured series the two
        # can never sit on top of each other.
        ann_va, note_va = ("top", "bottom") if rising else ("bottom", "top")
        if no_rate:
            ax.text(0.02, (0.98 if note_va == "top" else 0.02),
                    f"{_plural(no_rate, 'row')} with no rate yet — not on this axis",
                    transform=ax.transAxes, ha="left", va=note_va, fontsize=8,
                    color=_ROW_WARN_FG)
        text = self.rate_annotation_text
        if text:
            warned = bool(self.rate_warnings)
            # facecolor white at alpha 0.8, rather than SPEC_rate_series' bare
            # alpha=0.15: the entire purpose of drawing this INTO the axes is that
            # a saved PNG carries it, and a caveat that cannot be read over the
            # data it sits near has not been carried anywhere. The warning-toned
            # border is the spec's own, and it is what makes a flagged figure
            # recognisable as one at thumbnail size.
            ax.text(0.02, (0.98 if ann_va == "top" else 0.02), text,
                    transform=ax.transAxes, ha="left", va=ann_va, fontsize=8,
                    linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              alpha=0.8,
                              edgecolor=(_ROW_WARN_FG if warned else "0.75"),
                              linewidth=(1.2 if warned else 0.6)))
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")
        self.rate_plot.figure.tight_layout()
        self.rate_plot.canvas.draw_idle()
    # ------------------------------------------------------------------
    # Rate series: export / reload
    # ------------------------------------------------------------------

    def rate_export_rows_csv(self, path) -> int:
        """Write the row table. Returns the number of rows written.

        Takes a path and opens no dialog, so a test exercises the writing rather
        than a monkeypatched `QFileDialog`. The warnings handed to the writer are
        the ones on screen, so the CSV and the banner can never disagree about
        what applies to this series.
        """
        import dta.dta_rate_series as drs
        drs.write_series_csv(self.rate_series, self.rate_warnings, Path(path))
        return len(self.rate_series.rows)

    def rate_export_fit_csv(self, path) -> int:
        """Write one line per evaluable model. Returns how many were written.

        `fit_all` shares ONE `check_series` pass across the models, so the
        Lasocka line and the Kissinger line of the file carry the same warning
        objects and a reader comparing them is comparing models, not two
        independently recomputed opinions about the same rows.
        """
        import dta.dta_rate_series as drs
        fits = drs.fit_all(self.rate_series)
        drs.write_fit_csv(self.rate_series, fits, self.rate_warnings, Path(path))
        return len(fits)

    def rate_save_plot_png(self, path) -> None:
        """Save the rate figure, annotation and all.

        `cancel_pending()` first: `PlotWidget.request_redraw` can fire up to
        120 ms later, and a debounced repaint landing between the draw and the
        write would save a figure without its caveats -- which is the one thing
        drawing them into the axes was for.
        """
        self.rate_plot.cancel_pending()
        self._rate_draw()
        self.rate_plot.figure.savefig(Path(path), dpi=200, bbox_inches="tight")

    def rate_load_rows_csv(self, path):
        """Replace the series with one read back from a rows CSV.

        Values arrive as they were saved and are NOT re-measured: a Tg written to
        a file is a measurement someone made, not a cache this widget may
        silently refresh. The status line says so, and each row keeps its
        `value_source`, so the table still shows which were computed and which
        were typed. A row whose source file has moved keeps its value and gains a
        warning rather than being dropped.
        """
        import dta.dta_rate_series as drs
        series, warns = drs.read_series_csv(Path(path))
        if not series.rows and self.rate_series.rows:
            # THE guard. Replacing the series is what this method is for, but
            # replacing a MEASURED series with an empty one is not replacement,
            # it is deletion — and it used to happen silently, with the status
            # line reading "values are as saved", because `pd.read_csv` does not
            # raise on a non-CSV: it returns a 0-row frame. The Load filter is
            # "CSV (*.csv);;All files (*.*)" and the file a user is most likely
            # to mis-pick is the .txt export sitting in the same folder. Four
            # measurements, a hand-typed value and a hand-written note, gone,
            # with no dialog and no undo.
            #
            # An empty series is still replaceable, so loading a legitimately
            # empty export still works; and nothing is refused — the file was
            # read, it simply held no rows, and that is what gets said.
            note = ('Nothing was loaded from {}: it holds no rate-series rows. '
                    'The {} already in the table {} untouched.'
                    ).format(Path(path).name, _plural(len(self.rate_series.rows), "row"),
                             "is" if len(self.rate_series.rows) == 1 else "are")
            self._rate_loaded_warnings = list(warns)
            self._rate_update_warnings()
            self._rate_set_io_note(note)
            self._set_status(note)
            return 0, list(warns)
        self.rate_series = series
        self.rate_last_fit = None
        self._rate_loaded_warnings = list(warns)
        self._rate_after_change()
        # Said here rather than in the dialog wrapper: "these numbers were not
        # re-measured" is a property of the ACTION, and a caller reaching the
        # method directly (a script, a test, a future batch reload) needs to have
        # told the user as much just as much as a caller who clicked a button.
        note = ('Loaded {} from {}; values are as saved. Press "Compute all '
                'values" to re-measure them with the current settings.'
                ).format(_plural(len(series.rows), "row"), Path(path).name)
        self._rate_set_io_note(note)
        self._set_status(note)
        return len(series.rows), list(warns)

    def _on_rate_export_rows(self) -> None:
        """Dialog wrapper over `rate_export_rows_csv`."""
        if not self.rate_series.rows:
            QMessageBox.critical(self, "Export rows", "The series has no rows yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save series rows as CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            n = self.rate_export_rows_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export rows", str(exc))
            return
        self._rate_set_io_note("Exported {} to {}.".format(_plural(n, "row"), path))
        QMessageBox.information(self, "Export rows",
                                "{} saved:\n{}".format(_plural(n, "row"), path))

    def _on_rate_export_fit(self) -> None:
        """Dialog wrapper over `rate_export_fit_csv`."""
        if not self.rate_series.fitted_rows():
            QMessageBox.critical(
                self, "Export fit",
                'No row has both a rate and a value yet, so there is nothing to '
                'fit. Press "Compute all values", or type values into the Value '
                'column.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fit results as CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            n = self.rate_export_fit_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export fit", str(exc))
            return
        self._rate_set_io_note("Exported {} to {}.".format(_plural(n, "model"), path))
        QMessageBox.information(self, "Export fit",
                                "{} saved:\n{}".format(_plural(n, "model"), path))

    def _on_rate_save_png(self) -> None:
        """Dialog wrapper over `rate_save_plot_png`."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save rate plot as PNG", "", "PNG (*.png)")
        if not path:
            return
        try:
            self.rate_save_plot_png(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save plot", str(exc))
            return
        self._rate_set_io_note("Saved the figure to {}.".format(path))
        QMessageBox.information(self, "Save plot", "Saved:\n{}".format(path))

    def _on_rate_load_csv(self) -> None:
        """Dialog wrapper over `rate_load_rows_csv`."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load series rows from CSV", "", "CSV (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            self.rate_load_rows_csv(path)   # reports what it did itself
        except Exception as exc:
            QMessageBox.critical(self, "Load series", str(exc))
            return

    def _rate_set_io_note(self, text: str) -> None:
        """One place the export/reload group reports what it just did."""
        if hasattr(self, "rate_io_note"):
            self.rate_io_note.setText(text)
    # ------------------------------------------------------------------
    # Rate series: project persistence
    # ------------------------------------------------------------------

    def rate_series_records(self) -> List[Dict[str, Any]]:
        """The rate series as JSON-safe records for `.prism` (project v4).

        A list rather than a single object because one project holding several
        named series is the obvious next ask, and a list that happens to have one
        entry needs no format change to grow. An empty series returns `[]`, so
        `save_project` writes no section at all and a project made without ever
        opening this tab is byte-identical to a v3 one apart from its version.
        """
        if not self.rate_series.rows:
            return []
        return [self.rate_series.to_dict()]

    def restore_rate_series(self, records: Sequence[Any]) -> int:
        """Rebuild the series from `.prism` records. Returns the row count.

        Values are restored AS SAVED and never re-measured. Re-deriving them
        would mean a reopened project could show different Tg values from the one
        that was saved -- the window, signal and method are stored per row for
        exactly this reason.

        Tolerant by design: a malformed or newer record leaves the series empty
        and says so in the status line instead of taking the whole project load
        down with it, since every other workspace has already been restored by
        the time this runs.
        """
        import dta.dta_rate_series as drs
        self.rate_series = drs.RateSeries(quantity_kind="Tg")
        self.rate_last_fit = None
        self._rate_loaded_warnings = []
        status = ""
        for rec in (records or []):
            if not isinstance(rec, dict):
                continue
            try:
                self.rate_series = drs.RateSeries.from_dict(rec)
            except Exception as exc:
                self.rate_series = drs.RateSeries(quantity_kind="Tg")
                status = ("The saved rate series could not be read ({}); the tab "
                          "starts empty.".format(exc))
            break
        n = len(self.rate_series.rows)
        if n and not status:
            status = ('Restored {} in the rate series; values are as saved. Press '
                      '"Compute all values" to re-measure them with the current '
                      'settings.').format(_plural(n, "row"))
        self._rate_after_change()
        if status:
            self._rate_set_io_note(status)
            self._set_status(status)
        return n
