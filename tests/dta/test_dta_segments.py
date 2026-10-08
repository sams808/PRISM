"""Tests for dta/dta_segments.py — segmentation, cycles, alignment, arrays.

Three kinds of test live here and they are deliberately not mixed:

* **Known-truth tests** build their own arrays, so a failure means the code is
  wrong and nothing else. The rate estimator's unbiasedness and the
  coarse-sampling guard are both pinned this way.
* **Fixture tests** run on ``EXAMPLES/DTA_cycles_example.txt``, which is every
  4th row of a real 890 min, 4-cycle TA run. They always run.
* **Full-file tests** behind ``@requires_dta_bm`` pin the same quantities on the
  un-decimated 18996-row original, so the committed fixture cannot drift away
  from the reality it was cut from.

Every number in here was measured on 2026-09-26 with ``py -3.11`` against the
files named above; where a number differs from SPEC_cycles §4.4 the measured
value is used and the difference is called out in a comment next to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.io_universal as iou
from conftest import requires_dta_bm
from dta import dta_program as dp
from dta import dta_segments as ds
from dta.dta_science import compute_derivative

# ---------------------------------------------------------------------------
# Pinned measurements (fixture = EXAMPLES/DTA_cycles_example.txt, 4749 rows)
# ---------------------------------------------------------------------------
FIXTURE_ROWS = 4749
FIXTURE_ROW_STARTS = (0, 346, 448, 903, 1226, 1329, 1892, 2328, 2431, 3212, 3873, 3977)
# SPEC §4.4 wrote 449 for the third boundary; measured here and on the full file
# (whose [::4] decimation reproduces this list exactly) it is 448.
FIXTURE_KINDS = ("heat", "isothermal", "cool") * 4
FIXTURE_CYCLES = (1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
FIXTURE_ROLES = ("heat", "hold_hot", "cool") * 4
FIXTURE_MODE_RATES = (19.954, 0.039, -19.961, 19.965, 0.034, -14.984,
                      14.991, 0.036, -9.979, 9.986, 0.029, -9.980)
FIXTURE_STEPS = ((3,), (4,), (5, 6), (7,), (8,), (9, 10),
                 (11,), (12,), (13, 14), (15,), (16,), (17, 18))
FIXTURE_T_FIRST = 0.003444444
FIXTURE_T_LAST = 890.2292

# Full-resolution BM-0-cycles.txt, 18996 rows.
FULL_ROWS = 18996
FULL_ROW_STARTS = (0, 1381, 1794, 3610, 4903, 5315, 7565, 9309, 9723, 12844, 15492, 15907)
FULL_MODE_RATES = (19.949, 0.037, -19.960, 19.962, 0.036, -14.984,
                   14.991, 0.029, -9.980, 9.988, 0.030, -9.980)
FULL_T_LAST = 890.3708
FULL_ALIGNMENT_COST = 0.669159
PROGRAMMED_TOTAL_MIN = 890.7186


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load(path):
    """(header, time_min, temp_C, heat_flow, DataFrame) from a TA SDT export."""
    header, _signals, df = iou.parse_ta_sdt_txt(path)
    t_col = next(c for c in df.columns if c.lower().startswith("time"))
    T_col = next(c for c in df.columns if c.lower().startswith("temperature ("))
    y_col = next(c for c in df.columns if c.lower().startswith("heat flow"))
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    T = pd.to_numeric(df[T_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    return header, t, T, y, df


def _dta_signal(df):
    """The `Temperature Difference (C)` column — the DTA signal Tg is read off."""
    col = next(c for c in df.columns
               if c.lower().startswith("temperature difference (")
               and "v)" not in c.lower())
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _whole_file_tg_spread(T, y):
    """The three Tg methods on a WHOLE-FILE global sort, and their spread.

    Exactly what the pre-rework widget did: mask, `np.argsort` the temperature
    of the entire record, then run all three constructions over 350-700 C.
    """
    from dta import dta_science as sci
    mask = np.isfinite(T) & np.isfinite(y)
    x, yy = T[mask], y[mask]
    order = np.argsort(x)
    x, yy = x[order], yy[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        double = sci.compute_tg_double_tangent(x, yy, 350.0, 700.0).tg
        parallel = sci.compute_tg_parallel_improved(x, yy, 350.0, 700.0).tg
        deriv = sci.compute_tg_derivative(x, yy, 350.0, 700.0)
    values = [v for v in (double, parallel, deriv) if v is not None and np.isfinite(v)]
    return values, (max(values) - min(values) if len(values) > 1 else float("nan"))


def _program(header):
    """The 19-step OrgMethod program, read through TAHeader's repeated lines."""
    return dp.program_from_header(header, header_lines=getattr(header, "lines", None))


@pytest.fixture(scope="module")
def cycles(request):
    """(program, t, T, heat_flow, SegmentedRun) for the committed fixture.

    Module-scoped: the fixture is 318 KB and the segmentation is deterministic,
    so re-deriving it per test would add a second of I/O for no information.
    """
    path = request.config.rootpath / "EXAMPLES" / "DTA_cycles_example.txt"
    header, t, T, y, df = _load(path)
    program = _program(header)
    run = ds.segment_run(t, T, program=program)
    return program, t, T, y, run


@pytest.fixture(scope="module")
def full_cycles(request):
    """Same, for the un-decimated real run. Only used behind @requires_dta_bm."""
    from conftest import DTA_BM_DIR
    header, t, T, y, df = _load(DTA_BM_DIR / "BM-0-cycles.txt")
    program = _program(header)
    return program, t, T, y, ds.segment_run(t, T, program=program)


def _synthetic_two_cycles():
    """Six-segment program with the answer known by construction.

    heat 25->600 at 20, hold 5 min, cool at 20, heat at 10, hold 5, cool at 10.
    Mirrors the conftest `dta_synthetic_cycles` fixture but is a plain function
    so module-level helpers can use it too.
    """
    knots_t = [0.0, 28.75, 33.75, 62.5, 120.0, 125.0, 182.5]
    knots_T = [25.0, 600.0, 600.0, 25.0, 600.0, 600.0, 25.0]
    t = np.arange(0.0, knots_t[-1] + 1e-9, 0.05)
    T = np.interp(t, knots_t, knots_T)
    rng = np.random.default_rng(20250926)
    return t, T + rng.uniform(-0.02, 0.02, size=t.size)


# ===========================================================================
# 1. estimate_rate  — C4, the guard everything else rests on
# ===========================================================================
def test_estimate_rate_is_unbiased_on_nonuniform_sampling():
    """The windowed LS slope is exact on deliberately uneven sampling; a
    Savitzky-Golay derivative with a fixed `delta` is biased by exactly
    mean(dt)/median(dt) - 1, which is why savgol is not used here."""
    rng = np.random.default_rng(7)
    dt = rng.choice([0.02, 0.10], size=3000, p=[0.7, 0.3])
    t = np.concatenate(([0.0], np.cumsum(dt)))
    T = 25.0 + 20.0 * t                      # exactly 20 C/min, everywhere

    rate = ds.estimate_rate(t, T, window_min=0.5, min_points=5)
    assert np.max(np.abs(rate - 20.0)) < 1e-4
    assert rate.mean() == pytest.approx(20.0, abs=1e-5)

    savgol = pytest.importorskip("scipy.signal").savgol_filter
    biased = savgol(T, 11, 2, deriv=1, delta=float(np.median(dt)))[10:-10]
    predicted = float(np.mean(dt) / np.median(dt) - 1.0)
    assert biased.mean() / 20.0 - 1.0 == pytest.approx(predicted, rel=2e-3)
    assert predicted > 1.0                   # the bias is not subtle: +122 %


@pytest.mark.parametrize("every", [1, 2, 4, 6, 8])
def test_estimate_rate_survives_coarse_sampling(cycles, every):
    """SPEC C4 / MERGED M7, restated as what it actually is: the answer must
    not depend on the sampling.

    12 segments, the same kinds and the same 4 cycles at every sampling from
    0.19 to 1.5 min per point — i.e. from the committed fixture's own grid to
    8x coarser than it, and 33x coarser than the real file. The corresponding
    failure, and its real cause, are pinned by the three
    `test_the_prototype_rate_estimator_*` tests below; this one is the
    invariance the whole decimated-fixture strategy depends on. Rates are
    checked too, because at one decimation the prototype returns the right
    NUMBER of segments on rates that are noise."""
    _program_, t, T, _y, _run = cycles
    run = ds.segment_run(t[::every], T[::every])
    assert len(run) == 12, (f"every={every}: {len(run)} segments, "
                            f"{[s.kind for s in run]}")
    assert [s.kind for s in run] == list(FIXTURE_KINDS)
    assert sorted(run.cycles()) == [1, 2, 3, 4]
    for seg, expected in zip(run, FIXTURE_MODE_RATES):
        assert seg.rate_mode == pytest.approx(expected, abs=0.2), (every, seg.index)
    for hold in run.by_kind(ds.KIND_ISOTHERMAL):
        # The programmed 20 min soaks. The tolerance widens with the sampling
        # because a boundary can only be placed on a sample: at x8 one sample
        # is 1.5 min, and the shortest hold reads 16.5.
        assert 19.4 - 2.0 * every * 0.1875 < hold.duration_min < 22.0, \
            (every, hold.index, hold.duration_min)


def test_estimate_rate_is_conditioned_for_a_time_offset():
    """FAULT 1 of the prefix-sum formulation, and the dangerous one.

    Summing raw t makes `den = N*Sxx - Sx**2` a difference of two numbers whose
    size is set by t, while the answer's size is set by the variance inside the
    window. A time column that merely STARTS at 1e5 min then gives a 95 % slope
    error. Centring both axes removes it: the same ramp reads the same rate at
    any offset."""
    span = np.linspace(0.0, 890.0, 19000)
    T = 25.0 + 20.0 * span
    for offset in (0.0, 1e3, 1e5, 1e6):
        rate = ds.estimate_rate(span + offset, T)
        assert np.max(np.abs(rate - 20.0)) < 1e-4, offset


def test_estimate_rate_is_conditioned_for_a_long_record():
    """FAULT 2: one global prefix sum over a long record loses the window in
    the total. Fitting in overlapping chunks is exact, because the estimator is
    local. Measured: 2.1 C/min of error on 400 000 points in one pass."""
    for n, span in ((19_000, 890.0), (100_000, 5_000.0), (400_000, 20_000.0)):
        t = np.linspace(0.0, span, n)
        rate = ds.estimate_rate(t, 25.0 + 20.0 * t)
        assert np.max(np.abs(rate - 20.0)) < 1e-4, (n, span)

    # The chunked path must agree with a single pass, or the chunking would be
    # changing answers rather than protecting them. They agree to 2.4e-4 C/min
    # on a curved 890 min signal, and the residual is the SINGLE PASS drifting:
    # the disagreement is concentrated in the last few hundred samples, where
    # its prefix sums are longest.
    t = np.linspace(0.0, 890.0, 19000)
    T = 25.0 + 20.0 * t + np.sin(t)
    chunked = ds.estimate_rate(t, T, window_min=0.5)
    one_pass = ds._ls_slope_one_pass(t, T, 0.25, 5)
    assert np.max(np.abs(chunked - one_pass)) < 1e-3
    assert np.max(np.abs(chunked[:100] - one_pass[:100])) < 1e-6


def test_estimate_rate_refuses_a_one_point_window(cycles):
    """FAULT 3, and the source of the spectacular numbers in the bug report.

    When a window holds one point, `den`'s true value is exactly zero, so an
    ABSOLUTE floor of 1e-12 lets pure roundoff through as a slope. Measured on
    a perfectly smooth -10 C/min stretch of the real cooling ramp sampled at
    0.28 min/pt: +99.6 C/min. A line needs two points, and `den` has to clear
    its own cancellation noise; with both tests the same point reads -9.97."""
    _program_, t, T, _y, _run = cycles
    # dt = 0.75 min at x4, so the 0.5 min window holds exactly one point.
    t_d, T_d = t[::4], T[::4]
    window = (np.searchsorted(t_d, t_d + 0.25, "right")
              - np.searchsorted(t_d, t_d - 0.25, "left"))
    assert int(window.max()) == 1

    for min_points in (0, 5):
        rate = ds.estimate_rate(t_d, T_d, min_points=min_points)
        # Nothing in this run moves faster than 20 C/min.
        assert np.max(np.abs(rate)) < 35.0, (min_points, np.max(np.abs(rate)))
        assert np.all(np.isfinite(rate))


@pytest.mark.parametrize("dt", [0.1875, 0.375, 0.75])
def test_min_points_is_worth_a_factor_of_two_in_noise(dt):
    """WHAT THE FLOOR IS ACTUALLY FOR, measured.

    With the three numerical faults fixed, coarse sampling no longer destroys
    the segmentation with or without `rate_min_points` -- SPEC_cycles C4
    credited the floor with a correctness it does not provide. What it does
    provide is smoothing: wherever the time window holds fewer than
    `min_points` samples, widening it in index space halves the scatter of the
    pointwise estimate. On a 20 C/min ramp carrying +/-0.3 C of noise the
    standard deviation is 0.53 / 0.26 / 0.13 C/min with the floor against
    1.22 / 0.54 / 0.31 without it, at dt = 0.19 / 0.38 / 0.75 min."""
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 120.0, dt)
    T = 25.0 + 20.0 * t + rng.normal(0.0, 0.3, t.size)
    guarded = ds.estimate_rate(t, T, min_points=5)
    naked = ds.estimate_rate(t, T, min_points=0)
    assert naked.std() / guarded.std() > 2.0
    assert np.max(np.abs(naked - 20.0)) > 1.5 * np.max(np.abs(guarded - 20.0))
    # Both are still the right rate on average; this is scatter, not bias.
    assert np.median(guarded) == pytest.approx(20.0, abs=0.1)
    assert np.median(naked) == pytest.approx(20.0, abs=0.1)


def test_min_points_is_a_no_op_when_the_window_is_already_full():
    """Where the time window already holds `min_points` samples the widening
    must change nothing at all, bit for bit -- that is what makes it safe to
    leave on for every file."""
    t = np.arange(0.0, 120.0, 0.05)
    T = 25.0 + 20.0 * t + np.random.default_rng(3).normal(0.0, 0.3, t.size)
    assert np.array_equal(ds.estimate_rate(t, T, min_points=5),
                          ds.estimate_rate(t, T, min_points=0))


def _legacy_estimate_rate(time_min, temp_C, *, window_min=0.5, min_points=0):
    """The scratchpad prototype's rate estimator, verbatim.

    Raw `t` in the prefix sums, an ABSOLUTE `|den| > 1e-12` degeneracy floor,
    no minimum point count. Kept here, and only here, as the regression
    witness for the three numerical faults `estimate_rate` fixes.
    """
    t = np.asarray(time_min, float)
    T = np.asarray(temp_C, float)
    n = t.size
    if n < 2:
        return np.zeros(n)
    hw = max(float(window_min), 0.0) / 2.0
    out = np.gradient(T, t)
    if hw <= 0:
        return out
    lo = np.searchsorted(t, t - hw, "left")
    hi = np.searchsorted(t, t + hw, "right")
    z = np.zeros(1)
    c1 = np.concatenate((z, np.cumsum(t)))
    c2 = np.concatenate((z, np.cumsum(t * t)))
    cy = np.concatenate((z, np.cumsum(T)))
    cty = np.concatenate((z, np.cumsum(t * T)))
    cn = np.arange(n + 1, dtype=float)
    N = cn[hi] - cn[lo]
    Sx = c1[hi] - c1[lo]
    Sxx = c2[hi] - c2[lo]
    Sy = cy[hi] - cy[lo]
    Sxy = cty[hi] - cty[lo]
    den = N * Sxx - Sx * Sx
    good = np.abs(den) > 1e-12
    out[good] = (N[good] * Sxy[good] - Sx[good] * Sy[good]) / den[good]
    return out


@pytest.mark.parametrize("every", [2, 4, 6, 8])
def test_the_prototype_rate_estimator_produces_garbage(cycles, every):
    """THE REGRESSION WITNESS for the numerical rework.

    The prototype's formulation, run on the same decimated fixture, returns
    pointwise rates of 976 / 816 / 273 / 236 C/min at x2 / x4 / x6 / x8 on a
    run whose fastest programmed rate is 20, with dozens of points above 50.
    `estimate_rate` stays at 20-23 at every one of them. This is what
    SPEC_cycles C4 observed and attributed to the missing point floor; the
    floor is not what fixes it."""
    _program_, t, T, _y, _run = cycles
    t_d, T_d = t[::every], T[::every]
    legacy = _legacy_estimate_rate(t_d, T_d)
    fixed = ds.estimate_rate(t_d, T_d)
    assert np.max(np.abs(legacy)) > 200.0, np.max(np.abs(legacy))
    assert int(np.count_nonzero(np.abs(legacy) > 50.0)) >= 10
    assert np.max(np.abs(fixed)) < 25.0


@pytest.mark.parametrize("every", [2, 4, 8])
def test_the_prototype_rate_estimator_wrecks_the_segmentation(cycles, monkeypatch,
                                                              every):
    """... and the downstream consequence, which is what a user would see.

    Measured on the decimated fixture with the prototype's estimator:
    8 / 10 / 11 segments instead of 12, the four 19 min holds replaced by
    "holds" of 263 / 153 / 67 min, and the fastest cooling rate it can still
    find is +6.0 / -8.0 C/min where the method said -20, -15 and -10. (At
    x6 it even returns 12 segments, on rates that are pure noise — which is
    why a test that only counted segments would have missed this.)"""
    _program_, t, T, _y, _run = cycles
    t_d, T_d = t[::every], T[::every]
    assert len(ds.segment_run(t_d, T_d)) == 12

    monkeypatch.setattr(ds, "estimate_rate", _legacy_estimate_rate)
    broken = ds.segment_run(t_d, T_d)
    assert len(broken) != 12, len(broken)
    cools = [abs(s.rate_mode) for s in broken.by_kind(ds.KIND_COOL)]
    assert max(cools, default=0.0) < 9.0, cools
    holds = [s.duration_min for s in broken.by_kind(ds.KIND_ISOTHERMAL)]
    assert max(holds, default=0.0) > 22.0, holds


def test_the_prototype_rate_estimator_breaks_on_a_time_offset():
    """The fault that has nothing to do with sampling at all: shift the time
    column by 1e5 min and the prototype reports 1.1 C/min for a 20 C/min ramp.
    No point count, window or threshold would have caught this one."""
    t = np.linspace(0.0, 890.0, 19000)
    T = 25.0 + 20.0 * t
    assert np.max(np.abs(_legacy_estimate_rate(t, T) - 20.0)) < 1e-2
    shifted = _legacy_estimate_rate(t + 1e5, T)
    assert np.max(np.abs(shifted - 20.0)) > 10.0
    assert np.max(np.abs(ds.estimate_rate(t + 1e5, T) - 20.0)) < 1e-4


def test_estimate_rate_guard_is_a_no_op_at_native_sampling(cycles):
    """The guarded and unguarded estimators agree wherever the window was
    already full — the fix adds samples to empty windows, it does not change
    the estimator."""
    _program_, t, T, _y, _run = cycles
    guarded = ds.estimate_rate(t, T, window_min=2.0, min_points=5)
    naked = ds.estimate_rate(t, T, window_min=2.0, min_points=0)
    assert np.allclose(guarded, naked, atol=1e-12)


def test_estimate_rate_widens_in_index_space_not_time_space():
    """A 0.5 min window on 1 min sampling must still fit a line, and must do it
    by taking more POINTS rather than more MINUTES."""
    t = np.arange(0.0, 60.0, 1.0)
    T = 25.0 + 5.0 * t
    naked = ds.estimate_rate(t, T, window_min=0.5, min_points=0)
    guarded = ds.estimate_rate(t, T, window_min=0.5, min_points=5)
    # With no widening every window holds exactly one point, so the estimator
    # has nothing to fit and falls back to plain differences.
    assert np.allclose(guarded[3:-3], 5.0, atol=1e-9)
    assert guarded.size == naked.size == t.size


def test_estimate_rate_never_raises_on_bad_input():
    """A rate estimate is asked for on every record that loads, including the
    broken ones. It reports, it does not raise."""
    assert ds.estimate_rate([], []).size == 0
    assert ds.estimate_rate([1.0], [2.0]).tolist() == [0.0]
    assert ds.estimate_rate([0.0, 1.0], [0.0, 5.0]).size == 2
    assert ds.estimate_rate(np.arange(5.0), np.arange(5.0), window_min=-1.0).size == 5
    assert ds.estimate_rate(np.arange(5.0), np.arange(5.0), min_points="x").size == 5
    mismatched = ds.estimate_rate(np.arange(5.0), np.arange(3.0))
    assert mismatched.size == 5 and np.all(np.isnan(mismatched))

    t = np.array([0.0, 1.0, np.nan, 3.0, 3.0, 4.0])
    T = np.array([0.0, 2.0, 4.0, 6.0, 6.0, 8.0])
    out = ds.estimate_rate(t, T)
    assert out.size == t.size
    assert np.isnan(out[2]) and np.isnan(out[4])      # dropped rows stay NaN
    assert np.all(np.isfinite(out[[0, 1, 3, 5]]))


# ===========================================================================
# 2. clean_time_temperature — C8, the two index spaces
# ===========================================================================
def test_clean_time_temperature_is_the_identity_on_a_real_file(cycles):
    """On every real TA export here nothing is dropped. That is exactly why the
    mapping must be returned and used rather than assumed."""
    _program_, t, T, _y, run = cycles
    assert run.n_rows == FIXTURE_ROWS
    assert run.keep_index.size == FIXTURE_ROWS
    assert np.array_equal(run.keep_index, np.arange(FIXTURE_ROWS))
    assert run.n_dropped_rows == 0


def test_clean_time_temperature_drops_and_reports():
    """Non-finite rows and backwards/duplicate timestamps go; keep_index says
    which original rows survived."""
    t = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 4.0, 5.0, 3.0])
    T = np.array([10.0, 11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0])
    tc, Tc, keep = ds.clean_time_temperature(t, T)
    assert keep.tolist() == [0, 1, 7, 4, 6]        # sorted by time, first wins
    assert np.all(np.diff(tc) > 0)
    assert np.all(np.isfinite(tc)) and np.all(np.isfinite(Tc))
    assert np.array_equal(Tc, T[keep])


def test_clean_time_temperature_maps_rows_back(cycles):
    """C8, the latent bug: inject NaN temperatures and check that
    `row_indices` still returns ORIGINAL positions while `row_start/row_stop`
    stay in cleaned space. With an identity mapping on every shipped file an
    off-by-N here is otherwise invisible."""
    _program_, t, T, _y, _run = cycles
    holes = [10, 500, 1500, 3000, 4500]
    T_holed = T.copy()
    T_holed[holes] = np.nan

    run = ds.segment_run(t, T_holed, program=_program_)
    assert len(run) == 12
    assert run.n_rows == FIXTURE_ROWS
    assert run.keep_index.size == FIXTURE_ROWS - len(holes)
    assert run.n_dropped_rows == len(holes)
    assert not set(holes) & set(run.keep_index.tolist())

    for seg in run:
        rows = run.row_indices(seg)
        assert rows.size == seg.n_points
        assert rows.size == seg.row_stop - seg.row_start
        # The whole point: the two spaces have DIVERGED.
        offset = int(rows[0]) - int(seg.row_start)
        assert 0 <= offset <= len(holes)
        assert np.array_equal(T[rows], run.temp_C[seg.slice])
    assert int(run.row_indices(run.segments[-1])[-1]) == FIXTURE_ROWS - 1
    # And at least one segment really is shifted, or the test proves nothing.
    assert any(int(run.row_indices(s)[0]) != s.row_start for s in run)


def test_row_indices_slices_the_dataframe_correctly(cycles):
    """`SegmentedRun.row_indices` is the only DataFrame API; prove it lands on
    the right rows of the actual table."""
    _program_, t, T, _y, run = cycles
    frame = pd.DataFrame({"t": t, "T": T})
    for seg in run:
        rows = run.row_indices(seg)
        sub = frame.iloc[rows]
        assert float(sub["t"].iloc[0]) == pytest.approx(seg.t_start)
        assert float(sub["t"].iloc[-1]) == pytest.approx(seg.t_end)
        assert float(sub["T"].iloc[0]) == pytest.approx(seg.T_start)


# ===========================================================================
# 3. Segmentation of the committed fixture
# ===========================================================================
def test_segment_run_finds_twelve_segments_on_the_cycles_example(cycles):
    _program_, _t, _T, _y, run = cycles
    assert len(run) == 12
    assert [s.kind for s in run] == list(FIXTURE_KINDS)
    assert [s.role for s in run] == list(FIXTURE_ROLES)
    assert [s.cycle_index for s in run] == list(FIXTURE_CYCLES)
    assert sorted(run.cycles()) == [1, 2, 3, 4]
    assert run.n_cycles == 4
    assert all(len(v) == 3 for v in run.cycles().values())
    assert len(run.by_kind(ds.KIND_HEAT)) == 4
    assert len(run.by_kind(ds.KIND_COOL)) == 4
    assert len(run.by_role(ds.ROLE_HOLD_HOT)) == 4


def test_segments_tile_the_record_exactly(cycles):
    _program_, _t, _T, _y, run = cycles
    assert run.segments[0].row_start == 0
    assert run.segments[-1].row_stop == FIXTURE_ROWS
    assert sum(s.n_points for s in run) == FIXTURE_ROWS
    for a, b in zip(run.segments, run.segments[1:]):
        assert a.row_stop == b.row_start
    assert [s.index for s in run] == list(range(1, 13))


def test_segment_row_boundaries_are_pinned(cycles):
    _program_, _t, _T, _y, run = cycles
    assert tuple(s.row_start for s in run) == FIXTURE_ROW_STARTS
    assert run.segments[-1].row_stop == FIXTURE_ROWS


def test_segment_times_are_pinned(cycles):
    _program_, _t, _T, _y, run = cycles
    assert run.segments[0].t_start == pytest.approx(FIXTURE_T_FIRST, abs=1e-6)
    assert run.segments[-1].t_end == pytest.approx(FIXTURE_T_LAST, abs=1e-4)
    assert run.segments[2].t_start == pytest.approx(83.9792, abs=1e-4)
    assert run.segments[5].t_start == pytest.approx(249.1667, abs=1e-4)


def test_segment_mode_rates_match_the_program(cycles):
    _program_, _t, _T, _y, run = cycles
    got = [s.rate_mode for s in run]
    for value, expected in zip(got, FIXTURE_MODE_RATES):
        assert value == pytest.approx(expected, abs=0.05)
    # The four heats and the four cools recover their programmed magnitudes.
    assert [round(abs(s.rate_mode)) for s in run.by_kind(ds.KIND_HEAT)] == [20, 20, 15, 10]
    assert [round(abs(s.rate_mode)) for s in run.by_kind(ds.KIND_COOL)] == [20, 15, 10, 10]


def test_mode_rate_beats_mean_and_median_on_a_decaying_cool(cycles):
    """C3/C4: three estimators, and only one of them is usable here. On the
    first cooling segment the mean is wrecked by the passive tail and the median
    merely wounded; the mode recovers the programmed -20 C/min."""
    _program_, _t, _T, _y, run = cycles
    cool = run.segments[2]
    assert cool.rate_mean == pytest.approx(-14.15, abs=0.5)
    assert cool.rate_median == pytest.approx(-17.9, abs=0.5)
    assert cool.rate_mode == pytest.approx(-19.96, abs=0.1)
    assert abs(cool.rate_mode + 20.0) < 0.5
    assert abs(cool.rate_mean + 20.0) > 4.0
    assert abs(cool.rate_median + 20.0) > 1.0


def test_every_ramp_segment_is_monotone(cycles):
    """The property that makes a per-segment sort a no-op and the global sort a
    bug. `n_reversals` is measured against the segment's OWN direction, so a
    perfectly monotone COOL scores zero too."""
    _program_, _t, _T, _y, run = cycles
    for seg in run:
        if seg.is_ramp:
            assert seg.monotone_fraction > 0.99, seg
            assert seg.n_reversals <= 5, seg
            assert seg.n_duplicate_T == 0, seg
        else:
            assert seg.monotone_fraction < 1.0, seg
            assert seg.n_reversals >= 4, seg
    assert all(s.monotone_fraction == 1.0 for s in run.by_kind(ds.KIND_COOL))


def test_only_the_last_segment_is_partial(cycles):
    _program_, _t, _T, _y, run = cycles
    assert [s.is_partial for s in run] == [False] * 11 + [True]


def test_rate_fit_r2_separates_a_held_ramp_from_a_coasting_one(cycles):
    """R2 answers "did the furnace hold the rate", which no single rate can."""
    _program_, _t, _T, _y, run = cycles
    for seg in run.by_kind(ds.KIND_HEAT):
        assert seg.rate_fit_r2 > 0.999, seg
    for seg in run.by_kind(ds.KIND_COOL):
        assert 0.90 < seg.rate_fit_r2 < 0.999, seg     # 0.9598 .. 0.9913


# ===========================================================================
# 4. C5 — the flat-core hold test
# ===========================================================================
def test_flat_core_separates_a_hold_from_a_slow_cooling_tail(cycles):
    """C5. Measured on the fixture: 12.94-15.00 min of <=1 C flat core inside
    the four holds, 0.19-0.75 min inside all eight ramps. A 17x margin across
    an empty gap."""
    _program_, _t, _T, _y, run = cycles
    holds, ramps = [], []
    for seg in run:
        core = ds._longest_flat_core_min(run.time_min[seg.slice],
                                         run.temp_C[seg.slice], 1.0)
        (holds if seg.is_hold else ramps).append(core)
    assert len(holds) == 4 and len(ramps) == 8
    assert min(holds) >= 12.0, holds
    assert max(ramps) <= 1.0, ramps
    assert min(holds) / max(ramps) > 15.0


def test_holds_are_not_separable_by_a_rate_threshold(cycles):
    """C5, the negative half: why `iso_tol` is NOT a knob. The |dT/dt| range
    inside every hold overlaps the range inside the ramps, so no pointwise
    threshold — with or without hysteresis — can tell them apart."""
    _program_, _t, _T, _y, run = cycles
    hold_max = []
    ramp_min = []
    for seg in run:
        r = np.abs(run.rate_C_min[seg.slice])
        (hold_max if seg.is_hold else ramp_min).append((r.min(), r.max()))
    worst_hold = max(hi for _lo, hi in hold_max)
    slowest_ramp = min(lo for lo, _hi in ramp_min)
    assert slowest_ramp < worst_hold, (slowest_ramp, worst_hold)
    # ... and every hold's floor is below every ramp's floor, so the overlap is
    # not a one-sided artefact of a single bad ramp either.
    assert min(lo for lo, _hi in hold_max) < slowest_ramp


def test_segmentation_is_insensitive_to_the_rate_threshold(cycles):
    """C5: segmentation is constant over r_on in 1.0-8.0 C/min, which is what a
    knob would be tuning inside. Boundaries move by at most a few rows."""
    _program_, t, T, _y, run = cycles
    for thr in (1.0, 2.0, 3.25, 5.0, 8.0):
        other = ds.segment_run(t, T, program=_program_,
                               config=ds.SegmentationConfig(rate_threshold_C_min=thr))
        assert len(other) == 12, thr
        assert [s.kind for s in other] == list(FIXTURE_KINDS), thr
        assert [s.cycle_index for s in other] == list(FIXTURE_CYCLES), thr
        shifts = [abs(a.row_start - b.row_start)
                  for a, b in zip(other.segments, run.segments)]
        assert max(shifts) <= 10, (thr, shifts)


# ===========================================================================
# 5. C1 — the program is a labelling layer, not the segmenter
# ===========================================================================
def test_segmentation_works_without_any_program(cycles):
    """C1. Program-free segmentation gives the same 12 segments, the same
    kinds, the same 4 cycles, with boundaries within 2 rows."""
    _program_, t, T, _y, run = cycles
    bare = ds.segment_run(t, T, program=None)
    assert len(bare) == 12
    assert [s.kind for s in bare] == [s.kind for s in run]
    assert [s.role for s in bare] == [s.role for s in run]
    assert [s.cycle_index for s in bare] == [s.cycle_index for s in run]
    shifts = [abs(a.row_start - b.row_start) for a, b in zip(bare.segments, run.segments)]
    assert max(shifts) <= 2, shifts
    assert bare.alignment.method == "no-program"
    assert bare.program is None


def test_rate_nominal_is_nan_without_a_program(cycles):
    """MERGED M3's retargeted guard: an unaligned segment must not pretend to
    know a programmed rate, because `q_programmed_C_per_min` in the rate series
    reads straight off this field."""
    _program_, t, T, _y, _run = cycles
    bare = ds.segment_run(t, T, program=None)
    assert all(np.isnan(s.rate_nominal_C_min) for s in bare)
    assert all(np.isnan(s.target_gap_C) for s in bare)
    assert all(np.isnan(s.control_lost_below_C) for s in bare)
    assert all(s.step_indices == () for s in bare)


def test_measured_and_programmed_rates_are_different_fields(cycles):
    """C3's rename, pinned. `rate_mode` is MEASURED (19.954, not 20.000);
    `rate_nominal_C_min` is PROGRAMMED (exactly 20.0). Shipping one name for
    both is what would put a measured rate on an Arrhenius axis."""
    _program_, _t, _T, _y, run = cycles
    heat = run.segments[0]
    assert heat.rate_nominal_C_min == 20.0
    assert heat.rate_mode != 20.0
    assert heat.rate_mode == pytest.approx(19.954, abs=0.01)
    cool = run.segments[2]
    assert cool.rate_nominal_C_min == -20.0      # signed by travel direction
    assert cool.rate_mode == pytest.approx(-19.961, abs=0.01)
    assert not hasattr(heat, "rate_programmed")


# ===========================================================================
# 6. Alignment
# ===========================================================================
def test_alignment_is_many_to_one_and_absorbs_the_invisible_holds(cycles):
    """The headline alignment claim. Each cooling segment consumes the
    programmed `Ramp to 50 C` AND the `Isothermal 20 min` that follows it,
    because the furnace was still coasting through the hold."""
    _program_, _t, _T, _y, run = cycles
    assert tuple(s.step_indices for s in run) == FIXTURE_STEPS
    assert run.alignment.unmatched_steps == ()
    assert run.alignment.unmatched_segments == ()
    assert run.alignment.method == "dp"
    assert run.alignment.ok is True
    for seg, expected in zip(run, (None, None, (6,), None, None, (10,),
                                   None, None, (14,), None, None, (18,))):
        match = run.alignment.match_for(seg.index)
        assert match is not None
        assert match.absorbed_step_indices == (expected or ())
    assert run.alignment.absorbed_steps == (6, 10, 14, 18)


def test_alignment_says_absorbed_not_not_executed(cycles):
    """MERGED/C1 wording: those four holds RAN. A note that called them "not
    executed" would be a different and false claim about the experiment."""
    _program_, _t, _T, _y, run = cycles
    blob = " ".join(run.alignment.notes).lower()
    assert "absorbed" in blob
    assert "not executed" not in blob
    assert "never executed" not in blob
    assert "absorbed step(s) 6" in run.segments[2].note


def test_alignment_reports_the_missed_cool_targets(cycles):
    """The `Ramp to 50 C` steps never reached 50 C. `target_gap_C` is the
    proof, and it travels on the segment."""
    _program_, _t, _T, _y, run = cycles
    gaps = [s.target_gap_C for s in run.by_kind(ds.KIND_COOL)]
    for value, expected in zip(gaps, (44.4, 27.3, 9.9, 12.1)):
        assert value == pytest.approx(expected, abs=1.0)
    assert all(np.isfinite(g) and g > 0 for g in gaps)
    for seg in run.by_kind(ds.KIND_COOL):
        assert "short of its 50 °C target" in seg.note
    for seg in run.by_kind(ds.KIND_ISOTHERMAL):
        assert seg.target_gap_C == 0.0


def test_alignment_clock_matches_the_program(cycles):
    """The run did NOT end early: the programmed clock is 890.72 min and the
    record is 890.23 min of it (890.37 on the full file)."""
    _program_, _t, _T, _y, run = cycles
    assert run.alignment.programmed_total_min == pytest.approx(PROGRAMMED_TOTAL_MIN, abs=0.01)
    assert run.alignment.measured_total_min == pytest.approx(FIXTURE_T_LAST, abs=1e-3)
    assert run.alignment.programmed_total_min > run.alignment.measured_total_min


def test_control_lost_below_is_reported_on_cooling(cycles):
    """Finite on all four cools, NaN on every heat and every hold.

    A ramp decelerating into the hold that follows it is the setpoint arriving,
    not control being lost: measured, those corners are 0.5-1.2 % of the ramp
    while the four real cooling tails are 25.7-49.4 %."""
    _program_, _t, _T, _y, run = cycles
    for seg in run.by_kind(ds.KIND_COOL):
        assert np.isfinite(seg.control_lost_below_C), seg
        assert "rate control lost below" in seg.note
    for seg in run.by_kind(ds.KIND_HEAT):
        assert np.isnan(seg.control_lost_below_C), seg
    for seg in run.by_kind(ds.KIND_ISOTHERMAL):
        assert np.isnan(seg.control_lost_below_C), seg
    first_cool = run.segments[2]
    assert 400.0 < first_cool.control_lost_below_C < 550.0
    lost = [s.control_lost_below_C for s in run.by_kind(ds.KIND_COOL)]
    assert lost == sorted(lost, reverse=True)     # each cycle loses control later


def test_alignment_never_raises_on_the_wrong_program(cycles):
    """Load a 2-step method against a 12-segment run: report, never raise, and
    keep the matches that were found."""
    _program_, _t, _T, _y, run = cycles
    wrong = dp.parse_org_method_lines(["1: Ramp 10.00 C/min to 1300.00 C",
                                       "2: Isothermal for 5.00 min"])
    align = ds.align_segments_to_program(run.segments, wrong)
    assert align.method == "dp"
    assert align.ok is False
    assert align.unmatched_segments                      # 11 of the 12
    assert len(align.matches) >= 1
    assert np.isfinite(align.total_cost)
    assert any("no counterpart" in n for n in align.notes)


def test_alignment_with_a_program_that_has_no_thermal_steps(cycles):
    _program_, _t, _T, _y, run = cycles
    gas_only = dp.parse_org_method_lines(["1: Select gas 1"])
    align = ds.align_segments_to_program(run.segments, gas_only)
    assert align.method == "no-program"
    assert align.ok is False
    assert any("no thermal steps" in n for n in align.notes)
    assert align.matches == ()


def test_alignment_of_an_empty_segment_list():
    program = dp.parse_org_method_lines(["1: Ramp 10.00 C/min to 500.00 C"])
    align = ds.align_segments_to_program([], program)
    assert align.method == "no-program"
    assert align.matches == ()
    assert align.unmatched_steps == (1,)


def test_repeat_program_aligns_to_the_expanded_steps():
    """C10: a written `Repeat segment` is unrolled before alignment, because
    the written steps and the executed steps differ and the data saw the
    executed ones."""
    program = dp.parse_org_method_lines([
        "1: Ramp 20.00 C/min to 600.00 C",
        "2: Isothermal for 5.00 min",
        "3: Ramp 20.00 C/min to 25.00 C",
        "4: Repeat segment 1 1 times",
        "5: End of method",
    ])
    physical, _notes = program.physical_steps()
    assert [s.index for s in physical if s.is_thermal] == [1, 2, 3, 1, 2, 3]

    knots_t = [0.0, 28.75, 33.75, 62.5, 91.25, 96.25, 125.0]
    knots_T = [25.0, 600.0, 600.0, 25.0, 600.0, 600.0, 25.0]
    t = np.arange(0.0, knots_t[-1] + 1e-9, 0.05)
    T = np.interp(t, knots_t, knots_T)
    run = ds.segment_run(t, T, program=program)
    assert len(run) == 6
    assert [s.step_indices for s in run] == [(1,), (2,), (3,), (1,), (2,), (3,)]
    assert run.alignment.unmatched_steps == ()


def test_attach_alignment_without_arrays_leaves_control_lost_nan(cycles):
    """`control_lost_below_C` needs the pointwise rate, which the alignment does
    not carry. A caller that does not supply it gets NaN, not a guess."""
    _program_, t, T, _y, _run = cycles
    segments = ds.segment_temperature_program(t, T, program=_program_)
    align = ds.align_segments_to_program(segments, _program_)
    bare = ds.attach_alignment(segments, align)
    assert all(np.isnan(s.control_lost_below_C) for s in bare)
    assert all(np.isfinite(s.rate_nominal_C_min) for s in bare)


# ===========================================================================
# 7. prepare_segment_xy — C6 / C7
# ===========================================================================
def test_prepare_xy_is_identity_on_a_single_monotone_ramp(request):
    """BACKWARD-COMPATIBILITY KEYSTONE.

    On EXAMPLES/DTA_example.txt the record is ONE heating ramp and
    `np.argsort(T)` is the identity permutation, so slice + stable sort +
    duplicate collapse are all no-ops and the reworked arrays are BIT-IDENTICAL
    to the pre-rework `mask -> argsort` ones. This is what protects the pinned
    Tg value 354.4666531002641 through the Qt rework: if this test fails, the
    rework changed a number it had no business changing."""
    path = request.config.rootpath / "EXAMPLES" / "DTA_example.txt"
    _header, t, T, y, _df = _load(path)

    # Exactly what dta/qt_dta.py::_get_xy did before the rework.
    mask = np.isfinite(T) & np.isfinite(y)
    x_old, y_old = T[mask], y[mask]
    order = np.argsort(x_old)
    x_old, y_old = x_old[order], y_old[order]
    assert np.array_equal(order, np.arange(order.size))     # the identity

    run = ds.segment_run(t, T)
    assert len(run) == 1
    seg = run.segments[0]
    assert seg.kind == ds.KIND_HEAT
    assert (seg.row_start, seg.row_stop) == (0, 2221)

    prepared = ds.prepare_segment_xy(t, T, y, segment=seg,
                                     keep_index=run.keep_index, x_kind=ds.X_TEMPERATURE)
    assert np.array_equal(prepared.x, x_old)
    assert np.array_equal(prepared.y, y_old)
    assert prepared.was_sorted is True
    assert prepared.n_duplicates_collapsed == 0
    assert np.array_equal(prepared.source_index, np.flatnonzero(mask))

    # ... and the convenience wrapper gives the same arrays.
    via_run = run.xy(seg, y, x_kind=ds.X_TEMPERATURE)
    assert np.array_equal(via_run.x, x_old)
    assert np.array_equal(via_run.y, y_old)


def test_prepare_xy_collapses_duplicate_temperatures(cycles):
    """Unconditional duplicate-x collapse, AVERAGING y — the pre-rework code
    de-duplicated in one branch of one function and kept the first y."""
    _program_, t, T, y, run = cycles
    seg = run.segments[0]
    T_dup, y_dup = T.copy(), y.copy()
    T_dup[5] = T_dup[4]
    y_dup[4], y_dup[5] = 1.0, 3.0

    prepared = ds.prepare_segment_xy(t, T_dup, y_dup, segment=seg,
                                     keep_index=run.keep_index)
    assert prepared.n_duplicates_collapsed == 1
    assert prepared.x.size == seg.n_points - 1
    assert bool(np.all(np.diff(prepared.x) > 0))
    assert prepared.y[4] == pytest.approx(2.0)          # the MEAN, not the first


def test_prepare_xy_refuses_a_flood_of_duplicates(cycles):
    """More than `max_duplicate_frac` collapsed means the abscissa or the scope
    is wrong, and a quietly halved array is worse than an exception."""
    _program_, t, T, y, run = cycles
    seg = run.segments[0]
    T_bad = T.copy()
    T_bad[0:200:2] = T_bad[1:200:2]
    with pytest.raises(ds.SegmentScopeError) as err:
        ds.prepare_segment_xy(t, T_bad, y, segment=seg, keep_index=run.keep_index)
    assert "share an x value" in str(err.value)


def test_prepare_xy_refuses_temperature_on_a_hold(cycles):
    """C6: temperature is not an abscissa on a hold. Time is."""
    _program_, t, T, y, run = cycles
    hold = run.segments[1]
    assert hold.is_hold
    with pytest.raises(ds.SegmentScopeError) as err:
        ds.prepare_segment_xy(t, T, y, segment=hold, keep_index=run.keep_index,
                              x_kind=ds.X_TEMPERATURE)
    assert "not an abscissa on a hold" in str(err.value)

    ok = ds.prepare_segment_xy(t, T, y, segment=hold, keep_index=run.keep_index,
                               x_kind=ds.X_TIME)
    assert ok.x.size == hold.n_points
    assert ok.was_sorted is False
    assert bool(np.all(np.diff(ok.x) > 0))


def test_prepare_xy_on_a_hold_time_axis_is_finite(cycles):
    """The measured consequence of C6. Sorting a hold by temperature leaves
    4-24 non-finite derivative values (an `inf` among them); the time axis
    leaves zero."""
    _program_, t, T, y, run = cycles
    bad_total = 0
    for hold in run.by_kind(ds.KIND_ISOTHERMAL):
        rows = run.row_indices(hold)
        order = np.argsort(T[rows], kind="stable")
        with np.errstate(divide="ignore", invalid="ignore"):
            sorted_deriv = compute_derivative(y[rows][order], T[rows][order])
        bad_total += int(np.count_nonzero(~np.isfinite(sorted_deriv)))

        prepared = run.xy(hold, y, x_kind=ds.X_TIME)
        good = compute_derivative(prepared.y, prepared.x)
        assert int(np.count_nonzero(~np.isfinite(good))) == 0, hold
    assert bad_total > 20, bad_total          # measured 54 across the four holds


def test_prepare_xy_returns_ascending_x_on_a_cooling_segment(cycles):
    """A cool is traversed downwards but prepared ascending, and
    `source_index` keeps every point traceable to its original row."""
    _program_, _t, _T, y, run = cycles
    cool = run.segments[2]
    prepared = run.xy(cool, y, x_kind=ds.X_TEMPERATURE)
    assert prepared.x[0] < prepared.x[-1]
    assert bool(np.all(np.diff(prepared.x) > 0))
    assert prepared.was_sorted is True
    rows = run.row_indices(cool)
    assert int(prepared.source_index[0]) == int(rows[-1])
    assert int(prepared.source_index[-1]) == int(rows[0])
    assert np.array_equal(prepared.y, y[prepared.source_index])


def test_prepare_xy_time_axis_is_not_sorted(cycles):
    _program_, _t, _T, y, run = cycles
    cool = run.segments[2]
    prepared = run.xy(cool, y, x_kind=ds.X_TIME)
    assert prepared.was_sorted is False
    assert np.array_equal(prepared.source_index, run.row_indices(cool))


def test_prepare_xy_inverts_last(cycles):
    """Invert happens AFTER the sort and the collapse, so a collapsed duplicate
    is the mean of the measured values and not the mean of the negated ones —
    identical here, but the order is what makes it provably so."""
    _program_, _t, _T, y, run = cycles
    seg = run.segments[0]
    plain = run.xy(seg, y)
    flipped = run.xy(seg, y, invert=True)
    assert np.array_equal(flipped.x, plain.x)
    assert np.array_equal(flipped.y, -plain.y)


def test_prepare_xy_masks_y_and_y_alt_together(cycles):
    """One mask across x, y and y_alt: three arrays of different lengths is how
    the pre-rework code could pair a heat-flow value with the wrong temperature."""
    _program_, _t, _T, y, run = cycles
    seg = run.segments[0]
    alt = y.copy()
    alt[10] = np.nan
    prepared = run.xy(seg, y, y_alt=alt)
    assert prepared.y.size == prepared.x.size == prepared.y_alt.size
    assert prepared.x.size == seg.n_points - 1
    assert 10 not in prepared.source_index.tolist()


def test_prepare_xy_rejects_an_array_from_another_record(cycles):
    _program_, _t, _T, _y, run = cycles
    with pytest.raises(ds.SegmentScopeError):
        run.xy(run.segments[0], np.zeros(17))


def test_prepare_xy_rejects_an_unknown_x_kind(cycles):
    _program_, t, T, y, run = cycles
    with pytest.raises(ValueError):
        ds.prepare_segment_xy(t, T, y, segment=run.segments[0],
                              keep_index=run.keep_index, x_kind="wavenumber")


def test_xy_accepts_original_and_cleaned_length_y(cycles):
    """A DataFrame column is original-length; this object's own arrays are
    cleaned-length. Both must give the same curve when a row was dropped."""
    _program_, t, T, y, _run = cycles
    T_holed = T.copy()
    T_holed[100] = np.nan
    run = ds.segment_run(t, T_holed, program=_program_)
    assert run.n_rows != run.keep_index.size
    seg = run.segments[0]
    from_original = run.xy(seg, y)
    from_cleaned = run.xy(seg, y[run.keep_index])
    assert np.array_equal(from_original.x, from_cleaned.x)
    assert np.array_equal(from_original.y, from_cleaned.y)


def test_segment_scope_error_is_a_value_error():
    """The Qt layer catches ValueError and shows the message; the guardrail
    must land in that handler rather than crash the workspace."""
    assert issubclass(ds.SegmentScopeError, ValueError)


def test_global_sort_is_the_bug_we_removed(cycles):
    """The regression witness. A whole-file argsort of the fixture gives 93
    duplicate-temperature pairs and 151 non-finite dHF/dT; per-segment
    preparation gives zero of each."""
    _program_, _t, T, y, run = cycles
    mask = np.isfinite(T) & np.isfinite(y)
    x_global, y_global = T[mask], y[mask]
    order = np.argsort(x_global)
    x_global, y_global = x_global[order], y_global[order]
    assert int(np.count_nonzero(np.diff(x_global) == 0)) == 93
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = compute_derivative(y_global, x_global)
    assert int(np.count_nonzero(~np.isfinite(bad))) == 151

    total = 0
    for seg in run:
        prepared = run.xy(seg, y,
                          x_kind=ds.X_TEMPERATURE if seg.is_ramp else ds.X_TIME)
        total += int(np.count_nonzero(~np.isfinite(
            compute_derivative(prepared.y, prepared.x))))
    assert total == 0


def test_degenerate_tg_collapse_needs_a_1e_2_threshold(request):
    """C12, measured from the science layer so the Qt threshold is not a guess.

    A whole-file global sort of the cyclic fixture makes all three Tg methods
    return the SAME number, 455.951 C, with a spread of 1.2e-4 C — three
    independent constructions agreeing to four decimal places because they all
    landed on one meaningless index. The healthy single-ramp control
    (DTA_example) spreads 3.43 C over the same window.

    So a "methods agree" check needs a degeneracy floor, and where that floor
    goes matters: 1e-2 C catches the collapse with two orders of margin either
    side, while the 1e-6 C that was also proposed misses it entirely."""
    cyclic_path = request.config.rootpath / "EXAMPLES" / "DTA_cycles_example.txt"
    _h, _t, temp, _hf, frame = _load(cyclic_path)
    values, spread = _whole_file_tg_spread(temp, _dta_signal(frame))
    assert len(values) == 3
    assert all(v == pytest.approx(455.951, abs=1e-3) for v in values), values
    assert spread == pytest.approx(1.2e-4, rel=0.2)

    healthy_path = request.config.rootpath / "EXAMPLES" / "DTA_example.txt"
    _h2, _t2, temp2, _y2, frame2 = _load(healthy_path)
    _vals2, healthy = _whole_file_tg_spread(temp2, _dta_signal(frame2))
    assert healthy == pytest.approx(3.43, rel=0.2)

    assert spread < 1e-2 < healthy          # 1e-2 separates them, with margin
    assert not (spread < 1e-6)              # 1e-6 would call the collapse real


def test_default_window_is_the_central_sixty_percent(cycles):
    _program_, _t, _T, _y, run = cycles
    seg = run.segments[0]
    lo, hi = run.default_window(seg)
    assert lo == pytest.approx(273.7, abs=1.0)
    assert hi == pytest.approx(1037.5, abs=1.0)
    assert seg.T_min <= lo < hi <= seg.T_max
    assert (hi - lo) == pytest.approx(0.60 * (seg.T_max - seg.T_min), rel=1e-9)
    wide = run.default_window(seg, central_frac=1.0)
    assert wide == pytest.approx((seg.T_min, seg.T_max))


# ===========================================================================
# 8. Config, cycles, robustness
# ===========================================================================
def test_segmentation_config_validated_widens_only_when_needed():
    """`validated()` is an honest-window helper for a UI, NOT the coarse-sampling
    guard: that lives in `estimate_rate` and widens in index space."""
    cfg = ds.SegmentationConfig()
    assert cfg.validated(0.0459) is cfg                 # native sampling: untouched
    assert cfg.validated(0.1875).rate_window_min == pytest.approx(0.75)
    assert cfg.validated(0.75).rate_window_min == pytest.approx(3.0)
    assert cfg.validated(float("nan")) is cfg
    assert cfg.validated(0.0) is cfg
    assert cfg.validated("nonsense") is cfg
    assert cfg.validated(0.1875).rate_min_points == cfg.rate_min_points


def test_validated_config_still_finds_twelve_segments(cycles):
    """Widening the window in time space is not needed, but it must not break
    anything either."""
    _program_, t, T, _y, _run = cycles
    cfg = ds.SegmentationConfig().validated(float(np.median(np.diff(t))))
    assert cfg.rate_window_min > 0.5
    run = ds.segment_run(t, T, program=_program_, config=cfg)
    assert len(run) == 12
    assert [s.kind for s in run] == list(FIXTURE_KINDS)


def test_group_cycles_starts_a_cycle_at_every_heat_after_a_cool():
    """The convention, on hand-built segments so the rule is visible."""
    def _seg(index, kind):
        return ds.Segment(index=index, kind=kind, role="", cycle_index=0,
                          row_start=index, row_stop=index + 1, n_points=1,
                          t_start=float(index), t_end=float(index) + 1.0,
                          T_start=0.0, T_end=0.0, T_min=0.0, T_max=0.0,
                          rate_mean=0.0, rate_median=0.0, rate_mode=0.0,
                          rate_fit_r2=1.0, control_lost_below_C=float("nan"),
                          monotone_fraction=1.0, n_reversals=0, n_duplicate_T=0,
                          is_partial=False)

    seq = [ds.KIND_EQUILIBRATE, ds.KIND_HEAT, ds.KIND_ISOTHERMAL, ds.KIND_COOL,
           ds.KIND_ISOTHERMAL, ds.KIND_HEAT, ds.KIND_ISOTHERMAL, ds.KIND_COOL]
    grouped = ds.group_cycles([_seg(i, k) for i, k in enumerate(seq)])
    assert [s.cycle_index for s in grouped] == [1, 1, 1, 1, 1, 2, 2, 2]
    assert [s.role for s in grouped] == [
        ds.ROLE_EQUILIBRATE, ds.ROLE_HEAT, ds.ROLE_HOLD_HOT, ds.ROLE_COOL,
        ds.ROLE_HOLD_COLD, ds.ROLE_HEAT, ds.ROLE_HOLD_HOT, ds.ROLE_COOL]
    # A hot plateau belongs to the cycle that PRODUCED it, not the next one.
    assert grouped[2].cycle_index == grouped[1].cycle_index


def test_segment_run_on_synthetic_cycles(dta_synthetic_cycles):
    """Unconditional core coverage with the answer known by construction."""
    t, T = dta_synthetic_cycles
    run = ds.segment_run(t, T)
    assert len(run) == 6
    assert [s.kind for s in run] == [ds.KIND_HEAT, ds.KIND_ISOTHERMAL, ds.KIND_COOL,
                                     ds.KIND_HEAT, ds.KIND_ISOTHERMAL, ds.KIND_COOL]
    assert [s.cycle_index for s in run] == [1, 1, 1, 2, 2, 2]
    assert run.n_cycles == 2
    rates = [s.rate_mode for s in run]
    assert rates[0] == pytest.approx(20.0, abs=0.2)
    assert rates[2] == pytest.approx(-20.0, abs=0.2)
    assert rates[3] == pytest.approx(10.0, abs=0.2)
    assert rates[5] == pytest.approx(-10.0, abs=0.2)
    assert abs(rates[1]) < 0.2 and abs(rates[4]) < 0.2
    assert sum(s.n_points for s in run) == t.size
    assert run.segments[-1].row_stop == t.size


def test_segment_run_is_deterministic(cycles):
    _program_, t, T, _y, run = cycles
    again = ds.segment_run(t, T, program=_program_)
    assert run.table() == again.table()
    assert ds.format_segment_table(run.segments) == ds.format_segment_table(again.segments)


@pytest.mark.parametrize("name", [
    "empty", "one_point", "two_points", "constant", "pure_noise",
    "nan_row", "duplicate_timestamps", "spike", "all_nan", "descending_time",
    "single_value_time",
])
def test_segment_run_edge_cases_never_raise(name):
    """A workspace must survive every file a user can open. Nothing here
    raises; the degenerate cases return one honest `isothermal` segment or, for
    a record with no usable row at all, none."""
    ramp_t = np.arange(2000) * 0.05
    ramp_T = 25.0 + 20.0 * ramp_t
    cases = {
        "empty": (np.array([]), np.array([])),
        "one_point": (np.array([0.0]), np.array([25.0])),
        "two_points": (np.array([0.0, 1.0]), np.array([25.0, 26.0])),
        "constant": (np.arange(100) * 0.05, np.full(100, 500.0)),
        "pure_noise": (np.arange(4000) * 0.05,
                       500.0 + np.random.default_rng(1).normal(0.0, 0.5, 4000)),
        "all_nan": (np.arange(50) * 0.1, np.full(50, np.nan)),
        "descending_time": (np.arange(50, 0, -1) * 0.1, np.arange(50) * 1.0),
        "single_value_time": (np.zeros(20), np.arange(20) * 1.0),
    }
    holed = ramp_T.copy()
    holed[500] = np.nan
    cases["nan_row"] = (ramp_t.copy(), holed)
    dup = ramp_t.copy()
    dup[700] = dup[699]
    cases["duplicate_timestamps"] = (dup, ramp_T.copy())
    spiked = ramp_T.copy()
    spiked[900] = 5000.0
    cases["spike"] = (ramp_t.copy(), spiked)

    t, T = cases[name]
    run = ds.segment_run(t, T)
    assert isinstance(run, ds.SegmentedRun)
    assert run.n_rows == t.size
    if name in ("empty", "all_nan", "single_value_time"):
        expected = 0 if name != "single_value_time" else 1
        assert len(run) == expected
    else:
        assert len(run) >= 1
        assert run.segments[0].row_start == 0
        assert run.segments[-1].row_stop == run.keep_index.size
        assert sum(s.n_points for s in run) == run.keep_index.size
    if name == "pure_noise":
        assert len(run) == 1
        assert run.segments[0].kind == ds.KIND_ISOTHERMAL
    if name == "constant":
        assert len(run) == 1 and run.segments[0].kind == ds.KIND_ISOTHERMAL
    # The reporting surface must survive them too.
    assert isinstance(ds.format_segment_table(run.segments), str)
    assert isinstance(run.describe(), str)
    for seg in run:
        assert isinstance(ds.segment_label(seg), str)
        assert isinstance(ds.describe_segment(seg), str)


def test_a_leading_flat_stretch_becomes_an_equilibrate():
    """A flat run at the START of a record that later ramps is the instrument
    settling, not a programmed soak; it carries no kinetics and must be named
    differently from the soaks that do."""
    t = np.arange(0.0, 60.0, 0.05)
    T = np.where(t < 10.0, 25.0, 25.0 + 20.0 * (t - 10.0))
    run = ds.segment_run(t, T)
    assert len(run) == 2
    assert run.segments[0].kind == ds.KIND_EQUILIBRATE
    assert run.segments[0].role == ds.ROLE_EQUILIBRATE
    assert run.segments[0].is_hold is True
    assert run.segments[1].kind == ds.KIND_HEAT


def test_segment_temperature_program_return_rate_is_aligned(cycles):
    _program_, t, T, _y, run = cycles
    segments, rate = ds.segment_temperature_program(t, T, program=_program_,
                                                    return_rate=True)
    assert rate.size == run.keep_index.size
    assert len(segments) == 12
    assert np.allclose(rate, run.rate_C_min, equal_nan=True)


# ===========================================================================
# 9. Reporting surface
# ===========================================================================
def test_segment_label_reads_like_the_spec(cycles):
    _program_, _t, _T, _y, run = cycles
    label = ds.segment_label(run.segments[2])
    assert label.startswith("#3  cool  -20.0 ")
    assert "1299 -> 94" in label
    assert "cycle 1" in label
    assert "steps 5, 6" in label
    assert "(partial)" in ds.segment_label(run.segments[11])
    assert "cycle" not in ds.segment_label(run.segments[2], show_cycle=False)
    assert "step" not in ds.segment_label(run.segments[2], show_step=False)
    hold = ds.segment_label(run.segments[1])
    assert hold.startswith("#2  isothermal")
    assert "/min" not in hold                 # a hold has no rate to advertise


def test_describe_segment_carries_the_warnings(cycles):
    _program_, _t, _T, _y, run = cycles
    text = ds.describe_segment(run.segments[2])
    assert "cool" in text and "cycle 1" in text
    assert "mode" in text and "mean" in text and "median" in text
    assert "short of its 50 °C target" in text
    assert "absorbed" in text
    assert "rate control lost" in text
    assert text.endswith(".")
    bare = ds.describe_segment(ds.segment_run(np.arange(100) * 0.05,
                                              np.full(100, 500.0)).segments[0])
    assert "no programmed step matched" in bare


def test_segment_table_has_every_column(cycles):
    _program_, _t, _T, _y, run = cycles
    rows = ds.segment_table(run.segments)
    assert len(rows) == 12
    for row in rows:
        assert tuple(row) == ds.SEGMENT_TABLE_COLUMNS
    assert rows[2]["kind"] == "cool"
    assert rows[2]["steps"] == "5,6"
    assert rows[2]["rows"] == "448:903"
    assert rows[0]["rate_nominal"] == 20.0
    assert rows[1]["control_lost_below"] != rows[1]["control_lost_below"]   # NaN
    text = ds.format_segment_table(run.segments)
    assert text.count("\n") == 12
    assert "rate_mode" in text.splitlines()[0]
    assert ds.format_segment_table([]) == "(no segments)"


def test_segment_table_unit_reaches_the_note(cycles):
    _program_, _t, _T, _y, run = cycles
    rows = ds.segment_table(run.segments, unit="K")
    assert "°C" not in rows[2]["note"]
    assert " K" in rows[2]["note"]


def test_segment_to_dict_is_json_safe(cycles):
    import json
    _program_, _t, _T, _y, run = cycles
    payload = [s.to_dict() for s in run]
    text = json.dumps(payload)
    assert "NaN" not in text
    assert json.loads(text)[1]["control_lost_below_C"] is None
    assert json.loads(text)[2]["step_indices"] == [5, 6]


def test_segmented_run_navigation(cycles):
    _program_, _t, _T, _y, run = cycles
    assert len(run) == 12
    assert bool(run) is True
    assert [s.index for s in run] == list(range(1, 13))
    assert run.segment_by_index(3) is run.segments[2]
    assert run.segment_by_index(99) is None
    assert run.by_kind("nonsense") == ()
    assert "12 segments" in run.describe()
    assert "4 cycle" in run.describe()


def test_segment_properties(cycles):
    _program_, _t, _T, _y, run = cycles
    heat, hold, cool = run.segments[0], run.segments[1], run.segments[2]
    assert heat.is_ramp and not heat.is_hold and heat.direction == 1
    assert cool.is_ramp and cool.direction == -1
    assert hold.is_hold and not hold.is_ramp and hold.direction == 0
    assert heat.span_C > 0 and cool.span_C < 0
    assert heat.duration_min == pytest.approx(heat.t_end - heat.t_start)
    assert heat.slice == slice(0, 346)


# ===========================================================================
# 10. Full-resolution pins — skip when the raw folder is absent
# ===========================================================================
@requires_dta_bm
def test_full_cycles_file_segments_into_twelve(full_cycles):
    _program_, _t, _T, _y, run = full_cycles
    assert len(run) == 12
    assert "".join(s.kind[0].upper() for s in run) == "HICHICHICHIC"
    assert sorted(run.cycles()) == [1, 2, 3, 4]
    assert tuple(s.row_start for s in run) == FULL_ROW_STARTS
    assert run.segments[-1].row_stop == FULL_ROWS
    assert sum(s.n_points for s in run) == FULL_ROWS


@requires_dta_bm
def test_full_cycles_file_segment_times(full_cycles):
    _program_, _t, _T, _y, run = full_cycles
    assert run.segments[0].t_start == pytest.approx(0.0034, abs=1e-4)
    assert run.segments[-1].t_end == pytest.approx(FULL_T_LAST, abs=1e-4)
    assert run.segments[2].t_start == pytest.approx(84.0750, abs=1e-4)
    assert run.segments[2].t_end == pytest.approx(169.1500, abs=1e-4)
    assert run.segments[11].t_start == pytest.approx(745.6208, abs=1e-4)


@requires_dta_bm
def test_full_cycles_file_mode_rates(full_cycles):
    _program_, _t, _T, _y, run = full_cycles
    for seg, expected in zip(run, FULL_MODE_RATES):
        assert seg.rate_mode == pytest.approx(expected, abs=0.02), seg.index


@requires_dta_bm
def test_full_cycles_file_hold_durations_are_mutually_consistent(full_cycles):
    """Pins that ramp->hold corner refinement is ON and hold->ramp is OFF: the
    four programmed 20 min holds come out within 0.14 min of each other, against
    1.55 min with the refinement disabled."""
    _program_, t, T, _y, run = full_cycles
    durations = [s.duration_min for s in run.by_kind(ds.KIND_ISOTHERMAL)]
    assert len(durations) == 4
    for value, expected in zip(durations, (19.308, 19.267, 19.358, 19.408)):
        assert value == pytest.approx(expected, abs=0.01)
    assert max(durations) - min(durations) == pytest.approx(0.1415, abs=0.005)

    unrefined = ds.segment_run(t, T, program=_program_,
                               config=ds.SegmentationConfig(refine_boundaries=False))
    raw = [s.duration_min for s in unrefined.by_kind(ds.KIND_ISOTHERMAL)]
    assert max(raw) - min(raw) == pytest.approx(1.5458, abs=0.01)


@requires_dta_bm
def test_full_cycles_file_alignment(full_cycles):
    _program_, _t, _T, _y, run = full_cycles
    align = run.alignment
    assert align.method == "dp"
    assert align.ok is True
    assert align.total_cost == pytest.approx(FULL_ALIGNMENT_COST, abs=1e-3)
    assert align.unmatched_steps == ()
    assert align.unmatched_segments == ()
    assert align.absorbed_steps == (6, 10, 14, 18)
    assert tuple(s.step_indices for s in run) == FIXTURE_STEPS
    assert align.programmed_total_min == pytest.approx(PROGRAMMED_TOTAL_MIN, abs=0.01)
    assert align.measured_total_min == pytest.approx(FULL_T_LAST, abs=1e-4)
    gaps = [s.target_gap_C for s in run.by_kind(ds.KIND_COOL)]
    for value, expected in zip(gaps, (44.36, 27.32, 9.88, 11.81)):
        assert value == pytest.approx(expected, abs=0.05)


@requires_dta_bm
def test_full_cycles_file_control_lost_temperatures(full_cycles):
    _program_, _t, _T, _y, run = full_cycles
    lost = [s.control_lost_below_C for s in run.by_kind(ds.KIND_COOL)]
    for value, expected in zip(lost, (447.6, 344.9, 227.9, 227.2)):
        assert value == pytest.approx(expected, abs=1.0)
    assert all(np.isnan(s.control_lost_below_C) for s in run.by_kind(ds.KIND_HEAT))


@requires_dta_bm
def test_full_cycles_file_global_sort_pathology(full_cycles):
    """The full-resolution version of the bug: 719 duplicate-T pairs and 1002
    non-finite dHF/dT under a whole-file argsort."""
    _program_, _t, T, y, run = full_cycles
    mask = np.isfinite(T) & np.isfinite(y)
    x_global, y_global = T[mask], y[mask]
    order = np.argsort(x_global)
    x_global, y_global = x_global[order], y_global[order]
    assert int(np.count_nonzero(np.diff(x_global) == 0)) == 719
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = compute_derivative(y_global, x_global)
    assert int(np.count_nonzero(~np.isfinite(bad))) == 1002
    window = (x_global >= 350.0) & (x_global <= 700.0)
    assert int(window.sum()) == 4398
    assert int(np.count_nonzero(~np.isfinite(bad[window]))) == 4
    assert float(np.nanmax(np.abs(bad[window]))) == pytest.approx(240881.0, rel=1e-3)

    total = 0
    for seg in run:
        prepared = run.xy(seg, y,
                          x_kind=ds.X_TEMPERATURE if seg.is_ramp else ds.X_TIME)
        total += int(np.count_nonzero(~np.isfinite(
            compute_derivative(prepared.y, prepared.x))))
    assert total == 0


@requires_dta_bm
def test_full_cycles_file_decimation_matches_the_fixture(full_cycles, cycles):
    """The committed fixture is not drifting from reality: an in-test [::4] of
    the full file reproduces its segments boundary for boundary."""
    program, t, T, _y, _run = full_cycles
    _p2, _t2, _T2, _y2, fixture_run = cycles
    decimated = ds.segment_run(t[::4], T[::4], program=program)
    assert [s.row_start for s in decimated] == [s.row_start for s in fixture_run]
    assert [s.kind for s in decimated] == [s.kind for s in fixture_run]
    assert [s.step_indices for s in decimated] == [s.step_indices for s in fixture_run]


@requires_dta_bm
def test_full_cycles_file_survives_every_decimation(full_cycles):
    """The C4 guard on the original sampling grid: 12 segments from x1 to x16,
    i.e. dt 0.046 to 0.75 min per point."""
    program, t, T, _y, _run = full_cycles
    for every in (1, 2, 3, 4, 6, 8, 10, 16):
        run = ds.segment_run(t[::every], T[::every], program=program)
        assert len(run) == 12, (every, len(run))
        assert "".join(s.kind[0].upper() for s in run) == "HICHICHICHIC", every


@requires_dta_bm
def test_savgol_fixed_delta_is_biased_on_the_real_sampling(full_cycles):
    """The measured reason savgol is not the estimator: the real file's dt has
    mean/median - 1 = +2.12 %, and a fixed-`delta` Savitzky-Golay derivative is
    high by exactly that."""
    _program_, t, _T, _y, run = full_cycles
    dt = np.diff(t)
    ratio = float(np.mean(dt) / np.median(dt) - 1.0)
    assert ratio == pytest.approx(0.0212, abs=0.001)

    savgol = pytest.importorskip("scipy.signal").savgol_filter
    seg = run.segments[0]
    windowed = run.rate_C_min[seg.slice]
    biased = savgol(run.temp_C[seg.slice], 21, 2, deriv=1, delta=float(np.median(dt)))
    interior = slice(int(0.1 * windowed.size), int(0.9 * windowed.size))
    assert biased[interior].mean() / windowed[interior].mean() - 1.0 == \
        pytest.approx(ratio, abs=0.002)
    assert windowed[interior].mean() == pytest.approx(20.0, abs=0.05)


@requires_dta_bm
def test_full_cycles_file_flat_core_margin(full_cycles):
    """C5 at full resolution — the measurement the whole hold rule rests on.

    Flat core (<=1 C peak-to-peak): 13.175-15.233 min inside the four holds,
    0.329-0.704 min inside all eight ramps, a 19x margin across an empty gap.
    Pointwise |dT/dt|: 0.000-2.283 C/min inside the holds, while the slowest
    ramp visits 0.003 C/min and the LAST cool never gets below 1.744 C/min --
    so the hold range is a strict subset of the ramp range and no pointwise
    threshold, with or without hysteresis, can separate them."""
    _program_, _t, _T, _y, run = full_cycles
    cores = {seg.index: ds._longest_flat_core_min(run.time_min[seg.slice],
                                                  run.temp_C[seg.slice], 1.0)
             for seg in run}
    hold_cores = [cores[s.index] for s in run.by_kind(ds.KIND_ISOTHERMAL)]
    ramp_cores = [cores[s.index] for s in run if s.is_ramp]
    assert min(hold_cores) == pytest.approx(13.175, abs=0.01)
    assert max(hold_cores) == pytest.approx(15.233, abs=0.01)
    assert min(ramp_cores) == pytest.approx(0.329, abs=0.01)
    assert max(ramp_cores) == pytest.approx(0.704, abs=0.01)
    assert min(hold_cores) / max(ramp_cores) > 18.0

    hold_max = max(float(np.abs(run.rate_C_min[s.slice]).max())
                   for s in run.by_kind(ds.KIND_ISOTHERMAL))
    ramp_min = min(float(np.abs(run.rate_C_min[s.slice]).min())
                   for s in run if s.is_ramp)
    last_cool_min = float(np.abs(run.rate_C_min[run.segments[-1].slice]).min())
    assert hold_max == pytest.approx(2.283, abs=0.01)
    assert ramp_min == pytest.approx(0.003, abs=0.01)
    assert last_cool_min == pytest.approx(1.744, abs=0.01)
    assert ramp_min < hold_max                 # the ranges overlap: no threshold


@requires_dta_bm
def test_full_cycles_file_mode_beats_mean_and_median(full_cycles):
    """The three-estimator claim on the original data: mean -14.155, median
    -18.006, mode -19.960, against a programmed -20."""
    _program_, _t, _T, _y, run = full_cycles
    cool = run.segments[2]
    assert cool.rate_mean == pytest.approx(-14.155, abs=0.01)
    assert cool.rate_median == pytest.approx(-18.006, abs=0.01)
    assert cool.rate_mode == pytest.approx(-19.960, abs=0.01)


@requires_dta_bm
def test_full_cycles_file_segmentation_is_program_free(full_cycles):
    """C1 at full resolution: no program, same 12 segments, same cycles.

    Boundaries move by at most 3 of 18996 rows — 0.142 min — because the only
    thing the program contributes to segmentation is the rate threshold
    (2.5 C/min from the slowest programmed ramp, against 5.0 from p95 of the
    measured rate), and the answer is flat across that whole range. SPEC §4.4
    said 2 rows; measured on the full file it is 3."""
    _program_, t, T, _y, run = full_cycles
    bare = ds.segment_run(t, T, program=None)
    assert len(bare) == 12
    assert [s.kind for s in bare] == [s.kind for s in run]
    assert [s.cycle_index for s in bare] == [s.cycle_index for s in run]
    shifts = [abs(a.row_start - b.row_start) for a, b in zip(bare.segments, run.segments)]
    assert max(shifts) <= 3, shifts
    drift = max(abs(a.t_start - b.t_start) for a, b in zip(bare.segments, run.segments))
    assert drift < 0.15, drift


@requires_dta_bm
def test_full_cycles_file_tg_collapse_value(full_cycles):
    """C12 on the original data: a whole-file global sort makes all three Tg
    methods return 431.758 C with a spread of 3.7e-5, and the app used to call
    that agreement. The decimated fixture reproduces the same pathology at
    455.951 C / 1.2e-4, which is what makes it a legitimate stand-in."""
    _program_, _t, T, _y, _run = full_cycles
    from conftest import DTA_BM_DIR
    _h, _t2, _T2, _hf, frame = _load(DTA_BM_DIR / "BM-0-cycles.txt")
    values, spread = _whole_file_tg_spread(T, _dta_signal(frame))
    assert len(values) == 3
    assert all(v == pytest.approx(431.758, abs=1e-3) for v in values), values
    assert spread == pytest.approx(3.7e-5, rel=0.2)
    assert spread < 1e-2


@requires_dta_bm
def test_bm0_single_ramp_file_segments_into_two(dta_bm_dir):
    """The non-cyclic control file: one heat plus the 5 min soak it ends in."""
    header, t, T, _y, _df = _load(dta_bm_dir / "BM-0.txt")
    program = _program(header)
    run = ds.segment_run(t, T, program=program)
    assert len(run) == 2
    heat, hold = run.segments
    assert (heat.kind, heat.row_start, heat.row_stop) == (ds.KIND_HEAT, 0, 15375)
    assert (hold.kind, hold.row_start, hold.row_stop) == (ds.KIND_ISOTHERMAL, 15375, 15906)
    assert heat.rate_mode == pytest.approx(9.9934, abs=0.01)
    assert heat.is_partial is False
    assert hold.is_partial is True
    assert heat.step_indices == (1,)
    assert hold.step_indices == (2,)
    assert run.n_cycles == 1


@requires_dta_bm
def test_bm0_and_cycles_have_different_sampling(dta_bm_dir, full_cycles):
    """Pins that every window in the config is in MINUTES, never in points: the
    two real files differ by 5.5x in sampling interval."""
    _h, t_single, _T, _y, _df = _load(dta_bm_dir / "BM-0.txt")
    _program_, t_cycles, _T2, _y2, _run = full_cycles
    dt_single = float(np.median(np.diff(t_single)))
    dt_cycles = float(np.median(np.diff(t_cycles)))
    assert dt_single == pytest.approx(0.00833, abs=1e-5)
    assert dt_cycles == pytest.approx(0.04590, abs=1e-5)
    assert dt_cycles / dt_single == pytest.approx(5.51, abs=0.05)


# ===========================================================================
# Adversarial-review regressions (2026-09-26)
# ===========================================================================

def _repeat_program():
    """A four-cycle method written the natural way: one cycle plus a repeat."""
    return dp.parse_org_method_lines([
        "1: Ramp 20.00 °C/min to 1300.00 °C",
        "2: Isothermal for 20.00 min",
        "3: Ramp 20.00 °C/min to 50.00 °C",
        "4: Isothermal for 20.00 min",
        "5: Repeat segment 1 for 3 times",
        "6: End of method",
    ])


def _synthetic_cycles(n_cycles: int, dt: float = 0.2):
    """(t, T) for `n_cycles` of 19 → 1300 → hold → 50 → hold at 20 °C/min."""
    t, T = [0.0], [19.0]

    def ramp(rate, target):
        while (T[-1] < target) if rate > 0 else (T[-1] > target):
            t.append(t[-1] + dt)
            T.append(T[-1] + rate * dt)
        T[-1] = target

    def hold(minutes):
        for _ in range(int(minutes / dt)):
            t.append(t[-1] + dt)
            T.append(T[-1])

    for _ in range(n_cycles):
        ramp(20.0, 1300.0)
        hold(20.0)
        ramp(-20.0, 50.0)
        hold(20.0)
    return np.asarray(t), np.asarray(T)


def test_unmatched_steps_counts_repeat_passes_not_written_indices():
    """A run stopped after one pass of a repeated block must say so.

    `_thermal_physical_steps` unrolls `Repeat segment 1 for 3 times` into
    four passes that all carry written index 1..4, so de-duplicating coverage
    by written index let pass 1 stand in for passes 2-4. Measured before the
    fix: a run containing 1 of 4 programmed passes claimed all 12 physical
    thermal steps, reported `unmatched_steps == ()` and `alignment.ok is
    True` — it told the user the method had run to completion.
    """
    program = _repeat_program()
    physical = ds._thermal_physical_steps(program)
    assert len(physical) == 16
    assert [s.index for s in physical[:8]] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert [s.repeat_pass for s in physical[:8]] == [0, 0, 0, 0, 1, 1, 1, 1]

    full = ds.segment_run(*_synthetic_cycles(4), program=program)
    assert full.alignment.unmatched_steps == ()
    assert full.alignment.ok is True
    assert sum(len(m.step_positions) for m in full.alignment.matches) == 16
    # positions are unique even though the written indices repeat
    positions = [p for m in full.alignment.matches for p in m.step_positions]
    assert sorted(positions) == list(range(16))

    stopped = ds.segment_run(*_synthetic_cycles(1), program=program)
    assert stopped.alignment.unmatched_steps == (1, 2, 3, 4)
    assert stopped.alignment.ok is False
    note = " ".join(stopped.alignment.notes)
    assert "repeated pass" in note
    assert "12 unexecuted step slots" in note


def test_a_program_without_repeats_reports_unmatched_steps_unchanged(cycles):
    """The no-repeat path must be byte-for-byte what it always was."""
    _program, _t, _T, _y, run = cycles
    assert run.alignment.unmatched_steps == ()
    assert run.alignment.ok is True
    for m in run.alignment.matches:
        assert len(m.step_positions) == len(m.step_indices)
        assert m.repeat_passes == (0,) * len(m.step_indices)


def test_clamped_boundary_never_inverts_a_run():
    """The refinement clip must not be able to produce a negative-length run.

    `np.clip(v, lo, hi)` with lo > hi silently returns hi. Two adjacent runs
    spanning fewer than six samples between them give lo = a[0] + 3 above
    hi = b[1] - 3, so the old unconditional clip could set the boundary BELOW
    `a[0]` and hand `_make_segment` an empty slice.
    """
    # window empty: leave the boundary where the state machine put it
    assert ds._clamped_boundary(11, [10, 11, +1], [11, 13, 0]) is None
    # refinement declined (returned the original cut): nothing to write
    assert ds._clamped_boundary(20, [0, 20, +1], [20, 60, 0]) is None
    # a genuine move is clipped into the window, never outside it
    assert ds._clamped_boundary(5, [0, 20, +1], [20, 60, 0]) == 5
    assert ds._clamped_boundary(-100, [0, 20, +1], [20, 60, 0]) == 3
    assert ds._clamped_boundary(10_000, [0, 20, +1], [20, 60, 0]) == 57
    for lo, hi in ((10, 11), (10, 12), (10, 13), (10, 15), (10, 16)):
        cut = ds._clamped_boundary(hi, [10, hi, +1], [hi, hi + 2, 0])
        assert cut is None or 10 <= cut <= hi + 2


def test_segment_run_coerces_an_object_dtype_column_instead_of_raising():
    """"Never raises on data" must survive a non-numeric column.

    Both the `n_rows` count in `segment_run` and the coercion in
    `clean_time_temperature` went straight through
    `np.asarray(x, dtype=float)`, which throws on an object-dtype column, so
    a restarted run that appended a second header block took the workspace
    down with a bare `ValueError: could not convert string to float: 'a'`
    before any of the module's own guards were reached. Junk rows now become
    NaN and travel out through `keep_index`, the channel that already
    reports dropped rows.
    """
    run = ds.segment_run(["a", "b"], ["c", "d"])
    assert isinstance(run, ds.SegmentedRun)
    assert run.segments == ()
    assert run.n_rows == 2

    mixed_t = ["0.0", "1.0", "oops", "3.0", "4.0", "5.0"]
    mixed_T = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    run2 = ds.segment_run(mixed_t, mixed_T)
    assert run2.n_rows == 6
    assert run2.n_dropped_rows == 1                 # only the junk row went
    assert list(run2.keep_index) == [0, 1, 3, 4, 5]
    assert run2.temp_C[2] == 80.0                   # the 'oops' row, not shifted

    for junk in ([], [None, None], [object()], {"a": 1}):
        assert isinstance(ds.segment_run(junk, junk), ds.SegmentedRun)


def test_a_ramp_setpoint_is_judged_after_the_soak_that_follows_it(cycles):
    """A heating ramp that hands over to its soak a few degrees short REACHED
    its setpoint; a cooling ramp that never arrived did not.

    Measured at the ramp's own last point, all four heating ramps of this run
    look like missed setpoints -- 7.83 / 8.26 / 6.75 / 5.10 C below 1300 C on
    the full file -- because a Q600 ramp step hands control to the next step
    slightly early and the isothermal after it finishes the approach (the
    holds top out at 1298.6 C). `target_gap_C <= 5 C` is what the rate series
    turns into `setpoint_reached`, so measured in the wrong place that rule
    fired on 100 % of real heating ramps, and a warning that fires on
    everything tells the user nothing.

    The four cooling ramps are the control: their programmed 20-min hold at
    50 C is ABSORBED into the same segment (the furnace was still drifting
    down through it), so there is no following hold to extend into and they
    stay far short -- which is true, they never got near 50 C.
    """
    _program, _t, _T, _y, run = cycles
    heats = [s for s in run.segments if s.kind == ds.KIND_HEAT]
    cools = [s for s in run.segments if s.kind == ds.KIND_COOL]
    assert len(heats) == 4 and len(cools) == 4

    for s in heats:
        assert s.target_gap_C < 5.0, (
            f"heat segment {s.index} reports a {s.target_gap_C:.2f} C gap; the "
            "soak after it reached the setpoint")
    for s in cools:
        assert s.target_gap_C > 5.0, (
            f"cool segment {s.index} reports only a {s.target_gap_C:.2f} C gap "
            "but the furnace never reached 50 C")

    # The separation is wide, not marginal: worst heat vs best cool.
    assert max(s.target_gap_C for s in heats) * 2 < min(s.target_gap_C for s in cools)


def test_extending_a_ramp_gap_never_changes_the_alignment(cycles):
    """The soak correction is applied AFTER the DP, on purpose.

    The alignment's own cost term keeps using the ramp-end gap, so re-measuring
    cannot move a step from one segment to another -- only the reported number
    changes. Pinning it here means a future edit that folds the correction into
    `_block_info` (where it would feed the cost) has to justify itself.
    """
    _program, _t, _T, _y, run = cycles
    al = run.alignment
    assert al.method == "dp"
    assert al.unmatched_steps == ()
    assert [m.step_indices for m in al.matches] == [
        (3,), (4,), (5, 6), (7,), (8,), (9, 10),
        (11,), (12,), (13, 14), (15,), (16,), (17, 18),
    ]
