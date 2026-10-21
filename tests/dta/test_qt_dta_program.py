"""The Program tab (SPEC_cycles §2.5, §3 Phase B step 9, risk R6).

What this tab exists to prevent is a user believing a number that is true at
the top of a ramp and false at the bottom. So the tests are weighted towards
the four claims that are easy to get subtly wrong:

* **OrgMethod is authoritative.** A `.mth` the user loads by hand can confirm a
  step or disagree with it, and that is the whole of its authority. A
  mismatching file must never reach `widget.program`, the segment labels, the
  programmed rates or any Matched cell -- and R6 says the `.mth` format is
  n = 1, so "a file that decodes plausibly and wrongly" is the expected case,
  not the exotic one.
* **"absorbed" and "not executed" are different claims.** Four
  `Isothermal for 20.00 min` holds at 50 °C DID run on the real file; the
  furnace was simply still coasting down through them. A table that reported
  them as missing would be telling the user their method did not execute.
* **Nothing here blocks and nothing here opens a dialog.** The user asked that
  unusual runs still be processable, so every finding on this tab travels
  beside the computation instead of gating it, and a warning that has to be
  clicked away is a warning that gets clicked away.
* **The tab is in °C and stays in °C.** `segment_table(unit="K")` and
  `describe_segment(unit="K")` substitute the degree symbol and convert no
  number, so a kelvin-labelled Program tab would print °C values under a K
  suffix. The tab is labelled °C and left alone instead.
"""
from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QMessageBox

from dta.qt_dta import (
    METHOD_TABLE_COLUMNS, PROGRAM_ROW_ABSORBED, PROGRAM_ROW_DETECTED,
    PROGRAM_ROW_NOT_EXECUTED, PROGRAM_TABLE_KEYS, SCOPE_ALL_TIME,
    DtaWorkspace, program_table_column,
)
from dta.dta_segments import SegmentationConfig, StepMatch
from core.qt_shell import _load_spectrum_from_path
from conftest import requires_dta_bm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _record(path, *, rec_id=None, df=None, title=None):
    """A library-shaped record dict, with an optional stable id.

    `rec_id=None` reproduces the hand-built records the pinned tests use, which
    carry no `"id"` at all and are therefore deliberately NOT cacheable.
    """
    spectrum = _load_spectrum_from_path(str(path))
    rec = {"title": title or spectrum.title, "path": spectrum.path,
           "df": spectrum.df if df is None else df, "meta": spectrum.meta}
    if rec_id is not None:
        rec["id"] = rec_id
    return rec


def _col(key: str) -> int:
    """Segment-table column index by MEANING, never by literal position."""
    index = program_table_column(key)
    assert index >= 0, key
    return index


_METHOD_COL = {name: i for i, name in enumerate(METHOD_TABLE_COLUMNS)}


def _cell(table, row: int, col: int) -> str:
    item = table.item(row, col)
    return "" if item is None else item.text()


def _method_row(widget, step_number: int) -> int:
    """Row of the method table carrying this written step number."""
    for row in range(widget.method_table.rowCount()):
        if _cell(widget.method_table, row, _METHOD_COL["#"]) == str(step_number):
            return row
    raise AssertionError(f"step {step_number} is not in the method table")


def _program_rows(widget):
    """`[(status, index, steps, note), ...]` of the segment table."""
    out = []
    for row in range(widget.program_table.rowCount()):
        out.append((_cell(widget.program_table, row, 0),
                    _cell(widget.program_table, row, _col("index")),
                    _cell(widget.program_table, row, _col("steps")),
                    _cell(widget.program_table, row, _col("note"))))
    return out


def _all_cells(table):
    return [_cell(table, r, c)
            for r in range(table.rowCount())
            for c in range(table.columnCount())]


def _make_match(segment_index, step_indices, *, absorbed=()):
    """A minimal `StepMatch` for the repeated-step case.

    Only the four fields `_step_matches` reads are meaningful; the rest carry
    the dataclass defaults rather than invented measurements.
    """
    return StepMatch(segment_index=int(segment_index),
                     step_indices=tuple(step_indices),
                     programmed_kind="isothermal",
                     programmed_rate=0.0, programmed_target=float("nan"),
                     programmed_duration=20.0, measured_rate=0.0,
                     measured_duration=20.0, target_gap_C=0.0,
                     absorbed_step_indices=tuple(absorbed))


def _write_two_step_mth(path):
    """A syntactically valid but WRONG 2-step .mth: Ramp 5 °C/min to 800, hold 5 min.

    Written in the v1 layout `dta_program` decodes (24-byte records, uint16 LE
    opcode, two float32 LE arguments, twelve zero bytes) rather than by faking
    a `MethodProgram`, because the property under test is that a file which
    decodes PLAUSIBLY and wrongly is still refused the authority to relabel
    anything. R6 says that is the expected failure mode of a format known from
    one specimen.

    The tests that use it also assert `len(widget.mth_program) == 2`, so if the
    decoder ever stops reading this layout the test fails loudly instead of
    passing for the wrong reason.
    """
    blob = bytearray(b"\x00" * 900)
    magic = "TA Instruments Method File".encode("utf-16-le")
    blob[0x10:0x10 + len(magic)] = magic
    offset = 654
    for i, (opcode, arg1, arg2) in enumerate(((0x2635, 5.0, 800.0), (0x2625, 5.0, 0.0))):
        record = struct.pack("<HHff", opcode, 0, arg1, arg2) + b"\x00" * 12
        blob[offset + 24 * i: offset + 24 * i + 24] = record
    path.write_bytes(bytes(blob))
    return path


@pytest.fixture
def no_dialogs(monkeypatch):
    """Make every QMessageBox static a hard failure.

    The Program tab's whole contract is that bad news is shown, not
    interrupted with. `conftest`'s autouse patch merely silences the dialogs;
    this one turns any of them into a test failure.
    """
    def _boom(*_a, **_k):
        raise AssertionError("the Program tab opened a modal dialog")

    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))


# ===========================================================================
# 1. Shape of the tab
# ===========================================================================
def test_the_tab_order_is_tg_program_calculs_export(qtbot, dta_cycles_path):
    """Program sits at index 1, and the tabs around it are where MERGED_DECISIONS
    M5 puts them.

    "Rate series" was inserted at index 3 by the rate-series step, exactly where
    this test's earlier revision said it would land, and this list was updated
    with it — the assertion that matters is that Program is still second and
    "Export / Batch" is still last, i.e. that the insertion moved no existing tab
    relative to the others. Nothing in `qt_dta.py` addresses a tab by number.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    labels = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert labels == ["Tg", "Program", "Calculs", "Rate series", "Export / Batch"]
    assert widget.tabs.tabText(1) == "Program"


def test_program_tab_lists_nineteen_method_steps(qtbot, dta_cycles_path):
    """One row per WRITTEN step, in written order, and the Matched column
    resolves each thermal step to the segment that ran it."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    table = widget.method_table

    assert table.rowCount() == 19
    assert table.columnCount() == len(METHOD_TABLE_COLUMNS)
    assert [_cell(table, r, _METHOD_COL["#"]) for r in range(19)] == \
        [str(i) for i in range(1, 20)]

    row = _method_row(widget, 3)
    assert _cell(table, row, _METHOD_COL["Kind"]) == "ramp"
    assert _cell(table, row, _METHOD_COL["Rate (°C/min)"]) == "20.00"
    assert _cell(table, row, _METHOD_COL["Target (°C)"]) == "1300.00"
    assert _cell(table, row, _METHOD_COL["Duration (min)"]) == ""
    assert "Ramp 20.00" in _cell(table, row, _METHOD_COL["Step"])
    assert _cell(table, row, _METHOD_COL["Matched"]) == "#1 heat"

    # The hold at 1300 °C got its own segment; the hold at 50 °C did not.
    assert _cell(widget.method_table, _method_row(widget, 4),
                 _METHOD_COL["Matched"]) == "#2 isothermal"
    assert _cell(widget.method_table, _method_row(widget, 6),
                 _METHOD_COL["Matched"]) == "#3 (absorbed)"


def test_a_mass_flow_step_is_never_shown_as_a_duration(qtbot, dta_cycles_path):
    """MERGED_DECISIONS M9/D1. The prototype stored "Mass flow 100.00 mL/min"
    in `duration_min`, and any clock summing that column then adds 100 phantom
    minutes to an 890-minute run. The Duration cell of a flow step must be
    blank, and the flow value must still be visible in the verbatim text."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    row = _method_row(widget, 2)
    assert _cell(widget.method_table, row, _METHOD_COL["Kind"]) == "flow"
    assert _cell(widget.method_table, row, _METHOD_COL["Duration (min)"]) == ""
    assert _cell(widget.method_table, row, _METHOD_COL["Rate (°C/min)"]) == ""
    assert "100.00 mL/min" in _cell(widget.method_table, row, _METHOD_COL["Step"])


def test_a_non_thermal_step_is_not_reported_as_unexecuted(qtbot, dta_cycles_path):
    """"Select gas 1" moves no setpoint, so it CANNOT leave a signature in a
    temperature record. Reporting it as "not executed" would accuse the
    instrument of skipping an instruction it carried out."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    for step in (1, 2, 19):
        row = _method_row(widget, step)
        assert _cell(widget.method_table, row, _METHOD_COL["Matched"]) == "—"
        tip = widget.method_table.item(row, _METHOD_COL["Matched"]).toolTip()
        assert "not a thermal step" in tip.lower()
    statuses = [status for status, *_ in _program_rows(widget)]
    assert PROGRAM_ROW_NOT_EXECUTED not in statuses


def test_a_repeated_step_reports_every_pass_it_made(qtbot, dta_cycles_path):
    """`Repeat segment 1 for N times` makes one WRITTEN step run several
    times, and the furnace need not behave the same way on every pass: the
    same hold can settle into its own segment once and be absorbed into a
    cooling tail the next time round. Both have to reach the Matched cell.

    Built from dataclasses rather than from a file because no committed
    fixture contains a repeat, and reporting only the pass that produced a
    segment would show the flattering half of a run with nothing on screen to
    say the other half existed.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    run = widget.run
    first, second = run.segments[1], run.segments[4]      # two real holds

    # Written step 4 ran twice: segment #2 settled into it, segment #5 did not.
    matches = (
        _make_match(2, (4,), absorbed=()),
        _make_match(5, (4,), absorbed=(4,)),
    )
    widget.run = dataclasses.replace(
        run,
        segments=(dataclasses.replace(first, step_indices=(4,)),
                  dataclasses.replace(second, step_indices=(4,))),
        alignment=dataclasses.replace(run.alignment, matches=matches),
    )

    text, segment_index, tip = widget._step_matches()[4]
    assert text == "#2 isothermal, #5 (absorbed)"
    assert segment_index == 2
    assert "Absorbed is not 'not executed'" in tip
    assert "Segment #2" in tip                           # the settled pass too


def test_no_cell_on_the_tab_renders_nan_or_none(qtbot, dta_cycles_path):
    """A NaN rendered "nan" reads as a broken number and a NaN rendered 0.0
    reads as a measurement. Missing is blank, everywhere."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    for table in (widget.method_table, widget.program_table):
        for text in _all_cells(table):
            assert "nan" not in text.lower()
            assert text != "None"


# ===========================================================================
# 2. Segment table: detected / absorbed / not executed
# ===========================================================================
def test_program_table_has_a_row_per_segment(qtbot, dta_cycles_path):
    """12 detected segments plus the 4 absorbed holds = 16 rows, and every
    absorbed row sits immediately after the segment that absorbed it, because
    that adjacency IS the finding being reported."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    rows = _program_rows(widget)

    assert widget.program_table.columnCount() == len(PROGRAM_TABLE_KEYS)
    assert len(rows) == 16
    detected = [r for r in rows if r[0] == PROGRAM_ROW_DETECTED]
    absorbed = [r for r in rows if r[0] == PROGRAM_ROW_ABSORBED]
    assert len(detected) == 12
    assert len(absorbed) == 4
    assert [r[1] for r in detected] == [str(i) for i in range(1, 13)]
    assert [r[2] for r in absorbed] == ["6", "10", "14", "18"]

    # #3 is the first cool of cycle 1: steps 5+6, a missed target and a lost rate.
    third = next(r for r in detected if r[1] == "3")
    assert third[2] == "5,6"
    assert "44.4 °C short" in third[3]
    row3 = [r[0] for r in rows].index(PROGRAM_ROW_DETECTED, 2)
    assert _cell(widget.program_table, row3, _col("kind")) == "cool"
    assert _cell(widget.program_table, row3, _col("cycle")) == "1"
    assert float(_cell(widget.program_table, row3, _col("rate_mode"))) == \
        pytest.approx(-19.961, abs=0.02)
    assert float(_cell(widget.program_table, row3, _col("rate_nominal"))) == \
        pytest.approx(-20.0, abs=1e-6)
    assert float(_cell(widget.program_table, row3, _col("target_gap"))) == \
        pytest.approx(44.36, abs=0.05)
    assert float(_cell(widget.program_table, row3, _col("control_lost_below"))) == \
        pytest.approx(447.6, abs=1.0)
    # The absorbed row for step 6 follows it immediately.
    assert rows[row3 + 1][0] == PROGRAM_ROW_ABSORBED
    assert rows[row3 + 1][2] == "6"


def test_absorbed_and_not_executed_are_different_claims(qtbot, dta_cycles_path):
    """The headline honesty requirement of this tab.

    The first 1300 rows of the cyclic fixture are cycle 1 plus the start of
    cycle 2, so steps 9-18 genuinely never ran while step 6 ran and left no
    separate signature. The two must reach the table as two different statuses
    with two different sentences: one says the step happened, the other says
    the data never shows it.
    """
    spectrum = _load_spectrum_from_path(str(dta_cycles_path))
    truncated = spectrum.df.iloc[:1300].copy()
    widget = DtaWorkspace(records=[_record(dta_cycles_path, df=truncated,
                                           title="cycle 1 only")])
    qtbot.addWidget(widget)

    rows = _program_rows(widget)
    absorbed = [r for r in rows if r[0] == PROGRAM_ROW_ABSORBED]
    missing = [r for r in rows if r[0] == PROGRAM_ROW_NOT_EXECUTED]
    assert [r[2] for r in absorbed] == ["6"]
    assert [r[2] for r in missing] == [str(i) for i in range(9, 19)]

    assert "ran but left no separate signature" in absorbed[0][3]
    assert "never appears in the data" in missing[0][3]
    # Neither sentence may be reachable from the other's status.
    assert "never appears" not in absorbed[0][3]
    assert "ran but left" not in missing[0][3]

    warnings = "\n".join(widget.program_warning_lines())
    assert "never appear in the data" in warnings
    assert "NOT the same claim" in warnings
    assert "absorbed" in warnings


def test_the_absorbed_row_measures_nothing(qtbot, dta_cycles_path):
    """An absorbed step has no measured duration, rate or temperature: it was
    never separated from the ramp it happened in. Those cells are blank, not
    zero -- a 0.0 in a duration column is a claim about the furnace."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    row = next(r for r in range(widget.program_table.rowCount())
               if _cell(widget.program_table, r, 0) == PROGRAM_ROW_ABSORBED)
    for key in ("index", "cycle", "rows", "n", "t_start", "t_end", "dur_min",
                "T_start", "T_end", "rate_mean", "rate_mode", "rate_nominal",
                "target_gap", "control_lost_below"):
        assert _cell(widget.program_table, row, _col(key)) == "", key
    # What IS known about it is the programmed step and the sentence.
    assert _cell(widget.program_table, row, _col("steps")) == "6"
    assert _cell(widget.program_table, row, _col("kind")) == "isothermal"


def test_a_single_ramp_file_gets_one_method_row_and_one_segment_row(
        qtbot, dta_example_path):
    """`EXAMPLES/DTA_example.txt` is one heating ramp under a one-step method.
    Nothing about the Program tab may require a cyclic file."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.method_table.rowCount() == 1
    assert widget.program_table.rowCount() == 1
    assert _cell(widget.program_table, 0, 0) == PROGRAM_ROW_DETECTED
    assert _cell(widget.method_table, 0, _METHOD_COL["Matched"]) == "#1 heat"
    assert "1 step" in widget.program_summary_label.text()
    assert "1 segment" in widget.program_summary_label.text()


def test_an_unsegmented_record_says_so_instead_of_showing_nothing(qtbot):
    """No clock and no furnace-temperature column means no segmentation and no
    programmed method. Both tables are empty and the tab says why -- an empty
    table with no explanation is indistinguishable from a bug."""
    df = pd.DataFrame({"A": np.linspace(0.0, 1.0, 64),
                       "B": np.linspace(2.0, 3.0, 64)})
    widget = DtaWorkspace(records=[{"title": "synthetic", "df": df, "meta": {}}])
    qtbot.addWidget(widget)
    assert widget.run is None
    assert widget.method_table.rowCount() == 0
    assert widget.program_table.rowCount() == 0
    assert "not segmented" in widget.program_summary_label.text()
    text = widget.program_warning_label.text()
    assert "not split into segments" in text
    assert "No programmed method could be read" in text


def test_a_bare_workspace_builds_an_empty_program_tab(qtbot):
    """`test_dta_workspace_constructs_standalone` pins that a DtaWorkspace with
    no records is constructible. The Program tab must not change that."""
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    assert widget.method_table.rowCount() == 0
    assert widget.program_table.rowCount() == 0
    assert widget.program_summary_label.text() == "No record loaded."
    assert widget.program_warning_label.text() == ""
    assert widget.program_warning_lines() == []
    assert widget.mth_program is None


# ===========================================================================
# 3. Warnings: aggregated, plain, and never a dialog
# ===========================================================================
def test_the_warnings_name_control_lost_and_the_target_gap(qtbot, dta_cycles_path):
    """The two diagnostics SPEC_cycles §6.8 puts in MUST scope, in the words a
    user can act on.

    "-20 °C/min" is true at 1300 °C and false at 200 °C on this file, and the
    only way a user can know that is if the tab says which temperature the
    furnace stopped following the programme at. All four cools are named, with
    their own numbers, in one aggregated paragraph and not four dialogs.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    lines = widget.program_warning_lines()

    lost = next(l for l in lines if "held the programmed rate only part of the way" in l)
    for segment, temperature in (("#3", "448"), ("#6", "346"),
                                 ("#9", "228"), ("#12", "228")):
        assert segment in lost
        assert temperature in lost
    assert "-20 °C/min" in lost
    assert "1299 °C" in lost           # where it IS true
    assert "no longer true below 448 °C" in lost
    assert "coasting" in lost

    gap = next(l for l in lines if "targets that were not reached" in l)
    assert "44.4 °C short" in gap
    assert "27.3 °C short" in gap
    assert "9.9 °C short" in gap
    assert "12.1 °C short" in gap
    # The four heating ramps stop 1.36-1.39 °C short and reach the setpoint
    # during the soak that follows. They are NOT a finding.
    assert "#1 heat" not in gap
    assert "1.4 °C short" not in gap

    # All of it is on screen, marked, and visually distinct.
    shown = widget.program_warning_label.text()
    assert shown.count("⚠") == len(lines)
    assert "color:" in widget.program_warning_label.styleSheet()


def test_the_partial_last_segment_is_reported(qtbot, dta_cycles_path):
    """Acquisition stopped inside segment #12: its duration and end temperature
    are where the recording stopped, not where the programme did."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    line = next(l for l in widget.program_warning_lines()
                if "Acquisition stopped inside" in l)
    assert "12" in line
    row = next(r for r in range(widget.program_table.rowCount())
               if _cell(widget.program_table, r, _col("index")) == "12")
    assert _cell(widget.program_table, row, _col("partial")) == "yes"


def test_a_clean_record_reports_only_what_is_actually_wrong(qtbot,
                                                            dta_synthetic_cycles):
    """The warnings are findings, not decoration.

    The synthetic program is a clean 2-cycle run with no header, so exactly two
    things are true of it: there is no programmed method to compare against,
    and the record happens to stop inside its last cool. Neither a missed target
    nor a lost rate may be invented on data that has neither -- a tab that
    warns about everything is a tab nobody reads.
    """
    t, temp = dta_synthetic_cycles
    df = pd.DataFrame({"Time (min)": t, "Temperature (°C)": temp,
                       "Heat Flow (mW)": np.zeros_like(t)})
    widget = DtaWorkspace(records=[{"title": "clean", "df": df, "meta": {}}])
    qtbot.addWidget(widget)
    assert widget.run is not None and len(widget.run) == 6

    lines = widget.program_warning_lines()
    assert any("No programmed method could be read" in l for l in lines)
    assert any("Acquisition stopped inside" in l for l in lines)
    assert len(lines) == 2, lines
    assert not any("held the programmed rate only part of the way" in l for l in lines)
    assert not any("targets that were not reached" in l for l in lines)
    assert not any("absorbed" in l for l in lines)
    # With no method there is nothing to compare, so neither "unmatched" claim
    # may be made either: every segment is trivially unmatched here.
    assert not any("match no programmed step" in l for l in lines)
    assert not any("never appear in the data" in l for l in lines)
    # With no method there is nothing to align, so no step is claimed either way.
    assert widget.method_table.rowCount() == 0
    assert all(status == PROGRAM_ROW_DETECTED
               for status, *_ in _program_rows(widget))


def test_nothing_on_the_program_tab_opens_a_dialog(qtbot, no_dialogs,
                                                   dta_cycles_path,
                                                   dta_cycles_mth_path, tmp_path):
    """Loading a record with four missed targets and four lost-control ramps,
    loading a matching .mth, loading a wrong one, loading a file that is not a
    method file at all, re-detecting, resetting and clicking rows -- none of it
    may interrupt the user. Every QMessageBox static raises in this test."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget.program_warning_lines()

    widget.load_mth(dta_cycles_mth_path)
    widget.load_mth(_write_two_step_mth(tmp_path / "wrong.mth"))
    garbage = tmp_path / "not-a-method.mth"
    garbage.write_text("this is plain text, not a TA method file\n", encoding="utf-8")
    widget.load_mth(garbage)
    widget.clear_mth()
    widget._on_program_row_clicked(0, 0)
    widget._on_method_row_clicked(2, 0)
    widget.seg_hold_edit.setText("")
    widget._on_redetect()
    widget._on_seg_params_reset()


def test_no_finding_disables_the_computation(qtbot, dta_cycles_path):
    """"Still make this possible to process such data": every warning on this
    tab is information carried beside the computation, never a gate in front of
    it. A segment with a missed target, a lost rate AND a partial acquisition
    still computes."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget.select_segment(12) is True          # the worst segment there is
    segment = widget._scope().segment
    assert segment.is_partial
    assert segment.target_gap_C > 5.0
    assert np.isfinite(segment.control_lost_below_C)
    assert widget.compute_btn.isEnabled()
    assert widget.integrate_btn.isEnabled()
    assert widget.export_csv_btn.isEnabled()
    widget._compute()
    assert np.isfinite(widget.tg_deriv)


# ===========================================================================
# 4. The .mth cross-check. OrgMethod is authoritative.
# ===========================================================================
def test_the_matching_mth_confirms_every_step(qtbot, dta_cycles_path,
                                              dta_cycles_mth_path):
    """`EXAMPLES/DTA_cycles_example.mth` is a byte copy of the method that
    produced this run, so all 19 steps come back tagged "both" -- and
    `widget.program` is still the OrgMethod one, not the merge."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert widget.mth_program is None

    comparison = widget.load_mth(dta_cycles_mth_path)
    assert comparison.verdict == "identical"
    assert comparison.score == pytest.approx(1.0)
    assert len(widget.mth_program) == 19

    sources = {_cell(widget.method_table, r, _METHOD_COL["Source"])
               for r in range(widget.method_table.rowCount())}
    assert sources == {"both"}
    assert widget.method_table.rowCount() == 19
    assert widget.program.source == "orgmethod"
    assert "identical" in widget.mth_note_label.text()
    assert not any("DOES NOT MATCH" in l for l in widget.program_warning_lines())


def test_loading_a_wrong_mth_does_not_override_orgmethod(qtbot, dta_cycles_path,
                                                         tmp_path):
    """THE test for risk R6.

    A 2-step .mth that decodes perfectly well describes a different furnace
    programme. It must be reported prominently and it must change nothing: not
    `widget.program`, not the method table, not the Source column, not a single
    segment label, not a programmed rate.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    labels_before = [widget.segment_combo.itemText(i)
                     for i in range(widget.segment_combo.count())]
    rates_before = [s.rate_nominal_C_min for s in widget.run.segments]
    method_before = [_cell(widget.method_table, r, _METHOD_COL["Step"])
                     for r in range(widget.method_table.rowCount())]

    comparison = widget.load_mth(_write_two_step_mth(tmp_path / "wrong.mth"))

    assert len(widget.mth_program) == 2            # it really did decode
    assert comparison.verdict == "mismatch"
    assert comparison.is_same_method is False

    # The authoritative program is untouched, and so is everything built on it.
    assert widget.program is not None and len(widget.program) == 19
    assert widget.program.source == "orgmethod"
    assert widget.method_table.rowCount() == 19
    assert [_cell(widget.method_table, r, _METHOD_COL["Step"])
            for r in range(19)] == method_before
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"OrgMethod"}
    assert [widget.segment_combo.itemText(i)
            for i in range(widget.segment_combo.count())] == labels_before
    assert [s.rate_nominal_C_min for s in widget.run.segments] == rates_before

    # And it is impossible to miss.
    note = widget.mth_note_label.text()
    assert "DOES NOT MATCH" in note
    assert "relabels nothing" in note
    lines = widget.program_warning_lines()
    assert "DOES NOT MATCH" in lines[0]            # first, above every other finding
    assert "Where they differ" in lines[1]
    assert "rate 20.00 vs 5.00" in lines[1]
    assert "color:" in widget.program_warning_label.styleSheet()


def test_a_real_mth_for_another_run_is_a_mismatch_too(qtbot, dta_example_path,
                                                      dta_cycles_mth_path):
    """The same rule with no hand-built binary anywhere: the real cyclic .mth
    loaded against the single-ramp record is a genuine, fully parsed method
    file that simply belongs to a different experiment."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    comparison = widget.load_mth(dta_cycles_mth_path)
    assert widget.mth_program.status == "parsed"
    assert comparison.verdict == "mismatch"
    assert widget.method_table.rowCount() == 1           # still the 1-step method
    assert _cell(widget.method_table, 0, _METHOD_COL["Source"]) == "OrgMethod"
    assert "DOES NOT MATCH" in widget.program_warning_lines()[0]


def test_a_file_that_is_not_a_method_file_degrades_instead_of_raising(
        qtbot, dta_cycles_path, tmp_path):
    """"Incomparable" is deliberately not "equivalent": a caller asking "does
    this .mth belong to this run?" must never be talked into yes by a file it
    could not read."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    path = tmp_path / "junk.mth"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    comparison = widget.load_mth(path)
    assert comparison.verdict == "incomparable"
    assert comparison.is_same_method is False
    assert not widget.mth_program                     # falsy, not an exception
    assert "could not be read as a TA method file" in widget.mth_note_label.text()
    assert widget.method_table.rowCount() == 19
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"OrgMethod"}


def test_clearing_the_mth_restores_the_orgmethod_only_view(
        qtbot, dta_cycles_path, dta_cycles_mth_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.load_mth(dta_cycles_mth_path)
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"both"}

    widget.clear_mth()
    assert widget.mth_program is None
    assert widget.mth_path is None
    assert widget.mth_comparison is None
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"OrgMethod"}
    assert "No .mth loaded" in widget.mth_note_label.text()


def test_switching_records_rejudges_a_loaded_mth(qtbot, dta_cycles_path,
                                                 dta_example_path,
                                                 dta_cycles_mth_path):
    """A .mth that matched record A says nothing about record B.

    It is KEPT -- silently dropping it would be its own surprise -- and
    re-compared, so the stale "identical" cannot survive the record switch. A
    verdict that looks current and is not is worse than no verdict.
    """
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc"),
                                   _record(dta_example_path, rec_id="ramp")])
    qtbot.addWidget(widget)
    widget.load_mth(dta_cycles_mth_path)
    assert widget.mth_comparison.verdict == "identical"

    widget.record_combo.setCurrentIndex(1)
    assert widget.method_table.rowCount() == 1
    assert widget.mth_program is not None                 # kept
    assert widget.mth_comparison.verdict == "mismatch"    # re-judged
    assert "DOES NOT MATCH" in widget.program_warning_lines()[0]

    widget.record_combo.setCurrentIndex(0)
    assert widget.mth_comparison.verdict == "identical"
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"both"}


def test_load_mth_is_reachable_without_a_dialog(qtbot, dta_cycles_path,
                                                dta_cycles_mth_path, monkeypatch):
    """House rule: every file-dialog method is a thin wrapper over a directly
    callable one. `_on_load_mth` must do nothing but choose a path."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    seen = []
    monkeypatch.setattr(widget, "load_mth", lambda p: seen.append(p))
    from dta import qt_dta as module
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    widget._on_load_mth()
    assert seen == []                    # cancelled: nothing happens
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(dta_cycles_mth_path), "")))
    widget._on_load_mth()
    assert seen == [dta_cycles_mth_path]


# ===========================================================================
# 5. Clicking a row selects the segment
# ===========================================================================
def test_program_table_row_click_selects_the_segment(qtbot, dta_cycles_path):
    """The scope combo lives on the Tg tab, so a click here has to drive THAT
    widget -- the scope memory, the guardrails, the default window and the
    redraw all hang off its signal."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc")])
    qtbot.addWidget(widget)
    assert widget._scope_key == SCOPE_ALL_TIME

    row_of_six = next(r for r in range(widget.program_table.rowCount())
                      if _cell(widget.program_table, r, 0) == PROGRAM_ROW_DETECTED
                      and _cell(widget.program_table, r, _col("index")) == "6")
    widget._on_program_row_clicked(row_of_six, 0)

    assert widget._scope_key == "segment:6"
    assert widget._scope().segment.index == 6
    assert "#6" in widget.segment_combo.currentText()
    assert "#6" in widget.program_selection_label.text()
    # It went through the combo, so the per-record scope memory followed.
    assert widget._scope_memory[widget._record_id] == "segment:6"


def test_an_absorbed_row_click_selects_its_absorbing_segment(qtbot, dta_cycles_path):
    """Step 6 has no segment of its own, and the honest place to send a user
    who clicks it is the cool it was absorbed into."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    row = next(r for r in range(widget.program_table.rowCount())
               if _cell(widget.program_table, r, 0) == PROGRAM_ROW_ABSORBED)
    widget._on_program_row_clicked(row, 0)
    assert widget._scope_key == "segment:3"


def test_a_method_row_click_selects_the_segment_that_ran_it(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget._on_method_row_clicked(_method_row(widget, 11), 0)
    assert widget._scope_key == "segment:7"          # step 11 is heat 3
    widget._on_method_row_clicked(_method_row(widget, 5), 0)
    assert widget._scope_key == "segment:3"          # step 5 is cool 1


def test_clicking_a_row_with_no_segment_changes_nothing(qtbot, dta_cycles_path):
    """"Select gas 1" and a never-executed step both resolve to no segment. A
    click there must be inert, not an exception and not a silent jump to
    segment #1."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.select_segment(3)
    assert widget._scope_key == "segment:3"
    widget._on_method_row_clicked(_method_row(widget, 1), 0)     # gas
    assert widget._scope_key == "segment:3"
    assert widget.select_segment(None) is False
    assert widget.select_segment(99) is False                    # no such segment
    assert widget._scope_key == "segment:3"


def test_the_selected_row_is_highlighted_and_follows_the_combo(qtbot, dta_cycles_path):
    """Selection travels both ways: the table follows the combo too, or the two
    disagree about what the workspace is pointed at."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    index = next(i for i in range(widget.segment_combo.count())
                 if widget.segment_combo.itemData(i) == "segment:9")
    widget.segment_combo.setCurrentIndex(index)

    rows = widget.program_table.selectionModel().selectedRows()
    assert len(rows) == 1
    selected = rows[0].row()
    assert _cell(widget.program_table, selected, 0) == PROGRAM_ROW_DETECTED
    assert _cell(widget.program_table, selected, _col("index")) == "9"


# ===========================================================================
# 6. The three detection knobs
# ===========================================================================
def test_the_knobs_show_the_config_in_force(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    defaults = SegmentationConfig()
    assert float(widget.seg_window_edit.text()) == defaults.rate_window_min
    assert float(widget.seg_hold_edit.text()) == defaults.min_hold_min
    assert float(widget.seg_span_edit.text()) == defaults.hold_flat_span_C


def test_the_effective_rate_window_is_stated_honestly(qtbot, dta_cycles_path):
    """A "0.5 min" field is a lie on a coarsely sampled export: the estimator
    widens the window in index space to hold `rate_min_points`. The fixture
    samples at 0.1875 min/point, where 0.5 min holds only ~4 points, so the
    note has to say what actually happens."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    note = widget.seg_params_note.text()
    assert "0.1875 min/point" in note
    assert "widens it to 5 points" in note


def test_redetect_reruns_segmentation_with_the_typed_knobs(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert len(widget.run) == 12

    widget.seg_window_edit.setText("1.25")
    widget.seg_hold_edit.setText("3")
    widget.seg_span_edit.setText("2")
    widget._on_redetect()

    config = widget._seg_config
    assert config.rate_window_min == 1.25
    assert config.min_hold_min == 3.0
    assert config.hold_flat_span_C == 2.0
    # Segmentation of this run is stable across a wide band of settings, which
    # is the point of the measured defaults: the knobs are a safety valve.
    assert len(widget.run) == 12
    assert "Re-detected" in widget.status_label.text()


def test_a_non_positive_knob_is_refused_and_the_previous_value_kept(
        qtbot, dta_cycles_path):
    """A zero-length rate window is not a strict setting, it is an unusable
    one. Refuse it, keep the previous value, say so in the note -- never a
    silent fall back to a default the user did not ask for, and never a
    dialog."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.seg_window_edit.setText("0.8")
    widget._on_redetect()
    assert widget._seg_config.rate_window_min == 0.8

    for bad in ("0", "-2", "", "abc", "nan"):
        widget.seg_hold_edit.setText(bad)
        widget._on_redetect()
        assert widget._seg_config.min_hold_min == SegmentationConfig().min_hold_min
        assert "Kept the previous value for Min hold" in widget.seg_params_note.text()
        # A refused field does not poison the two good ones beside it.
        assert widget._seg_config.rate_window_min == 0.8
    # The field is put back to what is actually in force, so what is on screen
    # is what the segmentation used.
    assert float(widget.seg_hold_edit.text()) == SegmentationConfig().min_hold_min


def test_reset_restores_the_measured_defaults(qtbot, dta_cycles_path):
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.apply_detection_settings(rate_window_min=2.0, min_hold_min=9.0,
                                    hold_flat_span_C=7.0)
    assert widget._seg_config.rate_window_min == 2.0

    widget._on_seg_params_reset()
    defaults = SegmentationConfig()
    assert widget._seg_config == defaults
    assert float(widget.seg_window_edit.text()) == defaults.rate_window_min
    assert float(widget.seg_hold_edit.text()) == defaults.min_hold_min
    assert float(widget.seg_span_edit.text()) == defaults.hold_flat_span_C
    assert len(widget.run) == 12


def test_redetect_empties_the_whole_run_cache(qtbot, dta_cycles_path,
                                              dta_example_path):
    """The cache is keyed by record and remembers nothing about the config a
    run was detected under. Clearing only the record on screen would leave the
    other file segmented by the old rules, with nothing on screen to say so."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path, rec_id="cyc"),
                                   _record(dta_example_path, rec_id="ramp")])
    qtbot.addWidget(widget)
    widget.record_combo.setCurrentIndex(1)
    widget.record_combo.setCurrentIndex(0)
    assert set(widget._run_cache) == {"id:cyc", "id:ramp"}
    stale = widget._run_cache["id:ramp"]

    widget.apply_detection_settings(rate_window_min=0.9, min_hold_min=2.0,
                                    hold_flat_span_C=1.0)
    assert "id:ramp" not in widget._run_cache
    assert widget._run_cache["id:cyc"].config.rate_window_min == 0.9

    widget.record_combo.setCurrentIndex(1)
    assert widget.run is not stale
    assert widget.run.config.rate_window_min == 0.9


def test_redetect_forgets_a_tg_measured_on_the_old_boundaries(qtbot, dta_cycles_path):
    """A Tg measured before re-detection is not a measurement after it: the
    curve, the window and the baselines have all moved underneath it."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.select_segment(1)
    widget._compute()
    assert widget.res_parallel is not None

    widget._on_redetect()
    assert widget.res_double is None
    assert widget.res_parallel is None
    assert widget.tg_deriv is None
    assert widget.result_label.text() == "Tg: —"


def test_apply_detection_settings_is_callable_without_the_line_edits(
        qtbot, dta_cycles_path):
    """The house rule again: the logic takes explicit arguments and the button
    handler only reads three fields."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    config = widget.apply_detection_settings(rate_window_min=0.75,
                                             min_hold_min=2.5,
                                             hold_flat_span_C=1.5)
    assert config is widget._seg_config
    assert (config.rate_window_min, config.min_hold_min,
            config.hold_flat_span_C) == (0.75, 2.5, 1.5)
    assert float(widget.seg_window_edit.text()) == 0.75


def test_there_is_no_iso_tol_knob(qtbot, dta_cycles_path):
    """SPEC_cycles C5, measured and decided: no pointwise |dT/dt| tolerance can
    separate a hold from the tail of a cooling ramp on a real run, and a value
    of 2 °C/min invents two spurious hold segments inside the cooling tails.
    Offering the knob would invite tuning something with no working value, so
    it is a REJECTED feature rather than a missing one -- and the tab says why,
    where a user would otherwise go looking for it."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    assert not [name for name in dir(widget) if "iso_tol" in name.lower()]
    assert "iso_tol" not in {f.name for f in dataclasses.fields(SegmentationConfig)}
    program_tab = widget.tabs.widget(1)
    texts = " ".join(label.text() for label in program_tab.findChildren(type(
        widget.program_summary_label)))
    assert "no |dT/dt| tolerance" in texts
    assert "cannot work" in texts


# ===========================================================================
# 7. Units: the Program tab is in °C and stays there
# ===========================================================================
def test_the_program_tables_stay_in_celsius_in_kelvin_mode(qtbot, dta_cycles_path):
    """`segment_table(unit="K")` and `describe_segment(unit="K")` substitute the
    degree symbol and convert NOTHING, so a kelvin-labelled Program tab would
    print °C numbers under a K suffix -- and the Note column, being free text
    from the science layer, cannot be converted at all. The tab is therefore
    labelled °C and left alone, and it says so on screen."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.select_segment(1)                 # a ramp, so the switch is enabled
    before = [_all_cells(widget.method_table), _all_cells(widget.program_table)]

    widget.temp_k_btn.setChecked(True)
    assert widget._temp_unit() == "K"
    assert [_all_cells(widget.method_table),
            _all_cells(widget.program_table)] == before

    assert "Target (°C)" in METHOD_TABLE_COLUMNS
    headers = [widget.program_table.horizontalHeaderItem(c).text()
               for c in range(widget.program_table.columnCount())]
    assert "T start (°C)" in headers
    assert not any("(K)" in h for h in headers)
    assert "°C" in widget.program_table_note.text()
    assert "do not follow the °C/K switch" in widget.program_table_note.text()


def test_loading_a_mth_never_redraws_the_plot(qtbot, dta_cycles_path,
                                              dta_cycles_mth_path, monkeypatch):
    """Risk R5. A `.mth` changes nothing about the curve, so it must not queue
    a render of a 19k-point figure -- and, more importantly, must not clear the
    Tg a user has just computed by redrawing over it."""
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    widget.select_segment(1)
    widget._compute()
    tg = widget.res_parallel.tg

    renders = []
    monkeypatch.setattr(widget, "_render", lambda: renders.append(1))
    monkeypatch.setattr(widget.plot, "request_redraw",
                        lambda *a, **k: renders.append("queued"))
    widget.load_mth(dta_cycles_mth_path)
    widget.clear_mth()
    widget._refresh_program_tab()
    assert renders == []
    assert widget.res_parallel.tg == tg
    assert widget._scope_key == "segment:1"


def test_program_table_column_addresses_cells_by_meaning(qtbot, dta_cycles_path):
    """The helper the tests use is public, so a column inserted later moves the
    tests and the caller together instead of silently shifting an index."""
    assert program_table_column("status") == 0
    assert program_table_column("note") == len(PROGRAM_TABLE_KEYS) - 1
    assert program_table_column("not a column") == -1
    widget = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(widget)
    headers = [widget.program_table.horizontalHeaderItem(c).text()
               for c in range(widget.program_table.columnCount())]
    assert headers[program_table_column("status")] == "Status"
    assert headers[program_table_column("control_lost_below")] == \
        "Control lost below (°C)"
    # The two columns that read 0 on a hold carry their explanation in the
    # header tooltip, where a reader meets them.
    tip = widget.program_table.horizontalHeaderItem(
        program_table_column("target_gap")).toolTip()
    assert "no target temperature at all" in tip


def test_the_pinned_tg_is_untouched_by_the_program_tab(qtbot, dta_example_path):
    """Belt and braces around `test_dta_workspace_compute_matches_known_good_values`:
    the Program tab is built on every record load, and the pinned Tg depends on
    the 350/700 window surviving that."""
    widget = DtaWorkspace(records=[_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.xmin_edit.text() == "350"
    assert widget.xmax_edit.text() == "700"
    widget._compute()
    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-4)


# ===========================================================================
# 8. The real full-resolution files
# ===========================================================================
@requires_dta_bm
def test_full_cycles_file_program_tab(qtbot, dta_bm_dir):
    """The 18996-row original, at its native 0.046 min sampling.

    The committed fixture is every 4th row of this file, so a finding that
    holds on one and not on the other is a finding about the decimation rather
    than about the furnace. Both of the diagnostics this tab exists to show
    survive the 4x: control is lost at 448 / 345 / 228 / 227 °C here against
    448 / 346 / 228 / 228 °C on the fixture, and the four cooling target gaps
    agree to 0.25 °C. That is the point of measuring the rate over a window in
    MINUTES instead of in points.
    """
    path = dta_bm_dir / "BM-0-cycles.txt"
    widget = DtaWorkspace(records=[_record(path, rec_id="bm-cycles")])
    qtbot.addWidget(widget)

    assert widget.method_table.rowCount() == 19
    assert len(widget.run) == 12
    rows = _program_rows(widget)
    assert len([r for r in rows if r[0] == PROGRAM_ROW_DETECTED]) == 12
    assert [r[2] for r in rows if r[0] == PROGRAM_ROW_ABSORBED] == \
        ["6", "10", "14", "18"]
    assert not [r for r in rows if r[0] == PROGRAM_ROW_NOT_EXECUTED]

    lost = [s.control_lost_below_C for s in widget.run.segments
            if np.isfinite(s.control_lost_below_C)]
    assert len(lost) == 4
    assert lost == pytest.approx([447.59, 344.94, 227.86, 227.16], abs=0.05)
    # Loss of control is a COOLING finding on this file: every heating ramp
    # tracked its setpoint to the top.
    assert [s.index for s in widget.run.segments
            if np.isfinite(s.control_lost_below_C)] == [3, 6, 9, 12]

    line = next(l for l in widget.program_warning_lines()
                if "held the programmed rate only part of the way" in l)
    assert "-20 °C/min" in line
    assert "down to 448 °C" in line
    assert "no longer true below 448 °C" in line

    gaps = [s.target_gap_C for s in widget.run.by_kind("cool")]
    assert gaps == pytest.approx([44.357, 27.325, 9.884, 11.808], abs=0.01)
    assert "44.4 °C short" in next(l for l in widget.program_warning_lines()
                                   if "targets that were not reached" in l)


@requires_dta_bm
def test_the_fixture_reports_the_same_findings_as_the_full_file(
        qtbot, dta_bm_dir, dta_cycles_path):
    """The committed fixture must not drift away from the run it came from.

    Every conclusion in this file's tests is drawn on 4749 decimated rows, so
    the fixture earns its place only if the FINDINGS -- not merely the segment
    count -- survive the decimation. They do: same 12 segments, same four
    absorbed steps, same four cooling targets missed to 0.25 °C, same four
    control-loss temperatures to 1.1 °C.
    """
    full = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0-cycles.txt")])
    qtbot.addWidget(full)
    fixture = DtaWorkspace(records=[_record(dta_cycles_path)])
    qtbot.addWidget(fixture)

    assert len(full.run) == len(fixture.run) == 12
    assert ([s.kind for s in full.run.segments]
            == [s.kind for s in fixture.run.segments])
    assert (list(full.run.alignment.absorbed_steps)
            == list(fixture.run.alignment.absorbed_steps) == [6, 10, 14, 18])
    assert ([s.target_gap_C for s in full.run.by_kind("cool")]
            == pytest.approx([s.target_gap_C for s in fixture.run.by_kind("cool")],
                             abs=0.25))
    assert ([s.control_lost_below_C for s in full.run.by_kind("cool")]
            == pytest.approx([s.control_lost_below_C
                              for s in fixture.run.by_kind("cool")], abs=1.1))
    assert (len(full.program_warning_lines())
            == len(fixture.program_warning_lines()))


@requires_dta_bm
def test_the_real_mth_confirms_the_real_run_through_the_widget(qtbot, dta_bm_dir):
    """`cycles.mth` next to `BM-0-cycles.txt` is the method that ran it, so
    every Source cell says "both" and no mismatch is reported."""
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0-cycles.txt")])
    qtbot.addWidget(widget)
    comparison = widget.load_mth(dta_bm_dir / "cycles.mth")
    assert comparison.verdict == "identical"
    assert {_cell(widget.method_table, r, _METHOD_COL["Source"])
            for r in range(19)} == {"both"}
    assert not any("DOES NOT MATCH" in l for l in widget.program_warning_lines())


@requires_dta_bm
def test_the_single_ramp_bm_file_reports_its_trailing_hold(qtbot, dta_bm_dir):
    """`BM-0.txt` is one heat plus a trailing hold under a 2-step method: no
    absorbed rows, no never-executed rows, nothing missed by more than the
    warning threshold."""
    widget = DtaWorkspace(records=[_record(dta_bm_dir / "BM-0.txt")])
    qtbot.addWidget(widget)
    assert len(widget.run) == 2
    rows = _program_rows(widget)
    assert [r[0] for r in rows] == [PROGRAM_ROW_DETECTED, PROGRAM_ROW_DETECTED]
    assert [r[1] for r in rows] == ["1", "2"]
    assert not any("never appear in the data" in l
                   for l in widget.program_warning_lines())
