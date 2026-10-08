"""Tests for dta/dta_program.py -- TA OrgMethod text and .mth binary.

Three things these tests exist to protect:

1. **The round trip.** A step decoded from the binary .mth must render to the
   same string the instrument wrote into the data file's own header. All 19
   real lines of BM-0-cycles.txt are pinned here as REAL_ORGMETHOD_LINES, so
   the grammar and the renderer are checked against real bytes even on a
   machine that does not have the raw DTA folder; when the folder IS present,
   test_real_org_method_lines_match_the_pinned_constant proves the constant
   still equals the file.
2. **Never raises.** Empty, truncated, random and wrong-format method files
   all have to come back as status="unrecognized" with an error string,
   because a method file we cannot read must never block the data next to it.
3. **Never silently drops.** An unknown opcode or an unreadable record keeps
   its place in the step list and is counted; dropping one would renumber
   every step after it and corrupt the program-to-data alignment.
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import pytest

from dta import dta_program as dp
from dta.dta_program import (
    CONFIDENCE_INVALID,
    CONFIDENCE_UNKNOWN,
    MTH_FORMAT_TAG,
    MTH_MAGIC,
    MTH_RECORD_SIZE,
    MethodProgram,
    MethodStep,
    PROGRAM_PARSED,
    PROGRAM_PARTIAL,
    PROGRAM_UNRECOGNIZED,
    STEP_END,
    STEP_FLOW,
    STEP_GAS,
    STEP_INVALID,
    STEP_ISOTHERMAL,
    STEP_RAMP,
    STEP_UNKNOWN,
    compare_programs,
    expand_program,
    locate_mth_step_table,
    merge_programs,
    org_method_lines_from_header,
    parse_mth_bytes,
    parse_mth_file,
    parse_org_method_lines,
    parse_step_text,
    program_fingerprint,
    program_from_header,
    scan_utf16_strings,
)

PRISM_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PRISM_ROOT / "EXAMPLES"

#: U+00B0, spelled as an escape so this test file stays pure ASCII and the
#: byte-identical comparison cannot be broken by a re-encoding tool.
DEG = "°"

#: The 19 OrgMethod lines of the real BM-0-cycles.txt, verbatim (the file is
#: UTF-16LE and holds a real U+00B0 -- there is no mojibake in it, see
#: test_degree_sign_variants_all_parse for the lossy paths that do produce
#: some). Pinned here so the grammar tests always run.
REAL_ORGMETHOD_LINES = (
    "1: Select gas 1",
    "2: Mass flow 100.00 mL/min",
    f"3: Ramp 20.00 {DEG}C/min to 1300.00 {DEG}C",
    "4: Isothermal for 20.00 min",
    f"5: Ramp 20.00 {DEG}C/min to 50.00 {DEG}C",
    "6: Isothermal for 20.00 min",
    f"7: Ramp 20.00 {DEG}C/min to 1300.00 {DEG}C",
    "8: Isothermal for 20.00 min",
    f"9: Ramp 15.00 {DEG}C/min to 50.00 {DEG}C",
    "10: Isothermal for 20.00 min",
    f"11: Ramp 15.00 {DEG}C/min to 1300.00 {DEG}C",
    "12: Isothermal for 20.00 min",
    f"13: Ramp 10.00 {DEG}C/min to 50.00 {DEG}C",
    "14: Isothermal for 20.00 min",
    f"15: Ramp 10.00 {DEG}C/min to 1300.00 {DEG}C",
    "16: Isothermal for 20.00 min",
    f"17: Ramp 10.00 {DEG}C/min to 50.00 {DEG}C",
    "18: Isothermal for 20.00 min",
    "19: Mass flow 0.00 mL/min",
)

#: Measured start of the real run: t = 0.0034 min, T = 19.014 C.
REAL_T0_C = 19.014
REAL_T_START_MIN = 0.0034
#: Programmed total over the 19 steps from that start (measured 890.371 min
#: of data; the run did NOT end early, it fell 0.345 min short inside the
#: final hold).
REAL_PROGRAM_TOTAL_MIN = 890.716


def _find_dta_bm_dir() -> Path:
    """The real BM DTA folder (BM-0-cycles.txt, cycles.mth) is not part of
    the repo. PRISM_DTA_BM_DIR overrides; full-file tests skip when neither
    is present. Mirrors the conftest helper of the same name so this module
    stays runnable on its own."""
    override = os.environ.get("PRISM_DTA_BM_DIR")
    return Path(override) if override else PRISM_ROOT.parents[2] / "DTA" / "BM"


DTA_BM_DIR = _find_dta_bm_dir()
requires_dta_bm = pytest.mark.skipif(
    not DTA_BM_DIR.is_dir(), reason="real DTA/BM folder not present")


def _cycles_mth_bytes() -> bytes:
    """The real 6776-byte cycles.mth: the committed EXAMPLES copy when it
    exists, else the raw DTA/BM original, else skip."""
    fixture = EXAMPLES_DIR / "DTA_cycles_example.mth"
    if fixture.is_file():
        return fixture.read_bytes()
    original = DTA_BM_DIR / "cycles.mth"
    if original.is_file():
        return original.read_bytes()
    pytest.skip("neither EXAMPLES/DTA_cycles_example.mth nor DTA/BM/cycles.mth is present")


def _real_org_method_lines() -> list:
    """Every OrgMethod line of the real UTF-16 BM-0-cycles.txt, in order."""
    path = DTA_BM_DIR / "BM-0-cycles.txt"
    text = path.read_bytes().decode("utf-16")
    return [ln.split("\t", 1)[1].strip()
            for ln in text.splitlines() if ln.startswith("OrgMethod\t")]


def _record(opcode: int, arg1: float = 0.0, arg2: float = 0.0) -> bytes:
    """One synthetic 24-byte .mth step record."""
    return struct.pack("<HHff", opcode, 0, arg1, arg2) + b"\x00" * 12


def _synthetic_mth(records, *, magic: bool = True, pad_before: int = 40,
                   pad_after: int = 48) -> tuple:
    """Build a .mth-shaped blob. Returns (blob, table_offset).

    `pad_before` is varied by the locator test: the real table sits at 654
    only because of that file's preamble, so a test that hardcoded an offset
    would pass while the product broke on the next file.
    """
    head = bytearray()
    if magic:
        head += MTH_MAGIC.encode("utf-16-le")
    head += b"\x00" * pad_before
    offset = len(head)
    return bytes(head) + b"".join(records) + b"\x00" * pad_after, offset


#: The four real opcodes, so the synthetic tables exercise the same decoder.
OP_GAS, OP_FLOW, OP_RAMP, OP_ISO = 0x2645, 0x26B8, 0x2635, 0x2625


# =============================================================================
# 1) The header parser (core.io_universal) keeps repeated keys
# =============================================================================

def test_ta_header_equals_plain_dict():
    """TAHeader must be a drop-in dict: tests/core pins `header == {}`."""
    from core import io_universal
    TAHeader = getattr(io_universal, "TAHeader", None)
    if TAHeader is None:
        pytest.skip("core.io_universal.TAHeader not built yet (build plan step 2)")
    assert TAHeader() == {}
    assert TAHeader([("a", "1")])["a"] == "1"


@requires_dta_bm
def test_org_method_lines_survive_the_header_parser():
    """All 19 OrgMethod lines reach program_from_header, in order."""
    from core import io_universal
    if getattr(io_universal, "TAHeader", None) is None:
        pytest.skip("core.io_universal.TAHeader not built yet (build plan step 2)")
    header, _cols, _df = io_universal.parse_ta_sdt_txt(DTA_BM_DIR / "BM-0-cycles.txt")
    assert list(header.all("OrgMethod")) == list(REAL_ORGMETHOD_LINES)
    # First-wins on the plain-dict view is unchanged: existing callers see
    # exactly what they saw before.
    assert header["OrgMethod"] == "1: Select gas 1"
    for key in ("Xcomment", "InstCalDate", "InstCalFile"):
        assert len(header.all(key)) == 3
    program = program_from_header(header)
    assert len(program.steps) == 19
    assert program.status == PROGRAM_PARSED


# =============================================================================
# 2) OrgMethod grammar
# =============================================================================

def test_org_method_program_has_nineteen_steps():
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES, name="BM-0-cycles")
    assert len(program.steps) == 19
    assert len(program.thermal_steps) == 16
    assert sum(1 for s in program.steps if s.kind == STEP_RAMP) == 8
    assert sum(1 for s in program.steps if s.kind == STEP_ISOTHERMAL) == 8
    assert program.status == PROGRAM_PARSED
    assert program.n_unrecognized == 0
    assert program.ramp_rates == (20.0, 20.0, 20.0, 15.0, 15.0, 10.0, 10.0, 10.0)
    assert [s.index for s in program.steps] == list(range(1, 20))
    assert program.describe() == "19 steps (8 ramps, 8 holds), status=parsed"
    assert bool(program) is True


def test_org_method_round_trips_byte_identically():
    """Every one of the 19 real lines re-renders to itself, degree sign
    included. This is what lets a .mth be checked against the header."""
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    rendered = [f"{s.index}: {s.render()}" for s in program.steps]
    assert rendered == list(REAL_ORGMETHOD_LINES)


def test_org_method_vocabulary_probes():
    """19 step forms, one per row: kind in, kind out."""
    probes = [
        ("Select gas 1", STEP_GAS),
        ("Select gas: 2", STEP_GAS),
        ("Mass flow 100.00 mL/min", STEP_FLOW),
        (f"Ramp 20.00 {DEG}C/min to 1300.00 {DEG}C", STEP_RAMP),
        (f"Ramp to 1300 {DEG}C at 10 {DEG}C/min", STEP_RAMP),   # word order
        ("Ramp 5.00 �C/min to 800.00 �C", STEP_RAMP),  # lossy decode
        ("Ramp 5.00 Â°C/min to 800.00 Â°C", STEP_RAMP),  # utf8 as latin1
        ("Ramp 5.00 ºC/min to 800.00 ºC", STEP_RAMP),  # masculine ordinal
        ("Ramp 5.00 C/min to 800.00 C", STEP_RAMP),              # no unit marker
        ("Isothermal for 20.00 min", STEP_ISOTHERMAL),
        ("Isothermal 5 min", STEP_ISOTHERMAL),
        (f"Equilibrate at 30.00 {DEG}C", dp.STEP_EQUILIBRATE),
        (f"Jump to 500.00 {DEG}C", dp.STEP_JUMP),
        ("Sampling interval 0.50 sec/pt", dp.STEP_OTHER),
        ("Data storage: On", dp.STEP_OTHER),
        ("Repeat segment 2 3 times", dp.STEP_REPEAT),
        ("End of method", STEP_END),
        ("Abort next segment if weight loss is more than 5.00 %", dp.STEP_OTHER),
        ("Increment temperature 5 C", STEP_UNKNOWN),
    ]
    assert len(probes) == 19
    got = [(text, parse_step_text(text).kind) for text, _ in probes]
    assert got == [(text, kind) for text, kind in probes]


def test_degree_sign_variants_all_parse_to_the_same_numbers():
    """Every degree look-alike, and none at all, give one identical step."""
    variants = [f"Ramp 5.00 {DEG}C/min to 800.00 {DEG}C",
                "Ramp 5.00 �C/min to 800.00 �C",
                "Ramp 5.00 Â°C/min to 800.00 Â°C",
                "Ramp 5.00 ºC/min to 800.00 ºC",
                "Ramp 5.00 C/min to 800.00 C"]
    for text in variants:
        step = parse_step_text(text, index=3)
        assert step.kind == STEP_RAMP
        assert step.rate_C_min == pytest.approx(5.0)
        assert step.target_C == pytest.approx(800.0)
        # Rendering is canonical, so a mojibake line comes back clean.
        assert step.render() == f"Ramp 5.00 {DEG}C/min to 800.00 {DEG}C"


def test_word_order_variant_keeps_rate_and_target_apart():
    step = parse_step_text(f"Ramp to 1300 {DEG}C at 10 {DEG}C/min", index=4)
    assert (step.rate_C_min, step.target_C) == (10.0, 1300.0)


def test_unrecognized_lines_are_kept_verbatim_and_flagged():
    """An unknown line keeps its place, its text and its number.

    Dropping it would renumber every step after it, which is how a program
    silently stops describing the run it came from.
    """
    lines = ["1: Select gas 1",
             "2: Increment temperature 5 C",
             f"3: Ramp 10.00 {DEG}C/min to 900.00 {DEG}C"]
    program = parse_org_method_lines(lines)
    assert len(program.steps) == 3
    assert program.status == PROGRAM_PARTIAL
    assert program.n_unrecognized == 1
    odd = program.steps[1]
    assert odd.kind == STEP_UNKNOWN
    assert odd.confidence == CONFIDENCE_UNKNOWN
    assert odd.text == "Increment temperature 5 C"
    assert odd.render() == "Increment temperature 5 C"
    assert program.steps[2].index == 3
    assert any("not understood" in w for w in program.warnings)


def test_unnumbered_lines_get_their_position():
    program = parse_org_method_lines(["Select gas 1", "Isothermal for 3.00 min"])
    assert [s.index for s in program.steps] == [1, 2]


def test_empty_line_list_is_unrecognized_not_an_exception():
    program = parse_org_method_lines([])
    assert program.status == PROGRAM_UNRECOGNIZED
    assert bool(program) is False
    assert program.error


# =============================================================================
# 3) The program clock
# =============================================================================

def test_program_schedule_total_matches_the_run():
    """890.716 programmed min against 890.371 min of data: the run did not
    end early, it fell 0.345 min short inside the final hold."""
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    assert program.total_minutes(T0=REAL_T0_C) == pytest.approx(
        REAL_PROGRAM_TOTAL_MIN, abs=0.01)
    entries = program.schedule(t0=REAL_T_START_MIN, T0=REAL_T0_C)
    assert len(entries) == 19
    assert entries[-1].t_end == pytest.approx(
        REAL_T_START_MIN + REAL_PROGRAM_TOTAL_MIN, abs=0.01)
    # A ramp is TIME-budgeted: |target - setpoint| / rate, on the clock
    # whether or not the furnace arrived.
    first_ramp = entries[2]
    assert first_ramp.step_index == 3
    assert first_ramp.duration_min == pytest.approx((1300.0 - REAL_T0_C) / 20.0, abs=1e-9)
    assert first_ramp.T_end == pytest.approx(1300.0)
    # Every window abuts the next: the clock never jumps or overlaps.
    for a, b in zip(entries, entries[1:]):
        assert b.t_start == pytest.approx(a.t_end, abs=1e-12)
        assert b.t_end >= b.t_start


def test_mass_flow_contributes_zero_time_and_never_lands_in_duration():
    """The defect this field exists to prevent.

    "Mass flow 100.00 mL/min" stored as duration_min=100.0 injects 100
    phantom minutes into an 890-minute program. The value belongs in
    `value`, and the two flow steps must cost exactly zero minutes.
    """
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    flows = [s for s in program.steps if s.kind == STEP_FLOW]
    assert [s.index for s in flows] == [2, 19]
    assert [s.value for s in flows] == [100.0, 0.0]
    assert all(math.isnan(s.duration_min) for s in flows)
    gas = program.steps[0]
    assert gas.kind == STEP_GAS and gas.value == 1.0 and math.isnan(gas.duration_min)

    entries = {e.step_index: e for e in program.schedule(T0=REAL_T0_C)}
    assert entries[2].duration_min == 0.0
    assert entries[19].duration_min == 0.0
    assert entries[1].duration_min == 0.0
    # And the phantom-minute failure would be visible as exactly +100.
    assert program.total_minutes(T0=REAL_T0_C) < REAL_PROGRAM_TOTAL_MIN + 1.0


def test_equilibrate_and_jump_take_no_programmed_time_and_say_so():
    program = parse_org_method_lines([
        f"1: Equilibrate at 100.00 {DEG}C",
        f"2: Jump to 200.00 {DEG}C",
        "3: Isothermal for 5.00 min",
    ])
    entries = program.schedule(t0=0.0, T0=25.0)
    assert entries[0].duration_min == 0.0
    assert entries[0].T_end == pytest.approx(100.0)
    assert entries[1].T_end == pytest.approx(200.0)
    assert program.total_minutes(T0=25.0) == pytest.approx(5.0)
    assert any("lower bound" in w for w in program.warnings)


def test_a_ramp_that_cannot_be_budgeted_is_reported_not_raised():
    """0 C/min never arrives. That is an arithmetic impossibility, so the
    clock refuses to invent a duration -- and reports it instead of raising."""
    program = parse_org_method_lines([
        f"1: Ramp 0.00 {DEG}C/min to 500.00 {DEG}C",
        "2: Isothermal for 10.00 min",
    ])
    assert program.status == PROGRAM_PARSED          # the LINE parsed fine
    entries = program.schedule(t0=0.0, T0=25.0)
    assert entries[0].duration_min == 0.0
    assert entries[0].T_end == pytest.approx(25.0)   # setpoint did not move
    assert program.total_minutes(T0=25.0) == pytest.approx(10.0)
    assert any("no usable rate" in w for w in program.warnings)


def test_schedule_survives_nonsense_arguments():
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    entries = program.schedule(t0=float("nan"), T0=float("nan"))
    assert len(entries) == 19
    assert entries[0].t_start == 0.0


# =============================================================================
# 4) Repeat expansion
# =============================================================================

def test_repeat_segment_expands_to_physical_steps():
    """Equilibrate / Ramp / Iso / Ramp / Repeat 2 x3 / End -> 14 steps."""
    program = parse_org_method_lines([
        f"1: Equilibrate at 30.00 {DEG}C",
        f"2: Ramp 10.00 {DEG}C/min to 600.00 {DEG}C",
        "3: Isothermal for 5.00 min",
        f"4: Ramp 10.00 {DEG}C/min to 100.00 {DEG}C",
        "5: Repeat segment 2 3 times",
        "6: End of method",
    ])
    steps, notes = program.physical_steps()
    assert len(steps) == 14
    assert [s.index for s in steps] == [1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 6]
    assert [s.repeat_pass for s in steps] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
    assert notes == ()
    # The repeat instruction itself is flow control, never a physical step.
    assert all(s.kind != dp.STEP_REPEAT for s in steps)
    # ...and the written list is untouched.
    assert len(program.steps) == 6


def test_a_satisfied_repeat_rearms_for_an_enclosing_loop():
    """Nested cycles: the inner block runs once per OUTER pass, not once."""
    program = parse_org_method_lines([
        "1: Isothermal for 1.00 min",
        "2: Isothermal for 2.00 min",
        "3: Repeat segment 2 1 times",
        "4: Repeat segment 1 1 times",
    ])
    steps, notes = program.physical_steps()
    assert [s.index for s in steps] == [1, 2, 2, 1, 2, 2]
    assert notes == ()


def test_forward_and_missing_repeat_targets_are_ignored_with_a_note():
    program = parse_org_method_lines([
        "1: Isothermal for 1.00 min",
        "2: Repeat segment 3 2 times",     # forward: would skip work
        "3: Repeat segment 9 2 times",     # missing target
    ])
    steps, notes = program.physical_steps()
    assert [s.index for s in steps] == [1]
    assert len(notes) == 2
    assert any("forward repeat is ignored" in n for n in notes)
    assert any("does not exist" in n for n in notes)


def test_runaway_repeat_is_capped():
    program = parse_org_method_lines([
        "1: Isothermal for 1.00 min",
        "2: Repeat segment 1 999999 times",
    ])
    steps, notes = program.physical_steps(max_steps=50)
    assert len(steps) == 50
    assert any("capped at 50" in n for n in notes)


def test_expand_program_is_a_no_op_without_repeats():
    program = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    steps, notes = expand_program(program.steps)
    assert steps == program.steps
    assert notes == ()


def test_end_of_method_stops_expansion_and_says_what_is_unreachable():
    program = parse_org_method_lines([
        "1: Isothermal for 1.00 min",
        "2: End of method",
        "3: Isothermal for 9.00 min",
    ])
    steps, notes = program.physical_steps()
    assert [s.index for s in steps] == [1, 2]
    assert any("unreachable" in n for n in notes)


# =============================================================================
# 5) The .mth binary
# =============================================================================

def test_mth_step_table_is_located_by_scan_not_a_fixed_offset():
    """The same table found at three different offsets."""
    records = [_record(OP_GAS, 0.0), _record(OP_RAMP, 20.0, 1300.0),
               _record(OP_ISO, 20.0), _record(OP_FLOW, 50.0)]
    for pad in (0, 40, 654 - 2 * len(MTH_MAGIC), 1024):
        blob, offset = _synthetic_mth(records, pad_before=pad)
        candidates = locate_mth_step_table(blob)
        assert candidates, f"no table found with pad_before={pad}"
        assert candidates[0] == (offset, 4, 4, 0)
        program = parse_mth_bytes(blob)
        assert program.status == PROGRAM_PARSED
        assert len(program.steps) == 4
        assert program.meta["table_offset"] == offset


def test_mth_decodes_the_four_known_opcodes():
    blob, _off = _synthetic_mth([
        _record(OP_GAS, 0.0),              # 0-based in binary, 1-based in text
        _record(OP_FLOW, 100.0),
        _record(OP_RAMP, 20.0, 1300.0),
        _record(OP_ISO, 20.0),
    ])
    program = parse_mth_bytes(blob, name="synthetic")
    assert [s.kind for s in program.steps] == [STEP_GAS, STEP_FLOW, STEP_RAMP,
                                               STEP_ISOTHERMAL]
    assert [s.render() for s in program.steps] == [
        "Select gas 1",
        "Mass flow 100.00 mL/min",
        f"Ramp 20.00 {DEG}C/min to 1300.00 {DEG}C",
        "Isothermal for 20.00 min",
    ]
    assert [s.opcode for s in program.steps] == [OP_GAS, OP_FLOW, OP_RAMP, OP_ISO]
    assert all(s.source == dp.SOURCE_MTH for s in program.steps)


def test_mth_format_tag_is_pinned():
    """The .mth decode is reverse-engineered from ONE file.

    The tag travels with every program built from binary so that a second
    .mth which decodes wrongly forces a v2 with its own rules instead of a
    quiet change to what v1 means.
    """
    assert MTH_FORMAT_TAG == "ta_mth_reverse_engineered_v1"
    blob, _off = _synthetic_mth([_record(OP_ISO, 5.0), _record(OP_ISO, 6.0)])
    assert parse_mth_bytes(blob).meta["format"] == MTH_FORMAT_TAG
    # Even a failure carries it, so a log line can say which decoder tried.
    assert parse_mth_bytes(b"").meta["format"] == MTH_FORMAT_TAG


def test_mth_shifted_table_is_still_found():
    """Seven junk bytes in front: the table is at an ODD offset now, and the
    scan must still find it (byte-stride search, not 24-byte alignment)."""
    blob = _cycles_mth_bytes()
    shifted = b"\x00" * 7 + blob
    candidates = locate_mth_step_table(shifted)
    assert candidates[0] == (661, 19, 19, 0)
    program = parse_mth_bytes(shifted)
    assert program.status == PROGRAM_PARSED
    assert len(program.steps) == 19


@pytest.mark.parametrize("bad_index", [0, 9, 18])
def test_mth_unknown_opcode_is_kept_and_flagged(bad_index):
    """An opcode we do not know is a step we cannot name -- not a step that
    did not happen. It keeps its slot, so every later step keeps its number."""
    records = [_record(OP_ISO, float(i + 1)) for i in range(19)]
    records[bad_index] = _record(0x9999, 1.0, 2.0)
    blob, offset = _synthetic_mth(records)
    program = parse_mth_bytes(blob)
    assert len(program.steps) == 19
    assert program.status == PROGRAM_PARTIAL
    assert program.n_unrecognized == 1
    odd = program.steps[bad_index]
    assert odd.kind == STEP_UNKNOWN
    assert odd.confidence == CONFIDENCE_UNKNOWN
    assert odd.opcode == 0x9999
    assert "0x9999" in odd.text
    assert program.meta["n_known"] == 18
    assert program.meta["n_nonstandard"] == 1
    # The surviving steps kept their positions and values.
    assert program.steps[-1].index == 19
    for i, step in enumerate(program.steps):
        if i != bad_index:
            assert step.duration_min == pytest.approx(i + 1)


def test_mth_unreadable_record_inside_the_table_is_bridged_not_resynced():
    """A NaN argument makes record 3 unreadable. The table is still 19 long
    and record 3 is an explicit STEP_INVALID, because resyncing past it would
    renumber steps 4..19."""
    records = [_record(OP_ISO, float(i + 1)) for i in range(19)]
    records[2] = _record(OP_ISO, float("nan"), 0.0)
    blob, offset = _synthetic_mth(records)
    program = parse_mth_bytes(blob)
    assert len(program.steps) == 19
    assert program.status == PROGRAM_PARTIAL
    assert program.n_unrecognized == 1
    broken = program.steps[2]
    assert broken.kind == STEP_INVALID
    assert broken.confidence == CONFIDENCE_INVALID
    assert str(offset + 2 * MTH_RECORD_SIZE) in broken.text
    assert program.steps[3].duration_min == pytest.approx(4.0)


def test_mth_denormal_floats_are_rejected():
    """The denormal guard, on its own and in the locator.

    Four near-zero bytes decode as a DENORMAL float (~1e-45) which is finite
    and would pass a naive "is it a number?" test. In the real cycles.mth,
    offset 110 is exactly that shape. Here the same pattern is placed at
    stride 24 right after a real table: without the guard it is absorbed as a
    phantom 4th step reading "Isothermal for 8.4e-45 min".
    """
    assert dp._mth_float_is_plausible(0.0) is True
    assert dp._mth_float_is_plausible(20.0) is True
    assert dp._mth_float_is_plausible(8.407790785948902e-45) is False
    assert dp._mth_float_is_plausible(5.605193857299268e-45) is False
    assert dp._mth_float_is_plausible(float("nan")) is False
    assert dp._mth_float_is_plausible(float("inf")) is False
    assert dp._mth_float_is_plausible(1e9) is False

    denormal = struct.pack("<HH", OP_ISO, 0) + bytes([6, 0, 0, 0, 4, 0, 0, 0]) + b"\x00" * 12
    assert len(denormal) == MTH_RECORD_SIZE
    real = [_record(OP_RAMP, 20.0, 1300.0), _record(OP_ISO, 20.0), _record(OP_ISO, 5.0)]
    blob, offset = _synthetic_mth(real + [denormal])
    assert locate_mth_step_table(blob)[0] == (offset, 3, 3, 0)
    assert len(parse_mth_bytes(blob).steps) == 3


def test_mth_garbage_never_raises():
    """Empty, all-zero, pseudo-random, truncated, wrong-format, missing."""
    import random

    rng = random.Random(20260926)
    cases = {
        "empty": b"",
        "all zero": bytes(4096),
        "pseudo random": bytes(rng.randrange(256) for _ in range(4096)),
        "truncated": _cycles_mth_bytes()[:200],
        "wrong format": b"%PDF-1.7\n" + bytes(rng.randrange(256) for _ in range(512)),
        "one byte": b"\x01",
    }
    for label, blob in cases.items():
        program = parse_mth_bytes(blob, name=label)
        assert program.status == PROGRAM_UNRECOGNIZED, label
        assert bool(program) is False, label
        assert program.error, label
        assert program.steps == (), label
        assert program.meta["format"] == MTH_FORMAT_TAG, label

    missing = parse_mth_file(DTA_BM_DIR / "definitely_not_here.mth")
    assert missing.status == PROGRAM_UNRECOGNIZED
    assert "cannot read" in (missing.error or "")
    assert parse_mth_file(None).status == PROGRAM_UNRECOGNIZED
    assert parse_mth_bytes(object()).status == PROGRAM_UNRECOGNIZED


def test_mth_without_the_magic_string_still_parses_but_warns():
    blob, _off = _synthetic_mth([_record(OP_ISO, 5.0), _record(OP_ISO, 6.0)],
                                magic=False)
    program = parse_mth_bytes(blob)
    assert program.status == PROGRAM_PARSED
    assert program.meta["magic_found"] is False
    assert any("signature was not found" in w for w in program.warnings)


def test_scan_utf16_strings_only_walks_even_offsets():
    """Odd-aligned scanning drifts parity and silently misses ProcName."""
    blob = (b"\x00" + "ODD".encode("utf-16-le") + b"\x00" * 9
            + "EVEN".encode("utf-16-le") + b"\x00" * 8)
    found = scan_utf16_strings(blob)
    assert [text for _off, text in found] == ["EVEN"]
    offset = found[0][0]
    assert offset % 2 == 0
    assert blob[offset:offset + 8] == "EVEN".encode("utf-16-le")
    assert scan_utf16_strings(b"") == []
    assert scan_utf16_strings(None) == []


# =============================================================================
# 6) Program vs program
# =============================================================================

def test_program_fingerprint_ignores_gas_flow_and_formatting():
    thermal_only = [ln for ln in REAL_ORGMETHOD_LINES
                    if "gas" not in ln and "Mass flow" not in ln]
    full = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    trimmed = parse_org_method_lines(thermal_only)
    assert program_fingerprint(full) == program_fingerprint(trimmed)
    assert program_fingerprint(full).startswith(f"R20.00>1300.0|I20.00|")
    # A different furnace program gives a different signature.
    other = parse_org_method_lines([f"1: Ramp 10.00 {DEG}C/min to 1300.00 {DEG}C"])
    assert program_fingerprint(other) != program_fingerprint(full)
    assert program_fingerprint(parse_mth_bytes(b"")) == ""


def test_fingerprint_unrolls_repeats_so_a_loop_equals_its_longhand():
    looped = parse_org_method_lines([
        f"1: Ramp 10.00 {DEG}C/min to 600.00 {DEG}C",
        "2: Isothermal for 5.00 min",
        "3: Repeat segment 1 1 times",
    ])
    longhand = parse_org_method_lines([
        f"1: Ramp 10.00 {DEG}C/min to 600.00 {DEG}C",
        "2: Isothermal for 5.00 min",
        f"3: Ramp 10.00 {DEG}C/min to 600.00 {DEG}C",
        "4: Isothermal for 5.00 min",
    ])
    assert program_fingerprint(looped) == program_fingerprint(longhand)
    assert compare_programs(looped, longhand).verdict == "identical"


def test_compare_programs_detects_the_wrong_method_file():
    org = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    same = parse_mth_bytes(_cycles_mth_bytes())
    verdict = compare_programs(org, same)
    assert verdict.verdict == "identical"
    assert verdict.score == pytest.approx(1.0)
    assert verdict.is_same_method is True
    assert len(verdict.rows) == 16
    assert {row[3] for row in verdict.rows} == {"same"}

    wrong = parse_org_method_lines([f"1: Ramp 10.00 {DEG}C/min to 1300.00 {DEG}C",
                                    "2: Isothermal for 1.00 min"])
    bad = compare_programs(org, wrong)
    assert bad.verdict == "mismatch"
    assert bad.score == 0.0
    assert bad.is_same_method is False
    assert bad.reasons
    assert [row[3] for row in bad.rows][:2] == ["differs", "differs"]
    assert [row[3] for row in bad.rows][2:] == ["only_a"] * 14


def test_compare_programs_is_tolerant_but_not_blind():
    a = parse_org_method_lines([f"1: Ramp 20.00 {DEG}C/min to 1300.00 {DEG}C"])
    near = parse_org_method_lines([f"1: Ramp 20.02 {DEG}C/min to 1300.40 {DEG}C"])
    far = parse_org_method_lines([f"1: Ramp 20.40 {DEG}C/min to 1300.00 {DEG}C"])
    assert compare_programs(a, near).verdict == "equivalent"
    assert compare_programs(a, near).is_same_method is True
    assert compare_programs(a, far).verdict == "mismatch"


def test_comparing_against_an_unreadable_program_is_incomparable():
    """"Could not read it" must never be reported as "they match"."""
    org = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    broken = parse_mth_bytes(b"")
    verdict = compare_programs(org, broken)
    assert verdict.verdict == "incomparable"
    assert verdict.is_same_method is False
    assert verdict.score == 0.0
    assert compare_programs(broken, broken).verdict == "incomparable"
    assert compare_programs(org, None).verdict == "incomparable"


def test_merge_programs_never_overwrites_orgmethod():
    org = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    mth = parse_mth_bytes(_cycles_mth_bytes())
    merged = merge_programs(org, mth)
    assert merged.source == "orgmethod+mth"
    assert len(merged.steps) == 19
    assert {s.source for s in merged.steps} == {dp.SOURCE_BOTH}
    assert [f"{s.index}: {s.render()}" for s in merged.steps] == list(REAL_ORGMETHOD_LINES)

    wrong = parse_org_method_lines([f"{i}: Ramp 3.00 {DEG}C/min to 700.00 {DEG}C"
                                    for i in range(1, 20)])
    guarded = merge_programs(org, wrong)
    assert [f"{s.index}: {s.render()}" for s in guarded.steps] == list(REAL_ORGMETHOD_LINES)
    assert any("disagrees" in w for w in guarded.warnings)
    assert all(s.source != dp.SOURCE_MTH for s in guarded.steps)


def test_merge_falls_back_to_whichever_side_exists():
    org = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    empty = parse_mth_bytes(b"")
    assert merge_programs(org, empty) is not None
    assert len(merge_programs(org, empty).steps) == 19
    only_mth = merge_programs(empty, parse_mth_bytes(_cycles_mth_bytes()))
    assert len(only_mth.steps) == 19
    assert any("not confirmed" in w for w in only_mth.warnings)
    assert merge_programs(empty, empty).status == PROGRAM_UNRECOGNIZED


# =============================================================================
# 7) Headers and persistence
# =============================================================================

def test_legacy_collapsed_header_degrades_loudly():
    """A plain dict keeps the FIRST of 19 repeated OrgMethod keys.

    One step that looks perfectly parsed is the dangerous outcome, so the
    program is demoted to partial and the warning names the cause.
    """
    program = program_from_header({"OrgMethod": "1: Select gas 1", "ProcName": "Ramp"})
    assert program.status == PROGRAM_PARTIAL
    assert len(program.steps) == 1
    assert program.proc_name == "Ramp"
    assert program.warnings
    assert "repeated key" in program.warnings[0]
    assert program.meta["provenance"] == "collapsed_key"


def test_program_from_header_reads_header_lines_and_header_objects():
    header_lines = [["Sample", "BM-0-cycles"]] + [["OrgMethod", ln]
                                                  for ln in REAL_ORGMETHOD_LINES]
    from_lines = program_from_header({"OrgMethod": REAL_ORGMETHOD_LINES[0]}, header_lines)
    assert len(from_lines.steps) == 19
    assert from_lines.status == PROGRAM_PARSED
    assert from_lines.meta["provenance"] == "header_lines"
    assert org_method_lines_from_header({}, header_lines) == list(REAL_ORGMETHOD_LINES)

    class _Header(dict):
        """Stand-in for core.io_universal.TAHeader."""

        def __init__(self, pairs):
            super().__init__()
            self.lines = list(pairs)
            for key, value in pairs:
                self.setdefault(key, value)

        def all(self, key):
            return [v for k, v in self.lines if k == key]

    header = _Header([("OrgMethod", ln) for ln in REAL_ORGMETHOD_LINES])
    from_object = program_from_header(header)
    assert len(from_object.steps) == 19
    assert from_object.status == PROGRAM_PARSED
    assert from_object.meta["provenance"] == "header_object"


def test_header_without_any_org_method_is_unrecognized():
    program = program_from_header({"Sample": "x"})
    assert program.status == PROGRAM_UNRECOGNIZED
    assert bool(program) is False
    assert "no OrgMethod" in (program.error or "")
    assert program_from_header(None).status == PROGRAM_UNRECOGNIZED
    assert org_method_lines_from_header({}) == []


def test_program_round_trips_through_json():
    """NaN must not reach json.dumps: a bare NaN token is invalid JSON and a
    .prism project has to survive a strict reader."""
    program = parse_mth_bytes(_cycles_mth_bytes(), name="cycles")
    payload = json.dumps(program.to_dict(), allow_nan=False)
    restored = MethodProgram.from_dict(json.loads(payload))
    assert restored.status == program.status
    assert restored.meta["format"] == MTH_FORMAT_TAG
    assert [f"{s.index}: {s.render()}" for s in restored.steps] == \
           [f"{s.index}: {s.render()}" for s in program.steps]
    assert math.isnan(restored.steps[0].duration_min)
    assert restored.total_minutes(T0=REAL_T0_C) == pytest.approx(
        program.total_minutes(T0=REAL_T0_C))
    assert MethodProgram.from_dict({}).status == PROGRAM_UNRECOGNIZED
    assert MethodProgram.from_dict("junk").status == PROGRAM_UNRECOGNIZED
    assert MethodStep.from_dict({}).kind == STEP_UNKNOWN


# =============================================================================
# 8) Against the real files (skipped when the DTA/BM folder is absent)
# =============================================================================

@requires_dta_bm
def test_real_org_method_lines_match_the_pinned_constant():
    """Proves REAL_ORGMETHOD_LINES above is still the real file."""
    assert _real_org_method_lines() == list(REAL_ORGMETHOD_LINES)


@requires_dta_bm
def test_real_mth_renders_the_orgmethod_lines_byte_identically():
    """cycles.mth vs the 19 OrgMethod lines of BM-0-cycles.txt: no diff.

    Two independent encodings of one method -- a binary the operator saved
    and text the instrument wrote while running -- agreeing character for
    character is the whole evidence that the .mth decode is right.
    """
    program = parse_mth_file(DTA_BM_DIR / "cycles.mth")
    assert program.status == PROGRAM_PARSED
    assert len(program.steps) == 19
    rendered = [f"{s.index}: {s.render()}" for s in program.steps]
    expected = _real_org_method_lines()
    diff = [(a, b) for a, b in zip(rendered, expected) if a != b]
    assert diff == [], f"{len(diff)} of 19 lines differ: {diff[:3]}"
    assert rendered == expected
    assert program.proc_name == "Ramp"


@requires_dta_bm
def test_real_mth_table_is_found_by_scan_at_654():
    blob = (DTA_BM_DIR / "cycles.mth").read_bytes()
    assert len(blob) == 6776
    assert locate_mth_step_table(blob) == [(654, 19, 19, 0)]
    program = parse_mth_bytes(blob)
    assert program.meta["table_offset"] == 654
    assert program.meta["n_candidates"] == 1
    assert program.meta["magic_found"] is True


@requires_dta_bm
def test_real_mth_offset_110_is_not_a_record():
    """The denormal false positive, in the real file.

    Offset 110 has opcode 0x1207, a zero uint16, twelve zero bytes and two
    denormal floats -- a perfect shape match. The magnitude floor is what
    keeps it out of the record set.
    """
    blob = (DTA_BM_DIR / "cycles.mth").read_bytes()
    opcode, pad, a1, a2 = struct.unpack_from("<HHff", blob, 110)
    assert (opcode, pad) == (0x1207, 0)
    assert blob[110 + 12:110 + 24] == b"\x00" * 12
    assert 0 < abs(a1) < 1e-40 and 0 < abs(a2) < 1e-40
    records, _scanned = dp._scan_mth_records(blob)
    assert 110 not in records
    assert sorted(records) == list(range(654, 654 + 19 * MTH_RECORD_SIZE, MTH_RECORD_SIZE)) \
        + [1087]


@requires_dta_bm
def test_real_mth_matches_the_recorded_program_and_the_run_length():
    org = parse_org_method_lines(_real_org_method_lines(), name="BM-0-cycles")
    mth = parse_mth_file(DTA_BM_DIR / "cycles.mth")
    assert compare_programs(org, mth).verdict == "identical"
    assert org.total_minutes(T0=REAL_T0_C) == pytest.approx(890.716, abs=0.01)
    assert mth.total_minutes(T0=REAL_T0_C) == pytest.approx(890.716, abs=0.01)


@requires_dta_bm
def test_single_ramp_bm0_header_is_a_different_program():
    """BM-0.txt is one ramp to 1300 C: it must NOT match the cyclic method."""
    path = DTA_BM_DIR / "BM-0.txt"
    if not path.is_file():
        pytest.skip("BM-0.txt not present")
    text = path.read_bytes().decode("utf-16", errors="replace")
    lines = [ln.split("\t", 1)[1].strip()
             for ln in text.splitlines() if ln.startswith("OrgMethod\t")]
    single = parse_org_method_lines(lines, name="BM-0")
    cyclic = parse_org_method_lines(REAL_ORGMETHOD_LINES)
    assert single.status in (PROGRAM_PARSED, PROGRAM_PARTIAL)
    assert compare_programs(cyclic, single).is_same_method is False


# ===========================================================================
# Adversarial-review regressions (2026-09-26)
# ===========================================================================

def test_a_truncated_mth_reports_partial_not_parsed():
    """`MethodProgram.status` is this module's honesty channel (M9/D2).

    A step table with no room for one more record after it was cut off by
    the end of the buffer, not by TA: the real 6776-byte cycles.mth carries
    1110 bytes of table and a 5666-byte trailer. Before the fix every
    truncation came back `status="parsed"`, so an interrupted copy off an
    instrument share gave an 890 min run a 146 min program clock and NaN
    programmed rates on three of four ramps, with `warnings == ()`.
    """
    blob = _cycles_mth_bytes()
    intact = dp.parse_mth_bytes(blob)
    assert intact.status == PROGRAM_PARSED
    assert len(intact.steps) == 19
    assert intact.meta["table_end"] == 1110
    assert intact.meta["n_bytes"] == 6776
    assert intact.meta["table_reaches_eof"] is False
    assert intact.meta["trailer_bytes"] == 6776 - 1110

    # every cut, on a record boundary AND mid-record: an interrupted copy
    # lands at an arbitrary byte, and the scanner then stops at the last
    # COMPLETE record leaving 1-23 orphan bytes, so an exact-EOF test would
    # miss 23 of every 24 truncations
    for n_records in (3, 5, 12):
        for orphan in (0, 1, 11, 23):
            cut = blob[:654 + 24 * n_records + orphan]
            prog = dp.parse_mth_bytes(cut)
            assert len(prog.steps) == n_records, (n_records, orphan)
            assert prog.status == PROGRAM_PARTIAL, (n_records, orphan)
            assert prog.meta["table_reaches_eof"] is True
            assert prog.meta["trailer_bytes"] == orphan
            assert any("truncated" in w for w in prog.warnings), prog.warnings
            # the steps it DID recover are still the real ones
            assert prog.steps[0].render() == intact.steps[0].render()

    # a trailer of at least one record's worth reads as deliberate again, so
    # an ordinary file is never called truncated (the real one above has a
    # 5666-byte trailer; a synthetic one needs only 24)
    records = [_record(OP_RAMP, 20.0, 1300.0), _record(OP_ISO, 20.0),
               _record(OP_RAMP, 20.0, 50.0)]
    roomy, _off = _synthetic_mth(records, pad_after=48)
    assert dp.parse_mth_bytes(roomy).meta["table_reaches_eof"] is False
    assert dp.parse_mth_bytes(roomy).status == PROGRAM_PARSED
    tight, _off = _synthetic_mth(records, pad_after=0)
    assert dp.parse_mth_bytes(tight).meta["table_reaches_eof"] is True
    assert dp.parse_mth_bytes(tight).status == PROGRAM_PARTIAL


def test_parse_mth_file_reports_an_unreadable_path_of_any_kind():
    """`except OSError` was one clause too narrow.

    A path carrying an embedded NUL reaches the OS layer as a ValueError, and
    the module docstring says nothing here raises.
    """
    for bad in ("\x00badnull", "C:\\definitely\\not\\here.mth", "", "x" * 5000):
        prog = dp.parse_mth_file(bad)
        assert prog.status == PROGRAM_UNRECOGNIZED
        assert prog.error


def test_a_ramp_without_a_target_is_warned_about_without_a_refusal():
    """The warning says what happened to the step, not that the tool declined.

    Same reasoning as SPEC_rate_series' wording law: the user asked to be
    able to process this data.
    """
    steps = (MethodStep(index=1, kind=STEP_RAMP, rate_C_min=10.0,
                        target_C=float("nan")),)
    notes = dp._program_warnings(steps)
    text = " ".join(notes)
    assert "no usable target" in text
    assert "counted as zero" in text
    assert "cannot" not in text.lower()
