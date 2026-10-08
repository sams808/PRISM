"""Tests for the committed cyclic-DTA fixtures themselves.

EXAMPLES/DTA_cycles_example.txt is not a hand-written toy file: it is every
4th data row of a real 18996-row TA SDT export (BM-0-cycles.txt, 890 min, 4
heat/cool cycles), carrying its 62-line header verbatim. Every other cyclic
test — program parsing, segmentation, alignment, the rate series — reads its
"real world" out of this one file, so a silent change to it would quietly
change what all of those tests mean.

Hence these tests, which assert nothing about science and everything about the
bytes: exact size, exact row count, the UTF-8 BOM, the 19 repeated OrgMethod
lines, the 7 signal declarations and the U+00B0 degree sign. Regenerating the
fixture with a different `--every` (or with a tool that rewrites the header
instead of copying it) fails here, loudly, at the source, instead of showing
up as an inexplicable segment count three modules away.
"""
from __future__ import annotations

import re

import pytest

import core.io_universal as iou
from conftest import requires_dta_bm

# Pinned by measurement on the 2026-09-26 regeneration from the real file.
# `--every 4` over 18996 rows: 18996 = 4 x 4749 exactly, no ragged tail.
EXPECTED_BYTES = 318_440
EXPECTED_DATA_ROWS = 4749
EXPECTED_HEADER_LINES = 62          # includes the StartOfData line itself
EXPECTED_TOTAL_LINES = 4811         # 62 + 4749
EXPECTED_ORGMETHOD_LINES = 19       # gas + flow + 8 ramps + 8 isotherms + closing flow
EXPECTED_SIGNALS = 7
EXPECTED_HEADER_KEYS = 35
EXPECTED_MTH_BYTES = 6776

# UTF-8 BOM + "CLOSED" + CRLF — byte-for-byte the opening of the existing
# EXAMPLES/DTA_example.txt, so EXAMPLES/ has exactly one encoding story.
EXPECTED_MAGIC = bytes.fromhex("efbbbf434c4f5345440d0a")


@pytest.fixture(scope="module")
def fixture_bytes(dta_cycles_path) -> bytes:
    return dta_cycles_path.read_bytes()


@pytest.fixture(scope="module")
def fixture_lines(fixture_bytes) -> list[str]:
    return fixture_bytes.decode("utf-8-sig").splitlines()


@pytest.fixture(scope="module")
def fixture_header(fixture_lines) -> list[str]:
    """Header lines up to and including StartOfData, split the same way
    core.io_universal.parse_ta_sdt_txt splits them (whitespace- and
    case-insensitive marker match), so these tests cannot disagree with the
    parser about where the header ends."""
    sod = next(i for i, ln in enumerate(fixture_lines)
               if "startofdata" in re.sub(r"\s+", "", ln).lower())
    return fixture_lines[: sod + 1]


# --------------------------------------------------------------------------
# Byte-level pins
# --------------------------------------------------------------------------

def test_cycles_fixtures_are_present(dta_cycles_path, dta_cycles_mth_path):
    assert dta_cycles_path.is_file(), f"missing committed fixture: {dta_cycles_path}"
    assert dta_cycles_mth_path.is_file(), f"missing committed fixture: {dta_cycles_mth_path}"


def test_fixture_size_is_pinned(fixture_bytes):
    assert len(fixture_bytes) == EXPECTED_BYTES


def test_fixture_starts_with_a_utf8_bom_like_the_other_dta_example(fixture_bytes, dta_example_path):
    assert fixture_bytes[:11] == EXPECTED_MAGIC
    assert dta_example_path.read_bytes()[:11] == EXPECTED_MAGIC


def test_fixture_line_endings_are_all_crlf(fixture_bytes):
    """A stray bare LF would mean the file was rewritten by an editor rather
    than by the generator — and on a TA export the line terminator is part of
    what the fixture is supposed to reproduce."""
    assert fixture_bytes.count(b"\n") == fixture_bytes.count(b"\r\n") == EXPECTED_TOTAL_LINES
    assert fixture_bytes.endswith(b"\r\n")


def test_fixture_row_counts_are_pinned(fixture_lines, fixture_header):
    assert len(fixture_header) == EXPECTED_HEADER_LINES
    assert fixture_header[-1].strip() == "StartOfData"
    data = [ln for ln in fixture_lines[len(fixture_header):] if ln.strip()]
    assert len(data) == EXPECTED_DATA_ROWS
    assert len(fixture_lines) == EXPECTED_TOTAL_LINES


# --------------------------------------------------------------------------
# Header content — the reason this fixture exists
# --------------------------------------------------------------------------

def test_fixture_keeps_all_nineteen_orgmethod_lines(fixture_header):
    """The whole thermal program lives in these 19 repeated-key lines. The
    generator copies the header verbatim precisely so they survive; a
    header-rebuilding generator would collapse them to one."""
    org = [ln for ln in fixture_header if ln.startswith("OrgMethod\t")]
    assert len(org) == EXPECTED_ORGMETHOD_LINES
    assert org[0] == "OrgMethod\t1: Select gas 1"
    assert org[2].startswith("OrgMethod\t3: Ramp 20.00")
    assert org[-1] == "OrgMethod\t19: Mass flow 0.00 mL/min"


def test_fixture_keeps_every_other_repeated_header_key(fixture_header):
    for key in ("Xcomment", "InstCalDate", "InstCalFile"):
        n = sum(1 for ln in fixture_header if ln.startswith(key + "\t"))
        assert n == 3, f"{key}: expected 3 lines, found {n}"


def test_fixture_declares_seven_signals_with_the_degree_sign(fixture_header):
    """U+00B0 (and U+00B5 on Sig6) must survive the UTF-16 -> UTF-8
    re-encoding: the canonical-column mapper keys on these exact names, so a
    mojibaked degree sign turns 'Temperature (°C)' into an unrecognised
    column and every downstream test loses its abscissa."""
    sigs = [ln for ln in fixture_header if re.match(r"^Sig\d+\t", ln)]
    assert len(sigs) == EXPECTED_SIGNALS
    assert sigs[0] == "Sig1\tTime (min)"
    assert sigs[1] == "Sig2\tTemperature (°C)"
    assert sigs[4] == "Sig5\tTemperature Difference (°C)"
    assert sigs[5] == "Sig6\tTemperature Difference (µV)"


# --------------------------------------------------------------------------
# What the production parser makes of it
# --------------------------------------------------------------------------

def test_parser_reads_the_fixture_as_a_full_ta_export(dta_cycles_path):
    header, colnames, df = iou.parse_ta_sdt_txt(dta_cycles_path)
    assert len(header) == EXPECTED_HEADER_KEYS
    assert colnames == [
        "Time (min)",
        "Temperature (°C)",
        "Weight (mg)",
        "Heat Flow (mW)",
        "Temperature Difference (°C)",
        "Temperature Difference (µV)",
        "Sample Purge Flow (mL/min)",
    ]
    assert df.shape == (EXPECTED_DATA_ROWS, EXPECTED_SIGNALS)
    assert df.notna().all().all(), "the fixture must have no non-finite rows"


def test_fixture_covers_the_whole_cyclic_run(dta_cycles_path):
    """Content pins, so a fixture regenerated from the wrong source file (or
    truncated) fails here. NOTE t_end is 890.2292, not the real run's
    890.3708: [::4] keeps indices 0, 4, ... 18992, so the fixture stops one
    stride short of the final row. Tests that pin the end of the run must use
    this value, not the real file's."""
    _, colnames, df = iou.parse_ta_sdt_txt(dta_cycles_path)
    t = df[colnames[0]].to_numpy(dtype=float)
    temp = df[colnames[1]].to_numpy(dtype=float)
    assert t[0] == pytest.approx(0.003444444, abs=1e-9)
    assert t[-1] == pytest.approx(890.2292, abs=1e-4)
    assert float(temp.min()) == pytest.approx(19.02074, abs=1e-4)
    assert float(temp.max()) == pytest.approx(1298.642, abs=1e-3)
    # 0.1875 min/pt = 4 x the real 0.046875 min/pt. 1.5x below the 0.28 min
    # sampling at which an unguarded rate estimator fragments this run.
    import numpy as np
    assert float(np.median(np.diff(t))) == pytest.approx(0.1875, abs=1e-4)


# --------------------------------------------------------------------------
# The binary method fixture
# --------------------------------------------------------------------------

def test_method_fixture_is_the_expected_binary(dta_cycles_mth_path):
    """A TA .mth is a binary record stream; the fixture's only job is to be
    the instrument's own bytes. The magic string is stored UTF-16LE at offset
    0 — not ASCII — which is why the check encodes it."""
    raw = dta_cycles_mth_path.read_bytes()
    assert len(raw) == EXPECTED_MTH_BYTES
    assert raw.startswith("TA Instruments Method File".encode("utf-16-le"))


# --------------------------------------------------------------------------
# Drift checks against the real file (skipped when it is not on this machine)
# --------------------------------------------------------------------------

@requires_dta_bm
def test_fixture_is_a_faithful_decimation_of_the_real_file(dta_bm_dir, fixture_bytes):
    """Re-runs the committed generator against the real export and demands
    byte equality with the committed fixture. This is the test that keeps the
    fixture honest: it catches both a hand-edited fixture and a generator
    whose behaviour drifted from the bytes it once produced."""
    import tools.make_dta_cycles_example as gen
    rebuilt, header, kept = gen.build_fixture_bytes(dta_bm_dir / "BM-0-cycles.txt", 4)
    assert len(kept) == EXPECTED_DATA_ROWS
    assert len(header) == EXPECTED_HEADER_LINES
    assert rebuilt == fixture_bytes


@requires_dta_bm
def test_committed_mth_fixture_is_byte_identical_to_the_real_one(dta_bm_dir, dta_cycles_mth_path):
    assert dta_cycles_mth_path.read_bytes() == (dta_bm_dir / "cycles.mth").read_bytes()
