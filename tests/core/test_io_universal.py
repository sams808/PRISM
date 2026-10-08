"""Tests for io_universal.py — the universal file parser registry.

Fixtures are RAW inputs only (bundled EXAMPLES/* and archived real spectra).
Expected values are derived from physically-sane invariants or from
independently-verifiable facts about the input file, never from snapshotting
the old (pre-fix) pipeline's output.
"""
from __future__ import annotations


import core.io_universal as iou


# --------------------------------------------------------------------------
# Regression pin: the one format this module was clearly built/tested against.
# --------------------------------------------------------------------------

def test_ta_sdt_dta_example_still_works(dta_example_path):
    df, meta = iou.load_any(str(dta_example_path), return_meta=True)
    assert meta["selected_parser"] == "ta_sdt"
    canon = meta["canonical_map"]
    assert "T_C" in canon
    assert "HF_mW" in canon or "DSC_mW_mg" in canon
    # Real instrument data: no fully-empty columns, sane temperature range.
    t_col = df[canon["T_C"]]
    assert t_col.notna().sum() > 100
    assert t_col.min() > -50 and t_col.max() < 2000


# --------------------------------------------------------------------------
# Bug fix: parse_ta_sdt_txt no-StartOfData-marker fallback used to drop the
# real header/column-name line (off-by-one) and corrupt things further when
# data started at line 0 (negative-index slicing).
# --------------------------------------------------------------------------

def test_ta_sdt_txt_fallback_keeps_header_line_and_names_columns(tmp_path):
    text = (
        "Instrument: Fake STA\n"
        "Operator: test\n"
        "Time(min)\tTemp(C)\tDSC(mW)\n"
        "0.0\t25.0\t1.0\n"
        "1.0\t26.0\t1.1\n"
        "2.0\t27.0\t1.2\n"
    )
    p = tmp_path / "no_marker.txt"
    p.write_text(text, encoding="utf-8")

    header, colnames, df = iou.parse_ta_sdt_txt(p)

    # The header/column-name row must be recognized, not silently dropped.
    assert colnames == ["Time(min)", "Temp(C)", "DSC(mW)"]
    # No data rows lost or corrupted into the header dict.
    assert len(df) == 3
    assert list(df["Temp(C)"]) == [25.0, 26.0, 27.0]


def test_ta_sdt_txt_fallback_handles_data_starting_at_line_zero(tmp_path):
    # No header at all — first line is numeric data. The old code computed
    # start_idx = -1 here, which corrupted header parsing via negative
    # indexing (`lines[:-1]`) and could misparse data rows as header entries.
    text = "0.0\t1.0\n1.0\t2.0\n2.0\t3.0\n"
    p = tmp_path / "headerless.txt"
    p.write_text(text, encoding="utf-8")

    header, colnames, df = iou.parse_ta_sdt_txt(p)

    assert header == {}
    assert len(df) == 3
    assert list(df.iloc[:, 0]) == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------
# Bug fix: _read_table_flexible (used by parse_raman/parse_xrd) didn't pass
# header=None for headerless files, silently consuming the first data row.
# --------------------------------------------------------------------------

def test_parse_raman_keeps_all_rows_on_headerless_example(raman_example_path):
    n_lines_in_file = sum(1 for ln in raman_example_path.read_text().splitlines() if ln.strip())
    df, meta = iou.parse_raman(str(raman_example_path))
    assert len(df) == n_lines_in_file, "the first real data row must not be lost as a phantom header"


def test_parse_raman_still_detects_real_header_row(tmp_path):
    text = "Raman Shift (cm-1)\tIntensity (a.u.)\n100.0\t5.0\n200.0\t7.0\n300.0\t9.0\n"
    p = tmp_path / "with_header.txt"
    p.write_text(text, encoding="utf-8")
    df, meta = iou.parse_raman(str(p))
    assert len(df) == 3
    assert "Raman Shift" in str(list(df.columns)[0]) or "shift" in str(list(df.columns)[0]).lower()


def test_parse_xrd_keeps_all_rows_on_headerless_example(xrd_example_path):
    n_lines_in_file = sum(1 for ln in xrd_example_path.read_text(errors="ignore").splitlines() if ln.strip())
    df, meta = iou.parse_xrd(str(xrd_example_path))
    assert len(df) == n_lines_in_file


# --------------------------------------------------------------------------
# Bug fix: sniff_dta_table used to misclassify a Raman file whose comment
# header merely mentions "temperature" alongside an unrelated semicolon.
# --------------------------------------------------------------------------

def test_sniff_dta_table_does_not_misfire_on_incidental_temperature_mention():
    head = "# Raman spectrum; acquired at room temperature, laser 532nm\n100.0\t5.0\n200.0\t7.0\n"
    assert iou.sniff_dta_table("fake.txt", head) is False


def test_sniff_dta_table_still_fires_on_real_semicolon_header_row():
    head = "Time;Temperature;Heat Flow;Mass\n0.0;25.0;1.0;10.0\n"
    assert iou.sniff_dta_table("fake.txt", head) is True


# --------------------------------------------------------------------------
# Bug fix: SAXS column-canonicalization silently overwrote extra columns
# beyond the third (everything non-q/non-sig mapped to the same "I" key).
# --------------------------------------------------------------------------

def test_saxs_extra_columns_do_not_overwrite_each_other(tmp_path):
    text = (
        "# EDF_DataBlockID = 1\n"
        "q(A-1)  I(q)  Sig(q)  I2(q)\n"
        "0.01 100.0 1.0 50.0\n"
        "0.02 90.0 1.1 45.0\n"
        "0.03 80.0 1.2 40.0\n"
        "0.04 70.0 1.3 35.0\n"
        "0.05 60.0 1.4 30.0\n"
    )
    p = tmp_path / "saxs_extra.dat"
    p.write_text(text, encoding="utf-8")
    df, meta = iou.parse_saxs_edf_ascii(str(p))
    canon = meta["canonical_map"]
    # Both intensity-like columns must be retained under distinct keys.
    assert canon.get("I") is not None
    assert canon.get("I2") is not None
    assert canon["I"] != canon["I2"]


def test_saxs_example_still_parses(saxs_example_path):
    df, meta = iou.parse_saxs_edf_ascii(str(saxs_example_path))
    assert len(df) > 10
    assert meta["canonical_map"].get("q_A^-1") is not None


# --------------------------------------------------------------------------
# Bug fix: extra_encodings was unreachable dead code (latin-1 never fails).
# --------------------------------------------------------------------------

def test_extra_encodings_now_reachable_before_latin1_fallback():
    # Bytes 0x93/0x94 decode to curly quotes (U+201C/U+201D) under cp1252,
    # but to C1 control characters under latin-1 — a real, observable
    # difference that proves extra_encodings is actually being tried.
    raw = b"# comment \x93curly\x94"
    text, used_enc = iou._decode_text_autodetect(raw, extra_encodings=("cp1252",))
    assert used_enc == "cp1252"
    assert "curly" in text
    assert "“" in text and "”" in text


def test_decode_text_autodetect_falls_back_to_latin1_when_no_extra_encoding_matches():
    raw = b"\xff\xfeNotActuallyUTF16ButHasBOMBytes"  # exercise the plain fallback path
    text, used_enc = iou._decode_text_autodetect(raw)
    assert isinstance(text, str)


# --------------------------------------------------------------------------
# Dead code removal check
# --------------------------------------------------------------------------

def test_canonicalize_xy_was_removed():
    assert not hasattr(iou, "_canonicalize_xy")


# --------------------------------------------------------------------------
# Repeated TA header keys: TAHeader + meta["header_lines"].
#
# parse_ta_sdt_txt used to fold the header with dict.setdefault, so a
# repeated key kept only its FIRST value. A TA method writes one OrgMethod
# line per program step, so 18 of the 19 program lines of a cyclic run were
# destroyed at parse time (measured on the real BM-0-cycles.txt:
# header["OrgMethod"] == "1: Select gas 1"), and 2 of 3 Xcomment /
# InstCalDate / InstCalFile lines were lost on EVERY TA export.
#
# Imports live here rather than at the top of the file so this section could
# be appended without restructuring what is above it.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest

import core.project_io as project_io


_EXAMPLES_DIR = Path(iou.__file__).resolve().parents[1] / "EXAMPLES"
_CYCLES_FIXTURE = _EXAMPLES_DIR / "DTA_cycles_example.txt"


def _bm_cycles_path() -> Path:
    """The full-resolution cyclic run (2.5 MB, not in the repo; the committed
    decimated fixture is). PRISM_DTA_BM_DIR relocates the folder on another
    machine, and the tests that need it skip when it is absent."""
    override = os.environ.get("PRISM_DTA_BM_DIR")
    base = Path(override) if override else Path("C:/Users/samso/Desktop/WSU_work/DTA/BM")
    return base / "BM-0-cycles.txt"


_BM_CYCLES = _bm_cycles_path()

requires_bm_cycles = pytest.mark.skipif(
    not _BM_CYCLES.is_file(),
    reason="full-resolution BM-0-cycles.txt is not on this machine (set PRISM_DTA_BM_DIR)",
)
requires_cycles_fixture = pytest.mark.skipif(
    not _CYCLES_FIXTURE.is_file(),
    reason="EXAMPLES/DTA_cycles_example.txt has not been built yet",
)


def _raw_header_pairs(path: Path) -> list:
    """Re-derive every (key, value) header line, in file order, straight from
    the bytes. The expected values below are then facts about the FILE rather
    than a snapshot of the output of the parser under test."""
    pairs = []
    for line in iou._read_text_autodetect(Path(path)).splitlines():
        if "startofdata" in line.replace(" ", "").replace("\t", "").lower():
            break
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) >= 2:
            pairs.append((fields[0].strip(), "\t".join(fields[1:]).strip()))
    return pairs


def _count_raw_header_key(path: Path, key: str) -> int:
    return sum(1 for k, _ in _raw_header_pairs(path) if k == key)


def _write_synthetic_ta_export(tmp_path: Path, name: str) -> Path:
    """A minimal TA-style export with deliberately repeated header keys."""
    lines = [
        "Sample\tSynthetic cyclic run",
        "Operator\ttest",
        "OrgMethod\t1: Ramp 10.00 C/min to 500.00 C",
        "OrgMethod\t2: Isothermal for 5.00 min",
        "OrgMethod\t3: Ramp 10.00 C/min to 25.00 C",
        "Xcomment\tPan: Alumina",
        "Xcomment\tGas1: Nitrogen 100.0 ml/min",
        "Sig1\tTime (min)",
        "Sig2\tTemperature (C)",
        "Sig3\tHeat Flow (mW)",
        "StartOfData",
        "0.0\t25.0\t0.10",
        "1.0\t35.0\t0.20",
        "2.0\t45.0\t0.30",
        "3.0\t55.0\t0.25",
        "4.0\t65.0\t0.15",
    ]
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---- the container itself -------------------------------------------------

def test_ta_header_equals_a_plain_dict_and_keeps_first_wins():
    # A zero-item dict subclass, so the headerless-file contract survives.
    assert iou.TAHeader() == {}
    assert iou.TAHeader([("a", "1")])["a"] == "1"

    h = iou.TAHeader([("k", "1"), ("j", "x"), ("k", "2")])
    assert h["k"] == "1" and h.get("k") == "1"          # FIRST wins, unchanged
    assert h == {"k": "1", "j": "x"}                    # equality unchanged
    assert len(h) == 2 and sorted(h) == ["j", "k"]
    assert h.all("k") == ["1", "2"]                     # both, in file order
    assert h.all("j") == ["x"]
    assert h.all("absent") == []
    assert list(h.lines) == [("k", "1"), ("j", "x"), ("k", "2")]


def test_ta_header_all_falls_back_to_the_stored_value_without_lines():
    # A header rebuilt from a plain mapping (e.g. a .prism round-trip, where
    # _json_safe has flattened the subclass) must still answer .all().
    h = iou.TAHeader({"Sample": "S1"})
    assert h.all("Sample") == ["S1"]
    assert h.all("OrgMethod") == []


def test_ta_header_copy_from_another_ta_header_keeps_the_repeats():
    src = iou.TAHeader([("OrgMethod", "1: a"), ("OrgMethod", "2: b")])
    assert iou.TAHeader(src).all("OrgMethod") == ["1: a", "2: b"]


def test_ta_header_skips_malformed_pairs_instead_of_raising():
    # Header construction sits on the file-parsing path: a malformed export
    # must degrade to a thinner header, never to an import failure.
    h = iou.TAHeader([("a", "1"), None, ("b", "2"), ("c", "3", "4"), 7])
    assert h == {"a": "1", "b": "2"}


def test_headerless_file_still_gives_an_empty_header(tmp_path):
    text = "0.0\t1.0\n1.0\t2.0\n2.0\t3.0\n"
    p = tmp_path / "headerless_taheader.txt"
    p.write_text(text, encoding="utf-8")

    header, colnames, df = iou.parse_ta_sdt_txt(p)

    assert header == {}
    assert isinstance(header, iou.TAHeader)
    assert tuple(header.lines) == ()
    assert header.all("OrgMethod") == []
    assert len(df) == 3


# ---- committed EXAMPLES/DTA_example.txt: repeats exist on every TA export --

def test_repeated_xcomment_lines_survive_on_the_bundled_example(dta_example_path):
    header, colnames, df = iou.parse_ta_sdt_txt(dta_example_path)

    xcomments = header.all("Xcomment")
    assert len(xcomments) == 3 == _count_raw_header_key(dta_example_path, "Xcomment")
    # File order, not set order: the pan line first, then the two gas lines.
    assert xcomments[0] == "Pan: Alumina"
    assert xcomments[1].startswith("Gas1:")
    assert xcomments[2].startswith("Gas2:")
    # The same fold destroyed these two on every TA export, cyclic or not.
    assert len(header.all("InstCalDate")) == 3
    assert len(header.all("InstCalFile")) == 3
    assert header.all("InstCalDate")[0].startswith("Weight:")
    assert header.all("InstCalFile")[2].startswith("DTA Baseline:")


def test_first_wins_single_valued_keys_are_unchanged(dta_example_path):
    header, colnames, df = iou.parse_ta_sdt_txt(dta_example_path)
    raw = _raw_header_pairs(dta_example_path)

    assert header["Sample"] == "NaBiO3-400C-N2"
    for key in {k for k, _ in raw}:
        first_in_file = next(v for k, v in raw if k == key)
        assert header[key] == first_in_file
        assert header.get(key) == first_in_file
    assert len(header) == len({k for k, _ in raw})


# ---- the cyclic program: 19 OrgMethod lines -------------------------------

def _assert_nineteen_org_method_lines(path: Path) -> None:
    """Shared body: the 19-step cyclic method must come back whole, in file
    order, with first-wins lookup untouched."""
    header, colnames, df = iou.parse_ta_sdt_txt(Path(path))

    steps = header.all("OrgMethod")
    assert len(steps) == 19 == _count_raw_header_key(path, "OrgMethod")
    # TA numbers each line "<n>: ...", so file order is externally checkable.
    assert [s.split(":", 1)[0] for s in steps] == [str(i) for i in range(1, 20)]
    assert steps[0] == "1: Select gas 1"
    assert steps[-1].startswith("19: Mass flow")
    assert "Ramp" in steps[2] and "1300" in steps[2]
    # First-wins lookup is deliberately NOT changed by the fix.
    assert header["OrgMethod"] == "1: Select gas 1"
    # The other repeated keys of this export.
    assert len(header.all("Xcomment")) == 3
    assert len(header.all("InstCalDate")) == 3
    assert len(header.all("InstCalFile")) == 3
    assert header["Sample"] == "BM-0-cycles"
    assert len(df) > 100


@requires_bm_cycles
def test_org_method_lines_survive_the_header_parser_on_the_real_run():
    _assert_nineteen_org_method_lines(_BM_CYCLES)


@requires_cycles_fixture
def test_org_method_lines_survive_the_header_parser_on_the_committed_fixture():
    # The decimated fixture keeps the header verbatim, so the temperature
    # program must be recoverable from it on a machine with no access to the
    # full-resolution run.
    _assert_nineteen_org_method_lines(_CYCLES_FIXTURE)


@requires_bm_cycles
def test_xcomment_all_returns_three_entries_in_file_order_on_the_real_run():
    header, colnames, df = iou.parse_ta_sdt_txt(_BM_CYCLES)
    assert header.all("Xcomment") == [
        "Pan: Alumina",
        "Gas1: Nitrogen 0.0 ml/min",
        "Gas2: Nitrogen 0.0 ml/min",
    ]


# ---- meta["header_lines"] in both registered parsers ----------------------

def test_meta_header_lines_is_present_and_ordered_for_ta_sdt(dta_example_path):
    df, meta = iou.parse_ta_sdt(str(dta_example_path))
    expected = [[k, v] for k, v in _raw_header_pairs(dta_example_path)]

    assert meta["header_lines"] == expected
    assert all(isinstance(item, list) and len(item) == 2 for item in meta["header_lines"])
    assert [k for k, _ in meta["header_lines"]].count("Xcomment") == 3
    # raw_header is unchanged and still first-wins.
    assert meta["raw_header"]["Sample"] == "NaBiO3-400C-N2"


def test_meta_header_lines_is_present_and_ordered_for_both_parsers(tmp_path):
    p = _write_synthetic_ta_export(tmp_path, "synthetic_ta.txt")
    expected = [[k, v] for k, v in _raw_header_pairs(p)]
    assert [k for k, _ in expected].count("OrgMethod") == 3

    for parse in (iou.parse_ta_sdt, iou.parse_dta_table):
        df, meta = parse(str(p))
        assert meta["header_lines"] == expected, parse.__name__
        org = [v for k, v in meta["header_lines"] if k == "OrgMethod"]
        assert org == [
            "1: Ramp 10.00 C/min to 500.00 C",
            "2: Isothermal for 5.00 min",
            "3: Ramp 10.00 C/min to 25.00 C",
        ], parse.__name__
        assert meta["raw_header"]["OrgMethod"] == org[0], parse.__name__
        assert len(df) == 5, parse.__name__


def test_meta_header_lines_is_what_survives_a_project_round_trip(tmp_path):
    # WHY meta["header_lines"] exists at all: _json_safe flattens the TAHeader
    # subclass into a plain dict on .prism save, so .lines does not survive the
    # round trip - the plain list of [key, value] pairs does.
    p = _write_synthetic_ta_export(tmp_path, "synthetic_roundtrip.txt")
    df, meta = iou.parse_ta_sdt(str(p))

    saved = project_io._json_safe(meta)
    assert not hasattr(saved["raw_header"], "lines")
    assert saved["raw_header"] == dict(meta["raw_header"])
    assert [k for k, _ in saved["header_lines"]].count("OrgMethod") == 3
    assert saved["header_lines"] == meta["header_lines"]


@requires_bm_cycles
def test_meta_header_lines_carries_the_whole_program_on_the_real_run():
    df, meta = iou.parse_ta_sdt(str(_BM_CYCLES))
    org = [v for k, v in meta["header_lines"] if k == "OrgMethod"]
    assert len(org) == 19
    assert org == meta["raw_header"].all("OrgMethod")
