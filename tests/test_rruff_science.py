"""Tests for rruff_science.py (M10) — RRUFF mineral Raman database ingest.

Format verified directly against the real, current rruff.net bulk download
(2026-07-15) rather than assumed — see rruff_science.py's module docstring.
Uses EXAMPLES/RRUFF_fair_oriented_sample.zip, a real (not synthetic) small
RRUFF ZIP checked in as a fixture.
"""
from __future__ import annotations

import json

import pytest

import rruff_science as rs


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

def test_parse_rruff_filename_oriented_with_polarization():
    fields = rs.parse_rruff_filename(
        "Annite__R060211-3__Raman__514__0-000__depolarized__Raman_Data_Processed__989cc43b1a7a123bfdaaa6d30516.txt"
    )
    assert fields["mineral"] == "Annite"
    assert fields["rruff_id"] == "R060211-3"
    assert fields["wavelength_nm"] == 514.0
    assert fields["orientation_deg"] == "0-000"
    assert fields["polarization"] == "depolarized"
    assert fields["data_kind"] == "Processed"


def test_parse_rruff_filename_unoriented_has_empty_middle_fields():
    fields = rs.parse_rruff_filename(
        "Abramovite__R070037__Raman__785______Raman_Data_RAW__b8534873d9d9446834252f023c0e.txt"
    )
    assert fields["mineral"] == "Abramovite"
    assert fields["rruff_id"] == "R070037"
    assert fields["wavelength_nm"] == 785.0
    assert fields["orientation_deg"] is None
    assert fields["polarization"] is None
    assert fields["data_kind"] == "RAW"


def test_parse_rruff_filename_non_matching_returns_empty_dict():
    assert rs.parse_rruff_filename("not_a_rruff_file.txt") == {}
    assert rs.parse_rruff_filename("Weird__Format__NotRaman__x.txt") == {}


# --------------------------------------------------------------------------
# Header + data parsing
# --------------------------------------------------------------------------

_REAL_TEXT = """\
##NAMES=Corundum
##RRUFFID=R060020
##IDEAL CHEMISTRY=Al_2_O_3_
##LOCALITY=Yogo Gulch, Montana, USA
##OWNER=RRUFF
##SOURCE=American Museum of Natural History
##ORIENTATION=Laser parallel to c* (0 0 1)
##CELL PARAMETERS=a: 4.76274 b: 4.76274 c: 13.0016 alpha: 90 beta: 90 gamma: 120 volume: 255.412 crystal system: hexagonal
##FILETYPE=Raman Processed
##RAMAN WAVELENGTH=514

179.5050, 136.8613
180.7290, 44.83008
181.9520, 48.55469
"""

_MINIMAL_TEXT = """\
##NAMES=Abramovite
##RRUFFID=R070037
##FILETYPE=Raman RAW
##RAMAN WAVELENGTH=785

137.2198, 1591.066
137.7019, 1623.034
"""


def test_parse_rruff_txt_extracts_full_header_and_data():
    spectrum = rs.parse_rruff_txt(_REAL_TEXT, source_filename="Corundum__R060020__Raman__514__0-000____Raman_Data_Processed__abc.txt")
    assert spectrum.mineral == "Corundum"
    assert spectrum.rruff_id == "R060020"
    assert spectrum.wavelength_nm == 514.0
    assert spectrum.ideal_chemistry == "Al_2_O_3_"
    assert spectrum.locality == "Yogo Gulch, Montana, USA"
    assert spectrum.orientation_text.startswith("Laser parallel")
    assert spectrum.cell_parameters is not None
    assert len(spectrum.x) == 3
    assert spectrum.x[0] == pytest.approx(179.5050)
    assert spectrum.y[0] == pytest.approx(136.8613)


def test_parse_rruff_txt_handles_missing_optional_fields_gracefully():
    """Real unoriented/unconfirmed samples can omit ORIENTATION, MEASURED
    CHEMISTRY, PIN_ID entirely — must not raise or require them."""
    spectrum = rs.parse_rruff_txt(_MINIMAL_TEXT, source_filename="Abramovite__R070037__Raman__785______Raman_Data_RAW__b85.txt")
    assert spectrum.mineral == "Abramovite"
    assert spectrum.orientation_text is None
    assert spectrum.locality is None
    assert spectrum.wavelength_nm == 785.0
    assert len(spectrum.x) == 2


def test_parse_rruff_txt_falls_back_to_filename_when_header_sparse():
    """If NAMES/RRUFFID were ever absent from the header, the filename's
    own fields should still let mineral/rruff_id/wavelength resolve."""
    text = "##FILETYPE=Raman Processed\n\n100.0, 1.0\n200.0, 2.0\n"
    spectrum = rs.parse_rruff_txt(text, source_filename="Quartz__R040031__Raman__532__0-000____Raman_Data_Processed__xyz.txt")
    assert spectrum.mineral == "Quartz"
    assert spectrum.rruff_id == "R040031"
    assert spectrum.wavelength_nm == 532.0


def test_parse_rruff_txt_prefers_header_wavelength_over_mismatched_filename():
    """Regression guard for a real discrepancy found during the full-corpus
    ingest: Gysinite-(Ce) R250121's filename says wavelength "53" (a
    genuine truncated-digit typo in that one filename on rruff.net) while
    its own ##RAMAN WAVELENGTH header correctly says 532. The header must
    win — it's curated structured metadata, the filename is a derived,
    manually-typeable string."""
    text = "##NAMES=Gysinite-(Ce)\n##RRUFFID=R250121\n##RAMAN WAVELENGTH=532\n\n100.0, 1.0\n"
    spectrum = rs.parse_rruff_txt(text, source_filename="Gysinite-(Ce)__R250121__Raman__53______Raman_Data_Processed__abc.txt")
    assert spectrum.wavelength_nm == 532.0


# --------------------------------------------------------------------------
# Real-ZIP ingest (EXAMPLES/RRUFF_fair_oriented_sample.zip, 25 real spectra)
# --------------------------------------------------------------------------

def test_ingest_zip_on_real_sample_produces_sane_records(rruff_sample_zip_path, tmp_path):
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(tmp_path / "raw"), category="fair_oriented")
    assert len(records) == 25

    minerals = {r["mineral"] for r in records}
    assert "Corundum" in minerals
    assert "Annite" in minerals

    for r in records:
        assert r["wavelength_nm"] == 514.0  # this sample ZIP is all 514nm
        assert r["category"] == "fair_oriented"
        assert r["x_min"] is not None and r["x_max"] is not None
        assert r["x_min"] < r["x_max"]

    processed = [r for r in records if r["data_kind"].lower().startswith("process")]
    assert len(processed) > 0
    assert all(isinstance(r["peaks"], list) for r in processed)
    assert any(len(r["peaks"]) > 0 for r in processed)


def test_ingest_zip_writes_raw_text_files_for_later_overlay(rruff_sample_zip_path, tmp_path):
    raw_dir = tmp_path / "raw"
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(raw_dir), category="fair_oriented")
    for r in records:
        assert r["raw_path"]
        with open(r["raw_path"], encoding="utf-8") as f:
            content = f.read()
        assert "##NAMES=" in content


def test_build_index_and_load_index_round_trip(rruff_sample_zip_path, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    count = rs.build_index([(str(rruff_sample_zip_path), "fair_oriented")], cache_dir=str(cache_dir))
    assert count == 25

    loaded = rs.load_index(cache_dir=str(cache_dir))
    assert len(loaded) == 25
    assert (cache_dir / "index.json").is_file()

    with open(cache_dir / "index.json", encoding="utf-8") as f:
        raw_json = json.load(f)
    assert raw_json == loaded


def test_load_index_missing_cache_returns_empty_list(tmp_path):
    assert rs.load_index(cache_dir=str(tmp_path / "does_not_exist")) == []


def test_index_summary_reports_mineral_and_wavelength_coverage(rruff_sample_zip_path, tmp_path):
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(tmp_path / "raw"), category="fair_oriented")
    summary = rs.index_summary(records)
    assert summary["n_spectra"] == 25
    assert summary["n_minerals"] >= 4  # Annite, Corundum, Scholzite, Spinel
    assert summary["wavelengths_nm"] == [514.0]


def test_citation_constants_are_nonempty_strings():
    assert "RRUFF" in rs.RRUFF_CITATION
    assert "Lafuente" in rs.RRUFF_CITATION
    assert len(rs.RRUFF_ATTRIBUTION_NOTE) > 0


# --------------------------------------------------------------------------
# M12 match-assist scoring
# --------------------------------------------------------------------------

def test_score_match_counts_peaks_within_tolerance():
    query = [100.0, 500.0, 1000.0]
    candidate = [102.0, 501.0, 2000.0]  # 2 of 3 within tolerance=10
    result = rs.score_match(query, candidate, tolerance=10.0)
    assert result["matched"] == 2
    assert result["fraction"] == pytest.approx(2 / 3)


def test_score_match_empty_query_returns_zero():
    assert rs.score_match([], [100.0], tolerance=10.0) == {"matched": 0, "fraction": 0.0}


def test_score_match_no_overlap_returns_zero():
    result = rs.score_match([100.0], [500.0, 600.0], tolerance=5.0)
    assert result["matched"] == 0
    assert result["fraction"] == 0.0


def test_rank_rruff_matches_orders_best_first_and_excludes_zero_matches():
    index = [
        {"mineral": "Best", "peaks": [100.0, 200.0, 300.0]},
        {"mineral": "Partial", "peaks": [100.0, 999.0, 999.0]},
        {"mineral": "NoMatch", "peaks": [700.0, 800.0]},
        {"mineral": "NoPeaks", "peaks": []},
    ]
    query = [100.0, 200.0, 300.0]
    ranked = rs.rank_rruff_matches(query, index, tolerance=5.0, top_n=10)

    names = [r["mineral"] for r in ranked]
    assert names[0] == "Best"
    assert "NoMatch" not in names
    assert "NoPeaks" not in names
    assert ranked[0]["matched_peaks"] == 3
    assert ranked[0]["match_fraction"] == pytest.approx(1.0)


def test_rank_rruff_matches_respects_top_n():
    index = [{"mineral": f"M{i}", "peaks": [100.0]} for i in range(50)]
    ranked = rs.rank_rruff_matches([100.0], index, tolerance=5.0, top_n=5)
    assert len(ranked) == 5


def test_rank_rruff_matches_preserves_original_record_fields():
    index = [{"mineral": "Quartz", "rruff_id": "R040031", "wavelength_nm": 532.0, "peaks": [464.0]}]
    ranked = rs.rank_rruff_matches([464.0], index, tolerance=5.0)
    assert ranked[0]["rruff_id"] == "R040031"
    assert ranked[0]["wavelength_nm"] == 532.0
