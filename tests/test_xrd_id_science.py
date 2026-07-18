"""Tests for xrd_id_science.py — the QualX-style XRD phase-identification
engine. Uses a tiny synthetic QualX-format source database (same schema as
the user's real cod/pdf2 .sq files), never the real multi-GB ones."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

import xrd_id_science as xid


def _make_source_sq(path, cards):
    """cards: [(id, name, mineral, formula, sg, quality, d_list, i_list, elements)]"""
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE id (id int, name varchar, mineralname varchar, chemical_formula varchar,
                         spacegroup varchar, quality varchar, rir double, nrec int,
                         dvalue blob, intensita blob, nd int);
        CREATE TABLE chemical (id int, chemical_element varchar);
        CREATE TABLE infodb (id varchar, date varchar, ncard int, type varchar, source varchar);
    """)
    con.execute("INSERT INTO infodb VALUES ('t', '2026-01-01', ?, 'TEST', 'unit-test')", (len(cards),))
    for cid, name, mineral, formula, sg, quality, d, i, elements in cards:
        con.execute("INSERT INTO id VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, name, mineral, formula, sg, quality, 1.0, 0,
                     ",".join(f"{v:.6f}" for v in d), ",".join(f"{v:.4f}" for v in i), len(d)))
        for el in elements:
            con.execute("INSERT INTO chemical VALUES (?,?)", (cid, el))
    con.commit()
    con.close()


QUARTZ_D = [3.342, 4.257, 1.8179, 2.457, 2.282, 1.5418, 1.3718]
QUARTZ_I = [100.0, 22.0, 14.0, 8.0, 8.0, 9.0, 8.0]
CALCITE_D = [3.035, 2.285, 2.095, 1.913, 1.875, 3.86]
CALCITE_I = [100.0, 18.0, 18.0, 17.0, 17.0, 12.0]


@pytest.fixture()
def unified_db(tmp_path):
    src1 = tmp_path / "src1.sq"
    _make_source_sq(src1, [
        (1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
        (1020, "Calcite", "Calcite", "C Ca O3", "R -3 c", "A", CALCITE_D, CALCITE_I, ["C", "Ca", "O"]),
        (1030, "no-lines junk", "", "", "", "C", [3.1], [100.0], []),  # < min_lines -> skipped
    ])
    src2 = tmp_path / "src2.sq"
    _make_source_sq(src2, [
        (77, " $GB-Quartz synthetic ", "", "Si O2", "P 32 2 1", "B", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
    ])
    out = tmp_path / "unified.sq"
    counts = xid.build_xrd_database([(str(src1), "SRC1"), (str(src2), "SRC2")],
                                    out_path=str(out), log=lambda *a: None)
    assert counts == {"SRC1": 2, "SRC2": 1}
    return str(out)


def test_two_theta_d_round_trip():
    tt = np.array([20.0, 26.64, 60.0])
    d = xid.two_theta_to_d(tt)
    back = xid.d_to_two_theta(d)
    assert np.allclose(back, tt)
    # quartz's main line: d=3.342 Å -> 2θ ≈ 26.65° at Cu Kα
    assert xid.d_to_two_theta(np.array([3.342]))[0] == pytest.approx(26.65, abs=0.05)


def test_build_keeps_sources_codes_and_cleans_pdf2_names(unified_db):
    con = sqlite3.connect(unified_db)
    rows = con.execute("SELECT source, source_code, name FROM cards ORDER BY card_id").fetchall()
    con.close()
    assert ("SRC1", "1010", "Quartz low") in rows
    assert ("SRC2", "77", "Quartz synthetic") in rows  # $GB- code stripped, original id kept
    summary = xid.database_summary(unified_db)
    assert summary["total_cards"] == 3
    assert summary["by_source"] == {"SRC1": 2, "SRC2": 1}


def test_search_match_identifies_quartz_from_measured_peaks(unified_db):
    # simulated measured pattern: quartz's lines at Cu Kα with small offsets
    tt = xid.d_to_two_theta(np.array(QUARTZ_D)) + 0.03
    results = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, db_path=unified_db)
    assert results
    assert results[0].mineral == "Quartz" or "Quartz" in results[0].name
    assert results[0].fom > 0.9
    assert results[0].n_matched >= 6
    # Calcite scores far lower (only coincidental overlaps)
    calcite = [r for r in results if r.mineral == "Calcite"]
    assert not calcite or calcite[0].fom < 0.5
    # both quartz cards (both sources) are found
    quartz_sources = {r.source for r in results if "Quartz" in (r.mineral + r.name)}
    assert quartz_sources == {"SRC1", "SRC2"}


def test_search_match_element_filters(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    with_ca = xid.search_match(tt, QUARTZ_I, elements_all=["Ca"], db_path=unified_db)
    assert all(r.mineral == "Calcite" for r in with_ca)
    no_si = xid.search_match(tt, QUARTZ_I, elements_none=["Si"], db_path=unified_db)
    assert all("Quartz" not in (r.mineral + r.name) for r in no_si)


def test_search_match_source_filter(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    only2 = xid.search_match(tt, QUARTZ_I, sources=["SRC2"], db_path=unified_db)
    assert only2 and all(r.source == "SRC2" for r in only2)


def test_search_match_range_limits_card_coverage(unified_db):
    # only the strongest quartz line measured, in a narrow range: coverage of
    # the card must be computed against lines INSIDE that range only
    tt_main = float(xid.d_to_two_theta(np.array([3.342]))[0])
    results = xid.search_match([tt_main], [100.0], two_theta_range=(25.0, 28.0),
                               db_path=unified_db)
    quartz = next(r for r in results if "Quartz" in (r.mineral + r.name))
    assert quartz.cov_card == pytest.approx(1.0, abs=0.01)  # the only in-range line is matched


def test_search_match_missing_db_raises_helpfully(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_xrd_database"):
        xid.search_match([26.6], db_path=str(tmp_path / "absent.sq"))


def test_find_cards_by_text(unified_db):
    hits = xid.find_cards_by_text("quartz", db_path=unified_db)
    assert len(hits) == 2
    assert all(len(h["d"]) == len(QUARTZ_D) for h in hits)
    hits = xid.find_cards_by_text("Ca O3", db_path=unified_db)
    assert len(hits) == 1 and hits[0]["mineral"] == "Calcite"
