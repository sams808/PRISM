"""Tests for xrd_id_science.py — the QualX-style XRD phase-identification
engine. Uses tiny synthetic QualX-format source databases (same schema as
real downloadable card databases), never real multi-GB ones."""
from __future__ import annotations

import os
import pathlib
import sqlite3

import numpy as np
import pytest

import xrd.xrd_id_science as xid


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


def test_build_keeps_sources_codes_and_cleans_typeset_names(unified_db):
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


# ---------------------------------------------------------------------------
# Database registry + multi-database probing
# ---------------------------------------------------------------------------

def test_sniff_sq_format(tmp_path, unified_db):
    src = tmp_path / "qualx_src.sq"
    _make_source_sq(src, [(1, "X", "", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    assert xid.sniff_sq_format(str(src)) == "qualx"
    assert xid.sniff_sq_format(unified_db) == "prism"
    junk = tmp_path / "not_a_db.sq"
    junk.write_text("hello, not sqlite")
    assert xid.sniff_sq_format(str(junk)) is None
    assert xid.sniff_sq_format(str(tmp_path / "absent.sq")) is None


def test_register_qualx_database_converts_once(tmp_path):
    reg = str(tmp_path / "reg.json")
    imp = str(tmp_path / "imported")
    src = tmp_path / "MyCOD.sq"
    _make_source_sq(src, [(1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A",
                           QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    entry = xid.register_database(str(src), registry_path=reg, import_dir=imp, log=lambda *a: None)
    assert entry["name"] == "MyCOD"
    assert entry["origin"] == str(src)
    assert xid.sniff_sq_format(entry["path"]) == "prism"
    # provenance tag inside the converted file = the registry name
    assert xid.database_summary(entry["path"])["by_source"] == {"MyCOD": 1}
    # re-registering the ORIGINAL path is a no-op, not a duplicate
    xid.register_database(str(src), registry_path=reg, import_dir=imp, log=lambda *a: None)
    assert len(xid.load_registry(reg)) == 1


def test_register_prism_database_in_place_no_copy(tmp_path, unified_db):
    import os
    reg = str(tmp_path / "reg2.json")
    entry = xid.register_database(unified_db, registry_path=reg, log=lambda *a: None)
    assert entry["path"] == os.path.abspath(unified_db)
    assert "origin" not in entry


def test_register_rejects_non_database(tmp_path):
    bad = tmp_path / "junk.sq"
    bad.write_text("definitely not sqlite")
    with pytest.raises(ValueError, match="not a recognizable"):
        xid.register_database(str(bad), registry_path=str(tmp_path / "r.json"))


def test_enable_disable_and_unregister(tmp_path, unified_db):
    import os
    reg = str(tmp_path / "reg3.json")
    xid.register_database(unified_db, name="DB A", registry_path=reg, log=lambda *a: None)
    assert xid.enabled_database_paths(reg) == [os.path.abspath(unified_db)]
    xid.set_database_enabled("DB A", False, registry_path=reg)
    assert xid.enabled_database_paths(reg) == []
    xid.unregister_database("DB A", registry_path=reg)
    assert xid.load_registry(reg) == []


def test_load_registry_migrates_legacy_unified_db(tmp_path, monkeypatch, unified_db):
    """A pre-registry setup (only the old fixed-location unified .sq) must
    keep working: first load_registry() call registers it automatically."""
    reg = tmp_path / "migrated_registry.json"
    monkeypatch.setattr(xid, "XRD_ID_REGISTRY_PATH", str(reg))
    monkeypatch.setattr(xid, "XRD_ID_DB_PATH", unified_db)
    entries = xid.load_registry()
    assert len(entries) == 1
    assert entries[0]["path"] == unified_db
    assert entries[0]["enabled"] is True
    assert reg.is_file()  # persisted, so the migration happens exactly once


def test_search_match_across_multiple_databases(tmp_path):
    """Quartz lives only in DB1, calcite only in DB2 — one search over both
    must find both, each hit tagged with the database it came from."""
    src1, src2 = tmp_path / "s1.sq", tmp_path / "s2.sq"
    _make_source_sq(src1, [(1, "Quartz", "Quartz", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    _make_source_sq(src2, [(2, "Calcite", "Calcite", "C Ca O3", "", "A", CALCITE_D, CALCITE_I, ["C", "Ca", "O"])])
    db1, db2 = tmp_path / "db_quartz.sq", tmp_path / "db_calcite.sq"
    xid.build_xrd_database([(str(src1), "DB1")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src2), "DB2")], out_path=str(db2), log=lambda *a: None)

    tt = np.concatenate([xid.d_to_two_theta(np.array(QUARTZ_D)),
                         xid.d_to_two_theta(np.array(CALCITE_D))])
    ii = np.concatenate([QUARTZ_I, CALCITE_I])
    results = xid.search_match(tt, ii, db_paths=[str(db1), str(db2)])
    minerals = {r.mineral for r in results}
    assert {"Quartz", "Calcite"} <= minerals
    assert {r.db for r in results} == {"db_quartz", "db_calcite"}


def test_find_cards_by_text_across_multiple_databases(tmp_path):
    src1, src2 = tmp_path / "t1.sq", tmp_path / "t2.sq"
    _make_source_sq(src1, [(1, "Quartz", "Quartz", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    _make_source_sq(src2, [(2, "Quartz high", "Quartz", "Si O2", "", "B", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    db1, db2 = tmp_path / "ta.sq", tmp_path / "tb.sq"
    xid.build_xrd_database([(str(src1), "A")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src2), "B")], out_path=str(db2), log=lambda *a: None)
    hits = xid.find_cards_by_text("quartz", db_paths=[str(db1), str(db2)])
    assert len(hits) == 2
    assert {h["db"] for h in hits} == {"ta", "tb"}


def test_crystal_system_inference():
    cases = {
        "Fm-3m": "cubic", "Im-3m": "cubic", "P 41 32": "cubic", "Ia-3d": "cubic",
        "P 63/m m c": "hexagonal", "P 6222": "hexagonal",
        "R -3 c": "trigonal", "P 32 2 1": "trigonal", "P -3 m 1": "trigonal",
        "I 41/a m d": "tetragonal", "P 42/m n m": "tetragonal",
        "P n m a": "orthorhombic", "C m c m": "orthorhombic", "P 21 21 21": "orthorhombic",
        "P 21/c": "monoclinic", "C 2/m": "monoclinic", "P 2": "monoclinic",
        "P -1": "triclinic", "P 1": "triclinic",
        "": "",
    }
    for sg, expected in cases.items():
        assert xid.crystal_system(sg) == expected, f"{sg!r} -> {xid.crystal_system(sg)!r}, expected {expected!r}"


def test_search_match_quality_system_and_spacegroup_filters(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    # quartz exists as quality A (SRC1) and quality B (SRC2), both P 32 2 1
    only_b = xid.search_match(tt, QUARTZ_I, qualities=["B"], db_path=unified_db)
    assert only_b and all(r.quality == "B" for r in only_b)
    # nothing in the fixture DB is cubic
    assert xid.search_match(tt, QUARTZ_I, crystal_systems=["cubic"], db_path=unified_db) == []
    sg_hits = xid.search_match(tt, QUARTZ_I, spacegroup_contains="32 2 1", db_path=unified_db)
    assert sg_hits and all("32 2 1" in r.spacegroup for r in sg_hits)


def test_search_match_dedups_identical_cards_across_databases(tmp_path):
    """The same card (code + phase + space group + quality) carried by two
    registered databases must appear once, not twice (user request)."""
    src = tmp_path / "same.sq"
    _make_source_sq(src, [(1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A",
                           QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    db1, db2 = tmp_path / "d1.sq", tmp_path / "d2.sq"
    xid.build_xrd_database([(str(src), "TAG1")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src), "TAG2")], out_path=str(db2), log=lambda *a: None)
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    res = xid.search_match(tt, QUARTZ_I, db_paths=[str(db1), str(db2)])
    assert len([r for r in res if r.mineral == "Quartz"]) == 1


def test_database_summary_lists_qualities(unified_db):
    # kept cards: two quality-A (SRC1) + one quality-B (SRC2); the junk
    # quality-C card is skipped at build time (< min_lines)
    assert xid.database_summary(unified_db)["qualities"] == ["A", "B"]


def test_find_cards_by_elements_matches_regardless_of_formula_order(tmp_path):
    """User report: text search for 'TiO2' missed cards written 'O2 Ti'."""
    src = tmp_path / "src_ti.sq"
    _make_source_sq(src, [
        (1, "Anatase", "Anatase", "O2 Ti", "I 41/a m d", "A", [3.52, 2.38, 1.89], [100.0, 20.0, 30.0], ["Ti", "O"]),
        (2, "Rutile", "Rutile", "Ti O2", "P 42/m n m", "A", [3.25, 2.49, 1.69], [100.0, 50.0, 60.0], ["Ti", "O"]),
        (3, "LiTi oxide", "", "Li0.5 O2 Ti", "", "B", [4.0, 2.0, 1.5], [100.0, 40.0, 20.0], ["Li", "Ti", "O"]),
    ])
    db = tmp_path / "uni_ti.sq"
    xid.build_xrd_database([(str(src), "T")], out_path=str(db), log=lambda *a: None)

    exact = xid.find_cards_by_elements("TiO2", mode="exact", db_path=str(db))
    assert sorted(h["mineral"] or h["formula"] for h in exact) == ["Anatase", "Rutile"]
    contains = xid.find_cards_by_elements("TiO2", mode="contains", db_path=str(db))
    assert len(contains) == 3  # the Li-bearing card included
    assert xid.find_cards_by_elements("Zr2O", mode="exact", db_path=str(db)) == []


# ---------------------------------------------------------------------------
# Peak detection on a measured diffractogram
# ---------------------------------------------------------------------------

def _pattern(peaks, *, x=None, background=None, noise=0.0, width=0.06, seed=0):
    """Synthetic diffractogram: [(2theta, height), ...] as Gaussians on an
    optional background, on a realistic 0.01 deg grid."""
    x = np.arange(10.0, 80.0, 0.01) if x is None else x
    y = np.zeros_like(x)
    for c, h in peaks:
        y += h * np.exp(-0.5 * ((x - c) / width) ** 2)
    if background is not None:
        y = y + background(x)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=len(x))
    return x, y


def test_snip_background_follows_a_halo_without_eating_bragg_peaks():
    """The case the Raman finder could not handle: sharp reflections
    riding on a big amorphous hump."""
    halo = lambda x: 500.0 + 3000.0 * np.exp(-0.5 * ((x - 27.0) / 6.0) ** 2)
    x, y = _pattern([(30.0, 2000.0), (55.0, 800.0)], background=halo)
    bg = xid.snip_background(y, int(round(0.3 / 0.01)))
    # the hump is background: recovered within a few percent away from peaks
    clear = (np.abs(x - 30.0) > 1.0) & (np.abs(x - 55.0) > 1.0)
    assert np.allclose(bg[clear], halo(x)[clear], rtol=0.05, atol=60.0)
    # the peaks are NOT: their height survives the subtraction
    corr = y - bg
    assert corr[np.argmin(np.abs(x - 30.0))] > 0.9 * 2000.0
    assert corr[np.argmin(np.abs(x - 55.0))] > 0.9 * 800.0


def test_snip_window_width_is_an_aggressiveness_dial_not_a_cutoff():
    """Measured, because the intuition is wrong and the wrong default cost
    a factor of five on every intensity sitting on the halo: SNIP iterates
    once per channel of the window, so a window several times the halo's
    width does not 'keep' the halo — it erodes it away."""
    halo = lambda x: 300.0 + 4000.0 * np.exp(-0.5 * ((x - 22.0) / 4.0) ** 2)  # FWHM 9.4 deg
    x, y = _pattern([(22.0, 1000.0)], background=halo)
    off = int(np.argmin(np.abs(x - 24.0)))  # clear of the sharp peak
    truth = halo(x)[off] - 300.0
    narrow = xid.snip_background(y, int(round(0.3 / 0.01)))[off] - 300.0
    wide = xid.snip_background(y, int(round(2.0 / 0.01)))[off] - 300.0
    assert narrow > 0.90 * truth   # the hump is kept as background
    assert wide < 0.10 * truth     # ... and destroyed by the wide window


def test_background_width_adapts_to_the_pattern_peak_widths():
    """One fixed window cannot serve a sharp lab powder and a
    nanocrystalline sample: too wide swallows the halo into the peak
    intensities, too narrow clips broad reflections away as background."""
    halo = lambda x: 300.0 + 4000.0 * np.exp(-0.5 * ((x - 22.0) / 4.0) ** 2)
    widths = {}
    for sigma in (0.06, 0.20, 0.60):
        x, y = _pattern([(22.0, 1000.0), (44.0, 1000.0), (60.0, 1000.0)],
                        background=halo, width=sigma, noise=8.0)
        widths[sigma] = xid.estimate_background_width(x, y)
    assert widths[0.06] < widths[0.20] < widths[0.60]
    lo, hi = xid.BACKGROUND_WIDTH_LIMITS
    assert all(lo <= w <= hi for w in widths.values())
    # sharp peaks on a halo: the height that comes back is the PEAK's, not
    # the peak plus the halo underneath it
    x, y = _pattern([(22.0, 1000.0), (60.0, 1000.0)], background=halo, width=0.06)
    got = xid.peak_intensities_at(x, y, [22.0, 60.0])
    assert got[0] == pytest.approx(1000.0, rel=0.15)


def test_find_xrd_peaks_recovers_positions_and_relative_intensities():
    halo = lambda x: 400.0 + 2500.0 * np.exp(-0.5 * ((x - 25.0) / 5.0) ** 2)
    planted = [(21.5, 1200.0), (28.34, 3000.0), (44.0, 900.0), (62.17, 600.0)]
    x, y = _pattern(planted, background=halo, noise=12.0)
    found = xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0)
    assert [p.two_theta for p in found] == sorted(p.two_theta for p in found)
    for c, h in planted:
        near = [p for p in found if abs(p.two_theta - c) < 0.05]
        assert near, f"missed the planted reflection at {c}"
        assert near[0].intensity == pytest.approx(h, rel=0.15)
    # nothing invented on the halo itself
    assert not [p for p in found if abs(p.two_theta - 25.0) < 1.0]


def test_find_xrd_peaks_reaches_weak_high_angle_lines_under_a_strong_halo():
    """A glass-ceramic pattern whose amorphous hump is larger than every
    reflection, and whose informative lines are the weak high-angle ones.

    On the real pattern that prompted this (a 10-80 deg scan of a
    partly crystallized glass) the Raman 2nd-derivative finder returned
    25 candidates all between 11.9 and 36.9 deg, several of them on the
    rising flank of a peak rather than on a maximum; this finder returned
    33 spanning 16-79 deg. The synthetic here pins the contract that
    matters - weak high-angle lines found, nothing invented on the hump."""
    halo = lambda x: 400.0 + 6000.0 * np.exp(-0.5 * ((x - 25.0) / 5.0) ** 2)
    planted = [(21.5, 900.0), (28.34, 2500.0), (58.0, 260.0), (71.4, 200.0)]
    x, y = _pattern(planted, background=halo, noise=12.0)

    new = [p.two_theta for p in xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0)]
    for c, _h in planted:
        assert any(abs(v - c) < 0.05 for v in new), f"new finder missed {c}"
    # and nothing invented on the hump's flanks, where a curvature-based
    # finder puts most of its candidates
    assert not [v for v in new if abs(v - 25.0) < 2.0 and abs(v - 28.34) > 0.1]
    assert sum(1 for v in new if v > 45.0) >= 2


def test_find_xrd_peaks_flags_a_resolved_ka2_satellite():
    ratio = xid.ka2_ratio_for(xid.CU_KA1)
    ka1 = 75.0
    ka2 = float(xid.ka2_position([ka1], ratio)[0])
    assert ka2 - ka1 > 0.2  # resolved at this angle, by construction
    x, y = _pattern([(ka1, 1000.0), (ka2, 500.0), (40.0, 800.0)],
                    background=lambda v: np.full_like(v, 200.0), width=0.04)
    found = xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0)
    satellites = [p for p in found if p.is_ka2]
    assert len(satellites) == 1
    assert satellites[0].two_theta == pytest.approx(ka2, abs=0.05)
    assert satellites[0].parent_two_theta == pytest.approx(ka1, abs=0.05)
    # and the parent is NOT flagged, whatever the iteration order
    assert not [p for p in found if p.is_ka2 and abs(p.two_theta - ka1) < 0.05]
    # switching the flagging off leaves both as independent reflections
    assert not any(p.is_ka2 for p in xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0,
                                                        flag_ka2=False))


def test_ka2_ratio_follows_the_anode():
    assert xid.ka2_ratio_for(1.5406) == pytest.approx(1.544426 / 1.540598, rel=1e-9)
    assert xid.ka2_ratio_for(1.7890) == pytest.approx(1.792850 / 1.788965, rel=1e-9)
    assert xid.ka2_ratio_for(0.7093) == pytest.approx(0.713590 / 0.709300, rel=1e-9)
    # a synchrotron wavelength matches no anode: fall back to Cu rather than
    # silently using Ag's ratio because it happens to be the nearest
    assert xid.ka2_ratio_for(0.9) == pytest.approx(xid.ka2_ratio_for(1.5406))


def test_peak_intensities_are_background_subtracted():
    """Reading raw heights makes a peak on the halo look stronger than an
    equal peak on the flat part — and re-weights every figure of merit."""
    halo = lambda x: 300.0 + 4000.0 * np.exp(-0.5 * ((x - 22.0) / 4.0) ** 2)
    x, y = _pattern([(22.0, 1000.0), (60.0, 1000.0)], background=halo)
    corrected = xid.peak_intensities_at(x, y, [22.0, 60.0])
    assert corrected[0] == pytest.approx(corrected[1], rel=0.15)
    raw = xid.peak_intensities_at(x, y, [22.0, 60.0], subtract_background=False)
    assert raw[0] > 3.0 * raw[1]  # what the old rule saw


# ---------------------------------------------------------------------------
# Search-match: proximity weighting, zero-shift, filter ordering
# ---------------------------------------------------------------------------

def test_fom_rewards_lines_that_land_on_the_peaks(unified_db):
    """A card matched at the edge of the tolerance window must not score
    like one matched dead on — otherwise a tolerance wide enough for real
    sample displacement makes every candidate look equally good."""
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    exact = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, db_path=unified_db)
    sloppy = xid.search_match(tt + 0.18, QUARTZ_I, tol_two_theta=0.2, db_path=unified_db)
    assert exact[0].fom > sloppy[0].fom + 0.3
    assert exact[0].mean_abs_dev < 1e-6
    assert sloppy[0].mean_abs_dev == pytest.approx(0.18, abs=0.01)


def test_shift_tolerance_recovers_a_displaced_pattern(unified_db):
    """Sample displacement shifts every line the same way; allowing a
    bounded global offset must recover the score and REPORT the offset."""
    tt = xid.d_to_two_theta(np.array(QUARTZ_D)) + 0.12
    without = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, db_path=unified_db)
    with_shift = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, shift_tolerance=0.15,
                                  db_path=unified_db)
    assert with_shift[0].fom > without[0].fom + 0.2
    assert with_shift[0].shift == pytest.approx(0.12, abs=0.02)
    assert with_shift[0].mean_abs_dev < 0.02
    # the offset is bounded: it cannot be used to drag a card into place
    clamped = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, shift_tolerance=0.02,
                               db_path=unified_db)
    assert abs(clamped[0].shift) <= 0.02 + 1e-9


def test_matched_pairs_report_measured_peaks_not_shifted_ones(unified_db):
    """The accept step removes explained peaks by value, so a shifted match
    must still hand back the peaks AS MEASURED."""
    tt = xid.d_to_two_theta(np.array(QUARTZ_D)) + 0.10
    r = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, shift_tolerance=0.15,
                         db_path=unified_db)[0]
    measured = {round(v, 4) for v in tt}
    assert all(round(q, 4) in measured for q, _card in r.matched_pairs)


def test_match_reports_card_coverage_and_strongest_line(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    r = xid.search_match(tt, QUARTZ_I, db_path=unified_db)[0]
    assert r.n_card_lines == len(QUARTZ_D)
    assert r.n_matched == len(QUARTZ_D)
    assert r.strongest_matched is True
    # drop the 100% line from the measurement: the flag must go down
    weak = xid.search_match(tt[1:], QUARTZ_I[1:], db_path=unified_db)
    quartz = next(m for m in weak if "Quartz" in (m.mineral + m.name))
    assert quartz.strongest_matched is False


def test_min_matched_drops_one_line_coincidences(unified_db):
    tt = xid.d_to_two_theta(np.array([QUARTZ_D[0]]))
    assert xid.search_match(tt, [100.0], db_path=unified_db)  # default keeps it
    assert xid.search_match(tt, [100.0], min_matched=3, db_path=unified_db) == []


def test_chemistry_filter_is_applied_before_the_candidate_cut(unified_db):
    """Regression: filtering the TRUNCATED candidate pool threw away every
    chemically valid card that ranked past the cut on line coincidence
    alone — exactly the cards a chemistry filter is there to keep."""
    tt = np.concatenate([xid.d_to_two_theta(np.array(QUARTZ_D)),
                         xid.d_to_two_theta(np.array(CALCITE_D))])
    ii = np.concatenate([QUARTZ_I, [9.0] * len(CALCITE_D)])  # calcite barely present
    hits = xid.search_match(tt, ii, elements_all=["Ca"], max_candidates=1,
                            db_path=unified_db)
    assert [h.mineral for h in hits] == ["Calcite"]


# ---------------------------------------------------------------------------
# Structure (CIF) export
# ---------------------------------------------------------------------------

def test_cod_id_only_for_cards_that_really_carry_one():
    assert xid.cod_id_for_card("COD2205", "1011172") == "1011172"
    assert xid.cod_id_for_card("COD1906-INO", "4331511") == "4331511"
    assert xid.cod_id_for_card("PDF2", "500493") is None       # a PDF number
    assert xid.cod_id_for_card("COD2205", "12") is None        # too short to be one
    assert xid.cod_id_for_card("COD2205", "00-050-0493") is None
    assert xid.cod_id_for_card("", "1011172") is None


def test_find_local_cif_handles_flat_and_cod_tree_layouts(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "1011172.cif").write_text("data_flat")
    assert xid.find_local_cif("1011172", [str(flat)]) == str(flat / "1011172.cif")

    tree = tmp_path / "mirror"
    nested = tree / "cif" / "1" / "01" / "11"
    nested.mkdir(parents=True)
    (nested / "1011172.cif").write_text("data_tree")
    assert xid.find_local_cif("1011172", [str(tree)]) == str(nested / "1011172.cif")
    assert xid.find_local_cif("9999999", [str(tree), str(flat)]) is None
    assert xid.find_local_cif("not-a-number", [str(flat)]) is None


def test_export_phase_cifs_from_a_local_mirror_never_touches_the_network(tmp_path, monkeypatch):
    def _no_network(*a, **k):
        raise AssertionError("export must not download when the mirror has the file")
    monkeypatch.setattr(xid, "download_cod_cif", _no_network)

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "1011172.cif").write_text("data_quartz\n_cell_length_a 4.913\n")
    out = tmp_path / "rietveld"
    report = xid.export_phase_cifs(
        [{"source": "COD2205", "source_code": "1011172", "mineral": "Quartz",
          "formula": "Si O2", "fom": 0.93},
         {"source": "PDF2", "source_code": "500493", "mineral": "Lead Bismuth Vanadium Oxide",
          "formula": "Pb4 Bi V O8", "fom": 0.41}],
        str(out), local_folders=[str(mirror)], log=lambda *a: None)

    assert [label for label, _ in report["written"]] == ["Quartz"]
    written = report["written"][0][1]
    assert os.path.basename(written) == "Quartz_COD1011172.cif"
    assert "data_quartz" in open(written, encoding="utf-8").read()
    # the PDF-2 card is REPORTED, not silently skipped
    assert len(report["skipped"]) == 1
    assert "no structure available" in report["skipped"][0][1]
    summary = open(report["summary"], encoding="utf-8").read()
    assert "Quartz_COD1011172.cif" in summary and "Pb4 Bi V O8" in summary


def test_export_offline_reports_what_it_could_not_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(xid, "XRD_CIF_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "out"
    report = xid.export_phase_cifs(
        [{"source": "COD2205", "source_code": "1011172", "mineral": "Quartz"}],
        str(out), allow_download=False, log=lambda *a: None)
    assert report["written"] == []
    assert "downloads are off" in report["skipped"][0][1]


def test_download_cod_cif_is_cached_and_atomic(tmp_path, monkeypatch):
    """One fetch per entry; a re-export of the same phase works offline."""
    calls = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            calls.append(1)
            return b"data_1011172\n_cell_length_a 4.913\n"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    cache = str(tmp_path / "cache")
    first = xid.download_cod_cif("1011172", cache_dir=cache)
    second = xid.download_cod_cif("1011172", cache_dir=cache)
    assert first == second and len(calls) == 1
    assert not os.path.exists(first + ".part")


def test_find_card_by_code_round_trips_a_recorded_identification(unified_db):
    card = xid.find_card_by_code("SRC1", "1010", db_path=unified_db)
    assert card is not None
    assert card["mineral"] == "Quartz"
    assert len(card["d"]) == len(QUARTZ_D)
    assert xid.find_card_by_code("SRC1", "does-not-exist", db_path=unified_db) is None


# ---------------------------------------------------------------------------
# A folder of CIFs as a search-match source
# ---------------------------------------------------------------------------

def _write_cif(path, *, a, atoms, name, formula, sg="P 1"):
    """A P1 CIF with the given cell and atoms - enough for the calculator."""
    lines = [
        "data_x", f"_chemical_name_mineral '{name}'",
        f"_chemical_formula_sum '{formula}'",
        f"_symmetry_space_group_name_H-M '{sg}'",
        f"_cell_length_a {a[0]}", f"_cell_length_b {a[1]}", f"_cell_length_c {a[2]}",
        f"_cell_angle_alpha {a[3]}", f"_cell_angle_beta {a[4]}", f"_cell_angle_gamma {a[5]}",
        "loop_", "_symmetry_equiv_pos_as_xyz", "x,y,z",
        "loop_", "_atom_site_label", "_atom_site_type_symbol",
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
    ]
    for k, (el, x, y, z) in enumerate(atoms, 1):
        lines.append(f"{el}{k} {el} {x} {y} {z}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture()
def cif_folder(tmp_path):
    folder = tmp_path / "structures"
    (folder / "nested").mkdir(parents=True)
    _write_cif(folder / "halite.cif", a=(5.64, 5.64, 5.64, 90, 90, 90),
               atoms=[("Na", 0, 0, 0), ("Cl", 0.5, 0.5, 0.5)], name="Halite",
               formula="Na Cl", sg="F m -3 m")
    _write_cif(folder / "nested" / "periclase_1000053.cif", a=(4.21, 4.21, 4.21, 90, 90, 90),
               atoms=[("Mg", 0, 0, 0), ("O", 0.5, 0.5, 0.5)], name="Periclase",
               formula="Mg O", sg="F m -3 m")
    (folder / "notes.txt").write_text("not a structure")
    (folder / "broken.cif").write_text("data_broken\n_chemical_formula_sum 'X'\n")
    return str(folder)


def test_build_cif_database_calculates_indexes_and_skips_unreadable(cif_folder, tmp_path):
    out = tmp_path / "cifs.sq"
    logged = []
    counts = xid.build_cif_database(cif_folder, out_path=str(out), tag="MYCIF",
                                    log=logged.append)
    assert counts == {"MYCIF": 2}                 # the .txt and the broken .cif are not cards
    assert any("broken.cif" in m for m in logged)  # and the broken one is REPORTED
    assert xid.sniff_sq_format(str(out)) == "prism"
    summary = xid.database_summary(str(out))
    assert summary["by_source"] == {"MYCIF": 2}

    con = sqlite3.connect(str(out))
    names = sorted(r[0] for r in con.execute("SELECT name FROM cards"))
    codes = sorted(r[0] for r in con.execute("SELECT source_code FROM cards"))
    elements = sorted({r[0] for r in con.execute("SELECT element FROM elements")})
    con.close()
    assert names == ["Halite", "Periclase"]
    assert codes == ["halite", "periclase_1000053"]   # the filename stem is the code
    assert elements == ["Cl", "Mg", "Na", "O"]        # chemistry filters work on them


def test_a_cif_folder_database_identifies_its_own_structures(cif_folder, tmp_path):
    """The round trip that matters: calculate a structure's pattern, hand
    it back as a measurement, and get that structure at the top."""
    out = tmp_path / "cifs.sq"
    xid.build_cif_database(cif_folder, out_path=str(out), tag="MYCIF", log=lambda *a: None)
    from xrd import cif_patterns as cifp
    pat = cifp.powder_pattern_from_cif(os.path.join(cif_folder, "halite.cif"),
                                       two_theta_max=90, use_cache=False)
    res = xid.search_match(pat.two_theta, pat.intensity, db_path=str(out),
                           tol_two_theta=0.1, two_theta_range=(5.0, 90.0))
    assert res and res[0].name == "Halite"
    assert res[0].fom > 0.95
    assert res[0].source_code == "halite"
    # and the chemistry filter reaches the calculated element list
    mg = xid.search_match(pat.two_theta, pat.intensity, elements_all=["Mg"],
                          db_path=str(out), two_theta_range=(5.0, 90.0))
    assert all(r.name == "Periclase" for r in mg)


def test_cif_derived_card_remembers_its_file_and_exports_it(cif_folder, tmp_path, monkeypatch):
    def _no_network(*a, **k):
        raise AssertionError("a structure the user already has must never be downloaded")
    monkeypatch.setattr(xid, "download_cod_cif", _no_network)

    out = tmp_path / "cifs.sq"
    xid.build_cif_database(cif_folder, out_path=str(out), tag="MYCIF", log=lambda *a: None)
    src = xid.cif_path_for_card("MYCIF", "halite", db_paths=[str(out)])
    assert src is not None and os.path.samefile(src, os.path.join(cif_folder, "halite.cif"))
    assert xid.cif_path_for_card("MYCIF", "nope", db_paths=[str(out)]) is None

    dest = tmp_path / "rietveld"
    report = xid.export_phase_cifs(
        [{"source": "MYCIF", "source_code": "halite", "mineral": "Halite",
          "formula": "Na Cl", "fom": 0.98}],
        str(dest), db_paths=[str(out)], log=lambda *a: None)
    assert len(report["written"]) == 1 and not report["skipped"]
    written = report["written"][0][1]
    assert "matched against" in open(report["summary"], encoding="utf-8").read()
    assert open(written, encoding="utf-8").read() == \
        open(os.path.join(cif_folder, "halite.cif"), encoding="utf-8").read()


def test_register_cif_folder_recalculates_when_files_are_added(cif_folder, tmp_path):
    reg = str(tmp_path / "reg.json")
    imp = str(tmp_path / "imported")
    entry = xid.register_cif_folder(cif_folder, registry_path=reg, import_dir=imp,
                                    log=lambda *a: None)
    assert entry["kind"] == "cif"
    assert entry["cif_folder"] == os.path.abspath(cif_folder)
    assert entry["ncards"] == 2
    assert xid.enabled_database_paths(reg) == [entry["path"]]

    _write_cif(pathlib.Path(cif_folder) / "extra.cif", a=(3.0, 3.0, 3.0, 90, 90, 90),
               atoms=[("Si", 0, 0, 0)], name="Extra", formula="Si")
    again = xid.register_cif_folder(cif_folder, registry_path=reg, import_dir=imp,
                                    log=lambda *a: None)
    assert again["ncards"] == 3
    assert len(xid.load_registry(reg)) == 1  # re-registering updates, never duplicates
    assert xid.database_summary(again["path"])["total_cards"] == 3


def test_build_cif_database_on_a_folder_without_cifs_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No .cif files"):
        xid.build_cif_database(str(empty), out_path=str(tmp_path / "x.sq"),
                               log=lambda *a: None)


# ---------------------------------------------------------------------------
# Local noise: the detection limit has to mean the same thing everywhere
# ---------------------------------------------------------------------------

def _poisson_pattern(bg_fn, peaks=(), seed=0, step=0.02, lo=10.0, hi=80.0):
    x = np.arange(lo, hi, step)
    base = bg_fn(x)
    for c, h, w in peaks:
        base = base + h * np.exp(-0.5 * ((x - c) / w) ** 2)
    return x, np.random.default_rng(seed).poisson(np.clip(base, 0, None)).astype(float)


def test_local_noise_tracks_the_counting_statistics():
    """Truth is known: Poisson counts at level L have sigma = sqrt(L). A
    single global sigma cannot be right at both ends of a pattern whose
    background varies by a factor of fifteen."""
    bg = lambda x: 120.0 + 1800.0 * np.exp(-0.5 * ((x - 24.0) / 7.0) ** 2)
    x, y = _poisson_pattern(bg, seed=3)
    sigma = xid.local_noise_sigma(y, int(round(3.0 / 0.02)))
    truth = np.sqrt(bg(x))
    ratio = sigma / truth
    assert np.median(ratio) == pytest.approx(1.0, abs=0.12)
    assert np.mean((ratio > 0.65) & (ratio < 1.45)) > 0.97
    # and it really varies: the halo top is noisier than the tail
    i_halo = int(np.argmin(np.abs(x - 24.0)))
    i_tail = int(np.argmin(np.abs(x - 75.0)))
    assert sigma[i_halo] > 2.5 * sigma[i_tail]


def test_local_noise_is_sane_at_both_ends_of_the_scan():
    """Regression: filling the window at the edges by REPEATING the edge
    sample put ~75 copies of one noisy value in the window, and the median
    returned it — the estimate at the last point of a scan came out at a
    fifth of the truth, so the threshold collapsed exactly where weak
    high-angle reflections live."""
    bg = lambda x: 120.0 + 1800.0 * np.exp(-0.5 * ((x - 24.0) / 7.0) ** 2)
    for seed in range(6):
        x, y = _poisson_pattern(bg, seed=seed)
        sigma = xid.local_noise_sigma(y, int(round(3.0 / 0.02)))
        for i in (0, 1, 2, len(x) - 3, len(x) - 2, len(x) - 1):
            assert 0.5 < sigma[i] / np.sqrt(bg(x))[i] < 2.0, \
                f"seed {seed}, point {i}: {sigma[i]:.1f} vs {np.sqrt(bg(x))[i]:.1f}"


def test_local_noise_beats_a_global_threshold_on_a_halo_pattern():
    """The user-visible payoff: with one global sigma the counting noise on
    top of an amorphous halo comes back as dozens of 'reflections'.

    Averaged over several noise draws, because the spurious count of any
    single draw is itself a random variable — a one-seed assertion here
    would be a flake waiting to happen rather than a measurement."""
    bg = lambda x: 120.0 + 2200.0 * np.exp(-0.5 * ((x - 24.0) / 6.0) ** 2)
    planted = [(18.0, 9000.0, 0.06), (33.0, 9000.0, 0.06),
               (47.0, 4000.0, 0.06), (66.0, 2500.0, 0.06)]
    spur_loc, spur_glob = [], []
    for seed in range(6):
        x, y = _poisson_pattern(bg, planted, seed=seed)
        found = {}
        for tag, local in (("loc", True), ("glob", False)):
            found[tag] = [p.two_theta for p in xid.find_xrd_peaks(
                x, y, min_prominence_sigma=4.0, max_peaks=300, local_noise=local)]
            for c, _h, _w in planted:     # both must still find the real lines
                assert any(abs(v - c) < 0.05 for v in found[tag]), \
                    f"{tag} finder missed {c} on seed {seed}"
        spur_loc.append(sum(1 for v in found["loc"]
                            if all(abs(v - c) > 0.2 for c, _h, _w in planted)))
        spur_glob.append(sum(1 for v in found["glob"]
                             if all(abs(v - c) > 0.2 for c, _h, _w in planted)))
    mean_loc = float(np.mean(spur_loc))
    mean_glob = float(np.mean(spur_glob))
    assert mean_glob > 3.0 * mean_loc, f"local {mean_loc:.1f} vs global {mean_glob:.1f}"


def test_detection_limit_default_is_the_calibrated_one():
    """4 sigma, not 3: measured over 110 simulated patterns built from real
    structures, 3 sigma left ~9 spurious peaks per pattern and 4 left 1.4,
    for 0.1% of the genuine ones."""
    assert xid.PeakSearchOptions().min_prominence_sigma == 4.0
    import inspect
    sig = inspect.signature(xid._find_peaks_impl)
    assert sig.parameters["min_prominence_sigma"].default == 4.0


# ---------------------------------------------------------------------------
# Search options
# ---------------------------------------------------------------------------

def test_peak_search_options_round_trip_and_tolerate_junk():
    opts = xid.PeakSearchOptions(min_prominence_sigma=5.5, flag_ka2=False,
                                 two_theta_max=62.0, max_peaks=25)
    back = xid.PeakSearchOptions.from_dict(opts.to_dict())
    assert back.min_prominence_sigma == 5.5
    assert back.flag_ka2 is False
    assert back.two_theta_max == 62.0
    assert back.max_peaks == 25
    # strings (what a settings store hands back), unknown keys, and junk
    loose = xid.PeakSearchOptions.from_dict(
        {"min_prominence_sigma": "3", "local_noise": "false", "max_peaks": "12",
         "smooth_width": "not a number", "from_the_future": 1})
    assert loose.min_prominence_sigma == 3.0
    assert loose.local_noise is False
    assert loose.max_peaks == 12
    assert loose.smooth_width == xid.PeakSearchOptions().smooth_width  # junk ignored
    assert xid.PeakSearchOptions.from_dict(None).min_prominence_sigma == 4.0


def test_options_translate_the_range_to_search_kwargs():
    assert xid.PeakSearchOptions().as_kwargs()["two_theta_range"] is None
    lo, hi = xid.PeakSearchOptions(two_theta_min=20.0).as_kwargs()["two_theta_range"]
    assert lo == 20.0 and hi > 1e8
    lo, hi = xid.PeakSearchOptions(two_theta_max=70.0).as_kwargs()["two_theta_range"]
    assert lo < -1e8 and hi == 70.0


def test_search_range_selects_peaks_without_moving_them():
    """Narrowing the window must not change the background, the noise or
    the positions — only which peaks are reported."""
    bg = lambda x: 300.0 + 900.0 * np.exp(-0.5 * ((x - 22.0) / 6.0) ** 2)
    planted = [(15.0, 6000.0, 0.06), (35.0, 6000.0, 0.06), (65.0, 6000.0, 0.06)]
    x, y = _poisson_pattern(bg, planted, seed=5)
    everything = xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0)
    windowed = xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0,
                                  two_theta_range=(30.0, 70.0))
    assert [p.two_theta for p in windowed] == \
        [p.two_theta for p in everything if 30.0 <= p.two_theta <= 70.0]
    assert any(abs(p.two_theta - 35.0) < 0.05 for p in windowed)
    assert not any(p.two_theta < 30.0 for p in windowed)


def test_traced_search_reports_what_it_did():
    bg = lambda x: 200.0 + 1500.0 * np.exp(-0.5 * ((x - 25.0) / 6.0) ** 2)
    x, y = _poisson_pattern(bg, [(30.0, 8000.0, 0.06)], seed=7)
    peaks, trace = xid.find_xrd_peaks_traced(x, y, min_prominence_sigma=4.0)
    assert peaks and trace is not None
    assert len(trace.x) == len(trace.background) == len(trace.threshold) == len(x)
    # the reported background is the one actually removed
    assert np.allclose(trace.background + trace.corrected, y, atol=1e-6)
    # and it looks like a background: under the data, tracking the halo
    assert trace.background[np.argmin(np.abs(x - 25.0))] > \
        2.0 * trace.background[np.argmin(np.abs(x - 75.0))]
    # the threshold is the limit times the local noise, everywhere
    assert np.allclose(trace.threshold, 4.0 * trace.sigma)
    assert trace.background_width > 0
    # find_xrd_peaks is the same search without the bookkeeping
    assert [p.two_theta for p in xid.find_xrd_peaks(x, y, min_prominence_sigma=4.0)] == \
        [p.two_theta for p in peaks]
