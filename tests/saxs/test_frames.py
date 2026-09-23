"""Tests for saxs.core.frames: inventory of synthetic EDF frames, role detection, centre refinement,
lobe/streak diagnostics, the three pairing rules and the inventory table."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from saxs.core import edf_sectors as es
from saxs.core import frames as fr

import saxs2_synthetic as syn  # noqa: E402  (tests/saxs is on sys.path under pytest)

pytestmark = pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("frames"))
    paths = syn.make_campaign(root)
    records = fr.scan_frames(root)
    return root, paths, records


def _by_stem(records):
    return {r.stem: r for r in records}


def test_scan_finds_every_frame_with_roles_and_stem_names(campaign):
    root, paths, records = campaign
    assert len(records) == len(paths)
    by = _by_stem(records)
    assert by["Empty_0p5mm"].role == "empty"
    empties = sorted(r.rel for r in records if r.role == "empty")
    assert empties == ["sessionA/Empty.edf", "sessionB/Empty.edf", "sessionB/Empty_0p5mm.edf"]
    assert by["AgBeh"].role == "calibrant"
    assert {by[s].role for s in ("S1", "S2", "S3_0p5mm", "S4")} == {"sample"}
    assert by["S1"].name == "S1" and by["S1"].comment_name == "S1" and by["S1"].tag == "S1_1mm_SAXS"
    assert by["S1"].session == "sessionA" and by["S4"].session == "sessionB"
    assert by["S1"].configuration == "SAXS" and by["S1"].collimation == "prod"
    assert by["S3_0p5mm"].capillary_mm == 0.5 and by["Empty_0p5mm"].capillary_mm == 0.5
    assert by["S1"].capillary_mm == 1.0 and by["Empty"].capillary_mm == 1.0    # "_1mm_" in the comment tag
    assert by["S1"].shape == syn.SHAPE and by["S1"].exposure_s == 100.0
    assert abs(by["S1"].energy_ev - 8041.0) < 2


def test_scan_name_from_comment_and_exclude_pattern(campaign):
    root, paths, _ = campaign
    recs = fr.scan_frames(root, name_from="comment", exclude_pattern=r"0p5", diagnostics=False)
    stems = {r.stem for r in recs}
    assert "S3_0p5mm" not in stems and "Empty_0p5mm" not in stems
    assert all(r.name == r.comment_name for r in recs)
    assert all(r.cx is None for r in recs)      # no diagnostics: nothing refined


def test_centre_refinement_recovers_the_header_offset(campaign):
    _, _, records = campaign
    by = _by_stem(records)
    # session A headers are 6 px to the right of the true beam; the refined centre finds the beam
    for stem in ("Empty", "S1"):
        r = by[stem] if stem != "Empty" else [x for x in records if x.rel == "sessionA/Empty.edf"][0]
        assert r.cx is not None and abs(r.cx - syn.TRUE_CX) < 0.3 and abs(r.cy - syn.TRUE_CY) < 0.3
        assert abs(r.centre_offset_px[0] + 6.0) < 0.3
        assert abs(r.centre_offset_mm - 6.0 * syn.PIXEL_M * 1e3) < 0.05
    b_empty = [x for x in records if x.rel == "sessionB/Empty.edf"][0]
    assert abs(b_empty.centre_offset_px[0]) < 0.3


def test_lobe_and_streak_diagnostics(campaign):
    _, _, records = campaign
    by = _by_stem(records)
    assert by["S2"].lobe_q is not None                       # two lobes on the horizontal axis
    assert by["S2"].lobe_ratio > 3
    assert by["S1"].lobe_q is None                           # an isotropic ring is not a lobe
    assert by["S4"].streak_h > 1.5 and by["S4"].streak_v < 1.5
    assert by["S1"].streak_h < 1.5
    assert not by["S1"].saturated and by["S1"].direct_beam_cps > 0
    empty_a = [x for x in records if x.rel == "sessionA/Empty.edf"][0]
    assert by["S2"].direct_beam_cps < 0.01 * empty_a.direct_beam_cps
    # the dark-centre sample takes the session empty's centre (its own centroid would sit on a lobe)
    assert by["S2"].centre_source.startswith("session empty") and by["S2"].reference_empty == "sessionA/Empty.edf"
    assert abs(by["S2"].cx - empty_a.cx) < 1e-9 and by["S2"].cx_own is not None and abs(by["S2"].cx_own - syn.TRUE_CX) > 3
    assert by["S1"].centre_source == "own direct beam" and by["S1"].reference_empty == "sessionA/Empty.edf"
    assert empty_a.centre_source == "own direct beam" and empty_a.reference_empty == ""


def test_pair_same_folder_matches_capillary_and_configuration(campaign):
    _, _, records = campaign
    res = fr.pair_frames(records, rule="same_folder")
    got = {p.sample.stem: p.empty.rel for p in res.pairs}
    assert got == {"S1": "sessionA/Empty.edf", "S2": "sessionA/Empty.edf", "S3_0p5mm": "sessionB/Empty_0p5mm.edf",
                   "S4": "sessionB/Empty.edf"}
    assert not res.unpaired
    assert "AgBeh" not in got
    t = res.table()
    assert list(t.columns[:3]) == ["sample", "sample_frame", "empty_frame"] and len(t) == 4
    assert "4 sample(s) paired" in res.describe()


def test_pair_falls_back_to_session_then_reports_unpaired(campaign):
    _, _, records = campaign
    # move S4 into a sub-folder of session B without an empty: the session fallback still pairs it
    recs = [fr.FrameRecord(**{**r.__dict__}) for r in records]
    s4 = [r for r in recs if r.stem == "S4"][0]
    s4.folder = os.path.join(s4.folder, "sub")
    res = fr.pair_frames(recs, rule="same_folder", fallback_to_session=True)
    assert {p.sample.stem for p in res.pairs} >= {"S4"}
    res2 = fr.pair_frames(recs, rule="same_folder", fallback_to_session=False)
    assert [r.stem for r in res2.unpaired] == ["S4"]
    assert "unpaired" in res2.describe()


def test_pair_nearest_time_and_explicit(campaign):
    _, _, records = campaign
    res = fr.pair_frames(records, rule="nearest_time")
    got = {p.sample.stem: p.empty.rel for p in res.pairs}
    assert got["S1"] == "sessionA/Empty.edf" and got["S4"] == "sessionB/Empty.edf" and got["S3_0p5mm"] == "sessionB/Empty_0p5mm.edf"
    explicit = {"S1": "sessionB/Empty.edf", "S2": "sessionA/Empty.edf", "S3_0p5mm": "Empty_0p5mm", "S4": "sessionB/Empty.edf"}
    res = fr.pair_frames(records, rule="explicit", explicit=explicit)
    got = {p.sample.stem: p.empty.rel for p in res.pairs}
    assert got["S1"] == "sessionB/Empty.edf" and got["S3_0p5mm"] == "sessionB/Empty_0p5mm.edf"
    res = fr.pair_frames(records, rule="explicit", explicit={"S1": "sessionB/Empty.edf"})
    assert {r.stem for r in res.unpaired} == {"S2", "S3_0p5mm", "S4"}
    with pytest.raises(ValueError):
        fr.pair_frames(records, rule="no_such_rule")


def test_inventory_table_and_csv_round_trip(campaign, tmp_path):
    _, _, records = campaign
    df = fr.inventory_table(records)
    assert len(df) == len(records)
    for col in ("rel", "session", "name", "role", "date", "exposure_s", "distance_m", "cx", "cy", "centre_offset_px",
                "direct_beam_cps", "lobe_q", "streak_h", "capillary_mm"):
        assert col in df.columns
    assert "header" not in df.columns
    out = tmp_path / "inventory.csv"
    fr.write_inventory(records, str(out))
    back = pd.read_csv(out)
    assert len(back) == len(records) and set(back["role"]) == {"sample", "empty", "calibrant"}


def test_parse_comment_and_capillary_token():
    assert fr.parse_comment("X - X_Cu-HR-full_ESAXS_1200s", 1.8) == ("X", "X_Cu-HR-full_ESAXS_1200s", "HR", "ESAXS")
    assert fr.parse_comment("plain", 0.9) == ("plain", "", "prod", "SAXS")
    assert fr.parse_comment("", 0.37) == ("", "", "prod", "MAXS")
    assert fr.parse_comment("", 1.234) == ("", "", "prod", "D1.23")
    assert fr.capillary_token("Empty_cap_0p5mm") == 0.5 and fr.capillary_token("E_1.5 mm") == 1.5
    assert fr.capillary_token("P5Bi8-12") is None and fr.capillary_token("", "x_1mm_SAXS") == 1.0
    assert fr.classify_role("Test1_0_00017", "T1_Empty", r"(?i)empty", r"(?i)agbeh") == "empty"
    assert fr.classify_role("PBi_0_00000", "Cu_SAXS_AgBeh", r"(?i)empty", r"(?i)agbeh") == "calibrant"
    assert fr.parse_date("2026-01-09T18:46:36") is not None and fr.parse_date("") is None


def test_sector_anisotropy_flags_lobes_not_rings():
    geom = es.FrameGeometry(cx=syn.TRUE_CX, cy=syn.TRUE_CY, dist_m=syn.DIST_M, pixel_m=syn.PIXEL_M, wavelength_A=1.541891, exposure_s=100.0)
    with_lobes = 5.0 + syn.lobes(x0=20.0, amp=5e3)
    an = es.sector_anisotropy(with_lobes, geom)
    assert an.lobe_q is not None and abs(an.lobe_q - geom.q_of_radius(np.array([20.0]))[0]) < 1.5 * geom.q_of_radius(np.array([1.0]))[0]
    assert an.lobe_ratio > 5 and an.max_q_vertical is None
    iso = 5.0 + syn.ring(r0=20.0, amp=200.0)
    an2 = es.sector_anisotropy(iso, geom)
    assert an2.lobe_q is None and an2.max_q_horizontal is not None and an2.max_q_vertical is not None
    assert abs(an2.lobe_ratio - 1) < 0.05
    streaked = 5.0 + syn.streak(amp=50.0, axis="v")
    an3 = es.sector_anisotropy(streaked, geom)
    assert an3.streak_v > 1.5 and abs(an3.streak_h - 1) < 0.2
