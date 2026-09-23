"""Tests for saxs.core.tables: transliteration, the three writers and the standard sheet builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saxs.core import tables as tb


def test_ascii_transliteration_covers_the_characters_excel_garbles():
    assert tb.ascii_transliterate("0.15–0.20 Å⁻¹ ± 2 × 10⁻³ µm, R g, d ≈ 4, T≥3 °C") == \
        "0.15-0.20 A^-1 +- 2 x 10^-3 mum, R g, d ~ 4, T>=3  degC"
    assert tb.ascii_transliterate("Bi₂O₃ (mol%)") == "Bi2O3 (mol%)" and tb.ascii_transliterate("cm²·s⁻¹") == "cm^2.s^-1"
    assert tb.ascii_transliterate("naïve café") == "naive cafe"
    assert tb.ascii_transliterate(None) == "" and tb.ascii_transliterate(3.5) == "3.5"
    out = tb.ascii_transliterate("日本語 — test")
    assert out.isascii() and "test" in out


def test_format_cell_and_markdown():
    df = pd.DataFrame({"Sample": ["a", "b"], "Rg (nm)": [12.3456789, np.nan], "ok": [True, False], "note": [None, "x"]})
    md = tb.dataframe_to_markdown(df, notes=["first note", "second"])
    lines = md.splitlines()
    assert lines[0] == "| Sample | Rg (nm) | ok | note |" and lines[1] == "|---|---|---|---|"
    assert lines[2] == "| a | 12.3457 | yes | - |" and lines[3] == "| b | - | no | x |"
    assert "- first note" in md and "- second" in md
    assert tb.format_cell(1e-7) == "1e-07" and tb.format_cell(np.float64(2.5)) == "2.5" and tb.format_cell(pd.NA) == "-"


def test_write_tables_xlsx_csv_md(tmp_path):
    fits = pd.DataFrame({"Sample": ["s1", "s2"], "Rg₂ (nm)": [12.0, None], "Loading": ["opaque", "marginal"], "note": ["≈ 4 – 5", None]})
    mf = pd.DataFrame({"Sample": ["s1", "s2"], "plateau 0.15–0.20": [1e-3, 2e-3]})
    base = str(tmp_path / "out" / "tables")
    paths = tb.write_tables({"fits": fits, "model-free": mf}, base, notes=["a note with an en-dash –"])
    assert set(paths) == {"xlsx", "csv", "md"}
    import openpyxl
    wb = openpyxl.load_workbook(paths["xlsx"])
    assert wb.sheetnames == ["fits", "model-free", "notes"]
    ws = wb["fits"]
    assert ws.freeze_panes == "B2"
    assert ws["A1"].value == "Sample" and ws["B1"].value == "Rg₂ (nm)"       # xlsx keeps the unicode
    assert 8 <= ws.column_dimensions["A"].width <= 30
    assert wb["notes"]["A2"].value.startswith("a note")
    raw = open(paths["csv"], "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf")                                    # BOM
    text = raw.decode("utf-8-sig")
    assert text.isascii() and "Rg2 (nm)" in text and "~ 4 - 5" in text
    md = open(paths["md"], encoding="utf-8").read()
    assert md.startswith("| Sample | Rg₂ (nm) |") and "| s2 | - | marginal | - |" in md and "- a note" in md
    only_md = tb.write_tables({"x": fits}, str(tmp_path / "x"), formats=("md",), md_sheet="x")
    assert set(only_md) == {"md"}
    with pytest.raises(ValueError):
        tb.write_tables({}, str(tmp_path / "none"))


class _P:
    def __init__(self, value, stderr=None, at_bound=False):
        self.value, self.stderr, self.at_bound = value, stderr, at_bound


class _R:
    """A minimal stand-in for UnifiedFitResult."""

    def __init__(self, sample, l2=True, tail=True, resolved=True, fitted=True, relaxed=None, at_bound=()):
        self.sample, self.cls, self.fitted, self.skip_reason = sample, "marginal", fitted, None if fitted else "too short"
        self.l2, self.tail, self.resolved, self.relaxed = l2, tail, resolved, relaxed
        self.n_params = 10 if l2 else 6
        self.q_min, self.rms_log, self.dBIC_L2, self.covariance_ok = 0.0051, 0.0123, 66.4 if l2 else float("nan"), True
        self.params = {"Rg1": _P(600.0, 20.0, "Rg1" in at_bound), "d1": _P(4.5, None, "d1" in at_bound),
                       "Rg2": _P(120.0, 6.0, "Rg2" in at_bound), "d2": _P(3.5, 0.2, "d2" in at_bound), "logpf2": _P(-0.3, None, "logpf2" in at_bound)}
        self.derived = {"Rg1_nm": 60.0, "d1": 4.5, "G1": 1234.5, "Rg2_nm": 12.0, "D2_eq_nm": 31.0, "d2": 3.5, "G2": 137.2, "pf2": 0.5}
        self.name = "L1" + ("_L2" if l2 else "") + ("_T" if tail else "")


def test_fit_sheet_marks_bounds_and_missing_levels():
    relaxed = _R("a")
    relaxed.params["Rg2"] = _P(125.0)
    relaxed.derived["Rg2_nm"] = 12.5
    a = _R("a", relaxed=relaxed, at_bound=("d1",))
    b = _R("b", l2=False, tail=False, resolved=None)
    c = _R("c", fitted=False)
    summary = pd.DataFrame({"sample": ["a", "b"], "T_used": [0.1681, 0.0445], "t_eff_mm": [0.2529, 0.284]})
    df = tb.fit_sheet([a, b, c], summary, extra={"x (mol%)": {"a": 1.1, "b": 2.2}}, amplitude_unit="per mm")
    assert list(df["Sample"]) == ["a", "b", "c"]
    ra, rb, rc = (df.set_index("Sample").loc[k] for k in "abc")
    assert ra["d1"] == "4.50*" and ra["Rg1 (nm)"] == "60.0" and ra["Rg2 (nm)"] == "12.0" and ra["Rg2 err (nm)"] == 0.6
    assert ra["Level 2 resolved"] == "yes" and ra["Rg2, relaxed bounds (nm)"] == 12.5 and ra["dBIC (level 2)"] == 66
    assert ra["G1 (per mm)"] == 1230.0 and ra["G2 (per mm)"] == 137.0 and ra["T"] == 0.168 and ra["t_eff (mm)"] == 0.25
    assert ra["Rising tail"] == "yes" and ra["RMS log10 I"] == 0.012 and ra["x (mol%)"] == 1.1
    assert rb["Rg2 (nm)"] is None and rb["Level 2 resolved"] == "-" and rb["Rising tail"] == "no" and rb["Parameters"] == 6
    assert pd.isna(rc["Parameters"]) and rc["Note"] == "too short" and pd.isna(rc["Rg1 (nm)"])
    md = tb.dataframe_to_markdown(df)
    assert "| c | marginal |" in md and "4.50*" in md


def test_modelfree_and_robustness_sheets_and_notes():
    d = pd.DataFrame({"sample": ["a"], "cls": ["opaque"], "n_low": [4.734], "n_at_0.02": [-4.1], "n_at_0.04": [-3.94], "C_hi": [0.0123456],
                      "ratio_hi": [12.34], "K4_over_rhoe2": [1.23e-6], "shoulder_size_nm": [np.nan], "has_maximum": [False]})
    mf = tb.modelfree_sheet(d, extra={"x": {"a": 5}})
    r = mf.iloc[0]
    assert r["Sample"] == "a" and r["Low-q exponent"] == 4.73 and r["Local exponent at 0.04"] == 3.94 and r["Local exponent at 0.02"] == 4.1
    assert r["High-q plateau"] == 0.0123 and r["Sample/(T x empty) at high q"] == 12.3 and r["Porod shoulder (nm)"] is None
    assert r["Maximum in I or I q^2"] == "no" and r["x"] == 5
    assert list(mf.columns).index("Local exponent at 0.02") < list(mf.columns).index("Local exponent at 0.04")
    relaxed = _R("a")
    rs = tb.robustness_sheet([_R("a", relaxed=relaxed), _R("b", fitted=False)])
    assert len(rs) == 1 and rs.iloc[0]["Relaxed model"] == "L1_L2_T" and rs.iloc[0]["Level 2 resolved"] == "yes"
    notes = tb.default_notes({"q_max": 0.2, "dbic_min": 10.0, "relaxed_d1_max": 5.0, "relaxed_rg1_max": 3000.0, "robust_tol": 0.2})
    assert any("BIC by more than 10" in n for n in notes) and any("Rg1 <= 300 nm" in n for n in notes)
    assert len(tb.default_notes()) == 6
