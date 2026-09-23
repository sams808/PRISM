"""Tests for saxs.core.series_figures: the series table, panel counts, letters, labels at curve ends,
fit-panel pairs, trend markers (open / faded) and the export formats. Synthetic names only."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from saxs.core import series_figures as sf
from saxs.core import unified_levels as ul
from saxs.core.curve import Curve

Q = np.geomspace(5e-3, 0.22, 400)


def _table():
    rows = []
    for r_, fam in enumerate(("family X", "family Y")):
        for c_, grp in enumerate(("G1", "G2", "G3")):
            for k, lab in enumerate(("base", "one", "two")):
                if k == 0 and r_ == 1:
                    continue                                    # the parent is drawn once, in the first row
                rows.append(dict(sample=f"{grp}_{lab}_{r_}" if k else f"{grp}_{lab}", panel_row=r_, panel_col=c_, stack_index=k,
                                 label=lab, colour=["#7f7f7f", "#1f77b4", "#ff7f0e"][c_], marker="D" if k == 0 else ("o" if r_ == 0 else "s"),
                                 linestyle="-" if r_ == 0 else "--", x_value=k * 2.5, family_label=fam, col_title=f"{grp} series",
                                 quantified=(k < 2), note_symbol="" if k < 2 else "†", group=grp))
    return sf.SeriesTable.from_dataframe(pd.DataFrame(rows))


def _curve(name, level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    I = 1e-9 * Q ** -4 + level * np.exp(-(Q * 100.0) ** 2 / 3) + 5e-3
    I = I * (1 + rng.normal(scale=0.01, size=len(Q)))
    return Curve(q=Q, intensity=I, sigma=0.01 * I, name=name, metadata={"sample": name, "class_mu_t": "marginal",
                                                                          "columns": {"usable": np.ones(len(Q), int).tolist()}})


def test_series_table_validation_and_lookup(tmp_path):
    t = _table()
    assert t.shape() == (2, 3) and len(t.panels()) == 6
    assert t.samples_in_panel(0, 0) == ["G1_base", "G1_one_0", "G1_two_0"]
    assert t.samples_in_panel(1, 2) == ["G3_one_1", "G3_two_1"]
    assert t.groups() == ["G1", "G2", "G3"] and t.samples_in_group("G2")[0] == "G2_base"
    assert t.label("G1_two_0") == "two†" and t.label("G1_one_0") == "one"
    assert t.colour("G2_one_0") == "#1f77b4" and t.marker("G1_base") == "D" and t.linestyle("G1_one_1") == "--"
    assert t.quantified("G1_one_0") and not t.quantified("G1_two_0")
    assert t.get("G1_base", "x_value") == 0.0 and t.get("nope", "x_value", 7) == 7
    path = tmp_path / "series.csv"
    t.df.to_csv(path, index=False, encoding="utf-8-sig")
    back = sf.SeriesTable.from_csv(str(path))
    assert back.samples == t.samples and back.label("G1_two_0") == "two†"
    with pytest.raises(ValueError, match="lacks"):
        sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["a"], "label": ["a"]}))
    with pytest.raises(ValueError, match="duplicate"):
        sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["a", "a"], "panel_row": [0, 0], "panel_col": [0, 0], "stack_index": [0, 1], "label": ["a", "b"]}))
    with pytest.raises(ValueError, match="numeric"):
        sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["a"], "panel_row": ["x"], "panel_col": [0], "stack_index": [0], "label": ["a"]}))
    minimal = sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["a", "b"], "panel_row": [0, 0], "panel_col": [0, 1], "stack_index": [0, 0], "label": ["a", "b"]}))
    assert minimal.quantified("a") and minimal.marker("a") == "o" and minimal.colour("a") != minimal.colour("b")
    q = sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["a"], "panel_row": [0], "panel_col": [0], "stack_index": [0], "label": ["a"], "quantified": ["no"]}))
    assert not q.quantified("a")
    # a parent drawn in two panels, and an explicit fit order inside a group
    e = sf.SeriesTable.from_dataframe(pd.DataFrame({"sample": ["p", "x", "y"], "panel_row": [0, 0, 1], "panel_col": [0, 0, 0],
                                                    "stack_index": [0, 1, 1], "label": list("pxy"), "extra_panels": ["1:0", "", None],
                                                    "group": ["g", "g", "g"], "fit_order": [2, 3, 1]}))
    assert e.shape() == (2, 1) and e.samples_in_panel(1, 0) == ["p", "y"] and e.samples_in_panel(0, 0) == ["p", "x"]
    assert e.extra_panels("p") == [(1, 0)] and e.extra_panels("x") == []
    assert e.samples_in_group("g") == ["y", "p", "x"]
    curves = {n: _curve(n) for n in "pxy"}
    fig = sf.grid_figure(curves, e)
    assert len(fig.get_axes()) == 2 and len(fig.get_axes()[1].get_lines()) == 4        # p and y, each with a leader


def test_grid_figure_panels_letters_labels_and_export(tmp_path):
    t = _table()
    curves = {s: _curve(s, level=float(t.get(s, "stack_index")) * 20, seed=i) for i, s in enumerate(t.samples)}
    del curves["G3_two_1"]                                          # a table row without a curve is skipped
    fig = sf.grid_figure(curves, t, shift_decades=1.6, ylabel="Intensity (normalized, shifted)")
    axes = [a for a in fig.get_axes()]
    assert len(axes) == 6
    texts = {a: [tx.get_text() for tx in a.texts] for a in axes}
    letters = [[x for x in texts[a] if x in sf.LETTERS and len(x) == 1][0] for a in axes]
    assert letters == list("ABCDEF")
    assert all("family X" in texts[a] for a in axes[:3]) and all("family Y" in texts[a] for a in axes[3:])
    assert [a.get_title() for a in axes[:3]] == ["G1 series", "G2 series", "G3 series"] and axes[3].get_title() == ""
    # labels at the curve ends: 3 curves in the first panel (base, one, two†) and 2 in the last (one only: two was removed)
    assert sorted(x for x in texts[axes[0]] if x not in ("A", "family X")) == ["base", "one", "two†"]
    assert sorted(x for x in texts[axes[5]] if x not in ("F", "family Y")) == ["one"]
    # curves and leader lines: 2 line artists per curve
    assert len(axes[0].get_lines()) == 6 and len(axes[5].get_lines()) == 2
    # stacked curves are shifted upward by 1.6 decades per stack index
    ys = [np.nanmax(ln.get_ydata()) for ln in axes[0].get_lines()[::2]]      # normalised maxima: exactly the shift apart
    assert np.log10(ys[1] / ys[0]) == pytest.approx(1.6, abs=1e-9) and np.log10(ys[2] / ys[1]) == pytest.approx(1.6, abs=1e-9)
    assert axes[0].get_xscale() == "log" and axes[0].get_yscale() == "log" and axes[0].get_ylabel().startswith("Intensity")
    paths = sf.export_figure(fig, str(tmp_path / "figs" / "grid"), formats=("pdf", "svg", "png"), dpi=72)
    assert [os.path.basename(p) for p in paths] == ["grid.pdf", "grid.svg", "grid.png"] and all(os.path.getsize(p) > 1000 for p in paths)


@pytest.fixture(scope="module")
def fits():
    from test_unified_levels import poisson_curve, true_curve
    out = {}
    for i, name in enumerate(("G1_base", "G1_one_0", "G1_two_0", "G2_base")):
        c = poisson_curve(true_curve(l2=(i % 2 == 0)), name=name, seed=i)
        r = ul.fit_unified_levels(c, ul.UnifiedOptions(rg1_starts=(400.0,), rg2_starts=(80.0,)))
        r.relaxed = r
        r.resolved = True if (r.l2 and i == 0) else (False if r.l2 else None)
        out[name] = r
    return out


def test_fit_panels_figure_has_main_and_residual_pairs(fits, tmp_path):
    t = _table()
    fig = sf.fit_panels_figure(fits, ["G1_base", "G1_one_0", "G1_two_0", "G2_base", "missing"], t, ncol=3, title="series G")
    axes = fig.get_axes()
    assert len(axes) == 8                                            # 4 samples x (main + residual)
    mains = axes[0::2]
    assert mains[0].get_title().startswith("G1_base") and mains[0].texts[0].get_text() == "A" and mains[3].texts[0].get_text() == "D"
    labels = [h.get_label() for h in mains[0].get_legend().get_lines()]
    assert labels[0] == "data" and labels[1].startswith("fit (") and any(l.startswith("level 1") for l in labels)
    if fits["G1_base"].l2:
        assert any(l.startswith("level 2") for l in labels)
    assert axes[1].get_ylim() == (-0.25, 0.25) and axes[1].get_xscale() == "log"
    assert mains[0].get_xlim()[0] == pytest.approx(4.5e-3) and axes[-1].get_xlabel().startswith("Q")
    assert fig._suptitle.get_text() == "series G"
    fig2 = sf.fit_panels_figure(fits, ["G1_base"], None, ncol=1, letters=False, level_names={"L1": "grains", "L2": "nano"})
    lab2 = [h.get_label() for h in fig2.get_axes()[0].get_legend().get_lines()]
    assert any(l.startswith("grains") for l in lab2) and not fig2.get_axes()[0].texts
    with pytest.raises(ValueError):
        sf.fit_panels_figure(fits, ["missing"], t)
    sf.export_figure(fig, str(tmp_path / "fits"), formats=("png",), dpi=50)
    assert os.path.exists(tmp_path / "fits.png")


def test_trends_figure_marks_unquantified_open_and_unresolved_faded(fits):
    t = _table()
    desc = pd.DataFrame({"sample": t.samples, "n_at_0.04": -4.0 + 0.1 * np.arange(len(t.samples)),
                         "C_hi": 1e-3 * (1 + np.arange(len(t.samples))), "ratio_hi": [1.0] + [5.0] * (len(t.samples) - 1)})
    fit_df = ul.results_table(list(fits.values()))
    fig = sf.trends_figure(t, desc, fit_df, x_label="x (mol%)")
    axes = fig.get_axes()
    assert len(axes) == 4 and [a.texts[0].get_text() for a in axes] == list("ABCD")
    # panel A: every sample with an x-value; open marker (no face colour) for the non-quantified ones
    pts = [ln for ln in axes[0].get_lines() if ln.get_marker() not in ("", "None", None) and len(ln.get_xdata()) == 1]
    assert len(pts) == len(t.samples)
    open_faces = [ln for ln in pts if ln.get_markerfacecolor() == "none"]
    assert len(open_faces) == sum(not t.quantified(s) for s in t.samples)
    assert axes[0].get_ylabel().startswith("local exponent") and axes[0].get_xlabel() == "x (mol%)"
    # panel B: the sample with ratio_hi = 1 is excluded by the requirement
    pts_b = [ln for ln in axes[1].get_lines() if len(ln.get_xdata()) == 1]
    assert len(pts_b) == len(t.samples) - 1 and axes[1].get_yscale() == "log"
    # panels C / D: only fitted samples with a level 2; unresolved ones faded
    pts_d = [ln for ln in axes[3].get_lines() if len(ln.get_xdata()) == 1 and ln.get_marker() not in ("", "None", None)]
    faded = [ln for ln in pts_d if ln.get_alpha() == 0.35]
    n_l2 = sum(1 for r in fits.values() if r.l2)
    n_unres = sum(1 for r in fits.values() if r.l2 and r.resolved is False)
    assert len(pts_d) + len(axes[3].containers) >= n_l2 and len(faded) == n_unres
    legend_labels = [h.get_label() for h in axes[0].get_legend().legend_handles]
    assert "G1" in legend_labels and "family X" in legend_labels and "not quantified" in legend_labels and "level not resolved" in legend_labels
    # custom panels and no legend
    p = [sf.TrendPanel("k", "descriptors", "C_hi", "C", yscale="log", transform=lambda v: 2 * v, hlines=(1e-3,))]
    fig2 = sf.trends_figure(t, desc, None, panels=p, legend=False, letters=False)
    a2 = fig2.get_axes()[0]
    assert a2.get_legend() is None and not a2.texts and len(a2.get_lines()) == len(t.samples) + 1


@pytest.mark.skipif(not __import__("saxs.core.edf_sectors", fromlist=["HAVE_FABIO"]).HAVE_FABIO, reason="fabio not installed")
def test_frame_figure_three_panels(tmp_path):
    import saxs2_synthetic as syn
    paths = syn.make_campaign(str(tmp_path))
    fig = sf.frame_figure(paths["A/S2"], paths["A/Empty"], crop_px=40, q_marker=7e-3, titles=("sample S2", "empty"))
    axes = fig.get_axes()
    assert len(axes) == 3 and axes[0].get_title() == "sample S2" and axes[2].get_xscale() == "log"
    assert len(axes[0].images) == 1 and len(axes[0].patches) == 1 and len(axes[2].get_lines()) == 5    # 4 profiles + marker
    assert [a.texts[0].get_text() for a in axes] == ["A", "B", "C"]
    sf.export_figure(fig, str(tmp_path / "frame"), formats=("png",), dpi=50)
    assert os.path.exists(tmp_path / "frame.png")


def test_apply_style_and_rebinned_curve():
    import matplotlib
    with sf.apply_style("Publication", {"font.size": 11}):
        assert matplotlib.rcParams["font.size"] == 11 and matplotlib.rcParams["pdf.fonttype"] == 42
    with sf.apply_style("Presentation"):
        assert matplotlib.rcParams["font.size"] == 14
    c = _curve("x")
    q, I = sf.rebinned_curve(c, per_decade=60, q_max=0.1)
    assert q.max() <= 0.1 and 50 < len(q) < 120
    q2, I2 = sf.rebinned_curve((Q, np.ones_like(Q)), per_decade=20)
    assert len(q2) < len(q) and np.allclose(I2, 1.0)
