"""Tests for the SAXS/WAXS module (saxs_core port of pomme + qt_saxs) and
the Hephaestus-style XAS sample-mass calculator (xas_mass)."""
from __future__ import annotations

import numpy as np
import pytest

import xas_mass
from qt_models import SpectrumLibrary
from qt_saxs import SaxsWorkspace
from qt_shell import MODULES, NAV_ITEMS, PrismMainWindow
from saxs_core.analysis import fit_guinier, fit_pseudo_bragg_peak
from saxs_core.curve import Curve
from saxs_core.waxs import auto_find_peaks, fit_waxs_peaks


def _sphere_curve(rg=30.0, name="sample"):
    q = np.linspace(0.005, 0.5, 600)
    intensity = 1000.0 * np.exp(-(q * rg) ** 2 / 3.0) + 2.0
    return Curve(q=q, intensity=intensity, sigma=None, name=name)


def test_guinier_recovers_rg_on_synthetic_curve():
    c = _sphere_curve(rg=30.0)
    r = fit_guinier(c.q, c.intensity, 0.006, 1.0 / 30.0)
    assert r.Rg == pytest.approx(30.0, rel=0.05)
    assert r.r2 > 0.99


def test_pseudo_bragg_peak_d_spacing():
    q = np.linspace(0.05, 0.6, 800)
    intensity = 50.0 * np.exp(-((q - 0.30) / 0.02) ** 2) + 10.0
    r = fit_pseudo_bragg_peak(q, intensity, 0.2, 0.4)
    assert r.q0 == pytest.approx(0.30, abs=0.005)
    assert r.d_spacing == pytest.approx(2 * np.pi / 0.30, rel=0.02)


def test_waxs_multi_peak_fit_and_crystallinity():
    q = np.linspace(0.5, 4.0, 1200)
    rng = np.random.default_rng(0)
    intensity = (200.0 * np.exp(-((q - 1.5) / 0.03) ** 2)
                 + 120.0 * np.exp(-((q - 2.2) / 0.04) ** 2)
                 + 40.0 * np.exp(-((q - 1.8) / 0.5) ** 2)  # amorphous hump
                 + 5.0 + rng.normal(0, 1.0, q.shape))
    specs = auto_find_peaks(q, intensity)
    assert len(specs) >= 2
    result = fit_waxs_peaks(q, intensity, specs)
    centers = sorted(p.center for p in result.peaks if not p.is_amorphous)
    assert any(abs(c - 1.5) < 0.05 for c in centers)
    assert any(abs(c - 2.2) < 0.05 for c in centers)
    assert result.r2 > 0.9


def test_saxs_workspace_reduction_and_send_to_library(qtbot):
    library = SpectrumLibrary()
    calls = []
    widget = SaxsWorkspace(library=library, on_derived_added=lambda ids: calls.append(ids))
    qtbot.addWidget(widget)
    sample = _sphere_curve(name="samp")
    empty = Curve(q=sample.q, intensity=np.full_like(sample.q, 2.0), sigma=None, name="empty")
    widget.add_curve(sample)
    widget.add_curve(empty)
    widget.red_sample_combo.setCurrentText("samp")
    widget.red_empty_combo.setCurrentText("empty")
    widget.red_mode_combo.setCurrentText("manual")
    widget.run_reduction()
    qtbot.wait(20)
    assert any(c.name == "samp_corr" for c in widget.curves)

    widget.curve_list.selectAll()
    widget.send_to_library()
    assert len(library) == 3
    assert calls and len(calls[0]) == 3


def test_saxs_workspace_analysis_tab_guinier(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_sphere_curve(rg=25.0, name="c"))
    widget.ana_combo.setCurrentText("c")
    widget.run_guinier()  # auto region detection fills the range
    qtbot.wait(20)
    assert "Rg" in widget.ana_report.toPlainText()


def test_saxs_module_registered_in_shell(qtbot):
    assert "SAXS/WAXS" in MODULES
    assert "SAXS/WAXS" in NAV_ITEMS
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("SAXS/WAXS"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.saxs_page


# --------------------------------------------------------------------------
# Sample-mass calculator
# --------------------------------------------------------------------------

def test_parse_components_formula_and_table():
    assert xas_mass.parse_components("Fe2O3") == [("Fe2O3", 1.0)]
    comps = xas_mass.parse_components("SiO2 58.8\nNa2O 19.6; Bi2O3 19.6")
    assert comps == [("SiO2", 58.8), ("Na2O", 19.6), ("Bi2O3", 19.6)]
    with pytest.raises(Exception):
        xas_mass.parse_components("NotAnElementZz9")


def test_element_mass_fractions_pure_compound():
    w = xas_mass.element_mass_fractions([("Fe2O3", 1.0)])
    assert w["Fe"] == pytest.approx(0.6994, abs=0.001)  # textbook value
    assert w["O"] == pytest.approx(0.3006, abs=0.001)


def test_sample_mass_report_fe2o3_is_physically_sane():
    r = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", pellet_diameter_mm=13.0)
    assert r.edge_energy_ev == pytest.approx(7112.0, abs=5.0)
    # cross-check against xraydb directly
    import xraydb
    mu_direct = sum(w * xraydb.mu_elam(el, r.edge_energy_ev + 50)
                    for el, w in xas_mass.element_mass_fractions([("Fe2O3", 1.0)]).items())
    assert r.mu_rho_above == pytest.approx(mu_direct, rel=1e-6)
    assert r.edge_step_mu_rho > 0
    # mass for mu*t = 2.5: target/mu * area — recompute independently
    area = np.pi * 0.65 ** 2
    assert r.mass_mut_25_mg == pytest.approx(2.5 * area / r.mu_rho_above * 1000, rel=1e-6)
    assert 5.0 < r.mass_mut_25_mg < 100.0  # tens of mg — the realistic pellet range


def test_sample_mass_report_oxide_mixture_bi_l3():
    """The lab's actual case: a mol% oxide composition at the Bi L3 edge."""
    comp = "SiO2 58.8\nNa2O 19.6\nBi2O3 19.6\nUO3 2.0"
    r = xas_mass.sample_mass_report(comp, "Bi", "L3", basis="mol")
    assert r.edge_energy_ev == pytest.approx(13419.0, abs=10.0)
    assert 0.2 < r.absorber_fraction < 0.6  # Bi-heavy glass
    assert r.edge_step_mu_rho > 0
    assert r.mass_step_1_mg > r.mass_mut_1_mg  # step target always needs more mass
    with pytest.raises(ValueError, match="not in the composition"):
        xas_mass.sample_mass_report("SiO2", "Bi", "L3")


def test_xas_workspace_mass_tab(qtbot):
    from qt_xas import XasWorkspace
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    widget._compute_sample_mass()  # defaults: Bi L3 on the Bi glass composition
    text = widget.mass_report_text.toPlainText()
    assert "Bi L3 edge" in text
    assert "μt = 2.5" in text
