"""Tests for saxs/core/contrast.py against closed forms: electron density of silica, the invariant of a Debye term,
the two-phase volume fraction round trip, the minimum contrast, a phase contrast and the stoichiometric ceiling of a
hand-computed example."""
from __future__ import annotations

import math

import pytest

from saxs.core.contrast import (
    N_A, R_E_CM, as_composition, electron_density_cm3, invariant, min_contrast_1e10cm2, phase_contrast_1e10cm2,
    sld_1e10cm2, stoichiometric_ceiling, two_phase_product_cm4, volume_fraction_from_invariant, xray_sld_1e10cm2,
)
from saxs.core.models import Dab


def test_composition_parsing():
    assert as_composition("50(SiO2)25(Na2O)") == {"SiO2": 50.0, "Na2O": 25.0}
    assert as_composition("Na3BiP2O8") == {"Na3BiP2O8": 1.0}
    assert as_composition({"SiO2": 1, "B2O3": 0}) == {"SiO2": 1.0}


def test_electron_density_of_silica_and_sld():
    rho_e = electron_density_cm3({"SiO2": 1.0}, 2.2)
    assert rho_e == pytest.approx(2.2 * N_A * 30 / 60.0843, rel=1e-4)
    assert xray_sld_1e10cm2(rho_e) == pytest.approx(R_E_CM * rho_e / 1e10)
    assert electron_density_cm3("70(SiO2)30(Na2O)", 2.4) == pytest.approx(electron_density_cm3({"SiO2": 7, "Na2O": 3}, 2.4))


def test_phase_contrast_is_the_sld_difference():
    c = phase_contrast_1e10cm2("Al2O3", 3.95, "SiO2", 2.2)
    assert c == pytest.approx(sld_1e10cm2("Al2O3", 3.95) - sld_1e10cm2("SiO2", 2.2))
    assert c > 0


def test_stoichiometric_ceiling_hand_example():
    # sample 10 Na2O - 10 P2O5 - 80 SiO2 (mol), density 2.5; phase NaPO3 (density 2.5): per mole of oxide units the
    # sample has 0.2 Na and 0.2 P -> 0.2 mol NaPO3 at most; volume fraction = 0.2 M(NaPO3)/2.5 / (M_sample/2.5)
    import xraydb
    m = lambda f: sum(n * xraydb.atomic_mass(e) for e, n in xraydb.chemparse(f).items())      # noqa: E731
    M_sample = 0.1 * m("Na2O") + 0.1 * m("P2O5") + 0.8 * m("SiO2")
    expected = 0.2 * m("NaPO3") / 2.5 / (M_sample / 2.5)
    assert stoichiometric_ceiling("NaPO3", 2.5, "10(Na2O)10(P2O5)80(SiO2)", 2.5) == pytest.approx(expected, rel=1e-9)
    assert stoichiometric_ceiling("NaPO3", 2.5, "20(Na2O)80(SiO2)", 2.5) == 0.0          # no phosphorus


def test_invariant_of_dab_and_volume_fraction_round_trip():
    A, xi = 20.0, 80.0
    Q = invariant(lambda x: Dab().eval(x, A=A, xi=xi))
    assert Q == pytest.approx(A * math.pi / (4 * xi ** 3), rel=1e-3)
    drho, phi = 5.0, 0.03
    Q2 = 2 * math.pi ** 2 * phi * (1 - phi) * (drho * 1e10) ** 2 / 1e24
    assert two_phase_product_cm4(Q2) == pytest.approx(phi * (1 - phi) * (drho * 1e10) ** 2)
    assert volume_fraction_from_invariant(Q2, drho) == pytest.approx(phi, rel=1e-9)
    assert math.isnan(volume_fraction_from_invariant(Q2, 0.1))
    # the minimum contrast reproduces Q at phi = 1/2
    dmin = min_contrast_1e10cm2(Q2)
    assert 2 * math.pi ** 2 * 0.25 * (dmin * 1e10) ** 2 / 1e24 == pytest.approx(Q2, rel=1e-9)
