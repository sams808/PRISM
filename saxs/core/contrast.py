"""X-ray contrast, invariant and volume-fraction helpers for absolute SAXS intensities.

Constants: r_e and N_A from CODATA 2018; atomic numbers and masses from xraydb (Elam et al. tables).

    electron_density_cm3(composition, density)      electrons per cm^3 of a material
    xray_sld_1e10cm2(rho_e)                         r_e rho_e in 1e10 cm^-2 (forward scattering, no anomalous terms)
    phase_contrast_1e10cm2(phase, rho_p, sample, rho_s)    SLD(phase) - SLD(sample)
    stoichiometric_ceiling(phase, rho_p, sample, rho_s)    largest volume fraction of a phase the sample's elements
                                                     allow (oxygen and any element in `ignore` not limiting)
    invariant(func)                                 Q = integral q^2 I dq of a model component
    two_phase_product_cm4(Q)                        phi (1 - phi) drho^2 from Q = 2 pi^2 phi (1 - phi) drho^2
    volume_fraction_from_invariant(Q, drho)         the smaller root phi (NaN when the contrast is too small)
    min_contrast_1e10cm2(Q)                         the smallest contrast able to produce Q (at phi = 1/2)

A composition is a ``{formula: amount}`` mapping (oxides in mol %, or a single formula with amount 1) or a string in
any syntax the chemistry parser accepts (``50.6(SiO2)13.6(B2O3)...``, ``SiO2: 50.6, B2O3: 13.6``) or a bare formula.
Units: q in 1/Å, I in 1/cm, densities in g/cm^3.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Mapping, Union

import numpy as np

R_E_CM = 2.8179403262e-13      # classical electron radius, CODATA 2018
N_A = 6.02214076e23            # Avogadro constant, CODATA 2018 (exact)

Composition = Union[str, Mapping[str, float]]


def as_composition(comp: Composition) -> Dict[str, float]:
    """``{formula: amount}`` from a mapping, a mixture string or a bare formula."""
    if isinstance(comp, Mapping):
        return {str(k): float(v) for k, v in comp.items() if float(v) != 0}
    from saxs.core.chemistry import parse_mixture_expression

    text = str(comp).strip()
    try:
        parts = parse_mixture_expression(text)
    except ValueError:
        return {text: 1.0}
    out: Dict[str, float] = {}
    for formula, amount in parts:
        out[formula] = out.get(formula, 0.0) + float(amount)
    return out


def element_amounts(comp: Composition) -> Dict[str, float]:
    """Moles of each element per mole of formula units (amounts normalised to a total of 1)."""
    import xraydb

    c = as_composition(comp)
    total = sum(c.values())
    out: Dict[str, float] = {}
    for formula, amount in c.items():
        for el, n in xraydb.chemparse(formula).items():
            out[el] = out.get(el, 0.0) + amount / total * n
    return out


def molar_mass(comp: Composition) -> float:
    """Mass (g) of one mole of formula units of the (normalised) composition."""
    import xraydb

    return sum(n * xraydb.atomic_mass(el) for el, n in element_amounts(comp).items())


def electron_density_cm3(comp: Composition, density_g_cm3: float) -> float:
    """Electrons per cm^3: rho N_A sum(n_i Z_i) / sum(n_i M_i)."""
    import xraydb

    els = element_amounts(comp)
    z_tot = sum(n * xraydb.atomic_number(el) for el, n in els.items())
    m_tot = sum(n * xraydb.atomic_mass(el) for el, n in els.items())
    return float(density_g_cm3) * N_A * z_tot / m_tot


def xray_sld_1e10cm2(rho_e_cm3: float) -> float:
    """X-ray scattering-length density r_e rho_e in units of 1e10 cm^-2 (forward scattering, no anomalous terms)."""
    return R_E_CM * rho_e_cm3 / 1e10


def sld_1e10cm2(comp: Composition, density_g_cm3: float) -> float:
    return xray_sld_1e10cm2(electron_density_cm3(comp, density_g_cm3))


def phase_contrast_1e10cm2(phase: Composition, phase_density: float, sample: Composition, sample_density: float) -> float:
    """SLD(phase) - SLD(sample) in 1e10 cm^-2. The sample is taken as the matrix: valid while the phase is dilute (the
    phase depletes the matrix of its elements and slightly changes the contrast at large volume fractions)."""
    return sld_1e10cm2(phase, phase_density) - sld_1e10cm2(sample, sample_density)


def stoichiometric_ceiling(phase: Composition, phase_density: float, sample: Composition, sample_density: float,
                           ignore: Iterable[str] = ("O",)) -> float:
    """Largest volume fraction of ``phase`` the sample's elements allow: the most formula units of the phase one mole
    of sample formula units can supply (the scarcest element sets it; elements in ``ignore`` are never limiting),
    times the phase molar volume, divided by the sample molar volume."""
    ign = set(ignore)
    have = element_amounts(sample)
    need = {el: n for el, n in element_amounts(phase).items() if el not in ign}
    if not need:
        return float("nan")
    n_max = min(have.get(el, 0.0) / n for el, n in need.items())
    v_phase = molar_mass(phase) / float(phase_density)
    v_sample = molar_mass(sample) / float(sample_density)
    return float(n_max * v_phase / v_sample)


def invariant(func: Callable[[np.ndarray], np.ndarray], q_lo: float = 1e-6, q_hi: float = 100.0, n: int = 20000) -> float:
    """Q = integral q^2 I(q) dq (I in cm^-1, q in 1/Å -> Q in cm^-1 Å^-3), on a log grid. Only for components whose
    integrand converges inside [q_lo, q_hi] (any term decaying faster than q^-3)."""
    from core.np_compat import trapezoid

    x = np.logspace(math.log10(q_lo), math.log10(q_hi), n)
    return float(trapezoid(x ** 3 * func(x), np.log(x)))


def two_phase_product_cm4(Q_cm_invA3: float) -> float:
    """phi (1 - phi) drho^2 in cm^-4 from the invariant: Q = 2 pi^2 phi (1 - phi) drho^2 (Porod)."""
    return float(Q_cm_invA3) * 1e24 / (2 * math.pi ** 2)


def volume_fraction_from_invariant(Q_cm_invA3: float, drho_1e10cm2: float) -> float:
    """Smaller root phi of phi (1 - phi) = X, X = Q / (2 pi^2 drho^2); NaN if X > 1/4 (contrast too small)."""
    X = two_phase_product_cm4(Q_cm_invA3) / (float(drho_1e10cm2) * 1e10) ** 2
    if not np.isfinite(X) or X > 0.25 or X < 0:
        return float("nan")
    return float((1 - math.sqrt(1 - 4 * X)) / 2)


def min_contrast_1e10cm2(Q_cm_invA3: float) -> float:
    """The smallest contrast that can produce the invariant Q (reached at phi = 1/2)."""
    return 2.0 * math.sqrt(max(two_phase_product_cm4(Q_cm_invA3), 0.0)) / 1e10
