"""SAXS model library: the physically interpretable scattering contributions the fit engine sums.

Every component is a small class wrapping a vectorised I(q; parameters) on an absolute or relative scale, with
default values, bounds, units and documentation for each parameter, and (where meaningful) derived quantities:

    flat_background   I = C                                   high-q level (density + composition fluctuations)
    power_law         I = B q^-p                              grain / particle outer surfaces (Porod p = 4)
    lorentz_oz        I = A / (1 + q^2 xi^2)                  Ornstein-Zernike concentration fluctuations
    guinier           I = G exp(-q^2 Rg^2 / 3)
    guinier_porod     Hammouda (2010) Guinier -> power-law crossover
    beaucage_unified  one unified level, free Porod prefactor (Beaucage 1995)
    beaucage_tied     one unified level, Porod prefactor tied to G (Beaucage 1996 polydispersity relation)
    dab               Debye-Anderson-Brumberger random two-phase medium, correlation length xi
    teubner_strey     Teubner-Strey quasi-periodic two-phase medium, spacing d and correlation length xi
    sphere_schulz     polydisperse spheres on an absolute scale, optional Percus-Yevick structure factor

The order of preference for structural interpretation (Debye / Teubner-Strey as correlation-function models,
unified levels where there is no finite-q peak, spheres only for particulate morphologies, no empirical shapes) is
described in docs/SAXS_FITTING.md. Units: q in 1/Å, lengths in Å.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

import numpy as np
from scipy.special import erf


@dataclass
class Param:
    name: str
    value: float
    min: float
    max: float
    vary: bool = True
    unit: str = ""
    doc: str = ""


class Component:
    """One scattering contribution. Subclasses define ``name``, ``title``, ``reference``, ``params`` and ``eval``."""
    name: str = ""
    title: str = ""
    reference: str = ""

    def params(self) -> List[Param]:
        raise NotImplementedError

    def eval(self, q: np.ndarray, **kw: float) -> np.ndarray:
        raise NotImplementedError

    def derived(self, **kw: float) -> Dict[str, float]:
        return {}


# ------------------------------------------------------------------------------------------- background terms
class FlatBackground(Component):
    name = "flat_background"
    title = "flat level C (q-independent)"
    reference = "density and composition fluctuations of a homogeneous medium"

    def params(self) -> List[Param]:
        return [Param("C", 0.0, 0.0, np.inf, unit="I units", doc="Flat level.")]

    def eval(self, q: np.ndarray, C: float = 0.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.full_like(q, float(C))


class PowerLaw(Component):
    name = "power_law"
    title = "power law B q^-p"
    reference = "Porod (1951): p = 4 for sharp outer surfaces (e.g. powder grains)"

    def params(self) -> List[Param]:
        return [
            Param("B", 1.0, 0.0, np.inf, unit="I units x A^-p", doc="Power-law prefactor."),
            Param("p", 4.0, 1.0, 4.5, doc="Exponent: 4 sharp interfaces, 3-4 rough, < 3 mass-fractal-like."),
        ]

    def eval(self, q: np.ndarray, B: float = 1.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(B) * np.power(np.clip(q, 1e-300, None), -float(p))


class LorentzOZ(Component):
    name = "lorentz_oz"
    title = "Ornstein-Zernike A / (1 + q^2 xi^2)"
    reference = "Ornstein & Zernike (1914); concentration fluctuations with correlation length xi"

    def params(self) -> List[Param]:
        return [
            Param("A", 1.0, 0.0, np.inf, unit="I units", doc="Forward amplitude I(0)."),
            Param("xiL", 5.0, 0.5, 1e5, unit="Å", doc="Correlation length of the fluctuations."),
        ]

    def eval(self, q: np.ndarray, A: float = 1.0, xiL: float = 5.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(A) / (1.0 + (q ** 2) * (float(xiL) ** 2))


# ------------------------------------------------------------------------------------------- levels
class Guinier(Component):
    name = "guinier"
    title = "Guinier G exp(-q^2 Rg^2 / 3)"
    reference = "Guinier (1939)"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Forward scattering I(0)."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
        ]

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(G) * np.exp(-(q ** 2) * (float(Rg) ** 2) / 3.0)


class GuinierPorod(Component):
    name = "guinier_porod"
    title = "Guinier-Porod crossover"
    reference = "Hammouda (2010), J. Appl. Cryst. 43, 716 (s = 0)"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Forward scattering I(0)."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
            Param("p", 4.0, 1.0, 4.5, doc="High-q power-law exponent."),
        ]

    @staticmethod
    def _q1_D(G: float, Rg: float, p: float) -> Tuple[float, float]:
        Rg = max(float(Rg), 1e-12)
        p = float(p)
        q1 = (1.0 / Rg) * math.sqrt(3.0 * p / 2.0)
        D = float(G) * math.exp(-(q1 ** 2) * (Rg ** 2) / 3.0) * (q1 ** p)
        return q1, D

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q1, D = self._q1_D(G, Rg, p)
        out = np.empty_like(q)
        lo = q <= q1
        out[lo] = float(G) * np.exp(-(q[lo] ** 2) * (float(Rg) ** 2) / 3.0)
        hi = ~lo
        out[hi] = D / np.power(np.clip(q[hi], 1e-300, None), float(p))
        return out

    def derived(self, G: float = 1.0, Rg: float = 100.0, p: float = 4.0, **_) -> Dict[str, float]:
        q1, D = self._q1_D(G, Rg, p)
        return {"q1": q1, "D": D}


class BeaucageUnified(Component):
    name = "beaucage_unified"
    title = "unified level, free Porod prefactor"
    reference = "Beaucage (1995), J. Appl. Cryst. 28, 717"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Guinier prefactor."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
            Param("B", 1.0, 0.0, np.inf, doc="Porod-regime prefactor."),
            Param("p", 4.0, 1.0, 4.5, doc="Porod exponent."),
        ]

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0, B: float = 1.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        Rg, p = float(Rg), float(p)
        q_safe = np.where(q > 0, q, 1e-300)
        guinier_term = float(G) * np.exp(-q ** 2 * Rg ** 2 / 3)
        porod_term = float(B) * erf(q_safe * Rg / np.sqrt(6)) ** (3 * p) * q_safe ** (-p)
        return guinier_term + porod_term


def beaucage_polydisperse_prefactor(G: float, Rg: float, d: float) -> float:
    """Beaucage (1996) Porod prefactor of a polydisperse level:
    B = G d / Rg^d (6 d^2 / ((2 + d)(2 + 2 d)))^(d/2) Gamma(d/2)."""
    from scipy.special import gamma as Gamma

    G, Rg, d = float(G), max(float(Rg), 1e-12), float(d)
    return G * d / Rg ** d * (6 * d ** 2 / ((2 + d) * (2 + 2 * d))) ** (d / 2) * Gamma(d / 2)


class BeaucageTied(Component):
    """A unified level whose Porod prefactor is B = pf x B_poly(G, Rg, d): a population with a Guinier knee or
    nothing - it cannot degenerate into a free power law."""
    name = "beaucage_tied"
    title = "unified level, Porod prefactor tied to G"
    reference = "Beaucage (1995) form with the Beaucage (1996) polydispersity prefactor"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Guinier prefactor."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
            Param("d", 3.5, 1.0, 4.5, doc="Power-law exponent of the level."),
            Param("pf", 0.5, 0.1, 1.5, doc="Polydispersity factor: 0.16 monodisperse spheres, 1 broad distribution."),
        ]

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0, d: float = 3.5, pf: float = 0.5) -> np.ndarray:
        B = float(pf) * beaucage_polydisperse_prefactor(G, Rg, d)
        return BeaucageUnified().eval(q, G=G, Rg=Rg, B=B, p=d)

    def derived(self, G: float = 1.0, Rg: float = 100.0, d: float = 3.5, pf: float = 0.5, **_) -> Dict[str, float]:
        return {"B": float(pf) * beaucage_polydisperse_prefactor(G, Rg, d), "D_eq": 2.0 * float(Rg) * math.sqrt(5.0 / 3.0)}


# ------------------------------------------------------------------------------------------- two-phase media
class Dab(Component):
    name = "dab"
    title = "Debye (DAB) random two-phase medium"
    reference = "Debye, Anderson & Brumberger (1957), J. Appl. Phys. 28, 679"

    def params(self) -> List[Param]:
        return [
            Param("A", 1.0, 0.0, np.inf, doc="Forward amplitude I(0)."),
            Param("xi", 100.0, 1.0, 1e5, unit="Å", doc="Correlation length (no preferred spacing)."),
        ]

    def eval(self, q: np.ndarray, A: float = 1.0, xi: float = 100.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(A) / (1.0 + (q ** 2) * (float(xi) ** 2)) ** 2


def ts_classic_from_physical(d: float, xi: float) -> Tuple[float, float, float]:
    """(d, xi) -> classic Teubner-Strey coefficients (a2, c1, c2 = 1)."""
    kappa = 1.0 / float(xi)
    k = 2.0 * math.pi / float(d)
    a2 = (k ** 2 + kappa ** 2) ** 2
    c1 = -2.0 * (k ** 2 - kappa ** 2)
    return float(a2), float(c1), 1.0


def ts_physical_from_classic(a2: float, c1: float, c2: float = 1.0) -> Tuple[float, float]:
    """Classic (a2, c1, c2) -> (d, xi), exact inverse of ts_classic_from_physical when c2 = 1."""
    sqrt_a2c2 = math.sqrt(float(a2) / float(c2))
    inv_xi2 = sqrt_a2c2 / 2.0 + float(c1) / (4.0 * float(c2))
    inv_k2 = sqrt_a2c2 / 2.0 - float(c1) / (4.0 * float(c2))
    xi = 1.0 / math.sqrt(max(inv_xi2, 1e-300))
    d = 2.0 * math.pi / math.sqrt(max(inv_k2, 1e-300))
    return float(d), float(xi)


class TeubnerStrey(Component):
    name = "teubner_strey"
    title = "Teubner-Strey quasi-periodic two-phase medium"
    reference = "Teubner & Strey (1987), J. Chem. Phys. 87, 3195"

    def params(self) -> List[Param]:
        return [
            Param("S", 1.0, 0.0, np.inf, doc="Peak height: I(q_max) = S when a peak exists."),
            Param("d", 1000.0, 10.0, 1e5, unit="Å", doc="Repeat distance (characteristic spacing)."),
            Param("xi", 3000.0, 5.0, 1e5, unit="Å", doc="Correlation length (persistence of the order)."),
        ]

    @staticmethod
    def _kkappa(d: float, xi: float) -> Tuple[float, float]:
        return 2.0 * math.pi / float(d), 1.0 / float(xi)

    def eval(self, q: np.ndarray, S: float = 1.0, d: float = 1000.0, xi: float = 3000.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        k, kappa = self._kkappa(d, xi)
        Dq = (q ** 2 - (k ** 2 - kappa ** 2)) ** 2 + 4.0 * (k ** 2) * (kappa ** 2)
        return float(S) * 4.0 * (k ** 2) * (kappa ** 2) / np.clip(Dq, 1e-300, None)

    def derived(self, S: float = 1.0, d: float = 1000.0, xi: float = 3000.0, **_) -> Dict[str, float]:
        """q_max (NaN without a peak), classic coefficients, amphiphilicity fa = c1 / sqrt(4 a2 c2) (-1 lamellar,
        0 disorder line, +1 Lifshitz line)."""
        k, kappa = self._kkappa(d, xi)
        a2, c1, c2 = ts_classic_from_physical(d, xi)
        disc = k ** 2 - kappa ** 2
        q_max = math.sqrt(disc) if disc > 0 else float("nan")
        fa = c1 / math.sqrt(4.0 * a2 * c2)
        return {"q_max": q_max, "a2": a2, "c1": c1, "c2": c2, "fa": fa, "has_peak": bool(disc > 0)}


# ------------------------------------------------------------------------------------------- particles
_N_SCHULZ = 64


def sphere_amplitude(x: np.ndarray) -> np.ndarray:
    """F(x) = 3 (sin x - x cos x) / x^3, with its series below x = 1e-3 (F -> 1)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    f = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3
    return np.where(small, 1.0 - x ** 2 / 10.0, f)


def schulz_radii(R_mean: float, sig_rel: float, n: int = _N_SCHULZ):
    """Radii and weights of the Schulz (gamma) number distribution with mean R_mean and relative standard deviation
    sig_rel, by generalised Gauss-Laguerre quadrature (weight x^(a-1) e^-x, a = 1/sig_rel^2) computed with the
    Golub-Welsch eigenvalue method (no overflow for narrow distributions): every moment <R^k> up to k = 2n - 1 is
    integrated exactly, which matters because the forward intensity is carried by <R^6>/<R^3>.
    Returns (radii, weights) with weights summing to 1."""
    a = 1.0 / max(float(sig_rel), 1e-3) ** 2
    alpha = a - 1.0
    k = np.arange(int(n), dtype=float)
    diag = 2.0 * k + alpha + 1.0
    off = np.sqrt(k[1:] * (k[1:] + alpha))
    x, vec = np.linalg.eigh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    w = vec[0, :] ** 2
    w = w / w.sum()
    return x * float(R_mean) / a, w


def schulz_moment(R_mean: float, sig_rel: float, n: int) -> float:
    """<R^n> of the Schulz distribution (exact): R_mean^n Gamma(a + n) / (a^n Gamma(a)), a = 1/sig_rel^2."""
    from scipy.special import gammaln
    a = 1.0 / max(float(sig_rel), 1e-3) ** 2
    return float(R_mean) ** n * math.exp(gammaln(a + n) - gammaln(a) - n * math.log(a))


def percus_yevick_sq(q: np.ndarray, R: float, phi_hs: float) -> np.ndarray:
    """Hard-sphere structure factor, Percus-Yevick closure (radius R, hard-sphere volume fraction phi_hs)."""
    q = np.asarray(q, dtype=float)
    if phi_hs <= 0:
        return np.ones_like(q)
    eta = float(phi_hs)
    a = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    b = -6 * eta * (1 + eta / 2) ** 2 / (1 - eta) ** 4
    c = eta * a / 2
    x = 2 * q * float(R)
    x = np.where(x < 0.05, 0.05, x)              # below: cancellation; S is flat to < 1e-3 there
    s, co = np.sin(x), np.cos(x)
    g = (a * (s - x * co) / x ** 2
         + b * (2 * x * s + (2 - x ** 2) * co - 2) / x ** 3
         + c * (-x ** 4 * co + 4 * ((3 * x ** 2 - 6) * co + (x ** 3 - 6 * x) * s + 6)) / x ** 5)
    return 1.0 / (1.0 + 24 * eta * g / x)


class SphereSchulz(Component):
    """Polydisperse spheres, absolute scale: I(q) = phi drho^2 <V^2 P(qR)> / <V> (x S_PY(q)). With drho fixed from
    compositions the amplitude is the volume fraction; with drho = 1 it is the product phi drho^2."""
    name = "sphere_schulz"
    title = "polydisperse spheres (Schulz), absolute scale"
    reference = ("Kotlarchyk & Chen (1983), J. Chem. Phys. 79, 2461; Percus-Yevick S(q): Kinning & Thomas (1984), "
                 "Macromolecules 17, 1712")

    def params(self) -> List[Param]:
        return [
            Param("phi", 0.01, 0.0, 0.74, doc="Volume fraction of the spheres (absolute when drho is fixed)."),
            Param("R", 100.0, 5.0, 5000.0, unit="Å", doc="Number-mean radius of the Schulz distribution."),
            Param("sig", 0.3, 0.05, 0.7, doc=(
                "Relative standard deviation of the radius distribution (0.7 = the broadest distribution for which the "
                "component is accurate to ~3 %; a fit wanting more is not describing a particle population).")),
            Param("drho", 1.0, 0.0, 1e3, vary=False, unit="1e10 cm^-2",
                  doc="X-ray SLD contrast sphere - matrix (r_e x delta electron density); fix it from compositions."),
            Param("phi_hs", 0.0, 0.0, 0.5, vary=False,
                  doc="Hard-sphere volume fraction of the Percus-Yevick S(q) (0 = no interference)."),
        ]

    def eval(self, q: np.ndarray, phi: float = 0.01, R: float = 100.0, sig: float = 0.3,
             drho: float = 1.0, phi_hs: float = 0.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        shape = q.shape
        q = q.ravel()
        radii, wts = schulz_radii(R, sig)                             # number-weighted quadrature nodes
        vol = 4.0 / 3.0 * math.pi * radii ** 3                        # Å^3
        x = q * float(R)
        # Deep in the Porod regime a finite quadrature aliases (onset near q R_mean ~ 6 / sig, measured against a dense
        # 8000-radius average); there the polydisperse average is the Porod law 2 pi (S/V) q^-4 with S/V = 3 <R^2>/<R^3>
        # (exact Schulz moments). Smooth hand-over from q R_mean = 2.5/sig over a third of that: <= 2.5 % for sig <= 0.6.
        x0 = 2.5 / max(float(sig), 1e-3)
        w = np.clip((x - x0) / (x0 / 3.0), 0.0, 1.0)
        p_avg = np.empty_like(q)
        grid = w < 1
        if grid.any():
            F = sphere_amplitude(q[grid].reshape(-1, 1) * radii.reshape(1, -1))
            p_avg[grid] = (F ** 2 @ (wts * vol ** 2)) / np.sum(wts * vol)       # <V^2 P> / <V>, Å^3
        sv = 3.0 * schulz_moment(R, sig, 2) / schulz_moment(R, sig, 3)
        porod = 2.0 * math.pi * sv * q ** -4.0
        p_avg = np.where(grid, (1 - w) * np.where(grid, p_avg, 0.0) + w * porod, porod)
        # phi (1e10 cm^-2)^2 Å^3 = phi x 1e20 cm^-4 x 1e-24 cm^3 = 1e-4 cm^-1
        out = float(phi) * float(drho) ** 2 * p_avg * 1e-4
        if phi_hs > 0:
            out = out * percus_yevick_sq(q, float(R), float(phi_hs))
        return out.reshape(shape)

    def derived(self, phi: float = 0.01, R: float = 100.0, sig: float = 0.3, **_) -> Dict[str, float]:
        """Diameters in Å: number mean, volume-weighted mean <D^4>/<D^3>, the Scherrer-comparable volume-weighted
        column length L_vol = (3/4) <D^4>/<D^3> (spheres; Langford & Wilson 1978, J. Appl. Cryst. 11, 102), and the
        ensemble radius of gyration sqrt(3/5 <R^8>/<R^6>)."""
        m = lambda n: schulz_moment(R, sig, n)        # noqa: E731
        d43 = 2.0 * m(4) / m(3)
        return {"D_number": 2.0 * float(R), "D_vol_weighted": d43, "L_vol_scherrer": 0.75 * d43,
                "Rg_ensemble": math.sqrt(0.6 * m(8) / m(6))}


COMPONENTS: Dict[str, Type[Component]] = {
    cls.name: cls for cls in (
        FlatBackground, PowerLaw, LorentzOZ, Guinier, GuinierPorod, BeaucageUnified, BeaucageTied, Dab,
        TeubnerStrey, SphereSchulz,
    )
}
