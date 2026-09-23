"""Sector-aware reduction of 2D frames from a beamstopless SAXS instrument (Xenocs Xeuss type).

Why this exists. On a beamstopless instrument the direct beam is recorded on the detector. For
absorbing powders in capillaries two things then happen: the beam through the capillary centre is
extinguished, and the part of the beam that passes the capillary next to its walls is refracted
outward by the curved glass and lands as two lobes on the horizontal detector axis (the capillary
being vertical). Their azimuthal average is a maximum at low q that moves with absorption and
mounting and mimics a structural peak. The slit-scattering streaks along both detector axes add to
this. The remedy is geometric: integrate only the four diagonal wedges of the frame, mask the lobe
zone, take the transmission from the recorded direct beam averaged over the beam footprint (not
from a region-of-interest that integrates the lobes), and subtract the empty of the same session.

This module is pure numpy; ``fabio`` is needed only by :func:`read_frame`. It is instrument- and
sample-agnostic: nothing here knows about a particular series or naming scheme.

Typical use::

    red = reduce_frame_pair("sample.edf", "empty.edf", mu_mm=12.3)
    red.curve            # corrected intensity (counts/s/pixel, empty-transmission units), sigma, q
    red.curve_usable     # the same, cropped to the usable range
    red.transmission     # footprint transmission with its heterogeneity diagnostics
    red.write_dat("sample__sectors.dat")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .curve import Curve

try:  # fabio is optional, as for edf_reduction
    import fabio

    HAVE_FABIO = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_FABIO = False


# --------------------------------------------------------------------------------------- geometry
@dataclass
class FrameGeometry:
    """Beam centre (pixels, array axis 1 = ``Center_1``, axis 0 = ``Center_2``), distance (m),
    pixel size (m), wavelength (A) and exposure (s)."""
    cx: float
    cy: float
    dist_m: float
    pixel_m: float
    wavelength_A: float
    exposure_s: float

    @classmethod
    def from_header(cls, h: Dict[str, str], default_wavelength_A: Optional[float] = None) -> "FrameGeometry":
        """``default_wavelength_A`` is used when the header carries no WaveLength (virtual-detector
        exports); without it such a header raises."""
        wl = h.get("WaveLength", h.get("Wavelength"))
        if wl is None:
            if default_wavelength_A is None:
                raise ValueError("EDF header has no WaveLength/Wavelength")
            wavelength_A = float(default_wavelength_A)
        else:
            wavelength_A = float(wl) * 1e10
        return cls(cx=float(h["Center_1"]), cy=float(h["Center_2"]), dist_m=float(h["SampleDistance"]),
                   pixel_m=float(h.get("PSize_1", 7.5e-5)), wavelength_A=wavelength_A,
                   exposure_s=float(h["ExposureTime"]))

    def q_of_radius(self, r_px: np.ndarray) -> np.ndarray:
        """Exact q (1/A) of a pixel radius, from the scattering angle 2theta = atan(r pixel / D)."""
        tth = np.arctan(np.asarray(r_px, float) * self.pixel_m / self.dist_m)
        return 4 * np.pi / self.wavelength_A * np.sin(tth / 2)

    def geometric_correction(self, r_px: np.ndarray) -> np.ndarray:
        """Solid-angle (cos^3) times unpolarized-source polarization factor, to divide by."""
        tth = np.arctan(np.asarray(r_px, float) * self.pixel_m / self.dist_m)
        return np.cos(tth) ** 3 * (1 + np.cos(tth) ** 2) / 2


def read_frame(path: str, default_wavelength_A: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, str], FrameGeometry]:
    """Read an EDF frame with fabio -> (float64 array, header, geometry)."""
    if not HAVE_FABIO:
        raise ImportError("fabio is required to read EDF frames")
    im = fabio.open(str(path))
    h = dict(im.header)
    return im.data.astype(np.float64), h, FrameGeometry.from_header(h, default_wavelength_A)


def beam_centroid(data: np.ndarray, seed_cx: float, seed_cy: float, r_px: float = 25.0, pct: float = 99.0) -> Optional[Tuple[float, float]]:
    """Intensity-weighted centroid of the brightest pixels (top ``100 - pct`` %) within ``r_px`` of a
    seed: the recorded direct beam. Returns None when there is nothing bright there. Header beam
    centres can be off by several pixels from one session to the next; on a beamstopless
    instrument the frame itself carries the true centre."""
    yy, xx = np.indices(data.shape)
    rr = np.hypot(xx - seed_cx, yy - seed_cy)
    m = (rr < r_px) & (data > 0)
    if m.sum() < 20:
        return None
    thr = np.percentile(data[m], pct)
    m &= data >= thr
    return float((data[m] * xx[m]).sum() / data[m].sum()), float((data[m] * yy[m]).sum() / data[m].sum())


def refine_centres(sample: np.ndarray, gs: FrameGeometry, empty: np.ndarray, ge: FrameGeometry,
                   max_shift_px: float = 4.0, bright_fraction: float = 0.01) -> Tuple[FrameGeometry, FrameGeometry, str]:
    """Replace the header centres by the direct-beam centroids: the empty's own; the sample's own when
    its central disc holds more than ``bright_fraction`` of the empty's and lies within ``max_shift_px``
    of the empty's centre, otherwise the empty's centre (dark centre = opaque loading). Returns the two
    geometries and a description of the sample choice."""
    from dataclasses import replace

    ce = beam_centroid(empty, ge.cx, ge.cy)
    if ce is None:
        return gs, ge, "header (no direct beam found in the empty)"
    ge2 = replace(ge, cx=ce[0], cy=ce[1])
    yy, xx = np.indices(sample.shape)
    disc = np.hypot(xx - ce[0], yy - ce[1]) < 5.0
    s_level = float(np.mean(sample[disc & (sample >= 0)])) / gs.exposure_s
    e_level = float(np.mean(empty[disc & (empty >= 0)])) / ge.exposure_s
    cs = beam_centroid(sample, ce[0], ce[1])
    if cs is not None and e_level > 0 and s_level > bright_fraction * e_level and np.hypot(cs[0] - ce[0], cs[1] - ce[1]) < max_shift_px:
        return replace(gs, cx=cs[0], cy=cs[1]), ge2, "own direct beam"
    return replace(gs, cx=ce[0], cy=ce[1]), ge2, "session empty (centre too dark)"


def pixel_maps(shape: Tuple[int, int], cx: float, cy: float):
    """(r, phi180, dx, dy): radius in px, azimuth folded to [0, 180) deg (0 = +x axis), offsets."""
    yy, xx = np.indices(shape)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    phi = np.degrees(np.arctan2(dy, dx)) % 180.0
    return r, phi, dx, dy


# ------------------------------------------------------------------------------------------ masks
def valid_pixel_mask(data: np.ndarray, upper: float = 1e9) -> np.ndarray:
    """Detector gaps and dummies are written as negative values."""
    return (data >= 0) & (data < upper)


def hot_pixel_mask(data: np.ndarray, valid: Optional[np.ndarray] = None, factor: float = 4.0,
                   sqrt_factor: float = 5.0, floor: float = 10.0, size: int = 5) -> np.ndarray:
    """Isolated hot pixels / zingers: value > factor x local median + sqrt_factor sqrt(median) + floor.
    A smooth anisotropic intensity passes; an isolated spike does not."""
    from scipy.ndimage import median_filter

    if valid is None:
        valid = valid_pixel_mask(data)
    med = median_filter(np.where(data < 0, 0.0, data), size=size, mode="nearest")
    return valid & (data > factor * med + sqrt_factor * np.sqrt(np.maximum(med, 0)) + floor)


def flatfield_factor(data: np.ndarray, lo: float = 0.3, hi: float = 0.95) -> float:
    """Frames that were flat-field corrected are floats; the cluster of single-count pixels sits at
    the mean flat-field factor. var(value) = factor x value. Returns 1.0 when no such cluster exists
    (integer counts)."""
    pos = data[(data > lo) & (data < hi)]
    return float(np.mean(pos)) if pos.size > 100 else 1.0


def diagonal_sector_mask(shape: Tuple[int, int], cx: float, cy: float, axis_exclusion_deg: float = 20.0,
                         bar_rect_px: Tuple[float, float] = (40.0, 14.0), rmax_px: Optional[float] = None,
                         valid: Optional[np.ndarray] = None) -> np.ndarray:
    """The four diagonal wedges: more than ``axis_exclusion_deg`` from both detector axes, outside the
    lobe rectangle |dx| <= bar_rect_px[0], |dy| <= bar_rect_px[1], and (optionally) inside rmax."""
    r, phi, dx, dy = pixel_maps(shape, cx, cy)
    d_h = np.abs(((phi - 0.0) + 90) % 180 - 90)          # angular distance to the horizontal axis
    d_v = 90.0 - d_h                                       # to the vertical axis
    m = (d_h > axis_exclusion_deg) & (d_v > axis_exclusion_deg)
    m &= ~((np.abs(dy) <= bar_rect_px[1]) & (np.abs(dx) <= bar_rect_px[0]))
    if rmax_px is not None:
        m &= r < rmax_px
    if valid is not None:
        m &= valid
    return m


def vertical_sector_mask(shape: Tuple[int, int], cx: float, cy: float, half_deg: float = 25.0,
                         rmax_px: Optional[float] = None, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels within ``half_deg`` of the vertical axis (both directions); the direction free of the
    capillary lobes, useful as a cross-check of the diagonal wedges."""
    r, phi, _, _ = pixel_maps(shape, cx, cy)
    m = np.abs(phi - 90.0) <= half_deg
    if rmax_px is not None:
        m &= r < rmax_px
    if valid is not None:
        m &= valid
    return m


# --------------------------------------------------------------------------------------- profiles
@dataclass
class RadialProfile:
    r_px: np.ndarray
    q: np.ndarray
    intensity: np.ndarray        # counts per second per pixel, geometry corrected
    sigma: np.ndarray
    npix: np.ndarray


def radial_profile(data: np.ndarray, geom: FrameGeometry, mask: np.ndarray, rmax_px: int = 650,
                   ff: float = 1.0) -> RadialProfile:
    """Mean of the geometry-corrected values in 1-pixel rings, per second; Poisson sigma of the
    flat-field-corrected values (var = ff x value)."""
    r, _, _, _ = pixel_maps(data.shape, geom.cx, geom.cy)
    ri = np.round(r).astype(int)
    m = mask & (r < rmax_px)
    corr = geom.geometric_correction(r)
    n = np.bincount(ri[m], minlength=rmax_px + 1).astype(float)
    s = np.bincount(ri[m], weights=(data / corr)[m], minlength=rmax_px + 1)
    s_raw = np.bincount(ri[m], weights=data[m], minlength=rmax_px + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan) / geom.exposure_s
        sig = np.where(n > 0, np.sqrt(ff * np.maximum(s_raw, 0)) / np.maximum(n, 1), np.nan) / geom.exposure_s
    rr = np.arange(rmax_px + 1)
    return RadialProfile(r_px=rr, q=geom.q_of_radius(rr), intensity=mean, sigma=sig, npix=n)


def disc_counts(data: np.ndarray, geom: FrameGeometry, r_px: float, valid: np.ndarray,
                vertical_half_deg: Optional[float] = None, ff: float = 1.0) -> Tuple[float, float]:
    """Counts per second inside r < r_px (optionally restricted to the vertical sector) and their
    Poisson uncertainty."""
    r, phi, _, _ = pixel_maps(data.shape, geom.cx, geom.cy)
    m = valid & (r < r_px)
    if vertical_half_deg is not None:
        m &= np.abs(phi - 90.0) <= vertical_half_deg
    s = float(data[m].sum())
    return s / geom.exposure_s, float(np.sqrt(ff * max(s, 0.0))) / geom.exposure_s


# ------------------------------------------------------------------------------------- anisotropy
@dataclass
class SectorAnisotropy:
    """Raw (not geometry-corrected) ring means in counts/s/pixel along the horizontal axis, the
    vertical axis, the diagonals and the full azimuth, plus the lobe and streak diagnostics used by
    the frame inventory. ``lobe_q`` is the q of a local maximum of the horizontal profile that the
    vertical profile does not show (the capillary-wall lobes), None when there is none."""
    r_px: np.ndarray
    q: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    diagonal: np.ndarray
    full: np.ndarray
    lobe_q: Optional[float]
    lobe_prominence: float
    lobe_ratio: float           # horizontal / vertical in the band around lobe_q (or the default band)
    max_q_horizontal: Optional[float]
    max_q_vertical: Optional[float]
    max_q_full: Optional[float]
    streak_h: float             # horizontal / diagonal level in the streak band (1 = no streak)
    streak_v: float


def _local_max_q(prof: np.ndarray, q: np.ndarray, q_window: Tuple[float, float], prominence: float) -> Tuple[Optional[float], float]:
    from scipy.signal import find_peaks

    sel = (q >= q_window[0]) & (q <= q_window[1]) & np.isfinite(prof) & (prof > 0)
    if sel.sum() < 5:
        return None, 0.0
    lp = np.log10(prof[sel])
    pk, pr = find_peaks(lp, prominence=prominence)
    if not len(pk):
        return None, 0.0
    b = int(np.argmax(pr["prominences"]))
    return float(q[sel][pk[b]]), float(pr["prominences"][b])


def sector_anisotropy(data: np.ndarray, geom: FrameGeometry, valid: Optional[np.ndarray] = None,
                      half_deg: float = 25.0, rmax_px: int = 220, q_window: Tuple[float, float] = (2e-3, 1.5e-2),
                      prominence: float = 0.05, streak_band_px: Tuple[int, int] = (30, 60),
                      default_band_q: float = 6.5e-3, lobe_min_ratio: float = 2.0) -> SectorAnisotropy:
    """Horizontal / vertical / diagonal / full radial profiles about ``geom``'s centre and the lobe
    test: a maximum of the horizontal profile inside ``q_window`` (prominence in log10 units) where the
    horizontal level exceeds the vertical one by more than ``lobe_min_ratio`` (an isotropic ring gives
    the same maximum in both). Streak levels compare each axis with the diagonals in ``streak_band_px``."""
    if valid is None:
        valid = valid_pixel_mask(data)
    r, phi, _, _ = pixel_maps(data.shape, geom.cx, geom.cy)
    ri = np.round(r).astype(int)
    base = valid & (r < rmax_px)
    d_h = np.abs(((phi - 0.0) + 90) % 180 - 90)
    masks = {"full": base, "horizontal": base & (d_h <= half_deg), "vertical": base & (np.abs(phi - 90.0) <= half_deg),
             "diagonal": base & (np.abs(d_h - 45.0) <= half_deg)}
    prof: Dict[str, np.ndarray] = {}
    for key, m in masks.items():
        n = np.bincount(ri[m], minlength=rmax_px + 1)
        s = np.bincount(ri[m], weights=data[m], minlength=rmax_px + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof[key] = np.where(n > 0, s / np.maximum(n, 1), np.nan) / geom.exposure_s
    rr = np.arange(rmax_px + 1)
    q = geom.q_of_radius(rr)
    qh, ph = _local_max_q(prof["horizontal"], q, q_window, prominence)
    qv, pv = _local_max_q(prof["vertical"], q, q_window, prominence)
    qa, _ = _local_max_q(prof["full"], q, q_window, prominence)
    def hv_ratio(q0: float) -> float:
        band = (q >= 0.8 * q0) & (q <= 1.2 * q0)
        with np.errstate(invalid="ignore", divide="ignore"):
            hm, vm = np.nanmean(prof["horizontal"][band]), np.nanmean(prof["vertical"][band])
        return float(hm / vm) if vm > 0 else float("nan")

    lobe_q = None
    if qh is not None and hv_ratio(qh) > lobe_min_ratio:
        lobe_q = qh
    q0 = lobe_q if lobe_q is not None else (qa if qa is not None else default_band_q)
    lobe_ratio = hv_ratio(q0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b0, b1 = streak_band_px
        dg = np.nanmedian(prof["diagonal"][b0:b1])
        streak_h = float(np.nanmedian(prof["horizontal"][b0:b1]) / dg) if dg > 0 else float("nan")
        streak_v = float(np.nanmedian(prof["vertical"][b0:b1]) / dg) if dg > 0 else float("nan")
    return SectorAnisotropy(r_px=rr, q=q, horizontal=prof["horizontal"], vertical=prof["vertical"], diagonal=prof["diagonal"],
                            full=prof["full"], lobe_q=lobe_q, lobe_prominence=ph, lobe_ratio=lobe_ratio,
                            max_q_horizontal=qh, max_q_vertical=qv, max_q_full=qa, streak_h=streak_h, streak_v=streak_v)


# ----------------------------------------------------------------------------------- transmission
@dataclass
class Transmission:
    T: float                 # footprint-averaged (r < footprint_r_px, vertical +-half), lobes excluded
    dT: float
    T_disc4: float           # central r < 4 px
    T_disc8: float           # r < 8 px
    footprint_r_px: float
    vertical_half_deg: float

    @property
    def heterogeneity(self) -> float:
        """T_disc4 / T: 1 for a uniform powder column; far from 1 = uneven loading."""
        return self.T_disc4 / self.T if self.T > 0 else float("nan")


def footprint_transmission(sample: np.ndarray, empty: np.ndarray, gs: FrameGeometry, ge: FrameGeometry,
                           valid_s: np.ndarray, valid_e: np.ndarray, footprint_r_px: float = 16.0,
                           vertical_half_deg: float = 30.0, ff_s: float = 1.0, ff_e: float = 1.0) -> Transmission:
    """Sample/empty ratio of the recorded direct beam averaged over the beam footprint, in the vertical
    sector so that the capillary lobes (on the horizontal axis) cannot enter."""
    dS, dSe = disc_counts(sample, gs, footprint_r_px, valid_s, vertical_half_deg, ff_s)
    dE, dEe = disc_counts(empty, ge, footprint_r_px, valid_e, vertical_half_deg, ff_e)
    T = dS / dE if dE > 0 else float("nan")
    dT = T * np.sqrt((dSe / dS) ** 2 + (dEe / dE) ** 2) if dS > 0 and dE > 0 else float("nan")
    t4 = disc_counts(sample, gs, 4.0, valid_s, None, ff_s)[0] / max(disc_counts(empty, ge, 4.0, valid_e, None, ff_e)[0], 1e-300)
    t8 = disc_counts(sample, gs, 8.0, valid_s, None, ff_s)[0] / max(disc_counts(empty, ge, 8.0, valid_e, None, ff_e)[0], 1e-300)
    return Transmission(T=T, dT=dT, T_disc4=t4, T_disc8=t8, footprint_r_px=footprint_r_px, vertical_half_deg=vertical_half_deg)


# ---------------------------------------------------------------------------------- usable range
def usable_range(ratio: np.ndarray, I: np.ndarray, sigma: np.ndarray, r_px: np.ndarray, min_ratio: float = 1.25,
                 min_ratio_footprint: float = 5.0, r_footprint_px: float = 30.0, nsigma: float = 3.0,
                 min_run: int = 8) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """Bins where the sample exceeds the scaled empty by ``min_ratio`` (``min_ratio_footprint`` inside
    the beam footprint) and the corrected signal exceeds ``nsigma`` sigma; the longest run (single-bin
    gaps allowed) of at least ``min_run`` bins is kept. Returns (flag, first index, last index)."""
    need = np.where(r_px <= r_footprint_px, min_ratio_footprint, min_ratio)
    ok = np.isfinite(ratio) & np.isfinite(I) & (ratio >= need) & (I >= nsigma * sigma)
    best, start, length, gap = None, None, 0, 0
    for i, v in enumerate(ok):
        if v:
            if start is None:
                start = i
            length = i - start + 1
            gap = 0
        else:
            gap += 1
            if gap > 1 and start is not None:
                if length >= min_run and (best is None or length > best[1]):
                    best = (start, length)
                start, length = None, 0
    if start is not None and length >= min_run and (best is None or length > best[1]):
        best = (start, length)
    flag = np.zeros_like(ok)
    if best is None:
        return flag, None, None
    i0, i1 = best[0], best[0] + best[1] - 1
    while i1 > i0 and not ok[i1]:
        i1 -= 1
    flag[i0:i1 + 1] = ok[i0:i1 + 1]
    return flag, i0, i1


# ------------------------------------------------------------------------------------- reduction
@dataclass
class SectorReduction:
    curve: Curve                     # corrected intensity, all rings with pixels (usable flag in metadata)
    curve_usable: Curve              # cropped to the usable range (empty Curve fields if none)
    transmission: Transmission
    sample_profile: RadialProfile
    empty_profile: RadialProfile
    ratio: np.ndarray                # S / (T E)
    usable: np.ndarray
    q_min_use: Optional[float]
    q_max_use: Optional[float]
    t_eff_mm: Optional[float]
    settings: Dict[str, object] = field(default_factory=dict)

    def write_dat(self, path: str) -> None:
        """Tab-separated file the PRISM loader reads back (columns named so that ``corrected`` and
        ``sigma_corrected`` are picked up)."""
        c = self.curve
        md = c.metadata
        lines = [f"# {k}: {v}" for k, v in md.items() if not k.startswith("_")]     # arrays go to the columns
        lines.append("# columns: q_A^-1  corrected  sigma_corrected  sample_cps_px  empty_cps_px  ratio_S_over_TE  n_pix  usable")
        lines.append("q_A^-1\tcorrected\tsigma_corrected\tsample_cps_px\tempty_cps_px\tratio_S_over_TE\tn_pix\tusable")
        S, E, n, u = md["_sample_cps_px"], md["_empty_cps_px"], md["_n_pix"], md["_usable"]
        for i in range(len(c.q)):
            lines.append(f"{c.q[i]:.6e}\t{c.intensity[i]:.6e}\t{c.sigma[i]:.6e}\t{S[i]:.6e}\t{E[i]:.6e}\t{md['_ratio'][i]:.5e}\t{int(n[i])}\t{int(u[i])}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def reduce_arrays(sample: np.ndarray, gs: FrameGeometry, empty: np.ndarray, ge: FrameGeometry,
                  mu_mm: Optional[float] = None, axis_exclusion_deg: float = 20.0,
                  bar_rect_px: Tuple[float, float] = (40.0, 14.0), rmax_px: int = 650,
                  footprint_r_px: float = 16.0, vertical_half_deg: float = 30.0,
                  name: str = "sample", sample_path: str = "", empty_path: str = "",
                  refine_centre: bool = True, max_shift_px: float = 4.0, bright_fraction: float = 0.01,
                  min_ratio: float = 1.25, min_ratio_footprint: float = 5.0, r_footprint_px: float = 30.0,
                  nsigma: float = 3.0, min_run: int = 8) -> SectorReduction:
    """Sector-integrated, transmission-scaled empty subtraction of two frames already in memory.
    With ``refine_centre`` (default) the beam centres come from the recorded direct beam
    (:func:`refine_centres`); the header centres are kept in the metadata."""
    cx_h, cy_h = gs.cx, gs.cy
    ge_cx_h, ge_cy_h = ge.cx, ge.cy
    centre_note = "header"
    if refine_centre:
        gs, ge, centre_note = refine_centres(sample, gs, empty, ge, max_shift_px=max_shift_px, bright_fraction=bright_fraction)
    hot_s, hot_e = hot_pixel_mask(sample), hot_pixel_mask(empty)
    valid_s = valid_pixel_mask(sample) & ~hot_s
    valid_e = valid_pixel_mask(empty) & ~hot_e
    ff_s, ff_e = flatfield_factor(sample), flatfield_factor(empty)
    ms = diagonal_sector_mask(sample.shape, gs.cx, gs.cy, axis_exclusion_deg, bar_rect_px, rmax_px, valid_s)
    me = diagonal_sector_mask(empty.shape, ge.cx, ge.cy, axis_exclusion_deg, bar_rect_px, rmax_px, valid_e)
    ps = radial_profile(sample, gs, ms, rmax_px, ff_s)
    pe = radial_profile(empty, ge, me, rmax_px, ff_e)
    tr = footprint_transmission(sample, empty, gs, ge, valid_s, valid_e, footprint_r_px, vertical_half_deg, ff_s, ff_e)
    T = tr.T
    with np.errstate(invalid="ignore", divide="ignore"):
        I = ps.intensity / T - pe.intensity
        sig = np.sqrt((ps.sigma / T) ** 2 + pe.sigma ** 2 + ((ps.intensity / T) * (tr.dT / T)) ** 2)
        ratio = ps.intensity / (T * pe.intensity)
    flag, i0, i1 = usable_range(ratio, I, sig, ps.r_px, min_ratio=min_ratio, min_ratio_footprint=min_ratio_footprint,
                                r_footprint_px=r_footprint_px, nsigma=nsigma, min_run=min_run)
    t_eff = (-np.log(T) / mu_mm) if (mu_mm and mu_mm > 0 and 0 < T < 1) else None
    keep = (ps.npix > 0) & (pe.npix > 0) & np.isfinite(I)
    keep[0] = False
    md: Dict[str, object] = dict(
        reduction="edf_sectors.reduce_arrays", sample_frame=sample_path, empty_frame=empty_path,
        sample_exposure_s=gs.exposure_s, empty_exposure_s=ge.exposure_s, distance_m=gs.dist_m,
        wavelength_A=gs.wavelength_A, centre_px=(round(gs.cx, 2), round(gs.cy, 2)), centre_source=centre_note,
        centre_header_px=(cx_h, cy_h), centre_offset_from_header_px=(round(gs.cx - cx_h, 1), round(gs.cy - cy_h, 1)),
        empty_centre_px=(round(ge.cx, 2), round(ge.cy, 2)),
        empty_centre_offset_from_header_px=(round(ge.cx - ge_cx_h, 1), round(ge.cy - ge_cy_h, 1)), pixel_m=gs.pixel_m,
        n_hot_sample=int(hot_s.sum()), n_hot_empty=int(hot_e.sum()),
        sector=f"diagonal wedges, > {axis_exclusion_deg} deg from both axes, bar zone |dx|<={bar_rect_px[0]} |dy|<={bar_rect_px[1]} px excluded",
        transmission_T=T, transmission_dT=tr.dT, transmission_T_disc4=tr.T_disc4, transmission_T_disc8=tr.T_disc8,
        heterogeneity_T4_over_T=tr.heterogeneity, flatfield_factor_sample=ff_s, flatfield_factor_empty=ff_e,
        mu_mm=mu_mm, t_eff_mm=t_eff, mu_t_eff=(-np.log(T) if T > 0 else None),
        q_min_use=(float(ps.q[i0]) if i0 is not None else None), q_max_use=(float(ps.q[i1]) if i1 is not None else None),
        units="counts/s/pixel, empty-transmission units: (S/t_s - T E/t_e)/T",
        _sample_cps_px=ps.intensity[keep], _empty_cps_px=pe.intensity[keep], _ratio=ratio[keep], _n_pix=ps.npix[keep], _usable=flag[keep])
    curve = Curve(q=ps.q[keep], intensity=I[keep], sigma=sig[keep], name=name, path=sample_path, transmission=T,
                  file_role="corrected", metadata=md)
    curve.record("edf_sectors", transmission=T, sector=md["sector"], hot_pixels_sample=int(hot_s.sum()))
    u = flag[keep]
    if u.any():
        cu = Curve(q=ps.q[keep][u], intensity=I[keep][u], sigma=sig[keep][u], name=name + " (usable)", path=sample_path,
                   transmission=T, file_role="corrected", metadata={k: v for k, v in md.items() if not k.startswith("_")})
    else:
        cu = Curve(q=np.array([]), intensity=np.array([]), sigma=np.array([]), name=name + " (usable)", path=sample_path)
    return SectorReduction(curve=curve, curve_usable=cu, transmission=tr, sample_profile=ps, empty_profile=pe,
                           ratio=ratio, usable=flag, q_min_use=md["q_min_use"], q_max_use=md["q_max_use"], t_eff_mm=t_eff,
                           settings=dict(axis_exclusion_deg=axis_exclusion_deg, bar_rect_px=bar_rect_px, rmax_px=rmax_px,
                                         footprint_r_px=footprint_r_px, vertical_half_deg=vertical_half_deg,
                                         refine_centre=refine_centre, max_shift_px=max_shift_px, bright_fraction=bright_fraction,
                                         min_ratio=min_ratio, min_ratio_footprint=min_ratio_footprint,
                                         r_footprint_px=r_footprint_px, nsigma=nsigma, min_run=min_run))


def reduce_frame_pair(sample_path: str, empty_path: str, mu_mm: Optional[float] = None, **kw) -> SectorReduction:
    """Read two EDF frames (fabio) and reduce them with :func:`reduce_arrays`."""
    s, hs, gs = read_frame(sample_path)
    e, he, ge = read_frame(empty_path)
    name = hs.get("Comment", str(sample_path)).split(" - ")[0].strip() or str(sample_path)
    return reduce_arrays(s, gs, e, ge, mu_mm=mu_mm, name=name, sample_path=str(sample_path), empty_path=str(empty_path), **kw)
