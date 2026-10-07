"""Synthetic 2D frames and curves shared by the SAXS tests (frames, sector batch, pipeline, UI).

Frames are small (400 x 380) numpy arrays with a Gaussian direct beam, an optional isotropic ring,
optional capillary-wall lobes on the horizontal axis and an optional horizontal streak; the header
centre can be offset from the true beam so centre refinement is exercised. Written with fabio as EDF
into a folder tree that mimics a campaign (sessions with their own empty).

Synthetic curves for the physical fits are generated in fit_synthetic.py.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

SHAPE = (400, 380)
TRUE_CX, TRUE_CY = 190.0, 200.0
DIST_M, PIXEL_M, WAVELENGTH_M = 0.9, 7.5e-5, 1.541891e-10
COUNT_CUTOFF = 1.5e6


def beam(peak: float = 1e5, sigma_px: float = 5.0, cx: float = TRUE_CX, cy: float = TRUE_CY) -> np.ndarray:
    yy, xx = np.indices(SHAPE)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return peak * np.exp(-r2 / (2 * sigma_px ** 2)) + peak * 1e-4 * np.exp(-r2 / (2 * 25.0 ** 2))


def ring(r0: float = 80.0, amp: float = 200.0, width: float = 2.5, cx: float = TRUE_CX, cy: float = TRUE_CY) -> np.ndarray:
    yy, xx = np.indices(SHAPE)
    r = np.hypot(xx - cx, yy - cy)
    return amp * np.exp(-((r - r0) ** 2) / (2 * width ** 2))


def power_law(amp: float = 5e3, p: float = 4.0, cx: float = TRUE_CX, cy: float = TRUE_CY, r_min: float = 8.0) -> np.ndarray:
    """Isotropic q^-p scattering (amp at r = 10 px), flat inside r_min."""
    yy, xx = np.indices(SHAPE)
    r = np.maximum(np.hypot(xx - cx, yy - cy), r_min)
    return amp * (10.0 / r) ** p


def lobes(x0: float = 20.0, amp: float = 5e3, half_h: float = 10.0, half_w: float = 1.5,
          cx: float = TRUE_CX, cy: float = TRUE_CY) -> np.ndarray:
    yy, xx = np.indices(SHAPE)
    dx, dy = xx - cx, yy - cy
    return amp * (((np.abs(np.abs(dx) - x0) <= half_w) & (np.abs(dy) <= half_h)).astype(float))


def streak(amp: float = 50.0, half_w: float = 2.0, axis: str = "h", cx: float = TRUE_CX, cy: float = TRUE_CY) -> np.ndarray:
    yy, xx = np.indices(SHAPE)
    if axis == "h":
        return amp * (np.abs(yy - cy) <= half_w).astype(float)
    return amp * (np.abs(xx - cx) <= half_w).astype(float)


def empty_frame(peak: float = 2e5, background: float = 3.0, noise: Optional[np.random.Generator] = None,
                cx: float = TRUE_CX, cy: float = TRUE_CY) -> np.ndarray:
    a = beam(peak=peak, cx=cx, cy=cy) + background
    if noise is not None:
        a = a + noise.normal(scale=0.3, size=SHAPE)
    return a


def sample_frame(T: float = 0.2, empty: Optional[np.ndarray] = None, ring_amp: float = 400.0, ring_r: float = 90.0,
                 pl_amp: float = 0.0, with_lobes: bool = False, with_streak: bool = False,
                 noise: Optional[np.random.Generator] = None, cx: float = TRUE_CX, cy: float = TRUE_CY,
                 flat: float = 0.5) -> np.ndarray:
    """Attenuated empty + isotropic scattering (``flat`` counts everywhere, as any real powder gives,
    plus a ring and/or a power law) + optional lobes / streak."""
    e = empty_frame(cx=cx, cy=cy) if empty is None else empty
    a = T * e + flat
    if ring_amp:
        a = a + ring(r0=ring_r, amp=ring_amp, width=3.0, cx=cx, cy=cy)
    if pl_amp:
        a = a + power_law(amp=pl_amp, cx=cx, cy=cy)
    if with_lobes:
        a = a + lobes(cx=cx, cy=cy)
    if with_streak:
        a = a + streak(cx=cx, cy=cy)
    if noise is not None:
        a = a + noise.normal(scale=0.3, size=SHAPE)
    return a


def edf_header(comment: str, date: str, exposure_s: float = 100.0, cx_header: float = TRUE_CX, cy_header: float = TRUE_CY,
               distance_m: float = DIST_M) -> Dict[str, str]:
    return {"Center_1": f"{cx_header}", "Center_2": f"{cy_header}", "SampleDistance": f"{distance_m}",
            "WaveLength": f"{WAVELENGTH_M}", "PSize_1": f"{PIXEL_M}", "PSize_2": f"{PIXEL_M}",
            "ExposureTime": f"{exposure_s}", "Comment": comment, "Date": date, "CountCutoff": f"{COUNT_CUTOFF:.0f}"}


def write_edf(path: str, data: np.ndarray, header: Dict[str, str]) -> str:
    import fabio.edfimage

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fabio.edfimage.EdfImage(data=np.asarray(data, np.float32), header=header).write(path)
    return path


def make_campaign(root: str, header_offset_px: float = 6.0, seed: int = 7) -> Dict[str, str]:
    """A two-session tree. Session A (cx header offset by ``header_offset_px``): Empty, S1 (ring), S2 (opaque
    with lobes, T = e^-7; no scattering inside the beam footprint, as for a real powder). Session B: Empty, Empty_0p5mm, S3_0p5mm (T = 0.5), S4 (marginal, T = e^-4, with
    a horizontal streak), and a calibrant AgBeh. Returns ``{stem: path}``."""
    rng = np.random.default_rng(seed)
    cxh = TRUE_CX + header_offset_px
    paths: Dict[str, str] = {}
    # session A
    eA = empty_frame(noise=rng)
    paths["A/Empty"] = write_edf(os.path.join(root, "sessionA", "Empty.edf"), eA,
                                 edf_header("Empty - Empty_1mm_SAXS", "2026-01-09T10:00:00", cx_header=cxh))
    paths["A/S1"] = write_edf(os.path.join(root, "sessionA", "S1.edf"), sample_frame(T=0.2, empty=eA, ring_amp=400.0, noise=rng),
                              edf_header("S1 - S1_1mm_SAXS", "2026-01-09T10:20:00", cx_header=cxh))
    paths["A/S2"] = write_edf(os.path.join(root, "sessionA", "S2.edf"),
                              sample_frame(T=np.exp(-7.0), empty=eA, ring_amp=400.0, with_lobes=True, noise=rng),
                              edf_header("S2 - S2_1mm_SAXS", "2026-01-09T10:40:00", cx_header=cxh))
    # session B (header centre correct)
    eB = empty_frame(noise=rng)
    paths["B/Empty"] = write_edf(os.path.join(root, "sessionB", "Empty.edf"), eB,
                                 edf_header("Empty - Empty_1mm_SAXS", "2026-02-10T09:00:00"))
    eB5 = empty_frame(peak=1.5e5, noise=rng)
    paths["B/Empty_0p5mm"] = write_edf(os.path.join(root, "sessionB", "Empty_0p5mm.edf"), eB5,
                                       edf_header("Empty_0p5mm - Empty_0p5mm_SAXS", "2026-02-10T09:30:00"))
    paths["B/S3_0p5mm"] = write_edf(os.path.join(root, "sessionB", "S3_0p5mm.edf"), sample_frame(T=0.5, empty=eB5, ring_amp=300.0, noise=rng),
                                    edf_header("S3_0p5mm - S3_0p5mm_SAXS", "2026-02-10T09:45:00"))
    paths["B/S4"] = write_edf(os.path.join(root, "sessionB", "S4.edf"),
                              sample_frame(T=np.exp(-4.0), empty=eB, ring_amp=300.0, with_streak=True, noise=rng),
                              edf_header("S4 - S4_1mm_SAXS", "2026-02-10T10:00:00"))
    paths["B/AgBeh"] = write_edf(os.path.join(root, "sessionB", "AgBeh.edf"), sample_frame(T=0.6, empty=eB, ring_amp=800.0, ring_r=60.0, noise=rng),
                                 edf_header("AgBeh - AgBeh_SAXS", "2026-02-10T08:30:00", exposure_s=10.0))
    return paths


def campaign_truth() -> Dict[str, Tuple[float, str]]:
    """(T, expected class) per sample stem of :func:`make_campaign`."""
    return {"S1": (0.2, "measurable"), "S2": (float(np.exp(-7.0)), "opaque"),
            "S3_0p5mm": (0.5, "measurable"), "S4": (float(np.exp(-4.0)), "marginal")}
