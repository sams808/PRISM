"""Tests for saxs.core.edf_sectors: sector-aware reduction of beamstopless 2D frames.

Synthetic frames only (numpy arrays + a FrameGeometry); fabio is not needed here. The
scenarios mirror the real failure mode that motivated the module: an isotropic ring that must
survive, two bright vertical lobes on the horizontal axis that must not, an attenuated direct
beam whose footprint average gives the transmission, and isolated hot pixels.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxs.core import edf_sectors as es

SHAPE = (400, 380)          # (rows, cols)
CX, CY = 190.0, 200.0
GEOM = es.FrameGeometry(cx=CX, cy=CY, dist_m=0.9, pixel_m=7.5e-5, wavelength_A=1.54189, exposure_s=100.0)


def _beam(peak=1e5, sigma_px=5.0):
    yy, xx = np.indices(SHAPE)
    r2 = (xx - CX) ** 2 + (yy - CY) ** 2
    return peak * np.exp(-r2 / (2 * sigma_px ** 2)) + peak * 1e-4 * np.exp(-r2 / (2 * (25.0 ** 2)))


def _ring(r0=80.0, amp=200.0, width=2.5):
    yy, xx = np.indices(SHAPE)
    r = np.hypot(xx - CX, yy - CY)
    return amp * np.exp(-((r - r0) ** 2) / (2 * width ** 2))


def _bars(x0=20.0, amp=5e3, half_h=10, half_w=1.5):
    yy, xx = np.indices(SHAPE)
    dx, dy = xx - CX, yy - CY
    return amp * (((np.abs(np.abs(dx) - x0) <= half_w) & (np.abs(dy) <= half_h)).astype(float))


def test_q_of_radius_matches_small_angle_formula_within_one_percent():
    r = np.array([10.0, 100.0, 300.0])
    q = GEOM.q_of_radius(r)
    q_small = 2 * np.pi / GEOM.wavelength_A * r * GEOM.pixel_m / GEOM.dist_m
    assert np.all(np.abs(q / q_small - 1) < 0.01)
    assert q[2] < q_small[2]                      # the exact formula is below the small-angle one


def test_diagonal_mask_excludes_axes_and_lobe_zone():
    m = es.diagonal_sector_mask(SHAPE, CX, CY, axis_exclusion_deg=20.0, bar_rect_px=(40.0, 14.0))
    assert not m[int(CY), int(CX) + 100]           # on the horizontal axis
    assert not m[int(CY) + 100, int(CX)]           # on the vertical axis
    assert m[int(CY) + 70, int(CX) + 70]           # 45 degrees
    assert not m[int(CY) + 5, int(CX) + 30]        # inside the lobe rectangle
    assert m[int(CY) + 40, int(CX) + 45]           # outside the rectangle, 42 degrees from horizontal


def test_radial_profile_keeps_isotropic_ring_and_rejects_lobes():
    base = 5.0
    frame = base + _ring(r0=80.0, amp=200.0) + _bars(x0=20.0, amp=5e3)
    valid = es.valid_pixel_mask(frame)
    diag = es.diagonal_sector_mask(SHAPE, CX, CY, valid=valid)
    full = valid & (es.pixel_maps(SHAPE, CX, CY)[0] < 150)
    p_diag = es.radial_profile(frame, GEOM, diag, rmax_px=150)
    p_full = es.radial_profile(frame, GEOM, full, rmax_px=150)
    # the ring is recovered at r = 80 px in the diagonal wedges
    r = p_diag.r_px
    peak = r[np.nanargmax(np.where((r > 60) & (r < 100), p_diag.intensity, -1))]
    assert abs(peak - 80) <= 1
    # the lobes at r = 20 px dominate the full-azimuth average but not the diagonal one
    at20_full = np.nanmean(p_full.intensity[18:23])
    at20_diag = np.nanmean(p_diag.intensity[18:23])
    assert at20_full > 20 * base / GEOM.exposure_s
    assert at20_diag < 1.5 * base / GEOM.exposure_s


def test_hot_pixel_mask_catches_isolated_spike_but_not_smooth_gradient():
    frame = _beam(peak=1e4, sigma_px=30.0) + 20.0
    hot = es.hot_pixel_mask(frame)
    assert hot.sum() == 0                          # a smooth beam has no hot pixels
    frame[300, 100] = 5000.0
    hot = es.hot_pixel_mask(frame)
    assert hot[300, 100] and hot.sum() == 1


def test_footprint_transmission_recovers_uniform_attenuation():
    empty = _beam() + 2.0
    sample = 0.05 * empty
    vs, ve = es.valid_pixel_mask(sample), es.valid_pixel_mask(empty)
    tr = es.footprint_transmission(sample, empty, GEOM, GEOM, vs, ve)
    assert abs(tr.T / 0.05 - 1) < 1e-6
    assert abs(tr.heterogeneity - 1) < 1e-6
    assert abs(tr.T_disc8 / 0.05 - 1) < 1e-6


def test_footprint_transmission_ignores_lobes_on_horizontal_axis():
    empty = _beam() + 2.0
    sample = 1e-4 * empty + _bars(x0=12.0, amp=3e4, half_h=10)   # dark centre + bright lobes inside r < 16
    vs, ve = es.valid_pixel_mask(sample), es.valid_pixel_mask(empty)
    tr = es.footprint_transmission(sample, empty, GEOM, GEOM, vs, ve, footprint_r_px=16.0, vertical_half_deg=30.0)
    assert abs(tr.T / 1e-4 - 1) < 0.05            # vertical footprint sees the attenuated beam, not the lobes


def test_usable_range_requires_signal_above_scaled_empty_and_noise():
    r = np.arange(200)
    ratio = np.ones(200); I = np.zeros(200); sig = np.ones(200)
    ratio[50:150] = 3.0; I[50:150] = 10.0
    flag, i0, i1 = es.usable_range(ratio, I, sig, r)
    assert (i0, i1) == (50, 149)
    assert flag[50:150].all() and not flag[:50].any() and not flag[150:].any()
    # inside the footprint (r <= 30) a ratio of 3 is not enough
    ratio[10:40] = 3.0; I[10:40] = 10.0
    flag, i0, i1 = es.usable_range(ratio, I, sig, r)
    assert i0 == 31 or i0 == 50


def test_reduce_arrays_subtracts_scaled_empty_and_flags_ring_usable():
    rng = np.random.default_rng(1)
    T_true = 0.2
    empty = _beam(peak=2e5) + 3.0
    ring = _ring(r0=90.0, amp=400.0, width=3.0)
    sample = T_true * empty + ring
    sample = sample + rng.normal(scale=0.3, size=SHAPE)
    empty = empty + rng.normal(scale=0.3, size=SHAPE)
    red = es.reduce_arrays(sample, GEOM, empty, GEOM, mu_mm=10.0, name="synthetic")
    assert abs(red.transmission.T / T_true - 1) < 0.02
    assert red.t_eff_mm is not None and abs(red.t_eff_mm - (-np.log(red.transmission.T) / 10.0)) < 1e-9
    c = red.curve
    # at the ring the corrected intensity is ring / T (empty-transmission units) and it is usable
    i_ring = int(np.argmin(np.abs(c.q - GEOM.q_of_radius(np.array([90.0]))[0])))
    expected = 400.0 / T_true / GEOM.exposure_s
    assert abs(c.intensity[i_ring] / expected - 1) < 0.15
    assert c.metadata["_usable"][i_ring] == 1
    assert red.q_min_use is not None and red.q_min_use < c.q[i_ring] < red.q_max_use
    # far from the ring and the beam the corrected intensity is consistent with zero
    far = (c.q > GEOM.q_of_radius(np.array([120.0]))[0]) & (c.q < GEOM.q_of_radius(np.array([150.0]))[0])
    assert np.abs(np.mean(c.intensity[far])) < 5 * np.mean(c.sigma[far])
    assert len(red.curve_usable.q) > 0


def test_refine_centres_recovers_a_shifted_beam_and_falls_back_for_a_dark_centre():
    # beam actually 8 px to the left of the header centre, in both frames
    yy, xx = np.indices(SHAPE)
    def beam_at(cx, peak):
        return peak * np.exp(-((xx - cx) ** 2 + (yy - CY) ** 2) / (2 * 5.0 ** 2)) + 2.0
    empty = beam_at(CX - 8.0, 1e5)
    sample = 0.1 * beam_at(CX - 8.0, 1e5)
    gs2, ge2, note = es.refine_centres(sample, GEOM, empty, GEOM)
    assert abs(ge2.cx - (CX - 8.0)) < 0.3 and abs(ge2.cy - CY) < 0.3
    assert abs(gs2.cx - (CX - 8.0)) < 0.3 and note == "own direct beam"
    # opaque sample: dark centre with bright lobes 12 px either side of the TRUE centre -> the empty's centre is used
    lobes = 3e4 * (((np.abs(np.abs(xx - (CX - 8.0)) - 12.0) <= 1.5) & (np.abs(yy - CY) <= 10)).astype(float))
    dark = 1e-5 * beam_at(CX - 8.0, 1e5) + lobes
    gs3, ge3, note3 = es.refine_centres(dark, GEOM, empty, GEOM)
    assert abs(gs3.cx - ge3.cx) < 1e-9 and note3.startswith("session empty")
    # the reduction with a shifted beam recovers the transmission only when the centre is refined
    red = es.reduce_arrays(sample, GEOM, empty, GEOM, refine_centre=True)
    assert abs(red.transmission.T / 0.1 - 1) < 0.02
    assert red.curve.metadata["centre_source"] == "own direct beam"
    assert abs(red.curve.metadata["centre_offset_from_header_px"][0] + 8.0) < 0.5


def test_write_dat_round_trips_through_the_loader(tmp_path):
    empty = _beam(peak=2e5) + 3.0
    sample = 0.3 * empty + _ring(r0=70.0, amp=300.0)
    red = es.reduce_arrays(sample, GEOM, empty, GEOM, name="synthetic")
    out = tmp_path / "synthetic__sectors.dat"
    red.write_dat(str(out))
    from saxs.core.loader import load_curve
    c = load_curve(str(out))
    assert c.npts == len(red.curve.q)
    assert np.allclose(c.intensity, red.curve.intensity, rtol=1e-5)
    assert np.allclose(c.sigma, red.curve.sigma, rtol=1e-5)


@pytest.mark.skipif(not es.HAVE_FABIO, reason="fabio not installed")
def test_read_frame_geometry_from_synthetic_edf(tmp_path):
    import fabio.edfimage
    data = (_beam(peak=1e4) + 1.0).astype(np.float32)
    header = {"Center_1": str(CX), "Center_2": str(CY), "SampleDistance": "0.9", "WaveLength": "1.541891e-10",
              "PSize_1": "7.5e-05", "PSize_2": "7.5e-05", "ExposureTime": "100", "Comment": "syn - syn_test"}
    p = tmp_path / "syn.edf"
    fabio.edfimage.EdfImage(data=data, header=header).write(str(p))
    arr, h, g = es.read_frame(str(p))
    assert arr.shape == SHAPE and abs(g.wavelength_A - 1.541891) < 1e-9 and g.exposure_s == 100.0


def test_flatfield_factor_is_the_single_photon_value_that_makes_sigma_poisson():
    """A flat-field corrected frame = photon counts x per-pixel constants (0.97-1.03, plus 3 % low-gain pixels at
    0.6-0.9, as on the EIGER2 frames). Under uniform illumination var(value) / mean(value) is the factor that makes the
    Poisson sigma right; it equals the single-photon value (~1), which the legacy window (0.3-0.95) misses."""
    rng = np.random.default_rng(3)
    gain = rng.uniform(0.97, 1.03, SHAPE)
    low = rng.random(SHAPE) < 0.03
    gain[low] = rng.uniform(0.6, 0.9, int(low.sum()))
    counts = rng.poisson(0.4, SHAPE)
    data = counts * gain
    f = es.flatfield_factor(data)
    assert f == pytest.approx(1.0, abs=0.01)
    assert np.var(data) / np.mean(data) == pytest.approx(f, rel=0.03)
    assert es.flatfield_factor(data, "legacy") < 0.92
    assert es.flatfield_factor(counts.astype(float)) == 1.0                  # integer counts
    assert es.flatfield_factor(np.zeros(SHAPE)) == 1.0                       # no single photon at all
    with pytest.raises(ValueError):
        es.flatfield_factor(data, "nope")


def test_reduce_arrays_records_the_flatfield_method_and_scales_sigma_by_its_root():
    rng = np.random.default_rng(4)
    gain = rng.uniform(0.97, 1.03, SHAPE)
    low = rng.random(SHAPE) < 0.03
    gain[low] = rng.uniform(0.6, 0.9, int(low.sum()))
    empty = rng.poisson(0.4 + _beam(), SHAPE) * gain
    sample = rng.poisson(0.4 + 0.5 * _beam() + _ring(), SHAPE) * gain
    new = es.reduce_arrays(sample, GEOM, empty, GEOM, refine_centre=False)
    old = es.reduce_arrays(sample, GEOM, empty, GEOM, refine_centre=False, flatfield_method="legacy")
    assert new.curve.metadata["flatfield_method"] == "single_photon" and old.curve.metadata["flatfield_method"] == "legacy"
    assert np.allclose(new.curve.intensity, old.curve.intensity)              # the intensities do not depend on it
    f_new, f_old = new.curve.metadata["flatfield_factor_sample"], old.curve.metadata["flatfield_factor_sample"]
    assert f_new == pytest.approx(1.0, abs=0.01) and f_old < 0.92
    ratio = np.nanmedian(new.curve.sigma / old.curve.sigma)
    assert ratio == pytest.approx(np.sqrt(f_new / f_old), rel=0.02)
