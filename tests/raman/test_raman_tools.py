"""raman.raman_tools — peak table, axis conversions, calibration and the
Raman-side single-spectrum operations, checked against closed forms."""
from __future__ import annotations

import math

import numpy as np
import pytest

import raman.raman_tools as rt


def _two_peaks(noise=0.0, seed=0):
    x = np.linspace(100.0, 1500.0, 2801)
    y = (50.0 + 1000.0 * np.exp(-0.5 * ((x - 520.0) / 4.0) ** 2)
         + 300.0 * np.exp(-0.5 * ((x - 1050.0) / 8.0) ** 2))
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, x.size)
    return x, y


def test_peak_table_positions_heights_and_fwhm():
    x, y = _two_peaks()
    rows = rt.peak_table(x, y)
    assert [round(r.position, 1) for r in rows] == [520.0, 1050.0]
    assert rows[0].height == pytest.approx(1050.0, abs=1.0)
    # FWHM of a Gaussian = 2.3548 sigma, measured at half PROMINENCE above the base
    assert rows[0].fwhm == pytest.approx(2.3548 * 4.0, rel=0.05)
    assert rows[1].fwhm == pytest.approx(2.3548 * 8.0, rel=0.05)
    # area of the Gaussian above its base: A*sigma*sqrt(2pi), minus the tails outside the half-width window
    full = 1000.0 * 4.0 * math.sqrt(2 * math.pi)
    assert 0.7 * full < rows[0].area < full


def test_peak_table_threshold_drops_noise_and_inverted_finds_minima():
    x, y = _two_peaks(noise=5.0)
    rows = rt.peak_table(x, y, threshold_frac=0.05)
    assert [r.position for r in rows] == pytest.approx([520, 1050], abs=1.0)
    rows_inv = rt.peak_table(x, -y)
    assert all(abs(r.position - 520) > 5 and abs(r.position - 1050) > 5 for r in rows_inv)  # the real peaks are minima of -y
    rows_inv = rt.peak_table(x, -y, inverted=True)
    assert [r.position for r in rows_inv] == pytest.approx([520, 1050], abs=1.0)
    assert rows_inv[0].height == pytest.approx(-1050.0, abs=15.0)


def test_peak_table_text_and_range():
    x, y = _two_peaks()
    rows = rt.peak_table(x, y, x_range=(900, 1200))
    assert len(rows) == 1 and round(rows[0].position) == 1050
    txt = rt.peak_table_text(rows)
    assert txt.splitlines()[0].startswith("position") and "1050.0" in txt


def test_axis_conversions_round_trip_and_known_values():
    laser = 532.0
    shift = np.array([0.0, 520.7, 1000.0, 3000.0])
    lam = rt.raman_shift_to_wavelength(shift, laser)
    assert lam[0] == pytest.approx(532.0)
    assert lam[1] == pytest.approx(1e7 / (1e7 / 532.0 - 520.7))  # 547.0 nm
    assert np.allclose(rt.wavelength_to_raman_shift(lam, laser), shift)
    x, y = rt.convert_x_axis(lam, np.ones(4), "Wavelength (nm)", "Raman shift (cm⁻¹)", laser_nm=laser)
    assert np.allclose(x, shift) and np.allclose(y, 1.0)
    x2, _ = rt.convert_x_axis(shift, np.ones(4), "Raman shift (cm⁻¹)", "Photon energy (eV)", laser_nm=laser)
    assert x2[-1] == pytest.approx(1239.841984 / 532.0)  # the laser line is the highest energy; sorted ascending
    x3, _ = rt.convert_x_axis(np.array([500.0]), np.ones(1), "Wavelength (nm)", "Wavenumber (cm⁻¹)")
    assert x3[0] == pytest.approx(20000.0)
    with pytest.raises(ValueError):
        rt.convert_x_axis(shift, np.ones(4), "Raman shift (cm⁻¹)", "Wavelength (nm)")


def test_jacobian_preserves_the_integral():
    laser = 532.0
    lam = np.linspace(540.0, 640.0, 2001)
    y = np.exp(-0.5 * ((lam - 580.0) / 5.0) ** 2)
    area_nm = np.trapezoid(y, lam)
    x, yj = rt.convert_x_axis(lam, y, "Wavelength (nm)", "Raman shift (cm⁻¹)", laser_nm=laser, jacobian=True)
    assert np.trapezoid(yj, x) == pytest.approx(area_nm, rel=2e-3)


def test_calibration_shift_linear_and_residuals():
    cal = rt.calibrate_x_axis([520.3], [520.7])
    assert cal.apply([100.0])[0] == pytest.approx(100.4)
    measured = [300.0, 520.3, 1001.0]
    true = [1.002 * m - 0.5 for m in measured]
    cal = rt.calibrate_x_axis(measured, true, order=1)
    assert cal.order == 1 and cal.rms < 1e-9
    assert np.allclose(cal.apply(measured), true)
    cal3 = rt.calibrate_x_axis(measured, true, order=4)
    assert cal3.order == 2  # capped at n_pairs - 1
    m, r = rt.parse_pairs("520.3 -> 520.7; 1001.9 = 1001.4\n300 300.2")
    assert m == [520.3, 1001.9, 300.0] and r == [520.7, 1001.4, 300.2]
    with pytest.raises(ValueError):
        rt.parse_pairs("520.3")


def test_single_spectrum_ops():
    x, y = _two_peaks()
    xn, yn = rt.normalize_to_value(x, y, x0=520.0, target=2.0)
    assert np.interp(520.0, xn, yn) == pytest.approx(2.0)
    xo, yo = rt.subtract_range_mean(x, y, xmin=1300, xmax=1500)
    assert np.mean(yo[(x >= 1300) & (x <= 1500)]) == pytest.approx(0.0, abs=1e-9)
    xr, yr = rt.remove_peak(x, y, center=1050.0, fwhm=2.3548 * 8.0)
    assert yr[np.argmin(np.abs(x - 1050.0))] == pytest.approx(50.0, abs=1.0)
    assert np.allclose(yr[x < 1000], y[x < 1000])
    s = rt.snv(y)
    assert np.mean(s) == pytest.approx(0.0, abs=1e-12) and np.std(s) == pytest.approx(1.0)
    ref = y
    corrected = rt.msc(3.0 * y + 7.0, ref)
    assert np.allclose(corrected, ref)
    with pytest.raises(ValueError):
        rt.snv(np.ones(10))
