"""Regression tests for the defects found by the 2026-10-21 audit. Each test
fails on the pre-fix code (verified by reverting the fix) and names the
behaviour that was wrong."""
from __future__ import annotations


import numpy as np
import pytest

import core.io_universal as iou


def _write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return str(p)


X = np.linspace(100.0, 1500.0, 400)
Y = 1000.0 + 500.0 * np.exp(-((X - 520.0) / 10.0) ** 2)


def _xy_of(path):
    df, meta = iou.load_any(path, return_meta=True)
    c = meta["canonical_map"]
    return (df[c["X"]].astype(float).to_numpy(), df[c["Y"]].astype(float).to_numpy(), meta["selected_parser"])


def test_generic_parser_reads_decimal_comma_space_separated(tmp_path):
    """'100,0000 1000,0000' used to import as x = 100, y = 0: the separator
    split ran before the decimal-comma normaliser ever saw a token."""
    body = "\n".join(f"{x:.4f} {y:.4f}".replace(".", ",") for x, y in zip(X, Y))
    x, y, parser = _xy_of(_write(tmp_path, "dec_comma.txt", body))
    assert parser == "generic_xy"
    assert np.allclose(x, X, atol=1e-4) and np.allclose(y, Y, atol=1e-3)


def test_generic_parser_keeps_comma_as_separator_in_plain_csv(tmp_path):
    body = "\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(X, Y))
    x, y, _ = _xy_of(_write(tmp_path, "plain.csv", body))
    assert np.allclose(x, X, atol=1e-4) and np.allclose(y, Y, atol=1e-3)


@pytest.mark.parametrize("encoding, banner", [
    ("latin-1", "# acquired at 25 \xb0C, 1 \xb5m spot\n"),
    ("utf-16", ""),
])
def test_raman_header_files_in_other_encodings_import(tmp_path, encoding, banner):
    """A Latin-1 or UTF-16 export with a Raman header failed with 'Could
    not read table with flexible settings': pandas re-read the file in
    UTF-8 after the sniffer had decoded it correctly."""
    body = banner + "Raman Shift (cm-1)\tIntensity (counts)\n" + "\n".join(f"{x:.4f}\t{y:.4f}" for x, y in zip(X, Y))
    x, y, parser = _xy_of(_write(tmp_path, f"raman_{encoding}.txt", body, encoding=encoding))
    assert parser == "raman_xy"
    assert np.allclose(x, X, atol=1e-4) and np.allclose(y, Y, atol=1e-3)


def test_xrd_header_latin1_imports(tmp_path):
    body = "# Cu K\xe1 1.5406 \xc5\n2Theta (deg)\tIntensity (counts)\n" + "\n".join(f"{x:.4f}\t{y:.4f}" for x, y in zip(X, Y))
    x, y, parser = _xy_of(_write(tmp_path, "xrd.xy", body, encoding="latin-1"))
    assert parser == "xrd_xy" and np.allclose(x, X, atol=1e-4)


def test_moving_average_is_unbiased_at_the_edges():
    """np.convolve(mode='same') divided by the full window: a flat 100
    came out at 54.5 at both ends of the Calculations smoother."""
    import processing.calc_science as cs
    x = np.linspace(0, 1, 101)
    _, s = cs.smooth(x, np.full(101, 100.0), method="moving_average", window=11)
    assert np.allclose(s, 100.0)
    y = 1000 * np.exp(-x * 5)
    _, s2 = cs.smooth(x, y, method="moving_average", window=11)
    assert s2[0] == pytest.approx(y[:6].mean(), rel=1e-9)


def test_true_voigt_width_field_is_the_half_width_at_every_eta():
    """At eta = 0.5 the profile's measured HWHM was 8.15 for a field value of
    10 (component widths were split linearly instead of through the
    Olivero-Longbothum relation)."""
    import core.fitting_science as fs
    xx = np.linspace(-100, 100, 40001)
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = fs.voigt_peak(xx, 1.0, 0.0, 10.0, eta)
        idx = np.where(y >= 0.5)[0]
        hwhm = (xx[idx[-1]] - xx[idx[0]]) / 2
        assert hwhm == pytest.approx(10.0, abs=0.02), eta


def test_htxrd_fwhm_is_a_full_width_and_tracking_follows_a_fast_drift():
    """The HT-XRD table/CSV column said FWHM but carried the fit's half
    width, and a peak drifting more than 1.5 HWHM per pattern was reported
    absent after two patterns."""
    import xrd.htxrd_science as hs
    tt = np.arange(20, 40, 0.02)
    rng = np.random.default_rng(3)
    pats = []
    for i, Tc in enumerate(range(25, 625, 100)):
        c = 30.0 - 0.002 * (Tc - 25)  # 0.2 deg per step, HWHM 0.094 deg
        y = 50 + 2000 * np.exp(-0.5 * ((tt - c) / 0.08) ** 2) + 800 * np.exp(-0.5 * ((tt - 33.0) / 0.08) ** 2)
        pats.append(hs.HtxrdPattern(path="", name=f"p{i}", x=tt, y=rng.poisson(y).astype(float),
                                    ramp_value=float(Tc), ramp_source="metadata"))
    res = hs.track_peak(pats, window_lo=28.0, window_hi=34.0, initial_center=30.0)
    assert all(r.present for r in res)
    for r in res:
        assert r.center == pytest.approx(30.0 - 0.002 * (r.ramp_value - 25), abs=0.01)
        assert r.fwhm == pytest.approx(2.3548 * 0.08, rel=0.1)


def test_open_project_clears_the_previous_session(qtbot, tmp_path, monkeypatch):
    """A project without an HT-XRD series kept showing the series of the
    previous one, and the Library's undo stack could resurrect spectra of
    the project that was just replaced."""
    from PySide6.QtWidgets import QFileDialog
    from core.qt_shell import PrismMainWindow, _load_spectrum_from_path
    import xrd.htxrd_science as hs
    window = PrismMainWindow()
    qtbot.addWidget(window)
    data = tmp_path / "r.txt"
    data.write_text("\n".join(f"{x:.3f}\t{y:.3f}" for x, y in zip(X, Y)), encoding="utf-8")
    sp = _load_spectrum_from_path(str(data))
    window.library.add(sp)
    window.library_page.push_undo(("delete", [(0, sp)]))
    window.htxrd_page.set_series([hs.HtxrdPattern(path="", name="p", x=np.arange(10.0), y=np.arange(10.0),
                                                   ramp_value=1.0, ramp_source="index")])
    out = str(tmp_path / "empty.prism")
    import core.project_io as pio
    pio.save_project(out, [], {})
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (out, "")))
    window.open_project()
    assert len(window.library) == 0
    assert window.htxrd_page.series == []
    assert window.library_page._undo_stack == [] and not window.library_page.undo_btn.isEnabled()
