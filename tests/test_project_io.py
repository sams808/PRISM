"""Tests for project_io.py (M14) — .dataapp project save/load."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import project_io
from qt_models import Spectrum


def _spectrum(title="s1", with_df=True, with_match=False):
    x = np.linspace(100, 200, 50)
    y = np.sin(x / 10)
    df = pd.DataFrame({"col_x": x, "col_y": y, "extra": y * 2}) if with_df else None
    meta = {"selected_parser": "raman_xy", "canonical_map": {"X": "col_x", "Y": "col_y"}}
    if with_match:
        meta["rruff_match"] = {"mineral": "Quartz", "rruff_id": "R040031", "wavelength_nm": 532.0}
    return Spectrum(id=Spectrum.new_id(), title=title, path="C:/somewhere/original.txt",
                    kind="raman_xy", x=x, y=y, df=df, meta=meta)


def test_save_load_round_trip_preserves_everything(tmp_path):
    s1 = _spectrum("alpha", with_df=True, with_match=True)
    s2 = _spectrum("beta", with_df=False)
    fit_params = {s1.id: [{"shape": "G", "shift_val": 150.0, "fit_shift": True}]}

    path = tmp_path / "session.dataapp"
    project_io.save_project(str(path), [s1, s2], fit_params)
    spectra, loaded_params = project_io.load_project(str(path))

    assert [sp.title for sp in spectra] == ["alpha", "beta"]
    assert spectra[0].id == s1.id  # identity survives the round trip
    assert np.allclose(spectra[0].x, s1.x)
    assert np.allclose(spectra[0].y, s1.y)
    assert spectra[0].kind == "raman_xy"
    assert spectra[0].path == "C:/somewhere/original.txt"
    assert spectra[0].meta["rruff_match"]["mineral"] == "Quartz"

    assert spectra[0].df is not None
    assert list(spectra[0].df.columns) == ["col_x", "col_y", "extra"]
    assert spectra[1].df is None

    assert loaded_params[s1.id][0]["shift_val"] == 150.0


def test_meta_with_numpy_values_serializes(tmp_path):
    sp = _spectrum("np_meta", with_df=False)
    sp.meta["temp_start_C"] = np.float64(21.5)
    sp.meta["indices"] = np.array([1, 2, 3])

    path = tmp_path / "np.dataapp"
    project_io.save_project(str(path), [sp], {})
    spectra, _ = project_io.load_project(str(path))
    assert spectra[0].meta["temp_start_C"] == 21.5
    assert spectra[0].meta["indices"] == [1, 2, 3]


def test_load_rejects_non_project_zip(tmp_path):
    import zipfile
    bad = tmp_path / "notproject.dataapp"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("manifest.json", '{"format": "something-else"}')
    with pytest.raises(ValueError, match="Not a Dataapp project"):
        project_io.load_project(str(bad))


def test_load_rejects_newer_version(tmp_path):
    import zipfile
    newer = tmp_path / "future.dataapp"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("manifest.json", '{"format": "dataapp-project", "version": 99}')
    with pytest.raises(ValueError, match="newer"):
        project_io.load_project(str(newer))


def test_empty_project_round_trips(tmp_path):
    path = tmp_path / "empty.dataapp"
    project_io.save_project(str(path), [], {})
    spectra, params = project_io.load_project(str(path))
    assert spectra == []
    assert params == {}
