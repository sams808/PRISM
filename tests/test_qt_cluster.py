"""Tests for qt_cluster.py (M16) — the clustering workspace."""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

from qt_cluster import ClusterWorkspace
from qt_models import Spectrum, SpectrumLibrary
from qt_shell import NAV_ITEMS, DataappMainWindow


def _library_with_two_families() -> SpectrumLibrary:
    library = SpectrumLibrary()
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1000, 600)
    for i in range(4):
        y = rp.gaussian(x, 80.0, 300.0 + rng.normal(0, 2), 20.0) + rng.normal(0, 0.5, x.shape)
        library.add(Spectrum(id=Spectrum.new_id(), title=f"famA_{i}", path="", kind="raman_xy", x=x, y=y))
    for i in range(4):
        y = rp.gaussian(x, 80.0, 700.0 + rng.normal(0, 2), 20.0) + rng.normal(0, 0.5, x.shape)
        library.add(Spectrum(id=Spectrum.new_id(), title=f"famB_{i}", path="", kind="raman_xy", x=x, y=y))
    return library


def test_workspace_constructs_empty(qtbot):
    widget = ClusterWorkspace()
    qtbot.addWidget(widget)
    assert widget.file_list.count() == 0


def test_run_clustering_separates_families_and_renders(qtbot):
    library = _library_with_two_families()
    widget = ClusterWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    widget.n_clusters_spin.setValue(2)

    widget.run_clustering()
    qtbot.wait(20)

    labels = widget._last_result["labels"]
    assert len(set(labels[:4].tolist())) == 1
    assert len(set(labels[4:].tolist())) == 1
    assert labels[0] != labels[4]
    assert widget.assign_table.rowCount() == 8
    assert "silhouette" in widget.result_label.text()
    assert len(widget.scatter_plot.figure.get_axes()) == 1
    assert len(widget.means_plot.figure.get_axes()[0].lines) == 2  # one mean per cluster


def test_run_clustering_too_few_spectra_warns(qtbot):
    library = _library_with_two_families()
    widget = ClusterWorkspace(library=library)
    qtbot.addWidget(widget)
    ids = [s.id for s in library.all()][:2]
    widget.set_spectra(ids)
    widget.file_list.selectAll()
    widget.run_clustering()  # warning dialog neutralized by conftest fixture
    assert widget._last_result is None


def test_run_clustering_k_not_smaller_than_n_warns(qtbot):
    library = _library_with_two_families()
    widget = ClusterWorkspace(library=library)
    qtbot.addWidget(widget)
    ids = [s.id for s in library.all()][:3]
    widget.set_spectra(ids)
    widget.file_list.selectAll()
    widget.n_clusters_spin.setValue(3)
    widget.run_clustering()
    assert widget._last_result is None


def test_shell_cluster_page_picks_up_library_records(qtbot, raman_example_path):
    from qt_shell import _load_spectrum_from_path

    window = DataappMainWindow()
    qtbot.addWidget(window)
    window.library.add(_load_spectrum_from_path(str(raman_example_path)))

    window.nav.setCurrentRow(NAV_ITEMS.index("Clustering"))
    qtbot.wait(20)
    assert window.cluster_page.file_list.count() == 1
