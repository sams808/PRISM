"""
qt_cluster.py — cluster-analysis workspace (M16), thin Qt layer over
cluster_science.py. Select a series of library spectra (e.g. a multi-point
Raman map), cluster them (KMeans/hierarchical), and read the result from
two views: a PCA scatter colored by cluster and the per-cluster mean
spectra — the RamanLab-style "which points on my sample are the same
phase" workflow.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from cluster_science import build_feature_matrix, cluster_means, cluster_spectra, pca_scores
from qt_models import SpectrumLibrary
from qt_widgets import PlotWidget

CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]


def _to_float(text: str) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return None


class ClusterWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self._last_result = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Select spectra to cluster (3+)"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        left_layout.addWidget(self.file_list, 1)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["kmeans", "hierarchical"])
        method_row.addWidget(self.method_combo)
        method_row.addWidget(QLabel("Clusters"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(3)
        method_row.addWidget(self.n_clusters_spin)
        left_layout.addLayout(method_row)

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalize"))
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["area", "max", "none"])
        norm_row.addWidget(self.normalize_combo)
        left_layout.addLayout(norm_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("X range"))
        self.xmin_edit = QLineEdit()
        self.xmin_edit.setPlaceholderText("min")
        self.xmax_edit = QLineEdit()
        self.xmax_edit.setPlaceholderText("max")
        range_row.addWidget(self.xmin_edit)
        range_row.addWidget(self.xmax_edit)
        left_layout.addLayout(range_row)

        run_btn = QPushButton("Run clustering")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run_clustering)
        left_layout.addWidget(run_btn)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label)

        left_layout.addWidget(QLabel("Assignments"))
        self.assign_table = QTableWidget(0, 2)
        self.assign_table.setHorizontalHeaderLabels(["Spectrum", "Cluster"])
        self.assign_table.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self.assign_table, 1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.scatter_plot = PlotWidget(figsize=(7, 3.2))
        right_layout.addWidget(self.scatter_plot, 1)
        self.means_plot = PlotWidget(figsize=(7, 3.2))
        right_layout.addWidget(self.means_plot, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        selected = {self.file_list.item(i).data(Qt.UserRole)
                    for i in range(self.file_list.count()) if self.file_list.item(i).isSelected()}
        self.file_list.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is None:
                continue
            item = QListWidgetItem(spectrum.title)
            item.setData(Qt.UserRole, sid)
            self.file_list.addItem(item)
            if sid in selected:
                item.setSelected(True)

    def _selected_spectra(self):
        out = []
        for item in self.file_list.selectedItems():
            sp = self.library.get(item.data(Qt.UserRole))
            if sp is not None:
                out.append(sp)
        return out

    # ------------------------------------------------------------------
    def run_clustering(self) -> None:
        spectra = self._selected_spectra()
        if len(spectra) < 3:
            QMessageBox.warning(self, "Clustering", "Select at least 3 spectra.")
            return
        n_clusters = self.n_clusters_spin.value()
        if n_clusters >= len(spectra):
            QMessageBox.warning(self, "Clustering", f"Cluster count ({n_clusters}) must be smaller than the number of selected spectra ({len(spectra)}).")
            return

        normalize = self.normalize_combo.currentText()
        normalize = None if normalize == "none" else normalize
        try:
            matrix, grid = build_feature_matrix(
                [(sp.x, sp.y) for sp in spectra],
                x_min=_to_float(self.xmin_edit.text()), x_max=_to_float(self.xmax_edit.text()),
                normalize=normalize,
            )
            result = cluster_spectra(matrix, method=self.method_combo.currentText(), n_clusters=n_clusters)
            pca = pca_scores(matrix, n_components=2)
        except (ValueError, ImportError) as exc:
            QMessageBox.critical(self, "Clustering error", str(exc))
            return

        labels = result["labels"]
        self._last_result = {"spectra": spectra, "labels": labels, "matrix": matrix, "grid": grid}

        sil_txt = f"silhouette = {result['silhouette']:.3f}" if result["silhouette"] is not None else "silhouette n/a"
        var = pca["explained_variance_ratio"]
        var_txt = " + ".join(f"{v * 100:.0f}%" for v in var[:2])
        self.result_label.setText(f"{result['method']}, k={n_clusters}: {sil_txt}. PCA variance: {var_txt}.")

        self.assign_table.setRowCount(len(spectra))
        for row, (sp, lbl) in enumerate(zip(spectra, labels)):
            self.assign_table.setItem(row, 0, QTableWidgetItem(sp.title))
            self.assign_table.setItem(row, 1, QTableWidgetItem(str(int(lbl))))
        self.assign_table.resizeColumnsToContents()

        self._render_scatter(pca["scores"], labels, var)
        self._render_means(grid, matrix, labels)

    def _render_scatter(self, scores, labels, var) -> None:
        fig = self.scatter_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        for lbl in sorted(set(labels.tolist())):
            mask = labels == lbl
            color = CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            ax.scatter(scores[mask, 0], scores[mask, 1] if scores.shape[1] > 1 else np.zeros(mask.sum()),
                       color=color, s=36, label=f"cluster {lbl}")
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)" if len(var) > 1 else "PC2")
        ax.set_title("PCA overview")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.scatter_plot.canvas.draw_idle()

    def _render_means(self, grid, matrix, labels) -> None:
        fig = self.means_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        for lbl, mean in cluster_means(matrix, labels).items():
            color = CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            ax.plot(grid, mean, lw=1.3, color=color, label=f"cluster {lbl} mean")
        ax.set_xlabel("x")
        ax.set_ylabel("normalized intensity")
        ax.set_title("Per-cluster mean spectra")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.means_plot.canvas.draw_idle()
