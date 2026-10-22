"""
qt_raman.py — the Raman workbench: the one workspace a Raman user needs
(the replacement for Spectragryph on the spectrometer's computer).

Layout, top to bottom: a ribbon of tabs (Spectra · Display · Process ·
Transform · Analyze · Identify · Sequence · Export), a parameter panel
that opens under the ribbon when an operation is chosen, then the body:
the spectra list on the left (check boxes = shown on the plot, highlight =
target of an operation, colour swatches, right-click properties), the plot
in the centre, and a results pane below the plot (tables from Analyze,
the protocol log).

Operations come from raman.raman_ops: one declaration per function gives
the form, the preview, the derived spectra written into the Library
(undoable, with the choice of keeping or hiding the originals) and the
Sequence tab's record/replay. Display-only settings (layout, colours,
units, labels, cursor, CIF overlays) never change the data.

Every redraw-triggering control goes through PlotWidget.request_redraw(),
the debounce that prevents the historical per-keystroke redraw lag.
"""
from __future__ import annotations

import colorsys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.np_compat import trapezoid

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from raman.cif_tools import bragg_peaks_from_cif_generic, list_cif_files_case_insensitive
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float
import raman.raman_ops as ro

CIF_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "goldenrod"]
LINE_COLORS = ["navy", "darkred", "seagreen", "darkorange", "purple", "teal", "brown", "indigo"]
COLOR_SCHEMES = ["Distinct", "Matplotlib cycle", "Hash by name", "Monochrome",
                 "Viridis", "Plasma", "Cividis", "Rainbow", "Tab10", "Set2"]
LINE_STYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dash-dot": "-."}
X_UNIT_MEASURED = "as measured"


class CifManagerDialog(QDialog):
    """Per-CIF visibility/label/color/pad controls — a modeless-feeling
    dialog whose edits redraw the parent's plot via the shared debounce."""

    def __init__(self, workspace: "RamanWorkspace"):
        super().__init__(workspace)
        self.workspace = workspace
        self.setWindowTitle("CIF manager")
        self.resize(680, 320)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        folder_btn = QPushButton("Set CIF folder…")
        folder_btn.clicked.connect(workspace.choose_cif_folder_and_reload)
        top.addWidget(folder_btn)

        top.addWidget(QLabel("Label pos."))
        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["right_out", "left_out", "right_in", "left_in"])
        self.pos_combo.setCurrentText(workspace.cif_label_pos)
        self.pos_combo.currentTextChanged.connect(self._on_pos_changed)
        top.addWidget(self.pos_combo)

        top.addWidget(QLabel("Bragg height ×"))
        self.height_edit = QLineEdit(str(workspace.bragg_height_scale))
        self.height_edit.textChanged.connect(self._on_height_changed)
        top.addWidget(self.height_edit)
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.list_container)
        layout.addWidget(scroll)

        self.rebuild()

    def _on_pos_changed(self, text: str) -> None:
        self.workspace.cif_label_pos = text
        self.workspace.plot.request_redraw(self.workspace.render)

    def _on_height_changed(self, text: str) -> None:
        v = _to_float(text)
        self.workspace.bragg_height_scale = v if (v is not None and v > 0.1) else 1.0
        self.workspace.plot.request_redraw(self.workspace.render)

    def rebuild(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for serie in self.workspace.cif_series:
            row = QHBoxLayout()
            show_check = QCheckBox(serie.get("label", os.path.basename(serie.get("path", ""))))
            show_check.setChecked(serie.get("visible", False))
            show_check.toggled.connect(lambda checked, s=serie: self.workspace.set_cif_field(s, "visible", checked))
            row.addWidget(show_check)

            name_edit = QLineEdit(serie.get("plot_label", ""))
            name_edit.setPlaceholderText("display label")
            name_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "plot_label", text))
            row.addWidget(name_edit)

            color_edit = QLineEdit(serie.get("color", "crimson"))
            color_edit.setMaximumWidth(80)
            color_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "color", text or "crimson"))
            row.addWidget(color_edit)

            pad_edit = QLineEdit(str(serie.get("pad", 0.03)))
            pad_edit.setMaximumWidth(50)
            pad_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "pad", _to_float(text) or 0.03))
            row.addWidget(pad_edit)

            wrapper = QWidget()
            wrapper.setLayout(row)
            self.list_layout.addWidget(wrapper)


LEGEND_MAX = 25        # above this many spectra the legend is left out (a map's pixels froze the window)
PEAK_LABEL_MAX = 12    # above this many spectra no automatic peak labels
MANY_SPECTRA = 100     # above this many spectra: thin lines, no markers, wait cursor
PICK_COLOR = "#2a9d8f"


def _clear_layout(layout) -> None:
    """Remove every widget and sub-layout from a layout, now (hide + reparent),
    not only at the next event-loop turn: the parameter form is rebuilt for
    each operation and leftovers of the previous one must not stay visible."""
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            _clear_layout(it.layout())
            it.layout().deleteLater()


def _group(title: str) -> Tuple[QWidget, QVBoxLayout]:
    """A ribbon group: small uppercase label over its controls."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(5, 2, 5, 2)
    lay.setSpacing(3)
    lab = QLabel(title)
    lab.setObjectName("GroupLabel")
    lay.addWidget(lab)
    return w, lay


def _row(*widgets, stretch: bool = False) -> QHBoxLayout:
    """Controls side by side; a string becomes a label. stretch=True keeps
    them packed to the left instead of spreading over the row."""
    lay = QHBoxLayout()
    lay.setSpacing(4)
    for w in widgets:
        if isinstance(w, str):
            lay.addWidget(QLabel(w))
        else:
            lay.addWidget(w)
    if stretch:
        lay.addStretch(1)
    return lay


class RamanWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None, *,
                 on_derived_added: Optional[Callable[[List[str]], None]] = None,
                 on_navigate: Optional[Callable[[str], None]] = None,
                 import_paths: Optional[Callable[[List[str]], Tuple[int, list]]] = None,
                 on_undo_push: Optional[Callable[[tuple], None]] = None,
                 on_library_changed: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.on_derived_added = on_derived_added
        self.on_navigate = on_navigate
        self.import_paths = import_paths
        self.on_undo_push = on_undo_push
        self.on_library_changed = on_library_changed

        self.cif_series: List[Dict[str, Any]] = []
        self.cif_label_pos = "right_out"
        self.bragg_height_scale = 1.0
        self._cif_manager: Optional[CifManagerDialog] = None
        self.annotations: List[Dict[str, float]] = []  # persists across re-renders
        self._peak_table_dialog = None
        self._cursor_artists: List[Any] = []
        self.styles: Dict[str, Dict[str, Any]] = {}     # per-spectrum display overrides
        self.sequence: List[ro.Step] = []
        self._preview: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = None
        self._current_op: Optional[ro.Op] = None
        self._param_widgets: Dict[str, QWidget] = {}
        self._pick_target: Optional[tuple] = None   # (field, mode, pending clicks, button) while picking
        self.found_peaks: Dict[str, list] = {}       # spectrum id -> [(position, height, source)] of the last Peaks & FWHM
        self._drawn_colors: Dict[str, Any] = {}

        self._build_ui()
        self.plot.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.plot.canvas.mpl_connect("motion_notify_event", self._on_cursor_move)
        self.plot.canvas.mpl_connect("key_press_event", self._on_plot_key)

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        self.ribbon = QTabWidget()
        self.ribbon.setObjectName("Ribbon")
        # the detail panels (operation form, plot settings, sequence steps) are
        # built first because the ribbon tabs hold controls that live in them
        self.plot_panel = self._build_plot_panel()
        self.seq_panel = self._build_seq_panel()
        for name, builder in (("Spectra", self._build_tab_spectra), ("Display", self._build_tab_display),
                              ("Process", lambda: self._build_tab_ops("process")),
                              ("Transform", lambda: self._build_tab_ops("transform")),
                              ("Analyze", lambda: self._build_tab_ops(
                                  "analyze", extra=self._analyze_extras,
                                  extra_buttons={"Peaks": [("Peak table…", self.open_peak_table,
                                                            "The peak positions of the shown spectra in a separate window "
                                                            "(threshold from Display › Peak labels)")]})),
                              ("Identify", self._build_tab_identify), ("Sequence", self._build_tab_sequence),
                              ("Export", self._build_tab_export)):
            self.ribbon.addTab(self._scrolled(builder()), name)
        # every tab is at most two rows of controls: size the ribbon once for
        # the tallest of them so no tab is ever clipped
        inner_h = max(self.ribbon.widget(i).widget().sizeHint().height() for i in range(self.ribbon.count()))
        self.ribbon.setFixedHeight(inner_h + self.ribbon.tabBar().sizeHint().height() + 6)
        self.ribbon.currentChanged.connect(self._on_ribbon_tab_changed)
        root.addWidget(self.ribbon)

        # ---- operation panel (hidden until an operation is chosen)
        self.op_panel = QFrame()
        self.op_panel.setObjectName("Card")
        op_lay = QVBoxLayout(self.op_panel)
        op_lay.setContentsMargins(10, 6, 10, 6)
        head = QHBoxLayout()
        self.op_title = QLabel("")
        self.op_title.setObjectName("SectionTitle")
        head.addWidget(self.op_title, 1)
        self.op_tip = QLabel("")
        self.op_tip.setObjectName("SectionNote")
        self.op_tip.setWordWrap(True)
        head.addWidget(self.op_tip, 3)
        close_btn = QPushButton("Close")
        close_btn.setToolTip("Close this panel (the preview disappears)")
        close_btn.clicked.connect(self.close_op_panel)
        head.addWidget(close_btn)
        op_lay.addLayout(head)
        self.op_form = QGridLayout()
        self.op_form.setHorizontalSpacing(8)
        self.op_form.setVerticalSpacing(4)
        op_lay.addLayout(self.op_form)
        opts = QHBoxLayout()
        self.op_ref_label = QLabel("Reference spectrum")
        self.op_ref_combo = QComboBox()
        opts.addWidget(self.op_ref_label)
        opts.addWidget(self.op_ref_combo, 1)
        self.op_treat_all = QCheckBox("Treat all shown spectra")
        self.op_treat_all.setToolTip("Apply to every spectrum shown on the plot, not only the highlighted ones.")
        opts.addWidget(self.op_treat_all)
        self.op_replace = QCheckBox("Hide originals")
        self.op_replace.setToolTip("After Apply, uncheck the input spectra so only the results stay on the plot.")
        opts.addWidget(self.op_replace)
        opts.addSpacing(10)
        opts.addWidget(QLabel("suffix"))
        self.op_suffix = QLineEdit("")
        self.op_suffix.setMaximumWidth(80)
        opts.addWidget(self.op_suffix)
        opts.addStretch(1)
        self.op_preview_btn = QPushButton("Preview")
        self.op_preview_btn.setToolTip("Draw the result dashed on the plot; nothing is changed")
        self.op_preview_btn.clicked.connect(lambda: self.run_current_op(preview=True))
        opts.addWidget(self.op_preview_btn)
        self.op_apply_btn = QPushButton("Apply")
        self.op_apply_btn.setObjectName("Primary")
        self.op_apply_btn.setToolTip("Write the result into the Library as new spectra (undo: Ctrl+Z); tables land in Results")
        self.op_apply_btn.clicked.connect(lambda: self.run_current_op(preview=False))
        opts.addWidget(self.op_apply_btn)
        self.op_seq_btn = QPushButton("Add to sequence")
        self.op_seq_btn.setToolTip("Record this operation with its parameters as a step of the Sequence (Sequence tab)")
        self.op_seq_btn.clicked.connect(self.add_current_op_to_sequence)
        opts.addWidget(self.op_seq_btn)
        op_lay.addLayout(opts)
        self.op_panel.setVisible(False)
        root.addWidget(self.op_panel)
        root.addWidget(self.plot_panel)
        root.addWidget(self.seq_panel)
        self._panel: Optional[QWidget] = None

        # ---- body
        body = QSplitter(Qt.Horizontal)
        root.addWidget(body, 1)

        left = QWidget()
        left.setObjectName("Card")
        left.setMinimumWidth(220)
        left.setMaximumWidth(360)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(4)
        head_row = QHBoxLayout()
        head_row.addWidget(QLabel("Spectra"))
        head_row.addStretch(1)
        self.visible_count_label = QLabel("")
        self.visible_count_label.setObjectName("SectionNote")
        head_row.addWidget(self.visible_count_label)
        left_lay.addLayout(head_row)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter by name…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip("Show only the list entries whose name contains this text (a map's 2000 pixels, a series…)")
        self.filter_edit.textChanged.connect(self._apply_filter)
        left_lay.addWidget(self.filter_edit)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setToolTip("Tick = shown on the plot. Highlight = target of an operation (Ctrl/Shift-click for several). Right-click for properties.")
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.file_list.itemChanged.connect(self._on_item_changed)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._on_list_menu)
        left_lay.addWidget(self.file_list, 1)
        show_row = QHBoxLayout()
        for text, slot, tip in (("Show all", lambda: self._set_all_visible(True), "Tick every spectrum of the list"),
                                ("Hide all", lambda: self._set_all_visible(False), "Untick every spectrum (the list keeps them)"),
                                ("Only selected", self.show_only_selected, "Show the highlighted spectra and hide the others")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            show_row.addWidget(b)
        left_lay.addLayout(show_row)
        body.addWidget(left)

        centre = QSplitter(Qt.Vertical)
        self.centre_splitter = centre
        plot_card = QWidget()
        plot_card.setObjectName("Card")
        plot_lay = QVBoxLayout(plot_card)
        plot_lay.setContentsMargins(6, 6, 6, 6)
        self.plot = PlotWidget(figsize=(8, 5))
        plot_lay.addWidget(self.plot, 1)
        self.cursor_label = QLabel("")
        self.cursor_label.setObjectName("SectionNote")
        self.cursor_label.setWordWrap(True)
        plot_lay.addWidget(self.cursor_label)
        centre.addWidget(plot_card)

        results = QWidget()
        results.setObjectName("Card")
        res_lay = QVBoxLayout(results)
        res_lay.setContentsMargins(6, 4, 6, 4)
        self.results_tabs = QTabWidget()
        table_page = QWidget()
        tl = QVBoxLayout(table_page)
        tl.setContentsMargins(0, 0, 0, 0)
        trow = QHBoxLayout()
        self.results_note = QLabel("")
        self.results_note.setObjectName("SectionNote")
        trow.addWidget(self.results_note, 1)
        copy_res = QPushButton("Copy")
        copy_res.setToolTip("The results table as tab-separated text, ready to paste into Excel or Origin")
        copy_res.clicked.connect(self.copy_results)
        trow.addWidget(copy_res)
        save_res = QPushButton("Save CSV…")
        save_res.setToolTip("The results table as a CSV file")
        save_res.clicked.connect(self.save_results_csv)
        trow.addWidget(save_res)
        tl.addLayout(trow)
        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tl.addWidget(self.results_table, 1)
        self.results_tabs.addTab(table_page, "Results")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.results_tabs.addTab(self.log, "Log")
        res_lay.addWidget(self.results_tabs)
        centre.addWidget(results)
        centre.setStretchFactor(0, 4)
        centre.setStretchFactor(1, 1)
        centre.setSizes([620, 140])      # the results pane grows by itself when a table arrives
        body.addWidget(centre)
        body.setStretchFactor(1, 1)

    def _scrolled(self, inner: QWidget) -> QScrollArea:
        lay = inner.layout()
        if lay is not None:                      # groups hang from the top, whatever their height
            for i in range(lay.count()):
                w = lay.itemAt(i).widget()
                if w is not None:
                    lay.setAlignment(w, Qt.AlignTop)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        return sa

    def _ribbon_page(self) -> Tuple[QWidget, QHBoxLayout]:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignLeft)
        return page, lay

    def _btn(self, text: str, slot, tip: str = "", primary: bool = False) -> QPushButton:
        b = QPushButton(text)
        if tip:
            b.setToolTip(tip)
        if primary:
            b.setObjectName("Primary")
        b.clicked.connect(slot)
        return b

    # ---- detail panels under the ribbon --------------------------------
    def _panels(self) -> Tuple[QWidget, ...]:
        return (self.op_panel, self.plot_panel, self.seq_panel)

    def _show_panel(self, panel: QWidget) -> None:
        """One detail panel at a time under the ribbon."""
        for p in self._panels():
            p.setVisible(p is panel)
        self._panel = panel

    def _hide_panel(self, panel: QWidget) -> None:
        panel.setVisible(False)
        if self._panel is panel:
            self._panel = None

    def _on_ribbon_tab_changed(self, index: int) -> None:
        name = self.ribbon.tabText(index)
        if name == "Sequence":
            self._show_panel(self.seq_panel)
            return
        if self._panel is self.seq_panel:
            self._hide_panel(self.seq_panel)
        key = {"Process": "process", "Transform": "transform", "Analyze": "analyze"}.get(name)
        if key and self._current_op is not None and self._current_op.tab == key and self._panel is not self.op_panel:
            self._show_panel(self.op_panel)

    def open_plot_settings(self) -> None:
        self._show_panel(self.plot_panel)

    def _panel_head(self, panel: QWidget, title: str, note: str) -> QHBoxLayout:
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        head.addWidget(t, 1)
        n = QLabel(note)
        n.setObjectName("SectionNote")
        n.setWordWrap(True)
        head.addWidget(n, 3)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: self._hide_panel(panel))
        head.addWidget(close_btn)
        return head

    def _build_plot_panel(self) -> QFrame:
        """Titles, limits, font, the x unit of the data and the peak-label
        options: the typed plot settings, opened from Display > Plot settings."""
        redraw = lambda *_: self.plot.request_redraw(self.render)  # noqa: E731
        panel = QFrame()
        panel.setObjectName("Card")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addLayout(self._panel_head(panel, "Plot settings",
                                       "Titles, limits, font, the unit of the measured x axis and the peak-label "
                                       "options. Display only: the data are not changed."))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("plot title")
        self.title_edit.textChanged.connect(redraw)
        self.no_title_check = QCheckBox("No title")
        self.no_title_check.toggled.connect(redraw)
        self.no_yticks_check = QCheckBox("No y ticks")
        self.no_yticks_check.toggled.connect(redraw)
        self.font_combo = QComboBox()
        self.font_combo.addItems(["8", "9", "10", "12", "14", "16"])
        self.font_combo.setCurrentText("10")
        self.font_combo.currentTextChanged.connect(redraw)
        grid.addWidget(QLabel("Title"), 0, 0)
        grid.addWidget(self.title_edit, 0, 1)
        grid.addWidget(QLabel("Font size"), 0, 2)
        grid.addLayout(_row(self.font_combo, self.no_title_check, self.no_yticks_check, stretch=True), 0, 3)

        self.x_title_edit = QLineEdit("Raman Shift (cm⁻¹)")
        self.x_title_edit.textChanged.connect(redraw)
        self.y_title_edit = QLineEdit("Intensity (a.u.)")
        self.y_title_edit.textChanged.connect(redraw)
        grid.addWidget(QLabel("X title"), 1, 0)
        grid.addWidget(self.x_title_edit, 1, 1)
        grid.addWidget(QLabel("Y title"), 1, 2)
        grid.addWidget(self.y_title_edit, 1, 3)

        self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        for e, ph in ((self.xmin_edit, "x min"), (self.xmax_edit, "x max"), (self.ymin_edit, "y min"), (self.ymax_edit, "y max")):
            e.setMaximumWidth(72)
            e.setPlaceholderText(ph)
            e.textChanged.connect(lambda *_: self.plot.request_redraw(self._apply_axis_limits))
        grid.addWidget(QLabel("X limits"), 2, 0)
        grid.addLayout(_row(self.xmin_edit, self.xmax_edit, stretch=True), 2, 1)
        grid.addWidget(QLabel("Y limits"), 2, 2)
        grid.addLayout(_row(self.ymin_edit, self.ymax_edit, self._btn("Auto", self.autoscale_y, "Clear the y limits"), stretch=True), 2, 3)

        self.data_unit_combo = QComboBox()
        self.data_unit_combo.addItems(ro.UNITS)
        self.data_unit_combo.setToolTip("The unit of the measured x axis (what the files contain)")
        self.data_unit_combo.currentTextChanged.connect(redraw)
        self.laser_edit = QLineEdit("532")
        self.laser_edit.setMaximumWidth(52)
        self.laser_edit.setToolTip("Laser wavelength, needed to convert between wavelength and Raman shift")
        self.laser_edit.textChanged.connect(redraw)
        grid.addWidget(QLabel("Data x unit"), 3, 0)
        grid.addLayout(_row(self.data_unit_combo, "laser (nm)", self.laser_edit, stretch=True), 3, 1)

        self.peak_content_combo = QComboBox()
        self.peak_content_combo.addItems(["x", "y", "x, y"])
        self.peak_content_combo.currentTextChanged.connect(redraw)
        self.peak_digits_combo = QComboBox()
        self.peak_digits_combo.addItems(["0", "1", "2"])
        self.peak_digits_combo.setCurrentText("1")
        self.peak_digits_combo.currentTextChanged.connect(redraw)
        self.peak_angle_combo = QComboBox()
        self.peak_angle_combo.addItems(["90°", "45°", "0°"])
        self.peak_angle_combo.currentTextChanged.connect(redraw)
        grid.addWidget(QLabel("Peak labels"), 3, 2)
        grid.addLayout(_row("show", self.peak_content_combo, "digits", self.peak_digits_combo, "angle", self.peak_angle_combo, stretch=True), 3, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addLayout(grid)
        panel.setVisible(False)
        return panel

    def _build_seq_panel(self) -> QFrame:
        """The steps of the sequence, shown whenever the Sequence tab is open."""
        panel = QFrame()
        panel.setObjectName("Card")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addLayout(self._panel_head(panel, "Sequence",
                                       "Steps run in this order on each target spectrum. Add a step from any operation "
                                       "panel (Add to sequence); double-click a step to reopen it with its parameters."))
        self.seq_empty = QLabel("No step yet. Open an operation in Process, Transform or Analyze, set its parameters and click "
                                "Add to sequence; then run the steps from the ribbon.")
        self.seq_empty.setObjectName("SectionNote")
        self.seq_empty.setWordWrap(True)
        lay.addWidget(self.seq_empty)
        self.seq_list = QListWidget()
        self.seq_list.setMinimumHeight(96)
        self.seq_list.setMaximumHeight(150)
        self.seq_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.seq_list.itemDoubleClicked.connect(self._open_seq_step)
        lay.addWidget(self.seq_list)
        panel.setVisible(False)
        return panel

    def _open_seq_step(self, item: QListWidgetItem) -> None:
        i = self.seq_list.row(item)
        if 0 <= i < len(self.sequence):
            st = self.sequence[i]
            self.open_op(ro.OPS_BY_KEY[st.op], dict(st.params))
            if st.options.get("suffix"):
                self.op_suffix.setText(str(st.options["suffix"]))

    # ---- Spectra tab --------------------------------------------------
    def _build_tab_spectra(self) -> QWidget:
        page, lay = self._ribbon_page()
        g, gl = _group("Import")
        gl.addLayout(_row(self._btn("Files…", self.import_files, "Any text/CSV/JCAMP, Thermo OMNIC .spa/.mapx (Ctrl+O)"),
                          self._btn("Folder…", self.import_folder, "Every data file of a folder")))
        gl.addLayout(_row(self._btn("Paste", self.paste_clipboard, "A table copied from Excel, Origin or a text editor")))
        lay.addWidget(g)
        g, gl = _group("Spectra")
        gl.addLayout(_row(self._btn("Rename…", self.rename_selected, "Rename the highlighted spectrum"),
                          self._btn("Duplicate", self.duplicate_selected, "Copy the highlighted spectra as new entries"),
                          self._btn("Properties…", self.open_properties, "Colour, line style, offset, laser wavelength, notes")))
        gl.addLayout(_row(self._btn("Remove", self.remove_selected, "Remove the highlighted spectra (undo: Ctrl+Z)"),
                          self._btn("Remove all", self.remove_all, "Empty the library (undo: Ctrl+Z)"), stretch=True))
        lay.addWidget(g)
        g, gl = _group("Sort")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["A → Z", "Z → A", "Reverse order", "By laser λ", "By acquisition time", "By maximum"])
        gl.addWidget(self.sort_combo)
        gl.addWidget(self._btn("Sort", self.sort_spectra, "Reorder the list (and the Library) as chosen above"))
        lay.addWidget(g)
        g, gl = _group("Colour by property")
        self.colour_prop_combo = QComboBox()
        self.colour_prop_combo.addItems(["maximum", "area", "centre of gravity", "x at maximum", "mean", "laser λ", "list order"])
        gl.addWidget(self.colour_prop_combo)
        gl.addLayout(_row(self._btn("Apply", self.colour_by_property, "Colour the shown spectra by the property (Viridis scale)"),
                          self._btn("Reset colours", self.reset_colours, "Back to the colour scheme of the Display tab")))
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    # ---- Display tab --------------------------------------------------
    def _build_tab_display(self) -> QWidget:
        page, lay = self._ribbon_page()
        redraw = lambda *_: self.plot.request_redraw(self.render)  # noqa: E731

        g, gl = _group("Layout")
        self.mode_group = QButtonGroup(self)
        self.mode_overlay = QRadioButton("Overlay")
        self.mode_stacked = QRadioButton("Stacked")
        self.mode_separate = QRadioButton("Panels")
        self.mode_overlay.setChecked(True)
        for i, rb in enumerate((self.mode_overlay, self.mode_stacked, self.mode_separate)):
            self.mode_group.addButton(rb, i)
            rb.toggled.connect(redraw)
        gl.addLayout(_row(self.mode_overlay, self.mode_stacked, self.mode_separate))
        self.shift_edit = QLineEdit("0.5")
        self.shift_edit.setMaximumWidth(44)
        self.shift_edit.setToolTip("Stacked: offset between spectra, as a fraction of the largest range")
        self.shift_edit.textChanged.connect(redraw)
        self.diff_check = QCheckBox("Difference")
        self.diff_check.setToolTip("With exactly two spectra shown: plot A, B and A − B.")
        self.diff_check.toggled.connect(redraw)
        gl.addLayout(_row("Shift", self.shift_edit, self.diff_check, stretch=True))
        lay.addWidget(g)

        g, gl = _group("Style")
        self.color_combo = QComboBox()
        self.color_combo.addItems(COLOR_SCHEMES)
        self.color_combo.setToolTip("Colour scheme of the shown spectra (per-spectrum colours: Spectra > Properties)")
        self.color_combo.currentTextChanged.connect(redraw)
        self.color_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.color_combo.setMinimumContentsLength(9)
        self.linewidth_combo = QComboBox()
        self.linewidth_combo.addItems(["0.6", "1", "1.5", "2", "3"])
        self.linewidth_combo.setCurrentText("1")
        self.linewidth_combo.currentTextChanged.connect(redraw)
        gl.addLayout(_row("Colours", self.color_combo, "Width", self.linewidth_combo))
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["Lines", "Lines + points", "Points"])
        self.marker_combo.currentTextChanged.connect(redraw)
        self.legend_combo = QComboBox()
        self.legend_combo.addItems(["Legend inside", "Legend outside", "No legend"])
        self.legend_combo.setToolTip("Where the legend goes, or none")
        self.legend_combo.currentTextChanged.connect(redraw)
        gl.addLayout(_row(self.marker_combo, self.legend_combo))
        lay.addWidget(g)

        g, gl = _group("View")
        self.norm_check = QCheckBox("Normalize")
        self.norm_check.setToolTip("Area-normalise the shown spectra (area = 100 inside the x limits); display only.")
        self.norm_check.toggled.connect(redraw)
        self.zero_line_check = QCheckBox("Zero line")
        self.zero_line_check.toggled.connect(redraw)
        self.logy_check = QCheckBox("Log y")
        self.logy_check.toggled.connect(redraw)
        gl.addLayout(_row(self.norm_check, self.zero_line_check, self.logy_check))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(["None", "SG", "MA(5)"])
        self.smooth_combo.setToolTip("Display smoothing only; the data are not changed (Process › Smooth does).")
        self.smooth_combo.currentTextChanged.connect(redraw)
        self.grid_check = QCheckBox("Grid")
        self.grid_check.toggled.connect(redraw)
        gl.addLayout(_row("Smooth", self.smooth_combo, self.grid_check, stretch=True))
        lay.addWidget(g)

        g, gl = _group("Axes")
        self.xunit_combo = QComboBox()
        self.xunit_combo.addItem(X_UNIT_MEASURED)
        self.xunit_combo.addItems(ro.UNITS)
        self.xunit_combo.setToolTip("Show the x axis in another unit (display only). Raman shift needs the laser "
                                    "wavelength and the unit of the data: both in Plot settings.")
        self.xunit_combo.currentTextChanged.connect(redraw)
        self.xunit_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.xunit_combo.setMinimumContentsLength(13)
        gl.addLayout(_row("Show x as", self.xunit_combo))
        self.flipx_check = QCheckBox("Flip x")
        self.flipx_check.toggled.connect(redraw)
        gl.addLayout(_row(self.flipx_check,
                          self._btn("Plot settings…", self.open_plot_settings,
                                    "Titles, limits, font size, the unit of the data, peak-label options"),
                          stretch=True))
        lay.addWidget(g)

        g, gl = _group("Labels")
        self.peak_labels_check = QCheckBox("Peak labels")
        self.peak_labels_check.setToolTip("Label the peaks of every shown spectrum (content, digits and angle: Plot settings)")
        self.peak_labels_check.toggled.connect(redraw)
        self.peak_threshold_edit = QLineEdit("5")
        self.peak_threshold_edit.setMaximumWidth(32)
        self.peak_threshold_edit.setToolTip("Threshold, % of the intensity range")
        self.peak_threshold_edit.textChanged.connect(redraw)
        self.found_peaks_check = QCheckBox("Found peaks")
        self.found_peaks_check.setChecked(True)
        self.found_peaks_check.setToolTip("Mark on the plot the peaks of the last Peaks & FWHM run: detected (▼) and picked by hand (◆)")
        self.found_peaks_check.toggled.connect(redraw)
        gl.addLayout(_row(self.peak_labels_check, "%", self.peak_threshold_edit, self.found_peaks_check, stretch=True))
        self.cursor_check = QCheckBox("Cursor")
        self.cursor_check.setToolTip("Values of every shown spectrum at the mouse position")
        self.cursor_check.toggled.connect(self._on_cursor_toggled)
        self.annotate_check = QCheckBox("Annotate")
        self.annotate_check.setToolTip("Click on the plot to pin an x, y annotation")
        gl.addLayout(_row(self.cursor_check, self.annotate_check,
                          self._btn("Clear", self._clear_annotations, "Remove the click annotations"), stretch=True))
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    # ---- operation tabs -----------------------------------------------
    def _build_tab_ops(self, tab: str, extra: Optional[Callable[[QHBoxLayout], None]] = None,
                       extra_buttons: Optional[Dict[str, list]] = None) -> QWidget:
        page, lay = self._ribbon_page()
        groups: Dict[str, QGridLayout] = {}
        counts: Dict[str, int] = {}
        for op in ro.ops_for_tab(tab):
            if op.group not in groups:
                g, gl = _group(op.group)
                grid = QGridLayout()
                grid.setHorizontalSpacing(4)
                grid.setVerticalSpacing(3)
                gl.addLayout(grid)
                groups[op.group] = grid
                counts[op.group] = 0
                lay.addWidget(g)
            b = QPushButton(op.label.replace("&", "&&"))
            b.setToolTip(op.tip)
            b.clicked.connect(lambda _=False, o=op: self.open_op(o))
            i = counts[op.group]
            groups[op.group].addWidget(b, i % 2, i // 2)   # two rows per group, columns grow to the right
            counts[op.group] = i + 1
        for group, buttons in (extra_buttons or {}).items():
            grid = groups.get(group)
            if grid is None:
                continue
            for text, slot, tip in buttons:
                i = counts[group]
                grid.addWidget(self._btn(text, slot, tip), i % 2, i // 2)
                counts[group] = i + 1
        if extra is not None:
            extra(lay)
        lay.addStretch(1)
        return page

    def _analyze_extras(self, lay: QHBoxLayout) -> None:
        g, gl = _group("Modelling")
        gl.addWidget(self._btn("Fit peaks (Peak Fitting)", lambda: self._go("Peak Fitting"),
                               "Gaussian / pseudo-Voigt / Voigt / EMG peak fitting of the selected spectrum, with one-step iterations"))
        gl.addWidget(self._btn("Mixture analysis (Calculations)", lambda: self._go("Calculations"),
                               "Linear-combination fit of a target against reference spectra"))
        lay.addWidget(g)

    def _build_tab_identify(self) -> QWidget:
        page, lay = self._ribbon_page()
        g, gl = _group("Identify")
        gl.addWidget(self._btn("Search in Raman ID", lambda: self._go("Raman ID"),
                               "RRUFF and your reference library: by peak positions or whole-spectrum similarity"))
        gl.addWidget(self._btn("Add selected to my library…", self.add_selected_to_library,
                               "Store the highlighted spectra in your own reference library (name, laser, source, notes)"))
        lay.addWidget(g)
        g, gl = _group("Quick similarity (my library)")
        self.quick_sim_combo = QComboBox()
        for label, value in (("raw", "raw"), ("smoothed", "smooth"), ("1st derivative", "d1"), ("2nd derivative", "d2")):
            self.quick_sim_combo.addItem(label, value)
        gl.addLayout(_row("Compare", self.quick_sim_combo))
        gl.addWidget(self._btn("Rank my library", self.quick_similarity, "Pearson correlation of the highlighted spectrum against your reference library → Results"))
        lay.addWidget(g)
        g, gl = _group("Structure overlay")
        gl.addWidget(self._btn("Import CIF…", self.import_cif, "Overlay Bragg positions calculated from a structure file"))
        gl.addWidget(self._btn("CIF manager…", self.open_cif_manager, "Show, hide, colour or remove the structure overlays"))
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    def _build_tab_sequence(self) -> QWidget:
        page, lay = self._ribbon_page()
        g, gl = _group("Steps")
        gl.addWidget(self._btn("Show steps", lambda: self._show_panel(self.seq_panel),
                               "The steps of the sequence, under the ribbon (double-click one to reopen it)"))
        self.seq_count_label = QLabel("no step yet")
        self.seq_count_label.setObjectName("SectionNote")
        gl.addWidget(self.seq_count_label)
        lay.addWidget(g)
        g, gl = _group("Edit")
        gl.addLayout(_row(self._btn("Remove step", self.seq_remove, "Remove the steps highlighted in the list"),
                          self._btn("Clear", self.seq_clear, "Remove every step")))
        gl.addLayout(_row(self._btn("Save…", self.seq_save, "Save the sequence as a .json file to reuse or share"),
                          self._btn("Load…", self.seq_load, "Load a saved sequence (.json)")))
        lay.addWidget(g)
        g, gl = _group("Run")
        gl.addWidget(self._btn("On highlighted spectra", lambda: self.seq_run(mode="selected"),
                               "Run every step, in order, on the spectra highlighted in the list", primary=True))
        gl.addLayout(_row(self._btn("On all shown", lambda: self.seq_run(mode="visible"), "Run the sequence on every spectrum shown on the plot"),
                          self._btn("On a folder…", lambda: self.seq_run(mode="folder"),
                                    "Import every file of a folder, then run the sequence on them")))
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    def _build_tab_export(self) -> QWidget:
        page, lay = self._ribbon_page()
        g, gl = _group("Data")
        gl.addLayout(_row(self._btn("Text files…", self.export_text_files, "One .txt/.csv per highlighted spectrum"),
                          self._btn("Combined CSV…", self.export_combined_csv, "All shown spectra on a common grid, one column each")))
        gl.addLayout(_row(self._btn("JCAMP-DX…", self.export_jcamp, "One .jdx per highlighted spectrum"),
                          self._btn("Copy data", self.copy_data_to_clipboard,
                                    "Copy the shown spectra as tab-separated columns (x, y, y, …) for Excel or Origin")))
        lay.addWidget(g)
        g, gl = _group("Plot")
        self.export_png_check = QCheckBox("PNG")
        self.export_png_check.setChecked(True)
        self.export_svg_check = QCheckBox("SVG")
        self.export_pdf_check = QCheckBox("PDF")
        self.export_eps_check = QCheckBox("EPS")
        self.export_tif_check = QCheckBox("TIFF")
        self.export_dpi_edit = QLineEdit("300")
        self.export_dpi_edit.setMaximumWidth(44)
        gl.addLayout(_row(self.export_png_check, self.export_svg_check, self.export_pdf_check, self.export_eps_check, self.export_tif_check, "dpi", self.export_dpi_edit))
        gl.addLayout(_row(self._btn("Export plot…", self.export_all_formats, "Save the plot in the ticked formats at the chosen dpi"),
                          self._btn("Copy plot", self.copy_plot_to_clipboard, "Copy the plot as an image to the clipboard"),
                          self._btn("Print…", self.print_plot, "Print the plot")))
        lay.addWidget(g)
        g, gl = _group("Results")
        gl.addWidget(self._btn("Save results CSV…", self.save_results_csv, "The results table as a CSV file"))
        gl.addWidget(self._btn("Copy results", self.copy_results, "The results table as tab-separated text, ready to paste into Excel or Origin"))
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    def _go(self, nav_name: str) -> None:
        if self.on_navigate is not None:
            self.on_navigate(nav_name)

    # ==================================================================
    # Spectra list
    # ==================================================================
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        """Populate the list from the library ids (called by the shell when
        this workspace becomes active, and after every change). Check
        states and highlights survive; when nothing is shown, the first
        spectrum is shown so arriving on the page always gives a plot."""
        visible = {self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())
                   if self.file_list.item(i).checkState() == Qt.Checked}
        selected = {self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())
                    if self.file_list.item(i).isSelected()}
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is None:
                continue
            item = QListWidgetItem(spectrum.title)
            item.setData(Qt.UserRole, sid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if sid in visible else Qt.Unchecked)
            item.setToolTip(self._item_tip(spectrum))
            self.file_list.addItem(item)
            if sid in selected:
                item.setSelected(True)
        if self.file_list.count() and not any(self.file_list.item(i).checkState() == Qt.Checked for i in range(self.file_list.count())):
            self.file_list.item(0).setCheckState(Qt.Checked)
            self.file_list.item(0).setSelected(True)
        self.file_list.blockSignals(False)
        self._apply_filter()
        self._refresh_ref_combo()
        self.plot.request_redraw(self.render)

    def _item_tip(self, s: Spectrum) -> str:
        bits = [f"{len(s.x)} points, {np.nanmin(s.x):.1f}-{np.nanmax(s.x):.1f}", f"kind {s.kind}"]
        for key in ("laser_nm", "stage_um", "snr", "exposure_s"):
            if key in s.meta:
                bits.append(f"{key} {s.meta[key]}")
        om = s.meta.get("omnic") or {}
        for key in ("laser_mW", "n_scans", "objective"):
            if key in om:
                bits.append(f"{key} {om[key]}")
        return "\n".join(bits)

    def _on_selection_changed(self) -> None:
        # a highlighted spectrum is always shown (clicking one means "see it")
        self.file_list.blockSignals(True)
        for item in self.file_list.selectedItems():
            if item.checkState() != Qt.Checked:
                item.setCheckState(Qt.Checked)
        self.file_list.blockSignals(False)
        self.plot.request_redraw(self.render)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        self.plot.request_redraw(self.render)

    def _set_all_visible(self, on: bool) -> None:
        """Show all / Hide all: the entries the list shows (a name filter keeps the others as they are)."""
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.file_list.blockSignals(False)
        self.plot.request_redraw(self.render)

    def show_only_selected(self) -> None:
        sel = {id(it) for it in self.file_list.selectedItems()}
        if not sel:
            return
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Checked if id(it) in sel else Qt.Unchecked)
        self.file_list.blockSignals(False)
        self.plot.request_redraw(self.render)

    def _visible_spectra(self) -> List[Spectrum]:
        out = []
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.checkState() == Qt.Checked:
                s = self.library.get(it.data(Qt.UserRole))
                if s is not None:
                    out.append(s)
        return out

    def _selected_spectra(self) -> List[Spectrum]:
        """The shown spectra (what the plot draws)."""
        return self._visible_spectra()

    def _highlighted_spectra(self) -> List[Spectrum]:
        rows = sorted(self.file_list.row(it) for it in self.file_list.selectedItems())
        out = []
        for r in rows:
            s = self.library.get(self.file_list.item(r).data(Qt.UserRole))
            if s is not None:
                out.append(s)
        return out

    def _target_spectra(self, treat_all: bool = False) -> List[Spectrum]:
        hl = self._highlighted_spectra()
        return self._visible_spectra() if (treat_all or not hl) else hl

    def _on_list_menu(self, pos) -> None:
        if self.file_list.itemAt(pos) is None:
            return
        menu = QMenu(self)
        menu.addAction("Properties…", self.open_properties)
        menu.addAction("Rename…", self.rename_selected)
        menu.addAction("Duplicate", self.duplicate_selected)
        menu.addSeparator()
        menu.addAction("Show only these", self.show_only_selected)
        menu.addAction("Hide", lambda: [it.setCheckState(Qt.Unchecked) for it in self.file_list.selectedItems()])
        menu.addSeparator()
        menu.addAction("Remove", self.remove_selected)
        menu.exec(self.file_list.viewport().mapToGlobal(pos))

    def _library_changed(self) -> None:
        if self.on_library_changed is not None:
            self.on_library_changed()
        self.set_spectra([s.id for s in self.library.all()])

    # ---- Spectra tab actions ------------------------------------------
    def import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import spectra", "",
            "Spectra (*.txt *.dat *.csv *.xy *.asc *.spa *.mapx *.jdx *.dx);;Thermo OMNIC (*.spa *.mapx);;All files (*.*)")
        if paths:
            self._import(paths)

    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import every spectrum of a folder")
        if not folder:
            return
        exts = (".txt", ".dat", ".csv", ".xy", ".asc", ".spa", ".mapx", ".jdx", ".dx")
        paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))
        if not paths:
            QMessageBox.information(self, "Import folder", "No data file in that folder.")
            return
        self._import(paths)

    def _import(self, paths: List[str]) -> int:
        before = {s.id for s in self.library.all()}
        if self.import_paths is not None:
            added, _errors = self.import_paths(paths)
        else:
            from core.qt_library import load_spectra_from_path
            added = 0
            for p in paths:
                for s in load_spectra_from_path(p):
                    self.library.add(s)
                    added += 1
        self._library_changed()
        new_ids = [s.id for s in self.library.all() if s.id not in before]
        self._set_visible_ids(new_ids, only=False)
        self._append_log(f"imported {added} spectrum/spectra from {len(paths)} file(s)")
        return added

    def _set_visible_ids(self, ids: List[str], only: bool) -> None:
        wanted = set(ids)
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.data(Qt.UserRole) in wanted:
                it.setCheckState(Qt.Checked)
                it.setSelected(True)
            elif only:
                it.setCheckState(Qt.Unchecked)
                it.setSelected(False)
        self.file_list.blockSignals(False)
        self.plot.request_redraw(self.render)

    def paste_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication
        import tempfile
        from core.qt_library import _load_spectrum_from_path
        text = QApplication.clipboard().text()
        if not text.strip():
            QMessageBox.information(self, "Paste", "The clipboard holds no text.")
            return
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="clipboard_", delete=False, encoding="utf-8")
        try:
            tmp.write(text)
            tmp.close()
            s = _load_spectrum_from_path(tmp.name)
        except Exception as exc:
            QMessageBox.warning(self, "Paste", f"The clipboard text is not a spectrum table:\n{exc}")
            return
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        n = sum(1 for t in self.library.all() if t.title.startswith("clipboard"))
        s.title, s.path = f"clipboard{n + 1}", ""
        self.library.add(s)
        if self.on_undo_push is not None:
            self.on_undo_push(("add", [s.id]))
        self._library_changed()
        self._set_visible_ids([s.id], only=False)

    def rename_selected(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        hl = self._highlighted_spectra()
        if len(hl) != 1:
            QMessageBox.information(self, "Rename", "Highlight exactly one spectrum.")
            return
        s = hl[0]
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=s.title)
        if ok and name.strip() and name.strip() != s.title:
            if self.on_undo_push is not None:
                self.on_undo_push(("rename", s.id, s.title))
            s.title = name.strip()
            self._library_changed()

    def duplicate_selected(self) -> None:
        new_ids = []
        for s in self._highlighted_spectra():
            c = Spectrum(id=Spectrum.new_id(), title=f"{s.title}_copy", path=s.path, kind=s.kind,
                         x=np.array(s.x, float).copy(), y=np.array(s.y, float).copy(), df=s.df, meta=dict(s.meta), status="derived")
            self.library.add(c)
            new_ids.append(c.id)
        if new_ids:
            if self.on_undo_push is not None:
                self.on_undo_push(("add", new_ids))
            self._library_changed()
            self._set_visible_ids(new_ids, only=False)

    def remove_selected(self) -> None:
        hl = self._highlighted_spectra()
        if not hl:
            return
        order = [s.id for s in self.library.all()]
        batch = [(order.index(s.id), s) for s in hl]
        for _, s in batch:
            self.library.remove(s.id)
        if self.on_undo_push is not None:
            self.on_undo_push(("delete", batch))
        self._library_changed()

    def remove_all(self) -> None:
        """Empty the library (undoable, like Remove)."""
        n = len(self.library)
        if n == 0:
            return
        if QMessageBox.question(self, "Remove all", f"Remove all {n} spectra from the library? (Undo: Ctrl+Z)") != QMessageBox.Yes:
            return
        batch = [(i, s) for i, s in enumerate(self.library.all())]
        self.library.clear()
        if self.on_undo_push is not None:
            self.on_undo_push(("delete", batch))
        self._library_changed()

    def _apply_filter(self, text: Optional[str] = None) -> None:
        needle = (self.filter_edit.text() if text is None else text).strip().lower()
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            it.setHidden(bool(needle) and needle not in it.text().lower())

    def sort_spectra(self) -> None:
        mode = self.sort_combo.currentText()
        items = self.library.all()

        def laser(s):
            return float(s.meta.get("laser_nm") or (s.meta.get("omnic") or {}).get("laser_nm") or 0.0)

        def t_first(s):
            return str((s.meta.get("omnic") or {}).get("time_of_first_frame", "")) or str(s.meta.get("added_at", ""))

        if mode == "A → Z":
            items = sorted(items, key=lambda s: s.title.lower())
        elif mode == "Z → A":
            items = sorted(items, key=lambda s: s.title.lower(), reverse=True)
        elif mode == "Reverse order":
            items = list(reversed(items))
        elif mode == "By laser λ":
            items = sorted(items, key=laser)
        elif mode == "By acquisition time":
            items = sorted(items, key=t_first)
        elif mode == "By maximum":
            items = sorted(items, key=lambda s: float(np.nanmax(s.y)) if len(s.y) else 0.0)
        self.library.reorder([s.id for s in items])
        self._library_changed()

    def open_properties(self) -> None:
        from raman.qt_raman_dialogs import SpectrumPropertiesDialog
        hl = self._highlighted_spectra()
        if not hl:
            QMessageBox.information(self, "Properties", "Highlight one or more spectra.")
            return
        dlg = SpectrumPropertiesDialog(self, hl, self.styles)
        if dlg.exec():
            dlg.apply(self.styles)
            self.set_spectra([s.id for s in self.library.all()])

    def colour_by_property(self) -> None:
        import matplotlib
        prop = self.colour_prop_combo.currentText()
        shown = self._visible_spectra()
        if not shown:
            return
        vals = []
        for i, s in enumerate(shown):
            x, y = np.asarray(s.x, float), np.asarray(s.y, float)
            if prop == "maximum":
                v = float(np.nanmax(y))
            elif prop == "area":
                v = float(trapezoid(y, x))
            elif prop == "centre of gravity":
                pos = np.clip(y, 0, None)
                v = float(trapezoid(x * pos, x) / trapezoid(pos, x)) if trapezoid(pos, x) > 0 else float("nan")
            elif prop == "x at maximum":
                v = float(x[int(np.nanargmax(y))])
            elif prop == "mean":
                v = float(np.nanmean(y))
            elif prop == "laser λ":
                v = float(s.meta.get("laser_nm") or (s.meta.get("omnic") or {}).get("laser_nm") or float("nan"))
            else:
                v = float(i)
            vals.append(v)
        arr = np.array(vals, float)
        finite = np.isfinite(arr)
        lo, hi = (float(np.nanmin(arr)), float(np.nanmax(arr))) if finite.any() else (0.0, 1.0)
        cmap = matplotlib.colormaps["viridis"]
        for s, v in zip(shown, arr):
            t = 0.5 if not np.isfinite(v) or hi == lo else (v - lo) / (hi - lo)
            self.styles.setdefault(s.id, {})["color"] = matplotlib.colors.to_hex(cmap(t))
        self._append_log(f"coloured {len(shown)} spectra by {prop}: {lo:.4g} (dark) to {hi:.4g} (yellow)")
        self.set_spectra([s.id for s in self.library.all()])

    def reset_colours(self) -> None:
        for st in self.styles.values():
            st.pop("color", None)
        self.set_spectra([s.id for s in self.library.all()])

    # ==================================================================
    # Operations
    # ==================================================================
    def _refresh_ref_combo(self) -> None:
        current = self.op_ref_combo.currentData()
        self.op_ref_combo.blockSignals(True)
        self.op_ref_combo.clear()
        for s in self.library.all():
            self.op_ref_combo.addItem(s.title, s.id)
        if current is not None:
            idx = self.op_ref_combo.findData(current)
            if idx >= 0:
                self.op_ref_combo.setCurrentIndex(idx)
        self.op_ref_combo.blockSignals(False)

    def open_op(self, op: ro.Op, params: Optional[Dict[str, Any]] = None) -> None:
        """Show the parameter panel for an operation."""
        self._current_op = op
        self._end_pick()
        self.op_title.setText(op.label)
        self.op_tip.setText(op.tip)
        _clear_layout(self.op_form)
        self._param_widgets = {}
        values = dict(op.defaults())
        if params:
            values.update(params)
        ncols = 3
        for i, p in enumerate(op.params):
            r, c = divmod(i, ncols)
            lab = QLabel(p.label)
            if p.tip:
                lab.setToolTip(p.tip)
            cell = QHBoxLayout()
            cell.setSpacing(3)
            if p.kind == "choice":
                w = QComboBox()
                w.addItems(p.choices or [])
                w.setCurrentText(str(values.get(p.key, p.default)))
            elif p.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(values.get(p.key, p.default)))
            else:
                w = QLineEdit("" if values.get(p.key, p.default) in (None, "") else str(values.get(p.key, p.default)))
                if p.kind == "text":
                    w.setMinimumWidth(220)
            if p.tip:
                w.setToolTip(p.tip)
            cell.addWidget(w, 1)
            mode = ro.pick_mode(p)
            if mode and isinstance(w, QLineEdit):
                pick = QPushButton("Pick…")
                pick.setMinimumWidth(56)
                pick.setToolTip({"range": "Click the start and the end of each range on the plot; keep clicking for more ranges. "
                                          "Right-click, Esc or Done finishes.",
                                 "list": "Click once per value on the plot; keep clicking for more. Right-click, Esc or Done finishes.",
                                 "x": "Click on the plot to fill this field"}[mode])
                pick.clicked.connect(lambda _=False, e=w, m=mode, b=pick: self._start_pick(e, m, b))
                cell.addWidget(pick)
                w.textChanged.connect(lambda *_: self.plot.request_redraw(self.render))   # the overlay follows the text
            self.op_form.addWidget(lab, r, 2 * c)
            self.op_form.addLayout(cell, r, 2 * c + 1)
            self._param_widgets[p.key] = w
        # the scaled-subtraction slider
        if op.key == "scaled_subtraction":
            r = (len(op.params) + ncols - 1) // ncols
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 300)
            slider.setValue(100)
            slider.setToolTip("k from 0 to 3; the preview follows the slider")

            def on_slide(v):
                self._param_widgets["k"].setText(f"{v / 100:.2f}")
                self.run_current_op(preview=True, quiet=True)
            slider.valueChanged.connect(on_slide)
            self.op_form.addWidget(QLabel("k slider"), r, 0)
            self.op_form.addWidget(slider, r, 1, 1, 5)
        is_ref = op.arity == "reference"
        self.op_ref_label.setVisible(is_ref)
        self.op_ref_combo.setVisible(is_ref)
        self._refresh_ref_combo()
        self.op_suffix.setText(op.suffix)
        self.op_suffix.setEnabled(op.arity in ("each", "reference"))
        self.op_replace.setEnabled(op.arity != "table")
        self.op_preview_btn.setEnabled(op.arity != "table")
        self.op_apply_btn.setText("Compute" if op.arity == "table" else "Apply")
        self._show_panel(self.op_panel)
        tab_index = self._tab_index_for_op(op)
        if tab_index >= 0 and self.ribbon.currentIndex() != tab_index:
            self.ribbon.setCurrentIndex(tab_index)

    def _tab_index_for_op(self, op: ro.Op) -> int:
        name = {"process": "Process", "transform": "Transform", "analyze": "Analyze"}.get(op.tab, "")
        for i in range(self.ribbon.count()):
            if self.ribbon.tabText(i) == name:
                return i
        return -1

    def close_op_panel(self) -> None:
        self._current_op = None
        self._end_pick()
        self._hide_panel(self.op_panel)
        if self._preview is not None:
            self._preview = None
            self.plot.request_redraw(self.render)

    def _read_params(self) -> Dict[str, Any]:
        raw = {}
        for key, w in self._param_widgets.items():
            if isinstance(w, QComboBox):
                raw[key] = w.currentText()
            elif isinstance(w, QCheckBox):
                raw[key] = w.isChecked()
            else:
                raw[key] = w.text()
        return raw

    def _spec_in(self, s: Spectrum) -> ro.SpecIn:
        return ro.SpecIn(np.asarray(s.x, float), np.asarray(s.y, float), s.title, dict(s.meta), id=s.id)

    def _reference_spec(self) -> Optional[ro.SpecIn]:
        sid = self.op_ref_combo.currentData()
        s = self.library.get(sid) if sid else None
        return self._spec_in(s) if s is not None else None

    def run_current_op(self, preview: bool, quiet: bool = False) -> None:
        op = self._current_op
        if op is None:
            return
        targets = self._target_spectra(self.op_treat_all.isChecked())
        if not targets:
            if not quiet:
                QMessageBox.information(self, op.label, "Show or highlight at least one spectrum.")
            return
        try:
            res = ro.run_op(op, [self._spec_in(s) for s in targets], self._read_params(),
                            reference=self._reference_spec() if op.arity == "reference" else None,
                            suffix=self.op_suffix.text().strip() or None)
        except Exception as exc:
            if not quiet:
                QMessageBox.warning(self, op.label, str(exc))
            return
        if op.arity == "table":
            if op.key == "peak_table":
                self._remember_found_peaks(targets, res.rows or [])
            self._show_table(res.columns or [], res.rows or [], res.text)
            self._append_log(f"{op.label}: {res.text or len(res.rows or [])} rows")
            return
        if preview:
            self._preview = res.spectra
            self.plot.request_redraw(self.render)
            return
        self._add_results(res.spectra, targets, op, replace=self.op_replace.isChecked())

    def _add_results(self, spectra, sources: List[Spectrum], op: ro.Op, replace: bool) -> List[str]:
        new_ids = []
        for x, y, title in spectra:
            src = sources[0]
            derived = Spectrum(id=Spectrum.new_id(), title=title, path="", kind=src.kind,
                               x=np.asarray(x, float), y=np.asarray(y, float), df=None,
                               meta={"derived": f"raman:{op.key}", "params": self._read_params() if self._current_op is op else {},
                                     "sources": [s.title for s in sources], "laser_nm": src.meta.get("laser_nm")},
                               status="derived")
            if derived.meta.get("laser_nm") is None:
                derived.meta.pop("laser_nm", None)
            self.library.add(derived)
            new_ids.append(derived.id)
        if new_ids and self.on_derived_added is not None:
            self.on_derived_added(new_ids)
        self._preview = None
        self._library_changed()
        if replace:
            src_ids = {s.id for s in sources}
            self.file_list.blockSignals(True)
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it.data(Qt.UserRole) in src_ids:
                    it.setCheckState(Qt.Unchecked)
                    it.setSelected(False)
            self.file_list.blockSignals(False)
        self._set_visible_ids(new_ids, only=False)
        self._append_log(f"{op.label}: {len(new_ids)} new spectrum/spectra" + (" (originals hidden)" if replace else ""))
        return new_ids

    def _show_table(self, columns: List[str], rows: List[List[Any]], note: str = "") -> None:
        t = self.results_table
        t.setUpdatesEnabled(False)   # thousands of rows (a map): fill without a repaint per cell
        try:
            t.setRowCount(0)
            t.setColumnCount(len(columns))
            t.setHorizontalHeaderLabels(columns)
            t.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, v in enumerate(row):
                    it = QTableWidgetItem(f"{v:.6g}" if isinstance(v, float) else str(v))
                    if c:
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    t.setItem(r, c, it)
            if len(rows) <= 2000:
                t.resizeColumnsToContents()
        finally:
            t.setUpdatesEnabled(True)
        self.results_note.setText(note)
        self.results_tabs.setCurrentIndex(0)
        sizes = self.centre_splitter.sizes()
        if rows and len(sizes) == 2 and sum(sizes) > 400 and sizes[1] < 220:   # a table deserves a few visible rows
            self.centre_splitter.setSizes([sum(sizes) - 220, 220])

    def _results_text(self, sep: str = "\t") -> str:
        t = self.results_table
        cols = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        lines = [sep.join(cols)]
        for r in range(t.rowCount()):
            lines.append(sep.join(t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())))
        return "\n".join(lines)

    def copy_results(self) -> None:
        from PySide6.QtWidgets import QApplication
        if self.results_table.columnCount():
            QApplication.clipboard().setText(self._results_text())

    def save_results_csv(self) -> None:
        if not self.results_table.columnCount():
            QMessageBox.information(self, "Results", "No result table yet (Analyze tab).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save results", "results.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(self._results_text(","))

    def _append_log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    # ---- pick on plot --------------------------------------------------
    def _start_pick(self, edit: QLineEdit, mode: str, button: Optional[QPushButton] = None) -> None:
        """Pick… starts a pick session on the plot (the button again ends it).
        'x': one click fills the field. 'range': two clicks per range, each
        finished range is appended (';' separated). 'list': one click per
        value, appended (',' separated). A session ends with the button, a
        right-click on the plot, Esc, or when the panel changes."""
        if self._pick_target is not None:
            same = self._pick_target[0] is edit
            self._end_pick()
            if same:
                return
        self._pick_target = (edit, mode, [], button)
        if button is not None:
            button.setText("Done")
            button.setObjectName("Primary")
            button.style().unpolish(button)
            button.style().polish(button)
        if self.plot.toolbar.mode:
            m = str(self.plot.toolbar.mode)
            if "zoom" in m:
                self.plot.toolbar.zoom()
            elif "pan" in m:
                self.plot.toolbar.pan()
        self.plot.canvas.setFocus()
        self.plot.canvas.setCursor(Qt.CrossCursor)
        self._pick_status()

    def _pick_status(self) -> None:
        if self._pick_target is None:
            return
        edit, mode, buf, _button = self._pick_target
        if mode == "x":
            msg = "Click the x position on the plot."
        elif mode == "range":
            n = len(ro.parse_ranges(edit.text(), lenient=True))
            msg = ("Click the END of the range…" if buf else "Click the START of a range…") + \
                  f"   {n} range{'s' if n != 1 else ''} so far; right-click, Esc or Done to finish."
        else:
            n = len(ro.parse_positions(edit.text()))
            msg = f"Click each position…   {n} so far; right-click, Esc or Done to finish."
        self.cursor_label.setText(msg)

    def _end_pick(self) -> None:
        if self._pick_target is None:
            return
        _edit, _mode, _buf, button = self._pick_target
        self._pick_target = None   # session over
        if button is not None:
            try:
                button.setText("Pick…")
                button.setObjectName("")
                button.style().unpolish(button)
                button.style().polish(button)
            except RuntimeError:
                pass   # the form was rebuilt and the button is gone
        self.plot.canvas.unsetCursor()
        self.cursor_label.setText("")
        self.plot.request_redraw(self.render)

    def _handle_pick(self, xdata: float, *, button: int = 1, dblclick: bool = False) -> bool:
        """A click on the plot during a pick session; True when consumed."""
        if self._pick_target is None:
            return False
        if button == 3:            # right-click ends the session
            self._end_pick()
            return True
        if dblclick:               # the second press of a double click is not a new point
            return True
        edit, mode, buf, _button = self._pick_target
        if mode == "x":
            edit.setText(f"{xdata:.2f}")
            self._end_pick()
            return True
        if mode == "list":
            cur = edit.text().strip().rstrip(",;")
            edit.setText((cur + ", " if cur else "") + f"{xdata:.2f}")
        else:
            if buf and abs(xdata - buf[0]) <= 1e-9 * max(1.0, abs(xdata)):
                return True        # the same point twice: wait for a different second point
            buf.append(xdata)
            if len(buf) == 2:
                lo, hi = sorted(buf)
                buf.clear()
                cur = edit.text().strip().rstrip(";,")
                edit.setText((cur + "; " if cur else "") + f"{lo:.1f}-{hi:.1f}")
        self._pick_status()
        self.plot.request_redraw(self.render)
        return True

    def _draw_pick_overlays(self) -> None:
        """Shade the ranges and mark the points of the open operation panel
        (typed or picked) so the user sees what the operation will use."""
        op = self._current_op
        if op is None or not self.op_panel.isVisibleTo(self):
            return
        axes = self.plot.figure.get_axes()
        if not axes:
            return
        spans, points = [], []
        for p in op.params:
            w = self._param_widgets.get(p.key)
            mode = ro.pick_mode(p)
            if mode is None or not isinstance(w, QLineEdit):
                continue
            if mode == "range":
                spans += ro.parse_ranges(w.text(), lenient=True)
            elif mode == "list":
                points += ro.parse_positions(w.text())
            else:
                v = _to_float(w.text())
                if v is not None:
                    points.append(v)
        pending = list(self._pick_target[2]) if (self._pick_target is not None and self._pick_target[1] == "range") else []
        for ax in axes:
            for lo, hi in spans:
                ax.axvspan(lo, hi, color=PICK_COLOR, alpha=0.14, lw=0, zorder=0.5)
            for xv in points:
                ax.axvline(xv, color=PICK_COLOR, lw=0.9, ls=":", alpha=0.9, zorder=0.6)
            for xv in pending:
                ax.axvline(xv, color=PICK_COLOR, lw=0.9, ls="--", alpha=0.9, zorder=0.6)

    # ---- sequence ------------------------------------------------------
    def add_current_op_to_sequence(self) -> None:
        op = self._current_op
        if op is None:
            return
        try:
            params = ro.parse_params(op, self._read_params())
        except ValueError as exc:
            QMessageBox.warning(self, op.label, str(exc))
            return
        ref_title = self.op_ref_combo.currentText() if op.arity == "reference" else None
        step = ro.Step(op.key, params, reference_title=ref_title,
                       options={"suffix": self.op_suffix.text().strip() or op.suffix})
        self.sequence.append(step)
        self._refresh_seq_list()
        self._append_log(f"sequence + {step.describe()}")

    def _refresh_seq_list(self) -> None:
        self.seq_list.clear()
        for i, st in enumerate(self.sequence, 1):
            self.seq_list.addItem(f"{i}. {st.describe()}")
        n = len(self.sequence)
        self.seq_count_label.setText("no step yet" if n == 0 else f"{n} step" + ("s" if n > 1 else ""))
        self.seq_empty.setVisible(n == 0)

    def seq_remove(self) -> None:
        rows = sorted({i.row() for i in self.seq_list.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.sequence):
                del self.sequence[r]
        self._refresh_seq_list()

    def seq_clear(self) -> None:
        self.sequence = []
        self._refresh_seq_list()

    def seq_save(self) -> None:
        if not self.sequence:
            QMessageBox.information(self, "Sequence", "The sequence is empty: open an operation and use 'Add to sequence'.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save sequence", "sequence.json", "PRISM sequence (*.json)")
        if path:
            ro.save_sequence(self.sequence, path)

    def seq_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load sequence", "", "PRISM sequence (*.json)")
        if not path:
            return
        try:
            self.sequence = ro.load_sequence(path)
        except Exception as exc:
            QMessageBox.warning(self, "Sequence", str(exc))
            return
        self._refresh_seq_list()

    def seq_run(self, mode: str = "selected") -> None:
        if not self.sequence:
            QMessageBox.information(self, "Sequence", "The sequence is empty.")
            return
        if mode == "folder":
            folder = QFileDialog.getExistingDirectory(self, "Folder of spectra to process")
            if not folder:
                return
            exts = (".txt", ".dat", ".csv", ".xy", ".asc", ".spa", ".jdx", ".dx")
            paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))
            if not paths:
                QMessageBox.information(self, "Sequence", "No data file in that folder.")
                return
            before = {s.id for s in self.library.all()}
            self._import(paths)
            targets = [s for s in self.library.all() if s.id not in before]
        else:
            targets = self._target_spectra(treat_all=(mode == "visible"))
        if not targets:
            QMessageBox.information(self, "Sequence", "Nothing to process.")
            return

        def lookup(title):
            for s in self.library.all():
                if s.title == title:
                    return self._spec_in(s)
            return None

        try:
            final, results = ro.run_sequence(self.sequence, [self._spec_in(s) for s in targets],
                                             reference_lookup=lookup, log=self._append_log)
        except Exception as exc:
            QMessageBox.warning(self, "Sequence", str(exc))
            return
        last_table = next((r for r in reversed(results) if r.columns), None)
        if last_table is not None:
            self._show_table(last_table.columns, last_table.rows or [], last_table.text)
        if any(r.spectra for r in results):
            last_op = ro.OPS_BY_KEY[next(st.op for st in reversed(self.sequence) if ro.OPS_BY_KEY[st.op].arity != "table")]
            self._add_results([(s.x, s.y, s.title) for s in final], targets, last_op, replace=False)
        self._append_log(f"sequence finished on {len(targets)} spectrum/spectra")

    # ---- identify ------------------------------------------------------
    def add_selected_to_library(self) -> None:
        import raman.raman_library as rl
        from raman.qt_raman_dialogs import AddToLibraryDialog
        hl = self._highlighted_spectra()
        if not hl:
            QMessageBox.information(self, "Reference library", "Highlight the spectra to add.")
            return
        dlg = AddToLibraryDialog(self, [s.title for s in hl])
        if not dlg.exec():
            return
        v = dlg.values()
        entries = rl.load_library()
        for s in hl:
            entries.append(rl.entry_from_spectrum(v["name"] if (v["name"] and len(hl) == 1) else s.title, s.x, s.y,
                                                  laser_nm=v["laser_nm"] or s.meta.get("laser_nm"), source=v["source"], notes=v["notes"],
                                                  extra={"origin_path": s.path}))
        rl.save_library(entries)
        self._append_log(f"added {len(hl)} spectrum/spectra to the reference library ({len(entries)} entries)")

    def quick_similarity(self) -> None:
        import raman.raman_library as rl
        hl = self._highlighted_spectra()
        if len(hl) != 1:
            QMessageBox.information(self, "Quick similarity", "Highlight exactly one spectrum (the query).")
            return
        entries = rl.load_library()
        if not entries:
            QMessageBox.information(self, "Quick similarity", "Your reference library is empty (Identify › Add selected to my library…).")
            return
        q = hl[0]
        ranked = rl.rank_library(q.x, q.y, entries, how=self.quick_sim_combo.currentData(), top_n=50)
        rows = [[r["name"], round(r["score"], 4), r["meta"].get("laser_nm") or "", r["meta"].get("source", "")] for r in ranked]
        self._show_table(["Reference", "Score", "λ (nm)", "Source"], rows, f"query: {q.title}; 1 = identical shape")

    # ==================================================================
    # Export
    # ==================================================================
    def export_text_files(self) -> None:
        targets = self._target_spectra()
        if not targets:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the exported spectra")
        if not folder:
            return
        n = 0
        for s in targets:
            data = np.column_stack([np.asarray(s.x, float), np.asarray(s.y, float)])
            np.savetxt(os.path.join(folder, f"{s.title}.txt"), data, delimiter="\t", header=f"{s.title} (exported from PRISM)", comments="# ")
            n += 1
        self._append_log(f"exported {n} text file(s) to {folder}")

    def export_combined_csv(self) -> None:
        shown = self._visible_spectra()
        if not shown:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Combined CSV", "spectra.csv", "CSV (*.csv)")
        if not path:
            return
        from core.spectrum_math import _common_grid, _on_grid
        try:
            grid = _common_grid([(s.x, s.y) for s in shown])
        except ValueError as exc:
            QMessageBox.warning(self, "Combined CSV", str(exc))
            return
        cols = [grid] + [_on_grid(s.x, s.y, grid) for s in shown]
        header = ",".join(["x"] + [s.title for s in shown])
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")
        self._append_log(f"combined CSV: {len(shown)} spectra on {grid.size} points -> {path}")

    def export_jcamp(self) -> None:
        from core.io_universal import write_jcamp_dx
        targets = self._target_spectra()
        if not targets:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the JCAMP-DX files")
        if not folder:
            return
        for s in targets:
            write_jcamp_dx(os.path.join(folder, f"{s.title}.jdx"), s.x, s.y, title=s.title,
                           xunits="1/CM" if "Raman" in self.data_unit_combo.currentText() or "Wavenumber" in self.data_unit_combo.currentText() else "NANOMETERS")
        self._append_log(f"exported {len(targets)} JCAMP-DX file(s) to {folder}")

    def export_all_formats(self) -> None:
        base, _ = QFileDialog.getSaveFileName(self, "Export plot as…", "", "Image (*.png)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        dpi = int(_to_float(self.export_dpi_edit.text(), 300) or 300)
        errors, done = [], []
        for ext, cb in (("png", self.export_png_check), ("svg", self.export_svg_check), ("pdf", self.export_pdf_check),
                        ("eps", self.export_eps_check), ("tif", self.export_tif_check)):
            if cb.isChecked():
                try:
                    self.plot.figure.savefig(f"{base}.{ext}", dpi=dpi)
                    done.append(ext)
                except Exception as exc:
                    errors.append(f"{ext.upper()}: {exc}")
        if errors:
            QMessageBox.critical(self, "Export", "Some exports failed:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Export", f"Plot exported ({', '.join(done)}).")

    def copy_plot_to_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        self.plot.canvas.draw()
        buf = self.plot.canvas.buffer_rgba()
        w, h = self.plot.canvas.get_width_height(physical=True)
        img = QImage(bytes(buf), w, h, QImage.Format_RGBA8888).copy()
        QApplication.clipboard().setImage(img)
        self.cursor_label.setText("Plot copied to the clipboard.")

    def copy_data_to_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication
        spectra = self._target_spectra()
        if not spectra:
            return
        same_grid = all(len(s.x) == len(spectra[0].x) and np.allclose(s.x, spectra[0].x) for s in spectra)
        lines = []
        if same_grid:
            lines.append("\t".join(["x"] + [s.title for s in spectra]))
            for i in range(len(spectra[0].x)):
                lines.append("\t".join([f"{spectra[0].x[i]:.6g}"] + [f"{s.y[i]:.6g}" for s in spectra]))
        else:
            lines.append("\t".join(f"{s.title}_x\t{s.title}_y" for s in spectra))
            n = max(len(s.x) for s in spectra)
            for i in range(n):
                lines.append("\t".join(f"{s.x[i]:.6g}\t{s.y[i]:.6g}" if i < len(s.x) else "\t" for s in spectra))
        QApplication.clipboard().setText("\n".join(lines))
        self.cursor_label.setText(f"{len(spectra)} spectrum/spectra copied as text.")

    def print_plot(self) -> None:
        try:
            from PySide6.QtPrintSupport import QPrintDialog, QPrinter
            from PySide6.QtGui import QImage, QPainter
        except ImportError:
            QMessageBox.information(self, "Print", "Printing support is not available in this build.")
            return
        printer = QPrinter(QPrinter.HighResolution)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.plot.canvas.draw()
        buf = self.plot.canvas.buffer_rgba()
        w, h = self.plot.canvas.get_width_height(physical=True)
        img = QImage(bytes(buf), w, h, QImage.Format_RGBA8888).copy()
        painter = QPainter(printer)
        rect = painter.viewport()
        scaled = img.size()
        scaled.scale(rect.size(), Qt.KeepAspectRatio)
        painter.setViewport(rect.x(), rect.y(), scaled.width(), scaled.height())
        painter.setWindow(img.rect())
        painter.drawImage(0, 0, img)
        painter.end()

    # ==================================================================
    # Peak labels / table, cursor, annotations
    # ==================================================================
    def _peak_threshold(self) -> float:
        v = _to_float(self.peak_threshold_edit.text(), 5.0)
        return max(0.0, min(float(v if v is not None else 5.0), 100.0)) / 100.0

    def peak_rows_for(self, x: np.ndarray, y: np.ndarray):
        from raman.raman_tools import peak_table
        xr = self._effective_xlim(self.plot.figure.get_axes()[0]) if self.plot.figure.get_axes() else None
        return peak_table(x, y, threshold_frac=self._peak_threshold(), x_range=xr)

    def _draw_peak_labels(self, placed: list, spectra: Optional[List[Spectrum]] = None) -> None:
        if not self.peak_labels_check.isChecked():
            return
        # a spectrum whose Peaks & FWHM result is marked on the plot keeps those marks only
        skip = set()
        if spectra is not None and len(spectra) == len(placed) and self.found_peaks_check.isChecked():
            skip = {i for i, s in enumerate(spectra) if s.id in self.found_peaks}
        if len(placed) > PEAK_LABEL_MAX:
            self._note_on_axes(placed[0][0], f"peak labels hidden above {PEAK_LABEL_MAX} spectra")
            return
        content = self.peak_content_combo.currentText()
        digits = int(self.peak_digits_combo.currentText())
        angle = {"90°": 90, "45°": 45, "0°": 0}[self.peak_angle_combo.currentText()]
        for i, (ax, x, y, _label) in enumerate(placed):
            if i in skip:
                continue
            try:
                rows = self.peak_rows_for(x, y)
            except Exception:
                continue
            for r in rows:
                if content == "x":
                    text = f"{r.position:.{digits}f}"
                elif content == "y":
                    text = f"{r.height:.{max(digits, 0)}f}"
                else:
                    text = f"{r.position:.{digits}f}, {r.height:.{digits}f}"
                ax.annotate(text, xy=(r.position, r.height), xytext=(0, 4), textcoords="offset points",
                            ha="center" if angle else "left", va="bottom", fontsize=7, rotation=angle, color="#333333")

    def _remember_found_peaks(self, targets: List[Spectrum], rows: List[List[Any]]) -> None:
        """Keep the Peaks & FWHM result per spectrum so the plot can mark it
        (Display › Found peaks)."""
        by_title: Dict[str, list] = {}
        for row in rows:
            by_title.setdefault(str(row[0]), []).append((float(row[1]), float(row[2]), str(row[-1])))
        self.found_peaks = {s.id: by_title[s.title] for s in targets if s.title in by_title}
        self.plot.request_redraw(self.render)

    def _draw_found_peaks(self, placed: list, spectra: List[Spectrum]) -> None:
        if not self.found_peaks or not self.found_peaks_check.isChecked() or len(placed) != len(spectra):
            return
        digits = int(self.peak_digits_combo.currentText())
        angle = {"90°": 90, "45°": 45, "0°": 0}[self.peak_angle_combo.currentText()]
        for (ax, x, y, label), s in zip(placed, spectra):
            rows = self.found_peaks.get(s.id)
            if not rows:
                continue
            xs, ys = np.asarray(x, float), np.asarray(y, float)
            order = np.argsort(xs, kind="mergesort")
            for pos, _height, source in rows:
                if not (xs.min() <= pos <= xs.max()):
                    continue
                yv = float(np.interp(pos, xs[order], ys[order]))
                manual = source == "manual"
                color = PICK_COLOR if manual else self._drawn_colors.get(label, "#444444")
                ax.plot([pos], [yv], marker="D" if manual else "v", ms=5 if manual else 4, ls="none", color=color, zorder=4)
                ax.annotate(f"{pos:.{digits}f}", xy=(pos, yv), xytext=(0, 6), textcoords="offset points",
                            ha="center" if angle else "left", va="bottom", fontsize=7, rotation=angle,
                            color=PICK_COLOR if manual else "#333333")

    def open_peak_table(self) -> None:
        from raman.qt_raman_dialogs import PeakTableDialog
        spectra = self._target_spectra()
        if not spectra:
            QMessageBox.information(self, "Peak table", "Show at least one spectrum.")
            return
        if self._peak_table_dialog is None:
            self._peak_table_dialog = PeakTableDialog(self)
        self._peak_table_dialog.set_spectra(spectra, self._peak_threshold())
        self._peak_table_dialog.show()
        self._peak_table_dialog.raise_()

    def _on_cursor_toggled(self, on: bool) -> None:
        if not on:
            self._remove_cursor_artists()
            self.cursor_label.setText("")
            self.plot.canvas.draw_idle()

    def _remove_cursor_artists(self) -> None:
        for a in self._cursor_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._cursor_artists = []

    def _on_cursor_move(self, event) -> None:
        if not self.cursor_check.isChecked() or event.inaxes is None or event.xdata is None:
            return
        if self.plot.toolbar.mode:
            return
        x0 = float(event.xdata)
        self._remove_cursor_artists()
        lines = []
        for ax in self.plot.figure.get_axes():
            self._cursor_artists.append(ax.axvline(x0, color="#8a3033", lw=0.8, ls="--", alpha=0.8))
            for line in ax.get_lines():
                if len(lines) >= 12:        # the readout shows twelve values; hundreds would stall the mouse
                    break
                lab = line.get_label()
                if not lab or lab.startswith("_"):
                    continue
                xd, yd = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
                if len(xd) < 2 or x0 < xd.min() or x0 > xd.max():
                    continue
                order = np.argsort(xd)
                yv = float(np.interp(x0, xd[order], yd[order]))
                lines.append(f"{lab}: {yv:.4g}")
                self._cursor_artists.append(ax.plot([x0], [yv], "o", ms=4, color=line.get_color())[0])
        self.cursor_label.setText(f"x = {x0:.2f}   " + "   ".join(lines[:12]))
        self.plot.canvas.draw_idle()

    def _on_plot_key(self, event) -> None:
        if getattr(event, "key", None) in ("escape", "esc"):
            self._end_pick()

    def _on_plot_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if self.plot.toolbar.mode:  # zoom/pan tool active — don't hijack the click
            return
        btn = int(getattr(event, "button", None) or 1)
        if self._handle_pick(float(event.xdata), button=btn, dblclick=bool(getattr(event, "dblclick", False))):
            return
        if not self.annotate_check.isChecked():
            return
        self.annotations.append({"x": float(event.xdata), "y": float(event.ydata)})
        self.plot.request_redraw(self.render)

    def _clear_annotations(self) -> None:
        self.annotations.clear()
        self.plot.request_redraw(self.render)

    def _draw_annotations(self) -> None:
        axes = self.plot.figure.get_axes()
        if not axes or not self.annotations:
            return
        for ann in self.annotations:
            for ax in axes:
                xlo, xhi = sorted(ax.get_xlim())
                if not (xlo <= ann["x"] <= xhi):
                    continue
                ax.plot([ann["x"]], [ann["y"]], marker="o", ms=6, mew=0, color="#8a3033", zorder=5)
                ax.annotate(f"{ann['x']:.1f}", xy=(ann["x"], ann["y"]), xytext=(0, 12), textcoords="offset points",
                            ha="center", fontsize=8, color="#8a3033", arrowprops=dict(arrowstyle="-", color="#8a3033", lw=0.8))
                break

    # ==================================================================
    # Rendering
    # ==================================================================
    def _apply_smoothing(self, y: np.ndarray) -> np.ndarray:
        method = self.smooth_combo.currentText()
        if method == "None":
            return y
        from processing.calc_science import moving_average
        if method.startswith("MA"):
            return moving_average(y, 5)
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(y, window_length=7, polyorder=2, mode="interp")
        except Exception:
            return moving_average(y, 5)

    def _color_from_scheme(self, i: int, label: str, total: int):
        scheme = self.color_combo.currentText()
        if scheme == "Matplotlib cycle":
            return None
        if scheme == "Distinct":
            return colorsys.hsv_to_rgb(i / max(total, 1), 0.75, 0.9)
        if scheme == "Hash by name":
            return colorsys.hsv_to_rgb((hash(label) % 360) / 360.0, 0.70, 0.9)
        if scheme == "Monochrome":
            return "#3b82f6"
        import matplotlib
        name = {"Viridis": "viridis", "Plasma": "plasma", "Cividis": "cividis", "Rainbow": "rainbow", "Tab10": "tab10", "Set2": "Set2"}.get(scheme)
        if name:
            cmap = matplotlib.colormaps[name]
            if name in ("tab10", "Set2"):
                return cmap(i % cmap.N)
            return cmap(0.1 + 0.8 * (i / max(total - 1, 1)))
        return LINE_COLORS[i % len(LINE_COLORS)]

    def _line_kwargs(self) -> dict:
        lw = _to_float(self.linewidth_combo.currentText(), 1.0) or 1.0
        mode = self.marker_combo.currentText()
        kw = {"lw": lw}
        if mode == "Lines + points":
            kw.update(marker="o", ms=max(2.0, 2.5 * lw))
        elif mode == "Points":
            kw.update(marker="o", ms=max(2.0, 2.5 * lw), ls="none")
        return kw

    def _note_on_axes(self, ax, text: str) -> None:
        notes = getattr(self, "_notes_drawn", None)
        if notes is None:
            notes = self._notes_drawn = {}
        k = notes.get(id(ax), 0)
        ax.text(0.99, 0.98 - 0.05 * k, text, transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#888888")
        notes[id(ax)] = k + 1

    def _place_legend(self, ax, n_entries: int = 0) -> None:
        mode = self.legend_combo.currentText()
        if mode == "No legend":
            return
        if n_entries > LEGEND_MAX:
            self._note_on_axes(ax, f"{n_entries} spectra: legend hidden above {LEGEND_MAX}")
            return
        fs = int(self.font_combo.currentText()) - 2
        if mode == "Legend outside":
            ax.legend(fontsize=fs, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        else:
            ax.legend(fontsize=fs)

    def _display_x(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The x axis in the chosen display unit (data unchanged)."""
        target = self.xunit_combo.currentText()
        if target == X_UNIT_MEASURED or target == self.data_unit_combo.currentText():
            return x, y
        from raman.raman_tools import convert_x_axis
        try:
            return convert_x_axis(x, y, self.data_unit_combo.currentText(), target, laser_nm=_to_float(self.laser_edit.text()))
        except ValueError:
            return x, y

    def _prepare(self, s: Spectrum) -> Tuple[np.ndarray, np.ndarray, str, dict]:
        x, y = np.asarray(s.x, float), np.asarray(s.y, float)
        st = self.styles.get(s.id, {})
        norm_xmin, norm_xmax = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        if self.norm_check.isChecked():
            mask = np.isfinite(x) & np.isfinite(y)
            if norm_xmin is not None:
                mask &= x >= norm_xmin
            if norm_xmax is not None:
                mask &= x <= norm_xmax
            if mask.sum() > 1:
                area = trapezoid(y[mask], x[mask])
                if abs(area) > 1e-12:
                    y = y / area * 100
        y = self._apply_smoothing(y)
        y = y * float(st.get("scale", 1.0)) + float(st.get("offset", 0.0))
        x, y = self._display_x(x, y)
        return x, y, st.get("label") or s.title, st

    def _style_kwargs(self, i: int, label: str, total: int, st: dict) -> dict:
        kw = self._line_kwargs()
        if total > MANY_SPECTRA:   # hundreds of map pixels: thin plain lines or the draw takes minutes
            kw = {"lw": min(float(kw.get("lw", 1.0)), 0.6)}
        color = st.get("color") or self._color_from_scheme(i, label, total)
        if color is not None:
            kw["color"] = color
        if st.get("ls") and "ls" not in kw:
            kw["ls"] = LINE_STYLES.get(st["ls"], st["ls"])
        if st.get("lw"):
            kw["lw"] = float(st["lw"])
        return kw

    def _finish_axes(self, ax, fs: int) -> None:
        ax.tick_params(labelsize=fs)
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.title.set_size(fs + 1)
        if self.logy_check.isChecked():
            try:
                ax.set_yscale("log")
            except ValueError:
                pass
        if self.flipx_check.isChecked():
            ax.invert_xaxis()
        ax.grid(True, alpha=0.3) if self.grid_check.isChecked() else ax.grid(False)

    def render(self) -> None:
        """Redraw everything; a wait cursor while hundreds of spectra draw."""
        n = sum(1 for i in range(self.file_list.count()) if self.file_list.item(i).checkState() == Qt.Checked)
        if n <= MANY_SPECTRA:
            self._render_now()
            return
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._render_now()
        finally:
            QApplication.restoreOverrideCursor()

    def _render_now(self) -> None:
        spectra = self._selected_spectra()
        fig = self.plot.figure
        fig.clf()
        self._cursor_artists = []
        self._notes_drawn = {}
        self.visible_count_label.setText(f"{len(spectra)} shown" if spectra else "")
        if not spectra and not self._preview:
            self.plot.canvas.draw_idle()
            return
        fs = int(self.font_combo.currentText())
        global_title = self.title_edit.text().strip()
        xlabel = self.x_title_edit.text() if self.xunit_combo.currentText() == X_UNIT_MEASURED else self.xunit_combo.currentText()
        prepared = [self._prepare(s) for s in spectra]
        self._drawn_colors = {}
        placed = []

        if self.diff_check.isChecked() and len(prepared) == 2:
            (xa, ya, la, _), (xb, yb, lb, _) = prepared
            yb_on_a = np.interp(xa, xb, yb)
            ax = fig.add_subplot(111)
            ax.plot(xa, ya, lw=1, label=la)
            ax.plot(xa, yb_on_a, lw=1, label=lb)
            ax.plot(xa, ya - yb_on_a, lw=1.4, color="black", label=f"{la} − {lb}")
            ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(self.y_title_edit.text())
            if not self.no_title_check.isChecked():
                ax.set_title(global_title or f"Difference: {la} − {lb}")
            self._place_legend(ax)
            self._finish_axes(ax, fs)
            placed.append((ax, xa, ya - yb_on_a, "diff"))
        elif self.mode_separate.isChecked() and prepared:
            axes = np.atleast_1d(fig.subplots(1, len(prepared), sharey=True)).ravel()
            for i, (ax, (x, y, label, st)) in enumerate(zip(axes, prepared)):
                line = ax.plot(x, y, label=label, **self._style_kwargs(i, label, len(prepared), st))[0]
                self._drawn_colors[label] = line.get_color()
                placed.append((ax, x, y, label))
                ax.set_xlabel(xlabel)
                if not self.no_title_check.isChecked():
                    ax.set_title(global_title or label)
                if i == 0 and not self.no_yticks_check.isChecked():
                    ax.set_ylabel(self.y_title_edit.text())
                else:
                    ax.tick_params(labelleft=False)
                if self.zero_line_check.isChecked():
                    ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
                self._finish_axes(ax, fs)
        else:
            ax = fig.add_subplot(111)
            stacked = self.mode_stacked.isChecked()
            delta = 0.0
            if stacked and prepared:
                shift_factor = _to_float(self.shift_edit.text(), 0.5)
                shift_factor = 0.5 if shift_factor is None else shift_factor
                norm_xmin, norm_xmax = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
                max_dy = 0.0
                for x, y, _, _ in prepared:
                    mask = np.ones_like(x, dtype=bool)
                    if norm_xmin is not None:
                        mask &= x >= norm_xmin
                    if norm_xmax is not None:
                        mask &= x <= norm_xmax
                    yv = y[mask] if np.any(mask) else y
                    if len(yv):
                        max_dy = max(max_dy, (np.nanmax(yv) - np.nanmin(yv)) * 0.5)
                delta = shift_factor * max_dy
            for i, (x, y, label, st) in enumerate(prepared):
                y_plot = y + i * delta
                line = ax.plot(x, y_plot, label=label, **self._style_kwargs(i, label, len(prepared), st))[0]
                self._drawn_colors[label] = line.get_color()
                placed.append((ax, x, y_plot, label))
            if self._preview:
                for x, y, title in self._preview:
                    x, y = self._display_x(np.asarray(x, float), np.asarray(y, float))
                    ax.plot(x, y, lw=1.4, ls="--", color="#d62728", alpha=0.9, label=f"preview: {title}")
            n_entries = len(prepared) + (len(self._preview) if self._preview else 0)
            if n_entries > 1:
                self._place_legend(ax, n_entries)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(self.y_title_edit.text())
            if self.no_yticks_check.isChecked():
                ax.tick_params(left=False, labelleft=False)
            if self.zero_line_check.isChecked():
                ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
            if not self.no_title_check.isChecked() and (global_title or prepared):
                ax.set_title(global_title or (prepared[-1][2] if len(prepared) == 1 else ""))
            self._finish_axes(ax, fs)

        self._draw_cif_bragg_markers()
        self._apply_axis_limits(skip_redraw=True)
        self._draw_peak_labels(placed, spectra)
        self._draw_found_peaks(placed, spectra)
        self._draw_annotations()
        self._draw_pick_overlays()
        self._update_swatches()
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def _update_swatches(self) -> None:
        """Colour squares in the list matching the drawn lines. setIcon
        emits itemChanged, which would request another redraw and loop
        forever at the debounce period — hence the signal block."""
        import matplotlib.colors as mc
        self.file_list.blockSignals(True)
        try:
            self._paint_swatches(mc)
        finally:
            self.file_list.blockSignals(False)

    def _paint_swatches(self, mc) -> None:
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            s = self.library.get(it.data(Qt.UserRole))
            if s is None:
                continue
            label = self.styles.get(s.id, {}).get("label") or s.title
            color = self._drawn_colors.get(label)
            pm = QPixmap(12, 12)
            if color is not None and it.checkState() == Qt.Checked:
                try:
                    pm.fill(QColor(mc.to_hex(color)))
                except Exception:
                    pm.fill(QColor("#888888"))
            else:
                pm.fill(Qt.transparent)
            it.setIcon(QIcon(pm))

    def _apply_axis_limits(self, skip_redraw: bool = False) -> None:
        xmin, xmax = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        ymin, ymax = _to_float(self.ymin_edit.text()), _to_float(self.ymax_edit.text())
        for ax in self.plot.figure.get_axes():
            if xmin is not None or xmax is not None:
                lo, hi = ax.get_xlim()
                new_lo = xmin if xmin is not None else lo
                new_hi = xmax if xmax is not None else hi
                if new_lo != new_hi:
                    ax.set_xlim(new_lo, new_hi)
            if ymin is not None or ymax is not None:
                lo, hi = ax.get_ylim()
                new_lo = ymin if ymin is not None else lo
                new_hi = ymax if ymax is not None else hi
                if new_lo != new_hi:
                    ax.set_ylim(new_lo, new_hi)
        if not skip_redraw:
            self.plot.canvas.draw_idle()

    def autoscale_y(self) -> None:
        for ax in self.plot.figure.get_axes():
            ax.relim()
            ax.autoscale(axis="y")
        self.ymin_edit.clear()
        self.ymax_edit.clear()
        self.plot.canvas.draw_idle()

    # ==================================================================
    # CIF Bragg overlay
    # ==================================================================
    def import_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CIF file", "", "CIF files (*.cif);;All files (*.*)")
        if not path:
            return
        for s in self.cif_series:
            if os.path.abspath(s["path"]) == os.path.abspath(path):
                s["visible"] = True
                self.render()
                return
        try:
            peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
        except Exception as exc:
            QMessageBox.critical(self, "CIF import error", f"Could not read CIF:\n{exc}")
            return
        if not peaks:
            QMessageBox.information(self, "CIF import", "No Bragg peak found below 80° 2θ.")
            return
        color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
        self.cif_series.append({"path": path, "label": os.path.basename(path), "plot_label": "",
                                "peaks": peaks, "visible": True, "color": color, "pad": 0.03})
        self.render()

    def restore_cif_overlays(self, records: List[Dict[str, Any]]) -> int:
        restored = 0
        self.cif_series = []
        for rec in records:
            path = rec.get("path", "")
            if not path or not os.path.isfile(path):
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            self.cif_series.append({"path": path, "label": rec.get("label", os.path.basename(path)),
                                    "plot_label": rec.get("plot_label", ""), "peaks": peaks,
                                    "visible": bool(rec.get("visible", True)), "color": rec.get("color", "crimson"),
                                    "pad": float(rec.get("pad", 0.03))})
            restored += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.plot.request_redraw(self.render)
        return restored

    def add_cif_files(self, paths: List[str]) -> int:
        added = 0
        for path in paths:
            ap = os.path.abspath(path)
            existing = next((s for s in self.cif_series if os.path.abspath(s.get("path", "")) == ap), None)
            if existing is not None:
                existing["visible"] = True
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            if not peaks:
                continue
            color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
            self.cif_series.append({"path": path, "label": os.path.basename(path), "plot_label": "",
                                    "peaks": peaks, "visible": True, "color": color, "pad": 0.03})
            added += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.plot.request_redraw(self.render)
        return added

    def choose_cif_folder_and_reload(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select CIF folder")
        if not folder:
            return
        cif_paths = list_cif_files_case_insensitive(folder)
        if not cif_paths:
            QMessageBox.information(self, "CIF manager", "No CIF files found in the selected folder.")
            return
        added = 0
        for p in cif_paths:
            ap = os.path.abspath(p)
            if any(os.path.abspath(s.get("path", "")) == ap for s in self.cif_series):
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(p, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
            self.cif_series.append({"path": p, "label": os.path.basename(p), "plot_label": "",
                                    "peaks": peaks, "visible": False, "color": color, "pad": 0.03})
            added += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.render()
        QMessageBox.information(self, "CIF manager", f"Found {len(cif_paths)} CIF files.\nAdded {added} new entries.")

    def open_cif_manager(self) -> None:
        if self._cif_manager is None:
            self._cif_manager = CifManagerDialog(self)
        else:
            self._cif_manager.rebuild()
        self._cif_manager.show()
        self._cif_manager.raise_()

    def set_cif_field(self, serie: Dict[str, Any], key: str, value: Any) -> None:
        serie[key] = value
        self.plot.request_redraw(self.render)

    def _effective_xlim(self, ax):
        xmin_ax, xmax_ax = sorted(ax.get_xlim())
        xmin_in, xmax_in = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        if xmin_in is not None and xmax_in is not None and xmin_in < xmax_in:
            return xmin_in, xmax_in
        if xmin_in is not None:
            return xmin_in, xmax_ax
        if xmax_in is not None:
            return xmin_ax, xmax_in
        return xmin_ax, xmax_ax

    def _draw_cif_bragg_markers(self) -> None:
        visible_series = [c for c in self.cif_series if c.get("visible", True)]
        axes = self.plot.figure.get_axes()
        if not visible_series or not axes:
            return
        hmul = max(0.1, self.bragg_height_scale)
        for ax in axes:
            xmin, xmax = self._effective_xlim(ax)
            ymin_orig, ymax_orig = ax.get_ylim()
            height = (ymax_orig - ymin_orig) or 1.0
            n_in_view = [len([tt for (tt, _hkl, _d) in s["peaks"] if xmin <= tt <= xmax]) for s in visible_series]
            max_n = max(n_in_view) if n_in_view else 0
            row_space_factor = 1.0 + min(2.0, 0.02 * max_n)
            base_band_h = 0.03 * height
            base_spacing = 1.8 * base_band_h
            base_offset = 0.02 * height
            band_h = base_band_h * hmul
            spacing = base_spacing * row_space_factor
            off_down = base_offset * row_space_factor
            extra_down = off_down + len(visible_series) * spacing
            ax.set_ylim(ymin_orig - extra_down, ymax_orig)
            pos = self.cif_label_pos
            xspan = (xmax - xmin) or 1.0
            if pos in ("right_out", "left_out"):
                max_pad = max(float(c.get("pad", 0.03)) for c in visible_series)
                pad_abs = max_pad * xspan
                if pos == "right_out":
                    ax.set_xlim(xmin, xmax + pad_abs)
                else:
                    ax.set_xlim(xmin - pad_abs, xmax)
                xmin, xmax = ax.get_xlim()
                xspan = (xmax - xmin) or 1.0
            for i, serie in enumerate(visible_series):
                color = serie.get("color", "crimson")
                disp_label = serie.get("plot_label") or serie.get("label") or f"CIF {i + 1}"
                cif_pad = float(serie.get("pad", 0.03))
                y0 = (ymin_orig - off_down) - i * spacing
                y1 = y0 + band_h * 0.6
                for tt, hkl, d in serie["peaks"]:
                    if xmin <= tt <= xmax:
                        ax.vlines(tt, y0, y1, colors=color, linewidth=1.2)
                if pos == "right_out":
                    x_text, dytext, ha = xmax - cif_pad * xspan, band_h * 0.75, "right"
                elif pos == "left_out":
                    x_text, dytext, ha = xmin + cif_pad * xspan, band_h * 0.75, "left"
                elif pos == "right_in":
                    x_text, dytext, ha = xmax - cif_pad * xspan, 0, "right"
                else:
                    x_text, dytext, ha = xmin + cif_pad * xspan, 0, "left"
                ax.text(x_text, y0 + dytext, disp_label, va="bottom", ha=ha, fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.1", edgecolor="none", fc="white", lw=0.6, alpha=0.9), clip_on=False)
