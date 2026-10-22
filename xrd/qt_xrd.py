"""
qt_xrd.py — the XRD phase-identification workspace: a QualX-style
search-match (xrd_id_science engine) over the user's OWN registered card
databases, plus a reference-card browser and the Raman↔XRD bridge.

PRISM ships no reference data: any QualX-format or PRISM-format .sq the
user has the rights to can be registered ("Add database…" / "Add folder…"),
several can be enabled at once, and every search probes all enabled ones.

A folder of the user's own CIFs registers as a database as well — every
structure's pattern is calculated once and indexed like a card.

Same philosophy as Raman ID: candidates are RANKED with their figure of
merit, sources, and codes — identification only happens on an explicit
Accept, which works iteratively for mixtures (matched peaks removed, the
remaining ones re-searched), mirroring the Raman workflow. An Accept is a
hypothesis, so the Accepted-phases tab can withdraw one again (its peaks
return to the search) and export the accepted structures as CIFs for a
Rietveld refinement.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import xrd.xrd_id_science as xid
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float

RESULT_COLUMNS = ["FoM %", "Mineral / name", "Formula", "Source", "Code", "Q", "Space group",
                  "Matched", "Δ2θ"]
ACCEPTED_COLUMNS = ["Phase", "Formula", "Source", "Code", "FoM %", "Peaks explained", "Structure"]
CARD_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "teal"]
# Muted palette for phases already accepted (they stay overlaid, QualX-style)
ACCEPTED_COLORS = ["#9c6b74", "#6b7d9c", "#6b9c7d", "#9c8a6b", "#856b9c", "#6b969c"]


def _parse_elements(text: str) -> List[str]:
    return [t.strip() for t in (text or "").replace(";", ",").split(",") if t.strip()]


def _opt_float(text: str) -> Optional[float]:
    """A blank field means 'let PRISM decide', not zero."""
    t = (text or "").strip()
    if not t:
        return None
    return _to_float(t, None)


class PeakSearchOptionsDialog(QDialog):
    """Every peak-search setting, grouped by what it does to the search.

    Modeless on purpose: the whole point is to change a number, hit Apply,
    and watch the peak list and the detection threshold move on the plot
    behind the dialog. A setting you cannot see the effect of is a setting
    nobody can tune.
    """

    def __init__(self, parent, options: xid.PeakSearchOptions, on_apply):
        super().__init__(parent)
        self.setWindowTitle("Peak search options")
        self._on_apply = on_apply
        self._fields: Dict[str, QWidget] = {}
        root = QVBoxLayout(self)

        intro = QLabel(
            "Widths are in °2θ and are converted with the pattern's own step, so the "
            "same settings work on a fast survey and a slow scan. Blank = automatic.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        det = QGroupBox("Detection")
        dl = QFormLayout(det)
        self._add_line(dl, "min_prominence_sigma", "Detection limit (×σ)",
                       "How far above the local noise a peak must stand. Measured on 110 "
                       "simulated patterns: 3σ gave ~9 spurious peaks per pattern, 4σ gave "
                       "1.4 and cost 0.1% of the real ones. Lower it to chase weak "
                       "reflections, raise it on a noisy scan.")
        self.local_noise_check = QCheckBox("measure the noise locally")
        self.local_noise_check.setToolTip(
            "Counting noise scales as √counts, so it is several times larger under an "
            "amorphous halo than at high angle. Measuring it in a sliding window keeps "
            "'4σ' meaning the same thing across the scan. Switch off only to reproduce a "
            "single global threshold.")
        dl.addRow(self.local_noise_check)
        self._add_line(dl, "noise_window", "Noise window (°2θ)",
                       "How far around each point the local noise is measured. Wide enough "
                       "to hold many points, narrow enough to follow the background.")
        root.addWidget(det)

        bg = QGroupBox("Background")
        bl = QFormLayout(bg)
        self.subtract_bg_check = QCheckBox("subtract the background before searching")
        self.subtract_bg_check.setToolTip(
            "SNIP background removal. Off only if the pattern is already corrected.")
        bl.addRow(self.subtract_bg_check)
        self._add_line(bl, "background_width", "SNIP width (°2θ, blank = auto)",
                       "The SNIP window is an aggressiveness dial, not a cutoff: too wide "
                       "and it eats the amorphous halo and hands its counts to the peak "
                       "intensities, too narrow and it clips broad reflections away as "
                       "background. Blank estimates it from this pattern's own peak widths.")
        root.addWidget(bg)

        res = QGroupBox("Resolution")
        rl = QFormLayout(res)
        self._add_line(rl, "smooth_width", "Smoothing width (°2θ)",
                       "Savitzky-Golay window used before detection. Larger finds broad "
                       "peaks and merges close ones; smaller resolves doublets and lets "
                       "more noise through.")
        self._add_line(rl, "min_separation", "Minimum peak separation (°2θ)",
                       "Two maxima closer than this are reported as one peak.")
        root.addWidget(res)

        ka2 = QGroupBox("Kα2 satellites")
        kl = QFormLayout(ka2)
        self.ka2_check_dlg = QCheckBox("identify and drop Kα2 satellites")
        self.ka2_check_dlg.setToolTip(
            "Computed from λ, so it follows the anode. Only resolved doublets are "
            "affected — an unresolved one is a single peak already.")
        kl.addRow(self.ka2_check_dlg)
        self._add_line(kl, "ka2_tol", "Position tolerance (°2θ)",
                       "How close to the computed Kα2 position a peak must sit to be "
                       "called a satellite. Raise it if doublets are being missed; lower "
                       "it if a genuine reflection is being eaten.")
        self._add_line(kl, "ka2_max_ratio", "Max satellite / parent intensity",
                       "A Kα2 satellite is physically about half its parent. The bound "
                       "rejects accidental coincidences with a real reflection of "
                       "comparable strength.")
        root.addWidget(ka2)

        rng = QGroupBox("Search range and output")
        gl = QFormLayout(rng)
        self._add_line(gl, "two_theta_min", "Search from 2θ (blank = start of scan)",
                       "Restricts which detected peaks are reported. The background and "
                       "the noise are still measured on the whole scan, so narrowing the "
                       "range never moves the peaks that remain.")
        self._add_line(gl, "two_theta_max", "Search to 2θ (blank = end of scan)", "")
        self._add_line(gl, "edge_margin", "Ignore margin at each end (°2θ)",
                       "Drops peaks within this distance of either end of the scan, where "
                       "a partial profile is easily mistaken for a reflection.")
        self._add_line(gl, "max_peaks", "Maximum peaks reported",
                       "The strongest Kα1 lines are kept when there are more than this.")
        root.addWidget(rng)

        btns = QHBoxLayout()
        reset_btn = QPushButton("Restore defaults")
        reset_btn.clicked.connect(self._restore_defaults)
        btns.addWidget(reset_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        apply_btn = QPushButton("Apply and re-find")
        apply_btn.setObjectName("Primary")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._apply)
        btns.addWidget(apply_btn)
        root.addLayout(btns)

        self.set_options(options)

    def _add_line(self, form: QFormLayout, name: str, label: str, tip: str) -> None:
        edit = QLineEdit()
        edit.setMaximumWidth(110)
        if tip:
            edit.setToolTip(tip)
        edit.returnPressed.connect(self._apply)
        self._fields[name] = edit
        row_label = QLabel(label)
        if tip:
            row_label.setToolTip(tip)
        row_label.setWordWrap(True)
        form.addRow(row_label, edit)

    def set_options(self, options: xid.PeakSearchOptions) -> None:
        for name, edit in self._fields.items():
            value = getattr(options, name)
            edit.setText("" if value is None else (f"{value:g}" if isinstance(value, float)
                                                   else str(value)))
        self.local_noise_check.setChecked(options.local_noise)
        self.subtract_bg_check.setChecked(options.subtract_background)
        self.ka2_check_dlg.setChecked(options.flag_ka2)

    def options(self) -> xid.PeakSearchOptions:
        """Read the dialog back. A field that will not parse keeps the
        default rather than silently becoming zero, which would turn a
        typo into a search that finds everything or nothing."""
        opts = xid.PeakSearchOptions()
        for name, edit in self._fields.items():
            default = getattr(opts, name)
            value = _opt_float(edit.text())
            if value is None:
                # blank is meaningful for the optional settings only
                if default is None:
                    setattr(opts, name, None)
                continue
            setattr(opts, name, int(value) if name == "max_peaks" else float(value))
        opts.local_noise = self.local_noise_check.isChecked()
        opts.subtract_background = self.subtract_bg_check.isChecked()
        opts.flag_ka2 = self.ka2_check_dlg.isChecked()
        return opts

    def _restore_defaults(self) -> None:
        self.set_options(xid.PeakSearchOptions())
        self._apply()

    def _apply(self) -> None:
        self._on_apply(self.options())


class XrdIdWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 db_path: Optional[str] = None, on_accept=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Database registry: with an explicit db_path (tests / embedding) the
        # workspace runs on that single file and never touches the on-disk
        # registry; otherwise the JSON registry drives the list and toggles
        # persist there.
        if db_path:
            self._registry_backed = False
            self._db_entries: List[Dict[str, Any]] = [
                {"name": os.path.splitext(os.path.basename(db_path))[0], "path": db_path, "enabled": True}]
        else:
            self._registry_backed = True
            self._db_entries = xid.load_registry()
        # Shell callback (spectrum_id, previous_state) — accepted phases join
        # the Library undo stack, same as Raman ID.
        self.on_accept = on_accept
        self._results: List[xid.XrdMatch] = []
        self._query_peaks: List[float] = []
        self._query_int: List[float] = []
        # Per-spectrum session state (QualX-style): accepted phases stay
        # overlaid across iterative searches, and the query peaks they
        # explain turn gray instead of vanishing from the plot.
        self._session: Dict[str, Dict[str, Any]] = {}
        # spectra whose accepted list has already been reconstructed from
        # metadata (once per spectrum — a later empty list means the user
        # emptied it, not that it was never loaded)
        self._restored_sids: set = set()
        # The whole peak-search setup travels as one object, restored from
        # the last session so a user who tuned it once does not re-tune it
        # on every pattern.
        self.peak_options = xid.PeakSearchOptions.from_dict(self._load_peak_options())
        self._peak_dialog: Optional[PeakSearchOptionsDialog] = None
        self._peak_trace = None
        self._build_ui()
        self._refresh_db_status()
        self._refresh_accepted_table()

    def _state(self) -> Dict[str, Any]:
        sid = self.spec_combo.currentData() or "__none__"
        return self._session.setdefault(sid, {"accepted": [], "explained": []})

    def _enabled_paths(self) -> List[str]:
        return [e["path"] for e in self._db_entries
                if e.get("enabled", True) and os.path.isfile(e.get("path", ""))]

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Query pattern"))
        self.spec_combo = QComboBox()
        self.spec_combo.currentIndexChanged.connect(self._on_spectrum_changed)
        left_layout.addWidget(self.spec_combo)

        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto-find peaks")
        auto_btn.setToolTip(
            "XRD peak finder: SNIP background removal, then detection above the "
            "background-subtracted noise, sub-step centers, and Kα2 satellites "
            "identified geometrically rather than reported as extra reflections."
        )
        auto_btn.clicked.connect(self.auto_find_peaks)
        auto_row.addWidget(auto_btn)
        auto_row.addWidget(QLabel("limit ×σ"))
        self.detection_edit = QLineEdit(f"{self.peak_options.min_prominence_sigma:g}")
        self.detection_edit.setMaximumWidth(45)
        self.detection_edit.setToolTip(
            "Detection limit in units of the LOCAL noise. 4σ is the default because on "
            "110 simulated patterns it cut spurious peaks from ~9 per pattern to 1.4 "
            "while losing 0.1% of the real ones.")
        auto_row.addWidget(self.detection_edit)
        left_layout.addLayout(auto_row)

        opt_row = QHBoxLayout()
        self.background_check = QCheckBox("subtract background")
        self.background_check.setChecked(self.peak_options.subtract_background)
        self.background_check.setToolTip(
            "Estimate and remove the background (amorphous halo, air scatter, "
            "fluorescence) before detecting peaks and before reading their "
            "intensities. Off only if the pattern is already background-corrected."
        )
        opt_row.addWidget(self.background_check)
        self.ka2_check = QCheckBox("strip Kα2")
        self.ka2_check.setChecked(self.peak_options.flag_ka2)
        self.ka2_check.setToolTip(
            "Drop peaks sitting exactly where a stronger peak's Kα2 satellite must "
            "be (computed from λ, so it follows the anode). Only resolved doublets "
            "are affected — an unresolved one is a single peak already."
        )
        opt_row.addWidget(self.ka2_check)
        left_layout.addLayout(opt_row)

        more_row = QHBoxLayout()
        self.peak_options_btn = QPushButton("Peak search options…")
        self.peak_options_btn.setToolTip(
            "Everything the peak finder uses: detection limit and how the noise is "
            "measured, the background window, smoothing and minimum separation, the Kα2 "
            "test, the search range, and the peak cap.")
        self.peak_options_btn.clicked.connect(self.open_peak_options)
        more_row.addWidget(self.peak_options_btn)
        self.show_trace_check = QCheckBox("show threshold")
        self.show_trace_check.setToolTip(
            "Draw the estimated background and the detection threshold over the pattern, "
            "so a peak that was or was not found can be explained by looking at it.")
        self.show_trace_check.toggled.connect(
            lambda _=None: self.plot.request_redraw(self.render_preview))
        more_row.addWidget(self.show_trace_check)
        left_layout.addLayout(more_row)

        self.peak_status_label = QLabel("")
        self.peak_status_label.setWordWrap(True)
        left_layout.addWidget(self.peak_status_label)

        self.pick_peaks_btn = QPushButton("Pick peaks on plot")
        self.pick_peaks_btn.setCheckable(True)
        self.pick_peaks_btn.setToolTip("Toggle, then click peaks on the Phase ID preview — each click appends a 2θ value.")
        self.pick_peaks_btn.toggled.connect(self._on_pick_peaks_toggled)
        left_layout.addWidget(self.pick_peaks_btn)

        left_layout.addWidget(QLabel("Peaks 2θ (comma-sep — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        row = QHBoxLayout()
        row.addWidget(QLabel("λ (Å)"))
        self.wavelength_edit = QLineEdit(f"{xid.CU_KA1}")
        self.wavelength_edit.setMaximumWidth(60)
        row.addWidget(self.wavelength_edit)
        row.addWidget(QLabel("tol (°2θ)"))
        self.tol_edit = QLineEdit("0.2")
        self.tol_edit.setMaximumWidth(45)
        row.addWidget(self.tol_edit)
        row.addWidget(QLabel("shift ±"))
        self.shift_edit = QLineEdit("0.05")
        self.shift_edit.setMaximumWidth(45)
        self.shift_edit.setToolTip(
            "Let each card slide by up to this many °2θ before scoring, to absorb "
            "sample-displacement / zero error — the dominant systematic in "
            "Bragg-Brentano geometry. The offset each card actually used is shown "
            "in the Δ2θ column; a whole result list wanting the same large shift "
            "means the diffractometer, not the phases. 0 turns it off."
        )
        row.addWidget(self.shift_edit)
        left_layout.addLayout(row)

        left_layout.addWidget(QLabel("Chemistry filter"))
        el_row = QHBoxLayout()
        el_row.addWidget(QLabel("contains all:"))
        self.elements_all_edit = QLineEdit()
        self.elements_all_edit.setPlaceholderText("e.g. Si, O")
        el_row.addWidget(self.elements_all_edit)
        left_layout.addLayout(el_row)
        el_row2 = QHBoxLayout()
        el_row2.addWidget(QLabel("excludes:"))
        self.elements_none_edit = QLineEdit()
        self.elements_none_edit.setPlaceholderText("e.g. Pb")
        el_row2.addWidget(self.elements_none_edit)
        left_layout.addLayout(el_row2)

        left_layout.addWidget(QLabel("Card filters"))
        from core.qt_widgets import CheckComboBox
        filt_row = QHBoxLayout()
        self.system_filter = CheckComboBox("systems")
        self.system_filter.set_items(xid.CRYSTAL_SYSTEMS)
        filt_row.addWidget(self.system_filter, 1)
        self.quality_filter = CheckComboBox("qualities")
        filt_row.addWidget(self.quality_filter, 1)
        left_layout.addLayout(filt_row)
        self.sg_filter_edit = QLineEdit()
        self.sg_filter_edit.setPlaceholderText("space group contains… (e.g. Pnma)")
        left_layout.addWidget(self.sg_filter_edit)

        self.source_checks: Dict[str, QCheckBox] = {}
        self.sources_holder = QHBoxLayout()
        left_layout.addLayout(self.sources_holder)

        self._search_btn = QPushButton("Search match")
        self._search_btn.setObjectName("Primary")
        self._search_btn.clicked.connect(self.run_search)
        left_layout.addWidget(self._search_btn)

        accept_btn = QPushButton("Accept selected phase")
        accept_btn.clicked.connect(self.accept_selected)
        left_layout.addWidget(accept_btn)

        clear_row = QHBoxLayout()
        update_btn = QPushButton("Update plot")
        update_btn.clicked.connect(lambda: self.plot.request_redraw(self.render_preview))
        clear_row.addWidget(update_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Start over on this pattern: clears peaks, results, and the accepted-phase overlays (accepted identifications stay recorded on the spectrum; Ctrl+Z in the Library to undo those).")
        clear_btn.clicked.connect(self.clear_session)
        clear_row.addWidget(clear_btn)
        left_layout.addLayout(clear_row)

        self.raman_link_btn = QPushButton("Check Raman-identified phases here")
        self.raman_link_btn.setToolTip(
            "For each phase accepted in Mineral ID on this spectrum's sample, look the mineral up in "
            "the XRD database and overlay its reference lines on the query pattern."
        )
        self.raman_link_btn.clicked.connect(self.check_raman_phases)
        left_layout.addWidget(self.raman_link_btn)

        left_layout.addWidget(QLabel("Databases (check = probe)"))
        self.db_list = QListWidget()
        self.db_list.setMaximumHeight(110)
        self.db_list.setToolTip(
            "Register any card database you have the rights to use: a QualX-format .sq is "
            "indexed once locally; a PRISM-format .sq is used in place. A folder of CIFs "
            "becomes a database too — the pattern of each structure is calculated once. "
            "Check several to probe them all in one search."
        )
        self.db_list.itemChanged.connect(self._on_db_toggled)
        left_layout.addWidget(self.db_list)
        db_btn_row = QHBoxLayout()
        self.add_db_btn = QPushButton("Add database…")
        self.add_db_btn.clicked.connect(self.add_database)
        db_btn_row.addWidget(self.add_db_btn)
        self.add_folder_btn = QPushButton("Add folder…")
        self.add_folder_btn.clicked.connect(self.add_database_folder)
        db_btn_row.addWidget(self.add_folder_btn)
        self.remove_db_btn = QPushButton("Remove")
        self.remove_db_btn.setToolTip("Unregister the selected database (the .sq file itself is not deleted).")
        self.remove_db_btn.clicked.connect(self.remove_database)
        db_btn_row.addWidget(self.remove_db_btn)
        left_layout.addLayout(db_btn_row)

        self.add_cif_folder_btn = QPushButton("Add CIF folder…")
        self.add_cif_folder_btn.setToolTip(
            "Search against your OWN structure files: every .cif under a folder is turned "
            "into a reference pattern (positions AND intensities calculated from the "
            "structure, so a centred lattice contributes no extinct lines) and indexed "
            "like any other database. Re-add the same folder later to pick up new files.\n"
            "A phase identified this way exports its CIF straight back out — it is your "
            "file, so no download and no guessing."
        )
        self.add_cif_folder_btn.clicked.connect(self.add_cif_folder)
        left_layout.addWidget(self.add_cif_folder_btn)

        self.db_status_label = QLabel("")
        self.db_status_label.setWordWrap(True)
        left_layout.addWidget(self.db_status_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()

        id_tab = QWidget()
        idl = QVBoxLayout(id_tab)
        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        idl.addWidget(self.results_table, 1)
        self.plot = PlotWidget(figsize=(7, 4))
        idl.addWidget(self.plot, 1)
        self.tabs.addTab(id_tab, "Phase ID")

        browse_tab = QWidget()
        bl = QVBoxLayout(browse_tab)
        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("Name / mineral / formula"))
        self.browse_edit = QLineEdit()
        self.browse_edit.returnPressed.connect(self.browse_cards)
        browse_row.addWidget(self.browse_edit, 1)
        browse_btn = QPushButton("Look up")
        browse_btn.clicked.connect(self.browse_cards)
        browse_row.addWidget(browse_btn)
        bl.addLayout(browse_row)
        self.browse_list = QListWidget()
        self.browse_list.itemSelectionChanged.connect(lambda: self.browse_plot.request_redraw(self.render_browse))
        bl.addWidget(self.browse_list, 1)
        self.browse_plot = PlotWidget(figsize=(7, 4))
        bl.addWidget(self.browse_plot, 1)
        self.tabs.addTab(browse_tab, "Card browser")

        self.tabs.addTab(self._build_accepted_tab(), "Accepted phases")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self._browse_hits: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _build_accepted_tab(self) -> QWidget:
        """The accepted-phase ledger: what has been identified on this
        pattern, removable one by one (an accept is a hypothesis, and a
        hypothesis you cannot withdraw is not one), and exportable as
        structures for a Rietveld refinement."""
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(QLabel(
            "Phases accepted on the selected pattern. Removing one puts the peaks "
            "it explained back into the search."))
        self.accepted_table = QTableWidget(0, len(ACCEPTED_COLUMNS))
        self.accepted_table.setHorizontalHeaderLabels(ACCEPTED_COLUMNS)
        self.accepted_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.accepted_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.accepted_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.accepted_table, 1)

        btn_row = QHBoxLayout()
        self.remove_phase_btn = QPushButton("Remove selected phase")
        self.remove_phase_btn.setToolTip(
            "Withdraw the identification: it leaves the spectrum's record and the "
            "overlay, and the peaks it explained return to the search. Ctrl+Z in "
            "the Library undoes the removal.")
        self.remove_phase_btn.clicked.connect(self.remove_accepted_phase)
        btn_row.addWidget(self.remove_phase_btn)
        self.remove_all_phases_btn = QPushButton("Remove all")
        self.remove_all_phases_btn.clicked.connect(self.remove_all_accepted_phases)
        btn_row.addWidget(self.remove_all_phases_btn)
        lay.addLayout(btn_row)

        exp_row = QHBoxLayout()
        self.export_cif_btn = QPushButton("Export CIFs for Rietveld…")
        self.export_cif_btn.setObjectName("Primary")
        self.export_cif_btn.setToolTip(
            "Collect each accepted phase's crystal structure into one folder, with "
            "a phases.csv index — ready to load as starting models in GSAS-II, "
            "FullProf, TOPAS or Profex.")
        self.export_cif_btn.clicked.connect(self.export_accepted_cifs)
        exp_row.addWidget(self.export_cif_btn)
        self.cif_download_check = QCheckBox("download missing from COD")
        self.cif_download_check.setChecked(True)
        exp_row.addWidget(self.cif_download_check)
        lay.addLayout(exp_row)

        mirror_row = QHBoxLayout()
        mirror_row.addWidget(QLabel("Local CIF mirror:"))
        self.cif_folder_edit = QLineEdit()
        self.cif_folder_edit.setPlaceholderText("(optional) a folder of COD CIFs, searched before the web")
        mirror_row.addWidget(self.cif_folder_edit, 1)
        pick_btn = QPushButton("Browse…")
        pick_btn.clicked.connect(self._pick_cif_folder)
        mirror_row.addWidget(pick_btn)
        lay.addLayout(mirror_row)

        self.cif_status_label = QLabel(
            "A card database stores a d–I line list, not a structure: a CIF can only "
            "be fetched for cards whose code identifies a public entry (COD). PDF-2 "
            "cards have no structure to export — they are reported, not silently skipped.")
        self.cif_status_label.setWordWrap(True)
        lay.addWidget(self.cif_status_label)
        self._load_cif_settings()
        return tab

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------
    def _summary_cached(self, path: str):
        """database_summary scans the whole cards table — cache per file
        state so nav switches into this page stay instant."""
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            return None
        cache = getattr(self, "_summary_cache", None)
        if cache is None:
            cache = self._summary_cache = {}
        if key not in cache:
            cache[key] = xid.database_summary(path)
        return cache[key]

    def _refresh_db_status(self) -> None:
        self.db_list.blockSignals(True)
        self.db_list.clear()
        total = 0
        by_source: Dict[str, int] = {}
        for e in self._db_entries:
            item = QListWidgetItem(e.get("name", "?"))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            missing = not os.path.isfile(e.get("path", ""))
            item.setCheckState(Qt.Checked if e.get("enabled", True) and not missing else Qt.Unchecked)
            tip = e.get("path", "")
            if missing:
                item.setText(f"{e.get('name', '?')} (file missing)")
                tip += "\nThe .sq file is missing — reconnect the drive or Remove the entry."
            summary = self._summary_cached(e["path"]) if not missing else None
            if summary is not None:
                tip += f"\n{summary['total_cards']} cards"
                if e.get("enabled", True):
                    total += summary["total_cards"]
                    for k, v in summary["by_source"].items():
                        by_source[k] = by_source.get(k, 0) + v
            item.setToolTip(tip)
            self.db_list.addItem(item)
        self.db_list.blockSignals(False)

        if not self._db_entries:
            self.db_status_label.setText(
                "No card database registered yet. Download any QualX-format .sq database "
                "you have the rights to use, then click 'Add database…'."
            )
        else:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items()))
            self.db_status_label.setText(f"Enabled databases: {total} cards ({parts}).")

        # Source-tag checkboxes track the ENABLED databases' contents only:
        # tags from disabled/removed databases disappear (nobody sees tag
        # names their own files don't carry), new ones appear checked.
        for tag in list(self.source_checks):
            if tag not in by_source:
                cb = self.source_checks.pop(tag)
                self.sources_holder.removeWidget(cb)
                cb.deleteLater()
        for tag in sorted(by_source):
            if tag not in self.source_checks:
                cb = QCheckBox(tag)
                cb.setChecked(True)
                self.source_checks[tag] = cb
                self.sources_holder.addWidget(cb)

        # Quality filter entries = the quality codes actually present in
        # the enabled databases (check states preserved across refreshes).
        qualities: set = set()
        for e in self._db_entries:
            if e.get("enabled", True) and os.path.isfile(e.get("path", "")):
                summary = self._summary_cached(e["path"])
                qualities.update((summary or {}).get("qualities", []))
        self.quality_filter.set_items(sorted(qualities))

    def _on_db_toggled(self, item: QListWidgetItem) -> None:
        row = self.db_list.row(item)
        if not (0 <= row < len(self._db_entries)):
            return
        entry = self._db_entries[row]
        entry["enabled"] = item.checkState() == Qt.Checked
        if self._registry_backed:
            xid.set_database_enabled(entry["name"], entry["enabled"])
        # deferred: rebuilding the list from inside its own itemChanged
        # handler would delete the item mid-signal
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._refresh_db_status)

    def add_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add card database (.sq)", "",
            "Card databases (*.sq);;All files (*.*)")
        if path:
            self._register_paths([path])

    def add_database_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add every .sq database in a folder")
        if folder:
            self._register_paths([folder], folder_mode=True)

    def add_cif_folder(self) -> None:
        """Register a folder of structure files as a searchable database:
        each CIF's powder pattern is calculated once and indexed exactly
        like a card, so the whole search-match workflow applies to a
        lab's own structures."""
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of CIFs to search against (subfolders included)")
        if folder:
            self._register_paths([folder], cif_mode=True)

    def _register_paths(self, paths: List[str], folder_mode: bool = False,
                        cif_mode: bool = False) -> None:
        """Register in a worker thread — indexing a large QualX-format .sq
        takes minutes and must not freeze the UI."""
        self.add_db_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.add_cif_folder_btn.setEnabled(False)
        self.db_status_label.setText(
            "Calculating the powder pattern of every CIF in the folder… PRISM stays usable "
            "meanwhile." if cif_mode else
            "Registering… a QualX-format database is indexed once, which can take a few "
            "minutes for hundreds of thousands of cards. PRISM stays usable meanwhile."
        )

        registry_backed = self._registry_backed
        base_entries = list(self._db_entries)  # snapshot — the worker fn must not touch self
        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1) or xid.CU_KA1
        import_dir = None if registry_backed else os.path.join(
            os.path.dirname(base_entries[0]["path"]) if base_entries else ".", "cifdb")

        def work():
            added = []
            if cif_mode:
                # embedded/test mode has no registry: build beside the db it runs on
                reg = None if registry_backed else os.path.join(import_dir, "registry.json")
                for p in paths:
                    entry = xid.register_cif_folder(
                        p, wavelength=wavelength, registry_path=reg,
                        import_dir=import_dir, log=lambda *a: None)
                    added.append(entry)
                if registry_backed:
                    return xid.load_registry(), added
                return base_entries + [e for e in added
                                       if not any(os.path.normcase(x["path"]) ==
                                                  os.path.normcase(e["path"])
                                                  for x in base_entries)], added
            if registry_backed:
                for p in paths:
                    if folder_mode:
                        added += xid.register_folder(p, log=lambda *a: None)
                    else:
                        added.append(xid.register_database(p, log=lambda *a: None))
                return xid.load_registry(), added
            # single-db (embedded/test) mode: keep everything in memory
            entries = base_entries
            for p in paths:
                fmt = xid.sniff_sq_format(p)
                if fmt is None:
                    raise ValueError(f"{os.path.basename(p)} is not a recognizable card database.")
                entry = {"name": os.path.splitext(os.path.basename(p))[0], "path": p, "enabled": True}
                entries.append(entry)
                added.append(entry)
            return entries, added

        from core.qt_worker import run_in_thread
        run_in_thread(work, self._on_register_done, self._on_register_error)

    def _set_db_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self.add_db_btn, self.add_folder_btn, self.add_cif_folder_btn):
            btn.setEnabled(enabled)

    def _on_register_error(self, traceback_text: str) -> None:
        self._set_db_buttons_enabled(True)
        self._refresh_db_status()
        QMessageBox.critical(self, "Add database", traceback_text)

    def _on_register_done(self, result) -> None:
        entries, added = result
        self._set_db_buttons_enabled(True)
        self._db_entries = entries
        self._refresh_db_status()
        if added:
            names = ", ".join(e["name"] for e in added if e)
            self.db_status_label.setText(self.db_status_label.text() + f" Added: {names}.")
        else:
            QMessageBox.information(self, "Add database", "No new database found (already registered, or no .sq files).")

    def remove_database(self) -> None:
        row = self.db_list.currentRow()
        if not (0 <= row < len(self._db_entries)):
            QMessageBox.information(self, "Remove database", "Select a database in the list first.")
            return
        entry = self._db_entries[row]
        if self._registry_backed:
            xid.unregister_database(entry["name"])
            self._db_entries = xid.load_registry()
        else:
            self._db_entries = [e for e in self._db_entries if e is not entry]
        self._refresh_db_status()

    def set_spectra(self, spectrum_ids: List[str]) -> None:
        current = self.spec_combo.currentData()
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is not None:
                self.spec_combo.addItem(sp.title, sid)
        if current is not None:
            idx = self.spec_combo.findData(current)
            if idx >= 0:
                self.spec_combo.setCurrentIndex(idx)
        self.spec_combo.blockSignals(False)
        self._refresh_db_status()
        self._refresh_accepted_table()
        # Arriving on the page must show the diffractogram immediately, not
        # an empty axes (user request — same rule on every workspace).
        self.plot.request_redraw(self.render_preview)

    def _on_spectrum_changed(self, _index=None) -> None:
        """Every per-spectrum view follows the selection: the overlay, and
        the accepted-phase ledger that belongs to THIS pattern."""
        if hasattr(self, "accepted_table"):
            self._refresh_accepted_table()
        self.plot.request_redraw(self.render_preview)

    def _current_spectrum(self):
        sid = self.spec_combo.currentData()
        return self.library.get(sid) if sid else None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Peak search options
    # ------------------------------------------------------------------
    def _peak_settings_store(self):
        from core.app_settings import app_settings
        return app_settings()

    def _load_peak_options(self) -> Optional[Dict[str, Any]]:
        try:
            import json
            raw = self._peak_settings_store().value("xrd_id/peak_options", "")
            return json.loads(raw) if raw else None
        except Exception:  # a broken store must never stop the page loading
            return None

    def _save_peak_options(self) -> None:
        try:
            import json
            self._peak_settings_store().setValue(
                "xrd_id/peak_options", json.dumps(self.peak_options.to_dict()))
        except Exception:
            pass

    def _sync_options_from_panel(self) -> None:
        """The three settings that also live in the left panel are the
        common ones; the panel always wins, so the two views cannot
        disagree about them."""
        self.peak_options.min_prominence_sigma = (
            _to_float(self.detection_edit.text(), self.peak_options.min_prominence_sigma)
            or self.peak_options.min_prominence_sigma)
        self.peak_options.subtract_background = self.background_check.isChecked()
        self.peak_options.flag_ka2 = self.ka2_check.isChecked()

    def _sync_panel_from_options(self) -> None:
        self.detection_edit.setText(f"{self.peak_options.min_prominence_sigma:g}")
        for widget, value in ((self.background_check, self.peak_options.subtract_background),
                              (self.ka2_check, self.peak_options.flag_ka2)):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

    def open_peak_options(self) -> None:
        self._sync_options_from_panel()
        if self._peak_dialog is None:
            self._peak_dialog = PeakSearchOptionsDialog(
                self, self.peak_options, self._apply_peak_options)
        else:
            self._peak_dialog.set_options(self.peak_options)
        self._peak_dialog.show()
        self._peak_dialog.raise_()
        self._peak_dialog.activateWindow()

    def _apply_peak_options(self, options: xid.PeakSearchOptions) -> None:
        self.peak_options = options
        self._sync_panel_from_options()
        self._save_peak_options()
        if self._current_spectrum() is not None:
            self.auto_find_peaks()

    # ------------------------------------------------------------------
    def auto_find_peaks(self) -> None:
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Auto-find peaks", "Select a query pattern first.")
            return
        self._sync_options_from_panel()
        self._save_peak_options()
        opts = self.peak_options
        peaks, trace = xid.find_xrd_peaks_traced(
            sp.x, sp.y, wavelength=_to_float(self.wavelength_edit.text(), xid.CU_KA1),
            **opts.as_kwargs())
        self._peak_trace = trace
        primary = [p for p in peaks if not p.is_ka2]
        n_ka2 = len(peaks) - len(primary)
        if not primary:
            self.peak_status_label.setText(
                "No peaks above the detection limit — lower 'limit ×σ', or switch the "
                "background off if the pattern is already corrected.")
            self.plot.request_redraw(self.render_preview)
            QMessageBox.information(
                self, "Auto-find peaks",
                "No clear peaks at these settings. Tick 'show threshold' to see where the "
                "detection limit sits relative to the pattern.")
            return
        self.peaks_edit.setText(", ".join(f"{p.two_theta:.3f}" for p in primary))
        bits = [f"{len(primary)} peaks at ≥{opts.min_prominence_sigma:g}σ"]
        if opts.subtract_background:
            width = trace.background_width if trace is not None else 0.0
            bits.append(f"background removed ({width:.2f}°)" if width
                        else "background removed")
        bits.append("local noise" if opts.local_noise else "global noise")
        if n_ka2:
            bits.append(f"{n_ka2} Kα2 satellite(s) stripped")
        if opts.two_theta_min is not None or opts.two_theta_max is not None:
            lo = f"{opts.two_theta_min:g}" if opts.two_theta_min is not None else "start"
            hi = f"{opts.two_theta_max:g}" if opts.two_theta_max is not None else "end"
            bits.append(f"range {lo}–{hi}°")
        if len(primary) >= opts.max_peaks:
            bits.append(f"capped at {opts.max_peaks}")
        bits.append(f"weakest {min(p.snr for p in primary):.0f}σ")
        self.peak_status_label.setText(" · ".join(bits) + ".")
        self.plot.request_redraw(self.render_preview)

    def _on_pick_peaks_toggled(self, checked: bool) -> None:
        if checked:
            mode = str(self.plot.toolbar.mode)
            if "zoom" in mode:
                self.plot.toolbar.zoom()
            elif "pan" in mode:
                self.plot.toolbar.pan()
            self.plot.canvas.setCursor(Qt.CrossCursor)
            self._pick_cid = self.plot.canvas.mpl_connect("button_press_event", self._on_pick_peak_click)
            self.plot.request_redraw(self.render_preview)
        else:
            if getattr(self, "_pick_cid", None) is not None:
                self.plot.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.plot.canvas.unsetCursor()

    def _on_pick_peak_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or self.plot.toolbar.mode:
            return
        cur = self.peaks_edit.text().strip()
        xv = f"{float(event.xdata):.3f}"
        self.peaks_edit.setText(f"{cur}, {xv}" if cur else xv)

    def _parse_peaks(self):
        vals = []
        for tok in (self.peaks_edit.text() or "").split(","):
            v = _to_float(tok)
            if v is not None:
                vals.append(v)
        return vals

    def _peak_intensities(self, peaks: List[float]) -> List[float]:
        """Relative intensities for the peak list, background-subtracted.

        Reading raw heights and then subtracting the list's own minimum (the
        old rule) leaves each peak carrying whatever background sits under
        it: on a pattern with an amorphous halo that inflates the low-angle
        peaks by a factor of two and quietly re-weights every FoM. Works
        for hand-typed peaks too, not just auto-found ones."""
        sp = self._current_spectrum()
        if sp is None or not peaks:
            return [1.0] * len(peaks)
        return xid.peak_intensities_at(
            sp.x, sp.y, peaks, subtract_background=self.background_check.isChecked())

    def run_search(self) -> None:
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Search match", "Enter or auto-find at least one 2θ peak.")
            return
        db_paths = self._enabled_paths()
        if not db_paths:
            QMessageBox.information(
                self, "Search match",
                "No card database enabled. Register one with 'Add database…' (any "
                "QualX-format .sq you have the rights to use) and check it in the list.")
            return
        self._query_peaks = peaks
        self._query_int = self._peak_intensities(peaks)
        sp = self._current_spectrum()
        tt_range = (float(np.nanmin(sp.x)), float(np.nanmax(sp.x))) if sp is not None else None
        sources = [tag for tag, cb in self.source_checks.items() if cb.isChecked()]

        self._search_btn.setEnabled(False)
        self._search_btn.setText("Searching…")
        kwargs = dict(
            wavelength=_to_float(self.wavelength_edit.text(), xid.CU_KA1),
            tol_two_theta=_to_float(self.tol_edit.text(), 0.2),
            elements_all=_parse_elements(self.elements_all_edit.text()),
            elements_none=_parse_elements(self.elements_none_edit.text()),
            sources=sources if sources and len(sources) < len(self.source_checks) else (),
            two_theta_range=tt_range, db_paths=db_paths,
            qualities=self.quality_filter.checked(),
            crystal_systems=self.system_filter.checked(),
            spacegroup_contains=self.sg_filter_edit.text().strip(),
            shift_tolerance=max(0.0, _to_float(self.shift_edit.text(), 0.0) or 0.0),
        )
        from core.qt_worker import run_in_thread
        run_in_thread(
            lambda: xid.search_match(list(peaks), list(self._query_int), **kwargs),
            self._on_search_done, self._on_search_error,
        )

    def _on_search_error(self, traceback_text: str) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        QMessageBox.critical(self, "Search match error", traceback_text)

    def _on_search_done(self, results: List[xid.XrdMatch]) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        self._results = results
        self.results_table.setRowCount(len(results))
        for row, r in enumerate(results):
            display = r.mineral or r.name or r.formula
            matched = f"{r.n_matched}/{r.n_card_lines}" if r.n_card_lines else str(r.n_matched)
            values = [f"{100 * r.fom:.0f}", display, r.formula, r.source, r.source_code,
                      r.quality, r.spacegroup, matched, f"{r.shift:+.3f}" if r.shift else "—"]
            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if col == 7 and r.n_card_lines and not r.strongest_matched:
                    # the card's strongest in-range line is NOT in the data:
                    # the one disqualifier a good-looking FoM can hide
                    item.setToolTip("The card's strongest in-range line is not among "
                                    "your peaks — treat this match with suspicion.")
                    item.setText(f"{matched} ⚠")
                self.results_table.setItem(row, col, item)
        self.results_table.resizeColumnsToContents()
        if results:
            self.results_table.selectRow(0)
        else:
            self.plot.request_redraw(self.render_preview)

    def _selected_matches(self) -> List[xid.XrdMatch]:
        rows = sorted({i.row() for i in self.results_table.selectionModel().selectedRows()})
        return [self._results[r] for r in rows if 0 <= r < len(self._results)]

    def _on_result_selected(self) -> None:
        self.plot.request_redraw(self.render_preview)

    def render_preview(self) -> None:
        # keep the user's zoom across candidate-card re-renders (same spectrum)
        self.plot.preserve_zoom(("phase_id", self.spec_combo.currentData()))
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        sp = self._current_spectrum()
        ymax = 1.0
        if sp is not None:
            y = np.asarray(sp.y, float)
            ymax = float(np.nanmax(y)) or 1.0
            ax.plot(sp.x, y / ymax * 100.0, color="black", lw=1.0, label=f"query: {sp.title}")

        # The finder's own working, on the same axes and the same scale as
        # the pattern: the background it removed, and the level a peak had
        # to clear. With these on, "why was that shoulder not a peak?" is
        # a question you answer by looking.
        if self.show_trace_check.isChecked() and self._peak_trace is not None and sp is not None:
            tr = self._peak_trace
            scale = 100.0 / ymax
            ax.plot(tr.x, tr.background * scale, color="#2a7fb8", lw=0.9, ls="--",
                    label="background")
            ax.plot(tr.x, (tr.background + tr.threshold) * scale, color="#c05a2a", lw=0.9,
                    label=f"detection limit ({self.peak_options.min_prominence_sigma:g}σ)")

        # Query peak bars (QualX-style): the current (unexplained) peaks in
        # black; peaks already explained by an accepted phase turn gray.
        state = self._state()
        peaks = self._parse_peaks()
        if peaks:
            ax.vlines(peaks, 0, 104, color="black", lw=0.8, alpha=0.30,
                      label="query peaks")
        if state["explained"]:
            ax.vlines(state["explained"], 0, 104, color="gray", lw=0.8,
                      alpha=0.45, ls="--", label="explained by accepted phases")

        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        # Accepted phases stay overlaid across iterative searches (muted).
        for k, entry in enumerate(state["accepted"]):
            r = entry["match"]
            tt = xid.d_to_two_theta(r.d, wavelength)
            ok = np.isfinite(tt)
            label = f"accepted: {r.mineral or r.name or r.formula} [{r.source} {r.source_code}]"
            ax.vlines(tt[ok], 0, -r.i[ok], color=ACCEPTED_COLORS[k % len(ACCEPTED_COLORS)],
                      lw=1.1, alpha=0.75, label=label)
        for k, r in enumerate(self._selected_matches()):
            color = CARD_COLORS[k % len(CARD_COLORS)]
            tt = xid.d_to_two_theta(r.d, wavelength)
            ok = np.isfinite(tt)
            label = f"{r.mineral or r.name or r.formula} [{r.source} {r.source_code}] FoM {100 * r.fom:.0f}%"
            ax.vlines(tt[ok], 0, -r.i[ok], color=color, lw=1.2, label=label)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("query (up) / card lines (down)")
        if sp is not None or peaks or state["accepted"]:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        self.plot.restore_zoom(ax)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def clear_session(self) -> None:
        """Start over on the current pattern: peaks, results, overlays AND
        the accepted phases.

        Clearing used to leave the identifications on the spectrum while
        wiping their overlays, so a pattern could look untouched and still
        carry phases into a report. Clear now withdraws them too — one
        Ctrl+Z in the Library brings them all back."""
        self.remove_all_accepted_phases(confirm=False)
        sid = self.spec_combo.currentData() or "__none__"
        self._session.pop(sid, None)
        self.peaks_edit.setText("")
        self.peak_status_label.setText("")
        self._results = []
        self._query_peaks = []
        self._query_int = []
        self.results_table.setRowCount(0)
        self._restored_sids.discard(sid)
        self._refresh_accepted_table()
        self.plot.reset_zoom_memory()  # starting over autoscales again
        self.plot.request_redraw(self.render_preview)

    # ------------------------------------------------------------------
    # Accepted-phase ledger
    # ------------------------------------------------------------------
    def _restore_accepted_from_meta(self) -> None:
        """Rebuild the accepted list for a spectrum whose identifications
        were recorded in an earlier session (PRISM restarted, or a saved
        project reopened): the spectrum's metadata keeps source + code, and
        the card behind them is looked up again so the stick overlay and
        the CIF export work exactly as they did the first time. Cards whose
        database is no longer enabled are listed without an overlay rather
        than dropped — the identification was still made."""
        sp = self._current_spectrum()
        sid = self.spec_combo.currentData()
        if sp is None or sid is None or sid in self._restored_sids:
            return
        self._restored_sids.add(sid)
        state = self._state()
        if state["accepted"]:
            return
        records = list(sp.meta.get("xrd_matches", []) or [])
        if not records:
            return
        paths = self._enabled_paths()
        for rec in records:
            card = xid.find_card_by_code(rec.get("source", ""), rec.get("source_code", ""),
                                         db_paths=paths) if paths else None
            match = xid.XrdMatch(
                card_id=card["card_id"] if card else -1,
                source=rec.get("source", ""), source_code=rec.get("source_code", ""),
                name=rec.get("name", ""), mineral=rec.get("mineral", ""),
                formula=rec.get("formula", ""), spacegroup=rec.get("spacegroup", ""),
                quality=(card or {}).get("quality", ""), rir=None,
                fom=float(rec.get("fom") or 0.0), cov_card=0.0, cov_query=0.0,
                n_matched=int(rec.get("n_matched") or 0),
                d=(card or {}).get("d", np.array([])), i=(card or {}).get("i", np.array([])),
            )
            state["accepted"].append({"match": match, "peaks": [], "record": rec})

    def _refresh_accepted_table(self) -> None:
        self._restore_accepted_from_meta()
        entries = self._state()["accepted"]
        self.accepted_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            r = entry["match"]
            cod = xid.cod_id_for_card(r.source, r.source_code)
            values = [r.mineral or r.name or r.formula, r.formula, r.source, r.source_code,
                      f"{100 * r.fom:.0f}" if r.fom else "—",
                      str(len(entry["peaks"])) if entry["peaks"] else "—",
                      f"COD {cod}" if cod else "no structure (d–I list)"]
            for col, val in enumerate(values):
                self.accepted_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.accepted_table.resizeColumnsToContents()
        n_cif = sum(1 for e in entries
                    if xid.cod_id_for_card(e["match"].source, e["match"].source_code))
        if entries:
            self.cif_status_label.setText(
                f"{len(entries)} accepted phase(s); {n_cif} have a COD structure that can be "
                f"exported as a CIF. Cards from d–I-list databases (PDF-2 and the like) carry "
                f"no atomic coordinates, so no CIF exists to export for them.")

    def _withdraw(self, indices: List[int]) -> None:
        """Remove accepted phases by session-list index: drop the records
        from the spectrum (undoably), put their peaks back in the search
        box, and rebuild the explained-peak set from what is left."""
        sp = self._current_spectrum()
        state = self._state()
        entries = state["accepted"]
        indices = sorted({i for i in indices if 0 <= i < len(entries)}, reverse=True)
        if not indices or sp is None:
            return
        previous_state = {
            "xrd_match": sp.meta.get("xrd_match"),
            "xrd_matches": list(sp.meta["xrd_matches"]) if sp.meta.get("xrd_matches") else None,
        }
        returned: List[float] = []
        for i in indices:
            entry = entries.pop(i)
            returned.extend(entry["peaks"])
            record = entry.get("record")
            kept = [m for m in (sp.meta.get("xrd_matches") or []) if m is not record]
            if record is None:  # restored from metadata: match on identity of the card
                kept = [m for m in (sp.meta.get("xrd_matches") or [])
                        if not (str(m.get("source")) == entry["match"].source
                                and str(m.get("source_code")) == entry["match"].source_code)]
            if kept:
                sp.meta["xrd_matches"] = kept
                sp.meta["xrd_match"] = kept[-1]
            else:
                sp.meta.pop("xrd_matches", None)
                sp.meta.pop("xrd_match", None)
        state["explained"] = [p for e in entries for p in e["peaks"]]
        if returned:
            current = self._parse_peaks()
            merged = sorted(set(round(v, 4) for v in current + returned))
            self.peaks_edit.setText(", ".join(f"{v:.3f}" for v in merged))
            self._query_peaks = merged
            self._query_int = self._peak_intensities(merged)
        if self.on_accept is not None:
            self.on_accept(sp.id, previous_state)
        self._refresh_accepted_table()
        self.plot.request_redraw(self.render_preview)

    def remove_accepted_phase(self) -> None:
        rows = sorted({i.row() for i in self.accepted_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Remove phase",
                                    "Select an accepted phase in the table first.")
            return
        self._withdraw(rows)

    def remove_all_accepted_phases(self, *, confirm: bool = True) -> None:
        entries = self._state()["accepted"]
        if not entries:
            return
        if confirm:
            answer = QMessageBox.question(
                self, "Remove all phases",
                f"Withdraw all {len(entries)} accepted phase(s) from this pattern? "
                f"Their peaks return to the search; Ctrl+Z in the Library undoes it.")
            if answer != QMessageBox.Yes:
                return
        self._withdraw(list(range(len(entries))))

    # ------------------------------------------------------------------
    # CIF export
    # ------------------------------------------------------------------
    def _cif_settings(self):
        from core.app_settings import app_settings
        return app_settings()

    def _load_cif_settings(self) -> None:
        try:
            s = self._cif_settings()
            self.cif_folder_edit.setText(str(s.value("xrd_id/cif_mirror", "") or ""))
            stored = s.value("xrd_id/cif_download", True)
            self.cif_download_check.setChecked(str(stored).lower() not in ("false", "0"))
        except Exception:  # a broken/locked settings store must not block the page
            pass

    def _save_cif_settings(self) -> None:
        try:
            s = self._cif_settings()
            s.setValue("xrd_id/cif_mirror", self.cif_folder_edit.text().strip())
            s.setValue("xrd_id/cif_download", self.cif_download_check.isChecked())
        except Exception:
            pass

    def _pick_cif_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder holding COD CIFs (flat, or the cif/1/01/09/… tree)")
        if folder:
            self.cif_folder_edit.setText(folder)
            self._save_cif_settings()

    def export_accepted_cifs(self) -> None:
        """Write one CIF per accepted phase into a folder of the user's
        choice, plus a phases.csv index — the handover to a Rietveld
        program."""
        entries = self._state()["accepted"]
        if not entries:
            QMessageBox.information(
                self, "Export CIFs",
                "No accepted phases on this pattern yet. Accept the phases you have "
                "identified first — the export follows the ledger, not the candidate list.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export structures into…")
        if not out_dir:
            return
        self._save_cif_settings()
        phases = [{"source": e["match"].source, "source_code": e["match"].source_code,
                   "mineral": e["match"].mineral, "name": e["match"].name,
                   "formula": e["match"].formula, "fom": e["match"].fom}
                  for e in entries]
        # Every registered CIF folder is a mirror too — a structure the user
        # already has on disk must never be fetched from the web.
        folders = [self.cif_folder_edit.text().strip()] + \
                  [e["cif_folder"] for e in self._db_entries if e.get("cif_folder")]
        folders = [f for f in folders if f]
        db_paths = self._enabled_paths()
        allow = self.cif_download_check.isChecked()
        self.export_cif_btn.setEnabled(False)
        self.export_cif_btn.setText("Exporting…")
        self.cif_status_label.setText(
            "Fetching structures… COD downloads are cached, so a repeat export of the "
            "same phases needs no connection.")

        from core.qt_worker import run_in_thread
        run_in_thread(
            lambda: xid.export_phase_cifs(phases, out_dir, local_folders=folders,
                                          allow_download=allow, db_paths=db_paths,
                                          log=lambda *a: None),
            self._on_export_done, self._on_export_error)

    def _on_export_error(self, traceback_text: str) -> None:
        self.export_cif_btn.setEnabled(True)
        self.export_cif_btn.setText("Export CIFs for Rietveld…")
        QMessageBox.critical(self, "Export CIFs", traceback_text)

    def _on_export_done(self, report: Dict[str, Any]) -> None:
        self.export_cif_btn.setEnabled(True)
        self.export_cif_btn.setText("Export CIFs for Rietveld…")
        written, skipped = report["written"], report["skipped"]
        lines = [f"{len(written)} structure file(s) written."]
        if report.get("summary"):
            lines.append(f"Index: {os.path.basename(report['summary'])}")
        if skipped:
            lines.append("")
            lines.append(f"No CIF for {len(skipped)} phase(s):")
            lines += [f"  • {label} — {reason}" for label, reason in skipped[:8]]
            if len(skipped) > 8:
                lines.append(f"  … and {len(skipped) - 8} more (see the index).")
        self.cif_status_label.setText(" ".join(lines[:2]))
        QMessageBox.information(self, "Export CIFs", "\n".join(lines))

    # ------------------------------------------------------------------
    def accept_selected(self) -> None:
        """Iterative accept, mirroring Mineral ID: record the phase, drop
        the peaks it explains, re-search the remainder."""
        sp = self._current_spectrum()
        matches = self._selected_matches()
        if sp is None or not matches:
            QMessageBox.warning(self, "Accept phase", "Select a query pattern and a result row first.")
            return
        r = matches[0]
        previous_state = {
            "xrd_match": sp.meta.get("xrd_match"),
            "xrd_matches": list(sp.meta["xrd_matches"]) if sp.meta.get("xrd_matches") else None,
        }
        record = {
            "mineral": r.mineral, "name": r.name, "formula": r.formula,
            "source": r.source, "source_code": r.source_code, "spacegroup": r.spacegroup,
            "fom": r.fom, "n_matched": r.n_matched, "accepted_at": time.time(),
        }
        sp.meta["xrd_match"] = record
        accepted = list(sp.meta.get("xrd_matches", []))
        accepted.append(record)
        sp.meta["xrd_matches"] = accepted
        if self.on_accept is not None:
            self.on_accept(sp.id, previous_state)

        matched_q = {round(q, 4) for q, _ in r.matched_pairs}
        remaining = [p for p in self._query_peaks if round(p, 4) not in matched_q]
        explained = [p for p in self._query_peaks if round(p, 4) in matched_q]
        # session overlays: the accepted phase's bars stay, its peaks gray out.
        # The peaks each phase explains are kept WITH it, so withdrawing one
        # phase later returns exactly its own peaks and no one else's.
        state = self._state()
        state["accepted"].append({"match": r, "peaks": explained, "record": record})
        state["explained"].extend(explained)
        self._refresh_accepted_table()
        phase_list = ", ".join(f"{m['mineral'] or m['name'] or m['formula']} [{m['source']} {m['source_code']}]"
                               for m in accepted)
        if remaining:
            self.peaks_edit.setText(", ".join(f"{p:.3f}" for p in sorted(remaining)))
            QMessageBox.information(
                self, "Phase accepted",
                f"Recorded '{record['mineral'] or record['name'] or record['formula']}' "
                f"[{r.source} {r.source_code}] for '{sp.title}'.\n"
                f"{len(remaining)} peak(s) remain unexplained — searching those now.\n"
                f"Accepted so far: {phase_list}",
            )
            self.run_search()
        else:
            self.peaks_edit.setText("")
            self._results = []
            self.results_table.setRowCount(0)
            self.plot.request_redraw(self.render_preview)
            QMessageBox.information(
                self, "All peaks explained",
                f"Every query peak of '{sp.title}' is explained by {len(accepted)} phase(s): {phase_list}.",
            )

    # ------------------------------------------------------------------
    # Raman ↔ XRD bridge
    # ------------------------------------------------------------------
    def check_raman_phases(self) -> None:
        """Overlay the XRD reference lines of every phase accepted in
        Mineral ID (meta['rruff_matches'] on any library spectrum sharing…
        for now: on the selected query spectrum itself or any spectrum in
        the library) against the current pattern."""
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Raman link", "Select a query pattern first.")
            return
        # accepted Raman phases anywhere in the library (typically the Raman
        # spectrum of the same sample)
        minerals: List[str] = []
        for s in self.library.all():
            for m in s.meta.get("rruff_matches", []) or ([s.meta["rruff_match"]] if s.meta.get("rruff_match") else []):
                name = (m or {}).get("mineral")
                if name and name not in minerals:
                    minerals.append(name)
        if not minerals:
            QMessageBox.information(self, "Raman link",
                                    "No accepted Raman identifications found in the Library (Mineral ID → Accept).")
            return
        found: List[xid.XrdMatch] = []
        missing = []
        for name in minerals:
            hits = xid.find_cards_by_text(name, limit=3, db_paths=self._enabled_paths())
            if not hits:
                missing.append(name)
                continue
            h = hits[0]
            found.append(xid.XrdMatch(
                card_id=h["card_id"], source=h["source"], source_code=h["source_code"],
                name=h["name"], mineral=h["mineral"] or name, formula=h["formula"],
                spacegroup=h["spacegroup"], quality=h["quality"], rir=None,
                fom=0.0, cov_card=0.0, cov_query=0.0, n_matched=0, d=h["d"], i=h["i"],
            ))
        self._results = found
        self._on_search_done(found)
        if found:
            self.results_table.selectAll()
        msg = f"Overlaid XRD reference lines for {len(found)} Raman-identified phase(s): " \
              f"{', '.join(r.mineral or r.name for r in found)}."
        if missing:
            msg += f"\nNot in the XRD database: {', '.join(missing)}."
        self.db_status_label.setText(msg)

    # ------------------------------------------------------------------
    def browse_cards(self) -> None:
        """Text lookup PLUS element-set lookup: 'TiO2' also finds cards whose
        formula is written 'O2 Ti' (user-reported gap) — exact element set
        first, then contains-these-elements, then plain text; deduplicated."""
        text = self.browse_edit.text().strip()
        if not text:
            return
        db_paths = self._enabled_paths()
        hits, seen = [], set()
        for batch in (xid.find_cards_by_elements(text, mode="exact", db_paths=db_paths),
                      xid.find_cards_by_elements(text, mode="contains", limit=40, db_paths=db_paths),
                      xid.find_cards_by_text(text, limit=50, db_paths=db_paths)):
            for h in batch:
                # content-level dedup: the same card carried by several
                # databases/sources appears once (same rule as search results)
                key = xid._dedup_key(h["source_code"], h["name"], h["mineral"],
                                     h["formula"], h["spacegroup"], h["quality"])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        self._browse_hits = hits[:100]
        self.browse_list.clear()
        for h in self._browse_hits:
            label = h["mineral"] or h["name"] or h["formula"]
            self.browse_list.addItem(f"{label}  |  {h['formula']}  [{h['source']} {h['source_code']}]  {h['spacegroup']}")
        if not self._browse_hits:
            self.browse_list.addItem("(no cards found)")

    def render_browse(self) -> None:
        self.browse_plot.preserve_zoom("browse")  # keep zoom across card clicks
        fig = self.browse_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        rows = sorted({i.row() for i in self.browse_list.selectedIndexes()})
        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        for k, row in enumerate(rows):
            if 0 <= row < len(self._browse_hits):
                h = self._browse_hits[row]
                tt = xid.d_to_two_theta(h["d"], wavelength)
                ok = np.isfinite(tt)
                ax.vlines(tt[ok], 0, h["i"][ok], color=CARD_COLORS[k % len(CARD_COLORS)], lw=1.2,
                          label=f"{h['mineral'] or h['name']} [{h['source']} {h['source_code']}]")
        if rows:
            ax.legend(fontsize=7)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("relative intensity")
        ax.grid(alpha=0.2)
        self.browse_plot.restore_zoom(ax)
        fig.tight_layout()
        self.browse_plot.canvas.draw_idle()
