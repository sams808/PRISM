"""
qt_home.py — the Home page: where the app is configured and started from.

Three things live here, by request of the group lead (2026-10-21):

1. **Modules**: one card per module with a description, its workspaces and
   an Enabled switch. Everything starts switched OFF on a fresh install and
   the choice is remembered per user, so a single-technique computer (the
   Raman spectrometer's PC) shows exactly the workspaces that matter there
   and nothing else; presets apply a whole configuration in one click.
   Disabled modules are hidden, not removed: switch them back on here.
2. **Reference databases**: what is installed on this computer, and the
   export / import of a database pack (core.db_packs) so a licensed
   colleague's PC receives the databases as one folder PRISM reads
   directly.
3. **Quick actions**: import, open a project, the guides.

The page talks to the shell through a small interface (module state,
presets, navigation, library import) rather than reaching into widgets.
"""
from __future__ import annotations

import os
import queue
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


class ModuleCard(QFrame):
    """One module: colour stripe, name, description, workspace buttons and
    the Enabled switch."""

    toggled = Signal(str, bool)
    open_requested = Signal(str)

    def __init__(self, name: str, color: str, description: str, pages: List[str], parent=None):
        super().__init__(parent)
        self.name = name
        self.pages = list(pages)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 10, 0)
        outer.setSpacing(10)
        stripe = QFrame()
        stripe.setFixedWidth(6)
        stripe.setStyleSheet(f"background: {color}; border-top-left-radius: 4px; border-bottom-left-radius: 4px;")
        outer.addWidget(stripe)

        body = QVBoxLayout()
        body.setContentsMargins(4, 10, 0, 10)
        body.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel(name)
        title.setObjectName("SectionTitle")
        head.addWidget(title, 1)
        self.switch = QCheckBox("Enabled")
        self.switch.toggled.connect(lambda on: self.toggled.emit(self.name, bool(on)))
        head.addWidget(self.switch)
        body.addLayout(head)
        desc = QLabel(description)
        desc.setObjectName("SectionNote")
        desc.setWordWrap(True)
        body.addWidget(desc)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.page_buttons: Dict[str, QPushButton] = {}
        for page in self.pages:
            b = QPushButton(page)
            b.setToolTip(f"Open the {page} workspace")
            b.clicked.connect(lambda _=False, p=page: self.open_requested.emit(p))
            row.addWidget(b)
            self.page_buttons[page] = b
        row.addStretch(1)
        body.addLayout(row)
        outer.addLayout(body, 1)

    def set_state(self, enabled: bool) -> None:
        """Reflect the shell's state without re-emitting toggled."""
        self.switch.blockSignals(True)
        self.switch.setChecked(bool(enabled))
        self.switch.blockSignals(False)
        for b in self.page_buttons.values():
            b.setEnabled(bool(enabled))


class HomePage(QWidget):
    def __init__(self, shell, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.shell = shell
        self.cards: Dict[str, ModuleCard] = {}
        self._db_items = None
        self._db_log: "queue.Queue[str]" = queue.Queue()
        self._db_timer = QTimer(self)
        self._db_timer.setInterval(200)
        self._db_timer.timeout.connect(self._drain_db_log)
        self._db_busy = False
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        from core.qt_help import APP_NAME, APP_TAGLINE, APP_VERSION, asset_path

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        # ---- header
        head = QHBoxLayout()
        logo = QLabel()
        p = asset_path("prism_logo.png")
        if os.path.isfile(p):
            from PySide6.QtGui import QPixmap
            pm = QPixmap(p)
            if not pm.isNull():
                logo.setPixmap(pm.scaledToHeight(56, Qt.SmoothTransformation))
        head.addWidget(logo)
        titles = QVBoxLayout()
        t = QLabel(f"{APP_NAME} {APP_VERSION}")
        t.setStyleSheet("font-size: 22px; font-weight: 600;")
        titles.addWidget(t)
        tag = QLabel(APP_TAGLINE)
        tag.setObjectName("SectionNote")
        titles.addWidget(tag)
        head.addLayout(titles, 1)
        actions = QHBoxLayout()
        for text, slot in (("Import files…", self.shell.library_page._on_import_clicked),
                           ("Open project…", self.shell.open_project),
                           ("Quick-start guide", self.shell.show_help)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            actions.addWidget(b)
        head.addLayout(actions)
        lay.addLayout(head)

        # ---- modules
        sect = QLabel("Modules")
        sect.setObjectName("SectionTitle")
        lay.addWidget(sect)
        note = QLabel("Switch on the techniques used on this computer; the others stay hidden until you "
                      "come back here. The Library is always available. Your choice is remembered.")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        lay.addWidget(note)
        presets = QHBoxLayout()
        presets.addWidget(QLabel("Presets:"))
        self.preset_buttons: Dict[str, QPushButton] = {}
        for preset in self.shell.module_presets():
            b = QPushButton(preset)
            b.clicked.connect(lambda _=False, name=preset: self.shell.apply_module_preset(name))
            presets.addWidget(b)
            self.preset_buttons[preset] = b
        presets.addStretch(1)
        self.modules_summary = QLabel("")
        self.modules_summary.setObjectName("SectionNote")
        presets.addWidget(self.modules_summary)
        lay.addLayout(presets)

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (name, color, description, pages) in enumerate(self.shell.module_catalog()):
            card = ModuleCard(name, color, description, pages)
            card.toggled.connect(self.shell.set_module_enabled)
            card.open_requested.connect(self.shell.navigate_to)
            self.cards[name] = card
            grid.addWidget(card, i // 3, i % 3)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        lay.addLayout(grid)

        # ---- databases
        sect2 = QLabel("Reference databases on this computer")
        sect2.setObjectName("SectionTitle")
        lay.addWidget(sect2)
        note2 = QLabel("Export packs the installed databases into one folder that another PRISM imports as is "
                       "(for colleagues who hold the same database rights, e.g. the Raman spectrometer's PC). "
                       "RRUFF and AMCSD are downloaded from the Raman ID workspace; XRD card databases and CIF "
                       "folders are registered in the XRD ID workspace.")
        note2.setObjectName("SectionNote")
        note2.setWordWrap(True)
        lay.addWidget(note2)
        self.db_card = QWidget()
        self.db_card.setObjectName("Card")
        db_lay = QVBoxLayout(self.db_card)
        db_lay.setContentsMargins(12, 10, 12, 10)
        self.db_grid = QGridLayout()
        self.db_grid.setHorizontalSpacing(18)
        self.db_grid.setVerticalSpacing(4)
        db_lay.addLayout(self.db_grid)
        self.db_rows: Dict[str, tuple] = {}
        btns = QHBoxLayout()
        self.db_export_btn = QPushButton("Export databases…")
        self.db_export_btn.setObjectName("Primary")
        self.db_export_btn.clicked.connect(self.export_databases)
        btns.addWidget(self.db_export_btn)
        self.db_import_btn = QPushButton("Import a database pack…")
        self.db_import_btn.clicked.connect(self.import_databases)
        btns.addWidget(self.db_import_btn)
        self.db_refresh_btn = QPushButton("Refresh")
        self.db_refresh_btn.clicked.connect(lambda: self.refresh_databases(force=True))
        btns.addWidget(self.db_refresh_btn)
        btns.addStretch(1)
        db_lay.addLayout(btns)
        self.db_status = QLabel("")
        self.db_status.setObjectName("SectionNote")
        self.db_status.setWordWrap(True)
        db_lay.addWidget(self.db_status)
        lay.addWidget(self.db_card)
        lay.addStretch(1)

        self.refresh_modules()

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------
    def refresh_modules(self) -> None:
        on = []
        for name, card in self.cards.items():
            enabled = self.shell.is_module_enabled(name)
            card.set_state(enabled)
            if enabled:
                on.append(name)
        self.modules_summary.setText(
            f"{len(on)} of {len(self.cards)} modules on" + (f": {', '.join(on)}" if on else " — switch some on to start")
        )

    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().showEvent(event)
        if self._db_items is None:
            self.refresh_databases()

    def refresh_databases(self, force: bool = False) -> None:
        if self._db_busy:
            return
        if self._db_items is not None and not force:
            return
        import core.db_packs as dbp
        from core.qt_worker import run_in_thread
        self.db_status.setText("Reading the database inventory…")
        run_in_thread(lambda: dbp.inventory(sizes=False), self._on_inventory, self._on_db_error)

    def _on_inventory(self, items) -> None:
        self._db_items = list(items)
        # rebuild the rows
        while self.db_grid.count():
            it = self.db_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.db_rows = {}
        for r, item in enumerate(self._db_items):
            name = QLabel(item.label)
            name.setStyleSheet("font-weight: 600;")
            status = QLabel(("installed — " if item.installed else "") + item.detail)
            status.setObjectName("SectionNote" if not item.installed else "")
            status.setWordWrap(True)
            self.db_grid.addWidget(name, r, 0)
            self.db_grid.addWidget(status, r, 1)
            self.db_rows[item.key] = (name, status)
        self.db_grid.setColumnStretch(1, 1)
        n_inst = sum(1 for it in self._db_items if it.installed)
        self.db_status.setText(f"{n_inst} of {len(self._db_items)} database kinds installed.")

    def _on_db_error(self, traceback_text: str) -> None:
        self._db_busy = False
        self._db_timer.stop()
        self.db_export_btn.setEnabled(True)
        self.db_import_btn.setEnabled(True)
        self.db_status.setText("Database operation failed — see the dialog.")
        QMessageBox.critical(self, "Databases", traceback_text)

    def _drain_db_log(self) -> None:
        last = None
        while not self._db_log.empty():
            last = self._db_log.get_nowait()
        if last is not None:
            self.db_status.setText(last)

    def _start_db_job(self, fn, on_done) -> None:
        from core.qt_worker import run_in_thread
        self._db_busy = True
        self.db_export_btn.setEnabled(False)
        self.db_import_btn.setEnabled(False)
        self._db_timer.start()
        run_in_thread(fn, lambda result: self._finish_db_job(on_done, result), self._on_db_error)

    def _finish_db_job(self, on_done, result) -> None:
        self._db_busy = False
        self._db_timer.stop()
        self._drain_db_log()
        self.db_export_btn.setEnabled(True)
        self.db_import_btn.setEnabled(True)
        on_done(result)
        self._db_items = None
        self.refresh_databases(force=True)

    def export_databases(self) -> None:
        import core.db_packs as dbp
        items = [it for it in (self._db_items or dbp.inventory()) if it.installed]
        if not items:
            QMessageBox.information(self, "Export databases", "No database is installed on this computer yet.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder to write the database pack into")
        if not folder:
            return
        listing = "\n".join(f"  • {it.label}: {it.detail}" for it in items)
        resp = QMessageBox.question(
            self, "Export databases",
            f"Write these databases into\n{folder}\n\n{listing}\n\n"
            "Share the pack only with people who hold the same rights to every database in it "
            "(RRUFF and AMCSD are redistributable with attribution; a licensed card database is not). Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        keys = [it.key for it in items]
        self._start_db_job(lambda: dbp.export_pack(folder, keys, log=self._db_log.put),
                           lambda manifest: QMessageBox.information(
                               self, "Export databases",
                               f"Pack written to {folder}\n({len(manifest.get('items', {}))} item(s)). "
                               "Copy that folder to the other computer and use 'Import a database pack…' there."))

    def import_databases(self) -> None:
        import core.db_packs as dbp
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick the pack's manifest", "", f"PRISM database pack ({dbp.MANIFEST_NAME});;All files (*.*)")
        if not path:
            return
        try:
            manifest = dbp.read_pack_manifest(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import databases", str(exc))
            return
        items = manifest.get("items", {})
        if not items:
            QMessageBox.information(self, "Import databases", "This pack holds no database.")
            return
        listing = "\n".join(f"  • {dbp.LABELS.get(k, k)}" + (f": {v.get('detail')}" if isinstance(v, dict) and v.get("detail") else "")
                            for k, v in items.items())
        resp = QMessageBox.question(
            self, "Import databases",
            f"Install into this computer's PRISM cache:\n\n{listing}\n\n"
            "Existing databases of the same kind are replaced (RRUFF, AMCSD) or added to (XRD, CIF folders, "
            "Raman library). Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        src = manifest["_dir"]
        self._start_db_job(lambda: dbp.import_pack(src, log=self._db_log.put),
                           lambda done: QMessageBox.information(
                               self, "Import databases",
                               "Installed:\n" + "\n".join(f"  • {dbp.LABELS.get(k, k)}: {v}" for k, v in done.items())))
