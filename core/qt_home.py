"""
qt_home.py — the Home page and the Settings dialog.

Home shows what this computer is set up for: a card per ENABLED module
with buttons to its workspaces, the quick actions, and one line about the
reference databases. Nothing else — a Raman station shows the Raman
cards and no trace of SAXS or XAS.

Switching modules on and off, the presets and the database export /
import live in the Settings dialog (Settings › Modules…, Settings ›
Reference databases…), so the configuration is one menu away but never
in the way. A fresh install starts with every module off and Home says
so, with the button that opens the dialog.

The page and the dialog talk to the shell through a small interface
(module catalogue and state, presets, navigation, library import).
"""
from __future__ import annotations

import os
import queue
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)


class ModuleCard(QFrame):
    """One module: colour stripe, name, description, workspace buttons and,
    in the Settings dialog, the Enabled switch."""

    toggled = Signal(str, bool)
    open_requested = Signal(str)

    def __init__(self, name: str, color: str, description: str, pages: List[str], parent=None, *, with_switch: bool = True):
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
        self.switch: Optional[QCheckBox] = None
        if with_switch:
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
        if self.switch is not None:
            self.switch.blockSignals(True)
            self.switch.setChecked(bool(enabled))
            self.switch.blockSignals(False)
        for b in self.page_buttons.values():
            b.setEnabled(bool(enabled))


class DatabasePanel(QWidget):
    """Inventory of the reference databases on this computer, and the pack
    export / import (core.db_packs)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._db_items = None
        self._db_log: "queue.Queue[str]" = queue.Queue()
        self._db_timer = QTimer(self)
        self._db_timer.setInterval(200)
        self._db_timer.timeout.connect(self._drain_db_log)
        self._db_busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        note = QLabel("Export packs the installed databases into one folder that another PRISM imports as is "
                      "(for colleagues who hold the same database rights, e.g. the Raman spectrometer's PC). "
                      "RRUFF and AMCSD are downloaded from the Raman ID workspace; XRD card databases and CIF "
                      "folders are registered in the XRD ID workspace.")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        lay.addWidget(note)
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

    def summary(self) -> str:
        if self._db_items is None:
            return ""
        inst = [it.label for it in self._db_items if it.installed]
        return ("Installed: " + ", ".join(inst)) if inst else "No reference database installed yet."

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


class SettingsDialog(QDialog):
    """Settings › Modules… / Reference databases…: the module switches with
    their presets, and the database manager."""

    def __init__(self, shell, parent: Optional[QWidget] = None):
        super().__init__(parent or shell)
        self.shell = shell
        self.setWindowTitle("Settings")
        self.resize(900, 620)
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # ---- Modules
        page = QWidget()
        pl = QVBoxLayout(page)
        note = QLabel("Switch on the techniques used on this computer; the others stay hidden until switched back on "
                      "here. The Library is always available. The choice is remembered per user.")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        pl.addWidget(note)
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
        pl.addLayout(presets)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(10)
        self.cards: Dict[str, ModuleCard] = {}
        for i, (name, color, description, pages) in enumerate(self.shell.module_catalog()):
            card = ModuleCard(name, color, description, pages, with_switch=True)
            card.toggled.connect(self.shell.set_module_enabled)
            card.open_requested.connect(self._open_page)
            self.cards[name] = card
            grid.addWidget(card, i // 2, i % 2)
        grid.setRowStretch(grid.rowCount(), 1)
        scroll.setWidget(inner)
        pl.addWidget(scroll, 1)
        self.tabs.addTab(page, "Modules")

        # ---- Databases
        self.db_panel = DatabasePanel()
        self.tabs.addTab(self.db_panel, "Reference databases")

        btns = QHBoxLayout()
        btns.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        btns.addWidget(close)
        lay.addLayout(btns)
        self.refresh_modules()

    def _open_page(self, page: str) -> None:
        self.shell.navigate_to(page)
        self.close()

    def refresh_modules(self) -> None:
        on = []
        for name, card in self.cards.items():
            enabled = self.shell.is_module_enabled(name)
            card.set_state(enabled)
            if enabled:
                on.append(name)
        self.modules_summary.setText(f"{len(on)} of {len(self.cards)} modules on")

    def show_tab(self, name: str) -> None:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).lower().startswith(name.lower()):
                self.tabs.setCurrentIndex(i)
                break
        self.refresh_modules()
        self.show()
        self.raise_()
        self.activateWindow()


class HomePage(QWidget):
    def __init__(self, shell, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.shell = shell
        self.cards: Dict[str, ModuleCard] = {}
        self._build_ui()

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
                           ("Quick-start guide", self.shell.show_help),
                           ("Settings…", lambda: self.shell.open_settings("Modules"))):
            b = QPushButton(text)
            b.setToolTip({"Import files…": "Open data files into the Library (Ctrl+O); files can also be dropped on the window",
                          "Open project…": "Open a saved .prism project: every spectrum, result and setting of a session",
                          "Quick-start guide": "The guide to every page (F1)",
                          "Settings…": "Switch modules on or off, presets, reference databases"}.get(text, ""))
            b.clicked.connect(slot)
            actions.addWidget(b)
        head.addLayout(actions)
        lay.addLayout(head)

        self.modules_title = QLabel("Your modules")
        self.modules_title.setObjectName("SectionTitle")
        lay.addWidget(self.modules_title)
        self.empty_note = QLabel("No module is switched on. Choose the techniques used on this computer; "
                                 "the choice is remembered on this computer.")
        self.empty_note.setObjectName("SectionNote")
        self.empty_note.setWordWrap(True)
        lay.addWidget(self.empty_note)
        self.choose_btn = QPushButton("Choose modules…")
        self.choose_btn.setToolTip("Switch on the techniques used on this computer (Settings › Modules…)")
        self.choose_btn.setObjectName("Primary")
        self.choose_btn.setMaximumWidth(200)
        self.choose_btn.clicked.connect(lambda: self.shell.open_settings("Modules"))
        lay.addWidget(self.choose_btn)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        lay.addWidget(self.grid_widget)

        db_row = QHBoxLayout()
        self.db_line = QLabel("")
        self.db_line.setObjectName("SectionNote")
        self.db_line.setWordWrap(True)
        db_row.addWidget(self.db_line, 1)
        manage = QPushButton("Reference databases…")
        manage.setToolTip("What is installed on this computer, and the export / import of database packs")
        manage.clicked.connect(lambda: self.shell.open_settings("Reference"))
        db_row.addWidget(manage)
        lay.addLayout(db_row)
        lay.addStretch(1)
        self.refresh_modules()

    def refresh_modules(self) -> None:
        """Rebuild the grid with the enabled modules only."""
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)   # out of the layout AND off the page now, not at the next event-loop turn
                w.deleteLater()
        self.cards = {}
        enabled = [(name, color, desc, pages) for name, color, desc, pages in self.shell.module_catalog()
                   if self.shell.is_module_enabled(name)]
        for i, (name, color, desc, pages) in enumerate(enabled):
            card = ModuleCard(name, color, desc, pages, with_switch=False)
            card.open_requested.connect(self.shell.navigate_to)
            self.cards[name] = card
            self.grid.addWidget(card, i // 3, i % 3)
        for c in range(3):
            self.grid.setColumnStretch(c, 1)
        self.empty_note.setVisible(not enabled)
        self.choose_btn.setVisible(not enabled)
        self.grid_widget.setVisible(bool(enabled))
        self.modules_title.setText("Your modules" if enabled else "Welcome")

    def refresh_databases(self) -> None:
        try:
            import core.db_packs as dbp
            inst = [it.label for it in dbp.inventory(sizes=False) if it.installed]
        except Exception:
            inst = []
        self.db_line.setText(("Reference databases on this computer: " + ", ".join(inst))
                             if inst else "No reference database on this computer yet (RRUFF, AMCSD, your Raman library…).")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().showEvent(event)
        if not self.db_line.text():
            self.refresh_databases()
