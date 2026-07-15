"""
qt_main.py — Qt application entry point.

Run:
    python qt_main.py
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from qt_shell import DataappMainWindow
from qt_theme import apply_theme
import qt_exception_hook


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    app.setApplicationName("Dataapp")
    apply_theme(app)
    qt_exception_hook.install(app)

    window = DataappMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
