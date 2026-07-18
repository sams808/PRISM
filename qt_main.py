"""
qt_main.py — PRISM application entry point.

Run:
    python qt_main.py
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from qt_shell import DataappMainWindow
from qt_theme import apply_theme
import qt_exception_hook


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    from qt_help import APP_NAME, asset_path
    app.setApplicationName(APP_NAME)
    icon_path = asset_path("prism_logo.png")
    if os.path.isfile(icon_path):
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    splash = None
    splash_shown_at = 0.0
    splash_path = asset_path("prism_splash.png")
    if os.path.isfile(splash_path):
        import time
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QSplashScreen
        splash = QSplashScreen(QPixmap(splash_path))
        splash.show()
        splash_shown_at = time.time()
        app.processEvents()

    apply_theme(app)
    qt_exception_hook.install(app)

    window = DataappMainWindow()
    window.show()
    if splash is not None:
        # keep the logo up for at least 3 seconds (user request)
        import time
        while time.time() - splash_shown_at < 3.0:
            app.processEvents()
            time.sleep(0.02)
        splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
