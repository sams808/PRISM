"""
qt_main.py — PRISM application entry point.

Run:
    python qt_main.py

Import order matters here: the splash screen goes up FIRST, and only then
do the heavyweight imports (qt_shell pulls in matplotlib, scipy, pandas,
lmfit, every workspace) run — so the user sees the logo within ~1 s instead
of staring at nothing for the whole import phase.
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtWidgets import QApplication


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

    # Heavy imports happen behind the splash, not before it.
    from qt_shell import PrismMainWindow
    from qt_theme import apply_theme
    import qt_exception_hook

    apply_theme(app)
    qt_exception_hook.install(app)

    window = PrismMainWindow()
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
