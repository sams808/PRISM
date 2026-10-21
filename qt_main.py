"""
qt_main.py — PRISM application entry point.

Run:
    python qt_main.py

Import order matters here: the splash screen goes up FIRST, and only then
do the heavyweight imports (qt_shell pulls in matplotlib, scipy, pandas,
lmfit, every workspace) run — so the user sees the logo within ~1 s instead
of staring at nothing for the whole import phase.

Also doubles as a headless CLI for setting up the local reference databases
without ever opening the GUI — what the shipped Download-*.bat/.ps1 scripts
run, so a colleague with only the portable exe (no Python) can get RRUFF and
AMCSD data with a double-click, not a Python one-liner:
    PRISM.exe --build-rruff-cache [--categories excellent_oriented ...]
    PRISM.exe --build-amcsd-cache
(equivalently `python qt_main.py --build-rruff-cache` from source).
"""
from __future__ import annotations

import logging
import os
import sys


def _configure_headless_stdio() -> None:
    """A PyInstaller --windowed build has NO console attached: sys.stdout
    and sys.stderr are None, and the first bare print() anywhere (this
    module's own CLI progress logging, or any library's) raises
    AttributeError instead of doing nothing. Redirect to os.devnull so
    output is silently discarded rather than crashing. Idempotent — safe
    to call from both the GUI and CLI paths."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def _cli_log(log_path: str):
    """A log function for the headless CLI paths: prints (safe — stdio is
    already guarded by the time this is used) AND appends to a log file
    next to the exe/script, so `PRISM.exe --build-rruff-cache` run from a
    .bat with no visible console still leaves a trail the user can read."""
    import datetime

    def log(msg: str) -> None:
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        print(line)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    return log


def _cli_build_rruff_cache(argv, log=None) -> int:
    """`--build-rruff-cache [--categories a b c]`. Returns a process exit
    code (0 success, 1 failure) — never raises, so it's safe to call
    directly from main()."""
    import raman.rruff_science as rs

    log = log or _cli_log(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "rruff_download.log"))
    categories = None
    if "--categories" in argv:
        categories = argv[argv.index("--categories") + 1:]
        categories = [c for c in categories if not c.startswith("--")] or None
    try:
        n = rs.download_and_build_rruff_cache(categories=categories, log=log)
        log(f"DONE: {n} spectra indexed. Open the Raman ID workspace in PRISM.")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


def _cli_build_amcsd_cache(argv, log=None) -> int:
    """`--build-amcsd-cache` — the AMCSD counterpart, for the Raman ID
    workspace's "Overlay candidate's XRD (CIF)" button."""
    import raman.rruff_science as rs

    log = log or _cli_log(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "amcsd_download.log"))
    try:
        n = rs.download_and_build_amcsd_cache(log=log)
        log(f"DONE: {n} CIF structures indexed.")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


def _cli_distcheck(argv, log=None) -> int:
    """`--distcheck`: build the whole application off-screen, visit every
    workspace, open the bundled demo project if it is present, and exit 0.
    The installer's and the portable zip's self-test — a frozen build that
    misses a module or a data file dies here, in a log next to the exe,
    instead of on a colleague's desk. Returns 1 on any failure."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    log = log or _cli_log(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "prism_distcheck.log"))
    t0 = time.time()
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from core.qt_help import APP_VERSION, HELP_HTML, MODULE_GUIDES
        from core.qt_shell import NAV_ITEMS, PrismMainWindow
        from core.qt_theme import apply_theme
        apply_theme(app)
        log(f"PRISM {APP_VERSION} distcheck: python {sys.version.split()[0]}, frozen={hasattr(sys, '_MEIPASS')}")
        window = PrismMainWindow()
        window.show()
        app.processEvents()
        for mod in list(window.module_checks):
            window.set_module_enabled(mod, True)
        for row, name in enumerate(NAV_ITEMS):
            window.nav.setCurrentRow(row)
            app.processEvents()
            if window.stack.currentWidget() is not window._pages_by_nav[name]:
                raise RuntimeError(f"nav row {row} did not open {name}")
        log(f"  {len(NAV_ITEMS)} workspaces opened; {len(MODULE_GUIDES)} module guides, help {len(HELP_HTML)} chars")
        import numpy, scipy, matplotlib, lmfit, pandas, rampy, xraydb  # noqa: F401 - the frozen science stack
        log(f"  numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}, "
            f"lmfit {lmfit.__version__}, pandas {pandas.__version__}, xraydb {xraydb.__version__}")
        # a CIF powder pattern needs xraydb's data file to have been bundled
        import raman.raman_tools as rt
        rows = rt.peak_table([1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 5, 1, 0, 0])
        if len(rows) != 1:
            raise RuntimeError("peak_table self-test failed")
        import core.io_universal as iou
        base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(os.path.abspath(__file__))
        demo = os.path.join(base, "EXAMPLES", "Raman_example.txt")
        if os.path.isfile(demo):
            df = iou.load_any(demo)
            log(f"  EXAMPLES/Raman_example.txt: {len(df)} rows")
        deadline = time.time() + 0.5
        while time.time() < deadline:
            app.processEvents()
        window.close()
        app.processEvents()
        log(f"OK: distcheck passed in {time.time() - t0:.1f} s")
        return 0
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        return 1


def main() -> int:
    _configure_headless_stdio()

    argv = sys.argv[1:]
    if "--build-rruff-cache" in argv:
        return _cli_build_rruff_cache(argv)
    if "--build-amcsd-cache" in argv:
        return _cli_build_amcsd_cache(argv)
    if "--distcheck" in argv:
        return _cli_distcheck(argv)

    from PySide6.QtWidgets import QApplication

    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    from core.qt_help import APP_NAME, asset_path
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
    from core.qt_shell import PrismMainWindow
    from core.qt_theme import apply_theme
    import core.qt_exception_hook as qt_exception_hook

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
