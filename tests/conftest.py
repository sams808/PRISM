"""Shared pytest fixtures: paths to bundled EXAMPLES data and the archived
real-world dataset. Fixture files are RAW inputs only — never outputs from
the (historically buggy) pipeline. Expected values are derived from
hand-verified/fixed computations or invariant checks, not from snapshotting
old pipeline output.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATAAPP_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = DATAAPP_ROOT / "EXAMPLES"
ARCHIVE_DIR = DATAAPP_ROOT.parent / "data_and_notebooks"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def archive_dir() -> Path:
    return ARCHIVE_DIR


@pytest.fixture(scope="session")
def dta_example_path() -> Path:
    return EXAMPLES_DIR / "DTA_example.txt"


@pytest.fixture(scope="session")
def raman_example_path() -> Path:
    return EXAMPLES_DIR / "Raman_example.txt"


@pytest.fixture(scope="session")
def xrd_example_path() -> Path:
    return EXAMPLES_DIR / "XRD_example.xy"


@pytest.fixture(scope="session")
def saxs_example_path() -> Path:
    return EXAMPLES_DIR / "SAXS_example.dat"


@pytest.fixture(scope="session")
def rruff_sample_zip_path() -> Path:
    """A real, small (271K, 17 spectra) RRUFF bulk-download ZIP
    (fair_oriented.zip, fetched 2026-07-15 from
    https://www.rruff.net/zipped_data_files/raman/), checked in as a fixture
    rather than downloaded per-test-run — mirrors EXAMPLES/HTXRD_example.rasx."""
    return EXAMPLES_DIR / "RRUFF_fair_oriented_sample.zip"


@pytest.fixture(scope="session")
def isg_series_paths() -> list[Path]:
    """Real archived Raman series (headerless 2-column, same shape as EXAMPLES)."""
    return sorted(ARCHIVE_DIR.glob("ISG_*gpa.txt"))


@pytest.fixture(scope="session")
def pbi0_map_paths() -> list[Path]:
    """Real archived multi-file Raman map series — RAW map spectra only
    (`...-mapN.txt`). The same folder also holds derived outputs from past
    processing runs (`..._bl.txt` baseline-subtracted, `..._bl_fit....txt`
    fit-parameter tables with ~8 rows) that the old `*-map*.txt` glob
    silently swept in; a fit-parameter table is not a spectrum."""
    import re
    return sorted(
        p for p in (ARCHIVE_DIR / "PBi0-1").glob("*-map*.txt")
        if re.search(r"-map\d+\.txt$", p.name)
    )


def larch_available() -> bool:
    try:
        import larch  # noqa: F401
        return True
    except Exception:
        return False


requires_larch = pytest.mark.skipif(not larch_available(), reason="larch is not installed")


@pytest.fixture(autouse=True, scope="session")
def _synchronous_workers():
    """Run qt_worker background jobs inline during tests, so tests assert
    immediately after triggering an operation instead of polling for a
    worker thread to finish. Production code never flips this switch."""
    try:
        import qt_worker
    except Exception:
        yield
        return
    qt_worker.set_synchronous(True)
    yield
    qt_worker.set_synchronous(False)


@pytest.fixture(autouse=True)
def _prevent_blocking_qt_dialogs(monkeypatch):
    """Autouse safety net for Qt tests: an accidental REAL (unmocked)
    QMessageBox.information/warning/critical/question call opens a real
    modal dialog and blocks the event loop indefinitely waiting for a human
    click — fatal during an unattended/autonomous run. This is not
    theoretical: an unmocked success-path QMessageBox.information() in
    test_qt_single_fit.py's export-components test caused a real ~74-minute
    hang while verifying M8 (eventually returned exit 0, presumably via some
    outer supervisor timeout — not something to rely on). Default every
    convenience dialog to an immediate no-op/accept; a test that specifically
    wants to verify a dialog appeared can still monkeypatch over this within
    its own body (a test-local monkeypatch.setattr applied after this
    fixture's setup wins).

    The sys.modules guard (skip when PySide6 isn't already imported) is a
    holdover from when this suite also contained Tk-based tests and
    importing PySide6 process-wide would have corrupted Tk's Tcl
    interpreter on Windows; the Tk tests are gone now, but the guard is
    still harmlessly correct — pure-science test sessions simply never
    need QMessageBox patched.
    """
    import sys
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
