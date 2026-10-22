"""core.app_settings: one QSettings constructor, isolated when PRISM_SETTINGS_DIR is set."""
from __future__ import annotations

import os


def test_app_settings_uses_an_ini_file_when_the_env_var_is_set(monkeypatch, tmp_path):
    import PySide6.QtCore as qc
    from core.app_settings import app_settings
    calls = []

    class Recorder:
        IniFormat = "ini"

        def __init__(self, *args):
            calls.append(args)

    monkeypatch.setattr(qc, "QSettings", Recorder)
    monkeypatch.setenv("PRISM_SETTINGS_DIR", str(tmp_path / "settings"))
    app_settings()
    assert calls[-1][0].endswith("PRISM.ini") and calls[-1][1] == "ini" and (tmp_path / "settings").is_dir()
    monkeypatch.delenv("PRISM_SETTINGS_DIR")
    app_settings()
    assert calls[-1] == ("PRISM", "PRISM")


def test_distcheck_runs_on_a_throwaway_settings_file(qtbot, monkeypatch):
    """The installer self-test switches every module on and visits every
    page: that must land in a temporary settings file, never in the user's
    own settings (2.8.0 left a lab computer with all modules enabled)."""
    import qt_main
    monkeypatch.setenv("PRISM_SETTINGS_DIR", "")      # recorded, restored after the test
    lines = []
    assert qt_main._cli_distcheck([], log=lines.append) == 0, lines
    assert "prism_distcheck_" in os.environ.get("PRISM_SETTINGS_DIR", "")
    assert any("OK: distcheck passed" in line for line in lines)
