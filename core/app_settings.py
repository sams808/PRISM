"""The one place PRISM constructs its QSettings.

Normally this is the per-user native store (the registry on Windows,
organisation "PRISM", application "PRISM"). With the environment variable
PRISM_SETTINGS_DIR set, it is an .ini file in that folder instead: the
installer self-test (``PRISM.exe --distcheck``) and scripted runs use that so
they never overwrite the user's own window geometry, last page or module
switches. (The self-test used to switch every module on in the real
settings, which is why a lab computer woke up with all modules enabled.)
"""
from __future__ import annotations

import os

ORGANISATION = "PRISM"
APPLICATION = "PRISM"
ENV_SETTINGS_DIR = "PRISM_SETTINGS_DIR"
INI_NAME = "PRISM.ini"


def app_settings():
    """A QSettings for PRISM: native per-user store, or an .ini file under
    PRISM_SETTINGS_DIR when that variable is set."""
    import PySide6.QtCore as qc   # resolved at call time: the test suite substitutes QSettings

    folder = os.environ.get(ENV_SETTINGS_DIR)
    if folder:
        os.makedirs(folder, exist_ok=True)
        ini_format = getattr(qc.QSettings, "IniFormat", None)
        return qc.QSettings(os.path.join(folder, INI_NAME), ini_format)
    return qc.QSettings(ORGANISATION, APPLICATION)
