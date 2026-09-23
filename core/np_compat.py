"""Small numpy compatibility layer: names that moved between numpy 1.x and 2.x.

``trapezoid`` is ``numpy.trapezoid`` (numpy >= 2.0) or ``numpy.trapz`` (numpy 1.x, where the new
name does not exist; the old name was removed in numpy 2.4). Import it from here instead of
calling either directly, so PRISM runs on the pinned numpy 1.26 and on a newer install alike.
"""
from __future__ import annotations

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

__all__ = ["trapezoid"]
