"""Real-profile regression test (Phase 6, spec §5.3): one committed real
measured profile with a frozen expected FitResult. CI fails if the staged
pipeline's output on this exact file drifts beyond the stated tolerance —
the concrete proof that a future change to the components/engine/staged
pipeline hasn't silently changed what a real sample's fit means.

The fixture (tests/fixtures/P5Bi8-12__corr.dat) is a real reduced SAXS
profile from the user's own PBi glass series, committed to the repo by
explicit user decision (2026-07-22) so this test doesn't depend on the
external WSU_work\\SAXS\\... path existing on every machine/CI runner.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from saxs_core.composite_staged import fit_staged
from saxs_core.loader import load_curve

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "P5Bi8-12__corr.dat"

# Frozen reference (multistart_n=8, sample_id="P5Bi8-12" — deterministic,
# see test_composite_staged.py's own determinism test for why this is safe
# to freeze exactly). Captured directly from a real run of the pipeline
# against this exact fixture.
FROZEN_D = 875.3408256313131
FROZEN_XI = 3757.810215547226
FROZEN_PRESET = "BG_TS_GP"


def test_real_profile_regression_p5bi8_12():
    assert FIXTURE_PATH.is_file(), "committed fixture missing — see tests/fixtures/"
    curve = load_curve(str(FIXTURE_PATH))
    result = fit_staged(curve, sample_id="P5Bi8-12", multistart_n=8)

    assert result.preset_chosen == FROZEN_PRESET
    assert result.derived["d"] == pytest.approx(FROZEN_D, rel=0.01)
    assert result.derived["xi"] == pytest.approx(FROZEN_XI, rel=0.05)
    # sanity: within the spec's own stated observed ranges (§5.1)
    assert 700.0 <= result.derived["d"] <= 1700.0
    assert 2500.0 <= result.derived["xi"] <= 5000.0
    assert -1.0 < result.derived["fa"] < 0.0
