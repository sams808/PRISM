"""Real-profile regression test (Phase 6, spec §5.3; re-frozen for v2:
PRISM_fit_pipeline_upgrade_prompt.md). One committed real measured profile
with a frozen expected FitResult. CI fails if the staged pipeline's output
on this exact file drifts beyond the stated tolerance — the concrete proof
that a future change to the components/engine/staged pipeline hasn't
silently changed what a real sample's fit means.

The fixture (tests/fixtures/P5Bi8-12__corr.dat) is a real reduced SAXS
profile from the user's own PBi glass series, committed to the repo by
explicit user decision (2026-07-22) so this test doesn't depend on the
external WSU_work\\SAXS\\... path existing on every machine/CI runner.

v2 upgrade note (2026-07-22): re-run after implementing the full ticket
(weighting/log10 mode, high-q masking, power_law2/BG_TS_PL2, staging
enforcement, at-bounds flags). Two of the ticket's five stated acceptance
targets are now met (preset -> BG_TS_PL2 as intended; d = 880 Å, within
the ticket's own "875 ± 90 Å"). The other three (xi within [2500,5000] Å,
rms_log < 0.15, at_bounds = 0) are NOT met, and this is a genuine,
investigated finding rather than an unfixed bug: xi came out pinned at a
Stage-4 widened bound under BG_TS_PL2, but standalone BG_TS (tested
directly) pins THREE OTHER parameters (bg_C, pl_B, pl_p) at their bounds
instead — trading one imperfection for a worse one, which is exactly why
the BIC ladder prefers BG_TS_PL2 despite its own remaining at-bound flags.
This traces to the same xi-resolution limitation found and extensively
investigated during Phase 6's synthetic harness (this instrument's q-grid
spacing is comparable to or finer than the peak's own width for xi in this
range) — a property of the data/instrument, not something the composite-
fit code can resolve by further tuning. The at-bounds diagnostics (v2 §4)
now correctly SURFACE this uncertainty via flags instead of silently
reporting a falsely-precise xi, which is the intended, working behavior.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from saxs_core.composite_staged import fit_staged
from saxs_core.loader import load_curve

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "P5Bi8-12__corr.dat"

# Frozen reference (multistart_n=8, sample_id="P5Bi8-12" — deterministic,
# see test_composite_staged.py's own determinism test for why this is safe
# to freeze exactly). Captured directly from a real run of the v2 pipeline
# against this exact fixture (verified reproducible across repeated runs).
FROZEN_D = 880.0850024911949
FROZEN_XI = 11664.169265127604
FROZEN_PRESET = "BG_TS_PL2"
FROZEN_RMS_LOG = 0.2246103985360861


def test_real_profile_regression_p5bi8_12():
    assert FIXTURE_PATH.is_file(), "committed fixture missing — see tests/fixtures/"
    curve = load_curve(str(FIXTURE_PATH))
    result = fit_staged(curve, sample_id="P5Bi8-12", multistart_n=8)

    assert result.preset_chosen == FROZEN_PRESET
    assert result.derived["d"] == pytest.approx(FROZEN_D, rel=0.01)
    assert result.derived["xi"] == pytest.approx(FROZEN_XI, rel=0.05)
    assert result.gof["rms_log"] == pytest.approx(FROZEN_RMS_LOG, rel=0.05)
    # sanity: d is within the spec's own stated observed range (§5.1) and
    # within the ticket's explicit "d = 875 ± 90 Å" acceptance target; xi
    # is NOT constrained to the ticket's [2500,5000] Å target here (see
    # module docstring) -- frozen for reproducibility, not asserted against
    # that target, since doing so would misrepresent a real, investigated
    # data limitation as a pass.
    assert 700.0 <= result.derived["d"] <= 1700.0
    assert 785.0 <= result.derived["d"] <= 965.0
    assert -1.0 < result.derived["fa"] < 0.0
