"""Real-data regression of the SAXS frame-to-curve chain against the author's hand-made reference package.

Skipped unless two environment variables point at the reference material:

* ``PRISM_SAXS_REFERENCE_DIR``: the folder holding ``phase1_summary.csv`` and ``phase2_modelfree.csv``
  (optionally ``phase1_results.npz``, the full-precision arrays of the reference run);
* ``PRISM_SAXS_REDUCED_V2_DIR``: the folder of the reference reduced curves ``<sample>__v2.dat``.

The raw frames are located from the ``sample_frame`` headers of the reduced curves (override with
``PRISM_SAXS_RAW_DIR``). Nothing in this file names a sample: every identity comes from the reference
tables. Acceptance criteria: pairing identical for every frame, curves within 1e-9 relative of the
full-precision reference (2e-6 of its 7-digit text), reduction summary within 1e-6, descriptors within
1 % (1e-9 when computed from the same text curves). The reference predates the October 2026 correction of the
Poisson sigma (one photon = 1.00), so the chain runs with ``flatfield_method="legacy"`` here. The physical fits are
covered by the synthetic tests of test_fit_workflow.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

REF_DIR = os.environ.get("PRISM_SAXS_REFERENCE_DIR")
RED_DIR = os.environ.get("PRISM_SAXS_REDUCED_V2_DIR")
pytestmark = pytest.mark.skipif(not (REF_DIR and RED_DIR and os.path.isdir(REF_DIR) and os.path.isdir(RED_DIR)),
                                reason="PRISM_SAXS_REFERENCE_DIR / PRISM_SAXS_REDUCED_V2_DIR not set")


def _norm(p: str) -> str:
    return str(p).replace("\\", "/")


def _raw_root(summary: pd.DataFrame) -> str:
    override = os.environ.get("PRISM_SAXS_RAW_DIR")
    if override:
        return override
    # the reference headers carry absolute frame paths; the summary carries the same paths relative to the root
    from saxs.core.sector_batch import read_dat_header
    row = summary.iloc[0]
    meta = read_dat_header(os.path.join(RED_DIR, f"{row['sample']}__v2.dat"))
    abs_path = _norm(meta["sample_frame"])
    rel = _norm(row["sample_frame"])
    assert abs_path.endswith(rel), (abs_path, rel)
    root = abs_path[: -len(rel)].rstrip("/")
    if os.path.exists(os.path.join(root, rel)):
        return root
    # the session folders may since have been moved one level down (an archive subfolder, say);
    # look for them rather than fail with an empty reduction that would read as a science failure
    if os.path.isdir(root):
        for sub in sorted(os.listdir(root)):
            cand = os.path.join(root, sub)
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, rel)):
                return _norm(cand)
    pytest.skip(f"the reference frames are no longer under {root!r} ({rel}); set PRISM_SAXS_RAW_DIR")


@pytest.fixture(scope="module")
def reference():
    summary = pd.read_csv(os.path.join(REF_DIR, "phase1_summary.csv"))
    return dict(summary=summary, raw_root=_raw_root(summary),
                modelfree=pd.read_csv(os.path.join(REF_DIR, "phase2_modelfree.csv")).set_index("sample"))


@pytest.fixture(scope="module")
def pairing(reference):
    from saxs.core import frames as fr
    skipped = []
    records = fr.scan_frames(reference["raw_root"], diagnostics=False, on_error=lambda p, e: skipped.append((p, str(e))))
    # virtual-detector exports carry no wavelength and are not production frames
    assert all("WaveLength" in msg for _p, msg in skipped), skipped
    return records, fr.pair_frames(records, rule="same_folder")


def test_pairing_equals_reference_for_every_frame(reference, pairing):
    summary = reference["summary"]
    _records, res = pairing
    got = {_norm(p.sample.rel): _norm(p.empty.rel) for p in res.pairs}
    missing, wrong = [], []
    for _, r in summary.iterrows():
        s, e = _norm(r["sample_frame"]), _norm(r["empty_frame"])
        if s not in got:
            missing.append(s)
        elif got[s] != e:
            wrong.append((s, got[s], e))
    assert not missing, missing
    assert not wrong, wrong
    assert len(summary) == 43
    names = {p.sample.name for p in res.pairs}
    assert set(summary["sample"]) <= names


@pytest.fixture(scope="module")
def reduced(reference, pairing):
    from saxs.core import sector_batch as sb
    summary = reference["summary"]
    _records, res = pairing
    wanted = {_norm(r["sample_frame"]): r["sample"] for _, r in summary.iterrows()}
    pairs = [p for p in res.pairs if _norm(p.sample.rel) in wanted]
    mu = sb.MuSource.from_table({r["sample"]: r["mu_mm"] for _, r in summary.iterrows()})
    results = sb.reduce_pairs(pairs, sb.ReductionOptions(flatfield_method="legacy"), mu)     # as the reference was made
    return {r.name: r for r in results}


def _read_v2(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("q_A"):
                continue
            rows.append([float(x) for x in s.split()])
    return np.asarray(rows)


def _match_q(q, q_ref, rtol=1e-9):
    """(a, b): indices of the points of q that coincide with a point of q_ref (1-pixel rings)."""
    order = np.argsort(q_ref)
    pos = np.searchsorted(q_ref[order], q)
    a, b = [], []
    for i, v in enumerate(q):
        for j in (pos[i] - 1, pos[i]):
            if 0 <= j < len(order) and abs(q_ref[order[j]] - v) <= rtol * v:
                a.append(i); b.append(order[j])
                break
    return np.asarray(a, int), np.asarray(b, int)


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if len(b) else 0.0


def test_reduced_curves_equal_reference_within_1e_9(reference, reduced):
    """Against the full-precision arrays of the reference run (phase1_results.npz, 1e-9 relative) and
    against the published 7-digit .dat text (2e-6 relative)."""
    summary = reference["summary"].set_index("sample")
    npz_path = os.path.join(REF_DIR, "phase1_results.npz")
    npz = np.load(npz_path) if os.path.exists(npz_path) else None
    worst_I, worst_s, n_missing = 0.0, 0.0, 0
    for name, r in reduced.items():
        q, I, s = r.curve.q, r.curve.intensity, r.curve.sigma
        ref = _read_v2(os.path.join(RED_DIR, f"{name}__v2.dat"))
        q_ref, I_ref, s_ref, ipm_ref = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
        a, b = _match_q(q, q_ref, rtol=2e-6)              # the text carries 7 significant digits
        assert len(a) >= len(q_ref) - 2, (name, len(a), len(q_ref))
        n_missing += len(q_ref) - len(a)
        ok = np.isfinite(I_ref[b]) & np.isfinite(I[a])
        assert _rel(I[a][ok], I_ref[b][ok]) < 2e-6, (name, "I vs text")
        assert _rel(s[a][ok], s_ref[b][ok]) < 2e-6, (name, "sigma vs text")
        pm = r.curve_per_mm()
        if pm is not None and np.isfinite(ipm_ref[b][ok]).any():
            assert _rel(pm.intensity[a][ok], ipm_ref[b][ok]) < 2e-6, (name, "I per mm vs text")
        assert np.array_equal(r.columns["usable"][a], ref[:, 9][b].astype(int)), name
        if npz is not None and f"{name}__I" in npz.files:
            q_np, I_np, s_np, flag_np = (npz[f"{name}__{k}"] for k in ("q", "I", "sig", "flag"))
            a2, b2 = _match_q(q, q_np, rtol=1e-9)
            assert len(a2) == len(q), (name, "every ring of ours is a ring of the reference")
            ok2 = np.isfinite(I_np[b2]) & np.isfinite(I[a2])
            rel_I, rel_s = _rel(I[a2][ok2], I_np[b2][ok2]), _rel(s[a2][ok2], s_np[b2][ok2])
            worst_I, worst_s = max(worst_I, rel_I), max(worst_s, rel_s)
            assert rel_I < 1e-9, (name, rel_I)
            assert rel_s < 1e-9, (name, rel_s)
            assert np.array_equal(r.columns["usable"][a2], flag_np[b2].astype(int)), (name, "usable flags")
        row = summary.loc[name]
        assert r.loading_class == row["class_mu_t"], (name, r.loading_class, row["class_mu_t"])
        assert r.transmission.T == pytest.approx(row["T_used"], rel=1e-9), name
        assert r.t_eff_mm == pytest.approx(row["t_eff_mm"], rel=1e-9), name
        assert r.q_min_use == pytest.approx(row["q_min_use"], rel=1e-9) and r.q_max_use == pytest.approx(row["q_max_use"], rel=1e-9), name
        assert r.centre_source == row["centre_source"], (name, r.centre_source, row["centre_source"])
        assert r.centre_offset_px[0] == pytest.approx(row["centre_dx_px"], abs=0.06), name     # header rounds to 0.1 px
    assert n_missing <= 2 * len(reduced)
    print(f"worst relative deviation against the full-precision reference: I {worst_I:.2e}, sigma {worst_s:.2e}"
          + ("" if npz is not None else " (npz absent: text comparison only)"))


def test_reduction_summary_matches_reference_columns(reference, reduced):
    from saxs.core import sector_batch as sb
    df = sb.summary_table(reduced.values()).set_index("sample")
    ref = reference["summary"].set_index("sample")
    for name in ref.index:
        for col in ("T_disc4", "T_disc8", "heterogeneity_T4_over_T", "mu_t_eff", "ff_sample", "ff_empty", "n_use"):
            assert df.loc[name, col] == pytest.approx(ref.loc[name, col], rel=1e-6), (name, col)
        for col in ("ratio_plateau_6_12", "ratio_plateau_12_20"):
            a, b = df.loc[name, col], ref.loc[name, col]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-6), (name, col, a, b)
        for col in ("I_per_mm_at_0p02", "I_per_mm_at_0p05", "I_per_mm_at_0p15"):
            a, b = df.loc[name, col], ref.loc[name, col]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-6), (name, col, a, b)
        assert int(df.loc[name, "n_clipped_sample"]) == int(ref.loc[name, "n_clipped_sample"]), name
        assert int(df.loc[name, "n_clipped_empty"]) == int(ref.loc[name, "n_clipped_empty"]), name


# ------------------------------------------------------------------------------------ phase 2
@pytest.fixture(scope="module")
def described(reference, reduced):
    from saxs.core import descriptors as ds
    mf = reference["modelfree"]
    rho = {name: float(mf.loc[name, "rho_e"]) for name in mf.index if np.isfinite(mf.loc[name, "rho_e"])}
    curves = [r.usable_curve() for r in reduced.values()]
    results, df = ds.describe_batch(curves, ds.DescriptorOptions(), rho_e=rho)
    return {r.sample: r for r in results}, df.set_index("sample")


def test_descriptors_within_one_percent_of_reference(reference, described):
    results, df = described
    mf = reference["modelfree"]
    checked = 0
    for name in mf.index:
        ref = mf.loc[name]
        got = df.loc[name]
        assert int(got["n_use"]) == int(ref["n_use"]), name
        for col in ("n_low", "n_at_0.04", "C_hi", "ratio_hi"):
            a, b = float(got[col]), float(ref[col])
            if np.isnan(b):
                assert np.isnan(a), (name, col, a)
                continue
            assert a == pytest.approx(b, rel=0.01), (name, col, a, b)
            checked += 1
        for col in ("n_at_0.006", "n_at_0.01", "n_at_0.02", "K4", "K_low", "n_low_err", "C_hi_mad", "ratio_low",
                    "q_shoulder", "shoulder_prom", "shoulder_size_nm", "q_kratky_max", "Q_exc", "Q_exc_signed",
                    "excess_frac_max", "Q_tot", "I_mm_at_0.01", "I_mm_at_0.02", "I_mm_at_0.05", "K4_over_rhoe2"):
            a, b = float(got[col]), float(ref[col])
            if np.isnan(b):
                assert np.isnan(a), (name, col, a)
            else:
                # the reference computed from the 7-digit .dat text; ours from the full-precision arrays
                assert a == pytest.approx(b, rel=1e-4, abs=1e-30), (name, col, a, b)
        # the explicit maximum test agrees with the rethink: no maximum of I in any quantified curve
        # (opaque loadings with a 1e-6 transmission can rise at the usable-range edge; reported, not asserted)
        if ref["cls"] in ("measurable", "marginal"):
            assert not bool(got["has_maximum"]), (name, results[name].maxima)
    assert checked > 100
    flagged = [n for n in mf.index if bool(df.loc[n, "has_maximum"])]
    print("curves with a maximum of I (opaque only):", flagged)


def test_descriptors_from_the_reference_text_curves_are_exact(reference):
    """The same 7-digit text curves the reference analysed, through PRISM's descriptors: every
    number of phase2_modelfree.csv reproduced to 1e-9 (the port is exact)."""
    from saxs.core import descriptors as ds
    from saxs.core import sector_batch as sb
    mf = reference["modelfree"]
    rho = {name: float(mf.loc[name, "rho_e"]) for name in mf.index if np.isfinite(mf.loc[name, "rho_e"])}
    curves = {r.name: r.usable_curve() for r in sb.load_reduced_folder(RED_DIR, pattern=r".*__v2\.dat$")}
    _results, df = ds.describe_batch([curves[n] for n in mf.index], ds.DescriptorOptions(), rho_e=rho)
    df = df.set_index("sample")
    # table metadata (identity, loading class, electron density, composition columns *_mol) is not a descriptor
    meta = ("sample", "cls", "rho_e", "n_use", "n_low_pts")
    cols = [c for c in mf.columns if c not in meta and not c.endswith("_mol")]
    for name in mf.index:
        for col in cols:
            a, b = float(df.loc[name, col]), float(mf.loc[name, col])
            if np.isnan(b):
                assert np.isnan(a), (name, col, a)
            else:
                assert a == pytest.approx(b, rel=1e-9, abs=1e-300), (name, col, a, b)
        assert int(df.loc[name, "n_use"]) == int(mf.loc[name, "n_use"]) and int(df.loc[name, "n_low_pts"]) == int(mf.loc[name, "n_low_pts"])
