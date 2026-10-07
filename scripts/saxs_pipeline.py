"""PRISM SAXS command line: reduction of raw 2D frames and physical fitting of reduced curves, driven by a JSON
project file (saxs.core.pipeline) or run on single curves.

Usage (run with the Python that has fabio / xraydb / lmfit, e.g. ``py -3.11``)::

    saxs_pipeline.py init CONFIG.json [--frames-root DIR ...] [--curves-dir DIR] [--output-dir DIR]
                                                         write a template project (raw frames, or reduced curves)
    saxs_pipeline.py run CONFIG.json [--steps inventory reduce describe fit robustness figures tables]
    saxs_pipeline.py fit CONFIG.json [--no-robustness] [--no-figures]   fit -> robustness -> figures -> tables
    saxs_pipeline.py quick CURVE [--reference REF] [--contrast DRHO] [-o OUTDIR] [--iq4] [--families F ...]
                                                         fit one curve (and its reference), write tables and a figure
    saxs_pipeline.py inventory --config CONFIG.json          (the inventory step alone)
    saxs_pipeline.py reduce --config CONFIG.json             (inventory + reduce)
    saxs_pipeline.py inventory ROOT [ROOT ...] -o inventory.csv [--empty-pattern RE] [--calibrant-pattern RE]
                     [--exclude RE] [--name-from stem|comment] [--pairing same_folder|nearest_time] [--no-diagnostics]
    saxs_pipeline.py reduce ROOT [ROOT ...] -o OUTDIR [--samples-table samples.csv] [--energy-ev E]
                     [--exclude RE] [--pairing RULE] [--explicit pairs.csv] [--key-regex RE] [reduction options]
    saxs_pipeline.py sasview CURVES_DIR [-o OUTDIR] [--all-rings] [--i-corr]   three-column SasView files from reduced .dat curves

Exit codes: 0 success, 1 error, 2 finished with unpaired samples.

``samples.csv`` columns: ``sample`` and either ``mu_mm`` or ``composition`` + ``density_g_cm3``
(composition in any syntax the chemistry parser accepts, e.g. ``50.6(SiO2)13.6(B2O3)...``).
The fitting method, the project keys and the outputs are described in docs/SAXS_FITTING.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from saxs.core import frames as fr  # noqa: E402
from saxs.core import sector_batch as sb  # noqa: E402


def _progress(i: int, n: int, msg: str) -> None:
    print(f"  [{i:3d}/{n:3d}] {msg}", flush=True)


def _log(msg: str) -> None:
    print(msg, flush=True)                      # flushed: a long run redirected to a file shows where it is


def _run_config(path: str, steps) -> int:
    from saxs.core import pipeline as pl
    cfg = pl.ProjectConfig.load(path)
    manifest = pl.run(cfg, steps, progress=_progress, log=_log)
    inv = manifest.data.get("steps", {}).get("inventory", {}).get("notes", {})
    print(f"manifest -> {manifest.path}")
    return 2 if inv.get("n_unpaired") else 0


def cmd_init(a: argparse.Namespace) -> int:
    from saxs.core import pipeline as pl
    if os.path.exists(a.config) and not a.force:
        print(f"{a.config} exists (use --force to overwrite)")
        return 1
    cfg = pl.ProjectConfig.template(a.frames_root, a.output_dir, curves_dir=a.curves_dir)
    cfg.save(a.config)
    print(f"template configuration -> {a.config}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    return _run_config(a.config, a.steps or list(__import__("saxs.core.pipeline", fromlist=["STEPS"]).STEPS))


def cmd_fit(a: argparse.Namespace) -> int:
    steps = ["fit"] + ([] if a.no_robustness else ["robustness"]) + ([] if a.no_figures else ["figures"]) + ["tables"]
    return _run_config(a.config, steps)


def cmd_quick(a: argparse.Namespace) -> int:
    from saxs.core.fit_workflow import CurveInput, WorkflowOptions, curve_table, robustness, run_workflow
    from saxs.core.loader import load_tabular_curve
    from saxs.core.series_figures import export_figure, fit_panels_figure
    from saxs.core.tables import transliterate_frame, write_workflow_tables

    out = a.output or os.path.splitext(os.path.abspath(a.curve))[0] + "_fit"
    os.makedirs(out, exist_ok=True)
    curve = load_tabular_curve(a.curve)
    inputs = []
    if a.reference:
        ref = load_tabular_curve(a.reference)
        if ref.name == curve.name:
            ref.name = ref.name + "_reference"
        inputs.append(CurveInput.from_curve(ref))
    inputs.append(CurveInput.from_curve(curve, reference=inputs[0].name if inputs else None, contrast=a.contrast))
    opts = WorkflowOptions(gate="reference" if a.reference else "none", second_population=not a.no_second_population,
                           **({"families": tuple(a.families)} if a.families else {}))
    res = run_workflow(inputs, opts, progress=_progress)
    if not a.no_robustness:
        robustness(res, inputs)
    res.save(os.path.join(out, "fits.json"))
    write_workflow_tables(res, os.path.join(out, "tables"))
    for name, cr in res.curves.items():
        transliterate_frame(curve_table(cr)).to_csv(os.path.join(out, f"{name}_fit.csv"), index=False, encoding="utf-8-sig")
    fig = fit_panels_figure(res, ncol=len(res.curves), iq4=a.iq4, style="Presentation")
    export_figure(fig, os.path.join(out, "fits"), ("pdf", "png"), 300)
    for name, cr in res.curves.items():
        f = cr.retained_fit
        print(f"{name}: gate {cr.gate.method}{' pass' if cr.gate.passed else ''} -> {cr.retained} ({f.redchi:.2f} reduced chi2)")
        for k, v in cr.reasons.items():
            print(f"    {k:14s} {'; '.join(v)}")
    print(f"outputs -> {out}")
    return 0


def _add_scan_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("roots", nargs="*", help="folder(s) of raw frames (scanned recursively); or use --config")
    ap.add_argument("--config", default=None, help="JSON project configuration (then ROOT arguments are ignored)")
    ap.add_argument("--empty-pattern", default=r"(?i)empty")
    ap.add_argument("--calibrant-pattern", default=r"(?i)agbeh")
    ap.add_argument("--exclude", default=None, help="regex on the relative path / stem of frames to skip")
    ap.add_argument("--name-from", choices=("stem", "comment"), default="stem")
    ap.add_argument("--pairing", choices=("same_folder", "nearest_time", "explicit"), default="same_folder")
    ap.add_argument("--explicit", default=None, help="CSV (sample, empty) for --pairing explicit")
    ap.add_argument("--wavelength-A", type=float, default=None, help="wavelength for frames whose header has none (else skipped)")


def cmd_inventory(a: argparse.Namespace) -> int:
    if a.config:
        return _run_config(a.config, ["inventory"])
    if not a.roots:
        print("give frame folders or --config")
        return 1
    recs = fr.scan_frames(a.roots, a.empty_pattern, a.calibrant_pattern, a.exclude, a.name_from,
                          diagnostics=not a.no_diagnostics, progress=_progress,
                          on_error=lambda p, e: print(f"  !! {p}: {type(e).__name__}: {e}"),
                          default_wavelength_A=a.wavelength_A)
    df = fr.write_inventory(recs, a.output)
    print(f"{len(df)} frame(s) -> {a.output}")
    pairing = fr.pair_frames(recs, a.pairing, a.explicit)
    print(pairing.describe())
    pairs_csv = os.path.splitext(a.output)[0] + "_pairs.csv"
    pairing.table().to_csv(pairs_csv, index=False)
    print(f"pairs -> {pairs_csv}")
    return 0 if not pairing.unpaired else 2


def cmd_reduce(a: argparse.Namespace) -> int:
    if a.config:
        return _run_config(a.config, ["inventory", "reduce"])
    if not a.roots or not a.output:
        print("give frame folders and -o OUTDIR, or --config")
        return 1
    recs = fr.scan_frames(a.roots, a.empty_pattern, a.calibrant_pattern, a.exclude, a.name_from, diagnostics=False,
                          on_error=lambda p, e: print(f"  !! {p}: {type(e).__name__}: {e}"),
                          default_wavelength_A=a.wavelength_A)
    pairing = fr.pair_frames(recs, a.pairing, a.explicit)
    print(pairing.describe())
    mu = sb.MuSource.from_samples_table(a.samples_table, key_regex=a.key_regex, energy_ev=a.energy_ev) if a.samples_table else None
    opts = sb.ReductionOptions(axis_exclusion_deg=a.axis_exclusion_deg, bar_rect_px=tuple(a.bar_rect),
                               footprint_r_px=a.footprint_r_px, vertical_half_deg=a.vertical_half_deg,
                               refine_centre=not a.header_centre)
    results = sb.reduce_pairs(pairing.pairs, opts, mu, out_dir=a.output, progress=_progress,
                              on_error=lambda p, e: print(f"  !! {p.sample.name}: {type(e).__name__}: {e}"),
                              sasview_dir=os.path.join(a.output, "sasview"))
    df = sb.summary_table(results)
    summary = os.path.join(a.output, "reduction_summary.csv")
    df.to_csv(summary, index=False)
    for r in results:
        print(f"  {r.name:20s} T={r.transmission.T:.3g} (T4/T={r.heterogeneity:.2f}) mu t={r.mu_t:.2f} "
              f"t_eff={r.t_eff_mm if r.t_eff_mm is None else round(r.t_eff_mm, 3)} mm  {r.loading_class:10s} "
              f"usable q=[{r.q_min_use}, {r.q_max_use}] n={r.n_use}  centre: {r.centre_source}")
    print(f"{len(results)} curve(s) -> {a.output}; summary -> {summary}; SasView-ready copies -> {os.path.join(a.output, 'sasview')}")
    return 0


def cmd_sasview(a: argparse.Namespace) -> int:
    results = sb.load_reduced_folder(a.curves_dir, pattern=r".*\.dat$", exclude_pattern=sb.SASVIEW_SUFFIX)
    if not results:
        print(f"no reduced curve in {a.curves_dir}")
        return 1
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(a.curves_dir.rstrip("/\\"))), "curves_sasview")
    paths = sb.export_sasview(results, out, per_mm=not a.i_corr, usable_only=not a.all_rings)
    for p_ in paths:
        print(f"  {p_}")
    print(f"{len(paths)} SasView-ready file(s) -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="scan frames, write the inventory and the proposed pairs")
    _add_scan_args(inv)
    inv.add_argument("-o", "--output", default="inventory.csv")
    inv.add_argument("--no-diagnostics", action="store_true", help="headers only (fast)")
    inv.set_defaults(func=cmd_inventory)

    red = sub.add_parser("reduce", help="scan, pair and reduce every sample frame")
    _add_scan_args(red)
    red.add_argument("-o", "--output", default=None, help="output folder for the .dat curves and the summary")
    red.add_argument("--samples-table", default=None, help="CSV with sample + mu_mm or composition + density_g_cm3")
    red.add_argument("--key-regex", default=None, help="regex (one group) mapping a frame name to its samples-table key")
    red.add_argument("--energy-ev", type=float, default=None, help="override the frame energy for mu")
    red.add_argument("--axis-exclusion-deg", type=float, default=20.0)
    red.add_argument("--bar-rect", type=float, nargs=2, default=(40.0, 14.0), metavar=("DX_PX", "DY_PX"))
    red.add_argument("--footprint-r-px", type=float, default=16.0)
    red.add_argument("--vertical-half-deg", type=float, default=30.0)
    red.add_argument("--header-centre", action="store_true", help="keep the header beam centre (no refinement)")
    red.set_defaults(func=cmd_reduce)

    sv = sub.add_parser("sasview", help="three-column SasView files from a folder of reduced .dat curves")
    sv.add_argument("curves_dir")
    sv.add_argument("-o", "--output", default=None, help="output folder (default: curves_sasview next to the curves folder)")
    sv.add_argument("--all-rings", action="store_true", help="write every ring, not only the usable range")
    sv.add_argument("--i-corr", action="store_true", help="write I_corr (counts/s/pixel) instead of the per-mm intensity")
    sv.set_defaults(func=cmd_sasview)

    ini = sub.add_parser("init", help="write a template JSON configuration")
    ini.add_argument("config")
    ini.add_argument("--frames-root", nargs="*", default=None, help="raw-frame folders (reduction project)")
    ini.add_argument("--curves-dir", default=None, help="folder of reduced curves (fit-only project)")
    ini.add_argument("--output-dir", default="prism_saxs_output")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="run the project pipeline (all steps, or --steps)")
    run.add_argument("config")
    run.add_argument("--steps", nargs="*", default=None, choices=("inventory", "reduce", "describe", "fit", "robustness", "figures", "tables"))
    run.set_defaults(func=cmd_run)

    fit = sub.add_parser("fit", help="fit -> robustness -> figures -> tables for a project")
    fit.add_argument("config")
    fit.add_argument("--no-robustness", action="store_true")
    fit.add_argument("--no-figures", action="store_true")
    fit.set_defaults(func=cmd_fit)

    qk = sub.add_parser("quick", help="fit one curve (and its reference) without a project file")
    qk.add_argument("curve", help="named-column curve file (q, I, sigma [, dQ ...])")
    qk.add_argument("--reference", default=None, help="the reference curve (same material without the minority component)")
    qk.add_argument("--contrast", type=float, default=None, help="particle contrast in 1e10 cm^-2 (spheres on an absolute scale)")
    qk.add_argument("-o", "--output", default=None, help="output folder (default: <curve>_fit next to the curve)")
    qk.add_argument("--iq4", action="store_true", help="draw I q^4")
    qk.add_argument("--no-robustness", action="store_true")
    qk.add_argument("--families", nargs="*", default=None, choices=("sphere", "sphere_py", "teubner_strey", "debye", "unified"),
                    help="model families to offer (default: all)")
    qk.add_argument("--no-second-population", action="store_true", help="do not offer the second (small) population")
    qk.set_defaults(func=cmd_quick)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return int(a.func(a))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
