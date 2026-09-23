"""SAXS 2.0 command line: frame inventory, batch sector reduction and the whole project pipeline
driven by a JSON configuration.

Usage (run with the Python that has fabio / xraydb / lmfit, e.g. ``py -3.11``)::

    saxs_pipeline.py init CONFIG.json [--frames-root DIR ...] [--output-dir DIR]     write a template configuration
    saxs_pipeline.py run CONFIG.json [--steps inventory reduce describe fit robustness figures tables]
    saxs_pipeline.py inventory --config CONFIG.json          (the inventory step alone)
    saxs_pipeline.py reduce --config CONFIG.json             (inventory + reduce)
    saxs_pipeline.py inventory ROOT [ROOT ...] -o inventory.csv [--empty-pattern RE] [--calibrant-pattern RE]
                     [--exclude RE] [--name-from stem|comment] [--pairing same_folder|nearest_time] [--no-diagnostics]
    saxs_pipeline.py reduce ROOT [ROOT ...] -o OUTDIR [--samples-table samples.csv] [--energy-ev E]
                     [--exclude RE] [--pairing RULE] [--explicit pairs.csv] [--key-regex RE] [reduction options]

Exit codes: 0 success, 1 error, 2 finished with unpaired samples.

``samples.csv`` columns: ``sample`` and either ``mu_mm`` or ``composition`` + ``density_g_cm3``
(composition in any syntax the chemistry parser accepts, e.g. ``50.6(SiO2)13.6(B2O3)...``).
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


def _run_config(path: str, steps) -> int:
    from saxs.core import pipeline as pl
    cfg = pl.ProjectConfig.load(path)
    manifest = pl.run(cfg, steps, progress=None, log=print)
    inv = manifest.data.get("steps", {}).get("inventory", {}).get("notes", {})
    print(f"manifest -> {manifest.path}")
    return 2 if inv.get("n_unpaired") else 0


def cmd_init(a: argparse.Namespace) -> int:
    from saxs.core import pipeline as pl
    if os.path.exists(a.config) and not a.force:
        print(f"{a.config} exists (use --force to overwrite)")
        return 1
    cfg = pl.ProjectConfig.template(a.frames_root, a.output_dir)
    cfg.save(a.config)
    print(f"template configuration -> {a.config}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    return _run_config(a.config, a.steps or list(__import__("saxs.core.pipeline", fromlist=["STEPS"]).STEPS))


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
                              on_error=lambda p, e: print(f"  !! {p.sample.name}: {type(e).__name__}: {e}"))
    df = sb.summary_table(results)
    summary = os.path.join(a.output, "reduction_summary.csv")
    df.to_csv(summary, index=False)
    for r in results:
        print(f"  {r.name:20s} T={r.transmission.T:.3g} (T4/T={r.heterogeneity:.2f}) mu t={r.mu_t:.2f} "
              f"t_eff={r.t_eff_mm if r.t_eff_mm is None else round(r.t_eff_mm, 3)} mm  {r.loading_class:10s} "
              f"usable q=[{r.q_min_use}, {r.q_max_use}] n={r.n_use}  centre: {r.centre_source}")
    print(f"{len(results)} curve(s) -> {a.output}; summary -> {summary}")
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

    ini = sub.add_parser("init", help="write a template JSON configuration")
    ini.add_argument("config")
    ini.add_argument("--frames-root", nargs="*", default=None)
    ini.add_argument("--output-dir", default="prism_saxs_output")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="run the project pipeline (all steps, or --steps)")
    run.add_argument("config")
    run.add_argument("--steps", nargs="*", default=None, choices=("inventory", "reduce", "describe", "fit", "robustness", "figures", "tables"))
    run.set_defaults(func=cmd_run)
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
