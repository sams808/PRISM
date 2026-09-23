"""Batch sector-aware reduction of beamstopless 2D EDF frames (see saxs.core.edf_sectors).

Usage:
    py -3.11 scripts/reduce_edf_sectors.py pairs.csv OUTDIR [--mu-column mu_mm]

pairs.csv columns: name, sample (path to the sample EDF), empty (path to the empty EDF of the same
session and capillary type), optional mu_mm (linear attenuation coefficient of the material at the
working energy, 1/mm) to get the effective thickness and a per-mm normalisation.
Writes OUTDIR/<name>__sectors.dat per row and OUTDIR/reduction_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from saxs.core import edf_sectors as es  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs_csv")
    ap.add_argument("outdir")
    ap.add_argument("--mu-column", default="mu_mm")
    ap.add_argument("--axis-exclusion-deg", type=float, default=20.0)
    ap.add_argument("--bar-rect", type=float, nargs=2, default=(40.0, 14.0), metavar=("DX_PX", "DY_PX"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    with open(args.pairs_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mu = float(r[args.mu_column]) if r.get(args.mu_column) not in (None, "") else None
            red = es.reduce_frame_pair(r["sample"], r["empty"], mu_mm=mu, axis_exclusion_deg=args.axis_exclusion_deg,
                                       bar_rect_px=tuple(args.bar_rect))
            out = os.path.join(args.outdir, f"{r['name']}__sectors.dat")
            red.write_dat(out)
            tr = red.transmission
            rows.append(dict(name=r["name"], out=out, T=tr.T, T_disc4=tr.T_disc4, heterogeneity=tr.heterogeneity,
                             mu_t_eff=(-1.0 * __import__("math").log(tr.T) if tr.T > 0 else ""), t_eff_mm=red.t_eff_mm or "",
                             q_min_use=red.q_min_use or "", q_max_use=red.q_max_use or ""))
            print(f"{r['name']}: T={tr.T:.3g} (T4/T={tr.heterogeneity:.2f}) usable q=[{red.q_min_use}, {red.q_max_use}] -> {out}")
    with open(os.path.join(args.outdir, "reduction_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
