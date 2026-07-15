"""
project_io.py — .dataapp project file save/load (M14), framework-agnostic.

The starkest gap versus every reference tool (Origin .opju, Spectragryph
sessions): until now, closing Dataapp lost every imported spectrum, fit
parameter set, and accepted RRUFF identification. A .dataapp file is a ZIP:

    manifest.json   {"format": "dataapp-project", "version": 1, ...}
    spectra.json    ordered list of Spectrum records (id/title/path/kind/
                    meta/status) — everything except the arrays
    data.npz        x/y arrays for every spectrum, keyed "{id}_x"/"{id}_y"
    fit_params.json {spectrum_id: params_struct} from the shared
                    PerItemSettingsStore (Peak Fitting / Multi-Fit)
    df/{id}.csv     full imported DataFrame per spectrum, when present
                    (needed by e.g. the DTA workspace's column pickers)

Arrays are stored in the project itself rather than re-read from the
original source paths on load — source files move, network shares
disappear; a project must stand alone. The original path is still kept as
provenance metadata.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from qt_models import Spectrum

PROJECT_FORMAT = "dataapp-project"
PROJECT_VERSION = 1


def _json_safe(obj: Any) -> Any:
    """Recursively convert meta dicts to JSON-safe structures (numpy
    scalars/arrays and other exotic values appear in parser metadata)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_project(path: str, spectra: List[Spectrum], fit_params: Dict[str, list]) -> None:
    records = []
    arrays: Dict[str, np.ndarray] = {}
    dfs: Dict[str, pd.DataFrame] = {}

    for sp in spectra:
        records.append({
            "id": sp.id,
            "title": sp.title,
            "path": sp.path,
            "kind": sp.kind,
            "meta": _json_safe(sp.meta),
            "status": sp.status,
        })
        arrays[f"{sp.id}_x"] = np.asarray(sp.x, dtype=float)
        arrays[f"{sp.id}_y"] = np.asarray(sp.y, dtype=float)
        if sp.df is not None:
            dfs[sp.id] = sp.df

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **arrays)

    manifest = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_spectra": len(records),
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("spectra.json", json.dumps(records, indent=2))
        zf.writestr("data.npz", npz_buf.getvalue())
        zf.writestr("fit_params.json", json.dumps(_json_safe(fit_params), indent=2))
        for sid, df in dfs.items():
            zf.writestr(f"df/{sid}.csv", df.to_csv(index=False))


def load_project(path: str) -> Tuple[List[Spectrum], Dict[str, list]]:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != PROJECT_FORMAT:
            raise ValueError(f"Not a Dataapp project file: {path}")
        if int(manifest.get("version", 0)) > PROJECT_VERSION:
            raise ValueError(
                f"Project was saved by a newer Dataapp (project version "
                f"{manifest.get('version')}, this build reads up to {PROJECT_VERSION})."
            )

        records = json.loads(zf.read("spectra.json").decode("utf-8"))
        with np.load(io.BytesIO(zf.read("data.npz"))) as data:
            arrays = {k: data[k] for k in data.files}
        fit_params = json.loads(zf.read("fit_params.json").decode("utf-8"))

        df_names = {n for n in zf.namelist() if n.startswith("df/") and n.endswith(".csv")}

        spectra: List[Spectrum] = []
        for rec in records:
            sid = rec["id"]
            df = None
            df_name = f"df/{sid}.csv"
            if df_name in df_names:
                try:
                    df = pd.read_csv(io.BytesIO(zf.read(df_name)))
                except Exception:
                    df = None
            spectra.append(Spectrum(
                id=sid,
                title=rec.get("title", sid),
                path=rec.get("path", ""),
                kind=rec.get("kind", "generic_xy"),
                x=arrays.get(f"{sid}_x", np.array([])),
                y=arrays.get(f"{sid}_y", np.array([])),
                df=df,
                meta=rec.get("meta", {}) or {},
                status=rec.get("status", "imported"),
            ))

    return spectra, fit_params
