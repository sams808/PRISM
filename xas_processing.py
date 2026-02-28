from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import zipfile

import numpy as np
import pandas as pd


@dataclass
class XASData:
    path: str
    df: pd.DataFrame
    energy_col: str
    i0_col: str
    it_col: str
    energy: np.ndarray
    i0: np.ndarray
    it: np.ndarray


@dataclass
class Bundle:
    name: str
    df: pd.DataFrame
    scan_def: dict
    metadata: dict
    path: str
    npz_bytes: Optional[bytes] = None


def _safe_json_load_path(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _safe_json_load_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8", errors="ignore"))


def _group_zip_members_by_dataset(members: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Groups zip members into dataset groups keyed by a dataset root path.

    Returns:
      {group_key: {"csv": <member>, "npz": <member>, "scan_def": <member>, "metadata": <member>}}
    Works for zips where files are at root OR inside subfolders.
    """
    csvs = [m for m in members if re.search(r"_exd\.csv$", m, flags=re.IGNORECASE)]
    if not csvs:
        return {}

    groups: Dict[str, Dict[str, str]] = {}
    for csv in csvs:
        key = str(Path(csv).parent).replace("\\", "/")
        if key == ".":
            key = ""
        groups.setdefault(key, {})
        groups[key]["csv"] = csv

    def find_in_group(key: str, pattern: str) -> Optional[str]:
        prefix = (key.rstrip("/") + "/") if key else ""
        in_same = [m for m in members if m.startswith(prefix) and re.search(pattern, m, re.IGNORECASE)]
        if in_same:
            return in_same[0]
        in_root = [m for m in members if "/" not in m.strip("/") and re.search(pattern, m, re.IGNORECASE)]
        return in_root[0] if in_root else None

    for key in list(groups.keys()):
        groups[key]["npz"] = find_in_group(key, r"_mcas\.npz$")
        groups[key]["scan_def"] = find_in_group(key, r"(?:^|/)scan_def\.json$")
        groups[key]["metadata"] = find_in_group(key, r"(?:^|/)metadata\.json$")

    return {k: v for k, v in groups.items() if "csv" in v}


def read_bundle(path: Union[str, Path]) -> Bundle:
    path = Path(path)
    if path.is_dir():
        csvs = sorted(path.glob("*_exd.csv"))
        if not csvs:
            raise FileNotFoundError(f"No '*_exd.csv' found in {path}")

        npzs = sorted(path.glob("*_mcas.npz"))
        scan_p = path / "scan_def.json"
        meta_p = path / "metadata.json"

        df = pd.read_csv(csvs[0])
        df.attrs["source_csv"] = csvs[0].name

        return Bundle(
            name=path.name,
            df=df,
            scan_def=_safe_json_load_path(scan_p) if scan_p.exists() else {},
            metadata=_safe_json_load_path(meta_p) if meta_p.exists() else {},
            path=str(path),
            npz_bytes=npzs[0].read_bytes() if npzs else None,
        )

    if path.is_file() and path.suffix.lower() == ".zip":
        bundles = read_bundles_from_zip(path)
        if not bundles:
            raise FileNotFoundError(f"No '*_exd.csv' found in zip: {path}")
        if len(bundles) > 1:
            raise ValueError(f"Zip contains multiple datasets; use read_bundles(...) for {path}")
        return bundles[0]

    raise ValueError("Path must be a bundle directory or a .zip bundle.")


def read_bundles_from_zip(zip_path: Union[str, Path]) -> List[Bundle]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        groups = _group_zip_members_by_dataset(members)
        if not groups:
            raise FileNotFoundError(f"No '*_exd.csv' found in zip: {zip_path}")

        bundles: List[Bundle] = []
        for key, files in groups.items():
            csv_m = files["csv"]
            df = pd.read_csv(io.BytesIO(z.read(csv_m)))
            df.attrs["source_csv"] = csv_m

            scan_def = _safe_json_load_bytes(z.read(files["scan_def"])) if files.get("scan_def") else {}
            metadata = _safe_json_load_bytes(z.read(files["metadata"])) if files.get("metadata") else {}
            npz_bytes = z.read(files["npz"]) if files.get("npz") else None

            csv_stem = Path(csv_m).stem
            ds_name = Path(key).name if key else csv_stem
            name = f"{zip_path.stem}__{ds_name}" if len(groups) > 1 else zip_path.stem

            bundles.append(
                Bundle(
                    name=name,
                    df=df,
                    scan_def=scan_def,
                    metadata=metadata,
                    path=str(zip_path),
                    npz_bytes=npz_bytes,
                )
            )
        return bundles


def read_bundles(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Bundle]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: List[Bundle] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.append(read_bundle(p))
        elif p.is_file() and p.suffix.lower() == ".zip":
            out.extend(read_bundles_from_zip(p))
        else:
            raise ValueError(f"Unsupported path: {p}")
    return out


def _find_col(columns: list[str], patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        for col in columns:
            if pattern in col.lower():
                return col
    return None


def _xasdata_from_df(df: pd.DataFrame, path: str) -> XASData:
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    energy_col = _find_col(cols, ["energy", "ev", "angle"])
    i0_col = _find_col(cols, ["i0", "incident"])
    it_col = _find_col(cols, ["it", "trans", "transmitted", "if", "fluor"])

    numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))]
    if energy_col is None and numeric_cols:
        energy_col = numeric_cols[0]
    if i0_col is None and len(numeric_cols) >= 2:
        i0_col = numeric_cols[1]
    if it_col is None and len(numeric_cols) >= 3:
        it_col = numeric_cols[2]

    if not energy_col or not i0_col or not it_col:
        raise ValueError(
            "Could not detect Energy/I0/It columns. Expected columns like Energy, I0, It (or at least 3 numeric columns)."
        )

    energy = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_col], errors="coerce").to_numpy(float)
    it = pd.to_numeric(df[it_col], errors="coerce").to_numpy(float)

    mask = np.isfinite(energy) & np.isfinite(i0) & np.isfinite(it)
    energy = energy[mask]
    i0 = i0[mask]
    it = it[mask]
    order = np.argsort(energy, kind="mergesort")

    return XASData(
        path=path,
        df=df,
        energy_col=energy_col,
        i0_col=i0_col,
        it_col=it_col,
        energy=energy[order],
        i0=i0[order],
        it=it[order],
    )


def parse_xas_file(path: str) -> XASData:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    return _xasdata_from_df(df, str(Path(path)))


def parse_xas_bundle(bundle: Bundle) -> XASData:
    return _xasdata_from_df(bundle.df.copy(), bundle.path)


def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = int(max(1, window))
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def deglitch_robust(energy: np.ndarray, y: np.ndarray, z_thresh: float = 6.0, window: int = 21) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    y = np.asarray(y, dtype=float)
    window = int(max(5, window))
    if window % 2 == 0:
        window += 1
    n = y.size
    if n < window:
        return y.copy()

    half = window // 2
    med = np.empty_like(y)
    mad = np.empty_like(y)
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        seg = y[lo:hi]
        m0 = np.nanmedian(seg)
        med[idx] = m0
        mad[idx] = np.nanmedian(np.abs(seg - m0)) + 1e-12

    robust_z = 0.6745 * (y - med) / mad
    flagged = np.abs(robust_z) > float(z_thresh)
    if flagged.any():
        y_fixed = y.copy()
        good = ~flagged
        y_fixed[flagged] = np.interp(energy[flagged], energy[good], y[good])
        return y_fixed
    return y.copy()


def transmission_to_mu(i0: np.ndarray, it: np.ndarray, log_base: str = "ln") -> np.ndarray:
    i0 = np.asarray(i0, dtype=float)
    it = np.asarray(it, dtype=float)
    eps = 1e-12
    ratio = np.maximum(i0, eps) / np.maximum(it, eps)
    if log_base == "log10":
        return np.log10(ratio)
    return np.log(ratio)


def compute_mu(
    xas: XASData,
    *,
    log_base: str = "ln",
    deglitch: bool = False,
    deglitch_z: float = 6.0,
    deglitch_window: int = 21,
    smooth_window: int = 1,
) -> np.ndarray:
    mu = transmission_to_mu(xas.i0, xas.it, log_base=log_base)
    if deglitch:
        mu = deglitch_robust(xas.energy, mu, z_thresh=deglitch_z, window=deglitch_window)
    if int(smooth_window) > 1:
        mu = moving_average(mu, window=int(smooth_window))
    return mu
