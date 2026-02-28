from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib optional but recommended for helper plotting functions
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


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


def _find_col(columns: list[str], patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        for col in columns:
            if pattern in col.lower():
                return col
    return None


def parse_xas_file(path: str) -> XASData:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    energy_col = _find_col(cols, ["energy", "ev"])
    i0_col = _find_col(cols, ["i0", "incident"])
    it_col = _find_col(cols, ["it", "trans", "transmitted"])

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
        path=str(Path(path)),
        df=df,
        energy_col=energy_col,
        i0_col=i0_col,
        it_col=it_col,
        energy=energy[order],
        i0=i0[order],
        it=it[order],
    )


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


def _require_larch():
    try:
        from larch import Group
        from larch.xafs import find_e0, pre_edge, autobk, xftf
        return Group, find_e0, pre_edge, autobk, xftf
    except Exception as exc:
        raise ImportError(
            "xraylarch is required for Athena-like processing. Install with: pip install xraylarch"
        ) from exc


@dataclass
class AlignConfig:
    enabled: bool = False
    shift_range_ev: float = 5.0
    step_ev: float = 0.02
    windows: Optional[List[Tuple[float, float]]] = None


@dataclass
class PreEdgeConfig:
    pre1: float = -150.0
    pre2: float = -50.0
    norm1: float = 30.0
    norm2: float = 150.0
    nnorm: int = 0
    e0: Optional[float] = None


@dataclass
class AutobkConfig:
    rbkg: float = 1.0
    kmin: float = 0.0
    kmax: Optional[float] = None
    kweight: float = 2.0
    dk: float = 1.0


@dataclass
class FFTConfig:
    kmin: float = 2.0
    kmax: float = 12.0
    dk: float = 1.0
    window: str = "hanning"
    kweight: float = 2.0
    rmax_out: float = 10.0


@dataclass
class ProcessConfig:
    align: AlignConfig = field(default_factory=AlignConfig)
    preedge: PreEdgeConfig = field(default_factory=PreEdgeConfig)
    autobk: AutobkConfig = field(default_factory=AutobkConfig)
    fft: Optional[FFTConfig] = field(default_factory=FFTConfig)


def _interp_to_grid(x_src, y_src, x_new):
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_new = np.asarray(x_new, float)
    m = np.isfinite(x_src) & np.isfinite(y_src)
    x_src, y_src = x_src[m], y_src[m]
    idx = np.argsort(x_src)
    x_src, y_src = x_src[idx], y_src[idx]
    return np.interp(x_new, x_src, y_src)


def estimate_shift_derivative_xcorr(
    e_ref: np.ndarray,
    mu_ref: np.ndarray,
    e_mov: np.ndarray,
    mu_mov: np.ndarray,
    *,
    windows: Optional[List[Tuple[float, float]]] = None,
    shift_range_ev: float = 5.0,
    step_ev: float = 0.02,
    smooth_window: int = 11,
) -> Tuple[float, float]:
    e_ref = np.asarray(e_ref, float)
    mu_ref = np.asarray(mu_ref, float)
    e_mov = np.asarray(e_mov, float)
    mu_mov = np.asarray(mu_mov, float)

    if windows:
        mask = np.zeros_like(e_ref, dtype=bool)
        for lo, hi in windows:
            mask |= (e_ref >= lo) & (e_ref <= hi)
    else:
        mask = np.ones_like(e_ref, dtype=bool)

    mu_mov_i = _interp_to_grid(e_mov, mu_mov, e_ref)

    if smooth_window and smooth_window > 1:
        k = np.ones(int(smooth_window)) / float(smooth_window)
        mu_ref_s = np.convolve(mu_ref, k, mode="same")
        mu_mov_s = np.convolve(mu_mov_i, k, mode="same")
    else:
        mu_ref_s, mu_mov_s = mu_ref, mu_mov_i

    d_ref = np.gradient(mu_ref_s, e_ref)
    r = d_ref[mask]
    r = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)

    shifts = np.arange(-shift_range_ev, shift_range_ev + step_ev, step_ev)
    best_shift, best_score = 0.0, -np.inf

    for s in shifts:
        mu_shift = _interp_to_grid(e_mov, mu_mov, e_ref + s)
        if smooth_window and smooth_window > 1:
            mu_shift = np.convolve(mu_shift, k, mode="same")
        d_mov = np.gradient(mu_shift, e_ref)
        m = d_mov[mask]
        m = (m - np.nanmean(m)) / (np.nanstd(m) + 1e-12)
        score = float(np.nanmean(r * m))
        if score > best_score:
            best_score = score
            best_shift = float(s)

    return best_shift, best_score


def make_group(name: str, energy_ev: np.ndarray, mu: np.ndarray):
    Group, *_ = _require_larch()
    g = Group()
    g.filename = name
    g.label = name
    g.energy = np.asarray(energy_ev, float)
    g.mu = np.asarray(mu, float)
    return g


def run_preedge(g, cfg: PreEdgeConfig):
    _, find_e0, pre_edge, *_ = _require_larch()
    g.e0 = float(find_e0(energy=g.energy, mu=g.mu)) if cfg.e0 is None else float(cfg.e0)
    pre_edge(g, pre1=cfg.pre1, pre2=cfg.pre2, norm1=cfg.norm1, norm2=cfg.norm2, nnorm=cfg.nnorm)
    return g


def run_autobk(g, cfg: AutobkConfig):
    _, _, _, autobk, _ = _require_larch()
    autobk(g, rbkg=cfg.rbkg, kmin=cfg.kmin, kmax=cfg.kmax, kweight=cfg.kweight, dk=cfg.dk)
    return g


def run_fft(g, cfg: FFTConfig):
    _, _, _, _, xftf = _require_larch()
    xftf(g, kmin=cfg.kmin, kmax=cfg.kmax, dk=cfg.dk, window=cfg.window, kweight=cfg.kweight)
    return g


def process_mu_spectrum(
    name: str,
    energy_ev: np.ndarray,
    mu: np.ndarray,
    *,
    cfg: ProcessConfig = ProcessConfig(),
    reference: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    e = np.asarray(energy_ev, float)
    m = np.asarray(mu, float)
    align_report = None
    if cfg.align.enabled and reference is not None:
        eref, muref = reference
        shift, score = estimate_shift_derivative_xcorr(
            eref,
            muref,
            e,
            m,
            windows=cfg.align.windows,
            shift_range_ev=cfg.align.shift_range_ev,
            step_ev=cfg.align.step_ev,
        )
        e = e + shift
        align_report = {"shift_ev": float(shift), "score": float(score)}

    g = make_group(name, e, m)
    run_preedge(g, cfg.preedge)
    run_autobk(g, cfg.autobk)
    if cfg.fft is not None:
        run_fft(g, cfg.fft)

    out: Dict[str, np.ndarray] = {
        "energy": np.asarray(g.energy, float),
        "mu": np.asarray(g.mu, float),
        "e0": np.array([float(getattr(g, "e0", np.nan))]),
        "norm": np.asarray(getattr(g, "norm", np.full_like(g.mu, np.nan)), float),
        "flat": np.asarray(getattr(g, "flat", np.full_like(g.mu, np.nan)), float),
        "bkg": np.asarray(getattr(g, "bkg", np.full_like(g.mu, np.nan)), float),
        "k": np.asarray(getattr(g, "k", np.array([])), float),
        "chi": np.asarray(getattr(g, "chi", np.array([])), float),
    }

    if out["k"].size and out["chi"].size:
        kw = float(cfg.autobk.kweight)
        out[f"chi_k{kw:g}"] = out["chi"] * np.power(out["k"], kw)

    for key in ("r", "chir_mag", "chir_re", "chir_im", "chir_pha"):
        if hasattr(g, key):
            out[key] = np.asarray(getattr(g, key), float)

    if align_report is not None:
        out["align_shift_ev"] = np.array([align_report["shift_ev"]])
        out["align_score"] = np.array([align_report["score"]])

    return out


def process_many(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ProcessConfig = ProcessConfig(),
    reference_name: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    reference = None
    if cfg.align.enabled:
        if reference_name is None:
            raise ValueError("Alignment enabled but reference_name is None.")
        if reference_name not in spectra:
            raise KeyError(f"reference_name '{reference_name}' not found.")
        reference = spectra[reference_name]

    out = {}
    for name, (e, mu) in spectra.items():
        out[name] = process_mu_spectrum(name, e, mu, cfg=cfg, reference=reference)
    return out


def export_athena_columns(
    out: Dict[str, np.ndarray],
    filepath: Union[str, Path],
    *,
    comment: Optional[str] = None,
):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    energy = out["energy"]
    mu = out["mu"]
    cols = [energy, mu]
    headers = ["energy", "mu"]
    for key in ("norm", "flat", "bkg"):
        if key in out and out[key].size == energy.size:
            cols.append(out[key])
            headers.append(key)

    lines = ["# Athena column file exported from Python/Larch"]
    if "e0" in out:
        lines.append(f"# e0 = {float(out['e0'][0]):.6f}")
    if comment:
        lines.append(f"# {comment}")
    lines.append("# " + "  ".join(headers))

    np.savetxt(filepath, np.column_stack(cols), header="\n".join(lines), comments="")
    return filepath


def json_dumps_pretty(obj) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def export_athena_project_best_effort(
    processed: Dict[str, Dict[str, np.ndarray]],
    outdir: Union[str, Path],
    *,
    project_name: str = "export",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exported_files = {}
    for name, out in processed.items():
        fn = outdir / f"{name}.dat"
        export_athena_columns(out, fn)
        exported_files[name] = str(fn)

    prj_path = None
    writer_used = None
    try:
        write_athena = None
        try:
            from larch.io import write_athena  # type: ignore
        except Exception:
            write_athena = None
        if write_athena is None:
            try:
                from larch.io.athena import write_athena  # type: ignore
            except Exception:
                write_athena = None

        if write_athena is not None:
            from larch import Group

            groups = []
            for name, out in processed.items():
                g = Group()
                g.filename = name
                g.label = name
                g.energy = out["energy"]
                g.mu = out["mu"]
                for key in ("norm", "flat", "k", "chi", "r", "chir_mag", "chir_re", "chir_im", "chir_pha"):
                    if key in out and out[key].size > 0:
                        setattr(g, key, out[key])
                groups.append(g)

            prj_path = outdir / f"{project_name}.prj"
            write_athena(str(prj_path), groups)
            writer_used = "larch.write_athena"
    except Exception:
        prj_path = None
        writer_used = None

    manifest = {
        "project_name": project_name,
        "athena_prj": str(prj_path) if prj_path else None,
        "writer_used": writer_used,
        "files": exported_files,
        "import_hint": "In Athena: File -> Import -> Column Data (select the .dat files).",
    }
    manifest_path = outdir / f"{project_name}_manifest.json"
    manifest_path.write_text(json_dumps_pretty(manifest), encoding="utf-8")
    return {
        "outdir": str(outdir),
        "athena_prj": str(prj_path) if prj_path else None,
        "manifest": str(manifest_path),
        "column_files": exported_files,
        "writer_used": writer_used,
    }


def plot_mu_norm(out: Dict[str, np.ndarray], *, title: str = "", xlim=None):
    if plt is None:
        raise RuntimeError("matplotlib not available")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["energy"], out["mu"], label="mu(E)")
    if "norm" in out:
        ax.plot(out["energy"], out["norm"], label="norm")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("mu / norm")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_chik(out: Dict[str, np.ndarray], *, kweight: float = 2.0, title: str = "", xlim=None):
    if plt is None:
        raise RuntimeError("matplotlib not available")
    k = out.get("k", np.array([]))
    chi = out.get("chi", np.array([]))
    if k.size == 0 or chi.size == 0:
        raise ValueError("No chi(k) available in output (autobk did not run or failed).")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, chi * np.power(k, kweight), label=f"chi(k)*k^{kweight:g}")
    ax.set_xlabel("k (1/Å)")
    ax.set_ylabel(f"chi*k^{kweight:g}")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_ft(out: Dict[str, np.ndarray], *, title: str = "", xlim=None):
    if plt is None:
        raise RuntimeError("matplotlib not available")
    if "r" not in out or "chir_mag" not in out:
        raise ValueError("No FT data found (xftf outputs missing).")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["r"], out["chir_mag"], label="|chi(R)|")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("|chi(R)|")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def open_interactive_plot_energy_mu(energy: np.ndarray, mu: np.ndarray, *, title: str = ""):
    try:
        from wxmplot import plot
    except Exception as exc:
        raise ImportError("wxmplot not installed. Install with: pip install wxmplot") from exc
    plot(energy, mu, xlabel="Energy (eV)", ylabel="mu(E)", title=title)
