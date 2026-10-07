"""Publication figures for a series of reduced SAXS curves, driven by a caller-supplied series table.

Nothing here knows a sample: the grid layout, stacking order, labels, colours, markers, line styles,
family labels, column titles, x-values and "quantified" flags all come from a :class:`SeriesTable`
(a CSV or a DataFrame). The figures reproduce the style of the September 2026 manuscript package:

* :func:`grid_figure`: one panel per (panel_row, panel_col), curves normalised and shifted by
  ``shift_decades`` per stack index, labels with a short leader line at the curve ends, panel
  letters top-left, family label bottom-left, column titles on the top row;
* :func:`fit_panels_figure`: one main + residual pair per curve of a fit-workflow result (nested gridspec,
  3:1), data as open markers, the retained model and its labelled components, normalised residuals, I or I q^4;
* :func:`trends_figure`: descriptors or fit results against the table's x-value; open markers for
  non-quantified samples, faded markers for a parameter that is not robust, error bars when robust;
* :func:`frame_figure`: two 2D frames cropped about the refined beam centre plus their horizontal /
  vertical sector profiles (the instrumental-feature figure);
* :func:`export_figure`: pdf / svg / png at a given dpi.

Matplotlib only (Figure objects, no pyplot state); nothing Qt.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .curve import Curve

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_COLOURS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
COMPONENT_STYLE = {"grain": dict(color="0.45", ls="--", lw=0.9), "flat": dict(color="0.6", ls=":", lw=0.9),
                   "oz": dict(color="0.6", ls=":", lw=0.9), "sph": dict(color="#0072B2", ls="-.", lw=0.9),
                   "ts": dict(color="#D55E00", ls="-.", lw=0.9), "dab": dict(color="#009E73", ls="-.", lw=0.9),
                   "uni": dict(color="#CC79A7", ls="-.", lw=0.9), "uni2": dict(color="#56B4E9", ls=(0, (1, 1)), lw=0.9)}
SERIES_RC_COMMON = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "pdf.fonttype": 42,
                    "svg.fonttype": "none", "xtick.direction": "in", "ytick.direction": "in"}
SERIES_RC_PUBLICATION = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8, "xtick.top": False, "ytick.right": False}


# ------------------------------------------------------------------------------------ series table
class SeriesTable:
    """The caller's description of a sample series (one row per sample)."""
    REQUIRED = ("sample", "panel_row", "panel_col", "stack_index", "label")
    OPTIONAL = ("colour", "marker", "linestyle", "x_value", "family_label", "col_title", "quantified", "note_symbol",
                "group", "fit_letter", "extra_panels", "fit_order")
    # extra_panels: "row:col;row:col" where the same curve is drawn again (a parent shared by two families)
    # fit_order: number ordering the samples of a group in the per-group fit figures (else table order)

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.validate()

    @classmethod
    def from_csv(cls, path: str) -> "SeriesTable":
        return cls(pd.read_csv(path, encoding="utf-8-sig"))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SeriesTable":
        return cls(df)

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"series table lacks the columns {missing}")
        self.df["sample"] = self.df["sample"].astype(str)
        if self.df["sample"].duplicated().any():
            dup = list(self.df["sample"][self.df["sample"].duplicated()])
            raise ValueError(f"duplicate samples in the series table: {dup}")
        for c in ("panel_row", "panel_col", "stack_index"):
            try:
                self.df[c] = self.df[c].astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"column {c!r} must be numeric") from exc
            if (self.df[c] < 0).any() or self.df[c].isna().any():
                raise ValueError(f"column {c!r} must hold non-negative numbers")
        for c in ("panel_row", "panel_col"):
            self.df[c] = self.df[c].astype(int)
        if "quantified" in self.df.columns:
            self.df["quantified"] = self.df["quantified"].map(_to_bool)
        else:
            self.df["quantified"] = True
        for c in self.OPTIONAL:
            if c not in self.df.columns:
                self.df[c] = np.nan if c == "x_value" else ""
        self.df["x_value"] = pd.to_numeric(self.df["x_value"], errors="coerce")
        self.df["fit_order"] = pd.to_numeric(self.df["fit_order"], errors="coerce")
        self.df = self.df.set_index("sample", drop=False)

    # -- access ----------------------------------------------------------------------
    @property
    def samples(self) -> List[str]:
        return list(self.df["sample"])

    def has(self, sample: str) -> bool:
        return sample in self.df.index

    def row(self, sample: str) -> pd.Series:
        return self.df.loc[sample]

    def get(self, sample: str, column: str, default=None):
        if sample not in self.df.index or column not in self.df.columns:
            return default
        v = self.df.loc[sample, column]
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return default
        return v

    def extra_panels(self, sample: str) -> List[Tuple[int, int]]:
        text = self.get(sample, "extra_panels")
        out = []
        for item in str(text or "").replace(",", ";").split(";"):
            item = item.strip()
            if not item:
                continue
            r, c = item.split(":")
            out.append((int(float(r)), int(float(c))))
        return out

    def shape(self) -> Tuple[int, int]:
        rows = [int(self.df["panel_row"].max())] + [r for s in self.samples for r, _ in self.extra_panels(s)]
        cols = [int(self.df["panel_col"].max())] + [c for s in self.samples for _, c in self.extra_panels(s)]
        return max(rows) + 1, max(cols) + 1

    def panels(self) -> List[Tuple[int, int]]:
        own = {(int(r), int(c)) for r, c in zip(self.df["panel_row"], self.df["panel_col"])}
        own |= {p for s in self.samples for p in self.extra_panels(s)}
        return sorted(own)

    def samples_in_panel(self, row: int, col: int) -> List[str]:
        sub = self.df[(self.df["panel_row"] == row) & (self.df["panel_col"] == col)]
        names = list(sub["sample"]) + [s for s in self.samples if (row, col) in self.extra_panels(s) and s not in set(sub["sample"])]
        return sorted(names, key=lambda n: (float(self.get(n, "stack_index", 0)), self.samples.index(n)))

    def groups(self) -> List[str]:
        return [g for g in pd.unique(self.df["group"]) if g not in ("", None) and not (isinstance(g, float) and np.isnan(g))]

    def samples_in_group(self, group: str) -> List[str]:
        sub = self.df[self.df["group"] == group]
        if sub["fit_order"].notna().any():
            sub = sub.assign(_o=sub["fit_order"].fillna(1e9)).sort_values(["_o", "panel_row", "panel_col", "stack_index"])
        else:
            sub = sub.sort_values(["panel_row", "panel_col", "stack_index"])
        return list(sub["sample"])

    def colour(self, sample: str) -> str:
        c = self.get(sample, "colour")
        if c:
            return str(c)
        panels = self.panels()
        key = (int(self.get(sample, "panel_row", 0)), int(self.get(sample, "panel_col", 0)))
        return DEFAULT_COLOURS[panels.index(key) % len(DEFAULT_COLOURS)] if key in panels else DEFAULT_COLOURS[0]

    def marker(self, sample: str) -> str:
        return str(self.get(sample, "marker") or "o")

    def linestyle(self, sample: str) -> str:
        return str(self.get(sample, "linestyle") or "-")

    def label(self, sample: str) -> str:
        return str(self.get(sample, "label") or sample) + str(self.get(sample, "note_symbol") or "")

    def quantified(self, sample: str) -> bool:
        return bool(self.get(sample, "quantified", True))


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    return bool(v)


# ------------------------------------------------------------------------------------ style
@contextmanager
def apply_style(style: str = "Publication", overrides: Optional[Dict[str, object]] = None, dpi: int = 600):
    """rc context: the Figures module preset, the series-figure conventions, then ``overrides``."""
    import matplotlib

    try:
        from figures.figures_science import STYLE_PRESETS
        base = dict(STYLE_PRESETS.get(style, STYLE_PRESETS["Publication"]))
    except Exception:  # pragma: no cover - the Figures module is part of PRISM, but stay usable alone
        base = {}
    rc = {**base, **SERIES_RC_COMMON, **(SERIES_RC_PUBLICATION if style == "Publication" else {}), "savefig.dpi": dpi}
    rc.update(overrides or {})
    with matplotlib.rc_context(rc):
        yield


def light_grid(ax) -> None:
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.grid(True, which="minor", color="0.94", lw=0.3)
    ax.set_axisbelow(True)


def _curve_arrays(c: Union[Curve, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if isinstance(c, Curve):
        return np.asarray(c.q, float), np.asarray(c.intensity, float), (None if c.sigma is None else np.asarray(c.sigma, float))
    q, I = c[0], c[1]
    return np.asarray(q, float), np.asarray(I, float), None


def rebinned_curve(c: Union[Curve, Tuple[np.ndarray, np.ndarray]], per_decade: int = 60, q_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    from .descriptors import rebin_log

    q, I, sig = _curve_arrays(c)
    m = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
    if q_max is not None:
        m &= q <= q_max
    if isinstance(c, Curve):
        cols = (c.metadata or {}).get("columns") or {}
        if "usable" in cols and len(cols["usable"]) == len(q):
            m &= np.asarray(cols["usable"], float).astype(bool)
    q, I = q[m], I[m]
    if len(q) == 0:
        return q, I
    qb, Ib, _ = rebin_log(q, I, None if sig is None else sig[m], per_decade)
    ok = np.isfinite(Ib) & (Ib > 0)
    return qb[ok], Ib[ok]


# ------------------------------------------------------------------------------------ grid figure
def grid_figure(curves: Dict[str, Union[Curve, Tuple[np.ndarray, np.ndarray]]], table: SeriesTable, shift_decades: float = 1.6,
                normalise: bool = True, x_range: Tuple[float, float] = (4.0e-3, 0.62), q_max: float = 0.22, per_decade: int = 60,
                letters: bool = True, style: str = "Publication", figsize: Optional[Tuple[float, float]] = None,
                xlabel: str = "Q (Å$^{-1}$)", ylabel: str = "Intensity (normalized, shifted)", label_fontsize: float = 6.5,
                overrides: Optional[Dict[str, object]] = None):
    """One panel per (panel_row, panel_col) of the table; samples stacked by ``stack_index`` and
    shifted by ``10 ** (stack_index * shift_decades)``; labels at the curve ends. Samples of the
    table without a curve are skipped; curves not in the table are ignored."""
    from matplotlib.figure import Figure

    nrow, ncol = table.shape()
    with apply_style(style, overrides):
        fig = Figure(figsize=figsize or (2.6 * ncol, 3.5 * nrow))
        axes = fig.subplots(nrow, ncol, sharex=True, squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.12))
        k = 0
        for r_ in range(nrow):
            for c_ in range(ncol):
                ax = axes[r_, c_]
                drawn = []
                family, title = None, None
                for name in table.samples_in_panel(r_, c_):
                    family = family or table.get(name, "family_label")
                    title = title or table.get(name, "col_title")
                    if name not in curves:
                        continue
                    q, I = rebinned_curve(curves[name], per_decade, q_max)
                    if len(q) == 0:
                        continue
                    idx = float(table.get(name, "stack_index", 0))
                    y = (I / np.nanmax(I) if normalise else I) * 10 ** (idx * shift_decades)
                    col = table.colour(name)
                    ax.plot(q, y, color=col, lw=1.0, ls=table.linestyle(name))
                    x1, y1 = q[-1], float(np.median(y[-5:]))
                    ax.plot([x1 * 1.03, x1 * 1.12], [y1, y1], color=col, lw=0.7)
                    ax.text(x1 * 1.16, y1, table.label(name), fontsize=label_fontsize, ha="left", va="center")
                    drawn.append(y)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(*x_range)
                if drawn:
                    ax.set_ylim(min(np.nanmin(v) for v in drawn) / 3, max(np.nanmax(v) for v in drawn) * 6)
                ax.tick_params(labelleft=False, left=False, which="both")
                light_grid(ax)
                if letters:
                    ax.text(0.04, 0.96, LETTERS[k % 26], transform=ax.transAxes, va="top", ha="left", fontsize=9)
                if family:
                    ax.text(0.04, 0.03, str(family), transform=ax.transAxes, va="bottom", ha="left", fontsize=7)
                if r_ == 0 and title:
                    ax.set_title(str(title))
                if r_ == nrow - 1:
                    ax.set_xlabel(xlabel)
                if c_ == 0:
                    ax.set_ylabel(ylabel)
                k += 1
    return fig


# ------------------------------------------------------------------------------------ fit panels
def component_label(f, key: str) -> str:
    """Legend text of one component of a PhysFitResult, with its key parameters (* = at a bound)."""
    p = f.params
    star = lambda k: "*" if (k in p and p[k].vary and p[k].at_bound) else ""      # noqa: E731
    v = lambda k: p[k].value                                                    # noqa: E731
    if key == "grain":
        return f"grain law, p {v('grain_p'):.2f}{star('grain_p')}"
    if key == "flat":
        return f"flat level {v('flat_C'):.3g}"
    if key == "oz":
        return f"OZ, $\\xi$ {v('oz_xiL'):.1f} Å{star('oz_xiL')}"
    if key == "sph":
        return f"spheres D {2 * v('sph_R') / 10:.1f} nm{star('sph_R')}, $\\sigma$ {v('sph_sig'):.2f}{star('sph_sig')}"
    if key == "ts":
        return f"TS d {v('ts_d') / 10:.1f} nm{star('ts_d')}, $\\xi$ {v('ts_xi') / 10:.1f} nm{star('ts_xi')}"
    if key == "dab":
        return f"Debye $\\xi$ {v('dab_xi') / 10:.2f} nm{star('dab_xi')}"
    if key in ("uni", "uni2"):
        return f"{'unified' if key == 'uni' else 'small pop.'} R$_g$ {v(key + '_Rg') / 10:.1f} nm{star(key + '_Rg')}, d {v(key + '_d'):.1f}{star(key + '_d')}"
    return key


def draw_fit_panel(ax, axr, cr, table: Optional[SeriesTable] = None, letter: Optional[str] = None,
                   x_range: Optional[Tuple[float, float]] = None, iq4: bool = False, model_key: Optional[str] = None,
                   resid_ylim: float = 5.0, legend: bool = True) -> None:
    """Main panel (data, model, components) and residual panel (normalised, sigma units) of one
    :class:`saxs.core.fit_workflow.CurveResult` (the retained model unless ``model_key`` is given)."""
    f = cr.fits[model_key or cr.retained]
    name = cr.name
    q, I = f.q, f.I
    w = q ** 4 if iq4 else np.ones_like(q)
    col = table.colour(name) if (table is not None and table.has(name)) else "#1f77b4"
    marker = table.marker(name) if (table is not None and table.has(name)) else "o"
    ax.plot(q, I * w, ls="none", marker=marker, ms=2.6, mfc="none", mec=col, mew=0.6, label="data")
    key = model_key or cr.retained
    ax.plot(q, f.model * w, color="k", lw=1.1, label=f"{key}, $\\chi^2_r$ {f.redchi:.2f}")
    for ck, cv in f.components.items():
        ax.plot(q, cv * w, label=component_label(f, ck), **COMPONENT_STYLE.get(ck, dict(color="0.5", ls=":", lw=0.8)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*(x_range or (q.min() / 1.1, q.max() * 1.1)))
    yy = I * w
    ax.set_ylim(np.nanmin(yy[yy > 0]) / 5, np.nanmax(yy) * 3)
    light_grid(ax)
    ax.tick_params(labelbottom=False)
    ax.set_title(name, fontsize=8)
    if letter:
        ax.text(0.04, 0.05, letter, transform=ax.transAxes, fontsize=9, va="bottom")
    if legend:
        ax.legend(loc="upper right" if not iq4 else "upper left", frameon=False, fontsize=5.5, handlelength=1.8)
    axr.axhspan(-1, 1, color="0.92", lw=0)
    axr.plot(q, f.residual, ls="none", marker=".", ms=3, color=col)
    axr.axhline(0, color="k", lw=0.5)
    axr.set_xscale("log")
    axr.set_xlim(*(x_range or (q.min() / 1.1, q.max() * 1.1)))
    axr.set_ylim(-resid_ylim, resid_ylim)
    light_grid(axr)


def fit_panels_figure(results, samples: Optional[Sequence[str]] = None, table: Optional[SeriesTable] = None, ncol: int = 3,
                      letters: Union[bool, str] = True, title: Optional[str] = None, x_range: Optional[Tuple[float, float]] = None,
                      iq4: bool = False, panel_h: float = 3.6, panel_w: float = 4.33, style: str = "Publication",
                      xlabel: str = "Q (Å$^{-1}$)", ylabel: Optional[str] = None, resid_ylabel: str = "residual ($\\sigma$)",
                      overrides: Optional[Dict[str, object]] = None):
    """Main + residual pairs (nested gridspec 3:1) for ``samples`` (default: every curve) of a fit-workflow result
    (a WorkflowResult or a ``{name: CurveResult}`` mapping), ``ncol`` per row."""
    from matplotlib.figure import Figure

    curves = getattr(results, "curves", results)
    names = [s for s in (samples if samples is not None else list(curves)) if s in curves]
    if not names:
        raise ValueError("no fitted curve to draw")
    ylabel = ylabel or ("I q$^4$" if iq4 else "Intensity")
    nrow = int(np.ceil(len(names) / ncol))
    with apply_style(style, overrides):
        fig = Figure(figsize=(panel_w * ncol, panel_h * nrow + (0.4 if title else 0)))
        gs = fig.add_gridspec(nrow, ncol, hspace=0.32, wspace=0.28, top=0.95 if title else 0.97)
        for i, s in enumerate(names):
            rr, cc = divmod(i, ncol)
            sub = gs[rr, cc].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            letter = None
            if letters:
                letter = (table.get(s, "fit_letter") if table is not None else None) or (letters[i] if isinstance(letters, str) and i < len(letters) else LETTERS[i % 26])
            draw_fit_panel(ax, axr, curves[s], table, letter=letter, x_range=x_range, iq4=iq4)
            if rr == nrow - 1 or i + ncol >= len(names):
                axr.set_xlabel(xlabel)
            if cc == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
                axr.set_ylabel(resid_ylabel, fontsize=7)
        if title:
            fig.suptitle(title, y=0.995)
    return fig


# ------------------------------------------------------------------------------------ trends
@dataclass
class TrendPanel:
    key: str
    source: str                       # "descriptors" | "fits"
    column: str
    ylabel: str
    yscale: str = "linear"
    ylim: Optional[Tuple[float, float]] = None
    transform: Optional[Union[str, Callable[[float], float]]] = None      # "neg" | "pow10" | "div10" | callable
    err_column: Optional[str] = None
    only_quantified: bool = False
    fade_column: Optional[str] = None  # a boolean column of the source table: False -> faded marker, no error bar
    require: Optional[Tuple[str, str, float]] = None                     # (column, ">"|"<"|">="|"<=", value)
    hlines: Tuple[float, ...] = ()

    def apply(self, v: float) -> float:
        if self.transform is None:
            return v
        if callable(self.transform):
            return self.transform(v)
        return {"neg": -v, "pow10": 10 ** v, "div10": v / 10}[self.transform]


def default_trend_panels() -> List[TrendPanel]:
    """Grain exponent, high-q level, Teubner-Strey spacing, small-population Rg (columns of the workflow summary)."""
    return [
        TrendPanel("grain", "fits", "grain_p", "grain power-law exponent", err_column="grain_p_err", hlines=(4.0,)),
        TrendPanel("highq", "fits", "highq_at_report_q", "high-q term (I units)", yscale="log"),
        TrendPanel("ts_d", "fits", "ts_d_nm", "Teubner-Strey d (nm)", err_column="ts_d_err_nm", fade_column="robust_primary"),
        TrendPanel("uni2", "fits", "uni2_Rg_nm", "small population R$_g$ (nm)", err_column="uni2_Rg_err_nm", fade_column="robust_second"),
    ]


def _ok(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def trends_figure(table: SeriesTable, descriptors_df: Optional[pd.DataFrame], fits_df: Optional[pd.DataFrame],
                  panels: Optional[Sequence[TrendPanel]] = None, x_label: str = "x", style: str = "Publication",
                  figsize: Optional[Tuple[float, float]] = None, legend: bool = True, letters: bool = True,
                  overrides: Optional[Dict[str, object]] = None):
    """Descriptors / fit results against the table's ``x_value``: filled marker = quantified, open =
    not quantified, faded = the panel's ``fade_column`` is False (e.g. not robust), error bars
    (``err_column``) only when not faded. The fits table is the workflow summary (column ``curve``)."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    panels = list(panels) if panels is not None else default_trend_panels()
    sources = {"descriptors": _indexed(descriptors_df), "fits": _indexed(fits_df)}
    with apply_style(style, overrides):
        fig = Figure(figsize=figsize or (3.25 * len(panels), 3.2))
        axes = fig.subplots(1, len(panels), squeeze=False, gridspec_kw=dict(wspace=0.38))[0]
        for ax, spec in zip(axes, panels):
            df = sources[spec.source]
            for s in table.samples:
                x = table.get(s, "x_value")
                if df is None or s not in df.index or not _ok(x):
                    continue
                row = df.loc[s]
                if spec.column not in row or not _ok(row[spec.column]):
                    continue
                if spec.require is not None:
                    col, op, val = spec.require
                    if col not in row or not _ok(row[col]) or not {">": row[col] > val, "<": row[col] < val, ">=": row[col] >= val, "<=": row[col] <= val}[op]:
                        continue
                quantified = table.quantified(s)
                if spec.only_quantified and not quantified:
                    continue
                robust = None
                if spec.fade_column and spec.fade_column in row and row[spec.fade_column] is not None and not (
                        isinstance(row[spec.fade_column], float) and np.isnan(row[spec.fade_column])):
                    robust = bool(row[spec.fade_column])
                y = spec.apply(float(row[spec.column]))
                col = table.colour(s)
                kw = dict(marker=table.marker(s), ms=5, ls="none", color=col, mfc=col if quantified else "none", mew=0.9,
                          alpha=(0.35 if robust is False else 1.0))
                err = spec.apply(float(row[spec.err_column])) if (spec.err_column and spec.err_column in row and _ok(row[spec.err_column])) else None
                if err is not None and robust is not False:
                    ax.errorbar(float(x), y, yerr=abs(err), capsize=0, lw=0.7, **kw)
                else:
                    ax.plot(float(x), y, **kw)
            for h in spec.hlines:
                ax.axhline(h, color="0.5", lw=0.6, ls="--")
            ax.set_yscale(spec.yscale)
            if spec.ylim:
                ax.set_ylim(*spec.ylim)
            ax.set_ylabel(spec.ylabel)
            ax.set_xlabel(x_label)
            light_grid(ax)
        if letters:
            for ax, L in zip(axes, LETTERS):
                ax.text(0.04, 0.96, L, transform=ax.transAxes, fontsize=9, va="top")
        if legend and len(axes):
            handles = []
            seen = set()
            for s in table.samples:
                g = table.get(s, "group") or table.get(s, "col_title")
                key = (table.colour(s), g)
                if g and key not in seen:
                    seen.add(key)
                    handles.append(Line2D([], [], color=table.colour(s), marker="o", ls="none", ms=5, label=str(g)))
            seen_m = set()
            for s in table.samples:
                fam = table.get(s, "family_label")
                key = (table.marker(s), fam)
                if fam and key not in seen_m:
                    seen_m.add(key)
                    handles.append(Line2D([], [], color="0.3", marker=table.marker(s), ls="none", ms=5, label=str(fam)))
            handles.append(Line2D([], [], color="0.3", marker="o", ls="none", ms=5, mfc="none", label="not quantified"))
            if any(p.fade_column for p in panels):
                handles.append(Line2D([], [], color="0.3", marker="o", ls="none", ms=5, alpha=0.35, label="not robust"))
            axes[0].legend(handles=handles, loc="lower left", frameon=False, ncol=2, fontsize=6)
    return fig


def _indexed(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    for key in ("sample", "curve"):
        if key in df.columns:
            return df.set_index(key, drop=False)
    return df


# ------------------------------------------------------------------------------------ 2D frames
def frame_figure(sample_frame: str, empty_frame: str, crop_px: int = 45, q_marker: Optional[float] = None,
                 titles: Tuple[str, str] = ("sample", "empty"), style: str = "Publication", log_range: Tuple[float, float] = (-3, 5),
                 half_deg: float = 25.0, profile_x_range: Tuple[float, float] = (1e-3, 0.1), overrides: Optional[Dict[str, object]] = None,
                 default_wavelength_A: Optional[float] = None):
    """Two frames cropped about the direct beam (centre refined from the empty), log10 counts/s, plus
    their horizontal / vertical sector profiles; a dashed circle / line at ``q_marker`` (1/A)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    from . import edf_sectors as es

    s_data, _hs, gs = es.read_frame(sample_frame, default_wavelength_A)
    e_data, _he, ge = es.read_frame(empty_frame, default_wavelength_A)
    c = es.beam_centroid(e_data, ge.cx, ge.cy) or (ge.cx, ge.cy)
    cx, cy = float(c[0]), float(c[1])
    pix_mm = ge.pixel_m * 1e3
    with apply_style(style, overrides):
        fig = Figure(figsize=(9.5, 3.0))
        axes = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.35))
        x0, y0, w = int(round(cx)), int(round(cy)), int(crop_px)
        for k, (data, geom, ttl) in enumerate(((s_data, gs, titles[0]), (e_data, ge, titles[1]))):
            crop = data[max(y0 - w, 0):y0 + w, max(x0 - w, 0):x0 + w] / geom.exposure_s
            ax = axes[k]
            ax.imshow(np.where(crop > 0, np.log10(np.maximum(crop, 10 ** log_range[0] * 0.1)), np.nan), origin="lower", cmap="viridis",
                      vmin=log_range[0], vmax=log_range[1], extent=np.array([-w, w, -w, w]) * pix_mm)
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)" if k == 0 else "")
            ax.text(0.04, 0.96, LETTERS[k], transform=ax.transAxes, fontsize=9, va="top", bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.5))
            if q_marker and k == 0:
                r_px = float(np.interp(q_marker, geom.q_of_radius(np.arange(0, 4000)), np.arange(0, 4000)))
                ax.add_patch(Circle((0, 0), r_px * pix_mm, fill=False, ec="w", lw=0.7, ls="--"))
        ax = axes[2]
        for data, geom, lab, ls in ((s_data, gs, titles[0], "-"), (e_data, ge, titles[1], ":")):
            g = es.FrameGeometry(cx=cx, cy=cy, dist_m=geom.dist_m, pixel_m=geom.pixel_m, wavelength_A=geom.wavelength_A, exposure_s=geom.exposure_s)
            an = es.sector_anisotropy(data, g, half_deg=half_deg)
            ax.plot(an.q[2:], an.horizontal[2:], ls, color="#d62728", lw=1 if ls == "-" else 0.9, label=f"{lab}, horizontal ±{half_deg:.0f}°")
            ax.plot(an.q[2:], an.vertical[2:], ls, color="#1f77b4", lw=1 if ls == "-" else 0.9, label=f"{lab}, vertical ±{half_deg:.0f}°")
        if q_marker:
            ax.axvline(q_marker, color="k", ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*profile_x_range)
        ax.set_xlabel("Q (Å$^{-1}$)")
        ax.set_ylabel("counts s$^{-1}$ pixel$^{-1}$")
        ax.legend(frameon=False, fontsize=6, loc="upper right")
        light_grid(ax)
        ax.text(0.04, 0.96, "C", transform=ax.transAxes, fontsize=9, va="top")
    return fig


# ------------------------------------------------------------------------------------ export
def export_figure(fig, base: str, formats: Sequence[str] = ("pdf", "svg", "png"), dpi: int = 600) -> List[str]:
    """Save ``fig`` as ``base.<ext>`` for every format; returns the paths."""
    out = []
    d = os.path.dirname(base)
    if d:
        os.makedirs(d, exist_ok=True)
    for ext in formats:
        path = f"{base}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        out.append(path)
    return out
