"""
raman_ops.py — the operations of the Raman workbench, declared once.

Every Process / Transform / Analyze function of the workbench is an `Op`:
a key, a label, the tab and group it sits in, its parameters (with kinds
and defaults, so the UI builds the form and sequences store the values),
its arity (one spectrum at a time, each against a reference, all selected
into one, or a table over all) and the function that does the work on
plain arrays. The functions wrap the science modules that already exist
(baseline_science, calc_science, raman_tools, spectrum_math); the few new
ones (triangular and percentile smoothing, linear two-range baseline,
detrending, merge, integration, centre of gravity, extract data) live
here.

Because an operation is data, the Sequence tab records (key, params,
options) and replays them without any UI code, and a saved sequence is a
small JSON file. Nothing here imports Qt.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.np_compat import trapezoid

XY = Tuple[np.ndarray, np.ndarray]


@dataclass
class Param:
    key: str
    label: str
    default: Any
    kind: str = "float"            # float | int | str | bool | choice | text
    choices: Optional[List[str]] = None
    tip: str = ""
    pick: Any = False              # False, True (mode inferred from the key) or "x" | "range" | "list"


def pick_mode(p: "Param") -> Optional[str]:
    """How the UI picks this parameter on the plot: None (no picking), 'x'
    (one click fills the value), 'range' (two clicks per range, any number
    of ranges) or 'list' (one click per value, any number of values)."""
    if not p.pick:
        return None
    if isinstance(p.pick, str):
        return p.pick
    if p.kind == "str" and any(k in p.key for k in ("range", "regions")):
        return "range"
    if p.kind == "str" and any(k in p.key for k in ("positions", "points", "manual")):
        return "list"
    return "x"


@dataclass
class SpecIn:
    x: np.ndarray
    y: np.ndarray
    title: str
    meta: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class OpResult:
    spectra: List[Tuple[np.ndarray, np.ndarray, str]] = field(default_factory=list)
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    text: str = ""


@dataclass
class Op:
    key: str
    label: str
    tab: str                       # process | transform | analyze
    group: str
    params: List[Param]
    arity: str                     # each | reference | all | table
    func: Callable
    suffix: str = ""
    tip: str = ""

    def defaults(self) -> Dict[str, Any]:
        return {p.key: p.default for p in self.params}


# ----------------------------------------------------------------------
# Parameter conversion
# ----------------------------------------------------------------------

def _to_float(v, default=None):
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"'{v}' is not a number.") from exc


def parse_params(op: Op, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI / JSON values to typed values, validating choices.
    Missing keys take the default; empty floats become None."""
    out: Dict[str, Any] = {}
    for p in op.params:
        v = raw.get(p.key, p.default)
        if p.kind == "float":
            out[p.key] = _to_float(v, None if v in (None, "") else p.default)
        elif p.kind == "int":
            f = _to_float(v, None)
            out[p.key] = int(round(f)) if f is not None else (int(p.default) if p.default not in (None, "") else None)
        elif p.kind == "bool":
            out[p.key] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
        elif p.kind == "choice":
            v = str(v)
            if p.choices and v not in p.choices:
                raise ValueError(f"{p.label}: '{v}' is not one of {p.choices}.")
            out[p.key] = v
        else:
            out[p.key] = "" if v is None else str(v)
    return out


def parse_ranges(text: str, *, lenient: bool = False) -> List[Tuple[float, float]]:
    """'100-400; 1800-2600' -> [(100, 400), (1800, 2600)] (commas inside a
    pair are accepted too). A range given in the wrong order is swapped.
    lenient=True skips whatever cannot be read instead of raising (the plot
    overlay uses it while the user is still typing or clicking)."""
    out = []
    for chunk in re.split(r"[;\n]+", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        norm = re.sub(r"(?<=[\d.])\s*-\s*(?=[\d.+-])", ",", chunk)
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", norm)
        if len(nums) != 2:
            if lenient:
                continue
            raise ValueError(f"Range '{chunk}': expected 'min-max'.")
        lo, hi = sorted((float(nums[0]), float(nums[1])))
        if hi <= lo:
            if lenient:
                continue
            raise ValueError(f"Range '{chunk}': the two ends are the same point; click two different x positions.")
        out.append((lo, hi))
    return out


def parse_positions(text: str) -> List[float]:
    """'520.5, 960; 1100' -> [520.5, 960.0, 1100.0] (any separator)."""
    return [float(t) for t in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", text or "")]


# ----------------------------------------------------------------------
# New science helpers
# ----------------------------------------------------------------------

def _clean(x, y) -> XY:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[m], kind="mergesort")
    return x[m][order], y[m][order]


def triangular_average(y: np.ndarray, window: int) -> np.ndarray:
    """Moving average with a triangular kernel, normalised by the kernel
    mass that falls inside the array (no edge droop)."""
    y = np.asarray(y, float)
    w = max(3, int(window) | 1)
    half = w // 2
    kernel = np.array([half + 1 - abs(i - half) for i in range(w)], float)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones(y.size), kernel, mode="same")
    return num / den


def percentile_filter(y: np.ndarray, window: int, percentile: float) -> np.ndarray:
    """Running percentile (0 % = lower envelope, 50 % = median, 100 % = upper envelope)."""
    from scipy.ndimage import percentile_filter as _pf
    w = max(3, int(window) | 1)
    return _pf(np.asarray(y, float), float(np.clip(percentile, 0, 100)), size=w, mode="reflect")


def linear_baseline(x: np.ndarray, y: np.ndarray, ranges: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Straight baseline through the mean points of one or two ranges (one
    range: a horizontal line at its mean)."""
    x, y = _clean(x, y)
    pts = []
    for lo, hi in ranges:
        m = (x >= lo) & (x <= hi)
        if m.sum() == 0:
            raise ValueError(f"No data in {lo:g}-{hi:g}.")
        pts.append((float(np.mean(x[m])), float(np.mean(y[m]))))
    if not pts:
        raise ValueError("Give one or two baseline ranges.")
    if len(pts) == 1:
        return np.full(x.size, pts[0][1])
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    if x2 == x1:
        return np.full(x.size, 0.5 * (y1 + y2))
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def anchor_baseline(x: np.ndarray, y: np.ndarray, points: Sequence[float], *, kind: str = "straight segments",
                    half_width: float = 0.0) -> np.ndarray:
    """Baseline through the spectrum's own values at anchor x positions (the
    points clicked on the plot): each anchor takes the mean of y within
    ± half_width (the nearest sample when 0), the anchors are joined by
    straight segments or a natural cubic spline, and the baseline is flat
    beyond the outer anchors."""
    x, y = _clean(x, y)
    pts = sorted(set(float(p) for p in points))
    if len(pts) < 2:
        raise ValueError("Give at least two anchor points (Pick: one click per point).")
    ya = []
    for p in pts:
        m = np.abs(x - p) <= half_width if half_width > 0 else None
        if m is not None and m.any():
            ya.append(float(np.mean(y[m])))
        else:
            ya.append(float(y[int(np.argmin(np.abs(x - p)))]))
    xa, ya = np.array(pts, float), np.array(ya, float)
    xc = np.clip(x, xa[0], xa[-1])
    if kind.startswith("smooth") and len(pts) >= 3:
        from scipy.interpolate import CubicSpline
        return np.asarray(CubicSpline(xa, ya, bc_type="natural")(xc), float)
    return np.interp(xc, xa, ya)


def detrend(x: np.ndarray, y: np.ndarray, order: int = 1) -> np.ndarray:
    x, y = _clean(x, y)
    coef = np.polyfit(x, y, int(max(0, min(order, 6))))
    return y - np.polyval(coef, x)


def integrate(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float, baseline: str = "zero") -> Dict[str, float]:
    """Area between xmin and xmax: above zero, or above the straight line
    through the two end points ('linear', Spectragryph's tilted baseline)."""
    x, y = _clean(x, y)
    lo, hi = min(xmin, xmax), max(xmin, xmax)
    m = (x >= lo) & (x <= hi)
    if m.sum() < 2:
        raise ValueError(f"Fewer than two points in {lo:g}-{hi:g}.")
    xs, ys = x[m], y[m]
    if baseline == "linear":
        base = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / (xs[-1] - xs[0])
    else:
        base = np.zeros_like(ys)
    net = ys - base
    area = float(trapezoid(net, xs))
    pos = np.clip(net, 0, None)
    cog = float(trapezoid(xs * pos, xs) / trapezoid(pos, xs)) if trapezoid(pos, xs) > 0 else float("nan")
    i = int(np.argmax(net))
    return {"area": area, "cog": cog, "max_x": float(xs[i]), "max_net": float(net[i]), "n": int(m.sum())}


def extract_properties(x: np.ndarray, y: np.ndarray, xmin: Optional[float], xmax: Optional[float],
                       y_at: Optional[float] = None) -> Dict[str, float]:
    """The per-spectrum numbers of Spectragryph's Extract Data, inside a range."""
    from raman.raman_tools import peak_table
    x, y = _clean(x, y)
    if xmin is not None and xmax is not None:
        m = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
        if m.sum() < 2:
            raise ValueError("Fewer than two points in the range.")
        x, y = x[m], y[m]
    i_max, i_min = int(np.argmax(y)), int(np.argmin(y))
    out = {
        "x_at_max": float(x[i_max]), "max": float(y[i_max]), "x_at_min": float(x[i_min]), "min": float(y[i_min]),
        "mean": float(np.mean(y)), "std": float(np.std(y)), "area": float(trapezoid(y, x)),
        "cog": float(trapezoid(x * np.clip(y, 0, None), x) / trapezoid(np.clip(y, 0, None), x)) if trapezoid(np.clip(y, 0, None), x) > 0 else float("nan"),
        "first": float(y[0]), "last": float(y[-1]),
    }
    if y_at is not None:
        out["y_at"] = float(np.interp(float(y_at), x, y))
    rows = peak_table(x, y, threshold_frac=0.0, min_prominence_frac=0.02, max_peaks=200)
    if rows:
        top = max(rows, key=lambda r: r.height)
        out["fwhm_of_max"] = float(top.fwhm)
        out["n_peaks_2pct"] = float(len(rows))
    else:
        out["fwhm_of_max"] = float("nan"); out["n_peaks_2pct"] = 0.0
    return out


def merge_spectra(spectra: Sequence[XY]) -> XY:
    """Join spectra measured over different x ranges into one: a common grid
    at the finest step, averaging where they overlap and interpolating
    across gaps."""
    cleaned = [_clean(x, y) for x, y in spectra]
    if len(cleaned) < 2:
        raise ValueError("Merge needs at least two spectra.")
    lo = min(float(x[0]) for x, _ in cleaned); hi = max(float(x[-1]) for x, _ in cleaned)
    step = min(float(np.median(np.diff(x))) for x, _ in cleaned if x.size > 1)
    grid = np.arange(lo, hi + 0.5 * step, step)
    total = np.zeros(grid.size); count = np.zeros(grid.size)
    for x, y in cleaned:
        inside = (grid >= x[0]) & (grid <= x[-1])
        total[inside] += np.interp(grid[inside], x, y)
        count[inside] += 1
    merged = np.full(grid.size, np.nan)
    has = count > 0
    merged[has] = total[has] / count[has]
    if not has.all():
        merged[~has] = np.interp(grid[~has], grid[has], merged[has])
    return grid, merged


# ----------------------------------------------------------------------
# Operation functions (each: (spec, params) -> (x, y); reference: (spec, ref, params);
# all: (specs, params) -> (x, y, title); table: (specs, params) -> (columns, rows, text))
# ----------------------------------------------------------------------

def _op_baseline_simple(s: SpecIn, p):
    from raman.raman_tools import subtract_range_mean
    r = parse_ranges(p["range"])
    if len(r) != 1:
        raise ValueError("Give one range, e.g. 1800-2000.")
    return subtract_range_mean(s.x, s.y, xmin=r[0][0], xmax=r[0][1])


def _op_baseline_linear(s: SpecIn, p):
    x, y = _clean(s.x, s.y)
    base = linear_baseline(x, y, parse_ranges(p["ranges"]))
    return (x, base) if p["output"] == "baseline" else (x, y - base)


def _op_baseline_advanced(s: SpecIn, p):
    import processing.baseline_science as bs
    method = p["method"]
    if method == "anchors":
        x, y = _clean(s.x, s.y)
        base = anchor_baseline(x, y, parse_positions(p["points"]), kind=p["anchor_fit"],
                               half_width=float(p["anchor_window"] or 0.0))
        return (x, base) if p["output"] == "baseline" else (x, y - base)
    roi = None
    if p["regions"].strip():
        roi = np.array(parse_ranges(p["regions"]), float)
    params = {}
    if method in ("arPLS", "als"):
        params["lam"] = float(p["lam"])
        params["ratio" if method == "arPLS" else "p"] = float(p["ratio"])
    elif method == "poly":
        params["polynomial_order"] = int(p["order"])
    elif method == "unispline":
        params["s"] = float(p["spline_s"])
    x, y_sub, base = bs.compute_baseline(s.x, s.y, method=method, roi=roi, params=params)
    return (x, base) if p["output"] == "baseline" else (x, y_sub)


def _op_normalize(s: SpecIn, p):
    import processing.calc_science as cs
    from raman.raman_tools import normalize_to_value
    mode = p["mode"]
    if mode == "peak (max = 1)":
        return cs.transform(s.x, s.y, "normalize_max")
    if mode == "area (= 100)":
        return cs.transform(s.x, s.y, "normalize_area")
    if mode == "min-max (0..1)":
        return cs.transform(s.x, s.y, "normalize_minmax")
    if mode == "value at x":
        if p["x0"] is None:
            raise ValueError("Give x₀.")
        return normalize_to_value(s.x, s.y, x0=p["x0"], target=p["target"] if p["target"] is not None else 1.0)
    if mode == "peak in range":
        r = parse_ranges(p["range"])
        if len(r) != 1:
            raise ValueError("Give one range.")
        x, y = _clean(s.x, s.y)
        m = (x >= r[0][0]) & (x <= r[0][1])
        if m.sum() == 0:
            raise ValueError("No data in the range.")
        peak = float(np.max(y[m]))
        if peak == 0:
            raise ValueError("The maximum in the range is zero.")
        return x, y / peak
    raise ValueError(mode)


def _op_smooth(s: SpecIn, p):
    import processing.calc_science as cs
    method = p["method"]
    x, y = _clean(s.x, s.y)
    w = int(p["window"])
    if method == "Savitzky-Golay":
        return cs.smooth(x, y, method="savgol", window=w, polyorder=int(p["polyorder"]))
    if method == "moving average":
        return cs.smooth(x, y, method="moving_average", window=w)
    if method == "triangular average":
        return x, triangular_average(y, w)
    if method == "median":
        return cs.smooth(x, y, method="median", window=w)
    if method == "percentile filter":
        return x, percentile_filter(y, w, float(p["percentile"]))
    raise ValueError(method)


def _op_despike(s: SpecIn, p):
    import processing.calc_science as cs
    pos = parse_positions(p["positions"])
    return cs.despike(s.x, s.y, z=float(p["z"]), window=int(p["window"]), positions=pos or None)


def _op_remove_peak(s: SpecIn, p):
    from raman.raman_tools import remove_peak
    return remove_peak(s.x, s.y, center=float(p["center"]), fwhm=float(p["fwhm"]))


def _op_cut(s: SpecIn, p):
    import processing.calc_science as cs
    x, y = _clean(s.x, s.y)
    lo = p["xmin"] if p["xmin"] is not None else float(x[0])
    hi = p["xmax"] if p["xmax"] is not None else float(x[-1])
    return cs.crop(x, y, xmin=lo, xmax=hi)


def _op_offset_stretch(s: SpecIn, p):
    x, y = _clean(s.x, s.y)
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "×": lambda a, b: a * b, "÷": lambda a, b: a / b}
    if p["axis"] == "y":
        return x, ops[p["op"]](y, float(p["value"]))
    if p["op"] == "÷" and float(p["value"]) == 0:
        raise ValueError("Division by zero.")
    xn = ops[p["op"]](x, float(p["value"]))
    order = np.argsort(xn, kind="mergesort")
    return xn[order], y[order]


def _op_snv(s: SpecIn, p):
    from raman.raman_tools import snv
    x, y = _clean(s.x, s.y)
    return x, snv(y)


def _op_detrend(s: SpecIn, p):
    x, y = _clean(s.x, s.y)
    return x, detrend(x, y, int(p["order"]))


def _op_msc(s: SpecIn, ref: SpecIn, p):
    from raman.raman_tools import msc
    from core.spectrum_math import _common_grid, _on_grid
    grid = _common_grid([(s.x, s.y), (ref.x, ref.y)])
    return grid, msc(_on_grid(s.x, s.y, grid), _on_grid(ref.x, ref.y, grid))


def _op_arithmetic(s: SpecIn, ref: SpecIn, p):
    import processing.calc_science as cs
    op = {"A + B": "add", "A − B": "subtract", "A × B": "multiply", "A ÷ B": "divide"}[p["op"]]
    return cs.multi_op([(s.x, s.y), (ref.x, ref.y)], op)


def _op_scaled_subtraction(s: SpecIn, ref: SpecIn, p):
    from core.spectrum_math import combine_spectra
    return combine_spectra([(s.x, s.y), (ref.x, ref.y)], op="subtract", weights=[1.0, float(p["k"])])


def _op_divide_reference(s: SpecIn, ref: SpecIn, p):
    import processing.calc_science as cs
    return cs.multi_op([(s.x, s.y), (ref.x, ref.y)], "divide")


def _op_average(specs: Sequence[SpecIn], p):
    import processing.calc_science as cs
    if len(specs) < 2:
        raise ValueError("Select at least two spectra.")
    op = "average" if p["mode"] == "mean" else "median_spectra"
    x, y = cs.multi_op([(s.x, s.y) for s in specs], op)
    return x, y, f"{p['mode']}_of_{len(specs)}"


def _op_merge(specs: Sequence[SpecIn], p):
    x, y = merge_spectra([(s.x, s.y) for s in specs])
    return x, y, f"merged_{len(specs)}"


def _op_axis_convert(s: SpecIn, p):
    from raman.raman_tools import convert_x_axis
    x, y = convert_x_axis(s.x, s.y, p["from_unit"], p["to_unit"], laser_nm=p["laser_nm"], jacobian=bool(p["jacobian"]))
    if p["begin_at"] is not None:
        m = x >= float(p["begin_at"])
        x, y = x[m], y[m]
    return x, y


def _op_derivative(s: SpecIn, p):
    import processing.calc_science as cs
    return cs.derivative(s.x, s.y, order=int(p["order"]), window=int(p["window"]), polyorder=int(p["polyorder"]))


def _op_resample(s: SpecIn, p):
    import processing.calc_science as cs
    x, y = _clean(s.x, s.y)
    lo = p["xmin"] if p["xmin"] is not None else float(x[0])
    hi = p["xmax"] if p["xmax"] is not None else float(x[-1])
    if p["step"] is not None and float(p["step"]) > 0:
        n = int(round((hi - lo) / float(p["step"]))) + 1
    else:
        n = int(p["n_points"])
    return cs.resample(x, y, n_points=n, xmin=lo, xmax=hi)


def _op_calibrate_x(s: SpecIn, p):
    from raman.raman_tools import calibrate_x_axis, parse_pairs
    measured, reference = parse_pairs(p["pairs"])
    cal = calibrate_x_axis(measured, reference, order=int(p["order"]))
    x, y = _clean(s.x, s.y)
    xn = cal.apply(x)
    order = np.argsort(xn, kind="mergesort")
    return xn[order], y[order]


def _op_peak_table(specs: Sequence[SpecIn], p):
    from raman.raman_tools import manual_peak_rows, peak_table
    manual = parse_positions(p.get("manual", ""))
    only_manual = bool(manual) and p.get("manual_mode", "") == "the only peaks"
    rows = []
    for s in specs:
        r = parse_ranges(p["range"]) if p["range"].strip() else None
        found = [] if only_manual else peak_table(
            s.x, s.y, threshold_frac=float(p["threshold"]) / 100.0, min_prominence_frac=float(p["prominence"]) / 100.0,
            min_distance=float(p["min_distance"] or 0.0), inverted=bool(p["inverted"]), x_range=r[0] if r else None)
        hand = manual_peak_rows(s.x, s.y, manual, snap=float(p.get("snap") or 0.0), inverted=bool(p["inverted"])) if manual else []
        taken = {pk.index for pk in found}
        merged = [(pk, "auto") for pk in found] + [(pk, "manual") for pk in hand if pk.index not in taken]
        merged.sort(key=lambda t: t[0].position)
        for pk, source in merged:
            rows.append([s.title, round(pk.position, 2), float(f"{pk.height:.6g}"), round(pk.fwhm, 2),
                         float(f"{pk.area:.6g}"), float(f"{pk.prominence:.6g}"), source])
    n_hand = sum(1 for row in rows if row[-1] == "manual")
    text = f"{len(rows)} peak(s)" + (f", {n_hand} picked by hand" if n_hand else "")
    return ["Spectrum", "Position", "Height", "FWHM", "Area", "Prominence", "Source"], rows, text


def _op_integrate(specs: Sequence[SpecIn], p):
    rows = []
    r = parse_ranges(p["range"])
    if len(r) != 1:
        raise ValueError("Give one range.")
    for s in specs:
        d = integrate(s.x, s.y, r[0][0], r[0][1], baseline=p["baseline"])
        rows.append([s.title, r[0][0], r[0][1], p["baseline"], float(f"{d['area']:.6g}"), round(d["cog"], 2),
                     round(d["max_x"], 2), float(f"{d['max_net']:.6g}")])
    areas = [row[4] for row in rows]
    text = f"{len(rows)} spectrum/spectra; mean area {np.mean(areas):.6g}" if areas else ""
    return ["Spectrum", "x min", "x max", "Baseline", "Area", "Centre of gravity", "x at max", "Max above base"], rows, text


def _op_extract(specs: Sequence[SpecIn], p):
    rows = []
    cols = ["Spectrum", "x at max", "Max", "x at min", "Min", "Mean", "Std", "Area", "Centre of gravity",
            "FWHM of max peak", "Peaks ≥2 %", "First", "Last"]
    if p["y_at"] is not None:
        cols.append(f"y at {p['y_at']:g}")
    for s in specs:
        d = extract_properties(s.x, s.y, p["xmin"], p["xmax"], y_at=p["y_at"])
        row = [s.title, round(d["x_at_max"], 2), float(f"{d['max']:.6g}"), round(d["x_at_min"], 2), float(f"{d['min']:.6g}"),
               float(f"{d['mean']:.6g}"), float(f"{d['std']:.6g}"), float(f"{d['area']:.6g}"), round(d["cog"], 2),
               round(d["fwhm_of_max"], 2), int(d["n_peaks_2pct"]), float(f"{d['first']:.6g}"), float(f"{d['last']:.6g}")]
        if p["y_at"] is not None:
            row.append(float(f"{d['y_at']:.6g}"))
        rows.append(row)
    return cols, rows, f"{len(rows)} spectrum/spectra"


def _op_statistics(specs: Sequence[SpecIn], p):
    import processing.calc_science as cs
    rows = []
    for s in specs:
        st = cs.statistics_report(s.x, s.y)
        rows.append([s.title] + [float(f"{v:.6g}") for v in st.values()])
    cols = ["Spectrum"] + (list(cs.statistics_report(specs[0].x, specs[0].y).keys()) if specs else [])
    return cols, rows, ""


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------

UNITS = ["Raman shift (cm⁻¹)", "Wavelength (nm)", "Wavenumber (cm⁻¹)", "Photon energy (eV)"]

OPS: List[Op] = [
    # ---- Process
    Op("baseline_simple", "Simple baseline", "process", "Baseline",
       [Param("range", "signal-free range", "1800-2000", "str", tip="x-range whose mean is subtracted", pick=True)],
       "each", _op_baseline_simple, "offset", "Subtract the mean of a signal-free range (an offset correction)."),
    Op("baseline_linear", "Linear baseline", "process", "Baseline",
       [Param("ranges", "anchor ranges", "100-150; 1800-2000", "str", tip="one or two ranges; a line through their mean points", pick=True),
        Param("output", "output", "corrected", "choice", ["corrected", "baseline"])],
       "each", _op_baseline_linear, "lin", "Straight baseline through one or two signal-free ranges."),
    Op("baseline_advanced", "Advanced baseline", "process", "Baseline",
       [Param("method", "method", "arPLS", "choice", ["arPLS", "als", "poly", "unispline", "rubberband", "anchors"]),
        Param("lam", "λ (arPLS/ALS stiffness)", 1e5, "float", tip="1e5 default; 1e7 for broad glass bands"),
        Param("ratio", "ratio / p", 0.01, "float"),
        Param("order", "polynomial order", 3, "int"),
        Param("spline_s", "spline s", 1.0, "float"),
        Param("regions", "regions (poly/spline/rubberband)", "", "str",
              tip="x-ranges the baseline is fitted through, e.g. 100-400; 1800-2600 (Pick: two clicks per range, as many ranges as needed)", pick=True),
        Param("points", "anchor points (anchors method)", "", "str",
              tip="x positions where the baseline passes through the spectrum (Pick: one click per point, clicks append)", pick="list"),
        Param("anchor_fit", "anchors joined by", "straight segments", "choice", ["straight segments", "smooth spline"]),
        Param("anchor_window", "anchor half-width (x units)", 0.0, "float",
              tip="each anchor takes the mean of the spectrum within ± this; 0 = the nearest sample"),
        Param("output", "output", "corrected", "choice", ["corrected", "baseline"])],
       "each", _op_baseline_advanced, "bl",
       "Asymmetric least squares (arPLS, ALS), polynomial, spline or rubberband baseline, or a baseline through anchor points you click."),
    Op("normalize", "Normalize", "process", "Normalize",
       [Param("mode", "to", "peak (max = 1)", "choice", ["peak (max = 1)", "area (= 100)", "min-max (0..1)", "value at x", "peak in range"]),
        Param("x0", "x₀ (value mode)", "", "float", pick=True), Param("target", "target (value mode)", 1.0, "float"),
        Param("range", "range (peak-in-range mode)", "", "str", pick=True)],
       "each", _op_normalize, "norm", "Scale each spectrum to its peak, its area, 0..1, or a value at x₀."),
    Op("smooth", "Smooth", "process", "Smooth",
       [Param("method", "method", "Savitzky-Golay", "choice", ["Savitzky-Golay", "moving average", "triangular average", "median", "percentile filter"]),
        Param("window", "window (points)", 11, "int"), Param("polyorder", "polynomial order (SG)", 3, "int"),
        Param("percentile", "percentile (0 = lower envelope)", 50.0, "float")],
       "each", _op_smooth, "sm", "Savitzky-Golay keeps peak shapes; the median kills spikes; percentile gives envelopes."),
    Op("despike", "Remove spikes", "process", "Clean",
       [Param("z", "threshold (× robust σ)", 6.0, "float"), Param("window", "window (points)", 7, "int"),
        Param("positions", "forced positions (x, comma-separated)", "", "str", pick=True)],
       "each", _op_despike, "despiked", "Cosmic-ray spikes replaced by the rolling median; clicked positions are always removed."),
    Op("remove_peak", "Remove a peak", "process", "Clean",
       [Param("center", "centre", "", "float", pick=True), Param("fwhm", "FWHM", 10.0, "float")],
       "each", _op_remove_peak, "peakrm", "Replace a peak (±1.5 FWHM) by a straight line."),
    Op("cut", "Cut off", "process", "Clean",
       [Param("xmin", "x min", "", "float", pick=True), Param("xmax", "x max", "", "float", pick=True)],
       "each", _op_cut, "cut", "Keep only the x-range (empty = from the start / to the end)."),
    Op("offset_stretch", "Offset / stretch", "process", "Axes",
       [Param("axis", "axis", "y", "choice", ["x", "y"]), Param("op", "operation", "+", "choice", ["+", "-", "×", "÷"]),
        Param("value", "value", 0.0, "float")],
       "each", _op_offset_stretch, "scaled", "Add, subtract, multiply or divide every x or y value by a constant."),
    Op("snv", "SNV", "process", "Scatter",
       [], "each", _op_snv, "snv", "Standard normal variate: (y − mean) / std."),
    Op("detrend", "Detrend", "process", "Scatter",
       [Param("order", "polynomial order", 1, "int")], "each", _op_detrend, "detr", "Subtract a polynomial fitted to the whole spectrum."),
    Op("msc", "MSC against a reference", "process", "Scatter",
       [], "reference", _op_msc, "msc", "Multiplicative scatter correction: regress on the reference, remove offset and slope."),
    # ---- Transform
    Op("arithmetic", "Arithmetic with a reference", "transform", "Calc",
       [Param("op", "operation", "A − B", "choice", ["A + B", "A − B", "A × B", "A ÷ B"])],
       "reference", _op_arithmetic, "calc", "Each selected spectrum (A) with the reference spectrum (B) on the overlap grid."),
    Op("scaled_subtraction", "Scaled subtraction", "transform", "Calc",
       [Param("k", "factor k (A − k·B)", 1.0, "float")],
       "reference", _op_scaled_subtraction, "sub", "A − k·B; drag the slider in the panel to see the result live."),
    Op("divide_reference", "Sensitivity correction", "transform", "Calc",
       [], "reference", _op_divide_reference, "corr", "Divide by a system sensitivity (white-light) curve."),
    Op("average", "Average / median", "transform", "Combine",
       [Param("mode", "mode", "mean", "choice", ["mean", "median"])], "all", _op_average, "", "One spectrum from all selected spectra (common overlap grid)."),
    Op("merge", "Merge", "transform", "Combine",
       [], "all", _op_merge, "", "Join spectra of different ranges: overlaps averaged, gaps interpolated."),
    Op("axis_convert", "Convert x axis", "transform", "Axis",
       [Param("from_unit", "from", UNITS[1], "choice", UNITS), Param("to_unit", "to", UNITS[0], "choice", UNITS),
        Param("laser_nm", "laser λ (nm)", 532.0, "float"), Param("jacobian", "Jacobian (keep the integral)", False, "bool"),
        Param("begin_at", "begin spectrum at (cut below)", "", "float", tip="e.g. 50 to drop the Rayleigh region", pick=True)],
       "each", _op_axis_convert, "xconv", "Wavelength, wavenumber, Raman shift (needs the laser) and photon energy."),
    Op("derivative", "Derivative", "transform", "Axis",
       [Param("order", "order", "1", "choice", ["1", "2", "3", "4"]), Param("window", "window (points)", 11, "int"),
        Param("polyorder", "polynomial order", 3, "int")],
       "each", _op_derivative, "d", "Savitzky-Golay derivative (smoothing is part of it)."),
    Op("resample", "Interpolate / resample", "transform", "Axis",
       [Param("xmin", "x min", "", "float", pick=True), Param("xmax", "x max", "", "float", pick=True),
        Param("step", "step (x units)", "", "float", tip="leave empty to use the number of points"), Param("n_points", "number of points", 1000, "int")],
       "each", _op_resample, "resamp", "Linear interpolation onto a new regular grid."),
    Op("calibrate_x", "Calibrate x axis", "transform", "Axis",
       [Param("pairs", "measured -> reference; …", "520.3 -> 520.7", "text"), Param("order", "polynomial order", 1, "int")],
       "each", _op_calibrate_x, "xcal", "Polynomial correction of the x axis from known lines (silicon 520.7, neon, …)."),
    # ---- Analyze (tables)
    Op("peak_table", "Peaks & FWHM", "analyze", "Peaks",
       [Param("threshold", "threshold (% of range)", 5.0, "float"), Param("prominence", "min prominence (% of range)", 2.0, "float"),
        Param("min_distance", "min separation (x units)", "", "float"), Param("inverted", "find minima", False, "bool"),
        Param("range", "range (optional)", "", "str", pick=True),
        Param("manual", "manual peaks (x, comma-separated)", "", "str", pick="list",
              tip="peaks you point at yourself: Pick, then one click per peak (clicks append; right-click or Esc to finish)"),
        Param("manual_mode", "manual peaks are", "added to the detected", "choice", ["added to the detected", "the only peaks"]),
        Param("snap", "snap manual picks to the apex within (x units)", 5.0, "float", tip="0 = use the clicked x as is (a shoulder)")],
       "table", _op_peak_table, "",
       "Position, height, FWHM (at half prominence) and area of every peak: detected automatically and/or pointed at by hand."),
    Op("integrate", "Integrate", "analyze", "Areas",
       [Param("range", "range", "", "str", pick=True), Param("baseline", "baseline", "zero", "choice", ["zero", "linear"])],
       "table", _op_integrate, "", "Area in a range above zero or above the line through the end points, with the centre of gravity."),
    Op("extract", "Extract data", "analyze", "Series",
       [Param("xmin", "x min (optional)", "", "float", pick=True), Param("xmax", "x max (optional)", "", "float", pick=True),
        Param("y_at", "y at x (optional)", "", "float", pick=True)],
       "table", _op_extract, "", "One row per spectrum: maximum, minimum, mean, std, area, centre of gravity, FWHM, y at x."),
    Op("statistics", "Statistics", "analyze", "Series",
       [], "table", _op_statistics, "", "Summary statistics per spectrum."),
]

OPS_BY_KEY: Dict[str, Op] = {op.key: op for op in OPS}


def ops_for_tab(tab: str) -> List[Op]:
    return [op for op in OPS if op.tab == tab]


# ----------------------------------------------------------------------
# Running an operation
# ----------------------------------------------------------------------

def run_op(op: Op, inputs: Sequence[SpecIn], raw_params: Dict[str, Any], *,
           reference: Optional[SpecIn] = None, suffix: Optional[str] = None) -> OpResult:
    p = parse_params(op, raw_params)
    sfx = suffix if suffix is not None else op.suffix
    res = OpResult()
    if op.arity == "each":
        for s in inputs:
            x, y = op.func(s, p)
            res.spectra.append((np.asarray(x, float), np.asarray(y, float), f"{s.title}_{sfx}" if sfx else s.title))
    elif op.arity == "reference":
        if reference is None:
            raise ValueError(f"'{op.label}' needs a reference spectrum.")
        for s in inputs:
            if reference.id is not None and s.id == reference.id:
                continue
            x, y = op.func(s, reference, p)
            res.spectra.append((np.asarray(x, float), np.asarray(y, float), f"{s.title}_{sfx}" if sfx else s.title))
    elif op.arity == "all":
        x, y, title = op.func(list(inputs), p)
        res.spectra.append((np.asarray(x, float), np.asarray(y, float), title))
    elif op.arity == "table":
        cols, rows, text = op.func(list(inputs), p)
        res.columns, res.rows, res.text = cols, rows, text
    else:
        raise ValueError(op.arity)
    return res


# ----------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------

SEQUENCE_FORMAT = "prism-raman-sequence"


@dataclass
class Step:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    reference_title: Optional[str] = None     # for reference ops: the spectrum title to use
    options: Dict[str, Any] = field(default_factory=dict)  # suffix, replace

    def describe(self) -> str:
        op = OPS_BY_KEY[self.op]
        bits = [f"{p.label}={self.params.get(p.key, p.default)}" for p in op.params
                if str(self.params.get(p.key, p.default)) not in ("", "None")]
        s = op.label + ((" (" + ", ".join(bits) + ")") if bits else "")
        if self.reference_title:
            s += f" [ref: {self.reference_title}]"
        return s


def save_sequence(steps: Sequence[Step], path: str, *, name: str = "") -> None:
    data = {"format": SEQUENCE_FORMAT, "version": 1, "name": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [asdict(st) for st in steps]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_sequence(path: str) -> List[Step]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != SEQUENCE_FORMAT:
        raise ValueError(f"{os.path.basename(path)} is not a PRISM Raman sequence.")
    steps = []
    for st in data.get("steps", []):
        if st.get("op") not in OPS_BY_KEY:
            raise ValueError(f"Unknown operation in sequence: {st.get('op')!r}.")
        steps.append(Step(op=st["op"], params=dict(st.get("params", {})), reference_title=st.get("reference_title"),
                          options=dict(st.get("options", {}))))
    return steps


def run_sequence(steps: Sequence[Step], inputs: Sequence[SpecIn], *,
                 reference_lookup: Optional[Callable[[str], Optional[SpecIn]]] = None,
                 log: Optional[Callable[[str], None]] = None) -> Tuple[List[SpecIn], List[OpResult]]:
    """Apply the steps in order; the output spectra of one step are the
    inputs of the next (table steps leave the spectra unchanged). Returns
    (final spectra, every step's result)."""
    current = list(inputs)
    results: List[OpResult] = []
    for i, st in enumerate(steps, 1):
        op = OPS_BY_KEY[st.op]
        ref = None
        if op.arity == "reference":
            if reference_lookup is None or not st.reference_title:
                raise ValueError(f"Step {i} ({op.label}) needs a reference spectrum.")
            ref = reference_lookup(st.reference_title)
            if ref is None:
                raise ValueError(f"Step {i} ({op.label}): reference '{st.reference_title}' is not loaded.")
        res = run_op(op, current, st.params, reference=ref, suffix=st.options.get("suffix"))
        results.append(res)
        if log:
            log(f"step {i}: {st.describe()} -> " + (f"{len(res.spectra)} spectra" if res.spectra else (res.text or "table")))
        if res.spectra:
            current = [SpecIn(x, y, title, meta=dict(current[0].meta) if current else {}) for x, y, title in res.spectra]
    return current, results
