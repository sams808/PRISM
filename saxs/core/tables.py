"""Table writers for the SAXS 2.0 outputs: xlsx (several sheets, widths, frozen header, notes sheet),
csv (utf-8 with BOM after ASCII transliteration of every cell, because Excel garbles en-dashes in
plain UTF-8) and markdown ("-" for missing values, notes as bullets), plus the builders of the
standard sheets (fits / model-free / robustness) from the results of the other modules.

Sample-agnostic: extra columns (compositions, x-values) are supplied by the caller.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TRANSLITERATION = {
    "–": "-", "—": "-", "‑": "-", "−": "-", "≈": "~", "±": "+-", "×": "x", "·": ".", "µ": "mu", "μ": "mu", "Å": "A", "°": " deg",
    "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "…": "...", "′": "'", "″": '"', "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "θ": "theta", "λ": "lambda", "σ": "sigma", "χ": "chi", "ρ": "rho",
    "π": "pi", "ξ": "xi", "Δ": "Delta", "Σ": "Sigma", "Ω": "Omega", "ω": "omega", "τ": "tau", "η": "eta", "φ": "phi", "ε": "epsilon",
    "†": "+", "‡": "++", "✓": "yes", "✗": "no", "√": "sqrt",
}


SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ⁿ", "0123456789-+n")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋₊", "0123456789-+")
_SUP_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ⁿ]+")
_SUB_RUN = re.compile("[₀₁₂₃₄₅₆₇₈₉₋₊]+")


def ascii_transliterate(text) -> str:
    """ASCII-only text: superscript runs become ``^...`` (Å⁻¹ -> A^-1), subscripts plain digits
    (Bi₂O₃ -> Bi2O3), the table above, then compatibility decomposition, then dropping the rest."""
    if text is None:
        return ""
    s = str(text)
    s = _SUP_RUN.sub(lambda m: "^" + m.group(0).translate(SUPERSCRIPTS), s)
    s = _SUB_RUN.sub(lambda m: m.group(0).translate(SUBSCRIPTS), s)
    for k, v in TRANSLITERATION.items():
        if k in s:
            s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        return bool(pd.isna(v)) if not isinstance(v, (str, bytes, list, tuple, dict)) else False
    except (TypeError, ValueError):
        return False


def format_cell(v) -> str:
    if _is_missing(v):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


def dataframe_to_markdown(df: pd.DataFrame, notes: Optional[Sequence[str]] = None) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(format_cell(v) for v in r.values) + " |")
    text = "\n".join(lines) + "\n"
    if notes:
        text += "\n" + "\n".join("- " + str(n) for n in notes) + "\n"
    return text


def transliterate_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [ascii_transliterate(c) for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: v if _is_missing(v) or not isinstance(v, str) else ascii_transliterate(v))
    return out


def write_tables(sheets: Dict[str, pd.DataFrame], base_path: str, notes: Optional[Sequence[str]] = None,
                 md_sheet: Optional[str] = None, formats: Sequence[str] = ("xlsx", "csv", "md"), notes_sheet: str = "notes",
                 freeze_panes: str = "B2", max_width: int = 30, notes_width: int = 160, min_width: int = 8) -> Dict[str, str]:
    """Write ``base_path.xlsx`` (all sheets + a notes sheet), ``.csv`` (the ``md_sheet`` or first sheet,
    utf-8-sig, ASCII cells) and ``.md`` (same sheet, notes as bullets). Returns ``{format: path}``."""
    if not sheets:
        raise ValueError("no sheet to write")
    first = md_sheet if (md_sheet in sheets) else next(iter(sheets))
    d = os.path.dirname(base_path)
    if d:
        os.makedirs(d, exist_ok=True)
    out: Dict[str, str] = {}
    if "xlsx" in formats:
        path = base_path + ".xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
            if notes:
                pd.DataFrame({"Notes": list(notes)}).to_excel(xw, sheet_name=notes_sheet[:31], index=False)
            for ws in xw.book.worksheets:
                for col in ws.columns:
                    width = max((len(str(c.value)) if c.value is not None else 0) for c in col) + 2
                    cap = notes_width if ws.title == notes_sheet[:31] else max_width
                    ws.column_dimensions[col[0].column_letter].width = min(max(width, min_width), cap)
                ws.freeze_panes = freeze_panes
        out["xlsx"] = path
    if "csv" in formats:
        path = base_path + ".csv"
        transliterate_frame(sheets[first]).to_csv(path, index=False, encoding="utf-8-sig")
        out["csv"] = path
    if "md" in formats:
        path = base_path + ".md"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(dataframe_to_markdown(sheets[first], notes))
        out["md"] = path
    return out


# ------------------------------------------------------------------------------------ standard sheets
def _sig(v, n=3):
    return None if _is_missing(v) else float(f"{float(v):.{n}g}")


def _round(v, n):
    return None if _is_missing(v) else round(float(v), n)


def fit_sheet(results: Iterable[object], summary: Optional[pd.DataFrame] = None,
              extra: Optional[Dict[str, Dict[str, object]]] = None, amplitude_unit: str = "") -> pd.DataFrame:
    """The fits sheet (one row per :class:`UnifiedFitResult`): loading, T, t_eff, parameters of both
    levels with at-bound markers (*), dBIC, the relaxed-bounds Rg2, the resolved flag, the tail and
    the RMS; ``extra`` adds ``{column: {sample: value}}`` columns (compositions, x-values)."""
    summ = summary.set_index("sample") if (summary is not None and "sample" in summary.columns) else summary
    unit = f" ({amplitude_unit})" if amplitude_unit else ""
    rows = []
    for r in results:
        p = r.params
        star = lambda k: "*" if (k in p and p[k].at_bound) else ""      # noqa: E731
        s = r.sample
        row = {"Sample": s, "Loading": r.cls}
        if summ is not None and s in summ.index:
            row["T"] = _sig(summ.loc[s].get("T_used"), 3)
            row["t_eff (mm)"] = _round(summ.loc[s].get("t_eff_mm"), 2)
        if not r.fitted:
            row.update({"Parameters": None, "Note": r.skip_reason})
            rows.append(row)
            continue
        d = r.derived
        row.update({"q_min (1/A)": _sig(r.q_min, 2), "Parameters": int(r.n_params),
                    "Rg1 (nm)": f"{d['Rg1_nm']:.1f}{star('Rg1')}", "d1": f"{d['d1']:.2f}{star('d1')}", f"G1{unit}": _sig(d["G1"], 3),
                    "Rg2 (nm)": f"{d['Rg2_nm']:.1f}{star('Rg2')}" if r.l2 else None,
                    "Rg2 err (nm)": (_round(p["Rg2"].stderr / 10, 1) if (r.l2 and p["Rg2"].stderr is not None) else None),
                    "D_eq (nm)": _round(d.get("D2_eq_nm"), 1) if r.l2 else None,
                    "d2": f"{d['d2']:.2f}{star('d2')}" if r.l2 else None,
                    f"G2{unit}": _sig(d.get("G2"), 3) if r.l2 else None,
                    "pf2": f"{d['pf2']:.2f}{star('logpf2')}" if r.l2 else None,
                    "dBIC (level 2)": _round(r.dBIC_L2, 0) if _sig(r.dBIC_L2) is not None else None,
                    "Rg2, relaxed bounds (nm)": (_round(r.relaxed.params["Rg2"].value / 10, 1)
                                                 if (r.relaxed is not None and r.relaxed.fitted and r.relaxed.l2) else None),
                    "Level 2 resolved": ("yes" if r.resolved else "no") if r.l2 else "-",
                    "Rising tail": "yes" if r.tail else "no", "RMS log10 I": _round(r.rms_log, 3),
                    "Covariance": "ok" if r.covariance_ok else "unavailable"})
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, mapping in (extra or {}).items():
        df[col] = [mapping.get(s) for s in df["Sample"]]
    return df


def modelfree_sheet(descriptors: pd.DataFrame, extra: Optional[Dict[str, Dict[str, object]]] = None) -> pd.DataFrame:
    """The model-free sheet from :func:`saxs.core.descriptors.describe_batch`'s table."""
    d = descriptors.set_index("sample", drop=False) if "sample" in descriptors.columns else descriptors
    slope_cols = sorted([c for c in d.columns if c.startswith("n_at_")], key=lambda c: float(c[5:]))
    df = pd.DataFrame({"Sample": d["sample"] if "sample" in d.columns else d.index, "Loading": d.get("cls", "")})
    if "n_low" in d:
        df["Low-q exponent"] = d["n_low"].map(lambda v: _round(v, 2))
    for c in slope_cols:
        df[f"Local exponent at {c[5:]}"] = (-d[c]).map(lambda v: _round(v, 2))
    if "C_hi" in d:
        df["High-q plateau"] = d["C_hi"].map(lambda v: _sig(v, 3))
    if "ratio_hi" in d:
        df["Sample/(T x empty) at high q"] = d["ratio_hi"].map(lambda v: _round(v, 1))
    if "K4_over_rhoe2" in d:
        df["K4 / rho_e^2"] = d["K4_over_rhoe2"].map(lambda v: _sig(v, 3))
    if "shoulder_size_nm" in d:
        df["Porod shoulder (nm)"] = d["shoulder_size_nm"].map(lambda v: _round(v, 1))
    if "has_maximum" in d:
        df["Maximum in I or I q^2"] = d["has_maximum"].map(lambda v: "yes" if bool(v) else "no")
    df = df.reset_index(drop=True)
    for col, mapping in (extra or {}).items():
        df[col] = [mapping.get(s) for s in df["Sample"]]
    return df


def robustness_sheet(results: Iterable[object]) -> pd.DataFrame:
    """The relaxed-bounds variant next to the retained fit."""
    rows = []
    for r in results:
        if not r.fitted:
            continue
        w = r.relaxed
        row = {"Sample": r.sample, "Model": r.name, "Rg1 (nm)": _round(r.derived["Rg1_nm"], 1), "d1": _round(r.derived["d1"], 2),
               "Rg2 (nm)": _round(r.derived.get("Rg2_nm"), 1) if r.l2 else None, "RMS log10 I": _round(r.rms_log, 3)}
        if w is not None and w.fitted:
            row.update({"Relaxed model": w.name, "Relaxed Rg1 (nm)": _round(w.derived["Rg1_nm"], 1), "Relaxed d1": _round(w.derived["d1"], 2),
                        "Relaxed Rg2 (nm)": _round(w.derived.get("Rg2_nm"), 1) if w.l2 else None,
                        "Relaxed RMS log10 I": _round(w.rms_log, 3), "Level 2 resolved": ("yes" if r.resolved else "no") if r.l2 else "-"})
        rows.append(row)
    return pd.DataFrame(rows)


def default_notes(fit_options: Optional[Dict[str, object]] = None) -> List[str]:
    o = fit_options or {}
    d1 = o.get("d1_bounds", (3.0, 4.5))
    rg1 = o.get("rg1_bounds", (250.0, 1500.0))
    rg2 = o.get("rg2_bounds", (40.0, 200.0))
    d2 = o.get("d2_bounds", (2.0, 4.5))
    pf = o.get("logpf2_bounds", (-1.0, 0.18))
    return [
        f"Fits: two-level unified fits (Beaucage 1995; Tomita et al. 2024 Eq. 10) of the reduced profiles, q <= {o.get('q_max', 0.20)} 1/A, "
        f"{o.get('per_decade', 40)} points per decade, log10 I weighted by the propagated sigma with a {100 * float(o.get('sigma_floor', 0.03)):.0f} % floor.",
        f"Level 1: Rg1 {rg1[0] / 10:.0f}-{rg1[1] / 10:.0f} nm, d1 {d1[0]}-{d1[1]}, G1 and B1 free. Level 2: Rg2 {rg2[0] / 10:.0f}-{rg2[1] / 10:.0f} nm, "
        f"d2 {d2[0]}-{d2[1]}, B2 tied to G2 by the Beaucage (1996) relation times pf2 in {10 ** pf[0]:.1f}-{10 ** pf[1]:.1f}.",
        f"Level 2 and the rising tail (T q^{o.get('tail_exponent', 2.0):g}) are kept only when they lower the BIC by more than {o.get('dbic_min', 10.0):g} (column dBIC).",
        f"'Level 2 resolved' = level kept by both bound sets (d1 <= {d1[1]}, Rg1 <= {rg1[1] / 10:.0f} nm; d1 <= {o.get('relaxed_d1_max', 5.0)}, "
        f"Rg1 <= {float(o.get('relaxed_rg1_max', 3000.0)) / 10:.0f} nm) with Rg2 agreeing within {100 * float(o.get('robust_tol', 0.2)):.0f} %.",
        "D_eq = 2 Rg2 sqrt(5/3), diameter of the equivalent sphere. A star (*) marks a parameter at a bound.",
        "Amplitudes are comparable between quantified loadings only (measurable, marginal); opaque loadings: shapes only; "
        "unreliable: non-uniform loading.",
    ]
