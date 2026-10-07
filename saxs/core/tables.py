"""Table writers for the SAXS outputs: xlsx (several sheets, widths, frozen header, notes sheet), csv (utf-8 with BOM
after ASCII transliteration of every cell, because Excel garbles en-dashes in plain UTF-8) and markdown ("-" for
missing values, notes as bullets), the model-free sheet of saxs.core.descriptors, and the sheets of a fit-workflow
result (summary, models, parameters, robustness).

Sample-agnostic: extra columns (compositions, x-values) are supplied by the caller.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from typing import Dict, Optional, Sequence

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
        df["Maximum in I"] = d["has_maximum"].map(lambda v: "yes" if bool(v) else "no")
    if "has_kratky_maximum" in d:
        df["Maximum in I q^2 (knee)"] = d["has_kratky_maximum"].map(lambda v: "yes" if bool(v) else "no")
    df = df.reset_index(drop=True)
    for col, mapping in (extra or {}).items():
        df[col] = [mapping.get(s) for s in df["Sample"]]
    return df


def workflow_sheets(result, extra: Optional[Dict[str, Dict[str, object]]] = None,
                    descriptors: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """The sheets of a :class:`saxs.core.fit_workflow.WorkflowResult`: summary (one row per curve), models (every
    candidate with its verdict), parameters (long format), robustness (when run) and the model-free descriptors.
    ``extra``: ``{curve: {column: value}}`` columns placed first in the summary."""
    from .fit_workflow import comparison_table, parameters_table, summary_table

    sheets = {"summary": summary_table(result, extra), "models": comparison_table(result), "parameters": parameters_table(result)}
    if result.robustness is not None and len(result.robustness):
        sheets["robustness"] = result.robustness
    if descriptors is not None and len(descriptors):
        sheets["model-free"] = modelfree_sheet(descriptors)
    return sheets


def write_workflow_tables(result, base_path: str, extra: Optional[Dict[str, Dict[str, object]]] = None,
                          descriptors: Optional[pd.DataFrame] = None, formats: Sequence[str] = ("xlsx", "csv", "md")) -> Dict[str, str]:
    """``base_path.xlsx`` with every sheet + notes, ``.csv`` / ``.md`` of the summary, and one csv per further sheet
    (``base_path_models.csv`` ...)."""
    from .fit_workflow import method_notes

    sheets = workflow_sheets(result, extra, descriptors)
    notes = method_notes(result)
    out = write_tables(sheets, base_path, notes=notes, md_sheet="summary", formats=formats)
    if "csv" in formats:
        for name, df in sheets.items():
            if name == "summary":
                continue
            path = f"{base_path}_{name.replace('-', '_')}.csv"
            transliterate_frame(df).to_csv(path, index=False, encoding="utf-8-sig")
            out[f"csv:{name}"] = path
    return out
