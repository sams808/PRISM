# io_universal_v40.py
# --------------------------------------------------------------------------------------
# Universal loader (V40): robust sniffers + parsers + registry
# Supports:
#   1) TA SDT Q600 ASCII (DSC-TGA) with SigN + StartOfData
#   2) SAXS EDF ASCII exports (header with '#', then 'q(A-1)  I(q)  Sig(q)')
#   3) Raman XY (Raman shift vs intensity; common ASCII exports)
#   4) XRD XY   (2theta vs intensity; CSV/TSV variants)
#   5) Richer DTA/STA text exports (semicolon/comma/tab; decimal comma aware)
#   6) Generic XY text (spaces/commas/tabs/semicolons; decimal comma; Fortran 'D')
#
# Public API:
#   - load_any(path, *, x_key=None, y_key=None, prefer=None, return_meta=False)
#   - import_xy(path, *, x_key=None, y_key=None, deduplicate=None, ...)
#   - list_columns(path)
#   - register_parser(ParserSpec)
#
# Canonical keys exposed in meta["canonical_map"] for robust references:
#   TA SDT: 't_min', 'T_C', 'm_mg', 'HF_mW', 'dT_C', 'dT_uV', 'flow_mL_min'
#   SAXS  : 'q_A^-1', 'I', 'sigma'
#   Raman : 'shift_cm^-1', 'intensity'
#   XRD   : '2theta_deg', 'intensity'
#   DTA   : 'T_C', 'time_min', 'HF_mW', 'DSC_mW_mg', 'TG_pct', 'DTG_pct_min'
#
from __future__ import annotations

import io
import os
import re
import typing as _t
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ==============================
# Low-level numeric utilities
# ==============================
_DEC_COMMA = re.compile(r"(?<=\d),(?=\d)")
_FORTRAN_D = re.compile(r"([+-]?\d+(?:\.\d+)?)[dD]([+-]?\d+)")
_MULTI_SEP  = re.compile(r"[,\t; ]+")
_NUM_LIKE   = re.compile(r"""^[\s]*
    [+-]?                 # sign
    (?:\d+\.?\d*|\.\d+)   # '12', '12.3', '.004'
    (?:[eEdD][+-]?\d+)?   # exponent
    [\s]*$""", re.X)

def _read_text_with_fallbacks(path: str, encodings=("utf-8-sig","utf-8","latin-1")) -> tuple[str,str]:
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read(), enc
        except Exception as e:
            last_err = e
    # last resort: binary → utf-8 ignore
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore"), "binary->utf-8(ignore)"

def _normalize_num_token(tok: str) -> str | None:
    s = tok.strip()
    if not _NUM_LIKE.match(s):
        return None
    s = _DEC_COMMA.sub(".", s)                               # 12,34 -> 12.34
    s = _FORTRAN_D.sub(lambda m: f"{m.group(1)}E{m.group(2)}", s)  # 1.2D+3 -> 1.2E+3
    if s.startswith(".") and (len(s) == 1 or s[1].isdigit()):      # .004 -> 0.004
        s = "0" + s
    return s

def _split_numeric_line(line: str) -> list[str] | None:
    parts = _MULTI_SEP.split(line.strip())
    if len(parts) < 2:
        return None
    t0 = _normalize_num_token(parts[0])
    t1 = _normalize_num_token(parts[1])
    if t0 is None or t1 is None:
        return None
    parts[0], parts[1] = t0, t1
    return parts

def _coerce_2cols_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df.shape[1] < 2:
        raise ValueError("Found data, but fewer than two usable columns.")
    x = df.iloc[:,0].astype(float).to_numpy()
    y = df.iloc[:,1].astype(float).to_numpy()
    if x.size != y.size or x.size < 2:
        raise ValueError("X or Y length is invalid.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NaN/Inf found in X or Y.")
    return x, y

def _stable_sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray,bool]:
    sorted_now = bool(np.all(np.diff(x) > 0))
    if sorted_now:
        return x, y, True
    order = np.argsort(x, kind="mergesort")
    return x[order], y[order], False

def _deduplicate_xy(x: np.ndarray, y: np.ndarray, how: str, dec: int) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(x, dec)
    uniq, inv = np.unique(key, return_inverse=True)

    if how == "first":
        keep_idx = np.zeros_like(uniq, dtype=int)
        seen = np.zeros_like(uniq, dtype=bool)
        for i,g in enumerate(inv):
            if not seen[g]:
                keep_idx[g] = i
                seen[g] = True
        return uniq, y[keep_idx]
    if how == "last":
        keep_idx = np.zeros_like(uniq, dtype=int)
        seen = np.zeros_like(uniq, dtype=bool)
        for i in range(inv.size-1, -1, -1):
            g = inv[i]
            if not seen[g]:
                keep_idx[g] = i
                seen[g] = True
        return uniq, y[keep_idx]
    if how == "mean":
        sums = np.zeros_like(uniq, dtype=float)
        counts = np.zeros_like(uniq, dtype=int)
        for val, g in zip(y, inv):
            sums[g] += val
            counts[g] += 1
        return uniq, sums / np.maximum(counts, 1)
    raise ValueError("Invalid deduplicate method. Use 'first'|'last'|'mean'.")

# ==============================
# Parser registry
# ==============================
@dataclass
class ParserSpec:
    name: str
    sniff: _t.Callable[[str, str], bool]                      # (path, head_text) -> bool
    parse: _t.Callable[[str], tuple[pd.DataFrame, dict]]      # path -> (df, meta)
    priority: int = 100

_REGISTRY: list[ParserSpec] = []

def register_parser(spec: ParserSpec) -> None:
    _REGISTRY.append(spec)
    _REGISTRY.sort(key=lambda s: s.priority)

def _head_text(path: str, n_lines: int = 160) -> str:
    text, _ = _read_text_with_fallbacks(path)
    return "\n".join(text.splitlines()[:n_lines])

# ==============================
# Sniffers
# ==============================
def sniff_ta_sdt(path: str, head: str) -> bool:
    if "StartOfData" in head and re.search(r"^Sig\d+\s+", head, re.M) and re.search(r"^Nsig\s+\d+", head, re.M):
        return True
    if "StartOfData" in head and "Exotherm" in head and re.search(r"^Sig\d+\s+", head, re.M):
        return True
    return False

def sniff_saxs_edf_ascii(path: str, head: str) -> bool:
    if re.search(r"^#\s*EDF_DataBlockID", head, re.M) and re.search(r"^\s*q\(A-1\)\s+I\(q\)\s+Sig\(q\)", head, re.M):
        return True
    if re.search(r"^\s*q\s*\(A-?1\)\s+I\(q\)\s+Sig\(q\)", head, re.M):
        return True
    return False

def sniff_generic_xy(path: str, head: str) -> bool:
    lines = head.splitlines()
    numeric_count = 0
    for s in lines:
        s = s.strip()
        if not s or s.startswith("#") or s.lower().startswith(("version","language","run","sig1","sig2","nsig","startofdata")):
            continue
        tokens = _split_numeric_line(s)
        if tokens is not None:
            numeric_count += 1
        if numeric_count >= 5:
            return True
    return False

def sniff_raman(path: str, head: str) -> bool:
    low = head.lower()
    if "raman" in low and ("shift" in low or "cm-1" in low or "cm^" in low):
        return True
    if re.search(r"raman\s*shift", head, re.I):
        return True
    # simple two-column numeric tables with Raman-ish headers
    for ln in head.splitlines()[:20]:
        if re.search(r"raman\s*shift" , ln, re.I) and re.search(r"intensity|counts|a\.u", ln, re.I):
            return True
    return False

def sniff_xrd(path: str, head: str) -> bool:
    low = head.lower()
    if "xrd" in low and ("2theta" in low or "two theta" in low):
        return True
    for ln in head.splitlines()[:30]:
        if re.search(r"2\s*theta|two\s*theta", ln, re.I) and re.search(r"intensity|counts", ln, re.I):
            return True
    return False

def sniff_dta_table(path: str, head: str) -> bool:
    low = head.lower()
    if "netzsch" in low or "ta instruments" in low or "universal analysis" in low:
        return True
    if re.search(r"dsc|heat\s*flow|tg|dtg", low):
        return True
    if ";" in head and re.search(r"temperature", low):
        return True
    return False

# ==============================
# Parsers
# ==============================
def parse_ta_sdt(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_enc = _read_text_with_fallbacks(path)
    lines = text.splitlines()

    header_lines, data_lines, in_data = [], [], False
    for ln in lines:
        if not in_data:
            header_lines.append(ln)
            if ln.strip().lower().startswith("startofdata"):
                in_data = True
            continue
        s = ln.strip()
        if s:
            data_lines.append(s)

    header = "\n".join(header_lines)

    # SigN → column names
    sig_map: dict[int,str] = {}
    for m in re.finditer(r"^Sig(\d+)\s+(.+)$", header, re.M):
        sig_map[int(m.group(1))] = m.group(2).strip()

    cleaned_rows = []
    for raw in data_lines:
        parts = _MULTI_SEP.split(raw.strip())
        norm, ok = [], True
        for p in parts:
            p2 = _normalize_num_token(p)
            if p2 is None:
                ok = False; break
            norm.append(p2)
        if ok and len(norm) >= 2:
            cleaned_rows.append(" ".join(norm))
    if not cleaned_rows:
        raise ValueError("TA SDT: no numeric rows after StartOfData.")

    df = pd.read_csv(io.StringIO("\n".join(cleaned_rows)), sep=r"\s+", engine="python", header=None)
    # assign names
    ncol = df.shape[1]
    colnames = [sig_map.get(i, f"Sig{i}") for i in range(1, ncol+1)]
    df.columns = colnames

    # canonical map
    canonical = {}
    for col in df.columns:
        key = col.lower()
        if "time" in key and "(min" in key:
            canonical["t_min"] = col
        elif "temperature" in key and "difference" in key and "µv" in key:
            canonical["dT_uV"] = col
        elif "temperature" in key and "difference" in key:
            canonical["dT_C"] = col
        elif key.startswith("temperature"):
            canonical["T_C"] = col
        elif "weight" in key or "mass" in key:
            canonical["m_mg"] = col
        elif "heat flow" in key or "heatflow" in key:
            canonical["HF_mW"] = col
        elif "purge" in key and ("flow" in key or "ml/min" in key):
            canonical["flow_mL_min"] = col

    x_col = canonical.get("T_C") or canonical.get("t_min") or df.columns[0]
    y_col = canonical.get("HF_mW") or canonical.get("dT_C") or canonical.get("dT_uV") or canonical.get("m_mg") or df.columns[1]
    canonical["X"] = x_col
    canonical["Y"] = y_col

    meta = {
        "parser": "ta_sdt",
        "used_encoding": used_enc,
        "signals": list(df.columns),
        "canonical_map": canonical,
        "raw_header": header,
        "path": os.path.abspath(path),
    }
    return df, meta

def parse_saxs_edf_ascii(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_enc = _read_text_with_fallbacks(path)
    lines = text.splitlines()

    header = []
    header_columns_line = None
    data_rows: list[str] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(s); continue
        if header_columns_line is None:
            if re.search(r"q\s*\(A-?1\)", s) and "I(q)" in s:
                header_columns_line = s
                continue
        data_rows.append(s)

    if header_columns_line is None:
        # fallback: treat first non-# non-numeric as header row
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s or s.startswith("#"): continue
            parts = _MULTI_SEP.split(s)
            if len(parts) >= 2 and (_normalize_num_token(parts[0]) is None or _normalize_num_token(parts[1]) is None):
                header_columns_line = s
                data_rows = [l.strip() for l in lines[i+1:] if l.strip() and not l.strip().startswith("#")]
                break
    if header_columns_line is None:
        raise ValueError("SAXS EDF ASCII: Missing column header line (e.g., 'q(A-1) I(q) Sig(q)').")

    colnames = _MULTI_SEP.split(header_columns_line.strip())
    cleaned_rows = []
    for raw in data_rows:
        parts = _MULTI_SEP.split(raw.strip())
        norm, ok = [], True
        for p in parts:
            p2 = _normalize_num_token(p)
            if p2 is None:
                ok = False; break
            norm.append(p2)
        if ok and len(norm) == len(colnames):
            cleaned_rows.append(" ".join(norm))
    if not cleaned_rows:
        raise ValueError("SAXS EDF ASCII: no numeric data rows found.")

    df = pd.read_csv(io.StringIO("\n".join(cleaned_rows)), sep=r"\s+", engine="python", header=None)
    df.columns = colnames

    canonical = {}
    for col in df.columns:
        low = col.lower()
        if low.startswith("q"):
            canonical["q_A^-1"] = col
        elif "sig" in low:
            canonical["sigma"] = col
        else:
            canonical["I"] = col

    canonical.setdefault("X", canonical.get("q_A^-1"))
    canonical.setdefault("Y", canonical.get("I"))

    meta = {
        "parser": "saxs_edf_ascii",
        "used_encoding": used_enc,
        "raw_header": "\n".join(header),
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df, meta

def parse_generic_xy(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_enc = _read_text_with_fallbacks(path)
    raw_lines = text.splitlines()

    # gather candidate lines (text vs data)
    data_candidates = []
    for s in raw_lines:
        st = s.strip()
        if not st:
            continue
        if st.startswith(("#",";","//","%")):
            continue
        toks = _split_numeric_line(st)
        if toks is not None:
            data_candidates.append(("data", st, toks))
        else:
            data_candidates.append(("text", st, None))

    # first streak of >= 5 consecutive numeric lines
    start_idx, streak = None, 0
    for i, (kind, st, toks) in enumerate(data_candidates):
        if kind == "data":
            streak += 1
            if streak >= 5:
                start_idx = i - streak + 1
                break
        else:
            streak = 0
    if start_idx is None:
        # fallback: accept if >= 2 data lines exist
        numeric_total = sum(1 for k,_,_ in data_candidates if k == "data")
        if numeric_total < 2:
            raise ValueError("Generic XY: No numeric XY data detected in this file.")
        for i,(k, st, toks) in enumerate(data_candidates):
            if k == "data":
                start_idx = i; break

    header_row = None
    if start_idx and start_idx > 0:
        prev_kind, prev_st, _ = data_candidates[start_idx-1]
        if prev_kind == "text":
            header_row = prev_st

    cleaned_rows = []
    for kind, st, toks in data_candidates[start_idx:]:
        if kind != "data":
            continue
        parts = _MULTI_SEP.split(st.strip())
        norm = []
        for p in parts:
            p2 = _normalize_num_token(p)
            if p2 is None:
                break
            norm.append(p2)
        if len(norm) >= 2:
            cleaned_rows.append(" ".join(norm))
    if not cleaned_rows:
        raise ValueError("Generic XY: Could not build a numeric table from detected data rows.")

    df = pd.read_csv(io.StringIO("\n".join(cleaned_rows)), sep=r"\s+", engine="python", header=None)

    if header_row is not None:
        head_tokens = _MULTI_SEP.split(header_row.strip())
        if len(head_tokens) >= df.shape[1] and all(_normalize_num_token(t) is None for t in head_tokens[:2]):
            df.columns = head_tokens[:df.shape[1]]
        else:
            df.columns = [f"col{i+1}" for i in range(df.shape[1])]
    else:
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]

    meta = {
        "parser": "generic_xy",
        "used_encoding": used_enc,
        "canonical_map": {"X": df.columns[0], "Y": df.columns[1]},
        "path": os.path.abspath(path),
    }
    return df, meta

def _read_table_flexible(path: str) -> tuple[pd.DataFrame, str, str | None]:
    text, enc = _read_text_with_fallbacks(path)
    decimal = "," if len(re.findall(r"\d+,\d", text[:2000])) > len(re.findall(r"\d+\.\d", text[:2000])) else "."
    for sep in [";", "\t", ",", " ", None]:
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python")
            if df.shape[1] >= 2:
                return df, enc, sep
        except Exception:
            continue
    raise ValueError("Could not read table with flexible settings.")

def _canonicalize_xy(df: pd.DataFrame, prefer: list[str]) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in prefer:
        low = cand.lower()
        for key, col in cols.items():
            if low in key:
                for other_key, other_col in cols.items():
                    if other_col == col:
                        continue
                    if any(y in other_key for y in ["intensity","counts","absorb","a.u","signal","heat flow","dsc","tg","dtg"]):
                        return col, other_col
    # fallback: first two
    return df.columns[0], df.columns[1]

def parse_raman(path: str) -> tuple[pd.DataFrame, dict]:
    df, enc, sep = _read_table_flexible(path)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.columns.size < 2:
        raise ValueError("Raman parser needs at least two columns.")

    # detect header row as text if first row non-numeric
    first_row = df.iloc[0]
    if not pd.to_numeric(first_row, errors="coerce").notna().all():
        df.columns = first_row
        df = df.iloc[1:]
    df.columns = [str(c).strip() for c in df.columns]
    shift_col, intensity_col = None, None
    for col in df.columns:
        low = col.lower()
        if shift_col is None and ("raman" in low or "shift" in low or "cm" in low):
            shift_col = col
        elif intensity_col is None and re.search(r"intensity|counts|a\.u", low):
            intensity_col = col
    if shift_col is None or intensity_col is None:
        shift_col, intensity_col = df.columns[0], df.columns[1]

    canonical = {
        "shift_cm^-1": shift_col,
        "intensity": intensity_col,
        "X": shift_col,
        "Y": intensity_col,
    }
    meta = {
        "parser": "raman_xy",
        "used_encoding": enc,
        "sep": sep,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df.reset_index(drop=True), meta

def parse_xrd(path: str) -> tuple[pd.DataFrame, dict]:
    df, enc, sep = _read_table_flexible(path)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.columns.size < 2:
        raise ValueError("XRD parser needs at least two columns.")

    first_row = df.iloc[0]
    if not pd.to_numeric(first_row, errors="coerce").notna().all():
        df.columns = first_row
        df = df.iloc[1:]
    df.columns = [str(c).strip() for c in df.columns]
    two_theta, intensity = None, None
    d_spacing = None
    for col in df.columns:
        low = col.lower()
        if two_theta is None and re.search(r"2\s*theta|two\s*theta|2θ", low):
            two_theta = col
        elif intensity is None and re.search(r"intensity|counts", low):
            intensity = col
        elif d_spacing is None and re.search(r"\bd\b|spacing", low):
            d_spacing = col
    if two_theta is None or intensity is None:
        two_theta, intensity = df.columns[0], df.columns[1]

    canonical = {
        "2theta_deg": two_theta,
        "intensity": intensity,
        "X": two_theta,
        "Y": intensity,
    }
    if d_spacing:
        canonical["d_A"] = d_spacing
    meta = {
        "parser": "xrd_xy",
        "used_encoding": enc,
        "sep": sep,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df.reset_index(drop=True), meta

def parse_dta_table(path: str) -> tuple[pd.DataFrame, dict]:
    df, enc, sep = _read_table_flexible(path)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    first_row = df.iloc[0]
    if not pd.to_numeric(first_row, errors="coerce").notna().all():
        df.columns = first_row
        df = df.iloc[1:]
    df.columns = [str(c).strip() for c in df.columns]

    canonical = {}
    for col in df.columns:
        low = col.lower()
        if re.search(r"temp|t\s*\(\s*°?c\s*\)|°c| c\b", low):
            canonical.setdefault("T_C", col)
        if re.search(r"time|min", low):
            canonical.setdefault("time_min", col)
        if "heat" in low or "flow" in low or "dsc" in low:
            if "/" in low or "mw" in low:
                canonical.setdefault("DSC_mW_mg", col)
            else:
                canonical.setdefault("HF_mW", col)
        if re.search(r"tg|mass|weight", low) and "%" in low:
            canonical.setdefault("TG_pct", col)
        if re.search(r"dtg", low) or "%/min" in low:
            canonical.setdefault("DTG_pct_min", col)

    x_col, y_col = None, None
    if "T_C" in canonical and ("HF_mW" in canonical or "DSC_mW_mg" in canonical):
        x_col = canonical.get("T_C")
        y_col = canonical.get("DSC_mW_mg", canonical.get("HF_mW"))
    elif "time_min" in canonical and ("HF_mW" in canonical or "DSC_mW_mg" in canonical):
        x_col = canonical.get("time_min")
        y_col = canonical.get("DSC_mW_mg", canonical.get("HF_mW"))
    elif "T_C" in canonical and "TG_pct" in canonical:
        x_col = canonical.get("T_C")
        y_col = canonical.get("TG_pct")
    else:
        x_col, y_col = _canonicalize_xy(df, list(canonical.values()) if canonical else [])

    canonical["X"] = x_col
    canonical["Y"] = y_col

    meta = {
        "parser": "dta_table",
        "used_encoding": enc,
        "sep": sep,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df.reset_index(drop=True), meta

# ==============================
# Register parsers (priority)
# ==============================
register_parser(ParserSpec("ta_sdt", sniff_ta_sdt, parse_ta_sdt, priority=10))
register_parser(ParserSpec("saxs_edf_ascii", sniff_saxs_edf_ascii, parse_saxs_edf_ascii, priority=20))
register_parser(ParserSpec("dta_table", sniff_dta_table, parse_dta_table, priority=30))
register_parser(ParserSpec("raman_xy", sniff_raman, parse_raman, priority=40))
register_parser(ParserSpec("xrd_xy", sniff_xrd, parse_xrd, priority=50))
register_parser(ParserSpec("generic_xy", sniff_generic_xy, parse_generic_xy, priority=90))

# ==============================
# Public API
# ==============================
def load_any(path: str, *, x_key: str | None = None, y_key: str | None = None,
             prefer: str | None = None, return_meta: bool = False):
    """
    Load any supported file. If x_key & y_key are provided, return (x, y[, meta]) arrays
    for backward-compat, otherwise return (DataFrame[, meta]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    head = _head_text(path, n_lines=160)

    chosen = None
    if prefer:
        chosen = next((spec for spec in _REGISTRY if spec.name == prefer), None)
        if chosen is None:
            raise ValueError(f"Unknown parser requested via 'prefer': {prefer}")
    if chosen is None:
        for spec in _REGISTRY:
            try:
                if spec.sniff(path, head):
                    chosen = spec; break
            except Exception:
                continue
    if chosen is None:
        chosen = next((s for s in _REGISTRY if s.name == "generic_xy"), None)
        if chosen is None:
            raise RuntimeError("No parsers registered.")

    df, meta = chosen.parse(path)
    meta["selected_parser"] = chosen.name

    if x_key is not None and y_key is not None:
        if x_key not in df.columns or y_key not in df.columns:
            raise KeyError(f"x_key/y_key not found. Available columns: {list(df.columns)}")
        x = df[x_key].astype(float).to_numpy()
        y = df[y_key].astype(float).to_numpy()
        x, y, _ = _stable_sort_xy(x, y)
        return (x, y, meta) if return_meta else (x, y)

    return (df, meta) if return_meta else df

def import_xy(path: str, *, x_key: str | None = None, y_key: str | None = None,
              deduplicate: str | None = None, dedup_round_decimals: int = 12,
              force_ascending: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Backward-compatible helper: always returns (x, y, meta).
    If no x_key/y_key are given, the first two numeric columns are used.
    """
    df, meta = load_any(path, return_meta=True)

    if x_key is None or y_key is None:
        x, y = _coerce_2cols_to_xy(df)
        if force_ascending:
            x, y, _ = _stable_sort_xy(x, y)
        if deduplicate is not None:
            x, y = _deduplicate_xy(x, y, deduplicate, dedup_round_decimals)
        return x, y, meta

    if x_key not in df.columns or y_key not in df.columns:
        raise KeyError(f"x_key/y_key not found. Available: {list(df.columns)}")
    x = df[x_key].astype(float).to_numpy()
    y = df[y_key].astype(float).to_numpy()
    if force_ascending:
        x, y, _ = _stable_sort_xy(x, y)
    if deduplicate is not None:
        x, y = _deduplicate_xy(x, y, deduplicate, dedup_round_decimals)
    return x, y, meta

def list_columns(path: str) -> list[str]:
    df, _ = load_any(path, return_meta=True)
    return list(df.columns)
