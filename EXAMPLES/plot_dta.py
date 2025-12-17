# plot_dta.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pick_file_gui() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        p = filedialog.askopenfilename(
            title="Select a TA SDT text export",
            filetypes=[("Text files", "*.txt *.dat *.asc *.csv"), ("All files", "*.*")]
        )
        return Path(p) if p else None
    except Exception:
        return None


def read_text_autodetect(path: Path) -> str:
    """
    TA Instruments exports are often UTF-16 (BOM FF FE).
    This function auto-detects UTF-16 vs UTF-8-ish.
    """
    b = path.read_bytes()

    # BOM checks
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="replace")

    # Heuristic: lots of NUL bytes -> probably UTF-16LE/BE without BOM
    head = b[:4000]
    if head.count(b"\x00") > len(head) * 0.1:
        # Try utf-16 first; if it fails, fallback later
        try:
            return b.decode("utf-16", errors="replace")
        except Exception:
            pass

    # UTF-8 (with possible BOM) fallback, then latin-1 last resort
    try:
        return b.decode("utf-8-sig", errors="replace")
    except Exception:
        return b.decode("latin-1", errors="replace")


def parse_ta_sdt_txt(path: Path) -> Tuple[Dict[str, str], List[str], pd.DataFrame]:
    """
    Parse TA Instruments SDT/Q600-style text export:
    - Header ends at a line containing 'StartOfData' (case-insensitive)
    - Column names are in lines like 'Sig1\tTime (min)', 'Sig2\tTemperature (°C)', ...
    Returns: (header_dict, column_names, dataframe)
    """
    text = read_text_autodetect(path)
    lines = text.splitlines()

    # Find end-of-header marker (robust)
    start_idx = None
    for i, line in enumerate(lines):
        norm = re.sub(r"\s+", "", line).lower()
        if "startofdata" in norm:  # catches "StartOfData", "StartOfData:", etc.
            start_idx = i
            break

    # Fallback: find first numeric row and treat that as start of data
    if start_idx is None:
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() or s[0] in "+-.":
                start_idx = i - 1  # header ends just before first numeric row
                break

    if start_idx is None:
        raise ValueError("Could not find data start (no StartOfData marker and no numeric data rows).")

    header_lines = lines[:start_idx]
    data_lines = lines[start_idx + 1 :]

    # Parse header key/value pairs (best-effort)
    header: Dict[str, str] = {}
    for line in header_lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            key = parts[0].strip()
            val = "\t".join(parts[1:]).strip()
            header.setdefault(key, val)

    # Parse SigN column names (supports "Sig1\tName" and "Sig1 Name")
    sig_map: Dict[int, str] = {}
    sig_re = re.compile(r"^Sig(\d+)\s+(.*)$")
    for line in header_lines:
        m = sig_re.match(line.strip())
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            sig_map[idx] = name

    # Column list
    if sig_map:
        colnames = [sig_map[i] for i in sorted(sig_map.keys())]
    else:
        # Infer from first numeric line
        first_data = next((ln for ln in data_lines if ln.strip()), "")
        ncols = len(re.split(r"\s+", first_data.strip()))
        colnames = [f"col{i+1}" for i in range(ncols)]

    # Keep only numeric-looking data lines
    numeric_rows = []
    for ln in data_lines:
        s = ln.strip()
        if not s:
            continue
        if s[0].isdigit() or s[0] in "+-.":
            numeric_rows.append(s)

    if not numeric_rows:
        raise ValueError("No numeric data rows found after header.")

    # Read whitespace-separated block
    from io import StringIO
    df = pd.read_csv(
        StringIO("\n".join(numeric_rows)),
        sep=r"\s+",
        engine="python",
        names=colnames,
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all").reset_index(drop=True)

    return header, colnames, df


def choose_column(colnames: List[str], prompt: str) -> str:
    print("\nAvailable columns:")
    for i, name in enumerate(colnames):
        print(f"  [{i}] {name}")
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(colnames):
                return colnames[idx]
        matches = [c for c in colnames if raw.lower() in c.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"Ambiguous: matches {matches}. Please enter an index.")
        else:
            print("Invalid choice. Enter an index (e.g. 0) or a unique substring.")


def find_best_column(colnames: List[str], keyword: str) -> Optional[str]:
    hits = [c for c in colnames if keyword.lower() in c.lower()]
    return hits[0] if hits else None


def compute_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.full_like(y, np.nan, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    out[mask] = np.gradient(y[mask], x[mask])  # handles non-uniform spacing
    return out


def main():
    ap = argparse.ArgumentParser(description="Read & plot TA SDT/Q600 DTA/DSC/TGA text exports.")
    ap.add_argument("file", nargs="?", help="Path to .txt export (if omitted, a file dialog may open).")
    args = ap.parse_args()

    path: Optional[Path] = Path(args.file) if args.file else _pick_file_gui()
    if not path:
        raise SystemExit("No file selected/provided.")

    header, colnames, df = parse_ta_sdt_txt(path)

    sample = header.get("Sample", path.stem)
    print(f"\nLoaded: {path.name}")
    print(f"Sample: {sample}")
    print("\nPreview:")
    print(df.head(8).to_string(index=False))

    x_col = choose_column(colnames, "\nChoose X column (index or substring): ")
    y_col = choose_column(colnames, "Choose Y column (index or substring): ")

    print("\nDerivative options:")
    print("  [0] None (plot Y)")
    print("  [1] dY/dt  (derivative vs Time)")
    print("  [2] dY/dT  (derivative vs Temperature)")
    while True:
        dopt = input("Select 0/1/2: ").strip()
        if dopt in {"0", "1", "2"}:
            break

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    y_to_plot = y
    y_label = y_col

    if dopt == "1":
        t_col = find_best_column(colnames, "Time") or choose_column(colnames, "Pick Time column for dY/dt: ")
        t = df[t_col].to_numpy(dtype=float)
        y_to_plot = compute_derivative(y, t)
        y_label = f"d({y_col})/d({t_col})"
        print(f"Using '{t_col}' for derivative basis.")
    elif dopt == "2":
        T_col = find_best_column(colnames, "Temperature") or choose_column(colnames, "Pick Temperature column for dY/dT: ")
        T = df[T_col].to_numpy(dtype=float)
        y_to_plot = compute_derivative(y, T)
        y_label = f"d({y_col})/d({T_col})"
        print(f"Using '{T_col}' for derivative basis.")

    plt.figure()
    plt.plot(x, y_to_plot)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(sample)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

