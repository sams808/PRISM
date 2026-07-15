"""
rruff_science.py — RRUFF mineral Raman database ingest (M10), framework-
agnostic (no Qt/Tkinter dependency — the parallel-track milestone the plan
calls out as startable independent of the Qt rewrite).

Format confirmed directly against the real, current rruff.net bulk download
(https://www.rruff.net/about/download-data/ ->
https://www.rruff.net/zipped_data_files/raman/, 2026-07-15) rather than
assumed from memory — the site was reportedly updated, so this is a fresh
spike, not carried over from an old format:

Filename (double-underscore-delimited, extension .txt):
    {Mineral}__{RRUFF_ID}__Raman__{wavelength_nm}__{orientation_deg|''}__
        {polarization|''}__{Raman_Data_Processed|Raman_Data_RAW}__{hash}.txt
Fields are positional but several are legitimately empty for unoriented
samples (e.g. "Abramovite__R070037__Raman__785______Raman_Data_RAW__<hash>.txt"
has two empty fields between wavelength and data-kind).

In-file content: a JCAMP-like ##KEY=VALUE metadata header (keys vary per
sample — e.g. unoriented/unconfirmed samples may lack ##ORIENTATION,
##MEASURED CHEMISTRY, ##PIN_ID entirely — so the header is parsed as an
open-ended dict, not a fixed schema), then a blank line, then comma-
separated "raman_shift, intensity" data rows.

Multiple laser excitation wavelengths are used across the database (514,
532, 633, 780, 785 nm and others seen during ingest) — per the user's own
explicit requirement, this module never auto-labels a match; it only
prepares data for an assisted matching UI (M12) that always shows the
candidate's excitation wavelength and requires explicit user confirmation.

Citation (confirmed from rruff.net/about/): Lafuente, B., Downs, R. T.,
Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project.
Highlights in Mineralogical Crystallography, 1-30. RRUFF also asks that
per-sample external contributors (the ##OWNER/##SOURCE header fields) be
acknowledged "when applicable" — surfaced to the user via RRUFF_CITATION /
RRUFF_ATTRIBUTION_NOTE below (M12's UI should display these, not just this
module quietly knowing them).
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

RRUFF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "rruff")
RRUFF_INDEX_PATH = os.path.join(RRUFF_CACHE_DIR, "index.json")
RRUFF_RAW_DIR = os.path.join(RRUFF_CACHE_DIR, "raw")

RRUFF_CITATION = (
    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). "
    "The power of databases: the RRUFF project. "
    "Highlights in Mineralogical Crystallography, 1-30."
)
RRUFF_ATTRIBUTION_NOTE = (
    "Please also acknowledge the individual sample owner/source shown for "
    "each RRUFF match, when applicable (see that sample's OWNER/SOURCE "
    "fields), per RRUFF's stated attribution request."
)


@dataclass
class RruffSpectrum:
    mineral: str
    rruff_id: str
    wavelength_nm: Optional[float]
    orientation_deg: Optional[str]
    polarization: Optional[str]
    data_kind: str  # "Processed" | "RAW" | ""
    x: np.ndarray
    y: np.ndarray
    ideal_chemistry: Optional[str] = None
    locality: Optional[str] = None
    owner: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    orientation_text: Optional[str] = None
    cell_parameters: Optional[str] = None
    header: Dict[str, str] = field(default_factory=dict)
    source_filename: str = ""


def parse_rruff_filename(filename: str) -> Dict[str, Any]:
    """Split a RRUFF bulk-download filename into its positional fields.
    Returns {} if the filename doesn't match the expected double-underscore
    schema (e.g. a directory entry or an unrelated file some archives
    occasionally bundle) rather than raising — callers should skip those.
    """
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    parts = stem.split("__")
    if len(parts) < 7:
        return {}
    mineral, rruff_id, literal_raman = parts[0], parts[1], parts[2]
    if literal_raman != "Raman":
        return {}
    wavelength_txt, orientation_deg, polarization = parts[3], parts[4], parts[5]
    data_kind_field = parts[6]
    data_kind = data_kind_field.replace("Raman_Data_", "") if data_kind_field.startswith("Raman_Data_") else data_kind_field
    try:
        wavelength_nm = float(wavelength_txt) if wavelength_txt else None
    except ValueError:
        wavelength_nm = None
    return {
        "mineral": mineral,
        "rruff_id": rruff_id,
        "wavelength_nm": wavelength_nm,
        "orientation_deg": orientation_deg or None,
        "polarization": polarization or None,
        "data_kind": data_kind,
    }


def parse_rruff_txt(text: str, source_filename: str = "") -> RruffSpectrum:
    """Parse one RRUFF .txt file's content (## header + comma-separated
    x,y data) into a RruffSpectrum. Header keys are kept generically in
    `.header`; a handful of commonly-present ones are also promoted to
    named fields for convenience, but nothing is assumed mandatory —
    confirmed directly against real files that omit ORIENTATION,
    MEASURED CHEMISTRY, and PIN_ID entirely for some samples."""
    header: Dict[str, str] = {}
    xs: List[float] = []
    ys: List[float] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            key, _, value = line[2:].partition("=")
            header[key.strip()] = value.strip()
            continue
        if "," in line:
            left, _, right = line.partition(",")
            try:
                xs.append(float(left.strip()))
                ys.append(float(right.strip()))
            except ValueError:
                continue

    filename_fields = parse_rruff_filename(source_filename) if source_filename else {}
    # Header first, filename as fallback only — confirmed necessary on the
    # real database, not just theoretical: Gysinite-(Ce) R250121's filename
    # says wavelength "53" (a genuine truncated-digit typo in that one
    # filename) while its own ##RAMAN WAVELENGTH header correctly says 532.
    # The structured header is curated metadata; the filename is a derived,
    # manually-typeable string, so it's the less trustworthy of the two —
    # same direction already used for mineral/rruff_id below.
    wavelength_nm = None
    if "RAMAN WAVELENGTH" in header:
        try:
            wavelength_nm = float(header["RAMAN WAVELENGTH"])
        except ValueError:
            wavelength_nm = None
    if wavelength_nm is None:
        wavelength_nm = filename_fields.get("wavelength_nm")

    data_kind = filename_fields.get("data_kind", "")
    if not data_kind and "FILETYPE" in header:
        data_kind = header["FILETYPE"].replace("Raman ", "")

    return RruffSpectrum(
        mineral=header.get("NAMES") or filename_fields.get("mineral", ""),
        rruff_id=header.get("RRUFFID") or filename_fields.get("rruff_id", ""),
        wavelength_nm=wavelength_nm,
        orientation_deg=filename_fields.get("orientation_deg"),
        polarization=filename_fields.get("polarization"),
        data_kind=data_kind,
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        ideal_chemistry=header.get("IDEAL CHEMISTRY"),
        locality=header.get("LOCALITY"),
        owner=header.get("OWNER"),
        source=header.get("SOURCE"),
        url=header.get("URL"),
        orientation_text=header.get("ORIENTATION"),
        cell_parameters=header.get("CELL PARAMETERS"),
        header=header,
        source_filename=source_filename,
    )


def ingest_zip(
    zip_path: str, *, raw_dir: str = RRUFF_RAW_DIR, category: str = "",
    max_peaks: int = 15, progress_every: int = 500,
) -> List[Dict[str, Any]]:
    """Extract every .txt in `zip_path`, parse it, pre-extract peak
    candidates (fitting_science.find_peak_candidates — the same utility
    M8's "Auto-find peaks" uses, per the plan's reuse principle), and
    return one lightweight index record per spectrum. Only "Processed"
    spectra get peak-extracted (RAW files are kept for reference/overlay
    but a baseline-uncorrected RAW curve isn't a meaningful peak-finder
    input) — both kinds are indexed either way.

    Raw text is written under `raw_dir` for later full-spectrum overlay;
    the returned records are what actually goes into the searchable index.
    """
    from fitting_science import find_peak_candidates

    os.makedirs(raw_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        for i, name in enumerate(names):
            if progress_every and i and i % progress_every == 0:
                print(f"  ...{i}/{len(names)} in {os.path.basename(zip_path)}")
            try:
                raw_bytes = zf.read(name)
                text = raw_bytes.decode("utf-8", errors="replace")
            except (OSError, zipfile.BadZipFile):
                continue

            spectrum = parse_rruff_txt(text, source_filename=name)
            if len(spectrum.x) < 5 or not spectrum.mineral:
                continue

            out_path = os.path.join(raw_dir, name)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError:
                out_path = ""

            peaks: List[float] = []
            if spectrum.data_kind.lower().startswith("process"):
                try:
                    peaks = find_peak_candidates(spectrum.x, spectrum.y, max_peaks=max_peaks)
                except Exception:
                    peaks = []

            records.append({
                "mineral": spectrum.mineral,
                "rruff_id": spectrum.rruff_id,
                "wavelength_nm": spectrum.wavelength_nm,
                "orientation_deg": spectrum.orientation_deg,
                "polarization": spectrum.polarization,
                "data_kind": spectrum.data_kind,
                "ideal_chemistry": spectrum.ideal_chemistry,
                "locality": spectrum.locality,
                "owner": spectrum.owner,
                "source": spectrum.source,
                "url": spectrum.url,
                "peaks": peaks,
                "raw_path": out_path,
                "category": category,
                "x_min": float(spectrum.x.min()) if len(spectrum.x) else None,
                "x_max": float(spectrum.x.max()) if len(spectrum.x) else None,
            })

    return records


def build_index(zip_paths_with_categories: List[tuple], *, cache_dir: str = RRUFF_CACHE_DIR) -> int:
    """Ingest a list of (zip_path, category_label) pairs and write one
    consolidated index.json under cache_dir. Returns the total record
    count. Existing raw files/index are overwritten by a full rebuild —
    this is meant to be re-run wholesale when refreshing the database, not
    incrementally patched."""
    os.makedirs(cache_dir, exist_ok=True)
    raw_dir = os.path.join(cache_dir, "raw")
    all_records: List[Dict[str, Any]] = []
    for zip_path, category in zip_paths_with_categories:
        print(f"Ingesting {zip_path} (category={category})...")
        records = ingest_zip(zip_path, raw_dir=raw_dir, category=category)
        print(f"  -> {len(records)} spectra")
        all_records.extend(records)

    index_path = os.path.join(cache_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f)
    return len(all_records)


def load_index(cache_dir: str = RRUFF_CACHE_DIR) -> List[Dict[str, Any]]:
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small diagnostic summary — mineral/wavelength coverage — useful both
    for a one-off sanity check after ingest and for a future "database
    info" panel in the match-assist UI (M12)."""
    minerals = {r["mineral"] for r in records if r.get("mineral")}
    wavelengths = sorted({r["wavelength_nm"] for r in records if r.get("wavelength_nm")})
    return {
        "n_spectra": len(records),
        "n_minerals": len(minerals),
        "wavelengths_nm": wavelengths,
    }


# =============================================================================
# Match-assist scoring (M12) — ranks candidates, NEVER auto-labels. Per the
# user's explicit original requirement: RRUFF spectra span many laser
# excitation wavelengths, which affects relative peak intensities/fluorescence
# even though Raman shift (cm^-1) is nominally wavelength-independent — so
# every candidate must show its own wavelength, and the user always makes
# the final call. This module only ranks; qt_rruff.py's UI is what actually
# requires an explicit user accept/reject before anything gets applied.
# =============================================================================

def score_match(query_peaks: List[float], candidate_peaks: List[float], tolerance: float = 10.0) -> Dict[str, float]:
    """How many of the query's peaks have a candidate peak within
    `tolerance` cm^-1. A simple, explainable first-pass metric (peak-count
    overlap), not a claim of definitive identification — deliberately easy
    for a user to sanity-check by eye against the overlay this feeds."""
    if not query_peaks:
        return {"matched": 0, "fraction": 0.0}
    matched = sum(1 for qp in query_peaks if any(abs(qp - cp) <= tolerance for cp in candidate_peaks))
    return {"matched": matched, "fraction": matched / len(query_peaks)}


def rank_rruff_matches(
    query_peaks: List[float], index: List[Dict[str, Any]], *, tolerance: float = 10.0, top_n: int = 25,
) -> List[Dict[str, Any]]:
    """Rank indexed RRUFF spectra by peak-overlap against query_peaks.
    Returns a plain ranked list (each record + matched/fraction fields),
    best match first — never a single "the" answer. Records with zero
    matched peaks, or with no pre-extracted peaks at all (RAW spectra, per
    ingest_zip's Processed-only peak extraction), are excluded."""
    scored = []
    for rec in index:
        candidate_peaks = rec.get("peaks") or []
        if not candidate_peaks:
            continue
        result = score_match(query_peaks, candidate_peaks, tolerance=tolerance)
        if result["matched"] == 0:
            continue
        scored.append({**rec, "matched_peaks": result["matched"], "match_fraction": result["fraction"]})
    scored.sort(key=lambda r: (r["matched_peaks"], r["match_fraction"]), reverse=True)
    return scored[:top_n]
