"""Frame inventory and session pairing for 2D SAXS frames (EDF).

Scans one or more folders of raw frames, reads each header, classifies the frame (sample, empty,
calibrant, other) from regular expressions, refines the beam centre from the recorded direct beam
and records per-frame diagnostics (centre offset from the header, direct-beam level, saturation,
capillary-wall lobes, slit streaks). Then pairs every sample with the empty of the same session and
configuration. Nothing here knows about a particular sample series: the sample identity is the file
stem (or the header comment on request) and every rule is an option.

Typical use::

    records = scan_frames(["D:/saxs/2026-01", "D:/saxs/2026-02"])
    pairing = pair_frames(records, rule="same_folder")
    write_inventory(records, "inventory.csv")
    for pair in pairing.pairs: ...

``fabio`` is imported inside :func:`frame_record`; the module itself is pure numpy / pandas.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import edf_sectors as es

Progress = Optional[Callable[[int, int, str], None]]

CAPILLARY_RE = re.compile(r"(?i)(\d+(?:[p.]\d+)?)\s*mm")
COLLIMATION_RE = re.compile(r"Cu-(std|HR|HI)", re.I)
CONFIGURATION_RE = re.compile(r"(ESAXS|SAXS|MAXS|WAXS)", re.I)
DISTANCE_CONFIGURATIONS = {0.9: "SAXS", 1.8: "ESAXS", 0.37: "MAXS", 0.285: "WAXS"}
FRAME_EXTENSIONS = (".edf",)


# ------------------------------------------------------------------------------------ records
@dataclass
class FrameRecord:
    path: str
    rel: str                       # relative to the scanned root, forward slashes
    folder: str                    # immediate parent folder (absolute)
    session: str                   # top-level folder under the root (the root's name at root level)
    stem: str
    name: str                      # sample identity used downstream
    comment_name: str              # header Comment before " - "
    tag: str                       # header Comment after " - "
    collimation: str               # std | HR | HI | prod (from the comment tag)
    configuration: str             # SAXS | ESAXS | MAXS | WAXS | D<dist>
    role: str                      # sample | empty | calibrant | other
    date: str
    exposure_s: float
    distance_m: float
    wavelength_A: float
    pixel_m: float
    shape: Tuple[int, int]
    count_cutoff: float
    cx_header: float
    cy_header: float
    cx: Optional[float] = None     # centre used for the diagnostics (see centre_source)
    cy: Optional[float] = None
    cx_own: Optional[float] = None # this frame's own direct-beam centroid (None when not found)
    cy_own: Optional[float] = None
    centre_source: str = "header"  # own direct beam | session empty (...) | header (...)
    reference_empty: str = ""      # rel path of the empty whose centre was used or checked against
    centre_offset_px: Optional[Tuple[float, float]] = None
    centre_offset_mm: Optional[float] = None
    direct_beam_cps: float = float("nan")      # counts/s inside r < 4 px of the centre used
    direct_beam_cps_r5: float = float("nan")   # the same inside r < 5 px (the brightness test of refine_centres)
    max_cps: float = float("nan")
    saturated: bool = False
    n_negative: int = 0
    n_hot: int = 0
    lobe_q: Optional[float] = None
    lobe_prominence: float = float("nan")
    lobe_ratio: float = float("nan")
    streak_h: float = float("nan")
    streak_v: float = float("nan")
    capillary_mm: Optional[float] = None
    header: Dict[str, str] = field(default_factory=dict)

    @property
    def energy_ev(self) -> float:
        return 12398.42 / self.wavelength_A

    @property
    def datetime(self) -> Optional[datetime]:
        return parse_date(self.date)

    def geometry(self, refined: bool = True) -> es.FrameGeometry:
        cx, cy = (self.cx, self.cy) if (refined and self.cx is not None) else (self.cx_header, self.cy_header)
        return es.FrameGeometry(cx=cx, cy=cy, dist_m=self.distance_m, pixel_m=self.pixel_m,
                                wavelength_A=self.wavelength_A, exposure_s=self.exposure_s)


@dataclass
class FramePair:
    sample: FrameRecord
    empty: FrameRecord

    @property
    def name(self) -> str:
        return self.sample.name


@dataclass
class PairingResult:
    pairs: List[FramePair]
    unpaired: List[FrameRecord]
    rule: str

    def describe(self) -> str:
        lines = [f"{len(self.pairs)} sample(s) paired with the rule '{self.rule}'"]
        for p in self.pairs:
            lines.append(f"  {p.sample.name:24s} <- {p.empty.rel}")
        if self.unpaired:
            lines.append(f"{len(self.unpaired)} unpaired sample(s):")
            lines.extend(f"  {r.name} ({r.rel})" for r in self.unpaired)
        return "\n".join(lines)

    def table(self) -> pd.DataFrame:
        rows = [dict(sample=p.sample.name, sample_frame=p.sample.rel, empty_frame=p.empty.rel,
                     session=p.sample.session, configuration=p.sample.configuration,
                     sample_exposure_s=p.sample.exposure_s, empty_exposure_s=p.empty.exposure_s) for p in self.pairs]
        return pd.DataFrame(rows, columns=["sample", "sample_frame", "empty_frame", "session", "configuration",
                                           "sample_exposure_s", "empty_exposure_s"])


# ------------------------------------------------------------------------------------ helpers
def parse_date(text: str) -> Optional[datetime]:
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def parse_comment(comment: str, distance_m: float) -> Tuple[str, str, str, str]:
    """(name, tag, collimation, configuration) from a Xenocs-style ``Comment`` header."""
    comment = comment or ""
    if " - " in comment:
        name, tag = (x.strip() for x in comment.split(" - ", 1))
    else:
        name, tag = comment.strip(), ""
    m = COLLIMATION_RE.search(tag)
    coll = m.group(1).upper() if m else "prod"
    m = CONFIGURATION_RE.search(tag)
    if m:
        conf = m.group(1).upper()
    else:
        conf = DISTANCE_CONFIGURATIONS.get(round(float(distance_m), 2), f"D{float(distance_m):.2f}")
    return name, tag, coll, conf


def capillary_token(*texts: str) -> Optional[float]:
    """Capillary diameter in mm parsed from a name ("0p5mm", "1.5 mm"); None when absent."""
    for t in texts:
        if not t:
            continue
        m = CAPILLARY_RE.search(t)
        if m:
            return float(m.group(1).replace("p", "."))
    return None


def classify_role(stem: str, comment_name: str, empty_pattern: str, calibrant_pattern: str) -> str:
    texts = [stem, comment_name]
    if empty_pattern and any(re.search(empty_pattern, t) for t in texts if t):
        return "empty"
    if calibrant_pattern and any(re.search(calibrant_pattern, t) for t in texts if t):
        return "calibrant"
    return "sample"


def _rel_and_session(path: str, root: str) -> Tuple[str, str]:
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    parts = rel.split("/")
    session = parts[0] if len(parts) > 1 else os.path.basename(os.path.normpath(root))
    return rel, session


def list_frames(roots: Union[str, Sequence[str]], extensions: Sequence[str] = FRAME_EXTENSIONS) -> List[Tuple[str, str]]:
    """(path, root) for every frame file under the roots, sorted."""
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    out: List[Tuple[str, str]] = []
    for root in roots:
        root = os.path.abspath(str(root))
        if os.path.isfile(root):
            out.append((root, os.path.dirname(root)))
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith(tuple(extensions)):
                    out.append((os.path.join(dirpath, f), root))
    out.sort(key=lambda t: t[0].lower())
    return out


# ------------------------------------------------------------------------------------ scanning
def _base_record(path: str, root: str, h: Dict[str, str], geom: es.FrameGeometry, shape: Tuple[int, int],
                 empty_pattern: str, calibrant_pattern: str, name_from: str) -> FrameRecord:
    stem = os.path.splitext(os.path.basename(path))[0]
    comment_name, tag, coll, conf = parse_comment(h.get("Comment", ""), geom.dist_m)
    name = comment_name if (name_from == "comment" and comment_name) else stem
    rel, session = _rel_and_session(path, root)
    return FrameRecord(path=os.path.abspath(path), rel=rel, folder=os.path.dirname(os.path.abspath(path)), session=session,
                       stem=stem, name=name, comment_name=comment_name, tag=tag, collimation=coll, configuration=conf,
                       role=classify_role(stem, comment_name, empty_pattern, calibrant_pattern),
                       date=h.get("Date", ""), exposure_s=geom.exposure_s, distance_m=geom.dist_m,
                       wavelength_A=geom.wavelength_A, pixel_m=geom.pixel_m, shape=tuple(int(x) for x in shape),
                       count_cutoff=float(h.get("CountCutoff", "nan") or "nan"), cx_header=geom.cx, cy_header=geom.cy,
                       capillary_mm=capillary_token(stem, h.get("Comment", "")), header={k: str(v) for k, v in h.items()})


def diagnose_frame(rec: FrameRecord, data: np.ndarray, reference: Optional["FrameRecord"] = None,
                   max_shift_px: float = 4.0, bright_fraction: float = 0.01, disc_r_px: float = 4.0) -> FrameRecord:
    """Fill the data-dependent fields of ``rec`` in place. The centre used for the diagnostics follows
    the reduction's rule: the frame's own direct-beam centroid when the centre is bright and (with a
    ``reference`` empty) within ``max_shift_px`` of the empty's centre, otherwise the empty's centre
    (a dark centre = opaque loading whose centroid would land on the capillary lobes)."""
    valid = es.valid_pixel_mask(data)
    hot = es.hot_pixel_mask(data, valid)
    ok = valid & ~hot
    rec.n_negative = int((data < 0).sum())
    rec.n_hot = int(hot.sum())
    rec.max_cps = float(np.nanmax(np.where(valid, data, -np.inf))) / rec.exposure_s
    rec.saturated = bool(np.isfinite(rec.count_cutoff) and rec.max_cps * rec.exposure_s >= rec.count_cutoff)
    have_ref = reference is not None and reference.cx is not None
    seed = (reference.cx, reference.cy) if have_ref else (rec.cx_header, rec.cy_header)
    own = es.beam_centroid(data, seed[0], seed[1])
    if own is not None:
        rec.cx_own, rec.cy_own = float(own[0]), float(own[1])
    if have_ref:
        rec.reference_empty = reference.rel
        g_ref = es.FrameGeometry(cx=reference.cx, cy=reference.cy, dist_m=rec.distance_m, pixel_m=rec.pixel_m,
                                 wavelength_A=rec.wavelength_A, exposure_s=rec.exposure_s)
        level = es.disc_counts(data, g_ref, 5.0, ok)[0]
        ref_level = reference.direct_beam_cps_r5 if np.isfinite(reference.direct_beam_cps_r5) else reference.direct_beam_cps
        bright = ref_level > 0 and level > bright_fraction * ref_level
        if own is not None and bright and np.hypot(own[0] - reference.cx, own[1] - reference.cy) < max_shift_px:
            rec.cx, rec.cy, rec.centre_source = rec.cx_own, rec.cy_own, "own direct beam"
        else:
            rec.cx, rec.cy = float(reference.cx), float(reference.cy)
            rec.centre_source = "session empty (centre too dark)" if not bright else "session empty (own centroid off)"
    elif own is not None:
        rec.cx, rec.cy, rec.centre_source = rec.cx_own, rec.cy_own, "own direct beam"
    else:
        rec.cx, rec.cy, rec.centre_source = rec.cx_header, rec.cy_header, "header (no direct beam found)"
    rec.centre_offset_px = (round(rec.cx - rec.cx_header, 2), round(rec.cy - rec.cy_header, 2))
    rec.centre_offset_mm = float(np.hypot(*rec.centre_offset_px) * rec.pixel_m * 1e3)
    g = rec.geometry(refined=True)
    rec.direct_beam_cps = es.disc_counts(data, g, disc_r_px, ok)[0]
    rec.direct_beam_cps_r5 = es.disc_counts(data, g, 5.0, ok)[0]
    an = es.sector_anisotropy(data, g, ok)
    rec.lobe_q, rec.lobe_prominence, rec.lobe_ratio = an.lobe_q, an.lobe_prominence, an.lobe_ratio
    rec.streak_h, rec.streak_v = an.streak_h, an.streak_v
    return rec


def frame_record(path: str, root: Optional[str] = None, empty_pattern: str = r"(?i)empty",
                 calibrant_pattern: str = r"(?i)agbeh", name_from: str = "stem", diagnostics: bool = True,
                 reference: Optional[FrameRecord] = None, default_wavelength_A: Optional[float] = None) -> FrameRecord:
    """Read one frame and build its record; with ``diagnostics`` the data are analysed too
    (:func:`diagnose_frame`, using ``reference`` as the session empty when given)."""
    root = os.path.dirname(path) if root is None else root
    data, h, geom = es.read_frame(path, default_wavelength_A)
    rec = _base_record(path, root, h, geom, data.shape, empty_pattern, calibrant_pattern, name_from)
    if diagnostics:
        diagnose_frame(rec, data, reference)
    return rec


def reference_empty(rec: FrameRecord, empties: Sequence[FrameRecord], distance_tol_m: float = 0.05,
                    default_capillary_mm: float = 1.0, fallback_to_session: bool = True) -> Optional[FrameRecord]:
    """The empty of the same folder (else session) and configuration, nearest in time."""
    cand = _candidate_empties(rec, empties, distance_tol_m, default_capillary_mm, prefer_same_collimation=True)
    return _pick_in_scope(rec, cand, fallback_to_session)


def scan_frames(roots: Union[str, Sequence[str]], empty_pattern: str = r"(?i)empty", calibrant_pattern: str = r"(?i)agbeh",
                exclude_pattern: Optional[str] = None, name_from: str = "stem", diagnostics: bool = True,
                progress: Progress = None, on_error: Optional[Callable[[str, Exception], None]] = None,
                max_shift_px: float = 4.0, bright_fraction: float = 0.01,
                default_wavelength_A: Optional[float] = None) -> List[FrameRecord]:
    """Inventory of every frame under ``roots``. ``exclude_pattern`` is searched in the relative path
    and in the stem; a frame that cannot be read (or has no wavelength while ``default_wavelength_A``
    is None) is reported through ``on_error`` and skipped, or raises without ``on_error``. With
    ``diagnostics`` the empties are analysed first (their own direct beam gives the session centre),
    then every other frame with its session empty as the reference."""
    files = list_frames(roots)
    records: List[FrameRecord] = []
    pending: List[Tuple[FrameRecord, str]] = []
    empties: List[FrameRecord] = []
    n_total = len(files)
    n_done = 0

    def tick(msg: str) -> None:
        if progress is not None:
            progress(n_done, n_total, msg)

    for path, root in files:
        rel, _ = _rel_and_session(path, root)
        stem = os.path.splitext(os.path.basename(path))[0]
        if exclude_pattern and (re.search(exclude_pattern, rel) or re.search(exclude_pattern, stem)):
            n_total -= 1
            continue
        tick(rel)
        try:
            data, h, geom = es.read_frame(path, default_wavelength_A)
            rec = _base_record(path, root, h, geom, data.shape, empty_pattern, calibrant_pattern, name_from)
            if diagnostics and rec.role == "empty":
                diagnose_frame(rec, data, None, max_shift_px, bright_fraction)
                empties.append(rec)
            elif diagnostics:
                pending.append((rec, path))
        except Exception as exc:          # one unreadable frame must not stop the inventory
            n_total -= 1
            if on_error is not None:
                on_error(path, exc)
                continue
            raise
        records.append(rec)
        if not (diagnostics and rec.role != "empty"):
            n_done += 1
    for rec, path in pending:
        tick(rec.rel)
        try:
            data, _h, _g = es.read_frame(path, default_wavelength_A)
            diagnose_frame(rec, data, reference_empty(rec, empties), max_shift_px, bright_fraction)
        except Exception as exc:
            if on_error is not None:
                on_error(path, exc)
            else:
                raise
        n_done += 1
    if progress is not None:
        progress(n_total, n_total, "done")
    return records


# ------------------------------------------------------------------------------------ pairing
def _capillary(rec: FrameRecord, default_mm: float) -> float:
    return rec.capillary_mm if rec.capillary_mm is not None else default_mm


def _candidate_empties(s: FrameRecord, empties: Sequence[FrameRecord], distance_tol_m: float, default_capillary_mm: float,
                       prefer_same_collimation: bool) -> List[FrameRecord]:
    cand = [e for e in empties if abs(e.distance_m - s.distance_m) < distance_tol_m
            and _capillary(e, default_capillary_mm) == _capillary(s, default_capillary_mm)]
    if prefer_same_collimation:
        same = [e for e in cand if e.collimation == s.collimation]
        if same:
            cand = same
    return cand


def _pick_in_scope(s: FrameRecord, cand: List[FrameRecord], fallback_to_session: bool) -> Optional[FrameRecord]:
    scopes = [("folder", s.folder)] + ([("session", s.session)] if fallback_to_session else [])
    for attr, value in scopes:
        c2 = [e for e in cand if getattr(e, attr) == value]
        if c2:
            return _nearest_in_time(s, c2)
    return None


def _nearest_in_time(sample: FrameRecord, candidates: List[FrameRecord]) -> FrameRecord:
    ds = sample.datetime
    if ds is None or any(c.datetime is None for c in candidates):
        return candidates[0]
    return min(candidates, key=lambda c: abs((c.datetime - ds).total_seconds()))


def _resolve_explicit(sample: FrameRecord, explicit: Dict[str, str], empties: List[FrameRecord]) -> Optional[FrameRecord]:
    target = None
    for key in (sample.name, sample.rel, sample.path, sample.stem):
        if key in explicit:
            target = str(explicit[key])
            break
    if target is None:
        return None
    norm = target.replace("\\", "/")
    for e in empties:
        if norm in (e.name, e.rel, e.stem, e.path.replace("\\", "/")) or e.path.replace("\\", "/").endswith(norm):
            return e
    return None


def pair_frames(records: Iterable[FrameRecord], rule: str = "same_folder",
                explicit: Optional[Union[Dict[str, str], str]] = None, distance_tol_m: float = 0.05,
                default_capillary_mm: float = 1.0, fallback_to_session: bool = True,
                prefer_same_collimation: bool = True) -> PairingResult:
    """Pair each sample frame with an empty frame.

    Rules: ``same_folder`` (empties of the same immediate folder, same configuration (distance within
    ``distance_tol_m``) and same capillary token; the nearest in time; with ``fallback_to_session`` the
    session (top-level folder) is tried when the folder has none), ``nearest_time`` (any empty of the
    same configuration and capillary, nearest header date) and ``explicit`` (a ``{sample: empty}``
    mapping by name / relative path / path, or the path of a CSV with columns ``sample, empty``)."""
    records = list(records)
    samples = [r for r in records if r.role == "sample"]
    empties = [r for r in records if r.role == "empty"]
    if rule == "explicit":
        if isinstance(explicit, (str, os.PathLike)):
            df = pd.read_csv(explicit)
            explicit = {str(a): str(b) for a, b in zip(df["sample"], df["empty"])}
        explicit = dict(explicit or {})
    pairs: List[FramePair] = []
    unpaired: List[FrameRecord] = []
    for s in samples:
        chosen: Optional[FrameRecord] = None
        if rule == "explicit":
            chosen = _resolve_explicit(s, explicit, empties)
        else:
            cand = _candidate_empties(s, empties, distance_tol_m, default_capillary_mm, prefer_same_collimation)
            if rule == "same_folder":
                chosen = _pick_in_scope(s, cand, fallback_to_session)
            elif rule == "nearest_time":
                if cand:
                    chosen = _nearest_in_time(s, cand)
            else:
                raise ValueError(f"unknown pairing rule {rule!r}")
        if chosen is None:
            unpaired.append(s)
        else:
            pairs.append(FramePair(sample=s, empty=chosen))
    return PairingResult(pairs=pairs, unpaired=unpaired, rule=rule)


# ------------------------------------------------------------------------------------ tables
INVENTORY_COLUMNS = [f.name for f in fields(FrameRecord) if f.name != "header"]


def inventory_table(records: Iterable[FrameRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        d.pop("header", None)
        d["shape"] = f"{r.shape[0]}x{r.shape[1]}"
        d["centre_offset_px"] = (f"{r.centre_offset_px[0]:+.2f}, {r.centre_offset_px[1]:+.2f}" if r.centre_offset_px else "")
        rows.append(d)
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def write_inventory(records: Iterable[FrameRecord], path: str) -> pd.DataFrame:
    df = inventory_table(records)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df
