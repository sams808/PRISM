"""
xrd_id_science.py — XRD phase identification (framework-agnostic): the
QualX-style search-match engine, rebuilt on the user's own QualX-format
SQLite databases (COD 1906-inorganic, COD 2205, ICDD PDF-2), merged into
ONE local database that keeps every card's original source and code.

QualX .sq format (reverse-engineered from the three files): an `id` table
with one row per card — name / mineralname / chemical_formula / spacegroup /
quality / rir and the reference pattern as comma-separated text in `dvalue`
(d-spacings, Å) and `intensita` (intensities) — plus a `chemical` table
(card id → element) and `infodb` (provenance).

Unified database (built once into ~/.raman_cache/xrd_id/xrdid.sq):
  cards(card_id, source, source_code, name, mineral, formula, spacegroup,
        quality, rir, nd, d, i)          — lines capped at the strongest 60
  elements(card_id, element)             — for chemistry filters
  strong_lines(card_id, d, i)            — top-3 lines, d-indexed, the
                                           search-match prefilter
  sources(tag, date, ncard, origin)      — provenance of each merged input

Search-match: query peaks (2θ, I) at wavelength λ → d-spacings; candidate
cards are prefiltered by strong-line coincidence, then scored by the
geometric mean of intensity-weighted coverage in both directions (how much
of the card's pattern the measurement explains, and vice versa) — never an
automatic answer, always a ranked list, same philosophy as the RRUFF
match-assist.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

XRD_ID_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "xrd_id")
XRD_ID_DB_PATH = os.path.join(XRD_ID_CACHE_DIR, "xrdid.sq")

CU_KA1 = 1.5406  # Å — the default lab wavelength

MAX_LINES_PER_CARD = 60  # strongest lines kept; plenty for identification
N_STRONG = 3             # lines per card in the prefilter index


# =============================================================================
# d-spacing / 2θ conversions
# =============================================================================

def two_theta_to_d(two_theta_deg: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    tt = np.asarray(two_theta_deg, float)
    s = np.sin(np.radians(tt / 2.0))
    return np.where(s > 1e-9, wavelength / (2.0 * np.where(s > 1e-9, s, 1.0)), np.inf)


def d_to_two_theta(d: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    d = np.asarray(d, float)
    arg = wavelength / (2.0 * np.where(d > 1e-9, d, np.inf))
    out = np.full_like(d, np.nan)
    ok = (arg > 0) & (arg <= 1.0)
    out[ok] = 2.0 * np.degrees(np.arcsin(arg[ok]))
    return out


# =============================================================================
# Database build (one-time merge of the QualX-format sources)
# =============================================================================

def _parse_lines_text(d_text, i_text) -> Tuple[np.ndarray, np.ndarray]:
    """The dvalue/intensita columns are comma-separated float text (declared
    BLOB in PDF2 but stored as text in all three files)."""
    def _floats(t):
        if t is None:
            return np.array([])
        if isinstance(t, bytes):
            t = t.decode("ascii", errors="replace")
        vals = []
        for tok in str(t).split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        return np.array(vals, float)

    d = _floats(d_text)
    i = _floats(i_text)
    n = min(len(d), len(i))
    return d[:n], i[:n]


def _clean_name(name: Optional[str]) -> str:
    """PDF-2 names carry $-prefixed typesetting codes ('$GB-...')."""
    s = (name or "").strip()
    s = re.sub(r"\$[A-Z0-9]{1,3}[- ]?", "", s)
    return s.strip()


def build_xrd_database(
    sources: Sequence[Tuple[str, str]], *, out_path: str = XRD_ID_DB_PATH,
    min_lines: int = 3, progress_every: int = 50000, log=print,
) -> Dict[str, int]:
    """Merge QualX-format .sq files into the unified database.
    sources: [(path, tag), ...] e.g. [(cod1906ino.sq, 'COD1906-INO'), ...].
    Cards keep their original id as source_code. Cards with fewer than
    min_lines usable lines are skipped (nothing to match against).
    Returns {tag: n_cards_ingested}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    out = sqlite3.connect(out_path)
    out.executescript("""
        CREATE TABLE cards (
            card_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL, source_code TEXT NOT NULL,
            name TEXT, mineral TEXT, formula TEXT, spacegroup TEXT,
            quality TEXT, rir REAL, nd INTEGER, d TEXT, i TEXT
        );
        CREATE TABLE elements (card_id INTEGER NOT NULL, element TEXT NOT NULL);
        CREATE TABLE strong_lines (card_id INTEGER NOT NULL, d REAL NOT NULL, i REAL NOT NULL);
        CREATE TABLE sources (tag TEXT, date TEXT, ncard INTEGER, origin TEXT);
    """)

    counts: Dict[str, int] = {}
    for path, tag in sources:
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for row in src.execute("SELECT * FROM infodb"):
                out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, str(row[1]), int(row[2]), str(row[-1])))
        except sqlite3.Error:
            out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, "", 0, ""))

        n_in = 0
        cur = src.execute(
            "SELECT id, name, mineralname, chemical_formula, spacegroup, quality, rir, dvalue, intensita FROM id"
        )
        rows_cards, rows_strong = [], []
        for k, (cid, name, mineral, formula, sg, quality, rir, d_text, i_text) in enumerate(cur):
            if progress_every and k and k % progress_every == 0:
                log(f"  [{tag}] scanned {k} cards, kept {n_in}…")
            d, i = _parse_lines_text(d_text, i_text)
            if len(d) < min_lines:
                continue
            imax = float(i.max()) if len(i) else 0.0
            if imax <= 0:
                continue
            i = i / imax * 100.0
            order = np.argsort(i)[::-1][:MAX_LINES_PER_CARD]
            d, i = d[order], i[order]
            rows_cards.append((
                tag, str(cid), _clean_name(name), _clean_name(mineral),
                (formula or "").strip().strip('"'), (sg or "").strip(),
                (quality or "").strip(), float(rir) if rir not in (None, "", " ") else None,
                len(d), ",".join(f"{v:.5f}" for v in d), ",".join(f"{v:.2f}" for v in i),
            ))
            n_in += 1
        cur2 = out.executemany(
            "INSERT INTO cards (source, source_code, name, mineral, formula, spacegroup, quality, rir, nd, d, i) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows_cards)
        # strong lines need the new card_id: fetch back in insert order
        first_id = out.execute("SELECT MAX(card_id) FROM cards").fetchone()[0] - len(rows_cards) + 1
        for offset, rec in enumerate(rows_cards):
            d = [float(v) for v in rec[9].split(",")[:N_STRONG]]
            i = [float(v) for v in rec[10].split(",")[:N_STRONG]]
            for dv, iv in zip(d, i):
                rows_strong.append((first_id + offset, dv, iv))
        out.executemany("INSERT INTO strong_lines VALUES (?,?,?)", rows_strong)

        # elements (chemistry filter): only for cards we kept
        code_to_id = {rec[1]: first_id + off for off, rec in enumerate(rows_cards)}
        rows_el = []
        try:
            for cid, el in src.execute("SELECT id, chemical_element FROM chemical"):
                card_id = code_to_id.get(str(cid))
                if card_id is not None and el:
                    rows_el.append((card_id, str(el).strip().capitalize()))
        except sqlite3.Error:
            pass
        out.executemany("INSERT INTO elements VALUES (?,?)", rows_el)
        out.commit()
        counts[tag] = n_in
        log(f"[{tag}] ingested {n_in} cards ({len(rows_el)} element links)")
        src.close()

    log("indexing…")
    out.executescript("""
        CREATE INDEX idx_strong_d ON strong_lines(d);
        CREATE INDEX idx_elements ON elements(element, card_id);
        CREATE INDEX idx_cards_mineral ON cards(mineral);
        CREATE INDEX idx_cards_formula ON cards(formula);
    """)
    out.commit()
    out.close()
    return counts


def database_summary(db_path: str = XRD_ID_DB_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by_source = dict(con.execute("SELECT source, COUNT(*) FROM cards GROUP BY source"))
    total = sum(by_source.values())
    con.close()
    return {"total_cards": total, "by_source": by_source, "path": db_path}


# =============================================================================
# Search-match
# =============================================================================

@dataclass
class XrdMatch:
    card_id: int
    source: str
    source_code: str
    name: str
    mineral: str
    formula: str
    spacegroup: str
    quality: str
    rir: Optional[float]
    fom: float               # 0..1 figure of merit (geometric-mean coverage)
    cov_card: float          # how much of the card's pattern is in the data
    cov_query: float         # how much of the data the card explains
    n_matched: int
    matched_pairs: List[Tuple[float, float]] = field(default_factory=list)  # (query 2θ, card 2θ)
    d: np.ndarray = field(default_factory=lambda: np.array([]))
    i: np.ndarray = field(default_factory=lambda: np.array([]))


def search_match(
    query_two_theta: Sequence[float], query_intensity: Optional[Sequence[float]] = None, *,
    wavelength: float = CU_KA1, tol_two_theta: float = 0.2,
    elements_all: Sequence[str] = (), elements_none: Sequence[str] = (),
    sources: Sequence[str] = (), top_n: int = 40,
    two_theta_range: Optional[Tuple[float, float]] = None,
    max_candidates: int = 20000, db_path: str = XRD_ID_DB_PATH,
) -> List[XrdMatch]:
    """Rank database cards against a measured peak list.

    query_two_theta/intensity: the measured peaks (equal intensities are
    assumed when none given). tol_two_theta: match window in °2θ.
    elements_all: card must contain ALL of these; elements_none: none.
    sources: restrict to these source tags. two_theta_range: only card
    lines inside the measured range count against the card's coverage."""
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No unified XRD database at {db_path}. Build it once with "
            "xrd_id_science.build_xrd_database() — see the XRD workspace help.")
    q_tt = np.asarray(list(query_two_theta), float)
    if len(q_tt) == 0:
        return []
    q_i = (np.asarray(list(query_intensity), float) if query_intensity is not None
           else np.ones_like(q_tt))
    q_i = np.where(np.isfinite(q_i) & (q_i > 0), q_i, 1.0)
    q_d = two_theta_to_d(q_tt, wavelength)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # ---- candidate prefilter: cards whose strong lines coincide with the
    # query's strongest peaks (union, ranked by hit count)
    strongest = np.argsort(q_i)[::-1][:5]
    hits: Dict[int, int] = {}
    for qi in strongest:
        tt = q_tt[qi]
        # Δd from Δ2θ: |dd| = d·cot(θ)·Δθ(rad)/1 … evaluated at this line
        theta = np.radians(tt / 2.0)
        dd = abs(q_d[qi] / np.tan(theta) * np.radians(tol_two_theta / 2.0)) if theta > 0 else 0.01
        dd = max(dd, 1e-4)
        for (cid,) in con.execute("SELECT card_id FROM strong_lines WHERE d BETWEEN ? AND ?",
                                  (q_d[qi] - dd, q_d[qi] + dd)):
            hits[cid] = hits.get(cid, 0) + 1
    if not hits:
        con.close()
        return []
    candidates = sorted(hits, key=lambda c: hits[c], reverse=True)[:max_candidates]

    # ---- chemistry / source filters
    if elements_all:
        keep = None
        for el in elements_all:
            got = {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
            keep = got if keep is None else (keep & got)
        candidates = [c for c in candidates if c in (keep or set())]
    if elements_none:
        drop = set()
        for el in elements_none:
            drop |= {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
        candidates = [c for c in candidates if c not in drop]
    if not candidates:
        con.close()
        return []

    # ---- score
    marks = ",".join("?" * len(candidates))
    src_filter = ""
    params: List[Any] = list(candidates)
    if sources:
        src_filter = f" AND source IN ({','.join('?' * len(sources))})"
        params += list(sources)
    rows = con.execute(
        f"SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, rir, d, i "
        f"FROM cards WHERE card_id IN ({marks}){src_filter}", params).fetchall()
    con.close()

    results: List[XrdMatch] = []
    for (cid, source, code, name, mineral, formula, sg, quality, rir, d_text, i_text) in rows:
        c_d = np.array([float(v) for v in d_text.split(",")], float)
        c_i = np.array([float(v) for v in i_text.split(",")], float)
        c_tt = d_to_two_theta(c_d, wavelength)
        ok = np.isfinite(c_tt)
        if two_theta_range is not None:
            ok &= (c_tt >= two_theta_range[0]) & (c_tt <= two_theta_range[1])
        c_tt, c_i_in = c_tt[ok], c_i[ok]
        if len(c_tt) == 0:
            continue

        # greedy matching, card's strongest lines first (they're pre-sorted)
        matched_pairs: List[Tuple[float, float]] = []
        used_q = np.zeros(len(q_tt), bool)
        matched_ci, matched_qi = 0.0, 0.0
        for tt_c, i_c in zip(c_tt, c_i_in):
            diffs = np.abs(q_tt - tt_c)
            diffs[used_q] = np.inf
            j = int(np.argmin(diffs))
            if diffs[j] <= tol_two_theta:
                used_q[j] = True
                matched_pairs.append((float(q_tt[j]), float(tt_c)))
                matched_ci += float(i_c)
                matched_qi += float(q_i[j])
        if not matched_pairs:
            continue
        cov_card = matched_ci / float(c_i_in.sum()) if c_i_in.sum() > 0 else 0.0
        cov_query = matched_qi / float(q_i.sum()) if q_i.sum() > 0 else 0.0
        fom = float(np.sqrt(cov_card * cov_query))
        results.append(XrdMatch(
            card_id=cid, source=source, source_code=code, name=name or "", mineral=mineral or "",
            formula=formula or "", spacegroup=sg or "", quality=quality or "", rir=rir,
            fom=fom, cov_card=cov_card, cov_query=cov_query, n_matched=len(matched_pairs),
            matched_pairs=matched_pairs, d=c_d, i=c_i,
        ))
    results.sort(key=lambda r: (r.fom, r.n_matched), reverse=True)
    return results[:top_n]


def find_cards_by_text(text: str, *, limit: int = 50, db_path: str = XRD_ID_DB_PATH) -> List[Dict[str, Any]]:
    """Name/mineral/formula substring lookup — the Raman↔XRD bridge uses
    this to pull reference patterns for an already-identified mineral."""
    if not os.path.isfile(db_path):
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    like = f"%{text.strip()}%"
    rows = con.execute(
        "SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, d, i "
        "FROM cards WHERE mineral LIKE ? OR name LIKE ? OR formula LIKE ? LIMIT ?",
        (like, like, like, int(limit))).fetchall()
    con.close()
    out = []
    for (cid, source, code, name, mineral, formula, sg, quality, d_text, i_text) in rows:
        out.append({
            "card_id": cid, "source": source, "source_code": code, "name": name,
            "mineral": mineral, "formula": formula, "spacegroup": sg, "quality": quality,
            "d": np.array([float(v) for v in d_text.split(",")], float),
            "i": np.array([float(v) for v in i_text.split(",")], float),
        })
    return out
