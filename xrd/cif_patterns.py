"""
cif_patterns.py — calculated powder diffraction patterns from CIF files.

Why a calculator and not just the cell: PRISM already had a cell-only Bragg
generator (raman/cif_tools.bragg_peaks_from_cif_generic) that lists every
hkl the lattice metric allows, with no intensities. That is fine for "where
could a line be", and wrong for the two things asked of it here — phase
identification and stick overlays on a figure — for one concrete reason:
a centred lattice extinguishes half of those reflections. An F-centred
cubic CIF drawn from the cell alone shows roughly twice the bars the sample
can produce, and a search-match scored against them is penalized for every
reflection the structure was never going to give. Systematic absences are
not a table to look up; they fall out of the structure factor being zero,
which is what this module computes.

What is calculated, per reflection:

    F(hkl) = Σ_j  occ_j · f0_j(s) · exp(-B_j s²) · exp(2πi(h x_j + k y_j + l z_j))
    I(hkl) = |F(hkl)|² · LP(θ)              s = sinθ/λ = 1/(2d)
    LP(θ)  = (1 + cos²2θ) / (sin²θ · cosθ)   — Bragg-Brentano, unpolarized

over the atoms of the asymmetric unit expanded by the CIF's OWN symmetry
operations (a CIF that lists none is treated as P1, which is what it
says). Reflections are enumerated over the full hkl sphere and then merged
by d-spacing, so multiplicity — Friedel pairs included — is counted rather
than tabulated. f0 comes from xraydb when it is installed (PRISM already
uses it for the element filters) and falls back to the atomic number, which
is exact at s = 0 and degrades smoothly at high angle.

What is NOT modelled: preferred orientation, absorption, anomalous
dispersion, TDS, and any instrument profile. These are relative line
intensities of an ideal randomly-oriented powder, which is what a stick
pattern is.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CU_KA1 = 1.5406

CIF_PATTERN_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "xrd_id", "cifcalc")

# Elements as CIF type symbols may spell them: "O2-", "Si4+", "Fe(3)", "D".
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")
_ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
}


# =============================================================================
# CIF reading
# =============================================================================

def _strip_uncertainty(token: Optional[str]) -> Optional[float]:
    """'3.456(12)' -> 3.456; '?' / '.' / '' -> None."""
    if token is None:
        return None
    t = str(token).strip().strip("'\"")
    t = t.split("(", 1)[0].strip()
    if t in ("", "?", "."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _element_of(*candidates: Optional[str]) -> Optional[str]:
    """The element behind a CIF type symbol or site label, trying the type
    symbol first and the label only as a fallback (a label may be anything,
    but in practice it starts with the element)."""
    for cand in candidates:
        if not cand:
            continue
        t = str(cand).strip().strip("'\"")
        t = re.sub(r"^[0-9]+", "", t)  # some files write '1O'
        m = _ELEMENT_RE.match(t)
        if not m:
            continue
        sym = m.group(1)
        if sym in _ELEMENTS:
            return sym
        if sym[:1] in _ELEMENTS:  # 'Ow' (water oxygen) and friends
            return sym[:1]
        if sym == "D":
            return "H"
    return None


def _tokenize_cif_line(line: str) -> List[str]:
    """CIF values are whitespace separated but may be quoted."""
    out, i, n = [], 0, len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            j = i + 1
            while j < n and line[j] != ch:
                j += 1
            out.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out


def read_cif(path: str) -> Tuple[Dict[str, str], List[Tuple[List[str], List[List[str]]]]]:
    """Minimal CIF reader: (tags, loops). Handles quoted values and
    semicolon text blocks; only the first data block is read, which is what
    a structure file holds."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().splitlines()

    tags: Dict[str, str] = {}
    loops: List[Tuple[List[str], List[List[str]]]] = []
    i, n = 0, len(raw)
    seen_data = False
    while i < n:
        line = raw[i]
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("data_"):
            if seen_data:
                break
            seen_data = True
            i += 1
            continue
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if low.startswith("loop_"):
            i += 1
            loop_tags: List[str] = []
            while i < n and raw[i].strip().startswith("_"):
                loop_tags.append(raw[i].strip().split()[0].lower())
                i += 1
            rows: List[List[str]] = []
            buf: List[str] = []
            while i < n:
                s = raw[i].strip()
                if not s or s.startswith("#"):
                    i += 1
                    continue
                if s.startswith("_") or s.lower().startswith(("loop_", "data_")):
                    break
                if s.startswith(";"):  # a text field inside a loop row
                    block = [s[1:]]
                    i += 1
                    while i < n and not raw[i].strip().startswith(";"):
                        block.append(raw[i])
                        i += 1
                    i += 1
                    buf.append("\n".join(block).strip())
                else:
                    buf.extend(_tokenize_cif_line(s))
                    i += 1
                while len(buf) >= len(loop_tags):
                    rows.append(buf[:len(loop_tags)])
                    buf = buf[len(loop_tags):]
            if loop_tags:
                loops.append((loop_tags, rows))
            continue
        if stripped.startswith("_"):
            parts = _tokenize_cif_line(stripped)
            tag = parts[0].lower()
            if len(parts) > 1:
                tags[tag] = " ".join(parts[1:])
                i += 1
            else:
                i += 1
                if i < n and raw[i].strip().startswith(";"):
                    block = [raw[i].strip()[1:]]
                    i += 1
                    while i < n and not raw[i].strip().startswith(";"):
                        block.append(raw[i])
                        i += 1
                    i += 1
                    tags[tag] = "\n".join(block).strip()
                elif i < n:
                    vals = _tokenize_cif_line(raw[i].strip())
                    tags[tag] = " ".join(vals)
                    i += 1
            continue
        i += 1
    return tags, loops


_SYMOP_TERM = re.compile(r"([+-]?)\s*(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)?\s*\*?\s*([xyz]?)")


def parse_symop(text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """'-y, x-y, z+2/3' -> (3x3 rotation, 3 translation). None if
    unparseable — a malformed operator must not silently become identity
    and duplicate every atom onto itself."""
    text = str(text).strip().strip("'\"")
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    rot = np.zeros((3, 3), float)
    trans = np.zeros(3, float)
    for row, expr in enumerate(parts):
        expr = expr.replace(" ", "").lower()
        if not expr:
            return None
        pos = 0
        matched_any = False
        while pos < len(expr):
            m = _SYMOP_TERM.match(expr, pos)
            if not m or m.end() == pos:
                return None
            sign, number, var = m.group(1), m.group(2), m.group(3)
            pos = m.end()
            if not number and not var:
                return None
            scale = -1.0 if sign == "-" else 1.0
            if number:
                if "/" in number:
                    num, den = number.split("/")
                    scale *= float(num) / float(den)
                else:
                    scale *= float(number)
            if var:
                rot[row, "xyz".index(var)] += scale
            else:
                trans[row] += scale
            matched_any = True
        if not matched_any:
            return None
    return rot, trans


@dataclass
class CifStructure:
    path: str
    name: str = ""
    formula: str = ""
    spacegroup: str = ""
    cell: Tuple[float, float, float, float, float, float] = (1, 1, 1, 90, 90, 90)
    elements: List[str] = field(default_factory=list)      # per expanded site
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    occupancies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_iso: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_symops: int = 1
    n_asymmetric: int = 0

    @property
    def element_set(self) -> List[str]:
        return sorted(set(self.elements))


def parse_cif_structure(path: str) -> CifStructure:
    """Cell, symmetry-expanded atom sites, and the identifying metadata.
    Raises ValueError when the file carries no usable cell."""
    tags, loops = read_cif(path)

    def tag(*names: str) -> str:
        for nm in names:
            v = tags.get(nm)
            if v not in (None, "", "?", "."):
                return str(v).strip().strip("'\"")
        return ""

    cell_vals = [
        _strip_uncertainty(tags.get("_cell_length_a")),
        _strip_uncertainty(tags.get("_cell_length_b")),
        _strip_uncertainty(tags.get("_cell_length_c")),
        _strip_uncertainty(tags.get("_cell_angle_alpha")),
        _strip_uncertainty(tags.get("_cell_angle_beta")),
        _strip_uncertainty(tags.get("_cell_angle_gamma")),
    ]
    if any(v is None or v <= 0 for v in cell_vals[:3]):
        raise ValueError(f"{os.path.basename(path)}: no unit cell in the file.")
    angles = [v if (v is not None and 0 < v < 180) else 90.0 for v in cell_vals[3:]]
    cell = (float(cell_vals[0]), float(cell_vals[1]), float(cell_vals[2]),
            float(angles[0]), float(angles[1]), float(angles[2]))

    # --- symmetry operations
    symops: List[Tuple[np.ndarray, np.ndarray]] = []
    for loop_tags, rows in loops:
        cols = {t: k for k, t in enumerate(loop_tags)}
        key = next((t for t in ("_space_group_symop_operation_xyz",
                                "_symmetry_equiv_pos_as_xyz",
                                "_space_group_symop_id_operation_xyz") if t in cols), None)
        if key is None:
            continue
        for row in rows:
            op = parse_symop(row[cols[key]])
            if op is not None:
                symops.append(op)
        break
    if not symops:
        symops = [(np.eye(3), np.zeros(3))]

    # --- asymmetric unit
    sites: List[Tuple[str, np.ndarray, float, float]] = []
    for loop_tags, rows in loops:
        cols = {t: k for k, t in enumerate(loop_tags)}
        if not {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"} <= set(cols):
            continue
        for row in rows:
            xyz = [_strip_uncertainty(row[cols[f"_atom_site_fract_{c}"]]) for c in "xyz"]
            if any(v is None for v in xyz):
                continue
            el = _element_of(row[cols["_atom_site_type_symbol"]] if "_atom_site_type_symbol" in cols else None,
                             row[cols["_atom_site_label"]] if "_atom_site_label" in cols else None)
            if el is None:
                continue
            occ = _strip_uncertainty(row[cols["_atom_site_occupancy"]]) if "_atom_site_occupancy" in cols else None
            u = _strip_uncertainty(row[cols["_atom_site_U_iso_or_equiv"]]) if "_atom_site_U_iso_or_equiv" in cols else None
            b = _strip_uncertainty(row[cols["_atom_site_B_iso_or_equiv"]]) if "_atom_site_B_iso_or_equiv" in cols else None
            if b is None:
                b = 8.0 * math.pi ** 2 * u if u is not None else 0.5
            sites.append((el, np.array(xyz, float), 1.0 if occ is None else float(occ), float(b)))
        break

    # --- expand by symmetry, dropping images that coincide with one already placed
    els: List[str] = []
    pos: List[np.ndarray] = []
    occs: List[float] = []
    bs: List[float] = []
    for el, xyz, occ, b in sites:
        placed: List[np.ndarray] = []
        for rot, trans in symops:
            p = (rot @ xyz + trans) % 1.0
            p = np.where(p > 1.0 - 1e-5, 0.0, p)
            if any(np.all(np.abs(p - q) < 1e-4) for q in placed):
                continue
            placed.append(p)
            els.append(el)
            pos.append(p)
            occs.append(occ)
            bs.append(b)

    return CifStructure(
        path=path,
        name=tag("_chemical_name_mineral", "_chemical_name_common", "_chemical_name_systematic")
        or os.path.splitext(os.path.basename(path))[0],
        formula=re.sub(r"\s+", " ", tag("_chemical_formula_sum", "_chemical_formula_structural",
                                        "_chemical_formula_moiety")).strip(),
        spacegroup=tag("_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt",
                       "_symmetry_space_group_name_hall", "_space_group_name_hall"),
        cell=cell, elements=els,
        positions=np.array(pos, float).reshape(-1, 3),
        occupancies=np.array(occs, float), b_iso=np.array(bs, float),
        n_symops=len(symops), n_asymmetric=len(sites),
    )


# =============================================================================
# Powder pattern
# =============================================================================

def reciprocal_metric(cell: Sequence[float]) -> np.ndarray:
    """G* — the reciprocal metric tensor, so 1/d² = hᵀ G* h."""
    a, b, c, al, be, ga = [float(v) for v in cell]
    ca, cb, cg = (math.cos(math.radians(v)) for v in (al, be, ga))
    G = np.array([[a * a, a * b * cg, a * c * cb],
                  [a * b * cg, b * b, b * c * ca],
                  [a * c * cb, b * c * ca, c * c]], float)
    if abs(np.linalg.det(G)) < 1e-18:
        raise ValueError("degenerate unit cell")
    return np.linalg.inv(G)


def _scattering_factors(elements: Sequence[str], s: np.ndarray) -> Dict[str, np.ndarray]:
    """f0 per element over the s = sinθ/λ grid. xraydb when available (it is
    already a PRISM dependency for the element filters); atomic number
    otherwise — exact at s=0, increasingly generous at high angle, which
    shifts relative intensities but never invents or removes a line."""
    out: Dict[str, np.ndarray] = {}
    try:
        import xraydb
        for el in set(elements):
            try:
                out[el] = np.asarray(xraydb.f0(el, s), float)
            except Exception:
                out[el] = np.full(len(s), float(_ELEMENTS.get(el, 1)))
    except ImportError:  # pragma: no cover - xraydb ships with PRISM
        for el in set(elements):
            out[el] = np.full(len(s), float(_ELEMENTS.get(el, 1)))
    return out


@dataclass
class CifPattern:
    path: str
    name: str
    formula: str
    spacegroup: str
    cell: Tuple[float, float, float, float, float, float]
    wavelength: float
    d: np.ndarray
    two_theta: np.ndarray
    intensity: np.ndarray            # normalized so the strongest line is 100
    hkl: List[Tuple[int, int, int]]  # one representative per merged d
    elements: List[str] = field(default_factory=list)
    n_atoms: int = 0
    calculated_intensities: bool = True  # False => cell-only, equal weights


def powder_pattern_from_cif(
    path: str, *, wavelength: float = CU_KA1, two_theta_min: float = 5.0,
    two_theta_max: float = 90.0, min_intensity: float = 0.05,
    max_lines: int = 400, d_merge_tol: float = 1e-4, use_cache: bool = True,
) -> CifPattern:
    """Calculate the powder stick pattern of a CIF.

    min_intensity is a percentage of the strongest line; max_lines caps the
    result (strongest kept). A structure file with a cell but no usable
    atom sites still gives a pattern — positions from the lattice, equal
    intensities — flagged by calculated_intensities=False so callers can
    say so rather than implying a structure factor was computed.
    """
    key = _cache_key(path, wavelength, two_theta_min, two_theta_max, min_intensity, max_lines)
    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached

    st = parse_cif_structure(path)
    gstar = reciprocal_metric(st.cell)

    tt_max = min(float(two_theta_max), 179.0)
    d_min = wavelength / (2.0 * math.sin(math.radians(tt_max / 2.0)))
    a, b, c = st.cell[0], st.cell[1], st.cell[2]
    # |h| = |r* · a| <= |r*|·|a| = a/d  — an exact bound, so no reflection
    # inside the range can fall outside the enumerated box
    hmax, kmax, lmax = (max(1, int(math.floor(v / d_min))) for v in (a, b, c))
    if (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1) > 8_000_000:
        raise ValueError(
            f"{os.path.basename(path)}: cell too large for the requested 2θ range "
            f"({a:.1f}×{b:.1f}×{c:.1f} Å). Lower the maximum 2θ.")

    hs = np.arange(-hmax, hmax + 1)
    ks = np.arange(-kmax, kmax + 1)
    ls = np.arange(-lmax, lmax + 1)
    H = np.stack(np.meshgrid(hs, ks, ls, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    H = H[np.any(H != 0, axis=1)]
    inv_d2 = np.einsum("ij,jk,ik->i", H, gstar, H)
    sin_theta = np.sqrt(np.clip(inv_d2, 0, None)) * wavelength / 2.0
    tt = np.full(len(H), np.nan)
    ok = (sin_theta > 0) & (sin_theta <= math.sin(math.radians(tt_max / 2.0)))
    tt[ok] = 2.0 * np.degrees(np.arcsin(sin_theta[ok]))
    keep = ok & (tt >= two_theta_min) & (tt <= tt_max)
    H, tt, inv_d2 = H[keep], tt[keep], inv_d2[keep]
    if len(H) == 0:
        raise ValueError(f"{os.path.basename(path)}: no reflections in "
                         f"{two_theta_min:g}–{two_theta_max:g}°2θ at λ={wavelength} Å.")
    d = 1.0 / np.sqrt(inv_d2)
    s = 1.0 / (2.0 * d)  # sinθ/λ

    calculated = len(st.positions) > 0
    if calculated:
        f0 = _scattering_factors(st.elements, s)
        f_real = np.zeros(len(H))
        f_imag = np.zeros(len(H))
        dw = np.exp(-np.outer(st.b_iso, s ** 2))                    # (n_atom, n_hkl)
        phase_scale = 2.0 * np.pi
        chunk = max(1, int(2_000_000 // max(1, len(st.positions))))
        for start in range(0, len(H), chunk):
            stop = min(start + chunk, len(H))
            phase = phase_scale * (H[start:stop] @ st.positions.T)   # (n_hkl, n_atom)
            amp = np.stack([f0[el][start:stop] for el in st.elements], axis=1)
            amp = amp * st.occupancies[None, :] * dw[:, start:stop].T
            f_real[start:stop] = (amp * np.cos(phase)).sum(axis=1)
            f_imag[start:stop] = (amp * np.sin(phase)).sum(axis=1)
        f2 = f_real ** 2 + f_imag ** 2
    else:
        f2 = np.ones(len(H))

    theta = np.radians(tt / 2.0)
    lp = (1.0 + np.cos(2.0 * theta) ** 2) / np.clip(np.sin(theta) ** 2 * np.cos(theta), 1e-12, None)
    inten = f2 * lp

    # multiplicity by merging equal d-spacings — which is also what the
    # powder itself does to accidentally coincident reflections
    order = np.argsort(-d, kind="mergesort")
    d, tt, inten, H = d[order], tt[order], inten[order], H[order]
    group = np.concatenate([[0], np.cumsum(np.abs(np.diff(d)) > d_merge_tol * np.maximum(d[:-1], 1.0))])
    n_groups = int(group[-1]) + 1
    md = np.zeros(n_groups)
    mi = np.zeros(n_groups)
    mh: List[Tuple[int, int, int]] = []
    np.add.at(mi, group, inten)
    first = np.zeros(n_groups, int)
    first[group[::-1]] = np.arange(len(group))[::-1]
    md = d[first]
    mt = tt[first]
    for idx in first:
        mh.append(tuple(int(v) for v in H[idx]))

    peak = float(mi.max()) if len(mi) else 0.0
    if peak <= 0:
        raise ValueError(f"{os.path.basename(path)}: every reflection is extinct — "
                         f"check the atom sites and symmetry operators in the file.")
    mi = mi / peak * 100.0
    strong = mi >= float(min_intensity)
    md, mt, mi = md[strong], mt[strong], mi[strong]
    mh = [h for h, s_ok in zip(mh, strong) if s_ok]
    if max_lines and len(mi) > max_lines:
        top = np.argsort(-mi)[:max_lines]
        top = np.sort(top)
        md, mt, mi = md[top], mt[top], mi[top]
        mh = [mh[i] for i in top]

    pattern = CifPattern(
        path=os.path.abspath(path), name=st.name, formula=st.formula,
        spacegroup=st.spacegroup, cell=st.cell, wavelength=float(wavelength),
        d=md, two_theta=mt, intensity=mi, hkl=mh, elements=st.element_set,
        n_atoms=len(st.positions), calculated_intensities=calculated,
    )
    if use_cache:
        _save_cached(key, pattern)
    return pattern


# =============================================================================
# Cache (a big cell over a wide range takes seconds; a folder of them adds up)
# =============================================================================

def _cache_key(path: str, *parts) -> str:
    try:
        stat = os.stat(path)
        sig = f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}|" + \
              "|".join(f"{p}" for p in parts)
    except OSError:
        sig = os.path.abspath(path) + "|" + "|".join(f"{p}" for p in parts)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def _load_cached(key: str) -> Optional[CifPattern]:
    fp = os.path.join(CIF_PATTERN_CACHE_DIR, f"{key}.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            obj = json.load(f)
        return CifPattern(
            path=obj["path"], name=obj["name"], formula=obj["formula"],
            spacegroup=obj["spacegroup"], cell=tuple(obj["cell"]),
            wavelength=obj["wavelength"], d=np.array(obj["d"], float),
            two_theta=np.array(obj["two_theta"], float),
            intensity=np.array(obj["intensity"], float),
            hkl=[tuple(v) for v in obj["hkl"]], elements=list(obj["elements"]),
            n_atoms=int(obj["n_atoms"]),
            calculated_intensities=bool(obj["calculated_intensities"]),
        )
    except (OSError, ValueError, KeyError):
        return None


def _save_cached(key: str, pattern: CifPattern) -> None:
    try:
        os.makedirs(CIF_PATTERN_CACHE_DIR, exist_ok=True)
        with open(os.path.join(CIF_PATTERN_CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "path": pattern.path, "name": pattern.name, "formula": pattern.formula,
                "spacegroup": pattern.spacegroup, "cell": list(pattern.cell),
                "wavelength": pattern.wavelength, "d": pattern.d.tolist(),
                "two_theta": pattern.two_theta.tolist(),
                "intensity": pattern.intensity.tolist(),
                "hkl": [list(h) for h in pattern.hkl], "elements": list(pattern.elements),
                "n_atoms": pattern.n_atoms,
                "calculated_intensities": pattern.calculated_intensities,
            }, f)
    except OSError:
        pass


def list_cif_files(folder: str, *, recursive: bool = True) -> List[str]:
    """Every .cif under a folder. Recursive by default — a COD download
    unpacks into a deep cif/1/01/09/ tree, and a folder of structures a
    group shares is usually organized in subfolders too."""
    out: List[str] = []
    if not os.path.isdir(folder):
        return out
    if recursive:
        for root, _dirs, files in os.walk(folder):
            out += [os.path.join(root, f) for f in files if f.lower().endswith(".cif")]
    else:
        out += [os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(".cif") and os.path.isfile(os.path.join(folder, f))]
    return sorted(out)
