"""Tests for cif_patterns.py — calculated powder patterns from CIFs.

The patterns are checked against things outside the code: analytic
d-spacings for a cubic cell, the textbook systematic-absence rules for
centred lattices (which is the whole reason this calculates structure
factors instead of listing lattice-allowed hkl), and the known line
sequence of rock salt.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import xrd.cif_patterns as cifp


# --------------------------------------------------------------------------
# synthetic CIFs
# --------------------------------------------------------------------------

_FCC_SYMOPS = """loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,-y,z
-x,y,-z
x,-y,-z
x,y+1/2,z+1/2
-x,-y+1/2,z+1/2
-x,y+1/2,-z+1/2
x,-y+1/2,-z+1/2
x+1/2,y,z+1/2
-x+1/2,-y,z+1/2
-x+1/2,y,-z+1/2
x+1/2,-y,-z+1/2
x+1/2,y+1/2,z
-x+1/2,-y+1/2,z
-x+1/2,y+1/2,-z
x+1/2,-y+1/2,-z
"""

_BCC_SYMOPS = """loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,-y,z
-x,y,-z
x,-y,-z
x+1/2,y+1/2,z+1/2
-x+1/2,-y+1/2,z+1/2
-x+1/2,y+1/2,-z+1/2
x+1/2,-y+1/2,-z+1/2
"""


def _cif(path, *, a=5.64, symops=_FCC_SYMOPS, atoms=(("Na", 0, 0, 0), ("Cl", 0.5, 0.5, 0.5)),
         name="test", sg="F m -3 m", formula="Na Cl", extra=""):
    lines = [
        "data_test",
        f"_chemical_name_mineral '{name}'",
        f"_chemical_formula_sum '{formula}'",
        f"_symmetry_space_group_name_H-M '{sg}'",
        f"_cell_length_a {a}", f"_cell_length_b {a}", f"_cell_length_c {a}",
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        extra, symops,
    ]
    if atoms:
        lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol",
                  "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
        for k, (el, x, y, z) in enumerate(atoms, 1):
            lines.append(f"{el}{k} {el} {x} {y} {z}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_parse_symop_handles_signs_fractions_and_combinations():
    rot, trans = cifp.parse_symop("x,y,z")
    assert np.allclose(rot, np.eye(3)) and np.allclose(trans, 0)
    rot, trans = cifp.parse_symop("-y, x-y, z+2/3")
    assert np.allclose(rot, [[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    assert np.allclose(trans, [0, 0, 2 / 3])
    rot, trans = cifp.parse_symop("'-x+1/2, -y, z+1/2'")
    assert np.allclose(rot, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert np.allclose(trans, [0.5, 0.0, 0.5])
    # an unparseable operator is rejected, not silently taken as identity
    assert cifp.parse_symop("x,y") is None
    assert cifp.parse_symop("x,y,w") is None


def test_parse_structure_expands_symmetry_and_reads_metadata(tmp_path):
    st = cifp.parse_cif_structure(_cif(tmp_path / "nacl.cif"))
    assert st.name == "test"
    assert st.formula == "Na Cl"
    assert st.spacegroup == "F m -3 m"
    assert st.n_symops == 16
    assert st.n_asymmetric == 2
    # Na on 4a and Cl on 4b: 4 + 4 sites once duplicate images are dropped
    assert len(st.positions) == 8
    assert sorted(st.element_set) == ["Cl", "Na"]
    assert np.all((st.positions >= 0) & (st.positions < 1))


def test_uncertainties_and_ion_symbols_are_understood(tmp_path):
    path = _cif(tmp_path / "ion.cif", atoms=(("O2-", "0.1234(5)", 0, 0),),
                formula="O", sg="P 1", symops="loop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n")
    st = cifp.parse_cif_structure(path)
    assert st.elements == ["O"]
    assert st.positions[0][0] == pytest.approx(0.1234)


def test_missing_cell_is_an_explicit_error(tmp_path):
    bad = tmp_path / "nocell.cif"
    bad.write_text("data_x\n_chemical_formula_sum 'Si O2'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no unit cell"):
        cifp.parse_cif_structure(str(bad))


# --------------------------------------------------------------------------
# the pattern itself
# --------------------------------------------------------------------------

def test_cubic_line_positions_match_the_analytic_formula(tmp_path):
    a, lam = 5.64, 1.5406
    pat = cifp.powder_pattern_from_cif(_cif(tmp_path / "nacl.cif", a=a),
                                       wavelength=lam, two_theta_max=90, use_cache=False)
    for (h, k, l), d in zip(pat.hkl, pat.d):
        expected = a / math.sqrt(h * h + k * k + l * l)
        assert d == pytest.approx(expected, rel=1e-9)
    tt = 2.0 * np.degrees(np.arcsin(lam / (2.0 * pat.d)))
    assert np.allclose(pat.two_theta, tt)


def test_f_centring_extinguishes_mixed_parity_reflections(tmp_path):
    """The reason this calculates |F|² instead of listing lattice-allowed
    hkl: a cell-only generator reports (100), (110), (210)… for rock salt,
    and the sample produces none of them."""
    pat = cifp.powder_pattern_from_cif(_cif(tmp_path / "nacl.cif"),
                                       two_theta_max=90, use_cache=False)
    for h, k, l in pat.hkl:
        parities = {abs(h) % 2, abs(k) % 2, abs(l) % 2}
        assert len(parities) == 1, f"({h}{k}{l}) is mixed-parity and must be extinct in F-centring"
    # the known rock-salt line sequence (PDF 05-0628): 200 is the 100% line,
    # 220 the second at ~55%, and 111 is present but weak (~13%) because Na
    # and Cl scatter nearly in antiphase on the odd-index reflections
    order = np.argsort(-pat.intensity)
    by_hkl = {tuple(sorted(abs(v) for v in h)): i for h, i in zip(pat.hkl, pat.intensity)}
    assert tuple(sorted(abs(v) for v in pat.hkl[int(order[0])])) == (0, 0, 2)
    assert tuple(sorted(abs(v) for v in pat.hkl[int(order[1])])) == (0, 2, 2)
    assert by_hkl[(0, 2, 2)] == pytest.approx(55.0, abs=15.0)
    assert 5.0 < by_hkl[(1, 1, 1)] < 25.0


def test_i_centring_extinguishes_odd_h_plus_k_plus_l(tmp_path):
    pat = cifp.powder_pattern_from_cif(
        _cif(tmp_path / "bcc.cif", a=2.87, symops=_BCC_SYMOPS,
             atoms=(("Fe", 0, 0, 0),), sg="I m -3 m", formula="Fe"),
        two_theta_max=120, use_cache=False)
    assert pat.hkl
    for h, k, l in pat.hkl:
        assert (h + k + l) % 2 == 0, f"({h}{k}{l}) must be extinct in I-centring"


def test_a_cif_with_no_atom_sites_still_gives_lines_but_says_so(tmp_path):
    path = _cif(tmp_path / "cellonly.cif", atoms=(), symops="")
    pat = cifp.powder_pattern_from_cif(path, two_theta_max=60, use_cache=False)
    assert len(pat.d) > 0
    assert pat.calculated_intensities is False
    assert pat.n_atoms == 0
    # every line equally weighted — after the LP factor, not all equal, but
    # the point is that no structure factor was involved
    assert np.all(pat.intensity > 0)


def test_intensities_are_normalized_and_thresholded(tmp_path):
    pat = cifp.powder_pattern_from_cif(_cif(tmp_path / "nacl.cif"),
                                       two_theta_max=120, min_intensity=1.0, use_cache=False)
    assert pat.intensity.max() == pytest.approx(100.0)
    assert pat.intensity.min() >= 1.0
    assert len(pat.d) == len(pat.two_theta) == len(pat.intensity) == len(pat.hkl)
    assert np.all(np.diff(pat.d) < 0)  # returned largest-d first


def test_pattern_respects_the_two_theta_window(tmp_path):
    pat = cifp.powder_pattern_from_cif(_cif(tmp_path / "nacl.cif"),
                                       two_theta_min=30.0, two_theta_max=50.0, use_cache=False)
    assert pat.two_theta.min() >= 30.0 and pat.two_theta.max() <= 50.0
    with pytest.raises(ValueError, match="no reflections"):
        cifp.powder_pattern_from_cif(_cif(tmp_path / "n2.cif"), two_theta_min=1.0,
                                     two_theta_max=5.0, use_cache=False)


def test_wavelength_moves_the_lines_but_not_the_d_spacings(tmp_path):
    """d belongs to the structure; 2theta belongs to the experiment."""
    path = _cif(tmp_path / "nacl.cif")
    cu = cifp.powder_pattern_from_cif(path, wavelength=1.5406, two_theta_max=90, use_cache=False)
    mo = cifp.powder_pattern_from_cif(path, wavelength=0.7093, two_theta_max=90, use_cache=False)
    cu_by_d = {round(d, 6): tt for d, tt in zip(cu.d, cu.two_theta)}
    mo_by_d = {round(d, 6): tt for d, tt in zip(mo.d, mo.two_theta)}
    shared = sorted(set(cu_by_d) & set(mo_by_d))
    assert len(shared) >= 3          # the same planes, whatever the source
    for d in shared:
        # every shared plane sits at LOWER 2theta with the shorter wavelength
        assert mo_by_d[d] < cu_by_d[d]
        # d is the dict key, rounded to 6 dp, so the check is to 1e-4 deg
        assert mo_by_d[d] == pytest.approx(
            2 * np.degrees(np.arcsin(0.7093 / (2 * d))), abs=1e-4)
    # the same window reaches further into reciprocal space at short lambda,
    # so Mo returns MORE lines even though each one moved down
    assert len(mo.d) > len(cu.d)


def test_cache_returns_the_same_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(cifp, "CIF_PATTERN_CACHE_DIR", str(tmp_path / "cache"))
    path = _cif(tmp_path / "nacl.cif")
    first = cifp.powder_pattern_from_cif(path, two_theta_max=90)
    second = cifp.powder_pattern_from_cif(path, two_theta_max=90)
    assert np.allclose(first.d, second.d)
    assert np.allclose(first.intensity, second.intensity)
    assert first.hkl == second.hkl
    assert first.spacegroup == second.spacegroup


def test_list_cif_files_walks_subfolders(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    _cif(tmp_path / "a.cif")
    _cif(tmp_path / "sub" / "b.CIF")
    _cif(tmp_path / "sub" / "deeper" / "c.cif")
    (tmp_path / "notes.txt").write_text("not a cif")
    assert len(cifp.list_cif_files(str(tmp_path))) == 3
    assert len(cifp.list_cif_files(str(tmp_path), recursive=False)) == 1
    assert cifp.list_cif_files(str(tmp_path / "missing")) == []


def test_huge_cell_is_refused_with_a_usable_message(tmp_path):
    path = _cif(tmp_path / "huge.cif", a=900.0)
    with pytest.raises(ValueError, match="cell too large"):
        cifp.powder_pattern_from_cif(path, two_theta_max=90, use_cache=False)
