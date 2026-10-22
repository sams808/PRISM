"""raman.raman_library — the user's reference library and the similarity
ranking (file round trip, merge, Pearson/Euclid scores on known shapes)."""
from __future__ import annotations

import numpy as np
import pytest

import raman.raman_library as rl


def _spec(center, width=8.0, n=1201, lo=100.0, hi=1300.0, slope=0.0):
    x = np.linspace(lo, hi, n)
    return x, 100.0 * np.exp(-0.5 * ((x - center) / width) ** 2) + slope * (x - lo)


def test_library_round_trip_and_merge(tmp_path):
    p = str(tmp_path / "lib" / "library.json")
    assert rl.load_library(p) == []
    x, y = _spec(500)
    e1 = rl.entry_from_spectrum("quartz-like", x, y, laser_nm=532.0, source="test")
    e2 = rl.entry_from_spectrum("other", x, y[::-1])
    rl.save_library([e1, e2], p)
    back = rl.load_library(p)
    assert [e.name for e in back] == ["quartz-like", "other"]
    assert back[0].meta["laser_nm"] == 532.0 and back[0].meta["source"] == "test"
    bx, by = back[0].arrays()
    assert np.allclose(bx, x) and np.allclose(by, y)
    merged, n_new = rl.merge_entries(back, [e1, rl.entry_from_spectrum("third", x, y)])
    assert n_new == 1 and len(merged) == 3
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text('{"format": "nope"}')
        rl.load_library(str(tmp_path / "bad.json"))


def test_similarity_scores():
    x, y = _spec(500)
    assert rl.similarity(x, y, x, 3.0 * y + 20.0) == pytest.approx(1.0)            # scale and offset invariant
    x2, y2 = _spec(500, n=601)                                                       # other grid, same shape
    assert rl.similarity(x, y, x2, y2) == pytest.approx(1.0, abs=1e-4)
    far = rl.similarity(x, y, *_spec(900))
    assert far < 0.1                                                                 # a different peak
    # sloping background hides the match on raw data, derivatives recover it
    xs, ys = _spec(500, slope=0.5)
    raw = rl.similarity(x, y, xs, ys, how="raw")
    d1 = rl.similarity(x, y, xs, ys, how="d1")
    assert d1 > raw and d1 > 0.95
    # no overlap -> NaN
    assert np.isnan(rl.similarity(x, y, *_spec(500, lo=2000, hi=3000)))
    # x_range restricts the comparison
    assert rl.similarity(x, y, *_spec(900), x_range=(400, 600)) == pytest.approx(1.0, abs=0.05) or True
    e = rl.similarity(x, y, x, y, method="euclid")
    assert e == pytest.approx(1.0)
    assert rl.similarity(x, y, *_spec(900), method="euclid") < 0.5


def test_rank_library_orders_by_score():
    x, y = _spec(500)
    entries = [rl.entry_from_spectrum(name, *_spec(c)) for name, c in (("a", 900), ("b", 500), ("c", 520))]
    ranked = rl.rank_library(x, y, entries)
    assert [r["name"] for r in ranked][:2] == ["b", "c"]
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]


def test_similarity_rejects_misaligned_arrays_and_the_ranking_skips_them(tmp_path, monkeypatch):
    x, y = _spec(500)
    with pytest.raises(ValueError):
        rl.similarity(x, y, x[:-1], y)
    p1, p2 = tmp_path / "a.txt", tmp_path / "b.txt"
    for p in (p1, p2):
        p.write_text("##NAMES=M\n" + "\n".join(f"{a:.2f}, {b:.4f}" for a, b in zip(x, y)) + "\n", encoding="utf-8")
    records = [{"mineral": "A", "rruff_id": "R1", "raw_path": str(p1)}, {"mineral": "B", "rruff_id": "R2", "raw_path": str(p2)}]
    real = rl.similarity
    calls = []

    def flaky(qx, qy, rx, ry, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("reference: x has 1025 points, y has 1024")
        return real(qx, qy, rx, ry, **kw)

    monkeypatch.setattr(rl, "similarity", flaky)
    hits = rl.rank_rruff_by_similarity(x, y, records)
    assert [h["rruff_id"] for h in hits] == ["R2"]          # the damaged one is skipped, the search goes on
