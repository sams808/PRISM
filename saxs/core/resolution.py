"""Pinhole resolution smearing with a per-point Gaussian width.

I_smeared(q_i) = integral of I(q') N(q' - q_i; sigma_i) dq', with sigma_i = dQ_i the Gaussian standard deviation of the
q resolution at point i (as written by most reduction pipelines and by canSAS "Qdev"). Evaluated by Gauss-Hermite
quadrature: q'_k = q_i + sqrt(2) sigma_i x_k, weights w_k / sqrt(pi). Nodes that fall at q' <= 0 are dropped and the
remaining weights renormalised (the Gaussian is truncated at q = 0; for dQ/q < 0.3 the dropped mass is < 0.1 %).

This is the standard sasmodels/SasView pinhole convention (a Gaussian in q per point). A real beam with broader tails
than a Gaussian is not represented: it is the caller's job to restrict the q range where that matters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_NODES = {}


def _hermgauss(n: int):
    if n not in _NODES:
        x, w = np.polynomial.hermite.hermgauss(n)
        _NODES[n] = (x, w / np.sqrt(np.pi))
    return _NODES[n]


def smear_points(func: Callable[[np.ndarray], np.ndarray], q: np.ndarray, dq: np.ndarray, n_nodes: int = 20) -> np.ndarray:
    """Gaussian-smeared values of `func` at `q` with per-point standard deviations `dq` (same units as q).

    `func` must be vectorised (it is called once on the flattened node grid). Points with dq <= 0 or non-finite
    dq are returned unsmeared.
    """
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    if dq.shape != q.shape:
        raise ValueError("dq must have the same shape as q")
    x, w = _hermgauss(int(n_nodes))
    good = np.isfinite(dq) & (dq > 0)
    out = np.empty_like(q)
    if (~good).any():
        out[~good] = func(q[~good])
    if good.any():
        qq = q[good][:, None] + np.sqrt(2.0) * dq[good][:, None] * x[None, :]
        ww = np.broadcast_to(w, qq.shape).copy()
        neg = qq <= 0
        ww[neg] = 0.0
        qq = np.where(neg, q[good][:, None], qq)          # any positive value; its weight is zero
        vals = np.asarray(func(qq.ravel()), dtype=float).reshape(qq.shape)
        out[good] = np.sum(ww * vals, axis=1) / np.sum(ww, axis=1)
    return out
