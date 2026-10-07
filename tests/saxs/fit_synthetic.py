"""Synthetic series for the fit-workflow tests: two references (grain power laws with different exponents + a flat
level, as two batches of a reference material), a featureless curve, a Teubner-Strey structure and an absolute
sphere population (known contrast). Noise is multiplicative Gaussian (2 %); sigma is 3 % (a total uncertainty)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from saxs.core.fit_workflow import CurveInput, WorkflowOptions, run_workflow
from saxs.core.models import SphereSchulz, TeubnerStrey

Q = np.logspace(np.log10(0.0026), np.log10(0.7), 60)
TRUE = {"B_struct": {"ts_d": 500.0, "ts_xi": 200.0}, "A_sph": {"sph_R": 120.0, "sph_sig": 0.2, "sph_phi": 0.01}}
CONTRAST = 8.0


def intensities(seed: int = 5) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = Q
    out = {
        "refA": 1.0e-8 * q ** -4.15 + 0.05,
        "refB": 1.3e-8 * q ** -4.25 + 0.05,
        "A_plain": 1.4e-8 * q ** -4.15 + 0.30,
        "B_struct": 1.0e-8 * q ** -4.25 + 0.40 + TeubnerStrey().eval(q, S=400.0, d=500.0, xi=200.0),
        "A_sph": 1.2e-8 * q ** -4.15 + 0.30 + SphereSchulz().eval(q, phi=0.01, R=120.0, sig=0.2, drho=CONTRAST),
    }
    return {k: v * (1 + rng.normal(0, 0.02, q.size)) for k, v in out.items()}


REFERENCES = {"A_plain": "refA", "B_struct": "refB", "A_sph": "refA"}


def inputs(seed: int = 5) -> List[CurveInput]:
    out = []
    for name, I in intensities(seed).items():
        out.append(CurveInput(name=name, q=Q, I=I, sigma=0.03 * I, sigma_stat=0.015 * I, dq=0.04 * Q,
                              smeared_flag=Q < 0.0035, reference=REFERENCES.get(name),
                              contrast=CONTRAST if name == "A_sph" else None, tags={"group": name[0]}))
    return out


def fast_options(**kw) -> WorkflowOptions:
    o = dict(families=("teubner_strey", "sphere"), second_population=False, highq=("flat",), prescreen=3)
    o.update(kw)
    return WorkflowOptions(**o)


@lru_cache(maxsize=2)
def small_result(seed: int = 5):
    return run_workflow(inputs(seed), fast_options())


def write_tabular(folder: str, seed: int = 5) -> List[str]:
    """The series as named-column files (q_A, I_cm, sigma_total_cm, sigma_stat_cm, dQ_A, smeared flag, text column)."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, I in intensities(seed).items():
        p = os.path.join(folder, f"{name}_merged.dat")
        pd.DataFrame({"q_A": Q, "I_cm": I, "sigma_total_cm": 0.03 * I, "sigma_stat_cm": 0.015 * I, "dQ_A": 0.04 * Q,
                      "smeared_gt5pct": (Q < 0.0035).astype(int), "configurations": ["X"] * Q.size}).to_csv(p, sep="\t", index=False)
        paths.append(p)
    return paths


def curves(seed: int = 5):
    """The series as saxs.core.curve.Curve objects carrying dQ, the statistical sigma and the smeared flags the way
    saxs.core.loader.load_tabular_curve stores them."""
    from saxs.core.curve import Curve

    out = []
    for name, I in intensities(seed).items():
        out.append(Curve(q=Q, intensity=I, sigma=0.03 * I, name=name,
                         metadata={"dq": 0.04 * Q, "columns": {"sigma_stat_cm": 0.015 * I,
                                                               "smeared_gt5pct": (Q < 0.0035).astype(int)}}))
    return out
