"""Compare against the R lphom package (v0.3.5-7, lp_solve solver).

Data: 2026-10-25 vs 2026-11-01 presidential rounds, Bulgaria, by municipality
(see r_comparison/). R was run with new_and_exit_voters="raw" and
structural_zeros=list(c(10, 7)).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pylphom import lphom, tslphom

DATA = Path(__file__).parent / "data" / "r_reference"
X = pd.read_csv(DATA / "X.csv").to_numpy(float)
Y = pd.read_csv(DATA / "Y.csv").to_numpy(float)
FIXED_ZEROS = [(9, 6)]


def _r(name):
    return pd.read_csv(DATA / name, index_col=0)


def test_lphom_matches_r():
    P = lphom(X, Y, fixed_zeros=FIXED_ZEROS)
    np.testing.assert_allclose(P, _r("R_lphom.csv").to_numpy(), atol=1e-6)


def test_tslphom_matches_r():
    # Some local LPs have several optimal solutions even after the tie-break,
    # and lp_solve and HiGHS pick different ones. On this data that only
    # shows up in the small "invalid" row (and a little in "other").
    R = _r("R_tslphom.csv")
    P = pd.DataFrame(tslphom(X, Y, fixed_zeros=FIXED_ZEROS).P, index=R.index, columns=R.columns)
    diff = (P - R).abs()
    assert diff.to_numpy().max() < 0.1
    assert diff.drop(index=["invalid", "other"]).to_numpy().max() < 1e-3
