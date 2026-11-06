from typing import NamedTuple

import numpy as np

from .lphom import lphom
from .lphom_local import lphom_local


class TslphomResult(NamedTuple):
    P: np.ndarray
    P_local: np.ndarray


def tslphom(
    votes1: np.ndarray,
    votes2: np.ndarray,
    fixed_zeros: list[tuple[int, int]] | None = None,
) -> TslphomResult:
    """
    Two-stage lphom.

    Stage 1: run `lphom` to obtain a global transfer matrix P_global.
    Stage 2: run `lphom_local` anchored to P_global to obtain a per-precinct
    local matrix for each precinct.

    The local matrices are then re-aggregated into a refined global matrix
    P: for each election-1 category j, P[j, :] is the average of
    P_local[:, j, :] weighted by votes1[:, j] (the number of category-j
    voters in each precinct). Since each P_local[i, j, :] is row-stochastic,
    this weighted average is also row-stochastic.

    Categories j with zero votes across all precincts fall back to
    P_global[j, :].

    Parameters
    ----------
    votes1 : ndarray, shape (N, J)
    votes2 : ndarray, shape (N, K)
    fixed_zeros : list of (j, k), optional
        Entries of P that are constrained to be exactly 0. Passed through to
        both `lphom` and `lphom_local`.

    Returns
    -------
    TslphomResult
        Named tuple with fields:
          P : ndarray, shape (J, K) — the re-aggregated global transfer matrix
          P_local : ndarray, shape (N, J, K) — per-precinct local matrices
    """
    votes1 = np.asarray(votes1, dtype=float)
    votes2 = np.asarray(votes2, dtype=float)

    P_global = lphom(votes1, votes2, fixed_zeros=fixed_zeros)
    P_local = lphom_local(votes1, votes2, P_global, fixed_zeros=fixed_zeros)

    J = votes1.shape[1]
    row_totals = votes1.sum(axis=0)  # shape (J,)

    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.einsum("ij,ijk->jk", votes1, P_local) / row_totals[:, None]

    zero_rows = row_totals == 0
    P[zero_rows] = P_global[zero_rows]

    return TslphomResult(P=P, P_local=P_local)
