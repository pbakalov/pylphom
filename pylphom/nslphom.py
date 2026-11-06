from typing import NamedTuple

import numpy as np

from .lphom import lphom
from .lphom_local import lphom_local


class NslphomResult(NamedTuple):
    P: np.ndarray
    P_local: np.ndarray
    hi_history: list[float]


def _reaggregate(votes1: np.ndarray, P_local: np.ndarray, P_fallback: np.ndarray) -> np.ndarray:
    row_totals = votes1.sum(axis=0)  # (J,)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.einsum("ij,ijk->jk", votes1, P_local) / row_totals[:, None]
    P[row_totals == 0] = P_fallback[row_totals == 0]
    return P


def _heterogeneity_index(votes1: np.ndarray, P_local: np.ndarray, P: np.ndarray) -> float:
    # Weighted mean absolute deviation of local matrices from the global one.
    # Shape: (N, J, K); weights: votes1 (N, J).
    diff = np.abs(P_local - P[None, :, :])           # (N, J, K)
    weighted = np.einsum("ij,ijk->", votes1, diff)    # scalar
    total = votes1.sum()
    return float(weighted / total) if total > 0 else 0.0


def nslphom(
    votes1: np.ndarray,
    votes2: np.ndarray,
    fixed_zeros: list[tuple[int, int]] | None = None,
    max_iter: int = 10,
) -> NslphomResult:
    """
    Non-linear sequential lphom (nslphom).

    Extends tslphom by iterating the local-estimation and re-aggregation
    stages. After each iteration the re-aggregated global matrix becomes the
    new anchor for the next round of `lphom_local`, allowing the local
    matrices to converge toward a self-consistent solution.

    A heterogeneity index (HI) — the votes-weighted mean absolute deviation
    of the per-precinct local matrices from the global matrix — is computed
    after each iteration. The iteration with the lowest HI is returned.

    Parameters
    ----------
    votes1 : ndarray, shape (N, J)
    votes2 : ndarray, shape (N, K)
    fixed_zeros : list of (j, k), optional
        Entries of P constrained to exactly 0. Passed through to both
        `lphom` and `lphom_local`.
    max_iter : int, default 10
        Maximum number of lphom_local + re-aggregation iterations after
        the initial global lphom solve.

    Returns
    -------
    NslphomResult
        Named tuple with fields:
          P          : ndarray, shape (J, K) — global matrix of best iteration
          P_local    : ndarray, shape (N, J, K) — local matrices of best iteration
          hi_history : list of float, length max_iter — HI after each iteration
    """
    votes1 = np.asarray(votes1, dtype=float)
    votes2 = np.asarray(votes2, dtype=float)

    P_current = lphom(votes1, votes2, fixed_zeros=fixed_zeros)

    best_P = None
    best_P_local = None
    best_hi = np.inf
    hi_history = []

    for _ in range(max_iter):
        P_local = lphom_local(votes1, votes2, P_current, fixed_zeros=fixed_zeros)
        P_new = _reaggregate(votes1, P_local, P_current)
        hi = _heterogeneity_index(votes1, P_local, P_new)
        hi_history.append(hi)

        if hi < best_hi:
            best_hi = hi
            best_P = P_new
            best_P_local = P_local

        P_current = P_new

    return NslphomResult(P=best_P, P_local=best_P_local, hi_history=hi_history)
