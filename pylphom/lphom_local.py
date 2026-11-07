import warnings

import numpy as np
from scipy.optimize import linprog


def lphom_local(
    votes1: np.ndarray,
    votes2: np.ndarray,
    P_ref: np.ndarray,
    fixed_zeros: list[tuple[int, int]] | None = None,
    tiebreak: bool = True,
) -> np.ndarray:
    """
    Estimate a precinct-level transfer matrix for each precinct, anchored to
    a reference matrix (typically the global lphom solution).

    For each precinct i, solves a transportation-style LP for the transfer
    count matrix N_i (J x K):
      - row sums of N_i equal votes1[i]
      - column sums of N_i equal votes2[i]
      - N_i >= 0
    minimising sum_{j,k} |N_i[j,k] - votes1[i,j] * P_ref[j,k]|.

    This requires sum(votes1[i]) == sum(votes2[i]) for every precinct (the
    transportation problem is only feasible when the totals balance). When
    working with raw party vote counts that may differ due to turnout
    changes, build extended matrices that account for abstention, new
    voters, and removed voters first (see `build_extended`), so that the
    totals balance by construction.

    P_local[i] is N_i normalised by its row sums; rows where votes1[i,j] == 0
    fall back to P_ref[j] (that party has no voters in the precinct, so the
    row is irrelevant for aggregation but kept for shape consistency).

    Parameters
    ----------
    votes1 : ndarray, shape (N, J)
    votes2 : ndarray, shape (N, K)
    P_ref : ndarray, shape (J, K)
        Row-stochastic reference transfer matrix (e.g. from `lphom`).
    fixed_zeros : list of (j, k), optional
        Entries of P (and correspondingly N) that are constrained to be
        exactly 0. Should match the `fixed_zeros` used to compute `P_ref`.
    tiebreak : bool, default True
        The LP above often has many equally good solutions. If True, solve a
        second LP that picks, among them, the one whose proportions are
        closest to P_ref: minimise sum_{j,k} |P_i[j,k] - P_ref[j,k]|. This is
        what the R lphom package does. If False, return whichever optimal
        solution the solver finds.

    Returns
    -------
    P_local : ndarray, shape (N, J, K)
    """
    votes1 = np.asarray(votes1, dtype=float)
    votes2 = np.asarray(votes2, dtype=float)
    P_ref = np.asarray(P_ref, dtype=float)

    if votes1.ndim != 2 or votes2.ndim != 2:
        raise ValueError("votes1 and votes2 must be 2-D arrays")
    if votes1.shape[0] != votes2.shape[0]:
        raise ValueError("votes1 and votes2 must have the same number of rows (precincts)")
    if P_ref.shape != (votes1.shape[1], votes2.shape[1]):
        raise ValueError("P_ref must have shape (J, K) matching votes1/votes2 columns")

    N, J = votes1.shape
    K = votes2.shape[1]

    P_local = np.empty((N, J, K))

    for i in range(N):
        x = votes1[i]
        y = votes2[i]

        if not np.isclose(x.sum(), y.sum()):
            raise ValueError(
                f"precinct {i}: sum(votes1)={x.sum()} != sum(votes2)={y.sum()}; "
                "lphom_local requires balanced totals per precinct. Use "
                "build_extended to account for abstention, new voters, and "
                "removed voters so that electorate2 = electorate1 - "
                "removed_voters + new_voters holds exactly."
            )

        if x.sum() == 0:
            P_local[i] = P_ref
            continue

        N_i = _local_lp(x, y, P_ref, fixed_zeros, tiebreak)

        with np.errstate(invalid="ignore", divide="ignore"):
            P_i = N_i / x[:, None]
        zero_rows = x == 0
        P_i[zero_rows] = P_ref[zero_rows]
        P_local[i] = P_i

    return P_local


def _local_lp(
    x: np.ndarray,
    y: np.ndarray,
    P_ref: np.ndarray,
    fixed_zeros: list[tuple[int, int]] | None = None,
    tiebreak: bool = True,
) -> np.ndarray:
    """
    Solve the transportation linear program (LP) for a single precinct.

    Finds the J x K transfer count matrix N minimising
    sum_{j,k} |N[j,k] - x[j] * P_ref[j,k]| subject to:
      - sum_k N[j,k] = x[j] for each j (row sums match election-1 votes)
      - sum_j N[j,k] = y[k] for each k (column sums match election-2 votes)
      - N >= 0

    Parameters
    ----------
    x : ndarray, shape (J,)
        Election-1 vote counts for this precinct.
    y : ndarray, shape (K,)
        Election-2 vote counts for this precinct. Must satisfy
        sum(y) == sum(x).
    P_ref : ndarray, shape (J, K)
        Reference transfer matrix anchoring the L1 objective.
    fixed_zeros : list of (j, k), optional
        Entries of N that are constrained to be exactly 0.
    tiebreak : bool, default True
        See `lphom_local`.

    Returns
    -------
    N : ndarray, shape (J, K)
        Estimated transfer count matrix.
    """
    J = x.shape[0]
    K = y.shape[0]

    # Variables: N (J*K) + e_plus (J*K) + e_minus (J*K), flattened row-major.
    n_n = J * K
    n_vars = 3 * n_n

    c = np.zeros(n_vars)
    c[n_n : 2 * n_n] = 1.0  # e_plus
    c[2 * n_n :] = 1.0      # e_minus

    target = (x[:, None] * P_ref).ravel()  # votes1[j] * P_ref[j,k]

    # Equality constraints:
    #   (1) N[j,k] - e+[j,k] + e-[j,k] = target[j,k]   → J*K rows
    #   (2) sum_k N[j,k] = x[j]                         → J rows
    #   (3) sum_j N[j,k] = y[k]                         → K rows
    n_eq = n_n + J + K
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    # (1)
    idx = np.arange(n_n)
    A_eq[idx, idx] = 1.0
    A_eq[idx, n_n + idx] = -1.0
    A_eq[idx, 2 * n_n + idx] = 1.0
    b_eq[idx] = target

    # (2) row sums
    for j in range(J):
        A_eq[n_n + j, j * K : (j + 1) * K] = 1.0
        b_eq[n_n + j] = x[j]

    # (3) column sums
    for k in range(K):
        A_eq[n_n + J + k, np.arange(J) * K + k] = 1.0
        b_eq[n_n + J + k] = y[k]

    fixed_idx = {j * K + k for j, k in (fixed_zeros or [])}
    n_bounds = [(0.0, 0.0) if idx in fixed_idx else (0.0, None) for idx in range(n_n)]
    bounds = n_bounds + [(0.0, None)] * (2 * n_n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP solver did not converge for precinct: {result.message}")

    if not tiebreak:
        return result.x[:n_n].reshape(J, K)

    # Stage 2: keep the stage-1 error at its optimum and, among those
    # solutions, minimise sum |N[j,k]/x[j] - P_ref[j,k]|. That is the same
    # residual e+/e- divided by x[j] (rows with x[j] == 0 are all zero anyway).
    with np.errstate(divide="ignore"):
        w = np.where(x > 0, 1.0 / x, 0.0)
    w = np.repeat(w, K)
    c2 = np.zeros(n_vars)
    c2[n_n : 2 * n_n] = w
    c2[2 * n_n :] = w
    A_ub = c[None, :]
    b_ub = [result.fun * (1 + 1e-9) + 1e-7]
    result2 = linprog(c2, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result2.success:
        warnings.warn(f"tie-break LP failed ({result2.message}); using the stage-1 solution")
        return result.x[:n_n].reshape(J, K)

    return result2.x[:n_n].reshape(J, K)
