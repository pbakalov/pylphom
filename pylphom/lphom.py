import numpy as np
from scipy.optimize import linprog


def lphom(
    votes1: np.ndarray,
    votes2: np.ndarray,
    fixed_zeros: list[tuple[int, int]] | None = None,
    match_totals: bool = True,
) -> np.ndarray:
    """
    Estimate a global vote transfer matrix via linear programming.

    Finds the row-stochastic J×K matrix P minimising the L1 residual
    sum_i ||votes1[i] @ P - votes2[i]||_1 subject to P >= 0 and row sums = 1.

    Parameters
    ----------
    votes1 : ndarray, shape (N, J)
        Vote counts per precinct, election 1 (J parties / options).
    votes2 : ndarray, shape (N, K)
        Vote counts per precinct, election 2 (K parties / options).
    fixed_zeros : list of (j, k), optional
        Entries of P that are constrained to be exactly 0 (e.g. a
        structurally impossible transition such as new-voter -> removed-voter).
    match_totals : bool, default True
        Also require the estimated election-2 totals to equal the observed
        ones exactly: votes1.sum(0) @ P == votes2.sum(0). This is what the R
        lphom package does. It needs votes1.sum() == votes2.sum(); use
        `build_extended` to balance the totals, or pass False to only
        minimise the residual.

    Returns
    -------
    P : ndarray, shape (J, K)
        Row-stochastic transfer matrix.  P[j, k] is the estimated fraction
        of party-j voters at election 1 who voted for party k at election 2.
    """
    votes1 = np.asarray(votes1, dtype=float)
    votes2 = np.asarray(votes2, dtype=float)

    if votes1.ndim != 2 or votes2.ndim != 2:
        raise ValueError("votes1 and votes2 must be 2-D arrays")
    if votes1.shape[0] != votes2.shape[0]:
        raise ValueError("votes1 and votes2 must have the same number of rows (precincts)")

    N, J = votes1.shape
    K = votes2.shape[1]

    fixed_zeros = fixed_zeros or []
    for j, k in fixed_zeros:
        if not (0 <= j < J and 0 <= k < K):
            raise ValueError(f"fixed_zeros entry ({j}, {k}) out of bounds for P of shape ({J}, {K})")
    fixed_idx = {j * K + k for j, k in fixed_zeros}

    if match_totals and not np.isclose(votes1.sum(), votes2.sum()):
        raise ValueError(
            f"match_totals=True needs equal grand totals, got sum(votes1)={votes1.sum()} "
            f"and sum(votes2)={votes2.sum()}. Use build_extended to add new/removed "
            "voters, or pass match_totals=False."
        )

    # Decision variables (in order):
    #   p      : J*K  — entries of P, flattened row-major (P[j,k] = p[j*K + k])
    #   e_plus : N*K  — positive part of residual for each (precinct, party2)
    #   e_minus: N*K  — negative part of residual

    n_p = J * K
    n_e = N * K
    n_vars = n_p + 2 * n_e

    # Objective: minimise sum(e_plus) + sum(e_minus)
    c = np.zeros(n_vars)
    c[n_p : n_p + n_e] = 1.0
    c[n_p + n_e :] = 1.0

    # Equality constraints:
    #   (a) row-stochastic:  sum_k P[j, k] = 1  for each j         → J rows
    #   (b) residual def:    votes1[i] @ P[:, k] - e+[i,k] + e-[i,k] = votes2[i,k]
    #                        for each (i, k)                        → N*K rows
    #   (c) if match_totals: votes1.sum(0) @ P[:, k] = votes2.sum(0)[k]  → K rows

    n_eq = J + n_e + (K if match_totals else 0)
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    # (a) row-stochastic
    for j in range(J):
        A_eq[j, j * K : (j + 1) * K] = 1.0
    b_eq[:J] = 1.0

    # (b) residuals — build the X@P block via broadcasting to avoid the inner loop
    # For each k, the constraint matrix block over all precincts i is votes1
    for k in range(K):
        rows = J + np.arange(N) * K + k          # row indices in A_eq
        p_cols = np.arange(J) * K + k            # column indices for P[:,k]
        A_eq[np.ix_(rows, p_cols)] = votes1       # votes1[i, j] coefficient for P[j,k]
        A_eq[rows, n_p + np.arange(N) * K + k] = -1.0   # e_plus
        A_eq[rows, n_p + n_e + np.arange(N) * K + k] = 1.0  # e_minus
        b_eq[J + np.arange(N) * K + k] = votes2[:, k]

    # (c) aggregate totals
    if match_totals:
        xt = votes1.sum(axis=0)
        yt = votes2.sum(axis=0)
        for k in range(K):
            A_eq[J + n_e + k, np.arange(J) * K + k] = xt
            b_eq[J + n_e + k] = yt[k]

    # Bounds: P entries in [0, 1] (or fixed to 0); slack variables >= 0
    p_bounds = [(0.0, 0.0) if idx in fixed_idx else (0.0, 1.0) for idx in range(n_p)]
    bounds = p_bounds + [(0.0, None)] * (2 * n_e)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP solver did not converge: {result.message}")

    return result.x[:n_p].reshape(J, K)
