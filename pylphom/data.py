from typing import NamedTuple

import numpy as np
import pandas as pd


class ExtendedData(NamedTuple):
    X: np.ndarray
    Y: np.ndarray
    fixed_zeros: list[tuple[int, int]]
    row_labels: list[str]
    col_labels: list[str]
    groups: pd.DataFrame


def build_extended(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    new_voters_col: str = "new_voters",
    removed_voters_col: str = "removed_voters",
    group_by: str | list[str] | None = None,
    n_meta: int = 10,
) -> ExtendedData:
    """
    Build extended vote-count matrices that account for new voters entering
    the electorate and voters removed from the electorate (e.g. deaths,
    deregistrations) between two elections, so that totals balance per row.

    Column layout
    --------------
    In each dataframe, the *last* `n_meta` columns are treated as metadata
    (precinct/place identifiers, grouping columns, and the special
    `new_voters_col` / `removed_voters_col`). All other (leading) columns are
    treated as vote counts for parties/options — abstention is just another
    such column, with no special handling.

    `df1` must contain `new_voters_col` and `df2` must contain
    `removed_voters_col` among their last `n_meta` columns.

    The extended matrices add one category on each side:
      - X (election 1 / rows of P): party columns of df1, plus "new_voters"
      - Y (election 2 / cols of P): party columns of df2, plus "removed_voters"

    A person can't be both a new voter and a removed voter in the same
    transition, so P[new_voters_row, removed_voters_col] is fixed at 0
    (returned via `fixed_zeros`).

    For the resulting transportation problem (see `lphom_local`) to be
    feasible, the bookkeeping identity
        electorate2 = electorate1 - removed_voters + new_voters
    must hold for each row, i.e. sum(X row) == sum(Y row).

    Grouping
    --------
    Polling-place counts often differ between two elections, so `df1` and
    `df2` may have different numbers of rows. Pass `group_by` (a column
    name or list of column names present in the metadata of both
    dataframes, e.g. `["municipality_name", "place_name"]`) to aggregate
    both dataframes to a common geography before building X and Y. Groups
    that don't appear in both dataframes are dropped (with a warning).

    If `group_by` is None, `df1` and `df2` must already have the same
    number of rows and be row-aligned.

    Parameters
    ----------
    df1 : DataFrame
        Election-1 data: leading columns are party vote counts, last
        `n_meta` columns are metadata including `new_voters_col`.
    df2 : DataFrame
        Election-2 data: leading columns are party vote counts, last
        `n_meta` columns are metadata including `removed_voters_col`.
    new_voters_col : str, default "new_voters"
    removed_voters_col : str, default "removed_voters"
    group_by : str or list of str, optional
        Metadata column(s) to group both dataframes by before aligning.
    n_meta : int, default 10
        Number of trailing columns treated as metadata in each dataframe.

    Returns
    -------
    ExtendedData
        Named tuple with fields:
          X : ndarray, shape (M, J+1)
          Y : ndarray, shape (M, K+1)
          fixed_zeros : list[(J, K)]
          row_labels : list of str, length J+1 (party columns of df1 + "new_voters")
          col_labels : list of str, length K+1 (party columns of df2 + "removed_voters")
          groups : DataFrame, the group_by keys (or empty if group_by is None),
              one row per row of X/Y
    """
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    meta1 = list(df1.columns[-n_meta:])
    meta2 = list(df2.columns[-n_meta:])
    party_cols1 = list(df1.columns[:-n_meta])
    party_cols2 = list(df2.columns[:-n_meta])

    if new_voters_col not in meta1:
        raise ValueError(
            f"new_voters_col={new_voters_col!r} not found among the last {n_meta} columns of df1: {meta1}"
        )
    if removed_voters_col not in meta2:
        raise ValueError(
            f"removed_voters_col={removed_voters_col!r} not found among the last {n_meta} columns of df2: {meta2}"
        )

    if group_by is not None:
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        for col in group_cols:
            if col not in meta1:
                raise ValueError(f"group_by column {col!r} not found among the last {n_meta} columns of df1")
            if col not in meta2:
                raise ValueError(f"group_by column {col!r} not found among the last {n_meta} columns of df2")

        g1 = df1.groupby(group_cols, as_index=False)[party_cols1 + [new_voters_col]].sum()
        g2 = df2.groupby(group_cols, as_index=False)[party_cols2 + [removed_voters_col]].sum()

        # Avoid column-name collisions between g1 and g2 (e.g. shared party
        # names) by renaming g2's value columns before merging.
        y_cols = party_cols2 + [removed_voters_col]
        y_cols_renamed = [f"__y__{c}" for c in y_cols]
        g2 = g2.rename(columns=dict(zip(y_cols, y_cols_renamed)))

        merged = g1.merge(g2, on=group_cols, how="inner")

        n_dropped1 = g1.shape[0] - merged.shape[0]
        n_dropped2 = g2.shape[0] - merged.shape[0]
        if n_dropped1 or n_dropped2:
            import warnings

            warnings.warn(
                f"build_extended: dropped {n_dropped1} group(s) only in df1 and "
                f"{n_dropped2} group(s) only in df2 (groups must appear in both)."
            )

        groups = merged[group_cols].reset_index(drop=True)
        X = merged[party_cols1 + [new_voters_col]].to_numpy(dtype=float)
        Y = merged[y_cols_renamed].to_numpy(dtype=float)
    else:
        if len(df1) != len(df2):
            raise ValueError(
                "df1 and df2 have different numbers of rows and no group_by was given; "
                "either pass group_by to aggregate to a common geography, or "
                "pre-align the dataframes so rows correspond."
            )
        groups = pd.DataFrame(index=range(len(df1)))
        X = df1[party_cols1 + [new_voters_col]].to_numpy(dtype=float)
        Y = df2[party_cols2 + [removed_voters_col]].to_numpy(dtype=float)

    J = len(party_cols1)
    K = len(party_cols2)

    return ExtendedData(
        X=X,
        Y=Y,
        fixed_zeros=[(J, K)],
        row_labels=party_cols1 + ["new_voters"],
        col_labels=party_cols2 + ["removed_voters"],
        groups=groups,
    )
