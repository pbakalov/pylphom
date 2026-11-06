import numpy as np
import pandas as pd
import pytest

from pylphom import build_extended, lphom, lphom_local


class TestBuildExtendedNoGrouping:
    def test_row_aligned(self):
        df1 = pd.DataFrame({
            "party_A": [40.0],
            "party_B": [40.0],
            "place_name": ["P1"],
            "new_voters": [10.0],
        })
        df2 = pd.DataFrame({
            "party_A": [45.0],
            "party_B": [40.0],
            "place_name": ["P1"],
            "removed_voters": [5.0],
        })

        result = build_extended(df1, df2, n_meta=2)

        np.testing.assert_array_equal(result.X, [[40.0, 40.0, 10.0]])
        np.testing.assert_array_equal(result.Y, [[45.0, 40.0, 5.0]])
        assert result.fixed_zeros == [(2, 2)]
        assert result.row_labels == ["party_A", "party_B", "new_voters"]
        assert result.col_labels == ["party_A", "party_B", "removed_voters"]
        # totals balance per row
        np.testing.assert_allclose(result.X.sum(axis=1), result.Y.sum(axis=1))

    def test_mismatched_rows_without_group_by_raises(self):
        df1 = pd.DataFrame({
            "party_A": [40.0, 30.0],
            "place_name": ["P1", "P2"],
            "new_voters": [10.0, 5.0],
        })
        df2 = pd.DataFrame({
            "party_A": [45.0],
            "place_name": ["P1"],
            "removed_voters": [5.0],
        })
        with pytest.raises(ValueError, match="different numbers of rows"):
            build_extended(df1, df2, n_meta=2)

    def test_missing_new_voters_col_raises(self):
        df1 = pd.DataFrame({
            "party_A": [40.0],
            "place_name": ["P1"],
            "other_meta": [0.0],
        })
        df2 = pd.DataFrame({
            "party_A": [45.0],
            "place_name": ["P1"],
            "removed_voters": [5.0],
        })
        with pytest.raises(ValueError, match="new_voters_col"):
            build_extended(df1, df2, n_meta=2)


class TestBuildExtendedGrouping:
    def _dfs(self):
        df1 = pd.DataFrame({
            "party_A": [20.0, 20.0, 60.0, 10.0],
            "party_B": [20.0, 20.0, 60.0, 10.0],
            "place_id": [1, 2, 3, 4],
            "precinct_name": ["P1", "P2", "P3", "P4"],
            "municipality": ["M1", "M1", "M2", "M3"],
            "new_voters": [5.0, 5.0, 20.0, 2.0],
        })
        df2 = pd.DataFrame({
            "party_A": [45.0, 65.0],
            "party_B": [40.0, 60.0],
            "place_id": [10, 20],
            "precinct_name": ["Q1", "Q2"],
            "municipality": ["M1", "M2"],
            "removed_voters": [5.0, 15.0],
        })
        return df1, df2

    def test_grouping_aggregates_and_aligns(self):
        df1, df2 = self._dfs()
        with pytest.warns(UserWarning, match="dropped"):
            result = build_extended(df1, df2, group_by="municipality", n_meta=4)

        # M3 (only in df1) is dropped; M1, M2 remain
        assert result.groups["municipality"].tolist() == ["M1", "M2"]
        np.testing.assert_array_equal(result.X, [[40.0, 40.0, 10.0], [60.0, 60.0, 20.0]])
        np.testing.assert_array_equal(result.Y, [[45.0, 40.0, 5.0], [65.0, 60.0, 15.0]])
        np.testing.assert_allclose(result.X.sum(axis=1), result.Y.sum(axis=1))

    def test_group_by_column_not_in_metadata_raises(self):
        df1, df2 = self._dfs()
        with pytest.raises(ValueError, match="group_by column"):
            build_extended(df1, df2, group_by="party_A", n_meta=4)


class TestEndToEnd:
    def test_extended_data_feeds_lphom_and_local(self):
        df1, df2 = TestBuildExtendedGrouping()._dfs()
        with pytest.warns(UserWarning):
            result = build_extended(df1, df2, group_by="municipality", n_meta=4)

        P_global = lphom(result.X, result.Y, fixed_zeros=result.fixed_zeros)
        assert P_global.shape == (3, 3)
        np.testing.assert_allclose(P_global.sum(axis=1), 1.0, atol=1e-6)
        assert P_global[2, 2] == 0.0  # new_voters -> removed_voters fixed at 0

        P_local = lphom_local(result.X, result.Y, P_global, fixed_zeros=result.fixed_zeros)
        np.testing.assert_allclose(P_local.sum(axis=2), 1.0, atol=1e-6)
        np.testing.assert_allclose(P_local[:, 2, 2], 0.0, atol=1e-9)
