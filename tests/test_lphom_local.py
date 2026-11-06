import numpy as np
import pytest
from pylphom import lphom, lphom_local


def _synthetic(P_true, precinct_totals, rng):
    """Generate synthetic, perfectly balanced election data from a known transfer matrix."""
    N = len(precinct_totals)
    J = P_true.shape[0]
    votes1 = np.array([
        rng.multinomial(n, [1 / J] * J) for n in precinct_totals
    ], dtype=float)
    votes2 = np.array([
        rng.multinomial(int(votes1[i].sum()), votes1[i] / votes1[i].sum() @ P_true)
        for i in range(N)
    ], dtype=float)
    return votes1, votes2


class TestLphomLocal:
    def test_shape_and_row_stochastic(self):
        rng = np.random.default_rng(0)
        P_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
        votes1, votes2 = _synthetic(P_true, [500] * 10, rng)
        P_global = lphom(votes1, votes2)
        P_local = lphom_local(votes1, votes2, P_global)

        assert P_local.shape == (10, 3, 3)
        np.testing.assert_allclose(P_local.sum(axis=2), 1.0, atol=1e-6)
        assert (P_local >= -1e-9).all()

    def test_recovers_reference_under_homogeneity(self):
        rng = np.random.default_rng(1)
        P_true = np.array([[0.8, 0.2], [0.3, 0.7]])
        votes1, votes2 = _synthetic(P_true, [2000] * 20, rng)
        P_global = lphom(votes1, votes2)
        P_local = lphom_local(votes1, votes2, P_global)

        for i in range(votes1.shape[0]):
            np.testing.assert_allclose(P_local[i], P_global, atol=0.05)

    def test_unbalanced_totals_raise(self):
        votes1 = np.array([[10.0, 5.0]])
        votes2 = np.array([[8.0, 5.0]])  # totals 15 vs 13
        P_ref = np.array([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="balanced"):
            lphom_local(votes1, votes2, P_ref)

    def test_zero_total_precinct_falls_back_to_reference(self):
        votes1 = np.array([[0.0, 0.0], [10.0, 10.0]])
        votes2 = np.array([[0.0, 0.0], [10.0, 10.0]])
        P_ref = np.array([[0.6, 0.4], [0.3, 0.7]])
        P_local = lphom_local(votes1, votes2, P_ref)
        np.testing.assert_allclose(P_local[0], P_ref)

    def test_fixed_zeros_enforced(self):
        # 3rd category in each side acts like "new_voters" / "removed_voters";
        # the (2,2) entry (new->removed) must stay zero.
        rng = np.random.default_rng(2)
        votes1 = np.array([
            [50.0, 50.0, 10.0],
            [40.0, 60.0, 5.0],
        ])
        votes2 = np.array([
            [55.0, 50.0, 5.0],
            [42.0, 58.0, 5.0],
        ])
        # totals must balance per row
        assert np.allclose(votes1.sum(axis=1), votes2.sum(axis=1))

        fixed_zeros = [(2, 2)]
        P_global = lphom(votes1, votes2, fixed_zeros=fixed_zeros)
        assert P_global[2, 2] == 0.0

        P_local = lphom_local(votes1, votes2, P_global, fixed_zeros=fixed_zeros)
        np.testing.assert_allclose(P_local[:, 2, 2], 0.0, atol=1e-9)
        np.testing.assert_allclose(P_local.sum(axis=2), 1.0, atol=1e-6)
