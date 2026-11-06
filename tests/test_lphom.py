import numpy as np
import pytest
from pylphom import lphom


def _synthetic(P_true, precinct_totals, rng):
    """Generate synthetic election data from a known transfer matrix."""
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


class TestLphomProperties:
    def test_row_stochastic(self):
        rng = np.random.default_rng(0)
        P_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
        votes1, votes2 = _synthetic(P_true, [500] * 30, rng)
        P = lphom(votes1, votes2)
        assert P.shape == (3, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert (P >= -1e-9).all()

    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        P_true = np.array([[0.8, 0.2], [0.3, 0.7]])
        votes1, votes2 = _synthetic(P_true, [1000] * 20, rng)
        P = lphom(votes1, votes2)
        assert (P >= -1e-9).all()

    def test_recovers_known_matrix(self):
        # With a perfectly homogeneous dataset and many precincts the LP should
        # recover the true matrix closely.
        rng = np.random.default_rng(42)
        P_true = np.array([[0.6, 0.4], [0.2, 0.8]])
        votes1, votes2 = _synthetic(P_true, [2000] * 100, rng)
        P = lphom(votes1, votes2)
        np.testing.assert_allclose(P, P_true, atol=0.05)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            lphom(np.ones((5, 3)), np.ones((6, 2)))  # mismatched N

        with pytest.raises(ValueError):
            lphom(np.ones(10), np.ones(10))  # 1-D inputs
