import numpy as np
from pylphom import lphom, tslphom


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


class TestTslphom:
    def test_shapes_and_row_stochastic(self):
        rng = np.random.default_rng(0)
        P_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
        votes1, votes2 = _synthetic(P_true, [500] * 15, rng)

        result = tslphom(votes1, votes2)

        assert result.P.shape == (3, 3)
        assert result.P_local.shape == (15, 3, 3)
        np.testing.assert_allclose(result.P.sum(axis=1), 1.0, atol=1e-6)
        assert (result.P >= -1e-9).all()

    def test_recovers_known_matrix(self):
        rng = np.random.default_rng(42)
        P_true = np.array([[0.6, 0.4], [0.2, 0.8]])
        votes1, votes2 = _synthetic(P_true, [2000] * 100, rng)

        result = tslphom(votes1, votes2)

        np.testing.assert_allclose(result.P, P_true, atol=0.05)

    def test_fixed_zeros_enforced(self):
        votes1 = np.array([
            [50.0, 50.0, 10.0],
            [40.0, 60.0, 5.0],
        ])
        votes2 = np.array([
            [55.0, 50.0, 5.0],
            [42.0, 58.0, 5.0],
        ])
        assert np.allclose(votes1.sum(axis=1), votes2.sum(axis=1))

        fixed_zeros = [(2, 2)]
        result = tslphom(votes1, votes2, fixed_zeros=fixed_zeros)

        assert result.P[2, 2] == 0.0
        np.testing.assert_allclose(result.P_local[:, 2, 2], 0.0, atol=1e-9)
        np.testing.assert_allclose(result.P.sum(axis=1), 1.0, atol=1e-6)

    def test_zero_votes_for_a_party_falls_back_to_global(self):
        # party 1 (index 1) gets zero votes everywhere in election 1
        votes1 = np.array([
            [80.0, 0.0],
            [60.0, 0.0],
        ])
        votes2 = np.array([
            [50.0, 30.0],
            [40.0, 20.0],
        ])
        result = tslphom(votes1, votes2)
        P_global = lphom(votes1, votes2)

        np.testing.assert_allclose(result.P[1], P_global[1])
