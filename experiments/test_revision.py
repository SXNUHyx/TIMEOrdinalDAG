"""Regression checks for reviewer-driven changes; no fabricated study results."""
import unittest
import numpy as np
from data import windows, subject_split, fit_maps, encode
from simulations import generate, oracle, metrics
from osem import mask, oce, search, acyclic, TemporalOSEM


class RevisionTests(unittest.TestCase):
    def test_no_windows_cross_subject_or_segment(self):
        x = np.arange(30).reshape(15, 2)
        rows, ids = windows([[x, x + 100], [x + 200]], 2)
        self.assertEqual(len(rows), 39)
        np.testing.assert_array_equal(rows[0], np.r_[x[2], x[1], x[0]])
        self.assertTrue(np.all(np.ptp(rows, axis=1) < 20))
        self.assertEqual(set(ids), {0, 1})

    def test_independent_mapping_and_stationary_cuts(self):
        subjects, _ = generate(S=20, T=40, p=2, seed=17)
        groups, _, split = subject_split(subjects, 11, calibrate=False)
        self.assertFalse(set(split[0]) & set(split[1]))
        self.assertFalse(set(split[0]) & set(split[2]))
        x, _ = windows(groups[0], 2)
        cuts, _ = fit_maps(x, 2, 2, 4, 'cart')
        values = np.tile(np.array([.3, -.7]), 3)[None, :]
        z = encode(values, cuts)
        np.testing.assert_array_equal(z[:, :2], z[:, 2:4])
        np.testing.assert_array_equal(z[:, :2], z[:, 4:])

    def test_lag_one_effect_is_not_accidentally_null(self):
        _, meta = generate(S=5, T=30, p=3, seed=1)
        cuts = [np.array([-.5, 0., .5])] * 3
        effect = oracle(meta, cuts)
        self.assertGreater(effect[-1], .01)
        self.assertAlmostEqual(effect.sum(), 0., places=6)
        meta['heterogeneity'] = .5
        self.assertIsNone(oracle(meta, cuts))

    def test_temporal_order_and_zero_effect(self):
        allowed = mask(2, 2)
        self.assertTrue(allowed[4, 0])
        self.assertFalse(allowed[0, 4])
        self.assertFalse(mask(2, 2, False)[0, 1])
        unconstrained = TemporalOSEM(2, 1, temporal=False, contemporaneous=False)
        self.assertFalse(unconstrained.allowed[0, 1])
        self.assertTrue(unconstrained.allowed[0, 2])
        b = np.array([[0., 0.], [.4, 0.]])
        cuts = [np.array([-np.inf, 0., np.inf])] * 2
        np.testing.assert_array_equal(oce(b, np.ones(2), cuts, 1, 0), [0., 0.])

    def test_search_does_not_mutate_saved_candidate(self):
        # Seed 42 produced a score/graph mismatch with the supplied aliasing bug.
        rng = np.random.default_rng(42)
        covariance = np.corrcoef(rng.normal(size=(10, 5)), rowvar=False)
        graph, score, _ = search(covariance, 100, .1, ~np.eye(5, dtype=bool))
        actual = 0.
        for j in range(5):
            parents = np.flatnonzero(graph[:, j])
            residual = covariance[j, j]
            if len(parents):
                residual -= covariance[j, parents] @ np.linalg.solve(
                    covariance[np.ix_(parents, parents)], covariance[parents, j])
            actual += -50*np.log(max(residual, 1e-8)) - .05*np.log(100)*len(parents)
        self.assertTrue(acyclic(graph))
        self.assertAlmostEqual(actual, score, places=9)

    def test_calibration_never_uses_later_segments(self):
        first = np.arange(20., dtype=float)[:, None]
        later = np.full((80, 1), 1000.)
        subjects = [[first, later] for _ in range(10)]
        groups, _, _ = subject_split(subjects, 0, calibrate=True)
        transformed = groups[0][0]
        self.assertEqual(len(transformed), 1)
        np.testing.assert_allclose(transformed[0], (later-first.mean(0))/first.std(0))

    def test_noise_sensitivity_is_not_only_a_global_scale(self):
        low, _ = generate(S=5, T=30, p=3, noise=.5, seed=7)
        high, _ = generate(S=5, T=30, p=3, noise=2., seed=7)
        np.testing.assert_array_equal(low[0][0][:, 0], high[0][0][:, 0])
        self.assertFalse(np.allclose(low[0][0][:, -1], high[0][0][:, -1]))

    def test_stationary_oracle_matches_analytic_binary_contrast(self):
        # With rho=0, X0(t) is iid N(0,1); X1(t)=.3 X0(t-1)
        # +.2 X0(t-2)+epsilon(t). The sign-interval contrast has an
        # exact bivariate-normal arcsine expression, independent of our integral.
        for noise in [.5, 1., 2.]:
            _, meta = generate(S=5, T=10, p=2, noise=noise, rho=0., seed=5)
            result = oracle(meta, [np.array([0.]), np.array([0.])])
            expected = 2*np.arcsin(.3/np.sqrt(noise**2+.3**2+.2**2))/np.pi
            np.testing.assert_allclose(result, [-expected, expected], atol=1e-7)

    def test_contemporaneous_shd_counts_a_reversal_once(self):
        truth = dict(scenario='contemporaneous', A=np.zeros((2, 2, 2)),
                     B0=np.array([[0., 0.], [.3, 0.]]))
        predicted = np.array([[False, False], [True, False]])
        result = metrics(predicted, truth, 0, 2)
        self.assertEqual(result['contemp_SHD'], 1)
        self.assertEqual(result['contemp_TPR'], 0.)

    def test_heldout_loss_matches_independent_binary_probability(self):
        m = TemporalOSEM(2, 0)
        m.Sigma = np.eye(2)
        m.cuts = [np.array([-np.inf, 0., np.inf])] * 2
        m.levels = [2, 2]
        observations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertAlmostEqual(m.pair_logloss(observations), np.log(4.), places=7)


if __name__ == '__main__':
    unittest.main()
