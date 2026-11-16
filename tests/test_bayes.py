import math
import unittest

from kanm.bayes import TrustDistribution, update_belief


class BayesianUpdateTests(unittest.TestCase):
    def test_uniform_reliability_is_neutral(self):
        trust = TrustDistribution.uniform(points=101)
        self.assertAlmostEqual(trust.expected_reliability, 0.5)
        self.assertAlmostEqual(update_belief(0.37, trust, True), 0.37)
        self.assertAlmostEqual(update_belief(0.37, trust, False), 0.37)

    def test_beta_grid_preserves_mean_and_variance(self):
        # Moment-preserving discretization: the mean is exact at any
        # resolution, including J/U-shaped priors (alpha or beta < 1), which
        # v0.2.0 biased badly (e.g. requested 0.99 at c = 1 gave 0.85).
        for mean, concentration, points in ((0.6, 8, 21), (0.6, 8, 201), (0.99, 1, 201),
                                            (0.9, 0.2, 201), (0.5, 0.01, 51), (0.99, 4, 101)):
            trust = TrustDistribution.beta_prior(mean, concentration, points)
            self.assertAlmostEqual(trust.expected_reliability, mean, places=10)
            self.assertAlmostEqual(trust.expected_reliability + trust.expected_unreliability, 1.0)
            self.assertTrue(all(0.0 < x < 1.0 for x in trust.grid))
        fine = TrustDistribution.beta_prior(0.6, 8, 401)
        variance = sum((x - 0.6) ** 2 * w for x, w in zip(fine.grid, fine.weights))
        self.assertAlmostEqual(variance, 0.6 * 0.4 / 9, places=4)

    def test_incomplete_beta_matches_closed_forms(self):
        from kanm.bayes import _beta_cdf_pair
        cases = [(1, 1, 0.3, 0.3), (2, 1, 0.4, 0.16), (1, 3, 0.2, 1 - 0.8 ** 3),
                 (0.5, 0.5, 0.25, 2 / math.pi * math.asin(0.5))]
        for a, b, x, expected in cases:
            lower, upper = _beta_cdf_pair(a, b, x)
            self.assertAlmostEqual(lower, expected, places=12)
            self.assertAlmostEqual(lower + upper, 1.0, places=12)

    def test_trust_cannot_collapse_to_certainty(self):
        # v0.2.0 kept atoms at r = 0 and r = 1; thousands of consistent
        # reports drove E[r] to exactly 1.0 and a later contrary report crashed.
        trust = TrustDistribution.beta_prior(0.6, 8, points=201)
        for _ in range(20000):
            trust = trust.observe(1.0, True)
        self.assertLess(trust.expected_reliability, 1.0)
        self.assertGreater(trust.expected_unreliability, 0.0)
        self.assertTrue(math.isfinite(update_belief(0.99, trust, False)))

    def test_trusted_positive_report_increases_belief(self):
        trust = TrustDistribution.beta_prior(mean=0.8, concentration=30, points=501)
        self.assertGreater(update_belief(0.4, trust, True), 0.4)

    def test_untrusted_positive_report_decreases_belief(self):
        trust = TrustDistribution.beta_prior(mean=0.2, concentration=30, points=501)
        self.assertLess(update_belief(0.4, trust, True), 0.4)

    def test_positive_and_negative_are_symmetric(self):
        trust = TrustDistribution.beta_prior(mean=0.73, concentration=15, points=301)
        positive = update_belief(0.28, trust, True)
        swapped_negative = 1.0 - update_belief(1.0 - 0.28, trust, False)
        self.assertAlmostEqual(positive, swapped_negative, places=12)

    def test_trust_observation_preserves_normalization_and_changes_mean(self):
        trust = TrustDistribution.beta_prior(mean=0.7, concentration=8, points=301)
        posterior = trust.observe(0.8, positive=True)
        self.assertAlmostEqual(sum(posterior.weights), 1.0)
        self.assertGreater(posterior.expected_reliability, trust.expected_reliability)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            update_belief(1.2, TrustDistribution.uniform(11), True)


if __name__ == "__main__":
    unittest.main()
