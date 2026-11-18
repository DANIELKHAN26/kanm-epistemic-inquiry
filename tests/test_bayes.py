import math
import unittest

from kanm.bayes import TrustDistribution, _beta_cdf_pair, log_likelihood_ratio, update_belief
from kanm.laputa16 import Laputa16Trust


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
        cases = [(1, 1, 0.3, 0.3), (2, 1, 0.4, 0.16), (1, 3, 0.2, 1 - 0.8 ** 3),
                 (0.5, 0.5, 0.25, 2 / math.pi * math.asin(0.5))]
        for a, b, x, expected in cases:
            lower, upper = _beta_cdf_pair(a, b, x)
            self.assertAlmostEqual(lower, expected, places=12)
            self.assertAlmostEqual(lower + upper, 1.0, places=12)

    def test_incomplete_beta_handles_skewed_shapes_and_roundoff(self):
        # High-precision values from mpmath; these exercise a tiny Beta shape,
        # a large shape ratio, and a near-one tail where direct subtraction is
        # poorly conditioned.
        cases = [
            (0.0001, 1e6, 1e-6, 0.9999780593618636990),
            (10000.0, 1e-12, 0.99999, 1.8229646768954804e-12),
            (0.0001, 0.01, 5e-324, 0.9190702615988788522),
        ]
        for a, b, x, expected in cases:
            with self.subTest(a=a, b=b, x=x):
                lower, upper = _beta_cdf_pair(a, b, x)
                self.assertTrue(0.0 <= lower <= 1.0)
                self.assertTrue(0.0 <= upper <= 1.0)
                self.assertAlmostEqual(lower, expected, delta=1e-12)
                self.assertAlmostEqual(lower + upper, 1.0, places=14)

    def test_incomplete_beta_keeps_tiny_upper_tails(self):
        # mpmath at 400 digits. An upper tail evaluated directly used to be
        # returned as 1 - (1 - tail), so every upper tail below ~1e-16 became 0.
        cases = [
            (2.5, 21.0, 0.925, 1.67953051760025073e-22),
            (30.0, 70.0, 0.72, 1.524629795415189372e-18),
        ]
        for a, b, x, expected in cases:
            with self.subTest(a=a, b=b, x=x):
                _, upper = _beta_cdf_pair(a, b, x)
                self.assertAlmostEqual(upper / expected, 1.0, delta=1e-12)
        lower, _ = _beta_cdf_pair(21.0, 2.5, 0.075)
        self.assertAlmostEqual(lower / 1.679530517600270201e-22, 1.0, delta=1e-12)

    def test_beta_prior_is_mirror_symmetric(self):
        # A zero cell mass is a log-weight of -inf: that reliability could never be
        # learned. The far tail must survive on both sides of the mean.
        low = TrustDistribution.beta_prior(0.3, 100, 101)
        high = TrustDistribution.beta_prior(0.7, 100, 101)
        self.assertTrue(all(math.isfinite(w) for w in low.log_weights + high.log_weights))
        for w_low, w_high in zip(low.log_weights, reversed(high.log_weights)):
            self.assertAlmostEqual(w_low, w_high, delta=1e-9 * max(1.0, abs(w_low)))

    def test_incomplete_beta_rejects_invalid_parameters(self):
        for args in (
            (0.0, 1.0, 0.5),
            (1.0, float("inf"), 0.5),
            (1.0, 1.0, float("nan")),
            (1.0, 1.0, -0.1),
            (10**1000, 1.0, 0.5),
            (True, 1.0, 0.5),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                _beta_cdf_pair(*args)

    def test_incomplete_beta_rejects_unsupported_interior_shapes(self):
        for args in ((1e-13, 1.0, 0.4), (1.0, 1_000_001.0, 0.4)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "supports shape"):
                _beta_cdf_pair(*args)
        self.assertEqual(_beta_cdf_pair(1e-300, 1e300, 0.0), (0.0, 1.0))

    def test_beta_prior_rejects_shapes_outside_numerical_support(self):
        with self.assertRaisesRegex(ValueError, "imply Beta shapes"):
            TrustDistribution.beta_prior(0.5, concentration=1e-12, points=51)
        with self.assertRaisesRegex(ValueError, "imply Beta shapes"):
            TrustDistribution.beta_prior(0.5, concentration=2_000_000, points=51)
        with self.assertRaisesRegex(ValueError, "concentration must be finite"):
            TrustDistribution.beta_prior(0.5, concentration=10**1000, points=51)

    def test_trust_representations_share_the_update_interface(self):
        states = (
            TrustDistribution.beta_prior(0.7, 8.0, points=101),
            Laputa16Trust.from_beta(0.7, 8.0),
        )
        for trust in states:
            expected = 0.4 * trust.expected_reliability / (
                0.4 * trust.expected_reliability + 0.6 * trust.expected_unreliability
            )
            self.assertAlmostEqual(update_belief(0.4, trust, True), expected, places=12)
            self.assertAlmostEqual(log_likelihood_ratio(trust, True),
                                   math.log(trust.expected_reliability / trust.expected_unreliability),
                                   places=12)

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
        with self.assertRaises(ValueError):
            update_belief(10**1000, TrustDistribution.uniform(11), True)


if __name__ == "__main__":
    unittest.main()
