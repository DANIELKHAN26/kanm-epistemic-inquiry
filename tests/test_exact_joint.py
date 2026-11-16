"""Exact joint posterior: brute-force agreement, invariances and Laputa relation."""

from itertools import permutations, product
import math
import random
import unittest

from kanm.bayes import (
    ExactJointState,
    TrustDistribution,
    expit,
    logit,
    update_belief,
    update_log_odds,
)


def brute_force(prior, priors, reports):
    """Enumerate the full joint over (p, r_1, ..., r_k) on the grid."""
    sources = list(priors)
    grids = {s: priors[s].grid for s in sources}  # each prior has its own atoms
    total_true = total_false = 0.0
    marginal = {s: [0.0] * len(grids[s]) for s in sources}
    for indices in product(*(range(len(grids[s])) for s in sources)):
        weight = 1.0
        for s, i in zip(sources, indices):
            weight *= priors[s].weights[i]
        like_true = like_false = 1.0
        for source, positive in reports:
            r = grids[source][indices[sources.index(source)]]
            like_true *= r if positive else 1.0 - r
            like_false *= (1.0 - r) if positive else r
        wt = prior * weight * like_true
        wf = (1.0 - prior) * weight * like_false
        total_true += wt
        total_false += wf
        for s, i in zip(sources, indices):
            marginal[s][i] += wt + wf
    z = total_true + total_false
    return total_true / z, {s: [m / z for m in marginal[s]] for s in sources}


class ExactJointTests(unittest.TestCase):
    def setUp(self):
        self.a = TrustDistribution.beta_prior(0.7, 6, points=21)
        self.b = TrustDistribution.beta_prior(0.55, 3, points=21)

    def _state(self, prior=0.4):
        state = ExactJointState(prior)
        state.add_source("a", self.a)
        state.add_source("b", self.b)
        return state

    def test_matches_brute_force_joint_enumeration(self):
        reports = [("a", True), ("b", False), ("a", True), ("a", False), ("b", False)]
        state = self._state()
        for source, positive in reports:
            state.observe(source, positive)
        expected_belief, expected_marginals = brute_force(
            0.4, {"a": self.a, "b": self.b}, reports
        )
        self.assertAlmostEqual(state.belief, expected_belief, places=12)
        for source in ("a", "b"):
            for got, want in zip(state.trust(source).weights, expected_marginals[source]):
                self.assertAlmostEqual(got, want, places=12)

    def test_posterior_is_order_invariant(self):
        reports = [("a", True), ("a", True), ("b", False), ("a", False)]
        beliefs = set()
        for order in set(permutations(reports)):
            state = self._state()
            for source, positive in order:
                state.observe(source, positive)
            beliefs.add(round(state.belief, 13))
        self.assertEqual(len(beliefs), 1)

    def test_one_report_per_source_equals_laputa_belief(self):
        state = self._state(0.3)
        state.observe("a", True)
        state.observe("b", False)
        laputa = update_belief(update_belief(0.3, self.a, True), self.b, False)
        self.assertAlmostEqual(state.belief, laputa, places=12)

    def test_repeated_reports_from_one_source_differ_from_laputa(self):
        # Repeated reports share the unknown reliability, so they are
        # correlated; the factorized rule treats them as fresh evidence.
        trust = TrustDistribution.beta_prior(0.6, 8, points=201)
        state = ExactJointState(0.5)
        state.add_source("s", trust)
        q, t = 0.5, trust
        for _ in range(10):
            state.observe("s", True)
            q, t = update_belief(q, t, True), t.observe(q, True)
        self.assertLess(state.belief, q)
        self.assertAlmostEqual(state.belief, 0.897323, places=5)
        self.assertAlmostEqual(q, 0.999373, places=5)

    def test_marginal_trust_is_normalized_and_learns(self):
        state = self._state(0.9)
        for _ in range(5):
            state.observe("a", True)
        trust = state.trust("a")
        self.assertAlmostEqual(sum(trust.weights), 1.0)
        self.assertGreater(trust.expected_reliability, self.a.expected_reliability)

    def test_dogmatic_prior_is_stable(self):
        state = ExactJointState(1.0)
        state.add_source("a", self.a)
        state.observe("a", False)
        self.assertEqual(state.belief, 1.0)
        self.assertEqual(state.log_odds, math.inf)

    def test_unknown_and_duplicate_sources_rejected(self):
        state = self._state()
        with self.assertRaises(ValueError):
            state.observe("missing", True)
        with self.assertRaises(ValueError):
            state.add_source("a", self.a)

    def test_log_predictive_is_a_distribution(self):
        state = self._state()
        state.observe("a", True)
        total = math.exp(state.log_predictive("a", True)) + math.exp(
            state.log_predictive("a", False)
        )
        self.assertAlmostEqual(total, 1.0, places=12)


class LogOddsKernelTests(unittest.TestCase):
    def test_log_odds_update_matches_probability_update(self):
        rng = random.Random(5)
        for _ in range(200):
            q = rng.uniform(0.01, 0.99)
            trust = TrustDistribution.beta_prior(rng.uniform(0.05, 0.95), rng.uniform(1, 30), 51)
            positive = rng.random() < 0.5
            self.assertAlmostEqual(
                expit(update_log_odds(logit(q), trust, positive)),
                update_belief(q, trust, positive),
                places=12,
            )

    def test_expit_is_stable_at_extremes(self):
        self.assertEqual(expit(math.inf), 1.0)
        self.assertEqual(expit(-math.inf), 0.0)
        self.assertEqual(expit(1000.0), 1.0)
        self.assertEqual(expit(-1000.0), 0.0)
        self.assertAlmostEqual(expit(0.0), 0.5)
        with self.assertRaises(ValueError):
            expit(math.nan)

    def test_certain_source_meeting_contrary_certainty_fails_clearly(self):
        certain = TrustDistribution((0.5, 1.0), (0.0, 1.0))  # hand-built point mass at r = 1
        with self.assertRaisesRegex(ValueError, "zero probability"):
            update_log_odds(math.inf, certain, False)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            TrustDistribution.beta_prior(1.0, 5, points=11)

    def test_grid_size_is_bounded_and_integer(self):
        with self.assertRaises(ValueError):
            TrustDistribution.uniform(2.5)
        with self.assertRaises(ValueError):
            TrustDistribution.beta_prior(0.6, 8, points=10**7)


if __name__ == "__main__":
    unittest.main()
