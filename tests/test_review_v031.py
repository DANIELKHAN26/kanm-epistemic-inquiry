"""Regression tests for the second independent review (defects D16-D24).

Each test reproduces a defect that existed in the v0.3.0 draft; see
research/08_v0.3_audit_and_extensions.md, section 1.
"""

import math
import random
import unittest

from kanm.batch import paired_comparison
from kanm.bayes import ExactJointState, TrustDistribution
from kanm.dependence import ProvenanceNetwork
from kanm.evidence import EvidenceItem, EvidenceLedger
from kanm.inquiry import QueryAction, QuestionGraph, QuestionRelation, select_action
from kanm.model import AgentSpec, EdgeSpec, Simulation


def grid_posterior_mean(prior, positives, negatives):
    """Independent reference: prior * r^k (1-r)^m on the same grid, in log space."""
    logs = [
        math.log(w) + positives * math.log(r) + negatives * math.log1p(-r)
        for r, w in zip(prior.grid, prior.weights)
    ]
    peak = max(logs)
    ws = [math.exp(x - peak) for x in logs]
    return sum(r * w for r, w in zip(prior.grid, ws)) / sum(ws)


class TrustUnderflowTests(unittest.TestCase):
    """D16: linear weights underflowed and could never recover."""

    def test_trust_recovers_after_long_contrary_run(self):
        prior = TrustDistribution.beta_prior(0.6, 8.0, 201)
        trust = prior
        for _ in range(2000):
            trust = trust.observe(1.0, True)
        self.assertTrue(any(w == 0.0 for w in trust.weights))  # display view underflows
        for _ in range(2500):
            trust = trust.observe(1.0, False)
        expected = grid_posterior_mean(prior, 2000, 2500)
        self.assertAlmostEqual(trust.expected_reliability, expected, places=10)
        self.assertLess(trust.expected_reliability, 0.5)

    def test_exact_joint_state_recovers_too(self):
        prior = TrustDistribution.beta_prior(0.6, 8.0, 201)
        state = ExactJointState(1.0)
        state.add_source("s", prior)
        for _ in range(2000):
            state.observe("s", True)
        for _ in range(2500):
            state.observe("s", False)
        self.assertAlmostEqual(
            state.trust("s").expected_reliability, grid_posterior_mean(prior, 2000, 2500), places=10
        )

    def test_huge_hand_built_weights_do_not_overflow(self):
        trust = TrustDistribution((0.25, 0.75), (1e308, 1e308))
        self.assertEqual(trust.weights, (0.5, 0.5))
        self.assertAlmostEqual(trust.expected_reliability, 0.5)

    def test_from_log_weights_validation(self):
        with self.assertRaises(ValueError):
            TrustDistribution.from_log_weights((0.2, 0.8), (math.nan, 0.0))
        with self.assertRaises(ValueError):
            TrustDistribution.from_log_weights((0.2, 0.8), (-math.inf, -math.inf))
        trust = TrustDistribution.from_log_weights((0.2, 0.8), (-1000.0, -1000.0))
        self.assertEqual(trust.weights, (0.5, 0.5))


def mirror_pair(threshold, trust_mean, steps=3):
    """Sender a inquires once per step (always correct); b listens to a."""
    agents = [
        AgentSpec("a", 0.5, 1.0, 1.0, trust_mean, 50.0),
        AgentSpec("b", 0.5, 0.5, 0.0),
    ]
    edges = [EdgeSpec("a", "b", 1.0, threshold, 0.6, 8.0)]
    out = []
    for truth in (True, False):
        result = Simulation(agents, edges, truth=truth, steps=steps, seed=1).run()
        credence_in_truth = result.final_beliefs["b"] if truth else 1 - result.final_beliefs["b"]
        out.append((result.message_events, round(credence_in_truth, 12)))
    return out


class AssertionSymmetryTests(unittest.TestCase):
    """D17: probability comparisons broke p / not-p symmetry at ties and T = 1."""

    def test_threshold_tie_is_mirror_symmetric(self):
        for threshold in (0.6, 0.75, 0.8, 0.9):
            with self.subTest(threshold=threshold):
                first, second = mirror_pair(threshold, threshold, steps=1)
                self.assertEqual(first, second)

    def test_threshold_one_requires_dogmatism_on_both_sides(self):
        first, second = mirror_pair(1.0, 0.9, steps=200)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)


class ValidationTests(unittest.TestCase):
    def test_paired_comparison_rejects_non_integer_seed_start(self):  # D18
        agents = [AgentSpec("a")]
        conditions = {"x": {"update_rule": "laputa"}, "y": {"update_rule": "exact_joint"}}
        for bad in (True, "7", 1.5):
            with self.assertRaisesRegex(ValueError, "seed_start"):
                paired_comparison(agents, [], conditions, replicates=2, seed_start=bad, steps=1)

    def test_from_ledger_rejects_unused_tables(self):  # D21
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem("raw", "q", "yes", "lab"))
        ledger.add(EvidenceItem("copy", "q", "yes", "a", parents=("raw",)))
        kwargs = dict(question_id="q", positive_outcome="yes", negative_outcome="no",
                      primary_reliability=0.8, relay_fidelity=0.9)
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            ProvenanceNetwork.from_ledger(ledger, cpds={"Copy": {}}, **kwargs)
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            ProvenanceNetwork.from_ledger(
                ledger, **{**kwargs, "primary_reliability": {"raw": 0.8, "typo": 0.7}}
            )

    def test_final_trust_keys_cannot_collide(self):  # D22
        # Edge a -> "b:inquiry" and agent "a->b" both give key "a->b:inquiry".
        agents = [AgentSpec("a"), AgentSpec("b:inquiry"), AgentSpec("a->b")]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            Simulation(agents, [EdgeSpec("a", "b:inquiry")])


class QuestionGraphTests(unittest.TestCase):
    def test_abandoned_question_stays_abandoned(self):  # D19
        graph = QuestionGraph("goal", "Is p true?")
        graph.abandon("goal", rationale="no resources")
        graph.add_child(
            parent_id="goal", question_id="sub", text="Is q true?",
            relation=QuestionRelation.SUBSTITUTION, rationale="restart", mapping="q proxies p",
        )
        self.assertEqual(graph.status["goal"], "abandoned")
        with self.assertRaisesRegex(ValueError, "abandoned"):
            graph.activate_question("goal", rationale="back", mapping="same")


class SelectionTieTests(unittest.TestCase):
    def test_equal_scores_break_ties_by_id_not_rounding(self):  # D20
        rng = random.Random(0)
        for _ in range(300):
            hyps = ["h0", "h1", "h2"]
            raw = [rng.random() for _ in hyps]
            prior = {h: p / sum(raw) for h, p in zip(hyps, raw)}
            rows = {}
            for h in hyps:
                r = [rng.random() for _ in range(3)]
                rows[h] = tuple(x / sum(r) for x in r)
            a = QueryAction("a", "q", "x", ("o1", "o2", "o3"), rows)
            b = QueryAction("b", "q", "x", ("o3", "o2", "o1"),
                            {h: tuple(reversed(v)) for h, v in rows.items()})
            self.assertEqual(select_action(prior, [b, a])[0].action_id, "a")


if __name__ == "__main__":
    unittest.main()
