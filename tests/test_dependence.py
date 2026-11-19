from itertools import product
import math
import random
import unittest
import warnings

from kanm.dependence import (
    ProvenanceNetwork,
    ReportNode,
    corroborating_relay,
    primary,
    relay,
    symmetric_primary,
)
from kanm.evidence import EvidenceItem, EvidenceLedger


def full_enumeration(network, prior, observations):
    """Independent check: sum the joint over *all* nodes, no pruning."""
    ids = list(network.nodes)
    totals = {True: 0.0, False: 0.0}
    for truth in (True, False):
        for assignment in product((True, False), repeat=len(ids)):
            values = dict(zip(ids, assignment))
            if any(values[k] != v for k, v in observations.items()):
                continue
            weight = prior if truth else 1 - prior
            for node_id in ids:
                node = network.nodes[node_id]
                weight *= node.probability(
                    values[node_id], truth, tuple(values[p] for p in node.parents)
                )
            totals[truth] += weight
    return totals[True] / (totals[True] + totals[False])


class DependenceTests(unittest.TestCase):
    def test_single_primary_is_bayes_rule(self):
        net = ProvenanceNetwork()
        net.add(primary("s", sensitivity=0.9, false_positive_rate=0.2))
        expected = 0.3 * 0.9 / (0.3 * 0.9 + 0.7 * 0.2)
        self.assertAlmostEqual(net.posterior(0.3, {"s": True}), expected)

    def test_independent_primaries_match_independence_comparator(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("a", 0.8))
        net.add(primary("b", 0.7, 0.4))
        obs = {"a": True, "b": False}
        self.assertAlmostEqual(net.posterior(0.5, obs), net.independence_posterior(0.5, obs))

    def test_literal_repeats_count_once(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.8))
        net.add(relay("r1", "study", 1.0))
        net.add(relay("r2", "study", 1.0))
        report = net.dependence_report(0.5, {"r1": True, "r2": True})
        self.assertAlmostEqual(report["exact_posterior"], 0.8)
        self.assertAlmostEqual(report["overcounting_ratio"], 2.0)

    def test_uninformative_relay(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.9))
        net.add(relay("r", "study", 0.5))
        self.assertAlmostEqual(net.posterior(0.37, {"r": True}), 0.37)

    def test_predictive_distribution_conditions_on_parent(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.8))
        net.add(relay("summary", "study", 0.9))
        self.assertAlmostEqual(
            net.predictive_distribution("summary", 0.5, {"study": True})[True], 0.9
        )
        self.assertAlmostEqual(
            net.predictive_distribution("summary", 0.5, {"study": False})[True], 0.1
        )

    def test_predictive_distribution_supports_prior_endpoints(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.8))
        self.assertEqual(
            net.predictive_distribution("study", 1.0, {})[True], 0.8
        )
        self.assertAlmostEqual(
            net.predictive_distribution("study", 0.0, {})[True], 0.2
        )

    def test_predictive_distribution_is_degenerate_for_observed_node(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.8))
        self.assertEqual(
            net.predictive_distribution("study", 0.5, {"study": False}),
            {False: 1.0, True: 0.0},
        )

    def test_predictive_distribution_rejects_impossible_conditioning(self):
        net = ProvenanceNetwork()
        net.add(primary("certain", sensitivity=1.0, false_positive_rate=0.0))
        with self.assertRaisesRegex(ValueError, "zero probability"):
            net.predictive_distribution("certain", 1.0, {"certain": False})

    def test_partial_dependence_hand_computed(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("s", 0.8))
        net.add(corroborating_relay("c", "s", fidelity=0.95, own_reliability=0.8, mix=0.5))
        # P(both | p) = 0.8 * (0.5*0.8 + 0.5*0.95); P(both | not p) = 0.2 * (0.5*0.2 + 0.5*0.95)
        lr = (0.8 * 0.875) / (0.2 * 0.575)
        self.assertAlmostEqual(net.posterior(0.5, {"s": True, "c": True}), lr / (1 + lr))

    def test_random_networks_match_full_enumeration(self):
        rng = random.Random(3)
        for _ in range(25):
            net = ProvenanceNetwork()
            ids = []
            for i in range(rng.randint(2, 6)):
                node_id = f"n{i}"
                parents = tuple(p for p in ids if rng.random() < 0.4)[:2]
                cpd = {
                    (h, values): rng.uniform(0.05, 0.95)
                    for h in (True, False)
                    for values in product((True, False), repeat=len(parents))
                }
                net.add(ReportNode(node_id, parents, cpd))
                ids.append(node_id)
            observed = {n: rng.random() < 0.5 for n in ids if rng.random() < 0.6} or {ids[-1]: True}
            prior = rng.uniform(0.1, 0.9)
            self.assertAlmostEqual(
                net.posterior(prior, observed), full_enumeration(net, prior, observed), places=12
            )

    def test_cpd_must_be_complete_and_dag_ordered(self):
        with self.assertRaisesRegex(ValueError, "every"):
            ReportNode("x", ("p",), {(True, (True,)): 0.5})
        net = ProvenanceNetwork()
        with self.assertRaisesRegex(ValueError, "added first"):
            net.add(relay("r", "missing", 0.9))

    def test_read_only_nodes_view_preserves_dag_validation(self):
        net = ProvenanceNetwork()
        node = symmetric_primary("root", 0.8)
        net.add(node)
        with self.assertRaises(TypeError):
            net.nodes["bypass"] = node

    def test_exact_inference_warns_before_latent_enumeration_gets_expensive(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("n0", 0.8))
        for index in range(1, 17):
            net.add(relay(f"n{index}", f"n{index - 1}", 0.9))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            net.posterior(0.5, {"n16": True})
        self.assertTrue(any("2^16 latent assignments" in str(w.message) for w in caught))

    def test_exact_inference_stops_at_latent_cap(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("n0", 0.8))
        for index in range(1, 22):
            net.add(relay(f"n{index}", f"n{index - 1}", 0.9))
        with self.assertRaisesRegex(ValueError, "exact-enumeration limit 20"):
            net.posterior(0.5, {"n21": True})

    def test_from_ledger_with_latent_primary(self):
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem("raw", "q", "yes", "lab"))
        ledger.add(EvidenceItem("news-a", "q", "yes", "a", parents=("raw",)))
        ledger.add(EvidenceItem("news-b", "q", "yes", "b", parents=("raw",)))
        net, obs = ProvenanceNetwork.from_ledger(
            ledger, question_id="q", positive_outcome="yes", negative_outcome="no",
            primary_reliability=0.8, relay_fidelity=0.9, observed=["news-a", "news-b"],
        )
        self.assertEqual(set(obs), {"news-a", "news-b"})
        report = net.dependence_report(0.5, obs)
        self.assertLess(report["exact_posterior"], report["independence_posterior"])
        self.assertGreater(report["overcounting_ratio"], 1.0)
        self.assertLess(report["overcounting_ratio"], 2.0)

    def test_from_ledger_requires_table_for_multiple_parents(self):
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem("x", "q", "yes", "a"))
        ledger.add(EvidenceItem("y", "q", "yes", "b"))
        ledger.add(EvidenceItem("z", "q", "yes", "c", parents=("x", "y")))
        with self.assertRaisesRegex(ValueError, "several parents"):
            ProvenanceNetwork.from_ledger(
                ledger, question_id="q", positive_outcome="yes", negative_outcome="no",
                primary_reliability=0.8, relay_fidelity=0.9,
            )

    def test_latent_limit(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("root", 0.7))
        previous = "root"
        for i in range(22):
            net.add(relay(f"r{i}", previous, 0.9))
            previous = f"r{i}"
        with self.assertRaisesRegex(ValueError, "exceeds"):
            net.posterior(0.5, {previous: True})



class DependenceNumericsTests(unittest.TestCase):
    def test_many_observations_do_not_underflow(self):
        net = ProvenanceNetwork()
        observations = {}
        for i in range(1100):
            net.add(symmetric_primary(f"s{i}", 0.6))
            observations[f"s{i}"] = i % 2 == 0  # balanced evidence
        self.assertAlmostEqual(net.posterior(0.5, observations), 0.5)

    def test_shifts_stay_exact_when_posterior_saturates(self):
        net = ProvenanceNetwork()
        observations = {}
        for i in range(60):
            net.add(symmetric_primary(f"s{i}", 0.8))
            observations[f"s{i}"] = True
        net.add(relay("copy", "s0", 1.0))
        observations["copy"] = True
        report = net.dependence_report(0.5, observations)
        self.assertAlmostEqual(report["exact_log_odds_shift"], 60 * math.log(4))
        self.assertAlmostEqual(report["independence_log_odds_shift"], 61 * math.log(4))
        self.assertAlmostEqual(report["overcounting_ratio"], 61 / 60)

    def test_independence_likelihood_ratio_preserves_underflowed_marginals(self):
        net = ProvenanceNetwork()
        count = 16
        rare = 1e-30
        for index in range(count):
            node_id = f"r{index}"
            parents = () if index == 0 else (f"r{index - 1}",)
            cpd = {}
            for truth in (True, False):
                p_true = rare if truth else 2.0 * rare
                for parent_value in ((None,) if index == 0 else (True, False)):
                    probability = p_true if parent_value is not False else 0.0
                    key = (truth, ()) if index == 0 else (truth, (parent_value,))
                    cpd[key] = probability
            net.add(ReportNode(node_id, parents, cpd))

        target = {f"r{count - 1}": True}
        # Both exact marginal likelihoods are below the smallest positive
        # float, but their ratio is finite and well-defined in log space.
        self.assertEqual(net.marginal_likelihoods(f"r{count - 1}"), (0.0, 0.0))
        expected_llr = count * math.log(0.5)
        self.assertAlmostEqual(net.independence_log_likelihood_ratio(target), expected_llr)
        self.assertAlmostEqual(
            net.independence_posterior(0.5, target),
            1.0 / (1.0 + 2.0**count),
            places=15,
        )

    def test_independence_predictive_remains_normalized_with_tiny_likelihood(self):
        net = ProvenanceNetwork()
        for index in range(16):
            node_id = f"r{index}"
            parents = () if index == 0 else (f"r{index - 1}",)
            cpd = {}
            for truth in (True, False):
                p_true = 1e-30 if truth else 2e-30
                for parent_value in ((None,) if index == 0 else (True, False)):
                    key = (truth, ()) if index == 0 else (truth, (parent_value,))
                    cpd[key] = p_true if parent_value is not False else 0.0
            net.add(ReportNode(node_id, parents, cpd))
        prediction = net.independence_predictive_distribution("r15", 0.5, {})
        self.assertEqual(prediction, {False: 1.0, True: 0.0})

    def test_independence_predictive_matches_marginal_formula(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("source", 0.8))
        net.add(relay("relay", "source", 0.9))
        net.add(primary("audit", sensitivity=0.7, false_positive_rate=0.4))
        prior = 0.3
        observations = {"relay": True}
        # The relay marginal is itself averaged over its primary parent:
        # P(relay=True | H)=0.8*0.9 + 0.2*0.1 = 0.74 (and 0.26 under not-H).
        p_h = prior * 0.74 / (prior * 0.74 + (1.0 - prior) * 0.26)
        expected_true = p_h * 0.7 + (1.0 - p_h) * 0.4
        prediction = net.independence_predictive_distribution("audit", prior, observations)
        self.assertAlmostEqual(prediction[True], expected_true, places=14)
        self.assertAlmostEqual(prediction[False], 1.0 - expected_true, places=14)


if __name__ == "__main__":
    unittest.main()
