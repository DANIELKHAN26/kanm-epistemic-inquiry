from itertools import product
import math
import random
import unittest

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


if __name__ == "__main__":
    unittest.main()
