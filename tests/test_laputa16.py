"""Tests for the Laputa 1.6 trust representation and schedule (kanm.laputa16).

References are restated independently from the 1.6 source semantics (see the
module docstring); docking against the compiled source is done separately by
docking/laputa_1_6/compare.py.
"""

import random
import unittest

from kanm.laputa16 import NODES, RESOLUTION, Laputa16Trust, validated_node_values
from kanm.model import AgentSpec, EdgeSpec, Simulation


def segment_expectation(values):
    """Laputa's per-segment formula (TrustFunction::expectation, C++ branch)."""
    h = 1.0 / RESOLUTION
    total, r0 = 0.0, 0.0
    for i in range(RESOLUTION):
        f0, slope = values[i], values[i + 1] - values[i]
        total += r0 * f0 + (r0 * slope + f0 * h) * 0.5 + slope * h / 3.0
        r0 += h
    return total * h


def nodal_update(values, belief, positive):
    """TrustFunction::update restated: nodal likelihood, trapezoid normalisation."""
    out = []
    for i, v in enumerate(values):
        r = i / RESOLUTION
        likelihood = r * belief + (1 - r) * (1 - belief) if positive else (1 - r) * belief + r * (1 - belief)
        out.append(v * likelihood)
    total = 0.5 * out[0] + sum(out[1:-1]) + 0.5 * out[-1]
    return [v * RESOLUTION / total for v in out]


class TrustFunctionTests(unittest.TestCase):
    def test_uniform(self):
        trust = Laputa16Trust([1.0] * NODES)
        self.assertAlmostEqual(trust.mass, 1.0, places=14)
        self.assertAlmostEqual(trust.expected_reliability, 0.5, places=14)

    def test_expectation_matches_segment_formula(self):
        rng = random.Random(5)
        for _ in range(50):
            values = [rng.uniform(0, 3) for _ in range(NODES)]
            trust = Laputa16Trust(values)
            self.assertAlmostEqual(trust.expected_reliability, segment_expectation(values), places=12)

    def test_update_matches_nodal_rule(self):
        rng = random.Random(6)
        trust = Laputa16Trust.from_beta(0.7, 6.0)
        values = list(trust.values)
        for _ in range(40):
            belief, positive = rng.random(), rng.random() < 0.5
            trust = trust.observe(belief, positive)
            values = nodal_update(values, belief, positive)
            for got, want in zip(trust.values, values):
                self.assertAlmostEqual(got, want, places=9)
            self.assertAlmostEqual(trust.mass, 1.0, places=12)

    def test_report_direction(self):
        trust = Laputa16Trust.from_beta(0.6, 4.0)
        self.assertGreater(trust.observe(0.9, True).expected_reliability, trust.expected_reliability)
        self.assertLess(trust.observe(0.9, False).expected_reliability, trust.expected_reliability)
        # At belief 1/2 a report carries no information about reliability.
        self.assertEqual(trust.observe(0.5, True).values, trust.values)

    def test_expectation_never_reaches_one(self):
        trust = Laputa16Trust([1.0] * NODES)
        for _ in range(5000):
            trust = trust.observe(1.0, True)
        self.assertLess(trust.expected_reliability, 1.0)
        self.assertGreater(trust.expected_unreliability, 0.0)

    def test_presets(self):
        flat = Laputa16Trust.from_value_sharpness(0.5, 0.0)  # alpha = beta = 1
        self.assertAlmostEqual(flat.expected_reliability, 0.5, places=12)
        high = Laputa16Trust.from_value_sharpness(0.75, 0.5)
        alpha, beta = (0.1 + 0.75 * 99.9) ** 0.5, (0.1 + 0.25 * 99.9) ** 0.5
        self.assertAlmostEqual(high.expected_reliability, alpha / (alpha + beta), delta=0.01)

    def test_default_batch_metadistribution(self):
        # w = 0: Beta(1, 4) (mean 0.2); w = 1: Beta(4, 1) (mean 0.8); concentration 5.
        for weight, mean in ((0.0, 0.2), (0.5, 0.5), (1.0, 0.8)):
            trust = Laputa16Trust.default_batch(weight)
            self.assertAlmostEqual(trust.mass, 1.0, delta=1e-3)  # these rows are near-normalised
            self.assertAlmostEqual(trust.expected_reliability, mean, delta=2e-3)
        means = [Laputa16Trust.default_batch(w / 20).expected_reliability for w in range(21)]
        self.assertEqual(means, sorted(means))
        # Interpolation between precomputed rows is linear in the weight.
        lo, hi = Laputa16Trust.default_batch(10 / 48), Laputa16Trust.default_batch(11 / 48)
        mid = Laputa16Trust.default_batch(10.5 / 48)
        for a, b, m in zip(lo.values, hi.values, mid.values):
            self.assertAlmostEqual(m, (a + b) / 2, places=12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Laputa16Trust([1.0] * 48)
        with self.assertRaises(ValueError):
            Laputa16Trust([0.0] * NODES)
        with self.assertRaises(ValueError):
            Laputa16Trust([1.0] * 48 + [-1.0])
        with self.assertRaisesRegex(ValueError, ">= 1"):
            Laputa16Trust.from_beta(0.9, 5.0)  # beta = 0.5 < 1
        with self.assertRaisesRegex(ValueError, "mass"):
            validated_node_values([2.0] * NODES, "x")
        self.assertAlmostEqual(validated_node_values([1.0] * NODES, "x").mass, 1.0)

    def test_supplied_density_is_normalized_before_probability_updates(self):
        # A curve can pass the historical 5% mass tolerance while its raw
        # integral of r*f(r) exceeds one. Treat that scale error as numerical,
        # otherwise the report likelihood is not a probability.
        values = [0.0] * NODES
        values[-1] = 2.0 * RESOLUTION * 1.04
        imported = validated_node_values(values, "custom")
        self.assertAlmostEqual(imported.mass, 1.0, places=14)
        self.assertLessEqual(imported.expected_reliability, 1.0)
        self.assertGreaterEqual(imported.expected_unreliability, 0.0)
        for positive in (True, False):
            posterior = imported.observe(0.5, positive)
            self.assertTrue(0.0 <= posterior.expected_reliability <= 1.0)


def network(seed, n=5, density=0.5):
    rng = random.Random(seed)
    ids = [f"a{i}" for i in range(n)]
    agents = [
        AgentSpec(i, rng.uniform(0.1, 0.9), rng.uniform(0.55, 0.9), rng.uniform(0.2, 0.8), 0.6, 6.0)
        for i in ids
    ]
    edges = [
        EdgeSpec(s, r, rng.uniform(0.3, 1.0), rng.uniform(0.55, 0.95), 0.6, 6.0)
        for s in ids for r in ids if s != r and rng.random() < density
    ]
    return agents, edges


class SimulationTrustModelTests(unittest.TestCase):
    def test_validation(self):
        agents = [AgentSpec("a"), AgentSpec("b")]
        edges = [EdgeSpec("a", "b")]
        with self.assertRaisesRegex(ValueError, "update_rule 'laputa'"):
            Simulation(agents, edges, trust_model="laputa_1_6", update_rule="exact_joint")
        with self.assertRaisesRegex(ValueError, "omit trust_points"):
            Simulation(agents, edges, trust_model="laputa_1_6", trust_points=201)
        with self.assertRaisesRegex(ValueError, "only used with trust_model"):
            Simulation([AgentSpec("a", inquiry_trust_values=(1.0,) * NODES)], [])
        with self.assertRaisesRegex(ValueError, "mass"):
            Simulation([AgentSpec("a", inquiry_trust_values=(3.0,) * NODES)], [],
                       trust_model="laputa_1_6")
        sim = Simulation(agents, edges, trust_model="laputa_1_6")
        self.assertEqual(sim.trust_points, NODES)
        self.assertEqual(Simulation(agents, edges).trust_points, 201)

    def test_explicit_values_are_used(self):
        values = Laputa16Trust.default_batch(0.3).values
        result = Simulation([AgentSpec("a", 0.5, 1.0, 0.0, inquiry_trust_values=values)], [],
                            trust_model="laputa_1_6", steps=0).run()
        self.assertAlmostEqual(result.final_trust["a:inquiry"],
                               validated_node_values(values, "test").expected_reliability)
        self.assertEqual(result.as_dict()["assumptions"]["trust_model"], "laputa_1_6")


class Laputa16ScheduleTests(unittest.TestCase):
    def test_validation(self):
        agents = [AgentSpec("a"), AgentSpec("b")]
        edges = [EdgeSpec("a", "b")]
        with self.assertRaisesRegex(ValueError, "assertion_rule 'laputa_1_6'"):
            Simulation(agents, edges, schedule="laputa_1_6")
        with self.assertRaisesRegex(ValueError, "update_rule 'laputa'"):
            Simulation(agents, edges, schedule="laputa_1_6", assertion_rule="laputa_1_6",
                       update_rule="exact_joint")
        with self.assertRaisesRegex(ValueError, "only to schedule"):
            Simulation(agents, edges, evidence_policy="inquiry")
        with self.assertRaisesRegex(ValueError, "evidence_policy instead"):
            Simulation(agents, edges, schedule="laputa_1_6", assertion_rule="laputa_1_6",
                       publish_only_after_inquiry=True)

    def test_policy_none_equals_simultaneous_schedule(self):
        """Without a new-evidence policy, Society::evolve is the simultaneous update."""
        for seed in range(6):
            agents, edges = network(seed)
            for trust_model in ("beta_grid", "laputa_1_6"):
                kwargs = dict(steps=30, seed=seed, assertion_rule="laputa_1_6",
                              trust_model=trust_model)
                a = Simulation(agents, edges, schedule="simultaneous", **kwargs).run()
                b = Simulation(agents, edges, schedule="laputa_1_6", **kwargs).run()
                self.assertEqual(a.final_beliefs, b.final_beliefs)
                self.assertEqual(a.final_trust, b.final_trust)

    def run_policy(self, agents, edges, policy, **kwargs):
        return Simulation(agents, edges, schedule="laputa_1_6", assertion_rule="laputa_1_6",
                          evidence_policy=policy, steps=kwargs.pop("steps", 1), **kwargs).run()

    def test_inquiry_policy_is_index_order_dependent(self):
        """A lower-indexed source's same-step inquiry already counts as new evidence."""
        def pair(source, receiver):
            agents = [AgentSpec(source, 0.6, 1.0, 1.0), AgentSpec(receiver, 0.5, 0.5, 0.0)]
            return agents, [EdgeSpec(source, receiver, 1.0, 0.55)]
        early = self.run_policy(*pair("a", "b"), "inquiry")   # source processed first
        late = self.run_policy(*pair("b", "a"), "inquiry")    # source processed second
        self.assertEqual(early.message_events, 1)
        self.assertEqual(late.message_events, 0)

    def test_inquiry_policy_needs_fresh_inquiry(self):
        agents = [AgentSpec("a", 0.9, 1.0, 0.0), AgentSpec("b", 0.5, 0.5, 0.0)]
        edges = [EdgeSpec("a", "b", 1.0, 0.55)]
        self.assertEqual(self.run_policy(agents, edges, "inquiry", steps=5).message_events, 0)
        primed = self.run_policy(agents, edges, "inquiry", steps=5, count_prior_as_evidence=True)
        self.assertEqual(primed.message_events, 1)  # the prior counts once, on first use

    def test_any_policy_relays_received_evidence(self):
        # a inquires every step; b never inquires but listens to a; c listens to b.
        agents = [AgentSpec("a", 0.6, 1.0, 1.0), AgentSpec("b", 0.6, 0.5, 0.0),
                  AgentSpec("c", 0.5, 0.5, 0.0)]
        edges = [EdgeSpec("a", "b", 1.0, 0.55), EdgeSpec("b", "c", 1.0, 0.55)]

        def relayed(policy):
            result = Simulation(agents, edges, schedule="laputa_1_6", assertion_rule="laputa_1_6",
                                evidence_policy=policy, steps=6).run()
            return sum(1 for e in result.event_log if e["kind"] == "message" and e["sender"] == "b")

        self.assertEqual(relayed("inquiry"), 0)
        self.assertGreater(relayed("any"), 0)

    def test_common_random_numbers_across_policies(self):
        agents, edges = network(11, n=6)
        runs = [self.run_policy(agents, edges, p, steps=20, seed=3, record_events=True)
                for p in ("none", "inquiry", "any")]
        inquiries = [
            sorted((e["step"], e["agent"], e["positive"]) for e in r.event_log if e["kind"] == "inquiry")
            for r in runs
        ]
        self.assertEqual(inquiries[0], inquiries[1])
        self.assertEqual(inquiries[0], inquiries[2])


if __name__ == "__main__":
    unittest.main()
