import unittest

from kanm.batch import run_batch
from kanm.model import AgentSpec, EdgeSpec, Simulation


class SimulationTests(unittest.TestCase):
    def test_no_events_preserves_initial_beliefs(self):
        simulation = Simulation(
            [AgentSpec("a", prior_belief=0.2, inquiry_chance=0.0)], [], steps=20, seed=7
        )
        result = simulation.run()
        self.assertEqual(result.final_beliefs, {"a": 0.2})
        self.assertEqual(result.evidence_events, 0)
        self.assertEqual(result.message_events, 0)

    def test_seed_reproduces_same_result(self):
        agents = [
            AgentSpec("a", inquiry_chance=0.4, inquiry_accuracy=0.8),
            AgentSpec("b", inquiry_chance=0.3, inquiry_accuracy=0.7),
        ]
        edges = [EdgeSpec("a", "b", listen_chance=0.7)]
        first = Simulation(agents, edges, steps=30, seed=123).run().as_dict()
        second = Simulation(agents, edges, steps=30, seed=123).run().as_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["configuration"]["agents"][0]["agent_id"], "a")
        self.assertEqual(first["configuration"]["edges"][0]["sender"], "a")
        self.assertIn("python_version", first["runtime"])

    def test_edge_direction_is_respected(self):
        simulation = Simulation(
            [
                AgentSpec("a", prior_belief=0.9, inquiry_chance=0.0),
                AgentSpec("b", prior_belief=0.5, inquiry_chance=0.0),
            ],
            [EdgeSpec("a", "b", listen_chance=1.0, assertion_threshold=0.75, trust_mean=0.9)],
            steps=1,
            seed=1,
        )
        result = simulation.run()
        self.assertEqual(result.final_beliefs["a"], 0.9)
        self.assertGreater(result.final_beliefs["b"], 0.5)
        self.assertEqual(result.message_events, 1)

    def test_basic_metrics_are_in_range(self):
        simulation = Simulation(
            [AgentSpec("a", inquiry_chance=0.5, inquiry_accuracy=0.8)],
            [],
            truth=True,
            steps=20,
            seed=42,
        )
        result = simulation.run()
        self.assertTrue(0.0 <= result.mean_brier_score <= 1.0)
        self.assertTrue(0.0 <= result.accuracy_at_half <= 1.0)
        self.assertTrue(0.0 <= result.final_beliefs["a"] <= 1.0)

    def test_publish_only_after_inquiry_suppresses_recycled_opinions(self):
        agents = [
            AgentSpec("a", prior_belief=0.9, inquiry_chance=0.0),
            AgentSpec("b", prior_belief=0.5, inquiry_chance=0.0),
        ]
        edges = [EdgeSpec("a", "b", listen_chance=1.0, assertion_threshold=0.75)]
        baseline = Simulation(agents, edges, steps=4, seed=9).run()
        constrained = Simulation(
            agents, edges, steps=4, seed=9, publish_only_after_inquiry=True
        ).run()
        self.assertEqual(baseline.message_events, 4)
        self.assertEqual(constrained.message_events, 0)

    def test_batch_uses_consecutive_seeds_and_reports_mcse(self):
        summary = run_batch(
            [AgentSpec("a", inquiry_chance=0.3)],
            [],
            replicates=3,
            seed_start=20,
            steps=5,
        )
        self.assertEqual(summary["seeds"], [20, 21, 22])
        self.assertEqual(summary["configuration"]["agents"][0]["agent_id"], "a")
        self.assertIsNotNone(summary["summary"]["mean_brier_score"]["monte_carlo_se"])

    def test_single_replicate_does_not_claim_uncertainty_estimate(self):
        summary = run_batch([AgentSpec("a")], [], replicates=1, steps=1)
        self.assertIsNone(summary["summary"]["mean_brier_score"]["monte_carlo_se"])

    def test_string_truth_is_rejected_instead_of_becoming_true(self):
        with self.assertRaisesRegex(ValueError, "truth must be a boolean"):
            Simulation([AgentSpec("a")], [], truth="false")

    def test_non_integer_steps_are_rejected_at_boundary(self):
        with self.assertRaisesRegex(ValueError, "steps must"):
            Simulation([AgentSpec("a")], [], steps=2.5)


if __name__ == "__main__":
    unittest.main()
