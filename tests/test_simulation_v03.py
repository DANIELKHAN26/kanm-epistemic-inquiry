"""Schedules, update rules, metrics and regression tests added in v0.3.0."""

import math
import unittest

from kanm.bayes import TrustDistribution, update_belief
from kanm.batch import paired_comparison, run_batch, sweep
from kanm.model import AgentSpec, EdgeSpec, Simulation
from kanm.networks import agent_ids, agents_from_ids, complete, edges_from_pairs


def dense(n=4, **edge):
    agents = [AgentSpec(f"a{i}", inquiry_chance=0.5, inquiry_accuracy=0.7) for i in range(n)]
    pairs = complete([a.agent_id for a in agents])
    return agents, edges_from_pairs(pairs, **{"listen_chance": 0.6, **edge})


class ScheduleTests(unittest.TestCase):
    def test_simultaneous_messages_use_start_of_step_credence(self):
        agents = [
            AgentSpec("a", prior_belief=0.9, inquiry_chance=0.0),
            AgentSpec("b", prior_belief=0.2, inquiry_chance=0.0),
        ]
        edges = [EdgeSpec("a", "b", trust_mean=0.7), EdgeSpec("b", "a", trust_mean=0.7)]
        result = Simulation(agents, edges, steps=1, schedule="simultaneous").run()
        trust = TrustDistribution.beta_prior(0.7, 8.0, 201)
        self.assertAlmostEqual(result.final_beliefs["b"], update_belief(0.2, trust, True), places=12)
        self.assertAlmostEqual(result.final_beliefs["a"], update_belief(0.9, trust, False), places=12)
        # Trust updates use C_t of the receiver.
        self.assertAlmostEqual(
            result.final_trust["a->b"], trust.observe(0.2, True).expected_reliability, places=12
        )

    def test_input_order_does_not_change_results(self):
        agents, edges = dense()
        for schedule in ("simultaneous", "inquiry_then_messages"):
            for rule in ("laputa", "exact_joint"):
                first = Simulation(agents, edges, steps=25, seed=4, schedule=schedule,
                                   update_rule=rule).run()
                second = Simulation(list(reversed(agents)), list(reversed(edges)), steps=25,
                                    seed=4, schedule=schedule, update_rule=rule).run()
                self.assertEqual(first.final_beliefs, second.final_beliefs)

    def test_common_random_numbers_across_rules(self):
        agents, edges = dense()
        logs = []
        for rule in ("laputa", "exact_joint"):
            result = Simulation(agents, edges, steps=40, seed=11, update_rule=rule).run()
            logs.append([(e["step"], e["agent"], e["positive"])
                         for e in result.event_log if e["kind"] == "inquiry"])
        self.assertEqual(logs[0], logs[1])
        self.assertGreater(len(logs[0]), 0)

    def test_rules_agree_when_each_source_reports_once(self):
        agents, edges = dense()
        a = Simulation(agents, edges, steps=1, seed=2, update_rule="laputa").run()
        b = Simulation(agents, edges, steps=1, seed=2, update_rule="exact_joint").run()
        for agent_id in a.final_beliefs:
            self.assertAlmostEqual(a.final_beliefs[agent_id], b.final_beliefs[agent_id], places=12)

    def test_publish_only_after_inquiry_in_simultaneous_schedule(self):
        agents = [
            AgentSpec("a", prior_belief=0.9, inquiry_chance=1.0, inquiry_accuracy=1.0),
            AgentSpec("b", prior_belief=0.5, inquiry_chance=0.0),
        ]
        edges = [EdgeSpec("a", "b")]
        result = Simulation(agents, edges, steps=3, publish_only_after_inquiry=True).run()
        # Step 1: a has no inquiry before C_1, so it may not publish yet.
        self.assertEqual(result.message_events, 2)

    def test_invalid_rule_and_schedule_rejected(self):
        with self.assertRaises(ValueError):
            Simulation([AgentSpec("a")], [], update_rule="magic")
        with self.assertRaises(ValueError):
            Simulation([AgentSpec("a")], [], schedule="async")

    def test_simulation_runs_only_once(self):
        simulation = Simulation([AgentSpec("a")], [], steps=1)
        simulation.run()
        with self.assertRaises(RuntimeError):
            simulation.run()


class RegressionTests(unittest.TestCase):
    def test_saturated_credence_keeps_information(self):
        # In v0.2.0 credences rounded to exactly 1.0 and froze. Log-odds keep
        # the accumulated evidence, so the log score stays finite and positive.
        agents, edges = dense(listen_chance=1.0)
        result = Simulation(agents, edges, steps=300, seed=1).run()
        # All credences are saturated in floating point (here every agent has
        # locked onto one answer, which need not be the true one).
        self.assertTrue(set(result.final_beliefs.values()) <= {0.0, 1.0})
        self.assertIsNotNone(result.mean_log_score)
        self.assertTrue(math.isfinite(result.mean_log_score))

    def test_point_mass_trust_is_rejected_before_running(self):
        with self.assertRaisesRegex(ValueError, "strictly between"):
            Simulation([AgentSpec("a", inquiry_trust_mean=1.0)], [])
        with self.assertRaisesRegex(ValueError, "strictly between"):
            Simulation([AgentSpec("a"), AgentSpec("b")], [EdgeSpec("a", "b", trust_mean=0.0)])

    def test_accuracy_treats_exact_half_symmetrically(self):
        for truth in (True, False):
            result = Simulation([AgentSpec("a", inquiry_chance=0.0)], [], truth=truth,
                                steps=1).run()
            self.assertEqual(result.accuracy_at_half, 0.0)
            self.assertEqual(result.undecided_fraction, 1.0)

    def test_dogmatic_wrong_agent_makes_log_score_undefined(self):
        result = Simulation([AgentSpec("a", prior_belief=0.0, inquiry_chance=1.0)], [],
                            truth=True, steps=3).run()
        self.assertEqual(result.final_beliefs["a"], 0.0)
        self.assertIsNone(result.mean_log_score)

    def test_assertion_threshold_is_validated_as_probability(self):
        with self.assertRaises(ValueError):
            Simulation([AgentSpec("a"), AgentSpec("b")],
                       [EdgeSpec("a", "b", assertion_threshold=float("nan"))])


class MetricTests(unittest.TestCase):
    def test_veritistic_value_and_trajectory(self):
        agents, edges = dense()
        result = Simulation(agents, edges, steps=30, seed=3, truth=False,
                            record_trajectory=True).run()
        trajectory = result.truth_credence_trajectory
        self.assertEqual(len(trajectory), 31)
        self.assertAlmostEqual(result.veritistic_value, trajectory[-1] - trajectory[0])
        mean_false = sum(1 - b for b in result.final_beliefs.values()) / 4
        self.assertAlmostEqual(trajectory[-1], mean_false)

    def test_dispersion_and_spread(self):
        result = Simulation([AgentSpec("a", prior_belief=0.2, inquiry_chance=0.0),
                             AgentSpec("b", prior_belief=0.6, inquiry_chance=0.0)],
                            [], steps=0).run()
        self.assertAlmostEqual(result.belief_dispersion, 0.2)
        self.assertAlmostEqual(result.belief_spread, 0.4)

    def test_events_can_be_disabled_without_changing_results(self):
        agents, edges = dense()
        a = Simulation(agents, edges, steps=20, seed=8).run()
        b = Simulation(agents, edges, steps=20, seed=8, record_events=False).run()
        self.assertEqual(a.final_beliefs, b.final_beliefs)
        self.assertEqual(b.event_log, [])
        self.assertEqual(a.message_events, b.message_events)


class BatchTests(unittest.TestCase):
    def test_paired_comparison_reports_differences(self):
        agents, edges = dense()
        report = paired_comparison(
            agents, edges,
            {"laputa": {"update_rule": "laputa"}, "exact": {"update_rule": "exact_joint"}},
            replicates=5, steps=20,
        )
        diff = report["paired_difference_vs_baseline"]["exact"]["evidence_events"]
        self.assertEqual(diff["mean"], 0.0)  # identical inquiry draws
        self.assertIn("veritistic_value", report["summary"]["laputa"])

    def test_paired_comparison_rejects_seed_override(self):
        agents, edges = dense()
        with self.assertRaises(ValueError):
            paired_comparison(agents, edges, {"x": {"seed": 3}}, replicates=2)

    def test_sweep_is_full_factorial(self):
        def build(point):
            ids = agent_ids(3)
            agents = agents_from_ids(ids, inquiry_chance=0.3)
            edges = edges_from_pairs(complete(ids), listen_chance=point["listen"])
            return agents, edges, {"steps": 5, "update_rule": point["rule"]}

        rows = sweep(build, {"listen": [0.2, 0.8], "rule": ["laputa", "exact_joint"]},
                     replicates=2)
        self.assertEqual(len(rows), 4)
        self.assertEqual({(r["listen"], r["rule"]) for r in rows},
                         {(0.2, "laputa"), (0.2, "exact_joint"), (0.8, "laputa"),
                          (0.8, "exact_joint")})

    def test_batch_summary_counts_undefined_values(self):
        summary = run_batch([AgentSpec("a", prior_belief=0.0, inquiry_chance=1.0)], [],
                            replicates=3, steps=2)
        self.assertEqual(summary["summary"]["mean_log_score"]["n_undefined"], 3)
        self.assertIsNone(summary["summary"]["mean_log_score"]["mean"])


if __name__ == "__main__":
    unittest.main()
