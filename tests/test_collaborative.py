"""Regression tests for dependency-aware sequential inquiry."""

import unittest

from kanm.collaborative import (
    CollaborativeAction,
    optimal_collaborative_plan,
)
from kanm.dependence import ProvenanceNetwork, corroborating_relay, relay, symmetric_primary


class CollaborativePlanningTests(unittest.TestCase):
    def setUp(self):
        self.network = ProvenanceNetwork()
        self.network.add(symmetric_primary("study", 0.8))
        self.network.add(relay("summary", "study", 0.95))
        self.network.add(symmetric_primary("replication", 0.75))
        self.network.add(corroborating_relay(
            "audit", "study", fidelity=0.95, own_reliability=0.9, mix=0.5
        ))
        self.actions = [
            CollaborativeAction("study", "study", "primary_study", cost=0.2),
            CollaborativeAction("summary", "summary", "literature_summary", cost=0.1,
                                requires=("study",)),
            CollaborativeAction("replication", "replication", "independent_replication",
                                cost=0.3, requires=("study",)),
            CollaborativeAction("audit", "audit", "methods_audit", cost=0.25,
                                requires=("study",)),
        ]
        self.utilities = {
            "accept": {True: 1.0, False: -1.0},
            "reject": {True: -1.0, False: 1.0},
        }

    def test_joint_planner_does_not_pay_for_pure_relay_after_parent(self):
        actions = [
            CollaborativeAction("summary", "summary", "summary", cost=0.0,
                                requires=("study",)),
            CollaborativeAction("study", "study", "study", cost=0.0),
        ]
        result = optimal_collaborative_plan(
            0.5, self.network, actions, self.utilities, horizon=1,
            initial_observations={"study": True}, inference="joint",
        )
        self.assertIsNone(result.first_action)
        self.assertAlmostEqual(result.value, result.value_if_stopping_now)

    def test_independence_comparator_exposes_relay_overcounting(self):
        actions = [CollaborativeAction("summary", "summary", "summary", cost=0.0)]
        asymmetric_utilities = {
            "adopt": {True: 1.0, False: -3.0},
            "reject": {True: 0.0, False: 0.0},
        }
        joint = optimal_collaborative_plan(
            0.5, self.network, actions, asymmetric_utilities, horizon=1,
            initial_observations={"study": True}, inference="joint",
        )
        naive = optimal_collaborative_plan(
            0.5, self.network, actions, asymmetric_utilities, horizon=1,
            initial_observations={"study": True}, inference="independent",
        )
        self.assertIsNone(joint.first_action)
        self.assertEqual(naive.first_action, "summary")
        self.assertGreater(naive.value_of_planning, 0.0)

    def test_cost_can_make_the_planner_stop(self):
        actions = [CollaborativeAction("study", "study", "primary", cost=100.0)]
        result = optimal_collaborative_plan(
            0.5, self.network, actions, self.utilities, horizon=1, cost_weight=1.0
        )
        self.assertIsNone(result.first_action)
        self.assertAlmostEqual(result.value, 0.0)

    def test_prerequisites_control_availability(self):
        actions = [
            CollaborativeAction("study", "study", "study", cost=100.0),
            CollaborativeAction("replication", "replication", "replication",
                                cost=0.0, requires=("study",)),
        ]
        result = optimal_collaborative_plan(
            0.5, self.network, actions, self.utilities, horizon=1
        )
        self.assertIsNone(result.first_action)
        self.assertNotIn("replication", result.action_values)

    def test_observed_action_cannot_be_repeated(self):
        actions = [CollaborativeAction("study", "study", "primary", cost=0.0)]
        result = optimal_collaborative_plan(
            0.5, self.network, actions, self.utilities, horizon=3,
            initial_observations={"study": True},
        )
        self.assertIsNone(result.first_action)
        self.assertEqual(result.evaluated_states, 1)

    def test_horizon_zero_stops(self):
        result = optimal_collaborative_plan(
            0.5, self.network, self.actions, self.utilities, horizon=0
        )
        self.assertIsNone(result.first_action)
        self.assertEqual(result.action_values, {})

    def test_rejects_cycles_duplicate_evidence_and_bad_modes(self):
        cyclic = [
            CollaborativeAction("study", "study", "study", requires=("summary",)),
            CollaborativeAction("summary", "summary", "summary", requires=("study",)),
        ]
        with self.assertRaisesRegex(ValueError, "acyclic"):
            optimal_collaborative_plan(0.5, self.network, cyclic, self.utilities, horizon=1)
        duplicate_nodes = [
            CollaborativeAction("x", "study", "x"),
            CollaborativeAction("y", "study", "y"),
        ]
        with self.assertRaisesRegex(ValueError, "only one action"):
            optimal_collaborative_plan(
                0.5, self.network, duplicate_nodes, self.utilities, horizon=1
            )
        with self.assertRaisesRegex(ValueError, "inference"):
            optimal_collaborative_plan(
                0.5, self.network, self.actions, self.utilities, horizon=1,
                inference="approximate",
            )


if __name__ == "__main__":
    unittest.main()
