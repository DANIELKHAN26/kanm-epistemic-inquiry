"""Question partitions, scope-drift diagnostics, stopping and exact planning."""

import math
import unittest

from kanm.inquiry import (
    QueryAction,
    expected_information_gain,
    expected_value_of_sample_information,
    optimal_sequential_plan,
    question_relevance,
    refines,
    select_action_or_stop,
)

UNIFORM4 = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
GOAL = {"a": "yes", "b": "yes", "c": "no", "d": "no"}
ACTIVE_REFINEMENT = {"a": "a", "b": "b", "c": "c", "d": "d"}
PROXY = {"a": "x", "b": "y", "c": "x", "d": "y"}


def perfect_on(partition, action_id="q"):
    labels = sorted(set(partition.values()))
    return QueryAction(action_id, "q", "h", tuple(labels),
                       {h: tuple(1.0 if partition[h] == l else 0.0 for l in labels)
                        for h in partition})


class PartitionTests(unittest.TestCase):
    def test_target_partition_information(self):
        action = perfect_on(ACTIVE_REFINEMENT)
        self.assertAlmostEqual(expected_information_gain(UNIFORM4, action), math.log(4))
        self.assertAlmostEqual(
            expected_information_gain(UNIFORM4, action, target=GOAL), math.log(2)
        )

    def test_proxy_question_has_zero_goal_relevance(self):
        report = question_relevance(UNIFORM4, perfect_on(PROXY), GOAL, PROXY)
        self.assertAlmostEqual(report["information_about_active"], math.log(2))
        self.assertAlmostEqual(report["information_about_goal"], 0.0)
        self.assertEqual(report["goal_relevance_ratio"], 0.0)

    def test_refinement_obeys_data_processing_bound(self):
        noisy = QueryAction("n", "q", "h", ("a", "b", "c", "d"), {
            h: tuple(0.7 if h == o else 0.1 for o in "abcd") for h in "abcd"
        })
        self.assertTrue(refines(ACTIVE_REFINEMENT, GOAL))
        self.assertFalse(refines(PROXY, GOAL))
        report = question_relevance(UNIFORM4, noisy, GOAL, ACTIVE_REFINEMENT)
        self.assertLessEqual(report["goal_relevance_ratio"], 1.0 + 1e-12)

    def test_partition_must_cover_hypotheses(self):
        with self.assertRaises(ValueError):
            expected_information_gain(UNIFORM4, perfect_on(PROXY), target={"a": "x"})


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.prior = {"T": 0.8, "F": 0.2}
        self.test = QueryAction("t", "goal", "h", ("yes", "no"),
                                {"T": (0.55, 0.45), "F": (0.45, 0.55)}, cost=1.0)
        self.utilities = {"adopt": {"T": 1.0, "F": -3.0}, "reject": {"T": 0.0, "F": 0.0}}

    def test_horizon_one_equals_myopic_rule(self):
        for weight in (0.0, 0.005, 0.05):
            plan = optimal_sequential_plan(self.prior, [self.test], self.utilities,
                                           horizon=1, cost_weight=weight)
            evsi = expected_value_of_sample_information(self.prior, self.test, self.utilities)
            self.assertAlmostEqual(plan.value_of_planning, max(0.0, evsi - weight))

    def test_myopic_rule_stops_where_planning_continues(self):
        # One weak test can never change the decision (EVSI = 0), but two can:
        # the value of information is not concave (Radner & Stiglitz 1984).
        action, scores = select_action_or_stop(self.prior, [self.test], self.utilities,
                                               cost_weight=0.005)
        self.assertIsNone(action)
        self.assertAlmostEqual(
            expected_value_of_sample_information(self.prior, self.test, self.utilities), 0.0
        )
        plan = optimal_sequential_plan(self.prior, [self.test], self.utilities,
                                       horizon=2, cost_weight=0.005)
        self.assertEqual(plan.first_action, "t")
        self.assertGreater(plan.value_of_planning, 0.0)

    def test_value_is_monotone_in_horizon(self):
        values = [optimal_sequential_plan(self.prior, [self.test], self.utilities,
                                          horizon=h, cost_weight=0.005).value for h in range(5)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))

    def test_free_perfect_information_reaches_full_value(self):
        perfect = QueryAction("p", "goal", "h", ("yes", "no"),
                              {"T": (1.0, 0.0), "F": (0.0, 1.0)}, cost=0.0)
        plan = optimal_sequential_plan(self.prior, [perfect], self.utilities, horizon=1)
        # Expected value of perfect information: 0.8*1 + 0.2*0 - max(0.8-0.6, 0) = 0.6
        self.assertAlmostEqual(plan.value_of_planning, 0.6)

    def test_planner_rejects_oversized_problems(self):
        with self.assertRaises(ValueError):
            optimal_sequential_plan(self.prior, [self.test], self.utilities, horizon=40)



class PlannerGuardTests(unittest.TestCase):
    def test_size_guard_is_fast_for_huge_horizons(self):
        import time
        action = QueryAction("t", "g", "h", ("y", "n"), {"T": (0.6, 0.4), "F": (0.4, 0.6)})
        start = time.perf_counter()
        with self.assertRaises(ValueError):
            optimal_sequential_plan({"T": 0.5, "F": 0.5}, [action],
                                    {"a": {"T": 1, "F": 0}, "b": {"T": 0, "F": 1}},
                                    horizon=10**6)
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_tie_tolerance_is_scale_free(self):
        perfect = QueryAction("p", "g", "h", ("y", "n"), {"T": (1.0, 0.0), "F": (0.0, 1.0)},
                              cost=0.0)
        scale = 1e-13
        utilities = {"a": {"T": scale, "F": 0.0}, "b": {"T": 0.0, "F": scale}}
        plan = optimal_sequential_plan({"T": 0.5, "F": 0.5}, [perfect], utilities, horizon=1)
        self.assertEqual(plan.first_action, "p")


if __name__ == "__main__":
    unittest.main()
