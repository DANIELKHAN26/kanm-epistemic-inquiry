import math
import unittest

from kanm.inquiry import (
    QueryAction,
    QuestionGraph,
    QuestionRelation,
    expected_information_gain,
    expected_value_of_sample_information,
    select_action,
    select_action_by_value,
    update_hypothesis_posterior,
)


class InquiryTests(unittest.TestCase):
    def setUp(self):
        self.prior = {"h0": 0.5, "h1": 0.5}
        self.perfect = QueryAction(
            "perfect", "goal", "helper-a", ("no", "yes"),
            {"h0": (1.0, 0.0), "h1": (0.0, 1.0)}, cost=2.0
        )
        self.noisy = QueryAction(
            "noisy", "goal", "helper-b", ("no", "yes"),
            {"h0": (0.75, 0.25), "h1": (0.25, 0.75)}, cost=0.1
        )

    def test_information_gain_known_binary_values(self):
        self.assertAlmostEqual(expected_information_gain(self.prior, self.perfect), math.log(2))
        self.assertAlmostEqual(
            expected_information_gain(self.prior, self.noisy),
            math.log(2) + 0.75 * math.log(0.75) + 0.25 * math.log(0.25),
        )

    def test_bayesian_update_for_perfect_question(self):
        posterior = update_hypothesis_posterior(self.prior, self.perfect, "yes")
        self.assertEqual(posterior, {"h0": 0.0, "h1": 1.0})

    def test_value_of_information_uses_goal_utility(self):
        utilities = {
            "adopt": {"h0": 10.0, "h1": 0.0},
            "reject": {"h0": 0.0, "h1": 7.0},
        }
        self.assertAlmostEqual(
            expected_value_of_sample_information(self.prior, self.perfect, utilities), 3.5
        )
        uninformative = QueryAction(
            "uninformative",
            "goal",
            "helper-c",
            ("yes", "no"),
            {"h0": (0.5, 0.5), "h1": (0.5, 0.5)},
        )
        self.assertAlmostEqual(
            expected_value_of_sample_information(self.prior, uninformative, utilities), 0.0
        )

    def test_value_policy_can_prefer_lower_cost_action(self):
        utilities = {
            "adopt": {"h0": 10.0, "h1": 0.0},
            "reject": {"h0": 0.0, "h1": 7.0},
        }
        perfect = QueryAction(
            "perfect",
            "goal",
            "helper-a",
            ("no", "yes"),
            {"h0": (1.0, 0.0), "h1": (0.0, 1.0)},
            cost=3.0,
        )
        best_free, _ = select_action_by_value(
            self.prior, [perfect, self.noisy], utilities, cost_weight=0.0
        )
        best_costed, _ = select_action_by_value(
            self.prior, [perfect, self.noisy], utilities, cost_weight=1.0
        )
        self.assertEqual(best_free.action_id, "perfect")
        self.assertEqual(best_costed.action_id, "noisy")

    def test_cost_policy_is_explicit_and_deterministic(self):
        best_without_cost, _ = select_action(self.prior, [self.noisy, self.perfect])
        best_with_cost, _ = select_action(
            self.prior, [self.noisy, self.perfect], cost_weight=0.5
        )
        self.assertEqual(best_without_cost.action_id, "perfect")
        self.assertEqual(best_with_cost.action_id, "noisy")
        tied = QueryAction(
            "a", "goal", "helper", ("x", "y"),
            {"h0": (0.5, 0.5), "h1": (0.5, 0.5)}, cost=0
        )
        tied2 = QueryAction(
            "b", "goal", "helper", ("x", "y"),
            {"h0": (0.5, 0.5), "h1": (0.5, 0.5)}, cost=0
        )
        self.assertEqual(select_action(self.prior, [tied2, tied])[0].action_id, "a")

    def test_zero_probability_outcome_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "zero probability"):
            update_hypothesis_posterior({"h0": 1.0, "h1": 0.0}, self.perfect, "yes")

    def test_query_rejects_bad_likelihood_rows(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            QueryAction("bad", "q", "h", ("a", "b"), {"h0": (0.2, 0.2)})

    def test_question_change_keeps_original_goal_and_audit_trail(self):
        graph = QuestionGraph("goal", "Should policy P be adopted?")
        graph.add_child(
            parent_id="goal",
            question_id="sub1",
            text="Does P work in setting S?",
            relation=QuestionRelation.DECOMPOSITION,
            rationale="The parent question depends on effectiveness in S.",
            mapping="A positive result addresses one necessary component of the parent.",
        )
        graph.add_child(
            parent_id="sub1",
            question_id="alt1",
            text="Does P work in nearby setting S2?",
            relation=QuestionRelation.SUBSTITUTION,
            rationale="The target data for S are unavailable.",
            mapping="S2 is a proxy; results do not directly settle the original S question.",
        )
        self.assertEqual(graph.goal_id, "goal")
        self.assertEqual(graph.active_id, "alt1")
        self.assertEqual(graph.nodes["alt1"].parent_id, "sub1")
        self.assertEqual(graph.transitions[-1].mapping[:3], "S2 ")

    def test_question_substitution_requires_reason_and_parent(self):
        graph = QuestionGraph("goal", "Original")
        with self.assertRaisesRegex(ValueError, "rationale"):
            graph.add_child(
                parent_id="goal", question_id="new", text="New",
                relation=QuestionRelation.SUBSTITUTION, rationale=" ", mapping="mapping"
            )
        with self.assertRaisesRegex(ValueError, "unknown parent question"):
            graph.add_child(
                parent_id="missing", question_id="new", text="New",
                relation=QuestionRelation.SUBSTITUTION, rationale="reason", mapping="mapping"
            )

    def test_decomposition_can_record_multiple_branches_and_select_one(self):
        graph = QuestionGraph("goal", "Should policy P be adopted?")
        graph.add_child(
            parent_id="goal", question_id="effect", text="Does P work?",
            relation=QuestionRelation.DECOMPOSITION, rationale="Effect matters.",
            mapping="Necessary part of the goal."
        )
        graph.add_child(
            parent_id="goal", question_id="cost", text="What does P cost?",
            relation=QuestionRelation.DECOMPOSITION, rationale="Cost matters.",
            mapping="Necessary part of the goal.", activate=False
        )
        graph.activate_question(
            "cost",
            rationale="Cost is the next priority.",
            mapping="Still addresses the goal.",
        )
        self.assertEqual(graph.active_id, "cost")
        self.assertEqual(graph.nodes["cost"].parent_id, "goal")
        self.assertEqual(graph.transitions[-1].source_id, "effect")


if __name__ == "__main__":
    unittest.main()
