"""Checks for robust planning, credal posteriors and information gain.

The oracle here enumerates every deterministic plan of a small problem and
evaluates each under each candidate graph from the full joint distribution,
without using any of the code under test.
"""

from fractions import Fraction
from itertools import product
from math import log2
import random
import unittest

from kanm.collaborative import CollaborativeAction, optimal_collaborative_plan
from kanm.dependence import (
    ProvenanceNetwork,
    ReportNode,
    corroborating_relay,
    relay,
    symmetric_primary,
)
from kanm.model_uncertainty import ProvenanceModelSet
from kanm.robust_planning import (
    robust_collaborative_plan,
    solve_matrix_game,
    value_of_provenance_information,
)

UTILITIES = {"adopt": {True: 1.0, False: -1.0}, "reject": {True: -0.5, False: 0.5}}
PRIOR = 0.45


def candidate_models():
    """Three accounts of where two reports and an audit come from.

    In "copied" both reports relay one study; in "partial" the second report
    only partly depends on it; in "shared_audit" the reports are independent
    but the audit re-analyses the study's data. The graphs disagree about
    which inquiry is worth paying for.
    """
    copied = ProvenanceNetwork()
    copied.add(symmetric_primary("study", 0.7))
    copied.add(relay("a", "study", 0.9))
    copied.add(relay("b", "study", 0.9))
    copied.add(symmetric_primary("check", 0.8))
    partial = ProvenanceNetwork()
    partial.add(symmetric_primary("study", 0.7))
    partial.add(relay("a", "study", 0.9))
    partial.add(corroborating_relay("b", "study", 0.85, 0.7, 0.35))
    partial.add(symmetric_primary("check", 0.8))
    shared_audit = ProvenanceNetwork()
    shared_audit.add(symmetric_primary("study", 0.7))
    shared_audit.add(symmetric_primary("a", 0.7))
    shared_audit.add(symmetric_primary("b", 0.7))
    shared_audit.add(relay("check", "study", 0.9))
    return {"copied": copied, "partial": partial, "shared_audit": shared_audit}


ACTIONS = [
    CollaborativeAction("read_a", "a", "reader", cost=0.01),
    CollaborativeAction("read_b", "b", "reader", cost=0.01),
    CollaborativeAction("audit", "check", "auditor", cost=0.06),
]


def joint_table(network, prior):
    nodes = list(network.nodes)
    rows = []
    for truth in (True, False):
        for values in product((True, False), repeat=len(nodes)):
            assignment = dict(zip(nodes, values))
            mass = prior if truth else 1.0 - prior
            for node_id in nodes:
                node = network.nodes[node_id]
                mass *= node.probability(
                    assignment[node_id], truth, tuple(assignment[p] for p in node.parents)
                )
            rows.append((mass, truth, assignment))
    return rows


def conditional(rows, state):
    total = true_mass = 0.0
    for mass, truth, assignment in rows:
        if all(assignment[k] == v for k, v in state.items()):
            total += mass
            true_mass += mass if truth else 0.0
    return total, true_mass


def all_policies(state, remaining):
    """Every deterministic plan from ``state`` as nested tuples."""
    options = [("decide", d) for d in sorted(UTILITIES)]
    if remaining:
        for action in ACTIONS:
            if action.evidence_node in state:
                continue
            branches = [
                all_policies({**state, action.evidence_node: outcome}, remaining - 1)
                for outcome in (False, True)
            ]
            for if_false, if_true in product(*branches):
                options.append(("ask", action, if_false, if_true))
    return options


def evaluate(policy, rows, state):
    total, true_mass = conditional(rows, state)
    if policy[0] == "decide":
        p = true_mass / total
        return p * UTILITIES[policy[1]][True] + (1 - p) * UTILITIES[policy[1]][False]
    action = policy[1]
    value = -action.cost
    for outcome, branch in ((False, policy[2]), (True, policy[3])):
        child = {**state, action.evidence_node: outcome}
        mass, _ = conditional(rows, child)
        if mass > 0.0:
            value += mass / total * evaluate(branch, rows, child)
    return value


class Oracle:
    """All deterministic plans of the test problem and their values per model."""

    def __init__(self, horizon=2):
        models = candidate_models()
        self.names = list(models)
        tables = {name: joint_table(net, PRIOR) for name, net in models.items()}
        self.policies = all_policies({}, horizon)
        self.values = [
            {name: evaluate(policy, tables[name], {}) for name in self.names}
            for policy in self.policies
        ]
        self.optimal = {name: max(v[name] for v in self.values) for name in self.names}

    def bayes_value(self, weights):
        return max(sum(weights[n] * v[n] for n in self.names) for v in self.values)


ORACLE = Oracle()


class MatrixGameTest(unittest.TestCase):
    def test_known_games(self):
        value, x, y = solve_matrix_game([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        self.assertAlmostEqual(value, 0.0, delta=1e-15)
        for p in x + y:
            self.assertAlmostEqual(p, 1 / 3, delta=1e-15)
        # 2 x 2 without a saddle point: value (ad - bc) / (a + d - b - c).
        value, x, y = solve_matrix_game([[3, -1], [-2, 4]])
        self.assertAlmostEqual(value, (12 - 2) / (3 + 4 + 1 + 2), delta=1e-15)
        self.assertAlmostEqual(x[0], 0.5, delta=1e-15)
        # A dominated column never enters.
        value, x, _ = solve_matrix_game([[2, 1], [3, 0]])
        self.assertAlmostEqual(value, 2.0, delta=1e-15)
        self.assertEqual(x, (1.0, 0.0))

    def test_random_games_satisfy_the_saddle_point_certificate(self):
        rng = random.Random(7)
        for _ in range(60):
            rows, cols = rng.randint(1, 5), rng.randint(1, 7)
            payoff = [[rng.uniform(-3, 3) for _ in range(cols)] for _ in range(rows)]
            if rng.random() < 0.3:   # degenerate cases with repeated entries
                payoff = [[round(v) for v in row] for row in payoff]
            value, x, y = solve_matrix_game(payoff)
            self.assertAlmostEqual(sum(x), 1.0, delta=1e-12)
            self.assertAlmostEqual(sum(y), 1.0, delta=1e-12)
            self.assertTrue(all(p >= 0 for p in x + y))
            # x guarantees at least `value` against every row; y holds every column to it.
            for i in range(rows):
                self.assertGreaterEqual(sum(payoff[i][j] * x[j] for j in range(cols)), value - 1e-12)
            for j in range(cols):
                self.assertLessEqual(sum(payoff[i][j] * y[i] for i in range(rows)), value + 1e-12)

    def test_exact_arithmetic_on_a_near_degenerate_game(self):
        value, _, _ = solve_matrix_game([[1.0, 1.0 + 2 ** -50], [1.0 + 2 ** -50, 1.0]])
        self.assertEqual(Fraction(value), Fraction(1) + Fraction(1, 2 ** 51))


class RobustPlanAgainstEnumerationTest(unittest.TestCase):
    MODEL_SET = ProvenanceModelSet(candidate_models())

    def test_the_problem_is_not_trivial(self):
        # The graphs disagree about the best plan, and no single plan is best
        # in the worst case, so both criteria have to mix.
        best = {n: max(range(len(ORACLE.values)), key=lambda i: ORACLE.values[i][n]) for n in ORACLE.names}
        self.assertGreater(len({ORACLE.values[i][n] for n, i in best.items()}), 1)
        for criterion in ("maximin", "minimax_regret"):
            plan = robust_collaborative_plan(PRIOR, self.MODEL_SET, ACTIONS, UTILITIES,
                                             horizon=2, criterion=criterion)
            self.assertGreaterEqual(sum(p > 1e-9 for p in plan.probabilities), 2)
            self.assertGreater(abs(plan.guarantee - plan.deterministic_guarantee), 1e-4)

    def test_maximin_matches_the_best_randomized_plan_over_all_plans(self):
        plan = robust_collaborative_plan(
            PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2, cost_weight=1.0
        )
        matrix = [[v[n] for v in ORACLE.values] for n in ORACLE.names]
        best, _, _ = solve_matrix_game(matrix)
        self.assertAlmostEqual(plan.guarantee, best, delta=1e-12)
        self.assertAlmostEqual(plan.bound, best, delta=1e-12)
        # Certificate without the game solver: no plan, deterministic or not,
        # does better than `best` against the least favourable weights.
        w = plan.least_favourable_weights
        self.assertLessEqual(ORACLE.bayes_value(w), plan.guarantee + 1e-12)
        # The mixture really guarantees its value under each graph.
        for name in ORACLE.names:
            mixed = sum(p * v[name] for p, v in zip(plan.probabilities, plan.values))
            self.assertGreaterEqual(mixed, plan.guarantee - 1e-12)
        deterministic = max(min(v.values()) for v in ORACLE.values)
        self.assertLessEqual(plan.deterministic_guarantee, deterministic + 1e-12)
        self.assertLessEqual(deterministic, plan.guarantee + 1e-12)

    def test_policy_values_match_the_oracle(self):
        plan = robust_collaborative_plan(PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2)
        oracle_vectors = [tuple(round(v[n], 10) for n in ORACLE.names) for v in ORACLE.values]
        for values in plan.values:
            self.assertIn(tuple(round(values[n], 10) for n in ORACLE.names), oracle_vectors)
        for name in ORACLE.names:
            self.assertAlmostEqual(plan.model_optimal_values[name], ORACLE.optimal[name], delta=1e-12)

    def test_minimax_regret_and_the_value_of_provenance_information(self):
        plan = robust_collaborative_plan(
            PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2, criterion="minimax_regret"
        )
        matrix = [[v[n] - ORACLE.optimal[n] for v in ORACLE.values] for n in ORACLE.names]
        best, _, _ = solve_matrix_game(matrix)
        self.assertAlmostEqual(plan.guarantee, -best, delta=1e-12)
        self.assertAlmostEqual(plan.bound, -best, delta=1e-12)

        def evpi(weights):
            clairvoyant = sum(weights[n] * ORACLE.optimal[n] for n in ORACLE.names)
            return clairvoyant - ORACLE.bayes_value(weights)

        # Minimax regret equals the largest value of perfect provenance
        # information, attained at the least favourable weights.
        self.assertAlmostEqual(evpi(plan.least_favourable_weights), plan.guarantee, delta=1e-12)
        for a, b in product(range(11), repeat=2):
            if a + b <= 10:
                w = dict(zip(ORACLE.names, (a / 10, b / 10, (10 - a - b) / 10)))
                self.assertLessEqual(evpi(w), plan.guarantee + 1e-12)

    def test_value_of_provenance_information_against_oracle_and_bayes_planner(self):
        weights = {"copied": 0.5, "partial": 0.2, "shared_audit": 0.3}
        weighted = ProvenanceModelSet(candidate_models(), model_weights=weights)
        result = value_of_provenance_information(PRIOR, weighted, ACTIONS, UTILITIES, horizon=2)
        bayes = optimal_collaborative_plan(PRIOR, weighted, ACTIONS, UTILITIES, horizon=2).value
        self.assertAlmostEqual(result.bayes_value, bayes, delta=1e-12)
        self.assertAlmostEqual(result.bayes_value, ORACLE.bayes_value(weights), delta=1e-12)
        clairvoyant = sum(weights[n] * ORACLE.optimal[n] for n in ORACLE.names)
        self.assertAlmostEqual(result.clairvoyant_value, clairvoyant, delta=1e-12)
        self.assertGreaterEqual(result.value, 0.0)

    def test_value_of_provenance_information_after_an_observation(self):
        weights = {"copied": 0.5, "partial": 0.2, "shared_audit": 0.3}
        weighted = ProvenanceModelSet(candidate_models(), model_weights=weights)
        start = {"a": True}
        result = value_of_provenance_information(
            PRIOR, weighted, ACTIONS, UTILITIES, horizon=1, initial_observations=start
        )
        updated = weighted.model_averaged_posterior(PRIOR, start).model_weights
        for name in weights:
            self.assertAlmostEqual(result.model_weights[name], updated[name], delta=1e-12)
        bayes = optimal_collaborative_plan(
            PRIOR, weighted, ACTIONS, UTILITIES, horizon=1, initial_observations=start
        ).value
        self.assertAlmostEqual(result.bayes_value, bayes, delta=1e-12)

    def test_identical_models_have_no_provenance_value(self):
        net = candidate_models()["copied"]
        twins = ProvenanceModelSet({"one": net, "two": net}, model_weights={"one": 0.3, "two": 0.7})
        result = value_of_provenance_information(PRIOR, twins, ACTIONS, UTILITIES, horizon=2)
        self.assertAlmostEqual(result.value, 0.0, delta=1e-12)
        plan = robust_collaborative_plan(PRIOR, twins, ACTIONS, UTILITIES, horizon=2,
                                         criterion="minimax_regret")
        self.assertAlmostEqual(plan.guarantee, 0.0, delta=1e-12)

    def test_weight_bounds_restrict_the_adversary(self):
        lower = {"copied": 0.2, "partial": 0.1, "shared_audit": 0.2}
        upper = {"copied": 0.6, "partial": 0.4, "shared_audit": 0.7}
        plan = robust_collaborative_plan(PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2,
                                         lower=lower, upper=upper)
        vertices = []
        for free in ORACLE.names:
            others = [n for n in ORACLE.names if n != free]
            for picks in product((lower, upper), repeat=2):
                w = {n: pick[n] for n, pick in zip(others, picks)}
                w[free] = 1 - sum(w.values())
                if lower[free] - 1e-12 <= w[free] <= upper[free] + 1e-12:
                    vertices.append(w)
        matrix = [[sum(w[n] * v[n] for n in ORACLE.names) for v in ORACLE.values] for w in vertices]
        best, _, _ = solve_matrix_game(matrix)
        self.assertAlmostEqual(plan.guarantee, best, delta=1e-12)
        for name, value in plan.least_favourable_weights.items():
            self.assertGreaterEqual(value, lower[name] - 1e-12)
            self.assertLessEqual(value, upper[name] + 1e-12)
        unrestricted = robust_collaborative_plan(PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2)
        self.assertGreaterEqual(plan.guarantee, unrestricted.guarantee - 1e-12)

    def test_point_bounds_reduce_to_the_bayes_plan(self):
        w = {"copied": 0.5, "partial": 0.2, "shared_audit": 0.3}
        plan = robust_collaborative_plan(PRIOR, self.MODEL_SET, ACTIONS, UTILITIES, horizon=2,
                                         lower=w, upper=w)
        self.assertAlmostEqual(plan.guarantee, ORACLE.bayes_value(w), delta=1e-12)


class RandomizationTest(unittest.TestCase):
    """A two-graph case in which no single plan reaches the minimax regret."""

    def test_mixing_strictly_improves_on_every_single_plan(self):
        copied = ProvenanceNetwork()
        copied.add(symmetric_primary("study", 0.75))
        for k in range(3):
            copied.add(relay(f"r{k}", "study", 0.95))
        copied.add(symmetric_primary("check", 0.8))
        independent = ProvenanceNetwork()
        independent.add(symmetric_primary("study", 0.75))
        for k in range(3):
            independent.add(symmetric_primary(f"r{k}", 0.75))
        independent.add(symmetric_primary("check", 0.8))
        models = ProvenanceModelSet({"copied": copied, "independent": independent})
        actions = [CollaborativeAction(f"ask{k}", f"r{k}", "reader", cost=0.02) for k in range(3)]
        actions.append(CollaborativeAction("audit", "check", "auditor", cost=0.08))
        utilities = {"adopt": {True: 1.0, False: -1.0}, "reject": {True: -0.5, False: 0.5}}
        plan = robust_collaborative_plan(0.5, models, actions, utilities, horizon=3,
                                         criterion="minimax_regret")
        self.assertLess(plan.guarantee, plan.deterministic_guarantee - 1e-3)
        self.assertEqual(len([p for p in plan.probabilities if p > 0]), 2)


class RobustPlanValidationTest(unittest.TestCase):
    def test_rejects_bad_arguments(self):
        models = ProvenanceModelSet(candidate_models())
        with self.assertRaises(ValueError):
            robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=2, criterion="hurwicz")
        with self.assertRaises(ValueError):
            robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=2,
                                      lower={"copied": 0, "partial": 0, "shared_audit": 0})
        with self.assertRaises(ValueError):
            robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=2,
                                      lower={"copied": 0.5, "partial": 0.5, "shared_audit": 0.5},
                                      upper={"copied": 1, "partial": 1, "shared_audit": 1})
        with self.assertRaises(TypeError):
            robust_collaborative_plan(PRIOR, candidate_models()["copied"], ACTIONS, UTILITIES, horizon=2)
        with self.assertRaises(ValueError):
            value_of_provenance_information(PRIOR, models, ACTIONS, UTILITIES, horizon=2)

    def test_refuted_models_are_excluded_or_rejected(self):
        certain = ProvenanceNetwork()
        certain.add(symmetric_primary("study", 0.75))
        certain.add(relay("a", "study", 1.0))
        certain.add(relay("b", "study", 1.0))
        certain.add(symmetric_primary("check", 0.8))
        models = ProvenanceModelSet({"certain": certain, **candidate_models()})
        start = {"a": True, "b": False}   # impossible if both relay one study perfectly
        plan = robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=1,
                                         initial_observations=start)
        self.assertEqual(plan.excluded_models, ("certain",))
        bounds = {"certain": 0.0, "copied": 0.0, "partial": 0.0, "shared_audit": 0.0}
        with self.assertRaises(ValueError):
            robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=1,
                                      initial_observations=start, lower=bounds,
                                      upper={name: 1.0 for name in bounds})


class CredalPosteriorTest(unittest.TestCase):
    def setUp(self):
        self.models = ProvenanceModelSet(candidate_models())
        self.observed = {"a": True, "b": True}

    def brute_force(self, lower, upper, steps=60):
        names = self.models.model_names
        posteriors = {n: self.models.posterior_envelope(PRIOR, self.observed).by_model[n] for n in names}
        evidence = {}
        for name in names:
            rows = joint_table(self.models.model(name), PRIOR)
            evidence[name] = conditional(rows, self.observed)[0]
        values = []
        for a, b in product(range(steps + 1), repeat=2):
            w = dict(zip(names, (a / steps, b / steps, 1 - a / steps - b / steps)))
            if all(lower[n] - 1e-12 <= w[n] <= upper[n] + 1e-12 for n in names):
                z = sum(w[n] * evidence[n] for n in names)
                values.append(sum(w[n] * evidence[n] * posteriors[n] for n in names) / z)
        return min(values), max(values)

    def test_full_bounds_give_the_envelope_and_point_bounds_give_the_average(self):
        names = self.models.model_names
        full = self.models.credal_posterior(PRIOR, self.observed,
                                            lower=dict.fromkeys(names, 0.0), upper=dict.fromkeys(names, 1.0))
        envelope = self.models.posterior_envelope(PRIOR, self.observed)
        self.assertAlmostEqual(full.lower, envelope.lower, delta=1e-14)
        self.assertAlmostEqual(full.upper, envelope.upper, delta=1e-14)
        w = {"copied": 0.5, "partial": 0.2, "shared_audit": 0.3}
        point = self.models.credal_posterior(PRIOR, self.observed, lower=w, upper=w)
        averaged = ProvenanceModelSet(candidate_models(), model_weights=w).posterior(PRIOR, self.observed)
        self.assertAlmostEqual(point.lower, averaged, delta=1e-14)
        self.assertAlmostEqual(point.upper, averaged, delta=1e-14)

    def test_interval_bounds_against_a_fine_grid(self):
        lower = {"copied": 0.2, "partial": 0.0, "shared_audit": 0.1}
        upper = {"copied": 0.7, "partial": 0.5, "shared_audit": 0.6}
        result = self.models.credal_posterior(PRIOR, self.observed, lower=lower, upper=upper)
        grid_low, grid_high = self.brute_force(lower, upper)
        # Grid points with step 1/60 hit every vertex of this polytope exactly.
        self.assertAlmostEqual(result.lower, grid_low, delta=1e-12)
        self.assertAlmostEqual(result.upper, grid_high, delta=1e-12)
        for weights, target in ((result.weights_at_lower, result.lower),
                                (result.weights_at_upper, result.upper)):
            self.assertAlmostEqual(sum(weights.values()), 1.0, delta=1e-12)
            averaged = ProvenanceModelSet(candidate_models(), model_weights=dict(weights))
            self.assertAlmostEqual(averaged.posterior(PRIOR, self.observed), target, delta=1e-12)

    def test_rejects_infeasible_bounds(self):
        names = self.models.model_names
        with self.assertRaises(ValueError):
            self.models.credal_posterior(PRIOR, self.observed,
                                         lower=dict.fromkeys(names, 0.5), upper=dict.fromkeys(names, 1.0))
        with self.assertRaises(ValueError):
            self.models.credal_posterior(PRIOR, self.observed,
                                         lower=dict.fromkeys(names, 0.0), upper=dict.fromkeys(names, 0.2))


class InformationGainTest(unittest.TestCase):
    def test_against_the_full_joint_distribution(self):
        weights = {"copied": 0.5, "partial": 0.2, "shared_audit": 0.3}
        models = ProvenanceModelSet(candidate_models(), model_weights=weights)
        observed = {"a": True}
        for target in ("b", "check", "study"):
            gain = models.information_gain(target, PRIOR, observed)
            joint = {}
            for name in weights:
                for mass, truth, assignment in joint_table(models.model(name), PRIOR):
                    if assignment["a"]:
                        key = (name, truth, assignment[target])
                        joint[key] = joint.get(key, 0.0) + weights[name] * mass
            total = sum(joint.values())
            joint = {k: v / total for k, v in joint.items()}

            def mutual(index):
                pair, left, right = {}, {}, {}
                for key, p in joint.items():
                    a, y = key[index], key[2]
                    pair[a, y] = pair.get((a, y), 0.0) + p
                    left[a] = left.get(a, 0.0) + p
                    right[y] = right.get(y, 0.0) + p
                return sum(p * log2(p / (left[a] * right[y])) for (a, y), p in pair.items() if p > 0)

            with self.subTest(target=target):
                self.assertAlmostEqual(gain.about_model, mutual(0), delta=1e-12)
                self.assertAlmostEqual(gain.about_claim, mutual(1), delta=1e-12)

    def test_a_node_shared_by_every_graph_says_nothing_about_the_graph(self):
        # "study" has the same CPD in every candidate, so before any report it
        # carries no information about which graph is true.
        models = ProvenanceModelSet(candidate_models(),
                                    model_weights={"copied": 0.4, "partial": 0.3, "shared_audit": 0.3})
        gain = models.information_gain("study", PRIOR, {})
        self.assertAlmostEqual(gain.about_model, 0.0, delta=1e-12)
        self.assertGreater(gain.about_claim, 0.1)
        self.assertEqual(models.information_gain("a", PRIOR, {"a": True}).about_model, 0.0)


if __name__ == "__main__":
    unittest.main()


class ReviewRegressionTest(unittest.TestCase):
    """Cases found in an independent review of the first version."""

    def test_refuted_model_with_some_prior_weight_is_allowed(self):
        certain = ProvenanceNetwork()
        certain.add(symmetric_primary("study", 0.75))
        certain.add(relay("a", "study", 1.0))
        certain.add(relay("b", "study", 1.0))
        certain.add(symmetric_primary("check", 0.8))
        models = ProvenanceModelSet({"certain": certain, **candidate_models()})
        start = {"a": True, "b": False}
        lower = {"certain": 0.0, "copied": 0.2, "partial": 0.2, "shared_audit": 0.2}
        upper = {"certain": 0.3, "copied": 1.0, "partial": 1.0, "shared_audit": 1.0}
        plan = robust_collaborative_plan(PRIOR, models, ACTIONS, UTILITIES, horizon=1,
                                         initial_observations=start, lower=lower, upper=upper)
        self.assertEqual(plan.least_favourable_weights.get("certain", 0.0), 0.0)
        self.assertEqual(plan.excluded_models, ("certain",))

    def test_bounds_summing_to_one_within_tolerance(self):
        names = ("copied", "partial", "shared_audit")
        lower = dict(zip(names, (0.3, 0.3, 0.4 + 5e-10)))
        plan = robust_collaborative_plan(PRIOR, ProvenanceModelSet(candidate_models()), ACTIONS,
                                         UTILITIES, horizon=1, lower=lower,
                                         upper=dict.fromkeys(names, 1.0))
        self.assertAlmostEqual(sum(plan.least_favourable_weights.values()), 1.0, delta=1e-12)

    def test_credal_range_when_the_best_supported_model_is_capped_at_zero(self):
        def one_report(p_true, p_false):
            net = ProvenanceNetwork()
            net.add(ReportNode("r", (), {(True, ()): p_true, (False, ()): p_false}))
            return net

        models = ProvenanceModelSet({"A": one_report(0.9, 0.9), "B": one_report(1e-20, 1e-22),
                                     "C": one_report(1e-22, 1e-20)})
        result = models.credal_posterior(0.5, {"r": True}, lower=dict.fromkeys("ABC", 0.0),
                                         upper={"A": 0.0, "B": 1.0, "C": 1.0})
        self.assertAlmostEqual(result.lower, 1 / 101, delta=1e-12)
        self.assertAlmostEqual(result.upper, 100 / 101, delta=1e-12)

    def test_information_gain_survives_underflowing_marginals(self):
        def one_report(p):
            net = ProvenanceNetwork()
            net.add(ReportNode("r", (), {(True, ()): p, (False, ()): p}))
            return net

        models = ProvenanceModelSet({"main": one_report(1e-170), "rare": one_report(0.5)},
                                    model_weights={"main": 1 - 1e-170, "rare": 1e-170})
        gain = models.information_gain("r", 0.5, {})
        self.assertLess(gain.about_model, 1e-100)
        self.assertEqual(gain.about_claim, 0.0)

    def test_information_gain_rejects_impossible_data_on_both_paths(self):
        certain = ProvenanceNetwork()
        certain.add(symmetric_primary("study", 0.75))
        certain.add(relay("a", "study", 1.0))
        certain.add(relay("b", "study", 1.0))
        models = ProvenanceModelSet({"certain": certain}, model_weights={"certain": 1.0})
        for target in ("a", "b"):
            with self.assertRaises(ValueError):
                models.information_gain(target, 0.5, {"a": True, "b": False})
