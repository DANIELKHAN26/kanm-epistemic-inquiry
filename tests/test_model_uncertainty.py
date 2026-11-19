"""Brute-force checks for inference across alternative provenance models."""

from itertools import product
from functools import lru_cache
from math import exp
import unittest

from kanm.collaborative import CollaborativeAction, optimal_collaborative_plan
from kanm.dependence import ProvenanceNetwork, ReportNode, relay, symmetric_primary
from kanm.model_uncertainty import ProvenanceModelSet


def _table(network: ProvenanceNetwork, prior: float, model_weight: float = 1.0):
    node_ids = list(network.nodes)
    result = []
    for truth in (True, False):
        for values in product((True, False), repeat=len(node_ids)):
            assignment = dict(zip(node_ids, values))
            mass = model_weight * (prior if truth else 1.0 - prior)
            for node_id in node_ids:
                node = network.nodes[node_id]
                mass *= node.probability(
                    assignment[node_id],
                    truth,
                    tuple(assignment[parent] for parent in node.parents),
                )
            result.append((mass, truth, assignment))
    return result


def _condition(table, observations, target=None):
    total = true_mass = target_true = 0.0
    for mass, truth, assignment in table:
        if all(assignment[key] == value for key, value in observations.items()):
            total += mass
            if truth:
                true_mass += mass
            if target is not None and assignment[target]:
                target_true += mass
    if total == 0.0:
        return None
    result = {"posterior": true_mass / total}
    if target is not None:
        result["predictive_true"] = target_true / total
    return result


def _brute_force_plan_value(table, actions, utilities, horizon, cost_weight):
    by_id = {action.action_id: action for action in actions}

    @lru_cache(maxsize=None)
    def value(state_key, remaining):
        state = dict(state_key)
        conditioned = _condition(table, state)
        if conditioned is None:
            return 0.0
        p_true = conditioned["posterior"]
        best = max(
            p_true * utility[True] + (1.0 - p_true) * utility[False]
            for utility in utilities.values()
        )
        if remaining == 0:
            return best
        for action in actions:
            if action.evidence_node in state:
                continue
            if any(by_id[required].evidence_node not in state for required in action.requires):
                continue
            next_distribution = _condition(table, state, action.evidence_node)
            candidate = -cost_weight * action.cost
            for outcome in (False, True):
                probability = (
                    next_distribution["predictive_true"]
                    if outcome
                    else 1.0 - next_distribution["predictive_true"]
                )
                if probability == 0.0:
                    continue
                child = tuple(sorted((*state.items(), (action.evidence_node, outcome))))
                candidate += probability * value(child, remaining - 1)
            best = max(best, candidate)
        return best

    return value((), min(horizon, len(actions)))


def _candidate_networks():
    copied = ProvenanceNetwork()
    copied.add(symmetric_primary("a", 0.72))
    copied.add(relay("b", "a", 0.85))
    copied.add(symmetric_primary("c", 0.61))

    mixed = ProvenanceNetwork()
    mixed.add(symmetric_primary("a", 0.86))
    mixed.add(symmetric_primary("b", 0.65))
    mixed.add(ReportNode("c", ("a",), {
        (truth, (parent,)): 0.4 * (0.77 if truth else 0.23)
        + 0.6 * (0.91 if parent else 0.09)
        for truth in (True, False)
        for parent in (True, False)
    }))

    chain = ProvenanceNetwork()
    chain.add(symmetric_primary("a", 0.59))
    chain.add(relay("b", "a", 0.55))
    chain.add(relay("c", "b", 0.8))
    return {"copied": copied, "mixed": mixed, "chain": chain}


class ProvenanceModelSetTests(unittest.TestCase):
    def setUp(self):
        self.networks = _candidate_networks()
        self.weights = {"copied": 0.2, "mixed": 0.5, "chain": 0.3}
        self.models = ProvenanceModelSet(self.networks, model_weights=self.weights)
        self.prior = 0.37
        self.observations = {"a": True, "b": False}

    def test_weighted_posterior_and_predictive_match_full_enumeration(self):
        joint = []
        per_model = {}
        for name, network in self.networks.items():
            model_table = _table(network, self.prior)
            per_model[name] = _condition(model_table, self.observations)["posterior"]
            joint.extend(_table(network, self.prior, self.weights[name]))

        expected = _condition(joint, self.observations)
        result = self.models.model_averaged_posterior(self.prior, self.observations)
        self.assertAlmostEqual(result.posterior, expected["posterior"], places=12)
        for name, value in per_model.items():
            self.assertAlmostEqual(result.posterior_by_model[name], value, places=12)
        self.assertAlmostEqual(sum(result.model_weights.values()), 1.0, places=12)

        prediction = self.models.model_averaged_predictive("c", self.prior, self.observations)
        expected_prediction = _condition(joint, self.observations, "c")["predictive_true"]
        self.assertAlmostEqual(prediction.probabilities[True], expected_prediction, places=12)
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=12)

    def test_posterior_envelope_matches_candidate_posteriors(self):
        result = self.models.posterior_envelope(self.prior, self.observations)
        expected = [
            _condition(_table(network, self.prior), self.observations)["posterior"]
            for network in self.networks.values()
        ]
        self.assertAlmostEqual(result.lower, min(expected), places=12)
        self.assertAlmostEqual(result.upper, max(expected), places=12)
        self.assertEqual(set(result.by_model), set(self.networks))

        predictive = self.models.predictive_envelope("c", self.prior, self.observations)
        expected_predictions = [
            _condition(_table(network, self.prior), self.observations, "c")["predictive_true"]
            for network in self.networks.values()
        ]
        self.assertAlmostEqual(predictive.lower_true, min(expected_predictions), places=12)
        self.assertAlmostEqual(predictive.upper_true, max(expected_predictions), places=12)

    def test_empty_observations_preserve_prior_and_model_weights(self):
        result = self.models.model_averaged_posterior(self.prior, {})
        self.assertAlmostEqual(result.posterior, self.prior, places=12)
        for name, weight in self.weights.items():
            self.assertAlmostEqual(result.model_weights[name], weight, places=12)
        self.assertEqual(self.models.model_averaged_posterior(0.0, {}).posterior, 0.0)
        self.assertEqual(self.models.model_averaged_posterior(1.0, {}).posterior, 1.0)

    def test_set_envelope_includes_supported_zero_weight_candidates(self):
        weighted = ProvenanceModelSet(
            self.networks,
            model_weights={"copied": 1.0, "mixed": 0.0, "chain": 0.0},
        )
        envelope = weighted.posterior_envelope(self.prior, self.observations)
        average = weighted.model_averaged_posterior(self.prior, self.observations)
        self.assertEqual(set(envelope.by_model), set(self.networks))
        self.assertEqual(set(average.posterior_by_model), {"copied"})
        self.assertEqual(set(average.excluded_models), {"mixed", "chain"})

    def test_candidate_snapshots_and_model_copies_are_isolated(self):
        original = self.networks["copied"]
        model_set = ProvenanceModelSet({"copied": original})
        original.add(symmetric_primary("later", 0.8))
        self.assertNotIn("later", model_set.node_ids)
        returned_copy = model_set.model("copied")
        returned_copy.add(symmetric_primary("local_edit", 0.8))
        self.assertNotIn("local_edit", model_set.node_ids)

    def test_model_set_rejects_incompatible_ids_and_bad_weights(self):
        first = ProvenanceNetwork()
        first.add(symmetric_primary("a", 0.8))
        second = ProvenanceNetwork()
        second.add(symmetric_primary("b", 0.8))
        with self.assertRaisesRegex(ValueError, "same report IDs"):
            ProvenanceModelSet({"first": first, "second": second})
        with self.assertRaisesRegex(ValueError, "exactly the candidate"):
            ProvenanceModelSet({"first": first}, model_weights={"other": 1.0})
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            ProvenanceModelSet({"first": first}, model_weights={"first": 0.8})

    def test_log_space_model_update_survives_tiny_joint_likelihoods(self):
        models = {}
        for name, sensitivity in (("lower", 0.01), ("higher", 0.02)):
            network = ProvenanceNetwork()
            for index in range(150):
                network.add(ReportNode(
                    f"r{index}", (),
                    {(True, ()): sensitivity, (False, ()): 0.001},
                ))
            models[name] = network
        model_set = ProvenanceModelSet(
            models, model_weights={"lower": 0.5, "higher": 0.5}
        )
        observations = {f"r{index}": True for index in range(150)}
        posterior = model_set.model_averaged_posterior(0.5, observations)
        self.assertTrue(0.0 <= posterior.posterior <= 1.0)
        self.assertAlmostEqual(sum(posterior.model_weights.values()), 1.0, places=12)
        self.assertGreater(posterior.model_weights["higher"], 0.99)

    def test_independence_predictive_uses_log_marginals(self):
        rare = ProvenanceNetwork()
        ordinary = ProvenanceNetwork()
        for index in range(16):
            node_id = f"r{index}"
            parents = () if index == 0 else (f"r{index - 1}",)
            rare_cpd = {}
            ordinary_cpd = {}
            for truth in (True, False):
                rare_p = 1e-30 if truth else 2e-30
                ordinary_p = 0.2 if truth else 0.4
                for parent_value in ((None,) if index == 0 else (True, False)):
                    key = (truth, ()) if index == 0 else (truth, (parent_value,))
                    allow = parent_value is not False
                    rare_cpd[key] = rare_p if allow else 0.0
                    ordinary_cpd[key] = ordinary_p if allow else 0.0
            rare.add(ReportNode(node_id, parents, rare_cpd))
            ordinary.add(ReportNode(node_id, parents, ordinary_cpd))
        models = ProvenanceModelSet(
            {"rare": rare, "ordinary": ordinary},
            model_weights={"rare": 0.5, "ordinary": 0.5},
        )
        prediction = models.independence_predictive_distribution("r15", 0.5, {})
        self.assertGreaterEqual(prediction[True], 0.0)
        self.assertLessEqual(prediction[True], 1.0)
        self.assertAlmostEqual(sum(prediction.values()), 1.0, places=15)

    def test_independence_predictive_matches_explicit_marginal_mixture(self):
        posterior_true = posterior_false = 0.0
        audit_true = audit_false = 0.0
        for name, network in self.networks.items():
            weight = self.weights[name]
            for node_id, outcome, target in (
                ("b", True, False),
                ("c", True, True),
            ):
                likelihood_true = network.log_evidence({node_id: outcome}, True)
                likelihood_false = network.log_evidence({node_id: outcome}, False)
                if target:
                    audit_true += weight * exp(likelihood_true)
                    audit_false += weight * exp(likelihood_false)
                else:
                    posterior_true += weight * exp(likelihood_true)
                    posterior_false += weight * exp(likelihood_false)
        truth_weight = (
            self.prior * posterior_true
            / (self.prior * posterior_true + (1.0 - self.prior) * posterior_false)
        )
        expected_true = truth_weight * audit_true + (1.0 - truth_weight) * audit_false
        prediction = self.models.independence_predictive_distribution(
            "c", self.prior, {"b": True}
        )
        self.assertAlmostEqual(prediction[True], expected_true, places=12)
        self.assertAlmostEqual(prediction[False], 1.0 - expected_true, places=12)

    def test_model_weights_update_after_evidence_and_excluded_models_are_reported(self):
        certain_true = ProvenanceNetwork()
        certain_true.add(symmetric_primary("x", 1.0))
        certain_false = ProvenanceNetwork()
        certain_false.add(ReportNode("x", (), {(True, ()): 0.0, (False, ()): 0.0}))
        model_set = ProvenanceModelSet(
            {"true_only": certain_true, "false_only": certain_false},
            model_weights={"true_only": 0.5, "false_only": 0.5},
        )
        envelope = model_set.posterior_envelope(0.5, {"x": True})
        self.assertEqual(envelope.by_model, {"true_only": 1.0})
        self.assertEqual(envelope.excluded_models, ("false_only",))
        average = model_set.model_averaged_posterior(0.5, {"x": True})
        self.assertEqual(dict(average.model_weights), {"true_only": 1.0})
        self.assertEqual(average.posterior, 1.0)
        self.assertEqual(average.excluded_models, ("false_only",))

    def test_unweighted_set_does_not_silently_average(self):
        model_set = ProvenanceModelSet(self.networks)
        self.assertLessEqual(
            model_set.posterior_envelope(self.prior, self.observations).lower,
            model_set.posterior_envelope(self.prior, self.observations).upper,
        )
        with self.assertRaisesRegex(ValueError, "explicit model_weights"):
            model_set.model_averaged_posterior(self.prior, self.observations)
        actions = [CollaborativeAction("check", "a", "replication")]
        with self.assertRaisesRegex(ValueError, "requires explicit model_weights"):
            optimal_collaborative_plan(
                self.prior, model_set, actions,
                {"accept": {True: 1.0, False: -1.0}}, horizon=1,
            )

    def test_planner_matches_enumeration_for_shared_weighted_model(self):
        actions = [
            CollaborativeAction("a_check", "a", "primary", 0.04),
            CollaborativeAction("b_check", "b", "follow_up", 0.03,
                                requires=("a_check",)),
            CollaborativeAction("c_check", "c", "replication", 0.05,
                                requires=("a_check",)),
        ]
        utilities = {
            "accept": {True: 1.0, False: -1.4},
            "reject": {True: -0.2, False: 0.3},
        }
        joint = []
        for name, network in self.networks.items():
            joint.extend(_table(network, self.prior, self.weights[name]))
        plan = optimal_collaborative_plan(
            self.prior, self.models, actions, utilities,
            horizon=2, cost_weight=0.8,
        )
        total = 0.0
        for mass, truth, _ in joint:
            total += mass
        self.assertAlmostEqual(total, 1.0, places=12)
        expected = _brute_force_plan_value(joint, actions, utilities, 2, 0.8)
        self.assertAlmostEqual(plan.value, expected, places=12)


if __name__ == "__main__":
    unittest.main()
