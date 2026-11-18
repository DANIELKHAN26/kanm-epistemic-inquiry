"""Brute-force oracles for provenance inference and collaborative planning.

Each check enumerates the full joint distribution of a small random report network, with no pruning. It then compares the result with KANM's implementation:
- the posterior;
- the next-report predictive distribution;
- the planner's value under exact backward induction.

The networks include extreme CPD entries (0 or 1) and endpoint priors, so impossible observations must raise rather than return a number.
"""

from itertools import product
import random
import unittest

from kanm.collaborative import CollaborativeAction, optimal_collaborative_plan
from kanm.dependence import ProvenanceNetwork, ReportNode

TOLERANCE = 1e-9


def random_network(rng: random.Random, size: int, extreme: bool) -> ProvenanceNetwork:
    network = ProvenanceNetwork()
    ids: list[str] = []
    for index in range(size):
        parents = tuple(rng.sample(ids, rng.randint(0, min(3, len(ids)))))
        cpd = {}
        for truth in (True, False):
            for values in product((True, False), repeat=len(parents)):
                if extreme and rng.random() < 0.3:
                    cpd[(truth, values)] = rng.choice([0.0, 1.0])
                else:
                    cpd[(truth, values)] = rng.random()
        node_id = f"n{index}"
        network.add(ReportNode(node_id, parents, cpd))
        ids.append(node_id)
    return network


def joint_table(network: ProvenanceNetwork, prior: float):
    ids = list(network.nodes)
    table = []
    for truth in (True, False):
        for values in product((True, False), repeat=len(ids)):
            assignment = dict(zip(ids, values))
            weight = prior if truth else 1.0 - prior
            for node_id in ids:
                node = network.nodes[node_id]
                weight *= node.probability(assignment[node_id], truth,
                                           tuple(assignment[p] for p in node.parents))
            table.append((weight, truth, assignment))
    return table


def conditional(table, observations, target=None):
    """Return (P(obs), P(H | obs), P(target | obs)) by summation."""
    total = truth_mass = 0.0
    target_mass = {True: 0.0, False: 0.0}
    for weight, truth, assignment in table:
        if all(assignment[k] == v for k, v in observations.items()):
            total += weight
            truth_mass += weight if truth else 0.0
            if target is not None:
                target_mass[assignment[target]] += weight
    if total == 0.0:
        return 0.0, None, None
    return total, truth_mass / total, {k: v / total for k, v in target_mass.items()}


def brute_force_value(table, prior_actions, utilities, horizon, cost_weight, initial):
    actions = {action.action_id: action for action in prior_actions}

    def value(state, remaining):
        total, posterior, _ = conditional(table, state)
        best = max(posterior * u[True] + (1.0 - posterior) * u[False] for u in utilities.values())
        if remaining == 0:
            return best
        for action in prior_actions:
            if action.evidence_node in state:
                continue
            if any(actions[r].evidence_node not in state for r in action.requires):
                continue
            candidate = -cost_weight * action.cost
            for outcome in (False, True):
                successor = dict(state, **{action.evidence_node: outcome})
                p_successor, _, _ = conditional(table, successor)
                if p_successor > 0.0:
                    candidate += p_successor / total * value(successor, remaining - 1)
            best = max(best, candidate)
        return best

    return value(dict(initial), min(horizon, len(prior_actions)))


class ProvenanceOracleTest(unittest.TestCase):
    def test_posterior_and_predictive_match_full_enumeration(self):
        rng = random.Random(1)
        for trial in range(400):
            size = rng.randint(1, 5)
            network = random_network(rng, size, extreme=trial % 3 == 0)
            prior = rng.choice([0.0, 1.0, rng.random(), rng.random()])
            ids = list(network.nodes)
            observations = {k: rng.random() < 0.5 for k in rng.sample(ids, rng.randint(0, size))}
            target = rng.choice(ids)
            table = joint_table(network, prior)
            total, posterior, predictive = conditional(table, observations, target)
            with self.subTest(trial=trial):
                if total == 0.0:
                    with self.assertRaises(ValueError):
                        network.posterior(prior, observations)
                    with self.assertRaises(ValueError):
                        network.predictive_distribution(target, prior, observations)
                    continue
                self.assertAlmostEqual(network.posterior(prior, observations), posterior,
                                       delta=TOLERANCE)
                result = network.predictive_distribution(target, prior, observations)
                for value in (True, False):
                    self.assertAlmostEqual(result[value], predictive[value], delta=TOLERANCE)


class PlannerOracleTest(unittest.TestCase):
    def test_planner_value_matches_backward_induction_over_full_joint(self):
        rng = random.Random(7)
        checked = 0
        for trial in range(80):
            size = rng.randint(1, 4)
            network = random_network(rng, size, extreme=trial % 4 == 0)
            ids = list(network.nodes)
            chosen = rng.sample(ids, rng.randint(1, size))
            actions: list[CollaborativeAction] = []
            for index, node_id in enumerate(chosen):
                requires = tuple(rng.sample([a.action_id for a in actions],
                                            rng.randint(0, min(1, len(actions)))))
                actions.append(CollaborativeAction(f"a{index}", node_id, "role",
                                                   cost=rng.choice([0.0, 0.05, 0.2]),
                                                   requires=requires))
            utilities = {name: {True: rng.uniform(-2, 2), False: rng.uniform(-2, 2)}
                         for name in ("adopt", "reject")}
            prior = rng.choice([0.5, rng.random()])
            others = [x for x in ids if x not in chosen]
            initial = {k: rng.random() < 0.5 for k in rng.sample(others, rng.randint(0, len(others)))}
            horizon = rng.randint(0, 3)
            cost_weight = rng.choice([0.5, 1.0])
            table = joint_table(network, prior)
            if conditional(table, initial)[0] == 0.0:
                continue
            plan = optimal_collaborative_plan(prior, network, actions, utilities, horizon=horizon,
                                              cost_weight=cost_weight,
                                              initial_observations=initial)
            expected = brute_force_value(table, actions, utilities, horizon, cost_weight, initial)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(plan.value, expected, delta=TOLERANCE)
            checked += 1
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
