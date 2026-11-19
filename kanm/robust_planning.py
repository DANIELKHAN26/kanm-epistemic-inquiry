"""Inquiry plans for when the provenance graph itself is in doubt.

:func:`~kanm.collaborative.optimal_collaborative_plan` needs a probability for
each candidate graph. When such numbers cannot be defended, a plan has to be
chosen by a stated robust criterion instead. Two are offered here:

``"maximin"``
    Make the worst expected utility over the candidate graphs as large as
    possible (Wald, 1950; Gilboa & Schmeidler, 1989).
``"minimax_regret"``
    Make the worst expected shortfall, compared with someone told the true
    graph before starting, as small as possible (Savage, 1951).

One graph is true for the whole inquiry. The criterion is applied once, to
complete contingent plans, so the worst case cannot switch graphs from one
report to the next. Seen this way, the problem is a finite zero-sum game
between the plan and the choice of graph. By the minimax theorem the best
randomized plan is Bayes-optimal against a least favourable set of model
weights, and its guaranteed value equals the Bayes value under those weights.
The solver alternates between an exact matrix game over the plans found so far
and a Bayes plan against the current least favourable weights, which is the
double-oracle scheme of McMahan, Gordon & Blum (2003). It stops with a
certificate: the returned plan's guarantee and a bound that no plan can beat.

A randomized plan is drawn once, before the first query, and then followed.
Re-solving the criterion after some reports have arrived can recommend
something else; maximin and regret rules are not dynamically consistent in
general (Seidenfeld, 1988; Grünwald & Halpern, 2011), so the guarantee holds
for the plan as committed at the start.

The work is exact and exhaustive, like the Bayes planner, and is meant for
small declared problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import exp, fsum, inf, log, log1p
from types import MappingProxyType
from typing import Mapping, Sequence

from .collaborative import (
    MAX_COLLABORATIVE_STATES,
    CollaborativeAction,
    _validated_problem,
)
from .model_uncertainty import ProvenanceModelSet

StateKey = tuple[tuple[str, bool], ...]
_TIE_TOLERANCE = 1e-12
MAX_VERTEX_MODELS = 12


@dataclass(frozen=True)
class InquiryPolicy:
    """A deterministic contingent plan.

    ``rules`` maps each reachable state, a sorted tuple of observed
    ``(report, value)`` pairs, to ``("ask", action_id)`` or
    ``("decide", decision)``.
    """

    root: StateKey
    rules: Mapping[StateKey, tuple[str, str]]

    @property
    def first_step(self) -> tuple[str, str]:
        return self.rules[self.root]


@dataclass(frozen=True)
class RobustPlan:
    """A randomized plan chosen by a robust criterion, with its certificate.

    For ``"maximin"``, ``guarantee`` is the worst expected utility of the
    randomized plan over the allowed model weights and ``bound`` is an upper
    bound on what any plan can guarantee. For ``"minimax_regret"`` both are
    regrets: ``guarantee`` is the plan's worst expected regret and ``bound`` a
    lower bound on any plan's. ``least_favourable_weights`` are model
    probabilities at the starting state. ``deterministic_policy`` indexes the
    best single policy found, which is not always as good as the mixture.
    """

    criterion: str
    guarantee: float
    bound: float
    policies: tuple[InquiryPolicy, ...]
    probabilities: tuple[float, ...]
    values: tuple[Mapping[str, float], ...]
    least_favourable_weights: Mapping[str, float]
    model_optimal_values: Mapping[str, float]
    deterministic_policy: int
    deterministic_guarantee: float
    excluded_models: tuple[str, ...]
    iterations: int

    @property
    def first_step_probabilities(self) -> Mapping[tuple[str, str], float]:
        """Probability of each first step under the randomized plan."""
        result: dict[tuple[str, str], float] = {}
        for policy, probability in zip(self.policies, self.probabilities):
            if probability > 0.0:
                result[policy.first_step] = result.get(policy.first_step, 0.0) + probability
        return MappingProxyType(result)


@dataclass(frozen=True)
class ProvenanceInformationValue:
    """Expected value of learning the true graph before the inquiry starts.

    ``value = clairvoyant_value - bayes_value``. This is the expected value of
    partial perfect information about model identity: the truth of the claim
    remains uncertain after the graph is revealed. It is never negative, and
    it bounds what an audit of the provenance could be worth to this decision.
    """

    value: float
    bayes_value: float
    clairvoyant_value: float
    model_weights: Mapping[str, float]
    model_optimal_values: Mapping[str, float]


def solve_matrix_game(
    payoff: Sequence[Sequence[float]],
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Solve a finite zero-sum game exactly.

    ``payoff[i][j]`` is paid to the column player, who maximizes, when the row
    player, who minimizes, picks ``i``. Returns ``(value, column_mix, row_mix)``.
    The linear program is solved by the simplex method with Bland's rule in
    exact rational arithmetic, so the result is the exact solution of the game
    with the given floating-point entries.
    """
    if not payoff or not payoff[0] or any(len(row) != len(payoff[0]) for row in payoff):
        raise ValueError("payoff must be a nonempty rectangular matrix")
    rows, columns = len(payoff), len(payoff[0])
    matrix = [[Fraction(value) for value in row] for row in payoff]
    shift = 1 - min(min(row) for row in matrix)
    # Row player: maximize sum(u) subject to sum_i (A[i][j] + shift) u[i] <= 1.
    tableau = []
    for j in range(columns):
        coefficients = [matrix[i][j] + shift for i in range(rows)]
        slacks = [Fraction(int(k == j)) for k in range(columns)]
        tableau.append(coefficients + slacks + [Fraction(1)])
    objective = [Fraction(-1)] * rows + [Fraction(0)] * (columns + 1)
    basis = [rows + j for j in range(columns)]
    while True:
        entering = next((k for k in range(rows + columns) if objective[k] < 0), None)
        if entering is None:
            break
        leaving = None
        for index, line in enumerate(tableau):
            if line[entering] > 0:
                ratio = line[-1] / line[entering]
                if (
                    leaving is None
                    or ratio < leaving[0]
                    or (ratio == leaving[0] and basis[index] < basis[leaving[1]])
                ):
                    leaving = (ratio, index)
        pivot_row = leaving[1]
        pivot = tableau[pivot_row][entering]
        tableau[pivot_row] = [value / pivot for value in tableau[pivot_row]]
        for index, line in enumerate(tableau):
            factor = line[entering]
            if index != pivot_row and factor != 0:
                tableau[index] = [a - factor * b for a, b in zip(line, tableau[pivot_row])]
        factor = objective[entering]
        objective = [a - factor * b for a, b in zip(objective, tableau[pivot_row])]
        basis[pivot_row] = entering
    game_value = 1 / objective[-1]
    row_mix = [Fraction(0)] * rows
    for index, variable in enumerate(basis):
        if variable < rows:
            row_mix[variable] = tableau[index][-1] * game_value
    column_mix = [objective[rows + j] * game_value for j in range(columns)]
    return (
        float(game_value - shift),
        tuple(float(value) for value in column_mix),
        tuple(float(value) for value in row_mix),
    )


class _Problem:
    """Exact Bayes planning and policy evaluation over a finite model set."""

    def __init__(self, prior, model_set, actions, utilities, horizon, cost_weight, observations):
        self.prior = prior
        self.models = {name: model_set.model(name) for name in model_set.model_names}
        self.actions = sorted(actions, key=lambda action: action.action_id)
        self.by_id = {action.action_id: action for action in actions}
        self.utilities = utilities
        self.decisions = sorted(utilities)
        self.cost_weight = float(cost_weight)
        self.root: StateKey = tuple(sorted(observations.items()))
        self.max_actions = min(horizon, len(actions))
        self._masses: dict[tuple[str, StateKey], tuple[float, float]] = {}
        self.supported = tuple(
            name for name in self.models if self.log_mass(name, self.root) > -inf
        )
        self.excluded = tuple(name for name in self.models if name not in self.supported)
        if not self.supported:
            raise ValueError("initial observations have zero probability under every model")

    def masses(self, name: str, state: StateKey) -> tuple[float, float]:
        """``(log P(H, state | m), log P(state | m))``."""
        key = (name, state)
        if key not in self._masses:
            if len(self._masses) > MAX_COLLABORATIVE_STATES:
                raise ValueError("planning problem exceeds exact state limit")
            observations = dict(state)
            model = self.models[name]
            joint_true = (
                log(self.prior) + model.log_evidence(observations, True)
                if self.prior > 0.0 else -inf
            )
            joint_false = (
                log1p(-self.prior) + model.log_evidence(observations, False)
                if self.prior < 1.0 else -inf
            )
            self._masses[key] = (joint_true, _log_sum_exp([joint_true, joint_false]))
        return self._masses[key]

    def log_mass(self, name: str, state: StateKey) -> float:
        return self.masses(name, state)[1]

    def eligible(self, state: StateKey) -> list[CollaborativeAction]:
        if len(state) - len(self.root) >= self.max_actions:
            return []
        observed = dict(state)
        return [
            action for action in self.actions
            if action.evidence_node not in observed
            and all(self.by_id[r].evidence_node in observed for r in action.requires)
        ]

    def children(self, state: StateKey, action: CollaborativeAction):
        for outcome in (False, True):
            child = tuple(sorted(dict(state, **{action.evidence_node: outcome}).items()))
            if any(self.log_mass(name, child) > -inf for name in self.supported):
                yield outcome, child

    def expected_decision_utility(self, decision: str, p_true: float) -> float:
        values = self.utilities[decision]
        return p_true * values[True] + (1.0 - p_true) * values[False]

    def bayes_policy(self, weights: Mapping[str, float]) -> tuple[InquiryPolicy, float]:
        """Bayes-optimal policy for model probabilities ``weights`` at the root.

        States that have probability zero under ``weights`` but not under every
        model still get a rule, planned with equal weights on the models that
        allow them, so the policy can be evaluated under any candidate.
        """
        root_masses = {name: self.log_mass(name, self.root) for name in self.supported}
        planning = {
            name: (log(weights[name]) - root_masses[name] if weights.get(name, 0.0) > 0.0 else -inf)
            for name in self.supported
        }
        fallback = {name: -root_masses[name] for name in self.supported}
        rules: dict[StateKey, tuple[str, str]] = {}
        cache: dict[StateKey, float] = {}

        def solve(state: StateKey) -> float:
            if state in cache:
                return cache[state]
            log_weights = planning
            evidence = _log_sum_exp([planning[n] + self.log_mass(n, state) for n in self.supported])
            if evidence == -inf:
                log_weights = fallback
                evidence = _log_sum_exp(
                    [fallback[n] + self.log_mass(n, state) for n in self.supported]
                )
            p_true = exp(_log_sum_exp(
                [log_weights[n] + self.masses(n, state)[0] for n in self.supported]
            ) - evidence)
            best_decision = self.decisions[0]
            best = self.expected_decision_utility(best_decision, p_true)
            for decision in self.decisions[1:]:
                candidate = self.expected_decision_utility(decision, p_true)
                if candidate > best + _TIE_TOLERANCE * max(abs(best), abs(candidate), 1e-300):
                    best, best_decision = candidate, decision
            rule = ("decide", best_decision)
            for action in self.eligible(state):
                value = -self.cost_weight * float(action.cost)
                for _, child in self.children(state, action):
                    child_value = solve(child)
                    child_mass = _log_sum_exp(
                        [log_weights[n] + self.log_mass(n, child) for n in self.supported]
                    )
                    if child_mass > -inf:
                        value += exp(child_mass - evidence) * child_value
                if value > best + _TIE_TOLERANCE * max(abs(best), abs(value), 1e-300):
                    best, rule = value, ("ask", action.action_id)
            rules[state] = rule
            cache[state] = best
            return best

        value = solve(self.root)
        return InquiryPolicy(self.root, MappingProxyType(self._reachable(rules))), value

    def _reachable(self, rules: Mapping[StateKey, tuple[str, str]]) -> dict[StateKey, tuple[str, str]]:
        kept: dict[StateKey, tuple[str, str]] = {}
        pending = [self.root]
        while pending:
            state = pending.pop()
            if state in kept:
                continue
            kept[state] = rules[state]
            kind, label = rules[state]
            if kind == "ask":
                pending.extend(child for _, child in self.children(state, self.by_id[label]))
        return dict(sorted(kept.items()))

    def evaluate(self, policy: InquiryPolicy, name: str) -> float:
        """Expected utility of ``policy`` if model ``name`` is the true one."""
        cache: dict[StateKey, float] = {}

        def value(state: StateKey) -> float:
            if state in cache:
                return cache[state]
            log_true, evidence = self.masses(name, state)
            kind, label = policy.rules[state]
            if kind == "decide":
                p_true = exp(log_true - evidence) if log_true > -inf else 0.0
                result = self.expected_decision_utility(label, p_true)
            else:
                action = self.by_id[label]
                result = -self.cost_weight * float(action.cost)
                for _, child in self.children(state, action):
                    child_mass = self.log_mass(name, child)
                    if child_mass > -inf:
                        result += exp(child_mass - evidence) * value(child)
            cache[state] = result
            return result

        return value(self.root)


def robust_collaborative_plan(
    prior: float,
    model_set: ProvenanceModelSet,
    actions: Sequence[CollaborativeAction],
    decision_utilities: Mapping[str, Mapping[bool, float]],
    *,
    horizon: int,
    criterion: str = "maximin",
    lower: Mapping[str, float] | None = None,
    upper: Mapping[str, float] | None = None,
    cost_weight: float = 1.0,
    initial_observations: Mapping[str, bool] | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> RobustPlan:
    """Choose a randomized inquiry plan by maximin or minimax regret.

    Without ``lower``/``upper`` any weighting of the candidate graphs is
    allowed, so the criterion ranges over the graphs themselves. With bounds,
    the allowed prior weights are those within the bounds that sum to one, and
    each is updated on ``initial_observations`` before the plan starts. The
    model set's own ``model_weights``, if any, are not used.

    Regret is measured against a planner who is told the true graph, and is
    averaged over the allowed weights. Its minimax value therefore equals the
    largest expected value of perfect information about the graph over the
    allowed weights (see :func:`value_of_provenance_information`).
    """
    if not isinstance(model_set, ProvenanceModelSet):
        raise TypeError("model_set must be a ProvenanceModelSet")
    if criterion not in ("maximin", "minimax_regret"):
        raise ValueError("criterion must be 'maximin' or 'minimax_regret'")
    if (lower is None) != (upper is None):
        raise ValueError("give both lower and upper weight bounds, or neither")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not (isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and tolerance >= 0.0):
        raise ValueError("tolerance must be a nonnegative number")
    prior, utilities, _, observations = _validated_problem(
        prior, model_set, actions, decision_utilities, horizon, cost_weight, initial_observations
    )
    problem = _Problem(prior, model_set, actions, utilities, horizon, cost_weight, observations)
    names = problem.supported

    if lower is None:
        scenarios = [{name: float(name == other) for name in names} for other in names]
    else:
        low, high = model_set._weight_bounds(lower, upper)
        if len(model_set.model_names) > MAX_VERTEX_MODELS:
            raise ValueError(f"weight bounds are supported for at most {MAX_VERTEX_MODELS} models")
        scenarios = []
        for vertex in _box_simplex_vertices(model_set.model_names, low, high):
            # A refuted model may carry prior weight; it simply ends with none.
            # Only a prior with all its weight on refuted models is undefined.
            if not any(vertex[name] > 0.0 for name in names):
                raise ValueError(
                    "some allowed weights put all their mass on models that the initial "
                    "observations refute; tighten the bounds on the excluded models"
                )
            scenarios.append(_update_weights(problem, vertex))
        if not scenarios:
            raise ValueError("no weights within these bounds sum to 1")

    optimal = {name: problem.bayes_policy({name: 1.0})[1] for name in names}
    sign = 1.0 if criterion == "maximin" else -1.0

    def payoff(values: Mapping[str, float], scenario: Mapping[str, float]) -> float:
        expected = fsum(scenario[name] * values[name] for name in names)
        if criterion == "maximin":
            return expected
        return expected - fsum(scenario[name] * optimal[name] for name in names)

    policies: list[InquiryPolicy] = []
    values: list[dict[str, float]] = []

    def add(policy: InquiryPolicy) -> bool:
        if any(policy.rules == known.rules for known in policies):
            return False
        policies.append(policy)
        values.append({name: problem.evaluate(policy, name) for name in names})
        return True

    for name in names:
        add(problem.bayes_policy({name: 1.0})[0])
    add(problem.bayes_policy({name: 1.0 / len(names) for name in names})[0])

    iterations = 0
    while True:
        iterations += 1
        matrix = [[payoff(v, scenario) for v in values] for scenario in scenarios]
        game_value, policy_mix, scenario_mix = solve_matrix_game(matrix)
        weights = {
            name: fsum(p * scenario[name] for p, scenario in zip(scenario_mix, scenarios))
            for name in names
        }
        total = fsum(weights.values())
        weights = {name: value / total for name, value in weights.items()}
        response, _ = problem.bayes_policy(weights)
        response_values = {name: problem.evaluate(response, name) for name in names}
        best_response = fsum(
            p * payoff(response_values, scenario) for p, scenario in zip(scenario_mix, scenarios)
        )
        converged = best_response <= game_value + tolerance * max(1.0, abs(game_value))
        if converged or not add(response) or iterations >= max_iterations:
            break

    guarantee = min(
        fsum(p * payoff(v, scenario) for p, v in zip(policy_mix, values))
        for scenario in scenarios
    )
    worst_cases = [min(payoff(v, scenario) for scenario in scenarios) for v in values]
    best_single = max(range(len(values)), key=lambda index: (worst_cases[index], -index))
    probabilities = tuple(policy_mix) + (0.0,) * (len(policies) - len(policy_mix))
    return RobustPlan(
        criterion=criterion,
        guarantee=sign * guarantee + 0.0,
        bound=sign * max(best_response, game_value) + 0.0,
        policies=tuple(policies),
        probabilities=probabilities,
        values=tuple(MappingProxyType(v) for v in values),
        least_favourable_weights=MappingProxyType(weights),
        model_optimal_values=MappingProxyType(optimal),
        deterministic_policy=best_single,
        deterministic_guarantee=sign * worst_cases[best_single] + 0.0,
        excluded_models=problem.excluded,
        iterations=iterations,
    )


def value_of_provenance_information(
    prior: float,
    model_set: ProvenanceModelSet,
    actions: Sequence[CollaborativeAction],
    decision_utilities: Mapping[str, Mapping[bool, float]],
    *,
    horizon: int,
    cost_weight: float = 1.0,
    initial_observations: Mapping[str, bool] | None = None,
) -> ProvenanceInformationValue:
    """Expected value of learning which candidate graph is true, before inquiring.

    Uses the model set's explicit ``model_weights``, updated on
    ``initial_observations``. The clairvoyant planner still has to inquire
    about the claim; only the graph is revealed.
    """
    if not isinstance(model_set, ProvenanceModelSet):
        raise TypeError("model_set must be a ProvenanceModelSet")
    if model_set.model_weights is None:
        raise ValueError("the value of provenance information needs explicit model_weights")
    prior, utilities, _, observations = _validated_problem(
        prior, model_set, actions, decision_utilities, horizon, cost_weight, initial_observations
    )
    problem = _Problem(prior, model_set, actions, utilities, horizon, cost_weight, observations)
    weights = _update_weights(problem, model_set.model_weights)
    _, bayes_value = problem.bayes_policy(weights)
    optimal = {
        name: problem.bayes_policy({name: 1.0})[1]
        for name in problem.supported if weights[name] > 0.0
    }
    clairvoyant = fsum(weights[name] * optimal[name] for name in optimal)
    return ProvenanceInformationValue(
        value=max(0.0, clairvoyant - bayes_value),
        bayes_value=bayes_value,
        clairvoyant_value=clairvoyant,
        model_weights=MappingProxyType(weights),
        model_optimal_values=MappingProxyType(optimal),
    )


def _update_weights(problem: _Problem, prior_weights: Mapping[str, float]) -> dict[str, float]:
    """Model probabilities at the root, after the initial observations."""
    log_masses = {
        name: log(prior_weights[name]) + problem.log_mass(name, problem.root)
        for name in problem.supported if prior_weights.get(name, 0.0) > 0.0
    }
    normalizer = _log_sum_exp(list(log_masses.values()))
    if normalizer == -inf:
        raise ValueError("initial observations have zero probability under the weighted models")
    return {
        name: exp(log_masses[name] - normalizer) if name in log_masses else 0.0
        for name in problem.supported
    }


def _box_simplex_vertices(
    names: Sequence[str], low: Mapping[str, float], high: Mapping[str, float]
) -> list[dict[str, float]]:
    """Vertices of ``{w : low <= w <= high, sum(w) == 1}``.

    At a vertex every coordinate but at most one sits at a bound, so each
    vertex is found by fixing all but one coordinate at a bound and solving
    for the last.
    """
    vertices: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    for free in names:
        others = [name for name in names if name != free]
        for choice in product((False, True), repeat=len(others)):
            vertex = {name: (high if up else low)[name] for name, up in zip(others, choice)}
            rest = 1.0 - fsum(vertex.values())
            # Same tolerance as the bound check in ProvenanceModelSet._weight_bounds.
            if low[free] - 1e-9 <= rest <= high[free] + 1e-9:
                vertex[free] = min(high[free], max(low[free], rest))
                total = fsum(vertex.values())
                vertex = {name: value / total for name, value in vertex.items()}
                key = tuple(round(vertex[name], 12) for name in names)
                if key not in seen:
                    seen.add(key)
                    vertices.append(vertex)
    return vertices


def _log_sum_exp(values: list[float]) -> float:
    finite = [value for value in values if value > -inf]
    if not finite:
        return -inf
    peak = max(finite)
    return peak + log(fsum(exp(value - peak) for value in finite))
