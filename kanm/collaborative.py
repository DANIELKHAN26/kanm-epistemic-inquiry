"""Cost-aware inquiry over reports with explicit evidence ancestry.

This module joins two existing KANM components: finite-horizon decision
planning and exact inference over a binary provenance network. It is a small
reference planner for declared models, not a learned policy or a new active
learning algorithm. Role labels are descriptive metadata; reliability and
dependence come only from the supplied network CPDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from .bayes import _probability
from .dependence import ProvenanceNetwork
from .model_uncertainty import ProvenanceModelSet

MAX_COLLABORATIVE_STATES = 2_000_000
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CollaborativeAction:
    """One feasible inquiry action that observes a report node.

    ``role_id`` records who or what performs the action (for example,
    ``independent_replication`` or ``methods_audit``). It has no automatic
    effect on the likelihood model. ``requires`` lists action IDs that must
    already have been observed; this is a procedural prerequisite, distinct
    from statistical dependence, which is represented by the report DAG.
    Each report node may be queried at most once by a plan.
    """

    action_id: str
    evidence_node: str
    role_id: str
    cost: float = 1.0
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("action_id", self.action_id),
            ("evidence_node", self.evidence_node),
            ("role_id", self.role_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, (int, float))
            or not isfinite(self.cost)
            or self.cost < 0.0
        ):
            raise ValueError("cost must be finite and >= 0")
        if not isinstance(self.requires, tuple) or any(
            not isinstance(action_id, str) or not action_id.strip()
            for action_id in self.requires
        ):
            raise ValueError("requires must be a tuple of nonempty action IDs")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("requires must not contain duplicates")
        if self.action_id in self.requires:
            raise ValueError("an action cannot require itself")


@dataclass(frozen=True)
class CollaborativePlan:
    """Exact finite-horizon result under the selected inference assumption."""

    first_action: str | None
    value: float
    value_if_stopping_now: float
    action_values: Mapping[str, float]
    horizon: int
    inference: str
    evaluated_states: int
    # Keys are sorted tuples of (observed evidence node, value). Values are
    # the optimal next action ID, or None when the policy stops at that state.
    policy: Mapping[tuple[tuple[str, bool], ...], str | None]

    @property
    def value_of_planning(self) -> float:
        """Expected gain from inquiry over taking a terminal decision now."""
        return self.value - self.value_if_stopping_now


def optimal_collaborative_plan(
    prior: float,
    network: ProvenanceNetwork | ProvenanceModelSet,
    actions: Sequence[CollaborativeAction],
    decision_utilities: Mapping[str, Mapping[bool, float]],
    *,
    horizon: int,
    cost_weight: float = 1.0,
    initial_observations: Mapping[str, bool] | None = None,
    inference: str = "joint",
) -> CollaborativePlan:
    """Choose an optimal sequential inquiry policy over dependent reports.

    The terminal decision maximizes expected utility under ``decision_utilities``.
    Each action has a cost in caller-defined units, scaled by ``cost_weight``.
    The planner may stop at any state and may query each listed report at most
    once. ``requires`` controls when an action becomes available; the network
    conditional-probability tables control how its outcome depends on truth
    and previous reports.

    ``inference="joint"`` conditions reports on the complete provenance DAG.
    A :class:`~kanm.model_uncertainty.ProvenanceModelSet` is also accepted; in
    that case ``joint`` uses its explicitly weighted Bayesian model average,
    and the planner updates model probabilities after each observed report.
    ``inference="independent"`` is a diagnostic comparator: it uses each
    report's exact marginal likelihood, then treats acquired reports as
    conditionally independent given truth. The latter can overcount relays.
    This is exhaustive backward induction over a finite plan and report DAG,
    not ECED or a scalable active-learning guarantee. The calculation refuses
    more than :data:`MAX_COLLABORATIVE_STATES` distinct states.
    """
    if not isinstance(network, (ProvenanceNetwork, ProvenanceModelSet)):
        raise TypeError("network must be a ProvenanceNetwork or ProvenanceModelSet")
    if isinstance(network, ProvenanceModelSet) and network.model_weights is None:
        raise ValueError(
            "collaborative planning over a ProvenanceModelSet requires explicit model_weights"
        )
    prior, utilities, by_id, observations = _validated_problem(
        prior, network, actions, decision_utilities, horizon, cost_weight, initial_observations,
        inference=inference,
    )
    # Validate that the starting state has positive probability under the
    # prior before search (also covers endpoint priors and certain CPDs).
    if inference == "joint":
        network.posterior(prior, observations)
    else:
        network.independence_posterior(prior, observations)

    max_actions = min(horizon, len(actions))
    state_count = 0
    value_cache: dict[tuple[tuple[tuple[str, bool], ...], int], float] = {}
    policy: dict[tuple[tuple[str, bool], ...], str | None] = {}

    def signature(state: Mapping[str, bool]) -> tuple[tuple[str, bool], ...]:
        return tuple(sorted(state.items()))

    def posterior(state: Mapping[str, bool]) -> float:
        if inference == "joint":
            return network.posterior(prior, state)
        return network.independence_posterior(prior, state)

    def terminal_value(state: Mapping[str, bool]) -> float:
        p_true = posterior(state)
        return max(
            p_true * values[True] + (1.0 - p_true) * values[False]
            for values in utilities.values()
        )

    def predictive(action: CollaborativeAction, state: Mapping[str, bool]) -> dict[bool, float]:
        if inference == "joint":
            return network.predictive_distribution(action.evidence_node, prior, state)
        return network.independence_predictive_distribution(
            action.evidence_node, prior, state
        )

    def eligible(action: CollaborativeAction, state: Mapping[str, bool]) -> bool:
        if action.evidence_node in state:
            return False
        return all(by_id[required].evidence_node in state for required in action.requires)

    def action_value(action: CollaborativeAction, state: dict[str, bool], remaining: int) -> float:
        value = -float(cost_weight) * float(action.cost)
        outcome_probs = predictive(action, state)
        for outcome in (False, True):
            probability = outcome_probs[outcome]
            if probability <= 0.0:
                continue
            child = dict(state)
            child[action.evidence_node] = outcome
            value += probability * best_value(child, remaining - 1)
        return value

    def best_value(state: dict[str, bool], remaining: int) -> float:
        nonlocal state_count
        state_key = (signature(state), remaining)
        if state_key in value_cache:
            return value_cache[state_key]
        state_count += 1
        if state_count > MAX_COLLABORATIVE_STATES:
            raise ValueError("collaborative planning problem exceeds exact state limit")
        stop_value = terminal_value(state)
        best = stop_value
        best_action: str | None = None
        if remaining > 0:
            for action in sorted(actions, key=lambda candidate: candidate.action_id):
                if not eligible(action, state):
                    continue
                candidate = action_value(action, state, remaining)
                tolerance = _TIE_TOLERANCE * max(abs(best), abs(candidate), 1e-300)
                if candidate > best + tolerance:
                    best, best_action = candidate, action.action_id
        policy[signature(state)] = best_action
        value_cache[state_key] = best
        return best

    stop_now = terminal_value(observations)
    total_value = best_value(observations, max_actions)
    root_values: dict[str, float] = {}
    if max_actions:
        for action in sorted(actions, key=lambda candidate: candidate.action_id):
            if eligible(action, observations):
                root_values[action.action_id] = action_value(action, observations, max_actions)
    first_action = policy[signature(observations)]
    return CollaborativePlan(
        first_action=first_action,
        value=total_value,
        value_if_stopping_now=stop_now,
        action_values=MappingProxyType(root_values),
        horizon=horizon,
        inference=inference,
        evaluated_states=state_count,
        policy=MappingProxyType(policy),
    )


def _validated_problem(
    prior: float,
    network: ProvenanceNetwork | ProvenanceModelSet,
    actions: Sequence[CollaborativeAction],
    decision_utilities: Mapping[str, Mapping[bool, float]],
    horizon: int,
    cost_weight: float,
    initial_observations: Mapping[str, bool] | None,
    *,
    inference: str | None = None,
) -> tuple[float, dict[str, dict[bool, float]], dict[str, CollaborativeAction], dict[str, bool]]:
    """Check a planning problem; shared by the Bayes and robust planners.

    ``inference`` is checked only when given, at the point where the Bayes
    planner always checked it, so the first error reported is unchanged.
    """
    prior = _probability(prior, "prior")
    if not isinstance(actions, Sequence) or not actions:
        raise ValueError("at least one action is required")
    if any(not isinstance(action, CollaborativeAction) for action in actions):
        raise TypeError("actions must contain CollaborativeAction objects")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("action IDs must be unique")
    if len({action.evidence_node for action in actions}) != len(actions):
        raise ValueError("each evidence node may appear in only one action")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be an integer >= 0")
    if (
        isinstance(cost_weight, bool)
        or not isinstance(cost_weight, (int, float))
        or not isfinite(cost_weight)
        or cost_weight < 0.0
    ):
        raise ValueError("cost_weight must be finite and >= 0")
    if inference is not None and inference not in ("joint", "independent"):
        raise ValueError("inference must be 'joint' or 'independent'")
    if not isinstance(decision_utilities, Mapping) or not decision_utilities:
        raise ValueError("decision_utilities must contain at least one decision")
    utilities: dict[str, dict[bool, float]] = {}
    for decision, values in decision_utilities.items():
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("decision names must be nonempty strings")
        if not isinstance(values, Mapping) or len(values) != 2 or any(
            type(truth) is not bool for truth in values
        ) or set(values) != {True, False}:
            raise ValueError("each decision must define utility for True and False")
        utilities[decision] = {}
        for truth in (True, False):
            utility = values[truth]
            if (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or not isfinite(utility)
            ):
                raise ValueError("decision utilities must be finite numbers")
            utilities[decision][truth] = float(utility)

    by_id = {action.action_id: action for action in actions}
    for action in actions:
        if action.evidence_node not in network.node_ids:
            raise ValueError(f"unknown evidence node: {action.evidence_node}")
        unknown = sorted(set(action.requires) - set(by_id))
        if unknown:
            raise ValueError(f"unknown prerequisite action IDs: {unknown}")
    _check_prerequisite_cycles(by_id)
    if initial_observations is None:
        initial_observations = {}
    observations = network.validate_observations(initial_observations)
    return prior, utilities, by_id, observations


def _check_prerequisite_cycles(actions: Mapping[str, CollaborativeAction]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            raise ValueError("action prerequisites must be acyclic")
        if action_id in visited:
            return
        visiting.add(action_id)
        for required in actions[action_id].requires:
            visit(required)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in actions:
        visit(action_id)
