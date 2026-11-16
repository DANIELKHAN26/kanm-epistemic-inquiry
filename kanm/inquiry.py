"""Auditable question histories and finite-state inquiry planning.

This module is an optional research extension. It does not alter the frozen
single-proposition Laputa-style simulation in :mod:`kanm.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log
import random
from types import MappingProxyType
from typing import Mapping, Sequence


class QuestionRelation(str, Enum):
    ROOT = "root"
    CLARIFICATION = "clarification"
    DECOMPOSITION = "decomposition"
    REFINEMENT = "refinement"
    ALTERNATIVE = "alternative"
    SUBSTITUTION = "substitution"


@dataclass(frozen=True)
class QuestionNode:
    question_id: str
    text: str
    parent_id: str | None
    relation: QuestionRelation


@dataclass(frozen=True)
class QuestionTransition:
    source_id: str
    target_id: str | None
    relation: QuestionRelation | None
    rationale: str
    mapping: str


class QuestionGraph:
    """A question tree with an immutable root and explicit transition history."""

    def __init__(self, goal_id: str, goal_text: str) -> None:
        self._validate_text(goal_id, "goal_id")
        self._validate_text(goal_text, "goal_text")
        self.goal_id = goal_id
        self.nodes: dict[str, QuestionNode] = {
            goal_id: QuestionNode(goal_id, goal_text, None, QuestionRelation.ROOT)
        }
        self.status: dict[str, str] = {goal_id: "active"}
        self.active_id: str | None = goal_id
        self._last_active_id: str | None = goal_id
        self.transitions: list[QuestionTransition] = []

    @staticmethod
    def _validate_text(value: str, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")

    def add_child(
        self,
        *,
        parent_id: str,
        question_id: str,
        text: str,
        relation: QuestionRelation,
        rationale: str,
        mapping: str,
        activate: bool = True,
    ) -> QuestionNode:
        """Add and activate a question while retaining its parent and goal.

        ``mapping`` records how work on the child bears on its parent. It is
        required even for substitution so downstream analyses can audit scope
        drift rather than infer an unstated equivalence.
        """
        if not isinstance(activate, bool):
            raise ValueError("activate must be a boolean")
        if not isinstance(parent_id, str) or parent_id not in self.nodes:
            raise ValueError(f"unknown parent question: {parent_id}")
        if self.active_id is None and self.status.get(parent_id) != "abandoned" and activate:
            raise ValueError("after abandonment, resume from an abandoned question")
        self._validate_text(question_id, "question_id")
        self._validate_text(text, "text")
        self._validate_text(rationale, "rationale")
        self._validate_text(mapping, "mapping")
        if question_id in self.nodes:
            raise ValueError(f"duplicate question ID: {question_id}")
        if not isinstance(relation, QuestionRelation) or relation is QuestionRelation.ROOT:
            raise ValueError("child relation must be a non-root QuestionRelation")
        if activate and self.active_id is not None and parent_id != self.active_id:
            raise ValueError("activate a child only from the active question")
        node = QuestionNode(question_id, text, parent_id, relation)
        self.nodes[question_id] = node
        self.status[question_id] = "active" if activate else "open"
        if activate and self.status[parent_id] != "abandoned":
            # An abandoned question stays abandoned (terminal) when work resumes
            # from it on a new branch.
            self.status[parent_id] = "superseded"
        self.transitions.append(
            QuestionTransition(parent_id, question_id, relation, rationale.strip(), mapping.strip())
        )
        if activate:
            self.active_id = question_id
            self._last_active_id = parent_id
        return node

    def activate_question(self, question_id: str, *, rationale: str, mapping: str) -> None:
        """Select an existing branch explicitly, preserving the switch reason."""
        if not isinstance(question_id, str) or question_id not in self.nodes:
            raise ValueError(f"unknown question: {question_id}")
        self._validate_text(rationale, "rationale")
        self._validate_text(mapping, "mapping")
        if self.status[question_id] == "abandoned":
            raise ValueError("an abandoned question cannot be activated")
        previous = self.active_id
        if previous == question_id:
            return
        if previous is not None:
            self.status[previous] = "open"
        self.status[question_id] = "active"
        self.transitions.append(
            QuestionTransition(
                previous or self._last_active_id or self.goal_id,
                question_id,
                None,
                rationale.strip(),
                mapping.strip(),
            )
        )
        self.active_id = question_id
        self._last_active_id = previous or self._last_active_id

    def abandon(self, question_id: str, *, rationale: str) -> None:
        if not isinstance(question_id, str) or question_id not in self.nodes:
            raise ValueError(f"unknown question: {question_id}")
        if question_id != self.active_id:
            raise ValueError("only the active question can be abandoned")
        self._validate_text(rationale, "rationale")
        self.status[question_id] = "abandoned"
        self.transitions.append(
            QuestionTransition(question_id, None, None, rationale.strip(), "")
        )
        self._last_active_id = question_id
        self.active_id = None


@dataclass(frozen=True)
class QueryAction:
    """One candidate question/helper action as a likelihood table.

    ``likelihoods[h][j]`` is P(outcome j | hypothesis h, action). This makes
    helper competence explicit and allows different helpers or methods to
    supply different likelihood tables for the same question.
    """

    action_id: str
    question_id: str
    helper_id: str
    outcomes: tuple[str, ...]
    likelihoods: Mapping[str, tuple[float, ...]]
    cost: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("action_id", self.action_id),
            ("question_id", self.question_id),
            ("helper_id", self.helper_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.outcomes, tuple) or not self.outcomes or any(
            not isinstance(x, str) or not x.strip() for x in self.outcomes
        ):
            raise ValueError("outcomes must be nonempty strings")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be unique")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise ValueError("cost must be finite and >= 0")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("cost must be finite and >= 0")
        if not isinstance(self.likelihoods, Mapping) or not self.likelihoods:
            raise ValueError("likelihoods must include at least one hypothesis")
        if len(self.likelihoods) != len(set(self.likelihoods)):
            raise ValueError("hypothesis names must be unique")
        normalized_rows: dict[str, tuple[float, ...]] = {}
        for hypothesis, row in self.likelihoods.items():
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError("hypothesis names must be nonempty strings")
            if not isinstance(row, tuple):
                raise ValueError(f"likelihood row for {hypothesis} must be a tuple")
            if len(row) != len(self.outcomes):
                raise ValueError(f"likelihood row for {hypothesis} has wrong length")
            if any(
                isinstance(p, bool)
                or not isinstance(p, (int, float))
                or not isfinite(p)
                or p < 0.0
                or p > 1.0
                for p in row
            ):
                raise ValueError(f"likelihoods for {hypothesis} must lie in [0, 1]")
            if abs(sum(row) - 1.0) > 1e-9:
                raise ValueError(f"likelihoods for {hypothesis} must sum to 1")
            row_total = sum(row)
            normalized_rows[hypothesis] = tuple(float(p) / row_total for p in row)
        object.__setattr__(self, "likelihoods", MappingProxyType(normalized_rows))


def _validated_prior(prior: Mapping[str, float], action: QueryAction) -> dict[str, float]:
    if set(prior) != set(action.likelihoods):
        raise ValueError("prior hypotheses must match the action likelihood table")
    if not prior:
        raise ValueError("prior must contain at least one hypothesis")
    values: dict[str, float] = {}
    for hypothesis, probability in prior.items():
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not isfinite(probability)
            or probability < 0.0
        ):
            raise ValueError("prior probabilities must be finite and nonnegative")
        values[hypothesis] = float(probability)
    if abs(sum(values.values()) - 1.0) > 1e-9:
        raise ValueError("prior probabilities must sum to 1")
    return values


def _entropy(distribution: Sequence[float]) -> float:
    return -sum(p * log(p) for p in distribution if p > 0.0)


def _validated_partition(
    partition: Mapping[str, str] | None, hypotheses: Sequence[str]
) -> dict[str, str]:
    """Map every hypothesis to an answer cell; ``None`` is the finest partition."""
    if partition is None:
        return {h: h for h in hypotheses}
    if not isinstance(partition, Mapping) or set(partition) != set(hypotheses):
        raise ValueError("a question partition must assign every hypothesis to one answer")
    for hypothesis, cell in partition.items():
        if not isinstance(cell, str) or not cell.strip():
            raise ValueError(f"answer label for {hypothesis} must be a nonempty string")
    return dict(partition)


def _cell_distribution(distribution: Mapping[str, float], partition: Mapping[str, str]) -> list[float]:
    cells: dict[str, float] = {}
    for hypothesis, probability in distribution.items():
        cells[partition[hypothesis]] = cells.get(partition[hypothesis], 0.0) + probability
    return [cells[c] for c in sorted(cells)]


def expected_information_gain(
    prior: Mapping[str, float],
    action: QueryAction,
    *,
    target: Mapping[str, str] | None = None,
) -> float:
    """Return mutual information I(T; outcome | action), in nats.

    ``target`` is a question represented as a partition of the hypothesis
    space (hypothesis -> answer label), following the partition semantics of
    questions. By default every hypothesis is its own answer, giving
    I(H; outcome). With a coarser target the result is the information the
    action carries about *that* question only.
    """
    probabilities = _validated_prior(prior, action)
    partition = _validated_partition(target, list(probabilities))
    predictive = [
        sum(probabilities[h] * action.likelihoods[h][j] for h in probabilities)
        for j in range(len(action.outcomes))
    ]
    expected_posterior_entropy = 0.0
    for outcome_index, outcome_probability in enumerate(predictive):
        if outcome_probability == 0.0:
            continue
        posterior = {
            h: probabilities[h] * action.likelihoods[h][outcome_index] / outcome_probability
            for h in probabilities
        }
        expected_posterior_entropy += outcome_probability * _entropy(
            _cell_distribution(posterior, partition)
        )
    prior_entropy = _entropy(_cell_distribution(probabilities, partition))
    return max(0.0, prior_entropy - expected_posterior_entropy)


def question_relevance(
    prior: Mapping[str, float],
    action: QueryAction,
    goal: Mapping[str, str],
    active: Mapping[str, str] | None = None,
) -> dict[str, float | None]:
    """How much an action informs the original goal versus the active question.

    Returns the information about the goal question, about the active
    (possibly substituted) question, and their ratio. If the active question
    refines the goal (every active answer lies inside one goal answer), the
    data-processing inequality gives ratio <= 1. A low ratio flags scope
    drift: the action answers the working question but says little about the
    goal. For non-nested questions no such bound holds.
    """
    about_goal = expected_information_gain(prior, action, target=goal)
    about_active = expected_information_gain(prior, action, target=active)
    ratio = about_goal / about_active if about_active > 0.0 else None
    return {
        "information_about_goal": about_goal,
        "information_about_active": about_active,
        "goal_relevance_ratio": ratio,
    }


def refines(fine: Mapping[str, str], coarse: Mapping[str, str]) -> bool:
    """True when each answer of ``fine`` lies within a single answer of ``coarse``."""
    if set(fine) != set(coarse):
        raise ValueError("partitions must cover the same hypotheses")
    image: dict[str, str] = {}
    for hypothesis, cell in fine.items():
        if image.setdefault(cell, coarse[hypothesis]) != coarse[hypothesis]:
            return False
    return True


def expected_value_of_sample_information(
    prior: Mapping[str, float],
    action: QueryAction,
    decision_utilities: Mapping[str, Mapping[str, float]],
) -> float:
    """Return expected decision value gained by sampling, in utility units.

    This criterion is goal-specific: callers provide the terminal decisions
    and utility under each hypothesis. It differs from entropy reduction, which
    values information even when it cannot change a decision.
    """
    probabilities = _validated_prior(prior, action)
    if not isinstance(decision_utilities, Mapping) or not decision_utilities:
        raise ValueError("decision_utilities must contain at least one decision")
    utilities: dict[str, dict[str, float]] = {}
    for decision, by_hypothesis in decision_utilities.items():
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("decision names must be nonempty strings")
        if not isinstance(by_hypothesis, Mapping) or set(by_hypothesis) != set(probabilities):
            raise ValueError("each decision must define utility for every hypothesis")
        utilities[decision] = {}
        for hypothesis, utility in by_hypothesis.items():
            if (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or not isfinite(utility)
            ):
                raise ValueError("decision utilities must be finite numbers")
            utilities[decision][hypothesis] = float(utility)

    def best_expected_utility(distribution: Mapping[str, float]) -> float:
        return max(
            sum(distribution[h] * utilities[d][h] for h in distribution)
            for d in utilities
        )

    prior_best = best_expected_utility(probabilities)
    expected_posterior_best = 0.0
    for outcome_index in range(len(action.outcomes)):
        outcome_probability = sum(
            probabilities[h] * action.likelihoods[h][outcome_index]
            for h in probabilities
        )
        if outcome_probability == 0.0:
            continue
        posterior = {
            h: probabilities[h] * action.likelihoods[h][outcome_index] / outcome_probability
            for h in probabilities
        }
        expected_posterior_best += (
            outcome_probability * best_expected_utility(posterior)
        )
    return max(0.0, expected_posterior_best - prior_best)


def update_hypothesis_posterior(
    prior: Mapping[str, float], action: QueryAction, outcome: str
) -> dict[str, float]:
    """Bayesian posterior over the finite hypothesis space after an outcome."""
    probabilities = _validated_prior(prior, action)
    try:
        outcome_index = action.outcomes.index(outcome)
    except ValueError as exc:
        raise ValueError(f"unknown outcome {outcome!r}") from exc
    unnormalized = {
        h: probabilities[h] * action.likelihoods[h][outcome_index]
        for h in probabilities
    }
    normalizer = sum(unnormalized.values())
    if normalizer <= 0.0:
        raise ValueError("observed outcome has zero probability under the prior model")
    return {h: p / normalizer for h, p in unnormalized.items()}


def sample_outcome(hypothesis: str, action: QueryAction, rng: random.Random) -> str:
    """Sample an action result from a stated ground-truth hypothesis."""
    if hypothesis not in action.likelihoods:
        raise ValueError(f"unknown ground-truth hypothesis: {hypothesis}")
    draw = rng.random()
    cumulative = 0.0
    for outcome, probability in zip(action.outcomes, action.likelihoods[hypothesis]):
        cumulative += probability
        if draw < cumulative:
            return outcome
    return action.outcomes[-1]  # absorb only floating-point summation residue


def select_action(
    prior: Mapping[str, float],
    actions: Sequence[QueryAction],
    *,
    cost_weight: float = 0.0,
) -> tuple[QueryAction, dict[str, float]]:
    """Choose the highest ``information gain - cost_weight * cost`` action.

    The objective is an explicit policy, not a universal definition of
    scientific or social value. Ties resolve by lexical action ID so runs are
    reproducible regardless of input ordering.
    """
    if not actions:
        raise ValueError("at least one candidate action is required")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("candidate action IDs must be unique")
    if (
        isinstance(cost_weight, bool)
        or not isinstance(cost_weight, (int, float))
        or not isfinite(cost_weight)
        or cost_weight < 0.0
    ):
        raise ValueError("cost_weight must be finite and >= 0")
    scores = {
        action.action_id: expected_information_gain(prior, action) - cost_weight * action.cost
        for action in actions
    }
    return _best_with_ties(actions, scores), scores


def select_action_by_value(
    prior: Mapping[str, float],
    actions: Sequence[QueryAction],
    decision_utilities: Mapping[str, Mapping[str, float]],
    *,
    cost_weight: float = 1.0,
) -> tuple[QueryAction, dict[str, float]]:
    """Rank actions by expected value of sample information minus cost.

    ``cost_weight`` converts action-cost units to the units of the caller's
    decision utility. State that conversion explicitly in any experiment.
    """
    if not actions:
        raise ValueError("at least one candidate action is required")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("candidate action IDs must be unique")
    if (
        isinstance(cost_weight, bool)
        or not isinstance(cost_weight, (int, float))
        or not isfinite(cost_weight)
        or cost_weight < 0.0
    ):
        raise ValueError("cost_weight must be finite and >= 0")
    scores = {
        action.action_id: expected_value_of_sample_information(
            prior, action, decision_utilities
        ) - cost_weight * action.cost
        for action in actions
    }
    return _best_with_ties(actions, scores), scores


#: Relative tolerance under which two scores count as tied (rounding noise).
TIE_TOLERANCE = 1e-12


def _best_with_ties(actions: Sequence[QueryAction], scores: Mapping[str, float]) -> QueryAction:
    """Highest score; scores equal up to rounding go to the smallest action ID.

    Without the tolerance, mathematically equal scores computed in different
    summation orders would be split by floating-point noise rather than by
    the documented lexical tie-break.
    """
    best_score = max(scores.values())
    tied = [
        action
        for action in actions
        if scores[action.action_id]
        >= best_score - TIE_TOLERANCE * max(abs(best_score), abs(scores[action.action_id]), 1e-300)
    ]
    return min(tied, key=lambda action: action.action_id)


def select_action_or_stop(
    prior: Mapping[str, float],
    actions: Sequence[QueryAction],
    decision_utilities: Mapping[str, Mapping[str, float]],
    *,
    cost_weight: float = 1.0,
) -> tuple[QueryAction | None, dict[str, float]]:
    """Myopic value-of-information rule with an explicit stop option.

    Returns ``None`` (stop and decide now) unless some action has strictly
    positive ``EVSI - cost_weight * cost``. This one-step rule can stop too
    early, because the value of information is not concave: several cheap
    observations may be worth buying even when none is worth buying alone.
    :func:`optimal_sequential_plan` gives the exact finite-horizon answer.
    """
    best, scores = select_action_by_value(
        prior, actions, decision_utilities, cost_weight=cost_weight
    )
    return (best if scores[best.action_id] > 0.0 else None), scores


@dataclass(frozen=True)
class SequentialPlan:
    """Exact finite-horizon solution at the current belief state."""

    first_action: str | None
    value: float
    value_if_stopping_now: float
    action_values: Mapping[str, float]
    horizon: int

    @property
    def value_of_planning(self) -> float:
        """Expected gain from optimal inquiry over deciding immediately."""
        return self.value - self.value_if_stopping_now


MAX_PLAN_NODES = 2_000_000


def optimal_sequential_plan(
    prior: Mapping[str, float],
    actions: Sequence[QueryAction],
    decision_utilities: Mapping[str, Mapping[str, float]],
    *,
    horizon: int,
    cost_weight: float = 1.0,
) -> SequentialPlan:
    """Bayes-optimal sequential inquiry by backward induction.

    The agent may perform up to ``horizon`` actions (repeats allowed), each
    observed before choosing the next, and then takes the terminal decision
    with the highest posterior expected utility. Repeated actions are assumed
    conditionally independent given the hypothesis. Each action costs
    ``cost_weight * cost`` in utility units. Ties prefer stopping, then the
    lexically smallest action ID. The search is exponential in the horizon and
    refuses problems larger than :data:`MAX_PLAN_NODES` evaluations.
    """
    if not actions:
        raise ValueError("at least one candidate action is required")
    if len({a.action_id for a in actions}) != len(actions):
        raise ValueError("candidate action IDs must be unique")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be an integer >= 0")
    if (
        isinstance(cost_weight, bool)
        or not isinstance(cost_weight, (int, float))
        or not isfinite(cost_weight)
        or cost_weight < 0.0
    ):
        raise ValueError("cost_weight must be finite and >= 0")
    branching = sum(len(a.outcomes) for a in actions)
    nodes, level = 0, 1
    for _ in range(horizon + 1):
        nodes += level
        if nodes > MAX_PLAN_NODES:
            raise ValueError("planning problem too large for exact backward induction")
        level *= branching
    probabilities = _validated_prior(prior, actions[0])
    for action in actions[1:]:
        _validated_prior(probabilities, action)
    # Reuse the EVSI validator for the utility table.
    expected_value_of_sample_information(probabilities, actions[0], decision_utilities)
    hypotheses = sorted(probabilities)
    utilities = {d: {h: float(decision_utilities[d][h]) for h in hypotheses}
                 for d in decision_utilities}

    def terminal(belief: Mapping[str, float]) -> float:
        return max(sum(belief[h] * utilities[d][h] for h in hypotheses) for d in utilities)

    def action_value(belief: dict[str, float], action: QueryAction, remaining: int) -> float:
        value = -cost_weight * action.cost
        for j in range(len(action.outcomes)):
            joint = {h: belief[h] * action.likelihoods[h][j] for h in hypotheses}
            mass = sum(joint.values())
            if mass <= 0.0:
                continue
            posterior = {h: joint[h] / mass for h in hypotheses}
            value += mass * best_value(posterior, remaining - 1)
        return value

    def best_value(belief: dict[str, float], remaining: int) -> float:
        stop = terminal(belief)
        if remaining == 0:
            return stop
        return max([stop] + [action_value(belief, a, remaining) for a in actions])

    stop_now = terminal(probabilities)
    values = {a.action_id: action_value(probabilities, a, horizon) for a in actions} if horizon else {}
    first: str | None = None
    best = stop_now
    for action_id in sorted(values):
        # Relative tolerance: ties (up to rounding) prefer stopping, at any utility scale.
        tolerance = 1e-12 * max(abs(best), abs(values[action_id]))
        if values[action_id] > best + tolerance:
            best, first = values[action_id], action_id
    return SequentialPlan(first, best, stop_now, MappingProxyType(values), horizon)
