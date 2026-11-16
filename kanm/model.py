"""A deterministic Laputa-style network simulation.

The single-report update kernel follows the published binary model. Two
within-step schedules and two update rules are available; every combination is
recorded in the result so that runs are never ambiguous about which model was
used.

Schedules
---------
``"simultaneous"``
    Every inquiry result and every assertion at step ``t`` is determined by the
    credences held at the start of the step, ``C_t``. Each agent then
    conditions on all reports it received during the step. Under the Laputa
    rule each source's trust density is updated with ``C_t``. Results do not
    depend on the order in which agents or edges are processed.
``"inquiry_then_messages"``
    The v0.2.0 schedule: agents inquire and update first (in lexical ID
    order); messages are then generated from post-inquiry credences and
    delivered one at a time in stable ``(receiver, sender)`` order, each
    update using the receiver's current credence.

``"laputa_1_6"``
    A re-statement of ``Society::evolve`` in the Laputa 1.6 source, including
    its new-evidence policies (``evidence_policy`` = ``"none"``, ``"inquiry"``
    or ``"any"``; ``count_prior_as_evidence``). Requires the Laputa update
    rule and ``assertion_rule="laputa_1_6"``. With ``evidence_policy="none"``
    it gives exactly the ``simultaneous`` result; under the other policies the
    outcome can depend on agent order (sorted IDs play the role of Laputa's
    inquirer index), as in the original.

The simultaneous schedule matches the order of operations in the Laputa 1.6
source (``Society::evolve``: all inquirers compute new credences from the old
ones, then all are replaced), as read from the archived code. A harness that
compiles the 1.6 model core for numerical docking is in ``docking/laputa_1_6``;
until its comparison has been run, no schedule counts as docked.

Trust models
------------
``"beta_grid"`` (default)
    Moment-preserving discretisation of a Beta prior on ``trust_points``
    cells, held in log space (:class:`kanm.bayes.TrustDistribution`).
``"laputa_1_6"``
    The 49-node piecewise-linear trust function of the 1.6 source
    (:class:`kanm.laputa16.Laputa16Trust`); specify node values explicitly
    (``inquiry_trust_values``/``trust_values``) or a Beta mean and
    concentration with both shape parameters >= 1.

Assertion rules
---------------
``"published"``
    Olsson (2013, section 3): with threshold ``T > 0.5`` say ``p`` only if
    ``C >= T`` and ``not p`` only if ``C <= 1 - T``; with ``T < 0.5`` the
    inverted "liar" rule; with ``T = 0.5`` choose at random.
``"laputa_1_6"``
    Behaviour of the Laputa 1.6 source under its default "no new-evidence
    requirement" policy: with ``T > 0.5`` say ``p`` whenever ``C > 0.5`` and
    ``not p`` whenever ``C < 0.5`` (the size of ``T`` is not consulted), the
    reverse for ``T < 0.5``, and a random choice when ``C = 0.5`` or
    ``T = 0.5``. Provided for docking; it differs from the published rule.

Update rules
------------
``"laputa"``
    The published rule: credence uses the expected reliability; each trust
    density is updated separately (product of marginals).
``"exact_joint"``
    The exact joint posterior over ``p`` and all source reliabilities under the
    same assumptions (:class:`kanm.bayes.ExactJointState`). It is an exact
    reference for the agent's *assumed* model, not for the true data-generating
    process of the network, in which reports can share upstream evidence.

Randomness
----------
Each agent's inquiry process and each edge's communication process draws from
its own pseudo-random stream derived from ``seed``, and every stream draws the
same number of variates each step whatever the agents believe. Different
update rules or schedules run with the same seed therefore see identical
inquiry outcomes and listening opportunities (common random numbers), which
makes paired comparisons much more precise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import isfinite, sqrt
import platform
import random
from typing import Any

from . import __version__
from .bayes import (
    ExactJointState,
    TrustDistribution,
    _log_sigmoid,
    _probability,
    expit,
    log_likelihood_ratio,
    logit,
)
from .laputa16 import NODES as LAPUTA16_NODES, Laputa16Trust, validated_node_values

UPDATE_RULES = ("laputa", "exact_joint")
SCHEDULES = ("simultaneous", "inquiry_then_messages", "laputa_1_6")
DEFAULT_SCHEDULE = "simultaneous"
DEFAULT_UPDATE_RULE = "laputa"
ASSERTION_RULES = ("published", "laputa_1_6")
DEFAULT_ASSERTION_RULE = "published"
TRUST_MODELS = ("beta_grid", "laputa_1_6")
DEFAULT_TRUST_MODEL = "beta_grid"
DEFAULT_TRUST_POINTS = 201
EVIDENCE_POLICIES = ("none", "inquiry", "any")
MAX_STEPS = 10_000_000


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    prior_belief: float = 0.5
    inquiry_accuracy: float = 0.75
    inquiry_chance: float = 0.25
    inquiry_trust_mean: float = 0.6
    inquiry_trust_concentration: float = 8.0
    #: Explicit 49 node values; only with ``trust_model="laputa_1_6"``.
    inquiry_trust_values: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EdgeSpec:
    sender: str
    receiver: str
    listen_chance: float = 1.0
    assertion_threshold: float = 0.75
    trust_mean: float = 0.6
    trust_concentration: float = 8.0
    #: Explicit 49 node values; only with ``trust_model="laputa_1_6"``.
    trust_values: tuple[float, ...] | None = None


_INQUIRY = ("inquiry",)


@dataclass
class _Agent:
    spec: AgentSpec
    log_odds: float
    initial_belief: float
    rng: random.Random
    inquiry_trust: TrustDistribution | Laputa16Trust | None = None
    joint: ExactJointState | None = None
    received_evidence: int = 0
    inquired_this_round: bool = False
    inquired_last_round: bool = False
    # Laputa 1.6 schedule state (Inquirer::lastInquiryResult, Inquirer::message).
    last_inquiry_step: int = -1
    last_inquiry_result: bool | None = None

    @property
    def belief(self) -> float:
        # Report the configured prior exactly until evidence arrives, so the
        # logit/expit round trip never perturbs an unchanged credence.
        if self.received_evidence == 0:
            return self.initial_belief
        return expit(self.log_odds)


@dataclass
class _Edge:
    spec: EdgeSpec
    rng: random.Random
    trust: TrustDistribution | Laputa16Trust | None = None
    # Laputa 1.6 schedule state (Link::lastUsed, Link::message).
    last_used_step: int = -1
    message: bool | None = None


@dataclass
class _Report:
    receiver: str
    source_key: tuple
    positive: bool
    kind: str
    sender: str | None = None


@dataclass
class SimulationResult:
    truth: bool
    steps: int
    seed: int
    initial_mean_belief: float
    final_mean_belief: float
    final_beliefs: dict[str, float]
    mean_brier_score: float
    mean_log_score: float | None
    accuracy_at_half: float
    undecided_fraction: float
    veritistic_value: float
    belief_dispersion: float
    belief_spread: float
    evidence_events: int
    message_events: int
    publish_only_after_inquiry: bool
    trust_points: int
    update_rule: str
    schedule: str
    assertion_rule: str
    configuration: dict[str, Any]
    final_trust: dict[str, float] = field(default_factory=dict)
    truth_credence_trajectory: list[float] | None = None
    event_log: list[dict[str, Any]] = field(default_factory=list)
    trust_model: str = DEFAULT_TRUST_MODEL
    evidence_policy: str = "none"
    count_prior_as_evidence: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = {
            "truth": self.truth,
            "steps": self.steps,
            "seed": self.seed,
            "initial_mean_belief": self.initial_mean_belief,
            "final_mean_belief": self.final_mean_belief,
            "final_beliefs": dict(self.final_beliefs),
            "final_trust": dict(self.final_trust),
            "mean_brier_score": self.mean_brier_score,
            "mean_log_score": self.mean_log_score,
            "accuracy_at_half": self.accuracy_at_half,
            "undecided_fraction": self.undecided_fraction,
            "veritistic_value": self.veritistic_value,
            "belief_dispersion": self.belief_dispersion,
            "belief_spread": self.belief_spread,
            "evidence_events": self.evidence_events,
            "message_events": self.message_events,
            "configuration": self.configuration,
            "model_variant": model_variant(self.update_rule, self.schedule),
            "kanm_version": __version__,
            "runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "random_generator": "random.Random (MT19937), one stream per agent and edge",
            },
            "assumptions": {
                "binary_proposition": True,
                "source_symmetry": True,
                "conditional_source_independence": True,
                "update_rule": self.update_rule,
                "trust_and_credence_factorized": self.update_rule == "laputa",
                "repeated_opinions_are_fresh": not self.publish_only_after_inquiry,
                "publish_only_after_inquiry": self.publish_only_after_inquiry,
                "trust_grid_points": self.trust_points,
                "trust_model": self.trust_model,
                "schedule": self.schedule,
                "assertion_rule": self.assertion_rule,
                "evidence_policy": self.evidence_policy,
                "count_prior_as_evidence": self.count_prior_as_evidence,
            },
            "event_log": list(self.event_log),
        }
        if self.truth_credence_trajectory is not None:
            result["truth_credence_trajectory"] = list(self.truth_credence_trajectory)
        return result


def model_variant(update_rule: str, schedule: str) -> str:
    return f"kanm-binary-{update_rule}-{schedule}-{__version__}"


def _positive_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{label} must be finite and > 0")
    return float(value)


def _open_probability(value: Any, label: str) -> float:
    value = _probability(value, label)
    if value in (0.0, 1.0):
        raise ValueError(
            f"{label} must lie strictly between 0 and 1; certain reliability makes "
            "contrary reports impossible (use a high concentration instead)"
        )
    return value


def _stream(seed: int, *parts: str) -> random.Random:
    # JSON encoding gives an unambiguous key for arbitrary agent IDs; string
    # seeding of random.Random is hashed with SHA-512 and is platform-stable.
    return random.Random(json.dumps([seed, *parts]))


class Simulation:
    """Simulate inquiry and testimony about one binary proposition."""

    def __init__(
        self,
        agents: list[AgentSpec],
        edges: list[EdgeSpec],
        *,
        truth: bool = True,
        steps: int = 100,
        seed: int = 0,
        trust_points: int | None = None,
        publish_only_after_inquiry: bool = False,
        update_rule: str = DEFAULT_UPDATE_RULE,
        schedule: str = DEFAULT_SCHEDULE,
        assertion_rule: str = DEFAULT_ASSERTION_RULE,
        trust_model: str = DEFAULT_TRUST_MODEL,
        evidence_policy: str = "none",
        count_prior_as_evidence: bool = False,
        record_trajectory: bool = False,
        record_events: bool = True,
    ) -> None:
        if not isinstance(agents, (list, tuple)) or not agents:
            raise ValueError("at least one agent is required")
        if any(not isinstance(agent, AgentSpec) for agent in agents):
            raise TypeError("agents must be AgentSpec instances")
        if not isinstance(edges, (list, tuple)) or any(
            not isinstance(edge, EdgeSpec) for edge in edges
        ):
            raise TypeError("edges must be a list of EdgeSpec instances")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 0 <= steps <= MAX_STEPS
        ):
            raise ValueError(f"steps must be an integer in [0, {MAX_STEPS}]")
        if trust_model not in TRUST_MODELS:
            raise ValueError(f"trust_model must be one of {TRUST_MODELS}")
        if trust_points is None:
            trust_points = (
                LAPUTA16_NODES if trust_model == "laputa_1_6" else DEFAULT_TRUST_POINTS
            )
        if isinstance(trust_points, bool) or not isinstance(trust_points, int) or trust_points < 3:
            raise ValueError("trust_points must be an integer >= 3")
        if trust_model == "laputa_1_6" and trust_points != LAPUTA16_NODES:
            raise ValueError(
                f"trust_model 'laputa_1_6' has exactly {LAPUTA16_NODES} trust nodes; "
                "omit trust_points"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(truth, bool):
            raise ValueError("truth must be a boolean")
        for flag, label in (
            (publish_only_after_inquiry, "publish_only_after_inquiry"),
            (count_prior_as_evidence, "count_prior_as_evidence"),
            (record_trajectory, "record_trajectory"),
            (record_events, "record_events"),
        ):
            if not isinstance(flag, bool):
                raise ValueError(f"{label} must be a boolean")
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if assertion_rule not in ASSERTION_RULES:
            raise ValueError(f"assertion_rule must be one of {ASSERTION_RULES}")
        if evidence_policy not in EVIDENCE_POLICIES:
            raise ValueError(f"evidence_policy must be one of {EVIDENCE_POLICIES}")
        if trust_model == "laputa_1_6" and update_rule != "laputa":
            raise ValueError("trust_model 'laputa_1_6' is only defined for update_rule 'laputa'")
        if schedule == "laputa_1_6":
            if update_rule != "laputa":
                raise ValueError("schedule 'laputa_1_6' requires update_rule 'laputa'")
            if assertion_rule != "laputa_1_6":
                raise ValueError("schedule 'laputa_1_6' requires assertion_rule 'laputa_1_6'")
            if publish_only_after_inquiry:
                raise ValueError(
                    "schedule 'laputa_1_6' uses evidence_policy instead of "
                    "publish_only_after_inquiry"
                )
        elif evidence_policy != "none" or count_prior_as_evidence:
            raise ValueError(
                "evidence_policy and count_prior_as_evidence apply only to schedule 'laputa_1_6'"
            )
        ids = [agent.agent_id for agent in agents]
        if (
            any(not isinstance(agent_id, str) or not agent_id.strip() for agent_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("agent IDs must be nonempty and unique")
        known = set(ids)
        for agent in agents:
            _probability(agent.prior_belief, f"{agent.agent_id}.prior_belief")
            _probability(agent.inquiry_accuracy, f"{agent.agent_id}.inquiry_accuracy")
            _probability(agent.inquiry_chance, f"{agent.agent_id}.inquiry_chance")
            _open_probability(agent.inquiry_trust_mean, f"{agent.agent_id}.inquiry_trust_mean")
            _positive_finite(
                agent.inquiry_trust_concentration,
                f"{agent.agent_id}.inquiry_trust_concentration",
            )
        for edge in edges:
            if edge.sender not in known or edge.receiver not in known:
                raise ValueError(f"edge references unknown agent: {edge}")
            if edge.sender == edge.receiver:
                raise ValueError("self-edges are not supported in this prototype")
            label = f"edge {edge.sender}->{edge.receiver}"
            _probability(edge.listen_chance, f"{label}.listen_chance")
            _probability(edge.assertion_threshold, f"{label}.assertion_threshold")
            _open_probability(edge.trust_mean, f"{label}.trust_mean")
            _positive_finite(edge.trust_concentration, f"{label}.trust_concentration")
        edge_pairs = [(edge.sender, edge.receiver) for edge in edges]
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("duplicate directed edges are not supported")

        self.truth = truth
        self.steps = steps
        self.seed = seed
        self.publish_only_after_inquiry = publish_only_after_inquiry
        self.trust_points = trust_points
        self.update_rule = update_rule
        self.schedule = schedule
        self.assertion_rule = assertion_rule
        self.trust_model = trust_model
        self.evidence_policy = evidence_policy
        self.count_prior_as_evidence = count_prior_as_evidence
        self.record_trajectory = record_trajectory
        self.record_events = record_events
        self._agent_specs = list(agents)

        exact = update_rule == "exact_joint"
        self.agents: dict[str, _Agent] = {}
        for spec in agents:
            prior = self._trust_prior(
                spec.inquiry_trust_mean,
                spec.inquiry_trust_concentration,
                spec.inquiry_trust_values,
                f"{spec.agent_id}.inquiry_trust_values",
            )
            agent = _Agent(
                spec=spec,
                log_odds=logit(spec.prior_belief),
                initial_belief=spec.prior_belief,
                rng=_stream(seed, "inquiry", spec.agent_id),
            )
            if exact:
                agent.joint = ExactJointState(spec.prior_belief)
                agent.joint.add_source(_INQUIRY, prior)
            else:
                agent.inquiry_trust = prior
            self.agents[spec.agent_id] = agent

        self.edges: list[_Edge] = []
        for spec in sorted(edges, key=lambda edge: (edge.receiver, edge.sender)):
            prior = self._trust_prior(
                spec.trust_mean,
                spec.trust_concentration,
                spec.trust_values,
                f"edge {spec.sender}->{spec.receiver}.trust_values",
            )
            edge = _Edge(spec=spec, rng=_stream(seed, "edge", spec.sender, spec.receiver))
            if exact:
                self.agents[spec.receiver].joint.add_source(("edge", spec.sender), prior)
            else:
                edge.trust = prior
            self.edges.append(edge)
        self._edge_by_key = {(e.spec.receiver, ("edge", e.spec.sender)): e for e in self.edges}
        trust_keys = [f"{e.spec.sender}->{e.spec.receiver}" for e in self.edges] + [
            f"{agent_id}:inquiry" for agent_id in self.agents
        ]
        if len(set(trust_keys)) != len(trust_keys):
            raise ValueError(
                "agent IDs make final_trust keys ambiguous ('<sender>-><receiver>' and "
                "'<agent>:inquiry'); avoid '->' and ':inquiry' in IDs"
            )
        self._incoming: dict[str, list[_Edge]] = {agent_id: [] for agent_id in self.agents}
        for edge in self.edges:  # already sorted by (receiver, sender)
            self._incoming[edge.spec.receiver].append(edge)
        if schedule == "laputa_1_6":
            self._reset_laputa_1_6()
        self.event_log: list[dict[str, Any]] = []
        self.evidence_events = 0
        self.message_events = 0
        self._ran = False

    def _trust_prior(
        self, mean: float, concentration: float, values: Any, label: str
    ) -> TrustDistribution | Laputa16Trust:
        if self.trust_model == "laputa_1_6":
            if values is not None:
                return validated_node_values(values, label)
            return Laputa16Trust.from_beta(mean, concentration)
        if values is not None:
            raise ValueError(f"{label} is only used with trust_model 'laputa_1_6'")
        return TrustDistribution.beta_prior(mean, concentration, self.trust_points)

    # ------------------------------------------------------------------ trust
    def _trust(self, agent: _Agent, key: tuple) -> TrustDistribution:
        if agent.joint is not None:
            return agent.joint.trust(key)
        if key == _INQUIRY:
            return agent.inquiry_trust
        return self._edge_by_key[(agent.spec.agent_id, key)].trust

    def _apply(self, agent: _Agent, key: tuple, positive: bool, basis: float) -> None:
        """Condition ``agent`` on one report.

        ``basis`` is the credence used in the Laputa trust update (``C_t`` in
        the simultaneous schedule, the running credence otherwise). The
        exact-joint rule needs no basis because its updates commute.
        """
        if agent.joint is not None:
            agent.joint.observe(key, positive)
            agent.log_odds = agent.joint.log_odds
            return
        trust = self._trust(agent, key)
        new_log_odds = agent.log_odds + log_likelihood_ratio(trust, positive)
        if new_log_odds != new_log_odds:
            raise ValueError("observation has zero probability under the current model")
        agent.log_odds = new_log_odds
        updated = trust.observe(basis, positive)
        if key == _INQUIRY:
            agent.inquiry_trust = updated
        else:
            self._edge_by_key[(agent.spec.agent_id, key)].trust = updated

    # ------------------------------------------------------------- assertion
    def _assertion(self, log_odds: float, threshold: float, coin: float) -> bool | None:
        """Decide what a sender asserts, from its credence held as log-odds.

        Comparisons are made in log-odds against ``logit(T)``, so mirrored
        configurations (swap ``p`` and ``not p``) give mirrored decisions
        exactly. Comparing rounded probabilities would not: one positive
        report at prior 0.5 and trust 0.75 yields ``C = 0.7499999999999999``
        (no assertion at ``T = 0.75``) but its mirror yields exactly ``0.25``
        (assertion); and ``C`` rounds to 1.0 near log-odds 37 but to 0.0
        only near -745. With ``T = 1`` a sender asserts only when dogmatic.
        """
        if self.assertion_rule == "laputa_1_6":
            if (log_odds > 0.0 and threshold > 0.5) or (log_odds < 0.0 and threshold < 0.5):
                return True
            if (log_odds < 0.0 and threshold > 0.5) or (log_odds > 0.0 and threshold < 0.5):
                return False
            return coin < 0.5
        if threshold == 0.5:
            return coin < 0.5
        cut = logit(max(threshold, 1.0 - threshold))  # > 0, +inf when T in {0, 1}
        towards_p = log_odds >= cut
        towards_not_p = -log_odds >= cut
        if threshold > 0.5:
            return True if towards_p else (False if towards_not_p else None)
        # Liar rule (T < 0.5): say p when C <= T and not-p when C >= 1 - T.
        return True if towards_not_p else (False if towards_p else None)

    # ------------------------------------------------------------------ step
    def _draw_inquiries(self) -> dict[str, bool | None]:
        results: dict[str, bool | None] = {}
        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            chance_draw = agent.rng.random()
            accuracy_draw = agent.rng.random()
            if chance_draw < agent.spec.inquiry_chance:
                correct = accuracy_draw < agent.spec.inquiry_accuracy
                results[agent_id] = self.truth if correct else not self.truth
            else:
                results[agent_id] = None
        return results

    def _sender_may_publish(self, sender: _Agent) -> bool:
        if not self.publish_only_after_inquiry:
            return True
        if self.schedule == "simultaneous":
            # C_t already includes the previous step's inquiry, not this one's.
            return sender.inquired_last_round
        return sender.inquired_this_round

    def _draw_messages(self, log_odds: dict[str, float]) -> list[_Report]:
        reports = []
        for edge in self.edges:
            listen_draw = edge.rng.random()
            coin = edge.rng.random()
            sender = self.agents[edge.spec.sender]
            if not self._sender_may_publish(sender):
                continue
            positive = self._assertion(
                log_odds[edge.spec.sender], edge.spec.assertion_threshold, coin
            )
            if positive is not None and listen_draw < edge.spec.listen_chance:
                reports.append(
                    _Report(
                        edge.spec.receiver,
                        ("edge", edge.spec.sender),
                        positive,
                        "message",
                        edge.spec.sender,
                    )
                )
        return reports

    def _log(self, step: int, agent: _Agent, report: _Report, before: float,
             trust_before: float) -> None:
        if report.kind == "inquiry":
            self.evidence_events += 1
        else:
            self.message_events += 1
        if not self.record_events:
            return
        entry: dict[str, Any] = {"step": step, "kind": report.kind}
        if report.kind == "inquiry":
            entry["agent"] = report.receiver
        else:
            entry["sender"] = report.sender
            entry["receiver"] = report.receiver
        entry.update(
            {
                "positive": report.positive,
                "belief_before": before,
                "belief_after": agent.belief,
                "trust_before": trust_before,
                "trust_after": self._trust(agent, report.source_key).expected_reliability,
            }
        )
        self.event_log.append(entry)

    def _deliver(self, step: int, report: _Report, basis: float) -> None:
        agent = self.agents[report.receiver]
        before = agent.belief
        trust_before = (
            self._trust(agent, report.source_key).expected_reliability
            if self.record_events else float("nan")
        )
        self._apply(agent, report.source_key, report.positive, basis)
        agent.received_evidence += 1
        self._log(step, agent, report, before, trust_before)

    def _step_simultaneous(self, step: int) -> None:
        start = {agent_id: agent.belief for agent_id, agent in self.agents.items()}
        start_log_odds = {agent_id: agent.log_odds for agent_id, agent in self.agents.items()}
        inquiries = self._draw_inquiries()
        messages = self._draw_messages(start_log_odds)
        reports = [
            _Report(agent_id, _INQUIRY, positive, "inquiry")
            for agent_id, positive in inquiries.items()
            if positive is not None
        ]
        reports.extend(messages)
        # Log-odds contributions add, so the cumulative credence after the last
        # report of each agent is order independent; trust uses C_t.
        for report in sorted(
            reports,
            key=lambda r: (r.receiver, r.kind != "inquiry", r.sender or ""),
        ):
            self._deliver(step, report, start[report.receiver])
        for agent_id, agent in self.agents.items():
            agent.inquired_last_round = inquiries[agent_id] is not None

    def _step_inquiry_then_messages(self, step: int) -> None:
        inquiries = self._draw_inquiries()
        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            positive = inquiries[agent_id]
            agent.inquired_this_round = positive is not None
            if positive is not None:
                self._deliver(
                    step, _Report(agent_id, _INQUIRY, positive, "inquiry"), agent.belief
                )
        log_odds = {agent_id: agent.log_odds for agent_id, agent in self.agents.items()}
        for report in self._draw_messages(log_odds):
            self._deliver(step, report, self.agents[report.receiver].belief)

    # ----------------------------------------------------- Laputa 1.6 schedule
    # A re-statement of Society::evolve and Inquirer::doInquiry in the Laputa
    # 1.6 source, including the new-evidence policies. Agents are processed in
    # sorted-ID order, which plays the role of Laputa's inquirer index; each
    # agent's incoming links are consulted in sorted sender order (Laputa's
    # link-map order). Under the "inquiry" and "any" policies the outcome can
    # depend on this order, exactly as in Laputa: an inquiry made earlier in
    # the same step by a lower-indexed source already counts as new evidence.
    # Laputa's step counter starts at 0; ``step - 1`` is used to match it.

    def _reset_laputa_1_6(self) -> None:
        """Simulation::reset: evidence counters and initial link messages."""
        for agent in self.agents.values():
            agent.last_inquiry_step = -1
            agent.last_inquiry_result = None
        for edge in self.edges:
            edge.last_used_step = -1
            edge.message = None
            if self.evidence_policy != "none":
                belief = self.agents[edge.spec.sender].initial_belief
                threshold = edge.spec.assertion_threshold
                if belief > 0.5 and belief > threshold:
                    edge.message = True
                elif belief < 0.5 and belief < 1.0 - threshold:
                    edge.message = False

    def _has_new_evidence(self, edge: _Edge, receiver_id: str) -> bool:
        sender = self.agents[edge.spec.sender]
        new = edge.last_used_step < sender.last_inquiry_step
        if self.count_prior_as_evidence and edge.last_used_step == -1:
            new = True
        if self.evidence_policy == "any" and not new:
            new = any(
                other.spec.sender != receiver_id and other.last_used_step > edge.last_used_step
                for other in self._incoming[edge.spec.sender]
            )
        return new

    def _step_laputa_1_6(self, step: int) -> None:
        t = step - 1
        start = {agent_id: agent.belief for agent_id, agent in self.agents.items()}
        start_log_odds = {agent_id: agent.log_odds for agent_id, agent in self.agents.items()}
        # Fixed number of draws per stream and step (common random numbers).
        agent_draws = {
            agent_id: (agent.rng.random(), agent.rng.random())
            for agent_id, agent in sorted(self.agents.items())
        }
        edge_draws = {id(edge): (edge.rng.random(), edge.rng.random()) for edge in self.edges}
        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            chance_draw, accuracy_draw = agent_draws[agent_id]
            if chance_draw < agent.spec.inquiry_chance:
                agent.last_inquiry_step = t
                correct = accuracy_draw < agent.spec.inquiry_accuracy
                positive = self.truth if correct else not self.truth
                agent.last_inquiry_result = positive
                self._deliver(step, _Report(agent_id, _INQUIRY, positive, "inquiry"),
                              start[agent_id])
            for edge in self._incoming[agent_id]:
                listen_draw, coin = edge_draws[id(edge)]
                if self.evidence_policy == "none":
                    said = self._assertion(
                        start_log_odds[edge.spec.sender], edge.spec.assertion_threshold, coin
                    )
                elif self._has_new_evidence(edge, agent_id):
                    said = edge.message
                else:
                    said = None
                if listen_draw < edge.spec.listen_chance and said is not None:
                    edge.last_used_step = t
                    self._deliver(
                        step,
                        _Report(agent_id, ("edge", edge.spec.sender), said, "message",
                                edge.spec.sender),
                        start[agent_id],
                    )
        if self.evidence_policy != "none":
            for edge in self.edges:
                sender_id = edge.spec.sender
                new_belief = self.agents[sender_id].belief
                threshold = edge.spec.assertion_threshold
                if self.evidence_policy == "inquiry":
                    last = self.agents[sender_id].last_inquiry_result
                    if last is True and new_belief >= threshold:
                        edge.message = True
                    elif last is False and new_belief <= 1.0 - threshold:
                        edge.message = False
                else:
                    old_belief = start[sender_id]
                    if new_belief > old_belief and new_belief >= threshold:
                        edge.message = True
                    elif new_belief < old_belief and new_belief <= 1.0 - threshold:
                        edge.message = False

    # ------------------------------------------------------------------- run
    def _truth_credence(self) -> float:
        values = [a.belief if self.truth else 1.0 - a.belief for a in self.agents.values()]
        return sum(values) / len(values)

    def run(self) -> SimulationResult:
        if self._ran:
            raise RuntimeError("a Simulation instance can be run only once")
        self._ran = True
        initial_mean = sum(a.initial_belief for a in self.agents.values()) / len(self.agents)
        initial_truth_credence = self._truth_credence()
        trajectory = [initial_truth_credence] if self.record_trajectory else None
        step_fn = {
            "simultaneous": self._step_simultaneous,
            "inquiry_then_messages": self._step_inquiry_then_messages,
            "laputa_1_6": self._step_laputa_1_6,
        }[self.schedule]
        for step in range(1, self.steps + 1):
            step_fn(step)
            if trajectory is not None:
                trajectory.append(self._truth_credence())

        n = len(self.agents)
        final = {agent_id: agent.belief for agent_id, agent in self.agents.items()}
        final_mean = sum(final.values()) / n
        target = 1.0 if self.truth else 0.0
        brier = sum((belief - target) ** 2 for belief in final.values()) / n
        # Log score -log P(truth), computed from log-odds to stay finite for
        # saturated but non-dogmatic credences.
        log_losses = [
            -_log_sigmoid(a.log_odds if self.truth else -a.log_odds)
            for a in self.agents.values()
        ]
        mean_log = (
            sum(log_losses) / n if all(isfinite(x) for x in log_losses) else None
        )
        correct = sum(
            (belief > 0.5) if self.truth else (belief < 0.5) for belief in final.values()
        )
        undecided = sum(belief == 0.5 for belief in final.values())
        final_truth_credence = self._truth_credence()
        dispersion = sqrt(sum((b - final_mean) ** 2 for b in final.values()) / n)
        final_trust = {
            f"{edge.spec.sender}->{edge.spec.receiver}": self._trust(
                self.agents[edge.spec.receiver], ("edge", edge.spec.sender)
            ).expected_reliability
            for edge in self.edges
        }
        for agent_id, agent in self.agents.items():
            final_trust[f"{agent_id}:inquiry"] = self._trust(agent, _INQUIRY).expected_reliability
        return SimulationResult(
            truth=self.truth,
            steps=self.steps,
            seed=self.seed,
            initial_mean_belief=initial_mean,
            final_mean_belief=final_mean,
            final_beliefs=final,
            mean_brier_score=brier,
            mean_log_score=mean_log,
            accuracy_at_half=correct / n,
            undecided_fraction=undecided / n,
            veritistic_value=final_truth_credence - initial_truth_credence,
            belief_dispersion=dispersion,
            belief_spread=max(final.values()) - min(final.values()),
            evidence_events=self.evidence_events,
            message_events=self.message_events,
            publish_only_after_inquiry=self.publish_only_after_inquiry,
            trust_points=self.trust_points,
            update_rule=self.update_rule,
            schedule=self.schedule,
            assertion_rule=self.assertion_rule,
            configuration=self.configuration(),
            final_trust=final_trust,
            truth_credence_trajectory=trajectory,
            event_log=self.event_log,
            trust_model=self.trust_model,
            evidence_policy=self.evidence_policy,
            count_prior_as_evidence=self.count_prior_as_evidence,
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "truth": self.truth,
            "steps": self.steps,
            "seed": self.seed,
            "trust_points": self.trust_points,
            "publish_only_after_inquiry": self.publish_only_after_inquiry,
            "update_rule": self.update_rule,
            "schedule": self.schedule,
            "assertion_rule": self.assertion_rule,
            "trust_model": self.trust_model,
            "evidence_policy": self.evidence_policy,
            "count_prior_as_evidence": self.count_prior_as_evidence,
            "agents": [asdict(spec) for spec in self._agent_specs],
            "edges": [asdict(edge.spec) for edge in self.edges],
        }
