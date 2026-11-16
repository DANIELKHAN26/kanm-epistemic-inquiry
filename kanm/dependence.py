"""Exact inference for dependent binary reports with explicit provenance.

A provenance graph says *which* reports descend from which; it does not by
itself say *how strongly* they are dependent. This module makes that missing
generative assumption explicit. Each report node ``X_i`` is binary (it
supports ``p`` or ``not p``) and has a conditional probability table
``P(X_i = True | H, parents)``, where ``H`` is the truth of ``p``. The graph is
then a Bayesian network (Pearl 1988) and the posterior ``P(H | observed
reports)`` is computed exactly by enumerating the unobserved ancestors.

The module also provides the *independence comparator*: the posterior an
agent would reach by multiplying each observed report's exact marginal
likelihood ratio as if the reports were conditionally independent given ``H``.
That comparator isolates the error due to ignoring dependence alone, because
each report's individual diagnosticity is computed correctly.

Exact enumeration is exponential in the number of latent ancestors and is
limited to :data:`MAX_LATENT` nodes. It is intended for verification and for
small case studies, and as ground truth for approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, inf, isfinite, log
from types import MappingProxyType
from typing import Mapping, Sequence

from .bayes import _probability, expit, logit

MAX_LATENT = 20

CPD = Mapping[tuple[bool, tuple[bool, ...]], float]


@dataclass(frozen=True)
class ReportNode:
    """One binary report with a conditional probability table.

    ``cpd[(h, parent_values)]`` is ``P(X = True | H = h, parents = parent_values)``
    with ``parent_values`` ordered as in ``parents``. Every combination must be
    present.
    """

    node_id: str
    parents: tuple[str, ...]
    cpd: CPD
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a nonempty string")
        if not isinstance(self.parents, tuple) or any(
            not isinstance(p, str) or not p.strip() for p in self.parents
        ):
            raise ValueError("parents must be a tuple of node IDs")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("parents must be unique")
        if self.node_id in self.parents:
            raise ValueError("a node cannot be its own parent")
        if not isinstance(self.cpd, Mapping):
            raise ValueError("cpd must be a mapping")
        expected = {
            (h, values)
            for h in (True, False)
            for values in product((True, False), repeat=len(self.parents))
        }
        if set(self.cpd) != expected:
            raise ValueError(
                f"cpd for {self.node_id} must define every (truth, parent values) combination"
            )
        table = {key: _probability(value, f"{self.node_id} cpd") for key, value in self.cpd.items()}
        object.__setattr__(self, "cpd", MappingProxyType(table))

    def probability(self, value: bool, truth: bool, parent_values: tuple[bool, ...]) -> float:
        p_true = self.cpd[(truth, parent_values)]
        return p_true if value else 1.0 - p_true


def primary(node_id: str, sensitivity: float, false_positive_rate: float,
            description: str = "") -> ReportNode:
    """A report generated directly from the world.

    ``sensitivity = P(X = True | p)``; ``false_positive_rate = P(X = True | not p)``.
    Unequal error rates relax Laputa's symmetric-reliability assumption.
    """
    sensitivity = _probability(sensitivity, "sensitivity")
    false_positive_rate = _probability(false_positive_rate, "false_positive_rate")
    return ReportNode(
        node_id, (), {(True, ()): sensitivity, (False, ()): false_positive_rate}, description
    )


def symmetric_primary(node_id: str, reliability: float, description: str = "") -> ReportNode:
    """Laputa-style source: correct with probability ``reliability`` in both worlds."""
    reliability = _probability(reliability, "reliability")
    return primary(node_id, reliability, 1.0 - reliability, description)


def relay(node_id: str, parent: str, fidelity: float, description: str = "") -> ReportNode:
    """A report that copies its parent with probability ``fidelity``.

    The relayer adds no independent access to the world: given the parent's
    value, the relay is independent of ``H``. ``fidelity = 1`` is a literal
    repeat; ``fidelity = 0.5`` makes the relay uninformative.
    """
    fidelity = _probability(fidelity, "fidelity")
    cpd = {
        (h, (parent_value,)): fidelity if parent_value else 1.0 - fidelity
        for h in (True, False)
        for parent_value in (True, False)
    }
    return ReportNode(node_id, (parent,), cpd, description)


def corroborating_relay(node_id: str, parent: str, fidelity: float, own_reliability: float,
                        mix: float, description: str = "") -> ReportNode:
    """Mixture of copying the parent and checking the world independently.

    With probability ``mix`` the reporter consults the world with symmetric
    ``own_reliability``; otherwise it copies the parent with ``fidelity``. This
    is one explicit model of *partial* dependence; it is not the only one.
    """
    fidelity = _probability(fidelity, "fidelity")
    own_reliability = _probability(own_reliability, "own_reliability")
    mix = _probability(mix, "mix")
    cpd = {}
    for h in (True, False):
        own = own_reliability if h else 1.0 - own_reliability
        for parent_value in (True, False):
            copied = fidelity if parent_value else 1.0 - fidelity
            cpd[(h, (parent_value,))] = mix * own + (1.0 - mix) * copied
    return ReportNode(node_id, (parent,), cpd, description)


class ProvenanceNetwork:
    """Append-only DAG of report nodes with exact posterior inference."""

    def __init__(self) -> None:
        self.nodes: dict[str, ReportNode] = {}

    def add(self, node: ReportNode) -> None:
        if not isinstance(node, ReportNode):
            raise TypeError("node must be a ReportNode")
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node ID: {node.node_id}")
        for parent in node.parents:
            if parent not in self.nodes:
                raise ValueError(f"parent must be added first: {parent}")
        self.nodes[node.node_id] = node

    def ancestors(self, node_ids: Sequence[str]) -> set[str]:
        found: set[str] = set()
        stack = list(node_ids)
        while stack:
            current = stack.pop()
            for parent in self.nodes[current].parents:
                if parent not in found:
                    found.add(parent)
                    stack.append(parent)
        return found

    def _check_observations(self, observations: Mapping[str, bool]) -> dict[str, bool]:
        if not isinstance(observations, Mapping):
            raise ValueError("observations must map node IDs to booleans")
        checked = {}
        for node_id, value in observations.items():
            if node_id not in self.nodes:
                raise ValueError(f"unknown node: {node_id}")
            if not isinstance(value, bool):
                raise ValueError(f"observation for {node_id} must be a boolean")
            checked[node_id] = value
        return checked

    def _log_evidence(self, observations: Mapping[str, bool]) -> tuple[float, float]:
        """Return ``(log P(obs | p), log P(obs | not p))`` exactly."""
        observations = self._check_observations(observations)
        relevant = set(observations) | self.ancestors(list(observations))
        latent = [n for n in self.nodes if n in relevant and n not in observations]
        if len(latent) > MAX_LATENT:
            raise ValueError(
                f"{len(latent)} latent ancestors exceeds exact-enumeration limit {MAX_LATENT}"
            )
        order = [n for n in self.nodes if n in relevant]  # insertion order is topological
        results = []
        for truth in (True, False):
            terms = []
            for assignment in product((True, False), repeat=len(latent)):
                values = dict(observations)
                values.update(zip(latent, assignment))
                log_weight = 0.0
                for node_id in order:
                    node = self.nodes[node_id]
                    parent_values = tuple(values[p] for p in node.parents)
                    probability = node.probability(values[node_id], truth, parent_values)
                    if probability <= 0.0:
                        log_weight = -inf
                        break
                    log_weight += log(probability)
                if log_weight > -inf:
                    terms.append(log_weight)
            results.append(_log_sum_exp(terms))
        return results[0], results[1]

    def marginal_likelihoods(self, node_id: str) -> tuple[float, float]:
        """``(P(X = True | p), P(X = True | not p))`` with ancestors summed out."""
        log_true, log_false = self._log_evidence({node_id: True})
        return (exp(log_true) if log_true > -inf else 0.0,
                exp(log_false) if log_false > -inf else 0.0)

    def log_likelihood_ratio(self, observations: Mapping[str, bool]) -> float:
        """Exact ``log P(obs | p) - log P(obs | not p)`` (the log-odds shift)."""
        log_true, log_false = self._log_evidence(observations)
        return _log_ratio(log_true, log_false)

    def independence_log_likelihood_ratio(self, observations: Mapping[str, bool]) -> float:
        """Sum of each observed report's exact marginal log-likelihood ratio."""
        observations = self._check_observations(observations)
        total = 0.0
        for node_id, value in observations.items():
            p_true, p_false = self.marginal_likelihoods(node_id)
            a = p_true if value else 1.0 - p_true
            b = p_false if value else 1.0 - p_false
            ratio = _log_ratio(log(a) if a > 0.0 else -inf, log(b) if b > 0.0 else -inf)
            total = _combine(total, ratio)
        return total

    def posterior(self, prior: float, observations: Mapping[str, bool]) -> float:
        """Exact ``P(p | observations)`` under the network's generative model."""
        lo = logit(_probability(prior, "prior"))
        return expit(_combine(lo, self.log_likelihood_ratio(observations)))

    def independence_posterior(self, prior: float, observations: Mapping[str, bool]) -> float:
        """Posterior if observed reports were treated as conditionally independent.

        Each report contributes its own exact marginal likelihood ratio, so
        any difference from :meth:`posterior` is due to ignored dependence.
        """
        lo = logit(_probability(prior, "prior"))
        return expit(_combine(lo, self.independence_log_likelihood_ratio(observations)))

    def dependence_report(self, prior: float, observations: Mapping[str, bool]) -> dict[str, float]:
        """Exact versus independence posterior and the overcounting in log-odds."""
        prior_lo = logit(_probability(prior, "prior"))
        exact_llr = self.log_likelihood_ratio(observations)
        naive_llr = self.independence_log_likelihood_ratio(observations)
        exact = expit(_combine(prior_lo, exact_llr))
        naive = expit(_combine(prior_lo, naive_llr))
        # Shifts come straight from the likelihood ratios, so they stay exact
        # even when the posteriors themselves round to 0 or 1.
        exact_shift = _finite_or_none(exact_llr)
        naive_shift = _finite_or_none(naive_llr)
        ratio = None
        if exact_shift not in (None, 0.0) and naive_shift is not None:
            ratio = naive_shift / exact_shift
        return {
            "exact_posterior": exact,
            "independence_posterior": naive,
            "exact_log_odds_shift": exact_shift,
            "independence_log_odds_shift": naive_shift,
            "overcounting_ratio": ratio,
        }

    @classmethod
    def from_ledger(
        cls,
        ledger,
        *,
        question_id: str,
        positive_outcome: str,
        negative_outcome: str,
        primary_reliability: Mapping[str, float] | float,
        relay_fidelity: Mapping[str, float] | float,
        cpds: Mapping[str, CPD] | None = None,
        observed: Sequence[str] | None = None,
    ) -> tuple["ProvenanceNetwork", dict[str, bool]]:
        """Build a network and observations from a :class:`kanm.evidence.EvidenceLedger`.

        Items without parents become symmetric primary reports; single-parent
        items become relays. Multi-parent items need an explicit table in
        ``cpds`` because no default combination rule is justified. Reliability
        and fidelity mappings are keyed by evidence ID. All items must address
        ``question_id`` and report one of the two named outcomes. ``observed``
        lists the evidence IDs the reasoner has actually seen (default: all);
        the others are treated as latent and summed out.
        """
        from .evidence import EvidenceLedger

        if not isinstance(ledger, EvidenceLedger):
            raise TypeError("ledger must be an EvidenceLedger")
        cpds = dict(cpds or {})
        unused = sorted(set(cpds) - set(ledger.items))
        if unused:
            raise ValueError(f"cpds given for unknown evidence IDs: {unused}")
        for table, label in ((primary_reliability, "primary_reliability"),
                             (relay_fidelity, "relay_fidelity")):
            if isinstance(table, Mapping):
                extra = sorted(set(table) - set(ledger.items))
                if extra:
                    raise ValueError(f"{label} given for unknown evidence IDs: {extra}")

        def lookup(table, key, label):
            if isinstance(table, Mapping):
                if key not in table:
                    raise ValueError(f"missing {label} for evidence {key}")
                return table[key]
            return table

        network = cls()
        observations: dict[str, bool] = {}
        for evidence_id, item in ledger.items.items():
            if item.question_id != question_id:
                raise ValueError(f"evidence {evidence_id} addresses another question")
            if item.outcome not in (positive_outcome, negative_outcome):
                raise ValueError(f"evidence {evidence_id} has a non-binary outcome")
            observations[evidence_id] = item.outcome == positive_outcome
            if evidence_id in cpds:
                node = ReportNode(evidence_id, item.parents, cpds[evidence_id])
            elif not item.parents:
                node = symmetric_primary(
                    evidence_id, lookup(primary_reliability, evidence_id, "reliability")
                )
            elif len(item.parents) == 1:
                node = relay(
                    evidence_id, item.parents[0], lookup(relay_fidelity, evidence_id, "fidelity")
                )
            else:
                raise ValueError(
                    f"evidence {evidence_id} has several parents; supply its table in cpds"
                )
            network.add(node)
        if observed is not None:
            observed = list(observed)
            missing = [e for e in observed if e not in observations]
            if missing:
                raise ValueError(f"unknown observed evidence: {missing}")
            observations = {e: observations[e] for e in observed}
        return network, observations


def _log_sum_exp(terms: list[float]) -> float:
    if not terms:
        return -inf
    peak = max(terms)
    return peak + log(sum(exp(t - peak) for t in terms))


def _log_ratio(log_a: float, log_b: float) -> float:
    if log_a == -inf and log_b == -inf:
        raise ValueError("observations have zero probability under the model")
    if log_b == -inf:
        return inf
    if log_a == -inf:
        return -inf
    return log_a - log_b


def _combine(a: float, b: float) -> float:
    total = a + b
    if total != total:
        raise ValueError("contradictory certain evidence (infinite log-odds of both signs)")
    return total


def _finite_or_none(x: float) -> float | None:
    return x if isfinite(x) else None
