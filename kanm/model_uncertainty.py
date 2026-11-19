"""Posterior and predictive sensitivity across provenance models.

``ProvenanceNetwork`` is exact conditional on one declared graph and its
conditional-probability tables. This module makes uncertainty about that
declaration explicit. It offers posterior/predictive envelopes across a set of
candidate networks and Bayesian model averaging when prior model weights are
supplied. It does not learn provenance graphs or assign model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, inf, isfinite, log, log1p
from types import MappingProxyType
from typing import Mapping

from .bayes import _probability
from .dependence import ProvenanceNetwork


@dataclass(frozen=True)
class PosteriorEnvelope:
    """Range of posterior belief across candidate models supporting the data."""

    lower: float
    upper: float
    by_model: Mapping[str, float]
    excluded_models: tuple[str, ...]


@dataclass(frozen=True)
class ModelAveragedPosterior:
    """Posterior under Bayesian averaging over a stated prior on models."""

    posterior: float
    model_weights: Mapping[str, float]
    posterior_by_model: Mapping[str, float]
    excluded_models: tuple[str, ...]


@dataclass(frozen=True)
class PredictiveEnvelope:
    """Range of a report's predictive probability across candidate models."""

    lower_true: float
    upper_true: float
    by_model: Mapping[str, Mapping[bool, float]]
    excluded_models: tuple[str, ...]


@dataclass(frozen=True)
class CredalPosterior:
    """Posterior range when model weights are known only within bounds.

    ``weights_at_lower`` and ``weights_at_upper`` are prior model weights in
    the allowed set that attain the two ends of the range.
    """

    lower: float
    upper: float
    weights_at_lower: Mapping[str, float]
    weights_at_upper: Mapping[str, float]
    excluded_models: tuple[str, ...]


@dataclass(frozen=True)
class InformationGain:
    """Expected information from one report about the graph and the claim, in bits.

    ``about_model`` is the mutual information between the report and model
    identity, ``about_claim`` between the report and the claim, both given the
    reports already observed and the stated model weights.
    """

    about_model: float
    about_claim: float


@dataclass(frozen=True)
class ModelAveragedPredictive:
    """Predictive report distribution and updated model weights."""

    probabilities: Mapping[bool, float]
    model_weights: Mapping[str, float]
    by_model: Mapping[str, Mapping[bool, float]]
    excluded_models: tuple[str, ...]


def _log_sum_exp(values: list[float]) -> float:
    if not values:
        return -inf
    peak = max(values)
    if peak == -inf:
        return -inf
    return peak + log(fsum(exp(value - peak) for value in values))


def _posterior_from_log_masses(log_true: float, log_false: float) -> float:
    normalizer = _log_sum_exp([log_true, log_false])
    if normalizer == -inf:
        raise ValueError("observations have zero probability under the prior model")
    return 0.0 if log_true == -inf else exp(log_true - normalizer)


def _readonly(values: Mapping) -> Mapping:
    return MappingProxyType(dict(values))


class ProvenanceModelSet:
    """A finite set of alternative provenance networks for one binary claim.

    Every network must use the same report IDs, but may specify different
    edges or conditional-probability tables. The robust envelope methods treat
    these networks as a set and do not need model probabilities. The
    model-averaged methods require explicit ``model_weights``; those weights
    are prior probabilities for the model and are assumed independent of the
    claim truth before any reports are observed. The data update both truth and
    model probabilities by Bayes' rule.

    Networks are defensively copied on construction. Each network still obeys
    the exact-inference latent-node cap in :mod:`kanm.dependence`; total work
    grows with the number of candidate networks.
    """

    def __init__(
        self,
        models: Mapping[str, ProvenanceNetwork],
        *,
        model_weights: Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(models, Mapping) or not models:
            raise ValueError("models must be a nonempty mapping of names to networks")
        snapshots: dict[str, ProvenanceNetwork] = {}
        reference_ids: set[str] | None = None
        for name, network in models.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("model names must be nonempty strings")
            if not isinstance(network, ProvenanceNetwork):
                raise TypeError(f"model {name!r} must be a ProvenanceNetwork")
            node_ids = set(network.node_ids)
            if reference_ids is None:
                reference_ids = node_ids
            elif node_ids != reference_ids:
                raise ValueError("all candidate networks must contain the same report IDs")
            snapshot = ProvenanceNetwork()
            for node in network.nodes.values():
                snapshot.add(node)
            snapshots[name] = snapshot
        self._models = MappingProxyType(snapshots)
        self._model_names = tuple(snapshots)
        self._node_ids = frozenset(reference_ids or ())

        if model_weights is None:
            self._model_weights: Mapping[str, float] | None = None
        else:
            if not isinstance(model_weights, Mapping):
                raise ValueError("model_weights must map every model name to a probability")
            if set(model_weights) != set(self._model_names):
                raise ValueError("model_weights must have exactly the candidate model names")
            weights: dict[str, float] = {}
            for name in self._model_names:
                value = model_weights[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("model weights must be finite nonnegative probabilities")
                try:
                    weights[name] = float(value)
                except OverflowError as error:
                    raise ValueError(
                        "model weights must be finite nonnegative probabilities"
                    ) from error
                if not isfinite(weights[name]) or not 0.0 <= weights[name] <= 1.0:
                    raise ValueError("model weights must be finite nonnegative probabilities")
            total = fsum(weights.values())
            if total <= 0.0 or abs(total - 1.0) > 1e-9:
                raise ValueError("model weights must sum to 1")
            self._model_weights = _readonly(
                {name: weight / total for name, weight in weights.items()}
            )

    @property
    def model_names(self) -> tuple[str, ...]:
        """Candidate model names in their supplied order."""
        return self._model_names

    @property
    def node_ids(self) -> frozenset[str]:
        """Report IDs shared by every candidate model."""
        return self._node_ids

    @property
    def model_weights(self) -> Mapping[str, float] | None:
        """Prior model probabilities, or ``None`` when only a set was given."""
        return self._model_weights

    def model(self, name: str) -> ProvenanceNetwork:
        """Return a defensive copy of one candidate for inspection or editing."""
        if not isinstance(name, str):
            raise ValueError("model name must be a string")
        try:
            source = self._models[name]
        except KeyError as error:
            raise ValueError(f"unknown model: {name}") from error
        copied = ProvenanceNetwork()
        for node in source.nodes.values():
            copied.add(node)
        return copied

    def validate_observations(self, observations: Mapping[str, bool]) -> dict[str, bool]:
        """Validate and copy observations against the shared report IDs."""
        if not isinstance(observations, Mapping):
            raise ValueError("observations must map node IDs to booleans")
        checked: dict[str, bool] = {}
        for node_id, value in observations.items():
            if not isinstance(node_id, str) or node_id not in self._node_ids:
                raise ValueError(f"unknown node: {node_id}")
            if not isinstance(value, bool):
                raise ValueError(f"observation for {node_id} must be a boolean")
            checked[node_id] = value
        return checked

    def _model_log_masses(
        self, model: ProvenanceNetwork, prior: float, observations: Mapping[str, bool]
    ) -> tuple[float, float]:
        log_true = model.log_evidence(observations, True)
        log_false = model.log_evidence(observations, False)
        joint_true = log(prior) + log_true if prior > 0.0 else -inf
        joint_false = log1p(-prior) + log_false if prior < 1.0 else -inf
        return joint_true, joint_false

    def posterior_envelope(
        self, prior: float, observations: Mapping[str, bool]
    ) -> PosteriorEnvelope:
        """Return min/max ``P(H | observations, model)`` over supported models.

        Candidate models that assign zero probability to the observed data
        under the stated claim prior are listed in ``excluded_models``. If all
        candidates do so, the observation is inconsistent with the model set.
        The interval covers this finite set only; it is not a confidence or
        credible interval over models that were never specified. This set-based
        result ignores any optional ``model_weights`` and includes every
        supported candidate, including one assigned zero averaging weight.
        """
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        by_model: dict[str, float] = {}
        excluded: list[str] = []
        for name, model in self._models.items():
            log_true, log_false = self._model_log_masses(model, prior, checked)
            if _log_sum_exp([log_true, log_false]) == -inf:
                excluded.append(name)
                continue
            by_model[name] = _posterior_from_log_masses(log_true, log_false)
        if not by_model:
            raise ValueError("observations have zero probability under every candidate model")
        frozen = _readonly(by_model)
        return PosteriorEnvelope(
            min(by_model.values()), max(by_model.values()), frozen, tuple(excluded)
        )

    def model_averaged_posterior(
        self, prior: float, observations: Mapping[str, bool]
    ) -> ModelAveragedPosterior:
        """Update claim and model probabilities under explicit Bayesian weights.

        For model ``m``, the posterior mass is proportional to
        ``w_m P(observations | m)``. The reported claim posterior then averages
        the model-conditional posteriors using these *updated* weights, not the
        prior weights. A model with zero prior weight or impossible data has
        zero posterior weight and appears in ``excluded_models``.
        """
        weights = self._require_model_weights()
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        log_model_masses: dict[str, float] = {}
        by_model: dict[str, float] = {}
        excluded: list[str] = []
        for name, model in self._models.items():
            weight = weights[name]
            if weight == 0.0:
                excluded.append(name)
                continue
            log_true, log_false = self._model_log_masses(model, prior, checked)
            log_evidence = _log_sum_exp([log_true, log_false])
            if log_evidence == -inf:
                excluded.append(name)
                continue
            log_model_masses[name] = log(weight) + log_evidence
            by_model[name] = _posterior_from_log_masses(log_true, log_false)
        normalizer = _log_sum_exp(list(log_model_masses.values()))
        if normalizer == -inf:
            raise ValueError("observations have zero probability under every weighted model")
        posterior_weights = {
            name: exp(log_mass - normalizer)
            for name, log_mass in log_model_masses.items()
        }
        total_weight = fsum(posterior_weights.values())
        posterior_weights = {
            name: weight / total_weight for name, weight in posterior_weights.items()
        }
        posterior = fsum(posterior_weights[name] * by_model[name] for name in by_model)
        return ModelAveragedPosterior(
            min(1.0, max(0.0, posterior)),
            _readonly(posterior_weights),
            _readonly(by_model),
            tuple(excluded),
        )

    def posterior(self, prior: float, observations: Mapping[str, bool]) -> float:
        """Return the Bayesian model-averaged posterior; explicit weights required."""
        return self.model_averaged_posterior(prior, observations).posterior

    def predictive_envelope(
        self, node_id: str, prior: float, observations: Mapping[str, bool]
    ) -> PredictiveEnvelope:
        """Bound ``P(report=True | observations, model)`` across supported models."""
        if not isinstance(node_id, str) or node_id not in self._node_ids:
            raise ValueError(f"unknown node: {node_id}")
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        by_model: dict[str, Mapping[bool, float]] = {}
        excluded: list[str] = []
        for name, model in self._models.items():
            log_true, log_false = self._model_log_masses(model, prior, checked)
            if _log_sum_exp([log_true, log_false]) == -inf:
                excluded.append(name)
                continue
            by_model[name] = _readonly(
                model.predictive_distribution(node_id, prior, checked)
            )
        if not by_model:
            raise ValueError("observations have zero probability under every candidate model")
        probabilities = [distribution[True] for distribution in by_model.values()]
        return PredictiveEnvelope(
            min(probabilities), max(probabilities), _readonly(by_model), tuple(excluded)
        )

    def model_averaged_predictive(
        self, node_id: str, prior: float, observations: Mapping[str, bool]
    ) -> ModelAveragedPredictive:
        """Predict a report after updating model weights on the observed reports."""
        if not isinstance(node_id, str) or node_id not in self._node_ids:
            raise ValueError(f"unknown node: {node_id}")
        checked = self.validate_observations(observations)
        averaged = self.model_averaged_posterior(prior, checked)
        by_model: dict[str, Mapping[bool, float]] = {}
        for name in averaged.model_weights:
            by_model[name] = _readonly(
                self._models[name].predictive_distribution(node_id, prior, checked)
            )
        p_true = fsum(
            averaged.model_weights[name] * distribution[True]
            for name, distribution in by_model.items()
        )
        p_true = min(1.0, max(0.0, p_true))
        probabilities = {False: 1.0 - p_true, True: p_true}
        return ModelAveragedPredictive(
            _readonly(probabilities),
            averaged.model_weights,
            _readonly(by_model),
            averaged.excluded_models,
        )

    def predictive_distribution(
        self, node_id: str, prior: float, observations: Mapping[str, bool]
    ) -> dict[bool, float]:
        """Return the model-averaged predictive distribution for a report."""
        return dict(self.model_averaged_predictive(node_id, prior, observations).probabilities)

    def credal_posterior(
        self,
        prior: float,
        observations: Mapping[str, bool],
        *,
        lower: Mapping[str, float],
        upper: Mapping[str, float],
    ) -> CredalPosterior:
        """Range of the model-averaged posterior over interval model weights.

        The allowed prior weights are every ``w`` with
        ``lower[m] <= w[m] <= upper[m]`` that sums to one. Each such ``w`` is
        updated by Bayes' rule, and the result is the smallest and largest
        model-averaged posterior that any of them gives. Weights under which
        the observations are impossible are left out, as in Walley's regular
        extension. With bounds ``[0, 1]`` the range equals
        :meth:`posterior_envelope`; with equal bounds it is the single
        model-averaged posterior.

        The posterior is a ratio of two linear functions of ``w``. Each end is
        found exactly by Dinkelbach's method, whose inner step is a greedy
        allocation over the bounds.
        """
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        low, high = self._weight_bounds(lower, upper)
        log_evidence: dict[str, float] = {}
        by_model: dict[str, float] = {}
        excluded: list[str] = []
        for name, model in self._models.items():
            log_true, log_false = self._model_log_masses(model, prior, checked)
            evidence = _log_sum_exp([log_true, log_false])
            if evidence == -inf:
                excluded.append(name)
                continue
            log_evidence[name] = evidence
            by_model[name] = _posterior_from_log_masses(log_true, log_false)
        # Scale evidence by the largest weighted evidence any allowed model can
        # carry, so that a well-supported model capped at weight 0 cannot push
        # the others below floating-point resolution.
        usable = [name for name in log_evidence if high[name] > 0.0]
        if not usable:
            raise ValueError("observations have zero probability under every allowed weighting")
        peak = max(log_evidence[name] + log(high[name]) for name in usable)
        try:
            scale = {
                name: exp(log_evidence[name] - peak) if name in usable else 0.0
                for name in self._model_names
            }
        except OverflowError as error:
            raise ValueError("evidence ratios between models exceed floating-point range") from error
        posterior = {name: by_model.get(name, 0.0) for name in self._model_names}
        if _greedy_weights(scale, low, high)[1] <= 0.0:
            raise ValueError("observations have zero probability under every allowed weighting")
        upper_value, upper_weights = _max_ratio(scale, posterior, low, high)
        complement = {name: 1.0 - value for name, value in posterior.items()}
        lower_complement, lower_weights = _max_ratio(scale, complement, low, high)
        return CredalPosterior(
            min(1.0, max(0.0, 1.0 - lower_complement)),
            min(1.0, max(0.0, upper_value)),
            _readonly(lower_weights),
            _readonly(upper_weights),
            tuple(excluded),
        )

    def _weight_bounds(
        self, lower: Mapping[str, float], upper: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        bounds = []
        for label, values in (("lower", lower), ("upper", upper)):
            if not isinstance(values, Mapping) or set(values) != set(self._model_names):
                raise ValueError(f"{label} weight bounds must name every candidate model")
            checked = {}
            for name in self._model_names:
                value = values[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("weight bounds must be numbers in [0, 1]")
                value = float(value)
                if not isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("weight bounds must be numbers in [0, 1]")
                checked[name] = value
            bounds.append(checked)
        low, high = bounds
        if any(low[name] > high[name] for name in self._model_names):
            raise ValueError("each lower weight bound must not exceed its upper bound")
        if fsum(low.values()) > 1.0 + 1e-9 or fsum(high.values()) < 1.0 - 1e-9:
            raise ValueError("no weights within these bounds sum to 1")
        return low, high

    def information_gain(
        self, node_id: str, prior: float, observations: Mapping[str, bool]
    ) -> InformationGain:
        """Expected information from observing ``node_id`` next; weights required.

        A report can be informative about which graph is true without being
        informative about the claim, and the reverse. Keeping the two numbers
        apart lets a study of provenance be valued separately from a decision
        about the claim; the planner values only the latter.
        """
        if not isinstance(node_id, str) or node_id not in self._node_ids:
            raise ValueError(f"unknown node: {node_id}")
        weights = self._require_model_weights()
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        if node_id in checked:
            self.model_averaged_posterior(prior, checked)   # rejects impossible data
            return InformationGain(0.0, 0.0)
        log_prior = {True: log(prior) if prior > 0.0 else -inf,
                     False: log1p(-prior) if prior < 1.0 else -inf}
        joint: dict[tuple[str, bool, bool], float] = {}
        for name, model in self._models.items():
            if weights[name] == 0.0:
                continue
            for truth in (True, False):
                for outcome in (True, False):
                    extended = dict(checked)
                    extended[node_id] = outcome
                    term = model.log_evidence(extended, truth)
                    if term > -inf and log_prior[truth] > -inf:
                        joint[(name, truth, outcome)] = log(weights[name]) + log_prior[truth] + term
        normalizer = _log_sum_exp(list(joint.values()))
        if normalizer == -inf:
            raise ValueError("observations have zero probability under every weighted model")
        mass = {key: exp(value - normalizer) for key, value in joint.items()}
        return InformationGain(
            _mutual_information(mass, lambda key: key[0]),
            _mutual_information(mass, lambda key: key[1]),
        )

    def _require_model_weights(self) -> Mapping[str, float]:
        if self._model_weights is None:
            raise ValueError(
                "Bayesian model averaging requires explicit model_weights; "
                "use posterior_envelope for an unweighted model set"
            )
        return self._model_weights

    def _log_marginal_outcome(self, node_id: str, outcome: bool, truth: bool) -> float:
        weights = self._require_model_weights()
        terms = []
        for name, model in self._models.items():
            weight = weights[name]
            if weight == 0.0:
                continue
            log_likelihood = model.log_evidence({node_id: outcome}, truth)
            if log_likelihood > -inf:
                terms.append(log(weight) + log_likelihood)
        return _log_sum_exp(terms)

    def marginal_likelihoods(self, node_id: str) -> tuple[float, float]:
        """Model-averaged ``P(report=True | H)`` for both truth values."""
        if not isinstance(node_id, str) or node_id not in self._node_ids:
            raise ValueError(f"unknown node: {node_id}")
        true_given_true = self._log_marginal_outcome(node_id, True, True)
        true_given_false = self._log_marginal_outcome(node_id, True, False)
        return (
            exp(true_given_true) if true_given_true > -inf else 0.0,
            exp(true_given_false) if true_given_false > -inf else 0.0,
        )

    def independence_log_likelihood_ratio(
        self, observations: Mapping[str, bool]
    ) -> float:
        """Sum marginal report log-likelihood ratios, treating reports independent.

        Model uncertainty is integrated into each single-report likelihood;
        the comparator then deliberately ignores both provenance dependence
        and dependence induced by sharing one latent model across reports.
        """
        checked = self.validate_observations(observations)
        total = 0.0
        for node_id, outcome in checked.items():
            log_true = self._log_marginal_outcome(node_id, outcome, True)
            log_false = self._log_marginal_outcome(node_id, outcome, False)
            if log_true == -inf and log_false == -inf:
                raise ValueError("observed report has zero probability under every weighted model")
            if log_false == -inf:
                ratio = inf
            elif log_true == -inf:
                ratio = -inf
            else:
                ratio = log_true - log_false
            total += ratio
        if total != total:
            raise ValueError("contradictory certain reports under independence approximation")
        return total

    def independence_posterior(
        self, prior: float, observations: Mapping[str, bool]
    ) -> float:
        """Posterior under the model-averaged marginal independence comparator."""
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        log_true, log_false = self._independence_log_masses(prior, checked)
        return _posterior_from_log_masses(log_true, log_false)

    def _independence_log_masses(
        self, prior: float, observations: Mapping[str, bool]
    ) -> tuple[float, float]:
        """Unnormalized truth masses under the independent-report comparator."""
        log_true = log(prior) if prior > 0.0 else -inf
        log_false = log1p(-prior) if prior < 1.0 else -inf
        for node_id, outcome in observations.items():
            term_true = self._log_marginal_outcome(node_id, outcome, True)
            term_false = self._log_marginal_outcome(node_id, outcome, False)
            log_true = log_true + term_true if log_true > -inf and term_true > -inf else -inf
            log_false = log_false + term_false if log_false > -inf and term_false > -inf else -inf
        return log_true, log_false

    def independence_predictive_distribution(
        self, node_id: str, prior: float, observations: Mapping[str, bool]
    ) -> dict[bool, float]:
        """Predict under the model-averaged marginal independence comparator.

        The target is treated as conditionally independent of observed reports
        given truth, after integrating its own one-report likelihood over the
        prior model weights. Log masses are retained through normalization so
        a tiny candidate-model-averaged likelihood is not rounded to zero
        prematurely.
        """
        if not isinstance(node_id, str) or node_id not in self._node_ids:
            raise ValueError(f"unknown node: {node_id}")
        prior = _probability(prior, "prior")
        checked = self.validate_observations(observations)
        base_true, base_false = self._independence_log_masses(prior, checked)
        log_base = _log_sum_exp([base_true, base_false])
        if log_base == -inf:
            raise ValueError("observations have zero probability under the independence model")
        if node_id in checked:
            value = checked[node_id]
            return {False: float(not value), True: float(value)}

        log_masses: dict[bool, float] = {}
        for outcome in (False, True):
            likelihood_true = self._log_marginal_outcome(node_id, outcome, True)
            likelihood_false = self._log_marginal_outcome(node_id, outcome, False)
            log_masses[outcome] = _log_sum_exp([
                base_true + likelihood_true
                if base_true > -inf and likelihood_true > -inf else -inf,
                base_false + likelihood_false
                if base_false > -inf and likelihood_false > -inf else -inf,
            ])
        probabilities = {
            outcome: exp(mass - log_base) if mass > -inf else 0.0
            for outcome, mass in log_masses.items()
        }
        total = probabilities[False] + probabilities[True]
        if total <= 0.0:
            raise ValueError("observations have zero probability under the independence model")
        return {
            outcome: probability / total
            for outcome, probability in probabilities.items()
        }


def _greedy_weights(
    coefficients: Mapping[str, float], low: Mapping[str, float], high: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    """Maximize ``sum(c[m] * w[m])`` over ``low <= w <= high`` with ``sum(w) == 1``."""
    weights = dict(low)
    remaining = 1.0 - fsum(low.values())
    for name in sorted(coefficients, key=lambda key: -coefficients[key]):
        if remaining <= 0.0:
            break
        step = min(high[name] - low[name], remaining)
        weights[name] += step
        remaining -= step
    total = fsum(weights.values())
    weights = {name: value / total for name, value in weights.items()}
    return weights, fsum(coefficients[name] * weights[name] for name in weights)


def _max_ratio(
    scale: Mapping[str, float],
    values: Mapping[str, float],
    low: Mapping[str, float],
    high: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Maximize ``sum(w s v) / sum(w s)`` over the weight bounds (Dinkelbach, 1967)."""
    weights, _ = _greedy_weights(scale, low, high)

    def ratio(candidate: Mapping[str, float]) -> float | None:
        denominator = fsum(candidate[name] * scale[name] for name in candidate)
        if denominator <= 0.0:
            return None
        return fsum(candidate[name] * scale[name] * values[name] for name in candidate) / denominator

    level = ratio(weights)
    # Each step moves to a vertex with a strictly larger ratio, so the loop
    # ends after at most as many steps as the polytope has vertices.
    for _ in range(10_000):
        candidate, gain = _greedy_weights(
            {name: scale[name] * (values[name] - level) for name in scale}, low, high
        )
        new_level = ratio(candidate) if gain > 0.0 else None
        if new_level is None or new_level <= level:
            return level, weights
        level, weights = new_level, candidate
    raise RuntimeError("Dinkelbach iteration did not converge")


def _mutual_information(mass: Mapping[tuple[str, bool, bool], float], variable) -> float:
    """Mutual information in bits between the report outcome and ``variable(key)``."""
    joint: dict[tuple[object, bool], float] = {}
    left: dict[object, float] = {}
    right: dict[bool, float] = {}
    for key, probability in mass.items():
        pair = (variable(key), key[2])
        joint[pair] = joint.get(pair, 0.0) + probability
        left[pair[0]] = left.get(pair[0], 0.0) + probability
        right[pair[1]] = right.get(pair[1], 0.0) + probability
    information = fsum(
        probability * (log(probability) - log(left[a]) - log(right[b]))
        for (a, b), probability in joint.items()
        if probability > 0.0
    )
    return max(0.0, information / log(2.0))
