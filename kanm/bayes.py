"""Bayesian update kernel for the binary Laputa model.

Reliability is represented by a normalized discrete distribution on [0, 1]
with one atom per equal-width cell, placed at the cell's exact conditional
mean under the Beta prior (never at 0 or 1). This approximates the continuous
trust density in the published equations; the point count is a
numerical-resolution parameter, not a psychological assumption. Once a source
has produced many more reports than there are grid cells, the posterior
concentrates in one or two cells and the grid resolution limits accuracy;
choose ``trust_points`` accordingly for long runs.

Weights are held as normalised *log*-weights. Linear weights of cells far from
the mode underflow to 0.0 after a few hundred consistent reports; if they were
stored that way, a later reversal of the evidence could never move mass back
to those cells and the trust estimate could end on the wrong side of 0.5 (a
v0.3 audit finding). The public ``weights`` tuple is a derived, display-only
view; every computation uses ``log_weights``.

Two update rules are provided for the *same* generative assumptions (one
binary proposition ``p``; each source has a fixed but unknown symmetric
reliability ``r``; reliabilities are a priori independent of ``p`` and of
each other; reports are conditionally independent given ``p`` and the
reliabilities):

* :func:`update_belief` with :meth:`TrustDistribution.observe` is the
  published Laputa rule. After each report it keeps only the marginal credence
  and the marginal trust density of each source. For a single report these
  marginals are exact; after repeated reports from the same source the
  discarded correlation between ``p`` and ``r`` makes the rule an
  approximation (a product-of-marginals projection).
* :class:`ExactJointState` keeps the exact joint posterior. Because the
  reliabilities stay conditionally independent given ``p``, the exact state
  needs only one conditional density per source and truth value, so its cost
  is linear in the number of sources.

Credences are handled in log-odds internally by the simulator so that
evidence accumulated beyond floating-point resolution of a probability is
not silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, inf, isfinite, isinf, lgamma, log, log1p, pi
from typing import Hashable, Iterable

from .trust import TrustState

MAX_TRUST_POINTS = 100_001


def _probability(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite probability in [0, 1]") from error
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    return value


def _points(points: int, minimum: int) -> int:
    if isinstance(points, bool) or not isinstance(points, int):
        raise ValueError("points must be an integer")
    if not minimum <= points <= MAX_TRUST_POINTS:
        raise ValueError(f"points must lie in [{minimum}, {MAX_TRUST_POINTS}]")
    return points


def logit(probability: float) -> float:
    """Log-odds of a probability; returns +/-inf for 1 and 0."""
    q = _probability(probability, "probability")
    if q == 0.0:
        return -inf
    if q == 1.0:
        return inf
    return log(q) - log1p(-q)


def expit(log_odds: float) -> float:
    """Numerically stable inverse of :func:`logit` (accepts +/-inf)."""
    if log_odds != log_odds:  # NaN
        raise ValueError("log-odds is NaN")
    if log_odds >= 0.0:
        return 1.0 / (1.0 + exp(-log_odds)) if not isinf(log_odds) else 1.0
    if isinf(log_odds):
        return 0.0
    z = exp(log_odds)
    return z / (1.0 + z)


def _log_sigmoid(x: float) -> float:
    """log(expit(x)) without overflow; -inf at x = -inf."""
    if isinf(x):
        return 0.0 if x > 0 else -inf
    if x >= 0.0:
        return -log1p(exp(-x))
    return x - log1p(exp(x))


_CF_MAX_ITER = 200_000
_CF_EPS = 1e-15
_CF_TINY = 1e-300
_MIN_BETA_SHAPE = 1e-12
_MAX_BETA_SHAPE = 1e6


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
    h = d
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
            c = 1.0 + aa / c
            c = c if abs(c) > _CF_TINY else _CF_TINY
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ValueError("incomplete beta continued fraction did not converge")


def _stirling_correction(x: float) -> float:
    """Remainder after the leading terms of Stirling's log-gamma formula."""
    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    return inverse * (
        1.0 / 12.0
        + inverse_squared
        * (
            -1.0 / 360.0
            + inverse_squared
            * (
                1.0 / 1260.0
                + inverse_squared * (-1.0 / 1680.0 + inverse_squared / 1188.0)
            )
        )
    )


def _log_beta(a: float, b: float) -> float:
    """Compute ``log(Beta(a, b))`` without subtracting huge log-gammas.

    Direct ``lgamma(a) + lgamma(b) - lgamma(a + b)`` loses precision when
    one shape is large and can overflow even when the log-beta value is
    representable. The mixed-shape branch evaluates the log-gamma difference
    with ``log1p``; the large-shape branch uses a Stirling expansion.
    """
    total = a + b
    if not isfinite(total):
        raise ValueError("Beta shape parameters are too large for double-precision evaluation")
    small, large = min(a, b), max(a, b)
    if large < 100.0:
        return lgamma(a) + lgamma(b) - lgamma(total)
    if small >= 100.0:
        return (
            (a - 0.5) * log(a / total)
            + (b - 0.5) * log(b / total)
            - 0.5 * log(total)
            + 0.5 * log(2.0 * pi)
            + _stirling_correction(a)
            + _stirling_correction(b)
            - _stirling_correction(total)
        )

    ratio = small / large
    log_gamma_ratio = (
        small * log(large)
        + (large + small - 0.5) * log1p(ratio)
        - small
        + _stirling_correction(total)
        - _stirling_correction(large)
    )
    return lgamma(small) - log_gamma_ratio


def _beta_tail(a: float, b: float, x: float, log_front: float) -> float:
    """Evaluate one incomplete-beta tail from its continued fraction."""
    fraction = _beta_continued_fraction(a, b, x)
    if not isfinite(fraction) or fraction <= 0.0:
        raise ValueError("incomplete beta continued fraction returned an invalid value")
    log_tail = log_front + log(fraction) - log(a)
    if not isfinite(log_tail):
        if log_tail == -inf:
            return 0.0
        raise ValueError("incomplete beta tail is not representable")
    if log_tail > 0.0:
        if log_tail <= 1e-12:
            return 1.0
        raise ValueError("incomplete beta tail exceeds one; numerical evaluation is unstable")
    return exp(log_tail)


def _beta_cdf_pair(a: float, b: float, x: float) -> tuple[float, float]:
    """Return ``(I_x(a, b), 1 - I_x(a, b))`` from the numerically smaller side.

    Interior CDF evaluation is supported for each shape in
    ``[_MIN_BETA_SHAPE, _MAX_BETA_SHAPE]``. Exact endpoints and the symmetric
    midpoint are returned directly for any finite positive shapes.
    """
    converted = []
    for value in (a, b, x):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Beta shapes and x must be finite real numbers")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError("Beta shapes and x must be finite real numbers") from error
        if not isfinite(value):
            raise ValueError("Beta shapes and x must be finite real numbers")
        converted.append(value)
    a, b, x = converted
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Beta shape parameters must be > 0")
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return 0.0, 1.0
    if x == 1.0:
        return 1.0, 0.0
    if a == b and x == 0.5:
        return 0.5, 0.5
    if min(a, b) < _MIN_BETA_SHAPE or max(a, b) > _MAX_BETA_SHAPE:
        raise ValueError(
            "interior incomplete beta evaluation supports shape parameters "
            f"in [{_MIN_BETA_SHAPE:g}, {_MAX_BETA_SHAPE:g}]"
        )

    total = a + b
    if not isfinite(total):
        raise ValueError("Beta shape parameters are too large for double-precision evaluation")
    log_front = -_log_beta(a, b) + a * log(x) + b * log1p(-x)
    lower_side = x < (a + 1.0) / (total + 2.0)
    first = (a, b, x) if lower_side else (b, a, 1.0 - x)
    try:
        tail = _beta_tail(*first, log_front)
        if tail > 1.0 - 1e-8:
            # Near-one tails lose relative precision when represented as
            # 1 - epsilon. Try evaluating epsilon directly, while retaining
            # the standard branch if its complement is not computable.
            second = (b, a, 1.0 - x) if lower_side else (a, b, x)
            try:
                alternate = _beta_tail(*second, log_front)
            except ValueError:
                alternate = None
            if alternate is not None and alternate <= 0.5:
                return _tail_pair(alternate, lower_tail=not lower_side)
    except ValueError as first_error:
        second = (b, a, 1.0 - x) if lower_side else (a, b, x)
        try:
            alternate = _beta_tail(*second, log_front)
        except ValueError as second_error:
            raise ValueError(
                f"incomplete beta evaluation failed for shapes ({a}, {b}) at x={x}"
            ) from second_error
        if alternate > 0.5:
            raise ValueError(
                f"incomplete beta evaluation failed for shapes ({a}, {b}) at x={x}"
            ) from first_error
        return _tail_pair(alternate, lower_tail=not lower_side)

    return _tail_pair(tail, lower_tail=lower_side)


def _tail_pair(tail: float, *, lower_tail: bool) -> tuple[float, float]:
    """Return ``(lower, upper)`` keeping the directly evaluated tail unrounded.

    Writing the evaluated tail as ``1 - (1 - tail)`` would flush any tail below
    about 1e-16 to zero, and a zero cell mass is an impossible reliability.
    """
    return (tail, 1.0 - tail) if lower_tail else (1.0 - tail, tail)


def _beta_cell_masses(a: float, b: float, cells: int) -> list[float]:
    """Exact Beta(a, b) probability of each cell ``[k/cells, (k+1)/cells]``."""
    pairs = [_beta_cdf_pair(a, b, k / cells) for k in range(cells + 1)]
    switch = a / (a + b)
    masses = []
    for k in range(cells):
        upper_edge = (k + 1) / cells
        if upper_edge <= switch:
            mass = pairs[k + 1][0] - pairs[k][0]
        else:
            mass = pairs[k][1] - pairs[k + 1][1]
        if mass < -1e-12:
            raise ValueError(
                "incomplete beta CDFs are not monotone at the requested grid resolution"
            )
        masses.append(max(0.0, mass))
    return masses


def _log_sum_exp(values: Iterable[float]) -> float:
    values = list(values)
    peak = max(values)
    if peak == -inf:
        return -inf
    return peak + log(sum(exp(v - peak) for v in values))


def _log_or_minus_inf(x: float) -> float:
    return log(x) if x > 0.0 else -inf


def _validated_grid(grid: tuple[float, ...]) -> None:
    if len(grid) < 1:
        raise ValueError("grid and weights must be nonempty and of equal length")
    if any(not isfinite(x) or not 0.0 <= x <= 1.0 for x in grid):
        raise ValueError("reliability grid values must lie in [0, 1]")
    if any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError("reliability grid must be strictly increasing")


@dataclass(frozen=True)
class TrustDistribution:
    """Discrete approximation to an agent's credence density over reliability.

    Constructors place one atom in each of ``points`` equal cells of [0, 1]
    and never at 0 or 1, so a finite run of reports can never make a source
    certainly reliable or certainly unreliable. Hand-built distributions may
    still use any strictly increasing grid in [0, 1].

    ``weights`` are the normalised linear weights (for inspection); the
    authoritative state is ``log_weights``, which cannot underflow.
    """

    grid: tuple[float, ...]
    weights: tuple[float, ...]
    log_weights: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        grid, weights = tuple(self.grid), tuple(self.weights)
        if len(grid) != len(weights):
            raise ValueError("grid and weights must be nonempty and of equal length")
        _validated_grid(grid)
        if any(not isfinite(w) or w < 0.0 for w in weights):
            raise ValueError("trust weights must be finite and nonnegative")
        peak = max(weights)
        if peak <= 0.0:
            raise ValueError("trust weights must have positive total mass")
        # Rescale by the largest weight before taking logs, so that huge but
        # finite weights cannot overflow the total.
        self._set_state(grid, tuple(_log_or_minus_inf(w / peak) for w in weights))

    def _set_state(self, grid: tuple[float, ...], log_weights: tuple[float, ...]) -> None:
        peak = max(log_weights)
        if peak == -inf or not isfinite(peak):
            raise ValueError("trust weights must have positive total mass")
        # Subtract the peak before adding log(sum): avoids cancellation when
        # the unnormalised log-weights are large in magnitude.
        shifted = tuple(lw - peak for lw in log_weights)
        log_sum = log(sum(exp(lw) for lw in shifted))
        normalised = tuple(lw - log_sum for lw in shifted)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "log_weights", normalised)
        object.__setattr__(self, "weights", tuple(exp(lw) for lw in normalised))

    @classmethod
    def from_log_weights(
        cls, grid: Iterable[float], log_weights: Iterable[float]
    ) -> "TrustDistribution":
        """Construct from unnormalised log-weights (``-inf`` means zero mass)."""
        grid, log_weights = tuple(grid), tuple(log_weights)
        if len(grid) != len(log_weights):
            raise ValueError("grid and weights must be nonempty and of equal length")
        _validated_grid(grid)
        if any(lw != lw or lw == inf for lw in log_weights):
            raise ValueError("log-weights must be finite or -inf")
        obj = object.__new__(cls)
        obj._set_state(grid, log_weights)
        return obj

    @classmethod
    def uniform(cls, points: int = 1001) -> "TrustDistribution":
        """Uniform density: equal mass at the midpoint of each cell."""
        points = _points(points, 2)
        grid = tuple((i + 0.5) / points for i in range(points))
        return cls(grid, tuple(1.0 for _ in grid))

    @classmethod
    def beta_prior(
        cls, mean: float = 0.6, concentration: float = 8.0, points: int = 1001
    ) -> "TrustDistribution":
        """Moment-preserving discretization of Beta(mean*c, (1-mean)*c).

        Cell ``k`` receives its exact Beta probability, placed at the exact
        conditional mean of the density within the cell. The discrete
        distribution therefore has exactly the requested mean (up to
        floating-point error) within the supported shape range, including the U- and
        J-shaped densities that arise when ``mean * c < 1`` or
        ``(1 - mean) * c < 1``. Interior CDF evaluation requires both shapes,
        including ``alpha + 1`` for the first moment, to lie in
        ``[1e-12, 1e6]``; unsupported implied shapes raise ``ValueError``.
        """
        mean = _probability(mean, "mean")
        if mean in (0.0, 1.0):
            raise ValueError(
                "mean must lie strictly between 0 and 1; certain reliability makes "
                "contrary reports impossible (use a high concentration instead)"
            )
        if isinstance(concentration, bool) or not isinstance(concentration, (int, float)):
            raise ValueError("concentration must be finite and > 0")
        try:
            concentration = float(concentration)
        except OverflowError as error:
            raise ValueError("concentration must be finite and > 0") from error
        if not isfinite(concentration) or concentration <= 0.0:
            raise ValueError("concentration must be finite and > 0")
        points = _points(points, 2)
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        if (
            min(alpha, beta) < _MIN_BETA_SHAPE
            or max(alpha + 1.0, beta) > _MAX_BETA_SHAPE
        ):
            raise ValueError(
                "mean and concentration imply Beta shapes outside the supported "
                f"range [{_MIN_BETA_SHAPE:g}, {_MAX_BETA_SHAPE:g}]"
            )
        masses = _beta_cell_masses(alpha, beta, points)
        # Int_cell x f(x; a, b) dx = mean * P_cell(Beta(a + 1, b)).
        first_moments = _beta_cell_masses(alpha + 1.0, beta, points)
        grid = []
        for k, (mass, moment) in enumerate(zip(masses, first_moments)):
            low, high = k / points, (k + 1) / points
            location = mean * moment / mass if mass > 0.0 else (low + high) / 2.0
            grid.append(min(max(location, low), high))
        # Guard strict monotonicity if two locations coincide at a shared edge.
        for k in range(1, points):
            if grid[k] <= grid[k - 1]:
                grid[k] = (k + 0.5) / points if grid[k - 1] < (k + 0.5) / points else grid[k - 1] + 1e-15
        if sum(masses) <= 0.0:
            raise ValueError("Beta prior has no representable mass on this grid")
        return cls(tuple(grid), tuple(masses))

    @property
    def expected_reliability(self) -> float:
        return sum(x * weight for x, weight in zip(self.grid, self.weights))

    @property
    def expected_unreliability(self) -> float:
        """``E[1 - r]``, summed directly to avoid cancellation near 1."""
        return sum((1.0 - x) * weight for x, weight in zip(self.grid, self.weights))

    @property
    def is_point_mass(self) -> bool:
        return sum(1 for lw in self.log_weights if lw > -inf) == 1

    def _reweighted(self, log_likelihood: Iterable[float]) -> tuple["TrustDistribution", float]:
        unnormalised = tuple(lw + ll for lw, ll in zip(self.log_weights, log_likelihood))
        log_total = _log_sum_exp(unnormalised)
        if log_total == -inf:
            return self, -inf
        return TrustDistribution.from_log_weights(self.grid, unnormalised), log_total

    def observe(self, prior_belief: float, positive: bool) -> "TrustDistribution":
        """Bayesian trust update after a positive or negative report.

        Implements the published single-message density update. The caller
        should pass the recipient's credence immediately before this report.
        """
        q = _probability(prior_belief, "prior_belief")
        if positive:
            likelihood = [r * q + (1.0 - r) * (1.0 - q) for r in self.grid]
        else:
            likelihood = [(1.0 - r) * q + r * (1.0 - q) for r in self.grid]
        if min(likelihood) == max(likelihood) > 0.0:
            # A report whose likelihood is the same for every reliability (as at
            # credence 1/2) carries no information: return the prior unchanged.
            # Renormalising would add rounding noise that, accumulated, breaks
            # exact ties downstream.
            return self
        posterior, log_total = self._reweighted(_log_or_minus_inf(x) for x in likelihood)
        if log_total == -inf:
            raise ValueError("observation has zero probability under the trust prior")
        return posterior

    def condition(self, positive_given_true: bool) -> tuple["TrustDistribution", float]:
        """Condition on a report whose likelihood is ``r`` (or ``1-r``).

        Returns the normalized posterior and the log normalizer
        ``log E[r]`` (or ``log E[1-r]``). Used by :class:`ExactJointState`.
        """
        if positive_given_true:
            log_likelihood = (_log_or_minus_inf(r) for r in self.grid)
        else:
            log_likelihood = (log1p(-r) if r < 1.0 else -inf for r in self.grid)
        return self._reweighted(log_likelihood)


def log_likelihood_ratio(trust: TrustState, positive: bool) -> float:
    """Log-likelihood ratio of a report under the Laputa rule.

    Equal to ``log(mu / (1 - mu))`` for a positive report and its negative for
    a negative report, where ``mu`` is the expected reliability. ``1 - mu`` is
    summed directly rather than obtained by subtraction.
    """
    mu = trust.expected_reliability
    nu = trust.expected_unreliability
    if mu <= 0.0 and nu <= 0.0:
        raise ValueError("trust distribution has no mass")
    if nu <= 0.0:
        ratio = inf
    elif mu <= 0.0:
        ratio = -inf
    else:
        ratio = log(mu) - log(nu)
    return ratio if positive else -ratio


def update_belief(prior_belief: float, trust: TrustState, positive: bool) -> float:
    """Update credence in p after one positive/negative report from a source."""
    q = _probability(prior_belief, "prior_belief")
    mu = trust.expected_reliability
    nu = trust.expected_unreliability
    if positive:
        numerator = q * mu
        denominator = numerator + (1.0 - q) * nu
    else:
        numerator = q * nu
        denominator = numerator + (1.0 - q) * mu
    if denominator <= 0.0:
        raise ValueError("observation has zero probability under the current model")
    return numerator / denominator


def update_log_odds(log_odds: float, trust: TrustState, positive: bool) -> float:
    """Log-odds form of :func:`update_belief` (algebraically identical).

    Raises ``ValueError`` when the report is impossible under the model, for
    example a certain prior meeting a certain source with the opposite report.
    """
    ratio = log_likelihood_ratio(trust, positive)
    result = log_odds + ratio
    if result != result:
        raise ValueError("observation has zero probability under the current model")
    return result


def update_independent_reports(
    prior_belief: float,
    reports: Iterable[tuple[TrustDistribution, bool]],
) -> float:
    """Sequentially condition on reports under the source-independence model.

    Each tuple must represent a distinct conditionally independent source. The
    function does not mutate trust distributions; callers that need to retain
    and update source trust should call ``update_belief`` and ``observe`` for
    each source event in sequence.
    """
    belief = _probability(prior_belief, "prior_belief")
    for trust, positive in reports:
        belief = update_belief(belief, trust, positive)
    return belief


class ExactJointState:
    """Exact joint posterior over ``p`` and every source's reliability.

    State: ``log P(p, data)`` and ``log P(not p, data)`` (unnormalized), plus
    for each source ``i`` the conditional densities ``tau_i(r | p, data)`` and
    ``tau_i(r | not p, data)``. Under the stated assumptions the joint
    posterior is exactly

        P(w, r_1..r_k | data) = P(w | data) * prod_i tau_i(r_i | w, data),

    because each source's reports depend only on ``w`` and its own ``r_i``.
    Updates therefore commute: the posterior depends only on how many positive
    and negative reports each source has produced, not on their order.
    """

    def __init__(self, prior_belief: float) -> None:
        q = _probability(prior_belief, "prior_belief")
        self._log_true = log(q) if q > 0.0 else -inf
        self._log_false = log1p(-q) if q < 1.0 else -inf
        self._conditional: dict[Hashable, tuple[TrustDistribution, TrustDistribution]] = {}

    def add_source(self, source_id: Hashable, prior: TrustDistribution) -> None:
        if source_id is None or source_id == "":
            raise ValueError("source_id must be a nonempty hashable key")
        hash(source_id)
        if source_id in self._conditional:
            raise ValueError(f"duplicate source: {source_id}")
        if not isinstance(prior, TrustDistribution):
            raise TypeError("prior must be a TrustDistribution")
        self._conditional[source_id] = (prior, prior)

    @property
    def sources(self) -> tuple[Hashable, ...]:
        return tuple(self._conditional)

    @property
    def log_odds(self) -> float:
        if self._log_true == -inf and self._log_false == -inf:
            raise ValueError("posterior has zero total probability")
        if self._log_false == -inf:
            return inf
        if self._log_true == -inf:
            return -inf
        return self._log_true - self._log_false

    @property
    def belief(self) -> float:
        return expit(self.log_odds)

    def observe(self, source_id: Hashable, positive: bool) -> None:
        """Condition the joint posterior on one report from ``source_id``."""
        if source_id not in self._conditional:
            raise ValueError(f"unknown source: {source_id}")
        given_true, given_false = self._conditional[source_id]
        # Positive report: likelihood r if p, 1-r if not p. Negative: reversed.
        new_true, log_z_true = given_true.condition(positive_given_true=positive)
        new_false, log_z_false = given_false.condition(positive_given_true=not positive)
        log_true = self._log_true + log_z_true
        log_false = self._log_false + log_z_false
        if log_true == -inf and log_false == -inf:
            raise ValueError("observation has zero probability under the current model")
        self._log_true, self._log_false = log_true, log_false
        self._conditional[source_id] = (new_true, new_false)

    def trust(self, source_id: Hashable) -> TrustDistribution:
        """Marginal posterior density over the reliability of ``source_id``."""
        if source_id not in self._conditional:
            raise ValueError(f"unknown source: {source_id}")
        given_true, given_false = self._conditional[source_id]
        lo = self.log_odds
        log_q, log_not_q = _log_sigmoid(lo), _log_sigmoid(-lo)
        log_weights = tuple(
            _log_sum_exp((log_q + a, log_not_q + b))
            for a, b in zip(given_true.log_weights, given_false.log_weights)
        )
        return TrustDistribution.from_log_weights(given_true.grid, log_weights)

    def log_predictive(self, source_id: Hashable, positive: bool) -> float:
        """Log predictive probability of the next report from ``source_id``."""
        if source_id not in self._conditional:
            raise ValueError(f"unknown source: {source_id}")
        given_true, given_false = self._conditional[source_id]
        mu_true = given_true.expected_reliability
        mu_false = given_false.expected_reliability
        p_true = mu_true if positive else 1.0 - mu_true
        p_false = (1.0 - mu_false) if positive else mu_false
        lo = self.log_odds
        lt, lf = _log_sigmoid(lo), _log_sigmoid(-lo)
        terms = [t for t in (lt + _safe_log(p_true), lf + _safe_log(p_false)) if t != -inf]
        if not terms:
            return -inf
        peak = max(terms)
        return peak + log(sum(exp(t - peak) for t in terms))


def _safe_log(x: float) -> float:
    return log(x) if x > 0.0 else -inf
