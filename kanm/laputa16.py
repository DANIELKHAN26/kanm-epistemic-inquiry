"""The trust representation of the Laputa 1.6 source, for docking.

Laputa 1.6 (``Trust.h``/``Trust.cpp``) stores a trust function as density
values at the ``N + 1 = 49`` equally spaced nodes ``r_i = i / N`` of [0, 1]
(``N = TRUST_FUNCTION_RESOLUTION = 48``) and treats the density as piecewise
linear between nodes. This module re-states that behaviour from the archived
source; no code was copied.

* **Expectation** is the exact integral of ``r f(r)`` for the piecewise-linear
  density. Laputa does not divide by the total mass: the function "must be
  normalised" by the caller.
* **Update** after a report multiplies each node value by the report's
  likelihood *at that node*, ``r_i q + (1 - r_i)(1 - q)`` for a positive report
  (mirrored for a negative one), then rescales so that the trapezoid integral
  is 1. If every value becomes 0 the function is reset to uniform.
* **Presets** (``setValueSharpness``) evaluate an unnormalised Beta density at
  ``x_i = 0.001 + 0.998 i / N`` and normalise.
* **The default batch trust metadistribution** (``MetaDistribution::setDefault``
  with the distribution files shipped in ``build/data/distributions``) mixes
  ``Beta(1, 4)`` ("beta_lower") and ``Beta(4, 1)`` ("beta_upper") by
  interpolating their *parameters*: weight ``w ~ U(0, 1)`` gives
  ``Beta(1 + 3w, 4 - 3w)``, so the concentration is always 5 and the prior
  mean ``(1 + 3w) / 5`` is uniform on [0.2, 0.8]. Before a batch run Laputa
  precomputes the density at the nodes for ``w = j / N`` and linearly
  interpolates between neighbouring rows; the result is **not** renormalised
  before its first use. Measured over 4 801 weights, the trapezoid mass of
  these defaults lies in [0.968, 1.0004], so the expected trust used for the
  first report from a source is biased by up to -0.025 (mean absolute bias
  0.003); the bias is largest where a shape parameter is just above 1 and the
  density is 0 at an end node but rises steeply. The first update renormalises.

Differences that remain by design: Laputa stores node values and computes the
expectation in single precision (``float``), whereas this module uses double
precision. Docking therefore compares to a stated tolerance, not bit for bit.
"""

from __future__ import annotations

from math import exp, floor, isfinite, lgamma, log, log1p
from typing import Iterable, Sequence

RESOLUTION = 48
NODES = RESOLUTION + 1
_H = 1.0 / RESOLUTION
_R = tuple(i / RESOLUTION for i in range(NODES))
#: Allowed deviation of the trapezoid mass from 1 for user-supplied node values.
#: Laputa's own default trust functions deviate by up to 0.033 (see above).
MASS_TOLERANCE = 0.05


def _beta_pdf(x: float, a: float, b: float) -> float:
    """Beta density with the GSL convention at the end points."""
    if x < 0.0 or x > 1.0:
        return 0.0
    log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
    if x == 0.0 or x == 1.0:
        if a > 1.0 and b > 1.0:
            return 0.0
        return exp(log_norm) * (x ** (a - 1.0)) * ((1.0 - x) ** (b - 1.0))
    return exp(log_norm + (a - 1.0) * log(x) + (b - 1.0) * log1p(-x))


def _trapezoid_total(values: Sequence[float]) -> float:
    return 0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1]


class Laputa16Trust:
    """Immutable 49-node piecewise-linear trust function (Laputa 1.6)."""

    __slots__ = ("values",)

    def __init__(self, values: Iterable[float]) -> None:
        values = tuple(values)
        if len(values) != NODES:
            raise ValueError(f"a Laputa 1.6 trust function has exactly {NODES} node values")
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError("trust node values must be finite and >= 0")
        if not any(value > 0.0 for value in values):
            raise ValueError("trust node values must not all be zero")
        self.values = tuple(float(v) for v in values)

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_value_sharpness(cls, value: float, sharpness: float) -> "Laputa16Trust":
        """``TrustFunction::setValueSharpness`` (used by the GUI presets)."""
        for x, label in ((value, "value"), (sharpness, "sharpness")):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0.0 <= x <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1]")
        alpha = (0.1 + value * 99.9) ** sharpness
        beta = (0.1 + (1.0 - value) * 99.9) ** sharpness
        raw = [
            (x ** (alpha - 1.0)) * ((1.0 - x) ** (beta - 1.0))
            for x in (i / RESOLUTION * 0.998 + 0.001 for i in range(NODES))
        ]
        return cls._normalised(raw)

    @classmethod
    def default_batch(cls, weight: float) -> "Laputa16Trust":
        """Trust function drawn by the 1.6 default metadistribution at ``weight``.

        In a batch run ``weight`` is uniform on [0, 1]. The values are returned
        exactly as Laputa produces them, i.e. without renormalisation.
        """
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 <= weight <= 1.0:
            raise ValueError("weight must lie in [0, 1]")
        k = floor(weight * RESOLUTION)
        if k == RESOLUTION:
            k -= 1
        over = weight * RESOLUTION - k
        rows = [
            [_beta_pdf(r, 1.0 + 3.0 * d, 4.0 - 3.0 * d) for r in _R]
            for d in (k / RESOLUTION, (k + 1) / RESOLUTION)
        ]
        return cls(lo * (1.0 - over) + hi * over for lo, hi in zip(*rows))

    @classmethod
    def from_beta(cls, mean: float, concentration: float) -> "Laputa16Trust":
        """Beta(mean, concentration) density sampled at the nodes, normalised.

        A KANM convenience for specifying 1.6-style trust functions; it is not
        a Laputa feature. Both Beta shape parameters must be >= 1 so that the
        density is finite at the end-point nodes.
        """
        alpha, beta = mean * concentration, (1.0 - mean) * concentration
        if alpha < 1.0 or beta < 1.0:
            raise ValueError(
                "the laputa_1_6 trust model samples the Beta density at r = 0 and r = 1; "
                "mean * concentration and (1 - mean) * concentration must both be >= 1"
            )
        return cls._normalised(_beta_pdf(r, alpha, beta) for r in _R)

    @classmethod
    def _normalised(cls, raw: Iterable[float]) -> "Laputa16Trust":
        raw = list(raw)
        total = _trapezoid_total(raw)
        if total > 0.0:
            scale = RESOLUTION / total
            return cls(v * scale for v in raw)
        return cls([1.0] * NODES)

    # ------------------------------------------------------------- statistics
    @property
    def mass(self) -> float:
        """Trapezoid integral of the node values (1 for a normalised function)."""
        return _trapezoid_total(self.values) * _H

    @property
    def expected_reliability(self) -> float:
        """Exact ``integral r f(r) dr`` of the piecewise-linear density.

        Summing the per-segment integrals gives node weights ``h * r_i`` for
        interior nodes, ``h^2 / 6`` for ``r = 0`` and ``h (1/2 - h/6)`` for
        ``r = 1``. Like Laputa, no division by the mass is made.
        """
        v = self.values
        interior = sum(_R[i] * v[i] for i in range(1, RESOLUTION))
        return _H * (v[0] * _H / 6.0 + interior + v[RESOLUTION] * (0.5 - _H / 6.0))

    @property
    def expected_unreliability(self) -> float:
        """``1 - E[r]``, as Laputa uses it (``Amount::inverted``)."""
        return 1.0 - self.expected_reliability

    # ----------------------------------------------------------------- update
    def observe(self, prior_belief: float, positive: bool) -> "Laputa16Trust":
        """``TrustFunction::update(belief, pTrue)`` as a pure function."""
        q = float(prior_belief)
        if not 0.0 <= q <= 1.0:
            raise ValueError("prior_belief must lie in [0, 1]")
        if positive:
            factors = (r * q + (1.0 - r) * (1.0 - q) for r in _R)
        else:
            factors = ((1.0 - r) * q + r * (1.0 - q) for r in _R)
        updated = [value * factor for value, factor in zip(self.values, factors)]
        total = _trapezoid_total(updated)
        if total > 0.0:
            scale = RESOLUTION / total
            return Laputa16Trust(v * scale for v in updated)
        return Laputa16Trust([1.0] * NODES)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Laputa16Trust) and self.values == other.values

    def __hash__(self) -> int:
        return hash(self.values)

    def __repr__(self) -> str:
        return f"Laputa16Trust(E={self.expected_reliability:.6g}, mass={self.mass:.6g})"


def validated_node_values(values: Sequence[float], label: str) -> Laputa16Trust:
    """Build a trust function from user-supplied node values.

    The input must be nonnegative and its trapezoid mass must be within
    :data:`MASS_TOLERANCE` of 1. Accepted values are then normalized exactly.
    This treats small discretization and serialization error as numerical
    error while ensuring the values used as a probability density really have
    unit mass. The historical ``default_batch`` preset is kept separate and
    retains Laputa 1.6's original, slightly unnormalized tabulated values.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a list of {NODES} numbers")
    try:
        trust = Laputa16Trust(values)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None
    mass = trust.mass
    if abs(mass - 1.0) > MASS_TOLERANCE:
        raise ValueError(
            f"{label}: trapezoid mass {mass:.6g} is not within {MASS_TOLERANCE} of 1; "
            "Laputa assumes normalised trust functions"
        )
    if mass <= 0.0:
        raise ValueError(f"{label}: trapezoid mass must be positive")
    return Laputa16Trust(value / mass for value in trust.values)
