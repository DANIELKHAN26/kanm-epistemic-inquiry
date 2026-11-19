"""Inferring others' private evidence from their graded judgments.

A re-implementation of the models of Fränken, Valentin, Lucas & Bramley (2024), *Cognition* 242, 105633, following their published code (github.com/janphilippfranken/spacefish, commit a1681e8).

**Belief:** an agent's belief about a binary state is Beta(1 + red, 1 + blue), where red and blue are (expected) counts of evidence for each side.

**Judgments** use a 7-point scale, −3 (blue) … +3 (red).
- **Belief to bins:** a belief gives a probability for each of the 7 equal bins of [0, 1]. These are Beta cell masses, computed exactly by :func:`kanm.bayes._beta_cell_masses`.
- **Bins to response:** responses follow a softmax with temperature ``tau`` over negative expected squared loss.

**Inverting a judgment sequence** (:func:`infer_evidence`): at each trial, the agent's latest evidence (red, blue or none, with ``evidence_prior``) is inferred from its judgment. The *expected* counts are accumulated; this is the authors' approximation, not exact inference over evidence histories.
- **An agent that watches others:** its belief at trial t also includes its parents' inferred counts from trial t − 1 (the authors' recursive, structure-sensitive filter, "Level-2").
- **Ignoring the structure:** treating every agent as independent gives the structure-blind "Level-1" inversion.

**Prior used here:** the saved inferences the authors used for their published predictions were produced with a uniform evidence prior and ``tau = 10``. Those are the defaults; ``tests/test_judgment_inversion.py`` checks them against the saved inferences.
"""

from __future__ import annotations

from math import exp, log
from typing import Mapping, Sequence

from .bayes import _beta_cell_masses

N_CHOICES = 7
SCALE = tuple(range(-3, 4))
UNIFORM_EVIDENCE = {"red": 1 / 3, "blue": 1 / 3, "none": 1 / 3}
_EVIDENCE = ("red", "blue", "none")


def bin_probabilities(red: float, blue: float) -> list[float]:
    """Probability of each of the 7 judgment bins under Beta(1 + red, 1 + blue)."""
    if red < 0 or blue < 0:
        raise ValueError("evidence counts must be non-negative")
    return _beta_cell_masses(1.0 + red, 1.0 + blue, N_CHOICES)


def response_probabilities(bins: Sequence[float], tau: float) -> list[float]:
    """Softmax with temperature ``tau`` over negative expected squared loss."""
    if not tau > 0:
        raise ValueError("tau must be positive")
    scores = [-sum((j - i) ** 2 * bins[j] for j in range(N_CHOICES)) for i in range(N_CHOICES)]
    peak = max(tau * s for s in scores)
    weights = [exp(tau * s - peak) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def expected_judgment(bins: Sequence[float]) -> float:
    """Mean judgment on the −3 … +3 scale under the bin probabilities."""
    return sum(k * p for k, p in zip(SCALE, bins))


def _index(judgment: int) -> int:
    if judgment not in SCALE:
        raise ValueError(f"judgment {judgment!r} is not on the −3 … +3 scale")
    return judgment + 3


def infer_evidence(judgments: Sequence[int], *, parents: Sequence[Sequence[tuple[float, float]]] = (),
                   evidence_prior: Mapping[str, float] = UNIFORM_EVIDENCE,
                   tau: float = 10.0) -> list[tuple[float, float]]:
    """Cumulative expected (red, blue) evidence behind an agent's judgments.

    ``parents`` holds the inferred cumulative counts of agents this agent watches; at trial t
    the agent is assumed to have incorporated their counts from trial t − 1.
    """
    if set(evidence_prior) != set(_EVIDENCE) or abs(sum(evidence_prior.values()) - 1) > 1e-12:
        raise ValueError("evidence_prior must give probabilities for red, blue and none")
    for parent in parents:
        if len(parent) < len(judgments):
            raise ValueError("each parent needs inferred counts for every trial")
    red = blue = 0.0
    out: list[tuple[float, float]] = []
    for t, judgment in enumerate(judgments):
        index = _index(judgment)
        if t == 0:
            base_red = base_blue = 0.0
        else:
            base_red = red + sum(p[t - 1][0] for p in parents)
            base_blue = blue + sum(p[t - 1][1] for p in parents)
        weights = {}
        for evidence in _EVIDENCE:
            a = base_red + (evidence == "red")
            b = base_blue + (evidence == "blue")
            weights[evidence] = (response_probabilities(bin_probabilities(a, b), tau)[index]
                                 * evidence_prior[evidence])
        total = sum(weights.values())
        red += weights["red"] / total
        blue += weights["blue"] / total
        out.append((red, blue))
    return out


def listener_bins(own_evidence: Sequence[str | None], others: Sequence[Sequence[tuple[float, float]]]
                  ) -> list[list[float]]:
    """Bin probabilities for a listener at each trial.

    The listener's own evidence is counted from the trial it arrives on. Each other agent's inferred counts are counted from the previous trial, because judgments made at trial t are seen only at t + 1.
    """
    out = []
    own_red = own_blue = 0
    for t, evidence in enumerate(own_evidence):
        if evidence not in ("red", "blue", None):
            raise ValueError("own evidence must be 'red', 'blue' or None")
        own_red += evidence == "red"
        own_blue += evidence == "blue"
        red = own_red + (sum(o[t - 1][0] for o in others) if t else 0.0)
        blue = own_blue + (sum(o[t - 1][1] for o in others) if t else 0.0)
        out.append(bin_probabilities(red, blue))
    return out


def judgment_log_likelihood(judgments: Sequence[int | None], bins: Sequence[Sequence[float]],
                            tau: float, sticky: float = 0.0) -> list[float | None]:
    """Log-probability of each judgment under the response model.

    With ``sticky > 0`` the prediction is mixed with repeating the previous judgment. This is the authors' "sticky" family; it applies from the second trial on, when that trial's previous judgment is observed. ``None`` judgments are skipped and return ``None``.
    """
    if not 0.0 <= sticky < 1.0:
        raise ValueError("sticky must lie in [0, 1)")
    out: list[float | None] = []
    for t, judgment in enumerate(judgments):
        if judgment is None:
            out.append(None)
            continue
        probs = response_probabilities(bins[t], tau)
        p = probs[_index(judgment)]
        previous = judgments[t - 1] if t else None
        if sticky and previous is not None:
            p = (1 - sticky) * p + sticky * (judgment == previous)
        out.append(log(p) if p > 0 else float("-inf"))
    return out
