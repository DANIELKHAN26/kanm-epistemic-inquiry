"""Exact models of chained testimony about a binary hypothesis.

A set of informants each privately observes a noisy signal about a binary hypothesis H and states which value they believe. Two structures are covered:
- **Independent testimony:** each informant sees only their own signal.
- **Sequential testimony:** each informant also hears every earlier informant's statement before speaking. This is the classic information cascade.

A listener hears the statements, then observes a private signal of their own. The model is that of the balls-and-urns tasks in Whalen, Griffiths & Buchsbaum (2018) and Xie & Hayes (2022). Signals are symmetric with reliability ``q``; in those tasks q = 100/120.

Each case is built as a :class:`kanm.dependence.ProvenanceNetwork`:
- **Latent nodes:** every informant's signal.
- **Statement nodes:** each statement's parents are the informant's own signal and, in the sequential case, every earlier statement.
- **Statement probabilities:** they come from the informant's own *exact* posterior, which KANM computes on the network built so far. The informants are therefore fully Bayesian about one another.

Two response rules are supported:
- ``"match"`` (probability matching): an informant states a value with its posterior probability. This is the rule of Whalen et al. (2018) and Xie & Hayes (2022).
- ``"max"`` (maximising): an informant states the more probable value. An exact tie, within 1e-12, is resolved in favour of the informant's own signal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from random import Random
from typing import Literal

from .bayes import _probability
from .dependence import ProvenanceNetwork, ReportNode, symmetric_primary

Rule = Literal["match", "max"]
MAX_INFORMANTS = 10
_TIE = 1e-12


def _check(n_informants: int, reliability: float, rule: str, sequential: bool) -> None:
    if isinstance(n_informants, bool) or not isinstance(n_informants, int):
        raise ValueError("n_informants must be an integer")
    if not 1 <= n_informants <= MAX_INFORMANTS:
        raise ValueError(f"n_informants must lie in 1..{MAX_INFORMANTS}")
    if not isinstance(sequential, bool):
        raise ValueError("sequential must be a boolean")
    if (
        isinstance(reliability, bool)
        or not isinstance(reliability, (int, float))
        or not 0.0 < reliability < 1.0
    ):
        raise ValueError("reliability must lie strictly between 0 and 1")
    if rule not in ("match", "max"):
        raise ValueError("rule must be 'match' or 'max'")


@dataclass(frozen=True)
class UrnTaskOutcome:
    """One simulated balls-and-urns trial under the stated generative model.

    Signals and statements use the convention of :func:`informant_network`:
    ``True`` indicates that H is true, whichever urn was drawn, so a correct
    signal equals ``hypothesis_true``. ``listener_posterior``
    is the exact model posterior after hearing the statements and seeing the
    listener's own ball; it is a model quantity, not a simulated human rating.
    """

    hypothesis_true: bool
    informant_signals: tuple[bool, ...]
    statements: tuple[bool, ...]
    listener_signal: bool
    listener_posterior: float
    prior: float
    signal_reliability: float
    sequential: bool
    rule: Rule


def informant_network(n_informants: int, reliability: float, *, sequential: bool,
                      rule: Rule = "match", prior: float = 0.5) -> ProvenanceNetwork:
    """Network with latent signals ``signal{k}``, statements ``statement{k}`` and ``own``.

    A statement value of ``True`` means "H is true"; ``own`` is the listener's signal.
    ``prior`` is shared by informants and listener, matching common-prior models.
    """
    _check(n_informants, reliability, rule, sequential)
    prior = _probability(prior, "prior")
    network = ProvenanceNetwork()
    for k in range(n_informants):
        signal = f"signal{k}"
        network.add(symmetric_primary(signal, reliability))
        heard = [f"statement{j}" for j in range(k)] if sequential else []
        cpd = {}
        for values in itertools.product((True, False), repeat=1 + len(heard)):
            observed = {signal: values[0], **dict(zip(heard, values[1:]))}
            try:
                belief = network.posterior(prior, observed)
            except ValueError:
                # The earlier statements are impossible under the model (only under "max").
                # The entry is never reached, so any probability is correct; use 1/2.
                belief = None
            if belief is None:
                p_true = 0.5
            elif rule == "match":
                p_true = belief
            elif belief > 0.5 + _TIE:
                p_true = 1.0
            elif belief < 0.5 - _TIE:
                p_true = 0.0
            else:
                p_true = 1.0 if values[0] else 0.0
            for truth in (True, False):
                cpd[(truth, values)] = p_true
        network.add(ReportNode(f"statement{k}", (signal, *heard), cpd))
    network.add(symmetric_primary("own", reliability))
    return network


def simulate_urn_task(
    n_informants: int,
    *,
    majority_balls: int = 100,
    minority_balls: int = 20,
    prior: float = 0.5,
    sequential: bool = True,
    rule: Rule = "match",
    seed: int | str | bytes | bytearray | None = None,
) -> UrnTaskOutcome:
    """Draw one reproducible balls-and-urns trial from the testimony model.

    Each signal is an independent draw *with replacement* from the selected
    urn. The default 100:20 composition is the design reported by Xie and
    Hayes (2022), giving signal reliability 100/120. Informants make statements
    using the same exact posterior model as :func:`informant_network`; the
    listener then receives one private signal. Use different ball counts to
    study other urn designs.
    """
    if (
        isinstance(majority_balls, bool)
        or not isinstance(majority_balls, int)
        or majority_balls < 1
    ):
        raise ValueError("majority_balls must be a positive integer")
    if (
        isinstance(minority_balls, bool)
        or not isinstance(minority_balls, int)
        or minority_balls < 1
    ):
        raise ValueError("minority_balls must be a positive integer")
    if majority_balls < minority_balls:
        raise ValueError("majority_balls must be >= minority_balls")
    prior = _probability(prior, "prior")
    reliability = majority_balls / (majority_balls + minority_balls)
    network = informant_network(
        n_informants, reliability, sequential=sequential, rule=rule, prior=prior
    )
    rng = Random(seed)
    hypothesis_true = rng.random() < prior
    signals = tuple(
        hypothesis_true if rng.random() < reliability else not hypothesis_true
        for _ in range(n_informants)
    )
    statements = []
    for index, signal in enumerate(signals):
        node = network.nodes[f"statement{index}"]
        heard = tuple(statements) if sequential else ()
        p_true = node.cpd[(hypothesis_true, (signal, *heard))]
        statements.append(rng.random() < p_true)
    listener_signal = hypothesis_true if rng.random() < reliability else not hypothesis_true
    observations = {
        **{f"statement{index}": value for index, value in enumerate(statements)},
        "own": listener_signal,
    }
    listener_posterior = network.posterior(prior, observations)
    return UrnTaskOutcome(
        hypothesis_true=hypothesis_true,
        informant_signals=signals,
        statements=tuple(statements),
        listener_signal=listener_signal,
        listener_posterior=listener_posterior,
        prior=prior,
        signal_reliability=reliability,
        sequential=sequential,
        rule=rule,
    )


def trajectory(n_informants: int, reliability: float = 100 / 120, *, sequential: bool,
               rule: Rule = "match", inference: Literal["exact", "independence"] = "exact",
               prior: float = 0.5) -> tuple[float, ...]:
    """P(H) after each of ``n_informants`` statements for H, then after a contrary own signal.

    ``inference="independence"`` treats the observed statements as conditionally
    independent given H, each with its exact marginal likelihood
    (:meth:`ProvenanceNetwork.independence_posterior`).
    """
    if inference not in ("exact", "independence"):
        raise ValueError("inference must be 'exact' or 'independence'")
    prior = _probability(prior, "prior")
    network = informant_network(
        n_informants, reliability, sequential=sequential, rule=rule, prior=prior
    )
    evaluate = network.posterior if inference == "exact" else network.independence_posterior
    observed: dict[str, bool] = {}
    out = []
    for k in range(n_informants):
        observed[f"statement{k}"] = True
        out.append(evaluate(prior, observed))
    observed["own"] = False
    out.append(evaluate(prior, observed))
    return tuple(out)
