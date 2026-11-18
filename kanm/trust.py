"""Structural interface shared by KANM's trust representations."""

from __future__ import annotations

from typing import Protocol, Self


class TrustState(Protocol):
    """A trust model usable by the belief-update kernel.

    The protocol is intentionally small: implementations may store a discrete
    distribution, a sampled density, or another representation, provided they
    expose the same reliability summaries and return an updated state from
    ``observe``.
    """

    @property
    def expected_reliability(self) -> float: ...

    @property
    def expected_unreliability(self) -> float: ...

    def observe(self, prior_belief: float, positive: bool) -> Self: ...
