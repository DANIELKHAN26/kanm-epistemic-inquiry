"""Seeded directed-network generators for batch experiments.

Every generator returns a sorted list of ``(sender, receiver)`` pairs with no
self-loops and no duplicates, and is deterministic for a given seed. Use
:func:`edges_from_pairs` to attach communication parameters. Network structure
is an experimental factor: published Laputa findings about connectivity and
clustering are conditional on their generators and parameter ranges, so the
generator and its parameters must be reported with any result.
"""

from __future__ import annotations

from dataclasses import fields
import json
import random
from typing import Any, Mapping, Sequence

from .model import AgentSpec, EdgeSpec

Pair = tuple[str, str]


def agent_ids(n: int, prefix: str = "a") -> list[str]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")
    width = len(str(n - 1))
    return [f"{prefix}{i:0{width}d}" for i in range(n)]


def _ids(ids: Sequence[str]) -> list[str]:
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("ids must be a nonempty sequence of unique agent IDs")
    return ids


def _rng(seed: int, name: str) -> random.Random:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return random.Random(json.dumps([seed, "network", name]))


def _probability(p: float, label: str) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return float(p)


def _both_ways(pairs: set[Pair]) -> list[Pair]:
    return sorted(pairs | {(b, a) for a, b in pairs})


def complete(ids: Sequence[str]) -> list[Pair]:
    """Every ordered pair of distinct agents."""
    ids = _ids(ids)
    return sorted((a, b) for a in ids for b in ids if a != b)


def empty(ids: Sequence[str]) -> list[Pair]:
    _ids(ids)
    return []


def cycle(ids: Sequence[str], *, bidirectional: bool = True) -> list[Pair]:
    """Ring in the given ID order; one-way if ``bidirectional`` is False."""
    ids = _ids(ids)
    if len(ids) < 2:
        return []
    pairs = {(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))}
    pairs.discard((ids[0], ids[0]))
    return _both_ways(pairs) if bidirectional else sorted(pairs)


def star(ids: Sequence[str], *, center: str | None = None,
         direction: str = "both") -> list[Pair]:
    """Hub-and-spoke network. ``direction`` is 'out', 'in' or 'both'."""
    ids = _ids(ids)
    center = ids[0] if center is None else center
    if center not in ids:
        raise ValueError("center must be one of ids")
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be 'out', 'in' or 'both'")
    pairs = set()
    for other in ids:
        if other == center:
            continue
        if direction in ("out", "both"):
            pairs.add((center, other))
        if direction in ("in", "both"):
            pairs.add((other, center))
    return sorted(pairs)


def erdos_renyi(ids: Sequence[str], p: float, *, seed: int,
                directed: bool = True) -> list[Pair]:
    """G(n, p): each ordered (or unordered, if undirected) pair with chance p."""
    ids = _ids(ids)
    p = _probability(p, "p")
    rng = _rng(seed, "erdos_renyi")
    pairs: set[Pair] = set()
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if a == b or (not directed and j <= i):
                continue
            if rng.random() < p:
                pairs.add((a, b))
    return sorted(pairs) if directed else _both_ways(pairs)


def watts_strogatz(ids: Sequence[str], k: int, beta: float, *, seed: int) -> list[Pair]:
    """Small-world ring lattice with rewiring (Watts & Strogatz 1998).

    Each node links to its ``k // 2`` nearest neighbours on each side
    (``k`` even, ``2 <= k < n``); each lattice edge is rewired with
    probability ``beta`` to a uniformly chosen new endpoint that avoids
    self-loops and duplicates. Edges are reciprocal (undirected).
    """
    ids = _ids(ids)
    n = len(ids)
    if isinstance(k, bool) or not isinstance(k, int) or k < 2 or k % 2 or k >= n:
        raise ValueError("k must be an even integer with 2 <= k < n")
    beta = _probability(beta, "beta")
    rng = _rng(seed, "watts_strogatz")
    undirected: set[frozenset[str]] = set()
    for i in range(n):
        for offset in range(1, k // 2 + 1):
            undirected.add(frozenset((ids[i], ids[(i + offset) % n])))
    for offset in range(1, k // 2 + 1):
        for i in range(n):
            edge = frozenset((ids[i], ids[(i + offset) % n]))
            if edge not in undirected or rng.random() >= beta:
                continue
            source = ids[i]
            candidates = [
                other for other in ids
                if other != source and frozenset((source, other)) not in undirected
            ]
            if not candidates:
                continue
            undirected.remove(edge)
            undirected.add(frozenset((source, rng.choice(candidates))))
    pairs = {tuple(sorted(edge)) for edge in undirected}
    return _both_ways(pairs)  # type: ignore[arg-type]


def edges_from_pairs(pairs: Sequence[Pair], **edge_defaults: Any) -> list[EdgeSpec]:
    """Build :class:`EdgeSpec` objects with shared communication parameters."""
    allowed = {f.name for f in fields(EdgeSpec)} - {"sender", "receiver"}
    unknown = set(edge_defaults) - allowed
    if unknown:
        raise ValueError(f"unknown edge parameter(s): {sorted(unknown)}")
    return [EdgeSpec(sender, receiver, **edge_defaults) for sender, receiver in pairs]


def agents_from_ids(ids: Sequence[str], **agent_defaults: Any) -> list[AgentSpec]:
    allowed = {f.name for f in fields(AgentSpec)} - {"agent_id"}
    unknown = set(agent_defaults) - allowed
    if unknown:
        raise ValueError(f"unknown agent parameter(s): {sorted(unknown)}")
    return [AgentSpec(agent_id, **agent_defaults) for agent_id in _ids(ids)]


GENERATORS = {
    "complete": complete,
    "empty": empty,
    "cycle": cycle,
    "star": star,
    "erdos_renyi": erdos_renyi,
    "watts_strogatz": watts_strogatz,
}


def generate(spec: Mapping[str, Any], ids: Sequence[str]) -> list[Pair]:
    """Build pairs from a JSON-style spec such as ``{"type": "cycle"}``."""
    if not isinstance(spec, Mapping) or "type" not in spec:
        raise ValueError("network spec must be an object with a 'type'")
    kind = spec["type"]
    if kind not in GENERATORS:
        raise ValueError(f"unknown network type {kind!r}; choose from {sorted(GENERATORS)}")
    options = {key: value for key, value in spec.items() if key != "type"}
    try:
        return GENERATORS[kind](ids, **options)
    except TypeError as exc:
        raise ValueError(f"invalid options for network type {kind!r}: {exc}") from exc


def density(pairs: Sequence[Pair], n: int) -> float:
    """Directed density: edges / (n (n - 1))."""
    return 0.0 if n < 2 else len(set(pairs)) / (n * (n - 1))


def mean_clustering(pairs: Sequence[Pair], ids: Sequence[str]) -> float:
    """Average local clustering of the undirected skeleton (Watts & Strogatz)."""
    ids = _ids(ids)
    neighbours: dict[str, set[str]] = {i: set() for i in ids}
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)
    total = 0.0
    for node in ids:
        nbrs = sorted(neighbours[node])
        k = len(nbrs)
        if k < 2:
            continue
        links = sum(
            1 for i in range(k) for j in range(i + 1, k) if nbrs[j] in neighbours[nbrs[i]]
        )
        total += 2.0 * links / (k * (k - 1))
    return total / len(ids)
