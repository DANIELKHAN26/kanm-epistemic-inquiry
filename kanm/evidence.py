"""Structured evidence provenance and exact-scope audit utilities.

The ledger records provenance and repeated exposure. It deliberately does not
pretend that an ancestry graph alone supplies a correct likelihood for
partially dependent reports; inference over correlated sources remains a
separate, unimplemented research module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    question_id: str
    outcome: str
    source_id: str
    parents: tuple[str, ...] = ()
    scope: Mapping[str, str] | None = None
    kind: str = "observation"

    def __post_init__(self) -> None:
        for label in ("evidence_id", "question_id", "outcome", "source_id", "kind"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonempty string")
        if not isinstance(self.parents, tuple):
            raise ValueError("parents must be a tuple of evidence IDs")
        if any(not isinstance(parent, str) or not parent.strip() for parent in self.parents):
            raise ValueError("parent evidence IDs must be nonempty strings")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("parent evidence IDs must be unique")
        if self.scope is not None:
            if not isinstance(self.scope, Mapping):
                raise ValueError("scope must be a mapping of dimensions to values")
            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in self.scope.items()
            ):
                raise ValueError("scope keys and values must be nonempty strings")
            object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))


@dataclass(frozen=True)
class ScopeAssessment:
    status: str
    mismatched_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]


class EvidenceLedger:
    """Append-only evidence DAG and per-recipient first-seen tracking."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}
        self._items_view = MappingProxyType(self._items)
        self._seen_by_recipient: dict[str, set[str]] = {}

    @property
    def items(self) -> Mapping[str, EvidenceItem]:
        """Read-only view; use :meth:`add` to preserve DAG validation."""
        return self._items_view

    def add(self, item: EvidenceItem) -> None:
        if not isinstance(item, EvidenceItem):
            raise TypeError("item must be an EvidenceItem")
        if item.evidence_id in self._items:
            raise ValueError(f"duplicate evidence ID: {item.evidence_id}")
        for parent in item.parents:
            if parent not in self._items:
                raise ValueError(f"parent evidence must be added first: {parent}")
        self._items[item.evidence_id] = item

    def receive(self, recipient_id: str, evidence_id: str) -> bool:
        """Record exposure; return True only for a recipient's first exposure."""
        if not isinstance(recipient_id, str) or not recipient_id.strip():
            raise ValueError("recipient_id must be a nonempty string")
        if not isinstance(evidence_id, str) or evidence_id not in self.items:
            raise ValueError(f"unknown evidence ID: {evidence_id}")
        seen = self._seen_by_recipient.setdefault(recipient_id, set())
        if evidence_id in seen:
            return False
        seen.add(evidence_id)
        return True

    def unseen(self, recipient_id: str, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Return exact IDs not yet seen; shared ancestry is not collapsed."""
        if not isinstance(recipient_id, str) or not recipient_id.strip():
            raise ValueError("recipient_id must be a nonempty string")
        if not isinstance(evidence_ids, tuple):
            raise ValueError("evidence_ids must be a tuple")
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str) or evidence_id not in self.items:
                raise ValueError(f"unknown evidence ID: {evidence_id}")
        seen = self._seen_by_recipient.get(recipient_id, set())
        return tuple(
            dict.fromkeys(evidence_id for evidence_id in evidence_ids if evidence_id not in seen)
        )


def assess_scope(
    required_scope: Mapping[str, str], evidence: EvidenceItem
) -> ScopeAssessment:
    """Check exact categorical scope; no implicit transfer/generalization.

    Missing dimensions produce ``indeterminate`` unless another known field
    already proves a mismatch. The function never converts scope fit into a
    probability or a weight.
    """
    if not isinstance(required_scope, Mapping):
        raise ValueError("required_scope must be a mapping")
    if not isinstance(evidence, EvidenceItem):
        raise TypeError("evidence must be an EvidenceItem")
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in required_scope.items()
    ):
        raise ValueError("required scope keys and values must be nonempty strings")
    known_scope = evidence.scope or {}
    mismatched = tuple(
        sorted(
            key
            for key, value in required_scope.items()
            if key in known_scope and known_scope[key] != value
        )
    )
    unknown = tuple(sorted(key for key in required_scope if key not in known_scope))
    status = "out_of_scope" if mismatched else "indeterminate" if unknown else "in_scope"
    return ScopeAssessment(status, mismatched, unknown)
