"""
Hypothesis generation (Section 25).

"Hypothesis generation: a hypothesis is only constructed by binding a
statement to `supporting_evidence_ids`; generation refuses to emit a
hypothesis with an empty evidence list (hard gate). No unsupported
hypothesis generation is permitted anywhere in this engine."

This module enforces that gate structurally: there is no code path in
`generate_hypothesis` that returns a Hypothesis without at least one
`supporting_evidence_id` bound to it.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List
from uuid import UUID


class EmptyEvidenceListError(Exception):
    """
    Raised whenever hypothesis construction is attempted with zero
    supporting_evidence_ids. This is a hard gate — callers must not catch
    this and substitute a placeholder or default evidence id; the correct
    response is to not generate a hypothesis at all.
    """


@dataclass
class Hypothesis:
    hypothesis_id: UUID
    question_id: UUID
    statement: str
    supporting_evidence_ids: List[UUID]
    reframed_from_failure: bool = False
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def generate_hypothesis(
    question_id: UUID,
    statement: str,
    supporting_evidence_ids: List[UUID],
    reframed_from_failure: bool = False,
) -> Hypothesis:
    """
    Construct a Hypothesis by binding `statement` to
    `supporting_evidence_ids`.

    Hard gate (Section 25): raises EmptyEvidenceListError if
    `supporting_evidence_ids` is empty. There is no unsupported-hypothesis
    path — every Hypothesis returned by this function carries at least one
    real, linked Evidence Object id.
    """
    if not statement or not statement.strip():
        raise ValueError("statement must be non-empty")

    if not supporting_evidence_ids:
        raise EmptyEvidenceListError(
            "Refusing to generate a hypothesis with an empty evidence list "
            "(Section 25 hard gate) — no unsupported hypothesis generation "
            "is permitted."
        )

    return Hypothesis(
        hypothesis_id=uuid.uuid4(),
        question_id=question_id,
        statement=statement,
        supporting_evidence_ids=list(supporting_evidence_ids),
        reframed_from_failure=reframed_from_failure,
    )
