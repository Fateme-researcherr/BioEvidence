"""
Evidence Object storage (Section 8, Section 20.1) — stage 4/6 of the
Evidence Pipeline (Section 23):

    Evidence Object creation (schema validation)
            -> [Confidence calculation, Section 24]
    Evidence linking (evidence_link rows per target)   <- this module

Evidence Objects are independent, first-class entities: `evidence_object`
holds the object itself, and `evidence_link` is the polymorphic
association to whatever it supports or conflicts with (entity,
relationship, hypothesis, experiment, question, or claim). One Evidence
Object may link to multiple targets simultaneously via multiple
`evidence_link` rows, without duplicating the evidence object itself.

Storage note (Section 8): Postgres is used for MVP persistence as an
implementation convenience only; the association (`evidence_link`) is a
first-class link, never ownership in either direction.
"""

import json
import uuid
from typing import List, Optional

import psycopg2
import psycopg2.extras

from schemas.evidence_object_schema import EvidenceObject, EvidenceObjectValidationError


class EvidenceLinkTarget:
    """A single polymorphic link target for an Evidence Object."""

    def __init__(self, target_type: str, target_id: uuid.UUID):
        self.target_type = target_type
        self.target_id = target_id


class EvidenceObjectRepository:
    """
    Persists Evidence Objects to `evidence_object` and their polymorphic
    associations to `evidence_link`, per the migration order defined in
    Section 20.1 (0001_create_evidence_object, 0002_create_evidence_link).
    """

    def __init__(self, connection: psycopg2.extensions.connection):
        self._conn = connection

    def store(
        self,
        evidence: EvidenceObject,
        additional_targets: Optional[List[EvidenceLinkTarget]] = None,
    ) -> uuid.UUID:
        """
        Store a validated EvidenceObject and create its evidence_link
        row(s). The object's own `target_type`/`target_id` always produces
        one link row; `additional_targets` allows the same Evidence Object
        to simultaneously support/conflict with further targets (e.g. a
        single clinical trial result supporting a Result node while also
        conflicting with an older Hypothesis) without duplicating the
        evidence object itself.

        Raises EvidenceObjectValidationError if `evidence` is not a
        validated EvidenceObject instance — construction-time validation
        (pydantic) already enforces the write-path gates (Section 8.3
        source_database enforcement; Section 21.2 confidence_components
        completeness), so a valid instance reaching this method has
        already passed those gates.
        """
        if not isinstance(evidence, EvidenceObject):
            raise EvidenceObjectValidationError(
                "store() requires a validated EvidenceObject instance"
            )

        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evidence_object (
                        evidence_id, claim_summary, source_database, source_id,
                        retrieval_method, source_authority_level, primary_or_secondary,
                        evidence_type, study_design, supports_or_conflicts,
                        confidence_score, confidence_components,
                        retrieved_at, claim_asserted_date
                    ) VALUES (
                        %(evidence_id)s, %(claim_summary)s, %(source_database)s, %(source_id)s,
                        %(retrieval_method)s, %(source_authority_level)s, %(primary_or_secondary)s,
                        %(evidence_type)s, %(study_design)s, %(supports_or_conflicts)s,
                        %(confidence_score)s, %(confidence_components)s,
                        %(retrieved_at)s, %(claim_asserted_date)s
                    )
                    """,
                    {
                        "evidence_id": str(evidence.evidence_id),
                        "claim_summary": evidence.claim_summary,
                        "source_database": evidence.source_database.value,
                        "source_id": evidence.source_id,
                        "retrieval_method": evidence.retrieval_method,
                        "source_authority_level": evidence.source_authority_level.value,
                        "primary_or_secondary": evidence.primary_or_secondary.value,
                        "evidence_type": evidence.evidence_type.value,
                        "study_design": evidence.study_design.value,
                        "supports_or_conflicts": evidence.supports_or_conflicts.value,
                        "confidence_score": evidence.confidence_score,
                        "confidence_components": json.dumps(
                            evidence.confidence_components.model_dump()
                        ),
                        "retrieved_at": evidence.retrieved_at,
                        "claim_asserted_date": evidence.claim_asserted_date,
                    },
                )

                targets = [
                    EvidenceLinkTarget(
                        target_type=evidence.target_type.value, target_id=evidence.target_id
                    )
                ]
                targets.extend(additional_targets or [])

                for target in targets:
                    cur.execute(
                        """
                        INSERT INTO evidence_link (link_id, evidence_id, target_type, target_id)
                        VALUES (%(link_id)s, %(evidence_id)s, %(target_type)s, %(target_id)s)
                        """,
                        {
                            "link_id": str(uuid.uuid4()),
                            "evidence_id": str(evidence.evidence_id),
                            "target_type": target.target_type,
                            "target_id": str(target.target_id),
                        },
                    )

        return evidence.evidence_id

    def get_evidence_object(self, evidence_id: uuid.UUID) -> Optional[dict]:
        """
        Retrieve a single Evidence Object by its evidence_id. Used by the
        Explainability Agent (Section 21.4) to resolve each
        supporting_evidence_id / failure evidence id into a full
        evidence_trace entry.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM evidence_object WHERE evidence_id = %(evidence_id)s",
                {"evidence_id": str(evidence_id)},
            )
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def find_by_target(self, target_type: str, target_id: uuid.UUID) -> List[dict]:
        """
        Retrieve all Evidence Objects linked to a given target, joining
        through the polymorphic `evidence_link` table. Used by the
        Reasoning Agent (read-only, Section 21.3) and Explainability Agent
        (Section 21.4) to assemble evidence traces.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT eo.*
                FROM evidence_object eo
                JOIN evidence_link el ON el.evidence_id = eo.evidence_id
                WHERE el.target_type = %(target_type)s AND el.target_id = %(target_id)s
                ORDER BY eo.retrieved_at DESC
                """,
                {"target_type": target_type, "target_id": str(target_id)},
            )
            return [dict(row) for row in cur.fetchall()]

    def find_conflicting(self, target_type: str, target_id: uuid.UUID) -> List[dict]:
        """
        Retrieve only Evidence Objects tagged `conflicts` for a given
        target — used by the Confidence Engine's consistency/conflict
        component (Section 24) and by failure-opportunity detection
        (Section 21.3).
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT eo.*
                FROM evidence_object eo
                JOIN evidence_link el ON el.evidence_id = eo.evidence_id
                WHERE el.target_type = %(target_type)s
                  AND el.target_id = %(target_id)s
                  AND eo.supports_or_conflicts = 'conflicts'
                ORDER BY eo.retrieved_at DESC
                """,
                {"target_type": target_type, "target_id": str(target_id)},
            )
            return [dict(row) for row in cur.fetchall()]

    def count_supporting(self, target_type: str, target_id: uuid.UUID) -> int:
        """
        Count non-conflicting Evidence Objects for a given target — used
        as `replication_count` input to the Confidence Engine (Section 24:
        'whether independently replicated across multiple non-conflicting
        Evidence Objects').
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM evidence_object eo
                JOIN evidence_link el ON el.evidence_id = eo.evidence_id
                WHERE el.target_type = %(target_type)s
                  AND el.target_id = %(target_id)s
                  AND eo.supports_or_conflicts = 'supports'
                """,
                {"target_type": target_type, "target_id": str(target_id)},
            )
            return cur.fetchone()[0]
