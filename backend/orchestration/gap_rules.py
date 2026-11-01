"""
gap_rules.py (Section 20.1/21.5, Patch 10).

Cross-store referential integrity: `question.disease_kg_id` /
`gene_target_kg_id` store Neo4j node IDs as opaque text — Postgres cannot
enforce a foreign key against Neo4j. `kg_ids_resolve()` is an
application-layer structural existence check confirming these IDs resolve
in Neo4j before a Question row is persisted. This is a structural check,
not a biological judgment, and lives inside Orchestration.
"""

import os
from typing import Optional

from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


class StructuralExistenceError(Exception):
    """
    Raised when a disease_kg_id or gene_target_kg_id does not resolve to an
    existing node in Neo4j. Purely structural — never a statement about
    whether a biological claim is true.
    """


class GapRules:
    """
    Structural, non-biological existence checks used by Orchestration
    before persisting a Question row (Patch 10).
    """

    def __init__(self, driver=None):
        self._driver = driver or GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self._driver.close()

    def kg_ids_resolve(
        self, disease_kg_id: str, gene_target_kg_id: Optional[str] = None
    ) -> bool:
        """
        Confirms disease_kg_id (required) and gene_target_kg_id (optional)
        each resolve to an existing node in Neo4j. Returns True only if
        every supplied ID resolves; returns False (never raises) so
        Orchestration can decide how to halt (Section 21.5, step 3).
        """
        if not self._node_exists_by_id(disease_kg_id):
            return False

        if gene_target_kg_id is not None and not self._node_exists_by_id(gene_target_kg_id):
            return False

        return True

    def _node_exists_by_id(self, node_id: str) -> bool:
        query = "MATCH (n {id: $node_id}) RETURN count(n) AS c"
        with self._driver.session() as session:
            record = session.run(query, node_id=node_id).single()
            return bool(record and record["c"] > 0)
