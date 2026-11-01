"""
KG traversal logic per framework (Section 25, corrected directions per
Patches 7/8; denormalization per Patch 5).

KG traversal strategy per framework:
- PICO — traverses (:Disease)<-[:STUDIES]-(:ClinicalTrial)-[:TESTS]->(:Therapy)
  (corrected direction, Patch 7 — no direct Disease->Therapy edge exists).
- Mechanistic Biology — traverses Gene->Protein->Pathway->Phenotype,
  prioritizing functional_mechanistic evidence.
- Bradford Hill — traverses Disease->Gene causal edges plus all linked
  Evidence Objects, checking temporality/consistency/dose-response signals
  across independent Evidence Objects.
- Target Validation — traverses (:Therapy)-[:TARGETS]->(:Protein)<-[:ENCODES]-(:Gene)
  (corrected direction, Patch 8 — TARGETS points Therapy->Protein, not the
  reverse).

`kg_traversal_path` denormalization requirement (Patch 5, Section 21.3):
each traversal step is self-describing (type, id, display name, source id)
— the Reasoning Agent is responsible for this denormalization at traversal
time, since it is the only agent with a Neo4j dependency. This confirms
the Explainability Agent never needs a second Neo4j lookup.

Evidence selection strategy: each framework module queries `evidence_link`
for all Evidence Objects tied to traversed nodes/edges, filtered by
`evidence_type` relevant to that framework (see FRAMEWORK_EVIDENCE_TYPES
below; the actual evidence_link query is performed by the caller via
EvidenceObjectRepository, since evidence storage lives in Postgres, not
Neo4j).
"""

import os
from typing import List, TypedDict

from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# Which evidence_type values are prioritized/relevant per framework
# (Section 25: "Evidence selection strategy").
FRAMEWORK_EVIDENCE_TYPES = {
    "PICO": ["clinical_trial_result"],
    "MechanisticBiology": ["functional_mechanistic", "genetic_association"],
    "BradfordHill": [
        "genetic_association",
        "functional_mechanistic",
        "case_report",
        "clinical_trial_result",
    ],
    "TargetValidation": ["target_disease_association_score", "functional_mechanistic"],
}


class TraversalStep(TypedDict):
    node_or_edge_type: str
    id: str
    display_name: str
    source_id: str


class KGTraversal:
    """
    Executes framework-specific Neo4j traversals and denormalizes each
    step into a self-describing TraversalStep at traversal time (Patch 5).
    """

    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self._driver.close()

    def traverse_pico(self, disease_orpha_id: str) -> List[TraversalStep]:
        """
        PICO — (:Disease)<-[:STUDIES]-(:ClinicalTrial)-[:TESTS]->(:Therapy)
        (Patch 7: no direct Disease->Therapy edge exists).
        """
        query = """
            MATCH (d:Disease {orpha_id: $orpha_id})<-[s:STUDIES]-(ct:ClinicalTrial)-[t:TESTS]->(th:Therapy)
            RETURN d, s, ct, t, th
        """
        with self._driver.session() as session:
            records = session.run(query, orpha_id=disease_orpha_id)
            steps: List[TraversalStep] = []
            for record in records:
                steps.append(self._node_step("Disease", record["d"], "orpha_id"))
                steps.append(self._edge_step("STUDIES", record["s"]))
                steps.append(self._node_step("ClinicalTrial", record["ct"], "nct_id"))
                steps.append(self._edge_step("TESTS", record["t"]))
                steps.append(self._node_step("Therapy", record["th"], "chembl_id"))
            return steps

    def traverse_mechanistic_biology(self, gene_hgnc_id: str) -> List[TraversalStep]:
        """
        Mechanistic Biology — Gene->Protein->Pathway->Phenotype,
        prioritizing functional_mechanistic evidence downstream.
        """
        query = """
            MATCH (g:Gene {hgnc_id: $hgnc_id})-[e:ENCODES]->(p:Protein)
                  -[pi:PARTICIPATES_IN]->(pw:Pathway)-[a:AFFECTS]->(ph:Phenotype)
            RETURN g, e, p, pi, pw, a, ph
        """
        with self._driver.session() as session:
            records = session.run(query, hgnc_id=gene_hgnc_id)
            steps: List[TraversalStep] = []
            for record in records:
                steps.append(self._node_step("Gene", record["g"], "hgnc_id"))
                steps.append(self._edge_step("ENCODES", record["e"]))
                steps.append(self._node_step("Protein", record["p"], "uniprot_id"))
                steps.append(self._edge_step("PARTICIPATES_IN", record["pi"]))
                steps.append(self._node_step("Pathway", record["pw"], "reactome_id"))
                steps.append(self._edge_step("AFFECTS", record["a"]))
                steps.append(self._node_step("Phenotype", record["ph"], "hpo_id"))
            return steps

    def traverse_bradford_hill(self, disease_orpha_id: str) -> List[TraversalStep]:
        """
        Bradford Hill — Disease->Gene causal edges plus all linked Evidence
        Objects (fetched separately from Postgres by the caller), checking
        temporality/consistency/dose-response signals across independent
        Evidence Objects.
        """
        query = """
            MATCH (d:Disease {orpha_id: $orpha_id})-[hcg:HAS_CAUSAL_GENE]->(g:Gene)
            RETURN d, hcg, g
        """
        with self._driver.session() as session:
            records = session.run(query, orpha_id=disease_orpha_id)
            steps: List[TraversalStep] = []
            for record in records:
                steps.append(self._node_step("Disease", record["d"], "orpha_id"))
                steps.append(self._edge_step("HAS_CAUSAL_GENE", record["hcg"]))
                steps.append(self._node_step("Gene", record["g"], "hgnc_id"))
            return steps

    def traverse_target_validation(self, gene_hgnc_id: str) -> List[TraversalStep]:
        """
        Target Validation — (:Therapy)-[:TARGETS]->(:Protein)<-[:ENCODES]-(:Gene)
        (Patch 8: TARGETS points Therapy->Protein, not the reverse).
        """
        query = """
            MATCH (g:Gene {hgnc_id: $hgnc_id})-[e:ENCODES]->(p:Protein)<-[tg:TARGETS]-(th:Therapy)
            RETURN g, e, p, tg, th
        """
        with self._driver.session() as session:
            records = session.run(query, hgnc_id=gene_hgnc_id)
            steps: List[TraversalStep] = []
            for record in records:
                steps.append(self._node_step("Gene", record["g"], "hgnc_id"))
                steps.append(self._edge_step("ENCODES", record["e"]))
                steps.append(self._node_step("Protein", record["p"], "uniprot_id"))
                steps.append(self._edge_step("TARGETS", record["tg"]))
                steps.append(self._node_step("Therapy", record["th"], "chembl_id"))
            return steps

    # -- denormalization helpers (Patch 5) --------------------------------

    def _node_step(self, label: str, node, source_id_property: str) -> TraversalStep:
        display_name = node.get("name") or node.get("symbol") or node.get("id")
        return {
            "node_or_edge_type": label,
            "id": node.get("id"),
            "display_name": display_name,
            "source_id": node.get(source_id_property),
        }

    def _edge_step(self, edge_type: str, relationship) -> TraversalStep:
        return {
            "node_or_edge_type": edge_type,
            "id": relationship.get("id"),
            "display_name": edge_type,
            "source_id": relationship.get("id"),
        }
