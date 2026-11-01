"""
Orchestration (Workflow Controller) — Section 21.5.

FIXED SEQUENCE (no branching beyond explicit gap-detection rules):

1. receive_query(disease, gene, therapy)
2. IF disease/gene not in KG -> call Ingestion & Curation Agent
3. IF gap_rules.kg_ids_resolve() fails -> halt with structural error (Patch 10)
4. call Evidence Agent for any target lacking sufficient Evidence Objects
5. call Reasoning Agent
6. call Explainability Agent
7. GATE: if trace_complete == false -> halt, return structured gap notice
   else -> assemble final Intelligence Brief and return

No planning, no autonomous tool selection — every branch is a named,
pre-defined rule.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from agents.explainability_agent import GapNotice
from orchestration.gap_rules import GapRules, StructuralExistenceError


@dataclass
class StructuralError:
    reason: str
    disease_kg_id: Optional[str] = None
    gene_target_kg_id: Optional[str] = None


@dataclass
class IntelligenceBrief:
    question_id: str
    framework_selected: str
    framework_rationale: str
    report_sections: list
    trace_complete: bool


MIN_SUFFICIENT_EVIDENCE_OBJECTS = 1


class WorkflowController:
    """
    Runs the fixed Orchestration sequence. Every step below corresponds to
    a numbered step in the Section 21.5 sequence diagram; no additional
    branches exist beyond the explicit gap-detection rules already named
    there.
    """

    def __init__(
        self,
        ingestion_agent,
        evidence_agent,
        reasoning_agent,
        explainability_agent,
        evidence_repository,
        gap_rules: Optional[GapRules] = None,
        kg_lookup=None,
    ):
        self._ingestion_agent = ingestion_agent
        self._evidence_agent = evidence_agent
        self._reasoning_agent = reasoning_agent
        self._explainability_agent = explainability_agent
        self._evidence_repository = evidence_repository
        self._gap_rules = gap_rules or GapRules()
        self._kg_lookup = kg_lookup  # callable(disease, gene) -> (disease_kg_id, gene_target_kg_id) or None

    def receive_query(self, disease: str, gene: Optional[str], therapy: Optional[str]):
        """
        Step 1: receive_query(disease, gene, therapy). Returns either an
        IntelligenceBrief, a StructuralError, or a GapNotice — never a
        partial/degraded report.
        """
        # Step 2: IF disease/gene not in KG -> call Ingestion & Curation Agent.
        disease_kg_id, gene_target_kg_id = self._resolve_or_ingest(disease, gene, therapy)

        # Step 3: IF gap_rules.kg_ids_resolve() fails -> halt with structural error (Patch 10).
        if not self._gap_rules.kg_ids_resolve(disease_kg_id, gene_target_kg_id):
            return StructuralError(
                reason=(
                    "disease_kg_id/gene_target_kg_id does not resolve in Neo4j "
                    "(Patch 10 structural existence check) — halting before "
                    "the Question row is persisted."
                ),
                disease_kg_id=disease_kg_id,
                gene_target_kg_id=gene_target_kg_id,
            )

        question = {
            "question_id": uuid.uuid4(),
            "disease_kg_id": disease_kg_id,
            "gene_target_kg_id": gene_target_kg_id,
            "therapeutic_approach": therapy,
        }

        # Step 4: call Evidence Agent for any target lacking sufficient Evidence Objects.
        self._ensure_sufficient_evidence(question)

        # Step 5: call Reasoning Agent.
        reasoning_bundle = self._reasoning_agent.run(question)

        # Step 6: call Explainability Agent.
        report = self._explainability_agent.build_report(reasoning_bundle)

        # Step 7: GATE — provenance gate (Section 26), mandatory, not optional narration.
        if not report["trace_complete"]:
            gap_notice = self._explainability_agent.build_gap_notice(report)
            return gap_notice if gap_notice is not None else GapNotice(
                reason="trace_complete is false", failed_sections=[]
            )

        return IntelligenceBrief(
            question_id=str(question["question_id"]),
            framework_selected=reasoning_bundle["framework_selected"],
            framework_rationale=reasoning_bundle["framework_rationale"],
            report_sections=report["report_sections"],
            trace_complete=report["trace_complete"],
        )

    # -- step helpers --------------------------------------------------------

    def _resolve_or_ingest(self, disease: str, gene: Optional[str], therapy: Optional[str]):
        existing = self._kg_lookup(disease, gene) if self._kg_lookup else None
        if existing is not None:
            return existing

        ingestion_result = self._ingestion_agent.run(
            disease_query=disease, gene_filter=gene, therapy_filter=therapy
        )

        disease_kg_id = None
        gene_target_kg_id = None
        for node in ingestion_result.get("kg_nodes_written", []):
            if node.label == "Disease" and disease_kg_id is None:
                disease_kg_id = node.id
            if node.label == "Gene" and gene_target_kg_id is None:
                gene_target_kg_id = node.id

        return disease_kg_id, gene_target_kg_id

    def _ensure_sufficient_evidence(self, question: dict) -> None:
        targets = self._collect_evidence_targets(question)
        insufficient_targets = [
            target
            for target in targets
            if self._evidence_repository.count_supporting(target["target_type"], target["target_id"])
            < MIN_SUFFICIENT_EVIDENCE_OBJECTS
        ]

        if not insufficient_targets:
            return

        self._evidence_agent.run(
            targets=insufficient_targets,
            query_context={
                "disease": question["disease_kg_id"],
                "gene": question["gene_target_kg_id"],
                "therapy": question["therapeutic_approach"],
            },
        )

    def _collect_evidence_targets(self, question: dict) -> List[dict]:
        targets = []
        if question.get("disease_kg_id"):
            targets.append({"target_type": "entity", "target_id": question["disease_kg_id"]})
        if question.get("gene_target_kg_id"):
            targets.append({"target_type": "entity", "target_id": question["gene_target_kg_id"]})
        return targets
