"""
Reasoning Agent v0 (Section 21.3).

INPUT:  { question: QuestionRecord }
OUTPUT: {
    framework_selected: enum,
    framework_rationale: string,
    kg_traversal_path: [{node_or_edge_type, id, display_name, source_id}],
    hypotheses: [HypothesisRecord],
    failure_opportunities: [{
        source_failure_evidence_ids: [uuid],
        limitation_extracted: string,
        technology_delta_finding: string,
        alternative_perspective: string,
        reframed_question: string
    }]
}

`kg_traversal_path` denormalization (Patch 5): each traversal step is
self-describing — the Reasoning Agent is the only agent with a Neo4j
dependency, so the Explainability Agent never needs a second Neo4j lookup.

Validation gate: every HypothesisRecord must have >=1 non-empty
supporting_evidence_ids before being returned; a hypothesis with zero
linked evidence is rejected at generation time (enforced by
hypothesis_generation.generate_hypothesis's hard gate).

No external API calls originate here — reasons only over already-populated
KG/Evidence Object state.
"""

import uuid
from typing import Optional

from reasoning.failure_pipeline import FailurePipeline
from reasoning.framework_classifier import FrameworkClassifier
from reasoning.hypothesis_generation import EmptyEvidenceListError, generate_hypothesis
from reasoning.kg_traversal import FRAMEWORK_EVIDENCE_TYPES, KGTraversal
from repositories.evidence_object_repository import EvidenceObjectRepository


class ReasoningAgent:
    """
    Integrates framework classification, KG traversal, evidence selection,
    hard-gated hypothesis generation, and the failure-intelligence pipeline
    into a single read-only reasoning pass over already-populated KG and
    Evidence Object state.
    """

    def __init__(
        self,
        kg_traversal: Optional[KGTraversal] = None,
        evidence_repository: Optional[EvidenceObjectRepository] = None,
        framework_classifier: Optional[FrameworkClassifier] = None,
        failure_pipeline: Optional[FailurePipeline] = None,
    ):
        self._kg_traversal = kg_traversal or KGTraversal()
        self._evidence_repository = evidence_repository
        self._framework_classifier = framework_classifier or FrameworkClassifier()
        self._failure_pipeline = failure_pipeline or FailurePipeline()

    def run(self, question: dict) -> dict:
        classification = self._classify(question)
        framework_selected = classification.framework_selected
        framework_rationale = classification.framework_rationale

        kg_traversal_path = self._traverse(framework_selected, question)

        supporting_evidence, conflicting_evidence = self._select_evidence(
            framework_selected, kg_traversal_path
        )

        hypotheses = self._generate_hypotheses(question, supporting_evidence)

        failure_opportunities = self._process_failures(conflicting_evidence, question)

        return {
            "framework_selected": framework_selected,
            "framework_rationale": framework_rationale,
            "kg_traversal_path": kg_traversal_path,
            "hypotheses": hypotheses,
            "failure_opportunities": failure_opportunities,
        }

    # -- internal steps ----------------------------------------------------

    def _classify(self, question: dict):
        question_text = question.get("question_text") or self._build_question_text(question)
        return self._framework_classifier.classify(question_text)

    def _build_question_text(self, question: dict) -> str:
        return (
            f"For disease {question.get('disease_kg_id')}, gene/target "
            f"{question.get('gene_target_kg_id')}, evaluate therapeutic "
            f"approach: {question.get('therapeutic_approach')}"
        )

    def _traverse(self, framework_selected: str, question: dict):
        disease_id = question.get("disease_kg_id")
        gene_id = question.get("gene_target_kg_id")

        if framework_selected == "PICO":
            return self._kg_traversal.traverse_pico(disease_id)
        if framework_selected == "MechanisticBiology":
            return self._kg_traversal.traverse_mechanistic_biology(gene_id)
        if framework_selected == "BradfordHill":
            return self._kg_traversal.traverse_bradford_hill(disease_id)
        if framework_selected == "TargetValidation":
            return self._kg_traversal.traverse_target_validation(gene_id)
        raise ValueError(f"Unknown framework_selected: {framework_selected}")

    def _select_evidence(self, framework_selected: str, kg_traversal_path):
        relevant_evidence_types = FRAMEWORK_EVIDENCE_TYPES.get(framework_selected, [])
        supporting_evidence = []
        conflicting_evidence = []
        seen_evidence_ids = set()

        if self._evidence_repository is None:
            return supporting_evidence, conflicting_evidence

        for step in kg_traversal_path:
            node_or_edge_type = step["node_or_edge_type"]
            target_type = "relationship" if node_or_edge_type.isupper() else "entity"

            found = self._evidence_repository.find_by_target(target_type, step["id"])
            for evidence in found:
                if evidence.get("evidence_type") not in relevant_evidence_types:
                    continue
                # An Evidence Object may be linked to multiple KG nodes/edges
                # traversed here; dedupe by evidence_id so the same object
                # isn't double-counted when it surfaces on more than one step.
                evidence_id = evidence.get("evidence_id")
                if evidence_id in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(evidence_id)
                if evidence.get("supports_or_conflicts") == "conflicts":
                    conflicting_evidence.append(evidence)
                elif evidence.get("supports_or_conflicts") == "supports":
                    supporting_evidence.append(evidence)

        return supporting_evidence, conflicting_evidence

    def _generate_hypotheses(self, question: dict, supporting_evidence: list) -> list:
        if not supporting_evidence:
            # Hard gate (Section 25): refuse to generate an unsupported hypothesis.
            return []

        supporting_evidence_ids = [e["evidence_id"] for e in supporting_evidence]
        statement = (
            f"{question.get('therapeutic_approach')} is a plausible approach "
            f"for {question.get('disease_kg_id')} given available evidence."
        )

        try:
            hypothesis = generate_hypothesis(
                question_id=question.get("question_id", uuid.uuid4()),
                statement=statement,
                supporting_evidence_ids=supporting_evidence_ids,
            )
        except EmptyEvidenceListError:
            return []

        return [hypothesis]

    def _process_failures(self, conflicting_evidence: list, question: dict) -> list:
        failure_opportunities = []
        question_context = self._build_question_text(question)

        for evidence in conflicting_evidence:
            reframed = self._failure_pipeline.process_failure(evidence, question_context)
            failure_opportunities.append(
                {
                    "source_failure_evidence_ids": [reframed.original_failure_evidence_id],
                    "limitation_extracted": reframed.limitation.limitation_description,
                    "technology_delta_finding": reframed.technology_delta.technology_delta_description,
                    "alternative_perspective": reframed.alternative_perspective.adjacent_approach,
                    "reframed_question": reframed.reframed_question_statement,
                }
            )

        return failure_opportunities
