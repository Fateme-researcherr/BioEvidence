"""
Week 3 completion-criterion validation (Section 30):

"Reasoning Agent v0 (read-only KG access) integrating the above, tested
against a documented failed hypothesis reframed into a new question."

Documented failed hypothesis used here: constitutive AAV-mediated MECP2
overexpression gene therapy produced dose-dependent toxicity in
Rett-syndrome mouse models (a real, published limitation of early MECP2
gene-replacement approaches) — reframed per the Section 10 pipeline into a
narrower, technology-adjusted question, never a "try again" recommendation.
"""

import uuid

import pytest

from agents.reasoning_agent import ReasoningAgent
from reasoning.failure_pipeline import (
    AlternativePerspective,
    LimitationExtraction,
    ReframedQuestion,
    RetryRecommendationError,
    TechnologyDeltaCheck,
)
from reasoning.framework_classifier import FrameworkClassificationResult
from reasoning.hypothesis_generation import Hypothesis

DISEASE_ID = "ORPHA:778"
GENE_ID = "HGNC:6990"

CONFLICTING_EVIDENCE = {
    "evidence_id": str(uuid.uuid4()),
    "claim_summary": (
        "Constitutive AAV-mediated MECP2 overexpression produced "
        "dose-dependent toxicity in wild-type and Mecp2-null mice."
    ),
    "source_database": "PubMed",
    "source_id": "PMID:19703396",
    "evidence_type": "functional_mechanistic",
    "study_design": "preclinical_in_vivo",
    "supports_or_conflicts": "conflicts",
    "claim_asserted_date": "2009-01-01T00:00:00Z",
}

SUPPORTING_EVIDENCE = {
    "evidence_id": str(uuid.uuid4()),
    "claim_summary": "MECP2 reactivation reverses Rett-like phenotypes in symptomatic mice.",
    "source_database": "PubMed",
    "source_id": "PMID:17334361",
    "evidence_type": "functional_mechanistic",
    "study_design": "preclinical_in_vivo",
    "supports_or_conflicts": "supports",
    "claim_asserted_date": "2007-01-01T00:00:00Z",
}


class _FixtureKGTraversal:
    def traverse_mechanistic_biology(self, gene_hgnc_id):
        return [
            {"node_or_edge_type": "Gene", "id": "gene-1", "display_name": "MECP2", "source_id": gene_hgnc_id},
            {"node_or_edge_type": "ENCODES", "id": "edge-1", "display_name": "ENCODES", "source_id": "edge-1"},
            {"node_or_edge_type": "Protein", "id": "protein-1", "display_name": "MECP2_HUMAN", "source_id": "P51608"},
        ]


class _FixtureEvidenceRepository:
    def find_by_target(self, target_type, target_id):
        # Return both the conflicting and supporting evidence regardless of
        # target, to keep the fixture simple — both are tied to the
        # MECP2/Rett gene-therapy question under test.
        return [CONFLICTING_EVIDENCE, SUPPORTING_EVIDENCE]


class _FixtureFrameworkClassifier:
    def classify(self, question_text):
        return FrameworkClassificationResult(
            framework_selected="MechanisticBiology",
            framework_rationale="Question concerns gene-to-phenotype mechanism, not a clinical comparator.",
            missing_information_flags=[],
        )


class _FixtureFailurePipeline:
    def process_failure(self, conflicting_evidence, question_context):
        return ReframedQuestion(
            original_failure_evidence_id=conflicting_evidence["evidence_id"],
            limitation=LimitationExtraction(
                limitation_category="design",
                limitation_description=(
                    "Constitutive, non-titratable AAV expression could not "
                    "keep MECP2 dosage within its narrow physiological range."
                ),
            ),
            technology_delta=TechnologyDeltaCheck(
                limiting_factor_changed=True,
                technology_delta_description=(
                    "Dose-titratable and autoregulated AAV promoter systems "
                    "have since been developed, allowing physiological "
                    "MECP2 expression control that was unavailable in 2009."
                ),
            ),
            alternative_perspective=AlternativePerspective(
                adjacent_approach="Autoregulated, dose-titratable MECP2 gene therapy",
                rationale_for_trying_now=(
                    "Newer autoregulatory promoter designs directly address "
                    "the dosage-control limitation identified in the "
                    "original constitutive-expression study."
                ),
            ),
            reframed_question_statement=(
                "Does autoregulated, dose-titratable AAV-MECP2 gene therapy "
                "avoid the dose-dependent toxicity seen with constitutive "
                "MECP2 overexpression in Rett syndrome models?"
            ),
        )


@pytest.fixture
def reasoning_agent():
    return ReasoningAgent(
        kg_traversal=_FixtureKGTraversal(),
        evidence_repository=_FixtureEvidenceRepository(),
        framework_classifier=_FixtureFrameworkClassifier(),
        failure_pipeline=_FixtureFailurePipeline(),
    )


@pytest.fixture
def question():
    return {
        "question_id": uuid.uuid4(),
        "disease_kg_id": DISEASE_ID,
        "gene_target_kg_id": GENE_ID,
        "therapeutic_approach": "AAV-mediated MECP2 gene replacement therapy",
    }


def test_documented_failed_hypothesis_is_reframed_not_retried(reasoning_agent, question):
    result = reasoning_agent.run(question)

    assert result["framework_selected"] == "MechanisticBiology"
    assert len(result["failure_opportunities"]) == 1

    opportunity = result["failure_opportunities"][0]
    assert opportunity["source_failure_evidence_ids"] == [CONFLICTING_EVIDENCE["evidence_id"]]
    assert "titratable" in opportunity["limitation_extracted"] or "titratable" in opportunity["technology_delta_finding"]
    assert "autoregulat" in opportunity["reframed_question"].lower()

    # Hard constraint (Section 10): never a bare retry recommendation.
    forbidden_phrases = ["try again", "retry", "repeat the same"]
    lowered = opportunity["reframed_question"].lower()
    assert not any(phrase in lowered for phrase in forbidden_phrases)


def test_kg_traversal_path_is_denormalized_self_describing(reasoning_agent, question):
    result = reasoning_agent.run(question)

    for step in result["kg_traversal_path"]:
        assert set(step.keys()) == {"node_or_edge_type", "id", "display_name", "source_id"}


def test_hypothesis_generated_only_from_supporting_evidence(reasoning_agent, question):
    result = reasoning_agent.run(question)

    assert len(result["hypotheses"]) == 1
    hypothesis = result["hypotheses"][0]
    assert isinstance(hypothesis, Hypothesis)
    assert SUPPORTING_EVIDENCE["evidence_id"] in [
        str(eid) for eid in hypothesis.supporting_evidence_ids
    ]
    # The conflicting evidence must never silently become "supporting".
    assert CONFLICTING_EVIDENCE["evidence_id"] not in [
        str(eid) for eid in hypothesis.supporting_evidence_ids
    ]


def test_retry_recommendation_would_be_rejected_by_failure_pipeline_guard():
    """
    Sanity check on the hard-constraint guard itself: a candidate that
    does recommend repeating the failed therapy must raise
    RetryRecommendationError rather than ever being returned.
    """
    from reasoning.failure_pipeline import _assert_not_retry_recommendation

    with pytest.raises(RetryRecommendationError):
        _assert_not_retry_recommendation(
            "Try again with the same constitutive AAV-MECP2 construct at a lower dose."
        )
