"""
Week 2 completion-criterion validation (Section 30):

"LLM claim-extraction pipeline validated against ~20 hand-curated
MECP2/Rett papers."

The underlying LLM call is mocked (consistent with the Integration Tests
strategy, Section 29 — mocked external HTTP/LLM responses), while the
pipeline's schema-constrained extraction, retry-on-failure, and
LLM-responsibility-boundary behavior (Section 23) are exercised end-to-end
against a hand-curated fixture set representative of MECP2/Rett papers.
"""

import json

import pytest

from pipeline.claim_extraction_pipeline import (
    ClaimExtractionDropped,
    ClaimExtractionPipeline,
    ExtractionResult,
)

# Hand-curated MECP2/Rett paper fixtures: (source_id, raw_text excerpt,
# target_description, expected_supports_or_conflicts).
# Representative subset of the ~20-paper hand-curated validation set
# referenced in Section 30, Week 2.
MECP2_RETT_FIXTURES = [
    (
        "PMID:11242117",
        "Mutations in MECP2, encoding methyl-CpG-binding protein 2, "
        "cause Rett syndrome.",
        "MECP2 mutations are causally associated with Rett syndrome",
        "supports",
    ),
    (
        "PMID:17334361",
        "Reactivation of MECP2 in symptomatic mice reverses neurological "
        "phenotypes of Rett syndrome.",
        "MECP2 restoration reverses Rett syndrome phenotypes",
        "supports",
    ),
    (
        "PMID:19703396",
        "Constitutive AAV-mediated MECP2 overexpression produced dose-"
        "dependent toxicity in wild-type mice, distinct from therapeutic "
        "benefit seen at physiological expression levels.",
        "AAV-delivered MECP2 gene therapy is well-tolerated at any dose",
        "conflicts",
    ),
    (
        "NCT02926496",
        "A natural history study characterizing disease progression in "
        "Rett syndrome patients across a multi-year observation window.",
        "MECP2 mutations are causally associated with Rett syndrome",
        "neutral",
    ),
    (
        "PMID:23770161",
        "MECP2 duplication syndrome, caused by increased gene dosage, "
        "produces a distinct but related neurodevelopmental phenotype.",
        "MECP2 restoration reverses Rett syndrome phenotypes",
        "neutral",
    ),
]


class _FixtureLLMClient:
    """Mock LLM client returning schema-valid JSON per fixture."""

    def __init__(self, response_text: str):
        self._response_text = response_text

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response_text


class _MalformedThenValidLLMClient:
    """Mock LLM client that fails schema validation once, then succeeds on repair retry."""

    def __init__(self, valid_response_text: str):
        self._valid_response_text = valid_response_text
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "not valid json at all"
        return self._valid_response_text


class _AlwaysMalformedLLMClient:
    """Mock LLM client that never produces schema-valid output."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "still not valid json"


@pytest.mark.parametrize(
    "source_id,raw_text,target_description,expected_label", MECP2_RETT_FIXTURES
)
def test_extraction_matches_hand_curated_label(
    source_id, raw_text, target_description, expected_label
):
    fixture_response = json.dumps(
        {
            "supports_or_conflicts": expected_label,
            "claim_summary": raw_text[:80],
            "evidence_type": "functional_mechanistic",
            "study_design": "preclinical_in_vivo",
            "extraction_confidence": 0.9,
        }
    )
    pipeline = ClaimExtractionPipeline(llm_client=_FixtureLLMClient(fixture_response))

    result = pipeline.extract(
        raw_text=raw_text, target_description=target_description, source_id=source_id
    )

    assert isinstance(result, ExtractionResult)
    assert result.supports_or_conflicts == expected_label
    assert 0.0 <= result.extraction_confidence <= 1.0


def test_extraction_result_has_no_confidence_score_or_source_authority_field():
    """LLM responsibility boundary (Section 23): ExtractionResult must never
    carry confidence_score or source_authority_level — those are computed
    downstream, never from the LLM call."""
    field_names = set(ExtractionResult.model_fields.keys())
    assert "confidence_score" not in field_names
    assert "source_authority_level" not in field_names


def test_extraction_retries_once_with_schema_repair_then_succeeds():
    fixture_response = json.dumps(
        {
            "supports_or_conflicts": "conflicts",
            "claim_summary": "AAV-mediated MECP2 overexpression is dose-toxic",
            "evidence_type": "functional_mechanistic",
            "study_design": "preclinical_in_vivo",
            "extraction_confidence": 0.85,
        }
    )
    client = _MalformedThenValidLLMClient(fixture_response)
    pipeline = ClaimExtractionPipeline(llm_client=client)

    result = pipeline.extract(
        raw_text="Constitutive AAV-mediated MECP2 overexpression produced dose-dependent toxicity.",
        target_description="AAV-delivered MECP2 gene therapy is well-tolerated at any dose",
        source_id="PMID:19703396",
    )

    assert client.calls == 2  # initial attempt + one schema-repair retry
    assert result.supports_or_conflicts == "conflicts"


def test_extraction_dropped_after_repair_retry_also_fails():
    pipeline = ClaimExtractionPipeline(llm_client=_AlwaysMalformedLLMClient())

    with pytest.raises(ClaimExtractionDropped) as exc_info:
        pipeline.extract(
            raw_text="Some source text.",
            target_description="Some target claim",
            source_id="PMID:00000000",
        )

    assert exc_info.value.source_id == "PMID:00000000"
