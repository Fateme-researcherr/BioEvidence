"""
Claim-extraction pipeline (Section 23) — stage 3 of the Evidence Pipeline:

Official source retrieval
        -> Raw record normalization
        -> Claim extraction (LLM, schema-constrained)     <- this module
        -> Evidence Object creation (schema validation)
        -> Confidence calculation (Section 24)
        -> Evidence linking (evidence_link rows per target)
        -> Validation gates (enum + completeness checks)

LLM responsibility boundary, enforced structurally: `ExtractionResult` has
no field for `confidence_score` or `source_authority_level` — those are
computed downstream from configuration and metadata, never from the LLM
call. The LLM may only estimate "was the claim extracted correctly?" It
must never decide "is this biological claim true?"

Error handling (Section 21.2): extraction failures trigger one retry with
a stricter schema-repair prompt, then are dropped with a logged reason.
"""

import json
import logging
import os
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("claim_extraction_pipeline")

EXTRACTION_MODEL = os.environ.get("LLM_EXTRACTION_MODEL", "claude-sonnet-4-6")

EXTRACTION_PROMPT_INSTRUCTIONS = (
    "Extract whether this text supports, conflicts with, or is neutral "
    "toward the given claim. Report your confidence that you extracted "
    "this correctly (0-1). Do not assess whether the underlying biology "
    "is true — only whether you read the source text correctly."
)

SCHEMA_REPAIR_INSTRUCTIONS = (
    "Your previous response did not match the required output schema. "
    "Re-extract using exactly the following JSON schema, with no "
    "additional fields and no free text outside the JSON object: "
    '{"supports_or_conflicts": "supports|conflicts|neutral", '
    '"claim_summary": "string", "evidence_type": "string", '
    '"study_design": "string", "extraction_confidence": "float between '
    '0 and 1"}. ' + EXTRACTION_PROMPT_INSTRUCTIONS
)


class ExtractionResult(BaseModel):
    """
    Structured output schema (LLM-facing subset), Section 23.

    No `confidence_score` or `source_authority_level` field — those are
    computed downstream, never from the LLM call.
    """

    supports_or_conflicts: Literal["supports", "conflicts", "neutral"]
    claim_summary: str
    evidence_type: str
    study_design: str
    extraction_confidence: float  # the ONLY confidence-relevant field the LLM produces


class ClaimExtractionDropped(Exception):
    """Raised (and caught by the caller) when extraction is dropped after retry exhaustion."""

    def __init__(self, source_id: str, reason: str):
        self.source_id = source_id
        self.reason = reason
        super().__init__(f"Extraction dropped for {source_id}: {reason}")


class LLMClient:
    """
    Thin wrapper boundary around the underlying LLM call, so the pipeline
    logic is independent of the specific SDK/transport used. Consumers may
    substitute a mock in tests.
    """

    def __init__(self, model: str = EXTRACTION_MODEL):
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion. Real implementation calls the LLM API."""
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks)


class ClaimExtractionPipeline:
    """
    Runs schema-constrained claim extraction over a single raw source
    record against a single target claim/entity/hypothesis.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client or LLMClient()

    def extract(
        self,
        raw_text: str,
        target_description: str,
        source_id: str,
    ) -> ExtractionResult:
        """
        Extract an ExtractionResult from raw_text, evaluated against
        target_description. Retries once with a schema-repair prompt on
        failure; raises ClaimExtractionDropped if the retry also fails.
        """
        user_prompt = self._build_user_prompt(
            raw_text, target_description, EXTRACTION_PROMPT_INSTRUCTIONS
        )

        result = self._attempt_extraction(user_prompt, source_id, attempt_label="initial")
        if result is not None:
            return result

        repair_prompt = self._build_user_prompt(
            raw_text, target_description, SCHEMA_REPAIR_INSTRUCTIONS
        )
        result = self._attempt_extraction(repair_prompt, source_id, attempt_label="schema-repair")
        if result is not None:
            return result

        reason = "extraction failed schema validation on initial attempt and schema-repair retry"
        logger.warning("Dropping extraction for %s: %s", source_id, reason)
        raise ClaimExtractionDropped(source_id=source_id, reason=reason)

    def _build_user_prompt(self, raw_text: str, target_description: str, instructions: str) -> str:
        return (
            f"{instructions}\n\n"
            f"Target claim/entity being evaluated:\n{target_description}\n\n"
            f"Source text:\n{raw_text}"
        )

    def _attempt_extraction(
        self, user_prompt: str, source_id: str, attempt_label: str
    ) -> Optional[ExtractionResult]:
        try:
            raw_response = self._llm_client.complete(
                system_prompt=EXTRACTION_PROMPT_INSTRUCTIONS, user_prompt=user_prompt
            )
            parsed = json.loads(raw_response)
            return ExtractionResult(**parsed)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.info(
                "Extraction attempt (%s) failed for %s: %s", attempt_label, source_id, exc
            )
            return None
