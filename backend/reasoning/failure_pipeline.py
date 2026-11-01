"""
Failure Intelligence pipeline (Section 10, realized per Section 25):

    Failure (conflicting Evidence Object)
            v
    Limitation extraction (what specifically limited the original finding)
            v
    Technology delta analysis (has the limiting factor changed since?)
            v
    Alternative perspective (what adjacent approach wasn't tried?)
            v
    Reframed research question (never a retry recommendation)

Every step operates only on Evidence Objects already fetched from
Postgres — nothing is invented independent of a real, linked Evidence
Object. The output is always a candidate research opportunity, never a
stored verdict (Section 10).

Hard constraint, preserved without exception: the system never
recommends repeating a failed therapy. The output is always a new,
narrower, technology- or perspective-adjusted question — never "try
again."
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("failure_pipeline")

FAILURE_PIPELINE_MODEL = os.environ.get("LLM_FAILURE_PIPELINE_MODEL", "claude-sonnet-4-6")

# Hard constraint guard: phrases that signal a bare retry recommendation.
RETRY_RECOMMENDATION_PATTERNS = [
    r"\btry again\b",
    r"\bretry\b",
    r"\brepeat the (same )?(trial|study|therapy|treatment)\b",
    r"\bre-?run the same\b",
]


class RetryRecommendationError(Exception):
    """
    Raised when a candidate reframed question would amount to recommending
    repeating the failed therapy. This is a hard constraint with no
    exception path (Section 10) — the pipeline must reframe again or
    surface no opportunity at all, never return this candidate.
    """


class LimitationExtraction(BaseModel):
    limitation_category: str  # technological | biological_assumption | design
    limitation_description: str


class TechnologyDeltaCheck(BaseModel):
    limiting_factor_changed: bool
    technology_delta_description: str


class AlternativePerspective(BaseModel):
    adjacent_approach: str
    rationale_for_trying_now: str


@dataclass
class ReframedQuestion:
    original_failure_evidence_id: str
    limitation: LimitationExtraction
    technology_delta: TechnologyDeltaCheck
    alternative_perspective: AlternativePerspective
    reframed_question_statement: str


class LLMClient:
    """Thin wrapper boundary around the underlying LLM call; substitutable in tests."""

    def __init__(self, model: str = FAILURE_PIPELINE_MODEL):
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
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


def _parse_or_raise(raw_response: str, schema_cls, stage_name: str):
    try:
        parsed = json.loads(raw_response)
        return schema_cls(**parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.error("Failure pipeline stage '%s' failed to parse: %s", stage_name, exc)
        raise ValueError(f"Failure pipeline stage '{stage_name}' returned an invalid response") from exc


def _assert_not_retry_recommendation(statement: str) -> None:
    lowered = statement.lower()
    for pattern in RETRY_RECOMMENDATION_PATTERNS:
        if re.search(pattern, lowered):
            raise RetryRecommendationError(
                "Reframed question would recommend repeating the failed "
                "therapy — rejected per Section 10 hard constraint: "
                f"'{statement}'"
            )


class FailurePipeline:
    """
    Runs the four-stage Failure Intelligence pipeline over a single
    conflicting (failure-tagged) Evidence Object already fetched from
    Postgres.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client or LLMClient()

    def process_failure(
        self,
        conflicting_evidence: dict,
        question_context: str,
    ) -> ReframedQuestion:
        """
        `conflicting_evidence` is a dict as returned by
        EvidenceObjectRepository (a real, already-linked Evidence Object
        with supports_or_conflicts == 'conflicts'). Nothing here is
        invented independent of that record.
        """
        limitation = self._extract_limitation(conflicting_evidence)
        technology_delta = self._check_technology_delta(conflicting_evidence, limitation)
        alternative_perspective = self._surface_alternative_perspective(
            conflicting_evidence, limitation, technology_delta
        )
        reframed_statement = self._reframe_question(
            conflicting_evidence,
            question_context,
            limitation,
            technology_delta,
            alternative_perspective,
        )

        _assert_not_retry_recommendation(reframed_statement)

        return ReframedQuestion(
            original_failure_evidence_id=str(conflicting_evidence.get("evidence_id")),
            limitation=limitation,
            technology_delta=technology_delta,
            alternative_perspective=alternative_perspective,
            reframed_question_statement=reframed_statement,
        )

    # -- stage 1: limitation extraction -----------------------------------

    def _extract_limitation(self, conflicting_evidence: dict) -> LimitationExtraction:
        system_prompt = (
            "Given a claim summary from a scientific source, identify what "
            "specifically limited the original finding: technological "
            "(instrumentation, resolution, model organisms available at "
            "the time), biological-assumption-based, or design-based "
            "(power, controls, endpoints). Respond with JSON only: "
            '{"limitation_category": "technological|biological_assumption|'
            'design", "limitation_description": "string"}'
        )
        user_prompt = (
            f"Claim summary: {conflicting_evidence.get('claim_summary')}\n"
            f"Source: {conflicting_evidence.get('source_database')} "
            f"({conflicting_evidence.get('source_id')})\n"
            f"Study design: {conflicting_evidence.get('study_design')}"
        )
        raw_response = self._llm_client.complete(system_prompt, user_prompt)
        return _parse_or_raise(raw_response, LimitationExtraction, "limitation_extraction")

    # -- stage 2: technology-delta check -----------------------------------

    def _check_technology_delta(
        self, conflicting_evidence: dict, limitation: LimitationExtraction
    ) -> TechnologyDeltaCheck:
        system_prompt = (
            "Given a specific limiting factor that blocked an original "
            "finding, determine whether that specific factor has changed "
            "since — not a generic 'have things improved' scan, but "
            "targeted against what actually blocked the original work. "
            "Respond with JSON only: "
            '{"limiting_factor_changed": true|false, '
            '"technology_delta_description": "string"}'
        )
        user_prompt = (
            f"Original limiting factor ({limitation.limitation_category}): "
            f"{limitation.limitation_description}\n"
            f"Claim asserted date: {conflicting_evidence.get('claim_asserted_date')}"
        )
        raw_response = self._llm_client.complete(system_prompt, user_prompt)
        return _parse_or_raise(raw_response, TechnologyDeltaCheck, "technology_delta_check")

    # -- stage 3: alternative-perspective surfacing -------------------------

    def _surface_alternative_perspective(
        self,
        conflicting_evidence: dict,
        limitation: LimitationExtraction,
        technology_delta: TechnologyDeltaCheck,
    ) -> AlternativePerspective:
        system_prompt = (
            "Given the original finding, its limitation, and whether the "
            "limiting technology has changed, identify what adjacent "
            "discipline, biological scale, or mechanistic angle was not "
            "tried, and whether current evidence suggests it's worth "
            "trying now. Respond with JSON only: "
            '{"adjacent_approach": "string", '
            '"rationale_for_trying_now": "string"}'
        )
        user_prompt = (
            f"Claim summary: {conflicting_evidence.get('claim_summary')}\n"
            f"Limitation: {limitation.limitation_description}\n"
            f"Technology delta: {technology_delta.technology_delta_description}"
        )
        raw_response = self._llm_client.complete(system_prompt, user_prompt)
        return _parse_or_raise(raw_response, AlternativePerspective, "alternative_perspective")

    # -- stage 4: reframed research question --------------------------------

    def _reframe_question(
        self,
        conflicting_evidence: dict,
        question_context: str,
        limitation: LimitationExtraction,
        technology_delta: TechnologyDeltaCheck,
        alternative_perspective: AlternativePerspective,
    ) -> str:
        system_prompt = (
            "Produce a single reframed research question statement. It "
            "must explicitly state what was tried before, why it didn't "
            "work, what's different now, and what should be asked instead. "
            "It must NEVER recommend repeating the same failed therapy or "
            "study — that is a hard constraint with no exception. Respond "
            "with the statement as plain text, nothing else."
        )
        user_prompt = (
            f"Original question context: {question_context}\n"
            f"Original failed claim: {conflicting_evidence.get('claim_summary')}\n"
            f"Limitation: {limitation.limitation_description}\n"
            f"Technology delta: {technology_delta.technology_delta_description}\n"
            f"Alternative perspective: {alternative_perspective.adjacent_approach} — "
            f"{alternative_perspective.rationale_for_trying_now}"
        )
        return self._llm_client.complete(system_prompt, user_prompt).strip()
