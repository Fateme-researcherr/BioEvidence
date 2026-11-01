"""
Framework classifier (Section 25, Section 13).

"Framework classification: rule-based pre-filter (does the question name a
comparator? a causal claim? a target-feasibility question?) followed by an
LLM call constrained to pick exactly one of the four MVP framework labels
(Section 13) plus a rationale string."

MVP frameworks (final, simplified set — Section 13):
1. PICO — clinical/comparative questions.
2. MechanisticBiology — pathway/mechanism questions.
3. BradfordHill — causal-inference questions, especially where RCT-style
   evidence is unavailable.
4. TargetValidation — drug-target feasibility questions.

The layer's job is only to select and parameterize a framework, explain the
choice, and surface missing information required by that framework — it
never reasons biologically itself (Section 13).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("framework_classifier")

CLASSIFIER_MODEL = os.environ.get("LLM_CLASSIFIER_MODEL", "claude-sonnet-4-6")

FRAMEWORK_LABELS = ("PICO", "MechanisticBiology", "BradfordHill", "TargetValidation")

# Rule-based pre-filter keyword signals (Section 25).
COMPARATOR_PATTERNS = [
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompared to\b",
    r"\bcompared with\b",
    r"\brelative to\b",
]
CAUSAL_CLAIM_PATTERNS = [
    r"\bcauses?\b",
    r"\bcausal\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bassociated with\b",
    r"\bcontributes? to\b",
]
TARGET_FEASIBILITY_PATTERNS = [
    r"\bdruggable\b",
    r"\btarget feasibility\b",
    r"\btractab\w*\b",
    r"\bworth pursuing\b",
    r"\bdrug target\b",
]
MECHANISM_PATTERNS = [
    r"\bpathway\b",
    r"\bmechanism\w*\b",
    r"\bhow does\b",
    r"\bencodes?\b",
    r"\bfunctional\b",
]


@dataclass
class RuleBasedSignals:
    has_comparator: bool = False
    has_causal_claim: bool = False
    has_target_feasibility_signal: bool = False
    has_mechanism_signal: bool = False
    matched_terms: List[str] = field(default_factory=list)


class FrameworkClassificationResult(BaseModel):
    """LLM-facing structured output — constrained to exactly one of the four labels."""

    framework_selected: Literal["PICO", "MechanisticBiology", "BradfordHill", "TargetValidation"]
    framework_rationale: str
    missing_information_flags: List[str] = []


class FrameworkClassifierError(Exception):
    pass


def run_rule_based_prefilter(question_text: str) -> RuleBasedSignals:
    """
    Rule-based pre-filter (Section 25): does the question name a
    comparator? a causal claim? a target-feasibility question? Signals are
    passed to the LLM to inform — not replace — its constrained
    classification.
    """
    signals = RuleBasedSignals()
    lowered = question_text.lower()

    for pattern in COMPARATOR_PATTERNS:
        if re.search(pattern, lowered):
            signals.has_comparator = True
            signals.matched_terms.append(pattern)

    for pattern in CAUSAL_CLAIM_PATTERNS:
        if re.search(pattern, lowered):
            signals.has_causal_claim = True
            signals.matched_terms.append(pattern)

    for pattern in TARGET_FEASIBILITY_PATTERNS:
        if re.search(pattern, lowered):
            signals.has_target_feasibility_signal = True
            signals.matched_terms.append(pattern)

    for pattern in MECHANISM_PATTERNS:
        if re.search(pattern, lowered):
            signals.has_mechanism_signal = True
            signals.matched_terms.append(pattern)

    return signals


class LLMClient:
    """Thin wrapper boundary around the underlying LLM call; substitutable in tests."""

    def __init__(self, model: str = CLASSIFIER_MODEL):
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


class FrameworkClassifier:
    """
    Classifies a research question against the four MVP frameworks
    (Section 13), combining a rule-based pre-filter with an LLM call
    constrained to exactly one framework label plus a rationale.
    """

    SYSTEM_PROMPT = (
        "You classify a biomedical research question into exactly one of "
        "four frameworks: PICO (clinical/comparative questions), "
        "MechanisticBiology (pathway/mechanism questions), BradfordHill "
        "(causal-inference questions, especially where RCT-style evidence "
        "is unavailable), or TargetValidation (drug-target feasibility "
        "questions). State your choice and explain why it fits. Also list "
        "any information the question is missing that this framework would "
        "need to answer it well. You are selecting and parameterizing a "
        "framework only — you do not reason about the underlying biology."
    )

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client or LLMClient()

    def classify(self, question_text: str) -> FrameworkClassificationResult:
        signals = run_rule_based_prefilter(question_text)
        user_prompt = self._build_user_prompt(question_text, signals)

        raw_response = self._llm_client.complete(
            system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt
        )
        try:
            parsed = json.loads(raw_response)
            return FrameworkClassificationResult(**parsed)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.error("Framework classification failed to parse: %s", exc)
            raise FrameworkClassifierError(
                f"Framework classification did not match the required schema: {exc}"
            ) from exc

    def _build_user_prompt(self, question_text: str, signals: RuleBasedSignals) -> str:
        return (
            f"Research question:\n{question_text}\n\n"
            f"Rule-based pre-filter signals (advisory only):\n"
            f"- names a comparator: {signals.has_comparator}\n"
            f"- names a causal claim: {signals.has_causal_claim}\n"
            f"- names a target-feasibility question: {signals.has_target_feasibility_signal}\n"
            f"- names a pathway/mechanism question: {signals.has_mechanism_signal}\n\n"
            f"Respond with a JSON object matching this schema exactly, no "
            f"additional fields, no free text outside the JSON object: "
            f'{{"framework_selected": "PICO|MechanisticBiology|BradfordHill|'
            f'TargetValidation", "framework_rationale": "string", '
            f'"missing_information_flags": ["string", ...]}}'
        )
