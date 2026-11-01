"""
Explainability Agent (Section 21.4, Section 26 provenance gate).

INPUT:  { reasoning_bundle: ReasoningAgentOutput }
OUTPUT: {
    report_sections: [{
        claim_or_hypothesis: string,
        evidence_trace: [{evidence_id, source_database, source_id, confidence_score, primary_or_secondary}],
        reasoning_path: [NodeRef | EdgeRef],
        framework_used: string
    }],
    trace_complete: boolean
}

Dependencies: Postgres repository only (`pg_repo`) — no Neo4j dependency
(Patch 5): `kg_traversal_path` was already denormalized into self-describing
steps by the Reasoning Agent, so this agent never needs a second Neo4j
lookup.

Provenance requirement (Section 26, core MVP requirement, not deferred):
every claim, hypothesis, or recommendation must carry a resolvable link to
(1) supporting Evidence Object(s), (2) the original source (PMID/NCT/
ontology ID), (3) confidence level, and (4) the reasoning path used to
arrive at it. `validate_trace_complete` enforces this per section; failure
in any section means the entire run returns a structured GapNotice — never
a partial or degraded report.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from schemas.evidence_object_schema import PRIMARY_SOURCE_DATABASES


@dataclass
class EvidenceTraceEntry:
    evidence_id: str
    source_database: str
    source_id: str
    confidence_score: float
    primary_or_secondary: str


@dataclass
class ReportSection:
    claim_or_hypothesis: str
    evidence_trace: List[EvidenceTraceEntry] = field(default_factory=list)
    reasoning_path: List[dict] = field(default_factory=list)
    framework_used: str = ""

    def _as_validation_namespace(self):
        # validate_trace_complete expects attribute access
        # (section.evidence_trace, section.reasoning_path, section.framework_used).
        return self


@dataclass
class GapNotice:
    reason: str
    failed_sections: List[str] = field(default_factory=list)


def validate_trace_complete(section: ReportSection) -> bool:
    """
    Section 26 trace validation function — reproduced exactly:

        return (
            len(section.evidence_trace) > 0 and
            all(e["source_database"] in PRIMARY_SOURCE_ENUM for e in section.evidence_trace) and
            len(section.reasoning_path) > 0 and
            bool(section.framework_used)
        )
    """
    return (
        len(section.evidence_trace) > 0
        and all(e.source_database in PRIMARY_SOURCE_DATABASES for e in section.evidence_trace)
        and len(section.reasoning_path) > 0
        and bool(section.framework_used)
    )


class ExplainabilityAgent:
    """
    Renders the four-part provenance trace (evidence + source + confidence
    + reasoning path) for every claim/hypothesis/failure-opportunity in a
    ReasoningAgentOutput, using only the Postgres evidence repository — no
    Neo4j dependency, since kg_traversal_path arrives already denormalized.
    """

    def __init__(self, pg_repo):
        self._pg_repo = pg_repo

    def build_report(self, reasoning_bundle: dict) -> dict:
        framework_used = reasoning_bundle.get("framework_selected", "")
        reasoning_path = reasoning_bundle.get("kg_traversal_path", [])

        report_sections: List[ReportSection] = []

        for hypothesis in reasoning_bundle.get("hypotheses", []):
            section = self._build_hypothesis_section(hypothesis, reasoning_path, framework_used)
            report_sections.append(section)

        for opportunity in reasoning_bundle.get("failure_opportunities", []):
            section = self._build_failure_opportunity_section(
                opportunity, reasoning_path, framework_used
            )
            report_sections.append(section)

        trace_complete = len(report_sections) > 0 and all(
            validate_trace_complete(section) for section in report_sections
        )

        return {
            "report_sections": report_sections,
            "trace_complete": trace_complete,
        }

    # -- section builders ---------------------------------------------------

    def _build_hypothesis_section(self, hypothesis, reasoning_path, framework_used) -> ReportSection:
        evidence_trace = self._resolve_evidence_trace(hypothesis.supporting_evidence_ids)
        return ReportSection(
            claim_or_hypothesis=hypothesis.statement,
            evidence_trace=evidence_trace,
            reasoning_path=list(reasoning_path),
            framework_used=framework_used,
        )

    def _build_failure_opportunity_section(
        self, opportunity: dict, reasoning_path, framework_used
    ) -> ReportSection:
        evidence_trace = self._resolve_evidence_trace(
            opportunity.get("source_failure_evidence_ids", [])
        )
        return ReportSection(
            claim_or_hypothesis=opportunity.get("reframed_question", ""),
            evidence_trace=evidence_trace,
            reasoning_path=list(reasoning_path),
            framework_used=framework_used,
        )

    def _resolve_evidence_trace(self, evidence_ids: List[str]) -> List[EvidenceTraceEntry]:
        evidence_trace: List[EvidenceTraceEntry] = []
        for evidence_id in evidence_ids:
            record = self._pg_repo.get_evidence_object(evidence_id)
            if record is None:
                continue
            evidence_trace.append(
                EvidenceTraceEntry(
                    evidence_id=str(record["evidence_id"]),
                    source_database=record["source_database"],
                    source_id=record["source_id"],
                    confidence_score=record["confidence_score"],
                    primary_or_secondary=record["primary_or_secondary"],
                )
            )
        return evidence_trace

    def build_gap_notice(self, report: dict) -> Optional[GapNotice]:
        """
        Section 26 blocking logic: if any section fails validation, return
        a structured GapNotice instead of a partial/degraded report.
        """
        if report["trace_complete"]:
            return None

        failed_sections = [
            section.claim_or_hypothesis
            for section in report["report_sections"]
            if not validate_trace_complete(section)
        ]
        if not report["report_sections"]:
            failed_sections = ["no report sections were produced"]

        return GapNotice(
            reason=(
                "Provenance requirement not met: every claim/hypothesis "
                "must carry a resolvable link to supporting Evidence "
                "Object(s), the original source, confidence level, and "
                "reasoning path (Section 26). Returning a structured gap "
                "notice rather than a partial or degraded report."
            ),
            failed_sections=failed_sections,
        )
