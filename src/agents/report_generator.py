"""
Report Generator - Executive Summary Generation.

Generates formal investment banking memo-style reports.
Uses LLM to synthesize findings into partner-ready summaries.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.agents.multi_doc_analyzer import MultiDocConflict, MultiDocReport
from src.agents.reference_detector import MissingDocumentReport
from src.agents.schemas import ConflictSeverity
from src.knowledge.timeline_builder import Timeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# LLM Prompts - Formal Investment Banking Style
# ============================================================================

EXECUTIVE_SUMMARY_PROMPT = """You are a senior associate at a top-tier investment bank preparing a due diligence memorandum for Managing Directors.

MEMORANDUM

TO: Deal Team
FROM: Due Diligence Review
DATE: {current_date}
RE: Cross-Document Analysis - {document_count} Documents Reviewed

---

## DOCUMENTS ANALYZED

{document_list}

## KEY METRICS

- Total Entities Extracted: {total_entities}
- Cross-References Resolved: {resolved_entities}
- Conflicts Identified: {conflict_count}
- Missing Documents: {missing_count}

## FINDINGS

### Critical Issues ({critical_count})
{critical_issues}

### Material Discrepancies ({high_count})
{high_issues}

### Other Observations ({other_count})
{other_issues}

### Missing Documents
{missing_documents}

### Timeline Summary
{timeline_summary}

---

Based on the above findings, generate a formal investment banking executive summary memorandum.

The summary should:
1. Open with a one-paragraph executive overview suitable for senior partners
2. Highlight material risks in order of severity (CRITICAL first)
3. Use precise, professional language (avoid colloquialisms)
4. Include specific page citations for all findings
5. Conclude with prioritized action items for the deal team

Format the output as a professional markdown document with clear headers.
Use bullet points for action items. Be concise but comprehensive.
"""

CONFLICT_ANALYSIS_PROMPT = """Analyze the following conflict and provide a professional assessment:

CONFLICT TYPE: {conflict_type}
SEVERITY: {severity}

VALUE IN DOCUMENT A: {value_a}
VALUE IN DOCUMENT B: {value_b}

CONTEXT A: {context_a}
CONTEXT B: {context_b}

Provide:
1. A one-sentence professional description of this discrepancy
2. Potential business impact (Low/Medium/High/Critical)
3. Recommended due diligence follow-up action

Format as JSON with keys: description, impact, action
"""


# ============================================================================
# Schemas
# ============================================================================


class ExecutiveSummary(BaseModel):
    """Generated executive summary memorandum."""

    summary_id: UUID = Field(default_factory=uuid4)

    # Content
    summary_markdown: str = Field(..., description="Full summary in markdown")

    # Extracted highlights
    key_findings: list[str] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)

    # Metadata
    document_count: int = 0
    conflict_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = ""
    generation_time_seconds: float = 0.0

    @property
    def has_critical_issues(self) -> bool:
        return len(self.critical_issues) > 0


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    include_timeline: bool = True
    include_missing_docs: bool = True
    max_issues_per_section: int = 10
    include_page_citations: bool = True
    formal_style: bool = True  # Investment banking memo style


# ============================================================================
# Report Generator
# ============================================================================


class ReportGenerator:
    """
    Generate executive summaries and formal reports.

    Uses LLM to synthesize findings into partner-ready memoranda
    following investment banking conventions.

    Usage:
        generator = ReportGenerator()
        summary = await generator.generate_executive_summary(report)
    """

    def __init__(self, llm: Any = None) -> None:
        """
        Initialize the Report Generator.

        Args:
            llm: Optional LLM instance (lazy-loaded if not provided)
        """
        self._llm = llm

    @property
    def llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            try:
                from src.utils.llm_factory import get_llm

                self._llm = get_llm()
            except Exception as e:
                logger.warning(f"Could not load LLM: {e}")
        return self._llm

    async def generate_executive_summary(
        self,
        multi_doc_report: MultiDocReport,
        missing_doc_report: MissingDocumentReport | None = None,
        timeline: Timeline | None = None,
        config: ReportConfig | None = None,
    ) -> ExecutiveSummary:
        """
        Generate an executive summary memorandum.

        Args:
            multi_doc_report: Multi-document analysis results
            missing_doc_report: Optional missing document findings
            timeline: Optional timeline analysis
            config: Report configuration

        Returns:
            ExecutiveSummary with formatted memorandum
        """
        import time

        start_time = time.time()

        config = config or ReportConfig()

        # Build document list
        doc_list = self._format_document_list(multi_doc_report)

        # Categorize conflicts by severity
        critical, high, other = self._categorize_conflicts(
            multi_doc_report.conflicts,
            config.max_issues_per_section,
        )

        # Format missing documents
        missing_summary = (
            self._format_missing_documents(missing_doc_report)
            if missing_doc_report and config.include_missing_docs
            else "None identified."
        )

        # Format timeline
        timeline_summary = (
            self._format_timeline(timeline)
            if timeline and config.include_timeline
            else "Not analyzed."
        )

        # Build prompt
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            current_date=datetime.now().strftime("%B %d, %Y"),
            document_count=multi_doc_report.document_set.count,
            document_list=doc_list,
            total_entities=multi_doc_report.total_entities,
            resolved_entities=multi_doc_report.total_resolved_entities,
            conflict_count=len(multi_doc_report.conflicts),
            missing_count=missing_doc_report.missing_count if missing_doc_report else 0,
            critical_count=len(critical),
            critical_issues=self._format_issues(critical) or "None.",
            high_count=len(high),
            high_issues=self._format_issues(high) or "None.",
            other_count=len(other),
            other_issues=self._format_issues(other) or "None.",
            missing_documents=missing_summary,
            timeline_summary=timeline_summary,
        )

        # Generate with LLM
        summary_text = ""
        model_used = "none"

        if self.llm:
            try:
                response = await self.llm.acomplete(prompt)
                summary_text = response.text
                model_used = getattr(self.llm, "model", "unknown")
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                summary_text = self._generate_fallback_summary(
                    multi_doc_report,
                    missing_doc_report,
                    timeline,
                    critical,
                    high,
                    other,
                )
                model_used = "fallback"
        else:
            summary_text = self._generate_fallback_summary(
                multi_doc_report,
                missing_doc_report,
                timeline,
                critical,
                high,
                other,
            )
            model_used = "fallback"

        # Extract highlights from summary
        key_findings = self._extract_key_findings(summary_text)
        critical_issues = [self._issue_to_string(c) for c in critical]
        action_items = self._extract_action_items(summary_text)

        generation_time = time.time() - start_time

        summary = ExecutiveSummary(
            summary_markdown=summary_text,
            key_findings=key_findings,
            critical_issues=critical_issues,
            action_items=action_items,
            document_count=multi_doc_report.document_set.count,
            conflict_count=len(multi_doc_report.conflicts),
            model_used=model_used,
            generation_time_seconds=generation_time,
        )

        logger.info(
            f"Generated executive summary: {len(summary_text)} chars in {generation_time:.2f}s"
        )

        return summary

    def _format_document_list(self, report: MultiDocReport) -> str:
        """Format list of analyzed documents."""
        lines = []
        for i, doc_id in enumerate(report.document_set.document_ids, 1):
            doc_name = report.document_set.get_name(doc_id)
            lines.append(f"{i}. {doc_name}")
        return "\n".join(lines)

    def _categorize_conflicts(
        self,
        conflicts: list[MultiDocConflict],
        max_per_category: int,
    ) -> tuple[list, list, list]:
        """Categorize conflicts by severity."""
        critical = []
        high = []
        other = []

        for conflict in conflicts:
            if conflict.severity == ConflictSeverity.CRITICAL:
                if len(critical) < max_per_category:
                    critical.append(conflict)
            elif conflict.severity == ConflictSeverity.HIGH:
                if len(high) < max_per_category:
                    high.append(conflict)
            else:
                if len(other) < max_per_category:
                    other.append(conflict)

        return critical, high, other

    def _format_issues(self, conflicts: list[MultiDocConflict]) -> str:
        """Format conflicts as bullet points."""
        if not conflicts:
            return ""

        lines = []
        for conflict in conflicts:
            lines.append(f"- **{conflict.conflict_type.value}**: {conflict.description}")

        return "\n".join(lines)

    def _format_missing_documents(
        self,
        report: MissingDocumentReport,
    ) -> str:
        """Format missing document references."""
        if not report or not report.missing_documents:
            return "None identified."

        lines = []
        for ref in report.missing_documents[:10]:
            lines.append(
                f"- **{ref.reference_text}** - Referenced in "
                f"{ref.source_document_name}, page {ref.source_page}"
            )

        if report.missing_count > 10:
            lines.append(f"- ...and {report.missing_count - 10} additional references")

        return "\n".join(lines)

    def _format_timeline(self, timeline: Timeline) -> str:
        """Format timeline summary."""
        if not timeline or not timeline.events:
            return "No timeline events identified."

        lines = [
            f"**Date Range**: {timeline.earliest_date} to {timeline.latest_date}",
            f"**Events Identified**: {timeline.event_count}",
        ]

        if timeline.conflicts:
            lines.append(f"**Temporal Conflicts**: {timeline.conflict_count}")
            for conflict in timeline.conflicts[:3]:
                lines.append(f"  - {conflict.description}")

        return "\n".join(lines)

    def _issue_to_string(self, conflict: MultiDocConflict) -> str:
        """Convert a conflict to a string description."""
        return f"{conflict.conflict_type.value}: {conflict.title}"

    def _extract_key_findings(self, summary_text: str) -> list[str]:
        """Extract key findings from generated summary."""
        findings = []

        # Look for bullet points or numbered items
        import re

        patterns = [
            r"^\s*[-•]\s*(.+)$",
            r"^\s*\d+\.\s*(.+)$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, summary_text, re.MULTILINE)
            findings.extend(matches[:10])

        return findings[:10]

    def _extract_action_items(self, summary_text: str) -> list[str]:
        """Extract action items from generated summary."""
        actions = []

        # Look for action-oriented language
        import re

        # Find section after "Action Items" or "Recommendations"
        action_section = re.search(
            r"(?:action items|recommendations|next steps).*?$(.*?)(?:\n#|\Z)",
            summary_text,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if action_section:
            section_text = action_section.group(1)
            # Extract bullet points
            items = re.findall(r"^\s*[-•\[\]]\s*(.+)$", section_text, re.MULTILINE)
            actions.extend(items)

        return actions[:10]

    def _generate_fallback_summary(
        self,
        multi_doc_report: MultiDocReport,
        missing_doc_report: MissingDocumentReport | None,
        timeline: Timeline | None,
        critical: list[MultiDocConflict],
        high: list[MultiDocConflict],
        other: list[MultiDocConflict],
    ) -> str:
        """Generate a summary without LLM."""
        lines = [
            "# Due Diligence Review Memorandum",
            "",
            f"**Date**: {datetime.now().strftime('%B %d, %Y')}",
            f"**Documents Reviewed**: {multi_doc_report.document_set.count}",
            "",
            "---",
            "",
            "## Executive Overview",
            "",
            f"This memorandum summarizes the automated cross-document analysis of "
            f"{multi_doc_report.document_set.count} documents. The analysis identified "
            f"{len(multi_doc_report.conflicts)} potential discrepancies requiring attention.",
            "",
        ]

        # Critical issues
        if critical:
            lines.extend(
                [
                    "## Critical Issues ⚠️",
                    "",
                    "The following issues require immediate attention:",
                    "",
                ]
            )
            for c in critical:
                lines.append(f"- **{c.title}**: {c.description}")
            lines.append("")

        # High priority
        if high:
            lines.extend(
                [
                    "## Material Discrepancies",
                    "",
                ]
            )
            for c in high:
                lines.append(f"- **{c.title}**: {c.description}")
            lines.append("")

        # Missing documents
        if missing_doc_report and missing_doc_report.has_missing:
            lines.extend(
                [
                    "## Missing Document References",
                    "",
                    "The following referenced documents were not provided for review:",
                    "",
                ]
            )
            for ref in missing_doc_report.missing_documents[:5]:
                lines.append(
                    f"- {ref.reference_text} (cited in {ref.source_document_name}, p.{ref.source_page})"
                )
            lines.append("")

        # Timeline
        if timeline and timeline.conflicts:
            lines.extend(
                [
                    "## Timeline Conflicts",
                    "",
                ]
            )
            for tc in timeline.conflicts[:3]:
                lines.append(f"- {tc.description}")
            lines.append("")

        # Action items
        lines.extend(
            [
                "## Recommended Actions",
                "",
                "- [ ] Review all CRITICAL issues before proceeding",
                "- [ ] Request missing referenced documents from counterparty",
                "- [ ] Clarify discrepancies with legal counsel",
                "- [ ] Update transaction timeline if necessary",
                "",
                "---",
                "",
                "*This report was generated by automated document analysis. "
                "All findings should be verified by qualified professionals.*",
            ]
        )

        return "\n".join(lines)

    def generate_conflict_table(
        self,
        conflicts: list[MultiDocConflict],
    ) -> str:
        """Generate a markdown table of conflicts."""
        if not conflicts:
            return "No conflicts identified."

        lines = [
            "| Severity | Type | Description | Documents |",
            "|----------|------|-------------|-----------|",
        ]

        for conflict in conflicts:
            severity_icon = {
                ConflictSeverity.CRITICAL: "🔴",
                ConflictSeverity.HIGH: "🟠",
                ConflictSeverity.MEDIUM: "🟡",
                ConflictSeverity.LOW: "🟢",
            }.get(conflict.severity, "⚪")

            lines.append(
                f"| {severity_icon} {conflict.severity.value} | "
                f"{conflict.conflict_type.value} | "
                f"{conflict.title} | "
                f"{conflict.document_count} docs |"
            )

        return "\n".join(lines)

    def generate_entity_matrix(
        self,
        entity_type: str,
        occurrences: dict[str, list[tuple[str, str]]],
    ) -> str:
        """
        Generate a matrix showing entity values per document.

        Args:
            entity_type: Type of entity
            occurrences: Dict mapping entity value to list of (doc_name, value)

        Returns:
            Markdown table
        """
        if not occurrences:
            return f"No {entity_type} entities found."

        # Get all document names
        all_docs: set[str] = set()
        for values in occurrences.values():
            for doc_name, _ in values:
                all_docs.add(doc_name)

        doc_list = sorted(all_docs)

        # Build header
        header = f"| {entity_type} | " + " | ".join(doc_list) + " |"
        separator = "|" + "---|" * (len(doc_list) + 1)

        lines = [header, separator]

        # Build rows
        for entity_value, doc_values in occurrences.items():
            doc_value_map = {d: v for d, v in doc_values}
            row = f"| {entity_value} | "
            row += " | ".join(doc_value_map.get(d, "-") for d in doc_list)
            row += " |"
            lines.append(row)

        return "\n".join(lines)
