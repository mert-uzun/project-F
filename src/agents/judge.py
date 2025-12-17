"""
Judge Agent - Conflict Verification.

The Judge Agent verifies conflicts found by the Comparator:
1. Reviews each conflict with additional context
2. Determines if it's a true positive or false positive
3. Updates severity if needed
4. Generates the final Red Flag Report

This is the "verification" half of the conflict detection pipeline.
The Judge prevents hallucinated or spurious conflicts from reaching the report.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.agents.schemas import (
    Conflict,
    ConflictReport,
    ConflictSeverity,
    ConflictStatus,
    RedFlag,
    VerificationResult,
)
from src.knowledge.graph_store import GraphStore
from src.knowledge.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JudgeError(Exception):
    """Raised when verification fails."""
    pass


# ============================================================================
# Verification Prompts
# ============================================================================

VERIFICATION_PROMPT = """You are a senior legal/financial analyst tasked with verifying potential conflicts.

## Conflict to Verify:
**Type:** {conflict_type}
**Severity:** {severity}
**Description:** {description}

## Evidence from Document A:
- Value: {value_a}
- Context: {context_a}
- Page: {page_a}

## Evidence from Document B:
- Value: {value_b}
- Context: {context_b}
- Page: {page_b}

## Additional Context:
{additional_context}

## Your Task:
Determine if this is a REAL conflict or a FALSE POSITIVE.

Consider:
1. Are these values actually referring to the same thing?
2. Could this be a formatting difference rather than a real conflict?
3. Is there missing context that explains the difference?
4. What is the actual risk if this conflict is real?

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed reasoning",
    "updated_severity": "low/medium/high/critical" or null,
    "recommendations": ["List of recommended actions"],
    "impact": "Potential business impact if conflict is real"
}}
"""


class VerificationOutput(BaseModel):
    """LLM output schema for verification."""

    is_valid: bool = Field(..., description="Is this a real conflict?")
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Detailed reasoning")
    updated_severity: str | None = Field(None)
    recommendations: list[str] = Field(default_factory=list)
    impact: str = Field(default="")


# ============================================================================
# Judge Agent
# ============================================================================

class JudgeAgent:
    """
    Agent that verifies conflicts and generates the final report.

    The Judge acts as a "second opinion" to filter out false positives
    and ensure only real conflicts reach the final report.

    Usage:
        judge = JudgeAgent(vector_store, graph_store)
        report = await judge.verify_and_report(conflicts, query)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm: Any = None,
    ) -> None:
        """
        Initialize the Judge Agent.

        Args:
            vector_store: Vector store for additional context
            graph_store: Graph store for relationship context
            llm: Optional LLM instance
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Get LLM instance."""
        if self._llm is None:
            from src.utils.llm_factory import get_llm
            self._llm = get_llm()
        return self._llm

    async def verify_and_report(
        self,
        conflicts: list[Conflict],
        document_ids: list[UUID],
        document_names: list[str],
    ) -> ConflictReport:
        """
        Verify conflicts and generate the final report.

        Args:
            conflicts: Conflicts detected by Comparator
            document_ids: IDs of documents compared
            document_names: Names of documents compared

        Returns:
            Final ConflictReport with verified red flags
        """
        logger.info(f"Judge verifying {len(conflicts)} conflicts")

        # Verify each conflict
        verified_conflicts: list[tuple[Conflict, VerificationResult]] = []
        rejected_count = 0

        for conflict in conflicts:
            result = await self.verify_conflict(conflict)

            if result.is_valid:
                # Update conflict status
                conflict.status = result.updated_status
                if result.updated_severity:
                    try:
                        conflict.severity = ConflictSeverity(result.updated_severity)
                    except ValueError:
                        pass
                conflict.verification_notes = result.reasoning
                verified_conflicts.append((conflict, result))
            else:
                conflict.status = ConflictStatus.REJECTED
                conflict.is_false_positive = True
                conflict.verification_notes = result.reasoning
                rejected_count += 1

        logger.info(
            f"Verification complete: {len(verified_conflicts)} verified, "
            f"{rejected_count} rejected"
        )

        # Generate red flags from verified conflicts
        red_flags = [
            self._create_red_flag(conflict, result)
            for conflict, result in verified_conflicts
        ]

        # Sort by severity and priority
        red_flags.sort(key=lambda f: (
            self._severity_order(f.conflict.severity),
            f.priority,
        ))

        # Create the report
        report = ConflictReport(
            document_ids=document_ids,
            document_names=document_names,
            total_conflicts_detected=len(conflicts),
            total_verified=len(verified_conflicts),
            total_rejected=rejected_count,
            red_flags=red_flags,
        )

        logger.info(f"Generated report: {report.to_summary()}")

        return report

    async def verify_conflict(self, conflict: Conflict) -> VerificationResult:
        """
        Verify a single conflict.

        Args:
            conflict: The conflict to verify

        Returns:
            VerificationResult with decision and reasoning
        """
        logger.debug(f"Verifying conflict: {conflict.title}")

        # Get additional context from vector store
        additional_context = await self._get_additional_context(conflict)

        # Build verification prompt
        prompt = VERIFICATION_PROMPT.format(
            conflict_type=conflict.conflict_type.value,
            severity=conflict.severity.value,
            description=conflict.description,
            value_a=conflict.value_a,
            value_b=conflict.value_b,
            context_a=conflict.evidence_a.citation.excerpt,
            context_b=conflict.evidence_b.citation.excerpt,
            page_a=conflict.evidence_a.citation.page_number,
            page_b=conflict.evidence_b.citation.page_number,
            additional_context=additional_context,
        )

        try:
            # Call LLM for verification
            response = await self.llm.acomplete(prompt)
            text = response.text.strip()

            # Parse JSON response
            import json

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            output = VerificationOutput.model_validate(data)

            # Determine status
            if output.is_valid:
                if output.confidence >= 0.8:
                    status = ConflictStatus.VERIFIED
                else:
                    status = ConflictStatus.NEEDS_REVIEW
            else:
                status = ConflictStatus.REJECTED

            return VerificationResult(
                conflict_id=conflict.conflict_id,
                is_valid=output.is_valid,
                confidence=output.confidence,
                reasoning=output.reasoning,
                updated_severity=output.updated_severity,
                updated_status=status,
                additional_context=additional_context[:500] if additional_context else None,
                recommendations=output.recommendations,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification output: {e}")
            # Default to needs review if LLM output is unparseable
            return VerificationResult(
                conflict_id=conflict.conflict_id,
                is_valid=True,
                confidence=0.5,
                reasoning="Unable to verify automatically - manual review required",
                updated_status=ConflictStatus.NEEDS_REVIEW,
                recommendations=["Manual review required"],
            )
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                conflict_id=conflict.conflict_id,
                is_valid=True,
                confidence=0.5,
                reasoning=f"Verification error: {e}",
                updated_status=ConflictStatus.NEEDS_REVIEW,
                recommendations=["Manual review required due to system error"],
            )

    async def _get_additional_context(self, conflict: Conflict) -> str:
        """Get additional context from vector store for verification."""
        # Search for related content
        search_queries = [
            conflict.value_a,
            conflict.value_b,
        ]

        additional_chunks: list[str] = []

        for query in search_queries:
            results = self.vector_store.search(query, top_k=2)
            for result in results:
                # Skip if it's the same as original context
                if result.content not in conflict.evidence_a.citation.excerpt:
                    if result.content not in conflict.evidence_b.citation.excerpt:
                        additional_chunks.append(f"[Page {result.metadata.get('page_number', '?')}]: {result.content[:300]}")

        return "\n\n".join(additional_chunks[:3]) if additional_chunks else "No additional context found."

    def _create_red_flag(
        self,
        conflict: Conflict,
        verification: VerificationResult,
    ) -> RedFlag:
        """Create a RedFlag from verified conflict."""
        # Generate summary
        summary = f"{conflict.conflict_type.value.replace('_', ' ').title()}: {conflict.value_a} vs {conflict.value_b}"

        # Generate impact based on severity
        impact_map = {
            ConflictSeverity.CRITICAL: "Critical risk - could affect deal terms or valuation",
            ConflictSeverity.HIGH: "High risk - requires immediate attention from legal/finance",
            ConflictSeverity.MEDIUM: "Moderate risk - should be clarified before closing",
            ConflictSeverity.LOW: "Low risk - minor discrepancy to note",
        }
        impact = impact_map.get(conflict.severity, "Impact unknown")

        # Generate recommended action
        if verification.recommendations:
            recommended_action = verification.recommendations[0]
        else:
            action_map = {
                ConflictSeverity.CRITICAL: "Escalate to deal team immediately",
                ConflictSeverity.HIGH: "Request clarification from counterparty",
                ConflictSeverity.MEDIUM: "Document in DD report for discussion",
                ConflictSeverity.LOW: "Note for awareness, no action needed",
            }
            recommended_action = action_map.get(conflict.severity, "Review manually")

        # Determine priority (1 = highest)
        priority_map = {
            ConflictSeverity.CRITICAL: 1,
            ConflictSeverity.HIGH: 2,
            ConflictSeverity.MEDIUM: 3,
            ConflictSeverity.LOW: 4,
        }
        priority = priority_map.get(conflict.severity, 5)

        return RedFlag(
            conflict=conflict,
            verification=verification,
            summary=summary,
            impact=impact,
            recommended_action=recommended_action,
            priority=priority,
        )

    def _severity_order(self, severity: ConflictSeverity) -> int:
        """Get sort order for severity (lower = more severe)."""
        order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
        }
        return order.get(severity, 4)


# ============================================================================
# Convenience Function
# ============================================================================

async def detect_and_verify(
    document_ids: list[UUID],
    document_names: list[str],
    vector_store: VectorStore,
    graph_store: GraphStore,
    focus_areas: list[str] | None = None,
) -> ConflictReport:
    """
    Full conflict detection pipeline: Comparator → Judge → Report.

    Args:
        document_ids: Documents to compare
        document_names: Document filenames
        vector_store: Vector store instance
        graph_store: Graph store instance
        focus_areas: Areas to focus on

    Returns:
        Final ConflictReport
    """
    from src.agents.comparator import ComparatorAgent
    from src.agents.schemas import ComparisonQuery

    # Create query
    query = ComparisonQuery(
        document_ids=document_ids,
        focus_areas=focus_areas or ["salary", "equity", "dates", "parties"],
    )

    # Run Comparator
    comparator = ComparatorAgent(vector_store, graph_store)
    conflicts = await comparator.compare(query)

    # Run Judge
    judge = JudgeAgent(vector_store, graph_store)
    report = await judge.verify_and_report(
        conflicts,
        document_ids,
        document_names
    )

    return report
