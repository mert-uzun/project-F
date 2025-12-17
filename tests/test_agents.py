"""
Tests for Agents Layer - REAL Integration Tests.

These tests use actual LLM calls when available.
Run with Ollama running locally or OPENAI_API_KEY set.

Tests that require LLM are marked with @requires_llm() and will skip
if no LLM backend is available.
"""

from uuid import uuid4

import pytest

from src.agents.comparator import ComparatorAgent
from src.agents.judge import JudgeAgent
from src.agents.schemas import (
    ComparisonQuery,
    Conflict,
    ConflictEvidence,
    ConflictReport,
    ConflictSeverity,
    ConflictStatus,
    ConflictType,
    RedFlag,
    SourceCitation,
    VerificationResult,
)
from src.knowledge.schemas import Entity, EntityType
from tests.conftest import requires_llm, requires_ollama

# ============================================================================
# ComparatorAgent Tests
# ============================================================================


class TestComparatorAgent:
    """Tests for ComparatorAgent - conflict detection."""

    def test_comparator_initialization(self, vector_store, graph_store) -> None:
        """Test ComparatorAgent can be initialized."""
        comparator = ComparatorAgent(vector_store, graph_store)

        assert comparator.vector_store is vector_store
        assert comparator.graph_store is graph_store

    def test_normalize_amount(self, vector_store, graph_store) -> None:
        """Test monetary amount normalization."""
        comparator = ComparatorAgent(vector_store, graph_store)

        assert comparator._normalize_amount("$500,000") == 500000.0
        assert comparator._normalize_amount("$1.5M") == 1500000.0
        assert comparator._normalize_amount("100k") == 100000.0
        assert comparator._normalize_amount("€50,000") == 50000.0
        assert comparator._normalize_amount("invalid") is None

    def test_normalize_percentage(self, vector_store, graph_store) -> None:
        """Test percentage normalization."""
        comparator = ComparatorAgent(vector_store, graph_store)

        assert comparator._normalize_percentage("5%") == 5.0
        assert comparator._normalize_percentage("10.5%") == 10.5
        assert comparator._normalize_percentage("0.5") == 0.5
        assert comparator._normalize_percentage("invalid%") is None

    def test_parse_date(self, vector_store, graph_store) -> None:
        """Test date parsing with various formats."""
        from datetime import date

        comparator = ComparatorAgent(vector_store, graph_store)

        # ISO format
        assert comparator._parse_date("2024-01-15") == date(2024, 1, 15)

        # US format
        assert comparator._parse_date("01/15/2024") == date(2024, 1, 15)

        # Long format
        assert comparator._parse_date("January 15, 2024") == date(2024, 1, 15)

        # Short month
        assert comparator._parse_date("Jan 15, 2024") == date(2024, 1, 15)

        # Year only (defaults to Jan 1)
        assert comparator._parse_date("2024") == date(2024, 1, 1)

        # Invalid
        assert comparator._parse_date("invalid date") is None

    def test_compare_monetary_values_detects_conflict(
        self, vector_store, graph_store, conflicting_entities
    ) -> None:
        """Test that monetary value conflicts are detected."""
        entities_a, entities_b = conflicting_entities

        comparator = ComparatorAgent(vector_store, graph_store)

        # Get salary entities
        salary_a = [e for e in entities_a if e.entity_type == EntityType.SALARY]
        salary_b = [e for e in entities_b if e.entity_type == EntityType.SALARY]

        conflicts = comparator._compare_monetary_values(salary_a, salary_b)

        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == ConflictType.AMOUNT_MISMATCH
        assert "$500,000" in conflicts[0].value_a or "$450,000" in conflicts[0].value_a

    def test_compare_percentage_values_detects_conflict(
        self, vector_store, graph_store, conflicting_entities
    ) -> None:
        """Test that percentage/equity conflicts are detected."""
        entities_a, entities_b = conflicting_entities

        comparator = ComparatorAgent(vector_store, graph_store)

        # Get equity entities
        equity_a = [e for e in entities_a if e.entity_type == EntityType.EQUITY]
        equity_b = [e for e in entities_b if e.entity_type == EntityType.EQUITY]

        conflicts = comparator._compare_percentage_values(equity_a, equity_b)

        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == ConflictType.PERCENTAGE_MISMATCH
        # 5% vs 8% = 3 percentage points difference
        assert "5%" in conflicts[0].value_a or "8%" in conflicts[0].value_a

    def test_compare_date_values_detects_conflict(
        self, vector_store, graph_store, conflicting_entities
    ) -> None:
        """Test that date conflicts are detected."""
        entities_a, entities_b = conflicting_entities

        comparator = ComparatorAgent(vector_store, graph_store)

        # Get date entities
        dates_a = [e for e in entities_a if e.entity_type == EntityType.DATE]
        dates_b = [e for e in entities_b if e.entity_type == EntityType.DATE]

        conflicts = comparator._compare_date_values(dates_a, dates_b)

        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == ConflictType.DATE_CONFLICT
        # January 1 vs March 15 = 73 days
        assert "days" in conflicts[0].difference

    def test_conflict_severity_based_on_difference(self, vector_store, graph_store) -> None:
        """Test that severity is correctly assigned based on difference magnitude."""
        comparator = ComparatorAgent(vector_store, graph_store)

        doc_id = uuid4()
        chunk_id = uuid4()

        # Small difference (5% = LOW)
        small_a = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$100,000",
            normalized_value="100000",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="$100,000",
            confidence=0.9,
        )
        small_b = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$103,000",
            normalized_value="103000",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="$103,000",
            confidence=0.9,
        )

        conflicts = comparator._compare_monetary_values([small_a], [small_b])
        assert len(conflicts) > 0
        assert conflicts[0].severity == ConflictSeverity.LOW

        # Large difference (>50% = CRITICAL)
        big_a = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$100,000",
            normalized_value="100000",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="$100,000",
            confidence=0.9,
        )
        big_b = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$200,000",
            normalized_value="200000",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="$200,000",
            confidence=0.9,
        )

        conflicts = comparator._compare_monetary_values([big_a], [big_b])
        assert len(conflicts) > 0
        assert conflicts[0].severity == ConflictSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_compare_returns_empty_when_no_conflicts(self, vector_store, graph_store) -> None:
        """Test that compare returns empty list when documents have no conflicts."""
        comparator = ComparatorAgent(vector_store, graph_store)

        # Empty query - no documents to compare
        query = ComparisonQuery(
            document_ids=[uuid4(), uuid4()],
            focus_areas=["salary"],
        )

        # Since there's no data in the stores, should return empty
        conflicts = await comparator.compare(query)

        assert isinstance(conflicts, list)
        # No data means no conflicts found
        assert len(conflicts) == 0


# ============================================================================
# JudgeAgent Tests
# ============================================================================


class TestJudgeAgent:
    """Tests for JudgeAgent - conflict verification."""

    def test_judge_initialization(self, vector_store, graph_store) -> None:
        """Test JudgeAgent can be initialized."""
        judge = JudgeAgent(vector_store, graph_store)

        assert judge.vector_store is vector_store
        assert judge.graph_store is graph_store

    def test_severity_order(self, vector_store, graph_store) -> None:
        """Test severity ordering for report sorting."""
        judge = JudgeAgent(vector_store, graph_store)

        assert judge._severity_order(ConflictSeverity.CRITICAL) == 0
        assert judge._severity_order(ConflictSeverity.HIGH) == 1
        assert judge._severity_order(ConflictSeverity.MEDIUM) == 2
        assert judge._severity_order(ConflictSeverity.LOW) == 3

    def test_create_red_flag(self, vector_store, graph_store) -> None:
        """Test RedFlag creation from verified conflict."""
        judge = JudgeAgent(vector_store, graph_store)

        # Create a sample conflict
        doc_id = uuid4()
        chunk_id = uuid4()

        conflict = Conflict(
            conflict_type=ConflictType.SALARY_MISMATCH,
            severity=ConflictSeverity.HIGH,
            status=ConflictStatus.VERIFIED,
            title="Salary Mismatch",
            description="$500,000 vs $450,000",
            evidence_a=ConflictEvidence(
                entity=Entity(
                    entity_type=EntityType.SALARY,
                    value="$500,000",
                    source_document_id=doc_id,
                    source_chunk_id=chunk_id,
                    source_page=1,
                    source_text="Salary: $500,000",
                ),
                citation=SourceCitation(
                    document_id=doc_id,
                    document_name="Doc A",
                    page_number=1,
                    chunk_id=chunk_id,
                    excerpt="Salary: $500,000",
                ),
                extracted_value="$500,000",
            ),
            evidence_b=ConflictEvidence(
                entity=Entity(
                    entity_type=EntityType.SALARY,
                    value="$450,000",
                    source_document_id=doc_id,
                    source_chunk_id=chunk_id,
                    source_page=1,
                    source_text="Salary: $450,000",
                ),
                citation=SourceCitation(
                    document_id=doc_id,
                    document_name="Doc B",
                    page_number=1,
                    chunk_id=chunk_id,
                    excerpt="Salary: $450,000",
                ),
                extracted_value="$450,000",
            ),
            value_a="$500,000",
            value_b="$450,000",
            difference="10%",
        )

        verification = VerificationResult(
            conflict_id=conflict.conflict_id,
            is_valid=True,
            confidence=0.9,
            reasoning="Real salary discrepancy",
            updated_status=ConflictStatus.VERIFIED,
            recommendations=["Request clarification"],
        )

        red_flag = judge._create_red_flag(conflict, verification)

        assert isinstance(red_flag, RedFlag)
        assert red_flag.conflict == conflict
        assert red_flag.verification == verification
        assert "Salary" in red_flag.summary
        assert red_flag.priority == 2  # HIGH = priority 2
        assert "clarification" in red_flag.recommended_action.lower()


# ============================================================================
# Integration Tests (Require LLM)
# ============================================================================


class TestAgentsIntegration:
    """Integration tests that require real LLM calls."""

    @pytest.mark.asyncio
    @requires_llm()
    async def test_full_detection_pipeline_with_llm(self, populated_stores) -> None:
        """
        Test the full Comparator → Judge pipeline with real LLM.
        Validates that responses exist, not specific content.
        """
        vector_store, graph_store = populated_stores

        # This will use the real LLM
        from src.agents.judge import detect_and_verify

        doc_ids = [uuid4(), uuid4()]  # Placeholder IDs
        doc_names = ["Contract_A.pdf", "Contract_B.pdf"]

        report = await detect_and_verify(
            document_ids=doc_ids,
            document_names=doc_names,
            vector_store=vector_store,
            graph_store=graph_store,
            focus_areas=["salary", "equity"],
        )

        # Validate report structure (not specific content)
        assert isinstance(report, ConflictReport)
        assert report.document_ids == doc_ids
        assert report.document_names == doc_names
        assert isinstance(report.total_conflicts_detected, int)
        assert isinstance(report.total_verified, int)
        assert isinstance(report.red_flags, list)

    @pytest.mark.asyncio
    @requires_ollama()
    async def test_judge_verification_with_real_llm(self, vector_store, graph_store) -> None:
        """
        Test Judge verification with real Ollama LLM.
        Validates that a response is returned.
        """
        judge = JudgeAgent(vector_store, graph_store)

        # Create a test conflict
        doc_id = uuid4()
        chunk_id = uuid4()

        conflict = Conflict(
            conflict_type=ConflictType.SALARY_MISMATCH,
            severity=ConflictSeverity.HIGH,
            status=ConflictStatus.DETECTED,
            title="Salary Conflict",
            description="Document A says $500,000, Document B says $450,000",
            evidence_a=ConflictEvidence(
                entity=Entity(
                    entity_type=EntityType.SALARY,
                    value="$500,000",
                    source_document_id=doc_id,
                    source_chunk_id=chunk_id,
                    source_page=1,
                    source_text="Annual base salary of $500,000.",
                ),
                citation=SourceCitation(
                    document_id=doc_id,
                    document_name="Employment_Agreement.pdf",
                    page_number=1,
                    chunk_id=chunk_id,
                    excerpt="Annual base salary of $500,000.",
                ),
                extracted_value="$500,000",
            ),
            evidence_b=ConflictEvidence(
                entity=Entity(
                    entity_type=EntityType.SALARY,
                    value="$450,000",
                    source_document_id=doc_id,
                    source_chunk_id=chunk_id,
                    source_page=1,
                    source_text="Compensation: $450,000 per year.",
                ),
                citation=SourceCitation(
                    document_id=doc_id,
                    document_name="Offer_Letter.pdf",
                    page_number=1,
                    chunk_id=chunk_id,
                    excerpt="Compensation: $450,000 per year.",
                ),
                extracted_value="$450,000",
            ),
            value_a="$500,000",
            value_b="$450,000",
        )

        # This calls the real LLM
        result = await judge.verify_conflict(conflict)

        # Validate response exists and has required fields
        assert isinstance(result, VerificationResult)
        assert result.conflict_id == conflict.conflict_id
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0  # Must have some reasoning
        assert result.updated_status in [
            ConflictStatus.VERIFIED,
            ConflictStatus.REJECTED,
            ConflictStatus.NEEDS_REVIEW,
        ]


# ============================================================================
# Schema Tests
# ============================================================================


class TestAgentSchemas:
    """Tests for agent layer schemas."""

    def test_conflict_creation(self) -> None:
        """Test Conflict model creation."""
        doc_id = uuid4()
        chunk_id = uuid4()

        entity = Entity(
            entity_type=EntityType.MONETARY_AMOUNT,
            value="$100,000",
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_page=1,
            source_text="$100,000",
        )

        evidence = ConflictEvidence(
            entity=entity,
            citation=SourceCitation(
                document_id=doc_id,
                document_name="test.pdf",
                page_number=1,
                chunk_id=chunk_id,
                excerpt="test excerpt",
            ),
            extracted_value="$100,000",
        )

        conflict = Conflict(
            conflict_type=ConflictType.AMOUNT_MISMATCH,
            title="Test Conflict",
            description="Test description",
            evidence_a=evidence,
            evidence_b=evidence,
            value_a="$100,000",
            value_b="$90,000",
        )

        assert conflict.conflict_type == ConflictType.AMOUNT_MISMATCH
        assert conflict.severity == ConflictSeverity.MEDIUM  # default
        assert conflict.status == ConflictStatus.DETECTED  # default

    def test_conflict_report_properties(self) -> None:
        """Test ConflictReport computed properties."""
        report = ConflictReport(
            document_ids=[uuid4(), uuid4()],
            document_names=["A.pdf", "B.pdf"],
            total_conflicts_detected=10,
            total_verified=8,
            total_rejected=2,
            red_flags=[],
        )

        assert len(report.critical_flags) == 0
        assert len(report.high_flags) == 0
        assert "0 issues" in report.to_summary()

    def test_comparison_query_validation(self) -> None:
        """Test ComparisonQuery validation."""
        # Valid query
        query = ComparisonQuery(
            document_ids=[uuid4(), uuid4()],
            focus_areas=["salary"],
        )
        assert len(query.document_ids) == 2

        # Query with defaults
        query = ComparisonQuery(document_ids=[uuid4(), uuid4()])
        assert "salary" in query.focus_areas
        assert query.min_confidence == 0.7
        assert query.include_low_severity is False
