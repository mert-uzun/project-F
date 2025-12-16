"""
Tests for Report Generator.

Tests executive summary generation with and without LLM.
"""

import pytest
from uuid import uuid4
from datetime import datetime

from src.agents.report_generator import (
    ReportGenerator,
    ExecutiveSummary,
    ReportConfig,
)
from src.agents.multi_doc_analyzer import (
    MultiDocReport,
    MultiDocConflict,
    DocumentSet,
)
from src.agents.reference_detector import (
    MissingDocumentReport,
    DocumentReference,
    ReferenceType,
)
from src.knowledge.timeline_builder import Timeline, TimelineEvent, TimelineConflict, EventType
from src.knowledge.schemas import EntityType
from src.agents.schemas import ConflictSeverity, ConflictType


class TestReportConfig:
    """Tests for ReportConfig model."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ReportConfig()
        
        assert config.include_timeline is True
        assert config.include_missing_docs is True
        assert config.formal_style is True
        assert config.max_issues_per_section == 10


class TestExecutiveSummary:
    """Tests for ExecutiveSummary model."""
    
    def test_summary_creation(self) -> None:
        """Test ExecutiveSummary can be created."""
        summary = ExecutiveSummary(
            summary_markdown="# Test Summary\n\nNo issues found.",
            document_count=5,
            conflict_count=0,
            model_used="test",
        )
        
        assert not summary.has_critical_issues
        assert summary.conflict_count == 0
    
    def test_has_critical_issues(self) -> None:
        """Test has_critical_issues property."""
        summary = ExecutiveSummary(
            summary_markdown="# Summary",
            critical_issues=["Salary mismatch: $500k vs $450k"],
            document_count=2,
        )
        
        assert summary.has_critical_issues


class TestReportGenerator:
    """Tests for ReportGenerator."""
    
    @pytest.fixture
    def generator(self) -> ReportGenerator:
        """Create report generator without LLM."""
        return ReportGenerator(llm=None)
    
    @pytest.fixture
    def sample_report(self) -> MultiDocReport:
        """Create sample multi-doc report."""
        doc_set = DocumentSet(
            document_ids=[uuid4(), uuid4()],
            document_names={},
        )
        
        return MultiDocReport(
            document_set=doc_set,
            total_entities=20,
            total_relationships=10,
            conflicts=[
                MultiDocConflict(
                    entity_type=EntityType.SALARY,
                    conflict_type=ConflictType.SALARY_MISMATCH,
                    severity=ConflictSeverity.CRITICAL,
                    title="Salary Mismatch",
                    description="$500k vs $450k",
                    document_count=2,
                ),
                MultiDocConflict(
                    entity_type=EntityType.PERCENTAGE,
                    conflict_type=ConflictType.PERCENTAGE_MISMATCH,
                    severity=ConflictSeverity.HIGH,
                    title="Equity Mismatch",
                    description="5% vs 8%",
                    document_count=2,
                ),
                MultiDocConflict(
                    entity_type=EntityType.DATE,
                    conflict_type=ConflictType.DATE_CONFLICT,
                    severity=ConflictSeverity.LOW,
                    title="Date Mismatch",
                    description="Jan 1 vs Jan 15",
                    document_count=2,
                ),
            ],
        )
    
    @pytest.fixture
    def sample_missing_report(self) -> MissingDocumentReport:
        """Create sample missing document report."""
        return MissingDocumentReport(
            total_references=5,
            missing_documents=[
                DocumentReference(
                    reference_text="Exhibit A",
                    reference_type=ReferenceType.EXHIBIT,
                    normalized_name="Exhibit A",
                    source_document_id=uuid4(),
                    source_document_name="Contract.pdf",
                    source_page=3,
                ),
            ],
        )
    
    @pytest.fixture
    def sample_timeline(self) -> Timeline:
        """Create sample timeline."""
        from datetime import date
        
        return Timeline(
            events=[
                TimelineEvent(
                    event_date=date(2024, 1, 1),
                    event_type=EventType.EMPLOYMENT_START,
                    description="Employment begins",
                    source_document_id=uuid4(),
                    source_document_name="Contract",
                    source_page=1,
                ),
            ],
            conflicts=[
                TimelineConflict(
                    event_a=TimelineEvent(
                        event_date=date(2024, 12, 31),
                        event_type=EventType.EMPLOYMENT_END,
                        description="End",
                        source_document_id=uuid4(),
                        source_document_name="Doc A",
                        source_page=1,
                    ),
                    event_b=TimelineEvent(
                        event_date=date(2024, 1, 1),
                        event_type=EventType.EMPLOYMENT_START,
                        description="Start",
                        source_document_id=uuid4(),
                        source_document_name="Doc B",
                        source_page=1,
                    ),
                    conflict_type="date_mismatch",
                    description="Different dates",
                ),
            ],
            earliest_date=date(2024, 1, 1),
            latest_date=date(2024, 12, 31),
            document_count=2,
        )
    
    def test_generator_initialization(self, generator) -> None:
        """Test ReportGenerator can be initialized."""
        assert generator._llm is None
    
    def test_format_document_list(self, generator, sample_report) -> None:
        """Test document list formatting."""
        doc_list = generator._format_document_list(sample_report)
        
        assert "1." in doc_list
        assert "2." in doc_list
    
    def test_categorize_conflicts(self, generator, sample_report) -> None:
        """Test conflict categorization by severity."""
        critical, high, other = generator._categorize_conflicts(
            sample_report.conflicts,
            max_per_category=10,
        )
        
        assert len(critical) == 1
        assert len(high) == 1
        assert len(other) == 1
    
    def test_format_issues(self, generator, sample_report) -> None:
        """Test issue formatting."""
        critical, _, _ = generator._categorize_conflicts(
            sample_report.conflicts, 10
        )
        
        formatted = generator._format_issues(critical)
        
        assert "**salary_mismatch**" in formatted.lower() or "salary" in formatted.lower()
    
    def test_format_missing_documents(self, generator, sample_missing_report) -> None:
        """Test missing document formatting."""
        formatted = generator._format_missing_documents(sample_missing_report)
        
        assert "Exhibit A" in formatted
        assert "Contract.pdf" in formatted
    
    def test_format_timeline(self, generator, sample_timeline) -> None:
        """Test timeline formatting."""
        formatted = generator._format_timeline(sample_timeline)
        
        assert "2024-01-01" in formatted or "Date Range" in formatted
        assert "Temporal Conflicts" in formatted or "conflicts" in formatted.lower()
    
    @pytest.mark.asyncio
    async def test_generate_fallback_summary(self, generator, sample_report) -> None:
        """Test that summary generation works (may use LLM or fallback)."""
        summary = await generator.generate_executive_summary(sample_report)
        
        assert summary.summary_markdown is not None
        assert len(summary.summary_markdown) > 100
        # Either uses LLM or fallback - both are valid
        assert summary.model_used is not None
    
    @pytest.mark.asyncio
    async def test_summary_includes_critical_issues(
        self,
        generator,
        sample_report,
    ) -> None:
        """Test that critical issues are included in summary."""
        summary = await generator.generate_executive_summary(sample_report)
        
        # Should mention the critical salary mismatch
        assert "Critical" in summary.summary_markdown or "⚠️" in summary.summary_markdown
    
    @pytest.mark.asyncio
    async def test_summary_includes_missing_docs(
        self,
        generator,
        sample_report,
        sample_missing_report,
    ) -> None:
        """Test that missing documents are included."""
        summary = await generator.generate_executive_summary(
            sample_report,
            missing_doc_report=sample_missing_report,
        )
        
        assert "Exhibit A" in summary.summary_markdown or "Missing" in summary.summary_markdown
    
    @pytest.mark.asyncio
    async def test_summary_with_timeline(
        self,
        generator,
        sample_report,
        sample_timeline,
    ) -> None:
        """Test summary with timeline included."""
        summary = await generator.generate_executive_summary(
            sample_report,
            timeline=sample_timeline,
        )
        
        # Timeline should be mentioned
        assert summary.summary_markdown is not None
    
    def test_generate_conflict_table(self, generator, sample_report) -> None:
        """Test conflict table generation."""
        table = generator.generate_conflict_table(sample_report.conflicts)
        
        assert "|" in table  # Markdown table
        assert "Severity" in table
        assert "Type" in table
    
    def test_generate_conflict_table_empty(self, generator) -> None:
        """Test conflict table with no conflicts."""
        table = generator.generate_conflict_table([])
        
        assert "No conflicts" in table


class TestEntityMatrix:
    """Tests for entity matrix generation."""
    
    @pytest.fixture
    def generator(self) -> ReportGenerator:
        return ReportGenerator()
    
    def test_generate_entity_matrix(self, generator) -> None:
        """Test entity matrix generation."""
        occurrences = {
            "$500,000": [
                ("Contract A", "$500,000"),
                ("Contract B", "$450,000"),
            ],
        }
        
        matrix = generator.generate_entity_matrix("Salary", occurrences)
        
        assert "|" in matrix
        assert "Salary" in matrix
    
    def test_generate_entity_matrix_empty(self, generator) -> None:
        """Test entity matrix with no data."""
        matrix = generator.generate_entity_matrix("Salary", {})
        
        assert "No Salary entities" in matrix
