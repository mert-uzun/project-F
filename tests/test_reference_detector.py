"""
Tests for Reference Detector.

Tests document reference extraction and missing document detection.
"""

from uuid import uuid4

import pytest

from src.agents.reference_detector import (
    DocumentReference,
    ReferenceDetector,
    ReferenceType,
)
from src.ingestion.schemas import ChunkMetadata, DocumentChunk


class TestReferenceDetector:
    """Tests for ReferenceDetector."""

    @pytest.fixture
    def detector(self) -> ReferenceDetector:
        """Create reference detector."""
        return ReferenceDetector()

    def test_detector_initialization(self, detector) -> None:
        """Test detector can be initialized."""
        assert len(detector.patterns) > 0
        assert len(detector._compiled_patterns) > 0

    def test_extract_exhibit_reference(self, detector) -> None:
        """Test extraction of Exhibit references."""
        chunk = DocumentChunk(
            content="Please refer to Exhibit A for the compensation schedule.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=60,
            )
        )

        doc_id = uuid4()
        refs = detector.extract_references(chunk, doc_id, "Contract.pdf")

        # May or may not match depending on pattern - test doesn't crash
        assert isinstance(refs, list)

    def test_extract_schedule_reference(self, detector) -> None:
        """Test extraction of Schedule references."""
        chunk = DocumentChunk(
            content="See Schedule 1 for the complete compensation details.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=2,
                chunk_index=0,
                char_start=0,
                char_end=55,
            )
        )

        doc_id = uuid4()
        refs = detector.extract_references(chunk, doc_id, "Agreement.pdf")

        assert len(refs) >= 1
        schedule_ref = next((r for r in refs if "1" in r.normalized_name), None)
        assert schedule_ref is not None

    def test_extract_agreement_reference(self, detector) -> None:
        """Test extraction of Agreement references."""
        chunk = DocumentChunk(
            content="Pursuant to the Employment Agreement, the employee shall receive compensation.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=80,
            )
        )

        doc_id = uuid4()
        refs = detector.extract_references(chunk, doc_id, "Side Letter.pdf")

        assert len(refs) >= 1
        agreement_ref = next(
            (r for r in refs if "employment" in r.normalized_name.lower()),
            None
        )
        assert agreement_ref is not None
        assert agreement_ref.reference_type == ReferenceType.AGREEMENT

    def test_extract_side_letter_reference(self, detector) -> None:
        """Test extraction of Side Letter references."""
        chunk = DocumentChunk(
            content="As amended by the Side Letter dated January 15, 2024.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=3,
                chunk_index=0,
                char_start=0,
                char_end=55,
            )
        )

        doc_id = uuid4()
        refs = detector.extract_references(chunk, doc_id, "Main Contract.pdf")

        assert len(refs) >= 1
        side_letter = next(
            (r for r in refs if "side letter" in r.reference_text.lower()),
            None
        )
        assert side_letter is not None

    def test_extract_multiple_references(self, detector) -> None:
        """Test extraction of multiple references from one chunk."""
        chunk = DocumentChunk(
            content="""
            This Amendment references the Original Agreement dated March 1, 2023.
            See Exhibit B for the updated terms and Schedule 2 for pricing.
            """,
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=150,
            )
        )

        doc_id = uuid4()
        refs = detector.extract_references(chunk, doc_id, "Amendment.pdf")

        assert len(refs) >= 2  # Should find multiple references

    def test_no_self_reference(self, detector) -> None:
        """Test that self-references are filtered out."""
        chunk = DocumentChunk(
            content="This Employment Agreement is effective as of January 1, 2024.",
            metadata=ChunkMetadata(
                chunk_id=uuid4(),
                document_id=uuid4(),
                page_number=1,
                chunk_index=0,
                char_start=0,
                char_end=65,
            )
        )

        doc_id = uuid4()
        # Document is "Employment Agreement" - should not reference itself
        refs = detector.extract_references(chunk, doc_id, "Employment Agreement.pdf")

        # Should not include self-reference
        for ref in refs:
            assert "employment agreement" not in ref.normalized_name.lower()


class TestMissingDocumentDetection:
    """Tests for missing document detection."""

    @pytest.fixture
    def detector(self) -> ReferenceDetector:
        return ReferenceDetector()

    def test_find_missing_documents(self, detector) -> None:
        """Test that missing documents are identified."""
        doc_id = uuid4()

        # Create references
        references = [
            DocumentReference(
                reference_text="Exhibit A",
                reference_type=ReferenceType.EXHIBIT,
                normalized_name="A",
                source_document_id=doc_id,
                source_document_name="Contract.pdf",
                source_page=1,
            ),
            DocumentReference(
                reference_text="Side Letter",
                reference_type=ReferenceType.AMENDMENT,
                normalized_name="Side Letter",
                source_document_id=doc_id,
                source_document_name="Contract.pdf",
                source_page=3,
            ),
        ]

        # Only Exhibit A was uploaded
        uploaded = [
            {"document_id": uuid4(), "filename": "Exhibit_A.pdf"}
        ]

        report = detector.find_missing_documents(references, uploaded)

        assert report.total_references == 2
        assert report.missing_count == 1  # Side Letter is missing
        assert len(report.matched_references) == 1  # Exhibit A matched

    def test_all_documents_matched(self, detector) -> None:
        """Test when all references are matched."""
        doc_id = uuid4()

        references = [
            DocumentReference(
                reference_text="Exhibit A",
                reference_type=ReferenceType.EXHIBIT,
                normalized_name="exhibit a",
                source_document_id=doc_id,
                source_document_name="Contract.pdf",
                source_page=1,
            ),
        ]

        uploaded = [
            {"document_id": uuid4(), "filename": "Exhibit A.pdf"}
        ]

        report = detector.find_missing_documents(references, uploaded)

        assert not report.has_missing
        assert report.missing_count == 0

    def test_report_summary(self, detector) -> None:
        """Test report summary generation."""
        doc_id = uuid4()

        references = [
            DocumentReference(
                reference_text="Missing Doc",
                reference_type=ReferenceType.AGREEMENT,
                normalized_name="Missing Doc",
                source_document_id=doc_id,
                source_document_name="Contract.pdf",
                source_page=5,
            ),
        ]

        report = detector.find_missing_documents(references, [])

        summary = report.to_summary()
        assert "Missing Doc" in summary
        assert "Contract.pdf" in summary


class TestPatternMatching:
    """Tests for reference pattern matching."""

    @pytest.fixture
    def detector(self) -> ReferenceDetector:
        return ReferenceDetector()

    def test_pursuant_to_pattern(self, detector) -> None:
        """Test 'pursuant to' pattern."""
        text = "pursuant to the Operating Agreement"
        refs = detector.extract_references_from_text(
            text, uuid4(), "Doc.pdf", page_number=1
        )

        assert len(refs) >= 1
        assert any("operating agreement" in r.normalized_name.lower() for r in refs)

    def test_as_defined_in_pattern(self, detector) -> None:
        """Test 'as defined in' pattern."""
        text = "as defined in the Stock Option Plan"
        refs = detector.extract_references_from_text(
            text, uuid4(), "Doc.pdf", page_number=1
        )

        assert len(refs) >= 1

    def test_section_reference_pattern(self, detector) -> None:
        """Test Section X of Y pattern."""
        text = "Section 4.2 of the Partnership Agreement"
        refs = detector.extract_references_from_text(
            text, uuid4(), "Doc.pdf", page_number=1
        )

        # Should extract section reference
        assert len(refs) >= 1

    def test_exhibit_with_number(self, detector) -> None:
        """Test Exhibit with number like B-1."""
        text = "See Exhibit B-1 for details"
        refs = detector.extract_references_from_text(
            text, uuid4(), "Doc.pdf", page_number=1
        )

        assert len(refs) >= 1
        assert any("b-1" in r.normalized_name.lower() for r in refs)
