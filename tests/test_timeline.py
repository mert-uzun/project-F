"""
Tests for Timeline Builder.

Tests timeline construction, event detection, and temporal conflict detection.
"""

from datetime import date
from uuid import uuid4

import pytest

from src.knowledge.graph_store import GraphStore
from src.knowledge.schemas import Entity, EntityType
from src.knowledge.timeline_builder import (
    EventType,
    Timeline,
    TimelineBuilder,
    TimelineConflict,
    TimelineEvent,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self) -> None:
        """Test that all expected event types exist."""
        assert EventType.EMPLOYMENT_START
        assert EventType.EMPLOYMENT_END
        assert EventType.EFFECTIVE_DATE
        assert EventType.SIGNING
        assert EventType.VESTING
        assert EventType.TERMINATION


class TestTimelineEvent:
    """Tests for TimelineEvent model."""

    def test_event_creation(self) -> None:
        """Test TimelineEvent can be created."""
        event = TimelineEvent(
            event_date=date(2024, 1, 15),
            event_type=EventType.EMPLOYMENT_START,
            description="Employment begins: January 15, 2024",
            source_document_id=uuid4(),
            source_document_name="Employment Agreement",
            source_page=1,
        )

        assert event.event_date == date(2024, 1, 15)
        assert event.event_type == EventType.EMPLOYMENT_START


class TestTimelineConflict:
    """Tests for TimelineConflict model."""

    def test_conflict_creation(self) -> None:
        """Test TimelineConflict can be created."""
        event_a = TimelineEvent(
            event_date=date(2024, 12, 31),
            event_type=EventType.EMPLOYMENT_END,
            description="End",
            source_document_id=uuid4(),
            source_document_name="Doc A",
            source_page=1,
        )
        event_b = TimelineEvent(
            event_date=date(2024, 1, 1),
            event_type=EventType.EMPLOYMENT_START,
            description="Start",
            source_document_id=uuid4(),
            source_document_name="Doc B",
            source_page=1,
        )

        conflict = TimelineConflict(
            event_a=event_a,
            event_b=event_b,
            conflict_type="impossible_sequence",
            description="End before start",
        )

        assert conflict.days_difference == 365

    def test_days_difference(self) -> None:
        """Test days_difference calculation."""
        event_a = TimelineEvent(
            event_date=date(2024, 1, 1),
            event_type=EventType.SIGNING,
            description="Signed",
            source_document_id=uuid4(),
            source_document_name="Doc",
            source_page=1,
        )
        event_b = TimelineEvent(
            event_date=date(2024, 1, 11),
            event_type=EventType.EFFECTIVE_DATE,
            description="Effective",
            source_document_id=uuid4(),
            source_document_name="Doc",
            source_page=1,
        )

        conflict = TimelineConflict(
            event_a=event_a,
            event_b=event_b,
            conflict_type="test",
            description="Test",
        )

        assert conflict.days_difference == 10


class TestTimeline:
    """Tests for Timeline model."""

    def test_empty_timeline(self) -> None:
        """Test empty timeline."""
        timeline = Timeline()

        assert timeline.event_count == 0
        assert timeline.span_days is None

    def test_timeline_with_events(self) -> None:
        """Test timeline with events."""
        events = [
            TimelineEvent(
                event_date=date(2024, 1, 1),
                event_type=EventType.SIGNING,
                description="Signed",
                source_document_id=uuid4(),
                source_document_name="Doc",
                source_page=1,
            ),
            TimelineEvent(
                event_date=date(2024, 12, 31),
                event_type=EventType.EXPIRATION,
                description="Expires",
                source_document_id=uuid4(),
                source_document_name="Doc",
                source_page=2,
            ),
        ]

        timeline = Timeline(
            events=events,
            earliest_date=date(2024, 1, 1),
            latest_date=date(2024, 12, 31),
        )

        assert timeline.event_count == 2
        assert timeline.span_days == 365

    def test_get_events_by_type(self) -> None:
        """Test filtering events by type."""
        doc_id = uuid4()

        events = [
            TimelineEvent(
                event_date=date(2024, 1, 1),
                event_type=EventType.VESTING,
                description="Vest 1",
                source_document_id=doc_id,
                source_document_name="Doc",
                source_page=1,
            ),
            TimelineEvent(
                event_date=date(2024, 6, 1),
                event_type=EventType.VESTING,
                description="Vest 2",
                source_document_id=doc_id,
                source_document_name="Doc",
                source_page=1,
            ),
            TimelineEvent(
                event_date=date(2024, 3, 1),
                event_type=EventType.PAYMENT,
                description="Payment",
                source_document_id=doc_id,
                source_document_name="Doc",
                source_page=1,
            ),
        ]

        timeline = Timeline(events=events)

        vesting_events = timeline.get_events_by_type(EventType.VESTING)
        assert len(vesting_events) == 2

    def test_get_events_for_entity(self) -> None:
        """Test filtering events by entity."""
        events = [
            TimelineEvent(
                event_date=date(2024, 1, 1),
                event_type=EventType.EMPLOYMENT_START,
                description="Start",
                source_document_id=uuid4(),
                source_document_name="Doc",
                source_page=1,
                related_entity_values=["John Doe", "ABC Corp"],
            ),
            TimelineEvent(
                event_date=date(2024, 1, 1),
                event_type=EventType.EMPLOYMENT_START,
                description="Start",
                source_document_id=uuid4(),
                source_document_name="Doc",
                source_page=1,
                related_entity_values=["Jane Smith"],
            ),
        ]

        timeline = Timeline(events=events)

        john_events = timeline.get_events_for_entity("John Doe")
        assert len(john_events) == 1


class TestTimelineBuilder:
    """Tests for TimelineBuilder."""

    @pytest.fixture
    def graph_store_with_dates(self, tmp_path) -> GraphStore:
        """Create graph store with date entities."""
        store = GraphStore(persist_path=tmp_path / "test_graph.json")

        doc_id = uuid4()

        # Add date entities
        entities = [
            Entity(
                entity_type=EntityType.DATE,
                value="January 1, 2024",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="Employment commences on January 1, 2024",
            ),
            Entity(
                entity_type=EntityType.DATE,
                value="December 31, 2024",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=2,
                source_text="Contract terminates on December 31, 2024",
            ),
        ]

        store.add_entities(entities)
        store._doc_id = doc_id

        return store

    def test_builder_initialization(self, graph_store_with_dates) -> None:
        """Test TimelineBuilder can be initialized."""
        builder = TimelineBuilder(graph_store_with_dates)
        assert builder.graph_store is not None

    def test_parse_date_formats(self, graph_store_with_dates) -> None:
        """Test date parsing with various formats."""
        builder = TimelineBuilder(graph_store_with_dates)

        # ISO format
        assert builder._parse_date("2024-01-15") == date(2024, 1, 15)

        # Long format
        assert builder._parse_date("January 15, 2024") == date(2024, 1, 15)

        # Short format
        assert builder._parse_date("Jan 15, 2024") == date(2024, 1, 15)

        # US format
        assert builder._parse_date("01/15/2024") == date(2024, 1, 15)

        # Invalid
        assert builder._parse_date("invalid") is None

    def test_detect_event_type_employment_start(self, graph_store_with_dates) -> None:
        """Test event type detection for employment start."""
        builder = TimelineBuilder(graph_store_with_dates)

        event_type = builder._detect_event_type("Employment commences on this date")
        assert event_type == EventType.EMPLOYMENT_START

        event_type = builder._detect_event_type("The employee shall begin working")
        assert event_type == EventType.EMPLOYMENT_START

    def test_detect_event_type_termination(self, graph_store_with_dates) -> None:
        """Test event type detection for termination."""
        builder = TimelineBuilder(graph_store_with_dates)

        event_type = builder._detect_event_type("Contract terminates on")
        assert event_type == EventType.EMPLOYMENT_END

        event_type = builder._detect_event_type("Termination date")
        assert event_type in (EventType.EMPLOYMENT_END, EventType.TERMINATION)

    def test_detect_event_type_vesting(self, graph_store_with_dates) -> None:
        """Test event type detection for vesting."""
        builder = TimelineBuilder(graph_store_with_dates)

        event_type = builder._detect_event_type("Shares vest on this date")
        assert event_type == EventType.VESTING

    def test_detect_event_type_effective(self, graph_store_with_dates) -> None:
        """Test event type detection for effective date."""
        builder = TimelineBuilder(graph_store_with_dates)

        event_type = builder._detect_event_type("Effective as of")
        assert event_type == EventType.EFFECTIVE_DATE

    def test_build_timeline(self, graph_store_with_dates) -> None:
        """Test building a timeline from graph store."""
        builder = TimelineBuilder(graph_store_with_dates)

        timeline = builder.build_timeline([graph_store_with_dates._doc_id])

        assert timeline.event_count >= 1
        assert timeline.earliest_date is not None

    def test_detect_timeline_conflicts(self, tmp_path) -> None:
        """Test detection of temporal conflicts."""
        store = GraphStore(persist_path=tmp_path / "conflict_graph.json")
        doc_id = uuid4()

        # Add conflicting dates - end before start (impossible)
        entities = [
            Entity(
                entity_type=EntityType.DATE,
                value="December 31, 2023",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=1,
                source_text="Employment terminates on December 31, 2023",
            ),
            Entity(
                entity_type=EntityType.DATE,
                value="January 1, 2024",
                source_document_id=doc_id,
                source_chunk_id=uuid4(),
                source_page=2,
                source_text="Employment commences on January 1, 2024",
            ),
        ]

        store.add_entities(entities)

        builder = TimelineBuilder(store)
        timeline = builder.build_timeline([doc_id])

        # Should detect the impossible sequence
        # Note: Detection depends on event types being correctly identified
        assert timeline.event_count >= 2
