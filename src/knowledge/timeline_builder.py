"""
Timeline Builder - Chronological Event Extraction.

Builds a timeline of events across documents.
Detects temporal conflicts (e.g., termination before start date).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.knowledge.schemas import Entity, EntityType
from src.knowledge.graph_store import GraphStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Event Type Detection
# ============================================================================

class EventType(str, Enum):
    """Types of timeline events."""
    
    # Employment
    EMPLOYMENT_START = "employment_start"
    EMPLOYMENT_END = "employment_end"
    
    # Documents
    SIGNING = "signing"
    EFFECTIVE_DATE = "effective_date"
    AMENDMENT = "amendment"
    
    # Financial
    VESTING = "vesting"
    PAYMENT = "payment"
    
    # Legal
    EXPIRATION = "expiration"
    TERMINATION = "termination"
    RENEWAL = "renewal"
    
    # General
    OTHER = "other"


# Patterns to detect event types from context
EVENT_PATTERNS = {
    EventType.EMPLOYMENT_START: [
        r"commenc(?:es?|ing)",
        r"start(?:s|ing)?",
        r"begin(?:s|ning)?",
        r"hire\s*date",
        r"employment\s*(?:shall\s*)?begin",
    ],
    EventType.EMPLOYMENT_END: [
        r"terminat(?:es?|ing|ion)",
        r"end(?:s|ing)?",
        r"conclud(?:es?|ing)",
        r"expir(?:es?|ation)",
        r"last\s*day",
    ],
    EventType.EFFECTIVE_DATE: [
        r"effective\s*(?:as\s*of)?",
        r"takes?\s*effect",
        r"becomes?\s*effective",
    ],
    EventType.SIGNING: [
        r"sign(?:ed|ing)?",
        r"execut(?:ed|ion)",
        r"dated\s*(?:as\s*of)?",
    ],
    EventType.VESTING: [
        r"vest(?:s|ing|ed)?",
        r"vesting\s*(?:date|schedule|period)",
    ],
    EventType.AMENDMENT: [
        r"amend(?:ed|ment)?",
        r"modif(?:y|ied|ication)",
        r"restat(?:ed|ement)",
    ],
    EventType.PAYMENT: [
        r"pay(?:able|ment)?",
        r"due\s*(?:on|date)?",
        r"disburs(?:e|ement)",
    ],
    EventType.RENEWAL: [
        r"renew(?:al|ed)?",
        r"extend(?:ed|sion)?",
    ],
    EventType.TERMINATION: [
        r"terminat(?:e|ed|ion)",
        r"cancel(?:led|lation)?",
    ],
}


# ============================================================================
# Schemas
# ============================================================================

class TimelineEvent(BaseModel):
    """An event on the timeline."""
    
    event_id: UUID = Field(default_factory=uuid4)
    event_date: date
    event_type: EventType
    description: str
    
    # Source
    source_document_id: UUID
    source_document_name: str
    source_page: int
    source_text: str = Field(default="", description="Original text context")
    
    # Related entities
    related_entity_ids: list[UUID] = Field(default_factory=list)
    related_entity_values: list[str] = Field(default_factory=list)
    
    # Metadata
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class TimelineConflict(BaseModel):
    """A temporal inconsistency between events."""
    
    conflict_id: UUID = Field(default_factory=uuid4)
    event_a: TimelineEvent
    event_b: TimelineEvent
    
    conflict_type: str = Field(description="Type of temporal conflict")
    description: str
    severity: str = Field(default="medium")  # low, medium, high, critical
    
    @property
    def days_difference(self) -> int:
        """Days between the two events."""
        delta = abs((self.event_a.event_date - self.event_b.event_date).days)
        return delta


class Timeline(BaseModel):
    """Complete timeline from documents."""
    
    timeline_id: UUID = Field(default_factory=uuid4)
    
    # Events
    events: list[TimelineEvent] = Field(default_factory=list)
    
    # Conflicts
    conflicts: list[TimelineConflict] = Field(default_factory=list)
    
    # Date range
    earliest_date: date | None = None
    latest_date: date | None = None
    
    # Metadata
    document_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def event_count(self) -> int:
        return len(self.events)
    
    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)
    
    @property
    def span_days(self) -> int | None:
        """Total days spanned by timeline."""
        if self.earliest_date and self.latest_date:
            return (self.latest_date - self.earliest_date).days
        return None
    
    def get_events_by_type(self, event_type: EventType) -> list[TimelineEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_for_entity(self, entity_value: str) -> list[TimelineEvent]:
        """Get all events involving an entity."""
        return [
            e for e in self.events 
            if entity_value.lower() in [v.lower() for v in e.related_entity_values]
        ]


# ============================================================================
# Timeline Builder
# ============================================================================

class TimelineBuilder:
    """
    Build and analyze document timelines.
    
    Extracts:
    - Date entities from knowledge graph
    - Event types from surrounding context
    - Relationships between events
    
    Detects:
    - Temporal conflicts (impossible sequences)
    - Date mismatches across documents
    
    Usage:
        builder = TimelineBuilder(graph_store)
        timeline = builder.build_timeline(document_ids)
        conflicts = builder.detect_conflicts(timeline)
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        document_names: dict[UUID, str] | None = None,
    ) -> None:
        """
        Initialize the Timeline Builder.
        
        Args:
            graph_store: Graph store with entities
            document_names: Optional mapping of doc IDs to names
        """
        self.graph_store = graph_store
        self.document_names = document_names or {}
        
        # Compile event patterns
        self._event_patterns: dict[EventType, list[re.Pattern]] = {}
        for event_type, patterns in EVENT_PATTERNS.items():
            self._event_patterns[event_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def build_timeline(
        self,
        document_ids: list[UUID],
    ) -> Timeline:
        """
        Build chronological timeline from documents.
        
        Args:
            document_ids: List of document IDs to include
            
        Returns:
            Timeline with events and conflicts
        """
        events: list[TimelineEvent] = []
        
        # Get all DATE entities from documents
        for doc_id in document_ids:
            doc_name = self.document_names.get(doc_id, f"Document {str(doc_id)[:8]}")
            
            date_entities = self.graph_store.find_entities_by_type(
                EntityType.DATE,
                document_id=doc_id,
            )
            
            for node in date_entities:
                entity = node.entity
                
                # Parse the date
                parsed_date = self._parse_date(entity.value)
                if not parsed_date:
                    continue
                
                # Determine event type from context
                event_type = self._detect_event_type(entity.source_text)
                
                # Get related entities
                neighborhood = self.graph_store.get_entity_neighborhood(
                    entity.entity_id,
                    hops=1,
                    max_nodes=10,
                )
                
                related_ids = [n.entity.entity_id for n in neighborhood.nodes if n.id != str(entity.entity_id)]
                related_values = [n.entity.value for n in neighborhood.nodes if n.id != str(entity.entity_id)]
                
                event = TimelineEvent(
                    event_date=parsed_date,
                    event_type=event_type,
                    description=self._generate_description(entity, event_type),
                    source_document_id=doc_id,
                    source_document_name=doc_name,
                    source_page=entity.source_page,
                    source_text=entity.source_text[:200],
                    related_entity_ids=related_ids,
                    related_entity_values=related_values,
                    confidence=entity.confidence,
                )
                
                events.append(event)
        
        # Sort by date
        events.sort(key=lambda e: e.event_date)
        
        # Calculate date range
        earliest = events[0].event_date if events else None
        latest = events[-1].event_date if events else None
        
        # Build timeline
        timeline = Timeline(
            events=events,
            earliest_date=earliest,
            latest_date=latest,
            document_count=len(document_ids),
        )
        
        # Detect conflicts
        timeline.conflicts = self.detect_timeline_conflicts(timeline)
        
        logger.info(
            f"Built timeline: {timeline.event_count} events, "
            f"{timeline.conflict_count} conflicts"
        )
        
        return timeline
    
    def detect_timeline_conflicts(
        self,
        timeline: Timeline,
    ) -> list[TimelineConflict]:
        """
        Find temporal inconsistencies in the timeline.
        
        Looks for:
        - Termination before start
        - End before effective date
        - Same event type with different dates across docs
        """
        conflicts: list[TimelineConflict] = []
        
        events = timeline.events
        
        # 1. Check for impossible sequences within same document
        for i, event_a in enumerate(events):
            for event_b in events[i + 1:]:
                # Only compare within same document
                if event_a.source_document_id != event_b.source_document_id:
                    continue
                
                conflict = self._check_sequence_conflict(event_a, event_b)
                if conflict:
                    conflicts.append(conflict)
        
        # 2. Check for same event type with different dates across docs
        by_type: dict[EventType, list[TimelineEvent]] = {}
        for event in events:
            by_type.setdefault(event.event_type, []).append(event)
        
        for event_type, type_events in by_type.items():
            if len(type_events) <= 1:
                continue
            
            # Check if same event type has different dates across docs
            unique_dates = set((e.event_date, e.source_document_id) for e in type_events)
            
            if len(unique_dates) > 1:
                # Check if dates actually conflict
                dates = set(e.event_date for e in type_events)
                if len(dates) > 1:
                    # Different dates for same event type = potential conflict
                    conflict = self._create_date_mismatch_conflict(type_events)
                    if conflict:
                        conflicts.append(conflict)
        
        return conflicts
    
    def _check_sequence_conflict(
        self,
        event_a: TimelineEvent,
        event_b: TimelineEvent,
    ) -> TimelineConflict | None:
        """Check if two events have an impossible sequence."""
        # Termination before start
        if event_a.event_type == EventType.EMPLOYMENT_END and \
           event_b.event_type == EventType.EMPLOYMENT_START:
            return TimelineConflict(
                event_a=event_a,
                event_b=event_b,
                conflict_type="impossible_sequence",
                description=(
                    f"Termination date ({event_a.event_date}) is before "
                    f"start date ({event_b.event_date})"
                ),
                severity="critical",
            )
        
        # Expiration before effective
        if event_a.event_type == EventType.EXPIRATION and \
           event_b.event_type == EventType.EFFECTIVE_DATE:
            return TimelineConflict(
                event_a=event_a,
                event_b=event_b,
                conflict_type="impossible_sequence",
                description=(
                    f"Expiration date ({event_a.event_date}) is before "
                    f"effective date ({event_b.event_date})"
                ),
                severity="critical",
            )
        
        return None
    
    def _create_date_mismatch_conflict(
        self,
        events: list[TimelineEvent],
    ) -> TimelineConflict | None:
        """Create a conflict for date mismatches across documents."""
        if len(events) < 2:
            return None
        
        # Get unique dates
        dates = sorted(set(e.event_date for e in events))
        if len(dates) < 2:
            return None
        
        # Use first and last as the conflicting pair
        first_event = next(e for e in events if e.event_date == dates[0])
        last_event = next(e for e in events if e.event_date == dates[-1])
        
        days_diff = (dates[-1] - dates[0]).days
        
        # Determine severity based on difference
        if days_diff > 365:
            severity = "critical"
        elif days_diff > 90:
            severity = "high"
        elif days_diff > 30:
            severity = "medium"
        else:
            severity = "low"
        
        return TimelineConflict(
            event_a=first_event,
            event_b=last_event,
            conflict_type="date_mismatch",
            description=(
                f"Same event type ({first_event.event_type.value}) has "
                f"different dates: {dates[0]} vs {dates[-1]} "
                f"({days_diff} days apart)"
            ),
            severity=severity,
        )
    
    def _parse_date(self, value: str) -> date | None:
        """Parse a date string into a date object."""
        if not value:
            return None
        
        # Common date formats
        formats = [
            "%Y-%m-%d",           # 2024-01-15
            "%B %d, %Y",          # January 15, 2024
            "%b %d, %Y",          # Jan 15, 2024
            "%d %B %Y",           # 15 January 2024
            "%m/%d/%Y",           # 01/15/2024
            "%d/%m/%Y",           # 15/01/2024
            "%m-%d-%Y",           # 01-15-2024
            "%Y",                 # 2024 (assume Jan 1)
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
        
        # Try to extract year only
        year_match = re.search(r"\b(19|20)\d{2}\b", value)
        if year_match:
            try:
                return date(int(year_match.group(0)), 1, 1)
            except ValueError:
                pass
        
        return None
    
    def _detect_event_type(self, context: str) -> EventType:
        """Detect event type from surrounding context."""
        if not context:
            return EventType.OTHER
        
        context_lower = context.lower()
        
        for event_type, patterns in self._event_patterns.items():
            for pattern in patterns:
                if pattern.search(context_lower):
                    return event_type
        
        return EventType.OTHER
    
    def _generate_description(
        self,
        entity: Entity,
        event_type: EventType,
    ) -> str:
        """Generate a human-readable description for an event."""
        date_str = entity.value
        
        descriptions = {
            EventType.EMPLOYMENT_START: f"Employment begins: {date_str}",
            EventType.EMPLOYMENT_END: f"Employment ends: {date_str}",
            EventType.EFFECTIVE_DATE: f"Effective date: {date_str}",
            EventType.SIGNING: f"Document signed: {date_str}",
            EventType.VESTING: f"Vesting date: {date_str}",
            EventType.AMENDMENT: f"Amendment date: {date_str}",
            EventType.PAYMENT: f"Payment due: {date_str}",
            EventType.RENEWAL: f"Renewal date: {date_str}",
            EventType.TERMINATION: f"Termination date: {date_str}",
            EventType.EXPIRATION: f"Expiration date: {date_str}",
            EventType.OTHER: f"Date: {date_str}",
        }
        
        return descriptions.get(event_type, f"Event on {date_str}")
    
    def get_events_for_entity(
        self,
        entity_id: UUID,
        timeline: Timeline | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[TimelineEvent]:
        """
        Get all timeline events involving a specific entity.
        
        Args:
            entity_id: Entity to find events for
            timeline: Existing timeline (or pass document_ids)
            document_ids: Documents to search (if no timeline)
            
        Returns:
            List of TimelineEvent involving the entity
        """
        if timeline is None and document_ids:
            timeline = self.build_timeline(document_ids)
        
        if timeline is None:
            return []
        
        return [
            event for event in timeline.events
            if entity_id in event.related_entity_ids
        ]
