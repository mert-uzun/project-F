"""
Graph Visualizer - Interactive Knowledge Graph Visualization.

Generates PyVis HTML visualizations with investment banking-focused
color schemes designed for quick scanning and attention prioritization.

Color Psychology for Investment Bankers:
- They're time-pressured, need to spot issues FAST
- Used to RAG (Red-Amber-Green) traffic light systems
- Bloomberg terminal aesthetics (high contrast)
- Critical info must "pop" immediately
"""

from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID

from pyvis.network import Network

from src.knowledge.schemas import Entity, EntityType, Relationship, RelationshipType
from src.knowledge.graph_store import GraphStore, GraphNode, GraphEdge
from src.utils.logger import get_logger

# Import ConflictSeverity lazily to avoid circular import
if TYPE_CHECKING:
    from src.agents.schemas import ConflictSeverity as ConflictSeverityType

logger = get_logger(__name__)


# Local definition to avoid circular import
class _Severity(str, Enum):
    """Conflict severity levels (local copy to avoid circular import)."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Color Scheme - Investment Banking Focused
# ============================================================================

# Severity-based colors (RAG system - Red/Amber/Green)
# These draw immediate attention in order of priority
SEVERITY_COLORS = {
    ConflictSeverity.CRITICAL: "#DC2626",  # Vivid red - STOP, urgent
    ConflictSeverity.HIGH: "#EA580C",      # Deep orange - serious warning
    ConflictSeverity.MEDIUM: "#CA8A04",    # Amber/gold - notable, review
    ConflictSeverity.LOW: "#65A30D",       # Olive green - minor, low priority
}

# Entity type colors - Professional, high contrast
# Based on color psychology for business/finance context
ENTITY_COLORS = {
    # People - Professional blue (trust, human element)
    EntityType.PERSON: "#0284C7",
    
    # Organizations - Navy/dark blue (corporate, stable, reliable)
    EntityType.ORGANIZATION: "#1E3A5F",
    
    # Financial - Bronze/gold (money, value, importance)
    EntityType.MONETARY_AMOUNT: "#B45309",
    EntityType.SALARY: "#B45309",
    EntityType.EQUITY: "#B45309",
    EntityType.PERCENTAGE: "#D97706",
    
    # Dates/Time - Deep purple (milestones, deadlines)
    EntityType.DATE: "#6B21A8",
    EntityType.DURATION: "#7C3AED",
    
    # Legal/Terms - Slate gray (neutral, supporting)
    EntityType.TERM: "#475569",
    EntityType.CLAUSE: "#475569",
    EntityType.ROLE: "#64748B",
    
    # General - Light gray
    EntityType.OTHER: "#94A3B8",
}

# Relationship colors
RELATIONSHIP_COLORS = {
    # Conflict - RED, must stand out
    RelationshipType.CONFLICTS_WITH: "#DC2626",
    
    # Financial relationships - Gold (money)
    RelationshipType.HAS_SALARY: "#B45309",
    RelationshipType.HAS_EQUITY: "#B45309",
    RelationshipType.OWNS: "#B45309",
    
    # Organizational - Blue (hierarchy)  
    RelationshipType.EMPLOYED_BY: "#0284C7",
    RelationshipType.REPORTS_TO: "#0284C7",
    RelationshipType.HAS_ROLE: "#0284C7",
    
    # References - Light gray (supporting)
    RelationshipType.REFERENCES: "#94A3B8",
    RelationshipType.CONTAINS: "#94A3B8",
    RelationshipType.HAS_SECTION: "#94A3B8",
    
    # Comparison - Amber
    RelationshipType.SAME_AS: "#CA8A04",
    RelationshipType.SUPERSEDES: "#CA8A04",
}

# Edge styles
CONFLICT_EDGE_STYLE = {
    "color": "#DC2626",
    "width": 4,
    "dashes": True,
}

NORMAL_EDGE_STYLE = {
    "color": "#64748B",
    "width": 2,
    "dashes": False,
}


# ============================================================================
# Graph Visualizer
# ============================================================================

class GraphVisualizer:
    """
    Generate interactive graph visualizations using PyVis.
    
    Designed for investment banking due diligence:
    - High contrast colors for quick scanning
    - RAG severity indicators
    - Conflict edges stand out immediately
    - Professional, not "toy-like" aesthetics
    
    Usage:
        viz = GraphVisualizer(graph_store)
        viz.generate_network(document_ids=[...])
        viz.save_html(Path("graph.html"))
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        conflict_entities: set[UUID] | None = None,
    ) -> None:
        """
        Initialize the Graph Visualizer.
        
        Args:
            graph_store: Graph store with entities/relationships
            conflict_entities: Optional set of entity IDs involved in conflicts
        """
        self.graph_store = graph_store
        self.conflict_entities = conflict_entities or set()
        self._network: Network | None = None
    
    def generate_network(
        self,
        document_ids: list[UUID] | None = None,
        entity_types: list[EntityType] | None = None,
        highlight_conflicts: bool = True,
        max_nodes: int = 200,
        physics_enabled: bool = True,
    ) -> Network:
        """
        Generate a PyVis network from the knowledge graph.
        
        Args:
            document_ids: Optional filter by documents
            entity_types: Optional filter by entity types
            highlight_conflicts: Whether to highlight conflict nodes
            max_nodes: Maximum nodes to include
            physics_enabled: Enable physics simulation
            
        Returns:
            PyVis Network object
        """
        # Create network with professional styling
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#FAFAFA",  # Light gray background
            font_color="#1E293B",  # Dark slate text
            directed=True,
        )
        
        # Get entities
        if document_ids:
            entities_by_doc = self.graph_store.get_entities_by_document(document_ids)
            all_nodes = []
            for nodes in entities_by_doc.values():
                all_nodes.extend(nodes)
        else:
            all_nodes = self.graph_store.get_all_entities()
        
        # Filter by entity types
        if entity_types:
            all_nodes = [n for n in all_nodes if n.entity.entity_type in entity_types]
        
        # Limit nodes
        all_nodes = all_nodes[:max_nodes]
        
        # Add nodes
        node_ids: set[str] = set()
        
        for graph_node in all_nodes:
            entity = graph_node.entity
            node_id = str(entity.entity_id)
            
            if node_id in node_ids:
                continue
            node_ids.add(node_id)
            
            # Determine color
            is_conflict = entity.entity_id in self.conflict_entities
            
            if is_conflict and highlight_conflicts:
                color = SEVERITY_COLORS[ConflictSeverity.CRITICAL]
                border_width = 4
            else:
                color = ENTITY_COLORS.get(entity.entity_type, "#94A3B8")
                border_width = 2
            
            # Determine size based on entity type importance
            size = self._get_node_size(entity.entity_type, is_conflict)
            
            # Build tooltip
            title = self._build_tooltip(entity)
            
            # Truncate label for display
            label = self._truncate_label(entity.value)
            
            net.add_node(
                node_id,
                label=label,
                title=title,
                color={
                    "background": color,
                    "border": "#1E293B" if is_conflict else color,
                    "highlight": {
                        "background": "#FEF08A",
                        "border": "#CA8A04",
                    },
                },
                size=size,
                borderWidth=border_width,
                font={"size": 14, "face": "Inter, sans-serif"},
            )
        
        # Add edges
        for source, target, edge_data in self.graph_store.graph.edges(data=True):
            if source not in node_ids or target not in node_ids:
                continue
            
            rel_type = edge_data.get("relationship_type", RelationshipType.REFERENCES)
            if isinstance(rel_type, str):
                try:
                    rel_type = RelationshipType(rel_type)
                except ValueError:
                    rel_type = RelationshipType.REFERENCES
            
            # Style based on relationship
            if rel_type == RelationshipType.CONFLICTS_WITH:
                edge_color = CONFLICT_EDGE_STYLE["color"]
                edge_width = CONFLICT_EDGE_STYLE["width"]
                dashes = CONFLICT_EDGE_STYLE["dashes"]
            else:
                edge_color = RELATIONSHIP_COLORS.get(rel_type, NORMAL_EDGE_STYLE["color"])
                edge_width = NORMAL_EDGE_STYLE["width"]
                dashes = NORMAL_EDGE_STYLE["dashes"]
            
            net.add_edge(
                source,
                target,
                title=rel_type.value,
                color=edge_color,
                width=edge_width,
                dashes=dashes,
                arrows="to",
            )
        
        # Configure physics
        if physics_enabled:
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -30000,
                        "centralGravity": 0.3,
                        "springLength": 200,
                        "springConstant": 0.04,
                        "damping": 0.09
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 100,
                    "navigationButtons": true,
                    "keyboard": {
                        "enabled": true
                    }
                },
                "nodes": {
                    "shape": "dot",
                    "scaling": {
                        "min": 10,
                        "max": 50
                    }
                },
                "edges": {
                    "smooth": {
                        "type": "continuous"
                    }
                }
            }
            """)
        else:
            net.toggle_physics(False)
        
        self._network = net
        
        logger.info(
            f"Generated network: {len(node_ids)} nodes, "
            f"{len(list(self.graph_store.graph.edges))} edges"
        )
        
        return net
    
    def save_html(
        self,
        output_path: Path,
        include_legend: bool = True,
    ) -> Path:
        """
        Save the network as an interactive HTML file.
        
        Args:
            output_path: Where to save the HTML
            include_legend: Whether to include a color legend
            
        Returns:
            Path to saved file
        """
        if self._network is None:
            self.generate_network()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save base graph
        self._network.save_graph(str(output_path))
        
        # Inject legend if requested
        if include_legend:
            self._inject_legend(output_path)
        
        logger.info(f"Saved graph visualization to {output_path}")
        
        return output_path
    
    def _get_node_size(self, entity_type: EntityType, is_conflict: bool) -> int:
        """Determine node size based on type and conflict status."""
        base_sizes = {
            EntityType.PERSON: 35,
            EntityType.ORGANIZATION: 40,
            EntityType.MONETARY_AMOUNT: 30,
            EntityType.SALARY: 30,
            EntityType.EQUITY: 30,
            EntityType.DATE: 25,
            EntityType.CLAUSE: 20,
            EntityType.TERM: 20,
        }
        
        base = base_sizes.get(entity_type, 25)
        
        # Conflicts are larger
        if is_conflict:
            return base + 15
        
        return base
    
    def _build_tooltip(self, entity: Entity) -> str:
        """Build HTML tooltip for a node."""
        lines = [
            f"<b>{entity.value}</b>",
            f"<em>Type:</em> {entity.entity_type.value}",
            f"<em>Page:</em> {entity.source_page}",
        ]
        
        if entity.confidence < 1.0:
            lines.append(f"<em>Confidence:</em> {entity.confidence:.0%}")
        
        if entity.source_text:
            # Truncate context
            context = entity.source_text[:100]
            lines.append(f"<br/><em>Context:</em> {context}...")
        
        return "<br/>".join(lines)
    
    def _truncate_label(self, value: str, max_length: int = 25) -> str:
        """Truncate label for display."""
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."
    
    def _inject_legend(self, html_path: Path) -> None:
        """Inject a color legend into the HTML."""
        legend_html = """
        <div id="graph-legend" style="
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            font-family: Inter, sans-serif;
            font-size: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 1000;
        ">
            <div style="font-weight: 600; margin-bottom: 8px; color: #1E293B;">Legend</div>
            <div style="margin-bottom: 4px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #DC2626; border-radius: 50%; margin-right: 6px;"></span>
                Critical/Conflict
            </div>
            <div style="margin-bottom: 4px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #0284C7; border-radius: 50%; margin-right: 6px;"></span>
                Person
            </div>
            <div style="margin-bottom: 4px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #1E3A5F; border-radius: 50%; margin-right: 6px;"></span>
                Organization
            </div>
            <div style="margin-bottom: 4px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #B45309; border-radius: 50%; margin-right: 6px;"></span>
                Financial
            </div>
            <div style="margin-bottom: 4px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #6B21A8; border-radius: 50%; margin-right: 6px;"></span>
                Date
            </div>
            <div>
                <span style="display: inline-block; width: 12px; height: 12px; background: #475569; border-radius: 50%; margin-right: 6px;"></span>
                Terms/Clauses
            </div>
        </div>
        """
        
        try:
            content = html_path.read_text()
            # Insert legend after body tag
            content = content.replace("<body>", f"<body>\n{legend_html}")
            html_path.write_text(content)
        except Exception as e:
            logger.warning(f"Could not inject legend: {e}")
    
    def get_node_color(self, entity: Entity) -> str:
        """Get color for an entity node."""
        if entity.entity_id in self.conflict_entities:
            return SEVERITY_COLORS[ConflictSeverity.CRITICAL]
        return ENTITY_COLORS.get(entity.entity_type, "#94A3B8")
    
    def get_edge_color(self, relationship: Relationship) -> str:
        """Get color for a relationship edge."""
        return RELATIONSHIP_COLORS.get(
            relationship.relationship_type,
            NORMAL_EDGE_STYLE["color"]
        )


def visualize_knowledge_graph(
    graph_store: GraphStore,
    output_path: Path,
    document_ids: list[UUID] | None = None,
    conflict_entities: set[UUID] | None = None,
) -> Path:
    """
    Convenience function to generate graph visualization.
    
    Args:
        graph_store: Graph store with entities
        output_path: Where to save HTML
        document_ids: Optional document filter
        conflict_entities: Entities involved in conflicts
        
    Returns:
        Path to saved HTML file
    """
    viz = GraphVisualizer(graph_store, conflict_entities)
    viz.generate_network(document_ids=document_ids)
    return viz.save_html(output_path)
