#!/usr/bin/env python3
"""
CLI Script for Conflict Detection.

Usage:
    python scripts/detect_conflicts.py --docs doc1_id doc2_id
    python scripts/detect_conflicts.py --docs doc1_id doc2_id --focus salary equity
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def run_detection(
    document_ids: list[str],
    focus_areas: list[str],
    graph_path: Path,
    chroma_path: Path,
) -> None:
    """Run the conflict detection pipeline."""
    from src.agents.comparator import ComparatorAgent
    from src.agents.judge import JudgeAgent
    from src.agents.schemas import ComparisonQuery
    from src.knowledge.vector_store import VectorStore, VectorStoreConfig
    from src.knowledge.graph_store import GraphStore
    
    console.print("\n[bold blue]Initializing stores...[/]")
    
    # Initialize stores
    vector_store = VectorStore(VectorStoreConfig(
        persist_directory=chroma_path,
    ))
    graph_store = GraphStore(persist_path=graph_path)
    
    console.print(f"  Vector store: {vector_store.count()} chunks")
    console.print(f"  Graph store: {graph_store.node_count()} nodes, {graph_store.edge_count()} edges")
    
    # Parse UUIDs
    try:
        uuids = [UUID(doc_id) for doc_id in document_ids]
    except ValueError as e:
        console.print(f"[red]Invalid document ID: {e}[/]")
        return
    
    # Create query
    query = ComparisonQuery(
        document_ids=uuids,
        focus_areas=focus_areas,
    )
    
    console.print(f"\n[bold blue]Running Comparator Agent...[/]")
    console.print(f"  Focus areas: {', '.join(focus_areas)}")
    
    # Run Comparator
    comparator = ComparatorAgent(vector_store, graph_store)
    conflicts = await comparator.compare(query)
    
    console.print(f"  [green]Found {len(conflicts)} potential conflicts[/]")
    
    if not conflicts:
        console.print("\n[yellow]No conflicts detected between documents.[/]")
        return
    
    console.print(f"\n[bold blue]Running Judge Agent...[/]")
    
    # Run Judge
    judge = JudgeAgent(vector_store, graph_store)
    report = await judge.verify_and_report(
        conflicts,
        uuids,
        document_ids,
    )
    
    # Display results
    console.print(f"\n[bold green]" + "=" * 60 + "[/]")
    console.print(f"[bold green]CONFLICT DETECTION REPORT[/]")
    console.print(f"[bold green]" + "=" * 60 + "[/]")
    
    console.print(f"\n{report.to_summary()}")
    
    if report.red_flags:
        # Create table for red flags
        table = Table(title="Red Flags", show_header=True, header_style="bold red")
        table.add_column("#", style="dim", width=3)
        table.add_column("Severity", width=10)
        table.add_column("Type", width=20)
        table.add_column("Summary", width=40)
        table.add_column("Pages", width=10)
        
        for i, flag in enumerate(report.red_flags, 1):
            severity_colors = {
                "critical": "red bold",
                "high": "red",
                "medium": "yellow",
                "low": "dim",
            }
            severity_style = severity_colors.get(flag.conflict.severity.value, "white")
            
            table.add_row(
                str(i),
                f"[{severity_style}]{flag.conflict.severity.value.upper()}[/]",
                flag.conflict.conflict_type.value.replace("_", " ").title(),
                flag.summary[:40],
                f"{flag.conflict.evidence_a.citation.page_number} vs {flag.conflict.evidence_b.citation.page_number}",
            )
        
        console.print(table)
        
        # Show details for each red flag
        console.print("\n[bold]Detailed Findings:[/]\n")
        
        for i, flag in enumerate(report.red_flags, 1):
            console.print(Panel(
                f"[bold]{flag.summary}[/]\n\n"
                f"**Type:** {flag.conflict.conflict_type.value}\n"
                f"**Severity:** {flag.conflict.severity.value.upper()}\n\n"
                f"**Document A (Page {flag.conflict.evidence_a.citation.page_number}):**\n"
                f"  Value: {flag.conflict.value_a}\n"
                f"  Context: {flag.conflict.evidence_a.citation.excerpt[:100]}...\n\n"
                f"**Document B (Page {flag.conflict.evidence_b.citation.page_number}):**\n"
                f"  Value: {flag.conflict.value_b}\n"
                f"  Context: {flag.conflict.evidence_b.citation.excerpt[:100]}...\n\n"
                f"**Impact:** {flag.impact}\n"
                f"**Recommended Action:** {flag.recommended_action}",
                title=f"[red]Red Flag #{i}[/]",
                border_style="red" if flag.conflict.severity.value in ["critical", "high"] else "yellow",
            ))
    else:
        console.print("\n[green]✓ All detected conflicts were rejected by the Judge as false positives.[/]")
    
    console.print("\n[dim]Report complete.[/]")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect conflicts across documents"
    )
    parser.add_argument(
        "--docs", "-d",
        type=str,
        nargs="+",
        required=True,
        help="Document IDs to compare (UUIDs)",
    )
    parser.add_argument(
        "--focus", "-f",
        type=str,
        nargs="+",
        default=["salary", "equity", "dates", "parties"],
        help="Areas to focus on (default: salary, equity, dates, parties)",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("./data/graphs/knowledge_graph.json"),
        help="Path to knowledge graph",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("./data/chroma"),
        help="Path to ChromaDB directory",
    )
    
    args = parser.parse_args()
    
    console.print("[bold]Cross-Document Conflict Detector[/]")
    console.print("=" * 50)
    console.print(f"Documents to compare: {len(args.docs)}")
    
    if len(args.docs) < 2:
        console.print("[red]Error: At least 2 documents required for comparison[/]")
        sys.exit(1)
    
    # Run async detection
    asyncio.run(run_detection(
        args.docs,
        args.focus,
        args.graph_path,
        args.chroma_path,
    ))


if __name__ == "__main__":
    main()
