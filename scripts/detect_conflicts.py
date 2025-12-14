#!/usr/bin/env python3
"""
CLI Script for Conflict Detection.

Usage:
    python scripts/detect_conflicts.py --docs doc1.pdf doc2.pdf
    python scripts/detect_conflicts.py --query "Check for salary inconsistencies"
    
NOTE: This is a placeholder for Phase 3 implementation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect conflicts across documents"
    )
    parser.add_argument(
        "--docs", "-d",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to documents to compare",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Find all inconsistencies",
        help="Specific query for conflict detection",
    )
    
    args = parser.parse_args()
    
    console.print("[bold]Cross-Document Conflict Detector[/]")
    console.print("=" * 50)
    
    console.print("\n[yellow]⚠ Conflict detection not yet implemented.[/yellow]")
    console.print("This will be implemented in Phase 3.")
    console.print("\nThe pipeline will:")
    console.print("  1. Load documents from vector/graph stores")
    console.print("  2. Run Comparator Agent to find mismatches")
    console.print("  3. Run Judge Agent to verify findings")
    console.print("  4. Generate Red Flag Report with citations")
    
    console.print(f"\n[dim]Documents to compare: {[str(d) for d in args.docs]}[/dim]")
    console.print(f"[dim]Query: {args.query}[/dim]")


if __name__ == "__main__":
    main()
