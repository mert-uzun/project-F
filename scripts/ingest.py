#!/usr/bin/env python3
"""
CLI Script for Document Ingestion.

Usage:
    python scripts/ingest.py --file path/to/document.pdf
    python scripts/ingest.py --dir path/to/documents/
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import chunk_document
from src.utils.logger import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)


async def ingest_file(file_path: Path, output_dir: Path | None = None) -> None:
    """
    Ingest a single PDF file.
    
    Args:
        file_path: Path to PDF file
        output_dir: Optional output directory for processed files
    """
    settings = get_settings()
    
    console.print(f"\n[bold blue]Processing:[/] {file_path.name}")
    
    # Initialize parser
    parser = PDFParser(
        llama_api_key=settings.llama_cloud_api_key or None,
        prefer_llamaparse=bool(settings.llama_cloud_api_key),
    )
    
    try:
        # Parse document
        with console.status("[bold green]Parsing document..."):
            result = await parser.parse(file_path)
        
        # Chunk document
        with console.status("[bold green]Chunking document..."):
            result = chunk_document(result)
        
        # Display results
        _display_results(result)
        
        # Save processed output if output_dir specified
        if output_dir:
            await _save_output(result, output_dir)
        
        console.print(f"\n[bold green]✓[/] Successfully processed {file_path.name}")
        
    except Exception as e:
        console.print(f"\n[bold red]✗[/] Failed to process {file_path.name}: {e}")
        logger.exception(f"Processing failed for {file_path}")


def _display_results(result) -> None:
    """Display parsing results in a nice table."""
    # Metadata table
    meta_table = Table(title="Document Metadata")
    meta_table.add_column("Property", style="cyan")
    meta_table.add_column("Value", style="green")
    
    meta_table.add_row("Filename", result.metadata.filename)
    meta_table.add_row("Pages", str(result.metadata.num_pages))
    meta_table.add_row("Tables", str(result.metadata.num_tables))
    meta_table.add_row("Chunks", str(len(result.chunks)))
    meta_table.add_row("Parser", result.metadata.parser_used)
    meta_table.add_row("Duration", f"{result.metadata.parse_duration_seconds:.2f}s")
    
    console.print(meta_table)
    
    # Tables summary
    if result.tables:
        table_summary = Table(title="Extracted Tables")
        table_summary.add_column("Table #", style="cyan")
        table_summary.add_column("Page", style="green")
        table_summary.add_column("Rows", style="yellow")
        table_summary.add_column("Cols", style="yellow")
        
        for i, table in enumerate(result.tables, 1):
            table_summary.add_row(
                str(i),
                str(table.page_number),
                str(table.num_rows),
                str(table.num_cols),
            )
        
        console.print(table_summary)
    
    # Chunk summary
    if result.chunks:
        console.print(f"\n[bold]Sample Chunk (first 200 chars):[/]")
        console.print(f"[dim]{result.chunks[0].content[:200]}...[/dim]")


async def _save_output(result, output_dir: Path) -> None:
    """Save parsed output to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = result.metadata.filename.replace(".pdf", "")
    
    # Save full text as markdown
    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(result.full_text)
    console.print(f"[dim]Saved markdown to {md_path}[/dim]")
    
    # Save metadata as JSON
    import json
    meta_path = output_dir / f"{base_name}_meta.json"
    meta_path.write_text(result.metadata.model_dump_json(indent=2))
    console.print(f"[dim]Saved metadata to {meta_path}[/dim]")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents for conflict detection"
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to a single PDF file",
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        help="Path to directory of PDF files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.file and not args.dir:
        parser.error("Must specify --file or --dir")
    
    console.print("[bold]Cross-Document Conflict Detector - Ingestion[/]")
    console.print("=" * 50)
    
    if args.file:
        if not args.file.exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            sys.exit(1)
        await ingest_file(args.file, args.output)
    
    elif args.dir:
        if not args.dir.is_dir():
            console.print(f"[red]Directory not found: {args.dir}[/red]")
            sys.exit(1)
        
        pdf_files = list(args.dir.glob("*.pdf"))
        console.print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            await ingest_file(pdf_file, args.output)


if __name__ == "__main__":
    asyncio.run(main())
