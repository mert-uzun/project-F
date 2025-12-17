#!/usr/bin/env python3
"""
Generate sample PDF fixtures for testing.

Creates a simple employment agreement PDF for testing the ingestion pipeline.
Requires: pip install reportlab
"""

from pathlib import Path


def generate_sample_pdf(output_path: Path) -> None:
    """Generate a sample employment agreement PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return
    
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    
    # Page 1: Employment Agreement
    y = height - inch
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(inch, y, "EMPLOYMENT AGREEMENT")
    y -= 0.5 * inch
    
    c.setFont("Helvetica", 12)
    c.drawString(inch, y, "Effective Date: January 1, 2024")
    y -= 0.3 * inch
    
    c.drawString(inch, y, "This Employment Agreement (the \"Agreement\") is entered into between:")
    y -= 0.4 * inch
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Employer: Acme Corporation")
    y -= 0.25 * inch
    c.drawString(inch, y, "Employee: John Smith")
    y -= 0.5 * inch
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "1. COMPENSATION")
    y -= 0.3 * inch
    
    c.setFont("Helvetica", 12)
    lines = [
        "The Employee shall receive the following compensation:",
        "",
        "Base Salary: $500,000 per annum",
        "Signing Bonus: $100,000 (payable within 30 days)",
        "Annual Bonus Target: 50% of base salary",
        "",
        "Salary shall be paid in bi-weekly installments.",
    ]
    
    for line in lines:
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    
    y -= 0.3 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "2. EQUITY COMPENSATION")
    y -= 0.3 * inch
    
    c.setFont("Helvetica", 12)
    equity_lines = [
        "The Employee shall be granted the following equity:",
        "",
        "Stock Options: 100,000 shares",
        "Equity Percentage: 5% of fully diluted shares",
        "Vesting Schedule: 4 years with 1-year cliff",
        "Exercise Price: $10.00 per share",
    ]
    
    for line in equity_lines:
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    
    y -= 0.3 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "3. EMPLOYMENT TERM")
    y -= 0.3 * inch
    
    c.setFont("Helvetica", 12)
    term_lines = [
        "Start Date: January 1, 2024",
        "Initial Term: 3 years",
        "Renewal: Automatic annual renewal thereafter",
        "Notice Period: 90 days",
    ]
    
    for line in term_lines:
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    
    # Page 2
    c.showPage()
    y = height - inch
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "4. BENEFITS")
    y -= 0.3 * inch
    
    c.setFont("Helvetica", 12)
    benefits = [
        "Health Insurance: Full family coverage",
        "401(k): 6% employer match",
        "Vacation: 25 days per year",
        "Professional Development: $10,000 annual allowance",
    ]
    
    for line in benefits:
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    
    y -= 0.5 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "5. TERMINATION")
    y -= 0.3 * inch
    
    c.setFont("Helvetica", 12)
    termination = [
        "Severance: 12 months base salary if terminated without cause",
        "Acceleration: 50% equity acceleration on change of control",
        "Non-compete: 12 months post-termination",
    ]
    
    for line in termination:
        c.drawString(inch, y, line)
        y -= 0.25 * inch
    
    # Signature block
    y -= inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "SIGNATURES")
    y -= 0.5 * inch
    
    c.setFont("Helvetica", 12)
    c.drawString(inch, y, "_________________________")
    c.drawString(4 * inch, y, "_________________________")
    y -= 0.25 * inch
    
    c.drawString(inch, y, "John Smith (Employee)")
    c.drawString(4 * inch, y, "Jane Doe (Acme Corporation)")
    y -= 0.25 * inch
    
    c.drawString(inch, y, "Date: January 1, 2024")
    c.drawString(4 * inch, y, "Date: January 1, 2024")
    
    c.save()
    print(f"Created sample PDF: {output_path}")


if __name__ == "__main__":
    fixture_dir = Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(exist_ok=True)
    
    generate_sample_pdf(fixture_dir / "sample.pdf")
