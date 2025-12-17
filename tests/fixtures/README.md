# Tests Fixtures

This directory contains test fixture files.

## Files

- `sample.pdf` - Sample employment agreement for testing PDF parsing

## Generating Fixtures

To generate the sample PDF:

```bash
pip install reportlab
python tests/generate_fixtures.py
```

The generated PDF contains a mock employment agreement with:
- Salary: $500,000
- Equity: 5%
- Start Date: January 1, 2024
- Various benefits and termination clauses
