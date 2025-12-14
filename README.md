# Cross-Document Conflict Detector

A GraphRAG-based Due Diligence tool for Private Equity/M&A sector that identifies logical contradictions across financial/legal documents.

## 🎯 The Problem

In M&A due diligence, analysts spend days manually comparing hundreds of documents to find inconsistencies:
- "Document A says CEO gets **5% equity**"
- "Document B says CEO gets **3% equity**"

Standard LLMs miss these because they process documents separately. **We solve this.**

## 🏗️ Architecture

```
Layer 1: Ingestion Engine (The Moat)
├── LlamaParse → Table-aware PDF parsing
├── Semantic Chunking → Preserve clause boundaries
└── Metadata Extraction → Page numbers, sections

Layer 2: Knowledge Layer (GraphRAG)
├── Vector Store (ChromaDB) → Semantic search
├── Graph Store (NetworkX) → Entity relationships
└── Entity Extraction → Structured data from text

Layer 3: Logic Agents
├── Comparator → Detect value mismatches
└── Judge → Verify and prevent hallucinations

Layer 4: Interface
├── FastAPI Backend
└── Streamlit UI (coming soon)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM) or OpenAI API key
- [LlamaParse API key](https://cloud.llamaindex.ai/) (for table extraction)

### Installation

```bash
# Clone and setup
cd project-F
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the API

```bash
uvicorn app.main:app --reload
```

### Local LLM Setup (Privacy Mode)

```bash
# Install Ollama and pull Llama 3
ollama pull llama3

# Set in .env
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3
```

## 📁 Project Structure

```
project-F/
├── app/                 # FastAPI application
├── src/
│   ├── ingestion/      # PDF parsing & chunking
│   ├── knowledge/      # Vector & Graph stores
│   ├── agents/         # Conflict detection logic
│   └── utils/          # LLM factory, logging
├── tests/              # Test suite
├── data/               # Uploads, processed files, graphs
└── scripts/            # CLI utilities
```

## 🔒 Privacy First

The system is designed for on-premise deployment:
- Swap LLM backend in one line of config
- Local embeddings with HuggingFace models
- All data stays on your infrastructure

## 📄 License

MIT
