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
├── Entity Resolution → Deduplicate aliases
├── Cross-Reference Engine → Find mentions across docs
└── Timeline Builder → Chronological event tracking

Layer 3: Logic Agents
├── Comparator → Detect value mismatches
├── Judge → Verify and prevent hallucinations
├── Multi-Doc Analyzer → N-way conflict detection
├── Reference Detector → Find missing documents
└── Report Generator → Executive summaries

Layer 4: Interface
├── FastAPI Backend → 11 REST endpoints
└── Streamlit UI → Investor-facing demo
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

### Run the Application

```bash
# Terminal 1: Start the API
uvicorn app.main:app --reload

# Terminal 2: Start the UI
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

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
├── app/                 # FastAPI application (11 endpoints)
├── src/
│   ├── ingestion/       # PDF parsing & chunking
│   ├── knowledge/       # Vector/Graph stores, entity resolution
│   ├── agents/          # Conflict detection, reports
│   └── utils/           # LLM factory, logging
├── ui/                  # Streamlit UI (9 components)
│   ├── components/      # Upload, Analysis, Conflicts, Graph, Timeline, Report
│   ├── utils/           # API client, formatters
│   └── static/          # CSS styling
├── tests/               # Test suite (184 tests)
├── data/                # Uploads, processed files, graphs
└── scripts/             # CLI utilities
```

## 🎨 UI Features

- **Document Upload**: Drag-and-drop PDF upload with progress tracking
- **Data Inspector**: Side-by-side PDF vs parsed output view
- **Analysis Dashboard**: Live reasoning trace with audit log
- **Conflict Workbench**: Master-detail view with PDF citations
- **Knowledge Graph**: Interactive PyVis visualization
- **Timeline View**: Chronological events with conflict highlighting
- **Executive Summary**: Downloadable markdown reports

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest` | POST | Upload and process PDF |
| `/detect-conflicts` | POST | Pairwise conflict detection |
| `/analyze` | POST | Multi-document analysis |
| `/timeline` | POST | Build event timeline |
| `/search` | GET | Entity search |
| `/report` | POST | Generate executive summary |
| `/graph` | GET | Graph data JSON |
| `/graph/html` | GET | Interactive graph HTML |
| `/missing-documents` | POST | Find referenced but not uploaded docs |

## 🔒 Privacy First

The system is designed for on-premise deployment:
- Swap LLM backend in one line of config
- Local embeddings with HuggingFace models
- All data stays on your infrastructure

## 📄 License

MIT
