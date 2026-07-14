# DOC-RAG

Local RAG system for documents — chat with PDFs, text files, markdown, source code, or any document via AI.

Ask questions, get answers based **only on your document content**, with citations.

## How it works

```
Document → text extraction → chunks → embeddings (local) → Qdrant (vector DB)
                                                               ↓
OpenCode (MCP) ← search_book_tool ← retriever ← similarity search
                                                    ↓
                                              cross-encoder re-ranking
                                                    ↓
                                              LLM answers based on found fragments
```

- **100% local** — no data leaves your machine
- **Two-stage retrieval** — fast bi-encoder + precise cross-encoder re-ranking
- **OpenCode integration** — works as an MCP tool
- **Any document** — PDFs, markdown, source code, text files
- **Always cites sources** — document/chapter/page

## Quick start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Git

### Setup

```bash
# Clone
git clone git@github.com:erykjarocki/doc-rag.git
cd doc-rag

# One command — installs everything, starts Qdrant, prints OpenCode config
make setup
```

Then add the printed config snippet to `~/.config/opencode/opencode.json` and restart OpenCode.

**That's it.** Ask OpenCode to ingest and search your documents.

### Ingesting documents

```bash
# Via OpenCode — just ask:
# "Ingest /path/to/document.pdf"
# "Ingest all files in /path/to/documents/"

# Via API
curl -X POST localhost:8000/ingest -F "file=@document.pdf"
curl -X POST localhost:8000/ingest-folder -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/documents"}'

# Via CLI
python src/ingest.py /path/to/document.pdf
python src/ingest.py --folder /path/to/documents/
```

### Day-to-day

```bash
# Start Qdrant (if stopped)
make start

# Run API server (optional)
make serve

# Run MCP server (standalone test)
make mcp
```

## Supported formats

| Format | Extensions | Sections detected |
|--------|-----------|-------------------|
| PDF | `.pdf` | Chapter detection (TOC → font analysis → regex) |
| Markdown | `.md`, `.markdown` | `#` headings |
| Source code | `.py`, `.js`, `.ts`, `.rs`, `.go`, `.java`, etc. | Functions, classes |
| Plain text | `.txt`, `.log`, `.csv` | None (single section) |

40+ extensions supported. See `src/adapters.py` for the full list.

## OpenCode integration

`make setup` prints the config automatically. If you need to add it manually:

```json
"doc-rag": {
  "type": "local",
  "command": ["/path/to/doc-rag/venv/bin/python", "-m", "src.mcp_server"],
  "cwd": "/path/to/doc-rag",
  "enabled": true
}
```

### MCP tools

- `search_book_tool(question, book=None)` — search all documents or filter by name
- `search_book_raw(question, book=None)` — returns JSON with scores
- `list_books_tool()` — list available documents
- `ingest_document(file_path, reindex=False)` — ingest a single file
- `ingest_folder(directory, reindex=False)` — ingest all supported files in a directory

## API

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Search the knowledge base |
| `/ingest` | POST | Ingest a file (upload or path) |
| `/ingest-folder` | POST | Ingest all files in a directory |
| `/collections` | GET | List indexed collections |
| `/formats` | GET | List supported file formats |
| `/health` | GET | Health check |

## Project structure

```
doc-rag/
├── data/
│   ├── extracted/      # Raw text from documents
│   ├── chunks/         # Processed chunks
│   └── metadata/       # Index metadata
├── vector_db/qdrant/   # Qdrant storage
├── src/
│   ├── config.py       # Configuration
│   ├── adapters.py     # Format adapters (PDF, MD, code, text)
│   ├── ingest.py       # Document → Qdrant pipeline
│   ├── embeddings.py   # Local embedding model
│   ├── retriever.py    # search_book() function
│   ├── mcp_server.py   # MCP server for OpenCode
│   ├── api.py          # REST API
│   └── qdrant_store.py # Qdrant client helpers
├── venv/
├── pyproject.toml
├── Makefile
└── README.md
```

## Requirements

- Python 3.10+
- Docker (for Qdrant vector database)
- RAM: min 4 GB
- Disk: ~1.5 GB

## Re-ranking (Optional)

DOC-RAG supports cross-encoder re-ranking for higher precision. When enabled, it retrieves more candidates initially, then rescores them using a more accurate model.

```json
// ~/.config/doc-rag/config.json
{
  "rerank": {
    "enabled": true,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_n": 20
  }
}
```

**When to enable:**
- Answer quality matters more than latency
- Complex queries requiring precise matching
- ~100ms additional latency is acceptable

**When to keep disabled:**
- You need maximum speed
- Simple keyword searches
- Resource-constrained hardware

See [Architecture](docs/architecture.md) for detailed explanation of two-stage retrieval.

## Changing the embedding model

The default model is `intfloat/multilingual-e5-small` (384 dim) — fast, local, handles Polish well.

```bash
# Edit src/config.py:
#    EMBED_MODEL = "intfloat/multilingual-e5-large"
#    EMBED_DIM = 1024

# Re-index existing collections
python src/ingest.py /path/to/file.pdf --reindex
```

| Model | Dimensions | Quality | Speed |
|---|---|---|---|
| `multilingual-e5-small` | 384 | good | fast |
| `multilingual-e5-base` | 768 | better | medium |
| `multilingual-e5-large` | 1024 | best | slow |
