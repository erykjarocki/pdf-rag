# Testing

PDF-RAG uses a three-layer testing strategy to ensure correctness at every level — from pure utility functions to the full retrieval pipeline.

## Test Layers

### Unit Tests

Fast, no I/O. Test pure logic only — chunking, chapter detection, formatting, config.

```bash
make test-unit
# or directly:
pytest tests/unit/ -v -m unit
```

**What's covered:**
- `collection_name()` sanitization (special chars, casing, empty input)
- `detect_chapter()` regex for Polish and English headings
- `get_page_boundaries()` and `_page_at_position()` page tracking
- `format_fragments_for_prompt()` citation formatting

### Integration Tests

Use in-memory Qdrant (`QdrantClient(":memory:")`) and mocked embeddings. No Docker, no network.

```bash
make test-integration
# or directly:
pytest tests/integration/ -v -m integration
```

**What's covered:**
- Qdrant collection CRUD (create, delete, list)
- Vector upsert and cosine similarity query
- Score ranking (exact match ranks higher than partial)
- FastAPI endpoint contract (`/health`, `/query`, `/books`, `/delete`)

### Eval Tests (planned)

Full pipeline with real embeddings and labeled questions. Marked `@pytest.mark.eval`, not included in default CI — run nightly or manually.

```bash
pytest tests/ -v -m eval
```

## Running All Tests

```bash
make test            # all layers
make test-unit       # unit only
make test-integration # integration only
```

## Test Structure

```
tests/
├── conftest.py                # shared fixtures (in-memory Qdrant, sample data)
├── unit/
│   ├── test_config.py         # collection_name() sanitization
│   ├── test_ingest.py         # chapter detection, page boundaries
│   └── test_retriever.py      # formatting, result parsing
└── integration/
    ├── test_retrieval.py      # Qdrant round-trip (in-memory)
    └── test_api.py            # FastAPI TestClient endpoint tests
```

## How Mocking Works

The key challenge in testing a RAG system is avoiding expensive/slow operations (embedding model, vector DB). PDF-RAG handles this two ways:

1. **In-memory Qdrant** — `QdrantClient(":memory:")` runs the full vector DB in-process. No Docker needed. Tests that need Qdrant get a fresh client via the `qdrant_memory` fixture.

2. **Mocked embeddings** — `@patch("src.embeddings.get_model")` replaces the SentenceTransformer model with a `MagicMock`. This lets us test chunking and formatting logic without loading the 100MB+ model.

**Important:** Always patch where the function is *used*, not where it's defined. For example, `ingest.py` imports `get_model` from `src.embeddings`, so patch `"src.ingest.get_model"` — not `"src.embeddings.get_model"`.

## CI Integration

GitHub Actions runs a 2-stage pipeline on every PR (see `.github/workflows/ci.yml`):

- **Triggers:** Push/PR to `main` when `src/`, `tests/`, or `pyproject.toml` change
- **Python:** 3.12 (single version — project is a local tool, not a library)
- **Stage 1:** Install dependencies (warms pip cache)
- **Stage 2 (parallel):** Lint (ruff), Unit tests, Integration tests
- **Optimization:** CPU-only PyTorch installed first (~150MB vs ~3GB with CUDA)
- **No Docker needed** — all tests use in-memory Qdrant

### Testing CI Locally

```bash
# Run the same commands CI executes:
ruff check src/ tests/
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
```

## Adding New Tests

1. **Pure function?** → Add to `tests/unit/test_<module>.py`, mark `@pytest.mark.unit`
2. **Needs Qdrant?** → Use the `qdrant_memory` fixture, mark `@pytest.mark.integration`
3. **Needs embeddings?** → Mock `get_model()` with a `MagicMock` that returns deterministic vectors
4. **Needs FastAPI?** → Use `TestClient(app)` from `fastapi.testclient`

### Example: Testing a New Function

```python
# tests/unit/test_example.py
import pytest

@pytest.mark.unit
def test_my_new_function():
    from src.my_module import my_new_function
    result = my_new_function("input")
    assert result == "expected"
```

```python
# tests/integration/test_example.py
import pytest
from unittest.mock import patch

@pytest.mark.integration
def test_with_qdrant(qdrant_memory):
    from src.qdrant_store import ensure_collection
    ensure_collection("test", qdrant_memory)
    assert "test" in qdrant_memory.get_collections().collections
```
