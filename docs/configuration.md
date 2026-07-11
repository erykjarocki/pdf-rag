# Configuration

All configuration lives in `src/config.py`.

## Configuration Options

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | Project root | Auto-detected from file location |
| `BOOKS_DIR` | `books/` | Directory containing PDF files |
| `EXTRACTED_DIR` | `data/extracted/` | Where raw extracted text is saved (.txt) |
| `CHUNKS_FILE` | `data/chunks/chunks.json` | Reserved for cached chunks (not yet used) |
| `METADATA_FILE` | `data/metadata/metadata.json` | Reserved for index metadata (not yet used) |
| `QDRANT_PATH` | `vector_db/qdrant` | Qdrant storage path (unused, Docker mode active) |

### Embedding Model

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Sentence-transformers model name |
| `EMBED_DIM` | `384` | Vector dimensions (must match model) |

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant Docker host |
| `QDRANT_PORT` | `6333` | Qdrant Docker port |

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `384` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between adjacent chunks |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `8` | Default number of results returned per query |

---

## Changing the Embedding Model

Edit `src/config.py`:

```python
# Options: multilingual-e5-small (384d), multilingual-e5-base (768d),
#          multilingual-e5-large (1024d), BAAI/bge-m3, etc.
EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_DIM = 1024  # Must match the model's output dimension
```

Then **re-index all books** (old collections have wrong dimensions):

```bash
python src/ingest.py --reindex investor-tom1
python src/ingest.py --reindex investor-tom2
python src/ingest.py --reindex investor-tom3
python src/ingest.py --reindex investor-tom4
```

Or delete and re-index all:

```bash
python src/ingest.py --delete investor-tom1
python src/ingest.py  # re-indexes all
```

---

## Qdrant Setup

### First time

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/vector_db/qdrant:/qdrant/storage \
  qdrant/qdrant
```

### Subsequent starts

```bash
docker start qdrant
```

### Check status

```bash
curl http://localhost:6333/health
```

---

## Changing Chunk Size

Larger chunks = more context per result, but fewer total chunks. Smaller chunks = more precise retrieval, but may split related content.

```python
# In src/config.py
CHUNK_SIZE = 200   # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 30 # Proportionally reduce overlap
```

After changing, re-index affected books.
