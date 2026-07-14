# API Reference

## REST API

Base URL: `http://localhost:8000` (when running `python src/api.py`)

### POST /ingest

Ingest a single document into the knowledge base.

**Request (JSON — file path):**
```json
{
  "file_path": "/path/to/document.pdf",
  "reindex": false
}
```

**Request (multipart — file upload):**
```
Content-Type: multipart/form-data
file: <binary>
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes* | Absolute path to document on server |
| `file` | file | Yes* | Uploaded file (alternative to path) |
| `reindex` | bool | No | Delete existing collection first (default: false) |

*One of `file_path` or `file` is required.

**Response:**
```json
{
  "status": "indexed",
  "book": "my-document",
  "chunks": 142,
  "format": "pdf"
}
```

### POST /ingest-folder

Ingest all supported documents from a directory.

**Request:**
```json
{
  "directory": "/path/to/documents",
  "reindex": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `directory` | string | Yes | Absolute path to directory on server |
| `reindex` | bool | No | Re-index all documents (default: false) |

**Response:**
```json
{
  "results": [
    {"name": "doc1", "status": "indexed", "chunks": 42, "error": null},
    {"name": "doc2", "status": "skipped", "chunks": null, "error": null},
    {"name": "doc3", "status": "indexed", "chunks": 18, "error": null}
  ],
  "total_indexed": 2,
  "total_skipped": 1,
  "total_errors": 0
}
```

### POST /query

Search the knowledge base for relevant text fragments.

**Request:**
```json
{
  "question": "What are the main investment strategies?",
  "book": "investor-tom1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | Natural language query |
| `book` | string | No | Filter to specific document (omit to search all) |

**Response:**
```json
{
  "context": [
    {
      "text": "Dywersyfikacja portfela jest kluczowa...",
      "book": "investor-tom1",
      "chapter": "Rozdział 3",
      "start_page": 45,
      "end_page": 47
    }
  ],
  "formatted": "[1] Dywersyfikacja portfela jest kluczowa...\n\nŹródło: investor-tom1, Rozdział 3, str. 45-47\n---"
}
```

### GET /collections

List all indexed collections with chunk counts.

**Response:**
```json
{
  "collections": [
    {"name": "investor-tom1", "chunks": 312},
    {"name": "investor-tom2", "chunks": 287}
  ]
}
```

### GET /formats

List all supported document formats.

**Response:**
```json
{
  "formats": [".pdf", ".txt", ".md", ".py", ".js", ".ts", ".rs", ".go", ".java", "..."]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## MCP Tools

Used by OpenCode when configured as an MCP server.

### search_book_tool(question, book?)

Search the knowledge base and return formatted text fragments.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | Natural language query |
| `book` | string | No | Filter to specific document name |

**Returns:** Formatted string with numbered text blocks and Polish source citations (`Źródło: book, chapter, str. X-Y`).

### search_book_raw(question, book?)

Search and return structured JSON with relevance scores.

**Parameters:** Same as `search_book_tool`.

**Returns:** JSON string with array of objects containing `text`, `book`, `chapter`, `start_page`, `end_page`, and `score`.

### list_books_tool()

List all indexed document collections.

**Returns:** Formatted string listing available document names.

### ingest_document(file_path, reindex?)

Ingest any supported document format into the knowledge base.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Absolute path to document |
| `reindex` | bool | No | Delete existing collection first (default: false) |

**Returns:** Status message with document name and chunk count.

### ingest_folder(directory, reindex?)

Ingest all supported documents from a directory.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `directory` | string | Yes | Absolute path to directory |
| `reindex` | bool | No | Re-index all documents (default: false) |

**Returns:** Summary with per-file results (indexed/skipped/error) and totals.
