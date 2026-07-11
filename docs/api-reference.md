# API Reference

## REST API

Base URL: `http://localhost:8000` (when running `python src/api.py`)

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
| `book` | string | No | Filter to specific book (omit to search all) |

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

| Field | Type | Description |
|-------|------|-------------|
| `context` | array | Structured list of matching fragments |
| `context[].text` | string | The matching text chunk |
| `context[].book` | string | Book name (collection name) |
| `context[].chapter` | string | Detected chapter heading |
| `context[].start_page` | int | Starting page number |
| `context[].end_page` | int | Ending page number |
| `formatted` | string | Pre-formatted text ready for LLM consumption |

### GET /books

List all indexed book collections.

**Response:**
```json
{
  "books": ["investor-tom1", "investor-tom2", "investor-tom3", "investor-tom4"]
}
```

### DELETE /books/{name}

Remove a book collection from the knowledge base.

**Response (200):**
```json
{
  "status": "deleted",
  "book": "investor-tom1"
}
```

**Error (404):**
```json
{
  "detail": "Book 'investor-tom1' not found"
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
| `book` | string | No | Filter to specific book name |

**Returns:** Formatted string with numbered text blocks and Polish source citations (`Źródło: book, chapter, str. X-Y`).

### search_book_raw(question, book?)

Search and return structured JSON with relevance scores.

**Parameters:** Same as `search_book_tool`.

**Returns:** JSON string with array of objects containing `text`, `book`, `chapter`, `start_page`, `end_page`, and `score`.

### list_books_tool()

List all indexed book collections.

**Returns:** Formatted string listing available book names.
