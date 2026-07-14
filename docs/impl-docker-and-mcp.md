# Implementation Plan: Docker + MCP Integration

**Goal**
1. One-command setup via Docker Compose
2. Works with any AI tool (Claude Desktop, Cursor, Windsurf, Cline, OpenCode)
3. Document everything so users know exactly how to connect

## Phase 1: Docker Compose

**Files to create/modify:**

| File | Action | Purpose |
|------|--------|---------|
| `Dockerfile` | CREATE | Python 3.12 slim image, install deps, copy source |
| `docker-compose.yml` | CREATE | Qdrant + pdf-rag services, one-command start |
| `Makefile` | MODIFY | Add docker-up, docker-down targets |
| `README.md` | MODIFY | Add "Quick Start with Docker" section |

**Dockerfile design:**
```dockerfile
FROM python:3.12-slim
# Install only CPU PyTorch (smaller image)
# Copy pyproject.toml, install deps
# Copy src/
# Default: MCP server on stdio
```

**docker-compose.yml design:**
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["./vector_db/qdrant:/qdrant/storage"]
  
  pdf-rag:
    build: .
    depends_on: [qdrant]
    volumes:
      - ./data:/app/data
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    # Default: MCP server via stdio
```

**User experience after this phase:**
```bash
git clone https://github.com/erykjarocki/pdf-rag
cd pdf-rag
docker compose up -d                    # start everything
docker compose run pdf-rag python src/ingest.py /path/to/doc.pdf  # index
```

## Phase 2: MCP Config for Every AI Tool

**File to create:** `docs/mcp-setup.md`

Contains copy-paste config for each tool:

| Tool | Config location |
|------|----------------|
| OpenCode | `~/.config/opencode/opencode.json` |
| Claude Desktop | `~/.claude/claude_desktop_config.json` |
| Cursor | `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) |
| Windsurf | `~/.windsurf/mcp.json` |
| Cline | VS Code settings or `.clinerules` |

Key insight: All these tools use the same MCP protocol. The config format is nearly identical — just the file path differs.

## Phase 3: README Overhaul

Restructure for three audiences:

1. **Quick Start (Docker)** — copy 3 commands, done
2. **AI Tool Integration** — link to `docs/mcp-setup.md`, show one example
3. **Developer Setup** — current content (Makefile, venv, etc.)

## Phase 4: Enhance MCP Server (Optional)

Add tools that make the system more useful:

| New Tool | Purpose |
|----------|---------|
| `summarize_book(book)` | Generate a summary of an entire book |
| `get_chapters(book)` | List all chapters in a book |
| `get_page(book, page)` | Get text from a specific page |
| `get_stats()` | Show total books, chunks, pages indexed |

These tools let AI agents do more than just search — they can explore the knowledge base.

## Files Changed Summary

| Phase | Files |
|-------|-------|
| 1 | `Dockerfile` (new), `docker-compose.yml` (new), `Makefile` (modify), `README.md` (modify) |
| 2 | `docs/mcp-setup.md` (new), `src/mcp_server.py` (modify for SSE transport) |
| 3 | `README.md` (major rewrite) |
| 4 | `src/mcp_server.py` (add 4 tools) |

## What NOT to Do

| Skip | Why |
|------|-----|
| Web UI | Your users are developers with AI tools — they don't need a browser interface |
| PyPI package | Docker is sufficient for distribution; PyPI adds maintenance burden |
| Hybrid search / reranking | Important but separate concern — can be Improvement #13/#14 later |
| Async ingestion | Nice-to-have, not blocking adoption |

## Success Criteria

After this plan is executed:
1. `docker compose up -d` starts everything
2. A user can copy-paste MCP config into Claude Desktop and start querying documents in < 5 minutes
3. README clearly explains the value proposition in 3 sentences
4. Each AI tool's setup is documented with exact file paths and JSON
