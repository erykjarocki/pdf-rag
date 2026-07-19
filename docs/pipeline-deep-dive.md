# Pipeline Deep Dive

This document explains every step of the DOC-RAG ingestion and retrieval pipelines in detail, including the architectural decisions behind each choice and how they affect search quality.

For a high-level overview, see [Architecture](architecture.md).

---

## Ingestion Pipeline

The ingestion pipeline transforms a raw document file into searchable vector embeddings stored in Qdrant.

### Step 1: Format Detection & Adapter Selection

**Code:** `src/adapters.py:443` — `get_adapter()`

When a file is ingested, the system selects a format-specific adapter based on the file extension:

```
.pdf  → PDFAdapter
.md   → MarkdownAdapter
.py   → CodeAdapter       (+ 25 other code extensions)
.txt  → PlainTextAdapter   (+ .csv, .log)
```

**Why adapters?** Each format requires fundamentally different extraction logic. PDFs need PyMuPDF with font analysis; Markdown needs heading parsing; code needs function/class detection. The adapter pattern encapsulates this complexity behind a unified `Document` interface, so the rest of the pipeline (chunking, embedding, storage) works unchanged regardless of format.

**Impact on results:** A well-chosen adapter determines what text is extracted and how sections are identified. Poor extraction means missing content or wrong chapter assignments, which directly degrades retrieval quality and citation accuracy.

---

### Step 2: Text Extraction

Each adapter extracts text differently:

#### PDF (`src/extraction.py:50` — `extract_pdf()`)

1. **Table detection** — PyMuPDF's `page.find_tables()` identifies table bounding boxes on each page
2. **Block-level extraction** — `page.get_text("dict")` returns structured blocks with font metadata
3. **Table exclusion** — Text blocks whose center point falls inside a table bbox are skipped (`_block_center_in_table()`)
4. **Table-to-markdown** — Detected tables are converted to markdown format and appended inline after the page text
5. **Paragraph reconstruction** — Lines within each block are joined; blocks are separated by double newlines

**Key decision — block-center table exclusion:** Checking the center point of a text block (rather than any overlap) prevents false positives where a paragraph briefly touches a table edge. This ensures table content isn't duplicated as plain text.

**Impact on results:** Tables become searchable markdown text rather than being lost or garbled. Without this, table data would be extracted as unstructured text fragments that embedding models struggle to encode meaningfully.

#### Markdown (`src/adapters.py:221`)

- Parses `#`–`######` headings as section boundaries
- Content before the first heading becomes a "preamble" section
- Each heading and its content until the next heading becomes a section

#### Code (`src/adapters.py:319`)

- Regex-based detection of `def`, `class`, `fn`, `func`, `function`, `interface`, `struct`, `impl`, `trait`, `enum`, and more
- Supports 20+ languages via a single regex pattern (`_CODE_SECTION_RE`)
- Content before the first function/class becomes "imports_and_preamble"

#### Plain Text (`src/adapters.py:177`)

- Entire file treated as a single section named "full_text"
- Single virtual page (no page concept for text files)

**All adapters return a `Document` dataclass** (`src/adapters.py:41`):

```python
@dataclass
class Document:
    name: str                    # filename without extension
    full_text: str               # concatenated text from all pages
    page_boundaries: list[int]   # cumulative char offsets per page
    page_nums: list[int]         # actual page numbers
    sections: list[DocumentSection]  # logical sections (chapters, headings, functions)
    tables: list[TableInfo]      # detected tables (PDF only)
```

---

### Step 3: Chapter Detection (PDF only)

**Code:** `src/chapter_detection.py:256` — `ChapterDetector`

Chapter detection runs inside `PDFAdapter.extract()` and assigns each page to a chapter. This is critical for source citations (e.g., "Źródło: book, Chapter 3, str. 42").

The detector uses a **three-layer fallback** — only the first successful strategy is used:

#### Layer 1: PDF Outline / TOC (`_build_toc_map()`, line 55)

- Reads the document's built-in bookmark structure via `doc.get_toc(simple=True)`
- Builds a hierarchical breadcrumb: `"Book I > Chapter 3 > Section 1"`
- Fills gaps between TOC entries so every page inherits the preceding entry's chapter

**When it works:** Professionally produced PDFs with bookmarks. Fastest and most reliable.

**Impact on results:** Perfect chapter assignments with correct hierarchy. Citations are precise.

#### Layer 2: Font-Size Analysis (`_build_font_map()`, line 167)

- Extracts per-span font metadata from every page via `page.get_text("dict")`
- Computes mean font size across the entire document (body text dominates the distribution)
- Classifies spans using calibrated thresholds:
  - `font_size >= mean + 4pt` → **heading**
  - `font_size >= mean + 2pt` AND bold AND ALL-CAPS → **heading**
  - `font_size >= mean + 2pt` → **subheading**
  - Otherwise → **content**

**When it works:** PDFs without bookmarks that use visual hierarchy (larger/bolder fonts for headings). Works for any language.

**Impact on results:** Chapter assignments are derived from the document's visual structure. The +4pt/+2pt thresholds are calibrated for formally structured documents — works well for textbooks, legal documents, and academic papers. May miss chapters in documents with unusual formatting.

#### Layer 3: Regex Fallback (`_regex_fallback()`, line 222)

- Scans the first ~500 characters of each page for heading patterns
- Language-agnostic patterns cover:
  - English: `Chapter 1`, `Section 2.1`, `Part I`, `Article 3`
  - Polish: `Rozdział V`, `CZĘŚĆ II`, `Tom 1`, `Dział 3`
  - Generic: `1. Introduction`, `1.1 Overview`
  - Legal: `§ 42`, `Paragraph 3`

**When it works:** Any PDF with recognizable heading patterns in the text. Last resort.

**Impact on results:** Only pages with visible heading patterns get chapter assignments. Pages without a matching pattern get no chapter, resulting in "unknown" in citations. Less precise than TOC or font-based detection.

**Design decision — lazy evaluation:** Each strategy is only computed if all previous ones failed. This avoids expensive font analysis when the TOC is available, and avoids regex when font analysis worked.

---

### Step 4: Text Chunking

**Code:** `src/chunking.py:19` — `chunk_text()`

The full document text is split into chunks that will become individual vector embeddings.

#### How it works

1. **Target size:** `min(CHUNK_SIZE, max_seq_length - 10)` tokens. Default: `min(384, 512 - 10) = 384` tokens. The `-10` buffer prevents truncation during embedding.

2. **Initial estimation:** `target_tokens * 2.5` characters as starting guess (rough average for multilingual text).

3. **Binary search adjustment:** If the guessed chunk has too many tokens, shrink by `(token_count - target_tokens) * 2` characters iteratively until within target.

4. **Page-boundary clamping:** `end_pos` is clamped to the current page's end boundary. Chunks never span across pages.

5. **Overlap:** Each chunk overlaps the previous by `CHUNK_OVERLAP * 4` characters (~200 chars for default 50-token overlap). This preserves context at chunk boundaries.

6. **Page tracking:** Each chunk records `start_page` and `end_page` using `page_at_position()`.

#### Why tokenizer-aware chunking?

Character-count heuristics (e.g., "4 chars per token") are inaccurate for multilingual text. A Polish character like "ź" is one character but may be 1-2 tokens. Using the actual tokenizer produces consistent chunk sizes regardless of language.

**Impact on results:**

- **Chunk size (384 tokens):** Too small → fragments lack context, embedding model can't capture meaning. Too large → multiple topics in one chunk, retrieval returns irrelevant content. 384 tokens is a well-tested balance for `multilingual-e5-base` (512 max seq length).
- **Overlap (50 tokens):** Without overlap, information at chunk boundaries gets split and may not match queries. 50 tokens (~200 chars) provides enough context bridging without excessive duplication.
- **Page clamping:** Ensures page numbers in citations are always accurate. A chunk never claims to be on page 5 when it spans pages 5-6.

---

### Step 5: Chapter Assignment for Chunks

**Code:** `src/ingest.py:56` — inside `process_document()`

After chunking, each chunk is assigned to a chapter based on its character position in the full text:

```python
char_offset = doc.full_text.find(chunk["text"][:50])
chapter = section_for_position(doc.sections, char_offset, doc.full_text)
```

`section_for_position()` (`src/adapters.py:52`) builds cumulative character offsets from section boundaries and checks which section contains the given position.

**Impact on results:** Correct chapter assignment means citations show the right chapter name. If a chunk straddles two chapters, it inherits the chapter of its starting position — a deliberate tradeoff since the chunk's first tokens (most relevant for embedding) belong to that chapter.

---

### Step 6: Embedding

**Code:** `src/embeddings.py:50` — `embed()`

All chunk texts are encoded into 768-dimensional vectors using `intfloat/multilingual-e5-base`.

#### E5 prefix strategy

E5 models are trained with special prefixes:

- **Chunks (indexing):** Each text is prefixed with `"passage: "` → `"passage: The safety protocols require..."`
- **Queries (searching):** The query is prefixed with `"query: "` → `"query: What are the safety protocols?"`

This tells the model to differentiate between text to be searched and text doing the searching.

**Impact on results:** E5 models without prefixes lose ~5-10% retrieval quality. The prefix mechanism creates an asymmetric embedding space where queries and passages are projected differently, improving cosine similarity accuracy.

#### Normalized vectors

```python
model.encode(texts, normalize_embeddings=True)
```

Normalization ensures all vectors lie on the unit hypersphere, making cosine similarity equivalent to dot product. This is both mathematically cleaner and faster for Qdrant (which uses dot product internally with normalized vectors).

**Impact on results:** No quality difference vs raw cosine similarity, but enables Qdrant to use optimized dot product search internally.

---

### Step 7: Qdrant Storage

**Code:** `src/ingest.py:128` — inside `index_document()`

Vectors and metadata are stored in Qdrant in batches of 500 points.

#### Per-document collections

Each document gets its own Qdrant collection (e.g., `safety_manual` for `safety_manual.pdf`). Collection names are sanitized via `collection_name()` (`src/utils.py`) — lowercased, spaces→underscores, special chars stripped, Polish characters transliterated.

#### Point structure

```python
{
    "id": i + 1,
    "vector": [0.023, -0.156, ...],  # 768-dim
    "payload": {
        "text": "chunk text...",
        "book": "safety_manual",
        "chapter": "Chapter 3 > Fire Safety",
        "start_page": 42,
        "end_page": 43,
    }
}
```

#### Batch upserts

Qdrant recommends upserting ≤500 points per batch. Large single upserts can cause timeouts or memory issues with large documents.

**Impact on results:**

- **Per-document collections** enable searching a specific document without noise from others, and easy deletion/re-indexing of individual documents.
- **Cosine distance** (`Distance.COSINE`) ensures score range is always 0-1, making scores interpretable and comparable across searches.

---

## Retrieval Pipeline

The retrieval pipeline takes a natural language query and returns the most relevant text fragments from the indexed knowledge base.

### Step 1: Query Embedding

**Code:** `src/embeddings.py:67` — `embed_query()`

The query is encoded using the same model as indexing, but with the `"query: "` prefix:

```python
text = f"query: {text}"
vector = model.encode([text], normalize_embeddings=True)[0]
```

**Key detail:** The query embedding uses a different prefix than chunk embedding (`query:` vs `passage:`). This asymmetry is intentional — E5 models project queries and passages into different regions of the vector space, improving retrieval accuracy.

**Impact on results:** Using the wrong prefix (or no prefix) for queries degrades retrieval quality by 5-10%. The model was specifically trained to expect these prefixes.

---

### Step 2: Vector Search

**Code:** `src/retriever.py:66` — inside `search_book()`

The query vector is compared against all chunks using cosine similarity in Qdrant.

#### Single collection search

When a specific `book` is requested:

```python
resp = client.query_points(
    collection_name=coll,
    query=query_vector,
    limit=retrieval_limit,  # 20 if reranking, else top_k
)
```

#### All-collection search

When no specific book is specified, the system searches all collections proportionally:

```python
per_collection = max(1, retrieval_limit // len(collections)) + 2
```

Each collection gets `retrieval_limit / num_collections + 2` candidates (the +2 prevents starvation with many collections). Results from all collections are merged and sorted by score, then truncated to `retrieval_limit`.

**Why `+2`?** Without it, searching 5 collections with limit=20 would give each collection only 4 candidates, potentially missing relevant results. The +2 provides a small buffer.

**Impact on results:**

- **Single collection:** Precise, no cross-contamination from other documents.
- **All collections:** Enables cross-document queries but may dilute results if one document dominates relevance. The proportional allocation ensures fair representation.

---

### Step 3: Cross-Encoder Re-ranking

**Code:** `src/reranker.py:48` — `rerank()`

This is the "Advanced RAG" step that separates DOC-RAG from basic vector search.

#### How it works

1. The bi-encoder (Step 2) retrieves 20 candidates based on vector similarity
2. The cross-encoder processes each `(query, candidate)` pair **jointly** through a transformer
3. Each pair gets a new relevance score (not bounded to 0-1, unlike cosine similarity)
4. Candidates are re-sorted by cross-encoder score
5. Top 8 results are returned

#### Bi-encoder vs cross-encoder

| Aspect | Bi-encoder | Cross-encoder |
|--------|-----------|---------------|
| Input | Query and document encoded **separately** | Query and document encoded **jointly** |
| Speed | Fast (pre-computed vectors) | Slow (must process each pair) |
| Accuracy | Good | Excellent |
| Use case | Narrow millions → 20 | Rescore 20 → 8 |

**Why two stages?** The cross-encoder is too slow to search all chunks (it must process each pair individually). The bi-encoder acts as a fast filter, reducing thousands of candidates to a manageable set that the cross-encoder can precisely evaluate.

**Impact on results:**

- Cross-encoder re-ranking typically improves Recall@2 by 5-10% over bi-encoder alone
- It catches semantic matches that cosine similarity misses (e.g., paraphrases, synonyms)
- The tradeoff is added latency (~100-200ms for 20 candidates)

#### The rerank_with_analysis() function

For debugging, `rerank_with_analysis()` (`src/reranker.py:87`) tracks before/after rank positions for each fragment, enabling the trace system to show exactly how the cross-encoder reordered results:

```
Before rerank:  [1] score=0.82  [2] score=0.79  [3] score=0.75
After rerank:   [1] score=0.91  [3] score=0.87  [2] score=0.84
                ↑ promoted      ↑ promoted      ↓ demoted
```

---

### Step 4: Score Thresholding & Low-Quality Detection

**Code:** `src/retriever.py:12`

```python
LOW_SCORE_THRESHOLD = 0.3
```

After retrieval, the system logs warnings when chunks have scores below 0.3. This signals that the query may not match any indexed content well — useful for detecting "no answer" scenarios.

**Impact on results:** Currently informational only (logged, not used for filtering). Future work could use this threshold to return "no relevant information found" instead of forcing low-quality matches.

---

### Step 5: Result Formatting

**Code:** `src/retriever.py:284` — `format_fragments_for_prompt()`

Results are formatted as numbered text blocks with Polish-language source citations:

```
[1] Text content of the chunk...

Źródło: safety_manual, Chapter 3 > Fire Safety, str. 42-43
---

[2] Another relevant fragment...

Źródło: safety_manual, Appendix A, str. 98
---
```

**Why Polish citations?** The system was built for Polish legal/technical documents where "Źródło:" (Source:) is the standard citation format.

**Impact on results:** The LLM agent receiving these fragments gets both the content and precise source attribution, enabling it to cite specific documents and page numbers in its response.

---

## Configuration Impact on Results

| Parameter | Default | Effect on Speed | Effect on Quality | Notes |
|-----------|---------|----------------|-------------------|-------|
| `CHUNK_SIZE` | 384 tokens | Smaller = more chunks = slower | Smaller = more precise retrieval, less context per chunk | Sweet spot for multilingual-e5-base (512 max) |
| `CHUNK_OVERLAP` | 50 tokens | Larger = more chunks = slower | Larger = better context continuity at boundaries | Too much overlap = redundant embeddings |
| `TOP_K` | 8 | Larger = more results to format | Larger = more context for LLM, but may include noise | Per-request override now available via MCP |
| `RERANK_ENABLED` | true | Adds ~100-200ms | +5-10% Recall@2 | Disable for speed-critical applications |
| `RERANK_TOP_N` | 20 | Larger = slower reranking | Larger = cross-encoder sees more candidates | Must be ≥ TOP_K |
| `EMBED_MODEL` | multilingual-e5-base | Base model is slower than small | Base (768d) > Small (384d) | Upgrading to large improves quality further |
| `EMBED_DIM` | 768 | Higher = more storage, slower search | Higher = richer representations | Must match model; changing requires re-indexing |

---

## Retrieval Quality Baseline

Current metrics from the eval suite (80+ labeled queries against 20 benchmark documents):

| Metric | Bi-encoder only | With reranking |
|--------|----------------|----------------|
| Recall@2 | 0.82 | 0.89 |
| Precision@2 | 0.78 | 0.82 |
| MRR | 0.81 | 0.89 |

These baselines are used in CI to detect regressions — any code change that drops below thresholds triggers a CI failure.

---

## Future Improvements

See [Improvement Roadmap](improvements.md) for planned enhancements. The highest-impact next steps:

1. **Hybrid search (BM25 + vector)** — Combines keyword matching with semantic search to catch exact-term matches that vector search misses
2. **Query expansion/rewriting** — Generates multiple query variants to improve recall for varied phrasing
3. **Async MCP tools** — Enables concurrent tool calls without blocking the MCP event loop
