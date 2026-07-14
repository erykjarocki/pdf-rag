# DOC-RAG Roadmap

## Current Architecture Level: Advanced RAG (Partial)

DOC-RAG currently implements **Naive RAG** with partial **Advanced RAG** features:

| Pattern | Status | Notes |
|---------|--------|-------|
| Naive RAG | ✅ Complete | Basic retrieve-then-generate pipeline |
| Advanced RAG | ⚠️ Partial | ✅ Cross-encoder re-ranking, ❌ Query rewriting, ❌ Hybrid search |
| Modular RAG | ✅ Complete | Adapter pattern, composable pipeline |
| Agentic RAG | ❌ Not started | — |
| Self-RAG | ❌ Not started | — |
| Graph RAG | ❌ Not started | — |
| Reasoning-Based RAG | ❌ Not started | — |

---

## Implemented: Cross-Encoder Re-ranking

**Status:** ✅ Implemented (v0.1.0)

**What it does:**
- After initial fast retrieval (bi-encoder), rescores results using a cross-encoder
- Cross-encoder sees query and document jointly, producing much higher accuracy
- Trades ~100ms latency for significantly better precision

**Configuration:**
```json
{
  "rerank": {
    "enabled": false,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_n": 20
  }
}
```

**How it works:**
1. Bi-encoder (`multilingual-e5-small`) retrieves top-20 candidates (fast)
2. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores each pair (accurate)
3. Returns top-8 results with both `score` (bi-encoder) and `rerank_score` (cross-encoder)

**When to enable:**
- Answer quality matters more than latency
- Complex queries requiring precise matching
- ~100ms additional latency is acceptable

---

## Next: Advanced RAG Features

### Priority 1: Query Rewriting & Expansion

**Impact:** Medium | **Effort:** Low | **Status:** Not started

**What it does:**
- Expands vague queries with synonyms and related terms
- Generates sub-queries for complex questions
- Merges results from multiple query variations

**Implementation plan:**
```python
# src/query_rewrite.py
def rewrite_query(query: str) -> list[str]:
    """Expand query into multiple search variations."""
    # Simple: add synonyms
    # Advanced: use LLM to generate sub-queries
    return [query, f"{query} definition", f"{query} examples"]
```

**Verify it works:**
- Compare retrieval results with/without rewriting
- Test on vague queries ("Tell me about history")
- Expect: 10-15% improvement in recall for ambiguous queries

---

### Priority 2: Hybrid Search (Dense + BM25)

**Impact:** High | **Effort:** High | **Status:** Not started

**What it does:**
- Combines semantic search (dense vectors) with keyword search (BM25)
- Dense search misses exact matches; BM25 catches them
- Qdrant supports sparse vectors natively

**Implementation plan:**
1. During ingestion: also store BM25 sparse vectors
2. During retrieval: run both dense and sparse searches
3. Combine scores with configurable weighting (e.g., 0.7 semantic + 0.3 BM25)

**Verify it works:**
- Test with exact-match queries ("Section 3.2.1")
- Test with technical terms ("PyMuPDF", "Qdrant")
- Expect: Find results that pure dense search misses

---

### Priority 3: Relevance Thresholding (Self-RAG)

**Impact:** Medium | **Effort:** Low | **Status:** Not started

**What it does:**
- Filters out low-score results instead of returning garbage
- Configurable threshold per collection
- Returns "No relevant fragments found" when nothing matches

**Implementation plan:**
```python
# In config.py
SCORE_THRESHOLD = 0.25  # Configurable per collection

# In retriever.py
results = [r for r in results if r["score"] >= SCORE_THRESHOLD]
```

**Verify it works:**
- Measure false positive rate (irrelevant results returned)
- Should decrease without hurting recall
- Test threshold values: 0.2, 0.25, 0.3

---

## Future: Advanced Patterns

### Graph RAG (Long-term)

**Impact:** Very High | **Effort:** Very High | **Status:** Not started

**What it does:**
- Extracts entities and relationships from documents
- Builds a knowledge graph alongside vector storage
- Enables multi-hop reasoning across documents

**When to consider:**
- After Advanced RAG is solid
- You have multi-document use cases
- Users ask complex questions requiring cross-document reasoning

**Implementation plan:**
1. Add entity extraction during ingestion (spaCy or LLM-based)
2. Store entities and relationships in Qdrant or Neo4j
3. Implement graph traversal for multi-hop queries
4. Combine graph results with vector search results

---

### Agentic RAG (Long-term)

**Impact:** High | **Effort:** High | **Status:** Not started

**What it does:**
- LLM decides when and how to retrieve information
- Dynamic strategy selection based on query type
- Self-reflection on retrieval quality

**When to consider:**
- After Graph RAG is implemented
- You need adaptive retrieval strategies
- Users have diverse query types

**Implementation plan:**
1. Define retrieval strategies (keyword, semantic, graph, multi-hop)
2. Add query classifier to determine strategy
3. Implement LLM-driven retrieval planning
4. Add self-reflection to verify answer quality

---

## Verification Strategy

### Current Eval System

- 13 labeled queries with relevant pages
- Metrics: Recall@2, Precision@2, MRR
- Baseline comparison in CI
- Regression detection on PRs

### Adding New Metrics

| Metric | What It Measures | How to Add |
|--------|------------------|------------|
| **Recall@5** | Coverage with more results | Change k=5 in eval |
| **NDCG@5** | Ranking quality | Weight by position |
| **Latency** | Speed of retrieval | Time each search_book call |
| **False Positive Rate** | Irrelevant results returned | Count below threshold |
| **Query Rewrite Accuracy** | Better results after expansion | Compare before/after |

### Test Scenarios to Add

| Scenario | Why | Priority |
|----------|-----|----------|
| Multi-hop queries | "What do France and Germany have in common?" | High |
| Vague queries | "Tell me about history" | High |
| Exact match | "Section 3.2" | Medium |
| Cross-document | Queries requiring info from multiple docs | Medium |
| Adversarial | Queries designed to fail | Low |

---

## Implementation Priority

1. ✅ **Cross-Encoder Re-ranking** — done
2. **Query Rewriting** — easy win, improves recall
3. **Relevance Thresholding** — reduces false positives
4. **Hybrid Search** — biggest quality improvement
5. **Web UI** — makes system accessible to non-technical users
6. **Graph RAG** — enables multi-hop reasoning
7. **Agentic RAG** — adaptive retrieval strategies

---

## Documentation Updates

- [x] Architecture docs updated with re-ranking explanation
- [ ] Configuration docs updated with rerank settings
- [ ] API docs updated with rerank parameter
- [ ] README updated to mention re-ranking feature
