import json
from pathlib import Path

import pytest

from src.config import RERANK_TOP_N
from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book

LABELS_PATH = Path(__file__).parent / "benchmark_labels.json"
BOOK = "eval_benchmark"

# ---------------------------------------------------------------------------
# Cached search helpers — each query searched exactly ONCE per pipeline stage
# ---------------------------------------------------------------------------

_bi_cache = {}  # query -> list[dict] (up to RERANK_TOP_N results)
_ce_cache = {}  # query -> SearchResult (with .fragments and .trace)


def _bi(query, top_k=8):
    """Get bi-encoder results for a query (cached, top_k=20 stored)."""
    if query not in _bi_cache:
        _bi_cache[query] = search_book(
            query, book=BOOK, rerank=False, top_k=RERANK_TOP_N
        )
    return _bi_cache[query][:top_k]


def _bi_all(query):
    """Get all bi-encoder candidates (cached, up to RERANK_TOP_N)."""
    if query not in _bi_cache:
        _bi_cache[query] = search_book(
            query, book=BOOK, rerank=False, top_k=RERANK_TOP_N
        )
    return _bi_cache[query]


def _ce(query):
    """Get cross-encoder results for a query (cached, with trace)."""
    if query not in _ce_cache:
        _ce_cache[query] = search_book(
            query, book=BOOK, rerank=True, trace=True
        )
    return _ce_cache[query]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def _answerable_labels():
    return [
        lb for lb in _load_labels() if lb.get("category") != "no_answer"
    ]


def _no_answer_labels():
    return [
        lb for lb in _load_labels() if lb.get("category") == "no_answer"
    ]


# ---------------------------------------------------------------------------
# Indexing Tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestFullPipelineIndexing:
    def test_chunks_are_indexed(self, benchmark_indexed_qdrant):
        client, coll_name, chunks = benchmark_indexed_qdrant
        collections = list_collections(client)
        assert coll_name in collections

        count_result = client.count(
            collection_name=coll_name, exact=True
        )
        assert count_result.count == len(chunks)

    def test_chunks_have_required_fields(self, benchmark_indexed_qdrant):
        _, _, chunks = benchmark_indexed_qdrant
        for chunk in chunks:
            assert "text" in chunk and chunk["text"], (
                "chunk must have non-empty text"
            )
            assert (
                "source_file" in chunk and chunk["source_file"]
            ), "chunk must have source_file"

    def test_vectors_match_chunk_count(self, benchmark_indexed_qdrant):
        client, coll_name, chunks = benchmark_indexed_qdrant
        count_result = client.count(
            collection_name=coll_name, exact=True
        )
        assert count_result.count == len(chunks)
        result, _ = client.scroll(
            collection_name=coll_name,
            limit=count_result.count,
            with_vectors=True,
        )
        from src.config import EMBED_DIM

        for point in result:
            assert len(point.vector) == EMBED_DIM


# ---------------------------------------------------------------------------
# Bi-Encoder Quality Tests (uncached, 4 ad-hoc queries)
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestRetrievalQuality:
    def test_game_query(self, benchmark_indexed_qdrant):
        results = search_book(
            "What do keybullet kin drop?", book=BOOK
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "key" in text_lower or "drop" in text_lower
        assert top["score"] > 0.2

    def test_rpg_query(self, benchmark_indexed_qdrant):
        results = search_book(
            "What happens on day 2 of the campaign?", book=BOOK
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "day" in text_lower or "2" in text_lower
        assert top["score"] > 0.2

    def test_tech_query(self, benchmark_indexed_qdrant):
        results = search_book(
            "What is STICI-note and how does it work?",
            book=BOOK,
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "stici" in text_lower or "chatbot" in text_lower
        assert top["score"] > 0.2

    def test_retrieval_system_query(self, benchmark_indexed_qdrant):
        results = search_book(
            "What are the key components of a RAG system?",
            book=BOOK,
        )
        assert len(results) > 0
        top = results[0]
        assert top["score"] > 0.2


# ---------------------------------------------------------------------------
# Formatted Output Tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestFormattedOutput:
    def test_citations_have_polish_format(
        self, benchmark_indexed_qdrant
    ):
        results = search_book("keybullet kin", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "\u0179r\u00f3d\u0142o:" in formatted

    def test_numbered_blocks(self, benchmark_indexed_qdrant):
        results = search_book("keybullet kin", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "[1]" in formatted
        if len(results) > 1:
            assert "[2]" in formatted

    def test_separator_between_blocks(self, benchmark_indexed_qdrant):
        results = search_book("keybullet kin", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "---" in formatted


# ---------------------------------------------------------------------------
# Bi-Encoder Metric Tests (cached: 80 queries searched once)
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestRetrievalMetrics:
    """Evaluate retrieval quality on answerable queries only."""

    def test_recall_at_2(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        recalls = []
        for item in labels:
            results = _bi(item["query"])
            r, _, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            recalls.append(r)
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(f"\n  recall@2 = {avg:.2f} (threshold: 0.80)")
        assert avg >= 0.80, f"Recall@2 = {avg:.2f}, expected >= 0.80"

    def test_recall_at_4(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _recall_at_k

        labels = _answerable_labels()
        recalls = []
        for item in labels:
            results = _bi(item["query"])
            recalls.append(
                _recall_at_k(results, item["relevant_documents"], 4)
            )
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(f"\n  recall@4 = {avg:.2f} (threshold: 0.83)")
        assert avg >= 0.83, f"Recall@4 = {avg:.2f}, expected >= 0.83"

    def test_precision_at_2(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        precisions = []
        for item in labels:
            results = _bi(item["query"])
            _, p, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            precisions.append(p)
        avg = sum(precisions) / len(precisions) if precisions else 0
        print(f"\n  precision@2 = {avg:.2f} (threshold: 0.70)")
        assert avg >= 0.70, (
            f"Precision@2 = {avg:.2f}, expected >= 0.70"
        )

    def test_precision_at_4(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _precision_at_k

        labels = _answerable_labels()
        precisions = []
        for item in labels:
            results = _bi(item["query"])
            precisions.append(
                _precision_at_k(
                    results, item["relevant_documents"], 4
                )
            )
        avg = sum(precisions) / len(precisions) if precisions else 0
        print(f"\n  precision@4 = {avg:.2f} (threshold: 0.65)")
        assert avg >= 0.65, (
            f"Precision@4 = {avg:.2f}, expected >= 0.65"
        )

    def test_mrr(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        rrs = []
        for item in labels:
            results = _bi(item["query"])
            _, _, rr = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            rrs.append(rr)
        avg = sum(rrs) / len(rrs) if rrs else 0
        print(f"\n  mrr = {avg:.2f} (threshold: 0.80)")
        assert avg >= 0.80, f"MRR = {avg:.2f}, expected >= 0.80"


# ---------------------------------------------------------------------------
# Cross-Encoder Quality Tests (uncached, 4 ad-hoc queries)
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestRerankRetrievalQuality:
    """Test retrieval quality WITH cross-encoder re-ranking enabled."""

    def test_game_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "What do keybullet kin drop?",
            book=BOOK,
            rerank=True,
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "key" in text_lower or "drop" in text_lower
        assert "rerank_score" in top, (
            "rerank_score must be present when rerank=True"
        )

    def test_rpg_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "What happens on day 2 of the campaign?",
            book=BOOK,
            rerank=True,
        )
        assert len(results) > 0
        top = results[0]
        assert "rerank_score" in top

    def test_tech_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "What is STICI-note and how does it work?",
            book=BOOK,
            rerank=True,
        )
        assert len(results) > 0
        top = results[0]
        assert "rerank_score" in top

    def test_retrieval_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "What are the key components of a RAG system?",
            book=BOOK,
            rerank=True,
        )
        assert len(results) > 0
        assert "rerank_score" in results[0]


# ---------------------------------------------------------------------------
# Cross-Encoder Metric Tests (cached: 80 queries searched once)
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestRerankMetrics:
    """Evaluate retrieval quality WITH re-ranking on answerable queries."""

    def test_recall_at_2_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        recalls = []
        for item in labels:
            sr = _ce(item["query"])
            results = sr.fragments
            r, _, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            recalls.append(r)
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(
            f"\n  recall@2 (reranked) = {avg:.2f}"
            " (threshold: 0.70)"
        )
        assert avg >= 0.70, (
            f"Recall@2 = {avg:.2f}, expected >= 0.70"
        )

    def test_recall_at_4_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _recall_at_k

        labels = _answerable_labels()
        recalls = []
        for item in labels:
            sr = _ce(item["query"])
            recalls.append(
                _recall_at_k(
                    sr.fragments, item["relevant_documents"], 4
                )
            )
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(
            f"\n  recall@4 (reranked) = {avg:.2f}"
            " (threshold: 0.78)"
        )
        assert avg >= 0.78, (
            f"Recall@4 = {avg:.2f}, expected >= 0.78"
        )

    def test_precision_at_2_reranked(
        self, request, benchmark_indexed_qdrant
    ):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        precisions = []
        for item in labels:
            sr = _ce(item["query"])
            _, p, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                sr.fragments,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            precisions.append(p)
        avg = sum(precisions) / len(precisions) if precisions else 0
        print(
            f"\n  precision@2 (reranked) = {avg:.2f}"
            " (threshold: 0.65)"
        )
        assert avg >= 0.65, (
            f"Precision@2 = {avg:.2f}, expected >= 0.65"
        )

    def test_precision_at_4_reranked(
        self, request, benchmark_indexed_qdrant
    ):
        from tests.eval.conftest import _precision_at_k

        labels = _answerable_labels()
        precisions = []
        for item in labels:
            sr = _ce(item["query"])
            precisions.append(
                _precision_at_k(
                    sr.fragments, item["relevant_documents"], 4
                )
            )
        avg = sum(precisions) / len(precisions) if precisions else 0
        print(
            f"\n  precision@4 (reranked) = {avg:.2f}"
            " (threshold: 0.60)"
        )
        assert avg >= 0.60, (
            f"Precision@4 = {avg:.2f}, expected >= 0.60"
        )

    def test_mrr_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _answerable_labels()
        rrs = []
        for item in labels:
            sr = _ce(item["query"])
            _, _, rr = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                sr.fragments,
                item["relevant_documents"],
                k=2,
                category=item["category"],
            )
            rrs.append(rr)
        avg = sum(rrs) / len(rrs) if rrs else 0
        print(f"\n  mrr (reranked) = {avg:.2f} (threshold: 0.75)")
        assert avg >= 0.75, f"MRR = {avg:.2f}, expected >= 0.75"


# ---------------------------------------------------------------------------
# Reranker Behavior Tests (uncached, 1 ad-hoc query)
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestRerankBehavior:
    """Verify reranker mechanics: scores present, ordering correct."""

    def test_rerank_score_populated(self, benchmark_indexed_qdrant):
        results = search_book("keybullet kin", book=BOOK, rerank=True)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" in r, (
                f"Missing rerank_score in result: {r}"
            )
            assert isinstance(r["rerank_score"], float)

    def test_results_sorted_by_rerank_score(
        self, benchmark_indexed_qdrant
    ):
        results = search_book("keybullet kin", book=BOOK, rerank=True)
        assert len(results) > 1
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(
            scores, reverse=True
        ), "Results not sorted by rerank_score"

    def test_rerank_returns_fewer_or_equal_results(
        self, benchmark_indexed_qdrant
    ):
        results = search_book("keybullet kin", book=BOOK, rerank=True)
        assert len(results) <= 8, (
            f"Expected <= 8 results, got {len(results)}"
        )

    def test_rerank_disabled_has_no_rerank_score(
        self, benchmark_indexed_qdrant
    ):
        results = search_book("keybullet kin", book=BOOK, rerank=False)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" not in r, (
                "Unexpected rerank_score when rerank=False"
            )


# ---------------------------------------------------------------------------
# No-Answer Queries
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestNoAnswerQueries:
    """Verify no-answer queries are handled correctly.

    Our system does not yet detect unanswerable queries, so it will
    always return chunks. These tests document the current behavior.
    """

    def test_no_answer_queries_return_results(
        self, benchmark_indexed_qdrant
    ):
        labels = _no_answer_labels()
        assert len(labels) > 0, "Expected at least some no-answer labels"
        for item in labels:
            results = search_book(
                item["query"], book=BOOK, rerank=False
            )
            assert isinstance(results, list)

    def test_no_answer_labels_have_no_answer_text(self):
        labels = _no_answer_labels()
        for item in labels:
            assert (
                "answer_text" not in item or not item.get("answer_text")
            ), (
                "No-answer query should not have answer_text: "
                f"{item['query']}"
            )


# ---------------------------------------------------------------------------
# Two-Stage Pipeline Comparison (reuses bi+ce cache)
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestPipelineComparison:
    """Run both stages of the retrieval pipeline in one session.

    Stage 1: Bi-encoder retrieves candidates (baseline metrics)
    Stage 2: Cross-encoder re-ranks candidates (final metrics)

    Both stages are stored on the session so the report shows a single
    comparison table with Before -> After -> Delta, plus per-query
    reranking detail showing how items were reordered.
    """

    def test_pipeline_comparison(self, request, benchmark_indexed_qdrant):
        from src.trace import SearchResult
        from tests.eval.conftest import (
            collect_eval_result,
            collect_rerank_detail,
            collect_rerank_result,
        )

        labels = _answerable_labels()
        recalls_before, recalls_after = [], []
        precisions_before, precisions_after = [], []
        rrs_before, rrs_after = [], []

        for item in labels:
            query = item["query"]
            relevant = item["relevant_documents"]
            cat = item["category"]

            # Stage 1: bi-encoder only (retrieve all candidates)
            results_before = _bi_all(query)
            r1, p1, rr1 = collect_rerank_result(
                request.session,
                query,
                results_before,
                relevant,
                k=2,
                category=cat,
            )
            recalls_before.append(r1)
            precisions_before.append(p1)
            rrs_before.append(rr1)

            # Stage 2: bi-encoder + cross-encoder (cached)
            search_result = _ce(query)
            assert isinstance(search_result, SearchResult)
            results_after = search_result.fragments

            rank_changes = None
            for stage in search_result.trace.stages:
                if stage.name == "rerank":
                    rank_changes = stage.details.get("rank_changes")
                    break

            r2, p2, rr2 = collect_eval_result(
                request.session,
                f"{query} (pipeline)",
                results_after,
                relevant,
                k=2,
                category=cat,
            )
            recalls_after.append(r2)
            precisions_after.append(p2)
            rrs_after.append(rr2)

            collect_rerank_detail(
                request.session,
                query,
                results_before,
                results_after,
                rank_changes,
                relevant,
            )

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0

        m_before = {
            "recall": avg(recalls_before),
            "precision": avg(precisions_before),
            "mrr": avg(rrs_before),
        }
        m_after = {
            "recall": avg(recalls_after),
            "precision": avg(precisions_after),
            "mrr": avg(rrs_after),
        }

        print("\n")
        print("=" * 60)
        print("TWO-STAGE PIPELINE COMPARISON")
        print("=" * 60)
        print(
            f"  {'Metric':<15} {'Bi-Encoder':>12}"
            f" {'+Rerank':>12} {'Delta':>10}"
        )
        print(f"  {'-' * 49}")
        for key, label in [
            ("recall", "Recall@2"),
            ("precision", "Precision@2"),
            ("mrr", "MRR"),
        ]:
            b, a = m_before[key], m_after[key]
            d = a - b
            sign = "+" if d > 0 else ""
            print(
                f"  {label:<15} {b:>12.2f}"
                f" {a:>12.2f} {sign}{d:>9.2f}"
            )
        print("=" * 60)
