import json
from pathlib import Path

import pytest

from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book

LABELS_PATH = Path(__file__).parent / "labels.json"


def _load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


@pytest.mark.eval
class TestFullPipelineIndexing:
    def test_chunks_are_indexed(self, indexed_qdrant):
        client, coll_name, chunks = indexed_qdrant
        collections = list_collections(client)
        assert coll_name in collections

        count_result = client.count(collection_name=coll_name, exact=True)
        assert count_result.count == len(chunks)

    def test_chunks_have_required_fields(self, indexed_qdrant):
        _, _, chunks = indexed_qdrant
        for chunk in chunks:
            assert "text" in chunk and chunk["text"], "chunk must have non-empty text"
            assert "book" in chunk and chunk["book"], "chunk must have non-empty book"
            assert "start_page" in chunk and chunk["start_page"] >= 1
            assert "end_page" in chunk and chunk["end_page"] >= chunk["start_page"]

    def test_vectors_match_chunk_count(self, indexed_qdrant):
        client, coll_name, chunks = indexed_qdrant
        result, _ = client.scroll(
            collection_name=coll_name, limit=100, with_vectors=True
        )
        assert len(result) == len(chunks)
        for point in result:
            assert len(point.vector) == 384


@pytest.mark.eval
class TestRetrievalQuality:
    def test_paris_query_retrieves_france_chunks(self, indexed_qdrant):
        results = search_book("What is the capital of France?", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "paris" in text_lower or "france" in text_lower
        assert top["score"] > 0.3

    def test_berlin_query_retrieves_germany_chunks(self, indexed_qdrant):
        results = search_book("Tell me about Berlin", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "berlin" in text_lower or "germany" in text_lower
        assert top["score"] > 0.3

    def test_tokyo_query_retrieves_japan_chunks(self, indexed_qdrant):
        results = search_book("Shibuya Crossing in Tokyo Japan", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "tokyo" in text_lower or "japan" in text_lower
        assert top["score"] > 0.3

    def test_museum_query_prefers_france(self, indexed_qdrant):
        results = search_book("famous museums and art", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "louvre" in text_lower or "paris" in text_lower or "museum" in text_lower
        )

    def test_wall_query_prefers_germany(self, indexed_qdrant):
        results = search_book("the Berlin Wall and Cold War", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "berlin wall" in text_lower
            or "cold war" in text_lower
            or "berlin" in text_lower
        )

    def test_fuji_query_prefers_japan(self, indexed_qdrant):
        results = search_book("Mount Fuji and cherry blossoms", book="tiny_sample")
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "fuji" in text_lower
            or "cherry" in text_lower
            or "sakura" in text_lower
            or "japan" in text_lower
        )


@pytest.mark.eval
class TestFormattedOutput:
    def test_citations_have_polish_format(self, indexed_qdrant):
        results = search_book("capital of France", book="tiny_sample")
        formatted = format_fragments_for_prompt(results)

        assert "\u0179r\u00f3d\u0142o:" in formatted
        assert "str." in formatted

    def test_numbered_blocks(self, indexed_qdrant):
        results = search_book("France", book="tiny_sample")
        formatted = format_fragments_for_prompt(results)

        assert "[1]" in formatted
        if len(results) > 1:
            assert "[2]" in formatted

    def test_separator_between_blocks(self, indexed_qdrant):
        results = search_book("France", book="tiny_sample")
        formatted = format_fragments_for_prompt(results)

        assert "---" in formatted


@pytest.mark.eval
class TestRetrievalMetrics:
    """Evaluate retrieval quality using precision, recall, and MRR over labeled data."""

    def test_recall_at_2(self, request, indexed_qdrant):
        """Every relevant chunk should appear in top-2 results."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample")
            r, _, _ = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 = {avg_recall:.2f} (threshold: 0.80)")
        assert avg_recall >= 0.8, f"Recall@2 = {avg_recall:.2f}, expected >= 0.8"

    def test_precision_at_2(self, request, indexed_qdrant):
        """At least half of top-2 results should be relevant."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample")
            _, p, _ = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 = {avg_precision:.2f} (threshold: 0.50)")
        assert avg_precision >= 0.5, f"Precision@2 = {avg_precision:.2f}, expected >= 0.5"

    def test_mrr(self, request, indexed_qdrant):
        """First relevant result should appear early (MRR measures rank position)."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample")
            _, _, rr = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr = {avg_mrr:.2f} (threshold: 0.70)")
        assert avg_mrr >= 0.7, f"MRR = {avg_mrr:.2f}, expected >= 0.7"


# ---------------------------------------------------------------------------
# Cross-Encoder Re-ranking Tests
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestRerankRetrievalQuality:
    """Test retrieval quality WITH cross-encoder re-ranking enabled.

    These mirror TestRetrievalQuality but call search_book(rerank=True).
    Run separately in CI to compare against the bi-encoder baseline.
    """

    def test_paris_query_reranked(self, indexed_qdrant):
        results = search_book(
            "What is the capital of France?", book="tiny_sample", rerank=True
        )
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "paris" in text_lower or "france" in text_lower
        assert "rerank_score" in top, "rerank_score must be present when rerank=True"

    def test_berlin_query_reranked(self, indexed_qdrant):
        results = search_book("Tell me about Berlin", book="tiny_sample", rerank=True)
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "berlin" in text_lower or "germany" in text_lower
        assert "rerank_score" in top

    def test_tokyo_query_reranked(self, indexed_qdrant):
        results = search_book(
            "Shibuya Crossing in Tokyo Japan", book="tiny_sample", rerank=True
        )
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert "tokyo" in text_lower or "japan" in text_lower
        assert "rerank_score" in top

    def test_museum_query_reranked(self, indexed_qdrant):
        results = search_book("famous museums and art", book="tiny_sample", rerank=True)
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "louvre" in text_lower or "paris" in text_lower or "museum" in text_lower
        )

    def test_wall_query_reranked(self, indexed_qdrant):
        results = search_book(
            "the Berlin Wall and Cold War", book="tiny_sample", rerank=True
        )
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "berlin wall" in text_lower
            or "cold war" in text_lower
            or "berlin" in text_lower
        )

    def test_fuji_query_reranked(self, indexed_qdrant):
        results = search_book(
            "Mount Fuji and cherry blossoms", book="tiny_sample", rerank=True
        )
        assert len(results) > 0

        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "fuji" in text_lower
            or "cherry" in text_lower
            or "sakura" in text_lower
            or "japan" in text_lower
        )


@pytest.mark.rerank
class TestRerankMetrics:
    """Evaluate retrieval quality WITH re-ranking enabled.

    Runs the same labeled queries as TestRetrievalMetrics but with rerank=True.
    Expect same or better scores compared to bi-encoder baseline.
    """

    def test_recall_at_2_reranked(self, request, indexed_qdrant):
        """Recall@2 with reranking should meet baseline threshold."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample", rerank=True)
            r, _, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 (reranked) = {avg_recall:.2f} (threshold: 0.80)")
        assert avg_recall >= 0.8, f"Recall@2 = {avg_recall:.2f}, expected >= 0.8"

    def test_precision_at_2_reranked(self, request, indexed_qdrant):
        """Precision@2 with reranking should meet or exceed baseline."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample", rerank=True)
            _, p, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 (reranked) = {avg_precision:.2f} (threshold: 0.50)")
        assert avg_precision >= 0.5, f"Precision@2 = {avg_precision:.2f}, expected >= 0.5"

    def test_mrr_reranked(self, request, indexed_qdrant):
        """MRR with reranking should meet or exceed baseline."""
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book="tiny_sample", rerank=True)
            _, _, rr = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr (reranked) = {avg_mrr:.2f} (threshold: 0.70)")
        assert avg_mrr >= 0.7, f"MRR = {avg_mrr:.2f}, expected >= 0.7"


@pytest.mark.rerank
class TestRerankBehavior:
    """Verify reranker mechanics: scores present, ordering correct."""

    def test_rerank_score_populated(self, indexed_qdrant):
        """All results must have rerank_score when rerank=True."""
        results = search_book("France", book="tiny_sample", rerank=True)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" in r, f"Missing rerank_score in result: {r}"
            assert isinstance(r["rerank_score"], float)

    def test_results_sorted_by_rerank_score(self, indexed_qdrant):
        """Results should be sorted by rerank_score descending."""
        results = search_book("France", book="tiny_sample", rerank=True)
        assert len(results) > 1
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by rerank_score"

    def test_rerank_returns_fewer_or_equal_results(self, indexed_qdrant):
        """Reranking should not produce more results than requested top_k."""
        results = search_book("France", book="tiny_sample", rerank=True)
        assert len(results) <= 8, f"Expected <= 8 results, got {len(results)}"

    def test_rerank_disabled_has_no_rerank_score(self, indexed_qdrant):
        """Without rerank, results should NOT have rerank_score."""
        results = search_book("France", book="tiny_sample", rerank=False)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" not in r, "Unexpected rerank_score when rerank=False"
