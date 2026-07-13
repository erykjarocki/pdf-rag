import json
import os
import sys
from pathlib import Path

import fitz
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import src.config as config
import src.ingest as ingest
import src.qdrant_store as qdrant_store

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings import embed
from src.ingest import process_book
from src.qdrant_store import ensure_collection

BOOK_NAME = "tiny_sample"
COLLECTION_NAME = "tiny_sample"

REPORT_PATH = Path(__file__).parent / "eval-report.json"


def pytest_sessionfinish(session, exitstatus):
    """Print terminal summary and write eval-report.json after all tests."""
    results = getattr(session, "eval_results", [])
    if not results:
        return

    # Compute metrics
    recalls = [item["recall_at_k"] for item in results]
    precisions = [item["precision_at_k"] for item in results]
    rrs = [item["reciprocal_rank"] for item in results]

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_mrr = sum(rrs) / len(rrs)

    # Terminal summary
    terminal = session.config.get_terminal_writer()
    terminal.write("\n")
    terminal.write("=" * 70 + "\n")
    terminal.write("EVAL RESULTS\n")
    terminal.write("=" * 70 + "\n\n")

    for item in results:
        terminal.write(f'Query: "{item["query"]}"\n')
        for frag in item["retrieved_fragments"]:
            relevant = frag["is_relevant"]
            mark = "  \u2713 RELEVANT" if relevant else ""
            terminal.write(
                f'  [{frag["rank"]}] score={frag["score"]:.2f}'
                f'  page={frag["start_page"]}{mark}\n'
            )
            for line in frag["text"].split("\n"):
                terminal.write(f"      {line}\n")
            terminal.write("\n")
        terminal.write("\n")

    terminal.write("-" * 70 + "\n")
    terminal.write(
        f"Recall@2: {avg_recall:.2f} | "
        f"Precision@2: {avg_precision:.2f} | "
        f"MRR: {avg_mrr:.2f}\n"
    )
    terminal.write("-" * 70 + "\n\n")

    # Write JSON report
    report = {
        "queries": results,
        "metrics": {
            "recall_at_2": round(avg_recall, 4),
            "precision_at_2": round(avg_precision, 4),
            "mrr": round(avg_mrr, 4),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def collect_eval_result(session, query, results, relevant_pages, k=2):
    """Run metrics on a query result and store on session for the summary hook."""
    if not hasattr(session, "eval_results"):
        session.eval_results = []

    # Deduplicate: only store once per query
    if any(item["query"] == query for item in session.eval_results):
        top_k = results[:k]
        return (
            _recall_at_k(results, relevant_pages, k),
            _precision_at_k(results, relevant_pages, k),
            _mrr(results, relevant_pages),
        )

    top_k = results[:k]
    rr = _mrr(results, relevant_pages)
    recall = _recall_at_k(results, relevant_pages, k)
    precision = _precision_at_k(results, relevant_pages, k)

    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "chapter": r.get("chapter", ""),
                "is_relevant": r["start_page"] in relevant_pages,
            }
        )

    session.eval_results.append(
        {
            "query": query,
            "relevant_pages": relevant_pages,
            "retrieved_fragments": fragments,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": rr,
        }
    )

    return recall, precision, rr


def _precision_at_k(results, relevant_pages, k):
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r["start_page"] in relevant_pages)
    return relevant / k


def _recall_at_k(results, relevant_pages, k):
    if not relevant_pages:
        return 1.0
    top_k = results[:k]
    found = sum(1 for r in top_k if r["start_page"] in relevant_pages)
    return found / len(relevant_pages)


def _mrr(results, relevant_pages):
    for i, r in enumerate(results, 1):
        if r["start_page"] in relevant_pages:
            return 1.0 / i
    return 0.0


@pytest.fixture(scope="module")
def tiny_pdf(tmp_path_factory):
    """Generate a 2-page PDF with headings and paragraphs about France and Germany."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = str(tmp_dir / "tiny_sample.pdf")

    doc = fitz.open()

    page = doc.new_page()
    text = (
        "Chapter 1: France\n\n"
        "Paris is the capital and most populous city of France. The city is "
        "known for its iconic Eiffel Tower, which was built in 1889 for the "
        "World's Fair. Paris attracts millions of tourists every year.\n\n"
        "The Louvre Museum in Paris is the world's largest art museum. It "
        "houses the Mona Lisa painting by Leonardo da Vinci. The museum "
        "receives over nine million visitors annually."
    )
    page.insert_text((72, 72), text, fontsize=11)

    page = doc.new_page()
    text = (
        "Chapter 2: Germany\n\n"
        "Berlin is the capital and largest city of Germany. The city is known "
        "for the Brandenburg Gate, a neoclassical monument built in the 18th "
        "century. Berlin has a rich and complex history.\n\n"
        "The Berlin Wall divided the city from 1961 to 1989. Its fall "
        "symbolized the end of the Cold War. Today Berlin is a major European "
        "cultural center."
    )
    page.insert_text((72, 72), text, fontsize=11)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def indexed_qdrant(tiny_pdf, tmp_path_factory):
    """Run the full pipeline: extract -> chunk -> detect -> embed -> store.

    Uses real embedding model, real extraction, in-memory Qdrant.
    Returns (qdrant_client, collection_name, chunks).
    """
    tmp_dir = tmp_path_factory.mktemp("data")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(COLLECTION_NAME, in_memory_client)

        result = process_book(tiny_pdf)
        chunks = result["chunks"]
        assert len(chunks) > 0, "Expected at least one chunk from the test PDF"

        texts = [c["text"] for c in chunks]
        vectors = embed(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "book": chunk["book"],
                        "chapter": chunk["chapter"],
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                    },
                )
            )

        in_memory_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, COLLECTION_NAME, chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
