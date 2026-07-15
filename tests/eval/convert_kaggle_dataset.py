"""Convert Kaggle Single-Topic RAG Evaluation Dataset to benchmark format.

Reads CSVs from kaggle_raw/ and produces:
  - benchmark_docs/doc_XX.txt  (one per document)
  - benchmark_labels.json      (all 120 questions with relevance labels)

Usage:
    python tests/eval/convert_kaggle_dataset.py
"""

import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

EVAL_DIR = Path(__file__).parent
RAW_DIR = EVAL_DIR / "kaggle_raw"
DOCS_DIR = EVAL_DIR / "benchmark_docs"
LABELS_PATH = EVAL_DIR / "benchmark_labels.json"


def load_documents():
    """Load documents.csv -> {index: {text, source_url}}."""
    docs = {}
    with open(RAW_DIR / "documents.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            docs[idx] = {
                "text": row["text"],
                "source_url": row["source_url"],
            }
    return docs


def load_questions(filename, category):
    """Load a question CSV -> list of dicts."""
    questions = []
    with open(RAW_DIR / filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = {
                "document_index": int(row["document_index"]),
                "question": row["question"],
                "category": category,
            }
            if "answer" in row and row["answer"].strip():
                q["answer"] = row["answer"].strip()
            questions.append(q)
    return questions


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    single = load_questions("single_passage_answer_questions.csv", "single_passage")
    multi = load_questions("multi_passage_answer_questions.csv", "multi_passage")
    no_answer = load_questions("no_answer_questions.csv", "no_answer")
    print(
        f"Loaded {len(single)} single-passage, {len(multi)} multi-passage, "
        f"{len(no_answer)} no-answer questions"
    )

    # Write documents as .txt files
    DOCS_DIR.mkdir(exist_ok=True)
    doc_index_to_file = {}
    for idx in sorted(docs.keys()):
        filename = f"doc_{idx:02d}.txt"
        doc_index_to_file[idx] = filename
        (DOCS_DIR / filename).write_text(docs[idx]["text"])
    print(f"Wrote {len(docs)} documents to {DOCS_DIR}/")

    # Build labels
    labels = []
    for q in single + multi + no_answer:
        doc_idx = q["document_index"]
        label = {
            "query": q["question"],
            "relevant_documents": [doc_index_to_file[doc_idx]],
            "category": q["category"],
        }
        if "answer" in q:
            label["answer_text"] = q["answer"]
        labels.append(label)

    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"Wrote {len(labels)} labels to {LABELS_PATH}")

    # Summary
    categories = {}
    for label in labels:
        cat = label["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"Categories: {categories}")


if __name__ == "__main__":
    main()
