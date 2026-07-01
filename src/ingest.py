import json
import os
import re
import glob

import fitz

from src.config import BOOKS_DIR, EXTRACTED_DIR, CHUNKS_FILE, METADATA_FILE
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_COLLECTION
from src.embeddings import get_model, embed
from src.qdrant_store import get_qdrant_client, ensure_collection

CHAPTER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Rozdział\s+[IVXLCDM\d]+|Chapter\s+\d+|CZĘŚĆ\s+[IVXLCDM\d]+|Tom\s+\d+)",
    re.IGNORECASE,
)


def extract_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text.strip(),
            "labels": page.get_labels(),
        })
    doc.close()
    return pages


def detect_chapter(text: str) -> str | None:
    match = re.search(r"(Rozdział\s+[\dIVXLCDM]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(Chapter\s+\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(CZĘŚĆ\s+[\dIVXLCDM]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def token_count(text: str) -> int:
    model = get_model()
    return len(model.tokenizer.encode(text))


def chunk_text(text: str, pages: list[int]) -> list[dict]:
    model = get_model()
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text)
    chunks = []

    char_pos = 0
    chunk_start_page = pages[0] if pages else 1

    while char_pos < len(text):
        end_pos = min(char_pos + int(CHUNK_SIZE * 4), len(text))
        chunk_text_excerpt = text[char_pos:end_pos]

        chunk_tokens = tokenizer.encode(chunk_text_excerpt)
        while len(chunk_tokens) > CHUNK_SIZE and end_pos > char_pos + 50:
            end_pos -= 10
            chunk_text_excerpt = text[char_pos:end_pos]
            chunk_tokens = tokenizer.encode(chunk_text_excerpt)

        chunk_text_excerpt = chunk_text_excerpt.strip()
        if not chunk_text_excerpt:
            char_pos = end_pos
            continue

        chunk_end_page = _find_page_for_position(text, pages, end_pos)

        chunks.append({
            "text": chunk_text_excerpt,
            "start_page": chunk_start_page,
            "end_page": chunk_end_page,
        })

        overlap_chars = int(CHUNK_OVERLAP * 4)
        char_pos = max(end_pos - overlap_chars, char_pos + 1)
        chunk_start_page = _find_page_for_position(text, pages, char_pos)

    return chunks


def _find_page_for_position(text: str, page_boundaries: list[int], position: int) -> int:
    """Find which page a character position falls on, given page boundary positions.

    page_boundaries is a list of character positions where each page ends.
    """
    for page_idx, boundary in enumerate(page_boundaries):
        if position < boundary:
            return page_idx + 1
    return len(page_boundaries)


def get_page_boundaries(pages_data: list[dict]) -> list[int]:
    """Get cumulative character positions for page boundaries."""
    boundaries = []
    total = 0
    for p in pages_data:
        total += len(p["text"]) + 1
        boundaries.append(total)
    return boundaries


def get_full_text_with_page_info(pages_data: list[dict]) -> tuple[str, list[int]]:
    """Join all pages with page markers and track page boundaries."""
    segments = []
    page_nums = []
    for p in pages_data:
        segments.append(p["text"])
        page_nums.append(p["page_num"])
    full_text = "\n".join(segments)
    boundaries = get_page_boundaries(pages_data)
    return full_text, boundaries


def process_book(pdf_path: str) -> dict:
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"  Processing: {book_name}")

    pages_data = extract_pdf(pdf_path)
    full_text, page_boundaries = get_full_text_with_page_info(pages_data)

    print(f"    Pages: {len(pages_data)}, text length: {len(full_text)} chars")

    chunks = chunk_text(full_text, [p["page_num"] for p in pages_data])

    result_chunks = []
    current_chapter = None
    for chunk in chunks:
        detected = detect_chapter(chunk["text"])
        if detected:
            current_chapter = detected
        result_chunks.append({
            "text": chunk["text"],
            "book": book_name,
            "chapter": current_chapter or "unknown",
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
        })

    extracted_path = os.path.join(EXTRACTED_DIR, f"{book_name}.txt")
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"    Chunks created: {len(result_chunks)}")
    return {"book": book_name, "chunks": result_chunks, "total_pages": len(pages_data)}


def ingest_all():
    pdf_files = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.pdf")))
    if not pdf_files:
        print("No PDF files found in books/ directory.")
        return

    qdrant = get_qdrant_client()
    ensure_collection(qdrant)

    all_chunks = []

    for pdf_path in pdf_files:
        result = process_book(pdf_path)
        all_chunks.extend(result["chunks"])

    if not all_chunks:
        print("No chunks to index.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    vectors = embed(texts)

    print("Storing in Qdrant...")
    points = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        points.append({
            "id": i + 1,
            "vector": vector,
            "payload": {
                "text": chunk["text"],
                "book": chunk["book"],
                "chapter": chunk["chapter"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            },
        })

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    metadata = {
        "books": [os.path.splitext(os.path.basename(p))[0] for p in pdf_files],
        "pdf_files": pdf_files,
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": "intfloat/multilingual-e5-small",
    }
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Indexed {len(all_chunks)} chunks from {len(pdf_files)} books.")


if __name__ == "__main__":
    ingest_all()
