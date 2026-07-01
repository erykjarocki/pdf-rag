from src.config import TOP_K
from src.embeddings import embed_query
from src.qdrant_store import get_qdrant_client
from src.config import QDRANT_COLLECTION


def search_book(query: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = embed_query(query)

    client = get_qdrant_client()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )

    fragments = []
    for hit in results:
        p = hit.payload
        fragments.append({
            "text": p["text"],
            "book": p["book"],
            "chapter": p.get("chapter", ""),
            "start_page": p.get("start_page", ""),
            "end_page": p.get("end_page", ""),
            "score": round(hit.score, 4),
        })

    return fragments


def format_fragments_for_prompt(fragments: list[dict]) -> str:
    lines = []
    for i, f in enumerate(fragments, 1):
        source = f"Źródło: {f['book']}"
        if f.get("chapter"):
            source += f", {f['chapter']}"
        if f.get("start_page"):
            source += f", str. {f['start_page']}"
            if f.get("end_page") and f["end_page"] != f["start_page"]:
                source += f"-{f['end_page']}"

        lines.append(f"[{i}] {f['text']}\n\n{source}\n---")
    return "\n".join(lines)
