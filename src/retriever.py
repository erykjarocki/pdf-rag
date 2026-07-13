from src.config import TOP_K, collection_name
from src.embeddings import embed_query
from src.qdrant_store import get_qdrant_client, list_collections


def search_book(query: str, top_k: int = TOP_K, book: str | None = None) -> list[dict]:
    """Search the vector database for chunks relevant to a query.

    If book is specified, searches only that collection. Otherwise searches
    all collections proportionally and merges results by score.

    Args:
        query: Natural language question or search terms.
        top_k: Maximum number of results to return (default: 8).
        book: Optional book name to filter search to a single collection.

    Returns:
        List of dicts with 'text', 'book', 'chapter', 'start_page',
        'end_page', and 'score' keys, sorted by relevance.
    """
    query_vector = embed_query(query)
    client = get_qdrant_client()

    if book:
        coll = collection_name(book)
        if coll not in list_collections(client):
            return []
        resp = client.query_points(
            collection_name=coll,
            query=query_vector,
            limit=top_k,
        )
        return _format_results(resp.points)

    collections = [c for c in list_collections(client)]
    if not collections:
        return []

    all_results = []
    per_collection = max(1, top_k // len(collections)) + 2

    for coll in collections:
        try:
            resp = client.query_points(
                collection_name=coll,
                query=query_vector,
                limit=per_collection,
            )
            all_results.extend(_format_results(resp.points))
        except Exception:
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def _format_results(points) -> list[dict]:
    """Convert Qdrant result points into a standardized dict format.

    Args:
        points: List of Qdrant ScoredPoint objects.

    Returns:
        List of dicts with text, book, chapter, pages, and score.
    """
    fragments = []
    for hit in points:
        p = hit.payload
        fragments.append(
            {
                "text": p["text"],
                "book": p["book"],
                "chapter": p.get("chapter", ""),
                "start_page": p.get("start_page", ""),
                "end_page": p.get("end_page", ""),
                "score": round(hit.score, 4),
            }
        )
    return fragments


def format_fragments_for_prompt(fragments: list[dict]) -> str:
    """Format search results as numbered text blocks with Polish citations.

    Args:
        fragments: List of fragment dicts from search_book().

    Returns:
        Formatted string with numbered blocks and source citations
        (e.g. "[1] text... Źródło: book, chapter, str. X-Y").
    """
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
