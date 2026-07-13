import re

_COLLECTION_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def collection_name(book_name: str) -> str:
    """Convert a book filename into a safe Qdrant collection name.

    Strips non-alphanumeric characters (except underscore/hyphen),
    lowercases, and falls back to "book" if empty.

    Args:
        book_name: Original book name or filename (e.g. "investor-tom1.pdf").

    Returns:
        Sanitized collection name (e.g. "investor-tom1").
    """
    safe = _COLLECTION_SAFE.sub("_", book_name).strip("_").lower()
    return safe or "book"
