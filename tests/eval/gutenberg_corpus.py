"""Fetch and split a Project Gutenberg text into chapter-based pages."""

import re
import urllib.request
from functools import lru_cache

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1232/pg1232.txt"
CHAPTER_RE = re.compile(r"CHAPTER [IVXLC]+\.?\[?1?\]?")

_CHAPTER_NUMS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
    "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20, "XXI": 21, "XXII": 22, "XXIII": 23,
    "XXIV": 24, "XXV": 25, "XXVI": 26,
}


def _chapter_num(m: re.Match) -> int:
    """Extract chapter number from a regex match."""
    raw = m.group()
    roman = raw.replace("CHAPTER ", "").split(".")[0].split("[")[0].strip()
    return _CHAPTER_NUMS.get(roman, 99)


@lru_cache(maxsize=1)
def fetch_and_split() -> tuple[str, list[int], list[int]]:
    """Fetch The Prince from Gutenberg and split into chapters.

    Returns:
        (full_text, page_boundaries, page_nums) where each chapter
        is treated as one "page" for chunk_text().
    """
    raw = urllib.request.urlopen(GUTENBERG_URL).read().decode("utf-8")
    text = raw.replace("\r\n", "\n")

    # Strip Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE PRINCE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE PRINCE ***"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1:
        text = text[start + len(start_marker):]
    if end != -1:
        text = text[:end]
    text = text.strip()

    # Find all chapter markers
    all_matches = list(CHAPTER_RE.finditer(text))
    if not all_matches:
        raise RuntimeError("No chapter markers found in Gutenberg text")

    # The text has: TOC entries (small, <200 chars) then actual content.
    # Find where actual content starts: first chapter with >200 chars to next marker
    content_start = 0
    for i, m in enumerate(all_matches):
        end_pos = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(text)
        if end_pos - m.start() > 200:
            content_start = m.start()
            break

    # Only consider matches at or after content_start, deduplicate by chapter number
    seen = set()
    chapter_matches = []
    for m in all_matches:
        if m.start() < content_start:
            continue
        num = _chapter_num(m)
        if num in seen:
            continue
        seen.add(num)
        chapter_matches.append(m)

    if not chapter_matches:
        raise RuntimeError("No full chapters found after filtering TOC entries")

    # Sort by chapter number to get correct order (XXVI may appear before I)
    chapter_matches.sort(key=_chapter_num)

    # Build pages: each chapter = one page, in chapter-number order
    page_boundaries = []
    page_nums = []
    full_parts = []

    for i, m in enumerate(chapter_matches):
        start_pos = m.start()
        # Find this chapter's end: next marker after it in original text order
        next_after = None
        for other in all_matches:
            if other.start() > start_pos:
                next_after = other
                break
        end_pos = next_after.start() if next_after else len(text)
        chapter_text = text[start_pos:end_pos].strip()
        full_parts.append(chapter_text)
        page_boundaries.append(len("\n\n".join(full_parts)))
        page_nums.append(i + 1)

    full_text = "\n\n".join(full_parts)
    return full_text, page_boundaries, page_nums


def chapter_count() -> int:
    """Return the number of chapters found."""
    _, _, page_nums = fetch_and_split()
    return len(page_nums)
