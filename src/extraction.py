"""PDF text extraction and page-boundary utilities."""

from __future__ import annotations

import fitz


def _point_in_table(point: tuple[float, float], table_bboxes: list[tuple]) -> bool:
    """Check if a point falls inside any table bounding box."""
    x, y = point
    for bbox in table_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _block_center_in_table(block: dict, table_bboxes: list[tuple]) -> bool:
    """Check if a block's center point is inside any table bbox."""
    x0, y0, x1, y1 = block["bbox"]
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    return _point_in_table(center, table_bboxes)


def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF Table object to markdown-formatted text."""
    data = table.extract()
    if not data or not data[0]:
        return ""

    # Normalize: ensure all rows have same column count
    max_cols = max(len(row) for row in data)
    normalized = []
    for row in data:
        padded = list(row) + [""] * (max_cols - len(row))
        normalized.append([str(cell).strip() if cell else "" for cell in padded])

    lines = []
    # Header row
    lines.append("| " + " | ".join(normalized[0]) + " |")
    # Separator
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # Data rows
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF using PyMuPDF.

    Uses structured block-level extraction to preserve paragraph boundaries
    and detect tables. Tables are extracted separately and formatted as
    markdown, while regular text blocks are reconstructed with proper
    paragraph separation.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with 'page_num' (1-indexed), 'text' (paragraph-aware),
        and 'tables' (list of table metadata dicts) per page.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        page_num = i + 1

        # Detect tables and collect their bboxes
        table_finder = page.find_tables()
        table_bboxes = [t.bbox for t in table_finder.tables]

        # Extract tables as markdown
        tables = []
        for table in table_finder.tables:
            md = table.to_markdown()
            if md:
                tables.append({
                    "page_num": page_num,
                    "bbox": table.bbox,
                    "rows": table.row_count,
                    "cols": table.col_count,
                    "markdown": md,
                    "confidence": 1.0,
                })

        # Extract text blocks, skipping table regions
        page_dict = page.get_text("dict", sort=True)
        paragraphs: list[str] = []

        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            if _block_center_in_table(block, table_bboxes):
                continue

            block_lines: list[str] = []
            for line in block.get("lines", []):
                span_text = "".join(span["text"] for span in line.get("spans", []))
                if span_text.strip():
                    block_lines.append(span_text)
            if block_lines:
                paragraphs.append("\n".join(block_lines))

        text = "\n\n".join(paragraphs)
        pages.append({
            "page_num": page_num,
            "text": text.strip(),
            "tables": tables,
        })

    doc.close()
    return pages


def _page_segment_length(p: dict) -> int:
    """Compute the total character length of a page's text plus table markdown."""
    length = len(p["text"])
    for table in p.get("tables", []):
        md = table.get("markdown", "")
        if md:
            length += len(md) + 2  # +2 for the "\n\n" separator
    return length


def get_page_boundaries(pages_data: list[dict]) -> list[int]:
    """Calculate cumulative character positions marking page boundaries.

    Each page's contribution includes its text plus any inline table markdown,
    matching the layout produced by get_full_text_with_page_info().

    Args:
        pages_data: List of dicts with 'text' and optional 'tables' per page.

    Returns:
        List of cumulative character offsets (one per page).
    """
    boundaries = []
    total = 0
    for p in pages_data:
        total += _page_segment_length(p) + 1  # +1 for newline separator
        boundaries.append(total)
    return boundaries


def get_full_text_with_page_info(pages_data: list[dict]) -> tuple[str, list[int]]:
    """Join all page texts into one string and compute page boundary offsets.

    Each page's text is concatenated with table markdown appended inline
    so tables appear in the final text at their correct page position.

    Args:
        pages_data: List of dicts with 'page_num', 'text', and optional
            'tables' per page.

    Returns:
        Tuple of (full_text, page_boundaries) where page_boundaries is a
        list of cumulative character positions marking where each page ends.
    """
    segments = []
    page_nums = []
    for p in pages_data:
        page_text = p["text"]
        # Append table markdown inline after the page text
        for table in p.get("tables", []):
            if table.get("markdown"):
                page_text += "\n\n" + table["markdown"]
        segments.append(page_text)
        page_nums.append(p["page_num"])
    full_text = "\n".join(segments)
    boundaries = get_page_boundaries(pages_data)
    return full_text, boundaries


def page_at_position(page_boundaries: list[int], page_nums: list[int], pos: int) -> int:
    """Find which page a character position falls on.

    Args:
        page_boundaries: Cumulative character offsets per page.
        page_nums: Corresponding page numbers (1-indexed).
        pos: Character position in the full text.

    Returns:
        Page number containing the given position.
    """
    for i, boundary in enumerate(page_boundaries):
        if pos < boundary:
            return page_nums[i]
    return page_nums[-1] if page_nums else 1
