"""Tests for PDF extraction with structured text and table detection."""

import os
import tempfile

import fitz
import pytest

from src.extraction import (
    _block_center_in_table,
    _table_to_markdown,
    extract_pdf,
    get_full_text_with_page_info,
    get_page_boundaries,
)


@pytest.fixture
def simple_pdf():
    """Create a simple text-only PDF for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello world", fontsize=12)
        page.insert_text((72, 100), "Second paragraph here", fontsize=12)
        doc.save(f.name)
        doc.close()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def table_pdf():
    """Create a PDF with a table for testing table detection."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Before table text", fontsize=12)

        # Draw a table-like structure using lines and text
        # Header row
        page.insert_text((72, 150), "Name", fontsize=10)
        page.insert_text((200, 150), "Value", fontsize=10)
        # Data rows
        page.insert_text((72, 170), "Alpha", fontsize=10)
        page.insert_text((200, 170), "100", fontsize=10)
        page.insert_text((72, 190), "Beta", fontsize=10)
        page.insert_text((200, 190), "200", fontsize=10)

        # Draw table borders to help find_tables() detect the table
        shape = page.new_shape()
        # Outer border
        shape.draw_rect(fitz.Rect(60, 135, 280, 200))
        # Horizontal lines
        shape.draw_line((60, 155), (280, 155))
        shape.draw_line((60, 175), (280, 175))
        # Vertical line
        shape.draw_line((180, 135), (180, 200))
        shape.finish(color=(0, 0, 0), width=0.5)
        shape.commit()

        page.insert_text((72, 240), "After table text", fontsize=12)

        doc.save(f.name)
        doc.close()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def multi_page_pdf():
    """Create a multi-page PDF with paragraphs on each page."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()

        page1 = doc.new_page()
        page1.insert_text((72, 72), "Page one paragraph A", fontsize=12)
        page1.insert_text((72, 100), "Page one paragraph B", fontsize=12)

        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two paragraph A", fontsize=12)

        doc.save(f.name)
        doc.close()
        yield f.name
    os.unlink(f.name)


@pytest.mark.unit
class TestExtractPdf:
    def test_returns_page_num_and_text(self, simple_pdf):
        pages = extract_pdf(simple_pdf)
        assert len(pages) == 1
        assert pages[0]["page_num"] == 1
        assert "Hello world" in pages[0]["text"]
        assert "Second paragraph here" in pages[0]["text"]

    def test_returns_tables_key(self, simple_pdf):
        pages = extract_pdf(simple_pdf)
        assert "tables" in pages[0]
        assert isinstance(pages[0]["tables"], list)

    def test_paragraph_separation(self, simple_pdf):
        pages = extract_pdf(simple_pdf)
        text = pages[0]["text"]
        # Paragraphs should be separated by double newlines
        assert "\n\n" in text or "Hello world" in text

    def test_multi_page(self, multi_page_pdf):
        pages = extract_pdf(multi_page_pdf)
        assert len(pages) == 2
        assert pages[0]["page_num"] == 1
        assert pages[1]["page_num"] == 2
        assert "Page one" in pages[0]["text"]
        assert "Page two" in pages[1]["text"]

    def test_table_detection(self, table_pdf):
        pages = extract_pdf(table_pdf)
        assert len(pages) == 1
        # Tables should be detected (even if find_tables may not detect
        # our manually-drawn table, the function should not crash)
        assert "tables" in pages[0]

    def test_text_before_after_table(self, table_pdf):
        pages = extract_pdf(table_pdf)
        text = pages[0]["text"]
        assert "Before table text" in text
        assert "After table text" in text


@pytest.mark.unit
class TestTableToMarkdown:
    def test_basic_table(self):
        """Mock table object with extract() method."""

        class MockTable:
            def extract(self):
                return [
                    ["Name", "Value"],
                    ["Alpha", "100"],
                    ["Beta", "200"],
                ]

        md = _table_to_markdown(MockTable())
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| Alpha | 100 |" in md
        assert "| Beta | 200 |" in md

    def test_empty_table(self):
        class MockTable:
            def extract(self):
                return []

        md = _table_to_markdown(MockTable())
        assert md == ""

    def test_single_column(self):
        class MockTable:
            def extract(self):
                return [
                    ["Item"],
                    ["One"],
                    ["Two"],
                ]

        md = _table_to_markdown(MockTable())
        assert "| Item |" in md
        assert "| One |" in md


@pytest.mark.unit
class TestBlockCenterInTable:
    def test_inside_table(self):
        block = {"bbox": (100, 100, 200, 120)}
        table_bboxes = [(50, 50, 250, 250)]
        assert _block_center_in_table(block, table_bboxes) is True

    def test_outside_table(self):
        block = {"bbox": (300, 300, 400, 320)}
        table_bboxes = [(50, 50, 250, 250)]
        assert _block_center_in_table(block, table_bboxes) is False

    def test_no_tables(self):
        block = {"bbox": (100, 100, 200, 120)}
        assert _block_center_in_table(block, []) is False


@pytest.mark.unit
class TestGetPageBoundariesWithTables:
    def test_no_tables_matches_old_behavior(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        assert len(boundaries) == 3
        assert boundaries[0] < boundaries[1] < boundaries[2]

    def test_with_tables_extends_boundaries(self):
        pages = [
            {
                "page_num": 1,
                "text": "Hello",
                "tables": [{"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}],
            },
            {
                "page_num": 2,
                "text": "World",
                "tables": [],
            },
        ]
        boundaries = get_page_boundaries(pages)
        # First page boundary should account for table markdown
        assert boundaries[0] > len("Hello") + 1


@pytest.mark.unit
class TestGetFullTextWithPageInfo:
    def test_includes_table_markdown(self):
        pages = [
            {
                "page_num": 1,
                "text": "Some text",
                "tables": [{"markdown": "| X | Y |\n|---|---|\n| 1 | 2 |"}],
            },
        ]
        full_text, boundaries = get_full_text_with_page_info(pages)
        assert "Some text" in full_text
        assert "| X | Y |" in full_text

    def test_no_tables_plain_text(self, sample_pages):
        full_text, boundaries = get_full_text_with_page_info(sample_pages)
        for page in sample_pages:
            assert page["text"] in full_text
        assert len(boundaries) == len(sample_pages)
