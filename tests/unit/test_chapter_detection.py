"""Tests for universal chapter detection (src/chapter_detection.py)."""

from unittest.mock import MagicMock, patch

import pytest

from src.chapter_detection import (
    ChapterDetector,
    _build_font_map,
    _build_toc_map,
    _classify_span,
    _regex_fallback,
)

# ---------------------------------------------------------------------------
# _build_toc_map
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTocMapBuilder:
    def test_simple_toc(self):
        """Flat TOC produces correct page→chapter mapping."""
        doc = MagicMock()
        doc.get_toc.return_value = [
            [1, "Introduction", 1],
            [1, "Methods", 5],
            [1, "Results", 10],
        ]
        doc.page_count = 15

        result = _build_toc_map(doc)

        assert result[1] == "Introduction"
        assert result[5] == "Methods"
        assert result[10] == "Results"
        # Pages between entries inherit the previous chapter
        assert result[3] == "Introduction"
        assert result[7] == "Methods"
        assert result[12] == "Results"

    def test_hierarchical_toc(self):
        """Nested TOC produces breadcrumb paths with ' > '."""
        doc = MagicMock()
        doc.get_toc.return_value = [
            [1, "Book I", 1],
            [2, "Chapter 1", 2],
            [3, "Section 1.1", 3],
            [2, "Chapter 2", 5],
            [1, "Book II", 8],
        ]
        doc.page_count = 12

        result = _build_toc_map(doc)

        assert result[1] == "Book I"
        assert result[2] == "Book I > Chapter 1"
        assert result[3] == "Book I > Chapter 1 > Section 1.1"
        assert result[5] == "Book I > Chapter 2"
        assert result[8] == "Book II"

    def test_empty_toc(self):
        """Empty TOC returns empty dict."""
        doc = MagicMock()
        doc.get_toc.return_value = []
        doc.page_count = 5

        result = _build_toc_map(doc)
        assert result == {}

    def test_toc_with_invalid_page(self):
        """TOC entries with page_num < 1 are skipped."""
        doc = MagicMock()
        doc.get_toc.return_value = [
            [1, "Chapter 1", -1],  # invalid
            [1, "Chapter 2", 3],
        ]
        doc.page_count = 5

        result = _build_toc_map(doc)
        assert 1 not in result
        assert result[3] == "Chapter 2"


# ---------------------------------------------------------------------------
# _classify_span
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifySpan:
    def test_large_font_is_heading(self):
        span = {"size": 16.0, "flags": 0, "text": "Chapter 1"}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "heading"

    def test_medium_bold_caps_is_heading(self):
        span = {"size": 13.0, "flags": 16, "text": "METHODS"}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "heading"

    def test_medium_bold_not_caps_is_subheading(self):
        span = {"size": 13.0, "flags": 16, "text": "Data Collection"}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "subheading"

    def test_medium_not_bold_is_subheading(self):
        span = {"size": 13.0, "flags": 0, "text": "Some text"}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "subheading"

    def test_body_text(self):
        span = {"size": 11.0, "flags": 0, "text": "Regular paragraph."}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "content"

    def test_empty_text(self):
        span = {"size": 16.0, "flags": 0, "text": "   "}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "content"

    def test_long_caps_not_promoted(self):
        """ALL-CAPS text longer than 100 chars is not promoted to heading."""
        span = {"size": 13.0, "flags": 16, "text": "A" * 101}
        assert _classify_span(span, 11.0, 15.0, 13.0) == "subheading"


# ---------------------------------------------------------------------------
# _build_font_map
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFontMapBuilder:
    def _make_mock_doc(self, pages_spans):
        """Create a mock document with per-page span data."""
        doc = MagicMock()
        doc.page_count = len(pages_spans)

        pages = []
        for spans in pages_spans:
            page = MagicMock()
            blocks = [{"type": 0, "lines": [{"spans": spans}]}]
            page.get_text.return_value = {"blocks": blocks}
            pages.append(page)

        doc.load_page = lambda idx: pages[idx]
        doc.__iter__ = MagicMock(return_value=iter(pages))
        return doc

    def test_heading_detected(self, sample_font_spans):
        """Larger font is detected as heading."""
        doc = self._make_mock_doc([sample_font_spans])
        result = _build_font_map(doc)
        # Page 1 should have the heading detected
        assert 1 in result
        assert "Chapter 3" in result[1] or "Methods" in result[1]

    def test_no_headings_returns_empty(self):
        """Pages with only body text produce no entries."""
        body_spans = [
            {
                "size": 11.0,
                "flags": 0,
                "text": "Just regular text.",
                "bbox": (50, 50, 200, 65),
            },
        ]
        doc = self._make_mock_doc([body_spans])
        result = _build_font_map(doc)
        assert result == {}

    def test_empty_document(self):
        """Document with no text produces empty result."""
        doc = MagicMock()
        doc.page_count = 0
        doc.__iter__ = lambda self: iter([])
        result = _build_font_map(doc)
        assert result == {}


# ---------------------------------------------------------------------------
# _regex_fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegexFallback:
    def _make_mock_doc(self, page_texts):
        """Create a mock document with per-page text."""
        doc = MagicMock()
        doc.page_count = len(page_texts)

        pages = []
        for text in page_texts:
            page = MagicMock()
            page.get_text.return_value = text
            pages.append(page)

        doc.load_page = lambda idx: pages[idx]
        return doc

    def test_english_chapter(self):
        doc = self._make_mock_doc(["Chapter 3: Methods\nBody text here."])
        result = _regex_fallback(doc)
        assert result[1] == "Chapter 3"

    def test_polish_chapter(self):
        doc = self._make_mock_doc(["Rozdział V: Wyniki\nTekst."])
        result = _regex_fallback(doc)
        assert result[1] == "Rozdział V"

    def test_section_pattern(self):
        doc = self._make_mock_doc(["Section 2.1 Overview\nBody."])
        result = _regex_fallback(doc)
        assert result[1] == "Section 2.1"

    def test_numbered_heading(self):
        doc = self._make_mock_doc(["1. Introduction\nSome text."])
        result = _regex_fallback(doc)
        assert result[1] == "1. Introduction"

    def test_paragraph_symbol(self):
        doc = self._make_mock_doc(["§ 42 Important rule\nText."])
        result = _regex_fallback(doc)
        assert result[1] == "§ 42"

    def test_no_heading(self):
        doc = self._make_mock_doc(["Just some regular text without headings."])
        result = _regex_fallback(doc)
        assert result == {}

    def test_heading_not_on_first_line(self):
        """Heading found on second line still detected (scans first 500 chars)."""
        doc = self._make_mock_doc(["Some intro text.\nChapter 7: Analysis\nMore text."])
        result = _regex_fallback(doc)
        assert result[1] == "Chapter 7"

    def test_chapter_inheritance(self):
        """Pages without headings don't appear in result (no inheritance)."""
        doc = self._make_mock_doc(
            [
                "Chapter 1: Start\nBody.",
                "No heading here.",
                "Chapter 2: Next\nBody.",
            ]
        )
        result = _regex_fallback(doc)
        assert 1 in result
        assert 2 not in result  # no heading on page 2
        assert 3 in result


# ---------------------------------------------------------------------------
# ChapterDetector orchestration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChapterDetectorOrchestration:
    def _make_doc(self, toc=None, pages_spans=None, page_texts=None):
        """Helper to create a ChapterDetector with mocked fitz.open."""
        doc = MagicMock()

        # TOC
        if toc is not None:
            doc.get_toc.return_value = toc
        else:
            doc.get_toc.return_value = []

        # Pages for font analysis and regex fallback
        if pages_spans is None and page_texts is None:
            page_texts = ["No headings here."] * 3

        if page_texts:
            doc.page_count = len(page_texts)
            pages = []
            for text in page_texts:
                page = MagicMock()
                # Use default_value pattern to avoid closure issues
                page.get_text.side_effect = lambda fmt, _text=text, **kw: (
                    {"blocks": [{"type": 0, "lines": [{"spans": []}]}]}
                    if fmt == "dict"
                    else _text
                )
                pages.append(page)
            doc.load_page = lambda idx: pages[idx]
            doc.__iter__ = MagicMock(return_value=iter(pages))
        elif pages_spans:
            doc.page_count = len(pages_spans)
            pages = []
            for spans in pages_spans:
                page = MagicMock()
                page.get_text.side_effect = lambda fmt, _spans=spans, **kw: (
                    {"blocks": [{"type": 0, "lines": [{"spans": _spans}]}]}
                    if fmt == "dict"
                    else "No heading"
                )
                pages.append(page)
            doc.load_page = lambda idx: pages[idx]
            doc.__iter__ = MagicMock(return_value=iter(pages))

        return doc

    @patch("src.chapter_detection.fitz")
    def test_toc_strategy_preferred(self, mock_fitz):
        """TOC is used when available."""
        mock_fitz.open.return_value = self._make_doc(
            toc=[[1, "Chapter 1", 1], [1, "Chapter 2", 5]],
            page_texts=["text"] * 10,
        )
        with ChapterDetector("test.pdf") as detector:
            assert detector.detect_strategy() == "toc"
            assert detector.get_chapter_for_page(1) == "Chapter 1"
            assert detector.get_chapter_for_page(5) == "Chapter 2"

    @patch("src.chapter_detection.fitz")
    def test_font_strategy_fallback(self, mock_fitz):
        """Font analysis is used when TOC is empty."""
        heading_spans = [
            {"size": 16.0, "flags": 16, "text": "Methods", "bbox": (50, 50, 200, 70)},
        ]
        body_spans = [
            {"size": 11.0, "flags": 0, "text": "Body text.", "bbox": (50, 80, 400, 95)},
        ]
        mock_fitz.open.return_value = self._make_doc(
            toc=[],
            pages_spans=[heading_spans, body_spans, body_spans],
        )
        with ChapterDetector("test.pdf") as detector:
            assert detector.detect_strategy() == "font"
            assert detector.get_chapter_for_page(1) == "Methods"

    @patch("src.chapter_detection.fitz")
    def test_regex_strategy_last_resort(self, mock_fitz):
        """Regex is used when both TOC and font analysis fail."""
        mock_fitz.open.return_value = self._make_doc(
            toc=[],
            page_texts=[
                "Chapter 5: Analysis\nBody text.",
                "No heading here.",
                "Section 2.1 Overview\nMore text.",
            ],
        )
        with ChapterDetector("test.pdf") as detector:
            assert detector.detect_strategy() == "regex"
            assert detector.get_chapter_for_page(1) == "Chapter 5"
            assert detector.get_chapter_for_page(3) == "Section 2.1"

    @patch("src.chapter_detection.fitz")
    def test_none_strategy(self, mock_fitz):
        """Returns 'none' when no headings found by any strategy."""
        mock_fitz.open.return_value = self._make_doc(
            toc=[],
            page_texts=["No headings.", "Just text.", "Nothing special."],
        )
        with ChapterDetector("test.pdf") as detector:
            assert detector.detect_strategy() == "none"
            assert detector.get_chapter_for_page(1) is None

    @patch("src.chapter_detection.fitz")
    def test_context_manager(self, mock_fitz):
        """ChapterDetector works as context manager."""
        mock_doc = self._make_doc(toc=[[1, "Ch 1", 1]], page_texts=["text"])
        mock_fitz.open.return_value = mock_doc

        with ChapterDetector("test.pdf") as detector:
            detector.get_chapter_for_page(1)

        mock_doc.close.assert_called_once()

    @patch("src.chapter_detection.fitz")
    def test_hierarchical_toc_breadcrumb(self, mock_fitz):
        """Hierarchical TOC produces breadcrumb chapter paths."""
        mock_fitz.open.return_value = self._make_doc(
            toc=[
                [1, "Part I", 1],
                [2, "Chapter 1", 2],
                [3, "Section 1.1", 3],
            ],
            page_texts=["text"] * 5,
        )
        with ChapterDetector("test.pdf") as detector:
            assert (
                detector.get_chapter_for_page(3) == "Part I > Chapter 1 > Section 1.1"
            )
