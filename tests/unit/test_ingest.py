import pytest

from src.ingest import (
    _page_at_position,
    detect_chapter,
    get_full_text_with_page_info,
    get_page_boundaries,
)


@pytest.mark.unit
class TestDetectChapter:
    def test_polish_roman_numeral(self):
        assert detect_chapter("Rozdział III: Poczatki") == "Rozdział III"

    def test_polish_arabic_numeral(self):
        assert detect_chapter("Rozdział 5: Wyniki") == "Rozdział 5"

    def test_english_chapter(self):
        assert detect_chapter("Chapter 12: Discussion") == "Chapter 12"

    def test_polish_part(self):
        assert detect_chapter("CZĘŚĆ II: Dialogi") == "CZĘŚĆ II"

    def test_no_chapter(self):
        assert detect_chapter("Zwykly tekst bez naglowka.") is None

    def test_chapter_at_start(self):
        assert detect_chapter("Rozdział I\nTekst po naglowku.") == "Rozdział I"

    def test_chapter_in_middle(self):
        text = "Troche tekstu.\nRozdział VII: Nowy\nWiecej tekstu."
        assert detect_chapter(text) == "Rozdział VII"

    def test_english_chapter_in_text(self):
        text = "Some text.\nChapter 3: Methods\nMore text."
        assert detect_chapter(text) == "Chapter 3"

    def test_chapter_with_digit(self):
        assert detect_chapter("Chapter 3") == "Chapter 3"

    def test_empty_string(self):
        assert detect_chapter("") is None

    def test_roman_numeral_variants(self):
        assert detect_chapter("Rozdział IV") == "Rozdział IV"
        assert detect_chapter("Rozdział XLV") == "Rozdział XLV"


@pytest.mark.unit
class TestPageBoundaries:
    def test_single_page(self, sample_pages):
        boundaries = get_page_boundaries([sample_pages[0]])
        assert len(boundaries) == 1
        assert boundaries[0] == len(sample_pages[0]["text"]) + 1

    def test_multiple_pages(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        assert len(boundaries) == 3
        assert boundaries[0] < boundaries[1] < boundaries[2]

    def test_cumulative(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        for i in range(1, len(boundaries)):
            assert boundaries[i] > boundaries[i - 1]


@pytest.mark.unit
class TestGetFullTextWithPageInfo:
    def test_returns_tuple(self, sample_pages):
        result = get_full_text_with_page_info(sample_pages)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_full_text_contains_all_pages(self, sample_pages):
        full_text, _ = get_full_text_with_page_info(sample_pages)
        for page in sample_pages:
            assert page["text"] in full_text

    def test_boundaries_length_matches_pages(self, sample_pages):
        _, boundaries = get_full_text_with_page_info(sample_pages)
        assert len(boundaries) == len(sample_pages)


@pytest.mark.unit
class TestPageAtPosition:
    def test_first_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        assert _page_at_position(boundaries, page_nums, 0) == 1

    def test_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        last_pos = boundaries[-1] - 1
        assert _page_at_position(boundaries, page_nums, last_pos) == 3

    def test_beyond_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        assert _page_at_position(boundaries, page_nums, 999999) == 3
