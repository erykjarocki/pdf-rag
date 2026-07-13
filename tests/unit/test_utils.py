import pytest

from src.utils import collection_name


@pytest.mark.unit
class TestCollectionName:
    def test_strips_pdf_extension(self):
        assert collection_name("investor-tom1.pdf") == "investor-tom1_pdf"

    def test_lowercases(self):
        assert collection_name("MYBOOK") == "mybook"

    def test_replaces_spaces_with_underscores(self):
        assert collection_name("My Book Title") == "my_book_title"

    def test_replaces_special_characters(self):
        assert collection_name("Book (2024)!") == "book__2024"

    def test_preserves_hyphens_and_underscores(self):
        assert collection_name("my-book_name") == "my-book_name"

    def test_strips_leading_trailing_underscores(self):
        assert collection_name("_book_") == "book"

    def test_empty_string_returns_book(self):
        assert collection_name("") == "book"

    def test_only_special_chars_returns_book(self):
        assert collection_name("!!!") == "book"

    def test_polish_characters_replaced(self):
        result = collection_name("żółć.xlsx")
        assert result == "xlsx"

    def test_realistic_book_name(self):
        result = collection_name("Biblia-Tysiąclecia-Pallotinum.pdf")
        assert result == "biblia-tysi_clecia-pallotinum_pdf"

    def test_simple_name(self):
        assert collection_name("test") == "test"
