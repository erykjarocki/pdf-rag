import pytest

from src.retriever import _format_results, format_fragments_for_prompt


@pytest.mark.unit
class TestFormatFragmentsForPrompt:
    def test_single_fragment(self):
        fragments = [
            {
                "text": "Hello world",
                "book": "test-book",
                "chapter": "Ch1",
                "start_page": 1,
                "end_page": 1,
                "score": 0.9,
            }
        ]
        result = format_fragments_for_prompt(fragments)
        assert "[1]" in result
        assert "Hello world" in result
        assert "Źródło: test-book, Ch1, str. 1" in result

    def test_multiple_fragments(self, sample_fragments):
        result = format_fragments_for_prompt(sample_fragments)
        assert "[1]" in result
        assert "[2]" in result

    def test_page_range(self):
        fragments = [
            {
                "text": "Text",
                "book": "book",
                "chapter": "",
                "start_page": 5,
                "end_page": 8,
                "score": 0.9,
            }
        ]
        result = format_fragments_for_prompt(fragments)
        assert "str. 5-8" in result

    def test_single_page_no_range(self):
        fragments = [
            {
                "text": "Text",
                "book": "book",
                "chapter": "",
                "start_page": 5,
                "end_page": 5,
                "score": 0.9,
            }
        ]
        result = format_fragments_for_prompt(fragments)
        assert "str. 5" in result
        # Verify no page range dash (only the citation line, not the separator)
        citation_line = [line for line in result.split("\n") if "Źródło" in line][0]
        assert "-" not in citation_line

    def test_no_chapter_omits_chapter_from_source(self):
        fragments = [
            {
                "text": "Text",
                "book": "book",
                "chapter": "",
                "start_page": 1,
                "end_page": 1,
                "score": 0.9,
            }
        ]
        result = format_fragments_for_prompt(fragments)
        # Source should be "Źródło: book, str. 1" without a chapter
        assert "Źródło: book, str. 1" in result

    def test_empty_fragments(self):
        result = format_fragments_for_prompt([])
        assert result == ""


@pytest.mark.unit
class TestFormatResults:
    def test_formats_point(self):
        class MockPayload:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def get(self, key, default=None):
                return self._data.get(key, default)

        class MockPoint:
            def __init__(self, payload, score):
                self.payload = MockPayload(payload)
                self.score = score

        points = [
            MockPoint(
                {
                    "text": "chunk text",
                    "book": "my-book",
                    "chapter": "Ch1",
                    "start_page": 1,
                    "end_page": 2,
                },
                0.95,
            )
        ]
        result = _format_results(points)
        assert len(result) == 1
        assert result[0]["text"] == "chunk text"
        assert result[0]["score"] == 0.95
