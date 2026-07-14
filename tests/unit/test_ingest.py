import os
import tempfile
from unittest.mock import patch

import pytest

from src.extraction import (
    get_full_text_with_page_info,
    get_page_boundaries,
    page_at_position,
)
from src.ingest import ingest_folder


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


@pytest.mark.unit
class TestGetFullTextWithPageInfo:
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
        assert page_at_position(boundaries, page_nums, 0) == 1

    def test_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        last_pos = boundaries[-1] - 1
        assert page_at_position(boundaries, page_nums, last_pos) == 3

    def test_beyond_last_page(self, sample_pages):
        boundaries = get_page_boundaries(sample_pages)
        page_nums = [p["page_num"] for p in sample_pages]
        assert page_at_position(boundaries, page_nums, 999999) == 3


@pytest.mark.unit
class TestIngestFolder:
    def test_not_a_directory(self):
        with pytest.raises(NotADirectoryError):
            ingest_folder("/nonexistent/path")

    @patch("src.ingest.index_document")
    @patch("src.ingest.list_collections")
    @patch("src.ingest.get_qdrant_client")
    def test_indexes_supported_files(self, mock_client, mock_list, mock_index):
        mock_list.return_value = []
        mock_index.return_value = {"book": "test", "chunks": [{"text": "x"}], "total_pages": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create supported files
            for name in ["doc1.txt", "doc2.md", "doc3.py"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test content")

            results = ingest_folder(tmpdir)
            assert len(results) == 3
            assert all(r["status"] == "indexed" for r in results)
            assert mock_index.call_count == 3

    @patch("src.ingest.index_document")
    @patch("src.ingest.list_collections")
    @patch("src.ingest.get_qdrant_client")
    def test_skips_already_indexed(self, mock_client, mock_list, mock_index):
        mock_list.return_value = ["doc1"]
        mock_index.return_value = {"book": "doc2", "chunks": [{"text": "x"}], "total_pages": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["doc1.txt", "doc2.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test content")

            results = ingest_folder(tmpdir, reindex=False)
            assert sum(1 for r in results if r["status"] == "skipped") == 1
            assert sum(1 for r in results if r["status"] == "indexed") == 1
            assert mock_index.call_count == 1

    @patch("src.ingest.index_document")
    @patch("src.ingest.list_collections")
    @patch("src.ingest.get_qdrant_client")
    def test_empty_directory(self, mock_client, mock_list, mock_index):
        mock_list.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            results = ingest_folder(tmpdir)
            assert results == []
            assert mock_index.call_count == 0

    @patch("src.ingest.index_document")
    @patch("src.ingest.list_collections")
    @patch("src.ingest.get_qdrant_client")
    def test_handles_index_error(self, mock_client, mock_list, mock_index):
        mock_list.return_value = []
        mock_index.side_effect = ValueError("bad format")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bad.txt"), "w") as f:
                f.write("test")

            results = ingest_folder(tmpdir)
            assert len(results) == 1
            assert results[0]["status"] == "error"
            assert "bad format" in results[0]["error"]
