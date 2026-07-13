import os
import sys

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def qdrant_memory():
    """In-memory Qdrant client for testing."""
    return QdrantClient(":memory:")


@pytest.fixture
def sample_fragments():
    """Sample search result fragments for formatting tests."""
    return [
        {
            "text": "To jest przykladowy tekst z ksiazki.",
            "book": "test-book",
            "chapter": "Rozdzial I",
            "start_page": 10,
            "end_page": 12,
            "score": 0.95,
        },
        {
            "text": "Drugi fragment odpowiedzi.",
            "book": "test-book",
            "chapter": "",
            "start_page": 15,
            "end_page": 15,
            "score": 0.87,
        },
    ]


@pytest.fixture
def sample_pages():
    """Sample PDF page data for ingestion tests."""
    return [
        {"page_num": 1, "text": "Pierwsza strona tekstu. " * 20},
        {"page_num": 2, "text": "Druga strona z trescia. " * 20},
        {"page_num": 3, "text": "Rozdzial III: Nowy rozdzial. " * 20},
    ]
