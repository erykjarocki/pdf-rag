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


@pytest.fixture
def sample_font_spans():
    """Sample font spans simulating a PDF page with heading hierarchy.

    Simulates a page with:
    - A large bold heading ("Chapter 3: Methods") at 16pt
    - A medium subheading ("3.1 Data Collection") at 13pt
    - Body text at 11pt
    """
    return [
        # Large heading
        {
            "size": 16.0,
            "flags": 16,  # bold
            "font": "Helvetica-Bold",
            "text": "Chapter 3: Methods",
            "bbox": (50.0, 80.0, 250.0, 100.0),
        },
        # Subheading
        {
            "size": 13.0,
            "flags": 16,  # bold
            "font": "Helvetica-Bold",
            "text": "3.1 Data Collection",
            "bbox": (50.0, 120.0, 200.0, 135.0),
        },
        # Body text lines
        {
            "size": 11.0,
            "flags": 0,
            "font": "Helvetica",
            "text": "The data was collected over a period of six months",
            "bbox": (50.0, 150.0, 500.0, 165.0),
        },
        {
            "size": 11.0,
            "flags": 0,
            "font": "Helvetica",
            "text": "from multiple sources including surveys and interviews.",
            "bbox": (50.0, 170.0, 500.0, 185.0),
        },
    ]
