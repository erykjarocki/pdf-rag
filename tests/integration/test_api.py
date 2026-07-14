from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api import app
    return TestClient(app)


@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.integration
class TestQueryEndpoint:
    @patch("src.retriever.embed_query")
    @patch("src.retriever.get_qdrant_client")
    def test_query_returns_results(self, mock_get_client, mock_embed_query, client):
        mock_embed_query.return_value = [0.1] * 384

        mock_qdrant = mock_get_client.return_value
        mock_qdrant.get_collections.return_value.collections = []
        mock_qdrant.query_points.return_value.points = []

        response = client.post("/query", json={"question": "test question"})
        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert "formatted" in data

    def test_query_empty_body(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_missing_question(self, client):
        response = client.post("/query", json={"book": "test"})
        assert response.status_code == 422


@pytest.mark.integration
class TestBooksEndpoint:
    @patch("src.api.list_collections")
    def test_list_books(self, mock_list, client):
        mock_list.return_value = ["book1", "book2"]
        response = client.get("/books")
        assert response.status_code == 200
        assert "books" in response.json()
        assert len(response.json()["books"]) == 2

    @patch("src.api.list_collections")
    @patch("src.api.get_qdrant_client")
    @patch("src.api.delete_collection")
    def test_delete_book(self, mock_delete, mock_get_client, mock_list, client):
        mock_list.return_value = ["existing-book"]
        mock_get_client.return_value = mock_get_client

        response = client.delete("/books/existing-book")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    @patch("src.api.list_collections")
    @patch("src.api.get_qdrant_client")
    def test_delete_nonexistent_book(self, mock_get_client, mock_list, client):
        mock_list.return_value = ["other-book"]

        response = client.delete("/books/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
class TestIngestFolderEndpoint:
    @patch("src.api.ingest_folder")
    @patch("src.api.os.path.isdir", return_value=True)
    def test_ingest_folder_returns_results(self, mock_isdir, mock_ingest, client):
        mock_ingest.return_value = [
            {"name": "doc1", "status": "indexed", "chunks": 10},
            {"name": "doc2", "status": "skipped"},
        ]
        response = client.post(
            "/ingest-folder",
            json={"directory": "/tmp/test-docs"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_indexed"] == 1
        assert data["total_skipped"] == 1
        assert len(data["results"]) == 2

    def test_ingest_folder_invalid_directory(self, client):
        response = client.post(
            "/ingest-folder",
            json={"directory": "/nonexistent/path"},
        )
        assert response.status_code == 400

    @patch("src.api.ingest_folder")
    @patch("src.api.os.path.isdir", return_value=True)
    def test_ingest_folder_with_reindex(self, mock_isdir, mock_ingest, client):
        mock_ingest.return_value = [
            {"name": "doc1", "status": "indexed", "chunks": 5},
        ]
        response = client.post(
            "/ingest-folder",
            json={"directory": "/tmp/test-docs", "reindex": True},
        )
        assert response.status_code == 200
        mock_ingest.assert_called_once_with("/tmp/test-docs", reindex=True)
