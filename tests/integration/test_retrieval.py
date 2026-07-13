import pytest
from qdrant_client.models import PointStruct

from src.config import EMBED_DIM
from src.qdrant_store import delete_collection, ensure_collection, list_collections


@pytest.mark.integration
class TestQdrantCollections:
    def test_ensure_collection_creates(self, qdrant_memory):
        ensure_collection("test-coll", qdrant_memory)
        assert "test-coll" in list_collections(qdrant_memory)

    def test_ensure_collection_idempotent(self, qdrant_memory):
        ensure_collection("test-coll", qdrant_memory)
        ensure_collection("test-coll", qdrant_memory)
        assert list_collections(qdrant_memory).count("test-coll") == 1

    def test_delete_collection(self, qdrant_memory):
        ensure_collection("to-delete", qdrant_memory)
        delete_collection("to-delete", qdrant_memory)
        assert "to-delete" not in list_collections(qdrant_memory)


@pytest.mark.integration
class TestQdrantUpsertAndQuery:
    def test_upsert_and_retrieve(self, qdrant_memory):
        ensure_collection("upsert-test", qdrant_memory)

        vector = [0.1] * EMBED_DIM
        qdrant_memory.upsert(
            collection_name="upsert-test",
            points=[
                PointStruct(
                    id=1,
                    vector=vector,
                    payload={
                        "text": "test chunk",
                        "book": "test",
                        "chapter": "Ch1",
                        "start_page": 1,
                        "end_page": 1,
                    },
                )
            ],
        )

        results = qdrant_memory.query_points(
            collection_name="upsert-test",
            query=vector,
            limit=5,
        )
        assert len(results.points) == 1
        assert results.points[0].payload["text"] == "test chunk"

    def test_query_empty_collection(self, qdrant_memory):
        ensure_collection("empty-test", qdrant_memory)
        results = qdrant_memory.query_points(
            collection_name="empty-test",
            query=[0.1] * EMBED_DIM,
            limit=5,
        )
        assert len(results.points) == 0

    def test_multiple_vectors_ranked_by_score(self, qdrant_memory):
        ensure_collection("rank-test", qdrant_memory)

        import math

        identical = [1.0 / math.sqrt(EMBED_DIM)] * EMBED_DIM
        orthogonal = [0.0] * EMBED_DIM
        orthogonal[0] = 1.0

        qdrant_memory.upsert(
            collection_name="rank-test",
            points=[
                PointStruct(
                    id=1,
                    vector=identical,
                    payload={
                        "text": "exact match",
                        "book": "b",
                        "chapter": "",
                        "start_page": 1,
                        "end_page": 1,
                    },
                ),
                PointStruct(
                    id=2,
                    vector=orthogonal,
                    payload={
                        "text": "different topic",
                        "book": "b",
                        "chapter": "",
                        "start_page": 2,
                        "end_page": 2,
                    },
                ),
            ],
        )

        results = qdrant_memory.query_points(
            collection_name="rank-test",
            query=identical,
            limit=2,
        )
        assert results.points[0].payload["text"] == "exact match"
        assert results.points[0].score > results.points[1].score
