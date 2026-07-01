from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBED_DIM

_client = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(client: QdrantClient | None = None):
    if client is None:
        client = get_qdrant_client()

    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="book",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"Created collection '{QDRANT_COLLECTION}'")
    else:
        print(f"Collection '{QDRANT_COLLECTION}' already exists")
