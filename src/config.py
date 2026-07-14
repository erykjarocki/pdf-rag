import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks", "chunks.json")
METADATA_FILE = os.path.join(BASE_DIR, "data", "metadata", "metadata.json")
QDRANT_PATH = os.path.join(BASE_DIR, "vector_db", "qdrant")

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "8"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
