from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    return model.encode([text], normalize_embeddings=True).tolist()[0]
