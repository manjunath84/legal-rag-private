# src/raglab/embed.py
"""Local, in-process embeddings via sentence-transformers.

mlx_vlm.server has no embeddings endpoint, so embeddings run here on the same
machine — keeping the "nothing leaves the laptop" guarantee. The model is loaded
once and cached (it's a one-time online download, then fully offline).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from raglab.config import settings


@lru_cache(maxsize=2)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    model = _model(model_name or settings.embed_model)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str, model_name: str | None = None) -> list[float]:
    return embed_texts([text], model_name)[0]
