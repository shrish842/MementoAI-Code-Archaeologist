from __future__ import annotations

import hashlib
import os
from typing import List, Protocol

from .config import ServiceConfig


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        buckets = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return buckets
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            slot = int(digest[:8], 16) % self.dimensions
            buckets[slot] += 1.0
        scale = max(sum(value * value for value in buckets) ** 0.5, 1.0)
        return [value / scale for value in buckets]


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()


def build_embedding_provider(config: ServiceConfig) -> EmbeddingProvider:
    if os.getenv("MEMENTO_EMBEDDINGS_MODE", "").lower() == "hash":
        return HashingEmbeddingProvider(config.embedding_dimension)
    try:
        return SentenceTransformerEmbeddingProvider(config.embedding_model_name)
    except Exception:
        return HashingEmbeddingProvider(config.embedding_dimension)
