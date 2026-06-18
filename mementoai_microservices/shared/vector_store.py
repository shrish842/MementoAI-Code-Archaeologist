from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Protocol

from .config import ServiceConfig


class VectorStore(Protocol):
    def upsert(self, namespace: str, records: List[Dict[str, Any]]) -> None:
        ...

    def query(self, namespace: str, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def upsert(self, namespace: str, records: List[Dict[str, Any]]) -> None:
        existing = {record["id"]: record for record in self._data[namespace]}
        for record in records:
            existing[record["id"]] = record
        self._data[namespace] = list(existing.values())

    def query(self, namespace: str, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for record in self._data.get(namespace, []):
            score = _cosine_similarity(vector, record["vector"])
            hits.append({"id": record["id"], "score": score, "metadata": record["metadata"]})
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:top_k]


class PineconeVectorStore:
    def __init__(self, config: ServiceConfig) -> None:
        from pinecone import Pinecone, ServerlessSpec

        if not config.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for PineconeVectorStore")

        client = Pinecone(api_key=config.pinecone_api_key)
        existing = {index.name for index in client.list_indexes()}
        if config.pinecone_index_name not in existing:
            client.create_index(
                name=config.pinecone_index_name,
                dimension=config.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = client.Index(config.pinecone_index_name)

    def upsert(self, namespace: str, records: List[Dict[str, Any]]) -> None:
        vectors = [
            (record["id"], record["vector"], record["metadata"])
            for record in records
        ]
        self._index.upsert(vectors=vectors, namespace=namespace)

    def query(self, namespace: str, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        result = self._index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        return [
            {"id": match.id, "score": match.score, "metadata": match.metadata or {}}
            for match in result.matches
        ]


def build_vector_store(config: ServiceConfig) -> VectorStore:
    if config.pinecone_api_key:
        try:
            return PineconeVectorStore(config)
        except Exception:
            return InMemoryVectorStore()
    return InMemoryVectorStore()


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)