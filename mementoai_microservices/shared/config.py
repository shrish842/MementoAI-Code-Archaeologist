from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceConfig:
    service_name: str
    host: str
    port: int
    pinecone_api_key: str | None
    pinecone_index_name: str
    embedding_model_name: str
    embedding_dimension: int
    gemini_model_name: str
    google_api_key: str | None


def load_config(service_name: str, default_host: str, default_port: int) -> ServiceConfig:
    prefix = service_name.upper()
    host = os.getenv(f"{prefix}_HOST", default_host)
    port = int(os.getenv(f"{prefix}_PORT", str(default_port)))
    return ServiceConfig(
        service_name=service_name,
        host=host,
        port=port,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "mementoai"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )