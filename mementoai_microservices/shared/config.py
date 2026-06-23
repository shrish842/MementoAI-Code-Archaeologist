from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

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

def _load_local_secrets_module() -> Any | None:
    try:
        return importlib.import_module("shared.secrets")
    except ModuleNotFoundError:
        return None


def _read_setting(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value

    local_secrets = _load_local_secrets_module()
    if local_secrets is not None and hasattr(local_secrets, name):
        secret_value = getattr(local_secrets, name)
        if secret_value not in (None, ""):
            return str(secret_value)

    return default

def load_config(service_name: str, default_host: str, default_port: int) -> ServiceConfig:
    prefix = service_name.upper()
    host = _read_setting(f"{prefix}_HOST", default_host) or default_host
    port = int(_read_setting(f"{prefix}_PORT", str(default_port)) or str(default_port))
    return ServiceConfig(
        service_name=service_name,
        host=host,
        port=port,
        pinecone_api_key=_read_setting("PINECONE_API_KEY"),
        pinecone_index_name=_read_setting("PINECONE_INDEX_NAME", "mementoai") or "mementoai",
        embedding_model_name=_read_setting("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2",
        embedding_dimension=int(_read_setting("EMBEDDING_DIMENSION", "384") or "384"),
        gemini_model_name=_read_setting("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") or "gemini-1.5-flash-latest",
        google_api_key=_read_setting("GOOGLE_API_KEY"),
    )