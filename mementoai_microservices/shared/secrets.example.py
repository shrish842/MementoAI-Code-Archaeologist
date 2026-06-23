"""
Copy this file to `shared/secrets.py` for local development secrets.

Do not commit `shared/secrets.py`.
Do not rely on this pattern for production deployments or Docker image builds.
"""

PINECONE_API_KEY = "replace-me"
PINECONE_INDEX_NAME = "mementoai"
GOOGLE_API_KEY = "replace-me"

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = "8000"
INGESTION_HOST = "127.0.0.1"
INGESTION_PORT = "8001"
QUERY_HOST = "127.0.0.1"
QUERY_PORT = "8002"
SUMMARY_HOST = "127.0.0.1"
SUMMARY_PORT = "8003"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = "384"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"