import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration for the MementoAI application.
    Loads environment variables with sensible defaults.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    PINECONE_API_KEY: str = os.environ.get('PINECONE_API_KEY', "YOUR_DEFAULT_PINECONE_KEY")
    PINECONE_INDEX_NAME: str = "mementoai"
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    EMBEDDING_DIMENSION: int = 384
    NUM_COMMITS_TO_EXTRACT_FOR_INDEXING: int = 5000
    MAX_DIFF_CHARS_FOR_STORAGE: int = 5000
    MAX_DIFF_CHARS_FOR_LLM: int = 2000
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"
    CELERY_BROKER_URL: str = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND: str = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    GOOGLE_API_KEY: str = os.environ.get('GOOGLE_API_KEY', "YOUR_DEFAULT_GOOGLE_API_KEY")
    FRONTEND_URL: str = "http://localhost:8501" # Default for Streamlit frontend

settings = Settings()

