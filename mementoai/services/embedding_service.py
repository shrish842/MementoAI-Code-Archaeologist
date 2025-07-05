from sentence_transformers import SentenceTransformer
from config.settings import settings
from core.exceptions import ModelLoadingError

embedding_model = None

def load_embedding_model():
    """
    Loads the SentenceTransformer embedding model.
    """
    global embedding_model
    try:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        print(f"Embedding model loaded. Device: {embedding_model.device}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load embedding model: {e}")
        raise ModelLoadingError(f"Failed to load embedding model: {e}")

# Load embedding model when the module is imported
try:
    load_embedding_model()
except ModelLoadingError:
    embedding_model = None # Ensure it's None if loading fails

