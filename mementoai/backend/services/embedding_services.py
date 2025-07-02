from sentence_transformers import SentenceTransformer
from .pinecone_services import get_pinecone_index
from backend.config import Config

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return embedding_model

def generate_embeddings(texts: list):
    model = get_embedding_model()
    return model.encode(texts, batch_size=64, show_progress_bar=False)