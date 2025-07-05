from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
from core.exceptions import PineconeConnectionError
import time

pinecone_index = None

def initialize_pinecone():
    """
    Initializes the Pinecone connection and ensures the index exists.
    """
    global pinecone_index
    if not settings.PINECONE_API_KEY:
        print("WARNING: Pinecone API Key not set. Pinecone functionality disabled.")
        return

    try:
        print("Initializing Pinecone connection...")
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        existing_indexes = pc.list_indexes()
        index_names = [index_info.name for index_info in existing_indexes]

        if settings.PINECONE_INDEX_NAME not in index_names:
            print(f"Creating Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{settings.PINECONE_INDEX_NAME}' created. Waiting for initialization...")

            while True:
                status = pc.describe_index(settings.PINECONE_INDEX_NAME).status
                if status.get('ready') and status.get('state') == 'Ready':
                    print(f"Index '{settings.PINECONE_INDEX_NAME}' is ready.")
                    break
                print(f"Index '{settings.PINECONE_INDEX_NAME}' not ready yet. Waiting...")
                time.sleep(10)

        pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}. Stats: {pinecone_index.describe_index_stats()}")
    except Exception as e:
        print(f"ERROR: Could not connect/create Pinecone index: {e}")
        raise PineconeConnectionError(f"Failed to connect to Pinecone: {e}")

# Initialize Pinecone when the module is imported
try:
    initialize_pinecone()
except PineconeConnectionError:
    pinecone_index = None # Ensure it's None if connection fails

