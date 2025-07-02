import time
import os
from pinecone import Pinecone, ServerlessSpec
from backend.config import Config

pinecone_index = None

def initialize_pinecone():
    global pinecone_index
    if pinecone_index is None and Config.PINECONE_API_KEY:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        if Config.PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=Config.PINECONE_INDEX_NAME,
                dimension=Config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while True:
                status = pc.describe_index(Config.PINECONE_INDEX_NAME).status
                if status.get('ready') and status.get('state') == 'Ready': 
                    break
                time.sleep(10)
        
        pinecone_index = pc.Index(Config.PINECONE_INDEX_NAME)
    return pinecone_index

def get_pinecone_index():
    if pinecone_index is None:
        return initialize_pinecone()
    return pinecone_index