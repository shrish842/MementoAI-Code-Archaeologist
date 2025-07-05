import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from config.settings import settings
from api.endpoints import router
from services.embedding_service import embedding_model
from services.pinecone_service import pinecone_index

app = FastAPI(
    title="MementoAI Backend API",
    description="Codebase Archaeology System",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:8501",
    settings.FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize core services and log status"""
    startup_messages = []
    
    if embedding_model:
        startup_messages.append(f"✅ Embedding model loaded on {embedding_model.device}")
    else:
        startup_messages.append("❌ Embedding model NOT loaded")
    
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            startup_messages.append(f"✅ Pinecone connected: {stats}")
        except Exception:
            startup_messages.append("❌ Pinecone connection failed")
    else:
        startup_messages.append("❌ Pinecone NOT connected")
    
    print("\n".join(["MementoAI Backend API starting up..."] + startup_messages))

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MementoAI Backend API is running",
        "status": "operational",
        "services": {
            "embedding_model": "loaded" if embedding_model else "not loaded",
            "pinecone": "connected" if pinecone_index else "not connected"
        }
    }

if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )