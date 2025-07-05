import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from config.settings import settings
from api.endpoints import router # Import the router
from services.embedding_service import embedding_model
from services.pinecone_service import pinecone_index

app = FastAPI(title="MementoAI Backend API")

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:8501",
    settings.FRONTEND_URL, # Use setting for frontend URL
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
    """
    Ensures core services are loaded at startup.
    """
    if embedding_model is None:
        print("FATAL: Embedding model not loaded at startup.")
    if pinecone_index is None:
        print("FATAL: Pinecone index not connected at startup.")
    print("MementoAI Backend API starting up...")

# --- ADD THIS LINE ---
app.include_router(router) # Include the router here!
# ---------------------

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "MementoAI Backend API with Pinecone & Celery is running!"}

if __name__ == "__main__":
    print("Starting Uvicorn server directly for MementoAI API...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)