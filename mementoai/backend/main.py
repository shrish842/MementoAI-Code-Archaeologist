from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import google.generativeai as genai
import json
from celery.result import AsyncResult

# Import modular components
from backend.config import Config
from backend.services import (
    git_services,
    analysis_services,
    embedding_services,
    pinecone_services
)
from backend.celery_app import celery_app
from backend.utils.helpers import validate_repo_url, generate_repo_namespace
from backend.models.schemas import (  # Import models from schemas
    IndexRepoRequest,
    IndexRepoResponse,
    FunctionChange,
    CommitInfo,
    QueryRepoRequest,
    QueryRepoResponse,
    JobStatusResponse,
    TechnicalDebtMetrics
)

print("MementoAI API starting...")

# Initialize services
pinecone_index = pinecone_services.initialize_pinecone()
embedding_model = embedding_services.get_embedding_model()

# Check Gemini configuration
genai_configured = False
if Config.GOOGLE_API_KEY:
    try:
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        genai_configured = True
        print("INFO: Google API Key configured.")
    except Exception as e:
        print(f"ERROR during Gemini setup: {e}")
else:
    print("WARNING: Google API Key not found/configured.")

# --- FastAPI App ---
app = FastAPI(title="MementoAI Backend API")

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
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
    print("Initializing services on startup...")
    pinecone_services.initialize_pinecone()
    embedding_services.get_embedding_model()

@app.post("/index_repository", response_model=IndexRepoResponse)
async def index_repository_endpoint(request: IndexRepoRequest):
    if not validate_repo_url(request.repo_url):
        raise HTTPException(status_code=400, detail="Invalid Git repository URL. Must be a valid URL ending with .git")
    
    if not pinecone_services.get_pinecone_index():
        raise HTTPException(status_code=503, detail="Vector database (Pinecone) not configured or unavailable.")
    
    repo_id_for_namespace = generate_repo_namespace(request.repo_url)
    task = celery_app.send_task(
        "tasks.process_and_index_repository",
        args=[request.repo_url, repo_id_for_namespace]
    )
    
    return IndexRepoResponse(
        job_id=task.id,
        status="queued",
        message=f"Repo indexing job queued. Repo ID/Namespace: {repo_id_for_namespace}"
    )

@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)
    response = JobStatusResponse(job_id=job_id, status=task_result.status)
    
    if task_result.ready():
        if task_result.successful():
            response.result = task_result.result
        else:
            response.result = {"error": str(task_result.result)}
    
    return response

@app.post("/query_repository", response_model=QueryRepoResponse)
async def query_repository_endpoint(request: QueryRepoRequest): 
    print(f"Query for repo_id: {request.repo_id}, Question: {request.question}")
    
    embedding_model = embedding_services.get_embedding_model()
    pinecone_index = pinecone_services.get_pinecone_index()
    
    if not embedding_model or not pinecone_index:
        raise HTTPException(status_code=503, detail="Core models/DB not loaded.")
    if not request.question: 
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    analysis_error = None
    retrieved_commits_info = [] 

    try:
        question_embedding = embedding_model.encode([request.question]).tolist()
        query_response = pinecone_index.query(
            namespace=request.repo_id, 
            vector=question_embedding,
            top_k=5,
            include_metadata=True 
        )

        if query_response.matches:
            for match in query_response.matches:
                meta = match.metadata or {}
                commit_date = "Unknown"
                
                if meta.get("timestamp"):
                    try: 
                        commit_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(meta.get("timestamp")))
                    except: 
                        pass

                diff_snippet = meta.get("diff_snippet", "(Diff not stored or available in metadata)")
                
                # Process function changes from AST analysis
                function_changes = []
                
                # Added functions
                for func_name in meta.get("functions_added", []):
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="added"
                    ))
                
                # Removed functions
                for func_name in meta.get("functions_removed", []):
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="removed"
                    ))
                
                # Modified functions
                complexity_changes = json.loads(meta.get("complexity_changes", "{}"))
                for func_name in meta.get("functions_modified", []):
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="modified",
                        complexity_change=complexity_changes.get(func_name, 0)
                    ))

                # Handle technical debt data
                technical_debt = None
                old_technical_debt = None
                debt_delta = meta.get("debt_delta", 0.0)
                
                if meta.get("technical_debt"):
                    try:
                        technical_debt = TechnicalDebtMetrics(**json.loads(meta["technical_debt"]))
                    except:
                        pass
                
                if meta.get("old_technical_debt"):
                    try:
                        old_technical_debt = TechnicalDebtMetrics(**json.loads(meta["old_technical_debt"]))
                    except:
                        pass

                retrieved_commits_info.append(CommitInfo(
                    hash=match.id,
                    message=meta.get("message", "Message not found in metadata."),
                    author=meta.get("author", "N/A"),
                    date=commit_date,
                    similarity=match.score,
                    diff=diff_snippet,
                    function_changes=function_changes,
                    technical_debt=technical_debt,
                    old_technical_debt=old_technical_debt,
                    debt_delta=debt_delta
                ))

            print(f"Retrieved {len(retrieved_commits_info)} commits from Pinecone.")

    except Exception as e:
        analysis_error = f"Error querying Pinecone for repo {request.repo_id}: {e}"
        print(f"ERROR: {analysis_error}")
        return QueryRepoResponse(status="error", message=analysis_error)

    # Generate AI summary if configured
    ai_summary_text = None
    gemini_call_attempted = False
    
    if genai_configured and retrieved_commits_info and not analysis_error:
        print(f"Attempting Gemini summarization for repo_id: {request.repo_id}...")
        gemini_call_attempted = True
        
        context_parts = []
        for i, commit_data in enumerate(retrieved_commits_info):
            diff_for_prompt = "(Diff not available)" if commit_data.diff.startswith("Error") else commit_data.diff
            
            # Include function changes in the prompt
            func_changes = []
            for change in commit_data.function_changes:
                if change.change_type == "added":
                    func_changes.append(f"Added function: {change.name}")
                elif change.change_type == "removed":
                    func_changes.append(f"Removed function: {change.name}")
                else:
                    complexity = f" (complexity {'+' if change.complexity_change > 0 else ''}{change.complexity_change})" if change.complexity_change else ""
                    func_changes.append(f"Modified function: {change.name}{complexity}")
            
            func_changes_text = "\n".join(func_changes) if func_changes else "No function changes detected"
            
            context_parts.append(
                f"Commit {i+1} ({commit_data.hash[:7]}):\n"
                f"Message: {commit_data.message}\n"
                f"Function Changes:\n{func_changes_text}\n"
                f"Diff Snippet:\n```diff\n{diff_for_prompt}\n```\n---\n"
            )
        
        full_context = "\n".join(context_parts)
        repo_name_guess = request.repo_id 
        prompt = (
            f"Analyze commit info from repo '{repo_name_guess}'...\n"
            f"Context:\n---\n{full_context}---\n"
            f"User Question: {request.question}\n"
            f"Focus on explaining the functional changes and their significance.\n"
            f"Answer:"
        )
        
        try:
            model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
            response = model.generate_content(prompt, request_options={"timeout": 60})
            ai_summary_text = response.text if response.parts else "Gemini empty response."
        except Exception as e:
            print(f"ERROR calling Gemini: {e}")
            ai_summary_text = f"Error during AI summarization: {e}"
    
    if not gemini_call_attempted:
        if analysis_error: 
            ai_summary_text = "AI Summarization skipped due to earlier analysis error."
        elif not retrieved_commits_info: 
            ai_summary_text = "AI Summarization skipped (no relevant commits found)."
        else: 
            ai_summary_text = "AI Summarization skipped (API Key not configured)."
    elif ai_summary_text is None:
        ai_summary_text = "AI Summarization failed for an unknown reason after attempt."

    final_status = "success" if not analysis_error else "partial_success"
    final_message = analysis_error if analysis_error else "Query complete."
    
    if ai_summary_text and ("Error:" in ai_summary_text or "Blocked:" in ai_summary_text) and final_status == "success":
        final_status = "partial_success"

    return QueryRepoResponse(
        status=final_status,
        message=final_message,
        relevant_commits=retrieved_commits_info,
        ai_summary=ai_summary_text
    )

@app.get("/")
async def root(): 
    return {"message": "MementoAI Backend API with Pinecone & Celery is running!"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly for MementoAI API...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)