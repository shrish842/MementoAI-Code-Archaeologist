import os
import time
import subprocess
import tempfile
import torch 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import validators
from pinecone import Pinecone, ServerlessSpec
from celery import Celery
import uuid
 
print("API Script loading...")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', "")# PASTE YOUR ORIGINAL PINECONE API KEY HERE
PINECONE_INDEX_NAME = "mementoai"   

# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"



genai_configured = False
google_api_key_to_use = None
try:
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key: google_api_key_to_use = env_key
    else:
        manual_key = "" # <-- PASTE YOUR VALID GEMINI KEY
        if manual_key and manual_key != "YOUR_GOOGLE_API_KEY_GOES_HERE":
            google_api_key_to_use = manual_key
    if google_api_key_to_use:
        genai.configure(api_key=google_api_key_to_use)
        genai_configured = True
        print("INFO: Google API Key configured.")
    else:
        print("WARNING: Google API Key not found/configured.")
except Exception as e:
    print(f"ERROR during API Key setup: {e}")
    genai_configured = False
    
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384
    
NUM_COMMITS_TO_EXTRACT_FOR_INDEXING = 5000 
MAX_DIFF_CHARS_FOR_STORAGE = 5000 
MAX_DIFF_CHARS_FOR_LLM = 2000  
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
celery_app = Celery('memento_tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(task_serializer='json', accept_content=['json'], result_serializer='json', timezone='UTC', enable_utc=True)

embedding_model = None
try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded. Device: {embedding_model.device}")
except Exception as e: print(f"FATAL ERROR: Could not load embedding model: {e}")


# --- Initialize Pinecone ---
pinecone_index = None
if PINECONE_API_KEY != "YOUR_PINECONE_API_KEY": 
    try:
        print("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)

        
        existing_indexes = pc.list_indexes()
        # Extract the names into a list
        index_names = [index_info.name for index_info in existing_indexes]

        if PINECONE_INDEX_NAME not in index_names:
        # --- END OF CORRECTION ---
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating it...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1") 
            )
            print(f"Index '{PINECONE_INDEX_NAME}' created. Waiting for initialization...")
            
            while True:
                status = pc.describe_index(PINECONE_INDEX_NAME).status
                if status.get('ready') and status.get('state') == 'Ready': 
                    print(f"Index '{PINECONE_INDEX_NAME}' is ready.")
                    break
                print(f"Index '{PINECONE_INDEX_NAME}' not ready yet, current status: {status}. Waiting...")
                time.sleep(10) 
        
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}. Stats: {pinecone_index.describe_index_stats()}")
    except Exception as e:
        print(f"ERROR: Could not connect/create Pinecone index: {e}")
        pinecone_index = None # Ensuring it's None on failure
else:
    print("WARNING: Pinecone API Key not set. Pinecone functionality disabled.")
    
    
# --- Helper Functions ---
def get_git_diff(commit_hash, repo_path):
    """Fetches the code diff for a given commit hash within the specified repo path."""
    if not commit_hash or not repo_path or not os.path.isdir(repo_path) or not os.path.isdir(os.path.join(repo_path, '.git')):
        print(f"WARN: Invalid input for get_git_diff: hash={commit_hash}, path={repo_path}")
        return f"Error: Invalid repo path or commit hash for diff: {repo_path}"
    try:
        diff_command = ["git", "show", "--patch", "--pretty=format:", commit_hash]
        diff_result = subprocess.run(
            diff_command, cwd=repo_path, capture_output=True, text=True, check=False,
            encoding='utf-8', errors='replace', timeout=30
        )
        if diff_result.returncode == 0:
            diff_text = diff_result.stdout.strip()
            return diff_text if diff_text else "(No code changes detected)"
        else:
            error_message = diff_result.stderr.strip()
            print(f"ERROR getting diff for {commit_hash[:7]}: {error_message}")
            return f"Error getting diff (Code {diff_result.returncode})"
    except FileNotFoundError: return "Error: 'git' command not found."
    except subprocess.TimeoutExpired: return f"Error: `git show` timed out for {commit_hash[:7]}"
    except Exception as e: return f"Exception getting diff: {e}"


@celery_app.task(name="tasks.process_and_index_repository")
def process_and_index_repository_task(repo_url: str, repo_id_for_namespace: str):
    task_id = process_and_index_repository_task.request.id # Get Celery task ID
    print(f"CELERY TASK [{task_id}]: Starting indexing for {repo_url} into namespace {repo_id_for_namespace}")
    if not embedding_model or not pinecone_index:
        message = "Backend models or Pinecone not loaded. Aborting indexing."
        print(f"CELERY TASK [{task_id}] ERROR: {message}")
        # Update job status here (e.g., in Redis or another DB, Celery updates its own backend)
        return {"status": "error", "message": message, "indexed_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 1. Clone
            print(f"CELERY TASK [{task_id}]: Cloning {repo_url} into {tmpdir}")
            clone_command = ["git", "clone", "--depth", str(NUM_COMMITS_TO_EXTRACT_FOR_INDEXING), "--no-single-branch", repo_url, tmpdir]
            subprocess.run(clone_command, check=True, timeout=300, capture_output=True, text=True, encoding='utf-8', errors='replace')

            # 2. Extract Commits
            print(f"CELERY TASK [{task_id}]: Extracting commits from {tmpdir}")
            log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
            log_command = ["git", "log", f"--pretty=format:{log_format}"]
            log_result = subprocess.run(log_command, cwd=tmpdir, capture_output=True, text=True, check=True, timeout=120)

            commits_to_embed = [] 
            commit_messages_for_embedding = []

            commit_entries = log_result.stdout.strip().split("-----COMMIT_END-----")
            for entry in commit_entries:
                 if not entry.strip(): continue
                 parts = entry.strip().split('||', 3)
                 if len(parts) == 4:
                     h, a, t, m_part = parts; s_b = m_part.split('\n', 1); s=s_b[0].strip(); b=s_b[1].strip() if len(s_b)>1 else ""
                     full_msg = f"{s}\n{b}".strip();
                     if h and full_msg:
                         commits_to_embed.append({
                             "hash": h, "author": a, "timestamp": int(t) if t.isdigit() else 0,
                             "subject": s, "full_message": full_msg
                         })
                         commit_messages_for_embedding.append(full_msg)
            if not commits_to_embed:
                message = "No commit messages extracted."
                print(f"CELERY TASK [{task_id}] WARN: {message}")
                return {"status": "warning", "message": message, "indexed_count": 0}

            # 3. Embed Messages
            print(f"CELERY TASK [{task_id}]: Embedding {len(commit_messages_for_embedding)} messages...")
            embeddings = embedding_model.encode(commit_messages_for_embedding, batch_size=64, show_progress_bar=False)

            # 4. Prepare for Pinecone Upsert (Fetch Diffs HERE)
            vectors_for_pinecone = []
            for i, commit_details in enumerate(commits_to_embed):
                print(f"CELERY TASK [{task_id}]: Processing commit {i+1}/{len(commits_to_embed)} - Fetching diff for {commit_details['hash'][:7]}")
                
                diff_text = get_git_diff(commit_details["hash"], tmpdir)
                truncated_diff = diff_text[:MAX_DIFF_CHARS_FOR_STORAGE] + \
                                 ("..." if len(diff_text) > MAX_DIFF_CHARS_FOR_STORAGE else "")

                vectors_for_pinecone.append((
                    commit_details["hash"],       # ID
                    embeddings[i].tolist(),       # Vector
                    {                             # Metadata
                        "message": commit_details["full_message"], # Store full message
                        "author": commit_details["author"],
                        "timestamp": commit_details["timestamp"],
                        "subject": commit_details["subject"],
                        "diff_snippet": truncated_diff # Store the (truncated) diff
                    }
                ))
            print(f"CELERY TASK [{task_id}]: Prepared {len(vectors_for_pinecone)} vectors with diffs for Pinecone.")

            # 5. Upsert to Pinecone in Batches
            print(f"CELERY TASK [{task_id}]: Upserting to Pinecone namespace '{repo_id_for_namespace}'...")
            batch_size = 100 
            for i in range(0, len(vectors_for_pinecone), batch_size):
                batch = vectors_for_pinecone[i : i + batch_size]
                pinecone_index.upsert(vectors=batch, namespace=repo_id_for_namespace)
                print(f"CELERY TASK [{task_id}]: Upserted batch {i//batch_size + 1}")
            
            final_count = len(vectors_for_pinecone)
            print(f"CELERY TASK [{task_id}]: Upsert complete. Indexed {final_count} commits.")
            return {"status": "completed", "indexed_count": final_count, "message": "Indexing successful"}

        except subprocess.CalledProcessError as e:
            message = f"Git command failed during indexing: {e.cmd} - {e.stderr or e.stdout or 'No output'}"
            print(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}
        except Exception as e:
            message = f"Error during indexing for {repo_url}: {e}"
            print(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}


#Pydantic models 
class IndexRepoRequest(BaseModel): repo_url: str
class IndexRepoResponse(BaseModel): job_id: str; status: str; message: str
class QueryRepoRequest(BaseModel): repo_id: str; question: str 
class CommitInfo(BaseModel): hash: str; message: str; author: str | None = None; date: str | None = None; similarity: float; diff: str 
class QueryRepoResponse(BaseModel): status: str; message: str | None = None; relevant_commits: list[CommitInfo] = []; ai_summary: str | None = None
class JobStatusResponse(BaseModel): job_id: str; status: str; result: dict | None = None




app = FastAPI(title="MementoAI Backend API")


origins = [
    "http://localhost",
    "http://localhost:8501",
]
app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    if embedding_model is None: print("FATAL: Embedding model not loaded at startup.")
    if pinecone_index is None: print("FATAL: Pinecone index not connected at startup.")

@app.post("/index_repository", response_model=IndexRepoResponse)
async def index_repository_endpoint(request: IndexRepoRequest):
   
    if not validators.url(request.repo_url) or not request.repo_url.endswith(".git"):
        raise HTTPException(status_code=400, detail="Invalid Git repository URL.")
    if not pinecone_index:
        raise HTTPException(status_code=503, detail="Vector database (Pinecone) not configured or unavailable.")
    repo_id_for_namespace = str(uuid.uuid5(uuid.NAMESPACE_URL, request.repo_url))
    task = process_and_index_repository_task.delay(request.repo_url, repo_id_for_namespace)
    return IndexRepoResponse(job_id=task.id, status="queued", message=f"Repo indexing job queued. Repo ID/Namespace: {repo_id_for_namespace}")

@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    
    task_result = celery_app.AsyncResult(job_id)
    response = JobStatusResponse(job_id=job_id, status=task_result.status)
    if task_result.successful(): response.result = task_result.result
    elif task_result.failed(): response.result = {"error": str(task_result.info)}
    return response

@app.post("/query_repository", response_model=QueryRepoResponse)
async def query_repository_endpoint(request: QueryRepoRequest): 
    print(f"Query for repo_id: {request.repo_id}, Question: {request.question}")
    if not embedding_model or not pinecone_index:
        raise HTTPException(status_code=503, detail="Core models/DB not loaded.")
    if not request.question: raise HTTPException(status_code=400, detail="Question empty.")

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

        if not query_response.matches:
            return QueryRepoResponse(status="success", message="No relevant commits found.", relevant_commits=[])

        for match in query_response.matches:
            meta = match.metadata or {}
            commit_date = "Unknown"
            if meta.get("timestamp"):
                try: commit_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(meta.get("timestamp")))
                except: pass

            
            diff_snippet = meta.get("diff_snippet", "(Diff not stored or available in metadata)")

            retrieved_commits_info.append(CommitInfo(
                hash=match.id,
                message=meta.get("message", "Message not found in metadata."),
                author=meta.get("author", "N/A"),
                date=commit_date,
                similarity=match.score,
                diff=diff_snippet # Use the diff from metadata
            ))
        print(f"Retrieved {len(retrieved_commits_info)} commits from Pinecone.")

    except Exception as e:
        analysis_error = f"Error querying Pinecone for repo {request.repo_id}: {e}"
        print(f"ERROR: {analysis_error}")
        return QueryRepoResponse(status="error", message=analysis_error)

    
    ai_summary_text = None
    gemini_call_attempted = False
    
    if genai_configured and retrieved_commits_info and not analysis_error:
        print(f"Attempting Gemini summarization for repo_id: {request.repo_id}...")
        gemini_call_attempted = True
        context_parts = []
        for i, commit_data in enumerate(retrieved_commits_info):
            
            diff_for_prompt = "(Diff not available or applicable)" if commit_data.diff.startswith("Error") or commit_data.diff.startswith("(Diff not stored") or commit_data.diff.startswith("(No code changes") else commit_data.diff # use the diff from metadata
            
            context_parts.append(f"Commit {i+1} ({commit_data.hash[:7]}):\nMessage: {commit_data.message}\nDiff Snippet:\n```diff\n{diff_for_prompt}\n```\n---\n")
        full_context = "\n".join(context_parts)
        repo_name_guess = request.repo_id 
        prompt = f"Analyze commit info (message & diff snippets) from repo '{repo_name_guess}'...\nContext:\n---\n{full_context}---\nUser Question: {request.question}\nAnswer:"
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = model.generate_content(prompt, request_options={"timeout": 60})
            if response.parts: ai_summary_text = response.text
            elif response.prompt_feedback: ai_summary_text = f"Blocked: {response.prompt_feedback.block_reason}."
            else: ai_summary_text = "Gemini empty response."
        except Exception as e:
            print(f"ERROR calling Gemini: {e}")
            ai_summary_text = f"Error during AI summarization: {e}"
    
    if not gemini_call_attempted:
        if analysis_error: ai_summary_text = "AI Summarization skipped due to earlier analysis error."
        elif not retrieved_commits_info: ai_summary_text = "AI Summarization skipped (no relevant commits found)."
        else: ai_summary_text = "AI Summarization skipped (API Key not configured)."
    elif ai_summary_text is None:
         ai_summary_text = "AI Summarization failed for an unknown reason after attempt."

    final_status = "success" if not analysis_error else "partial_success"
    final_message = analysis_error if analysis_error else "Query complete."
    if ai_summary_text and ("Error:" in ai_summary_text or "Blocked:" in ai_summary_text) and final_status == "success":
        final_status = "partial_success"

    return QueryRepoResponse(
        status=final_status, message=final_message,
        relevant_commits=retrieved_commits_info, ai_summary=ai_summary_text
    )

@app.get("/")
async def root(): return {"message": "MementoAI Backend API with Pinecone & Celery is running!"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly for MementoAI API...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


