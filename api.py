import os
import time
import subprocess
import tempfile
import torch 
import ast
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import validators
from pinecone import Pinecone, ServerlessSpec
from celery import Celery
import uuid
from difflib import unified_diff
import radon
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import lizard

print("API Script loading...") 

# Configuration 

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', "pcsk_77BFmL_KGUSpB11n15Mj6EodPPaUEBATfvnfGTDt5djVeqgGdTkv1YBbVLziZz3oEhg5Db")
PINECONE_INDEX_NAME = "mementoai"   
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384
NUM_COMMITS_TO_EXTRACT_FOR_INDEXING = 5000 
MAX_DIFF_CHARS_FOR_STORAGE = 5000 
MAX_DIFF_CHARS_FOR_LLM = 2000  
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Initialize Celery
celery_app = Celery('memento_tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True
)

# Initialize Google Gemini
genai_configured = False
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCl21GaVEZuD3kfhZjJ-axQ6SQj-Y5PFgc")
    if google_api_key:
        genai.configure(api_key=google_api_key)
        genai_configured = True
        print("INFO: Google API Key configured.")
    else:
        print("WARNING: Google API Key not found/configured.")
except Exception as e:
    print(f"ERROR during API Key setup: {e}")
    genai_configured = False

# Load embedding model
try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded. Device: {embedding_model.device}")
except Exception as e:
    print(f"FATAL ERROR: Could not load embedding model: {e}")
    embedding_model = None


# Initialize Pinecone
pinecone_index = None
if PINECONE_API_KEY:
    try:
        print("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pc.list_indexes()
        index_names = [index_info.name for index_info in existing_indexes]

        if PINECONE_INDEX_NAME not in index_names:
            print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
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
                print(f"Index '{PINECONE_INDEX_NAME}' not ready yet. Waiting...")
                time.sleep(10) 
        
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}. Stats: {pinecone_index.describe_index_stats()}")
    except Exception as e:
        print(f"ERROR: Could not connect/create Pinecone index: {e}")
        pinecone_index = None
else:
    print("WARNING: Pinecone API Key not set. Pinecone functionality disabled.")


# --- AST Analysis Functions ---
class CodeChangeAnalysis(BaseModel):
    functions_added: List[str] = []
    functions_removed: List[str] = []
    functions_modified: List[str] = []
    complexity_changes: Dict[str, int] = {}

def analyze_code_changes(diff_text: str) -> CodeChangeAnalysis:
    """Analyze code changes using AST to detect function modifications."""
    analysis = CodeChangeAnalysis()
    
    if not diff_text or "Error" in diff_text:
        return analysis
    
    try:
        # Parse the diff to get old and new code sections
        old_code = []
        new_code = []
        
        for line in diff_text.split('\n'):
            if line.startswith('-') and not line.startswith('---'):
                old_code.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                new_code.append(line[1:])
        
        # Parse functions from old and new code
        old_functions = parse_functions('\n'.join(old_code))
        new_functions = parse_functions('\n'.join(new_code))
        
        # Compare functions
        old_func_names = {f['name'] for f in old_functions}
        new_func_names = {f['name'] for f in new_functions}
        
        analysis.functions_added = list(new_func_names - old_func_names)
        analysis.functions_removed = list(old_func_names - new_func_names)
        
        # Find modified functions
        modified = []
        complexity_changes = {}
        
        for new_func in new_functions:
            if new_func['name'] in old_func_names:
                old_func = next(f for f in old_functions if f['name'] == new_func['name'])
                if new_func['body'] != old_func['body']:
                    modified.append(new_func['name'])
                    # Calculate complexity change
                    old_complexity = calculate_complexity(old_func['body'])
                    new_complexity = calculate_complexity(new_func['body'])
                    if old_complexity != new_complexity:
                        complexity_changes[new_func['name']] = new_complexity - old_complexity
        
        analysis.functions_modified = modified
        analysis.complexity_changes = complexity_changes
        
    except Exception as e:
        print(f"Error during AST analysis: {e}")
    
    return analysis

def parse_functions(code: str) -> List[Dict]:
    """Parse functions from code using AST."""
    if not code:
        return []
    
    try:
        tree = ast.parse(code)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'body': ast.unparse(node.body),
                    'args': [arg.arg for arg in node.args.args]
                })
        
        return functions
    except Exception as e:
        print(f"Error parsing functions: {e}")
        return []

def calculate_complexity(code: str) -> int:
    """Calculate simple complexity metric (number of control flow statements)."""
    try:
        tree = ast.parse(code)
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 0.5  # Weight function calls less than control flow
        
        return int(complexity)
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0


# --- Technical Debt Analysis Functions ---
class TechnicalDebtMetrics(BaseModel):
    code_smells: List[str] = []
    cyclomatic_complexity: Dict[str, int] = {}
    maintainability_index: float = 0.0
    duplication: float = 0.0
    lines_of_code: int = 0
    technical_debt_score: float = 0.0
    
    
def analyze_technical_debt(code: str, language: str = 'python') -> TechnicalDebtMetrics:
    """Analyze code for technical debt indicators."""
    metrics = TechnicalDebtMetrics()
    
    if not code:
        return metrics
    
    try:
        # Radon analysis
        try:
            # Cyclomatic complexity
            complexity_results = cc_visit(code)
            metrics.cyclomatic_complexity = {
                func.name: func.complexity 
                for func in complexity_results
            }
            
            # Maintainability index
            metrics.maintainability_index = mi_visit(code, multi=True)
            
            # Raw metrics (lines of code)
            raw_metrics = analyze(code)
            metrics.lines_of_code = raw_metrics.loc
            
        except Exception as e:
            print(f"Radon analysis error: {e}")
        
        # Lizard analysis (for duplication)
        try:
            lizard_analysis = lizard.analyze_file.analyze_source_code(
                "temp.py" if language == 'python' else "temp.file", 
                code
            )
            metrics.duplication = lizard_analysis.average_duplication * 100  # as percentage
        except Exception as e:
            print(f"Lizard analysis error: {e}")
        
        # Code smell detection (simplified)
        metrics.code_smells = detect_code_smells(code)
        
        # Calculate composite technical debt score
        metrics.technical_debt_score = calculate_debt_score(metrics)
        
    except Exception as e:
        print(f"Error in technical debt analysis: {e}")
    
    return metrics

def detect_code_smells(code: str) -> List[str]:
    """Detect common code smells."""
    smells = []
    try:
        tree = ast.parse(code)
        
        # Long Method/Function detection
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 30:
                    smells.append(f"Long function: {node.name} ({func_lines} lines)")
                
                # Count parameters
                if len(node.args.args) > 5:
                    smells.append(f"Many parameters in {node.name} ({len(node.args.args)})")
        
        # Duplicate code detection (simplified)
        # In production, you'd use a more sophisticated approach
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        function_bodies = [ast.unparse(f.body) for f in functions]
        
        for i, body1 in enumerate(function_bodies):
            for j, body2 in enumerate(function_bodies[i+1:], i+1):
                if body1 == body2:
                    smells.append(f"Duplicate code between {functions[i].name} and {functions[j].name}")
                    break
        
    except Exception as e:
        print(f"Error detecting code smells: {e}")
    
    return smells

def calculate_debt_score(metrics: TechnicalDebtMetrics) -> float:
    """Calculate composite technical debt score (0-100)."""
    score = 0
    
    # Weighted components
    if metrics.cyclomatic_complexity:
        avg_complexity = sum(metrics.cyclomatic_complexity.values()) / len(metrics.cyclomatic_complexity)
        score += min(avg_complexity * 2, 30)  # max 30 points for complexity
    
    score += (100 - metrics.maintainability_index) * 0.3  # max 30 points for MI
    
    score += metrics.duplication * 0.4  # max 40 points for duplication
    
    # Add points for each code smell
    score += min(len(metrics.code_smells) * 2, 20)  # max 20 points for smells
    
    return min(score, 100)
    

# --- Git Helper Functions ---
def get_git_diff(commit_hash: str, repo_path: str) -> str:
    """Fetches the code diff for a given commit hash."""
    if not commit_hash or not repo_path or not os.path.isdir(repo_path) or not os.path.isdir(os.path.join(repo_path, '.git')):
        print(f"WARN: Invalid input for get_git_diff: hash={commit_hash}, path={repo_path}")
        return f"Error: Invalid repo path or commit hash for diff: {repo_path}"
    
    try:
        diff_command = ["git", "show", "--patch", "--pretty=format:", commit_hash]
        diff_result = subprocess.run(
            diff_command, 
            cwd=repo_path, 
            capture_output=True, 
            text=True,
            encoding='utf-8', 
            errors='replace', 
            timeout=30
        )
        
        if diff_result.returncode == 0:
            diff_text = diff_result.stdout.strip()
            return diff_text if diff_text else "(No code changes detected)"
        else:
            error_message = diff_result.stderr.strip()
            print(f"ERROR getting diff for {commit_hash[:7]}: {error_message}")
            return f"Error getting diff (Code {diff_result.returncode})"
    except FileNotFoundError: 
        return "Error: 'git' command not found."
    except subprocess.TimeoutExpired: 
        return f"Error: `git show` timed out for {commit_hash[:7]}"
    except Exception as e: 
        return f"Exception getting diff: {e}"

# --- Celery Task ---
@celery_app.task(name="tasks.process_and_index_repository")
def process_and_index_repository_task(repo_url: str, repo_id_for_namespace: str):
    task_id = process_and_index_repository_task.request.id
    print(f"CELERY TASK [{task_id}]: Starting indexing for {repo_url} into namespace {repo_id_for_namespace}")
    
    if not embedding_model or not pinecone_index:
        message = "Backend models or Pinecone not loaded. Aborting indexing."
        print(f"CELERY TASK [{task_id}] ERROR: {message}")
        return {"status": "error", "message": message, "indexed_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 1. Clone repository
            print(f"CELERY TASK [{task_id}]: Cloning {repo_url} into {tmpdir}")
            clone_command = [
                "git", "clone", 
                "--depth", str(NUM_COMMITS_TO_EXTRACT_FOR_INDEXING), 
                "--no-single-branch", 
                repo_url, 
                tmpdir
            ]
            subprocess.run(
                clone_command, 
                check=True, 
                timeout=300, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='replace'
            )

            # 2. Extract Commits
            print(f"CELERY TASK [{task_id}]: Extracting commits from {tmpdir}")
            log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
            log_command = ["git", "log", f"--pretty=format:{log_format}"]
            log_result = subprocess.run(
                log_command, 
                cwd=tmpdir, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=120
            )

            commits_to_embed = [] 
            commit_messages_for_embedding = []

            for entry in log_result.stdout.strip().split("-----COMMIT_END-----"):
                if not entry.strip(): 
                    continue
                
                parts = entry.strip().split('||', 3)
                if len(parts) == 4:
                    h, a, t, m_part = parts
                    s_b = m_part.split('\n', 1)
                    s = s_b[0].strip()
                    b = s_b[1].strip() if len(s_b) > 1 else ""
                    full_msg = f"{s}\n{b}".strip()
                    
                    if h and full_msg:
                        commits_to_embed.append({
                            "hash": h, 
                            "author": a, 
                            "timestamp": int(t) if t.isdigit() else 0,
                            "subject": s, 
                            "full_message": full_msg
                        })
                        commit_messages_for_embedding.append(full_msg)

            if not commits_to_embed:
                message = "No commit messages extracted."
                print(f"CELERY TASK [{task_id}] WARN: {message}")
                return {"status": "warning", "message": message, "indexed_count": 0}

            # 3. Embed Messages
            print(f"CELERY TASK [{task_id}]: Embedding {len(commit_messages_for_embedding)} messages...")
            embeddings = embedding_model.encode(
                commit_messages_for_embedding, 
                batch_size=64, 
                show_progress_bar=False
            )

            # 4. Prepare for Pinecone Upsert with AST Analysis
            vectors_for_pinecone = []
            for i, commit_details in enumerate(commits_to_embed):
                print(f"CELERY TASK [{task_id}]: Processing commit {i+1}/{len(commits_to_embed)} - {commit_details['hash'][:7]}")
                
                diff_text = get_git_diff(commit_details["hash"], tmpdir)
                old_code, new_code = extract_code_states(commit_details["hash"], tmpdir)
                old_debt = analyze_technical_debt(old_code) if old_code else None
                new_debt = analyze_technical_debt(new_code) if new_code else None
                debt_delta = new_debt.technical_debt_score - old_debt.technical_debt_score if old_debt and new_debt else None
                truncated_diff = diff_text[:MAX_DIFF_CHARS_FOR_STORAGE] + \
                                 ("..." if len(diff_text) > MAX_DIFF_CHARS_FOR_STORAGE else "")
                
                # Perform AST analysis on the diff
                code_analysis = analyze_code_changes(diff_text)
                
                vectors_for_pinecone.append((
                    commit_details["hash"],       # ID
                    embeddings[i].tolist(),       # Vector
                    {                            # Metadata
                        "message": commit_details["full_message"],
                        "author": commit_details["author"],
                        "timestamp": commit_details["timestamp"],
                        "subject": commit_details["subject"],
                        "diff_snippet": truncated_diff,
                        "functions_added": code_analysis.functions_added,
                        "functions_removed": code_analysis.functions_removed,
                        "functions_modified": code_analysis.functions_modified,
                        "complexity_changes": code_analysis.complexity_changes,
                        "technical_debt": new_debt.dict() if new_debt else None,
                        "old_technical_debt": old_debt.dict() if old_debt else None,
                        "debt_delta": debt_delta    
                    }
                ))

            print(f"CELERY TASK [{task_id}]: Prepared {len(vectors_for_pinecone)} vectors with AST analysis.")

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


def extract_code_states(commit_hash: str, repo_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Get the before and after code states for a commit."""
    try:
        # Get the parent commit
        parent_cmd = ["git", "rev-parse", f"{commit_hash}^"]
        parent_result = subprocess.run(
            parent_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if parent_result.returncode != 0:
            return None, None
        
        parent_hash = parent_result.stdout.strip()
        
        # Get changed files in the commit
        files_cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
        files_result = subprocess.run(
            files_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if files_result.returncode != 0:
            return None, None
        
        changed_files = files_result.stdout.strip().split('\n')
        if not changed_files:
            return None, None
        
        # For simplicity, we'll analyze the first changed file
        target_file = changed_files[0]
        
        # Get old version (parent)
        old_cmd = ["git", "show", f"{parent_hash}:{target_file}"]
        old_result = subprocess.run(
            old_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # Get new version (current commit)
        new_cmd = ["git", "show", f"{commit_hash}:{target_file}"]
        new_result = subprocess.run(
            new_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        old_code = old_result.stdout if old_result.returncode == 0 else None
        new_code = new_result.stdout if new_result.returncode == 0 else None
        
        return old_code, new_code
    
    except Exception as e:
        print(f"Error extracting code states: {e}")
        return None, None


# --- Pydantic Models ---
class IndexRepoRequest(BaseModel): 
    repo_url: str

class IndexRepoResponse(BaseModel): 
    job_id: str
    status: str
    message: str

class FunctionChange(BaseModel):
    name: str
    change_type: str  # 'added', 'removed', 'modified'
    complexity_change: Optional[int] = None

class CommitInfo(BaseModel): 
    hash: str
    message: str
    author: Optional[str] = None
    date: Optional[str] = None
    similarity: float
    diff: str
    function_changes: List[FunctionChange] = []
    technical_debt: Optional[TechnicalDebtMetrics] = None
    old_technical_debt: Optional[TechnicalDebtMetrics] = None
    debt_delta: Optional[float] = None

class QueryRepoRequest(BaseModel): 
    repo_id: str
    question: str

class QueryRepoResponse(BaseModel): 
    status: str
    message: Optional[str] = None
    relevant_commits: List[CommitInfo] = []
    ai_summary: Optional[str] = None

class JobStatusResponse(BaseModel): 
    job_id: str
    status: str
    result: Optional[dict] = None


# --- FastAPI App ---
app = FastAPI(title="MementoAI Backend API")

# CORS Configuration
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

@app.on_event("startup")
async def startup_event():
    if embedding_model is None: 
        print("FATAL: Embedding model not loaded at startup.")
    if pinecone_index is None: 
        print("FATAL: Pinecone index not connected at startup.")

@app.post("/index_repository", response_model=IndexRepoResponse)
async def index_repository_endpoint(request: IndexRepoRequest):
    if not validators.url(request.repo_url) or not request.repo_url.endswith(".git"):
        raise HTTPException(status_code=400, detail="Invalid Git repository URL.")
    if not pinecone_index:
        raise HTTPException(status_code=503, detail="Vector database (Pinecone) not configured or unavailable.")
    
    repo_id_for_namespace = str(uuid.uuid5(uuid.NAMESPACE_URL, request.repo_url))
    task = process_and_index_repository_task.delay(request.repo_url, repo_id_for_namespace)
    
    return IndexRepoResponse(
        job_id=task.id,
        status="queued",
        message=f"Repo indexing job queued. Repo ID/Namespace: {repo_id_for_namespace}"
    )

@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    task_result = celery_app.AsyncResult(job_id)
    response = JobStatusResponse(job_id=job_id, status=task_result.status)
    
    if task_result.successful(): 
        response.result = task_result.result
    elif task_result.failed(): 
        response.result = {"error": str(task_result.info)}
    
    return response

@app.post("/query_repository", response_model=QueryRepoResponse)
async def query_repository_endpoint(request: QueryRepoRequest): 
    print(f"Query for repo_id: {request.repo_id}, Question: {request.question}")
    
    if not embedding_model or not pinecone_index:
        raise HTTPException(status_code=503, detail="Core models/DB not loaded.")
    if not request.question: 
        raise HTTPException(status_code=400, detail="Question empty.")

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
                complexity_changes = meta.get("complexity_changes", {})
                for func_name in meta.get("functions_modified", []):
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="modified",
                        complexity_change=complexity_changes.get(func_name, 0)
                    ))

                retrieved_commits_info.append(CommitInfo(
                    hash=match.id,
                    message=meta.get("message", "Message not found in metadata."),
                    author=meta.get("author", "N/A"),
                    date=commit_date,
                    similarity=match.score,
                    diff=diff_snippet,
                    function_changes=function_changes
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
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
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
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)





































