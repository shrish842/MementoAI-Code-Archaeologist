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

print("API Script loading...")


EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
NUM_COMMITS_TO_ANALYZE = 1000 
MAX_DIFF_CHARS = 2000 # Limit diff size sent to LLM


genai_configured = False
google_api_key_to_use = None

try:
    # Prioritize Environment Variable
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        google_api_key_to_use = env_key
        print("INFO: Found Google API Key in Environment Variable.")
    else:
        manual_key = "" # <-- PASTE YOUR *VALID* GEMINI KEY HERE
        if manual_key and manual_key != "YOUR_GOOGLE_API_KEY_GOES_HERE":
            google_api_key_to_use = manual_key
            print("INFO: Found Manual Google API Key in script.")
        else:
             print("WARNING: Google API key not found in env vars or script placeholder.")

   
    if google_api_key_to_use:
        try:
            genai.configure(api_key=google_api_key_to_use)
            
            genai_configured = True
            print(f"INFO: genai.configure called successfully. Will use model '{GEMINI_MODEL_NAME}' when needed.")
        except Exception as config_e:
            
            print(f"ERROR: genai.configure failed: {config_e}")
            genai_configured = False 
    else:
        genai_configured = False 
except Exception as e:
    
    print(f"ERROR during API Key setup phase: {e}")
    genai_configured = False


embedding_model = None
try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    t_start = time.time()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded ({time.time() - t_start:.2f}s). Device: {embedding_model.device}")
except Exception as e:
    print(f"FATAL ERROR: Could not load embedding model: {e}")
    embedding_model = None 

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


class AnalyzeRequest(BaseModel):
    repo_url: str
    question: str

class CommitInfo(BaseModel):
    hash: str
    message: str
    author: str | None = None
    date: str | None = None
    similarity: float
    diff: str

class AnalyzeResponse(BaseModel):
    status: str = "processing"
    message: str | None = None 
    relevant_commits: list[CommitInfo] = []
    ai_summary: str | None = None


app = FastAPI(title="MementoAI Backend API")

origins = [
    "http://localhost",
    "http://localhost:8501",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(request: AnalyzeRequest):
    """
    Analyzes the recent commit history of a given public Git repository URL.
    """
    print(f"Received request for URL: {request.repo_url}, Question: {request.question}")

    
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded on server.")
    if not validators.url(request.repo_url) or not request.repo_url.endswith(".git"):
         raise HTTPException(status_code=400, detail="Invalid Git repository URL format (must be public URL ending in .git).")
    if not request.question:
         raise HTTPException(status_code=400, detail="Question cannot be empty.")

    
    analysis_error = None
    top_commits_for_summary = [] 

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory: {tmpdir}")
        try:
            # 1. Shallow Clone
            print(f"Cloning recent {NUM_COMMITS_TO_ANALYZE} commits from {request.repo_url}...")
            clone_command = ["git", "clone", "--depth", str(NUM_COMMITS_TO_ANALYZE), "--no-single-branch", request.repo_url, tmpdir]
            subprocess.run(clone_command, capture_output=True, text=True, check=True, timeout=90, encoding='utf-8', errors='replace')
            print("Clone successful.")

            # 2. Extract Commits
            print("Extracting commits...")
            log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
            log_command = ["git", "log", f"-n{NUM_COMMITS_TO_ANALYZE}", f"--pretty=format:{log_format}"]
            log_result = subprocess.run(log_command, cwd=tmpdir, capture_output=True, text=True, check=True, timeout=30, encoding='utf-8', errors='replace')

            commits_data_list = []
            commit_messages_list = []
            commit_entries = log_result.stdout.strip().split("-----COMMIT_END-----")
            for entry in commit_entries:
                 if not entry.strip(): continue
                 parts = entry.strip().split('||', 3)
                 if len(parts) == 4:
                     h, a, t, m_part = parts; s_b = m_part.split('\n', 1); s=s_b[0].strip(); b=s_b[1].strip() if len(s_b)>1 else ""
                     full_msg = f"{s}\n{b}".strip();
                     if h and full_msg:
                         commits_data_list.append({"hash": h, "author": a, "timestamp": int(t) if t.isdigit() else 0, "full_message": full_msg})
                         commit_messages_list.append(full_msg)
            print(f"Extracted {len(commits_data_list)} commits.")
            if not commits_data_list: raise ValueError("No valid commits found in recent history.")

            # 3. Embed
            print("Embedding messages and question...")
            commit_embeddings = embedding_model.encode(commit_messages_list)
            question_embedding = embedding_model.encode([request.question])[0]

            # 4. Search
            print("Performing semantic search...")
            search_hits = util.semantic_search(torch.tensor(question_embedding), torch.tensor(commit_embeddings), top_k=5)[0]

            # 5. Getting Top Commits & Fetch Diffs
            print("Fetching diffs...")
            if not search_hits: print("No relevant commits found from semantic search.")
            for hit in search_hits:
                 commit_index = hit['corpus_id']; retrieved_commit_dict = commits_data_list[commit_index]
                 similarity_score = hit['score']; commit_date = "Unknown"
                 timestamp = retrieved_commit_dict.get('timestamp')
                 if timestamp:
                     try: commit_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp))
                     except: pass
                 # Fetching diff using the temporary directory path
                 diff_text = get_git_diff(retrieved_commit_dict['hash'], tmpdir)
                
                 top_commits_for_summary.append(CommitInfo(
                     hash=retrieved_commit_dict['hash'], message=retrieved_commit_dict['full_message'],
                     author=retrieved_commit_dict.get('author', 'N/A'), date=commit_date,
                     similarity=similarity_score, diff=diff_text ))
            print(f"Processed {len(top_commits_for_summary)} top commits.")

        # Catch specific errors from the process
        except subprocess.CalledProcessError as e:
            analysis_error = f"Git command failed during analysis: {e.stderr or e.stdout or 'No output'}"
            print(f"ERROR: {analysis_error}")
        except subprocess.TimeoutExpired as e:
            analysis_error = f"Git command timed out during analysis: {e.cmd}"
            print(f"ERROR: {analysis_error}")
        except ValueError as e: 
             analysis_error = str(e)
             print(f"WARN: {analysis_error}")
        except Exception as e:
            analysis_error = f"Unexpected error during analysis: {e}"
            print(f"ERROR: {analysis_error}")

    
    print("Temporary directory cleaned up.")

    # If a critical error stopped processing before getting commits
    if analysis_error and not top_commits_for_summary:
        
        return AnalyzeResponse(status="error", message=analysis_error)

    
    ai_summary_text = None
    gemini_call_attempted = False

    
    if genai_configured and top_commits_for_summary and not analysis_error:
        print(f"Attempting Gemini summarization using model: {GEMINI_MODEL_NAME}...")
        gemini_call_attempted = True
        
        context_parts = []
        for i, commit_data in enumerate(top_commits_for_summary):
            # Using dot notation for Pydantic model attributes
            diff_to_truncate = commit_data.diff
            truncated_diff = diff_to_truncate[:MAX_DIFF_CHARS] + \
                             ("..." if len(diff_to_truncate) > MAX_DIFF_CHARS else "")
            diff_for_prompt = "(Diff not available or applicable)" if truncated_diff.startswith("Error") or truncated_diff.startswith("(No code changes") else truncated_diff
            context_parts.append(
                f"Commit {i+1} ({commit_data.hash[:7]}):\nMessage: {commit_data.message}\nDiff Snippet:\n```diff\n{diff_for_prompt}\n```\n---\n"
            )
        full_context = "\n".join(context_parts)
        repo_name = request.repo_url.split('/')[-1].replace('.git','')
        prompt = f"Analyze commit info (message & diff snippets) from RECENT history of '{repo_name}' repo...\nContext:\n---\n{full_context}---\nUser Question: {request.question}\nAnswer:"

        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = model.generate_content(prompt, request_options={"timeout": 60})

            if response.parts:
                ai_summary_text = response.text
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                ai_summary_text = f"Content blocked by Gemini: {response.prompt_feedback.block_reason}."
                print(f"WARN: Gemini blocked content. Reason: {response.prompt_feedback.block_reason}")
            else:
                ai_summary_text = "Gemini returned an empty response."
                print("WARN: Gemini returned empty response.")
            print("Gemini summarization attempt finished.")

        except Exception as e: 
            print(f"ERROR calling Google Gemini API: {e}")
            error_str = str(e)
            
            if "API key not valid" in error_str or "PERMISSION_DENIED" in error_str or "API_KEY_INVALID" in error_str:
                 ai_summary_text = "Error: Invalid or unauthorized Google API Key. Please check configuration."
            elif "404" in error_str and "Model" in error_str and "not found" in error_str:
                 ai_summary_text = f"Error: Gemini Model '{GEMINI_MODEL_NAME}' not found or accessible with your key."
            else:
                 ai_summary_text = f"Error during AI summarization: {e}"

    
    if not gemini_call_attempted:
        if analysis_error:
            ai_summary_text = "AI Summarization skipped due to earlier analysis error."
        elif not top_commits_for_summary:
            ai_summary_text = "AI Summarization skipped (no relevant commits found)."
        else: 
            ai_summary_text = "AI Summarization skipped (API Key not configured)."
    elif ai_summary_text is None: 
         ai_summary_text = "AI Summarization failed for an unknown reason after attempt."


    final_status = "success"
    final_message = "Analysis complete."
    if analysis_error:
        final_status = "partial_success" # Core analysis had issues
        final_message = analysis_error
    if ai_summary_text and "Error:" in ai_summary_text:
        final_status = "partial_success" if final_status == "success" else final_status # Keep 'error' if analysis failed
       

    return AnalyzeResponse(
        status=final_status,
        message=final_message,
        relevant_commits=top_commits_for_summary,
        ai_summary=ai_summary_text
    )

@app.get("/")
async def root():
    return {"message": "MementoAI Backend API is running!"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True) 