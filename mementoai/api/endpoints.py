# mementoai/api/endpoints.py

from fastapi import APIRouter, HTTPException, Request
import time
import json
import uuid
import os

from core.models import (
    IndexRepoRequest, IndexRepoResponse,
    QueryRepoRequest, QueryRepoResponse,
    JobStatusResponse, CommitInfo, FunctionChange, TechnicalDebtMetrics
)
from services.celery_tasks import process_and_index_repository_task
from services.embedding_service import embedding_model
from services.pinecone_service import pinecone_index
from services.gemini_service import generate_summary, genai_configured
from utils.validators import is_valid_git_url
from core.exceptions import InvalidInputError
from pinecone import Pinecone

router = APIRouter()

@router.post("/index_repository", response_model=IndexRepoResponse)
async def index_repository_endpoint(request: IndexRepoRequest):
    """
    Endpoint to initiate the indexing of a Git repository.
    """
    if not is_valid_git_url(request.repo_url):
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

@router.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    """
    Endpoint to check the status of a background indexing job.
    """
    task_result = process_and_index_repository_task.AsyncResult(job_id)
    response = JobStatusResponse(job_id=job_id, status=task_result.status)

    print("Getting status")

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', "pcsk_77BFmL_KGUSpB11n15Mj6EodPPaUEBATfvnfGTDt5djVeqgGdTkv1YBbVLziZz3oEhg5Db")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    print(existing_indexes)

    if task_result.successful():
        response.result = task_result.result
    elif task_result.failed():
        response.result = {"error": str(task_result.info)}

    return response

@router.post("/query_repository", response_model=QueryRepoResponse)
async def query_repository_endpoint(request: QueryRepoRequest):
    """
    Endpoint to query an indexed repository using natural language.
    """
    print(f"Query for repo_id: {request.repo_id}, Question: {request.question}")

    if not embedding_model or not pinecone_index:
        raise HTTPException(status_code=503, detail="Core models/DB not loaded.")
    if not request.question:
        raise HTTPException(status_code=400, detail="Question empty.")

    analysis_error = None
    retrieved_commits_info = []
    ai_summary_text = None

    try:
        question_embedding = embedding_model.encode([request.question]).tolist()
        query_response = pinecone_index.query(
            namespace=request.repo_id,
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )

        print("Query response", query_response)

        if query_response.matches:
            for match in query_response.matches:
                meta = match.metadata or {}
                commit_date = "Unknown"

                if meta.get("timestamp"):
                    try:
                        commit_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(meta.get("timestamp")))
                    except Exception:
                        pass

                diff_snippet = meta.get("diff_snippet", "(Diff not stored or available in metadata)")

                function_changes = []

                functions_added = meta.get("functions_added", []) or []
                functions_removed = meta.get("functions_removed", []) or []
                functions_modified = meta.get("functions_modified", []) or []

                complexity_changes_str = meta.get("complexity_changes", "{}")
                complexity_changes = {}
                try:
                    complexity_changes = json.loads(complexity_changes_str)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode complexity_changes: {complexity_changes_str}")

                for func_name in functions_added:
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="added"
                    ))
                for func_name in functions_removed:
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="removed"
                    ))
                for func_name in functions_modified:
                    function_changes.append(FunctionChange(
                        name=func_name,
                        change_type="modified",
                        complexity_change=complexity_changes.get(func_name, 0)
                    ))

                technical_debt_data = None
                old_technical_debt_data = None
                debt_delta = meta.get("debt_delta", 0.0)

                if meta.get("technical_debt"):
                    try:
                        td_dict = json.loads(meta["technical_debt"])
                        if td_dict:
                            technical_debt_data = TechnicalDebtMetrics(**td_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Warning: Could not parse technical_debt metadata: {e}")

                if meta.get("old_technical_debt"):
                    try:
                        old_td_dict = json.loads(meta["old_technical_debt"])
                        if old_td_dict:
                            old_technical_debt_data = TechnicalDebtMetrics(**old_td_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Warning: Could not parse old_technical_debt metadata: {e}")

                retrieved_commits_info.append(CommitInfo(
                    hash=match.id,
                    message=meta.get("message", "Message not found in metadata."),
                    author=meta.get("author", "N/A"),
                    date=commit_date,
                    similarity=match.score,
                    diff=diff_snippet,
                    function_changes=function_changes,
                    technical_debt=technical_debt_data,
                    old_technical_debt=old_technical_debt_data,
                    debt_delta=debt_delta
                ))

            print(f"Retrieved {len(retrieved_commits_info)} commits from Pinecone.")

    except Exception as e:
        analysis_error = f"Error querying Pinecone for repo {request.repo_id}: {e}"
        print(f"ERROR: {analysis_error}")

    gemini_call_attempted = False

    if genai_configured and retrieved_commits_info and not analysis_error:
        print(f"Attempting Gemini summarization for repo_id: {request.repo_id}...")
        gemini_call_attempted = True
        try:
            context_parts = []
            for i, commit_data in enumerate(retrieved_commits_info):
                diff_for_prompt = "(Diff not available)" if commit_data.diff.startswith("Error") else commit_data.diff
                # Truncate diff to avoid excessive token usage
                truncated_diff = diff_for_prompt[:6000]

                func_changes = []
                for change in commit_data.function_changes:
                    if change.change_type == "added":
                        func_changes.append(f"Added function: {change.name}")
                    elif change.change_type == "removed":
                        func_changes.append(f"Removed function: {change.name}")
                    else:
                        complexity = ""
                        if change.complexity_change is not None:
                            sign = "+" if change.complexity_change > 0 else ""
                            complexity = f" (complexity {sign}{change.complexity_change})"
                        func_changes.append(f"Modified function: {change.name}{complexity}")

                func_changes_text = "\n".join(func_changes) if func_changes else "No function changes detected"

                debt_line = ""
                if commit_data.technical_debt or commit_data.old_technical_debt:
                    debt_line = f"TechDebt Δ: {commit_data.debt_delta}"

                context_parts.append(
                    f"Commit {i+1} ({commit_data.hash[:7]}):\n"
                    f"Message: {commit_data.message}\n"
                    f"{debt_line}\n"
                    f"Function Changes:\n{func_changes_text}\n"
                    f"Diff Snippet:\n{truncated_diff}\n"
                    "----\n"
                )

            context_text = "\n".join(context_parts)
            ai_summary_text = generate_summary(
                question=request.question,
                context=context_text
            )

        except Exception as e:
            print(f"Gemini summarization failed: {e}")
            # Do not overwrite existing analysis_error if one already exists
            if not analysis_error:
                analysis_error = f"Gemini summarization failed: {e}"

    return QueryRepoResponse(
        relevant_commits=retrieved_commits_info,
        ai_summary=ai_summary_text,
        message=analysis_error,
        status="FINISHED"
        # gemini_call_attempted=gemini_call_attempted
    )

# status: str
#     message: Optional[str] = None
#     relevant_commits: List[CommitInfo] = []
#     ai_summary: Optional[str] = None