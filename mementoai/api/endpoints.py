# mementoai/api/endpoints.py

from fastapi import APIRouter, HTTPException, Request
import time
import json
import uuid

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
    ai_summary_text = None # Initialize to None

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

                # Ensure these are always lists, even if empty in metadata
                functions_added = meta.get("functions_added", []) or []
                functions_removed = meta.get("functions_removed", []) or []
                functions_modified = meta.get("functions_modified", []) or []

                # Deserialize complexity_changes from JSON string to dict
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

                # Technical Debt
                technical_debt_data = None
                old_technical_debt_data = None
                debt_delta = meta.get("debt_delta", 0.0)

                # Deserialize technical_debt from JSON string to Pydantic model
                # Ensure that if json.loads fails or dict is empty, technical_debt_data remains None
                if meta.get("technical_debt"):
                    try:
                        td_dict = json.loads(meta["technical_debt"])
                        if td_dict: # Only create TechnicalDebtMetrics if td_dict is not empty
                            technical_debt_data = TechnicalDebtMetrics(**td_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Warning: Could not parse technical_debt metadata: {e}")

                # Deserialize old_technical_debt from JSON string to Pydantic model
                # Ensure that if json.loads fails or dict is empty, old_technical_debt_data remains None
                if meta.get("old_technical_debt"):
                    try:
                        old_td_dict = json.loads(meta["old_technical_debt"])
                        if old_td_dict: # Only create TechnicalDebtMetrics if old_td_dict is not empty
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
                    technical_debt=technical_debt_data, # This will be None if parsing failed or dict was empty
                    old_technical_debt=old_technical_debt_data, # This will be None if parsing failed or dict was empty
                    debt_delta=debt_delta
                ))

            print(f"Retrieved {len(retrieved_commits_info)} commits from Pinecone.")

    except Exception as e:
        analysis_error = f"Error querying Pinecone for repo {request.repo_id}: {e}"
        print(f"ERROR: {analysis_error}")
        # Do not return here, let the AI summary logic run and then return the QueryRepoResponse
        # with the error message and potentially empty commits.

    # Generate AI summary if configured
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
                    # Ensure complexity_change is not None before formatting
                    complexity = f" (complexity {'+' if change.complexity_change > 0 else ''}{change.complexity_change})" if change.complexity_change is not None else ""
                    func_changes.append(f"Modified function: {change.name}{complexity}")

            func_changes_text = "\n".join(func_changes) if func_changes else "No function changes detected"

            context_parts.append(
                f"Commit {i+1} ({commit_data.hash[:7]}):\n"
                f"Message: {commit_data.message}\n"
                f"Function Changes:\n{func_changes_text}\n"
                f"Diff Snippet:\n")