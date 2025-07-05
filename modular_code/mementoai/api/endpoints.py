from fastapi import APIRouter, HTTPException, Response
import time
import json
import uuid
import logging

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

router = APIRouter()
logger = logging.getLogger("api")

@router.post("/index_repository", response_model=IndexRepoResponse)
async def index_repository_endpoint(request: IndexRepoRequest):
    """Initiate repository indexing"""
    if not is_valid_git_url(request.repo_url):
        raise HTTPException(400, "Invalid Git URL. Must end with .git")
    
    if not pinecone_index:
        raise HTTPException(503, "Pinecone service unavailable")

    try:
        repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, request.repo_url))
        task = process_and_index_repository_task.delay(request.repo_url, repo_id)
        
        return IndexRepoResponse(
            job_id=task.id,
            status="queued",
            message=f"Indexing queued. Namespace: {repo_id}"
        )
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise HTTPException(500, "Internal server error")

@router.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    """Check indexing job status"""
    task = process_and_index_repository_task.AsyncResult(job_id)
    response = JobStatusResponse(job_id=job_id, status=task.status)
    
    if task.ready():
        if task.successful():
            response.result = task.result
        else:
            response.result = {"error": str(task.result)}
    
    return response

@router.post("/query_repository", response_model=QueryRepoResponse)
async def query_repository_endpoint(request: QueryRepoRequest, response: Response):
    """Query indexed repository with natural language"""
    # Validate inputs
    if not embedding_model or not pinecone_index:
        raise HTTPException(503, "Core services unavailable")
    
    if not request.repo_id or not request.question.strip():
        raise HTTPException(400, "Missing repo_id or question")
    
    # Set cache header
    response.headers["Cache-Control"] = "public, max-age=60"
    
    # Initialize response components
    analysis_error = None
    commits = []
    ai_summary = "AI summary not generated"
    
    try:
        # Generate query embedding
        query_embed = embedding_model.encode([request.question]).tolist()[0]
        
        # Query Pinecone
        pinecone_response = pinecone_index.query(
            namespace=request.repo_id,
            vector=query_embed,
            top_k=5,
            include_metadata=True
        )
        
        # Process results
        for match in pinecone_response.get('matches', []):
            meta = match.get('metadata', {})
            commit_id = match.get('id', '')
            
            # Format date
            commit_date = "Unknown"
            if 'timestamp' in meta:
                try:
                    commit_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', 
                                              time.gmtime(meta['timestamp']))
                except Exception:
                    pass
            
            # Parse function changes
            functions = []
            for change_type in ['added', 'removed', 'modified']:
                for func in meta.get(f'functions_{change_type}', []):
                    functions.append(FunctionChange(
                        name=func,
                        change_type=change_type
                    ))
            
            # Parse technical debt
            tech_debt = None
            old_tech_debt = None
            debt_delta = meta.get('debt_delta', 0.0)
            
            for debt_type, debt_var in [('technical_debt', tech_debt), 
                                      ('old_technical_debt', old_tech_debt)]:
                if debt_type in meta:
                    try:
                        debt_data = json.loads(meta[debt_type])
                        if debt_var == tech_debt:
                            tech_debt = TechnicalDebtMetrics(**debt_data)
                        else:
                            old_tech_debt = TechnicalDebtMetrics(**debt_data)
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            # Create commit info
            commits.append(CommitInfo(
                hash=commit_id,
                message=meta.get('message', 'No message'),
                author=meta.get('author', 'Unknown'),
                date=commit_date,
                similarity=match.get('score', 0),
                diff=meta.get('diff_snippet', 'Diff not available'),
                function_changes=functions,
                technical_debt=tech_debt,
                old_technical_debt=old_tech_debt,
                debt_delta=debt_delta
            ))
        
        logger.info(f"Found {len(commits)} relevant commits")
        
    except Exception as e:
        analysis_error = f"Query failed: {str(e)}"
        logger.error(analysis_error)
    
    # Generate AI summary if possible
    if genai_configured and commits and not analysis_error:
        try:
            # Build context
            context = []
            for i, commit in enumerate(commits):
                context.append(
                    f"Commit {i+1} ({commit.hash[:7]}):\n"
                    f"Message: {commit.message}\n"
                    f"Debt Score: {commit.technical_debt.technical_debt_score if commit.technical_debt else 'N/A'}"
                )
            
            # FIXED: Proper string formatting
            context_block = "\n\n".join(context)
            prompt = (
                f"User question: {request.question}\n\n"
                "Relevant commit summaries:\n"
                f"{'-'*40}\n"
                f"{context_block}\n"
                f"{'-'*40}\n"
                "Provide a concise technical summary connecting these changes to the question:"
            )
            
            ai_summary = generate_summary(prompt)
            logger.info(f"Generated summary: {ai_summary[:100]}...")
        except Exception as e:
            ai_summary = f"Summary error: {str(e)}"
            logger.error(f"Gemini failed: {str(e)}")
    else:
        # Fallback reasons
        if analysis_error:
            ai_summary = "Summary skipped: analysis error"
        elif not commits:
            ai_summary = "Summary skipped: no relevant commits"
        elif not genai_configured:
            ai_summary = "Summary skipped: Gemini not configured"
    
    # Determine response status
    status = "success"
    message = "Query completed"
    
    if analysis_error:
        status = "partial_success"
        message = analysis_error
    elif "error" in ai_summary.lower():
        status = "partial_success"
        message = "Summary generation issues"
    
    return QueryRepoResponse(
        status=status,
        message=message,
        relevant_commits=commits,
        ai_summary=ai_summary
    )