from __future__ import annotations

from fastapi import FastAPI, HTTPException

from shared.config import load_config
from shared.logging import get_logger, log_event
from shared.models import (
    IndexRepositoryAccepted,
    IndexRepositoryRequest,
    JobStatusResponse,
    QueryRepositoryRequest,
    QueryRepositoryResponse,
    SummaryRequest,
)

from .clients import IngestionClient, QueryClient, SummaryClient

config = load_config("gateway", "127.0.0.1", 8000)
logger = get_logger(config.service_name)
ingestion_client = IngestionClient()
query_client = QueryClient()
summary_client = SummaryClient()

app = FastAPI(title="MementoAI Gateway")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": config.service_name}


@app.post("/index_repository", response_model=IndexRepositoryAccepted)
async def index_repository(request: IndexRepositoryRequest) -> IndexRepositoryAccepted:
    log_event(logger, "gateway.index_repository.request", repo_url=request.repo_url)
    try:
        return await ingestion_client.index_repository(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingestion service unavailable: {exc}") from exc


@app.get("/job_status/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    try:
        return await ingestion_client.get_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Job status unavailable: {exc}") from exc


@app.post("/query_repository", response_model=QueryRepositoryResponse)
async def query_repository(request: QueryRepositoryRequest) -> QueryRepositoryResponse:
    log_event(logger, "gateway.query_repository.request", repo_id=request.repo_id)
    try:
        query_result = await query_client.query_repository(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query service unavailable: {exc}") from exc

    if not query_result.relevant_commits:
        return query_result

    try:
        summary_result = await summary_client.summarize(
            SummaryRequest(
                repo_id=request.repo_id,
                question=request.question,
                commits=query_result.relevant_commits,
            )
        )
    except Exception as exc:
        return QueryRepositoryResponse(
            status="partial_success",
            message=f"Query succeeded but summary service failed: {exc}",
            relevant_commits=query_result.relevant_commits,
            ai_summary=None,
        )

    return QueryRepositoryResponse(
        status=summary_result.status if summary_result.status != "error" else "partial_success",
        message=summary_result.message or query_result.message,
        relevant_commits=query_result.relevant_commits,
        ai_summary=summary_result.summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)