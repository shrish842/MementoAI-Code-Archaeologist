from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException

from shared.config import load_config
from shared.models import IndexRepositoryAccepted, IndexRepositoryRequest, JobStatusResponse

from .git_history import validate_public_git_url
from .service import IngestionService

config = load_config("ingestion", "127.0.0.1", 8001)
service = IngestionService(config)
app = FastAPI(title="MementoAI Ingestion Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": config.service_name}


@app.post("/v1/repositories/index", response_model=IndexRepositoryAccepted)
async def index_repository(request: IndexRepositoryRequest, background_tasks: BackgroundTasks) -> IndexRepositoryAccepted:
    if not validate_public_git_url(request.repo_url):
        raise HTTPException(status_code=400, detail="Only public https Git URLs ending in .git are allowed.")
    response = service.submit(request.repo_url)
    background_tasks.add_task(service.run_job, response.job_id)
    return response


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = None
    if job.status == "completed":
        result = {"status": "completed", "indexed_count": job.indexed_count}
    elif job.status == "failed":
        result = {"status": "failed", "error": job.error}
    return JobStatusResponse(job_id=job.job_id, status=job.status, repo_id=job.repo_id, result=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)