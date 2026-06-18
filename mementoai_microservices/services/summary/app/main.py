from __future__ import annotations

from fastapi import FastAPI

from shared.config import load_config
from shared.models import SummaryRequest, SummaryResponse

from .service import SummaryService

config = load_config("summary", "127.0.0.1", 8003)
service = SummaryService(config)
app = FastAPI(title="MementoAI Summary Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": config.service_name}


@app.post("/v1/summaries/repository-query", response_model=SummaryResponse)
async def summarize(request: SummaryRequest) -> SummaryResponse:
    return service.summarize(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)