from __future__ import annotations

from fastapi import FastAPI

from shared.config import load_config
from shared.models import QueryRepositoryRequest, QueryRepositoryResponse

from .service import QueryService

config = load_config("query", "127.0.0.1", 8002)
service = QueryService(config)
app = FastAPI(title="MementoAI Query Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": config.service_name}


@app.post("/v1/repositories/query", response_model=QueryRepositoryResponse)
async def query_repository(request: QueryRepositoryRequest) -> QueryRepositoryResponse:
    return service.query(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)