from __future__ import annotations

import os

import httpx

from shared.models import (
    IndexRepositoryAccepted,
    IndexRepositoryRequest,
    JobStatusResponse,
    QueryRepositoryRequest,
    QueryRepositoryResponse,
    SummaryRequest,
    SummaryResponse,
)


class IngestionClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("INGESTION_BASE_URL", "http://127.0.0.1:8001")

    async def index_repository(self, request: IndexRepositoryRequest) -> IndexRepositoryAccepted:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{self.base_url}/v1/repositories/index", json=request.model_dump())
            response.raise_for_status()
            return IndexRepositoryAccepted.model_validate(response.json())

    async def get_job(self, job_id: str) -> JobStatusResponse:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            return JobStatusResponse.model_validate(response.json())


class QueryClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("QUERY_BASE_URL", "http://127.0.0.1:8002")

    async def query_repository(self, request: QueryRepositoryRequest) -> QueryRepositoryResponse:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/v1/repositories/query", json=request.model_dump())
            response.raise_for_status()
            return QueryRepositoryResponse.model_validate(response.json())


class SummaryClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8003")

    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{self.base_url}/v1/summaries/repository-query", json=request.model_dump())
            response.raise_for_status()
            return SummaryResponse.model_validate(response.json())