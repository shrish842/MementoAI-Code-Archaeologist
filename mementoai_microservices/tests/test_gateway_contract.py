from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FASTAPI_IMPORT_ERROR = None

try:
    from fastapi.testclient import TestClient
    from services.gateway.app import main as gateway_main
except ModuleNotFoundError as exc:
    FASTAPI_IMPORT_ERROR = exc
    TestClient = None
    gateway_main = None

from shared.models import (
    CommitHit,
    IndexRepositoryAccepted,
    JobStatusResponse,
    QueryRepositoryResponse,
)


@unittest.skipIf(FASTAPI_IMPORT_ERROR is not None, f"fastapi dependencies not installed: {FASTAPI_IMPORT_ERROR}")
class GatewayContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(gateway_main.app)

    def test_index_repository_returns_structured_repo_id(self) -> None:
        gateway_main.ingestion_client.index_repository = AsyncMock(
            return_value=IndexRepositoryAccepted(
                job_id="job-1",
                repo_id="repo-1",
                status="queued",
                message="Repository queued for indexing. Repo ID: repo-1",
            )
        )

        response = self.client.post("/index_repository", json={"repo_url": "https://github.com/psf/requests.git"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], "job-1")
        self.assertEqual(body["repo_id"], "repo-1")

    def test_query_repository_returns_partial_success_when_summary_fails(self) -> None:
        gateway_main.query_client.query_repository = AsyncMock(
            return_value=QueryRepositoryResponse(
                status="success",
                message="Query complete.",
                relevant_commits=[CommitHit(hash="abc1234", message="fix auth flow", similarity=0.92)],
                ai_summary=None,
            )
        )
        gateway_main.summary_client.summarize = AsyncMock(side_effect=RuntimeError("summary timeout"))

        response = self.client.post("/query_repository", json={"repo_id": "repo-1", "question": "why auth changed"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "partial_success")
        self.assertEqual(len(body["relevant_commits"]), 1)

    def test_job_status_passthrough(self) -> None:
        gateway_main.ingestion_client.get_job = AsyncMock(
            return_value=JobStatusResponse(
                job_id="job-1",
                status="completed",
                repo_id="repo-1",
                result={"status": "completed", "indexed_count": 10},
            )
        )

        response = self.client.get("/job_status/job-1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["result"]["indexed_count"], 10)


if __name__ == "__main__":
    unittest.main()