from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FASTAPI_IMPORT_ERROR = None
os.environ.setdefault("MEMENTO_EMBEDDINGS_MODE", "hash")

try:
    from fastapi.testclient import TestClient
    from services.ingestion.app import main as ingestion_main
except ModuleNotFoundError as exc:
    FASTAPI_IMPORT_ERROR = exc
    TestClient = None
    ingestion_main = None


@unittest.skipIf(FASTAPI_IMPORT_ERROR is not None, f"fastapi dependencies not installed: {FASTAPI_IMPORT_ERROR}")
class IngestionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(ingestion_main.app)

    def test_rejects_suspicious_repo_url(self) -> None:
        response = self.client.post("/v1/repositories/index", json={"repo_url": "https://127.0.0.1/private.git"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only public https Git URLs", response.text)

    def test_accepts_valid_repo_url_without_running_background_work(self) -> None:
        with patch.object(ingestion_main.service, "submit") as submit_mock, patch.object(ingestion_main.service, "run_job") as run_job_mock:
            submit_mock.return_value = ingestion_main.IndexRepositoryAccepted(
                job_id="job-1",
                repo_id="repo-1",
                status="queued",
                message="Repository queued for indexing. Repo ID: repo-1",
            )

            response = self.client.post("/v1/repositories/index", json={"repo_url": "https://github.com/psf/requests.git"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], "job-1")
        self.assertEqual(body["repo_id"], "repo-1")
        run_job_mock.assert_called_once_with("job-1")


if __name__ == "__main__":
    unittest.main()