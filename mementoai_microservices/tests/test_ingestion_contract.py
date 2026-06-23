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
        response = self.client.post("/v1/repositories/index", json={"repo_url": "http://github.com/psf/requests.git"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only public https Git URLs", response.text)

    def test_duplicate_index_request_reuses_existing_job(self) -> None:
        with patch.object(ingestion_main.service, "run_job") as run_job_mock:
            first = self.client.post("/v1/repositories/index", json={"repo_url": "https://github.com/psf/requests.git"})
            second = self.client.post("/v1/repositories/index", json={"repo_url": "https://github.com/psf/requests.git"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)

        first_body = first.json()
        second_body = second.json()
        self.assertEqual(first_body["job_id"], second_body["job_id"])
        self.assertFalse(first_body["deduplicated"])
        self.assertTrue(second_body["deduplicated"])
        run_job_mock.assert_called_once_with(first_body["job_id"])


if __name__ == "__main__":
    unittest.main()