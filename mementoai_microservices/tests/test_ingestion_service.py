from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEMENTO_EMBEDDINGS_MODE", "hash")

from services.ingestion.app.service import IngestionService


class DummyConfig:
    service_name = "ingestion-test"
    pinecone_api_key = None
    pinecone_index_name = "mementoai"
    embedding_model_name = "all-MiniLM-L6-v2"
    embedding_dimension = 384
    gemini_model_name = "gemini-1.5-flash-latest"
    google_api_key = None


class IngestionServiceIdempotencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = IngestionService(DummyConfig())

    def test_submit_reuses_existing_job_for_same_repo(self) -> None:
        first = self.service.submit("https://github.com/psf/requests.git")
        second = self.service.submit("https://github.com/psf/requests.git")

        self.assertEqual(first.repo_id, second.repo_id)
        self.assertEqual(first.job_id, second.job_id)
        self.assertFalse(first.deduplicated)
        self.assertTrue(second.deduplicated)

    def test_submit_allows_new_job_after_failure(self) -> None:
        first = self.service.submit("https://github.com/psf/requests.git")
        self.service.jobs.update(first.job_id, status="failed", error="clone timeout")

        second = self.service.submit("https://github.com/psf/requests.git")

        self.assertNotEqual(first.job_id, second.job_id)
        self.assertFalse(second.deduplicated)


if __name__ == "__main__":
    unittest.main()