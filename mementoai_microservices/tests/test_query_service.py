from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.models import QueryRepositoryRequest
from services.query.app.service import QueryService


class FakeEmbedder:
    def __init__(self) -> None:
        self.last_texts = None

    def embed_texts(self, texts):
        self.last_texts = texts
        return [[0.25, 0.75]]


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_namespace = None
        self.last_vector = None
        self.last_top_k = None

    def query(self, namespace, vector, top_k):
        self.last_namespace = namespace
        self.last_vector = vector
        self.last_top_k = top_k
        return [
            {
                "id": "abc123",
                "score": 0.91,
                "metadata": {
                    "message": "fix auth flow",
                    "author": "dev",
                    "timestamp": 1710000000,
                    "diff_snippet": "@@\n-def a():\n+def a():\n",
                    "function_changes": [],
                    "technical_debt": None,
                    "old_technical_debt": None,
                    "debt_delta": None,
                },
            }
        ]


class QueryServiceTests(unittest.TestCase):
    def test_query_uses_single_flattened_vector(self) -> None:
        service = QueryService.__new__(QueryService)
        service.embedder = FakeEmbedder()
        service.vector_store = FakeVectorStore()

        response = service.query(QueryRepositoryRequest(repo_id="repo-1", question="why did auth change"))

        self.assertEqual(response.status, "success")
        self.assertEqual(service.vector_store.last_namespace, "repo-1")
        self.assertEqual(service.vector_store.last_vector, [0.25, 0.75])
        self.assertEqual(len(response.relevant_commits), 1)


if __name__ == "__main__":
    unittest.main()