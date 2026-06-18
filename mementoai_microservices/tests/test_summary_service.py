from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.models import CommitHit, SummaryRequest
from services.summary.app.service import SummaryService


class ConfigWithoutGemini:
    google_api_key = None
    gemini_model_name = "gemini-1.5-flash-latest"


class SummaryServiceTests(unittest.TestCase):
    def test_summary_falls_back_when_gemini_is_not_configured(self) -> None:
        service = SummaryService(ConfigWithoutGemini())
        request = SummaryRequest(
            repo_id="repo-1",
            question="why did auth change",
            commits=[
                CommitHit(hash="abcdef1", message="fix auth flow", similarity=0.91),
                CommitHit(hash="abcdef2", message="add retries", similarity=0.88),
            ],
        )

        response = service.summarize(request)

        self.assertEqual(response.status, "partial_success")
        self.assertIn("Question: why did auth change", response.summary)
        self.assertIn("abcdef1", response.summary)


if __name__ == "__main__":
    unittest.main()