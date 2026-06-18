from pathlib import Path
import unittest


class FrontendSecurityTests(unittest.TestCase):
    def test_frontend_does_not_use_unsafe_allow_html(self) -> None:
        frontend_path = Path(__file__).resolve().parents[1] / "services" / "frontend" / "app.py"
        source = frontend_path.read_text(encoding="utf-8")
        self.assertNotIn("unsafe_allow_html=True", source)


if __name__ == "__main__":
    unittest.main()