from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ingestion.app.git_history import aggregate_changed_file_versions, validate_public_git_url


@unittest.skipIf(shutil.which("git") is None, "git is required for git-history integration tests")
class GitHistoryTests(unittest.TestCase):
    def test_aggregate_changed_file_versions_includes_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir:
            repo_path = Path(repo_dir)
            self._run_git(["init"], repo_path)
            self._run_git(["config", "user.name", "Test User"], repo_path)
            self._run_git(["config", "user.email", "test@example.com"], repo_path)

            first = repo_path / "first.py"
            second = repo_path / "second.py"
            first.write_text("def alpha():\n    return 'a'\n", encoding="utf-8")
            second.write_text("def beta():\n    return 'b'\n", encoding="utf-8")
            self._run_git(["add", "."], repo_path)
            self._run_git(["commit", "-m", "initial"], repo_path)

            first.write_text("def alpha():\n    if True:\n        return 'aa'\n    return 'a'\n", encoding="utf-8")
            second.write_text("def beta():\n    return 'bb'\n", encoding="utf-8")
            self._run_git(["add", "."], repo_path)
            self._run_git(["commit", "-m", "modify two files"], repo_path)

            commit_hash = self._run_git(["rev-parse", "HEAD"], repo_path).strip()
            old_code, new_code = aggregate_changed_file_versions(str(repo_path), commit_hash)

            self.assertIsNotNone(old_code)
            self.assertIsNotNone(new_code)
            self.assertIn("# file: first.py", old_code)
            self.assertIn("# file: second.py", old_code)
            self.assertIn("def alpha():", new_code)
            self.assertIn("def beta():", new_code)

    def test_validate_public_git_url_rejects_suspicious_hosts(self) -> None:
        self.assertTrue(validate_public_git_url("https://github.com/psf/requests.git"))
        self.assertFalse(validate_public_git_url("http://github.com/psf/requests.git"))
        self.assertFalse(validate_public_git_url("https://localhost/repo.git"))
        self.assertFalse(validate_public_git_url("https://127.0.0.1/repo.git"))
        self.assertFalse(validate_public_git_url("https://169.254.169.254/repo.git"))
        self.assertFalse(validate_public_git_url("https://github.com/psf/requests.git?token=abc"))
        self.assertFalse(validate_public_git_url("ssh://github.com/psf/requests.git"))

    def _run_git(self, args: list[str], repo_path: Path) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()