from __future__ import annotations

import ipaddress
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urlparse


TEXT_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".md",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class CommitSnapshot:
    commit_hash: str
    author: str
    timestamp: int
    subject: str
    message: str


def clone_repository(repo_url: str) -> tempfile.TemporaryDirectory[str]:
    tmpdir = tempfile.TemporaryDirectory()
    subprocess.run(
        ["git", "clone", "--no-single-branch", repo_url, tmpdir.name],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return tmpdir


def iter_commit_snapshots(repo_path: str, limit: int = 250) -> Iterator[CommitSnapshot]:
    log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
    result = subprocess.run(
        ["git", "log", f"--pretty=format:{log_format}", f"-n{limit}"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    for entry in result.stdout.split("-----COMMIT_END-----"):
        if not entry.strip():
            continue
        parts = entry.strip().split("||", 3)
        if len(parts) != 4:
            continue
        commit_hash, author, timestamp_text, body = parts
        lines = body.splitlines()
        subject = lines[0].strip() if lines else ""
        message = "\n".join(lines).strip()
        yield CommitSnapshot(
            commit_hash=commit_hash,
            author=author,
            timestamp=int(timestamp_text) if timestamp_text.isdigit() else 0,
            subject=subject,
            message=message,
        )


def get_commit_diff(repo_path: str, commit_hash: str) -> str:
    result = subprocess.run(
        ["git", "show", "--patch", "--pretty=format:", commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git show failed for {commit_hash}")
    return result.stdout.strip()

def get_changed_file_versions(repo_path: str, commit_hash: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    parent_hash = _get_parent_hash(repo_path, commit_hash)
    if not parent_hash:
        return []

    versions: List[Tuple[str, Optional[str], Optional[str]]] = []
    for target_file in _list_changed_files(repo_path, commit_hash):
        if not _is_supported_text_file(target_file):
            continue

        old_code = _git_show_file(repo_path, f"{parent_hash}:{target_file}")
        new_code = _git_show_file(repo_path, f"{commit_hash}:{target_file}")
        if old_code is None and new_code is None:
            continue
        versions.append((target_file, old_code, new_code))

    return versions


def aggregate_changed_file_versions(repo_path: str, commit_hash: str) -> Tuple[Optional[str], Optional[str]]:
    versions = get_changed_file_versions(repo_path, commit_hash)
    if not versions:
        return None, None

    old_parts: List[str] = []
    new_parts: List[str] = []
    for path, old_code, new_code in versions:
        header = f"# file: {path}"
        if old_code is not None:
            old_parts.append(header)
            old_parts.append(old_code)
        if new_code is not None:
            new_parts.append(header)
            new_parts.append(new_code)

    old_combined = "\n\n".join(old_parts).strip() or None
    new_combined = "\n\n".join(new_parts).strip() or None
    return old_combined, new_combined


def validate_public_git_url(repo_url: str) -> bool:
    try:
        parsed = urlparse(repo_url)
    except Exception:
        return False

    if parsed.scheme != "https" or not parsed.netloc or not parsed.path.endswith(".git"):
        return False
    if parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    if hostname in {"localhost", "0.0.0.0"}:
        return False

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False
    except ValueError:
        pass

    return True


def _get_parent_hash(repo_path: str, commit_hash: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit_hash}^"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _list_changed_files(repo_path: str, commit_hash: str) -> List[str]:
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return []
    return [path for path in result.stdout.splitlines() if path.strip()]


def _git_show_file(repo_path: str, object_path: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "show", object_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _is_supported_text_file(path: str) -> bool:
    lower_path = path.lower()
    return any(lower_path.endswith(suffix) for suffix in TEXT_SOURCE_SUFFIXES)