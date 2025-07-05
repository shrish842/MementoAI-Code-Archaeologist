import subprocess
import os
from typing import Optional, Tuple
from config.settings import settings
from core.exceptions import GitOperationError

def get_git_diff(commit_hash: str, repo_path: str) -> str:
    """
    Fetches the code diff for a given commit hash.
    """
    if not commit_hash or not repo_path or not os.path.isdir(repo_path) or not os.path.isdir(os.path.join(repo_path, '.git')):
        print(f"WARN: Invalid input for get_git_diff: hash={commit_hash}, path={repo_path}")
        return f"Error: Invalid repo path or commit hash for diff: {repo_path}"

    try:
        diff_command = ["git", "show", "--patch", "--pretty=format:", commit_hash]
        diff_result = subprocess.run(
            diff_command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30
        )

        if diff_result.returncode == 0:
            diff_text = diff_result.stdout.strip()
            return diff_text if diff_text else "(No code changes detected)"
        else:
            error_message = diff_result.stderr.strip()
            print(f"ERROR getting diff for {commit_hash[:7]}: {error_message}")
            raise GitOperationError(f"Git show failed for {commit_hash[:7]}: {error_message}")
    except FileNotFoundError:
        raise GitOperationError("Error: 'git' command not found. Please ensure Git is installed and in your PATH.")
    except subprocess.TimeoutExpired:
        raise GitOperationError(f"Error: `git show` timed out for {commit_hash[:7]}")
    except Exception as e:
        raise GitOperationError(f"Exception getting diff: {e}")

def clone_repository(repo_url: str, target_path: str):
    """
    Clones a Git repository into a specified target path.
    """
    print(f"Cloning {repo_url} into {target_path}")
    clone_command = [
        "git", "clone",
        "--depth", str(settings.NUM_COMMITS_TO_EXTRACT_FOR_INDEXING),
        "--no-single-branch",
        repo_url,
        target_path
    ]
    try:
        subprocess.run(
            clone_command,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    except subprocess.CalledProcessError as e:
        raise GitOperationError(f"Git clone failed: {e.cmd} - {e.stderr or e.stdout or 'No output'}")
    except subprocess.TimeoutExpired:
        raise GitOperationError(f"Git clone timed out for {repo_url}")
    except Exception as e:
        raise GitOperationError(f"Error during git clone: {e}")

def get_commit_log(repo_path: str) -> str:
    """
    Extracts commit log from a repository.
    """
    log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
    log_command = ["git", "log", f"--pretty=format:{log_format}"]
    try:
        log_result = subprocess.run(
            log_command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace',
            timeout=120
        )
        return log_result.stdout
    except subprocess.CalledProcessError as e:
        raise GitOperationError(f"Git log failed: {e.cmd} - {e.stderr or e.stdout or 'No output'}")
    except subprocess.TimeoutExpired:
        raise GitOperationError(f"Git log timed out for {repo_path}")
    except Exception as e:
        raise GitOperationError(f"Error during git log: {e}")

def extract_code_states(commit_hash: str, repo_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the before and after code states for a commit for the first *Python* changed file.
    Returns (old_code, new_code) for the first Python file found, or (None, None) if none.
    """
    try:
        # Get the parent commit
        parent_cmd = ["git", "rev-parse", f"{commit_hash}^"]
        parent_result = subprocess.run(
            parent_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if parent_result.returncode != 0:
            # This might happen for the very first commit (initial commit)
            # In this case, there's no "old" state, only "new"
            parent_hash = None # Indicate no parent
        else:
            parent_hash = parent_result.stdout.strip()

        # Get changed files in the commit
        files_cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
        files_result = subprocess.run(
            files_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if files_result.returncode != 0:
            print(f"Warning: Could not get changed files for commit {commit_hash[:7]}: {files_result.stderr.strip()}")
            return None, None

        changed_files = files_result.stdout.strip().split('\n')
        
        # Find the first Python file
        target_file = None
        for f in changed_files:
            if f.endswith('.py'):
                target_file = f
                break
        
        if not target_file:
            # No Python files changed in this commit
            return None, None

        old_code = None
        if parent_hash:
            # Get old version (parent)
            old_cmd = ["git", "show", f"{parent_hash}:{target_file}"]
            old_result = subprocess.run(
                old_cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            old_code = old_result.stdout if old_result.returncode == 0 else None
            if old_result.returncode != 0:
                print(f"Warning: Could not get old code for {target_file} in commit {commit_hash[:7]}: {old_result.stderr.strip()}")


        # Get new version (current commit)
        new_cmd = ["git", "show", f"{commit_hash}:{target_file}"]
        new_result = subprocess.run(
            new_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        new_code = new_result.stdout if new_result.returncode == 0 else None
        if new_result.returncode != 0:
            print(f"Warning: Could not get new code for {target_file} in commit {commit_hash[:7]}: {new_result.stderr.strip()}")

        return old_code, new_code

    except Exception as e:
        print(f"Error extracting code states: {e}")
        return None, None