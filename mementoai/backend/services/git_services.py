import os
import subprocess
import tempfile
from logging import Logger
from typing import List, Dict, Optional, Tuple

def get_git_diff(commit_hash: str, repo_path: str) -> str:
    """Fetches the code diff for a given commit hash."""
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
            return f"Error getting diff (Code {diff_result.returncode})"
    except FileNotFoundError: 
        return "Error: 'git' command not found."
    except subprocess.TimeoutExpired: 
        return f"Error: `git show` timed out for {commit_hash[:7]}"
    except Exception as e: 
        return f"Exception getting diff: {e}"

# Add this function to backend/services/git_services.py
def extract_code_states(commit_hash: str, repo_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the file name and before/after code states for a commit."""
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
            return None, None, None
        
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
            return None, None, None
        
        changed_files = files_result.stdout.strip().split('\n')
        if not changed_files:
            return None, None, None
        
        # Return the first changed file
        target_file = changed_files[0]
        
        # Check if file is binary
        file_cmd = ["git", "check-attr", "diff", target_file]
        file_result = subprocess.run(
            file_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if "binary" in file_result.stdout.lower():
            return target_file, None, None
            
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
        
        old_code = old_result.stdout if old_result.returncode == 0 else None
        new_code = new_result.stdout if new_result.returncode == 0 else None
        
        return target_file, old_code, new_code
    
    except Exception as e:
        logger.error(f"Error extracting code states: {e}")
        return None, None, None

def clone_repository(repo_url: str, tmpdir: str, depth: int) -> None:
    """Clone a repository into a temporary directory with a specified depth."""
    clone_command = [
        "git", "clone", 
        "--depth", str(depth), 
        "--no-single-branch", 
        repo_url, 
        tmpdir
    ]
    subprocess.run(
        clone_command, 
        check=True, 
        timeout=300, 
        capture_output=True, 
        text=True, 
        encoding='utf-8', 
        errors='replace'
    )

def extract_commit_history(repo_path: str, num_commits: int) -> List[Dict]:
    """Extract commit history from the repository."""
    log_format = "%H||%an||%at||%s%n%b-----COMMIT_END-----"
    log_command = ["git", "log", f"--pretty=format:{log_format}"]
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

    commits = []
    for entry in log_result.stdout.strip().split("-----COMMIT_END-----"):
        if not entry.strip(): 
            continue
        
        parts = entry.strip().split('||', 3)
        if len(parts) == 4:
            h, a, t, m_part = parts
            s_b = m_part.split('\n', 1)
            s = s_b[0].strip()
            b = s_b[1].strip() if len(s_b) > 1 else ""
            full_msg = f"{s}\n{b}".strip()
            
            if h and full_msg:
                commits.append({
                    "hash": h, 
                    "author": a, 
                    "timestamp": int(t) if t.isdigit() else 0,
                    "subject": s, 
                    "full_message": full_msg
                })
    return commits