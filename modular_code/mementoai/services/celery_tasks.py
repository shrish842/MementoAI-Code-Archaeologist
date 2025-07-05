# mementoai/services/celery_tasks.py

import tempfile
import json
import time
import uuid
from typing import Dict, List

from celery.utils.log import get_task_logger
from celery_worker import celery_app # Import the Celery app instance
from config.settings import settings
from services.embedding_service import embedding_model
from services.pinecone_service import pinecone_index
from services.git_service import clone_repository, get_commit_log, get_git_diff, extract_code_states
from services.ast_analysis import analyze_code_changes
from services.technical_debt import analyze_technical_debt
from core.exceptions import GitOperationError
from core.models import TechnicalDebtMetrics # Import TechnicalDebtMetrics

@celery_app.task(name="tasks.process_and_index_repository")
def process_and_index_repository_task(repo_url: str, repo_id_for_namespace: str):
    """
    Celery task to clone a repository, extract commits, analyze them,
    and upsert the data into Pinecone.
    """
    start_time = time.time()
    task_id = process_and_index_repository_task.request.id
    print(f"CELERY TASK [{task_id}]: Starting indexing for {repo_url} into namespace {repo_id_for_namespace}")

    if not embedding_model or not pinecone_index:
        message = "Backend models or Pinecone not loaded. Aborting indexing."
        print(f"CELERY TASK [{task_id}] ERROR: {message}")
        return {"status": "error", "message": message, "indexed_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 1. Clone repository
            clone_repository(repo_url, tmpdir)

            # 2. Extracting Commits
            print(f"CELERY TASK [{task_id}]: Extracting commits from {tmpdir}")
            log_output = get_commit_log(tmpdir)

            commits_to_embed = []
            commit_messages_for_embedding = []

            for entry in log_output.strip().split("-----COMMIT_END-----"):
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
                        commits_to_embed.append({
                            "hash": h,
                            "author": a,
                            "timestamp": int(t) if t.isdigit() else 0,
                            "subject": s,
                            "full_message": full_msg
                        })
                        commit_messages_for_embedding.append(full_msg)

            if not commits_to_embed:
                message = "No commit messages extracted."
                print(f"CELERY TASK [{task_id}] WARN: {message}")
                return {"status": "warning", "message": message, "indexed_count": 0}

            # 3. Embedding Messages
            print(f"CELERY TASK [{task_id}]: Embedding {len(commit_messages_for_embedding)} messages...")
            embeddings = embedding_model.encode(
                commit_messages_for_embedding,
                batch_size=64,
                show_progress_bar=False
            )

            # 4. Preparing for Pinecone Upsert with AST Analysis
            vectors_for_pinecone = []
            for i, commit_details in enumerate(commits_to_embed):
                print(f"CELERY TASK [{task_id}]: Processing commit {i+1}/{len(commits_to_embed)} - {commit_details['hash'][:7]}")

                diff_text = get_git_diff(commit_details["hash"], tmpdir)
                old_code, new_code = extract_code_states(commit_details["hash"], tmpdir)

                # Ensure these always return a TechnicalDebtMetrics object, even if empty
                # If old_code/new_code is None, analyze_technical_debt will return an empty TechnicalDebtMetrics
                old_debt = analyze_technical_debt(old_code)
                new_debt = analyze_technical_debt(new_code)

                debt_delta = new_debt.technical_debt_score - old_debt.technical_debt_score

                truncated_diff = diff_text[:settings.MAX_DIFF_CHARS_FOR_STORAGE] + \
                                 ("..." if len(diff_text) > settings.MAX_DIFF_CHARS_FOR_STORAGE else "")

                # Perform AST analysis on the diff
                code_analysis = analyze_code_changes(diff_text)

                vectors_for_pinecone.append((
                    commit_details["hash"],       # ID
                    embeddings[i].tolist(),       # Vector
                    {                            # Metadata
                        "message": commit_details["full_message"],
                        "author": commit_details["author"],
                        "timestamp": commit_details["timestamp"],
                        "subject": commit_details["subject"],
                        "diff_snippet": truncated_diff,
                        # Ensure these are always lists, even if empty
                        "functions_added": code_analysis.functions_added or [],
                        "functions_removed": code_analysis.functions_removed or [],
                        "functions_modified": code_analysis.functions_modified or [],
                        # Ensure complexity_changes is always a JSON string representing a dict
                        "complexity_changes": json.dumps(code_analysis.complexity_changes or {}),
                        # Ensure technical_debt is always a JSON string representing a dict
                        "technical_debt": json.dumps(new_debt.dict()),
                        # Ensure old_technical_debt is always a JSON string representing a dict
                        "old_technical_debt": json.dumps(old_debt.dict()),
                        "debt_delta": debt_delta
                    }
                ))

            print(f"CELERY TASK [{task_id}]: Prepared {len(vectors_for_pinecone)} vectors with AST analysis.")

            # 5. Upsert to Pinecone in Batches
            print(f"CELERY TASK [{task_id}]: Upserting to Pinecone namespace '{repo_id_for_namespace}'...")
            batch_size = 100
            for i in range(0, len(vectors_for_pinecone), batch_size):
                batch = vectors_for_pinecone[i : i + batch_size]
                pinecone_index.upsert(vectors=batch, namespace=repo_id_for_namespace)
                print(f"CELERY TASK [{task_id}]: Upserted batch {i//batch_size + 1}")

            final_count = len(vectors_for_pinecone)
            print(f"CELERY TASK [{task_id}]: Upsert complete. Indexed {final_count} commits.")
            return {"status": "completed", "indexed_count": final_count, "message": "Indexing successful"}

        except GitOperationError as e:
            message = f"Git operation failed during indexing: {e}"
            print(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}
        except Exception as e:
            message = f"Error during indexing for {repo_url}: {e}"
            print(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}
    duration = time.time() - start_time
    logger.info(f"Indexed {final_count} commits in {duration:.2f}s")
    return {"status": "completed", "indexed_count": final_count} 