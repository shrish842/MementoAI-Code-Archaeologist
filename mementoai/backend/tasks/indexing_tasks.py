from celery import shared_task
import tempfile
import time
import json
import os
from typing import List, Dict, Optional, Tuple
import subprocess
import logging

# Import services and config
from backend.config import Config
from backend.services import git_services, analysis_services
from backend.services.embedding_services import generate_embeddings
from backend.services.pinecone_services import get_pinecone_index
from backend.celery_app import celery_app

# Set up logging
logger = logging.getLogger(__name__)

@celery_app.task(name="tasks.process_and_index_repository")
def process_and_index_repository_task(repo_url: str, repo_id_for_namespace: str):
    task_id = process_and_index_repository_task.request.id
    logger.info(f"CELERY TASK [{task_id}]: Starting indexing for {repo_url} into namespace {repo_id_for_namespace}")
    
    pinecone_index = get_pinecone_index()
    if not pinecone_index:
        message = "Pinecone index not available. Aborting indexing."
        logger.error(f"CELERY TASK [{task_id}] ERROR: {message}")
        return {"status": "error", "message": message, "indexed_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 1. Clone repository
            logger.info(f"CELERY TASK [{task_id}]: Cloning {repo_url} into {tmpdir}")
            git_services.clone_repository(repo_url, tmpdir, Config.NUM_COMMITS_TO_EXTRACT_FOR_INDEXING)

            # 2. Extracting Commits
            logger.info(f"CELERY TASK [{task_id}]: Extracting commits from {tmpdir}")
            commits_to_embed = git_services.extract_commit_history(tmpdir, Config.NUM_COMMITS_TO_EXTRACT_FOR_INDEXING)
            
            if not commits_to_embed:
                message = "No commit messages extracted."
                logger.warning(f"CELERY TASK [{task_id}] WARN: {message}")
                return {"status": "warning", "message": message, "indexed_count": 0}

            commit_messages_for_embedding = [commit['full_message'] for commit in commits_to_embed]

            # 3. Embedding Messages
            logger.info(f"CELERY TASK [{task_id}]: Embedding {len(commit_messages_for_embedding)} messages...")
            embeddings = generate_embeddings(commit_messages_for_embedding)

            # 4. Preparing for Pinecone Upsert with AST Analysis
            vectors_for_pinecone = []
            for i, commit_details in enumerate(commits_to_embed):
                logger.info(f"CELERY TASK [{task_id}]: Processing commit {i+1}/{len(commits_to_embed)} - {commit_details['hash'][:7]}")
                
                diff_text = git_services.get_git_diff(commit_details["hash"], tmpdir)
                
                # Extract code states and file type
                target_file, old_code, new_code = git_services.extract_code_states(commit_details["hash"], tmpdir)
                
                # Initialize analysis variables
                old_debt = None
                new_debt = None
                debt_delta = None
                code_analysis = analysis_services.CodeChangeAnalysis()
                
                # Only analyze Python files
                if target_file and target_file.endswith('.py'):
                    if old_code:
                        try:
                            old_debt = analysis_services.analyze_technical_debt(old_code)
                        except Exception as e:
                            logger.error(f"Error analyzing old technical debt: {e}")
                    if new_code:
                        try:
                            new_debt = analysis_services.analyze_technical_debt(new_code)
                        except Exception as e:
                            logger.error(f"Error analyzing new technical debt: {e}")
                    
                    if old_debt and new_debt:
                        debt_delta = new_debt.technical_debt_score - old_debt.technical_debt_score
                    
                    # Perform AST analysis on the diff
                    try:
                        code_analysis = analysis_services.analyze_code_changes(diff_text)
                    except Exception as e:
                        logger.error(f"Error in AST analysis: {e}")
                
                # Truncate diff for storage
                truncated_diff = diff_text[:Config.MAX_DIFF_CHARS_FOR_STORAGE] + \
                                 ("..." if len(diff_text) > Config.MAX_DIFF_CHARS_FOR_STORAGE else "")
                
                # Prepare metadata
                metadata = {
                    "message": commit_details["full_message"],
                    "author": commit_details["author"],
                    "timestamp": commit_details["timestamp"],
                    "subject": commit_details["subject"],
                    "diff_snippet": truncated_diff,
                    "functions_added": code_analysis.functions_added or [],
                    "functions_removed": code_analysis.functions_removed or [],
                    "functions_modified": code_analysis.functions_modified or [],
                    "complexity_changes": json.dumps(code_analysis.complexity_changes) if code_analysis.complexity_changes else "{}",
                    "debt_delta": debt_delta if debt_delta is not None else 0.0
                }
                
                # Add technical debt if available
                if new_debt:
                    try:
                        metadata["technical_debt"] = json.dumps(new_debt.dict())
                    except:
                        pass
                if old_debt:
                    try:
                        metadata["old_technical_debt"] = json.dumps(old_debt.dict())
                    except:
                        pass
                
                vectors_for_pinecone.append((
                    commit_details["hash"],       # ID
                    embeddings[i].tolist(),       # Vector
                    metadata
                ))

            logger.info(f"CELERY TASK [{task_id}]: Prepared {len(vectors_for_pinecone)} vectors with AST analysis.")

            # 5. Upsert to Pinecone in Batches
            logger.info(f"CELERY TASK [{task_id}]: Upserting to Pinecone namespace '{repo_id_for_namespace}'...")
            batch_size = 100 
            for i in range(0, len(vectors_for_pinecone), batch_size):
                batch = vectors_for_pinecone[i : i + batch_size]
                try:
                    pinecone_index.upsert(vectors=batch, namespace=repo_id_for_namespace)
                    logger.info(f"CELERY TASK [{task_id}]: Upserted batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
            
            final_count = len(vectors_for_pinecone)
            logger.info(f"CELERY TASK [{task_id}]: Upsert complete. Indexed {final_count} commits.")
            return {"status": "completed", "indexed_count": final_count, "message": "Indexing successful"}

        except subprocess.CalledProcessError as e:
            message = f"Git command failed during indexing: {e.cmd} - {e.stderr or e.stdout or 'No output'}"
            logger.error(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}
        except Exception as e:
            message = f"Error during indexing for {repo_url}: {str(e)}"
            logger.error(f"CELERY TASK [{task_id}] ERROR: {message}")
            return {"status": "error", "message": message, "indexed_count": 0}