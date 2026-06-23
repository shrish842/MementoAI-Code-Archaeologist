from __future__ import annotations

import time
import uuid
from typing import List

from shared.embedding import EmbeddingProvider, build_embedding_provider
from shared.jobs import InMemoryJobStore
from shared.logging import get_logger, log_event
from shared.models import IndexRepositoryAccepted, JobState
from shared.vector_store import VectorStore, build_vector_store

from .analysis import analyze_code_changes, analyze_technical_debt
from .git_history import (
    aggregate_changed_file_versions,
    clone_repository,
    get_commit_diff,
    iter_commit_snapshots,
)


class IngestionService:
    def __init__(self, config) -> None:
        self.config = config
        self.logger = get_logger(config.service_name)
        self.jobs = InMemoryJobStore()
        self.embedder: EmbeddingProvider = build_embedding_provider(config)
        self.vector_store: VectorStore = build_vector_store(config)
    
    def submit(self, repo_url: str) -> IndexRepositoryAccepted:
        repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
        existing_job = self.jobs.get_latest_for_repo(repo_id)
        if existing_job and existing_job.status in {"queued", "running", "completed"}:
            log_event(
                self.logger,
                "ingestion.job.deduplicated",
                existing_job_id=existing_job.job_id,
                repo_id=repo_id,
                existing_status=existing_job.status,
            )
            return IndexRepositoryAccepted(
                job_id=existing_job.job_id,
                repo_id=repo_id,
                status=existing_job.status,
                message=f"Repository already has an index job with status '{existing_job.status}'. Reusing existing job.",
                deduplicated=True,
            )

        job_id = str(uuid.uuid4())
        job = JobState(job_id=job_id, repo_id=repo_id, repo_url=repo_url, status="queued")
        self.jobs.create(job)
        log_event(self.logger, "ingestion.job.created", job_id=job_id, repo_id=repo_id)
        return IndexRepositoryAccepted(
            job_id=job_id,
            repo_id=repo_id,
            status="queued",
            message=f"Repository queued for indexing. Repo ID: {repo_id}",
            deduplicated=False,
        )
    
    
    def get_job(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    def run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        self.jobs.update(job_id, status="running")
        started = time.time()
        try:
            temp_repo = clone_repository(job.repo_url)
            try:
                commits = list(iter_commit_snapshots(temp_repo.name))
                embeddings = self.embedder.embed_texts([commit.message for commit in commits])
                records: List[dict] = []
                for index, commit in enumerate(commits):
                    diff_text = get_commit_diff(temp_repo.name, commit.commit_hash)
                    old_code, new_code = aggregate_changed_file_versions(temp_repo.name, commit.commit_hash)
                    old_debt = analyze_technical_debt(old_code or "")
                    new_debt = analyze_technical_debt(new_code or "")
                    debt_delta = new_debt.technical_debt_score - old_debt.technical_debt_score
                    records.append(
                        {
                            "id": commit.commit_hash,
                            "vector": embeddings[index],
                            "metadata": {
                                "message": commit.message,
                                "author": commit.author,
                                "timestamp": commit.timestamp,
                                "subject": commit.subject,
                                "diff_snippet": diff_text[:5000],
                                "function_changes": [change.model_dump() for change in analyze_code_changes(diff_text)],
                                "technical_debt": new_debt.model_dump(),
                                "old_technical_debt": old_debt.model_dump(),
                                "debt_delta": debt_delta,
                            },
                        }
                    )
                for offset in range(0, len(records), 100):
                    self.vector_store.upsert(job.repo_id, records[offset : offset + 100])
                self.jobs.update(job_id, status="completed", indexed_count=len(records))
                log_event(
                    self.logger,
                    "ingestion.job.completed",
                    job_id=job_id,
                    repo_id=job.repo_id,
                    indexed_count=len(records),
                    duration_seconds=round(time.time() - started, 2),
                )
            finally:
                temp_repo.cleanup()
        except Exception as exc:
            self.jobs.update(job_id, status="failed", error=str(exc))
            log_event(self.logger, "ingestion.job.failed", job_id=job_id, repo_id=job.repo_id, error=str(exc))
