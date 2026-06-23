from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from .models import JobState


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._latest_job_by_repo: Dict[str, str] = {}
        self._lock = Lock()

    def create(self, job: JobState) -> JobState:
        with self._lock:
            self._jobs[job.job_id] = job
            self._latest_job_by_repo[job.repo_id] = job.job_id
            return job

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        indexed_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> JobState:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status  # type: ignore[assignment]
            if indexed_count is not None:
                job.indexed_count = indexed_count
            if error is not None:
                job.error = error
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)
        
    def get_latest_for_repo(self, repo_id: str) -> Optional[JobState]:
        with self._lock:
            job_id = self._latest_job_by_repo.get(repo_id)
            if not job_id:
                return None
            return self._jobs.get(job_id)    