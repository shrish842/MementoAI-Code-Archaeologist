from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "completed", "failed"]


class IndexRepositoryRequest(BaseModel):
    repo_url: str


class IndexRepositoryAccepted(BaseModel):
    job_id: str
    repo_id: str
    status: JobStatus
    message: str


class JobState(BaseModel):
    job_id: str
    repo_id: str
    repo_url: str
    status: JobStatus
    indexed_count: int = 0
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    repo_id: Optional[str] = None
    result: Optional[dict] = None


class FunctionChange(BaseModel):
    name: str
    change_type: Literal["added", "removed", "modified"]
    complexity_change: Optional[int] = None


class TechnicalDebtSnapshot(BaseModel):
    code_smells: List[str] = Field(default_factory=list)
    cyclomatic_complexity: Dict[str, int] = Field(default_factory=dict)
    maintainability_index: float = 0.0
    duplication: float = 0.0
    lines_of_code: int = 0
    technical_debt_score: float = 0.0
    avg_complexity: float = 0.0


class CommitHit(BaseModel):
    hash: str
    message: str
    author: Optional[str] = None
    date: Optional[str] = None
    similarity: float
    diff: str = ""
    function_changes: List[FunctionChange] = Field(default_factory=list)
    technical_debt: Optional[TechnicalDebtSnapshot] = None
    old_technical_debt: Optional[TechnicalDebtSnapshot] = None
    debt_delta: Optional[float] = None


class QueryRepositoryRequest(BaseModel):
    repo_id: str
    question: str


class QueryRepositoryResponse(BaseModel):
    status: Literal["success", "partial_success", "error"]
    message: Optional[str] = None
    relevant_commits: List[CommitHit] = Field(default_factory=list)
    ai_summary: Optional[str] = None


class SummaryRequest(BaseModel):
    repo_id: str
    question: str
    commits: List[CommitHit] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    status: Literal["success", "partial_success", "error"]
    summary: str
    message: Optional[str] = None