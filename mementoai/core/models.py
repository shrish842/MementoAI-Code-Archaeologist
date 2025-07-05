from typing import Dict, List, Optional
from pydantic import BaseModel

class CodeChangeAnalysis(BaseModel):
    """
    Represents the results of AST-based code change analysis.
    """
    functions_added: List[str] = []
    functions_removed: List[str] = []
    functions_modified: List[str] = []
    complexity_changes: Dict[str, int] = {}

class TechnicalDebtMetrics(BaseModel):
    """
    Represents various technical debt metrics for a given code snippet.
    """
    code_smells: List[str] = []
    cyclomatic_complexity: Dict[str, int] = {}
    maintainability_index: float = 0.0
    duplication: float = 0.0
    lines_of_code: int = 0
    technical_debt_score: float = 0.0

class IndexRepoRequest(BaseModel):
    """
    Request model for indexing a Git repository.
    """
    repo_url: str

class IndexRepoResponse(BaseModel):
    """
    Response model for repository indexing job.
    """
    job_id: str
    status: str
    message: str

class FunctionChange(BaseModel):
    """
    Represents a specific function change detected in a commit.
    """
    name: str
    change_type: str  # 'added', 'removed', 'modified'
    complexity_change: Optional[int] = None

class CommitInfo(BaseModel):
    """
    Detailed information about a relevant commit.
    """
    hash: str
    message: str
    author: Optional[str] = None
    date: Optional[str] = None
    similarity: float
    diff: str
    function_changes: List[FunctionChange] = []
    technical_debt: Optional[TechnicalDebtMetrics] = None
    old_technical_debt: Optional[TechnicalDebtMetrics] = None
    debt_delta: Optional[float] = None

class QueryRepoRequest(BaseModel):
    """
    Request model for querying an indexed repository.
    """
    repo_id: str
    question: str

class QueryRepoResponse(BaseModel):
    """
    Response model for repository query results.
    """
    status: str
    message: Optional[str] = None
    relevant_commits: List[CommitInfo] = []
    ai_summary: Optional[str] = None

class JobStatusResponse(BaseModel):
    """
    Response model for checking the status of a background job.
    """
    job_id: str
    status: str
    result: Optional[dict] = None

