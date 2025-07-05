from pydantic import BaseModel
from typing import Dict, List, Optional, ForwardRef

# Define TechnicalDebtMetrics first since it's referenced early
class TechnicalDebtMetrics(BaseModel):
    code_smells: List[str] = []
    cyclomatic_complexity: Dict[str, int] = {}
    maintainability_index: float = 0.0
    duplication: float = 0.0
    lines_of_code: int = 0
    technical_debt_score: float = 0.0

class CodeChangeAnalysis(BaseModel):
    functions_added: List[str] = []
    functions_removed: List[str] = []
    functions_modified: List[str] = []
    complexity_changes: Dict[str, int] = {}

class IndexRepoRequest(BaseModel): 
    repo_url: str

class IndexRepoResponse(BaseModel): 
    job_id: str
    status: str
    message: str

class FunctionChange(BaseModel):
    name: str
    change_type: str  # 'added', 'removed', 'modified'
    complexity_change: Optional[int] = None

# Create forward reference for CommitInfo if needed for recursive models
CommitInfoRef = ForwardRef('CommitInfo')

class CommitInfo(BaseModel): 
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
    repo_id: str
    question: str

class QueryRepoResponse(BaseModel): 
    status: str
    message: Optional[str] = None
    relevant_commits: List[CommitInfo] = []
    ai_summary: Optional[str] = None

class JobStatusResponse(BaseModel): 
    job_id: str
    status: str
    result: Optional[dict] = None

# Resolve forward references
CommitInfo.update_forward_refs()