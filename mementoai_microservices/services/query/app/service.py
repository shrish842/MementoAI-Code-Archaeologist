from __future__ import annotations

import time
from typing import List

from shared.embedding import build_embedding_provider
from shared.models import CommitHit, QueryRepositoryRequest, QueryRepositoryResponse
from shared.vector_store import build_vector_store


class QueryService:
    def __init__(self, config) -> None:
        self.embedder = build_embedding_provider(config)
        self.vector_store = build_vector_store(config)

    def query(self, request: QueryRepositoryRequest) -> QueryRepositoryResponse:
        if not request.question.strip():
            return QueryRepositoryResponse(status="error", message="Question cannot be empty.")

        vector = self.embedder.embed_texts([request.question])[0]
        matches = self.vector_store.query(request.repo_id, vector, top_k=5)
        commits: List[CommitHit] = []
        for match in matches:
            meta = match["metadata"]
            commit_date = None
            if meta.get("timestamp"):
                commit_date = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(meta["timestamp"]))
            commits.append(
                CommitHit(
                    hash=match["id"],
                    message=meta.get("message", ""),
                    author=meta.get("author"),
                    date=commit_date,
                    similarity=float(match["score"]),
                    diff=meta.get("diff_snippet", ""),
                    function_changes=meta.get("function_changes", []),
                    technical_debt=meta.get("technical_debt"),
                    old_technical_debt=meta.get("old_technical_debt"),
                    debt_delta=meta.get("debt_delta"),
                )
            )
        return QueryRepositoryResponse(status="success", message="Query complete.", relevant_commits=commits)