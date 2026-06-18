from __future__ import annotations

from shared.models import SummaryRequest, SummaryResponse


class SummaryService:
    def __init__(self, config) -> None:
        self.config = config

    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        if not request.commits:
            return SummaryResponse(status="partial_success", summary="No relevant commits found.", message="Skipped summary generation.")

        if not self.config.google_api_key:
            return SummaryResponse(
                status="partial_success",
                summary=self._fallback_summary(request),
                message="GOOGLE_API_KEY not configured; returned deterministic summary.",
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.google_api_key)
            model = genai.GenerativeModel(self.config.gemini_model_name)
            response = model.generate_content(self._build_prompt(request), request_options={"timeout": 30})
            summary_text = response.text if getattr(response, "parts", None) else self._fallback_summary(request)
            return SummaryResponse(status="success", summary=summary_text, message="Summary generated.")
        except Exception as exc:
            return SummaryResponse(
                status="partial_success",
                summary=self._fallback_summary(request),
                message=f"Summary model failed: {exc}",
            )

    def _build_prompt(self, request: SummaryRequest) -> str:
        commit_blocks = []
        for index, commit in enumerate(request.commits, start=1):
            changes = ", ".join(f"{change.change_type}:{change.name}" for change in commit.function_changes) or "no function-level changes detected"
            commit_blocks.append(
                f"Commit {index} ({commit.hash[:7]}):\n"
                f"Message: {commit.message}\n"
                f"Similarity: {commit.similarity:.3f}\n"
                f"Function changes: {changes}\n"
                f"Diff snippet:\n{commit.diff[:2000]}\n"
            )
        return (
            f"Repository ID: {request.repo_id}\n"
            f"User question: {request.question}\n\n"
            f"Commits:\n{chr(10).join(commit_blocks)}\n"
            "Answer the question using only the supplied commit evidence. Explain what changed and why it matters."
        )

    def _fallback_summary(self, request: SummaryRequest) -> str:
        lines = [f"Question: {request.question}", "Top commit evidence:"]
        for commit in request.commits[:3]:
            lines.append(f"- {commit.hash[:7]} {commit.message}")
        return "\n".join(lines)