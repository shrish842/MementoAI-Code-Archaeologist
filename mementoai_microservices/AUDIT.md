# Code-Only Audit for MementoAI

This audit is based only on:

- [app.py](/C:/Users/agraw/Downloads/app.py:1)
- [api.py](/C:/Users/agraw/Downloads/api.py:1)
- [README (2).md](</C:/Users/agraw/Downloads/README (2).md:1>)

No deployment or incident data was provided, so severity is judged as if this MVP is moving toward a hosted multi-user service.

## 1. Audit

### Real dependency graph

```text
app.py
  -> requests
  -> FastAPI backend `/index_repository`
  -> FastAPI backend `/job_status/{job_id}`
  -> FastAPI backend `/query_repository`
  -> brittle parsing of backend message text to recover repo namespace

api.py module import
  -> env vars / hardcoded secrets
  -> Celery app bootstrap
  -> Gemini client bootstrap
  -> SentenceTransformer model load
  -> Pinecone client bootstrap and possible index creation loop

POST /index_repository
  -> validators.url
  -> uuid5(repo_url)
  -> Celery delay()
  -> process_and_index_repository_task

process_and_index_repository_task
  -> tempfile.TemporaryDirectory
  -> subprocess git clone
  -> subprocess git log
  -> embedding_model.encode
  -> get_git_diff
  -> extract_code_states
  -> analyze_code_changes
  -> analyze_technical_debt
  -> pinecone_index.upsert

POST /query_repository
  -> embedding_model.encode
  -> pinecone_index.query
  -> CommitInfo assembly
  -> optional Gemini summary call
```

### Hidden coupling and implicit contracts

- `app.py:251-255` parses `Repo ID/Namespace` out of a free-form message string returned by `api.py:634-637`. That is a hidden API contract that will break on a wording change.
- `app.py:170-188` expects debt history and `avg_complexity`, but `api.py` never returns those fields in query responses.
- `app.py:379-410` expects `technical_debt` and `debt_delta`, but `api.py:712-720` does not populate those fields on `CommitInfo`.
- `api.py` relies on module-level globals for `embedding_model`, `pinecone_index`, `genai_configured`, and `celery_app`, so API and worker behavior depends on import order and process role.

### Single points of failure

- Import-time external initialization in `api.py:49-104` can stall or fail startup for the entire API process.
- Pinecone is a single shared dependency for both indexing and query in `api.py:75-104`, `api.py:475-477`, and `api.py:666-671`.
- Redis-backed Celery results are the only job-status store in `api.py:40-47` and `api.py:640-649`.
- Gemini summary generation is synchronous inside the request path at `api.py:733-777`.

### Missing failure handling and idempotency

- No retry/backoff for `git clone`, `git log`, `embedding_model.encode`, `pinecone_index.upsert`, `pinecone_index.query`, or Gemini generation.
- No deduplication or idempotency guard on `/index_repository`; repeated requests enqueue repeated full index jobs for the same repo.
- No checkpointing inside the indexing loop, so a failure after 4,900 commits loses the whole job.
- No timeout around Pinecone query or describe-index polling loop.

### No-test / no-observability paths

- No tests were provided for the frontend request flow, indexing task, query path, AST analysis, or technical debt scoring.
- Logging is `print(...)` everywhere in `api.py`; there are no structured logs, metrics, trace IDs, or per-job correlation fields.
- The frontend has no user-visible retry state, request metrics, or error taxonomy beyond raw exception text.

### Dead code and god objects

- [api.py](/C:/Users/agraw/Downloads/api.py:1) is the god-file. It owns config, infra bootstrap, data models, Git access, AST analysis, debt analysis, background tasks, HTTP API, and summary generation.
- `torch` is imported but unused at `api.py:5`.
- `Request` is imported but unused at `api.py:8`.
- `util` is imported but unused at `api.py:11`.
- `unified_diff` is imported but unused at `api.py:17`.
- `radon` is imported but unused directly at `api.py:18`.
- `MAX_DIFF_CHARS_FOR_LLM` is defined but unused at `api.py:34`.
- Several CSS classes in `app.py:26-83` are defined but never used by the rendered markup.

## 2. Findings

### P0

1. Hardcoded live secrets in source code.
   Evidence: `api.py:28` and `api.py:52`.
   Blast radius: full compromise of paid external services, cost exposure, and emergency key rotation across every deployment artifact that ever included this file.

2. Untrusted user input can trigger server-side `git clone` against arbitrary URLs.