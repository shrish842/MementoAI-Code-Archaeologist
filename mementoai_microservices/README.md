# MementoAI Microservices Scaffold

This scaffold splits the original two-file MVP into explicit service boundaries without changing the user-facing workflow:

1. Streamlit frontend submits repository indexing requests and query requests.
2. Gateway API preserves the public contract used by the frontend.
3. Ingestion service owns repo cloning, commit extraction, AST/debt enrichment, and vector indexing.
4. Query service owns question embeddings and vector retrieval.
5. Summary service owns Gemini-backed answer synthesis and fallback behavior.

## Why this split

The original [api.py](/C:/Users/agraw/Downloads/api.py:1) mixed API serving, Celery bootstrapping, model loading, Git subprocess calls, Pinecone access, and Gemini calls in one file. That creates hard failure coupling and makes scaling impossible without duplicating everything.

This scaffold keeps some helpers as in-process modules on purpose:

- AST analysis stays inside the ingestion service because it is CPU-local work on already-fetched diffs.
- Technical debt scoring stays inside the ingestion service for the same reason.
- The gateway remains thin and does not own domain logic.

## Target directory tree

```text
mementoai_microservices/
  AUDIT.md
  README.md
  docker-compose.yml
  requirements.txt
  shared/
    __init__.py
    config.py
    embedding.py
    jobs.py
    logging.py
    models.py
    vector_store.py
  services/
    frontend/
      app.py
    gateway/
      app/
        __init__.py
        clients.py
        main.py
    ingestion/
      app/
        __init__.py
        analysis.py
        git_history.py
        main.py
        service.py
    query/
      app/
        __init__.py
        main.py
        service.py
    summary/
      app/
        __init__.py
        main.py
        service.py
```

## Service contracts

### Gateway

- `POST /index_repository`
- `GET /job_status/{job_id}`
- `POST /query_repository`

This preserves the current frontend contract while allowing the backend to change underneath it.

### Ingestion service

- `POST /v1/repositories/index`
- `GET /v1/jobs/{job_id}`

### Query service

- `POST /v1/repositories/query`

### Summary service

- `POST /v1/summaries/repository-query`

## Local dev notes

- The shared vector store includes both an in-memory adapter and a Pinecone adapter.
- The in-memory adapter is only useful for single-process smoke tests.
- Real multi-service deployments should use Pinecone or another shared vector store.
- The summary service degrades gracefully when `GOOGLE_API_KEY` is not configured.

## What this scaffold is and is not

- It is a concrete starting point for extracting the monolith safely.
- It is not a full replacement for production infrastructure like a durable job database, queue workers, or rate limits.
- The audit in `AUDIT.md` explains the gaps that should be filled first.