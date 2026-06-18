# How To Test This Architecture

This architecture should be tested in four layers, because the risky failures are different at each layer.

## 1. Unit tests: catch logic bugs inside a service

Target files:

- [services/ingestion/app/analysis.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/services/ingestion/app/analysis.py)
- [services/query/app/service.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/services/query/app/service.py)
- [services/summary/app/service.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/services/summary/app/service.py)

What to test:

- diff parsing on added, removed, and modified functions
- technical-debt scoring returns stable shapes
- query embedding is flattened to a single vector before vector-store lookup
- summary service falls back cleanly when Gemini is unavailable

## 2. Contract tests: keep the frontend and gateway in sync

Target files:

- [services/gateway/app/main.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/services/gateway/app/main.py)
- [services/frontend/app.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/services/frontend/app.py)
- [shared/models.py](/C:/Users/agraw/OneDrive/Documents/New project/mementoai_microservices/shared/models.py)

What to test:

- `POST /index_repository` returns `job_id` and `repo_id` as structured fields
- `GET /job_status/{job_id}` keeps stable status values
- `POST /query_repository` returns `relevant_commits` plus partial-success behavior when summary fails

This is the exact place where the original code drifted between [app.py](/C:/Users/agraw/Downloads/app.py:254) and [api.py](/C:/Users/agraw/Downloads/api.py:634).

## 3. Integration tests: prove the service boundaries work together

Target flow:

```text
frontend -> gateway -> ingestion/query/summary -> vector store
```

What to test:

- indexing a small fixture repo creates a completed job
- querying that repo returns at least one commit hit
- summary service failure does not fail the whole query

For local repeatability, run these against the in-memory vector store first. Then repeat against Pinecone before production.

## 4. Adversarial and reliability tests: catch production pain

These matter because the original MVP accepted public repo content and ran Git/networked dependencies directly.

What to test:

- malicious diff text does not get rendered as raw HTML
- invalid or suspicious repo URLs are rejected by ingestion
- slow or failing summary service returns partial success
- duplicate indexing requests for the same repo do not corrupt results once durable idempotency is added

## Commands

Run the architecture tests:

```powershell
python -m unittest discover -s mementoai_microservices\tests -p "test_*.py" -v
```

Run just the gateway contract tests:

```powershell
python -m unittest mementoai_microservices.tests.test_gateway_contract -v
```

## What these tests do not cover yet

- real Pinecone integration
- real Gemini integration
- real background worker durability
- fixture Git repositories spanning many changed files in one commit

Those should be the next layer once the local contract suite is stable.