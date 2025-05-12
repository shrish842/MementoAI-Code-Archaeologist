# 🏛️ MementoAI: Codebase Archaeologist (Scalable Edition)

**Unearthing Deep Insights from Your Code's History with Natural Language, AI, and a Scalable Backend.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other relevant badges -->

---

**Problem:** Developers waste critical time manually deciphering complex Git histories to understand the 'why' and 'what' behind code changes. Traditional tools offer limited keyword search, failing to grasp semantic intent or provide comprehensive context (like associated code diffs), hindering debugging, onboarding, and effective maintenance.

**Solution:** MementoAI is an intelligent platform for navigating code history. Users can submit any public Git repository for analysis. MementoAI's backend **asynchronously clones the repository, extracts commit history, generates semantic embeddings for messages, fetches code diffs, and indexes this rich data into Pinecone.** Users can then ask natural language questions via a Streamlit frontend. The system performs a semantic search on the indexed messages, retrieves relevant commits with their diffs, and utilizes the Google Gemini API in a Retrieval-Augmented Generation (RAG) workflow to provide concise, context-aware summaries based on both the developer's narrative and the actual code changes.

---

## ✨ Key Features

*   **🔗 Analyze Any Public Repo:** Submit a Git URL for on-demand, asynchronous indexing.
*   **💬 Natural Language Queries:** Ask questions like "Why was the session object changed?"
*   **🧠 Semantic Understanding:** Finds relevant commits by *meaning* (via Sentence Transformers), not just keywords.
*   **📊 Rich Context:** Retrieves commit messages, author/date, similarity scores, AND the **full code diffs**.
*   **🤖 AI Summarization (Gemini):** Generates concise summaries synthesizing insights from both commit messages and code changes.
*   **☁️ Scalable Backend:** Built with FastAPI, Celery (for background tasks), and Pinecone (for a managed, scalable vector database).
*   **🌐 Interactive UI:** Streamlit frontend for easy interaction, job submission, and result viewing.
*   **⚙️ Asynchronous Indexing:** Long-running repository processing happens in the background without blocking the user.

---

## 🚀 Demo & Workflow

**1. Index Repository:** User submits a public Git URL. A background job is queued.
**2. Check Status:** User monitors the indexing job status.
**3. Query:** Once indexed, user asks natural language questions about the repository's history.
**4. Get Insights:** MementoAI displays relevant commits (messages + diffs) and an AI-generated summary.

**Video Demonstration:**
[![MementoAI Demo Video Thumbnail](./assets/loom_thumbnail.png)](https://www.loom.com/share/7e53cd79f26a44469cceb3da4e1994b8?sid=f93f02a6-3630-436b-9500-4bed8818aeb6)
*Caption: Watch MementoAI index a repository and answer historical queries.*

*(Ensure you have a `loom_thumbnail.png` in an `assets` folder or update the path/link)*

---

## ⚙️ Technology Stack

*   **Backend API:** Python 3, FastAPI
*   **Frontend UI:** Streamlit
*   **Asynchronous Tasks:** Celery
*   **Message Broker (for Celery):** Redis
*   **NLP Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2` model)
*   **Vector Database:** `Pinecone` (Managed, Serverless)
*   **LLM Integration:** `google-generativeai` library (`gemini-1.5-flash-latest` model)
*   **Git Interaction:** Python `subprocess` module (`git clone`, `git log`, `git show`)
*   **Data Validation:** Pydantic
*   **API Server:** Uvicorn
*   **URL Validation:** `validators`
*   **Data Handling:** `json`

---

## 🛠️ Setup & Installation (Local Development)

MementoAI now has a separated frontend, backend API, and a background worker.

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git # Replace with your repo URL
    cd YourRepositoryName
    ```

2.  **Prerequisites:**
    *   **Python 3.9+:** Ensure Python is installed and added to your PATH.
    *   **Git:** Ensure Git is installed and accessible from your terminal.
    *   **Redis:** Install and run a Redis server. Easiest with Docker: `docker run -d -p 6379:6379 redis`
    *   **(Windows Specific) Microsoft C++ Build Tools:** May be required for some Python package dependencies if they compile C/C++ code. Download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and install the "Desktop development with C++" workload.

3.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    # source .venv/bin/activate
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` is up-to-date with `fastapi`, `uvicorn`, `celery`, `redis`, `pinecone-client`, `google-generativeai`, `sentence-transformers`, `streamlit`, `validators`, etc.)*

5.  **Configure API Keys & Environment Variables:**
    *   Create a `.env` file in the project root (this file is listed in `.gitignore` and should NOT be committed).
    *   Add your API keys to the `.env` file:
        ```env
        # .env
        PINECONE_API_KEY="YOUR_ACTUAL_PINECONE_KEY"
        GOOGLE_API_KEY="AIzaYourActualGoogleApiKey..."
        # Optional: Override Celery broker/backend if not using default localhost Redis
        # CELERY_BROKER_URL="redis://other_host:6379/0"
        # CELERY_RESULT_BACKEND="redis://other_host:6379/0"
        ```
    *   The `app/core/config.py` file is set up to load these environment variables.

6.  **Pinecone Index Setup:**
    *   Ensure you have a Pinecone account and have created an index named `mementoai` (or update `PINECONE_INDEX_NAME` in `app/core/config.py`).
    *   The index **must** have `384` dimensions and use `cosine` similarity metric.
    *   The `api.py` script will attempt to create the index if it doesn't exist, but it's better to ensure it's ready in your Pinecone console.

---

## 💻 Running the Application (Local Development)

You'll need to run three separate processes, typically in three different terminal windows (make sure your virtual environment is activated in each).

1.  **Start Redis Server:** (If not already running via Docker or as a service).
    *   If installed locally (e.g., on Windows): navigate to Redis installation dir and run `redis-server.exe`.

2.  **Start Celery Worker:**
    *   Open Terminal 1 (activate venv).
    *   Run:
        ```bash
        celery -A app.tasks.celery_tasks.celery_app_instance worker -l INFO --pool=solo
        ```
    *   This worker will pick up indexing tasks.

3.  **Start FastAPI Backend API Server:**
    *   Open Terminal 2 (activate venv).
    *   Run:
        ```bash
        uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dirs app
        ```
    *   This serves the backend API. Check for any startup errors related to API keys or model loading.

4.  **Start Streamlit Frontend UI:**
    *   Open Terminal 3 (activate venv).
    *   Run:
        ```bash
        streamlit run streamlit_app.py
        ```
    *   This opens the web interface in your browser (usually `http://localhost:8501`).

**Using the Application:**

1.  In the Streamlit UI, go to "1. Index New Repository".
2.  Enter a public Git repository URL (ending in `.git`).
3.  Click "Start Indexing Repository". Note the `Job ID` and `Repo ID/Namespace` provided.
4.  Go to the "Check Indexing Job Status" section and click "Refresh Indexing Status" periodically. Wait for the status to become `SUCCESS` or `completed` (indexing can take several minutes).
5.  Once indexing is complete, go to "2. Query an Indexed Repository".
6.  Enter the `Repo ID/Namespace` from the indexing step.
7.  Type your question about the repository's history and click "Ask MementoAI".
8.  View the retrieved commits (with messages and diffs) and the AI-generated summary.

---

## ✨ Future Work & Roadmap

MementoAI is poised for significant enhancements to become a comprehensive code intelligence platform:

*   [ ] **Private Repository Support:** Securely handle authentication (OAuth) for GitHub, GitLab, etc.
*   [ ] **Advanced Diff Analysis & Search:** Semantically index code *within* diffs, enabling queries directly on code changes.
*   [ ] **Issue Tracker Integration:** Link commits to issues/PRs from platforms like Jira, GitHub Issues.
*   [ ] **IDE & Platform Plugins:** Bring MementoAI insights directly into VS Code, JetBrains IDEs, and GitHub/GitLab UIs.
*   [ ] **Fine-tuned LLMs:** Train custom LLMs specifically for code history summarization and Q&A.
*   [ ] **Robust Incremental Indexing:** Efficiently update Pinecone with new commits for active repositories.
*   [ ] **Team & Collaboration Features:** Shared repositories, access controls for teams.
*   [ ] **UI/UX Polish:** Enhance visualizations and user experience.
*   [ ] **Scalable Cloud Deployment:** Full cloud-native deployment architecture for the backend and workers.

---

## 📄 License

This project is licensed under the MIT License. (Consider adding a `LICENSE` file with the MIT license text).

---

## 👥 Author / Startup Idea

*   **Shish Agrawal**
*   *This project represents the foundational concept for a startup aiming to revolutionize how developers interact with and understand code history.*

---
