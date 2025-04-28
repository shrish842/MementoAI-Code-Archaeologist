# 🏛️ MementoAI: Codebase Archaeologist

**Unearthing Insights from Your Code's History with Natural Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add license badge -->
<!-- Add other badges if relevant (e.g., build status, code coverage) -->

---

**Problem:** Understanding the evolution of large, complex codebases is a major challenge for developers. Manually digging through thousands of Git commits to find the *why* behind a change is slow, tedious, and often frustrating. Critical historical context gets lost, slowing down debugging, onboarding, and feature development.

**Solution:** MementoAI acts as your intelligent assistant for navigating code history. Ask questions in plain English, and MementoAI uses semantic search powered by modern NLP techniques (Sentence Embeddings & Vector Databases) to instantly find the most relevant commit messages **and their associated code changes (diffs)** from the repository's history. This provides crucial context, including both the developer's narrative and implementation details, in seconds.

---

## ✨ Key Features

*   **💬 Natural Language Queries:** Ask questions like "Why was the session object changed?" instead of complex `git log` commands.
*   **🧠 Semantic Understanding:** Goes beyond simple keyword matching to grasp the *meaning* behind your query and commit messages.
*   **⚡ Fast Retrieval:** Uses efficient vector search (ChromaDB) to scan thousands of commits instantly based on message relevance.
*   **🎯 Relevant Results:** Pinpoints the most historically relevant commits based on semantic similarity.
*   **📊 View Code Changes:** Displays the actual `diff` for retrieved commits, showing exactly what code was modified alongside the commit message.
*   **🤖 AI Summarization (Optional):** Utilizes Google Gemini Pro to generate concise summaries based on both the commit messages and the code changes found (requires API key).
*   **🌐 Web Interface:** Simple and interactive UI built with Streamlit.
*   **🔧 Modular Design:** Easily adaptable to different repositories.

---

## 🚀 Live Demo / Screenshot

*(Ideally, replace this section with an animated GIF or a clear screenshot showing the app displaying commits AND code diffs)*

[![](./path/to/your/demo.gif)](./path/to/your/demo.gif) <!-- Example embedding a GIF -->
*Caption: MementoAI retrieving relevant commits and code changes for a query about the 'requests' library.*

**[Link to Live Demo (if deployed)]** <!-- Optional: Add link if you deploy it -->

---

## ⚙️ Technology Stack

*   **Backend & Core Logic:** Python 3
*   **NLP Embeddings:** `sentence-transformers` (using `all-MiniLM-L6-v2` model)
*   **Vector Database:** `ChromaDB` (for local persistent storage & similarity search)
*   **Web Framework:** `Streamlit`
*   **Git Interaction:** Python `subprocess` module (for `git log` and `git show`)
*   **LLM Integration:** `google-generativeai` library (using `gemini-1.0-pro` model)
*   **Data Handling:** `json`

---

## 🛠️ Setup & Installation

Follow these steps to set up and run MementoAI locally:

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git # Replace with your repo URL
    cd YourRepositoryName
    ```

2.  **Prerequisites:**
    *   **Python 3.9+:** Ensure Python is installed and added to your PATH.
    *   **Git:** Ensure Git is installed and accessible from your terminal.
    *   **(Windows Specific) Microsoft C++ Build Tools:** Required for compiling `chromadb` dependencies. Download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and install the "Desktop development with C++" workload. Restart your terminal after installation.

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes `streamlit`, `google-generativeai`, `sentence-transformers`, `chromadb`, etc. Create/update using `pip freeze > requirements.txt`)*

4.  **Download Target Repository Data:**
    *   MementoAI needs the commit history of the repository you want to analyze. For the demo, we use `psf/requests`.
    *   Clone the target repository locally. **Note the full path to this cloned repository.**
        ```bash
        # Example: Clone 'requests' inside the CodeBase_Archaelogist directory
        git clone https://github.com/psf/requests.git
        # Or clone it elsewhere and note its path
        ```
    *   **Important:** Update the `REPO_PATH` variable inside `extract_git_log.py` to point to the exact location of the *cloned target repository* (e.g., `./requests` or `C:/path/to/requests`).
    *   **Equally Important:** Update the `REPO_PATH_FOR_GIT_COMMANDS` variable inside `app.py` to point to the *same location* of the cloned target repository, as `app.py` now also runs `git show`.

5.  **Extract Commit History:**
    *   Run the extraction script. This will create `requests_commits.json` in the MementoAI project directory.
    ```bash
    python extract_git_log.py
    ```

6.  **Generate Embeddings & Build Database:**
    *   Run the embedding script. This will process `requests_commits.json`, generate embeddings based on commit messages, and save them into the `./chroma_db` directory. This can take several minutes.
    ```bash
    python embed_commits.py
    ```

7.  **(Optional but Recommended for Summaries) Configure Google Gemini API Key:**
    *   Get an API key from [Google AI Studio](https://aistudio.google.com/).
    *   The most secure way is via an environment variable:
        ```bash
        # Example (Linux/macOS):
        export GOOGLE_API_KEY='AIzaYourKeyHere...'
        # Example (Windows CMD):
        set GOOGLE_API_KEY=AIzaYourKeyHere...
        # Example (Windows PowerShell):
        $env:GOOGLE_API_KEY='AIzaYourKeyHere...'
        ```
    *   Alternatively, for quick testing, you can temporarily paste it into the `google_api_key_manual` variable in `app.py` **(BUT REMOVE BEFORE COMMITTING/SHARING!)**. If no key is configured, the summarization feature will be disabled.

---

## 💻 Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    *   This starts a local web server and opens the app in your browser (usually `http://localhost:8501`).

2.  **Ask Questions:**
    *   Type your question about the codebase's history into the text input box (e.g., "Tell me about authentication changes", "Fixes related to CVE-...", "Why was this dependency added?").
    *   Press Enter.

3.  **View Results:**
    *   MementoAI embeds your question, queries the database based on commit messages, and displays the top N most relevant results.
    *   For each result, you will see:
        *   The commit hash, author, date, and similarity score.
        *   The full commit message.
        *   An expandable section showing the actual **code changes (diff)** for that commit.
        *   A link to view the commit on GitHub.
    *   If a valid Google Gemini API key is configured, an AI-generated summary based on the retrieved messages and diffs will be displayed. Otherwise, an informational note appears.

---

## ✨ Future Work & Roadmap

MementoAI has exciting potential for expansion:

*   [ ] **Semantic Search on Code Diffs:** Index embeddings *of the code changes themselves* for more direct code-related queries.
*   [ ] **Integrate Issue Tracker Data:** Connect to GitHub/GitLab APIs to include context from issues and pull requests.
*   [ ] **IDE Plugin:** Develop extensions for VS Code, PyCharm, etc., to bring MementoAI insights directly into the developer workflow.
*   [ ] **Alternative LLMs:** Add support for other open-source or commercial LLMs for summarization.
*   [ ] **Advanced Filtering:** Add options to filter search results by author, date range, file path affected in the diff, etc.
*   [ ] **Cross-Repo Analysis:** Support querying across multiple related repositories.
*   [ ] **Incremental Indexing:** Implement efficient updates to the vector database as new commits arrive in the target repository.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you add one).

---
