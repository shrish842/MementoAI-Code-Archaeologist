# 🏛️ MementoAI: Codebase Archaeologist

**Unearthing Insights from Your Code's History with Natural Language.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add license badge -->
<!-- Add other badges if relevant (e.g., build status, code coverage) -->

---

**Problem:** Understanding the evolution of large, complex codebases is a major challenge for developers. Manually digging through thousands of Git commits to find the *why* behind a change is slow, tedious, and often frustrating. Critical historical context gets lost, slowing down debugging, onboarding, and feature development.

**Solution:** MementoAI acts as your intelligent assistant for navigating code history. Ask questions in plain English, and MementoAI uses semantic search powered by modern NLP techniques (Sentence Embeddings & Vector Databases) to instantly find the most relevant commit messages from the repository's history, providing crucial context in seconds.

---

## ✨ Key Features

*   **💬 Natural Language Queries:** Ask questions like "Why was the session object changed?" instead of complex `git log` commands.
*   **🧠 Semantic Understanding:** Goes beyond simple keyword matching to grasp the *meaning* behind your query and commit messages.
*   **⚡ Fast Retrieval:** Uses efficient vector search (ChromaDB) to scan thousands of commits instantly.
*   **🎯 Relevant Results:** Pinpoints the most historically relevant commits based on semantic similarity.
*   **🌐 Web Interface:** Simple and interactive UI built with Streamlit.
*   **🔧 Modular Design:** Easily adaptable to different repositories and potentially other data sources.

---

## 🚀 Live Demo / Screenshot

*(Ideally, replace this section with an animated GIF or a clear screenshot of the app in action)*

[![](./path/to/your/demo.gif)](./path/to/your/demo.gif) <!-- Example embedding a GIF -->
*Caption: Asking MementoAI about session refactoring in the 'requests' library.*

**[Link to Live Demo (if deployed)]** <!-- Optional: Add link if you deploy it -->

---

## ⚙️ Technology Stack

*   **Backend & Core Logic:** Python 3
*   **NLP Embeddings:** `sentence-transformers` (using `all-MiniLM-L6-v2` model)
*   **Vector Database:** `ChromaDB` (for local persistent storage & similarity search)
*   **Web Framework:** `Streamlit`
*   **Git Interaction:** Python `subprocess` (or `GitPython`)
*   **LLM Integration (Optional/Future):** `openai` library (Currently disabled in UI due to API quota limits, but backend code exists)

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
    *(Note: Make sure you have created a `requirements.txt` file using `pip freeze > requirements.txt` and potentially cleaned it up to include only necessary packages like `streamlit`, `openai`, `sentence-transformers`, `chromadb`)*

4.  **Download Target Repository Data:**
    *   MementoAI needs the commit history of the repository you want to analyze. For the demo, we use `psf/requests`.
    *   Clone the target repository *outside* the MementoAI project folder or ensure its path is correctly configured in `extract_git_log.py`.
        ```bash
        # Example: Clone 'requests' into the parent directory (adjust path as needed)
        # git clone https://github.com/psf/requests.git ../requests
        ```
    *   **Important:** Update the `REPO_PATH` variable inside `extract_git_log.py` to point to the exact location of the *cloned target repository* (e.g., `../requests` or an absolute path).

5.  **Extract Commit History:**
    *   Run the extraction script. This will create a `requests_commits.json` file (or similar, depending on config) in the MementoAI project directory.
    ```bash
    python extract_git_log.py
    ```

6.  **Generate Embeddings & Build Database:**
    *   Run the embedding script. This will process the `.json` file, generate embeddings, and save them into the `./chroma_db` directory. This step can take several minutes depending on the repository size and your hardware (CPU vs GPU).
    ```bash
    python embed_commits.py
    ```

7.  **(Optional) Configure OpenAI API Key:**
    *   The app includes code to potentially use the OpenAI API for answer summarization (currently disabled in the UI).
    *   If you want the underlying code to be ready, you can set your key. The most secure way is via an environment variable:
        ```bash
        # Example (Linux/macOS):
        export OPENAI_API_KEY='sk-YourKeyHere'
        # Example (Windows CMD):
        set OPENAI_API_KEY=sk-YourKeyHere
        # Example (Windows PowerShell):
        $env:OPENAI_API_KEY='sk-YourKeyHere'
        ```
    *   Alternatively, you can temporarily paste it into the `openai_api_key_manual` variable in `app.py` **(BUT REMOVE BEFORE COMMITTING/SHARING!)**.

---

## 💻 Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    *   This will start a local web server and should automatically open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Ask Questions:**
    *   In the web interface, type your question about the codebase's history into the text input box (e.g., "Tell me about authentication changes", "Fixes related to CVE-...", "Why was this dependency added?").
    *   Press Enter.

3.  **View Results:**
    *   MementoAI will embed your question, query the ChromaDB database, and display the top N most semantically relevant commit messages it found, along with author, date, and a link to the commit on GitHub.
    *   *(Note: The AI summarization section is currently disabled and will display an informational message).*

---

## ✨ Future Work & Roadmap

MementoAI has exciting potential for expansion:

*   [ ] **Enable LLM Summarization:** Integrate GPT (or alternative open-source LLMs) to provide concise, synthesized answers based on retrieved commits (pending API access/quota).
*   [ ] **Index Source Code:** Embed code chunks (functions, classes) to allow questions about specific code blocks ("Why was this function written this way?").
*   [ ] **Integrate Issue Tracker Data:** Connect to GitHub/GitLab APIs to include context from issues and pull requests.
*   [ ] **IDE Plugin:** Develop extensions for VS Code, PyCharm, etc., to bring MementoAI insights directly into the developer workflow.
*   [ ] **Advanced Filtering:** Add options to filter search results by author, date range, file path, etc.
*   [ ] **Cross-Repo Analysis:** Support querying across multiple related repositories.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you add one).

---

## 👥 Team / Author

*   **[Your Team Name / Your Name]**
*   Hackathon: [Name of Hackathon]
*   Members: [List Member Names - Optional]

---
