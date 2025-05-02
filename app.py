import streamlit as st
import requests # Library to make HTTP requests
import time
import os

# --- Configuration ---
# URL of your *running* FastAPI backend API
# Use environment variable in production/deployment!
BACKEND_API_URL = os.environ.get("MEMENTO_API_URL", "http://127.0.0.1:8000/analyze")

# --- Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="MementoAI - Code Archaeologist")

# --- Streamlit App UI ---
st.title("🏛️ MementoAI: Codebase Archaeologist")
st.markdown("Analyze the **recent commit history** of any **public** Git repository using natural language.")


# --- Input Area ---
repo_url = st.text_input("Enter PUBLIC Git Repository URL (.git):", placeholder="e.g., https://github.com/psf/requests.git", key="repo_url_input")
user_question = st.text_input("Your question about the recent history:", placeholder="e.g., What changed recently regarding sessions?", key="question_input")
analyze_button = st.button("Analyze Recent History", key="analyze_button")

# Placeholder for results
results_placeholder = st.empty()

# --- Processing Logic ---
if analyze_button:
    if not repo_url or not user_question:
        st.warning("Please enter both a repository URL and a question.")
    elif not repo_url.endswith(".git"): # Basic validation
         st.error("Please enter a valid Git repository URL ending in .git")
    else:
        results_placeholder.empty() # Clear previous results/messages
        with st.spinner(f"Contacting MementoAI API... Analyzing {repo_url.split('/')[-1]}... Please wait."):
            try:
                # --- Call the Backend API ---
                payload = {"repo_url": repo_url, "question": user_question}
                response = requests.post(BACKEND_API_URL, json=payload, timeout=120) # Increased timeout

                # Check response status code
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                # Parse the JSON response from the API
                api_data = response.json()

                # --- Display Results ---
                with results_placeholder.container(): # Use container to replace spinner smoothly
                    st.subheader("Analysis Results (Recent History Only)")

                    if api_data.get("status") == "error":
                        st.error(f"Analysis failed on backend: {api_data.get('message', 'Unknown error')}")
                    else:
                        # Display warning message if analysis had partial success
                        if api_data.get("status") == "partial_success":
                             st.warning(f"Analysis completed with issues: {api_data.get('message', '')}")

                        # Display AI Summary
                        st.subheader("💡 AI Summary")
                        if api_data.get("ai_summary"):
                            st.markdown(api_data["ai_summary"])
                        else:
                            st.info("AI Summary not available or not generated.")

                        st.markdown("---")

                        # Display Relevant Commits
                        st.subheader("🔍 Relevant History Found")
                        commits = api_data.get("relevant_commits", [])
                        if commits:
                            st.write(f"Found {len(commits)} relevant commits:")
                            for i, commit in enumerate(commits):
                                st.markdown(f"**{i+1}. Commit:** `{commit['hash'][:7]}` (Similarity: {commit['similarity']:.4f})")
                                st.markdown(f"*Author: {commit['author']} | Date: {commit['date']}*")
                                st.text_area(f"Commit Message {i+1}", commit['message'], height=100, key=f"msg_{i}")
                                with st.expander(f"View Code Changes (Diff) for Commit {i+1}"):
                                     st.code(commit['diff'], language='diff', line_numbers=False)
                                # Generate GitHub/GitLab links if possible
                                if "github.com" in repo_url:
                                    base_url = repo_url.replace(".git", "") + "/commit/"
                                    st.markdown(f"[View on GitHub]({base_url}{commit['hash']})", unsafe_allow_html=True)
                                elif "gitlab.com" in repo_url:
                                    base_url = repo_url.replace(".git", "") + "/-/commit/"
                                    st.markdown(f"[View on GitLab]({base_url}{commit['hash']})", unsafe_allow_html=True)
                                st.markdown("---")
                        else:
                             st.info("No specific relevant commits were identified in the recent history.")

            except requests.exceptions.RequestException as e:
                 with results_placeholder.container():
                    st.error(f"Error contacting MementoAI API: {e}")
                    st.error("Please ensure the backend server (api.py) is running.")
            except Exception as e:
                 with results_placeholder.container():
                    st.error(f"An unexpected error occurred in the Streamlit app: {e}")


# Footer
st.markdown("---")
st.markdown("MementoAI | Hackathon Project")