import streamlit as st
import requests
import time
import os

# --- Configuration ---
BACKEND_API_BASE_URL = os.environ.get("MEMENTO_API_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="wide", page_title="MementoAI - Code Archaeologist")
st.title("🏛️ MementoAI: Codebase Archaeologist")
st.markdown("Analyze the commit history of any **public** Git repository.")

# --- Session State ---
if 'indexing_job_id' not in st.session_state: st.session_state.indexing_job_id = None
if 'current_repo_id_for_query' not in st.session_state: st.session_state.current_repo_id_for_query = None
if 'current_repo_url_display' not in st.session_state: st.session_state.current_repo_url_display = None
if 'last_job_status' not in st.session_state: st.session_state.last_job_status = None


# --- UI for Indexing ---
with st.expander("1. Index New Repository (Run this first!)", expanded=True):
    repo_url_to_index = st.text_input("Enter PUBLIC Git Repository URL (.git) to Index:",
                                      placeholder="e.g., https://github.com/psf/requests.git",
                                      key="repo_url_index_input")
    index_button = st.button("Start Indexing Repository", key="index_button")

    if index_button and repo_url_to_index:
        if not repo_url_to_index.endswith(".git"):
            st.error("Please enter a valid Git repository URL ending in .git")
        else:
            with st.spinner(f"Requesting indexing for {repo_url_to_index}..."):
                try:
                    response = requests.post(f"{BACKEND_API_BASE_URL}/index_repository", json={"repo_url": repo_url_to_index}, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    st.session_state.indexing_job_id = data.get("job_id")
                    # Extract namespace from message (example specific, make robust if needed)
                    msg = data.get("message", "")
                    if "Namespace: " in msg:
                        st.session_state.current_repo_id_for_query = msg.split("Namespace: ")[-1].split(".")[0]
                    st.session_state.current_repo_url_display = repo_url_to_index
                    st.success(f"Indexing job started! Job ID: {st.session_state.indexing_job_id}. Repo ID/Namespace: {st.session_state.current_repo_id_for_query}")
                    st.info("Indexing runs in the background and may take several minutes. Check status below.")
                    st.session_state.last_job_status = None # Clear previous status
                except requests.exceptions.RequestException as e: st.error(f"Error requesting indexing: {e}")
                except Exception as e: st.error(f"An unexpected error occurred: {e}")

# --- UI for Checking Job Status ---
if st.session_state.indexing_job_id:
    with st.expander("Check Indexing Job Status", expanded=True):
        st.write(f"Current Job ID: `{st.session_state.indexing_job_id}` for `{st.session_state.current_repo_url_display}`")
        if st.button("Refresh Indexing Status", key="check_status_button"):
            with st.spinner(f"Checking status..."):
                try:
                    response = requests.get(f"{BACKEND_API_BASE_URL}/job_status/{st.session_state.indexing_job_id}", timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    st.session_state.last_job_status = data # Store the whole status object
                    st.write(f"Job Status: **{data.get('status')}**")
                    if data.get('result'): st.json(data.get('result')) # Display Celery task result
                except requests.exceptions.RequestException as e: st.error(f"Error checking job status: {e}")
                except Exception as e: st.error(f"An unexpected error: {e}")
        
        # Display last known status if available
        if st.session_state.last_job_status:
            status_data = st.session_state.last_job_status
            st.write(f"Last Known Status: **{status_data.get('status')}**")
            if status_data.get('result'): st.json(status_data.get('result'))
            if status_data.get("status") == "SUCCESS" or (status_data.get("result") and status_data["result"].get("status") == "completed"):
                st.success(f"Indexing for {st.session_state.current_repo_url_display} (Namespace: {st.session_state.current_repo_id_for_query}) appears complete! You can now query it.")


# --- UI for Querying ---
st.markdown("---")
st.header("2. Query an Indexed Repository")

repo_id_to_query_default = st.session_state.current_repo_id_for_query or ""
repo_id_to_query = st.text_input("Enter Repository ID/Namespace (from indexing step above):", value=repo_id_to_query_default, key="repo_id_query_input")
user_question = st.text_input("Your question about the repository's history:", placeholder="e.g., What changed regarding sessions?", key="question_input_query")
query_button = st.button("Ask MementoAI", key="query_button")

results_placeholder = st.empty()

if query_button and repo_id_to_query and user_question:
    with st.spinner(f"Querying repository {repo_id_to_query}... This may take a moment."):
        try:
            payload = {"repo_id": repo_id_to_query, "question": user_question}
            response = requests.post(f"{BACKEND_API_BASE_URL}/query_repository", json=payload, timeout=120)
            response.raise_for_status()
            api_data = response.json()

            with results_placeholder.container():
                st.subheader(f"Query Results for Repository ID: {repo_id_to_query}")
                if api_data.get("status") == "error":
                    st.error(f"Query failed: {api_data.get('message', 'Unknown error')}")
                else:
                    if api_data.get("status") == "partial_success":
                        st.warning(f"Query completed with issues: {api_data.get('message', '')}")

                    st.subheader("💡 AI Summary")
                    if api_data.get("ai_summary"): st.markdown(api_data["ai_summary"])
                    else: st.info("AI Summary not available or not generated for this query.")
                    st.markdown("---")

                    st.subheader("🔍 Relevant History Found")
                    commits = api_data.get("relevant_commits", [])
                    if commits:
                        st.write(f"Found {len(commits)} relevant commits:")
                        for i, commit in enumerate(commits):
                            st.markdown(f"**{i+1}. Commit:** `{commit['hash'][:7]}` (Similarity: {commit['similarity']:.4f})")
                            st.markdown(f"*Author: {commit['author']} | Date: {commit['date']}*")
                            st.text_area(f"Message {i+1}", commit['message'], height=100, key=f"q_msg_{i}_{commit['hash']}") # More unique key
                            with st.expander(f"View Code Changes (Diff Stored During Indexing) for Commit {i+1}"):
                                 st.code(commit['diff'], language='diff', line_numbers=False)
                            # Attempt to reconstruct GitHub link if repo_url was stored or can be inferred
                            # This part is tricky as current_repo_url_display might not match repo_id_to_query
                            # For now, we'll omit the direct GitHub link here for simplicity
                            st.markdown("---")
                    else: st.info("No specific relevant commits were identified for this query.")

        except requests.exceptions.RequestException as e:
            with results_placeholder.container(): st.error(f"Error contacting MementoAI API: {e}")
        except Exception as e:
            with results_placeholder.container(): st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("MementoAI | Real-World Startup Idea")