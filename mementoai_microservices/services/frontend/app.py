import html
import os
from difflib import unified_diff
from typing import Dict, Tuple

import plotly.express as px
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="MementoAI", page_icon="M")

BACKEND_API_BASE_URL = os.environ.get("MEMENTO_API_URL", "http://127.0.0.1:8000")

for key in ("indexing_job_id", "current_repo_id_for_query", "current_repo_url_display", "last_job_status"):
    if key not in st.session_state:
        st.session_state[key] = None


def extract_code_from_diff(diff_text: str) -> Tuple[str, str]:
    old_lines = []
    new_lines = []
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        elif not line.startswith("@"):
            old_lines.append(line)
            new_lines.append(line)
    return "\n".join(old_lines), "\n".join(new_lines)


def render_diff_viewer(old_code: str, new_code: str) -> None:
    diff = list(
        unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile="Old Version",
            tofile="New Version",
            n=3,
        )
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Old Version**")
        for line in diff:
            if line.startswith("-") and not line.startswith("---"):
                st.code(html.escape(line[1:]), language="diff")
    with col2:
        st.markdown("**New Version**")
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                st.code(html.escape(line[1:]), language="diff")


def render_technical_debt(debt_data: Dict) -> None:
    if not debt_data:
        return
    with st.expander("Technical Debt Analysis", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Debt Score", f"{debt_data.get('technical_debt_score', 0):.1f}/100")
        col2.metric("Maintainability", f"{debt_data.get('maintainability_index', 0):.1f}")
        col3.metric("Duplication", f"{debt_data.get('duplication', 0):.1f}%")
        if debt_data.get("cyclomatic_complexity"):
            fig = px.bar(
                x=list(debt_data["cyclomatic_complexity"].keys()),
                y=list(debt_data["cyclomatic_complexity"].values()),
                title="Cyclomatic Complexity by Function",
            )
            st.plotly_chart(fig, use_container_width=True)


st.title("MementoAI")
st.write("Analyze the history of any public Git repository with indexing, semantic retrieval, and optional AI summarization.")

with st.expander("1. Index New Repository", expanded=True):
    repo_url_to_index = st.text_input("Enter public Git repository URL", placeholder="https://github.com/psf/requests.git")
    if st.button("Start Indexing Repository", type="primary"):
        if not repo_url_to_index.endswith(".git"):
            st.error("Please enter a valid Git repository URL ending in .git")
        else:
            response = requests.post(
                f"{BACKEND_API_BASE_URL}/index_repository",
                json={"repo_url": repo_url_to_index},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.indexing_job_id = data.get("job_id")
            st.session_state.current_repo_id_for_query = data.get("repo_id")
            st.session_state.current_repo_url_display = repo_url_to_index
            st.success(f"Indexing job started. Job ID: {data.get('job_id')}")


if st.session_state.indexing_job_id:
    with st.expander("Check Indexing Job Status", expanded=True):
        st.write(f"Current Job ID: `{st.session_state.indexing_job_id}`")
        st.write(f"Repository: `{st.session_state.current_repo_url_display}`")

        if st.button("Refresh Indexing Status"):
            response = requests.get(
                f"{BACKEND_API_BASE_URL}/job_status/{st.session_state.indexing_job_id}",
                timeout=15,
            )

            response.raise_for_status()
            st.session_state.last_job_status = response.json()

        if st.session_state.last_job_status:
            st.json(st.session_state.last_job_status)

st.markdown("---")

st.header("2. Query an Indexed Repository")

repo_id_to_query = st.text_input(
    "Repository ID",
    value=st.session_state.current_repo_id_for_query or ""
)

user_question = st.text_area(
    "Your question",
    height=100
)

if st.button("Ask MementoAI", type="primary") and repo_id_to_query and user_question:
    response = requests.post(
        f"{BACKEND_API_BASE_URL}/query_repository",
        json={
            "repo_id": repo_id_to_query,
            "question": user_question,
        },
        timeout=60,
    )

    response.raise_for_status()
    api_data = response.json()

    st.subheader(f"Query Results for: {user_question}")

    if api_data.get("ai_summary"):
        with st.expander("AI Summary", expanded=True):
            st.write(api_data["ai_summary"])

    for commit in api_data.get("relevant_commits", []):
        with st.container(border=True):
            st.markdown(
                f"**{commit['hash'][:7]}**  Similarity: {commit['similarity']:.2f}"
            )

            st.markdown(
                f"Author: {commit.get('author', 'N/A')}  "
                f"Date: {commit.get('date', 'Unknown')}"
            )

            st.markdown(
                f"Message: {commit.get('message', '')}"
            )

            if commit.get("function_changes"):
                st.json(commit["function_changes"])

            if commit.get("technical_debt"):
                render_technical_debt(commit["technical_debt"])

            if commit.get("diff"):
                old_code, new_code = extract_code_from_diff(
                    commit["diff"]
                )
                render_diff_viewer(old_code, new_code)
            