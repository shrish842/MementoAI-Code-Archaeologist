import streamlit as st
import requests
import time
import os
import plotly.express as px
from typing import List, Dict, Optional, Tuple
from difflib import unified_diff
import json # Import json to parse technical_debt string

st.set_page_config(layout="wide", page_title="MementoAI", page_icon="🏛️")

# Configuration
BACKEND_API_BASE_URL = os.environ.get("MEMENTO_API_URL", "http://127.0.0.1:8000")

# Session State
if 'indexing_job_id' not in st.session_state:
    st.session_state.indexing_job_id = None
if 'current_repo_id_for_query' not in st.session_state:
    st.session_state.current_repo_id_for_query = None
if 'current_repo_url_display' not in st.session_state:
    st.session_state.current_repo_url_display = None
if 'last_job_status' not in st.session_state:
    st.session_state.last_job_status = None

# Custom CSS for diff viewer and debt visualization
st.markdown("""
<style>
    .diff-container {
        display: flex;
        width: 100%;
        overflow-x: auto;
        font-family: monospace;
        background: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
    }
    .diff-pane {
        flex: 1;
        min-width: 45%;
        padding: 0 5px;
    }
    .diff-line {
        white-space: pre;
        line-height: 1.5;
        font-size: 14px;
    }
    .diff-added {
        background: rgba(46, 160, 67, 0.2);
        border-left: 3px solid #2ea043;
    }
    .diff-removed {
        background: rgba(248, 81, 73, 0.2);
        border-left: 3px solid #f85149;
    }
    .diff-unchanged {
        color: #d4d4d4;
    }
    .diff-gutter {
        color: #858585;
        padding: 0 10px;
        user-select: none;
    }
    .debt-card {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f8f9fa;
    }
    .debt-high {
        border-left: 4px solid #e74c3c;
    }
    .debt-medium {
        border-left: 4px solid #f39c12;
    }
    .debt-low {
        border-left: 4px solid #2ecc71;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

def extract_code_from_diff(diff_text: str) -> Tuple[str, str]:
    """Extract old and new code versions from unified diff"""
    old_lines = []
    new_lines = []

    for line in diff_text.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            old_lines.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            new_lines.append(line[1:])
        elif not line.startswith('@'):
            # Context line - add to both
            old_lines.append(line)
            new_lines.append(line)

    return '\n'.join(old_lines), '\n'.join(new_lines)

def render_diff_viewer(old_code: str, new_code: str):
    """Enhanced VS Code-like diff viewer"""
    diff = list(unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile="Old Version",
        tofile="New Version",
        n=3
    ))

    with st.container():
        st.markdown("### Code Changes")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Old Version**")
            old_container = st.container(height=400, border=True)

        with col2:
            st.markdown("**New Version**")
            new_container = st.container(height=400, border=True)

        # Streamlit's markdown doesn't support direct HTML for complex diffs well
        # Reverting to simpler display or using custom components for true side-by-side
        # For now, a simplified display within two columns
        with old_container:
            for line in diff:
                if line.startswith('-') and not line.startswith('---'):
                    st.markdown(f'<div class="diff-line diff-removed">{line[1:].strip()}</div>',
                               unsafe_allow_html=True)
                elif not line.startswith('+') and not line.startswith('@'):
                    st.markdown(f'<div class="diff-line diff-unchanged">{line.strip()}</div>',
                               unsafe_allow_html=True)

        with new_container:
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    st.markdown(f'<div class="diff-line diff-added">{line[1:].strip()}</div>',
                               unsafe_allow_html=True)
                elif not line.startswith('-') and not line.startswith('@'):
                    st.markdown(f'<div class="diff-line diff-unchanged">{line.strip()}</div>',
                               unsafe_allow_html=True)


def render_technical_debt(debt_data: Dict, debt_delta: Optional[float] = None):
    """Interactive technical debt visualization"""
    if not debt_data:
        return

    # Ensure debt_data is a dictionary, not a Pydantic object
    # The API should already return it as a dict, but this is a safeguard
    if hasattr(debt_data, 'dict'):
        debt_data = debt_data.dict()

    with st.expander("🧮 Technical Debt Analysis", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Metrics", "Trends", "Breakdown"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Debt Score",
                         f"{debt_data.get('technical_debt_score', 0):.1f}/100",
                         delta=f"{debt_delta:+.1f} change" if debt_delta is not None else None,
                         delta_color="inverse")

            with col2:
                st.metric("Maintainability",
                         f"{debt_data.get('maintainability_index', 0):.1f}",
                         help="Higher is better (0-100 scale)")

            with col3:
                st.metric("Duplication",
                         f"{debt_data.get('duplication', 0):.1f}%",
                         help="Percentage of duplicated code")

        with tab2:
            # This part requires historical data which is not directly passed per commit
            # For a real trend, you'd need to query historical debt data for the repo
            st.info("Historical trend data is not available per commit in this view. It would require a separate historical data store.")
            # if 'history' in debt_data:
            #     fig = px.line(
            #         debt_data['history'],
            #         x='date',
            #         y='technical_debt_score',
            #         title='Technical Debt Trend',
            #         markers=True
            #     )
            #     st.plotly_chart(fig, use_container_width=True)
            # else:
            #     st.info("No historical data available for this commit")

        with tab3:
            # Cyclomatic Complexity
            cc_data = debt_data.get('cyclomatic_complexity', {})
            if cc_data:
                st.markdown("##### Cyclomatic Complexity per Function")
                for func_name, complexity in cc_data.items():
                    st.markdown(f"- `{func_name}`: {complexity}")
            else:
                st.info("No cyclomatic complexity data available.")

            # Maintainability Index
            mi = debt_data.get('maintainability_index', 0)
            st.markdown(f"##### Maintainability Index: {mi:.1f}")
            st.progress(mi / 100.0)

            # Duplication
            duplication = debt_data.get('duplication', 0)
            st.markdown(f"##### Duplication: {duplication:.1f}%")
            st.progress(duplication / 100.0)

            if debt_data.get('code_smells'):
                with st.expander(f"🚨 Code Smells ({len(debt_data['code_smells'])})"):
                    for smell in debt_data['code_smells'][:5]:
                        st.markdown(f"- {smell}")
                    if len(debt_data['code_smells']) > 5:
                        st.markdown(f"*+ {len(debt_data['code_smells']) - 5} more...*")


# UI Setup
st.title("🏛️ MementoAI: Codebase Archaeologist")
st.markdown("""
    <div style="margin-bottom: 20px;">
        <p>Analyze the commit history of any <strong>public</strong> Git repository with:</p>
        <ul>
            <li>🔍 AST-powered code change analysis</li>
            <li>📊 Technical debt tracking</li>
            <li>🔄 VS Code-like diff visualization</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# --- Indexing UI ---
with st.expander("1. Index New Repository (Run this first!)", expanded=True):
    repo_url_to_index = st.text_input(
        "Enter PUBLIC Git Repository URL (.git):",
        placeholder="e.g., https://github.com/psf/requests.git",
        key="repo_url_index_input"
    )
    index_button = st.button("Start Indexing Repository", key="index_button", type="primary")

    if index_button and repo_url_to_index:
        if not repo_url_to_index.endswith(".git"):
            st.error("Please enter a valid Git repository URL ending in .git")
        else:
            with st.spinner(f"Requesting indexing for {repo_url_to_index}..."):
                try:
                    response = requests.post(
                        f"{BACKEND_API_BASE_URL}/index_repository",
                        json={"repo_url": repo_url_to_index},
                        timeout=20
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.session_state.indexing_job_id = data.get("job_id")
                    msg = data.get("message", "")

                    # Extract namespace from message
                    if "Namespace: " in msg:
                        st.session_state.current_repo_id_for_query = msg.split("Namespace: ")[-1].strip()

                    st.session_state.current_repo_url_display = repo_url_to_index
                    st.success(
                        f"Indexing job started! Job ID: {st.session_state.indexing_job_id}\n"
                        f"Repo ID/Namespace: {st.session_state.current_repo_id_for_query}"
                    )
                    st.info("Indexing runs in the background and may take several minutes. Check status below.")
                    st.session_state.last_job_status = None
                except requests.exceptions.RequestException as e:
                    st.error(f"Error requesting indexing: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

# --- Job Status UI ---
if st.session_state.indexing_job_id:
    with st.expander("Check Indexing Job Status", expanded=True):
        st.write(f"Current Job ID: `{st.session_state.indexing_job_id}`")
        st.write(f"Repository: `{st.session_state.current_repo_url_display}`")

        if st.button("🔄 Refresh Indexing Status", key="check_status_button"):
            with st.spinner("Checking status..."):
                try:
                    response = requests.get(
                        f"{BACKEND_API_BASE_URL}/job_status/{st.session_state.indexing_job_id}",
                        timeout=20
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.session_state.last_job_status = data

                    status = data.get("status", "").upper()
                    if status == "SUCCESS" or (data.get("result") and data["result"].get("status") == "completed"):
                        st.success(f"✅ Status: {status}")
                    elif status == "FAILURE" or (data.get("result") and data["result"].get("status") == "error"):
                        st.error(f"❌ Status: {status}")
                    else:
                        st.info(f"🔄 Status: {status}")

                    if data.get('result'):
                        with st.expander("Detailed Results"):
                            st.json(data.get('result'))
                except requests.exceptions.RequestException as e:
                    st.error(f"Error checking job status: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error: {str(e)}")

        if st.session_state.last_job_status:
            status_data = st.session_state.last_job_status
            if status_data.get("status") == "SUCCESS" or \
               (status_data.get("result") and status_data["result"].get("status") == "completed"):
                st.balloons()
                st.success("Indexing complete! You can now query the repository.")

st.markdown("---")
st.header("2. Query an Indexed Repository")

repo_id_to_query_default = st.session_state.current_repo_id_for_query or ""
repo_id_to_query = st.text_input(
    "Repository ID/Namespace (from indexing step above):",
    value=repo_id_to_query_default,
    key="repo_id_query_input"
)

user_question = st.text_area(
    "Your question about the repository's history:",
    placeholder="e.g., Why was the session handling changed?\n"
                "What functions were modified for the authentication system?\n"
                "Show me changes related to API rate limiting",
    height=100,
    key="question_input_query"
)

query_button = st.button("🔍 Ask MementoAI", key="query_button", type="primary")

results_placeholder = st.empty()

if query_button and repo_id_to_query and user_question:
    with st.spinner(f"Querying repository {repo_id_to_query}..."):
        try:
            payload = {"repo_id": repo_id_to_query, "question": user_question}
            response = requests.post(
                f"{BACKEND_API_BASE_URL}/query_repository",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            api_data = response.json()

            with results_placeholder.container():
                st.subheader(f"🔎 Query Results for: *{user_question}*")

                if api_data.get("status") == "error":
                    st.error(f"Query failed: {api_data.get('message', 'Unknown error')}")
                else:
                    if api_data.get("status") == "partial_success":
                        st.warning(f"Note: {api_data.get('message', '')}")

                    # AI Summary Section
                    with st.expander("💡 AI Summary", expanded=True):
                        if api_data.get("ai_summary"):
                            st.markdown(api_data["ai_summary"])
                        else:
                            st.info("AI Summary not available for this query")

                    st.markdown("---")

                    # Relevant Commits Section
                    commits = api_data.get("relevant_commits", [])

                    if commits:
                        st.subheader(f"📜 Relevant Commits ({len(commits)})")

                        for i, commit in enumerate(commits):
                            with st.container(border=True):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"**Commit {i+1}:** `{commit['hash'][:7]}` "
                                                f"(Similarity: {commit['similarity']:.2f})")
                                    st.markdown(f"*👤 {commit['author']} | 📅 {commit['date']}*")
                                    st.markdown(f"**Message:** {commit['message']}")

                                with col2:
                                    # Parse the technical_debt string back to a dict for display
                                    if commit.get('technical_debt'):
                                        try:
                                            td_data = json.loads(commit['technical_debt'])
                                            debt_score = td_data.get('technical_debt_score', 0)
                                            debt_class = "debt-high" if debt_score > 60 else "debt-medium" if debt_score > 30 else "debt-low"
                                            st.markdown(f"""
                                            <div class="debt-card {debt_class}">
                                                <h4>Debt Score: {debt_score:.1f}/100</h4>
                                                <p>Δ {commit.get('debt_delta', 0):+.1f} from parent</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        except json.JSONDecodeError:
                                            st.warning("Could not parse technical debt data for summary card.")


                                # Function Changes
                                if commit.get('function_changes'):
                                    with st.expander("🔄 Function Changes", expanded=False):
                                        cols = st.columns(3)
                                        with cols[0]:
                                            st.markdown("**Added**")
                                            for change in [c for c in commit['function_changes'] if c['change_type'] == 'added']:
                                                st.markdown(f"- 🟢 `{change['name']}`")
                                        with cols[1]:
                                            st.markdown("**Removed**")
                                            for change in [c for c in commit['function_changes'] if c['change_type'] == 'removed']:
                                                st.markdown(f"- 🔴 `{change['name']}`")
                                        with cols[2]:
                                            st.markdown("**Modified**")
                                            for change in [c for c in commit['function_changes'] if c['change_type'] == 'modified']:
                                                delta = change.get('complexity_change', 0)
                                                arrow = "↑" if delta > 0 else "↓" if delta < 0 else ""
                                                st.markdown(f"- 🔵 `{change['name']}` {arrow}{abs(delta) if delta else ''}")

                                # Technical Debt Analysis
                                if commit.get('technical_debt'):
                                    try:
                                        td_data = json.loads(commit['technical_debt'])
                                        # Pass debt_delta to render_technical_debt for display
                                        render_technical_debt(td_data, commit.get('debt_delta'))
                                    except json.JSONDecodeError:
                                        st.warning("Could not parse technical debt data for detailed view.")

                                # Diff Viewer
                                if commit['diff'] and not commit['diff'].startswith("Error"):
                                    try:
                                        old_code, new_code = extract_code_from_diff(commit['diff'])
                                        render_diff_viewer(old_code, new_code)
                                    except Exception as e:
                                        st.error(f"Error rendering diff: {str(e)}")
                                else:
                                    st.info("No diff available for this commit")

                                st.markdown("---")
                    else:
                        st.info("No relevant commits found for this query")

        except requests.exceptions.RequestException as e:
            with results_placeholder.container():
                st.error(f"Error contacting MementoAI API: {str(e)}")
        except Exception as e:
            with results_placeholder.container():
                st.error(f"An unexpected error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        <p>MementoAI - Codebase Archaeology System</p>
        <p>Patent Pending - © 2024</p>
    </div>
""", unsafe_allow_html=True)

