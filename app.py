import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import openai # OpenAI library
import os
import time

# --- Configuration ---
# Use environment variable for API key for better security
# In a real app, use .env files or secrets management
# For hackathon quick start, you can paste your key here temporarily,
# BUT REMOVE IT BEFORE SHARING/COMMITTING CODE
try:
    # Try getting key from environment variable first
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    # If env var not set, fallback to Streamlit secrets or manual paste
    # For local running without Streamlit Cloud secrets:
    openai_api_key_manual = "YOUR_OPENAI_API_KEY_GOES_HERE" # <-- PASTE YOUR sk-XXXX KEY HERE TEMPORARILY (REMOVE LATER)
    if openai_api_key_manual:
         openai.api_key = openai_api_key_manual
    else:
         # Placeholder if no key is found anywhere
         openai.api_key = "YOUR_API_KEY_NOT_SET"
         print("Warning: OPENAI_API_KEY environment variable or manual key not set.")


CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "requests_commits"
MODEL_NAME = 'all-MiniLM-L6-v2' # Must match the model used for embedding
LLM_MODEL = "gpt-3.5-turbo-0125" # Or "gpt-4-turbo", etc.

# --- Load Models and DB (Cache to avoid reloading on every interaction) ---

@st.cache_resource # Use Streamlit's caching for resources
def load_resources():
    """Loads the embedding model and connects to ChromaDB."""
    print("Loading resources...")
    start_time = time.time()
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Embedding model loaded ({time.time() - start_time:.2f}s). Device: {model.device}")
    except Exception as e:
        st.error(f"Error loading embedding model '{MODEL_NAME}': {e}")
        return None, None

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to ChromaDB collection '{COLLECTION_NAME}'. Items: {collection.count()}")
        return model, collection
    except Exception as e:
        st.error(f"Error connecting to ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_DB_PATH}': {e}")
        st.error("Ensure './chroma_db' folder exists and 'embed_commits.py' ran successfully.")
        return model, None # Return model even if DB fails, maybe show partial error

# Load resources when the script runs
embedding_model, chroma_collection = load_resources()

# --- Streamlit App Interface ---

st.set_page_config(layout="wide") # Use wider layout
st.title(" MementoAI 🏛️")
st.markdown("Ask questions about the **Requests** library's commit history.")

# Input box for user question
user_question = st.text_input("Your question:", placeholder="e.g., Why was the session object refactored?")

if user_question and embedding_model and chroma_collection:
    if openai.api_key == "YOUR_API_KEY_NOT_SET" or not openai.api_key:
         st.error("OpenAI API key not set. Please add it to the script or environment variable.")
    else:
        st.markdown("---") # Separator
        st.subheader("Processing your question...")

        with st.spinner("Finding relevant commits..."):
            # 1. Embed the user's question
            start_time = time.time()
            question_embedding = embedding_model.encode([user_question])[0].tolist()
            # print(f"Question embedded in {time.time() - start_time:.2f}s")

            # 2. Query ChromaDB for relevant commits
            try:
                # Query for top 5 most similar commits
                results = chroma_collection.query(
                    query_embeddings=[question_embedding],
                    n_results=5, # Number of relevant commits to retrieve
                    include=['documents', 'metadatas', 'distances'] # Include text, metadata, and similarity score
                )
                # print(f"ChromaDB query took {time.time() - start_time:.2f}s")

                relevant_commits_docs = results.get('documents', [[]])[0]
                relevant_commits_metas = results.get('metadatas', [[]])[0]
                relevant_commits_distances = results.get('distances', [[]])[0]

                if not relevant_commits_docs:
                     st.warning("Could not find relevant commits for your question.")
                     st.stop() # Stop execution for this run if nothing found

            except Exception as e:
                st.error(f"Error querying ChromaDB: {e}")
                st.stop() # Stop if DB query fails

        # Display retrieved commits (optional, good for debugging/transparency)
        with st.expander("🔍 View Relevant Commits Found"):
             if relevant_commits_docs:
                  for i, (doc, meta, dist) in enumerate(zip(relevant_commits_docs, relevant_commits_metas, relevant_commits_distances)):
                       st.markdown(f"**Commit {i+1} (Distance: {dist:.4f})**")
                       st.markdown(f"*Author: {meta.get('author', 'N/A')} | Timestamp: {time.strftime('%Y-%m-%d', time.gmtime(meta.get('timestamp', 0)))}*")
                       st.text_area(f"Commit Message {i+1}", doc, height=100, key=f"commit_text_{i}")
                       st.markdown("---")
             else:
                  st.write("No relevant commits found.")


        with st.spinner("🧠 Asking the AI Assistant (LLM) for an answer..."):
            # 3. Prepare context and prompt for LLM
            context = ""
            for i, doc in enumerate(relevant_commits_docs):
                context += f"Commit {i+1}:\n{doc}\n\n" # Combine retrieved commit messages

            prompt = f"""
            You are an AI assistant acting as a "Codebase Archaeologist".
            Your task is to answer the user's question based *only* on the provided commit message context from the 'requests' library history.
            Do not make assumptions or use external knowledge.
            If the context does not contain enough information to answer the question, explicitly state that.
            Be concise and helpful.

            Provided Context (Commit Messages):
            ---
            {context}
            ---

            User Question: {user_question}

            Answer:
            """

            # 4. Call OpenAI API
            start_time = time.time()
            try:
                response = openai.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant analyzing Git commit history."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2, # Lower temperature for more factual answers
                    max_tokens=300 # Limit response length
                )
                ai_answer = response.choices[0].message.content
                # print(f"OpenAI call took {time.time() - start_time:.2f}s")

            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")
                st.error("Check your API key and OpenAI account status.")
                ai_answer = "Error: Could not get answer from AI Assistant."


        # 5. Display the final answer
        st.subheader("💡 Answer from MementoAI")
        st.markdown(ai_answer)

elif user_question and (not embedding_model or not chroma_collection):
    st.error("Resources (Embedding Model or Database) failed to load. Cannot process question. Check terminal logs.")

# Add some footer or instructions
st.markdown("---")
st.markdown("Built for Hackathon | Data Source: `requests` library commit history")