import json
import chromadb # Vector Database client
from sentence_transformers import SentenceTransformer # Embedding model library
import time # To time the process
import os # To construct file paths reliably

# --- Configuration ---
# Input file from the previous step
COMMIT_DATA_FILE = 'requests_commits.json'
# Directory to store the Chroma database files
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
# Name for our collection (like a table) in ChromaDB
COLLECTION_NAME = "requests_commits"
# Name of the embedding model to use (a good default)
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Main Script Logic ---
if __name__ == "__main__":
    # Get the absolute path to the commit data file
    script_dir = os.path.dirname(__file__) # Directory where the script is running
    commit_file_path = os.path.join(script_dir, COMMIT_DATA_FILE)

    # 1. Load the extracted commit data
    print(f"Loading commit data from: {commit_file_path}...")
    try:
        with open(commit_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        print(f"Loaded {len(commits)} commits.")
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"Commit data file not found: {commit_file_path}")
        print("Please make sure 'extract_git_log.py' ran successfully and created the file.")
        print("-------------")
        exit() # Stop the script if data isn't found
    except json.JSONDecodeError:
        print(f"--- ERROR ---")
        print(f"Error decoding JSON from file: {commit_file_path}")
        print("The file might be corrupted or empty. Try running 'extract_git_log.py' again.")
        print("-------------")
        exit()
    except Exception as e:
        print(f"--- ERROR ---")
        print(f"An unexpected error occurred loading commit data: {e}")
        print("-------------")
        exit()

    if not commits:
        print("No commits found in the data file. Exiting.")
        exit()

    # 2. Load the Sentence Transformer model
    # This might download the model files (a few hundred MB) the first time.
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    print("(This may take a moment, especially the first time...)")
    start_time = time.time()
    try:
       # Attempt to use GPU if available, otherwise fallback is automatic usually
       # Forcing CPU: device='cpu'
       # Forcing CUDA: device='cuda'
       model = SentenceTransformer(MODEL_NAME) # Let the library decide CPU/GPU first
       # Check which device it ended up on:
       print(f"Model loaded successfully. Using device: {model.device}")
    except Exception as e:
       print(f"--- FATAL ERROR ---")
       print(f"Could not load Sentence Transformer model '{MODEL_NAME}': {e}")
       print("Check your internet connection if this is the first time.")
       print("Ensure 'pip install sentence-transformers' completed successfully.")
       print("-------------------")
       exit()

    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")


    # 3. Prepare data lists for ChromaDB
    # ChromaDB expects separate lists for IDs, the text documents, and metadata.
    print("Preparing data for database...")
    ids = []
    documents = []
    metadatas = []

    for commit in commits:
        # Ensure required fields exist and the message is not empty
        if commit.get('hash') and commit.get('full_message'):
            ids.append(commit['hash']) # Use commit hash as unique ID
            documents.append(commit['full_message']) # The text to be embedded
            metadatas.append({
                # Store useful info to retrieve alongside search results
                "author": commit.get('author', 'N/A'),
                "timestamp": commit.get('timestamp', 0),
                "subject": commit.get('subject', '')
                # Avoid putting the very long 'full_message' or 'body' in metadata
            })
        # else: # Optional: Print warning for skipped commits
            # print(f"Warning: Skipping commit with missing hash or message: {commit.get('hash', 'N/A')}")


    if not ids:
        print("--- ERROR ---")
        print("No valid commits found to process after filtering. Check the JSON data.")
        print("-------------")
        exit()

    # Sanity check: Ensure all lists have the same number of items
    if not (len(ids) == len(documents) == len(metadatas)):
        print("--- FATAL ERROR ---")
        print("Internal error: Data list lengths mismatch after preparation!")
        print(f"IDs: {len(ids)}, Docs: {len(documents)}, Metas: {len(metadatas)}")
        print("-------------------")
        exit()

    print(f"Prepared {len(ids)} valid commits for embedding.")

    # 4. Generate Embeddings
    # This is the most time-consuming step.
    print(f"Generating embeddings for {len(documents)} documents...")
    print(f"(Device: {model.device}. This may take several minutes...)")
    start_time = time.time()
    # The model.encode() function converts the list of text strings into a list of vectors (embeddings)
    embeddings = model.encode(
        documents,
        show_progress_bar=True, # Displays a handy progress bar in the terminal
        batch_size=32 # Process in batches for efficiency (adjust if memory issues occur)
        )
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

    # 5. Initialize ChromaDB Client and Collection
    # We use PersistentClient to save the database to disk in the specified path.
    print(f"Initializing ChromaDB Client (saving to: {CHROMA_DB_PATH})...")
    try:
        # Creates the directory if it doesn't exist
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not initialize ChromaDB client at path '{CHROMA_DB_PATH}': {e}")
        print("Check permissions or if the path is valid.")
        print("-------------------")
        exit()

    print(f"Getting or creating ChromaDB collection: '{COLLECTION_NAME}'")
    try:
        # This command tries to get the collection if it already exists,
        # or creates a new one if it doesn't. This prevents errors if you re-run the script.
        # It's important that the embedding model (or its dimension) matches if the collection exists.
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity, standard for sentence transformers
            )
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not get or create ChromaDB collection '{COLLECTION_NAME}': {e}")
        print("If the collection exists and you changed the MODEL_NAME,")
        print("you may need to delete the './chroma_db' folder and re-run.")
        print("-------------------")
        exit()


    # 6. Add data (Embeddings, Documents, Metadatas) to ChromaDB Collection
    print(f"Adding {len(ids)} items to the ChromaDB collection...")
    print("(This might take a moment...)")
    start_time = time.time()
    try:
        # Add the data in batches to avoid potential memory issues with very large datasets
        batch_size = 500 # Add 500 commits at a time
        for i in range(0, len(ids), batch_size):
            # Calculate batch indices
            end_index = min(i + batch_size, len(ids))
            print(f"  Adding batch {i // batch_size + 1} ({i+1} to {end_index})...")

            # Get slices for the current batch
            batch_ids = ids[i:end_index]
            batch_embeddings = embeddings[i:end_index]
            # Convert numpy arrays in embeddings to lists for JSON serialization compatibility with Chroma
            batch_embeddings_list = [emb.tolist() for emb in batch_embeddings]
            batch_documents = documents[i:end_index]
            batch_metadatas = metadatas[i:end_index]

            # Add the batch to the collection
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings_list,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        end_time = time.time()
        print(f"Data added successfully to ChromaDB in {end_time - start_time:.2f} seconds.")

    except Exception as e:
         print(f"\n--- ERROR ---")
         print(f"An error occurred while adding data to ChromaDB: {e}")
         print("Check data format or ChromaDB status.")
         print("---------------")
         exit()

    # 7. Final Verification
    try:
        count = collection.count()
        print(f"\n--- Verification ---")
        print(f"Collection '{COLLECTION_NAME}' in '{CHROMA_DB_PATH}' now contains {count} items.")
        print("Phase 2 (Embedding and Storage) Complete!")
        print("--------------------")
    except Exception as e:
        print(f"\n--- Warning ---")
        print(f"Could not verify collection count: {e}")
        print("Data *might* still be added, but verification failed.")
        print("-----------------")