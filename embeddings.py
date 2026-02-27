import os
import numpy as np
import faiss
import pickle


# --------------------------------------------------
# Paths (Cloud Safe)
# --------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(CURRENT_DIR, "vector_store")

INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")


# --------------------------------------------------
# Load Precomputed Index
# --------------------------------------------------

def load_index():
    if not os.path.exists(INDEX_PATH):
        return None, None

    if not os.path.exists(CHUNKS_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# --------------------------------------------------
# Embedding (Query Only)
# --------------------------------------------------

def get_embedding(text, client):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text],
    )
    return np.array(response.embeddings[0].values)


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_top_k(query, chunks, index, client, k=3):
    query_embedding = get_embedding(query, client)
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]