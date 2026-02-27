import os
import pickle
import faiss
import numpy as np

from google.oauth2 import service_account
from google import genai

from pdf_utils import extract_text_from_pdf, chunk_text


# ----------------------------
# CONFIG
# ----------------------------

PDF_FOLDER = "data/knowledge_base"
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")


# ----------------------------
# Load Google Credentials
# ----------------------------

import json

with open("service_account.json") as f:
    creds_dict = json.load(f)

credentials = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = genai.Client(
    vertexai=True,
    credentials=credentials,
    project=creds_dict["project_id"],
    location="us-central1",
)


# ----------------------------
# Embedding Function
# ----------------------------

def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text],
    )
    return np.array(response.embeddings[0].values)


# ----------------------------
# Build Index
# ----------------------------

def build_index():

    print("Reading PDFs...")

    all_text = ""

    for file_name in os.listdir(PDF_FOLDER):
        if file_name.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file_name)
            print("Processing:", file_name)

            with open(path, "rb") as f:
                all_text += extract_text_from_pdf(f)

    print("Chunking...")
    chunks = chunk_text(all_text)

    print("Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector store created successfully!")


if __name__ == "__main__":
    build_index()