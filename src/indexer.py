import json
import pickle
import numpy as np
import faiss
import nltk
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# -------------------------
# Load chunks
# -------------------------
def load_chunks(path="data/processed/chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Build embeddings
# -------------------------
def build_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    )

    faiss.normalize_L2(embeddings)
    return embeddings, model

# -------------------------
# FAISS index
# -------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# -------------------------
# BM25 index
# -------------------------
def build_bm25(chunks):
    corpus = [c["text"].lower() for c in chunks]
    tokenized = [word_tokenize(text) for text in corpus]
    return BM25Okapi(tokenized)
