import pickle
import numpy as np
import faiss
from src.indexer import (
    load_chunks,
    build_embeddings,
    build_faiss_index,
    build_bm25
)

print("Loading chunks...")
chunks = load_chunks("data/processed/chunks.json")
print(f"Loaded {len(chunks)} chunks")

print("Building embeddings (this may take a few minutes)...")
embeddings, _ = build_embeddings(chunks)
np.save("data/processed/embeddings.npy", embeddings)

print("Building FAISS index...")
faiss_index = build_faiss_index(embeddings)
faiss.write_index(faiss_index, "data/processed/faiss_index.bin")

print("Building BM25 index...")
bm25 = build_bm25(chunks)
with open("data/processed/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("Saving chunks.pkl...")
with open("data/processed/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Indexing completed successfully")
