import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Indexes
# ----------------------------
def load_indexes():
    chunks = pickle.load(open("data/processed/chunks.pkl", "rb"))
    embeddings = np.load("data/processed/embeddings.npy")
    bm25 = pickle.load(open("data/processed/bm25.pkl", "rb"))
    faiss_index = faiss.read_index("data/processed/faiss_index.bin")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return chunks, embeddings, bm25, faiss_index, embed_model


# ----------------------------
# Hybrid Search
# ----------------------------
def hybrid_search(query, top_k=5, alpha=0.5):
    """
    alpha = weight for BM25 (0.5 recommended)
    top_k = number of final retrieved chunks
    """
    chunks, embeddings, bm25, faiss_index, embed_model = load_indexes()

    # ----------------------------
    # 1. BM25 retrieval
    # ----------------------------
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # ----------------------------
    # 2. FAISS embedding search
    # ----------------------------
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = faiss_index.search(q_emb, top_k * 10)  # get more candidates
    emb_scores = np.zeros(len(chunks))
    for score, idx in zip(D[0], I[0]):
        emb_scores[idx] = score

    emb_norm = (emb_scores - np.min(emb_scores)) / (np.max(emb_scores) - np.min(emb_scores))

    # ----------------------------
    # 3. Combine BM25 + Embeddings
    # ----------------------------
    final_scores = alpha * bm25_norm + (1 - alpha) * emb_norm

    # Top K results
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            "doc_id": chunks[i]["doc_id"],
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"]
        })

    return results
