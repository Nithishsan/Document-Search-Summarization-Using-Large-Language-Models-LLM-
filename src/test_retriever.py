from src.retriever import hybrid_search

queries = [
    "machine learning optimization techniques",
    "neural networks and backpropagation",
    "reinforcement learning algorithms",
    "natural language processing transformers",
]

for q in queries:
    print("\n" + "="*80)
    print("QUERY:", q)

    results = hybrid_search(q, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Doc ID:", r["doc_id"])
        print("Chunk ID:", r["chunk_id"])
        print("Score:", round(r["score"], 4))
        print("Text Preview:", r["text"][:250].replace("\n", " "), "...")
