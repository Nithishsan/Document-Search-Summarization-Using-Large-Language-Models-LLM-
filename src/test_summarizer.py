from src.retriever import hybrid_search
from src.summarizer import summarize_chunks

query = "optimization techniques in machine learning"

results = hybrid_search(query, top_k=5)

summary = summarize_chunks(
    query=query,
    chunks=results,
    length="medium"
)

print("\nQUERY:")
print(query)

print("\nSUMMARY:")
print(summary)
