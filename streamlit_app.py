import streamlit as st
from src.retriever import hybrid_search
from src.summarizer import summarize_chunks

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Document Search & Summarization",
    layout="wide"
)

st.title("📄 Document Search & Summarization using LLMs")
st.markdown(
    "Hybrid search (BM25 + embeddings) with LLM-powered summarization."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=3,
    max_value=10,
    value=5
)

summary_length = st.sidebar.selectbox(
    "Summary length",
    ["short", "medium", "long"]
)

alpha = st.sidebar.slider(
    "BM25 vs Embeddings weight (alpha)",
    0.0, 1.0, 0.5
)

# -----------------------------
# Query input
# -----------------------------
query = st.text_input(
    "🔍 Enter your query",
    placeholder="e.g. optimization techniques in machine learning"
)

search_button = st.button("Search & Summarize")

# -----------------------------
# Main logic
# -----------------------------
if search_button and query.strip():

    with st.spinner("Retrieving relevant documents..."):
        results = hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha
        )

    if not results:
        st.warning("No relevant documents found.")
    else:
        st.success(f"Retrieved {len(results)} relevant chunks")

        # -----------------------------
        # Show retrieved chunks
        # -----------------------------
        st.subheader("📚 Retrieved Chunks")

        for i, r in enumerate(results, 1):
            with st.expander(f"Chunk {i} | Doc ID: {r['doc_id']}"):
                st.write(r["text"])

        # -----------------------------
        # Summarization
        # -----------------------------
        with st.spinner("Generating summary using LLM..."):
            summary = summarize_chunks(
                query=query,
                chunks=results,
                length=summary_length
            )

        st.subheader("📝 LLM Summary")
        st.write(summary)

elif search_button:
    st.warning("Please enter a query.")
