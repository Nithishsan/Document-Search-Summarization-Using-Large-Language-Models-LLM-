📄 Document Search & Summarization Using Large Language Models (LLM)

A hybrid retrieval + LLM summarization system that enables fast, accurate search across large document corpora.
Built using BM25, FAISS vector search, SBERT embeddings, and GPT-4o-mini for production-grade retrieval augmented generation (RAG).

🚀 Project Overview

This project demonstrates a scalable end-to-end pipeline for:

🔎 Document Search

🧩 Hybrid Retrieval (BM25 + FAISS + Embeddings)

📝 LLM-based Summarization

🖥️ Interactive Streamlit UI

Users can query a text corpus, retrieve the most relevant document chunks, and generate concise, query-focused summaries powered by GPT-4o-mini.

This architecture mirrors real-world enterprise RAG systems used in search engines, chatbots, knowledge assistants, and research tools.

🏗️ System Architecture
High-Level Architecture

        User Query (Streamlit)
              │
              ▼
       Hybrid Retrieval Engine
      (BM25 + FAISS + Embeddings)
              │
       Top-K Relevant Chunks
              │
              ▼
     LLM Summarizer (GPT-4o-mini)
              │
              ▼
      Final Summarized Answer

📂 Directory Structure

<img width="296" height="558" alt="Screenshot 2025-12-15 140913" src="https://github.com/user-attachments/assets/43e901af-876d-4f2b-a9fa-c0a9e63313d9" />

## 🧠 Why This Tech Stack?

| Component        | Technology                  | Why It Matters                                       |
|------------------|-----------------------------|----------------------------------------------------- |
| Embeddings       | SBERT (all-MiniLM-L6-v2)     | Fast, lightweight, accurate semantic encoding       |
| Vector Index     | FAISS                       | Industry-standard for million-scale similarity search|
| Lexical Search   | BM25 (Rank-BM25)            | Handles keywords, acronyms, exact matches            |
| Hybrid Retrieval | Score Fusion                | Best of semantic + lexical relevance                 |
| Summarization    | GPT-4o-mini                 | Accurate, fast, low hallucination                    |
| UI               | Streamlit                   | Clean, interactive, production-ready                 |
| Preprocessing    | NLTK                        | Reliable tokenization                                |


This hybrid design reflects real-world RAG architecture used in enterprise AI search systems.

⚙️ Installation
1. Clone Repository
git clone https://github.com/<your-username>/Document-Search-Summarization-Using-Large-Language-Models-LLM
cd Document-Search-Summarization-Using-Large-Language-Models-LLM

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

🏗️ Running the Pipeline
1. Preprocess the Dataset
python -m src.run_preprocess

2. Chunk the Documents
python -m src.run_chunking

3. Build Indexes (FAISS + BM25 + Embeddings)
python -m src.run_indexing

4. Test Retrieval
python -m src.test_retriever

5. Launch Streamlit App
streamlit run streamlit_app.py

🧩 How Hybrid Retrieval Works

The system combines:
BM25 lexical scores (keyword relevance)
FAISS semantic scores (meaning relevance)
Weighted fusion to rank documents more accurately
This hybrid approach significantly outperforms pure semantic search or pure BM25-only search.

🤖 Summarization Logic

Retrieved chunks are passed to an OpenAI prompt:

"You are a document summarizer. Based on the retrieved context, 
summarize the information in a concise and query-focused manner."


The LLM generates:
Coherent summaries
Query-relevant insights
Optional adjustable summary length

🧪 Evaluation
Retrieval Accuracy: BM25 + FAISS fusion tested against sample queries
Summary Quality: Assessed using:
ROUGE scores
Human evaluation
Relevance scoring

🎨 User Interface (Streamlit)
Provides:
Search bar
Parameter controls (top_k, summary length)
Chunk preview
Final summarized answer

       
