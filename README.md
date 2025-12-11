# 🚀 Document Search & Summarization Using Large Language Models (LLM)
*A Production-Ready Hybrid Retrieval & Summarization System*

---

## 📌 Overview

This project implements a **production-style document search and summarization system** using a hybrid retrieval pipeline (BM25 + dense embeddings) combined with LLM-based abstractive summarization.

The system is designed to:
- Efficiently search thousands of documents  
- Return the most relevant chunks using hybrid retrieval  
- Generate coherent, query-focused summaries  
- Provide an interactive UI using Streamlit  

This architecture is inspired by real-world Retrieval-Augmented Generation (RAG) systems used in enterprise AI products.

---

## 🧠 Key Features

### 🔍 Hybrid Search Engine
- **BM25** for keyword-level relevance  
- **Sentence-BERT embeddings** for semantic search  
- **Weighted score fusion** ensures optimal retrieval precision & recall  

### 📄 LLM-Based Summarization
- Uses GPT models for concise, high-quality summaries  
- Adjustable summary lengths (`short`, `medium`, `long`)  
- Strict grounding to retrieved content to avoid hallucinations  

### ⚙️ Modular Pipeline
- Data preprocessing  
- Document chunking  
- Embedding generation  
- FAISS vector indexing  
- BM25 lexical indexing  
- Streamlit UI  

---

## 🏗️ System Architecture

                   ┌──────────────────────────┐
                   │        User (UI)         │
                   │  Streamlit Application   │
                   └─────────────┬────────────┘
                                 │ Query
                                 ▼
               ┌────────────────────────────────────┐
               │      Application Backend (Python)   │
               └─────────────────┬───────────────────┘
                                 │
        ┌────────────────────────────────────────────────────────┐
        │ Hybrid Retrieval Engine (Core Search Logic)            │
        │                                                        │
        │  ┌──────────────┐    ┌─────────────────┐              │
        │  │  BM25 Index   │    │  FAISS Vector   │              │
        │  │ (Lexical IR)  │    │    Index        │              │
        │  └──────────────┘    └─────────────────┘              │
        │          │                       │                    │
        │          └────── Score Fusion ───┘                    │
        │                                                        │
        └─────────────────┬──────────────────────────────────────┘
                          │ Top-K Retrieved Chunks
                          ▼
             ┌────────────────────────────────┐
             │     LLM Summarization Layer    │
             │ (OpenAI GPT-4o-mini / GPT-4o)   │
             └─────────────────┬──────────────┘
                               │ Summary
                               ▼
                   ┌──────────────────────────┐
                   │ Streamlit UI Output      │
                   └──────────────────────────┘

## 📂 Folder Structure

project/
├── data/
│ └── processed/
│ ├── chunks.json
│ ├── chunks.pkl
│ ├── embeddings.npy
│ ├── faiss_index.bin
│ └── bm25.pkl
│
├── src/
│ ├── init.py
│ ├── preprocess.py
│ ├── chunker.py
│ ├── indexer.py
│ ├── retriever.py
│ ├── summarizer.py
│ ├── run_preprocess.py
│ ├── run_chunking.py
│ ├── run_indexing.py
│ └── test_retriever.py
│
├── streamlit_app.py
└── README.md

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
git clone <repo-url>
cd project

2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set OpenAI API key
setx OPENAI_API_KEY "your_api_key_here"

📘 Pipeline Execution
Step 1 — Preprocess Documents
python -m src.run_preprocess

Step 2 — Chunk Documents
python -m src.run_chunking

Step 3 — Build Retrieval Indexes
python -m src.run_indexing

Step 4 — Test Retrieval
python -m src.test_retriever

🖥️ Streamlit Interface

Run:

streamlit run streamlit_app.py

