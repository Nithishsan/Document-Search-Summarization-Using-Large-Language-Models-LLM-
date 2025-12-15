from src.preprocess import load_arxiv_json, save_clean_docs

docs = load_arxiv_json()
print(f"Loaded {len(docs)} documents")

save_clean_docs(docs)
print("Saved cleaned documents to data/processed/")
