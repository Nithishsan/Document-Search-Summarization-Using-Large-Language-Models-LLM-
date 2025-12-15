import json
import re
from pathlib import Path

def clean_text(text: str) -> str:
    """Remove extra whitespace, control characters, and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\x0c', ' ', text)
    return text.strip()

def load_arxiv_json(path="data/raw/arxiv_data.json", limit=5000):
    """Loads only the first N ArXiv records to avoid huge dataset."""
    docs = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= limit:
                break
            entry = json.loads(line)
            doc = {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "text": clean_text(entry.get("abstract", "")),
            }
            docs.append(doc)
            count += 1

    return docs

def save_clean_docs(docs, out_dir="data/processed"):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    for doc in docs:
        out_path = Path(out_dir) / f"{doc['id']}.txt"
        out_path.write_text(doc["text"], encoding="utf-8")
