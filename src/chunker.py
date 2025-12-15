from pathlib import Path

def chunk_text(text, chunk_size=3500, overlap=500):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def create_chunks_from_processed(folder="data/processed", max_chunks=20000):
    all_chunks = []

    for idx, p in enumerate(Path(folder).glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        doc_chunks = chunk_text(text)

        for cidx, chunk in enumerate(doc_chunks):
            all_chunks.append({
                "doc_id": p.stem,
                "chunk_id": cidx,
                "text": chunk
            })

            if len(all_chunks) >= max_chunks:
                print(f"Reached max_chunks={max_chunks}. Stopping.")
                return all_chunks

        if idx % 500 == 0:
            print(f"Processed {idx} source documents...")

    return all_chunks