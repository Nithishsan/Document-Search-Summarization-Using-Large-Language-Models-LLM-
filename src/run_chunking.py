from chunker import create_chunks_from_processed
import json

print("Chunking started...")

chunks = create_chunks_from_processed()
print(f"Total chunks created: {len(chunks)}")

with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

print("Chunking complete!")
