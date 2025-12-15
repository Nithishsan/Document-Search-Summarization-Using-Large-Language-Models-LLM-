from openai import OpenAI

client = OpenAI()

def summarize_chunks(
    query,
    chunks,
    length="medium"
):
    """
    query: user query
    chunks: retrieved chunks (list of dicts)
    length: short | medium | long
    """

    length_map = {
        "short": "3-4 sentences",
        "medium": "1 short paragraph",
        "long": "2-3 detailed paragraphs"
    }

    context = "\n\n".join(
        f"- {c['text']}" for c in chunks
    )

    prompt = f"""
You are an expert technical summarizer.

User Query:
{query}

Relevant Document Excerpts:
{context}

Task:
Generate a {length_map[length]} summary that directly answers the query.
The summary must be factual, coherent, and based only on the provided text.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
