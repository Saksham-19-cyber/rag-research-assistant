from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def format_context(retrieved_chunks):
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_chunks):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        filename = source.split("\\")[-1]
        context_parts.append(
            f"[{i+1}] (File: {filename}, Page: {page+1})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

def generate_answer(query: str, retrieved_chunks: list):
    if not retrieved_chunks:
        return "No relevant content found in the uploaded papers.", []

    context = format_context(retrieved_chunks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
For every claim you make, cite the source using [1], [2], etc.
If the context does not contain enough information, say so explicitly — do not hallucinate.

Context:
{context}

Question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    citations = []
    for i, (doc, score) in enumerate(retrieved_chunks):
        citations.append({
            "ref": i + 1,
            "file": doc.metadata.get("source", "").split("\\")[-1],
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:200] + "...",
            "score": round(float(score), 3)
        })

    return answer, citations

if __name__ == "__main__":
    from src.embedder import get_embedder
    from src.vector_store import load_index
    from src.retriever import retrieve

    embedder = get_embedder()
    vector_store = load_index(embedder)

    query = "What is the attention mechanism?"
    retrieved = retrieve(query, vector_store)
    answer, citations = generate_answer(query, retrieved)

    print(f"Answer:\n{answer}\n")
    print("Citations:")
    for c in citations:
        print(f"  [{c['ref']}] {c['file']}, Page {c['page']} (score: {c['score']})")