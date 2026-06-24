from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(query: str) -> str:
    """Expand user query into technical terms for better retrieval."""
    prompt = f"""Convert this question into 2-3 specific technical search terms that would appear in an ML research paper. 
Output only the search terms, no explanation, no punctuation.

Question: {query}
Search terms:"""
    
    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"[Query rewriter] '{query}' → '{rewritten}'")
    return rewritten

def retrieve(query: str, vector_store, k=8):
    rewritten = rewrite_query(query)
    results = vector_store.similarity_search_with_score(rewritten, k=k)
    filtered = [(doc, score) for doc, score in results if score < 1.5]
    return filtered

if __name__ == "__main__":
    from src.embedder import get_embedder
    from src.vector_store import load_index

    embedder = get_embedder()
    vector_store = load_index(embedder)

    query = "What is the attention mechanism?"
    results = retrieve(query, vector_store)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks\n")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] Score: {round(score, 4)}")
        print(f"     Source: {doc.metadata['source']}, Page: {doc.metadata['page']+1}")
        print(f"     Snippet: {doc.page_content[:150]}\n")