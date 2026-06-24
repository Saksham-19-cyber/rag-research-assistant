from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

if __name__ == "__main__":
    embedder = get_embedder()
    test = embedder.embed_query("What is attention mechanism?")
    print(f"Embedding dimension: {len(test)}")
    print(f"First 5 values: {test[:5]}")