from langchain_community.vectorstores import FAISS
import os

INDEX_PATH = "faiss_index"

def build_index(chunks, embedder):
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embedder)
    vector_store.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}/")
    return vector_store

def load_index(embedder):
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("No index found. Run ingestion first.")
    return FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    from src.ingestion import load_and_chunk
    from src.embedder import get_embedder
    import os

    pdf_folder = "papers"
    pdf_paths = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.endswith(".pdf")
    ]

    chunks = load_and_chunk(pdf_paths)
    embedder = get_embedder()
    vector_store = build_index(chunks, embedder)
    print(f"Total vectors in index: {vector_store.index.ntotal}")