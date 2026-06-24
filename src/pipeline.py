from src.ingestion import load_and_chunk
from src.embedder import get_embedder
from src.vector_store import build_index, load_index
from src.retriever import retrieve
from src.generator import generate_answer
import os

class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_store = None

    def ingest(self, pdf_paths: list):
        chunks = load_and_chunk(pdf_paths)
        self.vector_store = build_index(chunks, self.embedder)
        return f"Ingested {len(pdf_paths)} PDFs → {len(chunks)} chunks indexed."

    def load_existing(self):
        self.vector_store = load_index(self.embedder)
        return "Existing index loaded."

    def query(self, question: str, k=8):
        if self.vector_store is None:
            raise RuntimeError("No index loaded. Call ingest() or load_existing() first.")
        retrieved = retrieve(question, self.vector_store, k=k)
        answer, citations = generate_answer(question, retrieved)
        return answer, citations