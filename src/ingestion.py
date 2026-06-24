from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk(pdf_paths: list, chunk_size=512, chunk_overlap=64):
    all_docs = []
    
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_documents(all_docs)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Loaded {len(all_docs)} pages → {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    # Quick test
    pdf_folder = "papers"
    pdf_paths = [
        os.path.join(pdf_folder, f) 
        for f in os.listdir(pdf_folder) 
        if f.endswith(".pdf")
    ]
    
    print(f"Found PDFs: {pdf_paths}")
    chunks = load_and_chunk(pdf_paths)
    print(f"\nSample chunk:\n{chunks[0].page_content[:300]}")
    print(f"\nMetadata: {chunks[0].metadata}")