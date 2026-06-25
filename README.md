---
title: RAG Research Assistant
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
python_version: "3.11"
---
# RAG Research Assistant

A retrieval-augmented generation (RAG) system that lets you ask natural language questions across multiple ML research papers and receive cited, grounded answers pointing to exact pages.

Built from scratch as a portfolio project demonstrating end-to-end ML engineering: document ingestion, vector search, LLM integration, query optimization, and quantitative evaluation.

---

## What It Does

- Upload any set of research PDFs and index them into a FAISS vector store
- Ask questions in plain English
- Get answers grounded strictly in the papers, with citations to exact filenames and page numbers
- Automatically rewrites vague queries into technical search terms for better retrieval

---

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Loading & Chunking | LangChain + PyMuPDF |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| Query Rewriting | Groq — `llama-3.1-8b-instant` |
| Answer Generation | Groq — `llama-3.1-8b-instant` |
| Frontend | Gradio |
| Evaluation | Custom faithfulness + relevancy scorer via Groq |

---

## Project Structure

```
rag-research-assistant/
├── src/
│   ├── ingestion.py       # PDF loading and text chunking
│   ├── embedder.py        # HuggingFace sentence embeddings
│   ├── vector_store.py    # FAISS index build and load
│   ├── retriever.py       # Query rewriting + similarity search
│   ├── generator.py       # Groq LLM answer generation with citations
│   └── pipeline.py        # End-to-end pipeline class
├── eval/
│   └── ragas_eval.py      # Faithfulness and relevancy evaluation
├── papers/                # Place your PDF files here
├── faiss_index/           # Auto-generated vector index (after ingestion)
├── app.py                 # Gradio web interface
├── requirements.txt
└── .env                   # API keys (not committed)
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/Saksham-19-cyber/rag-research-assistant.git
cd rag-research-assistant
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 4. Add research papers

Place PDF files into the `papers/` folder. This project was built and tested on:

| Paper | arXiv ID |
|-------|----------|
| Attention Is All You Need | 1706.03762 |
| BERT | 1810.04805 |
| RAG (Retrieval-Augmented Generation) | 2005.11401 |
| LLaMA 2 | 2307.09288 |
| FAISS: Billion-Scale Similarity Search | 1702.08734 |

Download directly: `https://arxiv.org/pdf/<arxiv_id>`

---

## Running the App

### Step 1 — Build the vector index (first time only)

```bash
python -m src.vector_store
```

This loads all PDFs from `papers/`, chunks them, embeds them, and saves the FAISS index to `faiss_index/`.

### Step 2 — Launch the UI

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Sample Questions to Try

```
What optimizer was used in the Transformer paper?
How does BERT differ from a standard Transformer?
What is the role of the document retriever in RAG?
What is multi-head attention and why is it useful?
How does FAISS perform billion-scale similarity search?
What training hardware was used for the Transformer?
How does LLaMA 2 differ from LLaMA 1?
What loss function does BERT use for pre-training?
```

---

## Evaluation Results

Evaluated on 20 question-answer pairs across 5 papers using a custom LLM-based scorer (Groq):

| Metric | Baseline (no rewriting) | Optimized (with rewriting) | Delta |
|--------|------------------------|---------------------------|-------|
| Faithfulness | 0.40 | 0.60 | +0.20 |
| Answer Relevancy | 0.77 | 0.74 | -0.03 |

Query rewriting produced a 50% relative improvement in faithfulness — meaning retrieved chunks
more reliably contain the information used in the final answer. Relevancy difference is within
noise margin for this eval set size.

---

## Key Design Decisions

**Query Rewriting:** The system uses a lightweight LLM call to expand user queries into technical terminology before retrieval. This bridges the semantic gap between natural language questions and academic paper text — for example, converting *"What optimizer did they use?"* into *"Adam optimizer beta learning rate"* before searching the vector store.

**Chunk Size:** Documents are split into 512-token chunks with 64-token overlap. This balances retrieval precision (smaller chunks = less noise) against context completeness (overlap prevents answers from being cut at boundaries).

**Embedding Model:** `all-MiniLM-L6-v2` was chosen for CPU compatibility and speed. It produces 384-dimensional embeddings and runs inference in under 1 second per query on a standard laptop.

**Score Filtering:** Chunks with FAISS L2 distance above 1.5 are filtered out as insufficiently relevant, preventing low-quality context from reaching the LLM.

---

## Limitations

- Retrieval quality depends on semantic similarity between query and chunk text; highly technical or formula-heavy content may not retrieve well without query rewriting
- General-purpose embeddings (`all-MiniLM-L6-v2`) are not fine-tuned on academic text; a domain-specific model would improve faithfulness scores
- No persistent session state — the index must be pre-built before launching the UI

---

## License

MIT
