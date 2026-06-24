import time
from groq import Groq
import os
from dotenv import load_dotenv
from src.pipeline import RAGPipeline
from src.embedder import get_embedder
from src.vector_store import load_index
from src.retriever import retrieve
from src.generator import generate_answer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

eval_samples = [
    {"question": "What optimizer was used in the Transformer paper?", "ground_truth": "Adam optimizer with β1=0.9, β2=0.98 and ε=10^-9"},
    {"question": "How does BERT differ from a standard Transformer?", "ground_truth": "BERT uses bidirectional self-attention, standard Transformer uses left-to-right attention"},
    {"question": "What is the role of the document retriever in RAG?", "ground_truth": "Returns top-K relevant documents from a corpus as context for generation"},
    {"question": "What is multi-head attention?", "ground_truth": "Allows model to jointly attend to information from different representation subspaces at different positions"},
    {"question": "How does FAISS perform billion scale similarity search?", "ground_truth": "Uses GPU acceleration with optimized brute force and approximate nearest neighbor search using product quantization"},
    {"question": "What masking strategy does BERT use during pre-training?", "ground_truth": "Masked Language Model - randomly masks 15% of tokens and predicts them"},
    {"question": "What is the BLEU score of the Transformer on English to German translation?", "ground_truth": "28.4 BLEU score on WMT 2014 English-to-German translation task"},
    {"question": "How many attention heads does the base Transformer use?", "ground_truth": "8 attention heads"},
    {"question": "What is product quantization in FAISS?", "ground_truth": "A compression technique that splits vectors into sub-vectors and quantizes each separately for memory-efficient approximate search"},
    {"question": "What datasets were used to evaluate RAG?", "ground_truth": "Natural Questions, TriviaQA, WebQuestions and CuratedTrec"},
    {"question": "What is the context window size used in LLaMA 2?", "ground_truth": "4096 tokens"},
    {"question": "What is grouped query attention in LLaMA 2?", "ground_truth": "An attention mechanism that shares key and value heads across multiple query heads to reduce memory usage during inference"},
    {"question": "What is the warmup schedule used in the Transformer?", "ground_truth": "Linear warmup for 4000 steps followed by inverse square root decay"},
    {"question": "What is the Next Sentence Prediction task in BERT?", "ground_truth": "A pre-training task where the model predicts whether two sentences appear consecutively in the original text"},
    {"question": "How does RAG combine retrieval and generation?", "ground_truth": "RAG marginalizes over retrieved documents using a parametric seq2seq model combined with a non-parametric dense retrieval component"},
    {"question": "What is the dimensionality of the feedforward layers in the base Transformer?", "ground_truth": "2048 dimensions"},
    {"question": "How does LLaMA 2 handle safety alignment?", "ground_truth": "Through reinforcement learning from human feedback and rejection sampling fine-tuning"},
    {"question": "What is the IVF index structure in FAISS?", "ground_truth": "Inverted file index that partitions vectors into clusters and searches only nearest clusters for approximate nearest neighbor search"},
    {"question": "What is label smoothing and why was it used in the Transformer?", "ground_truth": "A regularization technique that softens one-hot targets; used with value 0.1 to improve BLEU score despite hurting perplexity"},
    {"question": "What fine-tuning approach does BERT use for downstream tasks?", "ground_truth": "Adding a simple classification layer on top of the CLS token and fine-tuning all parameters end-to-end"},
]

def score_faithfulness(question, answer, context):
    prompt = f"""Rate whether the answer is faithful to the context (only uses information present in context).
Score 1 if faithful, 0 if it contains hallucinated information not in context.
Respond with only a single number: 0 or 1.

Context: {context}
Question: {question}
Answer: {answer}

Score:"""
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0

def score_relevancy(question, answer):
    prompt = f"""Rate how relevant the answer is to the question on a scale of 0 to 1.
1 = directly answers the question, 0 = completely irrelevant.
Respond with only a decimal number between 0 and 1.

Question: {question}
Answer: {answer}

Score:"""
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0

def retrieve_without_rewriting(question, vector_store, k=8):
    """Baseline: direct retrieval with no query rewriting."""
    results = vector_store.similarity_search_with_score(question, k=k)
    return [(doc, score) for doc, score in results if score < 1.5]

def run_eval(use_rewriting: bool):
    embedder = get_embedder()
    vector_store = load_index(embedder)

    faithfulness_scores = []
    relevancy_scores = []

    for sample in eval_samples:
        if use_rewriting:
            retrieved = retrieve(sample["question"], vector_store, k=8)
        else:
            retrieved = retrieve_without_rewriting(sample["question"], vector_store, k=8)

        answer, citations = generate_answer(sample["question"], retrieved)
        context = " ".join([c["snippet"] for c in citations])

        f_score = score_faithfulness(sample["question"], answer, context)
        r_score = score_relevancy(sample["question"], answer)
        time.sleep(4)
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        label = "WITH rewriting" if use_rewriting else "WITHOUT rewriting"
        print(f"[{label}] Q: {sample['question'][:55]}...")
        print(f"  Faithfulness: {f_score} | Relevancy: {r_score}\n")

    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_r = sum(relevancy_scores) / len(relevancy_scores)
    return avg_f, avg_r

print("=" * 50)
print("BASELINE — No Query Rewriting")
print("=" * 50)
f_base, r_base = run_eval(use_rewriting=False)
print(f"\nBaseline Faithfulness: {f_base:.2f}")
print(f"Baseline Relevancy:    {r_base:.2f}")

print("\n" + "=" * 50)
print("OPTIMIZED — With Query Rewriting")
print("=" * 50)
f_opt, r_opt = run_eval(use_rewriting=True)
print(f"\nOptimized Faithfulness: {f_opt:.2f}")
print(f"Optimized Relevancy:    {r_opt:.2f}")

print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Metric':<25} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
print(f"{'Faithfulness':<25} {f_base:>10.2f} {f_opt:>10.2f} {f_opt-f_base:>+10.2f}")
print(f"{'Answer Relevancy':<25} {r_base:>10.2f} {r_opt:>10.2f} {r_opt-r_base:>+10.2f}")