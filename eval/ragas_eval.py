from groq import Groq
import os
from dotenv import load_dotenv
from src.pipeline import RAGPipeline

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

eval_samples = [
    {
        "question": "What optimizer was used in the Transformer paper?",
        "ground_truth": "Adam optimizer with β1=0.9, β2=0.98 and ε=10^-9"
    },
    {
        "question": "How does BERT differ from a standard Transformer?",
        "ground_truth": "BERT uses bidirectional self-attention, standard Transformer uses left-to-right attention"
    },
    {
        "question": "What is the role of the document retriever in RAG?",
        "ground_truth": "Returns top-K relevant documents from a corpus as context for generation"
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Allows model to jointly attend to information from different representation subspaces at different positions"
    },
    {
        "question": "How does FAISS perform billion scale similarity search?",
        "ground_truth": "Uses GPU acceleration with optimized brute force and approximate nearest neighbor search using product quantization"
    },
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

pipeline = RAGPipeline()
pipeline.load_existing()

print("Running evaluation...\n")
faithfulness_scores = []
relevancy_scores = []

for sample in eval_samples:
    answer, citations = pipeline.query(sample["question"])
    context = " ".join([c["snippet"] for c in citations])

    f_score = score_faithfulness(sample["question"], answer, context)
    r_score = score_relevancy(sample["question"], answer)

    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)

    print(f"Q: {sample['question'][:60]}")
    print(f"A: {answer[:100]}")
    print(f"Faithfulness: {f_score} | Relevancy: {r_score}\n")

print("=" * 40)
print(f"Avg Faithfulness:  {sum(faithfulness_scores)/len(faithfulness_scores):.2f}")
print(f"Avg Relevancy:     {sum(relevancy_scores)/len(relevancy_scores):.2f}")