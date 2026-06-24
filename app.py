import gradio as gr
import os
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_existing()  # Load the index we already built

def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""

    try:
        answer, citations = pipeline.query(question)

        citation_text = "### Sources\n"
        for c in citations:
            citation_text += f"**[{c['ref']}]** {c['file']}, Page {c['page']} (score: {c['score']})\n\n"
            citation_text += f"> {c['snippet']}\n\n"

        return answer, citation_text
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks(title="RAG Research Assistant") as demo:
    gr.Markdown("# Research Paper Q&A")
    gr.Markdown("Ask questions across 5 foundational ML papers. Answers include citations to exact pages.")

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="What optimizer did the authors use in the Transformer paper?",
            lines=2
        )

    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_output = gr.Markdown(label="Answer")

    citation_output = gr.Markdown(label="Sources")

    ask_btn.click(
        answer_question,
        inputs=question_input,
        outputs=[answer_output, citation_output]
    )

if __name__ == "__main__":
    demo.launch()