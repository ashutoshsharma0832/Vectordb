import os
import gradio as gr
from datetime import datetime
from typing import List, Dict
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

#CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "...")
client = Groq(api_key=GROQ_API_KEY)

# Persistent ChromaDB client
vectordb = chromadb.PersistentClient(path="./chroma_storage", settings=Settings(allow_reset=True))
collection = vectordb.get_or_create_collection("document_chunks")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#DOCUMENT PROCESSING

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    return "Unsupported file format"

def chunk_text(text: str, chunk_size=300, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def store_document(file_path):
    text = extract_text(file_path)
    chunks = chunk_text(text)
    ids = []
    for i, chunk in enumerate(chunks):
        vector = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"{os.path.basename(file_path)}_{i}"]
        )
        ids.append(f"{os.path.basename(file_path)}_{i}")
    return f"Uploaded and indexed {len(ids)} chunks from {os.path.basename(file_path)}"

#VECTOR SEARCH

def get_context(query: str, top_k=3):
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""

#CHAT HANDLING

def generate_reply(message: str, history: List[Dict], model_choice: str, max_tokens: int):
    context = get_context(message)
    system_prompt = {
        "role": "system",
        "content": f"You are an assistant that answers questions based on uploaded documents.\n\nRelevant Info:\n{context}"
    }

    messages = [system_prompt] + history + [{"role": "user", "content": message}]
    try:
        chat_completion = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        full_response = ""
        for chunk in chat_completion:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                yield full_response

    except Exception as e:
        yield f"Error: {e}"

#GRADIO UI

models = {
    "gemma2-9b-it": "Gemma2-9b-it",
    "llama3-8b-8192": "LLaMA3-8b-8192",
    "llama3-70b-8192": "LLaMA3-70b-8192",
    "mixtral-8x7b-32768": "Mixtral-8x7b-Instruct-v0.1"
}

with gr.Blocks(title="Vector DB Chatbot") as demo:
    gr.Markdown("## Document Reader Bot")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".csv", ".txt", ".docx"],
                type="filepath"
            )
            upload_status = gr.Textbox(label="", interactive=False)

        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                label="Choose Groq Model",
                choices=list(models.keys()),
                value="llama3-8b-8192"
            )
            token_slider = gr.Slider(
                label="Max Tokens",
                minimum=512,
                maximum=32768,
                step=512,
                value=4096
            )
            chatbot = gr.Chatbot(label="Chat with Document", height=500)
            user_input = gr.Textbox(
                placeholder="Ask a question about your document...",
                show_label=False
            )

    history_state = gr.State([])

    def handle_user_input(message, history, model_choice, max_tokens):
        if not message.strip():
            return history, "", history

        history = history or []
        stream = generate_reply(message, history, model_choice, max_tokens)
        full_reply = ""
        for partial in stream:
            full_reply = partial
            updated_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_reply}
            ]
            formatted = [
                [updated_history[i]["content"], updated_history[i + 1]["content"]]
                for i in range(0, len(updated_history), 2)
            ]
            yield formatted, "", updated_history

    user_input.submit(
        fn=handle_user_input,
        inputs=[user_input, history_state, model_dropdown, token_slider],
        outputs=[chatbot, user_input, history_state]
    )

    file_upload.change(
        fn=store_document,
        inputs=[file_upload],
        outputs=[upload_status]
    )

demo.launch(share=True)

