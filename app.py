from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import nltk
import os
from dotenv import load_dotenv

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load env vars
load_dotenv()

# --- MODEL SETUP ---
def load_mistral_model(model_name="mistralai/Mistral-7B-Instruct-v0.1", hf_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=hf_token
    ).to(device)
    model.eval()
    return tokenizer, model, device

# --- KNOWLEDGE BASE FUNCTIONS ---
def chunk_text_by_sentences(text, chunk_size=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def create_kb_index(text):
    chunks = chunk_text_by_sentences(text)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return chunks, index, embedder

# --- CHATBOT FUNCTIONS ---
def search_kb(query, kb_chunks, index, embedder, top_k=3):
    query_vec = embedder.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    return [kb_chunks[idx] for idx in I[0] if 0 <= idx < len(kb_chunks)]

def is_bug_report(message, tokenizer, model, device):
    prompt = f"""You are an AI that detects if a customer message reports a technical bug or problem.
Respond only with one word: 'bug_report' or 'general'.

Message: "{message}"

Response:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    return decoded == "bug_report"

def reformulate_query(message, tokenizer, model, device):
    prompt = f"""You are a helpful assistant that reformulates customer questions to match knowledge base entries.
Original: "{message}"
Rewritten:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def generate_answer(message, kb_context, tokenizer, model, device):
    if not kb_context:
        kb_context = "No relevant information found in the knowledge base."

    prompt = f"""You are a helpful customer support AI. Use the knowledge base below to answer the customer's question.

Knowledge base:
{kb_context}

Customer question:
{message}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# --- INIT ---
app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

tokenizer, model, device = load_mistral_model(hf_token=HF_TOKEN)

# Global storage (MVP only – replace with DB/session storage for production)
kb_chunks, index, embedder = [], None, None

# --- API ROUTES ---

@app.route("/")
def home():
    return "🧠 AI Support Chatbot is running!"

@app.route("/upload_kb", methods=["POST"])
def upload_kb():
    global kb_chunks, index, embedder
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        kb_chunks, index, embedder = create_kb_index(text)
        return jsonify({"message": "Knowledge base uploaded successfully.", "chunks": len(kb_chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not index:
            return jsonify({"error": "Knowledge base not uploaded yet."}), 400

        data = request.get_json(force=True)
        message = data.get("message")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        if is_bug_report(message, tokenizer, model, device):
            return jsonify({"response": "Thank you for reporting the issue. Our technical team will look into it."})

        revised_query = reformulate_query(message, tokenizer, model, device)
        context = search_kb(revised_query, kb_chunks, index, embedder)

        if not context:
            return jsonify({"response": "Sorry, I couldn't find any relevant info."})

        answer = generate_answer(message, "\n".join(context), tokenizer, model, device)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
