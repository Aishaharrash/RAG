from flask import Blueprint, request, jsonify
from rag_utils import handle_customer_message
from model_utils import load_model_and_tokenizer
import os

predict_bp = Blueprint('predict', __name__)
tokenizer, model, device = load_model_and_tokenizer()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        customer_email = data.get("message")
        user_id = data.get("user_id")  # Get user_id to load their knowledge base

        if not customer_email or not user_id:
            return jsonify({"error": "Missing message or user_id"}), 400

        # Load user's knowledge base
        kb_path = f"user_data/{user_id}/knowledge.txt"
        if not os.path.exists(kb_path):
            return jsonify({"error": "Knowledge base not found for user"}), 404

        with open(kb_path, 'r') as f:
            kb_chunks = f.readlines()

        # Dummy embedder/index (replace with real)
        embedder, index = None, None

        response = handle_customer_message(
            message=customer_email,
            tokenizer=tokenizer,
            model=model,
            device=device,
            kb_chunks=kb_chunks,
            index=index,
            embedder=embedder
        )

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
