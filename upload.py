from flask import Blueprint, request, render_template
import os

upload_bp = Blueprint('upload', __name__)

@upload_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        user_id = request.form.get("user_id")

        if not file or not user_id:
            return "Missing file or user ID", 400

        user_dir = f"user_data/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        file.save(os.path.join(user_dir, "knowledge.txt"))
        return f"File uploaded successfully for user {user_id}"

    return render_template("upload.html")
