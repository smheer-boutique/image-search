from flask import Flask, request, jsonify
import os
from search import search_similar
from flask import render_template
from flask import send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/images', filename)


@app.route("/search", methods=["POST"])
def search():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    results = search_similar(file_path)

    formatted_results = []

    for item in results:
        formatted_results.append({
            "image": item["filename"],
            "category": item["category"],
            "price": item.get("price", "N/A"),
            "description": item.get("description", ""),
            "url": f"/images/{item['category']}/{item['filename']}",
            "similarity": item.get("similarity", 0.0)
        })

    return jsonify({"results": formatted_results})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

