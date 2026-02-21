import os
import json
import faiss
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pymongo import MongoClient

# ---------------- APP ----------------
app = FastAPI(title="Image Based Product Search")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "image_index.faiss")
# IMAGE_PATHS_PATH = os.path.join(BASE_DIR, "image_paths.json")

# ---------------- STATIC FILES ----------------
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---------------- LOAD FAISS ----------------
if not os.path.exists(FAISS_INDEX_PATH):
    raise RuntimeError("FAISS index not found. Run build_index_and_db.py first.")

# if not os.path.exists(IMAGE_PATHS_PATH):
#     raise RuntimeError("image_paths.json not found.")

index = faiss.read_index(FAISS_INDEX_PATH)

# with open(IMAGE_PATHS_PATH, "r") as f:
#     image_paths = json.load(f)

# ---------------- MONGODB ----------------
client = MongoClient("mongodb://localhost:27017")
db = client["image_search"]   
collection = db["products"]  

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image: Image.Image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    vector = image.flatten().astype("float32")

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector.reshape(1, -1)

# ---------------- SEARCH API ----------------
@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    query_vector = extract_features(image)

    k = 6
    distances, indices = index.search(query_vector, k)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        # 🔥 Fetch from MongoDB using faiss_id
        product = collection.find_one({"faiss_id": int(idx)})

        if not product:
            continue

        similarity = max(0, 100 - (dist * 100))

        results.append({
            "image": product.get("name"),
            "url": product.get("image_path"),
            "category": product.get("category"),
            "price": product.get("price"),
            "description": product.get("description"),
            "similarity": f"{round(similarity, 2)}%"
        })

    return {"results": results}


# ---------------- FRONTEND ----------------
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
