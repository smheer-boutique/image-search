import os
import faiss
import torch
import clip
import numpy as np
from PIL import Image
from pymongo import MongoClient

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_FOLDER = "data/images"
INDEX_FILE = "faiss.index"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CLIP MODEL
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -----------------------------
# MONGODB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["image_search"]
collection = db["products"]
collection.delete_many({})

# -----------------------------
# FAISS INDEX
# -----------------------------
dimension = 512
index = faiss.IndexFlatIP(dimension)

embeddings = []
metadata = []

print("Building index...")

for category in os.listdir(IMAGE_FOLDER):

    category_path = os.path.join(IMAGE_FOLDER, category)

    if not os.path.isdir(category_path):
        continue

    for image_name in os.listdir(category_path):

        image_path = os.path.join(category_path, image_name)

        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image)

            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding_np = embedding.cpu().numpy()

            embeddings.append(embedding_np)

            metadata.append({
                "image_path": image_path,
                "category": category,
                "filename": image_name
            })

            print(f"Processed: {image_name}")

        except Exception as e:
            print("Error:", e)

# Convert to numpy
embeddings_array = np.vstack(embeddings)

index.add(embeddings_array)
faiss.write_index(index, INDEX_FILE)

# Insert metadata in same order
collection.insert_many(metadata)

print("Index built successfully!")
