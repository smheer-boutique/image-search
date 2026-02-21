import faiss
import torch
import clip
import numpy as np
from PIL import Image
from pymongo import MongoClient

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
index = faiss.read_index("faiss.index")

# -----------------------------
# MONGODB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["image_search"]
collection = db["products"]


def search_similar(image_path, top_k=5):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding_np = embedding.cpu().numpy()

    D, I = index.search(embedding_np, top_k)

    results = []
    all_docs = list(collection.find())

    for rank, idx in enumerate(I[0]):
        doc = all_docs[int(idx)].copy()
        
        similarity_score = float(D[0][rank])

        doc["similarity"] = similarity_score
        results.append(doc)


    return results
