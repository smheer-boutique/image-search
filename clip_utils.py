import numpy as np
from PIL import Image, ImageOps
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)

def get_image_embedding(image: Image.Image):
    image = ImageOps.exif_transpose(image)
    image = preprocess(image).unsqueeze(0)

    with open_clip.no_grad():
        features = model.encode_image(image)

    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
