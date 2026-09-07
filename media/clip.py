"""CLIP image embedding: model loading and inference.

Shared by main.py (a standalone CLI for manual testing) and worker.py (the
stdio server scout's Go side actually talks to) - neither imports from the
other's entry point.
"""

import os

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_REPO = "openai/clip-vit-base-patch32"


def load_model(model_dir: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Loads CLIP from model_dir, downloading it there first if missing.

    model_dir is scout's media.model_dir config value - the directory this
    worker is responsible for resolving its own model files within (see
    config/config.go's MediaConfig).
    """
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        # ignore_patterns skips the TensorFlow/Flax weight files - only
        # pytorch_model.bin is ever loaded (CLIPModel.from_pretrained
        # below), and the other two frameworks' weights are ~1.1GB of
        # pure waste otherwise.
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=model_dir,
            ignore_patterns=["*.h5", "*.msgpack"],
        )

    model = CLIPModel.from_pretrained(model_dir)
    model.eval()

    processor = CLIPProcessor.from_pretrained(model_dir)

    return model, processor


def embed_image(model: CLIPModel, processor: CLIPProcessor, image_path: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # get_image_features returns a BaseModelOutputWithPooling, not a
        # bare tensor - the projected image embedding is its pooler_output
        # (see CLIPModel.get_image_features's source: it computes the
        # vision model's pooled output, overwrites pooler_output with the
        # visual projection of it, then returns the whole output object).
        features = model.get_image_features(**inputs).pooler_output

    # L2-normalize so cosine similarity can be computed as a plain dot
    # product downstream - the same convention the text embedder already
    # uses (see embedder/local_embedder.go's meanPool), needed if
    # image and text embeddings are ever compared against each other.
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features[0].tolist()
