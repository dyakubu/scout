import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_REPO = "openai/clip-vit-base-patch32"

# Default when no --model-dir/SCOUT_MEDIA_MODEL_DIR is given: a directory
# next to this script, not the current working directory, so behavior
# doesn't depend on where the script happens to be invoked from.
DEFAULT_MODEL_DIR = Path(__file__).parent / "model"


def load_model(model_dir: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Loads CLIP from model_dir, downloading it there first if missing.

    model_dir is scout's media.model_dir config value - the directory this
    worker is responsible for resolving its own model files within (see
    config/config.go's MediaConfig).
    """
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        snapshot_download(repo_id=MODEL_REPO, local_dir=model_dir)

    model = CLIPModel.from_pretrained(model_dir)
    model.eval()

    processor = CLIPProcessor.from_pretrained(model_dir)

    return model, processor


def embed_image(model: CLIPModel, processor: CLIPProcessor, image_path: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # L2-normalize so cosine similarity can be computed as a plain dot
    # product downstream - the same convention the text embedder already
    # uses (see embedder/local_embedder.go's meanPool), needed if
    # image and text embeddings are ever compared against each other.
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Embed an image with CLIP")
    parser.add_argument("image", help="path to the image file")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SCOUT_MEDIA_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="directory to load/download the CLIP model from",
    )
    args = parser.parse_args()

    model, processor = load_model(args.model_dir)
    embedding = embed_image(model, processor, args.image)

    print(f"embedding dim: {len(embedding)}")
    print(embedding)


if __name__ == "__main__":
    main()
