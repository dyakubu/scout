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

# The file CLIPModel.from_pretrained actually needs to load weights
WEIGHTS_FILENAME = "pytorch_model.bin"


def load_model(model_dir: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Loads CLIP from model_dir, downloading it there first if missing.

    model_dir is scout's media.model_dir config value - the directory this
    worker is responsible for resolving its own model files within (see
    config/config.go's MediaConfig).
    """
    if not os.path.isfile(os.path.join(model_dir, WEIGHTS_FILENAME)):
        # Gate on the weights file, not config.json: a run interrupted
        # mid-download (killed process, lost network) can leave config.json
        # and the other small files fully written while pytorch_model.bin
        # is still partial or absent, since snapshot_download fetches files
        # one at a time and config.json is tiny. Checking the weights file
        # directly means an interrupted download is retried next time
        # instead of looking permanently "complete". snapshot_download
        # itself only renames a file into place once its download finishes,
        # so an interrupted attempt never leaves a same-named partial file
        # behind to fool this check.
        #
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

    return _normalize(features)


def embed_text(model: CLIPModel, processor: CLIPProcessor, text: str) -> list[float]:
    """Embeds a search query into CLIP's joint image/text space, so it's
    directly comparable (cosine similarity) to embed_image's output for
    stored images - this is the only way a text string becomes comparable
    to a CLIP image embedding at all. Indexed text file chunks are still
    embedded entirely separately, by the text embedder in embedder/ - this
    function is only ever used to query vec_media, never vec_chunks.
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Same BaseModelOutputWithPooling quirk as get_image_features - see
        # CLIPModel.get_text_features's source.
        features = model.get_text_features(**inputs).pooler_output

    return _normalize(features)


def _normalize(features):
    # L2-normalize so cosine similarity can be computed as a plain dot
    # product downstream - the same convention the text embedder already
    # uses (see embedder/local_embedder.go's meanPool).
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].tolist()
