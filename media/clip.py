"""CLIP image/text embedding: model loading and inference.

Runs CLIP ViT-B/32 through ONNX Runtime. preprocess_image and _encode do
what transformers' CLIPProcessor would, and have to match it exactly - a
subtly wrong pipeline here returns worse embeddings rather than raising,
which is what test_clip.py exists to catch.

Shared by main.py (a standalone CLI for manual testing) and worker.py (the
stdio server scout's Go side talks to). See README.md for the reasoning
behind the runtime, the quantization, and the model's input contract.
"""

import os
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image
from tokenizers import Tokenizer

# Teaches Image.open to read HEIC/HEIF, so preprocess_image needs no
# special case for them.
pillow_heif.register_heif_opener()

# ONNX exports of openai/clip-vit-base-patch32, quantized to q4f16.
# scripts/fetch-deps.sh downloads them at build time; nothing here fetches
# anything. README.md covers the choice of repository and quantization.
MODEL_REPO = "Xenova/clip-vit-base-patch32"

VISION_MODEL_FILE = "vision_model_q4f16.onnx"
TEXT_MODEL_FILE = "text_model_q4f16.onnx"
TOKENIZER_FILE = "tokenizer.json"

REQUIRED_FILES = (VISION_MODEL_FILE, TEXT_MODEL_FILE, TOKENIZER_FILE)

# CLIP ViT-B/32's preprocessing, from the model's preprocessor_config.json:
# resize the shortest edge to 224 (bicubic), center crop to 224x224, scale
# to 0..1, then normalize per channel.
IMAGE_SIZE = 224
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# CLIP's context length. Every input is padded to it, so a query's
# embedding doesn't depend on how many tokens it produced.
CONTEXT_LENGTH = 77

# CLIPTokenizer's pad token is "!", id 0. The text tower locates the end of
# the sequence by argmax over input_ids, which finds <|endoftext|> (49407)
# because padding is a lower id - so no attention mask is needed, and this
# export doesn't accept one.
PAD_ID = 0
EOT_TOKEN = "<|endoftext|>"


class Sessions(NamedTuple):
    """The two CLIP towers, as separate ONNX Runtime sessions.

    Indexing only runs the vision tower and searching only runs the text
    tower, so each call loads just the graph it needs.
    """

    vision: ort.InferenceSession
    text: ort.InferenceSession


def load_model(model_dir: str) -> tuple[Sessions, Tokenizer]:
    """Loads CLIP from model_dir, which must already contain the model.

    model_dir is scout's media.model_dir config value. The model ships in
    the release archive beside the binary; scout makes no network calls, so
    missing files raise rather than triggering a download.
    """
    missing = missing_files(model_dir)
    if missing:
        raise FileNotFoundError(
            f"media model incomplete: {model_dir} is missing "
            + ", ".join(missing)
            + f". These ship in scout's release archive; to rebuild the directory by hand, "
            f"take them from the {MODEL_REPO} repository."
        )

    sessions = Sessions(
        vision=_session(model_dir, VISION_MODEL_FILE),
        text=_session(model_dir, TEXT_MODEL_FILE),
    )

    tokenizer = Tokenizer.from_file(os.path.join(model_dir, TOKENIZER_FILE))
    tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)
    tokenizer.enable_padding(length=CONTEXT_LENGTH, pad_id=PAD_ID, pad_token="!")

    return sessions, tokenizer


def missing_files(model_dir: str) -> list[str]:
    """Which of REQUIRED_FILES aren't in model_dir.

    Returns all of them, so a partially assembled directory is reported in
    one go rather than one file per run.
    """
    return [
        name
        for name in REQUIRED_FILES
        if not os.path.isfile(os.path.join(model_dir, name))
    ]


def _session(model_dir: str, name: str) -> ort.InferenceSession:
    # CPU explicitly: scout runs on whatever machine it's installed on, and
    # no accelerator is assumed to be there.
    return ort.InferenceSession(
        os.path.join(model_dir, name),
        providers=["CPUExecutionProvider"],
    )


def embed_image(sessions: Sessions, image_path: str) -> list[float]:
    pixel_values = preprocess_image(image_path)
    features = sessions.vision.run(["image_embeds"], {"pixel_values": pixel_values})[0]
    return _normalize(features)


def embed_text(sessions: Sessions, tokenizer: Tokenizer, text: str) -> list[float]:
    """Embeds a search query into CLIP's joint image/text space, so it's
    directly comparable (cosine similarity) to embed_image's output for
    stored images - this is the only way a text string becomes comparable
    to a CLIP image embedding at all. Indexed text file chunks are still
    embedded entirely separately, by the text embedder in embedder/ - this
    function is only ever used to query vec_media, never vec_chunks.
    """
    input_ids = _encode(tokenizer, text)
    features = sessions.text.run(["text_embeds"], {"input_ids": input_ids})[0]
    return _normalize(features)


def preprocess_image(image_path: str) -> np.ndarray:
    """Turns an image file into the vision tower's pixel_values input.

    Reproduces transformers' CLIPImageProcessor. Two details are load
    bearing: the resized long edge is truncated rather than rounded, and
    the resample filter is bicubic.
    """
    image = Image.open(image_path).convert("RGB")

    width, height = image.size
    short, long = min(width, height), max(width, height)
    new_long = int(IMAGE_SIZE * long / short)
    new_size = (new_long, IMAGE_SIZE) if width >= height else (IMAGE_SIZE, new_long)

    image = image.resize(new_size, Image.BICUBIC)

    new_width, new_height = image.size
    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD

    # HWC -> CHW, with a batch dimension of 1: an image isn't chunked, so
    # there's always exactly one per call.
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def _encode(tokenizer: Tokenizer, text: str) -> np.ndarray:
    # An over-length query stays terminated: the tokenizer adds special
    # tokens after truncating, so <|endoftext|> survives for the text tower
    # to find. That's a property of tokenizer.json, so test_clip.py pins it.
    return np.array([tokenizer.encode(text).ids], dtype=np.int64)


def _normalize(features: np.ndarray) -> list[float]:
    # L2-normalize so cosine similarity is a plain dot product downstream,
    # the same convention embedder/local_embedder.go's meanPool uses.
    vector = features[0]
    return (vector / np.linalg.norm(vector)).tolist()
