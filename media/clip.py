"""CLIP image/text embedding: model loading and inference.

Shared by main.py (a standalone CLI for manual testing) and worker.py (the
stdio server scout's Go side actually talks to) - neither imports from the
other's entry point.

Inference runs through ONNX Runtime, not PyTorch. The model is the same one
either way (CLIP ViT-B/32), but torch is ~557MB installed against
onnxruntime's ~77MB, and torch's fp32 checkpoint is ~605MB against ~126MB
for the quantized ONNX towers - which is the difference between a media worker
that can realistically ship inside scout's release archive and one that
can't. It also cuts worker startup from ~2.4s to ~0.5s, which scout pays on
every `find` (see search.Searcher.searchMedia: the query embed blocks on
the worker's model load).

What torch/transformers were doing for free and this file now does itself:
CLIPProcessor's image preprocessing (preprocess_image) and tokenization
(_encode). Both are reimplemented to match transformers' CLIPProcessor
exactly rather than approximately - see the comments on each - because a
subtly wrong preprocessing pipeline doesn't raise, it just quietly returns
worse embeddings. test_clip.py guards that with reference vectors captured
from the original torch implementation.
"""

import os
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

# Where these files come from, for whoever assembles the release archive:
# Xenova's repo, not openai's, since it holds ONNX exports of exactly the
# same openai/clip-vit-base-patch32 weights and openai's own repo doesn't
# publish any. Nothing here fetches them - see load_model.
#
# q4f16 (4-bit weights, fp16 activations) is the pick of the ten
# quantizations the repo publishes for each tower, measured against this
# file's original torch implementation on the same images and queries:
#
#     tower    variant   size     cosine vs torch    latency
#     vision   int8      89.1MB   0.911-0.954        11ms/image
#     vision   q4f16     53.3MB   0.938-0.967        28ms/image
#     text     int8      64.5MB   0.878-0.923        21ms/query
#     text     q4f16     72.5MB   0.980-0.987        21ms/query
#     text     fp16     127.3MB   1.0000             21ms/query
#
# So q4f16 is both smaller and closer to torch than int8 on the vision
# side, and far closer on the text side for 8MB more. The vision tower's
# 2.5x slower inference is the one real cost, and it's paid per indexed
# image against ~15ms of decode and preprocessing either way.
#
# The text tower's fidelity matters more than its size: it embeds the
# query, which every media result is then ranked against, and it runs once
# per `scout find` rather than once per file. int8 was measurably the weak
# link there - it ranked near-tied results differently from torch - which
# is what ruled it out despite being the smaller file.
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

# CLIP's fixed context length. This export accepts a variable sequence
# length, but every input is padded to 77 anyway: that's what CLIP was
# trained with, and it keeps a query's embedding independent of how many
# tokens the query happened to produce.
CONTEXT_LENGTH = 77

# CLIPTokenizer's pad token is "!" (id 0). The text tower finds the end-of-
# text position itself by taking the argmax over input_ids, which lands on
# <|endoftext|> (49407) as long as padding is a lower id - so padding needs
# no attention mask, and this export doesn't accept one.
PAD_ID = 0
EOT_TOKEN = "<|endoftext|>"


class Sessions(NamedTuple):
    """The two CLIP towers, loaded as separate ONNX Runtime sessions.

    Separate rather than one combined model.onnx: indexing only ever runs
    the vision tower and searching only ever runs the text tower, so
    splitting them keeps each call's graph to what it actually needs.
    """

    vision: ort.InferenceSession
    text: ort.InferenceSession


def load_model(model_dir: str) -> tuple[Sessions, Tokenizer]:
    """Loads CLIP from model_dir, which must already contain the model.

    model_dir is scout's media.model_dir config value - the directory this
    worker is responsible for resolving its own model files within (see
    config/config.go's MediaConfig). The model ships in scout's release
    archive alongside the binary and the text embedder's own model; it is
    never fetched at runtime.

    That's a hard requirement, not a default: scout makes no network calls
    at all, at any point, and a missing-model fallback that quietly
    downloaded one would break that guarantee exactly when a user is least
    expecting it. Missing files are an error naming every one of them, so
    an incomplete archive is obvious immediately rather than after a
    silent 126MB fetch.
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
    """Which of REQUIRED_FILES aren't in model_dir, if any.

    Every file, not just one representative: a directory assembled by hand
    or a release archive unpacked partially can easily hold the 2MB
    tokenizer and be missing a 53MB model, and reporting only the first
    gap means finding out about the next one a run later.
    """
    return [
        name
        for name in REQUIRED_FILES
        if not os.path.isfile(os.path.join(model_dir, name))
    ]


def _session(model_dir: str, name: str) -> ort.InferenceSession:
    # CPU only, explicitly: scout indexes on whatever machine it's run on,
    # and letting onnxruntime pick would mean a provider that isn't there
    # on most of them.
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

    This reproduces transformers' CLIPImageProcessor step for step,
    including two details worth stating outright, since getting either
    wrong degrades embeddings silently rather than raising:

      * the resized long edge is truncated, not rounded (what
        get_resize_output_image_size does), and
      * the resample filter is bicubic (preprocessor_config.json's
        "resample": 3), not PIL's default.
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

    # HWC -> CHW, then a batch dimension of 1: one image per call, since
    # (unlike text) an image isn't chunked.
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def _encode(tokenizer: Tokenizer, text: str) -> np.ndarray:
    # The text tower locates the end of the sequence by argmax over
    # input_ids, so <|endoftext|> has to survive truncation - if it didn't,
    # argmax would land on an arbitrary token and pool the wrong position.
    # It does: the tokenizer's post-processor adds the special tokens after
    # truncating, so an over-length query comes back truncated and still
    # terminated. test_clip.py asserts that, since it's a property of
    # tokenizer.json rather than of anything in this file.
    return np.array([tokenizer.encode(text).ids], dtype=np.int64)


def _normalize(features: np.ndarray) -> list[float]:
    # L2-normalize so cosine similarity can be computed as a plain dot
    # product downstream - the same convention the text embedder already
    # uses (see embedder/local_embedder.go's meanPool).
    vector = features[0]
    return (vector / np.linalg.norm(vector)).tolist()
