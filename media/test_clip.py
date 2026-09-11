"""Guards clip.py's embeddings against silent numerical regressions.

A broken preprocessing step or a badly quantized model still returns
well-formed 512-float vectors, so shape and finiteness checks would pass.
These compare against reference vectors in testdata/reference.json,
captured from the torch implementation clip.py replaced. See README.md.

Run with `uv run pytest` from media/. They skip when no model is present,
since it ships in the release archive rather than in git.
"""

import json
import math
import os
from pathlib import Path

import pytest

from clip import (
    CONTEXT_LENGTH,
    EOT_TOKEN,
    _encode,
    embed_image,
    embed_text,
    load_model,
    missing_files,
)

TESTDATA = Path(__file__).parent / "testdata"

# Quantized towers track the fp32 reference at 0.938-0.987 cosine. This
# threshold sits below that spread and far above a broken pipeline, which
# lands near zero.
MIN_COSINE = 0.90

MODEL_DIR = os.environ.get(
    "SCOUT_MEDIA_MODEL_DIR",
    str(Path(__file__).parent.parent / "models" / "media"),
)


@pytest.fixture(scope="module")
def model():
    missing = missing_files(MODEL_DIR)
    if missing:
        pytest.skip(
            f"no CLIP model in {MODEL_DIR} (missing {', '.join(missing)}) - "
            f"point SCOUT_MEDIA_MODEL_DIR at one"
        )
    return load_model(MODEL_DIR)


@pytest.fixture(scope="module")
def reference():
    with open(TESTDATA / "reference.json") as f:
        return json.load(f)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_image_embeddings_match_reference(model, reference):
    sessions, _ = model

    for name, expected in reference["images"].items():
        actual = embed_image(sessions, str(TESTDATA / name))
        similarity = cosine(actual, expected)

        assert similarity >= MIN_COSINE, (
            f"{name}: cosine {similarity:.4f} against the torch reference is "
            f"below {MIN_COSINE} - the image pipeline has drifted, not just quantized"
        )


def test_text_embeddings_match_reference(model, reference):
    sessions, tokenizer = model

    for query, expected in reference["queries"].items():
        actual = embed_text(sessions, tokenizer, query)
        similarity = cosine(actual, expected)

        assert similarity >= MIN_COSINE, (
            f"{query!r}: cosine {similarity:.4f} against the torch reference is "
            f"below {MIN_COSINE} - the text pipeline has drifted, not just quantized"
        )


def test_embeddings_are_normalized(model):
    """scout compares these with a plain dot product, which is only cosine
    similarity if both sides are unit length."""
    sessions, tokenizer = model

    vectors = [
        embed_image(sessions, str(TESTDATA / "gradient.png")),
        embed_text(sessions, tokenizer, "a colorful abstract gradient"),
    ]

    for vector in vectors:
        assert len(vector) == 512
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-5)


def test_distinct_inputs_produce_distinct_embeddings(model):
    """Catches a collapsed model, which returns near-identical vectors for
    every input and would survive a regenerated reference file."""
    sessions, _ = model

    gradient = embed_image(sessions, str(TESTDATA / "gradient.png"))
    checker = embed_image(sessions, str(TESTDATA / "checker.png"))

    assert cosine(gradient, checker) < 0.95


def test_truncated_query_stays_terminated(model):
    """<|endoftext|> must survive truncation of an over-length query, or
    the text tower's argmax pools the wrong position. That's a property of
    tokenizer.json rather than of clip.py, so a tokenizer change could take
    it away with nothing else noticing."""
    sessions, tokenizer = model

    prefix = "a photograph of a mountain range at sunrise with clouds below the peaks"
    long_query = prefix + " and" + " more words about the scene" * 20

    ids = _encode(tokenizer, long_query)[0]

    assert len(ids) == CONTEXT_LENGTH
    assert ids[-1] == tokenizer.token_to_id(EOT_TOKEN)

    # The embedding is still about the query: closer to its own prefix than
    # to an unrelated one, though not very close, since most of its tokens
    # are the repeated filler.
    truncated = embed_text(sessions, tokenizer, long_query)
    to_prefix = cosine(truncated, embed_text(sessions, tokenizer, prefix))
    to_unrelated = cosine(truncated, embed_text(sessions, tokenizer, "a bowl of soup on a wooden table"))

    assert to_prefix > to_unrelated
