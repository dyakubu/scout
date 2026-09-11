"""Guards clip.py's embeddings against silent numerical regressions.

Nothing in the ONNX path raises when it goes wrong. A mismatched
preprocessing step, a resample filter that isn't bicubic, an input padded
the way some other CLIP export expects, or a quantized model file that
simply doesn't survive quantization all produce well-formed 512-float
vectors that are quietly worse or outright meaningless - during this
backend switch, one candidate model returned vectors that ranked every
query identically, with no error anywhere.

So these tests don't check shapes and finiteness, which would pass in all
of those cases. They check the embeddings against reference vectors in
testdata/reference.json, captured from the original torch
CLIPModel/CLIPProcessor implementation this file's subject replaced.

Run with `uv run pytest` from media/. The model isn't a test fixture -
it's the ~126MB one that ships in scout's release archive - so these skip
when it's absent rather than failing a checkout that doesn't have it.
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

# The ONNX towers are quantized (q4f16), so they don't reproduce the fp32
# torch reference bit for bit - they track it at 0.938-0.987 cosine. This
# threshold sits below the observed spread but far above a broken pipeline,
# which lands near zero: it's here to catch "meaningless", not "different
# in the last decimal place".
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
    """scout compares these with a plain dot product (see
    search/searcher.go), which is only cosine similarity if both sides are
    unit length."""
    sessions, tokenizer = model

    vectors = [
        embed_image(sessions, str(TESTDATA / "gradient.png")),
        embed_text(sessions, tokenizer, "a colorful abstract gradient"),
    ]

    for vector in vectors:
        assert len(vector) == 512
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-5)


def test_distinct_inputs_produce_distinct_embeddings(model):
    """A collapsed model - one that returns near-identical vectors for
    everything - still passes a reference check per input if the reference
    itself were regenerated from it. This catches that independently."""
    sessions, _ = model

    gradient = embed_image(sessions, str(TESTDATA / "gradient.png"))
    checker = embed_image(sessions, str(TESTDATA / "checker.png"))

    assert cosine(gradient, checker) < 0.95


def test_truncated_query_stays_terminated(model):
    """The text tower finds the end of the sequence by argmax over
    input_ids, so <|endoftext|> has to survive truncation of an over-length
    query - otherwise argmax lands on an arbitrary token and pools the
    wrong position, silently. That it survives is a property of
    tokenizer.json's post-processor (which adds special tokens after
    truncating), not of clip.py, so it's worth pinning here: a tokenizer
    change could take it away without anything else noticing."""
    sessions, tokenizer = model

    prefix = "a photograph of a mountain range at sunrise with clouds below the peaks"
    long_query = prefix + " and" + " more words about the scene" * 20

    ids = _encode(tokenizer, long_query)[0]

    assert len(ids) == CONTEXT_LENGTH
    assert ids[-1] == tokenizer.token_to_id(EOT_TOKEN)

    # And the embedding it produces is still about the query: closer to its
    # own prefix than to an unrelated one. (Not *very* close to the prefix -
    # 60 of its 77 tokens are the repeated filler, so it shouldn't be.)
    truncated = embed_text(sessions, tokenizer, long_query)
    to_prefix = cosine(truncated, embed_text(sessions, tokenizer, prefix))
    to_unrelated = cosine(truncated, embed_text(sessions, tokenizer, "a bowl of soup on a wooden table"))

    assert to_prefix > to_unrelated
