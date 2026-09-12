# media worker

Embeds images and search queries with CLIP ViT-B/32, for scout's image
search. Runs as a subprocess of scout, one JSON job per line over stdin and
stdout.

Reads JPEG, PNG and HEIC. HEIC needs `pillow-heif`, whose opener `clip.py`
registers at import - Pillow has no HEIF support of its own, and the wheels
bundle libheif so nothing is required on the user's machine.

```
worker.py    the stdio protocol and job loop
clip.py      model loading, preprocessing, inference
main.py      standalone CLI, for embedding one image by hand
test_clip.py numerical regression tests
```

The Go side is `mediaworker.Client`, which spawns this and implements
`embedder.MediaEmbedder`.

## Setup

```
uv sync
uv run pytest
```

The model isn't in git. In a release archive it sits in `models/media/`
next to the binary; for development, `scripts/fetch-deps.sh` downloads it
there. Both the tests and `main.py` look in that directory unless
`SCOUT_MEDIA_MODEL_DIR` says otherwise.

## Why ONNX Runtime instead of PyTorch

The model is the same either way. The runtime around it is not:

|                        | torch  | onnxruntime |
| ---------------------- | ------ | ----------- |
| dependencies installed | 764MB  | 173MB       |
| model files            | 605MB  | 126MB       |
| worker startup         | ~2.4s  | ~0.5s       |

Size is what made this worth doing: at ~1.4GB installed, the worker could
not ship inside scout's release archive, and image search stayed a
development-only feature. At ~300MB it ships.

Startup matters because scout pays it per invocation. `Searcher.searchMedia`
blocks on the first job, which the worker can't answer until its model is
loaded, so ~2s of that sat on the critical path of every `scout find`.
Almost all of it was `import torch`.

The cost is that `CLIPProcessor` no longer preprocesses images and
tokenizes text for us. Both are reimplemented in `clip.py`.

## Choosing a quantization

`Xenova/clip-vit-base-patch32` publishes ten quantizations of each tower.
Measured against the original torch implementation, on the same images and
queries:

| tower  | variant | size    | cosine vs torch | latency    |
| ------ | ------- | ------- | --------------- | ---------- |
| vision | int8    | 89.1MB  | 0.911-0.954     | 11ms/image |
| vision | q4f16   | 53.3MB  | 0.938-0.967     | 28ms/image |
| text   | int8    | 64.5MB  | 0.878-0.923     | 21ms/query |
| text   | q4f16   | 72.5MB  | 0.980-0.987     | 21ms/query |
| text   | fp16    | 127.3MB | 1.0000          | 21ms/query |

q4f16 on both. On the vision side it is smaller *and* closer to torch than
int8; the 2.5x slower inference is paid per indexed image, against ~15ms of
decode and preprocessing either way.

On the text side it costs 8MB over int8 and is worth it. That tower embeds
the query every media result is ranked against, and runs once per search
rather than once per file. int8 there measurably reordered near-tied
results relative to torch.

Quantization means embeddings sit 0.94-0.99 cosine from the fp32 originals
rather than reproducing them. That is close enough to rank the same and far
enough that vectors from two different quantizations can't be compared, so
the model identity recorded with each stored vector
(`mediaModelID` in `main.go`) names the quantization too. Changing it
strands whatever is already indexed.

## Why the weights come from Xenova

They are OpenAI's CLIP ViT-B/32 weights. `openai/clip-vit-base-patch32`
publishes PyTorch, TensorFlow and Flax formats but no ONNX;
`Xenova/clip-vit-base-patch32` is a conversion of those same weights, which
is the only reason the download points there.

## Failure is silent, so the tests are numerical

Nothing in this path raises when it goes subtly wrong. A resample filter
that isn't bicubic, a sequence padded the way some other CLIP export
expects, a tower that didn't survive quantization - each returns
well-formed 512-float vectors that are quietly worse or entirely
meaningless. During the switch to ONNX, one candidate model returned
vectors that ranked every query identically, with no error anywhere.

So `test_clip.py` checks embeddings against reference vectors in
`testdata/reference.json`, captured from the torch implementation this
replaced, rather than checking shapes and finiteness. It skips when no
model is present.

## Things worth knowing before changing this

- **Preprocessing must match `CLIPImageProcessor` exactly.** Two details
  are easy to get wrong: the resized long edge is truncated rather than
  rounded, and the resample filter is bicubic rather than PIL's default.
- **The text tower takes `input_ids` only.** This export rejects an
  `attention_mask`. It finds the end of the sequence by argmax over
  `input_ids`, which works because CLIP's pad token is id 0 and
  `<|endoftext|>` is 49407 - and because the tokenizer adds special tokens
  after truncating, so an over-length query stays terminated.
- **Sequence length is padded to 77** even though this export accepts a
  variable length, so a query's embedding doesn't depend on how many tokens
  it happened to produce.
- **The interpreter in a release archive is scout's own.** It is
  relocatable, resolves its stdlib from the location of its own executable,
  and needs no Python on the user's machine. `main.go` runs it with `-E -s`
  so the user's Python environment can't reach into it, while leaving the
  script's own directory on `sys.path` - `worker.py` imports `clip.py` from
  there.
- **HEIC works only because of an import side effect.** `clip.py` calls
  `register_heif_opener()` at import; drop that and `Image.open` stops
  recognising `.heic` without any other symptom.
- **No network calls, ever.** Model files are fetched at build time by
  `scripts/fetch-deps.sh` and shipped in the archive. A missing file is an
  error naming it, not a download.
