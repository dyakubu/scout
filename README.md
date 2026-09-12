# scout

Scout is a local-first semantic search engine for your own filesystem, a `grep` that understands meaning instead of just matching characters. Point it at a directory, and it indexes text (and, optionally, images) into vector embeddings so you can later search by what a file *means*, not just what substring it contains, entirely on your own machine, with no cloud API calls and no data leaving your disk.

```
$ scout index ~/notes

Indexed 412 file(s), 6981 chunk(s) embedded, in 8.42s

$ scout find "that thing about retry backoff"

~/notes/systems/queues.md:112-134  (score: 0.81)

    Exponential backoff with jitter avoids thundering-herd retries when a
    downstream service comes back up after an outage...
```

## Architecture

```
                  scout index / scout find
                             │
                             ▼
                       cli / commands
                             │
                      ┌───────┴───────┐
                      ▼               ▼
                   indexer         search
        ╭┄┄┄┄┄┄┄┄┄┄┄┄┄┤               ├┄┄┄┄┄┄┄┄┄┄┄┄┄╮
        ┊             │               │             ┊
        ┊             └───────┬───────┘             ┊
        ┊                     ▼                     ┊
        ┊     embedder + tokenizer (ONNX, cgo)      ┊
        ┊                     │                     ┊
        ┊                     ▼                     ┊
        ┊                  SQLite                   ┊
        ┊        files · chunks · vec_chunks        ┊
        ┊       media_embeddings · vec_media        ┊
        ┊                                           ┊
        ╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┬┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                              ▼
                      media/worker.py
                 (Python subprocess, CLIP)
              stdio JSON lines · optional path
```

Everything down to SQLite runs inside a single Go binary, except the media worker.
It's a separate Python subprocess, started only if `media.model_dir` is configured, and text indexing/search
never depends on it. Any failure to reach it just means media results are
skipped.

## Why

Traditional file search (`grep`, `fzf`, Spotlight) matches literal text. It can't find a note about "the outage from last quarter" unless it contains those exact words. Scout embeds file contents into a vector space using a local transformer model, so semantically similar text ranks highly even when the wording is completely different, while keeping everything private and offline, since nothing is ever sent to a third-party API.

## How it works

```
scout index <dir>

        │

        ▼

    walk directory (gitignore-aware, extension/size filtered)

        │

        ▼

    chunk each file's text (fixed-size, line-tracked)

        │

        ▼

    embed each chunk (local ONNX model, mean-pooled + L2-normalized)

        │

        ▼

    store chunks + vectors in SQLite (sqlite-vec virtual table)

scout find "<query>"

        │

        ▼

    embed the query with the same model

        │

        ▼

    cosine-similarity kNN search over stored vectors

        │

        ▼

    ranked results: path, line range, snippet, score
```

Everything above runs as a single self-contained Go binary. There's no server, no database to install, and no network calls at query or index time. The embedding model, tokenizer, and ONNX Runtime library all ship alongside the binary.

### Core pieces

| Package             | Responsibility                                                                                                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `indexer/`          | Walks a directory (honoring `.gitignore`), chunks file text, extracts text from PDFs, and drives concurrent embedding via a bounded worker pool + single DB-writer goroutine |
| `embedder/`         | Loads a local ONNX model via cgo (`onnxruntime_go`) and runs batched inference, mean-pooling token outputs into one L2-normalized vector per chunk                           |
| `tokenizer/`        | A pure-Go tokenizer (no cgo, no Rust) that turns chunk text into the model's expected `input_ids`/`attention_mask`/`token_type_ids`                                          |
| `db/`               | SQLite schema: `files`, `chunks`, and `vec_chunks` (a [sqlite-vec](https://github.com/asg017/sqlite-vec) virtual table holding the actual embedding vectors)                 |
| `search/`           | Embeds a query and runs a kNN scan over `vec_chunks`, filtering by embedding model and (optionally) a `--restrict` path prefix                                               |
| `config/`           | Per-OS default config (TOML), resolving asset paths relative to the running binary rather than the working directory                                                         |
| `cli/`, `commands/` | Argument parsing and the `find`/`index`/`sync`/`config`/`clean` command implementations                                                                                      |
| `mediaworker/`      | A stdio JSON-lines client that talks to an optional Python subprocess for image embeddings (see below)                                                                       |

### Why a local model instead of an API

Every embedding is produced by a small transformer model running entirely in-process via [ONNX Runtime](https://onnxruntime.ai/), loaded once at startup. This means:

* **Privacy**: file contents never leave your machine.
* **Offline**: indexing and search work with no network access.
* **No per-query cost**: there's no API to rate-limit or bill against.

The tradeoff is that scout ships a real (if small, quantized) ML model and a native ONNX Runtime library per platform, rather than a pure-Go binary. See [Installing](#installing) below.

### Image search (media worker)

Scout can also embed images using [CLIP](https://openai.com/research/clip), so `scout find` can turn up matching photos alongside matching text. It runs as a **separate Python subprocess** (`media/worker.py`) that scout's `mediaworker` package talks to over stdin/stdout, one JSON job per line.

The worker runs CLIP ViT-B/32 through ONNX Runtime rather than PyTorch. It's the same model either way, but the quantized ONNX towers are ~126MB against torch's ~605MB checkpoint, the dependencies are ~144MB installed against ~764MB, and the worker starts in ~0.5s instead of ~2.4s - which scout pays on every `find`, since the query embed waits on the worker's model load. Quantization costs some fidelity: embeddings sit 0.94-0.99 cosine from the fp32 torch ones, close enough to rank the same. `media/test_clip.py` pins that against reference vectors captured from the original torch implementation.

**You don't need Python installed.** The release archive carries its own, in `media/python` - a relocatable [python-build-standalone](https://github.com/astral-sh/python-build-standalone) interpreter with the worker's four dependencies (onnxruntime, numpy, Pillow, tokenizers) already in its `site-packages`. It resolves its own stdlib from the path of its own executable, so it works wherever the archive is unpacked, needs no virtualenv, never goes on your `PATH`, and can't collide with a Python you already have. Scout runs it with `-E -s`, so `PYTHONPATH`, `PYTHONHOME`, and user site-packages can't reach into it either. It's scout's private file, the same as the ONNX Runtime library next to it.

The model is resolved like every other scout asset - `media.model_dir`, relative to the binary's own directory, defaulting to `models/media` - and is never downloaded. Set `media.model_dir = ""` to turn media support off.

Every media failure degrades to text-only rather than breaking a run: no `media.model_dir`, nothing at the path it names, a model directory missing files, or a worker that won't start. Scout logs it and carries on. Indexing and search never hard-fail because of the media path.

In a repo checkout there's no bundled interpreter, so the worker falls back to `uv run --project media media/worker.py`, which needs `uv` and a `uv sync` in `media/` (see [Development](#development)).

## Installing

### Homebrew (macOS Apple Silicon, Linux)

```
brew install dyakubu/scout/scout
```

Installs from the [homebrew-scout](https://github.com/dyakubu/homebrew-scout)
tap, pointing at the same release archives as below. Intel Macs aren't
supported: ONNX Runtime publishes no macOS x86_64 build after v1.23.2, and
that release is older than scout's Go bindings can drive (they request C
API 29; 1.23.2 offers 23), so pinning back to it isn't a way around this.
Windows isn't available via Homebrew - use the release archive below
instead.

### From a release

Download the archive for your platform from the [Releases](https://github.com/dyakubu/scout/releases) page and extract it. Add the extracted directory to your `PATH`, or invoke `./scout` directly from inside it. It's self-contained - everything is pre-wired to find everything else, and nothing is fetched on first run:

```
scout                                the binary
models/model_qint8_avx512_vnni.onnx  text embedding model
models/tokenizer.json                its tokenizer's vocab
models/media/                        CLIP model for image search
third_party/onnxruntime/             ONNX Runtime, loaded at runtime
media/                               image search worker + its Python interpreter
```

That comes to ~220MB compressed, ~430MB unpacked, a bit over half of which is image search. Deleting `media/` and `models/media` leaves text search working exactly as before, if you'd rather not carry it.

Supported platforms: macOS (Apple Silicon), Linux (x86_64/arm64), and Windows (x86_64).

### From source

Requires Go 1.25+ and a C compiler (cgo is required, `onnxruntime_go` and the model runtime link native code):

```
git clone https://github.com/dyakubu/scout.git
cd scout

scripts/fetch-deps.sh          # onnxruntime, CLIP model, and python, for your platform

VERSION=dev scripts/package-release.sh
```

This produces `dist/scout-dev-<os>-<arch>.tar.gz`, the same self-contained archive a release build produces. Packaging also needs [uv](https://docs.astral.sh/uv/), which installs the media worker's dependencies into the bundled interpreter - a build-time tool only, never required of anyone running scout. On Windows, cgo needs a MinGW-w64 toolchain (see `.github/workflows/release.yml` for the CI setup via MSYS2).

## Usage

Two commands get you going: point `scout index` at a directory, then search it
with `scout find`.

```
$ scout index ~/notes
Indexed 3 file(s), 3 chunk(s) embedded, in 20ms

$ scout find "why do retries make an outage worse"
~/notes/eng/queues.md:1-7  (score: 0.42)
    # Queue design Exponential backoff with jitter avoids thundering-herd
    retries when a downstream service comes back up after an outage...
```

Indexing is incremental, so re-running it on the same directory only embeds
what changed. Searching never touches the network, and neither does indexing.

| Command                                | What it does                            |
| -------------------------------------- | --------------------------------------- |
| `scout index [path]`                   | Index a directory (default: cwd)         |
| `scout find <query>`                   | Search what's indexed                    |
| `scout config [path\|get\|set] ...`     | View or change settings                  |
| `scout clean`                          | Delete the index and log, start over     |
| `scout version`                        | Print the version                        |
| `scout help`                           | Print usage                              |

### `index`

```
scout index                       # index the current directory
scout index ~/notes               # index a specific directory
scout index ~/notes --recursive=false   # just that directory, no subdirectories
```

Walks `path`, skipping anything matched by a `.gitignore` in the tree,
`index.ignore_dirs` / `index.ignore_patterns` / `index.ignore_dir_markers`,
oversized files, and extensions not listed in `index.allowed_extensions`.
Directories it has no permission to read are skipped and counted.

Size limits are per file type: `index.max_file_size_mb` is the fallback
(2MB), and `index.max_file_size_mb_by_type` overrides it by extension.
Formats that carry their own media are mostly container - a PDF's page
images, a photo's full resolution - so they ship with far more room than
source files get.

Re-running it is cheap: files whose modification time hasn't changed are
skipped without re-embedding.

```
$ scout index ~/notes
Indexed 3 file(s), 3 chunk(s) embedded, in 20ms

$ scout index ~/notes          # nothing changed since last time
Indexed 0 file(s), 0 chunk(s) embedded, in 0s
  3 file(s) unchanged, skipped
```

The summary lines below the first are only printed when they're non-zero, so a
run that reports nothing but the first line had nothing to skip.

### `find`

```
scout find "how does retry backoff work"
scout find "the design doc about caching" --max=3
scout find "cache invalidation" --restrict=~/notes/eng
scout find "screenshot of the login screen" --media-max=5
```

* `--max=N` - how many text results (default: `search.max_results`).
* `--media-max=N` - how many image results (default: `search.max_media_results`).
  Only meaningful with image search configured.
* `--restrict[=path]` - only return results under `path`. With no value, it
  means the current directory, so `scout find "todo" --restrict` searches
  where you're standing.

Quote the query. Without quotes the shell splits it into separate arguments
and scout only sees the first word.

Results are ranked by cosine similarity, printed highest first:

```
$ scout find "stale data problems" --max=1
~/notes/eng/caching.md:1-7  (score: 0.60)
    # Cache invalidation We settled on write-through caching with a short TTL
    rather than explicit invalidation. Explicit invalidation was correct...
```

Results are spread across documents: one file contributes at most
`search.max_results_per_file` (2 by default) before others get a turn, so a
long document can't fill every slot with near-identical chunks. Leftover
slots are still filled by score, so a query that genuinely matches one
document still returns a full set.

Scores run from about 1.0 (nearly identical meaning) down to 0 and below
(unrelated). A nearest-neighbour search always returns *something*, so a query
about a topic you've never written about still comes back with results - they
just score low. Treat anything under ~0.2 as "no real match" rather than
expecting an empty list.

In a terminal that supports OSC 8 hyperlinks, the file paths are clickable.
Piping the output strips the escape codes, so `scout find ... | grep` is safe.

### `config`

```
scout config                              # print the whole config file
scout config path                         # print where that file lives
scout config get search.max_results       # read one value
scout config set search.max_results 3     # change one value
scout config set index.max_file_size_mb 20
```

```
$ scout config get search.max_results
5

$ scout config set search.max_results 3
search.max_results = 3
```

List-valued fields - `index.allowed_extensions`, `index.ignore_dirs`,
`index.ignore_patterns`, `index.ignore_dir_markers`,
`media.allowed_extensions` - can't be set from the command line. Edit them in
the file (`scout config path` tells you where it is).

The config file is created from a built-in default the first time scout runs,
and is never rewritten after that. New settings added by a later version of
scout won't appear in a config file that already exists - copy them across by
hand, or delete the file to have it recreated.

### `sync`

Listed by `scout help`, but not implemented yet - it prints what it would do
and exits. Re-running `scout index` on a directory already picks up changed
files and skips unchanged ones, which is what `sync` is eventually for.

### `clean`

```
scout clean
```

Deletes the search index and the log, so the next `index` starts from nothing.
Your config file is left alone. Useful after changing an embedding model, since
vectors from different models aren't comparable and the old ones are ignored
rather than re-embedded.

## Where scout keeps its files

Everything lives in one directory, whose location follows each OS's own
convention. `scout config path` prints it.

| OS      | Directory                             |
| ------- | ------------------------------------- |
| macOS   | `~/Library/Application Support/scout` |
| Linux   | `~/.config/scout`                     |
| Windows | `%AppData%\scout`                     |

| File               | What it is                                                     |
| ------------------ | -------------------------------------------------------------- |
| `scoutconfig.toml` | Settings. Yours to edit; scout only writes it on first run.     |
| `scout.db`         | The search index: SQLite, including the embedding vectors.      |
| `scout.log`        | Timing and diagnostics, appended on every run. Not truncated.   |

`scout.db` grows with what you index, and the log grows slowly forever -
`scout clean` removes both.

The model files and the ONNX Runtime library don't live here. They ship inside
the release archive and are found relative to the `scout` binary itself, which
is why moving the binary out of its extracted directory breaks it.

## Development

```
go build ./...
go test ./...
```

* `scripts/fetch-deps.sh`: fetches everything scout ships but doesn't commit - the ONNX Runtime library and a standalone Python into `third_party/`, the CLIP model into `models/media/` (all gitignored). Model files are pinned to an exact upstream revision and checksum-verified.
* `scripts/package-release.sh`: builds the binary and assembles a self-contained release archive in `dist/` (gitignored).
* `embedder/embeddertest/`: a fake `Embedder`/`MediaEmbedder` for tests that don't want to load a real model.

To develop against the media worker locally, from the repo root:

```
cd media

uv sync

uv run pytest        # skips unless media.model_dir holds a CLIP model
```

then put the CLIP model - ~126MB: the two q4f16 ONNX towers and `tokenizer.json`, named in `media/clip.py` - in `models/media/`, which is where `media.model_dir` points by default, and run `scout index`/`scout find` as usual from the repo root. Nothing fetches the model for you: scout makes no network calls at any point, and the worker exits naming whichever files are missing. Set `media.model_dir = ""` to turn media support off entirely.

Media embeddings record which model produced them and are only compared against vectors from that same model, so changing the media model (or its quantization) strands whatever is already indexed - `scout clean` and re-index after one.

## Status

Scout is an early, personal-scale project, not yet hardened for huge corpora or concurrent multi-user use. In particular:

* `scout sync` is a placeholder; re-run `scout index` instead.
* Chunking is fixed-size, not semantic (no sentence/paragraph awareness yet).
* The kNN search pulls a bounded candidate pool before filtering, so a narrow `--restrict` can occasionally return fewer results than `--max`.
* Image search adds ~185MB to the release archive, including a Python interpreter, and text-only users pay for it too.
