"""Media worker stdio protocol.

Reads one JSON job per line from stdin, writes one matching JSON result per
line to stdout, then repeats until stdin closes. --model-dir is a startup
argument, not a per-job field: one worker process always serves one fixed
model directory (mediaworker.Client passes it once at spawn time), so the
model is loaded eagerly right after startup, not lazily on the first job -
load time and any load failure show up immediately and predictably, the
same way embedder.NewLocalEmbedder loads the text model eagerly on the Go
side rather than deferring it to an arbitrary later point. A load failure
here is left to crash the process rather than caught and retried per job -
a broken model directory doesn't get less broken by trying again later.

Set SCOUT_MEDIA_DUMMY=1 to skip real inference entirely and return a fixed
zero vector instead - this is a test-only escape hatch (see
mediaworker/client_test.go) so the Go<->Python wiring can be exercised
without network access or the real dependencies in pyproject.toml
installed. Not a user-facing mode.

Job:    {"id": str, "kind": "image" | "text", "payload": str}
Result: {"id": str, "embedding": [float, ...] | null, "error": str | null}

"kind" selects which CLIP tower embeds payload: "image" (a file path) via
embed_image, or "text" (a search query) via embed_text. Both land in the
same 512-dim CLIP space, but embed_text is only ever used to query
vec_media - indexed text file chunks are embedded entirely separately, by
the text embedder in embedder/, never by this worker.
"""

import argparse
import json
import os
import sys

# Matches openai/clip-vit-base-patch32's image embedding dimension.
EMBEDDING_DIM = 512

DUMMY = os.environ.get("SCOUT_MEDIA_DUMMY") == "1"

_model_dir_configured = False
_sessions = None
_tokenizer = None


def handle(job: dict) -> dict:
    job_id = job.get("id")

    if not _model_dir_configured:
        return {"id": job_id, "embedding": None, "error": "missing model_dir"}

    if DUMMY:
        return {"id": job_id, "embedding": [0.0] * EMBEDDING_DIM, "error": None}

    payload = job.get("payload")
    if not payload:
        return {"id": job_id, "embedding": None, "error": "missing payload"}

    kind = job.get("kind", "image")

    try:
        if kind == "text":
            from clip import embed_text

            embedding = embed_text(_sessions, _tokenizer, payload)
        elif kind == "image":
            from clip import embed_image

            embedding = embed_image(_sessions, payload)
        else:
            return {"id": job_id, "embedding": None, "error": f"unknown kind {kind!r}"}
    except Exception as e:
        return {"id": job_id, "embedding": None, "error": str(e)}

    return {"id": job_id, "embedding": embedding, "error": None}


def main() -> None:
    global _model_dir_configured, _sessions, _tokenizer

    parser = argparse.ArgumentParser(description="scout media worker")
    parser.add_argument("--model-dir", default="", help="directory to load the CLIP model from")
    args = parser.parse_args()

    _model_dir_configured = bool(args.model_dir)

    # Imported lazily, not at module load, so SCOUT_MEDIA_DUMMY=1 tests
    # never need onnxruntime/pillow installed at all.
    if _model_dir_configured and not DUMMY:
        from clip import load_model

        try:
            _sessions, _tokenizer = load_model(args.model_dir)
        except Exception as e:
            # Fatal, but reported as one line: scout wires this process's
            # stderr to its own, so a traceback would land in the middle of
            # a user's search results.
            print(f"scout media worker: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"id": None, "embedding": None, "error": f"invalid json: {e}"}), flush=True)
            continue

        print(json.dumps(handle(job)), flush=True)


if __name__ == "__main__":
    main()
