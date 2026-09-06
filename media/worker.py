
"""Media worker stdio protocol (dummy mode).

Reads one JSON job per line from stdin, writes one matching JSON result per
line to stdout, then repeats until stdin closes. No model is loaded here
and nothing outside the standard library is imported - this lets the
Go<->Python wiring be exercised without network access or the (large) real
dependencies in pyproject.toml installed. Swap handle() for a real call
into main.py's embed_image once that's needed.

Job:    {"id": str, "model_dir": str, "payload": str}
Result: {"id": str, "embedding": [float, ...] | null, "error": str | null}
"""

import json
import sys

# Matches openai/clip-vit-base-patch32's image embedding dimension, so
# swapping this dummy vector for a real one later doesn't change the
# protocol's shape.
EMBEDDING_DIM = 512


def handle(job: dict) -> dict:
    job_id = job.get("id")

    model_dir = job.get("model_dir")
    if not model_dir:
        return {"id": job_id, "embedding": None, "error": "missing model_dir"}

    # Dummy embedding: proves a job round-tripped through Go -> stdin ->
    # this process -> stdout -> Go, without a real model ever loading.
    return {"id": job_id, "embedding": [0.0] * EMBEDDING_DIM, "error": None}


def main() -> None:
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
