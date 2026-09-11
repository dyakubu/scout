import argparse
import os
from pathlib import Path

from clip import embed_image, load_model

# Default when no --model-dir/SCOUT_MEDIA_MODEL_DIR is given: the same
# models/media the shipped config's media.model_dir points at, resolved
# from this script's own location rather than the current working
# directory, so behavior doesn't depend on where it's invoked from.
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "media"


def main():
    parser = argparse.ArgumentParser(description="Embed an image with CLIP")
    parser.add_argument("image", help="path to the image file")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SCOUT_MEDIA_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="directory to load/download the CLIP model from",
    )
    args = parser.parse_args()

    sessions, _ = load_model(args.model_dir)
    embedding = embed_image(sessions, args.image)

    print(f"embedding dim: {len(embedding)}")
    print(embedding)


if __name__ == "__main__":
    main()
