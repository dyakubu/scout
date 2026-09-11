#!/usr/bin/env bash
set -euo pipefail

# Builds scout and assembles everything it needs at runtime into one
# self-contained archive - so `scout` runs fully offline immediately after
# being unpacked, with no separate download step, for media search as much
# as for text. Run scripts/fetch-deps.sh first.
#
#   scout                              the binary
#   models/                            text embedder model + tokenizer
#   models/media/                      CLIP model for the media worker
#   third_party/onnxruntime/           dlopened by the Go embedder
#   media/                             the media worker and its interpreter
#
# models/tokenizer.json is scout's own pure-Go tokenizer's vocab/config,
# not a native library - no separate tokenizer runtime dependency ships
# there. models/media/tokenizer.json is a different thing: CLIP's own
# tokenizer, read by the Python worker.
#
# Usage: VERSION=v0.1.0 scripts/package-release.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:?set VERSION, e.g. VERSION=v0.1.0}"

GOOS="$(go env GOOS)"
GOARCH="$(go env GOARCH)"

BIN_NAME="scout"

case "$GOOS" in
  darwin)  ORT_LIB_NAME="libonnxruntime.dylib" ;;
  linux)   ORT_LIB_NAME="libonnxruntime.so" ;;
  windows) ORT_LIB_NAME="onnxruntime.dll"; BIN_NAME="scout.exe" ;;
  *)       echo "error: unsupported platform $GOOS/$GOARCH" >&2; exit 1 ;;
esac

ORT_LIB="$ROOT_DIR/third_party/onnxruntime/$ORT_LIB_NAME"
if [[ ! -f "$ORT_LIB" ]]; then
  echo "error: $ORT_LIB not found - run scripts/fetch-deps.sh first" >&2
  exit 1
fi

PYTHON_SRC="$ROOT_DIR/third_party/python"
if [[ ! -d "$PYTHON_SRC" ]]; then
  echo "error: $PYTHON_SRC not found - run scripts/fetch-deps.sh first" >&2
  exit 1
fi

CLIP_SRC="$ROOT_DIR/models/media"
for name in vision_model_q4f16.onnx text_model_q4f16.onnx tokenizer.json; do
  if [[ ! -f "$CLIP_SRC/$name" ]]; then
    echo "error: $CLIP_SRC/$name not found - run scripts/fetch-deps.sh first" >&2
    exit 1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found - it installs the media worker's dependencies into the bundled interpreter" >&2
  exit 1
fi

DIST_DIR="$ROOT_DIR/dist"
PKG_NAME="scout-${VERSION}-${GOOS}-${GOARCH}"
PKG_DIR="$DIST_DIR/$PKG_NAME"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/models/media" "$PKG_DIR/third_party/onnxruntime" "$PKG_DIR/media"

echo "building scout $VERSION for $GOOS/$GOARCH"
( cd "$ROOT_DIR" && go build -ldflags "-X main.version=${VERSION}" -o "$PKG_DIR/$BIN_NAME" . )

cp "$ROOT_DIR/models/model_qint8_avx512_vnni.onnx" "$PKG_DIR/models/"
cp "$ROOT_DIR/models/tokenizer.json" "$PKG_DIR/models/"
cp "$ORT_LIB" "$PKG_DIR/third_party/onnxruntime/"

cp "$CLIP_SRC/vision_model_q4f16.onnx" "$CLIP_SRC/text_model_q4f16.onnx" "$CLIP_SRC/tokenizer.json" "$PKG_DIR/models/media/"

# The media worker: its two source files and the interpreter that runs
# them. Only worker.py and clip.py - main.py is a manual-testing CLI and
# test_clip.py is a test, neither of which scout ever invokes.
cp "$ROOT_DIR/media/worker.py" "$ROOT_DIR/media/clip.py" "$PKG_DIR/media/"

echo "assembling the media worker's Python runtime"

# A pristine copy of the fetched interpreter, with the worker's
# dependencies installed straight into its own site-packages - no
# virtualenv. A venv would record an absolute path to its base interpreter
# in pyvenv.cfg and break the moment the archive is unpacked somewhere
# else; this tree resolves its stdlib from the location of its own
# executable, so it works wherever it lands. See media/clip.py.
cp -R "$PYTHON_SRC" "$PKG_DIR/media/python"

case "$GOOS" in
  windows) PKG_PYTHON="$PKG_DIR/media/python/python.exe" ;;
  *)       PKG_PYTHON="$PKG_DIR/media/python/bin/python3" ;;
esac

# Installed from the lockfile, not from pyproject.toml's ranges, so the
# archive pins exactly what `uv sync` gives a developer. --no-dev leaves
# out pytest and friends.
uv export --project "$ROOT_DIR/media" --no-dev --no-hashes --format requirements-txt \
  | uv pip install --quiet --python "$PKG_PYTHON" -r -

# Bytecode caches, pip, and the build headers are all dead weight in a
# shipped tree: nothing in the archive compiles against this interpreter
# or installs into it after packaging.
rm -rf "$PKG_DIR/media/python/include"
find "$PKG_DIR/media/python" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

"$PKG_PYTHON" -E -s -c "import onnxruntime, numpy, PIL, tokenizers" \
  || { echo "error: bundled interpreter can't import the media worker's dependencies" >&2; exit 1; }

( cd "$DIST_DIR" && tar -czf "${PKG_NAME}.tar.gz" "$PKG_NAME" )

# shasum is macOS/BSD-native; Windows' Git Bash ships sha256sum instead.
if command -v shasum >/dev/null 2>&1; then
  ( cd "$DIST_DIR" && shasum -a 256 "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256" )
else
  ( cd "$DIST_DIR" && sha256sum "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256" )
fi

rm -rf "$PKG_DIR"

echo "packaged $DIST_DIR/${PKG_NAME}.tar.gz"
