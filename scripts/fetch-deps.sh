#!/usr/bin/env bash
set -euo pipefail

# Fetches everything scout needs at runtime but doesn't vendor in git:
#
#   * libonnxruntime - loaded via dlopen, see embedder/local_embedder.go
#   * the CLIP model  - the media worker's, see media/clip.py
#   * CPython         - the interpreter that media worker runs on
#
# Used both for local dev setup and, unmodified, as the per-platform step in
# CI's release build matrix - run natively on each target OS/arch, since CGO
# rules out cross-compiling this from one machine.
#
# Every download here happens at build time. scout itself makes no network
# calls at any point: whatever it needs is in the release archive beside it
# by the time a user runs it, which is what scripts/package-release.sh
# assembles out of what this script fetches.
ONNXRUNTIME_VERSION="1.29.0"

# The media worker's CLIP model: two q4f16 ONNX towers plus their
# tokenizer, pinned to an exact repo revision rather than main so a build
# is reproducible and can't silently pick up re-uploaded weights. See
# media/clip.py for why these particular files.
CLIP_REPO="Xenova/clip-vit-base-patch32"
CLIP_REVISION="d15189d7028b43f1d3e65039190477f6af591c2a"

# CPython for the media worker, built by astral-sh/python-build-standalone
# to be relocatable - it resolves its own stdlib from the path of its own
# executable, so it works from wherever the archive is unpacked without a
# venv, a PYTHONHOME, or any Python on the user's machine. "install_only"
# is the runtime-only layout (no build artifacts); "_stripped" has debug
# symbols removed, roughly halving it on Linux and Windows.
PYTHON_VERSION="3.11.16"
PYTHON_RELEASE="20260901"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_DIR="$ROOT_DIR/third_party/onnxruntime"
PYTHON_DIR="$ROOT_DIR/third_party/python"
CLIP_DIR="$ROOT_DIR/models/media"

GOOS="$(go env GOOS)"
GOARCH="$(go env GOARCH)"

case "$GOOS-$GOARCH" in
  darwin-arm64)
    ORT_ASSET="onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.dylib"
    PYTHON_TRIPLE="aarch64-apple-darwin"
    ;;
  linux-amd64)
    ORT_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
    PYTHON_TRIPLE="x86_64-unknown-linux-gnu"
    ;;
  linux-arm64)
    ORT_ASSET="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
    PYTHON_TRIPLE="aarch64-unknown-linux-gnu"
    ;;
  windows-amd64)
    ORT_ASSET="onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip"
    ORT_LIB_NAME="onnxruntime.dll"
    PYTHON_TRIPLE="x86_64-pc-windows-msvc"
    ;;
  darwin-amd64)
    echo "error: onnxruntime v${ONNXRUNTIME_VERSION} publishes no osx-x64 build (Intel Mac isn't supported upstream) - build onnxruntime from source, or use an older ONNX Runtime release that still ships one" >&2
    exit 1
    ;;
  *)
    echo "error: unsupported platform $GOOS/$GOARCH" >&2
    exit 1
    ;;
esac

# shasum is macOS/BSD-native; Linux and Git Bash ship sha256sum instead.
verify_sha256() {
  local file="$1" want="$2" got

  if command -v shasum >/dev/null 2>&1; then
    got="$(shasum -a 256 "$file" | cut -d" " -f1)"
  else
    got="$(sha256sum "$file" | cut -d" " -f1)"
  fi

  if [[ "$got" != "$want" ]]; then
    rm -f "$file"
    echo "error: checksum mismatch for $file" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    exit 1
  fi
}

mkdir -p "$ORT_DIR"

if [[ -f "$ORT_DIR/$ORT_LIB_NAME" && -z "${FORCE:-}" ]]; then
  echo "onnxruntime already present at $ORT_DIR/$ORT_LIB_NAME (set FORCE=1 to re-fetch)"
else
  echo "fetching $ORT_ASSET"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  archive="$tmp/ort.${ORT_ASSET##*.}"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_ASSET}" -o "$archive"

  # The release ships .tgz for macOS/Linux and .zip for Windows.
  case "$ORT_ASSET" in
    *.zip) unzip -q "$archive" -d "$tmp" ;;
    *)     tar -xzf "$archive" -C "$tmp" ;;
  esac

  pkg_dir="$(find "$tmp" -maxdepth 1 -type d -name 'onnxruntime-*')"
  # -L dereferences the release's libonnxruntime.{dylib,so} -> libonnxruntime.<version>.<ext>
  # symlink (the Windows .zip has no such symlink - onnxruntime.dll is a
  # plain file already - so -L is a no-op there), so third_party/ ends up
  # with one plain file either way, no symlink chain to preserve.
  cp -L "$pkg_dir/lib/$ORT_LIB_NAME" "$ORT_DIR/$ORT_LIB_NAME"
  rm -rf "$tmp"
  trap - EXIT
fi

# The CLIP model. Checked per file rather than per directory: a download
# interrupted partway leaves the directory looking populated while the
# worker still can't start (see media/clip.py's missing_files).
mkdir -p "$CLIP_DIR"

for entry in \
  "vision_model_q4f16.onnx:onnx/vision_model_q4f16.onnx:d238c4e0afe798c47c5991a046b923c5bcbeed19c2d75d7c0db845ba73bb7b87" \
  "text_model_q4f16.onnx:onnx/text_model_q4f16.onnx:7640460b1389da7d4fa3b3ebcc80299fbe6850b209a009b5067d25c4a9d027d2" \
  "tokenizer.json:tokenizer.json:f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"
do
  name="${entry%%:*}"
  rest="${entry#*:}"
  repo_path="${rest%%:*}"
  want_sha="${rest#*:}"

  if [[ -f "$CLIP_DIR/$name" && -z "${FORCE:-}" ]]; then
    echo "clip model $name already present (set FORCE=1 to re-fetch)"
    continue
  fi

  echo "fetching clip model $name"
  # Downloaded to a temporary name and renamed only after its checksum
  # matches, so an interrupted or corrupted fetch can't leave a file that
  # later runs treat as complete.
  curl -fsSL "https://huggingface.co/${CLIP_REPO}/resolve/${CLIP_REVISION}/${repo_path}" -o "$CLIP_DIR/$name.partial"
  verify_sha256 "$CLIP_DIR/$name.partial" "$want_sha"
  mv "$CLIP_DIR/$name.partial" "$CLIP_DIR/$name"
done

# CPython for the media worker.
if [[ -x "$PYTHON_DIR/bin/python3" || -x "$PYTHON_DIR/python.exe" ]] && [[ -z "${FORCE:-}" ]]; then
  echo "python already present at $PYTHON_DIR (set FORCE=1 to re-fetch)"
else
  PYTHON_ASSET="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${PYTHON_TRIPLE}-install_only_stripped.tar.gz"
  echo "fetching $PYTHON_ASSET"

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${PYTHON_ASSET}" -o "$tmp/python.tar.gz"
  tar -xzf "$tmp/python.tar.gz" -C "$tmp"

  # The tarball unpacks to a single "python/" directory. Replaced wholesale
  # rather than merged into an existing one, so a re-fetch can't leave
  # files from a previous version behind.
  rm -rf "$PYTHON_DIR"
  mkdir -p "$(dirname "$PYTHON_DIR")"
  mv "$tmp/python" "$PYTHON_DIR"

  rm -rf "$tmp"
  trap - EXIT
fi

echo "native deps ready for $GOOS/$GOARCH"
