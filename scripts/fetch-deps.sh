#!/usr/bin/env bash
set -euo pipefail

# Fetches the native libraries scout needs but doesn't vendor in git:
# libtokenizers.a (linked into the binary at compile time via CGO) and
# libonnxruntime (loaded at runtime via dlopen, see embedder/local_embedder.go).
#
# Used both for local dev setup and, unmodified, as the per-platform step in
# CI's release build matrix - run natively on each target OS/arch, since CGO
# rules out cross-compiling this from one machine.
#
# Versions are pinned here rather than tracking "latest": TOKENIZERS_VERSION
# must match the version of github.com/daulet/tokenizers required in go.mod,
# since the Go bindings and the prebuilt static lib share an ABI that isn't
# guaranteed stable across releases. Bump both together.
ONNXRUNTIME_VERSION="1.29.0"
TOKENIZERS_VERSION="1.27.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_DIR="$ROOT_DIR/third_party/onnxruntime"
TOK_DIR="$ROOT_DIR/third_party/tokenizers"

GOOS="$(go env GOOS)"
GOARCH="$(go env GOARCH)"

case "$GOOS-$GOARCH" in
  darwin-arm64)
    ORT_ASSET="onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.dylib"
    TOK_ASSET="libtokenizers.darwin-arm64.tar.gz"
    ;;
  linux-amd64)
    ORT_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
    TOK_ASSET="libtokenizers.linux-amd64.tar.gz"
    ;;
  linux-arm64)
    ORT_ASSET="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
    TOK_ASSET="libtokenizers.linux-arm64.tar.gz"
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

mkdir -p "$ORT_DIR" "$TOK_DIR"

if [[ -f "$ORT_DIR/$ORT_LIB_NAME" && -z "${FORCE:-}" ]]; then
  echo "onnxruntime already present at $ORT_DIR/$ORT_LIB_NAME (set FORCE=1 to re-fetch)"
else
  echo "fetching $ORT_ASSET"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_ASSET}" -o "$tmp/ort.tgz"
  tar -xzf "$tmp/ort.tgz" -C "$tmp"
  pkg_dir="$(find "$tmp" -maxdepth 1 -type d -name 'onnxruntime-*')"
  # -L dereferences the release's libonnxruntime.{dylib,so} -> libonnxruntime.<version>.<ext>
  # symlink, so third_party/ ends up with one plain file and no symlink chain to preserve.
  cp -L "$pkg_dir/lib/$ORT_LIB_NAME" "$ORT_DIR/$ORT_LIB_NAME"
  rm -rf "$tmp"
  trap - EXIT
fi

if [[ -f "$TOK_DIR/libtokenizers.a" && -z "${FORCE:-}" ]]; then
  echo "tokenizers already present at $TOK_DIR/libtokenizers.a (set FORCE=1 to re-fetch)"
else
  echo "fetching $TOK_ASSET"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}/${TOK_ASSET}" -o "$tmp/tok.tar.gz"
  tar -xzf "$tmp/tok.tar.gz" -C "$tmp"
  cp "$tmp/libtokenizers.a" "$TOK_DIR/libtokenizers.a"
  rm -rf "$tmp"
  trap - EXIT
fi

echo "native deps ready for $GOOS/$GOARCH"
