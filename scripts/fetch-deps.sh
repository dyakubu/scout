#!/usr/bin/env bash
set -euo pipefail

# Fetches the native library scout needs but doesn't vendor in git:
# libonnxruntime (loaded at runtime via dlopen, see embedder/local_embedder.go).
#
# Used both for local dev setup and, unmodified, as the per-platform step in
# CI's release build matrix - run natively on each target OS/arch, since CGO
# rules out cross-compiling this from one machine.
ONNXRUNTIME_VERSION="1.29.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_DIR="$ROOT_DIR/third_party/onnxruntime"

GOOS="$(go env GOOS)"
GOARCH="$(go env GOARCH)"

case "$GOOS-$GOARCH" in
  darwin-arm64)
    ORT_ASSET="onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.dylib"
    ;;
  linux-amd64)
    ORT_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
    ;;
  linux-arm64)
    ORT_ASSET="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
    ORT_LIB_NAME="libonnxruntime.so"
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

mkdir -p "$ORT_DIR"

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

echo "native deps ready for $GOOS/$GOARCH"
