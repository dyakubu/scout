#!/usr/bin/env bash
set -euo pipefail

# Builds scout and assembles it, its model/tokenizer config, and the
# onnxruntime shared library it dlopens at runtime into one self-contained
# archive - so `scout` runs fully offline immediately after being
# unpacked, with no separate download step. Run scripts/fetch-deps.sh
# first.
#
# tokenizer.json is scout's own pure-Go tokenizer's vocab/config, not a
# native library - no separate tokenizer runtime dependency ships here.
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

DIST_DIR="$ROOT_DIR/dist"
PKG_NAME="scout-${VERSION}-${GOOS}-${GOARCH}"
PKG_DIR="$DIST_DIR/$PKG_NAME"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/models" "$PKG_DIR/third_party/onnxruntime"

echo "building scout $VERSION for $GOOS/$GOARCH"
( cd "$ROOT_DIR" && go build -ldflags "-X main.version=${VERSION}" -o "$PKG_DIR/$BIN_NAME" . )

cp "$ROOT_DIR/models/model_qint8_avx512_vnni.onnx" "$PKG_DIR/models/"
cp "$ROOT_DIR/models/tokenizer.json" "$PKG_DIR/models/"
cp "$ORT_LIB" "$PKG_DIR/third_party/onnxruntime/"

( cd "$DIST_DIR" && tar -czf "${PKG_NAME}.tar.gz" "$PKG_NAME" )

# shasum is macOS/BSD-native; Windows' Git Bash ships sha256sum instead.
if command -v shasum >/dev/null 2>&1; then
  ( cd "$DIST_DIR" && shasum -a 256 "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256" )
else
  ( cd "$DIST_DIR" && sha256sum "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256" )
fi

rm -rf "$PKG_DIR"

echo "packaged $DIST_DIR/${PKG_NAME}.tar.gz"
