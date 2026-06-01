#!/usr/bin/env bash
set -euo pipefail

VERSION="18.0.6"
DEST="tools/llvm"
ARCHIVE_PATH="${1:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_PATH="$ROOT/$DEST"
mkdir -p "$DEST_PATH"

candidates=(
  "clang+llvm-$VERSION-x86_64-linux-gnu.tar.xz"
  "clang+llvm-$VERSION-x86_64-linux-musl.tar.xz"
)
archive_path=""

if [ -n "$ARCHIVE_PATH" ]; then
  if [ -f "$ARCHIVE_PATH" ]; then
    echo "Using pre-downloaded archive: $ARCHIVE_PATH"
    archive_path="$ARCHIVE_PATH"
  else
    echo "Error: archive file '$ARCHIVE_PATH' does not exist." >&2
    exit 1
  fi
else
  for archive_name in "${candidates[@]}"; do
  url="https://github.com/llvm/llvm-project/releases/download/llvmorg-$VERSION/$archive_name"
  echo "Trying LLVM download: $url"
  if command -v curl >/dev/null 2>&1; then
    if curl -fL -o "$DEST_PATH/$archive_name" "$url"; then
      archive_path="$DEST_PATH/$archive_name"
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -O "$DEST_PATH/$archive_name" "$url"; then
      archive_path="$DEST_PATH/$archive_name"
      break
    fi
  else
    echo "Error: curl or wget is required." >&2
    exit 1
  fi
  rm -f "$DEST_PATH/$archive_name"
done

if [ -z "$archive_path" ]; then
  echo "Could not download LLVM automatically." >&2
  echo "Please download one of the following archives manually and place it in $DEST_PATH or pass it as the first argument to this script:" >&2
  for archive_name in "${candidates[@]}"; do
    echo "  https://github.com/llvm/llvm-project/releases/download/llvmorg-$VERSION/$archive_name" >&2
  done
  echo "Example: ./scripts/install_clang.sh /path/to/clang+llvm-18.0.6-x86_64-linux-gnu.tar.xz" >&2
  exit 1
fi

echo "Extracting to $DEST_PATH ..."
tar -xJf "$ARCHIVE_PATH" -C "$DEST_PATH"
rm "$ARCHIVE_PATH"

EXTRACTED_DIR=$(find "$DEST_PATH" -maxdepth 1 -type d -name 'LLVM-*' | head -n 1)
if [ -z "$EXTRACTED_DIR" ]; then
  echo "Error: could not find extracted LLVM folder" >&2
  exit 1
fi

mkdir -p "$DEST_PATH/bin"
mv "$EXTRACTED_DIR/bin"/* "$DEST_PATH/bin/"
rm -rf "$EXTRACTED_DIR"

echo "LLVM installed to $DEST_PATH/bin"
echo "Use clang from $DEST_PATH/bin/clang or pass --clang-path to the CLI."
