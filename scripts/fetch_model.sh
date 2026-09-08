#!/usr/bin/env bash
# Fetch the released model checkpoint. Used by CI and by humans.
set -euo pipefail

TAG="${MODEL_TAG:-model-v1.0.0}"
DEST="checkpoints/unet_lung_best.pt"

mkdir -p checkpoints
if [ -f "$DEST" ]; then
  echo "already have $DEST"
  exit 0
fi

echo "fetching $TAG -> $DEST"
gh release download "$TAG" \
  --repo Himachiz/Unet-serving \
  --pattern "unet_lung_best.pt" \
  --output "$DEST"
