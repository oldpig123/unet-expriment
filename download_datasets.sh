#!/bin/bash
set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR/verse"
mkdir -p "$DATA_DIR/lumbar_mri"

echo "=== Downloading VerSe 19 dataset ==="
if [ ! -f "$DATA_DIR/dataset-verse19training.zip" ]; then
    wget -c -O "$DATA_DIR/dataset-verse19training.zip" "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip"
else
    echo "VerSe zip already downloaded."
fi

echo "=== Extracting VerSe 19 dataset ==="
# Check if directory has files (excluding hidden files)
if [ ! -d "$DATA_DIR/verse" ] || [ -z "$(ls -A "$DATA_DIR/verse")" ]; then
    echo "Extracting VerSe zip..."
    unzip -q "$DATA_DIR/dataset-verse19training.zip" -d "$DATA_DIR/verse"
else
    echo "VerSe dataset already extracted."
fi

echo "=== Downloading Mendeley Lumbar Spine MRI dataset ==="
if [ ! -f "$DATA_DIR/k57fr854j2-2.zip" ]; then
    wget -c -O "$DATA_DIR/k57fr854j2-2.zip" "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/k57fr854j2-2.zip"
else
    echo "Mendeley zip already downloaded."
fi

echo "=== Extracting Mendeley Lumbar Spine MRI dataset ==="
if [ ! -d "$DATA_DIR/lumbar_mri" ] || [ -z "$(ls -A "$DATA_DIR/lumbar_mri")" ]; then
    echo "Extracting Mendeley zip..."
    unzip -q "$DATA_DIR/k57fr854j2-2.zip" -d "$DATA_DIR/lumbar_mri"
else
    echo "Mendeley dataset already extracted."
fi

echo "=== Dataset download and extraction completed successfully! ==="
