#!/bin/bash
set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "=== Cleaning up invalid or empty zip files ==="
find "$DATA_DIR" -name "*.zip" -size 0 -delete

# 1. VerSe 19
echo "=== Downloading VerSe 19 dataset ==="
if [ ! -f "$DATA_DIR/dataset-verse19training.zip" ]; then
    echo "Downloading VerSe 19 zip..."
    wget -c -O "$DATA_DIR/dataset-verse19training.zip" "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip"
else
    echo "VerSe 19 zip already exists and is non-empty."
fi

echo "=== Extracting VerSe 19 dataset ==="
if [ ! -d "$DATA_DIR/verse19" ] || [ ! -d "$DATA_DIR/verse19/dataset-verse19training" ]; then
    echo "Extracting VerSe 19 zip..."
    mkdir -p "$DATA_DIR/verse19"
    unzip -q -o "$DATA_DIR/dataset-verse19training.zip" -d "$DATA_DIR/verse19"
else
    echo "VerSe 19 dataset already extracted."
fi

# 2. VerSe 20
echo "=== Downloading VerSe 20 dataset ==="
if [ ! -f "$DATA_DIR/dataset-verse20training.zip" ]; then
    echo "Downloading VerSe 20 zip..."
    wget -c -O "$DATA_DIR/dataset-verse20training.zip" "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip"
else
    echo "VerSe 20 zip already exists and is non-empty."
fi

echo "=== Extracting VerSe 20 dataset ==="
if [ ! -d "$DATA_DIR/verse20" ] || [ ! -d "$DATA_DIR/verse20/dataset-verse20training" ]; then
    echo "Extracting VerSe 20 zip..."
    mkdir -p "$DATA_DIR/verse20"
    unzip -q -o "$DATA_DIR/dataset-verse20training.zip" -d "$DATA_DIR/verse20"
else
    echo "VerSe 20 dataset already extracted."
fi

# 3. Mendeley Lumbar Spine MRI
echo "=== Downloading Mendeley Lumbar Spine MRI dataset ==="
if [ ! -f "$DATA_DIR/k57fr854j2-2.zip" ]; then
    echo "Downloading Mendeley zip..."
    wget -c -O "$DATA_DIR/k57fr854j2-2.zip" "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/k57fr854j2-2.zip"
else
    echo "Mendeley zip already exists and is non-empty."
fi

echo "=== Extracting Mendeley Lumbar Spine MRI dataset ==="
if [ ! -d "$DATA_DIR/lumbar_mri" ] || [ ! -d "$DATA_DIR/lumbar_mri/images" ] || [ "$(ls -A "$DATA_DIR/lumbar_mri/images" | wc -l)" -lt 100 ]; then
    echo "Extracting Mendeley zip..."
    unzip -q -o "$DATA_DIR/k57fr854j2-2.zip" -d "$DATA_DIR/lumbar_mri"
else
    echo "Mendeley dataset already extracted."
fi

echo "=== Dataset download and extraction completed successfully! ==="
