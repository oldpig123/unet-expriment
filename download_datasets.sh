#!/bin/bash
set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "=== Cleaning up invalid or empty zip files ==="
find "$DATA_DIR" -name "*.zip" -size 0 -delete

echo "=== Downloading VerSe 19 dataset ==="
if [ ! -f "$DATA_DIR/dataset-verse19training.zip" ]; then
    echo "Downloading VerSe zip..."
    wget -c -O "$DATA_DIR/dataset-verse19training.zip" "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip"
else
    echo "VerSe zip already exists and is non-empty."
fi

echo "=== Extracting VerSe 19 dataset ==="
if [ ! -d "$DATA_DIR/verse" ] || [ ! -d "$DATA_DIR/verse/dataset-verse19training" ]; then
    echo "Extracting VerSe zip..."
    unzip -q -o "$DATA_DIR/dataset-verse19training.zip" -d "$DATA_DIR/verse"
else
    echo "VerSe dataset already extracted."
fi

echo "=== Downloading Mendeley Lumbar Spine MRI dataset ==="
if [ ! -f "$DATA_DIR/k57fr854j2-2.zip" ]; then
    echo "Downloading Mendeley zip..."
    wget -c -O "$DATA_DIR/k57fr854j2-2.zip" "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/k57fr854j2-2.zip"
else
    echo "Mendeley zip already exists and is non-empty."
fi

echo "=== Extracting Mendeley Lumbar Spine MRI dataset ==="
# Check if Mendeley is fully extracted (the full dataset has hundreds of images)
if [ ! -d "$DATA_DIR/lumbar_mri" ] || [ ! -d "$DATA_DIR/lumbar_mri/images" ] || [ "$(ls -A "$DATA_DIR/lumbar_mri/images" | wc -l)" -lt 100 ]; then
    echo "Extracting Mendeley zip..."
    unzip -q -o "$DATA_DIR/k57fr854j2-2.zip" -d "$DATA_DIR/lumbar_mri"
else
    echo "Mendeley dataset already extracted."
fi

echo "=== Dataset download and extraction completed successfully! ==="
