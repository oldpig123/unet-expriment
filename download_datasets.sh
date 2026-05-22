#!/bin/bash
set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "=== Cleaning up invalid or empty zip files ==="
find "$DATA_DIR" -name "*.zip" -size 0 -delete

# 1. Mendeley Lumbar Spine MRI PNG Dataset (zbf6b4pttk version 2)
echo "=== Downloading Mendeley Lumbar Spine MRI PNG dataset ==="
MENDELEY_ZIP="$DATA_DIR/zbf6b4pttk.zip"
MENDELEY_URL="https://data.mendeley.com/public-files/datasets/zbf6b4pttk/files/0a216a09-5349-4f4b-9a36-df28d436ae38/file_downloaded"

if [ ! -d "$DATA_DIR/lumbar_mri" ] || [ ! -d "$DATA_DIR/lumbar_mri/images" ] || [ "$(find "$DATA_DIR/lumbar_mri/images" -name "*.png" | wc -l)" -lt 100 ]; then
    if [ ! -f "$MENDELEY_ZIP" ]; then
        echo "Downloading Mendeley PNG dataset zip..."
        wget -c -O "$MENDELEY_ZIP" "$MENDELEY_URL"
    fi

    echo "Extracting Mendeley dataset..."
    mkdir -p "$DATA_DIR/lumbar_mri/images"
    mkdir -p "$DATA_DIR/lumbar_mri/labels"
    
    RAW_EXTRACT_DIR="$DATA_DIR/lumbar_mri_raw"
    mkdir -p "$RAW_EXTRACT_DIR"
    unzip -q -o "$MENDELEY_ZIP" -d "$RAW_EXTRACT_DIR"

    echo "Organizing Mendeley files..."
    mv "$RAW_EXTRACT_DIR"/05_Final_Ground_Truth_Data/Resized_Composite_Images/*.png "$DATA_DIR/lumbar_mri/images/"
    mv "$RAW_EXTRACT_DIR"/05_Final_Ground_Truth_Data/Resized_Label_Images/*.png "$DATA_DIR/lumbar_mri/labels/"

    echo "Cleaning up Mendeley raw extraction folder and zip..."
    rm -rf "$RAW_EXTRACT_DIR"
    rm -f "$MENDELEY_ZIP"
else
    echo "Mendeley Lumbar Spine MRI dataset already downloaded and extracted."
fi

# 2. VerSe '19
echo "=== Downloading and extracting VerSe '19 from Bonescreen ==="
VERSE19_RAW="$DATA_DIR/verse19_raw"
mkdir -p "$VERSE19_RAW"

for split in training validation test; do
    ZIP_NAME="dataset-verse19${split}.zip"
    ZIP_PATH="$DATA_DIR/$ZIP_NAME"
    URL="https://s3.bonescreen.de/public/VerSe-complete/$ZIP_NAME"
    
    echo "Downloading VerSe '19 $split split..."
    wget -c -O "$ZIP_PATH" "$URL"
    
    echo "Extracting VerSe '19 $split split..."
    unzip -q -o "$ZIP_PATH" -d "$VERSE19_RAW"
    
    echo "Cleaning up $ZIP_NAME..."
    rm -f "$ZIP_PATH"
done

# 3. VerSe '20
echo "=== Downloading and extracting VerSe '20 from Bonescreen ==="
VERSE20_RAW="$DATA_DIR/verse20_raw"
mkdir -p "$VERSE20_RAW"

for split in training validation test; do
    ZIP_NAME="dataset-verse20${split}.zip"
    ZIP_PATH="$DATA_DIR/$ZIP_NAME"
    URL="https://s3.bonescreen.de/public/VerSe-complete/$ZIP_NAME"
    
    echo "Downloading VerSe '20 $split split..."
    wget -c -O "$ZIP_PATH" "$URL"
    
    echo "Extracting VerSe '20 $split split..."
    unzip -q -o "$ZIP_PATH" -d "$VERSE20_RAW"
    
    echo "Cleaning up $ZIP_NAME..."
    rm -f "$ZIP_PATH"
done

echo "=== Dataset download and organization scripts completed successfully! ==="
