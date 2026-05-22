#!/bin/bash

# Kill any lingering python processes running main.py or auto_update.py to ensure a clean start
echo "Cleaning up any lingering main.py or auto_update.py processes..."
pkill -f "main.py --dataset" || true
pkill -f "auto_update.py" || true

# Check GPU resources
echo "Checking GPU status..."
nvidia-smi

# 1. Start Mendeley MRI -> VerSe 20 CT chain on GPU 0 in the background
echo "Starting Mendeley MRI training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py \
  --dataset lumbar_mri \
  --epochs 50 \
  --batch_size 4 \
  --base_channels 32 \
  --checkpoint_path best_model_lumbar_mri.pt \
  --plot_path verification_plot_lumbar_mri.png > mri_train.log 2>&1 && \
echo "Mendeley MRI complete. Starting VerSe '20 CT training on GPU 0..." && \
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --dataset verse20 \
  --epochs 50 \
  --batch_size 6 \
  --base_channels 32 \
  --checkpoint_path best_model_verse20.pt \
  --plot_path verification_plot_verse20.png > verse20_train.log 2>&1 &

# 2. Start VerSe 19 CT on GPU 1 in the background
echo "Starting VerSe '19 CT training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 uv run python main.py \
  --dataset verse19 \
  --epochs 50 \
  --batch_size 6 \
  --base_channels 32 \
  --checkpoint_path best_model_verse19.pt \
  --plot_path verification_plot_verse19.png > verse19_train.log 2>&1 &

# Wait 5 seconds to let the python processes initialize
echo "Waiting for processes to initialize..."
sleep 5

# Check if they started successfully
echo "Active main.py training processes:"
pgrep -fl "main.py --dataset" || echo "No active main.py training processes found!"

# 3. Start auto_update.py in the foreground (since this script will be executed as a background task, auto_update.py will block it until training is finished)
echo "Starting auto_update.py daemon to monitor progress and commit/push results..."
uv run python auto_update.py
