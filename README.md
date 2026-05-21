# PyTorch U-Net Semantic Segmentation Experiment

This repository contains a modular, parameterizable, and GPU-optimized implementation of the **U-Net** architecture in PyTorch, designed to run on high-performance dual Quadro RTX 8000 systems.

---

## 1. What is U-Net?

**U-Net** was introduced by Olaf Ronneberger et al. in 2015 for Biomedical Image Segmentation. It has since become a cornerstone architecture for semantic segmentation, image-to-image translation, and even generative models (like the denoising U-Net in Stable Diffusion).

Its architecture is characterized by a symmetric "U" shape consisting of:
1. **An Encoder (Contracting Path):** Reduces spatial resolution to capture global context (semantic features).
2. **A Decoder (Expanding Path):** Restores spatial resolution to allow precise object localization.
3. **Skip Connections:** Connects corresponding levels of the encoder and decoder to preserve high-resolution spatial details.

### U-Net Spatial Structure

```text
Input Image (C=3, H=256, W=256)                                       Output Mask (C=2, H=256, W=256)
        │                                                                     ▲
        ▼                                                                     │
   [Level 1 Encoder] ───(64 channels skip connection)───► [Level 1 Decoder] ──┘
        │ MaxPool2d(2x2)                                      ▲ UpConv/Upsample
        ▼                                                     │
   [Level 2 Encoder] ───(128 channels skip connection)──► [Level 2 Decoder]
        │ MaxPool2d(2x2)                                      ▲ UpConv/Upsample
        ▼                                                     │
   [Level 3 Encoder] ───(256 channels skip connection)──► [Level 3 Decoder]
        │ MaxPool2d(2x2)                                      ▲ UpConv/Upsample
        ▼                                                     │
   [Level 4 Encoder] ───(512 channels skip connection)──► [Level 4 Decoder]
        │ MaxPool2d(2x2)                                      ▲ UpConv/Upsample
        ▼                                                     │
        └─────────────────► [Level 5: Bottleneck] ────────────┘
                            (1024 channels)
```

---

## 2. Module Implementations in `unet.py`

The architecture is implemented in a clean, object-oriented style inside [unet.py](file:///media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet/unet.py):

*   **`DoubleConv`:** 
    *   Applies a sequence of `[Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU]`.
    *   *Note:* The original paper did not include Batch Normalization (which wasn't popular in 2015). We have added it here to stabilize training and accelerate convergence.
    *   *Padding:* We use `padding=1` to ensure that spatial dimensions do not shrink after convolutions. This avoids the need for center-cropping in the skip connections.
*   **`Down`:**
    *   Applies `MaxPool2d(kernel_size=2, stride=2)` to halve the height and width, followed by the `DoubleConv` block to increase the number of channels.
*   **`Up`:**
    *   Upsamples the feature map (doubling spatial dimensions) using either a learnable `ConvTranspose2d` or bilinear `Upsample` interpolation.
    *   Concatenates the upsampled feature map with the high-resolution feature map from the encoder along the channel dimension (`dim=1`).
    *   Applies `DoubleConv` on the merged representation.
*   **`OutConv`:**
    *   A final 1x1 convolution mapping the $64$-channel feature representation to the target number of output classes.

---

## 3. Hardware Architecture & Multi-GPU Capabilities

This project is optimized to run on high-performance deep learning systems:
*   **Dual Quadro RTX 8000 GPUs:** The system detects and utilizes two identical GPUs, each featuring **48 GB of GDDR6 VRAM** (96 GB combined).
*   **Device Allocation:** By default, [main.py](file:///media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet/main.py) identifies if CUDA is available, maps execution to `cuda:0`, and runs a complete forward/backward pipeline with synthetic data.
*   **Parameter Count:** The base U-Net model contains **31,037,698 parameters**, all of which are fully trainable.

---

## 4. Local Environment Setup & Execution

We manage our dependencies cleanly and securely using **`uv`**, an extremely fast Python package manager and resolver.

### Running the Verification Script

1. **Synchronize Dependencies:**
   First, build and update the local virtual environment by running:
   ```bash
   uv sync
   ```
   *Note: This command will download and link the correct PyTorch and CUDA libraries to your local `.venv` environment.*

2. **Execute the Code:**
   To verify that PyTorch, CUDA, and the U-Net model are working correctly on your GPU, execute:
   ```bash
   uv run python main.py
   ```

### Verification Logs

When running `main.py`, you should see output similar to the following:
```text
============================================================
GPU / HARDWARE CHECK
============================================================
CUDA Available: True
Number of CUDA Devices: 2
  Device 0: Quadro RTX 8000 (47.27 GB VRAM)
  Device 1: Quadro RTX 8000 (47.26 GB VRAM)
Using device: cuda:0
============================================================
INITIALIZING U-NET FORWARD/BACKWARD TEST
============================================================
Creating UNet: 3 input channels -> 2 output classes
Total Parameters: 31,037,698
Trainable Parameters: 31,037,698

1. Generating dummy input tensor of shape: (2, 3, 256, 256)
2. Running forward pass through U-Net...
   Output tensor shape: [2, 2, 256, 256]
   ✔ Success! Output shape matches expectations.

3. Testing backward pass (gradient flow)...
   Computed CrossEntropyLoss: 0.7218
   ✔ Success! Gradients calculated and backpropagated correctly.
============================================================
```
