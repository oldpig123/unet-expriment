import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U-Net Architecture Explanation:
================================

U-Net is a convolutional neural network designed for fast and precise image segmentation.
Its name comes from its symmetric "U" shape, consisting of two main parts:
1. The Contracting Path (Encoder): Captures context/features but loses spatial resolution.
2. The Expanding Path (Decoder): Restores spatial resolution for precise localization.

Key Feature: Skip Connections
-----------------------------
As the network goes deeper, spatial details (like edges and fine boundaries) are lost.
U-Net solves this by copying high-resolution feature maps from the Encoder and concatenating
them with the corresponding upsampled feature maps in the Decoder. This allows the network
to retain precise spatial information.

Structure of the Implementation:
--------------------------------
- DoubleConv: A basic block of two 3x3 Convolutions, each followed by Batch Normalization
  (which stabilizes training) and a ReLU activation function.
- Down: Encoder step. Performs 2x2 Max Pooling (halving spatial size) followed by DoubleConv.
- Up: Decoder step. Upsamples the feature map (doubling spatial size), concatenates the
  corresponding skip connection from the Encoder, and applies a DoubleConv.
- OutConv: A final 1x1 Convolution to map the channels to the target output classes/channels.
"""

class DoubleConv(nn.Module):
    """
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    This represents the repeating convolution block used at each level of U-Net.
    We use padding=1 to keep the spatial dimensions the same (padded convolutions).
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with MaxPool2d (stride=2) then double conv.
    This halves the height and width of the input image while doubling/increasing channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    1. Upsamples the input map (doubling height and width).
    2. Concatenates the skip connection tensor (from the encoder path) along the channel dimension (dim 1).
    3. Passes the concatenated tensor through a DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # If bilinear upsampling, we use regular bilinear interpolation and standard 1x1 convolution
        # to reduce channels. Otherwise, we use learnable ConvTranspose2d (transposed convolution).
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After bilinear upsampling, the channel size is in_channels // 2.
            # We will concatenate the skip connection (which also has in_channels // 2 channels).
            # Thus, the input to DoubleConv will be in_channels.
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: incoming tensor from the lower decoder layer
        # x2: skip connection tensor from the corresponding encoder layer
        x1 = self.up(x1)
        
        # In case the input sizes are not divisible by 16, there might be a 1-pixel mismatch.
        # We check and pad x1 if necessary to match x2's spatial dimensions.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # F.pad format is (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension (dim=1)
        # e.g. if x1 has shape (B, C, H, W) and x2 has shape (B, C, H, W),
        # the concatenated tensor has shape (B, 2C, H, W)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution layer.
    Maps the high-dimensional feature representation (e.g. 64 channels)
    down to the desired number of output channels/classes (e.g. 1 for binary classification,
    or C for multi-class segmentation).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Full U-Net Model.
    
    Parameters:
    -----------
    n_channels: int
        Number of input channels (e.g. 3 for RGB images, 1 for grayscale).
    n_classes: int
        Number of target output classes/channels.
    bilinear: bool (default=False)
        Whether to use bilinear interpolation for upsampling or Transposed Convolution.
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Standard U-Net channel progression: 64 -> 128 -> 256 -> 512 -> 1024
        # We start with a DoubleConv mapping input channels to 64.
        self.inc = DoubleConv(n_channels, 64)
        
        # Encoder (Contracting Path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # If bilinear upsampling is used, the bottleneck channel layout is adjusted.
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final Output Layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder / Contracting Path ---
        x1 = self.inc(x)       # Level 1 features
        x2 = self.down1(x1)    # Level 2 features
        x3 = self.down2(x2)    # Level 3 features
        x4 = self.down3(x3)    # Level 4 features
        x5 = self.down4(x4)    # Bottleneck representation
        
        # --- Decoder / Expanding Path with Skip Connections ---
        # We upsample the lower feature map and concatenate it with the corresponding encoder level map.
        x = self.up1(x5, x4)   # Upsample and concat Level 4
        x = self.up2(x, x3)    # Upsample and concat Level 3
        x = self.up3(x, x2)    # Upsample and concat Level 2
        x = self.up4(x, x1)    # Upsample and concat Level 1
        
        # --- Output Classification ---
        logits = self.outc(x)
        return logits
