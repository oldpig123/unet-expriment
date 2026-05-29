import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityAdaptiveNormalization(nn.Module):
    """
    Modality-Adaptive Normalization (MAN)
    Standardizes features across the spatial domain (height and width) for each channel and sample.
    This is mathematically equivalent to Instance Normalization with learnable affine parameters.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, eps=eps, affine=True)

    def forward(self, x):
        return self.norm(x)

def gaussian_kernel_2d(kernel_size=5, sigma=1.5):
    """
    Generates a 2D Gaussian kernel.
    """
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    return gaussian_2d

class GaussianBlur(nn.Module):
    """
    Applies a fixed 2D Gaussian smoothing filter channel-wise (depthwise convolution).
    """
    def __init__(self, channels, kernel_size=5, sigma=1.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        kernel = gaussian_kernel_2d(kernel_size, sigma)
        kernel = kernel.repeat(channels, 1, 1, 1) # Shape: (C, 1, K, K)
        self.register_buffer('weight', kernel)
        self.pad = kernel_size // 2

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)

class SKConv(nn.Module):
    """
    Selective Kernel (SK) Convolution - Dynamic Receptive Field Block
    Fuses features from different receptive fields (dilation=1 and dilation=2) using channel attention.
    """
    def __init__(self, in_channels, out_channels, branches=2, reduction=16, min_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branches = branches

        # Path 1: Standard 3x3 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ModalityAdaptiveNormalization(out_channels),
            nn.ReLU(inplace=True)
        )

        # Path 2: 3x3 dilated convolution (dilation=2, effective RF is 5x5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            ModalityAdaptiveNormalization(out_channels),
            nn.ReLU(inplace=True)
        )

        # Dimensionality reduction for channel attention
        mid_dim = max(out_channels // reduction, min_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True)
        )

        # Attention mappings for each branch
        self.fcs = nn.ModuleList([
            nn.Conv2d(mid_dim, out_channels, kernel_size=1, bias=True)
            for _ in range(branches)
        ])

    def forward(self, x):
        # Split features into multi-scale receptive field branches
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)

        # Fuse branch features
        U = feat1 + feat2

        # Global average pooling + squeeze projection
        S = self.gap(U)
        Z = self.fc(S)

        # Compute soft attention weights across branches
        att_weights = [fc(Z) for fc in self.fcs]
        att_weights = torch.cat(att_weights, dim=1) # (B, 2*C, 1, 1)
        att_weights = F.softmax(att_weights.view(x.size(0), self.branches, self.out_channels, 1, 1), dim=1)

        # Dynamically select and merge features
        V = feat1 * att_weights[:, 0] + feat2 * att_weights[:, 1]
        return V

class ResidualBlock(nn.Module):
    """
    U-ResNet Residual Block with Gradient Smoothing and Modality-Adaptive Normalization.
    Mathematical formulation:
    F(x) = F_0(x) + lambda * grad(F_0(x))
    H(x) = F(x) + shortcut(x)
    """
    def __init__(self, in_channels, out_channels, stride=1, use_drf=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = ModalityAdaptiveNormalization(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_drf:
            self.conv2 = SKConv(out_channels, out_channels)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = ModalityAdaptiveNormalization(out_channels)

        # Shortcut projection if spatial dims or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                ModalityAdaptiveNormalization(out_channels)
            )

        # Trainable gradient smoothing coefficient (lambda)
        self.lambd = nn.Parameter(torch.tensor(0.1))

    def spatial_gradient(self, x):
        """
        Computes the spatial gradient magnitude using central differences.
        Matches the gradient term: grad(F_0)
        """
        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)

    def forward(self, x):
        identity = self.shortcut(x)

        # First convolution block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # Second convolution block F_0(x)
        F0 = self.conv2(out)
        F0 = self.norm2(F0)

        # Gradient smoothing term: F(x) = F_0(x) + lambda * spatial_gradient(F_0(x))
        grad_F0 = self.spatial_gradient(F0)
        F_x = F0 + self.lambd * grad_F0

        # Residual mapping
        out = F_x + identity
        out = self.relu(out)
        return out

class AdaptiveSkipConnection(nn.Module):
    """
    Adaptive Skip Connection (ASC)
    Performs spatial adaptive fusion instead of simple channel-wise concatenation.
    Formula: F_fused = alpha * F_shallow + (1 - alpha) * F_deep
    """
    def __init__(self, channels):
        super().__init__()
        # Outputs a spatial-only gating weight map alpha(s) in [0, 1]
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, F_shallow, F_deep):
        # Concatenate features along the channel dimension to estimate spatial weights
        concat_features = torch.cat([F_shallow, F_deep], dim=1)
        alpha = self.gate_conv(concat_features) # Shape: (B, 1, H, W)
        
        # Weighted sum fusion
        F_fused = alpha * F_shallow + (1.0 - alpha) * F_deep
        return F_fused

class ShapeAwareAttentionModule(nn.Module):
    """
    Shape-Aware Attention Module (SAAM)
    Fuses semantic features with pre-extracted contour priors and active contours.
    """
    def __init__(self, channels, kernel_size=5, sigma=1.5):
        super().__init__()
        self.channels = channels
        self.gaussian_blur = GaussianBlur(channels, kernel_size=kernel_size, sigma=sigma)
        self.mean_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, F_sem, C_s, C_hat):
        # F_sem: (B, C, H, W) - Semantic features
        # C_s: (B, 1, H_img, W_img) - Pre-extracted contour prior distance map
        # C_hat: (B, 1, H_img, W_img) - Active contour from input image
        B, C, H, W = F_sem.shape

        # Downsample/interpolate contour maps to match current feature spatial dimensions
        C_s_resized = F.interpolate(C_s, size=(H, W), mode='bilinear', align_corners=True)
        C_hat_resized = F.interpolate(C_hat, size=(H, W), mode='bilinear', align_corners=True)

        # 1. Shape correlation / consistency: Corr(s) = S(s) * C(s)
        Corr = F_sem * C_s_resized # (B, C, H, W)

        # 2. Initial spatial attention A_0(s) using Softmax over spatial dimensions
        A0 = F.softmax(Corr.view(B, C, -1), dim=-1).view(B, C, H, W)

        # 3. Dynamic shape adaptation factor beta(s) to handle pathological deformations
        # beta(s) = 1 - |C(s) - C_hat(s)| / (|C(s)| + |C_hat(s)| + eps)
        diff = torch.abs(C_s_resized - C_hat_resized)
        denom = torch.abs(C_s_resized) + torch.abs(C_hat_resized) + 1e-6
        beta = 1.0 - (diff / denom) # Shape: (B, 1, H, W)

        # 4. Final attention weight A(s) combining global prior A_0 and local neighborhood context
        mean_pool_A0 = self.mean_pool(A0)
        A = beta * A0 + (1.0 - beta) * mean_pool_A0 # Shape: (B, C, H, W)

        # 5. Feature optimization: F'(s) = F(s) * A(s) + GaussianFilter(F(s) * (1 - A(s)))
        term1 = F_sem * A
        term2 = self.gaussian_blur(F_sem * (1.0 - A))
        F_prime = term1 + term2

        return F_prime

class UResNet_Attention(nn.Module):
    """
    U-ResNet + Shape-Aware Attention Model for Spinal Segmentation.
    Backbone: 4 encoder stages, 4 decoder stages.
    - Shallow stages (Encoder 1/2, Decoder 3/4) contain 2 residual blocks.
    - Deep stages (Encoder 3/4, Decoder 1/2) contain 3 residual blocks with SKConv (DRF).
    - Fuses features via Adaptive Skip Connections (ASC) and Shape-Aware Attention Modules (SAAM).
    """
    def __init__(self, in_channels=3, n_classes=2, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        # Initial Projection Layer
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.init_norm = ModalityAdaptiveNormalization(base_channels)
        self.init_relu = nn.ReLU(inplace=True)

        # ============================================================
        # ENCODER PATH (Contracting Path)
        # ============================================================
        # Stage 1 (Shallow, 2 blocks): operates at original size (512x512)
        self.enc1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False),
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 512 -> 256

        # Stage 2 (Shallow, 2 blocks): operates at 256x256
        self.enc2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=1, use_drf=False),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1, use_drf=False)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 256 -> 128

        # Stage 3 (Deep, 3 blocks, uses SKConv): operates at 128x128
        self.enc3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=1, use_drf=True),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, use_drf=True),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, use_drf=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 128 -> 64

        # Stage 4 (Deep, 3 blocks, Bottleneck, uses SKConv): operates at 64x64
        self.enc4 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=1, use_drf=True),
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1, use_drf=True),
            ResidualBlock(base_channels * 8, base_channels * 8, stride=1, use_drf=True)
        )

        # ============================================================
        # DECODER PATH (Expanding Path)
        # ============================================================
        # Decoder Stage 1 (Deep): Upsamples 64x64 -> 128x128. Fuses with Encoder Stage 3 (128x128)
        self.up_conv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.asc1 = AdaptiveSkipConnection(base_channels * 4)
        self.saam1 = ShapeAwareAttentionModule(base_channels * 4)
        self.dec1 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, use_drf=True),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, use_drf=True),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, use_drf=True)
        )

        # Decoder Stage 2 (Deep): Upsamples 128x128 -> 256x256. Fuses with Encoder Stage 2 (256x256)
        self.up_conv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.asc2 = AdaptiveSkipConnection(base_channels * 2)
        self.saam2 = ShapeAwareAttentionModule(base_channels * 2)
        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1, use_drf=True),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1, use_drf=True),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1, use_drf=True)
        )

        # Decoder Stage 3 (Shallow): Upsamples 256x256 -> 512x512. Fuses with Encoder Stage 1 (512x512)
        self.up_conv3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.asc3 = AdaptiveSkipConnection(base_channels)
        self.saam3 = ShapeAwareAttentionModule(base_channels)
        self.dec3 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False),
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False)
        )

        # Decoder Stage 4 (Shallow): operates at 512x512. Fuses with Initial Projection Features (512x512)
        self.asc4 = AdaptiveSkipConnection(base_channels)
        self.saam4 = ShapeAwareAttentionModule(base_channels)
        self.dec4 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False),
            ResidualBlock(base_channels, base_channels, stride=1, use_drf=False)
        )

        # Output Classification Layer
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x, C_s, C_hat, use_saam=True):
        # Initial projection
        x_init = self.init_conv(x)
        x_init = self.init_norm(x_init)
        x_init = self.init_relu(x_init)

        # --- Encoder / Contracting Path ---
        e1 = self.enc1(x_init)              # Stage 1: (B, 64, 512, 512)
        e2 = self.enc2(self.pool1(e1))      # Stage 2: (B, 128, 256, 256)
        e3 = self.enc3(self.pool2(e2))      # Stage 3: (B, 256, 128, 128)
        e4 = self.enc4(self.pool3(e3))      # Stage 4 / Bottleneck: (B, 512, 64, 64)

        # --- Decoder / Expanding Path ---
        # Level 1 (Deep)
        d1_up = self.up_conv1(e4)
        d1_fused = self.asc1(e3, d1_up)
        d1_att = self.saam1(d1_fused, C_s, C_hat) if use_saam else d1_fused
        d1 = self.dec1(d1_att)              # (B, 256, 128, 128)

        # Level 2 (Deep)
        d2_up = self.up_conv2(d1)
        d2_fused = self.asc2(e2, d2_up)
        d2_att = self.saam2(d2_fused, C_s, C_hat) if use_saam else d2_fused
        d2 = self.dec2(d2_att)              # (B, 128, 256, 256)

        # Level 3 (Shallow)
        d3_up = self.up_conv3(d2)
        d3_fused = self.asc3(e1, d3_up)
        d3_att = self.saam3(d3_fused, C_s, C_hat) if use_saam else d3_fused
        d3 = self.dec3(d3_att)              # (B, 64, 512, 512)

        # Level 4 (Shallow)
        # Fuses Decoder Level 3 with Initial Projection Features (both at 512x512)
        d4_fused = self.asc4(x_init, d3)
        d4_att = self.saam4(d4_fused, C_s, C_hat) if use_saam else d4_fused
        d4 = self.dec4(d4_att)              # (B, 64, 512, 512)

        # Final Segmentation Output
        logits = self.out_conv(d4)
        return logits
