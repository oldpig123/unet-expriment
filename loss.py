import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel_2d(kernel_size=5, sigma=1.5):
    """
    Generates a 2D Gaussian kernel.
    """
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    return gaussian_2d

class SpineLoss(nn.Module):
    """
    Dynamically Weighted Combined Loss Function (SpineLoss)
    Integrates:
    1. Region Loss (L_region): Density-weighted cross entropy loss focusing on small vertebrae and discs.
    2. Boundary Loss (L_boundary): Distance-weighted boundary loss.
    3. Volume Loss (L_vol): Absolute difference in target volume.
    
    Dynamic weights alpha and beta balance regional and boundary losses based on target proportion.
    """
    def __init__(self, gamma=0.5, lambda_density=1.0, lambda_boundary=1.0, kernel_size=5, sigma=1.5):
        super().__init__()
        self.gamma = gamma
        self.lambda_density = lambda_density
        self.lambda_boundary = lambda_boundary
        
        # Setup static Gaussian kernel for smoothing the boundary
        kernel = gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel.unsqueeze(0).unsqueeze(0)) # Shape: (1, 1, K, K)
        self.pad = kernel_size // 2

    def gaussian_blur2d(self, x):
        # x shape: (B, 1, H, W) or (B, C, H, W)
        B, C, H, W = x.shape
        # Apply convolution channel-wise
        kernel = self.gaussian_kernel.to(x.device).repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.pad, groups=C)

    def forward(self, logits, targets):
        # logits shape: (B, C_classes, H, W)
        # targets shape: (B, H, W)
        B, C_classes, H, W = logits.shape
        device = logits.device

        # Convert targets to one-hot encoding
        # G shape: (B, C_classes, H, W)
        G = F.one_hot(targets, num_classes=C_classes).permute(0, 3, 1, 2).float()

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)
        probs_clamp = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # 1. Calculate dynamic weights alpha and beta based on target proportion
        # G_foreground represents all target classes (class >= 1)
        if C_classes > 1:
            G_foreground = (targets > 0).float()
            target_proportion = G_foreground.sum(dim=(1, 2)) / (H * W) # Shape: (B,)
            # alpha = 1 - gamma * target_proportion
            alpha = 1.0 - self.gamma * target_proportion # Shape: (B,)
            beta = 1.0 - alpha
        else:
            alpha = torch.ones(B, device=device)
            beta = torch.zeros(B, device=device)

        # Reshape alpha and beta for broadcasting: (B, 1, 1, 1)
        alpha = alpha.view(B, 1, 1, 1)
        beta = beta.view(B, 1, 1, 1)

        # 2. Compute density weight w(s) and boundary distance weight d(s) for each class
        w = torch.ones_like(G)
        d = torch.ones_like(G)

        for c in range(1, C_classes):
            G_c = G[:, c:c+1] # (B, 1, H, W)

            # Target region density: local density via average pooling
            local_density = F.avg_pool2d(G_c, kernel_size=15, stride=1, padding=7)
            # w(s) = 1.0 + lambda_density * G_c(s) * exp(-local_density)
            w[:, c:c+1] = 1.0 + self.lambda_density * G_c * torch.exp(-local_density)

            # Boundary extraction: dilation - erosion
            dilation = F.max_pool2d(G_c, kernel_size=3, stride=1, padding=1)
            erosion = -F.max_pool2d(-G_c, kernel_size=3, stride=1, padding=1)
            boundary = dilation - erosion

            # d(s) = 1.0 + lambda_boundary * GaussianBlur(boundary)
            smoothed_boundary = self.gaussian_blur2d(boundary)
            d[:, c:c+1] = 1.0 + self.lambda_boundary * smoothed_boundary

        # 3. Region Loss (L_region)
        # Formula: w(s) * [-G(s) * ln P(s) - (1 - G(s)) * ln(1 - P(s))]
        loss_reg = w * (-G * torch.log(probs_clamp) - (1.0 - G) * torch.log(1.0 - probs_clamp))
        L_region = loss_reg.mean(dim=(1, 2, 3)) # Shape: (B,)

        # 4. Boundary Loss (L_boundary)
        # Formula: d(s) * |P(s) - G(s)|
        loss_bound = d * torch.abs(probs - G)
        L_boundary = loss_bound.mean(dim=(1, 2, 3)) # Shape: (B,)

        # 5. Volume Loss (L_vol)
        # Formula: | \sum P(s) - V_gt | / |Omega|
        # Evaluated for all target (foreground) classes
        if C_classes > 1:
            L_vol = torch.abs(probs[:, 1:].mean(dim=(2, 3)) - G[:, 1:].mean(dim=(2, 3))).mean(dim=1) # Shape: (B,)
        else:
            L_vol = torch.zeros(B, device=device)

        # 6. Final Loss
        # L_total = alpha * L_region + beta * L_boundary
        # L_final = L_total + 0.1 * L_vol
        L_total = alpha.squeeze() * L_region + beta.squeeze() * L_boundary
        L_final = L_total + 0.1 * L_vol

        return L_final.mean(), L_region.mean(), L_boundary.mean(), L_vol.mean()
