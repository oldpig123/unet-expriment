import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

# Import custom model and loss
from model import UResNet_Attention
from loss import SpineLoss

def verify_gpu():
    """
    Checks the system's hardware configuration and outputs GPU information.
    """
    print("=" * 60)
    print("GPU / HARDWARE CHECK")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) # in GB
            print(f"  Device {i}: {name} ({memory:.2f} GB VRAM)")
        
        # Select device 0 for verification
        device = torch.device("cuda:0")
    else:
        print("No GPU detected. Falling back to CPU.")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print("=" * 60 + "\n")
    return device

def compute_distance_map_pytorch(mask):
    """
    Vectorized, GPU-compatible Euclidean Distance Transform (EDT) approximation in PyTorch.
    Computes the distance from each pixel to the nearest boundary/edge pixel.
    """
    B, _, H, W = mask.shape
    device = mask.device
    
    # Create spatial coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    ) # (H, W) each
    
    dist_maps = []
    for b in range(B):
        edge_indices = torch.nonzero(mask[b, 0]) # (N, 2)
        if len(edge_indices) == 0:
            # If no edge pixels, return maximum possible distance map
            max_dist = math.sqrt(H**2 + W**2)
            dist_maps.append(torch.full((1, H, W), max_dist, device=device))
            continue
            
        ey = edge_indices[:, 0].float() # (N,)
        ex = edge_indices[:, 1].float() # (N,)
        
        min_dist_sq = torch.full((H, W), float('inf'), device=device)
        # Process in chunks of edge pixels to avoid GPU memory overflow (O(N*H*W))
        chunk_size = 256
        for i in range(0, len(ey), chunk_size):
            ey_chunk = ey[i:i+chunk_size].view(-1, 1, 1) # (C, 1, 1)
            ex_chunk = ex[i:i+chunk_size].view(-1, 1, 1) # (C, 1, 1)
            
            # Squared Euclidean Distance from grid to all chunk edge pixels: (C, H, W)
            dist_sq = (grid_y.unsqueeze(0) - ey_chunk)**2 + (grid_x.unsqueeze(0) - ex_chunk)**2
            chunk_min, _ = torch.min(dist_sq, dim=0) # (H, W)
            min_dist_sq = torch.minimum(min_dist_sq, chunk_min)
            
        dist_maps.append(torch.sqrt(min_dist_sq).unsqueeze(0))
        
    return torch.stack(dist_maps, dim=0) # (B, 1, H, W)

def extract_edges_and_distances(targets, images):
    """
    Computes contour prior C_s from target masks and active contour C_hat from input images.
    Returns:
        C_s: normalized target boundary distance map of shape (B, 1, H, W)
        C_hat: normalized image edge boundary distance map of shape (B, 1, H, W)
    """
    B, H, W = targets.shape
    device = targets.device
    
    # 1. Contour Prior C_s from Targets
    fg_mask = (targets > 0).float().unsqueeze(1) # (B, 1, H, W)
    
    # Morphological boundary: dilation - erosion
    dilation = F.max_pool2d(fg_mask, kernel_size=3, stride=1, padding=1)
    erosion = -F.max_pool2d(-fg_mask, kernel_size=3, stride=1, padding=1)
    boundary_gt = (dilation - erosion) > 0.5
    
    C_s = compute_distance_map_pytorch(boundary_gt)
    
    # Min-max normalization for C_s to [0, 1] per sample
    C_s_min = C_s.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    C_s_max = C_s.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    C_s = (C_s - C_s_min) / (C_s_max - C_s_min + 1e-8)
    
    # 2. Active Contour C_hat from Input Image (Sobel filter-based edge map)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    
    gx = F.conv2d(images, kx, padding=1)
    gy = F.conv2d(images, ky, padding=1)
    
    edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    # Thresholding to obtain binary edge map (Canny edge approximation)
    edge_mag_max = edge_mag.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    boundary_img = edge_mag > (0.1 * edge_mag_max)
    
    C_hat = compute_distance_map_pytorch(boundary_img)
    
    # Min-max normalization for C_hat to [0, 1] per sample
    C_hat_min = C_hat.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    C_hat_max = C_hat.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    C_hat = (C_hat - C_hat_min) / (C_hat_max - C_hat_min + 1e-8)
    
    return C_s, C_hat

def generate_simulated_spine_slice(batch_size, height, width, device):
    """
    Generates simulated spinal sagittal CT/MRI slices with:
    - 5 vertebrae (class 1, blue/cyan in plots)
    - 4 intervertebral discs (class 2, orange/red in plots)
    - Spinal canal structures and random noise.
    """
    images = torch.zeros((batch_size, 1, height, width), device=device)
    targets = torch.zeros((batch_size, height, width), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Slightly jitter position and dimensions to simulate anatomical variation
        center_x = width // 2 + torch.randint(-15, 16, (1,)).item()
        vert_w = width // 5 + torch.randint(-10, 11, (1,)).item()
        vert_h = height // 12 + torch.randint(-4, 5, (1,)).item()
        disc_h = height // 30 + torch.randint(-2, 3, (1,)).item()
        
        y = torch.arange(height, device=device).view(height, 1)
        x = torch.arange(width, device=device).view(1, width)
        
        start_y = height // 7
        spacing = vert_h + disc_h + 8
        
        for i in range(5):
            v_cy = start_y + i * spacing + vert_h // 2
            
            # Vertebra mask (class 1)
            dist_vert = ((y - v_cy) / (vert_h / 2))**4 + ((x - center_x) / (vert_w / 2))**4
            vert_mask = dist_vert < 1.0
            targets[b][vert_mask] = 1
            
            # Intervertebral disc mask (class 2)
            if i < 4:
                d_cy = v_cy + vert_h // 2 + disc_h // 2 + 4
                dist_disc = ((y - d_cy) / (disc_h / 2))**2 + ((x - center_x) / (vert_w * 0.85 / 2))**2
                disc_mask = dist_disc < 1.0
                targets[b][disc_mask] = 2
                
        # Fill image signal intensities
        img = torch.zeros((height, width), device=device)
        img[targets[b] == 0] = 0.15 # Background
        img[targets[b] == 1] = 0.65 # Vertebrae
        img[targets[b] == 2] = 0.85 # Discs
        
        # Add a simulated spinal canal CSF band (bright strip behind vertebrae)
        canal_mask = (torch.abs(torch.arange(width, device=device) - (center_x - vert_w // 2 - 12)) < 8)
        img[:, canal_mask] = 0.5
        
        # Add random Gaussian noise to simulate MRI/CT scanner noise
        noise = torch.randn((height, width), device=device) * 0.05
        img = torch.clamp(img + noise, 0.0, 1.0)
        
        images[b, 0] = img
        
    return images, targets

def save_verification_plot(image, target, contour_prior, pred_prob, pred_mask, filepath="verification_plot.png"):
    """
    Saves a high-quality visualization panel containing:
    1. Input Spine slice
    2. Ground truth mask (vertebrae and discs)
    3. Contour prior distance map C_s
    4. Model prediction probability map
    5. Predicted segmentation mask
    """
    image_np = image.cpu().squeeze().numpy()
    target_np = target.cpu().numpy()
    contour_prior_np = contour_prior.cpu().squeeze().numpy()
    
    h, w = target_np.shape
    
    # RGB image mapping for target
    target_rgb = np.zeros((h, w, 3))
    target_rgb[target_np == 1] = [0.1, 0.6, 0.9] # Blue/Cyan for vertebrae
    target_rgb[target_np == 2] = [0.9, 0.4, 0.1] # Orange/Red for discs
    
    # RGB image mapping for prediction
    pred_mask_np = pred_mask.cpu().squeeze().numpy()
    pred_mask_rgb = np.zeros((h, w, 3))
    pred_mask_rgb[pred_mask_np == 1] = [0.1, 0.6, 0.9]
    pred_mask_rgb[pred_mask_np == 2] = [0.9, 0.4, 0.1]
    
    # Probability map of target foreground (sum of classes 1 and 2)
    pred_prob_fg = pred_prob[1] + pred_prob[2]
    pred_prob_np = pred_prob_fg.cpu().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), facecolor='#121212')
    
    # 1. Input Image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Input Spinal Slice", color='white', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth Mask
    axes[1].imshow(target_rgb)
    axes[1].set_title("Ground Truth Mask", color='white', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Contour Prior
    axes[2].imshow(contour_prior_np, cmap='jet')
    axes[2].set_title("Contour Prior C_s", color='white', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Predicted Probability
    axes[3].imshow(pred_prob_np, cmap='hot')
    axes[3].set_title("Predicted Prob Map", color='white', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # 5. Predicted Mask
    axes[4].imshow(pred_mask_rgb)
    axes[4].set_title("Predicted Mask", color='white', fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[INFO] Verification plot successfully saved to: {filepath}\n")

def check_dataset_paths():
    """
    Checks for the local VerSe and Lumbar Spine MRI datasets.
    Prints instructions and warnings if not found.
    """
    data_root = "./data"
    verse_path = os.path.join(data_root, "verse")
    mri_path = os.path.join(data_root, "lumbar_mri")
    
    print("=" * 60)
    print("CHECKING DATASET CONFIGURATION")
    print("=" * 60)
    
    mri_exists = os.path.exists(mri_path)
    verse_exists = os.path.exists(verse_path)
    
    if mri_exists:
        print(f"✔ Lumbar Spine MRI Dataset detected at {mri_path}")
    else:
        print(f"⚠ Lumbar Spine MRI Dataset NOT detected at {mri_path}")
        print("  - To configure, download the dataset from Mendeley Data:")
        print("    URL: https://data.mendeley.com/datasets/k57fr854j2/2")
        print(f"    and extract it to {mri_path}/")
        
    if verse_exists:
        print(f"✔ VerSe Dataset detected at {verse_path}")
    else:
        print(f"⚠ VerSe Dataset NOT detected at {verse_path}")
        print("  - To configure, download the dataset from GitHub / OSF:")
        print("    URL: https://github.com/anjany/verse")
        print(f"    and extract it to {verse_path}/")
        
    if not mri_exists or not verse_exists:
        print("\n--> Fallback activated: Generating high-quality simulated spine slices.")
        print("    This enables complete network execution, verification, and loss backpropagation.")
    print("=" * 60 + "\n")
    return not (mri_exists and verse_exists)

def main():
    device = verify_gpu()
    is_fallback = check_dataset_paths()
    
    # 1. Initialize U-ResNet + Shape-Aware Attention Network
    # in_channels = 1 (grayscale CT/MRI), n_classes = 3 (background, vertebrae, discs)
    print("Initializing UResNet_Attention model...")
    model = UResNet_Attention(in_channels=1, n_classes=3, base_channels=32) # Using base_channels=32 for lightweight testing
    model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Target Parameter Count: ~14.5M (when base_channels=64, base_channels=32 is ~3.6M)")
    
    # 2. Initialize SpineLoss
    criterion = SpineLoss(gamma=0.5, lambda_density=1.5, lambda_boundary=1.5)
    
    # 3. Create dummy batch to test forward shape tracking
    print("\n" + "-" * 50)
    print("FORWARD PASS & INTERMEDIATE RESOLUTION CHECK")
    print("-" * 50)
    images, targets = generate_simulated_spine_slice(batch_size=2, height=512, width=512, device=device)
    C_s, C_hat = extract_edges_and_distances(targets, images)
    
    print(f"Input image shape: {list(images.shape)}")
    print(f"Ground truth target shape: {list(targets.shape)}")
    print(f"Contour prior C_s shape: {list(C_s.shape)}")
    print(f"Active contour C_hat shape: {list(C_hat.shape)}")
    
    # Forward pass
    model.train()
    logits = model(images, C_s, C_hat)
    print(f"Output logits shape: {list(logits.shape)}")
    
    expected_shape = [2, 3, 512, 512]
    if list(logits.shape) == expected_shape:
        print("✔ Success! Output shape matches [Batch, Classes, Height, Width] perfectly.")
    else:
        print(f"❌ Error: Expected shape {expected_shape}, but got {list(logits.shape)}")
        sys.exit(1)
        
    # 4. Verify loss computation & backpropagation
    print("\n" + "-" * 50)
    print("LOSS COMPUTATION & BACKPROPAGATION TEST")
    print("-" * 50)
    loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
    print(f"Loss Components:")
    print(f"  - Total Loss: {loss.item():.6f}")
    print(f"  - Region Loss: {loss_reg.item():.6f}")
    print(f"  - Boundary Loss: {loss_bound.item():.6f}")
    print(f"  - Volume Loss: {loss_vol.item():.6f}")
    
    print("\nTesting backward pass...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    # Check if gradients flow to initial conv layers
    init_grad = model.init_conv.weight.grad
    if init_grad is not None and init_grad.abs().sum() > 0:
        print("✔ Success! Gradients calculated and backpropagated correctly through the whole network.")
    else:
        print("❌ Error: Gradients were not computed or are all zero.")
        sys.exit(1)
        
    # 5. Run a mini training sequence of 20 steps to show the loss decrease
    print("\n" + "-" * 50)
    print("RUNNING DEMO TRAINING SEQUENCE (20 STEPS)")
    print("-" * 50)
    model.train()
    for step in range(1, 21):
        optimizer.zero_grad()
        logits = model(images, C_s, C_hat)
        loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d}/20 | Loss: {loss.item():.6f} (Reg: {loss_reg.item():.4f}, Bound: {loss_bound.item():.4f}, Vol: {loss_vol.item():.4f})")
            
    # 6. Save visual verification plot from step 20
    model.eval()
    with torch.no_grad():
        logits = model(images, C_s, C_hat)
        probs = F.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1)
        
        # Plot the first item in the batch
        save_verification_plot(
            image=images[0],
            target=targets[0],
            contour_prior=C_s[0],
            pred_prob=probs[0],
            pred_mask=pred_mask[0],
            filepath="verification_plot.png"
        )
        
    print("=" * 60)
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
