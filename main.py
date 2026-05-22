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
from dataset import get_dataloaders
import argparse

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

def compute_distance_map_pytorch(mask, downsample_res=128):
    """
    Vectorized, GPU-compatible Euclidean Distance Transform (EDT) approximation in PyTorch.
    Computes the distance from each pixel to the nearest boundary/edge pixel.
    To avoid O(N*H*W) overhead on large resolutions, we perform computation on a downsampled
    grid (e.g. 128x128) and interpolate back to original size.
    """
    B, C, H, W = mask.shape
    device = mask.device
    
    # If the mask is already small or equal to downsample resolution, do not downsample
    down_h = min(H, downsample_res)
    down_w = min(W, downsample_res)
    
    if H != down_h or W != down_w:
        mask_down = F.interpolate(mask.float(), size=(down_h, down_w), mode='bilinear', align_corners=False) > 0.2
    else:
        mask_down = mask
        
    # Create spatial coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(down_h, dtype=torch.float32, device=device),
        torch.arange(down_w, dtype=torch.float32, device=device),
        indexing='ij'
    ) # (down_h, down_w) each
    
    dist_maps = []
    for b in range(B):
        edge_indices = torch.nonzero(mask_down[b, 0]) # (N, 2)
        if len(edge_indices) == 0:
            # If no edge pixels, return maximum possible distance map
            max_dist = math.sqrt(down_h**2 + down_w**2)
            dist_maps.append(torch.full((1, down_h, down_w), max_dist, device=device))
            continue
            
        ey = edge_indices[:, 0].float() # (N,)
        ex = edge_indices[:, 1].float() # (N,)
        
        min_dist_sq = torch.full((down_h, down_w), float('inf'), device=device)
        # Larger chunk size can be used due to smaller spatial resolution
        chunk_size = 512
        for i in range(0, len(ey), chunk_size):
            ey_chunk = ey[i:i+chunk_size].view(-1, 1, 1) # (C, 1, 1)
            ex_chunk = ex[i:i+chunk_size].view(-1, 1, 1) # (C, 1, 1)
            
            # Squared Euclidean Distance from grid to all chunk edge pixels: (C, down_h, down_w)
            dist_sq = (grid_y.unsqueeze(0) - ey_chunk)**2 + (grid_x.unsqueeze(0) - ex_chunk)**2
            chunk_min, _ = torch.min(dist_sq, dim=0) # (down_h, down_w)
            min_dist_sq = torch.minimum(min_dist_sq, chunk_min)
            
        # Convert squared distance to actual distance
        dist_map = torch.sqrt(min_dist_sq).unsqueeze(0)
        
        # Scale distance values back to match the original image coordinates scale
        if H != down_h:
            dist_map = dist_map * (H / down_h)
            
        dist_maps.append(dist_map)
        
    dist_maps_tensor = torch.stack(dist_maps, dim=0) # (B, 1, down_h, down_w)
    
    if H != down_h or W != down_w:
        dist_maps_tensor = F.interpolate(dist_maps_tensor, size=(H, W), mode='bilinear', align_corners=False)
        
    return dist_maps_tensor

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
    
    # Probability map of target foreground (sum of classes 1 and 2 if multi-class, else class 1)
    if pred_prob.shape[0] > 2:
        pred_prob_fg = pred_prob[1] + pred_prob[2]
    else:
        pred_prob_fg = pred_prob[1]
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
    parser = argparse.ArgumentParser(description="U-ResNet Spine Segmentation")
    parser.add_argument("--dataset", type=str, choices=["simulated", "verse19", "verse20", "lumbar_mri"], default="simulated", help="Dataset to use for training/eval")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where datasets are stored")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (ignored for simulated)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--plot_path", type=str, default="verification_plot.png", help="Path to save the verification plot")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per epoch for fast verification")
    args = parser.parse_args()

    device = verify_gpu()
    
    # 1. Initialize U-ResNet + Shape-Aware Attention Network
    # in_channels = 1 (grayscale CT/MRI), n_classes = 3 (background, vertebrae, discs) or 2 (background, vertebrae)
    n_classes = 3 if args.dataset in ["lumbar_mri", "simulated"] else 2
    print(f"Initializing UResNet_Attention model with {n_classes} classes...")
    model = UResNet_Attention(in_channels=1, n_classes=n_classes, base_channels=32) # Using base_channels=32 for lightweight testing
    model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Target Parameter Count: ~14.5M (when base_channels=64, base_channels=32 is ~3.6M)")
    
    # 2. Initialize SpineLoss
    criterion = SpineLoss(gamma=0.5, lambda_density=1.5, lambda_boundary=1.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.dataset == "simulated":
        check_dataset_paths()
        print("\n" + "-" * 50)
        print("RUNNING SIMULATED DATA DEMO")
        print("-" * 50)
        images, targets = generate_simulated_spine_slice(batch_size=2, height=512, width=512, device=device)
        C_s, C_hat = extract_edges_and_distances(targets, images)
        
        print(f"Input image shape: {list(images.shape)}")
        print(f"Ground truth target shape: {list(targets.shape)}")
        print(f"Contour prior C_s shape: {list(C_s.shape)}")
        print(f"Active contour C_hat shape: {list(C_hat.shape)}")
        
        # Test shape tracking
        model.train()
        logits = model(images, C_s, C_hat)
        print(f"Output logits shape: {list(logits.shape)}")
        
        expected_shape = [2, 3, 512, 512]
        if list(logits.shape) != expected_shape:
            print(f"❌ Error: Expected shape {expected_shape}, but got {list(logits.shape)}")
            sys.exit(1)
            
        # Verify loss and backward pass
        loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Run 20 steps
        for step in range(1, 21):
            optimizer.zero_grad()
            logits = model(images, C_s, C_hat)
            loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if step % 5 == 0 or step == 1:
                print(f"Step {step:02d}/20 | Loss: {loss.item():.6f} (Reg: {loss_reg.item():.4f}, Bound: {loss_bound.item():.4f}, Vol: {loss_vol.item():.4f})")
                
        model.eval()
        with torch.no_grad():
            logits = model(images, C_s, C_hat)
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1)
            save_verification_plot(
                image=images[0],
                target=targets[0],
                contour_prior=C_s[0],
                pred_prob=probs[0],
                pred_mask=pred_mask[0],
                filepath=args.plot_path
            )
            
    else:
        # Real dataset path
        print(f"\n" + "-" * 50)
        print(f"TRAINING ON REAL DATASET: {args.dataset.upper()}")
        print("-" * 50)
        
        train_loader, val_loader = get_dataloaders(args.dataset, args.data_dir, batch_size=args.batch_size)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_reg = 0.0
            train_bound = 0.0
            train_vol = 0.0
            
            num_batches = 0
            for step_idx, (images, targets) in enumerate(train_loader):
                if args.max_steps is not None and step_idx >= args.max_steps:
                    break
                images, targets = images.to(device), targets.to(device)
                C_s, C_hat = extract_edges_and_distances(targets, images)
                
                optimizer.zero_grad()
                logits = model(images, C_s, C_hat)
                loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_reg += loss_reg.item()
                train_bound += loss_bound.item()
                train_vol += loss_vol.item()
                num_batches += 1
                
            if num_batches > 0:
                train_loss /= num_batches
                train_reg /= num_batches
                train_bound /= num_batches
                train_vol /= num_batches
            
            # Validation
            model.eval()
            val_dice_list = []
            val_iou_list = []
            val_hd_list = []
            
            with torch.no_grad():
                for val_step_idx, (val_images, val_targets) in enumerate(val_loader):
                    if args.max_steps is not None and val_step_idx >= max(1, args.max_steps // 4):
                        break
                    val_images, val_targets = val_images.to(device), val_targets.to(device)
                    val_C_s, val_C_hat = extract_edges_and_distances(val_targets, val_images)
                    
                    val_logits = model(val_images, val_C_s, val_C_hat)
                    val_probs = F.softmax(val_logits, dim=1)
                    val_preds = torch.argmax(val_probs, dim=1)
                    
                    # Compute Dice and IoU
                    for c in [1, 2]:
                        # Skip class 2 (discs) if verse since it doesn't exist
                        if args.dataset.startswith("verse") and c == 2:
                            continue
                        pred_c = (val_preds == c)
                        target_c = (val_targets == c)
                        
                        intersection = (pred_c & target_c).float().sum()
                        union = (pred_c | target_c).float().sum()
                        sum_total = pred_c.float().sum() + target_c.float().sum()
                        
                        if target_c.sum() > 0:
                            dice = (2.0 * intersection) / (sum_total + 1e-8)
                            iou = intersection / (union + 1e-8)
                            val_dice_list.append(dice.item())
                            val_iou_list.append(iou.item())
                            
                    # Compute Hausdorff Distance
                    # Edge mapping
                    val_pred_boundary = (F.max_pool2d((val_preds > 0).float().unsqueeze(1), kernel_size=3, stride=1, padding=1) -
                                         -F.max_pool2d(-(val_preds > 0).float().unsqueeze(1), kernel_size=3, stride=1, padding=1)) > 0.5
                    val_target_boundary = (F.max_pool2d((val_targets > 0).float().unsqueeze(1), kernel_size=3, stride=1, padding=1) -
                                           -F.max_pool2d(-(val_targets > 0).float().unsqueeze(1), kernel_size=3, stride=1, padding=1)) > 0.5
                    
                    for b in range(val_images.shape[0]):
                        pb = val_pred_boundary[b, 0]
                        tb = val_target_boundary[b, 0]
                        
                        if pb.sum() > 0 and tb.sum() > 0:
                            dist_t = compute_distance_map_pytorch(tb.unsqueeze(0).unsqueeze(0)).squeeze()
                            dist_p = compute_distance_map_pytorch(pb.unsqueeze(0).unsqueeze(0)).squeeze()
                            hd_t = dist_t[pb].max().item()
                            hd_p = dist_p[tb].max().item()
                            val_hd_list.append(max(hd_t, hd_p))
                            
            mean_dice = np.mean(val_dice_list) if val_dice_list else 0.0
            mean_iou = np.mean(val_iou_list) if val_iou_list else 0.0
            mean_hd = np.mean(val_hd_list) if val_hd_list else 0.0
            
            print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | Val HD: {mean_hd:.2f} px")

        # Save verification plot for a validation sample
        model.eval()
        with torch.no_grad():
            # Get one batch from validation
            val_iter = iter(val_loader)
            images, targets = next(val_iter)
            images, targets = images.to(device), targets.to(device)
            C_s, C_hat = extract_edges_and_distances(targets, images)
            logits = model(images, C_s, C_hat)
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1)
            
            save_verification_plot(
                image=images[0],
                target=targets[0],
                contour_prior=C_s[0],
                pred_prob=probs[0],
                pred_mask=pred_mask[0],
                filepath=args.plot_path
            )

    print("=" * 60)
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
