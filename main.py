import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import re

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

def compute_distance_map_pytorch(mask, downsample_res=None):
    """
    Computes Exact Euclidean Distance Transform (EDT) using scipy on CPU.
    This is drastically faster and mathematically exact compared to pairwise GPU distances.
    """
    from scipy.ndimage import distance_transform_edt
    
    B, C, H, W = mask.shape
    device = mask.device
    
    # We convert to numpy, do EDT on CPU, and convert back.
    mask_np = mask.detach().cpu().numpy()
    dist_maps = []
    
    for b in range(B):
        # EDT requires background to be 1 and foreground to be 0 for distance to foreground
        bg_mask = ~(mask_np[b, 0] > 0)
        if not np.any(~bg_mask):
            # No edge pixels found
            max_dist = math.sqrt(H**2 + W**2)
            dist_maps.append(np.full((1, H, W), max_dist, dtype=np.float32))
            continue
            
        dist_map = distance_transform_edt(bg_mask).astype(np.float32)
        dist_maps.append(dist_map[np.newaxis, ...])
        
    dist_maps_tensor = torch.tensor(np.stack(dist_maps, axis=0), device=device)
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

# --- 3D Volumetric Evaluation Helpers ---

# Physical voxel spacing (D, H, W) in mm for each dataset
SPACING_MM = {
    'lumbar_mri': (3.0, 0.586, 0.586),   # (slice_thickness, pixel_y, pixel_x)
    'verse19': (1.0, 1.0, 1.0),           # isotropic 1mm (resampled)
    'verse20': (1.0, 1.0, 1.0),           # isotropic 1mm (resampled)
}

def get_boundary_3d(mask):
    """Extract surface boundary of a 3D binary mask via erosion."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    return mask ^ eroded

def compute_3d_hd95(mask_pred, mask_true, spacing=(1.0, 1.0, 1.0)):
    """Compute 3D 95th-percentile Hausdorff Distance in physical mm."""
    from scipy.ndimage import distance_transform_edt
    if np.sum(mask_pred) == 0 or np.sum(mask_true) == 0:
        return np.nan
    b_pred = get_boundary_3d(mask_pred)
    b_true = get_boundary_3d(mask_true)
    if np.sum(b_pred) == 0 or np.sum(b_true) == 0:
        return np.nan
    dt_true = distance_transform_edt(~b_true, sampling=spacing)
    dt_pred = distance_transform_edt(~b_pred, sampling=spacing)
    hd95_to_true = np.percentile(dt_true[b_pred], 95)
    hd95_to_pred = np.percentile(dt_pred[b_true], 95)
    return max(hd95_to_true, hd95_to_pred)

def get_slice_sort_key(fname, dataset_name):
    """Extract slice index from filename for correct 3D volume ordering."""
    if dataset_name == 'lumbar_mri':
        match = re.search(r'_D(\d+)\.png', fname)
    else:
        match = re.search(r'_slice(\d+)\.png', fname)
    return int(match.group(1)) if match else 0

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
    parser.add_argument("--base_channels", type=int, default=32, help="Number of base channels for U-ResNet (default: 32)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save the best model weights checkpoint (e.g. best_model.pt)")
    parser.add_argument("--min_epochs", type=int, default=20, help="Minimum epochs before early stopping can trigger (default: 20)")
    args = parser.parse_args()

    device = verify_gpu()
    
    # 1. Initialize U-ResNet + Shape-Aware Attention Network
    # in_channels = 1 (grayscale CT/MRI), n_classes = 3 (background, vertebrae, discs) or 2 (background, vertebrae)
    n_classes = 3 if args.dataset in ["lumbar_mri", "simulated"] else 2
    print(f"Initializing UResNet_Attention model with {n_classes} classes...")
    model = UResNet_Attention(in_channels=1, n_classes=n_classes, base_channels=args.base_channels)
    model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Target Parameter Count: ~14.5M (achieved when base_channels=42)")
    
    # 2. Initialize SpineLoss
    criterion = SpineLoss(gamma=0.5, lambda_density=1.5, lambda_boundary=1.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

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
        
        train_loader, val_loader = get_dataloaders(args.dataset, args.data_dir, batch_size=args.batch_size, val_return_filename=True)
        
        best_val_dice = -1.0
        best_val_hd = float('inf')
        epochs_no_improve = 0
        patience = 5
        cosine_scheduler = None
        start_epoch = 1
        start_step = 0
        
        temp_checkpoint_path = None
        if args.checkpoint_path:
            dir_name, file_name = os.path.split(args.checkpoint_path)
            if file_name.startswith("best_model_"):
                new_file_name = file_name.replace("best_model_", "checkpoint_")
            else:
                new_file_name = "checkpoint_" + file_name
            temp_checkpoint_path = os.path.join(dir_name, new_file_name)

        load_path = None
        if temp_checkpoint_path and os.path.exists(temp_checkpoint_path):
            load_path = temp_checkpoint_path
        elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
            load_path = args.checkpoint_path

        if load_path:
            try:
                print(f"[Checkpoint] Loading checkpoint from {load_path} to resume training...")
                checkpoint = torch.load(load_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                saved_step = checkpoint.get('step', -1)
                if saved_step == -1:
                    start_epoch = checkpoint['epoch'] + 1
                    start_step = 0
                else:
                    start_epoch = checkpoint['epoch']
                    start_step = saved_step + 1
                
                best_val_dice = checkpoint.get('val_dice', -1.0)
                best_val_hd = checkpoint.get('val_hd', float('inf'))
                print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch:02d}, step {start_step} with best Val Dice: {best_val_dice:.4f}, best Val HD: {best_val_hd:.2f}")
                
                if start_epoch >= 21:
                    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, args.epochs - 20),
                        eta_min=1e-6
                    )
                    # Step the scheduler to catch up to the resumed epoch
                    for _ in range(21, start_epoch):
                        cosine_scheduler.step()
                    print(f"[LR Scheduler] Re-initialized Cosine Annealing scheduler and caught up to epoch {start_epoch:02d}")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint: {e}. Starting training from scratch.")
                best_val_dice = -1.0
                start_epoch = 1
                start_step = 0
        
        for epoch in range(start_epoch, args.epochs + 1):
            # Paper's learning rate schedule:
            # - Reduced by 10% every 10 epochs before switching to cosine annealing after epoch 20.
            # - Switch to CosineAnnealingLR for remaining epochs.
            if epoch <= 20:
                if epoch == 11:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr * 0.9
                    print(f"\n[LR Scheduler] Reduced learning rate by 10% to {args.lr * 0.9:.6f} at epoch {epoch}")
            elif epoch == 21:
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - 20),
                    eta_min=1e-6
                )
                print(f"\n[LR Scheduler] Switched to Cosine Annealing scheduler for remaining epochs")
                
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n--- Epoch {epoch:02d}/{args.epochs} | Learning Rate: {current_lr:.6f} ---")
            
            model.train()
            train_loss = 0.0
            train_reg = 0.0
            train_bound = 0.0
            train_vol = 0.0
            
            num_batches = 0
            for step_idx, (images, targets) in enumerate(train_loader):
                if step_idx < start_step:
                    continue
                if args.max_steps is not None and step_idx >= args.max_steps:
                    break
                images, targets = images.to(device), targets.to(device)
                
                C_s, C_hat = extract_edges_and_distances(targets, images)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(images, C_s, C_hat)
                    loss, loss_reg, loss_bound, loss_vol = criterion(logits, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_reg += loss_reg.item()
                train_bound += loss_bound.item()
                train_vol += loss_vol.item()
                num_batches += 1
                
                # Save intermediate checkpoint every 500 steps to protect against server restarts
                if (step_idx + 1) % 500 == 0:
                    if temp_checkpoint_path:
                        torch.save({
                            'epoch': epoch,
                            'step': step_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_dice': best_val_dice
                        }, temp_checkpoint_path)
                        print(f"[Checkpoint] Saved intermediate checkpoint at Epoch {epoch:02d}, Step {step_idx+1}")
                
            if num_batches > 0:
                train_loss /= num_batches
                train_reg /= num_batches
                train_bound /= num_batches
                train_vol /= num_batches
            
            # Validation
            model.eval()
            val_dice_list = []
            val_iou_list = []
            # Accumulate per-slice predictions for 3D volume reconstruction
            patient_slices = {}  # pid -> [(slice_idx, pred_np, true_np)]
            
            with torch.no_grad():
                for val_step_idx, batch_data in enumerate(val_loader):
                    if args.max_steps is not None and val_step_idx >= max(1, args.max_steps // 4):
                        break
                    val_images, val_targets, val_fnames = batch_data
                    val_images, val_targets = val_images.to(device), val_targets.to(device)
                    val_C_s, val_C_hat = extract_edges_and_distances(val_targets, val_images)
                    
                    val_logits = model(val_images, val_C_s, val_C_hat)
                    val_probs = F.softmax(val_logits, dim=1)
                    val_preds = torch.argmax(val_probs, dim=1)
                    
                    # Compute Dice and IoU (unchanged — per-slice is fine for DSC)
                    for c in [1, 2]:
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
                    
                    # Accumulate predictions per patient for 3D HD
                    val_preds_np = val_preds.cpu().numpy().astype(np.uint8)
                    val_targets_np = val_targets.cpu().numpy().astype(np.uint8)
                    for b in range(val_images.shape[0]):
                        fname = val_fnames[b]
                        pid = fname.split("_")[0]
                        slice_idx = get_slice_sort_key(fname, args.dataset)
                        if pid not in patient_slices:
                            patient_slices[pid] = []
                        patient_slices[pid].append((slice_idx, val_preds_np[b], val_targets_np[b]))
            
            # Compute 3D volumetric 95% HD per patient (subsample for speed)
            spacing = SPACING_MM.get(args.dataset, (1.0, 1.0, 1.0))
            all_pids = sorted(patient_slices.keys())
            max_patients_for_hd = 10
            if len(all_pids) > max_patients_for_hd:
                np.random.seed(epoch)  # deterministic per epoch but varies across epochs
                selected_pids = list(np.random.choice(all_pids, max_patients_for_hd, replace=False))
            else:
                selected_pids = all_pids
            
            patient_hd_list = []
            for pid in selected_pids:
                slices = sorted(patient_slices[pid], key=lambda x: x[0])
                pred_vol = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)
                true_vol = np.stack([s[2] for s in slices], axis=0)
                hd = compute_3d_hd95(pred_vol > 0, true_vol > 0, spacing=spacing)
                if not np.isnan(hd):
                    patient_hd_list.append(hd)
            
            mean_dice = np.mean(val_dice_list) if val_dice_list else 0.0
            mean_iou = np.mean(val_iou_list) if val_iou_list else 0.0
            mean_hd = np.mean(patient_hd_list) if patient_hd_list else 0.0
            
            print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | Val 3D-HD95: {mean_hd:.2f} mm ({len(selected_pids)} patients)")
            
            # Step the cosine scheduler if active
            if cosine_scheduler is not None:
                cosine_scheduler.step()
                
            # Checkpoint & Early Stopping (Dual-Metric: DSC + 95% HD)
            dice_improved = mean_dice > best_val_dice
            hd_improved = mean_hd < best_val_hd and mean_hd > 0
            either_improved = dice_improved or hd_improved
            
            if dice_improved:
                best_val_dice = mean_dice
            if hd_improved:
                best_val_hd = mean_hd
            
            if either_improved:
                epochs_no_improve = 0
                improve_reasons = []
                if dice_improved:
                    improve_reasons.append(f"DSC: {best_val_dice:.4f}")
                if hd_improved:
                    improve_reasons.append(f"HD: {best_val_hd:.2f}")
                print(f"[Checkpoint] Metric improved ({', '.join(improve_reasons)})")
                
                if args.checkpoint_path:
                    ckpt_dict = {
                        'epoch': epoch,
                        'step': -1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': best_val_dice,
                        'val_iou': mean_iou,
                        'val_hd': best_val_hd
                    }
                    torch.save(ckpt_dict, args.checkpoint_path)
                    if temp_checkpoint_path:
                        torch.save(ckpt_dict, temp_checkpoint_path)
                    print(f"[Checkpoint] Saved best model to {args.checkpoint_path} (Val Dice: {best_val_dice:.4f}, Val HD: {best_val_hd:.2f})")
            else:
                if epoch >= args.min_epochs:
                    epochs_no_improve += 1
                    print(f"[Early Stopping] Neither DSC nor HD improved. Count: {epochs_no_improve}/{patience} (epoch {epoch}/{args.epochs})")
                    if epochs_no_improve >= patience:
                        print(f"[Early Stopping] No improvement in either metric for {patience} consecutive epochs after min_epochs={args.min_epochs}. Stopping.")
                        break
                else:
                    print(f"[Info] No improvement, but epoch {epoch} < min_epochs={args.min_epochs}. Continuing.")
            
            # Reset start_step for the next epoch
            start_step = 0

        # Save verification plot for a validation sample
        model.eval()
        with torch.no_grad():
            # Get one batch from validation
            val_iter = iter(val_loader)
            images, targets, _ = next(val_iter)
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
