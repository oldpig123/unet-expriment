import os
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion

from model import UResNet_Attention
from main import extract_edges_and_distances

def get_boundary_3d(mask):
    """
    Extracts the boundary voxels of a binary 3D mask using binary erosion.
    """
    eroded = binary_erosion(mask)
    return mask ^ eroded

def compute_3d_hd95(mask_pred, mask_true, spacing=(1.0, 1.0, 1.0)):
    """
    Computes the 95th percentile 3D Hausdorff Distance in millimeters.
    """
    if np.sum(mask_pred) == 0 or np.sum(mask_true) == 0:
        return np.nan
        
    b_pred = get_boundary_3d(mask_pred)
    b_true = get_boundary_3d(mask_true)
    
    if np.sum(b_pred) == 0 or np.sum(b_true) == 0:
        return np.nan
        
    # Distance transforms in physical spacing (mm)
    dt_true = distance_transform_edt(~b_true, sampling=spacing)
    dt_pred = distance_transform_edt(~b_pred, sampling=spacing)
    
    distances_to_true = dt_true[b_pred]
    distances_to_pred = dt_pred[b_true]
    
    hd95_to_true = np.percentile(distances_to_true, 95)
    hd95_to_pred = np.percentile(distances_to_pred, 95)
    
    return max(hd95_to_true, hd95_to_pred)

def compute_3d_dice(mask_pred, mask_true):
    """
    Computes the 3D Dice Similarity Coefficient.
    """
    intersection = np.sum(mask_pred & mask_true)
    total = np.sum(mask_pred) + np.sum(mask_true)
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total

def evaluate_mri_3d(device, checkpoint_path, data_dir, base_channels):
    print("\n" + "="*60)
    print(f"3D VOLUMETRIC EVALUATION: Mendeley Lumbar Spine MRI")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60)
    
    images_dir = os.path.join(data_dir, "lumbar_mri", "images")
    labels_dir = os.path.join(data_dir, "lumbar_mri", "labels")
    
    if not os.path.exists(images_dir):
        print(f"[Error] Images directory not found: {images_dir}")
        return
        
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    patient_ids = sorted(list(set([f.split("_")[0] for f in all_files])))
    
    # 80/20 train/validation split by patient ID
    np.random.seed(42)
    indices = np.random.permutation(len(patient_ids))
    split_idx = int(len(patient_ids) * 0.8)
    val_pids = set([patient_ids[i] for i in indices[split_idx:]])
    
    val_files = [f for f in all_files if f.split("_")[0] in val_pids]
    
    # Group slices by patient ID
    patient_slices = {}
    for f in val_files:
        pid = f.split("_")[0]
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append(f)
        
    # MRI is a 3-class segmentation problem (Background, Vertebrae, Discs)
    model = UResNet_Attention(in_channels=1, n_classes=3, base_channels=base_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    patient_dice_c1 = []
    patient_dice_c2 = []
    patient_dice_combined = []
    patient_hd95 = []
    
    # MRI Spacing: (slice_thickness = 3.0 mm, pixel_y = 0.586 mm, pixel_x = 0.586 mm)
    spacing = (3.0, 0.586, 0.586)
    
    with torch.no_grad():
        for pid, slices in sorted(patient_slices.items()):
            # Sort slices by their slice position (naming convention: e.g. 0001_D3.png)
            def get_slice_idx(fname):
                match = re.search(r'_D(\d+)\.png', fname)
                return int(match.group(1)) if match else 0
            slices = sorted(slices, key=get_slice_idx)
            
            pred_vol_list = []
            true_vol_list = []
            
            for fname in slices:
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir, fname)
                
                img_pil = Image.open(img_path).convert("L")
                lbl_pil = Image.open(lbl_path)
                
                img_resized = img_pil.resize((512, 512), Image.BILINEAR)
                lbl_resized = lbl_pil.resize((512, 512), Image.NEAREST)
                
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                lbl_np = np.array(lbl_resized, dtype=np.uint8)
                
                mapped_lbl = np.zeros_like(lbl_np)
                mapped_lbl[lbl_np == 100] = 1
                mapped_lbl[lbl_np == 50] = 2
                
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                seg_tensor = torch.tensor(mapped_lbl, dtype=torch.long).unsqueeze(0).to(device)
                
                C_s, C_hat = extract_edges_and_distances(seg_tensor, img_tensor)
                
                logits = model(img_tensor, C_s, C_hat)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                pred_vol_list.append(preds)
                true_vol_list.append(mapped_lbl)
                
            pred_vol = np.stack(pred_vol_list, axis=0) # (D, H, W)
            true_vol = np.stack(true_vol_list, axis=0) # (D, H, W)
            
            # Compute patient-level 3D metrics
            dice_c1 = compute_3d_dice(pred_vol == 1, true_vol == 1)
            dice_c2 = compute_3d_dice(pred_vol == 2, true_vol == 2)
            patient_dice_c1.append(dice_c1)
            patient_dice_c2.append(dice_c2)
            patient_dice_combined.append((dice_c1 + dice_c2) / 2)
            
            hd = compute_3d_hd95(pred_vol > 0, true_vol > 0, spacing=spacing)
            if not np.isnan(hd):
                patient_hd95.append(hd)
                
    print(f"Vertebrae (Class 1) 3D DSC:            {np.mean(patient_dice_c1):.6f}")
    print(f"Intervertebral Disc (Class 2) 3D DSC:  {np.mean(patient_dice_c2):.6f}")
    print(f"Combined Mean 3D DSC:                  {np.mean(patient_dice_combined):.6f}")
    print(f"Validation 95% HD (3D physical mm):    {np.mean(patient_hd95):.6f} mm")
    print("="*60 + "\n")

def evaluate_verse_3d(device, checkpoint_path, data_dir, dataset_name, base_channels):
    print("\n" + "="*60)
    print(f"3D VOLUMETRIC EVALUATION: {dataset_name.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60)
    
    dataset_path = os.path.join(data_dir, dataset_name)
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(images_dir):
        print(f"[Error] Images directory not found: {images_dir}")
        return
        
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    patient_ids = sorted(list(set([f.split("_")[0] for f in all_files])))
    
    # 80/20 train/validation split by patient ID
    np.random.seed(42)
    indices = np.random.permutation(len(patient_ids))
    split_idx = int(len(patient_ids) * 0.8)
    val_pids = set([patient_ids[i] for i in indices[split_idx:]])
    
    val_files = [f for f in all_files if f.split("_")[0] in val_pids]
    
    # Group slices by patient ID
    patient_slices = {}
    for f in val_files:
        pid = f.split("_")[0]
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append(f)
        
    # VerSe is a 2-class segmentation problem (Background, Vertebrae)
    model = UResNet_Attention(in_channels=1, n_classes=2, base_channels=base_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    patient_dice = []
    patient_hd95 = []
    
    # VerSe Spacing: Resampled to isotropic (1.0 mm, 1.0 mm, 1.0 mm)
    spacing = (1.0, 1.0, 1.0)
    
    sorted_pids = sorted(patient_slices.keys())
    
    with torch.no_grad():
        for idx, pid in enumerate(sorted_pids):
            print(f"Evaluating patient {idx+1}/{len(sorted_pids)} (ID: {pid})")
            slices = patient_slices[pid]
            
            # Sort slices by their index: sub-verse004_slice060.png
            def get_slice_idx(fname):
                match = re.search(r'_slice(\d+)\.png', fname)
                return int(match.group(1)) if match else 0
            slices = sorted(slices, key=get_slice_idx)
            
            pred_vol_list = []
            true_vol_list = []
            
            for fname in slices:
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir, fname)
                
                img_pil = Image.open(img_path).convert("L")
                lbl_pil = Image.open(lbl_path)
                
                img_resized = img_pil.resize((512, 512), Image.BILINEAR)
                lbl_resized = lbl_pil.resize((512, 512), Image.NEAREST)
                
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                lbl_np = np.array(lbl_resized, dtype=np.uint8)
                
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                seg_tensor = torch.tensor(lbl_np, dtype=torch.long).unsqueeze(0).to(device)
                
                C_s, C_hat = extract_edges_and_distances(seg_tensor, img_tensor)
                
                logits = model(img_tensor, C_s, C_hat)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                pred_vol_list.append(preds)
                true_vol_list.append(lbl_np)
                
            pred_vol = np.stack(pred_vol_list, axis=0) # (D, H, W)
            true_vol = np.stack(true_vol_list, axis=0) # (D, H, W)
            
            dice = compute_3d_dice(pred_vol == 1, true_vol == 1)
            patient_dice.append(dice)
            
            hd = compute_3d_hd95(pred_vol == 1, true_vol == 1, spacing=spacing)
            if not np.isnan(hd):
                patient_hd95.append(hd)
                
    print(f"Vertebrae (Class 1) 3D DSC:            {np.mean(patient_dice):.6f}")
    print(f"Validation 95% HD (3D physical mm):    {np.mean(patient_hd95):.6f} mm")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Standalone 3D Volumetric Evaluation for U-ResNet + SAAM")
    parser.add_argument("--dataset", type=str, choices=['lumbar_mri', 'verse19', 'verse20', 'all'], default='all',
                        help="Dataset to evaluate (default: all)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to datasets root folder")
    parser.add_argument("--base_channels", type=int, default=42,
                        help="Base channel count matching the checkpoint configuration (default: 42 for Run 2)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Custom checkpoint file path (uses default paths if not provided)")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Evaluate Lumbar Spine MRI
    if args.dataset in ['lumbar_mri', 'all']:
        chk = args.checkpoint if args.checkpoint else "best_model_lumbar_mri.pt"
        if os.path.exists(chk):
            evaluate_mri_3d(device, chk, args.data_dir, args.base_channels)
        else:
            print(f"[Warning] Checkpoint not found for lumbar_mri: {chk}")
            
    # 2. Evaluate VerSe 19
    if args.dataset in ['verse19', 'all']:
        chk = args.checkpoint if (args.checkpoint and args.dataset == 'verse19') else "best_model_verse19.pt"
        if os.path.exists(chk):
            evaluate_verse_3d(device, chk, args.data_dir, "verse19", args.base_channels)
        else:
            print(f"[Warning] Checkpoint not found for verse19: {chk}")
            
    # 3. Evaluate VerSe 20
    if args.dataset in ['verse20', 'all']:
        chk = args.checkpoint if (args.checkpoint and args.dataset == 'verse20') else "best_model_verse20.pt"
        if os.path.exists(chk):
            evaluate_verse_3d(device, chk, args.data_dir, "verse20", args.base_channels)
        else:
            print(f"[Warning] Checkpoint not found for verse20: {chk}")

if __name__ == "__main__":
    main()
