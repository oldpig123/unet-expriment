import os
import re
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion

from model import UResNet_Attention
from main import extract_edges_and_distances

WORKSPACE_DIR = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet"
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")

def get_boundary_3d(mask):
    eroded = binary_erosion(mask)
    return mask ^ eroded

def compute_3d_hd95(mask_pred, mask_true, spacing=(1.0, 1.0, 1.0)):
    if np.sum(mask_pred) == 0 or np.sum(mask_true) == 0:
        return np.nan
    b_pred = get_boundary_3d(mask_pred)
    b_true = get_boundary_3d(mask_true)
    if np.sum(b_pred) == 0 or np.sum(b_true) == 0:
        return np.nan
    dt_true = distance_transform_edt(~b_true, sampling=spacing)
    dt_pred = distance_transform_edt(~b_pred, sampling=spacing)
    distances_to_true = dt_true[b_pred]
    distances_to_pred = dt_pred[b_true]
    return max(np.percentile(distances_to_true, 95), np.percentile(distances_to_pred, 95))

def evaluate_verse(device, checkpoint_path, data_dir, dataset_name, base_channels):
    print(f"\n[Fixer] Evaluating {dataset_name} using checkpoint {checkpoint_path}...")
    dataset_path = os.path.join(data_dir, dataset_name)
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(images_dir):
        print(f"[Error] Images directory not found: {images_dir}")
        return None
        
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    patient_ids = sorted(list(set([f.split("_")[0] for f in all_files])))
    
    np.random.seed(42)
    indices = np.random.permutation(len(patient_ids))
    split_idx = int(len(patient_ids) * 0.8)
    val_pids = set([patient_ids[i] for i in indices[split_idx:]])
    
    val_files = [f for f in all_files if f.split("_")[0] in val_pids]
    
    patient_slices = {}
    for f in val_files:
        pid = f.split("_")[0]
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append(f)
        
    model = UResNet_Attention(in_channels=1, n_classes=2, base_channels=base_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_dices = []
    all_ious = []
    patient_hds = []
    
    spacing = (1.0, 1.0, 1.0)
    sorted_pids = sorted(patient_slices.keys())
    
    with torch.no_grad():
        for pid in sorted_pids:
            slices = patient_slices[pid]
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
            
            # Slice-level Dice and IoU
            for preds, lbl_np in zip(pred_vol_list, true_vol_list):
                pred_c = (preds == 1)
                target_c = (lbl_np == 1)
                intersection = np.sum(pred_c & target_c)
                union = np.sum(pred_c | target_c)
                sum_total = np.sum(pred_c) + np.sum(target_c)
                if np.sum(target_c) > 0:
                    dice = (2.0 * intersection) / (sum_total + 1e-8)
                    iou = intersection / (union + 1e-8)
                    all_dices.append(dice)
                    all_ious.append(iou)
                    
            # 3D-HD95
            hd = compute_3d_hd95(pred_vol == 1, true_vol == 1, spacing=spacing)
            if not np.isnan(hd):
                patient_hds.append(hd)
                
    mean_dice = np.mean(all_dices) if all_dices else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_hd = np.mean(patient_hds) if patient_hds else 0.0
    print(f"[{dataset_name}] True Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | 3D-HD95: {mean_hd:.2f} mm")
    return mean_dice, mean_iou, mean_hd

def evaluate_mri(device, checkpoint_path, data_dir, base_channels):
    print(f"\n[Fixer] Evaluating lumbar_mri using checkpoint {checkpoint_path}...")
    images_dir = os.path.join(data_dir, "lumbar_mri", "images")
    labels_dir = os.path.join(data_dir, "lumbar_mri", "labels")
    
    if not os.path.exists(images_dir):
        print(f"[Error] Images directory not found: {images_dir}")
        return None
        
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    patient_ids = sorted(list(set([f.split("_")[0] for f in all_files])))
    
    np.random.seed(42)
    indices = np.random.permutation(len(patient_ids))
    split_idx = int(len(patient_ids) * 0.8)
    val_pids = set([patient_ids[i] for i in indices[split_idx:]])
    
    val_files = [f for f in all_files if f.split("_")[0] in val_pids]
    
    patient_slices = {}
    for f in val_files:
        pid = f.split("_")[0]
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append(f)
        
    model = UResNet_Attention(in_channels=1, n_classes=3, base_channels=base_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_dices = []
    all_ious = []
    patient_hds = []
    
    spacing = (3.0, 0.586, 0.586)
    sorted_pids = sorted(patient_slices.keys())
    
    with torch.no_grad():
        for pid in sorted_pids:
            slices = patient_slices[pid]
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
            
            # Slice-level Dice and IoU (across classes 1 and 2)
            for preds, lbl_np in zip(pred_vol_list, true_vol_list):
                for c in [1, 2]:
                    pred_c = (preds == c)
                    target_c = (lbl_np == c)
                    intersection = np.sum(pred_c & target_c)
                    union = np.sum(pred_c | target_c)
                    sum_total = np.sum(pred_c) + np.sum(target_c)
                    if np.sum(target_c) > 0:
                        dice = (2.0 * intersection) / (sum_total + 1e-8)
                        iou = intersection / (union + 1e-8)
                        all_dices.append(dice)
                        all_ious.append(iou)
                        
            # 3D-HD95
            hd = compute_3d_hd95(pred_vol > 0, true_vol > 0, spacing=spacing)
            if not np.isnan(hd):
                patient_hds.append(hd)
                
    mean_dice = np.mean(all_dices) if all_dices else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_hd = np.mean(patient_hds) if patient_hds else 0.0
    print(f"[lumbar_mri] True Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | 3D-HD95: {mean_hd:.2f} mm")
    return mean_dice, mean_iou, mean_hd

def is_main_running():
    try:
        res = subprocess.run(["pgrep", "-f", "main.py"], capture_output=True, text=True)
        # Exclude our own script if it matches (it won't since we match main.py specifically)
        pids = [p for p in res.stdout.strip().split("\n") if p]
        return len(pids) > 0
    except Exception:
        return False

def update_pt_metadata(filepath, dice, iou, hd):
    if not os.path.exists(filepath):
        print(f"[Warning] Checkpoint not found to update metadata: {filepath}")
        return
    try:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        checkpoint['val_dice'] = dice
        checkpoint['val_iou'] = iou
        checkpoint['val_hd'] = hd
        torch.save(checkpoint, filepath)
        print(f"[Fixer] Successfully updated metadata in {filepath} to: Dice={dice:.4f}, IoU={iou:.4f}, HD={hd:.2f}")
    except Exception as e:
        print(f"[Error] Failed to update checkpoint {filepath}: {e}")

def main():
    print("[Fixer] Starting background training monitor and re-evaluation daemon...")
    
    # 1. Wait for training runs to complete
    while True:
        if not is_main_running():
            print("[Fixer] No active main.py training processes found. Starting re-evaluation...")
            break
        print("[Fixer] Training processes still running. Sleeping for 300 seconds...")
        time.sleep(300)
        
    # Wait a little bit for files to finish flushing to disk
    time.sleep(10)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Fixer] Using device: {device}")
    
    # 2. Re-evaluate lumbar_mri
    mri_chk = os.path.join(WORKSPACE_DIR, "best_model_lumbar_mri.pt")
    if os.path.exists(mri_chk):
        mri_metrics = evaluate_mri(device, mri_chk, DATA_DIR, base_channels=42)
        if mri_metrics:
            update_pt_metadata(mri_chk, *mri_metrics)
            
    # 3. Re-evaluate verse19
    v19_chk = os.path.join(WORKSPACE_DIR, "best_model_verse19.pt")
    if os.path.exists(v19_chk):
        v19_metrics = evaluate_verse(device, v19_chk, DATA_DIR, "verse19", base_channels=42)
        if v19_metrics:
            update_pt_metadata(v19_chk, *v19_metrics)
            
    # 4. Re-evaluate verse20
    v20_chk = os.path.join(WORKSPACE_DIR, "best_model_verse20.pt")
    if os.path.exists(v20_chk):
        v20_metrics = evaluate_verse(device, v20_chk, DATA_DIR, "verse20", base_channels=42)
        if v20_metrics:
            update_pt_metadata(v20_chk, *v20_metrics)
            
    # 5. Run auto-update manually to rebuild plots, table, and push to GitHub
    print("[Fixer] Running manual auto_update iteration to sync corrected metrics to README/GitHub...")
    try:
        subprocess.run(["uv", "run", "python", "-c", "import auto_update; auto_update.perform_iteration()"], cwd=WORKSPACE_DIR, check=True)
        print("[Fixer] Checkpoints successfully fixed and README updated/pushed!")
    except Exception as e:
        print(f"[Error] Failed to run manual auto-update: {e}")

if __name__ == "__main__":
    main()
