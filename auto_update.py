import os
import re
import time
import subprocess
import numpy as np
import torch


# Paths
README_PATH = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet/README.md"
WORKSPACE_DIR = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet"

LOG_FILES = {
    'mri': os.path.join(WORKSPACE_DIR, 'mri_train_v7.log'),
    'v19': os.path.join(WORKSPACE_DIR, 'verse19_train_v7.log'),
    'v20': os.path.join(WORKSPACE_DIR, 'verse20_train_v7.log'),
}

def is_dataset_running(dataset_name):
    try:
        res = subprocess.run(["pgrep", "-f", f"main.py --dataset {dataset_name}"], capture_output=True, text=True)
        return len(res.stdout.strip()) > 0
    except Exception:
        return False

def get_running_status():
    return {
        'mri': is_dataset_running('lumbar_mri'),
        'v19': is_dataset_running('verse19'),
        'v20': is_dataset_running('verse20')
    }

def parse_best_metrics(filepath, ckpt_path=None):
    """
    Parses a log file and optionally a checkpoint file, returning the metrics
    of the epoch with the highest Val Dice.
    """
    best_ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'epoch' in checkpoint and 'val_dice' in checkpoint:
                best_ckpt = {
                    'epoch': checkpoint['epoch'],
                    'dice': checkpoint['val_dice'],
                    'iou': checkpoint.get('val_iou', 0.0),
                    'hd': checkpoint.get('val_hd', 0.0)
                }
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {ckpt_path} in auto-updater: {e}")

    best_log = None
    if os.path.exists(filepath):
        pattern = re.compile(
            r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+Loss:\s*([\d.]+)\s*\|\s*Val\s+Dice:\s*([\d.]+)\s*\|\s*Val\s+IoU:\s*([\d.]+)\s*\|\s*Val\s+3D-HD95:\s*([\d.]+)\s*mm"
        )
        
        epochs = []
        dices = []
        ious = []
        hds = []
        
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    dices.append(float(match.group(3)))
                    ious.append(float(match.group(4)))
                    hds.append(float(match.group(5)))
                    
        if dices:
            best_idx = np.argmax(dices)
            best_log = {
                'epoch': epochs[best_idx],
                'dice': dices[best_idx],
                'iou': ious[best_idx],
                'hd': hds[best_idx]
            }

    # Prioritize the checkpoint metrics if available, as they represent the actual saved weights.
    res = None
    if best_ckpt:
        res = best_ckpt
    elif best_log:
        res = best_log

    return res


def update_readme_table(mri_res, v19_res, v20_res, running_status):
    if not os.path.exists(README_PATH):
        print(f"[Error] README.md not found at {README_PATH}")
        return
        
    with open(README_PATH, 'r') as f:
        content = f.read()
        
    # Format rows
    def format_row(name, res, status_key):
        is_running = running_status.get(status_key, False)
        is_real_res = res and res.get('dice', -1.0) > 0.0
        
        config_str = "Run 2 (`ch=42`, 14.5M)"
        
        def format_hd(hd_val):
            if hd_val == float('inf') or np.isinf(hd_val) or hd_val is None:
                return "inf"
            return f"{hd_val:.2f} mm"
            
        if is_real_res:
            status_str = "🔄 Training" if is_running else "✅ Completed"
            epoch_str = f"{res['epoch']}/50" if is_running else f"{res['epoch']}"
            return f"| **{name}** | {config_str} | {epoch_str} | {res['dice']:.4f} | {res['iou']:.4f} | {format_hd(res['hd'])} | {status_str} |"
        else:
            if is_running:
                return f"| **{name}** | {config_str} | 1/50 | *In Progress* | *TBD* | *TBD* | 🔄 Training |"
            else:
                if name == "VerSe '20 CT" and running_status.get('v19', False):
                    return f"| **{name}** | {config_str} | 50 | *Queued* | *TBD* | *TBD* | Queued |"
                return f"| **{name}** | {config_str} | 50 | *N/A* | *N/A* | *N/A* | Failed/Aborted |"

    mri_row = format_row("Mendeley Lumbar MRI", mri_res, 'mri')
    v19_row = format_row("VerSe '19 CT", v19_res, 'v19')
    v20_row = format_row("VerSe '20 CT", v20_res, 'v20')
    
    # Generate the table content
    table_pattern = re.compile(
        r"(#### Run 2 — `base_channels=42` \(14\.5M parameters\), 3D patient-level HD95 in mm\s*\n\s*\n\| Dataset \| Config \| Epochs \| Best Val Dice \| Val IoU \| Best 3D-HD95 \| Status \|\n\| :--- \| :--- \| :---: \| :---: \| :---: \| :---: \| :---: \|\n)(.*?)(\n\n---)",
        re.DOTALL
    )
    
    new_rows = f"{mri_row}\n{v19_row}\n{v20_row}"
    
    if table_pattern.search(content):
        content = table_pattern.sub(r"\1" + new_rows + r"\3", content)
        print("[Info] Table updated in README memory")
    else:
        print("[Error] Could not find matching table pattern in README.md")

    # Update VerSe Comparison Table rows if they exist
    v19_comp_pattern = re.compile(
        r"(\| \*\*Run 2 \(V19\)\*\* \(`ch=42`, 14\.5M\) \| Ours \(U-ResNet \+ SAAM\) \| VerSe '19 \| Vertebrae \(Combined\) \| )(.*?)(\n)"
    )
    v20_comp_pattern = re.compile(
        r"(\| \*\*Run 2 \(V20\)\*\* \(`ch=42`, 14\.5M\) \| Ours \(U-ResNet \+ SAAM\) \| VerSe '20 \| Vertebrae \(Combined\) \| )(.*?)(\n)"
    )
    
    def format_hd(hd_val):
        if hd_val == float('inf') or np.isinf(hd_val) or hd_val is None:
            return "inf"
        return f"{hd_val:.2f}"

    if v19_res and v19_res.get('dice', -1.0) > 0.0:
        v19_status = ", 🔄 training" if running_status.get('v19', False) else ""
        v19_comp_row = f"**{v19_res['dice']:.4f}** (Epoch {v19_res['epoch']}{v19_status}) | **{format_hd(v19_res['hd'])} mm** (3D-HD95) |"
        content = v19_comp_pattern.sub(r"\1" + v19_comp_row + r"\3", content)

    if v20_res and v20_res.get('dice', -1.0) > 0.0:
        v20_status = ", 🔄 training" if running_status.get('v20', False) else ""
        v20_comp_row = f"**{v20_res['dice']:.4f}** (Epoch {v20_res['epoch']}{v20_status}) | **{format_hd(v20_res['hd'])} mm** (3D-HD95) |"
        content = v20_comp_pattern.sub(r"\1" + v20_comp_row + r"\3", content)

    with open(README_PATH, 'w') as f:
        f.write(content)

def has_changes_to_commit():
    target_files = [
        "README.md",
        "mendeley_lumbar_mri_curves.png",
        "verse_19_ct_curves.png",
        "verse_20_ct_curves.png",
        "dataset_comparison_chart.png",
        "verification_plot_lumbar_mri.png",
        "verification_plot_verse19.png",
        "verification_plot_verse20.png"
    ]
    # Check if any of these files are modified or untracked
    res = subprocess.run(
        ["git", "status", "--porcelain"] + target_files,
        cwd=WORKSPACE_DIR,
        capture_output=True,
        text=True
    )
    return len(res.stdout.strip()) > 0

def run_git_push():
    if not has_changes_to_commit():
        print("[Info] No metric or README changes detected. Skipping git push.")
        return
        
    print("[Info] Staging, committing and pushing results...")
    try:
        target_files = [
            "README.md",
            "mendeley_lumbar_mri_curves.png",
            "verse_19_ct_curves.png",
            "verse_20_ct_curves.png",
            "dataset_comparison_chart.png",
            "verification_plot_lumbar_mri.png",
            "verification_plot_verse19.png",
            "verification_plot_verse20.png"
        ]
        # Only add files that exist
        existing_files = [f for f in target_files if os.path.exists(os.path.join(WORKSPACE_DIR, f))]
        
        subprocess.run(["git", "add"] + existing_files, cwd=WORKSPACE_DIR, check=True)
        commit_msg = "Live-update: training metrics and plots updated"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=WORKSPACE_DIR, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=WORKSPACE_DIR, check=True)
        print("[Info] Successfully pushed updates to origin main.")
    except Exception as e:
        print(f"[Error] Git push failed: {e}")

def perform_iteration():
    # 1. Run plot_metrics.py
    print("[Info] Generating learning curves and comparison chart...")
    try:
        subprocess.run(["uv", "run", "python", "plot_metrics.py"], cwd=WORKSPACE_DIR, check=True)
    except Exception as e:
        print(f"[Error] Failed to run plot_metrics.py: {e}")
        
    # 2. Parse log files
    print("[Info] Parsing log files...")
    mri_res = parse_best_metrics(LOG_FILES['mri'], os.path.join(WORKSPACE_DIR, 'best_model_lumbar_mri.pt'))
    v19_res = parse_best_metrics(LOG_FILES['v19'], os.path.join(WORKSPACE_DIR, 'best_model_verse19.pt'))
    v20_res = parse_best_metrics(LOG_FILES['v20'], os.path.join(WORKSPACE_DIR, 'best_model_verse20.pt'))
    
    # 3. Get running statuses
    running_status = get_running_status()
    
    # 4. Update README.md
    update_readme_table(mri_res, v19_res, v20_res, running_status)
    
    # 5. Push updates to Github
    run_git_push()

def main():
    print(f"[Info] Starting auto-updater loop, monitoring main.py processes dynamically...")
    
    while True:
        # Run update step
        perform_iteration()
        
        # Check running state
        mri_active = is_dataset_running('lumbar_mri')
        v19_active = is_dataset_running('verse19')
        v20_active = is_dataset_running('verse20')
        
        if not mri_active and not v19_active and not v20_active:
            print("[Monitor] All processes finished. Running final update...")
            time.sleep(5)
            perform_iteration()
            break
            
        print("[Monitor] Processes still running. Sleeping for 300 seconds...")
        time.sleep(300)
        
    print("[Info] Auto-update flow finished successfully!")

if __name__ == "__main__":
    main()
