import os
import re
import time
import subprocess
import numpy as np

# Paths
README_PATH = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet/README.md"
WORKSPACE_DIR = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet"

LOG_FILES = {
    'mri': os.path.join(WORKSPACE_DIR, 'mri_train.log'),
    'v19': os.path.join(WORKSPACE_DIR, 'verse19_train.log'),
    'v20': os.path.join(WORKSPACE_DIR, 'verse20_train.log'),
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

def parse_best_metrics(filepath):
    """
    Parses a log file and returns the metrics of the epoch with the highest Val Dice.
    """
    if not os.path.exists(filepath):
        return None
        
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+Loss:\s*([\d.]+)\s*\|\s*Val\s+Dice:\s*([\d.]+)\s*\|\s*Val\s+IoU:\s*([\d.]+)\s*\|\s*Val\s+HD:\s*([\d.]+)\s*px"
    )
    
    epochs = []
    losses = []
    dices = []
    ious = []
    hds = []
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                dices.append(float(match.group(3)))
                ious.append(float(match.group(4)))
                hds.append(float(match.group(5)))
                
    if not dices:
        return None
        
    best_idx = np.argmax(dices)
    return {
        'epoch': epochs[best_idx],
        'dice': dices[best_idx],
        'iou': ious[best_idx],
        'hd': hds[best_idx]
    }

def update_readme_table(mri_res, v19_res, v20_res, running_status):
    if not os.path.exists(README_PATH):
        print(f"[Error] README.md not found at {README_PATH}")
        return
        
    with open(README_PATH, 'r') as f:
        content = f.read()
        
    # Format rows
    def format_row(name, res, ckpt, status_key):
        is_running = running_status.get(status_key, False)
        if res:
            status_str = "Running" if is_running else "Completed"
            return f"| **{name}** | {res['epoch']} | {res['dice']:.4f} | {res['iou']:.4f} | {res['hd']:.2f} px | `{ckpt}` | {status_str} |"
        else:
            if is_running:
                return f"| **{name}** | 50 | *In Progress* | *TBD* | *TBD* | `{ckpt}` | Running |"
            else:
                # If VerSe 20 is not running yet but MRI was running or just finished
                if name == "VerSe '20 CT" and not running_status.get('v20', False):
                    return f"| **{name}** | 50 | *Queued* | *TBD* | *TBD* | `{ckpt}` | Queued |"
                return f"| **{name}** | 50 | *N/A* | *N/A* | *N/A* | `{ckpt}` | Failed/Aborted |"

    mri_row = format_row("Mendeley Lumbar MRI", mri_res, "best_model_lumbar_mri.pt", 'mri')
    v19_row = format_row("VerSe '19 CT", v19_res, "best_model_verse19.pt", 'v19')
    v20_row = format_row("VerSe '20 CT", v20_res, "best_model_verse20.pt", 'v20')
    
    # Generate the table content
    table_pattern = re.compile(
        r"(### Quantitative Evaluation \(Updating dynamically upon completion\):\s*\n\s*\n\| Dataset \| Epochs \| Best Val Dice \| Val IoU \| Val HD \(px\) \| Checkpoint Path \| Status \|\n\| :--- \| :---: \| :---: \| :---: \| :---: \| :--- \| :---: \|\n)(.*?)(\n\n---)",
        re.DOTALL
    )
    
    new_rows = f"{mri_row}\n{v19_row}\n{v20_row}"
    
    if table_pattern.search(content):
        updated_content = table_pattern.sub(r"\1" + new_rows + r"\3", content)
        with open(README_PATH, 'w') as f:
            f.write(updated_content)
        print("[Info] Table updated in README memory")
    else:
        print("[Error] Could not find matching table pattern in README.md")

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
    mri_res = parse_best_metrics(LOG_FILES['mri'])
    v19_res = parse_best_metrics(LOG_FILES['v19'])
    v20_res = parse_best_metrics(LOG_FILES['v20'])
    
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
