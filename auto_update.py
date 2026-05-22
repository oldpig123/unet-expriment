import os
import re
import time
import subprocess
import numpy as np

# Monitored PIDs
# 4014932: VerSe '19 training process (GPU 1)
# 4015137: VerSe '20 sequential bash script runner (GPU 0)
MONITORED_PIDS = [4014932, 4015137]

# Paths
LOG_DIR = "/home/nmlab326/.gemini/antigravity-ide/brain/a8113a7b-dd8a-4176-ba18-eb043fe77afb/.system_generated/tasks"
README_PATH = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet/README.md"
WORKSPACE_DIR = "/media/nmlab326/b2cd0f5f-2bd7-46c8-8a50-58708471c1bf1/experiments/unet"

LOG_FILES = {
    'mri': os.path.join(LOG_DIR, 'task-3184.log'),
    'v19': os.path.join(LOG_DIR, 'task-3188.log'),
    'v20': os.path.join(LOG_DIR, 'task-3194.log'),
}

def is_any_pid_running(pids):
    for pid in pids:
        try:
            # os.kill(pid, 0) checks if process exists without sending a signal
            os.kill(pid, 0)
            return True
        except OSError:
            pass
    return False

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

def update_readme_table(mri_res, v19_res, v20_res):
    if not os.path.exists(README_PATH):
        print(f"[Error] README.md not found at {README_PATH}")
        return
        
    with open(README_PATH, 'r') as f:
        content = f.read()
        
    # Format rows
    def format_row(name, res, ckpt):
        if res:
            return f"| **{name}** | {res['epoch']} | {res['dice']:.4f} | {res['iou']:.4f} | {res['hd']:.2f} px | `{ckpt}` | Completed |"
        else:
            return f"| **{name}** | 50 | *N/A* | *N/A* | *N/A* | `{ckpt}` | Failed/Aborted |"

    mri_row = format_row("Mendeley Lumbar MRI", mri_res, "best_model_lumbar_mri.pt")
    v19_row = format_row("VerSe '19 CT", v19_res, "best_model_verse19.pt")
    v20_row = format_row("VerSe '20 CT", v20_res, "best_model_verse20.pt")
    
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
        print("[Info] Successfully updated Quantitative Evaluation table in README.md")
    else:
        print("[Error] Could not find matching table pattern in README.md")

def run_git_push():
    print("[Info] Staging, committing and pushing results...")
    try:
        # Run git operations
        subprocess.run(["git", "add", "README.md", "*.png"], cwd=WORKSPACE_DIR, check=True)
        commit_msg = "Auto-update: training complete, plots generated and README performance table updated"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=WORKSPACE_DIR, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=WORKSPACE_DIR, check=True)
        print("[Info] Successfully pushed updates to origin main.")
    except Exception as e:
        print(f"[Error] Git push failed: {e}")

def main():
    print(f"[Info] Starting auto-updater, monitoring PIDs: {MONITORED_PIDS}")
    
    # Wait for completion
    while is_any_pid_running(MONITORED_PIDS):
        print(f"[Monitor] Process(es) still running. Sleeping for 60 seconds...")
        time.sleep(60)
        
    print("[Monitor] All processes finished! Waiting 5 seconds for filesystem logs to flush...")
    time.sleep(5)
    
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
    
    print(f"Mendeley MRI: {mri_res}")
    print(f"VerSe 19: {v19_res}")
    print(f"VerSe 20: {v20_res}")
    
    # 3. Update README.md
    update_readme_table(mri_res, v19_res, v20_res)
    
    # 4. Push updates to Github
    run_git_push()
    
    print("[Info] Auto-update flow finished successfully!")

if __name__ == "__main__":
    main()
