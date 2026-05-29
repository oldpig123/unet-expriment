import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Use a clean, modern style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8f9fa'
})

# --- Colors ---
# Run 1 uses muted/lighter tones, Run 2 uses vivid tones
COLORS_R1 = {'loss': '#e88e93', 'dice': '#6eaafc', 'iou': '#5fbf8a', 'hd': '#a98bd4'}
COLORS_R2 = {'loss': '#dc3545', 'dice': '#0d6efd', 'iou': '#198754', 'hd': '#6f42c1'}


def parse_log_file(filepath, hd_pattern_type='v2'):
    """
    Parses training metrics from a log file.
    Keeps only the latest metrics for each epoch to handle resumes.
    
    hd_pattern_type:
        'v1' — old format: "Val HD: X.XX px"
        'v2' — new format: "Val 3D-HD95: X.XX mm (N patients)"
    """
    epoch_dict = {}
    
    if not os.path.exists(filepath):
        print(f"[Warning] Log file not found: {filepath}")
        return [], [], [], [], []
    
    if hd_pattern_type == 'v1':
        pattern = re.compile(
            r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+Loss:\s*([\d.]+)\s*\|\s*Val\s+Dice:\s*([\d.]+)\s*\|\s*Val\s+IoU:\s*([\d.]+)\s*\|\s*Val\s+HD:\s*([\d.]+)\s*px"
        )
    else:
        pattern = re.compile(
            r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+Loss:\s*([\d.]+)\s*\|\s*Val\s+Dice:\s*([\d.]+)\s*\|\s*Val\s+IoU:\s*([\d.]+)\s*\|\s*Val\s+3D-HD95:\s*([\d.]+)\s*mm"
        )
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                loss = float(match.group(2))
                dice = float(match.group(3))
                iou = float(match.group(4))
                hd = float(match.group(5))
                epoch_dict[ep] = (loss, dice, iou, hd)
                
    epochs = sorted(epoch_dict.keys())
    losses = [epoch_dict[ep][0] for ep in epochs]
    dices = [epoch_dict[ep][1] for ep in epochs]
    ious = [epoch_dict[ep][2] for ep in epochs]
    hds = [epoch_dict[ep][3] for ep in epochs]
    
    return epochs, losses, dices, ious, hds


def plot_overlay_curves(r1_data, r2_data, title, output_path):
    """
    Plots 4-panel learning curves with Run 1 (dashed) and Run 2 (solid) overlaid.
    HD panel uses dual y-axes since Run 1 is in px and Run 2 is in mm.
    
    r1_data / r2_data: tuple of (epochs, losses, dices, ious, hds) or None
    """
    has_r1 = r1_data is not None and len(r1_data[0]) > 0
    has_r2 = r2_data is not None and len(r2_data[0]) > 0
    
    if not has_r1 and not has_r2:
        print(f"[Info] No data to plot for {title}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"U-ResNet + SAAM Learning Curves: {title}", weight='bold', y=0.98)
    
    # --- 1. Training Loss ---
    ax = axes[0, 0]
    if has_r1:
        ax.plot(r1_data[0], r1_data[1], color=COLORS_R1['loss'], linewidth=2,
                linestyle='--', marker='o', markersize=3, alpha=0.8,
                label='Run 1 (ch=32, 8.57M)')
    if has_r2:
        ax.plot(r2_data[0], r2_data[1], color=COLORS_R2['loss'], linewidth=2.5,
                marker='o', markersize=4,
                label='Run 2 (ch=42, 14.5M)')
    ax.set_title('Training Loss Curve', weight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    
    # --- 2. Validation Dice ---
    ax = axes[0, 1]
    if has_r1:
        ax.plot(r1_data[0], r1_data[2], color=COLORS_R1['dice'], linewidth=2,
                linestyle='--', marker='s', markersize=3, alpha=0.8,
                label='Run 1 (ch=32, 8.57M)')
    if has_r2:
        ax.plot(r2_data[0], r2_data[2], color=COLORS_R2['dice'], linewidth=2.5,
                marker='s', markersize=4,
                label='Run 2 (ch=42, 14.5M)')
    ax.set_title('Validation Dice Score', weight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    
    # --- 3. Validation IoU ---
    ax = axes[1, 0]
    if has_r1:
        ax.plot(r1_data[0], r1_data[3], color=COLORS_R1['iou'], linewidth=2,
                linestyle='--', marker='^', markersize=3, alpha=0.8,
                label='Run 1 (ch=32, 8.57M)')
    if has_r2:
        ax.plot(r2_data[0], r2_data[3], color=COLORS_R2['iou'], linewidth=2.5,
                marker='^', markersize=4,
                label='Run 2 (ch=42, 14.5M)')
    ax.set_title('Validation Intersection over Union (IoU)', weight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    
    # --- 4. Validation HD (dual y-axis) ---
    ax = axes[1, 1]
    lines = []
    labels = []
    if has_r1:
        l1, = ax.plot(r1_data[0], r1_data[4], color=COLORS_R1['hd'], linewidth=2,
                       linestyle='--', marker='d', markersize=3, alpha=0.8)
        ax.set_ylabel('HD — Run 1 (2D, pixels)', color=COLORS_R1['hd'], weight='semibold')
        ax.tick_params(axis='y', labelcolor=COLORS_R1['hd'])
        lines.append(l1)
        labels.append('Run 1 HD (2D px)')
    
    if has_r2:
        if has_r1:
            ax2 = ax.twinx()
        else:
            ax2 = ax
        l2, = ax2.plot(r2_data[0], r2_data[4], color=COLORS_R2['hd'], linewidth=2.5,
                        marker='d', markersize=4)
        ax2.set_ylabel('HD — Run 2 (3D-HD95, mm)', color=COLORS_R2['hd'], weight='semibold')
        ax2.tick_params(axis='y', labelcolor=COLORS_R2['hd'])
        lines.append(l2)
        labels.append('Run 2 HD (3D mm)')
    
    ax.set_title('Validation Hausdorff Distance', weight='semibold')
    ax.set_xlabel('Epoch')
    ax.legend(lines, labels, fontsize=9, loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved overlay learning curves to: {output_path}")


def plot_comparison_chart(results, output_path):
    """
    Plots a grouped bar chart comparing Run 1 vs Run 2 performance across datasets.
    """
    datasets = list(results.keys())
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle("Dataset Performance Comparison — Run 1 vs Run 2", weight='bold', y=0.98)
    
    x = np.arange(len(datasets))
    width = 0.18
    
    # Extract values per run
    r1_dices = [results[d].get('r1_dice', 0) for d in datasets]
    r1_ious = [results[d].get('r1_iou', 0) for d in datasets]
    r2_dices = [results[d].get('r2_dice', 0) for d in datasets]
    r2_ious = [results[d].get('r2_iou', 0) for d in datasets]
    
    # Bars
    ax1.bar(x - 1.5*width, r1_dices, width, label='Run 1 Dice', color='#6eaafc', edgecolor='#0d6efd', linewidth=0.8)
    ax1.bar(x - 0.5*width, r1_ious, width, label='Run 1 IoU', color='#5fbf8a', edgecolor='#198754', linewidth=0.8)
    ax1.bar(x + 0.5*width, r2_dices, width, label='Run 2 Dice', color='#0d6efd')
    ax1.bar(x + 1.5*width, r2_ious, width, label='Run 2 IoU', color='#198754')
    
    ax1.set_ylabel('Score (Dice / IoU)', weight='semibold')
    ax1.set_ylim(0, 1.08)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, weight='semibold')
    
    # Annotate each bar
    for bars in [ax1.containers[i] for i in range(4)]:
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax1.annotate(f'{h:.4f}',
                             xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=7.5, weight='bold')
    
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved comparison bar chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse training log files and plot metrics.")
    parser.add_argument("--log_dir", type=str, default=".", help="Tasks logs directory")
    args = parser.parse_args()
    
    # Dataset configs: (display_name, run1_log, run2_log, output_png)
    datasets = [
        ('Mendeley Lumbar MRI',
         os.path.join(args.log_dir, 'mri_train.log'),
         os.path.join(args.log_dir, 'mri_train_v7.log'),
         'mendeley_lumbar_mri_curves.png'),
        ('VerSe 19 CT',
         os.path.join(args.log_dir, 'verse19_train.log'),
         os.path.join(args.log_dir, 'verse19_train_v7.log'),
         'verse_19_ct_curves.png'),
        ('VerSe 20 CT',
         os.path.join(args.log_dir, 'verse20_train.log'),
         os.path.join(args.log_dir, 'verse20_train_v7.log'),
         'verse_20_ct_curves.png'),
    ]
    
    comparison_results = {}
    
    for name, r1_log, r2_log, out_png in datasets:
        print(f"Parsing {name}...")
        
        # Parse Run 1 (old format: Val HD: X.XX px)
        r1 = parse_log_file(r1_log, hd_pattern_type='v1')
        # Parse Run 2 (new format: Val 3D-HD95: X.XX mm)
        r2 = parse_log_file(r2_log, hd_pattern_type='v2')
        
        r1_data = r1 if r1[0] else None
        r2_data = r2 if r2[0] else None
        
        # Plot overlay curves
        plot_overlay_curves(r1_data, r2_data, name, out_png)
        
        # Collect comparison data
        entry = {}
        if r1_data:
            best_idx = np.argmax(r1_data[2])  # best Dice
            entry['r1_dice'] = r1_data[2][best_idx]
            entry['r1_iou'] = r1_data[3][best_idx]
        if r2_data:
            best_idx = np.argmax(r2_data[2])
            entry['r2_dice'] = r2_data[2][best_idx]
            entry['r2_iou'] = r2_data[3][best_idx]
        if entry:
            comparison_results[name] = entry
    
    if comparison_results:
        plot_comparison_chart(comparison_results, 'dataset_comparison_chart.png')
    else:
        print("[Warning] No training data successfully parsed.")


if __name__ == "__main__":
    main()
