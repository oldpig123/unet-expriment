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

def parse_log_file(filepath):
    """
    Parses training metrics from a log file.
    Looks for lines like:
    Epoch 01/50 | Train Loss: 0.180670 | Val Dice: 0.9214 | Val IoU: 0.8561 | Val HD: 30.77 px
    """
    epochs = []
    losses = []
    dices = []
    ious = []
    hds = []
    
    if not os.path.exists(filepath):
        print(f"[Warning] Log file not found: {filepath}")
        return epochs, losses, dices, ious, hds
        
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s*\|\s*Train\s+Loss:\s*([\d.]+)\s*\|\s*Val\s+Dice:\s*([\d.]+)\s*\|\s*Val\s+IoU:\s*([\d.]+)\s*\|\s*Val\s+HD:\s*([\d.]+)\s*px"
    )
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                dices.append(float(match.group(3)))
                ious.append(float(match.group(4)))
                hds.append(float(match.group(5)))
                
    return epochs, losses, dices, ious, hds

def plot_learning_curves(epochs, losses, dices, ious, hds, title, output_path):
    if not epochs:
        print(f"[Info] No epochs to plot for {title}")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"U-ResNet + SAAM Learning Curves: {title}", weight='bold', y=0.98)
    
    # 1. Training Loss
    axes[0, 0].plot(epochs, losses, color='#dc3545', linewidth=2.5, marker='o', markersize=4, label='SpineLoss')
    axes[0, 0].set_title('Training Loss Curve', weight='semibold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 2. Validation Dice
    axes[0, 1].plot(epochs, dices, color='#0d6efd', linewidth=2.5, marker='s', markersize=4, label='Dice Coefficient')
    axes[0, 1].set_title('Validation Dice Score', weight='semibold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend()
    
    # 3. Validation IoU
    axes[1, 0].plot(epochs, ious, color='#198754', linewidth=2.5, marker='^', markersize=4, label='IoU (Jaccard Index)')
    axes[1, 0].set_title('Validation Intersection over Union (IoU)', weight='semibold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()
    
    # 4. Validation HD
    axes[1, 1].plot(epochs, hds, color='#6f42c1', linewidth=2.5, marker='d', markersize=4, label='Hausdorff Distance')
    axes[1, 1].set_title('Validation Hausdorff Distance (HD)', weight='semibold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('HD (pixels)')
    axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved learning curves to: {output_path}")

def plot_comparison_chart(results, output_path):
    """
    Plots a bar chart comparing performance metrics across datasets.
    """
    datasets = list(results.keys())
    dices = [results[d]['best_dice'] for d in datasets]
    ious = [results[d]['best_iou'] for d in datasets]
    hds = [results[d]['best_hd'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Dataset Performance Comparison (Best Validation Metrics)", weight='bold', y=0.98)
    
    # Left axis for Dice and IoU
    rects1 = ax1.bar(x - width/2, dices, width, label='Best Val Dice', color='#0d6efd')
    rects2 = ax1.bar(x + width/2, ious, width, label='Val IoU', color='#198754')
    ax1.set_ylabel('Score (Dice / IoU)', weight='semibold')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, weight='semibold')
    
    # Right axis for HD
    ax2 = ax1.twinx()
    rects3 = ax2.plot(x, hds, color='#dc3545', marker='o', markersize=8, linewidth=2.5, label='Val HD (px)')
    ax2.set_ylabel('Hausdorff Distance (pixels)', color='#dc3545', weight='semibold')
    ax2.tick_params(axis='y', labelcolor='#dc3545')
    ax2.set_ylim(0, max(hds + [50]) * 1.2)
    
    # Legend construction
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Annotate bars with values
    for rect in rects1:
        height = rect.get_height()
        if height > 0:
            ax1.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, weight='bold')
                        
    for rect in rects2:
        height = rect.get_height()
        if height > 0:
            ax1.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, weight='bold')
                        
    for i, hd_val in enumerate(hds):
        if hd_val > 0:
            ax2.annotate(f'{hd_val:.2f} px',
                        xy=(i, hd_val),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom', color='#dc3545', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved comparison bar chart to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse training log files and plot metrics.")
    parser.add_argument("--log_dir", type=str, default=".", help="Tasks logs directory")
    args = parser.parse_args()
    
    # Task log filenames
    logs = {
        'Mendeley Lumbar MRI': os.path.join(args.log_dir, 'mri_train.log'),
        'VerSe 19 CT': os.path.join(args.log_dir, 'verse19_train.log'),
        'VerSe 20 CT': os.path.join(args.log_dir, 'verse20_train.log'),
    }
    
    results = {}
    
    for dataset_name, log_path in logs.items():
        print(f"Parsing {dataset_name} log from: {log_path}")
        epochs, losses, dices, ious, hds = parse_log_file(log_path)
        
        if epochs:
            # Save individual learning curves
            out_name = dataset_name.lower().replace(' ', '_') + '_curves.png'
            plot_learning_curves(epochs, losses, dices, ious, hds, dataset_name, out_name)
            
            # Find best metrics (based on Dice score)
            best_idx = np.argmax(dices)
            results[dataset_name] = {
                'best_dice': dices[best_idx],
                'best_iou': ious[best_idx],
                'best_hd': hds[best_idx]
            }
        else:
            print(f"No metric data found/parsed for {dataset_name}")
            
    if results:
        plot_comparison_chart(results, 'dataset_comparison_chart.png')
    else:
        print("[Warning] No training data successfully parsed; comparison chart was not generated.")

if __name__ == "__main__":
    main()
