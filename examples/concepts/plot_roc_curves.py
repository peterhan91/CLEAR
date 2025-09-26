#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
import seaborn as sns
from pathlib import Path

def bootstrap_roc_auc(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """
    Calculate ROC curve with confidence intervals using bootstrapping
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities 
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level for confidence interval
    
    Returns:
        fpr_mean, tpr_mean, tpr_lower, tpr_upper, auc_mean, auc_ci
    """
    # Calculate original ROC curve
    fpr_orig, tpr_orig, _ = roc_curve(y_true, y_pred)
    auc_orig = roc_auc_score(y_true, y_pred)
    
    # Common FPR points for interpolation
    fpr_mean = np.linspace(0, 1, 100)
    
    # Bootstrap sampling
    tpr_bootstrap = []
    auc_bootstrap = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        # Calculate ROC for bootstrap sample
        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_pred_boot)
        auc_boot = roc_auc_score(y_true_boot, y_pred_boot)
        
        # Interpolate TPR at common FPR points
        tpr_interp = np.interp(fpr_mean, fpr_boot, tpr_boot)
        tpr_bootstrap.append(tpr_interp)
        auc_bootstrap.append(auc_boot)
    
    # Calculate confidence intervals
    tpr_bootstrap = np.array(tpr_bootstrap)
    auc_bootstrap = np.array(auc_bootstrap)
    
    # TPR confidence intervals
    tpr_lower = np.percentile(tpr_bootstrap, 100 * alpha/2, axis=0)
    tpr_upper = np.percentile(tpr_bootstrap, 100 * (1 - alpha/2), axis=0)
    tpr_mean = np.mean(tpr_bootstrap, axis=0)
    
    # AUC confidence intervals
    auc_lower = np.percentile(auc_bootstrap, 100 * alpha/2)
    auc_upper = np.percentile(auc_bootstrap, 100 * (1 - alpha/2))
    auc_mean = np.mean(auc_bootstrap)
    
    return fpr_mean, tpr_mean, tpr_lower, tpr_upper, auc_mean, (auc_lower, auc_upper)

def plot_roc_curves_with_ci():
    """Plot ROC-AUC curves with confidence intervals for all labels"""
    
    # Load data
    results_dir = Path("results/concept_based_linear_probing_torch")
    
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    y_true = np.load(results_dir / "y_true.npy")
    y_pred = np.load(results_dir / "y_pred.npy")
    labels = results['labels']
    
    print(f"Loaded data: {y_true.shape[0]} samples, {len(labels)} labels")
    
    # Setup plotting
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    
    # Use Set3 colormap for better color distinction across subplots
    try:
        # For newer matplotlib versions
        cmap = plt.colormaps['Set3']
    except:
        # For older matplotlib versions
        cmap = plt.cm.get_cmap('Set3')
    colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
    
    # Create subplots - 3x5 grid for 13 labels + 2 empty spaces
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot individual ROC curves
    for i, label in enumerate(labels):
        ax = axes[i]
        
        # Get data for this label
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        # Skip if no positive samples
        if np.sum(y_true_label) == 0:
            ax.text(0.5, 0.5, f'{label}\nNo positive samples', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            continue
        
        # Calculate ROC with confidence intervals
        fpr_mean, tpr_mean, tpr_lower, tpr_upper, auc_mean, auc_ci = bootstrap_roc_auc(
            y_true_label, y_pred_label, n_bootstrap=1000
        )
        
        # Plot ROC curve with confidence band
        ax.plot(fpr_mean, tpr_mean, color=colors[i], linewidth=2, 
               label=f'AUC = {auc_mean:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]')
        ax.fill_between(fpr_mean, tpr_lower, tpr_upper, 
                       color=colors[i], alpha=0.2)
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{label}\n(n_pos = {int(np.sum(y_true_label))})')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(labels), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    # Remove title as requested
    # plt.suptitle('ROC-AUC Curves with 95% Confidence Intervals\nConcept-Based Linear Probing', 
    #             y=0.98, fontsize=16, fontweight='bold')
    
    # Save plot
    output_dir = Path("results/concept_based_linear_probing_torch")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "roc_curves_with_ci.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "roc_curves_with_ci.pdf", bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}/roc_curves_with_ci.png and .pdf")
    
    # Create a summary plot with all curves
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(labels):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        if np.sum(y_true_label) == 0:
            continue
            
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true_label, y_pred_label)
        auc = roc_auc_score(y_true_label, y_pred_label)
        
        plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                label=f'{label} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Labels\nConcept-Based Linear Probing', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "roc_curves_summary.pdf", bbox_inches='tight')
    
    print(f"Summary plot saved to {output_dir}/roc_curves_summary.png and .pdf")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Overall AUC: {results['test_auc']:.4f}")
    print("\nPer-label AUCs with confidence intervals:")
    
    for i, label in enumerate(labels):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        if np.sum(y_true_label) == 0:
            print(f"  {label:25}: No positive samples")
            continue
            
        _, _, _, _, auc_mean, auc_ci = bootstrap_roc_auc(
            y_true_label, y_pred_label, n_bootstrap=1000
        )
        
        n_pos = int(np.sum(y_true_label))
        print(f"  {label:25}: {auc_mean:.4f} [{auc_ci[0]:.4f}-{auc_ci[1]:.4f}] (n_pos={n_pos})")

    plt.show()

if __name__ == "__main__":
    plot_roc_curves_with_ci() 