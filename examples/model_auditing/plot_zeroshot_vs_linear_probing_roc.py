#!/usr/bin/env python3
"""
Plot ROC curves comparing Zero-shot and Linear Probing models on 
Enlarged Cardiomediastinum label for CheXpert test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import os

# Set style
plt.style.use('default')

def load_zero_shot_predictions(results_dir, filename_timestamp):
    """Load zero-shot model predictions from CSV files."""
    pred_file = os.path.join(results_dir, f"predictions_{filename_timestamp}.csv")
    gt_file = os.path.join(results_dir, f"ground_truth_{filename_timestamp}.csv")
    
    # Load predictions and ground truth
    pred_df = pd.read_csv(pred_file)
    gt_df = pd.read_csv(gt_file)
    
    # Extract Enlarged Cardiomediastinum data
    y_pred = pred_df['Enlarged Cardiomediastinum_pred'].values
    y_true = gt_df['Enlarged Cardiomediastinum_true'].values
    
    return y_true, y_pred

def load_linear_probing_predictions(results_file):
    """Load linear probing model predictions from pickle file."""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract test data
    y_true_list = data['test']['y_true']
    y_pred_list = data['test']['y_pred']
    labels = data['labels']
    
    # Convert to numpy arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # Get index for Enlarged Cardiomediastinum
    target_idx = labels.index('Enlarged Cardiomediastinum')
    
    return y_true[:, target_idx], y_pred[:, target_idx]

def plot_roc_comparison(zero_shot_ours_data, zero_shot_chexzero_data, 
                       linear_probing_ours_data, linear_probing_chexzero_data, output_path):
    """Plot ROC curve comparison between all four models."""
    
    # Create figure with white background and 1:1 aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    
    # Unpack data
    y_true_zs_ours, y_pred_zs_ours = zero_shot_ours_data
    y_true_zs_chexzero, y_pred_zs_chexzero = zero_shot_chexzero_data
    y_true_lp_ours, y_pred_lp_ours = linear_probing_ours_data
    y_true_lp_chexzero, y_pred_lp_chexzero = linear_probing_chexzero_data
    
    # Compute ROC curves
    fpr_zs_ours, tpr_zs_ours, _ = roc_curve(y_true_zs_ours, y_pred_zs_ours)
    fpr_zs_chexzero, tpr_zs_chexzero, _ = roc_curve(y_true_zs_chexzero, y_pred_zs_chexzero)
    fpr_lp_ours, tpr_lp_ours, _ = roc_curve(y_true_lp_ours, y_pred_lp_ours)
    fpr_lp_chexzero, tpr_lp_chexzero, _ = roc_curve(y_true_lp_chexzero, y_pred_lp_chexzero)
    
    # Compute AUC scores
    auc_zs_ours = auc(fpr_zs_ours, tpr_zs_ours)
    auc_zs_chexzero = auc(fpr_zs_chexzero, tpr_zs_chexzero)
    auc_lp_ours = auc(fpr_lp_ours, tpr_lp_ours)
    auc_lp_chexzero = auc(fpr_lp_chexzero, tpr_lp_chexzero)
    
    # Plot ROC curves with updated colors
    ax.plot(fpr_zs_ours, tpr_zs_ours, color='lightcoral', linewidth=3, alpha=1,
            label=f'CheXomni (zero-shot AUC = {auc_zs_ours:.3f})')
    
    ax.plot(fpr_zs_chexzero, tpr_zs_chexzero, color='lightblue', linewidth=3, alpha=1,
            label=f'CheXzero (zero-shot AUC = {auc_zs_chexzero:.3f})')
    
    ax.plot(fpr_lp_ours, tpr_lp_ours, color='indianred', linewidth=3, alpha=1,
            label=f'CheXomni (linear probe AUC = {auc_lp_ours:.3f})')
    
    ax.plot(fpr_lp_chexzero, tpr_lp_chexzero, color='steelblue', linewidth=3, alpha=1,
            label=f'CheXzero (linear probe AUC = {auc_lp_chexzero:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Customize plot
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate', fontsize=30)
    ax.set_ylabel('True Positive Rate', fontsize=30)
    ax.set_title('Enlarged Cardiomediastinum \n (zero-shot vs. probing)', 
                fontsize=30, pad=10)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Customize legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             fontsize=20, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    # Print results
    print(f"CheXomni: Zero-shot, AUC = {auc_zs_ours:.3f}")
    print(f"CheXzero: Zero-shot, AUC = {auc_zs_chexzero:.3f}")
    print(f"CheXomni: Linear Probing, AUC = {auc_lp_ours:.3f}")
    print(f"Linear Probing: AUC = {auc_lp_chexzero:.3f}")
    print(f"ROC curve saved to: {output_path}")

def main():
    # Configuration
    zero_shot_ours_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/benchmark/results/benchmark_evaluation_chexpert_test_dinov2-multi-v1.0_vitb_res448"
    zero_shot_chexzero_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/benchmark/results/benchmark_evaluation_chexpert_test_chexzero"
    linear_probing_ours_file = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_based_linear_probing_torch/predictions/seed_42_predictions.pkl"
    linear_probing_chexzero_file = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/baseline_linear_probing_chexzero/predictions/seed_42_predictions.pkl"
    output_path = "model_auditing/roc_curve_enlarged_cardiomediastinum_comparison.png"
    
    print("=" * 80)
    print("COMPARING ZERO-SHOT AND LINEAR PROBING MODELS ON ENLARGED CARDIOMEDIASTINUM")
    print("=" * 80)
    
    # Load zero-shot ours predictions
    print("Loading zero-shot ours (DINOv2) predictions...")
    y_true_zs_ours, y_pred_zs_ours = load_zero_shot_predictions(zero_shot_ours_dir, "20250614_213522")
    print(f"Zero-shot ours: {len(y_true_zs_ours)} samples loaded")
    
    # Load zero-shot CheXzero predictions
    print("Loading zero-shot CheXzero predictions...")
    y_true_zs_chexzero, y_pred_zs_chexzero = load_zero_shot_predictions(zero_shot_chexzero_dir, "20250614_213950")
    print(f"Zero-shot CheXzero: {len(y_true_zs_chexzero)} samples loaded")
    
    # Load linear probing ours predictions
    print("Loading linear probing ours predictions...")
    y_true_lp_ours, y_pred_lp_ours = load_linear_probing_predictions(linear_probing_ours_file)
    print(f"Linear probing ours: {len(y_true_lp_ours)} samples loaded")
    
    # Load linear probing CheXzero predictions
    print("Loading linear probing CheXzero predictions...")
    y_true_lp_chexzero, y_pred_lp_chexzero = load_linear_probing_predictions(linear_probing_chexzero_file)
    print(f"Linear probing CheXzero: {len(y_true_lp_chexzero)} samples loaded")
    
    # Check data consistency
    print(f"Sample counts - ZS Ours: {len(y_true_zs_ours)}, ZS CheXzero: {len(y_true_zs_chexzero)}, "
          f"LP Ours: {len(y_true_lp_ours)}, LP CheXzero: {len(y_true_lp_chexzero)}")
    
    # Plot comparison
    print("\nGenerating ROC curve comparison...")
    plot_roc_comparison(
        (y_true_zs_ours, y_pred_zs_ours), 
        (y_true_zs_chexzero, y_pred_zs_chexzero),
        (y_true_lp_ours, y_pred_lp_ours),
        (y_true_lp_chexzero, y_pred_lp_chexzero),
        output_path
    )
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main() 