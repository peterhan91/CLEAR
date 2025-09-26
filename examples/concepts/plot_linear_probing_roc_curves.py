#!/usr/bin/env python3
"""
Plot ROC curves for Linear Probing models using 20-seed repeated experiments
with confidence intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("deep")

def load_linear_probing_predictions(results_dir, seeds):
    """
    Load predictions from all seeds for linear probing model.
    
    Args:
        results_dir: Directory containing seed prediction files
        seeds: List of seed values to load
    
    Returns:
        predictions: List of (y_true, y_pred) across seeds
        labels: Label names
        model_name: Name of the model
    """
    predictions_dir = Path(results_dir) / "predictions"
    
    predictions = []
    labels = None
    model_name = None
    
    for seed in seeds:
        seed_file = predictions_dir / f"seed_{seed}_predictions.pkl"
        
        if not seed_file.exists():
            print(f"Warning: {seed_file} not found, skipping seed {seed}")
            continue
            
        # Load seed predictions
        with open(seed_file, 'rb') as f:
            seed_data = pickle.load(f)
        
        # Convert lists to numpy arrays
        y_true = np.array(seed_data['test']['y_true'])
        y_pred = np.array(seed_data['test']['y_pred'])
        
        predictions.append((y_true, y_pred))
        
        # Get labels and model name (should be the same across all seeds)
        if labels is None:
            labels = seed_data['labels']
            # Get model name - handle both 'method' and 'model_name' keys
            if 'method' in seed_data:
                model_name = seed_data['method']
            elif 'model_name' in seed_data:
                model_name = seed_data['model_name']
            else:
                model_name = "unknown"
    
    print(f"Loaded predictions from {len(predictions)} seeds for {model_name}")
    return predictions, labels, model_name

def compute_roc_curves_with_ci(predictions_list, label_idx):
    """
    Compute ROC curves across multiple seeds and calculate confidence intervals.
    
    Args:
        predictions_list: List of (y_true, y_pred) tuples from different seeds
        label_idx: Index of the label to analyze
    
    Returns:
        fpr_grid: Common FPR grid for interpolation
        tpr_mean: Mean TPR across seeds
        tpr_lower: 2.5th percentile TPR
        tpr_upper: 97.5th percentile TPR
        auc_scores: AUC scores for each seed
    """
    # Common FPR grid for interpolation
    fpr_grid = np.linspace(0, 1, 100)
    
    tpr_curves = []
    auc_scores = []
    skipped_seeds = 0
    
    for y_true, y_pred in predictions_list:
        # Extract data for this label
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        
        # Skip if only one class present
        if len(np.unique(y_true_label)) < 2:
            skipped_seeds += 1
            continue
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_label, y_pred_label)
        
        # Interpolate to common grid
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_curves.append(tpr_interp)
        
        # Compute AUC
        auc_scores.append(auc(fpr, tpr))
    
    # Check if we have any valid curves
    if len(tpr_curves) == 0:
        print(f"    Warning: No valid ROC curves (all {len(predictions_list)} seeds skipped - only one class present)")
        # Return dummy values
        return fpr_grid, np.full(len(fpr_grid), 0.5), np.full(len(fpr_grid), 0.5), np.full(len(fpr_grid), 0.5), np.array([0.5])
    
    if skipped_seeds > 0:
        print(f"    Warning: {skipped_seeds}/{len(predictions_list)} seeds skipped (only one class present)")
    
    tpr_curves = np.array(tpr_curves)
    
    # Compute statistics across seeds
    tpr_mean = np.mean(tpr_curves, axis=0)
    tpr_lower = np.percentile(tpr_curves, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_curves, 97.5, axis=0)
    
    return fpr_grid, tpr_mean, tpr_lower, tpr_upper, np.array(auc_scores)

def plot_roc_curve_for_label(model_predictions, label_name, label_idx, output_dir):
    """Plot ROC curve for a single label comparing all linear probing models."""
    
    # Create figure with white background and 1:1 aspect ratio
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    
    # Colors for different models
    colors = ['indianred', 'steelblue', 'darkgreen', 'darkorange']
    
    # Pretty model names for legend
    model_display_names = {
        'concept_based_linear_probing': 'CheXomni',
        'chexzero': 'CheXzero',
        'biomedclip': 'BiomedCLIP', 
        'openai_clip': 'OpenAI CLIP'
    }
    
    valid_models = 0
    
    # Plot ROC curves for each model
    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        print(f"  Computing ROC curve for {model_name}...")
        
        fpr_grid, tpr_mean, tpr_lower, tpr_upper, auc_scores = \
            compute_roc_curves_with_ci(predictions, label_idx)
        
        # Check if this is a valid curve
        is_dummy = (len(auc_scores) == 1 and auc_scores[0] == 0.5)
        
        if is_dummy:
            print(f"  Warning: {model_name} has invalid ROC curve for {label_name}")
            continue
        
        valid_models += 1
        
        # Get display name
        display_name = model_display_names.get(model_name, model_name)
        
        # Plot mean ROC curve
        ax.plot(fpr_grid, tpr_mean, color=colors[i % len(colors)], linewidth=3, alpha=1,
                label = f'{display_name} (AUC = {np.mean(auc_scores):.3f})'
                # label=f'{display_name} (AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f})'
                )
        
        # Plot confidence interval as shaded area
        ax.fill_between(fpr_grid, tpr_lower, tpr_upper, 
                       color=colors[i % len(colors)], alpha=0.2)
        
        # Print summary statistics
        print(f"  {model_name}: AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f} "
              f"(range: {np.min(auc_scores):.3f} - {np.max(auc_scores):.3f})")
    
    # Skip plotting if no valid models
    if valid_models == 0:
        print(f"  Skipping plot for {label_name} - no valid ROC curves")
        plt.close()
        return
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Customize plot
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False positive rate', fontsize=34)
    ax.set_ylabel('True positive rate', fontsize=34)
    ax.set_title(f'{label_name}', fontsize=34, pad=10)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Customize legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             fontsize=22, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save figure
    output_file = os.path.join(output_dir, f'roc_curve_{label_name.replace(" ", "_").lower()}_linear_probing.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"  Saved ROC curve to {output_file}")

def main():
    # Configuration
    base_results_dir = "concepts/results/"
    output_dir = "concepts/results/roc_curves_linear_probing/"
    
    # Model directories
    model_dirs = {
        'concept_based_linear_probing': 'concept_based_linear_probing_torch',
        'chexzero': 'baseline_linear_probing_chexzero',
        'biomedclip': 'baseline_linear_probing_biomedclip',
        'openai_clip': 'baseline_linear_probing_openai_clip'
    }
    
    # Use all available labels from the model
    target_labels = None  # Will be set from loaded data
    
    # Seeds (20 repeated experiments)
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PLOTTING LINEAR PROBING ROC CURVES FROM 20-SEED EXPERIMENTS")
    print("=" * 60)
    print(f"Base results directory: {base_results_dir}")
    print(f"Models: {list(model_dirs.keys())}")
    print(f"Seeds: {len(seeds)} ({min(seeds)} to {max(seeds)})")
    print(f"Output directory: {output_dir}")
    
    # Load predictions for all models
    print(f"\nLoading predictions from all models...")
    all_model_predictions = {}
    all_labels = None
    
    for model_name, model_dir in model_dirs.items():
        results_dir = os.path.join(base_results_dir, model_dir)
        
        print(f"\nLoading {model_name} from {results_dir}...")
        predictions, labels, loaded_model_name = load_linear_probing_predictions(results_dir, seeds)
        
        if not predictions:
            print(f"Error: No predictions loaded for {model_name}. Skipping...")
            continue
        
        all_model_predictions[model_name] = predictions
        
        # Store labels from first model (should be same across all)
        if all_labels is None:
            all_labels = labels
            print(f"Available labels: {all_labels}")
    
    if not all_model_predictions:
        print("Error: No model predictions loaded. Exiting...")
        return
    
    # Set target_labels to all available labels
    target_labels = all_labels
    print(f"Target labels: {target_labels}")
    
    print(f"\nSuccessfully loaded predictions from {len(all_model_predictions)} models")
    
    # Check data consistency
    for model_name, predictions in all_model_predictions.items():
        y_true_shape = predictions[0][0].shape
        y_pred_shape = predictions[0][1].shape
        print(f"{model_name}: y_true={y_true_shape}, y_pred={y_pred_shape}, seeds={len(predictions)}")
    
    # Generate ROC curves for target labels
    print(f"\nGenerating ROC curves for {len(target_labels)} target labels...")
    
    total_plots_generated = 0
    
    for label_name in target_labels:
        if label_name not in all_labels:
            print(f"Warning: {label_name} not found in model labels, skipping...")
            continue
            
        label_idx = all_labels.index(label_name)
        print(f"\nProcessing {label_name} (index {label_idx})...")
        
        plot_roc_curve_for_label(all_model_predictions, label_name, label_idx, output_dir)
        total_plots_generated += 1
    
    print(f"\n" + "=" * 60)
    print("COMPLETED LINEAR PROBING ROC CURVES!")
    print(f"All ROC curves saved to: {output_dir}")
    print(f"Total plots generated: {total_plots_generated}")
    print(f"Models compared: {len(all_model_predictions)} ({', '.join(all_model_predictions.keys())})")
    print(f"Labels plotted: {len(target_labels)} ({', '.join(target_labels)})")
    print("=" * 60)

if __name__ == "__main__":
    main() 