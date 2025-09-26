#!/usr/bin/env python3
"""
Plot ROC curves for Standard & LLM-based CBM models using 20-seed repeated experiments
with confidence intervals and human expert results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import json
import os
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("deep")

def load_seed_predictions(results_dir, seeds, dataset_name='chexpert'):
    """
    Load predictions from all seeds for both Standard and Improved CBM.
    
    Args:
        results_dir: Directory containing seed prediction files
        seeds: List of seed values to load
        dataset_name: Test dataset to use ('chexpert', 'padchest', 'vindrcxr', 'indiana')
    
    Returns:
        standard_predictions: List of (y_true, y_pred) for standard CBM across seeds
        improved_predictions: List of (y_true, y_pred) for improved CBM across seeds
        labels: Label names
    """
    predictions_dir = Path(results_dir) / "predictions"
    
    standard_predictions = []
    improved_predictions = []
    labels = None
    
    for seed in seeds:
        seed_file = predictions_dir / f"seed_{seed}_predictions.pkl"
        
        if not seed_file.exists():
            print(f"Warning: {seed_file} not found, skipping seed {seed}")
            continue
            
        # Load seed predictions
        with open(seed_file, 'rb') as f:
            seed_data = pickle.load(f)
        
        # Extract data for the specified dataset
        if dataset_name in seed_data['standard'] and dataset_name in seed_data['improved']:
            # Standard CBM
            std_data = seed_data['standard'][dataset_name]
            standard_predictions.append((std_data['y_true'], std_data['y_pred']))
            
            # Improved CBM
            imp_data = seed_data['improved'][dataset_name]
            improved_predictions.append((imp_data['y_true'], imp_data['y_pred']))
            
            # Get labels (should be the same across all seeds)
            if labels is None:
                labels = std_data['labels']
        else:
            print(f"Warning: Dataset {dataset_name} not found in seed {seed}")
    
    print(f"Loaded predictions from {len(standard_predictions)} seeds for {dataset_name}")
    return standard_predictions, improved_predictions, labels

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

def load_radiologist_data(data_path):
    """Load radiologist data from CSV files."""
    radiologist_files = ['bc4.csv', 'bc6.csv', 'bc8.csv']
    radiologist_names = ['Radiologist 1', 'Radiologist 2', 'Radiologist 3']
    markers = ['^', 'v', 'd']  # triangle, inverted triangle, diamond
    
    radiologist_data = {}
    
    for i, (file, name, marker) in enumerate(zip(radiologist_files, radiologist_names, markers)):
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            radiologist_data[name] = {
                'data': df,
                'marker': marker,
                'file': file
            }
        else:
            print(f"Warning: {file_path} not found")
    
    return radiologist_data

def compute_radiologist_metrics(radiologist_df, y_true, label_idx, label_name):
    """Compute sensitivity and specificity for radiologist."""
    # Get radiologist predictions for this label
    y_pred_rad = radiologist_df.iloc[:, label_idx + 1].values  # +1 to skip Study column
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred_rad))
    y_true_sub = y_true[:min_len]
    y_pred_rad_sub = y_pred_rad[:min_len]
    
    # Compute confusion matrix elements
    tp = np.sum((y_true_sub == 1) & (y_pred_rad_sub == 1))
    tn = np.sum((y_true_sub == 0) & (y_pred_rad_sub == 0))
    fp = np.sum((y_true_sub == 0) & (y_pred_rad_sub == 1))
    fn = np.sum((y_true_sub == 1) & (y_pred_rad_sub == 0))
    
    # Compute sensitivity (TPR) and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    fpr = 1 - specificity
    
    return fpr, sensitivity

def plot_roc_curve_for_label(standard_predictions, improved_predictions, label_name, 
                           label_idx, radiologist_data, output_dir, dataset_name):
    """Plot ROC curve for a single label with confidence intervals and radiologist results."""
    
    # Create figure with white background and 1:1 aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    
    # Compute ROC curves with confidence intervals
    print(f"  Computing Standard CBM ROC curve...")
    fpr_grid_std, tpr_mean_std, tpr_lower_std, tpr_upper_std, auc_scores_std = \
        compute_roc_curves_with_ci(standard_predictions, label_idx)
    
    print(f"  Computing LLM-based CBM ROC curve...")
    fpr_grid_imp, tpr_mean_imp, tpr_lower_imp, tpr_upper_imp, auc_scores_imp = \
        compute_roc_curves_with_ci(improved_predictions, label_idx)
    
    # Check if we have valid data to plot
    std_is_dummy = (len(auc_scores_std) == 1 and auc_scores_std[0] == 0.5)
    imp_is_dummy = (len(auc_scores_imp) == 1 and auc_scores_imp[0] == 0.5)
    
    if std_is_dummy and imp_is_dummy:
        print(f"  Skipping plot for {label_name} - no valid ROC curves (only one class present)")
        return
    elif std_is_dummy or imp_is_dummy:
        print(f"  Warning: One method has invalid ROC curve for {label_name}, proceeding with available data")
    
    # Plot mean ROC curves
    if not std_is_dummy:
        ax.plot(fpr_grid_std, tpr_mean_std, color='steelblue', linewidth=3, alpha=1,
                label=f'Standard CBM (AUC = {np.mean(auc_scores_std):.3f})')
    else:
        ax.plot(fpr_grid_std, tpr_mean_std, color='steelblue', linewidth=3, alpha=0.3,
                label='Standard CBM (Invalid - one class only)')
    
    if not imp_is_dummy:
        ax.plot(fpr_grid_imp, tpr_mean_imp, color='indianred', linewidth=3, alpha=1,
                label=f'CheXomni CBM (AUC = {np.mean(auc_scores_imp):.3f})')
    else:
        ax.plot(fpr_grid_imp, tpr_mean_imp, color='indianred', linewidth=3, alpha=0.3,
                label='LLM-based CBM (Invalid - one class only)')
    
    # Plot confidence intervals as shaded areas
    if not std_is_dummy:
        ax.fill_between(fpr_grid_std, tpr_lower_std, tpr_upper_std, 
                       color='steelblue', alpha=0.2, label='Standard CBM 95% CI')
    
    if not imp_is_dummy:
        ax.fill_between(fpr_grid_imp, tpr_lower_imp, tpr_upper_imp,
                       color='indianred', alpha=0.2, label='CheXomni CBM 95% CI')
    
    # Radiologist headers (order matters!)
    radiologist_headers = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
        "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    # Map label name to radiologist header
    label_mapping = {
        "Atelectasis": "Atelectasis",
        "Cardiomegaly": "Cardiomegaly", 
        "Consolidation": "Consolidation",
        "Edema": "Edema",
        "Pleural Effusion": "Pleural Effusion"
    }
    
    # Plot radiologist results if available
    if (radiologist_data and 
        label_name in label_mapping and 
        label_mapping[label_name] in radiologist_headers):
        
        rad_label_idx = radiologist_headers.index(label_mapping[label_name])
        
        # Use ground truth from first seed (should be the same across seeds)
        y_true_ref = standard_predictions[0][0][:, label_idx]
        
        print(f"  Adding radiologist results...")
        colors = ['orange', 'orange', 'orange']
        for i, (rad_name, rad_info) in enumerate(radiologist_data.items()):
            fpr_rad, tpr_rad = compute_radiologist_metrics(
                rad_info['data'], y_true_ref, rad_label_idx, label_name)
            
            ax.scatter(fpr_rad, tpr_rad, marker=rad_info['marker'], 
                       s=800, color=colors[i], edgecolors='black', linewidth=2,
                       label=rad_name, zorder=5)
    elif not radiologist_data:
        print(f"  No radiologist data available for {dataset_name}")
    else:
        print(f"  Radiologist data not available for {label_name} in {dataset_name}")
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Customize plot
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate', fontsize=34)
    ax.set_ylabel('True Positive Rate', fontsize=34)
    ax.set_title(f'{label_name}', 
                fontsize=34, pad=10)
    
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
    output_file = os.path.join(output_dir, f'roc_curve_{label_name.replace(" ", "_").lower()}_{dataset_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"  Saved ROC curve to {output_file}")
    
    # Print summary statistics
    if not std_is_dummy:
        print(f"  Standard CBM: AUC = {np.mean(auc_scores_std):.3f} ± {np.std(auc_scores_std):.3f} "
              f"(range: {np.min(auc_scores_std):.3f} - {np.max(auc_scores_std):.3f})")
    else:
        print(f"  Standard CBM: Invalid (only one class present)")
    
    if not imp_is_dummy:
        print(f"  LLM-based CBM: AUC = {np.mean(auc_scores_imp):.3f} ± {np.std(auc_scores_imp):.3f} "
              f"(range: {np.min(auc_scores_imp):.3f} - {np.max(auc_scores_imp):.3f})")
    else:
        print(f"  LLM-based CBM: Invalid (only one class present)")

def main():
    # Configuration
    results_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck/"
    radiologist_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_radiologist/"
    output_dir = "concepts/results/roc_curves_cbm/"
    
    # Target labels (5 shared labels)
    target_labels = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    # Seeds (20 repeated experiments)
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    
    # All test datasets to generate ROC curves for
    dataset_names = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PLOTTING CBM ROC CURVES FROM 20-SEED EXPERIMENTS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Datasets: {dataset_names}")
    print(f"Target labels: {target_labels}")
    print(f"Seeds: {len(seeds)} ({min(seeds)} to {max(seeds)})")
    print(f"Output directory: {output_dir}")
    
    # Load radiologist data (only available for CheXpert)
    print(f"\nLoading radiologist data...")
    radiologist_data = load_radiologist_data(radiologist_dir)
    print(f"Loaded {len(radiologist_data)} radiologist datasets (for CheXpert only)")
    
    total_plots_generated = 0
    
    # Generate ROC curves for each dataset
    for dataset_name in dataset_names:
        print(f"\n" + "=" * 50)
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print("=" * 50)
        
        # Load predictions from all seeds for this dataset
        print(f"Loading predictions from 20 seeds for {dataset_name}...")
        standard_predictions, improved_predictions, all_labels = load_seed_predictions(
            results_dir, seeds, dataset_name)
        
        if not standard_predictions or not improved_predictions:
            print(f"Error: No predictions loaded for {dataset_name}. Skipping...")
            continue
        
        print(f"Successfully loaded predictions from {len(standard_predictions)} seeds")
        print(f"Available labels: {all_labels}")
        
        # Check data consistency
        y_true_shape = standard_predictions[0][0].shape
        y_pred_shape = standard_predictions[0][1].shape
        print(f"Data shapes: y_true={y_true_shape}, y_pred={y_pred_shape}")
        print(f"Number of samples: {y_true_shape[0]}")
        print(f"Number of labels: {y_true_shape[1]}")
        
        # Generate ROC curves for target labels
        print(f"\nGenerating ROC curves for {len(target_labels)} target labels...")
        
        # Only use radiologist data for CheXpert
        current_radiologist_data = radiologist_data if dataset_name == 'chexpert' else {}
        
        for label_name in target_labels:
            if label_name not in all_labels:
                print(f"Warning: {label_name} not found in model labels for {dataset_name}, skipping...")
                continue
                
            label_idx = all_labels.index(label_name)
            print(f"\nProcessing {label_name} (index {label_idx}) for {dataset_name}...")
            
            plot_roc_curve_for_label(
                standard_predictions, improved_predictions, label_name, 
                label_idx, current_radiologist_data, output_dir, dataset_name)
            
            total_plots_generated += 1
        
        print(f"\nCompleted {dataset_name.upper()} dataset!")
    
    print(f"\n" + "=" * 60)
    print("COMPLETED ALL DATASETS!")
    print(f"All ROC curves saved to: {output_dir}")
    print(f"Total plots generated: {total_plots_generated}")
    print(f"Datasets processed: {len(dataset_names)} ({', '.join(dataset_names)})")
    print(f"Labels per dataset: {len(target_labels)}")
    print("Note: Radiologist comparisons only shown for CheXpert dataset")
    print("=" * 60)

if __name__ == "__main__":
    main()
