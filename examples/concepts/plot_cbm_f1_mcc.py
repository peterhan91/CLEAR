#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import seaborn as sns

# Set style for clean plots
plt.style.use('default')
sns.set_palette("husl")

def load_radiologist_data(data_path):
    """Load radiologist data from CSV files."""
    radiologist_files = ['bc4.csv', 'bc6.csv', 'bc8.csv']
    radiologist_names = ['Radiologist 1', 'Radiologist 2', 'Radiologist 3']
    
    radiologist_data = {}
    for file, name in zip(radiologist_files, radiologist_names):
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            radiologist_data[name] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    return radiologist_data

def compute_radiologist_f1_mcc(radiologist_df, y_true, label_idx):
    """Compute F1 and MCC scores for radiologist."""
    # Get radiologist predictions for this label (+1 to skip Study column)
    y_pred_rad = radiologist_df.iloc[:, label_idx + 1].values
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred_rad))
    y_true_sub = y_true[:min_len]
    y_pred_rad_sub = y_pred_rad[:min_len]
    
    # Skip if only one class present
    if len(np.unique(y_true_sub)) < 2:
        return 0.0, 0.0
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_sub, y_pred_rad_sub)
    if cm.size == 1:
        return 0.0, 0.0
    
    tn, fp, fn, tp = cm.ravel()
    
    # Compute F1 score
    if (2*tp + fp + fn) == 0:
        f1_score = 1.0
    else:
        f1_score = (2 * tp) / (2*tp + fp + fn)
    
    # Compute MCC
    try:
        mcc_score = matthews_corrcoef(y_true_sub, y_pred_rad_sub)
        if np.isnan(mcc_score):
            mcc_score = 0.0
    except:
        mcc_score = 0.0
    
    return f1_score, mcc_score

def load_cbm_results():
    """Load CBM F1 and MCC results from CSV files and calculate statistics."""
    f1_df = pd.read_csv('results/concept_bottleneck/metrics/cbm_experiments_f1_scores.csv')
    mcc_df = pd.read_csv('results/concept_bottleneck/metrics/cbm_experiments_mcc_scores.csv')
    
    # Filter for CheXpert dataset only
    f1_chexpert = f1_df[f1_df['Dataset'] == 'chexpert'].copy()
    mcc_chexpert = mcc_df[mcc_df['Dataset'] == 'chexpert'].copy()
    
    # Calculate statistics across seeds for each method and label
    target_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    cbm_stats = {
        'f1': {'Standard CBM': {}, 'Improved CBM': {}},
        'mcc': {'Standard CBM': {}, 'Improved CBM': {}}
    }
    
    for label in target_labels:
        for method_name, csv_method in [('Standard CBM', 'Standard'), ('Improved CBM', 'Improved')]:
            # F1 statistics
            f1_values = f1_chexpert[(f1_chexpert['Method'] == csv_method) & 
                                   (f1_chexpert['Label'] == label)]['F1_Score'].values
            f1_mean = np.mean(f1_values)
            f1_std = np.std(f1_values, ddof=1)  # Sample standard deviation
            f1_ci = 1.96 * f1_std / np.sqrt(len(f1_values))  # 95% CI
            
            cbm_stats['f1'][method_name][label] = {
                'mean': f1_mean,
                'std': f1_std,
                'ci': f1_ci,
                'values': f1_values
            }
            
            # MCC statistics
            mcc_values = mcc_chexpert[(mcc_chexpert['Method'] == csv_method) & 
                                     (mcc_chexpert['Label'] == label)]['MCC_Score'].values
            mcc_mean = np.mean(mcc_values)
            mcc_std = np.std(mcc_values, ddof=1)  # Sample standard deviation
            mcc_ci = 1.96 * mcc_std / np.sqrt(len(mcc_values))  # 95% CI
            
            cbm_stats['mcc'][method_name][label] = {
                'mean': mcc_mean,
                'std': mcc_std,
                'ci': mcc_ci,
                'values': mcc_values
            }
    
    return f1_chexpert, mcc_chexpert, cbm_stats

def load_ground_truth():
    """Load ground truth for CheXpert test set."""
    import pickle
    
    # Load from any seed predictions file (ground truth should be the same)
    with open('results/concept_bottleneck/predictions/seed_42_predictions.pkl', 'rb') as f:
        predictions = pickle.load(f)
    
    # Get ground truth from standard CBM results for chexpert
    y_true = predictions['standard']['chexpert']['y_true']
    labels = predictions['standard']['chexpert']['labels']
    
    return y_true, labels

def create_comparison_plots():
    """Create F1 and MCC comparison plots."""
    # Target labels
    target_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    # Radiologist headers (order matters!)
    radiologist_headers = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
        "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    # Label mapping
    label_mapping = {
        "Atelectasis": "Atelectasis",
        "Cardiomegaly": "Cardiomegaly", 
        "Consolidation": "Consolidation",
        "Edema": "Edema",
        "Pleural Effusion": "Pleural Effusion"
    }
    
    # Load data
    print("Loading CBM results...")
    f1_df, mcc_df, cbm_stats = load_cbm_results()
    
    print("Loading radiologist data...")
    radiologist_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_radiologist/"
    radiologist_data = load_radiologist_data(radiologist_dir)
    
    print("Loading ground truth...")
    y_true, cbm_labels = load_ground_truth()
    
    # Prepare data for plotting
    results_f1 = {label: {'Radiologist 1': [], 'Radiologist 2': [], 'Radiologist 3': [], 'Standard CBM': [], 'Improved CBM': []} 
                  for label in target_labels}
    results_mcc = {label: {'Radiologist 1': [], 'Radiologist 2': [], 'Radiologist 3': [], 'Standard CBM': [], 'Improved CBM': []} 
                   for label in target_labels}
    
    # Error bars for CBM models (95% CI)
    f1_errors = {label: {'Standard CBM': 0, 'Improved CBM': 0} for label in target_labels}
    mcc_errors = {label: {'Standard CBM': 0, 'Improved CBM': 0} for label in target_labels}
    
    # Get CBM results (mean and CI across seeds)
    for label in target_labels:
        # Standard CBM
        results_f1[label]['Standard CBM'] = cbm_stats['f1']['Standard CBM'][label]['mean']
        results_mcc[label]['Standard CBM'] = cbm_stats['mcc']['Standard CBM'][label]['mean']
        f1_errors[label]['Standard CBM'] = cbm_stats['f1']['Standard CBM'][label]['ci']
        mcc_errors[label]['Standard CBM'] = cbm_stats['mcc']['Standard CBM'][label]['ci']
        
        # Improved CBM
        results_f1[label]['Improved CBM'] = cbm_stats['f1']['Improved CBM'][label]['mean']
        results_mcc[label]['Improved CBM'] = cbm_stats['mcc']['Improved CBM'][label]['mean']
        f1_errors[label]['Improved CBM'] = cbm_stats['f1']['Improved CBM'][label]['ci']
        mcc_errors[label]['Improved CBM'] = cbm_stats['mcc']['Improved CBM'][label]['ci']
    
    # Get radiologist results
    for label in target_labels:
        if label in label_mapping and label_mapping[label] in radiologist_headers:
            rad_label_idx = radiologist_headers.index(label_mapping[label])
            cbm_label_idx = cbm_labels.index(label)
            
            for rad_name, rad_df in radiologist_data.items():
                f1_score, mcc_score = compute_radiologist_f1_mcc(rad_df, y_true[:, cbm_label_idx], rad_label_idx)
                results_f1[label][rad_name] = f1_score
                results_mcc[label][rad_name] = mcc_score
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('CheXpert Test Set Performance: Radiologists vs CBM Models', fontsize=16, fontweight='bold')
    
    # Colors and markers
    colors = ['#2E8B57', '#228B22', '#32CD32', '#4682B4', '#1E90FF']  # Green for radiologists, blue for models
    methods = ['Radiologist 1', 'Radiologist 2', 'Radiologist 3', 'Standard CBM', 'Improved CBM']
    
    x_pos = np.arange(len(target_labels))
    width = 0.15
    
    # Plot F1 scores
    for i, method in enumerate(methods):
        f1_values = [results_f1[label][method] for label in target_labels]
        color = colors[i]
        
        # Add error bars for CBM models only
        if method in ['Standard CBM', 'Improved CBM']:
            f1_err = [f1_errors[label][method] for label in target_labels]
            ax1.bar(x_pos + i * width, f1_values, width, label=method, color=color, alpha=0.8,
                   yerr=f1_err, capsize=0, error_kw={'linewidth': 1.5})
        else:
            ax1.bar(x_pos + i * width, f1_values, width, label=method, color=color, alpha=0.8)
    
    ax1.set_xlabel('Condition', fontweight='bold')
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score Comparison', fontweight='bold')
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels(target_labels, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Plot MCC scores
    for i, method in enumerate(methods):
        mcc_values = [results_mcc[label][method] for label in target_labels]
        color = colors[i]
        
        # Add error bars for CBM models only
        if method in ['Standard CBM', 'Improved CBM']:
            mcc_err = [mcc_errors[label][method] for label in target_labels]
            ax2.bar(x_pos + i * width, mcc_values, width, label=method, color=color, alpha=0.8,
                   yerr=mcc_err, capsize=0, error_kw={'linewidth': 1.5})
        else:
            ax2.bar(x_pos + i * width, mcc_values, width, label=method, color=color, alpha=0.8)
    
    ax2.set_xlabel('Condition', fontweight='bold')
    ax2.set_ylabel('MCC Score', fontweight='bold')
    ax2.set_title('MCC Score Comparison', fontweight='bold')
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels(target_labels, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "results/concept_bottleneck/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cbm_f1_mcc_comparison_chexpert.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CHEXPERT F1 AND MCC COMPARISON SUMMARY")
    print("="*60)
    for label in target_labels:
        print(f"\n{label}:")
        for method in methods:
            f1_val = results_f1[label][method]
            mcc_val = results_mcc[label][method]
            
            if method in ['Standard CBM', 'Improved CBM']:
                f1_ci = f1_errors[label][method]
                mcc_ci = mcc_errors[label][method]
                print(f"  {method:15s}: F1={f1_val:.3f}±{f1_ci:.3f}, MCC={mcc_val:.3f}±{mcc_ci:.3f}")
            else:
                print(f"  {method:15s}: F1={f1_val:.3f}, MCC={mcc_val:.3f}")
    
    plt.show()

if __name__ == "__main__":
    create_comparison_plots() 