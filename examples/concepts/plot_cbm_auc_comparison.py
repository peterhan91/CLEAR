#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Set style
plt.style.use('default')
sns.set_theme(style="ticks")

def load_predictions(predictions_dir):
    """Load all prediction files and compute AUC scores"""
    predictions_dir = Path(predictions_dir)
    
    # Load zero-shot predictions
    with open(predictions_dir / "zeroshot_predictions.pkl", 'rb') as f:
        zeroshot_data = pickle.load(f)
    
    # Load seed predictions (42-61)
    seeds = list(range(42, 62))
    seed_data = {}
    for seed in seeds:
        seed_file = predictions_dir / f"seed_{seed}_predictions.pkl"
        if seed_file.exists():
            with open(seed_file, 'rb') as f:
                seed_data[seed] = pickle.load(f)
    
    return zeroshot_data, seed_data

def load_metrics_from_csv(metrics_dir):
    """Load F1 and MCC scores from CSV files"""
    metrics_dir = Path(metrics_dir)
    
    # Load F1 scores
    f1_df = pd.read_csv(metrics_dir / "cbm_experiments_f1_scores.csv")
    # Load MCC scores
    mcc_df = pd.read_csv(metrics_dir / "cbm_experiments_mcc_scores.csv")
    
    return f1_df, mcc_df

def process_csv_metrics(df, metric_name):
    """Process CSV metrics data into the same format as AUC results"""
    datasets = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    labels = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    methods = ['Standard', 'Improved']  # CSV uses 'Standard' and 'Improved' instead of 'Standard CBM'
    
    # Map CSV method names to display names
    method_mapping = {
        'Standard': 'Standard CBM',
        'Improved': 'CheXomni CBM'
    }
    
    results = {method_mapping[method]: {dataset: {label: [] for label in labels} for dataset in datasets} for method in methods}
    
    # Filter out ZeroShot results and process Standard/Improved
    for method in methods:
        method_data = df[df['Method'] == method]
        
        for dataset in datasets:
            dataset_data = method_data[method_data['Dataset'] == dataset]
            
            for label in labels:
                label_data = dataset_data[dataset_data['Label'] == label]
                
                if len(label_data) > 0:
                    scores = label_data[f'{metric_name}_Score'].values
                    results[method_mapping[method]][dataset][label] = scores.tolist()
                else:
                    results[method_mapping[method]][dataset][label] = []
    
    return results

def compute_auc_scores_per_label(y_true, y_pred, labels):
    """Compute AUC for each label individually"""
    aucs = {}
    for i, label in enumerate(labels):
        if len(np.unique(y_true[:, i])) > 1:  # Check if both classes are present
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs[label] = auc
        else:
            aucs[label] = 0.0
    return aucs

def collect_auc_results_per_label(zeroshot_data, seed_data):
    """Collect AUC results for all methods, datasets, and labels"""
    datasets = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    methods = ['Standard CBM', 'CheXomni CBM']  # Only these two methods
    
    # Get labels from any dataset
    sample_dataset = list(zeroshot_data.keys())[0]
    labels = zeroshot_data[sample_dataset]['labels']
    
    # Initialize results structure: {method: {dataset: {label: [aucs]}}}
    results = {method: {dataset: {label: [] for label in labels} for dataset in datasets} for method in methods}
    
    # Standard and CheXomni CBM results (across seeds)
    for seed, data in seed_data.items():
        for dataset in datasets:
            # Standard CBM
            if dataset in data['standard']:
                y_true = data['standard'][dataset]['y_true']
                y_pred = data['standard'][dataset]['y_pred']
                aucs = compute_auc_scores_per_label(y_true, y_pred, labels)
                for label in labels:
                    results['Standard CBM'][dataset][label].append(aucs[label])
            
            # CheXomni CBM
            if dataset in data['improved']:
                y_true = data['improved'][dataset]['y_true']
                y_pred = data['improved'][dataset]['y_pred']
                aucs = compute_auc_scores_per_label(y_true, y_pred, labels)
                for label in labels:
                    results['CheXomni CBM'][dataset][label].append(aucs[label])
    
    return results, labels

def create_single_metric_plot(ax, results, labels, metric_name, y_label, show_xaxis=True):
    """Create a single metric plot (AUROC, F1, or MCC)"""
    datasets = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    dataset_labels = ['CheXpert', 'VinDr-CXR', 'PadChest', 'Indiana']
    methods = ['Standard CBM', 'CheXomni CBM']
    
    # Colors for datasets and methods
    palette = sns.color_palette("Set2")
    dataset_colors = [palette[3], palette[0], palette[2], palette[1]]
    method_alphas = [1, 0.6]  # Different alpha for Standard vs Improved
    
    # Calculate positions
    n_diseases = len(labels)
    n_datasets = len(datasets)
    n_methods = len(methods)
    
    # Width of each bar and spacing
    bar_width = 0.25
    dataset_spacing = n_methods * bar_width
    disease_spacing = n_datasets * dataset_spacing + 0.6
    
    # Create x positions for each disease
    disease_positions = np.arange(n_diseases) * disease_spacing
    
    # Plot data for each disease, dataset, and method
    for disease_idx, disease in enumerate(labels):
        for dataset_idx, dataset in enumerate(datasets):
            for method_idx, method in enumerate(methods):
                data = results[method][dataset][disease]
                
                if len(data) > 0:
                    mean_score = np.mean(data)
                    std_score = np.std(data)
                else:
                    mean_score = 0.0
                    std_score = 0.0
                
                # Calculate x position
                base_pos = disease_positions[disease_idx]
                dataset_offset = dataset_idx * dataset_spacing
                method_offset = method_idx * bar_width
                x_pos = base_pos + dataset_offset + method_offset
                
                # Create bar
                bar = ax.bar(x_pos, mean_score, bar_width,
                           color=dataset_colors[dataset_idx], alpha=method_alphas[method_idx],
                           edgecolor='black', linewidth=.8)
                
                # Add error bar with cap_width=0
                if mean_score > 0:  # Only add error bar if we have data
                    ax.errorbar(x_pos, mean_score, yerr=std_score,
                               fmt='none', color='black', capsize=0, capthick=1,
                               linewidth=3, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel(y_label, fontsize=18)
    
    # Set x-axis ticks and labels
    disease_centers = disease_positions + (n_datasets - 1) * dataset_spacing / 2
    ax.set_xticks(disease_centers)
    
    if show_xaxis:
        ax.set_xticklabels(labels, fontsize=16, rotation=30, ha='right')
    else:
        ax.set_xticklabels([])
    
    # Styling
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle='--')
    
    # Set y-limits based on metric type
    if metric_name == 'AUROC':
        ax.set_ylim(0.7, 1.02)
    else:  # F1 and MCC
        ax.set_ylim(0, 1.02)
    
    # Style the plot
    ax.tick_params(axis='y', labelsize=16)
    sns.despine(ax=ax, left=True, bottom=True)

def create_comprehensive_cbm_plot(auc_results, f1_results, mcc_results, labels, save_path=None):
    """Create a comprehensive plot with two vertical subplots for AUROC and MCC"""
    datasets = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    dataset_labels = ['CheXpert', 'VinDr-CXR', 'PadChest', 'Indiana']
    methods = ['Standard CBM', 'CheXomni CBM']
    
    # Create figure with two vertical subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    
    # Create each subplot - show x-axis labels on both plots
    create_single_metric_plot(axes[0], auc_results, labels, 'AUROC', 'AUROC', show_xaxis=True)
    create_single_metric_plot(axes[1], mcc_results, labels, 'MCC', 'MCC Score', show_xaxis=True)
    
    # Add subplot titles with more space for vertical layout
    # axes[0].set_title('a', fontsize=20, fontweight='bold', pad=15, loc='left')
    # axes[1].set_title('b', fontsize=20, fontweight='bold', pad=15, loc='left')
    
    # Create legends only for the first subplot
    palette = sns.color_palette("Set2")
    dataset_colors = [palette[3], palette[0], palette[2], palette[1]]
    method_alphas = [1, 0.6]
    
    # Create custom legend elements
    dataset_handles = []
    for i, dataset_label in enumerate(dataset_labels):
        dataset_handles.append(plt.Rectangle((0,0),1,1, color=dataset_colors[i], alpha=0.7))
    
    method_handles = []
    for i, method in enumerate(methods):
        method_handles.append(plt.Rectangle((0,0),1,1, color='gray', alpha=method_alphas[i]))
    
    # Create legends on the top subplot, positioned better for vertical layout
    legend1 = axes[0].legend(dataset_handles, dataset_labels, 
                           loc='upper left', 
                           title='Test Dataset',
                           bbox_to_anchor=(0, 1), fontsize=14, title_fontsize=16, ncol=2)
    legend2 = axes[0].legend(method_handles, methods, 
                           title='Method',
                           loc='upper left', 
                           bbox_to_anchor=(0.26, 1), fontsize=14, title_fontsize=16, ncol=2)
    
    # Add the first legend back
    axes[0].add_artist(legend1)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to: {save_path}")
    
    plt.show()

def print_summary_table_per_label(results, labels):
    """Print a summary table of results for each label"""
    datasets = ['chexpert', 'vindrcxr', 'padchest', 'indiana']
    dataset_labels = ['CheXpert', 'VinDr-CXR', 'PadChest', 'Indiana']
    methods = ['Standard CBM', 'CheXomni CBM']
    
    for label in labels:
        print(f"\n" + "="*70)
        print(f"CBM AUC COMPARISON - {label.upper()}")
        print("="*70)
        
        print(f"{'Dataset':<12} {'Standard CBM':<20} {'CheXomni CBM':<20}")
        print("-" * 70)
        
        for dataset, dataset_label in zip(datasets, dataset_labels):
            row = f"{dataset_label:<12} "
            
            for method in methods:
                data = results[method][dataset][label]
                if isinstance(data, list) and len(data) > 0:
                    mean = np.mean(data)
                    std = np.std(data)
                    row += f"{mean:.3f}±{std:.3f}      "
                else:
                    row += f"0.000±0.000      "
            
            print(row)
        
        print("-" * 70)

def main():
    # Paths
    predictions_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck/predictions"
    metrics_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck/metrics"
    save_path = "cbm_comprehensive_comparison.png"
    
    print("Loading prediction files...")
    zeroshot_data, seed_data = load_predictions(predictions_dir)
    
    print(f"Loaded data for {len(seed_data)} seeds")
    print(f"Available datasets: {list(zeroshot_data.keys())}")
    
    print("Computing AUC scores per label...")
    auc_results, labels = collect_auc_results_per_label(zeroshot_data, seed_data)
    
    print("Loading F1 and MCC scores from CSV files...")
    f1_df, mcc_df = load_metrics_from_csv(metrics_dir)
    
    print("Processing F1 and MCC data...")
    f1_results = process_csv_metrics(f1_df, 'F1')
    mcc_results = process_csv_metrics(mcc_df, 'MCC')
    
    print(f"Labels: {labels}")
    
    print("Creating comprehensive CBM comparison plot...")
    create_comprehensive_cbm_plot(auc_results, f1_results, mcc_results, labels, save_path)
    
    print_summary_table_per_label(auc_results, labels)

if __name__ == "__main__":
    main() 