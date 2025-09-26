#!/usr/bin/env python3
"""
Simplified ROC-AUC Comparison Analysis for CXR Models
Compares SFR-Mistral against benchmark models across multiple datasets.
Computes average AUC, 95% confidence intervals, and p-values.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations with dataset-specific thresholds
DATASET_CONFIGS = {
    'padchest': {
        'csv_path': '../data/padchest_test.csv',
        'phenotypes_path': '../data/pheotypes_padchest.json',
        'threshold': 25,
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_padchest_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_padchest_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_padchest_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_padchest_test_openai_clip",
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_padchest_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_padchest_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_padchest_openai_small"
        },
        'exclude_columns': ['Study', 'Path', 'image_id', 'ImageID', 'name', 'is_test']
    },
    'indiana': {
        'csv_path': '../data/indiana_test.csv',
        'phenotypes_path': None,
        'threshold': 25,
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_indiana_test_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_indiana_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_indiana_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_indiana_test_openai_clip",
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_indiana_test_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_indiana_test_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_indiana_test_openai_small"
        },
        'exclude_columns': ['uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 
                           'indication', 'comparison', 'findings', 'impression']
    },
    'vindrcxr': {
        'csv_path': '../data/vindrcxr_test.csv',
        'phenotypes_path': None,
        'threshold': 10,
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_vindrcxr_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_vindrcxr_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_vindrcxr_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_vindrcxr_test_openai_clip",
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_vindrcxr_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_vindrcxr_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_vindrcxr_openai_small"
        },
        'exclude_columns': ['image_id']
    }
}

def load_phenotypes(phenotypes_path):
    """Load diagnostic phenotypes from JSON file."""
    if phenotypes_path is None or not os.path.exists(phenotypes_path):
        return None
    with open(phenotypes_path, 'r') as f:
        data = json.load(f)
    return data['diagnostic_phenotypes']

def load_original_dataset_data(csv_path, exclude_columns):
    """Load original dataset CSV to get label counts."""
    df = pd.read_csv(csv_path)
    label_columns = [col for col in df.columns if col not in exclude_columns]
    label_counts = {}
    for label in label_columns:
        if label in df.columns:
            positive_count = df[label].sum() if df[label].dtype in ['int64', 'float64'] else 0
            label_counts[label] = positive_count
    return label_counts, label_columns

def filter_labels_by_threshold_and_phenotypes(label_counts, phenotypes, threshold=30):
    """Filter labels based on positive count threshold and diagnostic phenotypes."""
    labels_to_check = phenotypes if phenotypes else list(label_counts.keys())
    filtered_labels = []
    for label in labels_to_check:
        if label in label_counts and label_counts[label] > threshold:
            filtered_labels.append(label)
    return filtered_labels

def load_model_results(results_dir, model_name):
    """Load model predictions and ground truth from results directory."""
    if not os.path.exists(results_dir):
        return None, None, None
    
    pred_files = list(Path(results_dir).glob("predictions_*.csv"))
    gt_files = list(Path(results_dir).glob("ground_truth_*.csv"))
    
    if not pred_files or not gt_files:
        return None, None, None
    
    # Use the latest files
    pred_file = sorted(pred_files)[-1]
    gt_file = sorted(gt_files)[-1]
    
    predictions = pd.read_csv(pred_file)
    ground_truth = pd.read_csv(gt_file)
    
    # Extract label names
    pred_labels = [col[:-5] for col in predictions.columns if col.endswith('_pred')]
    gt_labels = [col[:-5] for col in ground_truth.columns if col.endswith('_true')]
    common_labels = sorted(list(set(pred_labels) & set(gt_labels)))
    
    # Create clean DataFrames
    pred_clean = pd.DataFrame({label: predictions[f"{label}_pred"] for label in common_labels})
    gt_clean = pd.DataFrame({label: ground_truth[f"{label}_true"] for label in common_labels})
    
    return pred_clean, gt_clean, common_labels

def bootstrap_auc(y_true, y_pred, n_bootstrap=1000):
    """Compute bootstrap confidence interval for AUC."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    
    np.random.seed(42)  # For reproducibility
    auc_scores = []
    
    for _ in tqdm(range(n_bootstrap), desc="    Bootstrap AUC", leave=False, disable=n_bootstrap < 100):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            auc = roc_auc_score(y_true_boot, y_pred_boot)
            auc_scores.append(auc)
        except:
            continue
    
    if len(auc_scores) == 0:
        return np.nan, np.nan, np.nan
    
    mean_auc = np.mean(auc_scores)
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)
    
    return mean_auc, ci_lower, ci_upper

def permutation_test_auc(y_true, y_pred1, y_pred2, n_permutations=1000):
    """Perform permutation test for AUC difference."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    np.random.seed(123)  # For reproducibility
    
    # Compute observed difference
    try:
        observed_diff = roc_auc_score(y_true, y_pred1) - roc_auc_score(y_true, y_pred2)
    except:
        return np.nan
    
    diffs = []
    for _ in tqdm(range(n_permutations), desc="    Permutation test", leave=False, disable=n_permutations < 100):
        mask = np.random.choice([True, False], size=len(y_true))
        y_pred1_perm = np.where(mask, y_pred1, y_pred2)
        y_pred2_perm = np.where(mask, y_pred2, y_pred1)
        
        try:
            diff = roc_auc_score(y_true, y_pred1_perm) - roc_auc_score(y_true, y_pred2_perm)
            diffs.append(diff)
        except:
            continue
    
    if len(diffs) == 0:
        return np.nan
    
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_value

def analyze_dataset_performance(dataset_name, config, filtered_labels, n_bootstrap=1000, n_permutations=1000):
    """Analyze performance for a single dataset."""
    print(f"\nAnalyzing {dataset_name.upper()}...")
    
    # Load SFR-Mistral results
    print("  Loading SFR-Mistral results...")
    sfr_data = load_model_results(config['sfr_results_dir'], "SFR-Mistral")
    if sfr_data[0] is None:
        print(f"  No SFR-Mistral results found for {dataset_name}")
        return None
    
    sfr_pred, sfr_gt, sfr_labels = sfr_data
    
    # Load all other models
    all_models = {}
    
    print("  Loading CLIP-based models...")
    # CLIP-based models
    for model_name, results_dir in tqdm(config['benchmark_dirs'].items(), desc="  Loading CLIP models"):
        model_data = load_model_results(results_dir, model_name)
        if model_data[0] is not None:
            all_models[model_name] = model_data
    
    print("  Loading LLM models...")
    # LLM models
    for model_name, results_dir in tqdm(config['llm_dirs'].items(), desc="  Loading LLM models"):
        model_data = load_model_results(results_dir, model_name)
        if model_data[0] is not None:
            all_models[model_name] = model_data
    
    # Find common labels across all models
    common_labels = set(filtered_labels) & set(sfr_labels)
    for model_name, (pred, gt, labels) in all_models.items():
        common_labels = common_labels & set(labels)
    
    common_labels = sorted(list(common_labels))
    print(f"  Common labels: {len(common_labels)}")
    print(f"  Available models: {list(all_models.keys())}")
    
    if len(common_labels) == 0:
        print(f"  No common labels found for {dataset_name}")
        return None
    
    # Compute AUC for each model and label
    results = []
    
    for label in tqdm(common_labels, desc=f"  Processing {dataset_name} labels"):
        y_true = sfr_gt[label].values
        
        if len(np.unique(y_true)) < 2:
            continue
        
        # SFR-Mistral performance
        y_pred_sfr = sfr_pred[label].values
        try:
            sfr_auc, sfr_ci_lower, sfr_ci_upper = bootstrap_auc(y_true, y_pred_sfr, n_bootstrap)
        except:
            continue
        
        label_result = {
            'dataset': dataset_name,
            'label': label,
            'n_positive': int(np.sum(y_true)),
            'n_total': len(y_true),
            'SFR-Mistral_auc': sfr_auc,
            'SFR-Mistral_ci_lower': sfr_ci_lower,
            'SFR-Mistral_ci_upper': sfr_ci_upper
        }
        
        # Compare against other models
        for model_name, (pred, gt, _) in tqdm(all_models.items(), desc=f"    Comparing models for {label}", leave=False):
            if label in pred.columns:
                y_pred_model = pred[label].values
                try:
                    model_auc, model_ci_lower, model_ci_upper = bootstrap_auc(y_true, y_pred_model, n_bootstrap)
                    p_value = permutation_test_auc(y_true, y_pred_sfr, y_pred_model, n_permutations)
                    
                    label_result[f'{model_name}_auc'] = model_auc
                    label_result[f'{model_name}_ci_lower'] = model_ci_lower
                    label_result[f'{model_name}_ci_upper'] = model_ci_upper
                    label_result[f'SFR-Mistral_vs_{model_name}_p_value'] = p_value
                    label_result[f'SFR-Mistral_vs_{model_name}_diff'] = sfr_auc - model_auc
                except:
                    continue
        
        results.append(label_result)
    
    return pd.DataFrame(results)

def compute_average_performance(all_results):
    """Compute average AUC across all datasets for each model."""
    if not all_results:
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Get all model names
    model_columns = [col for col in combined_df.columns if col.endswith('_auc')]
    models = [col[:-4] for col in model_columns]
    
    # Compute averages
    avg_results = []
    for model in models:
        auc_col = f'{model}_auc'
        if auc_col in combined_df.columns:
            valid_aucs = combined_df[auc_col].dropna()
            if len(valid_aucs) > 0:
                avg_results.append({
                    'model': model,
                    'avg_auc': valid_aucs.mean(),
                    'std_auc': valid_aucs.std(),
                    'n_labels': len(valid_aucs),
                    'min_auc': valid_aucs.min(),
                    'max_auc': valid_aucs.max()
                })
    
    avg_df = pd.DataFrame(avg_results)
    
    # Compute overall comparisons
    if 'SFR-Mistral' in avg_df['model'].values:
        sfr_avg = avg_df[avg_df['model'] == 'SFR-Mistral']['avg_auc'].iloc[0]
        
        comparison_results = []
        for _, row in avg_df.iterrows():
            if row['model'] != 'SFR-Mistral':
                # Count wins and significant wins
                comparison_col = f'SFR-Mistral_vs_{row["model"]}_diff'
                p_value_col = f'SFR-Mistral_vs_{row["model"]}_p_value'
                
                if comparison_col in combined_df.columns:
                    valid_comparisons = combined_df[[comparison_col, p_value_col]].dropna()
                    wins = (valid_comparisons[comparison_col] > 0).sum()
                    sig_wins = ((valid_comparisons[comparison_col] > 0) & 
                               (valid_comparisons[p_value_col] < 0.05)).sum()
                    total_comparisons = len(valid_comparisons)
                    
                    comparison_results.append({
                        'comparison': f'SFR-Mistral vs {row["model"]}',
                        'sfr_avg_auc': sfr_avg,
                        'other_avg_auc': row['avg_auc'],
                        'avg_diff': sfr_avg - row['avg_auc'],
                        'wins': wins,
                        'total_comparisons': total_comparisons,
                        'win_rate': wins / total_comparisons if total_comparisons > 0 else 0,
                        'significant_wins': sig_wins,
                        'significant_win_rate': sig_wins / total_comparisons if total_comparisons > 0 else 0
                    })
        
        comparison_df = pd.DataFrame(comparison_results)
        return avg_df, comparison_df
    
    return avg_df, None

def create_pvalue_matrices(all_results, results_dir):
    """Create comprehensive p-value matrices for all model comparisons on all labels."""
    if not all_results:
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Get all models that have p-value comparisons
    pvalue_columns = [col for col in combined_df.columns if col.endswith('_p_value')]
    models = [col.replace('SFR-Mistral_vs_', '').replace('_p_value', '') for col in pvalue_columns]
    
    if not models:
        print("  No p-value comparisons found")
        return
    
    # Create matrices for each dataset
    datasets = combined_df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        labels = dataset_df['label'].tolist()
        
        if len(labels) == 0:
            continue
        
        print(f"  Creating p-value matrix for {dataset}")
        
        # Create p-value matrix
        pvalue_matrix = pd.DataFrame(index=labels, columns=models)
        auc_diff_matrix = pd.DataFrame(index=labels, columns=models)
        significance_matrix = pd.DataFrame(index=labels, columns=models)
        
        for _, row in dataset_df.iterrows():
            label = row['label']
            for model in models:
                pvalue_col = f'SFR-Mistral_vs_{model}_p_value'
                diff_col = f'SFR-Mistral_vs_{model}_diff'
                
                if pvalue_col in row and not pd.isna(row[pvalue_col]):
                    pvalue_matrix.loc[label, model] = row[pvalue_col]
                    auc_diff_matrix.loc[label, model] = row[diff_col] if diff_col in row else np.nan
                    significance_matrix.loc[label, model] = 'Yes' if row[pvalue_col] < 0.05 else 'No'
        
        # Save matrices
        pvalue_matrix.to_csv(f"{results_dir}/{dataset}_pvalue_matrix.csv", float_format='%.6f')
        auc_diff_matrix.to_csv(f"{results_dir}/{dataset}_auc_diff_matrix.csv", float_format='%.6f')
        significance_matrix.to_csv(f"{results_dir}/{dataset}_significance_matrix.csv")
        
        # Create summary statistics
        summary_stats = []
        for model in models:
            valid_pvalues = pvalue_matrix[model].dropna()
            valid_diffs = auc_diff_matrix[model].dropna()
            
            if len(valid_pvalues) > 0:
                significant_count = (valid_pvalues < 0.05).sum()
                wins = (valid_diffs > 0).sum() if len(valid_diffs) > 0 else 0
                sig_wins = ((valid_diffs > 0) & (valid_pvalues < 0.05)).sum()
                
                summary_stats.append({
                    'model': model,
                    'total_comparisons': len(valid_pvalues),
                    'significant_pvalues': significant_count,
                    'significance_rate': significant_count / len(valid_pvalues),
                    'wins': wins,
                    'win_rate': wins / len(valid_diffs) if len(valid_diffs) > 0 else 0,
                    'significant_wins': sig_wins,
                    'significant_win_rate': sig_wins / len(valid_pvalues),
                    'mean_pvalue': valid_pvalues.mean(),
                    'median_pvalue': valid_pvalues.median(),
                    'mean_auc_diff': valid_diffs.mean() if len(valid_diffs) > 0 else np.nan
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{results_dir}/{dataset}_comparison_summary.csv", index=False, float_format='%.6f')
    
    # Create overall p-value matrix across all datasets
    print("  Creating overall p-value matrix across all datasets")
    
    # Pivot to create label-model matrix
    overall_pvalue_data = []
    overall_diff_data = []
    
    for _, row in combined_df.iterrows():
        dataset_label = f"{row['dataset']}_{row['label']}"
        for model in models:
            pvalue_col = f'SFR-Mistral_vs_{model}_p_value'
            diff_col = f'SFR-Mistral_vs_{model}_diff'
            
            if pvalue_col in row and not pd.isna(row[pvalue_col]):
                overall_pvalue_data.append({
                    'dataset_label': dataset_label,
                    'model': model,
                    'p_value': row[pvalue_col],
                    'auc_diff': row[diff_col] if diff_col in row else np.nan
                })
    
    if overall_pvalue_data:
        overall_df = pd.DataFrame(overall_pvalue_data)
        
        # Create pivot tables
        overall_pvalue_pivot = overall_df.pivot(index='dataset_label', columns='model', values='p_value')
        overall_diff_pivot = overall_df.pivot(index='dataset_label', columns='model', values='auc_diff')
        
        # Create significance matrix
        overall_sig_pivot = overall_pvalue_pivot.applymap(lambda x: 'Yes' if pd.notna(x) and x < 0.05 else ('No' if pd.notna(x) else ''))
        
        # Save overall matrices
        overall_pvalue_pivot.to_csv(f"{results_dir}/overall_pvalue_matrix.csv", float_format='%.6f')
        overall_diff_pivot.to_csv(f"{results_dir}/overall_auc_diff_matrix.csv", float_format='%.6f')
        overall_sig_pivot.to_csv(f"{results_dir}/overall_significance_matrix.csv")
        
        # Create overall summary
        overall_summary = []
        for model in models:
            valid_pvalues = overall_pvalue_pivot[model].dropna()
            valid_diffs = overall_diff_pivot[model].dropna()
            
            if len(valid_pvalues) > 0:
                significant_count = (valid_pvalues < 0.05).sum()
                wins = (valid_diffs > 0).sum() if len(valid_diffs) > 0 else 0
                sig_wins = ((valid_diffs > 0) & (valid_pvalues < 0.05)).sum()
                
                overall_summary.append({
                    'model': model,
                    'total_comparisons': len(valid_pvalues),
                    'significant_pvalues': significant_count,
                    'significance_rate': significant_count / len(valid_pvalues),
                    'wins': wins,
                    'win_rate': wins / len(valid_diffs) if len(valid_diffs) > 0 else 0,
                    'significant_wins': sig_wins,
                    'significant_win_rate': sig_wins / len(valid_pvalues),
                    'mean_pvalue': valid_pvalues.mean(),
                    'median_pvalue': valid_pvalues.median(),
                    'min_pvalue': valid_pvalues.min(),
                    'max_pvalue': valid_pvalues.max(),
                    'mean_auc_diff': valid_diffs.mean() if len(valid_diffs) > 0 else np.nan,
                    'std_auc_diff': valid_diffs.std() if len(valid_diffs) > 0 else np.nan
                })
        
        overall_summary_df = pd.DataFrame(overall_summary)
        overall_summary_df.to_csv(f"{results_dir}/overall_comparison_summary.csv", index=False, float_format='%.6f')
        
        print(f"  Saved comprehensive p-value matrices and summaries")
        print(f"  Overall comparisons: {len(overall_pvalue_data)} label-model pairs")
    
    return len(overall_pvalue_data) if overall_pvalue_data else 0

def main():
    parser = argparse.ArgumentParser(description='Simplified ROC-AUC Comparison Analysis')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['padchest', 'indiana', 'vindrcxr'],
                       default=['padchest', 'indiana', 'vindrcxr'],
                       help='Datasets to analyze')
    parser.add_argument('--n-bootstrap', type=int, default=1000, 
                       help='Number of bootstrap iterations')
    parser.add_argument('--n-permutations', type=int, default=1000, 
                       help='Number of permutation test iterations')
    args = parser.parse_args()
    
    # Create results directory with dataset-specific thresholds
    thresholds_str = "_".join([f"{d}t{DATASET_CONFIGS[d]['threshold']}" for d in args.datasets])
    results_dir = f"results/simplified_auc_comparison_{thresholds_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Simplified AUC Comparison Analysis")
    print(f"Datasets: {', '.join(args.datasets)}")
    dataset_thresholds = [f"{d}: >{DATASET_CONFIGS[d]['threshold']}" for d in args.datasets]
    print(f"Thresholds: {', '.join(dataset_thresholds)}")
    
    all_results = []
    
    for dataset_name in tqdm(args.datasets, desc="Processing datasets"):
        config = DATASET_CONFIGS[dataset_name]
        dataset_threshold = config['threshold']
        
        # Load data and filter labels using dataset-specific threshold
        phenotypes = load_phenotypes(config['phenotypes_path'])
        label_counts, _ = load_original_dataset_data(config['csv_path'], config['exclude_columns'])
        filtered_labels = filter_labels_by_threshold_and_phenotypes(label_counts, phenotypes, dataset_threshold)
        
        print(f"\n{dataset_name.upper()}: {len(filtered_labels)} labels selected (threshold >{dataset_threshold})")
        
        # Analyze dataset
        dataset_results = analyze_dataset_performance(
            dataset_name, config, filtered_labels, 
            args.n_bootstrap, args.n_permutations
        )
        
        if dataset_results is not None and len(dataset_results) > 0:
            all_results.append(dataset_results)
            # Save individual dataset results
            dataset_results.to_csv(f"{results_dir}/{dataset_name}_results.csv", index=False, float_format='%.6f')
            print(f"  Saved {len(dataset_results)} label results")
    
    # Compute and save average performance
    if all_results:
        print("\nComputing average performance across datasets...")
        avg_results, comparison_results = compute_average_performance(all_results)
        
        if avg_results is not None:
            avg_results.to_csv(f"{results_dir}/average_performance.csv", index=False, float_format='%.6f')
            print(f"\nAverage Performance Summary:")
            for _, row in avg_results.iterrows():
                print(f"  {row['model']}: {row['avg_auc']:.3f} ± {row['std_auc']:.3f} (n={row['n_labels']})")
        
        if comparison_results is not None:
            comparison_results.to_csv(f"{results_dir}/comparison_summary.csv", index=False, float_format='%.6f')
            print(f"\nComparison Summary:")
            for _, row in comparison_results.iterrows():
                print(f"  {row['comparison']}: {row['wins']}/{row['total_comparisons']} wins "
                      f"({row['win_rate']:.2%}), {row['significant_wins']} significant ({row['significant_win_rate']:.2%})")
    
    # Create comprehensive p-value matrices for all model comparisons
    if all_results:
        print("\nCreating comprehensive p-value matrices...")
        n_comparisons = create_pvalue_matrices(all_results, results_dir)
        if n_comparisons > 0:
            print(f"Created p-value matrices with {n_comparisons} total comparisons")
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()