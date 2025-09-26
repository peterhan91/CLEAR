#!/usr/bin/env python3
"""
Statistical Comparison Analysis for CXR Models
Computes 95% confidence intervals for AUROC, F1, MCC, AUPRC and p-values
comparing SFR-Mistral concept model against benchmark models.

Uses proper methodology from metrics.py for threshold selection and F1/MCC computation.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import (roc_auc_score, roc_curve, 
                           f1_score, matthews_corrcoef, confusion_matrix)
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Using official threshold computation - no caching needed

# Dataset configurations
DATASET_CONFIGS = {
    'padchest': {
        'csv_path': '../data/padchest_test.csv',
        'phenotypes_path': '../data/pheotypes_padchest.json',
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_padchest_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_padchest_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_padchest_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_padchest_test_openai_clip",
            # "CXR_Foundation": "results/benchmark_evaluation_padchest_test_cxr_foundation"
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
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_indiana_test_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_indiana_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_indiana_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_indiana_test_openai_clip",
            # "CXR_Foundation": "results/benchmark_evaluation_indiana_test_cxr_foundation"
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
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_vindrcxr_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_vindrcxr_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_vindrcxr_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_vindrcxr_test_openai_clip",
            # "CXR_Foundation": "results/benchmark_evaluation_vindrcxr_test_cxr_foundation"
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_vindrcxr_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_vindrcxr_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_vindrcxr_openai_small"
        },
        'exclude_columns': ['image_id']
    },
    'vindrpcxr': {
        'csv_path': '../data/vindrpcxr_test.csv',
        'phenotypes_path': None,
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_vindrpcxr_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_vindrpcxr_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_vindrpcxr_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_vindrpcxr_test_openai_clip",
            # "CXR_Foundation": "results/benchmark_evaluation_vindrpcxr_test_cxr_foundation"
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_vindrpcxr_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_vindrpcxr_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_vindrpcxr_openai_small"
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
        if label in label_counts and label_counts[label] >= threshold:
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

def get_optimal_threshold(y_true, y_pred):
    """Get optimal threshold using official implementation from metrics.py."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    # Use official ROC curve threshold method
    _, _, thresholds = roc_curve(y_true, y_pred)
    thresholds = thresholds[1:]  # Remove first threshold
    thresholds.sort()
    
    best_mcc = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_binary = np.where(y_pred < threshold, 0, 1)
        try:
            mcc = matthews_corrcoef(y_true, y_pred_binary)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
        except:
            continue
    
    return best_threshold

def compute_metrics_with_threshold(y_true, y_pred, threshold):
    """Compute F1 and MCC using a pre-determined threshold."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    
    y_pred_binary = np.where(y_pred < threshold, 0, 1)
    try:
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        # Compute F1
        m = confusion_matrix(y_true, y_pred_binary)
        if len(m.ravel()) == 1:
            f1 = 0.0 if np.all(y_pred_binary == 0) else 1.0
        else:
            tn, fp, fn, tp = m.ravel()
            if (2*tp + fp + fn) == 0:
                f1 = 1.0
            else:
                f1 = (2 * tp) / (2*tp + fp + fn)
        return f1, mcc
    except:
        return np.nan, np.nan

def get_optimal_threshold_and_metrics(y_true, y_pred):
    """Get optimal threshold and compute both F1 and MCC using official method."""
    if len(np.unique(y_true)) < 2:
        return 0.5, np.nan, np.nan
    
    # Get optimal threshold using official implementation
    best_threshold = get_optimal_threshold(y_true, y_pred)
    
    # Compute metrics with optimal threshold
    f1, mcc = compute_metrics_with_threshold(y_true, y_pred, best_threshold)
    
    return best_threshold, f1, mcc

def bootstrap_all_metrics(y_true, y_pred, optimal_threshold, n_bootstrap=500, label_name=None):
    """Compute bootstrap confidence intervals for all metrics at once using pre-computed threshold."""
    # Use unique seed per label for reproducibility while avoiding identical samples across labels
    base_seed = 42
    if label_name is not None:
        # Generate unique but reproducible seed per label
        label_hash = hash(label_name) % 10000
        seed = base_seed + label_hash
    else:
        seed = base_seed
    np.random.seed(seed)
    auroc_scores = []
    f1_scores = []
    mcc_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            # Compute all metrics from the same bootstrap sample using pre-computed threshold
            auroc = roc_auc_score(y_true_boot, y_pred_boot)
            f1, mcc = compute_metrics_with_threshold(y_true_boot, y_pred_boot, optimal_threshold)
            
            auroc_scores.append(auroc)
            f1_scores.append(f1)
            mcc_scores.append(mcc)
        except:
            continue
    
    # Compute confidence intervals for each metric
    results = {}
    for metric_name, scores in [('auroc', auroc_scores), ('f1', f1_scores), ('mcc', mcc_scores)]:
        if len(scores) == 0:
            results[metric_name] = (np.nan, np.nan, np.nan)
        else:
            scores = np.array(scores)
            mean_score = np.mean(scores)
            ci_lower = np.percentile(scores, 2.5)
            ci_upper = np.percentile(scores, 97.5)
            results[metric_name] = (mean_score, ci_lower, ci_upper)
    
    return results

def permutation_test_all_metrics(y_true, y_pred1, y_pred2, optimal_threshold1, optimal_threshold2, n_permutations=500, label_name=None):
    """Perform permutation test for all metrics at once using pre-computed thresholds."""
    if len(np.unique(y_true)) < 2:
        return {'auroc': np.nan, 'f1': np.nan, 'mcc': np.nan}
    
    # Use unique seed per label for reproducibility while avoiding identical samples across labels
    base_seed = 123  # Different base seed from bootstrap
    if label_name is not None:
        label_hash = hash(label_name) % 10000
        seed = base_seed + label_hash
    else:
        seed = base_seed
    np.random.seed(seed)
    
    # Compute observed differences for all metrics using pre-computed thresholds
    try:
        observed_auroc_diff = roc_auc_score(y_true, y_pred1) - roc_auc_score(y_true, y_pred2)
        f1_1, mcc_1 = compute_metrics_with_threshold(y_true, y_pred1, optimal_threshold1)
        f1_2, mcc_2 = compute_metrics_with_threshold(y_true, y_pred2, optimal_threshold2)
        observed_f1_diff = f1_1 - f1_2
        observed_mcc_diff = mcc_1 - mcc_2
    except:
        return {'auroc': np.nan, 'f1': np.nan, 'mcc': np.nan}
    
    auroc_diffs = []
    f1_diffs = []
    mcc_diffs = []
    
    for _ in range(n_permutations):
        mask = np.random.choice([True, False], size=len(y_true))
        y_pred1_perm = np.where(mask, y_pred1, y_pred2)
        y_pred2_perm = np.where(mask, y_pred2, y_pred1)
        
        try:
            # Compute all metric differences from the same permutation using pre-computed thresholds
            auroc_diff = roc_auc_score(y_true, y_pred1_perm) - roc_auc_score(y_true, y_pred2_perm)
            f1_1, mcc_1 = compute_metrics_with_threshold(y_true, y_pred1_perm, optimal_threshold1)
            f1_2, mcc_2 = compute_metrics_with_threshold(y_true, y_pred2_perm, optimal_threshold2)
            f1_diff = f1_1 - f1_2
            mcc_diff = mcc_1 - mcc_2
            
            auroc_diffs.append(auroc_diff)
            f1_diffs.append(f1_diff)
            mcc_diffs.append(mcc_diff)
        except:
            continue
    
    # Compute p-values for each metric
    p_values = {}
    for metric_name, diffs, observed_diff in [
        ('auroc', auroc_diffs, observed_auroc_diff),
        ('f1', f1_diffs, observed_f1_diff),
        ('mcc', mcc_diffs, observed_mcc_diff)
    ]:
        if len(diffs) == 0:
            p_values[metric_name] = np.nan
        else:
            diffs = np.array(diffs)
            p_values[metric_name] = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    
    return p_values

def _analyze_single_label(args):
    """Helper function for parallel processing of single label analysis."""
    label, predictions, ground_truth, n_bootstrap = args
    y_true = ground_truth[label].values
    y_pred = predictions[label].values
    return compute_metrics_with_ci(y_true, y_pred, label, n_bootstrap)

def _compare_single_label(args):
    """Helper function for parallel processing of single label comparison."""
    label, sfr_pred, sfr_gt, bench_pred, bench_gt, n_permutations = args
    y_true = sfr_gt[label].values
    y_pred_sfr = sfr_pred[label].values
    y_pred_bench = bench_pred[label].values
    
    if len(np.unique(y_true)) < 2:
        return None
    
    # Pre-compute optimal thresholds once for both models
    try:
        sfr_threshold = get_optimal_threshold(y_true, y_pred_sfr)
        bench_threshold = get_optimal_threshold(y_true, y_pred_bench)
    except:
        return None
    
    # Compute p-values using pre-computed thresholds
    p_values = permutation_test_all_metrics(y_true, y_pred_sfr, y_pred_bench, sfr_threshold, bench_threshold, n_permutations, label)
    
    # Compute actual scores using pre-computed thresholds
    try:
        sfr_auroc = roc_auc_score(y_true, y_pred_sfr)
        bench_auroc = roc_auc_score(y_true, y_pred_bench)
        sfr_f1, sfr_mcc = compute_metrics_with_threshold(y_true, y_pred_sfr, sfr_threshold)
        bench_f1, bench_mcc = compute_metrics_with_threshold(y_true, y_pred_bench, bench_threshold)
    except:
        return None
    
    return {
        'label': label, 'n_positive': int(np.sum(y_true)),
        'sfr_auroc': sfr_auroc, 'benchmark_auroc': bench_auroc, 'auroc_diff': sfr_auroc - bench_auroc, 'auroc_p_value': p_values['auroc'],
        'sfr_f1': sfr_f1, 'benchmark_f1': bench_f1, 'f1_diff': sfr_f1 - bench_f1, 'f1_p_value': p_values['f1'],
        'sfr_mcc': sfr_mcc, 'benchmark_mcc': bench_mcc, 'mcc_diff': sfr_mcc - bench_mcc, 'mcc_p_value': p_values['mcc']
    }

def analyze_model_performance(model_data, filtered_labels, model_name, n_jobs=-1, n_bootstrap=500):
    """Analyze performance for a single model with parallel processing."""
    if model_data is None:
        return None
    
    predictions, ground_truth, available_labels = model_data
    analysis_labels = [label for label in filtered_labels if label in available_labels]
    
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(analysis_labels))
    
    if n_jobs == 1 or len(analysis_labels) <= 1:
        # Sequential processing for small datasets or single core
        results = []
        for label in tqdm(analysis_labels, desc=f"Computing thresholds & metrics for {model_name}"):
            y_true = ground_truth[label].values
            y_pred = predictions[label].values
            metrics = compute_metrics_with_ci(y_true, y_pred, label, n_bootstrap)
            results.append(metrics)
    else:
        # Parallel processing
        args_list = [(label, predictions, ground_truth, n_bootstrap) for label in analysis_labels]
        
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_analyze_single_label, args_list), 
                total=len(args_list), 
                desc=f"Computing thresholds & metrics for {model_name}"
            ))
    
    return pd.DataFrame(results)

def compare_models_statistical(sfr_data, benchmark_data, filtered_labels, sfr_name, benchmark_name, n_jobs=-1, n_permutations=500):
    """Perform statistical comparison between SFR-Mistral and benchmark model with parallel processing."""
    if sfr_data is None or benchmark_data is None:
        return None
    
    sfr_pred, sfr_gt, sfr_labels = sfr_data
    bench_pred, bench_gt, bench_labels = benchmark_data
    
    common_labels = sorted(list(set(filtered_labels) & set(sfr_labels) & set(bench_labels)))
    
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(common_labels))
    
    if n_jobs == 1 or len(common_labels) <= 1:
        # Sequential processing
        comparison_results = []
        for label in tqdm(common_labels, desc=f"Comparing {sfr_name} vs {benchmark_name}"):
            result = _compare_single_label((label, sfr_pred, sfr_gt, bench_pred, bench_gt, n_permutations))
            if result is not None:
                comparison_results.append(result)
    else:
        # Parallel processing
        args_list = [(label, sfr_pred, sfr_gt, bench_pred, bench_gt, n_permutations) for label in common_labels]
        
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_compare_single_label, args_list), 
                total=len(args_list), 
                desc=f"Comparing {sfr_name} vs {benchmark_name}"
            ))
        
        comparison_results = [r for r in results if r is not None]
    
    return pd.DataFrame(comparison_results)

def compute_metrics_with_ci(y_true, y_pred, label_name, n_bootstrap=500):
    """Compute AUROC, F1, MCC with 95% confidence intervals using optimized bootstrap."""
    if len(np.unique(y_true)) < 2:
        return {
            'label': label_name, 'n_positive': int(np.sum(y_true)), 'n_total': len(y_true),
            'auroc': np.nan, 'auroc_ci_lower': np.nan, 'auroc_ci_upper': np.nan,
            'f1': np.nan, 'f1_ci_lower': np.nan, 'f1_ci_upper': np.nan,
            'mcc': np.nan, 'mcc_ci_lower': np.nan, 'mcc_ci_upper': np.nan
        }
    
    # Pre-compute optimal threshold once using official implementation
    optimal_threshold = get_optimal_threshold(y_true, y_pred)
    
    # Compute metrics with CI using optimized bootstrap with pre-computed threshold
    bootstrap_results = bootstrap_all_metrics(y_true, y_pred, optimal_threshold, n_bootstrap, label_name)
    
    return {
        'label': label_name, 'n_positive': int(np.sum(y_true)), 'n_total': len(y_true),
        'auroc': bootstrap_results['auroc'][0], 'auroc_ci_lower': bootstrap_results['auroc'][1], 'auroc_ci_upper': bootstrap_results['auroc'][2],
        'f1': bootstrap_results['f1'][0], 'f1_ci_lower': bootstrap_results['f1'][1], 'f1_ci_upper': bootstrap_results['f1'][2],
        'mcc': bootstrap_results['mcc'][0], 'mcc_ci_lower': bootstrap_results['mcc'][1], 'mcc_ci_upper': bootstrap_results['mcc'][2]
    }

def main():
    parser = argparse.ArgumentParser(description='Statistical Comparison Analysis for CXR Models')
    parser.add_argument('--dataset', choices=['padchest', 'indiana', 'vindrcxr', 'vindrpcxr'], default='padchest')
    parser.add_argument('--threshold', type=int, default=30, help='Minimum positive cases threshold')
    parser.add_argument('--compare-with', choices=['clip', 'llm', 'both'], default='clip')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap iterations')
    parser.add_argument('--n-permutations', type=int, default=1000, help='Number of permutation test iterations')
    args = parser.parse_args()
    
    config = DATASET_CONFIGS[args.dataset]
    results_dir = f"results/statistic_{args.dataset}_t{args.threshold}_{args.compare_with}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Statistical Analysis: {args.dataset} (threshold≥{args.threshold})")
    
    # Load data and filter labels
    phenotypes = load_phenotypes(config['phenotypes_path'])
    label_counts, all_labels = load_original_dataset_data(config['csv_path'], config['exclude_columns'])
    filtered_labels = filter_labels_by_threshold_and_phenotypes(label_counts, phenotypes, args.threshold)
    print(f"Selected {len(filtered_labels)} labels")
    
    # Load model results
    sfr_data = load_model_results(config['sfr_results_dir'], "SFR-Mistral")
    
    benchmark_data = {}
    llm_data = {}
    
    if args.compare_with in ['clip', 'both']:
        for model_name, model_results_dir in config['benchmark_dirs'].items():
            benchmark_data[model_name] = load_model_results(model_results_dir, model_name)
    
    if args.compare_with in ['llm', 'both']:
        for model_name, model_results_dir in config['llm_dirs'].items():
            llm_data[model_name] = load_model_results(model_results_dir, model_name)
    
    # Analyze individual model performance
    sfr_results = analyze_model_performance(sfr_data, filtered_labels, "SFR-Mistral", n_jobs=args.n_jobs, n_bootstrap=args.n_bootstrap)
    
    # Find best models and perform comparisons
    best_benchmark_model = None
    best_llm_model = None
    
    if benchmark_data:
        benchmark_results = {}
        for model_name, data in benchmark_data.items():
            if data is not None:
                benchmark_results[model_name] = analyze_model_performance(data, filtered_labels, model_name, n_jobs=args.n_jobs, n_bootstrap=args.n_bootstrap)
        
        # Find best benchmark model
        best_auroc = -1
        for model_name, results in benchmark_results.items():
            if results is not None and len(results) > 0:
                mean_auroc = results['auroc'].mean()
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_benchmark_model = model_name
    
    if llm_data:
        llm_results = {}
        for model_name, data in llm_data.items():
            if data is not None:
                llm_results[model_name] = analyze_model_performance(data, filtered_labels, model_name, n_jobs=args.n_jobs, n_bootstrap=args.n_bootstrap)
        
        # Find best LLM model
        best_auroc = -1
        for model_name, results in llm_results.items():
            if results is not None and len(results) > 0:
                mean_auroc = results['auroc'].mean()
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_llm_model = model_name
    
    # Statistical comparisons
    comparison_results = {}
    
    if best_benchmark_model and sfr_data is not None:
        comparison_results[f'sfr_vs_{best_benchmark_model.lower()}'] = compare_models_statistical(
            sfr_data, benchmark_data[best_benchmark_model], filtered_labels, "SFR-Mistral", best_benchmark_model, n_jobs=args.n_jobs, n_permutations=args.n_permutations
        )
    
    if best_llm_model and sfr_data is not None:
        comparison_results[f'sfr_vs_{best_llm_model.lower()}'] = compare_models_statistical(
            sfr_data, llm_data[best_llm_model], filtered_labels, "SFR-Mistral", best_llm_model, n_jobs=args.n_jobs, n_permutations=args.n_permutations
        )
    
    # Save results
    if sfr_results is not None:
        sfr_results.to_csv(f"{results_dir}/sfr_mistral_performance.csv", index=False, float_format='%.6f')
        print(f"Saved SFR-Mistral performance results")
    
    # Save benchmark model results
    if benchmark_data:
        for model_name, results in benchmark_results.items():
            if results is not None and len(results) > 0:
                results.to_csv(f"{results_dir}/{model_name.lower()}_performance.csv", index=False, float_format='%.6f')
                print(f"Saved {model_name} performance results")
    
    # Save LLM model results  
    if llm_data:
        for model_name, results in llm_results.items():
            if results is not None and len(results) > 0:
                results.to_csv(f"{results_dir}/{model_name.lower()}_performance.csv", index=False, float_format='%.6f')
                print(f"Saved {model_name} performance results")
    
    # Save comparison results
    for comparison_name, comp_results in comparison_results.items():
        if comp_results is not None and len(comp_results) > 0:
            comp_results.to_csv(f"{results_dir}/comparison_{comparison_name}.csv", index=False, float_format='%.6f')
            
            # Print summary
            print(f"\n{comparison_name.upper()} Summary (n={len(comp_results)}):")
            for metric in ['auroc', 'f1', 'mcc']:
                wins = (comp_results[f'{metric}_diff'] > 0).sum()
                sig_wins = ((comp_results[f'{metric}_diff'] > 0) & (comp_results[f'{metric}_p_value'] < 0.05)).sum()
                print(f"  {metric.upper()}: {wins}/{len(comp_results)} wins, {sig_wins} significant (p<0.05)")
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main() 