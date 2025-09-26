#!/usr/bin/env python3
"""
Linear Probing ROC-AUC Comparison Analysis for CXR Models
Compares SFR-Mistral (concept-based) against baseline models (CheXzero, BiomedCLIP, OpenAI CLIP) 
across linear probing experiments.
Uses prediction files to compute AUCs for each run, then calculates 95% confidence intervals 
and pairwise p-values using the distribution of AUC values across runs.
"""

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Model configurations for linear probing results
MODEL_CONFIGS = {
    'sfr_mistral': {
        'results_dir': 'results/concept_based_linear_probing_torch',
        'display_name': 'SFR-Mistral (Concept-based)',
    },
    'chexzero': {
        'results_dir': 'results/baseline_linear_probing_chexzero',
        'display_name': 'CheXzero',
    },
    'biomedclip': {
        'results_dir': 'results/baseline_linear_probing_biomedclip',
        'display_name': 'BiomedCLIP',
    },
    'openai_clip': {
        'results_dir': 'results/baseline_linear_probing_openai_clip',
        'display_name': 'OpenAI CLIP',
    }
}

def load_prediction_files(results_dir):
    """Load all prediction files from a model's results directory."""
    predictions_dir = os.path.join(results_dir, 'predictions')
    
    if not os.path.exists(predictions_dir):
        print(f"  Warning: {predictions_dir} not found")
        return None
    
    prediction_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('_predictions.pkl')])
    
    if not prediction_files:
        print(f"  Warning: No prediction files found in {predictions_dir}")
        return None
    
    all_predictions = []
    for pred_file in prediction_files:
        pred_path = os.path.join(predictions_dir, pred_file)
        try:
            with open(pred_path, 'rb') as f:
                data = pickle.load(f)
                all_predictions.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {pred_file}: {e}")
            continue
    
    return all_predictions

def compute_aucs_from_predictions(predictions_list):
    """Compute AUC for each label from list of prediction dictionaries."""
    if not predictions_list:
        return None, None
    
    # Get labels from first prediction file
    labels = predictions_list[0]['labels']
    
    # Compute AUCs for each run and each label
    all_aucs = []
    
    for pred_data in predictions_list:
        y_true = np.array(pred_data['test']['y_true'])  # Shape: (n_samples, n_labels)
        y_pred = np.array(pred_data['test']['y_pred'])  # Shape: (n_samples, n_labels)
        
        run_aucs = {}
        for i, label in enumerate(labels):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            
            # Only compute AUC if we have both classes
            if len(np.unique(y_true_label)) < 2:
                run_aucs[label] = np.nan
            else:
                try:
                    auc = roc_auc_score(y_true_label, y_pred_label)
                    run_aucs[label] = auc
                except Exception as e:
                    print(f"    Warning: Could not compute AUC for {label}: {e}")
                    run_aucs[label] = np.nan
        
        all_aucs.append(run_aucs)
    
    # Convert to label-wise arrays
    per_label_aucs = {}
    for label in labels:
        aucs = [run_aucs[label] for run_aucs in all_aucs if not np.isnan(run_aucs[label])]
        if aucs:
            per_label_aucs[label] = np.array(aucs)
    
    return per_label_aucs, labels

def compute_auc_statistics(aucs):
    """Compute statistics for AUC values from multiple runs."""
    if len(aucs) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_runs': 0
        }
    
    aucs = np.array(aucs)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0
    
    # 95% confidence interval using percentiles of the empirical distribution
    if len(aucs) >= 3:
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
    else:
        # For very small samples, use t-distribution
        if len(aucs) > 1:
            t_val = stats.t.ppf(0.975, len(aucs) - 1)
            margin = t_val * std_auc / np.sqrt(len(aucs))
            ci_lower = mean_auc - margin
            ci_upper = mean_auc + margin
        else:
            ci_lower = ci_upper = mean_auc
    
    return {
        'mean': mean_auc,
        'std': std_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_runs': len(aucs)
    }

def compare_auc_distributions(aucs1, aucs2, test_type='mannwhitney'):
    """Compare two AUC distributions using statistical tests."""
    if len(aucs1) == 0 or len(aucs2) == 0:
        return {
            'mean_diff': np.nan,
            'p_value': np.nan,
            'test_statistic': np.nan,
            'test_type': test_type
        }
    
    aucs1 = np.array(aucs1)
    aucs2 = np.array(aucs2)
    mean_diff = np.mean(aucs1) - np.mean(aucs2)
    
    # Choose statistical test
    if test_type == 'ttest' and len(aucs1) >= 3 and len(aucs2) >= 3:
        # Two-sample t-test (assumes normality)
        statistic, p_value = stats.ttest_ind(aucs1, aucs2)
    elif test_type == 'mannwhitney' and len(aucs1) >= 3 and len(aucs2) >= 3:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(aucs1, aucs2, alternative='two-sided')
    elif len(aucs1) >= 2 and len(aucs2) >= 2:
        # For small samples, use Wilcoxon rank-sum test
        try:
            statistic, p_value = stats.ranksums(aucs1, aucs2)
        except:
            statistic, p_value = np.nan, np.nan
    else:
        statistic, p_value = np.nan, np.nan
    
    return {
        'mean_diff': mean_diff,
        'p_value': p_value,
        'test_statistic': statistic,
        'test_type': test_type
    }

def analyze_linear_probing_performance(model_configs, test_type='mannwhitney'):
    """Analyze linear probing performance across all models."""
    print("Loading linear probing prediction files...")
    
    # Load all model results
    all_models = {}
    for model_name, config in model_configs.items():
        print(f"  Loading {config['display_name']}...")
        predictions_list = load_prediction_files(config['results_dir'])
        if predictions_list is not None:
            per_label_aucs, labels = compute_aucs_from_predictions(predictions_list)
            if per_label_aucs is not None:
                all_models[model_name] = {
                    'display_name': config['display_name'],
                    'per_label_aucs': per_label_aucs,
                    'labels': labels
                }
                print(f"    Loaded {len(predictions_list)} runs with {len(per_label_aucs)} labels")
    
    if not all_models:
        print("No model results found!")
        return None
    
    print(f"Successfully loaded {len(all_models)} models")
    
    # Find common labels across all models
    common_labels = None
    for model_name, model_data in all_models.items():
        labels = set(model_data['per_label_aucs'].keys())
        if common_labels is None:
            common_labels = labels
        else:
            common_labels = common_labels.intersection(labels)
    
    common_labels = sorted(list(common_labels))
    print(f"Common labels across all models: {len(common_labels)}")
    print(f"Labels: {common_labels}")
    
    if len(common_labels) == 0:
        print("No common labels found!")
        return None
    
    # Compute comparisons
    results = []
    
    # Get SFR-Mistral as reference model
    reference_model = 'sfr_mistral'
    if reference_model not in all_models:
        reference_model = list(all_models.keys())[0]
        print(f"SFR-Mistral not found, using {reference_model} as reference")
    
    reference_data = all_models[reference_model]
    
    for label in tqdm(common_labels, desc="Processing labels"):
        if label not in reference_data['per_label_aucs']:
            continue
            
        reference_aucs = reference_data['per_label_aucs'][label]
        
        # Compute statistics for reference model
        ref_stats = compute_auc_statistics(reference_aucs)
        
        label_result = {
            'label': label,
            f'{reference_model}_mean_auc': ref_stats['mean'],
            f'{reference_model}_std_auc': ref_stats['std'],
            f'{reference_model}_ci_lower': ref_stats['ci_lower'],
            f'{reference_model}_ci_upper': ref_stats['ci_upper'],
            f'{reference_model}_n_runs': ref_stats['n_runs']
        }
        
        # Compare against other models
        for model_name, model_data in all_models.items():
            if model_name == reference_model:
                continue
            
            if label in model_data['per_label_aucs']:
                other_aucs = model_data['per_label_aucs'][label]
                
                # Compute statistics for comparison model
                other_stats = compute_auc_statistics(other_aucs)
                
                label_result[f'{model_name}_mean_auc'] = other_stats['mean']
                label_result[f'{model_name}_std_auc'] = other_stats['std']
                label_result[f'{model_name}_ci_lower'] = other_stats['ci_lower']
                label_result[f'{model_name}_ci_upper'] = other_stats['ci_upper']
                label_result[f'{model_name}_n_runs'] = other_stats['n_runs']
                
                # Statistical comparison
                comparison = compare_auc_distributions(reference_aucs, other_aucs, test_type)
                
                label_result[f'{reference_model}_vs_{model_name}_diff'] = comparison['mean_diff']
                label_result[f'{reference_model}_vs_{model_name}_p_value'] = comparison['p_value']
                label_result[f'{reference_model}_vs_{model_name}_test_statistic'] = comparison['test_statistic']
        
        results.append(label_result)
    
    return pd.DataFrame(results)

def compute_overall_statistics(results_df, reference_model='sfr_mistral'):
    """Compute overall statistics across all labels."""
    if results_df is None or len(results_df) == 0:
        return None, None
    
    # Model performance summary
    model_columns = [col for col in results_df.columns if col.endswith('_mean_auc')]
    models = [col.replace('_mean_auc', '') for col in model_columns]
    
    performance_summary = []
    for model in models:
        mean_col = f'{model}_mean_auc'
        if mean_col in results_df.columns:
            mean_aucs = results_df[mean_col].dropna()
            if len(mean_aucs) > 0:
                performance_summary.append({
                    'model': model,
                    'overall_mean_auc': mean_aucs.mean(),
                    'overall_std_auc': mean_aucs.std(),
                    'n_labels': len(mean_aucs),
                    'min_auc': mean_aucs.min(),
                    'max_auc': mean_aucs.max()
                })
    
    performance_df = pd.DataFrame(performance_summary)
    
    # Comparison summary
    comparison_columns = [col for col in results_df.columns if col.endswith('_p_value')]
    
    comparison_summary = []
    for col in comparison_columns:
        comparison_name = col.replace('_p_value', '')
        diff_col = f'{comparison_name}_diff'
        
        if diff_col in results_df.columns:
            valid_comparisons = results_df[[diff_col, col]].dropna()
            
            if len(valid_comparisons) > 0:
                diffs = valid_comparisons[diff_col]
                p_values = valid_comparisons[col]
                
                wins = (diffs > 0).sum()
                sig_wins = ((diffs > 0) & (p_values < 0.05)).sum()
                sig_differences = (p_values < 0.05).sum()
                
                comparison_summary.append({
                    'comparison': comparison_name.replace('_', ' vs '),
                    'total_comparisons': len(valid_comparisons),
                    'wins': wins,
                    'win_rate': wins / len(valid_comparisons),
                    'significant_wins': sig_wins,
                    'significant_win_rate': sig_wins / len(valid_comparisons),
                    'significant_differences': sig_differences,
                    'significant_difference_rate': sig_differences / len(valid_comparisons),
                    'mean_difference': diffs.mean(),
                    'std_difference': diffs.std(),
                    'mean_p_value': p_values.mean(),
                    'median_p_value': p_values.median()
                })
    
    comparison_df = pd.DataFrame(comparison_summary)
    
    return performance_df, comparison_df

def main():
    parser = argparse.ArgumentParser(description='Linear Probing ROC-AUC Comparison Analysis')
    parser.add_argument('--models', nargs='+', 
                       choices=['sfr_mistral', 'chexzero', 'biomedclip', 'openai_clip'],
                       default=['sfr_mistral', 'chexzero', 'biomedclip', 'openai_clip'],
                       help='Models to compare')
    parser.add_argument('--test-type', choices=['ttest', 'mannwhitney', 'ranksums'],
                       default='mannwhitney',
                       help='Statistical test for comparing AUC distributions')
    args = parser.parse_args()
    
    # Create results directory
    results_dir = "results/linear_probing_auc_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Linear Probing AUC Comparison Analysis")
    print(f"Models: {', '.join([MODEL_CONFIGS[m]['display_name'] for m in args.models])}")
    print(f"Statistical test: {args.test_type}")
    
    # Filter model configs based on selected models
    selected_configs = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
    
    # Analyze performance
    results_df = analyze_linear_probing_performance(selected_configs, args.test_type)
    
    if results_df is not None:
        # Save detailed results
        detailed_results_path = f"{results_dir}/detailed_results.csv"
        results_df.to_csv(detailed_results_path, index=False, float_format='%.6f')
        print(f"\nDetailed results saved to: {detailed_results_path}")
        
        # Compute and save overall statistics
        performance_df, comparison_df = compute_overall_statistics(results_df)
        
        if performance_df is not None:
            performance_path = f"{results_dir}/model_performance_summary.csv"
            performance_df.to_csv(performance_path, index=False, float_format='%.6f')
            
            print(f"\nModel Performance Summary:")
            for _, row in performance_df.iterrows():
                model_name = MODEL_CONFIGS.get(row['model'], {}).get('display_name', row['model'])
                print(f"  {model_name}: {row['overall_mean_auc']:.4f} ± {row['overall_std_auc']:.4f} "
                      f"(n={row['n_labels']} labels)")
        
        if comparison_df is not None:
            comparison_path = f"{results_dir}/comparison_summary.csv"
            comparison_df.to_csv(comparison_path, index=False, float_format='%.6f')
            
            print(f"\nComparison Summary (using {args.test_type} test):")
            for _, row in comparison_df.iterrows():
                print(f"  {row['comparison']}: {row['wins']}/{row['total_comparisons']} wins "
                      f"({row['win_rate']:.2%}), {row['significant_wins']} significant wins "
                      f"({row['significant_win_rate']:.2%})")
                print(f"    Mean difference: {row['mean_difference']:.4f} ± {row['std_difference']:.4f}")
                print(f"    P-values: mean={row['mean_p_value']:.4f}, median={row['median_p_value']:.4f}")
        
        # Display confidence intervals for each model
        print(f"\nDetailed Performance with 95% Confidence Intervals:")
        model_columns = [col for col in results_df.columns if col.endswith('_mean_auc')]
        models = [col.replace('_mean_auc', '') for col in model_columns]
        
        for model in models:
            model_name = MODEL_CONFIGS.get(model, {}).get('display_name', model)
            mean_col = f'{model}_mean_auc'
            ci_lower_col = f'{model}_ci_lower'
            ci_upper_col = f'{model}_ci_upper'
            
            if all(col in results_df.columns for col in [mean_col, ci_lower_col, ci_upper_col]):
                mean_aucs = results_df[mean_col].dropna()
                ci_lowers = results_df[ci_lower_col].dropna()
                ci_uppers = results_df[ci_upper_col].dropna()
                
                if len(mean_aucs) > 0:
                    print(f"  {model_name}:")
                    print(f"    Overall: {mean_aucs.mean():.4f} "
                          f"[{ci_lowers.mean():.4f}, {ci_uppers.mean():.4f}]")
                    print(f"    Range: {mean_aucs.min():.4f} - {mean_aucs.max():.4f}")
        
        # Create summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total labels analyzed: {len(results_df)}")
        print(f"  Models compared: {len(selected_configs)}")
        
        if 'sfr_mistral' in selected_configs:
            sfr_cols = [col for col in results_df.columns if col.startswith('sfr_mistral_vs_') and col.endswith('_p_value')]
            if sfr_cols:
                all_p_values = []
                for col in sfr_cols:
                    p_vals = results_df[col].dropna()
                    all_p_values.extend(p_vals.tolist())
                
                if all_p_values:
                    sig_count = sum(1 for p in all_p_values if p < 0.05)
                    print(f"  Significant differences (p<0.05): {sig_count}/{len(all_p_values)} "
                          f"({sig_count/len(all_p_values):.2%})")
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()