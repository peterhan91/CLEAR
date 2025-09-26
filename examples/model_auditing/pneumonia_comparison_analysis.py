#!/usr/bin/env python3
"""
Pneumonia Performance Comparison Analysis between VinDr-CXR and PadChest datasets
for SFR-Mistral concept model.

SIMPLIFIED METHODOLOGY:
- Focus only on ROC-AUC (threshold-independent metric)
- Compare performance with bootstrapping CI and p-values
- Store all bootstrap values for error bar computation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def bootstrap_auc(y_true, y_pred, n_bootstrap=1000):
    """Bootstrap confidence intervals for AUC"""
    n_samples = len(y_true)
    
    auc_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if all labels are the same
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        # Compute AUC
        auc = roc_auc_score(y_true_boot, y_pred_boot)
        auc_scores.append(auc)
    
    # Compute confidence intervals
    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    
    return {
        'auc_mean': np.mean(auc_scores),
        'auc_ci': auc_ci,
        'auc_scores': auc_scores  # Store all bootstrap values
    }

def permutation_test(group1_scores, group2_scores, n_permutations=10000):
    """Permutation test to compare two groups of AUC scores"""
    observed_diff = np.mean(group1_scores) - np.mean(group2_scores)
    
    combined = np.concatenate([group1_scores, group2_scores])
    n1 = len(group1_scores)
    
    np.random.seed(42)
    permuted_diffs = []
    
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        permuted_diff = np.mean(perm_group1) - np.mean(perm_group2)
        permuted_diffs.append(permuted_diff)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return p_value, observed_diff

def load_predictions_from_results(results_dir, label_name):
    """Load predictions and ground truth from concept evaluation results"""
    results_path = Path(results_dir)
    
    # Find the most recent prediction files
    pred_files = list(results_path.glob("predictions_*.csv"))
    gt_files = list(results_path.glob("ground_truth_*.csv"))
    
    if not pred_files or not gt_files:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")
    
    # Use the most recent files
    pred_file = max(pred_files, key=lambda p: p.stat().st_mtime)
    gt_file = max(gt_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading predictions from: {pred_file.name}")
    print(f"Loading ground truth from: {gt_file.name}")
    
    # Load data
    predictions_df = pd.read_csv(pred_file)
    gt_df = pd.read_csv(gt_file)
    
    # Find the correct column name for pneumonia
    pred_col = None
    gt_col = None
    
    # Try different possible column names for pneumonia
    possible_names = [label_name, label_name.lower(), label_name.capitalize(), 
                     f"{label_name}_pred", f"{label_name.lower()}_pred",
                     f"{label_name}_true", f"{label_name.lower()}_true"]
    
    for col in predictions_df.columns:
        for name in possible_names:
            if name in col.lower():
                pred_col = col
                break
        if pred_col:
            break
    
    for col in gt_df.columns:
        for name in possible_names:
            if name in col.lower():
                gt_col = col
                break
        if gt_col:
            break
    
    if pred_col is None or gt_col is None:
        available_pred_cols = list(predictions_df.columns)
        available_gt_cols = list(gt_df.columns)
        raise ValueError(f"Could not find {label_name} columns.\n"
                        f"Available prediction columns: {available_pred_cols}\n"
                        f"Available ground truth columns: {available_gt_cols}")
    
    print(f"Using prediction column: {pred_col}")
    print(f"Using ground truth column: {gt_col}")
    
    y_pred = predictions_df[pred_col].values
    y_true = gt_df[gt_col].values
    
    return y_true, y_pred

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("🏥 PNEUMONIA AUC COMPARISON ANALYSIS")
    print("Simplified methodology: ROC-AUC only (threshold-independent)")
    print("="*60)
    
    # Configuration
    target_label = "Pneumonia"
    dataset_configs = {
        'vindrcxr': {
            'results_dir': '../concepts/results/concept_based_evaluation_vindrcxr_sfr_mistral',
            'label': 'Pneumonia',
            'display_name': 'VinDr-CXR'
        },
        'padchest': {
            'results_dir': '../concepts/results/concept_based_evaluation_padchest_sfr_mistral',
            'label': 'pneumonia',  # lowercase in padchest
            'display_name': 'PadChest'
        }
    }
    
    # Evaluate test datasets
    print(f"\n{'-'*50}")
    print("EVALUATING TEST DATASETS")
    print(f"{'-'*50}")
    
    results = {}
    all_results = []
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📊 Processing {config['display_name']}...")
        
        try:
            # Load test predictions
            y_true_test, y_pred_test = load_predictions_from_results(
                config['results_dir'], config['label']
            )
            
            print(f"✅ Loaded {len(y_true_test)} test samples")
            print(f"   Pneumonia prevalence: {np.mean(y_true_test):.3f}")
            
            # Compute AUC
            auc = roc_auc_score(y_true_test, y_pred_test)
            
            # Bootstrap confidence intervals
            bootstrap_results = bootstrap_auc(y_true_test, y_pred_test)
            
            # Store results
            result = {
                'Dataset': config['display_name'],
                'Samples': len(y_true_test),
                'Prevalence': np.mean(y_true_test),
                'AUROC': auc,
                'AUROC_CI_Lower': bootstrap_results['auc_ci'][0],
                'AUROC_CI_Upper': bootstrap_results['auc_ci'][1],
                'AUROC_Bootstrap_Mean': bootstrap_results['auc_mean'],
            }
            
            results[dataset_name] = {
                'metrics': result,
                'bootstrap_auc_scores': bootstrap_results['auc_scores']  # Store all bootstrap values
            }
            
            all_results.append(result)
            
            print(f"   📈 AUROC: {auc:.4f} [{bootstrap_results['auc_ci'][0]:.4f}, {bootstrap_results['auc_ci'][1]:.4f}]")
            print(f"   📊 Bootstrap samples: {len(bootstrap_results['auc_scores'])}")
            
        except Exception as e:
            print(f"❌ Error processing {config['display_name']}: {e}")
            continue
    
    # Statistical comparison
    print(f"\n{'-'*50}")
    print("STATISTICAL COMPARISON")
    print(f"{'-'*50}")
    
    if len(results) == 2:
        dataset_names = list(results.keys())
        dataset1, dataset2 = dataset_names
        
        scores1 = results[dataset1]['bootstrap_auc_scores']
        scores2 = results[dataset2]['bootstrap_auc_scores']
        
        p_value, observed_diff = permutation_test(scores1, scores2)
        
        print(f"AUC: {dataset_configs[dataset1]['display_name']} vs {dataset_configs[dataset2]['display_name']}")
        print(f"   Difference: {observed_diff:.4f}")
        print(f"   P-value: {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
    
    # Save results
    print(f"\n{'-'*50}")
    print("SAVING RESULTS")
    print(f"{'-'*50}")
    
    if all_results:
        # Save summary results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{results_dir}/pneumonia_auc_comparison.csv", index=False)
        
        # Save bootstrap values for error bars
        bootstrap_data = {}
        for dataset_name, result in results.items():
            bootstrap_data[dataset_name] = {
                'dataset': dataset_configs[dataset_name]['display_name'],
                'auc_scores': result['bootstrap_auc_scores']
            }
        
        # Save bootstrap values as numpy arrays for easy loading
        np.savez(f"{results_dir}/pneumonia_auc_bootstrap_values.npz", 
                vindrcxr_auc=results['vindrcxr']['bootstrap_auc_scores'] if 'vindrcxr' in results else [],
                padchest_auc=results['padchest']['bootstrap_auc_scores'] if 'padchest' in results else [])
        
        # Create summary text
        summary_text = f"""Pneumonia AUC Comparison Analysis
=====================================

Methodology: 
- ROC-AUC only (threshold-independent metric)
- 95% CI using 1000 bootstrap samples
- Statistical comparison: Permutation test (10,000 permutations)

Results:
"""
        
        for result in all_results:
            summary_text += f"""
{result['Dataset']} (n={result['Samples']:,}, prevalence={result['Prevalence']:.3f}):
  AUROC: {result['AUROC']:.4f} [{result['AUROC_CI_Lower']:.4f}, {result['AUROC_CI_Upper']:.4f}]
"""
        
        if len(results) == 2:
            summary_text += f"""
Statistical Comparison (Permutation Test):
  AUC difference: {observed_diff:.4f}
  P-value: {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})
"""
        
        summary_text += f"""
Bootstrap samples: 1000 per dataset
Files saved:
- pneumonia_auc_comparison.csv: Summary results
- pneumonia_auc_bootstrap_values.npz: All bootstrap AUC values for error bars
"""
        
        with open(f"{results_dir}/pneumonia_auc_summary.txt", 'w') as f:
            f.write(summary_text)
        
        print(f"✅ Results saved:")
        print(f"   📊 Summary: {results_dir}/pneumonia_auc_comparison.csv")
        print(f"   📈 Bootstrap values: {results_dir}/pneumonia_auc_bootstrap_values.npz")
        print(f"   📋 Summary: {results_dir}/pneumonia_auc_summary.txt")
        
        # Print how to load bootstrap values
        print(f"\n💡 To load bootstrap values for plotting:")
        print(f"   bootstrap_data = np.load('{results_dir}/pneumonia_auc_bootstrap_values.npz')")
        print(f"   vindrcxr_aucs = bootstrap_data['vindrcxr_auc']")
        print(f"   padchest_aucs = bootstrap_data['padchest_auc']")
        
    else:
        print("❌ No results to save - check that prediction files exist")
    
    print(f"\n🎯 Analysis complete!")

if __name__ == "__main__":
    main() 