#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_thresholds_data():
    """Load optimal thresholds from CSV file"""
    print("Loading thresholds data...")
    
    thresholds_file = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck/metrics/cbm_experiments_thresholds.csv"
    df = pd.read_csv(thresholds_file)
    
    # Filter for CheXpert dataset only
    df = df[df['Dataset'] == 'chexpert'].copy()
    
    # Organize thresholds by method and seed
    thresholds_dict = {}
    for _, row in df.iterrows():
        method = row['Method']
        seed = row['Seed']
        label = row['Label']
        threshold = row['Threshold']
        
        if method not in thresholds_dict:
            thresholds_dict[method] = {}
        if seed not in thresholds_dict[method]:
            thresholds_dict[method][seed] = {}
        
        thresholds_dict[method][seed][label] = threshold
    
    print(f"Loaded thresholds for methods: {list(thresholds_dict.keys())}")
    return thresholds_dict

def load_model_predictions():
    """Load model predictions from all seed files"""
    print("Loading model predictions...")
    
    predictions_dir = Path("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck/predictions")
    
    model_predictions = {}
    seeds = list(range(42, 62))  # Seeds 42-61
    
    for seed in seeds:
        seed_file = predictions_dir / f"seed_{seed}_predictions.pkl"
        if seed_file.exists():
            with open(seed_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract CheXpert predictions for Standard and Improved CBM
            if 'standard' in data and 'chexpert' in data['standard']:
                if 'Standard' not in model_predictions:
                    model_predictions['Standard'] = {}
                model_predictions['Standard'][seed] = {
                    'y_true': data['standard']['chexpert']['y_true'],
                    'y_pred': data['standard']['chexpert']['y_pred'],
                    'labels': data['standard']['chexpert']['labels']
                }
            
            if 'improved' in data and 'chexpert' in data['improved']:
                if 'Improved' not in model_predictions:
                    model_predictions['Improved'] = {}
                model_predictions['Improved'][seed] = {
                    'y_true': data['improved']['chexpert']['y_true'],
                    'y_pred': data['improved']['chexpert']['y_pred'],
                    'labels': data['improved']['chexpert']['labels']
                }
        else:
            print(f"Warning: {seed_file} not found")
    
    # Also load zero-shot predictions
    zeroshot_file = predictions_dir / "zeroshot_predictions.pkl"
    if zeroshot_file.exists():
        with open(zeroshot_file, 'rb') as f:
            zeroshot_data = pickle.load(f)
        if 'chexpert' in zeroshot_data:
            model_predictions['ZeroShot'] = {
                'NA': {  # Zero-shot doesn't have seeds
                    'y_true': zeroshot_data['chexpert']['y_true'],
                    'y_pred': zeroshot_data['chexpert']['y_pred'],
                    'labels': zeroshot_data['chexpert']['labels']
                }
            }
    
    print(f"Loaded predictions for methods: {list(model_predictions.keys())}")
    for method in model_predictions:
        print(f"  {method}: {len(model_predictions[method])} seeds/runs")
    
    return model_predictions

def load_human_annotations():
    """Load human radiologist annotations"""
    print("Loading human radiologist annotations...")
    
    radiologist_dir = Path("/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_radiologist")
    
    human_annotations = {}
    radiologist_files = ['bc4.csv', 'bc6.csv', 'bc8.csv']
    
    for i, filename in enumerate(radiologist_files):
        radiologist_id = f"Radiologist_{i+1}"
        file_path = radiologist_dir / filename
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            human_annotations[radiologist_id] = df
            print(f"  {radiologist_id}: {len(df)} annotations")
        else:
            print(f"Warning: {file_path} not found")
    
    return human_annotations

def load_ground_truth():
    """Load CheXpert test ground truth labels"""
    print("Loading ground truth labels...")
    
    gt_file = "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.csv"
    df = pd.read_csv(gt_file)
    
    print(f"Ground truth: {len(df)} samples")
    return df

def align_data_by_study_id(model_predictions, human_annotations, ground_truth, target_labels):
    """Align all data sources by study ID"""
    print("Aligning data by study ID...")
    
    # Get study IDs from ground truth
    gt_studies = set(ground_truth['Study'].values)
    print(f"Ground truth studies: {len(gt_studies)}")
    
    # Get study IDs from human annotations (assuming they're consistent across radiologists)
    human_studies = set()
    for radiologist_id, df in human_annotations.items():
        studies = set(df['Study'].values)
        human_studies.update(studies)
        print(f"{radiologist_id} studies: {len(studies)}")
    
    # Find common studies
    common_studies = gt_studies.intersection(human_studies)
    print(f"Common studies between GT and humans: {len(common_studies)}")
    
    # Create aligned datasets
    aligned_data = {
        'studies': sorted(list(common_studies)),
        'ground_truth': {},
        'human_annotations': {},
        'model_predictions': model_predictions  # Keep full model predictions for now
    }
    
    # Align ground truth
    gt_filtered = ground_truth[ground_truth['Study'].isin(common_studies)].copy()
    gt_filtered = gt_filtered.sort_values('Study').reset_index(drop=True)
    
    for label in target_labels:
        if label in gt_filtered.columns:
            aligned_data['ground_truth'][label] = gt_filtered[label].values
        else:
            print(f"Warning: {label} not found in ground truth")
            aligned_data['ground_truth'][label] = np.zeros(len(gt_filtered))
    
    # Align human annotations
    for radiologist_id, df in human_annotations.items():
        human_filtered = df[df['Study'].isin(common_studies)].copy()
        human_filtered = human_filtered.sort_values('Study').reset_index(drop=True)
        
        aligned_data['human_annotations'][radiologist_id] = {}
        for label in target_labels:
            if label in human_filtered.columns:
                aligned_data['human_annotations'][radiologist_id][label] = human_filtered[label].values
            else:
                print(f"Warning: {label} not found in {radiologist_id} annotations")
                aligned_data['human_annotations'][radiologist_id][label] = np.zeros(len(human_filtered))
    
    print(f"Aligned data: {len(aligned_data['studies'])} common studies")
    return aligned_data

def compute_mcc_score(y_true, y_pred):
    """Compute Matthews Correlation Coefficient with error handling"""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        mcc = matthews_corrcoef(y_true, y_pred)
        return mcc if not np.isnan(mcc) else 0.0
    except:
        return 0.0

def compute_f1_score(y_true, y_pred):
    """Compute F1 score with error handling"""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        f1 = f1_score(y_true, y_pred)
        return f1 if not np.isnan(f1) else 0.0
    except:
        return 0.0

def apply_threshold_and_compute_metrics(y_pred_prob, y_true, threshold):
    """Apply threshold to predictions and compute MCC and F1"""
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    mcc = compute_mcc_score(y_true, y_pred_binary)
    f1 = compute_f1_score(y_true, y_pred_binary)
    
    return mcc, f1

def permutation_test(group1_scores, group2_scores, n_permutations=10000, random_state=42):
    """
    Perform permutation test to compare two groups of scores
    
    Args:
        group1_scores: Array of scores for group 1 (e.g., model scores)
        group2_scores: Array of scores for group 2 (e.g., human scores)
        n_permutations: Number of permutations to perform
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with test results
    """
    np.random.seed(random_state)
    
    # Convert to numpy arrays
    group1 = np.array(group1_scores)
    group2 = np.array(group2_scores)
    
    # Observed difference in means
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Combine all scores
    all_scores = np.concatenate([group1, group2])
    n_group1 = len(group1)
    n_total = len(all_scores)
    
    # Perform permutations
    permuted_diffs = []
    for _ in range(n_permutations):
        # Randomly shuffle all scores
        shuffled_scores = np.random.permutation(all_scores)
        
        # Split into two groups
        perm_group1 = shuffled_scores[:n_group1]
        perm_group2 = shuffled_scores[n_group1:]
        
        # Calculate difference in means
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        permuted_diffs.append(perm_diff)
    
    permuted_diffs = np.array(permuted_diffs)
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    # Calculate confidence interval for the observed difference
    ci_lower = np.percentile(permuted_diffs, 2.5)
    ci_upper = np.percentile(permuted_diffs, 97.5)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0
    
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'group1_mean': np.mean(group1),
        'group1_std': np.std(group1, ddof=1),
        'group2_mean': np.mean(group2),
        'group2_std': np.std(group2, ddof=1),
        'n_permutations': n_permutations
    }

def compute_model_performance(model_predictions, thresholds_dict, aligned_data, target_labels):
    """Compute MCC and F1 scores for all model methods and seeds"""
    print("Computing model performance...")
    
    model_results = {}
    
    for method in ['Standard', 'Improved']:
        if method not in model_predictions:
            continue
            
        print(f"  Processing {method} CBM...")
        model_results[method] = {}
        
        for seed in model_predictions[method]:
            pred_data = model_predictions[method][seed]
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            model_labels = pred_data['labels']
            
            # Get thresholds for this method and seed
            if method in thresholds_dict and seed in thresholds_dict[method]:
                thresholds = thresholds_dict[method][seed]
            else:
                print(f"    Warning: No thresholds found for {method} seed {seed}")
                continue
            
            model_results[method][seed] = {}
            
            for i, label in enumerate(target_labels):
                if label not in model_labels:
                    print(f"    Warning: {label} not in model labels")
                    continue
                
                model_label_idx = model_labels.index(label)
                
                if label in thresholds:
                    threshold = thresholds[label]
                    mcc, f1 = apply_threshold_and_compute_metrics(
                        y_pred[:, model_label_idx], 
                        y_true[:, model_label_idx], 
                        threshold
                    )
                    
                    model_results[method][seed][label] = {
                        'mcc': mcc,
                        'f1': f1,
                        'threshold': threshold
                    }
                else:
                    print(f"    Warning: No threshold found for {label}")
    
    # Handle Zero-shot separately (no thresholds needed, use 0.5)
    if 'ZeroShot' in model_predictions:
        print("  Processing Zero-Shot CBM...")
        model_results['ZeroShot'] = {}
        
        pred_data = model_predictions['ZeroShot']['NA']
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        model_labels = pred_data['labels']
        
        model_results['ZeroShot']['NA'] = {}
        
        for i, label in enumerate(target_labels):
            if label not in model_labels:
                continue
                
            model_label_idx = model_labels.index(label)
            
            # Use threshold from ZeroShot in thresholds_dict if available
            threshold = 0.5  # Default
            if 'ZeroShot' in thresholds_dict and 'NA' in thresholds_dict['ZeroShot']:
                if label in thresholds_dict['ZeroShot']['NA']:
                    threshold = thresholds_dict['ZeroShot']['NA'][label]
            
            mcc, f1 = apply_threshold_and_compute_metrics(
                y_pred[:, model_label_idx], 
                y_true[:, model_label_idx], 
                threshold
            )
            
            model_results['ZeroShot']['NA'][label] = {
                'mcc': mcc,
                'f1': f1,
                'threshold': threshold
            }
    
    return model_results

def compute_human_performance(aligned_data, target_labels):
    """Compute MCC and F1 scores for human radiologists"""
    print("Computing human performance...")
    
    human_results = {}
    
    for radiologist_id in aligned_data['human_annotations']:
        print(f"  Processing {radiologist_id}...")
        human_results[radiologist_id] = {}
        
        for label in target_labels:
            y_true = aligned_data['ground_truth'][label]
            y_pred = aligned_data['human_annotations'][radiologist_id][label]
            
            mcc = compute_mcc_score(y_true, y_pred)
            f1 = compute_f1_score(y_true, y_pred)
            
            human_results[radiologist_id][label] = {
                'mcc': mcc,
                'f1': f1
            }
    
    return human_results

def perform_statistical_comparisons(model_results, human_results, target_labels):
    """Perform permutation tests comparing models vs humans"""
    print("Performing statistical comparisons...")
    
    comparison_results = {}
    
    # For each model method
    for method in ['Standard', 'Improved']:
        if method not in model_results:
            continue
            
        print(f"  Comparing {method} CBM vs Humans...")
        comparison_results[method] = {}
        
        for label in target_labels:
            # Collect model scores across all seeds
            model_mcc_scores = []
            model_f1_scores = []
            
            for seed in model_results[method]:
                if label in model_results[method][seed]:
                    model_mcc_scores.append(model_results[method][seed][label]['mcc'])
                    model_f1_scores.append(model_results[method][seed][label]['f1'])
            
            # Collect human scores
            human_mcc_scores = []
            human_f1_scores = []
            
            for radiologist_id in human_results:
                if label in human_results[radiologist_id]:
                    human_mcc_scores.append(human_results[radiologist_id][label]['mcc'])
                    human_f1_scores.append(human_results[radiologist_id][label]['f1'])
            
            if len(model_mcc_scores) > 0 and len(human_mcc_scores) > 0:
                # Perform permutation tests
                mcc_test = permutation_test(model_mcc_scores, human_mcc_scores)
                f1_test = permutation_test(model_f1_scores, human_f1_scores)
                
                comparison_results[method][label] = {
                    'mcc_test': mcc_test,
                    'f1_test': f1_test,
                    'model_mcc_scores': model_mcc_scores,
                    'model_f1_scores': model_f1_scores,
                    'human_mcc_scores': human_mcc_scores,
                    'human_f1_scores': human_f1_scores
                }
                
                print(f"    {label} - MCC p-value: {mcc_test['p_value']:.4f}, F1 p-value: {f1_test['p_value']:.4f}")
    
    # Handle Zero-shot separately (single run, no seeds)
    if 'ZeroShot' in model_results:
        print("  Comparing Zero-Shot CBM vs Humans...")
        comparison_results['ZeroShot'] = {}
        
        for label in target_labels:
            if label in model_results['ZeroShot']['NA']:
                model_mcc = model_results['ZeroShot']['NA'][label]['mcc']
                model_f1 = model_results['ZeroShot']['NA'][label]['f1']
                
                # Collect human scores
                human_mcc_scores = []
                human_f1_scores = []
                
                for radiologist_id in human_results:
                    if label in human_results[radiologist_id]:
                        human_mcc_scores.append(human_results[radiologist_id][label]['mcc'])
                        human_f1_scores.append(human_results[radiologist_id][label]['f1'])
                
                if len(human_mcc_scores) > 0:
                    # For zero-shot, we can't do permutation test with single value
                    # Instead, compute descriptive statistics
                    comparison_results['ZeroShot'][label] = {
                        'model_mcc': model_mcc,
                        'model_f1': model_f1,
                        'human_mcc_scores': human_mcc_scores,
                        'human_f1_scores': human_f1_scores,
                        'human_mcc_mean': np.mean(human_mcc_scores),
                        'human_mcc_std': np.std(human_mcc_scores, ddof=1),
                        'human_f1_mean': np.mean(human_f1_scores),
                        'human_f1_std': np.std(human_f1_scores, ddof=1)
                    }
    
    return comparison_results

def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval using t-distribution for small samples"""
    n = len(values)
    if n <= 1:
        return np.nan, np.nan
    
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    
    # Use t-distribution for small samples
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci_margin = t_val * sem
    
    return mean - ci_margin, mean + ci_margin

def save_results(model_results, human_results, comparison_results, target_labels, output_dir):
    """Save all results to CSV files"""
    print("Saving results...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model performance across seeds
    model_performance_data = []
    for method in model_results:
        for seed in model_results[method]:
            for label in target_labels:
                if label in model_results[method][seed]:
                    result = model_results[method][seed][label]
                    model_performance_data.append({
                        'Method': method,
                        'Seed': seed,
                        'Label': label,
                        'MCC': result['mcc'],
                        'F1': result['f1'],
                        'Threshold': result['threshold']
                    })
    
    model_df = pd.DataFrame(model_performance_data)
    model_df.to_csv(output_dir / "model_performance_by_seed.csv", index=False)
    
    # 2. Save human performance
    human_performance_data = []
    for radiologist_id in human_results:
        for label in target_labels:
            if label in human_results[radiologist_id]:
                result = human_results[radiologist_id][label]
                human_performance_data.append({
                    'Radiologist': radiologist_id,
                    'Label': label,
                    'MCC': result['mcc'],
                    'F1': result['f1']
                })
    
    human_df = pd.DataFrame(human_performance_data)
    human_df.to_csv(output_dir / "human_performance.csv", index=False)
    
    # 3. Save statistical comparison results with confidence intervals
    comparison_data = []
    for method in comparison_results:
        for label in comparison_results[method]:
            result = comparison_results[method][label]
            
            if method == 'ZeroShot':
                # Zero-shot has different structure
                human_mcc_ci_lower, human_mcc_ci_upper = compute_confidence_interval(result['human_mcc_scores'])
                human_f1_ci_lower, human_f1_ci_upper = compute_confidence_interval(result['human_f1_scores'])
                
                comparison_data.append({
                    'Method': method,
                    'Label': label,
                    'Model_MCC_Mean': result['model_mcc'],
                    'Model_MCC_Std': 0.0,  # Single value
                    'Model_MCC_CI_Lower': result['model_mcc'],  # Single value
                    'Model_MCC_CI_Upper': result['model_mcc'],  # Single value
                    'Model_F1_Mean': result['model_f1'],
                    'Model_F1_Std': 0.0,   # Single value
                    'Model_F1_CI_Lower': result['model_f1'],   # Single value
                    'Model_F1_CI_Upper': result['model_f1'],   # Single value
                    'Human_MCC_Mean': result['human_mcc_mean'],
                    'Human_MCC_Std': result['human_mcc_std'],
                    'Human_MCC_CI_Lower': human_mcc_ci_lower,
                    'Human_MCC_CI_Upper': human_mcc_ci_upper,
                    'Human_F1_Mean': result['human_f1_mean'],
                    'Human_F1_Std': result['human_f1_std'],
                    'Human_F1_CI_Lower': human_f1_ci_lower,
                    'Human_F1_CI_Upper': human_f1_ci_upper,
                    'MCC_P_Value': 'N/A',  # Can't compute with single model value
                    'F1_P_Value': 'N/A',
                    'MCC_Effect_Size': 'N/A',
                    'F1_Effect_Size': 'N/A'
                })
            else:
                # Standard and Improved CBM with permutation tests
                model_mcc_ci_lower, model_mcc_ci_upper = compute_confidence_interval(result['model_mcc_scores'])
                model_f1_ci_lower, model_f1_ci_upper = compute_confidence_interval(result['model_f1_scores'])
                human_mcc_ci_lower, human_mcc_ci_upper = compute_confidence_interval(result['human_mcc_scores'])
                human_f1_ci_lower, human_f1_ci_upper = compute_confidence_interval(result['human_f1_scores'])
                
                comparison_data.append({
                    'Method': method,
                    'Label': label,
                    'Model_MCC_Mean': result['mcc_test']['group1_mean'],
                    'Model_MCC_Std': result['mcc_test']['group1_std'],
                    'Model_MCC_CI_Lower': model_mcc_ci_lower,
                    'Model_MCC_CI_Upper': model_mcc_ci_upper,
                    'Model_F1_Mean': result['f1_test']['group1_mean'],
                    'Model_F1_Std': result['f1_test']['group1_std'],
                    'Model_F1_CI_Lower': model_f1_ci_lower,
                    'Model_F1_CI_Upper': model_f1_ci_upper,
                    'Human_MCC_Mean': result['mcc_test']['group2_mean'],
                    'Human_MCC_Std': result['mcc_test']['group2_std'],
                    'Human_MCC_CI_Lower': human_mcc_ci_lower,
                    'Human_MCC_CI_Upper': human_mcc_ci_upper,
                    'Human_F1_Mean': result['f1_test']['group2_mean'],
                    'Human_F1_Std': result['f1_test']['group2_std'],
                    'Human_F1_CI_Lower': human_f1_ci_lower,
                    'Human_F1_CI_Upper': human_f1_ci_upper,
                    'MCC_P_Value': result['mcc_test']['p_value'],
                    'F1_P_Value': result['f1_test']['p_value'],
                    'MCC_Effect_Size': result['mcc_test']['cohens_d'],
                    'F1_Effect_Size': result['f1_test']['cohens_d']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "statistical_comparisons.csv", index=False)
    
    # 4. Save summary statistics
    summary_data = []
    for method in ['Standard', 'Improved']:
        if method not in model_results:
            continue
            
        for label in target_labels:
            # Collect all scores for this method and label
            mcc_scores = []
            f1_scores = []
            
            for seed in model_results[method]:
                if label in model_results[method][seed]:
                    mcc_scores.append(model_results[method][seed][label]['mcc'])
                    f1_scores.append(model_results[method][seed][label]['f1'])
            
            if len(mcc_scores) > 0:
                summary_data.append({
                    'Method': method,
                    'Label': label,
                    'Metric': 'MCC',
                    'Mean': np.mean(mcc_scores),
                    'Std': np.std(mcc_scores, ddof=1),
                    'Min': np.min(mcc_scores),
                    'Max': np.max(mcc_scores),
                    'N_Seeds': len(mcc_scores)
                })
                
                summary_data.append({
                    'Method': method,
                    'Label': label,
                    'Metric': 'F1',
                    'Mean': np.mean(f1_scores),
                    'Std': np.std(f1_scores, ddof=1),
                    'Min': np.min(f1_scores),
                    'Max': np.max(f1_scores),
                    'N_Seeds': len(f1_scores)
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "model_summary_statistics.csv", index=False)
    
    # 5. Save average performance across all 5 labels
    average_performance_data = []
    
    # Calculate average performance for each method
    for method in ['Standard', 'Improved', 'ZeroShot']:
        if method not in model_results:
            continue
            
        all_mcc_scores = []
        all_f1_scores = []
        
        if method == 'ZeroShot':
            # Zero-shot: single values for each label
            for label in target_labels:
                if label in model_results[method]['NA']:
                    all_mcc_scores.append(model_results[method]['NA'][label]['mcc'])
                    all_f1_scores.append(model_results[method]['NA'][label]['f1'])
            
            if all_mcc_scores and all_f1_scores:
                average_performance_data.append({
                    'Method': method,
                    'MCC_Mean': np.mean(all_mcc_scores),
                    'MCC_Std': np.std(all_mcc_scores, ddof=1) if len(all_mcc_scores) > 1 else 0.0,
                    'MCC_CI_Lower': np.mean(all_mcc_scores),  # Single average value
                    'MCC_CI_Upper': np.mean(all_mcc_scores),  # Single average value
                    'F1_Mean': np.mean(all_f1_scores),
                    'F1_Std': np.std(all_f1_scores, ddof=1) if len(all_f1_scores) > 1 else 0.0,
                    'F1_CI_Lower': np.mean(all_f1_scores),    # Single average value
                    'F1_CI_Upper': np.mean(all_f1_scores),    # Single average value
                    'N_Seeds': 1,
                    'N_Labels': len(all_mcc_scores)
                })
        else:
            # Standard and Improved: multiple seeds
            seed_avg_mcc = []  # Average MCC across 5 labels for each seed
            seed_avg_f1 = []   # Average F1 across 5 labels for each seed
            
            for seed in model_results[method]:
                seed_mcc_scores = []
                seed_f1_scores = []
                
                for label in target_labels:
                    if label in model_results[method][seed]:
                        seed_mcc_scores.append(model_results[method][seed][label]['mcc'])
                        seed_f1_scores.append(model_results[method][seed][label]['f1'])
                
                if seed_mcc_scores and seed_f1_scores:
                    seed_avg_mcc.append(np.mean(seed_mcc_scores))
                    seed_avg_f1.append(np.mean(seed_f1_scores))
            
            if seed_avg_mcc and seed_avg_f1:
                mcc_ci_lower, mcc_ci_upper = compute_confidence_interval(seed_avg_mcc)
                f1_ci_lower, f1_ci_upper = compute_confidence_interval(seed_avg_f1)
                
                average_performance_data.append({
                    'Method': method,
                    'MCC_Mean': np.mean(seed_avg_mcc),
                    'MCC_Std': np.std(seed_avg_mcc, ddof=1),
                    'MCC_CI_Lower': mcc_ci_lower,
                    'MCC_CI_Upper': mcc_ci_upper,
                    'F1_Mean': np.mean(seed_avg_f1),
                    'F1_Std': np.std(seed_avg_f1, ddof=1),
                    'F1_CI_Lower': f1_ci_lower,
                    'F1_CI_Upper': f1_ci_upper,
                    'N_Seeds': len(seed_avg_mcc),
                    'N_Labels': len(target_labels)
                })
    
    # Calculate average human performance across all 5 labels
    human_mcc_by_radiologist = []  # Average MCC across 5 labels for each radiologist
    human_f1_by_radiologist = []   # Average F1 across 5 labels for each radiologist
    
    for radiologist_id in human_results:
        radiologist_mcc = []
        radiologist_f1 = []
        
        for label in target_labels:
            if label in human_results[radiologist_id]:
                radiologist_mcc.append(human_results[radiologist_id][label]['mcc'])
                radiologist_f1.append(human_results[radiologist_id][label]['f1'])
        
        if radiologist_mcc and radiologist_f1:
            human_mcc_by_radiologist.append(np.mean(radiologist_mcc))
            human_f1_by_radiologist.append(np.mean(radiologist_f1))
    
    if human_mcc_by_radiologist and human_f1_by_radiologist:
        human_mcc_ci_lower, human_mcc_ci_upper = compute_confidence_interval(human_mcc_by_radiologist)
        human_f1_ci_lower, human_f1_ci_upper = compute_confidence_interval(human_f1_by_radiologist)
        
        average_performance_data.append({
            'Method': 'Human',
            'MCC_Mean': np.mean(human_mcc_by_radiologist),
            'MCC_Std': np.std(human_mcc_by_radiologist, ddof=1),
            'MCC_CI_Lower': human_mcc_ci_lower,
            'MCC_CI_Upper': human_mcc_ci_upper,
            'F1_Mean': np.mean(human_f1_by_radiologist),
            'F1_Std': np.std(human_f1_by_radiologist, ddof=1),
            'F1_CI_Lower': human_f1_ci_lower,
            'F1_CI_Upper': human_f1_ci_upper,
            'N_Seeds': len(human_mcc_by_radiologist),
            'N_Labels': len(target_labels)
        })
    
    average_performance_df = pd.DataFrame(average_performance_data)
    average_performance_df.to_csv(output_dir / "average_performance_across_labels.csv", index=False)
    
    print(f"Results saved to: {output_dir}")
    print(f"  - model_performance_by_seed.csv: Individual seed results")
    print(f"  - human_performance.csv: Radiologist performance")
    print(f"  - statistical_comparisons.csv: Permutation test results with 95% CI")
    print(f"  - model_summary_statistics.csv: Summary statistics across seeds")
    print(f"  - average_performance_across_labels.csv: Average performance across all 5 labels with 95% CI")

def main():
    """Main function to run the complete analysis"""
    print("=== Comparing Model vs Human Performance ===")
    
    # Target labels (5 pathologies)
    target_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    print(f"Target labels: {target_labels}")
    
    # Output directory
    output_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/human_comparison"
    
    # Load all data
    thresholds_dict = load_thresholds_data()
    model_predictions = load_model_predictions()
    human_annotations = load_human_annotations()
    ground_truth = load_ground_truth()
    
    # Align data by study ID
    aligned_data = align_data_by_study_id(model_predictions, human_annotations, ground_truth, target_labels)
    
    # Compute performance metrics
    model_results = compute_model_performance(model_predictions, thresholds_dict, aligned_data, target_labels)
    human_results = compute_human_performance(aligned_data, target_labels)
    
    # Perform statistical comparisons
    comparison_results = perform_statistical_comparisons(model_results, human_results, target_labels)
    
    # Save results
    save_results(model_results, human_results, comparison_results, target_labels, output_dir)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for method in ['Standard', 'Improved']:
        if method not in comparison_results:
            continue
            
        print(f"\n{method} CBM vs Human Radiologists:")
        for label in target_labels:
            if label in comparison_results[method]:
                result = comparison_results[method][label]
                mcc_p = result['mcc_test']['p_value']
                f1_p = result['f1_test']['p_value']
                
                print(f"  {label}:")
                print(f"    MCC - Model: {result['mcc_test']['group1_mean']:.4f}±{result['mcc_test']['group1_std']:.4f}, "
                      f"Human: {result['mcc_test']['group2_mean']:.4f}±{result['mcc_test']['group2_std']:.4f}, "
                      f"p={mcc_p:.4f}")
                print(f"    F1  - Model: {result['f1_test']['group1_mean']:.4f}±{result['f1_test']['group1_std']:.4f}, "
                      f"Human: {result['f1_test']['group2_mean']:.4f}±{result['f1_test']['group2_std']:.4f}, "
                      f"p={f1_p:.4f}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()