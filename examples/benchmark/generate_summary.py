#!/usr/bin/env python3
"""
Generate benchmark summary from individual result files.
"""

import os
import sys
import argparse
import pandas as pd
import glob
from pathlib import Path

def collect_results(results_dir):
    """Collect all benchmark results from CSV files."""
    
    # Find all result CSV files
    pattern = os.path.join(results_dir, "*_results.csv")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return None
    
    # Parse results
    all_results = []
    
    for file_path in result_files:
        filename = os.path.basename(file_path)
        
        # Parse filename: {model}_{dataset}_results.csv
        parts = filename.replace('_results.csv', '').split('_')
        if len(parts) < 2:
            print(f"Warning: Could not parse filename {filename}")
            continue
        
        # Handle model names with underscores
        if len(parts) > 2:
            model_name = '_'.join(parts[:-1])
            dataset_name = parts[-1]
        else:
            model_name, dataset_name = parts
        
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                result_row = df.iloc[0].to_dict()
                result_row['model'] = model_name
                result_row['dataset'] = dataset_name
                all_results.append(result_row)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not all_results:
        print("No valid results found")
        return None
    
    return pd.DataFrame(all_results)

def create_summary_table(results_df):
    """Create summary table with models as rows and datasets as columns."""
    
    if results_df is None or len(results_df) == 0:
        return None
    
    # Create pivot table for mean AUC
    summary = results_df.pivot(index='model', columns='dataset', values='mean_auc')
    
    # Add overall mean across datasets for each model
    summary['overall_mean'] = summary.mean(axis=1, skipna=True)
    
    # Sort by overall mean (descending)
    summary = summary.sort_values('overall_mean', ascending=False)
    
    return summary

def create_detailed_summary(results_df):
    """Create detailed summary with key pathology performance."""
    
    if results_df is None or len(results_df) == 0:
        return None
    
    # Key pathologies to track
    key_pathologies = [
        'Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 
        'Edema_auc', 'Pleural effusion_auc', 'Pneumonia_auc', 'Pneumothorax_auc'
    ]
    
    detailed_results = []
    
    for _, row in results_df.iterrows():
        base_info = {
            'model': row['model'],
            'dataset': row['dataset'],
            'mean_auc': row.get('mean_auc', 0.0)
        }
        
        # Add pathology-specific results
        for pathology in key_pathologies:
            if pathology in row and pd.notna(row[pathology]):
                base_info[pathology] = row[pathology]
            else:
                base_info[pathology] = None
        
        detailed_results.append(base_info)
    
    return pd.DataFrame(detailed_results)

def print_summary_table(summary_df):
    """Print formatted summary table."""
    
    if summary_df is None:
        print("No summary data available")
        return
    
    print("\n" + "="*80)
    print("ZERO-SHOT CXR BENCHMARK SUMMARY (Mean AUC)")
    print("="*80)
    
    # Format to 4 decimal places
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(summary_df.to_string())
    
    print("\n" + "-"*80)
    print("RANKINGS (by overall mean AUC)")
    print("-"*80)
    
    for i, (model, row) in enumerate(summary_df.iterrows(), 1):
        overall_mean = row['overall_mean']
        print(f"{i:2d}. {model:20s} {overall_mean:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark summary")
    parser.add_argument('--results_dir', type=str, default='benchmark/results',
                        help='Directory containing result CSV files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (default: results_dir/benchmark_summary.csv)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Set output file
    if args.output is None:
        args.output = os.path.join(args.results_dir, "benchmark_summary.csv")
    
    print(f"Collecting results from: {args.results_dir}")
    
    # Collect results
    results_df = collect_results(args.results_dir)
    
    if results_df is None:
        print("No results to summarize")
        sys.exit(1)
    
    print(f"Found results for {len(results_df)} model-dataset combinations")
    
    # Check if we have resolution-specific results
    model_names = results_df['model'].unique()
    has_resolution_variants = any('_res' in model for model in model_names)
    if has_resolution_variants:
        print("📊 Detected resolution-specific results (e.g., model_res224, model_res448)")
        print("   These will be treated as separate models in the comparison")
    
    # Create summary table
    summary_df = create_summary_table(results_df)
    
    # Print summary
    print_summary_table(summary_df)
    
    # Save summary
    if summary_df is not None:
        summary_df.to_csv(args.output)
        print(f"\nSummary saved to: {args.output}")
    
    # Create detailed summary
    detailed_df = create_detailed_summary(results_df)
    if detailed_df is not None:
        detailed_output = args.output.replace('.csv', '_detailed.csv')
        detailed_df.to_csv(detailed_output, index=False)
        print(f"Detailed results saved to: {detailed_output}")
    
    print("\n" + "="*80)
    print("SUMMARY GENERATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 