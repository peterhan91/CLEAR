#!/usr/bin/env python3
"""
Generate prediction and ground truth CSV files for benchmark models.
This script runs the modified benchmark scripts that now save detailed predictions
in the same format as the concept-based evaluation results.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_benchmark_for_detailed_results(model_name, dataset_name, device='auto'):
    """Run benchmark script to generate detailed results with predictions and ground truth."""
    
    print(f"\n{'='*80}")
    print(f"Generating detailed results for {model_name} on {dataset_name}")
    print(f"{'='*80}")
    
    # Get benchmark directory
    benchmark_dir = Path(__file__).parent.parent.parent / "benchmark"
    
    # Model to script mapping
    script_mapping = {
        'chexzero': 'benchmark_chexzero.py',
        'cxr_foundation': 'benchmark_cxr_foundation.py', 
        'biomedclip': 'benchmark_biomedclip.py',
        'openai_clip': 'benchmark_openai_clip.py'
    }
    
    if model_name not in script_mapping:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    script_path = benchmark_dir / script_mapping[model_name]
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Prepare command
    cmd = f"conda activate chexzero2 && python {script_path} --datasets {dataset_name} --device {device}"
    
    print(f"Running command: {cmd}")
    print(f"Working directory: {benchmark_dir}")
    
    try:
        # Run the benchmark script
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(benchmark_dir),
            capture_output=True,
            text=True,
            check=True,
            executable='/bin/bash'  # Ensure bash is used for conda activate
        )
        
        print("✅ Benchmark completed successfully")
        
        # Check if detailed results were created
        analysis_dir = Path(__file__).parent.parent / "analysis"
        result_pattern = f"benchmark_evaluation_{dataset_name}_{model_name}"
        
        result_dirs = list(analysis_dir.glob(result_pattern))
        if result_dirs:
            latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
            print(f"📁 Detailed results saved in: {latest_dir}")
            
            # Check for required files
            required_files = ['ground_truth_*.csv', 'predictions_*.csv', 'config_*.json', 'summary_*.json']
            missing_files = []
            
            for pattern in required_files:
                if not list(latest_dir.glob(pattern)):
                    missing_files.append(pattern)
            
            if missing_files:
                print(f"⚠️  Warning: Missing files: {missing_files}")
            else:
                print("✅ All required files generated successfully")
            
            return True
        else:
            print(f"❌ No detailed results found in {analysis_dir}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-2000:])  # Last 2000 chars
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-2000:])  # Last 2000 chars
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate model predictions and ground truth CSV files")
    parser.add_argument('--models', nargs='+', 
                        default=['chexzero', 'biomedclip', 'cxr_foundation'],
                        choices=['chexzero', 'cxr_foundation', 'biomedclip', 'openai_clip'],
                        help='Models to evaluate')
    parser.add_argument('--dataset', type=str, default='padchest_test',
                        help='Dataset to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Device: {args.device}")
    
    # Track results
    results_summary = {}
    
    # Run each model
    for model_name in args.models:
        success = run_benchmark_for_detailed_results(model_name, args.dataset, args.device)
        results_summary[model_name] = success
    
    # Print summary
    print(f"\n{'='*80}")
    print("MODEL PREDICTION GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print()
    
    for model_name, success in results_summary.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} {model_name}")
    
    successful_models = [name for name, success in results_summary.items() if success]
    failed_models = [name for name, success in results_summary.items() if not success]
    
    if successful_models:
        print(f"\n✅ Successfully generated predictions for: {successful_models}")
        print(f"📁 Results are saved in: concepts/analysis/")
        print(f"📊 Ready for bootstrapping analysis and model comparison!")
        
        # Check what we have for comparison
        analysis_dir = Path(__file__).parent.parent / "analysis"
        print(f"\n📋 Available model results for comparison:")
        print(f"   🔹 Our SFR-Mistral concept model: concepts/results/concept_based_evaluation_{args.dataset}_sfr_mistral/")
        
        for model in successful_models:
            result_pattern = f"benchmark_evaluation_{args.dataset}_{model}"
            result_dirs = list(analysis_dir.glob(result_pattern))
            if result_dirs:
                latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
                print(f"   🔹 {model}: {latest_dir}")
    
    if failed_models:
        print(f"\n❌ Failed models: {failed_models}")
        print("Please check the error messages above and ensure all dependencies are installed.")
    
    return len(failed_models) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 