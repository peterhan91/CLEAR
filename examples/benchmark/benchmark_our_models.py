#!/usr/bin/env python3
"""
Benchmark script for our trained models.
"""

import os
import sys
import argparse
import torch

# Add the parent directory to Python path to import from the main codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_base import setup_test_data, run_zero_shot_evaluation, save_results
from train import load_clip

def find_model_paths(checkpoints_dir="checkpoints"):
    """Find all available trained models."""
    model_paths = {}
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return model_paths
    
    for model_dir in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, model_dir)
        if os.path.isdir(model_path):
            best_model_path = os.path.join(model_path, "best_model.pt")
            if os.path.exists(best_model_path):
                model_paths[model_dir] = best_model_path
                print(f"Found model: {model_dir} -> {best_model_path}")
    
    return model_paths

def load_our_model(model_path, model_name):
    """Load our trained model with EXACT training script configuration."""
    print(f"Loading model from: {model_path}")
    
    # Determine model configuration from name (following training script pattern)
    use_dinov2 = 'dinov2' in model_name.lower()
    dinov2_model_name = None
    
    if use_dinov2:
        if 'vitb' in model_name.lower():
            dinov2_model_name = 'dinov2_vitb14'
        elif 'vits' in model_name.lower():
            dinov2_model_name = 'dinov2_vits14'
        elif 'vitl' in model_name.lower():
            dinov2_model_name = 'dinov2_vitl14'
        elif 'vitg' in model_name.lower():
            dinov2_model_name = 'dinov2_vitg14'
        else:
            dinov2_model_name = 'dinov2_vitb14'  # default
    
    # CRITICAL FIX: Follow EXACT training script pattern
    # From run_train_improved.py: random_init=True by default, so pretrained=False
    # Our models are trained from scratch (not pretrained), so pretrained=False
    pretrained = False  # Our models are NOT using OpenAI pretrained weights
    
    # Load model using EXACT training script pattern
    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,  # FALSE for our trained models
        context_length=77,
        use_dinov2=use_dinov2,
        dinov2_model_name=dinov2_model_name,
        freeze_dinov2=False  # Not frozen during inference
    )
    
    print(f"Model loaded: use_dinov2={use_dinov2}, dinov2_model={dinov2_model_name}, pretrained={pretrained}")
    return model, use_dinov2

def determine_input_resolution(model_name, use_dinov2):
    """Determine input resolution following EXACT training script logic."""
    # From training script: input_resolution = 448 if (not config.random_init or config.use_dinov2) else 320
    # Since random_init=True by default, not config.random_init = False
    # So: input_resolution = 448 if (False or use_dinov2) else 320
    # Which simplifies to: input_resolution = 448 if use_dinov2 else 320
    
    if use_dinov2:
        return 448
    else:
        # Check if model name suggests it's a pretrained model
        if 'pretrained' in model_name.lower() or 'openai' in model_name.lower():
            return 448  # Pretrained models use 448
        else:
            return 320  # Our CLIP models trained from scratch use 320
    
def benchmark_model_at_resolution(model, model_name, datasets, device, input_resolution, resolution_suffix):
    """Benchmark a model at a specific resolution on all datasets."""
    print(f"\n--- Resolution: {input_resolution}x{input_resolution} ---")
    
    # Test on each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing on {dataset_name} at {input_resolution}x{input_resolution} ---")
        
        try:
            # Setup test data with specified resolution
            dataloader, y_true, labels, templates = setup_test_data(
                dataset_name, 
                batch_size=64,  # Our models can handle larger batch size
                input_resolution=input_resolution
            )
            
            # Run evaluation with detailed results
            model_name_with_res = f"{model_name}_{resolution_suffix}"
            results_df, y_pred = run_zero_shot_evaluation(
                model, dataloader, y_true, labels, templates, device, context_length=77,
                save_detailed=True, model_name=model_name_with_res, dataset_name=dataset_name
            )
            
            # Save results with resolution suffix
            save_results(results_df, model_name_with_res, dataset_name)
            
        except Exception as e:
            print(f"Error evaluating {model_name} on {dataset_name} at {input_resolution}x{input_resolution}: {e}")
            import traceback
            traceback.print_exc()
            continue

def benchmark_model(model_name, model_path, datasets, device):
    """Benchmark a single model on all datasets at both 224x224 and 448x448 resolutions."""
    print(f"\n{'='*60}")
    print(f"Benchmarking model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Testing at both 224x224 and 448x448 resolutions")
    print(f"{'='*60}")
    
    # Load model with proper configuration
    try:
        model, use_dinov2 = load_our_model(model_path, model_name)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        print(f"Model config: use_dinov2={use_dinov2}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test at both resolutions
    resolutions_to_test = [224, 448]
    
    for resolution in resolutions_to_test:
        resolution_suffix = f"res{resolution}"
        print(f"\n{'='*40}")
        print(f"Testing {model_name} at {resolution}x{resolution}")
        print(f"{'='*40}")
        
        benchmark_model_at_resolution(
            model=model,
            model_name=model_name, 
            datasets=datasets,
            device=device,
            input_resolution=resolution,
            resolution_suffix=resolution_suffix
        )

def main():
    parser = argparse.ArgumentParser(description="Benchmark our trained models")
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--datasets', nargs='+', 
                        default=['chexpert_test', 'padchest_test', 'vindrcxr_test', 'vindrpcxr_test', 'indiana_test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to evaluate (default: all found models)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Find available models
    model_paths = find_model_paths(args.checkpoints_dir)
    
    if not model_paths:
        print("No models found!")
        return
    
    # Filter models if specified
    if args.models:
        filtered_paths = {k: v for k, v in model_paths.items() if k in args.models}
        if not filtered_paths:
            print(f"None of the specified models found: {args.models}")
            print(f"Available models: {list(model_paths.keys())}")
            return
        model_paths = filtered_paths
    
    print(f"Will evaluate {len(model_paths)} models on {len(args.datasets)} datasets")
    print(f"Each model will be tested at both 224x224 and 448x448 resolutions")
    print(f"Total evaluations: {len(model_paths)} models x {len(args.datasets)} datasets x 2 resolutions = {len(model_paths) * len(args.datasets) * 2}")
    
    # Benchmark each model
    for model_name, model_path in model_paths.items():
        benchmark_model(model_name, model_path, args.datasets, device)
    
    print(f"\n{'='*60}")
    print("Benchmarking completed!")
    print("Results saved in benchmark/results/")
    print("Each model tested at both 224x224 and 448x448 resolutions:")
    print("  - *_res224_*.csv: Results at 224x224 (for comparison with other SOTA models)")
    print("  - *_res448_*.csv: Results at 448x448 (native training resolution)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 