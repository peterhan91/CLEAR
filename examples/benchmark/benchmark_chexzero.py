#!/usr/bin/env python3
"""
Benchmark script for CheXzero model.
Requires conda environment: chexzero2
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to Python path to import from the main codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the official CheXzero code directory to Python path  
# Use relative path from the current script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
chexzero_code_path = os.path.join(project_root, "external_sota", "chexzero", "code", "CheXzero")
sys.path.insert(0, chexzero_code_path)

# Force tqdm to use console mode to avoid jupyter widget issues
os.environ['TQDM_DISABLE_JUPYTER'] = '1'
import tqdm
tqdm.tqdm_notebook = tqdm.tqdm  # Patch to use standard tqdm instead of notebook tqdm

from benchmark_base import setup_test_data, save_results, evaluate_predictions
from tqdm import tqdm

def load_chexzero_model(device):
    """Load CheXzero model using official codebase pattern."""
    print("Loading CheXzero model...")
    
    try:
        # Import from the official CheXzero codebase
        from clear import clip
        from model import CLIP  # Import CLIP model from official CheXzero model.py
        from clear.zero_shot import load_clip  # Use official CheXzero zero_shot load_clip function
        
        # Try to find CheXzero checkpoint using relative paths
        possible_paths = [
            # Try relative paths from project root
            "../CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt",
            "external_sota/chexzero/checkpoints/best_64_5e-05_original_22000_0.864.pt",
            # Fallback to absolute path if others don't work
            "/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found CheXzero checkpoint: {path}")
                break
        
        if model_path is None:
            print("Warning: No CheXzero checkpoint found. Using OpenAI CLIP pretrained model.")
            print("For best results, download CheXzero checkpoint.")
            # Load OpenAI pretrained model as fallback
            model = load_clip(
                model_path=None,
                pretrained=True,  # Use pretrained when no custom checkpoint
                context_length=77
            )
        else:
            # Create CheXzero model with exact parameters from checkpoint analysis
            print("Creating CheXzero model with correct parameters from checkpoint...")
            params = {
                'embed_dim': 512,  # From text_projection: (512, 512) and visual.proj: (768, 512)
                'image_resolution': 224,  # 7x7 patches with 32x32 patch_size = 224x224
                'vision_layers': 12,
                'vision_width': 768,  # From visual.conv1.weight and visual.proj
                'vision_patch_size': 32,  # From visual.conv1.weight: (768, 3, 32, 32)
                'context_length': 77,
                'vocab_size': 49408,
                'transformer_width': 512,  # From text_projection: (512, 512)
                'transformer_heads': 8,
                'transformer_layers': 12
            }
            
            model = CLIP(**params)
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("Successfully loaded CheXzero checkpoint with correct parameters!")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise
        
        print("CheXzero model loaded successfully using official pattern")
        return model
        
    except ImportError as e:
        print(f"Error importing zero_shot modules: {e}")
        print("Make sure zero_shot.py is available in the codebase")
        raise
    except Exception as e:
        print(f"Error loading CheXzero model: {e}")
        raise

def run_chexzero_evaluation(model, dataloader, y_true, labels, templates, device, context_length=77, save_detailed=False, model_name=None, dataset_name=None):
    """
    Run zero-shot evaluation using CheXzero model with full device compatibility.
    Implements the same algorithm as CheXzero's run_softmax_eval but with proper device handling.
    """
    from clear.zero_shot import zeroshot_classifier
    from clear import clip
    import numpy as np
    import torch
    from tqdm import tqdm
    
    model.eval()
    pos_template, neg_template = templates[0]
    
    print(f"Running CheXzero evaluation on {len(dataloader.dataset)} images...")
    print(f"Using device-aware implementation of CheXzero's run_softmax_eval()")
    print(f"Template: ('{pos_template}', '{neg_template}')")
    
    # Patch clip.tokenize to ensure tokens are on the correct device
    original_tokenize = clip.tokenize
    def device_aware_tokenize(texts, context_length=77):
        tokens = original_tokenize(texts, context_length=context_length)
        return tokens.to(device)
    
    # Temporarily replace clip.tokenize with our device-aware version
    clip.tokenize = device_aware_tokenize
    
    try:
        # Generate text embeddings for positive and negative templates
        print("Generating text embeddings...")
        pos_weights = zeroshot_classifier([pos_template.format(label) for label in labels], ["{}"], model, context_length)
        neg_weights = zeroshot_classifier([neg_template.format(label) for label in labels], ["{}"], model, context_length)
        
        # Run image predictions
        print("Running image predictions...")
        pos_predictions = []
        neg_predictions = []
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Processing images")):
                images = data['img'].to(device)
                batch_size = images.shape[0]
                
                # Get image features
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute logits for positive template
                pos_logits = image_features @ pos_weights
                pos_logits = pos_logits.cpu().numpy()
                
                # Compute logits for negative template  
                neg_logits = image_features @ neg_weights
                neg_logits = neg_logits.cpu().numpy()
                
                # Handle each image in the batch individually
                for j in range(batch_size):
                    pos_predictions.append(pos_logits[j] if pos_logits.ndim > 1 else pos_logits)
                    neg_predictions.append(neg_logits[j] if neg_logits.ndim > 1 else neg_logits)
        
        pos_pred = np.array(pos_predictions)
        neg_pred = np.array(neg_predictions)
        
    finally:
        # Restore original tokenize function
        clip.tokenize = original_tokenize
    
    # Compute probabilities with softmax (same as run_softmax_eval)
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Evaluate using our common evaluation function
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    # Save detailed results if requested
    if save_detailed and model_name and dataset_name:
        from benchmark_base import save_detailed_results
        config_data = {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "chexzero_zero_shot",
            "test_batch_size": dataloader.batch_size,
            "templates": [pos_template, neg_template],
            "context_length": context_length,
            "num_images": len(y_pred),
            "num_labels": len(labels),
            "labels": labels,
            "input_resolution": 224
        }
        save_detailed_results(y_pred, y_true, labels, model_name, dataset_name, config_data)
        return results_df, y_pred
    
    return results_df

def benchmark_chexzero(datasets, device):
    """Benchmark CheXzero on all datasets."""
    print(f"\n{'='*60}")
    print(f"Benchmarking CheXzero model")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = load_chexzero_model(device)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading CheXzero model: {e}")
        return
    
    # Test on each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing on {dataset_name} ---")
        
        try:
            # Setup test data using official CheXzero resolution (224x224)
            dataloader, y_true, labels, templates = setup_test_data(
                dataset_name, 
                batch_size=32,  # Smaller batch size for external models
                input_resolution=224  # CheXzero uses 224x224 (7x7 patches * 32 patch_size)
            )
            
            # Run evaluation using official CheXzero pattern
            results_df, y_pred = run_chexzero_evaluation(
                model, dataloader, y_true, labels, templates, device, context_length=77, 
                save_detailed=True, model_name="chexzero", dataset_name=dataset_name
            )
            
            # Save results
            save_results(results_df, "chexzero", dataset_name)
            
        except Exception as e:
            print(f"Error evaluating CheXzero on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description="Benchmark CheXzero model")
    parser.add_argument('--datasets', nargs='+', 
                        default=['chexpert_test', 'padchest_test', 'vindrcxr_test', 'vindrpcxr_test', 'indiana_test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("Note: Make sure you're running in the chexzero2 conda environment")
    
    # Benchmark CheXzero
    benchmark_chexzero(args.datasets, device)
    
    print(f"\n{'='*60}")
    print("CheXzero benchmarking completed!")
    print("Results saved in benchmark/results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
