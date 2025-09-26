#!/usr/bin/env python3
"""
Benchmark script for BiomedCLIP model.
Requires conda environment: open_clip
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import h5py
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to Python path to import from the main codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_base import setup_test_data, save_results, evaluate_predictions
from tqdm import tqdm

def load_biomedclip_model(device):
    """Load BiomedCLIP model from Hugging Face following official example."""
    print("Loading BiomedCLIP model...")
    
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        from PIL import Image
        
        # Load BiomedCLIP from Hugging Face using the OFFICIAL method
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        
        # Official loading pattern from HuggingFace
        model, preprocess = create_model_from_pretrained(model_name)
        tokenizer = get_tokenizer(model_name)
        
        # Create wrapper with the official interface
        class BiomedCLIPWrapper:
            def __init__(self, model, tokenizer, preprocess):
                self.model = model
                self.tokenizer = tokenizer
                self.preprocess = preprocess
                
            def forward(self, images, texts):
                """Official forward pass returning (image_features, text_features, logit_scale)."""
                return self.model(images, texts)
                
            def tokenize(self, texts, context_length=256):
                """Tokenize text using BiomedCLIP tokenizer."""
                return self.tokenizer(texts, context_length=context_length)
                
            def eval(self):
                self.model.eval()
                
            def to(self, device):
                self.model.to(device)
                return self
        
        model_wrapper = BiomedCLIPWrapper(model, tokenizer, preprocess)
        print("BiomedCLIP model loaded successfully using official method")
        return model_wrapper
        
    except ImportError as e:
        print(f"Error importing open_clip modules: {e}")
        print("Make sure you're running in the open_clip conda environment")
        print("Install with: pip install open_clip_torch==2.23.0 transformers==4.35.2")
        raise
    except Exception as e:
        print(f"Error loading BiomedCLIP model: {e}")
        raise

def run_biomedclip_evaluation(model, dataloader, y_true, labels, templates, device, context_length=256, save_detailed=False, model_name=None, dataset_name=None):
    """
    Run zero-shot evaluation using BiomedCLIP model following EXACT official pattern.
    Based on official example: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    """
    model.eval()
    pos_template, neg_template = templates[0]
    
    print(f"Running BiomedCLIP evaluation on {len(dataloader.dataset)} images...")
    
    # Prepare ALL text queries (positive AND negative for each label)
    # Following zero-shot pattern: for each pathology, we have "pathology" and "no pathology"
    all_texts = []
    for label in labels:
        all_texts.append(pos_template.format(label.lower()))  # e.g., "pneumonia"
        all_texts.append(neg_template.format(label.lower()))  # e.g., "no pneumonia"
    
    print(f"Using context length: {context_length}")
    print(f"Total text queries: {len(all_texts)} (2 per pathology)")
    print(f"Example texts: {all_texts[:4]}")
    
    # Tokenize ALL texts at once (OFFICIAL PATTERN)
    text_tokens = model.tokenize(all_texts, context_length=context_length).to(device)
    
    # Process images and compute predictions
    all_predictions = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Processing images"):
            imgs = data['img'].to(device)
            batch_size = imgs.shape[0]
            
            # Single forward pass with ALL texts (OFFICIAL PATTERN)
            image_features, text_features, logit_scale = model.forward(imgs, text_tokens)
            
            # Compute logits using OFFICIAL method: (logit_scale * image_features @ text_features.t())
            logits = (logit_scale * image_features @ text_features.t())  # [batch_size, num_all_texts]
            
            # Convert to probabilities for each pathology
            batch_predictions = torch.zeros(batch_size, len(labels))
            
            # For each pathology, we have 2 texts: positive and negative
            for i, label in enumerate(labels):
                pos_idx = i * 2      # Index of positive text
                neg_idx = i * 2 + 1  # Index of negative text
                
                # Get logits for positive and negative texts
                pos_logits = logits[:, pos_idx]  # [batch_size]
                neg_logits = logits[:, neg_idx]  # [batch_size]
                
                # Apply softmax between positive and negative (standard CLIP approach)
                combined_logits = torch.stack([neg_logits, pos_logits], dim=1)  # [batch_size, 2]
                probs = torch.softmax(combined_logits, dim=1)  # [batch_size, 2]
                
                # Take probability of positive class
                batch_predictions[:, i] = probs[:, 1]  # Positive class probability
            
            all_predictions.append(batch_predictions.cpu())
    
    # Concatenate all predictions
    y_pred = torch.cat(all_predictions, dim=0).numpy()
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Evaluate
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    # Save detailed results if requested
    if save_detailed and model_name and dataset_name:
        from benchmark_base import save_detailed_results
        config_data = {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "biomedclip_zero_shot",
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

def benchmark_biomedclip(datasets, device):
    """Benchmark BiomedCLIP on all datasets."""
    print(f"\n{'='*60}")
    print(f"Benchmarking BiomedCLIP model")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = load_biomedclip_model(device)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading BiomedCLIP model: {e}")
        return
    
    # Test on each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing on {dataset_name} ---")
        
        try:
            # Get basic dataset configuration
            _, y_true, labels, templates = setup_test_data(
                dataset_name, 
                batch_size=32,  # Smaller batch size for external models
                input_resolution=224  # BiomedCLIP uses 224x224
            )
            
            # Create custom dataset with BiomedCLIP preprocessing
            dataset_configs = {
                'chexpert_test': '../data/chexpert_test.h5',
                'padchest_test': '../data/padchest_test.h5', 
                'vindrcxr_test': '../data/vindrcxr_test.h5',
                'vindrpcxr_test': '../data/vindrpcxr_test.h5',
                'indiana_test': '../data/indiana_test.h5'
            }
            
            img_path = dataset_configs[dataset_name]
            biomedclip_dataset = BiomedCLIPDataset(img_path, model.preprocess)
            dataloader = DataLoader(biomedclip_dataset, batch_size=32, shuffle=False, 
                                  num_workers=2, pin_memory=True)
            
            print(f"Created BiomedCLIP dataset with {len(biomedclip_dataset)} images")
            
            # Run evaluation
            results_df, y_pred = run_biomedclip_evaluation(
                model, dataloader, y_true, labels, templates, device, context_length=256,
                save_detailed=True, model_name="biomedclip", dataset_name=dataset_name
            )
            
            # Save results
            save_results(results_df, "biomedclip", dataset_name)
            
        except Exception as e:
            print(f"Error evaluating BiomedCLIP on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description="Benchmark BiomedCLIP model")
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
    print("Note: Make sure you're running in the open_clip conda environment")
    
    # Benchmark BiomedCLIP
    benchmark_biomedclip(args.datasets, device)
    
    print(f"\n{'='*60}")
    print("BiomedCLIP benchmarking completed!")
    print("Results saved in benchmark/results/")
    print(f"{'='*60}")

class BiomedCLIPDataset(Dataset):
    """Custom dataset that uses BiomedCLIP's preprocessing."""
    
    def __init__(self, img_path, preprocess_fn):
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.preprocess = preprocess_fn
        
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image as numpy array
        img = self.img_dset[idx]
        
        # Convert to PIL Image (BiomedCLIP preprocessing expects PIL)
        from PIL import Image
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert to PIL and RGB (3 channels)
        img = Image.fromarray(img).convert('RGB')
        
        # Apply BiomedCLIP preprocessing
        if self.preprocess:
            img = self.preprocess(img)
        
        return {'img': img}

if __name__ == "__main__":
    main() 