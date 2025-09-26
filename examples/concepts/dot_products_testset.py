#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import pickle
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
from clear import clip, zero_shot

def print_gpu_memory(stage=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"🖥️ GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("🖥️ CUDA not available")

def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    dataset_configs = {
        'vindrcxr_test': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv"
        },
        'padchest_test': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv"
        },
        'indiana_test': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv"
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name]

def generate_dot_products(dataset_name):
    """Generate dot product matrix [#CXR, #Concepts] for specified test set"""
    
    print(f"=== Generating {dataset_name.upper()} Test Set Dot Products ===")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    
    print("\n=== Step 1: Load CLIP Model ===")
    model = load_clip(
        model_path="../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ CLIP model loaded and moved to CUDA")
    print_gpu_memory("after CLIP model load")
    
    print(f"\n=== Step 2: Setup {dataset_name.upper()} Test Dataset ===")
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    test_dataset = zero_shot.CXRTestDataset(
        img_path=dataset_config['cxr_filepath'],
        transform=transform,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✅ {dataset_name.upper()} test dataset loaded: {len(test_dataset)} images")
    
    print("\n=== Step 3: Load Diagnostic Concepts ===")
    # Load from same directory (script is in concepts/, csv is also in concepts/)
    concepts_df = pd.read_csv("mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    print(f"✅ Loaded {len(concepts)} diagnostic concepts")
    
    print("\n=== Step 4: Encode Concepts with CLIP ===")
    concept_batch_size = 1024
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            # L2 normalize concept features
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_features = torch.cat(all_concept_features).to('cuda')
    print(f"✅ Concept features shape: {concept_features.shape}")
    
    print("\n=== Step 5: Encode Images with CLIP ===")
    all_img_features = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Encoding images"):
            imgs = data['img'].to('cuda')
            img_features = model.encode_image(imgs)
            # L2 normalize image features
            img_features /= img_features.norm(dim=-1, keepdim=True)
            all_img_features.append(img_features.cpu())
    
    img_features = torch.cat(all_img_features)
    print(f"✅ Image features shape: {img_features.shape}")
    
    print("\n=== Step 6: Compute Dot Products ===")
    # Compute dot products between normalized features in batches to avoid memory issues
    img_batch_size = 100
    all_dot_products = []
    
    for i in tqdm(range(0, len(img_features), img_batch_size), desc="Computing dot products"):
        batch_img_features = img_features[i:i+img_batch_size].to('cuda')
        # Dot product between normalized features gives cosine similarity
        batch_dot_products = batch_img_features @ concept_features.T
        all_dot_products.append(batch_dot_products.cpu())
        torch.cuda.empty_cache()
    
    dot_products_matrix = torch.cat(all_dot_products).numpy()
    print(f"✅ Dot products matrix shape: {dot_products_matrix.shape}")
    print(f"✅ Dot products range: [{dot_products_matrix.min():.3f}, {dot_products_matrix.max():.3f}]")
    
    print("\n=== Step 7: Save Results ===")
    
    # Create output directory
    output_dir = "results/dot_products"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dot products matrix with dataset name
    np.save(f"{output_dir}/{dataset_name}_dot_products.npy", dot_products_matrix)
    print(f"✅ Dot products matrix saved: {output_dir}/{dataset_name}_dot_products.npy")
    
    # Save concept list for reference
    with open(f"{output_dir}/concepts_list.txt", 'w') as f:
        for i, concept in enumerate(concepts):
            f.write(f"{i}: {concept}\n")
    print(f"✅ Concepts list saved: {output_dir}/concepts_list.txt")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'num_images': len(test_dataset),
        'num_concepts': len(concepts),
        'matrix_shape': list(dot_products_matrix.shape),
        'normalization': 'L2 normalized features (cosine similarity)',
        'dot_products_range': [float(dot_products_matrix.min()), float(dot_products_matrix.max())],
        'model_path': '../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt',
        'concepts_file': 'mimic_concepts.csv',
        'image_resolution': 448
    }
    
    import json
    with open(f"{output_dir}/{dataset_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {output_dir}/{dataset_name}_metadata.json")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Images processed: {len(test_dataset)}")
    print(f"   🧠 Concepts used: {len(concepts)}")
    print(f"   📈 Matrix shape: {dot_products_matrix.shape}")
    print(f"   🔢 Dot products are cosine similarities (L2 normalized features)")
    print(f"   💾 Files saved in: {output_dir}/")
    
    return dot_products_matrix, concepts, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dot products matrix for CXR test datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        choices=['vindrcxr_test', 'padchest_test', 'indiana_test'],
        default='vindrcxr_test',
        help='Dataset to generate dot products for'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Dot Products Generation")
    print(f"📊 Dataset: {args.dataset}")
    
    try:
        dot_products_matrix, concepts, metadata = generate_dot_products(args.dataset)
        print(f"\n✅ {args.dataset.upper()} dot products generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during dot products generation: {e}")
        raise 
