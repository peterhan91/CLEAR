#!/usr/bin/env python3
"""
MONET Data Auditing for EC (Enlarged Cardiomediastinum) and Atelectasis Contamination Analysis
Following MONET methodology to understand concept contamination in medical image classification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import h5py
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
from clear import clip

def load_mimic_data(features_path, labels_path):
    """Load MIMIC-CXR training data and labels"""
    print("📂 Loading MIMIC-CXR training data...")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    print(f"✅ Loaded labels CSV: {labels_df.shape}")
    
    # Use the correct EC column name
    ec_column = 'Enlarged Cardiomediastinum'
    if ec_column not in labels_df.columns:
        print("Available columns:", labels_df.columns.tolist())
        raise ValueError(f"Could not find EC column '{ec_column}' in labels")
    
    print(f"✅ Using EC column: '{ec_column}'")
    
    # Load pre-computed features
    with h5py.File(features_path, 'r') as f:
        print(f"✅ Loaded features H5: {f['cxr_feature'].shape}")
        total_images = f['cxr_feature'].shape[0]
    
    return labels_df, ec_column, total_images

def sample_ec_cases(labels_df, ec_column, n_samples=1000):
    """Sample EC positive and negative cases for robust MONET analysis"""
    print(f"🎯 Sampling {n_samples} EC positive and {n_samples} EC negative cases...")
    
    # Get EC positive and negative indices
    ec_positive_mask = labels_df[ec_column] == 1
    ec_negative_mask = labels_df[ec_column] == 0
    
    ec_positive_indices = labels_df[ec_positive_mask].index.tolist()
    ec_negative_indices = labels_df[ec_negative_mask].index.tolist()
    
    print(f"📊 Available EC positive cases: {len(ec_positive_indices)}")
    print(f"📊 Available EC negative cases: {len(ec_negative_indices)}")
    
    if len(ec_positive_indices) < n_samples:
        print(f"⚠️ Warning: Only {len(ec_positive_indices)} EC positive cases available, using all")
        sampled_positive = ec_positive_indices
    else:
        sampled_positive = np.random.choice(ec_positive_indices, n_samples, replace=False).tolist()
    
    if len(ec_negative_indices) < n_samples:
        print(f"⚠️ Warning: Only {len(ec_negative_indices)} EC negative cases available, using all")
        sampled_negative = ec_negative_indices
    else:
        sampled_negative = np.random.choice(ec_negative_indices, n_samples, replace=False).tolist()
    
    print(f"✅ Sampled {len(sampled_positive)} EC positive and {len(sampled_negative)} EC negative cases")
    
    return sampled_positive, sampled_negative

def load_model_and_concepts():
    """Load DINOv2-CLIP model and diagnostic concepts"""
    print("🤖 Loading DINOv2-CLIP model...")
    
    model = load_clip(
        model_path="checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ Model loaded and moved to CUDA")
    
    print("📚 Loading diagnostic concepts...")
    concepts_df = pd.read_csv("concepts/mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    print(f"✅ Loaded {len(concepts)} diagnostic concepts")
    
    return model, concepts

def load_precomputed_features(features_path, indices):
    """Load pre-computed normalized features for specified indices"""
    print(f"🖼️ Loading pre-computed features for {len(indices)} images...")
    
    # Sort indices for H5Py compatibility and keep track of original order
    sorted_indices = np.array(sorted(indices))
    original_order = np.argsort(np.argsort(indices))  # To restore original order
    
    with h5py.File(features_path, 'r') as f:
        # Load features for sorted indices
        features = f['cxr_feature'][sorted_indices]  # Shape: [len(indices), 768]
        features_tensor = torch.from_numpy(features).float()
        
        # Restore original order
        features_tensor = features_tensor[original_order]
        
        # L2 normalize the features
        features_tensor = features_tensor / features_tensor.norm(dim=-1, keepdim=True)
    
    print(f"✅ Pre-computed features loaded: {features_tensor.shape}")
    return features_tensor

def compute_concept_embeddings_cached(model, concepts, cache_dir="model_auditing/cache"):
    """Compute normalized concept embeddings with caching"""
    print("💭 Computing concept embeddings with caching...")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/concept_embeddings_{len(concepts)}.pt"
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"📦 Loading cached concept embeddings from {cache_file}")
        concept_embeddings = torch.load(cache_file, map_location='cpu')
        print(f"✅ Cached concept embeddings loaded: {concept_embeddings.shape}")
        return concept_embeddings
    
    print(f"🔄 Computing concept embeddings (will cache to {cache_file})")
    concept_batch_size = 512
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            concept_features = concept_features / concept_features.norm(dim=-1, keepdim=True)  # L2 normalize
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_embeddings = torch.cat(all_concept_features)
    
    # Save to cache
    torch.save(concept_embeddings, cache_file)
    print(f"💾 Concept embeddings cached to {cache_file}")
    print(f"✅ Concept embeddings computed: {concept_embeddings.shape}")
    return concept_embeddings

def perform_monet_analysis(positive_embeddings, negative_embeddings, concept_embeddings, concepts):
    """Perform MONET differential analysis"""
    print("🔬 Performing MONET differential analysis...")
    
    # Step 1: Compute prototype embeddings (m+ and m-)
    m_positive = torch.mean(positive_embeddings, dim=0)  # Average of positive embeddings
    m_negative = torch.mean(negative_embeddings, dim=0)  # Average of negative embeddings
    
    # Step 2: Compute displacement vector (m_Δ = m+ - m-)
    m_delta = m_positive - m_negative
    
    # Step 3: Compute concept scores (C_Δ,i = m_Δ^T · concept_i)
    concept_scores = torch.matmul(m_delta, concept_embeddings.T)  # [num_concepts]
    
    # Step 4: Rank concepts by differential expression
    ranked_indices = torch.argsort(concept_scores, descending=True)
    
    results = []
    for i, concept_idx in enumerate(ranked_indices):
        concept_idx = concept_idx.item()
        score = concept_scores[concept_idx].item()
        
        results.append({
            'rank': i + 1,
            'concept': concepts[concept_idx],
            'differential_score': score,
            'more_present_in': 'EC_positive' if score > 0 else 'EC_negative'
        })
    
    print(f"✅ MONET analysis completed for {len(concepts)} concepts")
    return results, {
        'm_positive': m_positive.numpy(),
        'm_negative': m_negative.numpy(), 
        'm_delta': m_delta.numpy(),
        'concept_scores': concept_scores.numpy()
    }

def save_results(results, analysis_data, output_dir):
    """Save analysis results"""
    print(f"💾 Saving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save concept ranking results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/ec_atelectasis_concept_analysis.csv", index=False)
    
    # Save top concepts more present in EC positive cases
    ec_positive_concepts = results_df[results_df['more_present_in'] == 'EC_positive'].head(20)
    ec_positive_concepts.to_csv(f"{output_dir}/top_concepts_ec_positive.csv", index=False)
    
    # Save top concepts more present in EC negative cases  
    ec_negative_concepts = results_df[results_df['more_present_in'] == 'EC_negative'].head(20)
    ec_negative_concepts.to_csv(f"{output_dir}/top_concepts_ec_negative.csv", index=False)
    
    # Save raw analysis data
    np.savez(f"{output_dir}/monet_analysis_data.npz", **analysis_data)
    
    # Save summary JSON
    summary = {
        'total_concepts_analyzed': len(results),
        'concepts_more_in_ec_positive': len(results_df[results_df['more_present_in'] == 'EC_positive']),
        'concepts_more_in_ec_negative': len(results_df[results_df['more_present_in'] == 'EC_negative']),
        'top_5_ec_positive_concepts': ec_positive_concepts.head(5)['concept'].tolist(),
        'top_5_ec_negative_concepts': ec_negative_concepts.head(5)['concept'].tolist(),
        'score_range': [float(results_df['differential_score'].min()), float(results_df['differential_score'].max())]
    }
    
    with open(f"{output_dir}/analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Results saved:")
    print(f"   📊 Full analysis: {output_dir}/ec_atelectasis_concept_analysis.csv")
    print(f"   📈 EC positive concepts: {output_dir}/top_concepts_ec_positive.csv")
    print(f"   📉 EC negative concepts: {output_dir}/top_concepts_ec_negative.csv")
    print(f"   📋 Summary: {output_dir}/analysis_summary.json")

def main():
    """Main execution function"""
    print("🔍 MONET Data Auditing: EC and Atelectasis Contamination Analysis")
    print("📊 ENHANCED EXPERIMENT: 1000 samples each with text embedding caching")
    print("="*70)
    
    # Configuration
    features_path = "/home/than/DeepLearning/conceptqa_vip/data/mimic_train_448.h5"  # Pre-computed features
    labels_path = "/home/than/DeepLearning/conceptqa_vip/data/mimic_cxr.csv"
    output_dir = "model_auditing/results/ec_atelectasis_analysis"
    n_samples = 1000  # Increased sample size for more robust analysis
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Step 1: Load data and sample cases
        labels_df, ec_column, total_images = load_mimic_data(features_path, labels_path)
        positive_indices, negative_indices = sample_ec_cases(labels_df, ec_column, n_samples)
        
        # Step 2: Load model and concepts (only need model for concept encoding)
        model, concepts = load_model_and_concepts()
        
        # Step 3: Load pre-computed features and compute concept embeddings
        positive_embeddings = load_precomputed_features(features_path, positive_indices)
        negative_embeddings = load_precomputed_features(features_path, negative_indices)
        concept_embeddings = compute_concept_embeddings_cached(model, concepts)
        
        # Step 4: Perform MONET analysis
        results, analysis_data = perform_monet_analysis(
            positive_embeddings, negative_embeddings, concept_embeddings, concepts
        )
        
        # Step 5: Save results
        save_results(results, analysis_data, output_dir)
        
        print("\n" + "="*70)
        print("✅ MONET EC-Atelectasis Analysis Completed Successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 
