#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LogisticRegressionModel(nn.Module):
    """Logistic Regression Model for multi-label classification"""
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def fix_numpy_compatibility():
    """Fix numpy compatibility for older pickle files"""
    try:
        import numpy.core as np_core
        if not hasattr(np, '_core'):
            np._core = np_core
    except (ImportError, AttributeError):
        pass

def load_concepts_and_embeddings():
    """Load diagnostic concepts and their embeddings"""
    print("Loading concepts and embeddings...")
    
    # Load concepts with their indices
    concepts_df = pd.read_csv("concepts/mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()
    
    # Load concept embeddings
    fix_numpy_compatibility()
    with open("concepts/embeddings/concepts_embeddings_sfr_mistral.pickle", 'rb') as f:
        embeddings_data = pickle.load(f)
    
    if isinstance(embeddings_data, dict):
        embedding_dim = len(list(embeddings_data.values())[0])
        concept_embeddings = np.zeros((len(concepts), embedding_dim))
        
        missing_count = 0
        for pos, concept_idx in enumerate(concept_indices):
            if concept_idx in embeddings_data:
                concept_embeddings[pos] = embeddings_data[concept_idx]
            else:
                concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} concepts missing embeddings, used random vectors")
    else:
        concept_embeddings = np.array(embeddings_data)
    
    concept_embeddings = torch.tensor(concept_embeddings).float()
    print(f"Loaded {len(concepts)} concepts with {concept_embeddings.shape[1]}-dim embeddings")
    
    return concepts, concept_indices, concept_embeddings

def load_linear_probing_weights_all_seeds(results_dir, concepts, concept_embeddings):
    """Load linear probing weights from all 20 seeds and compute concept importance"""
    print("Loading linear probing weights from all 20 seeds...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Move concept embeddings to device for faster computation
    concept_embeddings_gpu = concept_embeddings.to(device)
    print(f"Concept embeddings moved to {device}")
    
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    all_concept_weights = []
    labels = None
    
    for seed in tqdm(seeds, desc="Loading seeds"):
        # Load model info
        with open(f"{results_dir}/results_seed_{seed}.json", 'r') as f:
            info = json.load(f)
        
        if labels is None:
            labels = info['labels']
        
        # Load trained model
        try:
            fix_numpy_compatibility()
            model_data = torch.load(f"{results_dir}/models/seed_{seed}_model.pth", map_location=device, weights_only=False)
            
            # Create model and load weights
            model = LogisticRegressionModel(info['feature_dim'], len(labels))
            model.load_state_dict(model_data['state_dict'])
            model = model.to(device)
            
        except Exception as e:
            print(f"Warning: Could not load model for seed {seed}: {e}")
            continue
        
        # Get the linear layer weights [num_labels, feature_dim]
        linear_weights = model.linear.weight.detach()  # Keep on GPU: [13, 4096]
        
        # Compute concept importance using GPU-accelerated cosine similarity
        # Normalize vectors for cosine similarity
        linear_weights_norm = torch.nn.functional.normalize(linear_weights, dim=1)  # [13, 4096]
        concept_embeddings_norm = torch.nn.functional.normalize(concept_embeddings_gpu, dim=1)  # [num_concepts, 4096]
        
        # Compute all cosine similarities at once: [num_concepts, 13]
        concept_importance_matrix = torch.matmul(concept_embeddings_norm, linear_weights_norm.T)
        
        # Move back to CPU and convert to numpy
        concept_importance_matrix = concept_importance_matrix.cpu().numpy()
        
        all_concept_weights.append(concept_importance_matrix)
        
        # Clear GPU memory
        del model, linear_weights, linear_weights_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not all_concept_weights:
        raise ValueError("No model weights could be loaded!")
    
    # Convert to numpy array [num_seeds, num_concepts, num_labels]
    all_concept_weights = np.array(all_concept_weights)
    
    # Compute statistics across seeds
    avg_concept_weights = np.mean(all_concept_weights, axis=0)  # [num_concepts, num_labels]
    std_concept_weights = np.std(all_concept_weights, axis=0)   # [num_concepts, num_labels]
    
    print(f"Averaged linear probing concept importance shape: {avg_concept_weights.shape}")
    print(f"Averaged across {len(all_concept_weights)} seeds")
    return avg_concept_weights, std_concept_weights, all_concept_weights, labels

def plot_concept_importance(concept_weights, std_weights, all_weights, concepts, labels, output_dir):
    """Create bar plot with top 10 positive concepts for each label, with error bars and individual points"""
    print(f"Creating concept importance plots...")
    
    # Set up plotting style
    plt.style.use('default')
    
    # Use all available labels instead of filtering to specific ones
    target_labels = labels
    print(f"Target labels: {target_labels}")
    
    for target_label in tqdm(target_labels, desc="Creating plots"):
        if target_label not in labels:
            print(f"Warning: {target_label} not found in labels, skipping...")
            continue
            
        label_idx = labels.index(target_label)
        
        # Get weights for this label
        label_weights = concept_weights[:, label_idx]
        label_stds = std_weights[:, label_idx]
        label_all_weights = all_weights[:, :, label_idx]  # [num_seeds, num_concepts] for this label
        
        # Create DataFrame for easier handling
        df = pd.DataFrame({
            'concept': concepts,
            'weight': label_weights,
            'std': label_stds,
            'concept_idx': range(len(concepts))
        })
        
        # Get only positive concepts
        df_positive = df[df['weight'] > 0].copy()
        df_positive = df_positive.sort_values('weight', ascending=False)
        
        # Take top 10 positive concepts
        df_pos_top10 = df_positive.head(10)
        
        # Prepare data for plotting (top to bottom: highest to lowest)
        combined_concepts = []
        combined_weights = []
        combined_stds = []
        combined_labels = []
        combined_indices = []
        
        # Add positive concepts (top to bottom: highest to lowest)
        for _, row in df_pos_top10.iterrows():
            combined_concepts.append(row['concept'])
            combined_weights.append(row['weight'])
            combined_stds.append(row['std'])
            combined_labels.append(row['concept'])
            combined_indices.append(row['concept_idx'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(13, 12))  # Sized for 10 positive concepts
        
        # Use consistent salmon red color for all positive concepts
        colors = ['#CD5C5C'] * len(combined_weights)
        
        # Create horizontal bar plot with error bars
        y_positions = range(len(combined_weights))
        bars = ax.barh(y_positions, combined_weights, xerr=combined_stds,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1,
                      capsize=0, error_kw={'linewidth': 1.5, 'alpha': 0.8})
        
        # Add individual data points as open circles
        for i, concept_idx in enumerate(combined_indices):
            individual_values = label_all_weights[:, concept_idx]  # values for this concept across seeds
            y_jitter = np.random.normal(y_positions[i], 0.05, len(individual_values))  # Small jitter
            ax.scatter(individual_values, y_jitter, 
                      s=30, facecolors='none', edgecolors='black', 
                      alpha=0.6, linewidth=1)
        
        # Set labels
        ax.set_yticks(y_positions)
        # Clean up concept names (capitalize first letter, wrap long text)
        clean_labels = []
        for concept in combined_labels:
            clean_concept = concept.strip().capitalize()
            # Wrap long text instead of truncating
            words = clean_concept.split()
            if len(words) > 5:  # If more than 6 words, wrap to multiple lines
                # Split into lines of max 6 words each
                lines = []
                for i in range(0, len(words), 5):
                    lines.append(" ".join(words[i:i+5]))
                clean_concept = "\n".join(lines)
            clean_labels.append(clean_concept)
        
        ax.set_yticklabels(clean_labels, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=24)
        ax.set_xlabel('Concept importance score', fontsize=28)
        
        # Invert y-axis so positive concepts are at top
        ax.invert_yaxis()
        
        # Set adaptive x-axis limits based on the top 24 concepts being plotted
        all_values = combined_weights + combined_stds + [x for sublist in [label_all_weights[:, idx] for idx in combined_indices] for x in sublist]
        min_value = min(all_values)
        max_value = max(all_values)
        
        # Use the 10th concept's importance as the effective minimum for better zoom
        min_concept_importance = min(combined_weights)  # This is the 10th concept (lowest of top 10)
        
        # Set range from slightly below the 10th concept to slightly above the 1st concept
        value_range = max_value - min_concept_importance
        padding = value_range * 0.05  # Smaller padding for better zoom
        ax.set_xlim(min_concept_importance - padding, max_value + padding)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add title with label name
        ax.set_title(f'{target_label}', fontsize=32, pad=10)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        safe_label = target_label.replace(' ', '_').replace('/', '_')
        output_file = f"{output_dir}/SFR_Embedding_Mistral_{safe_label}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_file}")
        
        # Print summary stats
        positive_count = len(df_positive)
        max_positive = df_positive['weight'].max() if positive_count > 0 else 0
        min_positive = df_positive['weight'].min() if positive_count > 0 else 0
        
        print(f"  {target_label}: {positive_count} positive concepts")
        print(f"    Positive range: {min_positive:.4f} to {max_positive:.4f}")
        print(f"    Showing top 10 positive concepts with error bars and individual points")
        if len(df_pos_top10) > 0:
            print(f"    Top positive concept: {df_pos_top10.iloc[0]['concept'][:50]}...")
            print(f"    10th positive concept: {df_pos_top10.iloc[-1]['concept'][:50]}...")

def main():
    """Main function to run linear probing concept importance visualization"""
    print("=" * 60)
    print("LINEAR PROBING CONCEPT IMPORTANCE VISUALIZATION")
    print("=" * 60)
    
    # Paths
    results_dir = "concepts/results/concept_based_linear_probing_torch"
    output_dir = "concepts/results/linear_probing_concept_importance_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load concepts and embeddings
    concepts, concept_indices, concept_embeddings = load_concepts_and_embeddings()
    
    # Load and plot linear probing concept importance (averaged across all seeds)
    print("\n" + "=" * 40)
    print("SFR-EMBEDDING-MISTRAL (AVERAGED ACROSS 20 SEEDS)")
    print("=" * 40)
    avg_weights, std_weights, all_weights, labels = load_linear_probing_weights_all_seeds(
        results_dir, concepts, concept_embeddings)
    plot_concept_importance(avg_weights, std_weights, all_weights, concepts, labels, output_dir)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main() 