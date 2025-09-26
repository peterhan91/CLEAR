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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LogisticRegressionModel(nn.Module):
    """Logistic Regression Model for multi-label classification"""
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_concepts():
    """Load concepts from the CBM concepts JSON file in the same order as used in training"""
    print("Loading concepts...")
    with open("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/cbm_concepts.json", 'r') as f:
        cbm_data = json.load(f)
    
    concepts = []
    concept_indices = []
    
    # Extract concepts in the same order as used in the training script (exp_cbm.py)
    # This follows the order in load_filtered_concepts() from exp_cbm.py
    for label, concept_list in cbm_data.items():
        for item in concept_list:
            concepts.append(item['concept'])
            concept_indices.append(item['concept_idx'])
    
    print(f"Loaded {len(concepts)} concepts")
    return concepts, concept_indices

def load_concept_embeddings(concept_indices):
    """Load concept embeddings for improved CBM weight computation"""
    print("Loading concept embeddings...")
    
    # Use the same embedding path as in exp_cbm.py
    embedding_path = "/home/than/DeepLearning/CheXzero/embeddings_output/cxr_embeddings_sfr_mistral.pickle"
    
    # Try to fix numpy compatibility for older pickle files
    try:
        import numpy.core as np_core
        if not hasattr(np, '_core'):
            np._core = np_core
    except (ImportError, AttributeError):
        pass
    
    with open(embedding_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    print(f"Loaded embeddings from: {embedding_path}")
    
    if isinstance(embeddings_data, dict):
        embedding_dim = len(list(embeddings_data.values())[0])
        concept_embeddings = np.zeros((len(concept_indices), embedding_dim))
        
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
    print(f"Concept embeddings shape: {concept_embeddings.shape}")
    
    return concept_embeddings

def load_standard_cbm_weights_all_seeds(checkpoint_base_path):
    """Load standard CBM weights from all 20 seeds and return both averages and individual values"""
    print("Loading standard CBM weights from all 20 seeds...")
    
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    all_weights = []
    labels = None
    
    for seed in seeds:
        seed_path = f"{checkpoint_base_path}/seed_{seed}"
        
        # Load model info
        with open(f"{seed_path}/standard_cbm_info.json", 'r') as f:
            info = json.load(f)
        
        if labels is None:
            labels = info['labels']
        
        # Create model and load weights
        model = LogisticRegressionModel(info['input_dim'], info['output_dim'])
        model.load_state_dict(torch.load(f"{seed_path}/standard_cbm_weights.pth", map_location='cpu'))
        
        # Extract weights [68, 13]
        weights = model.linear.weight.detach().cpu().numpy().T  # Transpose to [68, 13]
        all_weights.append(weights)
    
    # Convert to numpy array [20, 68, 13]
    all_weights = np.array(all_weights)
    
    # Compute statistics across seeds
    avg_weights = np.mean(all_weights, axis=0)  # [68, 13]
    std_weights = np.std(all_weights, axis=0)   # [68, 13]
    
    print(f"Averaged standard CBM weights shape: {avg_weights.shape}")
    print(f"Averaged across {len(all_weights)} seeds")
    return avg_weights, std_weights, all_weights, labels

def load_improved_cbm_weights_all_seeds(checkpoint_base_path, concept_embeddings):
    """Load improved CBM weights from all 20 seeds and return both averages and individual values"""
    print("Loading improved CBM weights from all 20 seeds...")
    
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    all_concept_weights = []
    labels = None
    
    for seed in seeds:
        seed_path = f"{checkpoint_base_path}/seed_{seed}"
        
        # Load model info
        with open(f"{seed_path}/improved_cbm_info.json", 'r') as f:
            info = json.load(f)
        
        if labels is None:
            labels = info['labels']
        
        # Create model and load weights
        model = LogisticRegressionModel(info['input_dim'], info['output_dim'])
        model.load_state_dict(torch.load(f"{seed_path}/improved_cbm_weights.pth", map_location='cpu'))
        
        # Get the linear layer weights [13, 4096]
        linear_weights = model.linear.weight.detach().cpu().numpy()  # [13, 4096]
        
        # Compute concept importance: [#C, 4096] @ [4096, 13] → [#C, 13]
        concept_emb_np = concept_embeddings.numpy()  # [#C, 4096]
        concept_weights = concept_emb_np @ linear_weights.T  # [#C, 4096] @ [4096, 13] → [#C, 13]
        
        all_concept_weights.append(concept_weights)
    
    # Convert to numpy array [20, 68, 13]
    all_concept_weights = np.array(all_concept_weights)
    
    # Compute statistics across seeds
    avg_concept_weights = np.mean(all_concept_weights, axis=0)  # [68, 13]
    std_concept_weights = np.std(all_concept_weights, axis=0)   # [68, 13]
    
    print(f"Averaged improved CBM concept weights shape: {avg_concept_weights.shape}")
    print(f"Averaged across {len(all_concept_weights)} seeds")
    return avg_concept_weights, std_concept_weights, all_concept_weights, labels

def plot_concept_importance(concept_weights, std_weights, all_weights, concepts, labels, method_name, output_dir):
    """Create single bar plot with top 8 positive and top 4 negative concepts for each label, with error bars and individual points"""
    print(f"Creating plots for {method_name}...")
    
    # Set up plotting style
    plt.style.use('default')
    
    for label_idx, label in enumerate(labels):
        # Get weights for this label
        label_weights = concept_weights[:, label_idx]
        label_stds = std_weights[:, label_idx]
        label_all_weights = all_weights[:, :, label_idx]  # [20, 68] for this label
        
        # Create DataFrame for easier handling
        df = pd.DataFrame({
            'concept': concepts,
            'weight': label_weights,
            'std': label_stds,
            'concept_idx': range(len(concepts))
        })
        
        # Separate positive and negative concepts
        df_positive = df[df['weight'] > 0].copy()
        df_positive = df_positive.sort_values('weight', ascending=False)
        df_negative = df[df['weight'] < 0].copy()
        df_negative['abs_weight'] = df_negative['weight'].abs()
        
        # Take top 6 positive and top 3 negative (by largest absolute value)
        df_pos_top6 = df_positive.head(6)
        df_neg_top3 = df_negative.nlargest(3, 'abs_weight')  # Select 3 with largest absolute values
        
        # Now sort the selected negative concepts by smallest absolute value for plotting
        df_neg_top3 = df_neg_top3.sort_values('abs_weight', ascending=True)  # Smallest absolute value first
        
        # Combine and order: positive concepts first (high to low), then negative (low to high)
        combined_concepts = []
        combined_weights = []
        combined_stds = []
        combined_labels = []
        combined_indices = []
        
        # Add positive concepts (top to bottom: highest to lowest)
        for _, row in df_pos_top6.iterrows():
            combined_concepts.append(row['concept'])
            combined_weights.append(row['weight'])
            combined_stds.append(row['std'])
            combined_labels.append(row['concept'])
            combined_indices.append(row['concept_idx'])
        
        # Add negative concepts (top to bottom: most negative to least negative)
        for _, row in df_neg_top3.iterrows():
            combined_concepts.append(row['concept'])
            combined_weights.append(row['weight'])
            combined_stds.append(row['std'])
            combined_labels.append(row['concept'])
            combined_indices.append(row['concept_idx'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Create color list: salmon red for positive, steel blue for negative (matching reference image)
        colors = ['#CD5C5C' if weight > 0 else '#4682B4' for weight in combined_weights]
        
        # Create horizontal bar plot with error bars
        y_positions = range(len(combined_weights))
        bars = ax.barh(y_positions, combined_weights, xerr=combined_stds,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
                      capsize=0, error_kw={'linewidth': 1.5, 'alpha': 0.8})
        
        # Add individual data points as open circles
        for i, concept_idx in enumerate(combined_indices):
            individual_values = label_all_weights[:, concept_idx]  # 20 values for this concept
            y_jitter = np.random.normal(y_positions[i], 0.05, len(individual_values))  # Small jitter
            ax.scatter(individual_values, y_jitter, 
                      s=30, facecolors='none', edgecolors='black', 
                      alpha=0.6, linewidth=1)
        
        # Set labels
        ax.set_yticks(y_positions)
        # Clean up concept names (capitalize first letter, break long lines)
        clean_labels = []
        for concept in combined_labels:
            clean_concept = concept.strip().capitalize()
            # Break long text into multiple lines
            if len(clean_concept) > 40:
                words = clean_concept.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += (" " if current_line else "") + word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                clean_concept = "\n".join(lines)
            clean_labels.append(clean_concept)
        
        ax.set_yticklabels(clean_labels, fontsize=24)
        
        # Fix x-axis ticks to ensure proper coordinate system
        ax.tick_params(axis='x', labelsize=24)
        ax.set_xlabel('Concept importance score', fontsize=26)
        
        # Invert y-axis so positive concepts are at top
        ax.invert_yaxis()
        
        # Set x-axis limits with some padding, centered at 0
        all_values = combined_weights + combined_stds + [x for sublist in [label_all_weights[:, idx] for idx in combined_indices] for x in sublist]
        max_abs_weight = max([abs(val) for val in all_values])
        x_limit = max_abs_weight * 1.1
        ax.set_xlim(-x_limit, x_limit)
        
        # Add vertical line at x=0 (after setting xlim and before other plot elements)
        ax.axvline(x=0.0, color='black', linewidth=1.5, alpha=0.8, zorder=1)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add title only for Atelectasis plots
        if label == "Atelectasis":
            if "Standard_CBM" in method_name:
                method_display = "Standard CBM"
            elif "Improved_CBM" in method_name:
                method_display = "CheXomni CBM"
            else:
                method_display = method_name
            
            ax.set_title(method_display, fontsize=34, pad=10)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        safe_label = label.replace(' ', '_').replace('/', '_')
        output_file = f"{output_dir}/{method_name}_{safe_label}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_file}")
        
        # Print summary stats
        positive_count = len(df_positive)
        negative_count = len(df_negative)
        max_positive = df_positive['weight'].max() if positive_count > 0 else 0
        min_negative = df_negative['weight'].min() if negative_count > 0 else 0
        
        print(f"  {label}: {positive_count} positive, {negative_count} negative concepts")
        print(f"    Max positive: {max_positive:.4f}, Min negative: {min_negative:.4f}")
        print(f"    Showing top 6 positive + top 3 negative concepts with error bars and individual points")
        if len(df_pos_top6) > 0:
            print(f"    Top positive concept: {df_pos_top6.iloc[0]['concept'][:50]}...")
        if len(df_neg_top3) > 0:
            print(f"    Top negative concept: {df_neg_top3.iloc[0]['concept'][:50]}...")

def main():
    """Main function to run concept importance visualization"""
    print("=" * 60)
    print("CBM CONCEPT IMPORTANCE VISUALIZATION")
    print("=" * 60)
    
    # Paths
    checkpoint_path = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck"
    output_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/cbm_concept_importance_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load concepts
    concepts, concept_indices = load_concepts()
    
    # Load concept embeddings (needed for improved CBM)
    concept_embeddings = load_concept_embeddings(concept_indices)
    
    # Load and plot standard CBM (averaged across all seeds)
    print("\n" + "=" * 40)
    print("STANDARD CBM (AVERAGED ACROSS 20 SEEDS)")
    print("=" * 40)
    standard_weights, standard_stds, standard_all_weights, standard_labels = load_standard_cbm_weights_all_seeds(checkpoint_path)
    plot_concept_importance(standard_weights, standard_stds, standard_all_weights, concepts, standard_labels, 
                          "Standard_CBM_Averaged", output_dir)
    
    # Load and plot improved CBM (averaged across all seeds)
    print("\n" + "=" * 40)
    print("IMPROVED CBM (AVERAGED ACROSS 20 SEEDS)")
    print("=" * 40)
    improved_weights, improved_stds, improved_all_weights, improved_labels = load_improved_cbm_weights_all_seeds(checkpoint_path, concept_embeddings)
    plot_concept_importance(improved_weights, improved_stds, improved_all_weights, concepts, improved_labels, 
                          "Improved_CBM_Averaged", output_dir)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
