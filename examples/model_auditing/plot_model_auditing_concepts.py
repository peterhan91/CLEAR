#!/usr/bin/env python3
"""
Model Auditing Concept Analysis Plot
Visualization for model auditing results showing concept differences between clusters
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
import textwrap
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

def load_model_auditing_results(json_path: str) -> Dict:
    """Load model auditing results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_concept_differences(data: Dict, top_k: int = 25, output_dir: str = "results/concept_plots"):
    """Plot concept differences following the CBM script style"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract top concepts data
    top_concepts = data['top_concepts'][:top_k]  # Take top_k concepts
    
    # Extract data for plotting
    concept_texts = [item['concept_text'] for item in top_concepts]
    difference_scores = [item['difference_score'] for item in top_concepts]
    low_cluster_scores = [item['low_cluster_score'] for item in top_concepts]
    high_cluster_scores = [item['high_cluster_score'] for item in top_concepts]
    
    # Wrap long concept names into multiple lines instead of truncating
    max_chars_per_line = 30  # Maximum characters per line
    concept_texts_wrapped = []
    for concept in concept_texts:
        if len(concept) <= max_chars_per_line:
            concept_texts_wrapped.append(concept)
        else:
            # Split into multiple lines
            wrapped = textwrap.fill(concept, width=max_chars_per_line, break_long_words=False)
            concept_texts_wrapped.append(wrapped)
    
    # Convert to numpy arrays for easier manipulation
    difference_scores = np.array(difference_scores)
    
    # Create figure with larger size to accommodate longer concept names
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(10, 12))
    
    # Color negative scores in steelblue, positive scores in indianred (same as CBM script)
    colors = ['steelblue' if score < 0 else 'indianred' for score in difference_scores]
    
    # Create horizontal bar plot with wrapped concept names
    sns.barplot(x=difference_scores, y=concept_texts_wrapped, palette=colors, 
                edgecolor='black', linewidth=2.0, alpha=1.0, ax=ax)
    # Keep axes visible instead of despining
    sns.despine(ax=ax, trim=False, left=False, bottom=False)
    
    # Set labels and title
    ax.set_xlabel('Expression difference', fontsize=26)
    ax.set_ylabel('', fontsize=14)
    
    dataset = data.get('dataset', 'Unknown')
    cluster = data.get('cluster', 'Unknown')
    # ax.set_title(f'Top {top_k} Most Different Concepts\n{dataset.capitalize()} Dataset - Cluster {cluster} Analysis', 
    #              fontsize=14, fontweight='bold')
    
    # X-axis formatting - start from 0.0 as requested
    score_min = 0.0  # Force start from 0
    score_max = difference_scores.max()
    
    # Set limits starting from 0.0
    x_min = 0.015
    x_max = score_max + 0.001
    ax.set_xlim(x_min, x_max)
    
    # Generate tick marks starting from 0.0
    tick_start = 0.015
    tick_end = min(1.0, np.ceil(score_max * 1000) / 1000)  # Round up to nearest 0.001
    
    # Create ticks with appropriate spacing for smaller values
    tick_spacing = 0.005  # Smaller spacing for difference scores
    xticks = np.arange(tick_start, tick_end + 0.0001, tick_spacing)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.3f}" for x in xticks])
    
    # Make concept labels (y-axis) larger as requested
    ax.tick_params(axis='y', rotation=0, labelsize=24)  # Increased from 12 to 16
    ax.tick_params(axis='x', labelsize=24)  # Slightly increased x-axis too
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_dir}/model_auditing_{dataset}_cluster{cluster}_top{top_k}_concepts.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved concept difference plot: {plot_filename}")
    return plot_filename

def plot_cluster_comparison(data: Dict, top_k: int = 25, output_dir: str = "results/concept_plots"):
    """Plot side-by-side comparison of low vs high cluster scores"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract top concepts data
    top_concepts = data['top_concepts'][:top_k]
    
    # Extract data for plotting
    concept_texts = [item['concept_text'] for item in top_concepts]
    low_cluster_scores = np.array([item['low_cluster_score'] for item in top_concepts])
    high_cluster_scores = np.array([item['high_cluster_score'] for item in top_concepts])
    
    # Wrap long concept names into multiple lines instead of truncating
    max_chars_per_line = 45  # Slightly shorter for side-by-side plot
    concept_texts_wrapped = []
    for concept in concept_texts:
        if len(concept) <= max_chars_per_line:
            concept_texts_wrapped.append(concept)
        else:
            # Split into multiple lines
            wrapped = textwrap.fill(concept, width=max_chars_per_line, break_long_words=False)
            concept_texts_wrapped.append(wrapped)
    
    # Create figure with two subplots, larger to accommodate wrapped text
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(24, 12))
    
    # Plot low cluster scores (left)
    sns.barplot(x=low_cluster_scores, y=concept_texts_wrapped, 
                color='lightcoral', edgecolor='black', alpha=0.8, ax=ax1)
    sns.despine(ax=ax1, trim=False, left=False, bottom=False)  # Keep axes visible
    ax1.set_xlabel('Concept Activation Score', fontsize=14)
    ax1.set_ylabel('Medical Concepts', fontsize=14)
    ax1.set_title('Low Cluster\nConcept Activations', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=14)  # Larger y-axis labels
    ax1.tick_params(axis='x', labelsize=11)
    
    # Plot high cluster scores (right)
    sns.barplot(x=high_cluster_scores, y=concept_texts_wrapped, 
                color='lightblue', edgecolor='black', alpha=0.8, ax=ax2)
    sns.despine(ax=ax2, trim=False, left=False, bottom=False)  # Keep axes visible
    ax2.set_xlabel('Concept Activation Score', fontsize=14)
    ax2.set_ylabel('')  # No y-label for right plot
    ax2.set_title('High Cluster\nConcept Activations', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)  # Larger y-axis labels
    ax2.tick_params(axis='x', labelsize=11)
    
    # Synchronize x-axis limits
    all_scores = np.concatenate([low_cluster_scores, high_cluster_scores])
    x_min = all_scores.min() - 0.01
    x_max = all_scores.max() + 0.01
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Add overall title
    dataset = data.get('dataset', 'Unknown')
    cluster = data.get('cluster', 'Unknown')
    fig.suptitle(f'Cluster Comparison: Top {top_k} Most Different Concepts\n{dataset.capitalize()} Dataset - Cluster {cluster} Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_dir}/model_auditing_{dataset}_cluster{cluster}_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster comparison plot: {plot_filename}")
    return plot_filename

def generate_summary_stats(data: Dict) -> Dict:
    """Generate summary statistics for the model auditing results"""
    top_concepts = data['top_concepts']
    
    difference_scores = [item['difference_score'] for item in top_concepts]
    low_cluster_scores = [item['low_cluster_score'] for item in top_concepts]
    high_cluster_scores = [item['high_cluster_score'] for item in top_concepts]
    
    stats = {
        'num_concepts': len(top_concepts),
        'difference_score_stats': {
            'mean': np.mean(difference_scores),
            'std': np.std(difference_scores),
            'min': np.min(difference_scores),
            'max': np.max(difference_scores)
        },
        'low_cluster_stats': {
            'mean': np.mean(low_cluster_scores),
            'std': np.std(low_cluster_scores),
            'min': np.min(low_cluster_scores),
            'max': np.max(low_cluster_scores)
        },
        'high_cluster_stats': {
            'mean': np.mean(high_cluster_scores),
            'std': np.std(high_cluster_scores),
            'min': np.min(high_cluster_scores),
            'max': np.max(high_cluster_scores)
        }
    }
    
    return stats

def main():
    """Main function for model auditing concept visualization"""
    print("=== Model Auditing Concept Analysis ===")
    
    # Path to model auditing results
    json_path = "/home/than/DeepLearning/cxr_concept/CheXzero/model_auditing/results/padchest_pneumonia_cluster5.json"
    output_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/model_auditing/results/concept_plots"
    
    # Load model auditing results
    print(f"Loading model auditing results from: {json_path}")
    data = load_model_auditing_results(json_path)
    
    print(f"Dataset: {data.get('dataset', 'Unknown')}")
    print(f"Cluster: {data.get('cluster', 'Unknown')}")
    print(f"Number of concepts: {len(data['top_concepts'])}")
    
    # Generate summary statistics
    stats = generate_summary_stats(data)
    print(f"\nSummary Statistics:")
    print(f"Difference Score - Mean: {stats['difference_score_stats']['mean']:.4f}, "
          f"Std: {stats['difference_score_stats']['std']:.4f}")
    print(f"Low Cluster Score - Mean: {stats['low_cluster_stats']['mean']:.4f}, "
          f"Std: {stats['low_cluster_stats']['std']:.4f}")
    print(f"High Cluster Score - Mean: {stats['high_cluster_stats']['mean']:.4f}, "
          f"Std: {stats['high_cluster_stats']['std']:.4f}")
    
    # Create plots
    print("\nGenerating concept difference plot...")
    plot_concept_differences(data, top_k=25, output_dir=output_dir)
    
    print("Generating cluster comparison plot...")
    plot_cluster_comparison(data, top_k=25, output_dir=output_dir)
    
    print(f"\n✅ Model auditing concept analysis complete! Check {output_dir}/ for plots.")

if __name__ == "__main__":
    main() 