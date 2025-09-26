#!/usr/bin/env python3
"""
Highly Important Concepts Analysis with CLS Attention Maps
Based on concept_image.ipynb cell 4

This script analyzes the most important concepts for CXR images and visualizes them
alongside the original images and their CLS attention maps.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import h5py
import json
import argparse
import textwrap
import warnings
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_dataset_config(dataset_name: str) -> Dict:
    """Get dataset-specific configuration"""
    configs = {
        'vindrcxr_test': {
            'csv_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv',
            'h5_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.h5',
            'attention_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/attention_maps/vindrcxr_test_attention_maps.h5',
            'dot_products_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/dot_products/vindrcxr_test_dot_products.npy',
            'image_key': 'cxr'
        },
        'padchest_test': {
            'csv_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv',
            'h5_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5',
            'attention_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/attention_maps/padchest_test_attention_maps.h5',
            'dot_products_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/dot_products/padchest_test_dot_products.npy',
            'image_key': 'cxr'
        },
        'indiana_test': {
            'csv_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv',
            'h5_path': '/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.h5',
            'attention_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/attention_maps/indiana_test_attention_maps.h5',
            'dot_products_path': '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/dot_products/indiana_test_dot_products.npy',
            'image_key': 'cxr'
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    return configs[dataset_name]


def load_original_cxr(cxr_dataset_path: str, image_idx: int, image_key: str = 'cxr') -> Image.Image:
    """Load the original CXR image from dataset"""
    with h5py.File(cxr_dataset_path, 'r') as f:
        img_array = np.array(f[image_key][image_idx])
        
        # Normalize to [0, 255] if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # Convert to PIL Image
        if len(img_array.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(img_array, mode='L')
        else:  # RGB
            pil_img = Image.fromarray(img_array)
        
        return pil_img


def create_attention_overlay(h5_path: str, cxr_dataset_path: str, image_idx: int, 
                           head_idx: int = 0, text: str = None, image_key: str = 'cxr') -> Image.Image:
    """Create attention overlay visualization with optional title"""
    # Load original CXR
    original_cxr = load_original_cxr(cxr_dataset_path, image_idx, image_key)

    # Load attention map
    with h5py.File(h5_path, 'r') as f:
        if head_idx == -1:  # Use averaged
            attention_data = np.array(f[f'image_{image_idx:04d}_averaged'][:])
        else:  # Use specific head
            attention_data = np.array(f[f'image_{image_idx:04d}_heads'][head_idx])

    # Convert to RGB if needed
    if original_cxr.mode == 'L':
        original_cxr = original_cxr.convert('RGB')

    # Create attention heatmap
    norm_data = (attention_data - attention_data.min()) / (attention_data.max() - attention_data.min())
    colored = cm.turbo(norm_data)[:, :, :3]
    rgb_array = (colored * 255).astype(np.uint8)

    attention_img = Image.fromarray(rgb_array).resize(original_cxr.size, Image.NEAREST)

    # Create overlay
    overlay = Image.blend(original_cxr, attention_img, 0.5).convert('RGB')
    original_cxr = original_cxr.convert('RGB')

    # Stack original and overlay vertically
    combined = Image.new('RGB', (original_cxr.width, original_cxr.height * 2))
    combined.paste(original_cxr, (0, 0))
    combined.paste(overlay, (0, original_cxr.height))

    # If no text, return combined image
    if text is None:
        return combined

    # Add title area
    title_height = 40
    total_img = Image.new('RGB', (combined.width, combined.height + title_height), color=(0, 0, 0))
    total_img.paste(combined, (0, title_height))

    # Draw title
    draw = ImageDraw.Draw(total_img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Use textbbox instead of deprecated textsize
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((total_img.width - text_width) // 2, (title_height - text_height) // 2)

    draw.text(position, text, fill="white", font=font)

    return total_img


def filter_concepts(concepts_list: List[str], sensitive_words: List[str] = None) -> List[int]:
    """Filter out concepts containing sensitive words"""
    if sensitive_words is None:
        sensitive_words = []
    
    filtered_indices = [
        idx for idx, concept in enumerate(concepts_list)
        if not any(sw.lower() in concept.lower() for sw in sensitive_words)
    ]
    
    return filtered_indices


def get_disease_labels(df_row, dataset_name: str) -> List[str]:
    """Extract disease labels from dataset row"""
    if dataset_name == 'vindrcxr_test':
        # VinDr-CXR: exclude metadata columns
        exclude_cols = ['image_id', 'name', 'Path', 'is_test']
        disease_cols = [col for col in df_row.index if col not in exclude_cols]
        labels = [col for col in disease_cols if df_row[col] == 1]
    elif dataset_name == 'padchest_test':
        # PadChest: exclude metadata columns
        exclude_cols = ['ImageID', 'name', 'Path', 'is_test']
        disease_cols = [col for col in df_row.index if col not in exclude_cols]
        labels = [col for col in disease_cols if df_row[col] == 1]
    elif dataset_name == 'indiana_test':
        # Indiana: exclude metadata columns
        exclude_cols = ['uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 
                       'indication', 'comparison', 'findings', 'impression']
        disease_cols = [col for col in df_row.index if col not in exclude_cols]
        labels = [col for col in disease_cols if df_row[col] == 1]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return labels


def should_skip_image(df_row, dataset_name: str) -> bool:
    """Check if an image should be skipped based on normal/no finding labels"""
    labels = get_disease_labels(df_row, dataset_name)
    
    # Define normal/no finding patterns for each dataset
    normal_patterns = {
        'vindrcxr_test': ['no finding'],
        'padchest_test': ['normal'],
        'indiana_test': ['normal']
    }
    
    if dataset_name not in normal_patterns:
        return False
    
    patterns = normal_patterns[dataset_name]
    
    # Skip if image has only normal/no finding labels
    if len(labels) == 0:
        return False  # No labels at all, don't skip
    
    # Check if all labels are normal/no finding
    normal_labels = []
    pathology_labels = []
    
    for label in labels:
        is_normal = any(pattern.lower() in label.lower() for pattern in patterns)
        if is_normal:
            normal_labels.append(label)
        else:
            pathology_labels.append(label)
    
    # Skip if only normal/no finding labels are present
    return len(pathology_labels) == 0 and len(normal_labels) > 0


def break_long_concepts(concepts_list: List[str], max_chars_per_line: int = 50) -> List[str]:
    """Break long concepts into 2 lines without truncating"""
    broken_concepts = []
    for concept in concepts_list:
        if len(concept) <= max_chars_per_line:
            broken_concepts.append(concept)
        else:
            # Find a good breaking point (prefer breaking at spaces)
            words = concept.split()
            line1 = ""
            line2 = ""
            
            # Build first line
            for word in words:
                if len(line1 + " " + word) <= max_chars_per_line:
                    line1 += (" " + word if line1 else word)
                else:
                    # Remaining words go to second line
                    remaining_words = words[len(line1.split()):]
                    line2 = " ".join(remaining_words)
                    break
            
            # Keep full text without truncation
            broken_concepts.append(f"{line1}\n{line2}")
    
    return broken_concepts


def analyze_image_concepts(dataset_name: str, image_idx: int = None, top_k: int = 10, 
                          max_words: int = 12, head_idx: int = 0, 
                          save_plot: bool = True, output_dir: str = "results/concept_analysis") -> Dict:
    """Analyze top concepts for a specific image with attention visualization"""
    
    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    
    # Load dataset
    df = pd.read_csv(config['csv_path'])
    
    # Load concept dot products
    dot_products = np.load(config['dot_products_path'])
    
    # Load concepts list
    concepts_list_path = '/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/dot_products/concepts_list.txt'
    with open(concepts_list_path, 'r') as f:
        concepts_list = [line.strip().split(': ', 1)[1] for line in f.readlines()]
    
    # Select image
    if image_idx is None:
        # Find a random valid image (not "normal" or "no finding" only)
        max_attempts = 1000  # Prevent infinite loop
        attempts = 0
        while attempts < max_attempts:
            image_idx = np.random.randint(0, len(df))
            row = df.iloc[image_idx]
            
            # Skip images with only normal/no finding labels
            if should_skip_image(row, dataset_name):
                attempts += 1
                continue
            else:
                break
        
        if attempts >= max_attempts:
            raise ValueError(f"Could not find a valid image after {max_attempts} attempts. Dataset may contain only normal/no finding images.")
        
        labels = get_disease_labels(row, dataset_name)
    else:
        row = df.iloc[image_idx]
        labels = get_disease_labels(row, dataset_name)
        
        # Check if the specified image should be skipped
        if should_skip_image(row, dataset_name):
            print(f"⚠️  Warning: Image {image_idx} has only normal/no finding labels: {labels}")
            print("Proceeding with analysis anyway since image was explicitly specified.")
    
    # Filter concepts
    sensitive_words = []  # Add any words to filter out if needed
    filtered_indices = filter_concepts(concepts_list, sensitive_words)
    
    # Get top concepts for this image
    dot_products_filtered = dot_products[image_idx][filtered_indices]
    concepts_filtered = [concepts_list[i] for i in filtered_indices]
    
    # Sort by importance (highest dot product first)
    sorted_indices_desc = np.argsort(-dot_products_filtered)[:top_k]
    dot_products_sorted = dot_products_filtered[sorted_indices_desc]
    concepts_sorted = [concepts_filtered[i] for i in sorted_indices_desc]
    
    # Break long concepts into 2 lines instead of truncating
    concepts_sorted = break_long_concepts(concepts_sorted)
    
    # Create attention overlay
    label_str = ', '.join(labels) if labels else 'No findings'
    overlay_img = create_attention_overlay(
        config['attention_path'], 
        config['h5_path'], 
        image_idx, 
        head_idx=head_idx, 
        text=None,
        image_key=config['image_key']
    )
    
    # Create visualization
    fig, axs = plt.subplots(1, 2, dpi=300, figsize=(19, 8))
    
    # Left: Concepts barplot
    sns.barplot(x=dot_products_sorted, y=concepts_sorted, color='indianred', 
                edgecolor='black', alpha=1.0, ax=axs[0])
    sns.despine(ax=axs[0], trim=True, left=True, bottom=True)
    axs[0].set_xlabel('Concept Similarity Score', fontsize=18)
    axs[0].set_ylabel('')  # Remove y-axis label
    # axs[0].set_title(f'Top {top_k} Most Important Concepts\n{dataset_name.replace("_", " ").title()} - Image {image_idx}', 
    #                  fontsize=14, fontweight='bold')
    # Increase font size for concept labels (y-axis tick labels)
    axs[0].tick_params(axis='y', labelsize=16)
    axs[0].tick_params(axis='x', labelsize=16)
    
    # X-axis formatting
    xticks = np.arange(0.2, 1.001, 0.05)
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels([f"{x:.2f}" for x in xticks])
    
    # Dynamic limits
    x_min = max(0, dot_products_sorted.min() - 0.01)
    x_max = min(1.0, dot_products_sorted.max() + 0.01)
    axs[0].set_xlim(x_min, x_max)
    axs[0].tick_params(axis='y', rotation=0)
    
    # Right: CXR with attention overlay
    axs[1].imshow(overlay_img)
    axs[1].axis('off')
    # axs[1].set_title(f'CXR Image with CLS Attention Map\nHead {head_idx} (Top: Original, Bottom: Attention Overlay)', 
    #                  fontsize=14, fontweight='bold')
    
    # Text below image
    wrapped_text = "\n".join(textwrap.wrap(f'Ground Truth Labels: {label_str}', width=60))
    axs[1].text(
        0.5, -0.05, wrapped_text,
        transform=axs[1].transAxes,
        ha='center', va='top',
        fontsize=14, color='black',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7)
    )
    
    plt.tight_layout()
    
    # Save plot if requested
    plot_filename = None
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"{output_dir}/{dataset_name}_image_{image_idx:04d}_concepts_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
    else:
        plt.show()
    
    # Return analysis results
    results = {
        'dataset_name': dataset_name,
        'image_idx': image_idx,
        'ground_truth_labels': labels,
        'top_concepts': concepts_sorted,
        'concept_scores': dot_products_sorted.tolist(),
        'attention_head': head_idx,
        'plot_saved': plot_filename
    }
    
    return results


def batch_analyze_concepts(dataset_name: str, num_images: int = 5, top_k: int = 30,
                          save_results: bool = True, output_dir: str = "results/concept_analysis") -> List[Dict]:
    """Analyze concepts for multiple random images (automatically skips normal/no finding images)"""
    
    # Get dataset configuration and calculate statistics
    config = get_dataset_config(dataset_name)
    df = pd.read_csv(config['csv_path'])
    
    # Count normal/no finding images
    normal_count = 0
    total_count = len(df)
    
    for idx in range(total_count):
        row = df.iloc[idx]
        if should_skip_image(row, dataset_name):
            normal_count += 1
    
    pathology_count = total_count - normal_count
    
    print(f"📊 Dataset Statistics for {dataset_name}:")
    print(f"   Total images: {total_count}")
    print(f"   Normal/No finding images: {normal_count} ({normal_count/total_count*100:.1f}%)")
    print(f"   Pathology images: {pathology_count} ({pathology_count/total_count*100:.1f}%)")
    print(f"   Analyzing {num_images} random pathology images...")
    print()
    
    all_results = []
    
    # Use tqdm for progress bar
    for i in tqdm(range(num_images), desc=f"Analyzing {dataset_name}", unit="images"):
        try:
            results = analyze_image_concepts(
                dataset_name=dataset_name,
                image_idx=None,  # Random selection (will skip normal/no finding)
                top_k=top_k,
                save_plot=True,
                output_dir=output_dir
            )
            all_results.append(results)
            
        except Exception as e:
            tqdm.write(f"❌ Error analyzing image {i}: {e}")
            continue
    
    # Save batch results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_filename = f"{output_dir}/{dataset_name}_batch_analysis_results.json"
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Batch results saved to: {results_filename}")
    
    return all_results


def test_specific_images():
    """Test specific images from padchest dataset with predefined indices"""
    test_indices = [2077, 491, 31, 6546, 153, 2110]
    output_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_analysis_demo"
    
    print(f"🚀 Testing specific images from PadChest dataset")
    print(f"📋 Image indices: {test_indices}")
    print(f"📁 Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in test_indices:
        try:
            print(f"🔍 Analyzing image {idx}...")
            results = analyze_image_concepts(
                dataset_name='padchest_test',
                image_idx=idx,
                top_k=10,
                head_idx=0,
                save_plot=True,
                output_dir=output_dir
            )
            print(f"✅ Completed image {idx}")
        except Exception as e:
            print(f"❌ Error analyzing image {idx}: {e}")
    
    print(f"\n🎉 Demo analysis completed!")


def main():
    parser = argparse.ArgumentParser(description="Analyze highly important concepts with CLS attention maps")
    parser.add_argument('--dataset', type=str, 
                       choices=['vindrcxr_test', 'padchest_test', 'indiana_test'],
                       help='Dataset to analyze')
    parser.add_argument('--image_idx', type=int, default=None,
                       help='Specific image index to analyze (default: random)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of images to analyze (default: 1)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top concepts to show (default: 10)')
    parser.add_argument('--head_idx', type=int, default=0,
                       help='Attention head to visualize (default: 0, use -1 for averaged)')
    parser.add_argument('--output_dir', type=str, default="results/concept_analysis",
                       help='Output directory for results')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with specific PadChest images')
    
    args = parser.parse_args()
    
    if args.demo:
        test_specific_images()
        return
    
    # Check if dataset is required for non-demo mode
    if not args.dataset:
        parser.error("--dataset is required when not using --demo")
    
    print(f"🚀 Starting Highly Important Concepts Analysis")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🏆 Top-K concepts: {args.top_k}")
    print(f"🧠 Attention head: {args.head_idx}")
    
    if args.batch_size == 1:
        # Single image analysis
        results = analyze_image_concepts(
            dataset_name=args.dataset,
            image_idx=args.image_idx,
            top_k=args.top_k,
            head_idx=args.head_idx,
            save_plot=True,
            output_dir=args.output_dir
        )
        print(f"\n✅ Analysis completed for image {results['image_idx']}")
    else:
        # Batch analysis
        results = batch_analyze_concepts(
            dataset_name=args.dataset,
            num_images=args.batch_size,
            top_k=args.top_k,
            output_dir=args.output_dir
        )
        print(f"\n✅ Batch analysis completed for {len(results)} images")


if __name__ == "__main__":
    main() 