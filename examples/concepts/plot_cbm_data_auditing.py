#!/usr/bin/env python3
"""
CBM Data Auditing Plot
Data auditing visualization for Concept Bottleneck Model using dinov2-clip
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
import torch
import h5py
import textwrap
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_clip

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_filtered_concepts():
    """Load filtered diagnostic concepts from CBM concepts file"""
    with open("cbm_concepts.json", 'r') as f:
        cbm_data = json.load(f)
    
    concepts = []
    concept_indices = []
    label_to_concepts = {}
    
    for label, concept_list in cbm_data.items():
        label_concepts = []
        for item in concept_list:
            concepts.append(item['concept'])
            concept_indices.append(item['concept_idx'])
            label_concepts.append(len(concepts) - 1)
        label_to_concepts[label] = label_concepts
    
    return concepts, concept_indices, label_to_concepts

@torch.no_grad()
def extract_concept_features(model, concepts):
    """Extract CLIP concept features"""
    from clear import clip
    concept_batch_size = 512
    all_concept_features = []
    
    for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
        batch_concepts = concepts[i:i+concept_batch_size]
        concept_tokens = clip.tokenize(batch_concepts, context_length=77).to(device)
        concept_features = model.encode_text(concept_tokens)
        concept_features /= concept_features.norm(dim=-1, keepdim=True)
        all_concept_features.append(concept_features.cpu())
        torch.cuda.empty_cache()
    
    return torch.cat(all_concept_features)

@torch.no_grad()
def extract_image_features_sample(model, dataset_path, num_samples=None):
    """Extract image features from a sample of the dataset"""
    with h5py.File(dataset_path, 'r') as h5_file:
        total_samples = h5_file['cxr'].shape[0]
        print(f"Total samples in dataset: {total_samples}")
        
        if num_samples is None or num_samples >= total_samples:
            # Use all samples
            sample_indices = np.arange(total_samples)
            print(f"Using all {total_samples} samples")
        else:
            # Use random sample
            sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            print(f"Using {num_samples} random samples")
        
        features = []
        for idx in tqdm(sample_indices, desc="Extracting image features"):
            img_data = np.array(h5_file['cxr'][idx])
            img_data = np.expand_dims(img_data, axis=0)
            img_data = np.repeat(img_data, 3, axis=0)
            img = torch.from_numpy(img_data).float().unsqueeze(0).to(device)
            
            # Normalize
            img = img / 255.0
            img = (img - 0.485) / 0.229  # ImageNet normalization
            
            img_features = model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            features.append(img_features.cpu())
            
        return torch.cat(features), sample_indices

def compute_concept_dot_products(image_features, concept_features, label_to_concepts):
    """Compute dot products between images and concepts for each label"""
    target_labels = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    dot_products = {}
    for label in target_labels:
        if label in label_to_concepts:
            concept_indices = label_to_concepts[label]
            label_concept_features = concept_features[concept_indices]
            
            # Compute dot products: [num_images, num_concepts_for_label]
            label_dots = image_features @ label_concept_features.T
            dot_products[label] = label_dots.numpy() if hasattr(label_dots, 'numpy') else label_dots
    
    return dot_products

def analyze_sample_images(image_features, concept_features, concepts, label_to_concepts, 
                         sampled_indices, num_samples=5, top_k=25, output_dir="results/cbm_auditing"):
    """Analyze top concepts for sample images in the style of concept_image_analysis.py"""
    os.makedirs(output_dir, exist_ok=True)
    
    target_labels = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    # Paths for PadChest data
    attention_path = "results/attention_maps/padchest_test_attention_maps.h5"
    cxr_dataset_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5"
    
    saved_count = 0
    # Use provided sampled indices to match with attention maps
    for sample_idx in range(min(num_samples, len(sampled_indices))):
        img_idx = sampled_indices[sample_idx]
        
        # Compute dot products for this image with all concepts
        img_feature = image_features[sample_idx]  # Feature for this sample
        all_dot_products = img_feature @ concept_features.T  # Shape: [num_concepts]
        all_dot_products = all_dot_products.numpy() if hasattr(all_dot_products, 'numpy') else all_dot_products
        
        # Get top concepts overall
        top_concept_indices = np.argsort(-all_dot_products)[:top_k]
        top_concepts = [concepts[i] for i in top_concept_indices]
        top_scores = all_dot_products[top_concept_indices]
        
        # Truncate long concept names
        max_words = 12
        top_concepts_truncated = [
            c if len(c.split()) <= max_words else " ".join(c.split()[:max_words]) + "..."
            for c in top_concepts
        ]
        
        # Find which labels this image might have (based on concept activations)
        predicted_labels = []
        for label in target_labels:
            if label in label_to_concepts:
                label_concept_indices = label_to_concepts[label]
                label_dots = all_dot_products[label_concept_indices]
                max_label_score = np.max(label_dots)
                if max_label_score > 0.3:  # Threshold for prediction
                    predicted_labels.append(f"{label} ({max_label_score:.3f})")
        
        # Check filtering criteria (only require strong predictions)
        has_strong_predictions = len(predicted_labels) > 0
        
        # Skip this sample if it doesn't have strong predictions
        if not has_strong_predictions:
            print(f"Skipping sample {sample_idx+1}/{num_samples} (image {img_idx}): no strong predictions")
            continue
        
        # Create attention overlay
        label_str = ', '.join(predicted_labels)
        overlay_img = create_attention_overlay(
            attention_path, 
            cxr_dataset_path, 
            img_idx, 
            head_idx=0, 
            text=None,
            image_key='cxr'
        )
        
        # Create visualization in the same style as concept_image_analysis.py
        fig, axs = plt.subplots(1, 2, dpi=300, figsize=(19, 8))
        
        # Left: Concepts barplot (same style as concept_image_analysis.py)
        # Color negative scores in steelblue, positive scores in indianred
        colors = ['steelblue' if score < 0 else 'indianred' for score in top_scores]
        sns.barplot(x=top_scores, y=top_concepts_truncated, palette=colors, 
                    edgecolor='black', alpha=1.0, ax=axs[0])
        sns.despine(ax=axs[0], trim=True, left=True, bottom=True)
        axs[0].set_xlabel('Concept Similarity Score', fontsize=12)
        axs[0].set_ylabel('Medical Concepts', fontsize=12)
        axs[0].set_title(f'Top {top_k} Most Important CBM Concepts\nPadChest Test - Image {img_idx}', 
                         fontsize=14, fontweight='bold')
        
        # X-axis formatting (handle negative scores)
        score_min = top_scores.min()
        score_max = top_scores.max()
        
        # Dynamic limits (allow negative scores)
        x_min = score_min - 0.01
        x_max = score_max + 0.01
        axs[0].set_xlim(x_min, x_max)
        
        # Generate appropriate tick marks based on actual data range
        if score_min < 0:
            # Include negative range
            tick_start = max(-1.0, np.floor(score_min * 20) / 20)  # Round down to nearest 0.05
            tick_end = min(1.0, np.ceil(score_max * 20) / 20)      # Round up to nearest 0.05
        else:
            # Only positive range (original behavior)
            tick_start = max(0.0, np.floor(score_min * 20) / 20)
            tick_end = min(1.0, np.ceil(score_max * 20) / 20)
        
        xticks = np.arange(tick_start, tick_end + 0.001, 0.05)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels([f"{x:.2f}" for x in xticks])
        
        # Add vertical line at x=0 if we have negative scores
        if score_min < 0 and score_max > 0:
            axs[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axs[0].tick_params(axis='y', rotation=0)
        
        # Right: CXR with attention overlay (same style as concept_image_analysis.py)
        axs[1].imshow(overlay_img)
        axs[1].axis('off')
        axs[1].set_title(f'CXR Image with CLS Attention Map\nHead 0 (Top: Original, Bottom: Attention Overlay)', 
                         fontsize=14, fontweight='bold')
        
        # Text below image (same style as concept_image_analysis.py)
        wrapped_text = "\n".join(textwrap.wrap(f'CBM Predictions: {label_str}', width=30))
        axs[1].text(
            0.5, -0.05, wrapped_text,
            transform=axs[1].transAxes,
            ha='center', va='top',
            fontsize=11, color='black',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7)
        )
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{output_dir}/cbm_sample_{saved_count:03d}_image_{img_idx:04d}_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        
        saved_count += 1
        print(f"Saved analysis {saved_count} for sample {sample_idx+1}/{num_samples} (image {img_idx}): {plot_filename}")
    
    print(f"\n✅ Filtering complete: Saved {saved_count} out of {num_samples} samples with strong predictions")

def plot_label_concept_distributions(dot_products, output_dir="results/cbm_auditing"):
    """Plot concept activation distributions for each of the 5 target labels"""
    os.makedirs(output_dir, exist_ok=True)
    
    target_labels = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, label in enumerate(target_labels):
        if label in dot_products:
            # Get max dot product per image for this label
            max_dots = np.max(dot_products[label], axis=1)
            
            # Plot distribution
            axes[i].hist(max_dots, bins=50, alpha=0.7, color=sns.color_palette("husl", 5)[i])
            axes[i].set_title(f'{label}\nMax Concept Activation Distribution', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Max Dot Product Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add stats
            mean_val = np.mean(max_dots)
            std_val = np.std(max_dots)
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.6, label=f'+1σ: {mean_val+std_val:.3f}')
            axes[i].legend(fontsize=9)
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.suptitle('CBM Concept Activation Analysis\nDot Product Distributions by Label', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cbm_concept_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cbm_concept_distributions.png")

def main():
    """Main function for CBM data auditing"""
    print("=== CBM Data Auditing ===")
    
    # Load CLIP model
    print("Loading dinov2-CLIP model...")
    model = load_clip(
        model_path="/home/than/DeepLearning/cxr_concept/CheXzero/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    ).to(device).eval()
    
    # Load filtered concepts
    print("Loading filtered CBM concepts...")
    concepts, concept_indices, label_to_concepts = load_filtered_concepts()
    print(f"Loaded {len(concepts)} concepts across {len(label_to_concepts)} labels")
    
    # Extract concept features
    print("Extracting concept features...")
    concept_features = extract_concept_features(model, concepts)
    
    # Extract image features from ALL samples in PadChest
    print("Extracting image features from all PadChest samples...")
    dataset_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5"
    image_features, sample_indices = extract_image_features_sample(model, dataset_path, num_samples=None)
    
    # Compute dot products
    print("Computing concept dot products...")
    dot_products = compute_concept_dot_products(image_features, concept_features, label_to_concepts)
    
    # Generate plots in the style of concept_image_analysis.py
    print("Generating sample image analyses (concept_image_analysis.py style)...")
    total_samples = len(sample_indices)
    analyze_sample_images(image_features, concept_features, concepts, label_to_concepts, 
                         sample_indices, num_samples=total_samples, top_k=25)
    
    print("Generating label distribution plots...")
    plot_label_concept_distributions(dot_products)
    
    print("CBM data auditing complete! Check results/cbm_auditing/ for plots.")

if __name__ == "__main__":
    main()
