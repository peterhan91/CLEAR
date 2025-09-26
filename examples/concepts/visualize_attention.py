#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
from clear import zero_shot

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

def extract_dinov2_vision_encoder(clip_model):
    """Extract the DinoV2 vision encoder from CLIP model"""
    # The vision encoder is wrapped in DinoV2Visual, get the actual backbone
    vision_encoder = clip_model.visual.backbone
    return vision_encoder

def get_attention_maps(model, images, layer_idx=-1):
    """Extract attention maps from DinoV2 model"""
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # For DinoV2, we need to capture the attention weights
        # DinoV2 attention modules return different formats
        if hasattr(module, 'get_attention_map'):
            # Some implementations have this method
            attn = module.get_attention_map()
        else:
            # Try to get attention from the forward pass
            # The attention weights are usually the second element or accessible via attn_weights
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
            else:
                # For some DinoV2 implementations, we need to modify the attention module
                # Let's store the input and compute attention manually
                return
        
        if attn is not None:
            attention_weights.append(attn.cpu())
    
    # Register hook to the specified transformer layer
    if hasattr(model, 'blocks'):
        # DinoV2 structure
        target_layer = model.blocks[layer_idx].attn
    else:
        return None
    
    handle = target_layer.register_forward_hook(attention_hook)
    
    try:
        with torch.no_grad():
            # For DinoV2, we need to enable attention map saving
            if hasattr(model, 'get_intermediate_layers'):
                # Use DinoV2's method to get features with attention
                output = model.get_intermediate_layers(images, n=1, return_class_token=True)
            else:
                output = model(images)
        
        if attention_weights:
            return attention_weights[0]  # [batch, num_heads, seq_len, seq_len]
        else:
            return get_attention_maps_alternative(model, images, layer_idx)
    finally:
        handle.remove()

def get_attention_maps_alternative(model, images, layer_idx=-1):
    """Alternative method to get attention maps by modifying the attention computation"""
    attention_maps = []
    
    def modified_attention_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Store attention weights
        attention_maps.append(attn.cpu())
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    # Temporarily replace the attention forward method
    target_attn = model.blocks[layer_idx].attn
    original_forward = target_attn.forward
    target_attn.forward = modified_attention_forward.__get__(target_attn, target_attn.__class__)
    
    try:
        with torch.no_grad():
            _ = model(images)
        
        if attention_maps:
            return attention_maps[0]
        else:
            return None
    finally:
        # Restore original forward method
        target_attn.forward = original_forward

def generate_attention_maps(dataset_name, max_images=None):
    """Generate attention maps for specified test set"""
    
    print(f"=== Generating {dataset_name.upper()} Attention Maps ===")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    
    print("Loading CLIP model and extracting DinoV2...")
    clip_model = load_clip(
        model_path="../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    clip_model = clip_model.to('cuda').eval()
    
    # Extract DinoV2 vision encoder
    vision_encoder = extract_dinov2_vision_encoder(clip_model)
    
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    test_dataset = zero_shot.CXRTestDataset(
        img_path=dataset_config['cxr_filepath'],
        transform=transform,
    )
    
    # Limit number of images if specified
    if max_images is not None:
        total_images = min(max_images, len(test_dataset))
        indices = list(range(total_images))
    else:
        total_images = len(test_dataset)
        indices = list(range(total_images))
    
    print(f"Processing {total_images} images from {dataset_name.upper()} test set")
    
    # Create output directory
    output_dir = "results/attention_maps"
    os.makedirs(output_dir, exist_ok=True)
    
    # ACCUMULATE ALL ATTENTION DATA ACROSS BATCHES
    all_attention_data = {}
    
    # Process images in small batches to manage memory
    batch_size = 4  # Small batch size for attention map extraction
    
    for start_idx in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_images)
        batch_indices = indices[start_idx:end_idx]
        
        # Load batch of images
        batch_images = []
        for img_idx in batch_indices:
            sample = test_dataset[img_idx]
            batch_images.append(sample['img'])
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to('cuda')
        
        # Extract attention maps
        try:
            attention_maps = get_attention_maps(vision_encoder, batch_tensor)
            
            if attention_maps is not None:
                # Debug token sequence structure for the first batch
                if start_idx == 0:
                    debug_token_sequence(attention_maps, patch_size=14, image_size=448)
                
                # Process attention maps and ADD TO ACCUMULATOR
                batch_attention_data = process_attention_batch(
                    attention_maps, 
                    batch_indices,
                    patch_size=14,  # DinoV2 ViT-B/14
                    image_size=448
                )
                # Add to global accumulator
                all_attention_data.update(batch_attention_data)
            
            # Clear memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Error processing batch {start_idx//batch_size}: {e}")
            continue
    
    print("Saving attention maps...")
    
    # NOW SAVE ALL DATA AT ONCE
    h5_file_path = os.path.join(output_dir, f"{dataset_name}_attention_maps.h5")
    
    with h5py.File(h5_file_path, 'w') as f:
        # Save all attention maps
        for key, data in all_attention_data.items():
            f.create_dataset(key, data=data, compression='gzip', compression_opts=9)
        
        # Calculate metadata
        num_images = len([k for k in all_attention_data.keys() if k.endswith('_heads')])
        
        # Save metadata as attributes
        f.attrs['dataset'] = dataset_name
        f.attrs['model'] = 'dinov2_vitb14'
        f.attrs['patch_size'] = 14
        f.attrs['patches_per_side'] = 32
        f.attrs['num_heads'] = 12
        f.attrs['attention_format'] = 'CLS_token_to_spatial_patches'
        f.attrs['register_tokens_handled'] = True
        f.attrs['register_detection'] = 'automatic'
        f.attrs['original_image_size'] = 448
        f.attrs['attention_resolution'] = "32x32"
        f.attrs['total_images'] = num_images
    
    print(f"✅ Completed! Processed {num_images} images")
    print(f"📁 Results saved to: {output_dir}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'total_images_processed': num_images,
        'model': 'dinov2_vitb14',
        'patch_size': 14,
        'image_resolution': 448,
        'attention_layers': 'last_layer',
        'output_format': 'h5_raw_patches',
        'register_tokens_handled': True,
        'register_detection': 'automatic',
        'fix_applied': 'horizontal_shift_correction_for_dinov2_registers'
    }
    
    import json
    with open(f"{output_dir}/{dataset_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir

def process_attention_batch(attention_maps, image_indices, patch_size=14, image_size=448):
    """Process a batch of attention maps and return data dictionary"""
    batch_size, num_heads, seq_len, _ = attention_maps.shape
    
    # Calculate feature map dimensions based on image size and patch size
    patches_per_side = image_size // patch_size  # 448 // 14 = 32
    expected_patches = patches_per_side * patches_per_side  # 32 * 32 = 1024
    
    # Detect register tokens in DINOv2 models
    # Common configurations: CLS + 4 registers + spatial patches
    num_register_tokens = 0
    
    if seq_len == expected_patches + 1:
        # Standard ViT: [CLS, patch1, patch2, ..., patchN]
        num_register_tokens = 0
        num_patches_side = patches_per_side
        has_cls = True
    elif seq_len == expected_patches + 5:
        # DINOv2 with 4 registers: [CLS, reg1, reg2, reg3, reg4, patch1, patch2, ..., patchN]
        num_register_tokens = 4
        num_patches_side = patches_per_side
        has_cls = True
        print(f"🔍 Detected DINOv2 with {num_register_tokens} register tokens")
    elif seq_len == expected_patches:
        # No CLS token: [patch1, patch2, ..., patchN]
        num_register_tokens = 0
        num_patches_side = patches_per_side  
        has_cls = False
    else:
        # Try to infer the structure
        # Check if it could be CLS + registers + patches
        remaining_tokens = seq_len - 1  # subtract CLS
        if remaining_tokens > expected_patches:
            # Likely has register tokens
            possible_register_count = remaining_tokens - expected_patches
            if possible_register_count <= 8:  # reasonable number of registers
                num_register_tokens = possible_register_count
                num_patches_side = patches_per_side
                has_cls = True
                print(f"🔍 Inferred {num_register_tokens} register tokens from sequence length")
            else:
                # Fallback: assume square spatial arrangement
                num_patches_side = int(np.sqrt(remaining_tokens))
                has_cls = True
                num_register_tokens = 0
        else:
            num_patches_side = int(np.sqrt(seq_len - 1)) if seq_len > expected_patches else int(np.sqrt(seq_len))
            has_cls = seq_len > expected_patches
            num_register_tokens = 0
    
    # Prepare data for this batch
    batch_attention_data = {}
    
    for batch_idx in range(batch_size):
        img_idx = image_indices[batch_idx]
        
        # Extract spatial patches
        expected_spatial_patches = num_patches_side * num_patches_side
        
        # Calculate the starting index for spatial patches
        # Structure: [CLS] + [registers] + [spatial patches]
        spatial_patch_start_idx = 1 + num_register_tokens  # Skip CLS + register tokens
        spatial_patch_end_idx = spatial_patch_start_idx + expected_spatial_patches
        
        if seq_len >= spatial_patch_end_idx:
            # Extract attention from CLS token to spatial patches only
            cls_attention = attention_maps[batch_idx, :, 0, spatial_patch_start_idx:spatial_patch_end_idx]
        else:
            print(f"⚠️ Warning: Not enough tokens for image {img_idx}. Expected {spatial_patch_end_idx}, got {seq_len}")
            continue
        
        # Store all heads for this image
        image_attention_heads = []
        
        for head_idx in range(num_heads):
            attn_map = cls_attention[head_idx].numpy().reshape(num_patches_side, num_patches_side)
            image_attention_heads.append(attn_map)
        
        # Convert to numpy array [num_heads, patch_height, patch_width]
        image_attention_array = np.stack(image_attention_heads, axis=0)
        
        # Also create an averaged attention map across all heads
        avg_attention = cls_attention.mean(dim=0).numpy()
        avg_attn_map = avg_attention.reshape(num_patches_side, num_patches_side)
        
        # Store both individual heads and averaged attention
        batch_attention_data[f'image_{img_idx:04d}_heads'] = image_attention_array
        batch_attention_data[f'image_{img_idx:04d}_averaged'] = avg_attn_map
    
    return batch_attention_data

def debug_token_sequence(attention_maps, patch_size=14, image_size=448):
    """Debug function to analyze token sequence structure"""
    batch_size, num_heads, seq_len, _ = attention_maps.shape
    
    patches_per_side = image_size // patch_size
    expected_patches = patches_per_side * patches_per_side
    
    print(f"\n🔍 Token Sequence Analysis:")
    print(f"   Attention matrix shape: {attention_maps.shape}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Expected spatial patches: {expected_patches}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Patch size: {patch_size}x{patch_size}")
    print(f"   Patches per side: {patches_per_side}")
    
    # Analyze sequence structure
    if seq_len == expected_patches + 1:
        print("   🎯 Structure: [CLS] + [spatial patches] (Standard ViT)")
    elif seq_len == expected_patches + 5:
        print("   🎯 Structure: [CLS] + [4 registers] + [spatial patches] (DINOv2 with registers)")
    elif seq_len == expected_patches:
        print("   🎯 Structure: [spatial patches] (No CLS token)")
    else:
        extra_tokens = seq_len - expected_patches - 1
        if extra_tokens > 0:
            print(f"   🎯 Structure: [CLS] + [{extra_tokens} registers/special tokens] + [spatial patches]")
        else:
            print("   ⚠️ Unexpected sequence structure")
    
    # Analyze attention patterns to confirm structure
    sample_attention = attention_maps[0, 0, 0, :].cpu().numpy()  # First head, CLS token attention
    
    print(f"\n🔍 Attention Pattern Analysis:")
    print(f"   CLS token attention to all tokens - stats:")
    print(f"   Min: {sample_attention.min():.6f}")
    print(f"   Max: {sample_attention.max():.6f}")
    print(f"   Mean: {sample_attention.mean():.6f}")
    print(f"   Std: {sample_attention.std():.6f}")
    
    # Look for potential register tokens (might have different attention patterns)
    if seq_len > expected_patches + 1:
        register_region = sample_attention[1:seq_len-expected_patches]
        spatial_region = sample_attention[seq_len-expected_patches:]
        
        print(f"   Potential register region [1:{seq_len-expected_patches}]:")
        print(f"     Mean attention: {register_region.mean():.6f}")
        print(f"     Std attention: {register_region.std():.6f}")
        
        print(f"   Spatial patch region [{seq_len-expected_patches}:{seq_len}]:")
        print(f"     Mean attention: {spatial_region.mean():.6f}")
        print(f"     Std attention: {spatial_region.std():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate attention maps for CXR test datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        choices=['vindrcxr_test', 'padchest_test', 'indiana_test'],
        default='vindrcxr_test',
        help='Dataset to generate attention maps for'
    )
    parser.add_argument('--max_images', type=int, default=50, 
                        help='Maximum number of images to process (default: 50 for testing)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing (default: 4)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Attention Map Generation")
    print(f"📊 Dataset: {args.dataset}")
    print(f"📊 Processing up to {args.max_images} images")
    
    try:
        output_dir = generate_attention_maps(args.dataset, max_images=args.max_images)
        print(f"\n✅ Attention map generation completed!")
        print(f"🎯 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during attention map generation: {e}")
        raise 
