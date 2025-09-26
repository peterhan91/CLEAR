#!/usr/bin/env python3
"""
Image Retriever Script

Retrieves top-relevant images for a given text prompt from testing datasets 
(PadChest, VinDR-CXR, Indiana test sets) using various models.

Usage:
    python image_retriever.py --prompt "diffuse opacity in the right hemithorax due to layering pleural effusion" 
                             --datasets padchest vindrcxr indiana 
                             --model chexzero 
                             --top_k 100
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, InterpolationMode
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import json
from datetime import datetime
import shutil
import hashlib
import pickle

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model loading functions from benchmark modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))

# Import from cross_modal_retrieval module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cross_modal_retrieval"))

# Now import local modules after path setup
from clear import clip
try:
    from train import load_clip
except ImportError:
    print("⚠️ Warning: Could not import train module")

# Import LLM embedding generator
try:
    from get_embed import RadiologyEmbeddingGenerator
except ImportError:
    try:
        from concepts.get_embed import RadiologyEmbeddingGenerator
    except ImportError:
        print("⚠️ Warning: Could not import RadiologyEmbeddingGenerator")
        RadiologyEmbeddingGenerator = None

warnings.filterwarnings("ignore")

# Ensure required imports are available
try:
    import open_clip
except ImportError:
    print("⚠️ Warning: open_clip not available - some models may not work")
    open_clip = None

# Import necessary functions from the reference implementation
# We'll redefine them here to make this script self-contained
class CXRDataset(Dataset):
    """Dataset for loading CXR images from H5 file"""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.num_images = len(f['cxr'])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['cxr'][idx]  # numpy array, shape like (448, 448) or (320, 320)
            
        # Convert to 3-channel format exactly like CXRTestDataset
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        img = np.repeat(img, 3, axis=0)    # (3, H, W) - RGB format
        img = torch.from_numpy(img)        # Convert to tensor
        
        if self.transform:
            img = self.transform(img)
            
        return {'img': img, 'idx': idx}

class BiomedCLIPDataset(Dataset):
    """Dataset for BiomedCLIP that expects PIL Image input"""
    def __init__(self, h5_path, preprocess_fn):
        self.h5_path = h5_path
        self.preprocess = preprocess_fn
        with h5py.File(h5_path, 'r') as f:
            self.num_images = len(f['cxr'])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['cxr'][idx]  # numpy array
            
        # Convert to PIL Image (BiomedCLIP preprocessing expects PIL)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert to PIL and RGB (3 channels)
        img = Image.fromarray(img).convert('RGB')
        
        # Apply BiomedCLIP preprocessing
        if self.preprocess:
            img = self.preprocess(img)
            
        return {'img': img, 'idx': idx}

def load_chexzero_model_local(device):
    """Load CheXzero model with correct parameters matching the checkpoint"""
    try:
        from model import CLIP
        
        # Try to find CheXzero checkpoint
        possible_paths = [
            "external_sota/chexzero/checkpoints/best_64_5e-05_original_22000_0.864.pt",
            "/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Warning: No CheXzero checkpoint found. Using OpenAI CLIP pretrained model.")
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
        else:
            print(f"Loading CheXzero model from: {model_path}")
            # Use parameters that match the checkpoint file
            params = {
                'embed_dim': 512,
                'image_resolution': 224,
                'vision_layers': 12,
                'vision_width': 768,
                'vision_patch_size': 32,
                'context_length': 77, 
                'vocab_size': 49408,
                'transformer_width': 512,
                'transformer_heads': 8,
                'transformer_layers': 12
            }
            model = CLIP(**params)
            # Load the checkpoint
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Move model to the specified device
        model = model.to(device)
        return model, None
    except Exception as e:
        print(f"Error loading CheXzero model: {e}")
        raise

def load_biomedclip_model_local(device):
    """Load BiomedCLIP model and return with its built-in preprocessing"""
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        model, preprocess = create_model_from_pretrained(model_name)
        tokenizer = get_tokenizer(model_name)
        
        # Create wrapper
        class BiomedCLIPWrapper:
            def __init__(self, model, tokenizer, preprocess):
                self.model = model
                self.tokenizer = tokenizer
                self.preprocess = preprocess
                
            def encode_image(self, images):
                return self.model.encode_image(images)
            
            def encode_text(self, text_tokens):
                return self.model.encode_text(text_tokens)
                
            def tokenize(self, texts, context_length=256):
                return self.tokenizer(texts, context_length=context_length)
                
            def eval(self):
                self.model.eval()
                return self
                
            def to(self, device):
                self.model = self.model.to(device)
                return self
        
        return BiomedCLIPWrapper(model, tokenizer, preprocess), preprocess
    except Exception as e:
        print(f"Error loading BiomedCLIP model: {e}")
        raise

def load_openai_clip_model_local(device):
    """Load OpenAI CLIP model and return with its built-in preprocessing"""
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        return model, preprocess
    except Exception as e:
        print(f"Error loading OpenAI CLIP model: {e}")
        raise

def load_custom_dinov2_model_local(device):
    """Load custom trained DINOv2-based CLIP model"""
    try:
        # Load the custom trained model
        model_path = "checkpoints/dinov2-multi-v1.0_vitb/best_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Custom model not found: {model_path}")
        
        print(f"Loading custom DINOv2 model from: {model_path}")
        model = load_clip(
            model_path=model_path,
            pretrained=False,
            context_length=77,
            use_dinov2=True,
            dinov2_model_name='dinov2_vitb14'
        )
        model = model.to(device).eval()
        return model, None
    except Exception as e:
        print(f"Error loading custom DINOv2 model: {e}")
        raise

def get_image_embeddings_chexzero(model, images, device):
    """Get image embeddings using CheXzero model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_chexzero(model, texts, device, batch_size=32):
    """Get text embeddings using CheXzero model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = clip.tokenize(batch_texts, context_length=77).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_image_embeddings_biomedclip(model, images, device):
    """Get image embeddings using BiomedCLIP model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_biomedclip(model, texts, device, batch_size=32):
    """Get text embeddings using BiomedCLIP model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = model.tokenize(batch_texts, context_length=256).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_image_embeddings_openai_clip(model, images, device):
    """Get image embeddings using OpenAI CLIP model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_openai_clip(model, texts, device, batch_size=32):
    """Get text embeddings using OpenAI CLIP model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = clip.tokenize(batch_texts, context_length=77).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_image_embeddings_custom_dinov2(model, images, device):
    """Get image embeddings using custom DINOv2 model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        # L2 normalize features (like in dot_products_testset.py)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_custom_dinov2(model, texts, device, batch_size=32):
    """Get text embeddings using custom DINOv2 model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = clip.tokenize(batch_texts, context_length=77).to(device)
            text_features = model.encode_text(text_tokens)
            # L2 normalize features (like in dot_products_testset.py)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def ensure_cache_dir():
    """Ensure cache directory exists"""
    cache_dir = "model_auditing/cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cache_key(data, prefix=""):
    """Generate cache key from data"""
    if isinstance(data, dict):
        # Sort dict for consistent hashing
        data_str = str(sorted(data.items()))
    else:
        data_str = str(data)
    
    cache_key = hashlib.md5(data_str.encode()).hexdigest()
    return f"{prefix}_{cache_key}" if prefix else cache_key

def save_cached_embeddings(embeddings, cache_file):
    """Save embeddings to cache file"""
    ensure_cache_dir()
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"   💾 Cached embeddings saved: {cache_file}")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not save cache: {e}")

def load_cached_embeddings(cache_file):
    """Load embeddings from cache file"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"   🚀 Loaded cached embeddings: {cache_file}")
            return embeddings
    except Exception as e:
        print(f"   ⚠️ Warning: Could not load cache: {e}")
    return None

def get_image_embeddings_cached(model, model_name, dataset_name, dataset_config, device):
    """Get image embeddings with caching"""
    # Create cache key based on model, dataset, and parameters
    cache_data = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'h5_path': dataset_config['h5_path'],
        'resolution': 448 if model_name == 'custom_dinov2' else 224,
        'normalization': 'cxr_specific'
    }
    
    cache_key = get_cache_key(cache_data, "img_embeddings")
    cache_file = f"model_auditing/cache/{cache_key}.pkl"
    
    # Try to load from cache first
    cached_embeddings = load_cached_embeddings(cache_file)
    if cached_embeddings is not None:
        return cached_embeddings
    
    # Compute embeddings if not cached
    print(f"   🔄 Computing image embeddings (will be cached for future use)...")
    h5_path = dataset_config["h5_path"]
    
    # Create dataset and dataloader
    if model_name == "chexzero":
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
        ])
        dataset = CXRDataset(h5_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        get_image_embeddings = get_image_embeddings_chexzero
        
    elif model_name == "custom_dinov2":
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(448, interpolation=InterpolationMode.BICUBIC),
        ])
        dataset = CXRDataset(h5_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        get_image_embeddings = get_image_embeddings_custom_dinov2
        
    elif model_name in ["biomedclip", "openai_clip"]:
        # Need to get preprocess function again
        if model_name == "biomedclip":
            _, preprocess_fn = load_biomedclip_model_local(device)
            get_image_embeddings = get_image_embeddings_biomedclip
        else:
            _, preprocess_fn = load_openai_clip_model_local(device)
            get_image_embeddings = get_image_embeddings_openai_clip
            
        dataset = BiomedCLIPDataset(h5_path, preprocess_fn)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get image embeddings
    image_embeddings = []
    batch_indices = []
    
    for batch in tqdm(dataloader):
        batch_embeddings = get_image_embeddings(model, batch['img'], device)
        image_embeddings.append(batch_embeddings)
        batch_indices.extend(batch['idx'].tolist())
    
    image_embeddings = np.vstack(image_embeddings)
    image_embeddings = F.normalize(torch.from_numpy(image_embeddings), dim=-1).numpy()
    
    # Cache the results
    cache_result = {
        'embeddings': image_embeddings,
        'indices': np.array(batch_indices),
        'metadata': cache_data
    }
    save_cached_embeddings(cache_result, cache_file)
    
    return cache_result

def get_text_embeddings_cached(model, model_name, texts, device):
    """Get text embeddings with caching"""
    # Create cache key for text embeddings
    cache_data = {
        'model_name': model_name,
        'texts_hash': hashlib.md5(str(texts).encode()).hexdigest(),
        'num_texts': len(texts)
    }
    
    cache_key = get_cache_key(cache_data, "text_embeddings")
    cache_file = f"model_auditing/cache/{cache_key}.pkl"
    
    # Try to load from cache first
    cached_embeddings = load_cached_embeddings(cache_file)
    if cached_embeddings is not None:
        return cached_embeddings['embeddings']
    
    # Compute embeddings if not cached
    print(f"   🔄 Computing text embeddings (will be cached for future use)...")
    
    if model_name == "chexzero":
        text_embeddings = get_text_embeddings_chexzero(model, texts, device)
    elif model_name == "biomedclip":
        text_embeddings = get_text_embeddings_biomedclip(model, texts, device)
    elif model_name == "openai_clip":
        text_embeddings = get_text_embeddings_openai_clip(model, texts, device)
    elif model_name == "custom_dinov2":
        text_embeddings = get_text_embeddings_custom_dinov2(model, texts, device)
    
    # Cache the results
    cache_result = {
        'embeddings': text_embeddings,
        'metadata': cache_data
    }
    save_cached_embeddings(cache_result, cache_file)
    
    return text_embeddings

def clear_cache(cache_type="all"):
    """Clear cached data"""
    cache_dir = "model_auditing/cache"
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    if cache_type == "all":
        import shutil
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print("🧹 All cache cleared.")
    elif cache_type == "images":
        for file in os.listdir(cache_dir):
            if file.startswith("img_embeddings_"):
                os.remove(os.path.join(cache_dir, file))
        print("🧹 Image embeddings cache cleared.")
    elif cache_type == "text":
        for file in os.listdir(cache_dir):
            if file.startswith("text_embeddings_"):
                os.remove(os.path.join(cache_dir, file))
        print("🧹 Text embeddings cache cleared.")

def get_cache_info():
    """Get information about cached data"""
    cache_dir = "model_auditing/cache"
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    files = os.listdir(cache_dir)
    img_files = [f for f in files if f.startswith("img_embeddings_")]
    text_files = [f for f in files if f.startswith("text_embeddings_")]
    
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)
    
    print(f"📊 Cache Statistics:")
    print(f"   📁 Cache directory: {cache_dir}")
    print(f"   🖼️  Image embedding files: {len(img_files)}")
    print(f"   📝 Text embedding files: {len(text_files)}")
    print(f"   💾 Total cache size: {total_size / (1024**2):.2f} MB")
    
    if img_files:
        print(f"\n🖼️  Image Cache Files:")
        for f in img_files[:5]:  # Show first 5
            size = os.path.getsize(os.path.join(cache_dir, f)) / (1024**2)
            print(f"     {f} ({size:.2f} MB)")
        if len(img_files) > 5:
            print(f"     ... and {len(img_files) - 5} more")

def load_dataset_info(dataset_name):
    """Load dataset paths and metadata"""
    dataset_configs = {
        "padchest": {
            "h5_path": "data/padchest_test.h5",
            "csv_path": "data/padchest_test.csv",
            "image_id_col": "ImageID",
            "name": "PadChest Test"
        },
        "vindrcxr": {
            "h5_path": "data/vindrcxr_test.h5", 
            "csv_path": "data/vindrcxr_test.csv",
            "image_id_col": "image_id",
            "name": "VinDR-CXR Test"
        },
        "indiana": {
            "h5_path": "data/indiana_test.h5",
            "csv_path": "data/indiana_test.csv", 
            "image_id_col": "filename",
            "name": "Indiana Test"
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    # Verify files exist
    for path_key in ["h5_path", "csv_path"]:
        if not os.path.exists(config[path_key]):
            raise FileNotFoundError(f"Dataset file not found: {config[path_key]}")
    
    return config

def copy_images_from_h5(h5_path, indices, output_dir, dataset_name, prefix="img"):
    """Copy images from H5 file to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        for i, idx in enumerate(indices):
            img = f['cxr'][idx]
            
            # Convert to uint8 if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Save as PNG
            img_pil = Image.fromarray(img)
            filename = f"{prefix}_{dataset_name}_{idx:06d}_rank_{i+1:03d}.png"
            img_pil.save(os.path.join(output_dir, filename))

def retrieve_images_for_prompt(prompt: str, datasets: List[str], model_name: str, 
                              top_k: int = 100, device: str = 'auto'):
    """
    Retrieve top-k most relevant images for a given text prompt
    
    Args:
        prompt: Text prompt to search for
        datasets: List of dataset names to search in
        model_name: Model to use for embedding
        top_k: Number of top images to retrieve
        device: Device to run on
    
    Returns:
        Dictionary with results for each dataset
    """
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔍 Retrieving images for prompt: '{prompt}'")
    print(f"📊 Datasets: {datasets}")
    print(f"🤖 Model: {model_name}")
    print(f"🔢 Top-K: {top_k}")
    print(f"💻 Device: {device}")
    
    # Load model
    print(f"\n🔄 Loading {model_name} model...")
    if model_name == "chexzero":
        model, _ = load_chexzero_model_local(device)
        get_image_embeddings = get_image_embeddings_chexzero
        get_text_embeddings = get_text_embeddings_chexzero
        
    elif model_name == "biomedclip":
        model, preprocess_fn = load_biomedclip_model_local(device)
        model.to(device)
        get_image_embeddings = get_image_embeddings_biomedclip
        get_text_embeddings = get_text_embeddings_biomedclip
        
    elif model_name == "openai_clip":
        model, preprocess_fn = load_openai_clip_model_local(device)
        model.to(device)
        get_image_embeddings = get_image_embeddings_openai_clip
        get_text_embeddings = get_text_embeddings_openai_clip
        
    elif model_name == "custom_dinov2":
        model, _ = load_custom_dinov2_model_local(device)
        get_image_embeddings = get_image_embeddings_custom_dinov2
        get_text_embeddings = get_text_embeddings_custom_dinov2
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get text embedding for the prompt (with caching)
    print(f"📝 Computing text embedding for prompt...")
    text_embedding = get_text_embeddings_cached(model, model_name, [prompt], device)
    text_embedding = F.normalize(torch.from_numpy(text_embedding), dim=-1).numpy()
    
    results = {}
    all_similarities = []
    all_indices = []
    all_dataset_labels = []
    
    # Process each dataset
    for dataset_name in datasets:
        print(f"\n📂 Processing {dataset_name} dataset...")
        
        # Load dataset info
        dataset_config = load_dataset_info(dataset_name)
        csv_path = dataset_config["csv_path"]
        
        # Load CSV for metadata
        df = pd.read_csv(csv_path)
        image_ids = df[dataset_config["image_id_col"]].tolist()
        
        print(f"   📊 Loaded {len(image_ids)} images")
        
        # Get image embeddings (with caching)
        print(f"   🖼️  Getting image embeddings...")
        cache_result = get_image_embeddings_cached(model, model_name, dataset_name, dataset_config, device)
        
        image_embeddings = cache_result['embeddings']
        batch_indices = cache_result['indices']
        
        # Compute similarities
        print(f"   🔍 Computing similarities...")
        similarities = (image_embeddings @ text_embedding.T).flatten()
        
        # Store results for this dataset
        dataset_indices = np.array(batch_indices)
        results[dataset_name] = {
            'similarities': similarities,
            'indices': dataset_indices,
            'image_ids': [image_ids[i] for i in dataset_indices],
            'dataset_config': dataset_config
        }
        
        # Add to global lists for combined ranking
        all_similarities.extend(similarities)
        all_indices.extend([(dataset_name, idx) for idx in dataset_indices])
        all_dataset_labels.extend([dataset_name] * len(similarities))
    
    # Combine results and get top-k overall
    print(f"\n🏆 Finding top-{top_k} images across all datasets...")
    all_similarities = np.array(all_similarities)
    top_k_global_indices = np.argsort(all_similarities)[::-1][:top_k]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_clean = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()
    prompt_clean = prompt_clean.replace(' ', '_')[:50]  # Limit length
    output_dir = f"model_auditing/results/retrieved_images_{prompt_clean}_{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save top-k images and create summary
    top_results = []
    saved_count = 0
    
    print(f"💾 Saving top-{top_k} images to: {output_dir}")
    
    for rank, global_idx in enumerate(top_k_global_indices):
        dataset_name, local_idx = all_indices[global_idx]
        similarity = all_similarities[global_idx]
        
        # Get dataset config and image ID
        dataset_config = results[dataset_name]['dataset_config']
        df = pd.read_csv(dataset_config['csv_path'])
        image_id = df.iloc[local_idx][dataset_config['image_id_col']]
        
        # Copy image from H5 file
        try:
            copy_images_from_h5(
                dataset_config['h5_path'], 
                [local_idx], 
                output_dir, 
                dataset_name,
                f"rank_{rank+1:03d}"
            )
            saved_count += 1
            
            # Record result
            top_results.append({
                'rank': rank + 1,
                'similarity': float(similarity),
                'dataset': dataset_name,
                'image_id': image_id,
                'local_index': int(local_idx),
                'filename': f"rank_{rank+1:03d}_{dataset_name}_{local_idx:06d}_rank_{rank+1:03d}.png"
            })
            
        except Exception as e:
            print(f"   ⚠️ Error saving image {rank+1}: {e}")
    
    # Save summary JSON
    summary = {
        'prompt': prompt,
        'model': model_name,
        'datasets': datasets,
        'top_k': top_k,
        'timestamp': timestamp,
        'total_images_searched': len(all_similarities),
        'images_saved': saved_count,
        'output_directory': output_dir,
        'top_results': top_results,
        'dataset_stats': {
            name: {
                'total_images': len(data['similarities']),
                'max_similarity': float(np.max(data['similarities'])),
                'mean_similarity': float(np.mean(data['similarities'])),
                'top_5_count': sum(1 for r in top_results[:5] if r['dataset'] == name),
                'top_10_count': sum(1 for r in top_results[:10] if r['dataset'] == name),
                'top_50_count': sum(1 for r in top_results[:50] if r['dataset'] == name)
            }
            for name, data in results.items()
        }
    }
    
    summary_file = os.path.join(output_dir, "retrieval_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Retrieval complete!")
    print(f"   📊 Searched {len(all_similarities)} images across {len(datasets)} datasets")
    print(f"   💾 Saved {saved_count} images to {output_dir}")
    print(f"   📋 Summary saved to {summary_file}")
    
    # Print top-10 summary
    print(f"\n🏆 Top-10 Results:")
    for i, result in enumerate(top_results[:10]):
        print(f"   {i+1:2d}. {result['dataset']:10s} | Similarity: {result['similarity']:.4f} | ID: {result['image_id']}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Retrieve top-relevant images for a text prompt")
    parser.add_argument('--prompt', type=str,
                        help='Text prompt to search for (e.g., "diffuse opacity in the right hemithorax due to layering pleural effusion")')
    parser.add_argument('--datasets', nargs='+', 
                        default=['padchest', 'vindrcxr', 'indiana'],
                        choices=['padchest', 'vindrcxr', 'indiana'],
                        help='Datasets to search in')
    parser.add_argument('--model', type=str, 
                        default='chexzero',
                        choices=['chexzero', 'biomedclip', 'openai_clip', 'custom_dinov2'],
                        help='Model to use for embeddings')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top images to retrieve')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    parser.add_argument('--clear_cache', type=str, choices=['all', 'images', 'text'], 
                        help='Clear cached data before running')
    parser.add_argument('--cache_info', action='store_true',
                        help='Show cache information and exit')
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.cache_info:
        get_cache_info()
        return 0
    
    if args.clear_cache:
        clear_cache(args.clear_cache)
        if not args.prompt:  # If only clearing cache, exit
            return 0
    
    # Validate prompt is provided for retrieval
    if not args.prompt:
        parser.error("--prompt is required for image retrieval")
    
    # Run retrieval
    try:
        summary = retrieve_images_for_prompt(
            prompt=args.prompt,
            datasets=args.datasets,
            model_name=args.model,
            top_k=args.top_k,
            device=args.device
        )
        
        print(f"\n🎉 Image retrieval completed successfully!")
        print(f"Results saved to: {summary['output_directory']}")
        
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
