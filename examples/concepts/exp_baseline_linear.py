#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import h5py
import sys
import random
import shutil
import pickle
import hashlib
from pathlib import Path

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache directory for extracted features
CACHE_DIR = Path("cache/baseline_features")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class CXRDataset(Dataset):
    """Dataset class for loading CXR images from HDF5 files"""
    def __init__(self, h5_path, num_samples, transform=None):
        self.h5_path = h5_path
        self.num_samples = num_samples
        self.transform = transform
        self.h5_file = h5py.File(h5_path, 'r')
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Access by integer index from 'cxr' dataset
        img_data = self.h5_file['cxr'][idx]  # shape: (320, 320)
        
        # Convert to 3-channel format
        img_data = np.expand_dims(img_data, axis=0)  # (1, 320, 320)
        img_data = np.repeat(img_data, 3, axis=0)    # (3, 320, 320)
        img = torch.from_numpy(img_data).float()
        
        if self.transform:
            img = self.transform(img)
            
        return img
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

class LogisticRegressionModel(nn.Module):
    """Logistic Regression Model for multi-label classification"""
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_chexzero_model(device):
    """Load CheXzero model"""
    print("Loading CheXzero model...")
    
    # Add the CheXzero code directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    chexzero_code_path = os.path.join(project_root, "external_sota", "chexzero", "code", "CheXzero")
    sys.path.insert(0, chexzero_code_path)
    
    try:
        from model import CLIP
        
        # Load CheXzero checkpoint
        model_path = "/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt"
        
        if os.path.exists(model_path):
            # Create CheXzero model with correct parameters
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
            model.load_state_dict(torch.load(model_path, map_location='cuda'))
            print("CheXzero model loaded successfully!")
        else:
            raise FileNotFoundError(f"CheXzero checkpoint not found: {model_path}")
            
        return model
        
    except Exception as e:
        print(f"Error loading CheXzero model: {e}")
        raise

def load_biomedclip_model(device):
    """Load BiomedCLIP model"""
    print("Loading BiomedCLIP model...")
    
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        print(f"Loading model: {model_name}")
        
        model, preprocess = create_model_from_pretrained(model_name)
        
        # Validate model was loaded properly
        if model is None:
            raise ValueError("Failed to load BiomedCLIP model - model is None")
        
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        
        tokenizer = get_tokenizer(model_name)
        
        # Create wrapper with standard interface
        class BiomedCLIPWrapper:
            def __init__(self, model, preprocess):
                if model is None:
                    raise ValueError("Cannot create wrapper with None model")
                self.model = model
                self.preprocess = preprocess
                
            def encode_image(self, images):
                if self.model is None:
                    raise ValueError("Model is None in encode_image")
                with torch.no_grad():
                    features = self.model.encode_image(images)
                return features
                
            def eval(self):
                if self.model is not None:
                    self.model.eval()
                return self
                
            def to(self, device):
                if self.model is not None:
                    self.model = self.model.to(device)
                return self
        
        model_wrapper = BiomedCLIPWrapper(model, preprocess)
        print("BiomedCLIP model loaded successfully!")
        return model_wrapper, preprocess
        
    except ImportError as e:
        print(f"ImportError loading BiomedCLIP: {e}")
        print("Make sure open_clip is installed: pip install open_clip_torch")
        raise
    except Exception as e:
        print(f"Error loading BiomedCLIP model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_openai_clip_model(device):
    """Load OpenAI CLIP model"""
    print("Loading OpenAI CLIP model...")
    
    try:
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        print("OpenAI CLIP model loaded successfully!")
        return model, preprocess
        
    except Exception as e:
        print(f"Error loading OpenAI CLIP model: {e}")
        raise

def get_cache_key(model_name, dataset_type):
    """Generate a unique cache key for model+dataset combination"""
    # Include model name and dataset type in the key
    cache_str = f"{model_name}_{dataset_type}_v1"  # v1 for version control
    return cache_str

def save_cached_features(features, labels, label_names, model_name, dataset_type):
    """Save extracted features to cache"""
    cache_key = get_cache_key(model_name, dataset_type)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    
    cache_data = {
        'features': features,
        'labels': labels,
        'label_names': label_names,
        'model_name': model_name,
        'dataset_type': dataset_type,
        'feature_shape': features.shape,
        'label_shape': labels.shape
    }
    
    print(f"Saving features to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cached_features(model_name, dataset_type):
    """Load cached features if they exist"""
    cache_key = get_cache_key(model_name, dataset_type)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data['features'], cache_data['labels'], cache_data['label_names']
        except Exception as e:
            print(f"Error loading cache: {e}, will re-extract features")
            return None, None, None
    return None, None, None

def save_cached_aligned_features(train_features, train_labels, val_features, val_labels, 
                                test_features, test_labels, all_labels, model_name):
    """Save aligned features for all datasets for a specific model"""
    cache_key = f"{model_name}_aligned_v1"
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    
    cache_data = {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'all_labels': all_labels,
        'model_name': model_name
    }
    
    print(f"Saving aligned features to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cached_aligned_features(model_name):
    """Load cached aligned features if they exist"""
    cache_key = f"{model_name}_aligned_v1"
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_path.exists():
        print(f"Loading cached aligned features: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            return (cache_data['train_features'], cache_data['train_labels'],
                   cache_data['val_features'], cache_data['val_labels'],
                   cache_data['test_features'], cache_data['test_labels'],
                   cache_data['all_labels'])
        except Exception as e:
            print(f"Error loading aligned cache: {e}, will re-extract features")
            return None, None, None, None, None, None, None
    return None, None, None, None, None, None, None

def get_shared_labels():
    """Get labels that are shared between MIMIC and CheXpert datasets"""
    shared_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
        'Pneumothorax', 'Support Devices'
    ]
    return shared_labels

def load_dataset_labels(dataset_type):
    """Load labels for different datasets"""
    shared_labels = get_shared_labels()
    
    if dataset_type == "mimic_train":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/mimic_train.csv")
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
        
    elif dataset_type == "chexpert_val":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.csv")
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
        
    elif dataset_type == "chexpert_test":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.csv")
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

@torch.no_grad()
def extract_visual_features(model, model_name, dataset_type, preprocess=None, batch_size=32):
    """Extract visual features using model's visual encoder with caching"""
    print(f"\n=== Extracting visual features for {model_name} on {dataset_type} ===")
    
    # Check cache first
    cached_features, cached_labels, cached_label_names = load_cached_features(model_name, dataset_type)
    if cached_features is not None:
        print(f"✓ Using cached features (shape: {cached_features.shape})")
        return cached_features, cached_labels, cached_label_names
    
    print("Cache miss - extracting features...")
    
    # Load dataset labels
    df, available_labels = load_dataset_labels(dataset_type)
    print(f"Available labels: {available_labels}")
    
    # Get dataset path
    if dataset_type == "mimic_train":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/mimic.h5"
    elif dataset_type == "chexpert_val":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.h5"
    elif dataset_type == "chexpert_test":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.h5"
    
    # Check if files exist
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    num_samples = len(df)
    print(f"Dataset: {num_samples} images")
    
    # Setup model-specific preprocessing
    if model_name == "chexzero":
        # CheXzero uses CXR normalization + 224x224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
        ])
    elif model_name in ["biomedclip", "openai_clip"]:
        # Use model's built-in preprocessing
        if preprocess is None:
            raise ValueError(f"Preprocessing function required for {model_name}")
        # For CLIP models, we need to adapt the preprocess for our tensor format
        def tensor_preprocess(img_tensor):
            # Convert tensor back to PIL for CLIP preprocessing, then back to tensor
            from PIL import Image
            import torchvision.transforms.functional as F
            
            # Convert (3, H, W) tensor to PIL Image
            img_pil = F.to_pil_image(img_tensor)
            # Apply CLIP preprocessing
            processed = preprocess(img_pil)
            return processed
        transform = tensor_preprocess
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create dataset and dataloader
    dataset = CXRDataset(h5_path, num_samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Extract image features
    all_img_features = []
    
    for batch_imgs in tqdm(dataloader, desc="Processing images"):
        batch_imgs = batch_imgs.to(device)
        
        # Handle different model interfaces
        if hasattr(model, 'encode_image'):
            img_features_batch = model.encode_image(batch_imgs)
        else:
            # Fallback for models without encode_image method
            img_features_batch = model(batch_imgs)
        
        img_features_batch = img_features_batch / img_features_batch.norm(dim=-1, keepdim=True)
        all_img_features.append(img_features_batch.cpu())
        torch.cuda.empty_cache()
    
    img_features = torch.cat(all_img_features)
    print(f"Extracted image features shape: {img_features.shape}")
    
    # Prepare labels
    labels = np.zeros((num_samples, len(available_labels)))
    for i, label in enumerate(available_labels):
        if label in df.columns:
            # Handle different label formats (0/1, -1/0/1, etc.)
            label_values = df[label].fillna(0)
            # Convert to binary (positive = 1, others = 0)
            labels[:, i] = (label_values == 1).astype(int)
    
    print(f"Final feature shape: {img_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert to numpy and save to cache
    features_numpy = img_features.cpu().numpy()
    save_cached_features(features_numpy, labels, available_labels, model_name, dataset_type)
    
    return features_numpy, labels, available_labels

def train_epoch(model, criterion, optimizer, train_loader):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_epoch(model, data_loader):
    """Evaluate model on validation/test set"""
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    
    # Calculate AUC for each label
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
        else:
            aucs.append(0.0)
    
    mean_auc = np.mean(aucs)
    return mean_auc, y_true, y_pred, aucs

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_baseline_linear_probing(model_name, seed=42):
    """Main function to run baseline linear probing for a specific model"""
    print(f"=== Baseline Linear Probing: {model_name} (Seed: {seed}) ===")
    
    # Set random seed
    set_random_seed(seed)
    
    # Check for cached aligned features first (most efficient)
    cached_data = load_cached_aligned_features(model_name)
    if cached_data[0] is not None:
        print("✓ Using cached aligned features for all datasets")
        (train_features, train_labels_aligned, val_features, val_labels_aligned,
         test_features, test_labels_aligned, all_labels) = cached_data
    else:
        # Need to extract and align features
        print("Cache miss for aligned features - extracting...")
        
        # Load model only if we need to extract features
        print(f"Loading {model_name} model...")
        preprocess = None
        try:
            if model_name == "chexzero":
                model = load_chexzero_model(device).to(device).eval()
                preprocess = None  # CheXzero doesn't need separate preprocessing
            elif model_name == "biomedclip":
                model, preprocess = load_biomedclip_model(device)
                model = model.to(device).eval()
            elif model_name == "openai_clip":
                model, preprocess = load_openai_clip_model(device)
                model = model.to(device).eval()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Validate model was loaded successfully
            if model is None:
                raise ValueError(f"Model {model_name} loaded as None")
                
            print(f"✓ {model_name} model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load {model_name} model: {e}")
            print(f"Skipping {model_name} experiments")
            raise
        
        # Extract features for all datasets
        print("\n=== Feature Extraction ===")
        train_features, train_labels, train_label_names = extract_visual_features(
            model, model_name, "mimic_train", preprocess
        )
        
        val_features, val_labels, val_label_names = extract_visual_features(
            model, model_name, "chexpert_val", preprocess
        )
        
        test_features, test_labels, test_label_names = extract_visual_features(
            model, model_name, "chexpert_test", preprocess
        )
        
        # Free model memory immediately after feature extraction
        del model
        torch.cuda.empty_cache()
        
        # Align labels across datasets
        all_labels = list(set(train_label_names + val_label_names + test_label_names))
        all_labels.sort()
        print(f"Final aligned labels: {all_labels}")
        
        # Reformat labels to match aligned label set
        def align_labels(labels, label_names, all_labels):
            aligned = np.zeros((labels.shape[0], len(all_labels)))
            for i, label in enumerate(all_labels):
                if label in label_names:
                    idx = label_names.index(label)
                    aligned[:, i] = labels[:, idx]
            return aligned
        
        train_labels_aligned = align_labels(train_labels, train_label_names, all_labels)
        val_labels_aligned = align_labels(val_labels, val_label_names, all_labels)
        test_labels_aligned = align_labels(test_labels, test_label_names, all_labels)
        
        # Cache the aligned features for future runs
        save_cached_aligned_features(train_features, train_labels_aligned, 
                                   val_features, val_labels_aligned,
                                   test_features, test_labels_aligned, 
                                   all_labels, model_name)
    
    print(f"Training set: {train_features.shape[0]} samples")
    print(f"Validation set: {val_features.shape[0]} samples")
    print(f"Test set: {test_features.shape[0]} samples")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Number of labels: {len(all_labels)}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_features).float(),
        torch.tensor(train_labels_aligned).float()
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features).float(),
        torch.tensor(val_labels_aligned).float()
    )
    test_dataset = TensorDataset(
        torch.tensor(test_features).float(),
        torch.tensor(test_labels_aligned).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    input_dim = train_features.shape[1]
    output_dim = len(all_labels)
    
    lr_model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lr_model.parameters(), lr=2e-4, weight_decay=1e-8)
    
    # Training loop with validation
    print("\n=== Training ===")
    best_val_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(200):  # max epochs
        # Train
        train_loss = train_epoch(lr_model, criterion, optimizer, train_loader)
        
        # Validate
        val_auc, _, _, val_aucs = evaluate_epoch(lr_model, val_loader)
        
        if epoch % 20 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.4f}, Val AUC = {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = lr_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    lr_model.load_state_dict(best_model_state)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Final evaluation on validation and test sets
    val_auc, y_true_val, y_pred_val, val_aucs = evaluate_epoch(lr_model, val_loader)
    test_auc, y_true_test, y_pred_test, test_aucs = evaluate_epoch(lr_model, test_loader)
    print(f"Test AUC: {test_auc:.4f}")
    
    # Return results for aggregation (including predictions and model)
    results = {
        'model': model_name,
        'seed': seed,
        'test_auc': test_auc,
        'val_auc': best_val_auc,
        'labels': all_labels,
        'per_label_aucs': {label: auc for label, auc in zip(all_labels, test_aucs)},
        'feature_dim': input_dim,
        'train_samples': len(train_features),
        'val_samples': len(val_features), 
        'test_samples': len(test_features),
        # Add predictions and ground truth
        'predictions': {
            'val': {
                'y_true': y_true_val,
                'y_pred': y_pred_val
            },
            'test': {
                'y_true': y_true_test,
                'y_pred': y_pred_test
            }
        },
        # Add trained model state dict
        'trained_model': best_model_state,
        'model_config': {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'architecture': 'LogisticRegression'
        }
    }
    
    return results

def clear_cache():
    """Clear all cached features"""
    if CACHE_DIR.exists():
        print(f"Clearing cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print("Cache directory does not exist")

def get_cache_info():
    """Get information about cached files"""
    if not CACHE_DIR.exists():
        return "Cache directory does not exist"
    
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        return "No cached files found"
    
    info = f"Cache directory: {CACHE_DIR}\n"
    info += f"Cached files ({len(cache_files)}):\n"
    
    total_size = 0
    for cache_file in cache_files:
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        info += f"  - {cache_file.name}: {size_mb:.1f} MB\n"
    
    info += f"Total cache size: {total_size:.1f} MB"
    return info

def run_multiple_seeds_for_model(model_name):
    """Run experiment with multiple seeds for a specific model"""
    print(f"=== Running 20 Experiments for {model_name} with Different Seeds ===")
    print(f"Note: Features will be extracted once and cached for reuse across seeds")
    
    # Create results directory
    results_dir = f'results/baseline_linear_probing_{model_name}'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    predictions_dir = os.path.join(results_dir, 'predictions')
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Run experiments with different seeds
    seeds = list(range(42, 62))  # Seeds 42-61 (20 seeds)
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"EXPERIMENT {i+1}/20 (Model: {model_name}, Seed: {seed})")
        print(f"{'='*50}")
        
        try:
            results = run_baseline_linear_probing(model_name, seed)
            all_results.append(results)
            
            # Save individual run results (metrics only)
            results_clean = results.copy()
            if 'predictions' in results_clean:
                del results_clean['predictions']
            if 'trained_model' in results_clean:
                del results_clean['trained_model']
            
            with open(f'{results_dir}/results_seed_{seed}.json', 'w') as f:
                json.dump(results_clean, f, indent=2)
            
            # Save predictions and ground truth separately
            if 'predictions' in results:
                predictions_data = {
                    'seed': seed,
                    'model_name': model_name,
                    'labels': results['labels'],
                    'val': {
                        'y_true': results['predictions']['val']['y_true'].tolist(),
                        'y_pred': results['predictions']['val']['y_pred'].tolist()
                    },
                    'test': {
                        'y_true': results['predictions']['test']['y_true'].tolist(),
                        'y_pred': results['predictions']['test']['y_pred'].tolist()
                    }
                }
                
                with open(f'{predictions_dir}/seed_{seed}_predictions.pkl', 'wb') as f:
                    pickle.dump(predictions_data, f)
            
            # Save trained model checkpoint
            if 'trained_model' in results:
                model_data = {
                    'seed': seed,
                    'model_name': model_name,
                    'state_dict': results['trained_model'],
                    'model_config': results['model_config'],
                    'labels': results['labels'],
                    'test_auc': results['test_auc'],
                    'val_auc': results['val_auc']
                }
                
                torch.save(model_data, f'{models_dir}/seed_{seed}_model.pth')
                
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            continue
    
    # Aggregate results
    if all_results:
        test_aucs = [r['test_auc'] for r in all_results]
        val_aucs = [r['val_auc'] for r in all_results]
        
        # Per-label aggregation
        all_labels = all_results[0]['labels']
        per_label_stats = {}
        for label in all_labels:
            label_aucs = [r['per_label_aucs'][label] for r in all_results]
            per_label_stats[label] = {
                'mean': np.mean(label_aucs),
                'std': np.std(label_aucs),
                'aucs': label_aucs
            }
        
        # Clean individual results for JSON serialization
        individual_results_clean = []
        for result in all_results:
            result_clean = result.copy()
            if 'predictions' in result_clean:
                del result_clean['predictions']
            if 'trained_model' in result_clean:
                del result_clean['trained_model']
            individual_results_clean.append(result_clean)
        
        aggregated_results = {
            'method': f'baseline_linear_probing_{model_name}_20_seeds',
            'model': model_name,
            'num_runs': len(all_results),
            'seeds': [r['seed'] for r in all_results],
            'test_auc': {
                'mean': np.mean(test_aucs),
                'std': np.std(test_aucs),
                'all_aucs': test_aucs
            },
            'val_auc': {
                'mean': np.mean(val_aucs),
                'std': np.std(val_aucs),
                'all_aucs': val_aucs
            },
            'per_label_stats': per_label_stats,
            'individual_results': individual_results_clean
        }
        
        # Save aggregated results
        with open(f'{results_dir}/aggregated_results.json', 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS SUMMARY - {model_name}")
        print(f"{'='*60}")
        print(f"Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"Val AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        print(f"Successful runs: {len(all_results)}/20")
        print(f"Results saved to: {results_dir}/")
        print(f"Predictions saved to: {predictions_dir}/")
        print(f"Model checkpoints saved to: {models_dir}/")
        
        return aggregated_results
    else:
        print(f"No successful runs for {model_name}!")
        return None

def run_all_baselines():
    """Run experiments for all baseline models"""
    print("=== Running Baseline Linear Probing for All Models ===")
    
    models = ["chexzero", "biomedclip", "openai_clip"]
    all_model_results = {}
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENTS FOR {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            results = run_multiple_seeds_for_model(model_name)
            if results:
                all_model_results[model_name] = results
        except Exception as e:
            print(f"Error running experiments for {model_name}: {e}")
            continue
    
    # Save combined results
    results_dir = 'results/baseline_linear_probing_combined'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/all_models_results.json', 'w') as f:
        json.dump(all_model_results, f, indent=2)
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON ACROSS ALL MODELS")
    print(f"{'='*80}")
    
    for model_name, results in all_model_results.items():
        test_auc = results['test_auc']
        print(f"{model_name:12s}: {test_auc['mean']:.4f} ± {test_auc['std']:.4f}")
    
    return all_model_results

def load_seed_predictions(model_name, seed):
    """Load predictions for a specific model and seed"""
    predictions_dir = f'results/baseline_linear_probing_{model_name}/predictions'
    predictions_file = f'{predictions_dir}/seed_{seed}_predictions.pkl'
    
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    with open(predictions_file, 'rb') as f:
        predictions_data = pickle.load(f)
    
    return predictions_data

def load_seed_model(model_name, seed):
    """Load trained model for a specific model and seed"""
    models_dir = f'results/baseline_linear_probing_{model_name}/models'
    model_file = f'{models_dir}/seed_{seed}_model.pth'
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model_data = torch.load(model_file, map_location='cpu')
    
    # Create model instance
    config = model_data['model_config']
    model = LogisticRegressionModel(config['input_dim'], config['output_dim'])
    model.load_state_dict(model_data['state_dict'])
    
    return model, model_data

def load_all_predictions_for_model(model_name):
    """Load all predictions for a specific model across all seeds"""
    predictions_dir = f'results/baseline_linear_probing_{model_name}/predictions'
    
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    all_predictions = {}
    prediction_files = [f for f in os.listdir(predictions_dir) if f.startswith('seed_') and f.endswith('_predictions.pkl')]
    
    for pred_file in prediction_files:
        seed = int(pred_file.split('_')[1])
        with open(os.path.join(predictions_dir, pred_file), 'rb') as f:
            predictions_data = pickle.load(f)
        all_predictions[seed] = predictions_data
    
    return all_predictions

def get_predictions_summary(model_name):
    """Get summary of available predictions and models for a model"""
    results_dir = f'results/baseline_linear_probing_{model_name}'
    predictions_dir = os.path.join(results_dir, 'predictions')
    models_dir = os.path.join(results_dir, 'models')
    
    summary = {
        'model_name': model_name,
        'results_dir': results_dir,
        'predictions_available': [],
        'models_available': [],
        'total_seeds': 0
    }
    
    # Check predictions
    if os.path.exists(predictions_dir):
        pred_files = [f for f in os.listdir(predictions_dir) if f.startswith('seed_') and f.endswith('_predictions.pkl')]
        summary['predictions_available'] = [int(f.split('_')[1]) for f in pred_files]
    
    # Check models
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('seed_') and f.endswith('_model.pth')]
        summary['models_available'] = [int(f.split('_')[1]) for f in model_files]
    
    # Total seeds that have both predictions and models
    available_seeds = set(summary['predictions_available']) & set(summary['models_available'])
    summary['total_seeds'] = len(available_seeds)
    summary['complete_seeds'] = sorted(list(available_seeds))
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Linear Probing with Caching - Saves predictions, ground truth, and model checkpoints")
    parser.add_argument('--clear-cache', action='store_true', 
                       help='Clear all cached features before running')
    parser.add_argument('--cache-info', action='store_true',
                       help='Show cache information and exit')
    parser.add_argument('--models', nargs='+', 
                       choices=['chexzero', 'biomedclip', 'openai_clip'],
                       default=['chexzero', 'biomedclip', 'openai_clip'],
                       help='Models to run experiments for')
    
    args = parser.parse_args()
    
    # Handle cache info
    if args.cache_info:
        print(get_cache_info())
        exit(0)
    
    # Handle cache clearing
    if args.clear_cache:
        clear_cache()
    
    # Show initial cache info
    print("=== Cache Information ===")
    print(get_cache_info())
    print()
    
    # Run experiments
    if len(args.models) == 3:
        results = run_all_baselines()
    else:
        # Run specific models
        all_model_results = {}
        for model_name in args.models:
            print(f"\n{'='*80}")
            print(f"STARTING EXPERIMENTS FOR {model_name.upper()}")
            print(f"{'='*80}")
            
            try:
                results = run_multiple_seeds_for_model(model_name)
                if results:
                    all_model_results[model_name] = results
            except Exception as e:
                print(f"Error running experiments for {model_name}: {e}")
                continue
        
        # Save combined results for selected models
        results_dir = 'results/baseline_linear_probing_combined'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f'{results_dir}/selected_models_results.json', 'w') as f:
            json.dump(all_model_results, f, indent=2)
        
        # Print final comparison
        print(f"\n{'='*80}")
        print("FINAL COMPARISON ACROSS SELECTED MODELS")
        print(f"{'='*80}")
        
        for model_name, results in all_model_results.items():
            test_auc = results['test_auc']
            print(f"{model_name:12s}: {test_auc['mean']:.4f} ± {test_auc['std']:.4f}")
        
        results = all_model_results
    
    print("\n=== Final Cache Information ===")
    print(get_cache_info())
    print("\n=== All Baseline Experiments Complete ===")
    print("📁 Saved for each experiment:")
    print("   • JSON results with metrics and statistics")
    print("   • Pickle files with predictions (y_pred) and ground truth (y_true)")
    print("   • PyTorch model checkpoints (.pth) with trained weights")
    print("   • Aggregated results across all 20 seeds")
    print("🔧 Use utility functions to load data:")
    print("   • load_seed_predictions(model_name, seed)")
    print("   • load_seed_model(model_name, seed)")
    print("   • load_all_predictions_for_model(model_name)")
    print("   • get_predictions_summary(model_name)") 