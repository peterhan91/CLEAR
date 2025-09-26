#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
import pickle
import sys
import random
import shutil

# Import functions from existing files
# from zero_shot import load_clip
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_clip

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Convert to 3-channel format like in the original code
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

def fix_numpy_compatibility():
    """Fix numpy._core compatibility for older pickle files"""
    import numpy as np
    if not hasattr(np, '_core'):
        import types
        np._core = types.ModuleType('_core')
        if hasattr(np, 'core'):
            for attr in dir(np.core):
                if not attr.startswith('_'):
                    setattr(np._core, attr, getattr(np.core, attr))
        if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
            np._core.multiarray = np.core.multiarray
        elif hasattr(np, 'multiarray'):
            np._core.multiarray = np.multiarray
    sys.modules['numpy._core'] = np._core
    if hasattr(np._core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np._core.multiarray

def load_concepts_and_embeddings():
    """Load diagnostic concepts and their embeddings"""
    print("Loading diagnostic concepts...")
    concepts_df = pd.read_csv("/home/than/DeepLearning/CheXzero/data/mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()
    
    print("Loading concept embeddings...")
    fix_numpy_compatibility()
    
    with open("/home/than/DeepLearning/CheXzero/embeddings_output/cxr_embeddings_sfr_mistral.pickle", 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Process concept embeddings with proper alignment
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
    print(f"Concepts loaded: {len(concepts)}")
    print(f"Concept embeddings shape: {concept_embeddings.shape}")
    
    return concepts, concept_embeddings

def get_shared_labels():
    """Get labels that are shared between MIMIC and CheXpert datasets"""
    # All 14 overlapping labels across MIMIC, CheXpert valid, and CheXpert test
    shared_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        # 'No Finding', 
        'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
        'Pneumothorax', 'Support Devices'
    ]
    return shared_labels

def load_dataset_labels(dataset_type):
    """Load labels for different datasets"""
    shared_labels = get_shared_labels()
    
    if dataset_type == "mimic_train":
        df = pd.read_csv("/home/than/DeepLearning/conceptqa_vip/data/mimic_cxr.csv")
        # Filter for training set and available columns
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
        
    elif dataset_type == "chexpert_val":
        df = pd.read_csv("/home/than/DeepLearning/CheXzero/data/chexpert_valid.csv")
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
        
    elif dataset_type == "chexpert_test":
        df = pd.read_csv("/home/than/DeepLearning/CheXzero/data/chexpert_test.csv")
        available_labels = [label for label in shared_labels if label in df.columns]
        return df, available_labels
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

@torch.no_grad()
def extract_concept_features(model, concepts, concept_embeddings, dataset_type, batch_size=64):
    """Extract concept-weighted features for a dataset"""
    print(f"\n=== Extracting features for {dataset_type} ===")
    
    # Load dataset labels
    df, available_labels = load_dataset_labels(dataset_type)
    print(f"Available labels: {available_labels}")
    
    # Get dataset path
    if dataset_type == "mimic_train":
        h5_path = "/home/than/DeepLearning/conceptqa_vip/data/mimic_train_448.h5"
        is_preencoded = True
    elif dataset_type == "chexpert_val":
        h5_path = "/home/than/DeepLearning/CheXzero/data/chexpert_valid.h5"
        is_preencoded = False
    elif dataset_type == "chexpert_test":
        h5_path = "/home/than/DeepLearning/CheXzero/data/chexpert_test.h5"
        is_preencoded = False
    
    # Check if files exist
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    num_samples = len(df)
    print(f"Dataset: {num_samples} images")
    
    # Encode concepts with CLIP first (we'll need this for both cases)
    print("Encoding concepts...")
    
    # Check for cached concept features
    concept_cache_file = "cache/clip_concept_features.pkl"
    if os.path.exists(concept_cache_file):
        print("Loading cached CLIP concept features...")
        try:
            with open(concept_cache_file, 'rb') as f:
                cached_concept_data = pickle.load(f)
            # Verify cache matches current concepts
            if cached_concept_data['concepts'] == concepts:
                concept_features = cached_concept_data['concept_features'].to(device)
                print(f"✓ Loaded cached concept features: {concept_features.shape}")
            else:
                print("Concept list changed, re-encoding...")
                concept_features = None
        except Exception as e:
            print(f"Cache loading failed: {e}, re-encoding...")
            concept_features = None
    else:
        concept_features = None
    
    # Encode concepts if not cached or cache invalid
    if concept_features is None:
        print("Encoding concepts with CLIP...")
        from clear import clip
        concept_batch_size = 4096
        all_concept_features = []
        
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to(device)
            concept_features_batch = model.encode_text(concept_tokens)
            concept_features_batch /= concept_features_batch.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features_batch.cpu())
            torch.cuda.empty_cache()
        
        concept_features = torch.cat(all_concept_features)
        print(f"Concept features shape: {concept_features.shape}")
        
        # Cache the encoded concept features
        os.makedirs("cache", exist_ok=True)
        cache_data = {
            'concepts': concepts,
            'concept_features': concept_features
        }
        try:
            with open(concept_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cached CLIP concept features to: {concept_cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache concept features: {e}")
        
        concept_features = concept_features.to(device)
    
    if is_preencoded:
        # Load pre-encoded CLIP features (MIMIC train)
        print("Loading pre-encoded CLIP features...")
        with h5py.File(h5_path, 'r') as h5_file:
            clip_features = h5_file['cxr_feature'][:]  # Shape: (N, 768)
        
        print(f"Pre-encoded features shape: {clip_features.shape}")
        
        # Ensure we have the same number of features as labels
        if len(clip_features) != num_samples:
            print(f"Warning: Feature count ({len(clip_features)}) != label count ({num_samples})")
            # Take the minimum to avoid index errors
            min_samples = min(len(clip_features), num_samples)
            clip_features = clip_features[:min_samples]
            df = df.iloc[:min_samples]
            num_samples = min_samples
            print(f"Using {num_samples} samples")
        
        # Convert to torch and normalize
        img_features = torch.tensor(clip_features).float()
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        print(f"Normalized image features shape: {img_features.shape}")
        
    else:
        # Extract features from raw images (CheXpert val/test)
        print("Extracting features from raw images...")
        
        # Setup preprocessing (match the original format)
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(448, interpolation=InterpolationMode.BICUBIC),
        ])
        
        # Create dataset and dataloader
        dataset = CXRDataset(h5_path, num_samples, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Extract image features
        all_img_features = []
        
        for batch_imgs in tqdm(dataloader, desc="Processing images"):
            batch_imgs = batch_imgs.to(device)
            img_features_batch = model.encode_image(batch_imgs)
            img_features_batch /= img_features_batch.norm(dim=-1, keepdim=True)
            all_img_features.append(img_features_batch.cpu())
            torch.cuda.empty_cache()
        
        img_features = torch.cat(all_img_features)
        print(f"Extracted image features shape: {img_features.shape}")
    
    # Compute concept similarities and LLM representation in batches to save memory
    print("Computing concept-weighted features (memory efficient)...")
    img_batch_size = 500  # Smaller batches to save memory
    all_representations = []
    
    concept_embeddings = concept_embeddings.to(device)
    
    for i in tqdm(range(0, len(img_features), img_batch_size), desc="Processing image batches"):
        batch_img_features = img_features[i:i+img_batch_size].to(device)
        
        # Compute similarities: [batch_size, num_concepts]
        batch_similarity = batch_img_features @ concept_features.T
        
        # Immediately compute LLM representation: [batch_size, embedding_dim]
        batch_representation = batch_similarity @ concept_embeddings
        batch_representation /= batch_representation.norm(dim=-1, keepdim=True)
        
        all_representations.append(batch_representation.cpu())
        torch.cuda.empty_cache()
    
    llm_representation = torch.cat(all_representations)
    print(f"LLM representation computed efficiently")
    
    # Prepare labels
    labels = np.zeros((num_samples, len(available_labels)))
    for i, label in enumerate(available_labels):
        if label in df.columns:
            # Handle different label formats (0/1, -1/0/1, etc.)
            label_values = df[label].fillna(0)
            # Convert to binary (positive = 1, others = 0)
            labels[:, i] = (label_values == 1).astype(int)
    
    print(f"Final feature shape: {llm_representation.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return llm_representation.cpu().numpy(), labels, available_labels

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
        if len(np.unique(y_true[:, i])) > 1:  # Check if both classes are present
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
        else:
            aucs.append(0.0)  # or np.nan
    
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

def run_concept_based_linear_probing(seed=42):
    """Main function to run concept-based linear probing"""
    print(f"=== Concept-Based Linear Probing (Seed: {seed}) ===")
    
    # Set random seed
    set_random_seed(seed)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model = load_clip(
        model_path="/home/than/DeepLearning/cxr_concept/CheXzero/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    ).to(device).eval()
    
    # Load concepts and embeddings
    concepts, concept_embeddings = load_concepts_and_embeddings()
    
    # Extract features for all datasets
    print("\n=== Feature Extraction ===")
    train_features, train_labels, train_label_names = extract_concept_features(
        model, concepts, concept_embeddings, "mimic_train"
    )
    
    val_features, val_labels, val_label_names = extract_concept_features(
        model, concepts, concept_embeddings, "chexpert_val"
    )
    
    test_features, test_labels, test_label_names = extract_concept_features(
        model, concepts, concept_embeddings, "chexpert_test"
    )
    
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
        
        if epoch % 20 == 0 or epoch < 10:  # Reduce printing
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
        'seed': seed,
        'test_auc': test_auc,
        'val_auc': best_val_auc,
        'labels': all_labels,
        'per_label_aucs': {label: auc for label, auc in zip(all_labels, test_aucs)},
        'num_concepts': len(concepts),
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
            'architecture': 'LogisticRegression',
            'num_concepts': len(concepts)
        }
    }
    
    return results

def run_multiple_seeds():
    """Run experiment with 20 different random seeds"""
    print("=== Running 20 Experiments with Different Seeds ===")
    
    # Clean and create results directory
    results_dir = 'results/concept_based_linear_probing_torch'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    predictions_dir = os.path.join(results_dir, 'predictions')
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Run experiments with different seeds
    seeds = list(range(42, 62))  # Seeds 42-61
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"EXPERIMENT {i+1}/20 (Seed: {seed})")
        print(f"{'='*50}")
        
        try:
            results = run_concept_based_linear_probing(seed)
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
                    'method': 'concept_based_linear_probing',
                    'labels': results['labels'],
                    'num_concepts': results['num_concepts'],
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
                    'method': 'concept_based_linear_probing',
                    'state_dict': results['trained_model'],
                    'model_config': results['model_config'],
                    'labels': results['labels'],
                    'num_concepts': results['num_concepts'],
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
            'method': 'concept_based_linear_probing_20_seeds',
            'num_runs': len(all_results),
            'seeds': [r['seed'] for r in all_results],
            'num_concepts': all_results[0]['num_concepts'],
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
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"Val AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        print(f"Successful runs: {len(all_results)}/20")
        print(f"Results saved to: {results_dir}/")
        print(f"Predictions saved to: {predictions_dir}/")
        print(f"Model checkpoints saved to: {models_dir}/")
        
        return aggregated_results
    else:
        print("No successful runs!")
        return None

def load_concept_seed_predictions(seed):
    """Load predictions for a specific seed"""
    predictions_dir = 'results/concept_based_linear_probing_torch/predictions'
    predictions_file = f'{predictions_dir}/seed_{seed}_predictions.pkl'
    
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    with open(predictions_file, 'rb') as f:
        predictions_data = pickle.load(f)
    
    return predictions_data

def load_concept_seed_model(seed):
    """Load trained model for a specific seed"""
    models_dir = 'results/concept_based_linear_probing_torch/models'
    model_file = f'{models_dir}/seed_{seed}_model.pth'
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model_data = torch.load(model_file, map_location='cpu')
    
    # Create model instance
    config = model_data['model_config']
    model = LogisticRegressionModel(config['input_dim'], config['output_dim'])
    model.load_state_dict(model_data['state_dict'])
    
    return model, model_data

def load_all_concept_predictions():
    """Load all predictions across all seeds"""
    predictions_dir = 'results/concept_based_linear_probing_torch/predictions'
    
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

def get_concept_results_summary():
    """Get summary of available predictions and models"""
    results_dir = 'results/concept_based_linear_probing_torch'
    predictions_dir = os.path.join(results_dir, 'predictions')
    models_dir = os.path.join(results_dir, 'models')
    
    summary = {
        'method': 'concept_based_linear_probing',
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
    results = run_multiple_seeds()
    print("\n=== All Concept-Based Experiments Complete ===")
    print("📁 Saved for each experiment:")
    print("   • JSON results with metrics and statistics")
    print("   • Pickle files with predictions (y_pred) and ground truth (y_true)")
    print("   • PyTorch model checkpoints (.pth) with trained weights")
    print("   • Aggregated results across all 20 seeds")
    print("🔧 Use utility functions to load data:")
    print("   • load_concept_seed_predictions(seed)")
    print("   • load_concept_seed_model(seed)")
    print("   • load_all_concept_predictions()")
    print("   • get_concept_results_summary()") 
