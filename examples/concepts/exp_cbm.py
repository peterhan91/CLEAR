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
import hashlib
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_curve

# Import functions from existing files
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
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5_file:
            img_data = np.array(h5_file['cxr'][idx])  # shape: (320, 320)
        
        # Convert to 3-channel format like in the original code
        img_data = np.expand_dims(img_data, axis=0)  # (1, 320, 320)
        img_data = np.repeat(img_data, 3, axis=0)    # (3, 320, 320)
        img = torch.from_numpy(img_data).float()
        
        if self.transform:
            img = self.transform(img)
            
        return img

class LogisticRegressionModel(nn.Module):
    """Logistic Regression Model for multi-label classification"""
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_filtered_concepts():
    """Load filtered diagnostic concepts from CBM concepts file"""
    print("Loading filtered CBM concepts...")
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
            label_concepts.append(len(concepts) - 1)  # Store position in concepts list
        label_to_concepts[label] = label_concepts
    
    print(f"Loaded {len(concepts)} concepts across {len(cbm_data)} labels")
    return concepts, concept_indices, label_to_concepts

def load_concept_embeddings(concept_indices):
    """Load concept embeddings filtered by concept indices"""
    print("Loading concept embeddings...")
    
    # Try to fix numpy compatibility for older pickle files
    try:
        import numpy.core as np_core
        if not hasattr(np, '_core'):
            np._core = np_core
    except (ImportError, AttributeError):
        pass
    
    with open("/home/than/DeepLearning/CheXzero/embeddings_output/cxr_embeddings_sfr_mistral.pickle", 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Process concept embeddings with proper alignment
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

def get_shared_labels():
    """Get the 5 target labels for evaluation across all datasets"""
    return ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']

def normalize_label_name(label):
    """Normalize label names to handle variations like 'pulmonary edema' -> 'edema'"""
    return label.lower().replace('pulmonary ', '').strip()

def match_and_rename_labels(df, target_labels):
    """Match and rename dataset columns to target labels using normalization"""
    # Create normalized mapping from dataset columns
    normalized_col_map = {}
    for col in df.columns:
        norm = normalize_label_name(col)
        normalized_col_map[norm] = col
    
    # Find matches and create rename mapping
    matched_cols = []
    rename_map = {}
    available_labels = []
    
    for label in target_labels:
        norm_label = normalize_label_name(label)
        if norm_label in normalized_col_map:
            orig_col = normalized_col_map[norm_label]
            matched_cols.append(orig_col)
            rename_map[orig_col] = label
            available_labels.append(label)
    
    # Filter and rename columns
    if matched_cols:
        df_filtered = df[matched_cols].copy()
        df_filtered = df_filtered.rename(columns=rename_map)
        return df_filtered, available_labels
    else:
        return df[[]], []

def load_dataset_labels(dataset_type):
    """Load labels for different datasets with proper label matching"""
    target_labels = get_shared_labels()
    
    if dataset_type == "mimic_train":
        df = pd.read_csv("/home/than/DeepLearning/conceptqa_vip/data/mimic_cxr.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
        
    elif dataset_type == "chexpert_val":
        df = pd.read_csv("/home/than/DeepLearning/CheXzero/data/chexpert_valid.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
        
    elif dataset_type == "chexpert_test":
        df = pd.read_csv("/home/than/DeepLearning/CheXzero/data/chexpert_test.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
        
    elif dataset_type == "padchest_test":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
        
    elif dataset_type == "vindrcxr_test":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
        
    elif dataset_type == "indiana_test":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv")
        df_filtered, available_labels = match_and_rename_labels(df, target_labels)
        return df_filtered, available_labels
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def get_cache_key(dataset_type, model_type="dinov2"):
    """Generate cache key for features"""
    return f"{dataset_type}_{model_type}"

def cache_features(features, cache_key, cache_type="features"):
    """Save features to cache"""
    cache_dir = Path("cache/exp_cbm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{cache_type}_{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Cached {cache_type} to {cache_file}")

def load_cached_features(cache_key, cache_type="features"):
    """Load features from cache"""
    cache_dir = Path("cache/exp_cbm")
    cache_file = cache_dir / f"{cache_type}_{cache_key}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded cached {cache_type} from {cache_file}")
        return features
    return None

@torch.no_grad()
def extract_image_features(model, dataset_type, batch_size=64):
    """Extract CLIP image features for a dataset"""
    print(f"\n=== Extracting image features for {dataset_type} ===")
    
    # Check cache first
    cache_key = get_cache_key(dataset_type)
    cached_features = load_cached_features(cache_key, "image_features")
    if cached_features is not None:
        return cached_features
    
    # Load dataset labels
    df, available_labels = load_dataset_labels(dataset_type)
    
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
    elif dataset_type == "padchest_test":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5"
        is_preencoded = False
    elif dataset_type == "vindrcxr_test":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.h5"
        is_preencoded = False
    elif dataset_type == "indiana_test":
        h5_path = "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.h5"
        is_preencoded = False
    
    # Check if files exist
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    num_samples = len(df)
    print(f"Dataset: {num_samples} images")
    
    if is_preencoded:
        # Load pre-encoded CLIP features (MIMIC train)
        print("Loading pre-encoded CLIP features...")
        with h5py.File(h5_path, 'r') as h5_file:
            clip_features = np.array(h5_file['cxr_feature'])  # Shape: (N, 768)
        
        print(f"Pre-encoded features shape: {clip_features.shape}")
        
        # Ensure we have the same number of features as labels
        if len(clip_features) != num_samples:
            print(f"Warning: Feature count ({len(clip_features)}) != label count ({num_samples})")
            min_samples = min(len(clip_features), num_samples)
            clip_features = clip_features[:min_samples]
            df = df.iloc[:min_samples]
            num_samples = min_samples
            print(f"Using {num_samples} samples")
        
        # Convert to torch and normalize
        img_features = torch.tensor(clip_features).float()
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        
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
    
    print(f"Final image features shape: {img_features.shape}")
    
    # Prepare labels
    labels = np.zeros((num_samples, len(available_labels)))
    for i, label in enumerate(available_labels):
        if label in df.columns:
            label_values = df[label].fillna(0)
            labels[:, i] = (label_values == 1).astype(int)
    
    result = {
        'features': img_features.cpu().numpy(),
        'labels': labels,
        'label_names': available_labels,
        'num_samples': num_samples
    }
    
    # Cache the result
    cache_features(result, cache_key, "image_features")
    
    return result

@torch.no_grad()
def extract_concept_features(model, concepts):
    """Extract CLIP concept features"""
    print("Extracting concept features...")
    
    # Check cache first
    cache_key = "concept_features"
    cached_features = load_cached_features(cache_key, "concept_features")
    if cached_features is not None:
        return cached_features
    
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
    
    concept_features = torch.cat(all_concept_features)
    print(f"Concept features shape: {concept_features.shape}")
    
    result = concept_features.cpu().numpy()
    
    # Cache the result
    cache_features(result, cache_key, "concept_features")
    
    return result

def train_and_evaluate_model(train_features, train_labels, val_features, val_labels, 
                           test_features, test_labels, label_names, method_name, seed=42):
    """Train and evaluate a logistic regression model"""
    print(f"\n=== Training {method_name} (seed={seed}) ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_features).float(),
        torch.tensor(train_labels).float()
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features).float(),
        torch.tensor(val_labels).float()
    )
    test_dataset = TensorDataset(
        torch.tensor(test_features).float(),
        torch.tensor(test_labels).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    input_dim = train_features.shape[1]
    output_dim = len(label_names)
    
    lr_model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lr_model.parameters(), lr=2e-4, weight_decay=1e-8)
    
    # Training loop with validation
    print(f"Training with input dim: {input_dim}, output dim: {output_dim}")
    best_val_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    def train_epoch(model, criterion, optimizer, train_loader):
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
    
    for epoch in range(200):  # max epochs
        # Train
        train_loss = train_epoch(lr_model, criterion, optimizer, train_loader)
        
        # Validate
        val_auc, _, _, val_aucs = evaluate_epoch(lr_model, val_loader)
        
        if epoch % 20 == 0:
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
    if best_model_state is not None:
        lr_model.load_state_dict(best_model_state)
    
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Final evaluation
    test_auc, y_true, y_pred, test_aucs = evaluate_epoch(lr_model, test_loader)
    
    print(f"Test AUC: {test_auc:.4f}")
    print("\nPer-label AUCs:")
    for i, (label, auc) in enumerate(zip(label_names, test_aucs)):
        print(f"  {label}: {auc:.4f}")
    
    return {
        'method': method_name,
        'test_auc': test_auc,
        'val_auc': best_val_auc,
        'labels': label_names,
        'per_label_aucs': {label: auc for label, auc in zip(label_names, test_aucs)},
        'predictions': {'y_true': y_true, 'y_pred': y_pred},
        'model': lr_model  # Return the trained model
    }

@torch.no_grad()
def evaluate_trained_model(trained_model, test_features, test_labels, label_names, method_name):
    """Evaluate a trained model on test data without retraining"""
    print(f"Evaluating {method_name} on test data...")
    
    # Create test dataset and dataloader
    test_dataset = TensorDataset(
        torch.tensor(test_features).float(),
        torch.tensor(test_labels).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Evaluate the trained model
    trained_model.eval()
    all_targets = []
    all_predictions = []
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = trained_model(inputs)
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
    
    print(f"Test AUC: {mean_auc:.4f}")
    print("Per-label AUCs:")
    for i, (label, auc) in enumerate(zip(label_names, aucs)):
        print(f"  {label}: {auc:.4f}")
    
    return {
        'method': method_name,
        'test_auc': mean_auc,
        'labels': label_names,
        'per_label_aucs': {label: auc for label, auc in zip(label_names, aucs)},
        'predictions': {'y_true': y_true, 'y_pred': y_pred}
    }

class LLMEmbeddingGenerator:
    """LLM Embedding Generator class similar to exp_zeroshot.py"""
    def __init__(self):
        # Import the embedding generator from get_embed
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from get_embed import RadiologyEmbeddingGenerator
        self.RadiologyEmbeddingGenerator = RadiologyEmbeddingGenerator
        
        # Cache for local embedding generators to avoid reloading
        self._local_generators = {}

    def get_embeddings_batch(self, texts, model_name="sfr_mistral"):
        """Get embeddings batch similar to exp_zeroshot.py"""
        if model_name == "sfr_mistral":
            # Check if we already have this generator cached
            if model_name not in self._local_generators:
                print(f"🔄 Loading {model_name} model for the first time...")
                # Before loading, free all unused memory
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Create and cache the generator
                generator = self.RadiologyEmbeddingGenerator(
                    embedding_type="local",
                    local_model_name='Salesforce/SFR-Embedding-Mistral',
                    batch_size=16
                )
                
                self._local_generators[model_name] = generator
                print(f"✅ {model_name} model loaded and cached")
            else:
                print(f"♻️ Reusing cached {model_name} model")
            
            # Use the cached generator
            generator = self._local_generators[model_name]
            embeddings = generator.get_local_embeddings_batch(texts)
            return np.array(embeddings)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def cleanup(self):
        """Clean up cached models to free GPU memory"""
        # Make a copy of keys to avoid "dictionary changed size during iteration" error
        model_names = list(self._local_generators.keys())
        for model_name in model_names:
            print(f"🧹 Cleaning up {model_name} model...")
            if model_name in self._local_generators:
                del self._local_generators[model_name]
        self._local_generators.clear()
        
        # More aggressive cleanup for large models
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            import gc
            gc.collect()
        print("✅ All models cleaned up")

def generate_class_embeddings(labels, llm_model="sfr_mistral"):
    """Generate real LLM embeddings for class names using the same approach as exp_zeroshot.py"""
    print(f"Generating class embeddings using {llm_model}...")
    
    # Create positive and negative prompts exactly like exp_zeroshot.py
    pos_prompts = [f"{label.lower()}" for label in labels]
    neg_prompts = [f"no {label.lower()}" for label in labels]
    
    print(f"✅ Sample positive prompts: {pos_prompts[:3]}")
    print(f"✅ Sample negative prompts: {neg_prompts[:3]}")
    
    # Initialize the embedding generator like in exp_zeroshot.py
    embedding_generator = LLMEmbeddingGenerator()
    
    print(f"Getting {llm_model.upper()} embeddings for prompts...")
    pos_embeddings = embedding_generator.get_embeddings_batch(pos_prompts, llm_model)
    neg_embeddings = embedding_generator.get_embeddings_batch(neg_prompts, llm_model)
    
    print(f"✅ Positive {llm_model.upper()} embeddings shape: {pos_embeddings.shape}")
    print(f"✅ Negative {llm_model.upper()} embeddings shape: {neg_embeddings.shape}")
    
    # Clean up the generator to free memory
    embedding_generator.cleanup()
    
    return pos_embeddings, neg_embeddings

def compute_zeroshot_predictions(llm_features, pos_embeddings, neg_embeddings):
    """Compute zero-shot predictions exactly like exp_zeroshot.py"""
    print("\n=== Step 8: Compute Predictions ===")
    
    # Convert to tensors and move to GPU
    llm_representation = torch.tensor(llm_features).float().to(device)
    pos_class_embeddings = torch.tensor(pos_embeddings).float().to(device)
    neg_class_embeddings = torch.tensor(neg_embeddings).float().to(device)
    
    # Normalize embeddings like in exp_zeroshot.py
    llm_representation = llm_representation / llm_representation.norm(dim=-1, keepdim=True)
    pos_class_embeddings = pos_class_embeddings / pos_class_embeddings.norm(dim=-1, keepdim=True)
    neg_class_embeddings = neg_class_embeddings / neg_class_embeddings.norm(dim=-1, keepdim=True)
    
    with torch.no_grad():
        # Calculate similarities between LLM representation and class embeddings (exact copy from exp_zeroshot.py)
        logits_pos = llm_representation @ pos_class_embeddings.T
        logits_neg = llm_representation @ neg_class_embeddings.T
        
        # Apply softmax to get probabilities (exact copy from exp_zeroshot.py)
        exp_logits_pos = torch.exp(logits_pos)
        exp_logits_neg = torch.exp(logits_neg)
        probabilities = exp_logits_pos / (exp_logits_pos + exp_logits_neg)
        
        y_pred = probabilities.cpu().numpy()
    
    print(f"✅ Predictions shape: {y_pred.shape}")
    
    return y_pred

def evaluate_zeroshot_performance(y_pred, y_true, label_names):
    """Evaluate zero-shot performance using the same eval function as exp_zeroshot.py"""
    print("\n=== Step 9: Evaluate Performance ===")
    
    # Import the same evaluate function used in exp_zeroshot.py
    from clear.evaluation import evaluate
    
    # Use the same evaluation as exp_zeroshot.py
    results_df = evaluate(y_pred, y_true, label_names)
    print("📊 Performance Results:")
    print(results_df)
    
    # Calculate mean AUC like in exp_zeroshot.py
    auc_columns = [col for col in results_df.columns if col.endswith('_auc')]
    if auc_columns:
        auc_values = [results_df[col].iloc[0] for col in auc_columns]
        mean_auc = np.mean(auc_values)
    else:
        mean_auc = 0.0
    
    print(f"\n🎯 FINAL ZERO-SHOT RESULTS:")
    print(f"📈 Average AUC: {mean_auc:.4f}")
    print(f"📊 Number of test images: {len(y_pred)}")
    print(f"🏷️ Number of classes: {len(label_names)}")
    
    # Create per-label AUCs dictionary
    per_label_aucs = {}
    for label in label_names:
        auc_col = f"{label}_auc"
        if auc_col in results_df.columns:
            per_label_aucs[label] = float(results_df[auc_col].iloc[0])
        else:
            per_label_aucs[label] = 0.0
    
    print(f"\n📋 Individual AUCs:")
    for label, auc in per_label_aucs.items():
        print(f"  {label}: {auc:.4f}")
    
    return {
        'method': 'Zero_Shot_CBM',
        'test_auc': mean_auc,
        'val_auc': mean_auc,  # No validation for zero-shot
        'labels': label_names,
        'per_label_aucs': per_label_aucs,
        'results_df': results_df,
        'predictions': {'y_true': y_true, 'y_pred': y_pred}
    }

def check_seed_models_exist(seed):
    """Check if models for a specific seed already exist"""
    base_dir = Path("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck")
    seed_dir = base_dir / f"seed_{seed}"
    
    standard_weights = seed_dir / "standard_cbm_weights.pth"
    standard_info = seed_dir / "standard_cbm_info.json"
    improved_weights = seed_dir / "improved_cbm_weights.pth"
    improved_info = seed_dir / "improved_cbm_info.json"
    
    standard_exists = standard_weights.exists() and standard_info.exists()
    improved_exists = improved_weights.exists() and improved_info.exists()
    
    return {
        'standard_exists': standard_exists,
        'improved_exists': improved_exists,
        'both_exist': standard_exists and improved_exists,
        'seed_dir': seed_dir
    }

def load_seed_models(seed, all_labels):
    """Load existing models for a specific seed"""
    print(f"📂 Loading existing models for seed {seed}...")
    
    base_dir = Path("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck")
    seed_dir = base_dir / f"seed_{seed}"
    
    results = {'seed': seed}
    
    # Load Standard CBM
    standard_weights = seed_dir / "standard_cbm_weights.pth"
    standard_info = seed_dir / "standard_cbm_info.json"
    
    if standard_weights.exists() and standard_info.exists():
        # Load model info
        with open(standard_info, 'r') as f:
            info = json.load(f)
        
        # Create and load model
        standard_model = LogisticRegressionModel(info['input_dim'], info['output_dim']).to(device)
        standard_model.load_state_dict(torch.load(standard_weights, map_location=device))
        standard_model.eval()
        
        standard_result = {
            'method': 'Standard_CBM',
            'test_auc': info.get('test_auc_chexpert', 0.0),
            'val_auc': info.get('validation_auc', 0.0),
            'labels': all_labels,
            'per_label_aucs': {},  # Will be filled during evaluation
            'model': standard_model
        }
        print(f"  ✅ Standard CBM loaded (val_auc: {standard_result['val_auc']:.4f})")
    else:
        standard_result = None
        print(f"  ❌ Standard CBM not found")
    
    # Load Improved CBM
    improved_weights = seed_dir / "improved_cbm_weights.pth"
    improved_info = seed_dir / "improved_cbm_info.json"
    
    if improved_weights.exists() and improved_info.exists():
        # Load model info
        with open(improved_info, 'r') as f:
            info = json.load(f)
        
        # Create and load model
        improved_model = LogisticRegressionModel(info['input_dim'], info['output_dim']).to(device)
        improved_model.load_state_dict(torch.load(improved_weights, map_location=device))
        improved_model.eval()
        
        improved_result = {
            'method': 'Improved_CBM',
            'test_auc': info.get('test_auc_chexpert', 0.0),
            'val_auc': info.get('validation_auc', 0.0),
            'labels': all_labels,
            'per_label_aucs': {},  # Will be filled during evaluation
            'model': improved_model
        }
        print(f"  ✅ Improved CBM loaded (val_auc: {improved_result['val_auc']:.4f})")
    else:
        improved_result = None
        print(f"  ❌ Improved CBM not found")
    
    return standard_result, improved_result

def save_seed_models(seed, standard_model_result, improved_model_result, all_labels):
    """Save models for a specific seed"""
    base_dir = Path("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_bottleneck")
    seed_dir = base_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save standard CBM model
    if 'model' in standard_model_result:
        model = standard_model_result['model']
        torch.save(model.state_dict(), seed_dir / "standard_cbm_weights.pth")
        
        model_info = {
            'seed': seed,
            'method': 'standard_cbm',
            'input_dim': model.linear.in_features,
            'output_dim': model.linear.out_features,
            'labels': all_labels,
            'validation_auc': standard_model_result['val_auc'],
            'test_auc_chexpert': standard_model_result['test_auc']  # This is from CheXpert test
        }
        with open(seed_dir / "standard_cbm_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Save improved CBM model
    if 'model' in improved_model_result:
        model = improved_model_result['model']
        torch.save(model.state_dict(), seed_dir / "improved_cbm_weights.pth")
        
        model_info = {
            'seed': seed,
            'method': 'improved_cbm',
            'input_dim': model.linear.in_features,
            'output_dim': model.linear.out_features,
            'labels': all_labels,
            'validation_auc': improved_model_result['val_auc'],
            'test_auc_chexpert': improved_model_result['test_auc']  # This is from CheXpert test
        }
        with open(seed_dir / "improved_cbm_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    print(f"Models for seed {seed} saved to: {seed_dir}")

def get_best_thresholds(y_pred, y_true, label_names, metric_func=matthews_corrcoef):
    """
    Find optimal thresholds for each label using the specified metric.
    Based on get_best_p_vals from metrics.py
    
    Args:
        y_pred: Predicted probabilities (N, num_labels)
        y_true: True binary labels (N, num_labels) 
        label_names: List of label names
        metric_func: Metric function to optimize (default: Matthews Correlation Coefficient)
    
    Returns:
        Dictionary mapping label names to optimal thresholds
    """
    best_thresholds = {}
    
    for idx, label_name in enumerate(label_names):
        y_true_label = y_true[:, idx]
        y_pred_label = y_pred[:, idx]
        
        # Skip if only one class present
        if len(np.unique(y_true_label)) < 2:
            best_thresholds[label_name] = 0.5  # Default threshold
            continue
        
        # Get thresholds from ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_label, y_pred_label)
        thresholds = thresholds[1:]  # Remove first threshold (inf)
        thresholds = np.sort(thresholds)
        
        # Evaluate metric for each threshold
        metrics_list = []
        for threshold in thresholds:
            y_pred_binary = np.where(y_pred_label < threshold, 0, 1)
            try:
                metric_value = metric_func(y_true_label, y_pred_binary)
                # Handle NaN values
                if np.isnan(metric_value):
                    metric_value = -1.0  # Use a low value for NaN
                metrics_list.append(metric_value)
            except:
                # Handle any other errors in metric computation
                metrics_list.append(-1.0)
        
        # Find best threshold
        best_index = np.argmax(metrics_list)
        best_threshold = thresholds[best_index]
        best_thresholds[label_name] = best_threshold
        
        print(f"  {label_name}: threshold = {best_threshold:.4f}, {metric_func.__name__} = {metrics_list[best_index]:.4f}")
    
    return best_thresholds

def compute_f1_scores(y_pred, y_true, label_names, thresholds):
    """
    Compute F1 scores using the provided thresholds.
    Based on compute_f1 from metrics.py
    
    Args:
        y_pred: Predicted probabilities (N, num_labels)
        y_true: True binary labels (N, num_labels)
        label_names: List of label names
        thresholds: Dictionary mapping label names to thresholds
    
    Returns:
        Dictionary mapping label names to F1 scores
    """
    f1_scores = {}
    
    for idx, label_name in enumerate(label_names):
        y_true_label = y_true[:, idx]
        y_pred_label = y_pred[:, idx]
        threshold = thresholds[label_name]
        
        # Apply threshold to get binary predictions
        y_pred_binary = np.where(y_pred_label < threshold, 0, 1)
        
        # Handle case where only one class is present
        if len(np.unique(y_true_label)) < 2:
            f1_scores[label_name] = 0.0
            continue
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_label, y_pred_binary)
        if cm.size == 1:  # Only one class predicted
            f1_scores[label_name] = 0.0
            continue
            
        tn, fp, fn, tp = cm.ravel()
        
        # Compute F1 score: F1 = 2*TP / (2*TP + FP + FN)
        if (2*tp + fp + fn) == 0:
            f1_scores[label_name] = 1.0
        else:
            f1_scores[label_name] = (2 * tp) / (2*tp + fp + fn)
    
    return f1_scores

def compute_mcc_scores(y_pred, y_true, label_names, thresholds):
    """
    Compute Matthews Correlation Coefficient scores using the provided thresholds.
    Based on compute_mcc from metrics.py
    
    Args:
        y_pred: Predicted probabilities (N, num_labels)
        y_true: True binary labels (N, num_labels)
        label_names: List of label names
        thresholds: Dictionary mapping label names to thresholds
    
    Returns:
        Dictionary mapping label names to MCC scores
    """
    mcc_scores = {}
    
    for idx, label_name in enumerate(label_names):
        y_true_label = y_true[:, idx]
        y_pred_label = y_pred[:, idx]
        threshold = thresholds[label_name]
        
        # Apply threshold to get binary predictions
        y_pred_binary = np.where(y_pred_label < threshold, 0, 1)
        
        # Handle case where only one class is present
        if len(np.unique(y_true_label)) < 2:
            mcc_scores[label_name] = 0.0
            continue
        
        # Compute MCC
        try:
            mcc_score = matthews_corrcoef(y_true_label, y_pred_binary)
            # Handle NaN values
            if np.isnan(mcc_score):
                mcc_score = 0.0
            mcc_scores[label_name] = mcc_score
        except:
            # Handle any errors in MCC computation
            mcc_scores[label_name] = 0.0
    
    return mcc_scores

def evaluate_model_with_metrics(model, features, labels, label_names, method_name, thresholds=None):
    """
    Evaluate a model and compute AUC, F1, and MCC scores.
    
    Args:
        model: Trained model
        features: Input features
        labels: True labels
        label_names: List of label names
        method_name: Name of the method (for logging)
        thresholds: Optional pre-computed thresholds. If None, will be computed.
    
    Returns:
        Dictionary with evaluation results including thresholds, F1, and MCC scores
    """
    # Get model predictions
    result = evaluate_trained_model(model, features, labels, label_names, method_name)
    y_pred = result['predictions']['y_pred']
    y_true = result['predictions']['y_true']
    
    # Compute thresholds if not provided
    if thresholds is None:
        print(f"  Computing optimal thresholds for {method_name}...")
        thresholds = get_best_thresholds(y_pred, y_true, label_names)
    
    # Compute F1 and MCC scores
    print(f"  Computing F1 and MCC scores for {method_name}...")
    f1_scores = compute_f1_scores(y_pred, y_true, label_names, thresholds)
    mcc_scores = compute_mcc_scores(y_pred, y_true, label_names, thresholds)
    
    # Add metrics to result
    result['thresholds'] = thresholds
    result['f1_scores'] = f1_scores
    result['mcc_scores'] = mcc_scores
    
    return result

def save_metrics_to_csv(results_dict, output_dir, filename_prefix):
    """
    Save threshold, F1, and MCC results to CSV files.
    
    Args:
        results_dict: Dictionary containing results for all methods/seeds/datasets
        output_dir: Directory to save CSV files
        filename_prefix: Prefix for CSV filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV files
    threshold_data = []
    f1_data = []
    mcc_data = []
    
    # Process each result in the dictionary
    for key, result in results_dict.items():
        if 'thresholds' not in result:
            continue
            
        # Parse key format: Method_Dataset_Seed (e.g., "Standard_chexpert_42")
        parts = key.split('_')
        if len(parts) >= 3:
            method, dataset, seed = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            method, dataset, seed = parts[0], parts[1], 'N/A'
        else:
            method, dataset, seed = key, 'unknown', 'N/A'
        
        # Extract thresholds
        for label, threshold in result['thresholds'].items():
            threshold_data.append({
                'Method': method,
                'Dataset': dataset,
                'Seed': seed,
                'Label': label,
                'Threshold': threshold
            })
        
        # Extract F1 scores
        for label, f1_score in result['f1_scores'].items():
            f1_data.append({
                'Method': method,
                'Dataset': dataset,
                'Seed': seed,
                'Label': label,
                'F1_Score': f1_score
            })
        
        # Extract MCC scores
        for label, mcc_score in result['mcc_scores'].items():
            mcc_data.append({
                'Method': method,
                'Dataset': dataset,
                'Seed': seed,
                'Label': label,
                'MCC_Score': mcc_score
            })
    
    # Save to CSV files
    if threshold_data:
        threshold_df = pd.DataFrame(threshold_data)
        threshold_file = output_dir / f"{filename_prefix}_thresholds.csv"
        threshold_df.to_csv(threshold_file, index=False)
        print(f"Thresholds saved to: {threshold_file}")
    
    if f1_data:
        f1_df = pd.DataFrame(f1_data)
        f1_file = output_dir / f"{filename_prefix}_f1_scores.csv"
        f1_df.to_csv(f1_file, index=False)
        print(f"F1 scores saved to: {f1_file}")
    
    if mcc_data:
        mcc_df = pd.DataFrame(mcc_data)
        mcc_file = output_dir / f"{filename_prefix}_mcc_scores.csv"
        mcc_df.to_csv(mcc_file, index=False)
        print(f"MCC scores saved to: {mcc_file}")
    
    return threshold_df, f1_df, mcc_df

def run_zeroshot_cbm_evaluation(test_llm_features, test_labels, all_labels, concept_embeddings, llm_model="sfr_mistral"):
    """Run zero-shot CBM evaluation using LLM representations - following exp_zeroshot.py pattern"""
    print("🔬 Running zero-shot CBM evaluation...")
    print(f"✅ Test LLM features shape: {test_llm_features.shape}")
    print(f"✅ Number of labels: {len(all_labels)}")
    print(f"✅ Number of test images: {len(test_llm_features)}")
    
    print(f"\n=== Step 7: Generate {llm_model.upper()} Embeddings for Class Prompts ===")
    # Generate class embeddings using real LLM (same as exp_zeroshot.py)
    pos_embeddings, neg_embeddings = generate_class_embeddings(all_labels, llm_model)
    
    # Compute zero-shot predictions (same as exp_zeroshot.py)
    y_pred = compute_zeroshot_predictions(test_llm_features, pos_embeddings, neg_embeddings)
    
    # Evaluate performance using same eval function as exp_zeroshot.py
    results = evaluate_zeroshot_performance(y_pred, test_labels, all_labels)
    
    return results

def run_cbm_experiments():
    """Main function to run all three CBM experiments"""
    print("=== CBM Experiments ===")
    
    # Load CLIP model
    print("Loading CLIP model...")
    model = load_clip(
        model_path="/home/than/DeepLearning/cxr_concept/CheXzero/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    ).to(device).eval()
    
    # Load filtered concepts and embeddings
    concepts, concept_indices, label_to_concepts = load_filtered_concepts()
    concept_embeddings = load_concept_embeddings(concept_indices)
    
    # Extract image features for all datasets
    print("\n=== Extracting Image Features ===")
    train_data = extract_image_features(model, "mimic_train")
    val_data = extract_image_features(model, "chexpert_val")
    
    # Extract features for all test datasets
    chexpert_test_data = extract_image_features(model, "chexpert_test")
    padchest_test_data = extract_image_features(model, "padchest_test")
    vindrcxr_test_data = extract_image_features(model, "vindrcxr_test")
    indiana_test_data = extract_image_features(model, "indiana_test")
    
    # Extract concept features
    concept_features = extract_concept_features(model, concepts)
    
    # Collect all test datasets
    test_datasets = {
        'chexpert': chexpert_test_data,
        'padchest': padchest_test_data, 
        'vindrcxr': vindrcxr_test_data,
        'indiana': indiana_test_data
    }
    
    # Align labels across datasets - use the 5 target labels consistently
    all_labels = get_shared_labels()  # Use the 5 target labels
    print(f"Target labels: {all_labels}")
    
    # Reformat labels to match aligned label set
    def align_labels(labels, label_names, all_labels):
        aligned = np.zeros((labels.shape[0], len(all_labels)))
        for i, label in enumerate(all_labels):
            if label in label_names:
                idx = label_names.index(label)
                aligned[:, i] = labels[:, idx]
        return aligned
    
    train_labels = align_labels(train_data['labels'], train_data['label_names'], all_labels)
    val_labels = align_labels(val_data['labels'], val_data['label_names'], all_labels)
    
    # Align labels for all test datasets
    test_labels = {}
    for dataset_name, dataset_data in test_datasets.items():
        test_labels[dataset_name] = align_labels(dataset_data['labels'], dataset_data['label_names'], all_labels)
    
    print(f"MIMIC training set: {train_data['features'].shape[0]} samples")
    print(f"Validation set: {val_data['features'].shape[0]} samples")
    for dataset_name, dataset_data in test_datasets.items():
        print(f"{dataset_name.capitalize()} test set: {dataset_data['features'].shape[0]} samples")
    print(f"Number of labels: {len(all_labels)}")
    print(f"Number of concepts: {len(concepts)}")
    
    # Compute concept scores for all datasets (needed for all methods)
    train_concept_scores = train_data['features'] @ concept_features.T
    val_concept_scores = val_data['features'] @ concept_features.T
    
    # Compute concept scores for all test datasets
    test_concept_scores = {}
    for dataset_name, dataset_data in test_datasets.items():
        test_concept_scores[dataset_name] = dataset_data['features'] @ concept_features.T
    
    print(f"Concept scores shape: {train_concept_scores.shape}")
    
    # Compute concept-weighted LLM representations (needed for zero-shot and improved CBM)
    concept_embeddings_np = concept_embeddings.cpu().numpy()
    
    train_llm_features = train_concept_scores @ concept_embeddings_np
    val_llm_features = val_concept_scores @ concept_embeddings_np
    
    # Compute LLM features for all test datasets
    test_llm_features = {}
    for dataset_name, concept_scores in test_concept_scores.items():
        test_llm_features[dataset_name] = concept_scores @ concept_embeddings_np
    
    # Normalize all features
    train_llm_features = train_llm_features / np.linalg.norm(train_llm_features, axis=1, keepdims=True)
    val_llm_features = val_llm_features / np.linalg.norm(val_llm_features, axis=1, keepdims=True)
    
    for dataset_name in test_llm_features:
        test_llm_features[dataset_name] = test_llm_features[dataset_name] / np.linalg.norm(test_llm_features[dataset_name], axis=1, keepdims=True)
    
    print(f"LLM representation shape: {train_llm_features.shape}")
    
    # ============= Multiple Seed Experiments =============
    print("\n" + "="*50)
    print("RUNNING 20 SEED EXPERIMENTS")
    print("="*50)
    
    # Prepare results directory (but don't clean up existing models)
    results_dir = Path("results/concept_bottleneck")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Results directory ready: {results_dir}")
    
    seeds = list(range(42, 62))  # Seeds 42 to 61 (20 seeds total)
    all_seed_results = {}
    
    # Check which models already exist
    print("\n🔍 Checking for existing trained models...")
    existing_models_summary = {}
    for seed in seeds:
        model_status = check_seed_models_exist(seed)
        existing_models_summary[seed] = model_status
        if model_status['both_exist']:
            print(f"  Seed {seed}: ✅ Both Standard and Improved CBM models exist")
        elif model_status['standard_exists'] and not model_status['improved_exists']:
            print(f"  Seed {seed}: ⚠️  Only Standard CBM exists, need to train Improved CBM")
        elif not model_status['standard_exists'] and model_status['improved_exists']:
            print(f"  Seed {seed}: ⚠️  Only Improved CBM exists, need to train Standard CBM")
        else:
            print(f"  Seed {seed}: ❌ No models exist, need to train both")
    
    # Count how many models we need to train
    seeds_needing_training = [seed for seed in seeds if not existing_models_summary[seed]['both_exist']]
    seeds_with_models = [seed for seed in seeds if existing_models_summary[seed]['both_exist']]
    
    print(f"\n📊 Training Summary:")
    print(f"  Seeds with both models: {len(seeds_with_models)}/20")
    print(f"  Seeds needing training: {len(seeds_needing_training)}/20")
    
    if seeds_with_models:
        print(f"  Will load existing: {seeds_with_models}")
    if seeds_needing_training:
        print(f"  Will train: {seeds_needing_training}")
    
    # Run zero-shot once (no training needed)
    print("\n=== Zero-Shot CBM (seed-independent) ===")
    zeroshot_results = {}
    for dataset_name in test_datasets.keys():
        print(f"Zero-Shot CBM on {dataset_name.capitalize()}")
        zeroshot_results[dataset_name] = run_zeroshot_cbm_evaluation(
            test_llm_features[dataset_name], test_labels[dataset_name], all_labels, 
            concept_embeddings, llm_model="sfr_mistral"
        )
    
    # Run training experiments for each seed
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seeds.index(seed)+1}/20)")
        print(f"{'='*60}")
        
        seed_results = {'seed': seed}
        model_status = existing_models_summary[seed]
        
        # Handle Standard CBM
        if model_status['standard_exists']:
            print(f"\n--- Loading existing Standard CBM (seed={seed}) ---")
            standard_model_result, _ = load_seed_models(seed, all_labels)
        else:
            print(f"\n--- Training Standard CBM (seed={seed}) ---")
            standard_model_result = train_and_evaluate_model(
                train_concept_scores, train_labels,
                val_concept_scores, val_labels,
                test_concept_scores['chexpert'], test_labels['chexpert'],  # Use chexpert for initial eval
                all_labels, f"Standard_CBM", seed=seed
            )
        
        # Compute optimal thresholds for Standard CBM using validation set
        print(f"--- Computing thresholds for Standard CBM on validation set (seed={seed}) ---")
        val_result_std = evaluate_model_with_metrics(
            standard_model_result['model'], 
            val_concept_scores, val_labels,
            all_labels, f"Standard_CBM_validation"
        )
        standard_thresholds = val_result_std['thresholds']
        
        # Test Standard CBM on all datasets using validation thresholds
        print(f"--- Testing Standard CBM on all datasets (seed={seed}) ---")
        standard_results = {}
        for dataset_name in test_datasets.keys():
            print(f"  Testing on {dataset_name}...")
            test_result = evaluate_model_with_metrics(
                standard_model_result['model'], 
                test_concept_scores[dataset_name], test_labels[dataset_name],
                all_labels, f"Standard_CBM_{dataset_name}",
                thresholds=standard_thresholds  # Use validation thresholds
            )
            standard_results[dataset_name] = test_result
        seed_results['standard'] = standard_results
        
        # Handle Improved CBM
        if model_status['improved_exists']:
            print(f"\n--- Loading existing Improved CBM (seed={seed}) ---")
            _, improved_model_result = load_seed_models(seed, all_labels)
        else:
            print(f"\n--- Training Improved CBM (seed={seed}) ---")
            improved_model_result = train_and_evaluate_model(
                train_llm_features, train_labels,
                val_llm_features, val_labels,
                test_llm_features['chexpert'], test_labels['chexpert'],  # Use chexpert for initial eval
                all_labels, f"Improved_CBM", seed=seed
            )
        
        # Compute optimal thresholds for Improved CBM using validation set
        print(f"--- Computing thresholds for Improved CBM on validation set (seed={seed}) ---")
        val_result_imp = evaluate_model_with_metrics(
            improved_model_result['model'], 
            val_llm_features, val_labels,
            all_labels, f"Improved_CBM_validation"
        )
        improved_thresholds = val_result_imp['thresholds']
        
        # Test Improved CBM on all datasets using validation thresholds
        print(f"--- Testing Improved CBM on all datasets (seed={seed}) ---")
        improved_results = {}
        for dataset_name in test_datasets.keys():
            print(f"  Testing on {dataset_name}...")
            test_result = evaluate_model_with_metrics(
                improved_model_result['model'], 
                test_llm_features[dataset_name], test_labels[dataset_name],
                all_labels, f"Improved_CBM_{dataset_name}",
                thresholds=improved_thresholds  # Use validation thresholds
            )
            improved_results[dataset_name] = test_result
        seed_results['improved'] = improved_results
        
        # Save models for this seed (only if they were newly trained)
        if not model_status['both_exist']:
            save_seed_models(seed, standard_model_result, improved_model_result, all_labels)
            print(f"💾 Models saved for seed {seed}")
        else:
            print(f"📂 Using existing models for seed {seed}")
        
        all_seed_results[seed] = seed_results
    
    # ============= Save Metrics Results to CSV =============
    print("\n" + "="*50)
    print("SAVING THRESHOLD, F1, AND MCC RESULTS")
    print("="*50)
    
    # Collect all metrics results
    all_metrics_results = {}
    
    # Add zero-shot results (compute thresholds and metrics for each dataset)
    for dataset_name in test_datasets.keys():
        if 'predictions' in zeroshot_results[dataset_name]:
            y_pred = zeroshot_results[dataset_name]['predictions']['y_pred']
            y_true = zeroshot_results[dataset_name]['predictions']['y_true']
            
            print(f"Computing thresholds and metrics for Zero-Shot CBM on {dataset_name}...")
            thresholds = get_best_thresholds(y_pred, y_true, all_labels)
            f1_scores = compute_f1_scores(y_pred, y_true, all_labels, thresholds)
            mcc_scores = compute_mcc_scores(y_pred, y_true, all_labels, thresholds)
            
            all_metrics_results[f"ZeroShot_{dataset_name}_NA"] = {
                'thresholds': thresholds,
                'f1_scores': f1_scores,
                'mcc_scores': mcc_scores
            }
    
    # Add trained model results for all seeds and datasets
    for seed in seeds:
        if seed not in all_seed_results:
            continue
            
        seed_data = all_seed_results[seed]
        
        # Standard CBM results
        for dataset_name in test_datasets.keys():
            if dataset_name in seed_data['standard']:
                result = seed_data['standard'][dataset_name]
                if 'thresholds' in result:
                    all_metrics_results[f"Standard_{dataset_name}_{seed}"] = {
                        'thresholds': result['thresholds'],
                        'f1_scores': result['f1_scores'],
                        'mcc_scores': result['mcc_scores']
                    }
        
        # Improved CBM results
        for dataset_name in test_datasets.keys():
            if dataset_name in seed_data['improved']:
                result = seed_data['improved'][dataset_name]
                if 'thresholds' in result:
                    all_metrics_results[f"Improved_{dataset_name}_{seed}"] = {
                        'thresholds': result['thresholds'],
                        'f1_scores': result['f1_scores'],
                        'mcc_scores': result['mcc_scores']
                    }
    
    # Save metrics to CSV files
    metrics_output_dir = results_dir / "metrics"
    threshold_df, f1_df, mcc_df = save_metrics_to_csv(
        all_metrics_results, metrics_output_dir, "cbm_experiments"
    )
    
    print(f"Metrics results saved to: {metrics_output_dir}")
    print(f"Total results saved: {len(all_metrics_results)} method-dataset-seed combinations")
    
    # ============= Summary Results =============
    print("\n" + "="*50)
    print("FINAL STATISTICS ACROSS 20 SEEDS")
    print("="*50)
    
    # Compute statistics across seeds
    def compute_stats(values):
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Print results for each test dataset
    for dataset_name in test_datasets.keys():
        print(f"\n{dataset_name.upper()} Test Set:")
        print(f"  Zero-Shot CBM - Test AUC: {zeroshot_results[dataset_name]['test_auc']:.4f} (no training!)")
        
        # Collect AUC results across seeds
        standard_aucs = [all_seed_results[seed]['standard'][dataset_name]['test_auc'] for seed in seeds]
        improved_aucs = [all_seed_results[seed]['improved'][dataset_name]['test_auc'] for seed in seeds]
        
        std_auc_stats = compute_stats(standard_aucs)
        imp_auc_stats = compute_stats(improved_aucs)
        
        print(f"  Standard CBM  - Test AUC: {std_auc_stats['mean']:.4f} ± {std_auc_stats['std']:.4f} [{std_auc_stats['min']:.4f}, {std_auc_stats['max']:.4f}]")
        print(f"  Improved CBM  - Test AUC: {imp_auc_stats['mean']:.4f} ± {imp_auc_stats['std']:.4f} [{imp_auc_stats['min']:.4f}, {imp_auc_stats['max']:.4f}]")
        
        # Collect F1 and MCC results across seeds (compute mean across labels for each seed)
        standard_f1_means = []
        improved_f1_means = []
        standard_mcc_means = []
        improved_mcc_means = []
        
        for seed in seeds:
            # Standard CBM F1 and MCC
            std_result = all_seed_results[seed]['standard'][dataset_name]
            if 'f1_scores' in std_result:
                std_f1_mean = np.mean(list(std_result['f1_scores'].values()))
                std_mcc_mean = np.mean(list(std_result['mcc_scores'].values()))
                standard_f1_means.append(std_f1_mean)
                standard_mcc_means.append(std_mcc_mean)
            
            # Improved CBM F1 and MCC
            imp_result = all_seed_results[seed]['improved'][dataset_name]
            if 'f1_scores' in imp_result:
                imp_f1_mean = np.mean(list(imp_result['f1_scores'].values()))
                imp_mcc_mean = np.mean(list(imp_result['mcc_scores'].values()))
                improved_f1_means.append(imp_f1_mean)
                improved_mcc_means.append(imp_mcc_mean)
        
        # Compute and print F1 statistics
        if standard_f1_means and improved_f1_means:
            std_f1_stats = compute_stats(standard_f1_means)
            imp_f1_stats = compute_stats(improved_f1_means)
            
            print(f"  Standard CBM  - Test F1:  {std_f1_stats['mean']:.4f} ± {std_f1_stats['std']:.4f} [{std_f1_stats['min']:.4f}, {std_f1_stats['max']:.4f}]")
            print(f"  Improved CBM  - Test F1:  {imp_f1_stats['mean']:.4f} ± {imp_f1_stats['std']:.4f} [{imp_f1_stats['min']:.4f}, {imp_f1_stats['max']:.4f}]")
        
        # Compute and print MCC statistics
        if standard_mcc_means and improved_mcc_means:
            std_mcc_stats = compute_stats(standard_mcc_means)
            imp_mcc_stats = compute_stats(improved_mcc_means)
            
            print(f"  Standard CBM  - Test MCC: {std_mcc_stats['mean']:.4f} ± {std_mcc_stats['std']:.4f} [{std_mcc_stats['min']:.4f}, {std_mcc_stats['max']:.4f}]")
            print(f"  Improved CBM  - Test MCC: {imp_mcc_stats['mean']:.4f} ± {imp_mcc_stats['std']:.4f} [{imp_mcc_stats['min']:.4f}, {imp_mcc_stats['max']:.4f}]")
    
    # ============= Save Summary Results =============
    print("\n" + "="*50)
    print("SAVING SUMMARY RESULTS")
    print("="*50)
    
    # Save summary statistics
    summary_results = {
        'zeroshot': zeroshot_results,
        'seeds': seeds,
        'statistics': {},
        'all_results': all_seed_results,
        'concepts_used': concepts,
        'num_concepts': len(concepts),
        'labels': all_labels,
        'test_datasets': list(test_datasets.keys())
    }
    
    # Compute and save statistics for each dataset
    for dataset_name in test_datasets.keys():
        standard_aucs = [all_seed_results[seed]['standard'][dataset_name]['test_auc'] for seed in seeds]
        improved_aucs = [all_seed_results[seed]['improved'][dataset_name]['test_auc'] for seed in seeds]
        
        summary_results['statistics'][dataset_name] = {
            'standard_cbm': compute_stats(standard_aucs),
            'improved_cbm': compute_stats(improved_aucs),
            'zeroshot_cbm': zeroshot_results[dataset_name]['test_auc']
        }
    
    # Save predictions and ground truth separately before cleaning
    print("💾 Saving predictions and ground truth...")
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    
    # Save zeroshot predictions
    zeroshot_predictions = {}
    for dataset in zeroshot_results:
        if 'predictions' in zeroshot_results[dataset]:
            zeroshot_predictions[dataset] = {
                'y_true': zeroshot_results[dataset]['predictions']['y_true'],
                'y_pred': zeroshot_results[dataset]['predictions']['y_pred'],
                'method': 'Zero_Shot_CBM',
                'labels': all_labels
            }
    
    with open(predictions_dir / "zeroshot_predictions.pkl", 'wb') as f:
        pickle.dump(zeroshot_predictions, f)
    
    # Save trained model predictions for each seed
    for seed in all_seed_results:
        seed_predictions = {'seed': seed}
        
        # Standard CBM predictions
        standard_preds = {}
        for dataset in all_seed_results[seed]['standard']:
            if 'predictions' in all_seed_results[seed]['standard'][dataset]:
                standard_preds[dataset] = {
                    'y_true': all_seed_results[seed]['standard'][dataset]['predictions']['y_true'],
                    'y_pred': all_seed_results[seed]['standard'][dataset]['predictions']['y_pred'],
                    'method': 'Standard_CBM',
                    'labels': all_labels
                }
        seed_predictions['standard'] = standard_preds
        
        # Improved CBM predictions
        improved_preds = {}
        for dataset in all_seed_results[seed]['improved']:
            if 'predictions' in all_seed_results[seed]['improved'][dataset]:
                improved_preds[dataset] = {
                    'y_true': all_seed_results[seed]['improved'][dataset]['predictions']['y_true'],
                    'y_pred': all_seed_results[seed]['improved'][dataset]['predictions']['y_pred'],
                    'method': 'Improved_CBM',
                    'labels': all_labels
                }
        seed_predictions['improved'] = improved_preds
        
        # Save seed predictions
        with open(predictions_dir / f"seed_{seed}_predictions.pkl", 'wb') as f:
            pickle.dump(seed_predictions, f)
    
    print(f"✅ Predictions saved to: {predictions_dir}")
    
    # Save to JSON (without model objects and predictions)
    with open(results_dir / "summary_results.json", 'w') as f:
        # Clean data for JSON serialization
        clean_results = summary_results.copy()
        
        # Function to convert numpy/torch types to Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif hasattr(obj, 'item'):  # torch tensors
                return obj.item()
            else:
                return obj
        
        # Clean all_results section
        for seed in clean_results['all_results']:
            for method in ['standard', 'improved']:
                for dataset in clean_results['all_results'][seed][method]:
                    result = clean_results['all_results'][seed][method][dataset]
                    if 'model' in result:
                        del result['model']
                    if 'predictions' in result:
                        del result['predictions']
                    if 'results_df' in result:
                        del result['results_df']
                    # Convert numeric values
                    clean_results['all_results'][seed][method][dataset] = convert_to_json_serializable(result)
        
        # Clean zeroshot results section
        for dataset in clean_results['zeroshot']:
            result = clean_results['zeroshot'][dataset]
            if 'predictions' in result:
                del result['predictions']
            if 'results_df' in result:
                del result['results_df']
            clean_results['zeroshot'][dataset] = convert_to_json_serializable(result)
        
        # Clean statistics section
        clean_results['statistics'] = convert_to_json_serializable(clean_results['statistics'])
        
        json.dump(clean_results, f, indent=2)
    
    print(f"Summary results saved to: {results_dir / 'summary_results.json'}")
    print(f"Individual models saved in: {results_dir / 'seed_*'} directories")
    print(f"Predictions and ground truth saved in: {predictions_dir}")
    print(f"  - zeroshot_predictions.pkl: Zero-shot CBM predictions for all test datasets")
    print(f"  - seed_*_predictions.pkl: Standard and Improved CBM predictions for each seed")
    print(f"Metrics results (thresholds, F1, MCC) saved in: {metrics_output_dir}")
    print(f"  - cbm_experiments_thresholds.csv: Optimal thresholds for all methods/datasets/seeds")
    print(f"  - cbm_experiments_f1_scores.csv: F1 scores for all methods/datasets/seeds")
    print(f"  - cbm_experiments_mcc_scores.csv: MCC scores for all methods/datasets/seeds")
    # ============= Training/Loading Summary =============
    print("\n" + "="*50)
    print("TRAINING/LOADING SUMMARY")
    print("="*50)
    
    total_models_loaded = len([seed for seed in seeds if existing_models_summary[seed]['both_exist']])
    total_models_trained = len([seed for seed in seeds if not existing_models_summary[seed]['both_exist']])
    
    print(f"📊 Model Status Summary:")
    print(f"  ✅ Seeds with existing models loaded: {total_models_loaded}/20")
    print(f"  🔄 Seeds with newly trained models: {total_models_trained}/20")
    print(f"  💾 All models saved to: {results_dir}")
    
    print(f"\n🎉 20-seed CBM experiments complete!")
    
    return summary_results

if __name__ == "__main__":
    results = run_cbm_experiments()
    print("=== CBM Experiments Complete ===") 
