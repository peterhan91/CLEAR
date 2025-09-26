import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import h5py
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
except ImportError:
    print("Warning: torchvision not available. Some transforms may not work.")
    # Provide fallback minimal implementation
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img
    
    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
        def __call__(self, tensor):
            return (tensor - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
    
    class Resize:
        def __init__(self, size, interpolation=None):
            self.size = size
        def __call__(self, img):
            return torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.size, self.size), mode='bilinear'
            ).squeeze(0)
    
    class InterpolationMode:
        BICUBIC = 'bicubic'

class CXRTestDataset(Dataset):
    """Dataset class for loading CXR test data from H5 files."""
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
        
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)  # Convert to 3 channels
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
        
        return {'img': img}

def make_true_labels(label_path, labels, cutlabels=True):
    """Create ground truth labels array from CSV file."""
    df = pd.read_csv(label_path)
    # check if this is Indiana dataset (has Study column)
    if 'uid' in df.columns:
        exclude_cols = ['uid', 'filename', 'projection', 'MeSH', 
                        'Problems', 'image', 'indication', 'comparison', 
                        'findings', 'impression']
        available_labels = [col for col in df.columns if col not in exclude_cols]
        # Map labels to available ones
        final_labels = []
        for label in labels:
            if label in available_labels:
                final_labels.append(label)
            elif label.lower() in available_labels:
                final_labels.append(label.lower())
        labels = final_labels
        df.columns = [col.lower() if col not in exclude_cols else col for col in df.columns]

    # Check if this is PadChest (has ImageID column)
    if 'ImageID' in df.columns:
        # For PadChest, filter by available labels and convert to lowercase
        exclude_cols = ['ImageID', 'name', 'Path', 'is_test'] if 'is_test' in df.columns else ['ImageID', 'name', 'Path']
        available_labels = [col.lower() for col in df.columns if col not in exclude_cols]
        # Map labels to available ones
        final_labels = []
        for label in labels:
            if label.lower() in available_labels:
                final_labels.append(label.lower())
        labels = final_labels
        df.columns = [col.lower() if col not in ['ImageID', 'name', 'Path', 'is_test'] else col for col in df.columns]
    
    # For other datasets, check if labels exist in columns
    available_labels = [label for label in labels if label in df.columns or label.lower() in df.columns]
    if len(available_labels) != len(labels):
        print(f"Warning: Only {len(available_labels)}/{len(labels)} labels found in dataset")
    
    # Extract ground truth
    y_true = []
    for label in available_labels:
        if label in df.columns:
            y_true.append(df[label].values)
        elif label.lower() in df.columns:
            y_true.append(df[label.lower()].values)
    
    if not y_true:
        raise ValueError(f"No matching labels found in dataset. Available columns: {list(df.columns)}")
    
    y_true = np.column_stack(y_true)
    
    # Handle NaN values - convert to 0
    y_true = np.nan_to_num(y_true, nan=0.0)
    
    return y_true, available_labels

def evaluate_predictions(y_pred, y_true, labels):
    """Evaluate predictions and return AUC scores."""
    results = {}
    
    # Handle case where predictions and labels don't match exactly
    min_labels = min(y_pred.shape[1], len(labels))
    
    for i in range(min_labels):
        label = labels[i]
        
        # Skip if all ground truth labels are the same (no positive or negative cases)
        if len(np.unique(y_true[:, i])) < 2:
            print(f"Warning: Skipping {label} - all labels are the same value")
            continue
            
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results[f"{label}_auc"] = auc
        except Exception as e:
            print(f"Error computing AUC for {label}: {e}")
            results[f"{label}_auc"] = 0.0
    
    # Calculate mean AUC
    auc_values = [v for k, v in results.items() if k.endswith('_auc') and v > 0]
    if auc_values:
        results['mean_auc'] = np.mean(auc_values)
    else:
        results['mean_auc'] = 0.0
    
    return pd.DataFrame([results])

def setup_test_data(dataset_name, batch_size=64, input_resolution=448):
    """Setup test dataset and labels for a given dataset."""
    
    # Get the absolute path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Dataset configurations (paths only)
    dataset_configs = {
        'chexpert_test': {
            'img_path': os.path.join(data_dir, 'chexpert_test.h5'),
            'label_path': os.path.join(data_dir, 'chexpert_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'padchest_test': {
            'img_path': os.path.join(data_dir, 'padchest_test.h5'),
            'label_path': os.path.join(data_dir, 'padchest_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'vindrcxr_test': {
            'img_path': os.path.join(data_dir, 'vindrcxr_test.h5'),
            'label_path': os.path.join(data_dir, 'vindrcxr_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'vindrpcxr_test': {
            'img_path': os.path.join(data_dir, 'vindrpcxr_test.h5'),
            'label_path': os.path.join(data_dir, 'vindrpcxr_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'indiana_test': {
            'img_path': os.path.join(data_dir, 'indiana_test.h5'),
            'label_path': os.path.join(data_dir, 'indiana_test.csv'),
            'templates': [("{}", "no {}")]
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = dataset_configs[dataset_name]
    
    # Check if files exist
    if not os.path.exists(config['img_path']):
        raise FileNotFoundError(f"Image file not found: {config['img_path']}")
    if not os.path.exists(config['label_path']):
        raise FileNotFoundError(f"Label file not found: {config['label_path']}")
    
    # Dynamically read labels from CSV header
    df = pd.read_csv(config['label_path'], nrows=0)  # Read only header
    
    # Get all columns except non-label columns
    exclude_columns = {'Study', 'Path', 'image_id', 'ImageID', 'name', 'is_test', 'uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression'}
    all_columns = set(df.columns)
    label_columns = all_columns - exclude_columns
    labels = sorted(list(label_columns))  # Sort for consistency
    
    # Setup transforms
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])
    
    # Create dataset and dataloader
    dataset = CXRTestDataset(config['img_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Load ground truth
    y_true, actual_labels = make_true_labels(config['label_path'], labels)
    
    print(f"Loaded {dataset_name}:")
    print(f"  - Images: {len(dataset)}")
    print(f"  - Labels: {len(actual_labels)} ({actual_labels})")
    print(f"  - Ground truth shape: {y_true.shape}")
    
    return dataloader, y_true, actual_labels, config['templates']

def save_results(results_df, model_name, dataset_name, output_dir="benchmark/results"):
    """Save evaluation results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{model_name}_{dataset_name}_results.csv"
    filepath = os.path.join(output_dir, filename)
    
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    # Print key metrics
    if 'mean_auc' in results_df.columns:
        mean_auc = results_df['mean_auc'].iloc[0]
        print(f"Mean AUC: {mean_auc:.4f}")
    
    # Print individual AUCs for key pathologies
    key_pathologies = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural effusion_auc']
    for pathology in key_pathologies:
        if pathology in results_df.columns:
            auc = results_df[pathology].iloc[0]
            print(f"{pathology}: {auc:.4f}")

def save_detailed_results(y_pred, y_true, labels, model_name, dataset_name, config_data=None, output_dir=None):
    """
    Save detailed results including predictions and ground truth in the same format as concept-based evaluation.
    This enables bootstrapping analysis and detailed comparison.
    """
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default output directory to benchmark/results if not specified
    if output_dir is None:
        # Get the directory where this script is located (benchmark/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "results")
    
    # Create output directory following the concept-based evaluation structure
    results_dir = os.path.join(output_dir, f"benchmark_evaluation_{dataset_name}_{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving detailed results to: {results_dir}")
    
    # Create ground truth DataFrame with _true suffix
    gt_columns = [f"{label}_true" for label in labels]
    gt_df = pd.DataFrame(y_true, columns=gt_columns)
    
    # Create predictions DataFrame with _pred suffix  
    pred_columns = [f"{label}_pred" for label in labels]
    pred_df = pd.DataFrame(y_pred, columns=pred_columns)
    
    # Save ground truth and predictions
    gt_path = os.path.join(results_dir, f"ground_truth_{timestamp}.csv")
    pred_path = os.path.join(results_dir, f"predictions_{timestamp}.csv")
    
    gt_df.to_csv(gt_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    
    print(f"Saved ground truth: {gt_path}")
    print(f"Saved predictions: {pred_path}")
    
    # Compute detailed AUC metrics
    from sklearn.metrics import roc_auc_score
    detailed_aucs = {}
    valid_aucs = []
    
    for i, label in enumerate(labels):
        try:
            # Skip if all ground truth labels are the same
            if len(np.unique(y_true[:, i])) < 2:
                print(f"Warning: Skipping {label} - all labels are the same value")
                detailed_aucs[f"{label}_auc"] = np.nan
                continue
                
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            detailed_aucs[f"{label}_auc"] = auc
            valid_aucs.append(auc)
            
        except Exception as e:
            print(f"Error computing AUC for {label}: {e}")
            detailed_aucs[f"{label}_auc"] = np.nan
    
    # Calculate mean AUC
    if valid_aucs:
        detailed_aucs['mean_auc'] = np.mean(valid_aucs)
    else:
        detailed_aucs['mean_auc'] = np.nan
    
    # Save detailed AUCs
    aucs_df = pd.DataFrame([detailed_aucs])
    aucs_path = os.path.join(results_dir, f"detailed_aucs_{timestamp}.csv")
    aucs_df.to_csv(aucs_path, index=False)
    
    # Save configuration
    if config_data is None:
        config_data = {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "zero_shot_benchmark",
            "timestamp": timestamp,
            "num_images": len(y_pred),
            "num_labels": len(labels),
            "labels": labels
        }
    
    config_data["timestamp"] = timestamp
    config_path = os.path.join(results_dir, f"config_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Save summary
    summary_data = {
        "timestamp": timestamp,
        "model_name": model_name,
        "dataset": dataset_name,
        "mean_auc": float(detailed_aucs['mean_auc']) if not np.isnan(detailed_aucs['mean_auc']) else None,
        "total_labels": len(labels),
        "valid_labels": len(valid_aucs),
        "detailed_aucs": {k: float(v) if not np.isnan(v) else None for k, v in detailed_aucs.items()}
    }
    
    summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"Saved detailed AUCs: {aucs_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved summary: {summary_path}")
    print(f"📊 Mean AUC: {detailed_aucs['mean_auc']:.4f}" if not np.isnan(detailed_aucs['mean_auc']) else "📊 Mean AUC: N/A")
    
    return results_dir, timestamp

def run_zero_shot_evaluation(model, dataloader, y_true, labels, templates, device, context_length=77, save_detailed=False, model_name=None, dataset_name=None):
    """
    Run zero-shot evaluation on a dataset using a CLIP-like model.
    
    Args:
        model: The vision-language model with encode_image and encode_text methods
        dataloader: DataLoader for the test images
        y_true: Ground truth labels
        labels: List of label names
        templates: List of (positive_template, negative_template) tuples
        device: torch device
        context_length: Maximum sequence length for text encoding
        save_detailed: Whether to save detailed predictions and ground truth
        model_name: Name of the model (required if save_detailed=True)
        dataset_name: Name of the dataset (required if save_detailed=True)
        
    Returns:
        results_df: DataFrame with evaluation results
        y_pred (optional): Prediction probabilities if save_detailed=True
    """
    from clear import clip  # Import here to avoid dependency issues
    
    model.eval()
    pos_template, neg_template = templates[0]
    
    print(f"Running zero-shot evaluation on {len(dataloader.dataset)} images...")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c.lower()) for c in labels]
        neg_texts = [neg_template.format(c.lower()) for c in labels]
        
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting image features"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions
    img_feats_cat = torch.cat(all_img_feats).to(device)
    
    with torch.no_grad():
        logits_pos = img_feats_cat @ pos_features.T
        logits_neg = img_feats_cat @ neg_features.T
        probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    
    y_pred = probs.cpu().numpy()
    
    # Evaluate
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    # Save detailed results if requested
    if save_detailed and model_name and dataset_name:
        config_data = {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "zero_shot_benchmark",
            "test_batch_size": dataloader.batch_size,
            "templates": [pos_template, neg_template],
            "context_length": context_length,
            "num_images": len(y_pred),
            "num_labels": len(labels),
            "labels": labels
        }
        save_detailed_results(y_pred, y_true, labels, model_name, dataset_name, config_data)
        return results_df, y_pred
    
    return results_df 
