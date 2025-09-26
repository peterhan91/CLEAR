#!/usr/bin/env python3
"""
MONET-Adapted Model Auditing for CXR Zero-Shot Classification System
Follows MONET methodology: EfficientNetV2-S clustering + concept attribution
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
import torch.utils.data as data
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import h5py


class MONETAuditor:
    def __init__(self, results_dir: str, data_dir: str = "data", cache_dir: str = "model_auditing/cache"):
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load EfficientNetV2-S for clustering
        self.efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.efficientnet.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.efficientnet = self.efficientnet.to(self.device)
        
        # Image preprocessing for CLIP model (matching exp_zeroshot.py)
        self.clip_transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(448, interpolation=InterpolationMode.BICUBIC),
        ])
        
        # Separate preprocessing for EfficientNet (ImageNet pretrained)
        self.efficientnet_transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # MONET parameters
        self.n_clusters = 40
        self.n_pca_components = 50
        
        # Feature and clustering cache
        self._feature_cache = {}
        self._clustering_cache = {}
        
    def create_dataloader(self, dataset: str, for_efficientnet: bool = True):
        """Create dataloader - compatible with exp_zeroshot.py"""
        # Import zero_shot module to use the same dataset class
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from clear import zero_shot
        
        # Choose appropriate transform
        transform = self.efficientnet_transform if for_efficientnet else self.clip_transform
        
        # Handle dataset naming for file paths
        if dataset.endswith("_test"):
            file_dataset = dataset  # already has _test suffix
        else:
            file_dataset = f"{dataset}_test"
        
        # Use the same dataset class as exp_zeroshot.py
        test_dataset = zero_shot.CXRTestDataset(
            img_path=f"data/{file_dataset}.h5",
            transform=transform,
        )
        
        dataloader = data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
        )
        
        return dataloader
        
    def load_results(self, dataset: str, model: str = "sfr_mistral") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load existing zero-shot results"""
        # Map dataset names to results folder names
        dataset_mapping = {
            "padchest": "padchest",
            "indiana": "indiana_test",
            "indiana_test": "indiana_test", 
            "vindrcxr": "vindrcxr",
            "vindrcxr_test": "vindrcxr"
        }
        
        mapped_dataset = dataset_mapping.get(dataset, dataset)
        results_path = f"concepts/results/concept_based_evaluation_{mapped_dataset}_{model}"
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results directory not found: {results_path}")
        
        # Find latest results
        files = os.listdir(results_path)
        summary_files = [f for f in files if f.startswith('summary_')]
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {results_path}")
        
        # Extract timestamp and load latest results
        timestamps = ['_'.join(f.replace('.json', '').split('_')[-2:]) for f in summary_files if len(f.split('_')) >= 3]
        timestamp = sorted(timestamps)[-1]
        
        predictions = pd.read_csv(f"{results_path}/predictions_{timestamp}.csv")
        ground_truth = pd.read_csv(f"{results_path}/ground_truth_{timestamp}.csv")
        with open(f"{results_path}/summary_{timestamp}.json", 'r') as f:
            summary = json.load(f)
            
        print(f"✅ Loaded {dataset} data: {predictions.shape}")
        return predictions, ground_truth, summary
    
    def get_layer_feature(self, model, feature_layer_name, image):
        """Extract features from specific layer using forward hook"""
        feature_layer = model._modules.get(feature_layer_name)
        embedding = []

        def copyData(module, input, output):
            embedding.append(output.data)

        h = feature_layer.register_forward_hook(copyData)
        out = model(image.to(self.device))
        h.remove()
        embedding = embedding[0]
        assert embedding.shape[0] == image.shape[0], f"{embedding.shape[0]} != {image.shape[0]}"
        # For avgpool layer, check spatial dimensions
        if embedding.shape[2] == 1 and embedding.shape[3] == 1:
            return embedding[:, :, 0, 0]
        else:
            # Global average pooling if needed
            return torch.mean(embedding, dim=[2, 3])

    def extract_efficientnet_features(self, dataloader) -> np.ndarray:
        """Extract penultimate layer features from EfficientNetV2-S using batch processing"""
        features = []
        self.efficientnet.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                images = batch["img"]
                batch_features = self.get_layer_feature(self.efficientnet, "avgpool", images)
                features.append(batch_features.detach().cpu())
        
        return torch.cat(features, dim=0).numpy()

    def _get_features_cache_path(self, dataset: str) -> str:
        """Get cache file path for EfficientNet features"""
        return os.path.join(self.cache_dir, f"efficientnet_features_{dataset}.npy")
    
    def _get_clustering_cache_path(self, dataset: str) -> str:
        """Get cache file path for clustering results"""
        return os.path.join(self.cache_dir, f"clustering_{dataset}.npz")
    
    def _save_features_to_cache(self, dataset: str, features: np.ndarray):
        """Save EfficientNet features to cache"""
        cache_path = self._get_features_cache_path(dataset)
        try:
            np.save(cache_path, features)
            print(f"💾 Cached EfficientNet features to: {cache_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not cache features: {e}")
    
    def _load_features_from_cache(self, dataset: str) -> Optional[np.ndarray]:
        """Load EfficientNet features from cache"""
        cache_path = self._get_features_cache_path(dataset)
        
        if os.path.exists(cache_path):
            try:
                features = np.load(cache_path)
                print(f"🔄 Loaded cached EfficientNet features: {features.shape}")
                return features
            except Exception as e:
                print(f"⚠️ Warning: Could not load cached features: {e}")
                return None
        return None
    
    def _save_clustering_to_cache(self, dataset: str, cluster_labels: np.ndarray, kmeans: KMeans):
        """Save clustering results to cache"""
        cache_path = self._get_clustering_cache_path(dataset)
        try:
            np.savez(cache_path, 
                    cluster_labels=cluster_labels,
                    cluster_centers=kmeans.cluster_centers_,
                    n_clusters=self.n_clusters,
                    n_pca_components=self.n_pca_components)
            print(f"💾 Cached clustering results to: {cache_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not cache clustering results: {e}")
    
    def _load_clustering_from_cache(self, dataset: str) -> Optional[Tuple[np.ndarray, KMeans]]:
        """Load clustering results from cache"""
        cache_path = self._get_clustering_cache_path(dataset)
        
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                
                # Check if parameters match current settings
                if (data['n_clusters'] != self.n_clusters or 
                    data['n_pca_components'] != self.n_pca_components):
                    print(f"⚠️ Cached clustering parameters don't match current settings, recomputing...")
                    return None
                
                # Reconstruct KMeans object
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                kmeans.cluster_centers_ = data['cluster_centers']
                kmeans.n_clusters = self.n_clusters
                
                cluster_labels = data['cluster_labels']
                print(f"🔄 Loaded cached clustering results: {len(cluster_labels)} samples, {self.n_clusters} clusters")
                return cluster_labels, kmeans
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load cached clustering: {e}")
                return None
        return None

    def extract_efficientnet_features_cached(self, dataset: str) -> np.ndarray:
        """Extract EfficientNet features with caching"""
        # Check cache first
        cached_features = self._load_features_from_cache(dataset)
        if cached_features is not None:
            return cached_features
        
        # Extract features if not cached
        print(f"📊 Extracting EfficientNet features for {dataset}...")
        dataloader = self.create_dataloader(dataset, for_efficientnet=True)
        features = self.extract_efficientnet_features(dataloader)
        
        # Save to cache
        self._save_features_to_cache(dataset, features)
        
        return features
    
    def cluster_images(self, features: np.ndarray) -> Tuple[np.ndarray, KMeans]:
        """Apply PCA and K-means clustering"""
        print("🔄 PCA reduction...", end="")
        pca = PCA(n_components=self.n_pca_components)
        features_pca = pca.fit_transform(features)
        print(f" {features.shape} → {features_pca.shape}")
        
        print("🎯 K-means clustering...", end="")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_pca)
        print(f" {self.n_clusters} clusters")
        
        return cluster_labels, kmeans

    def cluster_images_cached(self, dataset: str) -> Tuple[np.ndarray, KMeans]:
        """Apply PCA and K-means clustering with caching"""
        # Check cache first
        cached_clustering = self._load_clustering_from_cache(dataset)
        if cached_clustering is not None:
            return cached_clustering
        
        # Extract features and cluster if not cached
        print(f"🎯 Clustering images for {dataset}...")
        features = self.extract_efficientnet_features_cached(dataset)
        cluster_labels, kmeans = self.cluster_images(features)
        
        # Save to cache
        self._save_clustering_to_cache(dataset, cluster_labels, kmeans)
        
        return cluster_labels, kmeans

    def clear_cache(self, dataset: str = None):
        """Clear cached features and clustering results"""
        if dataset is None:
            # Clear all cache
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                print("🗑️ Cleared all cached data")
        else:
            # Clear specific dataset cache
            features_path = self._get_features_cache_path(dataset)
            clustering_path = self._get_clustering_cache_path(dataset)
            
            if os.path.exists(features_path):
                os.remove(features_path)
                print(f"🗑️ Cleared cached features for {dataset}")
            
            if os.path.exists(clustering_path):
                os.remove(clustering_path)
                print(f"🗑️ Cleared cached clustering for {dataset}")
    
    def calculate_cluster_auroc_for_disease(self, cluster_labels: np.ndarray, 
                                           predictions: np.ndarray, 
                                           ground_truth: np.ndarray,
                                           disease_name: str,
                                           dataset: str, 
                                           model: str = "sfr_mistral") -> Tuple[float, Dict[int, float]]:
        """Calculate per-cluster AUROC for a specific disease"""
        # Get disease-specific overall AUROC from detailed AUCs file
        disease_auroc = self.get_disease_auroc(disease_name, dataset, model)
        print(f"📊 Overall AUROC for {disease_name}: {disease_auroc:.3f}")
        
        # Get disease index for array slicing
        disease_idx = self._get_disease_index(disease_name, dataset, model)
        if disease_idx is None:
            raise ValueError(f"Disease '{disease_name}' not found in dataset {dataset}")
        
        # Calculate per-cluster AUROC for the specific disease
        cluster_aurocs = {}
        for cluster_id in tqdm(range(self.n_clusters), desc="Computing cluster AUROCs", leave=False):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_pred = predictions[cluster_mask, disease_idx]
                cluster_gt = ground_truth[cluster_mask, disease_idx]
                
                try:
                    if len(np.unique(cluster_gt)) > 1:
                        cluster_aurocs[cluster_id] = roc_auc_score(cluster_gt, cluster_pred)
                    else:
                        cluster_aurocs[cluster_id] = np.nan
                except ValueError:
                    cluster_aurocs[cluster_id] = np.nan
        
        return disease_auroc, cluster_aurocs

    def _get_disease_index(self, disease_name: str, dataset: str, model: str = "sfr_mistral") -> Optional[int]:
        """Get the column index for a specific disease in the predictions/ground_truth arrays"""
        try:
            # Load the original predictions to get column order
            dataset_mapping = {
                "padchest": "padchest",
                "indiana": "indiana_test", 
                "indiana_test": "indiana_test",
                "vindrcxr": "vindrcxr",
                "vindrcxr_test": "vindrcxr"
            }
            
            mapped_dataset = dataset_mapping.get(dataset, dataset)
            results_dir = f"concepts/results/concept_based_evaluation_{mapped_dataset}_{model}"
            
            # Find detailed AUCs file to get column order
            detailed_files = [f for f in os.listdir(results_dir) if f.startswith("detailed_aucs_")]
            if not detailed_files:
                return None
            
            detailed_file = os.path.join(results_dir, sorted(detailed_files)[-1])
            detailed_aucs = pd.read_csv(detailed_file)
            
            # Find the index of the disease
            disease_col = f"{disease_name}_auc"
            if disease_col in detailed_aucs.columns:
                # Get column index (detailed_aucs columns should match predictions columns)
                return list(detailed_aucs.columns).index(disease_col)
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Error getting disease index for {disease_name}: {e}")
            return None

    def get_disease_auroc(self, disease_name: str, dataset: str, model: str = "sfr_mistral") -> float:
        """Get AUROC for a specific disease from detailed AUCs file"""
        try:
            dataset_mapping = {
                "padchest": "padchest",
                "indiana": "indiana_test", 
                "indiana_test": "indiana_test",
                "vindrcxr": "vindrcxr",
                "vindrcxr_test": "vindrcxr"
            }
            
            mapped_dataset = dataset_mapping.get(dataset, dataset)
            results_dir = f"concepts/results/concept_based_evaluation_{mapped_dataset}_{model}"
            
            # Find and load detailed AUCs file
            detailed_files = [f for f in os.listdir(results_dir) if f.startswith("detailed_aucs_")]
            if not detailed_files:
                raise ValueError(f"No detailed AUCs file found for {dataset}")
            
            detailed_file = os.path.join(results_dir, sorted(detailed_files)[-1])
            detailed_aucs = pd.read_csv(detailed_file)
            
            # Get disease AUROC
            disease_col = f"{disease_name}_auc"
            if disease_col in detailed_aucs.columns:
                auroc_value = detailed_aucs[disease_col].iloc[0]
                if pd.notna(auroc_value):
                    return float(auroc_value)
                else:
                    raise ValueError(f"AUROC for {disease_name} is NaN")
            else:
                raise ValueError(f"Disease '{disease_name}' not found in detailed AUCs")
                
        except Exception as e:
            print(f"⚠️ Error getting AUROC for {disease_name}: {e}")
            raise

    def find_closest_high_performing_cluster(self, low_cluster_id: int, 
                                           low_cluster_centers: np.ndarray,
                                           reference_cluster_aurocs: Dict[int, float],
                                           reference_disease_auroc: float,
                                           reference_cluster_centers: np.ndarray) -> int:
        """Find high-performing cluster from reference dataset closest to low-performing cluster
        
        Args:
            low_cluster_id: ID of the low-performing cluster
            low_cluster_centers: Cluster centers from the low-performing dataset
            reference_cluster_aurocs: Per-cluster AUROC scores for the disease in reference dataset
            reference_disease_auroc: Overall AUROC for the disease in reference dataset
            reference_cluster_centers: Cluster centers from reference dataset
        """
        low_center = low_cluster_centers[low_cluster_id]
        
        # Find clusters in reference dataset that perform better than reference dataset's overall AUROC
        high_performing = [cid for cid, auroc in reference_cluster_aurocs.items() 
                          if not np.isnan(auroc) and auroc > reference_disease_auroc]
        
        print(f"🔄 Found {len(high_performing)} high-performing clusters in reference dataset")
        
        if not high_performing:
            return None
        
        # Find closest high-performing cluster from reference dataset
        distances = [np.linalg.norm(low_center - reference_cluster_centers[hid]) 
                   for hid in high_performing]
        closest_high = high_performing[np.argmin(distances)]
        
        return closest_high
    
    def calculate_concept_presence(self, cluster_mask: np.ndarray, 
                                 concept_similarities: np.ndarray) -> np.ndarray:
        """Calculate mean concept presence scores for a cluster"""
        cluster_concepts = concept_similarities[cluster_mask]
        return np.mean(cluster_concepts, axis=0)
    


    def load_concept_similarities(self, dataset: str, model: str = "sfr_mistral") -> np.ndarray:
        """Load concept dot products from precomputed files"""
        dataset_mapping = {"padchest": "padchest_test", "indiana": "indiana_test", "indiana_test": "indiana_test", "vindrcxr": "vindrcxr", "vindrcxr_test": "vindrcxr"}
        mapped_dataset = dataset_mapping.get(dataset, dataset)
        dot_products_path = f"concepts/results/dot_products/{mapped_dataset}_dot_products.npy"
        
        if not os.path.exists(dot_products_path):
            print(f"⚠️ Concept similarities not found: {dot_products_path}")
            return None
        
        try:
            concept_similarities = np.load(dot_products_path)
            print(f"✅ Loaded concept similarities: {concept_similarities.shape}")
            return concept_similarities
        except Exception as e:
            print(f"⚠️ Error loading concept similarities: {e}")
            return None

    def load_concept_names(self) -> List[str]:
        """Load concept names from the concepts list file"""
        concepts_file = "concepts/results/dot_products/concepts_list.txt"
        if not os.path.exists(concepts_file):
            return []
        
        try:
            with open(concepts_file, 'r') as f:
                concepts = [line.split(':', 1)[1].strip() if ':' in line else line.strip() for line in f]
            print(f"✅ Loaded {len(concepts)} concept names")
            return concepts
        except Exception as e:
            print(f"⚠️ Error loading concept names: {e}")
            return []
    
    def analyze_concept_differences(self, low_cluster_id: int, 
                                  high_cluster_id: int, 
                                  cluster_labels: np.ndarray,
                                  dataset: str = None,
                                  model: str = "sfr_mistral") -> Dict:
        """Analyze concept differences between low and high-performing clusters"""
        try:
            # Load concept similarity matrix
            concept_similarities = self.load_concept_similarities(dataset, model)
            
            if concept_similarities is None:
                # Return empty result when concept similarities are not available
                return {
                    'status': 'concept_similarities_unavailable',
                    'note': 'Concept similarities not available - run save_concept_similarities.py first'
                }
            
            # Calculate concept presence scores for both clusters
            low_mask = cluster_labels == low_cluster_id
            high_mask = cluster_labels == high_cluster_id
            
            low_concepts = self.calculate_concept_presence(low_mask, concept_similarities)
            high_concepts = self.calculate_concept_presence(high_mask, concept_similarities)
            
            # Find concepts more present in low-performing cluster
            concept_diffs = low_concepts - high_concepts
            keep = concept_diffs > 0
            
            # Load concept names for interpretation
            concept_names = self.load_concept_names()
            
            # Rank all concepts by difference (using keep filter)
            ranked_concepts = []
            for i, diff in enumerate(concept_diffs):
                if keep[i]:  # Only concepts more present in low-performing cluster
                    concept_info = {
                        'concept_index': i,
                        'difference_score': float(diff),
                        'low_cluster_score': float(low_concepts[i]),
                        'high_cluster_score': float(high_concepts[i])
                    }
                    
                    # Add concept text if available
                    if i < len(concept_names):
                        concept_info['concept_text'] = concept_names[i]
                    
                    ranked_concepts.append(concept_info)
            
            # Sort by difference score (descending)
            ranked_concepts = sorted(ranked_concepts, key=lambda x: x['difference_score'], reverse=True)
            
            return {
                'top_concepts_low_performing': ranked_concepts[:20],  # Top 20 (increased from 10)
                'all_concepts_low_performing': ranked_concepts,  # All concepts for further analysis
                'concept_presence_scores': {
                    'low_cluster': float(np.mean(low_concepts)),
                    'high_cluster': float(np.mean(high_concepts))
                },
                'total_concepts': len(concept_similarities[0]),
                'num_concepts_more_present': len(ranked_concepts),
                'num_concepts_with_names': len(concept_names)
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Concept analysis failed for clusters {low_cluster_id} vs {high_cluster_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_monet_auditing(self, dataset: str, disease_name: str, 
                         model: str = "sfr_mistral", 
                         use_same_dataset_comparison: bool = False) -> Dict:
        """Run complete MONET auditing pipeline for a specific disease"""
        print(f"🔍 Running MONET auditing for {dataset} with {model}")
        print(f"🏥 Target disease: {disease_name}")
        
        try:
            # Load results
            predictions_df, ground_truth_df, summary = self.load_results(dataset, model)
            predictions = predictions_df.values
            ground_truth = ground_truth_df.values
            
            # Validate data shapes
            if predictions.shape != ground_truth.shape:
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs ground_truth {ground_truth.shape}")
            
            print(f"📊 Loaded {predictions.shape[0]} samples with {predictions.shape[1]} classes")
            
        except Exception as e:
            print(f"❌ Error in data loading: {e}")
            raise
        
        # Extract features and cluster images (with caching)
        cluster_labels, kmeans = self.cluster_images_cached(dataset)
        
        # Calculate per-cluster AUROC for the specific disease
        print("📈 Calculating cluster AUROC for disease...")
        disease_auroc, cluster_aurocs = self.calculate_cluster_auroc_for_disease(
            cluster_labels, predictions, ground_truth, disease_name, dataset, model)
        
        # Identify low-performing clusters for this disease (filter out NaN values)
        low_performing = [cid for cid, auroc in cluster_aurocs.items() 
                         if not np.isnan(auroc) and auroc < disease_auroc]
        
        print(f"🎯 Found {len(low_performing)} low-performing clusters for {disease_name}")
        
        # Create results structure
        results = {
            'dataset': dataset,
            'disease_name': disease_name,
            'model': model,
            'disease_auroc': disease_auroc,
            'n_clusters': self.n_clusters,
            'low_performing_clusters': low_performing,
            'cluster_aurocs': cluster_aurocs,
            'cluster_labels': cluster_labels.tolist(),
            'concept_attributions': {}  # Will be filled based on comparison type
        }
        
        # Perform same-dataset concept attribution analysis if requested
        if use_same_dataset_comparison:
            print("🔄 Performing same-dataset cluster analysis...")
            self._update_concept_attributions_same_dataset(results, dataset, disease_name, model)
        else:
            print("🔄 Concept attributions will be updated during cross-dataset comparison")
        
        return results
    
    def compare_datasets(self, dataset1: str, dataset2: str, disease_name: str,
                       model: str = "sfr_mistral", use_same_dataset_comparison: bool = False) -> Dict:
        """Compare performance between two datasets for a specific disease with configurable cluster analysis"""
        comparison_type = "same-dataset" if use_same_dataset_comparison else "cross-dataset"
        print(f"🔄 Comparing {dataset1} vs {dataset2} for disease: {disease_name} using {comparison_type} analysis")
        
        # Run MONET on both datasets for the specific disease
        results1 = self.run_monet_auditing(dataset1, disease_name, model, use_same_dataset_comparison)
        results2 = self.run_monet_auditing(dataset2, disease_name, model, use_same_dataset_comparison)
        
        # Perform concept attribution analysis based on comparison type
        if not use_same_dataset_comparison:
            # Cross-dataset cluster analysis
            print("🔄 Performing cross-dataset cluster analysis...")
            self._update_concept_attributions_cross_dataset(results1, results2, dataset1, dataset2, disease_name, model)
        else:
            # Same-dataset analysis already performed in run_monet_auditing
            print("✅ Same-dataset cluster analysis completed for both datasets")
        
        # Calculate disease-specific performance gap
        disease_gap = results1['disease_auroc'] - results2['disease_auroc']
        
        comparison = {
            'dataset1': dataset1,
            'dataset2': dataset2,
            'disease_name': disease_name,
            'model': model,
            'comparison_type': comparison_type,
            'disease_auroc1': results1['disease_auroc'],
            'disease_auroc2': results2['disease_auroc'],
            'disease_auroc_difference': disease_gap,
            'results1': results1,
            'results2': results2
        }
        
        return comparison

    def _update_concept_attributions_cross_dataset(self, results1: Dict, results2: Dict, 
                                                  dataset1: str, dataset2: str, 
                                                  disease_name: str, model: str):
        """Update concept attributions using cross-dataset cluster analysis"""
        
        # Get cluster centers for both datasets (using cache)
        _, kmeans1 = self.cluster_images_cached(dataset1)
        _, kmeans2 = self.cluster_images_cached(dataset2)
        
        # Update concept attributions for dataset1 (using dataset2 as reference)
        updated_attributions1 = {}
        for low_cluster_id in results1['low_performing_clusters']:
            high_cluster_id = self.find_closest_high_performing_cluster(
                low_cluster_id, 
                kmeans1.cluster_centers_,  # low dataset centers
                results2['cluster_aurocs'],  # reference dataset AUROCs
                results2['disease_auroc'],   # reference disease AUROC
                kmeans2.cluster_centers_     # reference dataset centers
            )
            
            if high_cluster_id is not None:
                # Analyze concepts between dataset1 low cluster and dataset2 high cluster
                concept_differences = self.analyze_concept_differences(
                    low_cluster_id, high_cluster_id, 
                    np.array(results1['cluster_labels']), dataset1, model
                )
                
                updated_attributions1[f"cluster_{low_cluster_id}"] = {
                    'low_cluster_id': low_cluster_id,
                    'low_dataset': dataset1,
                    'high_cluster_id': high_cluster_id,
                    'high_dataset': dataset2,
                    'low_auroc': results1['cluster_aurocs'][low_cluster_id],
                    'high_auroc': results2['cluster_aurocs'][high_cluster_id],
                    'concept_differences': concept_differences,
                    'comparison_type': 'cross_dataset'
                }
        
        # Update concept attributions for dataset2 (using dataset1 as reference)
        updated_attributions2 = {}
        for low_cluster_id in results2['low_performing_clusters']:
            high_cluster_id = self.find_closest_high_performing_cluster(
                low_cluster_id,
                kmeans2.cluster_centers_,  # low dataset centers
                results1['cluster_aurocs'],  # reference dataset AUROCs
                results1['disease_auroc'],   # reference disease AUROC
                kmeans1.cluster_centers_     # reference dataset centers
            )
            
            if high_cluster_id is not None:
                concept_differences = self.analyze_concept_differences(
                    low_cluster_id, high_cluster_id,
                    np.array(results2['cluster_labels']), dataset2, model
                )
                
                updated_attributions2[f"cluster_{low_cluster_id}"] = {
                    'low_cluster_id': low_cluster_id,
                    'low_dataset': dataset2,
                    'high_cluster_id': high_cluster_id,
                    'high_dataset': dataset1,
                    'low_auroc': results2['cluster_aurocs'][low_cluster_id],
                    'high_auroc': results1['cluster_aurocs'][high_cluster_id],
                    'concept_differences': concept_differences,
                    'comparison_type': 'cross_dataset'
                }
        
        # Update the results with cross-dataset concept attributions
        results1['concept_attributions'] = updated_attributions1
        results2['concept_attributions'] = updated_attributions2

    def _update_concept_attributions_same_dataset(self, results: Dict, dataset: str, 
                                                 disease_name: str, model: str):
        """Update concept attributions using same-dataset cluster analysis (low vs high performing)"""
        
        # Get cluster centers for the dataset
        _, kmeans = self.cluster_images_cached(dataset)
        
        # Find high-performing clusters within the same dataset
        high_performing = [cid for cid, auroc in results['cluster_aurocs'].items() 
                          if not np.isnan(auroc) and auroc > results['disease_auroc']]
        
        print(f"🔄 Found {len(high_performing)} high-performing clusters in {dataset}")
        print(f"🔄 Found {len(results['low_performing_clusters'])} low-performing clusters in {dataset}")
        
        # Update concept attributions for low-performing clusters
        updated_attributions = {}
        for low_cluster_id in results['low_performing_clusters']:
            high_cluster_id = self.find_closest_high_performing_cluster(
                low_cluster_id, 
                kmeans.cluster_centers_,  # same dataset centers
                results['cluster_aurocs'],  # same dataset AUROCs
                results['disease_auroc'],   # same dataset disease AUROC
                kmeans.cluster_centers_     # same dataset centers (reference)
            )
            
            if high_cluster_id is not None:
                # Analyze concepts between low and high clusters in same dataset
                concept_differences = self.analyze_concept_differences(
                    low_cluster_id, high_cluster_id, 
                    np.array(results['cluster_labels']), dataset, model
                )
                
                updated_attributions[f"cluster_{low_cluster_id}"] = {
                    'low_cluster_id': low_cluster_id,
                    'low_dataset': dataset,
                    'high_cluster_id': high_cluster_id,
                    'high_dataset': dataset,  # Same dataset
                    'low_auroc': results['cluster_aurocs'][low_cluster_id],
                    'high_auroc': results['cluster_aurocs'][high_cluster_id],
                    'concept_differences': concept_differences,
                    'comparison_type': 'same_dataset'
                }
            else:
                print(f"⚠️ No high-performing cluster found for low cluster {low_cluster_id} in {dataset}")
        
        # Update the results with same-dataset concept attributions
        results['concept_attributions'] = updated_attributions
        print(f"🔬 {dataset}: {len(updated_attributions)} same-dataset cluster comparisons completed")

    def save_worst_cluster_images(self, dataset: str, disease_name: str, 
                                model: str = "sfr_mistral", 
                                top_k: int = 5,
                                max_images_per_cluster: int = 20,
                                output_dir: str = "model_auditing/worst_clusters_analysis") -> Dict:
        """
        Save images from the top-k worst performing clusters as JPG files
        
        Args:
            dataset: Dataset name (e.g., "padchest", "indiana_test")
            disease_name: Disease to analyze
            model: Model name
            top_k: Number of worst clusters to save (default: 5)
            max_images_per_cluster: Maximum images to save per cluster
            output_dir: Base directory for saving images
            
        Returns:
            Dictionary with saving statistics and metadata
        """
        print(f"🖼️ Saving images from top-{top_k} worst performing clusters for {disease_name} in {dataset}")
        
        try:
            # Load existing results
            predictions_df, ground_truth_df, summary = self.load_results(dataset, model)
            predictions = predictions_df.values
            ground_truth = ground_truth_df.values
            
            # Get clustering results
            cluster_labels, _ = self.cluster_images_cached(dataset)
            
            # Calculate per-cluster AUROC for the disease
            disease_auroc, cluster_aurocs = self.calculate_cluster_auroc_for_disease(
                cluster_labels, predictions, ground_truth, disease_name, dataset, model)
            
            # Get disease index for extracting predictions/ground truth
            disease_idx = self._get_disease_index(disease_name, dataset, model)
            if disease_idx is None:
                raise ValueError(f"Disease '{disease_name}' not found in dataset {dataset}")
            
            # Sort clusters by AUROC (worst first, excluding NaN)
            valid_clusters = [(cid, auroc) for cid, auroc in cluster_aurocs.items() if not np.isnan(auroc)]
            worst_clusters = sorted(valid_clusters, key=lambda x: x[1])[:top_k]
            
            print(f"🎯 Identified {len(worst_clusters)} worst performing clusters:")
            for i, (cluster_id, auroc) in enumerate(worst_clusters):
                print(f"  {i+1}. Cluster {cluster_id}: AUROC = {auroc:.3f}")
            
            # Create output directory structure
            analysis_dir = os.path.join(output_dir, f"{dataset}_{disease_name}_{model}")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Load original images from H5 file
            if dataset.endswith("_test"):
                h5_dataset = dataset
            else:
                h5_dataset = f"{dataset}_test"
            h5_path = f"data/{h5_dataset}.h5"
            
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"H5 file not found: {h5_path}")
            
            # Load CSV to get original image names
            dataset_configs = {
                "padchest": {
                    "csv_path": "data/padchest_test.csv",
                    "image_id_col": "ImageID"
                },
                "vindrcxr": {
                    "csv_path": "data/vindrcxr_test.csv",
                    "image_id_col": "image_id"
                },
                "indiana_test": {
                    "csv_path": "data/indiana_test.csv",
                    "image_id_col": "filename"
                }
            }
            
            # Get the correct dataset key for CSV lookup
            csv_dataset_key = dataset if dataset in dataset_configs else h5_dataset
            if csv_dataset_key not in dataset_configs:
                raise ValueError(f"Dataset configuration not found for: {dataset}")
            
            csv_config = dataset_configs[csv_dataset_key]
            csv_path = csv_config["csv_path"]
            image_id_col = csv_config["image_id_col"]
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Load the CSV to get original image names
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Process each worst cluster
            cluster_stats = {}
            total_images_saved = 0
            
            for rank, (cluster_id, cluster_auroc) in enumerate(worst_clusters):
                print(f"📁 Processing cluster {cluster_id} (rank {rank+1}/{len(worst_clusters)})...")
                
                # Create cluster directory
                cluster_dir = os.path.join(analysis_dir, f"cluster_{cluster_id:02d}_auroc_{cluster_auroc:.3f}")
                images_dir = os.path.join(cluster_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                # Get indices of images in this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Limit number of images per cluster
                if len(cluster_indices) > max_images_per_cluster:
                    # Sort by prediction confidence (save most confident wrong predictions)
                    cluster_preds = predictions[cluster_mask, disease_idx]
                    cluster_gts = ground_truth[cluster_mask, disease_idx]
                    
                    # Calculate prediction errors (higher = more wrong)
                    errors = np.abs(cluster_preds - cluster_gts)
                    sorted_idx = np.argsort(errors)[::-1]  # Worst predictions first
                    
                    selected_indices = cluster_indices[sorted_idx[:max_images_per_cluster]]
                else:
                    selected_indices = cluster_indices
                
                # Save images
                images_saved = 0
                with h5py.File(h5_path, 'r') as h5_file:
                    for img_idx in tqdm(selected_indices, desc=f"Saving cluster {cluster_id} images", leave=False):
                        try:
                            # Load original image
                            img_array = h5_file['cxr'][img_idx]
                            
                            # Convert to PIL Image
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                            
                            # Create PIL image (grayscale)
                            pil_image = Image.fromarray(img_array, mode='L')
                            
                            # Get prediction and ground truth for this image
                            pred = predictions[img_idx, disease_idx]
                            gt = ground_truth[img_idx, disease_idx]
                            
                            # Get original image name from CSV
                            try:
                                if img_idx < len(df):
                                    original_name = str(df.iloc[img_idx][image_id_col])
                                    # Remove extension if present and clean up the name
                                    original_name = os.path.splitext(original_name)[0]
                                    # Create filename with original name and metadata
                                    filename = f"{original_name}_pred_{pred:.3f}_gt_{int(gt)}.jpg"
                                else:
                                    # Fallback to index-based naming if CSV doesn't have this index
                                    filename = f"img_{img_idx:05d}_pred_{pred:.3f}_gt_{int(gt)}.jpg"
                            except Exception as e:
                                print(f"⚠️ Warning: Could not get original name for image {img_idx}: {e}")
                                filename = f"img_{img_idx:05d}_pred_{pred:.3f}_gt_{int(gt)}.jpg"
                            
                            filepath = os.path.join(images_dir, filename)
                            
                            # Save as JPEG
                            pil_image.save(filepath, 'JPEG', quality=95)
                            images_saved += 1
                            
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to save image {img_idx}: {e}")
                            continue
                
                # Save cluster metadata
                cluster_metadata = {
                    'cluster_id': int(cluster_id),
                    'cluster_auroc': float(cluster_auroc),
                    'rank': rank + 1,
                    'total_images_in_cluster': int(len(cluster_indices)),
                    'images_saved': images_saved,
                    'disease_name': disease_name,
                    'dataset': dataset,
                    'model': model,
                    'overall_disease_auroc': float(disease_auroc),
                    'performance_gap': float(disease_auroc - cluster_auroc)
                }
                
                with open(os.path.join(cluster_dir, "cluster_metadata.json"), 'w') as f:
                    json.dump(cluster_metadata, f, indent=2)
                
                cluster_stats[f"cluster_{cluster_id}"] = cluster_metadata
                total_images_saved += images_saved
                
                print(f"  ✅ Saved {images_saved} images to {images_dir}")
            
            # Save overall analysis summary
            analysis_summary = {
                'dataset': dataset,
                'disease_name': disease_name,
                'model': model,
                'overall_disease_auroc': float(disease_auroc),
                'top_k_clusters': top_k,
                'max_images_per_cluster': max_images_per_cluster,
                'total_images_saved': total_images_saved,
                'cluster_stats': cluster_stats,
                'h5_source_path': h5_path,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_path = os.path.join(analysis_dir, "analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(analysis_summary, f, indent=2)
            
            print(f"\n✅ Image saving completed!")
            print(f"📁 Output directory: {analysis_dir}")
            print(f"🖼️ Total images saved: {total_images_saved}")
            print(f"📊 Analysis summary: {summary_path}")
            
            return analysis_summary
            
        except Exception as e:
            print(f"❌ Error saving cluster images: {e}")
            raise


def main(disease_name: str = "cardiomegaly", use_same_dataset_comparison: bool = False):
    """Main execution function for single disease analysis
    
    Args:
        disease_name: Specific disease to analyze
        use_same_dataset_comparison: If True, compare low vs high clusters within same dataset.
                                   If False, compare across datasets (default behavior).
    """
    auditor = MONETAuditor("concepts/results")
    
    comparison_type = "same-dataset" if use_same_dataset_comparison else "cross-dataset"
    print(f"🔬 Analysis type: {comparison_type} cluster comparison")
    
    # Compare datasets for specific disease
    comparison = auditor.compare_datasets(
        "padchest", "indiana_test", disease_name,
        model="sfr_mistral", use_same_dataset_comparison=use_same_dataset_comparison
    )
    
    # Save results
    os.makedirs("model_auditing/results", exist_ok=True)
    
    # Save main results
    with open("model_auditing/results/monet_auditing_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    with open("model_auditing/results/concept_differences.json", "w") as f:
        concept_data = {
            'disease_name': comparison['disease_name'],
            'concept_attributions': {
                'padchest': comparison['results1'].get('concept_attributions', {}),
                'indiana': comparison['results2'].get('concept_attributions', {})
            }
        }
        json.dump(concept_data, f, indent=2)
    
    print("\n" + "="*60)
    print("📋 ANALYSIS SUMMARY")
    print("="*60)
    print("✅ MONET auditing completed!")
    print("📁 Main results saved to: model_auditing/results/monet_auditing_results.json")
    print("📁 Concept differences saved to: model_auditing/results/concept_differences.json")
    print(f"🏥 Disease: {comparison['disease_name']}")
    print(f"📊 PadChest AUROC: {comparison['disease_auroc1']:.3f}")
    print(f"📊 Indiana AUROC: {comparison['disease_auroc2']:.3f}")
    print(f"📈 Performance gap: {comparison['disease_auroc_difference']:.3f}")
    
    # Print cluster analysis summary
    results1 = comparison['results1']
    results2 = comparison['results2']
    print(f"🎯 PadChest: {len(results1['low_performing_clusters'])} low-performing clusters")
    print(f"🎯 Indiana: {len(results2['low_performing_clusters'])} low-performing clusters")
    print("="*60)


if __name__ == "__main__":
    main() 
