#!/usr/bin/env python3
import os
import sys
import datetime
import json
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import time
import pickle
from tqdm import tqdm
from openai import AzureOpenAI
import gc

import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
from clear import clip, zero_shot
from clear.evaluation import evaluate
from get_embed import RadiologyEmbeddingGenerator


def print_gpu_memory(stage=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"🖥️ GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("🖥️ CUDA not available")


class LLMEmbeddingGenerator:
    def __init__(self):
        # Load API credentials from environment variables for security
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "sth")
        api_key = os.getenv("AZURE_OPENAI_KEY", "sth")
        
        self.openai_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
        self.openai_model = "text-embedding-3-small"
        
        # Cache for local embedding generators to avoid reloading
        self._local_generators = {}

    def get_openai_embedding(self, text: str, max_retries: int = 3) -> np.ndarray:
        cleaned_text = text.strip().replace("\\n", " ").replace("\n", " ")
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    input=[cleaned_text],
                    model=self.openai_model
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"Rate limit, waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt)
        return np.zeros(1536)  # OpenAI embedding dimension

    def get_embeddings_batch(self, texts: List[str], model_name: str) -> np.ndarray:
        embeddings = []
        if model_name == "openai_small":
            for text in tqdm(texts, desc=f"Getting {model_name} embeddings"):
                embedding = self.get_openai_embedding(text)
                time.sleep(0.1)
                embeddings.append(embedding)
            return np.array(embeddings)
        elif model_name in ["sfr_mistral", "qwen3_8b", "biomedbert"]:
            # Check if we already have this generator cached
            if model_name not in self._local_generators:
                print(f"🔄 Loading {model_name} model for the first time...")
                # Before loading, free all unused memory
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Create and cache the generator
                if model_name == "sfr_mistral":
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='Salesforce/SFR-Embedding-Mistral',
                        batch_size=16
                    )
                elif model_name == "qwen3_8b":
                    # Use smaller batch size for very large models
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='Qwen/Qwen3-Embedding-8B',
                        batch_size=4  # Reduced from 16 to 4 for memory efficiency
                    )
                elif model_name == "biomedbert":
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                        batch_size=16
                    )
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
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
        for model_name in self._local_generators:
            print(f"🧹 Cleaning up {model_name} model...")
            # The cleanup depends on the RadiologyEmbeddingGenerator implementation
            del self._local_generators[model_name]
        self._local_generators.clear()
        
        # More aggressive cleanup for large models
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
        print("✅ All models cleaned up")


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    if dataset_name == "chexpert":
        return {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.csv",
            'labels': ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema',
                      'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                      'Lung Opacity', 'No Finding','Pleural Effusion',
                      'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'],
            'core_conditions': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        }
    elif dataset_name == "padchest":
        # Get all disease labels from CSV
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv")
        all_disease_labels = [col for col in df.columns if col not in ['ImageID', 'name', 'Path', 'is_test']]   
        return {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv",
            'labels': all_disease_labels,
            'core_conditions': ['pleural effusion', 'atelectasis', 'pneumonia', 'pneumothorax', 'normal']
        }
    elif dataset_name == "vindrcxr":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv")
        all_disease_labels = [col for col in df.columns if col not in ['image_id', 'name', 'Path', 'is_test']]
        return {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv",
            'labels': all_disease_labels,
            'core_conditions': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        }
    elif dataset_name == "indiana_test":
        df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv")
        # Exclude Indiana-specific non-pathology columns
        exclude_cols = ['uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression']
        all_disease_labels = [col for col in df.columns if col not in exclude_cols]
        return {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv",
            'labels': all_disease_labels,
            'core_conditions': ['pulmonary atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema', 'pleural effusion', 'pneumonia', 'pneumothorax']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_concept_embeddings(model_name: str) -> Tuple[Dict[int, np.ndarray], int]:
    """Load concept embeddings for a specific model"""
    embeddings_file = f"/home/than/DeepLearning/cxr_concept/CheXzero/concepts/embeddings/concepts_embeddings_{model_name}.pickle"
    
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Get embedding dimension from first embedding
    embedding_dim = len(list(embeddings_data.values())[0])
    
    return embeddings_data, embedding_dim

def run_concept_based_evaluation(dataset_name="chexpert", llm_model="openai"):
    """Run concept-based evaluation using diagnostic concepts and specified LLM embeddings"""
    
    print(f"=== Concept-Based CLIP + {llm_model.upper()} Evaluation: {dataset_name.upper()} ===")
    
    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    test_labels = config['labels']
    core_conditions = config['core_conditions']
    
    print(f"✅ Dataset: {dataset_name}")
    print(f"✅ Number of labels: {len(test_labels)}")
    print(f"✅ Core conditions: {core_conditions}")
    
    # Initialize embedding generator once at the beginning
    print(f"\n=== Step 0: Initialize {llm_model.upper()} Embedding Generator ===")
    print_gpu_memory("before generator init")
    embedding_generator = LLMEmbeddingGenerator()
    print(f"✅ {llm_model.upper()} embedding generator initialized")
    print_gpu_memory("after generator init")
    
    print("\n=== Step 1: Load CLIP Model ===")
    model = load_clip(
        model_path="../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ CLIP model loaded and moved to CUDA")
    print_gpu_memory("after CLIP model load")
    
    print("\n=== Step 2: Setup Dataset ===")
    y_true = zero_shot.make_true_labels(
        cxr_true_labels_path=config['labels_path'],
        cxr_labels=test_labels,
        cutlabels=True
    )
    
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    test_dataset = zero_shot.CXRTestDataset(
        img_path=config['cxr_filepath'],
        transform=transform,
    )
    
    # Adjust batch size for large models to prevent OOM
    image_batch_size = 32 if llm_model == "qwen3_8b" else 64
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=image_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} images")
    print(f"✅ Ground truth labels shape: {y_true.shape}")
    
    print("\n=== Step 3: Load Diagnostic Concepts ===")
    concepts_df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()
    print(f"✅ Loaded {len(concepts)} diagnostic concepts")
    
    print(f"\n=== Step 4: Load {llm_model.upper()} Concept Embeddings ===")
    embeddings_data, embedding_dim = load_concept_embeddings(llm_model)
    
    # Process concept embeddings with proper alignment
    concept_embeddings = np.zeros((len(concepts), embedding_dim))
    missing_count = 0
    
    for pos, concept_idx in enumerate(concept_indices):
        if concept_idx in embeddings_data:
            concept_embeddings[pos] = embeddings_data[concept_idx]
        else:
            concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} concepts missing embeddings, used random vectors")
    else:
        print("✅ All concepts have corresponding embeddings")
    
    concept_embeddings = torch.tensor(concept_embeddings).float()
    print(f"✅ Concept embeddings shape: {concept_embeddings.shape}")
    
    print("\n=== Step 5: Get Concept Similarity Scores ===")
    # Process concepts in batches to avoid CUDA OOM - reduce batch size for large models
    concept_batch_size = 512 if llm_model == "qwen3_8b" else 1024
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_features = torch.cat(all_concept_features).to('cuda')
    print(f"✅ Concept features shape: {concept_features.shape}")
    
    # Encode images and compute concept similarities
    all_img_features = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Encoding images"):
            imgs = data['img'].to('cuda')
            img_features = model.encode_image(imgs)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            all_img_features.append(img_features.cpu())
    
    img_features = torch.cat(all_img_features)
    print(f"✅ Image features shape: {img_features.shape}")
    
    # Compute concept similarities in batches - reduce for large models
    img_batch_size = 50 if llm_model == "qwen3_8b" else 100
    all_similarities = []
    
    for i in tqdm(range(0, len(img_features), img_batch_size), desc="Computing concept similarities"):
        batch_img_features = img_features[i:i+img_batch_size].to('cuda')
        batch_similarity = batch_img_features @ concept_features.T
        all_similarities.append(batch_similarity.cpu())
        torch.cuda.empty_cache()
    
    concept_similarity = torch.cat(all_similarities)
    print(f"✅ Concept similarity shape: {concept_similarity.shape}")
    
    print("\n=== Step 6: Matrix Multiply to Get LLM Representation ===")
    concept_embeddings = concept_embeddings.to('cuda')
    concept_similarity = concept_similarity.to('cuda')
    
    llm_representation = concept_similarity @ concept_embeddings
    llm_representation /= llm_representation.norm(dim=-1, keepdim=True)
    print(f"✅ LLM representation shape: {llm_representation.shape}")
    
    print(f"\n=== Step 7: Generate {llm_model.upper()} Embeddings for Class Prompts ===")
    # For very large models like Qwen 3 8B, temporarily move CLIP model to CPU to free GPU memory
    if llm_model == "qwen3_8b":
        print("⚠️ Large model detected - temporarily moving CLIP to CPU to free GPU memory")
        model = model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Reuse the same embedding generator from Step 0
    
    # Generate class prompts
    pos_prompts = [f"{label.lower()}" for label in test_labels]
    neg_prompts = [f"no {label.lower()}" for label in test_labels]
    
    print(f"✅ Sample positive prompts: {pos_prompts[:3]}")
    print(f"✅ Sample negative prompts: {neg_prompts[:3]}")
    
    # Get embeddings for prompts (reusing the same generator)
    print(f"Getting {llm_model.upper()} embeddings for prompts...")
    print_gpu_memory("before embedding generation")
    pos_embeddings = embedding_generator.get_embeddings_batch(pos_prompts, llm_model)
    neg_embeddings = embedding_generator.get_embeddings_batch(neg_prompts, llm_model)
    print_gpu_memory("after embedding generation")
    
    # Move CLIP model back to GPU if it was moved to CPU
    if llm_model == "qwen3_8b":
        print("🔄 Moving CLIP model back to GPU after embedding generation")
        model = model.to('cuda')
        torch.cuda.empty_cache()
    
    # Convert to tensors
    pos_class_embeddings = torch.tensor(np.array(pos_embeddings)).float().to('cuda')
    neg_class_embeddings = torch.tensor(np.array(neg_embeddings)).float().to('cuda')
    pos_class_embeddings /= pos_class_embeddings.norm(dim=-1, keepdim=True)
    neg_class_embeddings /= neg_class_embeddings.norm(dim=-1, keepdim=True)
    
    print(f"✅ Positive {llm_model.upper()} embeddings shape: {pos_class_embeddings.shape}")
    print(f"✅ Negative {llm_model.upper()} embeddings shape: {neg_class_embeddings.shape}")
    
    print("\n=== Step 8: Compute Predictions ===")
    with torch.no_grad():
        # Calculate similarities between LLM representation and class embeddings
        logits_pos = llm_representation @ pos_class_embeddings.T
        logits_neg = llm_representation @ neg_class_embeddings.T
        
        # Apply softmax to get probabilities
        exp_logits_pos = torch.exp(logits_pos)
        exp_logits_neg = torch.exp(logits_neg)
        probabilities = exp_logits_pos / (exp_logits_pos + exp_logits_neg)
        
        y_pred = probabilities.cpu().numpy()
    
    print(f"✅ Predictions shape: {y_pred.shape}")
    
    print("\n=== Step 9: Evaluate Performance ===")
    results_df = evaluate(y_pred, y_true, test_labels)
    print("📊 Performance Results:")
    print(results_df)
    
    # Calculate average AUC for core conditions
    core_aucs = []
    for condition in core_conditions:
        if f"{condition}_auc" in results_df.columns:
            auc_value = results_df[f"{condition}_auc"].iloc[0]
            core_aucs.append(auc_value)
    
    avg_auc = np.mean(core_aucs) if core_aucs else results_df.mean().mean()
    
    print(f"\n🎯 FINAL RESULTS ({dataset_name.upper()}) - {llm_model.upper()}:")
    print(f"📈 Average AUC (core conditions): {avg_auc:.4f}")
    print(f"📊 Number of test images: {len(y_pred)}")
    print(f"🏷️ Number of classes: {len(test_labels)}")
    print(f"🧠 Number of concepts: {len(concepts)}")
    
    print(f"\n📋 Individual AUCs (core conditions):")
    for condition in core_conditions:
        if f"{condition}_auc" in results_df.columns:
            auc_value = results_df[f"{condition}_auc"].iloc[0]
            print(f"  {condition}: {auc_value:.4f}")
    
    # Save results
    save_results(results_df, y_pred, y_true, test_labels, dataset_name, avg_auc, config, len(concepts), llm_model)
    
    # Cleanup embedding generator to free GPU memory
    print(f"\n=== Cleanup ===")
    embedding_generator.cleanup()
    
    return results_df, y_pred, avg_auc

def save_results(results_df, y_pred, y_true, test_labels, dataset_name, avg_auc, config, num_concepts, llm_model):
    """Save evaluation results"""
    print(f"\n=== Saving Results ({dataset_name.upper()} - {llm_model.upper()}) ===")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = f"results/concept_based_evaluation_{dataset_name}_{llm_model}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Save performance metrics
    results_summary = {
        'timestamp': timestamp,
        'method': 'concept_based_clip_gpt',
        'dataset': dataset_name,
        'llm_model': llm_model,
        'model': 'best_model.pt',
        'test_images': len(y_pred),
        'num_classes': len(test_labels),
        'num_concepts': num_concepts,
        'avg_auc': avg_auc,
        'individual_aucs': {}
    }
    
    for label in test_labels:
        if f"{label}_auc" in results_df.columns:
            results_summary['individual_aucs'][label] = float(results_df[f"{label}_auc"].iloc[0])
    
    # Save summary as JSON
    with open(f"{results_dir}/summary_{timestamp}.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed results as CSV
    results_df.to_csv(f"{results_dir}/detailed_aucs_{timestamp}.csv", index=False)
    
    # 2. Save predictions and ground truth
    predictions_df = pd.DataFrame(y_pred, columns=[f"{label}_pred" for label in test_labels])
    predictions_df.to_csv(f"{results_dir}/predictions_{timestamp}.csv", index=False)
    
    gt_df = pd.DataFrame(y_true, columns=[f"{label}_true" for label in test_labels])
    gt_df.to_csv(f"{results_dir}/ground_truth_{timestamp}.csv", index=False)
    
    # 3. Save configuration
    config_save = {
        'dataset': dataset_name,
        'method': 'concept_based_clip_gpt',
        'llm_model': llm_model,
        'test_batch_size': 32 if llm_model == "qwen3_8b" else 64,
        'concept_batch_size': 512 if llm_model == "qwen3_8b" else 1024,
        'img_batch_size': 50 if llm_model == "qwen3_8b" else 100,
        'templates': ('{}', 'no {}'),
        'clip_model_path': '../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt',
        'concepts_file': 'concepts/mimic_concepts.csv',
        'concept_embeddings_file': f'concepts/embeddings/concepts_embeddings_{llm_model}.pickle',
        'cxr_filepath': config['cxr_filepath'],
        'labels_path': config['labels_path'],
        'normalization': {
            'mean': [101.48761, 101.48761, 101.48761],
            'std': [83.43944, 83.43944, 83.43944]
        },
        'resolution': 448
    }
    
    with open(f"{results_dir}/config_{timestamp}.json", 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"✅ Results saved to {results_dir}/")
    print(f"   📊 Summary: summary_{timestamp}.json")
    print(f"   📈 AUC scores: detailed_aucs_{timestamp}.csv") 
    print(f"   🎯 Predictions: predictions_{timestamp}.csv")
    print(f"   📋 Ground truth: ground_truth_{timestamp}.csv")
    print(f"   ⚙️ Configuration: config_{timestamp}.json")

if __name__ == "__main__":
    print("🚀 Starting Multi-LLM Concept-Based Evaluation")
    
    # Define datasets and LLM models
    datasets = [
                # "chexpert", 
                # "padchest",
                # "vindrcxr",
                "indiana_test"
                ]
    llm_models = [
                "openai_small", 
                "qwen3_8b", 
                "sfr_mistral", 
                "biomedbert"
                ]
    
    # Store all results
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        
        for llm_model in llm_models:
            try:
                print(f"\n{'='*60}")
                print(f"🔬 Evaluating {dataset.upper()} Dataset with {llm_model.upper()}")
                print('='*60)
                
                results_df, y_pred, avg_auc = run_concept_based_evaluation(dataset, llm_model)
                all_results[dataset][llm_model] = {
                    'results_df': results_df,
                    'avg_auc': avg_auc,
                    'num_images': len(y_pred)
                }
                
                print(f"\n✅ {dataset.upper()} evaluation with {llm_model.upper()} completed!")
                print(f"   📈 Average AUC: {avg_auc:.4f}")
                print(f"   📊 Images evaluated: {len(y_pred)}")
                
            except Exception as e:
                print(f"❌ Error evaluating {dataset} with {llm_model}: {e}")
                continue
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📋 FINAL COMPARISON SUMMARY")
    print('='*60)
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        for llm_model in llm_models:
            if dataset in all_results and llm_model in all_results[dataset]:
                results = all_results[dataset][llm_model]
                print(f"  {llm_model.upper()}:")
                print(f"    📈 Average AUC: {results['avg_auc']:.4f}")
                print(f"    📊 Images: {results['num_images']}")
    
    print("\n✅ Multi-LLM concept-based evaluation completed successfully!") 
