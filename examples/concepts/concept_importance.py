#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
import sys
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concepts.exp_linear import LogisticRegressionModel, fix_numpy_compatibility

class ConceptImportanceAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load results
        with open(f"{results_dir}/results.json", 'r') as f:
            self.results = json.load(f)
        
        self.labels = self.results['labels']
        
        # Load model
        input_dim = self.results['feature_dim']
        output_dim = len(self.labels)
        self.model = LogisticRegressionModel(input_dim, output_dim)
        self.model.load_state_dict(torch.load(f"{results_dir}/model.pth", map_location='cpu'))
        self.model.eval()
        
        self._load_concepts()
        
    def _load_concepts(self) -> None:
        """Load diagnostic concepts and embeddings"""
        print("Loading concepts...")
        
        # Load concepts with their indices
        concepts_df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/mimic_concepts.csv")
        self.concepts = concepts_df['concept'].tolist()
        self.concept_indices = concepts_df['concept_idx'].tolist()
        
        # Load concept embeddings
        fix_numpy_compatibility()
        with open("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/embeddings/concepts_embeddings_sfr_mistral.pickle", 'rb') as f:
            embeddings_data = pickle.load(f)
        
        if isinstance(embeddings_data, dict):
            embedding_dim = len(list(embeddings_data.values())[0])
            concept_embeddings = np.zeros((len(self.concepts), embedding_dim))
            
            for pos, concept_idx in enumerate(self.concept_indices):
                if concept_idx in embeddings_data:
                    concept_embeddings[pos] = embeddings_data[concept_idx]
                else:
                    concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01
        else:
            concept_embeddings = np.array(embeddings_data)
        
        self.concept_embeddings = torch.tensor(concept_embeddings).float()
        print(f"Loaded {len(self.concepts)} concepts")
    
    def analyze_concept_importance(self, k: int = 100) -> Dict:
        """Analyze positive and negative concept importance separately"""
        print(f"\nAnalyzing concept importance (Top {k} positive/negative per label)")
        
        weights = self.model.linear.weight.detach().cpu().numpy()  # [num_labels, feature_dim]
        concept_importance = {}
        
        for label_idx, label in enumerate(self.labels):
            label_weights = weights[label_idx]  # [feature_dim]
            concept_emb_np = self.concept_embeddings.numpy()  # [num_concepts, feature_dim]
            
            # Compute cosine similarity alignments
            alignments = []
            for i in range(len(self.concepts)):
                concept_emb = concept_emb_np[i]
                alignment = np.dot(label_weights, concept_emb) / (
                    np.linalg.norm(label_weights) * np.linalg.norm(concept_emb)
                )
                alignments.append(alignment)
            
            alignments = np.array(alignments)
            
            # Separate positive and negative alignments
            positive_indices = np.where(alignments > 0)[0]
            negative_indices = np.where(alignments < 0)[0]
            
            # Sort by magnitude within each group
            positive_sorted = positive_indices[np.argsort(alignments[positive_indices])[::-1]]
            negative_sorted = negative_indices[np.argsort(alignments[negative_indices])]  # Most negative first
            
            # Get top k from each group
            top_positive = []
            for idx in positive_sorted[:k]:
                top_positive.append({
                    'concept': self.concepts[idx],
                    'concept_idx': self.concept_indices[idx],
                    'alignment': float(alignments[idx])
                })
            
            top_negative = []
            for idx in negative_sorted[:k]:
                top_negative.append({
                    'concept': self.concepts[idx],
                    'concept_idx': self.concept_indices[idx],
                    'alignment': float(alignments[idx])
                })
            
            concept_importance[label] = {
                'positive_concepts': top_positive,
                'negative_concepts': top_negative,
                'stats': {
                    'total_positive': len(positive_indices),
                    'total_negative': len(negative_indices),
                    'max_positive': float(np.max(alignments[positive_indices])) if len(positive_indices) > 0 else 0.0,
                    'min_negative': float(np.min(alignments[negative_indices])) if len(negative_indices) > 0 else 0.0,
                    'auc': self.results['per_label_aucs'][label]
                }
            }
            
            # Print summary
            print(f"\n{label}: {len(positive_indices)} positive, {len(negative_indices)} negative concepts")
            
            print(f"  Top 5 Positive:")
            for i, concept_data in enumerate(top_positive[:5]):
                concept = concept_data['concept'][:60]
                alignment = concept_data['alignment']
                print(f"    {i+1}. +{alignment:.4f} | {concept}")
            
            print(f"  Top 5 Negative:")
            for i, concept_data in enumerate(top_negative[:5]):
                concept = concept_data['concept'][:60]
                alignment = concept_data['alignment']
                print(f"    {i+1}. {alignment:.4f} | {concept}")
        
        return concept_importance
    
    def save_results(self, concept_importance: Dict, output_dir: str) -> None:
        """Save concept importance results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete results
        with open(f"{output_dir}/concept_importance_pos_neg_top100.json", 'w') as f:
            json.dump(concept_importance, f, indent=2)
        
        # Save summary CSV for easy analysis
        summary_data = []
        for label, data in concept_importance.items():
            for concept_data in data['positive_concepts']:
                summary_data.append({
                    'label': label,
                    'type': 'positive',
                    'concept': concept_data['concept'],
                    'concept_idx': concept_data['concept_idx'],
                    'alignment': concept_data['alignment'],
                    'auc': data['stats']['auc']
                })
            for concept_data in data['negative_concepts']:
                summary_data.append({
                    'label': label,
                    'type': 'negative',
                    'concept': concept_data['concept'],
                    'concept_idx': concept_data['concept_idx'],
                    'alignment': concept_data['alignment'],
                    'auc': data['stats']['auc']
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{output_dir}/concept_importance_summary.csv", index=False)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- concept_importance_pos_neg_top100.json")
        print(f"- concept_importance_summary.csv")

def main():
    results_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/results/concept_based_linear_probing_torch"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    analyzer = ConceptImportanceAnalyzer(results_dir)
    concept_importance = analyzer.analyze_concept_importance(k=100)
    
    output_dir = f"{results_dir}/concept_importance_analysis"
    analyzer.save_results(concept_importance, output_dir)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 