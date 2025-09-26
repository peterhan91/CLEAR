#!/usr/bin/env python3
"""
MONet Auditing: Multi-Disease Analysis with Cross-Dataset Concept Attribution
Analyzes cardiomegaly, pleural effusion, and pneumonia between VinDR-CXR and Indiana datasets.
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_auditing.monet_auditing import MONETAuditor

# Disease name mappings for different datasets
DISEASE_MAPPINGS = {
    "vindrcxr": {
        "cardiomegaly": "Cardiomegaly",
        "pleural_effusion": "Pleural effusion", 
        "pneumonia": "Pneumonia"
    },
    "indiana_test": {
        "cardiomegaly": "cardiomegaly",
        "pleural_effusion": "pleural effusion",
        "pneumonia": "pneumonia"
    }
}

def print_concept_analysis(dataset_name: str, disease: str, concept_attributions: dict):
    """Print detailed concept analysis for a dataset and disease"""
    if not concept_attributions:
        print(f"❌ No concept attributions found for {dataset_name} - {disease}")
        return
    
    print(f"\n🔬 {dataset_name.upper()} - {disease.upper()} Concept Analysis:")
    print("=" * 60)
    
    for cluster_key, cluster_data in concept_attributions.items():
        low_cluster = cluster_data['low_cluster_id']
        high_cluster = cluster_data['high_cluster_id']
        low_auroc = cluster_data['low_auroc']
        high_auroc = cluster_data['high_auroc']
        
        print(f"\n🎯 Cluster {low_cluster} (AUROC: {low_auroc:.4f}) vs Reference Cluster {high_cluster} (AUROC: {high_auroc:.4f})")
        
        concept_differences = cluster_data['concept_differences']
        if 'status' in concept_differences:
            print(f"   ⚠️ Status: {concept_differences['status']}")
            continue
        
        if 'top_concepts_low_performing' in concept_differences:
            top_concepts = concept_differences['top_concepts_low_performing'][:3]  # Show top 3
            if top_concepts:
                print(f"   📊 Top distinguishing concepts:")
                for i, concept in enumerate(top_concepts, 1):
                    concept_text = concept.get('concept_text', f"Concept {concept['concept_index']}")
                    if len(concept_text) > 70:
                        concept_text = concept_text[:70] + "..."
                    diff_score = concept['difference_score']
                    low_score = concept['low_cluster_score']
                    high_score = concept['high_cluster_score']
                    print(f"   {i}. {concept_text}")
                    print(f"      Diff: {diff_score:.4f} | Low: {low_score:.4f} | High: {high_score:.4f}")
            else:
                print(f"   ❌ No significant concept differences found")

def save_disease_results(comparison: dict, disease: str, timestamp: str):
    """Save detailed results for a specific disease"""
    results_dir = "model_auditing/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save main comparison results for this disease
    main_results_path = f"{results_dir}/{disease}_vindrcxr_indiana_comparison_{timestamp}.json"
    with open(main_results_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"💾 {disease.capitalize()} comparison saved to: {main_results_path}")
    

def run_disease_analysis(auditor: MONETAuditor, disease: str, timestamp: str, 
                        use_same_dataset_comparison: bool = False):
    """Run MONet analysis for a specific disease"""
    print(f"\n" + "=" * 80)
    print(f"🏥 ANALYZING {disease.upper()}")
    print("=" * 80)
    
    try:
        # Get correct disease names for each dataset
        vindrcxr_disease = DISEASE_MAPPINGS["vindrcxr"][disease]
        indiana_disease = DISEASE_MAPPINGS["indiana_test"][disease]
        
        print(f"📊 VinDR disease name: '{vindrcxr_disease}'")
        print(f"📊 Indiana disease name: '{indiana_disease}'")
        
        # Run individual analyses 
        comparison_type = "same-dataset" if use_same_dataset_comparison else "cross-dataset"
        print(f"🔬 Using {comparison_type} cluster comparison")
        
        print(f"\n📊 Running VinDR analysis for {disease}...")
        vindrcxr_results = auditor.run_monet_auditing("vindrcxr", vindrcxr_disease, "sfr_mistral", use_same_dataset_comparison)
        
        print(f"📊 Running Indiana analysis for {disease}...")
        indiana_results = auditor.run_monet_auditing("indiana_test", indiana_disease, "sfr_mistral", use_same_dataset_comparison)
        
        # Perform cluster analysis based on comparison type
        if not use_same_dataset_comparison:
            print("🔄 Performing cross-dataset cluster analysis...")
            auditor._update_concept_attributions_cross_dataset(
                vindrcxr_results, indiana_results, "vindrcxr", "indiana_test", disease, "sfr_mistral"
            )
        else:
            print("✅ Same-dataset cluster analysis completed for both datasets")
        
        # Create comparison structure
        comparison = {
            'dataset1': 'vindrcxr',
            'dataset2': 'indiana_test', 
            'disease_name': disease,
            'vindrcxr_disease_name': vindrcxr_disease,
            'indiana_disease_name': indiana_disease,
            'model': 'sfr_mistral',
            'comparison_type': comparison_type,
            'disease_auroc1': vindrcxr_results['disease_auroc'],
            'disease_auroc2': indiana_results['disease_auroc'],
            'disease_auroc_difference': vindrcxr_results['disease_auroc'] - indiana_results['disease_auroc'],
            'results1': vindrcxr_results,
            'results2': indiana_results
        }
        
        # Print results summary
        print(f"\n✅ {disease.upper()} ANALYSIS RESULTS")
        print("-" * 50)
        print(f"📊 VinDR-CXR AUROC: {comparison['disease_auroc1']:.4f}")
        print(f"📊 Indiana AUROC: {comparison['disease_auroc2']:.4f}")
        print(f"📈 Performance gap: {abs(comparison['disease_auroc_difference']):.4f}")
        
        better_dataset = "VinDR-CXR" if comparison['disease_auroc1'] > comparison['disease_auroc2'] else "Indiana"
        print(f"🏆 Better performing: {better_dataset}")
        
        # Cluster analysis summary
        vindrcxr_concepts = vindrcxr_results.get('concept_attributions', {})
        indiana_concepts = indiana_results.get('concept_attributions', {})
        
        print(f"🎯 VinDR-CXR: {len(vindrcxr_results['low_performing_clusters'])} low-performing clusters")
        print(f"🎯 Indiana: {len(indiana_results['low_performing_clusters'])} low-performing clusters")
        print(f"🔬 VinDR-CXR concept attributions: {len(vindrcxr_concepts)} clusters analyzed")
        print(f"🔬 Indiana concept attributions: {len(indiana_concepts)} clusters analyzed")
        
        # Detailed concept analysis (show just first cluster for brevity)
        if vindrcxr_concepts:
            first_cluster = list(vindrcxr_concepts.keys())[0]
            cluster_data = vindrcxr_concepts[first_cluster]
            concept_differences = cluster_data['concept_differences']
            if 'top_concepts_low_performing' in concept_differences:
                top_concepts = concept_differences['top_concepts_low_performing'][:2]
                if top_concepts:
                    print(f"🔬 Sample VinDR concepts (Cluster {cluster_data['low_cluster_id']}):")
                    for concept in top_concepts:
                        concept_text = concept.get('concept_text', f"Concept {concept['concept_index']}")[:50]
                        print(f"   • {concept_text}... (diff: {concept['difference_score']:.3f})")
        
        # Save results for this disease
        save_disease_results(comparison, disease, timestamp)
        
        # Save worst performing cluster images for both datasets
        print(f"\n📸 Saving worst cluster images for {disease}...")
        try:
            # Save images for VinDR-CXR dataset
            print(f"🔍 Processing VinDR-CXR worst clusters...")
            vindrcxr_image_stats = auditor.save_worst_cluster_images(
                dataset="vindrcxr", 
                disease_name=vindrcxr_disease,
                model="sfr_mistral",
                top_k=5,
                max_images_per_cluster=15
            )
            
            # Save images for Indiana dataset  
            print(f"🔍 Processing Indiana worst clusters...")
            indiana_image_stats = auditor.save_worst_cluster_images(
                dataset="indiana_test",
                disease_name=indiana_disease, 
                model="sfr_mistral",
                top_k=5,
                max_images_per_cluster=30
            )
            
            print(f"✅ {disease} cluster images saved:")
            print(f"   📁 VinDR-CXR: {vindrcxr_image_stats['total_images_saved']} images")
            print(f"   📁 Indiana: {indiana_image_stats['total_images_saved']} images")
            
            # Add image stats to comparison
            comparison['image_stats'] = {
                'vindrcxr': vindrcxr_image_stats,
                'indiana': indiana_image_stats
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save worst cluster images for {disease}: {e}")
        
        return comparison
        
    except Exception as e:
        print(f"❌ Error analyzing {disease}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MONet Auditing: VinDR-CXR vs Indiana University Analysis")
    parser.add_argument("--same-dataset", action="store_true", 
                       help="Compare low vs high clusters within same dataset (default: cross-dataset)")
    args = parser.parse_args()
    
    comparison_type = "same-dataset" if args.same_dataset else "cross-dataset"
    
    print("🏥 MONet Auditing: Multi-Disease Cross-Dataset Analysis")
    print("📊 Datasets: VinDR-CXR vs Indiana")
    print("🦠 Diseases: Cardiomegaly, Pleural Effusion, Pneumonia")
    print(f"🔬 Comparison type: {comparison_type}")
    print("=" * 80)
    
    # Initialize auditor
    auditor = MONETAuditor("concepts/results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Diseases to analyze
    diseases = ["cardiomegaly", "pleural_effusion", "pneumonia"]
    all_results = {}
    
    # Run analysis for each disease
    for disease in diseases:
        result = run_disease_analysis(auditor, disease, timestamp, args.same_dataset)
        if result:
            all_results[disease] = result
    
    # Create summary report
    if all_results:
        print(f"\n" + "=" * 80)
        print("📋 MULTI-DISEASE ANALYSIS SUMMARY")
        print("=" * 80)
        
        summary_data = {
            "timestamp": timestamp,
            "total_diseases_analyzed": len(all_results),
            "diseases": {}
        }
        
        for disease, result in all_results.items():
            vindrcxr_auroc = result['disease_auroc1']
            indiana_auroc = result['disease_auroc2']
            gap = abs(result['disease_auroc_difference'])
            better = "VinDR-CXR" if vindrcxr_auroc > indiana_auroc else "Indiana"
            
            print(f"\n🦠 {disease.upper()}:")
            print(f"   VinDR: {vindrcxr_auroc:.4f} | Indiana: {indiana_auroc:.4f} | Gap: {gap:.4f} | Better: {better}")
            
            summary_data["diseases"][disease] = {
                "vindrcxr_auroc": vindrcxr_auroc,
                "indiana_auroc": indiana_auroc,
                "performance_gap": gap,
                "better_dataset": better,
                "vindrcxr_low_clusters": len(result['results1']['low_performing_clusters']),
                "indiana_low_clusters": len(result['results2']['low_performing_clusters'])
            }
        
        # Save overall summary
        summary_path = f"model_auditing/results/multi_disease_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n💾 Multi-disease summary saved to: {summary_path}")
        
        print(f"\n✅ All analyses completed! Results saved with timestamp: {timestamp}")
        return 0
    else:
        print("❌ No analyses completed successfully")
        return 1

if __name__ == "__main__":
    exit(main()) 