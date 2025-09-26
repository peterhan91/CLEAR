#!/usr/bin/env python3
"""
Create a 5x10 grid visualization of retrieved images for multiple medical conditions.

This script reads the top-10 retrieved images for each condition and creates
a grid plot similar to the style shown in the reference image.
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse

def find_latest_retrieval_dirs(results_dir, conditions, model_name="chexzero"):
    """Find the latest retrieval directories for each condition"""
    retrieval_dirs = {}
    
    for condition in conditions:
        # Look for directories matching the pattern
        pattern = f"retrieved_images_{condition.replace(' ', '_')}_{model_name}_*"
        dirs = glob.glob(os.path.join(results_dir, pattern))
        
        if not dirs:
            raise FileNotFoundError(f"No retrieval directory found for condition: {condition}")
        
        # Get the most recent one (based on timestamp in name)
        latest_dir = max(dirs, key=lambda x: os.path.getctime(x))
        retrieval_dirs[condition] = latest_dir
        print(f"Found {condition}: {os.path.basename(latest_dir)}")
    
    return retrieval_dirs

def load_images_from_directory(retrieval_dir, top_k=10):
    """Load the top-k images from a retrieval directory"""
    # Load the summary to get the ranked results
    summary_file = os.path.join(retrieval_dir, "retrieval_summary.json")
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    images = []
    filenames = []
    
    # Get the top-k results and load corresponding images
    for i, result in enumerate(summary['top_results'][:top_k]):
        # Try to find the actual file (they all seem to end with _rank_001.png)
        expected_filename = result['filename']
        
        # Try the expected filename first
        image_path = os.path.join(retrieval_dir, expected_filename)
        
        if not os.path.exists(image_path):
            # If expected file doesn't exist, try with _rank_001.png pattern
            # Extract the base name and replace the rank suffix
            base_name = expected_filename.split('_rank_')[0]
            actual_filename = base_name + "_rank_001.png"
            image_path = os.path.join(retrieval_dir, actual_filename)
            
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                images.append(img)
                filenames.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Warning: Could not load image {os.path.basename(image_path)}: {e}")
                # Create a placeholder black image
                img = Image.new('RGB', (224, 224), color='black')
                images.append(img)
                filenames.append(f"placeholder_{i}")
        else:
            # Try to find any image file with the rank pattern
            rank_pattern = f"rank_{i+1:03d}_*_rank_001.png"
            matching_files = glob.glob(os.path.join(retrieval_dir, rank_pattern))
            
            if matching_files:
                try:
                    img = Image.open(matching_files[0]).convert('RGB')
                    images.append(img)
                    filenames.append(os.path.basename(matching_files[0]))
                except Exception as e:
                    print(f"Warning: Could not load image {os.path.basename(matching_files[0])}: {e}")
                    # Create a placeholder black image
                    img = Image.new('RGB', (224, 224), color='black')
                    images.append(img)
                    filenames.append(f"placeholder_{i}")
            else:
                print(f"Warning: No image found for rank {i+1}")
                # Create a placeholder black image
                img = Image.new('RGB', (224, 224), color='black')
                images.append(img)
                filenames.append(f"placeholder_{i}")
    
    return images, filenames

def create_condition_grid(conditions, retrieval_dirs, output_path, top_k=10):
    """Create a 5x10 grid of retrieved images for the conditions"""
    
    print(f"Creating {len(conditions)}x{top_k} grid visualization...")
    
    # Set up the figure with proper spacing
    fig, axes = plt.subplots(len(conditions), top_k, figsize=(20, 4*len(conditions)))
    fig.suptitle('Top-10 Retrieved Chest X-rays for Medical Conditions', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # Process each condition (row)
    for row_idx, condition in enumerate(conditions):
        retrieval_dir = retrieval_dirs[condition]
        
        # Load images for this condition
        images, filenames = load_images_from_directory(retrieval_dir, top_k)
        
        # Add condition label on the left
        if row_idx < len(axes):
            # Format condition name (capitalize first letter)
            condition_label = condition.replace('_', ' ').title()
            
            # Add text annotation for the condition name
            fig.text(0.03, 0.84 - row_idx * 0.165, condition_label, 
                    fontsize=14, fontweight='bold', 
                    verticalalignment='center', rotation=90)
        
        # Display images in this row
        for col_idx in range(top_k):
            if row_idx < len(axes) and col_idx < len(axes[row_idx]):
                ax = axes[row_idx, col_idx] if len(conditions) > 1 else axes[col_idx]
                
                if col_idx < len(images):
                    # Display the image
                    ax.imshow(images[col_idx], cmap='gray' if len(np.array(images[col_idx]).shape) == 2 else None)
                else:
                    # Create placeholder for missing images
                    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
                    ax.imshow(placeholder)
                
                # Remove axes and ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                
                # Add black frame around each image
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(2)
    
    # First do tight layout with specific margins
    plt.tight_layout(rect=[0.08, 0.03, 0.97, 0.94])
    
    # Then adjust spacing between subplots for uniform white space
    # wspace: width spacing between columns, hspace: height spacing between rows
    plt.subplots_adjust(left=0.08, bottom=0.03, right=0.97, top=0.94, 
                       wspace=0.08, hspace=0.08)
    
    # Save the plot with consistent padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    
    print(f"Grid visualization saved to: {output_path}")
    print(f"PDF version saved to: {output_path.replace('.png', '.pdf')}")
    
    # Show the plot
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create grid visualization of retrieved medical images")
    parser.add_argument('--results_dir', type=str, 
                        default='model_auditing/results',
                        help='Directory containing retrieval results')
    parser.add_argument('--conditions', nargs='+',
                        default=['pneumothorax', 'pleural effusion', 'consolidation', 'edema', 'mediastinal shift'],
                        help='Medical conditions to visualize')
    parser.add_argument('--model', type=str, default='chexzero',
                        help='Model name used for retrieval')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top images per condition')
    parser.add_argument('--output', type=str,
                        help='Output path for the grid image (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Find the latest retrieval directories
    try:
        retrieval_dirs = find_latest_retrieval_dirs(args.results_dir, args.conditions, args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conditions_str = "_".join([c.replace(' ', '_') for c in args.conditions])
        args.output = os.path.join(args.results_dir, 
                                   f"retrieval_grid_{conditions_str}_{args.model}_{timestamp}.png")
    
    # Create the grid visualization
    try:
        create_condition_grid(args.conditions, retrieval_dirs, args.output, args.top_k)
        print(f"\n✅ Grid visualization created successfully!")
        return 0
    except Exception as e:
        print(f"Error creating grid: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 