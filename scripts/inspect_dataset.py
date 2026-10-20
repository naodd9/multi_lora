#!/usr/bin/env python3
"""
Script to inspect the loaded datasets and verify correctness.
"""

import os
# Explicitly unset proxy variables to avoid HF Hub connection issues
proxy_vars = [
    'ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 
    'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy'
]
for var in proxy_vars:
    os.environ.pop(var, None)  # Remove if exists
    os.environ[var] = ''  # Set to empty string

# Also set NO_PROXY to everything to bypass proxies
os.environ['NO_PROXY'] = '*'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multilora.data import load_and_preprocess_data, inspect_dataset
import yaml


def main():
    """Main function to inspect datasets."""
    print("Loading and inspecting datasets...")
    print("=" * 50)
    
    # Load the datasets
    datasets_dict, task_to_id, class_names = load_and_preprocess_data()
    
    print("Dataset loading completed successfully!")
    print(f"Task to ID mapping: {task_to_id}")
    print(f"Class names: {class_names}")
    
    print("\nDataset sizes:")
    for split, dataset in datasets_dict.items():
        print(f"  {split}: {len(dataset)} examples")
    
    # Inspect a few examples from each split
    for split in ["train", "validation", "test"]:
        if split in datasets_dict:
            print(f"\nInspecting {split} split:")
            print("-" * 30)
            inspect_dataset(datasets_dict[split], num_examples=3)
    
    # Verify that we can create DataLoaders
    from multilora.data import get_dataloaders
    
    dataloaders = get_dataloaders(datasets_dict)
    
    print("DataLoaders created successfully!")
    for split, loader in dataloaders.items():
        print(f"  {split} loader: {len(loader)} batches")
        
    # Inspect a batch from the training loader
    if "train" in dataloaders:
        print("\nInspecting a batch from training loader:")
        print("-" * 40)
        batch = next(iter(dataloaders["train"]))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Task IDs shape: {batch['task_id'].shape}")
        
        # Check a few examples from the batch
        print(f"\nFirst example in batch:")
        print(f"  Label: {batch['labels'][0].item()}")
        print(f"  Task ID: {batch['task_id'][0].item()}")


if __name__ == "__main__":
    main()