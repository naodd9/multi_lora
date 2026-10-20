"""
Data loading and preprocessing utilities for Multi-LoRA + MoE project.
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

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """
    Dataset class for handling multiple tasks with different label spaces.
    Each example includes task identifier to ensure correct routing to 
    task-specific classification heads.
    """
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        task_to_id: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            examples: List of dictionaries containing 'text', 'label', and 'task_id'
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
            task_to_id: Mapping from task names to integer IDs
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_to_id = task_to_id or {}
        
        # Pre-tokenize all examples for efficiency
        self._tokenize_examples()
    
    def _tokenize_examples(self):
        """Tokenize all examples and store the results."""
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.task_ids = []
        
        for example in self.examples:
            # Tokenize the text
            encoding = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            self.input_ids.append(encoding["input_ids"].squeeze())
            self.attention_masks.append(encoding["attention_mask"].squeeze())
            self.labels.append(example["label"])
            self.task_ids.append(example["task_id"])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "task_id": torch.tensor(self.task_ids[idx], dtype=torch.long)
        }


def load_and_preprocess_data(
    config_path: str = "configs/data.yaml"
) -> Tuple[Dict[str, Dataset], Dict[str, int], Dict[str, List[str]]]:
    """
    Load and preprocess AG News and SST-2 datasets.
    
    Returns:
        Tuple of (datasets_dict, task_to_id_mapping, class_names_dict)
        where datasets_dict contains 'train', 'validation', 'test' splits
        for each task combined into a single dataset per split
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Get configuration parameters
    dataset_names = data_config['dataset_names']
    text_columns = data_config['text_columns']
    label_columns = data_config['label_columns']
    num_classes = data_config['num_classes']
    class_names = data_config['class_names']
    max_length = data_config['max_sequence_length']
    use_subset = data_config['use_subset']
    subset_size = data_config['subset_size']
    cache_dir = data_config['cache_dir']
    seed = data_config['seed']
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Map task names to IDs
    task_to_id = {name: idx for idx, name in enumerate(dataset_names)}
    
    # Define valid label ranges for each dataset
    # This handles datasets where some splits might have placeholder/invalid labels
    valid_label_ranges = {
        "ag_news": lambda label: 0 <= label < 4,  # Labels 0, 1, 2, 3
        "sst2": lambda label: label in [0, 1]     # Labels 0, 1 (filter out -1 placeholders)
    }
    
    # Load datasets using specific identifiers that work with the current HF Hub setup
    datasets_dict = {}
    
    # Dataset loading mappings
    dataset_loaders = {
        "ag_news": ("fancyzhx/ag_news", None),  # (dataset_name, config_name)
        "sst2": ("nyu-mll/glue", "sst2")        # (dataset_name, config_name)
    }
    
    for dataset_name in dataset_names:
        if dataset_name not in dataset_loaders:
            raise ValueError(f"No loader configured for dataset: {dataset_name}")
        
        loader_dataset, loader_config = dataset_loaders[dataset_name]
        logger.info(f"Loading dataset: {dataset_name} as {loader_dataset}" + 
                   (f" with config {loader_config}" if loader_config else ""))
        
        # Load dataset from Hugging Face
        if loader_config is None:
            raw_dataset = load_dataset(loader_dataset, cache_dir=cache_dir)
        else:
            raw_dataset = load_dataset(loader_dataset, loader_config, cache_dir=cache_dir)
        
        # Get the text and label column names
        text_col = text_columns[dataset_name]
        label_col = label_columns[dataset_name]
        
        # Get validation function for this dataset
        is_valid_label = valid_label_ranges.get(dataset_name, lambda label: True)
        
        # Process each split
        processed_splits = {}
        for split in ["train", "validation", "test"]:
            # Handle different split names in datasets
            if split == "validation":
                # Some datasets use "validation", others use "dev"
                raw_split = raw_dataset.get("validation") or raw_dataset.get("dev")
                if raw_split is None:
                    logger.warning(f"No validation split found for {dataset_name}")
                    continue
            else:
                raw_split = raw_dataset.get(split)
                if raw_split is None:
                    logger.warning(f"No {split} split found for {dataset_name}")
                    continue
            
            # Extract examples with valid labels only
            examples = []
            invalid_count = 0
            for i, example in enumerate(raw_split):
                # Skip examples with missing text or label
                if example[text_col] is None or example[label_col] is None:
                    invalid_count += 1
                    continue
                
                # Convert label to integer (some datasets might have string labels)
                try:
                    label = int(example[label_col])
                except (ValueError, TypeError):
                    invalid_count += 1
                    continue
                
                # Validate label range
                if not is_valid_label(label):
                    invalid_count += 1
                    continue
                
                examples.append({
                    "text": str(example[text_col]),
                    "label": label,
                    "task_id": task_to_id[dataset_name]
                })
                
                # If using subset, limit the number of examples
                if use_subset and len(examples) >= subset_size:
                    break
            
            if invalid_count > 0:
                logger.info(f"Filtered {invalid_count} invalid examples from {dataset_name} {split}")
            
            logger.info(f"Loaded {len(examples)} valid examples for {dataset_name} {split}")
            processed_splits[split] = examples
        
        # Store processed splits for this dataset
        if dataset_name not in datasets_dict:
            datasets_dict[dataset_name] = {}
        
        for split, examples in processed_splits.items():
            if split not in datasets_dict[dataset_name]:
                datasets_dict[dataset_name][split] = []
            datasets_dict[dataset_name][split].extend(examples)
    
    # Combine datasets from different tasks for each split
    combined_datasets = {}
    for split in ["train", "validation", "test"]:
        combined_examples = []
        for dataset_name in dataset_names:
            if dataset_name in datasets_dict and split in datasets_dict[dataset_name]:
                combined_examples.extend(datasets_dict[dataset_name][split])
        
        # Shuffle the combined examples to mix tasks
        np.random.seed(seed)
        np.random.shuffle(combined_examples)
        
        # Create MultiTaskDataset
        combined_datasets[split] = MultiTaskDataset(
            examples=combined_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            task_to_id=task_to_id
        )
        
        logger.info(f"Created combined {split} dataset with {len(combined_examples)} examples")
    
    return combined_datasets, task_to_id, class_names


def get_dataloaders(
    datasets_dict: Dict[str, Dataset],
    batch_size: Dict[str, int] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        datasets_dict: Dictionary containing datasets for each split
        batch_size: Dictionary with batch sizes for each split
        
    Returns:
        Dictionary of DataLoaders for each split
    """
    if batch_size is None:
        batch_size = {"train": 16, "validation": 16, "test": 16}
    
    dataloaders = {}
    for split, dataset in datasets_dict.items():
        shuffle = (split == "train")  # Only shuffle training data
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size.get(split, 16),
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for simplicity; can be increased for faster loading
            pin_memory=True
        )
    
    return dataloaders


def inspect_dataset(dataset: Dataset, num_examples: int = 5):
    """
    Inspect a few examples from the dataset.
    
    Args:
        dataset: MultiTaskDataset instance
        num_examples: Number of examples to inspect
    """
    print(f"\nInspecting {num_examples} examples from dataset:")
    print("=" * 50)
    
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"Example {i+1}:")
        print(f"  Input IDs shape: {example['input_ids'].shape}")
        print(f"  Attention mask shape: {example['attention_mask'].shape}")
        print(f"  Label: {example['labels'].item()}")
        print(f"  Task ID: {example['task_id'].item()}")
        print()


if __name__ == "__main__":
    # Test the data loading functions
    datasets_dict, task_to_id, class_names = load_and_preprocess_data()
    
    print("Dataset loading completed successfully!")
    print(f"Task to ID mapping: {task_to_id}")
    print(f"Class names: {class_names}")
    
    for split, dataset in datasets_dict.items():
        print(f"{split} dataset size: {len(dataset)}")
        
    # Inspect a few examples
    if "train" in datasets_dict:
        inspect_dataset(datasets_dict["train"])