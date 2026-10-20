"""
Evaluation pipeline for Multi-LoRA + MoE models.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for Multi-LoRA + MoE models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use for evaluation (optional)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self, 
                 dataloader: DataLoader,
                 compute_per_task_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            compute_per_task_metrics: Whether to compute metrics per task
            
        Returns:
            Dictionary containing evaluation results
        """
        # Initialize metrics
        all_predictions = []
        all_labels = []
        all_task_ids = []
        total_loss = 0.0
        num_batches = 0
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for batch in dataloader:
                # Get batch data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                task_ids = batch["task_id"].to(self.device)
                
                # Forward pass
                logits, load_balancing_loss = self.model(input_ids, attention_mask, task_ids)
                task_loss = criterion(logits, labels)
                
                # Total loss includes load balancing loss
                batch_loss = task_loss + load_balancing_loss
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_task_ids.append(task_ids.cpu())
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_task_ids = torch.cat(all_task_ids, dim=0)
        
        # Calculate overall metrics
        overall_accuracy = (all_predictions == all_labels).float().mean().item()
        overall_loss = total_loss / max(num_batches, 1)
        
        # Prepare results
        results = {
            'loss': overall_loss,
            'accuracy': overall_accuracy,
            'predictions': all_predictions.numpy(),
            'labels': all_labels.numpy(),
            'task_ids': all_task_ids.numpy()
        }
        
        # Calculate per-task metrics if requested
        if compute_per_task_metrics:
            task_metrics = {}
            unique_task_ids = torch.unique(all_task_ids)
            
            for task_id in unique_task_ids:
                task_id_int = task_id.item()
                # Get indices for this task
                task_mask = (all_task_ids == task_id)
                
                if task_mask.any():
                    task_predictions = all_predictions[task_mask]
                    task_labels = all_labels[task_mask]
                    
                    # Calculate task-specific accuracy
                    task_accuracy = (task_predictions == task_labels).float().mean().item()
                    
                    # Get task name if available
                    task_name = f"task_{task_id_int}"
                    if hasattr(self.model, 'task_names') and task_id_int < len(self.model.task_names):
                        task_name = self.model.task_names[task_id_int]
                    
                    task_metrics[task_name] = {
                        'accuracy': task_accuracy,
                        'num_samples': task_mask.sum().item()
                    }
            
            results['task_metrics'] = task_metrics
        
        return results
    
    def evaluate_routing(self, 
                        dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate routing patterns in MoE models.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary containing routing statistics
        """
        # This is a placeholder for routing evaluation
        # In a full implementation, we would hook into the model to collect
        # expert assignment statistics
        
        logger.info("Routing evaluation not fully implemented in this placeholder")
        return {
            'routing_evaluation': 'placeholder',
            'message': 'Full routing evaluation would require model hooks to collect expert assignments'
        }


if __name__ == "__main__":
    # This is just a placeholder to show the evaluator can be imported
    print("Evaluator class defined. Use with actual model and data loaders.")