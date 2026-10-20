"""
Visualization utilities for Multi-LoRA + MoE models.
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization utilities for Multi-LoRA + MoE models.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualizer initialized. Output directory: {output_dir}")
    
    def plot_training_curves(self, 
                           train_history: List[Dict[str, float]],
                           val_history: Optional[List[Dict[str, float]]] = None,
                           metric: str = "accuracy",
                           title: str = "Training Curves",
                           filename: Optional[str] = None):
        """
        Plot training and validation curves.
        
        Args:
            train_history: List of dictionaries containing training metrics per epoch
            val_history: List of dictionaries containing validation metrics per epoch (optional)
            metric: Metric to plot (e.g., "accuracy", "loss")
            title: Plot title
            filename: Output filename (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training metrics
        train_values = [epoch[metric] for epoch in train_history]
        epochs = range(1, len(train_values) + 1)
        plt.plot(epochs, train_values, 'b-', label=f'Training {metric}', linewidth=2)
        
        # Plot validation metrics if available
        if val_history:
            val_values = [epoch[metric] for epoch in val_history]
            plt.plot(epochs, val_values, 'r-', label=f'Validation {metric}', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_task_accuracies(self, 
                           task_metrics: Dict[str, Dict[str, Any]],
                           title: str = "Task-wise Accuracies",
                           filename: Optional[str] = None):
        """
        Plot task-wise accuracies as a bar chart.
        
        Args:
            task_metrics: Dictionary containing task metrics
            title: Plot title
            filename: Output filename (optional)
        """
        # Extract task names and accuracies
        task_names = list(task_metrics.keys())
        accuracies = [task_metrics[task]['accuracy'] for task in task_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(task_names, accuracies, color=sns.color_palette("husl", len(task_names)))
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Task', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Task accuracies saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_expert_utilization(self, 
                              expert_counts: List[int],
                              title: str = "Expert Utilization",
                              filename: Optional[str] = None):
        """
        Plot expert utilization as a bar chart.
        
        Args:
            expert_counts: List of counts for each expert
            title: Plot title
            filename: Output filename (optional)
        """
        num_experts = len(expert_counts)
        expert_labels = [f'Expert {i}' for i in range(num_experts)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(expert_labels, expert_counts, color=sns.color_palette("husl", num_experts))
        
        # Add value labels on bars
        for bar, count in zip(bars, expert_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(expert_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Expert', fontsize=12)
        plt.ylabel('Number of Tokens Assigned', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Expert utilization saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_routing_heatmap(self, 
                           routing_matrix: np.ndarray,
                           task_names: List[str],
                           expert_names: Optional[List[str]] = None,
                           title: str = "Routing Heatmap",
                           filename: Optional[str] = None):
        """
        Plot routing heatmap showing task-to-expert assignments.
        
        Args:
            routing_matrix: Matrix of shape (num_tasks, num_experts) showing routing counts
            task_names: List of task names
            expert_names: List of expert names (optional)
            title: Plot title
            filename: Output filename (optional)
        """
        if expert_names is None:
            expert_names = [f'Expert {i}' for i in range(routing_matrix.shape[1])]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(routing_matrix, 
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd',
                   xticklabels=expert_names,
                   yticklabels=task_names)
        
        plt.xlabel('Expert', fontsize=12)
        plt.ylabel('Task', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Routing heatmap saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_parameter_efficiency(self, 
                                models_info: List[Dict[str, Any]],
                                title: str = "Parameter Efficiency",
                                filename: Optional[str] = None):
        """
        Plot parameter efficiency (accuracy vs. trainable parameters) for different models.
        
        Args:
            models_info: List of dictionaries containing model information
                      Each dict should have 'name', 'trainable_params', and 'accuracy' keys
            title: Plot title
            filename: Output filename (optional)
        """
        # Extract data
        model_names = [info['name'] for info in models_info]
        trainable_params = [info['trainable_params'] for info in models_info]
        accuracies = [info['accuracy'] for info in models_info]
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(trainable_params, accuracies, 
                            s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        
        # Add model names as labels
        for i, name in enumerate(model_names):
            plt.annotate(name, 
                        (trainable_params[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Trainable Parameters', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xscale('log')  # Log scale for parameters
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Index', fontsize=10)
        
        # Save or show plot
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter efficiency plot saved to {filepath}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # This is just a placeholder to show the visualizer can be imported
    print("Visualizer class defined. Use with actual training results.")