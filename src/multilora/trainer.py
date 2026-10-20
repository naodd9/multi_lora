"""
Training pipeline for Multi-LoRA + MoE models.
"""

import os

# Explicitly unset proxy variables to avoid HF Hub connection issues
proxy_vars = [
    'ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy',
    'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy'
]
for var in proxy_vars:
    os.environ.pop(var, None)

os.environ['NO_PROXY'] = '*'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import yaml
import os
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Multi-LoRA + MoE models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration (optional)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = self._init_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize learning rate scheduler if specified
        self.scheduler = self._init_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        
    def _init_optimizer(self):
        """Initialize optimizer based on config."""
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adamw')
        learning_rate = self.config.get('training', {}).get('learning_rate', 1e-4)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.get_trainable_parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler based on config."""
        scheduler_name = self.config.get('training', {}).get('lr_scheduler', 'linear')
        warmup_steps = self.config.get('training', {}).get('warmup_steps', 500)
        
        if scheduler_name.lower() == 'linear':
            # Simple linear warmup then constant
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_lbal_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Get batch data
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            task_ids = batch["task_id"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, load_balancing_loss = self.model(input_ids, attention_mask, task_ids)
            task_loss = self.criterion(logits, labels)
            
            # Total loss includes load balancing loss
            total_loss = task_loss + load_balancing_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_task_loss += task_loss.item()
            epoch_lbal_loss += load_balancing_loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.get('training', {}).get('log_steps', 100) == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Task Loss: {task_loss.item():.4f}, "
                    f"LBal Loss: {load_balancing_loss.item():.6f}, "
                    f"Accuracy: {accuracy.item():.4f}"
                )
        
        # Return average metrics
        return {
            'loss': epoch_loss / max(num_batches, 1),
            'task_loss': epoch_task_loss / max(num_batches, 1),
            'lbal_loss': epoch_lbal_loss / max(num_batches, 1),
            'accuracy': epoch_accuracy / max(num_batches, 1)
        }
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_task_loss = 0.0
        total_lbal_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Get batch data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                task_ids = batch["task_id"].to(self.device)
                
                # Forward pass
                logits, load_balancing_loss = self.model(input_ids, attention_mask, task_ids)
                task_loss = self.criterion(logits, labels)
                
                # Total loss includes load balancing loss
                total_loss += (task_loss + load_balancing_loss).item()
                total_task_loss += task_loss.item()
                total_lbal_loss += load_balancing_loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        # Return average metrics
        return {
            'loss': total_loss / max(num_batches, 1),
            'task_loss': total_task_loss / max(num_batches, 1),
            'lbal_loss': total_lbal_loss / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1)
        }
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model has {self.model.get_num_trainable_parameters():,} trainable parameters")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics
            logger.info(
                f"Epoch {epoch} Train - "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"Task Loss: {train_metrics['task_loss']:.4f}, "
                f"LBal Loss: {train_metrics['lbal_loss']:.6f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
            )
            
            # Evaluate on validation set if available
            if self.val_dataloader:
                val_metrics = self.evaluate(self.val_dataloader)
                
                # Log validation metrics
                logger.info(
                    f"Epoch {epoch} Val - "
                    f"Loss: {val_metrics['loss']:.4f}, "
                    f"Task Loss: {val_metrics['task_loss']:.4f}, "
                    f"LBal Loss: {val_metrics['lbal_loss']:.6f}, "
                    f"Accuracy: {val_metrics['accuracy']:.4f}"
                )
                
                # Save best model based on validation accuracy
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
                    logger.info(f"New best validation accuracy: {self.best_val_accuracy:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('output', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


if __name__ == "__main__":
    # This is just a placeholder to show the trainer can be imported
    print("Trainer class defined. Use with actual model and data loaders.")