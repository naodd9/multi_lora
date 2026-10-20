"""
Baseline model: Frozen DistilBERT encoder with task-specific classification heads.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    """
    Baseline model with frozen DistilBERT encoder and task-specific classification heads.
    
    Architecture:
    - Shared DistilBERT encoder (frozen)
    - Task-specific classification heads for each task
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes_per_task: dict = None,
                 dropout_rate: float = 0.1):
        """
        Initialize the baseline model.
        
        Args:
            model_name: Name of the pretrained DistilBERT model
            num_classes_per_task: Dictionary mapping task names to number of classes
            dropout_rate: Dropout rate for classification heads
        """
        super(BaselineModel, self).__init__()
        
        if num_classes_per_task is None:
            # Default to our two tasks
            num_classes_per_task = {
                "ag_news": 4,  # World, Sports, Business, Sci/Tech
                "sst2": 2      # negative, positive
            }
        
        self.num_classes_per_task = num_classes_per_task
        self.task_names = list(num_classes_per_task.keys())
        self.max_num_classes = max(num_classes_per_task.values()) if num_classes_per_task else 0
        
        # Load pretrained DistilBERT model
        logger.info(f"Loading pretrained model: {model_name}")
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.config = DistilBertConfig.from_pretrained(model_name)
        
        # Freeze the encoder parameters
        logger.info("Freezing DistilBERT encoder parameters")
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Get the hidden size from the model configuration
        self.hidden_size = self.config.dim
        
        # Create task-specific classification heads
        # Each head outputs max_num_classes logits, but we'll mask unused ones later
        self.classification_heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_per_task.items():
            self.classification_heads[task_name] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size, self.max_num_classes)
            )
        
        # Create a mask to zero out invalid class outputs for each task
        self.class_masks = {}
        for task_name, num_classes in num_classes_per_task.items():
            mask = torch.zeros(self.max_num_classes)
            mask[:num_classes] = 1.0  # Only keep the first 'num_classes' positions
            self.class_masks[task_name] = mask
        
        logger.info(f"Model initialized with {len(self.task_names)} tasks: {self.task_names}")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Max number of classes: {self.max_num_classes}")
        for task_name, num_classes in num_classes_per_task.items():
            logger.info(f"  {task_name}: {num_classes} classes (outputting {self.max_num_classes} logits)")
    
    def forward(self, input_ids, attention_mask, task_ids):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            task_ids: Task IDs of shape (batch_size,) indicating which task each example belongs to
            
        Returns:
            logits: Tensor of shape (batch_size, max_num_classes) with masked logits for invalid classes
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        cls_output = sequence_output[:, 0, :]        # (batch_size, hidden_size)
        
        # Apply task-specific classification heads
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, self.max_num_classes, 
                            device=input_ids.device)
        
        # For each example in the batch, apply the appropriate task's classification head
        for i in range(batch_size):
            task_id = task_ids[i].item()
            task_name = self.task_names[task_id]
            head = self.classification_heads[task_name]
            mask = self.class_masks[task_name].to(input_ids.device)
            
            # Get raw logits from the head and apply the mask
            raw_logits = head(cls_output[i:i+1]).squeeze(0)  # (max_num_classes,)
            masked_logits = raw_logits * mask
            logits[i] = masked_logits
        
        return logits
    
    def get_trainable_parameters(self):
        """
        Get the trainable parameters of the model.
        
        Returns:
            List of trainable parameter tensors
        """
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def get_num_total_parameters(self):
        """
        Get the total number of parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test the model
    model = BaselineModel()
    
    print("Model architecture:")
    print(model)
    print(f"\nNumber of trainable parameters: {model.get_num_trainable_parameters():,}")
    print(f"Number of total parameters: {model.get_num_total_parameters():,}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 128
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    # Task IDs: 0 for ag_news, 1 for sst2
    task_ids = torch.tensor([0, 1, 0, 1])
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask, task_ids)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  task_ids: {task_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")
    
    # Test that invalid classes are masked to zero
    print(f"\nTesting class masking:")
    print(f"For AG News examples (task_id=0), classes 2-3 should be non-zero, class 4+ should be zero")
    print(f"For SST-2 examples (task_id=1), classes 1+ should be zero")
    
    # Check first example (AG News, task_id=0)
    ag_news_logits = logits[0]  # First example
    print(f"AG News logits: {ag_news_logits}")
    print(f"  Classes 0-3: {ag_news_logits[:4]} (should be non-zero)")
    print(f"  Classes 4+: {ag_news_logits[4:]} (should be all zeros)")
    
    # Check second example (SST-2, task_id=1)
    sst2_logits = logits[1]  # Second example
    print(f"SST-2 logits: {sst2_logits}")
    print(f"  Class 0: {sst2_logits[0]} (should be non-zero)")
    print(f"  Class 1+: {sst2_logits[1:]} (should be all zeros)")