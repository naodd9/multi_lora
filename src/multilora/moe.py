"""
Mixture of Experts with LoRA adaptation.
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
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from peft import LoraConfig, get_peft_model
import logging

from multilora.lora_layer import LoRALayer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with LoRA experts.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_experts: int = 4,
                 expert_capacity: int = 32,
                 k: int = 2,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 load_balancing_loss_coeff: float = 0.01,
                 target_modules: list = None):
        """
        Initialize MoE layer with LoRA experts.
        
        Args:
            hidden_size: Hidden size of the model
            num_experts: Number of experts
            expert_capacity: Maximum number of tokens per expert
            k: Number of experts to select per token (top-k)
            lora_rank: Rank of LoRA adaptation
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            load_balancing_loss_coeff: Coefficient for load balancing loss
            target_modules: List of module names to apply LoRA to in each expert
        """
        super(MoELayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.k = k
        self.load_balancing_loss_coeff = load_balancing_loss_coeff
        
        # Create experts (each expert is a LoRA-wrapped FFN)
        # Each expert consists of two linear layers with LoRA (simulating FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                # First linear layer: hidden_size -> hidden_size * 4
                LoRALayer(hidden_size, hidden_size * 4, 
                         r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                nn.ReLU(),
                # Second linear layer: hidden_size * 4 -> hidden_size
                LoRALayer(hidden_size * 4, hidden_size,
                         r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network (router)
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Noise for training (helps with load balancing)
        self.noise_std = 0.1
        
    def cv_squared(self, x):
        """Compute the coefficient of variation squared of x."""
        eps = 1e-10
        return torch.var(x, dim=0) / (torch.mean(x, dim=0)**2 + eps)

    def init_experts_from_encoder(self, encoder):
        """Initialize each expert's frozen FFN weights from DistilBERT.

        Each expert is a LoRA-adapted copy of a pretrained FFN block.  Without
        this initialization the base weights would be random and frozen, so the
        only trainable signal would be a rank-r correction around a random
        projection.  Experts are assigned round-robin to the encoder's six
        transformer layers so that they start from distinct pretrained FFNs.
        """
        layers = encoder.transformer.layer
        n_layers = len(layers)
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                src = layers[i % n_layers].ffn
                expert[0].weight.copy_(src.lin1.weight)
                expert[0].bias.copy_(src.lin1.bias)
                expert[2].weight.copy_(src.lin2.weight)
                expert[2].bias.copy_(src.lin2.bias)
    
    def forward(self, x):
        """
        Forward pass of MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            output: Output tensor of same shape as input
            load_balancing_loss: Scalar loss for load balancing
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        num_tokens = x_flat.shape[0]
        
        # Compute router logits and probabilities
        router_logits = self.gate(x_flat)  # (num_tokens, num_experts)
        
        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits += noise
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (num_tokens, num_experts)
        
        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.k, dim=-1)  # (num_tokens, k)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        # Measures how evenly tokens are distributed across experts
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.k):
            expert_counts += torch.bincount(top_k_indices[:, i].flatten(), 
                                          minlength=self.num_experts)
        
        # Calculate fraction of tokens per expert
        expert_fraction = expert_counts / num_tokens
        
        # Calculate average probability per expert
        expert_prob_mean = router_probs.mean(dim=0)
        
        # Load balancing loss encourages uniform expert utilization
        load_balancing_loss = self.cv_squared(expert_fraction) * self.cv_squared(expert_prob_mean)
        load_balancing_loss *= self.load_balancing_loss_coeff
        
        # Compute final output via scatter-add instead of nested Python loops
        output = torch.zeros_like(x_flat)

        # Vectorized scatter-add: process each token slot independently
        for k_idx in range(self.k):
            expert_indices = top_k_indices[:, k_idx]    # (num_tokens,)
            expert_weights = top_k_probs[:, k_idx]      # (num_tokens,)

            for expert_id in range(self.num_experts):
                # Gather tokens routed to this expert at this k-slot
                mask = expert_indices == expert_id          # (num_tokens,)
                if not mask.any():
                    continue
                inputs = x_flat[mask]                       # (n_tokens_x_H,)
                out = self.experts[expert_id](inputs)       # (n_tokens_x_H,)
                w = expert_weights[mask].unsqueeze(-1)     # (n_tokens_x_1)
                output[mask] += out * w
        
        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, load_balancing_loss


class MoLoRAModel(nn.Module):
    """
    MoE-LoRA model with DistilBERT encoder and task-specific classification heads.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes_per_task: dict = None,
                 num_experts: int = 4,
                 expert_capacity: int = 32,
                 k: int = 2,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 load_balancing_loss_coeff: float = 0.01,
                 dropout_rate: float = 0.1):
        """
        Initialize the MoE-LoRA model.
        
        Args:
            model_name: Name of the pretrained DistilBERT model
            num_classes_per_task: Dictionary mapping task names to number of classes
            num_experts: Number of experts in MoE layer
            expert_capacity: Maximum number of tokens per expert
            k: Number of experts to select per token (top-k)
            lora_rank: Rank of LoRA adaptation
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            load_balancing_loss_coeff: Coefficient for load balancing loss
            dropout_rate: Dropout rate for classification heads
        """
        super(MoLoRAModel, self).__init__()
        
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
        base_model = DistilBertModel.from_pretrained(model_name)
        self.config = DistilBertConfig.from_pretrained(model_name)
        
        # Get the hidden size from the model configuration
        self.hidden_size = self.config.dim
        
        # We'll replace the transformer blocks with MoE layers
        # For simplicity, we'll just add an MoE layer after the embeddings
        # In a full implementation, we'd replace specific transformer blocks
        self.distilbert = base_model
        
        # Add MoE layer
        self.moe_layer = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            k=k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            load_balancing_loss_coeff=load_balancing_loss_coeff
        )
        self.moe_layer.init_experts_from_encoder(base_model)
        
        # Freeze the base model parameters
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Create task-specific classification heads
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
        logger.info(f"MoE configuration: {num_experts} experts, capacity={expert_capacity}, k={k}")
        for task_name, num_classes in num_classes_per_task.items():
            logger.info(f"  {task_name}: {num_classes} classes (outputting {self.max_num_classes} logits)")
        
        # Log trainable parameter info
        trainable_params = self.get_num_trainable_parameters()
        total_params = self.get_num_total_parameters()
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
    
    def forward(self, input_ids, attention_mask, task_ids):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            task_ids: Task IDs of shape (batch_size,) indicating which task each example belongs to
            
        Returns:
            logits: Tensor of shape (batch_size, max_num_classes) with masked logits for invalid classes
            load_balancing_loss: Scalar loss for load balancing (0 if not training)
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the sequence output (before pooling)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Apply MoE layer
        moe_output, load_balancing_loss = self.moe_layer(sequence_output)
        
        # Use mean pooling over sequence (alternatively could use [CLS] token)
        # For simplicity, we'll use mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(moe_output.size()).float()
        sum_embeddings = torch.sum(moe_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask  # (batch_size, hidden_size)
        
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
            raw_logits = head(pooled_output[i:i+1]).squeeze(0)  # (max_num_classes,)
            masked_logits = raw_logits * mask
            logits[i] = masked_logits
        
        return logits, load_balancing_loss
    
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
    model = MoLoRAModel()
    
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
    model.eval()
    with torch.no_grad():
        logits, lbal_loss = model(input_ids, attention_mask, task_ids)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  task_ids: {task_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Load balancing loss: {lbal_loss.item():.6f}")
    print(f"Logits:\n{logits}")
    
    # Test that invalid classes are masked to zero
    print(f"\nTesting class masking:")
    print(f"For AG News examples (task_id=0), classes 0-3 should be non-zero, class 4+ should be zero")
    print(f"For SST-2 examples (task_id=1), class 0 should be non-zero, class 1+ should be zero")
    
    # Check first example (AG News, task_id=0)
    ag_news_logits = logits[0]  # First example
    print(f"AG News logits: {ag_news_logits}")
    print(f"  Classes 0-3: {ag_news_logits[:4]} (should be non-zero)")
    if model.max_num_classes > 4:
        print(f"  Classes 4+: {ag_news_logits[4:]} (should be all zeros)")
    
    # Check second example (SST-2, task_id=1)
    sst2_logits = logits[1]  # Second example
    print(f"SST-2 logits: {sst2_logits}")
    print(f"  Class 0: {sst2_logits[0]} (should be non-zero)")
    if model.max_num_classes > 1:
        print(f"  Class 1+: {sst2_logits[1:]} (should be all zeros)")