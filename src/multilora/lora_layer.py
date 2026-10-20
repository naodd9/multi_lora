"""
Shared LoRA layer implementation used by MoE-LoRA and Combined Multi-LoRA + MoE models.

This module provides the LoRALayer class which wraps a linear layer with low-rank
adaptation matrices. It is reused by both moe.py and combined_moe.py to avoid code
duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    LoRA wrapper for a linear layer.

    A frozen pretrained weight matrix is paired with trainable low-rank matrices
    A (r x d) and B (h x r). The forward pass computes:
        out = x W_frozen + (x A^T B^T) * (alpha / r)
    with A initialized via Kaiming-uniform and B initialized to zeros so that
    the adapter contribution starts at exactly zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        base_weight: torch.Tensor = None,
        base_bias: torch.Tensor = None,
    ):
        super(LoRALayer, self).__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        # Trainable LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = self.lora_alpha / self.r

            # Initialize: A from Kaiming-uniform, B to zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_buffer("lora_A", None)
            self.register_buffer("lora_B", None)
            self.scaling = 0

        # Frozen base weight. When a pretrained weight is supplied (as in the
        # MoE experts copied from DistilBERT FFN blocks), use it directly;
        # otherwise fall back to a Kaiming-uniform initialization rather than
        # N(0,1), which would produce excessively large activations.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if base_weight is not None:
            if tuple(base_weight.shape) != (out_features, in_features):
                raise ValueError(
                    f"base_weight shape {tuple(base_weight.shape)} does not match "
                    f"({out_features}, {in_features})"
                )
            with torch.no_grad():
                self.weight.copy_(base_weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # Frozen bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        if base_bias is not None:
            if tuple(base_bias.shape) != (out_features,):
                raise ValueError(
                    f"base_bias shape {tuple(base_bias.shape)} does not match ({out_features},)"
                )
            with torch.no_grad():
                self.bias.copy_(base_bias)
        else:
            bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        self.bias.requires_grad = False

    def forward(self, x):
        """Forward pass: base linear transform + LoRA adaptation."""
        out = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            x_dropout = self.lora_dropout(x)
            lora_intermediate = F.linear(x_dropout, self.lora_A, bias=None)
            lora_output = F.linear(lora_intermediate, self.lora_B, bias=None)
            out += lora_output * self.scaling

        return out
