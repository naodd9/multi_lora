"""
Multi-LoRA + MoE for multi-task text classification.

Public API:
    BaselineModel              -- frozen DistilBERT encoder with task heads
    LoRAModel                  -- standard PEFT LoRA on the DistilBERT encoder
    MoLoRAModel                -- Mixture-of-Experts with LoRA-adapted experts
    CombinedMultiLoRAMoEModel  -- task-specific LoRA adapters + MoE layer
    LoRALayer                  -- shared low-rank linear wrapper
    Trainer / Evaluator        -- generic training and evaluation pipelines
    load_and_preprocess_data   -- AG News + SST-2 dataset pipeline
"""

__version__ = "1.0.0"

__all__ = [
    "BaselineModel",
    "LoRAModel",
    "MoLoRAModel",
    "CombinedMultiLoRAMoEModel",
    "LoRALayer",
    "Trainer",
    "Evaluator",
    "load_and_preprocess_data",
    "get_dataloaders",
]
