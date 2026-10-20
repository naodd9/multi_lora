# Multi-LoRA + MoE for Multi-Task Text Classification

Parameter-efficient adaptation of a **frozen DistilBERT** encoder for multi-task text
classification, implemented as four progressively richer variants:

| Variant | Adaptation mechanism | Trainable params |
|---|---|---|
| **Baseline** | frozen encoder + per-task linear heads | 6,152 (0.009%) |
| **LoRA** | PEFT LoRA (r=8) on `q_lin, v_lin, k_lin, out_lin, lin1, lin2` | 669,704 (0.999%) |
| **MoE-LoRA** | top-k router over LoRA-adapted FFN experts, post-encoder | 254,988 (0.298%) |
| **Combined** | task-specific LoRA adapters + trainable MoE layer | 19,144,716 (18.34%) |

Grounded in LoRA (Hu et al., 2021), Mixture-of-Experts (Shazeer et al., 2017),
MultiLoRA (Wang et al., 2023), MeteoRA (Xu et al., 2024) and MALoRA (Wang et al., 2025).

---

## Repository layout

```
github/
├── src/multilora/          # Library package (installable)
│   ├── data.py             # AG News + SST-2 loading, tokenization, dataloaders
│   ├── baseline_model.py   # Frozen DistilBERT + task-specific heads
│   ├── lora_model.py       # Standard LoRA via PEFT
│   ├── lora_layer.py       # Shared LoRA linear wrapper (frozen W + trainable A, B)
│   ├── moe.py              # MoE-LoRA (LoRA experts, top-k routing, load-balancing loss)
│   ├── combined_moe.py     # Task-specific LoRA adapters + MoE layer
│   ├── trainer.py          # Generic training pipeline
│   ├── evaluator.py        # Generic evaluation pipeline
│   └── visualization.py    # Plotting utilities
├── scripts/
│   ├── run_experiments.py  # Main experiment grid (produces every reported number)
│   ├── export_results.py   # Promotes measured artifacts from outputs/ to results/
│   ├── plot_results.py     # Regenerates the paper figures from results/
│   └── inspect_dataset.py  # Dataset sanity check
├── configs/                # YAML configs for the datasets and model variants
├── results/                # Measured result tables + JSON artifacts (version-controlled)
├── paper/                  # LaTeX project (ACM acmart): Multi_LORA.tex, main.tex,
│   │                       # references.bib, figures/, multi_lora.pdf
├── docs/reference_papers/  # Source texts of the three cited papers
├── requirements.txt        # Pip requirements
├── pyproject.toml          # Package metadata, setuptools src-layout
├── run_experiment.sh       # Environment setup (+ optional training)
└── .gitignore
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the `multilora` package
```

The package can also be used without installing: every script bootstraps `sys.path`
with `src/`, so `python scripts/run_experiments.py` works from a clean clone.

---

## Reproducing the results

```bash
# 1. full grid: 5 main variants + 4 ablations (CPU only, several hours)
python scripts/run_experiments.py --group all

# 2. promote the measured artifacts into the version-controlled results/ directory
python scripts/export_results.py

# 3. regenerate the figures
python scripts/plot_results.py
```

`run_experiment.sh --train` performs steps 1-3 in one go. `run_experiment.sh` without
arguments only installs the dependencies.

### Experimental protocol

- **Data**: AG News (`fancyzhx/ag_news`). Train = indices `[0, 5000)`, validation =
  `[5000, 7000)` from the official training split, test = the full official 7,600-example
  test split (1900 per class). SST-2 (`nyu-mll/glue`) is used **only** to measure how the
  trained routers respond to out-of-domain inputs; no SST-2 accuracy is reported.
- **Tokenization**: `distilbert-base-uncased`, truncation at 128 tokens, dynamic padding
  inside each batch.
- **Optimization**: AdamW (weight decay 0.01), gradient clipping at norm 1.0, linear
  warmup over 500 steps then constant LR. LR = 1e-4 (baseline, LoRA) and 5e-5 (MoE variants).
- **Protocol**: one epoch, batch size 16, seed 42, CPU only.

### Measured results (committed in `results/main_results.csv`)

| Model | Trainable | % | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|
| baseline | 6,152 | 0.009% | 0.6295 | 0.6645 | 0.6602 |
| lora | 669,704 | 0.999% | 0.8375 | **0.8722** | **0.8712** |
| moe_n2 (2 experts, top-1) | 130,570 | 0.172% | 0.4885 | 0.5137 | 0.5036 |
| moe_n4 (4 experts, top-2) | 254,988 | 0.298% | 0.5980 | 0.5345 | 0.4952 |
| combined | 19,144,716 | 18.34% | 0.7650 | 0.7703 | 0.7711 |

**The result is negative for the MoE variants**: standard LoRA is the strongest model,
and the MoE routers stay close to uniform (utilization entropy 1.34-1.36 nats out of a
maximum of `ln 4 = 1.3863`), i.e. routing adds cost without an accuracy gain at this
scale. The numbers are reported as measured; no value is estimated or inferred.

---

## Implementation notes worth knowing

- **Which code produced the results.** Every number in `results/` was produced by
  `scripts/run_experiments.py`. The generic pipelines in `src/multilora/trainer.py` and
  `src/multilora/evaluator.py` are *not* on that path, and `trainer.py:evaluate()` is
  currently broken (it reads an undefined `accuracy` variable). The YAML files in
  `configs/` describe the intended configuration; the executed hyperparameters are
  hard-coded in `scripts/run_experiments.py`.
- **Class masking** multiplies each logit vector by a task-specific 0/1 mask, so invalid
  class logits are exactly zero. This zeroes the logit but does not remove the class from
  the softmax denominator, so it is not a strict variable-label-space constraint. For
  AG News (4 valid classes) the mask is a no-op.
- **`expert_capacity` is inert.** It is stored and logged but never enforced: no token is
  ever dropped, and every token is processed by exactly `k` experts.
- **Load-balancing loss.** `L_lb = lambda * CV^2(expert_fractions) * CV^2(mean_probs)` with
  `lambda = 0.01`. The fraction term is built from `topk` indices and is therefore
  non-differentiable, so only the mean-probability term contributes gradient. Its measured
  magnitude (7.7e-5 to 2.6e-4) is four orders of magnitude below the task loss.
- **Padding tokens are routed.** The MoE layer runs over the padded hidden-state tensor;
  in a mixed-length batch roughly 37% of routed tokens are padding positions, which
  contaminates the router statistics and `L_lb`.
- **The MoE output replaces the encoder hidden state** (no residual connection), whereas
  the LoRA variant pools the encoder's `[CLS]` token directly.
- **Single-task training.** All training batches carry `task_id = 0` (AG News), so the
  SST-2 adapter branch of the Combined model never receives a gradient and the multi-task
  mechanism is not exercised by the reported numbers.

---

## Building the paper

```bash
cd paper
pdflatex -jobname=multi_lora Multi_LORA.tex
bibtex multi_lora
pdflatex -jobname=multi_lora Multi_LORA.tex
pdflatex -jobname=multi_lora Multi_LORA.tex
```

`paper/Multi_LORA.tex` is the final single-column (`manuscript`) version and builds
`paper/multi_lora.pdf`; `paper/main.tex` is the two-column (`sigconf`) version. Both
reference their figures as `figures/<name>.png`, so they must be compiled from within
the `paper/` directory of the repository.

---

## License

This project is for educational purposes only. The pretrained model
(`distilbert-base-uncased`) and the `peft`/`transformers` libraries carry their own
licenses; see the upstream projects.
