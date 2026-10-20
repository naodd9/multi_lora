"""Reduced-protocol AG News experiments for the Multi-LoRA + MoE paper.

Protocol: train on examples [0, 5000), validate on [5000, 7000), test on the
official 7,600-example test split. One epoch, batch 16, max length 128.
Dynamic padding is used inside each batch, so no computation is spent on
padding tokens; the maximum length remains 128.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["NO_PROXY"] = "*"
for _v in ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
    os.environ.pop(_v, None)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multilora.baseline_model import BaselineModel
from multilora.lora_model import LoRAModel
from multilora.moe import MoLoRAModel
from multilora.combined_moe import CombinedMultiLoRAMoEModel

torch.set_num_threads(3)
SEED = 42
TASKS = {"ag_news": 4, "sst2": 2}
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
OUT = ROOT / "outputs" / "experiments" / "ag_news"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "tokenized_data.pt"


class TokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels, task_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.task_ids = task_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_masks[i],
            "labels": self.labels[i],
            "task_id": self.task_ids[i],
        }


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        mask[i, :n] = 1
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "task_id": torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
    }


def tokenize_examples(tokenizer, examples, task_id=0):
    ids, masks, labels = [], [], []
    for text, label in examples:
        enc = tokenizer(str(text), truncation=True, max_length=128)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
        labels.append(int(label))
    return TokenizedDataset(ids, masks, labels, [task_id] * len(labels))


def prepare_data():
    if CACHE.exists():
        raw = torch.load(CACHE, weights_only=False)
        return {k: TokenizedDataset(*v) for k, v in raw.items()}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = load_dataset("fancyzhx/ag_news", cache_dir=str(ROOT / "data" / "cache"))["train"]
    test_ds = load_dataset("fancyzhx/ag_news", cache_dir=str(ROOT / "data" / "cache"))["test"]

    def rows(dataset, lo, hi):
        return [(dataset[i]["text"], dataset[i]["label"]) for i in range(lo, hi)]

    splits = {
        "train": tokenize_examples(tokenizer, rows(ds, 0, 5000), 0),
        "val": tokenize_examples(tokenizer, rows(ds, 5000, 7000), 0),
        "test": tokenize_examples(tokenizer, rows(test_ds, 0, len(test_ds)), 0),
    }
    torch.save(
        {k: (v.input_ids, v.attention_masks, v.labels, v.task_ids) for k, v in splits.items()},
        CACHE,
    )
    return splits


def prepare_sst2():
    path = OUT / "sst2_routing.pt"
    if path.exists():
        raw = torch.load(path, weights_only=False)
        return TokenizedDataset(*raw)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = load_dataset("nyu-mll/glue", "sst2", cache_dir=str(ROOT / "data" / "cache"))["validation"]
    examples = [(ds[i]["sentence"], ds[i]["label"]) for i in range(len(ds)) if ds[i]["label"] in (0, 1)]
    ds_obj = tokenize_examples(tokenizer, examples, 1)
    torch.save((ds_obj.input_ids, ds_obj.attention_masks, ds_obj.labels, ds_obj.task_ids), path)
    return ds_obj


def make_loaders(splits, batch_size=16):
    g = torch.Generator()
    g.manual_seed(SEED)
    return {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                            collate_fn=collate, generator=g, num_workers=0),
        "val": DataLoader(splits["val"], batch_size=64, shuffle=False,
                          collate_fn=collate, num_workers=0),
        "test": DataLoader(splits["test"], batch_size=64, shuffle=False,
                           collate_fn=collate, num_workers=0),
    }


def build_model(name, rank=8, experts=4, k=2, lr=None):
    common = dict(model_name="distilbert-base-uncased", num_classes_per_task=TASKS)
    if name == "baseline":
        return BaselineModel(**common), (lr or 1e-4)
    if name == "lora":
        return LoRAModel(lora_rank=rank, lora_alpha=16, **common), (lr or 1e-4)
    if name.startswith("moe"):
        return MoLoRAModel(num_experts=experts, expert_capacity=32, k=k,
                           lora_rank=rank, lora_alpha=16,
                           load_balancing_loss_coeff=0.01, **common), (lr or 5e-5)
    if name == "combined":
        return CombinedMultiLoRAMoEModel(num_experts_per_task=2, num_moe_experts=experts,
                                        moe_expert_capacity=32, moe_k=k,
                                        lora_rank=rank, lora_alpha=16,
                                        load_balancing_loss_coeff=0.01, **common), (lr or 5e-5)
    raise ValueError(name)


def unpack(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, torch.tensor(0.0, device=output.device)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            logits, _ = unpack(model(batch["input_ids"], batch["attention_mask"], batch["task_id"]))
            total_loss += ce(logits, batch["labels"]).item()
            preds.append(logits.argmax(-1).cpu())
            labels.append(batch["labels"].cpu())
            n += 1
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(4)), average=None, zero_division=0
    )
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in range(4)
        },
        "predictions": preds,
        "labels": labels,
    }


def routing_stats(model, dataset, max_examples=512):
    if not hasattr(model, "moe_layer"):
        return {"routing_probabilities": [0.0] * 4, "entropy": 0.0}
    subset = TokenizedDataset(
        dataset.input_ids[:max_examples], dataset.attention_masks[:max_examples],
        dataset.labels[:max_examples], dataset.task_ids[:max_examples],
    )
    loader = DataLoader(subset, batch_size=64, shuffle=False, collate_fn=collate)
    captured = []
    handle = model.moe_layer.gate.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().float().cpu())
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(batch["input_ids"], batch["attention_mask"], batch["task_id"])
    handle.remove()
    if not captured:
        return {"routing_probabilities": [], "entropy": 0.0}
    logits = torch.cat(captured, dim=0)
    probs = torch.softmax(logits, dim=-1)
    mean_probs = probs.mean(0).tolist()
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum(-1).mean())
    return {"routing_probabilities": mean_probs, "entropy": entropy}


def masked_logit_magnitude(model, dataset, max_examples=128):
    subset = TokenizedDataset(dataset.input_ids[:max_examples], dataset.attention_masks[:max_examples],
                              dataset.labels[:max_examples], dataset.task_ids[:max_examples])
    loader = DataLoader(subset, batch_size=64, shuffle=False, collate_fn=collate)
    worst = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, _ = unpack(model(batch["input_ids"], batch["attention_mask"], batch["task_id"]))
            if logits.shape[1] > 2:
                worst = max(worst, float(logits[:, 2:].abs().max()))
    return worst


def train_one(name, model, lr, loaders, sst2, epochs=1):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    opt = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr, weight_decay=0.01)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: float(s) / float(max(1, warmup)) if s < warmup else 1.0
    )
    ce = nn.CrossEntropyLoss()
    curve, grad_norms = [], []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "task": 0.0, "lb": 0.0, "acc": 0.0}
        nb = 0
        for batch in loaders["train"]:
            opt.zero_grad()
            logits, lb = unpack(model(batch["input_ids"], batch["attention_mask"], batch["task_id"]))
            task_loss = ce(logits, batch["labels"])
            loss = task_loss + lb
            loss.backward()
            total_norm = math.sqrt(sum(float(p.grad.detach().norm() ** 2)
                                       for p in model.get_trainable_parameters()
                                       if p.grad is not None))
            grad_norms.append(total_norm)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            sums["loss"] += loss.item()
            sums["task"] += task_loss.item()
            sums["lb"] += float(lb)
            sums["acc"] += float((logits.argmax(-1) == batch["labels"]).float().mean())
            nb += 1
        train_time = time.time() - start
        val = evaluate(model, loaders["val"])
        rout = routing_stats(model, loaders["val"].dataset)
        curve.append({
            "model": name,
            "epoch": epoch + 1,
            "train_loss": sums["loss"] / nb,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_f1": val["macro_f1"],
            "load_balancing_loss": sums["lb"] / nb,
            "expert_utilization_entropy": rout["entropy"],
            "task_loss": sums["task"] / nb,
            "train_acc": sums["acc"] / nb,
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
        })
        print(f"[{name}] epoch {epoch+1}: train_loss={sums['loss']/nb:.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} "
              f"gradmax={np.max(grad_norms):.4f} time={train_time:.1f}s", flush=True)

    test = evaluate(model, loaders["test"])
    ag_routing = routing_stats(model, loaders["val"].dataset, max_examples=512)
    sst_routing = routing_stats(model, sst2, max_examples=512) if sst2 else {}
    masked = masked_logit_magnitude(model, sst2) if sst2 else 0.0
    result = {
        "model": name,
        "trainable_params": int(model.get_num_trainable_parameters()),
        "total_params": int(model.get_num_total_parameters()),
        "pct_trainable": 100.0 * model.get_num_trainable_parameters() / model.get_num_total_parameters(),
        "val_acc": curve[-1]["val_acc"],
        "val_f1": curve[-1]["val_f1"],
        "test_acc": test["accuracy"],
        "test_f1": test["macro_f1"],
        "train_time_sec": train_time,
        "epochs": epochs,
        "per_class": test["per_class"],
        "routing_ag_news": ag_routing,
        "routing_sst2": sst_routing,
        "masked_logit_max": masked,
        "curve": curve,
    }
    return result


def write_outputs(results, curves, ablations=None):
    results = sorted(results, key=lambda r: ["baseline", "lora", "moe_n2", "moe_n4", "combined"].index(r["model"])
                     if r["model"] in ["baseline", "lora", "moe_n2", "moe_n4", "combined"] else 99)
    fields = ["model", "trainable_params", "total_params", "pct_trainable", "val_acc",
              "val_f1", "test_acc", "test_f1", "train_time_sec", "epochs"]
    if results:
        with open(OUT / "main_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in fields})
        main_curves = [c for c in curves if c["model"] in
                       ["baseline", "lora", "moe_n2", "moe_n4", "combined"]]
        with open(OUT / "per_epoch_metrics.csv", "w", newline="") as f:
            fields2 = ["model", "epoch", "train_loss", "val_loss", "val_acc", "val_f1",
                       "load_balancing_loss", "expert_utilization_entropy", "grad_norm_mean", "grad_norm_max"]
            w = csv.DictWriter(f, fieldnames=fields2)
            w.writeheader()
            for row in main_curves:
                w.writerow({k: row[k] for k in fields2})
        with open(OUT / "results_full.json", "w") as f:
            json.dump(results, f, indent=2)
    if ablations:
        afields = fields
        with open(OUT / "ablation_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=afields)
            w.writeheader()
            for r in ablations:
                w.writerow({k: r[k] for k in afields})
        for r in ablations:
            with open(OUT / f"{r['model']}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=afields)
                w.writeheader()
                w.writerow({k: r[k] for k in afields})
        with open(OUT / "ablation_full.json", "w") as f:
            json.dump(ablations, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["main", "ablation", "all"], default="main")
    args = ap.parse_args()
    splits = prepare_data()
    sst2 = prepare_sst2()
    loaders = make_loaders(splits)
    print(f"data: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} sst2={len(sst2)}", flush=True)

    specs = {
        "baseline": ("baseline", 8, 4, 2, 1e-4),
        "lora": ("lora", 8, 4, 2, 1e-4),
        "moe_n2": ("moe", 8, 2, 1, 5e-5),
        "moe_n4": ("moe", 8, 4, 2, 5e-5),
        "combined": ("combined", 8, 4, 2, 5e-5),
    }
    abl_specs = {
        "ablation_rank4": ("moe", 4, 4, 2, 5e-5),
        "ablation_rank16": ("moe", 16, 4, 2, 5e-5),
        "ablation_topk1": ("moe", 8, 4, 1, 5e-5),
        "ablation_n2": ("moe", 8, 2, 2, 5e-5),
    }

    results, curves, ablations = [], [], []
    if args.group in ("main", "all"):
        for name, (kind, rank, experts, k, lr) in specs.items():
            print(f"=== training {name} ===", flush=True)
            model, lr = build_model(kind, rank, experts, k, lr)
            r = train_one(name, model, lr, loaders, sst2)
            results.append(r)
            curves.extend(r["curve"])
            with open(OUT / f"{name}_full.json", "w") as f:
                json.dump(r, f, indent=2)
            write_outputs(results, curves)
    if args.group in ("ablation", "all"):
        for name, (kind, rank, experts, k, lr) in abl_specs.items():
            print(f"=== training {name} ===", flush=True)
            model, lr = build_model(kind, rank, experts, k, lr)
            r = train_one(name, model, lr, loaders, sst2)
            ablations.append(r)
            curves.extend(r["curve"])
            with open(OUT / f"{name}_full.json", "w") as f:
                json.dump(r, f, indent=2)
            write_outputs(results, curves, ablations)
    print("done", flush=True)


if __name__ == "__main__":
    main()
