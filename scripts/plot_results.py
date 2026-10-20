"""Build the three paper figures from the measured experiment outputs.

Reads the version-controlled artifacts in ``results/`` (populated by
``scripts/export_results.py``) and writes the figures to ``figures/``.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MAIN_ORDER = ["baseline", "lora", "moe_n2", "moe_n4", "combined"]
LABELS = {
    "baseline": "Baseline",
    "lora": "Standard LoRA",
    "moe_n2": "MoE-LoRA (n=2)",
    "moe_n4": "MoE-LoRA (n=4)",
    "combined": "Combined",
}


def load_json(name):
    with open(OUT / name) as f:
        return json.load(f)


def routing_figure(results):
    selected = [r for r in results if r["model"] in ("moe_n2", "moe_n4", "combined")]
    if not selected:
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.0), squeeze=False)
    for ax, r in zip(axes[0], selected):
        ag = r["routing_ag_news"]["routing_probabilities"]
        sst = r["routing_sst2"].get("routing_probabilities", []) if r.get("routing_sst2") else []
        n = max(len(ag), len(sst))
        x = np.arange(n)
        w = 0.38
        ax.bar(x - w / 2, ag, width=w, label="AG News", color="#3b6ea5")
        if sst:
            ax.bar(x + w / 2, sst, width=w, label="SST-2 inputs", color="#c26d3a")
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{i+1}" for i in range(n)])
        ax.set_ylabel("Mean routing probability")
        ax.set_title(LABELS.get(r["model"], r["model"]), fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Per-expert routing probability (measured)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "routing_analysis.png", dpi=200)
    plt.close(fig)


def curve_figure(rows, metric, ylabel, title, filename, logy=False):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for model in MAIN_ORDER:
        vals = [r[metric] for r in rows if r["model"] == model]
        if not vals:
            continue
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, marker="o", label=LABELS[model], linewidth=1.8)
        for x, y in zip(epochs, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(3, 4), fontsize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([1] if len(rows) else [])
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=200)
    plt.close(fig)


def main():
    with open(OUT / "per_epoch_metrics.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("train_loss", "val_acc", "val_f1"):
            r[k] = float(r[k])
    with open(OUT / "results_full.json") as f:
        results = json.load(f)
    routing_figure(results)
    curve_figure(rows, "train_loss", "Training loss", "Training loss per epoch (1-epoch protocol)",
                 "train_loss.png", logy=False)
    curve_figure(rows, "val_acc", "Validation accuracy", "Validation accuracy per epoch (1-epoch protocol)",
                 "val_acc.png", logy=False)
    print("figures written")


if __name__ == "__main__":
    main()
