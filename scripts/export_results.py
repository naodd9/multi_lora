#!/usr/bin/env python3
"""Export the measured experiment artifacts into the committed ``results/`` directory.

``scripts/run_experiments.py`` writes every artifact it measures to
``outputs/experiments/ag_news/`` (git-ignored, overwritten on each run).  This
script promotes the measured files into ``results/`` so that the numbers quoted
in the paper are version-controlled and the figures can be regenerated without
re-training.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "experiments" / "ag_news"
TARGET = ROOT / "results"

ARTIFACTS = [
    "main_results.csv",
    "per_epoch_metrics.csv",
    "results_full.json",
    "ablation_results.csv",
    "ablation_full.json",
    "baseline_full.json",
    "lora_full.json",
    "moe_n2_full.json",
    "moe_n4_full.json",
    "combined_full.json",
    "ablation_n2.csv",
    "ablation_n2_full.json",
    "ablation_rank4.csv",
    "ablation_rank4_full.json",
    "ablation_rank16.csv",
    "ablation_rank16_full.json",
    "ablation_topk1.csv",
    "ablation_topk1_full.json",
]


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: no measured artifacts at {SOURCE}", file=sys.stderr)
        print("Run 'python scripts/run_experiments.py --group all' first.", file=sys.stderr)
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in ARTIFACTS:
        src = SOURCE / name
        if src.exists():
            shutil.copy2(src, TARGET / name)
            copied.append(name)
        else:
            missing.append(name)

    print(f"exported {len(copied)} artifacts -> {TARGET.relative_to(ROOT)}")
    if missing:
        print(f"missing (not produced by this run): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
