#!/usr/bin/env bash
# Set up the environment for the Multi-LoRA + MoE experiments.
#
# Usage:  ./run_experiment.sh            # install dependencies
#         ./run_experiment.sh --train    # additionally run the full experiment grid
#
# NOTE: the full grid trains 5 main variants + 4 ablations on CPU and takes
# several hours. Use scripts/run_experiments.py --group main for a shorter run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "Multi-LoRA + MoE :: environment setup"
echo "====================================="

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON not found. Install Python 3.9 or newer." >&2
    exit 1
fi

echo "Python:  $($PYTHON --version 2>&1)"
echo "Installing requirements..."
"$PYTHON" -m pip install -r requirements.txt

echo
echo "Installing the 'multilora' package in editable mode..."
"$PYTHON" -m pip install -e . --no-deps

if [[ "${1:-}" == "--train" ]]; then
    echo
    echo "Running the main experiment grid (writes to outputs/experiments/ag_news/)..."
    "$PYTHON" scripts/run_experiments.py --group main

    echo
    echo "Exporting measured artifacts to results/..."
    "$PYTHON" scripts/export_results.py

    echo
    echo "Regenerating figures..."
    "$PYTHON" scripts/plot_results.py
fi

echo
echo "Setup complete."
echo "  - experiments: python3 scripts/run_experiments.py --group all"
echo "  - figures:     python3 scripts/plot_results.py"
echo "  - paper:       cd paper && pdflatex -jobname=multi_lora Multi_LORA.tex"
