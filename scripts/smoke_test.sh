#!/usr/bin/env bash
# Offline end-to-end check of the whole pipeline with a deterministic MOCK LLM (no API key, no internet).
# The numbers it prints are meaningless - it only proves the code path runs.
set -e
cd "$(dirname "$0")/.."
python scripts/make_mini_dataset.py
MOCK_LLM=1 python scripts/evaluate.py --data data/sample_mini.json --n 20 --sweep --sweep_n 10 --budgets 100 300 --judge --out results_smoke
python scripts/make_report.py --results results_smoke --paper /tmp/paper_smoke 2>/dev/null || (mkdir -p /tmp/paper_smoke && python scripts/make_report.py --results results_smoke --paper /tmp/paper_smoke)
echo "SMOKE TEST OK"
