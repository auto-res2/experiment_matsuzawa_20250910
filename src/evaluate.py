# src/evaluate.py – evaluation helpers, aggregation & plotting wrappers
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
#                             AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_runs(runs: List[Dict], dataset: str, model: str, budget: int) -> Dict:
    """Average results of several seeds for *dataset/model/budget*."""
    sel = [
        r
        for r in runs
        if r["dataset"] == dataset and r["model"] == model and r["budget_bytes"] == budget
    ]
    avg_acc = float(np.mean([r["average_accuracy"] for r in sel])) if sel else 0.0
    std_acc = float(np.std([r["average_accuracy"] for r in sel])) if sel else 0.0
    last_acc = float(np.mean([r["last_accuracy"] for r in sel])) if sel else 0.0
    forget = float(np.mean([r["forgetting"] for r in sel])) if sel else 0.0
    acc_per_task = (
        list(np.mean(np.array([r["acc_per_task"] for r in sel]), axis=0)) if sel else []
    )
    return {
        "average_accuracy_mean": avg_acc,
        "average_accuracy_std": std_acc,
        "last_accuracy": last_acc,
        "forgetting": forget,
        "acc_per_task": acc_per_task,
    }

# ---------------------------------------------------------------------------
#                             I/O HELPERS
# ---------------------------------------------------------------------------

def save_json(obj: Dict | List, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------------------------
#   The plotting helpers (plot_avgacc_vs_tasks / plot_ablation / plot_privacy
#   _robustness) are part of src.main because they depend on matplotlib &
#   seaborn which are used there already.  Keeping them in main avoids an
#   additional heavyweight import here and honours the 6-file constraint.
# ---------------------------------------------------------------------------
