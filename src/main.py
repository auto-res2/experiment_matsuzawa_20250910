"""High-level experiment runner.  Execute with
    python -m src.main
This script orchestrates the depth-scalability experiment (Exp-1) using
configuration loaded from config/config.yaml and stores artifacts under
.research/iteration1/ …
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.optim as optim
import yaml

from .train import (GNNStack, SEED_LIST, TrainState, set_seed,
                    train_one_epoch)
from .evaluate import eval_model, line_plot
from .preprocess import load_dataset

# ---------------------------------------------------------------------
# 1.  DIRECTORIES & DEVICE
# ---------------------------------------------------------------------
RESEARCH_DIR = Path(".research") / "iteration1"
IMG_DIR = RESEARCH_DIR / "images"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 2.  CONFIGURATION
# ---------------------------------------------------------------------
CONFIG_PATH = Path("config") / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Missing config/config.yaml – cannot continue.")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

# ---------------------------------------------------------------------
# 3.  EXP-1 DEPTH SCALABILITY
# ---------------------------------------------------------------------

def run_depth_scalability(cfg_exp: Dict[str, Any]):
    """Depth-scalability stress-test."""
    print("\n========== EXPERIMENT 1 – DEPTH SCALABILITY ==========")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset_name in cfg_exp["datasets"]:
        data = load_dataset(dataset_name).to(DEVICE)
        res_dataset: Dict[str, Dict[str, float]] = {}

        for depth in cfg_exp["depth_grid"]:
            cfg_model = {
                "backbone": cfg_exp["backbone"],
                "wrapper" : cfg_exp["wrapper"],
                "depth"   : depth,
                "hidden"  : cfg_exp["hidden"],
                "dropout" : cfg_exp["dropout"],
                "K"       : cfg_exp["K"],
            }

            acc_seeds: List[float] = []
            for seed in SEED_LIST:
                set_seed(seed)
                model = GNNStack(cfg_model, data).to(DEVICE)
                opt = optim.AdamW(model.parameters(), lr=cfg_exp["lr"],
                                   weight_decay=cfg_exp["weight_decay"])
                state = TrainState(epoch=0, best_val=0.0, best_epoch=0)

                best_state_dict = None
                for epoch in range(cfg_exp["epochs"]):
                    loss = train_one_epoch(model, data, opt)
                    val_acc, _ = eval_model(model, data, data.val_mask)

                    if val_acc > state.best_val:
                        state.best_val = val_acc
                        state.best_epoch = epoch
                        best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                    if epoch - state.best_epoch > cfg_exp["patience"]:
                        break

                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                test_acc, _ = eval_model(model, data, data.test_mask)
                acc_seeds.append(test_acc)

            mean_acc = float(torch.tensor(acc_seeds).mean().item())
            std_acc = float(torch.tensor(acc_seeds).std(unbiased=False).item())
            res_dataset[str(depth)] = {"mean": mean_acc, "std": std_acc}
            print(f"{dataset_name:12s} depth={depth:3d}: {mean_acc:.3f} ± {std_acc:.3f}")

        results[dataset_name] = res_dataset

    # ------------------------------------------------------------------
    # Save results & plot for first dataset
    # ------------------------------------------------------------------
    out_file = RESEARCH_DIR / "exp1_depth_scalability.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nJSON results saved to", out_file)
    print(json.dumps(results, indent=2))

    # Visualise on first dataset
    first = cfg_exp["datasets"][0]
    xs = list(map(int, results[first].keys()))
    ys = [results[first][str(d)]["mean"] for d in xs]
    line_plot(xs, ys, ylabel="Accuracy", fname="accuracy_depth.pdf", save_dir=IMG_DIR)
    print("Figure saved to", IMG_DIR / "accuracy_depth.pdf")

# ---------------------------------------------------------------------
# 4.  MAIN
# ---------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("medium")
    cfg_exp1 = CONFIG["exp1"]

    start = time.time()
    run_depth_scalability(cfg_exp1)
    elapsed = (time.time() - start) / 60.0
    print(f"\nTotal wall-clock time: {elapsed:.2f} min")

if __name__ == "__main__":
    main()
