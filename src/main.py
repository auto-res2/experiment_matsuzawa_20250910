# main.py
"""Entry-point orchestrating Experiment-1 (memory-budget sweep)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

# -----------------------------------------------------------------------------
# Local imports – use plain imports first; fall back to absolute if required
# -----------------------------------------------------------------------------
try:
    import evaluate as ev
    import preprocess as prep
    import train as tr
except ImportError:  # likely executed with `python -m src.main`
    from src import evaluate as ev  # type: ignore
    from src import preprocess as prep  # type: ignore
    from src import train as tr  # type: ignore

# -----------------------------------------------------------------------------
# Paths & configuration helpers
# -----------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PACKAGE_ROOT / "config" / "config.yaml"
RESEARCH_DIR = PACKAGE_ROOT / ".research" / "iteration2"
IMAGES_DIR = RESEARCH_DIR / "images"
# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def _load_cfg():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _device(cfg_common):
    return torch.device(cfg_common["device"] if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Experiment-1
# -----------------------------------------------------------------------------

def run_experiment_1(cfg):
    common = cfg["common"]
    e_cfg = cfg["experiment_1"]
    device = _device(common)
    json_results: dict[str, dict] = {}

    for dname, dcfg in e_cfg["datasets"].items():
        ds = prep.ContinualSplit(
            dname,
            dcfg["hf_repo"],
            dcfg["n_tasks"],
            dcfg["classes_per_task"],
            dcfg["img_size"],
            common["data_root"],
        )
        for budget in e_cfg["budgets_bytes"]:
            for method in e_cfg["methods"]:
                run_key = f"{dname}_{method}_{budget//1024}KB"
                print(f"\n=== Running {run_key} ===")
                torch.manual_seed(common["seeds"][0])
                model = tr.build_model(
                    method,
                    dcfg["n_tasks"] * dcfg["classes_per_task"],
                    k=e_cfg["sketch_sizes"][0],
                    sparsity=e_cfg["sparsities"][0],
                ).to(device)
                # training across tasks
                acc_curve = []
                for task_id in range(dcfg["n_tasks"]):
                    loader = ds.get_task_loader(
                        task_id, e_cfg["optim"]["batch_live"], common["num_workers"]
                    )
                    tr.train_one_task(
                        model, loader, {**e_cfg["optim"], **common}, device, budget=budget
                    )
                    acc = ev.evaluate(model, ds.get_test_loader(256, common["num_workers"]), device)
                    acc_curve.append(acc)
                    print(f"Task {task_id} acc: {acc:.2f}%")

                res = {
                    "accuracy_per_task": acc_curve,
                    "avg_final_accuracy": sum(acc_curve[-5:]) / 5,
                }
                json_results[run_key] = res

                # save json per-run
                json_fp = RESEARCH_DIR / f"{run_key}.json"
                with open(json_fp, "w") as f:
                    json.dump(res, f, indent=2)
                # print for immediate verification
                print(json.dumps(res, indent=2))

                # plot and save figure
                ev.plot_accuracy_curve(
                    acc_curve,
                    IMAGES_DIR / f"accuracy_{run_key}.pdf",
                    title=run_key,
                )

    return json_results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    try:
        cfg = _load_cfg()
    except FileNotFoundError:
        print("Configuration file not found. Abort.")
        sys.exit(1)

    try:
        run_experiment_1(cfg)
    except tr.MemoryBudgetExceeded as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
