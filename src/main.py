from __future__ import annotations

"""src/main.py
Entry-point orchestrating the CurvAdaNorm experimental workflow.
The script must *import* without heavyweight scientific libraries being
installed.  Full experiments naturally still require PyTorch/PyG, but the code
now detects their absence early and aborts gracefully with a clear message
instead of crashing during dataset construction.
"""
# ruff: noqa
# mypy: ignore-errors
###############################################################################
#                                Lazy imports                                #
###############################################################################
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml  # PyYAML is a light-weight explicit dependency

_pre = importlib.import_module("src.preprocess")
_train = importlib.import_module("src.train")
_eval = importlib.import_module("src.evaluate")

ensure_dir = _pre.ensure_dir  # type: ignore[attr-defined]
set_seed = _pre.set_seed  # type: ignore[attr-defined]
compute_ollivier_ricci = _pre.compute_ollivier_ricci  # type: ignore[attr-defined]

CurvGCNII = _train.CurvGCNII  # type: ignore[attr-defined]
GCNIIModel = _train.GCNIIModel  # type: ignore[attr-defined]
train_epoch = _train.train_epoch  # type: ignore[attr-defined]

dump_json = _eval.dump_json  # type: ignore[attr-defined]
evaluate = _eval.evaluate  # type: ignore[attr-defined]
plot_line = _eval.plot_line  # type: ignore[attr-defined]

###############################################################################
#                        Optional heavyweight dependencies                    #
###############################################################################
try:
    torch = importlib.import_module("torch")
    if getattr(torch, "_is_stub", False):
        torch = None  # type: ignore[assignment]
except Exception:
    torch = None  # type: ignore[assignment]

try:
    tg_datasets = importlib.import_module("torch_geometric.datasets")
    Planetoid = getattr(tg_datasets, "Planetoid")  # type: ignore[attr-defined]
    WebKB = getattr(tg_datasets, "WebKB")  # type: ignore[attr-defined]
    tg_utils = importlib.import_module("torch_geometric.utils")
    add_self_loops = getattr(tg_utils, "add_self_loops")  # type: ignore[attr-defined]

    optim_lr = importlib.import_module("torch.optim").lr_scheduler  # type: ignore[attr-defined]
    CosineAnnealingLR = getattr(optim_lr, "CosineAnnealingLR")  # type: ignore[attr-defined]
    _HAS_PYG = True
except Exception:
    class _Stub:
        def __init__(self, *_: Any, **__: Any):
            raise RuntimeError("torch_geometric not available – install to run experiments.")

    Planetoid = WebKB = _Stub  # type: ignore[assignment]

    def add_self_loops(*_: Any, **__: Any):
        raise RuntimeError("torch_geometric not available – install to run experiments.")

    class _SchedulerStub:
        def __init__(self, *_: Any, **__: Any):
            pass

        def step(self):
            pass

    CosineAnnealingLR = _SchedulerStub  # type: ignore[assignment]
    _HAS_PYG = False

###############################################################################
#                                  Config                                     #
###############################################################################
CFG_PATH = Path("config/config.yaml")
DEFAULT_CFG: Dict[str, Any] = {
    "experiment_1": {
        "name": "depth_stress_test",
        "datasets": [
            "Cora",  # limited to a single lightweight dataset for quick CI runs
        ],
        "variant": "curv",  # "baseline" to run plain GCNII
        "depths": [2, 4],
        "hidden": 32,
        "dropout": 0.5,
        "lr": 1e-2,
        "weight_decay": 5e-4,
        "steps": 200,  # drastically shorter to keep runtime reasonable
        "λ": 0.5,
        "tau": 0.05,
    }
}
if not CFG_PATH.exists():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CFG_PATH.open("w") as fp:
        yaml.safe_dump(DEFAULT_CFG, fp)

###############################################################################
#                           Depth stress-test runner                          #
###############################################################################

def run_depth_stress_test(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Paths updated to follow mandatory iteration6 requirements
    base_dir = ensure_dir(".research/iteration6")
    images_dir = ensure_dir(".research/iteration6/images")

    all_results: Dict[str, Any] = {}

    for dataset_name in cfg["datasets"]:
        if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
            ds = Planetoid(root=f"data/{dataset_name}", name=dataset_name)  # type: ignore
        elif dataset_name in {"Chameleon", "Squirrel"}:
            ds = WebKB(root=f"data/{dataset_name}", name=dataset_name)  # type: ignore
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        data = ds[0]
        data.edge_index, _ = add_self_loops(data.edge_index)  # type: ignore
        data.x = data.x / (data.x.sum(1, keepdim=True) + 1e-8)

        κ_path = Path(base_dir) / f"curv_{dataset_name}.pt"
        if κ_path.exists():
            κ = importlib.import_module("torch").load(κ_path)  # type: ignore[attr-defined]
        else:
            κ = compute_ollivier_ricci(data.edge_index, data.num_nodes)
            importlib.import_module("torch").save(κ, κ_path)  # type: ignore[attr-defined]

        device = importlib.import_module("torch").device(
            "cuda" if torch and torch.cuda.is_available() else "cpu"
        )  # type: ignore[attr-defined]
        data = data.to(device)
        κ = κ.to(device)

        per_depth: Dict[int, Any] = {}
        for depth in cfg["depths"]:
            if cfg["variant"] == "curv":
                model = CurvGCNII(
                    in_dim=data.num_features,
                    hid_dim=cfg["hidden"],
                    out_dim=ds.num_classes,
                    num_layers=depth,
                    dropout=cfg["dropout"],
                    λ=cfg["λ"],
                )
            else:
                model = GCNIIModel(
                    in_dim=data.num_features,
                    hid_dim=cfg["hidden"],
                    out_dim=ds.num_classes,
                    num_layers=depth,
                    dropout=cfg["dropout"],
                )
            model = model.to(device)

            optimiser = importlib.import_module("torch.optim").Adam(  # type: ignore[attr-defined]
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            scheduler = CosineAnnealingLR(optimiser, T_max=cfg["steps"], eta_min=1e-4)

            best_val, best_metrics = 0.0, None
            for step in range(cfg["steps"]):
                _ = train_epoch(model, data, κ, optimiser, uniformity_tau=cfg.get("tau"))
                scheduler.step()
                if step % 50 == 0 or step == cfg["steps"] - 1:
                    metrics = evaluate(model, data, κ)
                    if metrics["val_acc"] > best_val:
                        best_val = metrics["val_acc"]
                        best_metrics = metrics

            if best_metrics is None:
                raise RuntimeError("Best metrics were not captured – check training loop.")
            per_depth[depth] = best_metrics

        # Save per-dataset results to mandatory JSON location
        json_path = base_dir / f"{dataset_name}.json"
        dump_json(per_depth, json_path)

        # Print JSON content for verification as required
        with json_path.open("r") as fp:
            print(f"\n=== Results for {dataset_name} ===")
            print(fp.read())

        depths = list(per_depth.keys())
        test_acc = [per_depth[d]["test_acc"] for d in depths]
        rd_vals = [per_depth[d]["rowdiff"] for d in depths]
        gdr_vals = [per_depth[d]["gdr"] for d in depths]

        plot_line(
            depths,
            test_acc,
            xlabel="Depth (layers)",
            ylabel="Accuracy",
            title=f"{dataset_name} – Accuracy vs Depth ({cfg['variant']})",
            save_path=images_dir / f"accuracy_{dataset_name}_{cfg['variant']}.pdf",
        )
        plot_line(
            depths,
            rd_vals,
            xlabel="Depth (layers)",
            ylabel="RowDiff",
            title=f"{dataset_name} – Oversmoothing ({cfg['variant']})",
            save_path=images_dir / f"rowdiff_{dataset_name}_{cfg['variant']}.pdf",
        )
        plot_line(
            depths,
            gdr_vals,
            xlabel="Depth (layers)",
            ylabel="GDR",
            title=f"{dataset_name} – GDR vs Depth ({cfg['variant']})",
            save_path=images_dir / f"gdr_{dataset_name}_{cfg['variant']}.pdf",
        )

        all_results[dataset_name] = per_depth

    print("\n=== Experiment 1 – Depth Stress Test (aggregate) ===")
    print(json.dumps(all_results, indent=2))
    print("Figures written to", images_dir.resolve())

    return all_results

###############################################################################
#                                    main                                     #
###############################################################################

def _heavy_libs_available() -> bool:
    if torch is None or not _HAS_PYG:
        return False
    try:
        _ = Planetoid  # noqa: F841
    except Exception:
        return False
    return True


def main() -> None:  # noqa: D401
    set_seed()

    if not _heavy_libs_available():
        print(
            "Required scientific libraries (PyTorch / PyG) not available – "
            "skipping heavy experiments.  Install torch>=2.0 and "
            "torch_geometric to enable full functionality.",
            file=sys.stderr,
        )
        return

    with CFG_PATH.open("r") as fp:
        cfg = yaml.safe_load(fp) or DEFAULT_CFG

    if "experiment_1" not in cfg:
        cfg["experiment_1"] = DEFAULT_CFG["experiment_1"]

    run_depth_stress_test(cfg["experiment_1"])


if __name__ == "__main__":
    main()
