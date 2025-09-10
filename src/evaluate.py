"""src/evaluate.py
Evaluation utilities, metrics and plotting support for CurvAdaNorm.
All heavyweight dependencies are imported lazily so the module **imports even
without PyTorch / Matplotlib / Seaborn** available.
"""
# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

###############################################################################
#                              Lazy heavy imports                             #
###############################################################################
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, List


def _lazy(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

# Heavy libs (stubs if absent)
torch = _lazy("torch")
matplotlib = _lazy("matplotlib")
_lazy("matplotlib.pyplot")
plt = _lazy("matplotlib.pyplot")
sns = _lazy("seaborn")

CurvGCNII = getattr(importlib.import_module("src.train"), "CurvGCNII", object)

###############################################################################
#                           Metrics: RowDiff & GDR                            #
###############################################################################

def rowdiff(h: "torch.Tensor"):  # type: ignore
    """RowDiff – mean pairwise ℓ2 distance across node features."""
    n = h.size(0)
    with torch.no_grad():
        if n > 5_000:
            idx = torch.randperm(n, device=h.device)[:5_000]
            h = h[idx]
        dists = torch.cdist(h, h, p=2)
        return float(dists.mean().item())


def group_distance_ratio(h: "torch.Tensor", y: "torch.Tensor") -> float:  # type: ignore
    """GDR = inter-class distance / intra-class distance."""
    with torch.no_grad():
        intra = inter = cnt_intra = cnt_inter = 0.0
        for c in y.unique():
            idx_c = (y == c).nonzero(as_tuple=False).view(-1)
            idx_nc = (y != c).nonzero(as_tuple=False).view(-1)
            if idx_c.numel() < 2:
                continue
            h_c = h[idx_c]
            h_nc = h[idx_nc]
            intra += torch.pdist(h_c).mean().item()
            inter += torch.cdist(h_c, h_nc).mean().item()
            cnt_intra += 1
            cnt_inter += 1
        if cnt_intra == 0 or cnt_inter == 0:
            return float("nan")
        return (inter / cnt_inter) / (intra / cnt_intra + 1e-8)

###############################################################################
#                               Full evaluation                               #
###############################################################################

def evaluate(model, data, κ):  # noqa: N803
    """Run model on all splits and compute metrics."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, CurvGCNII):
            out = model(data.x, data.edge_index, κ)
        else:
            out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)
    accs: List[float] = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(float(pred[mask].eq(data.y[mask]).float().mean().item()))

    if isinstance(model, CurvGCNII):
        h_last = model.layers[-2].forward(data.x, data.edge_index, κ)
    else:
        h_last = out

    rd = rowdiff(h_last)
    gdr = group_distance_ratio(h_last, data.y)

    return {
        "train_acc": accs[0],
        "val_acc": accs[1],
        "test_acc": accs[2],
        "rowdiff": rd,
        "gdr": gdr,
    }

###############################################################################
#                     Helpers: save results & simple plots                    #
###############################################################################

def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(obj, fp, indent=2)


def plot_line(x: List[int], y: List[float], *, xlabel: str, ylabel: str, title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x, y=y, marker="o")
    for x_i, y_i in zip(x, y):
        plt.text(x_i, y_i, f"{y_i:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
