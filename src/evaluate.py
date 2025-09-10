from typing import List, Tuple

import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

#############################################
# 1.  ACCURACY & EVAL                      #
#############################################

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean classification accuracy."""
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def eval_model(model, data, mask) -> Tuple[float, torch.Tensor]:
    """Evaluate model on given node split mask."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        acc = accuracy(logits[mask], data.y.squeeze()[mask])
    return acc, logits

#############################################
# 2.  VISUALISATION                        #
#############################################

def line_plot(x: List[int], y: List[float], *, ylabel: str, fname, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label=ylabel)
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.3f}", (xi, yi))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / fname, bbox_inches="tight", format="pdf")
    plt.close()
