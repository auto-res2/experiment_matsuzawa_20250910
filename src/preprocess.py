import sys
from pathlib import Path

from torch_geometric.datasets import Planetoid, WebKB
from ogb.nodeproppred import PygNodePropPredDataset
import torch

ROOT = Path("data")

__all__ = ["load_dataset"]

def _ensure_1d_masks(data):
    """Convert 2-D split masks (N, S) into 1-D (N,) by selecting split 0."""
    for key in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, key, None)
        if mask is None:
            continue
        if mask.dim() == 2:
            # Select first split column – fail-fast if shape unexpected.
            if mask.size(1) < 1:
                raise RuntimeError(f"Empty split mask encountered for {key}.")
            setattr(data, key, mask[:, 0].clone())
    return data


def load_dataset(name: str):
    """Download or load a dataset by name.  Raises RuntimeError if unavailable (NO-FALLBACK)."""
    root = ROOT / name
    try:
        if name in ["Cora", "Citeseer", "PubMed"]:
            ds = Planetoid(root=str(root), name=name)
            return _ensure_1d_masks(ds[0])
        if name in ["Cornell", "Texas", "Wisconsin"]:
            ds = WebKB(root=str(root), name=name)
            return _ensure_1d_masks(ds[0])
        if name == "ogbn-arxiv":
            ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
            data = ds[0]
            split_idx = ds.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx["train"]] = True
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[split_idx["valid"]] = True
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[split_idx["test"]] = True
            return data
    except Exception as exc:  # pragma: no-cover  pylint: disable=broad-except
        raise RuntimeError(f"Failed to load dataset '{name}': {exc}") from exc

    raise RuntimeError(
        f"Dataset '{name}' not supported – aborting as per NO-FALLBACK policy."
    )
