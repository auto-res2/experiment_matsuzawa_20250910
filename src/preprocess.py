import sys
from pathlib import Path

from torch_geometric.datasets import Planetoid, WebKB
from ogb.nodeproppred import PygNodePropPredDataset
import torch

ROOT = Path("data")

__all__ = ["load_dataset"]

def load_dataset(name: str):
    """Download or load a dataset by name.  Raises RuntimeError if unavailable (NO-FALLBACK)."""
    root = ROOT / name
    try:
        if name in ["Cora", "Citeseer", "PubMed"]:
            ds = Planetoid(root=str(root), name=name)
            return ds[0]
        if name in ["Cornell", "Texas", "Wisconsin"]:
            ds = WebKB(root=str(root), name=name)
            return ds[0]
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
