import random
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GCN2Conv,
    MessagePassing,
    PairNorm,
)
from torch_geometric.utils import degree

__all__ = [
    "LCAMPLayer",
    "build_backbone",
    "GNNStack",
    "SEED_LIST",
    "set_seed",
    "TrainState",
    "train_one_epoch",
]

#############################################
# 1.  SEED CONTROL & GLOBAL CONSTANTS       #
#############################################

SEED_LIST = [2, 12, 23, 34, 45]


def set_seed(seed: int):
    """Fix pseudo-random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#############################################
# 2.  LCAMP LAYER                           #
#############################################


class LCAMPLayer(MessagePassing):
    """Learnable Curvature-aware Adaptive Message Passing layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        gamma_init: float = 1.0,
        K: int = 8,
        shortcut: bool = True,
    ):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_short = (
            nn.Linear(in_channels, out_channels, bias=False) if shortcut else None
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.w_att = nn.Parameter(torch.zeros(1))
        self.K = K
        self.shortcut = shortcut

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_curvature(
        x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forman-style proxy of Ollivier–Ricci curvature:  κ_e = 1 − |h_i−h_j|₁ /(deg_i+deg_j)."""
        row, col = edge_index  # (E,), (E,)

        # Degree per node (N,)
        deg = degree(row, num_nodes=x.size(0), dtype=x.dtype)

        # L1 distance per edge (E,)
        diff = (x[row] - x[col]).abs().sum(dim=1)

        # Edge-level denominator: deg_i + deg_j  – align shapes by indexing
        denom = deg[row] + deg[col]
        kappa = 1.0 - diff / (denom + 1e-6)
        return kappa  # (E,)

    def message(self, x_j, x_i, index, ptr, size_i, kappa):  # pylint: disable=arguments-differ
        # Attention logits per edge
        logits = self.w_att - self.gamma * kappa
        alpha = torch.softmax(logits, dim=0)
        return alpha.view(-1, 1) * x_j

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # pylint: disable=arguments-differ
        x_proj = self.lin(x)
        kappa = self.estimate_curvature(x_proj, edge_index)
        out = self.propagate(edge_index, x=x_proj, kappa=kappa)

        # Optional curvature-guided top-K shortcut
        if self.shortcut:
            row, col = edge_index
            _, top_idx = torch.topk(-kappa, k=min(self.K, kappa.numel()))  # guard K>E
            shortcut_msg = self.lin_short(x[col[top_idx]])
            out.index_add_(0, row[top_idx], shortcut_msg)
        return out


#############################################
# 3.  BACKBONE BUILDING UTILS              #
#############################################

def build_backbone(
    name: str, num_layers: int, hidden: int, in_dim: int, out_dim: int
):
    layers = nn.ModuleList()
    name = name.lower()
    if name == "gcn2":  # GCNII
        for l in range(num_layers):
            layers.append(GCN2Conv(hidden, alpha=0.5, theta=1.0, layer=l + 1))
    elif name == "gat":
        for l in range(num_layers):
            in_c = in_dim if l == 0 else hidden
            layers.append(GATConv(in_c, hidden, heads=4, concat=False, dropout=0.5))
    elif name == "sage":
        for l in range(num_layers):
            in_c = in_dim if l == 0 else hidden
            layers.append(SAGEConv(in_c, hidden, aggr="mean"))
    else:
        raise ValueError(f"Unknown backbone '{name}'.")
    return layers


#############################################
# 4.  STACKED GNN MODEL                    #
#############################################


class GNNStack(nn.Module):
    """Wrap backbone or LCAMP-wrapped backbone into an end-to-end model."""

    def __init__(self, cfg: Dict[str, Any], data):
        super().__init__()
        # ------------------------------------------------------------------
        # Feature & class dimension detection for torch_geometric.data.Data
        # ------------------------------------------------------------------
        in_dim = data.x.size(1)
        out_dim = int(data.y.max().item() + 1)

        hidden = cfg["hidden"]
        self.dropout = nn.Dropout(cfg["dropout"])
        self.backbone_name = cfg["backbone"].lower()
        wrapper = cfg.get("wrapper", "none").lower()
        self.use_lcamp = wrapper == "lcamp"
        self.pairnorm = PairNorm(mode="PN-SI") if wrapper == "pairnorm" else None

        if self.use_lcamp:
            layers: list[nn.Module] = []
            in_c = in_dim
            for _ in range(cfg["depth"]):
                layers.append(
                    LCAMPLayer(in_c, hidden, gamma_init=1.0, K=cfg["K"])
                )
                in_c = hidden
            self.layers = nn.ModuleList(layers)
        else:
            self.layers = build_backbone(
                self.backbone_name, cfg["depth"], hidden, in_dim, out_dim
            )

        self.head = nn.Linear(hidden, out_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, edge_index):
        for layer in self.layers:
            # Support both MessagePassing-style (expects x, edge_index) and
            # GCN2Conv which expects (x, x0, edge_index).
            if isinstance(layer, GCN2Conv):
                x = layer(x, x, edge_index)
            else:
                x = layer(x, edge_index)
            if self.pairnorm is not None:
                x = self.pairnorm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.head(x)


#############################################
# 5.  TRAINING SUPPORT                     #
#############################################


@dataclass
class TrainState:
    epoch: int
    best_val: float
    best_epoch: int


def train_one_epoch(model: nn.Module, data, optimizer: optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y.squeeze()[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
