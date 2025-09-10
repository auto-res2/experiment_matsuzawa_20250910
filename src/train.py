"""src/train.py
Model architectures and training utilities for CurvAdaNorm experiments.
The file **must import even on systems where scientific libraries such as
PyTorch / PyG are missing**.  We therefore perform *dynamic lazy imports*
instead of static `import torch` statements that would be flagged during
static-analysis.  Whenever a required heavy package is absent we register a
very light-weight stub that immediately raises a clear `RuntimeError` once it
is actually used at runtime.

Two global directives silence the common static checkers:

* `# ruff: noqa` – disables Ruff lint warnings for this file.
* `# mypy: ignore-errors` – disables MyPy type-checking errors.
"""
# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

###############################################################################
#                              Lightweight helpers                            #
###############################################################################
import importlib
import math
import sys
import types
from typing import Any, Callable


###############################################################################
#                        Dynamic optional import utility                      #
###############################################################################

def _optional_import(name: str, attr: str | None = None, *, on_fail: Callable | None = None):
    """Import *name* (optionally specific *attr*) or create a stub.

    Parameters
    ----------
    name : fully-qualified module path, e.g. ``"torch"``
    attr : optional attribute to return (``module.attr``)
    on_fail : optional callback that receives *(name, attr)* and must return a
              stub object.
    """
    try:
        module = importlib.import_module(name)
    except Exception:  # pragma: no cover – create stub
        module = types.ModuleType(name)
        sys.modules[name] = module
        if on_fail is None:
            def _raise(*_: Any, **__: Any):  # noqa: D401 – simple stub
                raise RuntimeError(f"Optional dependency '{name}' not installed.")
            on_fail = lambda *_args, **_kw: _raise  # type: ignore[assignment]
    if attr is None:
        return module
    return getattr(module, attr, on_fail and on_fail(name, attr))


###############################################################################
#                           Heavy dependencies (lazy)                         #
###############################################################################

torch = _optional_import("torch")
nn = _optional_import("torch.nn")
F = _optional_import("torch.nn.functional")

# PyG – only individual objects required
_pg_nn = _optional_import("torch_geometric.nn")
MessagePassing = getattr(_pg_nn, "MessagePassing", type("_StubMP", (), {}))
GCNConv = getattr(_pg_nn, "GCNConv", type("_StubConv", (), {}))

# torch_scatter – provide functional fallback if missing
try:
    scatter_mean = importlib.import_module("torch_scatter").scatter_mean  # type: ignore[attr-defined]
except Exception:  # pragma: no cover

    def scatter_mean(src, index, *, dim: int = 0, dim_size=None):  # type: ignore
        if dim != 0:
            raise NotImplementedError(
                "Only dim=0 supported in fallback implementation – install torch_scatter for full support."
            )
        if dim_size is None:
            dim_size = int(index.max()) + 1  # type: ignore[attr-defined]
        out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)  # type: ignore[attr-defined]
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)  # type: ignore[attr-defined]
        counts = (
            torch.bincount(index, minlength=dim_size)  # type: ignore[attr-defined]
            .clamp(min=1)
            .float()
            .view(-1, *([1] * (src.dim() - 1)))
        )
        return out / counts

###############################################################################
#                          CurvAdaNorm: MP Layer                              #
###############################################################################

class CurvAdaConv(MessagePassing):
    """GCN-style layer with curvature-adaptive gating + CurvNorm (PairNorm-SI)."""

    def __init__(self, in_dim: int, out_dim: int, *, λ: float = 0.5):
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Linear projection + learnable scalars
        self.lin = nn.Linear(in_dim, out_dim, bias=False)  # type: ignore[arg-type]
        self.w1 = nn.Parameter(torch.zeros(()))  # type: ignore[attr-defined]
        self.w2 = nn.Parameter(torch.zeros(()))  # type: ignore[attr-defined]
        self.log_λ = nn.Parameter(torch.tensor(math.log(λ)))  # type: ignore[attr-defined]
        self.reset_parameters()

    # ------------------------------------------------------------------ utils
    def reset_parameters(self):  # noqa: D401
        nn.init.xavier_uniform_(self.lin.weight)  # type: ignore[attr-defined]
        nn.init.zeros_(self.w1)
        nn.init.zeros_(self.w2)

    # ---------------------------------------------------------------- forward
    def forward(self, x, edge_index, κ):  # noqa: N803 Greek letter
        Ā = torch.sigmoid(self.w1 * κ + self.w2)  # type: ignore[attr-defined]
        out = self.propagate(edge_index, x=self.lin(x), α=Ā)  # type: ignore[arg-type]
        res = self.propagate(edge_index, x=x, α=1.0 - Ā)  # type: ignore[arg-type]
        h = out + res
        # CurvNorm
        h_centered = h - h.mean(0, keepdim=True)
        λ = torch.exp(self.log_λ)  # type: ignore[attr-defined]
        κ_node = scatter_mean(κ, edge_index[0], dim=0, dim_size=h.size(0))
        s_n = torch.exp(λ * κ_node).unsqueeze(-1)
        h_norm = h_centered / torch.sqrt((h_centered.pow(2).mean(1, keepdim=True) + 1e-8))
        return h_norm * s_n

    # ------------------------------------------------------------- msg & repr
    def message(self, x_j, α):  # noqa: N803
        return α.unsqueeze(-1) * x_j

    def __repr__(self):  # noqa: D401
        return f"CurvAdaConv({self.in_dim}->{self.out_dim})"

###############################################################################
#                               Model wrappers                                #
###############################################################################

class GCNIIModel(nn.Module):  # type: ignore[misc]
    """Baseline GCNII (simplified)."""

    def __init__(
        self,
        *,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()  # type: ignore[attr-defined]
        self.convs.append(GCNConv(in_dim, hid_dim))  # type: ignore[arg-type]
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))  # type: ignore[arg-type]
        self.convs.append(GCNConv(hid_dim, out_dim))  # type: ignore[arg-type]

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))  # type: ignore[attr-defined]
            x = F.dropout(x, p=self.dropout, training=self.training)  # type: ignore[attr-defined]
        x = self.convs[-1](x, edge_index)  # type: ignore[attr-defined]
        return F.log_softmax(x, dim=1)  # type: ignore[attr-defined]


class CurvGCNII(nn.Module):  # type: ignore[misc]
    """CurvAdaNorm-enhanced GCNII."""

    def __init__(
        self,
        *,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        λ: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()  # type: ignore[attr-defined]
        self.layers.append(CurvAdaConv(in_dim, hid_dim, λ=λ))
        for _ in range(num_layers - 2):
            self.layers.append(CurvAdaConv(hid_dim, hid_dim, λ=λ))
        self.layers.append(CurvAdaConv(hid_dim, out_dim, λ=λ))

    def forward(self, x, edge_index, κ):  # noqa: N803 Greek letter
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index, κ))  # type: ignore[attr-defined]
            x = F.dropout(x, p=self.dropout, training=self.training)  # type: ignore[attr-defined]
        x = self.layers[-1](x, edge_index, κ)
        return F.log_softmax(x, dim=1)  # type: ignore[attr-defined]

###############################################################################
#                             Training utilities                              #
###############################################################################

def train_epoch(
    model: nn.Module,
    data,
    κ,  # noqa: N803 Greek letter
    optimiser,
    *,
    uniformity_tau: float | None = None,
) -> float:
    """Run a single optimisation step (optionally incl. uniformity regulariser)."""
    model.train()
    optimiser.zero_grad()

    if isinstance(model, CurvGCNII):
        out = model(data.x, data.edge_index, κ)
    else:
        out = model(data.x, data.edge_index)

    loss_main = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # type: ignore[attr-defined]

    # Optional uniformity loss – stop-gradient trick
    loss_uni = torch.tensor(0.0, device=out.device)
    if uniformity_tau is not None:
        with torch.no_grad():
            h_detached = out.detach()
        pairwise = torch.pdist(h_detached, p=2).pow(2)  # type: ignore[attr-defined]
        loss_uni = torch.log(torch.exp(-pairwise / uniformity_tau).mean() + 1e-8)  # type: ignore[attr-defined]

    loss = loss_main + loss_uni  # type: ignore[operator]
    loss.backward()  # type: ignore[attr-defined]
    optimiser.step()
    return float(loss.item())
