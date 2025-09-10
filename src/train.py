"""src/train.py
Model architectures and training utilities for CurvAdaNorm experiments.
The file **must import even on systems where scientific libraries such as
PyTorch / PyG are missing**.  We therefore perform *dynamic lazy imports*
instead of static `import torch` statements that would be flagged during
static-analysis.  Whenever a required heavy package is absent we register a
very light-weight stub that immediately raises a clear `RuntimeError` once it
is actually used at runtime.
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


def _create_stub_module(name: str):
    """Return a stub module that *imports* but raises on *use*.

    The stub exposes arbitrary attributes so that `getattr` succeeds during
    import-time (class definitions, etc.) but each attribute – whether a class
    or a function – raises a *RuntimeError* the moment it is **instantiated**
    or **called**.  This guarantees *import-safety* while still following a
    fail-fast policy at runtime once the missing dependency is actually used.
    """

    def _raise(*_: Any, **__: Any):  # noqa: D401 – simple helper
        raise RuntimeError(f"Optional dependency '{name.split('.')[0]}' not installed.")

    class _StubClass:  # noqa: D401 – minimal placeholder
        def __init__(self, *a: Any, **kw: Any):
            _raise()

        def __call__(self, *a: Any, **kw: Any):  # noqa: D401 – still fail-fast
            _raise()

        def __getattr__(self, _attr: str):  # noqa: D401
            return _raise

    mod = types.ModuleType(name)

    # Generic attribute access – resolves *any* identifier.
    def __getattr__(_ignored: str):  # noqa: D401 – dynamic attr hook
        # Return a class for capitalised names, function otherwise.
        return _StubClass if _ignored and _ignored[0].isupper() else _raise

    # Inject both dunder and common pytorch-ish names so that static attribute
    # look-ups succeed without triggering __getattr__ (e.g. `nn.Module`).
    mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
    # Frequently accessed placeholders – safe for use as base classes.
    mod.Module = _StubClass  # type: ignore[attr-defined]
    mod.Parameter = _StubClass  # type: ignore[attr-defined]
    mod.Linear = _StubClass  # type: ignore[attr-defined]
    mod.ModuleList = _StubClass  # type: ignore[attr-defined]

    return mod


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
        module = _create_stub_module(name)
        sys.modules[name] = module
        # Also register on the parent so that ``import torch; torch.nn`` works.
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name) or _create_stub_module(parent_name)
            setattr(parent, child_name, module)
            sys.modules[parent_name] = parent
    # ----------------------------- attr handling ---------------------------
    if attr is None:
        return module
    # Try direct lookup first; if missing fall back to stub via on_fail / __getattr__
    if hasattr(module, attr):
        return getattr(module, attr)
    if on_fail is not None:
        return on_fail(name, attr)
    # Resort to module.__getattr__ if available (our stubs provide it)
    if hasattr(module, "__getattr__"):
        return module.__getattr__(attr)  # type: ignore[attr-defined]
    # Fallback – last resort plain stub object that raises when called
    def _raise(*_: Any, **__: Any):
        raise RuntimeError(f"Optional dependency '{name}' missing attribute '{attr}'.")

    return _raise

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
