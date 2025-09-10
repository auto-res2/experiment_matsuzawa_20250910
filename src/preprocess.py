"""src/preprocess.py
Utility helpers for reproducibility, file-system handling and curvature
computation.  Heavy scientific dependencies are imported lazily so the module
is importable on very lean systems.
"""
# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

###############################################################################
#                               Lazy imports                                 #
###############################################################################
import importlib
import os
import pathlib
import random
import sys
import types
from typing import Any


def _lazy(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

np = _lazy("numpy")
torch = _lazy("torch")

###############################################################################
#                               Reproducibility                               #
###############################################################################
SEED_GLOBAL = 42


def set_seed(seed: int = SEED_GLOBAL) -> None:
    random.seed(seed)
    np.random.seed(seed)  # type: ignore[attr-defined]
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)  # type: ignore[attr-defined]
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

###############################################################################
#                             File-system helpers                             #
###############################################################################

def ensure_dir(p: str | os.PathLike) -> pathlib.Path:
    path = pathlib.Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

###############################################################################
#                       Network resource availability                         #
###############################################################################

def ensure_availability(url: str, *, local_hint: str | None = None) -> None:
    import requests

    try:
        resp = requests.head(url, timeout=10)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Resource not reachable: {url} (HTTP {resp.status_code})"
            )
    except Exception as exc:  # pragma: no cover
        msg = f"Could not reach required resource {url}: {exc}"
        if local_hint:
            msg += f". Local fallback attempted at {local_hint}"
        raise RuntimeError(msg) from exc

###############################################################################
#              Approximate Ollivier–Ricci curvature (fast)                    #
###############################################################################

def compute_ollivier_ricci(edge_index: "torch.Tensor", num_nodes: int, *, sinkhorn_iter: int = 1):  # type: ignore
    nx = _lazy("networkx")
    try:
        OllivierRicci = importlib.import_module("GraphRicciCurvature.OllivierRicci").OllivierRicci  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pip install GraphRicciCurvature==1.6") from exc

    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(num_nodes))
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if u <= v:
            g_nx.add_edge(u, v)

    orc = OllivierRicci(
        g_nx,
        alpha=0.5,
        method="OTD",
        approximation="sinkhorn",
        sinkhorn_iter=sinkhorn_iter,
        verbose="ERROR",
    )
    orc.compute_ricci_curvature()

    curv: dict[tuple[int, int], float] = {}
    for (u, v), data in orc.G.edges.items():  # type: ignore[attr-defined]
        curv[(u, v)] = data["ricciCurvature"]

    kappa_vals: list[float] = []
    for u, v in zip(src.tolist(), dst.tolist()):
        kappa_vals.append(curv.get((u, v), curv.get((v, u), 0.0)))

    κ = torch.tensor(kappa_vals, dtype=torch.float32)  # noqa: N803 Greek letter
    κ = torch.tanh(κ)
    return κ
