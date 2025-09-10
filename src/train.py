# train.py
"""Model definitions, training utilities and memory-budget enforcement (episodic only).

IMPORTANT: We *do not* enforce the budget on the whole resident process
memory. Instead, we follow the CL-literature convention and measure only the
extra episodic memory used to store rehearsal/experience buffers (e.g.
SketchMemory in SKETCH-CL). This makes the provided budgets realistic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

############################################################
# Budget helpers
############################################################

class MemoryBudgetExceeded(RuntimeError):
    """Raised when the episodic memory size is larger than the user budget."""


def _tensor_bytes(t: torch.Tensor | None) -> int:
    if t is None:
        return 0
    return t.numel() * t.element_size()


def episodic_memory_bytes(model: nn.Module) -> int:
    """Return the bytes occupied by *episodic* memory buffers (not model params)."""
    bytes_used = 0
    # The only episodic memory in the current code base lives inside `SketchMemory`.
    for m in model.modules():
        if isinstance(m, SketchMemory):
            bytes_used += _tensor_bytes(getattr(m, "S", None))
            bytes_used += _tensor_bytes(getattr(m, "N", None))
    return bytes_used


def enforce_budget(used_bytes: int, max_bytes: int):
    """Abort if *used_bytes* is greater than *max_bytes* (fail-fast)."""
    if used_bytes > max_bytes:
        raise MemoryBudgetExceeded(
            f"Memory budget {max_bytes/1e6:.2f} MB exceeded: {used_bytes/1e6:.2f} MB"
        )

############################################################
#  Count-Sketch based SKETCH-CL components
############################################################

class CountSketch(nn.Module):
    """Count-Sketch projection C∈{−1,0,1}^{k×d}."""

    def __init__(self, in_dim: int, k: int):
        super().__init__()
        idx = torch.randint(0, k, (in_dim,))
        sgn = torch.randint(0, 2, (in_dim,)) * 2 - 1  # ±1
        self.register_buffer("idx", idx, persistent=False)
        # store as float for later multiplication, but we will cast on the fly
        self.register_buffer("sgn", sgn.float(), persistent=False)
        self.k = k
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor):  # x:(B,D)
        # Ensure dtype consistency to avoid `scatter` runtime errors under AMP
        sgn = self.sgn.to(dtype=x.dtype)
        res = torch.zeros(x.size(0), self.k, device=x.device, dtype=x.dtype)
        idx_exp = self.idx.expand(x.size(0), -1)
        res.scatter_add_(1, idx_exp, x * sgn)
        return res


class SketchMemory(nn.Module):
    """Running sum & count per class occupying O(k·C) memory."""

    def __init__(self, k: int, n_classes: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        # Register as buffers so that `.to(device)` moves them together with the model
        self.register_buffer("S", torch.zeros(n_classes, k, dtype=dtype), persistent=False)
        self.register_buffer("N", torch.zeros(n_classes, dtype=torch.int32), persistent=False)
        self.k = k
        self.n_classes = n_classes

    @torch.no_grad()
    def update(self, cs: torch.Tensor, labels: torch.Tensor):
        # Cast to match memory dtype to avoid dtype mismatch when accumulating
        cs = cs.to(self.S.dtype)
        for c in range(self.n_classes):
            mask = labels == c
            if mask.any():
                self.S[c] += cs[mask].sum(0)
                self.N[c] += int(mask.sum())

    def get_mean(self, c: torch.Tensor):
        return self.S[c] / torch.clamp(self.N[c].unsqueeze(-1), min=1)


class Decoder(nn.Module):
    def __init__(self, k: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SparseLinear(nn.Linear):
    """Linear layer with a dynamic binary pruning mask (RigL-style)."""

    def __init__(self, *args, sparsity: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.register_buffer("mask", torch.ones_like(self.weight, dtype=torch.bool), persistent=False)
        self._apply_prune()

    def _apply_prune(self):
        k = int(self.weight.numel() * self.sparsity)
        _, idx = torch.topk(self.weight.abs().flatten(), k=k, largest=False)
        new_mask = torch.ones_like(self.weight, dtype=torch.bool).flatten()
        new_mask[idx] = 0
        self.mask = new_mask.view_as(self.weight)
        # Ensure pruned parameters are zeroed *and* stay zero during optimisation
        self.weight.data *= self.mask

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class SketchCLNet(nn.Module):
    """Backbone + sketch memory + sparse classifier."""

    def __init__(self, backbone_name: str, num_classes: int, k: int, sparsity: float):
        super().__init__()

        # ------------------------------------------------------------------
        # Use timm-native models exclusively to avoid HF YAML resolution bugs
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.count_sketch = CountSketch(feat_dim, k)
        self.memory = SketchMemory(k, num_classes)
        self.decoder = Decoder(k, feat_dim)
        linear_cls: nn.Module
        if sparsity < 1:
            linear_cls = SparseLinear(feat_dim, num_classes, sparsity=sparsity)
        else:
            linear_cls = nn.Linear(feat_dim, num_classes)
        self.classifier = linear_cls

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f)

    def online_step(self, x, y):
        f = self.backbone(x)
        cs = self.count_sketch(f)
        self.memory.update(cs.detach(), y.detach())
        # replay
        rand_cls = torch.randint(0, self.classifier.out_features, (x.size(0),), device=x.device)
        replay_cs = self.memory.get_mean(rand_cls)
        f_hat = self.decoder(replay_cs)
        logits_replay = self.classifier(f_hat.detach())
        logits_cur = self.classifier(f)

        loss_cls = F.cross_entropy(logits_cur, y)
        loss_dec = F.mse_loss(self.count_sketch(f_hat), replay_cs)
        loss_replay = F.cross_entropy(
            logits_replay, torch.randint_like(y, 0, self.classifier.out_features)
        )
        return loss_cls + 0.1 * loss_dec + 0.2 * loss_replay

############################################################
#  Baseline wrapper – provides a minimal `online_step` for plain models
############################################################

class BaselineWrapper(nn.Module):
    """Wrap standard classification models so they can be trained with `train_one_task`."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):  # noqa: D401 – simple forward pass
        return self.model(x)

    def online_step(self, x, y):
        """Single forward/backward step using standard cross entropy loss."""
        logits = self.model(x)
        return F.cross_entropy(logits, y)

############################################################
#  Model factory & training loop
############################################################


def build_model(method: str, num_classes: int, k: int, sparsity: float):
    """Factory returning the model associated with *method*."""
    if method == "sketch_cl":
        return SketchCLNet("resnet18", num_classes, k, sparsity)

    # Fallback baseline – wrap the backbone so that `online_step` is available.
    backbone = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    return BaselineWrapper(backbone)


def train_one_task(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optim_cfg: Dict[str, Any],
    device: torch.device,
    budget: int | None = None,
):
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg.get("weight_decay", 0.0)
    )
    scaler = GradScaler(enabled=(optim_cfg.get("precision", "fp16") == "fp16" and device.type == "cuda"))

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=scaler.is_enabled()):
            loss = model.online_step(images, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if budget is not None:
            used = episodic_memory_bytes(model)
            enforce_budget(used, budget)
