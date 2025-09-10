# evaluate.py
"""Evaluation & visualisation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def plot_accuracy_curve(acc: List[float], save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(acc)), acc, marker="o")
    for i, v in enumerate(acc):
        plt.text(i, v + 0.5, f"{v:.1f}", fontsize=6, ha="center")
    plt.xlabel("Task ID")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
