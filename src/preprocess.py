# preprocess.py
"""Dataset download, preprocessing and continual split helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

# --- torchvision transforms -------------------------------------------------

cifar_aug = T.Compose([
    T.Pad(4),
    T.RandomCrop(32),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

cifar_test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def _tiny_aug(size):
    return T.Compose([
        T.RandomResizedCrop(size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.25, hue=0.1),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def _tiny_test_tf(size):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# -----------------------------------------------------------------------------

class _TorchWrapper(torch.utils.data.Dataset):
    """Lazy transform wrapper around hf datasets."""

    def __init__(self, ds, transform=None, img_key="img", label_key="fine_label"):
        self._ds = ds
        self.transform = transform
        self.img_key = img_key
        self.label_key = label_key

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        rec = self._ds[idx]
        img = rec[self.img_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(rec[self.label_key])


class ContinualSplit:
    """Generate sequential class-incremental tasks."""

    def __init__(
        self,
        dataset_name: str,
        hf_repo: str,
        n_tasks: int,
        classes_per_task: int,
        img_size: int,
        data_root: str,
        split: str = "train",
    ):
        self.name = dataset_name
        local_dir = Path(data_root) / dataset_name.replace("/", "_")
        local_dir.mkdir(parents=True, exist_ok=True)
        self.raw = load_dataset(hf_repo, split=split, cache_dir=str(local_dir))

        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
        self.tasks: List[List[int]] = []
        labels = [int(x.get("fine_label", x.get("label"))) for x in self.raw]

        # build index list for each task
        order: List[int] = []
        for t in range(n_tasks):
            cls_ids = list(range(t * classes_per_task, (t + 1) * classes_per_task))
            idxs = [i for i, y in enumerate(labels) if y in cls_ids]
            self.tasks.append(idxs)
            order.extend(idxs)
        self.raw = self.raw.select(order)  # reorder to stream order

        self.transform = cifar_aug if img_size == 32 else _tiny_aug(img_size)
        self.test_transform = cifar_test_tf if img_size == 32 else _tiny_test_tf(img_size)

    # ---------------------------------------------------------------------
    def get_task_loader(self, task_id: int, batch_size: int, num_workers: int):
        subset = torch.utils.data.Subset(_TorchWrapper(self.raw, transform=self.transform), self.tasks[task_id])
        return DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    def get_test_loader(self, batch_size: int, num_workers: int):
        split = "test" if "cifar" in self.name.lower() else "validation"
        test_ds = load_dataset(self.raw.builder_name, split=split, cache_dir=self.raw.cache_dir)
        test_ds = _TorchWrapper(test_ds, transform=self.test_transform)
        return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
