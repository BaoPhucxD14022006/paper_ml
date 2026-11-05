"""Tải dữ liệu thô và tiền xử lý ảnh theo từng batch.

Ảnh được giữ ở định dạng uint8 và độ phân giải gốc để tiết kiệm RAM,
chỉ chuyển sang float khi có yêu cầu lấy batch trong quá trình train.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

N_CLASSES = {"mnist": 10, "fashion": 10, "svhn": 10, "cifar10": 10, "cifar100": 100}


@dataclass
class RawDataset:
    images: torch.Tensor  # uint8, kích thước (N, C, H, W)
    labels: torch.Tensor  # int64, kích thước (N,)
    name: str

    def __len__(self):
        return self.labels.shape[0]


def load_raw(name: str, train: bool, root: str = "./data") -> RawDataset:
    """Tải (nếu chưa có) và trả về dữ liệu dưới dạng tensor uint8."""
    import torchvision.datasets as tvd

    name = name.lower()
    if name in ("mnist", "fashion"):
        cls = tvd.MNIST if name == "mnist" else tvd.FashionMNIST
        ds = cls(root, train=train, download=True)
        imgs = ds.data.unsqueeze(1)  # Kích thước (N, 1, 28, 28)
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
    elif name in ("cifar10", "cifar100"):
        cls = tvd.CIFAR10 if name == "cifar10" else tvd.CIFAR100
        ds = cls(root, train=train, download=True)
        imgs = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2)
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
    elif name == "svhn":
        ds = tvd.SVHN(root, split="train" if train else "test", download=True)
        imgs = torch.from_numpy(ds.data)
        labels = torch.as_tensor(ds.labels, dtype=torch.long)
    else:
        raise ValueError(f"Không hỗ trợ tập dữ liệu: {name}")
    return RawDataset(imgs.contiguous(), labels, name)


class Preprocess:
    """Chuyển tensor uint8 (N, C, H, W) thành float trong khoảng [0, 1] hoặc [-1, 1]."""

    def __init__(self, image_size: int | None = None, channels: int | None = None,
                 value_range: str = "01", flatten: bool = False):
        assert value_range in ("01", "pm1")
        self.image_size = image_size
        self.channels = channels
        self.value_range = value_range
        self.flatten = flatten

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div_(255.0)
        if self.channels is not None and x.shape[1] != self.channels:
            if self.channels == 3 and x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            elif self.channels == 1 and x.shape[1] == 3:
                x = x.mean(1, keepdim=True)
            else:
                raise ValueError("Không hỗ trợ chuyển đổi số kênh này")
        if self.image_size is not None and x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False, antialias=True)
        if self.value_range == "pm1":
            x = x * 2.0 - 1.0
        x = x.contiguous()
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        return x
