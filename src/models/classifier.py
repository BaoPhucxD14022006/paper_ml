"""Kiến trúc mạng phân loại phục vụ bài toán có giám sát (Mục 4.5, Bảng 4 trong bài báo)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Mạng Fully Connected gồm 2 tầng ẩn, mỗi tầng 400 nơ-ron (dùng cho Split MNIST)."""

    def __init__(self, in_dim: int = 784, hidden: int = 400, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    """Khối Residual cơ bản của ResNet với 2 tầng tích chập và kết nối tắt (shortcut)."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ReducedResNet18(nn.Module):
    """Phiên bản ResNet-18 thu nhỏ (nf=20 bộ lọc thay vì 64) thường dùng cho
    bài toán học liên tục trực tuyến trên Split CIFAR-10 để giảm chi phí tính toán.
    """

    def __init__(self, n_classes: int = 10, nf: int = 20, in_channels: int = 3):
        super().__init__()
        self.in_planes = nf
        self.conv1 = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(nf, 2, 1)
        self.layer2 = self._make_layer(nf * 2, 2, 2)
        self.layer3 = self._make_layer(nf * 4, 2, 2)
        self.layer4 = self._make_layer(nf * 8, 2, 2)
        self.linear = nn.Linear(nf * 8, n_classes)

    def _make_layer(self, planes, n_blocks, stride):
        layers = []
        for s in [stride] + [1] * (n_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def build_classifier(name: str, n_classes: int, **kw) -> nn.Module:
    """Hàm khởi tạo mạng phân loại từ tên cấu hình."""
    if name == "mlp":
        return MLP(n_classes=n_classes, **kw)
    if name == "reduced_resnet18":
        return ReducedResNet18(n_classes=n_classes, **kw)
    raise ValueError(f"Không nhận diện được mô hình phân loại: {name}")
