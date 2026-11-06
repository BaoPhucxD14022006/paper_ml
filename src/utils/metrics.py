"""Độ đo đánh giá thực nghiệm: Độ chính xác phân loại (Accuracy) và điểm số FID."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


@torch.no_grad()
def accuracy(model: nn.Module, batches: Iterable, device="cpu") -> float:
    """Tính tỉ lệ đoán đúng (Accuracy) của mô hình trên tập kiểm thử."""
    model.eval()
    correct = total = 0
    for x, y in batches:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += y.numel()
    model.train()
    return correct / max(total, 1)


class InceptionFeatures(nn.Module):
    """Mạng trích xuất đặc trưng Inception-v3 (chuẩn hóa trên ImageNet, vector 2048 chiều)."""

    def __init__(self, device="cpu"):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Đầu vào x là ảnh trong đoạn [-1, 1], kích thước (N, C, H, W)."""
        x = x.to(self.device)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def feature_stats(extractor: InceptionFeatures, images: torch.Tensor, batch_size: int = 100):
    """Tính vector kỳ vọng trung bình và ma trận hiệp phương sai của tập ảnh."""
    feats = [extractor(images[i:i + batch_size]).cpu().double() for i in range(0, len(images), batch_size)]
    f = torch.cat(feats).numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps: float = 1e-6) -> float:
    """Công thức khoảng cách Frechet giữa 2 phân phối Gaussian đa chiều:
    FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 * sqrt(S1 * S2)).
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(extractor: InceptionFeatures, real: torch.Tensor, fake: torch.Tensor) -> float:
    """Tính chỉ số FID giữa tập ảnh thật và tập ảnh sinh ra bởi mô hình."""
    m1, s1 = feature_stats(extractor, real)
    m2, s2 = feature_stats(extractor, fake)
    return frechet_distance(m1, s1, m2, s2)
