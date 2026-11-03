"""Đo lường sự khác biệt tri thức (Knowledge Discrepancy Measures - KDM).
Triển khai Mục 3.2 trong bài báo: F_KDM(x_g, x_h) = F_d(G_t(x_g), G_t(x_h)).

Bao gồm 2 biến thể:
1. SquaredErrorKDM  - Phương trình (7): Đo khoảng cách Euclidean bình phương L2.
2. JensenShannonKDM - Phương trình (8): Đo phân kỳ Jensen-Shannon giữa hai phân phối
   Gaussian sau khi khuếch tán thuận tại bước t* = 100.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch

LOG2 = math.log(2.0)


def _flat(x: torch.Tensor) -> torch.Tensor:
    """Làm phẳng tensor ảnh về dạng ma trận 2D (N, D)."""
    return x.reshape(x.shape[0], -1).to(torch.float32)


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tính ma trận khoảng cách bình phương L2 giữa mọi cặp mẫu (Eq. 7 vector hóa).
    Công thức: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y^T.
    Dùng float64 để tránh lỗi số âm do sai số làm tròn số thực.
    """
    x, y = _flat(x).double(), _flat(y).double()
    d = (x * x).sum(1, keepdim=True) + (y * y).sum(1)[None, :] - 2.0 * x @ y.T
    return d.clamp_min_(0.0).float()


def linear_alpha_bar(t_star: int, T: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02) -> float:
    """Tính hệ số alpha_bar tại bước t_star theo lịch trình tuyến tính của DDPM."""
    betas = np.linspace(beta_1, beta_T, T, dtype=np.float64)
    return float(np.prod(1.0 - betas[:t_star]))


class KDM(ABC):
    """Giao diện chung cho các độ đo KDM: nhận vào x và y, trả về ma trận khoảng cách."""

    name: str = "kdm"

    @abstractmethod
    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Trả về ma trận khoảng cách giữa x[i] và y[j] có kích thước (n, m)."""

    def __call__(self, x_g: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
        """Hàm tiện ích đo khoảng cách giữa 2 mẫu đơn lẻ."""
        return self.pairwise(x_g[None], x_h[None])[0, 0]


class SquaredErrorKDM(KDM):
    """DCM-SE (Phương trình 7): khoảng cách bình phương Euclidean trong không gian ảnh."""

    name = "se"

    def pairwise(self, x, y):
        return pairwise_sq_dist(x, y)


class JensenShannonKDM(KDM):
    """DCM-JS (Phương trình 8): phân kỳ Jensen-Shannon giữa 2 phân phối Gaussian.
    
    Khi thêm nhiễu tại bước t*, 2 ảnh x_g và x_h trở thành 2 phân phối Gaussian.
    Vì cùng phương sai đẳng hướng, phân kỳ JS quy về bài toán tích phân 1 chiều
    dọc theo đường nối hai tâm kỳ vọng.
    """

    name = "js"

    def __init__(self, t_star: int = 100, alpha_bar: float | None = None,
                 mode: str = "exact", n_quad: int = 80, T: int = 1000):
        if mode not in ("exact", "dimwise"):
            raise ValueError(f"Không hỗ trợ JS mode: {mode}")
        self.t_star = t_star
        self.alpha_bar = linear_alpha_bar(t_star, T) if alpha_bar is None else alpha_bar
        self.mode = mode
        # Dùng phương pháp cầu phương Gauss-Hermite để tính tích phân 1D chính xác
        z, w = np.polynomial.hermite_e.hermegauss(n_quad)
        self._z = torch.tensor(z, dtype=torch.float64)
        self._w = torch.tensor(w / math.sqrt(2.0 * math.pi), dtype=torch.float64)

    @property
    def scale(self) -> float:
        """Hệ số tỉ lệ: delta = scale * ||x_g - x_h||."""
        return math.sqrt(self.alpha_bar / (1.0 - self.alpha_bar))

    def js_1d(self, delta: torch.Tensor) -> torch.Tensor:
        """Tính phân kỳ JS 1 chiều giữa N(0, 1) và N(delta, 1)."""
        d = delta.to(torch.float64).unsqueeze(-1)
        z = self._z.to(d.device)
        w = self._w.to(d.device)
        integrand = LOG2 - torch.nn.functional.softplus(d * z - 0.5 * d * d)
        return (integrand * w).sum(-1).clamp(0.0, LOG2)

    def pairwise(self, x, y):
        if self.mode == "exact":
            # Chế độ exact: tính trực tiếp theo công thức chuẩn
            delta = self.scale * pairwise_sq_dist(x, y).to(torch.float64).sqrt()
            return self.js_1d(delta).to(torch.float32)

        # Chế độ dimwise: trung bình JS trên từng chiều ảnh (tránh bão hòa ln 2 ở số chiều cao)
        # Sử dụng bảng tra trước (LUT) để tính toán nhanh hơn
        if not hasattr(self, "_lut"):
            self._lut_max = 40.0
            grid = torch.linspace(0.0, self._lut_max, 8193, dtype=torch.float64)
            self._lut = self.js_1d(grid).to(torch.float32)

        xf, yf = _flat(x), _flat(y)
        lut = self._lut.to(xf.device)
        n_bins = lut.numel() - 1
        out = torch.empty(xf.shape[0], yf.shape[0], dtype=torch.float32, device=xf.device)
        chunk = max(1, 8_000_000 // (yf.shape[0] * xf.shape[1] + 1))
        for s in range(0, xf.shape[0], chunk):
            delta = (xf[s:s + chunk, None, :] - yf[None, :, :]).abs() * self.scale
            pos = (delta / self._lut_max * n_bins).clamp(max=n_bins - 1e-4)
            lo = pos.floor().long()
            frac = pos - lo
            val = lut[lo] * (1 - frac) + lut[lo + 1] * frac
            out[s:s + chunk] = val.mean(-1)
        return out


def build_kdm(name: str, **kwargs) -> KDM:
    """Hàm khởi tạo thước đo KDM theo tên cấu hình (se hoặc js)."""
    name = name.lower()
    if name == "se":
        return SquaredErrorKDM()
    if name == "js":
        return JensenShannonKDM(**kwargs)
    raise ValueError(f"Không tìm thấy KDM: {name}")
