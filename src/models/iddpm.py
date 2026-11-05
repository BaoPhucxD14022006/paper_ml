"""Mô hình khuếch tán Improved DDPM (Nichol & Dhariwal, 2021).
Triển khai các Phương trình (1) đến (5) trong bài báo:

- Quá trình khuếch tán thuận q(x_t | x_{t-1})                   (Eq. 1)
- Phân phối khử nhiễu ngược p(x_{t-1} | x_t)                     (Eq. 2)
- Hàm mất mát kết hợp L_hybrid = L_MSE + lambda * L_vlb          (Eq. 3)
- Cận biến phân L_vlb dựa trên phân kỳ KL                        (Eq. 4)
- Phân phối hậu nghiệm q(x_{t-1} | x_t, x_0)                     (Eq. 5)

Mô hình học nội suy phương sai qua tham số v và hỗ trợ sinh ảnh nhanh (respacing).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def make_betas(schedule: str, T: int) -> np.ndarray:
    """Tạo lịch trình thêm nhiễu beta_1, ..., beta_T theo dạng linear hoặc cosine."""
    if schedule == "linear":
        scale = 1000.0 / T
        return np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)
    if schedule == "cosine":
        s = 0.008
        f = lambda t: math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        return np.array([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=np.float64)
    raise ValueError(f"Không hỗ trợ schedule: {schedule}")


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Tính phân kỳ KL giữa 2 phân phối Gaussian một chiều N(mean1, var1) và N(mean2, var2)."""
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def _std_normal_cdf(x):
    """Hàm phân phối tích lũy CDF của phân phối chuẩn tắc N(0, 1)."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def discretized_gaussian_loglik(x, mean, log_scale):
    """Tính log-likelihood của ảnh số nguyên 8-bit rời rạc trong đoạn [-1, 1] (bước t=0)."""
    centered = x - mean
    inv_std = torch.exp(-log_scale)
    cdf_plus = _std_normal_cdf(inv_std * (centered + 1.0 / 255.0))
    cdf_min = _std_normal_cdf(inv_std * (centered - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(x < -0.999, log_cdf_plus,
                       torch.where(x > 0.999, log_one_minus_cdf_min,
                                   torch.log(cdf_delta.clamp(min=1e-12))))


def _mean_flat(x):
    """Tính trung bình trên tất cả các chiều ngoại trừ chiều batch đầu tiên."""
    return x.mean(dim=tuple(range(1, x.ndim)))


class GaussianDiffusion:
    def __init__(self, betas: np.ndarray, vlb_weight: float = 1e-3,
                 timestep_map: Optional[Sequence[int]] = None):
        self.betas = betas = np.asarray(betas, dtype=np.float64)
        self.T = len(betas)
        self.vlb_weight = vlb_weight
        self.timestep_map = torch.tensor(timestep_map if timestep_map is not None else range(self.T))

        # Tính toán các hằng số tích lũy của quá trình khuếch tán
        alphas = 1.0 - betas
        self.abar = np.cumprod(alphas)
        self.abar_prev = np.append(1.0, self.abar[:-1])
        self.sqrt_abar = np.sqrt(self.abar)
        self.sqrt_1m_abar = np.sqrt(1.0 - self.abar)
        self.sqrt_recip_abar = np.sqrt(1.0 / self.abar)
        self.sqrt_recipm1_abar = np.sqrt(1.0 / self.abar - 1.0)

        # Phương trình (5): Phương sai và các hệ số kỳ vọng của phân phối hậu nghiệm
        self.post_var = betas * (1.0 - self.abar_prev) / (1.0 - self.abar)
        self.post_logvar = np.log(np.append(self.post_var[1], self.post_var[1:]))
        self.post_c0 = betas * np.sqrt(self.abar_prev) / (1.0 - self.abar)
        self.post_ct = (1.0 - self.abar_prev) * np.sqrt(alphas) / (1.0 - self.abar)

    @staticmethod
    def _ex(arr: np.ndarray, t: torch.Tensor, shape) -> torch.Tensor:
        """Trích xuất giá trị tại bước t và định hình lại theo kích thước tensor ảnh."""
        out = torch.from_numpy(arr).to(t.device)[t].float()
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    # ---------------------------------------------------------------- Khuếch tán thuận
    def q_sample(self, x0, t, noise=None):
        """Khuếch tán thuận (Eq. 1): thêm nhiễu vào ảnh gốc x0 tại bước t."""
        noise = torch.randn_like(x0) if noise is None else noise
        return self._ex(self.sqrt_abar, t, x0.shape) * x0 + self._ex(self.sqrt_1m_abar, t, x0.shape) * noise

    def q_posterior(self, x0, xt, t):
        """Phân phối hậu nghiệm thực tế q(x_{t-1} | x_t, x_0) (Eq. 5)."""
        mean = self._ex(self.post_c0, t, xt.shape) * x0 + self._ex(self.post_ct, t, xt.shape) * xt
        return mean, self._ex(self.post_logvar, t, xt.shape)

    # ---------------------------------------------------------------- Khử nhiễu ngược
    def p_mean_var(self, model: nn.Module, xt, t, clip: bool = True, out=None):
        """Dự đoán kỳ vọng và phương sai của bước khử nhiễu ngược (Eq. 2)."""
        if out is None:
            out = model(xt, self.timestep_map.to(t.device)[t])
        C = xt.shape[1]
        eps, v = out[:, :C], out[:, C:]

        # Nội suy phương sai theo Improved DDPM
        frac = (v + 1.0) / 2.0
        min_log = self._ex(self.post_logvar, t, xt.shape)
        max_log = self._ex(np.log(self.betas), t, xt.shape)
        logvar = frac * max_log + (1.0 - frac) * min_log

        # Dự đoán x0 từ xt và eps
        x0 = self._ex(self.sqrt_recip_abar, t, xt.shape) * xt - self._ex(self.sqrt_recipm1_abar, t, xt.shape) * eps
        if clip:
            x0 = x0.clamp(-1, 1)
        mean, _ = self.q_posterior(x0, xt, t)
        return mean, logvar, x0, eps

    # ---------------------------------------------------------------- Hàm mất mát
    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> dict:
        """Eq. (3): L_hybrid = L_MSE + lambda * T * L_vlb."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        out = model(xt, self.timestep_map.to(t.device)[t])
        C = x0.shape[1]
        mse = _mean_flat((noise - out[:, :C]) ** 2)

        # Nhánh VLB chỉ truyền đạo hàm vào hệ số phương sai v
        frozen = torch.cat([out[:, :C].detach(), out[:, C:]], 1)
        mean, logvar, _, _ = self.p_mean_var(model, xt, t, clip=False, out=frozen)
        true_mean, true_logvar = self.q_posterior(x0, xt, t)
        kl = _mean_flat(normal_kl(true_mean, true_logvar, mean, logvar)) / math.log(2.0)
        nll = -_mean_flat(discretized_gaussian_loglik(x0, mean, 0.5 * logvar)) / math.log(2.0)
        vlb = torch.where(t == 0, nll, kl)
        loss = mse + self.vlb_weight * self.T * vlb
        return {"loss": loss.mean(), "mse": mse.mean().detach(), "vlb": vlb.mean().detach()}

    # -------------------------------------------------------------- Sinh ảnh
    @torch.no_grad()
    def sample(self, model: nn.Module, shape, device="cpu", generator=None) -> torch.Tensor:
        """Lấy mẫu sinh ảnh: khử nhiễu dần từ x_T lùi về x_0."""
        x = torch.randn(shape, device=device, generator=generator)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            mean, logvar, _, _ = self.p_mean_var(model, x, t)
            if i > 0:
                x = mean + torch.exp(0.5 * logvar) * torch.randn(x.shape, device=device, generator=generator)
            else:
                x = mean
        return x.clamp(-1, 1)

    def respaced(self, n_steps: int) -> "GaussianDiffusion":
        """Lấy mẫu nhanh (respacing): rút gọn số bước sinh ảnh nhưng vẫn giữ lịch trình alpha_bar."""
        keep = sorted(set(np.linspace(0, self.T - 1, n_steps).round().astype(int).tolist()))
        new_betas, last = [], 1.0
        for i in keep:
            new_betas.append(1.0 - self.abar[i] / last)
            last = self.abar[i]
        base_map = self.timestep_map.tolist()
        return GaussianDiffusion(np.array(new_betas), self.vlb_weight, [base_map[i] for i in keep])
