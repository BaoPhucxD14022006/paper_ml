"""Mạng U-Net khử nhiễu eps_theta(x_t, t) cho mô hình khuếch tán Diffusion.

Thiết kế gọn nhẹ cho ảnh kích thước 32x32 hoặc 64x64:
- Mã hóa thời gian (Sinusoidal Positional Embedding).
- Các khối ResBlock với GroupNorm và hàm kích hoạt SiLU.
- Tự chú ý (Self-Attention) ở các tầng có độ phân giải thấp.
- Khi bật learn_sigma=True, mạng dự đoán cả nhiễu eps và hệ số nội suy phương sai v.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Mã hóa bước thời gian t thành vector nhúng theo chuỗi sóng Sin/Cos."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def norm(ch: int) -> nn.GroupNorm:
    """Chuẩn hóa nhóm (GroupNorm) chuẩn cho mạng U-Net trong Diffusion."""
    return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)


class ResBlock(nn.Module):
    """Khối tích chập dư có tích hợp vector nhúng bước thời gian emb."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.n1 = norm(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.n2 = norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Khối Self-Attention đa đầu (Multi-Head Self-Attention)."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.n = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """Mạng U-Net đầy đủ cho mô hình khuếch tán."""

    def __init__(self, in_channels: int = 3, base: int = 64, mults: Sequence[int] = (1, 2, 2, 2),
                 n_res: int = 2, attn_res: Sequence[int] = (16,), image_size: int = 32,
                 dropout: float = 0.1, learn_sigma: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        emb_dim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = nn.Conv2d(in_channels, base, 3, padding=1)

        # Nhánh Downsampling (đi xuống)
        self.down = nn.ModuleList()
        chans = [base]
        ch, res = base, image_size
        for li, m in enumerate(mults):
            for _ in range(n_res):
                blocks = [ResBlock(ch, base * m, emb_dim, dropout)]
                ch = base * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                self.down.append(nn.ModuleList(blocks))
                chans.append(ch)
            if li != len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
                res //= 2

        # Khối Middle ở đáy U-Net
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_dim, dropout),
            AttnBlock(ch),
            ResBlock(ch, ch, emb_dim, dropout)
        ])

        # Nhánh Upsampling (đi lên, kết hợp Skip Connections)
        self.up = nn.ModuleList()
        for li, m in reversed(list(enumerate(mults))):
            for i in range(n_res + 1):
                blocks = [ResBlock(ch + chans.pop(), base * m, emb_dim, dropout)]
                ch = base * m
                if res in attn_res:
                    blocks.append(AttnBlock(ch))
                if li != 0 and i == n_res:
                    blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blocks.append(nn.Conv2d(ch, ch, 3, padding=1))
                    res *= 2
                self.up.append(nn.ModuleList(blocks))

        # Tầng xuất ảnh: 2*C nếu học cả phương sai v, C nếu chỉ đoán nhiễu
        out_ch = in_channels * (2 if learn_sigma else 1)
        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return h

    def forward(self, x, t):
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, emb)
            hs.append(h)
        h = self._run(self.mid, h, emb)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], 1)
            h = self._run(blocks, h, emb)
        return self.out(h)
