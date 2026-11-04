"""Bộ nhớ baseline dùng thuật toán Reservoir Sampling để so sánh với DCM."""
from __future__ import annotations

import time
from typing import Optional

import torch


class ReservoirMemory:
    """Lấy mẫu hồ chứa (Vitter, 1985): mỗi mẫu mới xuất hiện đều có xác suất
    capacity / n_seen được lưu vào bộ nhớ, đảm bảo phân phối đều."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.size = 0
        self.n_seen = 0
        self.g = torch.Generator().manual_seed(seed)
        self.time_spent = 0.0
        self.history_k = []

    def __len__(self):
        return self.size

    @torch.no_grad()
    def observe(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """Cập nhật bộ nhớ với batch dữ liệu mới."""
        t0 = time.perf_counter()
        x = x.detach().cpu()
        # Khởi tạo tensor lưu trữ khi nhận batch đầu tiên
        if self.x is None:
            self.x = torch.empty((self.capacity,) + tuple(x.shape[1:]), dtype=x.dtype)
            if y is not None:
                self.y = torch.empty(self.capacity, dtype=y.dtype)

        for j in range(x.shape[0]):
            if self.size < self.capacity:
                # Bộ nhớ chưa đầy: điền tiếp vào ô trống
                slot = self.size
                self.size += 1
            else:
                # Bộ nhớ đã đầy: bốc thăm ngẫu nhiên xem có thay thế mẫu cũ không
                r = int(torch.randint(0, self.n_seen + 1, (1,), generator=self.g))
                slot = r if r < self.capacity else -1

            if slot >= 0:
                self.x[slot] = x[j]
                if y is not None:
                    self.y[slot] = y[j]
            self.n_seen += 1

        self.history_k.append(1)
        self.time_spent += time.perf_counter() - t0

    def all_samples(self):
        """Lấy toàn bộ mẫu hiện có trong bộ nhớ."""
        return self.x[:self.size], (None if self.y is None else self.y[:self.size])

    def sample_batch(self, n: int, generator: Optional[torch.Generator] = None):
        """Lấy ngẫu nhiên n mẫu để replay huấn luyện mô hình."""
        idx = torch.randint(0, self.size, (min(n, self.size),), generator=generator)
        return self.x[idx], (None if self.y is None else self.y[idx])
