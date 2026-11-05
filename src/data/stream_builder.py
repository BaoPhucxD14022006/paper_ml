"""Dòng dữ liệu cho bài toán Học liên tục trực tuyến không nhãn tác vụ (OTFCL - Mục 3.1).

Tập train được chia thành C nhóm lớp rời rạc (ví dụ Split MNIST: {0,1}, {2,3}, ...).
Dòng dữ liệu được cắt thành các batch X_1 ... X_n có kích thước b.
Mỗi mẫu chỉ xuất hiện đúng một lần trong suốt quá trình huấn luyện.

Mô hình chỉ nhận được cặp (x, y) tại mỗi bước - hoàn toàn không có task ID hay cờ ranh giới.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import torch

from .datasets import N_CLASSES, Preprocess, RawDataset


def class_splits(n_classes: int, n_tasks: int) -> List[List[int]]:
    """Chia đều n_classes thành n_tasks danh sách các lớp rời nhau."""
    per = n_classes // n_tasks
    return [list(range(t * per, (t + 1) * per)) for t in range(n_tasks)]


class OnlineStream:
    """Tạo luồng dữ liệu trực tuyến không để lộ task ID."""

    def __init__(self, raw: RawDataset, n_tasks: int, batch_size: int,
                 preprocess: Preprocess, seed: int = 0,
                 splits: Optional[Sequence[Sequence[int]]] = None,
                 max_samples_per_task: Optional[int] = None):
        self.raw = raw
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.splits = [list(s) for s in (splits or class_splits(N_CLASSES[raw.name], n_tasks))]
        g = torch.Generator().manual_seed(seed)
        order, task_of_sample = [], []

        for t, classes in enumerate(self.splits):
            mask = torch.isin(raw.labels, torch.tensor(classes))
            idx = mask.nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(idx.numel(), generator=g)]
            if max_samples_per_task is not None:
                idx = idx[:max_samples_per_task]
            order.append(idx)
            task_of_sample.append(torch.full((idx.numel(),), t))

        self._order = torch.cat(order)
        self._task_of_sample = torch.cat(task_of_sample)
        self.n_steps = (self._order.numel() + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for i in range(self.n_steps):
            idx = self._order[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.preprocess(self.raw.images[idx]), self.raw.labels[idx]

    # Hàm nội bộ chỉ dùng để đánh giá và vẽ đồ thị (không bao giờ truyền cho mô hình)
    def oracle_task_of_step(self, i: int) -> int:
        return int(self._task_of_sample[min(i * self.batch_size, self._order.numel() - 1)])

    @property
    def n_tasks(self) -> int:
        return len(self.splits)


def iterate_test(raw: RawDataset, preprocess: Preprocess, batch_size: int = 1000,
                 classes: Optional[Sequence[int]] = None):
    """Hàm phát sinh từng batch dữ liệu kiểm thử."""
    idx = torch.arange(len(raw))
    if classes is not None:
        idx = torch.isin(raw.labels, torch.tensor(list(classes))).nonzero(as_tuple=True)[0]
    for s in range(0, idx.numel(), batch_size):
        j = idx[s:s + batch_size]
        yield preprocess(raw.images[j]), raw.labels[j]
