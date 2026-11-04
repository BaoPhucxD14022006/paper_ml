"""Cụm bộ nhớ đơn lẻ M^k (Mục 3.3 - 3.4 trong bài báo).

Mỗi cụm chứa tối đa `capacity` (= pi) mẫu kèm nhãn (nếu có).
Nhãn chỉ lưu kèm thụ động để phục vụ bộ phân loại học có giám sát,
toàn bộ cơ chế phân cụm của DCM hoàn toàn không đọc nhãn.
"""
from __future__ import annotations

from typing import Optional

import torch

from .kdm import KDM

INTRA_PRUNE_MODES = ("nearest", "farthest", "quantile")


class MemoryCluster:
    def __init__(self, samples: torch.Tensor, labels: Optional[torch.Tensor],
                 capacity: int, kdm: KDM, prototype_index: Optional[int] = None):
        if samples.shape[0] == 0:
            raise ValueError("Cụm phải được khởi tạo với ít nhất 1 mẫu")
        self.kdm = kdm
        self.capacity = capacity
        self.samples = samples.clone()
        self.labels = None if labels is None else labels.clone()

        # Prototype (medoid) chỉ tính một lần khi tạo cụm (theo lời văn bài báo sau Eq. 9)
        if prototype_index is None:
            prototype_index = self.find_medoid_index(self.samples, kdm)
        self.prototype = self.samples[prototype_index].clone()
        self.prototype_index = prototype_index

        # Lưu cache khoảng cách từ từng mẫu trong cụm tới prototype để không phải tính lại
        self.proto_dists = self.kdm.pairwise(self.samples, self.prototype[None])[:, 0]

        # Đánh dấu vị trí prototype ban đầu để chế độ "nearest" không bao giờ xóa nhầm prototype (khoảng cách 0)
        self.is_proto = torch.zeros(len(self), dtype=torch.bool)
        self.is_proto[prototype_index] = True

        # Nếu cụm tạo từ batch lớn ban đầu (b > pi), tỉa bớt cho vừa sức chứa
        while len(self) > capacity:
            self._remove_one("nearest")

    # ------------------------------------------------------------------ Phương trình (9)
    @staticmethod
    def find_medoid_index(samples: torch.Tensor, kdm: KDM) -> int:
        """Eq. (9): Tìm mẫu có tổng khoảng cách tới các mẫu khác trong cụm là nhỏ nhất."""
        if samples.shape[0] == 1:
            return 0
        d = kdm.pairwise(samples, samples)
        d.fill_diagonal_(0.0)  # Bỏ qua chính nó (j == c)
        return int(d.sum(1).argmin())

    # ------------------------------------------------------------------ Quản lý mẫu
    def __len__(self) -> int:
        return self.samples.shape[0]

    def add(self, x: torch.Tensor, y: Optional[torch.Tensor], intra_prune_mode: str,
            dist_to_proto: Optional[torch.Tensor] = None) -> None:
        """Thêm một mẫu vào cụm và thực hiện cắt tỉa nếu cụm bị tràn dung lượng (Eq. 12)."""
        if dist_to_proto is None:
            dist_to_proto = self.kdm.pairwise(x[None], self.prototype[None])[0, 0]
        else:
            dist_to_proto = dist_to_proto.reshape(()).to(self.proto_dists)

        # Nếu cụm chưa đầy: thêm thẳng vào cuối
        if len(self) < self.capacity:
            self.samples = torch.cat([self.samples, x[None]], 0)
            self.proto_dists = torch.cat([self.proto_dists, dist_to_proto[None]], 0)
            self.is_proto = torch.cat([self.is_proto, torch.zeros(1, dtype=torch.bool)], 0)
            if self.labels is not None and y is not None:
                self.labels = torch.cat([self.labels, y.reshape(1).to(self.labels)], 0)
            return

        # Cụm đã đầy (sĩ số = capacity): xác định mẫu nào cần bị loại và ghi đè tại chỗ
        if intra_prune_mode not in INTRA_PRUNE_MODES:
            raise ValueError(f"intra_prune_mode phải thuộc {INTRA_PRUNE_MODES}")

        if intra_prune_mode == "farthest":
            # Theo công thức in ở Eq. 12: xóa mẫu xa prototype nhất
            cur_idx = int(self.proto_dists.argmax())
            if float(dist_to_proto) >= float(self.proto_dists[cur_idx]):
                return  # Mẫu mới còn xa hơn mẫu xa nhất hiện có -> bỏ qua luôn
            e = cur_idx
        elif intra_prune_mode == "quantile":
            # Thuật toán đề xuất cải tiến: xóa mẫu nội điểm có khoảng cách lân cận hẹp nhất
            # để các mẫu giữ lại phân bố đều từ tâm ra biên
            all_d = torch.cat([self.proto_dists, dist_to_proto[None]])
            order = all_d.argsort()
            gaps = torch.full((all_d.numel(),), float("inf"))
            gaps[order[1:-1]] = all_d[order[2:]] - all_d[order[:-2]]
            gaps[:len(self)] = gaps[:len(self)].masked_fill(self.is_proto, float("inf"))
            victim = int(gaps.argmin())
            if victim == len(self):
                return
            e = victim
        else:
            # Theo lời văn trang 6 bài báo: xóa mẫu gần prototype nhất (ngoại trừ prototype)
            d = self.proto_dists.masked_fill(self.is_proto, float("inf"))
            cur_idx = int(d.argmin())
            if float(dist_to_proto) < float(d[cur_idx]):
                return  # Mẫu mới gần prototype hơn mẫu gần nhất hiện có -> bỏ qua luôn
            e = cur_idx

        # Ghi đè trực tiếp vào vị trí mẫu bị loại bỏ
        self.samples[e] = x
        self.proto_dists[e] = dist_to_proto
        self.is_proto[e] = False
        if self.labels is not None and y is not None:
            self.labels[e] = y.reshape(()).to(self.labels)

    # ----------------------------------------------------------------- Phương trình (12)
    def removal_index(self, mode: str) -> int:
        """Tìm vị trí mẫu cần loại bỏ khi cụm vượt quá dung lượng.
        Hỗ trợ so sánh giữa lời văn (nearest), công thức in (farthest) và đề xuất (quantile).
        """
        if mode not in INTRA_PRUNE_MODES:
            raise ValueError(f"intra_prune_mode phải thuộc {INTRA_PRUNE_MODES}")
        d = self.proto_dists
        if mode == "farthest":
            return int(d.argmax())
        if mode == "quantile":
            order = d.argsort()
            gaps = torch.full((d.numel(),), float("inf"))
            if d.numel() > 2:
                gaps[order[1:-1]] = d[order[2:]] - d[order[:-2]]
            gaps = gaps.masked_fill(self.is_proto, float("inf"))
            return int(gaps.argmin())
        if bool(self.is_proto.all()):
            return 0
        d = d.masked_fill(self.is_proto, float("inf"))
        return int(d.argmin())

    def _remove_one(self, mode: str) -> None:
        """Hàm nội bộ: xóa bỏ 1 mẫu khỏi cụm khi khởi tạo."""
        e = self.removal_index(mode)
        keep = torch.ones(len(self), dtype=torch.bool)
        keep[e] = False
        self.samples = self.samples[keep]
        self.proto_dists = self.proto_dists[keep]
        self.is_proto = self.is_proto[keep]
        if self.labels is not None:
            self.labels = self.labels[keep]
