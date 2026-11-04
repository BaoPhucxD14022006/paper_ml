"""Bộ nhớ Cụm Động (Dynamic Cluster Memory - DCM).
Triển khai Mục 3.3 - 3.6 và Thuật toán 1 (Algorithm 1) trong bài báo.

Bộ nhớ này hoàn toàn model-free và label-free:
Hàm observe chỉ dựa vào tensor dữ liệu x để quyết định gom cụm và lưu trữ.
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch

from .cluster import MemoryCluster
from .kdm import KDM

EXPANSION_MODES = ("batch", "sample")
MPP_MODES = ("consistent", "literal", "off")


class DynamicClusterMemory:
    def __init__(self, kdm: KDM, cluster_capacity: int = 128, max_clusters: int = 15,
                 lambda_val: float = 2000.0, expansion_mode: str = "batch",
                 expansion_reassign: bool = False, intra_prune_mode: str = "nearest",
                 mpp_mode: str = "consistent", eps: float = 1e-8):
        assert expansion_mode in EXPANSION_MODES, expansion_mode
        assert mpp_mode in MPP_MODES, mpp_mode
        self.kdm = kdm
        self.pi = cluster_capacity          # Sức chứa tối đa của 1 cụm
        self.rho = max_clusters             # Số lượng cụm tối đa cho phép
        self.lam = lambda_val               # Ngưỡng phát hiện phân phối mới để mở rộng cụm
        self.expansion_mode = expansion_mode
        self.expansion_reassign = expansion_reassign
        self.intra_prune_mode = intra_prune_mode
        self.mpp_mode = mpp_mode
        self.eps = eps
        self.clusters: List[MemoryCluster] = []

        # Các biến theo dõi thống kê thực nghiệm
        self.n_expansions = 0
        self.n_prunes = 0
        self.time_spent = 0.0
        self.history_k: List[int] = []
        self._cache = None

    # ================================================================ Thuộc tính chung
    @property
    def k(self) -> int:
        """Số lượng cụm hiện có."""
        return len(self.clusters)

    def __len__(self) -> int:
        """Tổng số mẫu hiện có trong toàn bộ các cụm."""
        return sum(len(c) for c in self.clusters)

    def prototypes(self) -> torch.Tensor:
        """Tập hợp tất cả các prototype đại diện của các cụm."""
        return torch.stack([c.prototype for c in self.clusters], 0)

    @torch.no_grad()
    def observe(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """Bước 1 - 3 trong Thuật toán 1: tiếp nhận batch mới và cập nhật bộ nhớ."""
        t0 = time.perf_counter()
        x = x.detach().cpu()
        y = None if y is None else y.detach().cpu()

        if self.k == 0:
            # Bước đầu tiên t_1: tạo cụm đầu tiên M^1 từ batch đầu X_1
            self.clusters.append(MemoryCluster(x, y, self.pi, self.kdm))
        elif self.expansion_mode == "batch":
            self._observe_batch(x, y)
        else:
            self._observe_per_sample(x, y)

        # Bước 3: Cắt tỉa liên cụm MPP nếu số cụm vượt quá giới hạn rho
        if self.mpp_mode != "off":
            while self.k > self.rho:
                self._prune_one_cluster()

        self._cache = None
        self.history_k.append(self.k)
        self.time_spent += time.perf_counter() - t0

    def all_samples(self):
        """Hợp nhất toàn bộ dữ liệu từ các cụm trong bộ nhớ."""
        if self._cache is None:
            xs = torch.cat([c.samples for c in self.clusters], 0)
            ys = None
            if self.clusters and self.clusters[0].labels is not None:
                ys = torch.cat([c.labels for c in self.clusters], 0)
            self._cache = (xs, ys)
        return self._cache

    def sample_batch(self, n: int, generator: Optional[torch.Generator] = None):
        """Bước 4: Lấy ngẫu nhiên n mẫu từ bộ nhớ để ôn tập (Replay) cho mô hình."""
        xs, ys = self.all_samples()
        idx = torch.randint(0, xs.shape[0], (min(n, xs.shape[0]),), generator=generator)
        return xs[idx], (None if ys is None else ys[idx])

    def cluster_sizes(self) -> List[int]:
        """Kích thước từng cụm hiện tại."""
        return [len(c) for c in self.clusters]

    # ======================================================= Các bước gom cụm & Mở rộng
    def _assign(self, xj, yj, dists_row: torch.Tensor) -> None:
        """Eq. (11): Gán mẫu vào cụm có prototype gần nhất."""
        c_star = int(dists_row.argmin())
        yj_ = None if yj is None else yj
        self.clusters[c_star].add(xj, yj_, self.intra_prune_mode, dist_to_proto=dists_row[c_star])

    def _new_cluster(self, xj, yj) -> None:
        """Tạo cụm mới với prototype là xj."""
        yy = None if yj is None else yj.reshape(1)
        self.clusters.append(MemoryCluster(xj[None], yy, self.pi, self.kdm, prototype_index=0))
        self.n_expansions += 1

    def _observe_batch(self, x, y) -> None:
        """Xử lý mở rộng cụm theo mức độ batch (Eq. 10)."""
        d = self.kdm.pairwise(x, self.prototypes())          # Kích thước (batch_size, k)
        min_per_sample = d.min(1).values
        # Mở rộng nếu khoảng cách nhỏ nhất của cả batch tới mọi prototype đều lớn hơn lambda
        expand = bool(min_per_sample.min() > self.lam)
        j_star = int(min_per_sample.argmax()) if expand else -1

        if expand and self.expansion_reassign:
            self._new_cluster(x[j_star], None if y is None else y[j_star])
            d = self.kdm.pairwise(x, self.prototypes())
            for j in range(x.shape[0]):
                if j != j_star:
                    self._assign(x[j], None if y is None else y[j], d[j])
            return

        for j in range(x.shape[0]):
            if j == j_star:
                continue
            self._assign(x[j], None if y is None else y[j], d[j])

        if expand:
            self._new_cluster(x[j_star], None if y is None else y[j_star])

    def _observe_per_sample(self, x, y) -> None:
        """Xử lý mở rộng cụm theo từng mẫu đơn lẻ (dạng vòng lặp Algorithm 1)."""
        for j in range(x.shape[0]):
            yj = None if y is None else y[j]
            d = self.kdm.pairwise(x[j:j + 1], self.prototypes())[0]
            if bool(d.min() > self.lam):
                self._new_cluster(x[j], yj)
            else:
                self._assign(x[j], yj, d)

    # ======================================================= Cắt tỉa liên cụm (MPP)
    def relation_matrix(self):
        """Tính ma trận khoảng cách D và ma trận quan hệ tương đồng B = 1 / (D + eps)."""
        P = self.prototypes()
        D = self.kdm.pairwise(P, P).to(torch.float64)
        D = 0.5 * (D + D.T)
        B = 1.0 / (D + self.eps)
        B.fill_diagonal_(0.0)
        return D, B

    def select_pair_and_victim(self):
        """Xác định cặp cụm chồng lấn nhiều nhất và cụm dư thừa cần xóa (Eq. 13-14)."""
        D, B = self.relation_matrix()
        k = D.shape[0]
        off = ~torch.eye(k, dtype=torch.bool)

        if self.mpp_mode == "literal":
            # Theo công thức in trong bài (Eq. 13): argmax D
            Dm = D.masked_fill(~off, -float("inf"))
            flat = int(Dm.argmax())
        else:
            # Theo bản chất logic: tìm cặp gần nhau nhất (tương đồng nhất, khoảng cách nhỏ nhất)
            Dm = D.masked_fill(~off, float("inf"))
            flat = int(Dm.argmin())

        g, h = divmod(flat, k)
        s_g, s_h = float(B[g].sum()), float(B[h].sum())        # Điểm tương đồng với toàn bộ bộ nhớ (Eq. 14)

        if self.mpp_mode == "literal":
            victim = h if s_g > s_h else g
        else:
            # Cụm nào có tổng tương đồng lớn hơn chứng tỏ nó nằm lẫn vào nhiều cụm khác -> xóa bỏ
            victim = g if s_g > s_h else h
        return g, h, victim

    def _prune_one_cluster(self) -> None:
        """Xóa 1 cụm dư thừa nhất để giảm sĩ số cụm về lại rho."""
        _, _, victim = self.select_pair_and_victim()
        del self.clusters[victim]
        self.n_prunes += 1
