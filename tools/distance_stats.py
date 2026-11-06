"""Công cụ khảo sát phân phối khoảng cách KDM trên tập dữ liệu thực tế.

Mục đích:
1. Tìm dải giá trị khoảng cách phù hợp để chọn ngưỡng lambda (Eq. 10).
2. Kiểm tra hiện tượng bão hòa về hằng số ln 2 của phân kỳ Jensen-Shannon (Eq. 8).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import Preprocess, class_splits, load_raw  # noqa: E402
from src.memory import JensenShannonKDM, MemoryCluster, SquaredErrorKDM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Khảo sát thống kê khoảng cách KDM")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--value_range", default="01")
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--n", type=int, default=128, help="Số mẫu lấy thử trên mỗi task")
    ap.add_argument("--t_star", type=int, default=100)
    args = ap.parse_args()

    raw = load_raw(args.dataset, train=True)
    pre = Preprocess(args.image_size, args.channels, args.value_range, args.flatten)
    se, js = SquaredErrorKDM(), JensenShannonKDM(t_star=args.t_star)
    jsd = JensenShannonKDM(t_star=args.t_star, mode="dimwise")
    g = torch.Generator().manual_seed(0)

    groups, protos = [], []
    for classes in class_splits(10 if args.dataset != "cifar100" else 100, args.n_tasks):
        idx = torch.isin(raw.labels, torch.tensor(classes)).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)[:args.n]]
        x = pre(raw.images[idx])
        groups.append(x)
        protos.append(x[MemoryCluster.find_medoid_index(x, se)])

    P = torch.stack(protos)
    q = torch.tensor([0.05, 0.5, 0.95])
    print(f"Số chiều ảnh = {groups[0][0].numel()}, abar(t*={args.t_star}) = {js.alpha_bar:.4f}, scale delta = {js.scale:.3f}")
    print("Task | SE tới medoid của chính nó [5%, 50%, 95%] | Min SE tới medoid các task khác [5%, 50%, 95%]")
    for t, x in enumerate(groups):
        d = se.pairwise(x, P)
        own = d[:, t]
        other = torch.cat([d[:, :t], d[:, t + 1:]], 1).min(1).values
        print(f"  {t}  | {own.quantile(q).numpy().round(1)} | {other.quantile(q).numpy().round(1)}")

    print("\nKhoảng cách SE giữa các medoid:\n", se.pairwise(P, P).numpy().round(1))
    print("\nPhân kỳ JS exact giữa các medoid (ln 2 = 0.6931):\n", js.pairwise(P, P).numpy().round(4))

    within = se.pairwise(groups[0][:32], groups[0][:32])
    delta = js.scale * within[within > 0].sqrt()
    print(f"\nDải giá trị delta trong cùng task 0: {delta.min():.1f} .. {delta.max():.1f} "
          f"(JS exact sẽ bão hòa về ln 2 khi delta > 75)")
    print("Giá trị JS exact nội bộ task 0 (5 mẫu đầu):\n", js.pairwise(groups[0][:5], groups[0][:5]).numpy().round(4))

    print("\nPhân kỳ JS dimwise (không bị bão hòa) giữa các medoid:\n", jsd.pairwise(P, P).numpy().round(4))
    dd = jsd.pairwise(torch.cat(groups), P)
    labels = torch.arange(len(groups)).repeat_interleave(args.n)
    own = dd[torch.arange(len(dd)), labels]
    other = dd.clone()
    other[torch.arange(len(dd)), labels] = float("inf")
    print(f"JS dimwise: median medoid cùng lớp {own.median():.4f}, median medoid khác lớp gần nhất {other.min(1).values.median():.4f}")


if __name__ == "__main__":
    main()
