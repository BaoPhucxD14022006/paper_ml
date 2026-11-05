"""Vòng lặp huấn luyện OTFCL cho bài toán phân loại (Mục 4.5, Bảng 4 trong bài báo)."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ..data import N_CLASSES, iterate_test
from ..models import build_classifier
from ..utils import get_device, set_seed
from ..utils.metrics import accuracy
from .common import build_data, build_memory


def run_classification(cfg: dict, seed: int, log_every: int = 500) -> dict:
    """Chạy một lượt huấn luyện phân loại trực tuyến với seed xác định."""
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    dcfg, mcfg, tcfg = cfg["data"], cfg["memory"], cfg["train"]

    # Khởi tạo dữ liệu và bộ nhớ
    stream, test, pre = build_data(dcfg, seed)
    memory = build_memory(mcfg, seed)
    n_classes = N_CLASSES[dcfg["name"]]
    model = build_classifier(cfg["model"]["name"], n_classes, **cfg["model"].get("kwargs", {})).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=tcfg.get("lr", 0.05))

    g = torch.Generator().manual_seed(seed + 1)
    replay_bs = tcfg.get("replay_batch_size", 10)
    source = tcfg.get("train_source", "stream+memory")
    memory_only = tcfg.get("memory_only", False)

    trace = {"step": [], "k": [], "task": [], "mem_size": []}
    t0 = time.perf_counter()

    # Vòng lặp duyệt qua dòng dữ liệu stream (chỉ thấy mỗi mẫu 1 lần)
    for i, (x, y) in enumerate(stream):
        if memory is not None and source == "memory":
            memory.observe(x, y)

        if not memory_only:
            for _ in range(tcfg.get("iters_per_step", 1)):
                if memory is None:
                    # Finetune: chỉ học trên batch dữ liệu mới hiện tại
                    bx, by = x, y
                elif source == "memory":
                    bx, by = memory.sample_batch(x.shape[0] + replay_bs, generator=g)
                elif len(memory) > 0:
                    # Experience Replay: ghép batch mới X_i với mẫu lấy từ bộ nhớ M
                    mx, my = memory.sample_batch(replay_bs, generator=g)
                    bx, by = torch.cat([x, mx]), torch.cat([y, my])
                else:
                    bx, by = x, y

                loss = F.cross_entropy(model(bx.to(device)), by.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if memory is not None and source != "memory":
            memory.observe(x, y)

        if memory is not None:
            trace["step"].append(i)
            trace["k"].append(getattr(memory, "k", 1))
            trace["mem_size"].append(len(memory))
            trace["task"].append(stream.oracle_task_of_step(i))  # Chỉ dùng để vẽ đồ thị đánh giá

        if log_every and i % log_every == 0:
            k = getattr(memory, "k", "-") if memory is not None else "-"
            print(f"  bước {i}/{len(stream)}: số cụm={k}, bộ nhớ={len(memory) if memory else 0}", flush=True)

    wall = time.perf_counter() - t0

    res = {
        "seed": seed,
        "wall_time": wall,
        "memory_time": getattr(memory, "time_spent", 0.0) if memory is not None else 0.0,
        "final_mem_size": len(memory) if memory is not None else 0,
        "final_clusters": getattr(memory, "k", None) if memory is not None else None,
        "trace": trace
    }

    if not memory_only:
        # Đánh giá độ chính xác tổng thể và trên từng task
        res["acc"] = accuracy(model, iterate_test(test, pre), device)
        res["acc_per_task"] = [
            accuracy(model, iterate_test(test, pre, classes=c), device)
            for c in stream.splits
        ]

    if memory is not None and hasattr(memory, "cluster_sizes"):
        res["cluster_sizes"] = memory.cluster_sizes()
        res["n_expansions"] = memory.n_expansions
        res["n_prunes"] = memory.n_prunes
        # Đếm phân phối nhãn trong từng cụm để khảo sát độ tinh khiết (chỉ dùng để phân tích)
        res["cluster_label_hist"] = [
            torch.bincount(c.labels, minlength=n_classes).tolist()
            for c in memory.clusters if c.labels is not None
        ]

    return res
