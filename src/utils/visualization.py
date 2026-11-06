"""Vẽ các biểu đồ phục vụ báo cáo và phân tích thực nghiệm (tái hiện Hình 3a, 3b, 3c)."""
from __future__ import annotations

import json
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_summary(path: str) -> dict:
    """Đọc file kết quả json từ thư mục thí nghiệm."""
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def plot_cluster_dynamics(summary: dict, out_png: str, run: int = 0) -> None:
    """Tái hiện Hình 3b: Sự biến thiên số lượng cụm bộ nhớ DCM so với số task thực tế."""
    tr = summary["runs"][run]["trace"]
    seen = [t + 1 for t in tr["task"]]
    plt.figure(figsize=(6, 3.5))
    plt.step(tr["step"], tr["k"], where="post", label="Số lượng cụm bộ nhớ (DCM)")
    plt.step(tr["step"], seen, where="post", ls="--", label="Số phân phối thực tế (Oracle)")
    plt.xlabel("Bước huấn luyện $t_i$")
    plt.ylabel("Số lượng")
    plt.title(summary["name"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_lambda_sweep(summaries: List[dict], lambdas: List[float], out_png: str,
                      metric: Optional[str] = None) -> None:
    """Tái hiện Hình 3a: Số lượng cụm (và Accuracy/FID) theo các giá trị ngưỡng lambda."""
    k = [s.get("final_clusters_mean") for s in summaries]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(lambdas, k, "o-", color="tab:blue", label="Số cụm")
    ax1.set_xlabel(r"Ngưỡng $\lambda$")
    ax1.set_ylabel("Số lượng cụm", color="tab:blue")
    if metric:
        vals = [s.get(f"{metric}_mean") for s in summaries]
        if all(v is not None for v in vals):
            ax2 = ax1.twinx()
            ax2.plot(lambdas, vals, "s--", color="tab:red")
            ax2.set_ylabel(metric.upper(), color="tab:red")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_memory_time(names: List[str], se_times: List[float], js_times: List[float], out_png: str) -> None:
    """Tái hiện Hình 3c: So sánh thời gian xử lý bộ nhớ giữa DCM-SE và DCM-JS."""
    import numpy as np
    x = np.arange(len(names))
    plt.figure(figsize=(6, 3.5))
    plt.bar(x - 0.2, se_times, 0.4, label="DCM-SE")
    plt.bar(x + 0.2, js_times, 0.4, label="DCM-JS")
    plt.xticks(x, names)
    plt.ylabel("Thời gian (giây)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
