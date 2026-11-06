"""Chạy toàn bộ bộ thực nghiệm và tự động xuất các bảng số liệu & biểu đồ.

Bao gồm:
1. Tái hiện Bảng 4: Phân loại Split MNIST (Finetune, Reservoir, DCM-SE, DCM-JS).
2. Đối chứng thực nghiệm các điểm mâu thuẫn công thức trong bài báo (Spec Mục 3):
   - Eq. (12): intra_prune_mode = "nearest" (lời văn) vs "farthest" (công thức in)
   - Eq. (13-14): mpp_mode = "consistent" (chuẩn logic) vs "literal" (công thức in)
3. Xuất các biểu đồ phân tích cắt bỏ (Hình 3a, 3b, 3c).
4. Thử nghiệm sinh ảnh không giám sát với IDDPM (R-DDPM vs DCM-SE).
"""
from __future__ import annotations

import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiment import run_one, set_by_path  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    plot_cluster_dynamics,
    plot_lambda_sweep,
    plot_memory_time,
)


def load_cfg(path: str) -> dict:
    """Đọc file YAML cấu hình."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cfg(base: dict, name: str, overrides: dict, seeds, out_root: str = "results/suite") -> dict:
    """Chạy một cấu hình với các tham số ghi đè (nếu đã có kết quả cache thì bỏ qua để tiết kiệm thời gian)."""
    cfg = copy.deepcopy(base)
    cfg["name"] = name
    for k, v in overrides.items():
        set_by_path(cfg, k, v)
    out_dir = os.path.join(out_root, name)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"[BỎ QUA - Đã có cache] {name}", flush=True)
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return run_one(cfg, seeds, out_dir)


def main():
    os.makedirs("results/suite", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    cls_base = load_cfg("configs/classification_mnist.yaml")
    gen_base = load_cfg("configs/generation_mnist.yaml")
    seeds = [0, 1, 2]

    report = {
        "classification_table4": {},
        "ablation_contradictions": {},
        "ablation_lambda": [],
        "memory_time": {},
        "generation": {}
    }

    # =========================================================================
    # 1. BẢNG 4: Phân loại OTFCL trên Split MNIST (b=10, bộ nhớ=2000)
    # =========================================================================
    cls_variants = [
        ("finetune", {"memory.type": "none"}),
        ("reservoir", {"memory.type": "reservoir", "memory.memory_size": 2000}),
        ("dcm_se_lam75", {"memory.type": "dcm", "memory.kdm": "se", "memory.lambda": 75.0}),
        ("dcm_se_lam85", {"memory.type": "dcm", "memory.kdm": "se", "memory.lambda": 85.0}),
        ("dcm_js_dimwise", {"memory.type": "dcm", "memory.kdm": "js", "memory.js_mode": "dimwise", "memory.lambda": 0.115}),
        ("dcm_js_exact", {"memory.type": "dcm", "memory.kdm": "js", "memory.js_mode": "exact", "memory.lambda": 0.5}),
    ]
    for name, ov in cls_variants:
        s = run_cfg(cls_base, name, ov, seeds)
        report["classification_table4"][name] = {
            "acc_mean": s.get("acc_mean"), "acc_std": s.get("acc_std"),
            "clusters_mean": s.get("final_clusters_mean"),
            "mem_size_mean": s.get("final_mem_size_mean"),
            "memory_time_mean": s.get("memory_time_mean"),
        }

    # =========================================================================
    # 2. ĐỐI CHỨNG CÁC ĐIỂM MÂU THUẪN TRONG BÀI BÁO (Spec Mục 3)
    # =========================================================================
    contra_variants = [
        # Eq. (12): nearest (lời văn) vs farthest (công thức in)
        ("eq12_nearest_text", {"memory.lambda": 75.0, "memory.intra_prune_mode": "nearest", "memory.mpp_mode": "consistent"}),
        ("eq12_farthest_printed", {"memory.lambda": 75.0, "memory.intra_prune_mode": "farthest", "memory.mpp_mode": "consistent"}),
        # Eq. (13-14): consistent (chuẩn logic) vs literal (công thức in) khi giới hạn rho nhỏ = 6
        ("mpp_consistent_rho6", {"memory.lambda": 70.0, "memory.rho": 6, "memory.mpp_mode": "consistent"}),
        ("mpp_literal_rho6", {"memory.lambda": 70.0, "memory.rho": 6, "memory.mpp_mode": "literal"}),
    ]
    for name, ov in contra_variants:
        s = run_cfg(cls_base, name, ov, seeds)
        report["ablation_contradictions"][name] = {
            "acc_mean": s.get("acc_mean"), "acc_std": s.get("acc_std"),
            "clusters_mean": s.get("final_clusters_mean"),
            "mem_size_mean": s.get("final_mem_size_mean"),
        }

    # =========================================================================
    # 3. PHÂN TÍCH CẮT BỎ: Hình 3a (quét lambda) & Hình 3b (động thái mở rộng cụm)
    # =========================================================================
    lambdas_cls = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    lam_summaries = []
    for lam in lambdas_cls:
        s = run_cfg(cls_base, f"ablation_lam_{int(lam)}", {"memory.lambda": lam}, [0])
        lam_summaries.append(s)
        report["ablation_lambda"].append({
            "lambda": lam, "clusters": s.get("final_clusters_mean"),
            "mem_size": s.get("final_mem_size_mean"), "acc": s.get("acc_mean"),
        })
    plot_lambda_sweep(lam_summaries, lambdas_cls, "results/figures/fig3a_lambda_sweep.png", metric="acc")
    plot_cluster_dynamics(lam_summaries[lambdas_cls.index(90.0)], "results/figures/fig3b_dynamic_expansion_lam90.png")
    plot_cluster_dynamics(lam_summaries[lambdas_cls.index(85.0)], "results/figures/fig3b_dynamic_expansion_lam85.png")

    # =========================================================================
    # 4. THỜI GIAN QUẢN LÝ BỘ NHỚ: Hình 3c (Split MNIST & Split Fashion)
    # =========================================================================
    se_times, js_times = [], []
    for ds_name in ("mnist", "fashion"):
        s_se = run_cfg(gen_base, f"memtime_{ds_name}_se",
                       {"data.name": ds_name, "train.memory_only": True, "memory.kdm": "se", "memory.lambda": 500.0}, [0])
        s_js = run_cfg(gen_base, f"memtime_{ds_name}_js",
                       {"data.name": ds_name, "train.memory_only": True, "memory.kdm": "js", "memory.js_mode": "exact", "memory.lambda": 0.5}, [0])
        se_times.append(s_se["memory_time_mean"])
        js_times.append(s_js["memory_time_mean"])
        report["memory_time"][ds_name] = {"dcm_se_sec": s_se["memory_time_mean"], "dcm_js_sec": s_js["memory_time_mean"]}
    plot_memory_time(["Split MNIST", "Split Fashion"], se_times, js_times, "results/figures/fig3c_memory_time.png")

    # =========================================================================
    # 5. THỰC NGHIỆM SINH ẢNH IDDPM (R-DDPM vs DCM-SE trên CPU)
    # =========================================================================
    fast_gen = {
        "data.image_size": 16, "data.channels": 1, "data.max_samples_per_task": 640,
        "model.unet.base": 32, "model.unet.mults": [1, 2], "model.unet.n_res": 1,
        "model.unet.attn_res": [8], "model.unet.dropout": 0.0,
        "train.iters_per_step": 2, "train.lr": 0.0005,
        "eval.n_fid": 200, "eval.sample_steps": 50, "eval.compute_fid": False,
    }
    s_rddpm = run_cfg(gen_base, "gen_fast_rddpm", {**fast_gen, "memory.type": "reservoir", "memory.memory_size": 640}, [0])
    s_dcm = run_cfg(gen_base, "gen_fast_dcm_se", {**fast_gen, "memory.type": "dcm", "memory.pi": 64, "memory.rho": 10, "memory.lambda": 60.0}, [0])
    report["generation"] = {
        "r_ddpm": {"wall_time": s_rddpm["runs"][0]["wall_time"], "final_mem_size": s_rddpm["final_mem_size_mean"]},
        "dcm_se": {"wall_time": s_dcm["runs"][0]["wall_time"], "final_mem_size": s_dcm["final_mem_size_mean"],
                   "final_clusters": s_dcm["final_clusters_mean"]},
    }

    with open("results/suite/full_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n=== HOÀN THÀNH TOÀN BỘ BỘ THỰC NGHIỆM ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
