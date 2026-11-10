"""Chạy toàn bộ các thực nghiệm và tự động đồng bộ liên kết Weights & Biases (W&B) vào README.md.

Script này tự động:
1. Chạy 12 thực nghiệm cốt lõi theo đúng bảng ánh xạ (Experiment Mapping Table) trong README.md.
2. Log toàn bộ thông số, metrics từng bước và đánh giá cuối cùng lên project W&B 'dcm-otfcl-cvpr2024'.
3. Thu thập URL thực tế của từng run trên W&B.
4. Tự động cập nhật các đường link W&B thật vào file README.md.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
import yaml

# Đảm bảo đường dẫn gốc của project nằm trong sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_experiment import run_one, set_by_path  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as f:
        return yaml.safe_load(f)


EXPERIMENTS = [
    {
        "row_key": "Finetune (No Memory)",
        "name": "finetune",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {"memory.type": "none"},
        "tags": ["lab1-reproduction", "table4", "finetune", "classification"],
    },
    {
        "row_key": "Reservoir Sampling",
        "name": "reservoir",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {"memory.type": "reservoir", "memory.memory_size": 2000},
        "tags": ["lab1-reproduction", "table4", "reservoir", "classification"],
    },
    {
        "row_key": "Eq. 12 in",
        "name": "dcm_se_lam75_farthest",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.type": "dcm",
            "memory.kdm": "se",
            "memory.lambda": 75.0,
            "memory.intra_prune_mode": "farthest",
            "memory.mpp_mode": "consistent",
        },
        "tags": ["lab1-reproduction", "table4", "dcm-se", "farthest", "classification"],
    },
    {
        "row_key": "Lời văn",
        "name": "dcm_se_lam75_nearest",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.type": "dcm",
            "memory.kdm": "se",
            "memory.lambda": 75.0,
            "memory.intra_prune_mode": "nearest",
            "memory.mpp_mode": "consistent",
        },
        "tags": ["lab1-reproduction", "table4", "dcm-se", "nearest", "classification"],
    },
    {
        "row_key": "Quantile (Đề xuất)",
        "name": "dcm_se_lam75_quantile",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.type": "dcm",
            "memory.kdm": "se",
            "memory.lambda": 75.0,
            "memory.intra_prune_mode": "quantile",
            "memory.mpp_mode": "consistent",
        },
        "tags": ["lab2-empirics", "table4", "dcm-se", "quantile", "classification"],
    },
    {
        "row_key": "DCM-JS (Exact",
        "name": "dcm_js_exact",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.type": "dcm",
            "memory.kdm": "js",
            "memory.js_mode": "exact",
            "memory.lambda": 0.5,
        },
        "tags": ["lab1-reproduction", "table4", "dcm-js", "exact", "classification"],
    },
    {
        "row_key": "DCM-JS (Dimwise)",
        "name": "dcm_js_dimwise",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.type": "dcm",
            "memory.kdm": "js",
            "memory.js_mode": "dimwise",
            "memory.lambda": 0.115,
        },
        "tags": ["lab2-empirics", "table4", "dcm-js", "dimwise", "classification"],
    },
    {
        "row_key": "MPP Cắt tỉa liên cụm",
        "name": "mpp_consistent_rho6",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {
            "memory.lambda": 70.0,
            "memory.rho": 6,
            "memory.mpp_mode": "consistent",
        },
        "tags": ["lab2-empirics", "ablation", "mpp", "classification"],
    },
    {
        "row_key": "Độ nhạy ngưỡng",
        "name": "lambda_sweep_lam90",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {"memory.lambda": 90.0},
        "tags": ["lab2-empirics", "lambda-sweep", "fig3a", "classification"],
    },
    {
        "row_key": "Mở rộng cụm động",
        "name": "dynamic_expansion_lam85",
        "config_path": "configs/classification_mnist.yaml",
        "overrides": {"memory.lambda": 85.0},
        "tags": ["lab2-empirics", "cluster-dynamics", "fig3b", "classification"],
    },
    {
        "row_key": "Thời gian xử lý bộ nhớ",
        "name": "memory_time_mnist_se",
        "config_path": "configs/generation_mnist.yaml",
        "overrides": {
            "data.name": "mnist",
            "train.memory_only": True,
            "memory.kdm": "se",
            "memory.lambda": 500.0,
        },
        "tags": ["lab1-reproduction", "memory-time", "fig3c"],
    },
    {
        "row_key": "Sinh ảnh IDDPM",
        "name": "gen_fast_dcm_se",
        "config_path": "configs/generation_mnist.yaml",
        "overrides": {
            "data.image_size": 16,
            "data.channels": 1,
            "data.max_samples_per_task": 640,
            "model.unet.base": 32,
            "model.unet.mults": [1, 2],
            "model.unet.n_res": 1,
            "model.unet.attn_res": [8],
            "model.unet.dropout": 0.0,
            "train.iters_per_step": 2,
            "train.lr": 0.0005,
            "eval.n_fid": 200,
            "eval.sample_steps": 50,
            "eval.compute_fid": False,
            "memory.type": "dcm",
            "memory.pi": 64,
            "memory.rho": 10,
            "memory.lambda": 60.0,
        },
        "tags": ["lab1-reproduction", "generation", "iddpm", "split-mnist"],
    },
]


def update_readme_links(run_links: dict[str, str]):
    """Cập nhật các liên kết W&B thực tế vào Experiment Mapping Table trong README.md."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_count = 0
    new_lines = []
    for line in lines:
        matched = False
        for exp in EXPERIMENTS:
            row_key = exp["row_key"]
            if row_key in line and exp["name"] in run_links:
                url = run_links[exp["name"]]
                # Thay thế liên kết [W&B Run](...) bằng URL thực tế
                new_line = re.sub(
                    r"\[W&B Run\]\([^)]+\)",
                    f"[W&B Run]({url})",
                    line
                )
                if new_line != line:
                    matched = True
                    new_lines.append(new_line)
                    updated_count += 1
                    break
        if not matched:
            new_lines.append(line)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n[README] Đã cập nhật thành công {updated_count}/{len(EXPERIMENTS)} liên kết W&B vào README.md!")


def main():
    project_name = "dcm-otfcl-cvpr2024"
    run_links = {}

    print("=" * 70)
    print("BẮT ĐẦU CHẠY VÀ LOG TOÀN BỘ 12 THỰC NGHIỆM LÊN WEIGHTS & BIASES")
    print(f"Project W&B: {project_name}")
    print("=" * 70)

    total_start = time.time()
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        name = exp["name"]
        print(f"\n>>> [{idx}/{len(EXPERIMENTS)}] Đang chạy: {exp['row_key']} ({name})...")
        cfg = load_yaml(exp["config_path"])
        cfg["name"] = name
        cfg["seeds"] = [0]  # Chạy seed 0 chuẩn để thu thập W&B link

        for k, v in exp["overrides"].items():
            set_by_path(cfg, k, v)

        # Bật W&B
        set_by_path(cfg, "wandb.enabled", True)
        set_by_path(cfg, "wandb.project", project_name)
        set_by_path(cfg, "wandb.tags", exp["tags"])

        out_dir = os.path.join(PROJECT_ROOT, "results", "runs", name)
        summary = run_one(cfg, [0], out_dir)

        url = summary.get("wandb_url")
        if url:
            run_links[name] = url
            print(f"    -> Đã log thành công: {url}")
        else:
            print(f"    -> [CẢNH BÁO] Không lấy được W&B URL cho {name}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH TẤT CẢ {len(EXPERIMENTS)} THỰC NGHIỆM TRONG {total_elapsed/60:.1f} PHÚT!")
    print("=" * 70)

    # Cập nhật README.md
    update_readme_links(run_links)

    # Lưu lại file ánh xạ kết quả
    mapping_path = os.path.join(PROJECT_ROOT, "results", "wandb_runs.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(run_links, f, indent=2)
    print(f"Bảng ánh xạ link đã được lưu tại: {mapping_path}")


if __name__ == "__main__":
    main()
