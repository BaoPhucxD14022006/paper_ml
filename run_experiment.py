"""File chạy chính: thực thi thí nghiệm OTFCL từ file cấu hình YAML.

Ví dụ cách chạy:
----------------
python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=se
python run_experiment.py --config configs/classification_mnist.yaml --sweep memory.lambda=20,30,40,50
python run_experiment.py --config configs/generation_mnist.yaml --seeds 0
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import statistics
import time

import yaml


def parse_value(v: str):
    """Chuyển chuỗi tham số CLI sang đúng kiểu dữ liệu (int, float, bool) qua YAML."""
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def set_by_path(cfg: dict, path: str, value) -> None:
    """Ghi đè giá trị trong dictionary lồng nhau theo đường dẫn dấu chấm (ví dụ: memory.lambda)."""
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_one(cfg: dict, seeds, out_dir: str) -> dict:
    """Chạy cấu hình qua danh sách các seed ngẫu nhiên và tổng hợp kết quả thống kê."""
    from src.trainers import run_classification, run_generation
    results = []
    for s in seeds:
        print(f"[{cfg['name']}] seed={s}", flush=True)
        if cfg["task"] == "classification":
            r = run_classification(cfg, s)
        else:
            r = run_generation(cfg, s, out_dir)
        results.append(r)
        brief = {k: v for k, v in r.items() if k not in ("trace", "cluster_label_hist")}
        print("   ->", json.dumps(brief), flush=True)

    summary = {"name": cfg["name"], "config": cfg, "runs": results}
    for metric in ("acc", "fid", "memory_time", "final_clusters", "final_mem_size"):
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if vals:
            summary[f"{metric}_mean"] = statistics.mean(vals)
            summary[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Chạy thực nghiệm học liên tục trực tuyến OTFCL")
    ap.add_argument("--config", required=True, help="Đường dẫn tới file YAML cấu hình")
    ap.add_argument("--set", nargs="*", default=[], help="Ghi đè giá trị dạng key=value")
    ap.add_argument("--sweep", nargs="*", default=[], help="Quét nhiều giá trị dạng key=v1,v2,...")
    ap.add_argument("--seeds", type=str, default=None, help="Danh sách seed ngẫu nhiên (cách nhau bởi dấu phẩy)")
    ap.add_argument("--out", default="results", help="Thư mục lưu kết quả")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # Ghi đè các tham số chỉ định qua --set
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_by_path(base, k, parse_value(v))
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else base.get("seeds", [0]))]

    # Xử lý quét lưới tham số qua --sweep
    sweep_keys, sweep_vals = [], []
    for kv in args.sweep:
        k, v = kv.split("=", 1)
        sweep_keys.append(k)
        sweep_vals.append([parse_value(x) for x in v.split(",")])

    rows = []
    for combo in itertools.product(*sweep_vals) if sweep_keys else [()]:
        cfg = copy.deepcopy(base)
        tag = []
        for k, v in zip(sweep_keys, combo):
            set_by_path(cfg, k, v)
            tag.append(f"{k.split('.')[-1]}={v}")
        name = base["name"] + ("_" + "_".join(tag) if tag else "")
        cfg["name"] = name
        out_dir = os.path.join(args.out, name)
        t0 = time.time()
        s = run_one(cfg, seeds, out_dir)
        rows.append({
            "name": name,
            **{k: v for k, v in s.items() if k.endswith(("_mean", "_std"))},
            "elapsed": time.time() - t0
        })

    print("\n==== TỔNG KẾT KẾT QUẢ ====")
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
