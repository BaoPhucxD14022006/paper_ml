"""Các hàm tiện ích chung: khởi tạo luồng dữ liệu, bộ nhớ và theo dõi Weights & Biases."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from ..baselines import ReservoirMemory
from ..data import OnlineStream, Preprocess, load_raw
from ..memory import DynamicClusterMemory, build_kdm

# Khắc phục lỗi WinError 6714 vô hại trên Windows khi dọn dẹp thư mục tạm lúc thoát chương trình
try:
    _orig_tempdir_cleanup = tempfile.TemporaryDirectory.cleanup

    def _safe_tempdir_cleanup(self):
        try:
            _orig_tempdir_cleanup(self)
        except OSError:
            pass

    tempfile.TemporaryDirectory.cleanup = _safe_tempdir_cleanup
except Exception:
    pass


def build_preprocess(dcfg: dict) -> Preprocess:
    """Tạo hàm tiền xử lý ảnh (chuẩn hóa kích thước, số kênh, khoảng giá trị)."""
    return Preprocess(
        image_size=dcfg.get("image_size"),
        channels=dcfg.get("channels"),
        value_range=dcfg.get("value_range", "01"),
        flatten=dcfg.get("flatten", False)
    )


def build_data(dcfg: dict, seed: int):
    """Tải dữ liệu thô và bọc vào đối tượng OnlineStream cho học liên tục trực tuyến."""
    pre = build_preprocess(dcfg)
    train = load_raw(dcfg["name"], train=True, root=dcfg.get("root", "./data"))
    test = load_raw(dcfg["name"], train=False, root=dcfg.get("root", "./data"))
    stream = OnlineStream(
        train,
        n_tasks=dcfg["n_tasks"],
        batch_size=dcfg["batch_size"],
        preprocess=pre,
        seed=seed,
        max_samples_per_task=dcfg.get("max_samples_per_task")
    )
    return stream, test, pre


def build_memory(mcfg: dict, seed: int) -> Optional[object]:
    """Khởi tạo đối tượng bộ nhớ (None, Reservoir hoặc DCM) từ cấu hình."""
    kind = mcfg["type"]
    if kind == "none":
        return None
    if kind == "reservoir":
        return ReservoirMemory(mcfg["memory_size"], seed=seed)
    if kind == "dcm":
        kdm_kwargs = {}
        if mcfg.get("kdm", "se") == "js":
            kdm_kwargs = dict(t_star=mcfg.get("t_star", 100), mode=mcfg.get("js_mode", "exact"))
        kdm = build_kdm(mcfg.get("kdm", "se"), **kdm_kwargs)
        pi = mcfg.get("pi", 128)
        rho = mcfg.get("rho") or max(1, mcfg["memory_size"] // pi)
        return DynamicClusterMemory(
            kdm,
            cluster_capacity=pi,
            max_clusters=rho,
            lambda_val=float(mcfg["lambda"]),
            expansion_mode=mcfg.get("expansion_mode", "batch"),
            expansion_reassign=mcfg.get("expansion_reassign", False),
            intra_prune_mode=mcfg.get("intra_prune_mode", "nearest"),
            mpp_mode=mcfg.get("mpp_mode", "consistent")
        )
    raise ValueError(f"Loại bộ nhớ không hợp lệ: {kind}")


def init_wandb(cfg: dict, seed: int):
    """Khởi tạo session Weights & Biases nếu được bật trong cấu hình."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    try:
        # Đặt thư mục đệm W&B cục bộ để tránh bị Windows Defender can thiệp vào AppData Temp
        cache_dir = os.path.abspath("./.wandb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", cache_dir)
        os.environ.setdefault("WANDB_CACHE_DIR", cache_dir)

        import wandb
        project = wcfg.get("project", "dcm-otfcl-cvpr2024")
        entity = wcfg.get("entity", None)
        run_name = f"{cfg['name']}_s{seed}"
        tags = list(wcfg.get("tags", []))
        if cfg.get("task"):
            tags.append(cfg["task"])
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=cfg.get("name"),
            tags=tags,
            config=cfg,
            reinit=True
        )
        return run
    except Exception as e:
        print(f"[W&B Warning] Không thể kết nối W&B ({e}). Chạy chế độ offline.", flush=True)
        return None
