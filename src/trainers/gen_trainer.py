"""Vòng lặp huấn luyện OTFCL cho mô hình sinh ảnh không giám sát với IDDPM (Mục 4.2 trong bài báo).

Nhãn lớp bị loại bỏ ngay khi rời khỏi stream: cả bộ nhớ và mô hình khuếch tán
đều hoàn toàn không nhìn thấy nhãn.
"""
from __future__ import annotations

import copy
import os
import time

import torch

from ..data import iterate_test
from ..models import GaussianDiffusion, UNet, make_betas
from ..utils import get_device, set_seed
from .common import build_data, build_memory, init_wandb


@torch.no_grad()
def ema_update(ema, model, decay):
    """Cập nhật trọng số trung bình trượt theo hàm mũ (EMA)."""
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def run_generation(cfg: dict, seed: int, out_dir: str, log_every: int = 100) -> dict:
    """Chạy một lượt huấn luyện sinh ảnh không giám sát trên dòng dữ liệu trực tuyến."""
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    dcfg, mcfg, tcfg, mdl = cfg["data"], cfg["memory"], cfg["train"], cfg["model"]

    # Khởi tạo kết nối W&B (nếu được bật trong cấu hình)
    wb_run = init_wandb(cfg, seed)

    stream, test, pre = build_data(dcfg, seed)
    memory = build_memory(mcfg, seed)
    memory_only = tcfg.get("memory_only", False)

    C = dcfg.get("channels", 3)
    model = UNet(in_channels=C, image_size=dcfg["image_size"], **mdl.get("unet", {})).to(device)
    ema = copy.deepcopy(model).eval()
    diffusion = GaussianDiffusion(
        make_betas(mdl.get("schedule", "linear"), mdl.get("T", 1000)),
        vlb_weight=mdl.get("vlb_weight", 1e-3)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.get("lr", 2e-4))
    g = torch.Generator().manual_seed(seed + 1)
    source = tcfg.get("train_source", "memory")
    train_bs = tcfg.get("train_batch_size", 64)

    trace = {"step": [], "k": [], "task": [], "mem_size": []}
    t0 = time.perf_counter()

    for i, (x, _labels_dropped) in enumerate(stream):
        # Bước 1-3: Cập nhật bộ nhớ DCM (hoàn toàn không dùng nhãn)
        if memory is not None:
            memory.observe(x)

        loss_dict = {"loss": 0.0, "mse": 0.0, "vlb": 0.0}
        if not memory_only:
            # Bước 4: Huấn luyện mô hình khuếch tán với dữ liệu từ bộ nhớ
            for _ in range(tcfg.get("iters_per_step", 1)):
                if memory is None:
                    x0 = x
                elif source == "memory":
                    x0, _ = memory.sample_batch(train_bs, generator=g)
                else:
                    mx, _ = memory.sample_batch(train_bs, generator=g)
                    x0 = torch.cat([x, mx])

                out = diffusion.training_loss(model, x0.to(device))
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema_update(ema, model, tcfg.get("ema", 0.999))
                loss_dict = {
                    "loss": float(out["loss"].item()),
                    "mse": float(out["mse"].item()),
                    "vlb": float(out["vlb"].item())
                }

        current_k = getattr(memory, "k", 1) if memory is not None else 0
        current_mem = len(memory) if memory is not None else 0
        current_task = stream.oracle_task_of_step(i)

        if memory is not None:
            trace["step"].append(i)
            trace["k"].append(current_k)
            trace["mem_size"].append(current_mem)
            trace["task"].append(current_task)

        # Log metrics từng bước lên W&B
        if wb_run is not None and (i % 50 == 0 or i == len(stream) - 1):
            wb_payload = {
                "train/loss": loss_dict["loss"],
                "train/mse": loss_dict["mse"],
                "train/vlb": loss_dict["vlb"],
                "memory/clusters": current_k,
                "memory/size": current_mem,
                "stream/oracle_task": current_task,
                "stream/step": i
            }
            wb_run.log(wb_payload, step=i)

        if log_every and i % log_every == 0:
            msg = f"  bước {i}/{len(stream)}: số cụm={getattr(memory, 'k', '-')} bộ nhớ={current_mem}"
            if not memory_only:
                msg += f" loss={loss_dict['loss']:.4f}"
            print(msg, flush=True)

    wall = time.perf_counter() - t0

    res = {
        "seed": seed,
        "wall_time": wall,
        "memory_time": getattr(memory, "time_spent", 0.0) if memory is not None else 0.0,
        "final_mem_size": len(memory) if memory is not None else 0,
        "final_clusters": getattr(memory, "k", None),
        "trace": trace
    }

    if memory is not None and hasattr(memory, "cluster_sizes"):
        res["cluster_sizes"] = memory.cluster_sizes()
        res["n_expansions"], res["n_prunes"] = memory.n_expansions, memory.n_prunes

    if memory_only:
        if wb_run is not None:
            wb_run.summary.update({
                "eval/final_clusters": res["final_clusters"],
                "eval/final_mem_size": res["final_mem_size"],
                "eval/memory_time_sec": res["memory_time"],
                "eval/wall_time_sec": res["wall_time"]
            })
            res["wandb_url"] = getattr(wb_run, "url", None)
            wb_run.finish()
        return res

    # Lưu trọng số mô hình EMA và sinh ảnh mẫu kiểm thử
    os.makedirs(out_dir, exist_ok=True)
    torch.save(ema.state_dict(), os.path.join(out_dir, f"ema_seed{seed}.pt"))
    ecfg = cfg.get("eval", {})
    sampler = diffusion.respaced(ecfg.get("sample_steps", 100))
    n_fid = ecfg.get("n_fid", 1000)
    fakes = []
    for s in range(0, n_fid, ecfg.get("sample_batch", 100)):
        n = min(ecfg.get("sample_batch", 100), n_fid - s)
        fakes.append(sampler.sample(ema, (n, C, dcfg["image_size"], dcfg["image_size"]), device=device).cpu())
    fakes = torch.cat(fakes)

    # Xuất lưới 64 ảnh sinh ra
    from torchvision.utils import save_image
    samples_path = os.path.join(out_dir, f"samples_seed{seed}.png")
    save_image((fakes[:64] + 1) / 2, samples_path, nrow=8)

    # Tính toán chỉ số FID nếu cấu hình yêu cầu
    if ecfg.get("compute_fid", True):
        from ..utils.metrics import InceptionFeatures, fid
        reals = torch.cat([xb for xb, _ in iterate_test(test, pre)])
        idx = torch.randperm(len(reals), generator=torch.Generator().manual_seed(0))[:n_fid]
        res["fid"] = fid(InceptionFeatures(device), reals[idx], fakes)

    # Đẩy ảnh mẫu và metrics cuối cùng lên W&B
    if wb_run is not None:
        try:
            import wandb
            wb_summary = {
                "eval/final_clusters": res["final_clusters"],
                "eval/final_mem_size": res["final_mem_size"],
                "eval/memory_time_sec": res["memory_time"],
                "eval/wall_time_sec": res["wall_time"],
            }
            if "fid" in res:
                wb_summary["eval/fid"] = res["fid"]
            if os.path.exists(samples_path):
                wb_run.log({"eval/generated_samples": wandb.Image(samples_path)})
            wb_run.summary.update(wb_summary)
            res["wandb_url"] = getattr(wb_run, "url", None)
        finally:
            wb_run.finish()

    return res
