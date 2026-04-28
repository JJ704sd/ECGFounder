"""
param_observer.py
独立参数统计观察者进程，监控训练进度并计算各层权重统计信息。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "VS vibe coding files/ecg-annotation-platform/proxy-server"))
from state import read_shared_state, write_param_stats

ECGFOUNDER_BASE = Path("D:/ECG founder/ECGFounder")
OUTPUTS_DIR = ECGFOUNDER_BASE / "outputs"
SHARED_STATE_FILE = ECGFOUNDER_BASE / "shared_state.json"
PARAM_STATS_FILE = ECGFOUNDER_BASE / "param_stats.json"


def find_best_checkpoint(round_name: str) -> Optional[Path]:
    """Return outputs/round_name/best_macro_f1.pth if it exists."""
    ckpt_path = OUTPUTS_DIR / round_name / "best_macro_f1.pth"
    if ckpt_path.exists():
        return ckpt_path
    return None


def compute_param_stats(checkpoint_path: Path) -> dict:
    """Load .pth checkpoint and compute per-layer statistics."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract state_dict from checkpoint (handle different formats)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    layers = []
    total_norm = 0.0
    trainable = 0
    frozen = 0

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        # 跳过 0 维标量（如 BatchNorm 的 num_batches_tracked）
        if param.dim() == 0:
            continue
        data = param.detach().cpu().float()
        grad = param.grad
        grad_data = grad.detach().cpu().float() if grad is not None else None

        item = {
            "name": name,
            "shape": list(data.shape),
            "mean": float(data.mean().item()),
            "std": float(data.std().item()) if data.numel() > 1 else 0.0,
            "min": float(data.min().item()),
            "max": float(data.max().item()),
            "grad_mean": float(grad_data.mean().item()) if grad_data is not None else None,
            "grad_std": float(grad_data.std().item()) if grad_data is not None and grad_data.numel() > 1 else None,
        }
        layers.append(item)

        # global gradient norm for training stability monitoring
        if grad_data is not None:
            total_norm += (grad_data.norm().item() ** 2)

        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    global_norm = total_norm ** 0.5

    return {
        "layers": layers,
        "global_norm": global_norm,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }


def append_to_param_history(round_name: str, epoch: int, stats: dict) -> None:
    """Append epoch stats to outputs/round_name/param_history.json."""
    hist_file = OUTPUTS_DIR / round_name / "param_history.json"

    if hist_file.exists():
        history = json.loads(hist_file.read_text(encoding="utf-8"))
    else:
        history = {"round": round_name, "epochs": []}

    # Skip if this epoch already exists
    existing_epochs = [e["epoch"] for e in history["epochs"]]
    if epoch in existing_epochs:
        return

    history["epochs"].append({
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "global_norm": stats["global_norm"],
        "trainable_params": stats["trainable_params"],
        "frozen_params": stats["frozen_params"],
        "layer_summary": [
            {"name": l["name"], "mean": l["mean"], "std": l["std"]}
            for l in stats["layers"]
        ],
    })

    hist_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def main(poll_interval: float = 5.0) -> None:
    """Main polling loop."""
    last_epoch: Optional[int] = None
    current_round: Optional[str] = None

    print("[param_observer] 启动，开始监控训练状态...")

    while True:
        state = read_shared_state()
        status = state.get("status", "idle")
        epoch = state.get("epoch")
        round_name = state.get("round")

        # Reset tracking when training is done/error/idle
        if status in ("done", "error", "idle"):
            last_epoch = None
            current_round = None
            time.sleep(poll_interval)
            continue

        # Only process when status is "training"
        if status == "training":
            # Check if new round or new epoch
            new_round = round_name != current_round
            new_epoch = epoch is not None and epoch != last_epoch

            if new_round:
                current_round = round_name
                last_epoch = None

            if new_epoch or new_round:
                last_epoch = epoch
                ckpt = find_best_checkpoint(current_round)
                if ckpt is not None:
                    stats = compute_param_stats(ckpt)
                    write_param_stats(stats)
                    append_to_param_history(current_round, epoch, stats)
                    print(f"[param_observer] round={current_round} epoch={epoch} "
                          f"layers={len(stats['layers'])} global_norm={stats['global_norm']:.4f}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
