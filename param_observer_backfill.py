"""
D:/ECG founder/ECGFounder/param_observer_backfill.py
对已有 round_1 ~ round_12 批量生成 param_history.json
一次性运行，不影响正在进行的训练
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加 state.py 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "VS vibe coding files/ecg-annotation-platform/proxy-server"))

OUTPUTS = Path("D:/ECG founder/ECGFounder/outputs")


def compute_checkpoint_stats(checkpoint_path: Path) -> dict:
    """加载 checkpoint，计算各层参数统计"""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # 实际结构: ckpt['model'] 直接是 state_dict OrderedDict
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    layers = []
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        # 跳过 0 维标量（如 BatchNorm 的 num_batches_tracked）
        if param.dim() == 0:
            continue
        data = param.detach().cpu().float()
        layers.append({
            "name": name,
            "shape": list(data.shape),
            "mean": float(data.mean().item()),
            "std": float(data.std().item()) if data.numel() > 1 else 0.0,
            "min": float(data.min().item()),
            "max": float(data.max().item()),
        })

    return {"layers": layers}


def backfill_round(round_name: str) -> bool:
    """为一个 round 生成 param_history.json。返回是否做了处理。"""
    ckpt_path = OUTPUTS / round_name / "best_macro_f1.pth"
    if not ckpt_path.exists():
        print(f"[SKIP] {round_name}: no checkpoint")
        return False

    history_file = OUTPUTS / round_name / "param_history.json"
    if history_file.exists():
        print(f"[SKIP] {round_name}: param_history.json already exists")
        return False

    print(f"[BACKFILL] {round_name}...")
    stats = compute_checkpoint_stats(ckpt_path)

    # 由于历史数据没有 epoch 粒度，只存最终 checkpoint 的统计
    history = {
        "round": round_name,
        "epochs": [{
            "epoch": "final",
            "timestamp": datetime.now().isoformat(),
            "global_norm": 0.0,
            "trainable_params": 0,
            "frozen_params": 0,
            "layer_summary": [
                {"name": l["name"], "mean": l["mean"], "std": l["std"]}
                for l in stats["layers"]
            ],
        }]
    }

    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {round_name}: {len(stats['layers'])} layers")
    return True


def main():
    """扫描所有 round_* 目录，批量补录"""
    print("=== ECGFounder 参数统计批量补录 ===")
    processed = 0
    for item in sorted(OUTPUTS.iterdir()):
        if item.is_dir() and item.name.startswith("round_"):
            if backfill_round(item.name):
                processed += 1
    print(f"=== 完成，已处理 {processed} 个 round ===")


if __name__ == "__main__":
    main()
