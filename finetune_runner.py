"""
finetune_runner.py
Controlled training runner for ECGFounder.
Watches train_task.json and launches finetune_mitbih_ecgfounder_gpu_amp.py as subprocess.
"""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# Add proxy-server to sys.path for state.py imports
sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "VS vibe coding files/ecg-annotation-platform/proxy-server")
)
from state import read_shared_state, write_shared_state, read_train_task, delete_train_task

ECGFOUNDER_BASE = Path("D:/ECG founder/ECGFounder")
OUTPUTS_DIR = ECGFOUNDER_BASE / "outputs"
TRAIN_SCRIPT = ECGFOUNDER_BASE / "finetune_mitbih_ecgfounder_gpu_amp.py"


# ------------------------------------------------------------------
# find_next_round
# ------------------------------------------------------------------

def find_next_round() -> str:
    """Find the next available round number by scanning outputs/round_* directories."""
    if not OUTPUTS_DIR.exists():
        return "round_1"

    max_num = 0
    for item in OUTPUTS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("round_"):
            try:
                num = int(item.name.split("_")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass

    return f"round_{max_num + 1}"


# ------------------------------------------------------------------
# _parse_log_epochs
# ------------------------------------------------------------------

def _parse_log_epochs(log_text: str) -> list[dict]:
    """
    Parse log text to extract epoch data.
    Each epoch dict: epoch, stage, train_loss, train_acc, train_f1,
                     val_acc, val_macro_f1, lr, is_best.
    Also updates shared_state.json with real-time progress.
    """
    results = []
    current = {}
    current_stage = None
    current_lr = None
    best_macro_f1 = 0.0
    best_epoch_num = 0

    for line in log_text.splitlines():
        line = line.strip()

        # Stage header: "========== Stage 1: Freeze backbone, lr=0.0001 =========="
        stage_match = re.match(r"=+\s*Stage \d+:\s*(.+?),\s*lr(?:_backbone)?=(.+?)\s*=+", line)
        if stage_match:
            current_stage = stage_match.group(1).strip()
            current_lr = float(stage_match.group(2).strip())
            if current and current.get("epoch") is not None:
                current["stage"] = current_stage
                current["lr"] = current_lr
                results.append(current)
                current = {}
            continue

        # Epoch line: "Epoch 1 (Stage 1)"
        epoch_match = re.match(r"Epoch\s+(\d+)\s+\(Stage\s+\d+\)", line)
        if epoch_match:
            if current and current.get("epoch") is not None:
                current["stage"] = current_stage
                current["lr"] = current_lr
                results.append(current)
            current = {"epoch": int(epoch_match.group(1)), "is_best": False}
            continue

        # Train metrics: "  Train Loss=1.2563 Acc=0.6031 F1=0.6089"
        train_match = re.match(r"Train\s+Loss=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)", line)
        if train_match:
            current["train_loss"] = float(train_match.group(1))
            current["train_acc"] = float(train_match.group(2))
            current["train_f1"] = float(train_match.group(3))
            continue

        # Val metrics: "  Val   Acc=0.6900 MacroF1=0.6852 WeightedF1=0.6852 LR=9.89e-06"
        val_match = re.match(
            r"Val\s+Acc=([\d.]+)\s+MacroF1=([\d.]+)\s+WeightedF1=([\d.]+)(?:\s+LR=([\d.e+-]+))?",
            line
        )
        if val_match:
            current["val_acc"] = float(val_match.group(1))
            current["val_macro_f1"] = float(val_match.group(2))
            # val_weighted_f1 is third but not in epoch dict spec — skip
            lr_group = val_match.group(4)
            if lr_group is not None:
                current["lr"] = float(lr_group)
            continue

        # Save best: "  [SAVE] best_macro_f1=0.6852"
        save_match = re.match(r"\[SAVE\]\s+best_macro_f1=([\d.]+)", line)
        if save_match:
            current["is_best"] = True
            val_f1 = float(save_match.group(1))
            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                best_epoch_num = current.get("epoch", 0)
            continue

    if current and current.get("epoch") is not None:
        current["stage"] = current_stage
        current["lr"] = current_lr
        results.append(current)

    # Update shared_state.json with latest progress
    if results:
        latest = results[-1]
        write_shared_state({
            "status": "running",
            "round": results[0].get("stage", "unknown") if results else "unknown",
            "current_epoch": latest.get("epoch"),
            "total_epochs": 20,  # STAGE1_EPOCHS(5) + STAGE2_EPOCHS(15) = 20
            "train_loss": latest.get("train_loss"),
            "train_acc": latest.get("train_acc"),
            "train_f1": latest.get("train_f1"),
            "val_acc": latest.get("val_acc"),
            "val_macro_f1": latest.get("val_macro_f1"),
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch_num,
            "lr": latest.get("lr"),
            "is_best": latest.get("is_best", False),
            "error": None,
        })

    return results


# ------------------------------------------------------------------
# run_training
# ------------------------------------------------------------------

def run_training(round_name: str, task: dict):
    """Launch the training subprocess and monitor its progress."""
    output_dir = OUTPUTS_DIR / round_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = OUTPUTS_DIR / f"train_{round_name}.log"

    # Build command — training script is called as-is; extra args are ignored
    # since it has no argparse, but the convention allows future extensibility.
    epochs = task.get("epochs", 20)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
    ]

    write_shared_state({
        "status": "running",
        "round": round_name,
        "current_epoch": None,
        "total_epochs": epochs,
        "train_loss": None,
        "train_acc": None,
        "train_f1": None,
        "val_acc": None,
        "val_macro_f1": None,
        "best_macro_f1": 0.0,
        "best_epoch": 0,
        "lr": None,
        "is_best": False,
        "error": None,
    })

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ECGFOUNDER_BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Monitoring thread: parse log file every 2 seconds
        def monitor():
            while True:
                time.sleep(2.0)
                if proc.poll() is not None:
                    break
                if log_file.exists():
                    try:
                        text = log_file.read_text(encoding="utf-8")
                        _parse_log_epochs(text)
                    except Exception:
                        pass

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        # Stream stdout to log file
        with open(log_file, "w", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()

        proc.wait()
        monitor_thread.join(timeout=5)

        # Final log parse
        if log_file.exists():
            final_text = log_file.read_text(encoding="utf-8")
            epochs_data = _parse_log_epochs(final_text)

            best_macro_f1 = 0.0
            best_epoch_num = 0
            for ep in epochs_data:
                if ep.get("is_best") and ep.get("val_macro_f1", 0) > best_macro_f1:
                    best_macro_f1 = ep["val_macro_f1"]
                    best_epoch_num = ep.get("epoch", 0)

            write_shared_state({
                "status": "done",
                "round": round_name,
                "best_macro_f1": best_macro_f1,
                "best_epoch": best_epoch_num,
                "epochs_data": epochs_data,
                "error": None,
            })
        else:
            write_shared_state({
                "status": "error",
                "round": round_name,
                "error": "Log file not found after training",
            })

    except Exception as e:
        write_shared_state({
            "status": "error",
            "round": round_name,
            "error": str(e),
        })


# ------------------------------------------------------------------
# wait_for_train_task
# ------------------------------------------------------------------

def wait_for_train_task(poll_interval: float = 5.0):
    """Main loop: poll train_task.json and run training when queued."""
    print(f"[finetune_runner] Polling every {poll_interval}s")

    while True:
        task = read_train_task()

        if task is not None and task.get("status") == "queued":
            round_name = find_next_round()
            print(f"[finetune_runner] Found queued task → {round_name}")

            run_training(round_name, task)
            delete_train_task()

            print(f"[finetune_runner] Task complete, deleted train_task.json")
        else:
            time.sleep(poll_interval)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("[finetune_runner] Starting, waiting for training tasks...")
    wait_for_train_task()
