import subprocess
import sys
import os
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "ablation_run_logs"
STATUS = LOG_DIR / "status.log"
ABLATIONS = [
    "distance_only",
    "no_distance",
    "ftm_only",
    "single_frame_distance",
]
EPOCHS = os.environ.get("VIFI_ABLATION_EPOCHS", "20")
BATCH_SIZE = os.environ.get("VIFI_ABLATION_BATCH_SIZE", "32")


def write_status(message):
    LOG_DIR.mkdir(exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def run_to_log(cmd, log_path):
    with log_path.open("ab", buffering=0) as log:
        header = f"\n===== {' '.join(cmd)} {datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        log.write(header.encode("utf-8"))
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def latest_best_checkpoint(ablation):
    ckpt_dir = ROOT / "checkpoints" / ablation
    epoch60 = ckpt_dir / "fold1_epoch60.pth"
    if epoch60.exists():
        return epoch60
    best = sorted(ckpt_dir.glob("fold1_best_epoch*_acc*.pth"), key=lambda p: p.stat().st_mtime)
    return best[-1] if best else None


def parse_inference_result(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    online = re.search(r"Online accuracy.*?([0-9.]+)%", text)
    offline = re.search(r"Offline accuracy.*?([0-9.]+)%", text)
    if not online or not offline:
        return None
    return float(online.group(1)), float(offline.group(1))


def main():
    LOG_DIR.mkdir(exist_ok=True)
    write_status(f"Starting ablation experiments via Python runner, epochs={EPOCHS}.")

    for ablation in ABLATIONS:
        write_status(f"TRAIN start: {ablation}")
        train_log = LOG_DIR / f"{ablation}.train.log"
        code = run_to_log([
            sys.executable, "-u", "train.py",
            "--fold", "1",
            "--epochs", EPOCHS,
            "--batch_size", BATCH_SIZE,
            "--num_workers", "0",
            "--ablation", ablation,
        ], train_log)
        write_status(f"TRAIN end: {ablation} exit={code}")

        checkpoint = latest_best_checkpoint(ablation)
        if checkpoint is None:
            write_status(f"INFERENCE skipped: {ablation} no checkpoint found")
            continue

        write_status(f"INFERENCE start: {ablation} checkpoint={checkpoint}")
        infer_log = LOG_DIR / f"{ablation}.inference.log"
        code = run_to_log([
            sys.executable, "-u", "inference.py",
            "--fold", "1",
            "--ablation", ablation,
            "--checkpoint", str(checkpoint),
        ], infer_log)
        write_status(f"INFERENCE end: {ablation} exit={code}")

        result = parse_inference_result(infer_log)
        if result:
            online, offline = result
            write_status(f"RESULT {ablation}: online={online:.2f}% offline={offline:.2f}%")

    write_status("All ablation experiments finished.")


if __name__ == "__main__":
    main()
