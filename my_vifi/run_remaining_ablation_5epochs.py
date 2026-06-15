import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "ablation_run_logs"
SUMMARY = LOG_DIR / "ablation_5epoch_summary.md"
ABLATIONS = ["no_distance", "ftm_only", "single_frame_distance"]


def run(cmd, log_path):
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        log.write(f"# {' '.join(cmd)}\n")
        log.write(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def newest_checkpoint(ablation):
    ckpt_dir = ROOT / "checkpoints" / ablation
    files = list(ckpt_dir.glob("fold1_best_epoch*_acc*.pth"))
    if not files:
        files = list(ckpt_dir.glob("fold1_epoch*.pth"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def parse_result(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    online = re.search(r"Online accuracy.*?([0-9.]+)%", text)
    offline = re.search(r"Offline accuracy.*?([0-9.]+)%", text)
    return (
        online.group(1) if online else "NA",
        offline.group(1) if offline else "NA",
    )


def main():
    LOG_DIR.mkdir(exist_ok=True)
    rows = [
        "# Vi-Fi Ablation Results, 5 Epochs",
        "",
        "| Ablation | Checkpoint | Online | Offline 30-frame vote |",
        "|---|---|---:|---:|",
    ]

    for ablation in ABLATIONS:
        print(f"\n===== TRAIN {ablation} =====", flush=True)
        train_log = LOG_DIR / f"{ablation}.5epoch.train.log"
        code = run([
            sys.executable, "-u", "train.py",
            "--fold", "1",
            "--epochs", "5",
            "--batch_size", "32",
            "--num_workers", "0",
            "--ablation", ablation,
        ], train_log)
        if code != 0:
            rows.append(f"| {ablation} | train failed exit={code} | NA | NA |")
            SUMMARY.write_text("\n".join(rows) + "\n", encoding="utf-8")
            continue

        ckpt = newest_checkpoint(ablation)
        if ckpt is None:
            rows.append(f"| {ablation} | no checkpoint | NA | NA |")
            SUMMARY.write_text("\n".join(rows) + "\n", encoding="utf-8")
            continue

        print(f"\n===== INFERENCE {ablation}: {ckpt.name} =====", flush=True)
        infer_log = LOG_DIR / f"{ablation}.5epoch.inference.log"
        code = run([
            sys.executable, "-u", "inference.py",
            "--fold", "1",
            "--ablation", ablation,
            "--checkpoint", str(ckpt),
        ], infer_log)
        online, offline = parse_result(infer_log)
        rows.append(f"| {ablation} | {ckpt.name} | {online}% | {offline}% |")
        SUMMARY.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"\nSummary written to {SUMMARY}")


if __name__ == "__main__":
    main()
