import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"D:\python\conda\envs\dlgpu\python.exe")


def run_and_log(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n===== RUN =====\n")
        log.write(" ".join(str(part) for part in cmd) + "\n\n")
        log.flush()
        return subprocess.run(
            [str(part) for part in cmd],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def find_best_checkpoint(checkpoint_dir, fold):
    pattern = f"fold{fold}_best_epoch*_acc*.pth"
    best = []
    for path in checkpoint_dir.glob(pattern):
        match = re.search(r"_acc([0-9.]+)\.pth$", path.name)
        if match:
            best.append((float(match.group(1)), path.stat().st_mtime, path))
    if not best:
        return None
    return max(best, key=lambda item: (item[0], item[1]))[2]


def main():
    parser = argparse.ArgumentParser(description="Run strict Vi-Fi train/val/test experiment")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_seed", type=int, default=2026)
    parser.add_argument("--ablation", type=str, default="full")
    args = parser.parse_args()

    out_dir = ROOT / "strict_run_logs"
    train_log = out_dir / f"{args.ablation}.fold{args.fold}.strict.train.log"
    test_log = out_dir / f"{args.ablation}.fold{args.fold}.strict.test.log"
    summary = out_dir / f"{args.ablation}.fold{args.fold}.strict.summary.md"

    checkpoint_root = ROOT / "checkpoints_strict"
    log_root = ROOT / "logs_strict"

    train_cmd = [
        PYTHON,
        "train.py",
        "--fold", args.fold,
        "--epochs", args.epochs,
        "--batch_size", args.batch_size,
        "--ablation", args.ablation,
        "--val_ratio", args.val_ratio,
        "--val_seed", args.val_seed,
        "--checkpoint_dir", checkpoint_root,
        "--log_dir", log_root,
        "--device", "cuda",
    ]
    code = run_and_log(train_cmd, train_log)
    if code != 0:
        summary.write_text(
            f"# Strict experiment failed\n\nTraining exited with code `{code}`.\n"
            f"See `{train_log}`.\n",
            encoding="utf-8",
        )
        return code

    ckpt_dir = checkpoint_root / args.ablation
    ckpt = find_best_checkpoint(ckpt_dir, args.fold)
    if ckpt is None:
        summary.write_text(
            "# Strict experiment failed\n\nNo best checkpoint was produced.\n",
            encoding="utf-8",
        )
        return 2

    test_cmd = [
        PYTHON,
        "inference.py",
        "--fold", args.fold,
        "--ablation", args.ablation,
        "--val_ratio", args.val_ratio,
        "--val_seed", args.val_seed,
        "--checkpoint", ckpt,
    ]
    code = run_and_log(test_cmd, test_log)

    train_tail = "\n".join(train_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    test_text = test_log.read_text(encoding="utf-8", errors="replace")
    result_lines = [
        line for line in test_text.splitlines()
        if "Online accuracy" in line or "Offline accuracy" in line
    ]

    summary.write_text(
        "# Strict Vi-Fi Full Model Experiment\n\n"
        f"- Fold: `{args.fold}`\n"
        f"- Epochs: `{args.epochs}`\n"
        f"- Batch size: `{args.batch_size}`\n"
        f"- Validation split: `val_ratio={args.val_ratio}`, `val_seed={args.val_seed}`\n"
        f"- Checkpoint selected by validation only: `{ckpt}`\n"
        f"- Test set is evaluated once after training.\n\n"
        "## Test Results\n\n"
        + ("\n".join(f"- {line}" for line in result_lines) if result_lines else "- No result lines parsed.")
        + "\n\n## Train Log Tail\n\n```text\n"
        + train_tail
        + "\n```\n\n"
        f"Full train log: `{train_log}`\n\n"
        f"Full test log: `{test_log}`\n",
        encoding="utf-8",
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
