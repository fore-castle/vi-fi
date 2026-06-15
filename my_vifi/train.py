import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as cfg_lib
import model as model_lib
import loss as loss_lib
from dataset import ViFiDataset, compute_norm_stats
from ablations import ABLATION_SPECS, configure_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Vi-Fi Affinity Matrix Learning")
    parser.add_argument("--fold", type=int, default=1, help="fold index (1-5)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_root", type=str,
                        default="d:/buaa_study/research/vi-fi/RAN4model_dfv4p4")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="d:/buaa_study/research/vi-fi/my_vifi/checkpoints")
    parser.add_argument("--log_dir", type=str,
                        default="d:/buaa_study/research/vi-fi/my_vifi/logs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=sorted(ABLATION_SPECS.keys()))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="fraction of non-test sequences reserved for validation")
    parser.add_argument("--val_seed", type=int, default=None,
                        help="seed for deterministic sequence-level validation split")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = cfg_lib.Config()
    configure_ablation(config, args.ablation)
    config.fold = args.fold
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.num_workers = args.num_workers
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.val_seed is not None:
        config.val_seed = args.val_seed
    config.data_root = Path(args.data_root)
    config.checkpoint_dir = Path(args.checkpoint_dir) / config.ablation
    config.log_dir = Path(args.log_dir) / config.ablation
    config.device = args.device

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablation: {config.ablation}")
    print(f"Camera feature indices: {config.camera_feature_indices}")
    print(f"Phone feature indices: {config.phone_feature_indices}")
    print(f"Window size: {config.window_size}")
    print(f"Validation split: ratio={config.val_ratio}, seed={config.val_seed}")

    torch.backends.cudnn.enabled = True

    # Compute normalization statistics from training data
    norm_stats = compute_norm_stats(config.data_root, config=config)

    # Data
    train_dataset = ViFiDataset(config.data_root, split="train", fold=config.fold,
                                config=config, norm_stats=norm_stats)
    val_dataset = ViFiDataset(config.data_root, split="val", fold=config.fold,
                              config=config, norm_stats=norm_stats)

    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Check fold and data paths.")
        return
    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty. Using training set for validation.")
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, drop_last=True,
                              persistent_workers=config.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, drop_last=False,
                            persistent_workers=config.num_workers > 0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = model_lib.MultimodalNetwork(config).to(device)
    criterion = loss_lib.AffinityLoss(config.Nm_phone, config.Nm_camera).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = torch.load(args.resume, map_location=device, weights_only=False)

    # Optimizer + AMP
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = MultiStepLR(optimizer,
                            milestones=[config.epochs // 2],
                            gamma=0.1)
    scaler = torch.amp.GradScaler(device.type)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(config.log_dir / f"fold{config.fold}"))

    best_acc = 0.0
    best_checkpoint = ""
    recent_checkpoints = []  # track last 5 regular checkpoints

    for epoch in range(1, config.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0

        for i, batch in enumerate(train_loader):
            cam = batch["camera"].to(device)
            ph = batch["phone"].to(device)
            cam_mask = batch["camera_mask"].to(device)
            ph_mask = batch["phone_mask"].to(device)
            aff_mat = batch["aff_mat"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device.type):
                output = model(cam, ph, cam_mask, ph_mask)
                _, _, _, loss, _, _, acc, _ = criterion(output, aff_mat, ph_mask, cam_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc += acc.item()
            train_batches += 1

            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch}, Iter {i+1}: loss={loss.item():.4f}, "
                      f"acc={acc.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}",
                      flush=True)

        train_loss /= train_batches
        train_acc /= train_batches

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                cam = batch["camera"].to(device)
                ph = batch["phone"].to(device)
                cam_mask = batch["camera_mask"].to(device)
                ph_mask = batch["phone_mask"].to(device)
                aff_mat = batch["aff_mat"].to(device)

                with torch.autocast(device.type):
                    output = model(cam, ph, cam_mask, ph_mask)
                    _, _, _, loss, _, _, acc, _ = criterion(output, aff_mat, ph_mask, cam_mask)

                val_loss += loss.item()
                val_acc += acc.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_acc /= max(val_batches, 1)

        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        print(f"Epoch {epoch}/{config.epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}",
              flush=True)

        scheduler.step()

        # Checkpoint — keep last 5 + best
        if epoch % config.save_every == 0:
            ckpt_name = f"fold{config.fold}_epoch{epoch}.pth"
            ckpt_path = config.checkpoint_dir / ckpt_name
            torch.save(model, ckpt_path)
            print(f"Saved: {ckpt_path}")

            # Rotate: keep only last 5 regular checkpoints
            recent_checkpoints.append(ckpt_name)
            while len(recent_checkpoints) > 5:
                old_name = recent_checkpoints.pop(0)
                old_path = config.checkpoint_dir / old_name
                if old_path.exists():
                    old_path.unlink()

        if val_acc > best_acc and val_acc < 1.0:
            # Remove old best
            if best_checkpoint:
                old_path = config.checkpoint_dir / best_checkpoint
                if old_path.exists():
                    old_path.unlink()
            best_acc = val_acc
            best_checkpoint = f"fold{config.fold}_best_epoch{epoch}_acc{val_acc:.4f}.pth"
            torch.save(model, config.checkpoint_dir / best_checkpoint)
            print(f"New best: {best_checkpoint}")

    writer.close()
    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
