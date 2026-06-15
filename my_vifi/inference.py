"""Offline inference with consistency voting for Vi-Fi deep affinity model.

Implements the paper's offline evaluation protocol:
1. Frame-by-frame affinity prediction with Hungarian bipartite matching
2. 30-frame consistency voting to refine per-tracklet assignments
"""
import sys
sys.path.insert(0, ".")

import argparse
import pickle
import re
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from model import MultimodalNetwork
from dataset import ViFiDataset, compute_norm_stats
from loss import AffinityEvaluator
from ablations import ABLATION_SPECS, configure_ablation


def hungarian_match(affinity_mat, valid_phone, valid_camera):
    """Run Hungarian algorithm on affinity matrix for 1:1 matching.

    Returns phone→camera assignment indices.
    """
    from scipy.optimize import linear_sum_assignment

    # Extract valid submatrix
    aff = affinity_mat[valid_phone][:, valid_camera]
    if aff.size == 0:
        return []

    # Convert to numpy for Hungarian
    aff_np = aff.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-aff_np)

    # Map back to original indices
    phone_indices = [valid_phone[i] for i in row_ind]
    camera_indices = [valid_camera[i] for i in col_ind]
    return list(zip(phone_indices, camera_indices))


def find_best_checkpoint(checkpoint_dir, fold):
    """Pick the best checkpoint for a fold by parsing acc from file names."""
    pattern = f"fold{fold}_best_epoch*_acc*.pth"
    best = []
    for path in checkpoint_dir.glob(pattern):
        match = re.search(r"_acc([0-9.]+)\.pth$", path.name)
        if match:
            best.append((float(match.group(1)), path.stat().st_mtime, path))

    if best:
        return max(best, key=lambda item: (item[0], item[1]))[2]

    regular = sorted(
        checkpoint_dir.glob(f"fold{fold}_epoch*.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    return regular[-1] if regular else None


def inference_with_voting(model, dataloader, device, config, vote_window=30):
    """Run inference with per-tracklet consistency voting.

    Returns:
        frame_results: list of per-frame accuracy dicts
        voted_accuracy: accuracy after voting
        raw_accuracy: accuracy without voting
    """
    model.eval()
    evaluator = AffinityEvaluator(config.Nm_phone, config.Nm_camera)

    # Tracklet assignment history: tracklet_id → [(frame_idx, voted_assignment, ground_truth)]
    tracklet_history = defaultdict(list)

    correct_raw = 0
    total_raw = 0
    correct_voted = 0
    total_voted = 0

    with torch.no_grad():
        for frame_idx, batch in enumerate(dataloader):
            cam = batch["camera"].to(device)
            ph = batch["phone"].to(device)
            cam_mask = batch["camera_mask"].to(device)
            ph_mask = batch["phone_mask"].to(device)
            aff_mat = batch["aff_mat"].to(device)

            # Forward pass
            output = model(cam, ph, cam_mask, ph_mask)

            # Evaluate with Hungarian matching
            B = output.size(0)
            for b in range(B):
                pred = output[b:b+1]
                target = aff_mat[b:b+1]
                pm = ph_mask[b]
                cm = cam_mask[b]

                # Softmax
                pred_sq = pred[0, 0]  # (Np+1, Nc+1)

                # Row-wise (phone→camera)
                mask_pc = torch.ones_like(pred_sq)
                mask_pc[config.Nm_phone, :] = 0
                pred_pc = mask_pc * pred_sq
                pred_pc = torch.softmax(pred_pc, dim=1)

                # Col-wise (camera→phone)
                mask_cp = torch.ones_like(pred_sq)
                mask_cp[:, config.Nm_camera] = 0
                pred_cp = mask_cp * pred_sq
                pred_cp = torch.softmax(pred_cp, dim=0)

                # Average both directions
                sub = pred_pc[:config.Nm_phone, :config.Nm_camera]
                sub_cp = pred_cp[:config.Nm_phone, :config.Nm_camera]
                averaged = (sub + sub_cp) / 2.0
                pred_pc[:config.Nm_phone, :config.Nm_camera] = averaged

                # Hungarian matching
                valid_p = pm.bool().nonzero(as_tuple=False).squeeze(-1)
                valid_c = cm.bool().nonzero(as_tuple=False).squeeze(-1)
                # Remove extra row/col from valid sets
                valid_p = [v.item() for v in valid_p if v < config.Nm_phone]
                valid_c = [v.item() for v in valid_c if v < config.Nm_camera]

                # Get raw prediction (argmax)
                _, pred_idx = pred_pc[:config.Nm_phone, :config.Nm_camera].max(dim=1)
                _, target_idx = target[0, :config.Nm_phone, :config.Nm_camera].max(dim=1)

                # Hungarian matching for refined assignment
                matches = hungarian_match(pred_pc[:config.Nm_phone, :config.Nm_camera],
                                          valid_p, valid_c)

                # Track per-phone assignment
                for p_idx, c_idx in matches:
                    # Get ground truth camera for this phone
                    gt_cam = target_idx[p_idx].item() if target[0, p_idx].sum() > 0 else -1

                    # Update tracklet history (using camera_idx as tracklet proxy)
                    tracklet_history[p_idx].append((frame_idx, c_idx, gt_cam))

                    # Raw accuracy: argmax match
                    raw_pred = pred_idx[p_idx].item()
                    if raw_pred == gt_cam and gt_cam < config.Nm_camera:
                        correct_raw += 1
                    total_raw += 1

    # Apply consistency voting
    from copy import deepcopy
    updated_history = deepcopy(tracklet_history)

    for phone_id, assignments in tracklet_history.items():
        for i in range(len(assignments)):
            frame_idx, current_pred, gt = assignments[i]

            # Get vote window (last `vote_window` assignments before this frame)
            if i >= vote_window:
                window = [a[1] for a in assignments[i - vote_window:i]]
            else:
                window = [a[1] for a in assignments[:vote_window]]

            # Majority vote
            vote_counts = Counter(window)
            winner = vote_counts.most_common(1)[0][0]

            if vote_counts[winner] > vote_counts.get(current_pred, 0):
                updated_history[phone_id][i] = (frame_idx, winner, gt)

    # Compute voted accuracy
    for phone_id, assignments in updated_history.items():
        for frame_idx, pred, gt in assignments:
            if pred == gt and gt < config.Nm_camera:
                correct_voted += 1
            total_voted += 1

    raw_acc = correct_raw / max(total_raw, 1)
    voted_acc = correct_voted / max(total_voted, 1)

    return raw_acc, voted_acc


def main():
    parser = argparse.ArgumentParser(description="Vi-Fi offline inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint path; defaults to best checkpoint for the fold")
    parser.add_argument("--fold", type=int, default=None,
                        help="fold index; defaults to Config.fold")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=sorted(ABLATION_SPECS.keys()))
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="must match training validation split for normalization")
    parser.add_argument("--val_seed", type=int, default=None,
                        help="must match training validation split for normalization")
    args = parser.parse_args()

    config = Config()
    configure_ablation(config, args.ablation)
    if args.fold is not None:
        config.fold = args.fold
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.val_seed is not None:
        config.val_seed = args.val_seed
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablation: {config.ablation}")
    print(f"Validation split used for normalization: ratio={config.val_ratio}, seed={config.val_seed}")

    # Load best model
    checkpoint_dir = config.checkpoint_dir / config.ablation
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(checkpoint_dir, config.fold)
    if checkpoint_path is None:
        print("No checkpoint found!")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.eval()

    # Compute normalization stats and load test dataset
    norm_stats = compute_norm_stats(config.data_root, config=config)
    test_ds = ViFiDataset(config.data_root, split="test", fold=config.fold,
                          config=config, norm_stats=norm_stats)

    if len(test_ds) == 0:
        print("Empty test dataset!")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Run inference with voting
    print("Running inference...")
    raw_acc, voted_acc = inference_with_voting(
        model, test_loader, device, config, vote_window=30
    )

    print(f"\n=== Results (Fold {config.fold}) ===")
    print(f"Online accuracy (no voting):   {raw_acc*100:.2f}%")
    print(f"Offline accuracy (30-frame vote): {voted_acc*100:.2f}%")

    # Compare with paper
    print(f"\n=== Paper Reference ===")
    print(f"Vi-Fi online:  ~81%")
    print(f"Vi-Fi offline: ~90%")


if __name__ == "__main__":
    main()
