import pickle
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class ViFiDataset(Dataset):
    """Loads RAN4model_dfv4p4 and builds sliding-window training samples.

    Each sample:
        camera_input: (Nm_camera, k*3 + 1)  — BBX (x,y,depth) × k frames + valid_len
        phone_input:  (Nm_phone, k*11 + 1)  — FTM(2)+IMU(9) × k frames + valid_len
        camera_mask:  (Nm_camera+1,)          — valid entity flags
        phone_mask:   (Nm_phone+1,)           — valid entity flags
        aff_mat:      (Nm_phone+1, Nm_camera+1) — ground truth affinity
    """

    def __init__(self, data_root, split="train", fold=1, config=None,
                 norm_stats=None):
        self.data_root = Path(data_root)
        self.split = split
        self.fold = fold
        self.k = config.window_size if config else 10
        self.Np = config.Nm_phone if config else 5
        self.Nc = config.Nm_camera if config else 15
        self.camera_indices = list(getattr(config, "camera_feature_indices", [0, 1, 2]))
        self.phone_indices = list(getattr(config, "phone_feature_indices", range(11)))
        self.camera_feat_dim = len(self.camera_indices)
        self.phone_feat_dim = len(self.phone_indices)
        self.shuffle_aug = (split == "train")
        self.max_others = self.Nc - 3
        self.val_ratio = float(getattr(config, "val_ratio", 0.1)) if config else 0.1
        self.val_seed = int(getattr(config, "val_seed", 2026)) if config else 2026

        # Feature normalization stats: {cam_mean, cam_std, ph_mean, ph_std}
        self.norm_stats = norm_stats

        # Test sequences for fold splitting. Supports both:
        #   ["seq_a", "seq_b"] for legacy leave-one-sequence folds, and
        #   [["seq_a", "seq_b"]] for author-style full fold test sets.
        self.test_sequences = config.test_sequences if config else [
            [
                "20210907_145202", "20211004_142306",
                "20211006_152208", "20211007_105924",
                "20211007_134632",
            ],
        ]

        self.sequences = self._discover_sequences()
        self.samples = self._build_sample_index()

    def _discover_sequences(self):
        """Scan dataset directory and return list of sequence metadata."""
        test_set = self._get_test_set()

        sequences = []
        seqs_dir = self.data_root / "seqs"

        # Indoor: scene0
        indoor_dir = seqs_dir / "indoor" / "scene0"
        if indoor_dir.exists():
            for seq_dir in sorted(indoor_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                sync_dir = seq_dir / "sync_ts16_dfv4p4"
                if sync_dir.exists():
                    sequences.append({
                        "path": sync_dir,
                        "name": seq_dir.name,
                        "is_indoor": True,
                        "is_test": seq_dir.name in test_set,
                    })

        # Outdoor: scene1-scene4
        outdoor_dir = seqs_dir / "outdoor"
        if outdoor_dir.exists():
            for scene_dir in sorted(outdoor_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                for seq_dir in sorted(scene_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    sync_dir = seq_dir / "sync_ts16_dfv4p4"
                    if sync_dir.exists():
                        sequences.append({
                            "path": sync_dir,
                            "name": seq_dir.name,
                            "is_indoor": False,
                            "is_test": seq_dir.name in test_set,
                        })

        print(f"Found {len(sequences)} sequences "
              f"({sum(1 for s in sequences if s['is_indoor'])} indoor, "
              f"{sum(1 for s in sequences if not s['is_indoor'])} outdoor)",
              flush=True)

        found_names = {s["name"] for s in sequences}
        missing_test = sorted(test_set - found_names)
        if missing_test:
            print(f"WARNING: configured test sequences not found: {missing_test}",
                  flush=True)

        val_set = self._get_validation_set(sequences, test_set)
        if val_set:
            print(f"Validation sequences ({len(val_set)}): {sorted(val_set)}",
                  flush=True)

        # Filter by split. The official fold test set is kept isolated:
        # train excludes both test and validation; validation is selected only
        # from non-test sequences and is used for checkpoint selection.
        if self.split == "train":
            sequences = [
                s for s in sequences
                if not s["is_test"] and s["name"] not in val_set
            ]
        elif self.split in {"val", "valid", "validation"}:
            sequences = [
                s for s in sequences
                if not s["is_test"] and s["name"] in val_set
            ]
        elif self.split == "test":
            sequences = [s for s in sequences if s["is_test"]]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        print(f"Using {len(sequences)} sequences for {self.split} (fold {self.fold})",
              flush=True)
        return sequences

    def _get_test_set(self):
        """Return the set of test sequence names for the requested fold."""
        if not (1 <= self.fold <= len(self.test_sequences)):
            return set()

        fold_sequences = self.test_sequences[self.fold - 1]
        if isinstance(fold_sequences, str):
            return {fold_sequences}
        return set(fold_sequences)

    def _get_validation_set(self, sequences, test_set):
        """Choose a deterministic sequence-level validation set from train sequences."""
        candidates = sorted({s["name"] for s in sequences if s["name"] not in test_set})
        if self.val_ratio <= 0 or not candidates:
            return set()

        n_val = max(1, int(round(len(candidates) * self.val_ratio)))
        n_val = min(n_val, max(len(candidates) - 1, 1))
        rng = np.random.default_rng(self.val_seed + self.fold)
        chosen = rng.choice(candidates, size=n_val, replace=False)
        return set(chosen.tolist())

    def _load_sequence_data(self, seq):
        """Load all PKL data for a sequence. Returns dict with pre-loaded arrays."""
        sync_dir = seq["path"]
        is_indoor = seq["is_indoor"]

        # Camera: BBX5 (outdoor) or BBX5H (indoor) — (T, Ns, 1, 5)
        bbx_name = "BBX5H_sync_dfv4p4.pkl" if is_indoor else "BBX5_sync_dfv4p4.pkl"
        with open(sync_dir / bbx_name, "rb") as f:
            bbx = pickle.load(f)
        T, Ns, _, _ = bbx.shape
        # Squeeze the DIM_PER_FRAME dimension → (T, Ns, 5)
        camera_data = bbx[:, :, 0, :].astype(np.float32)  # (T, Ns, 5)

        # Phone: FTM_li (T, Ns, 1, 2) + IMUagm9 (T, Ns, 1, 9) → 11 features
        with open(sync_dir / "FTM_li_sync_dfv4p4.pkl", "rb") as f:
            ftm = pickle.load(f)[:, :, 0, :].astype(np.float32)  # (T, Ns, 2)
        with open(sync_dir / "IMUagm9_sync_dfv4p4.pkl", "rb") as f:
            imu = pickle.load(f)[:, :, 0, :].astype(np.float32)  # (T, Ns, 9)
        phone_data = np.concatenate([ftm, imu], axis=-1)  # (T, Ns, 11)

        # Others (passersby): only outdoor
        others_data = None
        others_path = sync_dir / "BBX5_Others_sync_dfv4p4.pkl"
        if not is_indoor and others_path.exists():
            with open(others_path, "rb") as f:
                others = pickle.load(f)[:, :, 0, :].astype(np.float32)  # (T, No, 5)
            others_data = others

        # Remove NaN frames — replace NaN with 0 for later valid-length counting
        camera_nan = np.isnan(camera_data)
        phone_nan = np.isnan(phone_data)
        camera_data[camera_nan] = 0.0
        phone_data[phone_nan] = 0.0
        if others_data is not None:
            others_data[np.isnan(others_data)] = 0.0

        # Valid mask per frame per subject
        # A subject is valid if ANY feature is non-zero (non-NaN originally)
        camera_valid = (~camera_nan).any(axis=-1)  # (T, Ns)
        phone_valid = (~phone_nan).any(axis=-1)     # (T, Ns)

        return {
            "camera": camera_data,      # (T, Ns, 5)
            "phone": phone_data,        # (T, Ns, 11)
            "others": others_data,      # (T, No, 5) or None
            "camera_valid": camera_valid,  # (T, Ns)
            "phone_valid": phone_valid,    # (T, Ns)
            "T": T,
            "Ns": Ns,
            "name": seq["name"],
        }

    def _build_sample_index(self):
        """Pre-load all sequences and build flat sample index."""
        samples = []
        for seq_idx, seq in enumerate(self.sequences):
            if seq_idx % 10 == 0:
                print(f"  Loading sequence {seq_idx+1}/{len(self.sequences)}...",
                      flush=True)
            data = self._load_sequence_data(seq)
            T = data["T"]

            # Each frame t >= k-1 is a valid sample
            for t in range(self.k - 1, T):
                # Skip frames where ALL subjects are invalid in camera
                if not data["camera_valid"][t].any():
                    continue
                samples.append((seq_idx, t))

            # Store loaded data in sequence dict for fast access
            self.sequences[seq_idx]["_data"] = data

        print(f"Total samples: {len(samples)}", flush=True)
        return samples

    def __len__(self):
        return len(self.samples)

    def _build_window(self, data, t, feat_dim):
        """
        Build window of k frames ending at t for all subjects.
        Returns (N_subjects, k * feat_dim + 1) concatenated feature vector
        where last column is valid sequence length.

        NaN frames are shifted to the beginning (padded), valid frames at the end.
        """
        T, Ns = data.shape[0], data.shape[1]
        window = np.zeros((Ns, self.k * feat_dim), dtype=np.float32)
        valid_lens = np.zeros(Ns, dtype=np.float32)

        for s in range(Ns):
            seq = data[max(0, t - self.k + 1):t + 1, s, :]  # (w, D)
            # Find valid (non-zero after NaN→0 replacement) frames
            valid_mask = (np.abs(seq).sum(axis=-1) > 1e-6).astype(np.float32)
            n_valid = int(valid_mask.sum())

            if n_valid > 0:
                # Take last n_valid non-zero frames
                valid_seq = seq[valid_mask > 0.5][-self.k:]
                n_actual = valid_seq.shape[0]
                # Left-align: valid frames at start, padded at end (for pack_padded_sequence)
                window[s, :n_actual * feat_dim] = valid_seq.reshape(-1)
                valid_lens[s] = n_actual

        # Append valid_len as last column
        window_with_len = np.concatenate([
            window, valid_lens.reshape(-1, 1)
        ], axis=1)  # (Ns, k*D + 1)

        return window_with_len

    def _get_visible_others(self, others_data, t):
        """Get passersby visible at frame t."""
        if others_data is None:
            return np.zeros((0, 5), dtype=np.float32), 0

        t_other = min(t, others_data.shape[0] - 1)  # others may have fewer frames
        frame = others_data[t_other]  # (No, 5)
        visible_mask = (np.abs(frame).sum(axis=-1) > 1e-6)
        visible = frame[visible_mask]
        n_others = min(visible.shape[0], self.max_others)
        return visible[:n_others], n_others

    def __getitem__(self, idx):
        seq_idx, t = self.samples[idx]
        data = self.sequences[seq_idx]["_data"]

        Ns = data["Ns"]
        camera_data = data["camera"]      # (T, Ns, 5)
        phone_data = data["phone"]        # (T, Ns, 11)

        # Build window features with valid lengths
        cam_window = self._build_window(camera_data, t, 5)    # (Ns, k*5 + 1)
        # Use only (x, y, depth) — indices 0,1,2 of 5 BBX features
        # Reconstruct: features are interleaved as f0,f1,f2,f3,f4, f0,f1,...
        cam_5d = cam_window[:, :-1].reshape(Ns, self.k, 5)  # (Ns, k, 5)
        cam_selected = cam_5d[:, :, self.camera_indices]
        cam_flat = cam_selected.reshape(Ns, self.k * self.camera_feat_dim)
        cam_feats = np.concatenate([cam_flat, cam_window[:, -1:]], axis=1)  # valid_len in frames

        ph_window = self._build_window(phone_data, t, 11)  # (Ns, k*11 + 1)
        ph_11d = ph_window[:, :-1].reshape(Ns, self.k, 11)
        ph_selected = ph_11d[:, :, self.phone_indices]
        ph_feats = np.concatenate([
            ph_selected.reshape(Ns, self.k * self.phone_feat_dim),
            ph_window[:, -1:],
        ], axis=1)

        # Get visible others at frame t
        others, n_others = self._get_visible_others(data["others"], t)

        # Build camera input: Ns legit subjects + n_others passersby
        cam_valid = np.array([True] * Ns)
        if n_others > 0:
            # Build window for others (same window as subjects)
            others_window = np.zeros((n_others, self.k * self.camera_feat_dim + 1), dtype=np.float32)
            # For simplicity, only use current frame for others
            others_selected = others[:n_others, self.camera_indices]
            for oi in range(n_others):
                start = (self.k - 1) * self.camera_feat_dim
                others_window[oi, start:start + self.camera_feat_dim] = others_selected[oi]
                others_window[oi, -1] = 1  # valid_len = 1 frame
            cam_feats = np.concatenate([cam_feats, others_window], axis=0)
            cam_valid = np.concatenate([cam_valid, np.array([True] * n_others)])

        # Pad camera to Nc
        n_cam_actual = cam_feats.shape[0]
        cam_pad = np.ones((self.Nc - n_cam_actual, self.k * self.camera_feat_dim + 1), dtype=np.float32)  # ones so valid_len=1
        cam_feats = np.concatenate([cam_feats, cam_pad], axis=0)
        cam_valid = np.concatenate([cam_valid, np.array([False] * (self.Nc - n_cam_actual))])

        # Pad phone to Np
        n_ph_actual = ph_feats.shape[0]
        ph_pad = np.ones((self.Np - n_ph_actual, self.k * self.phone_feat_dim + 1), dtype=np.float32)  # ones so valid_len=1
        ph_feats = np.concatenate([ph_feats, ph_pad], axis=0)
        ph_valid = np.array([True] * n_ph_actual + [False] * (self.Np - n_ph_actual))

        # Build affinity matrix: (Np+1, Nc+1)
        aff_mat = np.zeros((self.Np + 1, self.Nc + 1), dtype=np.float32)
        for i in range(min(Ns, self.Np)):
            if data["phone_valid"][t, i]:
                for j in range(min(Ns, self.Nc)):
                    if data["camera_valid"][t, j]:
                        if i == j:
                            aff_mat[i, j] = 1.0
                # If this phone has no camera match, route to extra column
                if aff_mat[i, :self.Nc].sum() == 0:
                    aff_mat[i, -1] = 1.0
        for j in range(n_cam_actual):
            if aff_mat[:self.Np, j].sum() == 0:
                aff_mat[-1, j] = 1.0  # others → extra row

        # Extend masks
        cam_mask = np.concatenate([cam_valid, np.array([True])])  # extra col always valid
        ph_mask = np.concatenate([ph_valid, np.array([True])])    # extra row always valid

        # Apply feature normalization (Z-score)
        if self.norm_stats is not None:
            # Normalize camera features (excluding valid_len column)
            cam_feat_3d = cam_feats[:, :-1].reshape(-1, self.k, self.camera_feat_dim)
            cam_feat_3d = (cam_feat_3d - self.norm_stats["cam_mean"]) / (self.norm_stats["cam_std"] + 1e-8)
            cam_feats[:, :-1] = cam_feat_3d.reshape(-1, self.k * self.camera_feat_dim)
            # Normalize phone features (excluding valid_len column)
            ph_feat_11d = ph_feats[:, :-1].reshape(-1, self.k, self.phone_feat_dim)
            ph_feat_11d = (ph_feat_11d - self.norm_stats["ph_mean"]) / (self.norm_stats["ph_std"] + 1e-8)
            ph_feats[:, :-1] = ph_feat_11d.reshape(-1, self.k * self.phone_feat_dim)

        # Data augmentation: shuffle entity order (training only)
        if self.shuffle_aug and random.random() < 0.8:
            cam_feats, cam_mask, aff_mat = self._shuffle_camera(cam_feats, cam_mask, aff_mat)
            ph_feats, ph_mask, aff_mat = self._shuffle_phone(ph_feats, ph_mask, aff_mat)

        return {
            "camera": torch.from_numpy(cam_feats).float(),
            "phone": torch.from_numpy(ph_feats).float(),
            "camera_mask": torch.from_numpy(cam_mask).bool(),
            "phone_mask": torch.from_numpy(ph_mask).bool(),
            "aff_mat": torch.from_numpy(aff_mat).float(),
        }

    def _shuffle_camera(self, cam_feats, cam_mask, aff_mat):
        """Randomly permute camera entity order and adjust affinity matrix."""
        n_valid = cam_mask[:-1].sum()
        if n_valid <= 1:
            return cam_feats, cam_mask, aff_mat

        perm = np.random.permutation(n_valid)
        full_perm = np.arange(self.Nc)
        full_perm[:n_valid] = perm
        full_perm_ext = np.concatenate([full_perm, np.array([self.Nc])])  # extra col stays

        cam_feats = cam_feats[full_perm]
        cam_mask[:-1] = cam_mask[:-1][full_perm]
        aff_mat = aff_mat[:, full_perm_ext]
        return cam_feats, cam_mask, aff_mat

    def _shuffle_phone(self, ph_feats, ph_mask, aff_mat):
        """Randomly permute phone entity order and adjust affinity matrix."""
        n_valid = ph_mask[:-1].sum()
        if n_valid <= 1:
            return ph_feats, ph_mask, aff_mat

        perm = np.random.permutation(n_valid)
        full_perm = np.arange(self.Np)
        full_perm[:n_valid] = perm
        full_perm_ext = np.concatenate([full_perm, np.array([self.Np])])

        ph_feats = ph_feats[full_perm]
        ph_mask[:-1] = ph_mask[:-1][full_perm]
        aff_mat = aff_mat[full_perm_ext, :]
        return ph_feats, ph_mask, aff_mat


def compute_norm_stats(data_root, config=None):
    """Compute per-feature mean and std from training data for Z-score normalization.

    Samples a subset of windows from training sequences to estimate statistics.
    """
    print("Computing feature normalization statistics...", flush=True)
    # Create dataset without norm to access raw features
    fold = config.fold if config else 1
    ds = ViFiDataset(data_root, split="train", fold=fold, config=config, norm_stats=None)

    cam_dim = len(getattr(config, "camera_feature_indices", [0, 1, 2])) if config else 3
    ph_dim = len(getattr(config, "phone_feature_indices", range(11))) if config else 11
    cam_sum = np.zeros(cam_dim, dtype=np.float64)
    cam_sq = np.zeros(cam_dim, dtype=np.float64)
    ph_sum = np.zeros(ph_dim, dtype=np.float64)
    ph_sq = np.zeros(ph_dim, dtype=np.float64)
    n_cam = 0
    n_ph = 0

    # Sample a subset (every 50th sample from each sequence)
    indices = list(range(0, len(ds), max(1, len(ds) // 2000)))
    print(f"  Sampling {len(indices)} windows from {len(ds)} total...", flush=True)

    for idx in indices:
        sample = ds[idx]
        # Camera: (Nc, k*3+1) — extract (N_cam_valid, k, 3) for valid entities
        cam = sample["camera"].numpy()
        cam_mask = sample["camera_mask"].numpy()
        cam_feat = cam[:, :-1].reshape(-1, ds.k, ds.camera_feat_dim)
        valid = cam_mask[:-1]  # (Nc,)
        valid_cam = cam_feat[valid]  # (N_valid, k, 3)
        # Only count non-padded frames (valid_len > 1 means not a padding subject)
        cam_lens = cam[:, -1]  # (Nc,)
        for i, v in enumerate(valid):
            if v and cam_lens[i] > 1:
                n_frames = int(min(cam_lens[i], ds.k))
                frames = cam_feat[i, :n_frames]  # (n_frames, 3)
                cam_sum += frames.sum(axis=0)
                cam_sq += (frames ** 2).sum(axis=0)
                n_cam += n_frames

        # Phone: (Np, k*11+1)
        ph = sample["phone"].numpy()
        ph_mask = sample["phone_mask"].numpy()
        ph_feat = ph[:, :-1].reshape(-1, ds.k, ds.phone_feat_dim)
        valid_p = ph_mask[:-1]
        ph_lens = ph[:, -1]
        for i, v in enumerate(valid_p):
            if v and ph_lens[i] > 1:
                n_frames = int(min(ph_lens[i], ds.k))
                frames = ph_feat[i, :n_frames]
                ph_sum += frames.sum(axis=0)
                ph_sq += (frames ** 2).sum(axis=0)
                n_ph += n_frames

    cam_mean = cam_sum / max(n_cam, 1)
    cam_std = np.sqrt(cam_sq / max(n_cam, 1) - cam_mean ** 2)
    ph_mean = ph_sum / max(n_ph, 1)
    ph_std = np.sqrt(ph_sq / max(n_ph, 1) - ph_mean ** 2)

    stats = {
        "cam_mean": cam_mean.astype(np.float32),
        "cam_std": cam_std.astype(np.float32),
        "ph_mean": ph_mean.astype(np.float32),
        "ph_std": ph_std.astype(np.float32),
    }

    print(f"  Camera mean: {stats['cam_mean']}", flush=True)
    print(f"  Camera std:  {stats['cam_std']}", flush=True)
    print(f"  Phone mean:  {stats['ph_mean']}", flush=True)
    print(f"  Phone std:   {stats['ph_std']}", flush=True)

    return stats
