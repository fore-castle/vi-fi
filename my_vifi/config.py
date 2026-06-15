from pathlib import Path

class Config:
    # Data
    data_root = Path("d:/buaa_study/research/vi-fi/RAN4model_dfv4p4")

    # Model hyperparameters
    window_size = 10        # sliding window frames (~3s at 10fps)
    Nm_phone = 5            # max phone entities (indoor=5, outdoor=3)
    Nm_camera = 15          # max BBX entities (including passersby)
    lstm_hidden = 32        # LSTM hidden dimension
    lstm_layers = 2         # LSTM layers
    camera_feat_dim = 3     # BBX (x, y, depth) — indices [0,1,2] of BBX5
    phone_feat_dim = 11     # FTM_li (range, std) + IMUagm9 (accel×3, gyro×3, mag×3)
    camera_feature_indices = [0, 1, 2]
    phone_feature_indices = list(range(11))
    ablation = "full"
    false_constant = 1      # value for unmatched row/col in affinity matrix

    # Training
    batch_size = 64
    lr = 1e-3
    epochs = 200
    momentum = 0.9
    lr_milestone_ratio = 0.5  # halve LR at 50% of epochs
    val_ratio = 0.1           # split validation sequences from non-test sequences
    val_seed = 2026           # deterministic sequence-level validation split

    # Fold split (4 outdoor scenes → leave-one-scene-out)
    fold = 1
    # Fold split from the author's train_outdoor_test_split_v2/fold_info.txt.
    # Each inner list is the full test set for that fold.
    test_sequences = [
        [
            "20210907_145202",  # CAC spot1
            "20211004_142306",  # CAC spot2
            "20211006_152208",  # Busch
            "20211007_105924",  # Liv spot1
            "20211007_134632",  # Liv spot2
        ],
    ]

    # Output
    checkpoint_dir = Path("d:/buaa_study/research/vi-fi/my_vifi/checkpoints")
    log_dir = Path("d:/buaa_study/research/vi-fi/my_vifi/logs")
    save_every = 10

    # Device
    device = "cuda"
