"""Feature ablation settings for Vi-Fi experiments."""

ABLATION_SPECS = {
    "full": {
        "camera_indices": [0, 1, 2],
        "phone_indices": list(range(11)),
        "window_size": None,
        "description": "camera x/y/depth + FTM + IMU, 10-frame sequence",
    },
    "distance_only": {
        "camera_indices": [2],
        "phone_indices": [0],
        "window_size": None,
        "description": "camera depth + FTM range, 10-frame sequence",
    },
    "no_distance": {
        "camera_indices": [0, 1],
        "phone_indices": list(range(2, 11)),
        "window_size": None,
        "description": "camera x/y + IMU only, no camera depth or FTM",
    },
    "ftm_only": {
        "camera_indices": [0, 1, 2],
        "phone_indices": [0, 1],
        "window_size": None,
        "description": "camera x/y/depth + FTM range/std, no IMU",
    },
    "single_frame_distance": {
        "camera_indices": [2],
        "phone_indices": [0],
        "window_size": 1,
        "description": "current-frame camera depth + current-frame FTM range",
    },
}


def configure_ablation(config, ablation):
    """Attach feature-selection settings to a Config instance."""
    if ablation not in ABLATION_SPECS:
        valid = ", ".join(sorted(ABLATION_SPECS))
        raise ValueError(f"Unknown ablation '{ablation}'. Valid options: {valid}")

    spec = ABLATION_SPECS[ablation]
    config.ablation = ablation
    config.camera_feature_indices = list(spec["camera_indices"])
    config.phone_feature_indices = list(spec["phone_indices"])
    config.camera_feat_dim = len(config.camera_feature_indices)
    config.phone_feat_dim = len(config.phone_feature_indices)
    if spec["window_size"] is not None:
        config.window_size = spec["window_size"]
    return config
