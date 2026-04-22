"""Central configuration for Part A of the glucose forecasting project."""

from __future__ import annotations

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    # Reproducibility
    "seed": 42,

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "pin_memory": True,
    "dtype": torch.float32,
    "gradient_checkpointing": False,

    # Model
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 2,
    "d_ff": 256,
    "dropout": 0.1,
    "max_seq_len": 25,

    # Data
    "input_len": 24,
    "target_offsets": [6, 12],
    "resample_frequency": "5min",
    "ffill_limit": 3,
    "window_stride": 1,
    "train_patients": [559, 563, 570],
    "val_patients": [588],
    "test_patients": [575, 591],
    "ohio_year": "2018",

    # Training
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 200,
    "grad_clip_norm": 1.0,
    "adam_betas": (0.9, 0.98),
    "progress_print_every": 5,

    # Baseline
    "lstm_hidden_size": 64,
    "lstm_num_layers": 2,

    # Visualisation
    "attention_samples": 3,
    "figure_dpi": 150,
    "lag_zone_start_minutes": 30,
    "lag_zone_end_minutes": 45,

    # Paths
    "project_root": str(PROJECT_ROOT),
    "data_raw_dir": str(PROJECT_ROOT / "data" / "raw"),
    "data_processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "checkpoint_dir": str(PROJECT_ROOT / "part_a" / "checkpoints"),
    "figures_dir": str(PROJECT_ROOT / "part_a" / "figures"),
    "train_windows_path": str(PROJECT_ROOT / "data" / "processed" / "train_windows.pt"),
    "val_windows_path": str(PROJECT_ROOT / "data" / "processed" / "val_windows.pt"),
    "test_windows_path": str(PROJECT_ROOT / "data" / "processed" / "test_windows.pt"),
    "norm_stats_path": str(PROJECT_ROOT / "data" / "processed" / "normalisation_stats.json"),
    "transformer_checkpoint_path": str(PROJECT_ROOT / "part_a" / "checkpoints" / "best_model.pt"),
    "lstm_checkpoint_path": str(PROJECT_ROOT / "part_a" / "checkpoints" / "best_lstm_model.pt"),
    "history_path": str(PROJECT_ROOT / "part_a" / "checkpoints" / "training_history.json"),
    "lstm_history_path": str(PROJECT_ROOT / "part_a" / "checkpoints" / "lstm_training_history.json"),
    "comparison_path": str(PROJECT_ROOT / "part_a" / "checkpoints" / "model_comparison.json"),
}


def get_runtime_config(*, no_cuda: bool = False) -> dict:
    """Return a mutable config copy with optional runtime overrides."""

    runtime_config = dict(CONFIG)
    if no_cuda:
        runtime_config["device"] = "cpu"
        runtime_config["pin_memory"] = False
    return runtime_config
