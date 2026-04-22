"""Central configuration for the standalone non-invasive project."""

from __future__ import annotations

from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "pin_memory": True,
    "use_amp_inference": torch.cuda.is_available(),
    # Model
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 2,
    "d_ff": 256,
    "dropout": 0.15,
    "user_emb_dim": 16,
    "max_seq_len": 24,
    # Non-invasive specifics
    "mc_dropout_samples": 50,
    "uncertainty_threshold": 15.0,
    "uncertainty_acceptance_band": 15.0,
    "input_signals": ["hr", "ecg_features", "emg_features", "eeg_bands", "cbf"],
    "target": "glucose_current",
    "window_minutes": 30,
    "window_timesteps": 6,
    "timestep_minutes": 5,
    "eeg_band_names": ["delta", "theta", "alpha", "beta", "gamma"],
    # Calibration
    "n_calibration_readings": 3,
    "calibration_inner_lr": 0.01,
    "calibration_inner_steps": 3,
    # Training
    "batch_size": 16,
    "mc_dropout_batch_size": 8,
    "max_epochs": 80,
    "early_stopping_patience": 10,
    "learning_rate": 1e-4,
    "pretrained_encoder_lr": 1e-5,
    "weight_decay": 1e-4,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "adam_betas": (0.9, 0.98),
    "pretrained_freeze_epochs": 10,
    # Synthetic cohort defaults
    "synthetic_days_per_user": 4,
    "archetype_counts": {
        "athlete": 60,
        "sedentary": 80,
        "elderly": 50,
        "diabetic": 50,
    },
    "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    # Analysis
    "ig_steps": 50,
    "ig_windows_per_scenario": 10,
    "reliability_bins": 10,
    # Paths
    "data_dir": str(PACKAGE_ROOT / "data"),
    "raw_data_dir": str(PACKAGE_ROOT / "data" / "raw"),
    "processed_data_dir": str(PACKAGE_ROOT / "data" / "processed"),
    "synthetic_cohort_dir": str(PACKAGE_ROOT / "data" / "processed" / "synthetic_cohort"),
    "manifest_path": str(PACKAGE_ROOT / "data" / "processed" / "synthetic_cohort" / "cohort_manifest.csv"),
    "train_windows_path": str(PACKAGE_ROOT / "data" / "processed" / "train_windows.pt"),
    "val_windows_path": str(PACKAGE_ROOT / "data" / "processed" / "val_windows.pt"),
    "test_windows_path": str(PACKAGE_ROOT / "data" / "processed" / "test_windows.pt"),
    "norm_stats_path": str(PACKAGE_ROOT / "data" / "processed" / "norm_stats.json"),
    "checkpoint_dir": str(PACKAGE_ROOT / "checkpoints"),
    "eeg_pretrain_checkpoint": str(PACKAGE_ROOT / "checkpoints" / "eeg_encoder_pretrained.pt"),
    "ecg_pretrain_checkpoint": str(PACKAGE_ROOT / "checkpoints" / "ecg_encoder_pretrained.pt"),
    "model_checkpoint": str(PACKAGE_ROOT / "checkpoints" / "noninvasive_best.pt"),
    "figures_dir": str(PACKAGE_ROOT / "figures"),
    "results_dir": str(PACKAGE_ROOT / "results"),
    "reliability_diagram_path": str(PACKAGE_ROOT / "figures" / "reliability_diagram.png"),
    "noninvasive_attr_dir": str(PACKAGE_ROOT / "figures" / "noninvasive_attributions"),
    "noninvasive_attr_summary_path": str(PACKAGE_ROOT / "results" / "noninvasive_attribution_summary.csv"),
    "baseline_comparison_path": str(PACKAGE_ROOT / "results" / "baseline_comparison.csv"),
    "uncertainty_metrics_path": str(PACKAGE_ROOT / "results" / "uncertainty_metrics.json"),
    "finetune_history_path": str(PACKAGE_ROOT / "results" / "finetune_history.json"),
    "supervised_reference_path": "glucose_transformer/part_a/results/test_metrics.json",
}


def ensure_directories(config: dict | None = None) -> None:
    """Create the directory layout needed by the non-invasive pipeline."""

    active = CONFIG if config is None else config
    for key in [
        "data_dir",
        "raw_data_dir",
        "processed_data_dir",
        "synthetic_cohort_dir",
        "checkpoint_dir",
        "figures_dir",
        "results_dir",
        "noninvasive_attr_dir",
    ]:
        Path(active[key]).mkdir(parents=True, exist_ok=True)


__all__ = ["CONFIG", "PACKAGE_ROOT", "ensure_directories"]

