"""Configuration for Part B multimodal glucose forecasting."""

from __future__ import annotations

from pathlib import Path

from part_a.config import get_runtime_config as get_part_a_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PART_B_OVERRIDES = {
    # Hardware
    "batch_size": 16,
    "mixed_precision": True,

    # Model
    "n_encoder_layers_per_modal": 2,
    "hr_feature_dim": 2,
    "ecg_feature_dim": 5,
    "emg_feature_dim": 2,
    "early_fusion_feature_dim": 9,
    "modality_dropout_p": 0.15,

    # Synthetic features
    "ecg_feature_names": ["sdnn", "rmssd", "lf_power", "hf_power", "lf_hf_ratio"],
    "emg_feature_names": ["rms_envelope", "zero_crossing_rate"],
    "synthetic_validation_min_abs_corr": 0.10,
    "synthetic_validation_alpha": 0.05,

    # Results
    "results_dir": str(PROJECT_ROOT / "part_b" / "results"),
    "checkpoint_dir_part_b": str(PROJECT_ROOT / "part_b" / "checkpoints"),
    "part_b_processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "part_b_train_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_b_train_windows.pt"),
    "part_b_val_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_b_val_windows.pt"),
    "part_b_test_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_b_test_windows.pt"),
    "part_b_norm_stats_path": str(PROJECT_ROOT / "data" / "processed" / "part_b_normalisation_stats.json"),
    "part_b_manifest_path": str(PROJECT_ROOT / "data" / "processed" / "part_b_manifest.json"),
    "synthetic_validation_path": str(PROJECT_ROOT / "part_b" / "results" / "synthetic_feature_validation.json"),
    "early_fusion_checkpoint_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "best_early_fusion.pt"),
    "late_fusion_checkpoint_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "best_late_fusion.pt"),
    "cross_attention_checkpoint_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "best_cross_attention.pt"),
    "early_fusion_history_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "early_fusion_history.json"),
    "late_fusion_history_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "late_fusion_history.json"),
    "cross_attention_history_path": str(PROJECT_ROOT / "part_b" / "checkpoints" / "cross_attention_history.json"),
    "fusion_comparison_csv_path": str(PROJECT_ROOT / "part_b" / "results" / "fusion_comparison.csv"),
    "ablation_results_csv_path": str(PROJECT_ROOT / "part_b" / "results" / "ablation_results.csv"),
    "cross_attention_heatmap_path": str(PROJECT_ROOT / "part_b" / "results" / "cross_attention_heatmap.png"),
    "modality_contribution_path": str(PROJECT_ROOT / "part_b" / "results" / "modality_contribution.png"),
    "part_b_summary_path": str(PROJECT_ROOT / "part_b" / "results" / "part_b_summary.json"),
}


def get_runtime_config(*, no_cuda: bool = False) -> dict:
    """Return the Part B runtime configuration layered on top of Part A."""

    config = get_part_a_runtime_config(no_cuda=no_cuda)
    config.update(PART_B_OVERRIDES)
    config["device"] = "cpu" if no_cuda else config["device"]
    config["pin_memory"] = False if config["device"] == "cpu" else config["pin_memory"]
    config["amp_enabled"] = bool(config["mixed_precision"] and config["device"] == "cuda")
    return config
