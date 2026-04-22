"""Configuration for Part C efficient multimodal glucose forecasting."""

from __future__ import annotations

from pathlib import Path

from part_b.config import get_runtime_config as get_part_b_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PART_C_OVERRIDES = {
    # Hardware
    "batch_size": 8,
    "mixed_precision": True,
    "gradient_accumulation_steps": 4,
    "eeg_gradient_checkpointing": True,

    # EEG
    "eeg_sfreq": 256,
    "eeg_window_seconds": 120,
    "eeg_samples": 120 * 256,
    "eeg_band_window_seconds": 1,
    "eeg_band_tokens": 120,
    "eeg_patch_size": 64,
    "eeg_patch_tokens": 480,
    "eeg_local_window_seconds": 5,
    "eeg_local_window_samples": 5 * 256,
    "eeg_local_windows": 24,
    "eeg_local_patch_size": 64,
    "eeg_local_patches": 20,
    "local_d_model": 32,
    "eeg_feature_dim": 1,
    "cbf_feature_dim": 1,

    # Benchmarking
    "vram_budget_mb": 6144.0,
    "comfortable_vram_mb": 5500.0,
    "benchmark_warmup_batches": 1,

    # Paths
    "results_dir_part_c": str(PROJECT_ROOT / "part_c" / "results"),
    "figures_dir_part_c": str(PROJECT_ROOT / "part_c" / "figures"),
    "checkpoint_dir_part_c": str(PROJECT_ROOT / "part_c" / "checkpoints"),
    "part_c_processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "part_c_train_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_c_train_windows.pt"),
    "part_c_val_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_c_val_windows.pt"),
    "part_c_test_windows_path": str(PROJECT_ROOT / "data" / "processed" / "part_c_test_windows.pt"),
    "part_c_norm_stats_path": str(PROJECT_ROOT / "data" / "processed" / "part_c_normalisation_stats.json"),
    "part_c_manifest_path": str(PROJECT_ROOT / "data" / "processed" / "part_c_manifest.json"),
    "frequency_checkpoint_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "best_frequency_eeg.pt"),
    "patch_checkpoint_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "best_patch_eeg.pt"),
    "hierarchical_checkpoint_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "best_hierarchical_eeg.pt"),
    "frequency_history_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "frequency_history.json"),
    "patch_history_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "patch_history.json"),
    "hierarchical_history_path": str(PROJECT_ROOT / "part_c" / "checkpoints" / "hierarchical_history.json"),
    "benchmark_csv_path": str(PROJECT_ROOT / "part_c" / "results" / "benchmark.csv"),
    "sleep_stage_attention_path": str(PROJECT_ROOT / "part_c" / "figures" / "sleep_stage_attention.png"),
    "part_c_summary_path": str(PROJECT_ROOT / "part_c" / "results" / "part_c_summary.json"),
}


def get_runtime_config(*, no_cuda: bool = False) -> dict:
    """Return the Part C runtime configuration layered on top of Part B."""

    config = get_part_b_runtime_config(no_cuda=no_cuda)
    config.update(PART_C_OVERRIDES)
    config["device"] = "cpu" if no_cuda else config["device"]
    config["pin_memory"] = False if config["device"] == "cpu" else config["pin_memory"]
    config["amp_enabled"] = bool(config["mixed_precision"] and config["device"] == "cuda")
    return config
