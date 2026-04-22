"""Configuration for Part D population generalisation experiments."""

from __future__ import annotations

from pathlib import Path

from part_c.config import get_runtime_config as get_part_c_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PART_D_OVERRIDES = {
    # Cohort
    "synthetic_users_per_archetype": {
        "athlete": 250,
        "sedentary": 350,
        "elderly": 200,
        "diabetic": 200,
    },
    "cohort_days": 7,
    "cohort_workers": 4,
    "support_set_size": 12,
    "query_set_size": 24,
    "train_split_ratio": 0.70,
    "val_split_ratio": 0.15,
    "test_split_ratio": 0.15,
    "eeg_stats_windows_per_user": 2,
    # User conditioning
    "user_embedding_dim": 16,
    "maml_first_order": True,
    "maml_inner_lr": 0.01,
    "maml_outer_lr": 1e-4,
    "maml_inner_steps": 1,
    "maml_meta_epochs": 50,
    "meta_batch_size": 4,
    "meta_steps_per_epoch": 48,
    "meta_val_tasks": 24,
    "meta_early_stopping_patience": 10,
    "part_d_eeg_encoder_kind": "frequency_eeg",
    # Paths
    "synthetic_cohort_dir": str(PROJECT_ROOT / "data" / "synthetic_cohort"),
    "synthetic_cohort_manifest_path": str(PROJECT_ROOT / "data" / "synthetic_cohort" / "cohort_manifest.csv"),
    "synthetic_cohort_norm_stats_path": str(PROJECT_ROOT / "data" / "synthetic_cohort" / "cohort_norm_stats.json"),
    "synthetic_cohort_split_path": str(PROJECT_ROOT / "data" / "synthetic_cohort" / "cohort_splits.json"),
    "results_dir_part_d": str(PROJECT_ROOT / "part_d" / "results"),
    "figures_dir_part_d": str(PROJECT_ROOT / "part_d" / "figures"),
    "checkpoint_dir_part_d": str(PROJECT_ROOT / "part_d" / "checkpoints"),
    "best_meta_checkpoint_path": str(PROJECT_ROOT / "part_d" / "checkpoints" / "best_meta_model.pt"),
    "meta_history_path": str(PROJECT_ROOT / "part_d" / "checkpoints" / "meta_history.json"),
    "adaptation_curve_path": str(PROJECT_ROOT / "part_d" / "figures" / "adaptation_curve.png"),
    "cross_archetype_csv_path": str(PROJECT_ROOT / "part_d" / "results" / "cross_archetype_results.csv"),
    "embedding_space_path": str(PROJECT_ROOT / "part_d" / "figures" / "embedding_space.png"),
    "real_world_results_path": str(PROJECT_ROOT / "part_d" / "results" / "real_world_results.csv"),
    "part_d_summary_path": str(PROJECT_ROOT / "part_d" / "results" / "part_d_summary.json"),
}


def get_runtime_config(*, no_cuda: bool = False) -> dict:
    """Return the Part D runtime configuration layered on top of Part C."""

    config = get_part_c_runtime_config(no_cuda=no_cuda)
    config.update(PART_D_OVERRIDES)
    config["device"] = "cpu" if no_cuda else config["device"]
    config["pin_memory"] = False if config["device"] == "cpu" else config["pin_memory"]
    config["amp_enabled"] = False
    config["learning_rate"] = config["maml_outer_lr"]
    return config

