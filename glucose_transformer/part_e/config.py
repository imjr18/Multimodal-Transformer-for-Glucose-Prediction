"""Configuration for Part E interpretability analyses."""

from __future__ import annotations

from pathlib import Path

from part_d.config import get_runtime_config as get_part_d_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PART_E_OVERRIDES = {
    "attention_batch_size": 8,
    "rollout_batch_size": 1,
    "ig_batch_size": 1,
    "ig_n_steps": 50,
    "probing_batch_size": 32,
    "probing_max_train_windows": 1500,
    "probing_max_test_windows": 600,
    "analysis_max_windows_per_user": 96,
    "scenario_matches_per_condition": 10,
    "head_analysis_max_windows": 256,
    "spurious_batch_size": 8,
    "spurious_epochs": 5,
    "spurious_learning_rate": 5e-5,
    "spurious_train_max_windows": 2000,
    "spurious_val_max_windows": 500,
    "spurious_test_max_windows": 500,
    "results_dir_part_e": str(PROJECT_ROOT / "part_e" / "results"),
    "figures_dir_part_e": str(PROJECT_ROOT / "part_e" / "figures"),
    "checkpoint_dir_part_e": str(PROJECT_ROOT / "part_e" / "checkpoints"),
    "ig_scenarios_dir": str(PROJECT_ROOT / "part_e" / "figures" / "ig_scenarios"),
    "attention_rollout_json_path": str(PROJECT_ROOT / "part_e" / "results" / "attention_rollout.json"),
    "attention_rollout_plot_path": str(PROJECT_ROOT / "part_e" / "figures" / "attention_rollout.png"),
    "ig_scenario_summary_path": str(PROJECT_ROOT / "part_e" / "results" / "ig_scenario_summary.csv"),
    "ig_scenario_results_path": str(PROJECT_ROOT / "part_e" / "results" / "ig_scenarios.json"),
    "probing_results_path": str(PROJECT_ROOT / "part_e" / "results" / "probing_results.json"),
    "probing_plot_path": str(PROJECT_ROOT / "part_e" / "figures" / "probing_curves.png"),
    "spurious_results_path": str(PROJECT_ROOT / "part_e" / "results" / "spurious_correlation.json"),
    "spurious_checkpoint_path": str(PROJECT_ROOT / "part_e" / "checkpoints" / "noise_model.pt"),
    "head_specialisation_results_path": str(PROJECT_ROOT / "part_e" / "results" / "head_specialisation.json"),
    "head_specialisation_plot_path": str(PROJECT_ROOT / "part_e" / "figures" / "head_specialisation.png"),
    "final_report_path": str(PROJECT_ROOT / "part_e" / "FINAL_REPORT.md"),
    "part_e_summary_path": str(PROJECT_ROOT / "part_e" / "results" / "part_e_summary.json"),
}


def get_runtime_config(*, no_cuda: bool = False) -> dict:
    """Return the Part E runtime configuration layered on top of Part D."""

    config = get_part_d_runtime_config(no_cuda=no_cuda)
    config.update(PART_E_OVERRIDES)
    config["device"] = "cpu" if no_cuda else config["device"]
    config["pin_memory"] = False if config["device"] == "cpu" else config["pin_memory"]
    return config

