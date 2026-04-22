"""Single entry point for the full Part E interpretability pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from part_e.attention_rollout import compute_attention_rollout, plot_temporal_importance_profile
from part_e.common import (
    build_analysis_windows,
    ensure_runtime_dirs,
    load_json,
    load_model_and_dataset,
    load_part_d_summary,
    make_single_window_batch,
    save_json,
)
from part_e.config import get_runtime_config
from part_e.head_specialisation import analyse_head_specialisation
from part_e.integrated_gradients import run_biological_scenarios
from part_e.probing_classifiers import train_probing_classifiers
from part_e.report_generator import generate_final_report
from part_e.spurious_correlation import run_spurious_correlation_test


def _rollout_to_json_ready(result: dict) -> dict:
    """Convert NumPy arrays in the rollout payload into lists."""

    payload = {}
    for key, value in result.items():
        if hasattr(value, "tolist"):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload


def _load_cross_archetype(config: dict):
    """Load the saved Part D cross-archetype table if it exists."""

    cross_path = Path(config["cross_archetype_csv_path"])
    if not cross_path.exists():
        return []
    return pd.read_csv(cross_path).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part E of the glucose Transformer project.")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU execution for all analyses")
    parser.add_argument("--force", action="store_true", help="recompute analyses even if cached artifacts exist")
    parser.add_argument("--skip_spurious", action="store_true", help="skip the expensive spurious-correlation control")
    args = parser.parse_args()

    config = get_runtime_config(no_cuda=args.no_cuda)
    ensure_runtime_dirs(config)

    gpu_model, dataset = load_model_and_dataset(config)
    cpu_model, _ = load_model_and_dataset(get_runtime_config(no_cuda=True))

    all_results = {
        "part_d_summary": load_part_d_summary(config),
        "cross_archetype": _load_cross_archetype(config),
    }

    test_windows = build_analysis_windows(
        dataset,
        split="test",
        max_windows_per_user=1,
    )
    if not test_windows:
        raise RuntimeError("No test windows were available for Part E analyses.")
    rollout_batch = make_single_window_batch(test_windows[0], device="cpu")

    rollout_json_path = Path(config["attention_rollout_json_path"])
    rollout_plot_path = Path(config["attention_rollout_plot_path"])
    if not args.force and rollout_json_path.exists() and rollout_plot_path.exists():
        attention_rollout = load_json(rollout_json_path)
    else:
        rollout_result = compute_attention_rollout(cpu_model, rollout_batch)
        plot_path = plot_temporal_importance_profile(rollout_result, config["attention_rollout_plot_path"])
        attention_rollout = _rollout_to_json_ready(rollout_result)
        attention_rollout["plot_path"] = plot_path
        save_json(attention_rollout, rollout_json_path)
    all_results["attention_rollout"] = attention_rollout

    ig_json_path = Path(config["ig_scenario_results_path"])
    ig_csv_path = Path(config["ig_scenario_summary_path"])
    if not args.force and ig_json_path.exists() and ig_csv_path.exists():
        ig_results = {
            "results": load_json(ig_json_path),
            "csv_path": str(ig_csv_path),
        }
    else:
        ig_payload = run_biological_scenarios(
            cpu_model,
            dataset,
            dataset.norm_stats,
            config=config,
        )
        ig_results = {
            "results": ig_payload["results"],
            "csv_path": str(config["ig_scenario_summary_path"]),
        }
    all_results["integrated_gradients"] = ig_results

    probing_json_path = Path(config["probing_results_path"])
    probing_plot_path = Path(config["probing_plot_path"])
    if not args.force and probing_json_path.exists() and probing_plot_path.exists():
        probing_results = load_json(probing_json_path)
    else:
        probing_results = train_probing_classifiers(
            cpu_model,
            dataset,
            dataset,
            config=config,
        )
    all_results["probing"] = probing_results

    head_json_path = Path(config["head_specialisation_results_path"])
    head_plot_path = Path(config["head_specialisation_plot_path"])
    if not args.force and head_json_path.exists() and head_plot_path.exists():
        head_results = load_json(head_json_path)
    else:
        head_results = analyse_head_specialisation(gpu_model, dataset, config=config)
    all_results["head_specialisation"] = head_results

    if args.skip_spurious:
        spurious_results = {"skipped": True}
    else:
        spurious_json_path = Path(config["spurious_results_path"])
        if not args.force and spurious_json_path.exists():
            spurious_results = load_json(spurious_json_path)
        else:
            spurious_results = run_spurious_correlation_test(
                gpu_model,
                dataset,
                dataset,
                dataset.norm_stats,
                config=config,
            )
    all_results["spurious_correlation"] = spurious_results

    report_path = generate_final_report(all_results, config["final_report_path"])
    all_results["final_report"] = report_path
    save_json(all_results, config["part_e_summary_path"])

    print("Final summary")
    print("-" * 84)
    print(f"Attention rollout: {config['attention_rollout_plot_path']}")
    print(f"IG scenario summary: {config['ig_scenario_summary_path']}")
    print(f"Probing curves: {config['probing_plot_path']}")
    print(f"Head specialisation: {config['head_specialisation_plot_path']}")
    if not args.skip_spurious:
        print(f"Spurious correlation: {config['spurious_results_path']}")
    print(f"Final report: {report_path}")


if __name__ == "__main__":
    main()
