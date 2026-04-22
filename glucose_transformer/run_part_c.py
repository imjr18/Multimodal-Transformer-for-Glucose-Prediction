"""Single entry point for the full Part C efficient multimodal pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from part_a.config import get_runtime_config as get_part_a_runtime_config
from part_a.dataset import create_dataloader
from part_a.train import count_parameters, load_checkpoint, set_seed
from part_b.config import get_runtime_config as get_part_b_runtime_config
from part_c.benchmark import (
    analyse_sleep_stage_attention,
    demonstrate_vanilla_attention_failure,
    run_efficiency_benchmark,
)
from part_c.config import get_runtime_config
from part_c.dataset import FullModalDataset, build_full_modal_processed_windows
from part_c.models.full_modal import FullModalTransformer
from part_c.train import train_full_modal_model
from preprocessing.ohio_preprocessor import preprocess_ohio_dataset
from preprocessing.synthetic_ecg_emg import build_multimodal_processed_windows


def _part_a_processed_ready(config: dict) -> bool:
    """Check that the Part A processed windows exist."""

    required_files = [
        config["train_windows_path"],
        config["val_windows_path"],
        config["test_windows_path"],
        config["norm_stats_path"],
    ]
    return all(Path(file_path).exists() for file_path in required_files)


def _part_b_processed_ready(config: dict) -> bool:
    """Check that the Part B multimodal windows exist."""

    required_files = [
        config["part_b_train_windows_path"],
        config["part_b_val_windows_path"],
        config["part_b_test_windows_path"],
        config["part_b_norm_stats_path"],
    ]
    return all(Path(file_path).exists() for file_path in required_files)


def _part_c_processed_ready(config: dict) -> bool:
    """Check that the Part C full-modal windows exist."""

    required_files = [
        config["part_c_train_windows_path"],
        config["part_c_val_windows_path"],
        config["part_c_test_windows_path"],
        config["part_c_norm_stats_path"],
    ]
    return all(Path(file_path).exists() for file_path in required_files)


def _ensure_runtime_dirs(config: dict) -> None:
    """Create the output directories required by Part C."""

    for directory_key in ["results_dir_part_c", "figures_dir_part_c", "checkpoint_dir_part_c", "part_c_processed_dir"]:
        Path(config[directory_key]).mkdir(parents=True, exist_ok=True)


def _load_norm_stats(config: dict) -> dict:
    """Load the saved Part C normalisation statistics."""

    return json.loads(Path(config["part_c_norm_stats_path"]).read_text(encoding="utf-8"))


def _build_dataloaders(config: dict):
    """Load Part C datasets and wrap them in dataloaders."""

    train_dataset = FullModalDataset(config["part_c_train_windows_path"])
    val_dataset = FullModalDataset(config["part_c_val_windows_path"])
    test_dataset = FullModalDataset(config["part_c_test_windows_path"])

    dataloaders = {
        "train": create_dataloader(train_dataset, config, shuffle=True),
        "val": create_dataloader(val_dataset, config, shuffle=False),
        "test": create_dataloader(test_dataset, config, shuffle=False),
    }
    return train_dataset, val_dataset, test_dataset, dataloaders


def _model_specs(config: dict) -> list[tuple[str, str, str, str]]:
    """Return the checkpoint and history paths for the three EEG variants."""

    return [
        ("frequency_eeg", config["frequency_checkpoint_path"], config["frequency_history_path"], "FullModalTransformer[frequency_eeg]"),
        ("patch_eeg", config["patch_checkpoint_path"], config["patch_history_path"], "FullModalTransformer[patch_eeg]"),
        ("hierarchical_eeg", config["hierarchical_checkpoint_path"], config["hierarchical_history_path"], "FullModalTransformer[hierarchical_eeg]"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part C of the glucose Transformer project.")
    parser.add_argument("--preprocess_only", action="store_true", help="run only preprocessing")
    parser.add_argument("--eval_only", action="store_true", help="skip training and evaluate checkpoints only")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU execution for debugging")
    args = parser.parse_args()

    config = get_runtime_config(no_cuda=args.no_cuda)
    _ensure_runtime_dirs(config)
    set_seed(config["seed"])

    failure_report = demonstrate_vanilla_attention_failure(config)

    if not _part_a_processed_ready(config):
        print("Part A processed data not found. Running Part A preprocessing first.")
        preprocess_ohio_dataset(get_part_a_runtime_config(no_cuda=args.no_cuda))

    if not _part_b_processed_ready(config):
        print("Part B processed data not found. Building Part B multimodal windows first.")
        build_multimodal_processed_windows(get_part_b_runtime_config(no_cuda=args.no_cuda))

    if not _part_c_processed_ready(config) or args.preprocess_only:
        print("Building Part C full-modal processed windows.")
        manifest = build_full_modal_processed_windows(config)
        print(json.dumps(manifest["splits"], indent=2))
    else:
        manifest_path = Path(config["part_c_manifest_path"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "splits": {
                    "train": {"windows_path": config["part_c_train_windows_path"]},
                    "val": {"windows_path": config["part_c_val_windows_path"]},
                    "test": {"windows_path": config["part_c_test_windows_path"]},
                }
            }

    if args.preprocess_only:
        print("Part C preprocessing complete.")
        return

    norm_stats = _load_norm_stats(config)
    train_dataset, val_dataset, test_dataset, dataloaders = _build_dataloaders(config)

    print(f"Using device: {config['device']}")
    print(
        f"Dataset sizes -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise RuntimeError(
            "At least one Part C split has zero windows. Check the upstream "
            "Part B preprocessing and the EEG/CBF simulation pipeline."
        )

    models: dict[str, FullModalTransformer] = {}
    for model_key, _, _, display_name in _model_specs(config):
        model = FullModalTransformer(config, eeg_encoder_kind=model_key).to(config["device"])
        models[model_key] = model
        print(f"{display_name} parameters: {count_parameters(model):,}")

    if not args.eval_only:
        for model_key, checkpoint_path, history_path, display_name in _model_specs(config):
            train_full_modal_model(
                models[model_key],
                dataloaders["train"],
                dataloaders["val"],
                norm_stats,
                config,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                model_name=display_name,
            )

    for model_key, checkpoint_path, _, display_name in _model_specs(config):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Missing checkpoint for {display_name}: {checkpoint_path}")
        load_checkpoint(models[model_key], checkpoint_path, device=config["device"])

    benchmark_df, recommendation = run_efficiency_benchmark(
        models,
        dataloaders["test"],
        norm_stats,
        device=config["device"],
        csv_path=config["benchmark_csv_path"],
        config=config,
    )

    recommended_model = models[recommendation["recommended_model"]]
    sleep_analysis = analyse_sleep_stage_attention(
        recommended_model,
        dataloaders["test"],
        norm_stats,
        device=config["device"],
        save_path=config["sleep_stage_attention_path"],
    )

    summary = {
        "failure_demo": failure_report,
        "manifest": manifest,
        "benchmark": benchmark_df.to_dict(orient="records"),
        "recommendation": recommendation,
        "sleep_stage_analysis": sleep_analysis,
        "artifacts": {
            "benchmark_csv": config["benchmark_csv_path"],
            "sleep_stage_attention": config["sleep_stage_attention_path"],
        },
    }
    Path(config["part_c_summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Final summary")
    print("-" * 84)
    print(
        f"Recommended Part C backbone: {recommendation['recommended_model']} "
        f"({recommendation['reason']})"
    )
    print(f"Benchmark CSV: {config['benchmark_csv_path']}")
    print(f"Sleep-stage attention figure: {config['sleep_stage_attention_path']}")


if __name__ == "__main__":
    main()
