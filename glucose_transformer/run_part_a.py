"""Single entry point for the full Part A pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from part_a.config import get_runtime_config
from part_a.dataset import GlucoseWindowDataset, create_dataloader
from part_a.evaluate import evaluate_model
from part_a.lstm_baseline import LSTMBaseline
from part_a.model import TemporalTransformer
from part_a.train import count_parameters, load_checkpoint, set_seed, train_model
from part_a.visualise_attention import save_random_attention_visualisations
from preprocessing.ohio_preprocessor import preprocess_ohio_dataset


def _processed_data_ready(config: dict) -> bool:
    """Check whether the required processed files already exist."""

    required_files = [
        config["train_windows_path"],
        config["val_windows_path"],
        config["test_windows_path"],
        config["norm_stats_path"],
    ]
    return all(Path(file_path).exists() for file_path in required_files)


def _load_norm_stats(config: dict) -> dict:
    """Load the saved normalisation statistics JSON."""

    return json.loads(Path(config["norm_stats_path"]).read_text(encoding="utf-8"))


def _build_dataloaders(config: dict) -> tuple[GlucoseWindowDataset, GlucoseWindowDataset, GlucoseWindowDataset, dict]:
    """Load saved window datasets and construct their dataloaders."""

    train_dataset = GlucoseWindowDataset(config["train_windows_path"])
    val_dataset = GlucoseWindowDataset(config["val_windows_path"])
    test_dataset = GlucoseWindowDataset(config["test_windows_path"])

    dataloaders = {
        "train": create_dataloader(train_dataset, config, shuffle=True),
        "val": create_dataloader(val_dataset, config, shuffle=False),
        "test": create_dataloader(test_dataset, config, shuffle=False),
    }
    return train_dataset, val_dataset, test_dataset, dataloaders


def _ensure_runtime_dirs(config: dict) -> None:
    """Create the output directories used by the pipeline."""

    for directory_key in ["data_processed_dir", "checkpoint_dir", "figures_dir"]:
        Path(config[directory_key]).mkdir(parents=True, exist_ok=True)


def _report_parameter_count(model_name: str, model: torch.nn.Module) -> None:
    """Print a compact trainable parameter count summary."""

    print(f"{model_name} parameters: {count_parameters(model):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part A of the glucose Transformer project.")
    parser.add_argument("--preprocess_only", action="store_true", help="run only preprocessing")
    parser.add_argument("--eval_only", action="store_true", help="skip training, load checkpoints, and evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU execution for debugging")
    args = parser.parse_args()

    config = get_runtime_config(no_cuda=args.no_cuda)
    _ensure_runtime_dirs(config)
    set_seed(config["seed"])

    if not _processed_data_ready(config):
        print("Processed data not found. Running preprocessing.")
        manifest = preprocess_ohio_dataset(config)
        print(json.dumps(manifest["splits"], indent=2))
    elif args.preprocess_only:
        print("Processed data already exists. Re-running preprocessing by request.")
        manifest = preprocess_ohio_dataset(config)
        print(json.dumps(manifest["splits"], indent=2))
    else:
        manifest = None

    if args.preprocess_only:
        print("Preprocessing complete.")
        return

    norm_stats = _load_norm_stats(config)
    train_dataset, val_dataset, test_dataset, dataloaders = _build_dataloaders(config)

    device = config["device"]
    print(f"Using device: {device}")
    print(
        f"Dataset sizes -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise RuntimeError(
            "At least one split has zero windows. Check the raw XML placement, "
            "signal availability, and preprocessing gap filtering."
        )

    transformer = TemporalTransformer(config).to(device)
    _report_parameter_count("TemporalTransformer", transformer)

    if not args.eval_only:
        train_model(
            transformer,
            dataloaders["train"],
            dataloaders["val"],
            norm_stats,
            config,
            checkpoint_path=config["transformer_checkpoint_path"],
            history_path=config["history_path"],
            model_name="TemporalTransformer",
        )

    if not Path(config["transformer_checkpoint_path"]).exists():
        raise FileNotFoundError(
            f"Missing transformer checkpoint at {config['transformer_checkpoint_path']}."
        )

    load_checkpoint(transformer, config["transformer_checkpoint_path"], device=device)
    transformer_metrics = evaluate_model(
        transformer,
        dataloaders["test"],
        norm_stats,
        device=device,
        model_name="TemporalTransformer",
    )

    lstm_baseline = LSTMBaseline(config).to(device)
    _report_parameter_count("LSTMBaseline", lstm_baseline)

    if not args.eval_only:
        train_model(
            lstm_baseline,
            dataloaders["train"],
            dataloaders["val"],
            norm_stats,
            config,
            checkpoint_path=config["lstm_checkpoint_path"],
            history_path=config["lstm_history_path"],
            model_name="LSTMBaseline",
        )

    lstm_metrics = None
    if Path(config["lstm_checkpoint_path"]).exists():
        load_checkpoint(lstm_baseline, config["lstm_checkpoint_path"], device=device)
        lstm_metrics = evaluate_model(
            lstm_baseline,
            dataloaders["test"],
            norm_stats,
            device=device,
            model_name="LSTMBaseline",
        )
    else:
        print("LSTM baseline checkpoint not found. Skipping baseline evaluation.")

    attention_artifacts = save_random_attention_visualisations(
        transformer,
        test_dataset,
        device=device,
        output_dir=config["figures_dir"],
        sample_count=config["attention_samples"],
    )

    comparison = {
        "manifest": manifest,
        "transformer": {
            "rmse_30min": transformer_metrics["rmse_30min"],
            "rmse_60min": transformer_metrics["rmse_60min"],
            "mae_30min": transformer_metrics["mae_30min"],
            "mae_60min": transformer_metrics["mae_60min"],
        },
        "lstm": None
        if lstm_metrics is None
        else {
            "rmse_30min": lstm_metrics["rmse_30min"],
            "rmse_60min": lstm_metrics["rmse_60min"],
            "mae_30min": lstm_metrics["mae_30min"],
            "mae_60min": lstm_metrics["mae_60min"],
        },
        "attention_artifacts": attention_artifacts,
    }
    Path(config["comparison_path"]).write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("Final summary")
    print("-" * 56)
    print(
        f"Transformer RMSE -> 30 min: {transformer_metrics['rmse_30min']:.3f} mg/dL, "
        f"60 min: {transformer_metrics['rmse_60min']:.3f} mg/dL"
    )
    if lstm_metrics is not None:
        print(
            f"LSTM RMSE        -> 30 min: {lstm_metrics['rmse_30min']:.3f} mg/dL, "
            f"60 min: {lstm_metrics['rmse_60min']:.3f} mg/dL"
        )
        delta = lstm_metrics["rmse_60min"] - transformer_metrics["rmse_60min"]
        print(f"60-minute RMSE improvement (positive means transformer better): {delta:.3f} mg/dL")
    print(f"Attention figures saved: {len(attention_artifacts)} samples")
    print(f"Comparison JSON: {config['comparison_path']}")


if __name__ == "__main__":
    main()
