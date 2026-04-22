"""Single entry point for the full Part B multimodal pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from part_a.config import get_runtime_config as get_part_a_runtime_config
from part_a.dataset import create_dataloader
from part_a.train import load_checkpoint, set_seed
from part_b.ablation import run_modality_ablation
from part_b.config import get_runtime_config
from part_b.dataset import MultiModalWindowDataset
from part_b.evaluate import compare_fusion_strategies, save_cross_attention_heatmap
from part_b.models.cross_attention import CrossModalTransformer
from part_b.models.early_fusion import EarlyFusionTransformer
from part_b.models.late_fusion import LateFusionTransformer
from part_b.train import train_multimodal_model
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
    """Check that the Part B multimodal processed windows exist."""

    required_files = [
        config["part_b_train_windows_path"],
        config["part_b_val_windows_path"],
        config["part_b_test_windows_path"],
        config["part_b_norm_stats_path"],
    ]
    return all(Path(file_path).exists() for file_path in required_files)


def _ensure_runtime_dirs(config: dict) -> None:
    """Create the output directories needed for Part B execution."""

    for directory_key in ["results_dir", "checkpoint_dir_part_b", "part_b_processed_dir"]:
        Path(config[directory_key]).mkdir(parents=True, exist_ok=True)


def _load_norm_stats(config: dict) -> dict:
    """Load the saved Part B normalisation statistics."""

    return json.loads(Path(config["part_b_norm_stats_path"]).read_text(encoding="utf-8"))


def _build_dataloaders(config: dict):
    """Load Part B datasets and wrap them in dataloaders."""

    train_dataset = MultiModalWindowDataset(config["part_b_train_windows_path"])
    val_dataset = MultiModalWindowDataset(config["part_b_val_windows_path"])
    test_dataset = MultiModalWindowDataset(config["part_b_test_windows_path"])

    dataloaders = {
        "train": create_dataloader(train_dataset, config, shuffle=True),
        "val": create_dataloader(val_dataset, config, shuffle=False),
        "test": create_dataloader(test_dataset, config, shuffle=False),
    }
    return train_dataset, val_dataset, test_dataset, dataloaders


def _print_parameter_count(model_name: str, model) -> None:
    """Print the model parameter count using the loaded checkpoint-safe utility."""

    from part_a.train import count_parameters

    print(f"{model_name} parameters: {count_parameters(model):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part B of the glucose Transformer project.")
    parser.add_argument("--preprocess_only", action="store_true", help="run only preprocessing")
    parser.add_argument("--eval_only", action="store_true", help="skip training and evaluate checkpoints only")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU execution for debugging")
    args = parser.parse_args()

    config = get_runtime_config(no_cuda=args.no_cuda)
    _ensure_runtime_dirs(config)
    set_seed(config["seed"])

    if not _part_a_processed_ready(config):
        print("Part A processed data not found. Running Part A preprocessing first.")
        preprocess_ohio_dataset(get_part_a_runtime_config(no_cuda=args.no_cuda))

    if not _part_b_processed_ready(config) or args.preprocess_only:
        print("Building Part B multimodal processed windows.")
        manifest = build_multimodal_processed_windows(config)
        print(json.dumps(manifest["splits"], indent=2))
    else:
        manifest_path = Path(config["part_b_manifest_path"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "splits": {
                    "train": {"windows_path": config["part_b_train_windows_path"]},
                    "val": {"windows_path": config["part_b_val_windows_path"]},
                    "test": {"windows_path": config["part_b_test_windows_path"]},
                }
            }

    if args.preprocess_only:
        print("Part B preprocessing complete.")
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
            "At least one Part B split has zero windows. Check Part A preprocessing "
            "and the multimodal synthetic feature generation."
        )

    early_model = EarlyFusionTransformer(config).to(config["device"])
    late_model = LateFusionTransformer(config).to(config["device"])
    cross_model = CrossModalTransformer(config).to(config["device"])

    _print_parameter_count("EarlyFusionTransformer", early_model)
    _print_parameter_count("LateFusionTransformer", late_model)
    _print_parameter_count("CrossModalTransformer", cross_model)

    if not args.eval_only:
        train_multimodal_model(
            early_model,
            dataloaders["train"],
            dataloaders["val"],
            norm_stats,
            config,
            checkpoint_path=config["early_fusion_checkpoint_path"],
            history_path=config["early_fusion_history_path"],
            model_name="EarlyFusionTransformer",
        )
        train_multimodal_model(
            late_model,
            dataloaders["train"],
            dataloaders["val"],
            norm_stats,
            config,
            checkpoint_path=config["late_fusion_checkpoint_path"],
            history_path=config["late_fusion_history_path"],
            model_name="LateFusionTransformer",
        )
        train_multimodal_model(
            cross_model,
            dataloaders["train"],
            dataloaders["val"],
            norm_stats,
            config,
            checkpoint_path=config["cross_attention_checkpoint_path"],
            history_path=config["cross_attention_history_path"],
            model_name="CrossModalTransformer",
        )

    for model_name, model, checkpoint_path in [
        ("EarlyFusionTransformer", early_model, config["early_fusion_checkpoint_path"]),
        ("LateFusionTransformer", late_model, config["late_fusion_checkpoint_path"]),
        ("CrossModalTransformer", cross_model, config["cross_attention_checkpoint_path"]),
    ]:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, device=config["device"])

    comparison_df = compare_fusion_strategies(
        early_model,
        late_model,
        cross_model,
        dataloaders["test"],
        norm_stats,
        device=config["device"],
        csv_path=config["fusion_comparison_csv_path"],
    )

    ablation_df = run_modality_ablation(
        cross_model,
        dataloaders["test"],
        norm_stats,
        device=config["device"],
        csv_path=config["ablation_results_csv_path"],
        figure_path=config["modality_contribution_path"],
    )

    cross_attention_heatmap_path = save_cross_attention_heatmap(
        cross_model,
        test_dataset,
        device=config["device"],
        save_path=config["cross_attention_heatmap_path"],
        sample_index=0,
    )

    summary = {
        "manifest": manifest,
        "fusion_comparison": comparison_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
        "artifacts": {
            "fusion_comparison_csv": config["fusion_comparison_csv_path"],
            "ablation_results_csv": config["ablation_results_csv_path"],
            "cross_attention_heatmap": cross_attention_heatmap_path,
            "modality_contribution_plot": config["modality_contribution_path"],
            "synthetic_validation_report": config["synthetic_validation_path"],
        },
    }
    Path(config["part_b_summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_model_row = comparison_df.iloc[0]
    print("Final summary")
    print("-" * 72)
    print(
        f"Best model by 60-minute RMSE: {best_model_row['model']} "
        f"({best_model_row['rmse_60min']:.3f} mg/dL)"
    )
    print(f"Fusion comparison CSV: {config['fusion_comparison_csv_path']}")
    print(f"Ablation CSV: {config['ablation_results_csv_path']}")
    print(f"Cross-attention heatmap: {cross_attention_heatmap_path}")
    print(f"Modality contribution plot: {config['modality_contribution_path']}")


if __name__ == "__main__":
    main()
