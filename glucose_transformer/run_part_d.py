"""Single entry point for the full Part D population-generalisation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from part_a.train import count_parameters, load_checkpoint, set_seed
from part_d.cohort_simulator import generate_full_cohort
from part_d.config import get_runtime_config
from part_d.dataset import MetaLearningDataset
from part_d.evaluate import cross_archetype_evaluation, plot_adaptation_curve
from part_d.maml import FOMAML, load_meta_checkpoint
from part_d.user_embedding import UserConditionedFullModalTransformer
from part_d.visualise_embeddings import visualise_user_embedding_space


def _ensure_runtime_dirs(config: dict) -> None:
    """Create the Part D output directories."""

    for key in ["synthetic_cohort_dir", "results_dir_part_d", "figures_dir_part_d", "checkpoint_dir_part_d"]:
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def _part_c_recommendation(config: dict) -> str:
    """Return the preferred Part C EEG backbone for Part D initialisation."""

    summary_path = Path(config["part_c_summary_path"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recommendation = summary.get("recommendation", {})
        if recommendation.get("recommended_model") in {"frequency_eeg", "patch_eeg", "hierarchical_eeg"}:
            return recommendation["recommended_model"]
    return str(config["part_d_eeg_encoder_kind"])


def _part_c_checkpoint_for_backbone(config: dict, backbone_kind: str) -> str:
    """Map a Part C backbone kind to its checkpoint path."""

    checkpoint_lookup = {
        "frequency_eeg": config["frequency_checkpoint_path"],
        "patch_eeg": config["patch_checkpoint_path"],
        "hierarchical_eeg": config["hierarchical_checkpoint_path"],
    }
    return checkpoint_lookup[backbone_kind]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part D of the glucose Transformer project.")
    parser.add_argument("--cohort_only", action="store_true", help="generate only the synthetic cohort")
    parser.add_argument("--eval_only", action="store_true", help="skip FOMAML training and evaluate an existing checkpoint")
    parser.add_argument("--no_cuda", action="store_true", help="force CPU execution for debugging")
    parser.add_argument("--meta_epochs", type=int, default=None, help="override the number of meta-training epochs")
    args = parser.parse_args()

    config = get_runtime_config(no_cuda=args.no_cuda)
    if args.meta_epochs is not None:
        config["maml_meta_epochs"] = int(args.meta_epochs)

    _ensure_runtime_dirs(config)
    set_seed(int(config["seed"]))

    manifest_path = Path(config["synthetic_cohort_manifest_path"])
    if not manifest_path.exists():
        generate_full_cohort(config=config, output_dir=config["synthetic_cohort_dir"])

    if args.cohort_only:
        print(f"Synthetic cohort ready: {config['synthetic_cohort_manifest_path']}")
        return

    meta_dataset = MetaLearningDataset(config)
    backbone_kind = _part_c_recommendation(config)
    model = UserConditionedFullModalTransformer(
        config,
        eeg_encoder_kind=backbone_kind,
        n_users=len(meta_dataset.manifest),
        inject_conditioning=False,
    ).to(config["device"])

    print(f"Using device: {config['device']}")
    print(f"Backbone for Part D: {backbone_kind}")
    print(f"User-conditioned model parameters: {count_parameters(model):,}")
    print(
        f"Task counts -> train: {len(meta_dataset.splits['train'])}, "
        f"val: {len(meta_dataset.splits['val'])}, test: {len(meta_dataset.splits['test'])}"
    )

    part_c_checkpoint = Path(_part_c_checkpoint_for_backbone(config, backbone_kind))
    if part_c_checkpoint.exists():
        load_checkpoint(model.backbone, part_c_checkpoint, device=config["device"])
        print(f"Loaded Part C backbone weights from {part_c_checkpoint}")
    else:
        print(f"Part C checkpoint not found for {backbone_kind}; starting Part D from scratch.")

    model.inject_user_conditioning()
    model.set_known_user_ids(meta_dataset.get_known_user_ids())

    if not args.eval_only:
        trainer = FOMAML(model, meta_dataset, meta_dataset.norm_stats, config)
        trainer.train(
            checkpoint_path=config["best_meta_checkpoint_path"],
            history_path=config["meta_history_path"],
        )

    checkpoint_path = Path(config["best_meta_checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Part D checkpoint: {checkpoint_path}. "
            "Run without --eval_only or provide a trained checkpoint."
        )
    load_meta_checkpoint(model, checkpoint_path, device=config["device"])

    test_tasks = meta_dataset.get_split_tasks("test")
    figure = plot_adaptation_curve(
        model,
        test_tasks,
        meta_dataset.norm_stats,
        config=config,
        device=config["device"],
        save_path=config["adaptation_curve_path"],
    )
    cross_archetype_df = cross_archetype_evaluation(
        model,
        test_tasks,
        meta_dataset.norm_stats,
        config=config,
        device=config["device"],
        csv_path=config["cross_archetype_csv_path"],
    )
    if figure is not None:
        figure.clf()

    _, embedding_stats = visualise_user_embedding_space(
        model,
        meta_dataset,
        config=config,
        device=config["device"],
        save_path=config["embedding_space_path"],
    )

    mean_zero_shot = float(cross_archetype_df["rmse_0shot"].mean())
    mean_sixty_min = float(cross_archetype_df["rmse_60min"].mean())
    summary = {
        "backbone_kind": backbone_kind,
        "cohort_manifest_path": config["synthetic_cohort_manifest_path"],
        "norm_stats_path": config["synthetic_cohort_norm_stats_path"],
        "splits": meta_dataset.splits,
        "mean_rmse_0shot": mean_zero_shot,
        "mean_rmse_60min": mean_sixty_min,
        "personalisation_improvement_pct": ((mean_zero_shot - mean_sixty_min) / max(mean_zero_shot, 1e-6)) * 100.0,
        "embedding_stats": embedding_stats,
        "artifacts": {
            "adaptation_curve": config["adaptation_curve_path"],
            "cross_archetype_csv": config["cross_archetype_csv_path"],
            "embedding_space": config["embedding_space_path"],
            "checkpoint": config["best_meta_checkpoint_path"],
        },
    }
    Path(config["part_d_summary_path"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Final summary")
    print("-" * 84)
    print(f"Mean 0-shot RMSE: {mean_zero_shot:.3f} mg/dL")
    print(f"Mean 60-minute adapted RMSE: {mean_sixty_min:.3f} mg/dL")
    print(f"Adaptation curve: {config['adaptation_curve_path']}")
    print(f"Cross-archetype results: {config['cross_archetype_csv_path']}")
    print(f"Embedding-space figure: {config['embedding_space_path']}")


if __name__ == "__main__":
    main()
