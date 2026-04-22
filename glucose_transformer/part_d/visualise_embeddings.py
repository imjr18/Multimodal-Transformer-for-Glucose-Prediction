"""Embedding-space visualisation for Part D user personalisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from part_d.maml import compute_task_loss


def _adapt_embedding_only(
    model,
    task: dict,
    config: dict,
    *,
    device: str,
    support_size: int,
) -> np.ndarray:
    """Adapt only a per-user override vector while freezing the backbone."""

    criterion = torch.nn.MSELoss()
    model.eval()
    initial_embedding = model.get_initial_user_embedding(
        int(task["user_id"]),
        int(task["archetype_id"]),
        device=device,
    )
    user_embedding_override = initial_embedding.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([user_embedding_override], lr=float(config["maml_inner_lr"]))
    support_split = {key: value[:support_size] for key, value in task["support"].items()}

    if support_size > 0:
        for _ in range(int(config["maml_inner_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = compute_task_loss(
                model,
                support_split,
                device=device,
                user_id=int(task["user_id"]),
                archetype_id=int(task["archetype_id"]),
                criterion=criterion,
                batch_size=int(config["batch_size"]),
                user_embedding_override=user_embedding_override,
            )
            loss.backward()
            optimizer.step()

    return user_embedding_override.detach().cpu().numpy()


def visualise_user_embedding_space(
    model,
    dataset,
    *,
    config: dict,
    device: str,
    save_path: str | Path,
) -> tuple[plt.Figure, dict]:
    """Project personalised user embeddings into 2D and relate them to physiology."""

    metadata = dataset.get_user_metadata_frame().sort_values("user_id").reset_index(drop=True)
    tasks = [
        dataset.build_task(int(user_id))
        for user_id in metadata["user_id"].tolist()
    ]

    embeddings = np.stack(
        [
            _adapt_embedding_only(
                model,
                task,
                config,
                device=device,
                support_size=int(config["support_set_size"]),
            )
            for task in tasks
        ],
        axis=0,
    ).astype("float32")

    perplexity = max(5, min(30, embeddings.shape[0] - 1))
    tsne_projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=int(config["seed"]),
    ).fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=int(config["seed"]))
    pca_projection = pca.fit_transform(embeddings)

    pc1_corr_glucose = float(np.corrcoef(pca_projection[:, 0], metadata["mean_glucose"].to_numpy(dtype=np.float32))[0, 1])
    pc2_corr_hr = float(np.corrcoef(pca_projection[:, 1], metadata["hr_resting"].to_numpy(dtype=np.float32))[0, 1])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    palette = {
        "athlete": "#1b9e77",
        "sedentary": "#7570b3",
        "elderly": "#d95f02",
        "diabetic": "#e7298a",
    }

    for archetype, group in metadata.groupby("archetype", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        axes[0].scatter(
            tsne_projection[indices, 0],
            tsne_projection[indices, 1],
            s=16,
            alpha=0.75,
            color=palette.get(archetype, "#4c4c4c"),
            label=archetype,
        )
        scatter = axes[1].scatter(
            pca_projection[indices, 0],
            pca_projection[indices, 1],
            s=16,
            alpha=0.75,
            color=palette.get(archetype, "#4c4c4c"),
            label=archetype,
        )
        scatter.set_edgecolor("none")

    axes[0].set_title("t-SNE of Personalised User Embeddings")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(alpha=0.18)
    axes[0].legend()

    axes[1].set_title("PCA of Personalised User Embeddings")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(alpha=0.18)
    axes[1].text(
        0.02,
        0.98,
        f"corr(PC1, mean glucose) = {pc1_corr_glucose:.2f}\n"
        f"corr(PC2, resting HR) = {pc2_corr_hr:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    figure.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")

    stats = {
        "pc1_corr_mean_glucose": pc1_corr_glucose,
        "pc2_corr_resting_hr": pc2_corr_hr,
        "figure_path": str(output_path),
    }
    return figure, stats


__all__ = ["visualise_user_embedding_space"]
