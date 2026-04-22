"""Frozen-activation probing classifiers for Part E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from part_e.biological_validation import derive_window_labels
from part_e.common import make_window_batch, save_json, trace_model


PROBE_LAYERS = [
    "input_tokens",
    "hr_layer1_out",
    "hr_layer2_out",
    "ecg_layer2_out",
    "emg_layer2_out",
    "eeg_layer2_out",
    "cross_attn_out",
    "final_cls",
]


def _layer_vector(sequence_tensor, *, uses_cls: bool = True) -> np.ndarray:
    """Convert a sequence representation into one vector per sample."""

    tensor = sequence_tensor.detach().cpu().float()
    if tensor.dim() == 3:
        if uses_cls:
            return tensor[:, 0, :].numpy().astype("float32")
        return tensor.mean(dim=1).numpy().astype("float32")
    return tensor.numpy().astype("float32")


def _trace_to_probe_vectors(trace: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract one vector per requested probing layer."""

    hr_input = trace["hr"]["input_tokens"][:, 1:, :]
    eeg_trace = trace["eeg"]["trace"]
    eeg_uses_cls = bool(eeg_trace["uses_cls"])

    return {
        "input_tokens": hr_input.mean(dim=1).detach().cpu().numpy().astype("float32"),
        "hr_layer1_out": _layer_vector(trace["hr"]["layer_outputs"][0]),
        "hr_layer2_out": _layer_vector(trace["hr"]["output"]),
        "ecg_layer2_out": _layer_vector(trace["ecg"]["output"]),
        "emg_layer2_out": _layer_vector(trace["emg"]["output"]),
        "eeg_layer2_out": _layer_vector(trace["eeg"]["trace"]["output"], uses_cls=eeg_uses_cls),
        "cross_attn_out": trace["hr_fused"][:, 0, :].detach().cpu().numpy().astype("float32"),
        "final_cls": trace["final"]["output"][:, 0, :].detach().cpu().numpy().astype("float32"),
    }


def _collect_probe_dataset(
    model,
    dataset,
    *,
    split: str,
    max_windows: int,
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect frozen activations and labels for one probing split."""

    layer_rows = {layer_name: [] for layer_name in PROBE_LAYERS}
    labels = {
        "sleep_stage": [],
        "archetype": [],
        "post_meal": [],
        "post_exercise": [],
        "glucose_level": [],
    }

    window_entries = []
    for window_entry in dataset.iter_split_windows(
        split,
        max_windows_per_user=int(dataset.config["analysis_max_windows_per_user"]),
    ):
        window_entries.append(window_entry)
        if len(window_entries) >= max_windows:
            break

    for start in range(0, len(window_entries), batch_size):
        batch_entries = window_entries[start:start + batch_size]
        batch = make_window_batch(batch_entries, device=device)
        trace = trace_model(model, batch, capture_attention=False)
        batch_vectors = _trace_to_probe_vectors(trace)

        for layer_name, vectors in batch_vectors.items():
            layer_rows[layer_name].append(vectors)

        for entry in batch_entries:
            derived = derive_window_labels(entry)
            labels["sleep_stage"].append(derived["sleep_stage"])
            labels["archetype"].append(derived["archetype"])
            labels["post_meal"].append(int(derived["post_meal"]))
            labels["post_exercise"].append(int(derived["post_exercise"]))
            labels["glucose_level"].append(float(derived["target_glucose_mean"]))

    features = {
        layer_name: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)
        for layer_name, chunks in layer_rows.items()
    }
    label_arrays = {
        key: np.asarray(values)
        for key, values in labels.items()
    }
    return features, label_arrays


def _fit_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    """Fit a linear classifier and return test accuracy."""

    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return float("nan")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return float(accuracy_score(test_y, predictions))


def _fit_regressor(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    """Fit a linear regressor and return test R^2."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return float(r2_score(test_y, predictions))


def _plot_probe_curves(results: dict[str, dict[str, float]], *, save_path: str | Path) -> str:
    """Plot probing performance as a function of representation depth."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    properties = [
        ("sleep_stage", "Accuracy"),
        ("archetype", "Accuracy"),
        ("post_meal", "Accuracy"),
        ("post_exercise", "Accuracy"),
        ("glucose_level", "R²"),
    ]

    figure, axes = plt.subplots(len(properties), 1, figsize=(12, 14), sharex=True)
    x_positions = np.arange(len(PROBE_LAYERS))

    for axis, (property_name, y_label) in zip(axes, properties):
        scores = [results[property_name].get(layer_name, np.nan) for layer_name in PROBE_LAYERS]
        axis.plot(x_positions, scores, marker="o", linewidth=2.0)
        axis.set_ylabel(y_label)
        axis.set_title(property_name.replace("_", " ").title())
        axis.grid(alpha=0.22)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(PROBE_LAYERS, rotation=35, ha="right")
    axes[-1].set_xlabel("Representation")
    figure.suptitle("Frozen-Activation Probing Curves", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def train_probing_classifiers(model, train_data, test_data, *, config: dict) -> dict:
    """Train small linear probes on frozen Part D activations."""

    train_features, train_labels = _collect_probe_dataset(
        model,
        train_data,
        split="train",
        max_windows=int(config["probing_max_train_windows"]),
        batch_size=int(config["probing_batch_size"]),
        device="cpu",
    )
    test_features, test_labels = _collect_probe_dataset(
        model,
        test_data,
        split="test",
        max_windows=int(config["probing_max_test_windows"]),
        batch_size=int(config["probing_batch_size"]),
        device="cpu",
    )

    results = {
        "sleep_stage": {},
        "archetype": {},
        "post_meal": {},
        "post_exercise": {},
        "glucose_level": {},
    }

    for layer_name in PROBE_LAYERS:
        train_x = train_features[layer_name]
        test_x = test_features[layer_name]
        results["sleep_stage"][layer_name] = _fit_classifier(
            train_x,
            train_labels["sleep_stage"],
            test_x,
            test_labels["sleep_stage"],
        )
        results["archetype"][layer_name] = _fit_classifier(
            train_x,
            train_labels["archetype"],
            test_x,
            test_labels["archetype"],
        )
        results["post_meal"][layer_name] = _fit_classifier(
            train_x,
            train_labels["post_meal"],
            test_x,
            test_labels["post_meal"],
        )
        results["post_exercise"][layer_name] = _fit_classifier(
            train_x,
            train_labels["post_exercise"],
            test_x,
            test_labels["post_exercise"],
        )
        results["glucose_level"][layer_name] = _fit_regressor(
            train_x,
            train_labels["glucose_level"],
            test_x,
            test_labels["glucose_level"],
        )

    plot_path = _plot_probe_curves(results, save_path=config["probing_plot_path"])
    payload = {
        "results": results,
        "plot_path": plot_path,
        "n_train": len(train_labels["glucose_level"]),
        "n_test": len(test_labels["glucose_level"]),
    }
    save_json(payload, config["probing_results_path"])
    return payload


__all__ = ["train_probing_classifiers"]
