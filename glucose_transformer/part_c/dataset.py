"""Dataset building and loading utilities for Part C."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.cbf_simulation import generate_synthetic_cbf
from preprocessing.eeg_simulation import generate_synthetic_eeg


def _denormalise(values: torch.Tensor, stats: dict[str, float]) -> np.ndarray:
    """Convert normalised tensors back to approximate raw values."""

    return (values.detach().cpu().numpy() * float(stats["std"])) + float(stats["mean"])


def _update_running_stats(accumulator: dict[str, float], values: np.ndarray) -> None:
    """Update running sum statistics for a large feature array."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    accumulator["count"] += float(array.size)
    accumulator["sum"] += float(array.sum())
    accumulator["sum_sq"] += float(np.square(array).sum())


def _finalise_running_stats(accumulator: dict[str, float]) -> dict[str, float]:
    """Convert running sums into mean and standard deviation."""

    count = max(accumulator["count"], 1.0)
    mean = accumulator["sum"] / count
    variance = max((accumulator["sum_sq"] / count) - (mean**2), 1e-8)
    return {"mean": float(mean), "std": float(np.sqrt(variance))}


def build_full_modal_processed_windows(config: dict) -> dict:
    """Create Part C windows by extending Part B data with EEG and CBF."""

    processed_dir = Path(config["part_c_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    part_b_norm_stats = json.loads(Path(config["part_b_norm_stats_path"]).read_text(encoding="utf-8"))
    hr_stats = part_b_norm_stats["heart_rate_bpm"]
    glucose_stats = part_b_norm_stats["glucose_mg_dl"]

    split_sources = {
        "train": Path(config["part_b_train_windows_path"]),
        "val": Path(config["part_b_val_windows_path"]),
        "test": Path(config["part_b_test_windows_path"]),
    }
    split_targets = {
        "train": Path(config["part_c_train_windows_path"]),
        "val": Path(config["part_c_val_windows_path"]),
        "test": Path(config["part_c_test_windows_path"]),
    }

    raw_windows_by_split: dict[str, list[dict]] = {}
    eeg_accumulator = {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
    cbf_accumulator = {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}

    for split_name, source_path in split_sources.items():
        part_b_windows = torch.load(source_path, map_location="cpu", weights_only=False)
        split_windows: list[dict] = []

        for window in part_b_windows:
            patient_id = int(window["patient_id"])
            end_timestamp = pd.Timestamp(window["timestamp"])
            timestamps = pd.date_range(
                end=end_timestamp,
                periods=config["input_len"],
                freq=config["resample_frequency"],
            )

            hr_raw = _denormalise(window["hr_input"], hr_stats)
            glucose_raw = _denormalise(window["glucose_input"], glucose_stats)
            hr_series = pd.Series(hr_raw.astype("float32"), index=timestamps)
            glucose_series = pd.Series(glucose_raw.astype("float32"), index=timestamps)

            eeg_raw = generate_synthetic_eeg(
                hr_series,
                glucose_series,
                sfreq=config["eeg_sfreq"],
            )
            cbf_raw = generate_synthetic_cbf(hr_series, glucose_series)

            if split_name == "train":
                _update_running_stats(eeg_accumulator, eeg_raw)
                _update_running_stats(cbf_accumulator, cbf_raw.to_numpy(dtype=np.float32))

            split_windows.append(
                {
                    "hr_input": window["hr_input"].clone().to(dtype=torch.float32),
                    "glucose_input": window["glucose_input"].clone().to(dtype=torch.float32),
                    "ecg_features": window["ecg_features"].clone().to(dtype=torch.float32),
                    "emg_features": window["emg_features"].clone().to(dtype=torch.float32),
                    "eeg_signal_raw": eeg_raw.astype("float32"),
                    "cbf_raw": cbf_raw.to_numpy(dtype=np.float32),
                    "glucose_target": window["glucose_target"].clone().to(dtype=torch.float32),
                    "patient_id": patient_id,
                    "timestamp": window["timestamp"],
                }
            )

        raw_windows_by_split[split_name] = split_windows

    eeg_stats = _finalise_running_stats(eeg_accumulator)
    cbf_stats = _finalise_running_stats(cbf_accumulator)

    norm_stats = {
        **part_b_norm_stats,
        "eeg_signal": eeg_stats,
        "cbf_signal": cbf_stats,
    }

    manifest = {"splits": {}, "part_c_norm_stats_path": config["part_c_norm_stats_path"]}

    for split_name, raw_windows in raw_windows_by_split.items():
        processed_windows: list[dict] = []
        for raw_window in raw_windows:
            eeg_normalised = (
                (raw_window["eeg_signal_raw"] - float(eeg_stats["mean"])) / float(eeg_stats["std"])
            ).astype("float32")
            cbf_normalised = (
                (raw_window["cbf_raw"] - float(cbf_stats["mean"])) / float(cbf_stats["std"])
            ).astype("float32")

            processed_windows.append(
                {
                    "hr_input": raw_window["hr_input"],
                    "glucose_input": raw_window["glucose_input"],
                    "ecg_features": raw_window["ecg_features"],
                    "emg_features": raw_window["emg_features"],
                    "eeg_signal": torch.tensor(eeg_normalised, dtype=torch.float32),
                    "cbf_signal": torch.tensor(cbf_normalised, dtype=torch.float32),
                    "glucose_target": raw_window["glucose_target"],
                    "patient_id": raw_window["patient_id"],
                    "timestamp": raw_window["timestamp"],
                }
            )

        target_path = split_targets[split_name]
        torch.save(processed_windows, target_path)
        manifest["splits"][split_name] = {
            "num_windows": len(processed_windows),
            "windows_path": str(target_path),
        }

    Path(config["part_c_norm_stats_path"]).write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")
    Path(config["part_c_manifest_path"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class FullModalDataset(Dataset):
    """Dataset wrapper exposing HR, ECG, EMG, EEG, CBF, and glucose targets."""

    def __init__(self, windows_path: str | Path):
        self.windows_path = Path(windows_path)
        self.windows: list[dict] = torch.load(self.windows_path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        """Return the number of full-modal forecasting windows."""

        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return one full-modal forecasting sample."""

        window = self.windows[index]
        hr_sequence = window["hr_input"].to(dtype=torch.float32).unsqueeze(-1)
        glucose_context = window["glucose_input"].to(dtype=torch.float32).unsqueeze(-1)
        ecg_features = window["ecg_features"].to(dtype=torch.float32)
        emg_features = window["emg_features"].to(dtype=torch.float32)
        eeg_signal = window["eeg_signal"].to(dtype=torch.float32)
        cbf_signal = window["cbf_signal"].to(dtype=torch.float32).unsqueeze(-1)
        target = window["glucose_target"].to(dtype=torch.float32)
        patient_id = int(window["patient_id"])
        return (
            hr_sequence,
            glucose_context,
            ecg_features,
            emg_features,
            eeg_signal,
            cbf_signal,
            target,
            patient_id,
        )

    def get_metadata(self, index: int) -> dict:
        """Expose window-level metadata for evaluation and analysis."""

        return self.windows[index]


__all__ = ["FullModalDataset", "build_full_modal_processed_windows"]
