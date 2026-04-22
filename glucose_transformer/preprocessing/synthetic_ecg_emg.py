"""Synthetic ECG-HRV and EMG feature generation for Part B.

The OhioT1DM dataset used in Part A does not include ECG or EMG. Part B creates
deterministic, physiologically coupled synthetic features from the existing
heart-rate and glucose-context windows so the fusion architecture can be studied
without changing the forecasting task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr


def _timestamp_seed(timestamp: pd.Timestamp, patient_id: int, seed: int) -> int:
    """Create a deterministic RNG seed from patient ID, timestamp, and base seed."""

    timestamp_ns = int(timestamp.value % (2**32 - 1))
    return int((seed + patient_id * 1009 + timestamp_ns) % (2**32 - 1))


def generate_synthetic_ecg_features(
    hr_series: pd.Series,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic ECG-HRV features from a heart-rate sequence.

    The constructed features follow the Part B prompt:
    SDNN and RMSSD are tied to short-window HR variability, LF power follows a
    slow sinusoidal trend, HF power falls as HR rises, and LF/HF is the implied
    autonomic-balance ratio. The exact values are synthetic but deterministic
    for a given RNG seed.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    hr = hr_series.astype("float32")
    rolling_std = hr.rolling(window=5, min_periods=1).std().fillna(0.0)
    rolling_mean = hr.rolling(window=5, min_periods=1).mean().fillna(hr.mean())

    if isinstance(hr.index, pd.DatetimeIndex):
        minutes = (hr.index.hour * 60) + hr.index.minute
    else:
        minutes = np.arange(len(hr), dtype=np.float32) * 5.0

    inverse_hr_term = np.clip((95.0 - rolling_mean) / 35.0, -1.25, 1.25)
    sdnn = (0.025 * rolling_std) + (0.12 * inverse_hr_term)
    sdnn = sdnn + rng.normal(0.0, 0.005, size=len(hr))
    sdnn = np.clip(sdnn, 0.005, None)

    rmssd = (rolling_std * 0.08) + rng.normal(0.0, 0.004, size=len(hr))
    rmssd = np.clip(rmssd, 0.004, None)

    lf_power = 0.3 + (0.1 * np.sin((2.0 * np.pi * minutes) / 240.0)) + rng.normal(0.0, 0.015, size=len(hr))
    lf_power = np.clip(lf_power, 0.05, None)

    hf_power = 0.5 - (0.2 * (hr.to_numpy() - 60.0) / 40.0) + rng.normal(0.0, 0.015, size=len(hr))
    hf_power = np.clip(hf_power, 0.05, None)

    lf_hf_ratio = lf_power / np.clip(hf_power, 1e-3, None)

    return pd.DataFrame(
        {
            "sdnn": sdnn.astype("float32"),
            "rmssd": rmssd.astype("float32"),
            "lf_power": lf_power.astype("float32"),
            "hf_power": hf_power.astype("float32"),
            "lf_hf_ratio": lf_hf_ratio.astype("float32"),
        },
        index=hr.index,
    )


def generate_synthetic_emg_features(
    hr_series: pd.Series,
    glucose_series: pd.Series,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic EMG-envelope features from HR and glucose context.

    Exercise-like periods are inferred from elevated, rising HR. The RMS
    envelope spikes during those segments, persists for about 10 minutes, then
    decays over 5 minutes. Zero-crossing rate is constructed to track the RMS
    envelope while remaining noisy enough to avoid trivial redundancy.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    hr = hr_series.astype("float32").to_numpy()
    glucose = glucose_series.astype("float32").to_numpy()
    hr_diff = np.diff(hr, prepend=hr[0])
    glucose_diff = np.diff(glucose, prepend=glucose[0])

    rms_envelope = np.full_like(hr, fill_value=0.05, dtype=np.float32)
    exercise_steps_remaining = 0
    cooldown_steps_remaining = 0

    for index, (hr_value, hr_delta) in enumerate(zip(hr, hr_diff)):
        is_exercise_trigger = bool(hr_value > 90.0 and hr_delta > 0.0)
        if is_exercise_trigger:
            exercise_steps_remaining = 2
            cooldown_steps_remaining = 1

        if exercise_steps_remaining > 0:
            rms_envelope[index] = 0.5 + (0.3 * rng.random())
            exercise_steps_remaining -= 1
        elif cooldown_steps_remaining > 0:
            rms_envelope[index] = 0.15 + (0.05 * rng.random())
            cooldown_steps_remaining -= 1
        else:
            rms_envelope[index] = 0.05 + max(glucose_diff[index], 0.0) * 0.0002

    rms_envelope = rms_envelope + np.clip((hr - 75.0) / 400.0, 0.0, 0.12)
    rms_envelope = np.clip(rms_envelope + rng.normal(0.0, 0.02, size=len(hr)), 0.01, 0.95)

    zero_crossing_rate = 0.15 + (0.6 * rms_envelope) + rng.normal(0.0, 0.03, size=len(hr))
    zero_crossing_rate = np.clip(zero_crossing_rate, 0.01, 1.0)

    return pd.DataFrame(
        {
            "rms_envelope": rms_envelope.astype("float32"),
            "zero_crossing_rate": zero_crossing_rate.astype("float32"),
        },
        index=hr_series.index,
    )


def _denormalise(values: torch.Tensor, stats: dict[str, float]) -> np.ndarray:
    """Convert normalised tensors back to approximate raw values."""

    return (values.detach().cpu().numpy() * float(stats["std"])) + float(stats["mean"])


def _normalise_frame(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Apply feature-wise z-score normalisation using training-only statistics."""

    normalised = frame.copy()
    for column, column_stats in stats.items():
        std = float(column_stats["std"]) if float(column_stats["std"]) > 0 else 1.0
        normalised[column] = ((normalised[column] - float(column_stats["mean"])) / std).astype("float32")
    return normalised


def _compute_feature_stats(frames: list[pd.DataFrame]) -> dict[str, dict[str, float]]:
    """Compute training-only mean and std for synthetic features."""

    concatenated = pd.concat(frames, axis=0, ignore_index=True)
    stats: dict[str, dict[str, float]] = {}
    for column in concatenated.columns:
        series = concatenated[column].astype("float32")
        std = float(series.std(ddof=0))
        stats[column] = {
            "mean": float(series.mean()),
            "std": std if std > 0 else 1.0,
        }
    return stats


def validate_synthetic_feature_coupling(
    hr_values: np.ndarray,
    ecg_frame: pd.DataFrame,
    emg_frame: pd.DataFrame,
    *,
    min_abs_corr: float,
    alpha: float,
) -> dict:
    """Validate that the synthetic features preserve expected physiological trends."""

    validations = {}

    for feature_name, expected_sign in {
        "sdnn": -1,
        "hf_power": -1,
        "rms_envelope": 1,
        "zero_crossing_rate": 1,
    }.items():
        series = ecg_frame[feature_name] if feature_name in ecg_frame else emg_frame[feature_name]
        correlation, p_value = pearsonr(hr_values.astype("float64"), series.to_numpy(dtype="float64"))
        passed = (
            np.sign(correlation) == expected_sign
            and abs(correlation) >= min_abs_corr
            and p_value <= alpha
        )
        validations[feature_name] = {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "expected_sign": expected_sign,
            "passed": bool(passed),
        }

    if not all(item["passed"] for item in validations.values()):
        raise ValueError(f"Synthetic feature validation failed: {json.dumps(validations, indent=2)}")

    return validations


def build_multimodal_processed_windows(config: dict) -> dict:
    """Create and save Part B multimodal windows from the existing Part A outputs."""

    processed_dir = Path(config["part_b_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)

    part_a_norm_stats = json.loads(Path(config["norm_stats_path"]).read_text(encoding="utf-8"))
    hr_stats = part_a_norm_stats["heart_rate_bpm"]
    glucose_stats = part_a_norm_stats["glucose_mg_dl"]

    split_sources = {
        "train": Path(config["train_windows_path"]),
        "val": Path(config["val_windows_path"]),
        "test": Path(config["test_windows_path"]),
    }
    split_targets = {
        "train": Path(config["part_b_train_windows_path"]),
        "val": Path(config["part_b_val_windows_path"]),
        "test": Path(config["part_b_test_windows_path"]),
    }

    raw_windows_by_split: dict[str, list[dict]] = {}
    training_ecg_frames: list[pd.DataFrame] = []
    training_emg_frames: list[pd.DataFrame] = []
    training_hr_values: list[np.ndarray] = []

    for split_name, source_path in split_sources.items():
        part_a_windows = torch.load(source_path, map_location="cpu", weights_only=False)
        split_windows: list[dict] = []

        for window in part_a_windows:
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

            window_rng = np.random.default_rng(
                _timestamp_seed(end_timestamp, patient_id, config["seed"])
            )
            ecg_frame = generate_synthetic_ecg_features(hr_series, rng=window_rng)
            emg_frame = generate_synthetic_emg_features(hr_series, glucose_series, rng=window_rng)

            split_windows.append(
                {
                    "hr_input": window["hr_input"].clone().to(dtype=torch.float32),
                    "glucose_input": window["glucose_input"].clone().to(dtype=torch.float32),
                    "ecg_features_raw": ecg_frame.reset_index(drop=True),
                    "emg_features_raw": emg_frame.reset_index(drop=True),
                    "glucose_target": window["glucose_target"].clone().to(dtype=torch.float32),
                    "patient_id": patient_id,
                    "timestamp": window["timestamp"],
                }
            )

            if split_name == "train":
                training_ecg_frames.append(ecg_frame.reset_index(drop=True))
                training_emg_frames.append(emg_frame.reset_index(drop=True))
                training_hr_values.append(hr_raw.astype("float32"))

        raw_windows_by_split[split_name] = split_windows

    ecg_stats = _compute_feature_stats(training_ecg_frames)
    emg_stats = _compute_feature_stats(training_emg_frames)

    aggregated_train_ecg = pd.concat(training_ecg_frames, axis=0, ignore_index=True)
    aggregated_train_emg = pd.concat(training_emg_frames, axis=0, ignore_index=True)
    validation_report = validate_synthetic_feature_coupling(
        hr_values=np.concatenate(training_hr_values, axis=0),
        ecg_frame=aggregated_train_ecg,
        emg_frame=aggregated_train_emg,
        min_abs_corr=config["synthetic_validation_min_abs_corr"],
        alpha=config["synthetic_validation_alpha"],
    )

    normalisation_stats = {
        **part_a_norm_stats,
        "ecg_features": ecg_stats,
        "emg_features": emg_stats,
        "synthetic_validation": validation_report,
    }

    manifest = {"splits": {}, "part_b_norm_stats_path": config["part_b_norm_stats_path"]}

    for split_name, raw_windows in raw_windows_by_split.items():
        processed_windows: list[dict] = []
        for raw_window in raw_windows:
            ecg_normalised = _normalise_frame(raw_window["ecg_features_raw"], ecg_stats)
            emg_normalised = _normalise_frame(raw_window["emg_features_raw"], emg_stats)

            processed_windows.append(
                {
                    "hr_input": raw_window["hr_input"],
                    "glucose_input": raw_window["glucose_input"],
                    "ecg_features": torch.tensor(ecg_normalised.to_numpy(), dtype=torch.float32),
                    "emg_features": torch.tensor(emg_normalised.to_numpy(), dtype=torch.float32),
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

    Path(config["part_b_norm_stats_path"]).write_text(
        json.dumps(normalisation_stats, indent=2),
        encoding="utf-8",
    )
    Path(config["synthetic_validation_path"]).write_text(
        json.dumps(validation_report, indent=2),
        encoding="utf-8",
    )
    Path(config["part_b_manifest_path"]).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "build_multimodal_processed_windows",
    "generate_synthetic_ecg_features",
    "generate_synthetic_emg_features",
    "validate_synthetic_feature_coupling",
]
