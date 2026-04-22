"""Synthetic 1,000-user cohort generation for Part D."""

from __future__ import annotations

import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from part_d.archetype_classifier import archetype_to_index
from preprocessing.cbf_simulation import generate_synthetic_cbf
from preprocessing.synthetic_ecg_emg import (
    generate_synthetic_ecg_features,
    generate_synthetic_emg_features,
)


ARCHETYPES = {
    "athlete": {
        "n_users": 250,
        "hr_resting_mean": 52,
        "hr_resting_std": 5,
        "hr_max": 185,
        "glucose_baseline": 82,
        "glucose_std": 12,
        "insulin_sensitivity": "high",
        "exercise_frequency": "daily",
        "sleep_quality": "good",
        "cbf_baseline": 58,
    },
    "sedentary": {
        "n_users": 350,
        "hr_resting_mean": 78,
        "hr_resting_std": 8,
        "hr_max": 150,
        "glucose_baseline": 98,
        "glucose_std": 20,
        "insulin_sensitivity": "moderate",
        "exercise_frequency": "rare",
        "sleep_quality": "moderate",
        "cbf_baseline": 50,
    },
    "elderly": {
        "n_users": 200,
        "hr_resting_mean": 68,
        "hr_resting_std": 10,
        "hr_max": 140,
        "glucose_baseline": 105,
        "glucose_std": 25,
        "insulin_sensitivity": "low",
        "exercise_frequency": "light",
        "sleep_quality": "poor",
        "cbf_baseline": 44,
    },
    "diabetic": {
        "n_users": 200,
        "hr_resting_mean": 74,
        "hr_resting_std": 9,
        "hr_max": 155,
        "glucose_baseline": 145,
        "glucose_std": 45,
        "insulin_sensitivity": "very_low",
        "exercise_frequency": "light",
        "sleep_quality": "poor",
        "cbf_baseline": 47,
    },
}

ECG_COLUMNS = ["sdnn", "rmssd", "lf_power", "hf_power", "lf_hf_ratio"]
EMG_COLUMNS = ["rms_envelope", "zero_crossing_rate"]

_INSULIN_SENSITIVITY_FACTOR = {
    "high": 1.35,
    "moderate": 1.0,
    "low": 0.78,
    "very_low": 0.55,
}
_EXERCISE_EVENTS_PER_DAY = {
    "daily": (0.9, 1.6),
    "rare": (0.05, 0.25),
    "light": (0.2, 0.55),
}
_SLEEP_FRAGMENTATION_SCALE = {
    "good": 0.35,
    "moderate": 0.75,
    "poor": 1.25,
}


def _clipped_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    """Sample from a clipped Gaussian distribution."""

    return float(np.clip(rng.normal(mean, std), low, high))


def _sample_user_params(archetype: str, rng: np.random.Generator) -> dict:
    """Sample concrete physiology for one virtual user."""

    spec = ARCHETYPES[archetype]
    resting_hr = _clipped_normal(
        rng,
        mean=spec["hr_resting_mean"],
        std=spec["hr_resting_std"],
        low=42.0,
        high=95.0,
    )
    glucose_baseline = _clipped_normal(
        rng,
        mean=spec["glucose_baseline"],
        std=max(spec["glucose_std"] * 0.18, 4.0),
        low=65.0,
        high=220.0,
    )

    exercise_range = _EXERCISE_EVENTS_PER_DAY[spec["exercise_frequency"]]
    return {
        "hr_resting": resting_hr,
        "hr_max": float(spec["hr_max"] + rng.normal(0.0, 5.0)),
        "glucose_baseline": glucose_baseline,
        "glucose_std": float(max(spec["glucose_std"] + rng.normal(0.0, 2.0), 8.0)),
        "insulin_sensitivity": spec["insulin_sensitivity"],
        "insulin_sensitivity_factor": _INSULIN_SENSITIVITY_FACTOR[spec["insulin_sensitivity"]],
        "exercise_frequency": spec["exercise_frequency"],
        "exercise_events_per_day": float(rng.uniform(*exercise_range)),
        "sleep_quality": spec["sleep_quality"],
        "sleep_fragmentation_scale": _SLEEP_FRAGMENTATION_SCALE[spec["sleep_quality"]],
        "cbf_baseline": float(spec["cbf_baseline"] + rng.normal(0.0, 1.5)),
        "age": {
            "athlete": float(rng.integers(22, 38)),
            "sedentary": float(rng.integers(28, 56)),
            "elderly": float(rng.integers(66, 82)),
            "diabetic": float(rng.integers(35, 70)),
        }[archetype],
    }


def _event_kernel(
    length: int,
    onset: int,
    rise_steps: int,
    decay_steps: int,
    amplitude: float,
) -> np.ndarray:
    """Build a simple asymmetric rise-and-decay response kernel."""

    kernel = np.zeros(length, dtype=np.float32)
    peak_index = min(onset + rise_steps, length - 1)
    if peak_index > onset:
        kernel[onset:peak_index] = np.linspace(0.0, amplitude, peak_index - onset, endpoint=False, dtype=np.float32)
    decay_end = min(peak_index + decay_steps, length)
    kernel[peak_index:decay_end] = np.linspace(
        amplitude,
        0.0,
        max(decay_end - peak_index, 1),
        endpoint=False,
        dtype=np.float32,
    )
    return kernel


def _simulate_meals(
    timestamps: pd.DatetimeIndex,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate three meal responses per day with archetype-scaled impact."""

    n_steps = len(timestamps)
    response = np.zeros(n_steps, dtype=np.float32)
    insulin_factor = float(params["insulin_sensitivity_factor"])
    base_amplitude = {
        "high": (20.0, 38.0),
        "moderate": (28.0, 48.0),
        "low": (34.0, 54.0),
        "very_low": (40.0, 60.0),
    }[params["insulin_sensitivity"]]

    for day_start in pd.date_range(timestamps[0].normalize(), timestamps[-1].normalize(), freq="1D"):
        for hour in (8, 13, 19):
            meal_time = day_start + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(int(rng.integers(-30, 31)), unit="m")
            onset = int(np.argmin(np.abs((timestamps - meal_time).asi8)))
            amplitude = rng.uniform(*base_amplitude) * (1.0 / insulin_factor)
            rise_steps = int(rng.integers(6, 10))
            decay_steps = int(rng.integers(12, 25))
            response += _event_kernel(n_steps, onset, rise_steps, decay_steps, amplitude)

    return response


def _simulate_exercise(
    archetype: str,
    timestamps: pd.DatetimeIndex,
    params: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate archetype-specific exercise effects on HR and glucose."""

    n_steps = len(timestamps)
    hr_effect = np.zeros(n_steps, dtype=np.float32)
    glucose_effect = np.zeros(n_steps, dtype=np.float32)

    hr_peak_ranges = {
        "athlete": (86.0, 120.0),
        "sedentary": (32.0, 52.0),
        "elderly": (26.0, 42.0),
        "diabetic": (30.0, 58.0),
    }
    glucose_drop_ranges = {
        "athlete": (15.0, 30.0),
        "sedentary": (5.0, 15.0),
        "elderly": (5.0, 10.0),
        "diabetic": (5.0, 20.0),
    }

    days = pd.date_range(timestamps[0].normalize(), timestamps[-1].normalize(), freq="1D")
    for day_start in days:
        if rng.random() > min(float(params["exercise_events_per_day"]), 1.0):
            continue

        start_hour = {
            "athlete": rng.uniform(6.0, 19.5),
            "sedentary": rng.uniform(10.0, 18.5),
            "elderly": rng.uniform(8.0, 17.0),
            "diabetic": rng.uniform(7.0, 20.0),
        }[archetype]
        start_time = day_start + pd.to_timedelta(start_hour, unit="h")
        onset = int(np.argmin(np.abs((timestamps - start_time).asi8)))

        rise_steps = int(rng.integers(2, 4))
        decay_steps = int(rng.integers(4, 8))
        hr_effect += _event_kernel(
            n_steps,
            onset,
            rise_steps,
            decay_steps,
            rng.uniform(*hr_peak_ranges[archetype]),
        )

        glucose_delay = int(rng.integers(2, 5))
        glucose_effect -= _event_kernel(
            n_steps,
            onset + glucose_delay,
            rise_steps,
            int(rng.integers(8, 16)),
            rng.uniform(*glucose_drop_ranges[archetype]),
        )

    return hr_effect, glucose_effect


def _simulate_base_signals(
    archetype: str,
    params: dict,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """Simulate 5-minute HR and glucose for one user across seven days."""

    hours = timestamps.hour.to_numpy(dtype=np.float32) + (timestamps.minute.to_numpy(dtype=np.float32) / 60.0)
    sleeping = (hours >= 22.0) | (hours < 6.0)

    circadian = 4.5 * np.sin((2.0 * np.pi * (hours - 15.0)) / 24.0)
    wake_drive = np.where(sleeping, -6.0, 6.0).astype(np.float32)
    fragmentation = rng.normal(0.0, params["sleep_fragmentation_scale"], size=len(timestamps)).astype("float32")
    dawn_effect = np.where((hours >= 5.0) & (hours <= 8.0), np.sin((hours - 5.0) * math.pi / 3.0), 0.0).astype("float32")

    exercise_hr, exercise_glucose = _simulate_exercise(archetype, timestamps, params, rng)
    meal_response = _simulate_meals(timestamps, params, rng)

    hr = (
        params["hr_resting"]
        + circadian
        + wake_drive
        + (fragmentation * 2.0)
        + exercise_hr
        + rng.normal(0.0, 2.4, size=len(timestamps))
    ).astype("float32")
    hr = np.clip(hr, 38.0, params["hr_max"]).astype("float32")

    glucose_noise = rng.normal(0.0, params["glucose_std"] * 0.10, size=len(timestamps)).astype("float32")
    glucose = (
        params["glucose_baseline"]
        + meal_response
        + exercise_glucose
        + (dawn_effect * (0.15 * params["glucose_std"]))
        + glucose_noise
    ).astype("float32")
    glucose = np.clip(glucose, 55.0, 320.0).astype("float32")

    return (
        pd.Series(hr, index=timestamps, name="heart_rate_bpm"),
        pd.Series(glucose, index=timestamps, name="glucose_mg_dl"),
    )


def generate_user(user_id: int, archetype: str, seed: int, *, n_days: int = 7) -> dict:
    """Generate one virtual user with 5-minute multimodal biosignals.

    The cohort stores signals that remain tractable on disk: HR, glucose,
    synthetic ECG-HRV features, synthetic EMG features, and CBF at 5-minute
    resolution. EEG is intentionally regenerated on demand later from each
    support/query window because a literal 7-day raw 256 Hz trace per user would
    be unnecessarily large for the Part D storage budget.
    """

    rng = np.random.default_rng(seed)
    params = _sample_user_params(archetype, rng)

    timestamps = pd.date_range("2025-01-01", periods=n_days * 24 * 12, freq="5min")
    hr_series, glucose_series = _simulate_base_signals(archetype, params, timestamps, rng)

    ecg_features = generate_synthetic_ecg_features(hr_series, rng=rng)
    emg_features = generate_synthetic_emg_features(hr_series, glucose_series, rng=rng)
    cbf_series = generate_synthetic_cbf(hr_series, glucose_series)
    cbf_series = cbf_series + (float(params["cbf_baseline"]) - 50.0)

    return {
        "user_id": int(user_id),
        "archetype": archetype,
        "archetype_id": archetype_to_index(archetype),
        "params": params,
        "start_time": timestamps[0].isoformat(),
        "signals": {
            "hr": hr_series.to_numpy(dtype=np.float32),
            "glucose": glucose_series.to_numpy(dtype=np.float32),
            "ecg_features": ecg_features[ECG_COLUMNS].to_numpy(dtype=np.float32),
            "emg_features": emg_features[EMG_COLUMNS].to_numpy(dtype=np.float32),
            "cbf": cbf_series.to_numpy(dtype=np.float32),
            "eeg": None,
        },
        "metadata": {
            "n_days": n_days,
            "n_steps": len(timestamps),
            "mean_glucose": float(glucose_series.mean()),
            "hr_resting": float(np.percentile(hr_series.to_numpy(dtype=np.float32), 10)),
            "file_stub": f"{user_id}_{archetype}.pt",
        },
    }


def _save_user_record(user_record: dict, output_dir: str | Path) -> dict:
    """Save one user record and return manifest-ready metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    user_path = output_path / user_record["metadata"]["file_stub"]
    torch.save(user_record, user_path)
    return {
        "user_id": user_record["user_id"],
        "archetype": user_record["archetype"],
        "archetype_id": user_record["archetype_id"],
        "n_days": user_record["metadata"]["n_days"],
        "n_steps": user_record["metadata"]["n_steps"],
        "mean_glucose": user_record["metadata"]["mean_glucose"],
        "hr_resting": user_record["metadata"]["hr_resting"],
        "file_path": str(user_path),
    }


def _generate_and_save(job: tuple[int, str, int, str, int]) -> dict:
    """Pool worker entrypoint."""

    user_id, archetype, seed, output_dir, n_days = job
    user_record = generate_user(user_id, archetype, seed, n_days=n_days)
    return _save_user_record(user_record, output_dir)


def _build_jobs(config: dict, output_dir: str | Path) -> list[tuple[int, str, int, str, int]]:
    """Create deterministic cohort-generation jobs from the config."""

    jobs: list[tuple[int, str, int, str, int]] = []
    next_user_id = 0
    for archetype, spec in ARCHETYPES.items():
        n_users = int(config["synthetic_users_per_archetype"].get(archetype, spec["n_users"]))
        for local_index in range(n_users):
            seed = int(config["seed"] + (next_user_id * 97) + local_index)
            jobs.append((next_user_id, archetype, seed, str(output_dir), int(config["cohort_days"])))
            next_user_id += 1
    return jobs


def generate_full_cohort(output_dir: str = "data/synthetic_cohort/", config: dict | None = None) -> pd.DataFrame:
    """Generate the full Part D cohort and save a manifest CSV."""

    if config is None:
        from part_d.config import get_runtime_config

        config = get_runtime_config()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["synthetic_cohort_manifest_path"])
    jobs = _build_jobs(config, output_path)

    n_workers = int(max(1, config["cohort_workers"]))
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            records = pool.map(_generate_and_save, jobs)
    else:
        records = [_generate_and_save(job) for job in jobs]

    manifest = pd.DataFrame(records).sort_values(["archetype", "user_id"]).reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)

    print()
    print("Synthetic cohort summary")
    print("-" * 72)
    summary = manifest.groupby("archetype")[["user_id", "mean_glucose", "hr_resting"]].agg(
        n_users=("user_id", "count"),
        mean_glucose=("mean_glucose", "mean"),
        hr_resting=("hr_resting", "mean"),
    )
    print(summary.to_string(formatters={
        "mean_glucose": "{:.2f}".format,
        "hr_resting": "{:.2f}".format,
    }))
    print("-" * 72)
    print()

    return manifest


__all__ = [
    "ARCHETYPES",
    "ECG_COLUMNS",
    "EMG_COLUMNS",
    "generate_full_cohort",
    "generate_user",
]
