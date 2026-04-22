"""Synthetic multimodal cohort generation for non-invasive glucose estimation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ARCHETYPES = {
    "athlete": {
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
EEG_COLUMNS = ["delta", "theta", "alpha", "beta", "gamma"]
ARCHETYPE_TO_ID = {name: index for index, name in enumerate(ARCHETYPES)}

_INSULIN_FACTOR = {"high": 1.35, "moderate": 1.0, "low": 0.78, "very_low": 0.55}
_EXERCISE_EVENTS_PER_DAY = {"daily": (0.9, 1.6), "rare": (0.05, 0.25), "light": (0.2, 0.55)}
_SLEEP_FRAGMENTATION = {"good": 0.35, "moderate": 0.75, "poor": 1.25}


def _json_default(value: Any):
    """Serialise NumPy objects for JSON output."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _clipped_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    """Sample a clipped Gaussian scalar."""

    return float(np.clip(rng.normal(mean, std), low, high))


def _sample_user_params(archetype: str, rng: np.random.Generator) -> dict[str, float | str]:
    """Sample concrete physiology for one synthetic user."""

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
    return {
        "hr_resting": resting_hr,
        "hr_max": float(spec["hr_max"] + rng.normal(0.0, 5.0)),
        "glucose_baseline": glucose_baseline,
        "glucose_std": float(max(spec["glucose_std"] + rng.normal(0.0, 2.0), 8.0)),
        "insulin_sensitivity": spec["insulin_sensitivity"],
        "insulin_sensitivity_factor": _INSULIN_FACTOR[spec["insulin_sensitivity"]],
        "exercise_frequency": spec["exercise_frequency"],
        "exercise_events_per_day": float(rng.uniform(*_EXERCISE_EVENTS_PER_DAY[spec["exercise_frequency"]])),
        "sleep_quality": spec["sleep_quality"],
        "sleep_fragmentation_scale": _SLEEP_FRAGMENTATION[spec["sleep_quality"]],
        "cbf_baseline": float(spec["cbf_baseline"] + rng.normal(0.0, 1.5)),
    }


def _event_kernel(length: int, onset: int, rise_steps: int, decay_steps: int, amplitude: float) -> np.ndarray:
    """Create an asymmetric rise-and-decay response curve."""

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
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate daily meal events and the resulting glucose load."""

    n_steps = len(timestamps)
    response = np.zeros(n_steps, dtype=np.float32)
    meal_indicator = np.zeros(n_steps, dtype=np.float32)
    insulin_factor = float(params["insulin_sensitivity_factor"])
    amplitude_range = {
        "high": (20.0, 38.0),
        "moderate": (28.0, 48.0),
        "low": (34.0, 54.0),
        "very_low": (40.0, 60.0),
    }[params["insulin_sensitivity"]]

    for day_start in pd.date_range(timestamps[0].normalize(), timestamps[-1].normalize(), freq="1D"):
        for hour in (8, 13, 19):
            meal_time = day_start + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(int(rng.integers(-30, 31)), unit="m")
            onset = int(np.argmin(np.abs((timestamps - meal_time).asi8)))
            meal_indicator[onset] = 1.0
            amplitude = rng.uniform(*amplitude_range) * (1.0 / insulin_factor)
            rise_steps = int(rng.integers(3, 5))
            decay_steps = int(rng.integers(8, 16))
            response += _event_kernel(n_steps, onset, rise_steps, decay_steps, amplitude)

    return response, meal_indicator


def _simulate_exercise(
    archetype: str,
    timestamps: pd.DatetimeIndex,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate archetype-specific exercise effects."""

    n_steps = len(timestamps)
    hr_effect = np.zeros(n_steps, dtype=np.float32)
    glucose_effect = np.zeros(n_steps, dtype=np.float32)
    exercise_indicator = np.zeros(n_steps, dtype=np.float32)

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

    for day_start in pd.date_range(timestamps[0].normalize(), timestamps[-1].normalize(), freq="1D"):
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
        exercise_indicator[onset : min(onset + 3, n_steps)] = 1.0

        rise_steps = int(rng.integers(2, 4))
        decay_steps = int(rng.integers(4, 8))
        hr_effect += _event_kernel(
            n_steps,
            onset,
            rise_steps,
            decay_steps,
            rng.uniform(*hr_peak_ranges[archetype]),
        )

        glucose_delay = int(rng.integers(1, 3))
        glucose_effect -= _event_kernel(
            n_steps,
            onset + glucose_delay,
            rise_steps,
            int(rng.integers(4, 10)),
            rng.uniform(*glucose_drop_ranges[archetype]),
        )

    return hr_effect, glucose_effect, exercise_indicator


def _simulate_base_signals(
    archetype: str,
    params: dict[str, Any],
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Simulate 5-minute HR and glucose traces plus event indicators."""

    hours = timestamps.hour.to_numpy(dtype=np.float32) + (timestamps.minute.to_numpy(dtype=np.float32) / 60.0)
    sleeping = (hours >= 22.0) | (hours < 6.0)
    circadian = 4.5 * np.sin((2.0 * np.pi * (hours - 15.0)) / 24.0)
    wake_drive = np.where(sleeping, -6.0, 6.0).astype(np.float32)
    fragmentation = rng.normal(0.0, params["sleep_fragmentation_scale"], size=len(timestamps)).astype("float32")
    dawn_effect = np.where((hours >= 5.0) & (hours <= 8.0), np.sin((hours - 5.0) * math.pi / 3.0), 0.0).astype("float32")

    meal_response, meal_indicator = _simulate_meals(timestamps, params, rng)
    exercise_hr, exercise_glucose, exercise_indicator = _simulate_exercise(archetype, timestamps, params, rng)

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
        pd.Series(hr, index=timestamps, name="hr"),
        pd.Series(glucose, index=timestamps, name="glucose"),
        meal_indicator.astype("float32"),
        exercise_indicator.astype("float32"),
    )


def generate_synthetic_ecg_features(
    hr_series: pd.Series,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate physiologically coupled HRV features from heart rate."""

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
    sdnn = (0.025 * rolling_std) + (0.12 * inverse_hr_term) + rng.normal(0.0, 0.005, size=len(hr))
    rmssd = (rolling_std * 0.08) + rng.normal(0.0, 0.004, size=len(hr))
    lf_power = 0.3 + (0.1 * np.sin((2.0 * np.pi * minutes) / 240.0)) + rng.normal(0.0, 0.015, size=len(hr))
    hf_power = 0.5 - (0.2 * (hr.to_numpy() - 60.0) / 40.0) + rng.normal(0.0, 0.015, size=len(hr))

    sdnn = np.clip(sdnn, 0.005, None)
    rmssd = np.clip(rmssd, 0.004, None)
    lf_power = np.clip(lf_power, 0.05, None)
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
    """Generate EMG envelope features coupled to HR and glucose dynamics."""

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
    zero_crossing_rate = np.clip(0.15 + (0.6 * rms_envelope) + rng.normal(0.0, 0.03, size=len(hr)), 0.01, 1.0)

    return pd.DataFrame(
        {
            "rms_envelope": rms_envelope.astype("float32"),
            "zero_crossing_rate": zero_crossing_rate.astype("float32"),
        },
        index=hr_series.index,
    )


def generate_synthetic_cbf(hr_series: pd.Series, glucose_series: pd.Series, *, cbf_baseline: float = 50.0) -> pd.Series:
    """Simulate slow cerebral blood flow from posture, exercise, and glucose."""

    hr = hr_series.astype("float32")
    glucose = glucose_series.astype("float32")
    index = hr.index
    rng_seed = int(float(hr.mean()) * 10 + float(glucose.mean()) * 10 + len(hr))
    rng = np.random.default_rng(rng_seed % (2**32 - 1))

    baseline = np.full(len(hr), cbf_baseline, dtype=np.float32)
    if isinstance(index, pd.DatetimeIndex):
        hours = np.array([timestamp.hour + (timestamp.minute / 60.0) for timestamp in index], dtype=np.float32)
        waking_hours = (hours >= 7.0) & (hours < 22.0)
    else:
        waking_hours = np.ones(len(hr), dtype=bool)
    sleeping_hours = ~waking_hours

    postural_effect = np.where(waking_hours, -0.08 * baseline, 0.08 * baseline).astype("float32")
    exercise_indicator = pd.Series((hr > 85.0).astype("float32"), index=index).shift(1, fill_value=0.0)
    exercise_effect = (0.15 * baseline * exercise_indicator.to_numpy(dtype=np.float32)).astype("float32")
    glucose_effect = (-0.03 * (glucose.to_numpy(dtype=np.float32) - 110.0)).astype("float32")
    ageing_drift = np.linspace(0.0, -1.5, len(hr), dtype=np.float32)
    gaussian_noise = rng.normal(0.0, 2.0, size=len(hr)).astype("float32")

    cbf = baseline + postural_effect + exercise_effect + glucose_effect + ageing_drift + gaussian_noise
    return pd.Series(cbf.astype("float32"), index=index, name="cbf")


def generate_synthetic_eeg_bands(
    hr_series: pd.Series,
    glucose_series: pd.Series,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate 5-band EEG power features and coarse sleep-stage labels.

    The features are intentionally low-rate and state-centric. Deep sleep
    windows become delta-dominant, light sleep shifts toward theta/alpha, and
    awake periods become beta-heavy. Glucose modulates these templates slightly
    so the model can learn imperfect but meaningful physiological coupling.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    templates = {
        "deep_sleep": np.array([0.58, 0.16, 0.08, 0.08, 0.10], dtype=np.float32),
        "light_sleep": np.array([0.20, 0.34, 0.22, 0.14, 0.10], dtype=np.float32),
        "awake": np.array([0.08, 0.14, 0.18, 0.40, 0.20], dtype=np.float32),
    }
    rows: list[np.ndarray] = []
    stages: list[str] = []

    for timestamp, hr_value, glucose_value in zip(hr_series.index, hr_series.to_numpy(), glucose_series.to_numpy()):
        hour = timestamp.hour + (timestamp.minute / 60.0) if isinstance(timestamp, pd.Timestamp) else 12.0
        sleeping = (hour >= 22.0) or (hour < 6.0)
        if sleeping and hr_value < 55.0:
            stage = "deep_sleep"
        elif sleeping and hr_value < 65.0:
            stage = "light_sleep"
        else:
            stage = "awake"

        base = templates[stage].copy()
        glucose_delta = float((glucose_value - 110.0) / 120.0)
        if stage == "awake":
            base[3] += 0.04 * glucose_delta
            base[0] -= 0.02 * glucose_delta
        elif stage == "deep_sleep":
            base[0] += 0.03 * max(-glucose_delta, 0.0)
            base[3] += 0.02 * max(glucose_delta, 0.0)

        noisy = np.clip(base + rng.normal(0.0, 0.015, size=5).astype(np.float32), 1e-3, None)
        noisy = noisy / noisy.sum()
        rows.append(noisy.astype(np.float32))
        stages.append(stage)

    return pd.DataFrame(rows, columns=EEG_COLUMNS, index=hr_series.index), stages


def generate_user(user_id: int, archetype: str, seed: int, *, n_days: int = 4) -> dict[str, Any]:
    """Generate one synthetic user with multimodal 5-minute signals."""

    rng = np.random.default_rng(seed)
    params = _sample_user_params(archetype, rng)
    timestamps = pd.date_range("2025-01-01", periods=n_days * 24 * 12, freq="5min")

    hr_series, glucose_series, meal_indicator, exercise_indicator = _simulate_base_signals(archetype, params, timestamps, rng)
    ecg_features = generate_synthetic_ecg_features(hr_series, rng=rng)
    emg_features = generate_synthetic_emg_features(hr_series, glucose_series, rng=rng)
    eeg_bands, sleep_stages = generate_synthetic_eeg_bands(hr_series, glucose_series, rng=rng)
    cbf_series = generate_synthetic_cbf(hr_series, glucose_series, cbf_baseline=float(params["cbf_baseline"]))

    return {
        "user_id": int(user_id),
        "archetype": archetype,
        "archetype_id": ARCHETYPE_TO_ID[archetype],
        "params": params,
        "timestamps": [timestamp.isoformat() for timestamp in timestamps],
        "signals": {
            "hr": hr_series.to_numpy(dtype=np.float32),
            "glucose": glucose_series.to_numpy(dtype=np.float32),
            "ecg_features": ecg_features[ECG_COLUMNS].to_numpy(dtype=np.float32),
            "emg_features": emg_features[EMG_COLUMNS].to_numpy(dtype=np.float32),
            "eeg_bands": eeg_bands[EEG_COLUMNS].to_numpy(dtype=np.float32),
            "cbf": cbf_series.to_numpy(dtype=np.float32).reshape(-1, 1),
            "meal_indicator": meal_indicator.astype(np.float32),
            "exercise_indicator": exercise_indicator.astype(np.float32),
            "sleep_stage": sleep_stages,
        },
        "metadata": {
            "n_days": n_days,
            "n_steps": len(timestamps),
            "mean_glucose": float(glucose_series.mean()),
            "hr_resting": float(np.percentile(hr_series.to_numpy(dtype=np.float32), 10)),
        },
    }


def generate_synthetic_cohort(config: dict, *, force: bool = False) -> list[dict[str, Any]]:
    """Generate and save the synthetic cohort used by the non-invasive model."""

    output_dir = Path(config["synthetic_cohort_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["manifest_path"])

    if manifest_path.exists() and not force:
        return load_synthetic_cohort(config)

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    user_id = 0
    for archetype, count in config["archetype_counts"].items():
        for index in range(int(count)):
            record = generate_user(
                user_id=user_id,
                archetype=archetype,
                seed=int(config["seed"]) + (user_id * 17) + index,
                n_days=int(config["synthetic_days_per_user"]),
            )
            torch.save(record, output_dir / f"{user_id}_{archetype}.pt")
            records.append(record)
            manifest_rows.append(
                {
                    "user_id": record["user_id"],
                    "archetype": record["archetype"],
                    "n_days": record["metadata"]["n_days"],
                    "mean_glucose": record["metadata"]["mean_glucose"],
                    "hr_resting": record["metadata"]["hr_resting"],
                    "path": str(output_dir / f"{user_id}_{archetype}.pt"),
                }
            )
            user_id += 1

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return records


def load_synthetic_cohort(config: dict) -> list[dict[str, Any]]:
    """Load the saved synthetic cohort from disk."""

    output_dir = Path(config["synthetic_cohort_dir"])
    records = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(output_dir.glob("*.pt"))
    ]
    return records


def _window_target_states(
    *,
    timestamps: list[str],
    glucose: np.ndarray,
    meal_indicator: np.ndarray,
    exercise_indicator: np.ndarray,
    sleep_stages: list[str],
    end_index: int,
) -> dict[str, Any]:
    """Derive high-level metabolic-state labels for one window endpoint."""

    recent_glucose = glucose[max(0, end_index - 5) : end_index + 1]
    glucose_slope = float(recent_glucose[-1] - recent_glucose[0])
    recent_meal = float(meal_indicator[max(0, end_index - 12) : end_index + 1].sum())
    recent_exercise = float(exercise_indicator[max(0, end_index - 6) : end_index + 1].sum())
    timestamp = pd.Timestamp(timestamps[end_index])
    hour = timestamp.hour + (timestamp.minute / 60.0)
    fasting_state = bool(5.0 <= hour <= 10.0 and recent_meal < 0.5 and recent_exercise < 0.5 and abs(glucose_slope) < 12.0)
    post_meal_state = bool(recent_meal >= 0.5 and glucose_slope > 8.0)
    post_exercise_state = bool(recent_exercise >= 0.5 and glucose_slope < 5.0)
    deep_sleep_state = sleep_stages[end_index] == "deep_sleep"
    return {
        "sleep_stage_at_t": sleep_stages[end_index],
        "fasting_state": fasting_state,
        "post_meal_state": post_meal_state,
        "post_exercise_state": post_exercise_state,
        "deep_sleep_state": deep_sleep_state,
        "recent_meal_load": recent_meal,
        "recent_exercise_load": recent_exercise,
        "glucose_slope_30min": glucose_slope,
    }


def generate_noninvasive_windows(user_signals: dict, window_minutes: int = 30) -> list[dict[str, Any]]:
    """Convert a user's full recordings into non-invasive estimation windows.

    Each window ends at time `t` and predicts the current glucose at `t` from
    the preceding 30 minutes of non-invasive biosignals only.
    """

    timestep_minutes = 5
    window_steps = int(window_minutes // timestep_minutes)
    timestamps = user_signals["timestamps"]
    signals = user_signals["signals"]

    hr = np.asarray(signals["hr"], dtype=np.float32)
    ecg = np.asarray(signals["ecg_features"], dtype=np.float32)
    emg = np.asarray(signals["emg_features"], dtype=np.float32)
    eeg = np.asarray(signals["eeg_bands"], dtype=np.float32)
    cbf = np.asarray(signals["cbf"], dtype=np.float32)
    glucose = np.asarray(signals["glucose"], dtype=np.float32)
    meal_indicator = np.asarray(signals["meal_indicator"], dtype=np.float32)
    exercise_indicator = np.asarray(signals["exercise_indicator"], dtype=np.float32)
    sleep_stages = list(signals["sleep_stage"])

    windows: list[dict[str, Any]] = []
    for end_index in range(window_steps - 1, len(hr)):
        start_index = end_index - window_steps + 1
        state_metadata = _window_target_states(
            timestamps=timestamps,
            glucose=glucose,
            meal_indicator=meal_indicator,
            exercise_indicator=exercise_indicator,
            sleep_stages=sleep_stages,
            end_index=end_index,
        )
        windows.append(
            {
                "hr": hr[start_index : end_index + 1].reshape(window_steps, 1).astype(np.float32),
                "ecg_features": ecg[start_index : end_index + 1].astype(np.float32),
                "emg_features": emg[start_index : end_index + 1].astype(np.float32),
                "eeg_bands": eeg[start_index : end_index + 1].astype(np.float32),
                "cbf": cbf[start_index : end_index + 1].astype(np.float32),
                "glucose_current_raw": float(glucose[end_index]),
                "user_id": int(user_signals["user_id"]),
                "archetype": user_signals["archetype"],
                "archetype_id": int(user_signals["archetype_id"]),
                "timestamp": timestamps[end_index],
                **state_metadata,
            }
        )
    return windows


def _compute_feature_stats(array: np.ndarray, *, axis: int = 0) -> dict[str, Any]:
    """Return feature-wise mean and standard deviation for normalisation."""

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": np.asarray(mean, dtype=np.float32), "std": np.asarray(std, dtype=np.float32)}


def compute_normalisation_stats(train_windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute training-only normalisation statistics for every modality."""

    stats = {
        "hr": _compute_feature_stats(np.concatenate([window["hr"] for window in train_windows], axis=0)),
        "ecg_features": _compute_feature_stats(np.concatenate([window["ecg_features"] for window in train_windows], axis=0)),
        "emg_features": _compute_feature_stats(np.concatenate([window["emg_features"] for window in train_windows], axis=0)),
        "eeg_bands": _compute_feature_stats(np.concatenate([window["eeg_bands"] for window in train_windows], axis=0)),
        "cbf": _compute_feature_stats(np.concatenate([window["cbf"] for window in train_windows], axis=0)),
        "glucose_current": _compute_feature_stats(np.asarray([window["glucose_current_raw"] for window in train_windows], dtype=np.float32)),
    }
    return stats


def _normalise_array(array: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply z-score normalisation using precomputed stats."""

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((array.astype(np.float32) - mean) / std).astype(np.float32)


def apply_normalisation(windows: list[dict[str, Any]], norm_stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise all windows using training-only statistics."""

    normalised: list[dict[str, Any]] = []
    glucose_stats = norm_stats["glucose_current"]
    for window in windows:
        updated = dict(window)
        updated["hr"] = _normalise_array(window["hr"], norm_stats["hr"])
        updated["ecg_features"] = _normalise_array(window["ecg_features"], norm_stats["ecg_features"])
        updated["emg_features"] = _normalise_array(window["emg_features"], norm_stats["emg_features"])
        updated["eeg_bands"] = _normalise_array(window["eeg_bands"], norm_stats["eeg_bands"])
        updated["cbf"] = _normalise_array(window["cbf"], norm_stats["cbf"])
        updated["glucose_current"] = float(
            (np.float32(window["glucose_current_raw"]) - np.float32(glucose_stats["mean"])) / np.float32(glucose_stats["std"])
        )
        normalised.append(updated)
    return normalised


def denormalise_glucose(values: np.ndarray | torch.Tensor | float, norm_stats: dict[str, Any]) -> np.ndarray:
    """Convert normalised glucose values back to mg/dL."""

    glucose_stats = norm_stats["glucose_current"]
    values_array = np.asarray(values, dtype=np.float32)
    return (values_array * np.float32(glucose_stats["std"])) + np.float32(glucose_stats["mean"])


def _split_users_by_archetype(records: list[dict[str, Any]], config: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split users into train/val/test sets while preserving archetype balance."""

    ratios = config["split_ratios"]
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in ARCHETYPES}
    for record in records:
        grouped[record["archetype"]].append(record)

    train_users: list[dict[str, Any]] = []
    val_users: list[dict[str, Any]] = []
    test_users: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["seed"]))

    for archetype, users in grouped.items():
        users_copy = list(users)
        rng.shuffle(users_copy)
        n_total = len(users_copy)
        n_train = max(1, int(round(n_total * ratios["train"])))
        n_val = max(1, int(round(n_total * ratios["val"])))
        n_test = max(1, n_total - n_train - n_val)
        if n_train + n_val + n_test > n_total:
            n_train = max(1, n_total - n_val - n_test)

        train_users.extend(users_copy[:n_train])
        val_users.extend(users_copy[n_train : n_train + n_val])
        test_users.extend(users_copy[n_train + n_val : n_train + n_val + n_test])

    return train_users, val_users, test_users


def build_processed_datasets(config: dict, *, force: bool = False) -> dict[str, Any]:
    """Generate synthetic windows, normalise them, and save processed splits."""

    train_path = Path(config["train_windows_path"])
    val_path = Path(config["val_windows_path"])
    test_path = Path(config["test_windows_path"])
    norm_stats_path = Path(config["norm_stats_path"])

    if all(path.exists() for path in [train_path, val_path, test_path, norm_stats_path]) and not force:
        return load_processed_datasets(config)

    cohort = generate_synthetic_cohort(config, force=force)
    train_users, val_users, test_users = _split_users_by_archetype(cohort, config)

    train_windows_raw = [window for user in train_users for window in generate_noninvasive_windows(user, window_minutes=config["window_minutes"])]
    val_windows_raw = [window for user in val_users for window in generate_noninvasive_windows(user, window_minutes=config["window_minutes"])]
    test_windows_raw = [window for user in test_users for window in generate_noninvasive_windows(user, window_minutes=config["window_minutes"])]

    norm_stats = compute_normalisation_stats(train_windows_raw)
    train_windows = apply_normalisation(train_windows_raw, norm_stats)
    val_windows = apply_normalisation(val_windows_raw, norm_stats)
    test_windows = apply_normalisation(test_windows_raw, norm_stats)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_windows, train_path)
    torch.save(val_windows, val_path)
    torch.save(test_windows, test_path)
    norm_stats_path.write_text(json.dumps(norm_stats, default=_json_default, indent=2), encoding="utf-8")

    return {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "norm_stats": norm_stats,
    }


def load_processed_datasets(config: dict) -> dict[str, Any]:
    """Load the saved processed splits and normalisation stats."""

    return {
        "train_windows": torch.load(config["train_windows_path"], map_location="cpu", weights_only=False),
        "val_windows": torch.load(config["val_windows_path"], map_location="cpu", weights_only=False),
        "test_windows": torch.load(config["test_windows_path"], map_location="cpu", weights_only=False),
        "norm_stats": json.loads(Path(config["norm_stats_path"]).read_text(encoding="utf-8")),
    }


def window_to_model_inputs(window: dict[str, Any], *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Convert one processed window dictionary into model-ready tensors."""

    return {
        "hr": torch.tensor(window["hr"], dtype=torch.float32, device=device).unsqueeze(0),
        "ecg_features": torch.tensor(window["ecg_features"], dtype=torch.float32, device=device).unsqueeze(0),
        "emg_features": torch.tensor(window["emg_features"], dtype=torch.float32, device=device).unsqueeze(0),
        "eeg_bands": torch.tensor(window["eeg_bands"], dtype=torch.float32, device=device).unsqueeze(0),
        "cbf": torch.tensor(window["cbf"], dtype=torch.float32, device=device).unsqueeze(0),
        "user_ids": torch.tensor([window["user_id"]], dtype=torch.long, device=device),
        "archetype_ids": torch.tensor([window["archetype_id"]], dtype=torch.long, device=device),
    }


__all__ = [
    "ARCHETYPES",
    "ARCHETYPE_TO_ID",
    "EEG_COLUMNS",
    "EMG_COLUMNS",
    "ECG_COLUMNS",
    "apply_normalisation",
    "build_processed_datasets",
    "compute_normalisation_stats",
    "denormalise_glucose",
    "generate_noninvasive_windows",
    "generate_synthetic_cohort",
    "generate_user",
    "load_processed_datasets",
    "load_synthetic_cohort",
    "window_to_model_inputs",
]
