"""Scenario mining and physiological label extraction for Part E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from part_d.cohort_simulator import ECG_COLUMNS, EMG_COLUMNS
from preprocessing.eeg_simulation import extract_band_power_sequence, infer_sleep_stage_from_band_powers


def derive_window_labels(window_entry: dict[str, Any]) -> dict[str, Any]:
    """Derive interpretable labels from one synthetic-cohort window."""

    metadata = window_entry["metadata"]
    raw = metadata["raw"]
    timestamps = pd.to_datetime(metadata["timestamps"])
    end_time = pd.Timestamp(metadata["end_time"])

    hr = np.asarray(raw["hr"], dtype=np.float32)
    glucose = np.asarray(raw["glucose"], dtype=np.float32)
    ecg = np.asarray(raw["ecg_features"], dtype=np.float32)
    emg = np.asarray(raw["emg_features"], dtype=np.float32)
    eeg = np.asarray(raw["eeg"], dtype=np.float32)
    glucose_target = np.asarray(raw["glucose_target"], dtype=np.float32)

    band_power_sequence = extract_band_power_sequence(eeg, sfreq=256, window_seconds=1)
    sleep_stage = infer_sleep_stage_from_band_powers(band_power_sequence) if band_power_sequence.size else "awake"
    mean_band_powers = band_power_sequence.mean(axis=0) if band_power_sequence.size else np.zeros(5, dtype=np.float32)

    rms_envelope = emg[:, 0]
    sdnn = ecg[:, 0]
    hf_power = ecg[:, 3]
    glucose_rise = float(glucose[-1] - glucose[0])
    target_rise = float(glucose_target[-1] - glucose[0])
    hr_recovery = float(hr.max() - hr[-1])

    post_meal = bool(
        glucose[0] < 100.0
        and glucose.max() > 130.0
        and glucose_rise > 12.0
        and rms_envelope.max() < 0.22
    )
    post_exercise = bool(
        rms_envelope.max() > 0.30
        and hr.max() > np.mean(hr) + 10.0
        and hr_recovery > 3.0
    )
    dawn_window = bool(
        4 <= end_time.hour <= 7
        and target_rise > 10.0
        and rms_envelope.max() < 0.18
    )
    nocturnal_stability = bool(
        2 <= end_time.hour <= 4
        and sleep_stage == "deep_sleep"
        and float(np.std(glucose)) < 5.0
        and rms_envelope.max() < 0.12
    )

    return {
        "sleep_stage": sleep_stage,
        "archetype": metadata["archetype"],
        "is_athlete": metadata["archetype"] == "athlete",
        "is_healthy_user": metadata["archetype"] in {"athlete", "sedentary"},
        "post_meal": post_meal,
        "post_exercise": post_exercise,
        "dawn_phenomenon": dawn_window and metadata["archetype"] == "diabetic",
        "nocturnal_stability": nocturnal_stability and metadata["archetype"] in {"athlete", "sedentary"},
        "target_glucose_mean": float(glucose_target.mean()),
        "glucose_rise": glucose_rise,
        "target_rise": target_rise,
        "hr_recovery": hr_recovery,
        "mean_delta": float(mean_band_powers[0]),
        "mean_theta": float(mean_band_powers[1]),
        "mean_alpha": float(mean_band_powers[2]),
        "mean_beta": float(mean_band_powers[3]),
        "mean_gamma": float(mean_band_powers[4]),
        "max_rms_envelope": float(rms_envelope.max()),
        "mean_sdnn": float(sdnn.mean()),
        "mean_hf_power": float(hf_power.mean()),
        "window_end_hour": int(end_time.hour),
    }


def find_biological_scenario_windows(
    dataset,
    *,
    split: str,
    max_matches: int,
    max_windows_per_user: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Find windows matching the four Part E biological scenarios."""

    scenarios = {
        "post_meal": [],
        "post_exercise_athlete": [],
        "dawn_phenomenon_diabetic": [],
        "nocturnal_stability_healthy": [],
    }

    for window_entry in dataset.iter_split_windows(
        split,
        max_windows_per_user=max_windows_per_user,
    ):
        labels = derive_window_labels(window_entry)
        window_entry = dict(window_entry)
        window_entry["labels"] = labels

        if labels["post_meal"] and len(scenarios["post_meal"]) < max_matches:
            scenarios["post_meal"].append(window_entry)
        if labels["post_exercise"] and labels["is_athlete"] and len(scenarios["post_exercise_athlete"]) < max_matches:
            scenarios["post_exercise_athlete"].append(window_entry)
        if labels["dawn_phenomenon"] and len(scenarios["dawn_phenomenon_diabetic"]) < max_matches:
            scenarios["dawn_phenomenon_diabetic"].append(window_entry)
        if labels["nocturnal_stability"] and len(scenarios["nocturnal_stability_healthy"]) < max_matches:
            scenarios["nocturnal_stability_healthy"].append(window_entry)

        if all(len(matches) >= max_matches for matches in scenarios.values()):
            break

    return scenarios


__all__ = ["derive_window_labels", "find_biological_scenario_windows"]

