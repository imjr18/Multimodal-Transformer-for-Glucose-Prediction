"""Synthetic EEG generation and EEG feature helpers for Part C."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def _make_rng(hr_series: pd.Series, glucose_series: pd.Series, sfreq: int) -> np.random.Generator:
    """Create a deterministic RNG keyed by the input window."""

    if isinstance(hr_series.index, pd.DatetimeIndex):
        timestamp_seed = int(pd.Timestamp(hr_series.index[-1]).value % (2**32 - 1))
    else:
        timestamp_seed = int(len(hr_series) * 1009)
    value_seed = int(
        float(np.nanmean(hr_series.to_numpy(dtype="float64")) * 10)
        + float(np.nanmean(glucose_series.to_numpy(dtype="float64")) * 10)
        + sfreq
    )
    return np.random.default_rng((timestamp_seed + value_seed) % (2**32 - 1))


def _interpolated_terminal_signal(
    series: pd.Series,
    *,
    window_seconds: int,
    sfreq: int,
) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
    """Linearly interpolate the terminal part of a 5-minute series to EEG rate."""

    n_samples = window_seconds * sfreq
    signal = series.astype("float32")

    coarse_times = np.linspace(
        -((len(signal) - 1) * 300.0),
        0.0,
        len(signal),
        dtype=np.float32,
    )
    fine_times = np.linspace(-float(window_seconds), 0.0, n_samples, endpoint=False, dtype=np.float32)
    interpolated = np.interp(fine_times, coarse_times, signal.to_numpy(dtype=np.float32))

    if isinstance(signal.index, pd.DatetimeIndex):
        end_time = pd.Timestamp(signal.index[-1])
        fine_index = pd.date_range(
            end=end_time,
            periods=n_samples,
            freq=pd.to_timedelta(1 / sfreq, unit="s"),
        )
    else:
        fine_index = None

    return interpolated.astype("float32"), fine_index


def generate_synthetic_eeg(
    hr_series: pd.Series,
    glucose_series: pd.Series,
    sfreq: int = 256,
) -> np.ndarray:
    """Generate synthetic EEG coupled to HR, glucose, sleep state, and clock time.

    The function upsamples the terminal 2-minute portion of the heart-rate and
    glucose traces to 256 Hz, infers sleep stage from the interpolated HR level
    and clock time, then synthesises a stage-dependent oscillatory EEG with pink
    noise. The resulting signal is not intended as a clinical simulator, but it
    is structured enough to make the efficient-EEG experiments meaningful.
    """

    eeg_window_seconds = 120
    rng = _make_rng(hr_series, glucose_series, sfreq)

    interpolated_hr, fine_index = _interpolated_terminal_signal(
        hr_series,
        window_seconds=eeg_window_seconds,
        sfreq=sfreq,
    )
    interpolated_glucose, _ = _interpolated_terminal_signal(
        glucose_series,
        window_seconds=eeg_window_seconds,
        sfreq=sfreq,
    )

    n_samples = eeg_window_seconds * sfreq
    time_axis = np.arange(n_samples, dtype=np.float32) / float(sfreq)

    if fine_index is not None:
        hours = np.array([timestamp.hour + (timestamp.minute / 60.0) for timestamp in fine_index], dtype=np.float32)
        sleeping = (hours >= 22.0) | (hours < 6.0)
    else:
        sleeping = np.zeros(n_samples, dtype=bool)

    deep_sleep = sleeping & (interpolated_hr < 55.0)
    light_sleep = sleeping & (interpolated_hr >= 55.0) & (interpolated_hr < 65.0)
    awake = ~(deep_sleep | light_sleep)

    dominant_frequency = np.where(deep_sleep, 2.0, np.where(light_sleep, 6.0, 20.0)).astype("float32")
    amplitude = np.where(deep_sleep, 80.0, np.where(light_sleep, 40.0, 15.0)).astype("float32")
    noise_scale = np.where(deep_sleep, 0.3, np.where(light_sleep, 0.2, 0.4)).astype("float32")

    phase = 2.0 * np.pi * np.cumsum(dominant_frequency, dtype=np.float32) / float(sfreq)
    glucose_modulation = 1.0 + np.clip((110.0 - interpolated_glucose) / 250.0, -0.15, 0.15).astype("float32")
    rhythmic_signal = amplitude * glucose_modulation * np.sin(phase + (0.1 * time_axis))
    stage_noise = noise_scale * amplitude * rng.standard_normal(n_samples, dtype=np.float32)

    pink_noise = np.cumsum(rng.standard_normal(n_samples, dtype=np.float32)) * 0.1
    pink_noise = pink_noise - pink_noise.mean()
    pink_noise = pink_noise / (pink_noise.std() + 1e-6)
    pink_noise = pink_noise * 5.0

    eeg_signal = rhythmic_signal + stage_noise + pink_noise.astype("float32")
    return eeg_signal.astype("float32")


def extract_band_powers(eeg_segment: np.ndarray, sfreq: int = 256) -> np.ndarray:
    """Compute relative power in the canonical EEG frequency bands.

    Welch PSD estimation provides a stable per-segment estimate of band power.
    Each band's absolute power is divided by total power so the returned vector
    reflects the relative dominance of that band rather than raw amplitude.
    """

    signal = np.asarray(eeg_segment, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return np.zeros(5, dtype=np.float32)

    frequencies, power = welch(signal, fs=sfreq, nperseg=min(signal.size, sfreq))
    band_powers: list[float] = []
    for low, high in EEG_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = float(np.trapezoid(power[mask], frequencies[mask])) if np.any(mask) else 0.0
        band_powers.append(band_power)

    band_powers_array = np.asarray(band_powers, dtype=np.float32)
    total_power = float(band_powers_array.sum())
    if total_power <= 0:
        return np.zeros_like(band_powers_array)
    return (band_powers_array / total_power).astype("float32")


def extract_band_power_sequence(
    eeg_signal: np.ndarray,
    *,
    sfreq: int = 256,
    window_seconds: int = 1,
) -> np.ndarray:
    """Compute relative band powers for non-overlapping EEG windows."""

    signal = np.asarray(eeg_signal, dtype=np.float32).reshape(-1)
    samples_per_window = int(window_seconds * sfreq)
    n_windows = signal.size // samples_per_window
    if n_windows == 0:
        return np.zeros((0, 5), dtype=np.float32)

    segments = signal[: n_windows * samples_per_window].reshape(n_windows, samples_per_window)
    band_powers = [extract_band_powers(segment, sfreq=sfreq) for segment in segments]
    return np.stack(band_powers, axis=0).astype("float32")


def infer_sleep_stage_from_band_powers(band_powers: np.ndarray) -> str:
    """Map mean relative band powers to a coarse sleep/wake label."""

    mean_band_powers = np.asarray(band_powers, dtype=np.float32).mean(axis=0)
    delta_power, theta_power, _, beta_power, _ = mean_band_powers.tolist()
    if delta_power > 0.4:
        return "deep_sleep"
    if theta_power > 0.3 and delta_power < 0.3:
        return "light_sleep"
    if beta_power > 0.3:
        return "awake"
    return "awake"


__all__ = [
    "EEG_BANDS",
    "extract_band_power_sequence",
    "extract_band_powers",
    "generate_synthetic_eeg",
    "infer_sleep_stage_from_band_powers",
]
