"""Feature-extraction helpers documented for Part B and future waveform work."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch


def extract_ecg_features_from_hrv(rr_intervals: np.ndarray) -> dict:
    """Extract standard HRV features from RR intervals.

    For real ECG in later parts, RR intervals would come from R-peak detection
    on the waveform. Here the function provides a documented, testable feature
    extractor covering the core Part B quantities: SDNN, RMSSD, pNN50, LF, HF,
    and LF/HF. Frequency-domain power is estimated from an interpolated RR
    series using Welch's method.
    """

    rr = np.asarray(rr_intervals, dtype=np.float32).reshape(-1)
    if rr.size < 2:
        return {
            "sdnn": 0.0,
            "rmssd": 0.0,
            "pnn50": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
        }

    rr_diff = np.diff(rr)
    sdnn = float(np.std(rr, ddof=0))
    rmssd = float(np.sqrt(np.mean(rr_diff**2))) if rr_diff.size else 0.0
    pnn50 = float(np.mean(np.abs(rr_diff) > 0.05))

    cumulative_time = np.cumsum(rr)
    interpolation_times = np.arange(0.0, cumulative_time[-1], 0.25, dtype=np.float32)
    if interpolation_times.size < 8:
        lf_power = 0.0
        hf_power = 0.0
    else:
        interpolated_rr = np.interp(interpolation_times, cumulative_time, rr)
        frequencies, power = welch(interpolated_rr, fs=4.0, nperseg=min(256, interpolated_rr.size))
        lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)
        lf_power = float(np.trapz(power[lf_mask], frequencies[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(np.trapz(power[hf_mask], frequencies[hf_mask])) if np.any(hf_mask) else 0.0

    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else 0.0
    return {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


def extract_emg_envelope(emg_raw: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Extract an RMS envelope from raw EMG.

    For real EMG, the standard pipeline is bandpass filtering, full-wave
    rectification, and RMS pooling. Part B uses pre-generated synthetic
    envelopes, but this function documents and implements the waveform-side
    transformation expected in later extensions.
    """

    emg = np.asarray(emg_raw, dtype=np.float32).reshape(-1)
    if emg.size == 0:
        return emg

    if emg.size >= 20:
        b, a = butter(N=4, Wn=[20.0, 450.0], btype="bandpass", fs=1000.0)
        filtered = filtfilt(b, a, emg)
    else:
        filtered = emg

    rectified = np.abs(filtered)
    squared = rectified**2
    kernel = np.ones(window_size, dtype=np.float32) / max(window_size, 1)
    moving_average = np.convolve(squared, kernel, mode="same")
    return np.sqrt(np.clip(moving_average, 0.0, None)).astype(np.float32)
