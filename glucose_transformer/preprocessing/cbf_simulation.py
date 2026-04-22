"""Synthetic CBF generation for Part C."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_cbf(hr_series: pd.Series, glucose_series: pd.Series) -> pd.Series:
    """Simulate cerebral blood flow at the 5-minute resolution of the Ohio windows.

    The simulation combines a baseline level with posture, exercise, glucose,
    and slow drift effects. The 2-minute exercise lag from the prompt is
    approximated at 5-minute resolution with a one-step delayed high-HR term.
    """

    hr = hr_series.astype("float32")
    glucose = glucose_series.astype("float32")
    index = hr.index
    rng_seed = int(float(hr.mean()) * 10 + float(glucose.mean()) * 10 + len(hr))
    rng = np.random.default_rng(rng_seed % (2**32 - 1))

    baseline = np.full(len(hr), 50.0, dtype=np.float32)
    if isinstance(index, pd.DatetimeIndex):
        hours = np.array([timestamp.hour + (timestamp.minute / 60.0) for timestamp in index], dtype=np.float32)
        waking_hours = (hours >= 7.0) & (hours < 22.0)
        sleeping_hours = ~waking_hours
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
    return pd.Series(cbf.astype("float32"), index=index, name="cbf_ml_100g_min")


__all__ = ["generate_synthetic_cbf"]
