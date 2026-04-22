"""Calibration-session construction for new-user adaptation."""

from __future__ import annotations

from typing import Any

from noninvasive_glucose.simulation.noninvasive_simulator import generate_noninvasive_windows


def _pick_best(candidates: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    """Pick the candidate window with the strongest score on a given key."""

    if not candidates:
        return None
    return sorted(candidates, key=lambda item: float(item.get(key, 0.0)), reverse=True)[0]


def generate_calibration_session(user_signals: dict, n_readings: int = 3) -> list[tuple[dict[str, Any], float]]:
    """Select strategically spaced calibration pairs for one user.

    The session tries to cover fasting, post-meal, and post-exercise states.
    If one of these cannot be found, the remaining slots are filled with diverse
    windows sorted by absolute glucose slope and event intensity.
    """

    windows = generate_noninvasive_windows(user_signals, window_minutes=30)
    fasting = [window for window in windows if bool(window.get("fasting_state"))]
    post_meal = [window for window in windows if bool(window.get("post_meal_state"))]
    post_exercise = [window for window in windows if bool(window.get("post_exercise_state"))]

    selected: list[dict[str, Any]] = []
    for candidate in [
        _pick_best(fasting, key="glucose_current_raw"),
        _pick_best(post_meal, key="recent_meal_load"),
        _pick_best(post_exercise, key="recent_exercise_load"),
    ]:
        if candidate is not None:
            selected.append(candidate)

    if len(selected) < n_readings:
        fallback = sorted(
            windows,
            key=lambda window: (
                abs(float(window.get("glucose_slope_30min", 0.0))),
                float(window.get("recent_meal_load", 0.0)) + float(window.get("recent_exercise_load", 0.0)),
            ),
            reverse=True,
        )
        seen_timestamps = {window["timestamp"] for window in selected}
        for window in fallback:
            if window["timestamp"] in seen_timestamps:
                continue
            selected.append(window)
            seen_timestamps.add(window["timestamp"])
            if len(selected) >= n_readings:
                break

    return [
        (window, float(window["glucose_current_raw"]))
        for window in selected[:n_readings]
    ]


__all__ = ["generate_calibration_session"]

