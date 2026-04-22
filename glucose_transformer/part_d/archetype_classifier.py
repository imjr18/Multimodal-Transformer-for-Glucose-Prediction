"""Archetype helpers for Part D user initialisation."""

from __future__ import annotations


ARCHETYPE_ORDER = ("athlete", "sedentary", "elderly", "diabetic")
ARCHETYPE_TO_INDEX = {name: index for index, name in enumerate(ARCHETYPE_ORDER)}
INDEX_TO_ARCHETYPE = {index: name for name, index in ARCHETYPE_TO_INDEX.items()}


def archetype_to_index(archetype: str) -> int:
    """Map an archetype label to a stable integer index."""

    try:
        return ARCHETYPE_TO_INDEX[archetype]
    except KeyError as error:
        raise ValueError(f"Unknown archetype: {archetype}") from error


def index_to_archetype(index: int) -> str:
    """Map an archetype index back to its label."""

    try:
        return INDEX_TO_ARCHETYPE[int(index)]
    except KeyError as error:
        raise ValueError(f"Unknown archetype index: {index}") from error


def infer_archetype_from_metadata(metadata: dict) -> str:
    """Infer the closest archetype from lightweight onboarding metadata.

    The synthetic cohort already knows each user's archetype, but real-world
    onboarding may only provide basic physiology. The heuristic below mirrors
    the four Part D cohort definitions closely enough to choose a sensible warm
    start when a direct archetype label is unavailable.
    """

    resting_hr = float(metadata.get("resting_hr", metadata.get("hr_resting", 72.0)))
    mean_glucose = float(metadata.get("mean_glucose", metadata.get("glucose_baseline", 105.0)))
    age = float(metadata.get("age", 45.0))
    exercise_frequency = str(metadata.get("exercise_frequency", "light"))

    if mean_glucose >= 130.0:
        return "diabetic"
    if exercise_frequency in {"daily", "frequent"} and resting_hr <= 58.0:
        return "athlete"
    if age >= 65.0:
        return "elderly"
    return "sedentary"

