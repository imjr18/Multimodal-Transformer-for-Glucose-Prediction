"""OhioT1DM preprocessing utilities for Part A.

The pipeline in this module deliberately stays narrow:
it extracts only glucose and heart-rate data, aligns them on a 5-minute grid,
normalises them with training-only statistics, and converts them into sliding
windows suitable for the Part A models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

try:
    from lxml import etree
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments.
    import xml.etree.ElementTree as etree


PATIENT_FILE_PATTERN = re.compile(r"(?P<patient_id>\d+)-ws-(training|testing)\.xml$")
SIGNAL_COLUMNS = ("glucose_mg_dl", "heart_rate_bpm")


def _extract_patient_id(filepath: str | Path) -> int:
    """Infer the Ohio patient identifier from the XML filename."""

    match = PATIENT_FILE_PATTERN.search(Path(filepath).name)
    if match is None:
        raise ValueError(f"Could not infer patient ID from filename: {filepath}")
    return int(match.group("patient_id"))


def _locate_signal_section(root: Any, candidate_tags: Iterable[str]) -> Any | None:
    """Return the first matching XML section for a requested signal tag."""

    for tag in candidate_tags:
        section = root.find(f"./{tag}")
        if section is not None:
            return section

        nested_sections = root.findall(f".//{tag}")
        if nested_sections:
            return nested_sections[0]
    return None


def _events_to_frame(section: Any | None, value_name: str) -> pd.DataFrame:
    """Convert a signal section with `<event>` children into a typed dataframe."""

    if section is None:
        return pd.DataFrame(columns=["timestamp", value_name])

    records: list[dict[str, object]] = []
    for event in section.findall("./event"):
        timestamp = (
            event.attrib.get("ts")
            or event.attrib.get("timestamp")
            or event.attrib.get("date")
            or event.attrib.get("ts_begin")
        )
        value = event.attrib.get("value")
        if timestamp is None:
            continue
        records.append({"timestamp": timestamp, value_name: value})

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", value_name])

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.groupby("timestamp", as_index=False).mean(numeric_only=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def parse_ohio_xml(filepath: str) -> pd.DataFrame:
    """Parse one OhioT1DM XML file into glucose and heart-rate columns.

    The raw XML stores each signal in a separate section with repeated `event`
    elements. This function keeps missing values as `NaN` so the alignment step
    can distinguish short gaps that may be forward-filled from long gaps that
    must later be excluded from model windows.
    """

    tree = etree.parse(str(filepath))
    root = tree.getroot()

    glucose_section = _locate_signal_section(root, ("glucose_level",))
    heart_rate_section = _locate_signal_section(root, ("heart_rate", "basis_heart_rate"))

    glucose_frame = _events_to_frame(glucose_section, "glucose_mg_dl")
    heart_rate_frame = _events_to_frame(heart_rate_section, "heart_rate_bpm")

    parsed = pd.merge(glucose_frame, heart_rate_frame, on="timestamp", how="outer")
    parsed = parsed.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    parsed = parsed.reset_index(drop=True)
    return parsed[["timestamp", "glucose_mg_dl", "heart_rate_bpm"]]


def align_to_grid(
    df: pd.DataFrame,
    *,
    frequency: str = "5min",
    ffill_limit: int = 3,
) -> pd.DataFrame:
    """Align glucose and heart-rate values to a uniform 5-minute grid.

    OhioT1DM events can be irregular or partially missing. We resample onto a
    fixed 5-minute grid, forward-fill only short gaps up to 15 minutes, and
    leave longer gaps as `NaN`. The later window builder then drops any window
    that still touches these longer invalid regions.
    """

    if df.empty:
        return pd.DataFrame(columns=["timestamp", *SIGNAL_COLUMNS])

    aligned = df.copy()
    aligned = aligned.sort_values("timestamp").set_index("timestamp")
    aligned = aligned.resample(frequency).mean()
    aligned = aligned.ffill(limit=ffill_limit)
    aligned = aligned.reset_index()
    aligned["glucose_mg_dl"] = aligned["glucose_mg_dl"].astype("float32")
    aligned["heart_rate_bpm"] = aligned["heart_rate_bpm"].astype("float32")
    return aligned


def per_patient_normalise(
    df: pd.DataFrame,
    stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply z-score normalisation with explicit reusable statistics.

    The prompt calls for per-patient normalisation but also forbids deriving
    statistics from validation or test patients. To satisfy the no-leakage
    requirement, this function accepts externally supplied training statistics.
    When `stats` is omitted it computes them from the provided dataframe, which
    is useful for training-data-only statistics creation.
    """

    normalised = df.copy()

    if stats is None:
        stats = {}
        for column in SIGNAL_COLUMNS:
            series = normalised[column].dropna()
            mean = float(series.mean()) if not series.empty else 0.0
            std = float(series.std(ddof=0)) if not series.empty else 1.0
            stats[column] = {
                "mean": mean,
                "std": std if std > 0 else 1.0,
            }

    for column in SIGNAL_COLUMNS:
        mean = float(stats[column]["mean"])
        std = float(stats[column]["std"])
        std = std if std > 0 else 1.0
        normalised[column] = ((normalised[column] - mean) / std).astype("float32")

    return normalised, stats


def create_windows(
    df: pd.DataFrame,
    *,
    patient_id: int,
    input_len: int = 24,
    target_offsets: list[int] | tuple[int, ...] = (6, 12),
    stride: int = 1,
) -> list[dict]:
    """Convert an aligned patient dataframe into overlapping forecasting windows.

    Each example contains 24 historical heart-rate values, 24 glucose context
    values, and two future glucose targets. Windows touching unresolved long
    gaps are discarded because those rows still contain `NaN` after the limited
    forward-fill in `align_to_grid`.
    """

    windows: list[dict] = []
    if df.empty:
        return windows

    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    max_offset = max(target_offsets)
    last_start = len(sorted_df) - input_len - max_offset + 1

    for start_idx in range(0, max(last_start, 0), stride):
        end_idx = start_idx + input_len
        input_slice = sorted_df.iloc[start_idx:end_idx]
        target_rows = [sorted_df.iloc[end_idx - 1 + offset] for offset in target_offsets]

        if input_slice[list(SIGNAL_COLUMNS)].isna().any().any():
            continue
        if any(pd.isna(target_row["glucose_mg_dl"]) for target_row in target_rows):
            continue

        windows.append(
            {
                "hr_input": torch.tensor(
                    input_slice["heart_rate_bpm"].to_numpy(),
                    dtype=torch.float32,
                ),
                "glucose_input": torch.tensor(
                    input_slice["glucose_mg_dl"].to_numpy(),
                    dtype=torch.float32,
                ),
                "glucose_target": torch.tensor(
                    [float(target_row["glucose_mg_dl"]) for target_row in target_rows],
                    dtype=torch.float32,
                ),
                "patient_id": int(patient_id),
                "timestamp": input_slice.iloc[-1]["timestamp"].to_pydatetime(),
            }
        )

    return windows


def save_processed(
    windows: list[dict],
    output_dir: str | Path,
    split_name: str,
    norm_stats: dict | None = None,
) -> dict[str, str]:
    """Persist processed windows and optional normalisation statistics to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    windows_path = output_path / f"{split_name}_windows.pt"
    stats_path = output_path / f"{split_name}_stats.json"

    torch.save(windows, windows_path)
    if norm_stats is not None:
        stats_path.write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")

    return {
        "windows_path": str(windows_path),
        "stats_path": str(stats_path),
    }


def _discover_patient_files(raw_dir: str | Path, patient_id: int, year: str = "2018") -> list[Path]:
    """Locate all XML files that belong to a patient in the raw-data folder."""

    raw_path = Path(raw_dir)
    preferred_paths = [
        raw_path / "OhioT1DM" / year / "train" / f"{patient_id}-ws-training.xml",
        raw_path / "OhioT1DM" / year / "test" / f"{patient_id}-ws-testing.xml",
        raw_path / f"{patient_id}-ws-training.xml",
        raw_path / f"{patient_id}-ws-testing.xml",
    ]

    discovered_paths = [path for path in preferred_paths if path.exists()]
    recursive_paths = sorted(raw_path.rglob(f"{patient_id}-ws-*.xml"))

    unique_paths: list[Path] = []
    for path in [*discovered_paths, *recursive_paths]:
        if path not in unique_paths:
            unique_paths.append(path)

    return unique_paths


def load_patient_dataframe(raw_dir: str | Path, patient_id: int, year: str = "2018") -> pd.DataFrame:
    """Load and merge every available XML file for one patient."""

    xml_paths = _discover_patient_files(raw_dir, patient_id, year)
    if not xml_paths:
        raise FileNotFoundError(
            f"No OhioT1DM XML files found for patient {patient_id} under {raw_dir}."
        )

    frames = [parse_ohio_xml(str(xml_path)) for xml_path in xml_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def _compute_training_stats(patient_frames: dict[int, pd.DataFrame]) -> dict:
    """Compute pooled signal statistics using only the designated training patients."""

    concatenated = pd.concat(patient_frames.values(), ignore_index=True)
    _, training_stats = per_patient_normalise(concatenated)

    training_stats["meta"] = {
        "computed_from": "train_patients_only",
        "train_patient_ids": sorted(int(patient_id) for patient_id in patient_frames),
    }
    training_stats["per_training_patient"] = {}

    for patient_id, patient_frame in patient_frames.items():
        _, patient_stats = per_patient_normalise(patient_frame)
        training_stats["per_training_patient"][str(patient_id)] = patient_stats

    return training_stats


def preprocess_ohio_dataset(config: dict) -> dict:
    """Run the full OhioT1DM preprocessing pipeline and save split tensors."""

    processed_dir = Path(config["data_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_patients = {
        "train": [int(patient_id) for patient_id in config["train_patients"]],
        "val": [int(patient_id) for patient_id in config["val_patients"]],
        "test": [int(patient_id) for patient_id in config["test_patients"]],
    }

    all_patient_ids = [
        *split_patients["train"],
        *split_patients["val"],
        *split_patients["test"],
    ]

    aligned_patient_frames: dict[int, pd.DataFrame] = {}
    for patient_id in all_patient_ids:
        patient_frame = load_patient_dataframe(
            config["data_raw_dir"],
            patient_id,
            config["ohio_year"],
        )
        aligned_frame = align_to_grid(
            patient_frame,
            frequency=config["resample_frequency"],
            ffill_limit=config["ffill_limit"],
        )
        aligned_frame["patient_id"] = patient_id
        aligned_patient_frames[patient_id] = aligned_frame

    training_stats = _compute_training_stats(
        {patient_id: aligned_patient_frames[patient_id] for patient_id in split_patients["train"]}
    )

    manifest = {
        "splits": {},
        "normalisation_stats_path": str(processed_dir / "normalisation_stats.json"),
    }

    for split_name, patient_ids in split_patients.items():
        split_windows: list[dict] = []
        for patient_id in patient_ids:
            normalised_frame, _ = per_patient_normalise(
                aligned_patient_frames[patient_id],
                stats=training_stats,
            )
            split_windows.extend(
                create_windows(
                    normalised_frame,
                    patient_id=patient_id,
                    input_len=config["input_len"],
                    target_offsets=config["target_offsets"],
                    stride=config["window_stride"],
                )
            )

        saved_paths = save_processed(split_windows, processed_dir, split_name, training_stats)
        manifest["splits"][split_name] = {
            "patients": patient_ids,
            "num_windows": len(split_windows),
            **saved_paths,
        }

    norm_stats_path = processed_dir / "normalisation_stats.json"
    norm_stats_path.write_text(json.dumps(training_stats, indent=2), encoding="utf-8")

    manifest_path = processed_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "align_to_grid",
    "create_windows",
    "load_patient_dataframe",
    "parse_ohio_xml",
    "per_patient_normalise",
    "preprocess_ohio_dataset",
    "save_processed",
]
