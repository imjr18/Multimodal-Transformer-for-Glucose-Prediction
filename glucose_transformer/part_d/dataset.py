"""Task construction utilities for Part D meta-learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from part_d.archetype_classifier import archetype_to_index
from part_d.cohort_simulator import ECG_COLUMNS, EMG_COLUMNS
from preprocessing.eeg_simulation import generate_synthetic_eeg


def _init_running_scalar_stats() -> dict[str, float]:
    """Create a running scalar-statistics accumulator."""

    return {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}


def _update_scalar_stats(accumulator: dict[str, float], values: np.ndarray) -> None:
    """Update running scalar statistics."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    accumulator["count"] += float(array.size)
    accumulator["sum"] += float(array.sum())
    accumulator["sum_sq"] += float(np.square(array).sum())


def _finalise_scalar_stats(accumulator: dict[str, float]) -> dict[str, float]:
    """Convert running scalar sums into mean/std."""

    count = max(accumulator["count"], 1.0)
    mean = accumulator["sum"] / count
    variance = max((accumulator["sum_sq"] / count) - (mean**2), 1e-8)
    return {"mean": float(mean), "std": float(np.sqrt(variance))}


def _candidate_window_count(n_steps: int, input_len: int, target_offsets: list[int]) -> int:
    """Return the number of valid sliding windows for one user."""

    return max(int(n_steps) - int(input_len) - max(target_offsets) + 1, 0)


def _chunk_tensor_dict(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-window tensors into a batched task tensor dictionary."""

    stacked: dict[str, list[torch.Tensor]] = {}
    for sample in samples:
        for key, value in sample.items():
            stacked.setdefault(key, []).append(value)

    return {
        key: torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        for key, values in stacked.items()
    }


class MetaLearningDataset:
    """Organise the synthetic cohort into user-level support/query tasks.

    Each user contributes a sequence of forecast windows created on demand from
    the stored 5-minute multimodal signals. EEG is regenerated lazily from the
    local HR and glucose context so the cohort remains lightweight enough to
    store and iterate over on commodity hardware.
    """

    def __init__(self, config: dict):
        self.config = config
        self.cohort_dir = Path(config["synthetic_cohort_dir"])
        self.manifest_path = Path(config["synthetic_cohort_manifest_path"])
        self.split_path = Path(config["synthetic_cohort_split_path"])
        self.norm_stats_path = Path(config["synthetic_cohort_norm_stats_path"])

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Synthetic cohort manifest not found: {self.manifest_path}. "
                "Generate the cohort before building meta-learning tasks."
            )

        self.manifest = pd.read_csv(self.manifest_path)
        self.user_cache: dict[int, dict] = {}
        self.user_paths = {
            int(row.user_id): Path(row.file_path)
            for row in self.manifest.itertuples(index=False)
        }
        self.splits = self._load_or_create_splits()
        self.norm_stats = self._load_or_create_norm_stats()

    def _load_user(self, user_id: int) -> dict:
        """Load one user record and cache it in CPU memory."""

        user_id = int(user_id)
        if user_id not in self.user_cache:
            self.user_cache[user_id] = torch.load(self.user_paths[user_id], map_location="cpu", weights_only=False)
        return self.user_cache[user_id]

    def get_user_record(self, user_id: int) -> dict:
        """Return the cached raw synthetic user record."""

        return self._load_user(int(user_id))

    def _load_or_create_splits(self) -> dict[str, list[int]]:
        """Load a saved cohort split or create a stratified one."""

        if self.split_path.exists():
            return json.loads(self.split_path.read_text(encoding="utf-8"))

        rng = np.random.default_rng(self.config["seed"])
        splits = {"train": [], "val": [], "test": []}

        for archetype, archetype_frame in self.manifest.groupby("archetype", sort=True):
            user_ids = archetype_frame["user_id"].to_numpy(dtype=np.int64)
            user_ids = rng.permutation(user_ids)
            n_users = len(user_ids)
            n_train = int(round(n_users * float(self.config["train_split_ratio"])))
            n_val = int(round(n_users * float(self.config["val_split_ratio"])))
            n_train = min(n_train, n_users)
            n_val = min(n_val, max(n_users - n_train, 0))
            n_test = n_users - n_train - n_val

            splits["train"].extend(int(user_id) for user_id in user_ids[:n_train])
            splits["val"].extend(int(user_id) for user_id in user_ids[n_train:n_train + n_val])
            splits["test"].extend(int(user_id) for user_id in user_ids[n_train + n_val:n_train + n_val + n_test])

        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])

        self.split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
        return splits

    def _load_or_create_norm_stats(self) -> dict:
        """Compute training-only normalisation statistics for every modality."""

        if self.norm_stats_path.exists():
            return json.loads(self.norm_stats_path.read_text(encoding="utf-8"))

        hr_stats = _init_running_scalar_stats()
        glucose_stats = _init_running_scalar_stats()
        cbf_stats = _init_running_scalar_stats()
        eeg_stats = _init_running_scalar_stats()
        ecg_accumulators = {column: _init_running_scalar_stats() for column in ECG_COLUMNS}
        emg_accumulators = {column: _init_running_scalar_stats() for column in EMG_COLUMNS}

        eeg_windows_per_user = int(max(1, self.config["eeg_stats_windows_per_user"]))
        input_len = int(self.config["input_len"])
        target_offsets = list(self.config["target_offsets"])
        max_target_offset = max(target_offsets)

        for user_id in self.splits["train"]:
            user_record = self._load_user(user_id)
            signals = user_record["signals"]
            _update_scalar_stats(hr_stats, signals["hr"])
            _update_scalar_stats(glucose_stats, signals["glucose"])
            _update_scalar_stats(cbf_stats, signals["cbf"])

            ecg_matrix = np.asarray(signals["ecg_features"], dtype=np.float32)
            for column_index, column_name in enumerate(ECG_COLUMNS):
                _update_scalar_stats(ecg_accumulators[column_name], ecg_matrix[:, column_index])

            emg_matrix = np.asarray(signals["emg_features"], dtype=np.float32)
            for column_index, column_name in enumerate(EMG_COLUMNS):
                _update_scalar_stats(emg_accumulators[column_name], emg_matrix[:, column_index])

            n_steps = int(user_record["metadata"]["n_steps"])
            n_windows = _candidate_window_count(n_steps, input_len, target_offsets)
            if n_windows <= 0:
                continue

            sample_points = np.linspace(0, max(n_windows - 1, 0), num=min(eeg_windows_per_user, n_windows), dtype=int)
            start_time = pd.Timestamp(user_record["start_time"])
            for start_idx in np.unique(sample_points):
                end_idx = int(start_idx + input_len)
                window_timestamps = pd.date_range(
                    start=start_time + pd.to_timedelta(int(start_idx) * 5, unit="min"),
                    periods=input_len,
                    freq=self.config["resample_frequency"],
                )
                hr_window = pd.Series(np.asarray(signals["hr"], dtype=np.float32)[start_idx:end_idx], index=window_timestamps)
                glucose_window = pd.Series(np.asarray(signals["glucose"], dtype=np.float32)[start_idx:end_idx], index=window_timestamps)
                eeg_window = generate_synthetic_eeg(
                    hr_window,
                    glucose_window,
                    sfreq=int(self.config["eeg_sfreq"]),
                )
                _update_scalar_stats(eeg_stats, eeg_window)

        stats = {
            "heart_rate_bpm": _finalise_scalar_stats(hr_stats),
            "glucose_mg_dl": _finalise_scalar_stats(glucose_stats),
            "cbf_signal": _finalise_scalar_stats(cbf_stats),
            "eeg_signal": _finalise_scalar_stats(eeg_stats),
            "ecg_features": {
                column: _finalise_scalar_stats(accumulator)
                for column, accumulator in ecg_accumulators.items()
            },
            "emg_features": {
                column: _finalise_scalar_stats(accumulator)
                for column, accumulator in emg_accumulators.items()
            },
            "meta": {
                "computed_from_split": "train",
                "train_user_ids": self.splits["train"],
            },
        }
        self.norm_stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    def get_known_user_ids(self) -> list[int]:
        """Return the user IDs that should use the train-time lookup table."""

        return list(self.splits["train"])

    def _normalise_scalar(self, values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
        """Z-score normalise a scalar signal."""

        std = max(float(stats["std"]), 1e-6)
        return ((np.asarray(values, dtype=np.float32) - float(stats["mean"])) / std).astype("float32")

    def _normalise_matrix(self, values: np.ndarray, stats: dict[str, dict[str, float]], columns: list[str]) -> np.ndarray:
        """Normalise each feature channel independently."""

        matrix = np.asarray(values, dtype=np.float32).copy()
        for column_index, column_name in enumerate(columns):
            column_stats = stats[column_name]
            std = max(float(column_stats["std"]), 1e-6)
            matrix[:, column_index] = ((matrix[:, column_index] - float(column_stats["mean"])) / std).astype("float32")
        return matrix

    def _window_to_sample(self, user_record: dict, start_idx: int) -> dict[str, torch.Tensor]:
        """Create one normalised Part D forecasting window on demand."""

        input_len = int(self.config["input_len"])
        target_offsets = list(self.config["target_offsets"])
        end_idx = int(start_idx + input_len)
        start_time = pd.Timestamp(user_record["start_time"])
        timestamps = pd.date_range(
            start=start_time + pd.to_timedelta(int(start_idx) * 5, unit="min"),
            periods=input_len,
            freq=self.config["resample_frequency"],
        )

        hr_raw = np.asarray(user_record["signals"]["hr"], dtype=np.float32)[start_idx:end_idx]
        glucose_raw = np.asarray(user_record["signals"]["glucose"], dtype=np.float32)[start_idx:end_idx]
        ecg_raw = np.asarray(user_record["signals"]["ecg_features"], dtype=np.float32)[start_idx:end_idx]
        emg_raw = np.asarray(user_record["signals"]["emg_features"], dtype=np.float32)[start_idx:end_idx]
        cbf_raw = np.asarray(user_record["signals"]["cbf"], dtype=np.float32)[start_idx:end_idx]
        target_indices = [int(end_idx - 1 + offset) for offset in target_offsets]
        glucose_target_raw = np.asarray(user_record["signals"]["glucose"], dtype=np.float32)[target_indices]

        hr_series = pd.Series(hr_raw, index=timestamps)
        glucose_series = pd.Series(glucose_raw, index=timestamps)
        eeg_raw = generate_synthetic_eeg(
            hr_series,
            glucose_series,
            sfreq=int(self.config["eeg_sfreq"]),
        )

        return {
            "hr_sequence": torch.tensor(
                self._normalise_scalar(hr_raw, self.norm_stats["heart_rate_bpm"]),
                dtype=torch.float32,
            ).unsqueeze(-1),
            "glucose_context": torch.tensor(
                self._normalise_scalar(glucose_raw, self.norm_stats["glucose_mg_dl"]),
                dtype=torch.float32,
            ).unsqueeze(-1),
            "ecg_features": torch.tensor(
                self._normalise_matrix(ecg_raw, self.norm_stats["ecg_features"], ECG_COLUMNS),
                dtype=torch.float32,
            ),
            "emg_features": torch.tensor(
                self._normalise_matrix(emg_raw, self.norm_stats["emg_features"], EMG_COLUMNS),
                dtype=torch.float32,
            ),
            "eeg_signal": torch.tensor(
                self._normalise_scalar(eeg_raw, self.norm_stats["eeg_signal"]),
                dtype=torch.float32,
            ),
            "cbf_signal": torch.tensor(
                self._normalise_scalar(cbf_raw, self.norm_stats["cbf_signal"]),
                dtype=torch.float32,
            ).unsqueeze(-1),
            "targets": torch.tensor(
                self._normalise_scalar(glucose_target_raw, self.norm_stats["glucose_mg_dl"]),
                dtype=torch.float32,
            ),
        }

    def build_window(
        self,
        user_id: int,
        start_idx: int,
        *,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Build one arbitrary forecasting window plus raw metadata.

        Part E needs direct access to individual windows rather than only the
        contiguous support/query episodes used during meta-learning. This method
        exposes that window-level view while reusing the exact same
        normalisation and EEG-regeneration path as the training code.
        """

        user_record = self._load_user(int(user_id))
        sample = self._window_to_sample(user_record, int(start_idx))
        if not include_metadata:
            return {"sample": sample}

        input_len = int(self.config["input_len"])
        target_offsets = list(self.config["target_offsets"])
        end_idx = int(start_idx + input_len)
        start_time = pd.Timestamp(user_record["start_time"])
        timestamps = pd.date_range(
            start=start_time + pd.to_timedelta(int(start_idx) * 5, unit="min"),
            periods=input_len,
            freq=self.config["resample_frequency"],
        )
        hr_raw = np.asarray(user_record["signals"]["hr"], dtype=np.float32)[start_idx:end_idx]
        glucose_raw = np.asarray(user_record["signals"]["glucose"], dtype=np.float32)[start_idx:end_idx]
        ecg_raw = np.asarray(user_record["signals"]["ecg_features"], dtype=np.float32)[start_idx:end_idx]
        emg_raw = np.asarray(user_record["signals"]["emg_features"], dtype=np.float32)[start_idx:end_idx]
        cbf_raw = np.asarray(user_record["signals"]["cbf"], dtype=np.float32)[start_idx:end_idx]
        target_indices = [int(end_idx - 1 + offset) for offset in target_offsets]
        glucose_target_raw = np.asarray(user_record["signals"]["glucose"], dtype=np.float32)[target_indices]
        hr_series = pd.Series(hr_raw, index=timestamps)
        glucose_series = pd.Series(glucose_raw, index=timestamps)
        eeg_raw = generate_synthetic_eeg(
            hr_series,
            glucose_series,
            sfreq=int(self.config["eeg_sfreq"]),
        )
        target_timestamp = start_time + pd.to_timedelta(int(target_indices[-1]) * 5, unit="min")

        return {
            "sample": sample,
            "metadata": {
                "user_id": int(user_id),
                "archetype": str(user_record["archetype"]),
                "archetype_id": int(user_record["archetype_id"]),
                "start_idx": int(start_idx),
                "start_time": timestamps[0].to_pydatetime(),
                "end_time": timestamps[-1].to_pydatetime(),
                "target_time": target_timestamp.to_pydatetime(),
                "timestamps": timestamps.to_pydatetime().tolist(),
                "raw": {
                    "hr": hr_raw,
                    "glucose": glucose_raw,
                    "ecg_features": ecg_raw,
                    "emg_features": emg_raw,
                    "eeg": eeg_raw,
                    "cbf": cbf_raw,
                    "glucose_target": glucose_target_raw,
                },
                "params": user_record["params"],
            },
        }

    def iter_split_windows(
        self,
        split: str,
        *,
        limit_users: int | None = None,
        max_windows_per_user: int | None = None,
    ):
        """Yield arbitrary windows for every user in a split.

        The generator is deterministic and starts from the beginning of each
        user's recording, which is suitable for analysis tasks such as Part E
        scenario mining, probing-dataset construction, and attribution studies.
        """

        user_ids = list(self.splits[split])
        if limit_users is not None:
            user_ids = user_ids[: int(limit_users)]

        input_len = int(self.config["input_len"])
        target_offsets = list(self.config["target_offsets"])
        for user_id in user_ids:
            user_record = self._load_user(int(user_id))
            n_steps = int(user_record["metadata"]["n_steps"])
            n_windows = _candidate_window_count(n_steps, input_len, target_offsets)
            if max_windows_per_user is not None:
                n_windows = min(n_windows, int(max_windows_per_user))
            for start_idx in range(n_windows):
                yield self.build_window(int(user_id), start_idx, include_metadata=True)

    def _episode_start(self, user_record: dict, *, seed: int, support_size: int, query_size: int) -> int:
        """Sample a contiguous support/query episode start for one user."""

        n_steps = int(user_record["metadata"]["n_steps"])
        n_windows = _candidate_window_count(n_steps, int(self.config["input_len"]), list(self.config["target_offsets"]))
        n_required = int(support_size + query_size)
        if n_windows < n_required:
            raise ValueError(
                f"User {user_record['user_id']} has only {n_windows} valid windows, "
                f"but Part D requested {n_required}."
            )

        rng = np.random.default_rng(seed)
        max_start = max(n_windows - n_required, 0)
        return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

    def build_task(
        self,
        user_id: int,
        *,
        support_size: int | None = None,
        query_size: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build one deterministic support/query task for the requested user."""

        support_size = int(self.config["support_set_size"] if support_size is None else support_size)
        query_size = int(self.config["query_set_size"] if query_size is None else query_size)
        user_record = self._load_user(int(user_id))
        task_seed = int(self.config["seed"] + user_id * 9973) if seed is None else int(seed)
        episode_start = self._episode_start(
            user_record,
            seed=task_seed,
            support_size=support_size,
            query_size=query_size,
        )

        support_samples = [
            self._window_to_sample(user_record, episode_start + offset)
            for offset in range(support_size)
        ]
        query_samples = [
            self._window_to_sample(user_record, episode_start + support_size + offset)
            for offset in range(query_size)
        ]

        return {
            "user_id": int(user_id),
            "archetype": str(user_record["archetype"]),
            "archetype_id": int(user_record["archetype_id"]),
            "support": _chunk_tensor_dict(support_samples),
            "query": _chunk_tensor_dict(query_samples),
            "metadata": {
                **user_record["metadata"],
                "params": user_record["params"],
                "episode_start": episode_start,
            },
        }

    def sample_task_batch(self, *, split: str, batch_size: int | None = None) -> list[dict[str, Any]]:
        """Sample a meta-batch uniformly across archetypes."""

        batch_size = int(self.config["meta_batch_size"] if batch_size is None else batch_size)
        split_user_ids = set(int(user_id) for user_id in self.splits[split])
        manifest_split = self.manifest[self.manifest["user_id"].isin(split_user_ids)]
        archetype_to_users = {
            archetype: archetype_frame["user_id"].tolist()
            for archetype, archetype_frame in manifest_split.groupby("archetype", sort=True)
        }
        archetypes = sorted(archetype_to_users)
        rng = np.random.default_rng()

        tasks: list[dict[str, Any]] = []
        for index in range(batch_size):
            archetype = archetypes[index % len(archetypes)]
            user_id = int(rng.choice(archetype_to_users[archetype]))
            task_seed = int(self.config["seed"] + user_id * 9973 + rng.integers(0, 1_000_000))
            tasks.append(self.build_task(user_id, seed=task_seed))
        return tasks

    def get_split_tasks(self, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return deterministic tasks for every user in a split."""

        user_ids = list(self.splits[split])
        if limit is not None:
            user_ids = user_ids[: int(limit)]
        return [self.build_task(int(user_id)) for user_id in user_ids]

    def get_user_metadata_frame(self) -> pd.DataFrame:
        """Return the cohort manifest augmented with split information."""

        split_lookup = {}
        for split_name, user_ids in self.splits.items():
            for user_id in user_ids:
                split_lookup[int(user_id)] = split_name
        metadata = self.manifest.copy()
        metadata["split"] = metadata["user_id"].map(split_lookup)
        metadata["archetype_id"] = metadata["archetype"].map(archetype_to_index)
        return metadata


__all__ = ["MetaLearningDataset"]
