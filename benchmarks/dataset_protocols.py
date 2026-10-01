"""Reproducible classification protocols kept separate from builder splits.

The assignments are a project protocol, not published fixed FMD splits.
This module does not download data, decode images, train models, or write files.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import PurePosixPath

from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel


PROTOCOL_VERSION = "sds-split-v1"


@dataclass(frozen=True)
class DatasetProtocol:
    """Dataset views and a serializable record of the experimental assignment."""

    train_fit: StableDataset
    validation: StableDataset
    train_full: StableDataset
    test: StableDataset
    manifest: dict


@dataclass(frozen=True)
class _Record:
    image_id: str
    label: int
    source_split: str
    index: int


def _read_records(dataset: StableDataset, source_split: str, class_names: list[str]) -> list[_Record]:
    label_feature = dataset.features.get("label")
    if not isinstance(label_feature, ClassLabel) or list(label_feature.names) != class_names:
        raise ValueError(f"{source_split}: label ClassLabel names do not match the training collection.")
    if "image_id" not in dataset.features:
        raise ValueError(f"{source_split}: missing image_id feature.")

    records = []
    seen = set()
    # Raw formatting leaves images/masks encoded, and removes any caller transform.
    for index, sample in enumerate(dataset.with_format("raw").with_transform(None)):
        image_id = sample["image_id"]
        if (
            not isinstance(image_id, str)
            or not image_id
            or "\\" in image_id
            or "\x00" in image_id
            or PurePosixPath(image_id).is_absolute()
            or ".." in PurePosixPath(image_id).parts
            or PurePosixPath(image_id).as_posix() != image_id
            or image_id == "."
        ):
            raise ValueError(
                f"{source_split} row {index}: image_id must be a canonical POSIX relative path: {image_id!r}."
            )
        if image_id in seen:
            raise ValueError(f"{source_split}: duplicate image_id {image_id!r}.")
        label = sample["label"]
        if isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label < len(class_names):
            raise ValueError(f"{source_split}: invalid label {label!r} for {image_id!r}.")
        seen.add(image_id)
        records.append(_Record(image_id, int(label), source_split, index))
    return records


def _validate_class_counts(records: list[_Record], class_names: list[str], count: int, source_split: str) -> None:
    counts = Counter(record.label for record in records)
    for label, class_name in enumerate(class_names):
        if counts[label] != count:
            raise ValueError(f"{source_split}: expected {count} samples for {class_name!r}, found {counts[label]}.")


def _rank(records: list[_Record], dataset_name: str, stage: str, seed: int) -> list[_Record]:
    def key(record):
        payload = f"{PROTOCOL_VERSION}|{dataset_name}|{stage}|{seed}|{record.image_id}"
        return hashlib.sha256(payload.encode("utf-8")).digest(), record.image_id

    return sorted(records, key=key)


def _canonical_order(records: list[_Record]) -> list[_Record]:
    return sorted(records, key=lambda record: (record.label, record.image_id))


def build_dataset_protocol(
    dataset_name: str,
    splits: Mapping[str, StableDataset],
    *,
    split_seed: int = 42,
    source_fingerprints: Mapping[str, str] | None = None,
) -> DatasetProtocol:
    """Create benchmark views without changing the original builder collection.

    Args:
        dataset_name: ``"fmd"`` or ``"tf_flowers"``.
        splits: The untransformed builder result from ``split=None``.
        split_seed: Seed used in the versioned SHA-256 ranking, independent of
            training randomness. The default protocol uses 42.
        source_fingerprints: Caller-supplied SHA-256 hex digests of raw assets,
            keyed by the builder's SOURCE.assets keys. This function records
            these claims; it does not read or verify downloaded files. Empty
            fingerprints are allowed for offline tests and are marked missing.

    Returns:
        Fit/validation/refit/test views and a JSON-serializable manifest. Each
        manifest record has one disjoint role and an explicit refit membership.
    """
    if isinstance(dataset_name, str) and dataset_name.lower() == "tf_flowers":
        from .tf_flowers_protocol import build_tf_flowers_protocol

        return build_tf_flowers_protocol(splits, split_seed=split_seed, source_fingerprints=source_fingerprints)
    if not isinstance(dataset_name, str) or dataset_name.lower() != "fmd":
        raise ValueError(f"Unsupported classification protocol: {dataset_name!r}.")
    dataset_name = dataset_name.lower()
    if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
        raise ValueError("split_seed must be an integer.")
    split_seed = int(split_seed)
    expected_splits = {"train"}
    if set(splits) != expected_splits:
        raise ValueError(f"{dataset_name}: expected splits {sorted(expected_splits)}, got {sorted(splits)}.")

    fingerprints = dict(source_fingerprints or {})
    expected_assets = {"archive"}
    if set(fingerprints) - expected_assets:
        raise ValueError(
            f"{dataset_name}: unexpected source fingerprint assets: {sorted(set(fingerprints) - expected_assets)}."
        )
    for asset, fingerprint in fingerprints.items():
        if (
            not isinstance(asset, str)
            or not asset
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("source_fingerprints must map asset names to lowercase SHA-256 hex digests.")

    label_feature = splits["train"].features.get("label")
    if not isinstance(label_feature, ClassLabel):
        raise ValueError("train: label must be a ClassLabel with explicit class names.")
    class_names = list(label_feature.names)
    num_classes = 10
    if (
        len(class_names) != num_classes
        or any(not isinstance(name, str) or not name for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError(f"{dataset_name}: expected {num_classes} distinct, non-empty class names.")

    train_records = _read_records(splits["train"], "train", class_names)
    test_records = []
    _validate_class_counts(train_records, class_names, 100, "train")

    assignments = {"train_fit": [], "validation": [], "train_full": [], "test": []}
    for label in range(len(class_names)):
        class_records = [record for record in train_records if record.label == label]
        ranked_outer = _rank(class_records, dataset_name, "outer", split_seed)
        train_full = ranked_outer[:50]
        validation_count = 10
        assignments["test"].extend(ranked_outer[50:])
        ranked_inner = _rank(train_full, dataset_name, "inner", split_seed)
        assignments["validation"].extend(ranked_inner[:validation_count])
        assignments["train_fit"].extend(ranked_inner[validation_count:])
        assignments["train_full"].extend(train_full)

    assignments = {role: _canonical_order(records) for role, records in assignments.items()}
    role_by_id = {
        record.image_id: role for role in ("train_fit", "validation", "test") for record in assignments[role]
    }
    manifest = {
        "dataset": dataset_name,
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": split_seed,
        "class_names": class_names,
        "source_fingerprints": fingerprints,
        "source_fingerprints_status": "provided" if fingerprints else "not_verified",
        "source_fingerprints_complete": set(fingerprints) == expected_assets,
        "counts": {role: len(records) for role, records in assignments.items()},
        "per_class_counts": {
            role: {
                class_name: sum(record.label == label for record in records)
                for label, class_name in enumerate(class_names)
            }
            for role, records in {**assignments, "source_train": train_records, "source_test": test_records}.items()
        },
        "records": [
            {
                "image_id": record.image_id,
                "class_name": class_names[record.label],
                "label": record.label,
                "source_split": record.source_split,
                "role": role_by_id[record.image_id],
                "in_train_full": role_by_id[record.image_id] != "test",
            }
            for record in _canonical_order(train_records + test_records)
        ],
    }
    views = {
        role: splits["train"].select([record.index for record in records]) for role, records in assignments.items()
    }
    return DatasetProtocol(**views, manifest=manifest)
