"""Offline tests for native dataset views and classification split manifests."""

import json
import os
from collections import Counter
from types import SimpleNamespace

import pyarrow as pa
import pytest

from benchmarks.dataset_protocols import PROTOCOL_VERSION, build_dataset_protocol
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Value


FMD_NAMES = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
# Audited official list counts, not the paper's nominal 80/20 per-class ratio.
INDOOR_TRAIN_COUNTS = {
    "airport_inside": 80,
    "artstudio": 80,
    "auditorium": 82,
    "bakery": 81,
    "bar": 82,
    "bathroom": 82,
    "bedroom": 79,
    "bookstore": 80,
    "bowling": 80,
    "buffet": 80,
    "casino": 81,
    "children_room": 82,
    "church_inside": 81,
    "classroom": 82,
    "cloister": 80,
    "closet": 82,
    "clothingstore": 82,
    "computerroom": 82,
    "concert_hall": 80,
    "corridor": 79,
    "deli": 81,
    "dentaloffice": 79,
    "dining_room": 82,
    "elevator": 79,
    "fastfood_restaurant": 83,
    "florist": 81,
    "gameroom": 80,
    "garage": 82,
    "greenhouse": 80,
    "grocerystore": 79,
    "gym": 82,
    "hairsalon": 79,
    "hospitalroom": 80,
    "inside_bus": 77,
    "inside_subway": 79,
    "jewelleryshop": 78,
    "kindergarden": 80,
    "kitchen": 79,
    "laboratorywet": 78,
    "laundromat": 78,
    "library": 80,
    "livingroom": 80,
    "lobby": 80,
    "locker_room": 79,
    "mall": 80,
    "meeting_room": 78,
    "movietheater": 80,
    "museum": 77,
    "nursery": 80,
    "office": 79,
    "operating_room": 81,
    "pantry": 80,
    "poolinside": 80,
    "prisoncell": 80,
    "restaurant": 80,
    "restaurant_kitchen": 77,
    "shoeshop": 81,
    "stairscase": 80,
    "studiomusic": 81,
    "subway": 79,
    "toystore": 78,
    "trainstation": 80,
    "tv_studio": 82,
    "videostore": 78,
    "waitingroom": 79,
    "warehouse": 79,
    "winecellar": 79,
}


def _dataset(class_names, records):
    features = Features({"image_id": Value("string"), "label": ClassLabel(names=class_names), "image": Image()})
    return StableDataset(
        features=features,
        info=DatasetInfo(features=features),
        table=pa.Table.from_pylist(records, schema=features.to_arrow_schema()),
    )


def _fmd():
    # Deliberately undecodable images: assigning metadata must not decode pixels.
    return {
        "train": _dataset(
            FMD_NAMES,
            [
                {"image_id": f"{name}/image {index:03}.jpg", "label": label, "image": b"not an image"}
                for label, name in enumerate(FMD_NAMES)
                for index in range(100)
            ],
        )
    }


def _indoor67():
    names = list(INDOOR_TRAIN_COUNTS)
    records = {"train": [], "test": []}
    for label, (name, train_count) in enumerate(INDOOR_TRAIN_COUNTS.items()):
        for index in range(100):
            role = "train" if index < train_count else "test"
            records[role].append(
                {"image_id": f"{name}/image {index:03}.jpg", "label": label, "image": b"not an image"}
            )
    return {split: _dataset(names, rows) for split, rows in records.items()}


def _ids(dataset):
    return [row["image_id"] for row in dataset.with_format("raw").with_transform(None)]


def _change_row(dataset, index, **values):
    records = dataset.table.to_pylist()
    records[index].update(values)
    return _dataset(dataset.features["label"].names, records)


def _assert_partition(protocol, expected_counts):
    role_ids = {role: set(_ids(getattr(protocol, role))) for role in expected_counts}
    assert {role: len(ids) for role, ids in role_ids.items()} == expected_counts
    assert not role_ids["train_fit"] & role_ids["validation"]
    assert not role_ids["train_full"] & role_ids["test"]
    assert role_ids["train_full"] == role_ids["train_fit"] | role_ids["validation"]
    records = protocol.manifest["records"]
    assert len({row["image_id"] for row in records}) == len(records)
    assert {row["image_id"] for row in records} == role_ids["train_full"] | role_ids["test"]
    for record in records:
        assert record["image_id"] in role_ids[record["role"]]
        assert record["in_train_full"] == (record["image_id"] in role_ids["train_full"])
        assert record["class_name"] == protocol.manifest["class_names"][record["label"]]
    assert protocol.manifest["counts"] == expected_counts
    assert protocol.manifest["protocol_version"] == PROTOCOL_VERSION == "sds-split-v1"
    assert json.loads(json.dumps(protocol.manifest)) == protocol.manifest


def test_fmd_counts_roles_and_original_collection_preserved():
    splits = _fmd()
    protocol = build_dataset_protocol("fmd", splits)
    _assert_partition(protocol, {"train_fit": 400, "validation": 100, "train_full": 500, "test": 500})
    assert len(splits["train"]) == 1000
    for role, count in {"train_fit": 40, "validation": 10, "train_full": 50, "test": 50}.items():
        dataset = getattr(protocol, role)
        assert dataset._backend is splits["train"]._backend
        assert Counter(row["label"] for row in dataset.with_format("raw")) == dict.fromkeys(range(10), count)
        assert protocol.manifest["per_class_counts"][role] == dict.fromkeys(FMD_NAMES, count)
    assert protocol.manifest["source_fingerprints"] == {}
    assert protocol.manifest["source_fingerprints_status"] == "not_verified"
    assert not protocol.manifest["source_fingerprints_complete"]


def test_indoor67_preserves_uneven_official_membership_and_holds_out_16_per_class():
    splits = _indoor67()
    original_train = set(_ids(splits["train"]))
    original_test = set(_ids(splits["test"]))
    protocol = build_dataset_protocol("indoor67", splits)
    _assert_partition(protocol, {"train_fit": 4288, "validation": 1072, "train_full": 5360, "test": 1340})
    assert set(_ids(protocol.train_full)) == original_train
    assert set(_ids(protocol.test)) == original_test
    counts = protocol.manifest["per_class_counts"]
    assert counts["source_train"] == counts["train_full"] == INDOOR_TRAIN_COUNTS
    assert (
        counts["source_test"] == counts["test"] == {name: 100 - count for name, count in INDOOR_TRAIN_COUNTS.items()}
    )
    assert counts["validation"] == dict.fromkeys(INDOOR_TRAIN_COUNTS, 16)
    assert counts["train_fit"] == {name: count - 16 for name, count in INDOOR_TRAIN_COUNTS.items()}
    assert counts["train_fit"]["fastfood_restaurant"] == 67
    assert counts["train_fit"]["inside_bus"] == 61
    assert len(splits["train"]) == 5360
    assert len(splits["test"]) == 1340


def test_indoor67_input_order_independent_and_test_membership_unchanged_across_seeds():
    splits = _indoor67()
    first = build_dataset_protocol("indoor67", splits)
    reordered = {name: dataset.select(list(reversed(range(len(dataset))))) for name, dataset in splits.items()}
    reordered_protocol = build_dataset_protocol("indoor67", reordered)
    assert first.manifest == reordered_protocol.manifest
    for role in first.manifest["counts"]:
        assert _ids(getattr(first, role)) == _ids(getattr(reordered_protocol, role))
    second_seed = build_dataset_protocol("indoor67", splits, split_seed=43)
    assert set(_ids(first.validation)) != set(_ids(second_seed.validation))
    assert _ids(first.train_full) == _ids(second_seed.train_full)
    assert _ids(first.test) == _ids(second_seed.test)


def test_indoor67_inner_stage_golden_assignment():
    protocol = build_dataset_protocol("indoor67", _indoor67())
    assert [image_id for image_id in _ids(protocol.validation) if image_id.startswith("auditorium/")] == [
        "auditorium/image 005.jpg",
        "auditorium/image 007.jpg",
        "auditorium/image 015.jpg",
        "auditorium/image 017.jpg",
        "auditorium/image 020.jpg",
        "auditorium/image 024.jpg",
        "auditorium/image 035.jpg",
        "auditorium/image 049.jpg",
        "auditorium/image 052.jpg",
        "auditorium/image 053.jpg",
        "auditorium/image 054.jpg",
        "auditorium/image 058.jpg",
        "auditorium/image 061.jpg",
        "auditorium/image 066.jpg",
        "auditorium/image 073.jpg",
        "auditorium/image 076.jpg",
    ]


def test_indoor67_rejects_cross_split_overlap():
    splits = _indoor67()
    splits["test"] = _change_row(splits["test"], 0, image_id="airport_inside/image 000.jpg")
    with pytest.raises(ValueError, match="official train/test image_ids overlap"):
        build_dataset_protocol("indoor67", splits)


def test_indoor67_rejects_different_class_mappings():
    splits = _indoor67()
    splits["test"] = _dataset(list(reversed(INDOOR_TRAIN_COUNTS)), splits["test"].table.to_pylist())
    with pytest.raises(ValueError, match="ClassLabel names do not match"):
        build_dataset_protocol("indoor67", splits)


def test_indoor67_rejects_incomplete_official_totals():
    splits = _indoor67()
    splits["test"] = splits["test"].select(range(1339))
    with pytest.raises(ValueError, match="official totals train=5360/test=1340"):
        build_dataset_protocol("indoor67", splits)


def test_indoor67_rejects_wrong_combined_class_counts():
    splits = _indoor67()
    splits["test"] = _change_row(splits["test"], 0, label=1)
    with pytest.raises(ValueError, match="train \\+ test: expected 100 samples"):
        build_dataset_protocol("indoor67", splits)


def test_indoor67_rejects_missing_test_collection():
    with pytest.raises(ValueError, match="expected splits"):
        build_dataset_protocol("indoor67", {"train": _indoor67()["train"]})


def test_indoor67_rejects_missing_test_class_despite_matching_totals():
    splits = _indoor67()
    train_rows = splits["train"].table.to_pylist()
    test_rows = splits["test"].table.to_pylist()
    moved_to_train = [row for row in test_rows if row["label"] == 0]
    moved_to_test = [row for row in train_rows if row["label"] == 1][: len(moved_to_train)]
    moved_ids = {row["image_id"] for row in moved_to_test + moved_to_train}
    names = list(INDOOR_TRAIN_COUNTS)
    altered = {
        "train": _dataset(names, [row for row in train_rows if row["image_id"] not in moved_ids] + moved_to_train),
        "test": _dataset(names, [row for row in test_rows if row["image_id"] not in moved_ids] + moved_to_test),
    }
    with pytest.raises(ValueError, match="official test is missing class 'airport_inside'"):
        build_dataset_protocol("indoor67", altered)


def test_fmd_protocol_golden_assignment():
    protocol = build_dataset_protocol("fmd", _fmd())
    # Frozen expected IDs cover both outer and independent inner stage ranking.
    assert [image_id for image_id in _ids(protocol.validation) if image_id.startswith("fabric/")] == [
        "fabric/image 002.jpg",
        "fabric/image 015.jpg",
        "fabric/image 033.jpg",
        "fabric/image 035.jpg",
        "fabric/image 039.jpg",
        "fabric/image 041.jpg",
        "fabric/image 059.jpg",
        "fabric/image 071.jpg",
        "fabric/image 076.jpg",
        "fabric/image 089.jpg",
    ]
    assert "fabric/image 000.jpg" in _ids(protocol.train_fit)
    assert "fabric/image 001.jpg" in _ids(protocol.test)


def test_fmd_reproducible_independent_of_input_order_and_seed_changes_membership():
    splits = _fmd()
    first = build_dataset_protocol("fmd", splits, split_seed=42)
    again = build_dataset_protocol("fmd", splits, split_seed=42)
    reordered = {"train": splits["train"].select(list(reversed(range(1000))))}
    shuffled = build_dataset_protocol("fmd", reordered, split_seed=42)
    assert first.manifest == again.manifest == shuffled.manifest
    for role in first.manifest["counts"]:
        assert _ids(getattr(first, role)) == _ids(getattr(shuffled, role))
    second_seed = build_dataset_protocol("fmd", splits, split_seed=43)
    assert set(_ids(first.test)) != set(_ids(second_seed.test))
    assert set(_ids(first.validation)) != set(_ids(second_seed.validation))


def test_protocol_ignores_read_time_transforms_for_assignment():
    def forbidden_transform(sample):
        raise AssertionError("protocol assignment must not invoke image transforms")

    splits = _fmd()
    with_transform = {"train": splits["train"].with_transform(forbidden_transform)}
    assert build_dataset_protocol("fmd", with_transform).manifest == build_dataset_protocol("fmd", splits).manifest


def test_protocol_records_caller_fingerprints_without_claiming_verification():
    fingerprints = {"archive": "a" * 64}
    protocol = build_dataset_protocol("fmd", _fmd(), source_fingerprints=fingerprints)
    assert protocol.manifest["source_fingerprints"] == fingerprints
    assert protocol.manifest["source_fingerprints_status"] == "provided"
    assert protocol.manifest["source_fingerprints_complete"]
    fingerprints["archive"] = "b" * 64
    assert protocol.manifest["source_fingerprints"]["archive"] == "a" * 64


@pytest.mark.parametrize("seed", [None, True, 1.5, "42"])
def test_invalid_split_seed(seed):
    with pytest.raises(ValueError, match="split_seed"):
        build_dataset_protocol("fmd", _fmd(), split_seed=seed)


@pytest.mark.parametrize("name", ["cifar10", "", None])
def test_unsupported_protocol(name):
    with pytest.raises(ValueError, match="Unsupported classification protocol"):
        build_dataset_protocol(name, _fmd())


@pytest.mark.parametrize("split_names", [[], ["test"], ["train", "test"], ["train", "validation"]])
def test_fmd_rejects_missing_or_extra_builder_splits(split_names):
    dataset = _fmd()["train"]
    with pytest.raises(ValueError, match="expected splits"):
        build_dataset_protocol("fmd", dict.fromkeys(split_names, dataset))


@pytest.mark.parametrize(
    "image_id", ["", "../photo.jpg", "/photo.jpg", "fabric\\photo.jpg", "fabric//photo.jpg", "./photo.jpg"]
)
def test_protocol_rejects_noncanonical_image_ids(image_id):
    splits = _fmd()
    splits["train"] = _change_row(splits["train"], 0, image_id=image_id)
    with pytest.raises(ValueError, match="canonical POSIX relative path"):
        build_dataset_protocol("fmd", splits)


def test_protocol_rejects_duplicate_ids():
    splits = _fmd()
    splits["train"] = _change_row(splits["train"], 1, image_id="fabric/image 000.jpg")
    with pytest.raises(ValueError, match="duplicate image_id.*fabric/image 000"):
        build_dataset_protocol("fmd", splits)


@pytest.mark.parametrize("label", [-1, 10, None])
def test_protocol_rejects_invalid_labels(label):
    splits = _fmd()
    splits["train"] = _change_row(splits["train"], 0, label=label)
    with pytest.raises(ValueError, match="invalid label"):
        build_dataset_protocol("fmd", splits)


def test_fmd_rejects_incomplete_collection():
    splits = _fmd()
    splits["train"] = splits["train"].select(range(999))
    with pytest.raises(ValueError, match="expected 100 samples for 'wood', found 99"):
        build_dataset_protocol("fmd", splits)


@pytest.mark.parametrize("names", [FMD_NAMES[:-1], FMD_NAMES[:-1] + ["fabric"]])
def test_protocol_rejects_incomplete_or_duplicated_class_names(names):
    dataset = _dataset(names, _fmd()["train"].table.to_pylist())
    with pytest.raises(ValueError, match="10 distinct, non-empty class names"):
        build_dataset_protocol("fmd", {"train": dataset})


def test_protocol_requires_stable_ids():
    dataset = _fmd()["train"].remove_columns("image_id")
    with pytest.raises(ValueError, match="missing image_id feature"):
        build_dataset_protocol("fmd", {"train": dataset})


@pytest.mark.parametrize("fingerprints", [{"archive": "bad"}, {"archive": "A" * 64}, {"images": "a" * 64}])
def test_protocol_rejects_invalid_source_fingerprints(fingerprints):
    with pytest.raises(ValueError, match="fingerprint"):
        build_dataset_protocol("fmd", _fmd(), source_fingerprints=fingerprints)


def _patch_builder(monkeypatch, splits):
    from benchmarks import dataset as benchmark_dataset

    calls = []

    class LocalBuilder:
        def __new__(cls, split=None, **kwargs):
            calls.append((split, kwargs))
            if split is None:
                return splits
            if split not in splits:
                raise ValueError(f"Split {split!r} not found")
            return splits[split]

    monkeypatch.setattr(benchmark_dataset, "_get_dataset_class", lambda config: LocalBuilder)
    return calls


def test_new_benchmarks_are_explicit_without_changing_dataset_all():
    from benchmarks.dataset import get_image_dataset_names

    assert {"indoor67", "fmd"} <= set(get_image_dataset_names())
    assert not {"indoor67", "fmd"} & set(get_image_dataset_names(include_results_only=True))


@pytest.mark.parametrize("name,make_splits", [("fmd", _fmd), ("indoor67", _indoor67)])
def test_benchmark_uses_protocol_without_legacy_fallback(monkeypatch, tmp_path, name, make_splits):
    from benchmarks import dataset as benchmark_dataset

    splits = make_splits()
    calls = _patch_builder(monkeypatch, splits)

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("new datasets must not use legacy validation/test selection or the 10% holdout")

    monkeypatch.setattr(benchmark_dataset, "_load_validation_split", forbidden_fallback)
    monkeypatch.setattr(StableDataset, "train_test_split", forbidden_fallback)
    expected = build_dataset_protocol(name, splits, split_seed=43)
    data, config = benchmark_dataset.create_dataset(
        name,
        train_transform=None,
        val_transform=None,
        collate_fn=None,
        training_cfg=SimpleNamespace(batch_size=4, num_workers=0),
        data_dir=str(tmp_path),
        split_seed=43,
    )
    assert config.num_classes == (67 if name == "indoor67" else 10)
    assert config.channels == 3
    assert _ids(data.train.dataset) == _ids(expected.train_fit)
    assert _ids(data.val.dataset) == _ids(expected.validation)
    assert not set(_ids(data.val.dataset)) & set(_ids(expected.test))
    assert data.dataset_protocol_manifest == expected.manifest
    assert calls == [
        (None, {"download_dir": str(tmp_path / "downloads"), "processed_cache_dir": str(tmp_path / "processed")})
    ]


@pytest.mark.parametrize("has_test", [True, False])
def test_existing_dataset_keeps_legacy_split_behavior(monkeypatch, has_test):
    from benchmarks.dataset import create_dataset

    original = _fmd()["train"].remove_columns("image")
    splits = {"train": original.select(range(20))}
    if has_test:
        splits["test"] = original.select(range(20, 30))
    calls = _patch_builder(monkeypatch, splits)
    data, _ = create_dataset("cifar10", None, None, None, SimpleNamespace(batch_size=4, num_workers=0), split_seed=43)
    if has_test:
        assert _ids(data.train.dataset) == _ids(splits["train"])
        assert _ids(data.val.dataset) == _ids(splits["test"])
    else:
        expected = splits["train"].train_test_split(test_size=0.1, seed=42)
        assert _ids(data.train.dataset) == _ids(expected["train"])
        assert _ids(data.val.dataset) == _ids(expected["test"])
        assert len(data.train.dataset) == 18
        assert len(data.val.dataset) == 2
    assert [split for split, _ in calls] == ["train", "validation", "valid", "test"]
    assert not hasattr(data, "dataset_protocol_manifest")


def test_fmd_benchmark_collation_omits_mixed_mode_pil_masks(monkeypatch):
    from PIL import Image as PILImage

    from benchmarks.dataset import create_dataset, get_config
    from benchmarks.models import collate_single, val_transform

    records = _fmd()["train"].table.to_pylist()
    image_bytes = Image().encode(PILImage.new("RGB", (3, 2), color=(10, 40, 90)))
    masks = [Image().encode(PILImage.new(mode, (3, 2), color=0)) for mode in ("L", "RGB")]
    for index, record in enumerate(records):
        record["image"] = image_bytes
        record["mask"] = masks[index % 2]
    features = Features(
        {"image_id": Value("string"), "label": ClassLabel(names=FMD_NAMES), "image": Image(), "mask": Image()}
    )
    original = StableDataset(
        features=features,
        info=DatasetInfo(features=features),
        table=pa.Table.from_pylist(records, schema=features.to_arrow_schema()),
    )
    assert original[0]["mask"].mode == "L"
    assert original[1]["mask"].mode == "RGB"
    _patch_builder(monkeypatch, {"train": original})
    transform = val_transform(get_config("fmd"))
    data, _ = create_dataset("fmd", transform, transform, collate_single, SimpleNamespace(batch_size=4, num_workers=0))
    for loader in (data.train, data.val):
        batch = next(iter(loader))
        assert set(batch) == {"image", "label"}
        assert batch["image"].shape == (4, 3, 224, 224)
        assert batch["label"].shape == (4,)
    assert "mask" in original[0]
    assert len(original) == 1000


def test_runner_saves_complete_manifest_without_overwriting_different_protocol(tmp_path):
    from benchmarks.run import _save_dataset_protocol_manifest

    manifest = build_dataset_protocol("fmd", _fmd(), source_fingerprints={"archive": "a" * 64}).manifest
    data = SimpleNamespace(dataset_protocol_manifest=manifest)
    _save_dataset_protocol_manifest(data, str(tmp_path))
    path = tmp_path / "dataset_protocol.json"
    saved = path.read_bytes()
    assert json.loads(saved) == manifest
    _save_dataset_protocol_manifest(data, str(tmp_path))
    assert path.read_bytes() == saved
    changed = build_dataset_protocol("fmd", _fmd(), split_seed=43, source_fingerprints={"archive": "a" * 64}).manifest
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _save_dataset_protocol_manifest(SimpleNamespace(dataset_protocol_manifest=changed), str(tmp_path))
    assert path.read_bytes() == saved


def test_runner_requires_complete_source_fingerprints_before_training(tmp_path):
    from benchmarks.run import _save_dataset_protocol_manifest

    data = SimpleNamespace(dataset_protocol_manifest=build_dataset_protocol("fmd", _fmd()).manifest)
    with pytest.raises(ValueError, match="require source_fingerprints"):
        _save_dataset_protocol_manifest(data, str(tmp_path))
    assert not (tmp_path / "dataset_protocol.json").exists()
    _save_dataset_protocol_manifest(SimpleNamespace(), str(tmp_path))
    assert not (tmp_path / "dataset_protocol.json").exists()


def test_checkpoint_scope_binds_split_seed_and_source_fingerprints(tmp_path):
    from benchmarks.run import _get_run_checkpoint_dir

    name = "supervised_vit_tiny_patch16_224_fmd_seed42"
    first = build_dataset_protocol("fmd", _fmd(), source_fingerprints={"archive": "a" * 64}).manifest
    changed_seed = build_dataset_protocol(
        "fmd", _fmd(), split_seed=43, source_fingerprints={"archive": "a" * 64}
    ).manifest
    changed_source = build_dataset_protocol("fmd", _fmd(), source_fingerprints={"archive": "b" * 64}).manifest
    paths = [
        _get_run_checkpoint_dir(SimpleNamespace(dataset_protocol_manifest=manifest), str(tmp_path), name)
        for manifest in (first, changed_seed, changed_source)
    ]
    assert len(set(paths)) == 3
    from pathlib import Path

    first_path = Path(paths[0])
    assert first_path.parent == tmp_path / name
    assert first_path.name.startswith("protocol-")
    assert len(first_path.name.removeprefix("protocol-")) == 64
    (first_path / "last.ckpt").write_bytes(b"old protocol checkpoint")
    assert not (Path(paths[1]) / "last.ckpt").exists()
    assert not (Path(paths[2]) / "last.ckpt").exists()
    reordered_keys = dict(reversed(list(first.items())))
    assert (
        _get_run_checkpoint_dir(SimpleNamespace(dataset_protocol_manifest=reordered_keys), str(tmp_path), name)
        == paths[0]
    )
    assert json.loads((first_path / "dataset_protocol.json").read_text(encoding="utf-8")) == first


def test_checkpoint_scope_binds_protocol_version_and_rejects_mismatching_record(tmp_path):
    from pathlib import Path

    from benchmarks.run import _get_run_checkpoint_dir

    name = "supervised_vit_tiny_patch16_224_fmd_seed42"
    manifest = build_dataset_protocol("fmd", _fmd(), source_fingerprints={"archive": "a" * 64}).manifest
    data = SimpleNamespace(dataset_protocol_manifest=manifest)
    first_path = Path(_get_run_checkpoint_dir(data, str(tmp_path), name))
    changed = {**manifest, "protocol_version": "sds-split-v2"}
    second_path = Path(
        _get_run_checkpoint_dir(SimpleNamespace(dataset_protocol_manifest=changed), str(tmp_path), name)
    )
    assert first_path != second_path
    (first_path / "dataset_protocol.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _get_run_checkpoint_dir(data, str(tmp_path), name)


def test_existing_dataset_checkpoint_path_and_side_effects_unchanged(tmp_path):
    from benchmarks.run import _get_run_checkpoint_dir

    name = "supervised_vit_tiny_patch16_224_cifar10_seed42"
    assert _get_run_checkpoint_dir(SimpleNamespace(), str(tmp_path), name) == os.path.join(str(tmp_path), name)
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_cannot_resume_without_its_protocol_record(tmp_path):
    from pathlib import Path

    from benchmarks.run import _get_run_checkpoint_dir

    manifest = build_dataset_protocol("fmd", _fmd(), source_fingerprints={"archive": "a" * 64}).manifest
    data = SimpleNamespace(dataset_protocol_manifest=manifest)
    name = "supervised_vit_tiny_patch16_224_fmd_seed42"
    path = Path(_get_run_checkpoint_dir(data, str(tmp_path), name))
    (path / "last.ckpt").write_bytes(b"checkpoint without provenance")
    (path / "dataset_protocol.json").unlink()
    with pytest.raises(FileNotFoundError, match="without its protocol manifest"):
        _get_run_checkpoint_dir(data, str(tmp_path), name)
    assert not (path / "dataset_protocol.json").exists()
