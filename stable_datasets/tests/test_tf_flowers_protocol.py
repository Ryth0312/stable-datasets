"""Offline checks for archive-bound, global TFFlowers group assignments."""

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pyarrow as pa
import pytest

from benchmarks import tf_flowers_protocol as protocol
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Value


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _evidence(image_id, *, file=None, pixels=None, mode="RGB", size=(2, 3), photo=None, profile=None):
    attribution = []
    if photo:
        attribution.append(
            {
                "url_kind": "photo",
                "source_url": f"https://www.flickr.com/photos/test/{photo}/",
                "flickr_photo_id": photo,
            }
        )
    if profile:
        attribution.append(
            {
                "url_kind": "profile",
                "source_url": f"https://www.flickr.com/photos/{profile}/",
                "flickr_photo_id": "123",  # Must not become a verified photo edge.
                "author": "Same photographer",
            }
        )
    return {
        "image_id": image_id,
        "file_sha256": _sha(file or image_id),
        "decoded_source_pixels_sha256": _sha(pixels or image_id),
        "source_mode": mode,
        "size": list(size),
        "attribution": attribution,
        "filename_flickr_photo_id": "123",  # Also must not make an edge.
    }


def test_global_connectivity_is_transitive_across_labels_and_evidence_kinds():
    rows = [
        _evidence("roses/a.jpg", file="ab"),
        _evidence("tulips/b.jpg", file="ab", pixels="bc"),
        _evidence("daisy/c.jpg", pixels="bc", photo="17"),
        _evidence("sunflowers/d.jpg", photo="17"),
        _evidence("dandelion/e.jpg"),
    ]
    groups = protocol.connected_groups(rows)
    together = next(group for group in groups if len(group["image_ids"]) > 1)
    assert together["image_ids"] == sorted(row["image_id"] for row in rows[:4])
    assert together["class_counts"] == {"daisy": 1, "roses": 1, "sunflowers": 1, "tulips": 1}
    assert {edge["kind"] for edge in together["evidence"]} == {"file_sha256", "native_pixels", "verified_photo_id"}
    assert protocol.connected_groups(reversed(rows)) == groups
    assert together["group_id"] == _sha(json.dumps(together["image_ids"], separators=(",", ":")))


def test_native_pixel_equality_requires_mode_and_shape():
    rows = [
        _evidence("roses/a.jpg", pixels="same", mode="RGB", size=(2, 3)),
        _evidence("roses/b.jpg", pixels="same", mode="L", size=(2, 3)),
        _evidence("roses/c.jpg", pixels="same", mode="RGB", size=(3, 2)),
    ]
    assert len(protocol.connected_groups(rows)) == 3


def test_author_profile_filename_prefix_and_rgb_conversion_do_not_group():
    rows = [_evidence(f"{name}/123_variant.jpg", profile="same") for name in ("roses", "tulips")]
    for row in rows:
        row["decoded_rgb_pixels_sha256"] = _sha("equal only after RGB conversion")
    assert len(protocol.connected_groups(rows)) == 2


@pytest.mark.parametrize("change", ["different_id", "profile_as_photo", "different_host"])
def test_rejects_unverified_photo_identifiers(change):
    row = _evidence("roses/a.jpg", photo="17")
    attribution = row["attribution"][0]
    if change == "different_id":
        attribution["flickr_photo_id"] = "18"
    elif change == "profile_as_photo":
        attribution["source_url"] = "https://www.flickr.com/photos/test/"
    else:
        attribution["source_url"] = "https://example.org/photos/test/17/"
    with pytest.raises(ValueError, match="Flickr"):
        protocol.connected_groups([row])


def test_duplicate_full_ids_and_malformed_hashes_are_rejected():
    row = _evidence("roses/a.jpg")
    with pytest.raises(ValueError, match="Duplicate evidence"):
        protocol.connected_groups([row, row])
    row["file_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="SHA-256"):
        protocol.connected_groups([row])


def test_bundled_evidence_contains_three_groups_six_members_and_original_conflict():
    metadata = protocol.BUNDLED_GROUP_METADATA
    assert metadata["archive_sha256"] == protocol.ARCHIVE_SHA256
    assert metadata["image_count"] == 3670
    assert len(metadata["groups"]) == 3
    assert len(metadata["members"]) == 6
    assert protocol._components(metadata["members"]) == metadata["groups"]
    conflicts = [group for group in metadata["groups"] if len(group["class_counts"]) > 1]
    assert len(conflicts) == 1
    assert conflicts[0]["class_counts"] == {"roses": 1, "tulips": 1}
    assert conflicts[0]["image_ids"] == [
        "roses/15922772266_1167a06620.jpg",
        "tulips/15922772266_1167a06620.jpg",
    ]


@pytest.fixture
def synthetic_collection(monkeypatch):
    """Full-size offline observations; explicitly not the official archive."""
    rows = []
    for label, name in enumerate(protocol.CLASS_NAMES):
        for index in range(protocol.CLASS_COUNTS[name]):
            rows.append({"image_id": f"{name}/{index:05}.jpg", "label": label, "image": b"not an image"})
    # Include one cross-label connection and a within-class connection. Both
    # use the general global algorithm; neither is assigned a special role.
    observations = [_evidence(row["image_id"]) for row in rows]
    observations[0]["file_sha256"] = observations[-1]["file_sha256"]
    observations[1]["decoded_source_pixels_sha256"] = observations[2]["decoded_source_pixels_sha256"]
    metadata = protocol.reconstruct_group_metadata(observations, archive_sha256=protocol.ARCHIVE_SHA256)
    monkeypatch.setattr(protocol, "BUNDLED_GROUP_METADATA", metadata)
    features = Features(
        {"image_id": Value("string"), "label": ClassLabel(names=list(protocol.CLASS_NAMES)), "image": Image()}
    )
    ds = StableDataset(
        features=features,
        info=DatasetInfo(features=features),
        table=pa.Table.from_pylist(rows, schema=features.to_arrow_schema()),
    )
    return ds, observations


def test_full_metadata_verification_catches_changes_outside_six_members(synthetic_collection):
    _, observations = synthetic_collection
    assert protocol.verify_group_metadata(observations, archive_sha256=protocol.ARCHIVE_SHA256)
    changed = copy.deepcopy(observations)
    changed[50]["file_sha256"] = changed[51]["file_sha256"]
    with pytest.raises(ValueError, match="differs"):
        protocol.verify_group_metadata(changed, archive_sha256=protocol.ARCHIVE_SHA256)
    with pytest.raises(ValueError, match="3670"):
        protocol.reconstruct_group_metadata(observations[:-1], archive_sha256=protocol.ARCHIVE_SHA256)
    with pytest.raises(ValueError, match="bound"):
        protocol.reconstruct_group_metadata(observations, archive_sha256="0" * 64)


def test_all_rows_preserved_all_roles_disjoint_groups_never_split(synthetic_collection):
    ds, _ = synthetic_collection
    result = protocol.build_tf_flowers_protocol(
        {"train": ds}, source_fingerprints={"archive": protocol.ARCHIVE_SHA256}
    )
    manifest = result.manifest
    assert manifest["protocol_version"] == "sds-tfflowers-group-v1"
    assert manifest["source_fingerprints_status"] == "provided"
    assert manifest["source_fingerprints_complete"]
    assert len(ds) == len(manifest["records"]) == 3670
    assert sum(manifest["counts"][role] for role in ("train_fit", "validation", "test")) == 3670
    memberships = {
        role: {row["image_id"] for row in getattr(result, role).with_format("raw")}
        for role in ("train_fit", "validation", "test", "train_full")
    }
    assert not memberships["train_fit"] & memberships["validation"]
    assert not memberships["train_full"] & memberships["test"]
    assert memberships["train_full"] == memberships["train_fit"] | memberships["validation"]
    assert len(set.union(*memberships.values())) == 3670
    roles = {row["image_id"]: row["role"] for row in manifest["records"]}
    assert manifest["source_group_count"] == len(manifest["groups"])
    for role in ("train_fit", "validation", "test", "train_full"):
        expected_count = sum(
            group["in_train_full"] if role == "train_full" else group["role"] == role for group in manifest["groups"]
        )
        assert manifest["group_counts"][role] == expected_count
    assert manifest["duplicate_group_count"] == len(manifest["duplicate_groups"]) == 2
    assert manifest["cross_class_conflict_group_count"] == len(manifest["cross_class_conflict_groups"]) == 1
    assert manifest["duplicate_groups"] == [g for g in manifest["groups"] if len(g["image_ids"]) > 1]
    assert manifest["cross_class_conflict_groups"] == [g for g in manifest["groups"] if len(g["class_counts"]) > 1]
    for group in manifest["groups"]:
        assert {roles[image_id] for image_id in group["image_ids"]} == {group["role"]}
        assert group["group_id"] == protocol.group_id(group["image_ids"])
        assert group["in_train_full"] == (group["role"] != "test")
        assert group["label_set"] == sorted(protocol.CLASS_NAMES.index(name) for name in group["class_counts"])
    for row in manifest["records"]:
        assert row["image_id"] in memberships[row["role"]]
        assert row["class_name"] == protocol.CLASS_NAMES[row["label"]]
    assert json.loads(json.dumps(manifest)) == manifest
    for role, ratio in (("train_fit", 0.6), ("validation", 0.2), ("test", 0.2)):
        assert abs(manifest["counts"][role] - 3670 * ratio) <= 2
        for name, count in protocol.CLASS_COUNTS.items():
            assert abs(manifest["per_class_counts"][role][name] - count * ratio) <= 2


def test_assignment_order_independence_and_seed_effect(synthetic_collection):
    ds, _ = synthetic_collection
    first = protocol.build_tf_flowers_protocol({"train": ds})
    reordered = protocol.build_tf_flowers_protocol({"train": ds.select(list(reversed(range(len(ds)))))})
    assert first.manifest == reordered.manifest
    changed = protocol.build_tf_flowers_protocol({"train": ds}, split_seed=43)
    assert first.manifest["records"] != changed.manifest["records"]
    assert first.manifest["source_fingerprints_status"] == "not_verified"
    assert not first.manifest["source_fingerprints_complete"]


@pytest.mark.parametrize("seed", [True, 4.2, "42"])
def test_rejects_noninteger_split_seed(seed):
    with pytest.raises(ValueError, match="integer"):
        protocol.build_tf_flowers_protocol({}, split_seed=seed)


def test_rejects_unknown_collection_or_fingerprint(synthetic_collection):
    ds, _ = synthetic_collection
    with pytest.raises(ValueError, match="only"):
        protocol.build_tf_flowers_protocol({"train": ds, "test": ds})
    with pytest.raises(ValueError, match="assets"):
        protocol.build_tf_flowers_protocol({"train": ds}, source_fingerprints={"other": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256"):
        protocol.build_tf_flowers_protocol({"train": ds}, source_fingerprints={"archive": "0" * 64})
    with pytest.raises(ValueError, match="image_ids"):
        protocol.build_tf_flowers_protocol({"train": ds.select(range(3669))})


def test_rejects_swapped_labels_even_when_class_counts_are_preserved(synthetic_collection):
    ds, _ = synthetic_collection
    rows = ds.table.to_pylist()
    rows[0]["label"], rows[-1]["label"] = rows[-1]["label"], rows[0]["label"]
    changed = StableDataset(
        features=ds.features,
        info=ds.info,
        table=pa.Table.from_pylist(rows, schema=ds.features.to_arrow_schema()),
    )
    with pytest.raises(ValueError, match="labels disagree"):
        protocol.build_tf_flowers_protocol({"train": changed})


def test_real_tf_flowers_builder_registration():
    from benchmarks.dataset import _get_dataset_class, get_config
    from stable_datasets.images import TFFlowers

    assert _get_dataset_class(get_config("tf_flowers")) is TFFlowers


def test_real_registered_builder_routes_full_collection_without_legacy_fallback(
    synthetic_collection, monkeypatch, tmp_path
):
    from benchmarks import dataset as benchmark_dataset
    from stable_datasets.images import TFFlowers

    ds, _ = synthetic_collection
    fingerprints = {"archive": protocol.ARCHIVE_SHA256}
    expected = protocol.build_tf_flowers_protocol({"train": ds}, split_seed=43, source_fingerprints=fingerprints)
    calls = []

    def offline_builder_entry(cls, split=None, **kwargs):
        assert cls is TFFlowers
        calls.append((split, kwargs))
        return {"train": ds}

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("tf_flowers must use the group protocol, without legacy validation or 10% holdout")

    # Preserve the production registry lookup. Replace only the real builder's
    # I/O entry with full-size offline metadata; invalid image bytes cannot decode.
    monkeypatch.setattr(TFFlowers, "__new__", staticmethod(offline_builder_entry))
    monkeypatch.setattr(benchmark_dataset, "_load_validation_split", forbidden_fallback)
    monkeypatch.setattr(StableDataset, "train_test_split", forbidden_fallback)
    assert benchmark_dataset._get_dataset_class(benchmark_dataset.get_config("tf_flowers")) is TFFlowers
    data, config = benchmark_dataset.create_dataset(
        "tf_flowers",
        train_transform=None,
        val_transform=None,
        collate_fn=None,
        training_cfg=SimpleNamespace(batch_size=4, num_workers=0),
        data_dir=str(tmp_path),
        split_seed=43,
        source_fingerprints=fingerprints,
    )
    assert config.num_classes == 5
    assert config.channels == 3
    assert data.dataset_protocol_manifest == expected.manifest
    for actual, target in ((data.train.dataset, expected.train_fit), (data.val.dataset, expected.validation)):
        assert [row["image_id"] for row in actual.with_format("raw")] == [
            row["image_id"] for row in target.with_format("raw")
        ]
    assert calls == [
        (None, {"download_dir": str(tmp_path / "downloads"), "processed_cache_dir": str(tmp_path / "processed")})
    ]


def test_multilabel_group_allocation_uses_counts_not_a_conflict_role():
    groups = [
        {
            "group_id": protocol.group_id([f"roses/{i}.jpg", f"tulips/{i}.jpg"]),
            "image_ids": [f"roses/{i}.jpg", f"tulips/{i}.jpg"],
            "class_counts": {"roses": 1, "tulips": 1},
        }
        for i in range(20)
    ]
    assignments, balance = protocol._allocate(groups, {"train": 3, "test": 1}, seed=42, stage="test")
    assert len(assignments["train"]) == 15
    assert len(assignments["test"]) == 5
    assert balance["actual_class_counts"]["test"] == {
        "dandelion": 0,
        "daisy": 0,
        "tulips": 5,
        "sunflowers": 0,
        "roses": 5,
    }
    assert Counter(g["group_id"] for group_list in assignments.values() for g in group_list) == Counter(
        g["group_id"] for g in groups
    )
