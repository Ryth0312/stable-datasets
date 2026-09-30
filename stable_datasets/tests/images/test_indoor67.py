"""Offline Indoor67 fixtures plus an explicitly selected local-release test."""

import hashlib
import io
import os
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.images import indoor67
from stable_datasets.images.indoor67 import INDOOR67_CLASS_NAMES, Indoor67


def _jpeg(pixels):
    buffer = io.BytesIO()
    PILImage.fromarray(pixels).save(buffer, format="JPEG", quality=93, subsampling=0)
    data = buffer.getvalue()
    with PILImage.open(io.BytesIO(data)) as image:
        decoded = np.array(image.convert("RGB"))
    return data, decoded


def _write_tar(path, entries):
    with tarfile.open(path, "w") as archive:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = "bar/a.jpg"
                archive.addfile(member)


@pytest.fixture
def release(tmp_path):
    # Same basename across classes, spaces/case, and an unchanged upstream typo.
    image_ids = ["bar/Same.jpg", "mall/Same.jpg", "mall/shopping mall.jpg", "stairscase/Test.JPG"]
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 5
    entries = [("Images/", b"", tarfile.DIRTYPE)]
    expected = {}
    for index, image_id in enumerate(image_ids):
        data, decoded = _jpeg(pixels + index * 7)
        expected[image_id] = decoded
        entries.append(("Images/" + image_id, data, tarfile.REGTYPE))
    # This corrupt extra image must not be decoded or silently added to train.
    entries.append(("Images/bar/extra.jpg", b"not an image", tarfile.REGTYPE))
    paths = {
        key: tmp_path / name
        for key, name in {
            "images": "indoorCVPR_09.tar",
            "train_list": "TrainImages.txt",
            "test_list": "TestImages.txt",
        }.items()
    }
    paths["train_list"].write_text("\n".join(image_ids[:3]) + "\n", encoding="utf-8")
    paths["test_list"].write_text(image_ids[3] + "\n", encoding="utf-8")
    _write_tar(paths["images"], entries)
    return paths, entries, expected


def _load(monkeypatch, tmp_path, paths, *, split=None):
    def local_download(specs, dest_folder):
        assert Path(dest_folder) == tmp_path / "downloads"
        assert [spec.url for spec in specs] == [asset.url for asset in Indoor67.SOURCE.assets.values()]
        return [paths[key] for key in ("images", "train_list", "test_list")]

    monkeypatch.setattr(indoor67, "bulk_download", local_download)
    return Indoor67(split=split, processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads")


def test_metadata_does_not_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metadata must not download")

    monkeypatch.setattr(indoor67, "bulk_download", forbidden)
    builder = object.__new__(Indoor67)
    builder.__init__()
    assert str(builder.VERSION) == "1.0.0"
    assert builder._candidate_splits() == ["train", "test"]
    assert INDOOR67_CLASS_NAMES == sorted(INDOOR67_CLASS_NAMES)
    assert len(set(INDOOR67_CLASS_NAMES)) == 67
    assert "stairscase" in INDOOR67_CLASS_NAMES
    assert "staircase" not in INDOOR67_CLASS_NAMES
    assert builder.info.supervised_keys == ("image", "label")
    assert set(builder.info.features) == {"image", "label", "image_id"}
    assert builder.info.features["label"].names == INDOOR67_CLASS_NAMES
    assert builder.info.features["image"].encode_format == "PNG"
    assert "Research purposes only" in builder.info.license


def test_membership_pixels_formats_and_warm_cache(monkeypatch, tmp_path, release):
    import torch

    paths, _, expected = release
    real_open = tarfile.open
    archive_opens = []

    def counted_open(*args, **kwargs):
        archive_opens.append(kwargs.get("mode"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(indoor67.tarfile, "open", counted_open)
    datasets = _load(monkeypatch, tmp_path, paths)
    assert set(datasets) == {"train", "test"}
    assert [len(datasets[split]) for split in ("train", "test")] == [3, 1]
    assert archive_opens == ["r|*", "r|*"]
    for split, dataset in datasets.items():
        assert {row["image_id"] for row in dataset} == set(paths[f"{split}_list"].read_text().splitlines())
        for index, row in enumerate(dataset):
            image_id = row["image_id"]
            assert set(row) == {"image", "label", "image_id"}
            assert isinstance(row["image"], PILImage.Image)
            assert row["image"].mode == "RGB"
            np.testing.assert_array_equal(np.array(row["image"]), expected[image_id])
            assert row["label"] == INDOOR67_CLASS_NAMES.index(image_id.split("/")[0])
            arr = dataset.with_format("numpy")[index]["image"]
            assert arr.shape == (3, 5, 3) and arr.dtype == np.uint8
            np.testing.assert_array_equal(arr, expected[image_id])
            tensor = dataset.with_format("torch")[index]["image"]
            assert tensor.dtype == torch.float32
            np.testing.assert_array_equal(
                tensor.numpy(), expected[image_id].transpose(2, 0, 1).astype(np.float32) / 255
            )
            assert dataset.with_format("raw")[index]["image"].startswith(b"\x89PNG\r\n\x1a\n")

    cache_files = {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (tmp_path / "processed").rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("warm cache must not access raw assets")

    monkeypatch.setattr(indoor67, "bulk_download", forbidden)
    monkeypatch.setattr(indoor67.tarfile, "open", forbidden)
    for split in (None, "train", "test"):
        warm = Indoor67(split=split, processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads")
        loaded = warm if split is None else {split: warm}
        for dataset in loaded.values():
            for row in dataset:
                np.testing.assert_array_equal(np.array(row["image"]), expected[row["image_id"]])
    assert cache_files == {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in cache_files}


@pytest.mark.parametrize("prefix", ["", "./Images/"])
def test_supported_archive_layouts(monkeypatch, tmp_path, release, prefix):
    paths, entries, expected = release
    _write_tar(
        paths["images"], [(prefix + name.removeprefix("Images/"), data, kind) for name, data, kind in entries[1:]]
    )
    datasets = _load(monkeypatch, tmp_path, paths)
    assert set(expected) == {row["image_id"] for dataset in datasets.values() for row in dataset}


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        ("bar/Same.jpg\nbar/Same.jpg\n", "stairscase/Test.JPG\n", "duplicate image ID"),
        ("bar/Same.jpg\n", "bar/Same.jpg\n", "lists overlap"),
        ("", "stairscase/Test.JPG\n", "empty train list"),
        ("../bar/Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("/bar/Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("bar\\Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("bar/../Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("bar//Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("C:/Same.jpg\n", "stairscase/Test.JPG\n", "unsafe path"),
        ("staircase/Same.jpg\n", "stairscase/Test.JPG\n", "unknown class"),
        ("Bar/Same.jpg\n", "stairscase/Test.JPG\n", "unknown class"),
        ("bar/nested/Same.jpg\n", "stairscase/Test.JPG\n", "expected class/image.jpg"),
    ],
)
def test_bad_manifests(monkeypatch, tmp_path, release, train, test, message):
    paths, _, _ = release
    paths["train_list"].write_text(train, encoding="utf-8")
    paths["test_list"].write_text(test, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, tmp_path, paths)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("Images/../bar/not-listed.jpg", tarfile.REGTYPE, "unsafe path"),
        ("/Images/bar/not-listed.jpg", tarfile.REGTYPE, "unsafe path"),
        ("Images/bar\\not-listed.jpg", tarfile.REGTYPE, "unsafe path"),
        ("Images/bar/C:evil.jpg", tarfile.REGTYPE, "unsafe path"),
        ("unexpected/Images/bar/not-listed.jpg", tarfile.REGTYPE, "unknown class or archive prefix"),
        ("Images/unknown/not-listed.jpg", tarfile.REGTYPE, "unknown class"),
        ("Images/bar/Same.jpg", tarfile.REGTYPE, "duplicate/conflicting image ID"),
        ("bar/Same.jpg", tarfile.REGTYPE, "duplicate/conflicting image ID"),
        ("bar/different.jpg", tarfile.REGTYPE, "mixed wrapped and unwrapped"),
        ("Images/bar/link.jpg", tarfile.SYMTYPE, "archive links"),
        ("Images/bar/link.jpg", tarfile.LNKTYPE, "archive links"),
        ("Images/bar/device.jpg", tarfile.CHRTYPE, "unsupported archive entry"),
        ("Images/bar/nested/", tarfile.DIRTYPE, "unsupported archive directory"),
    ],
)
def test_bad_archive_members(monkeypatch, tmp_path, release, name, kind, message):
    paths, entries, _ = release
    _write_tar(paths["images"], entries + [(name, b"invalid", kind)])
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, tmp_path, paths)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_missing_listed_image_fails(monkeypatch, tmp_path, release):
    paths, entries, _ = release
    _write_tar(paths["images"], [entry for entry in entries if entry[0] != "Images/bar/Same.jpg"])
    with pytest.raises(FileNotFoundError, match="bar/Same.jpg"):
        _load(monkeypatch, tmp_path, paths)


def test_corrupt_listed_image_fails(monkeypatch, tmp_path, release):
    paths, entries, _ = release
    _write_tar(
        paths["images"],
        [(name, b"broken" if name == "Images/bar/Same.jpg" else data, kind) for name, data, kind in entries],
    )
    with pytest.raises(ValueError, match="cannot decode train image 'bar/Same.jpg'"):
        _load(monkeypatch, tmp_path, paths)


def test_grayscale_jpeg_is_loaded_as_rgb(monkeypatch, tmp_path, release):
    paths, entries, _ = release
    data, expected = _jpeg(np.arange(15, dtype=np.uint8).reshape(3, 5) * 13)
    _write_tar(
        paths["images"],
        [(name, data if name == "Images/stairscase/Test.JPG" else original, kind) for name, original, kind in entries],
    )
    dataset = _load(monkeypatch, tmp_path, paths, split="test")
    assert dataset[0]["image"].mode == "RGB"
    np.testing.assert_array_equal(np.array(dataset[0]["image"]), expected)


def test_listed_filename_case_is_not_normalized(monkeypatch, tmp_path, release):
    paths, entries, _ = release
    _write_tar(
        paths["images"],
        [
            ("Images/bar/same.jpg" if name == "Images/bar/Same.jpg" else name, data, kind)
            for name, data, kind in entries
        ],
    )
    with pytest.raises(FileNotFoundError, match="bar/Same.jpg"):
        _load(monkeypatch, tmp_path, paths)


@pytest.mark.large
def test_indoor67_local_release(monkeypatch, tmp_path):
    """Manually opt in with an approved archive and the two unchanged manifests.

    Set STABLE_DATASETS_INDOOR67_RELEASE_DIR to their directory. No network
    fallback is attempted, and invalid/missing opted-in assets fail the test.
    """
    release_dir = os.environ.get("STABLE_DATASETS_INDOOR67_RELEASE_DIR")
    if release_dir is None:
        pytest.skip("Set STABLE_DATASETS_INDOOR67_RELEASE_DIR to run the full local-release audit")
    root = Path(release_dir)
    paths = {
        "images": root / "indoorCVPR_09.tar",
        "train_list": root / "TrainImages.txt",
        "test_list": root / "TestImages.txt",
    }
    assert all(path.is_file() for path in paths.values())
    for key in ("train_list", "test_list"):
        assert "sha256:" + hashlib.sha256(paths[key].read_bytes()).hexdigest() == Indoor67.SOURCE.assets[key].checksum
    datasets = _load(monkeypatch, tmp_path, paths)
    expected = {
        split: set(paths[f"{split}_list"].read_text(encoding="utf-8").splitlines()) for split in ("train", "test")
    }
    assert len(datasets["train"]) == 5360
    assert len(datasets["test"]) == 1340
    assert not (expected["train"] & expected["test"])
    for split, dataset in datasets.items():
        actual = set()
        labels = Counter()
        for row in dataset:
            image_id = row["image_id"]
            assert image_id not in actual
            actual.add(image_id)
            assert row["image"].mode == "RGB"
            assert row["label"] == INDOOR67_CLASS_NAMES.index(image_id.split("/")[0])
            labels[image_id.split("/")[0]] += 1
        assert actual == expected[split]
        assert labels == Counter(image_id.split("/")[0] for image_id in expected[split])
        assert set(labels) == set(INDOOR67_CLASS_NAMES)
