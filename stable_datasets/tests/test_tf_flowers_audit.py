"""Offline CPU checks for explicit source import and the TF Flowers audit CLI."""

import builtins
import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from examples import audit_tf_flowers as audit
from stable_datasets import utils
from stable_datasets.images import TFFlowers


@pytest.fixture
def release(monkeypatch, tmp_path):
    """Only the release size/checksum/count contract is reduced for this fixture."""
    image_ids = ["roses/17_original.jpg", "tulips/18_original.jpg"]
    credits = [
        image_ids[0] + " CC-BY by Author  One - https://www.flickr.com/photos/example/17/",
        image_ids[1] + " CC-BY https://www.flickr.com/photos/example/ - by Author  Heart ♥",
    ]
    license_bytes = (audit.LICENSE_HEADER + "\n\n" + "\n".join(credits) + "\n").encode()
    archive = tmp_path / "source.tgz"
    pixels = bytes(range(18))
    buffer = io.BytesIO()
    Image.frombytes("RGB", (3, 2), pixels).save(buffer, format="JPEG")
    payload = buffer.getvalue()
    with tarfile.open(archive, "w:gz") as target:
        for name, content in [
            ("flower_photos/" + image_ids[0], payload),
            ("flower_photos/LICENSE.txt", license_bytes),
            ("flower_photos/" + image_ids[1], payload),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            target.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(audit, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(audit, "ARCHIVE_SHA256", digest)
    monkeypatch.setattr(audit, "CLASS_COUNTS", {"roses": 1, "tulips": 1})
    monkeypatch.setattr(TFFlowers.SOURCE.assets["archive"], "checksum", "sha256:" + digest)

    def forbid_network(*args, **kwargs):
        pytest.fail("This test may only use an explicitly verified local archive")

    monkeypatch.setattr(utils.requests.sessions.Session, "request", forbid_network)
    return archive, image_ids, credits, license_bytes, payload


def test_pixel_evidence_includes_unconverted_mode_and_dimensions():
    content = bytes(range(12))
    images = [Image.frombytes("RGB", (2, 2), content), Image.frombytes("RGB", (1, 4), content)]
    images.append(Image.frombytes("L", (2, 6), content))
    fingerprints = [audit.pixel_fingerprint(image) for image in images]
    assert len(set(fingerprints)) == 3
    assert fingerprints[0] == hashlib.sha256(b"RGB\x002,2\x00" + content).hexdigest()


def test_explicit_import_uses_existing_download_cache_and_reuses_verified_bytes(tmp_path, release):
    archive, _, _, _, _ = release
    downloads = tmp_path / "downloads"
    path, source = audit.prepare_source(downloads, archive)
    assert path == audit._archive_cache_path(downloads)
    assert path.read_bytes() == archive.read_bytes()
    assert source["method"] == "explicit_checksum_verified_local_archive_import"
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["bytes"] == path.stat().st_size
    before = path.stat().st_mtime_ns
    assert audit.prepare_source(downloads, archive)[0] == path
    assert path.stat().st_mtime_ns == before
    _, cached = audit.prepare_source(downloads)
    assert cached["method"] == "public_source_or_existing_verified_download_cache"


@pytest.mark.parametrize("change", ["size", "checksum"])
def test_explicit_import_rejects_invalid_input_before_creating_cache(tmp_path, release, change):
    archive, _, _, _, _ = release
    content = archive.read_bytes()
    archive.write_bytes(content + b"invalid" if change == "size" else bytes([content[0] ^ 1]) + content[1:])
    downloads = tmp_path / "downloads"
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        audit.prepare_source(downloads, archive)
    assert not downloads.exists()


@pytest.mark.parametrize("explicit_import", [True, False])
def test_existing_corrupt_cache_is_reverified_and_preserved(tmp_path, release, explicit_import):
    archive, _, _, _, _ = release
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    target = audit._archive_cache_path(downloads)
    target.write_bytes(b"preserve this invalid existing file")
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        audit.prepare_source(downloads, archive if explicit_import else None)
    assert target.read_bytes() == b"preserve this invalid existing file"


def test_audits_native_cold_warm_pixels_labels_and_exact_credit_lines(monkeypatch, tmp_path, release):
    archive, image_ids, credits, license_bytes, payload = release
    rows, actual_license = audit.source_records(archive)
    assert actual_license == license_bytes
    assert [row["image_id"] for row in rows] == image_ids
    assert [row["label"] for row in rows] == [4, 2]
    assert [row["attribution"][0]["original_line"] for row in rows] == credits
    assert [row["attribution"][0]["url_kind"] for row in rows] == ["photo", "author_profile"]
    assert [row["attribution_flickr_photo_id"] for row in rows] == ["17", None]
    assert [row["attribution"][0]["line"] for row in rows] == [5, 6]
    with Image.open(io.BytesIO(payload)) as image:
        expected = hashlib.sha256(b"RGB\x003,2\x00" + image.tobytes()).hexdigest()
    assert all(row["decoded_source_pixels_sha256"] == expected for row in rows)
    assert all(row["decoded_rgb_pixels_sha256"] == expected for row in rows)
    assert all(row["source_mode"] == "RGB" and row["size"] == [3, 2] for row in rows)
    monkeypatch.setattr(audit.builder_module, "bulk_download", lambda *args, **kwargs: [archive])
    kwargs = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    cold = TFFlowers(**kwargs)
    expected_rows = {row["image_id"]: row for row in rows}
    assert audit.check_cache(cold, expected_rows) == 2
    with audit._warm_cache_only(archive):
        warm = TFFlowers(**kwargs)
        assert audit.check_cache(warm, expected_rows) == 2
        assert list(warm.with_format("raw")) == list(cold.with_format("raw"))
    for field, value, message in (
        ("label", 0, "label"),
        ("decoded_rgb_pixels_sha256", "0" * 64, "pixels/mode/size"),
        ("size", [2, 3], "pixels/mode/size"),
        ("attribution", [{"original_line": "changed"}], "attribution"),
    ):
        altered = copy.deepcopy(expected_rows)
        altered[image_ids[0]][field] = value
        with pytest.raises(ValueError, match=message):
            audit.check_cache(cold, altered)


def test_warm_guard_blocks_actual_raw_access_but_allows_other_files(tmp_path, release):
    archive, _, _, _, _ = release
    processed = tmp_path / "processed.arrow"
    processed.write_bytes(b"processed cache access remains available")
    calls = [
        lambda: builtins.open(archive, "rb"),
        lambda: builtins.open(str(archive), "rb"),
        lambda: archive.open("rb"),
        lambda: archive.read_bytes(),
        lambda: audit.gzip.open(archive, "rb"),
        lambda: audit.tarfile.open(archive),
        lambda: audit.builder_module._verify_archive_checksum(archive, "unused"),
        lambda: audit.builder_module.bulk_download([], tmp_path),
        lambda: utils.download("https://example.invalid/never", dest_folder=tmp_path),
        lambda: utils.bulk_download([], tmp_path),
        lambda: utils.requests.Session().get("https://example.invalid/never"),
    ]
    with audit._warm_cache_only(archive) as blocked:
        for call in calls:
            with pytest.raises(AssertionError, match="Warm cache must not"):
                call()
        with builtins.open(processed, "rb") as stream:
            assert stream.read() == processed.read_bytes()
        assert "pathlib.Path.open (raw archive path)" in blocked
        assert "stable_datasets.images.tf_flowers.bulk_download" in blocked
    assert archive.read_bytes()


def test_verify_frozen_never_builds_a_missing_cache(tmp_path):
    frozen = tmp_path / "manifest.json"
    frozen.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(AssertionError, match="Warm cache must not"):
        audit.main(["--data-root", str(tmp_path / "data"), "--verify-frozen", str(frozen)])
    assert not list((tmp_path / "data").rglob("_metadata.json"))


def test_verify_frozen_cannot_import_archive(tmp_path, release):
    archive, _, _, _, _ = release
    with pytest.raises(SystemExit) as error:
        audit.main(
            [
                "--data-root",
                str(tmp_path / "data"),
                "--verify-frozen",
                str(tmp_path / "unused.json"),
                "--archive",
                str(archive),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "data").exists()


def test_cli_rejects_data_paths_inside_checkout():
    checkout = Path(audit.__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside the repository"):
        audit.main(["--data-root", str(checkout)])
