"""Verify the public TF Flowers archive, PNG caches and frozen grouped protocol.

No model is loaded. All data/results must be outside the checkout. The optional
--archive imports already authenticated release bytes into the existing download
cache; without it the builder's public SOURCE is used unchanged.
"""

from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import requests
from filelock import FileLock
from PIL import Image

from benchmarks.dataset_protocols import build_dataset_protocol
from benchmarks.tf_flowers_protocol import CLASS_COUNTS, verify_group_metadata
from examples.evaluate_image_classification import _outside_checkout, code_fingerprint, sha256_file, write_json
from stable_datasets.images import TFFlowers
from stable_datasets.images import tf_flowers as builder_module
from stable_datasets.utils import download


ARCHIVE_BYTES = 228813984
ARCHIVE_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
LICENSE_HEADER = (
    "All images in this archive are licensed under the Creative Commons By-Attribution License, available at:\n"
    "https://creativecommons.org/licenses/by/2.0/\n"
    "The photographers are listed below, thanks to all of them for making their work available, "
    "and please be sure to credit them for any use as per the license."
)


def pixel_fingerprint(image):
    """Hash original mode, dimensions and untransformed decoded pixels."""
    prefix = image.mode.encode() + b"\0" + f"{image.width},{image.height}".encode() + b"\0"
    return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def verify_archive(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"TFDS release archive is not a regular file: {path}")
    if path.stat().st_size != ARCHIVE_BYTES or sha256_file(path) != ARCHIVE_SHA256:
        raise ValueError(f"TFDS release size/SHA-256 mismatch: {path}")


def _archive_cache_path(download_dir):
    asset = TFFlowers.SOURCE.assets["archive"]
    name = Path(asset.filename or urlparse(asset.url).path)
    suffix = hashlib.sha256(asset.url.encode()).hexdigest()[:10]
    return _outside_checkout(Path(download_dir) / f"{name.stem}.{suffix}{name.suffix}")


def prepare_source(download_dir, archive=None):
    """Resolve the builder asset, optionally importing a verified local copy.

    This explicit import uses the existing downloader's documented implementation
    layout (URL-hash suffix); it is optional and never required for public loading.
    Existing different files are rejected, not overwritten.
    """
    asset = TFFlowers.SOURCE.assets["archive"]
    if asset.checksum != "sha256:" + ARCHIVE_SHA256:
        raise ValueError("Builder SOURCE must retain the published TFDS checksum")
    download_dir = _outside_checkout(download_dir)
    target = _archive_cache_path(download_dir)
    method = "public_source_or_existing_verified_download_cache"
    if archive is not None:
        archive = _outside_checkout(archive)
        verify_archive(archive)
        download_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(target.with_suffix(target.suffix + ".lock")):
            if target.exists():
                verify_archive(target)
            else:
                temporary = _outside_checkout(target.with_suffix(target.suffix + ".importing"))
                with archive.open("rb") as reader, temporary.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                verify_archive(temporary)
                temporary.replace(target)
        method = "explicit_checksum_verified_local_archive_import"
    path = download(asset, dest_folder=download_dir, progress_bar=False)
    verify_archive(path)
    return path, {
        "url": asset.url,
        "sha256": ARCHIVE_SHA256,
        "bytes": ARCHIVE_BYTES,
        "path": str(path),
        "method": method,
    }


@contextmanager
def _warm_cache_only(archive):
    """Block actual raw entry points while leaving processed Arrow access intact."""
    archive = Path(archive).resolve()
    builtin_open, path_open = builtins.open, Path.open

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm cache must not download, hash, or open the raw archive")

    def check_path(path):
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == archive:
            forbidden()

    def guarded_open(file, *args, **kwargs):
        check_path(file)
        return builtin_open(file, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        check_path(path)
        return path_open(path, *args, **kwargs)

    blocked = [
        "stable_datasets.images.tf_flowers.bulk_download",
        "stable_datasets.images.tf_flowers._verify_archive_checksum",
        "stable_datasets.utils.download",
        "stable_datasets.utils.bulk_download",
        "requests.sessions.Session.request",
        "tarfile.open",
        "gzip.open",
        "builtins.open (raw archive path)",
        "pathlib.Path.open (raw archive path)",
    ]
    with ExitStack() as stack:
        stack.enter_context(patch.object(builder_module, "bulk_download", forbidden))
        stack.enter_context(patch.object(builder_module, "_verify_archive_checksum", forbidden))
        stack.enter_context(patch("stable_datasets.utils.download", forbidden))
        stack.enter_context(patch("stable_datasets.utils.bulk_download", forbidden))
        stack.enter_context(patch.object(requests.sessions.Session, "request", forbidden))
        stack.enter_context(patch.object(tarfile, "open", forbidden))
        stack.enter_context(patch.object(gzip, "open", forbidden))
        stack.enter_context(patch.object(builtins, "open", guarded_open))
        stack.enter_context(patch.object(Path, "open", guarded_path_open))
        yield blocked


def source_records(archive):
    """Independently scan all original photos and both published credit formats."""
    names = list(builder_module.TFFLOWERS_CLASS_NAMES)
    records = []
    license_bytes = None
    seen = set()
    members = set()
    with gzip.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as source:
            for member in source:
                name = member.name.removesuffix("/") if member.isdir() else member.name
                parts = name.split("/")
                if (
                    name in members
                    or "\\" in name
                    or ":" in name
                    or any(ord(char) < 32 for char in name)
                    or any(part in ("", ".", "..") for part in parts)
                    or parts[0] != "flower_photos"
                ):
                    raise ValueError(f"Unsafe or repeated archive member: {name}")
                members.add(name)
                if member.isdir():
                    if len(parts) > 2 or (len(parts) == 2 and parts[1] not in names):
                        raise ValueError(f"Unexpected archive directory: {name}")
                    continue
                if not member.isfile():
                    raise ValueError(f"Unexpected archive member type: {name}")
                payload = source.extractfile(member).read()
                if name == "flower_photos/LICENSE.txt":
                    license_bytes = payload
                    continue
                if len(parts) != 3 or parts[1] not in names or not parts[2].endswith(".jpg"):
                    raise ValueError(f"Unexpected source photograph path: {name}")
                image_id = "/".join(parts[1:])
                seen.add(image_id)
                label = names.index(parts[1])
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.format != "JPEG" or image.mode != "RGB":
                        raise ValueError(f"Expected released RGB JPEG photograph: {image_id}")
                    native = pixel_fingerprint(image)
                    rgb = image.convert("RGB")
                    records.append(
                        {
                            "image_id": image_id,
                            "label": label,
                            "class_name": names[label],
                            "file_sha256": hashlib.sha256(payload).hexdigest(),
                            "source_mode": image.mode,
                            "size": list(image.size),
                            "decoded_source_pixels_sha256": native,
                            "decoded_rgb_pixels_sha256": pixel_fingerprint(rgb),
                        }
                    )
        # Tar end markers precede the gzip trailer, whose CRC must also verify.
        while stream.read(1024 * 1024):
            pass
    if license_bytes is None:
        raise ValueError("Missing original LICENSE.txt")
    credits = {}
    lines = license_bytes.decode("utf-8").splitlines()
    if "\n".join(lines[:3]) != LICENSE_HEADER:
        raise ValueError("Unexpected original license declaration")
    for line_no, line in enumerate(lines[3:], 4):
        if not line:
            continue
        match = re.fullmatch(r"(\S+\.jpg) CC-BY by (.+) - (https?://\S+)", line)
        alternate = re.fullmatch(r"(\S+\.jpg) CC-BY (https?://\S+) - by (.+)", line)
        if match:
            image_id, author, url = match.groups()
        elif alternate:
            image_id, url, author = alternate.groups()
        else:
            raise ValueError(f"Unparsed photo credit at line {line_no}")
        photo = re.fullmatch(r"https://www\.flickr\.com/photos/[^/]+/(\d+)/?", url)
        profile = re.fullmatch(r"https://www\.flickr\.com/photos/[^/]+/?", url)
        if not photo and not profile:
            raise ValueError(f"Unrecognized published attribution URL at line {line_no}")
        if image_id in credits:
            raise ValueError("Repeated attribution path")
        credits[image_id] = {
            "author": author,
            "source_url": url,
            "line": line_no,
            "original_line": line,
            "url_kind": "photo" if photo else "author_profile",
            "flickr_photo_id": photo.group(1) if photo else None,
        }
    if set(credits) != seen or Counter(row["class_name"] for row in records) != CLASS_COUNTS:
        raise ValueError("Source/attribution membership mismatch")
    for row in records:
        credit = credits[row["image_id"]]
        row["attribution"] = [credit]
        row["attribution_flickr_photo_id"] = credit["flickr_photo_id"]
    return records, license_bytes


def check_cache(dataset, expected):
    """Check every cached PNG, ID, label and unmodified attribution string."""
    seen = set()
    for row in dataset.with_format("raw").with_transform(None):
        image_id = row["image_id"]
        if image_id in seen or image_id not in expected:
            raise ValueError(f"Unexpected/repeated cache image_id: {image_id}")
        seen.add(image_id)
        original = expected[image_id]
        if int(row["label"]) != original["label"]:
            raise ValueError(f"Cached label differs from its original class: {image_id}")
        if row["attribution"] != original["attribution"][0]["original_line"]:
            raise ValueError(f"Cached attribution differs from its original line: {image_id}")
        with Image.open(io.BytesIO(row["image"])) as image:
            if image.format != "PNG":
                raise ValueError(f"Cached photograph is not PNG: {image_id}")
            image.load()
            if (
                image.mode != "RGB"
                or list(image.size) != original["size"]
                or pixel_fingerprint(image) != original["decoded_rgb_pixels_sha256"]
            ):
                raise ValueError(f"Cached photograph pixels/mode/size differ: {image_id}")
    if seen != set(expected):
        raise ValueError("Cached photograph membership differs from the source")
    return len(seen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--archive", type=Path, help="Optional verified flower_photos.tgz; default uses public HTTPS")
    parser.add_argument("--verify-frozen", type=Path, help="Only compare warm metadata to a pre-training manifest")
    args = parser.parse_args(argv)
    data_root = _outside_checkout(args.data_root)
    downloads = _outside_checkout(args.download_dir or data_root / "downloads")
    processed = _outside_checkout(data_root / "processed")
    if args.verify_frozen is not None:
        if args.archive is not None:
            parser.error("--archive cannot be imported during --verify-frozen")
        frozen = json.loads(args.verify_frozen.read_text(encoding="utf-8"))
        with _warm_cache_only(_archive_cache_path(downloads)):
            splits = TFFlowers(split=None, download_dir=downloads, processed_cache_dir=processed)
            actual = build_dataset_protocol(
                "tf_flowers", splits, split_seed=42, source_fingerprints={"archive": ARCHIVE_SHA256}
            ).manifest
        if actual != frozen:
            raise ValueError("Current protocol differs from the frozen pre-training manifest")
        print("Frozen protocol matches all fields, members, labels and roles")
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for a complete source/cache audit")
    output = _outside_checkout(args.output_dir)
    if processed.exists() and any(processed.iterdir()):
        raise FileExistsError("Cold audit requires an empty processed cache; preserve existing runs")
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "model_or_gpu_run": False}
    write_json(output / "audit_summary.json", status)
    started = time.perf_counter()
    try:
        archive, source = prepare_source(downloads, args.archive)
        rows, license_bytes = source_records(archive)
        write_json(output / "source_records.json", rows)
        (output / "LICENSE.txt").write_bytes(license_bytes)
        write_json(output / "source.json", source)
        # Canonical equality covers every source observation, not just the six
        # members of known duplicate groups. Fail before building an altered set.
        evidence = verify_group_metadata(rows, archive_sha256=ARCHIVE_SHA256)
        write_json(output / "group_evidence_verification.json", evidence)
        expected = {row["image_id"]: row for row in rows}
        build_started = time.perf_counter()
        cold = TFFlowers(split=None, download_dir=downloads, processed_cache_dir=processed)
        build_elapsed = time.perf_counter() - build_started
        scan_started = time.perf_counter()
        count = check_cache(cold["train"], expected)
        cold_scan_elapsed = time.perf_counter() - scan_started

        warm_started = time.perf_counter()
        with _warm_cache_only(archive) as blocked:
            warm = TFFlowers(split=None, download_dir=downloads, processed_cache_dir=processed)
            if check_cache(warm["train"], expected) != count:
                raise ValueError("Warm cache count differs from the cold cache")
        warm_elapsed = time.perf_counter() - warm_started
        protocol = build_dataset_protocol(
            "tf_flowers", cold, split_seed=42, source_fingerprints={"archive": ARCHIVE_SHA256}
        )
        warm_protocol = build_dataset_protocol(
            "tf_flowers", warm, split_seed=42, source_fingerprints={"archive": ARCHIVE_SHA256}
        )
        if protocol.manifest != warm_protocol.manifest:
            raise ValueError("Cold/warm protocol manifests differ")
        write_json(output / "manifest.json", protocol.manifest)
        write_json(output / "code_fingerprint.json", code_fingerprint())
        status.update(
            status="passed",
            source=source,
            verified_source_cold_warm_records=count,
            class_counts=dict(Counter(row["class_name"] for row in rows)),
            png_records=count,
            source_attribution_preserved=True,
            warm_download_and_raw_tar_blocked=True,
            warm_blocked_entry_points=blocked,
            cold_build_seconds=build_elapsed,
            cold_full_scan_seconds=cold_scan_elapsed,
            warm_load_and_full_scan_seconds=warm_elapsed,
            manifest_sha256=sha256_file(output / "manifest.json"),
            counts=protocol.manifest["counts"],
            protocol_version=protocol.manifest["protocol_version"],
        )
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "audit_summary.json", status, overwrite=True)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
