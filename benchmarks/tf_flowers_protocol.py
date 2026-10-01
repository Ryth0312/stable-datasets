"""Archive-bound, global group splits for the TensorFlow Flowers collection.

The native builder has one collection, not an official train/test partition.
This module assigns whole connected components before making dataset views.
It never downloads data, decodes images, or reads an external audit directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from numbers import Integral
from pathlib import PurePosixPath
from urllib.parse import urlsplit


PROTOCOL_VERSION = "sds-tfflowers-group-v1"
ARCHIVE_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
CLASS_NAMES = ("dandelion", "daisy", "tulips", "sunflowers", "roses")
CLASS_COUNTS = {"dandelion": 898, "daisy": 633, "tulips": 799, "sunflowers": 699, "roses": 641}

# Generated from a full audit of the SHA-256-pinned official archive. The compact
# evidence covers every non-singleton component; full-collection digests let an
# independent audit detect omitted relationships or changed source identities.
_BUNDLED_GROUP_METADATA_JSON = r"""{
  "archive_sha256": "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c",
  "class_counts": {
    "daisy": 633,
    "dandelion": 898,
    "roses": 641,
    "sunflowers": 699,
    "tulips": 799
  },
  "groups": [
    {
      "class_counts": {
        "sunflowers": 2
      },
      "evidence": [
        {
          "image_ids": [
            "sunflowers/14889392928_9742aed45b_m.jpg",
            "sunflowers/15066430311_fb57fa92b0_m.jpg"
          ],
          "key": "1235dc340983196227bcd84cdd1d598b97348373dc96b60871d16a63b9b30d5f",
          "kind": "file_sha256"
        },
        {
          "image_ids": [
            "sunflowers/14889392928_9742aed45b_m.jpg",
            "sunflowers/15066430311_fb57fa92b0_m.jpg"
          ],
          "key": {
            "mode": "RGB",
            "sha256": "c803aa7ddfca49215e8fa1f2d9418b9641af602bc6ef06d290c900147f1a0ba0",
            "size": [
              240,
              221
            ]
          },
          "kind": "native_pixels"
        }
      ],
      "group_id": "307bfc1bb65cfb0dae958e60e1c50195263feb328b4c3b96d96685b80d569bd7",
      "image_ids": [
        "sunflowers/14889392928_9742aed45b_m.jpg",
        "sunflowers/15066430311_fb57fa92b0_m.jpg"
      ]
    },
    {
      "class_counts": {
        "sunflowers": 2
      },
      "evidence": [
        {
          "image_ids": [
            "sunflowers/15069459615_7e0fd61914_n.jpg",
            "sunflowers/15072973261_73e2912ef2_n.jpg"
          ],
          "key": "65ea2fd6f038f541bea6e28f077e666c755b81441ee6e2224191fc48f69b8d21",
          "kind": "file_sha256"
        },
        {
          "image_ids": [
            "sunflowers/15069459615_7e0fd61914_n.jpg",
            "sunflowers/15072973261_73e2912ef2_n.jpg"
          ],
          "key": {
            "mode": "RGB",
            "sha256": "1f8e6463d0a9cedb9069f04c9252ae4f6342aaf9fe68801d382b260e98fae39c",
            "size": [
              320,
              239
            ]
          },
          "kind": "native_pixels"
        }
      ],
      "group_id": "4847572fba7d909746af25ed6c10f64624c3069c1e2a3d139b2de8415a8ba5e5",
      "image_ids": [
        "sunflowers/15069459615_7e0fd61914_n.jpg",
        "sunflowers/15072973261_73e2912ef2_n.jpg"
      ]
    },
    {
      "class_counts": {
        "roses": 1,
        "tulips": 1
      },
      "evidence": [
        {
          "image_ids": [
            "roses/15922772266_1167a06620.jpg",
            "tulips/15922772266_1167a06620.jpg"
          ],
          "key": "6784c09ff4eeec0e0ac148cdc74682f99c424ac89d85321d95161e20da9204e9",
          "kind": "file_sha256"
        },
        {
          "image_ids": [
            "roses/15922772266_1167a06620.jpg",
            "tulips/15922772266_1167a06620.jpg"
          ],
          "key": "15922772266",
          "kind": "verified_photo_id"
        },
        {
          "image_ids": [
            "roses/15922772266_1167a06620.jpg",
            "tulips/15922772266_1167a06620.jpg"
          ],
          "key": {
            "mode": "RGB",
            "sha256": "f87442547e7bcb81e278ada2546fe7c97c4ce40bfbfb4da448e75cb12287ce46",
            "size": [
              500,
              333
            ]
          },
          "kind": "native_pixels"
        }
      ],
      "group_id": "5034e7d826d4367ac5bde715979c586f79ec1ada58ba29b9a9bb1e04f0509d71",
      "image_ids": [
        "roses/15922772266_1167a06620.jpg",
        "tulips/15922772266_1167a06620.jpg"
      ]
    }
  ],
  "image_count": 3670,
  "image_ids_sha256": "5674fe020188d8f60128946c69dab61d7c133ee993329233541b08208bbd7fb0",
  "members": [
    {
      "file_sha256": "6784c09ff4eeec0e0ac148cdc74682f99c424ac89d85321d95161e20da9204e9",
      "image_id": "roses/15922772266_1167a06620.jpg",
      "mode": "RGB",
      "pixels_sha256": "f87442547e7bcb81e278ada2546fe7c97c4ce40bfbfb4da448e75cb12287ce46",
      "size": [
        500,
        333
      ],
      "verified_photo_ids": [
        "15922772266"
      ]
    },
    {
      "file_sha256": "1235dc340983196227bcd84cdd1d598b97348373dc96b60871d16a63b9b30d5f",
      "image_id": "sunflowers/14889392928_9742aed45b_m.jpg",
      "mode": "RGB",
      "pixels_sha256": "c803aa7ddfca49215e8fa1f2d9418b9641af602bc6ef06d290c900147f1a0ba0",
      "size": [
        240,
        221
      ],
      "verified_photo_ids": [
        "14889392928"
      ]
    },
    {
      "file_sha256": "1235dc340983196227bcd84cdd1d598b97348373dc96b60871d16a63b9b30d5f",
      "image_id": "sunflowers/15066430311_fb57fa92b0_m.jpg",
      "mode": "RGB",
      "pixels_sha256": "c803aa7ddfca49215e8fa1f2d9418b9641af602bc6ef06d290c900147f1a0ba0",
      "size": [
        240,
        221
      ],
      "verified_photo_ids": [
        "15066430311"
      ]
    },
    {
      "file_sha256": "65ea2fd6f038f541bea6e28f077e666c755b81441ee6e2224191fc48f69b8d21",
      "image_id": "sunflowers/15069459615_7e0fd61914_n.jpg",
      "mode": "RGB",
      "pixels_sha256": "1f8e6463d0a9cedb9069f04c9252ae4f6342aaf9fe68801d382b260e98fae39c",
      "size": [
        320,
        239
      ],
      "verified_photo_ids": [
        "15069459615"
      ]
    },
    {
      "file_sha256": "65ea2fd6f038f541bea6e28f077e666c755b81441ee6e2224191fc48f69b8d21",
      "image_id": "sunflowers/15072973261_73e2912ef2_n.jpg",
      "mode": "RGB",
      "pixels_sha256": "1f8e6463d0a9cedb9069f04c9252ae4f6342aaf9fe68801d382b260e98fae39c",
      "size": [
        320,
        239
      ],
      "verified_photo_ids": [
        "15072973261"
      ]
    },
    {
      "file_sha256": "6784c09ff4eeec0e0ac148cdc74682f99c424ac89d85321d95161e20da9204e9",
      "image_id": "tulips/15922772266_1167a06620.jpg",
      "mode": "RGB",
      "pixels_sha256": "f87442547e7bcb81e278ada2546fe7c97c4ce40bfbfb4da448e75cb12287ce46",
      "size": [
        500,
        333
      ],
      "verified_photo_ids": [
        "15922772266"
      ]
    }
  ],
  "schema_version": 1,
  "source_evidence_sha256": "594b9e6aaf3912b44d0020617f5be2eec8e54e64c6163ea4160788c5e1e8a272"
}"""
BUNDLED_GROUP_METADATA = json.loads(_BUNDLED_GROUP_METADATA_JSON)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def group_id(image_ids: Iterable[str]) -> str:
    """Identity of a component, based on every complete relative image path."""
    return _digest(sorted(image_ids))


def _check_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or PurePosixPath(value).as_posix() != value
        or value == "."
    ):
        raise ValueError(f"Invalid canonical relative image_id: {value!r}.")


def _check_sha(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Evidence digests must be lowercase SHA-256 hex strings.")


def _normalize_evidence(records: Iterable[Mapping]) -> list[dict]:
    """Read native-pixel observations and photo URLs from an archive audit.

    Only explicit Flickr photo URLs marked ``url_kind='photo'`` establish a
    photo-ID edge. Author names, profile URLs, and filename prefixes do not.
    Pixel equality includes original mode and width/height, without RGB/EXIF
    conversion. The caller must have measured these observations on the source.
    """
    normalized = []
    seen = set()
    for record in records:
        image_id = record["image_id"]
        _check_id(image_id)
        if image_id in seen:
            raise ValueError(f"Duplicate evidence image_id: {image_id!r}.")
        seen.add(image_id)
        file_hash = record["file_sha256"]
        pixel_hash = record["decoded_source_pixels_sha256"]
        _check_sha(file_hash)
        _check_sha(pixel_hash)
        mode = record["source_mode"]
        size = list(record["size"])
        if not isinstance(mode, str) or not mode or len(size) != 2:
            raise ValueError("Native pixel evidence needs an original mode and two dimensions.")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in size):
            raise ValueError("Native pixel dimensions must be positive integers.")
        photo_ids = set()
        for attribution in record.get("attribution", []):
            if attribution.get("url_kind") != "photo":
                continue
            url = urlsplit(attribution.get("source_url", ""))
            match = re.fullmatch(r"/photos/[^/]+/([0-9]+)/?", url.path)
            if url.scheme != "https" or url.netloc != "www.flickr.com" or not match or url.query or url.fragment:
                raise ValueError(f"Invalid verified Flickr photo URL for {image_id!r}.")
            photo_id = match[1]
            if attribution.get("flickr_photo_id") != photo_id:
                raise ValueError(f"Flickr photo ID disagrees with its attribution URL for {image_id!r}.")
            photo_ids.add(photo_id)
        normalized.append(
            {
                "image_id": image_id,
                "file_sha256": file_hash,
                "mode": mode,
                "size": [int(v) for v in size],
                "pixels_sha256": pixel_hash,
                "verified_photo_ids": sorted(photo_ids),
            }
        )
    return sorted(normalized, key=lambda record: record["image_id"])


def _components(records: list[dict]) -> list[dict]:
    parent = {record["image_id"]: record["image_id"] for record in records}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    buckets = defaultdict(list)
    for record in records:
        image_id = record["image_id"]
        buckets[("file_sha256", record["file_sha256"])].append(image_id)
        pixels = (record["mode"], tuple(record["size"]), record["pixels_sha256"])
        buckets[("native_pixels", pixels)].append(image_id)
        for photo_id in record["verified_photo_ids"]:
            buckets[("verified_photo_id", photo_id)].append(image_id)
    edges = []
    for (kind, key), members in buckets.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        for other in members[1:]:
            left, right = sorted((find(members[0]), find(other)))
            parent[right] = left
        if kind == "native_pixels":
            key = {"mode": key[0], "size": list(key[1]), "sha256": key[2]}
        edges.append({"kind": kind, "key": key, "image_ids": members})
    components = defaultdict(list)
    for image_id in sorted(parent):
        components[find(image_id)].append(image_id)
    result = []
    for members in components.values():
        evidence = [edge for edge in edges if edge["image_ids"][0] in members]
        result.append(
            {
                "group_id": group_id(members),
                "image_ids": members,
                "class_counts": dict(sorted(Counter(item.split("/")[0] for item in members).items())),
                "evidence": sorted(evidence, key=_digest),
            }
        )
    return sorted(result, key=lambda component: component["group_id"])


def connected_groups(records: Iterable[Mapping]) -> list[dict]:
    """Build global transitive components from independently observed evidence."""
    return _components(_normalize_evidence(records))


def reconstruct_group_metadata(records: Iterable[Mapping], *, archive_sha256: str) -> dict:
    """Rebuild the compact manifest from ALL 3670 original-image audit records.

    Input fields match the public audit observations: image_id, file_sha256,
    source_mode, size, decoded_source_pixels_sha256, and attribution entries
    (source_url, url_kind, flickr_photo_id). No archive is opened by this helper.
    ``archive_sha256`` must be computed from the full raw archive by the caller.
    """
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError("TFFlowers group evidence is bound to the official archive SHA-256.")
    normalized = _normalize_evidence(records)
    ids = [record["image_id"] for record in normalized]
    counts = dict(sorted(Counter(image_id.split("/")[0] for image_id in ids).items()))
    if counts != CLASS_COUNTS or len(ids) != 3670:
        raise ValueError("A full TFFlowers evidence audit needs all 3670 images and the original class counts.")
    groups = [component for component in _components(normalized) if len(component["image_ids"]) > 1]
    member_ids = {image_id for component in groups for image_id in component["image_ids"]}
    return {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "image_count": len(ids),
        "class_counts": counts,
        "image_ids_sha256": _digest(ids),
        "source_evidence_sha256": _digest(normalized),
        "groups": groups,
        "members": [record for record in normalized if record["image_id"] in member_ids],
    }


def verify_group_metadata(records: Iterable[Mapping], *, archive_sha256: str) -> dict:
    """Reject changed, incomplete, or newly connected full-archive evidence."""
    rebuilt = reconstruct_group_metadata(records, archive_sha256=archive_sha256)
    if rebuilt != BUNDLED_GROUP_METADATA:
        raise ValueError("Full TFFlowers source evidence differs from the bundled group manifest.")
    return rebuilt


def _seed_rank(seed: int, stage: str, *parts: str) -> str:
    return _digest([PROTOCOL_VERSION, stage, seed, *parts])


def _allocate(groups: list[dict], weights: Mapping[str, int], *, seed: int, stage: str) -> tuple[dict, dict]:
    """Whole-group greedy balance; exact rational costs make ties portable."""
    names = list(CLASS_NAMES)
    totals = Counter()
    for component in groups:
        totals.update(component["class_counts"])
    total = sum(totals.values())
    denominator = sum(weights.values())
    actual = {role: Counter() for role in weights}
    assigned = {role: [] for role in weights}
    order = sorted(groups, key=lambda g: (-len(g["image_ids"]), _seed_rank(seed, stage, g["group_id"]), g["group_id"]))
    for component in order:
        vector = component["class_counts"]
        size = len(component["image_ids"])

        def cost(role):
            # Increment in sum over roles of normalized squared per-class and
            # total-count errors from the unrounded target fractions.
            ratio = Fraction(weights[role], denominator)
            current = actual[role]
            delta = Fraction(0)
            for name in names:
                if totals[name]:
                    added = vector.get(name, 0)
                    delta += Fraction(2 * current[name] * added + added**2, totals[name] ** 2)
                    delta -= 2 * ratio * Fraction(added, totals[name])
            if total:
                delta += Fraction(2 * sum(current.values()) * size + size**2, total**2)
                delta -= 2 * ratio * Fraction(size, total)
            return delta, _seed_rank(seed, stage, component["group_id"], role), role

        role = min(weights, key=cost)
        actual[role].update(vector)
        assigned[role].append(component)
    target = {
        role: {name: str(Fraction(totals[name] * weight, denominator)) for name in names}
        for role, weight in weights.items()
    }
    stage_manifest = {
        "weights": dict(weights),
        "input_group_count": len(groups),
        "input_image_count": total,
        "target_total_counts": {role: str(Fraction(total * weight, denominator)) for role, weight in weights.items()},
        "target_class_counts": target,
        "actual_total_counts": {role: sum(counts.values()) for role, counts in actual.items()},
        "actual_class_counts": {role: {name: counts[name] for name in names} for role, counts in actual.items()},
        "deviation_class_counts": {
            role: {name: str(Fraction(actual[role][name]) - Fraction(target[role][name])) for name in names}
            for role in weights
        },
        "deviation_total_counts": {
            role: str(Fraction(sum(actual[role].values())) - Fraction(total * weight, denominator))
            for role, weight in weights.items()
        },
    }
    return assigned, stage_manifest


def build_tf_flowers_protocol(splits, *, split_seed=42, source_fingerprints=None):
    """Use the archive-bound global groups for approximately 60/20/20 views.

    Fingerprints are caller claims, not a new file verification. An absent
    archive fingerprint remains marked not_verified for offline use. A supplied
    fingerprint must match the one the bundled complete audit describes.
    """
    from .dataset_protocols import DatasetProtocol, _canonical_order, _read_records

    if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
        raise ValueError("split_seed must be an integer.")
    split_seed = int(split_seed)
    if set(splits) != {"train"}:
        raise ValueError("tf_flowers expects only the full native train collection.")
    fingerprints = dict(source_fingerprints or {})
    if set(fingerprints) - {"archive"}:
        raise ValueError("Unexpected tf_flowers source fingerprint assets.")
    if fingerprints and fingerprints.get("archive") != ARCHIVE_SHA256:
        raise ValueError("tf_flowers source archive SHA-256 differs from the audited group manifest.")
    records = _read_records(splits["train"], "train", list(CLASS_NAMES))
    ids = sorted(record.image_id for record in records)
    if len(ids) != BUNDLED_GROUP_METADATA["image_count"] or _digest(ids) != BUNDLED_GROUP_METADATA["image_ids_sha256"]:
        raise ValueError("tf_flowers image_ids differ from the complete audited collection.")
    if Counter(CLASS_NAMES[r.label] for r in records) != CLASS_COUNTS:
        raise ValueError("tf_flowers labels do not have the original per-class counts.")
    if any(r.image_id.split("/")[0] != CLASS_NAMES[r.label] for r in records):
        raise ValueError("tf_flowers labels disagree with their original image_id directories.")

    evidence_by_id = {r["image_id"]: r for r in BUNDLED_GROUP_METADATA["members"]}
    # Unlisted paths are singleton components of the complete pinned audit.
    groups = _components(list(evidence_by_id.values()))
    groups.extend(
        {
            "group_id": group_id([image_id]),
            "image_ids": [image_id],
            "class_counts": {image_id.split("/")[0]: 1},
            "evidence": [],
        }
        for image_id in ids
        if image_id not in evidence_by_id
    )
    outer, outer_manifest = _allocate(groups, {"train_full": 4, "test": 1}, seed=split_seed, stage="outer")
    inner, inner_manifest = _allocate(
        outer["train_full"], {"train_fit": 3, "validation": 1}, seed=split_seed, stage="inner"
    )
    assigned = {**inner, "train_full": outer["train_full"], "test": outer["test"]}
    by_id = {r.image_id: r for r in records}
    assignments = {
        role: _canonical_order([by_id[image_id] for component in components for image_id in component["image_ids"]])
        for role, components in assigned.items()
    }
    role_by_group = {g["group_id"]: role for role in ("train_fit", "validation", "test") for g in assigned[role]}
    group_by_id = {image_id: g["group_id"] for g in groups for image_id in g["image_ids"]}
    group_manifest = [
        {
            **component,
            "label_set": sorted(CLASS_NAMES.index(name) for name in component["class_counts"]),
            "role": role_by_group[component["group_id"]],
            "in_train_full": role_by_group[component["group_id"]] != "test",
        }
        for component in sorted(groups, key=lambda g: g["group_id"])
    ]
    duplicate_groups = [component for component in group_manifest if len(component["image_ids"]) > 1]
    cross_class_conflict_groups = [component for component in duplicate_groups if len(component["label_set"]) > 1]
    manifest = {
        "dataset": "tf_flowers",
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": split_seed,
        "class_names": list(CLASS_NAMES),
        "source_fingerprints": fingerprints,
        "source_fingerprints_status": "provided" if fingerprints else "not_verified",
        "source_fingerprints_complete": set(fingerprints) == {"archive"},
        "group_metadata_archive_sha256": ARCHIVE_SHA256,
        "group_metadata_sha256": _digest(BUNDLED_GROUP_METADATA),
        "source_evidence_sha256": BUNDLED_GROUP_METADATA["source_evidence_sha256"],
        "allocation": {
            "algorithm": "global connected components; outer 4:1, then inner 3:1; no label-conflict exception",
            "order": "descending group size, then SHA256(canonical JSON [version,stage,seed,group_id]), then group_id",
            "objective": "greedy minimum increase in sum of normalized squared per-class errors plus normalized squared total-count error, using exact Fractions and unrounded target ratios",
            "tie_break": "SHA256(canonical JSON [version,stage,seed,group_id,role]), then role name",
            "group_id": "SHA256(UTF8(canonical JSON of sorted full relative image_ids))",
            "outer": outer_manifest,
            "inner": inner_manifest,
        },
        "counts": {role: len(items) for role, items in assignments.items()},
        "group_counts": {role: len(components) for role, components in assigned.items()},
        "source_group_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "cross_class_conflict_group_count": len(cross_class_conflict_groups),
        "cross_class_conflict_groups": cross_class_conflict_groups,
        "per_class_counts": {
            role: {name: sum(r.label == label for r in items) for label, name in enumerate(CLASS_NAMES)}
            for role, items in {**assignments, "source_train": records, "source_test": []}.items()
        },
        "groups": group_manifest,
        "records": [
            {
                "image_id": r.image_id,
                "class_name": CLASS_NAMES[r.label],
                "label": r.label,
                "source_split": "train",
                "group_id": group_by_id[r.image_id],
                "role": role_by_group[group_by_id[r.image_id]],
                "in_train_full": role_by_group[group_by_id[r.image_id]] != "test",
            }
            for r in _canonical_order(records)
        ],
    }
    views = {role: splits["train"].select([record.index for record in items]) for role, items in assignments.items()}
    return DatasetProtocol(**views, manifest=manifest)
