"""Frozen ImageNet-supervised ResNet-50 + linear classification for Indoor67/FMD.

Run with ``python -m examples.evaluate_image_classification --help``.
All datasets, model weights, features, and results must live outside the checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
C_VALUES = (0.1, 1.0, 10.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, overwrite=False):
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def code_fingerprint(root=None):
    """Record HEAD and the tracked diff plus relevant untracked source files."""
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    # Also fingerprint actual source bytes when Git metadata is unavailable.
    # New builders and this script can still be untracked during local validation.
    implementation_files = [
        "examples/evaluate_image_classification.py",
        "benchmarks/dataset_protocols.py",
        "stable_datasets/images/indoor67.py",
        "stable_datasets/images/fmd.py",
        "stable_datasets/features/image.py",
        "stable_datasets/schema.py",
        "stable_datasets/cache.py",
        "stable_datasets/dataset.py",
        "stable_datasets/utils.py",
    ]
    implementation = {name: sha256_file(root / name) for name in implementation_files if (root / name).is_file()}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root, stderr=subprocess.DEVNULL)
        untracked = (
            subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .split("\0")
        )
    except (OSError, subprocess.CalledProcessError):
        snapshot_path = root / "CODE_SNAPSHOT.json"
        if snapshot_path.is_file():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            files = snapshot.get("files")
            if not isinstance(files, dict) or not files or not snapshot.get("head") or not snapshot.get("diff_sha256"):
                raise ValueError("CODE_SNAPSHOT.json must contain head, diff_sha256 and a non-empty files mapping")
            for name, expected in files.items():
                path = (root / name).resolve()
                if (
                    not name
                    or Path(name).is_absolute()
                    or "\\" in name
                    or ":" in name
                    or ".." in Path(name).parts
                    or not path.is_relative_to(root)
                    or not path.is_file()
                ):
                    raise ValueError(f"Invalid or missing code snapshot file: {name!r}")
                if sha256_file(path) != expected:
                    raise ValueError(f"Code snapshot hash mismatch: {name}")
            if set(implementation) - set(files):
                raise ValueError("Code snapshot omits relevant implementation files")
            return {**snapshot, "status": "verified_export_snapshot", "implementation_sha256": implementation}
        return {
            "head": None,
            "diff_sha256": None,
            "status": "git_metadata_unavailable",
            "implementation_sha256": implementation,
        }
    digest = hashlib.sha256(diff)
    files = {}
    for name in sorted(untracked):
        if name.startswith(("stable_datasets/", "benchmarks/", "examples/", "docs/")):
            path = root / name
            files[name] = sha256_file(path)
            digest.update(b"\0" + name.encode() + b"\0" + path.read_bytes())
    return {
        "head": head,
        "diff_sha256": digest.hexdigest(),
        "untracked_source_sha256": files,
        "implementation_sha256": implementation,
    }


def environment():
    versions = {}
    for name in ("torch", "torchvision", "numpy", "pillow", "scikit-learn", "pyarrow", "stable-pretraining"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


class FeatureImages(Dataset):
    """Read whole RGB images only; FMD masks never enter the classifier."""

    def __init__(self, sources, transform):
        self.sources = [source.with_format("raw").with_transform(None) for source in sources]
        self.transform = transform
        self.locations = [(source, index) for source in self.sources for index in range(len(source))]

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        source, row_index = self.locations[index]
        sample = source[row_index]
        # Decode only the image column, even when a builder also preserves masks.
        image = source.features["image"].format(sample["image"], format_type="default").convert("RGB")
        return self.transform(image), sample["image_id"], int(sample["label"])


def extract_features(model, loader, device):
    """Extract embeddings with frozen parameters and fixed batch statistics."""
    model.eval()
    model.requires_grad_(False)
    features, image_ids, labels = [], [], []
    with torch.inference_mode():
        for images, batch_ids, batch_labels in loader:
            features.append(model(images.to(device)).flatten(1).cpu().numpy())
            image_ids.extend(batch_ids)
            labels.extend(torch.as_tensor(batch_labels).tolist())
    if not features:
        raise ValueError("Cannot extract features from an empty dataset")
    return np.concatenate(features), image_ids, np.asarray(labels, dtype=np.int64)


def load_feature_cache(path, identity):
    """Fail on stale or misordered cache data instead of silently reusing it."""
    with np.load(path, allow_pickle=False) as cached:
        if json.loads(str(cached["identity"].item())) != identity:
            raise ValueError(f"Feature cache provenance mismatch: {path}")
        features = cached["features"]
        image_ids = cached["image_ids"].tolist()
        labels = cached["labels"]
    if image_ids != identity["image_ids"] or labels.tolist() != identity["labels"]:
        raise ValueError(f"Feature cache row order/labels mismatch: {path}")
    _validate_feature_rows(features, labels, image_ids)
    return features, image_ids, labels


def _validate_feature_rows(features, labels, image_ids):
    if (
        not isinstance(features, np.ndarray)
        or features.ndim != 2
        or len(features) == 0
        or features.shape[1] == 0
        or not np.issubdtype(features.dtype, np.number)
        or np.iscomplexobj(features)
        or len(features) != len(image_ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid feature matrix: expected finite real rows matching image IDs")
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or len(labels) != len(image_ids)
    ):
        raise ValueError("Invalid feature labels: expected one integer label per image ID")
    if any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise ValueError("Feature image IDs must be non-empty strings")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Feature image IDs must be unique")


def fit_and_evaluate(features, labels, image_ids, manifest, output_dir):
    """Select C on validation, refit on full training, then evaluate test once."""
    output_dir = _outside_checkout(output_dir)
    _validate_feature_rows(features, labels, image_ids)
    record_by_id = {record["image_id"]: record for record in manifest["records"]}
    if len(record_by_id) != len(manifest["records"]):
        raise ValueError("Protocol contains duplicate image IDs")
    if set(image_ids) != set(record_by_id):
        raise ValueError("Features and protocol must contain the same unique image IDs")
    if any(int(label) != record_by_id[image_id]["label"] for image_id, label in zip(image_ids, labels)):
        raise ValueError("Feature labels do not match the protocol")
    roles = np.asarray([record_by_id[image_id]["role"] for image_id in image_ids])
    fit = roles == "train_fit"
    validation = roles == "validation"
    test = roles == "test"
    train_full = fit | validation
    if not all(mask.any() for mask in (fit, validation, test)) or not (train_full | test).all():
        raise ValueError("Missing or unknown protocol roles")
    class_names = manifest["class_names"]
    if len(class_names) < 2 or len(class_names) != len(set(class_names)):
        raise ValueError("Protocol class names must be distinct and contain at least two classes")
    class_ids = np.arange(len(class_names))
    if any(set(labels[mask]) != set(class_ids) for mask in (fit, validation, test)):
        raise ValueError("Every protocol role must contain exactly the declared classes")
    scores = []
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        for c_value in C_VALUES:
            candidate = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000))
            candidate.fit(features[fit], labels[fit])
            prediction = candidate.predict(features[validation])
            scores.append(
                {
                    "C": c_value,
                    "validation_macro_accuracy": float(balanced_accuracy_score(labels[validation], prediction)),
                }
            )
        # C_VALUES is ordered, and the secondary key explicitly resolves ties.
        selected = min(scores, key=lambda item: (-item["validation_macro_accuracy"], item["C"]))
        write_json(output_dir / "validation_scores.json", {"candidates": scores, "selected_C": selected["C"]})
        final_model = make_pipeline(
            StandardScaler(), LogisticRegression(C=selected["C"], solver="lbfgs", max_iter=2000)
        )
        final_model.fit(features[train_full], labels[train_full])
        prediction = final_model.predict(features[test])
    matrix = confusion_matrix(labels[test], prediction, labels=class_ids)
    metrics = {
        "baseline": "ImageNet-supervised pretrained ResNet-50 + linear classifier",
        "input": "whole RGB image (no mask)",
        "metric_scale": "fraction in [0, 1]",
        "selected_C": selected["C"],
        "test_micro_top1": float(accuracy_score(labels[test], prediction)),
        "test_macro_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
        "test_macro_f1": float(f1_score(labels[test], prediction, labels=class_ids, average="macro", zero_division=0)),
        "counts": {role: int(np.sum(roles == role)) for role in ("train_fit", "validation", "test")},
        "train_full_count": int(train_full.sum()),
        "test_per_class": {name: int(matrix[index].sum()) for index, name in enumerate(class_names)},
    }
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "true_label", "predicted_label", "true_class", "predicted_class"))
        test_ids = np.asarray(image_ids)[test]
        for image_id, label, predicted in zip(test_ids, labels[test], prediction):
            writer.writerow(
                (image_id, int(label), int(predicted), class_names[int(label)], class_names[int(predicted)])
            )
    with (output_dir / "confusion_matrix.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("true/predicted", *class_names))
        writer.writerows((name, *row) for name, row in zip(class_names, matrix))
    import joblib

    with (output_dir / "linear_classifier.joblib").open("xb") as stream:
        joblib.dump(final_model, stream)
    return metrics


def _outside_checkout(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError(f"Data and results must be outside the repository: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("indoor67", "fmd"), required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-cache", type=Path, help="Optional external .npz cache; provenance must match")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.data_root = _outside_checkout(args.data_root)
    args.output_dir = _outside_checkout(args.output_dir)
    # Resolve descendants too, so an existing cache symlink cannot redirect
    # this experiment's writes back into the source checkout.
    download_dir = _outside_checkout(args.data_root / "downloads")
    processed_root = _outside_checkout(args.data_root / "processed")
    model_dir = _outside_checkout(args.data_root / "models")
    _outside_checkout(model_dir / "checkpoints")
    if args.feature_cache:
        args.feature_cache = _outside_checkout(args.feature_cache)
        if args.feature_cache.suffix != ".npz" or args.feature_cache.is_dir():
            parser.error("feature-cache must name an external .npz file")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("output-dir must be empty; use a new directory for each experiment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = {"status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(args.output_dir / "status.json", status)
    try:
        from benchmarks.dataset_protocols import build_dataset_protocol
        from stable_datasets import images
        from stable_datasets.utils import download

        builder = getattr(images, {"indoor67": "Indoor67", "fmd": "FMD"}[args.dataset])
        torch.set_num_threads(4)
        torch.hub.set_dir(str(model_dir))
        code = code_fingerprint()
        # Hash raw source files even when a processed cache already exists.
        # Missing raw provenance is not inferred from the cache directory name.
        source_fingerprints, source_files = {}, {}
        for key, asset in builder.SOURCE.assets.items():
            path = _outside_checkout(download(asset, dest_folder=download_dir, progress_bar=False))
            source_fingerprints[key] = sha256_file(path)
            if asset.checksum is not None and asset.checksum != "sha256:" + source_fingerprints[key]:
                raise ValueError(f"Source checksum mismatch for {key}: {path}")
            source_files[key] = {"url": asset.url, "path": str(path), "bytes": path.stat().st_size}
        # The public builder cache key does not incorporate raw-file fingerprints.
        # Isolate this experiment's processed cache by source and implementation.
        processed_identity = {"dataset": args.dataset, "sources": source_fingerprints, "code": code}
        processed_key = hashlib.sha256(json.dumps(processed_identity, sort_keys=True).encode()).hexdigest()
        processed_cache_dir = _outside_checkout(processed_root / processed_key)
        splits = builder(split=None, download_dir=download_dir, processed_cache_dir=processed_cache_dir)
        protocol = build_dataset_protocol(
            args.dataset, splits, split_seed=args.split_seed, source_fingerprints=source_fingerprints
        )
        write_json(args.output_dir / "manifest.json", protocol.manifest)
        transform = WEIGHTS.transforms()
        model = resnet50(weights=WEIGHTS)
        weight_path = _outside_checkout(
            Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(WEIGHTS.url).path).name
        )
        model_fingerprint = {"weights": str(WEIGHTS), "url": WEIGHTS.url, "sha256": sha256_file(weight_path)}
        model.fc = torch.nn.Identity()
        model.to(args.device)
        dataset = FeatureImages([protocol.train_full, protocol.test], transform)
        ordered_ids, ordered_labels = [], []
        for source in (protocol.train_full, protocol.test):
            for row in source.with_format("raw").with_transform(None):
                ordered_ids.append(row["image_id"])
                ordered_labels.append(int(row["label"]))
        env = environment()
        identity = {
            "cache_format_version": 1,
            "dataset": args.dataset,
            "source_fingerprints": source_fingerprints,
            "model": model_fingerprint,
            "transform": repr(transform),
            "torchvision": env["packages"]["torchvision"],
            "feature_runtime": {name: env["packages"][name] for name in ("torch", "torchvision", "numpy", "pillow")},
            "protocol_version": protocol.manifest["protocol_version"],
            "split_seed": args.split_seed,
            "class_names": protocol.manifest["class_names"],
            "code": code,
            "image_ids": ordered_ids,
            "labels": ordered_labels,
        }
        params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        params.update(
            {
                "C_candidates": C_VALUES,
                "solver": "lbfgs",
                "max_iter": 2000,
                "weights": model_fingerprint,
                "transform": repr(transform),
                "processed_cache_dir": str(processed_cache_dir),
            }
        )
        write_json(args.output_dir / "parameters.json", params)
        write_json(args.output_dir / "environment.json", env)
        write_json(args.output_dir / "code_fingerprint.json", code)
        write_json(args.output_dir / "source_files.json", source_files)
        if args.feature_cache and args.feature_cache.exists():
            features, image_ids, labels = load_feature_cache(args.feature_cache, identity)
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            features, image_ids, labels = extract_features(model, loader, args.device)
            if image_ids != ordered_ids or labels.tolist() != ordered_labels:
                raise ValueError("Extracted feature order differs from the frozen protocol")
            if args.feature_cache:
                args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
                with args.feature_cache.open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        features=features,
                        image_ids=np.asarray(image_ids),
                        labels=labels,
                        identity=json.dumps(identity, sort_keys=True),
                    )
        metrics = fit_and_evaluate(features, labels, image_ids, protocol.manifest, args.output_dir)
        status["status"] = "completed"
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        status["elapsed_seconds"] = time.perf_counter() - start
        write_json(args.output_dir / "status.json", status, overwrite=True)


if __name__ == "__main__":
    main()
