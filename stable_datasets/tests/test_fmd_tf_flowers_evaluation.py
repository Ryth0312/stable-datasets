"""Offline evaluation tests: native data, fake frozen network, real linear fits."""

import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pyarrow as pa
import pytest
import torch
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from examples import evaluate_image_classification as evaluation
from stable_datasets import images, utils
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Value


FMD_NAMES = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
SCALER_FITS = []
CLASSIFIER_PREDICTIONS = []


class TrackingScaler(StandardScaler):
    def fit(self, x, y=None, sample_weight=None):
        self.fitted_rows = np.array(x, copy=True)
        SCALER_FITS.append(self)
        return super().fit(x, y, sample_weight=sample_weight)


class TrackingClassifier(LogisticRegression):
    def predict(self, x):
        CLASSIFIER_PREDICTIONS.append((self.C, np.array(x, copy=True)))
        return super().predict(x)


class TinyTransform:
    def __call__(self, image):
        assert isinstance(image, PILImage.Image) and image.mode == "RGB"
        return torch.from_numpy(np.array(image).transpose(2, 0, 1).copy()).float() / 255

    def __repr__(self):
        return "TinyTransform(RGB, uint8/255, CHW)"


class TinyBNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.fc = torch.nn.Identity()
        self.forward_calls = 0

    def forward(self, images):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert not any(parameter.requires_grad for parameter in self.parameters())
        assert images.ndim == 4 and images.shape[1] == 3
        self.forward_calls += 1
        return self.fc(self.bn(images).mean(dim=(2, 3)))


def _native_fmd():
    features = Features(
        {"image_id": Value("string"), "label": ClassLabel(names=FMD_NAMES), "image": Image(), "mask": Image()}
    )
    rows = []
    for label, name in enumerate(FMD_NAMES):
        buffer = io.BytesIO()
        PILImage.new("RGB", (3, 2), (label * 23, (label % 3) * 80, (label // 3) * 70)).save(buffer, format="PNG")
        for index in range(100):
            rows.append(
                {
                    "image_id": f"{name}/{index:03}.jpg",
                    "label": label,
                    "image": buffer.getvalue(),
                    # Feature extraction must not even decode this annotation.
                    "mask": b"not an image; mask must remain unused",
                }
            )
    return StableDataset(
        features=features,
        info=DatasetInfo(features=features),
        table=pa.Table.from_pylist(rows, schema=features.to_arrow_schema()),
    )


@pytest.fixture(autouse=True)
def forbid_weight_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline tests must never download model weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)


def test_native_images_ignore_masks_and_transforms_and_freeze_bn():
    def forbidden_transform(sample):
        pytest.fail("A caller transform must not replace the frozen preprocessing")

    source = _native_fmd().select([0, 101, 202]).with_transform(forbidden_transform).with_format("torch")
    dataset = evaluation.FeatureImages([source], TinyTransform())
    model = TinyBNModel()
    before = {name: value.clone() for name, value in model.bn.state_dict().items()}
    features, image_ids, labels = evaluation.extract_features(
        model, DataLoader(dataset, batch_size=2, shuffle=False), "cpu"
    )
    assert features.shape == (3, 3)
    assert image_ids == ["fabric/000.jpg", "foliage/001.jpg", "glass/002.jpg"]
    assert labels.tolist() == [0, 1, 2]
    assert model.forward_calls == 2
    for name, value in model.bn.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def _fit_inputs():
    first = np.array([-2, -1, 1, 2, -1.5, 1.5, -10000, 10000], dtype=np.float32)
    features = np.column_stack([first, first * 0.5])
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    roles = ["train_fit"] * 4 + ["validation"] * 2 + ["test"] * 2
    image_ids = [f"image-{index}.jpg" for index in range(len(labels))]
    manifest = {
        "class_names": ["a", "b"],
        "records": [
            {"image_id": image_id, "label": int(label), "role": role}
            for image_id, label, role in zip(image_ids, labels, roles)
        ],
    }
    return features, labels, image_ids, manifest


def test_conflicting_duplicate_records_retain_both_original_labels_in_metrics(tmp_path):
    import csv

    features, labels, image_ids, manifest = _fit_inputs()
    # A synthetic conflict group tests metric semantics without observing the
    # real release's held-out images or moving its actual conflict group.
    features[-1] = features[-2]
    image_ids[-2:] = ["roses/shared.jpg", "tulips/shared.jpg"]
    manifest["class_names"] = ["roses", "tulips"]
    for index in (-2, -1):
        manifest["records"][index].update(image_id=image_ids[index], group_id="synthetic-conflict")
    metrics = evaluation.fit_and_evaluate(features, labels, image_ids, manifest, tmp_path)
    assert metrics["counts"]["test"] == 2
    assert metrics["test_per_class"] == {"roses": 1, "tulips": 1}
    assert metrics["test_micro_top1"] == metrics["test_macro_accuracy"] == 0.5
    with (tmp_path / "predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [row["image_id"] for row in predictions] == image_ids[-2:]
    assert [int(row["true_label"]) for row in predictions] == [0, 1]
    assert predictions[0]["predicted_label"] == predictions[1]["predicted_label"]


def test_selection_scaler_refit_test_isolation_and_ties(monkeypatch, tmp_path):
    SCALER_FITS.clear()
    CLASSIFIER_PREDICTIONS.clear()
    monkeypatch.setattr(evaluation, "StandardScaler", TrackingScaler)
    monkeypatch.setattr(evaluation, "LogisticRegression", TrackingClassifier)
    monkeypatch.setattr(evaluation, "C_VALUES", (10.0, 1.0, 0.1))
    features, labels, image_ids, manifest = _fit_inputs()
    output = tmp_path / "first"
    output.mkdir()
    metrics = evaluation.fit_and_evaluate(features, labels, image_ids, manifest, output)
    assert metrics["selected_C"] == 0.1
    assert metrics["counts"] == {"train_fit": 4, "validation": 2, "test": 2}
    assert metrics["train_full_count"] == 6
    assert len(SCALER_FITS) == len({id(scaler) for scaler in SCALER_FITS}) == 4
    for scaler in SCALER_FITS[:3]:
        np.testing.assert_array_equal(scaler.fitted_rows, features[:4])
    np.testing.assert_array_equal(SCALER_FITS[-1].fitted_rows, features[:6])
    assert [c_value for c_value, _ in CLASSIFIER_PREDICTIONS] == [10.0, 1.0, 0.1, 0.1]
    for index, (_, rows) in enumerate(CLASSIFIER_PREDICTIONS[:3]):
        np.testing.assert_array_equal(rows, SCALER_FITS[index].transform(features[4:6]))
    np.testing.assert_array_equal(CLASSIFIER_PREDICTIONS[-1][1], SCALER_FITS[-1].transform(features[6:]))
    saved = joblib.load(output / "linear_classifier.joblib")
    assert saved.steps[0][1].n_samples_seen_ == 6

    # Changing only held-out labels changes the reported test score, not C.
    changed = labels.copy()
    changed[-2:] = changed[-2:][::-1]
    modified_manifest = copy.deepcopy(manifest)
    for record, label in zip(modified_manifest["records"], changed):
        record["label"] = int(label)
    second = tmp_path / "second"
    second.mkdir()
    second_metrics = evaluation.fit_and_evaluate(features, changed, image_ids, modified_manifest, second)
    assert second_metrics["selected_C"] == metrics["selected_C"]
    assert second_metrics["test_micro_top1"] != metrics["test_micro_top1"]
    assert (second / "validation_scores.json").read_bytes() == (output / "validation_scores.json").read_bytes()


@pytest.mark.parametrize(
    "corruption", ["short_labels", "nan_features", "duplicate_ids", "duplicate_manifest", "bad_role"]
)
def test_fit_rejects_inconsistent_inputs(tmp_path, corruption):
    features, labels, image_ids, manifest = _fit_inputs()
    if corruption == "short_labels":
        labels = labels[:-1]
    elif corruption == "nan_features":
        features[0, 0] = np.nan
    elif corruption == "duplicate_ids":
        image_ids[0] = image_ids[1]
    elif corruption == "duplicate_manifest":
        manifest["records"].append(dict(manifest["records"][0]))
    else:
        manifest["records"][0]["role"] = "train"
    with pytest.raises(ValueError):
        evaluation.fit_and_evaluate(features, labels, image_ids, manifest, tmp_path)
    assert not list(tmp_path.iterdir())


def _save_features(path, identity, *, features=None, image_ids=None, labels=None):
    np.savez_compressed(
        path,
        features=np.ones((2, 3), dtype=np.float32) if features is None else features,
        image_ids=np.asarray(identity["image_ids"] if image_ids is None else image_ids),
        labels=np.asarray(identity["labels"] if labels is None else labels),
        identity=json.dumps(identity, sort_keys=True),
    )


@pytest.fixture
def cache_identity():
    return {
        "source_fingerprints": {"archive": "a" * 64},
        "model": {"sha256": "b" * 64},
        "transform": "RGB",
        "code": {"diff_sha256": "c" * 64},
        "image_ids": ["a.jpg", "b.jpg"],
        "labels": [0, 1],
    }


@pytest.mark.parametrize("key", ["source_fingerprints", "model", "transform", "code", "image_ids", "labels"])
def test_feature_cache_rejects_stale_provenance(tmp_path, cache_identity, key):
    path = tmp_path / "features.npz"
    _save_features(path, cache_identity)
    changed = copy.deepcopy(cache_identity)
    changed[key] = "changed"
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluation.load_feature_cache(path, changed)


@pytest.mark.parametrize("corruption", ["row_order", "labels", "float_labels", "nan", "length", "empty_features"])
def test_feature_cache_rejects_corrupt_rows(tmp_path, cache_identity, corruption):
    kwargs = {}
    if corruption == "row_order":
        kwargs["image_ids"] = ["b.jpg", "a.jpg"]
    elif corruption == "labels":
        kwargs["labels"] = [1, 0]
    elif corruption == "float_labels":
        kwargs["labels"] = np.array([0.0, 1.0])
    elif corruption == "nan":
        kwargs["features"] = np.full((2, 3), np.nan)
    elif corruption == "length":
        kwargs["features"] = np.ones((3, 3))
    else:
        kwargs["features"] = np.empty((2, 0))
    path = tmp_path / "features.npz"
    _save_features(path, cache_identity, **kwargs)
    with pytest.raises(ValueError):
        evaluation.load_feature_cache(path, cache_identity)


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    source_file = tmp_path / "fixture-release.bin"
    source_file.write_bytes(b"fixture release, never a real dataset")
    source = _native_fmd()
    builder_calls, models = [], []

    class FakeBuilder:
        SOURCE = DatasetSource(
            homepage="https://example.invalid/fixture",
            citation="Local test fixture",
            assets={"archive": DownloadInfo(url="https://example.invalid/fixture.zip")},
        )

        def __new__(cls, **kwargs):
            builder_calls.append(kwargs)
            return {"train": source}

    class FakeWeights:
        url = "https://example.invalid/fake-weights.pt"

        def transforms(self):
            return TinyTransform()

        def __str__(self):
            return "OfflineFakeWeights"

    weights = FakeWeights()

    def fake_resnet(*, weights):
        assert weights is evaluation.WEIGHTS
        checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        target = checkpoints / "fake-weights.pt"
        if not target.exists():
            target.write_bytes(b"offline fixture weights")
        model = TinyBNModel()
        models.append(model)
        return model

    monkeypatch.setattr(images, "FMD", FakeBuilder, raising=False)
    monkeypatch.setattr(images, "TFFlowers", FakeBuilder, raising=False)
    monkeypatch.setattr(utils, "download", lambda *args, **kwargs: source_file)
    monkeypatch.setattr(evaluation, "WEIGHTS", weights)
    monkeypatch.setattr(evaluation, "resnet50", fake_resnet)
    monkeypatch.setattr(evaluation, "code_fingerprint", lambda: {"head": "fixture", "diff_sha256": "d" * 64})
    original_hub_dir = torch.hub.get_dir()
    original_threads = torch.get_num_threads()
    args = ["--dataset", "fmd", "--data-root", str(tmp_path / "data"), "--batch-size", "128"]
    yield SimpleNamespace(
        args=args,
        source_file=source_file,
        builder=FakeBuilder,
        builder_calls=builder_calls,
        models=models,
        root=tmp_path,
    )
    torch.hub.set_dir(original_hub_dir)
    torch.set_num_threads(original_threads)


def test_main_native_fmd_fake_model_end_to_end_and_feature_cache(fake_run, monkeypatch):
    cache = fake_run.root / "features.npz"
    output = fake_run.root / "cold"
    evaluation.main(fake_run.args + ["--output-dir", str(output), "--feature-cache", str(cache)])
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed"
    assert manifest["counts"] == {"train_fit": 400, "validation": 100, "train_full": 500, "test": 500}
    assert metrics["counts"] == {"train_fit": 400, "validation": 100, "test": 500}
    assert metrics["test_per_class"] == dict.fromkeys(FMD_NAMES, 50)
    assert len((output / "predictions.csv").read_text().splitlines()) == 501
    assert len((output / "confusion_matrix.csv").read_text().splitlines()) == 11
    assert fake_run.models[0].forward_calls == 8
    torch.testing.assert_close(fake_run.models[0].bn.running_mean, torch.zeros(3), rtol=0, atol=0)
    assert fake_run.models[0].bn.num_batches_tracked.item() == 0
    assert joblib.load(output / "linear_classifier.joblib").steps[0][1].n_samples_seen_ == 500
    cache_before = cache.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("A matching feature cache must avoid feature extraction")

    monkeypatch.setattr(evaluation, "extract_features", forbidden)
    second = fake_run.root / "warm"
    evaluation.main(fake_run.args + ["--output-dir", str(second), "--feature-cache", str(cache)])
    assert cache.read_bytes() == cache_before
    assert fake_run.builder_calls[0]["processed_cache_dir"] == fake_run.builder_calls[1]["processed_cache_dir"]
    assert json.loads((second / "metrics.json").read_text()) == metrics
    assert json.loads((second / "code_fingerprint.json").read_text())["head"] == "fixture"

    fake_run.source_file.write_bytes(b"changed source release")
    failed = fake_run.root / "stale"
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluation.main(fake_run.args + ["--output-dir", str(failed), "--feature-cache", str(cache)])
    assert fake_run.builder_calls[-1]["processed_cache_dir"] != fake_run.builder_calls[0]["processed_cache_dir"]
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"
    assert cache.read_bytes() == cache_before


@pytest.mark.parametrize("dataset, unused_class", [("tf_flowers", "FMD"), ("fmd", "TFFlowers")])
def test_main_does_not_require_the_other_dataset_builder(fake_run, monkeypatch, dataset, unused_class):
    from stable_datasets import images, utils

    monkeypatch.delattr(images, unused_class)

    def stop_before_download(*args, **kwargs):
        raise RuntimeError("selected builder reached its download entry")

    monkeypatch.setattr(utils, "download", stop_before_download)
    args = list(fake_run.args)
    args[args.index("--dataset") + 1] = dataset
    output = fake_run.root / "selected-builder-only"
    with pytest.raises(RuntimeError, match="selected builder reached its download entry"):
        evaluation.main(args + ["--output-dir", str(output)])
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not fake_run.builder_calls and not fake_run.models


def test_main_rechecks_checksum_for_already_downloaded_source(fake_run):
    asset = fake_run.builder.SOURCE.assets["archive"]
    asset.checksum = "sha256:" + hashlib.sha256(fake_run.source_file.read_bytes()).hexdigest()
    fake_run.source_file.write_bytes(b"modified cached file")
    output = fake_run.root / "wrong-checksum"
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        evaluation.main(fake_run.args + ["--output-dir", str(output)])
    assert not fake_run.builder_calls and not fake_run.models
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_main_refuses_existing_output_without_changing_it(fake_run):
    output = fake_run.root / "existing"
    output.mkdir()
    marker = output / "status.json"
    marker.write_bytes(b"user data")
    with pytest.raises(SystemExit) as exc:
        evaluation.main(fake_run.args + ["--output-dir", str(output)])
    assert exc.value.code == 2
    assert marker.read_bytes() == b"user data"
    assert not fake_run.builder_calls and not fake_run.models


@pytest.mark.parametrize("option", ["--data-root", "--output-dir", "--feature-cache"])
def test_main_refuses_paths_inside_checkout(fake_run, option):
    inside = Path(evaluation.__file__).resolve().parents[1] / "uncreated-results.npz"
    args = fake_run.args + ["--output-dir", str(fake_run.root / "results"), option, str(inside)]
    with pytest.raises(ValueError, match="outside the repository"):
        evaluation.main(args)
    assert not fake_run.builder_calls and not fake_run.models


def test_result_helpers_do_not_overwrite_existing_files(tmp_path):
    target = tmp_path / "record.json"
    evaluation.write_json(target, {"original": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.write_json(target, {"original": False})
    assert target.read_bytes() == before


def _code_files(root):
    paths = ["examples/evaluate_image_classification.py", "stable_datasets/images/tf_flowers.py"]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {name}\n", encoding="utf-8")
    return {name: evaluation.sha256_file(root / name) for name in paths}


def test_code_fingerprint_tracks_relevant_untracked_bytes(monkeypatch, tmp_path):
    files = _code_files(tmp_path)

    def fake_git(command, **kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[1] == "diff":
            return b"tracked diff"
        assert command[1] == "ls-files"
        return ("\0".join(files) + "\0").encode()

    monkeypatch.setattr(evaluation.subprocess, "check_output", fake_git)
    first = evaluation.code_fingerprint(tmp_path)
    assert first["implementation_sha256"] == first["untracked_source_sha256"] == files
    (tmp_path / "stable_datasets/images/tf_flowers.py").write_text("# changed implementation\n", encoding="utf-8")
    second = evaluation.code_fingerprint(tmp_path)
    assert first["head"] == second["head"]
    assert first["diff_sha256"] != second["diff_sha256"]


@pytest.mark.parametrize("corruption", [None, "hash", "unsafe_path", "missing", "omitted"])
def test_export_snapshot_is_verified_without_git(monkeypatch, tmp_path, corruption):
    files = _code_files(tmp_path)
    snapshot = {"head": "a" * 40, "diff_sha256": "b" * 64, "files": files, "description": "test snapshot"}
    if corruption == "hash":
        files["stable_datasets/images/tf_flowers.py"] = "0" * 64
    elif corruption == "unsafe_path":
        files["../outside.py"] = "0" * 64
    elif corruption == "missing":
        files["missing.py"] = "0" * 64
    elif corruption == "omitted":
        del files["stable_datasets/images/tf_flowers.py"]
    (tmp_path / "CODE_SNAPSHOT.json").write_text(json.dumps(snapshot), encoding="utf-8")

    def no_git(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(evaluation.subprocess, "check_output", no_git)
    if corruption is not None:
        with pytest.raises(ValueError, match="snapshot"):
            evaluation.code_fingerprint(tmp_path)
    else:
        result = evaluation.code_fingerprint(tmp_path)
        assert result["status"] == "verified_export_snapshot"
        assert result["files"] == result["implementation_sha256"] == files
        assert result["head"] == snapshot["head"]
