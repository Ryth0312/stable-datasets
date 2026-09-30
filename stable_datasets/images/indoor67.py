"""MIT Indoor67 with membership defined by the official train/test lists."""

import io
import tarfile
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


# Sorted upstream directory names, including the original spellings.
INDOOR67_CLASS_NAMES = [
    "airport_inside",
    "artstudio",
    "auditorium",
    "bakery",
    "bar",
    "bathroom",
    "bedroom",
    "bookstore",
    "bowling",
    "buffet",
    "casino",
    "children_room",
    "church_inside",
    "classroom",
    "cloister",
    "closet",
    "clothingstore",
    "computerroom",
    "concert_hall",
    "corridor",
    "deli",
    "dentaloffice",
    "dining_room",
    "elevator",
    "fastfood_restaurant",
    "florist",
    "gameroom",
    "garage",
    "greenhouse",
    "grocerystore",
    "gym",
    "hairsalon",
    "hospitalroom",
    "inside_bus",
    "inside_subway",
    "jewelleryshop",
    "kindergarden",
    "kitchen",
    "laboratorywet",
    "laundromat",
    "library",
    "livingroom",
    "lobby",
    "locker_room",
    "mall",
    "meeting_room",
    "movietheater",
    "museum",
    "nursery",
    "office",
    "operating_room",
    "pantry",
    "poolinside",
    "prisoncell",
    "restaurant",
    "restaurant_kitchen",
    "shoeshop",
    "stairscase",
    "studiomusic",
    "subway",
    "toystore",
    "trainstation",
    "tv_studio",
    "videostore",
    "waitingroom",
    "warehouse",
    "winecellar",
]
_CLASS_TO_LABEL = {name: index for index, name in enumerate(INDOOR67_CLASS_NAMES)}


def _path_parts(name, *, context):
    """Validate POSIX names before interpreting them; no archive is extracted."""
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 for char in name)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"Indoor67 {context}: unsafe path {name!r}")
    return parts


def _image_id(parts, *, context):
    if not parts or parts[0] not in _CLASS_TO_LABEL:
        raise ValueError(f"Indoor67 {context}: unknown class or archive prefix {'/'.join(parts)!r}")
    if len(parts) != 2 or Path(parts[1]).suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError(f"Indoor67 {context}: expected class/image.jpg, got {'/'.join(parts)!r}")
    return "/".join(parts)


def _read_split_lists(path_map):
    """Read whole lines, preserving filename spaces, case, and class spelling."""
    split_ids = {}
    for split in (Split.TRAIN, Split.TEST):
        path = Path(path_map[f"{split}_list"])
        ids = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            context = f"{path.name}:{line_number}"
            image_id = _image_id(_path_parts(line, context=context), context=context)
            if image_id in ids:
                raise ValueError(f"Indoor67 {context}: duplicate image ID {image_id!r}")
            ids.add(image_id)
        if not ids:
            raise ValueError(f"Indoor67 {path.name}: empty {split} list")
        split_ids[split] = ids
    overlap = split_ids[Split.TRAIN] & split_ids[Split.TEST]
    if overlap:
        raise ValueError(f"Indoor67 train/test lists overlap: {sorted(overlap)[:5]}")
    return split_ids


class Indoor67(BaseDatasetBuilder):
    """Load only images named in MIT's official training and test manifests.

    Archive handling accepts ``Images/class/image.jpg`` or ``class/image.jpg``.
    The original archive was unavailable during development; these layouts have
    been tested with local fixtures and still require full-release validation.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://web.mit.edu/torralba/www/indoor.html",
        citation="""@inproceedings{quattoni2009recognizing,
  title={Recognizing Indoor Scenes},
  author={Quattoni, Ariadna and Torralba, Antonio},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2009}
}""",
        license="Research purposes only; see the official dataset homepage.",
        assets={
            # Official URL returned HTTP 404 on 2026-09-30. No unverified mirror
            # or archive checksum is substituted here.
            "images": DownloadInfo(url="https://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"),
            # These SHA-256 values fingerprint the files retrieved on 2026-09-30;
            # they are not checksums published by the dataset authors.
            "train_list": DownloadInfo(
                url="https://web.mit.edu/torralba/www/TrainImages.txt",
                checksum="sha256:7ec85f88735739d8159061e4bcf6a9765aab8606ab444605b3f8e0641be0260e",
            ),
            "test_list": DownloadInfo(
                url="https://web.mit.edu/torralba/www/TestImages.txt",
                checksum="sha256:906c529a5a0918feec371de5b07a465ddb1f15ebaebc90a33cc321c371133bf1",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "MIT Indoor67: the official 5,360-image training and 1,340-image test lists, "
                "with 67 scene classes. Images outside those lists are excluded. "
                "The retrieved lists have 77-83 training and 17-23 test images per class."
            ),
            features=Features(
                {
                    "image": ImageFeature(encode_format="PNG"),
                    "label": ClassLabel(names=INDOOR67_CLASS_NAMES),
                    "image_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        keys = ["images", "train_list", "test_list"]
        paths = bulk_download([self.SOURCE.assets[key] for key in keys], dest_folder=self._raw_download_dir)
        path_map = dict(zip(keys, paths))
        split_ids = _read_split_lists(path_map)
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={"archive_path": path_map["images"], "image_ids": split_ids[split], "split": split},
            )
            for split in self._candidate_splits()
        ]

    def _generate_examples(self, archive_path, image_ids, split):
        pending = set(image_ids)
        seen = {}
        layout = None
        # A single sequential pass per split avoids reopening a multi-GB archive
        # for each image. The base builder currently constructs caches per split.
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                context = f"archive member {member.name!r}"
                name = member.name
                if name.startswith("./"):
                    name = name[2:]
                if member.isdir():
                    name = name.removesuffix("/")
                parts = _path_parts(name, context=context)
                wrapped = parts[0] == "Images"
                if wrapped:
                    parts = parts[1:]
                if member.issym() or member.islnk():
                    raise ValueError(f"Indoor67 {context}: archive links are not allowed")
                if member.isdir():
                    if not (wrapped and not parts) and (len(parts) != 1 or parts[0] not in _CLASS_TO_LABEL):
                        raise ValueError(f"Indoor67 {context}: unsupported archive directory")
                    continue
                if not member.isfile():
                    raise ValueError(f"Indoor67 {context}: unsupported archive entry type")
                image_id = _image_id(parts, context=context)
                if image_id in seen:
                    raise ValueError(
                        f"Indoor67 {context}: duplicate/conflicting image ID {image_id!r}; first member {seen[image_id]!r}"
                    )
                seen[image_id] = member.name
                if layout is not None and layout != wrapped:
                    raise ValueError(f"Indoor67 {context}: mixed wrapped and unwrapped archive layouts")
                layout = wrapped
                if image_id not in image_ids:
                    # Extra images are path-validated, but never decoded or added
                    # to either official split.
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Indoor67 {context}: could not read image data")
                try:
                    with stream, PILImage.open(io.BytesIO(stream.read())) as source:
                        image = source.convert("RGB")
                        image.load()
                except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
                    raise ValueError(f"Indoor67 {context}: cannot decode {split} image {image_id!r}: {exc}") from exc
                # Converted PIL images have no source filename/format, so the
                # existing Image feature stores a lossless PNG of decoded pixels.
                pending.remove(image_id)
                yield (
                    image_id,
                    {
                        "image": image,
                        "label": _CLASS_TO_LABEL[parts[0]],
                        "image_id": image_id,
                    },
                )
        if pending:
            raise FileNotFoundError(
                f"Indoor67 {split}: {len(pending)} listed images missing from archive {archive_path}: {sorted(pending)[:5]}"
            )
