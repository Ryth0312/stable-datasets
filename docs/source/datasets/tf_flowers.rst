TensorFlow Flowers
==================

.. note::
   The official archive passed source preflight on September 30, 2026:
   its published checksum, complete archive, all 3,670 decoded photographs,
   attribution coverage and exact duplicates were checked. The native builder,
   grouped protocol and source/cache audit have passed offline checks and a
   complete 3,670-record real-data audit. The seed-42 experimental manifest is
   frozen below. The October 1, 2026 Colab run completed experiment A,
   the supervised smoke test and 20 training epochs. Its returned source
   fingerprints, protocol and predictions were independently checked;
   observed results and their limits are reported below.

Source and Scale
----------------

`TensorFlow Flowers <https://www.tensorflow.org/datasets/catalog/tf_flowers>`_
is the five-class photograph collection used by the
`TensorFlow image-loading tutorial
<https://www.tensorflow.org/tutorials/load_data/images>`_. It is a separate
dataset from Oxford Flowers-17 and Flowers-102.

The official HTTPS
`flower_photos.tgz archive
<https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz>`_
contains 3,670 JPEG photographs and ``flower_photos/LICENSE.txt``. The complete
download was 228,813,984 bytes with SHA-256
``4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c``.
Both values match the
`public TFDS checksum record at a fixed commit
<https://github.com/tensorflow/datasets/blob/2dff5ebbf41953c3f8eeb892c6354f98ee4ef564/tensorflow_datasets/datasets/tf_flowers/checksums.tsv>`_.
The checksum record names the original HTTP address; the preflight verified
that the HTTPS download contains the same expected bytes.

The gzip stream was consumed to EOF and the full tar traversed. All photographs
passed verification and complete decoding with truncated-image tolerance
disabled. No unsafe paths, links, duplicate member paths or unexpected files
were found. All released photographs decoded as RGB; observed widths range
from 143 to 1,024 pixels and heights from 159 to 442 pixels.

Labels follow the
`fixed TFDS builder's class order
<https://github.com/tensorflow/datasets/blob/2dff5ebbf41953c3f8eeb892c6354f98ee4ef564/tensorflow_datasets/datasets/tf_flowers/tf_flowers_dataset_builder.py>`_,
which is not alphabetical. Counts below are from the audited archive:

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Label
     - Class
     - Photographs
   * - 0
     - dandelion
     - 898
   * - 1
     - daisy
     - 633
   * - 2
     - tulips
     - 799
   * - 3
     - sunflowers
     - 699
   * - 4
     - roses
     - 641

Native Interface
----------------

The native ``TFFlowers`` interface uses
``supervised_keys=("image", "label")`` and the following sample fields:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Feature / default value
     - Meaning
   * - ``image``
     - ``Image`` / RGB PIL
     - Whole photograph at its released dimensions, explicitly encoded as PNG.
   * - ``label``
     - ``ClassLabel`` / integer
     - Original released category, using the fixed order above.
   * - ``image_id``
     - ``Value("string")``
     - Full category-relative POSIX path ``<class>/<source_filename>.jpg``.
   * - ``attribution``
     - ``Value("string")``
     - Original per-image line from ``LICENSE.txt``, including the photographer
       credit and its original photograph or profile URL.

``split="train"`` is the complete, unsplit 3,670-record collection.
``split=None`` returns a dictionary containing that single collection.
The release provides no fixed official train/validation/test partition;
``train`` here is an API container name. Experimental roles belong to a
separate protocol and do not remove records from the native collection.

The builder retains original IDs, duplicate records and released labels.
It does not crop, resize, normalize, apply EXIF reorientation or encode JPEG
a second time. RGB conversion and PNG storage preserve decoded photo pixels,
rather than the original JPEG byte stream. There is no mask field.
Classification batches use only ``image`` and ``label``; attribution remains
available as metadata.

Loading and Source Audit
------------------------

Use data and output directories outside the checkout. The public source is
the pinned HTTPS archive; no TensorFlow/TFDS installation or private data path
is needed. Load the complete native collection with:

.. code-block:: python

   from stable_datasets.images import TFFlowers

   dataset = TFFlowers(
       split="train",
       download_dir="/your/external/data/downloads",
       processed_cache_dir="/your/external/data/processed",
   )
   sample = dataset[0]
   print(sample["image_id"], sample["label"], sample["attribution"])

The archive remains in the download cache, including its original
``flower_photos/LICENSE.txt``. Its opening license declaration is also exposed
through dataset metadata. Archive members are read without unrestricted
filesystem extraction. Unsafe paths, links, duplicate canonical members,
unknown image categories, missing or conflicting attribution, and corrupt
photographs must fail explicitly. Every cold native build verifies the raw
archive checksum, including an existing download-cache hit.

The source/cache audit entry point uses fresh external outputs:

.. code-block:: bash

   python examples/audit_tf_flowers.py \
     --data-root /your/external/tf-flowers-data \
     --output-dir /your/external/tf-flowers-audit

With no local archive option, it uses the official HTTPS source. A previously
verified downloader cache may be reused with
``--download-dir /your/external/downloads``. An explicitly supplied archive,
for example after upload to Colab, may be imported with
``--archive /your/external/flower_photos.tgz`` and must still pass the same
size and SHA-256 checks. A local archive is optional, not a prerequisite for
public loading.

Real Data and Cache Validation
------------------------------

The executed audit compared all 3,670 original RGB photographs against both
the cold-built and warm-loaded native records. Original mode, dimensions,
decoded pixels and complete attribution lines matched for every record;
all cached images were PNG. It also rebuilt the grouping evidence from all
original images and required exact equality with the bundled compact metadata.
The cold and warm experimental manifests were identical.

This native cold build started with an empty processed cache and reused the
previously downloaded, checksum-verified source archive. It did not transfer
the 228,813,984-byte archive again. A complete public HTTPS download was
verified during the earlier source preflight. Separate tests exercise the
real downloader against controlled HTTP fixtures. These checks do not claim
a second combined public-network download and native-build run.

During the warm audit, the builder's actual module-local ``bulk_download``
reference and raw archive checksum function were blocked, together with
utility download functions, HTTP requests, gzip/tar opening and direct opening
of the raw archive path. All 3,670 records still loaded and passed comparison.

.. list-table:: Observed local CPU audit timings
   :header-rows: 1
   :widths: 75 25

   * - Operation
     - Seconds
   * - Cold native build from the verified download cache
     - 67.007
   * - Complete cold-cache record scan
     - 8.666
   * - Warm load and complete record scan
     - 8.617

These operations perform different work; their timings are observations,
not a speedup benchmark. No model or GPU computation was part of this audit.

Duplicates and Experimental Protocol
------------------------------------

The source audit found three distinct exact-duplicate pairs:

- ``sunflowers/14889392928_9742aed45b_m.jpg`` and
  ``sunflowers/15066430311_fb57fa92b0_m.jpg``.
- ``sunflowers/15069459615_7e0fd61914_n.jpg`` and
  ``sunflowers/15072973261_73e2912ef2_n.jpg``.
- ``roses/15922772266_1167a06620.jpg`` and
  ``tulips/15922772266_1167a06620.jpg``.

All three pairs have identical file bytes and decoded pixels. The first two
pairs have different Flickr IDs. The last pair has the same verified source
photograph ID and two different released labels. Full category-relative IDs
keep these two records distinct even though their basenames are identical.

The agreed TF Flowers protocol preserves all 3,670 records and labels. It
forms connected groups globally, before any class-specific assignment, using
identical source bytes, identical original decoded pixels including mode and
dimensions, or verified references to the same original photograph. A shared
photographer name or profile page alone does not join images into a group.

Each complete group belongs to exactly one fitting, validation or test role.
Cross-class groups stay together. The target is approximately 60% fitting,
20% validation and 20% test, with class balance where feasible. Group integrity
takes priority over exact quotas: no records are discarded, relabeled or split
apart to produce round counts. The final refit pool is fitting plus validation.

The implementation in ``benchmarks/tf_flowers_protocol.py`` uses seed 42 and
protocol version ``sds-tfflowers-group-v1``. FMD keeps its existing
``sds-split-v1`` assignments unchanged. The TF Flowers rule is deterministic
without depending on input row order or Python's randomized hash:

1. Form global connected components from the three evidence types above.
   Original pixel equality includes mode, width, height and decoded bytes,
   before conversion or EXIF changes. A source-photo edge requires an explicit
   Flickr photo URL whose numeric ID agrees with the attribution record;
   filename prefixes and photographer profile URLs create no such edge.
2. For a component, sort its complete category-relative image IDs and compute
   ``group_id = SHA256(J(sorted_image_ids))``. Here ``J`` is UTF-8 JSON produced
   with ``sort_keys=True``, ``separators=(",", ":")`` and ``ensure_ascii=False``.
   All digest comparisons use lowercase hexadecimal strings.
3. In stage ``outer``, assign whole groups to ``train_full`` or ``test`` with
   weights 4:1. Then, using only the chosen training-pool groups, stage
   ``inner`` assigns ``train_fit`` or ``validation`` with weights 3:1. Both
   stages process all classes together, so a cross-class component remains
   one unit with its complete vector of class counts.
4. Within each stage, process groups by descending member count, then by
   ``SHA256(J([version, stage, seed, group_id]))``, then by ``group_id``.
   For each group, select the role that gives the smallest increase in the
   balance objective below. Break exact cost ties by
   ``SHA256(J([version, stage, seed, group_id, role]))``, then by role name.

For a stage, let :math:`N_c` be its input count for class :math:`c`,
:math:`N=\sum_c N_c`, and :math:`q_r` a role's weight divided by the sum of
stage weights. With current assigned class counts :math:`a_{r,c}` and total
:math:`a_r=\sum_c a_{r,c}`, the balance objective is

.. math::

   E = \sum_r \left[
       \sum_{c:N_c>0}\left(\frac{a_{r,c}-q_r N_c}{N_c}\right)^2
       + \left(\frac{a_r-q_r N}{N}\right)^2
   \right].

The greedy choice minimizes :math:`E_{\mathrm{after}}-E_{\mathrm{before}}`
when the complete group's class-count vector is added to one role. Costs use
exact ``fractions.Fraction`` arithmetic and unrounded target ratios. This
defines a reproducible greedy allocation, not a claim of globally optimal
balance. Output sample views and manifest records are ordered by
``(label, image_id)``; manifest groups are ordered by ``group_id``.

The real-data audit froze the following seed-42 manifest before training.
``train_full`` is the union of ``train_fit`` and ``validation``, not a fourth
disjoint role.

.. list-table:: Frozen sample and group counts
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Class / unit
     - Train fit
     - Validation
     - Train full
     - Test
   * - dandelion
     - 538
     - 180
     - 718
     - 180
   * - daisy
     - 380
     - 126
     - 506
     - 127
   * - tulips
     - 479
     - 160
     - 639
     - 160
   * - sunflowers
     - 419
     - 140
     - 559
     - 140
   * - roses
     - 385
     - 128
     - 513
     - 128
   * - All photographs
     - 2,201
     - 734
     - 2,935
     - 735
   * - Whole groups
     - 2,198
     - 734
     - 2,932
     - 735

The collection has 3,667 groups. All three duplicate pairs listed above are
assigned to ``train_fit`` and hence also to ``train_full``. The roses/tulips
conflict group has ID
``5034e7d826d4367ac5bde715979c586f79ec1ada58ba29b9a9bb1e04f0509d71``;
its original labels remain unchanged and no special assignment rule applies.

The outer-stage totals differ from their targets by -1 for ``train_full``
and +1 for ``test``. Within the actual 2,935-record training pool, the exact
inner targets are 2,201.25 and 733.75, giving deviations of -0.25 for
``train_fit`` and +0.25 for ``validation``. The manifest records all per-class
targets and deviations, group evidence, memberships and roles. The local
Windows ``manifest.json`` has SHA-256
``8c73676281916cc033db42f3d55815748a31494d369e0c170e6178f57e5eb104``.
The Colab copy has SHA-256
``720833b1bcf22f0207023e8a087558fdd6753ebf3f1b65bc40f2890bd982187a``.
The difference is exclusively CRLF versus LF line endings: normalizing the
local file to LF produces exactly the Colab bytes. All parsed fields,
memberships, labels and groups are identical. Their shared canonical JSON
digest, using ``J`` above, is
``0d6d39bf5c6d061311294338aafc75283c3ddbfe7df070533428e3f4b541d4ca``.

Public Group Evidence and Reconstruction
----------------------------------------

The same module ships ``BUNDLED_GROUP_METADATA``, a compact, schema-version-1
JSON record bound to the official archive SHA-256. It contains the three
non-singleton components, all six member observations, their grouping evidence,
and digests covering the complete collection. All other audited image IDs
are singleton groups. No photographs or external private audit directory are
needed to read this public metadata or apply the saved grouping rule.

The complete sorted ID list has SHA-256
``5674fe020188d8f60128946c69dab61d7c133ee993329233541b08208bbd7fb0``;
the normalized complete source-evidence list has SHA-256
``594b9e6aaf3912b44d0020617f5be2eec8e54e64c6163ea4160788c5e1e8a272``.
Both use the canonical JSON function ``J`` above. The protocol checks the
full image-ID digest, original class counts and each ID's label directory.
A supplied archive fingerprint must match the pinned release. A fingerprint
is still a caller declaration: the protocol helper does not read or rehash
the archive, and omitted fingerprints remain marked ``not_verified``.

To reconstruct the evidence independently, audit all 3,670 original files,
recording ``image_id``, file SHA-256, original mode, dimensions, native pixel
SHA-256 and parsed attribution URLs. The native pixel fingerprint is SHA-256
of ``mode + NUL + "width,height" + NUL + image.tobytes()``; the text prefix is
UTF-8. No resizing, EXIF reorientation or RGB conversion precedes that hash.
``reconstruct_group_metadata(records, archive_sha256=...)`` normalizes these
observations, rebuilds all transitive components and regenerates the compact
metadata. ``verify_group_metadata(...)`` requires full equality with the
bundled record, so changed or additional relationships outside the known six
members cannot silently pass.

The audit command above performs this reconstruction from its own source
observations and writes ``source_records.json``,
``group_evidence_verification.json`` and the experimental ``manifest.json``.
The latter also records the metadata digest, source-evidence digest, stage
targets, actual counts and deviations. The complete real-data audit passed
this reconstruction and the cold/warm manifest comparison. The verified
compact metadata digest is
``f61b493e58dcb7390a59f16cd724ff048ac8b04ce82aeb1405c13cb5a2f094af``.

Executed Classification Checks
------------------------------

Run ``tfflowers-20261001T005537Z`` completed on October 1, 2026 using
Python 3.13.15 on Linux, an NVIDIA RTX PRO 6000 Blackwell Server Edition,
PyTorch 2.14.1 with CUDA 13.0, torchvision 0.29.1 and scikit-learn 1.9.1.
The stable-pretraining dependency resolved to commit
``ab836bf699a2be712dfcf980a5eb70d36391e876``. These are recorded run versions,
not minimum requirements or a claim that every dependency combination works.

The run used a reviewed working-source ZIP, based on Git commit
``185006f8418b49b9206045b0df071cd2ed1d518f`` plus the then-uncommitted
TF Flowers implementation. The ZIP SHA-256 is
``39996f4ade3cf0a969bc1b180a26294603b9f697b91103adb351cd664809dae8``;
all 277 returned source-file hashes match that ZIP. No implementation commit
is inferred from the base commit. The results ZIP SHA-256 is
``cd11c5e8a2d11de13a1f0c7ebfde8537b30c25988e623bee6b321ccae2f936e6``;
its 102 payload files passed the snapshot size/hash checks. The execution
ledger contains 26 successful commands, including all three experiments and
their before/after frozen-protocol checks.

The six explicitly selected offline test files passed in that Linux runtime:
184 tests passed, two opt-in large tests were deselected, and one PyTorch
deprecation warning was recorded. This is a focused test result, not a claim
that the complete repository suite passed. The separate source/cache audit
verified all 3,670 records again, using an explicitly imported archive with
the published size and checksum. Its cold build took 74.662 seconds, cold
scan 9.589 seconds, and warm load plus scan 9.547 seconds. This run did not
make another public-network archive download.

Experiment A uses frozen ImageNet-supervised ResNet-50
``IMAGENET1K_V2`` features and the weights' whole-RGB transform. StandardScaler
and logistic regression are fitted on ``train_fit`` for selection. Recorded
validation macro accuracies were 0.912976 for C=0.1, 0.915883 for C=1, and
0.918105 for C=10, selecting C=10. The final scaler and classifier are refitted
on all 2,935 ``train_full`` records before one evaluation of the 735-record
test role. Default metrics use every original test record and label.

.. list-table:: Experiment A, independently recomputed from 735 predictions
   :header-rows: 1
   :widths: 65 35

   * - Metric
     - Value
   * - Micro top-1 accuracy
     - 0.9115646259 (670 / 735)
   * - Macro accuracy (mean per-class recall)
     - 0.9090091082
   * - Macro F1
     - 0.9094370240

Every prediction ID, original label, confusion-matrix cell and metric was
checked independently. The C choice was checked against the recorded
validation-score grid and the hashed evaluator implementation. Validation
predictions, feature arrays and fitted model objects are not in the lightweight
result archive; model fitting and validation scores were not recomputed.

Experiment B uses the existing supervised random-initialization
``vit_tiny_patch16_224`` configuration, seed 42 and FP32, using only
``train_fit`` and ``validation``. Its smoke run completed one epoch with three
training and three validation batches at batch size 16. The full run completed
20 epochs at batch size 32: 68 training batches per epoch with ``drop_last``,
1,360 total training steps and 23 validation batches per epoch. Logs show
normal termination at ``max_epochs=20``; the last logged zero-based epoch and
step are 19 and 1,359.

The final ``validate/loss_epoch`` was 0.8473527431. The recorded validation
diagnostics were ``eval/linear_probe_top1_epoch=0.6024305820`` and
``eval/knn_probe_top1=0.6171824932``. These are the existing online probe metric
keys, not test accuracy or the primary supervised head's top-1 accuracy.
No B test-set result is reported.

Charged command wall times were 96.901 seconds for A, 8.478 seconds for B smoke
and 331.489 seconds for B20, totaling 436.868 seconds against the 36,000-second
experiment budget. The final snapshot records 747.369 elapsed session seconds
against the 86,400-second session limit. These include process overhead and
are not kernel-only GPU timings or billing measurements. The recorded run
contains one A evaluation and one run of each B stage.

FMD was not trained or retuned in this run. Its earlier source/result packages,
parameters, predictions and ``sds-split-v1`` membership remain frozen; the
shared-code migration was covered by separate FMD regression checks.

Evaluation Limits
-----------------

Grouping prevents the audited duplicates from crossing experimental roles.
It does not resolve the roses/tulips label conflict. Default metrics retain
the original sample-level labels and all records assigned to the evaluated
role, without special credit or silent exclusions for that group. Perceptual
near duplicates and overlap with pretraining data were not exhaustively
investigated. These are one fixed-seed baseline and execution checks, not
state-of-the-art claims. The returned lightweight evidence was checked without
executing enclosed scripts, loading model objects or rerunning a model.

License and Attribution
-----------------------

The archive's ``LICENSE.txt`` declares
`Creative Commons Attribution 2.0
<https://creativecommons.org/licenses/by/2.0/>`_ for the photographs and lists
their photographers. All 3,670 image paths have exactly one attribution entry.
Of these, 3,661 include an individual Flickr photo URL; nine include the
photographer's name and profile URL. Those nine entries have attribution and
archive-declared license coverage; a photograph-specific URL is not invented
for them. The audit did not independently authenticate every historical
Flickr page or ownership claim.

Retain the original license and per-image photographer credits when using
the photographs. The sample's ``attribution`` field preserves the original
line, including spacing, Unicode and the supplied URL. The image license is
distinct from the TFDS source-code license and the TensorFlow website's
content license. Raw photographs are not committed to this repository.

Dataset reference: TensorFlow, *TensorFlow Flowers (tf_flowers)*,
`dataset catalog <https://www.tensorflow.org/datasets/catalog/tf_flowers>`_.
Credit the individual photographers through the accompanying archive
attribution records.
