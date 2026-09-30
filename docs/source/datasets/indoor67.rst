Indoor67
========

.. warning::
   The native builder has passed synthetic offline tests. On 2026-09-30 the
   official image archive returned HTTP 404, so the full release, real cache
   round-trip, training, and evaluation remain unverified. The examples below
   require the official archive to be available; no third-party replacement is
   selected automatically.

Source and Membership
---------------------

`MIT Indoor Scene Recognition <https://web.mit.edu/torralba/www/indoor.html>`_
(Indoor67 / MIT67) describes a full collection of 15,620 JPG images in 67 classes.
This builder exposes only the 6,700 entries in the official
`training list <https://web.mit.edu/torralba/www/TrainImages.txt>`_ and
`test list <https://web.mit.edu/torralba/www/TestImages.txt>`_. Other images in the
archive are excluded; there is no ``all`` or ``extra`` configuration.

The homepage describes 80 training and 20 test images per class. The lists
retrieved on 2026-09-30 actually contain 77--83 training and 17--23 test images
per class, with 100 combined images in every class and totals of 5,360/1,340.
Their exact membership is preserved, including filename spaces, case, and
upstream spellings such as ``stairscase``. There is no official validation split.

The following SHA-256 values fingerprint the retrieved lists; they are not
checksums published by the dataset authors:

.. code-block:: text

   TrainImages.txt  7ec85f88735739d8159061e4bcf6a9765aab8606ab444605b3f8e0641be0260e
   TestImages.txt   906c529a5a0918feec371de5b07a465ddb1f15ebaebc90a33cc321c371133bf1

Fields and Class Mapping
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Default type
     - Meaning
   * - ``image``
     - RGB PIL image
     - Original spatial dimensions; no resizing or training normalization.
   * - ``label``
     - Integer, 0--66
     - Position in the fixed, sorted ``INDOOR67_CLASS_NAMES`` list in the builder.
   * - ``image_id``
     - String
     - Unique POSIX path ``class_name/file name.jpg`` independent of local paths.

Decoded RGB pixels are cached as PNG without another lossy JPEG encoding.
NumPy formatting returns H x W x 3 uint8; torch formatting returns 3 x H x W
float32 divided by 255. Invalid manifests, overlapping split IDs, missing or
corrupt selected images, unsafe archive paths, links, and conflicting IDs raise
errors. Extra images are not silently added to training.

Loading and Cache
-----------------

The following is the implemented API, conditional on availability of the source
archive. Use directories outside the repository:

.. code-block:: python

   from stable_datasets.images import Indoor67

   splits = Indoor67(
       split=None,
       download_dir="/path/outside/repo/data/downloads",
       processed_cache_dir="/path/outside/repo/data/processed",
   )
   train, test = splits["train"], splits["test"]
   sample = train[0]
   class_name = train.features["label"].names[sample["label"]]
   train_torch = train.with_format("torch")

``split="train"`` or ``split="test"`` returns a ``StableDataset``;
``split=None`` returns a ``StableDatasetDict``. A complete processed cache can
be reopened without downloading. Both unwrapped ``class/file.jpg`` and wrapped
``Images/class/file.jpg`` archives are covered by offline fixtures; the real
archive layout still needs verification.

Experimental Protocol
---------------------

The benchmark helper uses the versioned ``sds-split-v1`` SHA-256 ranking,
default ``split_seed=42``, within each official training class. Sixteen images
per class become validation, leaving 61--67 per class for fitting:

- ``train_fit``: 4,288 images.
- ``validation``: 1,072 images, drawn only from official training.
- ``train_full``: all 5,360 official training images for final refitting.
- ``test``: the unchanged 1,340 official test images.

These validation assignments are this contribution's protocol, not new official
splits. Manifests record stable IDs, classes, roles, seed, protocol version, and
source fingerprints. Test membership is independent of the split seed and is
never used to choose classifier settings. Images outside the official lists are
excluded from all stages, including pretraining and refitting.

The reference evaluation entry point is
``python -m examples.evaluate_image_classification --help``. It uses frozen
ImageNet-supervised ResNet-50 V2 features and a validation-selected linear
classifier, followed by training-pool refitting and final test evaluation.
No real Indoor67 result is reported while the archive remains unavailable.

Use and Citation
----------------

The source restricts the images to research purposes only. The repository's code
license does not replace these conditions. Images are not redistributed here.
Archive checksum, real image decoding, and duplicate-image auditing remain
outstanding; no claim about near duplicates or pretraining overlap is made.

Reference: A. Quattoni and A. Torralba, *Recognizing Indoor Scenes*,
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.
