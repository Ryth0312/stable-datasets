FMD
===

.. note::
   The native builder preserves the released RGB/L mask modes and decoded
   pixels in lossless PNG storage. The official ZIP has been downloaded and
   all 1,000 image/mask pairs compared pixel-for-pixel with the native cache.
   A warm load also passed while downloads and reopening the ZIP were blocked.
   A subsequent Colab run on September 30, 2026 passed the full cache audit,
   frozen-feature evaluation, supervised smoke run and 20-epoch training run.
   Measured results and their protocol limits are recorded below.

Source and Scale
----------------

The `Flickr Material Database (FMD)
<https://people.csail.mit.edu/lavanya/fmd.html>`_ contains 1,000 images in ten
material categories, with 100 images per category: 50 close-ups and 50 regular
views. Its source page states that the download includes photographs and
region-of-interest masks. The official
`download <https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip>`_ was audited
on September 30, 2026:

- 58,627,619 bytes; 1,000 photos and 1,000 exactly paired masks.
- Ten classes with 100 photos each; all photo/mask pairs are 512 by 384 pixels.
- Photos use ``image/<class>/<filename>.jpg``; the paired mask has the same
  relative name below ``mask/``. There is no published split file in this ZIP.
- No exact duplicate photos were found by file bytes or decoded RGB pixels.
  This does not exclude near duplicates or overlap with pretraining data.
- Three non-image files (two MATLAB source/backup files and ``Thumbs.db``) are
  not samples. No per-image license mapping was found in the archive.

The SHA-256 fingerprint of this downloaded ZIP is
``254475ddb37bf0fe1383ff9ea61eb71042db33af1f9a9aa8747391a72135b8f8``.
This was computed locally, not supplied as an official checksum.

Mask Preservation
-----------------

The released JPEG masks do not satisfy the initially proposed single-channel
binary-mask schema. Of the 1,000 masks, 743 decode as RGB and 257 as L; 653
contain values outside ``{0, 255}``. In 142 RGB masks the channels differ.
All remaining 347 masks are uniformly 255. Converting every mask to grayscale
or thresholding it would change the released annotation pixels.

The builder preserves each mask's original decoded mode, dimensions and pixel
values, explicitly encoding PNG bytes before passing them to the existing
``Image`` feature. It does not grayscale, threshold, merge channels, resize or
encode JPEG a second time. This preserves decoded annotation pixels rather
than the original compressed JPEG bytes. The public image feature and storage
framework are unchanged.

Schema and Collection
---------------------

``FMD`` is a native ``BaseDatasetBuilder`` with ``supervised_keys=("image", "label")``.
Default sample fields are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Feature / default value
     - Meaning
   * - ``image``
     - ``Image`` / RGB PIL
     - Whole photo at its original dimensions, stored as PNG.
   * - ``label``
     - ``ClassLabel`` / integer
     - Zero-based position in the fixed class order below.
   * - ``image_id``
     - ``Value("string")``
     - Unique POSIX path ``<class>/<filename>.jpg`` independent of cache location.
   * - ``mask``
     - ``Image`` / L or RGB PIL
     - Same dimensions as the photo, with original decoded mode and pixels.

The fixed class order is ``fabric``, ``foliage``, ``glass``, ``leather``,
``metal``, ``paper``, ``plastic``, ``stone``, ``water``, ``wood``.

``split="train"`` (the default) returns all 1,000 samples. ``split=None`` returns
a dictionary with that single collection. Here ``train`` is an unsplit API
container, not an official independent training set. The builder has no
validation or test split and does not resize or normalize samples.

NumPy images are H x W x 3 uint8; masks are H x W or H x W x 3 uint8. The
existing torch formatter produces CHW float tensors divided by 255: images
have three channels and masks have one or three. Do not stack mixed-mode
masks with a generic default classification collator. The benchmark's sample
transforms act on the image, and ``collate_single`` returns only ``image`` and
``label``; masks never enter its classification network.

Loading and Cache
-----------------

Use explicit directories outside the checkout:

.. code-block:: python

   from stable_datasets.images import FMD

   options = dict(
       download_dir="/your/external/data/downloads",
       processed_cache_dir="/your/external/data/processed",
   )
   dataset = FMD(split="train", **options)
   sample = dataset[0]
   assert sample["image"].mode == "RGB"
   assert sample["mask"].mode in {"L", "RGB"}
   assert sample["image"].size == sample["mask"].size
   collections = FMD(split=None, **options)
   assert set(collections) == {"train"}

The ZIP is read without extracting members to the filesystem. Exact relative
names pair photos and masks; metadata and unrelated thumbnails are not samples.
Missing or orphaned masks, duplicate/unsafe members, corrupt images,
unsupported mask modes and mismatched dimensions fail explicitly.

The cache stores both image fields as PNG. A subsequent call with the same
options reuses the processed cache. Conversion via ``with_format("numpy")``
or ``with_format("torch")`` changes the returned representation, not the
cached annotation. Raw encoded PNG bytes are available with
``with_format("raw")``.

Experimental Protocol
---------------------

The `authors' historical experiment
<https://people.csail.mit.edu/celiu/CVPR2010/index.html>`_ randomly selected
50 training and 50 test images per category. Close-up versus regular view
does not define that split. No fixed original split manifest has been verified.

The implemented benchmark protocol uses seed 42 and version
``sds-split-v1``. Within each class it ranks IDs by SHA-256 of
``sds-split-v1|fmd|outer|42|<image_id>`` to select 50 train-full and 50 test
samples, then independently ranks train-full with stage ``inner`` to reserve
10 validation samples. The resulting totals are 400 train-fit, 100 validation,
500 train-full for final refit, and 500 test. This is a new reproducible split,
not a recovered original author split. Test labels are not used for model or
hyperparameter selection. Experimental roles stay outside the builder's
sample cache, so the ``train`` collection still contains all 1,000 samples.

The classification baseline uses whole RGB images and ignores masks. It is
not a reproduction of the earlier mask-restricted feature pipeline. The
evaluation entry explicitly selects ImageNet-supervised ResNet50
``IMAGENET1K_V2``, freezes it with its matching preprocessing, selects a linear
classifier's C using validation only, then refits on train-full before final
test evaluation.

Measured FMD Run
----------------

Run ``sds-20260930T202922Z`` completed on September 30, 2026 in a fresh Colab
environment with an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. All
19 recorded commands exited successfully, including 156 offline tests
(two large tests deselected) and a separate full 1,000-pair source/cache audit.
Cold and warm image/mask PNG records matched the source modes, dimensions and
decoded pixels, including the mixed RGB/L masks.

On the seed-42 protocol above, the frozen ResNet-50 V2 linear baseline selected
``C=1`` using validation accuracy (81%, tied with ``C=10``; smaller C wins),
then refitted on all 500 training-pool samples. Its single final test evaluation
achieved 83.4% micro top-1 and 83.4% macro accuracy, with macro F1 of 0.833262.
There were 417 correct predictions among 500 test images, 50 per class.
The saved predictions independently reproduce these metrics and the confusion
matrix. This is one seeded split with pretrained features, not an estimate
over repeated splits or a reproduction of the authors' historical method.

The existing supervised runner also completed a one-epoch smoke run and
20 epochs with ``vit_tiny_patch16_224`` (``pretrained=False``) on 400 fit /
100 validation images.
This verifies the training path. Its online linear-probe and kNN diagnostics
must not be presented as the supervised classifier's final test accuracy;
the B runs do not evaluate the 500-image test partition.

The run used Python 3.13.15, torch 2.14.1, torchvision 0.29.1 and
``stable-pretraining`` commit
``ab836bf699a2be712dfcf980a5eb70d36391e876``.
Its delivered source ZIP fingerprint is
``28a0bd4e0b7fa259f7e2a40bc2de46768b72fa294cb0074f6cfece08efd09340``;
the verified result ZIP fingerprint is
``77946313891e4fbec01e9c671e16f4a25eefe753a219e85c7c3c0c46492c5030``.
These identify the executed artifacts; they are not Git implementation commits.

Use and Citation
----------------

The source page lists Creative Commons terms, image credits, and exceptions
with different licenses. Consult that page and the original image attributions;
this contribution does not assert one uniform dataset license. Per-image
conditions remain unresolved, and this contribution does not redistribute
photographs or masks.

Reference: L. Sharan, R. Rosenholtz, and E. H. Adelson, *Accuracy and speed of
material categorization in real-world images*, Journal of Vision 14(9),
article 12, 2014, `doi:10.1167/14.9.12 <https://doi.org/10.1167/14.9.12>`_.
Issue 9 is supported by the DOI and publication record; some author-page
BibTeX instead lists issue 10.
