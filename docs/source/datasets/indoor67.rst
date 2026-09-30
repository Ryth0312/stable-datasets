Indoor67 (planned)
==================

.. note::
   Work in progress: this page describes proposed support. The Indoor67 loader,
   package export, tests, and benchmark integration are not implemented in this
   contribution. No dataset download, training, or evaluation results are claimed.

Source and Scale
----------------

`MIT Indoor Scene Recognition <https://web.mit.edu/torralba/www/indoor.html>`_
(Indoor67 / MIT67) contains 15,620 JPG images across 67 indoor scene categories.
The official evaluation uses a 6,700-image subset: 80 training and 20 test images
per class, specified by the official
`training list <https://web.mit.edu/torralba/www/TrainImages.txt>`_ and
`test list <https://web.mit.edu/torralba/www/TestImages.txt>`_.

Planned Support
---------------

- Add a native ``BaseDatasetBuilder`` with ``image``, ``label``, and ``image_id``
  fields, retaining original image dimensions and stable category mappings.
- Expose only the official ``train`` (5,360) and ``test`` (1,340) subsets.
  No official validation split is provided; images outside these lists will
  not be added to training or exposed through an extra configuration.
- Keep experiment validation within the official training pool, with the
  official test set reserved for final evaluation.
- Add offline tests, documentation, and known-method validation in later work.

These are implementation plans, not an available loading API. Archive contents,
download behavior, cache behavior, and runtime counts remain unverified.

Use and Citation
----------------

The official source restricts the images to research purposes only. The
repository's code license does not replace the dataset's usage conditions.
This contribution does not redistribute images.

Reference: A. Quattoni and A. Torralba, *Recognizing Indoor Scenes*,
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.
