FMD (planned)
=============

.. note::
   Work in progress: this page describes proposed support. The FMD loader,
   package export, tests, and benchmark integration are not implemented in this
   contribution. No dataset download, training, or evaluation results are claimed.

Source and Scale
----------------

The `Flickr Material Database (FMD)
<https://people.csail.mit.edu/lavanya/fmd.html>`_ contains 1,000 images in ten
material categories, with 100 images per category: 50 close-ups and 50 regular
views. Its source page states that the download includes photographs and
region-of-interest masks.

Planned Support
---------------

- Add a native ``BaseDatasetBuilder`` with ``image``, ``label``, ``image_id``,
  and aligned single-channel ``mask`` fields, preserving original dimensions.
- Expose all 1,000 samples in one unsplit collection under ``split="train"``;
  ``split=None`` will return a dictionary containing only that collection.
  Here, ``train`` is a planned API container name, not an official training set.
- Keep reproducible, stratified experimental splits in the benchmark layer.
  The `authors' historical experiment
  <https://people.csail.mit.edu/celiu/CVPR2010/index.html>`_ randomly selected
  50 training and 50 test images per category. Close-up versus regular view
  does not define that split. No fixed original split manifest has been verified;
  a newly generated split will not be described as the authors' exact split.
- Retain masks as annotations; the planned classification baseline will use
  whole images. Add offline tests and known-method validation in later work.

These are implementation plans, not an available loading API. Archive layout,
image/mask pairing, mask encoding, and runtime counts remain unverified.

Use and Citation
----------------

The source page lists Creative Commons terms, image credits, and exceptions
with different licenses. Consult that page and the original image attributions;
this draft does not assert one uniform dataset license. Per-image conditions
remain to be audited, and this contribution does not redistribute photographs.

Reference: L. Sharan, R. Rosenholtz, and E. H. Adelson, *Accuracy and speed of
material categorization in real-world images*, Journal of Vision, 2014.
