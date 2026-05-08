# Upstream segmentation

Tumour segmentation is **not** part of this repository. We only require, as
input, a directory of predicted tumour masks that are geometrically aligned
to the raw DCE volumes.

## Why segmentation is out of scope

The manuscript's segmentation step was performed externally with nnU-Net.
nnU-Net has its own training pipeline, dataset configuration, preprocessing,
inference, and weight management, which we do not duplicate or vendor here.
This keeps the public release focused on the post-segmentation analysis
that is the methodological contribution of the paper.

## Recommended approach

We recommend using the official nnU-Net release
(https://github.com/MIC-DKFZ/nnUNet) to produce one predicted mask per
case. Other segmentation tools are equally acceptable as long as the
output masks satisfy the geometric requirements below.

## Mask requirements

For each case, the predicted tumour mask must:

1. Be a NIfTI file (`.nii` or `.nii.gz`), 3D (or 4D with a singleton time
   dimension that we squeeze).
2. Be aligned to the **reference DCE volume** of the same case, i.e. the
   timepoint that `scripts/01_extract_roi_tcc.py` will use as the geometry
   reference. By default that is the post-early timepoint (`post_early`
   column in the metadata) or the second discovered DCE timepoint if
   `post_early` is missing.
3. Match the reference volume in **shape**, **voxel spacing**, **axis
   orientation**, and **affine** within small numerical tolerance.

`scripts/01_extract_roi_tcc.py` performs explicit geometry checks and
**fails** by default on mismatches; it does not silently resample. Nearest-
neighbor resampling is available behind the explicit
`--allow_resample_mask_to_reference` flag.

## Mask filename conventions

The script's `build_mask_index` accepts several filename patterns, including:

- `<pid>.nii.gz`
- `<pid>_mask.nii.gz`
- `<pid>_spy2_vis1_mask.nii.gz`

You can also supply a `--case_map_csv` to map between an external
segmentation `case_id` and the `pid` used in `metadata.csv`.

## Mask values

Masks are interpreted as binary tumour masks via `mask > 0`. Multi-class
labels are reduced to a binary tumour foreground; if your segmentation
emits multiple classes for tumour subregions, please collapse them to a
single "tumour vs background" label before running this pipeline.
