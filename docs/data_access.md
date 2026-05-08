# Data access

This repository does **not** redistribute raw imaging data, segmentation
masks, clinical metadata, or pCR labels. To run the pipeline you must
independently obtain the data and produce the masks yourself.

## What you need

1. **Raw multi-timepoint DCE-MRI volumes** for each case, in NIfTI form
   (`.nii.gz`), one volume per acquisition timepoint per case.
2. A **`metadata.csv`** containing at minimum:
   - `pid` — case identifier
   - `pCR` — binary pathological complete response label (0 / 1)
   - and, if you want to use the manuscript's held-out split rule, a `test`
     column with values in `{0, 1, 2}` (1 and 2 = held out; see
     `docs/validation_and_leakage_control.md`).
3. **Predicted tumour masks**, one NIfTI per case, geometrically aligned
   (shape, spacing, orientation, affine) to the corresponding raw DCE
   reference volume. See `docs/upstream_segmentation.md`.

## Suggested directory layout

The expected on-disk layout used by `scripts/01_extract_roi_tcc.py` is shown
in [`examples/expected_directory_layout.md`](../examples/expected_directory_layout.md).
A schema (column names, requiredness, examples) for the metadata CSV is in
[`schemas/metadata_schema.csv`](../schemas/metadata_schema.csv) and a
synthetic example is in
[`examples/metadata_example_synthetic.csv`](../examples/metadata_example_synthetic.csv).

## Original source

The cohort used in our study is the publicly described BreastDCEDL / I-SPY2
breast DCE-MRI corpus. We do not redistribute it. Users must obtain access
through the original distributor and comply with all data-use agreements
that apply to the corpus, including any terms governing redistribution,
de-identification, and acceptable downstream uses.

## Privacy and de-identification

Do not commit any real patient data, real PIDs, or real predictions to a fork
of this repository. The shipped `.gitignore` excludes the file patterns that
typically carry such material (`*.nii.gz`, `*.dcm`, `metadata.csv`,
`xgboost_predictions*.csv`, etc.).
