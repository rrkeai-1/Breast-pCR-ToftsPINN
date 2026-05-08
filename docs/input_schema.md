# Input schema

This document describes the metadata fields recognised by the pipeline.
The canonical schema is also expressed in machine-readable form in
[`schemas/metadata_schema.csv`](../schemas/metadata_schema.csv).

## metadata.csv

One row per case. Required columns must be supplied; recommended columns
should be supplied to enable specific features; optional columns are
consulted when present and otherwise fall back to sensible defaults.

| column                | required    | type   | meaning                                                       |
|-----------------------|-------------|--------|---------------------------------------------------------------|
| `pid`                 | yes         | str    | case identifier; must match mask filename or `case_map_csv`   |
| `pCR`                 | yes         | 0/1    | binary pathological complete response label                   |
| `test`                | recommended | int    | held-out indicator: `0` = development, `1` or `2` = held out  |
| `clinical_age`        | recommended | float  | baseline age in years                                         |
| `clinical_hr`         | recommended | 0/1    | hormone receptor status (positive = 1)                        |
| `clinical_her2`       | recommended | 0/1    | HER2 status (positive = 1)                                    |
| `clinical_triple_neg` | recommended | 0/1    | triple-negative indicator (1 = TNBC)                          |
| `pre`                 | optional    | int    | index of the pre-contrast DCE timepoint                       |
| `post_early`          | optional    | int    | index of the early post-contrast DCE timepoint                |
| `post_late`           | optional    | int    | index of the late post-contrast DCE timepoint                 |
| `n_times`             | optional    | int    | total number of acquired DCE timepoints                       |
| `dataset` / `split`   | optional    | str    | free-form subset label, useful with `--subset_name`           |

### Notes on clinical fields

The XGBoost step accepts several common synonyms for the four clinical
columns. The script will recognise:

- `clinical_age`, `age`, `age_at_baseline`, `baseline_age`
- `clinical_hr`, `hr`, `hr_status`, `hr_positive`, `hormone_receptor`
- `clinical_her2`, `her2`, `her2_status`, `her2_positive`
- `clinical_triple_neg`, `triple_neg`, `triple_negative`, `tnbc`

For binary clinical fields, both numeric (`0` / `1`) and textual values
(`pos`, `neg`, `positive`, `negative`, `yes`, `no`, etc.) are accepted.

### Notes on the held-out split

The pipeline uses `test in {1, 2}` to identify held-out cases. Cases with
`test == 0` (or with `test` missing) form the development split that is
used for 5-fold CV and for both PINN training and XGBoost training. See
[`validation_and_leakage_control.md`](validation_and_leakage_control.md).

## DCE volume layout

For each case `pid`, the pipeline expects to find one NIfTI per timepoint
under:

```
<full_root>/dce/<pid>/<pid>_*_aqc_<t>.nii.gz
```

where `<t>` is the integer timepoint index (`1`, `2`, ...). The script
discovers timepoints by regex on the `_aqc_<t>.nii(.gz)` suffix. See
[`examples/expected_directory_layout.md`](../examples/expected_directory_layout.md).

## Mask layout

For each case, exactly one predicted tumour mask is required. Filename
conventions accepted out-of-the-box:

- `<pid>.nii.gz`
- `<pid>_mask.nii.gz`
- `<pid>_spy2_vis1_mask.nii.gz`

Or supply a `--case_map_csv` (columns: `pid`, `nnunet_case_id`) to map
arbitrary mask filenames to PIDs.
