# Breast-pCR-ToftsPINN

A post-segmentation reproducibility release of the analysis pipeline used in our
breast DCE-MRI pCR-prediction study. Given a tumour mask that is already
aligned to the raw multi-timepoint DCE volume, this repository runs the full
downstream analysis:

```
predicted tumor mask
  -> core15 / safe-rim ROI partitioning
  -> per-ROI time-concentration curve (TCC) extraction
  -> signal-to-concentration conversion
  -> Tofts-PINN pharmacokinetic feature estimation
  -> volume + clinical + Tofts feature table
  -> XGBoost pCR prediction
  -> 5-fold CV / held-out test evaluation
  -> metric export
```

## Repository scope

This is a **post-segmentation** release. It starts from predicted tumour masks
and performs all of the downstream analysis.

### What is included

- ROI partitioning: tumour mask cleaning, spacing-aware Euclidean distance
  transform, core15 (innermost 15%) and safe-rim (outer rim with an explicit
  uncertain-boundary trim) construction, and volume / volume-ratio computation.
- TCC extraction: per-ROI mean signal across all DCE timepoints, written to a
  long CSV (`tcc_long.csv`) plus per-case structured JSON.
- Tofts-PINN: variable-length DCE curve alignment, signal-to-concentration
  conversion (approximate / SPGR / linearised), population or fixed CSV AIF,
  the classic Tofts forward model, an MLP-based PINN with positivity
  constraints on K_trans and k_ep, masked reconstruction MSE, weak parameter
  regularisation, **fold-wise** PINN training inside CV, and a final
  development-only refit for held-out test inference.
- XGBoost: feature merging (Tofts + volume + clinical), 5-fold stratified CV,
  out-of-fold (OOF) prediction, threshold search on OOF predictions, optional
  refit on the full development split, and held-out test scoring.
- Metrics: AUROC, AUPRC / average precision, accuracy, balanced accuracy,
  precision (PPV), recall (sensitivity), specificity, F1, log loss,
  confusion-matrix counts.

### What is NOT included

- **No nnU-Net code, training pipeline, inference pipeline, or weights.** Users
  must produce predicted tumour masks themselves; see
  [`docs/upstream_segmentation.md`](docs/upstream_segmentation.md).
- **No raw imaging data.** The BreastDCEDL / I-SPY2 dataset is not redistributed
  here; see [`docs/data_access.md`](docs/data_access.md).
- **No real metadata, pCR labels, clinical tables, mask files, or per-case
  predictions.** Only schema templates and synthetic examples are shipped.
- **No model weights** of any kind (nnU-Net checkpoints, PINN checkpoints,
  XGBoost final model, or pretrained image-encoder weights).
- **No supplementary visual-feature branch.** Any visual / RadImageNet / PCA /
  PLS code that existed in our internal research tree has been removed; the
  public release covers the post-segmentation tabular pipeline only.

If this repository is currently a review-stage release, the complete research
tree (including dataset-specific configurations, supplementary experiments, and
fully populated outputs) can be released after publication, or made available
to reviewers and editors on reasonable request.

## External requirements

You will need to obtain or produce, on your own:

1. The raw multi-timepoint DCE-MRI for each case, organised under
   `<full_root>/dce/<pid>/<pid>_*_aqc_<t>.nii.gz`. See
   [`docs/data_access.md`](docs/data_access.md) and
   [`examples/expected_directory_layout.md`](examples/expected_directory_layout.md).
2. A `metadata.csv` with at minimum a `pid` column and a binary `pCR` label
   column. See [`schemas/metadata_schema.csv`](schemas/metadata_schema.csv) and
   [`examples/metadata_example_synthetic.csv`](examples/metadata_example_synthetic.csv).
3. Predicted tumour masks aligned to the raw DCE geometry, one NIfTI per case.
   See [`docs/upstream_segmentation.md`](docs/upstream_segmentation.md).

## Installation

```bash
git clone https://github.com/<your-org>/Breast-pCR-ToftsPINN.git
cd Breast-pCR-ToftsPINN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline is CPU-friendly. PyTorch is used only for the small Tofts-PINN
MLP, which trains comfortably on CPU; a GPU is supported but not required.

## Expected input structure

All paths are passed in via CLI flags or YAML configs and are never hardcoded.
The expected on-disk layout is documented in
[`examples/expected_directory_layout.md`](examples/expected_directory_layout.md);
the input field schema is documented in
[`docs/input_schema.md`](docs/input_schema.md).

## Step A — ROI / TCC extraction

```bash
python scripts/01_extract_roi_tcc.py \
  --full_root      /path/to/BreastDCEDL \
  --metadata_csv   /path/to/BreastDCEDL/metadata.csv \
  --mask_dir       /path/to/predicted_masks \
  --output_dir     /path/to/output_roi_tcc
```

Produces `volume_summary.csv`, `tcc_long.csv`, `dynamics_feature_summary.csv`,
`qc_report.csv`, per-case JSON, and optional QC plots.

## Step B — Tofts-PINN feature extraction

```bash
python scripts/02_extract_tofts_pinn_features.py \
  --tcc_csv            /path/to/output_roi_tcc/tcc_long.csv \
  --metadata_csv       /path/to/BreastDCEDL/metadata.csv \
  --output_dir         /path/to/output_tofts_pinn \
  --concentration_mode auto \
  --aif_mode           population
```

Produces `aligned_tcc_curves.csv`, `concentration_curves.csv`,
`tofts_pinn_features.csv`, `pinn_test_features.csv`, and
`tofts_pinn_training_summary.json`.

This step performs a **single** final-stage PINN refit on the development
split. The fold-wise PINN refits used for leakage-safe CV evaluation are
performed inside step C.

## Step C — XGBoost training and evaluation

```bash
python scripts/03_train_xgboost.py \
  --feature_csv        /path/to/output_roi_tcc/dynamics_feature_summary.csv \
  --volume_csv         /path/to/output_roi_tcc/volume_summary.csv \
  --metadata_csv       /path/to/BreastDCEDL/metadata.csv \
  --tcc_csv            /path/to/output_roi_tcc/tcc_long.csv \
  --output_dir         /path/to/output_xgb \
  --feature_mode       main \
  --concentration_mode auto \
  --aif_mode           population \
  --run_final_test
```

This runs the full leakage-safe protocol:

1. Held-out test split: `test in {1, 2}` (configurable via the `test` column).
2. Stratified 5-fold CV on the development split.
3. Inside each CV fold, the Tofts-PINN is refit on the fold's training PIDs
   only and applied to the fold's validation PIDs.
4. Threshold search on the aggregated OOF predictions.
5. (with `--run_final_test`) A final XGBoost refit on the full development
   split, scored on the held-out test set.

Outputs include `xgboost_cv_summary.json`, `xgboost_metrics.json`,
`xgboost_cv_oof_predictions.csv`, `xgboost_predictions.csv`, and the
final-stage Tofts feature table.

## Step D — Re-evaluate a predictions CSV

```bash
python scripts/04_evaluate_metrics.py \
  --predictions_csv /path/to/output_xgb/xgboost_predictions.csv \
  --output_dir      /path/to/output_xgb \
  --split_col       split_group \
  --split_value     heldout_test
```

Computes the standard metric panel (AUROC, AUPRC, F1, sensitivity,
specificity, log loss, etc.) on the requested rows and writes
`metrics.json`.

## Feature definitions

The 12-feature main preset reported in the manuscript is:

| group    | feature                                  |
|----------|------------------------------------------|
| Tofts    | `core15_Ktrans`, `core15_kep`            |
| Tofts    | `safe_rim_Ktrans`, `safe_rim_kep`        |
| volume   | `tumor_volume_mm3`, `core15_volume_mm3`  |
| volume   | `safe_rim_volume_mm3`                    |
| volume   | `safe_rim_to_core15_volume_ratio`        |
| clinical | `clinical_age`                           |
| clinical | `clinical_hr`, `clinical_her2`           |
| clinical | `clinical_triple_neg`                    |

Full definitions are in
[`docs/feature_definitions.md`](docs/feature_definitions.md).

## Leakage-control design

The development / held-out test split is fixed by the `test` column of
`metadata.csv`: rows with `test in {1, 2}` are held out and never seen by
either the PINN or XGBoost during model selection. Inside the development
split, the PINN is **refit per fold** so that its parameters never touch
validation cases. See
[`docs/validation_and_leakage_control.md`](docs/validation_and_leakage_control.md).

## Output files

A complete reference of every output file is in
[`docs/output_schema.md`](docs/output_schema.md).

## Data and model availability

This repository ships **no real patient data, no real labels, no model
weights, and no real predictions.** The pipeline is designed to be run by
users who have independently obtained the BreastDCEDL / I-SPY2 data and have
generated tumour masks with a segmentation tool of their choice (we used
nnU-Net externally).

For reviewers: see [`docs/reviewer_note.md`](docs/reviewer_note.md).

## Reproducibility caveat

Running this pipeline end-to-end on identical input data does not by itself
reproduce the manuscript's numerical results: full numerical reproduction
also requires the same dataset version, the same predicted masks, the same
clinical labels, and the same train / held-out test partition. We document
the partition rule we used (`test in {1, 2}`) and all PINN / XGBoost
hyperparameters in [`configs/`](configs/) and the script defaults.

## Citation

If you use this pipeline in academic work, please cite the associated
manuscript (citation block to be added on publication).

## License

MIT — see [`LICENSE`](LICENSE). Note that the license covers the source code
only; it does not grant any rights to any third-party dataset, weight, or
artefact that is not redistributed by this repository.
