# Reviewer note

This repository is the public, post-segmentation portion of the analysis
code for the associated manuscript. It is intended to make the
methodological contribution of the paper auditable and reproducible.

## Scope of this release

- This release covers, in code, the entire post-segmentation pipeline:
  ROI partitioning (core15 / safe-rim), per-ROI TCC extraction, signal-to-
  concentration conversion, Tofts-PINN parameter estimation, and the
  XGBoost classifier with leakage-safe 5-fold CV and a held-out test.
- This release does **not** include:
  - tumour segmentation (handled externally with nnU-Net),
  - the BreastDCEDL / I-SPY2 dataset,
  - any patient-level imaging, masks, metadata, or labels,
  - any pretrained or fine-tuned model weights.

The omitted items are not part of the methodological contribution of the
paper; they are upstream dependencies that have their own provenance and
licensing terms.

## What full numerical reproduction requires

To reproduce the headline numbers in the manuscript, a reviewer would need:

1. Access to the BreastDCEDL / I-SPY2 cohort under its original distribution
   terms.
2. Predicted tumour masks aligned to the raw DCE volumes (we used nnU-Net
   externally; any equivalent segmentation should yield comparable results
   provided geometric alignment is preserved).
3. The same `metadata.csv`, including the same `pCR` label encoding and the
   same held-out partition (`test in {1, 2}`).
4. The pipeline as configured in [`configs/main_tofts_pinn_xgb.template.yaml`](../configs/main_tofts_pinn_xgb.template.yaml),
   with `--random_state 42`.

We have published the partition rule and all hyperparameters; we have not
published the PIDs that fall in `test == 1` vs `test == 2`, because that
list is patient-level metadata.

## Availability on request

If editors or reviewers require:

- the exact patient-level held-out partition, or
- final-model checkpoints, or
- per-case OOF and held-out predictions,

we are willing to provide them on reasonable request through the journal's
confidential review channels, subject to the data-use agreements that
govern the BreastDCEDL / I-SPY2 cohort.

After acceptance, the broader research code (including supplementary
experiments and dataset-specific configurations) can be released either in
this same repository or as a clearly-versioned successor repository.

## Self-audit checklist

The release was produced under the following constraints, each of which can
be verified by inspection:

- [x] No real patient identifiers in source code, configs, schemas, or examples.
- [x] No real per-case predictions, no real labels, no real clinical tables.
- [x] No model checkpoints of any kind.
- [x] No nnU-Net code, weights, or training pipeline.
- [x] No hardcoded local or server paths; every path is a CLI flag or YAML
      placeholder of the form `/path/to/...`.
- [x] No dependence on a supplementary visual / RadImageNet / PCA / PLS branch
      (none of that code is in this repository).
- [x] `requirements.txt` lists only the dependencies actually used by the
      shipped code.
