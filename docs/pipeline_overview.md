# Pipeline overview

```
predicted tumor mask  +  raw multi-timepoint DCE-MRI  +  metadata.csv
                              |
                              v
              [Step 1] scripts/01_extract_roi_tcc.py
              ------------------------------------------------
              - geometry checks (shape / spacing / orientation / affine)
              - mask cleaning (binarize, optional largest component)
              - spacing-aware Euclidean distance transform
              - core15 (innermost 15% by EDT) construction
              - safe-rim construction (outer rim with uncertain-boundary trim)
              - per-ROI mean signal across all timepoints
              - per-ROI volume in mm^3, safe_rim/core15 ratio
              - simple dynamics: time-to-peak, wash-in slope,
                wash-out slope, normalised AUC
              outputs:
                volume_summary.csv
                tcc_long.csv
                dynamics_feature_summary.csv
                qc_report.csv
                case_json/<pid>.json
                              |
                              v
              [Step 2] scripts/02_extract_tofts_pinn_features.py
              ------------------------------------------------
              - variable-length DCE alignment to a fixed time grid
              - signal-to-concentration conversion
                  (auto / approximate / SPGR / linearised)
              - AIF: population (Parker) or fixed CSV
              - shared MLP-based PINN with positivity on K_trans, k_ep
              - masked reconstruction MSE + weak parameter regularisation
              - final-stage refit on the development split only
              outputs:
                aligned_tcc_curves.csv
                concentration_curves.csv
                tofts_pinn_features.csv
                pinn_test_features.csv
                tofts_pinn_training_summary.json
                              |
                              v
              [Step 3] scripts/03_train_xgboost.py
              ------------------------------------------------
              - merge dynamics + volume + clinical
              - apply held-out test split (test in {1, 2})
              - stratified 5-fold CV on development split
              - inside each fold: PINN refit on the fold's training PIDs
                only, infer K_trans / k_ep for the fold's validation PIDs
              - XGBoost with default hyperparameters from the manuscript
              - threshold search on aggregated OOF predictions
              - (optional) final XGBoost refit on the full development split
                + held-out test scoring
              outputs:
                xgboost_cv_summary.json
                xgboost_cv_oof_predictions.csv
                xgboost_metrics.json
                xgboost_predictions.csv
                tofts_pinn_features.csv
                pinn_test_features.csv
                tofts_pinn_training_summary.json
                              |
                              v
              [Step 4] scripts/04_evaluate_metrics.py
              ------------------------------------------------
              - take any predictions CSV, optionally filter by split
              - compute AUROC, AUPRC, accuracy, balanced accuracy,
                precision, recall, specificity, F1, log loss, confusion matrix
              outputs:
                metrics.json
```

## What this pipeline does NOT do

- It does **not** perform tumour segmentation.
- It does **not** perform any preprocessing of the raw DCE volumes other
  than what is needed to align timepoints to a common time grid.
- It does **not** train any deep image-encoder backbone. The only neural
  network in this pipeline is a small shared MLP used by the Tofts-PINN.
- It does **not** save final model checkpoints to disk; the design assumes
  the user will rerun the pipeline to reproduce predictions rather than
  ship a model artefact.
