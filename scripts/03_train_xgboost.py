#!/usr/bin/env python3
"""
Step 3: Train and evaluate the XGBoost pCR classifier.

This script implements the leakage-safe evaluation protocol described in the
manuscript:

1. Load the per-case dynamics, volume, metadata, and TCC tables.
2. Merge them into a single per-case feature dataframe.
3. Apply the held-out test rule (``test in {1, 2}``) to split out a test set
   that is never seen during model selection or PINN training.
4. Run stratified 5-fold CV on the development split. In each fold, refit the
   Tofts-PINN on the fold's training PIDs only, infer K_trans / k_ep features
   for the fold's validation PIDs, and train an XGBoost classifier on the
   selected feature preset.
5. Aggregate out-of-fold (OOF) predictions, search a single decision
   threshold on the OOF probabilities, and (optionally) refit on the full
   development split to score the held-out test set.

Outputs (under --output_dir):
- xgboost_cv_summary.json
- xgboost_cv_oof_predictions.csv
- xgboost_metrics.json                       (CV metrics, plus held-out test if --run_final_test)
- xgboost_predictions.csv                    (OOF + held-out test predictions, if --run_final_test)
- tofts_pinn_features.csv                    (final-stage Tofts features, if --feature_mode uses PINN)
- pinn_test_features.csv                     (Tofts features for held-out test only, same condition)
- tofts_pinn_training_summary.json           (final-stage fit summary, same condition)

This script does NOT save model checkpoints; the public release intentionally
omits patient-level model weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running this script directly without installing the project.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from metrics_utils import compute_binary_metrics
from tofts_pinn_utils import (
    HELDOUT_TEST_VALUES,
    add_shared_tofts_pinn_args,
    build_final_tofts_feature_table,
    pinn_config_from_args,
    prepare_curve_records,
    save_curve_artifacts,
    train_pinn_and_infer_for_pid_sets,
)
from xgboost_utils import (
    feature_columns_for_preset,
    fit_final_model_and_evaluate,
    run_stratified_kfold_cv,
    FEATURE_PRESETS,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: 5-fold CV + (optional) held-out test evaluation of an "
            "XGBoost pCR classifier on the Tofts + volume + clinical features."
        )
    )

    # Inputs.
    parser.add_argument("--feature_csv", type=Path, required=True,
                        help="Path to dynamics_feature_summary.csv from step 1.")
    parser.add_argument("--volume_csv", type=Path, required=True,
                        help="Path to volume_summary.csv from step 1.")
    parser.add_argument("--metadata_csv", type=Path, required=True,
                        help="Path to metadata.csv (clinical fields, labels, splits).")
    parser.add_argument("--tcc_csv", type=Path, required=True,
                        help="Path to tcc_long.csv from step 1 (used for fold-wise PINN refit).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for CV summary, metrics, and predictions.")

    # Column names.
    parser.add_argument("--pid_col", type=str, default="pid", help="Case ID column.")
    parser.add_argument("--label_col", type=str, default="pCR", help="Binary label column.")
    parser.add_argument("--test_col", type=str, default="test", help="Held-out test indicator column.")

    # Feature preset.
    parser.add_argument("--feature_mode", type=str, default="main",
                        choices=sorted(FEATURE_PRESETS.keys()),
                        help="Feature preset. 'main' is the 12-feature manuscript preset.")

    # CV / threshold.
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold_metric", type=str, default="f1",
                        choices=["f1", "balanced_accuracy"],
                        help="Metric used to select the decision threshold from OOF predictions.")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    # XGBoost hyperparameters.
    parser.add_argument("--n_estimators", type=int, default=1500)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.75)
    parser.add_argument("--colsample_bylevel", type=float, default=0.90)
    parser.add_argument("--colsample_bynode", type=float, default=0.90)
    parser.add_argument("--min_child_weight", type=float, default=2.0)
    parser.add_argument("--reg_lambda", type=float, default=6.0)
    parser.add_argument("--reg_alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--scale_pos_weight", type=float, default=1.0)
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--n_jobs", type=int, default=1)

    # Held-out test.
    parser.add_argument("--run_final_test", action="store_true",
                        help="If set, refit on the full development split and score the held-out test set.")

    # Tofts-PINN configuration (shared across scripts 02 and 03).
    add_shared_tofts_pinn_args(parser)

    return parser


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_xgboost")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train_xgboost.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _coerce_pid(df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    if pid_col not in df.columns:
        raise KeyError(f"required column {pid_col!r} not found in dataframe.")
    df = df.copy()
    df[pid_col] = df[pid_col].astype(str).str.strip()
    return df


def _coerce_binary_clinical(series: pd.Series) -> pd.Series:
    """Normalize clinical receptor fields to {0, 1, NaN}.

    Recognised positive tokens: "1", "+", "pos", "positive", "yes", "true".
    Recognised negative tokens: "0", "-", "neg", "negative", "no", "false".
    """
    if series is None:
        return series
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip().str.lower()
    pos_tokens = {"1", "1.0", "+", "pos", "positive", "yes", "y", "true", "t"}
    neg_tokens = {"0", "0.0", "-", "neg", "negative", "no", "n", "false", "f"}
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[text.isin(pos_tokens)] = 1.0
    out[text.isin(neg_tokens)] = 0.0
    return out


CLINICAL_COLUMNS = ["clinical_age", "clinical_hr", "clinical_her2", "clinical_triple_neg"]


def _extract_clinical_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort extraction of the four clinical columns from a free-form
    metadata table. Falls back to NaN-filled columns if a field is absent.
    """
    out = metadata_df[["pid"]].copy()
    lower_to_original = {str(c).lower(): str(c) for c in metadata_df.columns}

    def find(*candidates: str) -> str | None:
        for cand in candidates:
            key = cand.lower()
            if key in lower_to_original:
                return lower_to_original[key]
        return None

    age_col = find("clinical_age", "age", "age_at_baseline", "baseline_age")
    out["clinical_age"] = (
        pd.to_numeric(metadata_df[age_col], errors="coerce") if age_col else np.nan
    )

    hr_col = find("clinical_hr", "hr", "hr_status", "hr_positive", "hormone_receptor")
    out["clinical_hr"] = (
        _coerce_binary_clinical(metadata_df[hr_col]) if hr_col else np.nan
    )

    her2_col = find("clinical_her2", "her2", "her2_status", "her2_positive")
    out["clinical_her2"] = (
        _coerce_binary_clinical(metadata_df[her2_col]) if her2_col else np.nan
    )

    tn_col = find("clinical_triple_neg", "triple_neg", "triple_negative", "tnbc")
    out["clinical_triple_neg"] = (
        _coerce_binary_clinical(metadata_df[tn_col]) if tn_col else np.nan
    )
    return out


def _load_and_merge_features(
    feature_csv: Path,
    volume_csv: Path,
    metadata_csv: Path,
    pid_col: str,
    label_col: str,
    test_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Loading dynamics features: %s", feature_csv)
    feat = _coerce_pid(pd.read_csv(feature_csv), pid_col)

    logger.info("Loading volume features: %s", volume_csv)
    vol = _coerce_pid(pd.read_csv(volume_csv), pid_col)

    logger.info("Loading metadata: %s", metadata_csv)
    meta = pd.read_csv(metadata_csv)
    if pid_col not in meta.columns:
        # accept canonical pid synonyms
        for cand in ["pid", "PatientID", "patient_id", "case_id"]:
            if cand in meta.columns and cand != pid_col:
                meta = meta.rename(columns={cand: pid_col})
                break
    meta = _coerce_pid(meta, pid_col)

    # Pull labels and split flags from metadata.
    label_series = (
        pd.to_numeric(meta[label_col], errors="coerce") if label_col in meta.columns else pd.Series(np.nan, index=meta.index)
    )
    test_series = (
        pd.to_numeric(meta[test_col], errors="coerce") if test_col in meta.columns else pd.Series(np.nan, index=meta.index)
    )
    clinical_df = _extract_clinical_columns(meta.rename(columns={pid_col: "pid"}))
    clinical_df = clinical_df.rename(columns={"pid": pid_col})
    clinical_df[label_col] = label_series.to_numpy()
    clinical_df[test_col] = test_series.to_numpy()

    # Merge volume features. Avoid clobbering identical columns from feat.
    vol_cols_to_use = [pid_col] + [
        c for c in [
            "tumor_volume_mm3",
            "core15_volume_mm3",
            "safe_rim_volume_mm3",
            "safe_rim_to_core15_volume_ratio",
        ] if c in vol.columns
    ]
    merged = feat.merge(vol[vol_cols_to_use], on=pid_col, how="left", suffixes=("", "_vol"))
    # If volume columns already existed on feat, prefer the explicit volume table.
    for c in vol_cols_to_use[1:]:
        backup = c + "_vol"
        if backup in merged.columns:
            merged[c] = merged[backup]
            merged = merged.drop(columns=[backup])

    merged = merged.merge(clinical_df, on=pid_col, how="left")
    return merged


def _split_dev_test(
    df: pd.DataFrame,
    label_col: str,
    test_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_col in df.columns:
        test_mask = pd.to_numeric(df[test_col], errors="coerce").isin(list(HELDOUT_TEST_VALUES))
    else:
        test_mask = pd.Series(False, index=df.index)
    dev_df = df.loc[~test_mask & pd.to_numeric(df[label_col], errors="coerce").notna()].copy()
    test_df = df.loc[test_mask].copy()
    logger.info(
        "Split sizes | development=%d | held-out test=%d | test rule: %s in %s",
        len(dev_df), len(test_df), test_col, sorted(HELDOUT_TEST_VALUES),
    )
    return dev_df, test_df


def _build_pinn_bundle_for_dev(
    args: argparse.Namespace,
    dev_pids: list[str],
    test_pids: list[str],
    logger: logging.Logger,
):
    """Build a curve_records bundle from tcc_csv, restricted to dev + test PIDs.

    For CV we will only call train_pids on a subset of dev_pids per fold, so the
    curve bundle includes all eligible PIDs once and is reused across folds.
    """
    config = pinn_config_from_args(args)
    bundle = prepare_curve_records(
        tcc_csv=args.tcc_csv,
        metadata_csv=args.metadata_csv,
        pid_col=args.pid_col,
        test_col=args.test_col,
        label_col=args.label_col,
        config=config,
        logger=logger,
    )
    eligible = set(bundle["eligible_pids"])
    keep_pids = (set(dev_pids) | set(test_pids)) & eligible
    if not keep_pids:
        raise ValueError("No PIDs eligible for PINN feature generation after curve preparation.")
    return bundle, config


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)

    logger.info("Feature preset: %s", args.feature_mode)
    feature_cols = feature_columns_for_preset(args.feature_mode)
    logger.info("Selected %d feature columns: %s", len(feature_cols), feature_cols)

    df = _load_and_merge_features(
        feature_csv=args.feature_csv,
        volume_csv=args.volume_csv,
        metadata_csv=args.metadata_csv,
        pid_col=args.pid_col,
        label_col=args.label_col,
        test_col=args.test_col,
        logger=logger,
    )

    # Initialise Tofts feature columns on the merged dataframe so fold-wise
    # PINN refits in CV can write into them in place.
    for col in ("core15_Ktrans", "core15_kep", "safe_rim_Ktrans", "safe_rim_kep"):
        if col not in df.columns:
            df[col] = np.nan

    dev_df, test_df = _split_dev_test(df, args.label_col, args.test_col, logger)
    if len(dev_df) < args.n_splits * 2:
        raise ValueError(
            f"Development set is too small after filtering (n={len(dev_df)}); "
            f"need at least {args.n_splits * 2} labeled cases for stable CV."
        )

    # PINN bundle is required for any preset that contains Tofts features.
    needs_pinn = any(c in feature_cols for c in (
        "core15_Ktrans", "core15_kep", "safe_rim_Ktrans", "safe_rim_kep"
    ))
    pinn_bundle = None
    pinn_config = None
    if needs_pinn:
        logger.info("Preparing curve records for fold-wise Tofts-PINN refit.")
        pinn_bundle, pinn_config = _build_pinn_bundle_for_dev(
            args=args,
            dev_pids=dev_df[args.pid_col].astype(str).tolist(),
            test_pids=test_df[args.pid_col].astype(str).tolist(),
            logger=logger,
        )
        # Save curve artifacts for transparency.
        save_curve_artifacts(bundle=pinn_bundle, output_dir=args.output_dir)
        # Restrict dev_df / test_df to PIDs eligible for PINN feature generation.
        eligible = set(pinn_bundle["eligible_pids"])
        before_dev = len(dev_df)
        before_test = len(test_df)
        dev_df = dev_df.loc[dev_df[args.pid_col].astype(str).isin(eligible)].copy()
        test_df = test_df.loc[test_df[args.pid_col].astype(str).isin(eligible)].copy()
        logger.info(
            "After PINN-eligibility filtering | development: %d -> %d | held-out test: %d -> %d",
            before_dev, len(dev_df), before_test, len(test_df),
        )

    xgb_kwargs = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel,
        colsample_bynode=args.colsample_bynode,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        scale_pos_weight=args.scale_pos_weight,
        eval_metric=args.eval_metric,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
    )

    cv_result = run_stratified_kfold_cv(
        dev_df=dev_df,
        feature_cols=feature_cols,
        pid_col=args.pid_col,
        label_col=args.label_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
        xgb_kwargs=xgb_kwargs,
        threshold_metric=args.threshold_metric,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        pinn_bundle=pinn_bundle,
        pinn_config=pinn_config,
        logger=logger,
    )

    cv_result["fold_metrics_df"].to_csv(args.output_dir / "xgboost_cv_fold_metrics.csv", index=False)
    cv_result["oof_df"].to_csv(args.output_dir / "xgboost_cv_oof_predictions.csv", index=False)
    with open(args.output_dir / "xgboost_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_result["cv_summary"], f, indent=2, ensure_ascii=False)
    logger.info(
        "CV finished | OOF AUROC=%.4f | OOF AUPRC=%.4f | threshold=%.4f | final_n_estimators=%d",
        float(cv_result["cv_summary"]["oof_metrics"].get("roc_auc", float("nan"))),
        float(cv_result["cv_summary"]["oof_metrics"].get("average_precision", float("nan"))),
        float(cv_result["decision_threshold_used"]),
        int(cv_result["final_n_estimators_from_cv"]),
    )

    test_metrics: dict[str, Any] | None = None
    test_pred_df = pd.DataFrame()
    final_dev_df = dev_df
    final_test_df = test_df

    if needs_pinn:
        # Final-stage Tofts feature generation: train on full development split
        # only, infer for development + held-out test. This is the
        # "single-model" Tofts feature table that gets exported.
        logger.info("Final-stage Tofts-PINN refit on full development split.")
        train_pids = dev_df[args.pid_col].astype(str).tolist()
        infer_pids = (
            dev_df[args.pid_col].astype(str).tolist()
            + test_df[args.pid_col].astype(str).tolist()
        )
        final_pinn = train_pinn_and_infer_for_pid_sets(
            curve_records=pinn_bundle["curve_records"],
            train_pids=train_pids,
            infer_pids=infer_pids,
            time_grid_seconds=pinn_bundle["time_grid_seconds"],
            aif_cp=pinn_bundle["aif_cp"],
            random_state=int(args.random_state),
            config=pinn_config,
            logger=logger,
        )

        base_case_df = pd.concat([dev_df, test_df], ignore_index=True)
        final_feature_df = build_final_tofts_feature_table(
            base_case_df=base_case_df,
            pinn_wide_df=final_pinn["feature_wide_df"],
            pid_col=args.pid_col,
            test_col=args.test_col,
            label_col=args.label_col,
        )
        final_feature_df.to_csv(args.output_dir / "tofts_pinn_features.csv", index=False)

        # Held-out test subset.
        if "split_group" in final_feature_df.columns:
            test_only = final_feature_df.loc[final_feature_df["split_group"] == "heldout_test"].copy()
        else:
            test_only = pd.DataFrame(columns=final_feature_df.columns)
        test_only.to_csv(args.output_dir / "pinn_test_features.csv", index=False)

        with open(args.output_dir / "tofts_pinn_training_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    **final_pinn["fit_summary"],
                    "pinn_feature_generation_mode": "final_full_development_refit_only",
                    "whether_test_seen_in_pinn_training": False,
                    "development_pid_count": int(len(dev_df)),
                    "heldout_test_pid_count": int(len(test_df)),
                    "selected_feature_columns": [
                        "core15_Ktrans", "core15_kep", "safe_rim_Ktrans", "safe_rim_kep",
                    ],
                    "random_state": int(args.random_state),
                },
                f, indent=2, ensure_ascii=False,
            )

        # Use the final-stage Tofts features on dev_df / test_df for the final
        # XGBoost evaluation if requested.
        wide = final_pinn["feature_wide_df"].copy()
        wide["pid"] = wide["pid"].astype(str)
        for col in ("core15_Ktrans", "core15_kep", "safe_rim_Ktrans", "safe_rim_kep"):
            if col in wide.columns:
                lookup = wide.set_index("pid")[col]
                final_dev_df = final_dev_df.copy()
                final_dev_df[col] = final_dev_df[args.pid_col].astype(str).map(lookup)
                final_test_df = final_test_df.copy()
                final_test_df[col] = final_test_df[args.pid_col].astype(str).map(lookup)

    if args.run_final_test:
        final = fit_final_model_and_evaluate(
            dev_df=final_dev_df,
            test_df=final_test_df,
            feature_cols=feature_cols,
            pid_col=args.pid_col,
            label_col=args.label_col,
            decision_threshold=float(cv_result["decision_threshold_used"]),
            n_estimators=int(cv_result["final_n_estimators_from_cv"] or args.n_estimators),
            random_state=args.random_state,
            xgb_kwargs={k: v for k, v in xgb_kwargs.items() if k != "n_estimators"},
            logger=logger,
        )
        test_pred_df = final["test_pred_df"]
        test_metrics = final["test_metrics"]

    # Aggregate predictions table.
    predictions_frames = [cv_result["oof_df"].copy()]
    if not test_pred_df.empty:
        predictions_frames.append(test_pred_df)
    predictions_df = pd.concat(predictions_frames, ignore_index=True)
    predictions_df.to_csv(args.output_dir / "xgboost_predictions.csv", index=False)

    # Final metrics payload.
    metrics_payload: dict[str, Any] = {
        "feature_mode": args.feature_mode,
        "feature_columns": feature_cols,
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "decision_threshold_metric": args.threshold_metric,
        "decision_threshold_used": float(cv_result["decision_threshold_used"]),
        "development_n": int(len(dev_df)),
        "heldout_test_n": int(len(test_df)),
        "cv_oof_metrics": cv_result["cv_summary"]["oof_metrics"],
        "cv_fold_metrics": cv_result["cv_summary"]["fold_metrics"],
        "final_n_estimators_from_cv": int(cv_result["final_n_estimators_from_cv"]),
        "heldout_test_metrics": test_metrics,
        "pinn_used": bool(needs_pinn),
    }
    with open(args.output_dir / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    logger.info("Finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
