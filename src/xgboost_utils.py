"""
XGBoost utilities for the post-segmentation pCR pipeline.

This module provides:

- ``MAIN_FEATURE_COLUMNS``: the 12-feature main preset described in the manuscript.
- ``FEATURE_PRESETS``: dict of feature presets used by the ablations.
- ``build_xgb_classifier``: a thin factory wrapping ``xgboost.XGBClassifier`` with
  the project's default hyperparameters.
- ``prepare_feature_matrix``: median-imputed feature matrix ready for XGBoost.
- ``run_stratified_kfold_cv``: leakage-safe 5-fold CV on a feature dataframe,
  optionally regenerating Tofts-PINN features fold-wise from a pre-built curve
  bundle so test cases never appear in any fold's PINN training set.

This module deliberately omits all visual-branch / RadImageNet logic from the
original research code base: the public release covers the post-segmentation
tabular pipeline only.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from metrics_utils import compute_binary_metrics, select_best_threshold


# --- Feature presets ---------------------------------------------------------

#: The 12-feature main preset reported in the manuscript.
MAIN_FEATURE_COLUMNS: list[str] = [
    "core15_Ktrans",
    "core15_kep",
    "safe_rim_Ktrans",
    "safe_rim_kep",
    "tumor_volume_mm3",
    "core15_volume_mm3",
    "safe_rim_volume_mm3",
    "safe_rim_to_core15_volume_ratio",
    "clinical_age",
    "clinical_hr",
    "clinical_her2",
    "clinical_triple_neg",
]

#: 4 Tofts parameters from the PINN.
TOFTS_FEATURE_COLUMNS: list[str] = [
    "core15_Ktrans",
    "core15_kep",
    "safe_rim_Ktrans",
    "safe_rim_kep",
]

#: 4 volume features.
VOLUME_FEATURE_COLUMNS: list[str] = [
    "tumor_volume_mm3",
    "core15_volume_mm3",
    "safe_rim_volume_mm3",
    "safe_rim_to_core15_volume_ratio",
]

#: 4 clinical baseline features.
CLINICAL_FEATURE_COLUMNS: list[str] = [
    "clinical_age",
    "clinical_hr",
    "clinical_her2",
    "clinical_triple_neg",
]

FEATURE_PRESETS: dict[str, list[str]] = {
    # Main preset reported in the manuscript: Tofts + volumes + clinical (12).
    "main": list(MAIN_FEATURE_COLUMNS),
    # Ablations.
    "tofts_only": list(TOFTS_FEATURE_COLUMNS),
    "volume_only": list(VOLUME_FEATURE_COLUMNS),
    "clinical_only": list(CLINICAL_FEATURE_COLUMNS),
    "tofts_volume_clinical": list(MAIN_FEATURE_COLUMNS),  # alias of "main"
}


def feature_columns_for_preset(preset: str) -> list[str]:
    """Return the feature column list for the named preset."""
    if preset not in FEATURE_PRESETS:
        raise ValueError(
            f"Unknown feature preset {preset!r}. "
            f"Available: {sorted(FEATURE_PRESETS.keys())}"
        )
    return list(FEATURE_PRESETS[preset])


# --- Feature matrix preparation ---------------------------------------------


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Coerce the requested feature columns to numeric and median-impute missing
    values. Returns a new DataFrame with the same columns in the same order.
    """
    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not found in dataframe: {missing}")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.empty:
        return X.copy()
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(X_arr, columns=feature_cols, index=df.index)


# --- Classifier factory ------------------------------------------------------


def build_xgb_classifier(
    n_estimators: int = 1500,
    max_depth: int = 2,
    learning_rate: float = 0.02,
    subsample: float = 0.85,
    colsample_bytree: float = 0.75,
    colsample_bylevel: float = 0.90,
    colsample_bynode: float = 0.90,
    min_child_weight: float = 2.0,
    reg_lambda: float = 6.0,
    reg_alpha: float = 0.5,
    gamma: float = 0.1,
    scale_pos_weight: float = 1.0,
    eval_metric: str = "auc",
    tree_method: str = "hist",
    n_jobs: int = 1,
    random_state: int = 42,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    """Construct an ``XGBClassifier`` with the project's default hyperparameters."""
    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        colsample_bylevel=float(colsample_bylevel),
        colsample_bynode=float(colsample_bynode),
        min_child_weight=float(min_child_weight),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        gamma=float(gamma),
        scale_pos_weight=float(scale_pos_weight),
        eval_metric=str(eval_metric),
        tree_method=str(tree_method),
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
        objective="binary:logistic",
        verbosity=0,
    )


# --- Stratified K-fold CV ----------------------------------------------------


def _resolve_pinn_args(args_obj: Any) -> dict[str, Any]:
    """Pull PINN-related fields off an argparse Namespace into a plain dict."""
    from tofts_pinn_utils import pinn_config_from_args

    return pinn_config_from_args(args_obj)


def run_stratified_kfold_cv(
    dev_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pid_col: str = "pid",
    label_col: str = "pCR",
    n_splits: int = 5,
    random_state: int = 42,
    xgb_kwargs: dict[str, Any] | None = None,
    threshold_metric: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    pinn_bundle: dict[str, Any] | None = None,
    pinn_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Stratified K-fold CV with optional fold-wise Tofts-PINN feature regeneration.

    Parameters
    ----------
    dev_df : pd.DataFrame
        The development split (no held-out test cases).
    feature_cols : sequence of str
        Tabular feature columns to feed XGBoost. May include Tofts columns.
    pinn_bundle : dict, optional
        If provided, fold-wise PINN training is run on each fold's training PIDs
        only, and the resulting ``core15_Ktrans`` / ``core15_kep`` /
        ``safe_rim_Ktrans`` / ``safe_rim_kep`` columns on ``dev_df`` are
        overridden in each fold. When ``None``, the columns already present on
        ``dev_df`` are used as-is.
    pinn_config : dict, optional
        Passed to ``train_pinn_and_infer_for_pid_sets``. Required if
        ``pinn_bundle`` is provided.
    threshold_metric : {"f1", "balanced_accuracy"}
        Metric used to pick a single decision threshold from the OOF predictions.
    """
    feature_cols = list(feature_cols)
    xgb_kwargs = dict(xgb_kwargs or {})

    if pid_col not in dev_df.columns:
        raise KeyError(f"dev_df is missing required column {pid_col!r}")
    if label_col not in dev_df.columns:
        raise KeyError(f"dev_df is missing required column {label_col!r}")

    dev_df = dev_df.copy().reset_index(drop=True)
    dev_df[label_col] = pd.to_numeric(dev_df[label_col], errors="coerce")
    if dev_df[label_col].isna().any():
        raise ValueError("Development split contains NaN labels; clean before CV.")
    y_dev = dev_df[label_col].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))

    fold_metrics_rows: list[dict[str, Any]] = []
    fold_n_estimators: list[int] = []
    oof_records: list[dict[str, Any]] = []
    pinn_fold_summaries: list[dict[str, Any]] = []

    pinn_columns = [
        "core15_Ktrans",
        "core15_kep",
        "safe_rim_Ktrans",
        "safe_rim_kep",
    ]

    for fold_index, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(dev_df)), y_dev),
        start=1,
    ):
        fold_train_df = dev_df.iloc[train_idx].copy()
        fold_val_df = dev_df.iloc[val_idx].copy()

        # Optional fold-wise PINN refit. Strictly leakage-safe: only this fold's
        # training PIDs are passed as ``train_pids``; held-out test never enters
        # because ``pinn_bundle`` was already restricted to development PIDs by
        # the caller.
        if pinn_bundle is not None:
            if pinn_config is None:
                raise ValueError("pinn_config must be supplied when pinn_bundle is used.")
            from tofts_pinn_utils import train_pinn_and_infer_for_pid_sets

            train_pids = fold_train_df[pid_col].astype(str).tolist()
            val_pids = fold_val_df[pid_col].astype(str).tolist()
            if logger is not None:
                logger.info(
                    "Fold %d/%d PINN fit | train_pids=%d val_pids=%d "
                    "| heldout_test_seen_in_pinn_training=false",
                    fold_index,
                    n_splits,
                    len(set(train_pids)),
                    len(set(val_pids)),
                )
            pinn_result = train_pinn_and_infer_for_pid_sets(
                curve_records=pinn_bundle["curve_records"],
                train_pids=train_pids,
                infer_pids=train_pids + val_pids,
                time_grid_seconds=pinn_bundle["time_grid_seconds"],
                aif_cp=pinn_bundle["aif_cp"],
                random_state=int(random_state) + int(fold_index),
                config=pinn_config,
                logger=logger,
            )
            wide_df = pinn_result["feature_wide_df"]
            wide_df = wide_df.set_index(wide_df["pid"].astype(str))
            for col in pinn_columns:
                if col in wide_df.columns:
                    fold_train_df[col] = (
                        fold_train_df[pid_col].astype(str).map(wide_df[col])
                    )
                    fold_val_df[col] = (
                        fold_val_df[pid_col].astype(str).map(wide_df[col])
                    )
            pinn_fold_summaries.append(
                {
                    "fold_index": int(fold_index),
                    "train_pid_count": int(len(set(train_pids))),
                    "val_pid_count": int(len(set(val_pids))),
                    "whether_test_seen_in_pinn_training": False,
                    **pinn_result["fit_summary"],
                }
            )

        X_train = prepare_feature_matrix(fold_train_df, feature_cols)
        X_val = prepare_feature_matrix(fold_val_df, feature_cols)
        y_train = fold_train_df[label_col].astype(int).to_numpy()
        y_val = fold_val_df[label_col].astype(int).to_numpy()

        clf = build_xgb_classifier(random_state=int(random_state) + int(fold_index), **xgb_kwargs)
        if clf.early_stopping_rounds:
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_n = int(getattr(clf, "best_iteration", clf.n_estimators) or clf.n_estimators) + 1
        else:
            clf.fit(X_train, y_train)
            best_n = int(clf.n_estimators)
        fold_n_estimators.append(best_n)

        y_val_prob = clf.predict_proba(X_val)[:, 1]
        fold_threshold_choice = select_best_threshold(
            y_val,
            y_val_prob,
            metric=threshold_metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
        )
        fold_metrics = fold_threshold_choice["metrics_at_threshold"]
        fold_metrics_rows.append(
            {
                "fold_index": int(fold_index),
                "best_n_estimators": int(best_n),
                **fold_metrics,
            }
        )

        for pid_value, true_label, prob in zip(
            fold_val_df[pid_col].astype(str).tolist(),
            y_val.tolist(),
            y_val_prob.tolist(),
        ):
            oof_records.append(
                {
                    pid_col: pid_value,
                    "fold_index": int(fold_index),
                    "y_true": int(true_label),
                    "y_prob": float(prob),
                    "split_group": "development_oof",
                }
            )

    oof_df = pd.DataFrame(oof_records)
    oof_threshold = select_best_threshold(
        oof_df["y_true"].to_numpy(),
        oof_df["y_prob"].to_numpy(),
        metric=threshold_metric,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    decision_threshold = float(oof_threshold["decision_threshold"])
    oof_df["y_pred"] = (oof_df["y_prob"].to_numpy() >= decision_threshold).astype(int)

    final_n_estimators = int(round(float(np.mean(fold_n_estimators)))) if fold_n_estimators else 0

    cv_summary = {
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "feature_columns": list(feature_cols),
        "fold_metrics": fold_metrics_rows,
        "oof_metrics": oof_threshold["metrics_at_threshold"],
        "decision_threshold_from_oof": decision_threshold,
        "threshold_metric": threshold_metric,
        "final_n_estimators_from_cv": final_n_estimators,
        "fold_best_n_estimators": [int(v) for v in fold_n_estimators],
        "pinn_fold_summaries": pinn_fold_summaries if pinn_fold_summaries else None,
    }

    return {
        "cv_summary": cv_summary,
        "oof_df": oof_df,
        "decision_threshold_used": decision_threshold,
        "final_n_estimators_from_cv": final_n_estimators,
        "fold_metrics_df": pd.DataFrame(fold_metrics_rows),
    }


# --- Final development-fit + held-out test evaluation ------------------------


def fit_final_model_and_evaluate(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: Sequence[str],
    *,
    pid_col: str = "pid",
    label_col: str = "pCR",
    decision_threshold: float = 0.5,
    n_estimators: int = 1500,
    random_state: int = 42,
    xgb_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Fit a final XGBoost on the entire development split and (optionally) score
    the held-out test split. Returns a dict with the fitted classifier, the
    test predictions, and the test metrics at ``decision_threshold``.
    """
    feature_cols = list(feature_cols)
    xgb_kwargs = dict(xgb_kwargs or {})
    xgb_kwargs.setdefault("n_estimators", int(n_estimators))
    # Final refit on the full dev split: no held-out validation, no early stopping.
    xgb_kwargs["early_stopping_rounds"] = None

    X_dev = prepare_feature_matrix(dev_df, feature_cols)
    y_dev = dev_df[label_col].astype(int).to_numpy()
    clf = build_xgb_classifier(random_state=int(random_state), **xgb_kwargs)
    clf.fit(X_dev, y_dev)

    test_pred_df: pd.DataFrame
    test_metrics: dict[str, Any] | None = None

    if test_df is None or test_df.empty:
        test_pred_df = pd.DataFrame(columns=[pid_col, "y_true", "y_prob", "y_pred", "split_group"])
    else:
        X_test = prepare_feature_matrix(test_df, feature_cols)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= float(decision_threshold)).astype(int)
        test_pred_df = pd.DataFrame(
            {
                pid_col: test_df[pid_col].astype(str).to_numpy(),
                "y_true": pd.to_numeric(test_df.get(label_col, np.nan), errors="coerce").to_numpy(),
                "y_prob": prob,
                "y_pred": pred,
                "split_group": "heldout_test",
            }
        )
        labeled_mask = pd.notna(test_pred_df["y_true"])
        if labeled_mask.any():
            test_metrics = compute_binary_metrics(
                test_pred_df.loc[labeled_mask, "y_true"].to_numpy(),
                test_pred_df.loc[labeled_mask, "y_prob"].to_numpy(),
                decision_threshold=float(decision_threshold),
            )

    if logger is not None and test_metrics is not None:
        logger.info(
            "Held-out test | n=%d AUROC=%.4f AUPRC=%.4f F1=%.4f",
            test_metrics.get("n_samples", 0),
            float(test_metrics.get("roc_auc", float("nan"))),
            float(test_metrics.get("average_precision", float("nan"))),
            float(test_metrics.get("f1", float("nan"))),
        )

    return {
        "model": clf,
        "test_pred_df": test_pred_df,
        "test_metrics": test_metrics,
    }


__all__ = [
    "MAIN_FEATURE_COLUMNS",
    "TOFTS_FEATURE_COLUMNS",
    "VOLUME_FEATURE_COLUMNS",
    "CLINICAL_FEATURE_COLUMNS",
    "FEATURE_PRESETS",
    "feature_columns_for_preset",
    "prepare_feature_matrix",
    "build_xgb_classifier",
    "run_stratified_kfold_cv",
    "fit_final_model_and_evaluate",
]
