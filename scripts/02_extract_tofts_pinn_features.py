#!/usr/bin/env python3
"""
Step 2: Tofts-PINN pharmacokinetic feature extraction on existing per-ROI TCC.

Inputs:
- a tcc_long.csv produced by ``scripts/01_extract_roi_tcc.py``
- (optional) the original metadata CSV (for split / label / physical-time fields)

Outputs (under --output_dir):
- aligned_tcc_curves.csv               : per-case per-ROI fixed-grid signal curves
- concentration_curves.csv             : per-case per-ROI concentration curves C_t(t)
- tofts_pinn_features.csv              : per-case Tofts parameters (K_trans, k_ep, ...)
- pinn_test_features.csv               : the held-out subset of the above
- tofts_pinn_training_summary.json     : configuration + fit-quality summary

Design rules:
- the final-stage PINN here is fit only on the development split, so held-out
  test cases are never seen during training of this final-stage refit
- for the leakage-safe 5-fold CV in step 3, fold-wise PINN refits are performed
  inside ``scripts/03_train_xgboost.py`` instead
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

from tofts_pinn_utils import (
    HELDOUT_TEST_VALUES,
    add_shared_tofts_pinn_args,
    build_final_tofts_feature_table,
    pinn_config_from_args,
    prepare_curve_records,
    save_curve_artifacts,
    train_pinn_and_infer_for_pid_sets,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aligned curves, concentration curves, and final-stage Tofts-PINN features "
            "from an existing tcc_long.csv table."
        )
    )
    parser.add_argument("--tcc_csv", type=Path, required=True, help="Path to tcc_long.csv")
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="Optional metadata.csv for physical times and split/label supplementation.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--pid_col", type=str, default="pid", help="PID column name in tcc_csv.")
    parser.add_argument("--label_col", type=str, default="pCR", help="Binary label column.")
    parser.add_argument("--test_col", type=str, default="test", help="Held-out test indicator column.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    add_shared_tofts_pinn_args(parser)
    return parser


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_tofts_pinn_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "tofts_pinn_extract.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _build_case_level_base_df(
    tcc_csv: Path,
    pid_col: str,
    label_col: str,
    test_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(tcc_csv)
    if pid_col not in df.columns:
        if "pid" in df.columns and pid_col != "pid":
            df = df.rename(columns={"pid": pid_col})
        else:
            raise ValueError(f"tcc_csv must contain `{pid_col}` column.")
    df = df.copy()
    df[pid_col] = df[pid_col].astype(str).str.strip()
    keep_cols = [pid_col]
    for col in [label_col, test_col]:
        if col in df.columns:
            keep_cols.append(col)
    base_df = df[keep_cols].drop_duplicates(subset=[pid_col], keep="first").reset_index(drop=True)
    return base_df


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    config = pinn_config_from_args(args)

    logger.info("Preparing aligned and concentration curves from %s", args.tcc_csv)
    bundle = prepare_curve_records(
        tcc_csv=args.tcc_csv,
        metadata_csv=args.metadata_csv,
        pid_col=args.pid_col,
        test_col=args.test_col,
        label_col=args.label_col,
        config=config,
        logger=logger,
    )
    save_curve_artifacts(bundle=bundle, output_dir=args.output_dir)

    base_df = _build_case_level_base_df(
        tcc_csv=args.tcc_csv,
        pid_col=args.pid_col,
        label_col=args.label_col,
        test_col=args.test_col,
    )
    eligible_pid_set = set(bundle["eligible_pids"])
    base_df = base_df[base_df[args.pid_col].astype(str).isin(eligible_pid_set)].copy()
    if base_df.empty:
        raise ValueError("No eligible pids remain after requiring valid core15 + safe_rim curves.")

    if args.test_col in base_df.columns:
        test_series = pd.to_numeric(base_df[args.test_col], errors="coerce")
        test_mask = test_series.isin(list(HELDOUT_TEST_VALUES))
    else:
        test_mask = pd.Series(False, index=base_df.index)
    dev_df = base_df.loc[~test_mask].copy()
    test_df = base_df.loc[test_mask].copy()

    if args.label_col in dev_df.columns:
        dev_df = dev_df.loc[pd.to_numeric(dev_df[args.label_col], errors="coerce").notna()].copy()

    train_pids = dev_df[args.pid_col].astype(str).tolist()
    infer_pids = base_df[args.pid_col].astype(str).tolist()
    if not train_pids:
        raise ValueError("No development pids available for final-stage PINN fitting.")

    logger.info(
        "Final-stage PINN fitting | train_on_development=%d pids | infer_on_total=%d pids | test_seen_in_training=false",
        len(train_pids),
        len(infer_pids),
    )
    pinn_result = train_pinn_and_infer_for_pid_sets(
        curve_records=bundle["curve_records"],
        train_pids=train_pids,
        infer_pids=infer_pids,
        time_grid_seconds=bundle["time_grid_seconds"],
        aif_cp=bundle["aif_cp"],
        random_state=int(args.random_state),
        config=config,
        logger=logger,
    )

    final_feature_df = build_final_tofts_feature_table(
        base_case_df=base_df,
        pinn_wide_df=pinn_result["feature_wide_df"],
        pid_col=args.pid_col,
        test_col=args.test_col,
        label_col=args.label_col,
    )
    final_feature_df.to_csv(args.output_dir / "tofts_pinn_features.csv", index=False)

    if not test_df.empty:
        test_feature_df = final_feature_df.loc[final_feature_df["split_group"] == "heldout_test"].copy()
    else:
        test_feature_df = pd.DataFrame(columns=final_feature_df.columns)
    test_feature_df.to_csv(args.output_dir / "pinn_test_features.csv", index=False)

    summary: dict[str, Any] = {
        **bundle["preproc_summary"],
        **pinn_result["fit_summary"],
        "pinn_feature_generation_mode": "final_full_development_refit_only",
        "whether_foldwise_pinn_training_used": False,
        "whether_test_seen_in_pinn_training": False,
        "development_pid_count": int(len(dev_df)),
        "heldout_test_pid_count": int(len(test_df)),
        "selected_feature_columns": [
            "core15_Ktrans",
            "core15_kep",
            "safe_rim_Ktrans",
            "safe_rim_kep",
        ],
        "random_state": int(args.random_state),
    }
    with open(args.output_dir / "tofts_pinn_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
