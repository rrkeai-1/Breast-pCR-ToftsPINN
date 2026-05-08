#!/usr/bin/env python3
"""
Step 4: Evaluate a predictions CSV with the standard metric panel.

Inputs:
- a CSV containing at least a column of true labels (default name: ``y_true``)
  and a column of predicted positive-class probabilities (default name:
  ``y_prob``)

Outputs (under --output_dir, or stdout if --output_dir is omitted):
- metrics.json (or filename specified via --output_json)

Optional behaviour:
- ``--split_col SPLIT_COL`` and ``--split_value VAL`` restrict the evaluation
  to rows where ``SPLIT_COL == VAL`` (e.g. ``--split_col split_group
  --split_value heldout_test``); without these flags, all rows are used.
- ``--decision_threshold T`` sets a fixed threshold; otherwise the script
  searches for the best threshold under ``--threshold_metric`` on the supplied
  rows.

This script is dataset-agnostic: it does not load any project-specific data,
only a predictions CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running this script directly without installing the project.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from metrics_utils import compute_binary_metrics, select_best_threshold


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: compute AUROC, AUPRC, F1, sensitivity, specificity, etc. "
            "from a predictions CSV."
        )
    )
    parser.add_argument("--predictions_csv", type=Path, required=True,
                        help="CSV with at least y_true and y_prob columns.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory to write metrics.json. If omitted, prints to stdout.")
    parser.add_argument("--output_json", type=str, default="metrics.json",
                        help="Filename for the metrics JSON inside --output_dir.")
    parser.add_argument("--y_true_col", type=str, default="y_true")
    parser.add_argument("--y_prob_col", type=str, default="y_prob")
    parser.add_argument("--split_col", type=str, default=None,
                        help="Optional column name to filter rows by, e.g. 'split_group'.")
    parser.add_argument("--split_value", type=str, default=None,
                        help="Optional value of --split_col to keep, e.g. 'heldout_test'.")
    parser.add_argument("--decision_threshold", type=float, default=None,
                        help="Fixed decision threshold. If omitted, searched on these rows.")
    parser.add_argument("--threshold_metric", type=str, default="f1",
                        choices=["f1", "balanced_accuracy"],
                        help="Metric maximised when searching for a decision threshold.")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    df = pd.read_csv(args.predictions_csv)

    if args.split_col is not None and args.split_value is not None:
        if args.split_col not in df.columns:
            raise KeyError(f"--split_col {args.split_col!r} not found in predictions CSV.")
        df = df.loc[df[args.split_col].astype(str) == str(args.split_value)].copy()

    if args.y_true_col not in df.columns or args.y_prob_col not in df.columns:
        raise KeyError(
            f"predictions CSV must contain columns {args.y_true_col!r} and {args.y_prob_col!r}; "
            f"found columns: {list(df.columns)}"
        )

    if args.decision_threshold is None:
        choice = select_best_threshold(
            df[args.y_true_col].to_numpy(),
            df[args.y_prob_col].to_numpy(),
            metric=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        threshold = float(choice["decision_threshold"])
        metrics = choice["metrics_at_threshold"]
        threshold_source = "selected_from_input_rows"
    else:
        threshold = float(args.decision_threshold)
        metrics = compute_binary_metrics(
            df[args.y_true_col].to_numpy(),
            df[args.y_prob_col].to_numpy(),
            decision_threshold=threshold,
        )
        threshold_source = "user_supplied"

    payload: dict[str, Any] = {
        "predictions_csv": str(args.predictions_csv),
        "split_col": args.split_col,
        "split_value": args.split_value,
        "decision_threshold": threshold,
        "decision_threshold_source": threshold_source,
        "threshold_metric": args.threshold_metric,
        "metrics": metrics,
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / args.output_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Metrics written to: {out_path}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
