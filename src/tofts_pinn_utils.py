#!/usr/bin/env python3
"""
Utilities for leakage-safe Tofts-PINN feature generation on ROI-level DCE curves.

Design goals
------------
- keep the implementation lightweight and deterministic
- use a shared MLP encoder because aligned curves are short
- operate on ROI-level curves only (core15 / safe_rim)
- support fixed-grid alignment and explicit signal->concentration conversion
- support fold-wise PINN training so CV remains leakage-safe
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset



REGION_TYPES = ("core15", "safe_rim")
HELDOUT_TEST_VALUES = {1, 2}
EPS = 1e-8

TOFTS_PRIMARY_FEATURE_COLS = [
    "core15_Ktrans",
    "core15_kep",
    "safe_rim_Ktrans",
    "safe_rim_kep",
]

TOFTS_RECON_FEATURE_COLS = [
    "core15_recon_mse",
    "safe_rim_recon_mse",
]

TOFTS_DERIVED_FEATURE_COLS = [
    "core15_ve",
    "safe_rim_ve",
    "safe_rim_minus_core15_Ktrans",
    "safe_rim_minus_core15_kep",
    "safe_rim_minus_core15_ve",
    "log_safe_rim_to_core15_Ktrans_ratio",
    "log_safe_rim_to_core15_kep_ratio",
    "log_safe_rim_to_core15_ve_ratio",
]

TOFTS_EXPORT_FEATURE_COLS = (
    TOFTS_PRIMARY_FEATURE_COLS
    + TOFTS_RECON_FEATURE_COLS
    + TOFTS_DERIVED_FEATURE_COLS
)


def add_shared_tofts_pinn_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Append shared Tofts-PINN arguments to a CLI parser."""
    parser.add_argument(
        "--default_temporal_resolution_seconds",
        type=float,
        default=90.0,
        help=(
            "Fallback temporal resolution used when only discrete timepoint indices are available. "
            "Also used as the default fixed-grid spacing when --time_grid_resolution_seconds is omitted."
        ),
    )
    parser.add_argument(
        "--time_grid_resolution_seconds",
        type=float,
        default=None,
        help=(
            "Fixed aligned-grid spacing in seconds. If omitted, the code uses "
            "--default_temporal_resolution_seconds."
        ),
    )
    parser.add_argument(
        "--time_grid_max_time_seconds",
        type=float,
        default=None,
        help=(
            "Optional fixed end time of the aligned grid in seconds. If omitted, the code uses the "
            "largest observed original time rounded up to the grid spacing."
        ),
    )
    parser.add_argument(
        "--concentration_mode",
        type=str,
        default="auto",
        choices=["auto", "spgr", "linearized"],
        help=(
            "Signal-to-concentration conversion mode. 'auto' prefers SPGR when sufficient parameters "
            "are available, otherwise falls back to an explicit linearized approximation."
        ),
    )
    parser.add_argument(
        "--default_tr",
        type=float,
        default=0.004,
        help="Fallback TR in seconds for SPGR conversion.",
    )
    parser.add_argument(
        "--default_flip_angle_deg",
        type=float,
        default=15.0,
        help="Fallback flip angle in degrees for SPGR conversion.",
    )
    parser.add_argument(
        "--default_t10",
        type=float,
        default=1.6,
        help="Fallback baseline T1 / T10 in seconds for approximate conversion.",
    )
    parser.add_argument(
        "--relaxivity_r1",
        type=float,
        default=4.5,
        help="Gd relaxivity r1 in L/mmol/s.",
    )
    parser.add_argument(
        "--baseline_n_pre_points",
        type=int,
        default=1,
        help="Number of earliest points to average when explicit pre-contrast marking is absent.",
    )
    parser.add_argument(
        "--aif_mode",
        type=str,
        default="population",
        choices=["population", "fixed_csv"],
        help="Arterial input function mode.",
    )
    parser.add_argument(
        "--aif_csv",
        type=Path,
        default=None,
        help="Optional external AIF CSV for --aif_mode fixed_csv.",
    )
    parser.add_argument(
        "--pinn_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of the shared MLP encoder.",
    )
    parser.add_argument(
        "--pinn_epochs",
        type=int,
        default=250,
        help="Maximum PINN training epochs.",
    )
    parser.add_argument(
        "--pinn_batch_size",
        type=int,
        default=32,
        help="PINN mini-batch size.",
    )
    parser.add_argument(
        "--pinn_learning_rate",
        type=float,
        default=1e-3,
        help="PINN Adam learning rate.",
    )
    parser.add_argument(
        "--pinn_weight_decay",
        type=float,
        default=1e-5,
        help="PINN optimizer weight decay.",
    )
    parser.add_argument(
        "--pinn_early_stopping_patience",
        type=int,
        default=25,
        help="Early-stopping patience on reconstruction loss.",
    )
    parser.add_argument(
        "--pinn_val_fraction",
        type=float,
        default=0.15,
        help="Patient-level validation fraction for early stopping inside the fold-training subset.",
    )
    parser.add_argument(
        "--pinn_gradient_clip",
        type=float,
        default=5.0,
        help="Global gradient clipping norm. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--pinn_lambda_reg",
        type=float,
        default=1e-4,
        help="Weight of the weak parameter-range regularizer.",
    )
    parser.add_argument(
        "--pinn_soft_upper_ktrans",
        type=float,
        default=2.0,
        help="Soft upper range used by the weak Ktrans penalty (min^-1).",
    )
    parser.add_argument(
        "--pinn_soft_upper_kep",
        type=float,
        default=5.0,
        help="Soft upper range used by the weak kep penalty (min^-1).",
    )
    parser.add_argument(
        "--pinn_device",
        type=str,
        default="auto",
        help="PINN device: auto / cpu / cuda / cuda:0 ...",
    )
    return parser


def pinn_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "default_temporal_resolution_seconds": float(args.default_temporal_resolution_seconds),
        "time_grid_resolution_seconds": (
            None if args.time_grid_resolution_seconds is None else float(args.time_grid_resolution_seconds)
        ),
        "time_grid_max_time_seconds": (
            None if args.time_grid_max_time_seconds is None else float(args.time_grid_max_time_seconds)
        ),
        "concentration_mode": str(args.concentration_mode),
        "default_tr": float(args.default_tr),
        "default_flip_angle_deg": float(args.default_flip_angle_deg),
        "default_t10": float(args.default_t10),
        "relaxivity_r1": float(args.relaxivity_r1),
        "baseline_n_pre_points": int(args.baseline_n_pre_points),
        "aif_mode": str(args.aif_mode),
        "aif_csv": None if args.aif_csv is None else str(args.aif_csv),
        "pinn_hidden_dim": int(args.pinn_hidden_dim),
        "pinn_epochs": int(args.pinn_epochs),
        "pinn_batch_size": int(args.pinn_batch_size),
        "pinn_learning_rate": float(args.pinn_learning_rate),
        "pinn_weight_decay": float(args.pinn_weight_decay),
        "pinn_early_stopping_patience": int(args.pinn_early_stopping_patience),
        "pinn_val_fraction": float(args.pinn_val_fraction),
        "pinn_gradient_clip": float(args.pinn_gradient_clip),
        "pinn_lambda_reg": float(args.pinn_lambda_reg),
        "pinn_soft_upper_ktrans": float(args.pinn_soft_upper_ktrans),
        "pinn_soft_upper_kep": float(args.pinn_soft_upper_kep),
        "pinn_device": str(args.pinn_device),
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(np.nan_to_num(float(value), nan=np.nan, posinf=np.nan, neginf=np.nan))
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series, eps: float = EPS) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    denom = denominator.where(denominator.abs() > float(eps), np.nan)
    out = numerator / denom
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_log_ratio_series(numerator: pd.Series, denominator: pd.Series, eps: float = EPS) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid_mask = numerator.notna() & denominator.notna() & (numerator >= 0) & (denominator >= 0)
    out = pd.Series(np.nan, index=numerator.index, dtype=float)
    out.loc[valid_mask] = np.log((numerator.loc[valid_mask] + float(eps)) / (denominator.loc[valid_mask] + float(eps)))
    return out.replace([np.inf, -np.inf], np.nan)


def add_tofts_derived_feature_columns(base_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    required_cols = [
        "core15_Ktrans",
        "core15_kep",
        "safe_rim_Ktrans",
        "safe_rim_kep",
    ]
    if not set(required_cols).issubset(set(out.columns)):
        return out

    core_ktrans = _safe_numeric_series(out, "core15_Ktrans")
    core_kep = _safe_numeric_series(out, "core15_kep")
    rim_ktrans = _safe_numeric_series(out, "safe_rim_Ktrans")
    rim_kep = _safe_numeric_series(out, "safe_rim_kep")

    out["core15_ve"] = _safe_divide_series(core_ktrans, core_kep)
    out["safe_rim_ve"] = _safe_divide_series(rim_ktrans, rim_kep)

    out["safe_rim_minus_core15_Ktrans"] = rim_ktrans - core_ktrans
    out["safe_rim_minus_core15_kep"] = rim_kep - core_kep
    out["safe_rim_minus_core15_ve"] = _safe_numeric_series(out, "safe_rim_ve") - _safe_numeric_series(out, "core15_ve")

    out["log_safe_rim_to_core15_Ktrans_ratio"] = _safe_log_ratio_series(rim_ktrans, core_ktrans)
    out["log_safe_rim_to_core15_kep_ratio"] = _safe_log_ratio_series(rim_kep, core_kep)
    out["log_safe_rim_to_core15_ve_ratio"] = _safe_log_ratio_series(
        _safe_numeric_series(out, "safe_rim_ve"),
        _safe_numeric_series(out, "core15_ve"),
    )

    return out


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip()).strip("_")


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_to_original = {_normalize_name(col): str(col) for col in df.columns}
    for candidate in candidates:
        original = normalized_to_original.get(_normalize_name(candidate))
        if original is not None:
            return original
    return None


def _load_metadata_csv_minimal(path: Path) -> pd.DataFrame:
    """
    Local lightweight metadata loader for PINN-side use only.

    This avoids importing extraction-side imaging utilities when only CSV metadata
    is needed for time-axis resolution and clinical/test columns.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"metadata CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    pid_col = _find_first_existing_column(df, ["pid", "PatientID", "patient_id", "case_id"])
    if pid_col is None:
        raise ValueError(
            "metadata CSV must contain a PID-like column, one of: "
            "pid / PatientID / patient_id / case_id"
        )
    if pid_col != "pid":
        df = df.rename(columns={pid_col: "pid"})

    canonical_map = {
        "n_times": ["n_times", "ntimes", "num_times", "n_timepoints"],
        "pre": ["pre"],
        "post_early": ["post_early", "postearly", "early", "post1"],
        "post_late": ["post_late", "postlate", "late", "post2"],
        "pCR": ["pCR", "pcr"],
        "split": ["split", "subset", "partition", "set", "group"],
        "test": ["test", "is_test", "heldout"],
        "dataset": ["dataset"],
    }

    for target, candidates in canonical_map.items():
        col = _find_first_existing_column(df, candidates)
        if col is not None and col != target:
            df = df.rename(columns={col: target})

    df["pid"] = df["pid"].astype(str).str.strip()

    numeric_columns = ["n_times", "pre", "post_early", "post_late", "pCR", "test"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _parse_numeric_sequence(raw_value: Any) -> list[float] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None

    if isinstance(raw_value, (list, tuple, np.ndarray)):
        seq = list(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        seq = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple)):
                seq = list(parsed)
        except Exception:
            seq = None
        if seq is None:
            cleaned = text.strip("[]()")
            parts = [part for part in cleaned.replace("|", ",").replace(";", ",").split(",") if part.strip()]
            if not parts:
                parts = text.split()
            seq = [part for part in parts if str(part).strip()]
    else:
        return None

    out: list[float] = []
    for item in seq:
        try:
            out.append(float(item))
        except Exception:
            return None
    return out


@dataclass
class CurveRecord:
    pid: str
    region_type: str
    original_timepoint_idx: list[int]
    original_time_seconds: list[float]
    original_signal_curve: list[float]
    original_concentration_curve: list[float]
    aligned_time_grid_seconds: list[float]
    aligned_signal_curve: list[float]
    aligned_concentration_curve: list[float]
    aligned_valid_mask: list[int]
    original_n_timepoints: int
    time_alignment_mode: str
    temporal_resolution_seconds_used: float
    baseline_timepoint_idx: int
    baseline_signal: float
    concentration_mode_used: str
    concentration_conversion_note: str
    concentration_approximate: bool
    curve_valid_for_pinn: int
    curve_invalid_reason: str
    split_group: str | None
    test_value: int | None
    label_value: int | None
    curve_source: str

    def aligned_row(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "region_type": self.region_type,
            "original_timepoint_idx_json": json.dumps(self.original_timepoint_idx),
            "original_time_seconds_json": json.dumps([_finite_float(v) for v in self.original_time_seconds]),
            "original_signal_curve_json": json.dumps([_finite_float(v) for v in self.original_signal_curve]),
            "aligned_time_grid_seconds_json": json.dumps([_finite_float(v) for v in self.aligned_time_grid_seconds]),
            "aligned_signal_curve_json": json.dumps([_finite_float(v) for v in self.aligned_signal_curve]),
            "aligned_valid_mask_json": json.dumps([int(v) for v in self.aligned_valid_mask]),
            "original_n_timepoints": int(self.original_n_timepoints),
            "time_alignment_mode": self.time_alignment_mode,
            "temporal_resolution_seconds_used": _finite_float(self.temporal_resolution_seconds_used),
            "baseline_timepoint_idx": int(self.baseline_timepoint_idx),
            "curve_valid_for_pinn": int(self.curve_valid_for_pinn),
            "curve_invalid_reason": self.curve_invalid_reason,
            "split_group": self.split_group,
            "test": self.test_value,
            "pCR": self.label_value,
            "curve_source": self.curve_source,
        }

    def concentration_row(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "region_type": self.region_type,
            "original_timepoint_idx_json": json.dumps(self.original_timepoint_idx),
            "original_time_seconds_json": json.dumps([_finite_float(v) for v in self.original_time_seconds]),
            "original_concentration_curve_json": json.dumps(
                [_finite_float(v) for v in self.original_concentration_curve]
            ),
            "aligned_time_grid_seconds_json": json.dumps([_finite_float(v) for v in self.aligned_time_grid_seconds]),
            "aligned_concentration_curve_json": json.dumps(
                [_finite_float(v) for v in self.aligned_concentration_curve]
            ),
            "aligned_valid_mask_json": json.dumps([int(v) for v in self.aligned_valid_mask]),
            "baseline_signal": _finite_float(self.baseline_signal),
            "baseline_timepoint_idx": int(self.baseline_timepoint_idx),
            "concentration_mode_used": self.concentration_mode_used,
            "concentration_conversion_note": self.concentration_conversion_note,
            "concentration_approximate": int(self.concentration_approximate),
            "original_n_timepoints": int(self.original_n_timepoints),
            "time_alignment_mode": self.time_alignment_mode,
            "temporal_resolution_seconds_used": _finite_float(self.temporal_resolution_seconds_used),
            "curve_valid_for_pinn": int(self.curve_valid_for_pinn),
            "curve_invalid_reason": self.curve_invalid_reason,
            "split_group": self.split_group,
            "test": self.test_value,
            "pCR": self.label_value,
            "curve_source": self.curve_source,
        }


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_torch_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _load_tcc_df(tcc_csv: Path, pid_col: str) -> pd.DataFrame:
    if not Path(tcc_csv).exists():
        raise FileNotFoundError(f"tcc_csv not found: {tcc_csv}")
    df = pd.read_csv(tcc_csv)
    if pid_col not in df.columns:
        if "pid" in df.columns and pid_col != "pid":
            df = df.rename(columns={"pid": pid_col})
        else:
            raise ValueError(f"tcc_csv must contain `{pid_col}` column.")
    df = df.copy()
    df[pid_col] = df[pid_col].astype(str).str.strip()
    return df


def _load_metadata_for_pinn(metadata_csv: Path | None) -> pd.DataFrame | None:
    if metadata_csv is None:
        return None
    if not Path(metadata_csv).exists():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")
    df = _load_metadata_csv_minimal(metadata_csv)
    df = df.copy()
    df["pid"] = df["pid"].astype(str).str.strip()
    return df


def _build_metadata_lookup(metadata_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    if metadata_df is None or metadata_df.empty:
        return {}
    lookup: dict[str, pd.Series] = {}
    for _, row in metadata_df.drop_duplicates(subset=["pid"], keep="first").iterrows():
        lookup[str(row["pid"]).strip()] = row
    return lookup


def _resolve_split_fields(
    pid: str,
    case_rows: pd.DataFrame,
    metadata_row: pd.Series | None,
    test_col: str,
    label_col: str,
) -> tuple[str | None, int | None, int | None]:
    def _value_from_rows(col: str) -> Any:
        if col in case_rows.columns:
            values = case_rows[col].dropna().tolist()
            if values:
                return values[0]
        if metadata_row is not None and col in metadata_row.index:
            value = metadata_row.get(col)
            if not pd.isna(value):
                return value
        return None

    test_value_raw = _value_from_rows(test_col)
    label_value_raw = _value_from_rows(label_col)
    split_value_raw = _value_from_rows("split")

    test_value = None if test_value_raw is None else _finite_int(test_value_raw, default=0)
    label_value = None if label_value_raw is None or pd.isna(label_value_raw) else _finite_int(label_value_raw, default=0)
    split_group = None if split_value_raw is None else str(split_value_raw)

    if test_value in HELDOUT_TEST_VALUES:
        split_group = "heldout_test"
    elif label_value is not None:
        split_group = "development"

    return split_group, test_value, label_value


def _resolve_time_axis_from_metadata(
    metadata_row: pd.Series | None,
    discovered_timepoints: list[int],
    default_temporal_resolution_seconds: float,
) -> tuple[np.ndarray, str]:
    discovered_timepoints = [int(v) for v in sorted(discovered_timepoints)]
    if not discovered_timepoints:
        return np.asarray([], dtype=float), "no_timepoints"

    if metadata_row is not None:
        lower_to_original = {str(col).strip().lower(): str(col) for col in metadata_row.index}

        def _get_column_value(candidates: Sequence[str]) -> tuple[Any, str | None]:
            for candidate in candidates:
                original = lower_to_original.get(candidate.lower())
                if original is None:
                    continue
                value = metadata_row.get(original)
                if pd.isna(value):
                    continue
                return value, original
            return None, None

        def _validated(values_sec: Sequence[float]) -> np.ndarray | None:
            arr = np.asarray(values_sec, dtype=float)
            if arr.size != len(discovered_timepoints):
                return None
            if np.any(~np.isfinite(arr)):
                return None
            if arr.size >= 2 and np.any(np.diff(arr) <= 0):
                return None
            return arr

        sec_list_candidates = [
            "timepoints_sec",
            "timepoint_seconds",
            "acquisition_times_sec",
            "times_sec",
            "dce_times_sec",
            "time_axis_seconds",
            "time_axis_sec",
            "aqc_times_sec",
        ]
        min_list_candidates = [
            "timepoints_min",
            "timepoint_minutes",
            "acquisition_times_min",
            "times_min",
            "dce_times_min",
            "time_axis_min",
            "aqc_times_min",
        ]

        raw_value, _ = _get_column_value(sec_list_candidates)
        seq = _parse_numeric_sequence(raw_value) if raw_value is not None else None
        if seq is not None:
            arr = _validated(seq)
            if arr is not None:
                return arr, "metadata_physical_time_seconds"

        raw_value, _ = _get_column_value(min_list_candidates)
        seq = _parse_numeric_sequence(raw_value) if raw_value is not None else None
        if seq is not None:
            arr = _validated([float(v) * 60.0 for v in seq])
            if arr is not None:
                return arr, "metadata_physical_time_minutes"

        per_tp_sec_values: list[float] = []
        per_tp_sec_found = True
        for tp in discovered_timepoints:
            candidates = [
                f"tp{tp}_sec",
                f"time{tp}_sec",
                f"time_{tp}_sec",
                f"timepoint_{tp}_sec",
                f"aqc_{tp}_sec",
                f"acq_{tp}_sec",
                f"dce_{tp}_sec",
                f"t{tp}_sec",
            ]
            value, _ = _get_column_value(candidates)
            if value is None:
                per_tp_sec_found = False
                break
            try:
                per_tp_sec_values.append(float(value))
            except Exception:
                per_tp_sec_found = False
                break
        if per_tp_sec_found:
            arr = _validated(per_tp_sec_values)
            if arr is not None:
                return arr, "metadata_per_timepoint_seconds"

        per_tp_min_values: list[float] = []
        per_tp_min_found = True
        for tp in discovered_timepoints:
            candidates = [
                f"tp{tp}_min",
                f"time{tp}_min",
                f"time_{tp}_min",
                f"timepoint_{tp}_min",
                f"aqc_{tp}_min",
                f"acq_{tp}_min",
                f"dce_{tp}_min",
                f"t{tp}_min",
            ]
            value, _ = _get_column_value(candidates)
            if value is None:
                per_tp_min_found = False
                break
            try:
                per_tp_min_values.append(float(value) * 60.0)
            except Exception:
                per_tp_min_found = False
                break
        if per_tp_min_found:
            arr = _validated(per_tp_min_values)
            if arr is not None:
                return arr, "metadata_per_timepoint_minutes"

    min_tp = min(discovered_timepoints)
    seconds = np.asarray([(tp - min_tp) * float(default_temporal_resolution_seconds) for tp in discovered_timepoints], dtype=float)
    return seconds, "timepoint_index_default_resolution_fallback"


def _extract_time_axis_from_tcc_rows(
    region_rows: pd.DataFrame,
) -> tuple[np.ndarray | None, str | None]:
    timepoint_idx = pd.to_numeric(region_rows["timepoint_idx"], errors="coerce").astype(int).to_numpy()
    if "time_axis_seconds" in region_rows.columns:
        sec = pd.to_numeric(region_rows["time_axis_seconds"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(sec).sum() == len(sec) and len(sec) >= 2 and np.all(np.diff(sec[np.argsort(timepoint_idx)]) > 0):
            order = np.argsort(timepoint_idx)
            return sec[order], "tcc_time_axis_seconds_column"

    if "time_axis_value" in region_rows.columns and "time_axis_kind" in region_rows.columns:
        values = pd.to_numeric(region_rows["time_axis_value"], errors="coerce").to_numpy(dtype=float)
        kinds = region_rows["time_axis_kind"].astype(str).str.strip().str.lower().tolist()
        if all(kind == "physical_time" for kind in kinds) and np.isfinite(values).sum() == len(values):
            order = np.argsort(timepoint_idx)
            ordered = values[order]
            if len(ordered) >= 2 and np.all(np.diff(ordered) > 0):
                return ordered, "tcc_physical_time_axis_value_column"

    return None, None


def _resolve_baseline_from_rows(region_rows: pd.DataFrame, baseline_n_pre_points: int) -> tuple[int, float]:
    ordered = region_rows.sort_values("timepoint_idx", kind="stable").reset_index(drop=True)

    if "is_pre" in ordered.columns:
        pre_rows = ordered.loc[pd.to_numeric(ordered["is_pre"], errors="coerce").fillna(0).astype(int) == 1]
        if not pre_rows.empty:
            baseline_idx = int(pre_rows.iloc[0]["timepoint_idx"])
            baseline_signal = float(pd.to_numeric(pre_rows["mean_signal"], errors="coerce").mean())
            return baseline_idx, baseline_signal

    if "baseline_timepoint_idx" in ordered.columns:
        baseline_idx_raw = pd.to_numeric(ordered["baseline_timepoint_idx"], errors="coerce").dropna()
        if not baseline_idx_raw.empty:
            baseline_idx = int(baseline_idx_raw.iloc[0])
            matched = ordered.loc[pd.to_numeric(ordered["timepoint_idx"], errors="coerce").astype(int) == baseline_idx]
            if not matched.empty:
                baseline_signal = float(pd.to_numeric(matched["mean_signal"], errors="coerce").mean())
                return baseline_idx, baseline_signal

    n_points = max(1, int(baseline_n_pre_points))
    first_rows = ordered.iloc[:n_points]
    baseline_idx = int(first_rows.iloc[0]["timepoint_idx"])
    baseline_signal = float(pd.to_numeric(first_rows["mean_signal"], errors="coerce").mean())
    return baseline_idx, baseline_signal


def _resolve_spgr_parameters(
    metadata_row: pd.Series | None,
    default_tr: float,
    default_flip_angle_deg: float,
    default_t10: float,
) -> tuple[float | None, float | None, float | None, list[str]]:
    notes: list[str] = []

    def _extract_value(candidates: Sequence[str], scale: float = 1.0) -> float | None:
        if metadata_row is None:
            return None
        col = _find_first_existing_column(pd.DataFrame(columns=metadata_row.index), candidates)
        if col is None:
            return None
        value = metadata_row.get(col)
        if pd.isna(value):
            return None
        try:
            return float(value) * float(scale)
        except Exception:
            return None

    tr_sec = _extract_value(["tr_sec", "tr_seconds", "repetition_time_sec"], scale=1.0)
    if tr_sec is None:
        tr_ms = _extract_value(["tr", "tr_ms", "repetition_time", "repetition_time_ms"], scale=1e-3)
        tr_sec = tr_ms
        if tr_sec is None and default_tr is not None:
            tr_sec = float(default_tr)
            notes.append("TR used fallback default.")
    flip_angle_deg = _extract_value(["flip_angle_deg", "flip_angle", "fa_deg", "fa"], scale=1.0)
    if flip_angle_deg is None and default_flip_angle_deg is not None:
        flip_angle_deg = float(default_flip_angle_deg)
        notes.append("Flip angle used fallback default.")

    t10_sec = _extract_value(["t10_sec", "t10_seconds", "baseline_t1_sec", "baseline_t10_sec"], scale=1.0)
    if t10_sec is None:
        t10_ms = _extract_value(["t10", "baseline_t1", "baseline_t10", "t1_0_ms", "baseline_t1_ms"], scale=1e-3)
        t10_sec = t10_ms
    if t10_sec is None and default_t10 is not None:
        t10_sec = float(default_t10)
        notes.append("T10 used fallback default (approximate, not patient-specific).")

    return tr_sec, flip_angle_deg, t10_sec, notes


def _spgr_signal_to_concentration(
    signal_curve: np.ndarray,
    baseline_signal: float,
    tr_sec: float,
    flip_angle_deg: float,
    t10_sec: float,
    relaxivity_r1: float,
) -> np.ndarray:
    signal_curve = np.asarray(signal_curve, dtype=float)
    baseline_signal = float(max(EPS, baseline_signal))
    tr_sec = float(max(EPS, tr_sec))
    t10_sec = float(max(EPS, t10_sec))
    relaxivity_r1 = float(max(EPS, relaxivity_r1))

    alpha = math.radians(float(flip_angle_deg))
    cos_a = math.cos(alpha)
    e10 = math.exp(-tr_sec / t10_sec)

    a0 = (1.0 - e10) / max(EPS, (1.0 - cos_a * e10))
    ratio = np.nan_to_num(signal_curve / baseline_signal, nan=1.0, posinf=1.0, neginf=1.0)
    b = ratio * a0
    denom = 1.0 - b * cos_a
    denom = np.where(np.abs(denom) < EPS, np.sign(denom) * EPS + (denom == 0) * EPS, denom)
    e1 = (1.0 - b) / denom
    e1 = np.clip(e1, 1e-6, 0.999999)
    t1_t = -tr_sec / np.log(e1)
    ct = (1.0 / t1_t - 1.0 / t10_sec) / relaxivity_r1
    ct = np.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
    return ct.astype(float)


def _linearized_signal_to_concentration(
    signal_curve: np.ndarray,
    baseline_signal: float,
    default_t10: float,
    relaxivity_r1: float,
) -> np.ndarray:
    signal_curve = np.asarray(signal_curve, dtype=float)
    baseline_signal = float(max(EPS, baseline_signal))
    scale = max(EPS, float(default_t10) * float(relaxivity_r1))
    enhancement = (signal_curve - baseline_signal) / baseline_signal
    ct = enhancement / scale
    ct = np.maximum(ct, 0.0)
    ct = np.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
    return ct.astype(float)


def signal_to_concentration(
    signal_curve: Sequence[float],
    baseline_signal: float,
    metadata_row: pd.Series | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, str, str, bool]:
    signal_curve_np = np.asarray(signal_curve, dtype=float)
    requested_mode = str(config["concentration_mode"]).strip().lower()

    tr_sec, flip_angle_deg, t10_sec, spgr_notes = _resolve_spgr_parameters(
        metadata_row=metadata_row,
        default_tr=float(config["default_tr"]),
        default_flip_angle_deg=float(config["default_flip_angle_deg"]),
        default_t10=float(config["default_t10"]),
    )

    if requested_mode == "linearized":
        ct = _linearized_signal_to_concentration(
            signal_curve=signal_curve_np,
            baseline_signal=baseline_signal,
            default_t10=float(config["default_t10"]),
            relaxivity_r1=float(config["relaxivity_r1"]),
        )
        note = (
            "Approximate linearized conversion based on baseline-normalized signal enhancement. "
            "This is semi-quantitative / weakly quantitative."
        )
        return ct, "linearized", note, True

    if requested_mode == "spgr":
        if tr_sec is None or flip_angle_deg is None or t10_sec is None:
            raise ValueError(
                "SPGR conversion requested, but TR / flip angle / T10 could not be resolved from metadata or defaults."
            )
        ct = _spgr_signal_to_concentration(
            signal_curve=signal_curve_np,
            baseline_signal=baseline_signal,
            tr_sec=float(tr_sec),
            flip_angle_deg=float(flip_angle_deg),
            t10_sec=float(t10_sec),
            relaxivity_r1=float(config["relaxivity_r1"]),
        )
        approximate = bool(spgr_notes)
        note = (
            "SPGR conversion was used. "
            + ("Approximate because one or more parameters came from defaults. " if approximate else "Using available case-level sequence parameters. ")
            + "This implementation does not claim strict clinical PK quantitation without full T1 mapping."
        )
        return ct, ("spgr_approximate" if approximate else "spgr"), note, approximate

    # auto
    if tr_sec is not None and flip_angle_deg is not None and t10_sec is not None:
        ct = _spgr_signal_to_concentration(
            signal_curve=signal_curve_np,
            baseline_signal=baseline_signal,
            tr_sec=float(tr_sec),
            flip_angle_deg=float(flip_angle_deg),
            t10_sec=float(t10_sec),
            relaxivity_r1=float(config["relaxivity_r1"]),
        )
        approximate = bool(spgr_notes)
        note = (
            "Auto mode selected SPGR conversion. "
            + ("Approximate because one or more parameters came from defaults. " if approximate else "Using available case-level sequence parameters. ")
            + "This implementation should still be interpreted cautiously on low-temporal-resolution I-SPY2-style data."
        )
        return ct, ("spgr_approximate" if approximate else "spgr"), note, approximate

    ct = _linearized_signal_to_concentration(
        signal_curve=signal_curve_np,
        baseline_signal=baseline_signal,
        default_t10=float(config["default_t10"]),
        relaxivity_r1=float(config["relaxivity_r1"]),
    )
    note = (
        "Auto mode fell back to approximate linearized conversion because sufficient SPGR inputs were unavailable. "
        "This is semi-quantitative / weakly quantitative."
    )
    return ct, "linearized", note, True


def _build_fixed_time_grid(original_max_time_seconds: float, config: dict[str, Any]) -> np.ndarray:
    grid_resolution_seconds = (
        float(config["time_grid_resolution_seconds"])
        if config.get("time_grid_resolution_seconds") is not None
        else float(config["default_temporal_resolution_seconds"])
    )
    grid_resolution_seconds = max(EPS, grid_resolution_seconds)

    if config.get("time_grid_max_time_seconds") is not None:
        max_time = float(config["time_grid_max_time_seconds"])
    else:
        max_time = max(grid_resolution_seconds, float(original_max_time_seconds))
    n_steps = int(math.ceil(max_time / grid_resolution_seconds))
    grid = np.asarray([step * grid_resolution_seconds for step in range(n_steps + 1)], dtype=float)
    if grid.size < 2:
        grid = np.asarray([0.0, grid_resolution_seconds], dtype=float)
    return grid


def _interpolate_with_nan_edges(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_times.size == 0 or source_values.size == 0:
        return np.full_like(target_times, np.nan, dtype=float), np.zeros_like(target_times, dtype=int)

    order = np.argsort(source_times)
    source_times = np.asarray(source_times[order], dtype=float)
    source_values = np.asarray(source_values[order], dtype=float)

    target = np.asarray(target_times, dtype=float)
    aligned = np.full_like(target, np.nan, dtype=float)
    valid_mask = (target >= float(source_times[0])) & (target <= float(source_times[-1]))
    if valid_mask.any():
        aligned[valid_mask] = np.interp(target[valid_mask], source_times, source_values)
    return aligned, valid_mask.astype(int)


def parker_population_aif(time_grid_seconds: Sequence[float]) -> np.ndarray:
    """
    Parker population AIF, times expressed in minutes.

    Functional form:
        Cp(t) = A1/(sigma1*sqrt(2*pi))*exp(-(t-T1)^2/(2*sigma1^2))
              + A2/(sigma2*sqrt(2*pi))*exp(-(t-T2)^2/(2*sigma2^2))
              + alpha*exp(-beta*t) / (1 + exp(-s*(t-tau)))

    Parameter values follow Parker et al. 2006 and are reproduced in later open tables.
    """
    t_min = np.asarray(time_grid_seconds, dtype=float) / 60.0
    a1 = 0.809
    a2 = 0.330
    t1 = 0.17046
    t2 = 0.365
    sigma1 = 0.0563
    sigma2 = 0.132
    alpha = 1.050
    beta = 0.1685
    s = 38.078
    tau = 0.483

    gaussian1 = a1 / (sigma1 * math.sqrt(2.0 * math.pi)) * np.exp(-((t_min - t1) ** 2) / (2.0 * sigma1**2))
    gaussian2 = a2 / (sigma2 * math.sqrt(2.0 * math.pi)) * np.exp(-((t_min - t2) ** 2) / (2.0 * sigma2**2))
    sigmoid = 1.0 / (1.0 + np.exp(-s * (t_min - tau)))
    tail = alpha * np.exp(-beta * t_min) * sigmoid
    cp = gaussian1 + gaussian2 + tail
    return np.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


def _load_fixed_aif_from_csv(aif_csv: Path, time_grid_seconds: Sequence[float]) -> np.ndarray:
    if aif_csv is None:
        raise ValueError("--aif_csv is required when --aif_mode fixed_csv")
    if not Path(aif_csv).exists():
        raise FileNotFoundError(f"AIF CSV not found: {aif_csv}")

    df = pd.read_csv(aif_csv)
    time_col = _find_first_existing_column(
        df,
        [
            "time_seconds",
            "time_sec",
            "seconds",
            "time_s",
            "time_minutes",
            "time_min",
            "minutes",
            "time_m",
        ],
    )
    cp_col = _find_first_existing_column(
        df,
        [
            "Cp",
            "cp",
            "plasma_concentration",
            "aif",
            "aif_concentration",
            "concentration",
        ],
    )
    if time_col is None or cp_col is None:
        raise ValueError("fixed AIF CSV must contain time and Cp columns.")

    time_values = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    cp_values = pd.to_numeric(df[cp_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(cp_values)
    time_values = time_values[valid]
    cp_values = cp_values[valid]
    if time_values.size < 2:
        raise ValueError("fixed AIF CSV must contain at least 2 valid rows.")

    unit_name = str(time_col).strip().lower()
    if "min" in unit_name:
        time_values = time_values * 60.0

    order = np.argsort(time_values)
    time_values = time_values[order]
    cp_values = cp_values[order]
    target = np.asarray(time_grid_seconds, dtype=float)
    interpolated = np.interp(target, time_values, cp_values, left=0.0, right=float(cp_values[-1]))
    return np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


def load_aif_curve(config: dict[str, Any], time_grid_seconds: Sequence[float]) -> tuple[np.ndarray, str]:
    aif_mode = str(config["aif_mode"]).strip().lower()
    if aif_mode == "population":
        return parker_population_aif(time_grid_seconds), "population_parker"
    if aif_mode == "fixed_csv":
        aif_csv = None if config.get("aif_csv") is None else Path(str(config["aif_csv"]))
        return _load_fixed_aif_from_csv(aif_csv, time_grid_seconds), "fixed_csv"
    raise ValueError(f"Unsupported aif_mode: {config['aif_mode']}")


def prepare_curve_records(
    tcc_csv: Path,
    metadata_csv: Path | None,
    pid_col: str,
    test_col: str,
    label_col: str,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    tcc_df = _load_tcc_df(tcc_csv, pid_col=pid_col)
    tcc_df = tcc_df.copy()
    if pid_col != "pid":
        tcc_df = tcc_df.rename(columns={pid_col: "pid"})
    if "roi_type" not in tcc_df.columns:
        raise ValueError("tcc_csv must contain `roi_type` column.")
    if "timepoint_idx" not in tcc_df.columns:
        raise ValueError("tcc_csv must contain `timepoint_idx` column.")
    if "mean_signal" not in tcc_df.columns:
        raise ValueError("tcc_csv must contain `mean_signal` column.")

    metadata_df = _load_metadata_for_pinn(metadata_csv)
    metadata_lookup = _build_metadata_lookup(metadata_df)

    region_df = tcc_df.loc[tcc_df["roi_type"].astype(str).isin(REGION_TYPES)].copy()
    if region_df.empty:
        raise ValueError(f"tcc_csv did not contain any supported ROI rows: {REGION_TYPES}")

    all_original_max_time_seconds = 0.0
    preliminary_records: list[dict[str, Any]] = []

    for (pid, region_type), group in region_df.groupby(["pid", "roi_type"], sort=True):
        pid = str(pid).strip()
        region_type = str(region_type).strip()
        group = group.sort_values("timepoint_idx", kind="stable").reset_index(drop=True)

        metadata_row = metadata_lookup.get(pid)
        split_group, test_value, label_value = _resolve_split_fields(
            pid=pid,
            case_rows=group,
            metadata_row=metadata_row,
            test_col=test_col,
            label_col=label_col,
        )

        discovered_timepoints = pd.to_numeric(group["timepoint_idx"], errors="coerce").astype(int).tolist()
        source_seconds, source_name = _extract_time_axis_from_tcc_rows(group)
        if source_seconds is None:
            source_seconds, source_name = _resolve_time_axis_from_metadata(
                metadata_row=metadata_row,
                discovered_timepoints=discovered_timepoints,
                default_temporal_resolution_seconds=float(config["default_temporal_resolution_seconds"]),
            )
        source_seconds = np.asarray(source_seconds, dtype=float)
        if source_seconds.size >= 1:
            all_original_max_time_seconds = max(all_original_max_time_seconds, float(np.nanmax(source_seconds)))

        signal_curve = pd.to_numeric(group["mean_signal"], errors="coerce").to_numpy(dtype=float)
        baseline_idx, baseline_signal = _resolve_baseline_from_rows(
            region_rows=group,
            baseline_n_pre_points=int(config["baseline_n_pre_points"]),
        )
        concentration_curve, concentration_mode_used, concentration_note, concentration_approximate = signal_to_concentration(
            signal_curve=signal_curve,
            baseline_signal=baseline_signal,
            metadata_row=metadata_row,
            config=config,
        )

        preliminary_records.append({
            "pid": pid,
            "region_type": region_type,
            "group": group,
            "metadata_row": metadata_row,
            "split_group": split_group,
            "test_value": test_value,
            "label_value": label_value,
            "original_timepoint_idx": discovered_timepoints,
            "original_time_seconds": source_seconds,
            "original_signal_curve": signal_curve,
            "original_concentration_curve": concentration_curve,
            "time_alignment_mode": source_name,
            "baseline_timepoint_idx": baseline_idx,
            "baseline_signal": baseline_signal,
            "concentration_mode_used": concentration_mode_used,
            "concentration_conversion_note": concentration_note,
            "concentration_approximate": concentration_approximate,
        })

    if all_original_max_time_seconds <= 0:
        all_original_max_time_seconds = float(config["default_temporal_resolution_seconds"]) * 5.0
    time_grid_seconds = _build_fixed_time_grid(all_original_max_time_seconds, config=config)
    aif_cp, aif_mode_used = load_aif_curve(config=config, time_grid_seconds=time_grid_seconds)

    curve_records: list[CurveRecord] = []
    for item in preliminary_records:
        aligned_signal, aligned_mask = _interpolate_with_nan_edges(
            source_times=np.asarray(item["original_time_seconds"], dtype=float),
            source_values=np.asarray(item["original_signal_curve"], dtype=float),
            target_times=time_grid_seconds,
        )
        aligned_concentration, concentration_mask = _interpolate_with_nan_edges(
            source_times=np.asarray(item["original_time_seconds"], dtype=float),
            source_values=np.asarray(item["original_concentration_curve"], dtype=float),
            target_times=time_grid_seconds,
        )
        valid_mask = (aligned_mask.astype(bool) & concentration_mask.astype(bool)).astype(int)
        valid_points = int(np.sum(valid_mask))
        curve_valid_for_pinn = 1 if valid_points >= 2 else 0
        curve_invalid_reason = "" if curve_valid_for_pinn else "fewer_than_two_valid_aligned_points"

        curve_records.append(
            CurveRecord(
                pid=str(item["pid"]),
                region_type=str(item["region_type"]),
                original_timepoint_idx=[int(v) for v in item["original_timepoint_idx"]],
                original_time_seconds=[_finite_float(v) for v in item["original_time_seconds"]],
                original_signal_curve=[_finite_float(v) for v in item["original_signal_curve"]],
                original_concentration_curve=[_finite_float(v) for v in item["original_concentration_curve"]],
                aligned_time_grid_seconds=[_finite_float(v) for v in time_grid_seconds],
                aligned_signal_curve=[_finite_float(v, default=np.nan) for v in aligned_signal],
                aligned_concentration_curve=[_finite_float(v, default=np.nan) for v in aligned_concentration],
                aligned_valid_mask=[int(v) for v in valid_mask],
                original_n_timepoints=int(len(item["original_timepoint_idx"])),
                time_alignment_mode=str(item["time_alignment_mode"]),
                temporal_resolution_seconds_used=float(
                    config["time_grid_resolution_seconds"]
                    if config.get("time_grid_resolution_seconds") is not None
                    else config["default_temporal_resolution_seconds"]
                ),
                baseline_timepoint_idx=int(item["baseline_timepoint_idx"]),
                baseline_signal=_finite_float(item["baseline_signal"]),
                concentration_mode_used=str(item["concentration_mode_used"]),
                concentration_conversion_note=str(item["concentration_conversion_note"]),
                concentration_approximate=bool(item["concentration_approximate"]),
                curve_valid_for_pinn=int(curve_valid_for_pinn),
                curve_invalid_reason=str(curve_invalid_reason),
                split_group=item["split_group"],
                test_value=item["test_value"],
                label_value=item["label_value"],
                curve_source="tcc_long_csv",
            )
        )

    aligned_df = pd.DataFrame([record.aligned_row() for record in curve_records]).sort_values(
        ["pid", "region_type"], kind="stable"
    ).reset_index(drop=True)
    concentration_df = pd.DataFrame([record.concentration_row() for record in curve_records]).sort_values(
        ["pid", "region_type"], kind="stable"
    ).reset_index(drop=True)

    valid_regions_by_pid: dict[str, set[str]] = {}
    for record in curve_records:
        if int(record.curve_valid_for_pinn) == 1:
            valid_regions_by_pid.setdefault(record.pid, set()).add(record.region_type)
    eligible_pids = sorted([pid for pid, regions in valid_regions_by_pid.items() if set(REGION_TYPES).issubset(regions)])

    dropped_pid_reasons: dict[str, str] = {}
    all_pids = sorted({record.pid for record in curve_records})
    for pid in all_pids:
        regions = valid_regions_by_pid.get(pid, set())
        if not set(REGION_TYPES).issubset(regions):
            missing = sorted(list(set(REGION_TYPES) - set(regions)))
            dropped_pid_reasons[pid] = f"missing_valid_regions:{','.join(missing)}"

    approximate_any = bool(concentration_df["concentration_approximate"].astype(int).max()) if not concentration_df.empty else True
    time_grid_definition = {
        "grid_kind": "fixed_uniform_seconds",
        "grid_resolution_seconds": float(
            config["time_grid_resolution_seconds"]
            if config.get("time_grid_resolution_seconds") is not None
            else config["default_temporal_resolution_seconds"]
        ),
        "grid_times_seconds": [_finite_float(v) for v in time_grid_seconds.tolist()],
        "grid_max_time_seconds": _finite_float(time_grid_seconds[-1]),
    }

    preproc_summary = {
        "time_grid_definition": time_grid_definition,
        "temporal_resolution_seconds_used": float(time_grid_definition["grid_resolution_seconds"]),
        "concentration_mode_requested": str(config["concentration_mode"]),
        "concentration_mode_used": (
            concentration_df["concentration_mode_used"].mode().iloc[0]
            if not concentration_df.empty
            else str(config["concentration_mode"])
        ),
        "aif_mode_requested": str(config["aif_mode"]),
        "aif_mode_used": str(aif_mode_used),
        "aif_curve_json": json.dumps([_finite_float(v) for v in aif_cp.tolist()]),
        "curve_count_total": int(len(curve_records)),
        "pid_count_total": int(len(all_pids)),
        "eligible_pid_count_for_pinn": int(len(eligible_pids)),
        "eligible_pids": eligible_pids,
        "dropped_pid_reasons": dropped_pid_reasons,
        "approximate_concentration_conversion": bool(approximate_any),
        "note": (
            "Current concentration conversion is intentionally flagged as approximate / weakly quantitative "
            "unless true case-specific SPGR inputs are available for every record."
        ),
    }
    if logger is not None:
        logger.info(
            "Prepared %d ROI curves across %d pids; %d pids have valid core15 + safe_rim curves for PINN.",
            len(curve_records),
            len(all_pids),
            len(eligible_pids),
        )

    return {
        "curve_records": curve_records,
        "aligned_df": aligned_df,
        "concentration_df": concentration_df,
        "time_grid_seconds": np.asarray(time_grid_seconds, dtype=float),
        "aif_cp": np.asarray(aif_cp, dtype=float),
        "aif_mode_used": str(aif_mode_used),
        "eligible_pids": eligible_pids,
        "preproc_summary": preproc_summary,
    }


class _CurveDataset(Dataset):
    def __init__(self, records: Sequence[CurveRecord], curve_scale: float) -> None:
        self.records = list(records)
        self.curve_scale = float(max(EPS, curve_scale))
        self.curves = []
        self.masks = []
        self.targets = []
        self.region_flags = []
        for record in self.records:
            curve = np.asarray(record.aligned_concentration_curve, dtype=float)
            mask = np.asarray(record.aligned_valid_mask, dtype=float)
            filled_curve = np.where(np.isfinite(curve), curve, 0.0)
            filled_curve = np.nan_to_num(filled_curve, nan=0.0, posinf=0.0, neginf=0.0)
            self.curves.append(filled_curve / self.curve_scale)
            self.targets.append(filled_curve)
            self.masks.append(mask)
            self.region_flags.append([1.0 if record.region_type == "safe_rim" else 0.0])

        self.curves = torch.tensor(np.asarray(self.curves, dtype=np.float32))
        self.targets = torch.tensor(np.asarray(self.targets, dtype=np.float32))
        self.masks = torch.tensor(np.asarray(self.masks, dtype=np.float32))
        self.region_flags = torch.tensor(np.asarray(self.region_flags, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "curve": self.curves[index],
            "target": self.targets[index],
            "mask": self.masks[index],
            "region_flag": self.region_flags[index],
        }


class SharedToftsMLP(nn.Module):
    def __init__(self, n_timepoints: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = int(2 * n_timepoints + 1)
        hidden_dim = int(max(8, hidden_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        curve_scaled: torch.Tensor,
        valid_mask: torch.Tensor,
        region_flag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([curve_scaled, valid_mask, region_flag], dim=1)
        raw = self.encoder(x)
        ktrans = F.softplus(raw[:, 0]) + 1e-6
        kep = F.softplus(raw[:, 1]) + 1e-6
        return ktrans, kep


def tofts_forward_torch(
    time_grid_minutes: torch.Tensor,
    aif_cp: torch.Tensor,
    ktrans: torch.Tensor,
    kep: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete differentiable Tofts integral with trapezoidal integration over the fixed grid.
    """
    target_t = time_grid_minutes.view(1, -1, 1)
    source_t = time_grid_minutes.view(1, 1, -1)
    delta = target_t - source_t
    nonnegative = torch.clamp(delta, min=0.0)
    support_mask = (delta >= 0.0).to(dtype=aif_cp.dtype)
    kernel = torch.exp(-kep.view(-1, 1, 1) * nonnegative) * support_mask
    integrand = aif_cp.view(1, 1, -1) * kernel
    ct_hat = ktrans.view(-1, 1) * torch.trapz(integrand, x=time_grid_minutes, dim=-1)
    return ct_hat


def masked_reconstruction_mse(ct_hat: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    diff_sq = (ct_hat - target) ** 2 * valid_mask
    denom = valid_mask.sum(dim=1).clamp_min(1.0)
    return (diff_sq.sum(dim=1) / denom).mean()


def weak_parameter_range_regularizer(
    ktrans: torch.Tensor,
    kep: torch.Tensor,
    soft_upper_ktrans: float,
    soft_upper_kep: float,
) -> torch.Tensor:
    penalty_ktrans = torch.relu(ktrans - float(soft_upper_ktrans)) ** 2
    penalty_kep = torch.relu(kep - float(soft_upper_kep)) ** 2
    return (penalty_ktrans + penalty_kep).mean()


def _split_train_val_records(
    records: Sequence[CurveRecord],
    val_fraction: float,
    random_state: int,
) -> tuple[list[CurveRecord], list[CurveRecord]]:
    if len(records) <= 2 or val_fraction <= 0:
        return list(records), []

    patient_ids = sorted({record.pid for record in records})
    if len(patient_ids) <= 2:
        return list(records), []

    rng = np.random.default_rng(int(random_state))
    patient_ids = list(patient_ids)
    rng.shuffle(patient_ids)
    n_val_patients = int(round(len(patient_ids) * float(val_fraction)))
    n_val_patients = max(1, min(n_val_patients, len(patient_ids) - 1))
    val_pid_set = set(patient_ids[:n_val_patients])

    train_records = [record for record in records if record.pid not in val_pid_set]
    val_records = [record for record in records if record.pid in val_pid_set]
    if not train_records or not val_records:
        return list(records), []
    return train_records, val_records


def _curve_scale_from_records(records: Sequence[CurveRecord]) -> float:
    values: list[float] = []
    for record in records:
        curve = np.asarray(record.aligned_concentration_curve, dtype=float)
        mask = np.asarray(record.aligned_valid_mask, dtype=bool)
        valid = curve[mask & np.isfinite(curve)]
        if valid.size > 0:
            values.extend(np.abs(valid).tolist())
    if not values:
        return 1.0
    percentile = np.percentile(np.asarray(values, dtype=float), 95.0)
    return float(max(1e-4, percentile))


def train_shared_tofts_pinn(
    train_records: Sequence[CurveRecord],
    time_grid_seconds: Sequence[float],
    aif_cp: Sequence[float],
    random_state: int,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    records = [record for record in train_records if int(record.curve_valid_for_pinn) == 1]
    if not records:
        raise ValueError("No valid ROI curves available for PINN training.")

    set_global_seed(int(random_state))
    device = resolve_torch_device(str(config["pinn_device"]))
    train_split_records, val_split_records = _split_train_val_records(
        records=records,
        val_fraction=float(config["pinn_val_fraction"]),
        random_state=int(random_state),
    )
    curve_scale = _curve_scale_from_records(train_split_records or records)
    train_dataset = _CurveDataset(train_split_records or records, curve_scale=curve_scale)
    val_dataset = _CurveDataset(val_split_records, curve_scale=curve_scale) if val_split_records else None

    batch_size = int(max(1, min(int(config["pinn_batch_size"]), len(train_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=max(1, min(batch_size, len(val_dataset))), shuffle=False, drop_last=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    model = SharedToftsMLP(n_timepoints=len(time_grid_seconds), hidden_dim=int(config["pinn_hidden_dim"]))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["pinn_learning_rate"]),
        weight_decay=float(config["pinn_weight_decay"]),
    )
    time_grid_minutes = torch.tensor(np.asarray(time_grid_seconds, dtype=np.float32) / 60.0, device=device)
    aif_cp_tensor = torch.tensor(np.asarray(aif_cp, dtype=np.float32), device=device)

    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = int(max(1, config["pinn_early_stopping_patience"]))
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(config["pinn_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            curve = batch["curve"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            region_flag = batch["region_flag"].to(device)

            optimizer.zero_grad(set_to_none=True)
            ktrans, kep = model(curve, mask, region_flag)
            ct_hat = tofts_forward_torch(time_grid_minutes, aif_cp_tensor, ktrans, kep)
            recon_loss = masked_reconstruction_mse(ct_hat, target, mask)
            reg_loss = weak_parameter_range_regularizer(
                ktrans=ktrans,
                kep=kep,
                soft_upper_ktrans=float(config["pinn_soft_upper_ktrans"]),
                soft_upper_kep=float(config["pinn_soft_upper_kep"]),
            )
            loss = recon_loss + float(config["pinn_lambda_reg"]) * reg_loss
            loss.backward()
            if float(config["pinn_gradient_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["pinn_gradient_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")

        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    curve = batch["curve"].to(device)
                    target = batch["target"].to(device)
                    mask = batch["mask"].to(device)
                    region_flag = batch["region_flag"].to(device)
                    ktrans, kep = model(curve, mask, region_flag)
                    ct_hat = tofts_forward_torch(time_grid_minutes, aif_cp_tensor, ktrans, kep)
                    recon_loss = masked_reconstruction_mse(ct_hat, target, mask)
                    reg_loss = weak_parameter_range_regularizer(
                        ktrans=ktrans,
                        kep=kep,
                        soft_upper_ktrans=float(config["pinn_soft_upper_ktrans"]),
                        soft_upper_kep=float(config["pinn_soft_upper_kep"]),
                    )
                    loss = recon_loss + float(config["pinn_lambda_reg"]) * reg_loss
                    val_losses.append(float(loss.detach().cpu().item()))
            monitored_metric = float(np.mean(val_losses)) if val_losses else train_loss
        else:
            monitored_metric = train_loss

        history_rows.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "monitored_loss": float(monitored_metric),
        })

        if monitored_metric + 1e-10 < best_metric:
            best_metric = monitored_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if logger is not None and (epoch == 1 or epoch % 25 == 0 or epoch == int(config["pinn_epochs"])):
            logger.info(
                "PINN epoch %d/%d | train_loss=%.6f monitored_loss=%.6f",
                epoch,
                int(config["pinn_epochs"]),
                float(train_loss),
                float(monitored_metric),
            )

        if patience_counter >= patience:
            if logger is not None:
                logger.info("PINN early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    model.load_state_dict(best_state)
    fit_summary = {
        "train_record_count": int(len(records)),
        "train_patient_count": int(len({record.pid for record in records})),
        "train_split_record_count": int(len(train_split_records or records)),
        "val_split_record_count": int(0 if val_split_records is None else len(val_split_records)),
        "curve_scale": float(curve_scale),
        "best_monitored_loss": float(best_metric),
        "epochs_trained": int(len(history_rows)),
        "pinn_architecture": "shared_region_mlp(curve+mask+region_flag)->[Ktrans,kep]",
        "pinn_loss_definition": (
            "total_loss = masked_recon_mse(Ct_hat, Ct) + lambda_reg * weak_parameter_range_regularizer"
        ),
        "pinn_hyperparameters": {
            "hidden_dim": int(config["pinn_hidden_dim"]),
            "epochs_max": int(config["pinn_epochs"]),
            "batch_size": int(config["pinn_batch_size"]),
            "learning_rate": float(config["pinn_learning_rate"]),
            "weight_decay": float(config["pinn_weight_decay"]),
            "early_stopping_patience": int(config["pinn_early_stopping_patience"]),
            "val_fraction": float(config["pinn_val_fraction"]),
            "gradient_clip": float(config["pinn_gradient_clip"]),
            "lambda_reg": float(config["pinn_lambda_reg"]),
            "soft_upper_ktrans": float(config["pinn_soft_upper_ktrans"]),
            "soft_upper_kep": float(config["pinn_soft_upper_kep"]),
        },
        "training_history_tail": history_rows[-10:],
        "device_used": str(device),
    }
    return {
        "model": model,
        "device": device,
        "curve_scale": float(curve_scale),
        "fit_summary": fit_summary,
    }


def infer_tofts_pinn_features(
    model: SharedToftsMLP,
    records: Sequence[CurveRecord],
    time_grid_seconds: Sequence[float],
    aif_cp: Sequence[float],
    curve_scale: float,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_records = [record for record in records if int(record.curve_valid_for_pinn) == 1]
    if not valid_records:
        long_columns = [
            "pid",
            "region_type",
            "Ktrans",
            "kep",
            "recon_mse",
            "Ct_hat_curve_json",
            "curve_valid_for_pinn",
        ]
        return pd.DataFrame(columns=long_columns), pd.DataFrame(columns=["pid"])

    dataset = _CurveDataset(valid_records, curve_scale=float(curve_scale))
    loader = DataLoader(dataset, batch_size=max(1, min(64, len(dataset))), shuffle=False, drop_last=False)
    model = model.to(device)
    model.eval()

    time_grid_minutes = torch.tensor(np.asarray(time_grid_seconds, dtype=np.float32) / 60.0, device=device)
    aif_cp_tensor = torch.tensor(np.asarray(aif_cp, dtype=np.float32), device=device)

    long_rows: list[dict[str, Any]] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            curve = batch["curve"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            region_flag = batch["region_flag"].to(device)

            ktrans, kep = model(curve, mask, region_flag)
            ct_hat = tofts_forward_torch(time_grid_minutes, aif_cp_tensor, ktrans, kep)

            diff_sq = ((ct_hat - target) ** 2 * mask).detach().cpu().numpy()
            denom = np.maximum(mask.detach().cpu().numpy().sum(axis=1), 1.0)
            recon_mse = diff_sq.sum(axis=1) / denom

            ktrans_np = ktrans.detach().cpu().numpy()
            kep_np = kep.detach().cpu().numpy()
            ct_hat_np = ct_hat.detach().cpu().numpy()

            batch_size = curve.shape[0]
            for batch_index in range(batch_size):
                record = valid_records[cursor + batch_index]
                long_rows.append({
                    "pid": record.pid,
                    "region_type": record.region_type,
                    "Ktrans": _finite_float(ktrans_np[batch_index]),
                    "kep": _finite_float(kep_np[batch_index]),
                    "recon_mse": _finite_float(recon_mse[batch_index]),
                    "Ct_hat_curve_json": json.dumps([_finite_float(v) for v in ct_hat_np[batch_index].tolist()]),
                    "curve_valid_for_pinn": 1,
                    "baseline_signal": _finite_float(record.baseline_signal),
                    "concentration_mode_used": record.concentration_mode_used,
                    "time_alignment_mode": record.time_alignment_mode,
                    "temporal_resolution_seconds_used": _finite_float(record.temporal_resolution_seconds_used),
                    "split_group": record.split_group,
                    "test": record.test_value,
                    "pCR": record.label_value,
                })
            cursor += batch_size

    long_df = pd.DataFrame(long_rows).sort_values(["pid", "region_type"], kind="stable").reset_index(drop=True)

    wide_parts: list[pd.DataFrame] = []
    pivot_specs = {
        "Ktrans": "Ktrans",
        "kep": "kep",
        "recon_mse": "recon_mse",
        "Ct_hat_curve_json": "Ct_hat_curve_json",
        "curve_valid_for_pinn": "pinn_feature_valid",
    }
    for source_col, suffix in pivot_specs.items():
        pivot = long_df.pivot(index="pid", columns="region_type", values=source_col)
        pivot = pivot.rename(columns={region: f"{region}_{suffix}" for region in pivot.columns})
        wide_parts.append(pivot)

    wide_df = pd.concat(wide_parts, axis=1).reset_index()
    for region in REGION_TYPES:
        for suffix in ["Ktrans", "kep", "recon_mse", "Ct_hat_curve_json", "pinn_feature_valid"]:
            col = f"{region}_{suffix}"
            if col not in wide_df.columns:
                wide_df[col] = np.nan if suffix != "feature_valid" else 0

    wide_df["pinn_feature_valid"] = (
        pd.to_numeric(wide_df["core15_pinn_feature_valid"], errors="coerce").fillna(0).astype(int)
        * pd.to_numeric(wide_df["safe_rim_pinn_feature_valid"], errors="coerce").fillna(0).astype(int)
    ).astype(int)
    wide_df = wide_df.sort_values("pid", kind="stable").reset_index(drop=True)
    return long_df, wide_df


def train_pinn_and_infer_for_pid_sets(
    curve_records: Sequence[CurveRecord],
    train_pids: Sequence[str],
    infer_pids: Sequence[str],
    time_grid_seconds: Sequence[float],
    aif_cp: Sequence[float],
    random_state: int,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    train_pid_set = {str(pid) for pid in train_pids}
    infer_pid_set = {str(pid) for pid in infer_pids}
    train_records = [record for record in curve_records if record.pid in train_pid_set]
    infer_records = [record for record in curve_records if record.pid in infer_pid_set]

    fit_result = train_shared_tofts_pinn(
        train_records=train_records,
        time_grid_seconds=time_grid_seconds,
        aif_cp=aif_cp,
        random_state=int(random_state),
        config=config,
        logger=logger,
    )
    long_df, wide_df = infer_tofts_pinn_features(
        model=fit_result["model"],
        records=infer_records,
        time_grid_seconds=time_grid_seconds,
        aif_cp=aif_cp,
        curve_scale=float(fit_result["curve_scale"]),
        device=fit_result["device"],
    )
    return {
        "feature_long_df": long_df,
        "feature_wide_df": wide_df,
        "fit_summary": fit_result["fit_summary"],
        "curve_scale": float(fit_result["curve_scale"]),
        "device_used": str(fit_result["device"]),
        "model": fit_result["model"],
    }


def save_curve_artifacts(bundle: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle["aligned_df"].to_csv(output_dir / "aligned_tcc_curves.csv", index=False)
    bundle["concentration_df"].to_csv(output_dir / "concentration_curves.csv", index=False)


def merge_case_level_pinn_features(
    base_df: pd.DataFrame,
    pinn_wide_df: pd.DataFrame,
    pid_col: str,
) -> pd.DataFrame:
    if pid_col != "pid":
        pinn_wide_df = pinn_wide_df.rename(columns={"pid": pid_col})
    merged = base_df.merge(pinn_wide_df, on=pid_col, how="left")
    return add_tofts_derived_feature_columns(merged)


def build_final_tofts_feature_table(
    base_case_df: pd.DataFrame,
    pinn_wide_df: pd.DataFrame,
    pid_col: str,
    test_col: str,
    label_col: str,
) -> pd.DataFrame:
    merged = merge_case_level_pinn_features(base_case_df.copy(), pinn_wide_df.copy(), pid_col=pid_col)

    output_cols = [pid_col]
    output_cols.extend(TOFTS_EXPORT_FEATURE_COLS)

    for region in REGION_TYPES:
        output_cols.extend([
            f"{region}_Ct_hat_curve_json",
            f"{region}_pinn_feature_valid",
        ])
    output_cols.extend(["pinn_feature_valid"])

    for optional_col in [test_col, label_col]:
        if optional_col in merged.columns and optional_col not in output_cols:
            output_cols.append(optional_col)

    keep_cols = [col for col in output_cols if col in merged.columns]
    out = merged[keep_cols].copy()
    if test_col in out.columns:
        test_series = pd.to_numeric(out[test_col], errors="coerce")
        out["split_group"] = np.where(test_series.isin(list(HELDOUT_TEST_VALUES)), "heldout_test", "development")
    elif label_col in out.columns:
        out["split_group"] = np.where(out[label_col].notna(), "development", None)
    return out.sort_values(pid_col, kind="stable").reset_index(drop=True)
