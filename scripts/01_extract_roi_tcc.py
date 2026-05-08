#!/usr/bin/env python3
"""
Step 1: Extract per-ROI tumor volume and TCC (time-concentration curves) from
predicted tumor masks and raw multi-timepoint DCE-MRI.

Inputs (all paths are user-provided via CLI; nothing is hardcoded):
- raw DCE root with per-case multi-timepoint NIfTI volumes
- metadata CSV with one row per case
- a directory of predicted tumor masks aligned to the raw DCE geometry

Outputs (under --output_dir):
- volume_summary.csv                  : per-case tumor / core15 / safe_rim volumes
- tcc_long.csv                        : per-case per-ROI per-timepoint mean signal
- dynamics_feature_summary.csv        : per-case simple-dynamics summary features
- qc_report.csv                       : per-case QC flags, warnings, geometry status
- case_json/<pid>.json                : per-case structured detail
- run.log                             : log file
- (optional) qc_plots/<pid>.png       : QC overlays and TCC plots

Design rules:
- masks must already be aligned in shape, spacing, orientation, and affine to the
  reference DCE; mismatches are reported and never silently resampled
- nearest-neighbor resampling is only available behind an explicit opt-in flag
- this repository does NOT include nnU-Net code, weights, training data, or
  any real predicted masks. Users must supply their own predicted masks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running this script directly without installing the project.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geometry_utils import (
    compare_geometry,
    get_nifti_summary,
    get_spacing,
    load_nifti,
    resample_mask_to_reference,
    summarize_geometry_checks,
)
from io_utils import (
    apply_subset_filter,
    build_mask_index,
    configure_logging,
    discover_dce_timepoints,
    load_case_map_csv,
    load_metadata_csv,
    resolve_mask_path_for_pid,
    save_overlay_plot,
    save_tcc_plot,
    write_case_json,
    write_tables,
)
from roi_utils import (
    add_normalized_curve_fields,
    build_core_mask_by_fraction,
    build_safe_rim_mask_by_fraction,
    clean_mask,
    compute_simple_dynamics,
    compute_volume_metrics,
    compute_volume_ratio,
    extract_curve_array_fields,
    prefix_dynamics_fields,
    summarize_roi_signal,
)

CORE_FRACTION = 0.15
DEFAULT_UNCERTAIN_BOUNDARY_FRACTION = 0.10
DEFAULT_SAFE_RIM_FRACTION_AFTER_TRIM = 0.20


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: extract per-ROI tumor volume and TCC from predicted tumor "
            "masks and raw multi-timepoint DCE-MRI."
        )
    )
    parser.add_argument("--full_root", type=Path, required=True, help="Root of the raw multi-timepoint DCE dataset.")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Path to metadata CSV (one row per case).")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Directory of predicted tumor masks (NIfTI).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--case_map_csv", type=Path, default=None, help="Optional case_id -> pid mapping CSV.")
    parser.add_argument("--subset_csv", type=Path, default=None, help="Optional subset CSV.")
    parser.add_argument("--subset_name", type=str, default=None, help="Optional subset name (e.g. train/val/test).")
    parser.add_argument(
        "--allow_resample_mask_to_reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, resample mismatched mask to reference DCE with nearest-neighbor interpolation.",
    )
    parser.add_argument("--spacing_atol", type=float, default=1e-6, help="Absolute tolerance for spacing comparison.")
    parser.add_argument("--affine_atol", type=float, default=1e-3, help="Absolute tolerance for affine comparison.")
    parser.add_argument(
        "--keep_largest_component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only largest connected component of predicted mask.",
    )
    parser.add_argument(
        "--save_qc_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save TCC and overlay QC plots.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop at the first failing case instead of skipping and continuing.",
    )
    parser.add_argument(
        "--uncertain_boundary_fraction",
        type=float,
        default=DEFAULT_UNCERTAIN_BOUNDARY_FRACTION,
        help="Version 1.5 safe-rim: fraction of outer tumor voxels to trim as uncertain shell.",
    )
    parser.add_argument(
        "--safe_rim_fraction_after_trim",
        type=float,
        default=DEFAULT_SAFE_RIM_FRACTION_AFTER_TRIM,
        help="Version 1.5 safe-rim: fraction of remaining tumor voxels to keep as inner rim.",
    )
    parser.add_argument(
        "--safe_rim_min_voxel_warning",
        type=int,
        default=5,
        help="Warn when safe_rim contains fewer than this many voxels.",
    )
    return parser


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.nan_to_num(float(value), nan=default, posinf=default, neginf=default))
    except Exception:
        return float(default)


def _value_or_none(row: pd.Series, key: str, cast: type | None = None) -> Any:
    if key not in row.index:
        return None
    value = row.get(key)
    if pd.isna(value):
        return None
    if cast is None:
        return value
    try:
        return cast(value)
    except Exception:
        return value


def _safe_int_from_row(
    row: pd.Series,
    column: str,
    fallback: int,
    pid: str,
    warnings: list[str],
    logger,
) -> int:
    value = row.get(column, np.nan)
    if pd.isna(value):
        msg = f"{pid}: metadata missing `{column}`, fallback to {fallback}"
        logger.info(msg)
        warnings.append(msg)
        return int(fallback)
    try:
        return int(value)
    except Exception:
        msg = f"{pid}: metadata `{column}`={value!r} not parseable, fallback to {fallback}"
        logger.info(msg)
        warnings.append(msg)
        return int(fallback)


def _resolve_selected_timepoints(
    row: pd.Series,
    discovered_timepoints: list[int],
    pid: str,
    warnings: list[str],
    logger,
) -> dict[str, int]:
    discovered_timepoints = sorted(int(v) for v in discovered_timepoints)
    min_tp = min(discovered_timepoints)
    max_tp = max(discovered_timepoints)

    pre_idx = _safe_int_from_row(row, "pre", fallback=0, pid=pid, warnings=warnings, logger=logger)
    post_early_default = 1 if max_tp >= 1 else min_tp
    post_early_idx = _safe_int_from_row(
        row, "post_early", fallback=post_early_default, pid=pid, warnings=warnings, logger=logger
    )
    post_late_default = max_tp
    post_late_idx = _safe_int_from_row(
        row, "post_late", fallback=post_late_default, pid=pid, warnings=warnings, logger=logger
    )

    selected = {
        "pre": pre_idx,
        "post_early": post_early_idx,
        "post_late": post_late_idx,
    }

    for key, idx in list(selected.items()):
        if idx not in discovered_timepoints:
            fallback = discovered_timepoints[0] if key == "pre" else (
                discovered_timepoints[1] if (key == "post_early" and len(discovered_timepoints) > 1) else discovered_timepoints[-1]
            )
            msg = (
                f"{pid}: selected metadata timepoint `{key}`={idx} not present in discovered raw DCE timepoints "
                f"{discovered_timepoints}; fallback to {fallback}"
            )
            logger.warning(msg)
            warnings.append(msg)
            selected[key] = int(fallback)

    return selected


def _extract_case_level_metadata(row: pd.Series) -> dict[str, Any]:
    keys = ["pid", "pCR", "split", "test", "dataset", "n_times", "pre", "post_early", "post_late"]
    out = {}
    for key in keys:
        if key in row.index:
            val = row[key]
            out[key] = None if pd.isna(val) else val
    return out


def _status_from_warnings_and_errors(warnings: list[str], errors: list[str]) -> str:
    if errors:
        return "failed"
    if warnings:
        return "warning"
    return "ok"


def _parse_numeric_sequence(raw_value: Any) -> list[float] | None:
    """Parse a sequence of numbers from JSON/list/CSV-like metadata cells."""
    if raw_value is None or (not isinstance(raw_value, str) and pd.isna(raw_value)):
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
            parts = [p for p in re.split(r"[\s,;|]+", cleaned) if p]
            if not parts:
                return None
            seq = parts
    else:
        return None

    out: list[float] = []
    for item in seq:
        try:
            out.append(float(item))
        except Exception:
            return None
    return out


def _resolve_time_axis(
    row: pd.Series,
    discovered_timepoints: list[int],
    pid: str,
    warnings: list[str],
    logger,
) -> dict[str, Any]:
    """
    Resolve a case-level time axis.

    Preference order:
    1. explicit physical-time list columns in seconds
    2. explicit physical-time list columns in minutes (converted to seconds)
    3. per-timepoint columns in seconds
    4. per-timepoint columns in minutes (converted to seconds)
    5. fallback to raw timepoint indices

    This function is intentionally conservative: it only trusts columns that are
    clearly labeled as seconds/minutes.
    """
    discovered_timepoints = [int(v) for v in sorted(discovered_timepoints)]
    lower_to_original = {str(col).strip().lower(): str(col) for col in row.index}

    def _get_column_value(candidates: list[str]) -> tuple[Any, str | None]:
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is None:
                continue
            value = row.get(original)
            if pd.isna(value):
                continue
            return value, original
        return None, None

    def _validate_monotonic(values_sec: list[float], source_name: str) -> dict[str, Any] | None:
        arr = np.asarray(values_sec, dtype=float)
        if arr.size != len(discovered_timepoints):
            msg = (
                f"{pid}: time-axis source `{source_name}` had {arr.size} values but expected "
                f"{len(discovered_timepoints)}; fallback to timepoint index."
            )
            logger.warning(msg)
            warnings.append(msg)
            return None
        if np.any(~np.isfinite(arr)):
            msg = f"{pid}: time-axis source `{source_name}` contained non-finite values; fallback to timepoint index."
            logger.warning(msg)
            warnings.append(msg)
            return None
        if arr.size >= 2 and np.any(np.diff(arr) <= 0):
            msg = f"{pid}: time-axis source `{source_name}` was not strictly increasing; fallback to timepoint index."
            logger.warning(msg)
            warnings.append(msg)
            return None
        return {
            "kind": "physical_time",
            "unit": "seconds",
            "source": source_name,
            "values_by_timepoint_idx": {
                int(tp): float(arr[i]) for i, tp in enumerate(discovered_timepoints)
            },
        }

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
        "time_axis_minutes",
        "time_axis_min",
        "aqc_times_min",
    ]

    raw_value, source_name = _get_column_value(sec_list_candidates)
    if source_name is not None:
        seq = _parse_numeric_sequence(raw_value)
        if seq is not None:
            validated = _validate_monotonic(seq, source_name)
            if validated is not None:
                return validated

    raw_value, source_name = _get_column_value(min_list_candidates)
    if source_name is not None:
        seq = _parse_numeric_sequence(raw_value)
        if seq is not None:
            validated = _validate_monotonic([float(v) * 60.0 for v in seq], source_name)
            if validated is not None:
                return validated

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
        validated = _validate_monotonic(per_tp_sec_values, "per-timepoint *_sec columns")
        if validated is not None:
            return validated

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
        validated = _validate_monotonic(per_tp_min_values, "per-timepoint *_min columns")
        if validated is not None:
            return validated

    return {
        "kind": "timepoint_index",
        "unit": "timepoint_index",
        "source": "timepoint_idx_fallback",
        "values_by_timepoint_idx": {int(tp): float(tp) for tp in discovered_timepoints},
    }


def _append_curve_context_fields(
    rows: list[dict[str, Any]],
    selected_timepoints: dict[str, int],
    n_times: int,
    time_axis_info: dict[str, Any],
) -> list[dict[str, Any]]:
    for row in rows:
        row["pre_idx"] = int(selected_timepoints["pre"])
        row["post_early_idx"] = int(selected_timepoints["post_early"])
        row["post_late_idx"] = int(selected_timepoints["post_late"])
        row["n_times"] = int(n_times)
        row["time_axis_kind"] = str(time_axis_info["kind"])
        row["time_axis_unit"] = str(time_axis_info["unit"])
    return rows


def _build_curve_block(
    rows: list[dict[str, Any]],
    roi_type: str,
    time_axis_info: dict[str, Any],
) -> dict[str, Any]:
    rows_sorted = sorted(rows, key=lambda x: int(x["timepoint_idx"]))
    arrays = extract_curve_array_fields(
        rows_sorted,
        time_axis_lookup=time_axis_info["values_by_timepoint_idx"],
    )
    return {
        "roi_type": roi_type,
        "timepoints": rows_sorted,
        "baseline_definition": "mean signal at metadata pre timepoint (or fallback if unavailable)",
        "signal_type": "raw_signal",
        "normalized_signal_type": ["S/S0", "(S-S0)/S0"],
        "time_axis_kind": str(time_axis_info["kind"]),
        "time_axis_unit": str(time_axis_info["unit"]),
        "time_axis_source": str(time_axis_info["source"]),
        **arrays,
    }


def _build_dynamics_row(
    row: pd.Series,
    pid: str,
    n_times: int,
    time_axis_info: dict[str, Any],
    volumes: dict[str, Any],
    core15_dynamics: dict[str, Any],
    safe_rim_dynamics: dict[str, Any],
    case_status: str,
) -> dict[str, Any]:
    dynamics_row = {
        "pid": pid,
        "pCR": _value_or_none(row, "pCR", int),
        "split": _value_or_none(row, "split", str),
        "test": _value_or_none(row, "test", int),
        "n_times": int(n_times),
        "status": case_status,
        "time_axis_kind": str(time_axis_info["kind"]),
        "time_axis_unit": str(time_axis_info["unit"]),
        "time_axis_source": str(time_axis_info["source"]),
        "tumor_volume_mm3": _finite_float(volumes.get("tumor_volume_mm3", 0.0)),
        "tumor_volume_ml": _finite_float(volumes.get("tumor_volume_ml", 0.0)),
        "core15_volume_mm3": _finite_float(volumes.get("core15_volume_mm3", 0.0)),
        "core15_volume_ml": _finite_float(volumes.get("core15_volume_ml", 0.0)),
        "safe_rim_volume_mm3": _finite_float(volumes.get("safe_rim_volume_mm3", 0.0)),
        "safe_rim_volume_ml": _finite_float(volumes.get("safe_rim_volume_ml", 0.0)),
        "tumor_voxel_count": int(volumes.get("tumor_voxel_count", 0)),
        "core15_voxel_count": int(volumes.get("core15_voxel_count", 0)),
        "safe_rim_voxel_count": int(volumes.get("safe_rim_voxel_count", 0)),
        "safe_rim_to_tumor_volume_ratio": _finite_float(volumes.get("safe_rim_to_tumor_volume_ratio", 0.0)),
        "safe_rim_to_core15_volume_ratio": _finite_float(volumes.get("safe_rim_to_core15_volume_ratio", 0.0)),
    }
    dynamics_row.update(prefix_dynamics_fields(core15_dynamics, "core15"))
    dynamics_row.update(prefix_dynamics_fields(safe_rim_dynamics, "safe_rim"))

    for key, value in list(dynamics_row.items()):
        if isinstance(value, float):
            dynamics_row[key] = _finite_float(value)
    return dynamics_row


def process_one_case(
    row: pd.Series,
    args: argparse.Namespace,
    mask_index: dict[str, Path],
    case_map: dict[str, Any] | None,
    logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    pid = str(row["pid"]).strip()
    warnings: list[str] = []
    errors: list[str] = []
    geometry_comparisons = []
    resampled_mask = False

    case_result: dict[str, Any] = {
        "pid": pid,
        "status": "started",
        "metadata": _extract_case_level_metadata(row),
        "warnings": warnings,
        "errors": errors,
        "geometry": {},
        "volumes": {},
        "curves": {},
        "simple_dynamics": {},
        "safe_rim_definition": {},
        "time_axis": {},
        "concentration_conversion": {
            "status": "not_performed",
            "reason": (
                "This pipeline extracts raw signal curves and signal-normalized curves only. "
                "True pharmacokinetic concentration conversion was intentionally not attempted "
                "because TR / flip angle / baseline T1 / acquisition timing are not guaranteed."
            ),
        },
    }

    try:
        # -------------------------------
        # File discovery
        # -------------------------------
        mask_path = resolve_mask_path_for_pid(pid, mask_index, case_map=case_map)
        dce_pairs = discover_dce_timepoints(args.full_root, pid)
        discovered_timepoints = [idx for idx, _ in dce_pairs]
        dce_map = {idx: path for idx, path in dce_pairs}

        if "n_times" in row.index and not pd.isna(row["n_times"]):
            expected_n_times = int(row["n_times"])
            if expected_n_times != len(dce_pairs):
                msg = (
                    f"{pid}: metadata n_times={expected_n_times} but discovered "
                    f"{len(dce_pairs)} raw DCE files."
                )
                logger.warning(msg)
                warnings.append(msg)

        selected_timepoints = _resolve_selected_timepoints(
            row=row,
            discovered_timepoints=discovered_timepoints,
            pid=pid,
            warnings=warnings,
            logger=logger,
        )
        pre_idx = int(selected_timepoints["pre"])
        time_axis_info = _resolve_time_axis(
            row=row,
            discovered_timepoints=discovered_timepoints,
            pid=pid,
            warnings=warnings,
            logger=logger,
        )

        case_result["input_files"] = {
            "mask_path": str(mask_path),
            "dce_timepoints": {str(idx): str(path) for idx, path in dce_pairs},
        }
        case_result["all_timepoint_indices"] = discovered_timepoints
        case_result["selected_timepoints"] = selected_timepoints
        case_result["time_axis"] = {
            "kind": str(time_axis_info["kind"]),
            "unit": str(time_axis_info["unit"]),
            "source": str(time_axis_info["source"]),
            "values_by_timepoint_idx": {
                str(k): _finite_float(v) for k, v in time_axis_info["values_by_timepoint_idx"].items()
            },
        }

        # -------------------------------
        # Load reference + mask
        # -------------------------------
        reference_path = dce_map[pre_idx]
        reference_img = load_nifti(reference_path)
        reference_summary = get_nifti_summary(reference_img, path=reference_path)

        mask_img = load_nifti(mask_path)
        mask_summary_before = get_nifti_summary(mask_img, path=mask_path)

        mask_vs_ref = compare_geometry(
            left_img=mask_img,
            right_img=reference_img,
            left_name="mask",
            right_name=f"reference_tp_{pre_idx}",
            spacing_atol=args.spacing_atol,
            affine_atol=args.affine_atol,
        )
        geometry_comparisons.append(mask_vs_ref)

        if mask_vs_ref.status == "error":
            if args.allow_resample_mask_to_reference:
                msg = (
                    f"{pid}: mask geometry mismatch with reference tp={pre_idx}; "
                    "resampling mask to reference because "
                    "--allow_resample_mask_to_reference is enabled."
                )
                logger.warning(msg)
                warnings.append(msg)
                mask_img = resample_mask_to_reference(mask_img, reference_img)
                resampled_mask = True

                mask_vs_ref_after = compare_geometry(
                    left_img=mask_img,
                    right_img=reference_img,
                    left_name="mask_resampled",
                    right_name=f"reference_tp_{pre_idx}",
                    spacing_atol=args.spacing_atol,
                    affine_atol=args.affine_atol,
                )
                geometry_comparisons.append(mask_vs_ref_after)
                if mask_vs_ref_after.status == "error":
                    raise RuntimeError(
                        f"{pid}: mask still mismatched after resampling: "
                        f"{mask_vs_ref_after.messages}"
                    )
            else:
                raise RuntimeError(
                    f"{pid}: mask/reference geometry mismatch: {mask_vs_ref.messages}. "
                    "Resampling is disabled by default."
                )
        elif mask_vs_ref.status == "warning":
            warnings.extend([f"{pid}: {msg}" for msg in mask_vs_ref.messages])

        # -------------------------------
        # Build ROIs
        # -------------------------------
        full_rows: list[dict[str, Any]] = []
        core_rows: list[dict[str, Any]] = []
        safe_rim_rows: list[dict[str, Any]] = []

        mask_data = np.asarray(mask_img.dataobj)
        full_mask, clean_info = clean_mask(
            mask_data=mask_data,
            keep_largest_component=bool(args.keep_largest_component),
        )
        if int(full_mask.sum()) == 0:
            raise RuntimeError(f"{pid}: predicted mask is empty after cleaning.")

        spacing = get_spacing(reference_img)
        core_mask, core_info = build_core_mask_by_fraction(
            mask_bool=full_mask,
            spacing=spacing,
            fraction=CORE_FRACTION,
        )
        safe_rim_mask, uncertain_shell_mask, safe_rim_info = build_safe_rim_mask_by_fraction(
            mask_bool=full_mask,
            spacing=spacing,
            uncertain_boundary_fraction=float(args.uncertain_boundary_fraction),
            safe_rim_fraction_after_trim=float(args.safe_rim_fraction_after_trim),
            avoid_mask=core_mask,
            min_voxel_warning_threshold=int(args.safe_rim_min_voxel_warning),
        )

        if safe_rim_info.get("uncertain_shell_enters_safe_rim", False):
            raise RuntimeError(f"{pid}: safe_rim unexpectedly overlaps the uncertain shell.")
        if int(safe_rim_info.get("final_overlap_with_core15_voxel_count", 0)) > 0:
            raise RuntimeError(f"{pid}: safe_rim unexpectedly overlaps core15 after construction.")

        for msg in safe_rim_info.get("warning_messages", []):
            logger.warning("%s: %s", pid, msg)
            warnings.append(f"{pid}: {msg}")

        tumor_voxel_count = int(full_mask.sum())
        core_voxel_count = int(core_mask.sum())
        safe_rim_voxel_count = int(safe_rim_mask.sum())
        if tumor_voxel_count < 10:
            msg = (
                f"{pid}: tumor has very few voxels after cleaning "
                f"(n={tumor_voxel_count}); results may be unstable."
            )
            logger.warning(msg)
            warnings.append(msg)
        if core_voxel_count < 5:
            msg = (
                f"{pid}: core15 has very few voxels "
                f"(n={core_voxel_count}); results may be noisy."
            )
            logger.warning(msg)
            warnings.append(msg)
        if safe_rim_voxel_count == 0:
            msg = (
                f"{pid}: safe_rim is empty after applying strict trim + disjoint constraints; "
                "case-level rim dynamics will be zero-filled and TCC rows will be omitted."
            )
            logger.warning(msg)
            warnings.append(msg)

        tumor_volume = compute_volume_metrics(full_mask, spacing, prefix="tumor")
        core_volume = compute_volume_metrics(core_mask, spacing, prefix="core15")
        safe_rim_volume = compute_volume_metrics(safe_rim_mask, spacing, prefix="safe_rim")
        safe_rim_to_tumor_volume_ratio = compute_volume_ratio(
            safe_rim_volume["safe_rim_volume_mm3"],
            tumor_volume["tumor_volume_mm3"],
        )
        safe_rim_to_core15_volume_ratio = compute_volume_ratio(
            safe_rim_volume["safe_rim_volume_mm3"],
            core_volume["core15_volume_mm3"],
        )
        volumes_flat = {
            **tumor_volume,
            **core_volume,
            **safe_rim_volume,
            "safe_rim_to_tumor_volume_ratio": safe_rim_to_tumor_volume_ratio,
            "safe_rim_to_core15_volume_ratio": safe_rim_to_core15_volume_ratio,
        }

        case_result["volumes"] = {
            **volumes_flat,
            "tumor": {
                "voxel_count": int(tumor_volume["tumor_voxel_count"]),
                "volume_mm3": _finite_float(tumor_volume["tumor_volume_mm3"]),
                "volume_ml": _finite_float(tumor_volume["tumor_volume_ml"]),
            },
            "core15": {
                "voxel_count": int(core_volume["core15_voxel_count"]),
                "volume_mm3": _finite_float(core_volume["core15_volume_mm3"]),
                "volume_ml": _finite_float(core_volume["core15_volume_ml"]),
            },
            "safe_rim": {
                "voxel_count": int(safe_rim_volume["safe_rim_voxel_count"]),
                "volume_mm3": _finite_float(safe_rim_volume["safe_rim_volume_mm3"]),
                "volume_ml": _finite_float(safe_rim_volume["safe_rim_volume_ml"]),
            },
            "ratios": {
                "safe_rim_to_tumor_volume_ratio": _finite_float(safe_rim_to_tumor_volume_ratio),
                "safe_rim_to_core15_volume_ratio": _finite_float(safe_rim_to_core15_volume_ratio),
            },
        }
        case_result["mask_processing"] = {
            "clean_info": clean_info,
            "core15_info": core_info,
            "safe_rim_info": safe_rim_info,
            "keep_largest_component": bool(args.keep_largest_component),
            "core_fraction": CORE_FRACTION,
            "uncertain_boundary_fraction": float(args.uncertain_boundary_fraction),
            "safe_rim_fraction_after_trim": float(args.safe_rim_fraction_after_trim),
        }
        case_result["safe_rim_definition"] = {
            "uncertain_boundary_fraction": float(args.uncertain_boundary_fraction),
            "safe_rim_fraction_after_trim": float(args.safe_rim_fraction_after_trim),
            "uncertain_boundary_voxel_count": int(safe_rim_info.get("uncertain_boundary_voxel_count", 0)),
            "safe_rim_voxel_count": int(safe_rim_info.get("safe_rim_voxel_count", 0)),
            "disjoint_check_status": str(safe_rim_info.get("disjoint_check_status", "not_checked")),
            "warning_messages": list(safe_rim_info.get("warning_messages", [])),
            "requested_uncertain_boundary_voxel_count": int(safe_rim_info.get("requested_uncertain_boundary_voxel_count", 0)),
            "requested_safe_rim_voxel_count": int(safe_rim_info.get("requested_safe_rim_voxel_count", 0)),
            "remaining_voxel_count_after_trim": int(safe_rim_info.get("remaining_voxel_count_after_trim", 0)),
            "definition_adjustments": list(safe_rim_info.get("definition_adjustments", [])),
        }

        # -------------------------------
        # Check all timepoints vs reference and extract curves
        # -------------------------------
        reference_data = None

        for time_idx, time_path in dce_pairs:
            img = reference_img if time_idx == pre_idx else load_nifti(time_path)

            if time_idx != pre_idx:
                cmp_ = compare_geometry(
                    left_img=reference_img,
                    right_img=img,
                    left_name=f"reference_tp_{pre_idx}",
                    right_name=f"timepoint_{time_idx}",
                    spacing_atol=args.spacing_atol,
                    affine_atol=args.affine_atol,
                )
                geometry_comparisons.append(cmp_)
                if cmp_.status == "error":
                    raise RuntimeError(
                        f"{pid}: reference/timepoint geometry mismatch at tp={time_idx}: "
                        f"{cmp_.messages}"
                    )
                if cmp_.status == "warning":
                    warnings.extend([f"{pid}: {msg}" for msg in cmp_.messages])

            image_data = np.asarray(img.dataobj, dtype=np.float32)
            if reference_data is None and time_idx == pre_idx:
                reference_data = image_data

            full_stats = summarize_roi_signal(image_data, full_mask)
            core_stats = summarize_roi_signal(image_data, core_mask)

            common = {
                "pid": pid,
                "timepoint_idx": int(time_idx),
                "is_pre": int(time_idx == selected_timepoints["pre"]),
                "is_post_early": int(time_idx == selected_timepoints["post_early"]),
                "is_post_late": int(time_idx == selected_timepoints["post_late"]),
                "pCR": _value_or_none(row, "pCR", int),
                "split": _value_or_none(row, "split", str),
                "test": _value_or_none(row, "test", int),
                "dataset": _value_or_none(row, "dataset", str),
            }

            full_rows.append({
                **common,
                "roi_type": "full_tumor",
                **full_stats,
            })
            core_rows.append({
                **common,
                "roi_type": "core15",
                **core_stats,
            })
            if safe_rim_voxel_count > 0:
                safe_rim_stats = summarize_roi_signal(image_data, safe_rim_mask)
                safe_rim_rows.append({
                    **common,
                    "roi_type": "safe_rim",
                    **safe_rim_stats,
                })

        full_rows = add_normalized_curve_fields(full_rows, baseline_timepoint_idx=pre_idx)
        core_rows = add_normalized_curve_fields(core_rows, baseline_timepoint_idx=pre_idx)
        safe_rim_rows = add_normalized_curve_fields(safe_rim_rows, baseline_timepoint_idx=pre_idx)

        for rows_out in (full_rows, core_rows, safe_rim_rows):
            _append_curve_context_fields(
                rows_out,
                selected_timepoints=selected_timepoints,
                n_times=len(dce_pairs),
                time_axis_info=time_axis_info,
            )
            for row_out in rows_out:
                row_out["normalized_s_over_s0"] = _finite_float(row_out.get("normalized_s_over_s0", 0.0))
                row_out["normalized_delta_s_over_s0"] = _finite_float(
                    row_out.get("normalized_delta_s_over_s0", 0.0)
                )

        core15_dynamics = compute_simple_dynamics(
            rows=core_rows,
            roi_type="core15",
            pre_idx=int(selected_timepoints["pre"]),
            post_early_idx=int(selected_timepoints["post_early"]),
            post_late_idx=int(selected_timepoints["post_late"]),
            time_axis_kind=str(time_axis_info["kind"]),
            time_axis_lookup=time_axis_info["values_by_timepoint_idx"],
            time_axis_unit=str(time_axis_info["unit"]),
            time_axis_source=str(time_axis_info["source"]),
        )
        safe_rim_dynamics = compute_simple_dynamics(
            rows=safe_rim_rows,
            roi_type="safe_rim",
            pre_idx=int(selected_timepoints["pre"]),
            post_early_idx=int(selected_timepoints["post_early"]),
            post_late_idx=int(selected_timepoints["post_late"]),
            time_axis_kind=str(time_axis_info["kind"]),
            time_axis_lookup=time_axis_info["values_by_timepoint_idx"],
            time_axis_unit=str(time_axis_info["unit"]),
            time_axis_source=str(time_axis_info["source"]),
        )

        case_result["reference_image"] = reference_summary
        case_result["mask_image_before_optional_resample"] = mask_summary_before
        case_result["mask_resampled_to_reference"] = bool(resampled_mask)
        case_result["geometry"] = summarize_geometry_checks(geometry_comparisons)
        case_result["curves"] = {
            "full_tumor": _build_curve_block(full_rows, "full_tumor", time_axis_info),
            "core15": _build_curve_block(core_rows, "core15", time_axis_info),
            "safe_rim": _build_curve_block(safe_rim_rows, "safe_rim", time_axis_info),
            "future_fixed_length_hook": {
                "available": True,
                "description": (
                    "See roi_utils.interpolate_curve_to_target_times(). "
                    "Interpolation is intentionally not applied by default."
                ),
            },
        }
        case_result["simple_dynamics"] = {
            "core15": core15_dynamics,
            "safe_rim": safe_rim_dynamics,
        }

        if args.save_qc_plots:
            qc_plot_dir = args.output_dir / "qc_plots"
            save_tcc_plot(
                pid,
                full_rows,
                core_rows,
                qc_plot_dir / f"{pid}_tcc.png",
                safe_rim_rows=safe_rim_rows,
            )
            if reference_data is not None:
                save_overlay_plot(
                    pid=pid,
                    reference_data=reference_data,
                    full_mask=full_mask,
                    core_mask=core_mask,
                    safe_rim_mask=safe_rim_mask,
                    reference_timepoint_idx=pre_idx,
                    out_path=qc_plot_dir / f"{pid}_overlay_mid_slice.png",
                )

        status = _status_from_warnings_and_errors(warnings, errors)
        case_result["status"] = status

        volume_row = {
            "pid": pid,
            "tumor_voxel_count": volumes_flat["tumor_voxel_count"],
            "tumor_volume_mm3": volumes_flat["tumor_volume_mm3"],
            "tumor_volume_ml": volumes_flat["tumor_volume_ml"],
            "core15_voxel_count": volumes_flat["core15_voxel_count"],
            "core15_volume_mm3": volumes_flat["core15_volume_mm3"],
            "core15_volume_ml": volumes_flat["core15_volume_ml"],
            "n_times": len(dce_pairs),
            "pCR": _value_or_none(row, "pCR", int),
            "split": _value_or_none(row, "split", str),
            "test": _value_or_none(row, "test", int),
            "status": status,
            "warning_count": len(warnings),
            "error_message": "",
            "safe_rim_voxel_count": volumes_flat["safe_rim_voxel_count"],
            "safe_rim_volume_mm3": volumes_flat["safe_rim_volume_mm3"],
            "safe_rim_volume_ml": volumes_flat["safe_rim_volume_ml"],
            "safe_rim_to_tumor_volume_ratio": volumes_flat["safe_rim_to_tumor_volume_ratio"],
            "safe_rim_to_core15_volume_ratio": volumes_flat["safe_rim_to_core15_volume_ratio"],
        }

        qc_row = {
            "pid": pid,
            "status": status,
            "warning_count": len(warnings),
            "error_count": len(errors),
            "warnings": " | ".join(warnings),
            "errors": "",
            "mask_path": str(mask_path),
            "reference_path": str(reference_path),
            "mask_resampled_to_reference": int(resampled_mask),
            "geometry_overall_status": case_result["geometry"]["overall_status"],
            "safe_rim_status": str(safe_rim_info.get("disjoint_check_status", "not_checked")),
            "safe_rim_warning_messages": " | ".join(safe_rim_info.get("warning_messages", [])),
            "safe_rim_voxel_count": int(volumes_flat["safe_rim_voxel_count"]),
            "uncertain_boundary_voxel_count": int(safe_rim_info.get("uncertain_boundary_voxel_count", 0)),
        }

        dynamics_row = _build_dynamics_row(
            row=row,
            pid=pid,
            n_times=len(dce_pairs),
            time_axis_info=time_axis_info,
            volumes=volumes_flat,
            core15_dynamics=core15_dynamics,
            safe_rim_dynamics=safe_rim_dynamics,
            case_status=status,
        )

        return volume_row, full_rows + core_rows + safe_rim_rows, qc_row, case_result, dynamics_row

    except Exception as exc:
        errors.append(str(exc))
        case_result["status"] = "failed"
        case_result["geometry"] = summarize_geometry_checks(geometry_comparisons)
        case_result["exception_traceback"] = traceback.format_exc()

        volume_row = {
            "pid": pid,
            "tumor_voxel_count": np.nan,
            "tumor_volume_mm3": np.nan,
            "tumor_volume_ml": np.nan,
            "core15_voxel_count": np.nan,
            "core15_volume_mm3": np.nan,
            "core15_volume_ml": np.nan,
            "n_times": _value_or_none(row, "n_times", int),
            "pCR": _value_or_none(row, "pCR", int),
            "split": _value_or_none(row, "split", str),
            "test": _value_or_none(row, "test", int),
            "status": "failed",
            "warning_count": len(warnings),
            "error_message": str(exc),
            "safe_rim_voxel_count": np.nan,
            "safe_rim_volume_mm3": np.nan,
            "safe_rim_volume_ml": np.nan,
            "safe_rim_to_tumor_volume_ratio": np.nan,
            "safe_rim_to_core15_volume_ratio": np.nan,
        }

        qc_row = {
            "pid": pid,
            "status": "failed",
            "warning_count": len(warnings),
            "error_count": len(errors),
            "warnings": " | ".join(warnings),
            "errors": " | ".join(errors),
            "mask_path": case_result.get("input_files", {}).get("mask_path", ""),
            "reference_path": "",
            "mask_resampled_to_reference": int(resampled_mask),
            "geometry_overall_status": case_result["geometry"].get("overall_status", "not_checked"),
            "safe_rim_status": "not_available_due_to_case_failure",
            "safe_rim_warning_messages": "",
            "safe_rim_voxel_count": np.nan,
            "uncertain_boundary_voxel_count": np.nan,
        }

        return volume_row, [], qc_row, case_result, None


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "case_json").mkdir(parents=True, exist_ok=True)
    if args.save_qc_plots:
        (args.output_dir / "qc_plots").mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.output_dir)
    logger.info("Loading metadata: %s", args.metadata_csv)
    metadata_df = load_metadata_csv(args.metadata_csv)

    case_map = load_case_map_csv(args.case_map_csv) if args.case_map_csv is not None else None
    metadata_df = apply_subset_filter(metadata_df, args.subset_csv, args.subset_name, logger)

    logger.info("Number of cases after filtering: %d", len(metadata_df))
    mask_index = build_mask_index(args.mask_dir, case_map=case_map)

    volume_rows: list[dict[str, Any]] = []
    tcc_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []

    for _, row in metadata_df.iterrows():
        pid = str(row["pid"]).strip()
        logger.info("Processing %s", pid)

        volume_row, case_tcc_rows, qc_row, case_result, dynamics_row = process_one_case(
            row=row,
            args=args,
            mask_index=mask_index,
            case_map=case_map,
            logger=logger,
        )

        volume_rows.append(volume_row)
        tcc_rows.extend(case_tcc_rows)
        qc_rows.append(qc_row)
        if dynamics_row is not None:
            dynamics_rows.append(dynamics_row)

        write_case_json(case_result, args.output_dir / "case_json" / f"{pid}.json")

        if volume_row["status"] == "failed":
            logger.error("FAILED %s | %s", pid, volume_row["error_message"])
            if args.strict:
                write_tables(volume_rows, tcc_rows, qc_rows, args.output_dir, dynamics_rows=dynamics_rows)
                raise RuntimeError(f"Stopping in strict mode because case {pid} failed.")
        else:
            logger.info("DONE %s | status=%s", pid, volume_row["status"])

    write_tables(volume_rows, tcc_rows, qc_rows, args.output_dir, dynamics_rows=dynamics_rows)
    logger.info("Finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
