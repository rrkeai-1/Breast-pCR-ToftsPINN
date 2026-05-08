"""
ROI utilities for tumor volume computation, core-15 construction, safe-rim
construction, and simple DCE dynamics summarization.

Core-15 definition implemented here:
1. binarize mask (>0)
2. optionally keep only largest connected component
3. compute Euclidean distance transform inside tumor, with physical spacing
4. sort tumor voxels by distance-to-boundary descending
5. select the top ceil(0.15 * tumor_voxel_count), minimum 1 voxel

Version 1.5 safe-rim definition implemented here:
1. compute EDT inside the cleaned tumor mask, with physical spacing
2. sort tumor voxels by distance-to-boundary ascending (outer -> inner)
3. trim an uncertain boundary shell from the outermost ceil(0.10 * N) voxels
4. from the remaining voxels, take the outermost ceil(0.20 * remaining) voxels
5. explicitly enforce disjointness with core15 when an avoid-mask is supplied
"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a JSON/CSV-safe finite float."""
    try:
        return float(np.nan_to_num(float(value), nan=default, posinf=default, neginf=default))
    except Exception:
        return float(default)


def _finite_int(value: Any, default: int = 0) -> int:
    """Return a JSON/CSV-safe finite int."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe finite scalar division."""
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return float(default)
    return _finite_float(float(numerator) / denominator, default=default)


def binarize_mask(mask_data: np.ndarray) -> np.ndarray:
    """Convert any numeric mask array to a boolean tumor mask using > 0."""
    return np.asarray(mask_data) > 0


def keep_largest_connected_component(mask_bool: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Keep the largest connected component of a 3D binary mask.

    Connectivity is maximized for 3D volumes to avoid dropping diagonally
    connected tumor voxels.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={mask_bool.shape}")

    voxel_count = int(mask_bool.sum())
    if voxel_count == 0:
        return mask_bool.copy(), {
            "n_input_voxels": 0,
            "n_components": 0,
            "n_output_voxels": 0,
            "dropped_voxels": 0,
        }

    structure = ndi.generate_binary_structure(mask_bool.ndim, mask_bool.ndim)
    labeled, n_components = ndi.label(mask_bool, structure=structure)

    if n_components <= 1:
        return mask_bool.copy(), {
            "n_input_voxels": voxel_count,
            "n_components": int(n_components),
            "n_output_voxels": voxel_count,
            "dropped_voxels": 0,
        }

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    largest_label = int(np.argmax(counts))
    largest = labeled == largest_label

    output_voxels = int(largest.sum())
    return largest, {
        "n_input_voxels": voxel_count,
        "n_components": int(n_components),
        "n_output_voxels": output_voxels,
        "dropped_voxels": int(voxel_count - output_voxels),
    }


def clean_mask(
    mask_data: np.ndarray,
    keep_largest_component: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Binarize and optionally keep only the largest connected component.
    """
    mask_bool = binarize_mask(mask_data)
    info: dict[str, Any] = {
        "initial_voxel_count": int(mask_bool.sum()),
        "keep_largest_component": bool(keep_largest_component),
    }

    if keep_largest_component:
        cleaned, cc_info = keep_largest_connected_component(mask_bool)
        info["largest_component"] = cc_info
        info["final_voxel_count"] = int(cleaned.sum())
        return cleaned, info

    info["final_voxel_count"] = int(mask_bool.sum())
    return mask_bool, info


def compute_edt_distances(mask_bool: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """
    Compute physical Euclidean distance-to-boundary inside the tumor mask.

    Important:
    `sampling=spacing` makes the output distances physically meaningful (mm),
    which is critical for anisotropic voxels.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if mask_bool.sum() == 0:
        raise ValueError("Cannot compute EDT for an empty mask.")
    return ndi.distance_transform_edt(mask_bool, sampling=spacing)


def build_core_mask_by_fraction(
    mask_bool: np.ndarray,
    spacing: tuple[float, float, float],
    fraction: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build the 'core' mask as the top fraction of tumor voxels with the largest
    EDT distance-to-boundary values.

    Exact voxel count:
        ceil(fraction * tumor_voxel_count), minimum 1
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    mask_bool = np.asarray(mask_bool, dtype=bool)
    tumor_voxel_count = int(mask_bool.sum())
    if tumor_voxel_count == 0:
        raise ValueError("Cannot build core mask from an empty tumor mask.")

    edt = compute_edt_distances(mask_bool, spacing)
    target_voxels = max(1, int(ceil(fraction * tumor_voxel_count)))

    tumor_flat_idx = np.flatnonzero(mask_bool.ravel())
    tumor_distances = edt.ravel()[tumor_flat_idx]

    # Stable, reproducible exact selection:
    # primary key   = -distance (descending)
    # secondary key = flat index (ascending)
    order = np.lexsort((tumor_flat_idx, -tumor_distances))
    selected_flat_idx = tumor_flat_idx[order[:target_voxels]]

    core_flat = np.zeros(mask_bool.size, dtype=bool)
    core_flat[selected_flat_idx] = True
    core_mask = core_flat.reshape(mask_bool.shape)

    selected_distances = edt.ravel()[selected_flat_idx]
    info = {
        "fraction": float(fraction),
        "tumor_voxel_count": tumor_voxel_count,
        "target_voxel_count": target_voxels,
        "selected_voxel_count": int(core_mask.sum()),
        "max_distance_mm": _finite_float(np.max(tumor_distances)),
        "min_selected_distance_mm": _finite_float(np.min(selected_distances)),
        "mean_selected_distance_mm": _finite_float(np.mean(selected_distances)),
    }
    return core_mask, info


def build_safe_rim_mask_by_fraction(
    mask_bool: np.ndarray,
    spacing: tuple[float, float, float],
    uncertain_boundary_fraction: float = 0.10,
    safe_rim_fraction_after_trim: float = 0.20,
    avoid_mask: np.ndarray | None = None,
    min_voxel_warning_threshold: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build the Version 1.5 safe-rim mask.

    Steps:
    1. Rank tumor voxels from boundary -> center using spacing-aware EDT.
    2. Remove an outer uncertain shell.
    3. Take the next outer subset as the safe-rim.
    4. If `avoid_mask` is given, exclude those voxels to guarantee disjointness
       and top up from deeper eligible voxels when available.

    Returns
    -------
    safe_rim_mask, uncertain_shell_mask, info
    """
    if not (0.0 <= uncertain_boundary_fraction < 1.0):
        raise ValueError(
            "uncertain_boundary_fraction must be in [0, 1). "
            f"Got {uncertain_boundary_fraction}."
        )
    if not (0.0 < safe_rim_fraction_after_trim <= 1.0):
        raise ValueError(
            "safe_rim_fraction_after_trim must be in (0, 1]. "
            f"Got {safe_rim_fraction_after_trim}."
        )

    mask_bool = np.asarray(mask_bool, dtype=bool)
    tumor_voxel_count = int(mask_bool.sum())
    if tumor_voxel_count == 0:
        raise ValueError("Cannot build safe-rim mask from an empty tumor mask.")

    edt = compute_edt_distances(mask_bool, spacing)
    tumor_flat_idx = np.flatnonzero(mask_bool.ravel())
    tumor_distances = edt.ravel()[tumor_flat_idx]

    # Stable outer -> inner ordering:
    # primary key   = +distance (ascending)
    # secondary key = flat index (ascending)
    order_outer_to_inner = np.lexsort((tumor_flat_idx, tumor_distances))
    ranked_outer_to_inner = tumor_flat_idx[order_outer_to_inner]

    requested_uncertain_count = 0
    if uncertain_boundary_fraction > 0.0:
        requested_uncertain_count = max(1, int(ceil(uncertain_boundary_fraction * tumor_voxel_count)))
    applied_uncertain_count = requested_uncertain_count

    warning_messages: list[str] = []
    definition_adjustments: list[str] = []

    if applied_uncertain_count >= tumor_voxel_count:
        # Preserve downstream processing as much as possible, but record the deviation.
        adjusted = max(0, tumor_voxel_count - 1)
        definition_adjustments.append(
            "uncertain boundary shell count was capped to preserve at least one post-trim voxel"
        )
        warning_messages.append(
            "Tumor too small for strict uncertain-shell trimming; capped uncertain shell to keep at least one remaining voxel."
        )
        applied_uncertain_count = adjusted

    uncertain_flat_idx = ranked_outer_to_inner[:applied_uncertain_count]
    remaining_flat_idx = ranked_outer_to_inner[applied_uncertain_count:]
    remaining_voxel_count_after_trim = int(remaining_flat_idx.size)

    if remaining_voxel_count_after_trim <= 0:
        requested_safe_rim_count = 0
        selected_flat_idx = np.asarray([], dtype=np.int64)
        warning_messages.append(
            "No voxels remained after boundary trimming, so safe_rim is empty."
        )
        disjoint_check_status = "safe_rim_empty_after_trim"
        overlap_with_avoid_requested = 0
    else:
        requested_safe_rim_count = max(
            1,
            int(ceil(safe_rim_fraction_after_trim * remaining_voxel_count_after_trim)),
        )

        overlap_with_avoid_requested = 0
        eligible_flat_idx = remaining_flat_idx
        disjoint_check_status = "not_checked"

        if avoid_mask is not None:
            avoid_mask = np.asarray(avoid_mask, dtype=bool)
            avoid_flat_idx = set(np.flatnonzero(avoid_mask.ravel()))
            overlap_with_avoid_requested = int(
                sum(1 for idx in remaining_flat_idx[:requested_safe_rim_count] if idx in avoid_flat_idx)
            )
            eligible_flat_idx = np.asarray(
                [idx for idx in remaining_flat_idx if idx not in avoid_flat_idx],
                dtype=np.int64,
            )
            disjoint_check_status = "passed"
            if overlap_with_avoid_requested > 0:
                disjoint_check_status = "resolved_by_candidate_exclusion"

        selected_flat_idx = eligible_flat_idx[:requested_safe_rim_count]

        if int(selected_flat_idx.size) < requested_safe_rim_count:
            warning_messages.append(
                "Disjoint safe_rim could not reach the requested voxel count after excluding overlap with core15 or because the tumor is too small."
            )
            if int(selected_flat_idx.size) == 0:
                disjoint_check_status = "empty_after_disjoint_constraint"
            elif disjoint_check_status == "passed":
                disjoint_check_status = "warning_insufficient_voxels"

    uncertain_shell_flat = np.zeros(mask_bool.size, dtype=bool)
    if uncertain_flat_idx.size > 0:
        uncertain_shell_flat[uncertain_flat_idx] = True
    uncertain_shell_mask = uncertain_shell_flat.reshape(mask_bool.shape)

    safe_rim_flat = np.zeros(mask_bool.size, dtype=bool)
    if selected_flat_idx.size > 0:
        safe_rim_flat[selected_flat_idx] = True
    safe_rim_mask = safe_rim_flat.reshape(mask_bool.shape)

    uncertain_shell_intersection = int(
        np.logical_and(uncertain_shell_mask, safe_rim_mask).sum()
    )
    if uncertain_shell_intersection > 0:
        warning_messages.append(
            "Unexpected overlap detected between uncertain shell and safe_rim; safe_rim was corrected by mask construction."
        )

    overlap_with_avoid_final = 0
    if avoid_mask is not None:
        overlap_with_avoid_final = int(np.logical_and(safe_rim_mask, np.asarray(avoid_mask, dtype=bool)).sum())
        if overlap_with_avoid_final > 0:
            warning_messages.append(
                "Unexpected residual overlap detected between safe_rim and core15."
            )
            disjoint_check_status = "failed_overlap_remaining"

    safe_rim_voxel_count = int(safe_rim_mask.sum())
    if safe_rim_voxel_count < int(min_voxel_warning_threshold):
        warning_messages.append(
            f"safe_rim voxel count is very small (n={safe_rim_voxel_count}); rim signal may be noisy or unstable."
        )

    if tumor_voxel_count < 10:
        warning_messages.append(
            f"Tumor voxel count is very small (n={tumor_voxel_count}); safe_rim partition may be unstable."
        )

    selected_distances = edt.ravel()[selected_flat_idx] if selected_flat_idx.size > 0 else np.asarray([], dtype=float)
    info = {
        "uncertain_boundary_fraction": float(uncertain_boundary_fraction),
        "safe_rim_fraction_after_trim": float(safe_rim_fraction_after_trim),
        "tumor_voxel_count": tumor_voxel_count,
        "requested_uncertain_boundary_voxel_count": int(requested_uncertain_count),
        "uncertain_boundary_voxel_count": int(applied_uncertain_count),
        "remaining_voxel_count_after_trim": remaining_voxel_count_after_trim,
        "requested_safe_rim_voxel_count": int(requested_safe_rim_count),
        "safe_rim_voxel_count": safe_rim_voxel_count,
        "max_distance_mm": _finite_float(np.max(tumor_distances)),
        "safe_rim_min_distance_mm": _finite_float(np.min(selected_distances)) if selected_distances.size > 0 else 0.0,
        "safe_rim_max_distance_mm": _finite_float(np.max(selected_distances)) if selected_distances.size > 0 else 0.0,
        "safe_rim_mean_distance_mm": _finite_float(np.mean(selected_distances)) if selected_distances.size > 0 else 0.0,
        "initial_requested_overlap_with_core15_voxel_count": int(overlap_with_avoid_requested),
        "final_overlap_with_core15_voxel_count": int(overlap_with_avoid_final),
        "uncertain_shell_enters_safe_rim": bool(uncertain_shell_intersection > 0),
        "disjoint_check_status": disjoint_check_status,
        "definition_adjustments": definition_adjustments,
        "warning_messages": warning_messages,
    }
    return safe_rim_mask, uncertain_shell_mask, info


def compute_volume_metrics(
    mask_bool: np.ndarray,
    spacing: tuple[float, float, float],
    prefix: str,
) -> dict[str, Any]:
    """
    Compute voxel count and physical volume for a binary mask.

    Volume is computed strictly from NIfTI spacing:
      voxel_volume_mm3 = sx * sy * sz
      volume_mm3 = voxel_count * voxel_volume_mm3
      volume_ml = volume_mm3 / 1000
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    voxel_count = int(mask_bool.sum())
    voxel_volume_mm3 = _finite_float(np.prod(np.asarray(spacing, dtype=float)))
    volume_mm3 = _finite_float(voxel_count * voxel_volume_mm3)
    volume_ml = _finite_float(volume_mm3 / 1000.0)

    return {
        f"{prefix}_voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        f"{prefix}_volume_mm3": volume_mm3,
        f"{prefix}_volume_ml": volume_ml,
    }


def compute_volume_ratio(
    numerator_volume_mm3: float,
    denominator_volume_mm3: float,
) -> float:
    """Compute a finite volume ratio."""
    return _safe_divide(numerator_volume_mm3, denominator_volume_mm3, default=0.0)


def summarize_roi_signal(image_data: np.ndarray, roi_mask: np.ndarray) -> dict[str, Any]:
    """
    Compute summary statistics for one ROI on one timepoint image.
    """
    image_data = np.asarray(image_data, dtype=np.float32)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if image_data.shape != roi_mask.shape:
        raise ValueError(
            f"Image / ROI shape mismatch: image={image_data.shape}, roi={roi_mask.shape}"
        )

    voxel_count = int(roi_mask.sum())
    if voxel_count == 0:
        raise ValueError("ROI is empty; signal statistics are undefined.")

    values = image_data[roi_mask].astype(np.float64, copy=False)

    out = {
        "voxel_count": voxel_count,
        "mean_signal": _finite_float(np.mean(values)),
        "median_signal": _finite_float(np.median(values)),
        "std_signal": _finite_float(np.std(values, ddof=0)),
        "min_signal": _finite_float(np.min(values)),
        "max_signal": _finite_float(np.max(values)),
        "p10_signal": _finite_float(np.percentile(values, 10)),
        "p25_signal": _finite_float(np.percentile(values, 25)),
        "p75_signal": _finite_float(np.percentile(values, 75)),
        "p90_signal": _finite_float(np.percentile(values, 90)),
    }
    return out


def add_normalized_curve_fields(
    rows: list[dict[str, Any]],
    baseline_timepoint_idx: int,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    """
    Add normalized signal fields to per-timepoint ROI rows.

    Normalization is intentionally signal-based, not concentration-based:
      normalized_s_over_s0       = S / (S0 + eps)
      normalized_delta_s_over_s0 = (S - S0) / (S0 + eps)

    `S` and `S0` here refer to mean ROI signal.
    """
    if not rows:
        return rows

    lookup = {int(row["timepoint_idx"]): row for row in rows}
    if baseline_timepoint_idx not in lookup:
        baseline_timepoint_idx = int(rows[0]["timepoint_idx"])

    s0 = _finite_float(lookup[baseline_timepoint_idx]["mean_signal"])
    denom = float(s0 + epsilon)

    for row in rows:
        s = _finite_float(row["mean_signal"])
        s_over_s0 = np.nan_to_num(s / denom, nan=0.0, posinf=0.0, neginf=0.0)
        delta_over_s0 = np.nan_to_num((s - s0) / denom, nan=0.0, posinf=0.0, neginf=0.0)
        row["baseline_timepoint_idx"] = int(baseline_timepoint_idx)
        row["baseline_mean_signal"] = _finite_float(s0)
        row["normalized_s_over_s0"] = _finite_float(s_over_s0)
        row["normalized_delta_s_over_s0"] = _finite_float(delta_over_s0)

    return rows


def _sort_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted by timepoint_idx."""
    return sorted(rows, key=lambda x: int(x["timepoint_idx"]))


def extract_curve_array_fields(
    rows: list[dict[str, Any]],
    time_axis_lookup: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Convert per-timepoint row dicts into array-style fields for case JSON.
    """
    rows_sorted = _sort_curve_rows(rows)
    if not rows_sorted:
        return {
            "timepoint_idx_array": [],
            "mean_signal_array": [],
            "median_signal_array": [],
            "std_signal_array": [],
            "voxel_count_array": [],
            "normalized_s_over_s0_array": [],
            "normalized_delta_s_over_s0_array": [],
            "time_axis_array": [],
        }

    timepoint_idx_array = [int(row["timepoint_idx"]) for row in rows_sorted]
    return {
        "timepoint_idx_array": timepoint_idx_array,
        "mean_signal_array": [_finite_float(row.get("mean_signal", 0.0)) for row in rows_sorted],
        "median_signal_array": [_finite_float(row.get("median_signal", 0.0)) for row in rows_sorted],
        "std_signal_array": [_finite_float(row.get("std_signal", 0.0)) for row in rows_sorted],
        "voxel_count_array": [_finite_int(row.get("voxel_count", 0)) for row in rows_sorted],
        "normalized_s_over_s0_array": [
            _finite_float(row.get("normalized_s_over_s0", 0.0)) for row in rows_sorted
        ],
        "normalized_delta_s_over_s0_array": [
            _finite_float(row.get("normalized_delta_s_over_s0", 0.0)) for row in rows_sorted
        ],
        "time_axis_array": [
            _finite_float(time_axis_lookup.get(int(tp), float(tp)) if time_axis_lookup is not None else float(tp))
            for tp in timepoint_idx_array
        ],
    }


def _first_matching_value(
    rows_sorted: list[dict[str, Any]],
    timepoint_idx: int,
    value_key: str,
) -> tuple[float, bool]:
    for row in rows_sorted:
        if int(row["timepoint_idx"]) == int(timepoint_idx):
            return _finite_float(row.get(value_key, 0.0)), True
    return 0.0, False


def compute_simple_dynamics(
    rows: list[dict[str, Any]],
    roi_type: str,
    pre_idx: int,
    post_early_idx: int,
    post_late_idx: int,
    time_axis_kind: str,
    time_axis_lookup: dict[int, float] | None,
    time_axis_unit: str,
    time_axis_source: str,
) -> dict[str, Any]:
    """
    Compute simple per-ROI dynamics features from mean signal curves.

    Rules
    -----
    - peak is the maximum mean signal; ties use the earliest timepoint
    - wash-in slope = average slope from baseline to peak
    - wash-out slope = average slope from peak to final timepoint
    - if peak is the final timepoint (or no positive span exists), wash_out_slope = 0
    - all numeric outputs are finite and safe for CSV export
    """
    rows_sorted = _sort_curve_rows(rows)
    if not rows_sorted:
        return {
            "roi_type": str(roi_type),
            "feature_valid": 0,
            "status": "empty_curve",
            "time_axis_kind": str(time_axis_kind),
            "time_axis_unit": str(time_axis_unit),
            "time_axis_source": str(time_axis_source),
            "baseline_mean_signal": 0.0,
            "peak_mean_signal": 0.0,
            "peak_timepoint_idx": 0,
            "time_to_peak": 0.0,
            "wash_in_slope": 0.0,
            "wash_out_slope": 0.0,
            "auc_mean_signal": 0.0,
            "auc_normalized_s_over_s0": 0.0,
            "post_early_mean_signal": 0.0,
            "post_late_mean_signal": 0.0,
            "baseline_timepoint_idx": int(pre_idx),
            "wash_out_rule": "empty_curve_returned_zero",
            "warning_messages": ["ROI curve is empty; dynamics were filled with zeros."],
        }

    timepoint_idx_array = np.asarray([int(row["timepoint_idx"]) for row in rows_sorted], dtype=int)
    mean_signal_array = np.asarray([_finite_float(row.get("mean_signal", 0.0)) for row in rows_sorted], dtype=float)
    normalized_array = np.asarray(
        [_finite_float(row.get("normalized_s_over_s0", 0.0)) for row in rows_sorted],
        dtype=float,
    )

    if time_axis_lookup is None:
        x_axis = timepoint_idx_array.astype(float)
    else:
        x_axis = np.asarray([
            _finite_float(time_axis_lookup.get(int(tp), float(tp))) for tp in timepoint_idx_array
        ], dtype=float)

    warning_messages: list[str] = []

    baseline_matches = np.where(timepoint_idx_array == int(pre_idx))[0]
    if baseline_matches.size == 0:
        baseline_pos = 0
        baseline_timepoint_idx = int(timepoint_idx_array[0])
        warning_messages.append(
            f"Requested baseline timepoint {pre_idx} was absent in ROI curve; first available timepoint was used instead."
        )
    else:
        baseline_pos = int(baseline_matches[0])
        baseline_timepoint_idx = int(pre_idx)

    peak_pos = int(np.argmax(mean_signal_array))
    peak_timepoint_idx = int(timepoint_idx_array[peak_pos])

    baseline_mean_signal = _finite_float(mean_signal_array[baseline_pos])
    peak_mean_signal = _finite_float(mean_signal_array[peak_pos])

    time_to_peak = _finite_float(x_axis[peak_pos] - x_axis[baseline_pos])

    delta_x_wash_in = float(x_axis[peak_pos] - x_axis[baseline_pos])
    if delta_x_wash_in > 0:
        wash_in_slope = _finite_float((peak_mean_signal - baseline_mean_signal) / delta_x_wash_in)
    else:
        wash_in_slope = 0.0
        warning_messages.append(
            "Baseline and peak were located at the same axis coordinate; wash_in_slope was set to 0."
        )

    last_pos = int(len(rows_sorted) - 1)
    delta_x_wash_out = float(x_axis[last_pos] - x_axis[peak_pos])
    if peak_pos == last_pos or delta_x_wash_out <= 0:
        wash_out_slope = 0.0
        wash_out_rule = "peak_at_last_timepoint_or_nonpositive_span_set_to_zero"
    else:
        wash_out_slope = _finite_float((mean_signal_array[last_pos] - peak_mean_signal) / delta_x_wash_out)
        wash_out_rule = "standard_last_minus_peak_average_slope"

    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc_mean_signal = _finite_float(trapezoid(mean_signal_array, x_axis))
    auc_normalized_s_over_s0 = _finite_float(trapezoid(normalized_array, x_axis))

    post_early_mean_signal, found_post_early = _first_matching_value(rows_sorted, post_early_idx, "mean_signal")
    if not found_post_early:
        warning_messages.append(
            f"Requested post_early timepoint {post_early_idx} was absent in ROI curve; value was set to 0."
        )

    post_late_mean_signal, found_post_late = _first_matching_value(rows_sorted, post_late_idx, "mean_signal")
    if not found_post_late:
        warning_messages.append(
            f"Requested post_late timepoint {post_late_idx} was absent in ROI curve; value was set to 0."
        )

    return {
        "roi_type": str(roi_type),
        "feature_valid": 1,
        "status": "ok" if not warning_messages else "warning",
        "time_axis_kind": str(time_axis_kind),
        "time_axis_unit": str(time_axis_unit),
        "time_axis_source": str(time_axis_source),
        "baseline_timepoint_idx": int(baseline_timepoint_idx),
        "baseline_mean_signal": baseline_mean_signal,
        "peak_mean_signal": peak_mean_signal,
        "peak_timepoint_idx": int(peak_timepoint_idx),
        "time_to_peak": time_to_peak,
        "wash_in_slope": wash_in_slope,
        "wash_out_slope": wash_out_slope,
        "auc_mean_signal": auc_mean_signal,
        "auc_normalized_s_over_s0": auc_normalized_s_over_s0,
        "post_early_mean_signal": _finite_float(post_early_mean_signal),
        "post_late_mean_signal": _finite_float(post_late_mean_signal),
        "wash_out_rule": wash_out_rule,
        "warning_messages": warning_messages,
    }


def prefix_dynamics_fields(
    dynamics: dict[str, Any],
    prefix: str,
    include_meta: bool = False,
) -> dict[str, Any]:
    """
    Prefix numeric/simple-dynamics fields for flat case-level CSV export.
    """
    field_names = [
        "feature_valid",
        "baseline_mean_signal",
        "peak_mean_signal",
        "peak_timepoint_idx",
        "time_to_peak",
        "wash_in_slope",
        "wash_out_slope",
        "auc_mean_signal",
        "auc_normalized_s_over_s0",
        "post_early_mean_signal",
        "post_late_mean_signal",
    ]
    if include_meta:
        field_names.extend([
            "status",
            "time_axis_kind",
            "time_axis_unit",
            "time_axis_source",
        ])

    out: dict[str, Any] = {}
    for field in field_names:
        if field not in dynamics:
            continue
        value = dynamics[field]
        if isinstance(value, (int, np.integer)):
            out[f"{prefix}_{field}"] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[f"{prefix}_{field}"] = _finite_float(value)
        else:
            out[f"{prefix}_{field}"] = value
    return out


def interpolate_curve_to_target_times(
    source_times: list[float] | np.ndarray,
    source_values: list[float] | np.ndarray,
    target_times: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Optional future hook for fixed-length downstream modeling.

    Not used by default. No interpolation is performed unless the caller
    explicitly invokes this function.
    """
    source_times = np.asarray(source_times, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if source_times.ndim != 1 or source_values.ndim != 1:
        raise ValueError("source_times and source_values must be 1D.")
    if source_times.size != source_values.size:
        raise ValueError("source_times and source_values must have the same length.")
    if source_times.size < 2:
        raise ValueError("At least 2 source points are required for interpolation.")

    order = np.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]
    return np.interp(target_times, source_times, source_values)
