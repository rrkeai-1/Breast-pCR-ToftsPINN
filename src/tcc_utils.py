"""
TCC (time-concentration curve) extraction helpers.

This module collects the small per-case helpers used by
``scripts/01_extract_roi_tcc.py``. The script itself stays self-contained
(it carries the per-case orchestration) so that users can read the whole
extraction flow top-to-bottom without jumping between files; this module
re-exports the underlying utilities so they are importable as a clean API
from the ``src`` package as well.

Public surface
--------------
- ``add_normalized_curve_fields``: append S/S0 fields to ROI signal rows.
- ``compute_simple_dynamics``: compute scalar dynamics features
  (time-to-peak, wash-in slope, wash-out slope, normalized AUC).
- ``compute_volume_metrics`` / ``compute_volume_ratio``: tumor / core15 /
  safe_rim volumes in mm^3 and the safe_rim/core15 ratio.
- ``extract_curve_array_fields`` / ``prefix_dynamics_fields``: small helpers
  used to format per-case rows.
- ``summarize_roi_signal``: per-timepoint mean signal inside an ROI.

Notes
-----
- Nothing in this module loads images or talks to disk; concrete file IO
  lives in ``io_utils`` and concrete ROI partitioning lives in ``roi_utils``.
- This module imposes no path conventions and contains no real-data values.
"""

from __future__ import annotations

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

__all__ = [
    "add_normalized_curve_fields",
    "build_core_mask_by_fraction",
    "build_safe_rim_mask_by_fraction",
    "clean_mask",
    "compute_simple_dynamics",
    "compute_volume_metrics",
    "compute_volume_ratio",
    "extract_curve_array_fields",
    "prefix_dynamics_fields",
    "summarize_roi_signal",
]
