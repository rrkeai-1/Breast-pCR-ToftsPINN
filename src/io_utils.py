"""
I/O helpers for the DCE core15 + safe-rim extraction project.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry_utils import strip_nii_suffix


def configure_logging(output_dir: Path) -> logging.Logger:
    """Configure a project logger with both console and file handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logger = logging.getLogger("dce_core15")
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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def load_metadata_csv(path: Path) -> pd.DataFrame:
    """
    Load metadata CSV and standardize a small set of canonical columns.

    Required:
    - pid (or synonym)

    Common optional columns handled if present:
    - n_times, pre, post_early, post_late, pCR, split, test, dataset
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"metadata CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_column_names(df)

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


def load_case_map_csv(path: Path | None) -> dict[str, Any]:
    """
    Load an optional case map CSV to map nnUNet case IDs to PID.

    Expected columns:
    - pid (or synonym)
    - nnunet_case_id / case_id / mask_case_id / image_id
    """
    if path is None:
        return {
            "raw_df": None,
            "pid_to_case_ids": {},
            "case_id_to_pid": {},
        }

    if not Path(path).exists():
        raise FileNotFoundError(f"case_map CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_column_names(df)

    pid_col = _find_first_existing_column(df, ["pid", "PatientID", "patient_id"])
    case_col = _find_first_existing_column(
        df,
        ["nnunet_case_id", "case_id", "mask_case_id", "image_id", "mask_id"],
    )

    if pid_col is None or case_col is None:
        raise ValueError(
            "case_map CSV must contain PID and nnUNet-case columns, e.g. "
            "`pid` + `nnunet_case_id`"
        )

    df = df.rename(columns={pid_col: "pid", case_col: "nnunet_case_id"})
    df["pid"] = df["pid"].astype(str).str.strip()
    df["nnunet_case_id"] = df["nnunet_case_id"].astype(str).map(strip_nii_suffix).str.strip()

    pid_to_case_ids: dict[str, list[str]] = {}
    case_id_to_pid: dict[str, str] = {}

    for _, row in df.iterrows():
        pid = row["pid"]
        case_id = row["nnunet_case_id"]
        pid_to_case_ids.setdefault(pid, []).append(case_id)
        case_id_to_pid[case_id] = pid

    return {
        "raw_df": df,
        "pid_to_case_ids": pid_to_case_ids,
        "case_id_to_pid": case_id_to_pid,
    }


def apply_subset_filter(
    metadata_df: pd.DataFrame,
    subset_csv: Path | None,
    subset_name: str | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Filter metadata by either:
    - an external subset CSV
    - subset_name against metadata columns
    - both

    Supported conventions:
    - explicit pid list in subset CSV
    - subset/split column in subset CSV or metadata
    - boolean-like `test` column in subset CSV or metadata
    """
    df = metadata_df.copy()

    if subset_csv is not None:
        subset_df = load_metadata_csv(subset_csv)
        if "pid" not in subset_df.columns:
            raise ValueError("subset_csv must contain a pid column.")

        if subset_name is None:
            allowed_pids = set(subset_df["pid"].astype(str))
            logger.info("Applying subset_csv PID filter: %d cases", len(allowed_pids))
            df = df[df["pid"].isin(allowed_pids)].copy()
        else:
            subset_filtered = _filter_by_subset_name(subset_df, subset_name, logger)
            allowed_pids = set(subset_filtered["pid"].astype(str))
            logger.info(
                "Applying subset_csv + subset_name=%s filter: %d cases",
                subset_name,
                len(allowed_pids),
            )
            df = df[df["pid"].isin(allowed_pids)].copy()
    elif subset_name is not None:
        logger.info("Applying metadata subset filter: subset_name=%s", subset_name)
        df = _filter_by_subset_name(df, subset_name, logger)

    return df.reset_index(drop=True)


def _filter_by_subset_name(
    df: pd.DataFrame,
    subset_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    subset_name_lower = str(subset_name).strip().lower()

    if "split" in df.columns:
        mask = df["split"].astype(str).str.strip().str.lower() == subset_name_lower
        if mask.any():
            return df.loc[mask].copy()

    if "dataset" in df.columns:
        mask = df["dataset"].astype(str).str.strip().str.lower() == subset_name_lower
        if mask.any():
            return df.loc[mask].copy()

    if "test" in df.columns:
        test_series = pd.to_numeric(df["test"], errors="coerce")
        if subset_name_lower in {"test", "heldout", "holdout"}:
            mask = test_series.fillna(0).astype(int) == 1
            if mask.any():
                return df.loc[mask].copy()
        if subset_name_lower in {"train", "non-test", "nontest"}:
            mask = test_series.fillna(0).astype(int) == 0
            if mask.any():
                return df.loc[mask].copy()

    raise ValueError(
        f"Could not apply subset_name={subset_name!r}. Expected a compatible "
        f"split/dataset/test column or a matching subset_csv."
    )


def build_mask_index(mask_dir: Path, case_map: dict[str, Any] | None = None) -> dict[str, Path]:
    """
    Build a flexible alias -> file-path index for predicted masks.

    Aliases include:
    - bare filename stem
    - stem without _mask or _spy2_vis1_mask suffix
    - mapped pid from case_map, if provided
    """
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    case_id_to_pid = {}
    if case_map is not None:
        case_id_to_pid = case_map.get("case_id_to_pid", {}) or {}

    index: dict[str, Path] = {}
    files = sorted(list(mask_dir.glob("*.nii.gz")) + list(mask_dir.glob("*.nii")))

    for path in files:
        stem = strip_nii_suffix(path.name)
        aliases = {stem}

        if stem.endswith("_spy2_vis1_mask"):
            aliases.add(stem[: -len("_spy2_vis1_mask")])
        if stem.endswith("_mask"):
            aliases.add(stem[: -len("_mask")])

        if stem in case_id_to_pid:
            aliases.add(case_id_to_pid[stem])

        for alias in aliases:
            index.setdefault(alias, path)

    return index


def resolve_mask_path_for_pid(
    pid: str,
    mask_index: dict[str, Path],
    case_map: dict[str, Any] | None = None,
) -> Path:
    """
    Resolve the correct predicted-mask file for a given PID.
    """
    candidate_keys = [pid, f"{pid}_spy2_vis1_mask", f"{pid}_mask"]

    if case_map is not None:
        for case_id in case_map.get("pid_to_case_ids", {}).get(pid, []):
            candidate_keys.extend([case_id, f"{case_id}_mask", f"{case_id}_spy2_vis1_mask"])

    for key in candidate_keys:
        if key in mask_index:
            return mask_index[key]

    raise FileNotFoundError(
        f"Could not resolve predicted mask for pid={pid}. "
        f"Checked aliases: {candidate_keys}"
    )


_DCE_RE = re.compile(r"_aqc_(\d+)\.nii(?:\.gz)?$", re.IGNORECASE)


def discover_dce_timepoints(full_root: Path, pid: str) -> list[tuple[int, Path]]:
    """
    Discover and sort all raw DCE timepoints for one case.

    Expected location:
      full_root / dce / {pid} / {pid}_spy2_vis1_dce_aqc_{t}.nii.gz
    """
    case_dir = full_root / "dce" / pid
    if not case_dir.exists():
        raise FileNotFoundError(f"DCE case directory not found: {case_dir}")

    candidates = sorted(list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii")))
    pairs: list[tuple[int, Path]] = []
    seen = set()

    for path in candidates:
        match = _DCE_RE.search(path.name)
        if match is None:
            continue
        time_idx = int(match.group(1))
        if time_idx in seen:
            raise ValueError(f"Duplicate timepoint index {time_idx} for pid={pid}: {path}")
        seen.add(time_idx)
        pairs.append((time_idx, path))

    if not pairs:
        raise FileNotFoundError(f"No DCE timepoints found under {case_dir}")

    pairs.sort(key=lambda x: x[0])
    return pairs


def sanitize_jsonable(value: Any) -> Any:
    """Convert numpy / pandas / Path objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return value


def write_case_json(case_result: dict[str, Any], out_path: Path) -> None:
    """Write one per-case JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_jsonable(case_result), f, indent=2, ensure_ascii=False)


def write_tables(
    volume_rows: list[dict[str, Any]],
    tcc_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    output_dir: Path,
    dynamics_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write project-level CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_df = pd.DataFrame(volume_rows)
    tcc_df = pd.DataFrame(tcc_rows)
    qc_df = pd.DataFrame(qc_rows)
    dynamics_df = pd.DataFrame(dynamics_rows) if dynamics_rows is not None else None

    if not volume_df.empty:
        volume_df = volume_df.sort_values(["pid"], kind="stable")
    if not tcc_df.empty:
        tcc_df = tcc_df.sort_values(["pid", "roi_type", "timepoint_idx"], kind="stable")
    if not qc_df.empty:
        qc_df = qc_df.sort_values(["pid"], kind="stable")
    if dynamics_df is not None and not dynamics_df.empty:
        dynamics_df = dynamics_df.sort_values(["pid"], kind="stable")

    volume_df.to_csv(output_dir / "volume_summary.csv", index=False)
    tcc_df.to_csv(output_dir / "tcc_long.csv", index=False)
    qc_df.to_csv(output_dir / "qc_report.csv", index=False)
    if dynamics_df is not None:
        dynamics_df.to_csv(output_dir / "dynamics_feature_summary.csv", index=False)


def _extract_curve_xy(rows: list[dict[str, Any]], y_key: str) -> tuple[np.ndarray, np.ndarray]:
    rows_sorted = sorted(rows, key=lambda x: int(x["timepoint_idx"]))
    if not rows_sorted:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.asarray([int(r["timepoint_idx"]) for r in rows_sorted], dtype=float)
    y = np.asarray([float(r[y_key]) for r in rows_sorted], dtype=float)
    return x, y


def save_tcc_plot(
    pid: str,
    full_rows: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
    out_path: Path,
    safe_rim_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Save a two-panel QC plot for raw and normalized mean signal curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_full, y_full = _extract_curve_xy(full_rows, "mean_signal")
    x_core, y_core = _extract_curve_xy(core_rows, "mean_signal")
    x_rim, y_rim = _extract_curve_xy(safe_rim_rows or [], "mean_signal")

    x_full_n, y_full_n = _extract_curve_xy(full_rows, "normalized_s_over_s0")
    x_core_n, y_core_n = _extract_curve_xy(core_rows, "normalized_s_over_s0")
    x_rim_n, y_rim_n = _extract_curve_xy(safe_rim_rows or [], "normalized_s_over_s0")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax = axes[0]
    if x_full.size > 0:
        ax.plot(x_full, y_full, marker="o", label="full_tumor")
    if x_core.size > 0:
        ax.plot(x_core, y_core, marker="s", label="core15")
    if x_rim.size > 0:
        ax.plot(x_rim, y_rim, marker="^", label="safe_rim")
    ax.set_xlabel("Timepoint index")
    ax.set_ylabel("Mean signal")
    ax.set_title(f"{pid} - raw mean signal TCC")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if x_full_n.size > 0:
        ax.plot(x_full_n, y_full_n, marker="o", label="full_tumor S/S0")
    if x_core_n.size > 0:
        ax.plot(x_core_n, y_core_n, marker="s", label="core15 S/S0")
    if x_rim_n.size > 0:
        ax.plot(x_rim_n, y_rim_n, marker="^", label="safe_rim S/S0")
    ax.set_xlabel("Timepoint index")
    ax.set_ylabel("Normalized signal (S/S0)")
    ax.set_title(f"{pid} - normalized mean signal TCC")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overlay_plot(
    pid: str,
    reference_data: np.ndarray,
    full_mask: np.ndarray,
    core_mask: np.ndarray,
    reference_timepoint_idx: int,
    out_path: Path,
    safe_rim_mask: np.ndarray | None = None,
) -> None:
    """
    Save a mid-slice overlay for quick manual geometry verification.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reference_data = np.asarray(reference_data, dtype=float)
    full_mask = np.asarray(full_mask, dtype=bool)
    core_mask = np.asarray(core_mask, dtype=bool)
    safe_rim_mask = np.asarray(safe_rim_mask, dtype=bool) if safe_rim_mask is not None else None

    if full_mask.sum() > 0:
        z_indices = np.where(full_mask)[2]
        z = int(np.median(z_indices))
    else:
        z = reference_data.shape[2] // 2

    img2d = reference_data[:, :, z].T
    full2d = full_mask[:, :, z].T
    core2d = core_mask[:, :, z].T
    rim2d = safe_rim_mask[:, :, z].T if safe_rim_mask is not None else None

    vmin = float(np.percentile(img2d, 1))
    vmax = float(np.percentile(img2d, 99))

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(img2d, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if np.any(full2d):
        ax.contour(full2d.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=1.2)
    if np.any(core2d):
        ax.imshow(np.ma.masked_where(~core2d, core2d), origin="lower", alpha=0.45, cmap="spring")
    if rim2d is not None and np.any(rim2d):
        ax.contour(rim2d.astype(np.uint8), levels=[0.5], colors=["deepskyblue"], linewidths=1.2)

    ax.set_title(f"{pid} - reference tp={reference_timepoint_idx} - mid-slice overlay")
    ax.axis("off")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
