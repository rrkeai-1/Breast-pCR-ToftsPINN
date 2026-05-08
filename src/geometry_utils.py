"""
Geometry utilities for safe DCE-MRI / mask alignment checks.

Default behavior is intentionally conservative:
- detect shape / spacing / affine / orientation mismatches
- fail on substantial mismatch
- only tolerate very small affine floating-point discrepancies as warnings
- do not resample unless explicitly requested by the caller
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_from_to


def strip_nii_suffix(name: str) -> str:
    """Strip .nii or .nii.gz from a filename-like string."""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def load_nifti(path: Path | str) -> nib.spatialimages.SpatialImage:
    """Load a NIfTI image from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(str(path))
    return ensure_3d_image(img, path)


def ensure_3d_image(
    img: nib.spatialimages.SpatialImage,
    source: Path | str | None = None,
) -> nib.spatialimages.SpatialImage:
    """
    Ensure the loaded NIfTI behaves like a 3D spatial image.

    Accepted:
    - 3D images
    - 4D images with a singleton last dimension, which are squeezed to 3D

    Rejected:
    - all other dimensionalities
    """
    source_str = str(source) if source is not None else "<in-memory image>"
    shape = tuple(int(v) for v in img.shape)

    if len(shape) == 3:
        return img

    if len(shape) == 4 and shape[3] == 1:
        data = np.asarray(img.dataobj)[..., 0]
        header = img.header.copy()
        new_img = nib.Nifti1Image(data, img.affine, header=header)
        return new_img

    raise ValueError(
        f"Expected a 3D NIfTI (or 4D with singleton time dim), "
        f"but got shape={shape} from {source_str}"
    )


def get_spacing(img: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    """Return voxel spacing (mm) for the first 3 spatial dimensions."""
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError(f"Image has invalid zoom metadata: {zooms}")
    return tuple(float(v) for v in zooms[:3])


def get_orientation(img: nib.spatialimages.SpatialImage) -> tuple[str | None, ...]:
    """Return orientation codes inferred from affine."""
    return tuple(aff2axcodes(img.affine))


def get_nifti_summary(
    img: nib.spatialimages.SpatialImage,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Return a compact JSON-friendly geometry summary."""
    return {
        "path": str(path) if path is not None else None,
        "shape": [int(v) for v in img.shape[:3]],
        "spacing": [float(v) for v in get_spacing(img)],
        "orientation": list(get_orientation(img)),
        "affine": np.asarray(img.affine, dtype=float).round(6).tolist(),
    }


@dataclass
class GeometryComparison:
    """Structured comparison of two 3D NIfTI geometries."""

    left_name: str
    right_name: str
    left_shape: tuple[int, int, int]
    right_shape: tuple[int, int, int]
    left_spacing: tuple[float, float, float]
    right_spacing: tuple[float, float, float]
    left_orientation: tuple[str | None, ...]
    right_orientation: tuple[str | None, ...]
    shape_match: bool
    spacing_match: bool
    orientation_match: bool
    affine_match: bool
    max_spacing_abs_diff: float
    max_affine_abs_diff: float
    status: str
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["left_shape"] = list(self.left_shape)
        out["right_shape"] = list(self.right_shape)
        out["left_spacing"] = [float(v) for v in self.left_spacing]
        out["right_spacing"] = [float(v) for v in self.right_spacing]
        out["left_orientation"] = list(self.left_orientation)
        out["right_orientation"] = list(self.right_orientation)
        out["max_spacing_abs_diff"] = float(self.max_spacing_abs_diff)
        out["max_affine_abs_diff"] = float(self.max_affine_abs_diff)
        return out


def compare_geometry(
    left_img: nib.spatialimages.SpatialImage,
    right_img: nib.spatialimages.SpatialImage,
    left_name: str,
    right_name: str,
    spacing_atol: float = 1e-6,
    affine_atol: float = 1e-3,
    minor_affine_multiplier: float = 10.0,
) -> GeometryComparison:
    """
    Compare 3D geometry in a conservative way.

    Logic:
    - shape mismatch => error
    - spacing mismatch => error
    - orientation mismatch => error
    - affine mismatch with otherwise matching shape/spacing/orientation:
        * if very small -> warning
        * else -> error
    """
    left_shape = tuple(int(v) for v in left_img.shape[:3])
    right_shape = tuple(int(v) for v in right_img.shape[:3])

    left_spacing = get_spacing(left_img)
    right_spacing = get_spacing(right_img)

    left_orientation = get_orientation(left_img)
    right_orientation = get_orientation(right_img)

    shape_match = left_shape == right_shape
    spacing_match = bool(np.allclose(left_spacing, right_spacing, atol=spacing_atol, rtol=0.0))
    orientation_match = left_orientation == right_orientation

    left_aff = np.asarray(left_img.affine, dtype=float)
    right_aff = np.asarray(right_img.affine, dtype=float)

    affine_match = bool(np.allclose(left_aff, right_aff, atol=affine_atol, rtol=0.0))
    max_spacing_abs_diff = float(np.max(np.abs(np.asarray(left_spacing) - np.asarray(right_spacing))))
    max_affine_abs_diff = float(np.max(np.abs(left_aff - right_aff)))

    messages: list[str] = []
    status = "ok"

    if not shape_match:
        messages.append(f"Shape mismatch: {left_shape} vs {right_shape}")
        status = "error"

    if not spacing_match:
        messages.append(f"Spacing mismatch: {left_spacing} vs {right_spacing}")
        status = "error"

    if not orientation_match:
        messages.append(f"Orientation mismatch: {left_orientation} vs {right_orientation}")
        status = "error"

    if not affine_match:
        messages.append(f"Affine mismatch: max_abs_diff={max_affine_abs_diff:.6g}")
        if status == "ok":
            # Only affine differs. Decide if it is a tiny floating discrepancy or a real mismatch.
            if max_affine_abs_diff <= minor_affine_multiplier * affine_atol:
                status = "warning"
                messages.append(
                    "Affine difference is larger than affine_atol but within the "
                    "minor-affine warning tolerance."
                )
            else:
                status = "error"

    return GeometryComparison(
        left_name=left_name,
        right_name=right_name,
        left_shape=left_shape,
        right_shape=right_shape,
        left_spacing=left_spacing,
        right_spacing=right_spacing,
        left_orientation=left_orientation,
        right_orientation=right_orientation,
        shape_match=shape_match,
        spacing_match=spacing_match,
        orientation_match=orientation_match,
        affine_match=affine_match,
        max_spacing_abs_diff=max_spacing_abs_diff,
        max_affine_abs_diff=max_affine_abs_diff,
        status=status,
        messages=messages,
    )


def resample_mask_to_reference(
    mask_img: nib.spatialimages.SpatialImage,
    reference_img: nib.spatialimages.SpatialImage,
) -> nib.spatialimages.SpatialImage:
    """
    Resample a mask into reference space using nearest-neighbor interpolation.

    This is intentionally explicit and separate from normal processing because
    the default project behavior is to fail rather than silently resample.
    """
    resampled = resample_from_to(
        from_img=mask_img,
        to_vox_map=reference_img,
        order=0,          # nearest-neighbor for masks
        mode="constant",
        cval=0.0,
        out_class=nib.Nifti1Image,
    )
    data = np.asarray(resampled.dataobj)
    binary = (data > 0.5).astype(np.uint8)
    header = reference_img.header.copy()
    return nib.Nifti1Image(binary, reference_img.affine, header=header)


def summarize_geometry_checks(
    comparisons: Iterable[GeometryComparison],
) -> dict[str, Any]:
    """Collect multiple comparison results into one compact summary."""
    comparison_dicts = [cmp_.to_dict() for cmp_ in comparisons]
    n_error = sum(1 for c in comparison_dicts if c["status"] == "error")
    n_warning = sum(1 for c in comparison_dicts if c["status"] == "warning")
    overall_status = "ok"
    if n_error > 0:
        overall_status = "error"
    elif n_warning > 0:
        overall_status = "warning"

    return {
        "overall_status": overall_status,
        "n_checks": len(comparison_dicts),
        "n_error": n_error,
        "n_warning": n_warning,
        "checks": comparison_dicts,
    }
