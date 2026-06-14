"""Standardized local dataset repository materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from lsd_app.core.channels import DEFAULT_RGB_BANDS, extract_rgb_bands, normalize_rgb_bands, rgb_bands_to_indexes
from lsd_app.core.paths import DATASET_STORE_DIRNAME as TOKEN_DATASET_STORE_DIRNAME
from lsd_app.core.paths import portable_path, repair_portable_paths_in_payload
from lsd_app.core.raster import require_rasterio


IMAGE_SUFFIXES = (".tif", ".tiff", ".png")
LABEL_SUFFIXES = (".tif", ".tiff", ".png", ".gpkg", ".geojson", ".shp")
RASTER_LABEL_SUFFIXES = (".tif", ".tiff", ".png")
DATASET_CLASSES = ("TAFT", "DAFT", "TSFT", "Evaluation")
SUPERVISED_CLASSES = ("TAFT", "TSFT", "Evaluation")
DATASET_STORE_DIRNAME = "dataset_store"
MANIFEST_NAME = "manifest.json"
SUBSETS_DIRNAME = "subsets"
TAFT_TREE_STATE_NAME = "import_state.json"
TAFT_TREE_EVENTS_NAME = "import_events.jsonl"
DAFT_UPLOAD_STATE_NAME = "upload_state.json"
TAFT_TREE_PRESETS = ("landslide77k",)
DAFT_UPLOAD_METHOD_PACKAGED = "packaged_images"
DAFT_UPLOAD_METHOD_TIFF_SLICING = "tiff_slicing"
DAFT_UPLOAD_METHODS = (DAFT_UPLOAD_METHOD_PACKAGED, DAFT_UPLOAD_METHOD_TIFF_SLICING)
DEFAULT_DAFT_MAX_NODATA_RATIO = 0.20
DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO = 0.0
DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO = 0.001
DEFAULT_TSFT_NEGATIVE_RATIO = 1.0
TIFF_RASTER_SUFFIXES = (".tif", ".tiff")
TIFF_AUXILIARY_NAME_TOKENS = (
    "_date",
    "_qa",
    "_mask",
    "_provenance",
    "udm",
    "cloud",
)
TSFT_NO_FEATURE_POLICY_ONLY = "only_with_features"
TSFT_NO_FEATURE_POLICY_BALANCED = "balanced"
TSFT_NO_FEATURE_POLICY_ALL = "all_valid"
TSFT_NO_FEATURE_POLICIES = (
    TSFT_NO_FEATURE_POLICY_ONLY,
    TSFT_NO_FEATURE_POLICY_BALANCED,
    TSFT_NO_FEATURE_POLICY_ALL,
)
LABEL_POLARITY_FOREGROUND_WHITE = "foreground_white"
LABEL_POLARITY_FOREGROUND_BLACK = "foreground_black"
LABEL_POLARITIES = (LABEL_POLARITY_FOREGROUND_WHITE, LABEL_POLARITY_FOREGROUND_BLACK)
MIN_DATASET_STORE_FREE_BYTES = 2 * 1024 * 1024 * 1024
ProgressCallback = Callable[[dict[str, Any]], None]


class DatasetStandardizationError(ValueError):
    """Raised when a dataset cannot be standardized into the repository layout."""


@dataclass(frozen=True)
class StandardizedSample:
    """One materialized standardized image sample."""

    sample_id: str
    split: str
    image_path: Path
    label_path: Path | None
    shape: tuple[int, int, int]
    dtype: str
    normalized: bool
    linear_stretch_applied: bool
    source_raster: Path | None = None
    window: tuple[int, int, int, int] | None = None
    has_positive_label: bool | None = None
    label_positive_pixels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": str(self.image_path),
            "label_path": str(self.label_path) if self.label_path is not None else None,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "normalized": self.normalized,
            "linear_stretch_applied": self.linear_stretch_applied,
        }
        if self.source_raster is not None:
            data["source_raster"] = str(self.source_raster)
        if self.window is not None:
            data["window"] = list(self.window)
        if self.has_positive_label is not None:
            data["has_positive_label"] = bool(self.has_positive_label)
        if self.label_positive_pixels is not None:
            data["label_positive_pixels"] = int(self.label_positive_pixels)
        return data


@dataclass(frozen=True)
class StandardizedDatasetResult:
    """Summary returned after a dataset upload has been standardized."""

    dataset_id: str
    dataset_class: str
    version: str
    root_dir: Path
    imagery_path: Path
    label_path: Path | None
    has_gt: bool
    RGB_bands: str
    sample_shape_layout: str
    sample_dtype: str
    sample_count: int
    train_count: int
    test_count: int
    subsets: tuple[dict[str, Any], ...]
    manifest_path: Path
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root_dir"] = str(self.root_dir)
        data["imagery_path"] = str(self.imagery_path)
        data["label_path"] = str(self.label_path) if self.label_path is not None else None
        data["manifest_path"] = str(self.manifest_path)
        data["subsets"] = [dict(row) for row in self.subsets]
        return data


@dataclass(frozen=True)
class TaftTreeSubset:
    """One discovered leaf subset inside a hierarchical TAFT source tree."""

    source_rel: str
    subset_id: str
    parent_group: str
    source_dir: Path
    RGB_bands: str
    label_polarity: str
    auto_RGB_bands: str
    auto_label_polarity: str
    train_image_dir: Path
    train_label_dir: Path
    test_image_dir: Path
    test_label_dir: Path
    train_count: int
    test_count: int
    sample_count: int
    image_suffixes: tuple[str, ...]
    label_suffixes: tuple[str, ...]
    has_auxiliary: bool
    auxiliary_dirs: tuple[str, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    annotation_source: str = "auto_detected"
    user_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("source_dir", "train_image_dir", "train_label_dir", "test_image_dir", "test_label_dir"):
            data[key] = str(data[key])
        data["subset"] = self.source_rel
        data["has_label"] = True
        data["sample_layout"] = "(H, W, C)"
        return data


@dataclass(frozen=True)
class TaftTreeDiscovery:
    """Summary of a recursive TAFT tree discovery pass."""

    root_path: Path
    preset: str
    subsets: tuple[TaftTreeSubset, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def subset_count(self) -> int:
        return len(self.subsets)

    @property
    def train_count(self) -> int:
        return sum(subset.train_count for subset in self.subsets)

    @property
    def test_count(self) -> int:
        return sum(subset.test_count for subset in self.subsets)

    @property
    def sample_count(self) -> int:
        return sum(subset.sample_count for subset in self.subsets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_path": str(self.root_path),
            "preset": self.preset,
            "subset_count": self.subset_count,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "sample_count": self.sample_count,
            "has_errors": bool(self.errors or any(subset.errors for subset in self.subsets)),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "subsets": [subset.to_dict() for subset in self.subsets],
        }


@dataclass(frozen=True)
class TaftTreeRegistrationResult:
    """Summary returned after a hierarchical TAFT tree has been registered."""

    dataset_id: str
    dataset_class: str
    name: str
    version: str
    root_dir: Path
    manifest_path: Path
    source_root: Path
    preset: str
    standardized: bool
    split_policy: str
    has_gt: bool
    RGB_bands: str
    sample_shape_layout: str
    sample_dtype: str
    sample_count: int
    train_count: int
    test_count: int
    subset_count: int
    created_at: str
    subsets: tuple[dict[str, Any], ...]
    skipped_samples: int = 0
    written_samples: int = 0
    rewritten_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("root_dir", "manifest_path", "source_root"):
            data[key] = str(data[key])
        data["subsets"] = [dict(row) for row in self.subsets]
        return data


@dataclass(frozen=True)
class DaftSliceResult:
    """One standardized image chip generated from a source raster."""

    sample_id: str
    source_raster: Path
    image_path: Path
    window: tuple[int, int, int, int]
    shape: tuple[int, int, int]
    dtype: str
    normalized: bool
    linear_stretch_applied: bool

    def to_sample(self) -> StandardizedSample:
        return StandardizedSample(
            sample_id=self.sample_id,
            split="all",
            image_path=self.image_path,
            label_path=None,
            shape=self.shape,
            dtype=self.dtype,
            normalized=self.normalized,
            linear_stretch_applied=self.linear_stretch_applied,
            source_raster=self.source_raster,
            window=self.window,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_raster"] = str(self.source_raster)
        data["image_path"] = str(self.image_path)
        data["window"] = list(self.window)
        data["shape"] = list(self.shape)
        return data


@dataclass(frozen=True)
class TsftSliceWindow:
    """One valid TSFT slicing window selected for materialization."""

    sample_id: str
    source_raster: Path
    row_off: int
    col_off: int
    height: int
    width: int
    has_positive_label: bool
    label_positive_pixels: int
    nodata_ratio: float
    valid_pixels: int
    window_mode: str = "fixed"
    aoi_geometry: Any | None = None
    target_size: int | None = None

    @property
    def window(self) -> tuple[int, int, int, int]:
        return (self.row_off, self.col_off, self.height, self.width)


def standardize_dataset_upload(
    *,
    store_root: str | Path,
    dataset_id: str,
    dataset_class: str,
    version: str,
    imagery_path: str | Path,
    label_path: str | Path | None = None,
    test_imagery_path: str | Path | None = None,
    test_label_path: str | Path | None = None,
    RGB_bands: str,
    label_polarity: str = LABEL_POLARITY_FOREGROUND_WHITE,
    subset_name: str | None = None,
    source: str = "user",
    notes: str = "",
    overwrite: bool = False,
    resume: bool = False,
    upload_method: str = DAFT_UPLOAD_METHOD_PACKAGED,
    tile_size: int | None = None,
    stride: int | None = None,
    daft_max_nodata_ratio: float | None = DEFAULT_DAFT_MAX_NODATA_RATIO,
    slicing_aoi_path: str | Path | None = None,
    tsft_min_polygon_overlap_ratio: float | None = DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
    tsft_min_positive_pixel_ratio: float | None = DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
    tsft_no_feature_tile_policy: str = TSFT_NO_FEATURE_POLICY_BALANCED,
    tsft_negative_ratio: float | None = DEFAULT_TSFT_NEGATIVE_RATIO,
    progress_callback: ProgressCallback | None = None,
) -> StandardizedDatasetResult:
    """Materialize user data into the standard TAFT/DAFT/TSFT/Evaluation local repository."""

    dataset_class = _normalize_dataset_class(dataset_class)
    RGB_bands = normalize_rgb_bands(RGB_bands)
    label_polarity = _normalize_label_polarity(label_polarity)
    version = _required_text(version, "version")
    dataset_id = _required_text(dataset_id, "dataset_id")
    upload_method = _normalize_upload_method(upload_method)
    imagery = Path(imagery_path).expanduser().resolve()
    if not imagery.exists():
        raise DatasetStandardizationError(f"Imagery path does not exist: {imagery}")

    label = Path(label_path).expanduser().resolve() if label_path not in {None, ""} else None
    slicing_aoi = Path(slicing_aoi_path).expanduser().resolve() if slicing_aoi_path not in {None, ""} else None
    test_imagery = Path(test_imagery_path).expanduser().resolve() if test_imagery_path not in {None, ""} else None
    test_label = Path(test_label_path).expanduser().resolve() if test_label_path not in {None, ""} else None
    if dataset_class in SUPERVISED_CLASSES and label is None and not (
        dataset_class == "TSFT" and upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
    ):
        raise DatasetStandardizationError(f"{dataset_class} datasets require label_path.")
    if dataset_class == "DAFT" and label is not None:
        raise DatasetStandardizationError("DAFT datasets are images-only and must not include label_path.")
    if label is not None and not label.exists():
        raise DatasetStandardizationError(f"Label path does not exist: {label}")
    if slicing_aoi is not None and not slicing_aoi.exists():
        raise DatasetStandardizationError(f"Slicing AOI path does not exist: {slicing_aoi}")
    if test_imagery is not None and not test_imagery.exists():
        raise DatasetStandardizationError(f"Test imagery path does not exist: {test_imagery}")
    if test_label is not None and not test_label.exists():
        raise DatasetStandardizationError(f"Test label path does not exist: {test_label}")
    if test_imagery is not None or test_label is not None:
        raise DatasetStandardizationError(
            "TSFT test split inputs are no longer supported; register labeled evaluation data as an Evaluation dataset."
        )

    root_dir = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME / dataset_class / version
    if overwrite and root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if dataset_class == "TAFT":
        result = _standardize_taft(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery,
            label_path=label,
            RGB_bands=RGB_bands,
            label_polarity=label_polarity,
            subset_name=subset_name,
            progress_callback=progress_callback,
        )
    elif dataset_class == "DAFT":
        result = _standardize_daft(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery,
            RGB_bands=RGB_bands,
            subset_name=subset_name,
            upload_method=upload_method,
            tile_size=tile_size,
            stride=stride,
            daft_max_nodata_ratio=daft_max_nodata_ratio,
            slicing_aoi_path=slicing_aoi,
            resume=resume,
            progress_callback=progress_callback,
        )
    elif dataset_class == "TSFT":
        result = _standardize_tsft(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery,
            label_path=label,
            test_imagery_path=None,
            test_label_path=None,
            RGB_bands=RGB_bands,
            label_polarity=label_polarity,
            subset_name=subset_name,
            upload_method=upload_method,
            tile_size=tile_size,
            stride=stride,
            max_nodata_ratio=daft_max_nodata_ratio,
            slicing_aoi_path=slicing_aoi,
            min_polygon_overlap_ratio=tsft_min_polygon_overlap_ratio,
            min_positive_pixel_ratio=tsft_min_positive_pixel_ratio,
            no_feature_tile_policy=tsft_no_feature_tile_policy,
            negative_ratio=tsft_negative_ratio,
            resume=resume,
            progress_callback=progress_callback,
        )
    else:
        result = _standardize_evaluation(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery,
            label_path=label,
            RGB_bands=RGB_bands,
            label_polarity=label_polarity,
            subset_name=subset_name,
            upload_method=upload_method,
            tile_size=tile_size,
            stride=stride,
            max_nodata_ratio=daft_max_nodata_ratio,
            slicing_aoi_path=slicing_aoi,
            min_polygon_overlap_ratio=tsft_min_polygon_overlap_ratio,
            min_positive_pixel_ratio=tsft_min_positive_pixel_ratio,
            no_feature_tile_policy=tsft_no_feature_tile_policy,
            negative_ratio=tsft_negative_ratio,
            resume=resume,
            progress_callback=progress_callback,
        )

    previous_manifest = _read_json(result.manifest_path)
    previous_subsets = previous_manifest.get("subsets", []) if isinstance(previous_manifest, Mapping) else []
    subsets = _merge_subsets(previous_subsets, result.subsets)
    sample_count = sum(int(row.get("sample_count", 0)) for row in subsets)
    train_count = sum(int(row.get("train_count", 0)) for row in subsets)
    test_count = sum(int(row.get("test_count", 0)) for row in subsets)
    manifest = {
        **result.to_dict(),
        "subsets": subsets,
        "sample_count": sample_count,
        "train_count": train_count,
        "test_count": test_count,
        "source_imagery_path": str(imagery),
        "source_label_path": str(label) if label is not None else None,
        "source_slicing_aoi_path": str(slicing_aoi) if slicing_aoi is not None else None,
        "source_test_imagery_path": str(test_imagery) if test_imagery is not None else None,
        "source_test_label_path": str(test_label) if test_label is not None else None,
        "source": source,
        "notes": notes,
        "concat_ready": True,
            "standardization": {
                "sample_layout": "(H, W, C)",
                "normalization": "stored rasters keep compact source dtype when possible; training loaders convert to float32 0..1 on read",
                "dtype": result.sample_dtype,
                "image_dtype": result.sample_dtype,
                "label_dtype": "uint8" if result.dataset_class in SUPERVISED_CLASSES else None,
                "RGB_bands": RGB_bands,
                "RGB_band_indexes": list(rgb_bands_to_indexes(RGB_bands)),
                "label_polarity": label_polarity if result.dataset_class in SUPERVISED_CLASSES else None,
                "label_policy": "raster labels are standardized as compact uint8 masks with foreground=1 and background=0",
                "band_policy": "preserve all source bands; model adapters extract RGB_bands later",
            "upload_method": upload_method if result.dataset_class in {"DAFT", "TSFT", "Evaluation"} else "paired_images_labels",
            "tile_size": tile_size,
            "stride": stride,
            "max_nodata_ratio": (
                daft_max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            "slicing_aoi_path": (
                str(slicing_aoi) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and slicing_aoi is not None else None
            ),
            "min_polygon_overlap_ratio": (
                tsft_min_polygon_overlap_ratio if result.dataset_class in {"TSFT", "Evaluation"} else None
            ),
            "min_positive_pixel_ratio": (
                tsft_min_positive_pixel_ratio if result.dataset_class in {"TSFT", "Evaluation"} else None
            ),
            "no_feature_tile_policy": (
                tsft_no_feature_tile_policy if result.dataset_class in {"TSFT", "Evaluation"} else None
            ),
            "effective_no_feature_tile_policy": (
                TSFT_NO_FEATURE_POLICY_ALL
                if result.dataset_class == "TSFT"
                and upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
                and label is None
                else tsft_no_feature_tile_policy
                if result.dataset_class in {"TSFT", "Evaluation"}
                else None
            ),
            "negative_ratio": tsft_negative_ratio if result.dataset_class in {"TSFT", "Evaluation"} else None,
        },
    }
    _write_json(result.manifest_path, manifest, workspace_root=_workspace_root_from_store_root(store_root))
    if result.dataset_class == "DAFT":
        _update_daft_upload_state(
            result.root_dir,
            status="completed",
            processed=sample_count,
            total=sample_count,
            manifest_path=result.manifest_path,
        )
    return result


def build_dataset_repository_summary(*, store_root: str | Path) -> list[dict[str, Any]]:
    """Scan the standardized repository and return visualizable dataset/subset rows."""

    root = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for manifest_path in sorted(root.glob(f"*/*/{MANIFEST_NAME}")):
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            continue
        subset_rows = manifest.get("subsets")
        if isinstance(subset_rows, list) and subset_rows:
            for subset in subset_rows:
                if isinstance(subset, Mapping):
                    rows.append(_summary_row(manifest, subset, manifest_path))
        else:
            rows.append(_summary_row(manifest, {}, manifest_path))
    return rows


def inspect_daft_upload_target(
    *,
    store_root: str | Path,
    dataset_id: str,
    version: str,
    imagery_path: str | Path,
    RGB_bands: str,
    upload_method: str = DAFT_UPLOAD_METHOD_PACKAGED,
    tile_size: int | None = None,
    stride: int | None = None,
    daft_max_nodata_ratio: float | None = DEFAULT_DAFT_MAX_NODATA_RATIO,
    slicing_aoi_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return resumability metadata for a DAFT standardized target."""

    upload_method = _normalize_daft_upload_method(upload_method)
    config = _daft_upload_config(
        dataset_id=_required_text(dataset_id, "dataset_id"),
        version=_required_text(version, "version"),
        imagery_path=Path(imagery_path).expanduser().resolve(),
        RGB_bands=normalize_rgb_bands(RGB_bands),
        upload_method=upload_method,
        tile_size=tile_size,
        stride=stride,
        daft_max_nodata_ratio=daft_max_nodata_ratio,
        slicing_aoi_path=(
            Path(slicing_aoi_path).expanduser().resolve() if slicing_aoi_path not in {None, ""} else None
        ),
    )
    root_dir = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME / "DAFT" / config["version"]
    images_dir = root_dir / "images"
    manifest_path = root_dir / MANIFEST_NAME
    state_path = root_dir / DAFT_UPLOAD_STATE_NAME
    state = _read_json(state_path)
    state_map = state if isinstance(state, Mapping) else {}
    current_digest = _daft_config_digest(config)
    saved_digest = str(state_map.get("config_digest") or "")
    image_count = _count_repository_images(images_dir)
    has_target = root_dir.exists() and any(root_dir.iterdir())
    has_partial_images = image_count > 0 and not manifest_path.exists()
    config_matches = bool(saved_digest and saved_digest == current_digest)
    return {
        "exists": has_target,
        "root_dir": str(root_dir),
        "images_dir": str(images_dir),
        "manifest_path": str(manifest_path),
        "state_path": str(state_path),
        "has_manifest": manifest_path.exists(),
        "image_count": image_count,
        "has_partial_images": has_partial_images,
        "current_config": config,
        "current_config_digest": current_digest,
        "saved_config_digest": saved_digest,
        "config_matches": config_matches,
        "can_resume": bool(has_target and config_matches and image_count > 0),
        "state": dict(state_map),
    }



def create_concat_dataset_version(
    *,
    store_root: str | Path,
    dataset_class: str,
    new_version: str,
    source_versions: Sequence[str],
    dataset_id: str | None = None,
    name: str | None = None,
    source: str = "concat",
    notes: str = "",
) -> StandardizedDatasetResult:
    """Create a virtual DAFT/TSFT concat dataset-store version without copying samples."""

    dataset_class = _normalize_dataset_class(dataset_class)
    if dataset_class not in {"DAFT", "TSFT"}:
        raise DatasetStandardizationError("Concat datasets are only supported for DAFT or TSFT.")
    new_version = _required_text(new_version, "new_version")
    normalized_sources = [_required_text(version, "source_versions") for version in source_versions]
    if len(normalized_sources) < 2:
        raise DatasetStandardizationError("Concat datasets require at least two source_versions.")
    if len(set(normalized_sources)) != len(normalized_sources):
        raise DatasetStandardizationError("Concat source_versions must be unique.")

    workspace_root = _workspace_root_from_store_root(store_root)
    class_root = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME / dataset_class
    root_dir = class_root / new_version
    if root_dir.exists():
        raise DatasetStandardizationError(f"Concat dataset target already exists: {root_dir}.")

    sources: list[dict[str, Any]] = []
    source_manifests: list[Mapping[str, Any]] = []
    warnings: list[str] = []
    for version in normalized_sources:
        manifest_path = class_root / version / MANIFEST_NAME
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            raise DatasetStandardizationError(f"Missing source manifest for {dataset_class}/{version}: {manifest_path}")
        if manifest.get("dataset_class") != dataset_class:
            raise DatasetStandardizationError(f"Source {version} is not a {dataset_class} manifest.")
        source_manifests.append(manifest)
        sources.append(
            {
                "dataset_class": dataset_class,
                "version": version,
                "dataset_id": manifest.get("dataset_id", ""),
                "manifest_path": manifest_path,
                "root_dir": manifest.get("root_dir", str(manifest_path.parent)),
                "sample_count": _int_value(manifest.get("sample_count")),
                "train_count": _int_value(manifest.get("train_count")),
                "test_count": _int_value(manifest.get("test_count")),
                "subset_count": _manifest_subset_count(manifest),
                "has_gt": bool(manifest.get("has_gt", dataset_class == "TSFT")),
                "RGB_bands": manifest.get("RGB_bands", ""),
            }
        )

    sample_count = sum(_int_value(row.get("sample_count")) for row in sources)
    train_count = sum(_int_value(row.get("train_count")) for row in sources)
    test_count = sum(_int_value(row.get("test_count")) for row in sources)
    subset_count = sum(_int_value(row.get("subset_count")) for row in sources)
    rgb_values = [str(row.get("RGB_bands") or "") for row in sources if row.get("RGB_bands")]
    RGB_bands = rgb_values[0] if rgb_values else DEFAULT_RGB_BANDS
    if any(value != RGB_bands for value in rgb_values):
        warnings.append("Concat sources use mixed RGB_bands; manifest RGB_bands uses the first source value.")
    dtype_values = {str(manifest.get("sample_dtype") or "") for manifest in source_manifests if manifest.get("sample_dtype")}
    sample_dtype = next(iter(dtype_values)) if len(dtype_values) == 1 else ("mixed" if dtype_values else "source_dtype")
    layout_values = [str(manifest.get("sample_shape_layout") or "") for manifest in source_manifests if manifest.get("sample_shape_layout")]
    sample_shape_layout = layout_values[0] if layout_values else "(H, W, C)"

    subsets = [_concat_source_subset(source_row) for source_row in sources]
    created_at = _utc_now()
    dataset_id = _required_text(dataset_id or f"{dataset_class}-{new_version}", "dataset_id")
    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_class": dataset_class,
        "name": name or new_version,
        "version": new_version,
        "root_dir": root_dir,
        "imagery_path": root_dir,
        "label_path": root_dir if dataset_class == "TSFT" else None,
        "has_gt": dataset_class == "TSFT",
        "RGB_bands": RGB_bands,
        "sample_shape_layout": sample_shape_layout,
        "sample_dtype": sample_dtype,
        "sample_count": sample_count,
        "train_count": train_count,
        "test_count": test_count,
        "subset_count": subset_count,
        "subsets": subsets,
        "manifest_path": root_dir / MANIFEST_NAME,
        "created_at": created_at,
        "source": source,
        "notes": notes,
        "concat_ready": True,
        "is_concat": True,
        "concat_backend": "torch.utils.data.ConcatDataset",
        "concat_sources": sources,
        "warnings": warnings,
        "standardization": {
            "sample_layout": sample_shape_layout,
            "dtype": sample_dtype,
            "RGB_bands": RGB_bands,
            "RGB_band_indexes": list(rgb_bands_to_indexes(RGB_bands)),
            "upload_method": "concat",
            "virtual": True,
        },
    }
    if dataset_class == "TSFT":
        manifest.update(_concat_tsft_stats(source_manifests, normalized_sources, warnings))

    _write_json(root_dir / MANIFEST_NAME, manifest, workspace_root=workspace_root)
    return StandardizedDatasetResult(
        dataset_id=dataset_id,
        dataset_class=dataset_class,
        version=new_version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir if dataset_class == "TSFT" else None,
        has_gt=dataset_class == "TSFT",
        RGB_bands=RGB_bands,
        sample_shape_layout=sample_shape_layout,
        sample_dtype=sample_dtype,
        sample_count=sample_count,
        train_count=train_count,
        test_count=test_count,
        subsets=tuple(subsets),
        manifest_path=root_dir / MANIFEST_NAME,
        created_at=created_at,
    )

def browse_dataset_store(*, store_root: str | Path) -> dict[str, Any]:
    """Return folder-like TAFT/DAFT/TSFT/Evaluation repository contents for UI browsing."""

    root = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME
    classes: dict[str, Any] = {}
    for dataset_class in DATASET_CLASSES:
        class_root = root / dataset_class
        versions: list[dict[str, Any]] = []
        if class_root.exists():
            for version_dir in sorted(path for path in class_root.iterdir() if path.is_dir()):
                manifest_path = version_dir / MANIFEST_NAME
                manifest = _read_json(manifest_path)
                state = _read_json(version_dir / TAFT_TREE_STATE_NAME)
                subset_rows = manifest.get("subsets", []) if isinstance(manifest, Mapping) else []
                if not isinstance(subset_rows, list):
                    subset_rows = []
                versions.append(
                    {
                        "version": version_dir.name,
                        "path": str(version_dir),
                        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                        "has_manifest": manifest_path.exists(),
                        "dataset_id": manifest.get("dataset_id", "") if isinstance(manifest, Mapping) else "",
                        "sample_count": manifest.get("sample_count", _count_repository_images(version_dir)) if isinstance(manifest, Mapping) else _count_repository_images(version_dir),
                        "train_count": manifest.get("train_count", _count_repository_split_images(version_dir, "train")) if isinstance(manifest, Mapping) else _count_repository_split_images(version_dir, "train"),
                        "test_count": manifest.get("test_count", _count_repository_split_images(version_dir, "test")) if isinstance(manifest, Mapping) else _count_repository_split_images(version_dir, "test"),
                        "subset_count": manifest.get("subset_count", len(subset_rows)) if isinstance(manifest, Mapping) else len(subset_rows),
                        "has_gt": manifest.get("has_gt", False) if isinstance(manifest, Mapping) else False,
                        "sample_dtype": manifest.get("sample_dtype", "") if isinstance(manifest, Mapping) else "",
                        "sample_shape_layout": manifest.get("sample_shape_layout", "") if isinstance(manifest, Mapping) else "",
                        "is_concat": bool(manifest.get("is_concat", False)) if isinstance(manifest, Mapping) else False,
                        "concat_source_count": (
                            len(manifest.get("concat_sources", []))
                            if isinstance(manifest, Mapping) and isinstance(manifest.get("concat_sources"), list)
                            else 0
                        ),
                        "import_status": state.get("status", "") if isinstance(state, Mapping) else "",
                        "import_percent": state.get("percent", "") if isinstance(state, Mapping) else "",
                        "subsets": [_browse_subset_row(row) for row in subset_rows if isinstance(row, Mapping)],
                    }
                )
        classes[dataset_class] = {
            "path": str(class_root),
            "exists": class_root.exists(),
            "version_count": len(versions),
            "sample_count": sum(_int_value(row.get("sample_count")) for row in versions),
            "train_count": sum(_int_value(row.get("train_count")) for row in versions),
            "test_count": sum(_int_value(row.get("test_count")) for row in versions),
            "versions": versions,
        }
    return {"root": str(root), "classes": classes}


def delete_dataset_store_version(*, store_root: str | Path, dataset_class: str, version: str) -> dict[str, Any]:
    """Delete one standardized dataset-store version folder."""

    dataset_class = _normalize_dataset_class(dataset_class)
    version = _required_text(version, "version")
    root = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME
    target = root / dataset_class / version
    if not target.exists():
        return {
            "dataset_class": dataset_class,
            "version": version,
            "deleted_path": str(target),
            "already_missing": True,
        }
    if not target.is_dir():
        raise DatasetStandardizationError(f"Dataset-store version target is not a directory: {target}")
    _rmtree_dataset_store_target(target)
    return {
        "dataset_class": dataset_class,
        "version": version,
        "deleted_path": str(target),
        "already_missing": False,
    }


def _rmtree_dataset_store_target(target: Path) -> None:
    """Remove a dataset-store directory while tolerating disappearing partial files."""

    def onerror(func: Callable[..., Any], path: str, exc_info: tuple[Any, Any, Any]) -> None:
        exc = exc_info[1]
        if isinstance(exc, FileNotFoundError):
            return
        if isinstance(exc, PermissionError):
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                func(path)
                return
            except FileNotFoundError:
                return
            except OSError:
                pass
        raise exc

    if sys.version_info >= (3, 12):
        def onexc(func: Callable[..., Any], path: str, exc: BaseException) -> None:
            if isinstance(exc, FileNotFoundError):
                return
            if isinstance(exc, PermissionError):
                try:
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                    func(path)
                    return
                except FileNotFoundError:
                    return
                except OSError:
                    pass
            raise exc

        shutil.rmtree(target, onexc=onexc)
    else:
        shutil.rmtree(target, onerror=onerror)


def _count_repository_images(path: Path) -> int:
    if not path.exists():
        return 0
    suffixes = set(IMAGE_SUFFIXES)
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in suffixes)


def _daft_upload_config(
    *,
    dataset_id: str,
    version: str,
    imagery_path: Path,
    RGB_bands: str,
    upload_method: str,
    tile_size: int | None,
    stride: int | None,
    daft_max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None = None,
) -> dict[str, Any]:
    upload_method = _normalize_daft_upload_method(upload_method)
    normalized_ratio = (
        _normalize_ratio(daft_max_nodata_ratio, "daft_max_nodata_ratio")
        if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
        else None
    )
    return {
        "dataset_id": dataset_id,
        "version": version,
        "imagery_path": str(imagery_path),
        "RGB_bands": RGB_bands,
        "upload_method": upload_method,
        "tile_size": _positive_int(tile_size, "tile_size") if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "stride": _positive_int(stride, "stride") if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "max_nodata_ratio": normalized_ratio,
        "slicing_aoi_path": (
            str(slicing_aoi_path) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and slicing_aoi_path is not None else None
        ),
    }


def _daft_config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(_json_ready(dict(config)), sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_daft_target(root_dir: Path, upload_config: Mapping[str, Any], *, resume: bool) -> None:
    images_dir = root_dir / "images"
    if not images_dir.exists() or not any(images_dir.iterdir()):
        images_dir.mkdir(parents=True, exist_ok=True)
        return
    state = _read_json(root_dir / DAFT_UPLOAD_STATE_NAME)
    state_map = state if isinstance(state, Mapping) else {}
    config_matches = state_map.get("config_digest") == _daft_config_digest(upload_config)
    if resume and config_matches:
        return
    if resume and not config_matches:
        raise DatasetStandardizationError(
            f"Standardized DAFT target exists but its saved upload config does not match: {images_dir}. "
            "Choose overwrite to rebuild it from scratch."
        )
    raise DatasetStandardizationError(f"Standardized dataset target already exists: {images_dir}. Use overwrite=True.")


def _write_daft_upload_state(root_dir: Path, upload_config: Mapping[str, Any], *, status: str) -> None:
    state_path = root_dir / DAFT_UPLOAD_STATE_NAME
    previous = _read_json(state_path)
    previous_map = previous if isinstance(previous, Mapping) else {}
    state = {
        **previous_map,
        "dataset_id": upload_config.get("dataset_id", ""),
        "version": upload_config.get("version", ""),
        "dataset_class": "DAFT",
        "status": status,
        "config": dict(upload_config),
        "config_digest": _daft_config_digest(upload_config),
        "updated_at": _utc_now(),
    }
    if "created_at" not in state:
        state["created_at"] = state["updated_at"]
    _write_json(state_path, state)


def _update_daft_upload_state(
    root_dir: Path,
    *,
    status: str,
    processed: int,
    total: int,
    manifest_path: Path | None = None,
) -> None:
    state_path = root_dir / DAFT_UPLOAD_STATE_NAME
    state = _read_json(state_path)
    if not isinstance(state, Mapping):
        return
    updated = dict(state)
    updated.update(
        {
            "status": status,
            "processed": int(processed),
            "total": int(total),
            "percent": 1.0 if total else 1.0,
            "manifest_path": str(manifest_path) if manifest_path is not None else state.get("manifest_path", ""),
            "updated_at": _utc_now(),
        }
    )
    _write_json(state_path, updated)


def _count_repository_split_images(path: Path, split: str) -> int:
    if not path.exists():
        return 0
    suffixes = set(IMAGE_SUFFIXES)
    return sum(
        1
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in suffixes and split in file.parts
    )


def discover_taft_tree(root: str | Path, *, preset: str = "landslide77k") -> TaftTreeDiscovery:
    """Recursively discover train/test leaf subsets in a hierarchical TAFT tree."""

    preset = _normalize_taft_tree_preset(preset)
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise DatasetStandardizationError(f"TAFT tree root does not exist: {root_path}")
    if not root_path.is_dir():
        raise DatasetStandardizationError(f"TAFT tree root must be a directory: {root_path}")

    subsets: list[TaftTreeSubset] = []
    for directory in sorted(path for path in root_path.rglob("*") if path.is_dir()):
        if not _is_taft_leaf_subset(directory):
            continue
        rel = directory.relative_to(root_path).as_posix()
        train_image_dir = directory / "train" / "images"
        train_label_dir = directory / "train" / "labels"
        test_image_dir = directory / "test" / "images"
        test_label_dir = directory / "test" / "labels"
        train_images = _discover_images_in_dir(train_image_dir)
        train_labels = _discover_raster_labels_in_dir(train_label_dir)
        test_images = _discover_images_in_dir(test_image_dir)
        test_labels = _discover_raster_labels_in_dir(test_label_dir)
        errors = [
            *_pairing_errors("train", train_images, train_labels),
            *_pairing_errors("test", test_images, test_labels),
        ]
        warnings: list[str] = []
        RGB_bands, channel_warning = _taft_tree_RGB_bands(rel, preset=preset)
        if channel_warning:
            warnings.append(channel_warning)
        label_polarity, polarity_warning = _taft_tree_label_polarity(rel, preset=preset)
        if polarity_warning:
            warnings.append(polarity_warning)
        auxiliary_dirs = _discover_auxiliary_dirs(directory)
        subsets.append(
            TaftTreeSubset(
                source_rel=rel,
                subset_id=_subset_id_from_rel(rel),
                parent_group=rel.split("/", 1)[0],
                source_dir=directory,
                RGB_bands=RGB_bands,
                label_polarity=label_polarity,
                auto_RGB_bands=RGB_bands,
                auto_label_polarity=label_polarity,
                train_image_dir=train_image_dir,
                train_label_dir=train_label_dir,
                test_image_dir=test_image_dir,
                test_label_dir=test_label_dir,
                train_count=len(train_images),
                test_count=len(test_images),
                sample_count=len(train_images) + len(test_images),
                image_suffixes=_suffixes([*train_images, *test_images]),
                label_suffixes=_suffixes([*train_labels, *test_labels]),
                has_auxiliary=bool(auxiliary_dirs),
                auxiliary_dirs=tuple(auxiliary_dirs),
                errors=tuple(errors),
                warnings=tuple(warnings),
            )
        )

    errors: list[str] = []
    warnings: list[str] = []
    if not subsets:
        errors.append(f"No TAFT leaf subset found under {root_path}.")
    return TaftTreeDiscovery(
        root_path=root_path,
        preset=preset,
        subsets=tuple(subsets),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def register_taft_tree(
    *,
    store_root: str | Path,
    name: str | None = None,
    version: str,
    root: str | Path,
    preset: str = "landslide77k",
    standardize: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    force_rebuild: bool = False,
    append_to_existing: bool = False,
    split_policy: str = "preserve",
    source: str = "legacy",
    notes: str = "",
    subset_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TaftTreeRegistrationResult:
    """Register one hierarchical TAFT dataset version and optionally materialize a standardized copy."""

    if split_policy != "preserve":
        raise DatasetStandardizationError("TAFT tree import currently supports split_policy='preserve' only.")
    version = _required_text(version, "version")
    name = _required_text(name or version, "version")
    _emit_progress(
        progress_callback,
        {
            "status": "discovering",
            "processed": 0,
            "total": 0,
            "written": 0,
            "skipped": 0,
            "rewritten": 0,
            "percent": 0.0,
            "message": "Scanning TAFT source tree and validating image/label pairs.",
        },
    )
    discovery = discover_taft_tree(root, preset=preset)
    discovery_errors = _flatten_discovery_errors(discovery)
    if discovery_errors:
        preview = "; ".join(discovery_errors[:8])
        raise DatasetStandardizationError(f"TAFT tree discovery failed: {preview}")
    discovery = _apply_taft_subset_overrides(discovery, subset_overrides)
    annotations_digest = _taft_subset_annotations_digest(discovery)
    _emit_progress(
        progress_callback,
        {
            "status": "preparing",
            "processed": 0,
            "total": discovery.sample_count,
            "written": 0,
            "skipped": 0,
            "rewritten": 0,
            "percent": 0.0,
            "message": (
                f"Discovered {discovery.subset_count} subsets and "
                f"{discovery.sample_count} samples. Preparing standardized repository."
            ),
        },
    )

    dataset_id = f"TAFT-{_safe_name(version).lower()}"
    if append_to_existing and (overwrite or force_rebuild):
        raise DatasetStandardizationError("TAFT append mode cannot be combined with overwrite=True or force_rebuild=True.")
    if overwrite:
        if force_rebuild:
            raise DatasetStandardizationError("Use either overwrite=True or force_rebuild=True, not both.")
        # Backward compatibility: the former overwrite flag meant "reuse valid
        # files when the import state matches, and rebuild only invalid pairs".
        resume = True

    root_dir = Path(store_root).expanduser().resolve() / DATASET_STORE_DIRNAME / "TAFT" / version
    existing_manifest = _read_json(root_dir / MANIFEST_NAME)
    existing_subset_rows: list[dict[str, Any]] = []
    if append_to_existing:
        if not root_dir.exists() or not (root_dir / MANIFEST_NAME).exists():
            raise DatasetStandardizationError(f"TAFT version target does not exist for append: {root_dir}")
        if not isinstance(existing_manifest, Mapping) or existing_manifest.get("dataset_class") != "TAFT":
            raise DatasetStandardizationError(f"Existing TAFT manifest is invalid: {root_dir / MANIFEST_NAME}")
        raw_existing_subsets = existing_manifest.get("subsets", [])
        existing_subset_rows = [dict(row) for row in raw_existing_subsets if isinstance(row, Mapping)] if isinstance(raw_existing_subsets, list) else []
        _ensure_taft_append_has_no_subset_conflicts(discovery, existing_subset_rows)
    elif root_dir.exists() and any(root_dir.iterdir()):
        if force_rebuild:
            shutil.rmtree(root_dir)
        else:
            existing_state = _read_json(root_dir / TAFT_TREE_STATE_NAME)
            can_resume = _can_resume_taft_tree_import(
                state=existing_state,
                dataset_id=dataset_id,
                name=name,
                version=version,
                source_root=discovery.root_path,
                preset=discovery.preset,
                split_policy=split_policy,
                standardize=standardize,
                annotations_digest=annotations_digest,
            )
            if not resume:
                raise DatasetStandardizationError(
                    f"TAFT version target already exists: {root_dir}. "
                    "Use resume=True to continue a matching partial import, force_rebuild=True to delete and rebuild it, "
                    "or append_to_existing=True to add new leaf subsets to the existing version."
                )
            if not can_resume:
                raise DatasetStandardizationError(
                    f"TAFT version target already exists but cannot be resumed because the saved import state does not match "
                    f"the requested source/config: {root_dir}. Use force_rebuild=True to delete and rebuild it."
                )
    root_dir.mkdir(parents=True, exist_ok=True)
    if standardize:
        _ensure_output_space(root_dir, min_free_bytes=MIN_DATASET_STORE_FREE_BYTES)

    created_at = _utc_now()
    state_path = root_dir / TAFT_TREE_STATE_NAME
    events_path = root_dir / TAFT_TREE_EVENTS_NAME
    sample_total = discovery.sample_count
    import_state = _initial_taft_tree_state(
        dataset_id=dataset_id,
        name=name,
        version=version,
        source_root=discovery.root_path,
        preset=discovery.preset,
        split_policy=split_policy,
        standardize=standardize,
        annotations_digest=annotations_digest,
        sample_total=sample_total,
    )
    _write_json(state_path, import_state)
    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    subset_rows: list[dict[str, Any]] = []
    for subset in discovery.subsets:
        subset_target = root_dir / SUBSETS_DIRNAME / subset.subset_id
        standardized_samples: list[dict[str, Any]] = []
        target_auxiliary_dirs = _materialize_auxiliary_structure(subset, subset_target)
        if standardize:
            standardized_samples.extend(
                _standardize_taft_tree_split(
                    subset,
                    subset_target,
                    split="train",
                    counters=counters,
                    total_samples=sample_total,
                    state_path=state_path,
                    events_path=events_path,
                    progress_callback=progress_callback,
                )
            )
            standardized_samples.extend(
                _standardize_taft_tree_split(
                    subset,
                    subset_target,
                    split="test",
                    counters=counters,
                    total_samples=sample_total,
                    state_path=state_path,
                    events_path=events_path,
                    progress_callback=progress_callback,
                )
            )
        else:
            _ensure_taft_tree_split_dirs(subset_target)

        subset_manifest = {
            **subset.to_dict(),
            "dataset_id": dataset_id,
            "dataset_class": "TAFT",
            "version": version,
            "root_dir": str(subset_target),
            "standardized": bool(standardize),
            "split_policy": split_policy,
            "has_gt": True,
            "target_auxiliary_dirs": target_auxiliary_dirs,
            "samples": standardized_samples,
            "created_at": created_at,
        }
        _write_json(
            subset_target / "subset_manifest.json",
            subset_manifest,
            workspace_root=_workspace_root_from_store_root(store_root),
        )
        subset_rows.append(
            {
                "subset": subset.source_rel,
                "subset_id": subset.subset_id,
                "source_rel": subset.source_rel,
                "parent_group": subset.parent_group,
                "RGB_bands": subset.RGB_bands,
                "label_polarity": subset.label_polarity,
                "auto_RGB_bands": subset.auto_RGB_bands,
                "auto_label_polarity": subset.auto_label_polarity,
                "annotation_source": subset.annotation_source,
                "user_confirmed": subset.user_confirmed,
                "train_count": subset.train_count,
                "test_count": subset.test_count,
                "sample_count": subset.sample_count,
                "image_suffixes": list(subset.image_suffixes),
                "label_suffixes": list(subset.label_suffixes),
                "has_auxiliary": subset.has_auxiliary,
                "auxiliary_dirs": list(subset.auxiliary_dirs),
                "target_auxiliary_dirs": target_auxiliary_dirs,
                "has_label": True,
                "sample_layout": "(H, W, C)",
                "subset_manifest_path": str(subset_target / "subset_manifest.json"),
            }
        )
        _update_taft_tree_state(
            state_path,
            counters=counters,
            total_samples=sample_total,
            status="running",
            current_subset=subset.source_rel,
        )

    all_subset_rows = [*existing_subset_rows, *subset_rows] if append_to_existing else subset_rows
    aggregate_sample_count = sum(int(row.get("sample_count", 0)) for row in all_subset_rows)
    aggregate_train_count = sum(int(row.get("train_count", 0)) for row in all_subset_rows)
    aggregate_test_count = sum(int(row.get("test_count", 0)) for row in all_subset_rows)
    result = TaftTreeRegistrationResult(
        dataset_id=dataset_id,
        dataset_class="TAFT",
        name=name,
        version=version,
        root_dir=root_dir,
        manifest_path=root_dir / MANIFEST_NAME,
        source_root=discovery.root_path,
        preset=discovery.preset,
        standardized=bool(standardize),
        split_policy=split_policy,
        has_gt=True,
        RGB_bands="mixed",
        sample_shape_layout="(H, W, C)",
        sample_dtype="source_dtype",
        sample_count=aggregate_sample_count,
        train_count=aggregate_train_count,
        test_count=aggregate_test_count,
        subset_count=len(all_subset_rows),
        created_at=created_at,
        subsets=tuple(all_subset_rows),
        skipped_samples=counters["skipped"],
        written_samples=counters["written"],
        rewritten_samples=counters["rewritten"],
    )
    extension_sources = list(existing_manifest.get("extension_sources", [])) if append_to_existing and isinstance(existing_manifest, Mapping) else []
    if append_to_existing:
        extension_sources.append(
            {
                "source_root": str(discovery.root_path),
                "preset": discovery.preset,
                "subset_count": len(subset_rows),
                "sample_count": discovery.sample_count,
                "subsets": [row.get("source_rel", row.get("subset", "")) for row in subset_rows],
                "created_at": created_at,
            }
        )
    manifest_base = dict(existing_manifest) if append_to_existing and isinstance(existing_manifest, Mapping) else {}
    manifest = {
        **manifest_base,
        **result.to_dict(),
        "created_at": manifest_base.get("created_at", result.created_at),
        "updated_at": created_at,
        "dataset_type": "TAFT",
        "source": source,
        "notes": notes or str(manifest_base.get("notes", "")),
        "source_root": str(manifest_base.get("source_root", str(discovery.root_path))) if append_to_existing else str(discovery.root_path),
        "source_imagery_path": str(manifest_base.get("source_imagery_path", str(discovery.root_path))) if append_to_existing else str(discovery.root_path),
        "source_label_path": str(manifest_base.get("source_label_path", str(discovery.root_path))) if append_to_existing else str(discovery.root_path),
        "latest_import_source_root": str(discovery.root_path),
        "append_to_existing": bool(append_to_existing),
        "extension_sources": extension_sources,
        "import_state_path": str(state_path),
        "import_events_path": str(events_path),
        "resume_enabled": bool(resume),
        "force_rebuild": bool(force_rebuild),
        "concat_ready": True,
        "standardization": {
            "sample_layout": "(H, W, C)",
            "normalization": "stored rasters keep compact source dtype when possible; training loaders convert to float32 0..1 on read",
            "image_dtype": "source_dtype",
            "label_dtype": "uint8",
            "label_foreground_value": 1,
            "label_policy": "label_polarity is user-confirmed per subset; stored labels are compact uint8 masks with foreground=1 and background=0",
            "annotation_policy": "TAFT import requires user-confirmed RGB_bands and label_polarity for every leaf subset",
            "band_policy": "preserve all source bands; subset RGB_bands is consumed later by adapters",
            "split_policy": split_policy,
            "auxiliary_policy": "record source auxiliary directories and mirror empty target directory structure only",
        },
    }
    _write_json(result.manifest_path, manifest, workspace_root=_workspace_root_from_store_root(store_root))
    _update_taft_tree_state(
        state_path,
        counters=counters,
        total_samples=sample_total,
        status="completed",
        current_subset="",
    )
    _emit_progress(
        progress_callback,
        {
            "status": "completed",
            "processed": counters["processed"],
            "total": sample_total,
            "written": counters["written"],
            "skipped": counters["skipped"],
            "rewritten": counters["rewritten"],
            "percent": 1.0 if sample_total else 1.0,
            "message": "TAFT tree import completed.",
        },
    )
    return result


def _standardize_taft(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    label_path: Path | None,
    RGB_bands: str,
    label_polarity: str,
    subset_name: str | None,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    image_files = _discover_images(imagery_path)
    label_files = _match_labels(image_files, label_path) if label_path is not None else {}
    subset = _subset_name(subset_name)
    _ensure_new_target(root_dir / subset)
    train_files, test_files = _split_files(image_files, test_fraction=0.05)
    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    total_samples = len(train_files) + len(test_files)
    train_samples = _write_split(
        files=train_files,
        split_dir=root_dir / subset / "train",
        split="train",
        labels=label_files,
        RGB_bands=RGB_bands,
        label_polarity=label_polarity,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=total_samples,
        progress_message_prefix=f"TAFT/{version}/{subset}",
    )
    test_samples = _write_split(
        files=test_files,
        split_dir=root_dir / subset / "test",
        split="test",
        labels=label_files,
        RGB_bands=RGB_bands,
        label_polarity=label_polarity,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=total_samples,
        progress_message_prefix=f"TAFT/{version}/{subset}",
    )
    return _result(
        dataset_id=dataset_id,
        dataset_class="TAFT",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir / subset,
        has_gt=True,
        RGB_bands=RGB_bands,
        subsets=[
            {
                **_subset_stats(subset, train_samples, test_samples, has_label=True),
                "RGB_bands": RGB_bands,
                "label_polarity": label_polarity,
                "annotation_source": "user_confirmed",
                "user_confirmed": True,
            }
        ],
    )


def _standardize_daft(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    RGB_bands: str,
    subset_name: str | None,
    upload_method: str,
    tile_size: int | None,
    stride: int | None,
    daft_max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    upload_method = _normalize_daft_upload_method(upload_method)
    upload_config = _daft_upload_config(
        dataset_id=dataset_id,
        version=version,
        imagery_path=imagery_path,
        RGB_bands=RGB_bands,
        upload_method=upload_method,
        tile_size=tile_size,
        stride=stride,
        daft_max_nodata_ratio=daft_max_nodata_ratio,
        slicing_aoi_path=slicing_aoi_path,
    )
    _prepare_daft_target(root_dir, upload_config, resume=resume)
    _write_daft_upload_state(root_dir, upload_config, status="running")
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        samples = [
            result.to_sample()
            for result in _slice_rasters_to_daft_images(
                raster_path=imagery_path,
                output_dir=root_dir / "images",
                RGB_bands=RGB_bands,
                tile_size=tile_size,
                stride=stride,
                max_nodata_ratio=daft_max_nodata_ratio,
                slicing_aoi_path=slicing_aoi_path,
                resume_existing=resume,
                progress_callback=progress_callback,
                progress_message_prefix=f"DAFT/{version}/images",
            )
        ]
        max_nodata = _normalize_ratio(daft_max_nodata_ratio, "daft_max_nodata_ratio")
        tile_schema = _append_slicing_aoi_tile_schema(
            f"tiff_slicing;tile_size={int(tile_size or 0)};stride={int(stride or 0)};max_nodata_ratio={max_nodata:.2f}",
            slicing_aoi_path=slicing_aoi_path,
        )
    else:
        image_files = _discover_images(imagery_path)
        counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
        samples = _write_split(
            files=image_files,
            split_dir=root_dir / "images",
            split="all",
            labels={},
            RGB_bands=RGB_bands,
            direct_images_dir=True,
            resume_existing=resume,
            progress_callback=progress_callback,
            progress_counters=counters,
            progress_total=len(image_files),
            progress_message_prefix=f"DAFT/{version}/images",
        )
        tile_schema = "packaged_images"
    subset = _subset_stats(subset_name or "images", samples, [], has_label=False)
    subset["RGB_bands"] = RGB_bands
    subset["upload_method"] = upload_method
    subset["tile_schema"] = tile_schema
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and slicing_aoi_path is not None:
        subset["slicing_aoi_path"] = str(slicing_aoi_path)
    return _result(
        dataset_id=dataset_id,
        dataset_class="DAFT",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=None,
        has_gt=False,
        RGB_bands=RGB_bands,
        subsets=[subset],
    )


def _standardize_tsft(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    label_path: Path | None,
    test_imagery_path: Path | None,
    test_label_path: Path | None,
    RGB_bands: str,
    label_polarity: str,
    subset_name: str | None,
    upload_method: str,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None,
    min_polygon_overlap_ratio: float | None,
    min_positive_pixel_ratio: float | None,
    no_feature_tile_policy: str,
    negative_ratio: float | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    upload_method = _normalize_upload_method(upload_method)
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        if test_imagery_path is not None or test_label_path is not None:
            raise DatasetStandardizationError("TSFT Tiff Slicing does not support optional test split paths.")
        return _standardize_tsft_tiff_slicing(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery_path,
            vector_label_path=label_path,
            RGB_bands=RGB_bands,
            label_polarity=label_polarity,
            tile_size=tile_size,
            stride=stride,
            max_nodata_ratio=max_nodata_ratio,
            slicing_aoi_path=slicing_aoi_path,
            min_polygon_overlap_ratio=min_polygon_overlap_ratio,
            min_positive_pixel_ratio=min_positive_pixel_ratio,
            no_feature_tile_policy=no_feature_tile_policy,
            negative_ratio=negative_ratio,
            subset_name=subset_name,
            resume=resume,
            progress_callback=progress_callback,
        )
    if resume:
        raise DatasetStandardizationError("TSFT packaged upload does not support resume; use overwrite to rebuild.")
    image_files = _discover_images(imagery_path)
    label_files = _match_labels(image_files, label_path)
    _ensure_new_target(root_dir)
    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    total_samples = len(image_files)
    train_samples = _write_split(
        files=image_files,
        split_dir=root_dir / "train",
        split="train",
        labels=label_files,
        RGB_bands=RGB_bands,
        label_polarity=label_polarity,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=total_samples,
        progress_message_prefix=f"TSFT/{version}",
    )
    subset_label = _subset_name(subset_name) if subset_name else version
    subset = _subset_stats(subset_label, train_samples, [], has_label=True)
    subset["upload_method"] = DAFT_UPLOAD_METHOD_PACKAGED
    subset["tile_schema"] = "packaged_images_and_labels"
    subset["RGB_bands"] = RGB_bands
    subset["label_polarity"] = label_polarity
    subset["annotation_source"] = "user_confirmed"
    subset["user_confirmed"] = True
    return _result(
        dataset_id=dataset_id,
        dataset_class="TSFT",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir,
        has_gt=True,
        RGB_bands=RGB_bands,
        subsets=[subset],
    )


def _standardize_evaluation(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    label_path: Path | None,
    RGB_bands: str,
    label_polarity: str,
    subset_name: str | None,
    upload_method: str,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None,
    min_polygon_overlap_ratio: float | None,
    min_positive_pixel_ratio: float | None,
    no_feature_tile_policy: str,
    negative_ratio: float | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    upload_method = _normalize_upload_method(upload_method)
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        return _standardize_evaluation_tiff_slicing(
            dataset_id=dataset_id,
            version=version,
            root_dir=root_dir,
            imagery_path=imagery_path,
            vector_label_path=label_path,
            RGB_bands=RGB_bands,
            label_polarity=label_polarity,
            tile_size=tile_size,
            stride=stride,
            max_nodata_ratio=max_nodata_ratio,
            slicing_aoi_path=slicing_aoi_path,
            min_polygon_overlap_ratio=min_polygon_overlap_ratio,
            min_positive_pixel_ratio=min_positive_pixel_ratio,
            no_feature_tile_policy=no_feature_tile_policy,
            negative_ratio=negative_ratio,
            subset_name=subset_name,
            resume=resume,
            progress_callback=progress_callback,
        )
    if resume:
        raise DatasetStandardizationError("Evaluation packaged upload does not support resume; use overwrite to rebuild.")
    if label_path is None:
        raise DatasetStandardizationError("Evaluation datasets require label_path.")
    image_files = _discover_images(imagery_path)
    label_files = _match_labels(image_files, label_path)
    _ensure_new_target(root_dir)
    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    eval_samples = _write_split(
        files=image_files,
        split_dir=root_dir / "eval",
        split="eval",
        labels=label_files,
        RGB_bands=RGB_bands,
        label_polarity=label_polarity,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=len(image_files),
        progress_message_prefix=f"Evaluation/{version}",
    )
    subset_label = _subset_name(subset_name) if subset_name else version
    subset = _subset_stats(subset_label, [], eval_samples, has_label=True)
    subset["eval_count"] = len(eval_samples)
    subset["upload_method"] = DAFT_UPLOAD_METHOD_PACKAGED
    subset["tile_schema"] = "evaluation_images_and_labels"
    subset["RGB_bands"] = RGB_bands
    subset["label_polarity"] = label_polarity
    subset["annotation_source"] = "user_confirmed"
    subset["user_confirmed"] = True
    return _result(
        dataset_id=dataset_id,
        dataset_class="Evaluation",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir,
        has_gt=True,
        RGB_bands=RGB_bands,
        subsets=[subset],
    )


def _standardize_evaluation_tiff_slicing(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    vector_label_path: Path | None,
    RGB_bands: str,
    label_polarity: str,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None,
    min_polygon_overlap_ratio: float | None,
    min_positive_pixel_ratio: float | None,
    no_feature_tile_policy: str,
    negative_ratio: float | None,
    subset_name: str | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    if vector_label_path is None:
        raise DatasetStandardizationError("Evaluation Tiff Slicing requires a polygon label path.")
    tile_size_value = _positive_int(tile_size, "tile_size")
    stride_value = _positive_int(stride, "stride")
    max_nodata = _normalize_ratio_with_default(max_nodata_ratio, "max_nodata_ratio", DEFAULT_DAFT_MAX_NODATA_RATIO)
    min_overlap = _normalize_ratio_with_default(
        min_polygon_overlap_ratio,
        "min_polygon_overlap_ratio",
        DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
    )
    min_positive = _normalize_ratio_with_default(
        min_positive_pixel_ratio,
        "min_positive_pixel_ratio",
        DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
    )
    policy = _normalize_tsft_no_feature_policy(no_feature_tile_policy)
    negative_limit_ratio = _normalize_nonnegative_float(
        negative_ratio,
        "negative_ratio",
        DEFAULT_TSFT_NEGATIVE_RATIO,
    )
    if not resume:
        _ensure_new_target(root_dir)
    eval_dir = root_dir / "eval"
    (eval_dir / "images").mkdir(parents=True, exist_ok=True)
    (eval_dir / "labels").mkdir(parents=True, exist_ok=True)

    slicing_aoi = _read_slicing_aoi(slicing_aoi_path) if slicing_aoi_path is not None else None
    raster_files = _discover_tiff_rasters(
        imagery_path,
        RGB_bands=RGB_bands,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
    )
    vector_labels = _read_tsft_vector_labels(vector_label_path)
    positive_windows, negative_windows, slice_stats = _collect_tsft_slice_windows(
        raster_files=raster_files,
        vector_labels=vector_labels,
        vector_label_path=vector_label_path,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
        RGB_bands=RGB_bands,
        tile_size=tile_size_value,
        stride=stride_value,
        max_nodata_ratio=max_nodata,
        min_polygon_overlap_ratio=min_overlap,
        min_positive_pixel_ratio=min_positive,
    )
    if policy == TSFT_NO_FEATURE_POLICY_ONLY:
        selected_negatives: list[TsftSliceWindow] = []
    elif policy == TSFT_NO_FEATURE_POLICY_ALL:
        selected_negatives = list(negative_windows)
    else:
        limit = int(len(positive_windows) * negative_limit_ratio)
        selected_negatives = list(negative_windows[:limit])
    selected_windows = [*positive_windows, *selected_negatives]
    if not selected_windows:
        raise DatasetStandardizationError(f"Evaluation Tiff Slicing produced no image/label chips under {imagery_path}.")

    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    eval_samples = _write_tsft_slice_windows(
        windows=selected_windows,
        vector_labels=vector_labels,
        vector_label_path=vector_label_path,
        output_split_dir=eval_dir,
        output_split_name="eval",
        min_polygon_overlap_ratio=min_overlap,
        min_positive_pixel_ratio=min_positive,
        resume_existing=resume,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=len(selected_windows),
        progress_message_prefix=f"Evaluation/{version}/eval",
    )
    subset_label = _subset_name(subset_name) if subset_name else version
    subset = _subset_stats(subset_label, [], eval_samples, has_label=True)
    subset.update(
        {
            "eval_count": len(eval_samples),
            "upload_method": DAFT_UPLOAD_METHOD_TIFF_SLICING,
            "RGB_bands": RGB_bands,
            "label_polarity": label_polarity,
            "label_source": "polygon",
            "annotation_source": "user_confirmed",
            "user_confirmed": True,
            "tile_schema": _tsft_tiff_tile_schema(
                tile_size=tile_size_value,
                stride=stride_value,
                max_nodata_ratio=max_nodata,
                min_polygon_overlap_ratio=min_overlap,
                min_positive_pixel_ratio=min_positive,
                no_feature_tile_policy=policy,
                negative_ratio=negative_limit_ratio,
                slicing_aoi_path=slicing_aoi_path,
            ),
            "slicing_aoi_path": str(slicing_aoi_path) if slicing_aoi_path is not None else None,
            "positive_count": len(positive_windows),
            "negative_count": len(selected_negatives),
            "negative_candidate_count": len(negative_windows),
            "mean_positive_pixel_ratio": _mean_tsft_positive_pixel_ratio(positive_windows),
            **slice_stats,
        }
    )
    return _result(
        dataset_id=dataset_id,
        dataset_class="Evaluation",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir,
        has_gt=True,
        RGB_bands=RGB_bands,
        subsets=[subset],
    )


def _standardize_tsft_tiff_slicing(
    *,
    dataset_id: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    vector_label_path: Path | None,
    RGB_bands: str,
    label_polarity: str,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None,
    min_polygon_overlap_ratio: float | None,
    min_positive_pixel_ratio: float | None,
    no_feature_tile_policy: str,
    negative_ratio: float | None,
    subset_name: str | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
) -> StandardizedDatasetResult:
    tile_size_value = _positive_int(tile_size, "tile_size")
    stride_value = _positive_int(stride, "stride")
    max_nodata = _normalize_ratio_with_default(max_nodata_ratio, "max_nodata_ratio", DEFAULT_DAFT_MAX_NODATA_RATIO)
    min_overlap = _normalize_ratio_with_default(
        min_polygon_overlap_ratio,
        "min_polygon_overlap_ratio",
        DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
    )
    min_positive = _normalize_ratio_with_default(
        min_positive_pixel_ratio,
        "min_positive_pixel_ratio",
        DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
    )
    policy = _normalize_tsft_no_feature_policy(no_feature_tile_policy)
    negative_limit_ratio = _normalize_nonnegative_float(
        negative_ratio,
        "negative_ratio",
        DEFAULT_TSFT_NEGATIVE_RATIO,
    )
    if not resume:
        _ensure_new_target(root_dir)
    train_dir = root_dir / "train"
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_dir / "labels").mkdir(parents=True, exist_ok=True)

    slicing_aoi = _read_slicing_aoi(slicing_aoi_path) if slicing_aoi_path is not None else None
    raster_files = _discover_tiff_rasters(
        imagery_path,
        RGB_bands=RGB_bands,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
    )
    vector_labels = _read_tsft_vector_labels(vector_label_path) if vector_label_path is not None else None
    has_polygon_labels = vector_labels is not None
    positive_windows, negative_windows, slice_stats = _collect_tsft_slice_windows(
        raster_files=raster_files,
        vector_labels=vector_labels,
        vector_label_path=vector_label_path,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
        RGB_bands=RGB_bands,
        tile_size=tile_size_value,
        stride=stride_value,
        max_nodata_ratio=max_nodata,
        min_polygon_overlap_ratio=min_overlap,
        min_positive_pixel_ratio=min_positive,
    )
    small_aoi_fallback_windows: list[TsftSliceWindow] = []
    if not has_polygon_labels and slicing_aoi is not None and slicing_aoi_path is not None:
        small_aoi_fallback_windows, small_aoi_fallback_stats = _collect_tsft_negative_small_aoi_fallback_windows(
            raster_files=raster_files,
            slicing_aoi=slicing_aoi,
            slicing_aoi_path=slicing_aoi_path,
            RGB_bands=RGB_bands,
            tile_size=tile_size_value,
            max_nodata_ratio=max_nodata,
            existing_windows=negative_windows,
        )
        negative_windows = [*negative_windows, *small_aoi_fallback_windows]
        slice_stats.update(small_aoi_fallback_stats)
    else:
        slice_stats.update(
            {
                "small_aoi_fallback_count": 0,
                "skipped_small_aoi_empty": 0,
                "small_aoi_transform_policy": None,
            }
        )
    selection_policy = TSFT_NO_FEATURE_POLICY_ALL if not has_polygon_labels else policy
    if selection_policy == TSFT_NO_FEATURE_POLICY_ONLY:
        selected_negatives: list[TsftSliceWindow] = []
    elif selection_policy == TSFT_NO_FEATURE_POLICY_ALL:
        selected_negatives = list(negative_windows)
    else:
        limit = int(len(positive_windows) * negative_limit_ratio)
        selected_negatives = list(negative_windows[:limit])
    selected_windows = [*positive_windows, *selected_negatives]
    if not selected_windows:
        raise DatasetStandardizationError(f"TSFT Tiff Slicing produced no image/label chips under {imagery_path}.")

    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    train_samples = _write_tsft_slice_windows(
        windows=selected_windows,
        vector_labels=vector_labels,
        vector_label_path=vector_label_path,
        output_split_dir=train_dir,
        min_polygon_overlap_ratio=min_overlap,
        min_positive_pixel_ratio=min_positive,
        resume_existing=resume,
        progress_callback=progress_callback,
        progress_counters=counters,
        progress_total=len(selected_windows),
        progress_message_prefix=f"TSFT/{version}/train",
    )
    subset_label = _subset_name(subset_name) if subset_name else version
    subset = _subset_stats(subset_label, train_samples, [], has_label=True)
    subset.update(
        {
            "upload_method": DAFT_UPLOAD_METHOD_TIFF_SLICING,
            "RGB_bands": RGB_bands,
            "label_polarity": label_polarity,
            "label_source": "polygon" if has_polygon_labels else "none_negative_only",
            "annotation_source": "user_confirmed",
            "user_confirmed": True,
            "tile_schema": _tsft_tiff_tile_schema(
                tile_size=tile_size_value,
                stride=stride_value,
                max_nodata_ratio=max_nodata,
                min_polygon_overlap_ratio=min_overlap,
                min_positive_pixel_ratio=min_positive,
                no_feature_tile_policy=selection_policy,
                negative_ratio=negative_limit_ratio,
                slicing_aoi_path=slicing_aoi_path,
            ),
            "slicing_aoi_path": str(slicing_aoi_path) if slicing_aoi_path is not None else None,
            "positive_count": len(positive_windows),
            "negative_count": len(selected_negatives),
            "negative_candidate_count": len(negative_windows),
            "mean_positive_pixel_ratio": _mean_tsft_positive_pixel_ratio(positive_windows),
            **slice_stats,
        }
    )
    return _result(
        dataset_id=dataset_id,
        dataset_class="TSFT",
        version=version,
        root_dir=root_dir,
        imagery_path=root_dir,
        label_path=root_dir,
        has_gt=True,
        RGB_bands=RGB_bands,
        subsets=[subset],
    )


def _collect_tsft_slice_windows(
    *,
    raster_files: Sequence[Path],
    vector_labels: Any | None,
    vector_label_path: Path | None,
    slicing_aoi: Any | None,
    slicing_aoi_path: Path | None,
    RGB_bands: str,
    tile_size: int,
    stride: int,
    max_nodata_ratio: float,
    min_polygon_overlap_ratio: float,
    min_positive_pixel_ratio: float,
) -> tuple[list[TsftSliceWindow], list[TsftSliceWindow], dict[str, Any]]:
    import numpy as np

    rasterio = require_rasterio()
    from rasterio.windows import Window

    rgb_indexes = rgb_bands_to_indexes(RGB_bands)
    abnormal_threshold = -1.0
    zero_tolerance = 1e-6
    positive_windows: list[TsftSliceWindow] = []
    negative_windows: list[TsftSliceWindow] = []
    stats = {
        "total_windows": _count_daft_windows(
            raster_files,
            tile_size=tile_size,
            stride=stride,
            slicing_aoi=slicing_aoi,
            slicing_aoi_path=slicing_aoi_path,
        ),
        "skipped_nodata": 0,
        "skipped_empty_mask": 0,
        "skipped_zero_rgb": 0,
        "skipped_outside_aoi": 0,
        "low_positive_windows": 0,
    }
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            missing = [index for index in rgb_indexes if index > int(src.count)]
            if missing:
                raise DatasetStandardizationError(
                    f"RGB_bands {RGB_bands} requires band(s) {missing}, but {raster_file} only has {src.count} band(s)."
                )
            geometries = (
                _tsft_geometries_for_raster(vector_labels, vector_label_path, src.crs)
                if vector_labels is not None and vector_label_path is not None
                else []
            )
            aoi_filter = _slicing_aoi_filter_for_raster(slicing_aoi, slicing_aoi_path, src.crs)
            all_rows = _window_starts(src.height, tile_size, stride)
            all_cols = _window_starts(src.width, tile_size, stride)
            rows, cols = _candidate_window_starts_for_aoi(
                src,
                tile_size=tile_size,
                stride=stride,
                aoi_geometry=aoi_filter,
            )
            if aoi_filter is not None:
                stats["skipped_outside_aoi"] += len(all_rows) * len(all_cols) - len(rows) * len(cols)
            for row_off in rows:
                for col_off in cols:
                    height = min(tile_size, src.height - row_off)
                    width = min(tile_size, src.width - col_off)
                    if height <= 0 or width <= 0:
                        continue
                    window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
                    if aoi_filter is not None and not _window_is_strictly_inside_aoi(
                        aoi_filter,
                        window=window,
                        source_transform=src.transform,
                    ):
                        stats["skipped_outside_aoi"] += 1
                        continue
                    masks = [src.read_masks(b, window=window) > 0 for b in rgb_indexes]
                    valid_mask = np.any(np.stack(masks, axis=0), axis=0)
                    if not bool(np.any(valid_mask)):
                        stats["skipped_empty_mask"] += 1
                        continue
                    array = src.read(window=window).astype("float32", copy=False)
                    array[array < abnormal_threshold] = 0
                    rgb_array = extract_rgb_bands(array, RGB_bands, layout="chw")
                    any_nonzero_valid = np.any(np.abs(rgb_array) > zero_tolerance, axis=0) & valid_mask
                    valid_pixels = int(np.count_nonzero(any_nonzero_valid))
                    if valid_pixels <= 0:
                        stats["skipped_zero_rgb"] += 1
                        continue
                    nodata_ratio = 1.0 - (float(valid_pixels) / float(height * width))
                    if nodata_ratio > max_nodata_ratio:
                        stats["skipped_nodata"] += 1
                        continue

                    if geometries:
                        label_mask = _rasterize_tsft_window_label(
                            geometries=geometries,
                            window=window,
                            transform=src.window_transform(window),
                            source_transform=src.transform,
                            height=int(height),
                            width=int(width),
                            min_polygon_overlap_ratio=min_polygon_overlap_ratio,
                        )
                        positive_pixels = int(np.count_nonzero(label_mask > 0))
                    else:
                        positive_pixels = 0
                    if positive_pixels > 0:
                        positive_ratio = float(positive_pixels) / float(max(valid_pixels, 1))
                        if positive_ratio < min_positive_pixel_ratio:
                            stats["low_positive_windows"] += 1
                            continue
                    sample = TsftSliceWindow(
                        sample_id=f"{_safe_name(raster_file.stem)}_r{int(row_off):06d}_c{int(col_off):06d}",
                        source_raster=raster_file,
                        row_off=int(row_off),
                        col_off=int(col_off),
                        height=int(height),
                        width=int(width),
                        has_positive_label=positive_pixels > 0,
                        label_positive_pixels=positive_pixels,
                        nodata_ratio=nodata_ratio,
                        valid_pixels=valid_pixels,
                    )
                    if sample.has_positive_label:
                        positive_windows.append(sample)
                    else:
                        negative_windows.append(sample)
    return positive_windows, negative_windows, stats


def _collect_tsft_negative_small_aoi_fallback_windows(
    *,
    raster_files: Sequence[Path],
    slicing_aoi: Any,
    slicing_aoi_path: Path,
    RGB_bands: str,
    tile_size: int,
    max_nodata_ratio: float,
    existing_windows: Sequence[TsftSliceWindow],
) -> tuple[list[TsftSliceWindow], dict[str, Any]]:
    import numpy as np

    rasterio = require_rasterio()

    rgb_indexes = rgb_bands_to_indexes(RGB_bands)
    abnormal_threshold = -1.0
    zero_tolerance = 1e-6
    best_windows: dict[int, tuple[TsftSliceWindow, float]] = {}
    candidate_indexes: set[int] = set()
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            aoi_geometries = _polygon_geometries_for_raster(
                slicing_aoi,
                slicing_aoi_path,
                src.crs,
                purpose="slicing AOI",
            )
            raster_geometry = _raster_bounds_geometry(src)
            fixed_windows = [
                window for window in existing_windows
                if Path(window.source_raster) == Path(raster_file) and window.window_mode == "fixed"
            ]
            for aoi_index, geometry in enumerate(aoi_geometries, start=1):
                candidate_indexes.add(aoi_index)
                if geometry is None or geometry.is_empty or not geometry.intersects(raster_geometry):
                    continue
                intersection = geometry.intersection(raster_geometry)
                if intersection.is_empty:
                    continue
                try:
                    crop_window = _covering_window_from_bounds(intersection.bounds, transform=src.transform)
                except Exception:
                    continue
                crop_window = crop_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                height = int(crop_window.height)
                width = int(crop_window.width)
                row_off = int(crop_window.row_off)
                col_off = int(crop_window.col_off)
                if height <= 0 or width <= 0:
                    continue
                masks = [src.read_masks(b, window=crop_window) > 0 for b in rgb_indexes]
                valid_mask = np.any(np.stack(masks, axis=0), axis=0)
                if not bool(np.any(valid_mask)):
                    continue
                array = src.read(window=crop_window).astype("float32", copy=False)
                array[array < abnormal_threshold] = 0
                rgb_array = extract_rgb_bands(array, RGB_bands, layout="chw")
                polygon_mask = _rasterize_geometry_mask(
                    geometry=intersection,
                    transform=src.window_transform(crop_window),
                    height=height,
                    width=width,
                    all_touched=True,
                )
                candidate_mask = valid_mask & polygon_mask
                any_nonzero_valid = np.any(np.abs(rgb_array) > zero_tolerance, axis=0) & candidate_mask
                valid_pixels = int(np.count_nonzero(any_nonzero_valid))
                if valid_pixels <= 0:
                    continue
                nodata_ratio = 1.0 - (float(valid_pixels) / float(height * width))
                window = TsftSliceWindow(
                    sample_id=f"{_safe_name(raster_file.stem)}_aoi{aoi_index:06d}_letterbox",
                    source_raster=raster_file,
                    row_off=row_off,
                    col_off=col_off,
                    height=height,
                    width=width,
                    has_positive_label=False,
                    label_positive_pixels=0,
                    nodata_ratio=nodata_ratio,
                    valid_pixels=valid_pixels,
                    window_mode="aoi_letterbox",
                    aoi_geometry=intersection,
                    target_size=tile_size,
                )
                score = float(valid_pixels)
                previous = best_windows.get(aoi_index)
                if previous is None or score > previous[1]:
                    best_windows[aoi_index] = (window, score)
    fallback_windows = [best_windows[index][0] for index in sorted(best_windows)]
    skipped_empty = len(candidate_indexes) - len(fallback_windows)
    stats = {
        "aoi_candidate_count": len(candidate_indexes),
        "aoi_letterbox_count": len(fallback_windows),
        "skipped_aoi_letterbox_empty": skipped_empty,
        "aoi_letterbox_policy": "one_patch_per_aoi_polygon",
        "small_aoi_fallback_count": len(fallback_windows),
        "skipped_small_aoi_empty": skipped_empty,
        "small_aoi_transform_policy": "letterbox_resize_pad",
    }
    return fallback_windows, stats


def _aoi_geometry_has_fixed_window(geometry: Any, windows: Sequence[TsftSliceWindow], *, source_transform: Any) -> bool:
    from rasterio.windows import Window

    for window in windows:
        raster_window = Window(
            col_off=window.col_off,
            row_off=window.row_off,
            width=window.width,
            height=window.height,
        )
        if _window_is_strictly_inside_aoi(geometry, window=raster_window, source_transform=source_transform):
            return True
    return False


def _write_tsft_slice_windows(
    *,
    windows: Sequence[TsftSliceWindow],
    vector_labels: Any | None,
    vector_label_path: Path | None,
    output_split_dir: Path,
    min_polygon_overlap_ratio: float,
    min_positive_pixel_ratio: float,
    resume_existing: bool,
    progress_callback: ProgressCallback | None,
    progress_counters: dict[str, int],
    progress_total: int,
    progress_message_prefix: str,
    output_split_name: str = "train",
) -> list[StandardizedSample]:
    import numpy as np

    rasterio = require_rasterio()
    from rasterio.windows import Window

    images_dir = output_split_dir / "images"
    labels_dir = output_split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    samples: list[StandardizedSample] = []
    windows_by_raster: dict[Path, list[TsftSliceWindow]] = {}
    for window in windows:
        windows_by_raster.setdefault(window.source_raster, []).append(window)

    for raster_file, raster_windows in windows_by_raster.items():
        with rasterio.open(raster_file) as src:
            geometries = (
                _tsft_geometries_for_raster(vector_labels, vector_label_path, src.crs)
                if vector_labels is not None and vector_label_path is not None
                else []
            )
            base_profile = src.profile.copy()
            for slice_window in raster_windows:
                output_image = images_dir / f"{slice_window.sample_id}.tif"
                output_label = labels_dir / f"{slice_window.sample_id}.tif"
                if resume_existing and _is_valid_tsft_slice_pair(output_image, output_label):
                    image_shape, image_dtype = _validate_standardized_raster(output_image)
                    samples.append(
                        StandardizedSample(
                            sample_id=slice_window.sample_id,
                            split=output_split_name,
                            image_path=output_image,
                            label_path=output_label,
                            shape=image_shape,
                            dtype=image_dtype,
                            normalized=True,
                            linear_stretch_applied=False,
                            source_raster=slice_window.source_raster,
                            window=slice_window.window,
                            has_positive_label=slice_window.has_positive_label,
                            label_positive_pixels=slice_window.label_positive_pixels,
                        )
                    )
                    _emit_tsft_slice_progress(
                        progress_callback,
                        progress_counters,
                        total=progress_total,
                        sample_id=slice_window.sample_id,
                        action="skipped",
                        reason="existing standardized TSFT image/label pair passed integrity validation",
                        progress_message_prefix=progress_message_prefix,
                        split=output_split_name,
                    )
                    continue

                window = Window(
                    col_off=slice_window.col_off,
                    row_off=slice_window.row_off,
                    width=slice_window.width,
                    height=slice_window.height,
                )
                profile = dict(base_profile)
                if slice_window.window_mode == "aoi_letterbox":
                    array = src.read(window=window)
                    hwc = np.moveaxis(array, 0, -1)
                    crop_mask = _rasterize_geometry_mask(
                        geometry=slice_window.aoi_geometry,
                        transform=src.window_transform(window),
                        height=slice_window.height,
                        width=slice_window.width,
                    )
                    masked = np.where(crop_mask[:, :, None], hwc, 0)
                    target_size = int(slice_window.target_size or max(slice_window.height, slice_window.width))
                    letterboxed = _letterbox_resize_pad_hwc(masked, target_size=target_size)
                    stored_image, changed = _compact_image_for_storage(letterboxed, abnormal_threshold=-1.0)
                    profile.update(
                        height=target_size,
                        width=target_size,
                        transform=src.window_transform(window),
                    )
                    label_mask = np.zeros((target_size, target_size), dtype="uint8")
                    positive_pixels = 0
                    action_reason = "standardized TSFT small AOI letterbox negative chip and all-zero label written"
                else:
                    array = src.read(window=window)
                    hwc = np.moveaxis(array, 0, -1)
                    stored_image, changed = _compact_image_for_storage(hwc, abnormal_threshold=-1.0)
                    profile.update(height=slice_window.height, width=slice_window.width, transform=src.window_transform(window))
                    if slice_window.has_positive_label:
                        label_mask = _rasterize_tsft_window_label(
                            geometries=geometries,
                            window=window,
                            transform=src.window_transform(window),
                            source_transform=src.transform,
                            height=slice_window.height,
                            width=slice_window.width,
                            min_polygon_overlap_ratio=min_polygon_overlap_ratio,
                        )
                        positive_pixels = int(np.count_nonzero(label_mask > 0))
                        if positive_pixels <= 0:
                            label_mask = np.zeros((slice_window.height, slice_window.width), dtype="uint8")
                            positive_pixels = 0
                    else:
                        label_mask = np.zeros((slice_window.height, slice_window.width), dtype="uint8")
                        positive_pixels = 0
                    action_reason = "standardized TSFT TIFF window and equal-sized label written"
                _write_image_hwc_atomic(output_image, stored_image, profile)
                _write_label_mask_tif(output_label, label_mask, profile)
                samples.append(
                    StandardizedSample(
                        sample_id=slice_window.sample_id,
                        split=output_split_name,
                        image_path=output_image,
                        label_path=output_label,
                        shape=tuple(int(value) for value in stored_image.shape),
                        dtype=str(stored_image.dtype),
                        normalized=True,
                        linear_stretch_applied=changed,
                        source_raster=slice_window.source_raster,
                        window=slice_window.window,
                        has_positive_label=positive_pixels > 0,
                        label_positive_pixels=positive_pixels,
                    )
                )
                _emit_tsft_slice_progress(
                    progress_callback,
                    progress_counters,
                    total=progress_total,
                    sample_id=slice_window.sample_id,
                    action="written_positive" if positive_pixels > 0 else "written_negative",
                    reason=action_reason,
                    progress_message_prefix=progress_message_prefix,
                    split=output_split_name,
                )
    return samples


def _covering_window_from_bounds(bounds: Any, *, transform: Any) -> Any:
    from rasterio.windows import Window, from_bounds

    raw = from_bounds(*bounds, transform=transform)
    row_off = int(math.floor(raw.row_off))
    col_off = int(math.floor(raw.col_off))
    row_stop = int(math.ceil(raw.row_off + raw.height))
    col_stop = int(math.ceil(raw.col_off + raw.width))
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(1, col_stop - col_off),
        height=max(1, row_stop - row_off),
    )


def _rasterize_geometry_mask(*, geometry: Any, transform: Any, height: int, width: int, all_touched: bool = False) -> Any:
    import numpy as np
    from rasterio import features as raster_features
    from shapely.geometry import mapping

    if geometry is None or geometry.is_empty:
        return np.zeros((height, width), dtype=bool)
    return raster_features.rasterize(
        [(mapping(geometry), 1)],
        out_shape=(int(height), int(width)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=bool(all_touched),
    ).astype(bool)


def _letterbox_resize_pad_hwc(array: Any, *, target_size: int) -> Any:
    import numpy as np

    values = np.asarray(array)
    if values.ndim != 3:
        raise DatasetStandardizationError(f"Letterbox input must be H,W,C, got {values.shape}.")
    source_h, source_w, channels = (int(values.shape[0]), int(values.shape[1]), int(values.shape[2]))
    target_size = int(target_size)
    if target_size <= 0:
        raise DatasetStandardizationError("Letterbox target_size must be positive.")
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_size, target_size, channels), dtype=values.dtype)
    scale = min(float(target_size) / float(source_h), float(target_size) / float(source_w))
    new_h = max(1, min(target_size, int(round(source_h * scale))))
    new_w = max(1, min(target_size, int(round(source_w * scale))))
    resized = _resize_hwc_nearest(values, new_height=new_h, new_width=new_w)
    output = np.zeros((target_size, target_size, channels), dtype=values.dtype)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    output[top:top + new_h, left:left + new_w, :] = resized
    return output


def _resize_hwc_nearest(array: Any, *, new_height: int, new_width: int) -> Any:
    import numpy as np

    values = np.asarray(array)
    source_h, source_w = int(values.shape[0]), int(values.shape[1])
    row_index = np.floor((np.arange(new_height, dtype=np.float64) + 0.5) * source_h / new_height).astype(int)
    col_index = np.floor((np.arange(new_width, dtype=np.float64) + 0.5) * source_w / new_width).astype(int)
    row_index = np.clip(row_index, 0, source_h - 1)
    col_index = np.clip(col_index, 0, source_w - 1)
    return values[row_index[:, None], col_index[None, :], :]


def _mean_tsft_positive_pixel_ratio(windows: Sequence[TsftSliceWindow]) -> float | None:
    ratios = [
        float(window.label_positive_pixels) / float(max(window.valid_pixels, 1))
        for window in windows
        if window.has_positive_label and window.label_positive_pixels > 0
    ]
    if not ratios:
        return None
    return float(sum(ratios) / len(ratios))


def _emit_tsft_slice_progress(
    progress_callback: ProgressCallback | None,
    counters: dict[str, int],
    *,
    total: int,
    sample_id: str,
    action: str,
    reason: str,
    progress_message_prefix: str,
    split: str = "train",
) -> None:
    counters["processed"] = int(counters.get("processed", 0)) + 1
    if action == "skipped":
        counters["skipped"] = int(counters.get("skipped", 0)) + 1
    elif action == "rewritten":
        counters["rewritten"] = int(counters.get("rewritten", 0)) + 1
    else:
        counters["written"] = int(counters.get("written", 0)) + 1
    _emit_progress(
        progress_callback,
        {
            "status": "running",
            "processed": counters["processed"],
            "total": total,
            "written": int(counters.get("written", 0)),
            "skipped": int(counters.get("skipped", 0)),
            "rewritten": int(counters.get("rewritten", 0)),
            "percent": counters["processed"] / total if total else 1.0,
            "current_split": split,
            "current_sample": sample_id,
            "current_action": action,
            "current_reason": reason,
            "message": f"{progress_message_prefix}/{sample_id}".strip("/"),
        },
    )


def _write_split(
    *,
    files: Sequence[Path],
    split_dir: Path,
    split: str,
    labels: Mapping[str, Path],
    RGB_bands: str,
    label_polarity: str = LABEL_POLARITY_FOREGROUND_WHITE,
    direct_images_dir: bool = False,
    resume_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_counters: dict[str, int] | None = None,
    progress_total: int | None = None,
    progress_message_prefix: str = "",
) -> list[StandardizedSample]:
    images_dir = split_dir if direct_images_dir else split_dir / "images"
    labels_dir = None if direct_images_dir else split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
    samples: list[StandardizedSample] = []
    for index, image_path in enumerate(files):
        sample_id = image_path.stem
        output_image = images_dir / f"{sample_id}.tif"
        output_label = None
        matched_label = labels.get(image_path.stem)
        rewrite_existing = False
        if resume_existing and matched_label is None and labels_dir is None and output_image.exists():
            try:
                existing_shape, existing_dtype = _validate_standardized_raster(output_image)
            except Exception:
                rewrite_existing = True
            else:
                samples.append(
                    StandardizedSample(
                        sample_id=sample_id or f"sample_{index:06d}",
                        split=split,
                        image_path=output_image,
                        label_path=None,
                        shape=existing_shape,
                        dtype=existing_dtype,
                        normalized=True,
                        linear_stretch_applied=False,
                    )
                )
                if progress_callback is not None:
                    counters = progress_counters if progress_counters is not None else {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
                    counters["processed"] = int(counters.get("processed", 0)) + 1
                    counters["skipped"] = int(counters.get("skipped", 0)) + 1
                    total = int(progress_total if progress_total is not None else len(files))
                    _emit_progress(
                        progress_callback,
                        {
                            "status": "running",
                            "processed": counters["processed"],
                            "total": total,
                            "written": int(counters.get("written", 0)),
                            "skipped": counters["skipped"],
                            "rewritten": int(counters.get("rewritten", 0)),
                            "percent": counters["processed"] / total if total else 1.0,
                            "current_split": split,
                            "current_sample": sample_id,
                            "current_action": "skipped",
                            "current_reason": "existing standardized image passed integrity validation",
                            "message": f"{progress_message_prefix}/{split}/{sample_id}".strip("/"),
                        },
                    )
                continue

        array, profile = _read_image_hwc(image_path)
        stored_image, changed = _compact_image_for_storage(array)
        _write_image_hwc_atomic(output_image, stored_image, profile)

        if matched_label is not None and labels_dir is not None:
            if matched_label.suffix.lower() in RASTER_LABEL_SUFFIXES:
                label_array, label_profile = _read_image_hwc(matched_label)
                standardized_label = _standardize_label_hwc(
                    label_array,
                    label_polarity=label_polarity,
                )
                output_label = labels_dir / f"{sample_id}.tif"
                _write_label_mask_tif(output_label, standardized_label[:, :, 0], label_profile)
            else:
                output_label = labels_dir / f"{sample_id}{matched_label.suffix.lower()}"
                _copy_label(matched_label, output_label)

        samples.append(
            StandardizedSample(
                sample_id=sample_id or f"sample_{index:06d}",
                split=split,
                image_path=output_image,
                label_path=output_label,
                shape=tuple(int(v) for v in stored_image.shape),
                dtype=str(stored_image.dtype),
                normalized=True,
                linear_stretch_applied=changed,
            )
        )
        if progress_callback is not None:
            counters = progress_counters if progress_counters is not None else {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
            counters["processed"] = int(counters.get("processed", 0)) + 1
            if rewrite_existing:
                counters["rewritten"] = int(counters.get("rewritten", 0)) + 1
                current_action = "rewritten"
                current_reason = "existing standardized image was invalid and was rebuilt"
            else:
                counters["written"] = int(counters.get("written", 0)) + 1
                current_action = "written"
                current_reason = "standardized image copy written"
            total = int(progress_total if progress_total is not None else len(files))
            _emit_progress(
                progress_callback,
                {
                    "status": "running",
                    "processed": counters["processed"],
                    "total": total,
                    "written": int(counters.get("written", 0)),
                    "skipped": int(counters.get("skipped", 0)),
                    "rewritten": int(counters.get("rewritten", 0)),
                    "percent": counters["processed"] / total if total else 1.0,
                    "current_split": split,
                    "current_sample": sample_id,
                    "current_action": current_action,
                    "current_reason": current_reason,
                    "message": f"{progress_message_prefix}/{split}/{sample_id}".strip("/"),
                },
            )
    return samples


def _slice_rasters_to_daft_images(
    *,
    raster_path: Path,
    output_dir: Path,
    RGB_bands: str,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None = DEFAULT_DAFT_MAX_NODATA_RATIO,
    slicing_aoi_path: Path | None = None,
    resume_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_message_prefix: str = "",
) -> list[DaftSliceResult]:
    tile_size = _positive_int(tile_size, "tile_size")
    stride = _positive_int(stride, "stride")
    max_nodata_ratio = _normalize_ratio(max_nodata_ratio, "max_nodata_ratio")
    slicing_aoi = _read_slicing_aoi(slicing_aoi_path) if slicing_aoi_path is not None else None
    raster_files = _discover_tiff_rasters(
        raster_path,
        RGB_bands=RGB_bands,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DaftSliceResult] = []
    total_windows = _count_daft_windows(
        raster_files,
        tile_size=tile_size,
        stride=stride,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
    )
    counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    for raster_file in raster_files:
        results.extend(
            _slice_one_raster_to_daft_images(
                raster_file=raster_file,
                output_dir=output_dir,
                RGB_bands=RGB_bands,
                tile_size=tile_size,
                stride=stride,
                max_nodata_ratio=max_nodata_ratio,
                slicing_aoi=slicing_aoi,
                slicing_aoi_path=slicing_aoi_path,
                resume_existing=resume_existing,
                progress_callback=progress_callback,
                progress_counters=counters,
                progress_total=total_windows,
                progress_message_prefix=progress_message_prefix,
            )
        )
    if not results:
        raise DatasetStandardizationError(f"Tiff Slicing produced no DAFT image chips under {raster_path}.")
    return results


def _slice_one_raster_to_daft_images(
    *,
    raster_file: Path,
    output_dir: Path,
    RGB_bands: str,
    tile_size: int,
    stride: int,
    max_nodata_ratio: float,
    slicing_aoi: Any | None = None,
    slicing_aoi_path: Path | None = None,
    resume_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_counters: dict[str, int] | None = None,
    progress_total: int | None = None,
    progress_message_prefix: str = "",
) -> list[DaftSliceResult]:
    import numpy as np

    rasterio = require_rasterio()
    from rasterio.windows import Window

    rgb_indexes = rgb_bands_to_indexes(RGB_bands)
    abnormal_threshold = -1.0
    zero_tolerance = 1e-6
    results: list[DaftSliceResult] = []
    with rasterio.open(raster_file) as src:
        missing = [index for index in rgb_indexes if index > int(src.count)]
        if missing:
            raise DatasetStandardizationError(
                f"RGB_bands {RGB_bands} requires band(s) {missing}, but {raster_file} only has {src.count} band(s)."
            )
        aoi_filter = _slicing_aoi_filter_for_raster(slicing_aoi, slicing_aoi_path, src.crs)
        rows, cols = _candidate_window_starts_for_aoi(
            src,
            tile_size=tile_size,
            stride=stride,
            aoi_geometry=aoi_filter,
        )
        base_profile = src.profile.copy()
        total = int(progress_total if progress_total is not None else len(rows) * len(cols))
        for row_off in rows:
            for col_off in cols:
                height = min(tile_size, src.height - row_off)
                width = min(tile_size, src.width - col_off)
                if height <= 0 or width <= 0:
                    continue
                window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
                sample_id = f"{_safe_name(raster_file.stem)}_r{int(row_off):06d}_c{int(col_off):06d}"
                if aoi_filter is not None and not _window_is_strictly_inside_aoi(
                    aoi_filter,
                    window=window,
                    source_transform=src.transform,
                ):
                    _emit_daft_slice_progress(
                        progress_callback,
                        progress_counters,
                        total=total,
                        sample_id=sample_id,
                        action="skipped",
                        reason="TIFF window is outside AOI or touches the AOI boundary",
                        progress_message_prefix=progress_message_prefix,
                    )
                    continue
                output_image = output_dir / f"{sample_id}.tif"
                rewrite_existing = False
                if resume_existing and output_image.exists():
                    try:
                        existing_shape, existing_dtype = _validate_standardized_raster(output_image)
                    except Exception:
                        rewrite_existing = True
                    else:
                        results.append(
                            DaftSliceResult(
                                sample_id=sample_id,
                                source_raster=raster_file,
                                image_path=output_image,
                                window=(int(row_off), int(col_off), int(height), int(width)),
                                shape=existing_shape,
                                dtype=existing_dtype,
                                normalized=True,
                                linear_stretch_applied=False,
                            )
                        )
                        _emit_daft_slice_progress(
                            progress_callback,
                            progress_counters,
                            total=total,
                            sample_id=sample_id,
                            action="skipped",
                            reason="existing standardized TIFF window passed integrity validation",
                            progress_message_prefix=progress_message_prefix,
                        )
                        continue

                masks = [src.read_masks(b, window=window) > 0 for b in rgb_indexes]
                valid_mask = np.any(np.stack(masks, axis=0), axis=0)
                if not bool(np.any(valid_mask)):
                    _emit_daft_slice_progress(
                        progress_callback,
                        progress_counters,
                        total=total,
                        sample_id=sample_id,
                        action="skipped",
                        reason="TIFF window has no valid raster mask pixels",
                        progress_message_prefix=progress_message_prefix,
                    )
                    continue

                array = src.read(window=window)
                check_array = array.astype("float32", copy=False)
                check_array[check_array < abnormal_threshold] = 0
                rgb_array = extract_rgb_bands(check_array, RGB_bands, layout="chw")
                any_nonzero_valid = np.any(np.abs(rgb_array) > zero_tolerance, axis=0) & valid_mask
                if not bool(np.any(any_nonzero_valid)):
                    _emit_daft_slice_progress(
                        progress_callback,
                        progress_counters,
                        total=total,
                        sample_id=sample_id,
                        action="skipped",
                            reason=f"TIFF window valid pixels are zero in {RGB_bands}",
                        progress_message_prefix=progress_message_prefix,
                    )
                    continue
                nodata_ratio = 1.0 - (float(np.count_nonzero(any_nonzero_valid)) / float(height * width))
                if nodata_ratio > max_nodata_ratio:
                    _emit_daft_slice_progress(
                        progress_callback,
                        progress_counters,
                        total=total,
                        sample_id=sample_id,
                        action="skipped",
                        reason=f"TIFF window nodata ratio {nodata_ratio:.3f} exceeds {max_nodata_ratio:.3f}",
                        progress_message_prefix=progress_message_prefix,
                    )
                    continue

                hwc = np.moveaxis(array, 0, -1)
                stored_image, changed = _compact_image_for_storage(hwc, abnormal_threshold=abnormal_threshold)
                profile = dict(base_profile)
                profile.update(
                    height=int(height),
                    width=int(width),
                    transform=src.window_transform(window),
                )
                _write_image_hwc_atomic(output_image, stored_image, profile)
                results.append(
                    DaftSliceResult(
                        sample_id=sample_id,
                        source_raster=raster_file,
                        image_path=output_image,
                        window=(int(row_off), int(col_off), int(height), int(width)),
                        shape=tuple(int(value) for value in stored_image.shape),
                        dtype=str(stored_image.dtype),
                        normalized=True,
                        linear_stretch_applied=changed,
                    )
                )
                _emit_daft_slice_progress(
                    progress_callback,
                    progress_counters,
                    total=total,
                    sample_id=sample_id,
                    action="rewritten" if rewrite_existing else "written",
                    reason=(
                        "existing standardized TIFF window was invalid and was rebuilt"
                        if rewrite_existing
                        else "standardized TIFF window written"
                    ),
                    progress_message_prefix=progress_message_prefix,
                )
    return results


def _emit_daft_slice_progress(
    progress_callback: ProgressCallback | None,
    counters: dict[str, int] | None,
    *,
    total: int,
    sample_id: str,
    action: str,
    reason: str,
    progress_message_prefix: str,
) -> None:
    if counters is None:
        counters = {"processed": 0, "written": 0, "skipped": 0, "rewritten": 0}
    counters["processed"] = int(counters.get("processed", 0)) + 1
    if action == "written":
        counters["written"] = int(counters.get("written", 0)) + 1
    elif action == "rewritten":
        counters["rewritten"] = int(counters.get("rewritten", 0)) + 1
    elif action == "skipped":
        counters["skipped"] = int(counters.get("skipped", 0)) + 1
    _emit_progress(
        progress_callback,
        {
            "status": "running",
            "processed": counters["processed"],
            "total": total,
            "written": int(counters.get("written", 0)),
            "skipped": int(counters.get("skipped", 0)),
            "rewritten": int(counters.get("rewritten", 0)),
            "percent": counters["processed"] / total if total else 1.0,
            "current_sample": sample_id,
            "current_action": action,
            "current_reason": reason,
            "message": f"{progress_message_prefix}/{sample_id}".strip("/"),
        },
    )

def _standardize_taft_tree_split(
    subset: TaftTreeSubset,
    subset_target: Path,
    *,
    split: str,
    counters: dict[str, int],
    total_samples: int,
    state_path: Path,
    events_path: Path,
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    if split == "train":
        image_dir = subset.train_image_dir
        label_dir = subset.train_label_dir
    elif split == "test":
        image_dir = subset.test_image_dir
        label_dir = subset.test_label_dir
    else:
        raise DatasetStandardizationError(f"Unsupported TAFT tree split: {split}")

    image_files = _discover_images_in_dir(image_dir)
    label_files = _discover_raster_labels_in_dir(label_dir)
    labels_by_stem = {path.stem: path for path in label_files}
    output_image_dir = subset_target / split / "images"
    output_label_dir = subset_target / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_files):
        sample_id = image_path.stem or f"sample_{index:06d}"
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            raise DatasetStandardizationError(
                f"Missing label for {split} sample {subset.source_rel}/{image_path.name}."
            )

        output_image = output_image_dir / f"{sample_id}.tif"
        output_label = output_label_dir / f"{sample_id}.tif"
        existing_ok = _is_valid_standardized_pair(output_image, output_label)
        resume_action = "skipped" if existing_ok else "written"
        action_reason = (
            "standardized copy already exists and passed integrity validation"
            if existing_ok
            else "creating compact standardized image/label copy"
        )
        changed = False
        normalized_shape: tuple[int, int, int] | None = None
        label_shape: tuple[int, int, int] | None = None
        if existing_ok:
            counters["skipped"] += 1
            normalized_shape, label_shape = _read_standardized_pair_shapes(output_image, output_label)
        else:
            if output_image.exists() or output_label.exists():
                counters["rewritten"] += 1
                resume_action = "rewritten"
                action_reason = "existing standardized copy was incomplete or invalid, so it was rebuilt"
                _remove_partial_standardized_pair(output_image, output_label)
            image_array, image_profile = _read_image_hwc(image_path)
            stored_image, changed = _compact_image_for_storage(image_array)
            _write_image_hwc_atomic(output_image, stored_image, image_profile)

            label_array, label_profile = _read_image_hwc(label_path)
            standardized_label = _standardize_label_hwc(label_array, label_polarity=subset.label_polarity)
            _write_label_mask_tif(output_label, standardized_label[:, :, 0], label_profile)
            normalized_shape = tuple(int(value) for value in stored_image.shape)
            label_shape = tuple(int(value) for value in standardized_label.shape)
            counters["written"] += 1

        counters["processed"] += 1
        event = {
            "event": "taft_tree_sample_import",
            "created_at": _utc_now(),
            "action": resume_action,
            "reason": action_reason,
            "processed": counters["processed"],
            "total": total_samples,
            "subset": subset.source_rel,
            "subset_id": subset.subset_id,
            "split": split,
            "sample_id": sample_id,
            "source_image_path": str(image_path),
            "source_label_path": str(label_path),
            "output_image_path": str(output_image),
            "output_label_path": str(output_label),
            "shape": list(normalized_shape or ()),
            "label_shape": list(label_shape or ()),
            "RGB_bands": subset.RGB_bands,
            "label_polarity": subset.label_polarity,
            "auto_RGB_bands": subset.auto_RGB_bands,
            "auto_label_polarity": subset.auto_label_polarity,
            "annotation_source": subset.annotation_source,
            "user_confirmed": subset.user_confirmed,
            "linear_stretch_applied": changed,
        }
        _append_jsonl(events_path, event)
        progress = {
            "status": "running",
            "processed": counters["processed"],
            "total": total_samples,
            "written": counters["written"],
            "skipped": counters["skipped"],
            "rewritten": counters["rewritten"],
            "percent": counters["processed"] / total_samples if total_samples else 1.0,
            "current_subset": subset.source_rel,
            "current_split": split,
            "current_sample": sample_id,
            "current_action": resume_action,
            "current_reason": action_reason,
            "message": f"{subset.source_rel}/{split}/{sample_id}",
        }
        _update_taft_tree_state(
            state_path,
            counters=counters,
            total_samples=total_samples,
            status="running",
            current_subset=subset.source_rel,
            current_split=split,
            current_sample=sample_id,
        )
        _emit_progress(progress_callback, progress)

        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source_image_path": str(image_path),
                "source_label_path": str(label_path),
                "image_path": str(output_image),
                "label_path": str(output_label),
                "shape": list(normalized_shape or ()),
                "label_shape": list(label_shape or ()),
                "dtype": str(stored_image.dtype) if not existing_ok else "source_dtype",
                "label_dtype": "uint8",
                "normalized": True,
                "linear_stretch_applied": changed,
                "resume_action": resume_action,
                "resume_reason": action_reason,
                "RGB_bands": subset.RGB_bands,
                "label_polarity": subset.label_polarity,
                "auto_RGB_bands": subset.auto_RGB_bands,
                "auto_label_polarity": subset.auto_label_polarity,
                "annotation_source": subset.annotation_source,
                "user_confirmed": subset.user_confirmed,
            }
        )
    return rows


def _standardize_label_hwc(array: Any, *, label_polarity: str) -> Any:
    import numpy as np

    values = np.asarray(array)
    if values.ndim != 3:
        raise DatasetStandardizationError(f"Labels must be H,W,C arrays before standardization, got {values.shape}.")
    first_band = values[:, :, 0].astype(np.float32, copy=False)
    finite = first_band[np.isfinite(first_band)]
    if finite.size == 0:
        binary = np.zeros(first_band.shape, dtype="uint8")
        return binary[:, :, None]

    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        if label_polarity == LABEL_POLARITY_FOREGROUND_WHITE:
            foreground = first_band > 0
        elif label_polarity == LABEL_POLARITY_FOREGROUND_BLACK:
            foreground = first_band <= 0
        else:
            raise DatasetStandardizationError(f"Unsupported label_polarity: {label_polarity}")
        return foreground.astype("uint8")[:, :, None]
    if low >= 0.0 and high <= 1.0:
        scaled = first_band
    else:
        scaled = (first_band - low) / (high - low)

    if label_polarity == LABEL_POLARITY_FOREGROUND_WHITE:
        foreground = scaled > 0.5
    elif label_polarity == LABEL_POLARITY_FOREGROUND_BLACK:
        foreground = scaled <= 0.5
    else:
        raise DatasetStandardizationError(f"Unsupported label_polarity: {label_polarity}")
    return foreground.astype("uint8")[:, :, None]


def _ensure_taft_tree_split_dirs(subset_target: Path) -> None:
    for split in ("train", "test"):
        (subset_target / split / "images").mkdir(parents=True, exist_ok=True)
        (subset_target / split / "labels").mkdir(parents=True, exist_ok=True)


def _materialize_auxiliary_structure(subset: TaftTreeSubset, subset_target: Path) -> list[str]:
    target_dirs: list[str] = []
    for auxiliary_rel in subset.auxiliary_dirs:
        target = subset_target / "auxiliary" / auxiliary_rel
        target.mkdir(parents=True, exist_ok=True)
        target_dirs.append(str(target))
    return target_dirs


def _initial_taft_tree_state(
    *,
    dataset_id: str,
    name: str,
    version: str,
    source_root: Path,
    preset: str,
    split_policy: str,
    standardize: bool,
    annotations_digest: str,
    sample_total: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "name": name,
        "version": version,
        "source_root": str(source_root),
        "preset": preset,
        "split_policy": split_policy,
        "standardize": bool(standardize),
        "annotations_digest": annotations_digest,
        "sample_total": int(sample_total),
        "status": "running",
        "processed": 0,
        "written": 0,
        "skipped": 0,
        "rewritten": 0,
        "percent": 0.0,
        "current_subset": "",
        "current_split": "",
        "current_sample": "",
        "updated_at": _utc_now(),
    }


def _can_resume_taft_tree_import(
    *,
    state: Any,
    dataset_id: str,
    name: str,
    version: str,
    source_root: Path,
    preset: str,
    split_policy: str,
    standardize: bool,
    annotations_digest: str,
) -> bool:
    if not isinstance(state, Mapping):
        return False
    return (
        state.get("version") == version
        and state.get("source_root") == str(source_root)
        and state.get("preset") == preset
        and state.get("split_policy") == split_policy
        and bool(state.get("standardize")) == bool(standardize)
        and state.get("annotations_digest") == annotations_digest
    )


def _update_taft_tree_state(
    state_path: Path,
    *,
    counters: Mapping[str, int],
    total_samples: int,
    status: str,
    current_subset: str,
    current_split: str = "",
    current_sample: str = "",
) -> None:
    state = _read_json(state_path)
    if not isinstance(state, Mapping):
        state = {}
    processed = int(counters.get("processed", 0))
    updated = {
        **dict(state),
        "status": status,
        "processed": processed,
        "written": int(counters.get("written", 0)),
        "skipped": int(counters.get("skipped", 0)),
        "rewritten": int(counters.get("rewritten", 0)),
        "sample_total": int(total_samples),
        "percent": processed / total_samples if total_samples else 1.0,
        "current_subset": current_subset,
        "current_split": current_split,
        "current_sample": current_sample,
        "updated_at": _utc_now(),
    }
    _write_json(state_path, updated)


def _emit_progress(progress_callback: ProgressCallback | None, payload: Mapping[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(dict(payload))


def _read_image_hwc(path: Path) -> tuple[Any, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        rasterio = require_rasterio()
        with rasterio.open(path) as src:
            array = src.read()
            profile = src.profile.copy()
        import numpy as np

        return np.moveaxis(array, 0, -1), profile
    raise DatasetStandardizationError(f"Unsupported image suffix: {path.suffix}")


def _write_image_hwc(path: Path, array: Any, profile: Mapping[str, Any]) -> None:
    import numpy as np

    rasterio = require_rasterio()
    values = np.asarray(array)
    if values.ndim != 3:
        raise DatasetStandardizationError(f"Standardized samples must be H,W,C, got {values.shape}.")
    _ensure_output_space(path.parent, required_bytes=_estimate_geotiff_write_bytes(values))
    write_profile = dict(profile)
    write_profile.pop("blockxsize", None)
    write_profile.pop("blockysize", None)
    write_profile.pop("tiled", None)
    write_profile.pop("interleave", None)
    write_profile.pop("compress", None)
    write_profile.pop("predictor", None)
    write_profile.pop("BIGTIFF", None)
    write_profile.pop("bigtiff", None)
    write_profile.update(
        driver="GTiff",
        height=int(values.shape[0]),
        width=int(values.shape[1]),
        count=int(values.shape[2]),
        dtype=str(values.dtype),
        compress="LZW",
        predictor=3 if np.issubdtype(values.dtype, np.floating) else 2,
        BIGTIFF="IF_SAFER",
    )
    write_profile.pop("photometric", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path, "w", **write_profile) as dst:
            for band_idx in range(values.shape[2]):
                dst.write(values[:, :, band_idx], band_idx + 1)
    except Exception as exc:
        if _looks_like_no_space_error(exc):
            free_text = _format_bytes(shutil.disk_usage(path.parent).free)
            raise DatasetStandardizationError(
                f"Not enough free disk space while writing standardized GeoTIFF: {path}. "
                f"Free space on this filesystem is {free_text}. "
                "Choose a registry root on a larger disk or free space, then rerun with resume enabled."
            ) from exc
        raise


def _write_image_hwc_atomic(path: Path, array: Any, profile: Mapping[str, Any]) -> None:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    temp_path = path.with_name(f".{path.stem}.{token}.tmp.tif")
    if temp_path.exists():
        temp_path.unlink()
    try:
        _write_image_hwc(temp_path, array, profile)
        _validate_standardized_raster(temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _estimate_geotiff_write_bytes(values: Any) -> int:
    itemsize = int(getattr(getattr(values, "dtype", None), "itemsize", 4) or 4)
    size = int(getattr(values, "size", 0) or 0) * itemsize
    return max(size * 2, 64 * 1024 * 1024)


def _ensure_output_space(path: Path, *, required_bytes: int = 0, min_free_bytes: int = 0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    threshold = max(int(required_bytes), int(min_free_bytes))
    if free_bytes < threshold:
        raise DatasetStandardizationError(
            f"Not enough free disk space for dataset standardization at {path}. "
            f"Free={_format_bytes(free_bytes)}, required at least {_format_bytes(threshold)}. "
            "Choose a registry root on a larger disk, or free space and rerun with resume enabled."
        )


def _looks_like_no_space_error(exc: BaseException) -> bool:
    text = f"{exc}"
    cursor = exc.__cause__
    while cursor is not None:
        text += f" {cursor}"
        cursor = cursor.__cause__
    lowered = text.lower()
    return "no space left on device" in lowered or "write error" in lowered


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _validate_standardized_raster(path: Path) -> tuple[tuple[int, int, int], str]:
    rasterio = require_rasterio()
    if not path.exists() or path.stat().st_size <= 0:
        raise DatasetStandardizationError(f"Standardized raster is missing or empty: {path}")
    with rasterio.open(path) as src:
        if src.count <= 0 or src.width <= 0 or src.height <= 0:
            raise DatasetStandardizationError(f"Invalid standardized raster dimensions: {path}")
        dtype = str(src.dtypes[0])
        for band_dtype in src.dtypes:
            if str(band_dtype) != dtype:
                raise DatasetStandardizationError(f"All standardized raster bands must share one dtype: {path}")
        return (int(src.height), int(src.width), int(src.count)), dtype


def _is_valid_standardized_pair(image_path: Path, label_path: Path) -> bool:
    try:
        image_shape, image_dtype = _validate_standardized_raster(image_path)
        label_shape, label_dtype = _validate_label_mask_raster(label_path)
    except Exception:
        return False
    return (
        image_shape[0] == label_shape[0]
        and image_shape[1] == label_shape[1]
    )


def _read_standardized_pair_shapes(image_path: Path, label_path: Path) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    image_shape, _ = _validate_standardized_raster(image_path)
    label_shape, _ = _validate_label_mask_raster(label_path)
    return image_shape, label_shape


def _remove_partial_standardized_pair(image_path: Path, label_path: Path) -> None:
    for path in (image_path, label_path):
        if path.exists():
            path.unlink()


def _compact_image_for_storage(array: Any, *, abnormal_threshold: float | None = None) -> tuple[Any, bool]:
    import numpy as np

    values = np.asarray(array)
    if values.ndim != 3:
        raise DatasetStandardizationError(f"Samples must be H,W,C arrays, got shape {values.shape}.")
    stored = values.copy()
    if abnormal_threshold is None:
        return stored, False
    abnormal = stored < abnormal_threshold
    if bool(np.any(abnormal)):
        stored[abnormal] = 0
        return stored, True
    return stored, False


def _float_image_to_uint8(array: Any) -> Any:
    import numpy as np

    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype="uint8")
    low = float(finite.min())
    high = float(finite.max())
    if low >= 0.0 and high <= 1.0:
        scaled = values
    elif high > low:
        scaled = (values - low) / (high - low)
    else:
        scaled = np.zeros(values.shape, dtype=np.float32)
    return np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype("uint8")


def _ensure_linear_stretched_float32(array: Any) -> tuple[Any, bool]:
    import numpy as np

    original = np.asarray(array)
    values = original.astype(np.float32, copy=False)
    if values.ndim != 3:
        raise DatasetStandardizationError(f"Samples must be H,W,C arrays, got shape {values.shape}.")
    finite = values[np.isfinite(values)]
    already_normalized = finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0
    if already_normalized:
        return values.astype(np.float32, copy=True), False

    stretched = np.zeros(values.shape, dtype=np.float32)
    for channel in range(values.shape[2]):
        band = values[:, :, channel]
        finite_band = band[np.isfinite(band)]
        if finite_band.size == 0:
            continue
        low = float(finite_band.min())
        high = float(finite_band.max())
        if high <= low:
            stretched[:, :, channel] = 0
            continue
        scaled = (band - low) / (high - low)
        stretched[:, :, channel] = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return stretched, True


def _read_tsft_vector_labels(path: Path) -> Any:
    return _read_polygon_vector(path, purpose="TSFT Tiff Slicing labels")


def _read_slicing_aoi(path: Path | None) -> Any | None:
    if path is None:
        return None
    return _read_polygon_vector(path, purpose="slicing AOI")


def _read_polygon_vector(path: Path, *, purpose: str) -> Any:
    if path.suffix.lower() not in {".shp", ".gpkg", ".geojson", ".json"}:
        raise DatasetStandardizationError(f"{purpose} must be .shp, .gpkg, or .geojson polygons.")
    try:
        import geopandas as gpd
    except Exception as exc:
        raise DatasetStandardizationError(f"geopandas is required for {purpose}.") from exc
    try:
        labels = gpd.read_file(path)
    except Exception as exc:
        raise DatasetStandardizationError(f"Failed to read {purpose}: {path}: {exc}") from exc
    if labels.empty:
        raise DatasetStandardizationError(f"{purpose} file contains no features: {path}")
    labels = labels[labels.geometry.notna() & ~labels.geometry.is_empty]
    if labels.empty:
        raise DatasetStandardizationError(f"{purpose} file contains no polygon geometries: {path}")
    return labels


def _tsft_geometries_for_raster(vector_labels: Any, vector_label_path: Path, raster_crs: Any) -> list[Any]:
    return _polygon_geometries_for_raster(
        vector_labels,
        vector_label_path,
        raster_crs,
        purpose="TSFT polygon label",
    )


def _polygon_geometries_for_raster(labels: Any, vector_path: Path, raster_crs: Any, *, purpose: str) -> list[Any]:
    if raster_crs is not None:
        if labels.crs is None:
            raise DatasetStandardizationError(
                f"{purpose} CRS is missing while raster has CRS {raster_crs}: {vector_path}"
            )
        raster_crs_text = raster_crs.to_string() if hasattr(raster_crs, "to_string") else str(raster_crs)
        try:
            if str(labels.crs) != raster_crs_text:
                labels = labels.to_crs(raster_crs_text)
        except Exception as exc:
            raise DatasetStandardizationError(
                f"Failed to reproject {purpose} to raster CRS {raster_crs}: {exc}"
            ) from exc
    geometries: list[Any] = []
    for geometry in labels.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if getattr(geometry, "geom_type", "") not in {"Polygon", "MultiPolygon"}:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry is not None and not geometry.is_empty:
            geometries.append(geometry)
    if not geometries:
        raise DatasetStandardizationError(f"{purpose} file has no valid polygon geometries: {vector_path}")
    return geometries


def _slicing_aoi_filter_for_raster(slicing_aoi: Any | None, slicing_aoi_path: Path | None, raster_crs: Any) -> Any | None:
    if slicing_aoi is None:
        return None
    geometries = _polygon_geometries_for_raster(
        slicing_aoi,
        slicing_aoi_path or Path("<slicing_aoi>"),
        raster_crs,
        purpose="slicing AOI",
    )
    try:
        from shapely.ops import unary_union
    except Exception as exc:
        raise DatasetStandardizationError("shapely is required for slicing AOI filtering.") from exc
    union = unary_union(geometries)
    if not getattr(union, "is_valid", True):
        union = union.buffer(0)
    if union is None or union.is_empty:
        raise DatasetStandardizationError(f"slicing AOI file has no valid polygon geometries: {slicing_aoi_path}")
    return union


def _window_is_strictly_inside_aoi(aoi_geometry: Any, *, window: Any, source_transform: Any) -> bool:
    from rasterio.windows import bounds as window_bounds
    from shapely.geometry import box

    left, bottom, right, top = window_bounds(window, source_transform)
    window_geometry = box(left, bottom, right, top)
    if window_geometry.is_empty:
        return False
    return bool(aoi_geometry.contains(window_geometry) and not aoi_geometry.boundary.intersects(window_geometry))


def _rasterize_tsft_window_label(
    *,
    geometries: Sequence[Any],
    window: Any,
    transform: Any,
    source_transform: Any,
    height: int,
    width: int,
    min_polygon_overlap_ratio: float,
) -> Any:
    import numpy as np
    from rasterio import features as raster_features
    from rasterio.windows import bounds as window_bounds
    from shapely.geometry import box, mapping

    left, bottom, right, top = window_bounds(window, source_transform)
    window_geometry = box(left, bottom, right, top)
    shapes: list[tuple[dict[str, Any], int]] = []
    for geometry in geometries:
        if not geometry.intersects(window_geometry):
            continue
        intersection = geometry.intersection(window_geometry)
        if intersection.is_empty:
            continue
        geometry_area = float(getattr(geometry, "area", 0.0) or 0.0)
        if geometry_area > 0.0:
            overlap_ratio = float(intersection.area) / geometry_area
            if overlap_ratio < min_polygon_overlap_ratio:
                continue
        shapes.append((mapping(intersection), 255))
    if not shapes:
        return np.zeros((height, width), dtype="uint8")
    return raster_features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def _write_label_mask_tif(path: Path, mask: Any, image_profile: Mapping[str, Any]) -> None:
    token = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    temp_path = path.with_name(f".{path.stem}.{token}.tmp.tif")
    if temp_path.exists():
        temp_path.unlink()
    try:
        _write_label_mask_tif_direct(temp_path, mask, image_profile)
        _validate_label_mask_raster(temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_label_mask_tif_direct(path: Path, mask: Any, image_profile: Mapping[str, Any]) -> None:
    import numpy as np

    rasterio = require_rasterio()
    values = np.asarray(mask, dtype="uint8")
    write_profile = dict(image_profile)
    write_profile.pop("blockxsize", None)
    write_profile.pop("blockysize", None)
    write_profile.pop("tiled", None)
    write_profile.pop("photometric", None)
    write_profile.pop("compress", None)
    write_profile.pop("predictor", None)
    write_profile.pop("BIGTIFF", None)
    write_profile.pop("bigtiff", None)
    write_profile.pop("nodata", None)
    write_profile.update(
        driver="GTiff",
        height=int(values.shape[0]),
        width=int(values.shape[1]),
        count=1,
        dtype="uint8",
        compress="LZW",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(values, 1)


def _validate_label_mask_raster(path: Path) -> tuple[tuple[int, int, int], str]:
    shape, dtype = _validate_standardized_raster(path)
    if shape[2] != 1:
        raise DatasetStandardizationError(f"Standardized labels must have one band: {path}")
    if dtype != "uint8":
        raise DatasetStandardizationError(f"Standardized labels must be uint8, got {dtype}: {path}")
    return shape, dtype


def _is_valid_tsft_slice_pair(image_path: Path, label_path: Path) -> bool:
    try:
        image_shape, image_dtype = _validate_standardized_raster(image_path)
    except Exception:
        return False
    try:
        rasterio = require_rasterio()
        with rasterio.open(label_path) as src:
            label_shape = (int(src.height), int(src.width), int(src.count))
            label_dtype = str(src.dtypes[0])
    except Exception:
        return False
    return (
        label_dtype in {"uint8", "float32"}
        and label_shape[2] == 1
        and image_shape[0] == label_shape[0]
        and image_shape[1] == label_shape[1]
    )


def _discover_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise DatasetStandardizationError(f"Unsupported image file: {path}")
        return [path]
    files = sorted(file for suffix in IMAGE_SUFFIXES for file in path.rglob(f"*{suffix}"))
    if not files:
        raise DatasetStandardizationError(f"No supported image files found under {path}")
    return files


def _discover_tiff_rasters(
    path: Path,
    *,
    RGB_bands: str | None = None,
    slicing_aoi: Any | None = None,
    slicing_aoi_path: Path | None = None,
) -> list[Path]:
    required_band = max(rgb_bands_to_indexes(RGB_bands)) if RGB_bands else None
    if path.is_file():
        if path.suffix.lower() == ".txt":
            files = [
                Path(line.strip()).expanduser().resolve()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            invalid = [file for file in files if not file.exists() or file.suffix.lower() not in {".tif", ".tiff"}]
            if invalid:
                raise DatasetStandardizationError(
                    "Tiff Slicing raster list contains invalid entries: "
                    + ", ".join(str(file) for file in invalid[:5])
                )
            if not files:
                raise DatasetStandardizationError(f"Tiff Slicing raster list is empty: {path}")
            return _filter_tiff_rasters_by_metadata(
                files,
                required_band=required_band,
                slicing_aoi=slicing_aoi,
                slicing_aoi_path=slicing_aoi_path,
            )
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise DatasetStandardizationError(f"Tiff Slicing requires .tif/.tiff rasters, got: {path}")
        return _filter_tiff_rasters_by_metadata(
            [path],
            required_band=required_band,
            slicing_aoi=slicing_aoi,
            slicing_aoi_path=slicing_aoi_path,
        )
    files = discover_candidate_tiff_rasters(path)
    if not files:
        raise DatasetStandardizationError(f"No .tif/.tiff rasters found under {path}")
    return _filter_tiff_rasters_by_metadata(
        files,
        required_band=required_band,
        slicing_aoi=slicing_aoi,
        slicing_aoi_path=slicing_aoi_path,
    )


def _filter_tiff_rasters_by_metadata(
    files: Sequence[Path],
    *,
    required_band: int | None,
    slicing_aoi: Any | None = None,
    slicing_aoi_path: Path | None = None,
) -> list[Path]:
    if required_band is None and slicing_aoi is None:
        return list(files)
    rasterio = require_rasterio()
    selected: list[Path] = []
    skipped: list[tuple[Path, int]] = []
    skipped_outside_aoi: list[Path] = []
    for file in files:
        try:
            with rasterio.open(file) as src:
                band_count = int(src.count)
                if required_band is not None and band_count < required_band:
                    skipped.append((file, band_count))
                    continue
                if slicing_aoi is not None:
                    aoi_filter = _slicing_aoi_filter_for_raster(slicing_aoi, slicing_aoi_path, src.crs)
                    if not _raster_intersects_aoi(src, aoi_filter):
                        skipped_outside_aoi.append(file)
                        continue
        except Exception as exc:
            raise DatasetStandardizationError(f"Failed to inspect GeoTIFF metadata: {file}: {exc}") from exc
        selected.append(file)
    if not selected:
        reasons: list[str] = []
        if required_band is not None and skipped:
            examples = ", ".join(f"{file} ({count} band)" for file, count in skipped[:5])
            reasons.append(
                f"no raster has enough bands for {required_band}-band RGB_bands; skipped examples: {examples}"
            )
        if slicing_aoi is not None and skipped_outside_aoi:
            examples = ", ".join(str(file) for file in skipped_outside_aoi[:5])
            reasons.append(f"all band-compatible rasters are outside the AOI; skipped examples: {examples}")
        detail = "; ".join(reasons) if reasons else "no raster matched the requested metadata filters"
        raise DatasetStandardizationError(f"No GeoTIFF rasters under the selected path can be sliced: {detail}")
    return selected


def discover_candidate_tiff_rasters(path: Path) -> list[Path]:
    """Return likely primary GeoTIFF rasters under a folder, excluding common sidecars."""

    files = sorted(
        file.resolve()
        for suffix in TIFF_RASTER_SUFFIXES
        for file in path.rglob(f"*{suffix}")
        if file.is_file()
    )
    if not files:
        return []
    primary = [file for file in files if is_candidate_primary_tiff(file, single_file=False)]
    return primary or files


def is_candidate_primary_tiff(path: Path, *, single_file: bool = False) -> bool:
    """Heuristic used by DAFT/TSFT TIFF discovery to avoid auxiliary rasters."""

    if path.suffix.lower() not in TIFF_RASTER_SUFFIXES:
        return False
    if single_file:
        return True
    name = path.name.lower()
    return not any(token in name for token in TIFF_AUXILIARY_NAME_TOKENS)


def _raster_intersects_aoi(src: Any, aoi_geometry: Any | None) -> bool:
    if aoi_geometry is None:
        return True
    raster_geometry = _raster_bounds_geometry(src)
    if raster_geometry.is_empty:
        return False
    return bool(aoi_geometry.intersects(raster_geometry))


def _raster_bounds_geometry(src: Any) -> Any:
    from shapely.geometry import box

    return box(*src.bounds)


def _discover_images_in_dir(path: Path) -> list[Path]:
    suffixes = set(IMAGE_SUFFIXES)
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in suffixes)


def _discover_raster_labels_in_dir(path: Path) -> list[Path]:
    suffixes = set(RASTER_LABEL_SUFFIXES)
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in suffixes)


def _pairing_errors(split: str, image_files: Sequence[Path], label_files: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    image_stems = [path.stem for path in image_files]
    label_stems = [path.stem for path in label_files]
    duplicate_images = sorted({stem for stem in image_stems if image_stems.count(stem) > 1})
    duplicate_labels = sorted({stem for stem in label_stems if label_stems.count(stem) > 1})
    if duplicate_images:
        errors.append(f"{split}: duplicate image stems: {', '.join(duplicate_images[:5])}")
    if duplicate_labels:
        errors.append(f"{split}: duplicate label stems: {', '.join(duplicate_labels[:5])}")
    if len(image_files) != len(label_files):
        errors.append(f"{split}: image/label count mismatch ({len(image_files)} images, {len(label_files)} labels)")
    missing_labels = sorted(set(image_stems) - set(label_stems))
    extra_labels = sorted(set(label_stems) - set(image_stems))
    if missing_labels:
        errors.append(f"{split}: missing labels for stems: {', '.join(missing_labels[:5])}")
    if extra_labels:
        errors.append(f"{split}: labels without images: {', '.join(extra_labels[:5])}")
    return errors


def _match_labels(image_files: Sequence[Path], label_root: Path | None) -> dict[str, Path]:
    if label_root is None:
        return {}
    if label_root.is_file():
        if len(image_files) == 1:
            return {image_files[0].stem: label_root}
        raise DatasetStandardizationError("A single label file can only be paired with one image during upload.")
    labels = {
        file.stem.replace("_label", ""): file
        for suffix in LABEL_SUFFIXES
        for file in label_root.rglob(f"*{suffix}")
    }
    missing = [path.stem for path in image_files if path.stem not in labels]
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetStandardizationError(f"Missing labels for image samples: {preview}")
    return labels


def _copy_label(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".shp":
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            sidecar = source.with_suffix(suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, target.with_suffix(suffix))
        return
    shutil.copy2(source, target)


def _is_taft_leaf_subset(path: Path) -> bool:
    return all(
        (path / split / child).is_dir()
        for split in ("train", "test")
        for child in ("images", "labels")
    )


def _discover_auxiliary_dirs(subset_dir: Path) -> list[str]:
    auxiliary_dirs: list[str] = []
    for split in ("train", "test"):
        split_dir = subset_dir / split
        if not split_dir.is_dir():
            continue
        for child in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if child.name in {"images", "labels"}:
                continue
            auxiliary_dirs.append(child.relative_to(subset_dir).as_posix())
    return auxiliary_dirs


def _suffixes(paths: Sequence[Path]) -> tuple[str, ...]:
    return tuple(sorted({path.suffix.lower() for path in paths}))


def _normalize_taft_tree_preset(value: str) -> str:
    preset = str(value).strip().lower()
    if preset not in TAFT_TREE_PRESETS:
        allowed = ", ".join(TAFT_TREE_PRESETS)
        raise DatasetStandardizationError(f"Unsupported TAFT tree preset '{value}'. Expected one of: {allowed}.")
    return preset


def _taft_tree_RGB_bands(source_rel: str, *, preset: str) -> tuple[str, str | None]:
    if preset != "landslide77k":
        return DEFAULT_RGB_BANDS, f"Unknown preset {preset}; defaulting RGB_bands to {DEFAULT_RGB_BANDS} for {source_rel}."

    parts = source_rel.split("/")
    top = parts[0]
    if top in {"LMHLD"}:
        return "RGB_bands: 3,2,1", None
    if top in {"ELNT", "ELNT_overlap"}:
        return "RGB_bands: 3,2,1", None
    if source_rel in {"landslide4sense", "HR-GLDD"}:
        return "RGB_bands: 3,2,1", None
    if _taft_source_is_rlzx_split(source_rel, split="SAT"):
        return "RGB_bands: 3,2,1", None
    if top in {"CAS"}:
        return DEFAULT_RGB_BANDS, None
    if _taft_source_is_rlzx_split(source_rel, split="UAV"):
        return DEFAULT_RGB_BANDS, None
    if source_rel in {"bijie_2", "GVLM", "GDCLD", "LRSTTC"}:
        return DEFAULT_RGB_BANDS, None
    return DEFAULT_RGB_BANDS, f"No Landslide77K RGB_bands rule for {source_rel}; defaulting to {DEFAULT_RGB_BANDS}."


def _taft_tree_label_polarity(source_rel: str, *, preset: str) -> tuple[str, str | None]:
    if preset != "landslide77k":
        return LABEL_POLARITY_FOREGROUND_WHITE, (
            f"Unknown preset {preset}; defaulting label_polarity to foreground_white for {source_rel}."
        )

    parts = source_rel.split("/")
    top = parts[0]
    if source_rel in {"bijie_2", "GVLM", "GDCLD"}:
        return LABEL_POLARITY_FOREGROUND_WHITE, None
    if top in {"CAS", "LMHLD"}:
        return LABEL_POLARITY_FOREGROUND_BLACK, None
    if top in {"ELNT", "ELNT_overlap"}:
        return LABEL_POLARITY_FOREGROUND_BLACK, None
    if source_rel in {"landslide4sense", "HR-GLDD", "LRSTTC"}:
        return LABEL_POLARITY_FOREGROUND_BLACK, None
    if _taft_source_is_rlzx_split(source_rel, split="SAT") or _taft_source_is_rlzx_split(source_rel, split="UAV"):
        return LABEL_POLARITY_FOREGROUND_BLACK, None
    return LABEL_POLARITY_FOREGROUND_WHITE, f"No Landslide77K label_polarity rule for {source_rel}; defaulting to foreground_white."


def _taft_source_is_rlzx_split(source_rel: str, *, split: str) -> bool:
    parts = source_rel.split("/")
    if len(parts) != 2:
        return False
    top, leaf = parts
    return top in {"RLZX", "RLZX_overlap", "RLZX_LDD_overlap"} and leaf.upper() == split.upper()


def _subset_id_from_rel(source_rel: str) -> str:
    return "__".join(_safe_name(part) for part in source_rel.split("/"))


def _ensure_taft_append_has_no_subset_conflicts(
    discovery: TaftTreeDiscovery,
    existing_subsets: Sequence[Mapping[str, Any]],
) -> None:
    existing_ids = {str(row.get("subset_id", "")) for row in existing_subsets}
    existing_rels = {str(row.get("source_rel", row.get("subset", ""))) for row in existing_subsets}
    conflicts: list[str] = []
    for subset in discovery.subsets:
        if subset.subset_id in existing_ids or subset.source_rel in existing_rels:
            conflicts.append(subset.source_rel)
    if conflicts:
        raise DatasetStandardizationError(
            "TAFT append source contains subsets that already exist in this version: " + ", ".join(conflicts[:12])
        )


def _apply_taft_subset_overrides(
    discovery: TaftTreeDiscovery,
    subset_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> TaftTreeDiscovery:
    if not discovery.subsets:
        return discovery
    if not isinstance(subset_overrides, Mapping) or not subset_overrides:
        raise DatasetStandardizationError(
            "TAFT subset annotations are required before import. "
            "Run discover_taft_tree first and pass subset_overrides with confirmed=True for every leaf subset."
        )

    by_source = {subset.source_rel: subset for subset in discovery.subsets}
    by_subset_id = {subset.subset_id: subset for subset in discovery.subsets}
    known_keys = set(by_source) | set(by_subset_id)
    unknown = sorted(str(key) for key in subset_overrides if str(key) not in known_keys)
    if unknown:
        raise DatasetStandardizationError(f"Unknown TAFT subset override key(s): {', '.join(unknown[:8])}")

    annotated: list[TaftTreeSubset] = []
    missing: list[str] = []
    for subset in discovery.subsets:
        raw = subset_overrides.get(subset.source_rel)
        if raw is None:
            raw = subset_overrides.get(subset.subset_id)
        if not isinstance(raw, Mapping):
            missing.append(subset.source_rel)
            continue
        if not bool(raw.get("confirmed", False)):
            missing.append(subset.source_rel)
            continue
        try:
            RGB_bands = normalize_rgb_bands(raw.get("RGB_bands", ""))
        except ValueError as exc:
            raise DatasetStandardizationError(
                f"Unsupported RGB_bands for TAFT subset {subset.source_rel}: {raw.get('RGB_bands', '')}. "
                "Expected RGB_bands: R,G,B using 1-based band indexes."
            ) from exc
        label_polarity = _normalize_label_polarity(raw.get("label_polarity", ""))
        annotated.append(
            replace(
                subset,
                RGB_bands=RGB_bands,
                label_polarity=label_polarity,
                annotation_source="user_confirmed",
                user_confirmed=True,
            )
        )
    if missing:
        raise DatasetStandardizationError(
            "TAFT subset annotations must be confirmed before import. Missing/unconfirmed: "
            + ", ".join(missing[:12])
        )
    return replace(discovery, subsets=tuple(annotated))


def _taft_subset_annotations_digest(discovery: TaftTreeDiscovery) -> str:
    payload = [
        {
            "source_rel": subset.source_rel,
            "RGB_bands": subset.RGB_bands,
            "label_polarity": subset.label_polarity,
            "user_confirmed": bool(subset.user_confirmed),
        }
        for subset in sorted(discovery.subsets, key=lambda item: item.source_rel)
    ]
    return hashlib.sha1(json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_label_polarity(value: str) -> str:
    text = str(value).strip().lower()
    if text not in LABEL_POLARITIES:
        allowed = ", ".join(LABEL_POLARITIES)
        raise DatasetStandardizationError(f"Unsupported label_polarity '{value}'. Expected one of: {allowed}.")
    return text


def _flatten_discovery_errors(discovery: TaftTreeDiscovery) -> list[str]:
    errors = list(discovery.errors)
    for subset in discovery.subsets:
        for error in subset.errors:
            errors.append(f"{subset.source_rel}: {error}")
    return errors


def _split_files(files: Sequence[Path], *, test_fraction: float) -> tuple[list[Path], list[Path]]:
    ordered = list(files)
    if len(ordered) <= 1:
        return ordered, []
    test_count = max(1, int(round(len(ordered) * test_fraction)))
    test_count = min(test_count, len(ordered) - 1)
    train_count = len(ordered) - test_count
    return ordered[:train_count], ordered[train_count:]


def _window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _candidate_window_starts_for_aoi(
    src: Any,
    *,
    tile_size: int,
    stride: int,
    aoi_geometry: Any | None,
) -> tuple[list[int], list[int]]:
    rows = _window_starts(int(src.height), tile_size, stride)
    cols = _window_starts(int(src.width), tile_size, stride)
    if aoi_geometry is None:
        return rows, cols
    if not _raster_intersects_aoi(src, aoi_geometry):
        return [], []

    try:
        from rasterio.windows import from_bounds

        left, bottom, right, top = aoi_geometry.bounds
        aoi_window = from_bounds(left, bottom, right, top, transform=src.transform)
    except Exception:
        return rows, cols

    min_row = max(0, int(math.ceil(aoi_window.row_off)))
    max_row = min(int(src.height), int(math.floor(aoi_window.row_off + aoi_window.height)))
    min_col = max(0, int(math.ceil(aoi_window.col_off)))
    max_col = min(int(src.width), int(math.floor(aoi_window.col_off + aoi_window.width)))
    if max_row <= min_row or max_col <= min_col:
        return [], []

    filtered_rows = [
        row
        for row in rows
        if row >= min_row and row + min(tile_size, int(src.height) - row) <= max_row
    ]
    filtered_cols = [
        col
        for col in cols
        if col >= min_col and col + min(tile_size, int(src.width) - col) <= max_col
    ]
    return filtered_rows, filtered_cols


def _count_daft_windows(
    raster_files: Sequence[Path],
    *,
    tile_size: int,
    stride: int,
    slicing_aoi: Any | None = None,
    slicing_aoi_path: Path | None = None,
) -> int:
    rasterio = require_rasterio()
    total = 0
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            aoi_filter = _slicing_aoi_filter_for_raster(slicing_aoi, slicing_aoi_path, src.crs)
            rows, cols = _candidate_window_starts_for_aoi(
                src,
                tile_size=tile_size,
                stride=stride,
                aoi_geometry=aoi_filter,
            )
            total += len(rows) * len(cols)
    return total


def _subset_stats(
    name: str,
    train_samples: Sequence[StandardizedSample],
    test_samples: Sequence[StandardizedSample],
    *,
    has_label: bool,
) -> dict[str, Any]:
    return {
        "subset": name,
        "sample_count": len(train_samples) + len(test_samples),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "has_label": has_label,
        "sample_layout": "(H, W, C)",
        "samples": [sample.to_dict() for sample in [*train_samples, *test_samples]],
    }


def _infer_subset_sample_dtype(subsets: Sequence[dict[str, Any]]) -> str:
    dtypes: set[str] = set()
    for subset in subsets:
        samples = subset.get("samples", []) if isinstance(subset, Mapping) else []
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            continue
        for sample in samples:
            if isinstance(sample, Mapping):
                dtype = str(sample.get("dtype") or "").strip()
                if dtype:
                    dtypes.add(dtype)
    if not dtypes:
        return "source_dtype"
    if len(dtypes) == 1:
        return next(iter(dtypes))
    return "mixed"


def _result(
    *,
    dataset_id: str,
    dataset_class: str,
    version: str,
    root_dir: Path,
    imagery_path: Path,
    label_path: Path | None,
    has_gt: bool,
    RGB_bands: str,
    subsets: Sequence[dict[str, Any]],
) -> StandardizedDatasetResult:
    sample_count = sum(int(row.get("sample_count", 0)) for row in subsets)
    train_count = sum(int(row.get("train_count", 0)) for row in subsets)
    test_count = sum(int(row.get("test_count", 0)) for row in subsets)
    subset_rows = tuple(dict(row) for row in subsets)
    return StandardizedDatasetResult(
        dataset_id=dataset_id,
        dataset_class=dataset_class,
        version=version,
        root_dir=root_dir,
        imagery_path=imagery_path,
        label_path=label_path,
        has_gt=has_gt,
        RGB_bands=RGB_bands,
        sample_shape_layout="(H, W, C)",
        sample_dtype=_infer_subset_sample_dtype(subsets),
        sample_count=sample_count,
        train_count=train_count,
        test_count=test_count,
        subsets=subset_rows,
        manifest_path=root_dir / MANIFEST_NAME,
        created_at=_utc_now(),
    )



def _manifest_subset_count(manifest: Mapping[str, Any]) -> int:
    explicit = manifest.get("subset_count")
    if explicit not in {None, ""}:
        return _int_value(explicit)
    subsets = manifest.get("subsets")
    return len(subsets) if isinstance(subsets, list) else 0


def _concat_source_subset(source: Mapping[str, Any]) -> dict[str, Any]:
    version = str(source.get("version") or "")
    return {
        "subset": version,
        "subset_id": version,
        "parent_group": "concat_sources",
        "sample_count": _int_value(source.get("sample_count")),
        "train_count": _int_value(source.get("train_count")),
        "test_count": _int_value(source.get("test_count")),
        "has_label": bool(source.get("has_gt")),
        "RGB_bands": source.get("RGB_bands", ""),
        "upload_method": "concat_source",
        "tile_schema": "virtual_concat_source",
        "source_version": version,
        "source_manifest_path": source.get("manifest_path", ""),
    }


def _concat_tsft_stats(
    source_manifests: Sequence[Mapping[str, Any]],
    source_versions: Sequence[str],
    warnings: list[str],
) -> dict[str, Any]:
    sum_fields = (
        "positive_count",
        "negative_count",
        "negative_candidate_count",
        "total_windows",
        "skipped_nodata",
        "skipped_empty_mask",
        "skipped_zero_rgb",
        "low_positive_windows",
    )
    totals = {field: 0 for field in sum_fields}
    weighted_mean_total = 0.0
    weighted_positive_total = 0
    for manifest, version in zip(source_manifests, source_versions):
        stats_source = _concat_tsft_stat_source(manifest)
        for field in sum_fields:
            totals[field] += _int_value(stats_source.get(field))
        positive_count = _int_value(stats_source.get("positive_count"))
        mean_value = stats_source.get("mean_positive_pixel_ratio")
        if mean_value in {None, ""}:
            warnings.append(f"TSFT source {version} missing mean_positive_pixel_ratio; skipped weighted mean.")
            continue
        try:
            mean_float = float(mean_value)
        except (TypeError, ValueError):
            warnings.append(f"TSFT source {version} has invalid mean_positive_pixel_ratio; skipped weighted mean.")
            continue
        weighted_mean_total += mean_float * positive_count
        weighted_positive_total += positive_count
    result: dict[str, Any] = dict(totals)
    result["mean_positive_pixel_ratio"] = (
        weighted_mean_total / weighted_positive_total if weighted_positive_total > 0 else None
    )
    return result


def _concat_tsft_stat_source(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    subsets = manifest.get("subsets")
    if not isinstance(subsets, list):
        return manifest
    subset_rows = [row for row in subsets if isinstance(row, Mapping)]
    if not subset_rows:
        return manifest
    sum_fields = (
        "positive_count",
        "negative_count",
        "negative_candidate_count",
        "total_windows",
        "skipped_nodata",
        "skipped_empty_mask",
        "skipped_zero_rgb",
        "low_positive_windows",
    )
    totals: dict[str, Any] = {field: sum(_int_value(row.get(field)) for row in subset_rows) for field in sum_fields}
    weighted_mean_total = 0.0
    weighted_positive_total = 0
    for row in subset_rows:
        positive_count = _int_value(row.get("positive_count"))
        mean_value = row.get("mean_positive_pixel_ratio")
        if mean_value in {None, ""}:
            continue
        try:
            mean_float = float(mean_value)
        except (TypeError, ValueError):
            continue
        weighted_mean_total += mean_float * positive_count
        weighted_positive_total += positive_count
    if weighted_positive_total > 0:
        totals["mean_positive_pixel_ratio"] = weighted_mean_total / weighted_positive_total
    elif "mean_positive_pixel_ratio" in manifest:
        totals["mean_positive_pixel_ratio"] = manifest.get("mean_positive_pixel_ratio")
    return totals

def _summary_row(manifest: Mapping[str, Any], subset: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "dataset_id": manifest.get("dataset_id", ""),
        "dataset_class": manifest.get("dataset_class", ""),
        "version": manifest.get("version", ""),
        "subset": subset.get("subset", ""),
        "subset_id": subset.get("subset_id", ""),
        "parent_group": subset.get("parent_group", ""),
        "sample_count": subset.get("sample_count", manifest.get("sample_count", 0)),
        "train_count": subset.get("train_count", manifest.get("train_count", 0)),
        "test_count": subset.get("test_count", manifest.get("test_count", 0)),
        "has_label": subset.get("has_label", manifest.get("has_gt", False)),
        "has_gt": manifest.get("has_gt", False),
        "RGB_bands": subset.get("RGB_bands", manifest.get("RGB_bands", "")),
        "label_polarity": subset.get("label_polarity", ""),
        "has_auxiliary": subset.get("has_auxiliary", False),
        "upload_method": subset.get("upload_method", manifest.get("standardization", {}).get("upload_method", "")),
        "tile_schema": subset.get("tile_schema", ""),
        "sample_layout": subset.get("sample_layout", manifest.get("sample_shape_layout", "(H, W, C)")),
        "sample_dtype": manifest.get("sample_dtype", "source_dtype"),
        "is_concat": bool(manifest.get("is_concat", False)),
        "concat_source_count": len(manifest.get("concat_sources", [])) if isinstance(manifest.get("concat_sources"), list) else 0,
        "positive_count": subset.get("positive_count", manifest.get("positive_count", "")),
        "negative_count": subset.get("negative_count", manifest.get("negative_count", "")),
        "mean_positive_pixel_ratio": subset.get("mean_positive_pixel_ratio", manifest.get("mean_positive_pixel_ratio", "")),
        "root_dir": manifest.get("root_dir", str(manifest_path.parent)),
        "manifest_path": str(manifest_path),
    }


def _browse_subset_row(subset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_group": subset.get("parent_group", ""),
        "subset": subset.get("subset", subset.get("source_rel", "")),
        "subset_id": subset.get("subset_id", ""),
        "sample_count": subset.get("sample_count", 0),
        "train_count": subset.get("train_count", 0),
        "test_count": subset.get("test_count", 0),
        "RGB_bands": subset.get("RGB_bands", ""),
        "label_polarity": subset.get("label_polarity", ""),
        "has_auxiliary": subset.get("has_auxiliary", False),
        "upload_method": subset.get("upload_method", ""),
        "tile_schema": subset.get("tile_schema", ""),
        "positive_count": subset.get("positive_count", ""),
        "negative_count": subset.get("negative_count", ""),
        "mean_positive_pixel_ratio": subset.get("mean_positive_pixel_ratio", ""),
        "subset_manifest_path": subset.get("subset_manifest_path", ""),
    }


def _merge_subsets(existing: Any, new_subsets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in existing if isinstance(row, Mapping)] if isinstance(existing, list) else []
    for subset in new_subsets:
        name = subset.get("subset")
        rows = [row for row in rows if row.get("subset") != name]
        rows.append(dict(subset))
    return rows


def _ensure_new_target(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise DatasetStandardizationError(f"Standardized dataset target already exists: {path}. Use overwrite=True.")


def _subset_name(value: str | None) -> str:
    if value:
        return _safe_name(value)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: str) -> str:
    import re

    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-")
    return text or "subset"


def _required_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise DatasetStandardizationError(f"Missing required value: {field_name}.")
    return _safe_name(text)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_dataset_class(value: str) -> str:
    key = str(value).strip()
    aliases = {"taft": "TAFT", "daft": "DAFT", "tsft": "TSFT", "evaluation": "Evaluation", "eval": "Evaluation"}
    normalized = aliases.get(key.lower(), key if key in DATASET_CLASSES else key.upper())
    if normalized not in DATASET_CLASSES:
        raise DatasetStandardizationError("dataset_class must be TAFT, DAFT, TSFT, or Evaluation.")
    return normalized


def _normalize_upload_method(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DAFT_UPLOAD_METHODS:
        allowed = ", ".join(DAFT_UPLOAD_METHODS)
        raise DatasetStandardizationError(f"upload_method must be one of: {allowed}.")
    return normalized


def _normalize_daft_upload_method(value: str) -> str:
    return _normalize_upload_method(value)


def _normalize_tsft_no_feature_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in TSFT_NO_FEATURE_POLICIES:
        allowed = ", ".join(TSFT_NO_FEATURE_POLICIES)
        raise DatasetStandardizationError(f"TSFT no_feature_tile_policy must be one of: {allowed}.")
    return normalized


def _positive_int(value: int | None, field_name: str) -> int:
    try:
        number = int(value) if value is not None else 0
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        raise DatasetStandardizationError(f"{field_name} must be a positive integer.")
    return number


def _normalize_ratio(value: float | None, field_name: str) -> float:
    ratio = DEFAULT_DAFT_MAX_NODATA_RATIO if value is None else float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise DatasetStandardizationError(f"{field_name} must be between 0 and 1.")
    return ratio


def _normalize_ratio_with_default(value: float | None, field_name: str, default: float) -> float:
    try:
        ratio = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise DatasetStandardizationError(f"{field_name} must be a number between 0 and 1.") from exc
    if ratio < 0.0 or ratio > 1.0:
        raise DatasetStandardizationError(f"{field_name} must be between 0 and 1.")
    return ratio


def _normalize_nonnegative_float(value: float | None, field_name: str, default: float) -> float:
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise DatasetStandardizationError(f"{field_name} must be a non-negative number.") from exc
    if number < 0.0:
        raise DatasetStandardizationError(f"{field_name} must be a non-negative number.")
    return number


def _tsft_tiff_tile_schema(
    *,
    tile_size: int,
    stride: int,
    max_nodata_ratio: float,
    min_polygon_overlap_ratio: float,
    min_positive_pixel_ratio: float,
    no_feature_tile_policy: str,
    negative_ratio: float,
    slicing_aoi_path: Path | None = None,
) -> str:
    schema = (
        f"tiff_slicing;tile_size={tile_size};stride={stride};"
        f"max_nodata_ratio={max_nodata_ratio:.2f};"
        f"min_polygon_overlap_ratio={min_polygon_overlap_ratio:.3f};"
        f"min_positive_pixel_ratio={min_positive_pixel_ratio:.3f};"
        f"no_feature_tile_policy={no_feature_tile_policy};"
        f"negative_ratio={negative_ratio:.2f}"
    )
    return _append_slicing_aoi_tile_schema(schema, slicing_aoi_path=slicing_aoi_path)


def _append_slicing_aoi_tile_schema(schema: str, *, slicing_aoi_path: Path | None) -> str:
    if slicing_aoi_path is None:
        return schema
    return f"{schema};aoi=strict_inside"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, payload: Any, *, workspace_root: str | Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload, workspace_root=workspace_root)
    ready = repair_portable_paths_in_payload(ready, workspace_root=workspace_root)
    path.write_text(
        json.dumps(ready, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(dict(payload)), ensure_ascii=False))
        handle.write("\n")


def _json_ready(value: Any, *, workspace_root: str | Path | None = None) -> Any:
    if isinstance(value, Path):
        return portable_path(value, workspace_root=workspace_root) or str(value)
    if isinstance(value, tuple):
        return [_json_ready(item, workspace_root=workspace_root) for item in value]
    if isinstance(value, list):
        return [_json_ready(item, workspace_root=workspace_root) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item, workspace_root=workspace_root) for key, item in value.items()}
    return value


def _workspace_root_from_store_root(store_root: str | Path | None) -> Path | None:
    if store_root in {None, ""}:
        return None
    root = Path(store_root).expanduser().resolve(strict=False)
    if root.name == "registry":
        return root.parent
    if root.name == TOKEN_DATASET_STORE_DIRNAME and root.parent.name == "registry":
        return root.parent.parent
    return root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
