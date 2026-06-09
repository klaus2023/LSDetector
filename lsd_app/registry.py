"""Local registries for models, weights, datasets, and prediction runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from lsd_app.core.channels import DEFAULT_RGB_BANDS, rgb_bands_to_class, normalize_rgb_bands
from lsd_app.core.paths import (
    REGISTRY_DIRNAME,
    portable_path,
    registry_dir as _portable_registry_dir,
    repair_portable_paths_in_payload,
    resolve_user_path,
    workspace_root as _portable_workspace_root,
)
from lsd_app.data import (
    DAFT_UPLOAD_METHOD_PACKAGED,
    DAFT_UPLOAD_METHOD_TIFF_SLICING,
    LABEL_POLARITY_FOREGROUND_WHITE,
    DatasetStandardizationError,
    build_dataset_repository_summary,
    create_concat_dataset_version as create_concat_dataset_version_data,
    browse_dataset_store,
    discover_taft_tree as discover_taft_tree_data,
    delete_dataset_store_version as delete_dataset_store_version_data,
    inspect_daft_upload_target as inspect_daft_upload_target_data,
    register_taft_tree as register_taft_tree_data,
    standardize_dataset_upload,
)
from lsd_app.models import MODEL_SPECS, ModelSpec, canonical_model_names, normalize_model_name


DATASET_CLASSES = ("TAFT", "DAFT", "TSFT", "Evaluation")
DATASET_TYPES = DATASET_CLASSES
WEIGHT_STAGES = ("TAFT", "DAFT", "TSFT", "inference-ready")


class RegistryError(ValueError):
    """Raised when a local registry entry is invalid."""


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Serializable model repository entry."""

    model_id: str
    net_name: str
    adapter: str
    model_family: str
    input_mode: str
    channel_rule: str
    status: str
    description: str
    code_path: str = ""
    model_options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = _json_ready(asdict(self))
        data.update(
            {
                "name": data["net_name"],
                "adapter_name": data["adapter"],
                "role": data["description"],
                "edge_mode": "wavelets" if data["input_mode"] == "rgb_wavelets" else None,
            }
        )
        return data


@dataclass(frozen=True)
class WeightEntry:
    """Serializable weight repository entry."""

    weight_id: str
    model_id: str
    stage: str
    version: str
    path: Path
    exists: bool
    created_at: str
    metrics_summary: str
    notes: str
    weight_type: str = "checkpoint"
    base_checkpoint_path: Path | None = None
    adapter_profile: str = ""
    adapter_name: str = ""
    image_size: int | None = None
    input_mode: str = ""
    channel_rule: str = ""
    lora_base_r: int | None = None
    lora_base_alpha: int | None = None
    lora_task_r: int | None = None
    lora_task_alpha: int | None = None
    lora_dropout: float | None = None
    peft_adapter_name: str = ""
    target_modules: str = ""
    source_path: Path | None = None
    stored: bool = False
    file_size_bytes: int | None = None
    parameter_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class DatasetEntry:
    """Serializable local dataset repository entry."""

    dataset_id: str
    name: str
    dataset_class: str
    imagery_path: Path
    label_path: Path | None
    has_gt: bool
    RGB_bands: str
    RGB_bands_class: str
    tile_schema: str
    source: str
    created_at: str
    notes: str
    description: str
    dataset_type: str | None = None
    truth_path: Path | None = None
    version: str = ""
    repository_path: Path | None = None
    manifest_path: Path | None = None
    sample_shape_layout: str = ""
    sample_dtype: str = ""
    sample_count: int = 0
    train_count: int = 0
    test_count: int = 0
    subset_count: int = 0
    skipped_samples: int = 0
    written_samples: int = 0
    rewritten_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = _json_ready(asdict(self))
        data["dataset_type"] = data["dataset_class"]
        data["class"] = data["RGB_bands_class"]
        data["truth_path"] = data["label_path"]
        data["description"] = data["notes"]
        return data


@dataclass(frozen=True)
class PredictionRunEntry:
    """Serializable prediction deployment entry."""

    run_id: str
    model_id: str
    input_raster_path: Path
    output_dir: Path
    polygons_path: Path
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def registry_dir(root: str | Path) -> Path:
    """Return the registry directory under a local workspace/output root."""

    return _portable_registry_dir(root)


def default_model_entries(*, project_root: str | Path = ".") -> list[ModelRegistryEntry]:
    """Build default model repository entries from code specs."""

    return [
        _model_entry_from_spec(model_id, spec)
        for model_id, spec in ((name, MODEL_SPECS[name]) for name in canonical_model_names())
    ]


def list_models(*, project_root: str | Path = ".", registry_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return JSON-ready model registry rows."""

    rows = [entry.to_dict() for entry in default_model_entries(project_root=project_root)]
    if registry_root is not None:
        existing_ids = {row["model_id"] for row in rows}
        rows.extend(row for row in list_custom_models(registry_root=registry_root) if row.get("model_id") not in existing_ids)
    return rows


def register_model(
    *,
    registry_root: str | Path,
    model_id: str,
    net_name: str,
    adapter: str,
    model_family: str,
    input_mode: str,
    channel_rule: str,
    status: str = "custom",
    description: str = "",
) -> ModelRegistryEntry:
    """Persist one custom model repository entry."""

    model_id = _required_text(model_id, "model_id")
    if input_mode not in {"rgb", "rgb_wavelets"}:
        raise RegistryError("Model input_mode must be 'rgb' or 'rgb_wavelets'.")
    entry = ModelRegistryEntry(
        model_id=model_id,
        net_name=_required_text(net_name, "net_name"),
        adapter=_required_text(adapter, "adapter"),
        model_family=_required_text(model_family, "model_family"),
        input_mode=input_mode,
        channel_rule=_required_text(channel_rule, "channel_rule"),
        status=_required_text(status, "status"),
        description=description,
    )
    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    rows = _read_list(root / "models.json")
    rows = [row for row in rows if row.get("model_id") != model_id]
    rows.append(entry.to_dict())
    _write_json(root / "models.json", rows)
    return entry


def list_custom_models(*, registry_root: str | Path) -> list[dict[str, Any]]:
    """Return user-registered model rows."""

    return _read_list(registry_dir(registry_root) / "models.json")


def register_weight(
    *,
    registry_root: str | Path,
    model_id: str,
    stage: str,
    version: str,
    path: str | Path,
    metrics_summary: str = "",
    notes: str = "",
    weight_id: str | None = None,
    weight_type: str = "checkpoint",
    base_checkpoint_path: str | Path | None = None,
    adapter_profile: str = "",
    adapter_name: str = "",
    image_size: int | None = None,
    input_mode: str = "",
    channel_rule: str = "",
    lora_base_r: int | None = None,
    lora_base_alpha: int | None = None,
    lora_task_r: int | None = None,
    lora_task_alpha: int | None = None,
    lora_dropout: float | None = None,
    peft_adapter_name: str = "",
    target_modules: str = "",
    store_copy: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> WeightEntry:
    """Validate and persist one local checkpoint/weight entry."""

    model_id = _required_text(model_id, "model_id")
    try:
        canonical_model_id = normalize_model_name(model_id)
    except ValueError:
        canonical_model_id = model_id
    stage = _normalize_weight_stage(stage)
    weight_path = _resolve_user_path(path)
    if not weight_path.exists():
        raise RegistryError(f"Weight path does not exist: {weight_path}")
    version = str(version).strip() or weight_path.stem or _utc_compact()
    base_checkpoint = (
        _resolve_user_path(base_checkpoint_path)
        if base_checkpoint_path not in {None, ""}
        else None
    )
    if base_checkpoint is not None and not base_checkpoint.exists():
        raise RegistryError(f"Base checkpoint path does not exist: {base_checkpoint}")
    spec = MODEL_SPECS.get(canonical_model_id)
    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    rows = _read_list(root / "weights.json")
    weight_id_value = _stage_aware_weight_id(
        requested_weight_id=weight_id,
        canonical_model_id=canonical_model_id,
        stage=stage,
        version=version,
        existing_rows=rows,
    )
    stored_path = _store_weight_file(
        root=root,
        source_path=weight_path,
        weight_id=weight_id_value,
        stage=stage,
        model_id=canonical_model_id,
        store_copy=store_copy,
        progress_callback=progress_callback,
    )
    entry = WeightEntry(
        weight_id=weight_id_value,
        model_id=canonical_model_id,
        stage=stage,
        version=version,
        path=stored_path,
        exists=stored_path.exists(),
        created_at=_utc_now(),
        metrics_summary=metrics_summary,
        notes=notes,
        weight_type=_required_text(weight_type, "weight_type"),
        base_checkpoint_path=base_checkpoint,
        adapter_profile=adapter_profile,
        adapter_name=adapter_name or (spec.adapter_name if spec is not None else ""),
        image_size=image_size,
        input_mode=input_mode or (spec.input_mode if spec is not None else ""),
        channel_rule=channel_rule or (_channel_rule(spec) if spec is not None else ""),
        lora_base_r=lora_base_r,
        lora_base_alpha=lora_base_alpha,
        lora_task_r=lora_task_r,
        lora_task_alpha=lora_task_alpha,
        lora_dropout=lora_dropout,
        peft_adapter_name=peft_adapter_name,
        target_modules=target_modules,
        source_path=weight_path,
        stored=stored_path != weight_path,
        file_size_bytes=_weight_file_size(stored_path),
        parameter_count=_weight_parameter_count(stored_path),
    )
    rows = [row for row in rows if row.get("weight_id") != entry.weight_id]
    rows.append(entry.to_dict())
    _write_json(root / "weights.json", rows, workspace_root=workspace)
    return entry


def _stage_aware_weight_id(
    *,
    requested_weight_id: str | None,
    canonical_model_id: str,
    stage: str,
    version: str,
    existing_rows: list[Mapping[str, Any]],
) -> str:
    """Return a weight id that can coexist across TAFT/DAFT/TSFT stages."""

    base_id = _normalize_weight_id(
        requested_weight_id or _slug_id(f"{canonical_model_id}-{stage}-{version}", prefix="weight")
    )
    existing = next((row for row in existing_rows if row.get("weight_id") == base_id), None)
    if existing is None or str(existing.get("stage") or "").strip() == stage:
        return base_id
    candidate = _normalize_weight_id(f"{stage}-{base_id}")
    used_ids = {str(row.get("weight_id") or "") for row in existing_rows}
    if candidate not in used_ids:
        return candidate
    suffix = 2
    while True:
        numbered = _normalize_weight_id(f"{candidate}-{suffix}")
        if numbered not in used_ids:
            return numbered
        suffix += 1


def list_weights(*, registry_root: str | Path) -> list[dict[str, Any]]:
    """Return registered local checkpoint/weight rows."""

    workspace = _workspace_from_registry_root(registry_root)
    rows = _read_list(registry_dir(workspace) / "weights.json")
    listed: list[dict[str, Any]] = []
    for row in rows:
        path = _resolve_registry_path(row.get("path"), workspace_root=workspace) if row.get("path") else None
        source_path = _resolve_registry_path(row.get("source_path"), workspace_root=workspace) if row.get("source_path") else None
        path_exists = bool(path and path.exists())
        row_file_size = row.get("file_size_bytes")
        if row_file_size in {None, ""} and path_exists and path is not None:
            row_file_size = _weight_file_size(path)
        listed.append(
            {
                **row,
                "resolved_path": str(path) if path is not None else "",
                "resolved_source_path": str(source_path) if source_path is not None else "",
                "exists": path_exists,
                "source_exists": source_path.exists() if source_path is not None else None,
                "file_size_bytes": row_file_size,
                "parameter_count": row.get("parameter_count"),
            }
        )
    return listed


def delete_weight(*, registry_root: str | Path, weight_id: str, delete_stored_file: bool = True) -> dict[str, Any]:
    """Delete one weight registry entry and its internal stored copy when available."""

    normalized_weight_id = _normalize_weight_id(weight_id)
    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    weights_path = root / "weights.json"
    rows = _read_list(weights_path)
    match = next((row for row in rows if row.get("weight_id") == normalized_weight_id), None)
    if match is None:
        raise RegistryError(f"Unknown weight id: {normalized_weight_id}")

    removed_path: str | None = None
    if delete_stored_file and bool(match.get("stored")):
        stored_path = _resolve_registry_path(match.get("path"), workspace_root=workspace)
        if _is_relative_to(stored_path, root / "weight_repository"):
            target = stored_path.parent if stored_path.is_file() else stored_path
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed_path = str(target)

    kept = [row for row in rows if row.get("weight_id") != normalized_weight_id]
    _write_json(weights_path, kept, workspace_root=workspace)
    return {
        "weight_id": normalized_weight_id,
        "removed_registry_entry": True,
        "removed_path": removed_path,
    }


def register_dataset(
    *,
    registry_root: str | Path,
    name: str,
    dataset_type: str | None = None,
    dataset_class: str | None = None,
    imagery_path: str | Path,
    truth_path: str | Path | None = None,
    label_path: str | Path | None = None,
    test_imagery_path: str | Path | None = None,
    test_label_path: str | Path | None = None,
    RGB_bands: str,
    label_polarity: str = LABEL_POLARITY_FOREGROUND_WHITE,
    has_gt: bool | None = None,
    tile_schema: str = "unverified",
    source: str = "user",
    description: str = "",
    notes: str = "",
    version: str | None = None,
    subset_name: str | None = None,
    standardize: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    upload_method: str = DAFT_UPLOAD_METHOD_PACKAGED,
    tile_size: int | None = None,
    stride: int | None = None,
    daft_max_nodata_ratio: float | None = None,
    slicing_aoi_path: str | Path | None = None,
    tsft_min_polygon_overlap_ratio: float | None = None,
    tsft_min_positive_pixel_ratio: float | None = None,
    tsft_no_feature_tile_policy: str = "balanced",
    tsft_negative_ratio: float | None = None,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    """Validate and persist one local dataset entry."""

    dataset_class = _normalize_dataset_class(dataset_class or dataset_type)
    RGB_bands = _normalize_rgb_bands(RGB_bands)
    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    version_text = version or _utc_compact()
    dataset_id = _slug_id(version_text, prefix=dataset_class.lower())

    imagery = _resolve_user_path(imagery_path)
    if not imagery.exists():
        raise RegistryError(f"Imagery path does not exist: {imagery}")

    label_input = label_path if label_path not in {None, ""} else truth_path
    label = _resolve_user_path(label_input) if label_input not in {None, ""} else None
    if label is not None and not label.exists():
        raise RegistryError(f"Label/ground-truth path does not exist: {label}")
    slicing_aoi = _resolve_user_path(slicing_aoi_path) if slicing_aoi_path not in {None, ""} else None
    if slicing_aoi is not None and not slicing_aoi.exists():
        raise RegistryError(f"Slicing AOI path does not exist: {slicing_aoi}")
    test_imagery = _resolve_user_path(test_imagery_path) if test_imagery_path not in {None, ""} else None
    test_label = _resolve_user_path(test_label_path) if test_label_path not in {None, ""} else None
    if test_imagery is not None or test_label is not None:
        raise RegistryError(
            "TSFT test split inputs are no longer supported; register labeled evaluation data as an Evaluation dataset."
        )
    if dataset_class == "DAFT" and label is not None:
        raise RegistryError("DAFT datasets are images-only and must not include label/truth paths.")
    if dataset_class == "Evaluation" and label is None:
        raise RegistryError("Evaluation datasets require a label_path/ground-truth path.")
    generated_tsft_slicing_labels = (
        dataset_class == "TSFT"
        and upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
        and standardize
    )
    inferred_has_gt = label is not None or generated_tsft_slicing_labels
    has_gt = True if dataset_class == "Evaluation" else (inferred_has_gt if has_gt is None else bool(has_gt))
    if dataset_class == "Evaluation" and has_gt is not True:
        raise RegistryError("Evaluation datasets must have has_gt=True.")
    if has_gt and label is None and not generated_tsft_slicing_labels:
        raise RegistryError("Datasets marked has_gt require a label_path/ground-truth path.")

    standardized_result = None
    if standardize:
        try:
            standardized_result = standardize_dataset_upload(
                store_root=root,
                dataset_id=dataset_id,
                dataset_class=dataset_class,
                version=version_text,
                imagery_path=imagery,
                label_path=label,
                test_imagery_path=test_imagery,
                test_label_path=test_label,
                RGB_bands=RGB_bands,
                label_polarity=label_polarity,
                subset_name=subset_name,
                source=source,
                notes=notes or description,
                overwrite=overwrite,
                resume=resume,
                upload_method=upload_method,
                tile_size=tile_size,
                stride=stride,
                daft_max_nodata_ratio=daft_max_nodata_ratio,
                slicing_aoi_path=slicing_aoi,
                tsft_min_polygon_overlap_ratio=tsft_min_polygon_overlap_ratio,
                tsft_min_positive_pixel_ratio=tsft_min_positive_pixel_ratio,
                tsft_no_feature_tile_policy=tsft_no_feature_tile_policy,
                tsft_negative_ratio=tsft_negative_ratio,
                progress_callback=progress_callback,
            )
        except DatasetStandardizationError as exc:
            raise RegistryError(str(exc)) from exc
        imagery = standardized_result.imagery_path
        label = standardized_result.label_path
        has_gt = standardized_result.has_gt
        tile_schema = f"standardized:{standardized_result.sample_shape_layout}"
        if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and standardized_result.subsets:
            standardized_tile_schema = str(standardized_result.subsets[0].get("tile_schema") or "").strip()
            if standardized_tile_schema:
                tile_schema = standardized_tile_schema

    notes_text = notes or description
    entry = DatasetEntry(
        dataset_id=dataset_id,
        name=name,
        dataset_class=dataset_class,
        imagery_path=imagery,
        label_path=label,
        has_gt=has_gt,
        RGB_bands=RGB_bands,
        RGB_bands_class=rgb_bands_to_class(RGB_bands),
        tile_schema=(
            _daft_tiff_tile_schema(
                tile_size=tile_size,
                stride=stride,
                max_nodata_ratio=daft_max_nodata_ratio,
                slicing_aoi_path=slicing_aoi,
            )
            if standardized_result is None
            and dataset_class == "DAFT"
            and upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
            else _tsft_tiff_tile_schema(
                tile_size=tile_size,
                stride=stride,
                max_nodata_ratio=daft_max_nodata_ratio,
                min_polygon_overlap_ratio=tsft_min_polygon_overlap_ratio,
                min_positive_pixel_ratio=tsft_min_positive_pixel_ratio,
                no_feature_tile_policy=tsft_no_feature_tile_policy,
                negative_ratio=tsft_negative_ratio,
                slicing_aoi_path=slicing_aoi,
            )
            if standardized_result is None
            and dataset_class == "TSFT"
            and upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
            else tile_schema
        ),
        source=source,
        created_at=_utc_now(),
        notes=notes_text,
        description=notes_text,
        dataset_type=dataset_class,
        truth_path=label,
        version=standardized_result.version if standardized_result is not None else version_text,
        repository_path=standardized_result.root_dir if standardized_result is not None else None,
        manifest_path=standardized_result.manifest_path if standardized_result is not None else None,
        sample_shape_layout=standardized_result.sample_shape_layout if standardized_result is not None else "",
        sample_dtype=standardized_result.sample_dtype if standardized_result is not None else "",
        sample_count=standardized_result.sample_count if standardized_result is not None else _count_images(imagery),
        train_count=standardized_result.train_count if standardized_result is not None else _count_split_images(imagery, "train"),
        test_count=standardized_result.test_count if standardized_result is not None else _count_split_images(imagery, "test"),
        subset_count=len(standardized_result.subsets) if standardized_result is not None else 0,
    )
    rows = _read_list(root / "datasets.json")
    rows = [row for row in rows if row.get("dataset_id") != dataset_id]
    rows.append(entry.to_dict())
    _write_json(root / "datasets.json", rows, workspace_root=workspace)
    return entry



def register_concat_dataset_entry(
    *,
    registry_root: str | Path,
    dataset_class: str,
    version: str,
    source_versions: list[str] | tuple[str, ...],
    notes: str = "",
) -> DatasetEntry:
    """Register a virtual DAFT/TSFT concat dataset-store entry."""

    dataset_class = _normalize_dataset_class(dataset_class)
    if dataset_class not in {"DAFT", "TSFT"}:
        raise RegistryError("Concat datasets are only supported for DAFT or TSFT.")
    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    version_text = _path_segment(_required_text(version, "version"))
    dataset_id = _slug_id(version_text, prefix=dataset_class.lower())
    try:
        result = create_concat_dataset_version_data(
            store_root=root,
            dataset_class=dataset_class,
            new_version=version_text,
            source_versions=source_versions,
            dataset_id=dataset_id,
            name=version_text,
            notes=notes,
        )
    except DatasetStandardizationError as exc:
        raise RegistryError(str(exc)) from exc

    entry = DatasetEntry(
        dataset_id=result.dataset_id,
        name=version_text,
        dataset_class=dataset_class,
        imagery_path=result.imagery_path,
        label_path=result.label_path,
        has_gt=result.has_gt,
        RGB_bands=result.RGB_bands,
        RGB_bands_class=rgb_bands_to_class(result.RGB_bands),
        tile_schema="virtual_concat",
        source="concat",
        created_at=result.created_at,
        notes=notes,
        description=notes,
        dataset_type=dataset_class,
        truth_path=result.label_path,
        version=result.version,
        repository_path=result.root_dir,
        manifest_path=result.manifest_path,
        sample_shape_layout=result.sample_shape_layout,
        sample_dtype=result.sample_dtype,
        sample_count=result.sample_count,
        train_count=result.train_count,
        test_count=result.test_count,
        subset_count=len(result.subsets),
    )
    datasets_path = root / "datasets.json"
    rows = _read_list(datasets_path)
    rows = [row for row in rows if row.get("dataset_id") != entry.dataset_id]
    rows.append(entry.to_dict())
    _write_json(datasets_path, rows, workspace_root=workspace)
    return entry

def inspect_daft_upload(
    *,
    registry_root: str | Path,
    version: str,
    imagery_path: str | Path,
    RGB_bands: str,
    upload_method: str = DAFT_UPLOAD_METHOD_PACKAGED,
    tile_size: int | None = None,
    stride: int | None = None,
    daft_max_nodata_ratio: float | None = None,
    slicing_aoi_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return DAFT upload target status using the same path aliases as registration."""

    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    version_text = _required_text(version, "version")
    dataset_id = _slug_id(version_text, prefix="daft")
    imagery = _resolve_user_path(imagery_path)
    slicing_aoi = _resolve_user_path(slicing_aoi_path) if slicing_aoi_path not in {None, ""} else None
    return inspect_daft_upload_target_data(
        store_root=root,
        dataset_id=dataset_id,
        version=version_text,
        imagery_path=imagery,
        RGB_bands=RGB_bands,
        upload_method=upload_method,
        tile_size=tile_size,
        stride=stride,
        daft_max_nodata_ratio=daft_max_nodata_ratio,
        slicing_aoi_path=slicing_aoi,
    )


def list_datasets(*, registry_root: str | Path) -> list[dict[str, Any]]:
    """Return registered local datasets."""

    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    rows = _read_list(root / "datasets.json")
    listed: list[dict[str, Any]] = []
    for row in rows:
        resolved = dict(row)
        for key in ("imagery_path", "label_path", "truth_path", "repository_path", "manifest_path"):
            if row.get(key):
                try:
                    resolved[f"resolved_{key}"] = str(_resolve_registry_path(row.get(key), workspace_root=workspace))
                except RegistryError:
                    resolved[f"resolved_{key}"] = ""
        listed.append(resolved)
    return listed


def list_dataset_repository_summary(*, registry_root: str | Path) -> list[dict[str, Any]]:
    """Return standardized dataset/subset summary rows for UI and CLI tables."""

    return build_dataset_repository_summary(store_root=registry_dir(registry_root))


def browse_dataset_repository(*, registry_root: str | Path) -> dict[str, Any]:
    """Return TAFT/DAFT/TSFT repository folders and version/subset summaries."""

    return browse_dataset_store(store_root=registry_dir(registry_root))


def delete_dataset_store_entry(
    *,
    registry_root: str | Path,
    dataset_class: str,
    version: str,
    repository_path: str | Path | None = None,
) -> dict[str, Any]:
    """Delete a standardized dataset-store version and remove matching registry entries."""

    workspace = _workspace_from_registry_root(registry_root)
    registry = registry_dir(workspace)
    datasets_path = registry / "datasets.json"
    dataset_class = _normalize_dataset_class(dataset_class)
    version = _required_text(version, "version")
    expected_path = _expected_dataset_store_path(registry=registry, dataset_class=dataset_class, version=version)
    supplied_path: Path | None = None
    if repository_path not in {None, ""}:
        supplied_path = _resolve_registry_path(repository_path, workspace_root=workspace)
        if supplied_path != expected_path:
            raise RegistryError(
                "Refusing to delete dataset-store version because the selected path does not match "
                f"{dataset_class}/{version}: selected {supplied_path}, expected {expected_path}."
            )
    rows = _read_list(datasets_path)
    matching_rows = [
        row
        for row in rows
        if row.get("dataset_class") == dataset_class
        and row.get("version") == version
        and (
            repository_path in {None, ""}
            or _registry_row_repository_matches(row, expected_path=expected_path, workspace=workspace)
        )
    ]
    if not matching_rows and any(row.get("dataset_class") == dataset_class and row.get("version") == version for row in rows):
        raise RegistryError(
            f"Refusing to delete {dataset_class}/{version}: registry row path does not match the selected path."
        )
    try:
        result = delete_dataset_store_version_data(
            store_root=registry,
            dataset_class=dataset_class,
            version=version,
        )
    except DatasetStandardizationError as exc:
        if not matching_rows:
            raise RegistryError(str(exc)) from exc
        dataset_class = str(matching_rows[0].get("dataset_class") or dataset_class)
        version = str(matching_rows[0].get("version") or version)
        result = {
            "dataset_class": dataset_class,
            "version": version,
            "deleted_path": str(matching_rows[0].get("repository_path") or ""),
            "already_missing": True,
            "delete_warning": str(exc),
        }
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    deleted_path = str(result["deleted_path"])
    for row in rows:
        same_class_version = row.get("dataset_class") == result["dataset_class"] and row.get("version") == result["version"]
        same_repository = _registry_row_repository_matches(row, expected_path=expected_path, workspace=workspace)
        if repository_path not in {None, ""}:
            should_remove = same_class_version and same_repository
        else:
            should_remove = same_class_version or same_repository
        if should_remove:
            removed.append(row)
        else:
            kept.append(row)
    _write_json(datasets_path, kept, workspace_root=workspace)
    return {**result, "removed_registry_entries": len(removed)}


def _expected_dataset_store_path(*, registry: Path, dataset_class: str, version: str) -> Path:
    return (registry / "dataset_store" / dataset_class / version).expanduser().resolve()


def _registry_row_repository_matches(row: Mapping[str, Any], *, expected_path: Path, workspace: Path) -> bool:
    repository_path = row.get("repository_path")
    if repository_path in {None, ""}:
        return False
    try:
        resolved = _resolve_registry_path(repository_path, workspace_root=workspace)
    except RegistryError:
        return False
    if resolved == expected_path:
        return True
    portable_expected = portable_path(expected_path, workspace_root=workspace)
    return bool(portable_expected and row.get("repository_path") == portable_expected)


def discover_taft_tree(root: str | Path, *, preset: str = "landslide77k") -> dict[str, Any]:
    """Return a JSON-ready recursive discovery summary for a hierarchical TAFT source tree."""

    try:
        return discover_taft_tree_data(_resolve_user_path(root), preset=preset).to_dict()
    except DatasetStandardizationError as exc:
        raise RegistryError(str(exc)) from exc


def register_taft_tree(
    *,
    registry_root: str | Path,
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
    notes: str = "",
    subset_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: Any | None = None,
) -> DatasetEntry:
    """Register one hierarchical TAFT version as a single aggregate dataset entry."""

    workspace = _workspace_from_registry_root(registry_root)
    registry = registry_dir(workspace)
    registry.mkdir(parents=True, exist_ok=True)
    version = _required_text(version, "version")
    display_name = name or version
    try:
        result = register_taft_tree_data(
            store_root=registry,
            name=display_name,
            version=version,
            root=_resolve_user_path(root),
            preset=preset,
            standardize=standardize,
            overwrite=overwrite,
            resume=resume,
            force_rebuild=force_rebuild,
            append_to_existing=append_to_existing,
            split_policy=split_policy,
            source=preset,
            notes=notes,
            subset_overrides=subset_overrides,
            progress_callback=progress_callback,
        )
    except DatasetStandardizationError as exc:
        raise RegistryError(str(exc)) from exc

    entry = DatasetEntry(
        dataset_id=result.dataset_id,
        name=version,
        dataset_class="TAFT",
        imagery_path=result.root_dir,
        label_path=result.root_dir,
        has_gt=True,
        RGB_bands=result.RGB_bands,
        RGB_bands_class="MIXED",
        tile_schema=f"standardized-tree:{result.sample_shape_layout};split_policy={split_policy}",
        source=preset,
        created_at=result.created_at,
        notes=notes,
        description=notes,
        dataset_type="TAFT",
        truth_path=result.root_dir,
        version=result.version,
        repository_path=result.root_dir,
        manifest_path=result.manifest_path,
        sample_shape_layout=result.sample_shape_layout,
        sample_dtype=result.sample_dtype,
        sample_count=result.sample_count,
        train_count=result.train_count,
        test_count=result.test_count,
        subset_count=result.subset_count,
        skipped_samples=result.skipped_samples,
        written_samples=result.written_samples,
        rewritten_samples=result.rewritten_samples,
    )
    rows = _read_list(registry / "datasets.json")
    rows = [
        row
        for row in rows
        if not (row.get("dataset_class") == "TAFT" and row.get("version") == entry.version)
    ]
    rows.append(entry.to_dict())
    _write_json(registry / "datasets.json", rows, workspace_root=workspace)
    return entry


def register_prediction_run(
    *,
    registry_root: str | Path,
    run_id: str,
    model_id: str,
    input_raster_path: str | Path,
    output_dir: str | Path,
    polygons_path: str | Path,
) -> PredictionRunEntry:
    """Persist a completed prediction/deployment run."""

    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    entry = PredictionRunEntry(
        run_id=run_id,
        model_id=model_id,
        input_raster_path=_resolve_user_path(input_raster_path),
        output_dir=_resolve_user_path(output_dir),
        polygons_path=_resolve_user_path(polygons_path),
        created_at=_utc_now(),
    )
    rows = _read_list(root / "prediction_runs.json")
    rows = [row for row in rows if row.get("run_id") != run_id]
    rows.append(entry.to_dict())
    _write_json(root / "prediction_runs.json", rows, workspace_root=workspace)
    return entry


def repair_registry_paths(*, registry_root: str | Path) -> dict[str, Any]:
    """Rewrite internal absolute registry paths to portable token paths."""

    workspace = _workspace_from_registry_root(registry_root)
    root = registry_dir(workspace)
    changed_files: list[str] = []
    for name in ("models.json", "weights.json", "datasets.json", "prediction_runs.json"):
        path = root / name
        if not path.exists():
            continue
        current = _read_json_any(path)
        repaired = repair_portable_paths_in_payload(current, workspace_root=workspace)
        if repaired != current:
            backup = path.with_suffix(path.suffix + ".bak_portable_paths")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            _write_json(path, repaired, workspace_root=workspace)
            changed_files.append(str(path))

    dataset_root = root / "dataset_store"
    if dataset_root.exists():
        for path in sorted(dataset_root.rglob("*.json")):
            current = _read_json_any(path)
            if current in {None, ""}:
                continue
            repaired = repair_portable_paths_in_payload(current, workspace_root=workspace)
            if repaired != current:
                backup = path.with_suffix(path.suffix + ".bak_portable_paths")
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                _write_json(path, repaired, workspace_root=workspace)
                changed_files.append(str(path))
    return {
        "registry_root": str(workspace),
        "changed_files": changed_files,
        "changed_count": len(changed_files),
    }


def _read_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _model_entry_from_spec(model_id: str, spec: ModelSpec) -> ModelRegistryEntry:
    lower_model_id = model_id.lower()
    family = "LSDSAM" if "lsdsam" in lower_model_id else "LSDFormer" if "lsdformer" in lower_model_id else "Mock"
    return ModelRegistryEntry(
        model_id=model_id,
        net_name=spec.name,
        adapter=spec.adapter_name,
        model_family=family,
        input_mode=spec.input_mode,
        channel_rule=_channel_rule(spec),
        status=spec.status,
        description=spec.role,
        code_path=getattr(spec, "code_path", ""),
        model_options=dict(spec.options or {}) if getattr(spec, "options", None) else None,
    )


def _channel_rule(spec: ModelSpec) -> str:
    return "RGB + wavelets" if spec.input_mode == "rgb_wavelets" else "RGB"


def _read_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid registry JSON: {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RegistryError(f"Registry file must contain a list: {path}")
    return [dict(row) for row in data if isinstance(row, Mapping)]


def _write_json(path: Path, payload: Any, *, workspace_root: str | Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload, workspace_root=workspace_root)
    ready = repair_portable_paths_in_payload(ready, workspace_root=workspace_root)
    path.write_text(
        json.dumps(ready, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _store_weight_file(
    *,
    root: Path,
    source_path: Path,
    weight_id: str,
    stage: str,
    model_id: str,
    store_copy: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    if not store_copy:
        return source_path
    target_dir = root / "weight_repository" / stage / _path_segment(model_id) / weight_id
    target_dir.mkdir(parents=True, exist_ok=True)
    extension = source_path.suffix or ".pth"
    target_name = weight_id if weight_id.lower().endswith(extension.lower()) else f"{weight_id}{extension}"
    target = target_dir / target_name
    if target.resolve() != source_path.resolve():
        _copy_file_with_progress(source_path, target, progress_callback=progress_callback)
    return target.resolve()


def _copy_file_with_progress(
    source_path: Path,
    target_path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    total = int(source_path.stat().st_size)
    copied = 0
    progress_callback = progress_callback or (lambda _copied, _total: None)
    progress_callback(0, total)
    with source_path.open("rb") as source, target_path.open("wb") as target:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            target.write(chunk)
            copied += len(chunk)
            progress_callback(copied, total)
    shutil.copystat(source_path, target_path)
    progress_callback(total, total)


def _weight_file_size(path: Path) -> int | None:
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _weight_parameter_count(path: Path) -> int | None:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None

    state_dict = _extract_weight_state_dict(payload)
    if not isinstance(state_dict, Mapping):
        return None
    total = 0
    for value in state_dict.values():
        if hasattr(value, "numel"):
            try:
                total += int(value.numel())
            except Exception:
                continue
    return total or None


def _extract_weight_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    return None


def _path_segment(value: str) -> str:
    import re

    segment = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return segment or "unknown"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except (OSError, ValueError):
        return False
    return True


def _normalize_weight_id(value: str) -> str:
    text = _path_segment(_required_text(value, "weight_id"))
    if text in {".", ".."}:
        raise RegistryError("weight_id must contain letters, numbers, dots, underscores, or hyphens.")
    return text


def _slug_id(value: str, *, prefix: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    slug = slug or "dataset"
    return f"{prefix}-{slug}"


def _required_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise RegistryError(f"Missing required value: {field_name}.")
    return text


def _normalize_dataset_class(value: str | None) -> str:
    if value is None:
        raise RegistryError(f"Missing dataset class. Expected one of: {', '.join(DATASET_CLASSES)}.")
    aliases = {
        "pretraining": "TAFT",
        "baseline_evaluation": "DAFT",
        "taft": "TAFT",
        "daft": "DAFT",
        "tsft": "TSFT",
        "evaluation": "Evaluation",
        "eval": "Evaluation",
    }
    key = str(value).strip()
    normalized = aliases.get(key.lower(), key if key in DATASET_CLASSES else key.upper())
    if normalized not in DATASET_CLASSES:
        raise RegistryError(f"Unknown dataset class '{value}'. Expected one of: {', '.join(DATASET_CLASSES)}.")
    return normalized


def _normalize_weight_stage(value: str) -> str:
    key = str(value).strip()
    normalized = "inference-ready" if key.lower() in {"inference", "inference-ready", "ready"} else key.upper()
    if normalized not in WEIGHT_STAGES:
        raise RegistryError(f"Unknown weight stage '{value}'. Expected one of: {', '.join(WEIGHT_STAGES)}.")
    return normalized


def _normalize_rgb_bands(value: str) -> str:
    try:
        return normalize_rgb_bands(value)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    suffixes = {".tif", ".tiff", ".png"}
    if path.is_file():
        return int(path.suffix.lower() in suffixes)
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in suffixes)


def _count_split_images(path: Path, split: str) -> int:
    if not path.exists() or path.is_file():
        return 0
    suffixes = {".tif", ".tiff", ".png"}
    return sum(
        1
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in suffixes and split in file.parts
    )


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_user_path(value: str | Path) -> Path:
    """Resolve UI-entered paths with portable tokens and optional aliases."""

    return resolve_user_path(value)


def _workspace_from_registry_root(registry_root: str | Path) -> Path:
    return _portable_workspace_root(registry_root)


def _resolve_registry_path(value: Any, *, workspace_root: str | Path) -> Path:
    if value in {None, ""}:
        raise RegistryError("Missing registry path value.")
    try:
        return resolve_user_path(value, workspace_root=workspace_root)
    except (OSError, ValueError) as exc:
        raise RegistryError(f"Invalid registry path: {value}") from exc


def _daft_tiff_tile_schema(
    *,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: Path | None = None,
) -> str:
    ratio = 0.20 if max_nodata_ratio is None else float(max_nodata_ratio)
    return _append_slicing_aoi_tile_schema(
        f"tiff_slicing;tile_size={int(tile_size or 0)};stride={int(stride or 0)};max_nodata_ratio={ratio:.2f}",
        slicing_aoi_path=slicing_aoi_path,
    )


def _tsft_tiff_tile_schema(
    *,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    min_polygon_overlap_ratio: float | None,
    min_positive_pixel_ratio: float | None,
    no_feature_tile_policy: str | None,
    negative_ratio: float | None,
    slicing_aoi_path: Path | None = None,
) -> str:
    nodata = 0.20 if max_nodata_ratio is None else float(max_nodata_ratio)
    min_overlap = 0.0 if min_polygon_overlap_ratio is None else float(min_polygon_overlap_ratio)
    min_positive = 0.001 if min_positive_pixel_ratio is None else float(min_positive_pixel_ratio)
    neg_ratio = 1.0 if negative_ratio is None else float(negative_ratio)
    policy = str(no_feature_tile_policy or "balanced")
    schema = (
        f"tiff_slicing;tile_size={int(tile_size or 0)};stride={int(stride or 0)};"
        f"max_nodata_ratio={nodata:.2f};min_polygon_overlap_ratio={min_overlap:.3f};"
        f"min_positive_pixel_ratio={min_positive:.3f};no_feature_tile_policy={policy};negative_ratio={neg_ratio:.2f}"
    )
    return _append_slicing_aoi_tile_schema(schema, slicing_aoi_path=slicing_aoi_path)


def _append_slicing_aoi_tile_schema(schema: str, *, slicing_aoi_path: Path | None) -> str:
    if slicing_aoi_path is None:
        return schema
    return f"{schema};aoi=strict_inside"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_ready(value: Any, *, workspace_root: str | Path | None = None) -> Any:
    if isinstance(value, Path):
        return portable_path(value, workspace_root=workspace_root) or str(value)
    if isinstance(value, tuple):
        return [_json_ready(inner, workspace_root=workspace_root) for inner in value]
    if isinstance(value, list):
        return [_json_ready(inner, workspace_root=workspace_root) for inner in value]
    if isinstance(value, dict):
        return {key: _json_ready(inner, workspace_root=workspace_root) for key, inner in value.items()}
    return value
