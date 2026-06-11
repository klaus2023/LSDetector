"""Streamlit local operations console for LSDetector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import html
import io
import json
import csv
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsd_app.config import ConfigError, PlatformConfig, normalize_config as _normalize_platform_config
from lsd_app.core.channels import DEFAULT_RGB_BANDS, RGB_BAND_PRESETS, normalize_rgb_bands, rgb_bands_to_indexes
from lsd_app.core.paths import default_workspace_root, registry_dir, resolve_user_path
from lsd_app.core.inference import InferenceError, run_patch_inference
from lsd_app.core.metrics import EvaluationError, run_evaluate
from lsd_app.core.mosaic import MosaicError, run_mosaic
from lsd_app.core.raster import RasterError
from lsd_app.core.tiling import prepare_raster
from lsd_app.core.vectorize import VectorizeError, run_vectorize
from lsd_app.core.visualization import VisualizationError, run_visualize
from lsd_app.models import ModelAdapterError
from lsd_app.models.registry import get_adapter, get_model_spec, normalize_model_name
from lsd_app.registry import (
    DATASET_CLASSES,
    RegistryError,
    browse_dataset_repository,
    delete_weight,
    delete_dataset_store_entry,
    discover_taft_tree,
    inspect_daft_upload,
    list_datasets,
    list_dataset_repository_summary,
    list_models,
    list_weights,
    register_dataset,
    register_concat_dataset_entry,
    register_taft_tree,
    register_weight,
)
from lsd_app.data import (
    DAFT_UPLOAD_METHOD_PACKAGED,
    DAFT_UPLOAD_METHOD_TIFF_SLICING,
    DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
    DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
    DEFAULT_TSFT_NEGATIVE_RATIO,
    LABEL_POLARITIES,
    TSFT_NO_FEATURE_POLICIES,
    TSFT_NO_FEATURE_POLICY_ALL,
    TSFT_NO_FEATURE_POLICY_BALANCED,
    discover_candidate_tiff_rasters,
    is_candidate_primary_tiff,
)
from lsd_app.training.daft_lsdsam import (
    DaftTrainingError,
    _FocalDiceLoss,
    _LSDSAMSupervisedDataset,
    _collate_supervised,
    _evaluate_model,
    _require_module,
    resolve_daft_dataset_materialization,
    resolve_training_device,
)
from lsd_app.training.daft_lsdsam import (
    ALIGNMENT_MODES as DAFT_ALIGNMENT_MODES,
    DAFT_TRAINING_DEVICE_POLICY,
    DEFAULT_DAFT_NUM_WORKERS,
    DEFAULT_DAFT_GLOBAL_SEED,
    DaftLSDSAMConfig,
    build_daft_training_payload,
    parse_halve_epochs,
    run_lsdsam_daft,
)
from lsd_app.training.lsdformer import _LSDFormerSupervisedDataset
from lsd_app.training.tsft import (
    TSFT_TRAINING_DEVICE_POLICY,
    TsftLSDSAMConfig,
    TsftTrainingError,
    build_tsft_training_payload,
    finalize_tsft_checkpoint,
    run_lsdsam_tsft,
)
from lsd_app.web.jobs import (
    ACTIVE_STATUSES,
    _json_ready,
    clear_stale_job_blocker,
    list_job_states,
    read_job_state,
    request_job_cancel,
    stable_job_id,
    start_process_background_job,
    update_job_state,
)


STAGES = ("prepare", "infer", "mosaic", "vectorize", "evaluate", "visualize")
DEPLOYMENT_STAGES = ("prepare", "infer", "mosaic", "vectorize")
MODEL_CHOICES = ("LSDFormer", "LSDSAM-H", "LSDSAM-L", "LSDSAM-B")
LSDSAM_MODEL_CHOICES = ("LSDSAM-H", "LSDSAM-L", "LSDSAM-B")
WEIGHT_STAGES = ("TAFT", "DAFT", "TSFT", "inference-ready")
RGB_BANDS_CUSTOM_OPTION = "custom"
RGB_BANDS_UI_OPTIONS = (*RGB_BAND_PRESETS, RGB_BANDS_CUSTOM_OPTION)
SECTION_OPTIONS = (
    "Model Repository",
    "Weight Repository",
    "Dataset Repository",
    "DAFT Training",
    "TSFT Training",
    "Inference",
    "Evaluation",
)
APP_DISPLAY_NAME = "LSDetector"
WEB_ASSET_DIR = Path(__file__).resolve().parent / "assets"
SIDEBAR_LOGO_PATH = WEB_ASSET_DIR / "lsdetector-logo-sidebar-header.png"
MAIN_BACKGROUND_PATH = WEB_ASSET_DIR / "earth-dense-observation-satellites.png"
MODEL_REPOSITORY_HIDDEN_COLUMNS = frozenset(
    {
        "edge_mode",
        "adapter_name",
        "name",
        "model_options",
        "role",
        "input_mode",
        "adapter",
        "net_name",
    }
)
LSD_FORMER_DISPLAY_DESCRIPTION = "Lightweight design for fast deployment"
RUNNABLE_ERRORS = (
    ConfigError,
    RasterError,
    InferenceError,
    ModelAdapterError,
    MosaicError,
    VectorizeError,
    EvaluationError,
    VisualizationError,
    NotImplementedError,
)
TRAINING_RUNNABLE_ERRORS = (DaftTrainingError, TsftTrainingError, RegistryError, ValueError, RuntimeError)
ENABLE_STREAMLIT_FRAGMENTS = False
AUTO_REFRESH_INTERVAL_SECONDS = 1.5

StageRunner = Callable[[PlatformConfig], Any]
STAGE_RUNNERS: dict[str, StageRunner] = {
    "prepare": prepare_raster,
    "infer": run_patch_inference,
    "mosaic": run_mosaic,
    "vectorize": run_vectorize,
    "evaluate": run_evaluate,
    "visualize": run_visualize,
}


def _image_data_uri(path: Path) -> str:
    """Return a data URI for a local app image asset, or an empty string if missing."""

    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{payload}"


def _streamlit_fragment(st: Any) -> Any:
    """Return Streamlit fragment support when enabled.

    Streamlit 1.37 can leave stale fragment ids in an open browser tab after
    rapid navigation or app restarts. Keeping this disabled avoids user-facing
    "Could not find fragment" runtime errors while preserving the monitor UI.
    """

    if not ENABLE_STREAMLIT_FRAGMENTS:
        return None
    return getattr(st, "fragment", None)


def _schedule_active_job_refresh(
    st: Any,
    state: Mapping[str, Any] | None = None,
    *,
    registry_root: str | None = None,
    job_id: str | None = None,
    interval_seconds: float = AUTO_REFRESH_INTERVAL_SECONDS,
) -> None:
    """Force full-page refresh for active background jobs when fragments are disabled."""

    if _streamlit_fragment(st) is not None:
        return
    job_state: Mapping[str, Any] | None = state
    if job_state is None and registry_root and job_id:
        job_state = read_job_state(registry_root, job_id)
    if not job_state or not _job_is_active(job_state):
        return
    time.sleep(max(0.1, float(interval_seconds)))
    _request_streamlit_rerun(st)


@dataclass(frozen=True)
class StageStatus:
    """Serializable status for one UI-triggered stage run."""

    stage: str
    ok: bool
    message: str
    result: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    """User-facing validation result before launching a deployment run."""

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def build_raw_config(
    *,
    geotiff_path: str,
    output_dir: str,
    model: str,
    checkpoint_path: str = "",
    model_options: Mapping[str, Any] | None = None,
    tile_size: int = 512,
    stride: int = 256,
    batch_size: int = 1,
    threshold: float = 0.5,
    min_area_pixels: int = 0,
    device: str = "auto",
    truth_path: str = "",
    RGB_bands: str = DEFAULT_RGB_BANDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a raw platform config from Web GUI fields."""

    truth = _truth_config(truth_path)
    model_config: dict[str, Any] = {"name": model, "checkpoint_path": _blank_to_none(checkpoint_path)}
    if model_options:
        model_config["options"] = dict(model_options)
    return {
        "input": {"raster_path": geotiff_path, "RGB_bands": RGB_bands},
        "output": {"dir": output_dir, "vector_formats": ["gpkg", "geojson", "shapefile"]},
        "model": model_config,
        "tiling": {"tile_size": tile_size, "stride": stride},
        "inference": {"batch_size": batch_size, "threshold": threshold, "overwrite": overwrite},
        "postprocess": {"min_area_pixels": min_area_pixels, "simplify_tolerance": 0.0, "connectivity": 8},
        "vectorize": {"mode": "streaming"},
        "runtime": {"device": device, "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": truth,
    }


def build_evaluation_config(
    *,
    dataset: Mapping[str, Any],
    output_dir: str,
    model: str,
    checkpoint_path: str = "",
    batch_size: int = 1,
    threshold: float = 0.5,
    min_area_pixels: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build a legacy deployment-style evaluation config.

    Evaluation datasets in the registry are image/label sample sets; new UI
    runs use ``build_evaluation_dataset_payload`` instead.  This helper is kept
    for compatibility with older saved jobs/tests that evaluate a full raster
    prediction pipeline.
    """

    return build_raw_config(
        geotiff_path=str(dataset.get("imagery_path") or ""),
        output_dir=output_dir,
        model=model,
        checkpoint_path=checkpoint_path,
        tile_size=512,
        stride=256,
        batch_size=batch_size,
        threshold=threshold,
        min_area_pixels=min_area_pixels,
        device=device,
        truth_path=str(dataset.get("label_path") or dataset.get("truth_path") or ""),
        RGB_bands=str(dataset.get("RGB_bands") or DEFAULT_RGB_BANDS),
    )


def build_evaluation_dataset_payload(
    *,
    dataset_id: str,
    weight_id: str,
    model_id: str,
    output_dir: str,
    checkpoint_path: str = "",
    batch_size: int = 1,
    device: str = "auto",
) -> dict[str, Any]:
    """Build a payload for direct Evaluation images/labels sample evaluation."""

    return {
        "dataset_id": str(dataset_id or "").strip(),
        "weight_id": str(weight_id or "").strip(),
        "model_id": str(model_id or "").strip() or "LSDSAM-B",
        "checkpoint_path": str(checkpoint_path or "").strip(),
        "output_dir": str(output_dir or "outputs/evaluation_run").strip(),
        "batch_size": int(batch_size),
        "device": str(device or "auto").strip() or "auto",
    }


def _rgb_bands_input(
    st: Any,
    label: str,
    *,
    key: str,
    default: Any = DEFAULT_RGB_BANDS,
    help: str | None = None,
) -> str:
    default_value = _safe_rgb_bands(default)
    index = RGB_BANDS_UI_OPTIONS.index(default_value) if default_value in RGB_BANDS_UI_OPTIONS else len(RGB_BANDS_UI_OPTIONS) - 1
    selected = st.selectbox(label, RGB_BANDS_UI_OPTIONS, index=index, key=f"{key}_preset", help=help)
    if selected == RGB_BANDS_CUSTOM_OPTION:
        raw_value = st.text_input(f"{label} custom", value=default_value, key=key, help=help)
    else:
        raw_value = selected
    return normalize_rgb_bands(raw_value)


def _set_rgb_bands_preset_state(st: Any, key: str, value: Any) -> None:
    normalized = _safe_rgb_bands(value)
    preset_key = f"{key}_preset"
    if normalized in RGB_BANDS_UI_OPTIONS:
        st.session_state[preset_key] = normalized
        st.session_state.pop(key, None)
    else:
        st.session_state[preset_key] = RGB_BANDS_CUSTOM_OPTION
        st.session_state[key] = normalized


def _safe_rgb_bands(value: Any, fallback: Any = DEFAULT_RGB_BANDS) -> str:
    try:
        return normalize_rgb_bands(value)
    except ValueError:
        return normalize_rgb_bands(fallback)


def _evaluation_job_payload(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "dataset_id" in raw_config or "weight_id" in raw_config:
        return raw_config
    return {
        "dataset_id": _nested_value(raw_config, ("dataset", "dataset_id"), ("dataset_id",), default=""),
        "weight_id": _nested_value(raw_config, ("weight", "weight_id"), ("weight_id",), default=""),
        "model_id": _nested_value(raw_config, ("model", "name"), ("model_id",), ("model",), default="LSDSAM-B"),
        "checkpoint_path": _nested_value(raw_config, ("model", "checkpoint_path"), ("checkpoint_path",), default=""),
        "output_dir": _nested_value(raw_config, ("output", "dir"), ("output_dir",), default="outputs/evaluation_run"),
        "batch_size": _nested_value(raw_config, ("inference", "batch_size"), ("batch_size",), default=1),
        "device": _nested_value(raw_config, ("runtime", "device"), ("device",), default="auto"),
    }


def _evaluation_weight_row(*, registry_root: str, weight_id: str) -> dict[str, Any]:
    normalized_weight_id = str(weight_id or "").strip()
    if not normalized_weight_id:
        return {}
    try:
        weights = list_weights(registry_root=registry_root)
    except RegistryError as exc:
        raise EvaluationError(f"Could not read registered weight metadata for {normalized_weight_id}: {exc}") from exc
    return dict(next((row for row in weights if str(row.get("weight_id") or "") == normalized_weight_id), {}))


def _evaluation_model_id_from_weight(model_id: str, weight_row: Mapping[str, Any]) -> str:
    selected_model = _canonical_training_model_id(model_id) or str(model_id or "").strip() or "LSDSAM-B"
    weight_model = _canonical_training_model_id(weight_row.get("model_id")) if weight_row else ""
    return weight_model or selected_model


def _evaluation_adapter_options(model_id: str, weight_row: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    try:
        model_spec = get_model_spec(model_id)
    except ValueError:
        model_spec = None
    if model_spec is not None and model_spec.options:
        options.update(dict(model_spec.options))

    for key in (
        "weight_type",
        "image_size",
        "lora_base_r",
        "lora_base_alpha",
        "lora_task_r",
        "lora_task_alpha",
        "lora_dropout",
        "peft_adapter_name",
        "target_modules",
        "adapter_profile",
    ):
        value = weight_row.get(key) if weight_row else None
        if value is not None and str(value).strip() != "":
            options[key] = value

    base_checkpoint = weight_row.get("base_checkpoint_path") if weight_row else None
    if "sam_checkpoint_path" not in options and base_checkpoint is not None and str(base_checkpoint).strip() != "":
        options["sam_checkpoint_path"] = base_checkpoint
    return options


def build_training_dry_run(
    *,
    stage: str,
    model_id: str,
    weight_id: str,
    dataset_id: str,
    output_dir: str,
) -> dict[str, str]:
    """Return a serializable placeholder plan for DAFT/TSFT UI panels."""

    return {
        "stage": stage,
        "model_id": model_id,
        "weight_id": weight_id,
        "dataset_id": dataset_id,
        "output_dir": output_dir,
        "status": "reserved-dry-run-no-training-launched",
    }


def _training_dataset_rows_for_class(
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_class: str,
    *,
    require_gt: bool = False,
) -> list[Mapping[str, Any]]:
    selected_class = str(dataset_class or "").strip()
    return [
        row
        for row in dataset_rows
        if str(row.get("dataset_class") or row.get("dataset_type") or "").strip() == selected_class
        and (not require_gt or bool(row.get("has_gt")))
    ]


def _dataset_id_option_label(dataset_id: Any) -> str:
    value = str(dataset_id or "")
    lowered = value.lower()
    for dataset_class in ("TAFT", "DAFT", "TSFT"):
        prefix = f"{dataset_class.lower()}-"
        if lowered.startswith(prefix):
            return f"{dataset_class}-{value[len(prefix):]}"
    return value


def _canonical_training_model_id(model_id: Any) -> str:
    value = str(model_id or "").strip()
    if not value:
        return ""
    try:
        normalized = normalize_model_name(value)
    except ValueError:
        normalized = value
    return "LSDSAM-B" if normalized == "lsdsam-fewshot" else normalized


def _training_job_title(job_or_payload: Mapping[str, Any], *, stage: str, dataset_key: str, fallback: str) -> str:
    payload = job_or_payload.get("payload") if isinstance(job_or_payload.get("payload"), Mapping) else job_or_payload
    model_id = _canonical_training_model_id(payload.get("model_id")) or str(payload.get("model_id") or "").strip()
    dataset_id = str(payload.get(dataset_key) or "").strip() or fallback
    if model_id:
        return f"{stage} {model_id} {dataset_id}"
    return f"{stage} {dataset_id}"


def _training_model_widget_key(model_id: Any) -> str:
    normalized = _canonical_training_model_id(model_id) or "unknown"
    return normalized.replace(" ", "_").replace("-", "_")


def _training_weight_rows_for_model(
    weight_rows: Sequence[Mapping[str, Any]],
    model_id: Any,
) -> list[Mapping[str, Any]]:
    selected_model = _canonical_training_model_id(model_id)
    if not selected_model:
        return []
    return [
        row
        for row in weight_rows
        if row.get("exists", True)
        and _canonical_training_model_id(row.get("model_id")) == selected_model
    ]


def _weight_id_options(weight_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("weight_id") or "").strip() for row in weight_rows if str(row.get("weight_id") or "").strip()]


def _compact_label_part(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _append_unique_label_part(parts: list[str], value: Any) -> None:
    part = _compact_label_part(value)
    if part and part not in parts:
        parts.append(part)


def _weight_note_value(notes: str, key: str) -> str:
    match = re.search(rf"\b{re.escape(key)}=([^,.;)]+)", notes)
    return _compact_label_part(match.group(1)) if match else ""


def _timestamp_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _weight_option_detail_parts(row: Mapping[str, Any], *, weight_id: str) -> list[str]:
    parts: list[str] = []
    stage = _compact_label_part(row.get("stage"))
    version = _compact_label_part(row.get("version"))
    created_at = _compact_label_part(row.get("created_at"))
    notes = _compact_label_part(row.get("notes"))

    _append_unique_label_part(parts, stage)
    if version and version != weight_id:
        _append_unique_label_part(parts, version)

    for field, label in (
        ("source_dataset_id", "source"),
        ("target_dataset_id", "target"),
        ("dataset_id", "dataset"),
    ):
        value = _compact_label_part(row.get(field))
        if value:
            _append_unique_label_part(parts, f"{label}={value}")

    for key in ("source", "target", "dataset"):
        value = _weight_note_value(notes, key)
        if value:
            _append_unique_label_part(parts, f"{key}={value}")

    created_digits = _timestamp_digits(created_at)
    detail_digits = _timestamp_digits(" ".join(parts))
    if created_at and (not created_digits or created_digits not in detail_digits):
        _append_unique_label_part(parts, f"created={created_at}")

    return parts


def _weight_option_labels(weight_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in weight_rows:
        weight_id = str(row.get("weight_id") or "").strip()
        if not weight_id:
            continue
        detail = " | ".join(_weight_option_detail_parts(row, weight_id=weight_id))
        labels[weight_id] = f"{weight_id} ({detail})" if detail else weight_id
    return labels


def _weight_format_func(weight_rows: Sequence[Mapping[str, Any]]) -> Callable[[str], str]:
    labels = _weight_option_labels(weight_rows)
    return lambda weight_id: labels.get(str(weight_id), str(weight_id))


def daft_training_job_id(payload: Mapping[str, Any]) -> str:
    """Build a stable DAFT training job id from a UI payload."""

    target = str(payload.get("target_dataset_id") or "target")
    if payload.get("resume") and payload.get("resume_run_id"):
        target = f"{target}-resume-{payload.get('resume_run_id')}"
    return stable_job_id(kind="daft-training", version=_safe_job_version(target), payload=payload)


def _daft_resume_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _daft_config_from_payload(*, registry_root: str, payload: Mapping[str, Any]) -> DaftLSDSAMConfig:
    return DaftLSDSAMConfig(
        registry_root=registry_root,
        model_id=str(payload.get("model_id") or "LSDSAM-B"),
        base_weight_id=str(payload.get("base_weight_id") or ""),
        source_dataset_id=str(payload.get("source_dataset_id") or ""),
        target_dataset_id=str(payload.get("target_dataset_id") or ""),
        alignment_mode=str(payload.get("alignment_mode") or "IST"),
        total_epochs=int(payload.get("total_epochs") or 10),
        initial_lr=float(payload.get("initial_lr") or 2e-4),
        halve_epochs=tuple(int(value) for value in payload.get("halve_epochs", [])),
        device="auto",
        output_dir=str(payload.get("output_dir") or "outputs/daft_training"),
        evaluation_dataset_id=str(payload.get("evaluation_dataset_id") or ""),
        batch_size=int(payload.get("batch_size") or 2),
        num_workers=int(payload.get("num_workers", DEFAULT_DAFT_NUM_WORKERS)),
        seed=int(payload.get("global_seed", payload.get("seed", DEFAULT_DAFT_GLOBAL_SEED))),
        adv_lambda=float(payload.get("adv_lambda") or 0.05),
        align_lambda=float(payload.get("align_lambda") or 0.005),
        lora_task_r=int(payload.get("lora_task_r") or 16),
        lora_task_alpha=int(payload.get("lora_task_alpha") or 32),
        lora_dropout=float(payload.get("lora_dropout") if payload.get("lora_dropout") is not None else 0.05),
        output_weight_id=str(payload.get("output_weight_id") or ""),
        resume=bool(payload.get("resume")),
        resume_checkpoint_path=str(payload.get("resume_checkpoint_path") or ""),
        save_optimizer_state=bool(payload.get("save_optimizer_state")),
        pin_memory=bool(payload.get("pin_memory")),
        deterministic_cuda=bool(payload.get("deterministic_cuda")),
    )


def start_daft_training_background_job(*, registry_root: str, payload: Mapping[str, Any]) -> str:
    """Start an LSDSAM DAFT background job."""

    job_id = daft_training_job_id(payload)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        config = _daft_config_from_payload(registry_root=registry_root, payload=payload)
        return run_lsdsam_daft(config, progress_callback=progress_callback).to_dict()

    start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="daft-training",
        description=_training_job_title(payload, stage="DAFT", dataset_key="target_dataset_id", fallback="target"),
        payload=payload,
        target=target,
    )
    return job_id


def tsft_training_job_id(payload: Mapping[str, Any]) -> str:
    """Build a stable TSFT training job id from a UI payload."""

    dataset = str(payload.get("dataset_id") or "dataset")
    if payload.get("resume") and payload.get("resume_run_id"):
        dataset = f"{dataset}-resume-{payload.get('resume_run_id')}"
    return stable_job_id(kind="tsft-training", version=_safe_job_version(dataset), payload=payload)




def _tsft_resume_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _job_is_active(job: Mapping[str, Any]) -> bool:
    status = str(job.get("status") or "")
    return status in {"pending", "running", "cancelling", "stalled", "stalled_uninterruptible"}


def _job_is_live_blocking(job: Mapping[str, Any]) -> bool:
    status = str(job.get("status") or "")
    return status in {"pending", "running", "cancelling"}


def _job_monitor_priority(job: Mapping[str, Any]) -> tuple[int, str]:
    status = str(job.get("status") or "")
    if status in {"running", "pending", "cancelling"}:
        priority = 0
    elif status == "resource_blocked":
        priority = 1
    elif status in {"stalled", "stalled_uninterruptible"}:
        priority = 2
    else:
        priority = 3
    return (priority, _job_recency_age_seconds(job))


def _job_latest_priority(job: Mapping[str, Any]) -> tuple[int, float]:
    status = str(job.get("status") or "")
    if status in {"running", "pending", "cancelling", "completed"}:
        priority = 0
    elif status in {"failed", "cancelled"}:
        priority = 1
    elif status == "resource_blocked":
        priority = 2
    elif status in {"interrupted", "stalled", "stalled_uninterruptible"}:
        priority = 3
    else:
        priority = 4
    return (priority, _job_recency_age_seconds(job))


def _job_recency_age_seconds(job: Mapping[str, Any]) -> float:
    for key in ("updated_at", "finished_at", "created_at"):
        raw = str(job.get(key) or "").strip()
        if not raw:
            continue
        try:
            updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
    return float("inf")


def _latest_job_for_kind(registry_root: str, kind: str) -> Mapping[str, Any] | None:
    jobs = list_job_states(registry_root, kind=kind)
    if not jobs:
        return None
    return sorted(jobs, key=_job_latest_priority)[0]


def _active_jobs_for_kind(registry_root: str, kind: str) -> list[Mapping[str, Any]]:
    return [job for job in list_job_states(registry_root, kind=kind) if _job_is_active(job)]


def _active_job_for_kind(registry_root: str, kind: str) -> Mapping[str, Any] | None:
    jobs = _active_jobs_for_kind(registry_root, kind)
    if not jobs:
        return None
    return sorted(jobs, key=_job_monitor_priority)[0]


def _hidden_job_monitor_key(session_key: str) -> str:
    return f"{session_key}_hidden_job_ids"


def _hidden_job_monitor_ids(st: Any, session_key: str) -> set[str]:
    value = st.session_state.get(_hidden_job_monitor_key(session_key))
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    return set()


def _job_monitor_id_is_hidden(st: Any, *, session_key: str, job_id: str, state: Mapping[str, Any] | None = None) -> bool:
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id or normalized_job_id not in _hidden_job_monitor_ids(st, session_key):
        return False
    return not (state and _job_is_live_blocking(state))


def _job_monitor_is_hidden(st: Any, *, session_key: str, job: Mapping[str, Any]) -> bool:
    return _job_monitor_id_is_hidden(
        st,
        session_key=session_key,
        job_id=str(job.get("job_id") or ""),
        state=job,
    )


def _remember_hidden_job_monitor_id(st: Any, session_key: str, job_id: str) -> bool:
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        return False
    hidden_ids = _hidden_job_monitor_ids(st, session_key)
    is_new = normalized_job_id not in hidden_ids
    hidden_ids.add(normalized_job_id)
    st.session_state[_hidden_job_monitor_key(session_key)] = sorted(hidden_ids)
    return is_new


def _clear_frontend_job_monitor_history(
    st: Any,
    *,
    registry_root: str,
    session_keys: Sequence[str],
    job_kinds: Sequence[tuple[str, str]] = (),
) -> int:
    cleared = 0
    for session_key in session_keys:
        job_id = str(st.session_state.get(session_key) or "").strip()
        if not job_id:
            st.session_state.pop(session_key, None)
            continue
        state = read_job_state(registry_root, job_id)
        if state and _job_is_live_blocking(state):
            continue
        if _remember_hidden_job_monitor_id(st, session_key, job_id):
            cleared += 1
        st.session_state.pop(session_key, None)
    for session_key, kind in job_kinds:
        for job in list_job_states(registry_root, kind=kind):
            if _job_is_live_blocking(job):
                continue
            job_id = str(job.get("job_id") or "").strip()
            if _remember_hidden_job_monitor_id(st, session_key, job_id):
                cleared += 1
            if job_id and str(st.session_state.get(session_key) or "").strip() == job_id:
                st.session_state.pop(session_key, None)
    return cleared


def _render_clear_frontend_job_history_button(
    st: Any,
    *,
    registry_root: str,
    session_keys: Sequence[str],
    job_kinds: Sequence[tuple[str, str]] = (),
    label: str,
    button_key: str,
) -> None:
    st.divider()
    st.caption("Only clears this page's previous job monitor display; job records, outputs, and running jobs are kept.")
    if st.button(label, key=button_key):
        cleared = _clear_frontend_job_monitor_history(
            st,
            registry_root=registry_root,
            session_keys=session_keys,
            job_kinds=job_kinds,
        )
        if cleared:
            st.success(f"Cleared {cleared} previous job display item(s).")
            st.rerun()
        else:
            st.info("No previous terminal job display to clear.")


def _recover_active_job_id(
    st: Any,
    *,
    registry_root: str,
    kind: str,
    session_key: str,
    job_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    include_latest_terminal: bool = True,
) -> str | None:
    jobs = list_job_states(registry_root, kind=kind)
    if job_filter is not None:
        jobs = [job for job in jobs if job_filter(job)]
    jobs = [job for job in jobs if not _job_monitor_is_hidden(st, session_key=session_key, job=job)]

    active_jobs = [job for job in jobs if _job_is_active(job)]
    if active_jobs:
        active = sorted(active_jobs, key=_job_monitor_priority)[0]
        job_id = str(active.get("job_id") or "")
        if job_id:
            st.session_state[session_key] = job_id
            return job_id

    latest = sorted(jobs, key=_job_latest_priority)[0] if jobs else None
    remembered = str(st.session_state.get(session_key) or "").strip()
    if remembered:
        state = read_job_state(registry_root, remembered)
        if state and _job_monitor_id_is_hidden(st, session_key=session_key, job_id=remembered, state=state):
            st.session_state.pop(session_key, None)
            state = None
        if state and (job_filter is None or job_filter(state)):
            related = _latest_related_job(state, jobs)
            if related:
                related_job_id = str(related.get("job_id") or "")
                if related_job_id:
                    st.session_state[session_key] = related_job_id
                    return related_job_id
            if latest:
                remembered_updated = str(state.get("updated_at") or "")
                latest_updated = str(latest.get("updated_at") or "")
                if latest_updated > remembered_updated:
                    latest_status = str(latest.get("status") or "")
                    latest_job_id = str(latest.get("job_id") or "")
                    if latest_job_id and latest_status in {"interrupted", "cancelled", "failed", "completed", "resource_blocked"}:
                        st.session_state[session_key] = latest_job_id
                        return latest_job_id
            return remembered
        st.session_state.pop(session_key, None)
    if include_latest_terminal and latest and str(latest.get("status") or "") in {"interrupted", "cancelled", "failed", "completed"}:
        job_id = str(latest.get("job_id") or "")
        if job_id:
            st.session_state[session_key] = job_id
            return job_id
    return None


def _latest_related_job(reference: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    reference_id = str(reference.get("job_id") or "")
    candidates = [
        job
        for job in jobs
        if str(job.get("job_id") or "") != reference_id
        and str(job.get("status") or "") in {"pending", "running", "cancelling", "completed", "failed", "cancelled", "interrupted"}
        and _jobs_are_related(reference, job)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_job_latest_priority)[0]


def _jobs_are_related(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    ref_payload = reference.get("payload") if isinstance(reference.get("payload"), Mapping) else {}
    cand_payload = candidate.get("payload") if isinstance(candidate.get("payload"), Mapping) else {}
    if not ref_payload or not cand_payload:
        return False

    resume_path = str(cand_payload.get("resume_checkpoint_path") or "").strip()
    if resume_path:
        known_paths = {
            str(reference.get("latest_checkpoint_path") or "").strip(),
            str(reference.get("best_checkpoint_path") or "").strip(),
        }
        if resume_path in known_paths:
            return True

    keys = ("model_id", "base_weight_id", "source_dataset_id", "target_dataset_id", "output_dir")
    compared = [
        key
        for key in keys
        if str(ref_payload.get(key) or "").strip() and str(cand_payload.get(key) or "").strip()
    ]
    matches = [
        key
        for key in compared
        if str(ref_payload.get(key) or "").strip() == str(cand_payload.get(key) or "").strip()
    ]
    if "output_dir" in matches and any(key in matches for key in ("target_dataset_id", "source_dataset_id")):
        return True
    return len(matches) >= 3


def _job_updated_age_seconds(job: Mapping[str, Any]) -> float | None:
    raw = str(job.get("updated_at") or "").strip()
    if not raw:
        return None
    try:
        updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())


def _job_stale_caption(job: Mapping[str, Any]) -> str:
    parts: list[str] = []
    heartbeat_age = _timestamp_age_seconds(str(job.get("heartbeat_at") or ""))
    progress_age = _timestamp_age_seconds(str(job.get("last_progress_at") or ""))
    update_age = _job_updated_age_seconds(job)
    if heartbeat_age is not None:
        parts.append(f"heartbeat {_format_age(heartbeat_age)} ago")
    elif update_age is not None:
        parts.append(f"last update {_format_age(update_age)} ago")
    if progress_age is not None:
        parts.append(f"progress {_format_age(progress_age)} ago")
    lock_owner = str(job.get("device_lock_owner_job_id") or "").strip()
    lock_pid = str(job.get("device_lock_owner_pid") or "").strip()
    lock_id = str(job.get("device_lock_id") or "").strip()
    if lock_id:
        owner_text = lock_owner or "this job"
        parts.append(f"lock {lock_id} owner {owner_text}" + (f" pid {lock_pid}" if lock_pid else ""))
    return " | ".join(parts)


def _timestamp_age_seconds(raw: str) -> float | None:
    if not raw:
        return None
    try:
        updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())


def _format_age(age_seconds: float) -> str:
    if age_seconds < 120:
        return f"{age_seconds:.0f}s"
    return f"{age_seconds / 60:.1f} min"


def _render_job_health_notice(st: Any, job: Mapping[str, Any]) -> None:
    status = str(job.get("status") or "")
    reason = str(job.get("stall_reason") or job.get("message") or "").strip()
    if status == "stalled":
        st.warning(reason or "Task heartbeat has stalled. Stop it, then resume from the latest checkpoint/cache if available.")
    elif status == "stalled_uninterruptible":
        st.error(reason or "Worker is in uninterruptible kernel sleep; restart WSL/CUDA before launching another CUDA job.")
    elif status == "resource_blocked":
        st.error(reason or "CUDA device is blocked by another active job.")


def _render_stale_job_cleanup_button(
    st: Any,
    *,
    registry_root: str,
    job_id: str,
    state: Mapping[str, Any],
    session_key: str,
    label: str,
) -> None:
    status = str(state.get("status") or "")
    if status not in {"stalled", "stalled_uninterruptible", "resource_blocked", "cancelling"}:
        return
    st.caption(
        "Use this only after WSL/CUDA was restarted or you have confirmed the old worker is gone. "
        "It clears LSDetector state; it does not kill a Linux process."
    )
    if st.button(label, type="secondary", key=f"clear_stale_{job_id}"):
        updated = clear_stale_job_blocker(registry_root, job_id)
        if updated:
            st.session_state.pop(session_key, None)
            st.success("Cleared stale LSDetector blocker. You can launch a new run or resume from checkpoint.")
            st.rerun()
        else:
            st.warning("No stale blocker was cleared; refresh the page and check the current job state.")


def _training_resume_paths_from_state(
    state: Mapping[str, Any],
    *,
    default_output_dir: str,
    latest_names: Sequence[str],
    best_names: Sequence[str],
) -> tuple[str, str, str]:
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    output_dir = Path(str(payload.get("output_dir") or default_output_dir)).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    latest = str(state.get("latest_checkpoint_path") or "").strip()
    best = str(state.get("best_checkpoint_path") or "").strip()
    if not latest:
        latest = next((str(checkpoints_dir / name) for name in latest_names if (checkpoints_dir / name).exists()), str(checkpoints_dir / latest_names[0]))
    if not best:
        best = next((str(checkpoints_dir / name) for name in best_names if (checkpoints_dir / name).exists()), str(checkpoints_dir / best_names[0]))
    config_path = str(state.get("training_config_path") or "").strip() or str(output_dir / "training_config_latest.json")
    return latest, best, config_path


def _tsft_config_from_payload(*, registry_root: str, payload: Mapping[str, Any]) -> TsftLSDSAMConfig:
    max_epoch = int(payload.get("max_epoch") or payload.get("total_epochs") or 10)
    max_lr = float(payload.get("max_lr") or payload.get("initial_lr") or 2e-4)
    min_lr = float(payload.get("min_lr") if payload.get("min_lr") is not None else 1e-6)
    max_decay_epoch = int(payload.get("max_decay_epoch") or max_epoch)
    return TsftLSDSAMConfig(
        registry_root=registry_root,
        model_id=str(payload.get("model_id") or "LSDSAM-B"),
        base_weight_id=str(payload.get("base_weight_id") or ""),
        dataset_id=str(payload.get("dataset_id") or ""),
        evaluation_dataset_id=str(payload.get("evaluation_dataset_id") or ""),
        max_epoch=max_epoch,
        max_lr=max_lr,
        min_lr=min_lr,
        max_decay_epoch=max_decay_epoch,
        device="auto",
        output_dir=str(payload.get("output_dir") or "outputs/tsft_training"),
        batch_size=int(payload.get("batch_size") or 2),
        num_workers=int(payload.get("num_workers", DEFAULT_DAFT_NUM_WORKERS)),
        lora_task_r=int(payload.get("lora_task_r") or 16),
        lora_task_alpha=int(payload.get("lora_task_alpha") or 32),
        lora_dropout=float(payload.get("lora_dropout") if payload.get("lora_dropout") is not None else 0.05),
        output_weight_id=str(payload.get("output_weight_id") or ""),
        resume=bool(payload.get("resume")),
        resume_checkpoint_path=str(payload.get("resume_checkpoint_path") or ""),
        cleanup_lora_checkpoints_after_register=payload.get("cleanup_lora_checkpoints_after_register", True) is not False,
        save_optimizer_state=bool(payload.get("save_optimizer_state")),
    )


def start_tsft_training_background_job(*, registry_root: str, payload: Mapping[str, Any]) -> str:
    """Start an LSDSAM TSFT background job."""

    job_id = tsft_training_job_id(payload)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        config = _tsft_config_from_payload(registry_root=registry_root, payload=payload)
        return run_lsdsam_tsft(config, progress_callback=progress_callback).to_dict()

    start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="tsft-training",
        description=_training_job_title(payload, stage="TSFT", dataset_key="dataset_id", fallback="dataset"),
        payload=payload,
        target=target,
    )
    return job_id


def _format_progress_percent(percent: float) -> str:
    return f"{percent * 100:.2f}%"


def _taft_import_progress_html(progress: Mapping[str, Any]) -> str:
    percent = min(max(float(progress.get("percent") or 0.0), 0.0), 1.0)
    status = str(progress.get("status") or "pending").replace("-", " ").title()
    processed = int(progress.get("processed") or 0)
    total = int(progress.get("total") or 0)
    written = int(progress.get("written") or 0)
    skipped = int(progress.get("skipped") or 0)
    rewritten = int(progress.get("rewritten") or 0)
    current_action = str(progress.get("current_action") or "").strip()
    current_reason = str(progress.get("current_reason") or "").strip()
    message = str(progress.get("message") or "Waiting for import to start.")
    total_text = str(total) if total else "discovering"
    detail = f"{message} | {current_action}: {current_reason}" if current_action else message
    return f"""
    <div class="taft-import-panel">
      <div class="taft-import-head">
        <strong>{status}</strong>
        <span>{_format_progress_percent(percent)}</span>
      </div>
      <div class="taft-progress-track" aria-label="TAFT import progress">
        <div class="taft-progress-fill" style="width: {_format_progress_percent(percent)};"></div>
      </div>
      <div class="taft-import-stats">
        <span>processed {processed}/{total_text}</span>
        <span>written {written}</span>
        <span>skipped {skipped}</span>
        <span>rewritten {rewritten}</span>
      </div>
      <div class="taft-import-message">{detail}</div>
    </div>
    """


def normalize_config(raw_config: Mapping[str, Any], *, create_output_dirs: bool = True) -> PlatformConfig:
    """Normalize GUI-created config with the shared platform validator."""

    return _normalize_platform_config(raw_config, create_output_dirs=create_output_dirs)


def normalize_web_config(raw_config: Mapping[str, Any], *, create_output_dirs: bool = True) -> PlatformConfig:
    """Backward-friendly alias for the Web GUI normalizer."""

    return normalize_config(raw_config, create_output_dirs=create_output_dirs)


def _with_registry_root(raw_config: Mapping[str, Any], registry_root: str | Path) -> dict[str, Any]:
    """Carry the selected workspace into persisted worker payloads."""

    raw = json.loads(json.dumps(_json_ready(dict(raw_config))))
    raw["registry_root"] = str(Path(registry_root).expanduser().resolve())
    return raw


def run_stage(stage: str, config: PlatformConfig) -> StageStatus:
    """Run one pipeline stage and capture expected user-facing failures."""

    if stage not in STAGE_RUNNERS:
        available = ", ".join(STAGES)
        raise ValueError(f"Unknown stage '{stage}'. Expected one of: {available}.")

    try:
        result = STAGE_RUNNERS[stage](config)
    except RUNNABLE_ERRORS as exc:
        return StageStatus(
            stage=stage,
            ok=False,
            message=str(exc),
            artifacts=collect_artifacts(config),
            error_type=type(exc).__name__,
        )

    return StageStatus(
        stage=stage,
        ok=True,
        message="Completed.",
        result=_result_to_dict(result),
        artifacts=collect_artifacts(config),
    )


def run_stage_with_progress(
    stage: str,
    config: PlatformConfig,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> StageStatus:
    """Run one stage and pass progress callbacks to stage implementations that support it."""

    if stage not in STAGE_RUNNERS:
        available = ", ".join(STAGES)
        raise ValueError(f"Unknown stage '{stage}'. Expected one of: {available}.")

    runner = STAGE_RUNNERS[stage]
    try:
        if stage in {"prepare", "infer", "mosaic", "vectorize"}:
            result = runner(config, progress_callback=progress_callback)
        else:
            result = runner(config)
    except RUNNABLE_ERRORS as exc:
        return StageStatus(
            stage=stage,
            ok=False,
            message=str(exc),
            artifacts=collect_artifacts(config),
            error_type=type(exc).__name__,
        )

    return StageStatus(
        stage=stage,
        ok=True,
        message="Completed.",
        result=_result_to_dict(result),
        artifacts=collect_artifacts(config),
    )


def run_all(config: PlatformConfig) -> list[StageStatus]:
    """Run the deployment pipeline in order, stopping at the first failed stage."""

    statuses: list[StageStatus] = []
    for stage in DEPLOYMENT_STAGES:
        status = run_stage(stage, config)
        statuses.append(status)
        if not status.ok:
            break
    return statuses


def run_all_with_progress(
    config: PlatformConfig,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[StageStatus]:
    """Run the deployment pipeline and emit weighted stage progress."""

    statuses: list[StageStatus] = []
    stage_count = len(DEPLOYMENT_STAGES)
    for index, stage in enumerate(DEPLOYMENT_STAGES):
        stage_base = index / stage_count
        stage_weight = 1.0 / stage_count
        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            phase=stage,
            progress_kind=stage,
            status="running",
            percent=0.0,
            overall_percent=stage_base,
            message=f"Starting {stage}.",
        )

        def stage_progress(progress: Mapping[str, Any], *, stage_base: float = stage_base, stage_weight: float = stage_weight) -> None:
            stage_percent = min(max(float(progress.get("percent") or 0.0), 0.0), 1.0)
            _emit_deployment_progress(
                progress_callback,
                phase=progress.get("phase") or progress.get("stage") or stage,
                progress_kind=progress.get("progress_kind") or stage,
                **dict(progress),
                overall_percent=stage_base + stage_percent * stage_weight,
            )

        status = run_stage_with_progress(stage, config, progress_callback=stage_progress)
        statuses.append(status)
        if not status.ok:
            _emit_deployment_progress(
                progress_callback,
                stage=stage,
                status="failed",
                percent=1.0,
                overall_percent=stage_base + stage_weight,
                message=status.message,
                error_type=status.error_type,
            )
            break
        complete_payload = _deployment_stage_completion_payload(status)
        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            status="completed",
            percent=1.0,
            overall_percent=stage_base + stage_weight,
            message=f"Completed {stage}.",
            **complete_payload,
        )
    return statuses


def run_all_resumable_with_progress(
    config: PlatformConfig,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[StageStatus]:
    """Run deployment and reuse completed stage outputs from a matching output cache."""

    statuses: list[StageStatus] = []
    stage_count = len(DEPLOYMENT_STAGES)
    for index, stage in enumerate(DEPLOYMENT_STAGES):
        stage_base = index / stage_count
        stage_weight = 1.0 / stage_count
        if _deployment_stage_is_complete(stage, config):
            _emit_deployment_progress(
                progress_callback,
                stage=stage,
                status="skipped",
                percent=1.0,
                overall_percent=stage_base + stage_weight,
                current_action="using cached output",
                message=f"{stage} already has valid outputs for this configuration.",
            )
            statuses.append(
                StageStatus(
                    stage=stage,
                    ok=True,
                    message="Reused existing outputs.",
                    artifacts=collect_artifacts(config),
                )
            )
            continue

        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            phase=stage,
            progress_kind=stage,
            status="running",
            percent=0.0,
            overall_percent=stage_base,
            message=f"Starting {stage}.",
        )

        def stage_progress(progress: Mapping[str, Any], *, stage_base: float = stage_base, stage_weight: float = stage_weight) -> None:
            stage_percent = min(max(float(progress.get("percent") or 0.0), 0.0), 1.0)
            _emit_deployment_progress(
                progress_callback,
                phase=progress.get("phase") or progress.get("stage") or stage,
                progress_kind=progress.get("progress_kind") or stage,
                **dict(progress),
                overall_percent=stage_base + stage_percent * stage_weight,
            )

        status = run_stage_with_progress(stage, config, progress_callback=stage_progress)
        statuses.append(status)
        if not status.ok:
            _emit_deployment_progress(
                progress_callback,
                stage=stage,
                status="failed",
                percent=1.0,
                overall_percent=stage_base + stage_weight,
                message=status.message,
                error_type=status.error_type,
            )
            break
        complete_payload = _deployment_stage_completion_payload(status)
        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            status="completed",
            percent=1.0,
            overall_percent=stage_base + stage_weight,
            message=f"Completed {stage}.",
            **complete_payload,
        )
    return statuses



def _deployment_stage_completion_payload(status: StageStatus) -> dict[str, Any]:
    if status.stage != "infer" or not isinstance(status.result, Mapping):
        return {}
    result = status.result
    payload: dict[str, Any] = {}
    for source, target in (
        ("duration_seconds", "inference_duration_seconds"),
        ("patches_per_second", "inference_patches_per_second"),
        ("predictions_per_second", "inference_predictions_per_second"),
        ("patch_count", "inference_patch_count"),
        ("prediction_count", "inference_prediction_count"),
    ):
        if source in result:
            payload[target] = result[source]
    return payload

def preflight_deployment_config(
    raw_config: Mapping[str, Any],
    *,
    selected_weight: Mapping[str, Any] | None = None,
    create_output_dirs: bool = False,
) -> tuple[PlatformConfig | None, PreflightResult]:
    """Validate the one-click deployment inputs before heavy processing starts."""

    errors: list[str] = []
    warnings: list[str] = []

    try:
        config = normalize_web_config(raw_config, create_output_dirs=create_output_dirs)
    except ConfigError as exc:
        return None, PreflightResult(ok=False, errors=(f"Configuration is incomplete or invalid: {exc}",))

    if not config.input_raster_path.exists():
        errors.append(f"Input GeoTIFF does not exist: {config.input_raster_path}")
    elif not config.input_raster_path.is_file():
        errors.append(f"Input raster path is not a file: {config.input_raster_path}")

    if config.checkpoint_path is not None:
        if not config.checkpoint_path.exists():
            errors.append(f"Weight file does not exist: {config.checkpoint_path}")
        elif not config.checkpoint_path.is_file():
            errors.append(f"Weight path is not a file: {config.checkpoint_path}")

    if selected_weight:
        weight_id = str(selected_weight.get("weight_id") or "").strip()
        weight_model = str(selected_weight.get("model_id") or "").strip()
        if weight_id and weight_model and weight_model != config.model_name:
            errors.append(f"Weight {weight_id} is registered for model {weight_model}, but the current model is {config.model_name}.")
        if weight_id and not selected_weight.get("exists", True):
            errors.append(f"The weight registry entry points to a missing file: {weight_id}")

    output_error = _check_output_dir_writable(config.output_dir)
    if output_error:
        errors.append(output_error)

    if "shapefile" not in config.vector_formats:
        warnings.append("The current vector formats do not include shapefile, so landslides.shp will not be generated.")
    if config.stride > config.tile_size:
        errors.append(f"stride ({config.stride}) cannot be greater than tile size ({config.tile_size}).")

    return config, PreflightResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def _emit_deployment_progress(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _deployment_stage_is_complete(stage: str, config: PlatformConfig) -> bool:
    manifest = _deployment_cache_manifest(config.output_dir)
    completed_stages = _completed_deployment_stages(manifest)
    if stage in completed_stages:
        return True
    if stage == "prepare":
        index = config.output_dir / "patch_index.geojson"
        return index.exists() and bool(_patch_index_count(index))
    if stage == "infer":
        index = config.output_dir / "patch_index.geojson"
        expected = _patch_index_count(index)
        if expected <= 0:
            return False
        prediction_count = len(list(config.outputs.patch_predictions.glob("*.tif")))
        return prediction_count >= expected
    if stage == "mosaic":
        return (config.outputs.rasters / "probability.tif").exists() and (config.outputs.rasters / "mask.tif").exists()
    if stage == "vectorize":
        expected = [config.outputs.vectors / f"landslides.{_vector_suffix(item)}" for item in config.vector_formats]
        return bool(expected) and all(path.exists() for path in expected)
    return False


def _patch_index_count(index_path: Path) -> int:
    loaded = _read_json(index_path)
    if not isinstance(loaded, Mapping):
        return 0
    features = loaded.get("features", [])
    return len(features) if isinstance(features, list) else 0


def _deployment_cache_manifest(output_dir: str | Path) -> Mapping[str, Any]:
    manifest = _read_json(Path(output_dir) / "deployment_cache.json")
    return manifest if isinstance(manifest, Mapping) else {}


def _completed_deployment_stages(manifest: Mapping[str, Any]) -> set[str]:
    stages: set[str] = set()
    raw_statuses = manifest.get("statuses")
    if not isinstance(raw_statuses, list):
        return stages
    for raw_status in raw_statuses:
        if not isinstance(raw_status, Mapping):
            continue
        if raw_status.get("ok") is True and raw_status.get("stage"):
            stages.add(str(raw_status["stage"]))
    return stages


def _vector_suffix(format_name: str) -> str:
    if format_name == "shapefile":
        return "shp"
    if format_name == "geojson":
        return "geojson"
    return "gpkg"


def deployment_config_fingerprint(raw_config: Mapping[str, Any]) -> str:
    """Stable fingerprint for deciding whether an output cache matches the current deployment config."""

    normalized = _deployment_fingerprint_payload(raw_config)
    return hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def deployment_job_id(raw_config: Mapping[str, Any]) -> str:
    """Stable deployment job id for same-config resume in the same output folder."""

    output_dir = str(_nested_value(raw_config, ("output", "dir"), default="deployment")).strip() or "deployment"
    return stable_job_id(
        kind="deployment",
        version=_safe_job_version(Path(output_dir).name or "deployment"),
        payload={
            "output_dir": output_dir,
            "config_fingerprint": deployment_config_fingerprint(raw_config),
        },
    )


BATCH_DEPLOYMENT_CACHE_NAME = "batch_deployment_cache.json"
BATCH_DEPLOYMENT_JOB_KIND = "batch-deployment"


def build_batch_deployment_payload(
    raw_config: Mapping[str, Any],
    *,
    source_path: str,
    selected_rasters: Sequence[str],
    aoi: str = "",
    registry_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a serializable batch-deployment payload from the current Inference fields."""

    rasters = [str(Path(path).expanduser().resolve()) for path in selected_rasters if str(path or "").strip()]
    raw_payload = _json_ready(dict(raw_config))
    if registry_root not in {None, ""} and isinstance(raw_payload, dict):
        raw_payload["registry_root"] = str(Path(registry_root).expanduser().resolve())
    return {
        "deployment_mode": "batch",
        "source_path": str(source_path or "").strip(),
        "aoi": str(aoi or "").strip(),
        "selected_rasters": rasters,
        "raw_config": raw_payload,
        "registry_root": str(Path(registry_root).expanduser().resolve()) if registry_root not in {None, ""} else "",
    }


def batch_deployment_config_fingerprint(payload: Mapping[str, Any]) -> str:
    """Stable fingerprint for a multi-raster deployment."""

    normalized = {
        "deployment_mode": "batch",
        "source_path": str(payload.get("source_path") or ""),
        "aoi": str(payload.get("aoi") or ""),
        "selected_rasters": [str(path) for path in payload.get("selected_rasters", [])],
        "raw_config": _json_ready(dict(payload.get("raw_config") or {})),
    }
    return hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def batch_deployment_job_id(payload: Mapping[str, Any]) -> str:
    """Stable job id for same batch config/output resume."""

    raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    output_dir = str(_nested_value(raw_config, ("output", "dir"), default="batch-deployment")).strip() or "batch-deployment"
    return stable_job_id(
        kind=BATCH_DEPLOYMENT_JOB_KIND,
        version=_safe_job_version(Path(output_dir).name or "batch-deployment"),
        payload={
            "output_dir": output_dir,
            "config_fingerprint": batch_deployment_config_fingerprint(payload),
        },
    )


def batch_output_cache_status(output_dir: str | Path, config_fingerprint: str) -> dict[str, Any]:
    """Inspect whether an output directory contains a matching batch-deployment cache."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / BATCH_DEPLOYMENT_CACHE_NAME
    manifest = _read_json(manifest_path)
    existing_items = [
        item
        for item in ("items", "merged", BATCH_DEPLOYMENT_CACHE_NAME)
        if (root / item).exists()
    ]
    previous_fingerprint = str(manifest.get("config_fingerprint") or "") if isinstance(manifest, Mapping) else ""
    return {
        "output_dir": str(root),
        "manifest_path": str(manifest_path),
        "has_cache": bool(existing_items or manifest),
        "existing_items": existing_items,
        "previous_fingerprint": previous_fingerprint,
        "matches": bool(previous_fingerprint and previous_fingerprint == config_fingerprint),
        "manifest": manifest if isinstance(manifest, Mapping) else None,
    }


def output_cache_status(output_dir: str | Path, config_fingerprint: str) -> dict[str, Any]:
    """Inspect whether an output directory contains a deployment cache and if it matches the current config."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "deployment_cache.json"
    manifest = _read_json(manifest_path)
    existing_items = [
        item
        for item in ("patches", "patch_predictions", "rasters", "vectors", "patch_index.geojson", "infer_manifest.json")
        if (root / item).exists()
    ]
    previous_fingerprint = ""
    if isinstance(manifest, Mapping):
        previous_fingerprint = str(manifest.get("config_fingerprint") or "")
    return {
        "output_dir": str(root),
        "manifest_path": str(manifest_path),
        "has_cache": bool(existing_items or manifest),
        "existing_items": existing_items,
        "previous_fingerprint": previous_fingerprint,
        "matches": bool(previous_fingerprint and previous_fingerprint == config_fingerprint),
        "manifest": manifest if isinstance(manifest, Mapping) else None,
    }


def clear_deployment_cache(output_dir: str | Path) -> list[str]:
    """Remove intermediate and final deployment artifacts from one output folder."""

    return _remove_deployment_paths(
        output_dir,
        (
            "patches",
            "patch_predictions",
            "rasters",
            "vectors",
            "metrics",
            "visualizations",
            "logs",
            "patch_index.geojson",
            "infer_manifest.json",
            "deployment_cache.json",
        ),
    )


def clear_batch_deployment_cache(output_dir: str | Path) -> list[str]:
    """Remove batch deployment items and merged outputs from one output folder."""

    return _remove_deployment_paths(
        output_dir,
        (
            "items",
            "merged",
            BATCH_DEPLOYMENT_CACHE_NAME,
        ),
    )


def cleanup_completed_deployment_intermediates(output_dir: str | Path) -> list[str]:
    """Remove high-volume deployment fragments after final vectors are written."""

    return _remove_deployment_paths(
        output_dir,
        (
            "patches",
            "patch_predictions",
            "patch_index.geojson",
            "infer_manifest.json",
        ),
    )


def _remove_deployment_paths(output_dir: str | Path, names: Sequence[str]) -> list[str]:
    root = Path(output_dir).expanduser().resolve()
    removed: list[str] = []
    for name in names:
        path = root / name
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, onerror=_ignore_missing_rmtree_error)
        else:
            path.unlink(missing_ok=True)
        removed.append(name)
    return removed


def _ignore_missing_rmtree_error(
    _function: Callable[..., object],
    _path: str,
    exc_info: tuple[type[BaseException], BaseException, Any],
) -> None:
    if issubclass(exc_info[0], FileNotFoundError):
        return
    raise exc_info[1]


def run_deployment_background_task(
    config: PlatformConfig,
    *,
    raw_config: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Run deployment for a background job and persist cache metadata for resume decisions."""

    fingerprint = deployment_config_fingerprint(raw_config)
    _write_deployment_cache_manifest(
        config.output_dir,
        raw_config=raw_config,
        config_fingerprint=fingerprint,
        status="running",
    )
    statuses = run_all_resumable_with_progress(config, progress_callback=progress_callback)
    artifacts = collect_artifacts(config)
    failed = next((status for status in statuses if not status.ok), None)
    cleanup_removed: list[str] = []
    if failed is not None:
        _write_deployment_cache_manifest(
            config.output_dir,
            raw_config=raw_config,
            config_fingerprint=fingerprint,
            status="failed",
            statuses=statuses,
            artifacts=artifacts,
        )
        raise RuntimeError(f"Deployment failed at {failed.stage}: {failed.message}")
    cleanup_removed = cleanup_completed_deployment_intermediates(config.output_dir)
    artifacts = collect_artifacts(config)
    _write_deployment_cache_manifest(
        config.output_dir,
        raw_config=raw_config,
        config_fingerprint=fingerprint,
        status="completed",
        statuses=statuses,
        artifacts=artifacts,
        cleanup_removed=cleanup_removed,
    )
    return {
        "statuses": [status.__dict__ for status in statuses],
        "artifacts": artifacts,
        "config_fingerprint": fingerprint,
        "cleanup_removed": cleanup_removed,
    }


def start_deployment_background_job(
    *,
    registry_root: str,
    raw_config: Mapping[str, Any],
    config: PlatformConfig,
) -> str:
    """Start or resume a deployment background job for a stable config/output pair."""

    job_id = deployment_job_id(raw_config)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        return run_deployment_background_task(config, raw_config=raw_config, progress_callback=progress_callback)

    payload_raw_config = _with_registry_root(raw_config, registry_root)
    start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="deployment",
        description=f"Deployment {config.output_dir.name or config.output_dir}",
        payload={
            "output_dir": str(config.output_dir),
            "config_fingerprint": deployment_config_fingerprint(raw_config),
            "input_raster_path": str(config.input_raster_path),
            "model": config.model_name,
            "raw_config": payload_raw_config,
            "registry_root": str(Path(registry_root).expanduser().resolve()),
        },
        target=target,
    )
    return job_id


def run_batch_deployment_background_task(
    *,
    payload: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Run selected GeoTIFF rasters one by one and merge final vector outputs."""

    raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    if not raw_config:
        raise RuntimeError("Batch deployment payload is missing raw_config.")
    selected_rasters = [str(path) for path in payload.get("selected_rasters", []) if str(path or "").strip()]
    if not selected_rasters:
        raise RuntimeError("Batch deployment requires at least one selected raster.")
    aoi_text = str(payload.get("aoi") or "").strip()
    if aoi_text:
        try:
            selected_rasters = _filter_inference_rasters_by_aoi(selected_rasters, aoi_text)
        except RegistryError as exc:
            raise RuntimeError(str(exc)) from exc
        if not selected_rasters:
            raise RuntimeError("Batch deployment AOI excludes all selected rasters.")

    output_dir = Path(str(_nested_value(raw_config, ("output", "dir"), default="outputs/deployment_run"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = batch_deployment_config_fingerprint(payload)
    manifest = _batch_deployment_manifest(output_dir)
    if manifest.get("config_fingerprint") and str(manifest.get("config_fingerprint")) != fingerprint:
        raise RuntimeError("Output directory contains a different batch deployment cache. Clear it before starting a new batch.")

    items = _batch_deployment_items(output_dir=output_dir, raw_config=raw_config, selected_rasters=selected_rasters)
    manifest = _sync_batch_deployment_manifest(
        output_dir,
        payload=payload,
        config_fingerprint=fingerprint,
        status="running",
        items=items,
        manifest=manifest,
    )
    total = len(items)
    completed_count = sum(1 for item in manifest.get("items", []) if isinstance(item, Mapping) and _batch_item_is_complete(item))
    _emit_batch_deployment_progress(
        progress_callback,
        status="running",
        phase="deployment",
        current_index=0,
        total=total,
        completed=completed_count,
        message=f"Starting batch deployment for {total} raster(s).",
    )

    for index, item in enumerate(items, start=1):
        item_id = str(item["item_id"])
        current_item = _batch_manifest_item(manifest, item_id) or item
        if _batch_item_is_complete(current_item):
            _emit_batch_deployment_progress(
                progress_callback,
                status="running",
                phase="deployment",
                item_status="skipped",
                current_index=index,
                total=total,
                completed=completed_count,
                current_raster=item["raster_path"],
                current_item_id=item_id,
                message=f"Skipping completed raster {index}/{total}: {item['raster_name']}",
            )
            continue

        item = {**item, **{key: current_item.get(key, item.get(key)) for key in ("status", "artifacts", "cleanup_removed")}}
        item["status"] = "running"
        item["started_at"] = item.get("started_at") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        item["item_started_at"] = item.get("item_started_at") or item["started_at"]
        item["item_heartbeat_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        item["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        manifest = _update_batch_manifest_item(output_dir, manifest, item)

        single_raw_config = _raw_config_for_batch_item(raw_config, raster_path=item["raster_path"], output_dir=item["output_dir"])
        config = normalize_web_config(single_raw_config, create_output_dirs=True)
        item_base = (index - 1) / total
        item_weight = 1.0 / total

        def item_progress(progress: Mapping[str, Any], *, item_base: float = item_base, item_weight: float = item_weight) -> None:
            stage_percent = min(max(float(progress.get("overall_percent", progress.get("percent", 0.0)) or 0.0), 0.0), 1.0)
            progress_payload = _batch_stage_progress_payload(progress)
            item["item_heartbeat_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            _update_batch_manifest_item(output_dir, manifest, item)
            _emit_batch_deployment_progress(
                progress_callback,
                **progress_payload,
                status=progress.get("status", "running"),
                phase=progress.get("phase") or "deployment",
                progress_kind=progress.get("progress_kind") or "batch-item",
                item_status="running",
                current_index=index,
                total=total,
                completed=completed_count,
                current_raster=item["raster_path"],
                current_item_id=item_id,
                overall_percent=item_base + stage_percent * item_weight,
            )

        try:
            statuses = run_all_resumable_with_progress(config, progress_callback=item_progress)
            failed = next((status for status in statuses if not status.ok), None)
            artifacts = collect_artifacts(config)
            if failed is not None:
                item.update(
                    {
                        "status": "failed",
                        "error": f"{failed.stage}: {failed.message}",
                        "statuses": [status.__dict__ for status in statuses],
                        "artifacts": artifacts,
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    }
                )
                manifest = _update_batch_manifest_item(output_dir, manifest, item, status="failed")
                raise RuntimeError(f"Batch deployment failed at {item['raster_name']} ({failed.stage}): {failed.message}")
            cleanup_removed = cleanup_completed_deployment_intermediates(config.output_dir)
            artifacts = collect_artifacts(config)
            vector_stats = artifacts.get("vector_stats") if isinstance(artifacts.get("vector_stats"), Mapping) else {}
            polygon_count = int(vector_stats.get("polygon_count") or 0) if isinstance(vector_stats, Mapping) else 0
            item.update(
                {
                    "status": "completed" if polygon_count > 0 else "completed-empty",
                    "statuses": [status.__dict__ for status in statuses],
                    "artifacts": artifacts,
                    "cleanup_removed": cleanup_removed,
                    "polygon_count": polygon_count,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                }
            )
            completed_count += 1
            manifest = _update_batch_manifest_item(output_dir, manifest, item)
            _emit_batch_deployment_progress(
                progress_callback,
                status="running",
                phase="deployment",
                item_status=item["status"],
                current_index=index,
                total=total,
                completed=completed_count,
                current_raster=item["raster_path"],
                current_item_id=item_id,
                overall_percent=index / total,
                message=f"Completed raster {index}/{total}: {item['raster_name']}",
            )
        except Exception:
            raise

    _emit_batch_deployment_progress(
        progress_callback,
        status="running",
        phase="merge",
        current_index=total,
        total=total,
        completed=completed_count,
        overall_percent=1.0,
        message="Merging per-raster vector outputs.",
    )
    manifest = _batch_deployment_manifest(output_dir)
    completed_items = [item for item in manifest.get("items", []) if str(item.get("status")) in {"completed", "completed-empty"}]
    merged = merge_batch_deployment_vectors(output_dir, completed_items)
    manifest = _write_batch_deployment_manifest(
        output_dir,
        {
            **manifest,
            "status": "completed",
            "merged": merged,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )
    return {
        "status": "completed",
        "items": manifest.get("items", []),
        "merged": merged,
        "artifacts": collect_batch_artifacts(output_dir),
        "config_fingerprint": fingerprint,
    }


def start_batch_deployment_background_job(*, registry_root: str, payload: Mapping[str, Any]) -> str:
    """Start or resume a batch deployment background job."""

    job_id = batch_deployment_job_id(payload)
    raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    output_dir = str(_nested_value(raw_config, ("output", "dir"), default="outputs/deployment_run"))
    worker_payload = _json_ready(dict(payload))
    if isinstance(worker_payload.get("raw_config"), Mapping):
        worker_payload["raw_config"] = _with_registry_root(worker_payload["raw_config"], registry_root)
    worker_payload["registry_root"] = str(Path(registry_root).expanduser().resolve())

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        return run_batch_deployment_background_task(payload=payload, progress_callback=progress_callback)

    start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind=BATCH_DEPLOYMENT_JOB_KIND,
        description=f"Batch deployment {Path(output_dir).name or output_dir}",
        payload={
            **worker_payload,
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "config_fingerprint": batch_deployment_config_fingerprint(payload),
        },
        target=target,
    )
    return job_id


def evaluation_job_id(raw_config: Mapping[str, Any]) -> str:
    payload = _evaluation_job_payload(raw_config)
    output_dir = str(payload.get("output_dir") or "evaluation").strip() or "evaluation"
    return stable_job_id(
        kind="evaluation",
        version=_safe_job_version(Path(output_dir).name or "evaluation"),
        payload={
            "dataset_id": str(payload.get("dataset_id") or ""),
            "weight_id": str(payload.get("weight_id") or ""),
            "model_id": str(payload.get("model_id") or ""),
            "output_dir": output_dir,
            "checkpoint_path": str(payload.get("checkpoint_path") or ""),
        },
    )


def run_evaluation_background_task(
    *,
    raw_config: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    payload = _evaluation_job_payload(raw_config)
    result = run_evaluation_dataset_task(payload, progress_callback=progress_callback)
    output = {
        "statuses": [
            {
                "stage": "evaluate",
                "ok": True,
                "message": "Evaluation dataset metrics complete.",
                "result": result,
            }
        ],
        "artifacts": result.get("artifacts", {}),
        "config_fingerprint": hashlib.sha1(json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")).hexdigest(),
        **result,
    }
    _record_evaluation_history(
        registry_root=str(payload.get("registry_root") or default_workspace_root()),
        job_id=str(payload.get("job_id") or evaluation_job_id(payload)),
        payload=payload,
        result=output,
    )
    return output


def run_evaluation_dataset_task(
    payload: Mapping[str, Any],
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a registered Evaluation images/labels sample set directly."""

    registry_root = str(payload.get("registry_root") or default_workspace_root())
    dataset_id = str(payload.get("dataset_id") or "").strip()
    requested_model_id = str(payload.get("model_id") or "LSDSAM-B").strip() or "LSDSAM-B"
    weight_id = str(payload.get("weight_id") or "").strip()
    checkpoint_path = str(payload.get("checkpoint_path") or "").strip()
    output_dir = Path(str(payload.get("output_dir") or "outputs/evaluation_run")).expanduser().resolve()
    batch_size = max(1, int(payload.get("batch_size") or 1))
    requested_device = str(payload.get("device") or "auto")
    if not dataset_id:
        raise EvaluationError("Choose an Evaluation dataset before running evaluation.")
    if not checkpoint_path:
        raise EvaluationError("Choose a registered weight before running evaluation.")
    weight_row = _evaluation_weight_row(registry_root=registry_root, weight_id=weight_id)
    model_id = _evaluation_model_id_from_weight(requested_model_id, weight_row)
    adapter_options = _evaluation_adapter_options(model_id, weight_row)
    checkpoint_path_alias = checkpoint_path
    try:
        resolved_checkpoint_path = resolve_user_path(checkpoint_path, workspace_root=registry_root)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Invalid LSDSAM checkpoint path: {checkpoint_path}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    torch = _require_module("torch", "Install torch before running Evaluation.")
    cv2 = _require_module("cv2", "Install OpenCV before running Evaluation.")
    np = _require_module("numpy", "Install numpy before running Evaluation.")
    device = resolve_training_device(requested_device, torch)
    device_name = _training_device_name(torch, device)
    dataset = resolve_daft_dataset_materialization(registry_root=registry_root, dataset_id=dataset_id)
    if dataset.dataset_class != "Evaluation":
        raise EvaluationError(f"Dataset {dataset.dataset_id} is {dataset.dataset_class}, expected Evaluation.")
    samples = dataset.evaluation_samples()
    if not samples:
        raise EvaluationError(f"Evaluation dataset {dataset.dataset_id} has no labeled image/label samples.")

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluation_json_path = metrics_dir / "evaluation.json"
    evaluation_csv_path = metrics_dir / "evaluation.csv"
    progress_callback = progress_callback or (lambda _progress: None)
    progress_callback(
        {
            "status": "running",
            "stage": "evaluate",
            "phase": "model",
            "progress_kind": "model_load",
            "percent": 0.02,
            "overall_percent": 0.02,
            "device": device,
            "device_name": device_name,
            "sample_count": len(samples),
            "current_action": "loading model",
            "message": f"Loading {model_id} for Evaluation dataset {dataset_id}.",
        }
    )
    progress_callback({"status": "running", "stage": "evaluate", "phase": "dataset", "progress_kind": "dataset_build", "percent": 0.05, "overall_percent": 0.05, "device": device, "device_name": device_name, "sample_count": len(samples), "current_action": "building evaluation dataset", "message": f"Building Evaluation dataset {dataset_id}."})
    adapter_config = _adapter_config_for_evaluation(
        model_id=model_id,
        checkpoint_path=str(resolved_checkpoint_path),
        device=device,
        RGB_bands=dataset.RGB_bands,
        adapter_options=adapter_options,
    )
    adapter = get_adapter(model_id)
    adapter.load(adapter_config)
    model = getattr(adapter, "model", None)
    if model is None:
        raise EvaluationError(f"Model adapter {model_id} does not expose a loaded torch model for dataset evaluation.")
    image_size = int(getattr(adapter, "image_size", 512) or 512)
    criterion_seg = _FocalDiceLoss(torch)
    dataset_cls = _LSDFormerSupervisedDataset if _canonical_training_model_id(model_id) == "LSDFormer" else _LSDSAMSupervisedDataset
    eval_dataset = dataset_cls(samples, image_size=image_size, cv2=cv2, torch=torch, np=np)
    metrics = _evaluate_model(
        model,
        eval_dataset=eval_dataset,
        torch=torch,
        device=device,
        criterion_seg=criterion_seg,
        batch_size=batch_size,
        num_workers=0,
        progress_callback=progress_callback,
        progress_context={
            "stage": "evaluate",
            "phase": "evaluate",
            "evaluation_split": "evaluation",
            "current_action": "evaluating samples",
            "total": _ceil_batches_for_count(len(samples), batch_size),
            "device": device,
            "device_name": device_name,
            "sample_count": len(samples),
        },
        progress_every=10,
    )
    scores = _evaluation_scores_from_metrics(metrics)
    payload_out = {
        **scores,
        "loss": metrics.get("loss"),
        "model_id": model_id,
        "requested_model_id": requested_model_id,
        "weight_id": weight_id,
        "weight_version": str(
            payload.get("weight_version")
            or weight_row.get("version")
            or payload.get("weight_id")
            or ""
        ).strip(),
        "checkpoint_path": str(resolved_checkpoint_path),
        "checkpoint_path_alias": checkpoint_path_alias
        if checkpoint_path_alias != str(resolved_checkpoint_path)
        else "",
        "adapter_options": adapter_options,
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "device": device,
        "device_name": device_name,
    }
    progress_callback({"status": "running", "stage": "evaluate", "phase": "write_metrics", "progress_kind": "write_output", "percent": 0.98, "overall_percent": 0.98, "device": device, "device_name": device_name, "sample_count": len(samples), "current_action": "writing evaluation metrics", "message": "Writing evaluation metrics."})
    _write_json_file(evaluation_json_path, payload_out)
    _write_csv_file(evaluation_csv_path, [payload_out])
    artifacts = {
        "evaluation_json_path": str(evaluation_json_path),
        "evaluation_csv_path": str(evaluation_csv_path),
        "metrics": payload_out,
    }
    progress_callback(
        {
            "status": "running",
            "stage": "evaluate",
            "phase": "completed",
            "percent": 1.0,
            "overall_percent": 1.0,
            "current_action": "evaluation completed",
            "message": f"Evaluation finished: mIoU={scores.get('miou', 0.0):.4f}, IoU={scores.get('iou', 0.0):.4f}.",
            "device": device,
            "device_name": device_name,
            "sample_count": len(samples),
            "evaluation": metrics,
        }
    )
    return {
        "artifacts": artifacts,
        "evaluation_json_path": str(evaluation_json_path),
        "evaluation_csv_path": str(evaluation_csv_path),
        "scores": scores,
        "metrics": metrics,
        "sample_count": len(samples),
    }


@dataclass(frozen=True)
class _EvaluationAdapterConfig:
    model_name: str
    checkpoint_path: Path
    device: str
    threshold: float
    RGB_bands: str
    adapter_options: dict[str, Any]
    config_path: Path | None = None


def _adapter_config_for_evaluation(
    *,
    model_id: str,
    checkpoint_path: str,
    device: str,
    RGB_bands: str,
    adapter_options: Mapping[str, Any] | None = None,
) -> _EvaluationAdapterConfig:
    return _EvaluationAdapterConfig(
        model_name=model_id,
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        device=device,
        threshold=0.5,
        RGB_bands=RGB_bands,
        adapter_options=dict(adapter_options or {}),
    )


def _ceil_batches_for_count(count: int, batch_size: int) -> int:
    return max(1, (int(count) + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))


def _training_device_name(torch: Any, device: str) -> str:
    if str(device).startswith("cuda") and bool(torch.cuda.is_available()):
        try:
            return f"{device} ({torch.cuda.get_device_name(device)})"
        except Exception:
            return device
    return device


def _evaluation_scores_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    confusion = metrics.get("confusion_matrix")
    tp = fp = fn = tn = 0
    if isinstance(confusion, Sequence) and len(confusion) >= 2:
        try:
            tn = int(round(float(confusion[0][0])))
            fp = int(round(float(confusion[0][1])))
            fn = int(round(float(confusion[1][0])))
            tp = int(round(float(confusion[1][1])))
        except (TypeError, ValueError, IndexError):
            tp = fp = fn = tn = 0
    return {
        "iou": float(metrics.get("IoU_foreground", metrics.get("IoU_landslide", 0.0)) or 0.0),
        "miou": float(metrics.get("mIoU", 0.0) or 0.0),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("F1", 0.0) or 0.0),
        "kappa": float(metrics.get("kappa", metrics.get("Kappa", 0.0)) or 0.0),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "predicted_positive": tp + fp,
        "truth_positive": tp + fn,
        "total_pixels": tp + fp + fn + tn,
    }


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _evaluation_history_path(registry_root: str) -> Path:
    return registry_dir(registry_root) / "evaluation_history.json"


def _read_evaluation_history(registry_root: str) -> list[dict[str, Any]]:
    payload = _read_json(_evaluation_history_path(registry_root))
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _write_evaluation_history(registry_root: str, rows: Sequence[Mapping[str, Any]]) -> None:
    # Keep the history file a plain list because it is also user-facing and easy to export/debug.
    path = _evaluation_history_path(registry_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    tmp_path.write_text(json.dumps(_json_ready(list(rows)), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _ui_state_path(registry_root: str) -> Path:
    return registry_dir(registry_root) / "ui_state.json"


def _read_ui_state(registry_root: str) -> dict[str, Any]:
    payload = _read_json(_ui_state_path(registry_root))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_ui_state(registry_root: str, state: Mapping[str, Any]) -> None:
    path = _ui_state_path(registry_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    tmp_path.write_text(json.dumps(_json_ready(dict(state)), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _recent_path_key(*parts: str) -> str:
    return ".".join(str(part).strip().lower().replace(" ", "_") for part in parts if str(part).strip())


def _recent_path_values(registry_root: str, key: str) -> list[str]:
    state = _read_ui_state(registry_root)
    recent_paths = state.get("recent_paths") if isinstance(state.get("recent_paths"), Mapping) else {}
    values = recent_paths.get(key) if isinstance(recent_paths, Mapping) else []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    seen: set[str] = set()
    paths: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            paths.append(text)
    return paths


def _remember_recent_path(registry_root: str, key: str, value: str, *, limit: int = 3) -> None:
    text = str(value or "").strip()
    if not text:
        return
    state = _read_ui_state(registry_root)
    recent_paths = dict(state.get("recent_paths") if isinstance(state.get("recent_paths"), Mapping) else {})
    existing = [path for path in _recent_path_values(registry_root, key) if path != text]
    recent_paths[key] = [text, *existing][:limit]
    state["recent_paths"] = recent_paths
    _write_ui_state(registry_root, state)


def _recent_path_default(registry_root: str, key: str) -> str:
    values = _recent_path_values(registry_root, key)
    return values[0] if values else ""


def _render_recent_path_datalist(st: Any, *, label: str, values: Sequence[str], datalist_id: str) -> None:
    paths = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not paths:
        return
    try:
        import streamlit.components.v1 as components
    except ModuleNotFoundError:
        return
    payload = json.dumps({"label": label, "datalist_id": datalist_id, "values": paths}, ensure_ascii=False)
    components.html(
        f"""
<script>
(function() {{
  const payload = {payload};
  const doc = window.parent.document;
  const ensureList = () => {{
    let datalist = doc.getElementById(payload.datalist_id);
    if (!datalist) {{
      datalist = doc.createElement("datalist");
      datalist.id = payload.datalist_id;
      doc.body.appendChild(datalist);
    }}
    datalist.innerHTML = "";
    payload.values.forEach((value) => {{
      const option = doc.createElement("option");
      option.value = value;
      datalist.appendChild(option);
    }});
    return datalist;
  }};
  const attach = () => {{
    const input = Array.from(doc.querySelectorAll("input")).find(
      (candidate) => candidate.getAttribute("aria-label") === payload.label
    );
    if (!input) {{
      return false;
    }}
    ensureList();
    input.setAttribute("list", payload.datalist_id);
    input.setAttribute("autocomplete", "off");
    return true;
  }};
  if (!attach()) {{
    let attempts = 0;
    const timer = window.setInterval(() => {{
      attempts += 1;
      if (attach() || attempts > 30) {{
        window.clearInterval(timer);
      }}
    }}, 100);
  }}
}})();
</script>
        """,
        height=0,
    )


def _render_recent_path_selector(st: Any, *, registry_root: str, recent_key: str, label: str, target_state_key: str) -> None:
    values = _recent_path_values(registry_root, recent_key)
    if not values:
        return
    options = ["", *values]
    selected = st.selectbox(
        label,
        options,
        index=0,
        format_func=lambda value: "Select a recent path" if not value else value,
        key=f"{target_state_key}_recent_selector",
    )
    if selected:
        st.session_state[target_state_key] = selected


def _record_evaluation_history(
    *,
    registry_root: str,
    job_id: str,
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    row = _evaluation_history_row(job_id=job_id, payload=payload, result=result)
    rows = [entry for entry in _read_evaluation_history(registry_root) if entry.get("history_id") != row["history_id"]]
    rows.append(row)
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    _write_evaluation_history(registry_root, rows)


def _evaluation_history_row(
    *,
    job_id: str,
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), Mapping) else {}
    metrics = artifacts.get("metrics") if isinstance(artifacts.get("metrics"), Mapping) else {}
    scores = result.get("scores") if isinstance(result.get("scores"), Mapping) else {}
    raw_metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    merged = {**raw_metrics, **scores, **metrics}
    output_dir = str(payload.get("output_dir") or "").strip()
    return {
        "history_id": str(job_id or stable_job_id(kind="evaluation-history", version="entry", payload=payload)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "job_id": str(job_id or ""),
        "model_id": str(payload.get("model_id") or merged.get("model_id") or ""),
        "weight_id": str(payload.get("weight_id") or merged.get("weight_id") or ""),
        "weight_version": str(payload.get("weight_version") or payload.get("weight_id") or merged.get("weight_version") or ""),
        "dataset_id": str(payload.get("dataset_id") or merged.get("dataset_id") or ""),
        "output_dir": output_dir,
        "sample_count": _int_value(merged.get("sample_count") or result.get("sample_count")),
        "loss": _float_or_none(merged.get("loss")),
        "mIoU": _float_or_none(merged.get("miou", merged.get("mIoU"))),
        "IoU": _float_or_none(merged.get("iou", merged.get("IoU_foreground", merged.get("IoU_landslide")))),
        "precision": _float_or_none(merged.get("precision")),
        "recall": _float_or_none(merged.get("recall")),
        "F1": _float_or_none(merged.get("f1", merged.get("F1"))),
        "kappa": _float_or_none(merged.get("kappa", merged.get("Kappa"))),
        "evaluation_json_path": str(result.get("evaluation_json_path") or artifacts.get("evaluation_json_path") or ""),
        "evaluation_csv_path": str(result.get("evaluation_csv_path") or artifacts.get("evaluation_csv_path") or ""),
    }


def _delete_evaluation_history_entry(registry_root: str, history_id: str) -> bool:
    rows = _read_evaluation_history(registry_root)
    kept = [row for row in rows if str(row.get("history_id") or "") != str(history_id)]
    if len(kept) == len(rows):
        return False
    _write_evaluation_history(registry_root, kept)
    return True


def _evaluation_history_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        table.append(
            {
                "created_at": row.get("created_at", ""),
                "model": row.get("model_id", ""),
                "weight_version": row.get("weight_version") or row.get("weight_id", ""),
                "dataset": row.get("dataset_id", ""),
                "loss": _format_float(row.get("loss"), digits=4),
                "mIoU": _format_float(row.get("mIoU"), digits=4),
                "IoU": _format_float(row.get("IoU"), digits=4),
                "precision": _format_float(row.get("precision"), digits=4),
                "recall": _format_float(row.get("recall"), digits=4),
                "F1": _format_float(row.get("F1"), digits=4),
                "kappa": _format_float(row.get("kappa"), digits=4),
                "samples": row.get("sample_count", ""),
                "job_id": row.get("job_id", ""),
                "output_dir": row.get("output_dir", ""),
            }
        )
    return table


def _filter_evaluation_history(rows: Sequence[Mapping[str, Any]], query: str) -> list[dict[str, Any]]:
    query_text = str(query or "").strip().lower()
    if not query_text:
        return [dict(row) for row in rows]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        haystack = json.dumps(_json_ready(dict(row)), ensure_ascii=False, sort_keys=True).lower()
        if query_text in haystack:
            filtered.append(dict(row))
    return filtered


def _evaluation_history_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    table = _evaluation_history_table(rows)
    if not table:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)
    return output.getvalue()


def start_evaluation_background_job(
    *,
    registry_root: str,
    raw_config: Mapping[str, Any],
    config: PlatformConfig | None = None,
) -> str:
    job_id = evaluation_job_id(raw_config)
    payload = {**dict(_evaluation_job_payload(raw_config)), "registry_root": registry_root, "job_id": job_id}

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
        return run_evaluation_background_task(raw_config=payload, progress_callback=progress_callback)

    start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="evaluation",
        description=f"Evaluation {payload.get('dataset_id') or Path(str(payload.get('output_dir') or 'evaluation')).name}",
        payload=payload,
        target=target,
    )
    return job_id


def _config_from_deployment_job_state(state: Mapping[str, Any]) -> PlatformConfig | None:
    return _config_from_pipeline_job_state(state)


def _config_from_evaluation_job_state(state: Mapping[str, Any]) -> PlatformConfig | None:
    payload = state.get("payload")
    if isinstance(payload, Mapping) and payload.get("dataset_id"):
        return None
    return _config_from_pipeline_job_state(state)


def _config_from_pipeline_job_state(state: Mapping[str, Any]) -> PlatformConfig | None:
    raw_config = _raw_config_from_pipeline_job_state(state)
    if not raw_config:
        return None
    try:
        return normalize_web_config(raw_config, create_output_dirs=False)
    except ConfigError:
        return None


def _raw_config_from_pipeline_job_state(state: Mapping[str, Any]) -> dict[str, Any]:
    payload = state.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    output_dir = str(payload.get("output_dir") or "")
    if output_dir:
        manifest = _read_json(Path(output_dir) / "deployment_cache.json")
        if isinstance(manifest, Mapping) and isinstance(manifest.get("raw_config"), Mapping):
            return _deployment_payload_to_raw_config(manifest["raw_config"])
    raw_payload = payload.get("raw_config")
    if isinstance(raw_payload, Mapping):
        return _deployment_payload_to_raw_config(raw_payload)
    input_raster_path = str(payload.get("input_raster_path") or "")
    model = str(payload.get("model") or "LSDFormer")
    if output_dir and input_raster_path:
        return build_raw_config(
            geotiff_path=input_raster_path,
            output_dir=output_dir,
            model=model,
            overwrite=False,
        )
    return {}


def _deployment_payload_to_raw_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.loads(json.dumps(payload))
    if isinstance(raw, dict):
        raw.setdefault("inference", {})
        if isinstance(raw["inference"], dict):
            raw["inference"].setdefault("overwrite", False)
    return raw if isinstance(raw, dict) else {}


def _write_deployment_cache_manifest(
    output_dir: Path,
    *,
    raw_config: Mapping[str, Any],
    config_fingerprint: str,
    status: str,
    statuses: list[StageStatus] | None = None,
    artifacts: dict[str, Any] | None = None,
    cleanup_removed: Sequence[str] | None = None,
) -> None:
    payload = {
        "config_fingerprint": config_fingerprint,
        "raw_config": _deployment_fingerprint_payload(raw_config),
        "status": status,
        "statuses": [stage.__dict__ for stage in statuses] if statuses is not None else [],
        "artifacts": artifacts or {},
        "cleanup_removed": list(cleanup_removed or []),
    }
    path = output_dir / "deployment_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _deployment_fingerprint_payload(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registry_root": str(_nested_value(raw_config, ("registry_root",), default="")),
        "input": {
            "raster_path": str(_nested_value(raw_config, ("input", "raster_path"), default="")),
            "RGB_bands": str(_nested_value(raw_config, ("input", "RGB_bands"), default="")),
        },
        "output": {
            "dir": str(_nested_value(raw_config, ("output", "dir"), default="")),
            "vector_formats": list(_nested_value(raw_config, ("output", "vector_formats"), default=[])),
        },
        "model": {
            "name": str(_nested_value(raw_config, ("model", "name"), default="")),
            "checkpoint_path": str(_nested_value(raw_config, ("model", "checkpoint_path"), default="")),
        },
        "tiling": {
            "tile_size": _nested_value(raw_config, ("tiling", "tile_size"), default=None),
            "stride": _nested_value(raw_config, ("tiling", "stride"), default=None),
        },
        "inference": {
            "batch_size": _nested_value(raw_config, ("inference", "batch_size"), default=None),
            "threshold": _nested_value(raw_config, ("inference", "threshold"), default=None),
        },
        "postprocess": {
            "min_area_pixels": _nested_value(raw_config, ("postprocess", "min_area_pixels"), default=None),
            "simplify_tolerance": _nested_value(raw_config, ("postprocess", "simplify_tolerance"), default=None),
            "connectivity": _nested_value(raw_config, ("postprocess", "connectivity"), default=None),
        },
        "runtime": {
            "device": str(_nested_value(raw_config, ("runtime", "device"), default="")),
            "num_workers": _nested_value(raw_config, ("runtime", "num_workers"), default=None),
        },
        "geospatial": _nested_value(raw_config, ("geospatial",), default={}),
        "truth": _nested_value(raw_config, ("truth",), default={}),
    }


def _nested_value(raw: Mapping[str, Any], path: tuple[str, ...], *, default: Any = None) -> Any:
    value: Any = raw
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _safe_job_version(value: str) -> str:
    import re

    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return token or "deployment"


def collect_artifacts(config: PlatformConfig) -> dict[str, Any]:
    """Collect known output paths and JSON artifacts for display."""

    paths = {
        "patch_index": config.output_dir / "patch_index.geojson",
        "infer_manifest": config.output_dir / "infer_manifest.json",
        "probability_raster": config.outputs.rasters / "probability.tif",
        "mask_raster": config.outputs.rasters / "mask.tif",
        "landslides_gpkg": config.outputs.vectors / "landslides.gpkg",
        "landslides_geojson": config.outputs.vectors / "landslides.geojson",
        "landslides_shp": config.outputs.vectors / "landslides.shp",
        "vector_stats_json": config.outputs.metrics / "vector_stats.json",
        "evaluation_json": config.outputs.metrics / "evaluation.json",
        "overlay_png": config.outputs.visualizations / "overlay.png",
    }
    existing_paths = {name: str(path) for name, path in paths.items() if path.exists()}
    return {
        "paths": existing_paths,
        "expected_paths": {name: str(path) for name, path in paths.items()},
        "evaluation": _read_json(paths["evaluation_json"]),
        "vector_stats": _read_json(paths["vector_stats_json"]),
        "overlay_png": str(paths["overlay_png"]) if paths["overlay_png"].exists() else None,
    }


def preflight_batch_deployment_payload(
    payload: Mapping[str, Any],
    *,
    selected_weight: Mapping[str, Any] | None = None,
    create_output_dirs: bool = False,
) -> PreflightResult:
    """Validate a batch deployment before launching heavy processing."""

    raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    selected_rasters = [str(path) for path in payload.get("selected_rasters", []) if str(path or "").strip()]
    errors: list[str] = []
    warnings: list[str] = []
    if not selected_rasters:
        errors.append("Select at least one GeoTIFF for inference.")
        return PreflightResult(ok=False, errors=tuple(errors))
    aoi_text = str(payload.get("aoi") or "").strip()
    if aoi_text:
        try:
            selected_rasters = _filter_inference_rasters_by_aoi(selected_rasters, aoi_text)
        except RegistryError as exc:
            return PreflightResult(ok=False, errors=(str(exc),))

    first_raw = _raw_config_for_batch_item(
        raw_config,
        raster_path=selected_rasters[0],
        output_dir=_batch_item_output_dir(
            Path(str(_nested_value(raw_config, ("output", "dir"), default="outputs/deployment_run"))).expanduser().resolve(),
            selected_rasters[0],
        ),
    )
    try:
        config = normalize_web_config(first_raw, create_output_dirs=create_output_dirs)
    except ConfigError as exc:
        return PreflightResult(ok=False, errors=(f"Configuration is incomplete or invalid: {exc}",))

    for raster in selected_rasters:
        path = Path(raster).expanduser()
        if not path.exists():
            errors.append(f"Selected GeoTIFF does not exist: {path}")
        elif not path.is_file():
            errors.append(f"Selected raster path is not a file: {path}")
        elif path.suffix.lower() not in {".tif", ".tiff"}:
            errors.append(f"Selected raster must be a .tif/.tiff file: {path}")

    if config.checkpoint_path is not None:
        if not config.checkpoint_path.exists():
            errors.append(f"Weight file does not exist: {config.checkpoint_path}")
        elif not config.checkpoint_path.is_file():
            errors.append(f"Weight path is not a file: {config.checkpoint_path}")

    if selected_weight:
        weight_id = str(selected_weight.get("weight_id") or "").strip()
        weight_model = str(selected_weight.get("model_id") or "").strip()
        if weight_id and weight_model and weight_model != config.model_name:
            errors.append(f"Weight {weight_id} is registered for model {weight_model}, but the current model is {config.model_name}.")
        if weight_id and not selected_weight.get("exists", True):
            errors.append(f"The weight registry entry points to a missing file: {weight_id}")

    output_root = Path(str(_nested_value(raw_config, ("output", "dir"), default="outputs/deployment_run"))).expanduser().resolve()
    output_error = _check_output_dir_writable(output_root)
    if output_error:
        errors.append(output_error)
    if "shapefile" not in config.vector_formats:
        warnings.append("The current vector formats do not include shapefile, so landslides.shp will not be generated.")
    if config.stride > config.tile_size:
        errors.append(f"stride ({config.stride}) cannot be greater than tile size ({config.tile_size}).")
    return PreflightResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def collect_batch_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Collect known batch deployment output paths for display."""

    root = Path(output_dir).expanduser().resolve()
    paths = {
        "batch_manifest": root / BATCH_DEPLOYMENT_CACHE_NAME,
        "merged_gpkg": root / "merged" / "landslides_merged.gpkg",
        "merged_geojson": root / "merged" / "landslides_merged.geojson",
        "merged_shp": root / "merged" / "landslides_merged.shp",
    }
    manifest = _batch_deployment_manifest(root)
    return {
        "paths": {name: str(path) for name, path in paths.items() if path.exists()},
        "expected_paths": {name: str(path) for name, path in paths.items()},
        "manifest": manifest,
        "items": manifest.get("items", []) if isinstance(manifest, Mapping) else [],
        "merged": manifest.get("merged", {}) if isinstance(manifest, Mapping) else {},
    }


def _emit_batch_deployment_progress(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    total = max(1, int(payload.get("total") or 1))
    current_index = int(payload.get("current_index") or 0)
    overall = payload.get("overall_percent")
    if overall is None:
        overall = min(max(current_index / total, 0.0), 1.0)
    canonical_percent = min(max(float(overall or 0.0), 0.0), 1.0)
    progress_callback(
        {
            **_json_ready(payload),
            "stage": "batch-deployment",
            "percent": canonical_percent,
            "overall_percent": canonical_percent,
        }
    )


def _batch_stage_progress_payload(progress: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one raster-stage progress event before wrapping it in batch metadata."""

    stage_key_map = {
        "stage": "item_stage",
        "status": "item_stage_status",
        "phase": "item_stage_phase",
        "percent": "item_stage_percent",
        "overall_percent": "item_stage_overall_percent",
        "processed": "item_stage_processed",
        "total": "item_stage_total",
        "written": "item_stage_written",
        "skipped": "item_stage_skipped",
        "rewritten": "item_stage_rewritten",
        "current_index": "item_stage_current_index",
        "completed": "item_stage_completed",
        "current_raster": "item_stage_current_raster",
        "current_item_id": "item_stage_current_item_id",
        "item_status": "item_stage_item_status",
        "progress_kind": "item_stage_progress_kind",
    }
    stage_payload: dict[str, Any] = {}
    for key, value in progress.items():
        stage_payload[stage_key_map.get(str(key), str(key))] = value
    return stage_payload


def _batch_deployment_manifest(output_dir: str | Path) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir).expanduser().resolve() / BATCH_DEPLOYMENT_CACHE_NAME)
    return manifest if isinstance(manifest, dict) else {}


def _write_batch_deployment_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(dict(manifest))
    (root / BATCH_DEPLOYMENT_CACHE_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _sync_batch_deployment_manifest(
    output_dir: str | Path,
    *,
    payload: Mapping[str, Any],
    config_fingerprint: str,
    status: str,
    items: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    existing_by_id = {
        str(item.get("item_id")): dict(item)
        for item in manifest.get("items", [])
        if isinstance(item, Mapping) and item.get("item_id")
    }
    synced_items: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("item_id") or "")
        existing = existing_by_id.get(item_id, {})
        synced = {**dict(item), **existing}
        synced.setdefault("status", "pending")
        synced_items.append(synced)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return _write_batch_deployment_manifest(
        output_dir,
        {
            **dict(manifest),
            "deployment_mode": "batch",
            "config_fingerprint": config_fingerprint,
            "payload": _json_ready(dict(payload)),
            "status": status,
            "items": synced_items,
            "updated_at": now,
            "created_at": manifest.get("created_at") or now,
        },
    )


def _update_batch_manifest_item(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    status: str | None = None,
) -> dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    updated_item = {**dict(item)}
    if status is not None:
        updated_item["status"] = status
    updated_item["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    items: list[dict[str, Any]] = []
    replaced = False
    for existing in manifest.get("items", []):
        if isinstance(existing, Mapping) and str(existing.get("item_id") or "") == item_id:
            items.append(updated_item)
            replaced = True
        elif isinstance(existing, Mapping):
            items.append(dict(existing))
    if not replaced:
        items.append(updated_item)
    next_status = status or str(manifest.get("status") or "running")
    if next_status == "failed":
        manifest_status = "failed"
    else:
        manifest_status = str(manifest.get("status") or "running")
    return _write_batch_deployment_manifest(
        output_dir,
        {
            **dict(manifest),
            "status": manifest_status,
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )


def _batch_manifest_item(manifest: Mapping[str, Any], item_id: str) -> dict[str, Any] | None:
    for item in manifest.get("items", []):
        if isinstance(item, Mapping) and str(item.get("item_id") or "") == item_id:
            return dict(item)
    return None


def _batch_deployment_items(
    *,
    output_dir: Path,
    raw_config: Mapping[str, Any],
    selected_rasters: Sequence[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, raster_path in enumerate(selected_rasters, start=1):
        output = _batch_item_output_dir(output_dir, raster_path)
        fingerprint = deployment_config_fingerprint(
            _raw_config_for_batch_item(raw_config, raster_path=raster_path, output_dir=output)
        )
        items.append(
            {
                "item_id": _batch_item_id(raster_path),
                "source_index": index,
                "raster_path": str(Path(raster_path).expanduser().resolve()),
                "raster_name": Path(raster_path).name,
                "source_stem": Path(raster_path).stem,
                "output_dir": str(output),
                "config_fingerprint": fingerprint,
                "status": "pending",
            }
        )
    return items


def _batch_item_id(raster_path: str | Path) -> str:
    path = Path(raster_path).expanduser().resolve()
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _batch_item_output_dir(output_dir: Path, raster_path: str | Path) -> Path:
    path = Path(raster_path).expanduser().resolve()
    return output_dir / "items" / f"{_safe_job_version(path.stem)}-{_batch_item_id(path)}"


def _raw_config_for_batch_item(raw_config: Mapping[str, Any], *, raster_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    raw = json.loads(json.dumps(_json_ready(dict(raw_config))))
    raw.setdefault("input", {})
    raw.setdefault("output", {})
    registry_root = str(raw.get("registry_root") or "").strip()
    if isinstance(raw["input"], dict):
        raw["input"]["raster_path"] = str(Path(raster_path).expanduser().resolve())
    if isinstance(raw["output"], dict):
        raw["output"]["dir"] = str(Path(output_dir).expanduser().resolve())
    raw.setdefault("inference", {})
    if isinstance(raw["inference"], dict):
        raw["inference"].setdefault("overwrite", False)
    if registry_root:
        raw["registry_root"] = registry_root
    return raw


def _batch_item_is_complete(item: Mapping[str, Any]) -> bool:
    if str(item.get("status") or "") not in {"completed", "completed-empty"}:
        return False
    artifacts = item.get("artifacts") if isinstance(item.get("artifacts"), Mapping) else {}
    paths = artifacts.get("paths") if isinstance(artifacts.get("paths"), Mapping) else {}
    vector_path = paths.get("landslides_gpkg") or paths.get("landslides_geojson") or paths.get("landslides_shp")
    if vector_path and Path(str(vector_path)).exists():
        return True
    output_dir = item.get("output_dir")
    return bool(output_dir and (Path(str(output_dir)) / "vectors" / "landslides.gpkg").exists())


def merge_batch_deployment_vectors(output_dir: str | Path, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-raster landslide vectors into output_dir/merged."""

    geopandas = _require_geopandas_for_batch_merge()
    import pandas as pd

    root = Path(output_dir).expanduser().resolve()
    merged_dir = root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Any] = []
    empty_count = 0
    crs: Any = None
    for item in items:
        vector_path = _batch_item_vector_path(item)
        if vector_path is None or not vector_path.exists():
            empty_count += 1
            continue
        try:
            frame = geopandas.read_file(vector_path)
        except Exception:
            empty_count += 1
            continue
        if crs is None and getattr(frame, "crs", None) is not None:
            crs = frame.crs
        if frame.empty:
            empty_count += 1
            continue
        frame = frame.copy()
        frame["source_raster"] = str(item.get("raster_path") or "")
        frame["source_stem"] = str(item.get("source_stem") or Path(str(item.get("raster_path") or "")).stem)
        frame["source_index"] = int(item.get("source_index") or 0)
        frame["source_vector_path"] = str(vector_path)
        frames.append(frame)

    if frames:
        merged = geopandas.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    else:
        merged = geopandas.GeoDataFrame(
            {
                "source_raster": [],
                "source_stem": [],
                "source_index": [],
                "source_vector_path": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=crs,
        )

    gpkg_path = merged_dir / "landslides_merged.gpkg"
    geojson_path = merged_dir / "landslides_merged.geojson"
    shapefile_path = merged_dir / "landslides_merged.shp"
    _remove_shapefile_bundle(shapefile_path)
    for path in (gpkg_path, geojson_path):
        path.unlink(missing_ok=True)
    write_errors: dict[str, str] = {}
    if merged.empty:
        geojson_path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
        try:
            merged.to_file(gpkg_path, driver="GPKG", layer="landslides")
        except Exception as exc:
            write_errors["merged_gpkg"] = str(exc)
        try:
            merged.to_file(shapefile_path, driver="ESRI Shapefile")
        except Exception as exc:
            write_errors["merged_shp"] = str(exc)
            _remove_shapefile_bundle(shapefile_path)
    else:
        merged.to_file(gpkg_path, driver="GPKG", layer="landslides")
        merged.to_file(geojson_path, driver="GeoJSON")
        merged.to_file(shapefile_path, driver="ESRI Shapefile")
    paths = {
        "merged_geojson": str(geojson_path),
        **({"merged_gpkg": str(gpkg_path)} if gpkg_path.exists() else {}),
        **({"merged_shp": str(shapefile_path)} if shapefile_path.exists() else {}),
    }
    return {
        "paths": paths,
        "feature_count": int(len(merged)),
        "source_count": len(items),
        "empty_source_count": empty_count,
        "write_errors": write_errors,
    }


def _batch_item_vector_path(item: Mapping[str, Any]) -> Path | None:
    artifacts = item.get("artifacts") if isinstance(item.get("artifacts"), Mapping) else {}
    paths = artifacts.get("paths") if isinstance(artifacts.get("paths"), Mapping) else {}
    for key in ("landslides_gpkg", "landslides_geojson", "landslides_shp"):
        value = paths.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    output_dir = item.get("output_dir")
    if output_dir:
        for candidate in (
            Path(str(output_dir)) / "vectors" / "landslides.gpkg",
            Path(str(output_dir)) / "vectors" / "landslides.geojson",
            Path(str(output_dir)) / "vectors" / "landslides.shp",
        ):
            if candidate.exists():
                return candidate
    return None


def _require_geopandas_for_batch_merge() -> Any:
    try:
        import geopandas
    except ModuleNotFoundError as exc:
        raise RuntimeError("Batch vector merge requires geopandas. Start LSDetector with the geospatial environment.") from exc
    return geopandas


def _remove_shapefile_bundle(path: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path.with_suffix(suffix).unlink(missing_ok=True)


def main() -> None:
    """Run the Streamlit app."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError("Streamlit is required to run the Web GUI. Install the 'web' extra first.") from exc

    st.set_page_config(page_title=f"{APP_DISPLAY_NAME} Console", layout="wide")
    sidebar_logo_uri = _image_data_uri(SIDEBAR_LOGO_PATH)
    main_background_uri = _image_data_uri(MAIN_BACKGROUND_PATH)
    st.markdown(
        """
        <style>
        :root {
          --lsd-bg: #f5f7fb;
          --lsd-surface: #ffffff;
          --lsd-border: rgba(31, 41, 55, 0.13);
          --lsd-text: #111827;
          --lsd-muted: #5b6472;
          --lsd-primary: #e5484d;
          --lsd-primary-dark: #c93238;
          --lsd-accent: #0f766e;
          --lsd-radius: 8px;
          color-scheme: light;
        }
        .stApp {
          background: transparent;
          color: var(--lsd-text);
          height: 100vh;
          overflow-x: hidden;
          overflow-y: auto;
          position: relative;
        }
        body {
          background: var(--lsd-bg);
          overflow: hidden;
          color: var(--lsd-text);
          color-scheme: light;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .block-container {
          color: var(--lsd-text);
          color-scheme: light;
        }
        .stApp p,
        .stApp label,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p {
          color: var(--lsd-text);
        }
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        .stApp [data-testid="stCaptionContainer"] span {
          color: var(--lsd-muted);
        }
        .stApp input,
        .stApp textarea,
        .stApp [data-baseweb="input"] input,
        .stApp [data-baseweb="textarea"] textarea,
        .stApp [data-baseweb="select"] > div {
          background-color: rgba(255, 255, 255, 0.76) !important;
          color: var(--lsd-text) !important;
        }
        .stApp [data-baseweb="select"] span,
        .stApp [data-baseweb="select"] svg {
          color: var(--lsd-text) !important;
          fill: var(--lsd-text) !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
          color-scheme: light;
        }
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"] {
          background: #ffffff !important;
          color: var(--lsd-text) !important;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          background: #ffffff;
          color: var(--lsd-text);
        }
        button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"] {
          color: #ffffff !important;
        }
        body::before {
          content: "";
          position: fixed;
          inset: 0 0 0 23.4rem;
          z-index: 0;
          pointer-events: none;
          background:
            linear-gradient(90deg, rgba(245, 247, 251, 0.86) 0%, rgba(245, 247, 251, 0.54) 38%, rgba(245, 247, 251, 0.26) 100%),
            linear-gradient(180deg, rgba(245, 247, 251, 0.72) 0%, rgba(245, 247, 251, 0.28) 48%, rgba(245, 247, 251, 0.70) 100%),
            url("__MAIN_BACKGROUND_URI__");
          background-size: cover;
          background-position: 58% 18%;
          opacity: 0.78;
          filter: saturate(1.05) brightness(1.08);
        }
        body::after {
          content: "";
          position: fixed;
          inset: 0 0 0 23.4rem;
          z-index: 0;
          pointer-events: none;
          background:
            radial-gradient(circle at 66% 35%, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0.38) 28%, rgba(245,247,251,0.74) 72%),
            linear-gradient(90deg, rgba(245,247,251,0.70) 0%, rgba(245,247,251,0.20) 50%, rgba(245,247,251,0.34) 100%);
        }
        .stApp > header,
        .stApp > div,
        .stApp section[data-testid="stSidebar"] {
          position: relative;
          z-index: 1;
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.0); }
        section[data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--lsd-border);
          color: var(--lsd-text);
          color-scheme: light;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: var(--lsd-text) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          gap: 0.9rem;
        }
        section[data-testid="stSidebar"] [data-testid="stTextInput"] label,
        section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
          text-align: left;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] {
          display: grid;
          gap: 10px;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
          width: 100%;
          min-height: 52px;
          align-items: center;
          justify-content: center;
          padding: 10px 14px;
          border: 1px solid transparent;
          border-radius: var(--lsd-radius);
          background: transparent;
          white-space: normal;
          transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:last-child,
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
          font-size: 1.08rem;
          line-height: 1.25;
          text-align: center;
          color: var(--lsd-text) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
          background: #f3f5f9;
          border-color: var(--lsd-border);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
          background: #fff1f2;
          border-color: rgba(229, 72, 77, 0.38);
          color: var(--lsd-primary-dark);
          font-weight: 650;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
          color: var(--lsd-primary-dark) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
          color: var(--lsd-muted) !important;
          line-height: 1.45;
        }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p,
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] span {
          color: var(--lsd-muted) !important;
        }
        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 3rem;
          max-width: 1440px;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
          min-height: 130px;
          padding: 12px 24px 12px 24px;
          background-image: url("__SIDEBAR_LOGO_URI__");
          background-repeat: no-repeat;
          background-position: 50% 16px;
          background-size: min(300px, 82%) auto;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] [data-testid="stLogo"],
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] a[data-testid="stLogo"] {
          opacity: 0;
          pointer-events: none;
        }
        h1 {
          letter-spacing: 0;
          color: var(--lsd-text);
          font-size: 2rem;
          line-height: 1.15;
          margin-bottom: 0.25rem;
        }
        h2, h3 {
          letter-spacing: 0;
          color: var(--lsd-text);
        }
        button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"] {
          background: var(--lsd-primary);
          border-color: var(--lsd-primary);
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          border-radius: var(--lsd-radius);
          min-height: 40px;
        }
        input, textarea, select {
          border-radius: var(--lsd-radius) !important;
        }
        .repo-panel {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 14px 16px;
          margin: 8px 0 14px 0;
        }
        .repo-panel h3 {
          font-size: 1rem;
          line-height: 1.2;
          margin: 0 0 6px 0;
          color: #111827;
        }
        .repo-panel p {
          margin: 0;
          color: #4b5563;
          font-size: 0.9rem;
          line-height: 1.45;
        }
        .module-title {
          margin: 0 0 14px 0;
          color: #111827;
          font-size: 1.02rem;
          line-height: 1.25;
          font-weight: 700;
        }
        .module-note {
          margin: -6px 0 12px 0;
          color: #4b5563;
          font-size: 0.88rem;
          line-height: 1.45;
        }
        .repo-browser-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr));
          gap: 12px;
          margin: 10px 0 16px 0;
        }
        .repo-browser-card {
          border: 1px solid rgba(31, 41, 55, 0.12);
          border-radius: 8px;
          background: rgba(255,255,255,0.88);
          padding: 12px 14px;
        }
        .repo-browser-card strong {
          display: block;
          color: #111827;
          font-size: 0.95rem;
          line-height: 1.2;
          margin-bottom: 8px;
        }
        .repo-browser-card span {
          display: inline-block;
          color: #4b5563;
          font-size: 0.82rem;
          line-height: 1.45;
          margin-right: 12px;
        }
        .dataset-upload-shell {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 14px 16px;
          margin: 12px 0 18px 0;
        }
        .dataset-upload-shell h4 {
          margin: 0 0 4px 0;
          color: #111827;
          font-size: 1rem;
          line-height: 1.25;
        }
        .dataset-upload-shell p {
          margin: 0;
          color: #4b5563;
          font-size: 0.86rem;
          line-height: 1.45;
        }
        .taft-import-panel {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 14px 16px;
          margin: 10px 0 8px 0;
        }
        .taft-import-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          color: #111827;
          font-size: 0.95rem;
          line-height: 1.2;
          margin-bottom: 10px;
        }
        .taft-import-head span {
          color: #374151;
          font-variant-numeric: tabular-nums;
        }
        .taft-progress-track {
          width: 100%;
          height: 12px;
          border-radius: 999px;
          overflow: hidden;
          background: rgba(31, 41, 55, 0.10);
        }
        .taft-progress-fill {
          height: 100%;
          border-radius: inherit;
          background: linear-gradient(90deg, #2563eb, var(--lsd-accent));
          transition: width 160ms ease;
        }
        .taft-import-stats {
          display: flex;
          flex-wrap: wrap;
          gap: 8px 14px;
          margin-top: 10px;
          color: #4b5563;
          font-size: 0.82rem;
          line-height: 1.35;
          font-variant-numeric: tabular-nums;
        }
        .taft-import-message {
          margin-top: 8px;
          color: #374151;
          font-size: 0.84rem;
          line-height: 1.4;
          overflow-wrap: anywhere;
        }
        .module-job-card {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 12px 14px 14px 14px;
          margin: 12px 0 14px 0;
        }
        .module-job-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 12px;
          margin-bottom: 10px;
        }
        .module-job-head strong {
          color: #111827;
          font-size: 0.95rem;
          line-height: 1.2;
        }
        .module-job-head span {
          color: #4b5563;
          font-size: 0.78rem;
          font-variant-numeric: tabular-nums;
          white-space: nowrap;
        }
        .module-job-meta {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
          gap: 8px;
          margin-top: 10px;
        }
        .module-job-meta span {
          border: 1px solid rgba(31, 41, 55, 0.08);
          border-radius: 8px;
          background: rgba(249,250,251,0.82);
          color: #374151;
          font-size: 0.78rem;
          line-height: 1.25;
          padding: 7px 8px;
          font-variant-numeric: tabular-nums;
        }
        .module-job-meta label {
          display: block;
          color: #6b7280;
          font-size: 0.68rem;
          line-height: 1.1;
          margin-bottom: 3px;
        }
        .module-job-message {
          margin-top: 9px;
          color: #4b5563;
          font-size: 0.82rem;
          line-height: 1.4;
          overflow-wrap: anywhere;
        }
        .daft-train-panel {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 14px 16px;
          margin: 12px 0 10px 0;
        }
        .daft-train-head {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          align-items: baseline;
          color: #111827;
          font-size: 0.96rem;
          line-height: 1.25;
          margin-bottom: 10px;
        }
        .daft-train-head span {
          color: #4b5563;
          font-size: 0.82rem;
          font-variant-numeric: tabular-nums;
        }
        .daft-train-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 8px;
          margin-top: 10px;
        }
        .daft-train-stat {
          border: 1px solid rgba(31, 41, 55, 0.10);
          border-radius: 8px;
          background: rgba(249, 250, 251, 0.92);
          padding: 8px 10px;
        }
        .daft-train-stat label {
          display: block;
          color: #6b7280;
          font-size: 0.72rem;
          line-height: 1.2;
          margin-bottom: 4px;
        }
        .daft-train-stat strong {
          color: #111827;
          display: block;
          font-size: 0.96rem;
          line-height: 1.2;
          font-variant-numeric: tabular-nums;
          overflow-wrap: anywhere;
        }
        .daft-train-log {
          margin-top: 10px;
          padding: 9px 10px;
          border-radius: 8px;
          background: #111827;
          color: #f9fafb;
          font-size: 0.82rem;
          line-height: 1.45;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
          overflow-wrap: anywhere;
        }
        div[data-testid="stMetric"] {
          background: var(--lsd-surface);
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          padding: 10px 12px;
        }
        div[data-testid="stMetricLabel"] {
          color: #4b5563;
          font-size: 0.78rem;
        }
        .deployment-panel {
          border: 1px solid var(--lsd-border);
          border-radius: var(--lsd-radius);
          background: var(--lsd-surface);
          padding: 16px 18px;
          margin: 10px 0 16px 0;
        }
        .deployment-panel h3 {
          margin: 0 0 6px 0;
          color: #111827;
          font-size: 1.1rem;
          line-height: 1.2;
        }
        .deployment-panel p {
          margin: 0;
          color: #4b5563;
          font-size: 0.92rem;
          line-height: 1.5;
        }
        @media (max-width: 900px) {
          body::before,
          body::after {
            inset: 0;
          }
          body::before {
            opacity: 0.44;
          }
        }
        </style>
        """.replace("__MAIN_BACKGROUND_URI__", main_background_uri).replace("__SIDEBAR_LOGO_URI__", sidebar_logo_uri),
        unsafe_allow_html=True,
    )
    st.title(f"{APP_DISPLAY_NAME} Local Workbench")
    st.caption("Model, weight, data, inference, evaluation, and training repositories for landslide mapping.")

    if sidebar_logo_uri:
        st.logo(str(SIDEBAR_LOGO_PATH))

    registry_root = st.sidebar.text_input("Asset Workspace", str(default_workspace_root()))
    st.sidebar.caption(f"Registered Asset Registry: {registry_dir(registry_root)}")
    selected_section = st.sidebar.radio(
        "Workspace Section",
        SECTION_OPTIONS,
        key="workspace_section",
    )

    if selected_section == "Model Repository":
        st.subheader("Model Repository")
        model_rows = list_models(project_root=Path.cwd(), registry_root=registry_root)
        st.dataframe(_model_repository_table(model_rows), use_container_width=True, hide_index=True)

    elif selected_section == "Weight Repository":
        st.subheader("Weight Repository")
        weight_rows = list_weights(registry_root=registry_root)
        with st.container(border=True):
            st.markdown('<div class="module-title">Weight registry entries</div>', unsafe_allow_html=True)
            filtered_weight_rows = _render_weight_repository_filters(st, weight_rows)
            st.dataframe(_weight_repository_table(filtered_weight_rows), use_container_width=True, hide_index=True)
        with st.container(border=True):
            _render_weight_add_form(st, registry_root=registry_root, model_options=MODEL_CHOICES)
        with st.container(border=True):
            _render_weight_delete_panel(st, registry_root=registry_root, rows=filtered_weight_rows)

    elif selected_section == "Dataset Repository":
        st.subheader("Dataset Repository")
        _render_dataset_repository_header(st)
        summary_rows = list_dataset_repository_summary(registry_root=registry_root)
        dataset_rows = list_datasets(registry_root=registry_root)
        _render_dataset_repository_metrics(st, summary_rows=summary_rows, dataset_rows=dataset_rows)

        dataset_store_filtered_rows = _render_dataset_store_browser(st, registry_root=registry_root)

        upload_class = _render_dataset_upload_console(st, registry_root=registry_root)
        upload_session_key = _dataset_upload_session_key(upload_class)
        active_upload_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind="dataset-upload",
            session_key=upload_session_key,
            job_filter=_dataset_upload_job_filter(upload_class),
            include_latest_terminal=False,
        )
        if active_upload_job_id:
            active_upload_state = read_job_state(registry_root, active_upload_job_id)
            active_upload_status = str((active_upload_state or {}).get("status") or "")
            upload_title_prefix = "Current" if _job_is_active(active_upload_state or {}) else "Last submitted"
            _render_module_job_progress_auto(
                st,
                registry_root=registry_root,
                job_id=active_upload_job_id,
                title=f"{upload_title_prefix} {upload_class} dataset upload",
            )
            if active_upload_status in {"completed", "failed", "cancelled", "interrupted"}:
                if st.button(f"Clear {upload_class} upload monitor", key=f"clear_{upload_session_key}"):
                    st.session_state.pop(upload_session_key, None)
                    st.rerun()
        summary_rows = list_dataset_repository_summary(registry_root=registry_root)
        dataset_rows = list_datasets(registry_root=registry_root)
        with st.container(border=True):
            st.markdown('<div class="module-title">Dataset registry entries</div>', unsafe_allow_html=True)
            st.dataframe(_dataset_registry_table(dataset_rows), use_container_width=True, hide_index=True)
        if dataset_store_filtered_rows:
            _render_dataset_store_delete_panel(st, registry_root=registry_root, rows=dataset_store_filtered_rows)

    elif selected_section == "DAFT Training":
        st.subheader("DAFT Fine-tuning")
        _render_daft_training_panel(st, registry_root=registry_root)

    elif selected_section == "TSFT Training":
        st.subheader("TSFT Fine-tuning")
        _render_tsft_training_panel(st, registry_root=registry_root)

    elif selected_section == "Inference":
        st.subheader("Inference / Deployment")

        defaults = st.session_state.setdefault("lsd_app_defaults", {})
        inference_weights = list_weights(registry_root=registry_root)
        weight_options = _weight_id_options(inference_weights)
        weight_format = _weight_format_func(inference_weights)
        default_weight = defaults.get("selected_weight_id")
        default_weight_index = weight_options.index(default_weight) if default_weight in weight_options else 0

        st.markdown(
            """
            <div class="deployment-panel">
              <h3>One-click deployment</h3>
              <p>Select a registered weight, choose the input GeoTIFF, and generate final landslide polygons in one run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        primary_left, primary_right = st.columns((1.35, 1.0))
        with primary_left:
            selected_weight_id = st.selectbox(
                "registered weight",
                weight_options,
                index=default_weight_index,
                format_func=weight_format,
            )
            selected_weight = next(
                (row for row in inference_weights if row.get("weight_id") == selected_weight_id),
                {},
            )
            inferred_model = str(selected_weight.get("model_id") or defaults.get("model") or MODEL_CHOICES[0])
            if inferred_model not in MODEL_CHOICES:
                inferred_model = MODEL_CHOICES[0]
            checkpoint_default = str(selected_weight.get("path") or defaults.get("checkpoint_path", ""))
            channel_default = _default_RGB_bands_from_weight(selected_weight, defaults.get("RGB_bands", DEFAULT_RGB_BANDS))
            geotiff_path = st.text_input("GeoTIFF path", defaults.get("geotiff_path", ""))
            _render_inference_tiff_discovery_controls(st, default_source=geotiff_path, default_RGB_bands=channel_default)
        with primary_right:
            output_dir = st.text_input("Output Dir", defaults.get("output_dir", "outputs/deployment_run"))
            model = st.selectbox(
                "model",
                MODEL_CHOICES,
                index=MODEL_CHOICES.index(inferred_model) if inferred_model in MODEL_CHOICES else 0,
            )
            checkpoint_label = Path(checkpoint_default).name if checkpoint_default else "not configured"
            st.caption(f"checkpoint: {checkpoint_label}")

        with st.expander("Advanced parameters", expanded=False):
            advanced_left, advanced_right = st.columns(2)
            with advanced_left:
                RGB_bands = _rgb_bands_input(
                    st,
                    "RGB bands",
                    key="inference_RGB_bands",
                    default=channel_default,
                )
                tile_size = st.number_input("tile size", min_value=1, value=int(defaults.get("tile_size", 512)), step=1)
                stride = st.number_input("stride", min_value=1, value=int(defaults.get("stride", 256)), step=1)
            with advanced_right:
                batch_size = st.number_input("Batch Size", min_value=1, value=int(defaults.get("batch_size", 1)), step=1)
                threshold = st.slider("threshold", min_value=0.0, max_value=1.0, value=float(defaults.get("threshold", 0.5)))
                min_area_pixels = st.number_input(
                    "min area pixels",
                    min_value=0,
                    value=int(defaults.get("min_area_pixels", 0)),
                    step=1,
                    help="Remove connected predicted landslide regions smaller than this pixel count before vector export.",
                )
                st.caption("device: auto (GPU first, CPU fallback)")

        checkpoint_path = checkpoint_default
        model_options = _model_options_from_weight(selected_weight)
        truth_path = ""
        selected_inference_rasters = _selected_inference_rasters_from_state()
        inference_aoi = str(st.session_state.get("inference_selected_raster_aoi") or "").strip()
        if selected_inference_rasters:
            st.caption(f"Batch selection: {len(selected_inference_rasters)} raster(s) selected from discovery.")
            if inference_aoi:
                st.caption(f"Batch AOI filter: {inference_aoi}")

        action_cols = st.columns((1.0, 1.0, 4.0))
        run_everything = action_cols[0].button("Generate shp", type="primary", use_container_width=True)
        run_visual_preview = action_cols[1].button("Preview", use_container_width=True)
        confirm_cache_clear = False
        confirm_batch_cache_clear = False

        raw_config = build_raw_config(
            geotiff_path=geotiff_path,
            output_dir=output_dir,
            model=model,
            checkpoint_path=checkpoint_path,
            model_options=model_options,
            tile_size=int(tile_size),
            stride=int(stride),
            batch_size=int(batch_size),
            threshold=float(threshold),
            min_area_pixels=int(min_area_pixels),
            device="auto",
            truth_path=truth_path,
            RGB_bands=RGB_bands,
            overwrite=False,
        )
        raw_config = _with_registry_root(raw_config, registry_root)
        current_fingerprint = deployment_config_fingerprint(raw_config)
        current_job_id = deployment_job_id(raw_config)
        cache_status = output_cache_status(output_dir, current_fingerprint)
        batch_payload = build_batch_deployment_payload(
            raw_config,
            source_path=st.session_state.get("inference_discover_source_path") or geotiff_path,
            selected_rasters=selected_inference_rasters,
            aoi=inference_aoi,
            registry_root=registry_root,
        )
        batch_fingerprint = batch_deployment_config_fingerprint(batch_payload)
        current_batch_job_id = batch_deployment_job_id(batch_payload)
        batch_cache_status = batch_output_cache_status(output_dir, batch_fingerprint)
        active_deployment_jobs = [
            job
            for job in list_job_states(registry_root, kind="deployment")
            if job.get("status") in {"pending", "running", "cancelling"}
        ]
        active_batch_deployment_jobs = [
            job
            for job in list_job_states(registry_root, kind=BATCH_DEPLOYMENT_JOB_KIND)
            if job.get("status") in {"pending", "running", "cancelling"}
        ]
        same_output_active = [
            job
            for job in [*active_deployment_jobs, *active_batch_deployment_jobs]
            if str((job.get("payload") or {}).get("output_dir") or "") == str(Path(output_dir).expanduser().resolve())
        ]
        if selected_inference_rasters and batch_cache_status["has_cache"] and not batch_cache_status["matches"]:
            with st.expander("Output folder contains a different batch deployment cache", expanded=True):
                st.warning(
                    "The output folder already contains intermediate files from a different batch deployment configuration. "
                    "Confirm cache cleanup before reusing the same output folder."
                )
                st.caption("Existing cache items: " + ", ".join(batch_cache_status["existing_items"]))
                confirm_batch_cache_clear = st.checkbox(
                    "I understand this will clear batch deployment files in the output folder",
                    value=False,
                    key=f"confirm_batch_clear_{batch_fingerprint}",
                )
        elif selected_inference_rasters and batch_cache_status["matches"]:
            st.info("The current batch configuration matches the output-folder cache. Click Generate shp again to continue from completed TIFF files.")
        elif not selected_inference_rasters and cache_status["has_cache"] and not cache_status["matches"]:
            with st.expander("Output folder contains a different deployment cache", expanded=True):
                st.warning(
                    "The output folder already contains intermediate files from a different configuration. "
                    "Confirm cache cleanup before reusing the same output folder."
                )
                st.caption("Existing cache items: " + ", ".join(cache_status["existing_items"]))
                if not cache_status["previous_fingerprint"]:
                    st.caption("This output directory has no verifiable deployment_cache.json, so the cache is treated as unknown.")
                confirm_cache_clear = st.checkbox(
                    "I understand this will clear intermediate deployment files in the output folder",
                    value=False,
                    key=f"confirm_clear_{current_fingerprint}",
                )
        elif cache_status["matches"]:
            st.info("The current configuration matches the output-folder cache. Click Generate shp again to resume from completed stages.")
        if action_cols[2].button("Reset Inference state", use_container_width=False):
            _reset_inference_session_state(st)
            st.rerun()
        _render_inference_current_config_summary(
            st,
            registry_root=registry_root,
            selected_weight_id=selected_weight_id,
            model=model,
            checkpoint_path=checkpoint_path,
            input_path=geotiff_path,
            output_dir=output_dir,
            batch_count=len(selected_inference_rasters),
            batch_aoi=inference_aoi,
        )
        st.session_state["lsd_app_defaults"] = {
            "geotiff_path": geotiff_path,
            "output_dir": output_dir,
            "checkpoint_path": checkpoint_path,
            "selected_weight_id": selected_weight_id,
            "model": model,
            "tile_size": int(tile_size),
            "stride": int(stride),
            "batch_size": int(batch_size),
            "threshold": float(threshold),
            "min_area_pixels": int(min_area_pixels),
            "RGB_bands": RGB_bands,
        }

        statuses: list[StageStatus] = []

        if run_everything:
            if selected_inference_rasters:
                preflight = preflight_batch_deployment_payload(
                    batch_payload,
                    selected_weight=selected_weight,
                    create_output_dirs=True,
                )
                _render_preflight(st, preflight)
                if preflight.ok:
                    if same_output_active and all(job.get("job_id") != current_batch_job_id for job in same_output_active):
                        st.error("Another deployment job is already running in this output folder. Stop the current job before starting a new configuration.")
                    elif batch_cache_status["has_cache"] and not batch_cache_status["matches"] and not confirm_batch_cache_clear:
                        st.error("The batch configuration has changed. Confirm cache cleanup before starting a new job, or choose a different Output Dir.")
                    else:
                        if batch_cache_status["has_cache"] and not batch_cache_status["matches"] and confirm_batch_cache_clear:
                            removed = clear_batch_deployment_cache(output_dir)
                            st.warning(f"Cleared batch deployment cache: {', '.join(removed) if removed else 'nothing to remove'}")
                        job_id = start_batch_deployment_background_job(
                            registry_root=registry_root,
                            payload=batch_payload,
                        )
                        st.session_state["active_batch_deployment_job_id"] = job_id
                        st.success(f"Batch deployment started: {job_id}")
                        st.rerun()
                else:
                    with st.expander("Raw batch deployment payload", expanded=False):
                        st.json(batch_payload)
            else:
                config, preflight = preflight_deployment_config(raw_config, selected_weight=selected_weight, create_output_dirs=True)
                _render_preflight(st, preflight)
                if config is not None and preflight.ok:
                    if same_output_active and all(job.get("job_id") != current_job_id for job in same_output_active):
                        st.error("Another deployment job is already running in this output folder. Stop the current job before starting a new configuration.")
                    elif cache_status["has_cache"] and not cache_status["matches"] and not confirm_cache_clear:
                        st.error("The configuration has changed. Confirm cache cleanup before starting a new job, or choose a different Output Dir.")
                    else:
                        if cache_status["has_cache"] and not cache_status["matches"] and confirm_cache_clear:
                            removed = clear_deployment_cache(config.output_dir)
                            st.warning(f"Cleared deployment cache: {', '.join(removed) if removed else 'nothing to remove'}")
                        job_id = start_deployment_background_job(
                            registry_root=registry_root,
                            raw_config=raw_config,
                            config=config,
                        )
                        st.session_state["active_deployment_job_id"] = job_id
                        st.success(f"Deployment started: {job_id}")
                        st.rerun()
                elif config is not None:
                    with st.expander("Raw deployment config", expanded=False):
                        st.json(raw_config)
        elif run_visual_preview:
            config, preflight = preflight_deployment_config(raw_config, selected_weight=selected_weight, create_output_dirs=True)
            _render_preflight(st, preflight)
            if config is not None and preflight.ok:
                status = run_stage("visualize", config)
                _render_statuses(st, [status])
                _render_artifacts(st, collect_artifacts(config))
        else:
            st.info("Choose a registered weight and input GeoTIFF/folder, then click Generate shp. Evaluation remains separate.")

        recovered_deployment_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind="deployment",
            session_key="active_deployment_job_id",
        )
        recovered_batch_deployment_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind=BATCH_DEPLOYMENT_JOB_KIND,
            session_key="active_batch_deployment_job_id",
        )
        selected_job_id = recovered_deployment_job_id or st.session_state.get("active_deployment_job_id") or current_job_id
        selected_batch_job_id = (
            recovered_batch_deployment_job_id
            or st.session_state.get("active_batch_deployment_job_id")
            or (current_batch_job_id if selected_inference_rasters else "")
        )
        latest_kind, latest_job_id = _latest_inference_monitor_choice(
            registry_root=registry_root,
            deployment_job_id=str(selected_job_id or ""),
            batch_job_id=str(selected_batch_job_id or ""),
            prefer_batch=bool(selected_inference_rasters),
            st=st,
        )
        if latest_job_id:
            if latest_kind == BATCH_DEPLOYMENT_JOB_KIND:
                _render_batch_deployment_job_monitor_auto(st, registry_root=registry_root, job_id=latest_job_id)
            else:
                _render_deployment_job_monitor_auto(st, registry_root=registry_root, job_id=latest_job_id)
        _render_clear_frontend_job_history_button(
            st,
            registry_root=registry_root,
            session_keys=("active_deployment_job_id", "active_batch_deployment_job_id"),
            job_kinds=(("active_deployment_job_id", "deployment"), ("active_batch_deployment_job_id", BATCH_DEPLOYMENT_JOB_KIND)),
            label="Clear previous Inference job display",
            button_key="clear_inference_frontend_job_history",
        )

    elif selected_section == "Evaluation":
        st.subheader("Evaluation")
        active_eval_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind="evaluation",
            session_key="active_evaluation_job_id",
        )
        active_eval_state = read_job_state(registry_root, active_eval_job_id) if active_eval_job_id else {}
        active_eval_running = bool(active_eval_state and _job_is_active(active_eval_state))
        if active_eval_running:
            st.info("An evaluation job is already running. The monitor below will keep tracking it even if you close and reopen the page.")
        datasets = [
            row
            for row in list_datasets(registry_root=registry_root)
            if row.get("dataset_class") == "Evaluation" and row.get("has_gt")
        ]
        weights = list_weights(registry_root=registry_root)
        if not datasets:
            st.warning("Register an Evaluation dataset before running evaluation.")
        dataset_labels = [row["dataset_id"] for row in datasets]
        weight_labels = _weight_id_options(weights)
        weight_format = _weight_format_func(weights)
        with st.form("evaluation_config"):
            selected_dataset_id = st.selectbox("Evaluation dataset", dataset_labels) if dataset_labels else ""
            selected_weight_id = st.selectbox("weight", weight_labels, format_func=weight_format) if weight_labels else ""
            eval_model = st.selectbox("Evaluation Model", MODEL_CHOICES, index=0)
            eval_output_dir = st.text_input("Evaluation Output Dir", "outputs/evaluation_run")
            eval_batch_size = st.number_input("Evaluation Batch Size", min_value=1, value=1, step=1)
            run_eval = st.form_submit_button("Run Evaluation")
        if run_eval and selected_dataset_id:
            running_job = _active_job_for_kind(registry_root, "evaluation")
            if running_job:
                st.session_state["active_evaluation_job_id"] = str(running_job.get("job_id") or "")
                st.warning("An Evaluation job is already running. Switched back to the current job monitor.")
                st.rerun()
            dataset = next(row for row in datasets if row["dataset_id"] == selected_dataset_id)
            weight = next((row for row in weights if row["weight_id"] == selected_weight_id), {})
            if weight.get("model_id") and weight.get("model_id") != eval_model:
                st.warning(
                    f"Selected weight is registered for {weight.get('model_id')}; "
                    f"current Evaluation Model selection is {eval_model}."
                )
            raw_eval_config = build_evaluation_dataset_payload(
                dataset_id=selected_dataset_id,
                weight_id=selected_weight_id,
                model_id=eval_model,
                output_dir=eval_output_dir,
                checkpoint_path=str(weight.get("path") or ""),
                batch_size=int(eval_batch_size),
                device="auto",
            )
            try:
                job_id = start_evaluation_background_job(
                    registry_root=registry_root,
                    raw_config=raw_eval_config,
                )
            except RUNNABLE_ERRORS as exc:
                st.error(f"Evaluation launch failed: {exc}")
                st.json(raw_eval_config)
            else:
                st.session_state["active_evaluation_job_id"] = job_id
                st.success(f"Evaluation started: {job_id}")
                st.rerun()
        active_eval_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind="evaluation",
            session_key="active_evaluation_job_id",
        )
        if active_eval_job_id:
            _render_evaluation_job_monitor_auto(st, registry_root=registry_root, job_id=str(active_eval_job_id))
        _render_evaluation_history_panel_auto(st, registry_root=registry_root)
        _render_clear_frontend_job_history_button(
            st,
            registry_root=registry_root,
            session_keys=("active_evaluation_job_id",),
            job_kinds=(("active_evaluation_job_id", "evaluation"),),
            label="Clear previous Evaluation job display",
            button_key="clear_evaluation_frontend_job_history",
        )


def _render_statuses(st: Any, statuses: list[StageStatus]) -> None:
    st.subheader("Stage status")
    rows = [
        {
            "stage": status.stage,
            "status": "ok" if status.ok else "failed",
            "message": status.message,
            "error_type": status.error_type or "",
        }
        for status in statuses
    ]
    st.dataframe(rows, use_container_width=True)
    for status in statuses:
        if status.ok and status.result:
            with st.expander(f"{status.stage} result", expanded=False):
                st.json(status.result)
        if not status.ok:
            st.error(f"{status.stage}: {status.error_type}: {status.message}")


def _render_preflight(st: Any, result: PreflightResult) -> None:
    """Render compact deployment preflight feedback."""

    for warning in result.warnings:
        st.warning(warning)
    for error in result.errors:
        st.error(error)


def _render_deployment_result(st: Any, *, statuses: list[StageStatus], artifacts: dict[str, Any]) -> None:
    """Show delivery artifacts first and keep pipeline internals tucked away."""

    failed = next((status for status in statuses if not status.ok), None)
    if failed is not None:
        st.error(f"Deployment failed at {failed.stage}: {failed.message}")
        _render_statuses(st, statuses)
        with st.expander("Deployment artifacts and intermediate files", expanded=False):
            _render_artifacts(st, artifacts)
        return

    paths = artifacts.get("paths", {})
    delivery_keys = ("landslides_shp", "landslides_gpkg", "mask_raster", "probability_raster")
    delivery_rows = [
        {"artifact": key, "path": paths[key]}
        for key in delivery_keys
        if key in paths
    ]
    if delivery_rows:
        st.success("Deployment completed. Landslide polygon outputs are ready.")
        st.dataframe(delivery_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Deployment finished, but final delivery files were not found in the expected output folders.")

    stats = artifacts.get("vector_stats")
    if isinstance(stats, Mapping):
        cols = st.columns(3)
        cols[0].metric("polygons", _metric_value(stats.get("polygon_count")))
        cols[1].metric("total area m2", _metric_value(stats.get("total_area_m2")))
        cols[2].metric("mean confidence", _metric_value(stats.get("mean_confidence")))

    with st.expander("Pipeline status", expanded=False):
        _render_statuses(st, statuses)
    with st.expander("All artifacts and intermediate files", expanded=False):
        _render_artifacts(st, artifacts)


def _run_deployment_with_progress(st: Any, config: PlatformConfig) -> list[StageStatus]:
    """Run deployment while rendering stage and overall progress."""

    progress_panel = st.empty()
    overall_bar = st.progress(0.0)
    stage_rows = {
        stage: {
            "stage": stage,
            "status": "pending",
            "percent": 0.0,
            "message": "",
            "processed": "",
            "total": "",
        }
        for stage in DEPLOYMENT_STAGES
    }
    stage_table = st.empty()
    progress_text = st.empty()

    def update(progress: Mapping[str, Any]) -> None:
        stage = str(progress.get("stage") or "")
        if stage in stage_rows:
            stage_rows[stage] = {
                "stage": stage,
                "status": progress.get("status", "running"),
                "percent": _format_progress_percent(float(progress.get("percent") or 0.0)),
                "message": progress.get("message", ""),
                "processed": progress.get("processed", ""),
                "total": progress.get("total", ""),
            }
        overall = min(max(float(progress.get("overall_percent") or 0.0), 0.0), 1.0)
        current_action = str(progress.get("current_action") or progress.get("status") or "running")
        current_reason = str(progress.get("current_reason") or "")
        message = str(progress.get("message") or "")
        progress_panel.markdown(
            _deployment_progress_html(
                stage=stage or "deployment",
                action=current_action,
                reason=current_reason,
                message=message,
                overall_percent=overall,
            ),
            unsafe_allow_html=True,
        )
        overall_bar.progress(overall)
        stage_table.dataframe(list(stage_rows.values()), use_container_width=True, hide_index=True)
        progress_text.caption(f"{_format_progress_percent(overall)} overall | {stage or 'deployment'} | {message}")

    update(
        {
            "stage": "deployment",
            "status": "pending",
            "overall_percent": 0.0,
            "current_action": "starting deployment",
            "message": "Waiting to start prepare -> infer -> mosaic -> vectorize.",
        }
    )
    statuses = run_all_with_progress(config, progress_callback=update)
    final_status = "completed" if all(status.ok for status in statuses) else "failed"
    update(
        {
            "stage": "deployment",
            "status": final_status,
            "overall_percent": 1.0,
            "current_action": final_status,
            "message": "Deployment finished." if final_status == "completed" else "Deployment stopped with an error.",
        }
    )
    return statuses


def _deployment_progress_html(
    *,
    stage: str,
    action: str,
    reason: str,
    message: str,
    overall_percent: float,
) -> str:
    reason_text = f"<span>{reason}</span>" if reason else ""
    return f"""
    <div class="taft-import-panel">
      <div class="taft-import-head">
        <strong>{stage}: {action}</strong>
        <span>{_format_progress_percent(overall_percent)}</span>
      </div>
      <div class="taft-progress-track" aria-label="deployment progress">
        <div class="taft-progress-fill" style="width: {_format_progress_percent(overall_percent)};"></div>
      </div>
      <div class="taft-import-message">{message}</div>
      <div class="taft-import-stats">{reason_text}</div>
    </div>
    """


def _render_deployment_job_monitor(st: Any, *, registry_root: str, job_id: str) -> None:
    state = read_job_state(registry_root, job_id)
    if not state:
        latest = _latest_deployment_job(registry_root)
        if latest is not None:
            st.caption(f"Latest deployment job: {latest.get('job_id')} ({latest.get('status')})")
        return
    percent = min(max(float(state.get("overall_percent", state.get("percent", 0.0)) or 0.0), 0.0), 1.0)
    status = str(state.get("status") or "unknown")
    stage = str(state.get("stage") or "deployment")
    message = str(state.get("message") or "")
    action = str(state.get("current_action") or status)
    reason = str(state.get("current_reason") or "")
    st.markdown(
        _deployment_progress_html(
            stage=stage,
            action=action,
            reason=reason,
            message=message,
            overall_percent=percent,
        ),
        unsafe_allow_html=True,
    )
    st.progress(percent)
    caption = f"{_format_progress_percent(percent)} overall | status {status} | job {job_id}"
    stale_caption = _job_stale_caption(state)
    if stale_caption:
        caption = f"{caption} | {stale_caption}"
    st.caption(caption)
    _render_job_health_notice(st, state)
    _render_job_payload_summary(st, state)
    _render_inference_speed_summary(st, state)
    stage_rows = _deployment_stage_rows_from_job(state)
    if stage_rows:
        with st.expander("Stage details", expanded=status in {"failed", "cancelled", "interrupted"}):
            st.dataframe(stage_rows, use_container_width=True, hide_index=True)
    if status in {"running", "pending", "cancelling"}:
        if st.button("Stop current deployment", type="secondary", use_container_width=False, key=f"cancel_{job_id}"):
            if request_job_cancel(registry_root, job_id):
                st.warning("Stop requested. The current patch/batch will finish safely before the job stops.")
                st.rerun()
    if status == "completed":
        config = _config_from_deployment_job_state(state)
        result = state.get("result")
        artifacts = result.get("artifacts") if isinstance(result, Mapping) else None
        if config is not None:
            artifacts = collect_artifacts(config)
        if isinstance(artifacts, dict):
            statuses = _stage_statuses_from_job_result(result) if isinstance(result, Mapping) else []
            _render_deployment_result(st, statuses=statuses or _completed_stage_statuses(artifacts), artifacts=artifacts)
    elif status == "failed":
        st.error(f"Deployment failed: {state.get('error') or message}")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="deployment")
    elif status == "cancelled":
        st.warning("Deployment was stopped. Resume will continue from completed cached stages in the same output folder.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="deployment")
    elif status == "interrupted":
        st.warning("Deployment was interrupted when the app stopped. Resume will continue from completed cached stages in the same output folder.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="deployment")


def _render_batch_deployment_job_monitor(st: Any, *, registry_root: str, job_id: str) -> None:
    state = read_job_state(registry_root, job_id)
    if not state:
        latest = _latest_batch_deployment_job(registry_root)
        if latest is not None:
            st.caption(f"Latest batch deployment job: {latest.get('job_id')} ({latest.get('status')})")
        return
    percent = min(max(float(state.get("overall_percent", state.get("percent", 0.0)) or 0.0), 0.0), 1.0)
    status = str(state.get("status") or "unknown")
    message = str(state.get("message") or "")
    action = str(state.get("current_action") or state.get("item_status") or status)
    reason = str(state.get("current_raster") or state.get("current_reason") or "")
    st.markdown(
        _deployment_progress_html(
            stage="batch deployment",
            action=action,
            reason=reason,
            message=message,
            overall_percent=percent,
        ),
        unsafe_allow_html=True,
    )
    st.progress(percent)
    completed = _int_value(state.get("completed"))
    total = _int_value(state.get("total"))
    caption = f"{_format_progress_percent(percent)} overall | status {status} | completed {completed}/{total or '?'} | job {job_id}"
    stale_caption = _job_stale_caption(state)
    if stale_caption:
        caption = f"{caption} | {stale_caption}"
    st.caption(caption)
    _render_job_health_notice(st, state)
    _render_job_payload_summary(st, state)
    _render_inference_speed_summary(st, state)
    _render_current_batch_item_progress(st, state)
    rows = _batch_deployment_rows_from_job_state(state)
    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "progress": st.column_config.ProgressColumn("progress", min_value=0.0, max_value=1.0, format="%.1f"),
                "output_dir": st.column_config.TextColumn("output_dir", width="large"),
            },
        )
    if status in {"running", "pending", "cancelling"}:
        if st.button("Stop current batch deployment", type="secondary", key=f"cancel_batch_{job_id}"):
            if request_job_cancel(registry_root, job_id):
                st.warning("Stop requested. Resume will continue from completed rasters and cached stages.")
                st.rerun()
    if status == "completed":
        _render_batch_deployment_result(st, _batch_artifacts_from_job_state(state))
    elif status == "failed":
        st.error(f"Batch deployment failed: {state.get('error') or message}")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind=BATCH_DEPLOYMENT_JOB_KIND)
    elif status == "cancelled":
        st.warning("Batch deployment was stopped. Resume will continue from completed raster outputs.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind=BATCH_DEPLOYMENT_JOB_KIND)
    elif status == "interrupted":
        st.warning("Batch deployment was interrupted when the app stopped. Resume will continue from completed raster outputs.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind=BATCH_DEPLOYMENT_JOB_KIND)


def _render_pipeline_resume_controls(st: Any, *, registry_root: str, state: Mapping[str, Any], kind: str) -> None:
    if kind == BATCH_DEPLOYMENT_JOB_KIND:
        payload = state.get("payload")
        if not isinstance(payload, Mapping) or not payload.get("selected_rasters"):
            st.info("No recoverable batch deployment payload was saved for this job.")
            return
        key = f"resume_{kind}_{state.get('job_id') or ''}"
        if st.button("Resume batch deployment", type="secondary", key=key):
            try:
                resumed_job_id = start_batch_deployment_background_job(
                    registry_root=registry_root,
                    payload=dict(payload),
                )
            except RUNNABLE_ERRORS as exc:
                st.error(str(exc))
            else:
                st.session_state["active_batch_deployment_job_id"] = resumed_job_id
                st.success(f"Resume batch deployment started: {resumed_job_id}")
                st.rerun()
        return
    if kind == "evaluation":
        payload = state.get("payload")
        if isinstance(payload, Mapping) and payload.get("dataset_id"):
            label = "Resume evaluation"
            key = f"resume_{kind}_{state.get('job_id') or ''}"
            if st.button(label, type="secondary", key=key):
                try:
                    resumed_job_id = start_evaluation_background_job(
                        registry_root=registry_root,
                        raw_config=dict(payload),
                    )
                except RUNNABLE_ERRORS as exc:
                    st.error(str(exc))
                else:
                    st.session_state["active_evaluation_job_id"] = resumed_job_id
                    st.success(f"{label} started: {resumed_job_id}")
                    st.rerun()
            return
    raw_config = _raw_config_from_pipeline_job_state(state)
    if not raw_config:
        st.info("No recoverable configuration was saved for this job.")
        return
    try:
        config = normalize_web_config(raw_config, create_output_dirs=True)
    except ConfigError as exc:
        st.error(f"Cannot resume {kind}: {exc}")
        return
    label = "Resume deployment" if kind == "deployment" else "Resume evaluation"
    key = f"resume_{kind}_{state.get('job_id') or ''}"
    if st.button(label, type="secondary", key=key):
        try:
            if kind == "deployment":
                resumed_job_id = start_deployment_background_job(
                    registry_root=registry_root,
                    raw_config=raw_config,
                    config=config,
                )
                st.session_state["active_deployment_job_id"] = resumed_job_id
            elif kind == "evaluation":
                resumed_job_id = start_evaluation_background_job(
                    registry_root=registry_root,
                    raw_config=raw_config,
                    config=config,
                )
                st.session_state["active_evaluation_job_id"] = resumed_job_id
            else:
                st.error(f"Unsupported resume job kind: {kind}")
                return
        except RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.success(f"{label} started: {resumed_job_id}")
            st.rerun()


def _latest_deployment_job(registry_root: str) -> Mapping[str, Any] | None:
    return _latest_job_for_kind(registry_root, "deployment")


def _latest_batch_deployment_job(registry_root: str) -> Mapping[str, Any] | None:
    return _latest_job_for_kind(registry_root, BATCH_DEPLOYMENT_JOB_KIND)


def _latest_inference_monitor_choice(
    *,
    registry_root: str,
    deployment_job_id: str,
    batch_job_id: str,
    prefer_batch: bool,
    st: Any | None = None,
) -> tuple[str, str]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    session_keys = {
        "deployment": "active_deployment_job_id",
        BATCH_DEPLOYMENT_JOB_KIND: "active_batch_deployment_job_id",
    }
    for kind, job_id in (("deployment", deployment_job_id), (BATCH_DEPLOYMENT_JOB_KIND, batch_job_id)):
        if job_id:
            state = read_job_state(registry_root, job_id)
            if state:
                session_key = session_keys[kind]
                if st is not None and _job_monitor_id_is_hidden(st, session_key=session_key, job_id=job_id, state=state):
                    continue
                candidates.append((kind, state))
    if not candidates:
        for kind in ("deployment", BATCH_DEPLOYMENT_JOB_KIND):
            latest = _latest_job_for_kind(registry_root, kind)
            if latest:
                session_key = session_keys[kind]
                if st is not None and _job_monitor_is_hidden(st, session_key=session_key, job=latest):
                    continue
                candidates.append((kind, dict(latest)))
    if not candidates:
        return "", ""
    active = [(kind, state) for kind, state in candidates if _job_is_active(state)]
    if active:
        preferred = BATCH_DEPLOYMENT_JOB_KIND if prefer_batch else "deployment"
        same_kind = [(kind, state) for kind, state in active if kind == preferred]
        pool = same_kind or active
    else:
        pool = candidates
    kind, state = sorted(pool, key=lambda item: str(item[1].get("updated_at", "")), reverse=True)[0]
    return kind, str(state.get("job_id") or "")


def _render_deployment_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _deployment_fragment() -> None:
            _render_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _deployment_fragment()
    else:
        _render_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)
        _schedule_active_job_refresh(st, registry_root=registry_root, job_id=job_id)


def _render_batch_deployment_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _batch_deployment_fragment() -> None:
            _render_batch_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _batch_deployment_fragment()
    else:
        _render_batch_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)
        _schedule_active_job_refresh(st, registry_root=registry_root, job_id=job_id)


def _render_current_batch_item_progress(st: Any, state: Mapping[str, Any]) -> None:
    current_raster = str(state.get("current_raster") or "").strip()
    if not current_raster:
        return
    item_stage = str(state.get("item_stage") or "").strip() or "item"
    processed = _int_value(state.get("item_stage_processed"))
    total = _int_value(state.get("item_stage_total"))
    percent = _batch_current_item_progress(state)
    raster_name = Path(current_raster).name
    st.progress(percent, text=f"{raster_name} | {item_stage} | {_format_progress_percent(percent)}")
    count_text = f"{processed}/{total}" if total else str(processed or "")
    detail = str(state.get("current_action") or state.get("message") or "").strip()
    reason = str(state.get("current_reason") or "").strip()
    parts = [part for part in (count_text, detail, reason) if part]
    if parts:
        st.caption(" | ".join(parts))


def _batch_current_item_progress(state: Mapping[str, Any]) -> float:
    for key in ("item_stage_percent", "item_stage_overall_percent"):
        value = _float_or_none(state.get(key))
        if value is not None:
            return min(max(value, 0.0), 1.0)
    processed = _int_value(state.get("item_stage_processed"))
    total = _int_value(state.get("item_stage_total"))
    if total > 0:
        return min(max(processed / total, 0.0), 1.0)
    return 0.0


def _render_job_payload_summary(st: Any, state: Mapping[str, Any]) -> None:
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    output_dir = str(payload.get("output_dir") or _nested_value(raw_config, ("output", "dir"), default="")).strip()
    checkpoint = str(_nested_value(raw_config, ("model", "checkpoint_path"), default="") or payload.get("checkpoint_path") or "").strip()
    input_path = str(_nested_value(raw_config, ("input", "raster_path"), default="") or payload.get("source_path") or payload.get("input_raster_path") or "").strip()
    registry_root_value = str(payload.get("registry_root") or raw_config.get("registry_root") or "").strip()
    rows = [
        {"field": "input", "value": input_path},
        {"field": "output", "value": output_dir},
        {"field": "checkpoint", "value": checkpoint},
        {"field": "registry", "value": registry_root_value},
    ]
    rows = [row for row in rows if row["value"]]
    if rows:
        with st.expander("Running job configuration", expanded=False):
            st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_evaluation_job_monitor(st: Any, *, registry_root: str, job_id: str) -> None:
    state = read_job_state(registry_root, job_id)
    if not state:
        return
    percent_key = "percent" if _is_sample_evaluation_state(state) else "overall_percent"
    percent = min(max(float(state.get(percent_key, state.get("percent", 0.0)) or 0.0), 0.0), 1.0)
    status = str(state.get("status") or "unknown")
    stage = str(state.get("stage") or "evaluation")
    message = str(state.get("message") or "")
    action = str(state.get("current_action") or status)
    reason = str(state.get("current_reason") or "")
    st.markdown(
        _deployment_progress_html(
            stage=stage,
            action=action,
            reason=reason,
            message=message,
            overall_percent=percent,
        ),
        unsafe_allow_html=True,
    )
    st.progress(percent)
    caption = f"{_format_progress_percent(percent)} overall | status {status} | job {job_id}"
    stale_caption = _job_stale_caption(state)
    if stale_caption:
        caption = f"{caption} | {stale_caption}"
    st.caption(caption)
    _render_job_health_notice(st, state)
    stage_rows = _evaluation_stage_rows_from_job(state)
    if stage_rows:
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)
    if status in {"running", "pending", "cancelling"}:
        if st.button("Stop evaluation", type="secondary", key=f"cancel_eval_{job_id}"):
            if request_job_cancel(registry_root, job_id):
                st.warning("Stop requested. The current evaluation step will finish safely before the job stops.")
                st.rerun()
    if status == "completed":
        st.success("Evaluation completed. The latest metrics are recorded in Evaluation History below.")
    elif status == "failed":
        st.error(f"Evaluation failed: {state.get('error') or message}")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="evaluation")
    elif status == "cancelled":
        st.warning("Evaluation was stopped. Resume will continue from completed cached stages in the same output folder.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="evaluation")
    elif status == "interrupted":
        st.warning("Evaluation was interrupted when the app stopped. Resume will continue from completed cached stages in the same output folder.")
        _render_pipeline_resume_controls(st, registry_root=registry_root, state=state, kind="evaluation")


def _render_evaluation_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _evaluation_fragment() -> None:
            _render_evaluation_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _evaluation_fragment()
    else:
        _render_evaluation_job_monitor(st, registry_root=registry_root, job_id=job_id)
        _schedule_active_job_refresh(st, registry_root=registry_root, job_id=job_id)


def _render_evaluation_history_panel_auto(st: Any, *, registry_root: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _evaluation_history_fragment() -> None:
            _render_evaluation_history_panel(st, registry_root=registry_root)

        _evaluation_history_fragment()
    else:
        _render_evaluation_history_panel(st, registry_root=registry_root)


def _render_daft_training_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _daft_training_fragment() -> None:
            _render_daft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _daft_training_fragment()
    else:
        _render_daft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)
        _schedule_active_job_refresh(st, registry_root=registry_root, job_id=job_id)


def _render_tsft_training_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _tsft_training_fragment() -> None:
            _render_tsft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _tsft_training_fragment()
    else:
        _render_tsft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)
        _schedule_active_job_refresh(st, registry_root=registry_root, job_id=job_id)


def _evaluation_stage_rows_from_job(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = state.get("payload")
    if isinstance(payload, Mapping) and payload.get("dataset_id"):
        phase = str(state.get("phase") or state.get("stage") or "evaluate")
        raw_percent = state.get("percent")
        if raw_percent in {None, ""}:
            raw_percent = state.get("overall_percent", 0.0)
        return [
            {
                "stage": "evaluate dataset",
                "status": state.get("status", ""),
                "percent": _format_progress_percent(float(raw_percent or 0.0)),
                "message": state.get("message", ""),
                "processed": state.get("processed", state.get("eval_current_batch", "")),
                "total": state.get("total", state.get("eval_total_batches", "")),
                "phase": phase,
            }
        ]
    active_stage = str(state.get("stage") or "")
    stages = (*DEPLOYMENT_STAGES, "evaluate")
    rows: list[dict[str, Any]] = []
    active_index = stages.index(active_stage) if active_stage in stages else -1
    for index, stage in enumerate(stages):
        if index < active_index:
            status = "completed"
            percent = "100.00%"
        elif stage == active_stage:
            status = str(state.get("status") or "running")
            percent = _format_progress_percent(float(state.get("percent") or 0.0))
        else:
            status = "pending"
            percent = "0.00%"
        rows.append(
            {
                "stage": stage,
                "status": status,
                "percent": percent,
                "message": state.get("message", "") if stage == active_stage else "",
                "processed": state.get("processed", "") if stage == active_stage else "",
                "total": state.get("total", "") if stage == active_stage else "",
            }
        )
    return rows


def _is_sample_evaluation_state(state: Mapping[str, Any]) -> bool:
    payload = state.get("payload")
    return isinstance(payload, Mapping) and bool(payload.get("dataset_id"))


def _completed_evaluation_statuses(artifacts: dict[str, Any]) -> list[StageStatus]:
    return [
        StageStatus(stage=stage, ok=True, message="Completed or reused.", artifacts=artifacts)
        for stage in (*DEPLOYMENT_STAGES, "evaluate")
    ]


def _deployment_stage_rows_from_job(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_stage = str(state.get("stage") or "")
    rows: list[dict[str, Any]] = []
    active_index = DEPLOYMENT_STAGES.index(active_stage) if active_stage in DEPLOYMENT_STAGES else -1
    for index, stage in enumerate(DEPLOYMENT_STAGES):
        if index < active_index:
            status = "completed"
            percent = "100.00%"
        elif stage == active_stage:
            status = str(state.get("status") or "running")
            percent = _format_progress_percent(float(state.get("percent") or 0.0))
        else:
            status = "pending"
            percent = "0.00%"
        rows.append(
            {
                "stage": stage,
                "status": status,
                "percent": percent,
                "message": state.get("message", "") if stage == active_stage else "",
                "processed": state.get("processed", "") if stage == active_stage else "",
                "total": state.get("total", "") if stage == active_stage else "",
            }
        )
    return rows


def _batch_deployment_rows_from_job_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = _batch_artifacts_from_job_state(state)
    items = artifacts.get("items") if isinstance(artifacts, Mapping) else []
    if not isinstance(items, list):
        return []
    rows: list[dict[str, Any]] = []
    current_item_id = str(state.get("current_item_id") or "")
    for item in items:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "pending")
        if current_item_id and str(item.get("item_id") or "") == current_item_id and state.get("item_status"):
            status = str(state.get("item_status"))
        progress = _batch_item_row_progress(item, state=state, is_current=bool(current_item_id and str(item.get("item_id") or "") == current_item_id))
        rows.append(
            {
                "index": item.get("source_index", ""),
                "raster": item.get("raster_name", Path(str(item.get("raster_path") or "")).name),
                "status": status,
                "progress": progress,
                "polygons": item.get("polygon_count", ""),
                "output_dir": item.get("output_dir", ""),
            }
        )
    return rows


def _batch_item_row_progress(item: Mapping[str, Any], *, state: Mapping[str, Any], is_current: bool) -> float:
    if is_current:
        return _batch_current_item_progress(state)
    status = str(item.get("status") or "")
    if status in {"completed", "completed-empty", "skipped"}:
        return 1.0
    if status == "failed":
        value = _float_or_none(item.get("progress"))
        return min(max(value if value is not None else 0.0, 0.0), 1.0)
    return 0.0


def _batch_artifacts_from_job_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = state.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("artifacts"), Mapping):
        return dict(result["artifacts"])
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    output_dir = str(payload.get("output_dir") or "")
    if not output_dir and isinstance(payload.get("raw_config"), Mapping):
        output_dir = str(_nested_value(payload["raw_config"], ("output", "dir"), default=""))
    if output_dir:
        return collect_batch_artifacts(output_dir)
    return {}


def _render_batch_deployment_result(st: Any, artifacts: dict[str, Any]) -> None:
    paths = artifacts.get("paths") if isinstance(artifacts.get("paths"), Mapping) else {}
    delivery_keys = ("merged_shp", "merged_gpkg", "merged_geojson")
    delivery_rows = [
        {"artifact": key, "path": paths[key]}
        for key in delivery_keys
        if key in paths
    ]
    if delivery_rows:
        st.success("Batch deployment completed. Merged landslide polygon outputs are ready.")
        st.dataframe(delivery_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Batch deployment finished, but merged vector outputs were not found.")
    items = artifacts.get("items")
    if isinstance(items, list) and items:
        st.dataframe(_batch_deployment_rows_from_job_state({"result": {"artifacts": artifacts}}), use_container_width=True, hide_index=True)
    with st.expander("Batch deployment manifest", expanded=False):
        st.json(artifacts.get("manifest", {}))


def _stage_statuses_from_job_result(result: Mapping[str, Any]) -> list[StageStatus]:
    statuses = result.get("statuses")
    if not isinstance(statuses, list):
        return []
    parsed: list[StageStatus] = []
    for row in statuses:
        if not isinstance(row, Mapping):
            continue
        parsed.append(
            StageStatus(
                stage=str(row.get("stage") or ""),
                ok=bool(row.get("ok", False)),
                message=str(row.get("message") or ""),
                result=row.get("result") if isinstance(row.get("result"), dict) else None,
                artifacts=row.get("artifacts") if isinstance(row.get("artifacts"), dict) else None,
                error_type=str(row.get("error_type") or "") or None,
            )
        )
    return parsed


def _completed_stage_statuses(artifacts: dict[str, Any]) -> list[StageStatus]:
    return [
        StageStatus(stage=stage, ok=True, message="Completed or reused.", artifacts=artifacts)
        for stage in DEPLOYMENT_STAGES
    ]


def _check_output_dir_writable(output_dir: Path) -> str | None:
    """Return a validation error if the deployment output directory cannot be written."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".lsd_app_write_check_", dir=output_dir, delete=False) as handle:
            check_path = Path(handle.name)
            handle.write(b"ok")
        check_path.unlink(missing_ok=True)
    except OSError as exc:
        return f"Output directory is not writable: {output_dir} ({exc})"
    return None


def _metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _weight_repository_table(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the compact weight repository view shown in the UI."""

    return [
        {
            "weight_id": row.get("weight_id", ""),
            "Pretrained": row.get("version", ""),
            "stage": row.get("stage", ""),
            "registered_model": row.get("model_id", ""),
            "weight_size": _format_bytes(row.get("file_size_bytes")),
            "parameter_count": _format_count(row.get("parameter_count")),
            "registered_at": row.get("created_at", ""),
            "stored_path": row.get("path", ""),
        }
        for row in rows
    ]


def _model_repository_table(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the compact model repository view shown in the UI."""

    table: list[dict[str, Any]] = []
    for row in rows:
        item = {
            str(key): value
            for key, value in row.items()
            if str(key) not in MODEL_REPOSITORY_HIDDEN_COLUMNS
        }
        model_id = str(row.get("model_id") or row.get("name") or "").strip()
        if model_id == "LSDFormer":
            item["description"] = LSD_FORMER_DISPLAY_DESCRIPTION
        table.append(item)
    return table


def _render_weight_add_form(
    st: Any,
    *,
    registry_root: str,
    model_options: Sequence[str],
    locked_model_id: str | None = None,
    form_key: str = "weight_registry",
) -> None:
    st.markdown('<div class="module-title">Add Weight</div>', unsafe_allow_html=True)
    selected_model_id = locked_model_id or ""
    with st.form(form_key, border=False):
        if locked_model_id:
            st.text_input("Model ID", locked_model_id, disabled=True, key=f"{form_key}_model_id_locked")
            weight_model_id = locked_model_id
        else:
            options = list(model_options)
            default_index = options.index("LSDSAM-B") if "LSDSAM-B" in options else 0
            weight_model_id = st.selectbox("Model ID", options, index=default_index, key=f"{form_key}_model_id")
        selected_model_id = str(weight_model_id)
        weight_stage = st.selectbox("Weight Stage", WEIGHT_STAGES, key=f"{form_key}_stage")
        weight_id = st.text_input("Weight ID", "", key=f"{form_key}_weight_id")
        weight_path = st.text_input("Weight Path", "", key=f"{form_key}_path")
        add_weight = st.form_submit_button("Add Weight")
    if add_weight:
        progress_bar = st.progress(0, text="Preparing weight upload...")
        progress_status = st.empty()

        def update_weight_upload_progress(copied: int, total: int) -> None:
            ratio = 1.0 if total <= 0 else min(1.0, copied / total)
            progress_bar.progress(
                ratio,
                text=f"Uploading weight: {_format_bytes(copied)} / {_format_bytes(total)}",
            )
            progress_status.caption(f"{ratio * 100:.1f}% copied into the internal weight repository")

        try:
            entry = register_weight(
                registry_root=registry_root,
                model_id=selected_model_id,
                stage=weight_stage,
                version=Path(weight_path).expanduser().stem if weight_path.strip() else "",
                path=weight_path,
                weight_id=weight_id,
                progress_callback=update_weight_upload_progress,
            )
            progress_bar.progress(1.0, text="Weight upload completed")
            st.success(f"Weight registered: {entry.weight_id}")
            _request_streamlit_rerun(st)
        except RegistryError as exc:
            progress_bar.empty()
            progress_status.empty()
            st.error(str(exc))


def _render_weight_repository_filters(st: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = list(rows)
    query = st.text_input("Search Weights", "", key="weight_repo_search")
    if query.strip():
        terms = [part.lower() for part in query.split() if part.strip()]
        searchable_fields = ("weight_id", "model_id", "stage", "path", "source_path", "created_at")
        filtered = [
            row
            for row in filtered
            if all(
                any(term in str(row.get(field, "")).lower() for field in searchable_fields)
                for term in terms
            )
        ]

    controls = st.columns(3)
    filter_fields = [
        ("model_id", "Model ID"),
        ("stage", "Stage"),
        ("exists", "File Status"),
    ]
    for index, (field, label) in enumerate(filter_fields):
        if field == "exists":
            options = ["All", "available", "missing"]
            selected = controls[index].selectbox(label, options, key=f"weight_repo_filter_{field}")
            if selected != "All":
                expected = selected == "available"
                filtered = [row for row in filtered if bool(row.get("exists")) is expected]
            continue
        options = ["All", *_unique_values(filtered, field)]
        selected = controls[index].selectbox(label, options, key=f"weight_repo_filter_{field}")
        if selected != "All":
            filtered = [row for row in filtered if str(row.get(field, "")) == selected]
    return filtered


def _render_weight_delete_panel(
    st: Any,
    *,
    registry_root: str,
    rows: list[dict[str, Any]],
    form_key: str = "delete_weight_entry",
    title: str = "Delete weight",
    confirmation_label: str = "Delete this weight from the internal repository",
) -> None:
    st.markdown(f'<div class="module-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if not rows:
        st.info("No weight matches the current filters.")
        return

    options = _weight_id_options(rows)
    if not options:
        st.info("No deletable Weight ID is available.")
        return
    weight_format = _weight_format_func(rows)

    with st.form(form_key, border=False):
        selected_weight_id = st.selectbox(
            "Weight ID To Delete",
            options,
            key=f"{form_key}_weight_id",
            format_func=weight_format,
        )
        confirm_delete = st.checkbox(confirmation_label, key=f"{form_key}_confirm")
        delete_clicked = st.form_submit_button("Delete Weight")
    if delete_clicked:
        if not confirm_delete:
            st.warning("Please confirm deletion before removing a weight.")
            return
        try:
            result = delete_weight(registry_root=registry_root, weight_id=selected_weight_id)
        except RegistryError as exc:
            st.error(str(exc))
            return
        removed_path = result.get("removed_path") or "registry entry only"
        st.success(f"Deleted weight {result['weight_id']}; removed {removed_path}.")
        st.rerun()


def _render_evaluation_history_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### Evaluation History")
    rows = _read_evaluation_history(registry_root)
    if not rows:
        st.info("No evaluation history has been recorded yet.")
        return

    query = st.text_input("Search evaluation history", "", key="evaluation_history_search")
    filtered_rows = _filter_evaluation_history(rows, query)
    table_rows = _evaluation_history_table(filtered_rows)
    actions_left, actions_right = st.columns((1.0, 2.0))
    with actions_left:
        st.download_button(
            "Export evaluation history CSV",
            data=_evaluation_history_csv(filtered_rows),
            file_name="evaluation_history.csv",
            mime="text/csv",
            disabled=not filtered_rows,
        )
    with actions_right:
        st.caption(f"{len(filtered_rows)} / {len(rows)} history entries")
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    delete_options = [str(row.get("history_id") or "") for row in filtered_rows if row.get("history_id")]
    if not delete_options:
        return
    labels = {
        str(row.get("history_id") or ""): (
            f"{row.get('created_at', '')} | {row.get('weight_version') or row.get('weight_id', '')} "
            f"on {row.get('dataset_id', '')}"
        )
        for row in filtered_rows
        if row.get("history_id")
    }
    with st.form("delete_evaluation_history_entry"):
        selected_history_id = st.selectbox(
            "evaluation history entry to delete",
            delete_options,
            format_func=lambda value: labels.get(str(value), str(value)),
        )
        confirm_delete = st.checkbox("Delete this history entry")
        delete_clicked = st.form_submit_button("Delete Evaluation History Entry")
    if delete_clicked:
        if not confirm_delete:
            st.warning("Please confirm deletion before removing the history entry.")
            return
        if _delete_evaluation_history_entry(registry_root, selected_history_id):
            st.success("Evaluation history entry deleted.")
            st.rerun()
        else:
            st.warning("The selected evaluation history entry was not found.")


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "-"
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


def _format_count(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "-"
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def _render_training_placeholder(st: Any, *, registry_root: str, stage: str) -> None:
    datasets = [
        row
        for row in list_datasets(registry_root=registry_root)
        if row.get("dataset_class") in (stage, "TAFT")
    ]
    weights = list_weights(registry_root=registry_root)
    model_ids = [row["model_id"] for row in list_models(project_root=Path.cwd())]
    weight_options = _weight_id_options(weights)
    weight_format = _weight_format_func(weights)
    with st.form(f"{stage.lower()}_dry_run"):
        model_id = st.selectbox("Model ID", model_ids) if model_ids else ""
        weight_id = st.selectbox("Base Weight", weight_options, format_func=weight_format) if weight_options else ""
        dataset_id = st.selectbox("Dataset", [row["dataset_id"] for row in datasets]) if datasets else ""
        output_dir = st.text_input("Training Output Dir", f"outputs/{stage.lower()}_dry_run")
        submitted = st.form_submit_button(f"Build {stage} Dry-run")
    st.info(f"{stage} interface is reserved. Real training launch is disabled in this version.")
    if submitted:
        st.json(
            build_training_dry_run(
                stage=stage,
                model_id=model_id,
                weight_id=weight_id,
                dataset_id=dataset_id,
                output_dir=output_dir,
            )
        )


def _render_daft_training_panel(st: Any, *, registry_root: str) -> None:
    dataset_rows = list_datasets(registry_root=registry_root)
    weights = list_weights(registry_root=registry_root)
    source_rows = _training_dataset_rows_for_class(dataset_rows, "TAFT", require_gt=True)
    target_rows = _training_dataset_rows_for_class(dataset_rows, "DAFT")
    evaluation_rows = [
        row
        for row in dataset_rows
        if row.get("dataset_class") == "Evaluation" and row.get("has_gt")
    ]
    weight_rows = [row for row in weights if row.get("exists", True)]
    st.markdown(
        """
        <div class="deployment-panel">
          <h3>DAFT Training</h3>
          <p>Domain-adversarial fine-tuning uses the registered source and target repositories. Training device is selected automatically: GPU first, CPU fallback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Device Policy: {DAFT_TRAINING_DEVICE_POLICY}")
    if not source_rows:
        st.warning("Register a TAFT/TSFT source dataset with labels before launching DAFT.")
    if not target_rows:
        st.warning("Register a DAFT or TSFT target dataset before launching DAFT.")
    if not weight_rows:
        st.warning("Register a base weight before launching DAFT.")
    active_daft_job_id = _recover_active_job_id(
        st,
        registry_root=registry_root,
        kind="daft-training",
        session_key="active_daft_training_job_id",
    )
    active_daft_state = read_job_state(registry_root, active_daft_job_id) if active_daft_job_id else {}
    active_daft_running = bool(active_daft_state and _job_is_live_blocking(active_daft_state))
    if active_daft_running:
        st.info("A DAFT training job is already running. The monitor below will keep tracking it even if you close and reopen the page.")

    selector_left, selector_middle, _selector_right = st.columns(3)
    with selector_left:
        model_id = st.selectbox("model", MODEL_CHOICES, index=0, key="daft_model", disabled=active_daft_running)
    model_weight_rows = _training_weight_rows_for_model(weight_rows, model_id)
    base_weight_options = _weight_id_options(model_weight_rows)
    base_weight_format = _weight_format_func(model_weight_rows)
    model_key = _training_model_widget_key(model_id)
    with selector_middle:
        if base_weight_options:
            base_weight_id = st.selectbox(
                "base weight",
                base_weight_options,
                key=f"daft_base_weight_{model_key}",
                format_func=base_weight_format,
                disabled=active_daft_running,
            )
        else:
            st.selectbox(
                "base weight",
                ["No registered weight"],
                key=f"daft_base_weight_{model_key}_empty",
                disabled=True,
            )
            base_weight_id = ""
    if weight_rows and not base_weight_options:
        st.warning(f"No registered base weight is available for {model_id}. Check the Weight Repository model_id or register a matching weight.")

    with st.form("daft_lsdsam_training"):
        left, middle, right = st.columns(3)
        with left:
            source_dataset_id = st.selectbox(
                "source dataset",
                [row["dataset_id"] for row in source_rows],
                disabled=not source_rows,
                format_func=_dataset_id_option_label,
            ) if source_rows else ""
        with middle:
            target_dataset_id = st.selectbox(
                "target dataset",
                [row["dataset_id"] for row in target_rows],
                disabled=not target_rows,
                format_func=_dataset_id_option_label,
            ) if target_rows else ""
            alignment_mode = st.selectbox("Constraint", DAFT_ALIGNMENT_MODES, index=0)
            evaluation_dataset_id = st.selectbox(
                "Evaluation Dataset",
                ["None", *[row["dataset_id"] for row in evaluation_rows]],
                key="daft_evaluation_dataset",
            )
            evaluation_dataset_id = "" if evaluation_dataset_id == "None" else evaluation_dataset_id
            output_dir = st.text_input("Output Dir", "outputs/daft_training")
            output_weight_id = st.text_input(
                "DAFT Weight ID",
                "",
                help="Optional. Leave blank to let LSDetector generate one automatically.",
            )
        with right:
            total_epochs = st.number_input("Epochs", min_value=1, value=10, step=1)
            initial_lr = st.number_input("Initial LR", min_value=1e-8, value=2e-4, step=1e-5, format="%.8f")
            halve_epochs = st.text_input("Halve Epochs", "6,8")
        with st.expander("Advanced", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                batch_size = st.number_input("Batch Size", min_value=1, value=2, step=1)
                num_workers = st.number_input("Num Workers", min_value=0, value=DEFAULT_DAFT_NUM_WORKERS, step=1)
                global_seed = st.number_input("Global Seed", min_value=0, value=DEFAULT_DAFT_GLOBAL_SEED, step=1)
                adv_lambda = st.number_input("Adv Lambda", min_value=0.0, value=0.05, step=0.005, format="%.6f")
                align_lambda = st.number_input("Alignment Lambda", min_value=0.0, value=0.005, step=0.001, format="%.6f")
                save_optimizer_state = st.checkbox(
                    "Save Optimizer State For Exact Resume",
                    value=False,
                    help="Off by default to keep checkpoints small. Enable only when you need exact optimizer-state resume.",
                )
                pin_memory = st.checkbox(
                    "Pin Memory",
                    value=False,
                    help="Disabled by default for WSL/CUDA stability, especially when num_workers=0.",
                )
            with adv_right:
                lora_task_r = st.number_input("LoRA task r", min_value=1, value=16, step=1)
                lora_task_alpha = st.number_input("LoRA task alpha", min_value=1, value=32, step=1)
                lora_dropout = st.number_input("LoRA dropout", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
                deterministic_cuda = st.checkbox(
                    "Deterministic CUDA",
                    value=False,
                    help="Advanced reproducibility option. Disabled by default to avoid slower or unstable CUDA kernel paths on WSL.",
                )
        launch = st.form_submit_button("Launch DAFT", disabled=active_daft_running)

    selected_target = next((row for row in target_rows if row.get("dataset_id") == target_dataset_id), None)
    if selected_target and not evaluation_dataset_id:
        st.info("No Evaluation dataset selected. DAFT will use the last/latest checkpoint for registration; source-domain train metrics are not used for best-checkpoint selection.")

    if launch:
        running_job = _active_job_for_kind(registry_root, "daft-training")
        if running_job and not _job_is_live_blocking(running_job):
            running_job = None
        if running_job:
            st.session_state["active_daft_training_job_id"] = str(running_job.get("job_id") or "")
            st.warning("A DAFT job is already running. Switched back to the current job monitor.")
            st.rerun()
        if not (base_weight_id and source_dataset_id and target_dataset_id):
            st.error("Choose base weight, source dataset, and target dataset before launching DAFT.")
            return
        try:
            payload = build_daft_training_payload(
                registry_root=registry_root,
                model_id=model_id,
                base_weight_id=base_weight_id,
                source_dataset_id=source_dataset_id,
                target_dataset_id=target_dataset_id,
                evaluation_dataset_id=evaluation_dataset_id,
                alignment_mode=alignment_mode,
                total_epochs=int(total_epochs),
                initial_lr=float(initial_lr),
                halve_epochs=halve_epochs,
                device="auto",
                output_dir=output_dir,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                global_seed=int(global_seed),
                adv_lambda=float(adv_lambda),
                align_lambda=float(align_lambda),
                lora_task_r=int(lora_task_r),
                lora_task_alpha=int(lora_task_alpha),
                lora_dropout=float(lora_dropout),
                output_weight_id=output_weight_id,
                save_optimizer_state=bool(save_optimizer_state),
                pin_memory=bool(pin_memory),
                deterministic_cuda=bool(deterministic_cuda),
            )
            job_id = start_daft_training_background_job(registry_root=registry_root, payload=payload)
        except TRAINING_RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.session_state["active_daft_training_job_id"] = job_id
            st.success(f"DAFT training started: {job_id}")
            st.rerun()

    active_job = _recover_active_job_id(
        st,
        registry_root=registry_root,
        kind="daft-training",
        session_key="active_daft_training_job_id",
    )
    active_job_id = str(active_job) if active_job else None
    if active_job_id:
        _render_daft_training_job_monitor_auto(st, registry_root=registry_root, job_id=active_job_id)
    _render_clear_frontend_job_history_button(
        st,
        registry_root=registry_root,
        session_keys=("active_daft_training_job_id",),
        job_kinds=(("active_daft_training_job_id", "daft-training"),),
        label="Clear previous DAFT job display",
        button_key="clear_daft_frontend_job_history",
    )


def _render_daft_training_job_monitor(st: Any, *, registry_root: str, job_id: str) -> None:
    state = read_job_state(registry_root, job_id)
    if not state:
        return
    status = str(state.get("status") or "unknown")
    _render_daft_training_progress(st, state)
    stale_caption = _job_stale_caption(state)
    if stale_caption:
        st.caption(stale_caption)
    _render_job_health_notice(st, state)
    if status in {"running", "pending", "cancelling", "stalled"}:
        if st.button("Stop DAFT training", type="secondary", key=f"cancel_daft_{job_id}"):
            if request_job_cancel(registry_root, job_id):
                st.warning("Stop requested. The current training step will finish before the job stops.")
                st.rerun()
    if status == "completed":
        result = state.get("result")
        if isinstance(result, Mapping):
            st.success(f"DAFT weight registered: {result.get('registered_weight_id')}")
            st.dataframe(
                [
                    {"artifact": key, "path": result.get(key)}
                    for key in ("lora_checkpoint_path", "merged_checkpoint_path", "registered_weight_path", "metrics_path")
                    if result.get(key)
                ],
                use_container_width=True,
                hide_index=True,
            )
    elif status == "resource_blocked":
        _render_daft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status in {"stalled", "stalled_uninterruptible"}:
        _render_daft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "failed":
        st.error(f"DAFT training failed: {state.get('error') or state.get('message')}")
        _render_daft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "cancelled":
        st.warning("DAFT training was stopped. You can resume from the latest LoRA checkpoint if one was written.")
        _render_daft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "interrupted":
        st.warning("DAFT training was interrupted when the app stopped. You can resume from the latest LoRA checkpoint if one was written.")
        _render_daft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)


def _daft_checkpoint_paths_from_state(state: Mapping[str, Any]) -> tuple[str, str]:
    latest, best, _config_path = _training_resume_paths_from_state(
        state,
        default_output_dir="outputs/daft_training",
        latest_names=("lsdsam_daft_latest_lora.pth", "lsdformer_daft_latest_adapter.pth"),
        best_names=("lsdsam_daft_best_lora.pth", "lsdformer_daft_best_adapter.pth"),
    )
    return latest, best


def _daft_has_resume_checkpoint(state: Mapping[str, Any]) -> bool:
    latest, best = _daft_checkpoint_paths_from_state(state)
    return any(Path(path).exists() for path in (latest, best) if path)


def _render_daft_recovery_controls(st: Any, *, registry_root: str, job_id: str, state: Mapping[str, Any]) -> None:
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    latest_checkpoint, best_checkpoint = _daft_checkpoint_paths_from_state(state)
    resume_checkpoint = latest_checkpoint if latest_checkpoint and Path(latest_checkpoint).exists() else best_checkpoint
    _render_stale_job_cleanup_button(
        st,
        registry_root=registry_root,
        job_id=job_id,
        state=state,
        session_key="active_daft_training_job_id",
        label="Clear stale DAFT blocker",
    )
    if not resume_checkpoint or not Path(resume_checkpoint).exists():
        st.info("No DAFT LoRA checkpoint is available yet, so this job cannot be resumed.")
        return
    _latest, _best, config_path = _training_resume_paths_from_state(
        state,
        default_output_dir="outputs/daft_training",
        latest_names=("lsdsam_daft_latest_lora.pth", "lsdformer_daft_latest_adapter.pth"),
        best_names=("lsdsam_daft_best_lora.pth", "lsdformer_daft_best_adapter.pth"),
    )
    st.info(f"Resume checkpoint: {resume_checkpoint}")
    if config_path and Path(config_path).exists():
        st.caption(f"Training config snapshot: {config_path}")
    if st.button("Resume DAFT training", type="secondary", key=f"resume_daft_{job_id}"):
        resume_payload = dict(payload)
        resume_payload["resume"] = True
        resume_payload["resume_checkpoint_path"] = resume_checkpoint
        resume_payload["resume_run_id"] = _daft_resume_run_id()
        try:
            resumed_job_id = start_daft_training_background_job(registry_root=registry_root, payload=resume_payload)
        except TRAINING_RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.session_state["active_daft_training_job_id"] = resumed_job_id
            st.success(f"DAFT resume started: {resumed_job_id}")
            st.rerun()


def _render_daft_training_progress(st: Any, job: Mapping[str, Any]) -> None:
    percent = min(max(float(job.get("percent") or 0.0), 0.0), 1.0)
    title = _training_job_title(job, stage="DAFT", dataset_key="target_dataset_id", fallback=str(job.get("job_id") or "training"))
    status = str(job.get("status") or "unknown")
    phase = str(job.get("phase") or "queue")
    epoch = _int_value(job.get("current_epoch"))
    total_epochs = _int_value(job.get("total_epochs"))
    batch = _int_value(job.get("current_batch"))
    total_batches = _int_value(job.get("total_batches"))
    losses = job.get("losses") if isinstance(job.get("losses"), Mapping) else {}
    train_metrics = job.get("train") if isinstance(job.get("train"), Mapping) else {}
    evaluation = job.get("evaluation") if isinstance(job.get("evaluation"), Mapping) else {}
    evaluation_split = str(job.get("evaluation_split") or "").strip()
    eval_batch = _int_value(job.get("eval_current_batch"))
    eval_total_batches = _int_value(job.get("eval_total_batches"))
    device_label = str(job.get("device_name") or job.get("device") or "").strip()
    selection_metric = str(job.get("selection_metric") or "")
    best_score = job.get("best_score")
    best_epoch = _int_value(job.get("best_epoch"))
    latest_log = str(job.get("latest_log") or job.get("message") or "")
    if phase == "evaluation" and evaluation_split:
        train_position = (
            f"epoch {epoch}/{total_epochs or '?'} | eval {evaluation_split} {eval_batch}/{eval_total_batches or '?'}"
        )
    else:
        train_position = (
            f"epoch {epoch}/{total_epochs or '?'} | batch {batch}/{total_batches or '?'}"
            if epoch
            else phase
        )
    stats = [
        ("progress", f"{percent * 100:.2f}%"),
        ("status", status),
        ("position", train_position),
        ("device", device_label or "-"),
        ("pid", str(job.get("process_pid") or "-")),
        ("GPU now", _format_gpu_memory(job.get("gpu_memory_reserved_mb"))),
        ("GPU peak", _format_gpu_memory(job.get("gpu_memory_peak_reserved_mb"))),
        ("elapsed", _format_duration_seconds(_job_elapsed_seconds(job))),
        ("total loss", _format_float((losses or {}).get("total"), digits=4)),
        ("seg loss", _format_float((losses or {}).get("seg"), digits=4)),
        ("adv loss", _format_float((losses or {}).get("adv"), digits=4)),
        ("align loss", _format_float((losses or {}).get("align"), digits=6)),
        ("disc loss", _format_float((losses or {}).get("dis"), digits=4)),
        ("lr seg", _format_scientific(job.get("lr_seg"))),
        ("lr dis", _format_scientific(job.get("lr_dis"))),
        ("train mIoU", _format_float(train_metrics.get("mIoU"), digits=4) if train_metrics else "-"),
        ("eval mIoU", _format_float(evaluation.get("mIoU"), digits=4) if evaluation else "no eval"),
        ("best", _format_best_score(best_score, best_epoch, selection_metric)),
    ]
    html_stats = "\n".join(
        f'<div class="daft-train-stat"><label>{html.escape(label)}</label><strong>{html.escape(str(value))}</strong></div>'
        for label, value in stats
    )
    st.markdown(
        f"""
        <div class="daft-train-panel">
          <div class="daft-train-head">
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(phase)}</span>
          </div>
          <div class="taft-progress-track">
            <div class="taft-progress-fill" style="width: {_format_progress_percent(percent)};"></div>
          </div>
          <div class="daft-train-grid">
            {html_stats}
          </div>
          <div class="daft-train-log">{html.escape(latest_log or "Waiting for training output.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_daft_loss_charts(st, job)
    if status == "failed":
        st.error(f"{job.get('error_type', 'Error')}: {job.get('error', '')}")
        traceback_text = str(job.get("traceback") or "").strip()
        if traceback_text:
            with st.expander("Traceback", expanded=False):
                st.code(traceback_text, language="python")


def _render_daft_loss_charts(st: Any, job: Mapping[str, Any]) -> None:
    rows = _daft_loss_history_rows(job)
    if not rows or not hasattr(st, "columns"):
        return
    try:
        import altair as alt
        import pandas as pd
    except ModuleNotFoundError:
        if hasattr(st, "dataframe"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    st.markdown("##### Loss Trends")
    chart_specs = [
        ("total", "Total Loss"),
        ("seg", "Seg Loss"),
        ("adv", "Adv Loss"),
        ("align", "Align Loss"),
        ("dis", "Disc Loss"),
    ]
    observed_epochs = [int(row["epoch"]) for row in rows if _int_value(row.get("epoch")) > 0]
    min_epoch = min([1, *observed_epochs]) if observed_epochs else 1
    max_epoch = max(_int_value(job.get("total_epochs")), max(observed_epochs, default=1), 1)
    x_pad = 0.35 if max_epoch > min_epoch else 0.5
    x_domain = [min_epoch - x_pad, max_epoch + x_pad]
    if max_epoch - min_epoch <= 14:
        x_ticks = list(range(min_epoch, max_epoch + 1))
        x_axis = alt.Axis(values=x_ticks, format="d", labelFlush=False, tickMinStep=1)
    else:
        x_axis = alt.Axis(format="d", labelFlush=False, tickMinStep=1)
    cols = st.columns(len(chart_specs))
    for col, (key, label) in zip(cols, chart_specs):
        data_rows = []
        for row in rows:
            if key not in row:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if value != value or value in {float("inf"), float("-inf")}:
                continue
            data_rows.append({"epoch": int(row["epoch"]), "value": value})
        data = pd.DataFrame(data_rows)
        with col:
            st.caption(label)
            if data.empty:
                st.caption("No data yet")
            elif hasattr(st, "altair_chart"):
                values = [item["value"] for item in data_rows]
                y_min = min(values)
                y_max = max(values)
                y_pad = max(abs(y_max) * 0.08, 1e-6) if y_min == y_max else (y_max - y_min) * 0.12
                chart = (
                    alt.Chart(data)
                    .mark_line(point=True, clip=False)
                    .encode(
                        x=alt.X(
                            "epoch:Q",
                            title="epoch",
                            scale=alt.Scale(domain=x_domain, nice=False, zero=False),
                            axis=x_axis,
                        ),
                        y=alt.Y(
                            "value:Q",
                            title=key,
                            scale=alt.Scale(domain=[y_min - y_pad, y_max + y_pad], nice=False, zero=False),
                        ),
                        tooltip=[
                            alt.Tooltip("epoch:Q", title="epoch", format="d"),
                            alt.Tooltip("value:Q", title=label, format=".6f" if key == "align" else ".4f"),
                        ],
                    )
                    .properties(height=170)
                )
                st.altair_chart(chart, use_container_width=True)
            elif hasattr(st, "line_chart"):
                fallback = pd.DataFrame([{"epoch": item["epoch"], key: item["value"]} for item in data_rows])
                st.line_chart(fallback, x="epoch", y=key, height=170)


def _render_tsft_training_panel(st: Any, *, registry_root: str) -> None:
    dataset_rows = list_datasets(registry_root=registry_root)
    weights = list_weights(registry_root=registry_root)
    tsft_rows = _training_dataset_rows_for_class(dataset_rows, "TSFT", require_gt=True)
    evaluation_rows = [
        row
        for row in dataset_rows
        if row.get("dataset_class") == "Evaluation" and row.get("has_gt")
    ]
    weight_rows = [row for row in weights if row.get("exists", True)]
    st.markdown(
        """
        <div class="deployment-panel">
          <h3>LSDSAM TSFT</h3>
          <p>Task supervised fine-tuning uses registered TSFT samples with labels. Training device is selected automatically: GPU first, CPU fallback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Device Policy: {TSFT_TRAINING_DEVICE_POLICY}")
    if not tsft_rows:
        st.warning("Register a TSFT dataset with labels before launching TSFT.")
    if not weight_rows:
        st.warning("Register a base weight before launching TSFT.")
    active_tsft_job_id = _recover_active_job_id(
        st,
        registry_root=registry_root,
        kind="tsft-training",
        session_key="active_tsft_training_job_id",
    )
    active_tsft_state = read_job_state(registry_root, active_tsft_job_id) if active_tsft_job_id else {}
    active_tsft_running = bool(active_tsft_state and _job_is_live_blocking(active_tsft_state))
    if active_tsft_running:
        st.info("A TSFT training job is already running. The monitor below will keep tracking it even if you close and reopen the page.")

    selector_left, selector_middle, _selector_right = st.columns(3)
    with selector_left:
        model_id = st.selectbox("model", MODEL_CHOICES, index=0, key="tsft_model", disabled=active_tsft_running)
    model_weight_rows = _training_weight_rows_for_model(weight_rows, model_id)
    base_weight_options = _weight_id_options(model_weight_rows)
    base_weight_format = _weight_format_func(model_weight_rows)
    model_key = _training_model_widget_key(model_id)
    with selector_middle:
        if base_weight_options:
            base_weight_id = st.selectbox(
                "base weight",
                base_weight_options,
                key=f"tsft_base_weight_{model_key}",
                format_func=base_weight_format,
                disabled=active_tsft_running,
            )
        else:
            st.selectbox(
                "base weight",
                ["No registered weight"],
                key=f"tsft_base_weight_{model_key}_empty",
                disabled=True,
            )
            base_weight_id = ""
    if weight_rows and not base_weight_options:
        st.warning(f"No registered base weight is available for {model_id}. Check the Weight Repository model_id or register a matching weight.")

    with st.form("tsft_lsdsam_training"):
        left, middle, right = st.columns(3)
        with left:
            dataset_id = st.selectbox(
                "TSFT dataset",
                [row["dataset_id"] for row in tsft_rows],
                disabled=not tsft_rows,
                key="tsft_dataset",
                format_func=_dataset_id_option_label,
            ) if tsft_rows else ""
            evaluation_dataset_id = st.selectbox(
                "Evaluation Dataset",
                ["None", *[row["dataset_id"] for row in evaluation_rows]],
                key="tsft_evaluation_dataset",
            )
            evaluation_dataset_id = "" if evaluation_dataset_id == "None" else evaluation_dataset_id
            output_dir = st.text_input("Output Dir", "outputs/tsft_training", key="tsft_output_dir")
            output_weight_id = st.text_input(
                "TSFT Weight ID",
                "",
                help="Optional. Leave blank to let LSDetector generate one automatically.",
                key="tsft_output_weight_id",
            )
        with right:
            max_epoch = st.number_input("Max Epoch", min_value=1, value=10, step=1, key="tsft_max_epoch")
            max_lr = st.number_input("Max LR", min_value=1e-8, value=2e-4, step=1e-5, format="%.8f", key="tsft_max_lr")
            min_lr = st.number_input("Min LR", min_value=0.0, value=1e-6, step=1e-6, format="%.8f", key="tsft_min_lr")
            max_decay_epoch = st.number_input("Max Decay Epoch", min_value=1, value=10, step=1, key="tsft_max_decay_epoch")
        with st.expander("Advanced", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                batch_size = st.number_input("Batch Size", min_value=1, value=2, step=1, key="tsft_batch_size")
                num_workers = st.number_input("Num Workers", min_value=0, value=DEFAULT_DAFT_NUM_WORKERS, step=1, key="tsft_num_workers")
                save_optimizer_state = st.checkbox(
                    "Save Optimizer State For Exact Resume",
                    value=False,
                    help="Off by default to keep checkpoints small. Enable only when you need exact optimizer-state resume.",
                    key="tsft_save_optimizer_state",
                )
            with adv_right:
                lora_task_r = st.number_input("LoRA task r", min_value=1, value=16, step=1, key="tsft_lora_r")
                lora_task_alpha = st.number_input("LoRA task alpha", min_value=1, value=32, step=1, key="tsft_lora_alpha")
                lora_dropout = st.number_input("LoRA dropout", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="tsft_lora_dropout")
                resume = st.checkbox("Resume From Checkpoint", value=False, key="tsft_resume")
                resume_checkpoint_path = st.text_input(
                    "Resume Checkpoint Path",
                    "",
                    help="Optional. Leave blank to use the latest adapter checkpoint under Output Dir/checkpoints.",
                    key="tsft_resume_checkpoint_path",
                )
        launch = st.form_submit_button("Launch TSFT", disabled=active_tsft_running)

    if launch:
        running_job = _active_job_for_kind(registry_root, "tsft-training")
        if running_job and not _job_is_live_blocking(running_job):
            running_job = None
        if running_job:
            st.session_state["active_tsft_training_job_id"] = str(running_job.get("job_id") or "")
            st.warning("A TSFT job is already running. Switched back to the current job monitor.")
            st.rerun()
        if not (base_weight_id and dataset_id):
            st.error("Choose base weight and TSFT dataset before launching TSFT.")
            return
        try:
            payload = build_tsft_training_payload(
                registry_root=registry_root,
                model_id=model_id,
                base_weight_id=base_weight_id,
                dataset_id=dataset_id,
                evaluation_dataset_id=evaluation_dataset_id,
                max_epoch=int(max_epoch),
                max_lr=float(max_lr),
                min_lr=float(min_lr),
                max_decay_epoch=int(max_decay_epoch),
                device="auto",
                output_dir=output_dir,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                lora_task_r=int(lora_task_r),
                lora_task_alpha=int(lora_task_alpha),
                lora_dropout=float(lora_dropout),
                output_weight_id=output_weight_id,
                resume=bool(resume),
                resume_checkpoint_path=resume_checkpoint_path,
                save_optimizer_state=bool(save_optimizer_state),
            )
            job_id = start_tsft_training_background_job(registry_root=registry_root, payload=payload)
        except TRAINING_RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.session_state["active_tsft_training_job_id"] = job_id
            st.success(f"TSFT training started: {job_id}")
            st.rerun()

    active_job = _recover_active_job_id(
        st,
        registry_root=registry_root,
        kind="tsft-training",
        session_key="active_tsft_training_job_id",
    )
    active_job_id = str(active_job) if active_job else None
    if active_job_id:
        _render_tsft_training_job_monitor_auto(st, registry_root=registry_root, job_id=active_job_id)
    _render_clear_frontend_job_history_button(
        st,
        registry_root=registry_root,
        session_keys=("active_tsft_training_job_id",),
        job_kinds=(("active_tsft_training_job_id", "tsft-training"),),
        label="Clear previous TSFT job display",
        button_key="clear_tsft_frontend_job_history",
    )


def _render_tsft_training_job_monitor(st: Any, *, registry_root: str, job_id: str) -> None:
    state = read_job_state(registry_root, job_id)
    if not state:
        return
    status = str(state.get("status") or "unknown")
    _render_tsft_training_progress(st, state)
    stale_caption = _job_stale_caption(state)
    if stale_caption:
        st.caption(stale_caption)
    _render_job_health_notice(st, state)
    if status in {"running", "pending", "cancelling", "stalled"}:
        if st.button("Stop TSFT training", type="secondary", key=f"cancel_tsft_{job_id}"):
            if request_job_cancel(registry_root, job_id):
                st.warning("Stop requested. The current training step will finish before the job stops.")
                st.rerun()
    if status == "completed":
        result = state.get("result")
        if isinstance(result, Mapping):
            st.success(f"TSFT weight registered: {result.get('registered_weight_id')}")
            st.dataframe(
                [
                    {"artifact": key, "path": result.get(key)}
                    for key in ("lora_checkpoint_path", "merged_checkpoint_path", "registered_weight_path", "metrics_path")
                    if result.get(key)
                ],
                use_container_width=True,
                hide_index=True,
            )
        if _tsft_has_resume_checkpoint(state):
            st.info("A temporary LoRA checkpoint is still available, so you can resume training from this run.")
            _render_tsft_recovery_controls(
                st,
                registry_root=registry_root,
                job_id=job_id,
                state=state,
                allow_register=False,
            )
    elif status == "resource_blocked":
        _render_tsft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status in {"stalled", "stalled_uninterruptible"}:
        _render_tsft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "failed":
        st.error(f"TSFT training failed: {state.get('error') or state.get('message')}")
        _render_tsft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "cancelled":
        st.warning("TSFT training was stopped. You can register the best checkpoint or resume from latest.")
        _render_tsft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)
    elif status == "interrupted":
        st.warning("TSFT training was interrupted when the app stopped. You can register the best checkpoint or resume from latest.")
        _render_tsft_recovery_controls(st, registry_root=registry_root, job_id=job_id, state=state)


def _tsft_has_resume_checkpoint(state: Mapping[str, Any]) -> bool:
    latest, best, _config_path = _training_resume_paths_from_state(
        state,
        default_output_dir="outputs/tsft_training",
        latest_names=("lsdsam_tsft_latest_lora.pth", "lsdformer_tsft_latest_adapter.pth"),
        best_names=("lsdsam_tsft_best_lora.pth", "lsdformer_tsft_best_adapter.pth"),
    )
    return any(Path(path).exists() for path in (latest, best) if path)


def _render_tsft_recovery_controls(
    st: Any,
    *,
    registry_root: str,
    job_id: str,
    state: Mapping[str, Any],
    allow_register: bool = True,
) -> None:
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    _render_stale_job_cleanup_button(
        st,
        registry_root=registry_root,
        job_id=job_id,
        state=state,
        session_key="active_tsft_training_job_id",
        label="Clear stale TSFT blocker",
    )
    latest_checkpoint, best_checkpoint, config_path = _training_resume_paths_from_state(
        state,
        default_output_dir="outputs/tsft_training",
        latest_names=("lsdsam_tsft_latest_lora.pth", "lsdformer_tsft_latest_adapter.pth"),
        best_names=("lsdsam_tsft_best_lora.pth", "lsdformer_tsft_best_adapter.pth"),
    )
    checkpoint_for_register = best_checkpoint or latest_checkpoint
    if not checkpoint_for_register or not any(Path(path).exists() for path in (best_checkpoint, latest_checkpoint) if path):
        st.info("No TSFT checkpoint path is available yet, so this job cannot be finalized or resumed.")
        return
    if config_path and Path(config_path).exists():
        st.caption(f"Training config snapshot: {config_path}")
    if allow_register:
        st.info("A TSFT checkpoint is available. Register best to save a TSFT weight now, or resume from latest checkpoint.")
        register_col, resume_col = st.columns(2)
    else:
        st.info("A TSFT latest checkpoint is available for resume. The registered weight remains unchanged until the resumed run finishes.")
        resume_col = st.container()
    if allow_register and register_col.button("Register best TSFT weight", type="primary", key=f"finalize_tsft_{job_id}"):
        try:
            config = _tsft_config_from_payload(registry_root=registry_root, payload=payload)
            result = finalize_tsft_checkpoint(config, checkpoint_path=checkpoint_for_register)
            update_job_state(
                registry_root,
                job_id,
                {
                    "status": "completed",
                    "percent": 1.0,
                    "phase": "register",
                    "result": result.to_dict(),
                    "merged_checkpoint_path": str(result.merged_checkpoint_path),
                    "registered_weight_id": result.registered_weight_id,
                    "registered_weight_path": str(result.registered_weight_path),
                    "message": f"Registered TSFT weight {result.registered_weight_id} from checkpoint.",
                },
            )
        except TRAINING_RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.success("Registered best TSFT checkpoint as a TSFT weight.")
            st.rerun()
    if resume_col.button("Resume TSFT training", type="secondary", key=f"resume_tsft_{job_id}"):
        resume_payload = dict(payload)
        resume_payload["resume"] = True
        resume_payload["resume_checkpoint_path"] = latest_checkpoint or best_checkpoint
        resume_payload["resume_run_id"] = _tsft_resume_run_id()
        resume_payload.setdefault("cleanup_lora_checkpoints_after_register", True)
        try:
            resumed_job_id = start_tsft_training_background_job(registry_root=registry_root, payload=resume_payload)
        except TRAINING_RUNNABLE_ERRORS as exc:
            st.error(str(exc))
        else:
            st.session_state["active_tsft_training_job_id"] = resumed_job_id
            st.success(f"TSFT resume started: {resumed_job_id}")
            st.rerun()


def _render_tsft_training_progress(st: Any, job: Mapping[str, Any]) -> None:
    percent = min(max(float(job.get("percent") or 0.0), 0.0), 1.0)
    title = _training_job_title(job, stage="TSFT", dataset_key="dataset_id", fallback=str(job.get("job_id") or "training"))
    status = str(job.get("status") or "unknown")
    phase = str(job.get("phase") or "queue")
    epoch = _int_value(job.get("current_epoch"))
    total_epochs = _int_value(job.get("total_epochs"))
    batch = _int_value(job.get("current_batch"))
    total_batches = _int_value(job.get("total_batches"))
    train = job.get("train") if isinstance(job.get("train"), Mapping) else {}
    test = job.get("eval") if isinstance(job.get("eval"), Mapping) else job.get("test") if isinstance(job.get("test"), Mapping) else {}
    evaluation_split = str(job.get("evaluation_split") or "").strip()
    eval_batch = _int_value(job.get("eval_current_batch"))
    eval_total_batches = _int_value(job.get("eval_total_batches"))
    device_label = str(job.get("device_name") or job.get("device") or "").strip()
    selection_metric = str(job.get("selection_metric") or "")
    best_score = job.get("best_score")
    best_epoch = _int_value(job.get("best_epoch"))
    latest_log = str(job.get("latest_log") or job.get("message") or "")
    if phase == "evaluation" and evaluation_split:
        train_position = (
            f"epoch {epoch}/{total_epochs or '?'} | eval {evaluation_split} {eval_batch}/{eval_total_batches or '?'}"
        )
    else:
        train_position = (
            f"epoch {epoch}/{total_epochs or '?'} | batch {batch}/{total_batches or '?'}"
            if epoch
            else phase
        )
    stats = [
        ("progress", f"{percent * 100:.2f}%"),
        ("status", status),
        ("position", train_position),
        ("device", device_label or "-"),
        ("pid", str(job.get("process_pid") or "-")),
        ("GPU now", _format_gpu_memory(job.get("gpu_memory_reserved_mb"))),
        ("GPU peak", _format_gpu_memory(job.get("gpu_memory_peak_reserved_mb"))),
        ("elapsed", _format_duration_seconds(_job_elapsed_seconds(job))),
        ("lr", _format_scientific(job.get("lr"))),
        ("train loss", _format_float((train or {}).get("loss"), digits=4)),
        ("train mIoU", _format_float((train or {}).get("mIoU"), digits=4)),
        ("train IoU", _format_float((train or {}).get("IoU_foreground", (train or {}).get("IoU_landslide")), digits=4)),
        ("eval loss", _format_float((test or {}).get("loss"), digits=4) if test else "no eval"),
        ("eval mIoU", _format_float((test or {}).get("mIoU"), digits=4) if test else "no eval"),
        ("eval IoU", _format_float((test or {}).get("IoU_foreground", (test or {}).get("IoU_landslide")), digits=4) if test else "no eval"),
        ("best", _format_best_score(best_score, best_epoch, selection_metric)),
    ]
    html_stats = "\n".join(
        f'<div class="daft-train-stat"><label>{html.escape(label)}</label><strong>{html.escape(str(value))}</strong></div>'
        for label, value in stats
    )
    st.markdown(
        f"""
        <div class="daft-train-panel">
          <div class="daft-train-head">
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(phase)}</span>
          </div>
          <div class="taft-progress-track">
            <div class="taft-progress-fill" style="width: {_format_progress_percent(percent)};"></div>
          </div>
          <div class="daft-train-grid">
            {html_stats}
          </div>
          <div class="daft-train-log">{html.escape(latest_log or "Waiting for training output.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_tsft_metric_charts(st, job)
    if status == "failed":
        st.error(f"{job.get('error_type', 'Error')}: {job.get('error', '')}")


def _render_tsft_metric_charts(st: Any, job: Mapping[str, Any]) -> None:
    rows = _tsft_metric_history_rows(job)
    if not rows or not hasattr(st, "columns"):
        return
    try:
        import altair as alt
        import pandas as pd
    except ModuleNotFoundError:
        if hasattr(st, "dataframe"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    st.markdown("##### TSFT Metrics")
    chart_specs = [
        ("loss", "Loss"),
        ("mIoU", "mIoU"),
        ("IoU", "IoU"),
    ]
    cols = st.columns(len(chart_specs))
    for col, (metric, label) in zip(cols, chart_specs):
        data_rows = []
        for row in rows:
            if f"train_{metric}" in row:
                data_rows.append({"epoch": int(row["epoch"]), "split": "train", label: row[f"train_{metric}"]})
            if f"eval_{metric}" in row:
                data_rows.append({"epoch": int(row["epoch"]), "split": "eval", label: row[f"eval_{metric}"]})
        data = pd.DataFrame(data_rows)
        with col:
            st.caption(label)
            if data.empty:
                st.caption("No data yet")
            elif hasattr(st, "altair_chart"):
                chart = (
                    alt.Chart(data)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("epoch:O", title="epoch"),
                        y=alt.Y(f"{label}:Q", title=label),
                        color=alt.Color("split:N", title="split"),
                        tooltip=["epoch:O", "split:N", alt.Tooltip(f"{label}:Q", format=".4f")],
                    )
                    .properties(height=170)
                )
                st.altair_chart(chart, use_container_width=True)
            elif hasattr(st, "line_chart"):
                st.line_chart(data, x="epoch", y=label, color="split", height=170)


def _render_dataset_upload_console(st: Any, *, registry_root: str) -> str:
    with st.container(border=True):
        st.markdown('<div class="module-title">Dataset Upload</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="module-note">Choose one dataset class first. The form below only exposes the fields needed by that class.</div>',
            unsafe_allow_html=True,
        )
        upload_class = st.radio(
            "dataset upload class",
            DATASET_CLASSES,
            horizontal=True,
            key="dataset_upload_class",
        )
        if upload_class == "TAFT":
            _render_taft_upload_panel(st, registry_root=registry_root)
        elif upload_class == "DAFT":
            _render_daft_upload_panel(st, registry_root=registry_root)
        elif upload_class == "TSFT":
            _render_tsft_upload_panel(st, registry_root=registry_root)
        else:
            _render_evaluation_upload_panel(st, registry_root=registry_root)
    return str(upload_class)


def _dataset_upload_session_key(dataset_class: str) -> str:
    safe_class = str(dataset_class or "dataset").strip().lower().replace("-", "_") or "dataset"
    return f"active_dataset_upload_{safe_class}_job_id"


def _dataset_upload_job_filter(dataset_class: str) -> Callable[[Mapping[str, Any]], bool]:
    expected = str(dataset_class or "").strip()

    def _matches(job: Mapping[str, Any]) -> bool:
        payload = job.get("payload") if isinstance(job.get("payload"), Mapping) else {}
        return str(payload.get("dataset_class") or job.get("dataset_class") or "").strip() == expected

    return _matches


def _latest_module_job(registry_root: str, kind: str) -> Mapping[str, Any] | None:
    active = _active_job_for_kind(registry_root, kind)
    if active:
        return active
    latest = _latest_job_for_kind(registry_root, kind)
    if latest and str(latest.get("status") or "") == "completed":
        return None
    return latest


def _dataset_version_exists(registry_root: str, dataset_class: str, version: str) -> bool:
    dataset_class = str(dataset_class or "").strip()
    version = str(version or "").strip()
    if not dataset_class or not version:
        return False
    for row in list_datasets(registry_root=registry_root):
        if row.get("dataset_class") == dataset_class and str(row.get("version") or "") == version:
            return True
    return False


def _dataset_upload_payload_version_exists(registry_root: str, payload: Mapping[str, Any]) -> bool:
    return _dataset_version_exists(
        registry_root,
        str(payload.get("dataset_class") or ""),
        str(payload.get("version") or ""),
    )


def _job_state_for_id(registry_root: str, job_id: str) -> Mapping[str, Any]:
    return read_job_state(registry_root, job_id) if job_id else {}


def _job_is_blocking_resubmission(
    job: Mapping[str, Any],
    *,
    registry_root: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    status = str(job.get("status") or "")
    if status in ACTIVE_STATUSES:
        return True
    if status != "completed":
        return False
    if registry_root is None or payload is None:
        return True
    return _dataset_upload_payload_version_exists(registry_root, payload)


def _render_module_latest_job_auto(st: Any, *, registry_root: str, kind: str, title: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _module_latest_job_fragment() -> None:
            job = _latest_module_job(registry_root, kind)
            if job:
                _render_module_job_progress(st, registry_root=registry_root, job=job, title=title)

        _module_latest_job_fragment()
        return
    job = _latest_module_job(registry_root, kind)
    if job:
        _render_module_job_progress(st, registry_root=registry_root, job=job, title=title)
        _schedule_active_job_refresh(st, job)


def _render_module_job_progress_auto(st: Any, *, registry_root: str, job_id: str, title: str) -> None:
    fragment = _streamlit_fragment(st)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _job_fragment() -> None:
            state = read_job_state(registry_root, job_id)
            if state:
                _render_module_job_progress(st, registry_root=registry_root, job=state, title=title)

        _job_fragment()
        return
    state = read_job_state(registry_root, job_id)
    if state:
        _render_module_job_progress(st, registry_root=registry_root, job=state, title=title)
        _schedule_active_job_refresh(st, state)


def _render_module_job_progress(st: Any, *, registry_root: str, job: Mapping[str, Any], title: str) -> None:
    percent = min(max(float(job.get("percent") or job.get("overall_percent") or 0.0), 0.0), 1.0)
    status = str(job.get("status") or "unknown")
    description = str(job.get("description") or job.get("job_id") or title)
    message = str(job.get("message") or "")
    processed = _int_value(job.get("processed"))
    total = _int_value(job.get("total"))
    written = _int_value(job.get("written"))
    skipped = _int_value(job.get("skipped"))
    rewritten = _int_value(job.get("rewritten"))
    pid = str(job.get("process_pid") or "-")
    stale_caption = _job_stale_caption(job)
    result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
    meta = [
        ("status", status),
        ("progress", _format_progress_percent(percent)),
        ("processed", f"{processed}/{total or 'discovering'}"),
        ("pid", pid),
    ]
    if written or skipped or rewritten:
        meta.extend([("written", written), ("skipped", skipped), ("rewritten", rewritten)])
    if result:
        sample_count = result.get("sample_count")
        repository_path = result.get("repository_path")
        manifest_path = result.get("manifest_path")
        if sample_count not in {None, ""}:
            meta.append(("samples", sample_count))
        if repository_path:
            meta.append(("repository", repository_path))
        if manifest_path:
            meta.append(("manifest", manifest_path))
    meta_html = "\n".join(
        f"<span><label>{html.escape(str(label))}</label>{html.escape(str(value))}</span>"
        for label, value in meta
    )
    st.markdown(
        f"""
        <div class="module-job-card">
          <div class="module-job-head">
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(description)}</span>
          </div>
          <div class="taft-progress-track" aria-label="job progress">
            <div class="taft-progress-fill" style="width: {_format_progress_percent(percent)};"></div>
          </div>
          <div class="module-job-meta">{meta_html}</div>
          <div class="module-job-message">{html.escape(message or stale_caption or 'Waiting for job output.')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if stale_caption:
        st.caption(stale_caption)
    job_id = str(job.get("job_id") or "")
    if status in {"running", "pending", "cancelling"} and job_id:
        if st.button("Stop job", key=f"cancel_module_{job_id}", type="secondary"):
            if request_job_cancel(registry_root, job_id):
                st.rerun()
    if status == "failed":
        st.error(f"{job.get('error_type', 'Error')}: {job.get('error', '')}")
    elif status == "interrupted":
        st.warning("The app stopped before this job finished. Start the same action again with resume enabled to continue from existing outputs.")
    elif status == "completed":
        st.success(message or "Job completed.")


def _render_taft_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### TAFT Pretraining Tree")
    existing_versions = _registered_taft_versions(registry_root)
    with st.form("taft_tree_discover"):
        taft_tree_mode = st.radio(
            "TAFT import mode",
            ("Register new TAFT version", "Add subsets to existing version"),
            horizontal=True,
            key="taft_tree_mode",
        )
        append_to_existing = taft_tree_mode == "Add subsets to existing version"
        left, right = st.columns(2)
        with left:
            if append_to_existing and existing_versions:
                taft_tree_version = st.selectbox("TAFT version", existing_versions, key="taft_tree_existing_version")
            elif append_to_existing:
                taft_tree_version = st.text_input("TAFT version", "", key="taft_tree_existing_version_manual")
            else:
                taft_tree_version = st.text_input("TAFT version", "Landslide77K", key="taft_tree_new_version")
            taft_tree_root = st.text_input(
                "TAFT tree root",
                "/path/to/Landslide77K",
                key="taft_tree_root",
            )
        with right:
            taft_tree_standardize = True
            taft_tree_resume = st.checkbox(
                "resume matching partial import",
                value=True,
                key="taft_tree_resume",
                help=(
                    "All TAFT imports are standardized into the internal float32 dataset store. "
                    "When enabled, matching partial imports continue from valid existing files. "
                    "When disabled for a new-version import, an existing same-name version is rebuilt from scratch. "
                    "Append mode keeps existing subsets and adds only new leaf subsets."
                ),
            )
            taft_tree_notes = st.text_area("TAFT notes", "", key="taft_tree_notes")
        discover_tree = st.form_submit_button("Discover TAFT Tree")

    if discover_tree:
        try:
            discovery = discover_taft_tree(root=taft_tree_root)
            st.session_state["taft_tree_discovery"] = discovery
            st.session_state["taft_tree_annotation_rows"] = _taft_tree_annotation_rows(discovery)
            st.session_state["taft_tree_annotation_editor_version"] = (
                int(st.session_state.get("taft_tree_annotation_editor_version", 0)) + 1
            )
            st.session_state["taft_tree_import_config"] = {
                "mode": taft_tree_mode,
                "append_to_existing": append_to_existing,
                "version": taft_tree_version,
                "root": taft_tree_root,
                "standardize": taft_tree_standardize,
                "resume": taft_tree_resume,
                "force_rebuild": False if append_to_existing else not taft_tree_resume,
                "notes": taft_tree_notes,
            }
        except RegistryError as exc:
            st.error(str(exc))

    taft_tree_discovery = st.session_state.get("taft_tree_discovery")
    if not isinstance(taft_tree_discovery, Mapping):
        return

    _render_taft_tree_discovery_metrics(st, taft_tree_discovery)
    if taft_tree_discovery.get("has_errors"):
        st.error("Discovery found pairing or structure errors. Import is blocked until these are resolved.")

    annotation_rows = _coerce_record_rows(st.session_state.get("taft_tree_annotation_rows"))
    if not annotation_rows:
        annotation_rows = _taft_tree_annotation_rows(taft_tree_discovery)
        st.session_state["taft_tree_annotation_rows"] = annotation_rows

    form_version = int(st.session_state.get("taft_tree_annotation_editor_version", 0))
    control_cols = st.columns(3)
    with control_cols[0]:
        confirm_all_annotations = st.button("Confirm all", key="taft_tree_confirm_all")
    with control_cols[1]:
        clear_confirmations = st.button("Clear confirmations", key="taft_tree_clear_confirmations")
    with control_cols[2]:
        st.caption("Edit rows below, then apply.")
    if confirm_all_annotations or clear_confirmations:
        annotation_rows = _taft_set_rows_confirmed(annotation_rows, confirm_all_annotations)
        st.session_state["taft_tree_annotation_rows"] = annotation_rows
        for idx, row in enumerate(annotation_rows):
            row_key = _taft_annotation_row_key(row, idx, form_version)
            st.session_state[f"taft_confirmed_{row_key}"] = bool(confirm_all_annotations)

    with st.form("taft_tree_annotation_form"):
        header_cols = st.columns([1.3, 1.8, 0.8, 0.8, 1.3, 1.2, 0.8])
        header_cols[0].markdown("**parent_group**")
        header_cols[1].markdown("**subset**")
        header_cols[2].markdown("**train**")
        header_cols[3].markdown("**test**")
        header_cols[4].markdown("**RGB_bands**")
        header_cols[5].markdown("**label_polarity**")
        header_cols[6].markdown("**confirmed**")
        form_rows: list[dict[str, Any]] = []
        polarity_options = list(LABEL_POLARITIES)
        for idx, row in enumerate(annotation_rows):
            row_key = _taft_annotation_row_key(row, idx, form_version)
            row_cols = st.columns([1.3, 1.8, 0.8, 0.8, 1.3, 1.2, 0.8])
            parent_group = str(row.get("parent_group", ""))
            subset = str(row.get("subset", ""))
            row_cols[0].text(parent_group)
            row_cols[1].text(subset)
            row_cols[2].text(str(row.get("train_count", 0)))
            row_cols[3].text(str(row.get("test_count", 0)))
            RGB_bands = row_cols[4].text_input(
                "RGB_bands",
                value=str(row.get("RGB_bands", "")),
                key=f"taft_rgb_bands_{row_key}",
                label_visibility="collapsed",
            )
            current_polarity = str(row.get("label_polarity", ""))
            polarity_index = polarity_options.index(current_polarity) if current_polarity in polarity_options else 0
            label_polarity = row_cols[5].selectbox(
                "label_polarity",
                polarity_options,
                index=polarity_index,
                key=f"taft_label_polarity_{row_key}",
                label_visibility="collapsed",
            )
            confirmed_key = f"taft_confirmed_{row_key}"
            if confirmed_key not in st.session_state:
                st.session_state[confirmed_key] = bool(row.get("confirmed", False))
            confirmed = row_cols[6].checkbox(
                "confirmed",
                key=confirmed_key,
                label_visibility="collapsed",
            )
            form_rows.append(
                {
                    "parent_group": parent_group,
                    "subset": subset,
                    "train_count": row.get("train_count", 0),
                    "test_count": row.get("test_count", 0),
                    "RGB_bands": RGB_bands,
                    "label_polarity": label_polarity,
                    "confirmed": confirmed,
                }
            )
        apply_annotations = st.form_submit_button("Apply annotation edits")
    if apply_annotations:
        edited_rows = form_rows
        st.session_state["taft_tree_annotation_rows"] = edited_rows
    else:
        edited_rows = annotation_rows

    if _taft_annotations_complete(edited_rows):
        st.success("All TAFT leaf subsets are confirmed and ready to import.")
    else:
        st.warning("Confirm every subset before importing. Imports use the edited RGB_bands and label_polarity values.")

    import_config = st.session_state.get("taft_tree_import_config")
    if not isinstance(import_config, Mapping):
        import_config = {
            "mode": taft_tree_mode,
            "append_to_existing": append_to_existing,
            "version": taft_tree_version,
            "root": taft_tree_root,
            "standardize": True,
            "resume": taft_tree_resume,
            "force_rebuild": False if append_to_existing else not taft_tree_resume,
            "notes": taft_tree_notes,
        }
    append_import = bool(import_config.get("append_to_existing", False))
    import_disabled = bool(taft_tree_discovery.get("has_errors")) or not _taft_annotations_complete(edited_rows)
    import_label = "Add TAFT Subsets" if append_import else "Import TAFT Tree"
    if st.button(import_label, disabled=import_disabled, type="primary"):
        try:
            _start_taft_upload_job(
                st,
                registry_root=registry_root,
                version=str(import_config.get("version", taft_tree_version)),
                root=str(import_config.get("root", taft_tree_root)),
                standardize=True,
                resume=bool(import_config.get("resume", taft_tree_resume)),
                force_rebuild=bool(
                    import_config.get("force_rebuild", False if append_import else not bool(import_config.get("resume", taft_tree_resume)))
                ),
                append_to_existing=append_import,
                notes=str(import_config.get("notes", taft_tree_notes)),
                subset_overrides=_taft_subset_overrides_from_rows(edited_rows),
            )
        except RegistryError as exc:
            st.error(str(exc))

def _start_taft_upload_job(
    st: Any,
    *,
    registry_root: str,
    version: str,
    root: str,
    standardize: bool,
    resume: bool,
    force_rebuild: bool,
    append_to_existing: bool,
    notes: str,
    subset_overrides: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = {
        "dataset_class": "TAFT",
        "version": version,
        "root": root,
        "preset": "landslide77k",
        "standardize": standardize,
        "resume": resume,
        "force_rebuild": force_rebuild,
        "append_to_existing": append_to_existing,
        "split_policy": "preserve",
        "notes": notes,
        "subset_overrides": _json_ready(dict(subset_overrides)),
    }
    job_kind = "dataset-upload-taft-append" if append_to_existing else "dataset-upload-taft"
    job_id = stable_job_id(kind=job_kind, version=version, payload=payload)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
        return run_dataset_upload_background_task(
            registry_root=registry_root,
            payload=payload,
            progress_callback=progress_callback,
        )

    job = start_process_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="dataset-upload",
        description=f"TAFT upload {version}",
        payload=payload,
        target=target,
    )
    st.session_state[_dataset_upload_session_key("TAFT")] = job.job_id
    st.success(f"TAFT upload started: {job.job_id}")
    st.rerun()


def run_dataset_upload_background_task(
    *,
    registry_root: str,
    payload: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None],
) -> Any:
    dataset_class = str(payload.get("dataset_class") or "").strip()
    version = str(payload.get("version") or "").strip()
    progress_callback(
        {
            "status": "running",
            "message": f"Standardizing {dataset_class or 'dataset'} {version}.",
            "percent": 0.02,
        }
    )
    if dataset_class == "TAFT":
        entry = register_taft_tree(
            registry_root=registry_root,
            version=version,
            root=str(payload.get("root") or ""),
            preset=str(payload.get("preset") or "landslide77k"),
            standardize=payload.get("standardize", True) is not False,
            resume=bool(payload.get("resume")),
            force_rebuild=bool(payload.get("force_rebuild")),
            append_to_existing=bool(payload.get("append_to_existing")),
            split_policy=str(payload.get("split_policy") or "preserve"),
            notes=str(payload.get("notes") or ""),
            subset_overrides=_coerce_subset_overrides(payload.get("subset_overrides")),
            progress_callback=progress_callback,
        )
    elif dataset_class == "DAFT":
        upload_method = str(payload.get("upload_method") or DAFT_UPLOAD_METHOD_PACKAGED)
        tile_size = _optional_int(payload.get("tile_size"))
        stride = _optional_int(payload.get("stride"))
        max_nodata_ratio = _optional_float(payload.get("max_nodata_ratio"))
        slicing_aoi_path = str(payload.get("slicing_aoi_path") or "")
        entry = register_dataset(
            registry_root=registry_root,
            name=version,
            dataset_class="DAFT",
            imagery_path=str(payload.get("imagery_path") or ""),
            label_path=None,
            has_gt=False,
            RGB_bands=str(payload.get("RGB_bands") or DEFAULT_RGB_BANDS),
            tile_schema=str(
                payload.get("tile_schema")
                or _daft_tile_schema(
                    upload_method,
                    tile_size or 0,
                    stride or 0,
                    max_nodata_ratio or 0.20,
                    slicing_aoi_path=slicing_aoi_path,
                )
            ),
            notes=str(payload.get("notes") or ""),
            version=version,
            standardize=True,
            overwrite=bool(payload.get("overwrite")),
            resume=bool(payload.get("resume")),
            upload_method=upload_method,
            tile_size=tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            stride=stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            daft_max_nodata_ratio=max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            slicing_aoi_path=slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            progress_callback=progress_callback,
        )
    elif dataset_class == "Evaluation":
        upload_method = str(payload.get("upload_method") or DAFT_UPLOAD_METHOD_PACKAGED)
        tile_size = _optional_int(payload.get("tile_size")) or 0
        stride = _optional_int(payload.get("stride")) or 0
        max_nodata_ratio = _optional_float(payload.get("max_nodata_ratio"))
        min_polygon_overlap_ratio = _optional_float(payload.get("min_polygon_overlap_ratio"))
        min_positive_pixel_ratio = _optional_float(payload.get("min_positive_pixel_ratio"))
        negative_ratio = _optional_float(payload.get("negative_ratio"))
        no_feature_policy = str(payload.get("no_feature_tile_policy") or TSFT_NO_FEATURE_POLICY_ALL)
        slicing_aoi_path = str(payload.get("slicing_aoi_path") or "")
        entry = register_dataset(
            registry_root=registry_root,
            name=version,
            dataset_class="Evaluation",
            imagery_path=str(payload.get("imagery_path") or ""),
            label_path=str(payload.get("label_path") or ""),
            has_gt=True,
            RGB_bands=str(payload.get("RGB_bands") or DEFAULT_RGB_BANDS),
            label_polarity=str(payload.get("label_polarity") or LABEL_POLARITIES[0]),
            tile_schema=str(
                payload.get("tile_schema")
                or _tsft_tile_schema(
                    upload_method=upload_method,
                    tile_size=tile_size,
                    stride=stride,
                    max_nodata_ratio=max_nodata_ratio or 0.20,
                    min_polygon_overlap_ratio=min_polygon_overlap_ratio or DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
                    min_positive_pixel_ratio=min_positive_pixel_ratio or DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
                    no_feature_tile_policy=no_feature_policy,
                    negative_ratio=negative_ratio if negative_ratio is not None else DEFAULT_TSFT_NEGATIVE_RATIO,
                    slicing_aoi_path=slicing_aoi_path,
                )
            ),
            notes=str(payload.get("notes") or ""),
            version=version,
            standardize=True,
            overwrite=bool(payload.get("overwrite")),
            resume=bool(payload.get("resume")),
            upload_method=upload_method,
            tile_size=tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            stride=stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            daft_max_nodata_ratio=max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            slicing_aoi_path=slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            tsft_min_polygon_overlap_ratio=(
                min_polygon_overlap_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            tsft_min_positive_pixel_ratio=(
                min_positive_pixel_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            tsft_no_feature_tile_policy=no_feature_policy,
            tsft_negative_ratio=negative_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            progress_callback=progress_callback,
        )
    elif dataset_class == "TSFT":
        upload_method = str(payload.get("upload_method") or DAFT_UPLOAD_METHOD_PACKAGED)
        tile_size = _optional_int(payload.get("tile_size")) or 0
        stride = _optional_int(payload.get("stride")) or 0
        max_nodata_ratio = _optional_float(payload.get("max_nodata_ratio"))
        min_polygon_overlap_ratio = _optional_float(payload.get("min_polygon_overlap_ratio"))
        min_positive_pixel_ratio = _optional_float(payload.get("min_positive_pixel_ratio"))
        negative_ratio = _optional_float(payload.get("negative_ratio"))
        no_feature_policy = str(payload.get("no_feature_tile_policy") or TSFT_NO_FEATURE_POLICY_BALANCED)
        slicing_aoi_path = str(payload.get("slicing_aoi_path") or "")
        entry = register_dataset(
            registry_root=registry_root,
            name=version,
            dataset_class="TSFT",
            imagery_path=str(payload.get("train_images") or ""),
            label_path=str(payload.get("train_labels") or "") or None,
            has_gt=True,
            RGB_bands=str(payload.get("RGB_bands") or DEFAULT_RGB_BANDS),
            label_polarity=str(payload.get("label_polarity") or LABEL_POLARITIES[0]),
            tile_schema=str(
                payload.get("tile_schema")
                or _tsft_tile_schema(
                    upload_method=upload_method,
                    tile_size=tile_size,
                    stride=stride,
                    max_nodata_ratio=max_nodata_ratio or 0.20,
                    min_polygon_overlap_ratio=min_polygon_overlap_ratio or DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO,
                    min_positive_pixel_ratio=min_positive_pixel_ratio or DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO,
                    no_feature_tile_policy=no_feature_policy,
                    negative_ratio=negative_ratio if negative_ratio is not None else DEFAULT_TSFT_NEGATIVE_RATIO,
                    slicing_aoi_path=slicing_aoi_path,
                )
            ),
            notes=str(payload.get("notes") or ""),
            version=version,
            standardize=True,
            overwrite=bool(payload.get("overwrite")),
            resume=bool(payload.get("resume")),
            upload_method=upload_method,
            tile_size=tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            stride=stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            daft_max_nodata_ratio=max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            slicing_aoi_path=slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            tsft_min_polygon_overlap_ratio=(
                min_polygon_overlap_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            tsft_min_positive_pixel_ratio=(
                min_positive_pixel_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            tsft_no_feature_tile_policy=no_feature_policy,
            tsft_negative_ratio=negative_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            progress_callback=progress_callback,
        )
    else:
        raise RegistryError(f"Unsupported dataset upload class: {dataset_class}")

    sample_count = getattr(entry, "sample_count", 0)
    progress_callback(
        {
            "status": "running",
            "message": f"Registered {dataset_class} {version}.",
            "processed": sample_count,
            "total": sample_count,
            "percent": 1.0,
        }
    )
    return entry


def _coerce_subset_overrides(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, Mapping)}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _render_daft_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### DAFT Images-only Dataset")
    method_label = st.radio(
        "DAFT upload method",
        ("Packaged Images", "Tiff Slicing"),
        horizontal=True,
        key="daft_upload_method_label",
    )
    upload_method = (
        DAFT_UPLOAD_METHOD_TIFF_SLICING
        if method_label == "Tiff Slicing"
        else DAFT_UPLOAD_METHOD_PACKAGED
    )
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        _render_daft_tiff_discovery_controls(st)
    selected_raster_list = str(st.session_state.get("daft_selected_raster_list_path") or "")
    with st.form("daft_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("DAFT version", "DAFT-target-v1")
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and selected_raster_list:
                imagery_path = selected_raster_list
                st.text_input(
                    "image directory or GeoTIFF path",
                    value=selected_raster_list,
                    disabled=True,
                    help="Using the confirmed TIFF selection from Discover GeoTIFF rasters above.",
                )
                st.caption("Using confirmed TIFF selection from the discovery panel.")
            else:
                imagery_path = st.text_input(
                    "image directory or GeoTIFF path",
                    "",
                    help=(
                        "For Tiff Slicing, use Discover GeoTIFF rasters above to select main imagery "
                        "from a folder, or enter one .tif/.tiff path directly."
                    ),
                )
            slicing_aoi_path = st.text_input(
                "processing AOI path",
                "",
                disabled=upload_method != DAFT_UPLOAD_METHOD_TIFF_SLICING,
                help="Optional .shp/.gpkg/.geojson polygon AOI. Tiff Slicing keeps only windows strictly inside this AOI; windows touching the AOI boundary are skipped.",
            )
            RGB_bands = _rgb_bands_input(st, "RGB bands", key="daft_RGB_bands")
        with right:
            tile_size = st.number_input(
                "window size",
                min_value=1,
                value=512,
                step=1,
                disabled=upload_method != DAFT_UPLOAD_METHOD_TIFF_SLICING,
            )
            stride = st.number_input(
                "stride",
                min_value=1,
                value=256,
                step=1,
                disabled=upload_method != DAFT_UPLOAD_METHOD_TIFF_SLICING,
            )
            max_nodata_ratio = st.number_input(
                "max nodata ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.05,
                format="%.2f",
                disabled=upload_method != DAFT_UPLOAD_METHOD_TIFF_SLICING,
            )
            overwrite = st.checkbox("Overwrite Standardized Target", value=False)
            notes = st.text_area("DAFT notes", "")
        add_dataset = st.form_submit_button("Upload DAFT Dataset")
    if add_dataset:
        if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING and selected_raster_list:
            imagery_path = selected_raster_list
        target_status: dict[str, Any] = {}
        if not overwrite:
            try:
                target_status = inspect_daft_upload(
                    registry_root=registry_root,
                    version=version,
                    imagery_path=imagery_path,
                    RGB_bands=RGB_bands,
                    upload_method=upload_method,
                    tile_size=int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    stride=int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    daft_max_nodata_ratio=float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    slicing_aoi_path=slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                )
            except RegistryError as exc:
                st.error(str(exc))
                return
        if target_status.get("exists") and not overwrite:
            if target_status.get("config_matches") and not target_status.get("has_manifest"):
                st.session_state["pending_daft_upload"] = {
                    "version": version,
                    "imagery_path": imagery_path,
                    "RGB_bands": RGB_bands,
                    "notes": notes,
                    "upload_method": upload_method,
                    "tile_size": int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    "stride": int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    "max_nodata_ratio": float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                    "slicing_aoi_path": slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else "",
                    "target_status": target_status,
                }
                st.warning("Detected a partial DAFT upload with the same configuration. Choose resume, overwrite, or cancel.")
                st.rerun()
            elif target_status.get("has_manifest"):
                st.error("This DAFT version is already registered. Choose overwrite or use a different version.")
            else:
                st.error("This DAFT version already has a target directory from a different configuration. Choose overwrite or use a different version.")
            return
        _start_daft_upload_job(
            st,
            registry_root=registry_root,
            version=version,
            imagery_path=imagery_path,
            RGB_bands=RGB_bands,
            notes=notes,
            upload_method=upload_method,
            overwrite=overwrite,
            resume=False,
            tile_size=int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            stride=int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            max_nodata_ratio=float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            slicing_aoi_path=slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else "",
        )

    pending = st.session_state.get("pending_daft_upload")
    if isinstance(pending, Mapping):
        status = pending.get("target_status") if isinstance(pending.get("target_status"), Mapping) else {}
        with st.container(border=True):
            st.warning(
                "Found a partial DAFT upload with the same configuration: "
                f"{status.get('image_count', 0)} standardized images already exist."
            )
            st.caption(str(status.get("root_dir") or ""))
            resume_col, overwrite_col, cancel_col = st.columns(3)
            if resume_col.button("Resume", key="resume_daft_upload"):
                st.session_state.pop("pending_daft_upload", None)
                _start_daft_upload_job(
                    st,
                    registry_root=registry_root,
                    version=str(pending.get("version") or ""),
                    imagery_path=str(pending.get("imagery_path") or ""),
                    RGB_bands=str(pending.get("RGB_bands") or DEFAULT_RGB_BANDS),
                    notes=str(pending.get("notes") or ""),
                    upload_method=str(pending.get("upload_method") or DAFT_UPLOAD_METHOD_PACKAGED),
                    overwrite=False,
                    resume=True,
                    tile_size=pending.get("tile_size") if pending.get("tile_size") is not None else None,
                    stride=pending.get("stride") if pending.get("stride") is not None else None,
                    max_nodata_ratio=pending.get("max_nodata_ratio") if pending.get("max_nodata_ratio") is not None else None,
                    slicing_aoi_path=str(pending.get("slicing_aoi_path") or ""),
                )
            if overwrite_col.button("Overwrite", key="overwrite_daft_upload"):
                st.session_state.pop("pending_daft_upload", None)
                _start_daft_upload_job(
                    st,
                    registry_root=registry_root,
                    version=str(pending.get("version") or ""),
                    imagery_path=str(pending.get("imagery_path") or ""),
                    RGB_bands=str(pending.get("RGB_bands") or DEFAULT_RGB_BANDS),
                    notes=str(pending.get("notes") or ""),
                    upload_method=str(pending.get("upload_method") or DAFT_UPLOAD_METHOD_PACKAGED),
                    overwrite=True,
                    resume=False,
                    tile_size=pending.get("tile_size") if pending.get("tile_size") is not None else None,
                    stride=pending.get("stride") if pending.get("stride") is not None else None,
                    max_nodata_ratio=pending.get("max_nodata_ratio") if pending.get("max_nodata_ratio") is not None else None,
                    slicing_aoi_path=str(pending.get("slicing_aoi_path") or ""),
                )
            if cancel_col.button("Cancel", key="cancel_daft_upload_choice"):
                st.session_state.pop("pending_daft_upload", None)
                st.info("DAFT upload was cancelled.")
                st.rerun()


def _start_daft_upload_job(
    st: Any,
    *,
    registry_root: str,
    version: str,
    imagery_path: str,
    RGB_bands: str,
    notes: str,
    upload_method: str,
    overwrite: bool,
    resume: bool,
    tile_size: int | None,
    stride: int | None,
    max_nodata_ratio: float | None,
    slicing_aoi_path: str = "",
) -> None:
    payload = {
        "dataset_class": "DAFT",
        "version": version,
        "imagery_path": imagery_path,
        "RGB_bands": RGB_bands,
        "tile_schema": _daft_tile_schema(
            upload_method,
            tile_size or 0,
            stride or 0,
            max_nodata_ratio or 0.20,
            slicing_aoi_path=slicing_aoi_path,
        ),
        "notes": notes,
        "overwrite": overwrite,
        "upload_method": upload_method,
        "resume": resume,
        "tile_size": tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "stride": stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "max_nodata_ratio": max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "slicing_aoi_path": slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else "",
    }
    job_kind = "dataset-upload-daft-resume" if resume else "dataset-upload-daft"
    job_id = stable_job_id(kind=job_kind, version=version, payload=payload)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
        return run_dataset_upload_background_task(
            registry_root=registry_root,
            payload=payload,
            progress_callback=progress_callback,
        )

    try:
        job = start_process_background_job(
            registry_root=registry_root,
            job_id=job_id,
            kind="dataset-upload",
            description=f"DAFT upload {version}",
            payload=payload,
            target=target,
        )
        st.session_state[_dataset_upload_session_key("DAFT")] = job.job_id
        st.success(f"DAFT upload started: {job.job_id}")
        st.rerun()
    except RegistryError as exc:
        st.error(str(exc))


def _render_evaluation_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### Evaluation Dataset")
    method_label = st.radio(
        "Evaluation upload method",
        ("Packaged Images + Labels", "Tiff Slicing"),
        horizontal=True,
        key="evaluation_upload_method_label",
        help="Use packaged samples when images/labels are already chipped; use Tiff Slicing to cut evaluation chips from GeoTIFF imagery and polygon labels.",
    )
    upload_method = DAFT_UPLOAD_METHOD_TIFF_SLICING if method_label == "Tiff Slicing" else DAFT_UPLOAD_METHOD_PACKAGED
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        _render_evaluation_tiff_discovery_controls(st)
    selected_raster_list = str(st.session_state.get("evaluation_selected_raster_list_path") or "")
    with st.form("evaluation_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("Evaluation version", "evaluation-v1", key="evaluation_upload_version")
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                samples = ""
                if selected_raster_list:
                    imagery_path = selected_raster_list
                    st.text_input(
                        "source GeoTIFF path",
                        value=selected_raster_list,
                        disabled=True,
                        help="Using the confirmed TIFF selection from Discover GeoTIFF rasters above.",
                    )
                    st.caption("Using confirmed TIFF selection from the discovery panel.")
                else:
                    imagery_path = st.text_input(
                        "source GeoTIFF path",
                        "",
                        key="evaluation_tiff_source_path",
                        help="Single .tif/.tiff or a folder containing candidate GeoTIFF rasters. Use discovery above to select specific files.",
                    )
                label_path = st.text_input(
                    "landslide polygons",
                    "",
                    key="evaluation_tiff_label_path",
                    help="Required .shp, .gpkg, .geojson, or .json polygon file used to rasterize evaluation labels.",
                )
                slicing_aoi_path = st.text_input(
                    "processing AOI path",
                    "",
                    key="evaluation_tiff_aoi_path",
                    help="Optional .shp/.gpkg/.geojson polygon AOI. Tiff Slicing keeps fixed windows strictly inside this AOI.",
                )
            else:
                samples = st.text_input("evaluation samples", "", key="evaluation_upload_samples")
                imagery_path = ""
                label_path = ""
                slicing_aoi_path = ""
            RGB_bands = _rgb_bands_input(st, "RGB bands", key="evaluation_RGB_bands")
            label_polarity = st.selectbox("Label Foreground", LABEL_POLARITIES, key="evaluation_label_polarity")
        with right:
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                tile_size = st.number_input(
                    "Window Size",
                    min_value=1,
                    value=1024,
                    step=1,
                    key="evaluation_tiff_tile_size",
                    help="Sliding-window chip size in pixels.",
                )
                stride = st.number_input(
                    "stride",
                    min_value=1,
                    value=512,
                    step=1,
                    key="evaluation_tiff_stride",
                    help="Sliding-window step in pixels.",
                )
                max_nodata_ratio = st.number_input(
                    "max nodata ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.05,
                    format="%.2f",
                    key="evaluation_tiff_max_nodata_ratio",
                    help="Discard windows whose invalid or all-zero RGB pixels exceed this ratio.",
                )
                min_polygon_overlap_ratio = st.number_input(
                    "min polygon overlap ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO),
                    step=0.05,
                    format="%.3f",
                    key="evaluation_tiff_min_polygon_overlap_ratio",
                    help="Minimum fraction of a polygon that must overlap the window before it is rasterized into the label.",
                )
                min_positive_pixel_ratio = st.number_input(
                    "min positive pixel ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO),
                    step=0.001,
                    format="%.3f",
                    key="evaluation_tiff_min_positive_pixel_ratio",
                    help="Discard positive windows when landslide pixels are below this fraction of valid pixels.",
                )
                policy_labels = {
                    "only_with_features": "only_with_features - positives only",
                    "balanced": "balanced - positives plus sampled negatives",
                    "all_valid": "all_valid - positives plus all valid negatives",
                }
                policy_display_options = [policy_labels.get(policy, policy) for policy in TSFT_NO_FEATURE_POLICIES]
                selected_policy_label = st.selectbox(
                    "no-feature tile policy",
                    policy_display_options,
                    index=list(TSFT_NO_FEATURE_POLICIES).index(TSFT_NO_FEATURE_POLICY_ALL),
                    key="evaluation_tiff_no_feature_policy",
                    help="Controls no-landslide windows after nodata/validity filtering.",
                )
                no_feature_policy = next(
                    policy
                    for policy, label in policy_labels.items()
                    if label == selected_policy_label
                )
                negative_ratio_input = st.number_input(
                    "negative ratio",
                    min_value=0.0,
                    value=float(DEFAULT_TSFT_NEGATIVE_RATIO),
                    step=0.25,
                    format="%.2f",
                    disabled=no_feature_policy != TSFT_NO_FEATURE_POLICY_BALANCED,
                    key="evaluation_tiff_negative_ratio",
                    help="Only used by balanced policy: maximum negatives equals positive_count times this ratio.",
                )
                negative_ratio = (
                    float(negative_ratio_input)
                    if no_feature_policy == TSFT_NO_FEATURE_POLICY_BALANCED
                    else float(DEFAULT_TSFT_NEGATIVE_RATIO)
                )
                resume = st.checkbox(
                    "resume matching partial target",
                    value=False,
                    key="evaluation_tiff_resume",
                    help="Checked: continue an existing matching import. Unchecked: rebuild this version from scratch.",
                )
                overwrite = not bool(resume)
            else:
                tile_size = 0
                stride = 0
                max_nodata_ratio = 0.20
                min_polygon_overlap_ratio = DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO
                min_positive_pixel_ratio = DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO
                no_feature_policy = TSFT_NO_FEATURE_POLICY_ALL
                negative_ratio = DEFAULT_TSFT_NEGATIVE_RATIO
                resume = False
                overwrite = st.checkbox("Overwrite Standardized Target", value=False, key="evaluation_overwrite")
            notes = st.text_area("Evaluation notes", "", key="evaluation_notes")
        add_dataset = st.form_submit_button("Upload Evaluation Dataset")
    if add_dataset:
        try:
            if upload_method == DAFT_UPLOAD_METHOD_PACKAGED:
                imagery_path, label_path = _tsft_sample_folder_paths(samples, split="evaluation", required=True)
            elif selected_raster_list:
                imagery_path = selected_raster_list
        except RegistryError as exc:
            st.error(str(exc))
            return
        payload = {
            "dataset_class": "Evaluation",
            "version": version,
            "samples": samples if upload_method == DAFT_UPLOAD_METHOD_PACKAGED else "",
            "imagery_path": imagery_path,
            "label_path": label_path,
            "RGB_bands": RGB_bands,
            "label_polarity": label_polarity,
            "notes": notes,
            "overwrite": overwrite,
            "resume": resume,
            "upload_method": upload_method,
            "tile_schema": (
                "evaluation_images_and_labels"
                if upload_method == DAFT_UPLOAD_METHOD_PACKAGED
                else _tsft_tile_schema(
                    upload_method=upload_method,
                    tile_size=int(tile_size),
                    stride=int(stride),
                    max_nodata_ratio=float(max_nodata_ratio),
                    min_polygon_overlap_ratio=float(min_polygon_overlap_ratio),
                    min_positive_pixel_ratio=float(min_positive_pixel_ratio),
                    no_feature_tile_policy=no_feature_policy,
                    negative_ratio=float(negative_ratio),
                    slicing_aoi_path=slicing_aoi_path,
                )
            ),
            "tile_size": int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "stride": int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "max_nodata_ratio": float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "min_polygon_overlap_ratio": (
                float(min_polygon_overlap_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            "min_positive_pixel_ratio": (
                float(min_positive_pixel_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            "no_feature_tile_policy": no_feature_policy if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "negative_ratio": float(negative_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "slicing_aoi_path": slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else "",
        }
        job_id = stable_job_id(kind="dataset-upload-evaluation", version=version, payload=payload)
        existing_job = _job_state_for_id(registry_root, job_id)
        if _job_is_blocking_resubmission(existing_job, registry_root=registry_root, payload=payload):
            st.session_state[_dataset_upload_session_key("Evaluation")] = job_id
            status = str(existing_job.get("status") or "")
            if status == "completed":
                st.success(f"Evaluation dataset '{version}' was already created. Showing the completed job below.")
            else:
                st.info(f"Evaluation upload for '{version}' is already {status}. Showing the running job below.")
            _render_module_job_progress(st, registry_root=registry_root, job=existing_job, title="Current Evaluation dataset upload")
            return
        if _dataset_version_exists(registry_root, "Evaluation", version) and not bool(resume) and not bool(overwrite):
            st.warning(f"Evaluation dataset version '{version}' already exists. Choose a new version name or enable overwrite/resume.")
            return

        def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
            return run_dataset_upload_background_task(
                registry_root=registry_root,
                payload=payload,
                progress_callback=progress_callback,
            )

        try:
            job = start_process_background_job(
                registry_root=registry_root,
                job_id=job_id,
                kind="dataset-upload",
                description=f"Evaluation upload {version}",
                payload=payload,
                target=target,
            )
            st.session_state[_dataset_upload_session_key("Evaluation")] = job.job_id
            started_state = read_job_state(registry_root, job.job_id)
            st.success(f"Evaluation upload submitted: {job.job_id}. Progress is shown below.")
            if started_state:
                _render_module_job_progress(st, registry_root=registry_root, job=started_state, title="Current Evaluation dataset upload")
            st.rerun()
        except RegistryError as exc:
            st.error(str(exc))


def _render_tsft_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### TSFT Few-shot Dataset")
    method_label = st.radio(
        "TSFT upload method",
        ("Packaged Images + Labels", "Tiff Slicing"),
        horizontal=True,
        key="tsft_upload_method_label",
        help="Use packaged samples when images/labels are already chipped; use Tiff Slicing to cut chips from GeoTIFF imagery and polygon labels.",
    )
    upload_method = DAFT_UPLOAD_METHOD_TIFF_SLICING if method_label == "Tiff Slicing" else DAFT_UPLOAD_METHOD_PACKAGED
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        _render_tsft_tiff_discovery_controls(st, registry_root=registry_root)
    selected_raster_list = str(st.session_state.get("tsft_selected_raster_list_path") or "")
    with st.form("tsft_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("TSFT version", "TSFT-round-1", help="Name for this TSFT dataset version.")
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                train_samples = ""
                if selected_raster_list:
                    train_images = selected_raster_list
                    st.text_input(
                        "source GeoTIFF path",
                        value=selected_raster_list,
                        disabled=True,
                        help="Using the confirmed TIFF selection from Discover GeoTIFF rasters above.",
                    )
                    st.caption("Using confirmed TIFF selection from the discovery panel.")
                else:
                    train_images = st.text_input(
                        "source GeoTIFF path",
                        "",
                        help="Single .tif/.tiff or a folder containing candidate GeoTIFF rasters. Use discovery above to select specific files.",
                    )
                train_labels = st.text_input(
                    "landslide polygons",
                    "",
                    help="Optional .shp, .gpkg, .geojson, or .json polygon file. Leave empty to generate valid negative-only TSFT chips with all-zero labels.",
                )
                slicing_aoi_path = st.text_input(
                    "processing AOI path",
                    "",
                    help="Optional .shp/.gpkg/.geojson polygon AOI. Tiff Slicing keeps fixed windows strictly inside this AOI. In negative-only TSFT uploads, AOI polygons too small for one full window are letterbox-resized and padded into one negative chip.",
                )
            else:
                train_samples = st.text_input(
                    "train samples",
                    "",
                    help="Folder containing packaged TSFT samples under images/ and labels/.",
                )
                train_images = ""
                train_labels = ""
                slicing_aoi_path = ""
            RGB_bands = _rgb_bands_input(
                st,
                "RGB bands",
                key="tsft_RGB_bands",
                help="Model RGB input bands as 1-based GeoTIFF indexes, for example RGB_bands: 1,2,3.",
            )
            label_polarity = st.selectbox(
                "Label Foreground",
                LABEL_POLARITIES,
                key="tsft_label_polarity",
                help="Foreground convention for packaged labels; Tiff Slicing polygon labels are stored as foreground=255.",
            )
        with right:
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                test_images = ""
                test_labels = ""
                tile_size = st.number_input("Window Size", min_value=1, value=1024, step=1, help="Sliding-window chip size in pixels.")
                stride = st.number_input("stride", min_value=1, value=512, step=1, help="Sliding-window step in pixels.")
                max_nodata_ratio = st.number_input(
                    "max nodata ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.05,
                    format="%.2f",
                    help="Discard windows whose invalid or all-zero RGB pixels exceed this ratio.",
                )
                min_polygon_overlap_ratio = st.number_input(
                    "min polygon overlap ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO),
                    step=0.05,
                    format="%.3f",
                    help="Minimum fraction of a polygon that must overlap the window before it is rasterized into the label.",
                )
                min_positive_pixel_ratio = st.number_input(
                    "min positive pixel ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO),
                    step=0.001,
                    format="%.3f",
                    help="Discard positive windows when landslide pixels are below this fraction of valid pixels.",
                )
                policy_labels = {
                    "only_with_features": "only_with_features - positives only",
                    "balanced": "balanced - positives plus sampled negatives",
                    "all_valid": "all_valid - positives plus all valid negatives",
                }
                policy_display_options = [policy_labels.get(policy, policy) for policy in TSFT_NO_FEATURE_POLICIES]
                selected_policy_label = st.selectbox(
                    "no-feature tile policy",
                    policy_display_options,
                    index=list(TSFT_NO_FEATURE_POLICIES).index(TSFT_NO_FEATURE_POLICY_BALANCED),
                    key="tsft_no_feature_policy",
                    help="Controls no-landslide windows after nodata/validity filtering.",
                )
                no_feature_policy = next(
                    policy
                    for policy, label in policy_labels.items()
                    if label == selected_policy_label
                )
                negative_ratio_input = st.number_input(
                    "negative ratio",
                    min_value=0.0,
                    value=float(DEFAULT_TSFT_NEGATIVE_RATIO),
                    step=0.25,
                    format="%.2f",
                    disabled=no_feature_policy != TSFT_NO_FEATURE_POLICY_BALANCED,
                    help="Only used by balanced policy: maximum negatives equals positive_count times this ratio.",
                )
                negative_ratio = (
                    float(negative_ratio_input)
                    if no_feature_policy == TSFT_NO_FEATURE_POLICY_BALANCED
                    else float(DEFAULT_TSFT_NEGATIVE_RATIO)
                )
                resume = st.checkbox(
                    "resume matching partial target",
                    value=False,
                    help="Checked: continue an existing matching import. Unchecked: rebuild this version from scratch.",
                )
            else:
                test_images = ""
                test_labels = ""
                tile_size = 0
                stride = 0
                max_nodata_ratio = 0.20
                min_polygon_overlap_ratio = DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO
                min_positive_pixel_ratio = DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO
                no_feature_policy = TSFT_NO_FEATURE_POLICY_BALANCED
                negative_ratio = DEFAULT_TSFT_NEGATIVE_RATIO
                resume = False
            notes = st.text_area("TSFT notes", "", help="Optional notes saved into the dataset registry.")
        add_dataset = st.form_submit_button("Upload TSFT Dataset")
    if add_dataset:
        try:
            if upload_method == DAFT_UPLOAD_METHOD_PACKAGED:
                train_images, train_labels = _tsft_sample_folder_paths(train_samples, split="train", required=True)
            elif selected_raster_list:
                train_images = selected_raster_list
        except RegistryError as exc:
            st.error(str(exc))
            return
        payload = {
            "dataset_class": "TSFT",
            "version": version,
            "train_samples": train_samples if upload_method == DAFT_UPLOAD_METHOD_PACKAGED else "",
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "RGB_bands": RGB_bands,
            "label_polarity": label_polarity,
            "notes": notes,
            "overwrite": not bool(resume),
            "resume": resume,
            "upload_method": upload_method,
            "tile_size": int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "stride": int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "max_nodata_ratio": float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "min_polygon_overlap_ratio": (
                float(min_polygon_overlap_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            "min_positive_pixel_ratio": (
                float(min_positive_pixel_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
            ),
            "no_feature_tile_policy": no_feature_policy if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "negative_ratio": float(negative_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            "slicing_aoi_path": slicing_aoi_path if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else "",
            "small_aoi_letterbox_version": (
                1
                if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING
                and not str(train_labels or "").strip()
                and str(slicing_aoi_path or "").strip()
                else None
            ),
        }
        job_id = stable_job_id(kind="dataset-upload-tsft", version=version, payload=payload)
        existing_job = _job_state_for_id(registry_root, job_id)
        if _job_is_blocking_resubmission(existing_job, registry_root=registry_root, payload=payload):
            st.session_state[_dataset_upload_session_key("TSFT")] = job_id
            status = str(existing_job.get("status") or "")
            if status == "completed":
                st.success(f"TSFT dataset '{version}' was already created. Showing the completed job below.")
            else:
                st.info(f"TSFT upload for '{version}' is already {status}. Showing the running job below.")
            _render_module_job_progress(st, registry_root=registry_root, job=existing_job, title="Current TSFT dataset upload")
            return
        if _dataset_version_exists(registry_root, "TSFT", version) and not bool(resume):
            st.warning(f"TSFT dataset version '{version}' already exists. Choose a new version name or enable resume.")
            return

        def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
            return run_dataset_upload_background_task(
                registry_root=registry_root,
                payload=payload,
                progress_callback=progress_callback,
            )

        try:
            job = start_process_background_job(
                registry_root=registry_root,
                job_id=job_id,
                kind="dataset-upload",
                description=f"TSFT upload {version}",
                payload=payload,
                target=target,
            )
            st.session_state[_dataset_upload_session_key("TSFT")] = job.job_id
            started_state = read_job_state(registry_root, job.job_id)
            st.success(f"TSFT upload submitted: {job.job_id}. Progress is shown below.")
            if started_state:
                _render_module_job_progress(st, registry_root=registry_root, job=started_state, title="Current TSFT dataset upload")
            st.rerun()
        except RegistryError as exc:
            st.error(str(exc))


def _tsft_sample_folder_paths(sample_dir: str, *, split: str, required: bool) -> tuple[str, str]:
    text = str(sample_dir or "").strip()
    if not text:
        if required:
            raise RegistryError(f"{split} samples path is required.")
        return "", ""
    root = Path(text).expanduser()
    return str(root / "images"), str(root / "labels")



def _tiff_path_or_blank(path_text: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    try:
        path = Path(text).expanduser()
    except (OSError, ValueError):
        return ""
    return str(path) if path.suffix.lower() in {".tif", ".tiff"} else ""

def _render_inference_tiff_discovery_controls(st: Any, *, default_source: str, default_RGB_bands: str) -> None:
    with st.expander("Discover GeoTIFF rasters", expanded=False):
        source_path = st.text_input(
            "inference discover source path",
            str(default_source or ""),
            help="Optional. Enter a GeoTIFF file or folder, then discover and confirm rasters for batch Inference.",
            key="inference_discover_source_path",
        )
        RGB_bands = _rgb_bands_input(
            st,
            "inference discovery RGB bands",
            key="inference_discover_RGB_bands",
            default=default_RGB_bands,
            help="Used only to identify valid candidate rasters. Files with fewer bands than the largest requested band are not selected by default.",
        )
        aoi_text = st.text_input(
            "inference AOI",
            "",
            help=(
                "Optional geographic filter for batch Inference. Leave empty to use every discovered TIFF, "
                "or enter minx,miny,maxx,maxy in the TIFF CRS, or a .shp/.gpkg/.geojson AOI path."
            ),
            key="inference_discover_aoi",
        )
        controls = st.columns([1, 1, 2])
        discover_clicked = controls[0].button("Discover TIFFs", key="inference_discover_tiffs")
        clear_clicked = controls[1].button("Clear TIFF selection", key="inference_clear_tiffs")
        controls[2].caption("Discover, review, and manually adjust candidate rasters before deployment.")
        if clear_clicked:
            st.session_state.pop("inference_discovered_tiffs", None)
            st.session_state.pop("inference_selected_rasters", None)
            st.session_state.pop("inference_selected_raster_count", None)
            st.session_state.pop("inference_selected_raster_RGB_bands", None)
            st.session_state.pop("inference_selected_raster_aoi", None)
            st.session_state.pop("inference_discovered_tiff_aoi", None)
        if discover_clicked:
            try:
                st.session_state["inference_discovered_tiffs"] = _discover_tiff_candidates_for_ui(
                    source_path,
                    RGB_bands=RGB_bands,
                    aoi=aoi_text,
                )
                st.session_state.pop("inference_selected_rasters", None)
                st.session_state.pop("inference_selected_raster_count", None)
                st.session_state["inference_selected_raster_RGB_bands"] = RGB_bands
                st.session_state["inference_discovered_tiff_aoi"] = str(aoi_text or "").strip()
            except RegistryError as exc:
                st.error(str(exc))
        rows = st.session_state.get("inference_discovered_tiffs")
        if not isinstance(rows, list) or not rows:
            return
        with st.form("inference_tiff_selection"):
            st.caption("Confirm rasters for the next Inference deployment")
            edited_rows = st.data_editor(
                rows,
                use_container_width=True,
                hide_index=True,
                disabled=("name", "parent", "bands", "width", "height", "valid_for_RGB_bands", "reason", "path"),
                column_config={
                    "selected": st.column_config.CheckboxColumn("selected"),
                    "valid_for_RGB_bands": st.column_config.CheckboxColumn("valid"),
                    "path": st.column_config.TextColumn("path", width="large"),
                },
                key="inference_tiff_selection_editor",
            )
            selected = [
                str(row["path"])
                for row in _rows_from_data_editor_result(edited_rows)
                if bool(row.get("selected")) and bool(row.get("valid_for_RGB_bands", True))
            ]
            apply_selection = st.form_submit_button("Use selected TIFFs for Inference")
        if apply_selection:
            if not selected:
                st.warning("Select at least one GeoTIFF.")
                return
            st.session_state["inference_selected_rasters"] = selected
            st.session_state["inference_selected_raster_count"] = len(selected)
            st.session_state["inference_selected_raster_RGB_bands"] = RGB_bands
            st.session_state["inference_selected_raster_aoi"] = str(
                st.session_state.get("inference_discovered_tiff_aoi") or ""
            ).strip()
            _set_rgb_bands_preset_state(st, "inference_RGB_bands", RGB_bands)
            st.success(f"Using {len(selected)} selected raster(s) for batch Inference.")


def _reset_inference_session_state(st: Any) -> None:
    for key in (
        "lsd_app_defaults",
        "active_deployment_job_id",
        "active_batch_deployment_job_id",
        "inference_discovered_tiffs",
        "inference_selected_rasters",
        "inference_selected_raster_count",
        "inference_selected_raster_RGB_bands",
        "inference_selected_raster_aoi",
        "inference_discover_aoi",
        "inference_discovered_tiff_aoi",
        "inference_tiff_selection_editor",
    ):
        st.session_state.pop(key, None)


def _render_inference_current_config_summary(
    st: Any,
    *,
    registry_root: str,
    selected_weight_id: str,
    model: str,
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    batch_count: int,
    batch_aoi: str = "",
) -> None:
    checkpoint_label = checkpoint_path or "not configured"
    if checkpoint_label.startswith("@"):
        checkpoint_label = f"{checkpoint_label} -> {Path(registry_root).expanduser().resolve()}"
    rows = [
        {"field": "weight", "value": selected_weight_id},
        {"field": "model", "value": model},
        {"field": "input", "value": input_path},
        {"field": "output", "value": output_dir},
        {"field": "checkpoint", "value": checkpoint_label},
        {"field": "batch rasters", "value": str(batch_count) if batch_count else "single raster"},
        {"field": "batch AOI", "value": str(batch_aoi or "").strip() or "all discovered rasters"},
        {"field": "registry", "value": str(Path(registry_root).expanduser().resolve())},
    ]
    with st.expander("Current Inference configuration", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _selected_inference_rasters_from_state() -> list[str]:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return []
    value = st.session_state.get("inference_selected_rasters")
    if not isinstance(value, list):
        return []
    return [str(path) for path in value if str(path or "").strip()]


def _render_tsft_tiff_discovery_controls(st: Any, *, registry_root: str) -> None:
    recent_key = _recent_path_key("dataset_upload", "tsft", "tiff_discovery", "source_path")
    source_state_key = "tsft_discover_source_path"
    st.session_state.setdefault(source_state_key, "")
    with st.expander("Discover GeoTIFF rasters", expanded=True):
        source_path = st.text_input(
            "TSFT discover source path",
            help="Optional. Enter a GeoTIFF file or folder, then discover and confirm the rasters used by Tiff Slicing.",
            key=source_state_key,
        )
        _render_recent_path_datalist(
            st,
            label="TSFT discover source path",
            values=_recent_path_values(registry_root, recent_key),
            datalist_id="tsft_discover_source_path_recent",
        )
        RGB_bands = _rgb_bands_input(
            st,
            "discovery RGB bands",
            key="tsft_discover_RGB_bands",
            help="Used only to identify valid candidate rasters. Files with fewer bands than the largest requested band are not selected by default.",
        )
        controls = st.columns([1, 1, 2])
        discover_clicked = controls[0].button("Discover TIFFs", key="tsft_discover_tiffs")
        clear_clicked = controls[1].button("Clear TIFF selection", key="tsft_clear_tiffs")
        controls[2].caption("Discover, review, and manually adjust candidate rasters before upload.")
        if clear_clicked:
            st.session_state.pop("tsft_discovered_tiffs", None)
            st.session_state.pop("tsft_selected_raster_list_path", None)
            st.session_state.pop("tsft_selected_raster_count", None)
            st.session_state.pop("tsft_selected_raster_RGB_bands", None)
        if discover_clicked:
            try:
                st.session_state["tsft_discovered_tiffs"] = _discover_tiff_candidates_for_ui(
                    source_path,
                    RGB_bands=RGB_bands,
                )
                _remember_recent_path(registry_root, recent_key, source_path)
                st.session_state.pop("tsft_selected_raster_list_path", None)
                st.session_state.pop("tsft_selected_raster_count", None)
                st.session_state["tsft_selected_raster_RGB_bands"] = RGB_bands
            except RegistryError as exc:
                st.error(str(exc))
        rows = st.session_state.get("tsft_discovered_tiffs")
        if not isinstance(rows, list) or not rows:
            return
        with st.form("tsft_tiff_selection"):
            st.caption("Confirm rasters for the next TSFT upload")
            edited_rows = st.data_editor(
                rows,
                use_container_width=True,
                hide_index=True,
                disabled=("name", "parent", "bands", "width", "height", "valid_for_RGB_bands", "reason", "path"),
                column_config={
                    "selected": st.column_config.CheckboxColumn("selected"),
                    "valid_for_RGB_bands": st.column_config.CheckboxColumn("valid"),
                    "path": st.column_config.TextColumn("path", width="large"),
                },
                key="tsft_tiff_selection_editor",
            )
            selected = [
                str(row["path"])
                for row in _rows_from_data_editor_result(edited_rows)
                if bool(row.get("selected")) and bool(row.get("valid_for_RGB_bands", True))
            ]
            apply_selection = st.form_submit_button("Use selected TIFFs")
        if apply_selection:
            if not selected:
                st.warning("Select at least one GeoTIFF.")
                return
            list_path = _write_tsft_raster_list_file(selected)
            st.session_state["tsft_selected_raster_list_path"] = str(list_path)
            st.session_state["tsft_selected_raster_count"] = len(selected)
            st.session_state["tsft_selected_raster_RGB_bands"] = RGB_bands
            _remember_recent_path(registry_root, recent_key, source_path)
            _set_rgb_bands_preset_state(st, "tsft_RGB_bands", RGB_bands)
            st.success(f"Using {len(selected)} selected raster(s): {list_path}")


def _render_evaluation_tiff_discovery_controls(st: Any) -> None:
    with st.expander("Discover GeoTIFF rasters", expanded=True):
        source_path = st.text_input(
            "Evaluation discover source path",
            "",
            help="Optional. Enter a GeoTIFF file or folder, then discover and confirm the rasters used by Evaluation Tiff Slicing.",
            key="evaluation_discover_source_path",
        )
        RGB_bands = _rgb_bands_input(
            st,
            "Evaluation discovery RGB bands",
            key="evaluation_discover_RGB_bands",
            help="Used only to identify valid candidate rasters. Files with fewer bands than the largest requested band are not selected by default.",
        )
        controls = st.columns([1, 1, 2])
        discover_clicked = controls[0].button("Discover TIFFs", key="evaluation_discover_tiffs")
        clear_clicked = controls[1].button("Clear TIFF selection", key="evaluation_clear_tiffs")
        controls[2].caption("Discover, review, and manually adjust candidate rasters before Evaluation upload.")
        if clear_clicked:
            st.session_state.pop("evaluation_discovered_tiffs", None)
            st.session_state.pop("evaluation_selected_raster_list_path", None)
            st.session_state.pop("evaluation_selected_raster_count", None)
            st.session_state.pop("evaluation_selected_raster_RGB_bands", None)
        if discover_clicked:
            try:
                st.session_state["evaluation_discovered_tiffs"] = _discover_tiff_candidates_for_ui(
                    source_path,
                    RGB_bands=RGB_bands,
                )
                st.session_state.pop("evaluation_selected_raster_list_path", None)
                st.session_state.pop("evaluation_selected_raster_count", None)
                st.session_state["evaluation_selected_raster_RGB_bands"] = RGB_bands
            except RegistryError as exc:
                st.error(str(exc))
        rows = st.session_state.get("evaluation_discovered_tiffs")
        if not isinstance(rows, list) or not rows:
            return
        with st.form("evaluation_tiff_selection"):
            st.caption("Confirm rasters for the next Evaluation upload")
            edited_rows = st.data_editor(
                rows,
                use_container_width=True,
                hide_index=True,
                disabled=("name", "parent", "bands", "width", "height", "valid_for_RGB_bands", "reason", "path"),
                column_config={
                    "selected": st.column_config.CheckboxColumn("selected"),
                    "valid_for_RGB_bands": st.column_config.CheckboxColumn("valid"),
                    "path": st.column_config.TextColumn("path", width="large"),
                },
                key="evaluation_tiff_selection_editor",
            )
            selected = [
                str(row["path"])
                for row in _rows_from_data_editor_result(edited_rows)
                if bool(row.get("selected")) and bool(row.get("valid_for_RGB_bands", True))
            ]
            apply_selection = st.form_submit_button("Use selected TIFFs")
        if apply_selection:
            if not selected:
                st.warning("Select at least one GeoTIFF.")
                return
            list_path = _write_tiff_raster_list_file(selected, prefix="evaluation")
            st.session_state["evaluation_selected_raster_list_path"] = str(list_path)
            st.session_state["evaluation_selected_raster_count"] = len(selected)
            st.session_state["evaluation_selected_raster_RGB_bands"] = RGB_bands
            _set_rgb_bands_preset_state(st, "evaluation_RGB_bands", RGB_bands)
            st.success(f"Using {len(selected)} selected raster(s): {list_path}")


def _render_daft_tiff_discovery_controls(st: Any) -> None:
    with st.expander("Discover GeoTIFF rasters", expanded=True):
        source_path = st.text_input(
            "DAFT discover source path",
            "",
            help="Optional. Enter a GeoTIFF file or folder, then discover and confirm the rasters used by DAFT Tiff Slicing.",
            key="daft_discover_source_path",
        )
        RGB_bands = _rgb_bands_input(
            st,
            "DAFT discovery RGB bands",
            key="daft_discover_RGB_bands",
            help="Used only to identify valid candidate rasters. Files with fewer bands than the largest requested band are not selected by default.",
        )
        controls = st.columns([1, 1, 2])
        discover_clicked = controls[0].button("Discover TIFFs", key="daft_discover_tiffs")
        clear_clicked = controls[1].button("Clear TIFF selection", key="daft_clear_tiffs")
        controls[2].caption("Discover, review, and manually adjust candidate rasters before DAFT upload.")
        if clear_clicked:
            st.session_state.pop("daft_discovered_tiffs", None)
            st.session_state.pop("daft_selected_raster_list_path", None)
            st.session_state.pop("daft_selected_raster_count", None)
            st.session_state.pop("daft_selected_raster_RGB_bands", None)
        if discover_clicked:
            try:
                st.session_state["daft_discovered_tiffs"] = _discover_tiff_candidates_for_ui(
                    source_path,
                    RGB_bands=RGB_bands,
                )
                st.session_state.pop("daft_selected_raster_list_path", None)
                st.session_state.pop("daft_selected_raster_count", None)
                st.session_state["daft_selected_raster_RGB_bands"] = RGB_bands
            except RegistryError as exc:
                st.error(str(exc))
        rows = st.session_state.get("daft_discovered_tiffs")
        if not isinstance(rows, list) or not rows:
            return
        with st.form("daft_tiff_selection"):
            st.caption("Confirm rasters for the next DAFT upload")
            edited_rows = st.data_editor(
                rows,
                use_container_width=True,
                hide_index=True,
                disabled=("name", "parent", "bands", "width", "height", "valid_for_RGB_bands", "reason", "path"),
                column_config={
                    "selected": st.column_config.CheckboxColumn("selected"),
                    "valid_for_RGB_bands": st.column_config.CheckboxColumn("valid"),
                    "path": st.column_config.TextColumn("path", width="large"),
                },
                key="daft_tiff_selection_editor",
            )
            selected = [
                str(row["path"])
                for row in _rows_from_data_editor_result(edited_rows)
                if bool(row.get("selected")) and bool(row.get("valid_for_RGB_bands", True))
            ]
            apply_selection = st.form_submit_button("Use selected TIFFs")
        if apply_selection:
            if not selected:
                st.warning("Select at least one GeoTIFF.")
                return
            list_path = _write_tiff_raster_list_file(selected, prefix="daft")
            st.session_state["daft_selected_raster_list_path"] = str(list_path)
            st.session_state["daft_selected_raster_count"] = len(selected)
            st.session_state["daft_selected_raster_RGB_bands"] = RGB_bands
            _set_rgb_bands_preset_state(st, "daft_RGB_bands", RGB_bands)
            st.success(f"Using {len(selected)} selected raster(s): {list_path}")


def _discover_tiff_candidates_for_ui(
    path_text: str,
    *,
    RGB_bands: str = DEFAULT_RGB_BANDS,
    aoi: str = "",
) -> list[dict[str, Any]]:
    text = str(path_text or "").strip()
    if not text:
        raise RegistryError("Enter a GeoTIFF file or folder to discover.")
    root = Path(text).expanduser()
    if not root.exists():
        raise RegistryError(f"GeoTIFF discovery path does not exist: {root}")
    required_band = max(rgb_bands_to_indexes(normalize_rgb_bands(RGB_bands)))
    if root.is_file():
        if root.suffix.lower() not in {".tif", ".tiff"}:
            raise RegistryError(f"GeoTIFF discovery requires .tif/.tiff, got: {root}")
        files = [root.resolve()]
    else:
        files = sorted(
            discover_candidate_tiff_rasters(root),
            key=lambda item: (0 if item.name.lower().startswith("tile_") else 1, str(item)),
        )
    if not files:
        raise RegistryError(f"No .tif/.tiff files found under {root}.")
    files = [Path(path) for path in _filter_inference_rasters_by_aoi(files, aoi)]
    if not files:
        raise RegistryError(f"No discovered GeoTIFF intersects the Inference AOI: {aoi}")
    rows = [_inspect_tiff_candidate_for_ui(file, required_band=required_band, single_file=len(files) == 1) for file in files]
    valid_count = sum(1 for row in rows if row["valid_for_RGB_bands"])
    if valid_count <= 0:
        examples = ", ".join(f"{row['name']} ({row.get('bands', '?')} band)" for row in rows[:5])
        raise RegistryError(
            f"No discovered GeoTIFF has enough bands for {RGB_bands}. "
            f"Need at least band {required_band}. Examples: {examples}"
        )
    return rows


def _filter_inference_rasters_by_aoi(paths: Sequence[str | Path], aoi: str = "") -> list[str]:
    aoi_text = str(aoi or "").strip()
    raster_paths = [Path(path).expanduser().resolve() for path in paths if str(path or "").strip()]
    if not aoi_text:
        return [str(path) for path in raster_paths]
    aoi_filter = _parse_inference_aoi_filter(aoi_text)
    filtered: list[str] = []
    skipped: list[str] = []
    for path in raster_paths:
        if _inference_raster_intersects_aoi(path, aoi_filter):
            filtered.append(str(path))
        else:
            skipped.append(str(path))
    if not filtered:
        examples = ", ".join(skipped[:5])
        raise RegistryError(f"Inference AOI excludes all discovered GeoTIFF rasters. Skipped examples: {examples}")
    return filtered


def _parse_inference_aoi_filter(aoi: str) -> dict[str, Any]:
    text = str(aoi or "").strip()
    if not text:
        return {"kind": "none"}
    path = Path(text).expanduser()
    if path.exists():
        if path.suffix.lower() not in {".shp", ".gpkg", ".geojson", ".json"}:
            raise RegistryError("Inference AOI path must be .shp, .gpkg, .geojson, or .json.")
        try:
            import geopandas as gpd
        except Exception as exc:
            raise RegistryError("geopandas is required for Inference AOI file filtering.") from exc
        try:
            frame = gpd.read_file(path)
        except Exception as exc:
            raise RegistryError(f"Failed to read Inference AOI file: {path}: {exc}") from exc
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        if frame.empty:
            raise RegistryError(f"Inference AOI file contains no valid geometries: {path}")
        return {"kind": "vector", "path": str(path.resolve()), "frame": frame}

    parts = [part for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise RegistryError(
            "Inference AOI must be empty, a vector path, or bbox coordinates: minx,miny,maxx,maxy."
        )
    try:
        minx, miny, maxx, maxy = (float(part.strip()) for part in parts)
    except ValueError as exc:
        raise RegistryError("Inference AOI bbox must contain numeric minx,miny,maxx,maxy values.") from exc
    if maxx <= minx or maxy <= miny:
        raise RegistryError("Inference AOI bbox must satisfy maxx > minx and maxy > miny.")
    return {"kind": "bbox", "bounds": (minx, miny, maxx, maxy)}


def _inference_raster_intersects_aoi(path: Path, aoi_filter: Mapping[str, Any]) -> bool:
    kind = str(aoi_filter.get("kind") or "none")
    if kind == "none":
        return True
    try:
        import rasterio
        from shapely.geometry import box
        from shapely.ops import unary_union
    except Exception as exc:
        raise RegistryError("rasterio and shapely are required for Inference AOI filtering.") from exc

    try:
        with rasterio.open(path) as src:
            raster_bounds = box(*src.bounds)
            if raster_bounds.is_empty:
                return False
            if kind == "bbox":
                aoi_geometry = box(*aoi_filter["bounds"])
            elif kind == "vector":
                frame = aoi_filter["frame"]
                if src.crs is not None:
                    if frame.crs is None:
                        raise RegistryError(f"Inference AOI CRS is missing while raster has CRS {src.crs}: {aoi_filter.get('path')}")
                    raster_crs = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
                    if str(frame.crs) != raster_crs:
                        frame = frame.to_crs(raster_crs)
                aoi_geometry = unary_union([geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty])
            else:
                raise RegistryError(f"Unsupported Inference AOI filter kind: {kind}")
    except RegistryError:
        raise
    except Exception as exc:
        raise RegistryError(f"Failed to inspect GeoTIFF against Inference AOI: {path}: {exc}") from exc
    if not getattr(aoi_geometry, "is_valid", True):
        aoi_geometry = aoi_geometry.buffer(0)
    return bool(aoi_geometry is not None and not aoi_geometry.is_empty and aoi_geometry.intersects(raster_bounds))


def _rows_from_data_editor_result(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        try:
            return list(value.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _inspect_tiff_candidate_for_ui(file: Path, *, required_band: int, single_file: bool) -> dict[str, Any]:
    try:
        import rasterio

        with rasterio.open(file) as src:
            band_count = int(src.count)
            width = int(src.width)
            height = int(src.height)
    except Exception as exc:
        return {
            "selected": False,
            "name": file.name,
            "parent": str(file.parent),
            "bands": None,
            "width": None,
            "height": None,
            "valid_for_RGB_bands": False,
            "reason": f"inspect failed: {exc}",
            "path": str(file),
        }
    valid = band_count >= required_band
    looks_auxiliary = not is_candidate_primary_tiff(file, single_file=single_file)
    selected = bool(valid and (single_file or not looks_auxiliary))
    reason = "ok" if valid else f"needs band {required_band}, has {band_count}"
    if valid and looks_auxiliary:
        reason = "valid bands but name looks auxiliary; review before selecting"
    return {
        "selected": selected,
        "name": file.name,
        "parent": str(file.parent),
        "bands": band_count,
        "width": width,
        "height": height,
        "valid_for_RGB_bands": valid,
        "reason": reason,
        "path": str(file),
    }


def _write_tsft_raster_list_file(paths: Sequence[str]) -> Path:
    return _write_tiff_raster_list_file(paths, prefix="tsft")


def _write_tiff_raster_list_file(paths: Sequence[str], *, prefix: str) -> Path:
    safe_prefix = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(prefix or "rasters"))
    root = Path(tempfile.gettempdir()) / f"lsd_app_{safe_prefix}_raster_lists"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:12]
    path = root / f"{safe_prefix}_rasters_{digest}.txt"
    path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return path


def _daft_tile_schema(
    upload_method: str,
    tile_size: int,
    stride: int,
    max_nodata_ratio: float = 0.20,
    *,
    slicing_aoi_path: str = "",
) -> str:
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        return _append_slicing_aoi_tile_schema(
            f"tiff_slicing;tile_size={tile_size};stride={stride};max_nodata_ratio={float(max_nodata_ratio):.2f}",
            slicing_aoi_path=slicing_aoi_path,
        )
    return "packaged_images"


def _tsft_tile_schema(
    *,
    upload_method: str,
    tile_size: int,
    stride: int,
    max_nodata_ratio: float,
    min_polygon_overlap_ratio: float,
    min_positive_pixel_ratio: float,
    no_feature_tile_policy: str,
    negative_ratio: float,
    slicing_aoi_path: str = "",
) -> str:
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        schema = (
            f"tiff_slicing;tile_size={tile_size};stride={stride};"
            f"max_nodata_ratio={float(max_nodata_ratio):.2f};"
            f"min_polygon_overlap_ratio={float(min_polygon_overlap_ratio):.3f};"
            f"min_positive_pixel_ratio={float(min_positive_pixel_ratio):.3f};"
            f"no_feature_tile_policy={no_feature_tile_policy};"
            f"negative_ratio={float(negative_ratio):.2f}"
        )
        return _append_slicing_aoi_tile_schema(schema, slicing_aoi_path=slicing_aoi_path)
    return "packaged_images_and_labels"


def _append_slicing_aoi_tile_schema(schema: str, *, slicing_aoi_path: str = "") -> str:
    if not str(slicing_aoi_path or "").strip():
        return schema
    return f"{schema};aoi=strict_inside"


def _render_dataset_repository_header(st: Any) -> None:
    st.markdown(
        """
        <div class="repo-panel">
          <h3>Standardized sample contract</h3>
          <p>Samples are stored as H,W,C float32 values in 0..1. Original channels are preserved;
          RGB_bands is metadata consumed later by model adapters and training DataModules.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_repository_metrics(
    st: Any,
    *,
    summary_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> None:
    classes = ("TAFT", "DAFT", "TSFT", "Evaluation")
    columns = st.columns(4)
    columns[0].metric("registered datasets", len(dataset_rows))
    columns[1].metric("standardized subsets", len(summary_rows))
    columns[2].metric("samples", sum(_int_value(row.get("sample_count")) for row in summary_rows))
    columns[3].metric("with GT", sum(1 for row in dataset_rows if row.get("has_gt")))
    class_columns = st.columns(4)
    for column, dataset_class in zip(class_columns, classes):
        rows = [row for row in summary_rows if row.get("dataset_class") == dataset_class]
        column.metric(
            dataset_class,
            sum(_int_value(row.get("sample_count")) for row in rows),
            delta=f"{len(rows)} subsets",
        )


def _registered_taft_versions(registry_root: str) -> list[str]:
    try:
        rows = list_datasets(registry_root=registry_root)
    except RegistryError:
        return []
    versions = {
        str(row.get("version", "")).strip()
        for row in rows
        if row.get("dataset_class") == "TAFT" and str(row.get("version", "")).strip()
    }
    return sorted(versions)


def _render_taft_tree_discovery_metrics(st: Any, discovery: Mapping[str, Any]) -> None:
    columns = st.columns(5)
    columns[0].metric("leaf subsets", _int_value(discovery.get("subset_count")))
    columns[1].metric("train", _int_value(discovery.get("train_count")))
    columns[2].metric("test", _int_value(discovery.get("test_count")))
    columns[3].metric("samples", _int_value(discovery.get("sample_count")))
    subsets = discovery.get("subsets")
    auxiliary_count = 0
    if isinstance(subsets, list):
        auxiliary_count = sum(1 for row in subsets if isinstance(row, Mapping) and row.get("has_auxiliary"))
    columns[4].metric("with auxiliary", auxiliary_count)


def _taft_tree_discovery_table(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subsets = discovery.get("subsets")
    if not isinstance(subsets, list):
        return rows
    for subset in subsets:
        if not isinstance(subset, Mapping):
            continue
        rows.append(
            {
                "parent_group": subset.get("parent_group", ""),
                "leaf_subset": subset.get("source_rel", ""),
                "subset_id": subset.get("subset_id", ""),
                "train_images": subset.get("train_count", 0),
                "train_labels": subset.get("train_count", 0),
                "test_images": subset.get("test_count", 0),
                "test_labels": subset.get("test_count", 0),
                "RGB_bands": subset.get("RGB_bands", ""),
                "label_polarity": subset.get("label_polarity", ""),
                "has_auxiliary": subset.get("has_auxiliary", False),
                "auxiliary_dirs": ", ".join(str(item) for item in subset.get("auxiliary_dirs", []) or []),
                "warnings": "; ".join(str(item) for item in subset.get("warnings", []) or []),
                "errors": "; ".join(str(item) for item in subset.get("errors", []) or []),
            }
        )
    return rows



def _coerce_record_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        try:
            rows = rows.to_dict("records")
        except TypeError:
            rows = rows.to_dict()
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _request_streamlit_rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _taft_set_rows_confirmed(rows: Any, confirmed: bool) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in _coerce_record_rows(rows):
        item = dict(row)
        item["confirmed"] = bool(confirmed)
        updated.append(item)
    return updated


def _taft_annotation_row_key(row: Mapping[str, Any], idx: int, version: int = 0) -> str:
    subset = str(row.get("subset") or row.get("source_rel") or idx)
    digest = hashlib.sha1(subset.encode("utf-8")).hexdigest()[:12]
    return f"{version}_{digest}"


def _taft_tree_annotation_rows(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subsets = discovery.get("subsets")
    if not isinstance(subsets, list):
        return rows
    for subset in subsets:
        if not isinstance(subset, Mapping):
            continue
        RGB_bands = str(subset.get("RGB_bands", ""))
        label_polarity = str(subset.get("label_polarity", ""))
        rows.append(
            {
                "parent_group": subset.get("parent_group", ""),
                "subset": subset.get("source_rel", subset.get("subset", "")),
                "train_count": subset.get("train_count", 0),
                "test_count": subset.get("test_count", 0),
                "RGB_bands": RGB_bands,
                "label_polarity": label_polarity,
                "confirmed": bool(subset.get("user_confirmed", False)),
            }
        )
    return rows


def _taft_subset_overrides_from_rows(rows: Any) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for row in _coerce_record_rows(rows):
        subset = str(row.get("subset", "")).strip()
        if not subset:
            continue
        overrides[subset] = {
            "RGB_bands": str(row.get("RGB_bands", "")).strip(),
            "label_polarity": str(row.get("label_polarity", "")).strip(),
            "confirmed": bool(row.get("confirmed", False)),
        }
    return overrides


def _taft_annotations_complete(rows: Any) -> bool:
    records = _coerce_record_rows(rows)
    if not records:
        return False
    allowed_polarities = set(LABEL_POLARITIES)
    for row in records:
        if not bool(row.get("confirmed", False)):
            return False
        try:
            normalize_rgb_bands(row.get("RGB_bands", ""))
        except ValueError:
            return False
        if str(row.get("label_polarity", "")) not in allowed_polarities:
            return False
    return True

def _render_dataset_store_browser(st: Any, *, registry_root: str) -> list[dict[str, Any]]:
    browser = browse_dataset_repository(registry_root=registry_root)
    all_rows = _flatten_dataset_store_rows(browser)
    if not all_rows:
        with st.container(border=True):
            st.markdown('<div class="module-title">Filterable dataset browser</div>', unsafe_allow_html=True)
            st.info("No standardized datasets found under the dataset store.")
        return []

    with st.container(border=True):
        st.markdown('<div class="module-title">Filterable dataset browser</div>', unsafe_allow_html=True)
        filtered_rows = _render_dataset_store_filters(st, all_rows)
        st.dataframe(_dataset_store_browse_table(filtered_rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        _render_dataset_concat_panel(st, registry_root=registry_root, rows=all_rows)

    return filtered_rows


def _render_dataset_store_delete_panel(st: Any, *, registry_root: str, rows: list[Mapping[str, Any]]) -> None:
    with st.container(border=True):
        st.markdown('<div class="module-title">Delete standardized dataset version</div>', unsafe_allow_html=True)
        delete_candidates = _version_level_dataset_rows(rows)
        delete_options = {
            _dataset_delete_option_label(row): row
            for row in delete_candidates
            if str(row.get("dataset_class") or "") and str(row.get("version") or "")
        }
        if not delete_options:
            st.info("No dataset version matches the current filters.")
        else:
            with st.form("delete_dataset_store_entry", border=False):
                selected_delete_label = st.selectbox(
                    "dataset version to delete",
                    ["", *delete_options.keys()],
                    format_func=lambda value: "Select a dataset version" if not value else value,
                    key="delete_dataset_store_version_label_v2",
                )
                selected_delete_row = delete_options.get(selected_delete_label, {})
                selected_delete_class = str(selected_delete_row.get("dataset_class") or "")
                selected_delete_version = str(selected_delete_row.get("version") or "")
                selected_delete_path = str(selected_delete_row.get("path") or "")
                confirm_phrase = f"{selected_delete_class}/{selected_delete_version}" if selected_delete_row else ""
                if selected_delete_row:
                    st.caption(f"Will delete: {confirm_phrase}")
                    st.caption(f"Repository path: {selected_delete_path}")
                confirm_delete_text = st.text_input(
                    "type dataset/version to confirm",
                    "",
                    placeholder=confirm_phrase or "TSFT/example-version",
                    help="Manual confirmation prevents deleting a different dataset class after filters or widget state change.",
                    key="delete_dataset_store_confirm_text_v2",
                )
                delete_clicked = st.form_submit_button("Delete Dataset Version")
            if delete_clicked:
                if not selected_delete_row:
                    st.warning("Select a dataset version before deleting.")
                elif str(confirm_delete_text).strip() != confirm_phrase:
                    st.warning(f"Type exactly '{confirm_phrase}' before deleting.")
                else:
                    try:
                        result = delete_dataset_store_entry(
                            registry_root=registry_root,
                            dataset_class=selected_delete_class,
                            version=selected_delete_version,
                            repository_path=selected_delete_path,
                        )
                        st.success(
                            "Deleted {dataset_class}/{version}; removed {removed} registry rows.".format(
                                dataset_class=result["dataset_class"],
                                version=result["version"],
                                removed=result["removed_registry_entries"],
                            )
                        )
                    except RegistryError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()


def _render_dataset_concat_panel(st: Any, *, registry_root: str, rows: list[Mapping[str, Any]]) -> None:
    st.markdown('<div class="module-title">Concat standardized dataset versions</div>', unsafe_allow_html=True)
    concat_rows = [
        row
        for row in _version_level_dataset_rows(rows)
        if row.get("dataset_class") in {"DAFT", "TSFT"}
    ]
    if len(concat_rows) < 2:
        st.info("Register at least two DAFT or TSFT dataset versions before using concat.")
        return
    with st.form("concat_dataset_store_entry", border=False):
        left, right = st.columns(2)
        with left:
            concat_class = st.selectbox(
                "Dataset Class",
                ("TSFT", "DAFT"),
                help="Concat supports DAFT images-only datasets and TSFT supervised datasets.",
                key="concat_dataset_class",
            )
            candidate_versions = [
                str(row.get("version") or "")
                for row in concat_rows
                if row.get("dataset_class") == concat_class and str(row.get("version") or "")
            ]
            source_versions = st.multiselect(
                "source versions",
                candidate_versions,
                help="Choose two or more existing versions from the same dataset class.",
                key="concat_source_versions",
            )
        with right:
            new_version = st.text_input(
                "new concat version",
                "",
                help="Name for the virtual concat dataset version.",
                key="concat_new_version",
            )
            notes = st.text_area(
                "concat notes",
                "",
                help="Optional registry notes for the virtual concat dataset.",
                key="concat_notes",
            )
        confirm_concat = st.checkbox(
            "confirm virtual concat creation",
            value=False,
            help="Creates a virtual manifest only; source dataset files are not copied or modified.",
            key="concat_confirm",
        )
        create_concat = st.form_submit_button("Create Concat Dataset")
    if not create_concat:
        return
    if not confirm_concat:
        st.warning("Check the confirmation box before creating a concat dataset.")
        return
    if len(source_versions) < 2:
        st.error("Choose at least two source versions.")
        return
    if not str(new_version or "").strip():
        st.error("Enter a new concat version name.")
        return
    concat_version = str(new_version).strip()
    if _dataset_version_exists(registry_root, concat_class, concat_version):
        st.warning(f"{concat_class} dataset version '{concat_version}' already exists. Choose a new concat version name.")
        return
    last_concat_key = f"{concat_class}/{concat_version}"
    if st.session_state.get("last_concat_created_key") == last_concat_key:
        st.info(f"Concat dataset {last_concat_key} was already created from this form.")
        return
    try:
        result = register_concat_dataset_entry(
            registry_root=registry_root,
            dataset_class=concat_class,
            version=concat_version,
            source_versions=list(source_versions),
            notes=str(notes or ""),
        )
    except RegistryError as exc:
        st.error(str(exc))
        return
    st.success(
        "Created virtual concat dataset {dataset_class}/{version} with {count} sources.".format(
            dataset_class=getattr(result, "dataset_class", concat_class),
            version=getattr(result, "version", concat_version),
            count=len(source_versions),
        )
    )
    st.session_state["last_concat_created_key"] = last_concat_key
    st.rerun()


def _version_level_dataset_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    versions: list[dict[str, Any]] = []
    for row in rows:
        dataset_class = str(row.get("dataset_class") or "")
        version = str(row.get("version") or "")
        key = (dataset_class, version)
        if not dataset_class or not version or key in seen:
            continue
        seen.add(key)
        versions.append(dict(row))
    return versions


def _dataset_delete_option_label(row: Mapping[str, Any]) -> str:
    dataset_class = str(row.get("dataset_class") or "")
    version = str(row.get("version") or "")
    samples = _int_value(row.get("version_sample_count", row.get("sample_count", 0)))
    path = str(row.get("path") or "")
    return f"{dataset_class}/{version} | {samples} samples | {path}"


def _flatten_dataset_store_rows(browser: Mapping[str, Any]) -> list[dict[str, Any]]:
    classes = browser.get("classes", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(classes, Mapping):
        return rows
    for dataset_class in DATASET_CLASSES:
        class_summary = classes.get(dataset_class, {})
        versions = class_summary.get("versions", []) if isinstance(class_summary, Mapping) else []
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, Mapping):
                continue
            base = {
                "dataset_class": dataset_class,
                "version": version.get("version", ""),
                "version_sample_count": version.get("sample_count", 0),
                "version_train_count": version.get("train_count", 0),
                "version_test_count": version.get("test_count", 0),
                "version_subset_count": version.get("subset_count", 0),
                "has_gt": version.get("has_gt", False),
                "sample_dtype": version.get("sample_dtype", ""),
                "sample_shape_layout": version.get("sample_shape_layout", ""),
                "import_status": version.get("import_status", ""),
                "import_percent": version.get("import_percent", ""),
                "is_concat": version.get("is_concat", False),
                "concat_source_count": version.get("concat_source_count", 0),
                "path": version.get("path", ""),
                "manifest_path": version.get("manifest_path", ""),
                "delete_key": f"{dataset_class} / {version.get('version', '')}",
            }
            subsets = version.get("subsets", [])
            if base["is_concat"]:
                rows.append({**base, **_concat_dataset_store_subset_fields(version)})
            elif isinstance(subsets, list) and subsets:
                for subset in subsets:
                    if isinstance(subset, Mapping):
                        rows.append({**base, **_dataset_store_subset_fields(subset)})
            else:
                rows.append({**base, **_empty_dataset_store_subset_fields()})
    return rows


def _render_dataset_store_filters(st: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = list(rows)
    filter_columns = [
        ("dataset_class", "Dataset Class"),
        ("version", "version"),
        ("parent_group", "Parent Group"),
        ("subset_id", "Subset ID"),
        ("RGB_bands", "RGB bands"),
        ("label_polarity", "label polarity"),
        ("has_auxiliary", "Has Auxiliary"),
    ]
    controls = st.columns(4)
    for index, (field, label) in enumerate(filter_columns):
        options = ["All", *_unique_values(filtered, field)]
        selected = controls[index % len(controls)].selectbox(label, options, key=f"dataset_store_filter_{field}")
        if selected != "All":
            filtered = [row for row in filtered if str(row.get(field, "")) == selected]
    return filtered


def _dataset_store_browse_table(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "dataset_class",
        "version",
        "parent_group",
        "subset",
        "subset_id",
        "sample_count",
        "train_count",
        "test_count",
        "RGB_bands",
        "label_polarity",
        "upload_method",
        "tile_schema",
        "has_auxiliary",
        "version_sample_count",
        "version_subset_count",
        "is_concat",
        "concat_source_count",
        "positive_count",
        "negative_count",
        "mean_positive_pixel_ratio",
        "import_status",
        "path",
        "subset_manifest_path",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _dataset_store_subset_fields(subset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_group": subset.get("parent_group", ""),
        "subset": subset.get("subset", ""),
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


def _concat_dataset_store_subset_fields(version: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_group": "virtual_concat",
        "subset": f"{_int_value(version.get('concat_source_count'))} sources",
        "subset_id": "",
        "sample_count": version.get("sample_count", 0),
        "train_count": version.get("train_count", 0),
        "test_count": version.get("test_count", 0),
        "RGB_bands": "",
        "label_polarity": "",
        "has_auxiliary": "",
        "upload_method": "virtual_concat",
        "tile_schema": "virtual_concat",
        "positive_count": "",
        "negative_count": "",
        "mean_positive_pixel_ratio": "",
        "subset_manifest_path": version.get("manifest_path", ""),
    }


def _empty_dataset_store_subset_fields() -> dict[str, Any]:
    return {
        "parent_group": "",
        "subset": "",
        "subset_id": "",
        "sample_count": "",
        "train_count": "",
        "test_count": "",
        "RGB_bands": "",
        "label_polarity": "",
        "has_auxiliary": "",
        "upload_method": "",
        "tile_schema": "",
        "positive_count": "",
        "negative_count": "",
        "mean_positive_pixel_ratio": "",
        "subset_manifest_path": "",
    }


def _unique_values(rows: list[Mapping[str, Any]], field: str) -> list[str]:
    values = {str(row.get(field, "")) for row in rows if str(row.get(field, ""))}
    return sorted(values)


def _dataset_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "dataset_class",
        "version",
        "parent_group",
        "subset",
        "subset_id",
        "sample_count",
        "train_count",
        "test_count",
        "has_label",
        "RGB_bands",
        "label_polarity",
        "has_auxiliary",
        "sample_layout",
        "sample_dtype",
        "manifest_path",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _dataset_registry_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "dataset_class",
        "version",
        "sample_count",
        "train_count",
        "test_count",
        "subset_count",
        "has_gt",
        "RGB_bands",
        "sample_shape_layout",
        "sample_dtype",
        "repository_path",
        "manifest_path",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _render_artifacts(st: Any, artifacts: dict[str, Any]) -> None:
    st.subheader("Artifacts")
    paths = artifacts.get("paths")
    if paths is None:
        paths = {
            key: value
            for key, value in artifacts.items()
            if key.endswith("_path") and isinstance(value, str) and value
        }
    if paths:
        st.dataframe([{"artifact": key, "path": value} for key, value in paths.items()], use_container_width=True)
    else:
        st.info("No output files found yet.")

    evaluation_payload = artifacts.get("evaluation") or artifacts.get("metrics")
    if evaluation_payload is not None:
        st.subheader("evaluation.json")
        st.dataframe(_flatten_json_for_table(evaluation_payload), use_container_width=True)

    if artifacts.get("vector_stats") is not None:
        st.subheader("vector_stats.json")
        st.json(artifacts["vector_stats"])

    overlay_path = artifacts.get("overlay_png")
    if overlay_path is not None:
        st.subheader("overlay.png")
        st.image(overlay_path)


def _default_RGB_bands_from_weight(weight: Mapping[str, Any], fallback: Any) -> str:
    """Infer the UI RGB_bands default from registered weight metadata."""

    value = str(weight.get("RGB_bands") or "").strip()
    if value:
        return _safe_rgb_bands(value, fallback)
    return _safe_rgb_bands(fallback)


def _model_options_from_weight(weight: Mapping[str, Any]) -> dict[str, Any]:
    """Build model.options from registered weight metadata."""

    options: dict[str, Any] = {}
    mapping = {
        "weight_type": "weight_type",
        "base_checkpoint_path": "sam_checkpoint_path",
        "image_size": "image_size",
        "lora_base_r": "lora_base_r",
        "lora_base_alpha": "lora_base_alpha",
        "lora_task_r": "lora_task_r",
        "lora_task_alpha": "lora_task_alpha",
        "lora_dropout": "lora_dropout",
        "peft_adapter_name": "peft_adapter_name",
    }
    for source_key, option_key in mapping.items():
        value = weight.get(source_key)
        if value not in {None, ""}:
            options[option_key] = value
    return options


def _truth_config(truth_path: str) -> dict[str, str | None]:
    path_text = truth_path.strip()
    if not path_text:
        return {"vector_path": None, "mask_path": None}
    suffix = Path(path_text).suffix.lower()
    if suffix in {".tif", ".tiff", ".png"}:
        return {"vector_path": None, "mask_path": path_text}
    return {"vector_path": path_text, "mask_path": None}


def _blank_to_none(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        converted = result.to_dict()
        if isinstance(converted, dict):
            return converted
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": str(result)}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON artifact: {exc}"}


def _flatten_json_for_table(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                rows.append({"metric": f"{key}.{nested_key}", "value": str(nested_value)})
        else:
            rows.append({"metric": key, "value": str(value)})
    return rows


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, *, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}"


def _format_scientific(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.2e}"


def _format_gpu_memory(value: Any) -> str:
    try:
        mb = float(value)
    except (TypeError, ValueError):
        return "-"
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def _format_best_score(value: Any, epoch: int, selection_metric: str) -> str:
    if selection_metric == "final_epoch" and epoch:
        return f"final epoch {epoch}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    suffix = f" @ {epoch}" if epoch else ""
    return f"{number:.4f}{suffix}"


def _daft_loss_history_rows(job: Mapping[str, Any]) -> list[dict[str, float | int]]:
    rows = _daft_loss_rows_from_history(job.get("loss_history"))
    if rows:
        return rows

    rows = _daft_loss_rows_from_metrics_path(job.get("metrics_path"))
    if rows:
        return rows

    for progress_row in _job_progress_log_rows(job):
        rows = _daft_loss_rows_from_history(progress_row.get("loss_history"))
        if rows:
            return rows
    progress_rows = []
    for progress_row in _job_progress_log_rows(job):
        epoch = _int_value(progress_row.get("current_epoch"))
        losses = progress_row.get("losses") if isinstance(progress_row.get("losses"), Mapping) else {}
        if epoch > 0 and losses:
            progress_rows.append(_daft_loss_row(epoch, losses))
    progress_rows = [row for row in progress_rows if len(row) > 1]
    if progress_rows:
        return _dedupe_epoch_rows(progress_rows)

    losses = job.get("losses") if isinstance(job.get("losses"), Mapping) else {}
    epoch = _int_value(job.get("current_epoch"))
    if not losses or epoch <= 0:
        return []
    row = _daft_loss_row(epoch, losses)
    return [row] if len(row) > 1 else []


def _daft_loss_rows_from_history(history: Any) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not isinstance(history, list):
        return rows
    for item in history:
        if not isinstance(item, Mapping):
            continue
        epoch = _int_value(item.get("epoch"))
        if epoch <= 0:
            continue
        row = _daft_loss_row(epoch, item)
        if len(row) > 1:
            rows.append(row)
    return _dedupe_epoch_rows(rows)


def _daft_loss_rows_from_metrics_path(path_value: Any) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for item in _read_jsonl_rows(path_value):
        epoch = _int_value(item.get("epoch"))
        losses = item.get("losses") if isinstance(item.get("losses"), Mapping) else {}
        if epoch <= 0 or not losses:
            continue
        row = _daft_loss_row(epoch, losses)
        if len(row) > 1:
            rows.append(row)
    return _dedupe_epoch_rows(rows)


def _daft_loss_row(epoch: int, values: Mapping[str, Any]) -> dict[str, float | int]:
    row: dict[str, float | int] = {"epoch": int(epoch)}
    for key in ("total", "seg", "adv", "align", "dis"):
        try:
            row[key] = float(values.get(key))
        except (TypeError, ValueError):
            pass
    return row


def _tsft_metric_history_rows(job: Mapping[str, Any]) -> list[dict[str, float | int]]:
    rows = _tsft_metric_rows_from_history(job.get("metrics_history"))
    if rows:
        return rows

    rows = _tsft_metric_rows_from_metrics_path(job.get("metrics_path"))
    if rows:
        return rows

    for progress_row in _job_progress_log_rows(job):
        rows = _tsft_metric_rows_from_history(progress_row.get("metrics_history"))
        if rows:
            return rows
    progress_rows = []
    for progress_row in _job_progress_log_rows(job):
        epoch = _int_value(progress_row.get("current_epoch"))
        if epoch <= 0:
            continue
        row = _tsft_metric_row(epoch, progress_row)
        if len(row) > 1:
            progress_rows.append(row)
    if progress_rows:
        return _dedupe_epoch_rows(progress_rows)

    epoch = _int_value(job.get("current_epoch"))
    if epoch <= 0:
        return []
    row = _tsft_metric_row(epoch, job)
    return [row] if len(row) > 1 else []


def _tsft_metric_rows_from_history(history: Any) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not isinstance(history, list):
        return rows
    for item in history:
        if not isinstance(item, Mapping):
            continue
        epoch = _int_value(item.get("epoch"))
        if epoch <= 0:
            continue
        row = _tsft_metric_row(epoch, item)
        if len(row) > 1:
            rows.append(row)
    return _dedupe_epoch_rows(rows)


def _tsft_metric_rows_from_metrics_path(path_value: Any) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for item in _read_jsonl_rows(path_value):
        epoch = _int_value(item.get("epoch"))
        if epoch <= 0:
            continue
        row = _tsft_metric_row(epoch, item)
        if len(row) > 1:
            rows.append(row)
    return _dedupe_epoch_rows(rows)


def _tsft_metric_row(epoch: int, item: Mapping[str, Any]) -> dict[str, float | int]:
    row: dict[str, float | int] = {"epoch": int(epoch)}
    train = item.get("train") if isinstance(item.get("train"), Mapping) else {}
    test = item.get("eval") if isinstance(item.get("eval"), Mapping) else item.get("test") if isinstance(item.get("test"), Mapping) else {}
    _add_tsft_chart_metrics(row, "train", train)
    _add_tsft_chart_metrics(row, "eval", test)
    for split in ("train", "eval", "test"):
        prefix = "eval" if split == "test" else split
        for metric in ("loss", "mIoU", "IoU_foreground"):
            key = f"{split}.{metric}"
            if key not in item:
                continue
            chart_key = "IoU" if metric == "IoU_foreground" else metric
            try:
                row[f"{prefix}_{chart_key}"] = float(item.get(key))
            except (TypeError, ValueError):
                pass
    return row


def _dedupe_epoch_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float | int]]:
    by_epoch: dict[int, dict[str, float | int]] = {}
    for raw in rows:
        epoch = _int_value(raw.get("epoch"))
        if epoch <= 0:
            continue
        row: dict[str, float | int] = {"epoch": epoch}
        for key, value in raw.items():
            if key == "epoch":
                continue
            try:
                row[str(key)] = float(value)
            except (TypeError, ValueError):
                pass
        if len(row) > 1:
            by_epoch[epoch] = row
    return [by_epoch[key] for key in sorted(by_epoch)]



def _read_jsonl_rows(path_value: Any) -> list[dict[str, Any]]:
    raw = str(path_value or "").strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _job_progress_log_rows(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = str(job.get("progress_log_path") or "").strip()
    if not path:
        state_path = str(job.get("state_path") or "").strip()
        if state_path:
            path = str(Path(state_path).with_suffix(".progress.jsonl"))
    return _read_jsonl_rows(path)


def _job_elapsed_seconds(job: Mapping[str, Any]) -> float | None:
    explicit = _float_or_none(job.get("duration_seconds"))
    if explicit is not None and explicit > 0:
        return explicit
    start_raw = str(job.get("started_at") or job.get("created_at") or "").strip()
    if not start_raw:
        return None
    finish_raw = str(job.get("finished_at") or "").strip()
    try:
        started = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finish_raw.replace("Z", "+00:00")) if finish_raw else datetime.now(timezone.utc)
    except ValueError:
        return None
    return max(0.0, (finished - started).total_seconds())


def _format_duration_seconds(value: Any) -> str:
    seconds = _float_or_none(value)
    if seconds is None:
        return "-"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _render_inference_speed_summary(st: Any, state: Mapping[str, Any]) -> None:
    metrics = _inference_speed_metrics_from_state(state)
    if not metrics:
        return
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    st.caption(
        " | ".join(f"{row['metric']} {row['value']}" for row in rows)
    )


def _inference_speed_metrics_from_state(state: Mapping[str, Any]) -> dict[str, str]:
    candidates: list[Mapping[str, Any]] = [state]
    result = state.get("result")
    if isinstance(result, Mapping):
        statuses = result.get("statuses")
        if isinstance(statuses, list):
            for item in statuses:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("stage") or "") == "infer" and isinstance(item.get("result"), Mapping):
                    candidates.insert(0, item["result"])
        artifacts = result.get("artifacts")
        if isinstance(artifacts, Mapping):
            manifest = artifacts.get("infer_manifest") if isinstance(artifacts.get("infer_manifest"), Mapping) else None
            if manifest:
                candidates.append(manifest)
    for item in candidates:
        duration = _float_or_none(item.get("inference_duration_seconds", item.get("duration_seconds")))
        speed = _float_or_none(item.get("inference_patches_per_second", item.get("patches_per_second")))
        pred_speed = _float_or_none(item.get("inference_predictions_per_second", item.get("predictions_per_second")))
        patch_count = _int_value(item.get("inference_patch_count", item.get("patch_count")))
        pred_count = _int_value(item.get("inference_prediction_count", item.get("prediction_count")))
        if duration is None and speed is None and pred_speed is None:
            continue
        metrics: dict[str, str] = {}
        metrics["inference time"] = _format_duration_seconds(duration)
        if speed is not None:
            metrics["patch speed"] = f"{speed:.2f}/s"
        if pred_speed is not None:
            metrics["prediction speed"] = f"{pred_speed:.2f}/s"
        if patch_count:
            metrics["patches"] = str(patch_count)
        if pred_count:
            metrics["predictions"] = str(pred_count)
        return metrics
    return {}

def _add_tsft_chart_metrics(row: dict[str, float | int], split: str, metrics: Mapping[str, Any]) -> None:
    for metric in ("loss", "mIoU"):
        try:
            row[f"{split}_{metric}"] = float(metrics.get(metric))
        except (TypeError, ValueError):
            pass
    try:
        row[f"{split}_IoU"] = _tsft_foreground_iou_for_chart(metrics)
    except (TypeError, ValueError):
        pass


def _tsft_foreground_iou_for_chart(metrics: Mapping[str, Any]) -> float:
    for key in ("IoU_foreground", "IoU_landslide"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    iou = metrics.get("IoU")
    if isinstance(iou, Sequence) and not isinstance(iou, (str, bytes)):
        if len(iou) > 1:
            return float(iou[1])
        if len(iou) == 1:
            return float(iou[0])
        raise TypeError("IoU sequence is empty")
    return float(iou)


def _optional_positive_int(value: Any) -> int | None:
    number = _int_value(value)
    return number if number > 0 else None


if __name__ == "__main__":
    main()
