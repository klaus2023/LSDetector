"""Streamlit local operations console for LSD_APP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import csv
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsd_app.config import ConfigError, PlatformConfig, normalize_config as _normalize_platform_config
from lsd_app.core.channels import DEFAULT_RGB_BANDS, RGB_BAND_PRESETS, normalize_rgb_bands
from lsd_app.core.paths import default_workspace_root, registry_dir
from lsd_app.core.inference import InferenceError, run_patch_inference
from lsd_app.core.metrics import EvaluationError, run_evaluate
from lsd_app.core.mosaic import MosaicError, run_mosaic
from lsd_app.core.raster import RasterError
from lsd_app.core.tiling import prepare_raster
from lsd_app.core.vectorize import VectorizeError, run_vectorize
from lsd_app.core.visualization import VisualizationError, run_visualize
from lsd_app.models import ModelAdapterError
from lsd_app.models.registry import get_adapter, normalize_model_name
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
    TSFT_NO_FEATURE_POLICY_BALANCED,
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
    DaftLSDSAMConfig,
    build_daft_training_payload,
    parse_halve_epochs,
    run_lsdsam_daft,
)
from lsd_app.training.tsft import (
    TSFT_TRAINING_DEVICE_POLICY,
    TsftLSDSAMConfig,
    TsftTrainingError,
    build_tsft_training_payload,
    finalize_tsft_checkpoint,
    run_lsdsam_tsft,
)
from lsd_app.web.jobs import (
    _json_ready,
    list_job_states,
    read_job_state,
    request_job_cancel,
    stable_job_id,
    start_background_job,
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

StageRunner = Callable[[PlatformConfig], Any]
STAGE_RUNNERS: dict[str, StageRunner] = {
    "prepare": prepare_raster,
    "infer": run_patch_inference,
    "mosaic": run_mosaic,
    "vectorize": run_vectorize,
    "evaluate": run_evaluate,
    "visualize": run_visualize,
}


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
) -> str:
    default_value = _safe_rgb_bands(default)
    index = RGB_BANDS_UI_OPTIONS.index(default_value) if default_value in RGB_BANDS_UI_OPTIONS else len(RGB_BANDS_UI_OPTIONS) - 1
    selected = st.selectbox(label, RGB_BANDS_UI_OPTIONS, index=index, key=f"{key}_preset")
    if selected == RGB_BANDS_CUSTOM_OPTION:
        raw_value = st.text_input(f"{label} custom", value=default_value, key=key)
    else:
        raw_value = selected
    return normalize_rgb_bands(raw_value)


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
        adv_lambda=float(payload.get("adv_lambda") or 0.05),
        align_lambda=float(payload.get("align_lambda") or 0.005),
        lora_task_r=int(payload.get("lora_task_r") or 16),
        lora_task_alpha=int(payload.get("lora_task_alpha") or 32),
        lora_dropout=float(payload.get("lora_dropout") if payload.get("lora_dropout") is not None else 0.05),
        output_weight_id=str(payload.get("output_weight_id") or ""),
        resume=bool(payload.get("resume")),
        resume_checkpoint_path=str(payload.get("resume_checkpoint_path") or ""),
        save_optimizer_state=bool(payload.get("save_optimizer_state")),
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
        description=f"DAFT LSDSAM {payload.get('target_dataset_id') or 'target'}",
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
    return str(job.get("status") or "") in {"pending", "running", "cancelling", "stalled", "stalled_uninterruptible"}


def _latest_job_for_kind(registry_root: str, kind: str) -> Mapping[str, Any] | None:
    jobs = list_job_states(registry_root, kind=kind)
    if not jobs:
        return None
    return sorted(jobs, key=lambda item: str(item.get("updated_at", "")), reverse=True)[0]


def _active_jobs_for_kind(registry_root: str, kind: str) -> list[Mapping[str, Any]]:
    return [job for job in list_job_states(registry_root, kind=kind) if _job_is_active(job)]


def _active_job_for_kind(registry_root: str, kind: str) -> Mapping[str, Any] | None:
    jobs = _active_jobs_for_kind(registry_root, kind)
    if not jobs:
        return None
    return sorted(jobs, key=lambda item: str(item.get("updated_at", "")), reverse=True)[0]


def _recover_active_job_id(st: Any, *, registry_root: str, kind: str, session_key: str) -> str | None:
    active = _active_job_for_kind(registry_root, kind)
    if active:
        job_id = str(active.get("job_id") or "")
        if job_id:
            st.session_state[session_key] = job_id
            return job_id
    remembered = str(st.session_state.get(session_key) or "").strip()
    if remembered:
        state = read_job_state(registry_root, remembered)
        if state:
            return remembered
    latest = _latest_job_for_kind(registry_root, kind)
    if latest and str(latest.get("status") or "") in {"interrupted", "cancelled", "failed", "completed"}:
        job_id = str(latest.get("job_id") or "")
        if job_id:
            st.session_state[session_key] = job_id
            return job_id
    return None


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
    age = _job_updated_age_seconds(job)
    if age is None:
        return ""
    if age < 120:
        return f"last update {age:.0f}s ago"
    return f"last update {age / 60:.1f} min ago"


def _render_job_health_notice(st: Any, job: Mapping[str, Any]) -> None:
    status = str(job.get("status") or "")
    reason = str(job.get("stall_reason") or job.get("message") or "").strip()
    if status == "stalled":
        st.warning(reason or "Task progress has stalled. Stop it, then resume from the latest checkpoint/cache if available.")
    elif status == "stalled_uninterruptible":
        st.error(reason or "Worker is in uninterruptible kernel sleep; restart WSL/CUDA before launching another CUDA job.")
    elif status == "resource_blocked":
        st.error(reason or "CUDA device is blocked by another active job.")


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
        description=f"TSFT LSDSAM {payload.get('dataset_id') or 'dataset'}",
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
            status="running",
            percent=0.0,
            overall_percent=stage_base,
            message=f"Starting {stage}.",
        )

        def stage_progress(progress: Mapping[str, Any], *, stage_base: float = stage_base, stage_weight: float = stage_weight) -> None:
            stage_percent = min(max(float(progress.get("percent") or 0.0), 0.0), 1.0)
            _emit_deployment_progress(
                progress_callback,
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
        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            status="completed",
            percent=1.0,
            overall_percent=stage_base + stage_weight,
            message=f"Completed {stage}.",
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
            status="running",
            percent=0.0,
            overall_percent=stage_base,
            message=f"Starting {stage}.",
        )

        def stage_progress(progress: Mapping[str, Any], *, stage_base: float = stage_base, stage_weight: float = stage_weight) -> None:
            stage_percent = min(max(float(progress.get("percent") or 0.0), 0.0), 1.0)
            _emit_deployment_progress(
                progress_callback,
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
        _emit_deployment_progress(
            progress_callback,
            stage=stage,
            status="completed",
            percent=1.0,
            overall_percent=stage_base + stage_weight,
            message=f"Completed {stage}.",
        )
    return statuses


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

    start_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind="deployment",
        description=f"Deployment {config.output_dir.name or config.output_dir}",
        payload={
            "output_dir": str(config.output_dir),
            "config_fingerprint": deployment_config_fingerprint(raw_config),
            "input_raster_path": str(config.input_raster_path),
            "model": config.model_name,
            "raw_config": _deployment_fingerprint_payload(raw_config),
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
    model_id = str(payload.get("model_id") or "LSDSAM-B").strip() or "LSDSAM-B"
    checkpoint_path = str(payload.get("checkpoint_path") or "").strip()
    output_dir = Path(str(payload.get("output_dir") or "outputs/evaluation_run")).expanduser().resolve()
    batch_size = max(1, int(payload.get("batch_size") or 1))
    requested_device = str(payload.get("device") or "auto")
    if not dataset_id:
        raise EvaluationError("Choose an Evaluation dataset before running evaluation.")
    if not checkpoint_path:
        raise EvaluationError("Choose a registered weight before running evaluation.")

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
            "percent": 0.02,
            "overall_percent": 0.02,
            "device": device,
            "device_name": device_name,
            "sample_count": len(samples),
            "current_action": "loading model",
            "message": f"Loading {model_id} for Evaluation dataset {dataset_id}.",
        }
    )
    adapter_config = _adapter_config_for_evaluation(
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        device=device,
        RGB_bands=dataset.RGB_bands,
    )
    adapter = get_adapter(model_id)
    adapter.load(adapter_config)
    model = getattr(adapter, "model", None)
    if model is None:
        raise EvaluationError(f"Model adapter {model_id} does not expose a loaded torch model for dataset evaluation.")
    image_size = int(getattr(adapter, "image_size", 512) or 512)
    criterion_seg = _FocalDiceLoss(torch)
    eval_dataset = _LSDSAMSupervisedDataset(samples, image_size=image_size, cv2=cv2, torch=torch, np=np)
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
        "weight_id": str(payload.get("weight_id") or "").strip(),
        "weight_version": str(payload.get("weight_version") or payload.get("weight_id") or "").strip(),
        "checkpoint_path": checkpoint_path,
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "device": device,
        "device_name": device_name,
    }
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
) -> _EvaluationAdapterConfig:
    return _EvaluationAdapterConfig(
        model_name=model_id,
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        device=device,
        threshold=0.5,
        RGB_bands=RGB_bands,
        adapter_options={},
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


def main() -> None:
    """Run the Streamlit app."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError("Streamlit is required to run the Web GUI. Install the 'web' extra first.") from exc

    st.set_page_config(page_title="LSD_APP Console", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
          background: #f7f9fc;
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.0); }
        div[role="radiogroup"] {
          gap: 12px;
          border-bottom: 1px solid rgba(31, 41, 55, 0.14);
          padding-bottom: 8px;
          margin-bottom: 18px;
        }
        div[role="radiogroup"] label {
          padding: 4px 0;
          white-space: nowrap;
        }
        .repo-panel {
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.86);
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
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255,255,255,0.88);
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
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255,255,255,0.90);
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
          background: linear-gradient(90deg, #2563eb, #14b8a6);
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
        .background-job-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr));
          gap: 12px;
          margin: 10px 0 16px 0;
        }
        .job-card {
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255,255,255,0.90);
          padding: 12px 14px;
        }
        .job-card strong {
          color: #111827;
          display: block;
          font-size: 0.94rem;
          line-height: 1.2;
          margin-bottom: 8px;
        }
        .job-card span {
          display: inline-block;
          color: #4b5563;
          font-size: 0.82rem;
          line-height: 1.4;
          margin-right: 12px;
        }
        .daft-train-panel {
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255,255,255,0.92);
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
          background: rgba(255, 255, 255, 0.86);
          border: 1px solid rgba(31, 41, 55, 0.12);
          border-radius: 8px;
          padding: 10px 12px;
        }
        div[data-testid="stMetricLabel"] {
          color: #4b5563;
          font-size: 0.78rem;
        }
        .deployment-panel {
          border: 1px solid rgba(31, 41, 55, 0.14);
          border-radius: 8px;
          background: rgba(255,255,255,0.90);
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("LSD_APP Local Workbench")
    st.caption("Model, weight, data, inference, evaluation, and training repositories for landslide mapping.")

    registry_root = st.sidebar.text_input("asset workspace", str(default_workspace_root()))
    st.sidebar.caption(f"registered asset registry: {registry_dir(registry_root)}")
    section_control = getattr(st, "segmented_control", None)
    if section_control is not None:
        selected_section = section_control(
            "workspace section",
            SECTION_OPTIONS,
            default=SECTION_OPTIONS[0],
            label_visibility="collapsed",
            key="workspace_section",
            width="stretch",
        )
        selected_section = selected_section or SECTION_OPTIONS[0]
    else:
        selected_section = st.radio(
            "workspace section",
            SECTION_OPTIONS,
            horizontal=True,
            label_visibility="collapsed",
            key="workspace_section",
        )

    if selected_section == "Model Repository":
        st.subheader("Model Repository")
        model_rows = list_models(project_root=Path.cwd(), registry_root=registry_root)
        st.dataframe(model_rows, use_container_width=True)

    elif selected_section == "Weight Repository":
        st.subheader("Weight Repository")
        _render_weight_add_form(st, registry_root=registry_root, model_options=MODEL_CHOICES)
        weight_rows = list_weights(registry_root=registry_root)
        st.caption("Weight registry entries")
        filtered_weight_rows = _render_weight_repository_filters(st, weight_rows)
        st.dataframe(_weight_repository_table(filtered_weight_rows), use_container_width=True, hide_index=True)
        _render_weight_delete_panel(st, registry_root=registry_root, rows=filtered_weight_rows)

    elif selected_section == "Dataset Repository":
        st.subheader("Dataset Repository")
        _render_dataset_repository_header(st)
        summary_rows = list_dataset_repository_summary(registry_root=registry_root)
        dataset_rows = list_datasets(registry_root=registry_root)
        _render_dataset_repository_metrics(st, summary_rows=summary_rows, dataset_rows=dataset_rows)

        _render_dataset_upload_console(st, registry_root=registry_root)
        _render_background_jobs_panel(st, registry_root=registry_root, kind="dataset-upload")
        summary_rows = list_dataset_repository_summary(registry_root=registry_root)
        dataset_rows = list_datasets(registry_root=registry_root)
        st.caption("Dataset registry entries")
        st.dataframe(_dataset_registry_table(dataset_rows), use_container_width=True, hide_index=True)
        _render_dataset_store_browser(st, registry_root=registry_root)

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
        weight_options = [row["weight_id"] for row in inference_weights]
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
            selected_weight_id = st.selectbox("registered weight", weight_options, index=default_weight_index)
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
        with primary_right:
            output_dir = st.text_input("output dir", defaults.get("output_dir", "outputs/deployment_run"))
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
                batch_size = st.number_input("batch size", min_value=1, value=int(defaults.get("batch_size", 1)), step=1)
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

        action_cols = st.columns((1.0, 1.0, 4.0))
        run_everything = action_cols[0].button("Generate shp", type="primary", use_container_width=True)
        run_visual_preview = action_cols[1].button("Preview", use_container_width=True)
        confirm_cache_clear = False

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
        current_fingerprint = deployment_config_fingerprint(raw_config)
        current_job_id = deployment_job_id(raw_config)
        cache_status = output_cache_status(output_dir, current_fingerprint)
        active_deployment_jobs = [
            job
            for job in list_job_states(registry_root, kind="deployment")
            if job.get("status") in {"pending", "running", "cancelling"}
        ]
        same_output_active = [
            job
            for job in active_deployment_jobs
            if str((job.get("payload") or {}).get("output_dir") or "") == str(Path(output_dir).expanduser().resolve())
        ]
        if cache_status["has_cache"] and not cache_status["matches"]:
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
            config, preflight = preflight_deployment_config(raw_config, selected_weight=selected_weight, create_output_dirs=True)
            _render_preflight(st, preflight)
            if config is not None and preflight.ok:
                if same_output_active and all(job.get("job_id") != current_job_id for job in same_output_active):
                    st.error("Another deployment job is already running in this output folder. Stop the current job before starting a new configuration.")
                elif cache_status["has_cache"] and not cache_status["matches"] and not confirm_cache_clear:
                    st.error("The configuration has changed. Confirm cache cleanup before starting a new job, or choose a different output dir.")
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
            st.info("Choose a registered weight and input GeoTIFF, then click Generate shp. Evaluation remains separate.")

        recovered_deployment_job_id = _recover_active_job_id(
            st,
            registry_root=registry_root,
            kind="deployment",
            session_key="active_deployment_job_id",
        )
        selected_job_id = recovered_deployment_job_id or st.session_state.get("active_deployment_job_id") or current_job_id
        if selected_job_id:
            _render_deployment_job_monitor_auto(st, registry_root=registry_root, job_id=str(selected_job_id))

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
        weight_labels = [row["weight_id"] for row in weights]
        with st.form("evaluation_config"):
            selected_dataset_id = st.selectbox("Evaluation dataset", dataset_labels) if dataset_labels else ""
            selected_weight_id = st.selectbox("weight", weight_labels) if weight_labels else ""
            eval_model = st.selectbox("evaluation model", MODEL_CHOICES, index=0)
            eval_output_dir = st.text_input("evaluation output dir", "outputs/evaluation_run")
            eval_batch_size = st.number_input("evaluation batch size", min_value=1, value=1, step=1)
            run_eval = st.form_submit_button("Run Evaluation", disabled=active_eval_running)
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
                    f"current evaluation model selection is {eval_model}."
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
        _render_background_jobs_panel(st, registry_root=registry_root, kind="evaluation", skip_latest_job_id=str(active_eval_job_id or "") or None)
        _render_evaluation_history_panel_auto(st, registry_root=registry_root)


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
    st.caption(f"{_format_progress_percent(percent)} overall | status {status} | job {job_id}")
    stage_rows = _deployment_stage_rows_from_job(state)
    if stage_rows:
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


def _render_pipeline_resume_controls(st: Any, *, registry_root: str, state: Mapping[str, Any], kind: str) -> None:
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


def _render_deployment_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _deployment_fragment() -> None:
            _render_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _deployment_fragment()
    else:
        _render_deployment_job_monitor(st, registry_root=registry_root, job_id=job_id)


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
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _evaluation_fragment() -> None:
            _render_evaluation_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _evaluation_fragment()
    else:
        _render_evaluation_job_monitor(st, registry_root=registry_root, job_id=job_id)


def _render_evaluation_history_panel_auto(st: Any, *, registry_root: str) -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _evaluation_history_fragment() -> None:
            _render_evaluation_history_panel(st, registry_root=registry_root)

        _evaluation_history_fragment()
    else:
        _render_evaluation_history_panel(st, registry_root=registry_root)


def _render_daft_training_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _daft_training_fragment() -> None:
            _render_daft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _daft_training_fragment()
    else:
        _render_daft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)


def _render_tsft_training_job_monitor_auto(st: Any, *, registry_root: str, job_id: str) -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _tsft_training_fragment() -> None:
            _render_tsft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)

        _tsft_training_fragment()
    else:
        _render_tsft_training_job_monitor(st, registry_root=registry_root, job_id=job_id)


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
            "stage": row.get("stage", ""),
            "registered_model": row.get("model_id", ""),
            "weight_size": _format_bytes(row.get("file_size_bytes")),
            "parameter_count": _format_count(row.get("parameter_count")),
            "registered_at": row.get("created_at", ""),
            "stored_path": row.get("path", ""),
        }
        for row in rows
    ]


def _render_weight_add_form(
    st: Any,
    *,
    registry_root: str,
    model_options: Sequence[str],
    locked_model_id: str | None = None,
    form_key: str = "weight_registry",
) -> None:
    selected_model_id = locked_model_id or ""
    with st.form(form_key):
        if locked_model_id:
            st.text_input("model id", locked_model_id, disabled=True, key=f"{form_key}_model_id_locked")
            weight_model_id = locked_model_id
        else:
            options = list(model_options)
            default_index = options.index("LSDSAM-B") if "LSDSAM-B" in options else 0
            weight_model_id = st.selectbox("model id", options, index=default_index, key=f"{form_key}_model_id")
        selected_model_id = str(weight_model_id)
        weight_stage = st.selectbox("weight stage", WEIGHT_STAGES, key=f"{form_key}_stage")
        weight_id = st.text_input("weight id", "", key=f"{form_key}_weight_id")
        weight_path = st.text_input("weight path", "", key=f"{form_key}_path")
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
        except RegistryError as exc:
            progress_bar.empty()
            progress_status.empty()
            st.error(str(exc))


def _render_weight_repository_filters(st: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = list(rows)
    query = st.text_input("search weights", "", key="weight_repo_search")
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
        ("model_id", "model id"),
        ("stage", "stage"),
        ("exists", "file status"),
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
    st.caption(title)
    if not rows:
        st.info("No weight matches the current filters.")
        return

    options = [str(row.get("weight_id", "")) for row in rows if row.get("weight_id")]
    if not options:
        st.info("No deletable weight id is available.")
        return

    with st.form(form_key):
        selected_weight_id = st.selectbox("weight id to delete", options, key=f"{form_key}_weight_id")
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
    with st.form(f"{stage.lower()}_dry_run"):
        model_id = st.selectbox("model id", model_ids) if model_ids else ""
        weight_id = st.selectbox("base weight", [row["weight_id"] for row in weights]) if weights else ""
        dataset_id = st.selectbox("dataset", [row["dataset_id"] for row in datasets]) if datasets else ""
        output_dir = st.text_input("training output dir", f"outputs/{stage.lower()}_dry_run")
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
    st.caption(f"device policy: {DAFT_TRAINING_DEVICE_POLICY}")
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
    active_daft_running = bool(active_daft_state and _job_is_active(active_daft_state))
    if active_daft_running:
        st.info("A DAFT training job is already running. The monitor below will keep tracking it even if you close and reopen the page.")

    selector_left, selector_middle, _selector_right = st.columns(3)
    with selector_left:
        model_id = st.selectbox("model", MODEL_CHOICES, index=0, key="daft_model", disabled=active_daft_running)
    model_weight_rows = _training_weight_rows_for_model(weight_rows, model_id)
    base_weight_options = _weight_id_options(model_weight_rows)
    model_key = _training_model_widget_key(model_id)
    with selector_middle:
        if base_weight_options:
            base_weight_id = st.selectbox(
                "base weight",
                base_weight_options,
                key=f"daft_base_weight_{model_key}",
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
            alignment_mode = st.selectbox("constraint", DAFT_ALIGNMENT_MODES, index=0)
            evaluation_dataset_id = st.selectbox(
                "evaluation dataset",
                ["None", *[row["dataset_id"] for row in evaluation_rows]],
                key="daft_evaluation_dataset",
            )
            evaluation_dataset_id = "" if evaluation_dataset_id == "None" else evaluation_dataset_id
            output_dir = st.text_input("output dir", "outputs/daft_training")
            output_weight_id = st.text_input(
                "DAFT weight id",
                "",
                help="Optional. Leave blank to let LSD_APP generate one automatically.",
            )
        with right:
            total_epochs = st.number_input("epochs", min_value=1, value=10, step=1)
            initial_lr = st.number_input("initial lr", min_value=1e-8, value=2e-4, step=1e-5, format="%.8f")
            halve_epochs = st.text_input("halve epochs", "6,8")
        with st.expander("Advanced", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                batch_size = st.number_input("batch size", min_value=1, value=2, step=1)
                num_workers = st.number_input("num workers", min_value=0, value=DEFAULT_DAFT_NUM_WORKERS, step=1)
                adv_lambda = st.number_input("adv lambda", min_value=0.0, value=0.05, step=0.005, format="%.6f")
                align_lambda = st.number_input("alignment lambda", min_value=0.0, value=0.005, step=0.001, format="%.6f")
                save_optimizer_state = st.checkbox(
                    "save optimizer state for exact resume",
                    value=False,
                    help="Off by default to keep checkpoints small. Enable only when you need exact optimizer-state resume.",
                )
            with adv_right:
                lora_task_r = st.number_input("LoRA task r", min_value=1, value=16, step=1)
                lora_task_alpha = st.number_input("LoRA task alpha", min_value=1, value=32, step=1)
                lora_dropout = st.number_input("LoRA dropout", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        launch = st.form_submit_button("Launch DAFT", disabled=active_daft_running)

    selected_target = next((row for row in target_rows if row.get("dataset_id") == target_dataset_id), None)
    if selected_target and not evaluation_dataset_id:
        st.info("No Evaluation dataset selected. DAFT will use the last/latest checkpoint for registration; source-domain train metrics are not used for best-checkpoint selection.")

    if launch:
        running_job = _active_job_for_kind(registry_root, "daft-training")
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
                adv_lambda=float(adv_lambda),
                align_lambda=float(align_lambda),
                lora_task_r=int(lora_task_r),
                lora_task_alpha=int(lora_task_alpha),
                lora_dropout=float(lora_dropout),
                output_weight_id=output_weight_id,
                save_optimizer_state=bool(save_optimizer_state),
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
    _render_background_jobs_panel(st, registry_root=registry_root, kind="daft-training", skip_latest_job_id=active_job_id)


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
    if not resume_checkpoint or not Path(resume_checkpoint).exists():
        st.info("No DAFT LoRA checkpoint is available yet, so this job cannot be resumed.")
        return
    st.info(f"Resume checkpoint: {resume_checkpoint}")
    _latest, _best, config_path = _training_resume_paths_from_state(
        state,
        default_output_dir="outputs/daft_training",
        latest_names=("lsdsam_daft_latest_lora.pth", "lsdformer_daft_latest_adapter.pth"),
        best_names=("lsdsam_daft_best_lora.pth", "lsdformer_daft_best_adapter.pth"),
    )
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
    title = str(job.get("description") or job.get("job_id") or "DAFT training")
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
    st.caption(f"device policy: {TSFT_TRAINING_DEVICE_POLICY}")
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
    active_tsft_running = bool(active_tsft_state and _job_is_active(active_tsft_state))
    if active_tsft_running:
        st.info("A TSFT training job is already running. The monitor below will keep tracking it even if you close and reopen the page.")

    selector_left, selector_middle, _selector_right = st.columns(3)
    with selector_left:
        model_id = st.selectbox("model", MODEL_CHOICES, index=0, key="tsft_model", disabled=active_tsft_running)
    model_weight_rows = _training_weight_rows_for_model(weight_rows, model_id)
    base_weight_options = _weight_id_options(model_weight_rows)
    model_key = _training_model_widget_key(model_id)
    with selector_middle:
        if base_weight_options:
            base_weight_id = st.selectbox(
                "base weight",
                base_weight_options,
                key=f"tsft_base_weight_{model_key}",
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
                "evaluation dataset",
                ["None", *[row["dataset_id"] for row in evaluation_rows]],
                key="tsft_evaluation_dataset",
            )
            evaluation_dataset_id = "" if evaluation_dataset_id == "None" else evaluation_dataset_id
            output_dir = st.text_input("output dir", "outputs/tsft_training", key="tsft_output_dir")
            output_weight_id = st.text_input(
                "TSFT weight id",
                "",
                help="Optional. Leave blank to let LSD_APP generate one automatically.",
                key="tsft_output_weight_id",
            )
        with right:
            max_epoch = st.number_input("max epoch", min_value=1, value=10, step=1, key="tsft_max_epoch")
            max_lr = st.number_input("max lr", min_value=1e-8, value=2e-4, step=1e-5, format="%.8f", key="tsft_max_lr")
            min_lr = st.number_input("min lr", min_value=0.0, value=1e-6, step=1e-6, format="%.8f", key="tsft_min_lr")
            max_decay_epoch = st.number_input("max decay epoch", min_value=1, value=10, step=1, key="tsft_max_decay_epoch")
        with st.expander("Advanced", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                batch_size = st.number_input("batch size", min_value=1, value=2, step=1, key="tsft_batch_size")
                num_workers = st.number_input("num workers", min_value=0, value=DEFAULT_DAFT_NUM_WORKERS, step=1, key="tsft_num_workers")
                save_optimizer_state = st.checkbox(
                    "save optimizer state for exact resume",
                    value=False,
                    help="Off by default to keep checkpoints small. Enable only when you need exact optimizer-state resume.",
                    key="tsft_save_optimizer_state",
                )
            with adv_right:
                lora_task_r = st.number_input("LoRA task r", min_value=1, value=16, step=1, key="tsft_lora_r")
                lora_task_alpha = st.number_input("LoRA task alpha", min_value=1, value=32, step=1, key="tsft_lora_alpha")
                lora_dropout = st.number_input("LoRA dropout", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="tsft_lora_dropout")
                resume = st.checkbox("resume from checkpoint", value=False, key="tsft_resume")
                resume_checkpoint_path = st.text_input(
                    "resume checkpoint path",
                    "",
                    help="Optional. Leave blank to use the latest adapter checkpoint under output_dir/checkpoints.",
                    key="tsft_resume_checkpoint_path",
                )
        launch = st.form_submit_button("Launch TSFT", disabled=active_tsft_running)

    if launch:
        running_job = _active_job_for_kind(registry_root, "tsft-training")
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
    _render_background_jobs_panel(st, registry_root=registry_root, kind="tsft-training", skip_latest_job_id=active_job_id)


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
    title = str(job.get("description") or job.get("job_id") or "TSFT training")
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


def _render_dataset_upload_console(st: Any, *, registry_root: str) -> None:
    st.markdown(
        """
        <div class="dataset-upload-shell">
          <h4>Dataset Upload</h4>
          <p>Choose one dataset class first. The form below only exposes the fields needed by that class.</p>
        </div>
        """,
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


def _render_background_jobs_panel(
    st: Any,
    *,
    registry_root: str,
    kind: str | None = None,
    skip_latest_job_id: str | None = None,
) -> None:
    st.markdown("#### Background Jobs")

    def render_jobs() -> None:
        jobs = list_job_states(registry_root, kind=kind)
        if not jobs:
            st.info("No background dataset jobs have been started in this registry.")
            return
        active = [job for job in jobs if job.get("status") in {"pending", "running", "cancelling"}]
        completed = [job for job in jobs if job.get("status") == "completed"]
        failed = [job for job in jobs if job.get("status") == "failed"]
        cols = st.columns(4)
        cols[0].metric("jobs", len(jobs))
        cols[1].metric("running", len(active))
        cols[2].metric("completed", len(completed))
        cols[3].metric("failed", len(failed))
        latest_job = sorted(jobs, key=lambda item: str(item.get("updated_at", "")), reverse=True)[0]
        latest_job_id = str(latest_job.get("job_id") or "")
        if skip_latest_job_id and latest_job_id == skip_latest_job_id:
            return
        if kind == "daft-training":
            _render_daft_training_progress(st, latest_job)
        elif kind == "tsft-training":
            if latest_job_id:
                _render_tsft_training_job_monitor(st, registry_root=registry_root, job_id=latest_job_id)
            else:
                _render_tsft_training_progress(st, latest_job)
        else:
            _render_job_progress(st, latest_job, compact=True)

    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every=1.5)
        def _jobs_fragment() -> None:
            render_jobs()

        _jobs_fragment()
    else:
        render_jobs()


def _render_job_progress(st: Any, job: Mapping[str, Any], *, compact: bool = False) -> None:
    percent = min(max(float(job.get("percent") or 0.0), 0.0), 1.0)
    status = str(job.get("status") or "unknown")
    title = str(job.get("description") or job.get("job_id") or "background job")
    message = str(job.get("message") or "")
    processed = _int_value(job.get("processed"))
    total = _int_value(job.get("total"))
    written = _int_value(job.get("written"))
    skipped = _int_value(job.get("skipped"))
    rewritten = _int_value(job.get("rewritten"))
    st.markdown(
        f"""
        <div class="job-card">
          <strong>{title}</strong>
          <span>status {status}</span>
          <span>processed {processed}/{total or "discovering"}</span>
          <span>written {written}</span>
          <span>skipped {skipped}</span>
          <span>rewritten {rewritten}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(percent)
    if message:
        if compact:
            st.caption(f"Latest: {message}")
        else:
            st.caption(message)
    if status == "failed":
        st.error(f"{job.get('error_type', 'Error')}: {job.get('error', '')}")


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
            standardize_label = "standardize new subsets float32 copy" if append_to_existing else "standardize float32 copy"
            taft_tree_standardize = st.checkbox(standardize_label, value=False, key="taft_tree_standardize")
            taft_tree_resume = st.checkbox(
                "resume matching partial import",
                value=True,
                key="taft_tree_resume",
                help="Continue matching partial files where possible; append mode keeps existing subsets and adds only new leaf subsets.",
            )
            taft_tree_force_rebuild = st.checkbox(
                "force rebuild version",
                value=False,
                key="taft_tree_force_rebuild",
                disabled=append_to_existing,
                help="Delete the existing standardized version folder and rebuild it from scratch.",
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
                "force_rebuild": False if append_to_existing else taft_tree_force_rebuild,
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
            confirmed = row_cols[6].checkbox(
                "confirmed",
                value=bool(row.get("confirmed", False)),
                key=f"taft_confirmed_{row_key}",
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
            "standardize": taft_tree_standardize,
            "resume": taft_tree_resume,
            "force_rebuild": False if append_to_existing else taft_tree_force_rebuild,
            "notes": taft_tree_notes,
        }
    append_import = bool(import_config.get("append_to_existing", False))
    import_disabled = bool(taft_tree_discovery.get("has_errors")) or not _taft_annotations_complete(edited_rows)
    import_label = "Add TAFT Subsets" if append_import else "Import TAFT Tree"
    if st.button(import_label, disabled=import_disabled, type="primary"):
        try:
            _run_taft_import_with_progress(
                st,
                registry_root=registry_root,
                version=str(import_config.get("version", taft_tree_version)),
                root=str(import_config.get("root", taft_tree_root)),
                standardize=bool(import_config.get("standardize", taft_tree_standardize)),
                resume=bool(import_config.get("resume", taft_tree_resume)),
                force_rebuild=bool(import_config.get("force_rebuild", False if append_import else taft_tree_force_rebuild)),
                append_to_existing=append_import,
                notes=str(import_config.get("notes", taft_tree_notes)),
                subset_overrides=_taft_subset_overrides_from_rows(edited_rows),
            )
        except RegistryError as exc:
            st.error(str(exc))

def _run_taft_import_with_progress(
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
    progress_panel = st.empty()
    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    initial_progress = {
        "status": "pending",
        "processed": 0,
        "total": 0,
        "written": 0,
        "skipped": 0,
        "rewritten": 0,
        "percent": 0.0,
        "message": "Starting TAFT tree import.",
    }
    progress_panel.markdown(_taft_import_progress_html(initial_progress), unsafe_allow_html=True)
    progress_text.caption("pending: processed 0/discovering | written 0 | skipped 0 | rewritten 0")

    def update_progress(progress: Mapping[str, Any]) -> None:
        percent = float(progress.get("percent") or 0.0)
        clamped = min(max(percent, 0.0), 1.0)
        progress_panel.markdown(_taft_import_progress_html(progress), unsafe_allow_html=True)
        progress_bar.progress(clamped)
        progress_text.caption(
            "{status}: processed {processed}/{total} | written {written} | skipped {skipped} | rewritten {rewritten}".format(
                status=progress.get("status", "running"),
                processed=progress.get("processed", 0),
                total=progress.get("total", 0) or "discovering",
                written=progress.get("written", 0),
                skipped=progress.get("skipped", 0),
                rewritten=progress.get("rewritten", 0),
            )
        )

    entry = register_taft_tree(
        registry_root=registry_root,
        version=version,
        root=root,
        standardize=standardize,
        resume=resume,
        force_rebuild=force_rebuild,
        append_to_existing=append_to_existing,
        notes=notes,
        subset_overrides=subset_overrides,
        progress_callback=update_progress,
    )
    st.success(f"TAFT version registered: {entry.version}")


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
    with st.form("daft_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("DAFT version", "DAFT-target-v1")
            imagery_path = st.text_input(
                "image directory or GeoTIFF path",
                "",
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
            overwrite = st.checkbox("overwrite standardized target", value=False)
            notes = st.text_area("DAFT notes", "")
        add_dataset = st.form_submit_button("Upload DAFT Dataset")
    if add_dataset:
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
) -> None:
    payload = {
        "dataset_class": "DAFT",
        "version": version,
        "imagery_path": imagery_path,
        "RGB_bands": RGB_bands,
        "tile_schema": _daft_tile_schema(upload_method, tile_size or 0, stride or 0, max_nodata_ratio or 0.20),
        "notes": notes,
        "overwrite": overwrite,
        "upload_method": upload_method,
        "resume": resume,
        "tile_size": tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "stride": stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
        "max_nodata_ratio": max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
    }
    job_kind = "dataset-upload-daft-resume" if resume else "dataset-upload-daft"
    job_id = stable_job_id(kind=job_kind, version=version, payload=payload)

    def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
        return register_dataset(
            registry_root=registry_root,
            name=version,
            dataset_class="DAFT",
            imagery_path=imagery_path,
            label_path=None,
            has_gt=False,
            RGB_bands=RGB_bands,
            tile_schema=_daft_tile_schema(upload_method, tile_size or 0, stride or 0, max_nodata_ratio or 0.20),
            notes=notes,
            version=version,
            standardize=True,
            overwrite=overwrite,
            resume=resume,
            upload_method=upload_method,
            tile_size=tile_size if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            stride=stride if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            daft_max_nodata_ratio=max_nodata_ratio if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
            progress_callback=progress_callback,
        )

    try:
        job = start_background_job(
            registry_root=registry_root,
            job_id=job_id,
            kind="dataset-upload",
            description=f"DAFT upload {version}",
            payload=payload,
            target=target,
        )
        st.success(f"DAFT upload started: {job.job_id}")
        st.rerun()
    except RegistryError as exc:
        st.error(str(exc))


def _render_evaluation_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### Evaluation Dataset")
    with st.form("evaluation_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("Evaluation version", "evaluation-v1", key="evaluation_upload_version")
            samples = st.text_input("evaluation samples", "", key="evaluation_upload_samples")
            RGB_bands = _rgb_bands_input(st, "RGB bands", key="evaluation_RGB_bands")
            label_polarity = st.selectbox("label foreground", LABEL_POLARITIES, key="evaluation_label_polarity")
        with right:
            overwrite = st.checkbox("overwrite standardized target", value=False, key="evaluation_overwrite")
            notes = st.text_area("Evaluation notes", "", key="evaluation_notes")
        add_dataset = st.form_submit_button("Upload Evaluation Dataset")
    if add_dataset:
        try:
            imagery_path, label_path = _tsft_sample_folder_paths(samples, split="evaluation", required=True)
        except RegistryError as exc:
            st.error(str(exc))
            return
        payload = {
            "dataset_class": "Evaluation",
            "version": version,
            "samples": samples,
            "imagery_path": imagery_path,
            "label_path": label_path,
            "RGB_bands": RGB_bands,
            "label_polarity": label_polarity,
            "notes": notes,
            "overwrite": overwrite,
            "upload_method": DAFT_UPLOAD_METHOD_PACKAGED,
        }
        job_id = stable_job_id(kind="dataset-upload-evaluation", version=version, payload=payload)

        def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
            progress_callback({"status": "running", "message": f"Standardizing Evaluation {version}.", "percent": 0.05})
            entry = register_dataset(
                registry_root=registry_root,
                name=version,
                dataset_class="Evaluation",
                imagery_path=imagery_path,
                label_path=label_path,
                has_gt=True,
                RGB_bands=RGB_bands,
                label_polarity=label_polarity,
                tile_schema="evaluation_images_and_labels",
                notes=notes,
                version=version,
                standardize=True,
                overwrite=overwrite,
                upload_method=DAFT_UPLOAD_METHOD_PACKAGED,
                progress_callback=progress_callback,
            )
            progress_callback(
                {
                    "status": "running",
                    "message": f"Registered Evaluation {entry.version}.",
                    "processed": entry.sample_count,
                    "total": entry.sample_count,
                    "percent": 1.0,
                }
            )
            return entry

        try:
            job = start_background_job(
                registry_root=registry_root,
                job_id=job_id,
                kind="dataset-upload",
                description=f"Evaluation upload {version}",
                payload=payload,
                target=target,
            )
            st.success(f"Evaluation upload started: {job.job_id}")
        except RegistryError as exc:
            st.error(str(exc))


def _render_tsft_upload_panel(st: Any, *, registry_root: str) -> None:
    st.markdown("#### TSFT Few-shot Dataset")
    method_label = st.radio(
        "TSFT upload method",
        ("Packaged Images + Labels", "Tiff Slicing"),
        horizontal=True,
        key="tsft_upload_method_label",
    )
    upload_method = DAFT_UPLOAD_METHOD_TIFF_SLICING if method_label == "Tiff Slicing" else DAFT_UPLOAD_METHOD_PACKAGED
    with st.form("tsft_dataset_upload"):
        left, right = st.columns(2)
        with left:
            version = st.text_input("TSFT version", "TSFT-round-1")
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                train_samples = ""
                train_images = st.text_input("source GeoTIFF path", "")
                train_labels = st.text_input("landslide polygons", "")
            else:
                train_samples = st.text_input("train samples", "")
                train_images = ""
                train_labels = ""
            RGB_bands = _rgb_bands_input(st, "RGB bands", key="tsft_RGB_bands")
            label_polarity = st.selectbox("label foreground", LABEL_POLARITIES, key="tsft_label_polarity")
        with right:
            if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
                test_images = ""
                test_labels = ""
                tile_size = st.number_input("window size", min_value=1, value=1024, step=1)
                stride = st.number_input("stride", min_value=1, value=512, step=1)
                max_nodata_ratio = st.number_input(
                    "max nodata ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.05,
                    format="%.2f",
                )
                min_polygon_overlap_ratio = st.number_input(
                    "min polygon overlap ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POLYGON_OVERLAP_RATIO),
                    step=0.05,
                    format="%.3f",
                )
                min_positive_pixel_ratio = st.number_input(
                    "min positive pixel ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_TSFT_MIN_POSITIVE_PIXEL_RATIO),
                    step=0.001,
                    format="%.3f",
                )
                no_feature_policy = st.selectbox(
                    "no-feature tile policy",
                    TSFT_NO_FEATURE_POLICIES,
                    index=list(TSFT_NO_FEATURE_POLICIES).index(TSFT_NO_FEATURE_POLICY_BALANCED),
                    key="tsft_no_feature_policy",
                )
                negative_ratio = st.number_input(
                    "negative ratio",
                    min_value=0.0,
                    value=float(DEFAULT_TSFT_NEGATIVE_RATIO),
                    step=0.25,
                    format="%.2f",
                )
                resume = st.checkbox("resume matching partial target", value=False)
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
            overwrite = st.checkbox("overwrite standardized target", value=False)
            notes = st.text_area("TSFT notes", "")
        add_dataset = st.form_submit_button("Upload TSFT Dataset")
    if add_dataset:
        try:
            if upload_method == DAFT_UPLOAD_METHOD_PACKAGED:
                train_images, train_labels = _tsft_sample_folder_paths(train_samples, split="train", required=True)
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
            "overwrite": overwrite,
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
        }
        job_id = stable_job_id(kind="dataset-upload-tsft", version=version, payload=payload)

        def target(progress_callback: Callable[[Mapping[str, Any]], None]) -> Any:
            progress_callback({"status": "running", "message": f"Standardizing TSFT {version}.", "percent": 0.05})
            entry = register_dataset(
                registry_root=registry_root,
                name=version,
                dataset_class="TSFT",
                imagery_path=train_images,
                label_path=train_labels,
                has_gt=True,
                RGB_bands=RGB_bands,
                label_polarity=label_polarity,
                tile_schema=_tsft_tile_schema(
                    upload_method=upload_method,
                    tile_size=int(tile_size),
                    stride=int(stride),
                    max_nodata_ratio=float(max_nodata_ratio),
                    min_polygon_overlap_ratio=float(min_polygon_overlap_ratio),
                    min_positive_pixel_ratio=float(min_positive_pixel_ratio),
                    no_feature_tile_policy=str(no_feature_policy),
                    negative_ratio=float(negative_ratio),
                    ),
                notes=notes,
                version=version,
                standardize=True,
                overwrite=overwrite,
                resume=resume,
                upload_method=upload_method,
                tile_size=int(tile_size) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                stride=int(stride) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                daft_max_nodata_ratio=float(max_nodata_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                tsft_min_polygon_overlap_ratio=(
                    float(min_polygon_overlap_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
                ),
                tsft_min_positive_pixel_ratio=(
                    float(min_positive_pixel_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None
                ),
                tsft_no_feature_tile_policy=str(no_feature_policy),
                tsft_negative_ratio=float(negative_ratio) if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING else None,
                progress_callback=progress_callback,
            )
            progress_callback(
                {
                    "status": "running",
                    "message": f"Registered TSFT {entry.version}.",
                    "processed": entry.sample_count,
                    "total": entry.sample_count,
                    "percent": 1.0,
                }
            )
            return entry

        try:
            job = start_background_job(
                registry_root=registry_root,
                job_id=job_id,
                kind="dataset-upload",
                description=f"TSFT upload {version}",
                payload=payload,
                target=target,
            )
            st.success(f"TSFT upload started: {job.job_id}")
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


def _daft_tile_schema(upload_method: str, tile_size: int, stride: int, max_nodata_ratio: float = 0.20) -> str:
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        return f"tiff_slicing;tile_size={tile_size};stride={stride};max_nodata_ratio={float(max_nodata_ratio):.2f}"
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
) -> str:
    if upload_method == DAFT_UPLOAD_METHOD_TIFF_SLICING:
        return (
            f"tiff_slicing;tile_size={tile_size};stride={stride};"
            f"max_nodata_ratio={float(max_nodata_ratio):.2f};"
            f"min_polygon_overlap_ratio={float(min_polygon_overlap_ratio):.3f};"
            f"min_positive_pixel_ratio={float(min_positive_pixel_ratio):.3f};"
            f"no_feature_tile_policy={no_feature_tile_policy};"
            f"negative_ratio={float(negative_ratio):.2f}"
        )
    return "packaged_images_and_labels"


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

def _render_dataset_store_browser(st: Any, *, registry_root: str) -> None:
    browser = browse_dataset_repository(registry_root=registry_root)
    classes = browser.get("classes", {}) if isinstance(browser, Mapping) else {}
    st.markdown("#### Dataset Store Browser")
    card_html = ""
    for dataset_class in DATASET_CLASSES:
        summary = classes.get(dataset_class, {}) if isinstance(classes, Mapping) else {}
        card_html += (
            """
            <div class="repo-browser-card">
              <strong>{dataset_class}</strong>
              <span>{versions} versions</span>
              <span>{samples} samples</span>
              <span>{train} train</span>
              <span>{test} test</span>
            </div>
            """.format(
                dataset_class=dataset_class,
                versions=_int_value(summary.get("version_count")),
                samples=_int_value(summary.get("sample_count")),
                train=_int_value(summary.get("train_count")),
                test=_int_value(summary.get("test_count")),
            )
        )
    st.markdown(f"<div class=\"repo-browser-grid\">{card_html}</div>", unsafe_allow_html=True)

    all_rows = _flatten_dataset_store_rows(browser)
    if not all_rows:
        st.info("No standardized datasets found under the dataset store.")
        return

    st.caption("Filterable dataset browser")
    filtered_rows = _render_dataset_store_filters(st, all_rows)
    st.dataframe(_dataset_store_browse_table(filtered_rows), use_container_width=True, hide_index=True)

    st.caption("Delete standardized dataset version")
    delete_candidates = _version_level_dataset_rows(filtered_rows)
    delete_options = {
        _dataset_delete_option_label(row): row
        for row in delete_candidates
        if str(row.get("dataset_class") or "") and str(row.get("version") or "")
    }
    if not delete_options:
        st.info("No dataset version matches the current filters.")
    else:
        with st.form("delete_dataset_store_entry"):
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
        ("dataset_class", "dataset class"),
        ("version", "version"),
        ("parent_group", "parent group"),
        ("subset_id", "subset id"),
        ("RGB_bands", "RGB bands"),
        ("label_polarity", "label polarity"),
        ("has_auxiliary", "has auxiliary"),
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
    history = job.get("loss_history")
    rows: list[dict[str, float | int]] = []
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, Mapping):
                continue
            epoch = _int_value(item.get("epoch"))
            if epoch <= 0:
                continue
            row: dict[str, float | int] = {"epoch": epoch}
            for key in ("total", "seg", "adv", "align", "dis"):
                try:
                    row[key] = float(item.get(key))
                except (TypeError, ValueError):
                    pass
            if len(row) > 1:
                rows.append(row)
    if rows:
        return sorted(rows, key=lambda item: int(item["epoch"]))

    losses = job.get("losses") if isinstance(job.get("losses"), Mapping) else {}
    epoch = _int_value(job.get("current_epoch"))
    if not losses or epoch <= 0:
        return []
    row = {"epoch": epoch}
    for key in ("total", "seg", "adv", "align", "dis"):
        try:
            row[key] = float(losses.get(key))
        except (TypeError, ValueError):
            pass
    return [row] if len(row) > 1 else []



def _tsft_metric_history_rows(job: Mapping[str, Any]) -> list[dict[str, float | int]]:
    history = job.get("metrics_history")
    rows: list[dict[str, float | int]] = []
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, Mapping):
                continue
            epoch = _int_value(item.get("epoch"))
            if epoch <= 0:
                continue
            row: dict[str, float | int] = {"epoch": epoch}
            train = item.get("train") if isinstance(item.get("train"), Mapping) else {}
            test = item.get("eval") if isinstance(item.get("eval"), Mapping) else item.get("test") if isinstance(item.get("test"), Mapping) else {}
            _add_tsft_chart_metrics(row, "train", train)
            _add_tsft_chart_metrics(row, "eval", test)
            if len(row) > 1:
                rows.append(row)
    if rows:
        return sorted(rows, key=lambda item: int(item["epoch"]))

    epoch = _int_value(job.get("current_epoch"))
    train = job.get("train") if isinstance(job.get("train"), Mapping) else {}
    test = job.get("eval") if isinstance(job.get("eval"), Mapping) else job.get("test") if isinstance(job.get("test"), Mapping) else {}
    if epoch <= 0 or not (train or test):
        return []
    row = {"epoch": epoch}
    _add_tsft_chart_metrics(row, "train", train)
    _add_tsft_chart_metrics(row, "eval", test)
    return [row] if len(row) > 1 else []


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
