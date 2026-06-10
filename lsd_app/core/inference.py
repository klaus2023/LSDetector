"""Patch-level inference runner for model adapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from lsd_app.config import PlatformConfig
from lsd_app.core.raster import require_rasterio
from lsd_app.models import ModelAdapterError, get_adapter


class InferenceError(RuntimeError):
    """Raised when patch inference cannot be completed."""


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class InferenceResult:
    """Summary returned by patch inference."""

    model_name: str
    adapter_name: str
    checkpoint_path: Path | None
    patch_dir: Path
    prediction_dir: Path
    manifest_path: Path
    patch_count: int
    prediction_count: int
    skipped_count: int
    output_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]
    started_at: str
    finished_at: str
    duration_seconds: float
    patches_per_second: float
    predictions_per_second: float
    overwrite: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary."""

        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "patch_dir": str(self.patch_dir),
            "prediction_dir": str(self.prediction_dir),
            "manifest_path": str(self.manifest_path),
            "patch_count": self.patch_count,
            "prediction_count": self.prediction_count,
            "skipped_count": self.skipped_count,
            "output_paths": [str(path) for path in self.output_paths],
            "skipped_paths": [str(path) for path in self.skipped_paths],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "patches_per_second": self.patches_per_second,
            "predictions_per_second": self.predictions_per_second,
            "overwrite": self.overwrite,
        }


def run_patch_inference(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> InferenceResult:
    """Run adapter inference over prepared patch GeoTIFFs."""

    patch_paths = _discover_patch_paths(config)
    if not patch_paths:
        raise InferenceError(
            f"No patch GeoTIFFs found in {config.outputs.patches}. "
            "Run `lsd prepare --config ...` first or provide an output directory with existing patches."
        )

    started_at = _utc_now()
    started_perf = time.perf_counter()
    prediction_dir = config.outputs.patch_predictions
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "infer_manifest.json"

    tasks = [_PatchTask(patch_path=patch_path, output_path=prediction_dir / patch_path.name) for patch_path in patch_paths]
    skipped_tasks = [task for task in tasks if not config.overwrite and task.output_path.exists()]
    pending_tasks = [task for task in tasks if task not in skipped_tasks]
    output_paths = tuple(task.output_path.resolve() for task in tasks)
    skipped_paths = tuple(task.output_path.resolve() for task in skipped_tasks)

    adapter = get_adapter(config.model_name)
    prediction_count = 0
    processed_count = len(skipped_tasks)
    _emit_progress(
        progress_callback,
        stage="infer",
        status="running",
        percent=0.0,
        processed=processed_count,
        total=len(patch_paths),
        written=0,
        skipped=len(skipped_tasks),
        rewritten=0,
        current_action="loading model adapter",
        current_reason=adapter.name,
        message=f"Loading model adapter {adapter.name}.",
    )

    if pending_tasks:
        try:
            adapter.load(config)
        except ModelAdapterError:
            raise
        except NotImplementedError:
            raise
        except Exception as exc:
            raise InferenceError(f"Failed to load model adapter '{adapter.name}': {exc}") from exc

        pending_arrays: list[Any] = []
        pending_profiles: list[dict[str, Any]] = []
        pending_masks: list[Any] = []
        pending_output_paths: list[Path] = []

        for patch_data in _iter_patch_data(pending_tasks, config.num_workers):
            _emit_progress(
                progress_callback,
                stage="infer",
                status="running",
                percent=processed_count / len(patch_paths) if patch_paths else 1.0,
                processed=processed_count,
                total=len(patch_paths),
                written=prediction_count,
                skipped=len(skipped_tasks),
                rewritten=0,
                current_action="preprocessing patch",
                current_reason=patch_data.patch_path.name,
                message=f"Preprocessing patch {patch_data.patch_path.name}.",
            )
            model_ready_array = _prepare_model_input_array(patch_data.array, config=config)
            try:
                pending_arrays.append(adapter.preprocess_patch(model_ready_array))
            except Exception as exc:
                raise InferenceError(
                    f"Adapter '{adapter.name}' failed to preprocess patch {patch_data.patch_path}: {exc}"
                ) from exc
            pending_profiles.append(patch_data.profile)
            pending_masks.append(patch_data.mask)
            pending_output_paths.append(patch_data.output_path)

            if len(pending_arrays) >= config.batch_size:
                flushed = _flush_predictions(
                    adapter,
                    pending_arrays,
                    pending_profiles,
                    pending_masks,
                    pending_output_paths,
                )
                prediction_count += flushed
                processed_count += flushed
                _emit_progress(
                    progress_callback,
                    stage="infer",
                    status="running",
                    percent=processed_count / len(patch_paths) if patch_paths else 1.0,
                    processed=processed_count,
                    total=len(patch_paths),
                    written=prediction_count,
                    skipped=len(skipped_tasks),
                    rewritten=0,
                    current_action="writing prediction batch",
                    current_reason=f"{flushed} patches",
                    message=f"Wrote {prediction_count}/{len(patch_paths)} patch predictions.",
                )
                pending_arrays.clear()
                pending_profiles.clear()
                pending_masks.clear()
                pending_output_paths.clear()

        if pending_arrays:
            flushed = _flush_predictions(
                adapter,
                pending_arrays,
                pending_profiles,
                pending_masks,
                pending_output_paths,
            )
            prediction_count += flushed
            processed_count += flushed
            _emit_progress(
                progress_callback,
                stage="infer",
                status="running",
                percent=processed_count / len(patch_paths) if patch_paths else 1.0,
                processed=processed_count,
                total=len(patch_paths),
                written=prediction_count,
                skipped=len(skipped_tasks),
                rewritten=0,
                current_action="writing final prediction batch",
                current_reason=f"{flushed} patches",
                message=f"Wrote {prediction_count}/{len(patch_paths)} patch predictions.",
            )

    finished_at = _utc_now()
    duration_seconds = max(0.0, time.perf_counter() - started_perf)
    patches_per_second = (len(patch_paths) / duration_seconds) if duration_seconds > 0 else 0.0
    predictions_per_second = (prediction_count / duration_seconds) if duration_seconds > 0 else 0.0
    result = InferenceResult(
        model_name=config.model_name,
        adapter_name=adapter.name,
        checkpoint_path=config.checkpoint_path,
        patch_dir=config.outputs.patches,
        prediction_dir=prediction_dir,
        manifest_path=manifest_path,
        patch_count=len(patch_paths),
        prediction_count=prediction_count,
        skipped_count=len(skipped_tasks),
        output_paths=output_paths,
        skipped_paths=skipped_paths,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        patches_per_second=patches_per_second,
        predictions_per_second=predictions_per_second,
        overwrite=config.overwrite,
    )
    _write_manifest(
        manifest_path,
        result=result,
        config=config,
        written_paths=tuple(task.output_path.resolve() for task in pending_tasks),
    )
    _emit_progress(
        progress_callback,
        stage="infer",
        status="completed",
        percent=1.0,
        processed=len(patch_paths),
        total=len(patch_paths),
        written=prediction_count,
        skipped=len(skipped_tasks),
        rewritten=0,
        duration_seconds=duration_seconds,
        inference_duration_seconds=duration_seconds,
        patches_per_second=patches_per_second,
        predictions_per_second=predictions_per_second,
        inference_patches_per_second=patches_per_second,
        inference_predictions_per_second=predictions_per_second,
        inference_patch_count=len(patch_paths),
        inference_prediction_count=prediction_count,
        current_action="writing inference manifest",
        current_reason=str(manifest_path),
        message=f"Inference completed: {prediction_count} predictions, {len(skipped_tasks)} skipped.",
    )
    return result


@dataclass(frozen=True)
class _PatchTask:
    patch_path: Path
    output_path: Path


@dataclass(frozen=True)
class _PatchData:
    patch_path: Path
    output_path: Path
    array: Any
    profile: dict[str, Any]
    mask: Any


def _iter_patch_data(tasks: list[_PatchTask], num_workers: int) -> Any:
    if num_workers <= 0 or len(tasks) <= 1:
        for task in tasks:
            yield _read_patch_data(task)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(_read_patch_data, tasks)


def _read_patch_data(task: _PatchTask) -> _PatchData:
    rasterio = require_rasterio()
    try:
        with rasterio.open(task.patch_path) as src:
            return _PatchData(
                patch_path=task.patch_path,
                output_path=task.output_path,
                array=src.read(),
                profile=src.profile.copy(),
                mask=src.dataset_mask(),
            )
    except Exception as exc:
        raise InferenceError(f"Failed to read patch {task.patch_path}: {exc}") from exc


def _prepare_model_input_array(array: Any, *, config: PlatformConfig) -> Any:
    """Return raw patch channels; model adapters consume dataset channel metadata."""

    return array


def _write_manifest(
    manifest_path: Path,
    *,
    result: InferenceResult,
    config: PlatformConfig,
    written_paths: tuple[Path, ...],
) -> None:
    manifest = {
        "model_name": result.model_name,
        "adapter_name": result.adapter_name,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "patch_count": result.patch_count,
        "prediction_count": result.prediction_count,
        "skipped_count": result.skipped_count,
        "output_paths": [str(path) for path in result.output_paths],
        "written_output_paths": [str(path) for path in written_paths],
        "skipped_output_paths": [str(path) for path in result.skipped_paths],
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "patches_per_second": result.patches_per_second,
        "predictions_per_second": result.predictions_per_second,
        "patch_dir": str(result.patch_dir),
        "prediction_dir": str(result.prediction_dir),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "device": config.device,
        "overwrite": config.overwrite,
        "config_path": str(config.config_path) if config.config_path else None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _discover_patch_paths(config: PlatformConfig) -> list[Path]:
    index_path = config.output_dir / "patch_index.geojson"
    if index_path.exists():
        return _patch_paths_from_index(index_path)
    return sorted(
        path for path in config.outputs.patches.glob("*.tif") if not path.name.startswith("._")
    )


def _patch_paths_from_index(index_path: Path) -> list[Path]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Invalid patch index JSON: {index_path}: {exc}") from exc

    features = index.get("features", [])
    if not isinstance(features, list):
        raise InferenceError(f"Invalid patch index: features must be a list in {index_path}.")

    patch_paths: list[Path] = []
    missing_paths: list[Path] = []
    for feature in features:
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        patch_value = properties.get("patch_path") if isinstance(properties, dict) else None
        if not patch_value:
            continue
        patch_path = Path(str(patch_value))
        if not patch_path.is_absolute():
            patch_path = index_path.parent / patch_path
        if patch_path.exists():
            patch_paths.append(patch_path.resolve())
        else:
            missing_paths.append(patch_path)
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        suffix = "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing total)"
        raise InferenceError(f"Patch index references missing patch GeoTIFFs: {preview}{suffix}")
    return patch_paths


def _flush_predictions(
    adapter: Any,
    arrays: list[Any],
    profiles: list[dict[str, Any]],
    masks: list[Any],
    output_paths: list[Path],
) -> int:
    rasterio = require_rasterio()
    try:
        logits_batch = adapter.predict_batch(arrays)
    except Exception as exc:
        raise InferenceError(f"Adapter '{adapter.name}' failed during batch prediction: {exc}") from exc

    if len(logits_batch) != len(arrays):
        raise InferenceError(
            f"Adapter '{adapter.name}' returned {len(logits_batch)} predictions for {len(arrays)} input patches."
        )

    for logits, profile, mask, output_path in zip(logits_batch, profiles, masks, output_paths):
        prediction = adapter.postprocess_logits(logits)
        prediction = _match_prediction_shape(prediction, height=int(profile["height"]), width=int(profile["width"]))
        profile.update(count=1, dtype="uint8", nodata=None)
        try:
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(prediction, 1)
                dst.write_mask(mask)
        except Exception as exc:
            raise InferenceError(f"Failed to write prediction {output_path}: {exc}") from exc
    return len(arrays)


def _match_prediction_shape(prediction: Any, *, height: int, width: int) -> Any:
    import numpy as np

    prediction_array = np.asarray(prediction, dtype=np.uint8)
    if prediction_array.shape == (height, width):
        return prediction_array
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise InferenceError(
            "Prediction shape does not match patch shape and OpenCV is unavailable for resizing: "
            f"prediction={prediction_array.shape}, patch={(height, width)}."
        ) from exc
    return cv2.resize(prediction_array, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
