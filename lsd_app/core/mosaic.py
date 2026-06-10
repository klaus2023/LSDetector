"""Mosaic patch predictions back into source raster space."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from lsd_app.config import PlatformConfig
from lsd_app.core.raster import require_rasterio


class MosaicError(RuntimeError):
    """Raised when patch predictions cannot be mosaicked."""


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class MosaicResult:
    """Summary returned by the mosaic stage."""

    probability_path: Path
    mask_path: Path
    patch_count: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary."""

        return {
            "probability_path": str(self.probability_path),
            "mask_path": str(self.mask_path),
            "patch_count": self.patch_count,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class _SourceGrid:
    width: int
    height: int
    crs: str | None
    transform: Any


@dataclass(frozen=True)
class _PatchMosaicRecord:
    prediction_path: Path
    row_off: int
    col_off: int
    height: int
    width: int


def run_mosaic(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> MosaicResult:
    """Average-overlap patch predictions into full-resolution probability and mask rasters."""

    rasterio = require_rasterio()
    from rasterio.windows import Window
    import numpy as np

    index_path = config.output_dir / "patch_index.geojson"
    source_grid, records = _load_patch_index(index_path, prediction_dir=config.outputs.patch_predictions)
    if not records:
        raise MosaicError(f"Patch index contains no patch records: {index_path}")
    _emit_progress(
        progress_callback,
        stage="mosaic",
        status="running",
        percent=0.0,
        processed=0,
        total=len(records),
        written=0,
        skipped=0,
        rewritten=0,
        current_action="loading patch predictions",
        current_reason=str(index_path),
        message=f"Mosaicking {len(records)} patch predictions.",
    )

    missing_predictions = [record.prediction_path for record in records if not record.prediction_path.exists()]
    if missing_predictions:
        preview = ", ".join(str(path) for path in missing_predictions[:5])
        suffix = "" if len(missing_predictions) <= 5 else f", ... ({len(missing_predictions)} missing total)"
        raise MosaicError(f"Missing patch prediction GeoTIFFs: {preview}{suffix}")

    config.outputs.rasters.mkdir(parents=True, exist_ok=True)
    probability_path = config.outputs.rasters / "probability.tif"
    mask_path = config.outputs.rasters / "mask.tif"
    sum_path = config.outputs.rasters / "_mosaic_sum.tmp.tif"
    count_path = config.outputs.rasters / "_mosaic_count.tmp.tif"

    try:
        accumulator_profile = _base_profile(
            config,
            source_grid=source_grid,
            dtype="float32",
            nodata=None,
        )
        _create_zero_raster(sum_path, accumulator_profile)
        _create_zero_raster(count_path, accumulator_profile)

        with rasterio.open(sum_path, "r+") as sum_dst, rasterio.open(count_path, "r+") as count_dst:
            for index, record in enumerate(records, start=1):
                prediction, valid_mask = _read_prediction_window(record)
                if not np.any(valid_mask):
                    _emit_progress(
                        progress_callback,
                        stage="mosaic",
                        status="running",
                        percent=index / len(records) * 0.75,
                        processed=index,
                        total=len(records),
                        written=0,
                        skipped=0,
                        rewritten=0,
                        current_action="skipping empty prediction",
                        current_reason=record.prediction_path.name,
                        message=f"Skipping empty prediction {record.prediction_path.name}.",
                    )
                    continue

                target_window = Window(record.col_off, record.row_off, record.width, record.height)
                current_sum = sum_dst.read(1, window=target_window, out_dtype="float32")
                current_count = count_dst.read(1, window=target_window, out_dtype="float32")
                current_sum[valid_mask] += prediction[valid_mask]
                current_count[valid_mask] += 1.0
                sum_dst.write(current_sum, 1, window=target_window)
                count_dst.write(current_count, 1, window=target_window)
                _emit_progress(
                    progress_callback,
                    stage="mosaic",
                    status="running",
                    percent=index / len(records) * 0.75,
                    processed=index,
                    total=len(records),
                    written=index,
                    skipped=0,
                    rewritten=0,
                    current_action="accumulating prediction",
                    current_reason=record.prediction_path.name,
                    message=f"Accumulated prediction {index}/{len(records)}.",
                )

        probability_profile = _base_profile(
            config,
            source_grid=source_grid,
            dtype="float32",
            nodata=None,
        )
        mask_profile = _base_profile(
            config,
            source_grid=source_grid,
            dtype="uint8",
            nodata=0,
        )
        threshold = float(config.threshold)
        _emit_progress(
            progress_callback,
            stage="mosaic",
            status="running",
            percent=0.8,
            processed=len(records),
            total=len(records),
            written=len(records),
            skipped=0,
            rewritten=0,
            current_action="writing probability and mask rasters",
            current_reason=str(config.outputs.rasters),
            message="Writing full-scene probability.tif and mask.tif.",
        )
        with (
            rasterio.open(sum_path) as sum_src,
            rasterio.open(count_path) as count_src,
            rasterio.open(probability_path, "w", **probability_profile) as probability_dst,
            rasterio.open(mask_path, "w", **mask_profile) as mask_dst,
        ):
            for _, window in probability_dst.block_windows(1):
                sum_values = sum_src.read(1, window=window, out_dtype="float32")
                count_values = count_src.read(1, window=window, out_dtype="float32")
                has_prediction = count_values > 0
                probability = np.zeros(sum_values.shape, dtype=np.float32)
                probability[has_prediction] = sum_values[has_prediction] / count_values[has_prediction]
                probability = np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)
                mask = (probability >= threshold).astype(np.uint8) * 255
                dataset_mask = has_prediction.astype(np.uint8) * 255
                probability_dst.write(probability, 1, window=window)
                probability_dst.write_mask(dataset_mask, window=window)
                mask_dst.write(mask, 1, window=window)
                mask_dst.write_mask(dataset_mask, window=window)
    except MosaicError:
        raise
    except Exception as exc:
        raise MosaicError(f"Failed to mosaic patch predictions: {exc}") from exc
    finally:
        _safe_unlink(sum_path)
        _safe_unlink(count_path)

    _emit_progress(
        progress_callback,
        stage="mosaic",
        status="completed",
        percent=1.0,
        processed=len(records),
        total=len(records),
        written=len(records),
        skipped=0,
        rewritten=0,
        current_action="finished mosaic",
        current_reason=str(mask_path),
        message=f"Mosaic completed: {mask_path}.",
    )
    return MosaicResult(
        probability_path=probability_path,
        mask_path=mask_path,
        patch_count=len(records),
        width=source_grid.width,
        height=source_grid.height,
    )


def _load_patch_index(index_path: Path, *, prediction_dir: Path) -> tuple[_SourceGrid, list[_PatchMosaicRecord]]:
    if not index_path.exists():
        raise MosaicError(f"Patch index does not exist: {index_path}. Run `lsd prepare --config ...` first.")

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MosaicError(f"Invalid patch index JSON: {index_path}: {exc}") from exc

    if not isinstance(index, dict):
        raise MosaicError(f"Invalid patch index: expected JSON object in {index_path}.")

    source = index.get("properties", {}).get("source") if isinstance(index.get("properties"), dict) else None
    if not isinstance(source, dict):
        raise MosaicError(f"Invalid patch index: missing properties.source in {index_path}.")

    source_grid = _source_grid_from_metadata(source, index_path=index_path)
    features = index.get("features", [])
    if not isinstance(features, list):
        raise MosaicError(f"Invalid patch index: features must be a list in {index_path}.")

    records: list[_PatchMosaicRecord] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            continue
        patch_value = properties.get("patch_path")
        if not patch_value:
            continue
        window = properties.get("window", {})
        if not isinstance(window, dict):
            window = {}
        row_off = _int_value(properties.get("row_off", window.get("row_start")), "row_off")
        col_off = _int_value(properties.get("col_off", window.get("col_start")), "col_off")
        height = _int_value(window.get("height", properties.get("height")), "window.height")
        width = _int_value(window.get("width", properties.get("width")), "window.width")
        if height <= 0 or width <= 0:
            raise MosaicError(f"Invalid patch index window size for {patch_value}: width={width}, height={height}")
        if row_off < 0 or col_off < 0 or row_off + height > source_grid.height or col_off + width > source_grid.width:
            raise MosaicError(
                f"Patch window is outside source raster bounds for {patch_value}: "
                f"row={row_off}, col={col_off}, height={height}, width={width}"
            )
        records.append(
            _PatchMosaicRecord(
                prediction_path=prediction_dir / Path(str(patch_value)).name,
                row_off=row_off,
                col_off=col_off,
                height=height,
                width=width,
            )
        )
    return source_grid, records


def _source_grid_from_metadata(source: dict[str, Any], *, index_path: Path) -> _SourceGrid:
    from affine import Affine

    try:
        width = int(source["width"])
        height = int(source["height"])
        transform_values = source["transform"]
    except KeyError as exc:
        raise MosaicError(f"Invalid patch index source metadata in {index_path}: missing {exc.args[0]}.") from exc

    if width <= 0 or height <= 0:
        raise MosaicError(f"Invalid source raster shape in {index_path}: width={width}, height={height}")
    if not isinstance(transform_values, list) or len(transform_values) != 6:
        raise MosaicError(f"Invalid source transform in {index_path}: expected six GDAL transform values.")

    transform = Affine.from_gdal(*(float(value) for value in transform_values))
    crs = source.get("crs")
    return _SourceGrid(width=width, height=height, crs=str(crs) if crs else None, transform=transform)


def _read_prediction_window(record: _PatchMosaicRecord) -> tuple[Any, Any]:
    rasterio = require_rasterio()
    from rasterio.windows import Window
    import numpy as np

    try:
        with rasterio.open(record.prediction_path) as src:
            if src.count < 1:
                raise MosaicError(f"Patch prediction has no bands: {record.prediction_path}")
            if src.width < record.width or src.height < record.height:
                raise MosaicError(
                    f"Patch prediction is smaller than indexed source window: {record.prediction_path} "
                    f"prediction={src.width}x{src.height} window={record.width}x{record.height}"
                )
            crop_window = Window(0, 0, record.width, record.height)
            prediction = src.read(1, window=crop_window, out_dtype="float32")
            valid_mask = src.dataset_mask(window=crop_window) > 0
    except MosaicError:
        raise
    except Exception as exc:
        raise MosaicError(f"Failed to read patch prediction {record.prediction_path}: {exc}") from exc

    if prediction.size:
        finite = np.isfinite(prediction)
        if not np.all(finite):
            valid_mask &= finite
            prediction = np.where(finite, prediction, 0.0)
        if np.issubdtype(prediction.dtype, np.integer) or float(np.nanmax(prediction)) > 1.0:
            prediction = prediction / 255.0
    return np.clip(prediction, 0.0, 1.0).astype(np.float32, copy=False), valid_mask


def _base_profile(config: PlatformConfig, *, source_grid: _SourceGrid, dtype: str, nodata: int | float | None) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": source_grid.height,
        "width": source_grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": source_grid.crs,
        "transform": source_grid.transform,
        "nodata": nodata,
    }
    profile.update(_creation_options(config))
    return profile


def _creation_options(config: PlatformConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "BIGTIFF": "IF_SAFER",
        "tiled": True,
        "compress": "deflate",
        "blockxsize": 256,
        "blockysize": 256,
    }
    raw_options = getattr(config, "raster_options", {}) or {}
    for key, value in raw_options.items():
        normalized_key = "BIGTIFF" if key.lower() == "bigtiff" else key
        options[normalized_key] = value
    if not options.get("tiled", True):
        options.pop("blockxsize", None)
        options.pop("blockysize", None)
    return options


def _create_zero_raster(path: Path, profile: dict[str, Any]) -> None:
    rasterio = require_rasterio()
    import numpy as np

    with rasterio.open(path, "w", **profile) as dst:
        for _, window in dst.block_windows(1):
            zeros = np.zeros((int(window.height), int(window.width)), dtype=np.float32)
            dst.write(zeros, 1, window=window)


def _int_value(value: Any, field_name: str) -> int:
    if value is None:
        raise MosaicError(f"Invalid patch index: missing {field_name}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MosaicError(f"Invalid patch index {field_name}: expected integer, got {value!r}.") from exc


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
