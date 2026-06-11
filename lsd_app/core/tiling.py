"""Sliding-window tiling for prepare-stage raster processing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from lsd_app.config import PlatformConfig
from lsd_app.core.raster import RasterError, RasterMetadata, read_raster_metadata, require_rasterio


@dataclass(frozen=True)
class PatchRecord:
    """One prepared patch and its source-window metadata."""

    patch_path: Path
    row_off: int
    col_off: int
    height: int
    width: int
    tile_size: int
    stride: int
    padded: bool
    pad_bottom: int
    pad_right: int
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    crs: str | None

    def to_feature(self, *, index_dir: Path) -> dict[str, Any]:
        """Return a GeoJSON feature for this patch."""

        left, bottom, right, top = self.bounds
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [left, bottom],
                    [right, bottom],
                    [right, top],
                    [left, top],
                    [left, bottom],
                ]
            ],
        }
        rel_path = self.patch_path.relative_to(index_dir) if self.patch_path.is_relative_to(index_dir) else self.patch_path
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "patch_path": str(rel_path),
                "row_off": self.row_off,
                "col_off": self.col_off,
                "window": {
                    "row_start": self.row_off,
                    "row_stop": self.row_off + self.height,
                    "col_start": self.col_off,
                    "col_stop": self.col_off + self.width,
                    "height": self.height,
                    "width": self.width,
                },
                "tile_size": self.tile_size,
                "stride": self.stride,
                "transform": list(self.transform),
                "crs": self.crs,
                "padded": self.padded,
                "pad_bottom": self.pad_bottom,
                "pad_right": self.pad_right,
            },
        }


@dataclass(frozen=True)
class PrepareResult:
    """Summary returned by the prepare stage."""

    metadata: RasterMetadata
    patch_dir: Path
    index_path: Path
    saved_patches: int
    skipped_nodata: int
    skipped_all_zero: int
    total_windows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary."""

        return {
            "metadata": self.metadata.to_dict(),
            "patch_dir": str(self.patch_dir),
            "index_path": str(self.index_path),
            "saved_patches": self.saved_patches,
            "skipped_nodata": self.skipped_nodata,
            "skipped_all_zero": self.skipped_all_zero,
            "total_windows": self.total_windows,
        }


def iter_windows(width: int, height: int, tile_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    """Yield row/col offsets and source-window sizes for a sliding grid."""

    for row_off in range(0, height, stride):
        window_height = min(tile_size, height - row_off)
        for col_off in range(0, width, stride):
            window_width = min(tile_size, width - col_off)
            yield row_off, col_off, window_height, window_width


def normalize_patch_minmax(
    patch: Any,
    *,
    valid_mask: Any,
    nodata_values: tuple[Any, ...] = (),
    fill_value: float = 0.0,
    preserve_zero: bool = True,
) -> Any:
    """Normalize each channel to 0..1 while keeping zero and nodata pixels at fill_value."""

    import numpy as np

    normalized = np.full(patch.shape, fill_value, dtype=np.float32)
    finite_valid = np.isfinite(patch)
    for channel_index in range(patch.shape[0]):
        channel = patch[channel_index]
        channel_valid = valid_mask & finite_valid[channel_index]
        if channel_index < len(nodata_values) and nodata_values[channel_index] is not None:
            channel_valid &= ~_nodata_pixels(channel, nodata_values[channel_index])
        if preserve_zero:
            channel_valid &= channel != 0

        if not np.any(channel_valid):
            continue

        values = channel[channel_valid]
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value <= min_value:
            continue

        normalized[channel_index, channel_valid] = (channel[channel_valid] - min_value) / (max_value - min_value)
    return normalized


ProgressCallback = Callable[[Mapping[str, Any]], None]


def prepare_raster(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PrepareResult:
    """Cut a configured input raster into normalized sliding-window GeoTIFF patches."""

    metadata = read_raster_metadata(config.input_raster_path)
    rasterio = require_rasterio()
    from rasterio.windows import Window
    import numpy as np

    patch_dir = config.outputs.patches
    patch_dir.mkdir(parents=True, exist_ok=True)
    index_path = config.output_dir / "patch_index.geojson"

    records: list[PatchRecord] = []
    skipped_nodata = 0
    skipped_all_zero = 0
    total_windows = 0
    fill_value = _numeric_fill_value(config.nodata.output_value)

    try:
        with rasterio.open(config.input_raster_path) as src:
            total_windows = _count_windows(src.width, src.height, config.tile_size, config.stride)
            _emit_progress(
                progress_callback,
                stage="prepare",
                status="running",
                percent=0.0,
                processed=0,
                total=total_windows,
                written=0,
                skipped=0,
                rewritten=0,
                current_action="scanning source raster",
                message="Scanning source raster and preparing sliding windows.",
            )
            processed_windows = 0
            for row_off, col_off, window_height, window_width in iter_windows(
                src.width,
                src.height,
                config.tile_size,
                config.stride,
            ):
                processed_windows += 1
                window = Window(col_off, row_off, window_width, window_height)
                source_valid_mask = _read_valid_mask(src, window, config.nodata.mask_nodata)
                if config.nodata.mask_nodata and not np.any(source_valid_mask):
                    skipped_nodata += 1
                    _emit_progress(
                        progress_callback,
                        stage="prepare",
                        status="running",
                        percent=processed_windows / total_windows if total_windows else 1.0,
                        processed=processed_windows,
                        total=total_windows,
                        written=len(records),
                        skipped=skipped_nodata + skipped_all_zero,
                        rewritten=0,
                        current_action="skipping nodata window",
                        current_reason=f"row={row_off} col={col_off}",
                        message=f"Skipped nodata window at row={row_off}, col={col_off}.",
                    )
                    continue

                patch = src.read(window=window, out_dtype="float32")
                if config.nodata.mask_nodata and config.nodata.input_value is not None:
                    source_valid_mask &= ~np.any(_nodata_pixels(patch, config.nodata.input_value), axis=0)

                if not _has_nonzero_valid_pixel(patch, source_valid_mask):
                    skipped_all_zero += 1
                    _emit_progress(
                        progress_callback,
                        stage="prepare",
                        status="running",
                        percent=processed_windows / total_windows if total_windows else 1.0,
                        processed=processed_windows,
                        total=total_windows,
                        written=len(records),
                        skipped=skipped_nodata + skipped_all_zero,
                        rewritten=0,
                        current_action="skipping empty window",
                        current_reason=f"row={row_off} col={col_off}",
                        message=f"Skipped empty window at row={row_off}, col={col_off}.",
                    )
                    continue

                padded = window_width < config.tile_size or window_height < config.tile_size
                padded_patch = np.full(
                    (src.count, config.tile_size, config.tile_size),
                    fill_value,
                    dtype=np.float32,
                )
                padded_valid_mask = np.zeros((config.tile_size, config.tile_size), dtype=bool)
                padded_patch[:, :window_height, :window_width] = patch
                padded_valid_mask[:window_height, :window_width] = source_valid_mask

                normalized = normalize_patch_minmax(
                    padded_patch,
                    valid_mask=padded_valid_mask,
                    nodata_values=tuple(src.nodatavals),
                    fill_value=fill_value,
                    preserve_zero=True,
                )

                patch_transform = rasterio.windows.transform(window, src.transform)
                patch_bounds = rasterio.windows.bounds(window, src.transform)
                patch_path = patch_dir / f"{Path(src.name).stem}_r{row_off}_c{col_off}.tif"
                profile = {
                    "driver": "GTiff",
                    "height": config.tile_size,
                    "width": config.tile_size,
                    "count": src.count,
                    "dtype": "float32",
                    "transform": patch_transform,
                    "crs": src.crs,
                    "nodata": None,
                }
                with rasterio.open(patch_path, "w", **profile) as dst:
                    dst.write(normalized)
                    if config.nodata.mask_nodata:
                        dst.write_mask((padded_valid_mask.astype("uint8")) * 255)

                records.append(
                    PatchRecord(
                        patch_path=patch_path.resolve(),
                        row_off=row_off,
                        col_off=col_off,
                        height=window_height,
                        width=window_width,
                        tile_size=config.tile_size,
                        stride=config.stride,
                        padded=padded,
                        pad_bottom=config.tile_size - window_height,
                        pad_right=config.tile_size - window_width,
                        transform=tuple(float(value) for value in patch_transform.to_gdal()),
                        bounds=tuple(float(value) for value in patch_bounds),
                        crs=src.crs.to_string() if src.crs else None,
                    )
                )
                _emit_progress(
                    progress_callback,
                    stage="prepare",
                    status="running",
                    percent=processed_windows / total_windows if total_windows else 1.0,
                    processed=processed_windows,
                    total=total_windows,
                    written=len(records),
                    skipped=skipped_nodata + skipped_all_zero,
                    rewritten=0,
                    current_action="writing standardized patch",
                    current_reason=patch_path.name,
                    message=f"Prepared patch {patch_path.name}.",
                )
    except RasterError:
        raise
    except Exception as exc:
        raise RasterError(f"Failed to prepare raster patches: {exc}") from exc

    _write_patch_index(index_path, metadata=metadata, records=records)
    _emit_progress(
        progress_callback,
        stage="prepare",
        status="completed",
        percent=1.0,
        processed=total_windows,
        total=total_windows,
        written=len(records),
        skipped=skipped_nodata + skipped_all_zero,
        rewritten=0,
        current_action="writing patch index",
        message=f"Prepared {len(records)} patches from {total_windows} windows.",
    )
    return PrepareResult(
        metadata=metadata,
        patch_dir=patch_dir,
        index_path=index_path,
        saved_patches=len(records),
        skipped_nodata=skipped_nodata,
        skipped_all_zero=skipped_all_zero,
        total_windows=total_windows,
    )


def _read_valid_mask(src: Any, window: Any, mask_nodata: bool) -> Any:
    import numpy as np

    if not mask_nodata:
        return np.ones((int(window.height), int(window.width)), dtype=bool)
    masks = src.read_masks(window=window)
    if masks.ndim == 2:
        return masks > 0
    return np.any(masks > 0, axis=0)


def _count_windows(width: int, height: int, tile_size: int, stride: int) -> int:
    rows = len(range(0, height, stride))
    cols = len(range(0, width, stride))
    return rows * cols


def _has_nonzero_valid_pixel(patch: Any, valid_mask: Any, *, zero_tolerance: float = 1e-6) -> bool:
    import numpy as np

    band_count = min(3, patch.shape[0])
    front = patch[:band_count]
    nonzero = np.any(np.abs(front) > zero_tolerance, axis=0)
    return bool(np.any(nonzero & valid_mask))


def _numeric_fill_value(value: object) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(number):
        return number
    return 0.0


def _nodata_pixels(array: Any, value: object) -> Any:
    import numpy as np

    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.zeros(array.shape, dtype=bool)
    if math.isnan(number):
        return np.isnan(array)
    return array == number


def _write_patch_index(index_path: Path, *, metadata: RasterMetadata, records: list[PatchRecord]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "source": metadata.to_dict(),
            "patch_count": len(records),
        },
        "features": [record.to_feature(index_dir=index_path.parent) for record in records],
    }
    index_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
