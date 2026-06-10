"""Vectorize raster landslide masks into geospatial polygon outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping

from lsd_app.config import PlatformConfig
from lsd_app.core.raster import require_rasterio
from lsd_app.core.stats import VectorStats, summarize_vector_features, write_vector_stats


class VectorizeError(RuntimeError):
    """Raised when a raster mask cannot be vectorized."""


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class VectorFeature:
    """A vectorized landslide polygon and its attributes."""

    id: int
    geometry: dict[str, Any]
    area_m2: float
    area_px: int
    mean_confidence: float | None

    def properties(self) -> dict[str, int | float | None]:
        """Return the feature attributes required by downstream outputs."""

        return {
            "id": self.id,
            "area_m2": self.area_m2,
            "area_px": self.area_px,
            "mean_confidence": self.mean_confidence,
        }


@dataclass(frozen=True)
class VectorizeResult:
    """Summary returned by the vectorize stage."""

    gpkg_path: Path
    geojson_path: Path | None
    shapefile_path: Path | None
    stats_csv_path: Path
    stats_json_path: Path
    polygon_count: int
    total_area_m2: float
    mean_area_m2: float
    mean_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result mapping."""

        return {
            "gpkg_path": str(self.gpkg_path),
            "geojson_path": str(self.geojson_path) if self.geojson_path else None,
            "shapefile_path": str(self.shapefile_path) if self.shapefile_path else None,
            "stats_csv_path": str(self.stats_csv_path),
            "stats_json_path": str(self.stats_json_path),
            "polygon_count": self.polygon_count,
            "total_area_m2": self.total_area_m2,
            "mean_area_m2": self.mean_area_m2,
            "mean_confidence": self.mean_confidence,
        }


@dataclass(frozen=True)
class _RasterContext:
    width: int
    height: int
    transform: Any
    crs_string: str | None
    crs_wkt: str | None
    srs_id: int
    pixel_area_units: float
    pixel_area_m2: float


@dataclass
class _VectorStatsAccumulator:
    polygon_count: int = 0
    total_area_m2: float = 0.0
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def add(self, feature: VectorFeature) -> None:
        self.polygon_count += 1
        self.total_area_m2 += float(feature.area_m2)
        if feature.mean_confidence is not None:
            self.confidence_sum += float(feature.mean_confidence)
            self.confidence_count += 1

    def to_stats(self) -> VectorStats:
        return VectorStats(
            polygon_count=self.polygon_count,
            total_area_m2=float(self.total_area_m2),
            mean_area_m2=float(self.total_area_m2 / self.polygon_count) if self.polygon_count else 0.0,
            mean_confidence=(
                float(self.confidence_sum / self.confidence_count) if self.confidence_count else None
            ),
        )


class _ExitStack:
    def __init__(self, managers: list[Any]) -> None:
        self.managers = managers
        self.entered: list[Any] = []

    def __enter__(self) -> list[Any]:
        try:
            for manager in self.managers:
                self.entered.append(manager.__enter__())
            return self.entered
        except Exception:
            self.__exit__(*(__import__("sys").exc_info()))
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        raised: BaseException | None = None
        for manager in reversed(self.entered):
            try:
                manager.__exit__(exc_type, exc, traceback)
            except BaseException as close_exc:
                raised = close_exc
        if raised is not None:
            raise raised


def run_vectorize(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> VectorizeResult:
    """Extract landslide polygons from `rasters/mask.tif` and write vector outputs."""

    if getattr(config, "vectorize_mode", "streaming") == "legacy":
        return run_vectorize_legacy(config, progress_callback=progress_callback)
    return run_vectorize_streaming(config, progress_callback=progress_callback)


def run_vectorize_legacy(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> VectorizeResult:
    """Legacy in-memory vectorization path kept for explicit rollback."""

    rasterio = require_rasterio()
    import numpy as np

    mask_path = config.outputs.rasters / "mask.tif"
    probability_path = config.outputs.rasters / "probability.tif"
    if not mask_path.exists():
        raise VectorizeError(f"Mask raster does not exist: {mask_path}. Run `lsd mosaic --config ...` first.")
    _emit_progress(
        progress_callback,
        stage="vectorize",
        status="running",
        percent=0.0,
        processed=0,
        total=4,
        written=0,
        skipped=0,
        rewritten=0,
        current_action="reading mask raster",
        current_reason=str(mask_path),
        message="Reading mosaic mask and probability rasters.",
    )

    try:
        with rasterio.open(mask_path) as mask_src:
            if mask_src.count < 1:
                raise VectorizeError(f"Mask raster has no bands: {mask_path}")
            mask_values = mask_src.read(1)
            valid_mask = mask_src.dataset_mask() > 0
            positive_mask = (mask_values > 0) & valid_mask
            context = _raster_context(mask_src)
            probability_values, probability_valid = _read_probability(probability_path, mask_src=mask_src)
            _emit_progress(
                progress_callback,
                stage="vectorize",
                status="running",
                percent=0.3,
                processed=1,
                total=4,
                written=0,
                skipped=0,
                rewritten=0,
                current_action="extracting connected polygons",
                current_reason=str(mask_path),
                message="Extracting connected landslide components.",
            )
            features = _extract_features(
                positive_mask=positive_mask,
                transform=mask_src.transform,
                pixel_area_units=context.pixel_area_units,
                pixel_area_m2=context.pixel_area_m2,
                probability_values=probability_values,
                probability_valid=probability_valid,
                min_area_pixels=config.min_area_pixels,
                simplify_tolerance=config.simplify_tolerance,
                connectivity=config.connectivity,
            )
    except VectorizeError:
        raise
    except Exception as exc:
        raise VectorizeError(f"Failed to vectorize mask raster {mask_path}: {exc}") from exc

    config.outputs.vectors.mkdir(parents=True, exist_ok=True)
    gpkg_path = config.outputs.vectors / "landslides.gpkg"
    geojson_path = config.outputs.vectors / "landslides.geojson"
    shapefile_path = config.outputs.vectors / "landslides.shp"

    _emit_progress(
        progress_callback,
        stage="vectorize",
        status="running",
        percent=0.65,
        processed=2,
        total=4,
        written=0,
        skipped=0,
        rewritten=0,
        current_action="writing geopackage",
        current_reason=str(gpkg_path),
        message=f"Writing {len(features)} polygons to GeoPackage.",
    )
    _write_geopackage(gpkg_path, features, context=context)
    written_geojson_path = None
    written_shapefile_path = None
    if "geojson" in config.vector_formats:
        _emit_progress(
            progress_callback,
            stage="vectorize",
            status="running",
            percent=0.78,
            processed=3,
            total=4,
            written=len(features),
            skipped=0,
            rewritten=0,
            current_action="writing geojson",
            current_reason=str(geojson_path),
            message="Writing GeoJSON polygon output.",
        )
        _write_geojson(geojson_path, features, crs_string=context.crs_string)
        written_geojson_path = geojson_path
    if "shapefile" in config.vector_formats:
        _emit_progress(
            progress_callback,
            stage="vectorize",
            status="running",
            percent=0.88,
            processed=3,
            total=4,
            written=len(features),
            skipped=0,
            rewritten=0,
            current_action="writing shapefile",
            current_reason=str(shapefile_path),
            message="Writing ESRI Shapefile polygon output.",
        )
        _write_shapefile(shapefile_path, features, crs_wkt=context.crs_wkt)
        written_shapefile_path = shapefile_path

    stats = summarize_vector_features(features)
    stats_csv_path, stats_json_path = write_vector_stats(stats, config.outputs.metrics)
    _emit_progress(
        progress_callback,
        stage="vectorize",
        status="completed",
        percent=1.0,
        processed=4,
        total=4,
        written=len(features),
        skipped=0,
        rewritten=0,
        current_action="finished vectorization",
        current_reason=str(shapefile_path if written_shapefile_path else gpkg_path),
        message=f"Vectorization completed with {len(features)} polygons.",
    )
    return _result_from_stats(
        gpkg_path=gpkg_path,
        geojson_path=written_geojson_path,
        shapefile_path=written_shapefile_path,
        stats_csv_path=stats_csv_path,
        stats_json_path=stats_json_path,
        stats=stats,
    )


def run_vectorize_streaming(
    config: PlatformConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> VectorizeResult:
    """Low-memory vectorization path using GDAL polygonize plus streaming writers."""

    rasterio = require_rasterio()

    mask_path = config.outputs.rasters / "mask.tif"
    probability_path = config.outputs.rasters / "probability.tif"
    if not mask_path.exists():
        raise VectorizeError(f"Mask raster does not exist: {mask_path}. Run `lsd mosaic --config ...` first.")
    if config.connectivity not in {4, 8}:
        raise VectorizeError(f"Invalid connectivity: expected 4 or 8, got {config.connectivity}.")

    _emit_progress(
        progress_callback,
        stage="vectorize",
        status="running",
        percent=0.0,
        processed=0,
        total=4,
        written=0,
        skipped=0,
        rewritten=0,
        current_action="opening mask raster",
        current_reason=str(mask_path),
        message="Opening mosaic mask raster for low-memory polygonization.",
    )

    config.outputs.vectors.mkdir(parents=True, exist_ok=True)
    gpkg_path = config.outputs.vectors / "landslides.gpkg"
    geojson_path = config.outputs.vectors / "landslides.geojson"
    shapefile_path = config.outputs.vectors / "landslides.shp"
    written_geojson_path = geojson_path if "geojson" in config.vector_formats else None
    written_shapefile_path = shapefile_path if "shapefile" in config.vector_formats else None

    try:
        with rasterio.open(mask_path) as mask_src:
            if mask_src.count < 1:
                raise VectorizeError(f"Mask raster has no bands: {mask_path}")
            context = _raster_context(mask_src)
            probability_src = _open_probability_reader(probability_path, mask_src=mask_src)
            try:
                with tempfile.TemporaryDirectory(prefix="lsd_vectorize_") as tmp_dir:
                    raw_gpkg_path = Path(tmp_dir) / "raw_polygons.gpkg"
                    _emit_progress(
                        progress_callback,
                        stage="vectorize",
                        status="running",
                        percent=0.18,
                        processed=1,
                        total=4,
                        written=0,
                        skipped=0,
                        rewritten=0,
                        current_action="extracting connected polygons",
                        current_reason=str(mask_path),
                        message="Extracting connected landslide polygons with GDAL polygonize.",
                    )
                    _polygonize_mask_to_gpkg(
                        mask_path=mask_path,
                        output_path=raw_gpkg_path,
                        connectivity=config.connectivity,
                    )
                    raw_count = _raw_polygon_count(raw_gpkg_path)
                    _emit_progress(
                        progress_callback,
                        stage="vectorize",
                        status="running",
                        percent=0.52,
                        processed=2,
                        total=4,
                        written=0,
                        skipped=0,
                        rewritten=0,
                        current_action="writing vector outputs",
                        current_reason=str(gpkg_path),
                        message=f"Filtering and writing {raw_count} polygon candidates.",
                    )
                    stats = _write_streaming_outputs(
                        raw_gpkg_path=raw_gpkg_path,
                        gpkg_path=gpkg_path,
                        geojson_path=written_geojson_path,
                        shapefile_path=written_shapefile_path,
                        context=context,
                        probability_src=probability_src,
                        min_area_pixels=config.min_area_pixels,
                        simplify_tolerance=config.simplify_tolerance,
                        progress_callback=progress_callback,
                        raw_count=raw_count,
                    )
            finally:
                if probability_src is not None:
                    probability_src.close()
    except VectorizeError:
        raise
    except Exception as exc:
        raise VectorizeError(f"Failed to vectorize mask raster {mask_path}: {exc}") from exc

    stats_csv_path, stats_json_path = write_vector_stats(stats, config.outputs.metrics)
    _emit_progress(
        progress_callback,
        stage="vectorize",
        status="completed",
        percent=1.0,
        processed=4,
        total=4,
        written=stats.polygon_count,
        skipped=0,
        rewritten=0,
        current_action="finished vectorization",
        current_reason=str(shapefile_path if written_shapefile_path else gpkg_path),
        message=f"Vectorization completed with {stats.polygon_count} polygons.",
    )
    return _result_from_stats(
        gpkg_path=gpkg_path,
        geojson_path=written_geojson_path,
        shapefile_path=written_shapefile_path,
        stats_csv_path=stats_csv_path,
        stats_json_path=stats_json_path,
        stats=stats,
    )


def _raster_context(src: Any) -> _RasterContext:
    transform = src.transform
    crs = src.crs
    crs_string = crs.to_string() if crs else None
    crs_wkt = crs.to_wkt() if crs else None
    srs_id = int(crs.to_epsg() or 0) if crs else 0
    pixel_area_units = abs(float(transform.a) * float(transform.e) - float(transform.b) * float(transform.d))
    pixel_area_m2 = pixel_area_units
    if crs:
        try:
            _, factor = crs.linear_units_factor
            pixel_area_m2 = pixel_area_units * float(factor) * float(factor)
        except Exception:
            pixel_area_m2 = pixel_area_units
    return _RasterContext(
        width=int(src.width),
        height=int(src.height),
        transform=transform,
        crs_string=crs_string,
        crs_wkt=crs_wkt,
        srs_id=srs_id,
        pixel_area_units=pixel_area_units,
        pixel_area_m2=pixel_area_m2,
    )


def _open_probability_reader(probability_path: Path, *, mask_src: Any) -> Any | None:
    if not probability_path.exists():
        return None

    rasterio = require_rasterio()
    prob_src = rasterio.open(probability_path)
    try:
        if prob_src.count < 1:
            raise VectorizeError(f"Probability raster has no bands: {probability_path}")
        if prob_src.width != mask_src.width or prob_src.height != mask_src.height:
            raise VectorizeError(
                f"Probability raster shape does not match mask raster: "
                f"probability={prob_src.width}x{prob_src.height} mask={mask_src.width}x{mask_src.height}"
            )
        if not prob_src.transform.almost_equals(mask_src.transform):
            raise VectorizeError("Probability raster transform does not match mask raster.")
        if prob_src.crs != mask_src.crs:
            raise VectorizeError("Probability raster CRS does not match mask raster.")
        return prob_src
    except Exception:
        prob_src.close()
        raise


def _read_probability(probability_path: Path, *, mask_src: Any) -> tuple[Any | None, Any | None]:
    if not probability_path.exists():
        return None, None

    rasterio = require_rasterio()
    import numpy as np

    with rasterio.open(probability_path) as prob_src:
        if prob_src.count < 1:
            raise VectorizeError(f"Probability raster has no bands: {probability_path}")
        if prob_src.width != mask_src.width or prob_src.height != mask_src.height:
            raise VectorizeError(
                f"Probability raster shape does not match mask raster: "
                f"probability={prob_src.width}x{prob_src.height} mask={mask_src.width}x{mask_src.height}"
            )
        if not prob_src.transform.almost_equals(mask_src.transform):
            raise VectorizeError("Probability raster transform does not match mask raster.")
        if prob_src.crs != mask_src.crs:
            raise VectorizeError("Probability raster CRS does not match mask raster.")

        probability = prob_src.read(1, out_dtype="float32")
        valid = prob_src.dataset_mask() > 0

    finite = np.isfinite(probability)
    valid &= finite
    if np.any(valid) and float(np.nanmax(probability[valid])) > 1.0:
        probability = probability / 255.0
    probability = np.clip(np.where(finite, probability, 0.0), 0.0, 1.0).astype("float32", copy=False)
    return probability, valid


def _extract_features(
    *,
    positive_mask: Any,
    transform: Any,
    pixel_area_units: float,
    pixel_area_m2: float,
    probability_values: Any | None,
    probability_valid: Any | None,
    min_area_pixels: int,
    simplify_tolerance: float,
    connectivity: int,
) -> list[VectorFeature]:
    from rasterio import features as raster_features
    import numpy as np

    if connectivity not in {4, 8}:
        raise VectorizeError(f"Invalid connectivity: expected 4 or 8, got {connectivity}.")
    if not np.any(positive_mask):
        return []

    source = positive_mask.astype("uint8", copy=False)
    records: list[VectorFeature] = []
    for geometry, value in raster_features.shapes(source, mask=positive_mask, connectivity=connectivity, transform=transform):
        if int(value) != 1:
            continue
        normalized_geometry = _normalize_geometry(geometry)
        area_px = _area_pixels(normalized_geometry, pixel_area_units)
        if area_px < min_area_pixels:
            continue
        mean_confidence = _mean_confidence(
            normalized_geometry,
            positive_mask=positive_mask,
            transform=transform,
            probability_values=probability_values,
            probability_valid=probability_valid,
        )
        output_geometry = (
            _simplify_geometry(normalized_geometry, simplify_tolerance) if simplify_tolerance > 0 else normalized_geometry
        )
        records.append(
            VectorFeature(
                id=len(records) + 1,
                geometry=output_geometry,
                area_m2=float(area_px * pixel_area_m2),
                area_px=area_px,
                mean_confidence=mean_confidence,
            )
        )
    return records


def _area_pixels(geometry: dict[str, Any], pixel_area_units: float) -> int:
    if pixel_area_units <= 0:
        raise VectorizeError("Mask raster transform has zero-area pixels.")
    area_units = abs(_geometry_area(geometry))
    return int(round(area_units / pixel_area_units))


def _mean_confidence(
    geometry: dict[str, Any],
    *,
    positive_mask: Any,
    transform: Any,
    probability_values: Any | None,
    probability_valid: Any | None,
) -> float | None:
    if probability_values is None or probability_valid is None:
        return None

    from rasterio import features as raster_features
    import numpy as np

    component = raster_features.rasterize(
        [(geometry, 1)],
        out_shape=positive_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    component &= positive_mask
    component &= probability_valid
    if not np.any(component):
        return None
    return float(np.mean(probability_values[component]))


def _mean_confidence_windowed(geometry: dict[str, Any], *, probability_src: Any | None) -> float | None:
    if probability_src is None:
        return None

    from rasterio import features as raster_features
    from rasterio.windows import Window, from_bounds, transform as window_transform
    import numpy as np

    min_x, min_y, max_x, max_y = _geometry_bounds(geometry)
    try:
        raw_window = from_bounds(min_x, min_y, max_x, max_y, transform=probability_src.transform)
    except Exception:
        return None
    window = raw_window.round_offsets().round_lengths()
    try:
        window = window.intersection(Window(0, 0, probability_src.width, probability_src.height))
    except Exception:
        return None
    if window.width <= 0 or window.height <= 0:
        return None

    probability = probability_src.read(1, window=window, out_dtype="float32", masked=False)
    valid = probability_src.read_masks(1, window=window) > 0
    finite = np.isfinite(probability)
    valid &= finite
    if not np.any(valid):
        return None
    if float(np.nanmax(probability[valid])) > 1.0:
        probability = probability / 255.0
    probability = np.clip(np.where(finite, probability, 0.0), 0.0, 1.0).astype("float32", copy=False)

    component = raster_features.rasterize(
        [(geometry, 1)],
        out_shape=probability.shape,
        transform=window_transform(window, probability_src.transform),
        fill=0,
        dtype="uint8",
    ).astype(bool)
    component &= valid
    if not np.any(component):
        return None
    return float(np.mean(probability[component]))


def _polygonize_mask_to_gpkg(*, mask_path: Path, output_path: Path, connectivity: int) -> None:
    executable = shutil.which("gdal_polygonize.py")
    if executable is None:
        raise VectorizeError(
            "Streaming vectorize mode requires gdal_polygonize.py. "
            "Set vectorize.mode: legacy to use the in-memory fallback."
        )
    _safe_unlink(output_path)
    command = [
        executable,
        "-q",
        "-mask",
        str(mask_path),
        "-f",
        "GPKG",
    ]
    if connectivity == 8:
        command.append("-8")
    command.extend([str(mask_path), str(output_path), "landslides", "DN"])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise VectorizeError(f"GDAL polygonize failed with exit code {completed.returncode}: {detail}")
    if not output_path.exists():
        raise VectorizeError("GDAL polygonize did not create a GeoPackage output.")


def _raw_polygon_count(raw_gpkg_path: Path) -> int:
    with sqlite3.connect(raw_gpkg_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM landslides WHERE DN > 0").fetchone()
    return int(row[0] or 0)


def _raw_gpkg_features(raw_gpkg_path: Path) -> Iterable[dict[str, Any]]:
    with sqlite3.connect(raw_gpkg_path) as conn:
        for blob, value in conn.execute("SELECT geom, DN FROM landslides WHERE DN > 0 ORDER BY fid"):
            if int(value or 0) <= 0:
                continue
            yield _geometry_from_gpkg_blob(bytes(blob))


def _write_streaming_outputs(
    *,
    raw_gpkg_path: Path,
    gpkg_path: Path,
    geojson_path: Path | None,
    shapefile_path: Path | None,
    context: _RasterContext,
    probability_src: Any | None,
    min_area_pixels: int,
    simplify_tolerance: float,
    progress_callback: ProgressCallback | None,
    raw_count: int,
) -> VectorStats:
    stats = _VectorStatsAccumulator()
    geojson_writer = _StreamingGeoJsonWriter(geojson_path, crs_string=context.crs_string) if geojson_path else None
    shapefile_writer = _StreamingShapefileWriter(shapefile_path, crs_wkt=context.crs_wkt) if shapefile_path else None
    writers: list[Any] = [_StreamingGeoPackageWriter(gpkg_path, context=context)]
    if geojson_writer is not None:
        writers.append(geojson_writer)
    if shapefile_writer is not None:
        writers.append(shapefile_writer)

    with _ExitStack(writers) as active_writers:
        for index, geometry in enumerate(_raw_gpkg_features(raw_gpkg_path), start=1):
            normalized_geometry = _normalize_geometry(geometry)
            area_px = _area_pixels(normalized_geometry, context.pixel_area_units)
            if area_px < min_area_pixels:
                continue
            mean_confidence = _mean_confidence_windowed(normalized_geometry, probability_src=probability_src)
            output_geometry = (
                _simplify_geometry(normalized_geometry, simplify_tolerance)
                if simplify_tolerance > 0
                else normalized_geometry
            )
            feature = VectorFeature(
                id=stats.polygon_count + 1,
                geometry=output_geometry,
                area_m2=float(area_px * context.pixel_area_m2),
                area_px=area_px,
                mean_confidence=mean_confidence,
            )
            for writer in active_writers:
                writer.write(feature)
            stats.add(feature)
            if index == raw_count or stats.polygon_count % 100 == 0:
                _emit_progress(
                    progress_callback,
                    stage="vectorize",
                    status="running",
                    percent=0.52 + (0.35 * index / max(raw_count, 1)),
                    processed=3,
                    total=4,
                    written=stats.polygon_count,
                    skipped=max(0, index - stats.polygon_count),
                    rewritten=0,
                    current_action="writing vector outputs",
                    current_reason=str(gpkg_path),
                    message=f"Written {stats.polygon_count} polygons from {index}/{raw_count} candidates.",
                )
    return stats.to_stats()


def _write_geopackage(path: Path, features: list[VectorFeature], *, context: _RasterContext) -> None:
    _safe_unlink(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA application_id = 1196444487")
        conn.execute("PRAGMA user_version = 10200")
        _create_gpkg_metadata(conn, features, context=context)
        for feature in features:
            conn.execute(
                "INSERT INTO landslides (geom, id, area_m2, area_px, mean_confidence) VALUES (?, ?, ?, ?, ?)",
                (
                    sqlite3.Binary(_gpkg_geometry_blob(feature.geometry, context.srs_id)),
                    feature.id,
                    feature.area_m2,
                    feature.area_px,
                    feature.mean_confidence,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        _safe_unlink(path)
        raise
    finally:
        conn.close()


class _StreamingGeoPackageWriter:
    def __init__(self, path: Path, *, context: _RasterContext) -> None:
        self.path = path
        self.context = context
        self.conn: sqlite3.Connection | None = None
        self.bounds: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None)
        self.geometry_type = "POLYGON"

    def __enter__(self) -> "_StreamingGeoPackageWriter":
        _safe_unlink(self.path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA application_id = 1196444487")
        self.conn.execute("PRAGMA user_version = 10200")
        _create_gpkg_schema(self.conn, context=self.context)
        return self

    def write(self, feature: VectorFeature) -> None:
        assert self.conn is not None
        if feature.geometry.get("type") == "MultiPolygon":
            self.geometry_type = "MULTIPOLYGON"
        self.bounds = _merge_bounds(self.bounds, _geometry_bounds(feature.geometry))
        self.conn.execute(
            "INSERT INTO landslides (geom, id, area_m2, area_px, mean_confidence) VALUES (?, ?, ?, ?, ?)",
            (
                sqlite3.Binary(_gpkg_geometry_blob(feature.geometry, self.context.srs_id)),
                feature.id,
                feature.area_m2,
                feature.area_px,
                feature.mean_confidence,
            ),
        )

    def close(self) -> None:
        if self.conn is None:
            return
        try:
            _insert_gpkg_metadata(
                self.conn,
                context=self.context,
                bounds=self.bounds,
                geometry_type=self.geometry_type,
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            _safe_unlink(self.path)
            raise
        finally:
            self.conn.close()
            self.conn = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.conn is None:
            return
        if exc_type is not None:
            self.conn.rollback()
            self.conn.close()
            self.conn = None
            _safe_unlink(self.path)
            return
        self.close()


def _create_gpkg_metadata(conn: sqlite3.Connection, features: list[VectorFeature], *, context: _RasterContext) -> None:
    _create_gpkg_schema(conn, context=context)
    _insert_gpkg_metadata(
        conn,
        context=context,
        bounds=_feature_bounds(features),
        geometry_type=_geometry_type_name(features),
    )


def _create_gpkg_schema(conn: sqlite3.Connection, *, context: _RasterContext) -> None:
    conn.execute(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO gpkg_spatial_ref_sys
        (srs_name, srs_id, organization, organization_coordsys_id, definition, description)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        _srs_rows(context),
    )
    conn.execute(
        """
        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE landslides (
            fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            geom BLOB NOT NULL,
            id INTEGER NOT NULL,
            area_m2 REAL NOT NULL,
            area_px INTEGER NOT NULL,
            mean_confidence REAL
        )
        """
    )


def _insert_gpkg_metadata(
    conn: sqlite3.Connection,
    *,
    context: _RasterContext,
    bounds: tuple[float | None, float | None, float | None, float | None],
    geometry_type: str,
) -> None:
    min_x, min_y, max_x, max_y = bounds
    conn.execute(
        """
        INSERT INTO gpkg_contents
        (table_name, data_type, identifier, description, last_change, min_x, min_y, max_x, max_y, srs_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "landslides",
            "features",
            "landslides",
            "Vectorized landslide polygons from LSDetector mask.tif",
            _utc_timestamp(),
            min_x,
            min_y,
            max_x,
            max_y,
            context.srs_id,
        ),
    )
    conn.execute(
        """
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO gpkg_geometry_columns (table_name, column_name, geometry_type_name, srs_id, z, m)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("landslides", "geom", geometry_type, context.srs_id, 0, 0),
    )


def _srs_rows(context: _RasterContext) -> list[tuple[str, int, str, int, str, str]]:
    rows = [
        (
            "Undefined Cartesian SRS",
            -1,
            "NONE",
            -1,
            "undefined",
            "undefined Cartesian coordinate reference system",
        ),
        (
            "Undefined Geographic SRS",
            0,
            "NONE",
            0,
            "undefined",
            "undefined geographic coordinate reference system",
        ),
    ]
    if context.srs_id not in {-1, 0}:
        rows.append(
            (
                context.crs_string or f"EPSG:{context.srs_id}",
                context.srs_id,
                "EPSG",
                context.srs_id,
                context.crs_wkt or context.crs_string or "undefined",
                context.crs_string or "",
            )
        )
    return rows


def _gpkg_geometry_blob(geometry: dict[str, Any], srs_id: int) -> bytes:
    header = b"GP" + b"\x00" + b"\x01" + struct.pack("<i", int(srs_id))
    return header + _wkb_geometry(geometry)


def _wkb_geometry(geometry: dict[str, Any]) -> bytes:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        return _wkb_polygon(coordinates)
    if geometry_type == "MultiPolygon":
        payload = [struct.pack("<BI", 1, 6), struct.pack("<I", len(coordinates))]
        payload.extend(_wkb_polygon(polygon) for polygon in coordinates)
        return b"".join(payload)
    raise VectorizeError(f"Unsupported geometry type for GPKG output: {geometry_type!r}")


def _wkb_polygon(coordinates: Any) -> bytes:
    payload = [struct.pack("<BI", 1, 3), struct.pack("<I", len(coordinates))]
    for ring in coordinates:
        points = _closed_points(ring)
        payload.append(struct.pack("<I", len(points)))
        for x, y in points:
            payload.append(struct.pack("<2d", float(x), float(y)))
    return b"".join(payload)


def _write_geojson(path: Path, features: list[VectorFeature], *, crs_string: str | None) -> None:
    feature_collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": feature.id,
                "properties": feature.properties(),
                "geometry": feature.geometry,
            }
            for feature in features
        ],
    }
    if crs_string:
        feature_collection["crs"] = {"type": "name", "properties": {"name": crs_string}}
    path.write_text(json.dumps(feature_collection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _StreamingGeoJsonWriter:
    def __init__(self, path: Path, *, crs_string: str | None) -> None:
        self.path = path
        self.crs_string = crs_string
        self.file_obj: Any | None = None
        self.started = False

    def __enter__(self) -> "_StreamingGeoJsonWriter":
        self.file_obj = self.path.open("w", encoding="utf-8")
        self.file_obj.write('{"type":"FeatureCollection"')
        if self.crs_string:
            crs = {"type": "name", "properties": {"name": self.crs_string}}
            self.file_obj.write(',"crs":')
            self.file_obj.write(json.dumps(crs, ensure_ascii=False))
        self.file_obj.write(',"features":[')
        return self

    def write(self, feature: VectorFeature) -> None:
        assert self.file_obj is not None
        if self.started:
            self.file_obj.write(",")
        payload = {
            "type": "Feature",
            "id": feature.id,
            "properties": feature.properties(),
            "geometry": feature.geometry,
        }
        self.file_obj.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self.started = True

    def close(self) -> None:
        if self.file_obj is None:
            return
        self.file_obj.write("]}\n")
        self.file_obj.close()
        self.file_obj = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            if self.file_obj is not None:
                self.file_obj.close()
                self.file_obj = None
            _safe_unlink(self.path)
            return
        self.close()


def _write_shapefile(path: Path, features: list[VectorFeature], *, crs_wkt: str | None) -> None:
    base = path.with_suffix("")
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        _safe_unlink(base.with_suffix(suffix))

    records: list[bytes] = []
    shx_records: list[tuple[int, int]] = []
    offset_words = 50
    for record_number, feature in enumerate(features, start=1):
        content = _shapefile_polygon_content(feature.geometry)
        content_length_words = len(content) // 2
        records.append(struct.pack(">2i", record_number, content_length_words) + content)
        shx_records.append((offset_words, content_length_words))
        offset_words += 4 + content_length_words

    bbox = _feature_bounds(features)
    shp_header = _shapefile_header(offset_words, bbox)
    with base.with_suffix(".shp").open("wb") as shp:
        shp.write(shp_header)
        for record in records:
            shp.write(record)

    shx_length_words = 50 + len(shx_records) * 4
    with base.with_suffix(".shx").open("wb") as shx:
        shx.write(_shapefile_header(shx_length_words, bbox))
        for offset, content_length in shx_records:
            shx.write(struct.pack(">2i", offset, content_length))

    _write_dbf(base.with_suffix(".dbf"), features)
    base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
    if crs_wkt:
        base.with_suffix(".prj").write_text(crs_wkt + "\n", encoding="utf-8")


class _StreamingShapefileWriter:
    def __init__(self, path: Path, *, crs_wkt: str | None) -> None:
        self.path = path
        self.crs_wkt = crs_wkt
        self.base = path.with_suffix("")
        self.shp: Any | None = None
        self.dbf: Any | None = None
        self.shx_records: list[tuple[int, int]] = []
        self.offset_words = 50
        self.record_count = 0
        self.bounds: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None)
        self.fields = _dbf_fields()
        self.header_length = 32 + len(self.fields) * 32 + 1
        self.record_length = 1 + sum(field[2] for field in self.fields)

    def __enter__(self) -> "_StreamingShapefileWriter":
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            _safe_unlink(self.base.with_suffix(suffix))
        self.shp = self.base.with_suffix(".shp").open("wb")
        self.shp.write(b"\x00" * 100)
        self.dbf = self.base.with_suffix(".dbf").open("wb")
        self.dbf.write(b"\x00" * self.header_length)
        return self

    def write(self, feature: VectorFeature) -> None:
        assert self.shp is not None
        assert self.dbf is not None
        record_number = self.record_count + 1
        content = _shapefile_polygon_content(feature.geometry)
        content_length_words = len(content) // 2
        self.shp.write(struct.pack(">2i", record_number, content_length_words) + content)
        self.shx_records.append((self.offset_words, content_length_words))
        self.offset_words += 4 + content_length_words
        self.dbf.write(_dbf_record(feature))
        self.bounds = _merge_bounds(self.bounds, _geometry_bounds(feature.geometry))
        self.record_count += 1

    def close(self) -> None:
        if self.shp is None or self.dbf is None:
            return
        try:
            self.shp.seek(0)
            self.shp.write(_shapefile_header(self.offset_words, self.bounds))
            self.shp.close()
            self.shp = None

            shx_length_words = 50 + len(self.shx_records) * 4
            with self.base.with_suffix(".shx").open("wb") as shx:
                shx.write(_shapefile_header(shx_length_words, self.bounds))
                for offset, content_length in self.shx_records:
                    shx.write(struct.pack(">2i", offset, content_length))

            self.dbf.seek(0)
            self.dbf.write(_dbf_header(self.record_count, fields=self.fields))
            self.dbf.close()
            self.dbf = None
            self.base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
            if self.crs_wkt:
                self.base.with_suffix(".prj").write_text(self.crs_wkt + "\n", encoding="utf-8")
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if self.shp is not None:
            self.shp.close()
            self.shp = None
        if self.dbf is not None:
            self.dbf.close()
            self.dbf = None
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            _safe_unlink(self.base.with_suffix(suffix))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self._cleanup()
            return
        self.close()


def _shapefile_polygon_content(geometry: dict[str, Any]) -> bytes:
    parts = _geometry_to_shapefile_parts(geometry)
    points = [point for part in parts for point in part]
    min_x, min_y, max_x, max_y = _points_bounds(points)
    payload = [
        struct.pack("<i", 5),
        struct.pack("<4d", min_x, min_y, max_x, max_y),
        struct.pack("<2i", len(parts), len(points)),
    ]
    start = 0
    for part in parts:
        payload.append(struct.pack("<i", start))
        start += len(part)
    for x, y in points:
        payload.append(struct.pack("<2d", float(x), float(y)))
    return b"".join(payload)


def _geometry_to_shapefile_parts(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry_type == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise VectorizeError(f"Unsupported geometry type for Shapefile output: {geometry_type!r}")

    parts: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        for ring_index, ring in enumerate(polygon):
            points = _closed_points(ring)
            signed_area = _ring_signed_area(points)
            is_outer = ring_index == 0
            if is_outer and signed_area > 0:
                points = list(reversed(points))
            if not is_outer and signed_area < 0:
                points = list(reversed(points))
            parts.append(points)
    if not parts:
        raise VectorizeError("Cannot write empty polygon geometry to Shapefile.")
    return parts


def _shapefile_header(file_length_words: int, bbox: tuple[float | None, float | None, float | None, float | None]) -> bytes:
    min_x, min_y, max_x, max_y = (0.0 if value is None else float(value) for value in bbox)
    return b"".join(
        [
            struct.pack(">6i", 9994, 0, 0, 0, 0, 0),
            struct.pack(">i", int(file_length_words)),
            struct.pack("<2i", 1000, 5),
            struct.pack("<8d", min_x, min_y, max_x, max_y, 0.0, 0.0, 0.0, 0.0),
        ]
    )


def _write_dbf(path: Path, features: list[VectorFeature]) -> None:
    fields = _dbf_fields()
    header_length = 32 + len(fields) * 32 + 1

    with path.open("wb") as dbf:
        dbf.write(_dbf_header(len(features), fields=fields))
        for feature in features:
            dbf.write(_dbf_record(feature))
        dbf.write(b"\x1a")


def _dbf_fields() -> list[tuple[str, str, int, int]]:
    return [
        ("id", "N", 10, 0),
        ("area_m2", "N", 20, 6),
        ("area_px", "N", 12, 0),
        ("mean_conf", "N", 18, 6),
    ]


def _dbf_header(record_count: int, *, fields: list[tuple[str, str, int, int]]) -> bytes:
    header_length = 32 + len(fields) * 32 + 1
    record_length = 1 + sum(field[2] for field in fields)
    now = datetime.now()
    payload = [
        struct.pack(
            "<BBBBIHH20x",
            3,
            max(0, now.year - 1900),
            now.month,
            now.day,
            record_count,
            header_length,
            record_length,
        )
    ]
    for name, field_type, width, decimals in fields:
        name_bytes = name.encode("ascii")[:10]
        payload.append(name_bytes + b"\x00" * (11 - len(name_bytes)))
        payload.append(field_type.encode("ascii"))
        payload.append(b"\x00" * 4)
        payload.append(struct.pack("BB", width, decimals))
        payload.append(b"\x00" * 14)
    payload.append(b"\x0d")
    return b"".join(payload)


def _dbf_record(feature: VectorFeature) -> bytes:
    values = {
        "id": feature.id,
        "area_m2": feature.area_m2,
        "area_px": feature.area_px,
        "mean_conf": feature.mean_confidence,
    }
    payload = [b" "]
    for name, _, width, decimals in _dbf_fields():
        payload.append(_dbf_numeric(values[name], width=width, decimals=decimals).encode("ascii"))
    return b"".join(payload)


def _dbf_numeric(value: int | float | None, *, width: int, decimals: int) -> str:
    if value is None:
        return " " * width
    if decimals == 0:
        text = str(int(value))
    else:
        text = f"{float(value):.{decimals}f}"
    if len(text) > width:
        text = "*" * width
    return text.rjust(width)


def _normalize_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    return {"type": geometry["type"], "coordinates": _normalize_coordinates(geometry["coordinates"])}


def _normalize_coordinates(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
        return [float(value[0]), float(value[1])]
    return [_normalize_coordinates(item) for item in value]


def _simplify_geometry(geometry: dict[str, Any], tolerance: float) -> dict[str, Any]:
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        return {"type": "Polygon", "coordinates": _simplify_polygon(geometry["coordinates"], tolerance)}
    if geometry_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [_simplify_polygon(polygon, tolerance) for polygon in geometry["coordinates"]],
        }
    return geometry


def _simplify_polygon(polygon: list[Any], tolerance: float) -> list[Any]:
    simplified: list[Any] = []
    for ring_index, ring in enumerate(polygon):
        points = _closed_points(ring)
        core = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
        if len(core) < 4:
            simplified.append(ring)
            continue
        reduced = _douglas_peucker(core, tolerance)
        if len(reduced) < 3:
            if ring_index == 0:
                return polygon
            continue
        simplified.append([[x, y] for x, y in reduced + [reduced[0]]])
    return simplified or polygon


def _douglas_peucker(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_distance = -1.0
    max_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        distance = _point_line_distance(point, start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = index
    if max_distance > tolerance:
        left = _douglas_peucker(points[: max_index + 1], tolerance)
        right = _douglas_peucker(points[max_index:], tolerance)
        return left[:-1] + right
    return [start, end]


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denominator = math.hypot(dx, dy)
    if denominator == 0:
        return math.hypot(px - sx, py - sy)
    return abs(dy * px - dx * py + ex * sy - ey * sx) / denominator


def _geometry_area(geometry: dict[str, Any]) -> float:
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        return _polygon_area(geometry["coordinates"])
    if geometry_type == "MultiPolygon":
        return sum(_polygon_area(polygon) for polygon in geometry["coordinates"])
    raise VectorizeError(f"Unsupported geometry type for area calculation: {geometry_type!r}")


def _geometry_from_gpkg_blob(blob: bytes) -> dict[str, Any]:
    if len(blob) < 8 or blob[:2] != b"GP":
        raise VectorizeError("Invalid GPKG geometry blob.")
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb_offset = 8 + envelope_sizes.get(envelope_code, 0)
    return _geometry_from_wkb(blob[wkb_offset:])


def _geometry_from_wkb(data: bytes) -> dict[str, Any]:
    geometry, offset = _parse_wkb_geometry(data, 0)
    if offset > len(data):
        raise VectorizeError("Invalid WKB geometry length.")
    return geometry


def _parse_wkb_geometry(data: bytes, offset: int) -> tuple[dict[str, Any], int]:
    if offset + 5 > len(data):
        raise VectorizeError("Truncated WKB geometry.")
    byte_order = data[offset]
    endian = "<" if byte_order == 1 else ">"
    geometry_type = struct.unpack(endian + "I", data[offset + 1 : offset + 5])[0]
    offset += 5
    if geometry_type == 3:
        return _parse_wkb_polygon(data, offset, endian)
    if geometry_type == 6:
        if offset + 4 > len(data):
            raise VectorizeError("Truncated WKB MultiPolygon.")
        polygon_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
        offset += 4
        polygons = []
        for _ in range(polygon_count):
            polygon, offset = _parse_wkb_geometry(data, offset)
            if polygon["type"] != "Polygon":
                raise VectorizeError("Invalid WKB MultiPolygon member.")
            polygons.append(polygon["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polygons}, offset
    raise VectorizeError(f"Unsupported WKB geometry type {geometry_type}; expected Polygon or MultiPolygon.")


def _parse_wkb_polygon(data: bytes, offset: int, endian: str) -> tuple[dict[str, Any], int]:
    if offset + 4 > len(data):
        raise VectorizeError("Truncated WKB Polygon.")
    ring_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
    offset += 4
    rings = []
    for _ in range(ring_count):
        if offset + 4 > len(data):
            raise VectorizeError("Truncated WKB LinearRing.")
        point_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
        offset += 4
        points = []
        for _ in range(point_count):
            if offset + 16 > len(data):
                raise VectorizeError("Truncated WKB point.")
            x, y = struct.unpack(endian + "2d", data[offset : offset + 16])
            offset += 16
            points.append([float(x), float(y)])
        rings.append(points)
    return {"type": "Polygon", "coordinates": rings}, offset


def _polygon_area(polygon: list[Any]) -> float:
    if not polygon:
        return 0.0
    area = abs(_ring_signed_area(_closed_points(polygon[0])))
    for hole in polygon[1:]:
        area -= abs(_ring_signed_area(_closed_points(hole)))
    return area


def _ring_signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 4:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _closed_points(ring: Any) -> list[tuple[float, float]]:
    points = [(float(point[0]), float(point[1])) for point in ring]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _feature_bounds(features: list[VectorFeature]) -> tuple[float | None, float | None, float | None, float | None]:
    points: list[tuple[float, float]] = []
    for feature in features:
        points.extend(_geometry_points(feature.geometry))
    if not points:
        return (None, None, None, None)
    return _points_bounds(points)


def _geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    points = _geometry_points(geometry)
    if not points:
        raise VectorizeError("Cannot determine bounds for empty geometry.")
    return _points_bounds(points)


def _merge_bounds(
    current: tuple[float | None, float | None, float | None, float | None],
    update: tuple[float, float, float, float],
) -> tuple[float | None, float | None, float | None, float | None]:
    min_x, min_y, max_x, max_y = current
    next_min_x, next_min_y, next_max_x, next_max_y = update
    return (
        next_min_x if min_x is None else min(min_x, next_min_x),
        next_min_y if min_y is None else min(min_y, next_min_y),
        next_max_x if max_x is None else max(max_x, next_max_x),
        next_max_y if max_y is None else max(max_y, next_max_y),
    )


def _points_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _geometry_points(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry_type == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return []
    return [point for polygon in polygons for ring in polygon for point in _closed_points(ring)]


def _geometry_type_name(features: list[VectorFeature]) -> str:
    types = {feature.geometry.get("type") for feature in features}
    if "MultiPolygon" in types:
        return "MULTIPOLYGON"
    return "POLYGON"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _result_from_stats(
    *,
    gpkg_path: Path,
    geojson_path: Path | None,
    shapefile_path: Path | None,
    stats_csv_path: Path,
    stats_json_path: Path,
    stats: VectorStats,
) -> VectorizeResult:
    return VectorizeResult(
        gpkg_path=gpkg_path,
        geojson_path=geojson_path,
        shapefile_path=shapefile_path,
        stats_csv_path=stats_csv_path,
        stats_json_path=stats_json_path,
        polygon_count=stats.polygon_count,
        total_area_m2=stats.total_area_m2,
        mean_area_m2=stats.mean_area_m2,
        mean_confidence=stats.mean_confidence,
    )


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
