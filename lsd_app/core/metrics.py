"""Evaluation metrics for landslide prediction masks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import struct
from typing import Any

from lsd_app.config import PlatformConfig
from lsd_app.core.raster import require_rasterio
from lsd_app.core.vectorize import (
    VectorFeature,
    _extract_features,
    _raster_context,
    _write_geopackage,
)


class EvaluationError(RuntimeError):
    """Raised when prediction and truth masks cannot be evaluated."""


@dataclass(frozen=True)
class MaskGrid:
    """Raster grid metadata used for mask alignment and rasterization."""

    width: int
    height: int
    transform: Any
    crs: Any


@dataclass(frozen=True)
class EvaluationScores:
    """Binary segmentation metrics and confusion matrix values."""

    iou: float
    miou: float
    precision: float
    recall: float
    f1: float
    kappa: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    predicted_positive: int
    truth_positive: int
    total_pixels: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metrics mapping."""

        return {
            "iou": self.iou,
            "miou": self.miou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "kappa": self.kappa,
            "confusion_matrix": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
                "true_negative": self.true_negative,
            },
            "predicted_positive": self.predicted_positive,
            "truth_positive": self.truth_positive,
            "total_pixels": self.total_pixels,
        }

    def to_csv_row(self) -> dict[str, int | float]:
        """Return a flat row for CSV output."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Summary returned by the evaluate stage."""

    skipped: bool
    message: str
    evaluation_json_path: Path | None = None
    evaluation_csv_path: Path | None = None
    tp_vector_path: Path | None = None
    fp_vector_path: Path | None = None
    fn_vector_path: Path | None = None
    scores: EvaluationScores | None = None
    truth_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result mapping."""

        return {
            "skipped": self.skipped,
            "message": self.message,
            "evaluation_json_path": str(self.evaluation_json_path) if self.evaluation_json_path else None,
            "evaluation_csv_path": str(self.evaluation_csv_path) if self.evaluation_csv_path else None,
            "tp_vector_path": str(self.tp_vector_path) if self.tp_vector_path else None,
            "fp_vector_path": str(self.fp_vector_path) if self.fp_vector_path else None,
            "fn_vector_path": str(self.fn_vector_path) if self.fn_vector_path else None,
            "scores": self.scores.to_dict() if self.scores else None,
            "truth_source": self.truth_source,
        }


def run_evaluate(config: PlatformConfig) -> EvaluationResult:
    """Evaluate `rasters/mask.tif` against configured truth data."""

    truth_path = configured_truth_path(config)
    if truth_path is None:
        return EvaluationResult(
            skipped=True,
            message="No truth data configured; evaluate stage skipped.",
        )

    prediction_mask, valid_mask, grid, context = load_prediction_mask(config)
    truth_mask, truth_valid, truth_source = load_truth_mask(config, grid=grid)
    valid = valid_mask & truth_valid

    scores = compute_scores(prediction_mask & valid, truth_mask & valid, valid)
    config.outputs.metrics.mkdir(parents=True, exist_ok=True)
    config.outputs.vectors.mkdir(parents=True, exist_ok=True)
    evaluation_json_path = config.outputs.metrics / "evaluation.json"
    evaluation_csv_path = config.outputs.metrics / "evaluation.csv"
    tp_vector_path = config.outputs.vectors / "TP.gpkg"
    fp_vector_path = config.outputs.vectors / "FP.gpkg"
    fn_vector_path = config.outputs.vectors / "FN.gpkg"

    write_evaluation_metrics(scores, evaluation_json_path, evaluation_csv_path, truth_source=truth_source)
    _write_classification_vectors(
        tp_vector_path=tp_vector_path,
        fp_vector_path=fp_vector_path,
        fn_vector_path=fn_vector_path,
        prediction_mask=prediction_mask,
        truth_mask=truth_mask,
        valid_mask=valid,
        context=context,
    )

    return EvaluationResult(
        skipped=False,
        message="Evaluation complete.",
        evaluation_json_path=evaluation_json_path,
        evaluation_csv_path=evaluation_csv_path,
        tp_vector_path=tp_vector_path,
        fp_vector_path=fp_vector_path,
        fn_vector_path=fn_vector_path,
        scores=scores,
        truth_source=truth_source,
    )


def configured_truth_path(config: PlatformConfig) -> Path | None:
    """Return the configured truth path, preferring raster masks over vectors."""

    if config.truth_mask_path is not None:
        return config.truth_mask_path
    return config.truth_vector_path


def load_prediction_mask(config: PlatformConfig) -> tuple[Any, Any, MaskGrid, Any]:
    """Load the pipeline prediction mask and its valid data mask."""

    rasterio = require_rasterio()
    import numpy as np

    mask_path = config.outputs.rasters / "mask.tif"
    if not mask_path.exists():
        raise EvaluationError(f"Prediction mask does not exist: {mask_path}. Run `lsd mosaic --config ...` first.")

    try:
        with rasterio.open(mask_path) as src:
            if src.count < 1:
                raise EvaluationError(f"Prediction mask has no bands: {mask_path}")
            values = src.read(1)
            prediction_mask = values > 0
            valid_mask = _valid_mask_from_dataset(src)
            grid = MaskGrid(width=int(src.width), height=int(src.height), transform=src.transform, crs=src.crs)
            context = _raster_context(src)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Failed to read prediction mask {mask_path}: {exc}") from exc

    return prediction_mask.astype(bool, copy=False), valid_mask.astype(bool, copy=False), grid, context


def load_truth_mask(config: PlatformConfig, *, grid: MaskGrid) -> tuple[Any, Any, str]:
    """Load truth from raster or vector config and align it to the prediction grid."""

    if config.truth_mask_path is not None:
        return load_truth_raster(config.truth_mask_path, grid=grid)
    if config.truth_vector_path is not None:
        return load_truth_vector(config.truth_vector_path, grid=grid)
    raise EvaluationError("No truth data configured.")


def load_truth_raster(path: Path, *, grid: MaskGrid) -> tuple[Any, Any, str]:
    """Load a truth mask raster, reprojecting with nearest-neighbor when needed."""

    rasterio = require_rasterio()
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    import numpy as np

    if not path.exists():
        raise EvaluationError(f"Truth raster mask does not exist: {path}")

    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise EvaluationError(f"Truth raster mask has no bands: {path}")

            same_shape = src.width == grid.width and src.height == grid.height
            same_transform = src.transform.almost_equals(grid.transform)
            same_crs = src.crs == grid.crs
            if same_shape and (same_transform or _is_identity_like(src.transform)) and (same_crs or src.crs is None):
                truth_values = src.read(1)
                valid = _valid_mask_from_dataset(src)
            elif src.crs is None or grid.crs is None:
                raise EvaluationError(
                    f"Truth raster grid does not match prediction grid and CRS is missing: {path}"
                )
            else:
                truth_values = np.zeros((grid.height, grid.width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=truth_values,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    resampling=Resampling.nearest,
                )
                valid_source = _valid_mask_from_dataset(src).astype(np.uint8)
                valid = np.zeros((grid.height, grid.width), dtype=np.uint8)
                reproject(
                    source=valid_source,
                    destination=valid,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    resampling=Resampling.nearest,
                )
                valid = valid > 0
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Failed to read truth raster mask {path}: {exc}") from exc

    return truth_values > 0, valid.astype(bool, copy=False), str(path)


def load_truth_vector(path: Path, *, grid: MaskGrid) -> tuple[Any, Any, str]:
    """Rasterize vector truth polygons onto the prediction grid."""

    if not path.exists():
        raise EvaluationError(f"Truth vector does not exist: {path}")

    geometries = read_vector_geometries(path)
    truth_mask = rasterize_geometries(geometries, grid=grid)
    import numpy as np

    valid = np.ones((grid.height, grid.width), dtype=bool)
    return truth_mask, valid, str(path)


def read_vector_geometries(path: Path) -> list[dict[str, Any]]:
    """Read polygon geometries from GeoJSON, GPKG, or Shapefile without heavy GIS deps."""

    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return _read_geojson_geometries(path)
    if suffix == ".gpkg":
        return _read_gpkg_geometries(path)
    if suffix == ".shp":
        return _read_shapefile_geometries(path)
    raise EvaluationError(f"Unsupported truth vector format '{path.suffix}'. Use .shp, .geojson, or .gpkg.")


def rasterize_geometries(geometries: list[dict[str, Any]], *, grid: MaskGrid) -> Any:
    """Rasterize polygon geometries to a boolean mask."""

    from rasterio import features as raster_features
    import numpy as np

    if not geometries:
        return np.zeros((grid.height, grid.width), dtype=bool)
    values = raster_features.rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    )
    return values.astype(bool, copy=False)


def compute_scores(prediction_mask: Any, truth_mask: Any, valid_mask: Any) -> EvaluationScores:
    """Compute binary segmentation metrics from aligned boolean masks."""

    import numpy as np

    pred = np.asarray(prediction_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    truth = np.asarray(truth_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)

    tp = int(np.count_nonzero(pred & truth & valid))
    fp = int(np.count_nonzero(pred & ~truth & valid))
    fn = int(np.count_nonzero(~pred & truth & valid))
    tn = int(np.count_nonzero(~pred & ~truth & valid))
    total = tp + fp + fn + tn

    iou = _ratio(tp, tp + fp + fn)
    background_iou = _ratio(tn, tn + fp + fn)
    miou = (iou + background_iou) / 2.0
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    kappa = _kappa(tp=tp, fp=fp, fn=fn, tn=tn)

    return EvaluationScores(
        iou=iou,
        miou=miou,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        predicted_positive=tp + fp,
        truth_positive=tp + fn,
        total_pixels=total,
    )


def write_evaluation_metrics(
    scores: EvaluationScores,
    json_path: Path,
    csv_path: Path,
    *,
    truth_source: str,
) -> None:
    """Write evaluation metrics to JSON and CSV."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = scores.to_dict()
    payload["truth_source"] = truth_source
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    row = scores.to_csv_row()
    row["truth_source"] = truth_source
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_classification_vectors(
    *,
    tp_vector_path: Path,
    fp_vector_path: Path,
    fn_vector_path: Path,
    prediction_mask: Any,
    truth_mask: Any,
    valid_mask: Any,
    context: Any,
) -> None:
    import numpy as np

    masks = {
        tp_vector_path: np.asarray(prediction_mask, dtype=bool) & np.asarray(truth_mask, dtype=bool) & valid_mask,
        fp_vector_path: np.asarray(prediction_mask, dtype=bool) & ~np.asarray(truth_mask, dtype=bool) & valid_mask,
        fn_vector_path: ~np.asarray(prediction_mask, dtype=bool) & np.asarray(truth_mask, dtype=bool) & valid_mask,
    }
    for path, mask in masks.items():
        features = _features_from_mask(mask, context=context)
        _write_geopackage(path, features, context=context)


def _features_from_mask(mask: Any, *, context: Any) -> list[VectorFeature]:
    return _extract_features(
        positive_mask=mask,
        transform=context.transform,
        pixel_area_units=context.pixel_area_units,
        pixel_area_m2=context.pixel_area_m2,
        probability_values=None,
        probability_valid=None,
        min_area_pixels=0,
        simplify_tolerance=0.0,
        connectivity=8,
    )


def _valid_mask_from_dataset(src: Any) -> Any:
    """Use explicit dataset/alpha masks, but do not treat class-0 nodata as invalid."""

    import numpy as np

    flags = src.mask_flag_enums[0] if src.mask_flag_enums else []
    flag_names = {getattr(flag, "name", str(flag)).lower() for flag in flags}
    if "per_dataset" in flag_names or "alpha" in flag_names:
        return src.dataset_mask() > 0
    return np.ones((src.height, src.width), dtype=bool)


def _is_identity_like(transform: Any) -> bool:
    return (
        abs(float(transform.a) - 1.0) < 1e-9
        and abs(float(transform.b)) < 1e-9
        and abs(float(transform.c)) < 1e-9
        and abs(float(transform.d)) < 1e-9
        and abs(float(transform.e) - 1.0) < 1e-9
        and abs(float(transform.f)) < 1e-9
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def _kappa(*, tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    if total == 0:
        return 1.0
    observed = (tp + tn) / total
    predicted_positive = tp + fp
    predicted_negative = fn + tn
    truth_positive = tp + fn
    truth_negative = fp + tn
    expected = ((predicted_positive * truth_positive) + (predicted_negative * truth_negative)) / (total * total)
    denominator = 1.0 - expected
    if denominator == 0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / denominator


def _read_geojson_geometries(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid GeoJSON truth vector {path}: {exc}") from exc

    geometries: list[dict[str, Any]] = []
    if loaded.get("type") == "FeatureCollection":
        for feature in loaded.get("features", []):
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if geometry:
                geometries.extend(_flatten_geometry_collection(geometry))
    elif loaded.get("type") == "Feature":
        geometry = loaded.get("geometry")
        if geometry:
            geometries.extend(_flatten_geometry_collection(geometry))
    else:
        geometries.extend(_flatten_geometry_collection(loaded))
    return geometries


def _read_gpkg_geometries(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute("SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1").fetchone()
            if row is None:
                raise EvaluationError(f"GPKG truth vector has no geometry columns: {path}")
            table_name, column_name = row
            rows = conn.execute(f'SELECT "{column_name}" FROM "{table_name}"').fetchall()
    except EvaluationError:
        raise
    except sqlite3.Error as exc:
        raise EvaluationError(f"Failed to read GPKG truth vector {path}: {exc}") from exc

    return [_geometry_from_gpkg_blob(bytes(row[0])) for row in rows if row[0] is not None]


def _read_shapefile_geometries(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Failed to read Shapefile truth vector {path}: {exc}") from exc
    if len(data) < 100:
        raise EvaluationError(f"Invalid Shapefile truth vector: {path}")

    geometries: list[dict[str, Any]] = []
    offset = 100
    while offset + 8 <= len(data):
        _, content_length_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_length = content_length_words * 2
        content = data[offset : offset + content_length]
        offset += content_length
        if len(content) < 4:
            continue
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            continue
        if shape_type != 5:
            raise EvaluationError(f"Unsupported Shapefile geometry type {shape_type} in {path}; expected Polygon.")
        geometries.append(_geometry_from_shp_polygon_content(content))
    return geometries


def _geometry_from_gpkg_blob(blob: bytes) -> dict[str, Any]:
    if len(blob) < 8 or blob[:2] != b"GP":
        raise EvaluationError("Invalid GPKG geometry blob.")
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb_offset = 8 + envelope_sizes.get(envelope_code, 0)
    return _geometry_from_wkb(blob[wkb_offset:])


def _geometry_from_wkb(data: bytes) -> dict[str, Any]:
    geometry, offset = _parse_wkb_geometry(data, 0)
    if offset > len(data):
        raise EvaluationError("Invalid WKB geometry length.")
    return geometry


def _parse_wkb_geometry(data: bytes, offset: int) -> tuple[dict[str, Any], int]:
    if offset + 5 > len(data):
        raise EvaluationError("Truncated WKB geometry.")
    byte_order = data[offset]
    endian = "<" if byte_order == 1 else ">"
    geometry_type = struct.unpack(endian + "I", data[offset + 1 : offset + 5])[0]
    offset += 5
    if geometry_type == 3:
        return _parse_wkb_polygon(data, offset, endian)
    if geometry_type == 6:
        if offset + 4 > len(data):
            raise EvaluationError("Truncated WKB MultiPolygon.")
        polygon_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
        offset += 4
        polygons = []
        for _ in range(polygon_count):
            polygon, offset = _parse_wkb_geometry(data, offset)
            if polygon["type"] != "Polygon":
                raise EvaluationError("Invalid WKB MultiPolygon member.")
            polygons.append(polygon["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polygons}, offset
    raise EvaluationError(f"Unsupported WKB geometry type {geometry_type}; expected Polygon or MultiPolygon.")


def _parse_wkb_polygon(data: bytes, offset: int, endian: str) -> tuple[dict[str, Any], int]:
    if offset + 4 > len(data):
        raise EvaluationError("Truncated WKB Polygon.")
    ring_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
    offset += 4
    rings = []
    for _ in range(ring_count):
        if offset + 4 > len(data):
            raise EvaluationError("Truncated WKB LinearRing.")
        point_count = struct.unpack(endian + "I", data[offset : offset + 4])[0]
        offset += 4
        points = []
        for _ in range(point_count):
            if offset + 16 > len(data):
                raise EvaluationError("Truncated WKB point.")
            x, y = struct.unpack(endian + "2d", data[offset : offset + 16])
            offset += 16
            points.append([float(x), float(y)])
        rings.append(points)
    return {"type": "Polygon", "coordinates": rings}, offset


def _geometry_from_shp_polygon_content(content: bytes) -> dict[str, Any]:
    if len(content) < 44:
        raise EvaluationError("Truncated Shapefile Polygon record.")
    num_parts, num_points = struct.unpack("<2i", content[36:44])
    parts_offset = 44
    points_offset = parts_offset + num_parts * 4
    if len(content) < points_offset + num_points * 16:
        raise EvaluationError("Truncated Shapefile Polygon points.")
    part_starts = list(struct.unpack("<" + "i" * num_parts, content[parts_offset:points_offset]))
    part_starts.append(num_points)
    points = [
        list(struct.unpack("<2d", content[points_offset + index * 16 : points_offset + (index + 1) * 16]))
        for index in range(num_points)
    ]
    rings = [points[part_starts[index] : part_starts[index + 1]] for index in range(num_parts)]
    return {"type": "Polygon", "coordinates": rings}


def _flatten_geometry_collection(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    geometry_type = geometry.get("type")
    if geometry_type in {"Polygon", "MultiPolygon"}:
        return [geometry]
    if geometry_type == "GeometryCollection":
        geometries: list[dict[str, Any]] = []
        for child in geometry.get("geometries", []):
            if isinstance(child, dict):
                geometries.extend(_flatten_geometry_collection(child))
        return geometries
    raise EvaluationError(f"Unsupported truth geometry type {geometry_type!r}; expected Polygon or MultiPolygon.")
