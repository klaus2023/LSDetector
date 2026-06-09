#!/usr/bin/env python3
"""Create the Wuping landslide mapping result layout.

This script is intended to run on the WSL host where the source rasters and
shapefiles live. It keeps source GIS data read-only and writes versioned output
figures plus a small QA JSON summary.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Polygon as MplPolygon, Rectangle
from matplotlib.transforms import Affine2D
from osgeo import gdal, ogr, osr


gdal.UseExceptions()
ogr.UseExceptions()


WUPING_DIR = Path("/home/fuzijin/data/Wuping")
REFINEMENT_1 = Path("/home/fuzijin/data/Wuping/TSFT_AOI/test2.shp")
REFINEMENT_2 = Path("/home/fuzijin/data/Wuping/TSFT_AOI/wuping.shp")
LANDSLIDES = Path(
    "/home/fuzijin/data/LSD__APP/outputs/wuping_refine2_deploy/merged/landslides_merged.shp"
)
SHOW_AOI = Path("/home/fuzijin/data/Wuping/TSFT_AOI/show_AOI/all.shp")
OUT_DIR = Path("/home/fuzijin/data/LSD__APP/outputs/wuping_landslide_map")
VERSION = "v14"
RGB_BANDS = (6, 4, 2)
ARCGIS_MIN_MAX_CLIP_PERCENT = 0.25
ARCGIS_GAMMA = 2.3
LANDSLIDE_COLOR = "#ffe600"
SHOW_AOI_COLOR = "#ff1f1f"
FIG_WIDTH_IN = 14.6
FIG_HEIGHT_IN = 10.4
MAIN_AX = [0.055, 0.31, 0.73, 0.64]
RIGHT_COLUMN_ORDER = (1, 2, 0, 3)
CUTOUT_PAD_FRACTION = 0.0
CUTOUT_OUT_W = 640
CUTOUT_OUT_H = 480


@dataclass(frozen=True)
class Bounds:
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def tuple(self) -> tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy

    def intersects(self, other: "Bounds") -> bool:
        return not (
            self.maxx <= other.minx
            or self.minx >= other.maxx
            or self.maxy <= other.miny
            or self.miny >= other.maxy
        )

    def intersection(self, other: "Bounds") -> "Bounds | None":
        if not self.intersects(other):
            return None
        return Bounds(
            max(self.minx, other.minx),
            max(self.miny, other.miny),
            min(self.maxx, other.maxx),
            min(self.maxy, other.maxy),
        )

    def pad_to_aspect(self, target_aspect: float, pad_fraction: float = 0.02) -> "Bounds":
        minx = self.minx - self.width * pad_fraction
        maxx = self.maxx + self.width * pad_fraction
        miny = self.miny - self.height * pad_fraction
        maxy = self.maxy + self.height * pad_fraction
        b = Bounds(minx, miny, maxx, maxy)
        current_aspect = b.width / b.height
        if current_aspect < target_aspect:
            desired_width = b.height * target_aspect
            dx = (desired_width - b.width) / 2
            return Bounds(b.minx - dx, b.miny, b.maxx + dx, b.maxy)
        desired_height = b.width / target_aspect
        dy = (desired_height - b.height) / 2
        return Bounds(b.minx, b.miny - dy, b.maxx, b.maxy + dy)


@dataclass(frozen=True)
class AoiFeature:
    fid: int
    ring: list[tuple[float, float]]
    p0: np.ndarray
    p1: np.ndarray
    p3: np.ndarray
    center: tuple[float, float]
    bounds: Bounds
    placement: str


@dataclass(frozen=True)
class LocalBounds:
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def tuple(self) -> tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy


def raster_bounds(ds: gdal.Dataset) -> Bounds:
    gt = ds.GetGeoTransform()
    corners = [
        (0, 0),
        (ds.RasterXSize, 0),
        (ds.RasterXSize, ds.RasterYSize),
        (0, ds.RasterYSize),
    ]
    xs = [gt[0] + px * gt[1] + py * gt[2] for px, py in corners]
    ys = [gt[3] + px * gt[4] + py * gt[5] for px, py in corners]
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def raster_paths() -> list[Path]:
    paths = sorted(WUPING_DIR.glob("tile_*/tile_*_mosaic.tif"))
    return [
        path
        for path in paths
        if not any(token in path.name.lower() for token in ("date", "qa", "mask", "provenance"))
    ]


def union_bounds(bounds: Iterable[Bounds]) -> Bounds:
    bounds = list(bounds)
    return Bounds(
        min(b.minx for b in bounds),
        min(b.miny for b in bounds),
        max(b.maxx for b in bounds),
        max(b.maxy for b in bounds),
    )


def projection_name(wkt: str) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    auth_name = srs.GetAuthorityName(None)
    auth_code = srs.GetAuthorityCode(None)
    if auth_name and auth_code:
        return f"{auth_name}:{auth_code}"
    return srs.GetName() or "unknown"


def spatial_ref(epsg: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def coordinate_transform(src_epsg: int, dst_epsg: int) -> osr.CoordinateTransformation:
    return osr.CoordinateTransformation(spatial_ref(src_epsg), spatial_ref(dst_epsg))


TO_GEO = coordinate_transform(32650, 4326)
TO_UTM = coordinate_transform(4326, 32650)


def world_to_pixel(gt: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    inv = gdal.InvGeoTransform(gt)
    px = inv[0] + inv[1] * x + inv[2] * y
    py = inv[3] + inv[4] * x + inv[5] * y
    return px, py


def transform_xy(transform: osr.CoordinateTransformation, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_x = np.empty_like(xs, dtype="float64")
    out_y = np.empty_like(ys, dtype="float64")
    for i, (x, y) in enumerate(zip(xs, ys)):
        tx, ty, _ = transform.TransformPoint(float(x), float(y))
        out_x[i] = tx
        out_y[i] = ty
    return out_x, out_y


def natural_color_band_indices(ds: gdal.Dataset) -> tuple[int, int, int]:
    if ds.RasterCount < max(RGB_BANDS):
        raise RuntimeError(f"Raster has {ds.RasterCount} bands, cannot use RGB bands {RGB_BANDS}")
    return RGB_BANDS


def arcgis_rgb_stretch(bands: list[np.ndarray], nodata_mask: np.ndarray) -> np.ndarray:
    stretched = []
    for band in bands:
        values = band[~nodata_mask]
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if values.size < 10:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(
                values,
                [ARCGIS_MIN_MAX_CLIP_PERCENT, 100 - ARCGIS_MIN_MAX_CLIP_PERCENT],
            )
            if hi <= lo:
                lo, hi = float(values.min()), float(values.max())
        arr = np.clip((band.astype("float32") - lo) / (hi - lo + 1e-6), 0, 1)
        # ArcGIS Pro gamma > 1 brightens mid-tones for raster display.
        arr = np.power(arr, 1.0 / ARCGIS_GAMMA)
        stretched.append(arr)
    rgb = np.dstack(stretched)
    rgb[nodata_mask] = np.nan
    return rgb


def read_rgb_for_extent(path: Path, map_bounds: Bounds, max_pixels: int = 2_200_000) -> tuple[np.ndarray, Bounds] | None:
    ds = gdal.Open(str(path))
    rb = raster_bounds(ds)
    inter = rb.intersection(map_bounds)
    if inter is None:
        return None

    gt = ds.GetGeoTransform()
    px0, py0 = world_to_pixel(gt, inter.minx, inter.maxy)
    px1, py1 = world_to_pixel(gt, inter.maxx, inter.miny)
    xoff = max(0, int(math.floor(min(px0, px1))))
    yoff = max(0, int(math.floor(min(py0, py1))))
    xend = min(ds.RasterXSize, int(math.ceil(max(px0, px1))))
    yend = min(ds.RasterYSize, int(math.ceil(max(py0, py1))))
    xsize = max(1, xend - xoff)
    ysize = max(1, yend - yoff)

    scale = min(1.0, math.sqrt(max_pixels / (xsize * ysize)))
    out_x = max(1, int(round(xsize * scale)))
    out_y = max(1, int(round(ysize * scale)))

    r_i, g_i, b_i = natural_color_band_indices(ds)
    band_arrays = [
        ds.GetRasterBand(i).ReadAsArray(xoff, yoff, xsize, ysize, out_x, out_y).astype("float32")
        for i in (r_i, g_i, b_i)
    ]
    nodata_mask = np.zeros((out_y, out_x), dtype=bool)
    for i, arr in zip((r_i, g_i, b_i), band_arrays):
        nodata = ds.GetRasterBand(i).GetNoDataValue()
        if nodata is not None:
            nodata_mask |= arr == nodata
    nodata_mask |= np.all(np.stack(band_arrays, axis=0) <= 0, axis=0)
    rgb = arcgis_rgb_stretch(band_arrays, nodata_mask)

    win_gt = list(gt)
    win_gt[0] = gt[0] + xoff * gt[1] + yoff * gt[2]
    win_gt[3] = gt[3] + xoff * gt[4] + yoff * gt[5]
    actual = Bounds(
        win_gt[0],
        win_gt[3] + ysize * gt[5],
        win_gt[0] + xsize * gt[1],
        win_gt[3],
    )
    return rgb, actual


def layer_info(path: Path) -> dict:
    ds = ogr.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open vector: {path}")
    layer = ds.GetLayer(0)
    srs = layer.GetSpatialRef()
    invalid = 0
    empty = 0
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            empty += 1
        elif not geom_is_valid(geom):
            invalid += 1
    extent = layer.GetExtent()
    return {
        "path": str(path),
        "feature_count": layer.GetFeatureCount(),
        "empty_count": empty,
        "invalid_count": invalid,
        "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
        "extent": [extent[0], extent[2], extent[1], extent[3]],
        "crs": projection_name(srs.ExportToWkt()) if srs else "unknown",
    }


def iter_polygon_exteriors(path: Path) -> Iterable[list[tuple[float, float]]]:
    ds = ogr.Open(str(path))
    layer = ds.GetLayer(0)
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        if not geom_is_valid(geom):
            geom = geom_make_valid(geom)
        for ring in polygon_exterior_rings(geom):
            if len(ring) >= 3:
                yield ring


def polygon_exterior_rings(geom: ogr.Geometry) -> Iterable[list[tuple[float, float]]]:
    geom_type = geom.GetGeometryType()
    flat_type = ogr.GT_Flatten(geom_type)
    if flat_type == ogr.wkbPolygon:
        ring = geom.GetGeometryRef(0)
        if ring is not None:
            yield [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]
    elif flat_type == ogr.wkbMultiPolygon or flat_type == ogr.wkbGeometryCollection:
        for i in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(i)
            if child is not None:
                yield from polygon_exterior_rings(child)


def geometry_line_rings(geom: ogr.Geometry) -> Iterable[list[tuple[float, float]]]:
    geom_type = geom.GetGeometryType()
    flat_type = ogr.GT_Flatten(geom_type)
    if flat_type == ogr.wkbLineString or flat_type == ogr.wkbLinearRing:
        if geom.GetPointCount() >= 2:
            yield [(geom.GetX(i), geom.GetY(i)) for i in range(geom.GetPointCount())]
    elif flat_type == ogr.wkbPolygon:
        ring = geom.GetGeometryRef(0)
        if ring is not None and ring.GetPointCount() >= 2:
            yield [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]
    elif flat_type in (ogr.wkbMultiLineString, ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        for i in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(i)
            if child is not None:
                yield from geometry_line_rings(child)


def geom_is_valid(geom: ogr.Geometry) -> bool:
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        return bool(geom.IsValid())
    finally:
        gdal.PopErrorHandler()


def geom_make_valid(geom: ogr.Geometry) -> ogr.Geometry:
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        return geom.MakeValid()
    finally:
        gdal.PopErrorHandler()


def vector_union_geometry(path: Path) -> ogr.Geometry:
    ds = ogr.Open(str(path))
    layer = ds.GetLayer(0)
    union = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        if not geom_is_valid(geom):
            geom = geom_make_valid(geom)
        if union is None:
            union = geom
        else:
            union = union.Union(geom)
    if union is None:
        raise RuntimeError(f"No non-empty geometry in {path}")
    return union


def centroid_points(path: Path) -> tuple[np.ndarray, dict]:
    ds = ogr.Open(str(path))
    layer = ds.GetLayer(0)
    xs: list[float] = []
    ys: list[float] = []
    invalid = 0
    repaired = 0
    envelope_fallback = 0
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        if not geom_is_valid(geom):
            invalid += 1
            try:
                geom = geom_make_valid(geom)
                repaired += 1
            except Exception:
                pass
        try:
            point = geom.Centroid()
        except Exception:
            point = None
        if point is None or point.IsEmpty():
            minx, maxx, miny, maxy = geom.GetEnvelope()
            xs.append((minx + maxx) / 2)
            ys.append((miny + maxy) / 2)
            envelope_fallback += 1
        else:
            xs.append(point.GetX())
            ys.append(point.GetY())
    return np.column_stack([xs, ys]), {
        "centroid_points": len(xs),
        "invalid_polygon_count": invalid,
        "make_valid_attempts": repaired,
        "envelope_fallback_count": envelope_fallback,
    }


def ogr_polygon_from_ring(ring: list[tuple[float, float]]) -> ogr.Geometry:
    ogr_ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ring:
        ogr_ring.AddPoint(float(x), float(y))
    if ring[0] != ring[-1]:
        ogr_ring.AddPoint(float(ring[0][0]), float(ring[0][1]))
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ogr_ring)
    return polygon


def ogr_polygon_from_bounds(bounds: Bounds) -> ogr.Geometry:
    return ogr_polygon_from_ring(
        [
            (bounds.minx, bounds.miny),
            (bounds.maxx, bounds.miny),
            (bounds.maxx, bounds.maxy),
            (bounds.minx, bounds.maxy),
            (bounds.minx, bounds.miny),
        ]
    )


def read_show_aois(map_bounds: Bounds) -> list[AoiFeature]:
    ds = ogr.Open(str(SHOW_AOI))
    if ds is None:
        raise RuntimeError(f"Cannot open show AOI layer: {SHOW_AOI}")
    layer = ds.GetLayer(0)
    aois: list[AoiFeature] = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        if not geom_is_valid(geom):
            geom = geom_make_valid(geom)
        rings = list(polygon_exterior_rings(geom))
        if not rings:
            continue
        ring = rings[0]
        corners = np.array(ring[:-1] if ring[0] == ring[-1] else ring, dtype="float64")
        if len(corners) < 4:
            continue
        corners = corners[:4]
        env = geom.GetEnvelope()
        bounds = Bounds(env[0], env[2], env[1], env[3])
        center = tuple(corners.mean(axis=0))
        bottom_dist = bounds.miny - map_bounds.miny
        right_dist = map_bounds.maxx - bounds.maxx
        placement = "bottom" if bottom_dist <= right_dist else "right"
        if feature.GetFID() in RIGHT_COLUMN_ORDER:
            placement = "right"
        closed_ring = [(float(x), float(y)) for x, y in corners] + [(float(corners[0, 0]), float(corners[0, 1]))]
        aois.append(
            AoiFeature(
                fid=feature.GetFID(),
                ring=closed_ring,
                p0=corners[0],
                p1=corners[1],
                p3=corners[3],
                center=(float(center[0]), float(center[1])),
                bounds=bounds,
                placement=placement,
            )
        )
    aois.sort(key=lambda item: item.fid)
    if len(aois) != 8:
        raise RuntimeError(f"Expected 8 show AOIs, found {len(aois)}")
    return aois


def aoi_dimensions(aoi: AoiFeature) -> tuple[float, float, np.ndarray, np.ndarray]:
    e1 = aoi.p1 - aoi.p0
    e2 = aoi.p3 - aoi.p0
    width = float(np.linalg.norm(e1))
    height = float(np.linalg.norm(e2))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid AOI dimensions for fid={aoi.fid}")
    return width, height, e1 / width, e2 / height


def ring_to_local(ring: list[tuple[float, float]], aoi: AoiFeature) -> np.ndarray:
    width, height, e1_unit, e2_unit = aoi_dimensions(aoi)
    pts = np.array(ring, dtype="float64")
    rel = pts - aoi.p0
    local = np.column_stack([rel @ e1_unit, rel @ e2_unit])
    return local


def local_to_world(aoi: AoiFeature, x: float, y: float) -> tuple[float, float]:
    _, _, e1_unit, e2_unit = aoi_dimensions(aoi)
    point = aoi.p0 + x * e1_unit + y * e2_unit
    return float(point[0]), float(point[1])


def local_view_bounds(aoi: AoiFeature, pad_fraction: float = CUTOUT_PAD_FRACTION) -> LocalBounds:
    width, height, _, _ = aoi_dimensions(aoi)
    pad_x = width * pad_fraction
    pad_y = height * pad_fraction
    return LocalBounds(-pad_x, -pad_y, width + pad_x, height + pad_y)


def view_polygon_for_aoi(aoi: AoiFeature, view_bounds: LocalBounds) -> ogr.Geometry:
    ring = [
        local_to_world(aoi, view_bounds.minx, view_bounds.miny),
        local_to_world(aoi, view_bounds.maxx, view_bounds.miny),
        local_to_world(aoi, view_bounds.maxx, view_bounds.maxy),
        local_to_world(aoi, view_bounds.minx, view_bounds.maxy),
        local_to_world(aoi, view_bounds.minx, view_bounds.miny),
    ]
    return ogr_polygon_from_ring(ring)


def geom_intersection_quiet(left: ogr.Geometry, right: ogr.Geometry) -> ogr.Geometry | None:
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        return left.Intersection(right)
    except Exception:
        return None
    finally:
        gdal.PopErrorHandler()


def geom_intersects_quiet(left: ogr.Geometry, right: ogr.Geometry) -> bool:
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        return bool(left.Intersects(right))
    except Exception:
        return False
    finally:
        gdal.PopErrorHandler()


def envelopes_intersect(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (left[1] < right[0] or left[0] > right[1] or left[3] < right[2] or left[2] > right[3])


def geom_boundary_quiet(geom: ogr.Geometry) -> ogr.Geometry | None:
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        return geom.Boundary()
    except Exception:
        return None
    finally:
        gdal.PopErrorHandler()


def sample_rotated_aoi_rgb(
    rasters: list[Path],
    aoi: AoiFeature,
    out_w: int = CUTOUT_OUT_W,
    out_h: int = CUTOUT_OUT_H,
) -> tuple[np.ndarray, LocalBounds]:
    view_bounds = local_view_bounds(aoi)
    us = np.linspace(view_bounds.minx, view_bounds.maxx, out_w, dtype="float64")
    vs = np.linspace(view_bounds.miny, view_bounds.maxy, out_h, dtype="float64")
    uu, vv = np.meshgrid(us, vs)
    _, _, e1_unit, e2_unit = aoi_dimensions(aoi)
    pts = aoi.p0 + uu.ravel()[:, None] * e1_unit + vv.ravel()[:, None] * e2_unit
    xs = pts[:, 0]
    ys = pts[:, 1]
    bands_flat = [np.zeros(xs.shape, dtype="float32") for _ in range(3)]
    valid_flat = np.zeros(xs.shape, dtype=bool)

    for path in rasters:
        ds = gdal.Open(str(path))
        rb = raster_bounds(ds)
        in_raster = (
            (xs >= rb.minx)
            & (xs <= rb.maxx)
            & (ys >= rb.miny)
            & (ys <= rb.maxy)
            & (~valid_flat)
        )
        if not np.any(in_raster):
            continue
        gt = ds.GetGeoTransform()
        inv = gdal.InvGeoTransform(gt)
        src_idx_all = np.where(in_raster)[0]
        px = inv[0] + inv[1] * xs[src_idx_all] + inv[2] * ys[src_idx_all]
        py = inv[3] + inv[4] * xs[src_idx_all] + inv[5] * ys[src_idx_all]
        xi = np.floor(px).astype("int64")
        yi = np.floor(py).astype("int64")
        inside = (xi >= 0) & (yi >= 0) & (xi < ds.RasterXSize) & (yi < ds.RasterYSize)
        if not np.any(inside):
            continue
        src_idx = src_idx_all[inside]
        xi = xi[inside]
        yi = yi[inside]
        xoff, xend = int(xi.min()), int(xi.max()) + 1
        yoff, yend = int(yi.min()), int(yi.max()) + 1
        xsize, ysize = xend - xoff, yend - yoff
        sampled_bands = []
        ok = np.ones(src_idx.shape, dtype=bool)
        for band_number in RGB_BANDS:
            band = ds.GetRasterBand(band_number)
            arr = band.ReadAsArray(xoff, yoff, xsize, ysize).astype("float32")
            values = arr[yi - yoff, xi - xoff]
            nodata = band.GetNoDataValue()
            if nodata is not None:
                ok &= values != nodata
            sampled_bands.append(values)
        ok &= np.any(np.stack(sampled_bands, axis=0) > 0, axis=0)
        if not np.any(ok):
            continue
        write_idx = src_idx[ok]
        for out_band, values in zip(bands_flat, sampled_bands):
            out_band[write_idx] = values[ok]
        valid_flat[write_idx] = True

    bands = [band.reshape(out_h, out_w) for band in bands_flat]
    valid = valid_flat.reshape(out_h, out_w)
    nodata_mask = ~valid
    rgb = arcgis_rgb_stretch(bands, nodata_mask)
    return rgb, view_bounds


def landslide_outline_segments_for_bounds(bounds: Bounds) -> tuple[list[np.ndarray], dict]:
    filter_geom = ogr_polygon_from_bounds(bounds)
    ds = ogr.Open(str(LANDSLIDES))
    layer = ds.GetLayer(0)
    layer.SetSpatialFilter(filter_geom)
    segments: list[np.ndarray] = []
    feature_count = 0
    invalid = 0
    vertex_count = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        feature_count += 1
        if not geom_is_valid(geom):
            invalid += 1
        for ring in polygon_exterior_rings(geom):
            if len(ring) >= 2:
                arr = np.array(ring, dtype="float64")
                segments.append(arr)
                vertex_count += len(arr)
    layer.SetSpatialFilter(None)
    return segments, {
        "features_drawn": feature_count,
        "invalid_polygon_count": invalid,
        "outline_segments": len(segments),
        "outline_vertices": vertex_count,
        "display_rule": "original landslide polygon exterior outlines, no fill",
    }


def add_landslide_outline_collection(
    ax: plt.Axes,
    segments: list[np.ndarray],
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    if not segments:
        return
    collection = LineCollection(
        segments,
        colors=LANDSLIDE_COLOR,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    collection.set_rasterized(True)
    ax.add_collection(collection)


def add_landslide_points(ax: plt.Axes, points: np.ndarray) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=0.72,
        marker="o",
        facecolors=LANDSLIDE_COLOR,
        edgecolors="none",
        linewidths=0,
        alpha=0.96,
        zorder=16,
        rasterized=True,
    )


def landslide_segments_for_aoi(aoi: AoiFeature, view_bounds: LocalBounds) -> tuple[list[np.ndarray], dict]:
    view_geom = view_polygon_for_aoi(aoi, view_bounds)
    ds = ogr.Open(str(LANDSLIDES))
    layer = ds.GetLayer(0)
    segments: list[np.ndarray] = []
    filter_env = view_geom.GetEnvelope()
    envelope_candidates = 0
    intersecting_features = 0
    invalid = 0
    repaired = 0
    repair_failed = 0
    clipped_empty = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        if not envelopes_intersect(geom.GetEnvelope(), filter_env):
            continue
        envelope_candidates += 1
        geom = geom.Clone()
        if not geom_is_valid(geom):
            invalid += 1
            try:
                geom = geom_make_valid(geom)
                repaired += 1
            except Exception:
                repair_failed += 1
                continue
        if not geom_intersects_quiet(geom, view_geom):
            continue
        intersecting_features += 1
        clipped = geom_intersection_quiet(geom, view_geom)
        if clipped is None or clipped.IsEmpty():
            clipped_empty += 1
            continue
        boundary = geom_boundary_quiet(clipped)
        if boundary is None or boundary.IsEmpty():
            clipped_empty += 1
            continue
        for ring in geometry_line_rings(boundary):
            local = ring_to_local(ring, aoi)
            if len(local) >= 2:
                segments.append(local)
    return segments, {
        "filter_rule": "full scan with envelope prefilter; MakeValid before intersect; drawn as Boundary(landslide ∩ AOI window)",
        "envelope_candidates": envelope_candidates,
        "intersecting_features": intersecting_features,
        "invalid_seen": invalid,
        "make_valid_repairs": repaired,
        "make_valid_failures": repair_failed,
        "clipped_empty_after_intersection": clipped_empty,
        "drawn_ring_segments": len(segments),
    }


def count_landslides_intersecting(geom_filter: ogr.Geometry, use_spatial_filter: bool = False) -> dict:
    ds = ogr.Open(str(LANDSLIDES))
    layer = ds.GetLayer(0)
    if use_spatial_filter:
        layer.SetSpatialFilter(geom_filter)
    filter_env = geom_filter.GetEnvelope()
    feature_count = 0
    envelope_candidates = 0
    invalid = 0
    repaired = 0
    repair_failed = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        if not use_spatial_filter and not envelopes_intersect(geom.GetEnvelope(), filter_env):
            continue
        envelope_candidates += 1
        geom = geom.Clone()
        if not geom_is_valid(geom):
            invalid += 1
            try:
                geom = geom_make_valid(geom)
                repaired += 1
            except Exception:
                repair_failed += 1
                continue
        if geom_intersects_quiet(geom, geom_filter):
            feature_count += 1
    if use_spatial_filter:
        layer.SetSpatialFilter(None)
    return {
        "features": feature_count,
        "invalid_seen": invalid,
        "make_valid_repairs": repaired,
        "make_valid_failures": repair_failed,
        "envelope_candidates": envelope_candidates,
        "use_spatial_filter": use_spatial_filter,
    }


def cutout_landslide_qa(aoi: AoiFeature, view_bounds: LocalBounds, draw_qa: dict) -> dict:
    aoi_geom = ogr_polygon_from_ring(aoi.ring)
    view_geom = view_polygon_for_aoi(aoi, view_bounds)
    return {
        "strict_aoi_intersects": count_landslides_intersecting(aoi_geom),
        "view_window_intersects": count_landslides_intersecting(view_geom),
        "spatial_filter_crosscheck": count_landslides_intersecting(view_geom, use_spatial_filter=True),
        "view_pad_fraction": CUTOUT_PAD_FRACTION,
        "drawn_ring_segments": draw_qa["drawn_ring_segments"],
        "draw_qa": draw_qa,
        "display_rule": "sub-figure shows only the AOI extent; landslides drawn as clipped polygon boundaries inside AOI",
    }


def draw_cutout(
    ax: plt.Axes,
    aoi: AoiFeature,
    rgb: np.ndarray,
    view_bounds: LocalBounds,
    landslide_segments: list[np.ndarray],
) -> None:
    ax.imshow(
        rgb,
        extent=(view_bounds.minx, view_bounds.maxx, view_bounds.maxy, view_bounds.miny),
        origin="upper",
        interpolation="bilinear",
    )
    if landslide_segments:
        ax.add_collection(
            LineCollection(
                landslide_segments,
                colors=LANDSLIDE_COLOR,
                linewidths=0.72,
                alpha=0.95,
                zorder=5,
            )
        )
    ax.set_xlim(view_bounds.minx, view_bounds.maxx)
    ax.set_ylim(view_bounds.maxy, view_bounds.miny)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#2a2a2a")


def save_cutout_png(
    output_dir: Path,
    aoi: AoiFeature,
    rgb: np.ndarray,
    view_bounds: LocalBounds,
    landslide_segments: list[np.ndarray],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"aoi_{aoi.fid:02d}.png"
    fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    draw_cutout(ax, aoi, rgb, view_bounds, landslide_segments)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return str(path)


def add_scale_bar(ax: plt.Axes, bounds: Bounds, zorder: int = 220) -> None:
    # In EPSG:32650 map units are metres.
    target_m = bounds.width * 0.16
    nice_steps = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    length = min(nice_steps, key=lambda step: abs(step - target_m))
    x0 = bounds.minx + bounds.width * 0.055
    y0 = bounds.miny + bounds.height * 0.065
    height = bounds.height * 0.008
    segments = 2
    seg = length / segments
    for i in range(segments):
        ax.add_patch(
            Rectangle(
                (x0 + i * seg, y0),
                seg,
                height,
                facecolor="black" if i % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.7,
                zorder=zorder,
            )
        )
    ax.text(
        x0,
        y0 + height * 2.5,
        "0",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        zorder=zorder + 1,
    )
    ax.text(
        x0 + length,
        y0 + height * 2.5,
        f"{length / 1000:g} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        zorder=zorder + 1,
    )


def add_esri_style_north_arrow(ax: plt.Axes) -> None:
    x = 0.948
    y = 0.892
    h = 0.112
    w = 0.034
    angle = -26
    transform = Affine2D().rotate_deg_around(x, y, angle) + ax.transAxes
    ax.text(
        x - w * 0.84,
        y + h * 0.55,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        fontstyle="italic",
        rotation=62,
        color="#d8d8d8",
        zorder=284,
    )
    shadow = [
        (x - w * 0.18, y + h * 0.42),
        (x + w * 0.46, y - h * 0.46),
        (x + w * 0.03, y - h * 0.30),
        (x - w * 0.18, y - h * 0.02),
    ]
    main = [
        (x - w * 0.15, y + h * 0.48),
        (x + w * 0.26, y - h * 0.44),
        (x - w * 0.03, y - h * 0.25),
        (x - w * 0.35, y - h * 0.48),
        (x - w * 0.18, y - h * 0.08),
    ]
    inner = [
        (x - w * 0.08, y + h * 0.18),
        (x + w * 0.16, y - h * 0.30),
        (x - w * 0.03, y - h * 0.19),
        (x - w * 0.22, y - h * 0.34),
    ]
    ax.add_patch(
        MplPolygon(
            shadow,
            closed=True,
            transform=transform,
            facecolor="#4d4d4d",
            edgecolor="#1f1f1f",
            linewidth=0.45,
            zorder=280,
        )
    )
    ax.add_patch(
        MplPolygon(
            main,
            closed=True,
            transform=transform,
            facecolor="white",
            edgecolor="#252525",
            linewidth=0.55,
            zorder=282,
        )
    )
    ax.add_patch(
        MplPolygon(
            inner,
            closed=True,
            transform=transform,
            facecolor="#c7c7c7",
            edgecolor="#252525",
            linewidth=0.35,
            zorder=283,
        )
    )


def nice_degree_step(span: float) -> float:
    for step in (1 / 60, 2 / 60, 5 / 60, 10 / 60, 15 / 60, 30 / 60, 1):
        count = span / step
        if 4 <= count <= 9:
            return step
    return 10 / 60


def degree_minute_label(value: float, is_lon: bool) -> str:
    hemi = ("E" if value >= 0 else "W") if is_lon else ("N" if value >= 0 else "S")
    value = abs(value)
    deg = int(math.floor(value))
    minutes = int(round((value - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}°{minutes:02d}′{hemi}"


def sample_graticule_line(lon: float | None, lat: float | None, lon_range: tuple[float, float], lat_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if lon is not None:
        lons = np.full(120, lon)
        lats = np.linspace(lat_range[0], lat_range[1], 120)
    else:
        lons = np.linspace(lon_range[0], lon_range[1], 120)
        lats = np.full(120, lat)
    return transform_xy(TO_UTM, lons, lats)


def add_degree_minute_graticule(ax: plt.Axes, bounds: Bounds, show_labels: bool = False) -> dict:
    corners_x = np.array([bounds.minx, bounds.maxx, bounds.maxx, bounds.minx])
    corners_y = np.array([bounds.miny, bounds.miny, bounds.maxy, bounds.maxy])
    lon_corners, lat_corners = transform_xy(TO_GEO, corners_x, corners_y)
    lon_min, lon_max = float(lon_corners.min()), float(lon_corners.max())
    lat_min, lat_max = float(lat_corners.min()), float(lat_corners.max())

    lon_step = nice_degree_step(lon_max - lon_min)
    lat_step = nice_degree_step(lat_max - lat_min)
    lons = np.arange(math.ceil(lon_min / lon_step) * lon_step, lon_max, lon_step)
    lats = np.arange(math.ceil(lat_min / lat_step) * lat_step, lat_max, lat_step)

    line_segments = []
    for lon in lons:
        xs, ys = sample_graticule_line(float(lon), None, (lon_min, lon_max), (lat_min, lat_max))
        line_segments.append(np.column_stack([xs, ys]))
    for lat in lats:
        xs, ys = sample_graticule_line(None, float(lat), (lon_min, lon_max), (lat_min, lat_max))
        line_segments.append(np.column_stack([xs, ys]))
    ax.add_collection(LineCollection(line_segments, colors="#f4f0df", linewidths=0.48, alpha=0.46, zorder=6))

    ax.set_xticks([])
    ax.set_yticks([])
    tick_len = min(bounds.width, bounds.height) * 0.010
    label_zorder = 230
    for lon in lons:
        bottom_x, _, _ = TO_UTM.TransformPoint(float(lon), lat_min)
        top_x, _, _ = TO_UTM.TransformPoint(float(lon), lat_max)
        if bounds.minx <= bottom_x <= bounds.maxx:
            ax.plot([bottom_x, bottom_x], [bounds.miny, bounds.miny + tick_len], color="#252525", lw=0.6, zorder=25)
            if show_labels:
                ax.text(
                    bottom_x,
                    bounds.miny - bounds.height * 0.018,
                    degree_minute_label(float(lon), True),
                    ha="center",
                    va="top",
                    fontsize=7,
                    clip_on=False,
                    zorder=label_zorder,
                )
        if bounds.minx <= top_x <= bounds.maxx:
            ax.plot([top_x, top_x], [bounds.maxy - tick_len, bounds.maxy], color="#252525", lw=0.6, zorder=25)
            if show_labels:
                ax.text(
                    top_x,
                    bounds.maxy + bounds.height * 0.018,
                    degree_minute_label(float(lon), True),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    clip_on=False,
                    zorder=label_zorder,
                )
    for lat in lats:
        _, left_y, _ = TO_UTM.TransformPoint(lon_min, float(lat))
        _, right_y, _ = TO_UTM.TransformPoint(lon_max, float(lat))
        if bounds.miny <= left_y <= bounds.maxy:
            ax.plot([bounds.minx, bounds.minx + tick_len], [left_y, left_y], color="#252525", lw=0.6, zorder=25)
            if show_labels:
                ax.text(
                    bounds.minx - bounds.width * 0.010,
                    left_y,
                    degree_minute_label(float(lat), False),
                    ha="right",
                    va="center",
                    fontsize=7,
                    rotation=90,
                    clip_on=False,
                    zorder=label_zorder,
                )
        if bounds.miny <= right_y <= bounds.maxy:
            ax.plot([bounds.maxx - tick_len, bounds.maxx], [right_y, right_y], color="#252525", lw=0.6, zorder=25)
            if show_labels:
                ax.text(
                    bounds.maxx + bounds.width * 0.010,
                    right_y,
                    degree_minute_label(float(lat), False),
                    ha="left",
                    va="center",
                    fontsize=7,
                    rotation=90,
                    clip_on=False,
                    zorder=label_zorder,
                )
    return {
        "longitude_range": [lon_min, lon_max],
        "latitude_range": [lat_min, lat_max],
        "longitude_step_degrees": lon_step,
        "latitude_step_degrees": lat_step,
        "label_format": "degree-minute",
        "labels_visible": show_labels,
    }


def plot_aoi_layer(
    ax: plt.Axes,
    path: Path,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    patches = [MplPolygon(ring, closed=True) for ring in iter_polygon_exteriors(path)]
    collection = PatchCollection(
        patches,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_collection(collection)
    outline_segments = []
    for ring in iter_polygon_exteriors(path):
        outline_segments.append(ring)
    ax.add_collection(
        LineCollection(outline_segments, colors=edgecolor, linewidths=linewidth, zorder=zorder + 1)
    )


def write_qgis_hint(output_dir: Path) -> None:
    """Keep an editable hint for users who want to reproduce styling in QGIS."""
    hint = output_dir / f"wuping_landslide_map_{VERSION}_style_notes.txt"
    hint.write_text(
        "\n".join(
            [
                "Layer order:",
                "1. 12 Wuping tile_*_mosaic.tif rasters, RGB bands 6/4/2, 0.25% min/max clip, gamma 2.3.",
                "2. Landslide: centroids of landslides_merged.shp, small pure-yellow points without outline.",
                "3. Refinement-2: wuping.shp, cyan-blue semi-transparent polygon fill above landslide points.",
                "4. Refinement-1: test2.shp, coral semi-transparent polygon fill above landslide points.",
                "5. Show AOIs: all.shp, thin red locator rectangles with leader lines, no legend entry.",
                "6. Cutouts: all.shp exact AOI windows, RGB 6/4/2, landslide boundaries clipped to each AOI in yellow.",
                "",
                "CRS: EPSG:32650 (WGS 84 / UTM zone 50N).",
            ]
        ),
        encoding="utf-8",
    )


def add_legend_overlay(ax: plt.Axes) -> None:
    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#e66b5b",
            edgecolor="#8f2e24",
            alpha=0.58,
            linewidth=1.2,
            label="Refinement-1",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#1f9fb5",
            edgecolor="#075b66",
            alpha=0.50,
            linewidth=1.2,
            label="Refinement-2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=LANDSLIDE_COLOR,
            markeredgecolor=LANDSLIDE_COLOR,
            markeredgewidth=0,
            markersize=5.2,
            label="Landslide",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.055),
        bbox_transform=ax.transAxes,
        frameon=True,
        facecolor="white",
        edgecolor="#6b6b6b",
        framealpha=1.0,
        fontsize=8.6,
        borderpad=0.85,
        borderaxespad=0,
        handlelength=1.45,
        labelspacing=0.85,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.set_zorder(310)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    missing = [p for p in [REFINEMENT_1, REFINEMENT_2, LANDSLIDES, SHOW_AOI] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required vector inputs: {missing}")

    rasters = raster_paths()
    if len(rasters) != 12:
        raise RuntimeError(f"Expected 12 main Wuping rasters, found {len(rasters)}: {rasters}")

    raster_datasets = [gdal.Open(str(path)) for path in rasters]
    raster_crs = projection_name(raster_datasets[0].GetProjection())
    raster_bounds_list = [raster_bounds(ds) for ds in raster_datasets]
    data_bounds = union_bounds(raster_bounds_list)
    map_bounds = data_bounds

    vector_infos = {
        "Refinement-1": layer_info(REFINEMENT_1),
        "Refinement-2": layer_info(REFINEMENT_2),
        "Landslide source polygons": layer_info(LANDSLIDES),
        "Show AOIs": layer_info(SHOW_AOI),
    }
    for name, info in vector_infos.items():
        if info["feature_count"] <= 0 or info["empty_count"] == info["feature_count"]:
            raise RuntimeError(f"Required layer {name} has no usable features: {info}")
        if info["crs"] != raster_crs:
            raise RuntimeError(f"CRS mismatch for {name}: {info['crs']} vs raster {raster_crs}")

    refinement_1_geom = vector_union_geometry(REFINEMENT_1)
    refinement_2_geom = vector_union_geometry(REFINEMENT_2)
    topo = {
        "refinement_1_within_refinement_2": bool(refinement_1_geom.Within(refinement_2_geom)),
        "refinement_2_contains_refinement_1": bool(refinement_2_geom.Contains(refinement_1_geom)),
        "refinement_1_intersects_refinement_2": bool(refinement_1_geom.Intersects(refinement_2_geom)),
        "refinement_1_area_m2": refinement_1_geom.GetArea(),
        "refinement_2_area_m2": refinement_2_geom.GetArea(),
    }
    intersection = refinement_1_geom.Intersection(refinement_2_geom)
    topo["intersection_area_m2"] = intersection.GetArea()
    topo["refinement_1_intersection_ratio"] = (
        intersection.GetArea() / refinement_1_geom.GetArea()
        if refinement_1_geom.GetArea()
        else None
    )

    points, point_qa = centroid_points(LANDSLIDES)
    show_aois = read_show_aois(map_bounds)
    cutout_dir = OUT_DIR / f"cutouts_{VERSION}"
    cutout_payload = []
    for aoi in show_aois:
        rgb, view_bounds = sample_rotated_aoi_rgb(rasters, aoi)
        segments, draw_qa = landslide_segments_for_aoi(aoi, view_bounds)
        cutout_path = save_cutout_png(cutout_dir, aoi, rgb, view_bounds, segments)
        landslide_qa = cutout_landslide_qa(aoi, view_bounds, draw_qa)
        cutout_payload.append(
            {
                "aoi": aoi,
                "rgb": rgb,
                "view_bounds": view_bounds,
                "segments": segments,
                "landslide_qa": landslide_qa,
                "path": cutout_path,
            }
        )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    ax = fig.add_axes(MAIN_AX)
    ax.set_facecolor("#fbfbf7")

    raster_drawn = []
    for path in rasters:
        result = read_rgb_for_extent(path, map_bounds)
        if result is None:
            continue
        rgb, rb = result
        ax.imshow(
            rgb,
            extent=(rb.minx, rb.maxx, rb.miny, rb.maxy),
            origin="upper",
            interpolation="bilinear",
            zorder=1,
        )
        raster_drawn.append(str(path))

    add_landslide_points(ax, points)

    plot_aoi_layer(
        ax,
        REFINEMENT_2,
        facecolor="#1f9fb5",
        edgecolor="#003b46",
        alpha=0.42,
        linewidth=2.10,
        zorder=24,
    )
    plot_aoi_layer(
        ax,
        REFINEMENT_1,
        facecolor="#e66b5b",
        edgecolor="#7c1f17",
        alpha=0.48,
        linewidth=2.25,
        zorder=26,
    )
    for aoi in show_aois:
        ax.add_patch(
            MplPolygon(
                aoi.ring,
                closed=True,
                fill=False,
                edgecolor=SHOW_AOI_COLOR,
                linewidth=1.05,
                zorder=36,
            )
        )

    ax.set_xlim(map_bounds.minx, map_bounds.maxx)
    ax.set_ylim(map_bounds.miny, map_bounds.maxy)
    ax.set_aspect("equal", adjustable="box")
    graticule_qa = add_degree_minute_graticule(ax, map_bounds, show_labels=True)
    add_scale_bar(ax, map_bounds)
    add_esri_style_north_arrow(ax)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#252525")

    bottom_payload = [item for item in cutout_payload if item["aoi"].placement == "bottom"]
    right_payload = [item for item in cutout_payload if item["aoi"].placement == "right"]
    bottom_payload.sort(key=lambda item: item["aoi"].center[0])
    right_rank = {fid: i for i, fid in enumerate(RIGHT_COLUMN_ORDER)}
    right_payload.sort(key=lambda item: right_rank.get(item["aoi"].fid, 100 + item["aoi"].center[1]))

    bottom_axes = []
    if bottom_payload:
        n = len(bottom_payload)
        gap = 0.020
        total_w = MAIN_AX[2]
        panel_w = min(0.135, (total_w - gap * (n - 1)) / n)
        panel_h = panel_w * FIG_WIDTH_IN * 0.75 / FIG_HEIGHT_IN
        y = 0.105
        x0 = MAIN_AX[0] + (total_w - (panel_w * n + gap * (n - 1))) / 2
        for i, item in enumerate(bottom_payload):
            panel_ax = fig.add_axes([x0 + i * (panel_w + gap), y, panel_w, panel_h])
            draw_cutout(panel_ax, item["aoi"], item["rgb"], item["view_bounds"], item["segments"])
            bottom_axes.append((panel_ax, item["aoi"]))

    right_axes = []
    if right_payload:
        n = len(right_payload)
        panel_x = MAIN_AX[0] + MAIN_AX[2] + 0.045
        panel_w = bottom_axes[0][0].get_position().width if bottom_axes else 0.118
        panel_h = bottom_axes[0][0].get_position().height if bottom_axes else panel_w * FIG_WIDTH_IN * 0.75 / FIG_HEIGHT_IN
        gap = (MAIN_AX[3] - panel_h * n) / max(n - 1, 1)
        start_y = MAIN_AX[1] + MAIN_AX[3] - panel_h
        for i, item in enumerate(right_payload):
            y = start_y - i * (panel_h + gap)
            panel_ax = fig.add_axes([panel_x, y, panel_w, panel_h])
            draw_cutout(panel_ax, item["aoi"], item["rgb"], item["view_bounds"], item["segments"])
            right_axes.append((panel_ax, item["aoi"]))

    for panel_ax, aoi in bottom_axes:
        con = ConnectionPatch(
            xyA=aoi.center,
            coordsA=ax.transData,
            xyB=(0.5, 1.0),
            coordsB=panel_ax.transAxes,
            color="#d7191c",
            linewidth=0.75,
            alpha=0.82,
            zorder=45,
            clip_on=False,
        )
        ax.add_artist(con)
    for panel_ax, aoi in right_axes:
        con = ConnectionPatch(
            xyA=aoi.center,
            coordsA=ax.transData,
            xyB=(0.0, 0.5),
            coordsB=panel_ax.transAxes,
            color="#d7191c",
            linewidth=0.75,
            alpha=0.82,
            zorder=45,
            clip_on=False,
        )
        ax.add_artist(con)

    add_legend_overlay(ax)

    png_path = OUT_DIR / f"wuping_landslide_map_{VERSION}.png"
    pdf_path = OUT_DIR / f"wuping_landslide_map_{VERSION}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    qa = {
        "version": VERSION,
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "rasters_selected": raster_drawn,
        "raster_count": len(raster_drawn),
        "raster_crs": raster_crs,
        "data_bounds": data_bounds.tuple,
        "map_bounds": map_bounds.tuple,
        "vector_layers": vector_infos,
        "topology": topo,
        "landslide_point_processing": point_qa,
        "show_aoi_count": len(show_aois),
        "show_aoi_layout": [
            {
                "fid": item["aoi"].fid,
                "placement": item["aoi"].placement,
                "center": item["aoi"].center,
                "bounds": item["aoi"].bounds.tuple,
                "view_bounds_local": item["view_bounds"].tuple,
                "view_pad_fraction": CUTOUT_PAD_FRACTION,
                "cutout": item["path"],
                "landslide_outline_segments": len(item["segments"]),
                "landslide_qa": item["landslide_qa"],
            }
            for item in cutout_payload
        ],
        "raster_display": {
            "rgb_bands": list(RGB_BANDS),
            "min_max_clip_percent": ARCGIS_MIN_MAX_CLIP_PERCENT,
            "gamma": ARCGIS_GAMMA,
        },
        "symbology": {
            "Refinement-1": "coral semi-transparent polygon fill, dark red outline, drawn above landslide points",
            "Refinement-2": "cyan-blue semi-transparent polygon fill, dark teal outline, drawn above landslide points",
            "Landslide": "per-polygon centroid, small pure-yellow point without outline, drawn below refinement layers",
            "Show AOI": "thin red rotated rectangle on main map, no legend entry",
            "Cutout landslides": "landslide polygon boundaries clipped to each exact AOI window, no fill, yellow boundary",
        },
        "graticule": graticule_qa,
        "layout_elements": [
            "map body",
            "in-map legend",
            "8 rotated AOI cutouts outside the main map",
            "red AOI locator rectangles and leader lines",
            "scale bar",
            "small oblique ESRI-style north arrow",
            "degree-minute longitude/latitude frame labels",
        ],
    }
    qa_path = OUT_DIR / f"wuping_landslide_map_{VERSION}_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    write_qgis_hint(OUT_DIR)

    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
