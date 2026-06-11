"""Interactive map previews for GeoTIFF imagery and landslide vector outputs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from lsd_app.core.channels import normalize_rgb_bands, rgb_bands_to_indexes
from lsd_app.core.raster import require_rasterio
from lsd_app.core.visualization import _bands_to_rgb, _overview_shape, _write_png_rgb


DEFAULT_MAP_MAX_DIMENSION = 2048
DEFAULT_MAX_FEATURES = 5000
MAP_CRS = "EPSG:4326"


class MapVisualizationError(RuntimeError):
    """Raised when an interactive map preview cannot be generated."""


@dataclass(frozen=True)
class MapPreviewResult:
    """Artifacts and metadata produced for an interactive map preview."""

    map_html_path: Path
    base_preview_path: Path
    vector_preview_path: Path
    raster_path: Path
    raster_paths: tuple[Path, ...]
    vector_path: Path
    width: int
    height: int
    scale: float
    raster_crs: str | None
    raster_bounds_wgs84: tuple[float, float, float, float]
    feature_count: int
    displayed_feature_count: int
    truncated: bool
    RGB_bands: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_html_path": str(self.map_html_path),
            "base_preview_path": str(self.base_preview_path),
            "vector_preview_path": str(self.vector_preview_path),
            "raster_path": str(self.raster_path),
            "raster_paths": [str(path) for path in self.raster_paths],
            "raster_count": len(self.raster_paths),
            "vector_path": str(self.vector_path),
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "raster_crs": self.raster_crs,
            "raster_bounds_wgs84": list(self.raster_bounds_wgs84),
            "feature_count": self.feature_count,
            "displayed_feature_count": self.displayed_feature_count,
            "truncated": self.truncated,
            "RGB_bands": self.RGB_bands,
            "message": self.message,
        }


def build_interactive_map_preview(
    *,
    raster_path: str | Path | None = None,
    raster_paths: Sequence[str | Path] | None = None,
    vector_path: str | Path,
    output_dir: str | Path,
    RGB_bands: str,
    max_dimension: int = DEFAULT_MAP_MAX_DIMENSION,
    simplify_pixels: float = 1.0,
    max_features: int = DEFAULT_MAX_FEATURES,
    polygon_opacity: float = 0.35,
) -> MapPreviewResult:
    """Build a Leaflet/Folium map from GeoTIFF basemap imagery and one landslide vector file."""

    rasters = _normalize_raster_paths(raster_path=raster_path, raster_paths=raster_paths)
    first_raster = rasters[0]
    vector = Path(vector_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for raster in rasters:
        if not raster.exists() or not raster.is_file():
            raise MapVisualizationError(f"GeoTIFF does not exist or is not a file: {raster}")
        if raster.suffix.lower() not in {".tif", ".tiff"}:
            raise MapVisualizationError(f"Basemap must be a .tif/.tiff GeoTIFF: {raster}")
    if not vector.exists() or not vector.is_file():
        raise MapVisualizationError(f"Landslide vector file does not exist or is not a file: {vector}")
    if vector.suffix.lower() not in {".shp", ".gpkg", ".geojson", ".json"}:
        raise MapVisualizationError("Landslide vector must be .shp, .gpkg, .geojson, or .json.")

    output.mkdir(parents=True, exist_ok=True)
    base_preview_path = output / "base_rgb_preview.png"
    vector_preview_path = output / "landslides_preview.geojson"
    map_html_path = output / "interactive_map.html"

    rasterio = require_rasterio()
    try:
        if len(rasters) == 1:
            (
                rgb,
                out_width,
                out_height,
                scale,
                raster_crs,
                raster_bounds_native,
                raster_bounds_wgs84,
                native_pixel_size,
            ) = _read_single_raster_preview(
                rasters[0],
                RGB_bands=RGB_bands,
                max_dimension=max(1, int(max_dimension)),
                rasterio=rasterio,
            )
        else:
            (
                rgb,
                out_width,
                out_height,
                scale,
                raster_crs,
                raster_bounds_native,
                raster_bounds_wgs84,
                native_pixel_size,
            ) = _read_mosaic_raster_preview(
                rasters,
                RGB_bands=RGB_bands,
                max_dimension=max(1, int(max_dimension)),
                rasterio=rasterio,
            )
        _write_png_rgb(base_preview_path, rgb)
        raster_crs_string = raster_crs.to_string() if raster_crs else None
    except MapVisualizationError:
        raise
    except Exception as exc:
        raise MapVisualizationError(f"Failed to read GeoTIFF preview data: {exc}") from exc

    simplify_tolerance = max(0.0, float(simplify_pixels)) * native_pixel_size
    vector_payload, feature_count, displayed_feature_count, truncated = _prepare_vector_geojson(
        vector,
        raster_crs=raster_crs,
        raster_bounds_native=raster_bounds_native,
        simplify_tolerance=simplify_tolerance,
        max_features=max(0, int(max_features)),
    )
    vector_preview_path.write_text(json.dumps(vector_payload, ensure_ascii=False), encoding="utf-8")

    html = _build_map_html(
        base_preview_path=base_preview_path,
        vector_geojson=vector_payload,
        bounds_wgs84=raster_bounds_wgs84,
        polygon_opacity=float(min(max(polygon_opacity, 0.0), 1.0)),
    )
    map_html_path.write_text(html, encoding="utf-8")

    message = f"Interactive map generated with {displayed_feature_count}/{feature_count} polygon(s)."
    if len(rasters) > 1:
        message += f" Basemap mosaic uses {len(rasters)} GeoTIFF(s)."
    if truncated:
        message += f" Preview was limited to {displayed_feature_count} polygon(s) for browser performance."
    return MapPreviewResult(
        map_html_path=map_html_path,
        base_preview_path=base_preview_path,
        vector_preview_path=vector_preview_path,
        raster_path=first_raster,
        raster_paths=tuple(rasters),
        vector_path=vector,
        width=out_width,
        height=out_height,
        scale=scale,
        raster_crs=raster_crs_string,
        raster_bounds_wgs84=raster_bounds_wgs84,
        feature_count=feature_count,
        displayed_feature_count=displayed_feature_count,
        truncated=truncated,
        RGB_bands=normalize_rgb_bands(RGB_bands),
        message=message,
    )


def _normalize_raster_paths(
    *,
    raster_path: str | Path | None,
    raster_paths: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    raw_paths: list[str | Path] = []
    if raster_paths is not None:
        raw_paths.extend(raster_paths)
    if raster_path is not None:
        raw_paths.append(raster_path)
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in raw_paths:
        text = str(item or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    if not resolved:
        raise MapVisualizationError("Select at least one basemap GeoTIFF.")
    return tuple(resolved)


def _read_single_raster_preview(
    raster: Path,
    *,
    RGB_bands: str,
    max_dimension: int,
    rasterio: Any,
) -> tuple[Any, int, int, float, Any, tuple[float, float, float, float], tuple[float, float, float, float], float]:
    with rasterio.open(raster) as src:
        if src.crs is None:
            raise MapVisualizationError(
                "The selected GeoTIFF has no CRS. A georeferenced CRS is required for a lat/lon grid and scale bar."
            )
        out_width, out_height, scale = _overview_shape(
            int(src.width),
            int(src.height),
            max_dimension=max_dimension,
        )
        rgb = _read_rgb_preview(src, RGB_bands=RGB_bands, out_width=out_width, out_height=out_height, rasterio=rasterio)
        raster_bounds_native = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
        raster_bounds_wgs84 = _bounds_to_wgs84(src, rasterio=rasterio)
        native_pixel_size = max(abs(float(src.transform.a)), abs(float(src.transform.e)))
        return rgb, out_width, out_height, scale, src.crs, raster_bounds_native, raster_bounds_wgs84, native_pixel_size


def _read_mosaic_raster_preview(
    rasters: Sequence[Path],
    *,
    RGB_bands: str,
    max_dimension: int,
    rasterio: Any,
) -> tuple[Any, int, int, float, Any, tuple[float, float, float, float], tuple[float, float, float, float], float]:
    import numpy as np

    indexes = rgb_bands_to_indexes(RGB_bands)
    infos: list[dict[str, Any]] = []
    raster_crs = None
    pixel_sizes: list[float] = []
    for raster in rasters:
        with rasterio.open(raster) as src:
            if src.crs is None:
                raise MapVisualizationError(
                    f"GeoTIFF has no CRS: {raster}. Every basemap tile needs a georeferenced CRS."
                )
            if raster_crs is None:
                raster_crs = src.crs
            elif src.crs != raster_crs:
                raise MapVisualizationError(
                    "All selected basemap GeoTIFFs must use the same CRS for mosaic preview. "
                    f"First CRS is {raster_crs}; {raster} uses {src.crs}."
                )
            missing = [index for index in indexes if index > int(src.count)]
            if missing:
                raise MapVisualizationError(
                    f"{normalize_rgb_bands(RGB_bands)} requires band(s) {missing}, but {raster.name} has {src.count} band(s)."
                )
            bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
            pixel_size_x = abs(float(src.transform.a)) or (bounds[2] - bounds[0]) / max(1, int(src.width))
            pixel_size_y = abs(float(src.transform.e)) or (bounds[3] - bounds[1]) / max(1, int(src.height))
            pixel_sizes.append(max(pixel_size_x, pixel_size_y))
            infos.append(
                {
                    "path": raster,
                    "bounds": bounds,
                    "pixel_size_x": pixel_size_x,
                    "pixel_size_y": pixel_size_y,
                }
            )
    if raster_crs is None or not infos:
        raise MapVisualizationError("No readable GeoTIFFs were selected for mosaic preview.")

    left = min(info["bounds"][0] for info in infos)
    bottom = min(info["bounds"][1] for info in infos)
    right = max(info["bounds"][2] for info in infos)
    top = max(info["bounds"][3] for info in infos)
    raster_bounds_native = (left, bottom, right, top)
    native_pixel_size = min(pixel_sizes) if pixel_sizes else 1.0
    native_width = max(1, int(round((right - left) / native_pixel_size)))
    native_height = max(1, int(round((top - bottom) / native_pixel_size)))
    out_width, out_height, scale = _overview_shape(native_width, native_height, max_dimension=max_dimension)
    xres = (right - left) / out_width
    yres = (top - bottom) / out_height
    if xres <= 0 or yres <= 0:
        raise MapVisualizationError(f"Selected GeoTIFF bounds are invalid: {raster_bounds_native}")

    mosaic = np.full((3, out_height, out_width), np.nan, dtype=np.float32)
    coverage = np.zeros((out_height, out_width), dtype=bool)
    for info in infos:
        tile_left, tile_bottom, tile_right, tile_top = info["bounds"]
        col0 = max(0, int(math.floor((tile_left - left) / xres)))
        col1 = min(out_width, int(math.ceil((tile_right - left) / xres)))
        row0 = max(0, int(math.floor((top - tile_top) / yres)))
        row1 = min(out_height, int(math.ceil((top - tile_bottom) / yres)))
        tile_width = col1 - col0
        tile_height = row1 - row0
        if tile_width <= 0 or tile_height <= 0:
            continue
        with rasterio.open(info["path"]) as src:
            data = src.read(
                indexes,
                out_shape=(len(indexes), tile_height, tile_width),
                resampling=rasterio.enums.Resampling.bilinear,
                masked=True,
            )
        if hasattr(data, "filled"):
            values = data.astype("float32").filled(np.nan)
        else:
            values = np.asarray(data, dtype=np.float32)
        valid = np.all(np.isfinite(values), axis=0)
        target = mosaic[:, row0:row1, col0:col1]
        target[:, valid] = values[:, valid]
        coverage[row0:row1, col0:col1] |= valid

    if not np.any(coverage):
        raise MapVisualizationError("Selected GeoTIFFs did not produce any valid preview pixels.")
    _fill_missing_preview_values(mosaic, coverage)
    rgb = _bands_to_rgb(mosaic)
    raster_bounds_wgs84 = _native_bounds_to_wgs84(raster_crs, raster_bounds_native, rasterio=rasterio)
    return rgb, out_width, out_height, scale, raster_crs, raster_bounds_native, raster_bounds_wgs84, native_pixel_size


def _fill_missing_preview_values(values: Any, coverage: Any) -> None:
    import numpy as np

    for band_index in range(values.shape[0]):
        band = values[band_index]
        finite = np.isfinite(band)
        if np.any(finite):
            fill_value = float(np.nanpercentile(band[finite], 2))
        else:
            fill_value = 0.0
        band[~coverage | ~finite] = fill_value


def _read_rgb_preview(src: Any, *, RGB_bands: str, out_width: int, out_height: int, rasterio: Any) -> Any:
    import numpy as np

    indexes = rgb_bands_to_indexes(RGB_bands)
    missing = [index for index in indexes if index > int(src.count)]
    if missing:
        raise MapVisualizationError(
            f"{normalize_rgb_bands(RGB_bands)} requires band(s) {missing}, but the GeoTIFF has {src.count} band(s)."
        )
    bands = [
        src.read(
            index,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        for index in indexes
    ]
    return _bands_to_rgb(np.stack(bands, axis=0))


def _bounds_to_wgs84(src: Any, *, rasterio: Any) -> tuple[float, float, float, float]:
    bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
    return _native_bounds_to_wgs84(src.crs, bounds, rasterio=rasterio)


def _native_bounds_to_wgs84(crs: Any, bounds: tuple[float, float, float, float], *, rasterio: Any) -> tuple[float, float, float, float]:
    from rasterio.warp import transform_bounds

    try:
        west, south, east, north = transform_bounds(crs, MAP_CRS, *bounds, densify_pts=21)
    except Exception as exc:
        raise MapVisualizationError(f"Failed to transform raster bounds to {MAP_CRS}: {exc}") from exc
    bounds = (float(west), float(south), float(east), float(north))
    _validate_wgs84_bounds(bounds)
    return bounds


def _validate_wgs84_bounds(bounds: tuple[float, float, float, float]) -> None:
    west, south, east, north = bounds
    values = (west, south, east, north)
    if not all(math.isfinite(value) for value in values):
        raise MapVisualizationError(f"Raster bounds are not finite after CRS transform: {bounds}")
    if east <= west or north <= south:
        raise MapVisualizationError(f"Raster bounds are invalid after CRS transform: {bounds}")
    if south < -90.0 or north > 90.0 or west < -180.0 or east > 180.0:
        raise MapVisualizationError(f"Raster bounds fall outside valid longitude/latitude ranges: {bounds}")


def _prepare_vector_geojson(
    vector_path: Path,
    *,
    raster_crs: Any,
    raster_bounds_native: tuple[float, float, float, float],
    simplify_tolerance: float,
    max_features: int,
) -> tuple[dict[str, Any], int, int, bool]:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ModuleNotFoundError as exc:
        raise MapVisualizationError("Interactive vector overlay requires geopandas and shapely.") from exc

    try:
        gdf = gpd.read_file(vector_path)
    except Exception as exc:
        raise MapVisualizationError(f"Failed to read landslide vector file: {exc}") from exc
    feature_count = int(len(gdf))
    if feature_count <= 0:
        raise MapVisualizationError(f"Landslide vector file contains no features: {vector_path}")

    working = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if working.empty:
        raise MapVisualizationError(f"Landslide vector file contains no non-empty geometries: {vector_path}")
    if working.crs is None:
        working = working.set_crs(raster_crs, allow_override=True)
    elif raster_crs is not None and working.crs != raster_crs:
        working = working.to_crs(raster_crs)

    bounds_poly = box(*raster_bounds_native)
    working = working[working.intersects(bounds_poly)].copy()
    if working.empty:
        raise MapVisualizationError("No landslide polygons intersect the selected GeoTIFF extent. Check the raster/vector CRS pairing.")
    working["geometry"] = working.geometry.intersection(bounds_poly)
    working = working[working.geometry.notna() & ~working.geometry.is_empty].copy()
    if working.empty:
        raise MapVisualizationError("No landslide polygons remain after clipping to the selected GeoTIFF extent.")

    truncated = False
    if max_features and len(working) > max_features:
        working = _largest_features_first(working).head(max_features).copy()
        truncated = True
    if simplify_tolerance > 0:
        working["geometry"] = working.geometry.simplify(simplify_tolerance, preserve_topology=True)
        working = working[working.geometry.notna() & ~working.geometry.is_empty].copy()
    if working.crs is None:
        working = working.set_crs(raster_crs, allow_override=True)
    display = working.to_crs(MAP_CRS)
    display = _compact_preview_properties(display)
    payload = json.loads(display.to_json(drop_id=True))
    return payload, feature_count, int(len(display)), truncated


def _largest_features_first(gdf: Any) -> Any:
    if "area_m2" in gdf.columns:
        try:
            return gdf.sort_values("area_m2", ascending=False)
        except Exception:
            pass
    try:
        ranked = gdf.copy()
        ranked["_preview_area"] = ranked.geometry.area
        return ranked.sort_values("_preview_area", ascending=False).drop(columns=["_preview_area"])
    except Exception:
        return gdf


def _compact_preview_properties(gdf: Any) -> Any:
    known = (
        "id",
        "area_m2",
        "area_px",
        "mean_confidence",
        "confidence",
        "score",
        "probability",
    )
    keep = [column for column in known if column in gdf.columns]
    if not keep:
        gdf = gdf.copy()
        gdf["preview_id"] = list(range(1, len(gdf) + 1))
        keep = ["preview_id"]
    return gdf[[*keep, "geometry"]].copy()


def _build_map_html(
    *,
    base_preview_path: Path,
    vector_geojson: dict[str, Any],
    bounds_wgs84: tuple[float, float, float, float],
    polygon_opacity: float,
) -> str:
    try:
        import folium
    except ModuleNotFoundError as exc:
        raise MapVisualizationError("Interactive map rendering requires folium in the web environment.") from exc

    west, south, east, north = bounds_wgs84
    image_uri = "data:image/png;base64," + base64.b64encode(base_preview_path.read_bytes()).decode("ascii")
    center = [(south + north) / 2.0, (west + east) / 2.0]
    fmap = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.raster_layers.ImageOverlay(
        image=image_uri,
        bounds=[[south, west], [north, east]],
        name="GeoTIFF RGB basemap",
        opacity=1.0,
        interactive=True,
        zindex=1,
    ).add_to(fmap)

    _add_graticule(fmap, bounds_wgs84)
    tooltip_fields, aliases = _popup_fields(vector_geojson)
    popup = folium.GeoJsonPopup(fields=tooltip_fields, aliases=aliases, labels=True, localize=True) if tooltip_fields else None
    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, sticky=False, localize=True) if tooltip_fields else None
    folium.GeoJson(
        vector_geojson,
        name="Landslide polygons",
        style_function=lambda _feature: {
            "color": "#ff3b30",
            "weight": 2,
            "fillColor": "#ff6b35",
            "fillOpacity": polygon_opacity,
        },
        highlight_function=lambda _feature: {"weight": 3, "fillOpacity": min(polygon_opacity + 0.2, 0.9)},
        popup=popup,
        tooltip=tooltip,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    fmap.get_root().html.add_child(folium.Element(_map_overlay_controls_html()))
    return fmap.get_root().render()


def _popup_fields(vector_geojson: dict[str, Any]) -> tuple[list[str], list[str]]:
    features = vector_geojson.get("features")
    if not isinstance(features, list) or not features:
        return [], []
    properties = features[0].get("properties") if isinstance(features[0], dict) else None
    if not isinstance(properties, dict):
        return [], []
    preferred = ["id", "preview_id", "area_m2", "area_px", "mean_confidence", "confidence", "score", "probability"]
    fields = [field for field in preferred if field in properties]
    labels = {
        "id": "ID",
        "preview_id": "Preview ID",
        "area_m2": "Area m2",
        "area_px": "Area px",
        "mean_confidence": "Mean confidence",
        "confidence": "Confidence",
        "score": "Score",
        "probability": "Probability",
    }
    return fields, [labels.get(field, field) for field in fields]


def _add_graticule(fmap: Any, bounds: tuple[float, float, float, float]) -> None:
    import folium

    west, south, east, north = bounds
    group = folium.FeatureGroup(name="Latitude/longitude grid", show=True)
    lon_interval = _nice_interval(east - west)
    lat_interval = _nice_interval(north - south)
    for lon in _coordinate_values(west, east, lon_interval):
        folium.PolyLine(
            [[south, lon], [north, lon]],
            color="#111827",
            weight=1,
            opacity=0.34,
            dash_array="4,6",
            tooltip=f"Longitude {_format_lon(lon)}",
        ).add_to(group)
        folium.Marker(
            [south, lon],
            icon=folium.DivIcon(html=f'<div class="graticule-label">{_format_lon(lon)}</div>'),
        ).add_to(group)
    for lat in _coordinate_values(south, north, lat_interval):
        folium.PolyLine(
            [[lat, west], [lat, east]],
            color="#111827",
            weight=1,
            opacity=0.34,
            dash_array="4,6",
            tooltip=f"Latitude {_format_lat(lat)}",
        ).add_to(group)
        folium.Marker(
            [lat, west],
            icon=folium.DivIcon(html=f'<div class="graticule-label">{_format_lat(lat)}</div>'),
        ).add_to(group)
    group.add_to(fmap)


def _nice_interval(span: float, *, target_lines: int = 8) -> float:
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target_lines)
    exponent = math.floor(math.log10(raw))
    base = 10 ** exponent
    for multiplier in (1, 2, 5, 10):
        interval = multiplier * base
        if raw <= interval:
            return interval
    return 10 * base


def _coordinate_values(start: float, end: float, interval: float, *, max_lines: int = 60) -> list[float]:
    if interval <= 0:
        return []
    first = math.ceil(start / interval) * interval
    values: list[float] = []
    value = first
    epsilon = abs(interval) * 1e-9
    while value <= end + epsilon and len(values) <= max_lines:
        values.append(0.0 if abs(value) < epsilon else value)
        value += interval
    return values if len(values) <= max_lines else []


def _format_lon(value: float) -> str:
    direction = "E" if value >= 0 else "W"
    return f"{abs(value):.5g} deg {direction}"


def _format_lat(value: float) -> str:
    direction = "N" if value >= 0 else "S"
    return f"{abs(value):.5g} deg {direction}"


def _map_overlay_controls_html() -> str:
    return """
    <style>
      .north-arrow-control {
        position: absolute;
        top: 82px;
        right: 14px;
        z-index: 9999;
        width: 54px;
        min-height: 70px;
        border: 1px solid rgba(17, 24, 39, 0.20);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.14);
        color: #111827;
        text-align: center;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        pointer-events: none;
      }
      .north-arrow-control .needle {
        display: block;
        margin-top: 6px;
        font-size: 25px;
        line-height: 1;
        color: #ef4444;
      }
      .north-arrow-control strong {
        display: block;
        font-size: 15px;
        line-height: 1.1;
      }
      .north-arrow-control span {
        display: block;
        margin-top: 2px;
        color: #4b5563;
        font-size: 10px;
        line-height: 1.1;
      }
      .lsd-map-legend {
        position: absolute;
        right: 14px;
        bottom: 28px;
        z-index: 9999;
        border: 1px solid rgba(17, 24, 39, 0.16);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.94);
        color: #111827;
        font: 12px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 8px 10px;
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.14);
      }
      .lsd-map-legend .swatch {
        display: inline-block;
        width: 14px;
        height: 10px;
        margin-right: 6px;
        border: 2px solid #ff3b30;
        background: rgba(255, 107, 53, 0.35);
        vertical-align: -1px;
      }
      .graticule-label {
        color: #111827;
        background: rgba(255,255,255,0.76);
        border-radius: 4px;
        padding: 1px 4px;
        font: 10px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        white-space: nowrap;
        transform: translate(3px, -15px);
        box-shadow: 0 1px 4px rgba(17,24,39,0.10);
      }
    </style>
    <div class="north-arrow-control" title="Azimuth reference: north is 0 degrees">
      <span class="needle">▲</span>
      <strong>N</strong>
      <span>0 deg</span>
    </div>
    <div class="lsd-map-legend"><span class="swatch"></span>Landslide polygon</div>
    """
