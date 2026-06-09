from __future__ import annotations

from datetime import date
from itertools import count
from pathlib import Path
import json
import math
import warnings

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from pyproj import Transformer


ROOT = Path("/home/fuzijin/data/Wuping")
OUT_DIR = ROOT / "figures"
TMP_DIR = OUT_DIR / "_tmp"
AOI_PATH = ROOT / "TSFT_AOI/test2.shp"
LANDSLIDE_PATH = ROOT / "label/label/2024年滑坡面.shp"
RASTER_GLOB = "tile_*/tile_*_mosaic.tif"

OUTPUT_PREFIX = "wuping_test2_landslide_sci_map"

PAGE_W, PAGE_H = 7014, 4962  # A4 landscape at 600 dpi.
MAP_W, MAP_H = 5800, 4050
MAP_X = (PAGE_W - MAP_W) // 2
MAP_Y = 455

RGB_BANDS = (6, 4, 2)  # PlanetScope SuperDove: red, green, blue.
STRETCH_PERCENT_CLIP = 0.25
DISPLAY_GAMMA = 2.4

FONT_REG = TMP_DIR / "Times_New_Roman.ttf"
FONT_BOLD = TMP_DIR / "Times_New_Roman_Bold.ttf"
if not FONT_REG.exists():
    FONT_REG = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf")
if not FONT_BOLD.exists():
    FONT_BOLD = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf")


def next_output_stem() -> str:
    for idx in count(1):
        stem = f"{OUTPUT_PREFIX}_v{idx}"
        if not any((OUT_DIR / f"{stem}{suffix}").exists() for suffix in (".png", ".pdf", "_metadata.json")):
            return stem
    raise RuntimeError("Could not allocate output name.")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:
        from shapely.validation import make_valid

        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
        )
    except Exception:
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: geom.buffer(0) if geom is not None and not geom.is_valid else geom
        )
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def union_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return unary_union(list(gdf.geometry))


def expand_bounds_to_aspect(bounds, target_aspect: float, pad_ratio: float = 0.055):
    minx, miny, maxx, maxy = map(float, bounds)
    width = maxx - minx
    height = maxy - miny
    pad = max(width, height) * pad_ratio
    minx -= pad
    maxx += pad
    miny -= pad
    maxy += pad
    width = maxx - minx
    height = maxy - miny
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    current = width / height
    if current < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def intersection_area(bounds_a, bounds_b) -> float:
    aminx, aminy, amaxx, amaxy = bounds_a
    bminx, bminy, bmaxx, bmaxy = bounds_b
    ix = max(0.0, min(amaxx, bmaxx) - max(aminx, bminx))
    iy = max(0.0, min(amaxy, bmaxy) - max(aminy, bminy))
    return ix * iy


def choose_basemap(bounds, crs):
    candidates = []
    target_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    for path in sorted(ROOT.glob(RASTER_GLOB)):
        if any(token in path.name for token in ("qa_mask", "date", "provenance")):
            continue
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                raster_bounds = src.bounds
                if src.crs != crs:
                    continue
                area = intersection_area(bounds, raster_bounds)
                if area <= 0:
                    continue
                candidates.append(
                    {
                        "path": path,
                        "bounds": tuple(raster_bounds),
                        "overlap_area": float(area),
                        "coverage_ratio": float(area / target_area),
                        "bands": src.count,
                        "resolution": tuple(map(float, src.res)),
                    }
                )
        except Exception as exc:
            print(f"Skipped raster {path}: {exc}")
    if not candidates:
        raise RuntimeError("No valid raster basemap overlaps the AOI extent.")
    candidates.sort(key=lambda item: (item["coverage_ratio"], item["overlap_area"], -len(str(item["path"]))), reverse=True)
    return candidates[0], candidates


def clamp_bounds_to_raster(bounds, raster_bounds):
    minx = max(bounds[0], raster_bounds[0])
    miny = max(bounds[1], raster_bounds[1])
    maxx = min(bounds[2], raster_bounds[2])
    maxy = min(bounds[3], raster_bounds[3])
    if minx >= maxx or miny >= maxy:
        raise RuntimeError("Map extent does not overlap selected raster.")
    return minx, miny, maxx, maxy


def stretch_rgb(arr: np.ma.MaskedArray, nodata) -> Image.Image:
    rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
    for i in range(3):
        band = np.ma.filled(arr[i].astype("float32"), np.nan)
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)
        valid = np.isfinite(band) & (band > 0)
        if valid.any():
            lo, hi = np.nanpercentile(
                band[valid],
                [STRETCH_PERCENT_CLIP, 100 - STRETCH_PERCENT_CLIP],
            )
            band = (band - lo) / max(hi - lo, 1e-6)
            band = np.power(np.clip(band, 0, 1), 1.0 / DISPLAY_GAMMA)
        else:
            band = np.zeros_like(band)
        rgb[..., i] = np.nan_to_num(np.clip(band, 0, 1), nan=0.0)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"), "RGB")
    img = ImageEnhance.Contrast(img).enhance(1.04)
    img = ImageEnhance.Color(img).enhance(1.02)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=2))


def read_basemap(raster_path: Path, bounds):
    with rasterio.open(raster_path) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        indexes = [band for band in RGB_BANDS if band <= src.count]
        if len(indexes) != 3:
            raise RuntimeError(f"Raster {raster_path} does not contain requested RGB bands {RGB_BANDS}.")
        arr = src.read(indexes, window=window, masked=True)
        raster_window_bounds = rasterio.windows.bounds(window, src.transform)
        raster_window_shape = (int(window.height), int(window.width))
        img = stretch_rgb(arr, src.nodata)
    if img.size != (MAP_W, MAP_H):
        img = img.resize((MAP_W, MAP_H), Image.Resampling.LANCZOS)
    return img.convert("RGBA"), raster_window_bounds, raster_window_shape, indexes


def world_to_page_px(x: float, y: float, bounds) -> tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    px = MAP_X + ((x - minx) / (maxx - minx) * MAP_W)
    py = MAP_Y + ((maxy - y) / (maxy - miny) * MAP_H)
    return int(round(px)), int(round(py))


def exterior_rings(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom.exterior
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield poly.exterior
    elif isinstance(geom, GeometryCollection):
        for child in geom.geoms:
            yield from exterior_rings(child)


def polygon_rings(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom.exterior
        yield from geom.interiors
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield poly.exterior
            yield from poly.interiors
    elif isinstance(geom, GeometryCollection):
        for child in geom.geoms:
            yield from polygon_rings(child)


def draw_polygon_layer(draw, geoms, bounds, outline, width, fill=None, interiors=True):
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        if fill is not None:
            polys = []
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            for poly in polys:
                pts = [world_to_page_px(x, y, bounds) for x, y in poly.exterior.coords]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=fill)
        rings = polygon_rings(geom) if interiors else exterior_rings(geom)
        for ring in rings:
            pts = [world_to_page_px(x, y, bounds) for x, y in ring.coords]
            if len(pts) >= 2:
                draw.line(pts, fill=outline, width=width, joint="curve")


def draw_text_box(draw, xy, text, fnt, fill=(0, 0, 0), anchor="mm", pad=8, box_fill=(255, 255, 255, 218)):
    bbox = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rounded_rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), radius=3, fill=box_fill)
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def graticule_step(span: float) -> float:
    if span <= 0.08:
        return 0.02
    if span <= 0.16:
        return 0.04
    if span <= 0.28:
        return 0.05
    return 0.1


def format_dm(value: float, lon: bool) -> str:
    hemi = ("E" if value >= 0 else "W") if lon else ("N" if value >= 0 else "S")
    value = abs(value)
    deg = int(math.floor(value))
    minutes = int(round((value - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}°{minutes:02d}′{hemi}"


def line_parts(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for child in geom.geoms:
            parts.extend(line_parts(child))
        return parts
    return []


def draw_rotated_label(page, text, xy, fnt):
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=fnt)
    tmp = Image.new("RGBA", (bbox[2] - bbox[0] + 32, bbox[3] - bbox[1] + 32), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((tmp.width // 2, tmp.height // 2), text, font=fnt, fill=(0, 0, 0), anchor="mm")
    tmp = tmp.rotate(90, expand=True)
    page.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def draw_graticule(page, draw, bounds, crs):
    minx, miny, maxx, maxy = bounds
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    from_ll = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons, lats = to_ll.transform([minx, maxx, maxx, minx], [miny, miny, maxy, maxy])
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_step = graticule_step(lon_max - lon_min)
    lat_step = graticule_step(lat_max - lat_min)
    lon_values = np.arange(math.ceil(lon_min / lon_step) * lon_step, lon_max, lon_step)
    lat_values = np.arange(math.ceil(lat_min / lat_step) * lat_step, lat_max, lat_step)
    bbox = box(*bounds)
    grid_light = (255, 255, 255, 135)
    grid_dark = (80, 80, 80, 105)
    label_font = font(82, bold=True)
    tick = 26

    for lon in lon_values:
        ys = np.linspace(lat_min - lat_step, lat_max + lat_step, 180)
        xs = np.full_like(ys, lon)
        gx, gy = from_ll.transform(xs, ys)
        line = LineString(zip(gx, gy)).intersection(bbox)
        for part in line_parts(line):
            pts = [world_to_page_px(x, y, bounds) for x, y in part.coords]
            if len(pts) < 2:
                continue
            draw.line(pts, fill=grid_light, width=4)
            draw.line(pts, fill=grid_dark, width=2)
        label_x, _ = from_ll.transform(lon, lat_min)
        if minx <= label_x <= maxx:
            x, _ = world_to_page_px(label_x, miny, bounds)
            draw.line((x, MAP_Y + MAP_H, x, MAP_Y + MAP_H + tick), fill=(0, 0, 0), width=3)
            draw.line((x, MAP_Y - tick, x, MAP_Y), fill=(0, 0, 0), width=3)
            draw.text((x, MAP_Y + MAP_H + 72), format_dm(lon, True), font=label_font, fill=(0, 0, 0), anchor="mm")
            draw.text((x, MAP_Y - 66), format_dm(lon, True), font=label_font, fill=(0, 0, 0), anchor="mm")

    for lat in lat_values:
        xs = np.linspace(lon_min - lon_step, lon_max + lon_step, 180)
        ys = np.full_like(xs, lat)
        gx, gy = from_ll.transform(xs, ys)
        line = LineString(zip(gx, gy)).intersection(bbox)
        for part in line_parts(line):
            pts = [world_to_page_px(x, y, bounds) for x, y in part.coords]
            if len(pts) < 2:
                continue
            draw.line(pts, fill=grid_light, width=4)
            draw.line(pts, fill=grid_dark, width=2)
        _, label_y = from_ll.transform(lon_min, lat)
        if miny <= label_y <= maxy:
            _, y = world_to_page_px(minx, label_y, bounds)
            draw.line((MAP_X - tick, y, MAP_X, y), fill=(0, 0, 0), width=3)
            draw.line((MAP_X + MAP_W, y, MAP_X + MAP_W + tick, y), fill=(0, 0, 0), width=3)
            label = format_dm(lat, False)
            draw_rotated_label(page, label, (MAP_X - 116, y), label_font)
            draw_rotated_label(page, label, (MAP_X + MAP_W + 116, y), label_font)


def draw_scale_bar(draw, bounds):
    map_width_m = bounds[2] - bounds[0]
    length_m = 5000 if map_width_m >= 15000 else 3000
    segment_m = 1000
    px_per_m = MAP_W / map_width_m
    length_px = int(round(length_m * px_per_m))
    seg_px = int(round(segment_m * px_per_m))
    x0 = MAP_X + MAP_W - length_px - 250
    y0 = MAP_Y + MAP_H - 265
    h = 36
    draw.rounded_rectangle((x0 - 64, y0 - 178, x0 + length_px + 132, y0 + h + 92), radius=4, fill=(255, 255, 255, 226))
    segments = int(length_m / segment_m)
    for idx in range(segments):
        x1 = x0 + idx * seg_px
        x2 = x0 + (idx + 1) * seg_px if idx < segments - 1 else x0 + length_px
        fill = (0, 0, 0) if idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((x1, y0, x2, y0 + h), fill=fill, outline=(0, 0, 0), width=3)
    label_font = font(88, bold=True)
    draw.text((x0, y0 - 82), "0", font=label_font, fill=(0, 0, 0), anchor="mm")
    draw.text((x0 + length_px, y0 - 82), f"{int(length_m / 1000)} km", font=label_font, fill=(0, 0, 0), anchor="mm")


def draw_north_arrow(draw):
    cx = MAP_X + 310
    top = MAP_Y + 285
    size = 350
    outer = [(cx, top), (cx - 124, top + size), (cx, top + int(size * 0.71)), (cx + 124, top + size)]
    left = [(cx, top), (cx - 124, top + size), (cx, top + int(size * 0.71))]
    right = [(cx, top), (cx, top + int(size * 0.71)), (cx + 124, top + size)]
    draw.polygon(outer, fill=(255, 255, 255), outline=(255, 255, 255))
    draw.line(outer + [outer[0]], fill=(255, 255, 255), width=18)
    draw.polygon(left, fill=(0, 0, 0), outline=(0, 0, 0))
    draw.polygon(right, fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line(outer + [outer[0]], fill=(0, 0, 0), width=5)
    draw_text_box(draw, (cx, top - 58), "N", font(105, bold=True), anchor="mm", pad=4)


def draw_legend(draw):
    box_w, box_h = 1620, 385
    x0 = MAP_X + 130
    y0 = MAP_Y + MAP_H - box_h - 105
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=6, fill=(255, 255, 255, 226), outline=(45, 45, 45), width=3)
    title_font = font(92, bold=True)
    item_font = font(84, bold=True)
    draw.text((x0 + 60, y0 + 70), "Legend", font=title_font, fill=(0, 0, 0), anchor="lm")
    rows = [
        ("Wuping Benchmark Study Area", (228, 26, 28, 255), None, 8),
        ("Landslide", (255, 220, 0, 255), None, 5),
    ]
    y = y0 + 168
    for label, edge, fill, width in rows:
        rect = (x0 + 66, y - 34, x0 + 200, y + 34)
        draw.rectangle(rect, fill=(255, 255, 255, 0) if fill is None else fill, outline=edge, width=width)
        draw.text((x0 + 260, y), label, font=item_font, fill=(0, 0, 0), anchor="lm")
        y += 104


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = next_output_stem()
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    basemap_path = OUT_DIR / f"{stem}_basemap.png"
    meta_path = OUT_DIR / f"{stem}_metadata.json"
    clipped_path = TMP_DIR / f"{stem}_landslides_clipped_to_test2.gpkg"

    aoi_raw = gpd.read_file(AOI_PATH)
    landslide_raw = gpd.read_file(LANDSLIDE_PATH)
    if aoi_raw.empty:
        raise SystemExit(f"AOI layer has 0 features: {AOI_PATH}")
    if aoi_raw.crs is None or landslide_raw.crs is None:
        raise SystemExit("CRS missing in AOI or landslide layer.")

    aoi = fix_geometries(aoi_raw)
    landslide = fix_geometries(landslide_raw)
    if aoi.empty:
        raise SystemExit("AOI has no valid non-empty geometries after repair.")

    if landslide.crs != aoi.crs:
        landslide = landslide.to_crs(aoi.crs)

    aoi_geom = union_geometry(aoi)
    intersects_mask = landslide.intersects(aoi_geom)
    landslide_intersecting = landslide[intersects_mask].copy()
    if not landslide_intersecting.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            landslide_intersecting["geometry"] = landslide_intersecting.geometry.intersection(aoi_geom)
        landslide_clipped = landslide_intersecting[
            landslide_intersecting.geometry.notna() & ~landslide_intersecting.geometry.is_empty
        ].copy()
    else:
        landslide_clipped = landslide_intersecting

    if clipped_path.exists():
        clipped_path.unlink()
    landslide_clipped.to_file(clipped_path, driver="GPKG")

    target_bounds = expand_bounds_to_aspect(aoi.total_bounds, MAP_W / MAP_H, pad_ratio=0.055)
    selected, candidates = choose_basemap(target_bounds, aoi.crs)
    map_bounds = clamp_bounds_to_raster(target_bounds, selected["bounds"])
    basemap, raster_bounds, raster_window_shape, rgb_bands = read_basemap(selected["path"], map_bounds)

    page = Image.new("RGBA", (PAGE_W, PAGE_H), (255, 255, 255, 255))
    page.alpha_composite(basemap, (MAP_X, MAP_Y))
    draw = ImageDraw.Draw(page, "RGBA")

    draw.rectangle((MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H), outline=(0, 0, 0), width=7)
    draw_graticule(page, draw, raster_bounds, aoi.crs)

    draw_polygon_layer(draw, landslide_clipped.geometry, raster_bounds, outline=(255, 225, 0, 255), width=4, fill=None)
    draw_polygon_layer(draw, aoi.geometry, raster_bounds, outline=(255, 255, 255, 230), width=20, fill=None, interiors=False)
    draw_polygon_layer(draw, aoi.geometry, raster_bounds, outline=(228, 26, 28, 255), width=13, fill=None, interiors=False)

    draw_north_arrow(draw)
    draw_scale_bar(draw, raster_bounds)
    draw_legend(draw)

    title = "Wuping Test Area Landslide Inventory (2024)"
    draw.text((PAGE_W // 2, 202), title, font=font(108, bold=True), fill=(0, 0, 0), anchor="mm")
    note = (
        "Projection: WGS 84 / UTM zone 50N (EPSG:32650). "
        f"Basemap: PlanetScope mosaic, 3 m, RGB {rgb_bands[0]}/{rgb_bands[1]}/{rgb_bands[2]}. "
        "Landslide polygons clipped to the test2 study area."
    )
    draw.text((PAGE_W // 2, PAGE_H - 128), note, font=font(55), fill=(40, 40, 40), anchor="mm")

    page_rgb = page.convert("RGB")
    page_rgb.save(png_path, dpi=(600, 600), quality=96)
    page_rgb.save(pdf_path, "PDF", resolution=600.0)
    basemap.save(basemap_path)

    metadata = {
        "inputs": {
            "aoi": str(AOI_PATH),
            "landslides": str(LANDSLIDE_PATH),
            "selected_basemap": str(selected["path"]),
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "basemap_preview": str(basemap_path),
            "landslides_clipped_gpkg": str(clipped_path),
            "metadata": str(meta_path),
        },
        "crs": aoi.crs.to_string(),
        "aoi_feature_count_raw": int(len(aoi_raw)),
        "aoi_feature_count_valid": int(len(aoi)),
        "landslide_feature_count_raw": int(len(landslide_raw)),
        "landslide_feature_count_valid_after_repair": int(len(landslide)),
        "landslide_features_intersecting_aoi": int(len(landslide_intersecting)),
        "landslide_features_displayed_after_clip": int(len(landslide_clipped)),
        "landslide_clip_rule": "Landslide polygons were reprojected to the AOI CRS and geometrically intersected with test2; only clipped in-AOI parts are drawn.",
        "target_bounds": list(map(float, target_bounds)),
        "raster_window_bounds": list(map(float, raster_bounds)),
        "raster_window_shape_pixels": list(map(int, raster_window_shape)),
        "rgb_bands_1_based": list(rgb_bands),
        "raster_candidates": [
            {
                "path": str(item["path"]),
                "coverage_ratio": item["coverage_ratio"],
                "overlap_area": item["overlap_area"],
                "bands": item["bands"],
                "resolution": item["resolution"],
            }
            for item in candidates
        ],
        "layout": {
            "page": "A4 landscape",
            "dpi": 600,
            "elements": ["map frame", "legend", "scale bar", "north arrow", "degree-minute graticule", "projection/data note"],
            "styles": {
                "study_area_boundary": "white casing plus red outline, no fill, exterior rings only",
                "landslides": "thin yellow no-fill polygon outlines clipped to AOI",
                "basemap": f"full-resolution raster window, RGB {RGB_BANDS[0]}/{RGB_BANDS[1]}/{RGB_BANDS[2]}, {STRETCH_PERCENT_CLIP}% two-tail percent clip, gamma {DISPLAY_GAMMA}, light sharpening",
                "legend_and_scale_text": "larger bold black labels for print readability",
                "graticule_labels": "larger bold degree-minute longitude/latitude labels",
            },
        },
        "raster_display": {
            "rgb_bands_1_based": list(RGB_BANDS),
            "percent_clip_each_tail": STRETCH_PERCENT_CLIP,
            "gamma": DISPLAY_GAMMA,
            "arcgis_pro_reference": "Approximates ArcGIS Pro RGB Band_6/Band_4/Band_2 with min/max 0.250 and gamma 2.4 from user screenshot.",
        },
        "created_at": date.today().isoformat(),
        "backend": "Python GeoPandas/Rasterio/PIL fallback; QGIS/PyQGIS command-line tools were not available in WSL PATH.",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
