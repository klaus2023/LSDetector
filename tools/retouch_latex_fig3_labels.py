from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer


TARGET = Path("/Users/zijinfu/Documents/BaiduCloud/paper/ISGDR/isgdr_overleaf_template_v2__4_/figures/fig3.png")
META_PATH = Path("/Users/zijinfu/Documents/LSDAPP/out/wuping_test2_landslide_sci_map_v3_metadata.json")
PREVIEW = Path("/Users/zijinfu/Documents/LSDAPP/out/fig3_latex_labelscale_preview.png")

FONT_REG = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")

SCALE = 1.25


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(str(path), size)


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


def detect_map_frame(im: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(im.convert("RGB"))
    dark = (arr[:, :, 0] < 35) & (arr[:, :, 1] < 35) & (arr[:, :, 2] < 35)
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    rows = np.where(row_counts > arr.shape[1] * 0.45)[0]
    cols = np.where(col_counts > arr.shape[0] * 0.45)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("Could not detect map frame.")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def world_to_px(x: float, y: float, bounds, frame) -> tuple[int, int]:
    left, top, right, bottom = frame
    minx, miny, maxx, maxy = bounds
    px = left + ((x - minx) / (maxx - minx) * (right - left))
    py = top + ((maxy - y) / (maxy - miny) * (bottom - top))
    return int(round(px)), int(round(py))


def draw_rotated_label(page: Image.Image, text: str, xy: tuple[int, int], fnt: ImageFont.FreeTypeFont) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=fnt)
    tmp = Image.new("RGBA", (bbox[2] - bbox[0] + 44, bbox[3] - bbox[1] + 44), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((tmp.width // 2, tmp.height // 2), text, font=fnt, fill=(0, 0, 0), anchor="mm")
    tmp = tmp.rotate(90, expand=True)
    page.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def clear_graticule_labels(draw: ImageDraw.ImageDraw, frame: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = frame
    draw.rectangle((left - 120, 0, right + 120, top - 5), fill=(255, 255, 255, 255))
    draw.rectangle((left - 120, bottom + 5, right + 120, bottom + 144), fill=(255, 255, 255, 255))
    draw.rectangle((0, top + 12, left - 10, bottom - 12), fill=(255, 255, 255, 255))
    draw.rectangle((right + 10, top + 12, right + 210, bottom - 12), fill=(255, 255, 255, 255))


def draw_graticule_labels(page: Image.Image, draw: ImageDraw.ImageDraw, frame, bounds, crs) -> None:
    left, top, right, bottom = frame
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
    fnt = font(round(82 * SCALE), bold=True)
    for lon in lon_values:
        label_x, _ = from_ll.transform(lon, lat_min)
        if minx <= label_x <= maxx:
            x, _ = world_to_px(label_x, miny, bounds, frame)
            label = format_dm(float(lon), True)
            draw.text((x, top - 58), label, font=fnt, fill=(0, 0, 0), anchor="mm")
            draw.text((x, bottom + 78), label, font=fnt, fill=(0, 0, 0), anchor="mm")
    for lat in lat_values:
        _, label_y = from_ll.transform(lon_min, lat)
        if miny <= label_y <= maxy:
            _, y = world_to_px(minx, label_y, bounds, frame)
            label = format_dm(float(lat), False)
            draw_rotated_label(page, label, (left - 132, y), fnt)
            draw_rotated_label(page, label, (right + 132, y), fnt)


def clear_legend_and_scale(draw: ImageDraw.ImageDraw, frame: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = frame
    # Slightly larger than the existing white boxes; confined to low-density map corners.
    draw.rectangle((left + 130, bottom - 520, left + 1890, bottom - 70), fill=(255, 255, 255, 238))
    draw.rectangle((right - 1780, bottom - 470, right - 105, bottom - 95), fill=(255, 255, 255, 238))


def draw_legend(draw: ImageDraw.ImageDraw, frame: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = frame
    x0 = left + 140
    y0 = bottom - 500
    box_w = 1740
    box_h = 420
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(255, 255, 255), outline=(45, 45, 45), width=3)
    title_font = font(round(92 * SCALE), bold=True)
    item_font = font(round(84 * SCALE), bold=True)
    draw.text((x0 + 65, y0 + 74), "Legend", font=title_font, fill=(0, 0, 0), anchor="lm")
    rows = [
        ("Wuping Benchmark Study Area", (228, 26, 28), 9),
        ("Landslide", (255, 220, 0), 6),
    ]
    y = y0 + 190
    for label, edge, width in rows:
        rect = (x0 + 65, y - 38, x0 + 205, y + 38)
        draw.rectangle(rect, fill=(255, 255, 255), outline=edge, width=width)
        draw.text((x0 + 275, y), label, font=item_font, fill=(0, 0, 0), anchor="lm")
        y += 112


def draw_scale_bar(draw: ImageDraw.ImageDraw, frame: tuple[int, int, int, int], bounds) -> None:
    left, top, right, bottom = frame
    map_width_m = bounds[2] - bounds[0]
    length_m = 5000 if map_width_m >= 15000 else 3000
    segment_m = 1000
    px_per_m = (right - left) / map_width_m
    length_px = int(round(length_m * px_per_m))
    seg_px = int(round(segment_m * px_per_m))
    x0 = right - length_px - 260
    y0 = bottom - 270
    h = 42
    draw.rectangle((x0 - 78, y0 - 205, x0 + length_px + 155, y0 + h + 90), fill=(255, 255, 255), outline=None)
    label_font = font(round(88 * SCALE), bold=True)
    draw.text((x0, y0 - 104), "0", font=label_font, fill=(0, 0, 0), anchor="mm")
    draw.text((x0 + length_px, y0 - 104), f"{int(length_m / 1000)} km", font=label_font, fill=(0, 0, 0), anchor="mm")
    segments = int(length_m / segment_m)
    for idx in range(segments):
        x1 = x0 + idx * seg_px
        x2 = x0 + (idx + 1) * seg_px if idx < segments - 1 else x0 + length_px
        fill = (0, 0, 0) if idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((x1, y0, x2, y0 + h), fill=fill, outline=(0, 0, 0), width=3)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"{TARGET.name}.bak_{timestamp}_before_label_scale")
    shutil.copy2(TARGET, backup)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    bounds = tuple(float(v) for v in meta["raster_window_bounds"])
    crs = meta["crs"]

    page = Image.open(TARGET).convert("RGBA")
    frame = detect_map_frame(page)
    draw = ImageDraw.Draw(page, "RGBA")
    clear_graticule_labels(draw, frame)
    clear_legend_and_scale(draw, frame)
    draw_graticule_labels(page, draw, frame, bounds, crs)
    draw_legend(draw, frame)
    draw_scale_bar(draw, frame, bounds)

    out = page.convert("RGB")
    out.save(TARGET, "PNG", optimize=True)
    out.save(PREVIEW, "PNG", optimize=True)
    check = Image.open(TARGET)
    print(
        {
            "target": str(TARGET),
            "backup": str(backup),
            "preview": str(PREVIEW),
            "size": check.size,
            "frame": frame,
            "scale_factor": SCALE,
        }
    )


if __name__ == "__main__":
    main()
