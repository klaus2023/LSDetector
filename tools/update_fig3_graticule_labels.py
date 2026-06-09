from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer


ROOT = Path("/home/fuzijin/data/Wuping")
FIG_DIR = ROOT / "figures"
SRC_PNG = FIG_DIR / "wuping_test2_landslide_sci_map_v3.png"
META_PATH = FIG_DIR / "wuping_test2_landslide_sci_map_v3_metadata.json"
OUT_JPG = FIG_DIR / "fig3.jpg"
OUT_PREVIEW = FIG_DIR / "fig3_graticule_larger_preview.png"
TMP_DIR = FIG_DIR / "_tmp"

PAGE_W, PAGE_H = 7014, 4962
MAP_W, MAP_H = 5800, 4050
MAP_X = (PAGE_W - MAP_W) // 2
MAP_Y = 455
MAP_RIGHT = MAP_X + MAP_W
MAP_BOTTOM = MAP_Y + MAP_H

FONT_BOLD = TMP_DIR / "Times_New_Roman_Bold.ttf"
if not FONT_BOLD.exists():
    FONT_BOLD = Path("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD), size)


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


def world_to_page_px(x: float, y: float, bounds) -> tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    px = MAP_X + ((x - minx) / (maxx - minx) * MAP_W)
    py = MAP_Y + ((maxy - y) / (maxy - miny) * MAP_H)
    return int(round(px)), int(round(py))


def draw_rotated_label(page: Image.Image, text: str, xy: tuple[int, int], fnt: ImageFont.FreeTypeFont) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=fnt)
    tmp = Image.new("RGBA", (bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 40), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((tmp.width // 2, tmp.height // 2), text, font=fnt, fill=(0, 0, 0), anchor="mm")
    tmp = tmp.rotate(90, expand=True)
    page.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    bounds = tuple(float(v) for v in meta["raster_window_bounds"])
    crs = meta["crs"]

    page = Image.open(SRC_PNG).convert("RGBA")
    if page.size != (PAGE_W, PAGE_H):
        raise RuntimeError(f"Unexpected source page size: {page.size}")
    draw = ImageDraw.Draw(page, "RGBA")

    # Clear only the white page-margin label strips. Map frame, ticks, grid, and map content remain untouched.
    draw.rectangle((MAP_X - 80, MAP_Y - 150, MAP_RIGHT + 80, MAP_Y - 30), fill=(255, 255, 255, 255))
    draw.rectangle((MAP_X - 80, MAP_BOTTOM + 35, MAP_RIGHT + 80, MAP_BOTTOM + 185), fill=(255, 255, 255, 255))
    draw.rectangle((MAP_X - 260, MAP_Y + 30, MAP_X - 38, MAP_BOTTOM - 30), fill=(255, 255, 255, 255))
    draw.rectangle((MAP_RIGHT + 38, MAP_Y + 30, MAP_RIGHT + 260, MAP_BOTTOM - 30), fill=(255, 255, 255, 255))

    fnt = font(94)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    from_ll = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    minx, miny, maxx, maxy = bounds
    lons, lats = to_ll.transform([minx, maxx, maxx, minx], [miny, miny, maxy, maxy])
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_step = graticule_step(lon_max - lon_min)
    lat_step = graticule_step(lat_max - lat_min)
    lon_values = np.arange(math.ceil(lon_min / lon_step) * lon_step, lon_max, lon_step)
    lat_values = np.arange(math.ceil(lat_min / lat_step) * lat_step, lat_max, lat_step)

    for lon in lon_values:
        label_x, _ = from_ll.transform(lon, lat_min)
        if minx <= label_x <= maxx:
            x, _ = world_to_page_px(label_x, miny, bounds)
            label = format_dm(float(lon), True)
            draw.text((x, MAP_Y - 82), label, font=fnt, fill=(0, 0, 0), anchor="mm")
            draw.text((x, MAP_BOTTOM + 105), label, font=fnt, fill=(0, 0, 0), anchor="mm")

    for lat in lat_values:
        _, label_y = from_ll.transform(lon_min, lat)
        if miny <= label_y <= maxy:
            _, y = world_to_page_px(minx, label_y, bounds)
            label = format_dm(float(lat), False)
            draw_rotated_label(page, label, (MAP_X - 136, y), fnt)
            draw_rotated_label(page, label, (MAP_RIGHT + 136, y), fnt)

    page_rgb = page.convert("RGB")
    page_rgb.save(OUT_PREVIEW, dpi=(600, 600), quality=96)
    target = (round(PAGE_W * 800 / 600), round(PAGE_H * 800 / 600))
    jpg = page_rgb.resize(target, Image.Resampling.LANCZOS)
    jpg.save(OUT_JPG, "JPEG", quality=95, subsampling=0, optimize=True, dpi=(800, 800))

    check = Image.open(OUT_JPG)
    print(
        {
            "output": str(OUT_JPG),
            "preview": str(OUT_PREVIEW),
            "size": check.size,
            "dpi": check.info.get("dpi"),
            "graticule_label_font": "Times New Roman Bold 94 at 600 dpi source, resampled to 800 dpi",
        }
    )


if __name__ == "__main__":
    main()
