"""Quicklook visualizations for landslide prediction outputs."""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any
import zlib

from lsd_app.config import PlatformConfig
from lsd_app.core.channels import extract_rgb_bands
from lsd_app.core.metrics import (
    EvaluationError,
    MaskGrid,
    configured_truth_path,
    load_truth_mask,
)
from lsd_app.core.raster import require_rasterio


TP_COLOR = (255, 69, 0)
FP_COLOR = (0, 102, 255)
FN_COLOR = (255, 230, 0)
PREDICTION_COLOR = TP_COLOR
DEFAULT_MAX_DIMENSION = 1024


class VisualizationError(RuntimeError):
    """Raised when an overlay cannot be generated."""


@dataclass(frozen=True)
class VisualizationResult:
    """Summary returned by the visualize stage."""

    overlay_path: Path
    width: int
    height: int
    scale: float
    truth_used: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result mapping."""

        return {
            "overlay_path": str(self.overlay_path),
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "truth_used": self.truth_used,
            "message": self.message,
        }


def run_visualize(config: PlatformConfig) -> VisualizationResult:
    """Create `visualizations/overlay.png` as a downsampled prediction/truth overlay."""

    rasterio = require_rasterio()
    import numpy as np

    mask_path = config.outputs.rasters / "mask.tif"
    if not mask_path.exists():
        raise VisualizationError(f"Prediction mask does not exist: {mask_path}. Run `lsd mosaic --config ...` first.")

    try:
        with rasterio.open(mask_path) as mask_src:
            out_width, out_height, scale = _overview_shape(mask_src.width, mask_src.height)
            overview_transform = mask_src.transform * mask_src.transform.scale(
                mask_src.width / out_width,
                mask_src.height / out_height,
            )
            grid = MaskGrid(width=out_width, height=out_height, transform=overview_transform, crs=mask_src.crs)
            prediction = mask_src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.nearest) > 0
            base = _read_base_rgb(config, fallback_src=mask_src, out_width=out_width, out_height=out_height)

        truth_used = False
        message = "Prediction overlay generated without truth data."
        truth = np.zeros((out_height, out_width), dtype=bool)
        if configured_truth_path(config) is not None:
            try:
                truth, _, truth_source = load_truth_mask(config, grid=grid)
            except EvaluationError as exc:
                raise VisualizationError(str(exc)) from exc
            truth_used = True
            message = f"Prediction/truth overlay generated with truth: {truth_source}"

        overlay = _compose_overlay(base, prediction=prediction, truth=truth, truth_used=truth_used)
        config.outputs.visualizations.mkdir(parents=True, exist_ok=True)
        overlay_path = config.outputs.visualizations / "overlay.png"
        _write_png_rgb(overlay_path, overlay)
    except VisualizationError:
        raise
    except Exception as exc:
        raise VisualizationError(f"Failed to generate overlay: {exc}") from exc

    return VisualizationResult(
        overlay_path=overlay_path,
        width=out_width,
        height=out_height,
        scale=scale,
        truth_used=truth_used,
        message=message,
    )


def _overview_shape(width: int, height: int, *, max_dimension: int = DEFAULT_MAX_DIMENSION) -> tuple[int, int, float]:
    largest = max(width, height)
    if largest <= max_dimension:
        return width, height, 1.0
    scale = max_dimension / largest
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    return out_width, out_height, scale


def _read_base_rgb(config: PlatformConfig, *, fallback_src: Any, out_width: int, out_height: int) -> Any:
    import numpy as np

    source_path = config.input_raster_path
    if source_path.exists():
        try:
            rasterio = require_rasterio()
            with rasterio.open(source_path) as src:
                indexes = list(range(1, src.count + 1))
                if not indexes:
                    raise VisualizationError(f"Input raster has no bands: {source_path}")
                data = src.read(indexes, out_shape=(len(indexes), out_height, out_width))
                data = extract_rgb_bands(data, config.RGB_bands, layout="chw")
                return _bands_to_rgb(data)
        except VisualizationError:
            raise
        except Exception:
            pass

    data = fallback_src.read(1, out_shape=(out_height, out_width))
    return _bands_to_rgb(data[np.newaxis, ...])


def _bands_to_rgb(data: Any) -> Any:
    import numpy as np

    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise VisualizationError("Expected source raster data with shape (bands, height, width).")
    if values.shape[0] == 1:
        normalized = _normalize_band(values[0])
        return np.stack([normalized, normalized, normalized], axis=-1)
    channels = [_normalize_band(values[index]) for index in range(min(3, values.shape[0]))]
    while len(channels) < 3:
        channels.append(channels[-1])
    return np.stack(channels[:3], axis=-1)


def _normalize_band(band: Any) -> Any:
    import numpy as np

    values = np.asarray(band, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], [2, 98])
    if high <= low:
        high = float(values[finite].max())
        low = float(values[finite].min())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - low) / (high - low)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def _compose_overlay(base: Any, *, prediction: Any, truth: Any, truth_used: bool) -> Any:
    import numpy as np

    image = np.asarray(base, dtype=np.float32).copy()
    pred = np.asarray(prediction, dtype=bool)
    truth_mask = np.asarray(truth, dtype=bool)
    alpha = 0.65

    if truth_used:
        _blend(image, pred & truth_mask, TP_COLOR, alpha)
        _blend(image, pred & ~truth_mask, FP_COLOR, alpha)
        _blend(image, ~pred & truth_mask, FN_COLOR, alpha)
    else:
        _blend(image, pred, PREDICTION_COLOR, alpha)
    return np.clip(image, 0, 255).astype(np.uint8)


def _blend(image: Any, mask: Any, color: tuple[int, int, int], alpha: float) -> None:
    import numpy as np

    if not np.any(mask):
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[mask] = (1.0 - alpha) * image[mask] + alpha * color_array


def _write_png_rgb(path: Path, image: Any) -> None:
    """Write an RGB uint8 image as PNG using only the standard library."""

    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise VisualizationError("PNG writer expects an RGB image with shape (height, width, 3).")
    height, width, _ = array.shape
    raw_rows = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
