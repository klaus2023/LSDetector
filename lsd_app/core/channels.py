"""RGB band selection helpers for model-input normalization."""

from __future__ import annotations

import re
from typing import Any


RGB_BAND_PRESETS = (
    "RGB_bands: 1,2,3",
    "RGB_bands: 3,2,1",
    "RGB_bands: 4,3,2",
    "RGB_bands: 6,4,2",
)
DEFAULT_RGB_BANDS = "RGB_bands: 1,2,3"
MODEL_INPUT_RGB_BANDS = DEFAULT_RGB_BANDS
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


class RGBBandsError(ValueError):
    """Raised when an RGB_bands value or array layout is invalid."""


def normalize_rgb_bands(value: Any) -> str:
    """Return the canonical ``RGB_bands: R,G,B`` value."""

    if value is None or str(value).strip() == "":
        raise RGBBandsError("Missing RGB_bands. Expected format: RGB_bands: R,G,B.")
    text = str(value).strip()
    if text.lower() in {"rgb_first3", "bgr_first3", "rgb", "bgr", "gaofen", "channel_order"}:
        raise RGBBandsError(
            f"Invalid RGB_bands {value!r}. Legacy channel_order values are no longer supported; "
            "use RGB_bands: R,G,B."
        )
    if ":" in text:
        key, _, raw_bands = text.partition(":")
        normalized_key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_key != "rgb_bands":
            raise RGBBandsError(f"Invalid RGB_bands key {key!r}. Expected RGB_bands: R,G,B.")
    else:
        raw_bands = text
    parts = [part.strip() for part in raw_bands.split(",") if part.strip()]
    if len(parts) != 3:
        raise RGBBandsError(f"Invalid RGB_bands {value!r}. Expected exactly three 1-based band indexes.")
    indexes: list[int] = []
    for part in parts:
        if not re.fullmatch(r"[1-9][0-9]*", part):
            raise RGBBandsError(f"Invalid RGB_bands {value!r}. Band indexes must be positive 1-based integers.")
        indexes.append(int(part))
    return f"RGB_bands: {indexes[0]},{indexes[1]},{indexes[2]}"


def rgb_bands_to_indexes(value: Any) -> tuple[int, int, int]:
    """Return 1-based RGB band indexes from a canonical or shorthand value."""

    normalized = normalize_rgb_bands(value)
    _, _, raw_bands = normalized.partition(":")
    red, green, blue = (int(part.strip()) for part in raw_bands.split(","))
    return red, green, blue


def rgb_bands_to_display(value: Any) -> str:
    """Return compact display text for repository tables."""

    red, green, blue = rgb_bands_to_indexes(value)
    return f"{red},{green},{blue}"


def rgb_bands_to_class(value: Any) -> str:
    """Return a compact class label for repository tables."""

    return f"RGB bands {rgb_bands_to_display(value)}"


def extract_rgb_bands(array: Any, RGB_bands: Any, *, layout: str = "chw") -> Any:
    """Extract the configured source bands into a 3-channel RGB array."""

    import numpy as np

    values = np.asarray(array)
    if values.ndim != 3:
        raise RGBBandsError(f"Expected a 3D array for RGB band extraction, got shape {values.shape}.")
    if layout not in {"chw", "hwc"}:
        raise RGBBandsError(f"Unsupported array layout {layout!r}; expected 'chw' or 'hwc'.")

    indexes = rgb_bands_to_indexes(RGB_bands)
    channel_axis = 0 if layout == "chw" else 2
    band_count = int(values.shape[channel_axis])
    missing = [index for index in indexes if index > band_count]
    if missing:
        raise RGBBandsError(
            f"RGB_bands {normalize_rgb_bands(RGB_bands)} requires band(s) {missing}, "
            f"but the raster/sample only has {band_count} band(s)."
        )
    zero_based = [index - 1 for index in indexes]
    if layout == "chw":
        return values[zero_based, :, :].copy()
    return values[:, :, zero_based].copy()
