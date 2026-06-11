"""Raster I/O helpers for the prepare stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RasterError(RuntimeError):
    """Raised when raster metadata or I/O cannot be read."""


class RasterDependencyError(RasterError):
    """Raised when optional geospatial dependencies are unavailable."""


@dataclass(frozen=True)
class RasterMetadata:
    """Serializable metadata extracted from a GeoTIFF."""

    path: Path
    width: int
    height: int
    count: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | int | str | None
    nodatavals: tuple[float | int | str | None, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metadata mapping."""

        data = asdict(self)
        data["path"] = str(self.path)
        data["transform"] = list(self.transform)
        data["bounds"] = list(self.bounds)
        data["nodatavals"] = list(self.nodatavals)
        return data


def require_rasterio() -> Any:
    """Import rasterio lazily so non-geospatial commands stay lightweight."""

    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise RasterDependencyError(
            "The prepare stage requires rasterio. Start LSDetector with tools/start_lsd_app.sh so the "
            "'lsdector' conda environment is created automatically, or install geospatial dependencies "
            'with `python -m pip install ".[geospatial]"` and rerun the command.'
        ) from exc
    return rasterio


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    """Read core GeoTIFF metadata without loading the full image."""

    raster_path = Path(path).expanduser().resolve()
    if not raster_path.exists():
        raise RasterError(f"Input raster does not exist: {raster_path}")
    if not raster_path.is_file():
        raise RasterError(f"Input raster is not a file: {raster_path}")

    rasterio = require_rasterio()
    try:
        with rasterio.open(raster_path) as src:
            crs = src.crs.to_string() if src.crs else None
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            return RasterMetadata(
                path=raster_path,
                width=int(src.width),
                height=int(src.height),
                count=int(src.count),
                crs=crs,
                transform=tuple(float(value) for value in src.transform.to_gdal()),
                bounds=tuple(float(value) for value in bounds),
                nodata=src.nodata,
                nodatavals=tuple(src.nodatavals),
            )
    except RasterError:
        raise
    except Exception as exc:
        raise RasterError(f"Failed to read raster metadata from {raster_path}: {exc}") from exc
