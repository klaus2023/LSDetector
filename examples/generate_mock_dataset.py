"""Generate a tiny deterministic GeoTIFF dataset for LSD_APP examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a tiny mock LSD_APP dataset.")
    parser.add_argument("--output", required=True, help="Output directory for generated files.")
    parser.add_argument("--size", type=int, default=32, help="Raster width and height in pixels.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic content.")
    parser.add_argument("--with-truth", action="store_true", help="Also write truth.tif and wire it into the config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--config-format",
        choices=("yaml", "json"),
        default="yaml",
        help="Write config.yaml or config.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output).expanduser().resolve()

    if args.size < 4:
        raise SystemExit("--size must be at least 4 pixels.")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory already exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.tif"
    truth_path = output_dir / "truth.tif"
    config_path = output_dir / f"config.{args.config_format}"
    manifest_path = output_dir / "manifest.json"

    source, truth = _make_arrays(size=args.size, seed=args.seed)
    _write_tif(source_path, source, dtype="uint8")
    if args.with_truth:
        _write_tif(truth_path, truth, dtype="uint8")
    elif truth_path.exists():
        truth_path.unlink()

    config = _make_config(
        output_dir=output_dir,
        source_path=source_path,
        truth_path=truth_path if args.with_truth else None,
        tile_size=args.size,
    )
    _write_config(config_path, config, args.config_format)

    manifest = {
        "schema_version": 1,
        "dataset": "lsd_app_mock",
        "seed": args.seed,
        "size": args.size,
        "files": {
            "source": str(source_path),
            "truth": str(truth_path) if args.with_truth else None,
            "config": str(config_path),
        },
        "source": {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "min": int(source.min()),
            "max": int(source.max()),
        },
        "truth": {
            "enabled": args.with_truth,
            "positive_pixels": int(truth.sum()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"source: {source_path}")
    if args.with_truth:
        print(f"truth: {truth_path}")
    print(f"config: {config_path}")
    print(f"manifest: {manifest_path}")
    return 0


def _make_arrays(*, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    slope = ((xx * 5 + yy * 3) / max(size - 1, 1)).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=9.0, size=(size, size)).astype(np.float32)

    center_x = rng.uniform(size * 0.35, size * 0.65)
    center_y = rng.uniform(size * 0.35, size * 0.65)
    radius_x = rng.uniform(size * 0.12, size * 0.22)
    radius_y = rng.uniform(size * 0.10, size * 0.20)
    landslide = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0

    source = 70.0 + slope + noise
    source[landslide] += 105.0
    source = np.clip(source, 0, 255).astype(np.uint8)
    truth = landslide.astype(np.uint8)
    return source, truth


def _write_tif(path: Path, data: np.ndarray, *, dtype: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:3857",
        transform=from_origin(500000.0, 4100000.0, 10.0, 10.0),
        nodata=0 if dtype == "uint8" else None,
    ) as dst:
        dst.write(data, 1)


def _make_config(
    *,
    output_dir: Path,
    source_path: Path,
    truth_path: Path | None,
    tile_size: int,
) -> dict[str, Any]:
    return {
        "input": {"raster_path": str(source_path), "RGB_bands": "RGB_bands: 1,1,1"},
        "output": {"dir": str(output_dir / "outputs")},
        "model": {"name": "mock", "checkpoint_path": None},
        "tiling": {"tile_size": tile_size, "stride": tile_size},
        "inference": {"batch_size": 1, "threshold": 0.5, "overwrite": True},
        "postprocess": {"min_area_pixels": 0, "simplify_tolerance": 0.0, "connectivity": 8},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": str(truth_path) if truth_path else None},
    }


def _write_config(path: Path, config: dict[str, Any], config_format: str) -> None:
    if config_format == "json":
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return

    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        path.write_text(_simple_yaml(config), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _simple_yaml(value: Any, *, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{spaces}{key}:")
                lines.append(_simple_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{spaces}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    raise TypeError("simple YAML writer only supports mappings")


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


if __name__ == "__main__":
    raise SystemExit(main())
