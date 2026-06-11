import json
from pathlib import Path
import sqlite3
import struct

import pytest

from lsd_app.cli.main import main
from lsd_app.core.visualization import _overview_shape


def _config(tmp_path: Path, *, truth_mask_path: Path | None = None, truth_vector_path: Path | None = None) -> dict[str, object]:
    return {
        "input": {"raster_path": str(tmp_path / "source.tif"), "RGB_bands": "RGB_bands: 1,2,3"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "mock", "checkpoint_path": None},
        "tiling": {"tile_size": 4, "stride": 4},
        "inference": {"batch_size": 1, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 0, "simplify_tolerance": 0.0, "connectivity": 8},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": 0, "output": 0, "mask": True},
        },
        "truth": {
            "mask_path": str(truth_mask_path) if truth_mask_path is not None else None,
            "vector_path": str(truth_vector_path) if truth_vector_path is not None else None,
        },
    }


def _write_raster(path: Path, data: object, *, crs: str = "EPSG:3857", nodata: int | None = 0) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    array = np.asarray(data, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(0.0, float(array.shape[0]), 1.0, 1.0),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def _write_png_mask(path: Path, data: object) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    array = np.asarray(data, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="PNG",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(array, 1)


def _write_truth_vector(tmp_path: Path, *, vector_format: str) -> Path:
    truth_output = tmp_path / f"truth_{vector_format}"
    _write_raster(
        truth_output / "rasters" / "mask.tif",
        [
            [255, 0],
            [0, 0],
        ],
    )
    config = _config(tmp_path)
    config["output"] = {"dir": str(truth_output), "vector_formats": [vector_format]}
    config_path = truth_output / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["vectorize", "--config", str(config_path)]) == 0
    if vector_format == "shapefile":
        return truth_output / "vectors" / "landslides.shp"
    return truth_output / "vectors" / "landslides.gpkg"


def _write_rgb_source(path: Path, *, width: int = 4, height: int = 4) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rows, cols = np.indices((height, width))
    data = np.stack(
        [
            (cols * 40 + 20).astype(np.uint8),
            (rows * 40 + 20).astype(np.uint8),
            np.full((height, width), 80, dtype=np.uint8),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
        nodata=None,
    ) as dst:
        dst.write(data)


def _gpkg_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM landslides").fetchone()[0])


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">2I", data[16:24])
    return width, height


def test_evaluate_and_visualize_cli_dry_run_accept_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  raster_path: source.tif",
                "  RGB_bands: 1,2,3",
                "output:",
                f"  dir: {tmp_path / 'outputs'}",
                "model:",
                "  name: mock",
                "  checkpoint_path: null",
                "tiling:",
                "  tile_size: 4",
                "  stride: 4",
                "inference:",
                "  batch_size: 1",
                "  threshold: 0.5",
                "postprocess:",
                "  min_area_pixels: 0",
                "runtime:",
                "  device: cpu",
                "  num_workers: 0",
                "geospatial:",
                "  crs:",
                "    preserve_input: true",
                "    target: null",
                "    allow_missing: false",
                "  nodata:",
                "    input: 0",
                "    output: 0",
                "    mask: true",
                "truth:",
                "  vector_path: null",
                "  mask_path: null",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["evaluate", "--config", str(config_path), "--dry-run"]) == 0
    assert main(["visualize", "--config", str(config_path), "--dry-run"]) == 0

    stdout = capsys.readouterr().out
    assert "[dry-run] evaluate: model=mock" in stdout
    assert "[dry-run] visualize: model=mock" in stdout
    assert "truth=none" in stdout


def test_evaluate_with_truth_mask_outputs_metrics_and_tpfpfn(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    truth_path = tmp_path / "truth.tif"
    _write_raster(
        mask_path,
        [
            [255, 255, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
    )
    _write_raster(
        truth_path,
        [
            [255, 0, 0],
            [0, 255, 255],
            [0, 0, 0],
        ],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, truth_mask_path=truth_path)), encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 0

    evaluation = json.loads((tmp_path / "outputs" / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["confusion_matrix"] == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 5,
    }
    assert evaluation["iou"] == pytest.approx(0.5)
    assert evaluation["miou"] == pytest.approx((0.5 + 5 / 7) / 2)
    assert evaluation["precision"] == pytest.approx(2 / 3)
    assert evaluation["recall"] == pytest.approx(2 / 3)
    assert evaluation["f1"] == pytest.approx(2 / 3)

    assert (tmp_path / "outputs" / "metrics" / "evaluation.csv").exists()
    assert _gpkg_count(tmp_path / "outputs" / "vectors" / "TP.gpkg") == 1
    assert _gpkg_count(tmp_path / "outputs" / "vectors" / "FP.gpkg") == 1
    assert _gpkg_count(tmp_path / "outputs" / "vectors" / "FN.gpkg") == 1


def test_evaluate_with_truth_geojson_vector(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    truth_path = tmp_path / "truth.geojson"
    _write_raster(
        mask_path,
        [
            [255, 0],
            [0, 0],
        ],
    )
    truth_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, truth_vector_path=truth_path)), encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 0

    evaluation = json.loads((tmp_path / "outputs" / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["iou"] == pytest.approx(1.0)
    assert evaluation["confusion_matrix"]["true_positive"] == 1


@pytest.mark.parametrize("vector_format", ["gpkg", "shapefile"])
def test_evaluate_accepts_gpkg_and_shapefile_truth_vectors(tmp_path: Path, vector_format: str) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_raster(mask_path, [[255, 0], [0, 0]])
    truth_path = _write_truth_vector(tmp_path, vector_format=vector_format)
    config_path = tmp_path / f"config_{vector_format}.json"
    config_path.write_text(json.dumps(_config(tmp_path, truth_vector_path=truth_path)), encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 0

    evaluation = json.loads((tmp_path / "outputs" / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["iou"] == pytest.approx(1.0)
    assert evaluation["confusion_matrix"]["true_positive"] == 1


def test_evaluate_without_truth_skips_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_raster(mask_path, [[255, 0], [0, 0]])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 0

    stdout = capsys.readouterr().out
    assert "No truth data configured" in stdout
    assert not (tmp_path / "outputs" / "metrics" / "evaluation.json").exists()


def test_evaluate_accepts_png_truth_mask(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    truth_path = tmp_path / "truth.png"
    _write_raster(mask_path, [[255, 0], [0, 0]])
    _write_png_mask(truth_path, [[255, 0], [0, 0]])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, truth_mask_path=truth_path)), encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 0

    evaluation = json.loads((tmp_path / "outputs" / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["iou"] == pytest.approx(1.0)


def test_visualize_without_truth_writes_openable_overlay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_rgb_source(source_path)
    _write_raster(mask_path, [[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 0, 0], [0, 0, 0, 255]])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    assert main(["visualize", "--config", str(config_path)]) == 0

    stdout = capsys.readouterr().out
    assert "without truth data" in stdout
    overlay_path = tmp_path / "outputs" / "visualizations" / "overlay.png"
    assert overlay_path.exists()
    assert _png_size(overlay_path) == (4, 4)


def test_visualize_with_truth_uses_tp_fp_fn_colors(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    truth_path = tmp_path / "truth.tif"
    _write_rgb_source(source_path)
    _write_raster(mask_path, [[255, 255], [0, 0]])
    _write_raster(truth_path, [[255, 0], [255, 0]])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, truth_mask_path=truth_path)), encoding="utf-8")

    assert main(["visualize", "--config", str(config_path)]) == 0

    overlay_path = tmp_path / "outputs" / "visualizations" / "overlay.png"
    assert _png_size(overlay_path) == (2, 2)
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(overlay_path) as src:
        image = np.moveaxis(src.read(), 0, -1)
    assert image[0, 0, 0] > image[0, 0, 2]  # TP orange-red
    assert image[0, 1, 2] > image[0, 1, 0]  # FP blue
    assert image[1, 0, 0] > image[1, 0, 2] and image[1, 0, 1] > image[1, 0, 2]  # FN yellow


def test_overview_shape_downsamples_large_dimensions() -> None:
    assert _overview_shape(4000, 2000, max_dimension=1000) == (1000, 500, 0.25)
