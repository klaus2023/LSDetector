import csv
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from lsd_app.cli.main import main


def _config(tmp_path: Path) -> dict[str, object]:
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
        "truth": {"vector_path": None, "mask_path": None},
    }


def _write_mask(path: Path, data: object) -> None:
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
        crs="EPSG:3857",
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(array, 1)


def _write_probability(path: Path, data: object) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    array = np.asarray(data, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        nodata=None,
    ) as dst:
        dst.write(array, 1)
        dst.write_mask(np.ones(array.shape, dtype=np.uint8) * 255)


def _gpkg_rows(path: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT id, area_m2, area_px, mean_confidence, geom FROM landslides").fetchall()


def test_vectorize_cli_dry_run_accepts_yaml_platform_config(
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
                "  min_area_pixels: 2",
                "  simplify_tolerance: 0.0",
                "  connectivity: 8",
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

    assert main(["vectorize", "--config", str(config_path), "--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert "[dry-run] vectorize: model=mock" in stdout
    assert "connectivity=8" in stdout
    assert "formats=gpkg" in stdout


def test_vectorize_empty_mask_writes_empty_gpkg_and_stats(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_mask(mask_path, [[0, 0], [0, 0]])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    assert main(["vectorize", "--config", str(config_path)]) == 0

    gpkg_path = tmp_path / "outputs" / "vectors" / "landslides.gpkg"
    assert gpkg_path.exists()
    assert _gpkg_rows(gpkg_path) == []

    stats = json.loads((tmp_path / "outputs" / "metrics" / "vector_stats.json").read_text(encoding="utf-8"))
    assert stats == {
        "polygon_count": 0,
        "total_area_m2": 0.0,
        "mean_area_m2": 0.0,
        "mean_confidence": None,
    }

    with (tmp_path / "outputs" / "metrics" / "vector_stats.csv").open(encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows == [
        {"polygon_count": "0", "total_area_m2": "0.0", "mean_area_m2": "0.0", "mean_confidence": ""}
    ]


def test_vectorize_small_mask_writes_polygon_outputs_and_stats(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    probability_path = tmp_path / "outputs" / "rasters" / "probability.tif"
    _write_mask(
        mask_path,
        [
            [0, 0, 0, 0],
            [0, 255, 255, 0],
            [0, 255, 255, 0],
            [0, 0, 0, 0],
        ],
    )
    _write_probability(
        probability_path,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.0],
            [0.0, 0.6, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
    config_data = _config(tmp_path)
    config_data["output"] = {"dir": str(tmp_path / "outputs"), "vector_formats": ["geojson", "shapefile"]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["vectorize", "--config", str(config_path)]) == 0

    gpkg_path = tmp_path / "outputs" / "vectors" / "landslides.gpkg"
    rows = _gpkg_rows(gpkg_path)
    assert len(rows) == 1
    feature_id, area_m2, area_px, mean_confidence, geometry = rows[0]
    assert feature_id == 1
    assert area_px == 4
    assert area_m2 == pytest.approx(400.0)
    assert mean_confidence == pytest.approx(0.5)
    assert bytes(geometry).startswith(b"GP")

    geojson = json.loads((tmp_path / "outputs" / "vectors" / "landslides.geojson").read_text(encoding="utf-8"))
    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["area_px"] == 4
    assert geojson["features"][0]["properties"]["mean_confidence"] == pytest.approx(0.5)
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"

    assert (tmp_path / "outputs" / "vectors" / "landslides.shp").exists()
    assert (tmp_path / "outputs" / "vectors" / "landslides.dbf").exists()
    assert (tmp_path / "outputs" / "vectors" / "landslides.prj").exists()

    stats = json.loads((tmp_path / "outputs" / "metrics" / "vector_stats.json").read_text(encoding="utf-8"))
    assert stats["polygon_count"] == 1
    assert stats["total_area_m2"] == pytest.approx(400.0)
    assert stats["mean_area_m2"] == pytest.approx(400.0)
    assert stats["mean_confidence"] == pytest.approx(0.5)


def test_vectorize_legacy_mode_still_runs(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_mask(
        mask_path,
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
    )
    config_data = _config(tmp_path)
    config_data["vectorize"] = {"mode": "legacy"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["vectorize", "--config", str(config_path)]) == 0

    rows = _gpkg_rows(tmp_path / "outputs" / "vectors" / "landslides.gpkg")
    assert len(rows) == 1
    assert rows[0][2] == 1


def test_vectorize_min_area_pixels_filters_small_components(tmp_path: Path) -> None:
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    _write_mask(
        mask_path,
        [
            [255, 0, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 0, 0],
        ],
    )
    config_data = _config(tmp_path)
    config_data["postprocess"] = {"min_area_pixels": 2, "simplify_tolerance": 0.0, "connectivity": 8}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["vectorize", "--config", str(config_path)]) == 0

    rows = _gpkg_rows(tmp_path / "outputs" / "vectors" / "landslides.gpkg")
    assert len(rows) == 1
    assert rows[0][2] == 4


def test_vectorize_missing_mask_has_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["vectorize", "--config", str(config_path)])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    if importlib.util.find_spec("rasterio") is None:
        assert "requires rasterio" in err
    else:
        assert "Mask raster does not exist" in err
