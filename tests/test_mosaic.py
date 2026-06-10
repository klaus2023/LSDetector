import json
from pathlib import Path

import pytest

from lsd_app.cli.main import main


def _config(tmp_path: Path, raster_path: Path) -> dict[str, object]:
    return {
        "input": {"raster_path": str(raster_path), "RGB_bands": "RGB_bands: 1,2,3"},
        "output": {
            "dir": str(tmp_path / "outputs"),
            "raster_options": {"BIGTIFF": "IF_SAFER", "tiled": True, "compress": "deflate"},
        },
        "model": {"name": "mock", "checkpoint_path": None},
        "tiling": {"tile_size": 3, "stride": 2},
        "inference": {"batch_size": 2, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 0},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": 0, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def _write_source(path: Path, *, width: int, height: int, fill: int = 100) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = np.full((1, height, width), fill, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(10.0, 20.0, 2.0, 2.0),
        nodata=0,
    ) as dst:
        dst.write(data)


def test_mosaic_cli_dry_run_accepts_yaml_platform_config(
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
                "  tile_size: 3",
                "  stride: 2",
                "inference:",
                "  batch_size: 2",
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

    assert main(["mosaic", "--config", str(config_path), "--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert "[dry-run] mosaic: model=mock" in stdout
    assert "patch_predictions" in stdout


def test_mosaic_cli_reconstructs_probability_and_mask_after_mock_infer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    raster_path = tmp_path / "source.tif"
    _write_source(raster_path, width=4, height=4, fill=100)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0
    assert main(["mosaic", "--config", str(config_path)]) == 0

    stdout = capsys.readouterr().out
    assert "[mosaic] model=mock" in stdout
    probability_path = tmp_path / "outputs" / "rasters" / "probability.tif"
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    assert probability_path.exists()
    assert mask_path.exists()

    with rasterio.open(raster_path) as source_src, rasterio.open(probability_path) as prob_src, rasterio.open(mask_path) as mask_src:
        assert prob_src.count == 1
        assert prob_src.width == source_src.width
        assert prob_src.height == source_src.height
        assert prob_src.crs == source_src.crs
        assert prob_src.transform.almost_equals(source_src.transform)
        assert prob_src.dtypes == ("float32",)
        probability = prob_src.read(1)
        assert float(probability.min()) >= 0.0
        assert float(probability.max()) <= 1.0

        assert mask_src.count == 1
        assert mask_src.width == source_src.width
        assert mask_src.height == source_src.height
        assert mask_src.crs == source_src.crs
        assert mask_src.transform.almost_equals(source_src.transform)
        assert mask_src.dtypes == ("uint8",)
        assert set(np.unique(mask_src.read(1))).issubset({0, 255})


def test_mosaic_averages_overlap_and_crops_padded_edges(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    raster_path = tmp_path / "source.tif"
    _write_source(raster_path, width=3, height=3, fill=100)
    config_data = _config(tmp_path, raster_path)
    config_data["tiling"] = {"tile_size": 2, "stride": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    predictions_dir = tmp_path / "outputs" / "patch_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    positive_names = {"source_r0_c1.tif", "source_r1_c0.tif"}
    for patch_path in (tmp_path / "outputs" / "patches").glob("*.tif"):
        with rasterio.open(patch_path) as patch_src:
            profile = patch_src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=None)
            value = 255 if patch_path.name in positive_names else 0
            prediction = np.full((patch_src.height, patch_src.width), value, dtype=np.uint8)
            mask = patch_src.dataset_mask()
        with rasterio.open(predictions_dir / patch_path.name, "w", **profile) as dst:
            dst.write(prediction, 1)
            dst.write_mask(mask)

    assert main(["mosaic", "--config", str(config_path)]) == 0

    probability_path = tmp_path / "outputs" / "rasters" / "probability.tif"
    mask_path = tmp_path / "outputs" / "rasters" / "mask.tif"
    with rasterio.open(probability_path) as prob_src, rasterio.open(mask_path) as mask_src:
        assert prob_src.width == 3
        assert prob_src.height == 3
        probability = prob_src.read(1)
        assert probability.shape == (3, 3)
        assert probability[1, 1] == pytest.approx(0.5)
        assert mask_src.read(1)[1, 1] == 255


def test_mosaic_missing_predictions_has_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raster_path = tmp_path / "source.tif"
    _write_source(raster_path, width=3, height=3, fill=100)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0

    with pytest.raises(SystemExit) as exc:
        main(["mosaic", "--config", str(config_path)])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "Missing patch prediction GeoTIFFs" in stderr
