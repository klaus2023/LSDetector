import json
from pathlib import Path

import pytest

from lsd_app.cli.main import main
from lsd_app.config import normalize_config
from lsd_app.core.tiling import prepare_raster


def _prepare_config(tmp_path: Path, raster_path: Path) -> dict[str, object]:
    return {
        "input": {"raster_path": str(raster_path), "RGB_bands": "RGB_bands: 1,2,3"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "lsdsam", "checkpoint_path": None},
        "tiling": {"tile_size": 3, "stride": 2},
        "inference": {"batch_size": 1, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 0},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": -9999, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def test_prepare_missing_raster_has_clear_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "missing.json"
    config_path.write_text(json.dumps(_prepare_config(tmp_path, tmp_path / "missing.tif")), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["prepare", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "Input raster does not exist" in capsys.readouterr().err


def test_prepare_cuts_geotiff_patches_with_spatial_metadata(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    raster_path = tmp_path / "source.tif"
    transform = from_origin(100.0, 40.0, 2.0, 2.0)
    data = np.zeros((2, 4, 5), dtype=np.int16)
    data[0, 0:2, 0:2] = np.array([[1, 2], [3, 4]], dtype=np.int16)
    data[1, 0:2, 0:2] = np.array([[10, 20], [30, 40]], dtype=np.int16)
    data[:, 3, 4] = np.array([5, 50], dtype=np.int16)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data)

    config = normalize_config(_prepare_config(tmp_path, raster_path), create_output_dirs=True)
    result = prepare_raster(config)

    assert result.saved_patches > 0
    assert result.skipped_all_zero > 0
    assert result.index_path.is_file()

    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["properties"]["source"]["width"] == 5
    assert len(index["features"]) == result.saved_patches
    assert any(feature["properties"]["padded"] for feature in index["features"])

    first_patch = result.patch_dir / Path(index["features"][0]["properties"]["patch_path"]).name
    with rasterio.open(first_patch) as src:
        assert src.width == 3
        assert src.height == 3
        assert src.count == 2
        assert src.crs.to_string() == "EPSG:4326"
        expected = rasterio.windows.transform(Window(0, 0, 3, 3), transform)
        assert src.transform.almost_equals(expected)
        patch = src.read()
        assert patch.dtype == np.float32
        assert float(np.nanmin(patch)) >= 0.0
        assert float(np.nanmax(patch)) <= 1.0


def test_prepare_cli_runs_with_temporary_geotiff(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    data = np.zeros((1, 3, 3), dtype=np.uint8)
    data[0, 0, 0] = 10
    data[0, 1, 1] = 20
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=from_origin(0.0, 3.0, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(data)

    config_path = tmp_path / "prepare.json"
    config_path.write_text(json.dumps(_prepare_config(tmp_path, raster_path)), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert (tmp_path / "outputs" / "patches").is_dir()
    assert (tmp_path / "outputs" / "patch_index.geojson").is_file()
