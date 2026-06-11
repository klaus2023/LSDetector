import json
from pathlib import Path

import pytest

from lsd_app.cli.main import main


def _config(tmp_path: Path, raster_path: Path) -> dict[str, object]:
    return {
        "input": {"raster_path": str(raster_path), "RGB_bands": "RGB_bands: 1,2,3"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "lsdsam", "checkpoint_path": "weights/placeholder.pth"},
        "tiling": {"tile_size": 3, "stride": 2},
        "inference": {"batch_size": 2, "threshold": 0.4},
        "postprocess": {"min_area_pixels": 0},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": 0, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def test_infer_cli_mock_generates_patch_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    data = np.zeros((1, 4, 4), dtype=np.uint8)
    data[0, :3, :3] = np.array(
        [
            [0, 30, 80],
            [20, 100, 150],
            [60, 120, 200],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(10.0, 20.0, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(data)

    config_path = tmp_path / "config.json"
    config_data = _config(tmp_path, raster_path)
    config_data["runtime"] = {"device": "cpu", "num_workers": 2}
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0
    stdout = capsys.readouterr().out
    assert "[infer] model=mock adapter=mock" in stdout

    patch_paths = sorted((tmp_path / "outputs" / "patches").glob("*.tif"))
    prediction_paths = sorted((tmp_path / "outputs" / "patch_predictions").glob("*.tif"))
    manifest_path = tmp_path / "outputs" / "infer_manifest.json"
    assert len(prediction_paths) == len(patch_paths)
    assert prediction_paths
    assert [path.name for path in prediction_paths] == [path.name for path in patch_paths]
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "mock"
    assert manifest["adapter_name"] == "mock"
    assert manifest["checkpoint_path"].endswith("weights/placeholder.pth")
    assert manifest["patch_count"] == len(patch_paths)
    assert manifest["prediction_count"] == len(prediction_paths)
    assert manifest["skipped_count"] == 0
    assert manifest["batch_size"] == 2
    assert manifest["num_workers"] == 2
    assert manifest["device"] == "cpu"
    assert manifest["overwrite"] is True
    assert manifest["started_at"]
    assert manifest["finished_at"]
    assert sorted(Path(path).name for path in manifest["output_paths"]) == [path.name for path in prediction_paths]

    any_foreground = False
    for patch_path, prediction_path in zip(patch_paths, prediction_paths):
        with rasterio.open(patch_path) as patch_src, rasterio.open(prediction_path) as pred_src:
            assert pred_src.count == 1
            assert pred_src.width == patch_src.width
            assert pred_src.height == patch_src.height
            assert pred_src.crs == patch_src.crs
            assert pred_src.transform.almost_equals(patch_src.transform)
            np.testing.assert_array_equal(pred_src.dataset_mask(), patch_src.dataset_mask())
            prediction = pred_src.read(1)
            assert prediction.dtype == np.uint8
            assert set(np.unique(prediction)).issubset({0, 255})
            any_foreground = any_foreground or bool(np.any(prediction == 255))
    assert any_foreground


def test_infer_cli_mock_ignores_macos_metadata_patch_files(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    patch_dir = tmp_path / "outputs" / "patches"
    patch_dir.mkdir(parents=True)
    data = np.ones((1, 3, 3), dtype=np.float32)

    for path in (raster_path, patch_dir / "source_r0_c0.tif"):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=from_origin(10.0, 20.0, 1.0, 1.0),
            nodata=0,
        ) as dst:
            dst.write(data)

    (patch_dir / "._source_r0_c0.tif").write_bytes(b"not a geotiff")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")

    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0

    manifest = json.loads((tmp_path / "outputs" / "infer_manifest.json").read_text(encoding="utf-8"))
    predictions = sorted((tmp_path / "outputs" / "patch_predictions").glob("*.tif"))
    assert manifest["patch_count"] == 1
    assert [path.name for path in predictions] == ["source_r0_c0.tif"]


def test_infer_default_overwrites_existing_prediction(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    data = np.ones((1, 3, 3), dtype=np.uint8) * 100
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 3.0, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(data)

    config_data = _config(tmp_path, raster_path)
    config_data["tiling"] = {"tile_size": 3, "stride": 3}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    patch_path = next((tmp_path / "outputs" / "patches").glob("*.tif"))
    prediction_path = tmp_path / "outputs" / "patch_predictions" / patch_path.name
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(patch_path) as patch_src:
        profile = patch_src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)
        mask = patch_src.dataset_mask()
    with rasterio.open(prediction_path, "w", **profile) as dst:
        dst.write(np.full((3, 3), 123, dtype=np.uint8), 1)
        dst.write_mask(mask)

    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0

    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    assert 123 not in set(np.unique(prediction))

    manifest = json.loads((tmp_path / "outputs" / "infer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["patch_count"] == 1
    assert manifest["prediction_count"] == 1
    assert manifest["skipped_count"] == 0


def test_infer_overwrite_false_skips_existing_prediction(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    data = np.ones((1, 3, 3), dtype=np.uint8) * 100
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=from_origin(100.0, 200.0, 2.0, 2.0),
        nodata=0,
    ) as dst:
        dst.write(data)

    config_data = _config(tmp_path, raster_path)
    config_data["tiling"] = {"tile_size": 3, "stride": 3}
    config_data["inference"] = {"batch_size": 2, "threshold": 0.4, "overwrite": False}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    patch_path = next((tmp_path / "outputs" / "patches").glob("*.tif"))
    prediction_path = tmp_path / "outputs" / "patch_predictions" / patch_path.name
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(patch_path) as patch_src:
        profile = patch_src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)
        mask = patch_src.dataset_mask()
    with rasterio.open(prediction_path, "w", **profile) as dst:
        dst.write(np.full((3, 3), 123, dtype=np.uint8), 1)
        dst.write_mask(mask)

    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0

    with rasterio.open(prediction_path) as pred_src, rasterio.open(patch_path) as patch_src:
        np.testing.assert_array_equal(pred_src.read(1), np.full((3, 3), 123, dtype=np.uint8))
        assert pred_src.crs == patch_src.crs
        assert pred_src.transform.almost_equals(patch_src.transform)

    manifest = json.loads((tmp_path / "outputs" / "infer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["patch_count"] == 1
    assert manifest["prediction_count"] == 0
    assert manifest["skipped_count"] == 1
    assert manifest["overwrite"] is False
    assert manifest["skipped_output_paths"] == [str(prediction_path.resolve())]
    assert manifest["output_paths"] == [str(prediction_path.resolve())]


def test_infer_mock_without_prepare_has_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raster_path = tmp_path / "source.tif"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["infer", "--config", str(config_path), "--model", "mock"])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "No patch GeoTIFFs found" in stderr
    assert "Run `lsd prepare --config ...` first" in stderr


def test_infer_mock_cli_override_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    import builtins

    raster_path = tmp_path / "source.tif"
    data = np.ones((1, 3, 3), dtype=np.uint8) * 100
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

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")
    assert main(["prepare", "--config", str(config_path)]) == 0

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("mock infer should not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert main(["infer", "--config", str(config_path), "--model", "mock"]) == 0


def test_infer_lsdsam_missing_checkpoint_has_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "source.tif"
    data = np.ones((1, 3, 3), dtype=np.uint8) * 100
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 3.0, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(data)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path, raster_path)), encoding="utf-8")
    assert main(["prepare", "--config", str(config_path)]) == 0

    with pytest.raises(SystemExit) as exc:
        main(["infer", "--config", str(config_path), "--model", "lsdsam-fewshot"])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "LSDSAM checkpoint does not exist" in stderr
    assert "torch" not in stderr.lower()
