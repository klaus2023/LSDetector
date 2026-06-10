import json
from pathlib import Path

import pytest

from lsd_app.config import ConfigError, OUTPUT_SUBDIRS, load_config, normalize_config


def _base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "input": {"raster_path": "sample.tif", "RGB_bands": "RGB_bands: 3,2,1"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "lsdsam", "checkpoint_path": "weights/model.pth"},
        "tiling": {"tile_size": 512, "stride": 256},
        "inference": {"batch_size": 2, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 32},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def test_json_config_creates_standard_output_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_base_config(tmp_path)), encoding="utf-8")

    config = load_config(config_path)

    assert config.model_name == "lsdsam"
    for name in OUTPUT_SUBDIRS:
        assert (config.output_dir / name).is_dir()


def test_missing_input_path_has_clear_error(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["input"] = {}

    with pytest.raises(ConfigError, match="Missing required config value: input.raster_path"):
        normalize_config(raw, create_output_dirs=False)


def test_invalid_tile_stride_has_clear_error(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["tiling"] = {"tile_size": 256, "stride": 512}

    with pytest.raises(ConfigError, match="stride \\(512\\) must be <= tile_size \\(256\\)"):
        normalize_config(raw, create_output_dirs=False)


def test_unknown_model_has_clear_error(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["model"] = {"name": "unknown"}

    with pytest.raises(ConfigError, match="Unknown model name 'unknown'"):
        normalize_config(raw, create_output_dirs=False)


def test_inference_overwrite_defaults_and_validates(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    config = normalize_config(raw, create_output_dirs=False)
    assert config.overwrite is True

    raw = _base_config(tmp_path)
    raw["inference"] = {"batch_size": 2, "threshold": 0.5, "overwrite": False}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.overwrite is False

    raw = _base_config(tmp_path)
    raw["inference"] = {"batch_size": 2, "threshold": 0.5, "overwrite": "later"}
    with pytest.raises(ConfigError, match="Invalid inference.overwrite"):
        normalize_config(raw, create_output_dirs=False)


def test_raster_creation_options_are_preserved(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["output"] = {
        "dir": str(tmp_path / "outputs"),
        "raster_options": {"BIGTIFF": "YES", "tiled": True, "compress": "lzw"},
    }

    config = normalize_config(raw, create_output_dirs=False)

    assert config.raster_options == {"BIGTIFF": "YES", "tiled": True, "compress": "lzw"}


def test_vectorize_options_defaults_and_validates(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    config = normalize_config(raw, create_output_dirs=False)
    assert config.simplify_tolerance == 0.0
    assert config.connectivity == 8
    assert config.vectorize_mode == "streaming"
    assert config.vector_formats == ("gpkg",)

    raw = _base_config(tmp_path)
    raw["postprocess"] = {"min_area_pixels": 2, "simplify_tolerance": 1.5, "connectivity": 4}
    raw["vectorize"] = {"mode": "legacy"}
    raw["output"] = {"dir": str(tmp_path / "outputs"), "vector_formats": ["geojson", "shp"]}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.simplify_tolerance == 1.5
    assert config.connectivity == 4
    assert config.vectorize_mode == "legacy"
    assert config.vector_formats == ("geojson", "shapefile")

    raw = _base_config(tmp_path)
    raw["postprocess"] = {"min_area_pixels": 0, "connectivity": 6}
    with pytest.raises(ConfigError, match="Invalid connectivity"):
        normalize_config(raw, create_output_dirs=False)

    raw = _base_config(tmp_path)
    raw["vectorize"] = {"mode": "eager"}
    with pytest.raises(ConfigError, match="Invalid vectorize mode"):
        normalize_config(raw, create_output_dirs=False)


def test_truth_paths_support_nested_and_flat_keys(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["truth"] = {"vector_path": "truth.geojson", "mask_path": "truth.tif"}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.truth_vector_path == (Path.cwd() / "outputs" / "truth.geojson").resolve()
    assert config.truth_mask_path == (Path.cwd() / "outputs" / "truth.tif").resolve()

    raw = _base_config(tmp_path)
    raw.pop("truth")
    raw["truth_vector_path"] = "truth.gpkg"
    raw["truth_mask_path"] = "truth.png"
    config = normalize_config(raw, create_output_dirs=False)
    assert config.truth_vector_path == (Path.cwd() / "outputs" / "truth.gpkg").resolve()
    assert config.truth_mask_path == (Path.cwd() / "outputs" / "truth.png").resolve()


def test_config_relative_paths_resolve_from_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw = _base_config(tmp_path)
    raw["input"] = {"raster_path": "../data/sample.tif", "RGB_bands": "RGB_bands: 1,2,3"}
    raw["output"] = {"dir": "../runs/out"}
    raw["model"] = {"name": "mock", "checkpoint_path": "../weights/model.pth"}
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(config_path, create_output_dirs=False)

    assert config.input_raster_path == (tmp_path / "data" / "sample.tif").resolve()
    assert config.output_dir == (tmp_path / "runs" / "out").resolve()
    assert config.checkpoint_path == (tmp_path / "weights" / "model.pth").resolve()


def test_model_input_modes_are_normalized(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["model"] = {"name": "lsdsam", "checkpoint_path": "weights/model.pth", "options": {"edge": "sobel"}}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.input_mode == "rgb_wavelets"
    assert config.adapter_options["edge"] == "wavelets"
    assert "mode" not in config.adapter_options

    raw = _base_config(tmp_path)
    raw["model"] = {"name": "mock", "input_mode": "rgb_wavelets"}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.input_mode == "rgb"


def test_RGB_bands_requires_new_field_and_validates(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["input"] = {"raster_path": "sample.tif"}
    with pytest.raises(ConfigError, match="Missing RGB_bands"):
        normalize_config(raw, create_output_dirs=False)

    raw = _base_config(tmp_path)
    config = normalize_config(raw, create_output_dirs=False)
    assert config.RGB_bands == "RGB_bands: 3,2,1"

    raw = _base_config(tmp_path)
    raw["input"] = {"raster_path": "sample.tif", "RGB_bands": "RGB_bands: 1,2,3"}
    config = normalize_config(raw, create_output_dirs=False)
    assert config.RGB_bands == "RGB_bands: 1,2,3"

    raw = _base_config(tmp_path)
    raw["input"] = {"raster_path": "sample.tif", "RGB_bands": "nir_first"}
    with pytest.raises(ConfigError, match="Invalid RGB_bands"):
        normalize_config(raw, create_output_dirs=False)
