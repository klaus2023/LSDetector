from types import SimpleNamespace

import pytest

from lsd_app.config import ConfigError, normalize_config
from lsd_app.core.channels import (
    DEFAULT_RGB_BANDS,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    RGBBandsError,
    extract_rgb_bands,
    normalize_rgb_bands,
    rgb_bands_to_class,
    rgb_bands_to_indexes,
)
from lsd_app.core.inference import _prepare_model_input_array


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RGB_bands: 1,2,3", "RGB_bands: 1,2,3"),
        ("1,2,3", "RGB_bands: 1,2,3"),
        ("RGB_bands: 3,2,1", "RGB_bands: 3,2,1"),
        ("RGB_bands: 4,3,2", "RGB_bands: 4,3,2"),
        ("RGB_bands: 6,4,2", "RGB_bands: 6,4,2"),
    ],
)
def test_normalize_rgb_bands_accepts_presets_and_shorthand(raw: str, expected: str) -> None:
    assert normalize_rgb_bands(raw) == expected


@pytest.mark.parametrize("raw", ["RGB_bands: 0,1,2", "RGB_bands: 6,4", "RGB_bands: a,b,c", "rgb_first3", "bgr_first3"])
def test_normalize_rgb_bands_rejects_invalid_and_legacy_values(raw: str) -> None:
    with pytest.raises(RGBBandsError):
        normalize_rgb_bands(raw)


def test_rgb_band_display_and_indexes() -> None:
    assert rgb_bands_to_indexes("RGB_bands: 6,4,2") == (6, 4, 2)
    assert rgb_bands_to_class("RGB_bands: 6,4,2") == "RGB bands 6,4,2"


def test_extract_rgb_bands_from_multiband_chw() -> None:
    np = pytest.importorskip("numpy")
    array = np.zeros((6, 2, 2), dtype=np.uint8)
    for index in range(6):
        array[index, :, :] = index + 1

    converted = extract_rgb_bands(array, "RGB_bands: 6,4,2", layout="chw")

    assert converted.shape == (3, 2, 2)
    assert converted[:, 0, 0].tolist() == [6, 4, 2]


def test_extract_rgb_bands_rejects_missing_band() -> None:
    np = pytest.importorskip("numpy")
    array = np.zeros((3, 2, 2), dtype=np.uint8)

    with pytest.raises(RGBBandsError):
        extract_rgb_bands(array, "RGB_bands: 4,3,2", layout="chw")


def test_inference_preserves_raw_rgb_bands_for_adapter() -> None:
    np = pytest.importorskip("numpy")
    array = np.zeros((6, 1, 1), dtype=np.float32)
    array[:, 0, 0] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    converted = _prepare_model_input_array(array, config=SimpleNamespace(RGB_bands="RGB_bands: 6,4,2"))

    assert converted[:, 0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_config_requires_rgb_bands_and_rejects_channel_order(tmp_path) -> None:
    raw = {
        "input": {"raster_path": "sample.tif", "RGB_bands": "RGB_bands: 4,3,2"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "lsdsam", "checkpoint_path": "weights/model.pth", "options": {"mode": "bgr"}},
        "tiling": {"tile_size": 512, "stride": 256},
        "inference": {"batch_size": 1, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 0},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }

    config = normalize_config(raw, create_output_dirs=False)

    assert config.RGB_bands == "RGB_bands: 4,3,2"
    assert config.input_mode == "rgb_wavelets"
    assert config.adapter_options["edge"] == "wavelets"
    assert "mode" not in config.adapter_options

    raw["input"] = {"raster_path": "sample.tif", "channel_order": "bgr_first3"}
    with pytest.raises(ConfigError):
        normalize_config(raw, create_output_dirs=False)


def test_default_rgb_bands_constant() -> None:
    assert DEFAULT_RGB_BANDS == "RGB_bands: 1,2,3"


def test_imagenet_constants_are_shared_with_lsdsam_adapter() -> None:
    from lsd_app.models import lsdsam as lsdsam_module

    assert tuple(float(value) for value in lsdsam_module._PIXEL_MEAN) == pytest.approx(IMAGENET_RGB_MEAN)
    assert tuple(float(value) for value in lsdsam_module._PIXEL_STD) == pytest.approx(IMAGENET_RGB_STD)
