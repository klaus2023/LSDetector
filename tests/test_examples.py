import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lsd_app.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "examples" / "generate_mock_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_mock_dataset", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
generate_mock_dataset = GENERATOR.main


def test_generate_mock_dataset_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")

    output_dir = tmp_path / "mock"

    assert (
        generate_mock_dataset(
            [
                "--output",
                str(output_dir),
                "--size",
                "16",
                "--seed",
                "7",
                "--with-truth",
            ]
        )
        == 0
    )

    assert (output_dir / "source.tif").exists()
    assert (output_dir / "truth.tif").exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "manifest.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["size"] == 16
    assert manifest["truth"]["enabled"] is True


def test_generated_config_can_load_config(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    output_dir = tmp_path / "mock"

    generate_mock_dataset(["--output", str(output_dir), "--size", "16", "--config-format", "json"])

    config = load_config(output_dir / "config.json", create_output_dirs=False)

    assert config.input_raster_path == output_dir / "source.tif"
    assert config.output_dir == output_dir / "outputs"
    assert config.model_name == "mock"


def test_seed_is_reproducible(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_mock_dataset(["--output", str(first), "--size", "16", "--seed", "22", "--with-truth"])
    generate_mock_dataset(["--output", str(second), "--size", "16", "--seed", "22", "--with-truth"])

    assert _sha256(first / "source.tif") == _sha256(second / "source.tif")
    assert _sha256(first / "truth.tif") == _sha256(second / "truth.tif")


def test_existing_output_dir_requires_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    output_dir = tmp_path / "mock"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(SystemExit, match="already exists"):
        generate_mock_dataset(["--output", str(output_dir)])

    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_default_output_is_not_repo_examples_data(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    output_dir = tmp_path / "chosen-output"

    generate_mock_dataset(["--output", str(output_dir)])

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["files"]["source"]).parent == output_dir
    assert output_dir.name != "data"
    assert "examples/data" not in str(output_dir).replace("\\", "/")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
