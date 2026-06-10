import json
from pathlib import Path

import pytest

from lsd_app.cli.main import main


def _write_source(path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = np.array(
        [
            [80, 220, 220, 80],
            [80, 220, 220, 80],
            [80, 80, 80, 80],
            [220, 80, 80, 220],
        ],
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        nodata=None,
    ) as dst:
        dst.write(data, 1)


def _run_config(tmp_path: Path, raster_path: Path) -> dict[str, object]:
    return {
        "input": {"raster_path": str(raster_path), "RGB_bands": "RGB_bands: 1,2,3"},
        "output": {"dir": str(tmp_path / "outputs")},
        "model": {"name": "lsdsam", "checkpoint_path": None},
        "tiling": {"tile_size": 4, "stride": 4},
        "inference": {"batch_size": 1, "threshold": 0.5},
        "postprocess": {"min_area_pixels": 0, "simplify_tolerance": 0.0, "connectivity": 8},
        "runtime": {"device": "cpu", "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def test_top_level_help_lists_run_and_finetune(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    stdout = capsys.readouterr().out
    assert "run" in stdout
    assert "finetune" in stdout


def test_run_executes_full_pipeline_with_mock_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raster_path = tmp_path / "source.tif"
    _write_source(raster_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_run_config(tmp_path, raster_path)), encoding="utf-8")

    assert main(["run", "--config", str(config_path), "--model", "mock"]) == 0

    stdout = capsys.readouterr().out
    for stage in ["prepare", "infer", "mosaic", "vectorize"]:
        assert f"[run] starting {stage}" in stdout
        assert f"[run] completed {stage}" in stdout
    assert "[run] starting visualize" not in stdout
    assert "[run] starting evaluate" not in stdout
    assert "[run] pipeline completed" in stdout

    output_dir = tmp_path / "outputs"
    assert (output_dir / "patch_index.geojson").exists()
    assert (output_dir / "infer_manifest.json").exists()
    assert (output_dir / "rasters" / "probability.tif").exists()
    assert (output_dir / "rasters" / "mask.tif").exists()
    assert (output_dir / "vectors" / "landslides.gpkg").exists()
    assert (output_dir / "metrics" / "vector_stats.json").exists()
    assert not (output_dir / "metrics" / "evaluation.json").exists()
    assert not (output_dir / "visualizations" / "overlay.png").exists()
    assert (tmp_path / "registry" / "prediction_runs.json").exists()


def test_run_stops_at_failed_stage_with_check_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_run_config(tmp_path, tmp_path / "missing.tif")), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", str(config_path), "--model", "mock"])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "Run failed at stage 'prepare'" in stderr
    assert "Input raster does not exist" in stderr
    assert "Check:" in stderr


def test_finetune_dry_run_prints_training_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "finetune.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                f"  output_dir: {tmp_path / 'training-output'}",
                "  model:",
                "    name: mock",
                "  stages:",
                "    taft:",
                "      training:",
                "        epochs: 1",
                "      params:",
                "        learning_rate: 0.001",
                "    daft:",
                "      alignment_mode: mmd",
                "    tsft:",
                "      alignment_mode: none",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["finetune", "--config", str(config_path), "--dry-run"]) == 0

    stdout = capsys.readouterr().out
    assert "[dry-run] finetune: model=mock" in stdout
    assert "no training has been launched" in stdout
    assert "taft -> daft -> tsft" in stdout
    assert "alignment: mmd" in stdout
    assert "learning_rate=0.001" in stdout
    assert "lsdsam-finetune-target" in stdout


def test_finetune_output_and_model_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "finetune.yaml"
    override_output = tmp_path / "override-output"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                f"  output_dir: {tmp_path / 'training-output'}",
                "  model:",
                "    name: lsdsam",
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "finetune",
            "--config",
            str(config_path),
            "--dry-run",
            "--model",
            "mock",
            "--output-dir",
            str(override_output),
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert "model=mock" in stdout
    assert f"output={override_output.resolve()}" in stdout
