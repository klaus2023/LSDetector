import json
from pathlib import Path

import pytest

from lsd_app.config import ConfigError
from lsd_app.training.config import TRAINING_OUTPUT_SUBDIRS, load_training_config, normalize_training_config


def _base_config(tmp_path: Path) -> dict[str, object]:
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    (data_dir / "train" / "images").mkdir(parents=True)
    (data_dir / "train" / "masks").mkdir(parents=True)
    (data_dir / "val" / "images").mkdir(parents=True)
    (data_dir / "val" / "masks").mkdir(parents=True)
    ckpt_dir.mkdir()
    (ckpt_dir / "base.pt").write_text("dry-run checkpoint", encoding="utf-8")
    return {
        "training": {
            "output_dir": str(tmp_path / "training-output"),
            "data": {
                "train_images_dir": str(data_dir / "train" / "images"),
                "train_masks_dir": str(data_dir / "train" / "masks"),
                "val_images_dir": str(data_dir / "val" / "images"),
                "val_masks_dir": str(data_dir / "val" / "masks"),
            },
            "model": {
                "name": "lsdsam",
                "base_checkpoint": str(ckpt_dir / "base.pt"),
            },
            "stages": {
                "daft": {"alignment_mode": "mmd"},
                "tsft": {"alignment_mode": "none"},
            },
        }
    }


def test_json_training_config_loads_and_creates_standard_output_dirs(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps(_base_config(tmp_path)), encoding="utf-8")

    config = load_training_config(path)

    assert config.stage_order == ("taft", "daft", "tsft")
    assert config.stages["daft"].alignment_mode == "mmd"
    for name in TRAINING_OUTPUT_SUBDIRS:
        assert (config.output_dir / name).is_dir()


def test_create_output_dirs_false_does_not_create_dirs(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    output_dir = Path(raw["training"]["output_dir"])

    config = normalize_training_config(raw, create_output_dirs=False)

    assert config.output_dir == output_dir.resolve()
    assert not output_dir.exists()


def test_stage_order_can_be_overridden(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["training"]["stage_order"] = ["taft", "tsft"]

    config = normalize_training_config(raw, create_output_dirs=False)

    assert config.stage_order == ("taft", "tsft")


def test_unknown_stage_has_clear_error(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["training"]["stage_order"] = ["taft", "unknown"]

    with pytest.raises(ConfigError, match="Unknown training stage 'unknown'"):
        normalize_training_config(raw, create_output_dirs=False)


def test_alignment_mode_validates(tmp_path: Path) -> None:
    raw = _base_config(tmp_path)
    raw["training"]["stages"]["daft"] = {"alignment_mode": "mystery"}

    with pytest.raises(ConfigError, match="Invalid training.stages.daft.alignment_mode"):
        normalize_training_config(raw, create_output_dirs=False)


def test_env_vars_expand_and_missing_paths_warn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LSD_APP_TEST_DATA", str(tmp_path / "data"))
    raw = _base_config(tmp_path)
    raw["training"]["data"]["extra_index_path"] = "${LSD_APP_TEST_DATA}/missing-index.json"

    config = normalize_training_config(raw, create_output_dirs=False)

    assert config.data["extra_index_path"] == (tmp_path / "data" / "missing-index.json").resolve()
    assert any("extra_index_path" in warning for warning in config.warnings)
