import sys
from pathlib import Path

from lsd_app.training import build_training_plan
from lsd_app.training.config import normalize_training_config


def _base_config(tmp_path: Path) -> dict[str, object]:
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    (data_dir / "train" / "images").mkdir(parents=True)
    (data_dir / "train" / "masks").mkdir(parents=True)
    ckpt_dir.mkdir()
    (ckpt_dir / "base.pt").write_text("dry-run checkpoint", encoding="utf-8")
    return {
        "training": {
            "output_dir": str(tmp_path / "training-output"),
            "data": {
                "train_images_dir": str(data_dir / "train" / "images"),
                "train_masks_dir": str(data_dir / "train" / "masks"),
            },
            "model": {"base_checkpoint": str(ckpt_dir / "base.pt")},
            "stages": {
                "daft": {"alignment_mode": "coral"},
                "tsft": {"alignment_mode": "none"},
            },
        }
    }


def test_dry_run_plan_text_contains_default_stage_chain(tmp_path: Path) -> None:
    config = normalize_training_config(_base_config(tmp_path), create_output_dirs=False)

    plan = build_training_plan(config)

    assert [stage.name for stage in plan.stages] == ["taft", "daft", "tsft"]
    assert "taft -> daft -> tsft" in plan.dry_run_text()
    assert plan.stages[1].checkpoint_input == plan.stages[0].checkpoint_output
    assert plan.stages[2].checkpoint_input == plan.stages[1].checkpoint_output
    assert plan.stages[0].augmentation_profile is not None
    assert plan.stages[0].augmentation_profile["profile_id"] == "legacy-lsdsam-train-v1"
    assert any(step["name"] == "ImageNetNormalize" for step in plan.stages[0].augmentation_profile["post_transform"])


def test_dry_run_plan_does_not_import_torch(tmp_path: Path) -> None:
    sys.modules.pop("torch", None)
    config = normalize_training_config(_base_config(tmp_path), create_output_dirs=False)

    build_training_plan(config)

    assert "torch" not in sys.modules


def test_training_package_has_no_hardcoded_legacy_dataset_paths() -> None:
    forbidden = ("Noto", "Hunan", "Yaxia", "Landslide77K")
    for path in Path("lsd_app/training").glob("*.py"):
        if path.name.startswith("._"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in forbidden), path
