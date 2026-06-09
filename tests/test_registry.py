import json
from pathlib import Path

import pytest

import lsd_app.models.registry as model_registry_module
from lsd_app.core import paths as path_helpers
from lsd_app.registry import (
    RegistryError,
    delete_weight,
    list_datasets,
    list_models,
    list_weights,
    repair_registry_paths,
    register_dataset,
    register_weight,
)


def test_default_workspace_keeps_registry_out_of_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(path_helpers.ENV_WORKSPACE, raising=False)

    workspace = path_helpers.default_workspace_root()

    assert workspace == path_helpers.app_root().resolve()
    assert workspace.name != "outputs"
    assert path_helpers.registry_dir() == path_helpers.app_root().resolve() / "registry"


def test_model_registry_reports_input_modes(tmp_path: Path) -> None:
    rows = list_models(project_root=tmp_path)
    by_id = {row["model_id"]: row for row in rows}

    assert "lsdsam" not in by_id
    assert "lsdformer" not in by_id
    assert set(by_id) == {"LSDFormer", "LSDSAM-H", "LSDSAM-L", "LSDSAM-B"}
    assert by_id["LSDFormer"]["input_mode"] == "rgb"
    assert by_id["LSDFormer"]["channel_rule"] == "RGB"
    bundled_repo = Path(model_registry_module.__file__).resolve().parent / "model_repository"
    assert by_id["LSDFormer"]["code_path"] == str((bundled_repo / "LSDFormer" / "model").resolve())
    assert by_id["LSDFormer"]["model_options"]["model_root"] == str((bundled_repo / "LSDFormer").resolve())
    for model_id in ("LSDSAM-H", "LSDSAM-L", "LSDSAM-B"):
        assert by_id[model_id]["input_mode"] == "rgb_wavelets"
        assert by_id[model_id]["edge_mode"] == "wavelets"
        assert by_id[model_id]["channel_rule"] == "RGB + wavelets"
        assert by_id[model_id]["code_path"] == str((bundled_repo / "LSDSAM").resolve())
        assert by_id[model_id]["model_options"]["legacy_root"] == str((bundled_repo / "LSDSAM").resolve())


def test_register_weight_persists_and_checks_path(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"checkpoint")

    entry = register_weight(
        registry_root=tmp_path / "outputs",
        model_id="lsdsam",
        stage="TSFT",
        version="v1",
        path=weight_path,
        weight_id="tsft-main",
        metrics_summary="f1=0.8",
        notes="few-shot set",
        weight_type="lora_task",
        adapter_profile="legacy-lora",
        image_size=512,
        lora_base_r=32,
        lora_base_alpha=64,
        lora_task_r=16,
        lora_task_alpha=32,
        lora_dropout=0.05,
        peft_adapter_name="Task-Specific",
        target_modules="image_encoder.blocks.*",
    )

    assert entry.weight_id == "tsft-main"
    rows = list_weights(registry_root=tmp_path / "outputs")
    assert rows[0]["model_id"] == "LSDSAM-B"
    assert rows[0]["stage"] == "TSFT"
    assert rows[0]["exists"] is True
    assert rows[0]["adapter_name"] == "lsdsam-fewshot"
    assert rows[0]["input_mode"] == "rgb_wavelets"
    assert rows[0]["weight_type"] == "lora_task"
    assert rows[0]["adapter_profile"] == "legacy-lora"
    assert rows[0]["lora_task_r"] == 16
    assert rows[0]["peft_adapter_name"] == "Task-Specific"
    assert rows[0]["stored"] is True
    assert rows[0]["source_path"] == str(weight_path.resolve())
    assert rows[0]["path"] == "@weights/TSFT/LSDSAM-B/tsft-main/tsft-main.pth"
    assert rows[0]["resolved_path"].endswith(
        "outputs/registry/weight_repository/TSFT/LSDSAM-B/tsft-main/tsft-main.pth"
    )
    assert rows[0]["file_size_bytes"] == len(b"checkpoint")
    assert rows[0]["parameter_count"] is None
    assert Path(rows[0]["resolved_path"]).read_bytes() == b"checkpoint"

    with pytest.raises(RegistryError, match="Weight path does not exist"):
        register_weight(
            registry_root=tmp_path / "outputs",
            model_id="lsdsam",
            stage="DAFT",
            version="missing",
            path=tmp_path / "missing.pth",
        )


def test_delete_weight_removes_registry_entry_and_internal_copy(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"checkpoint")
    events: list[tuple[int, int]] = []

    entry = register_weight(
        registry_root=tmp_path / "outputs",
        model_id="lsdsam",
        stage="DAFT",
        version="v1",
        path=weight_path,
        weight_id="daft-main",
        progress_callback=lambda copied, total: events.append((copied, total)),
    )
    stored_path = Path(entry.path)
    assert stored_path.exists()
    assert events[0] == (0, len(b"checkpoint"))
    assert events[-1] == (len(b"checkpoint"), len(b"checkpoint"))

    result = delete_weight(registry_root=tmp_path / "outputs", weight_id="daft-main")

    assert result["weight_id"] == "daft-main"
    assert result["removed_registry_entry"] is True
    assert result["removed_path"] == str(stored_path.parent)
    assert not stored_path.exists()
    assert list_weights(registry_root=tmp_path / "outputs") == []
    assert weight_path.exists()


def test_register_weight_allows_same_version_across_stages(tmp_path: Path) -> None:
    daft_path = tmp_path / "daft.pth"
    tsft_path = tmp_path / "tsft.pth"
    daft_path.write_bytes(b"daft")
    tsft_path.write_bytes(b"tsft")

    daft_entry = register_weight(
        registry_root=tmp_path / "outputs",
        model_id="LSDFormer",
        stage="DAFT",
        version="LSDFormer_1",
        path=daft_path,
        weight_id="LSDFormer_1",
    )
    tsft_entry = register_weight(
        registry_root=tmp_path / "outputs",
        model_id="LSDFormer",
        stage="TSFT",
        version="LSDFormer_1",
        path=tsft_path,
        weight_id="LSDFormer_1",
    )

    rows = sorted(list_weights(registry_root=tmp_path / "outputs"), key=lambda row: row["stage"])
    assert daft_entry.weight_id == "LSDFormer_1"
    assert tsft_entry.weight_id == "TSFT-LSDFormer_1"
    assert [(row["stage"], row["version"], row["weight_id"]) for row in rows] == [
        ("DAFT", "LSDFormer_1", "LSDFormer_1"),
        ("TSFT", "LSDFormer_1", "TSFT-LSDFormer_1"),
    ]
    assert Path(rows[0]["resolved_path"]).read_bytes() == b"daft"
    assert Path(rows[1]["resolved_path"]).read_bytes() == b"tsft"


def test_registered_weight_token_survives_workspace_move(tmp_path: Path) -> None:
    workspace = tmp_path / "outputs"
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"checkpoint")
    register_weight(
        registry_root=workspace,
        model_id="LSDFormer",
        stage="TAFT",
        version="v1",
        path=weight_path,
        weight_id="portable-weight",
    )

    moved = tmp_path / "moved_outputs"
    workspace.rename(moved)

    rows = list_weights(registry_root=moved)
    assert rows[0]["path"] == "@weights/TAFT/LSDFormer/portable-weight/portable-weight.pth"
    assert rows[0]["exists"] is True
    assert Path(rows[0]["resolved_path"]).read_bytes() == b"checkpoint"


def test_repair_registry_paths_rewrites_internal_absolutes(tmp_path: Path) -> None:
    workspace = tmp_path / "outputs"
    registry = workspace / "registry"
    stored = registry / "weight_repository" / "TAFT" / "LSDFormer" / "w1" / "w1.pth"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"checkpoint")
    weights_path = registry / "weights.json"
    weights_path.write_text(
        json.dumps(
            [
                {
                    "weight_id": "w1",
                    "model_id": "LSDFormer",
                    "stage": "TAFT",
                    "version": "v1",
                    "path": str(stored),
                    "source_path": str(tmp_path / "source.pth"),
                    "stored": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = repair_registry_paths(registry_root=workspace)

    assert result["changed_count"] == 1
    repaired = json.loads(weights_path.read_text(encoding="utf-8"))
    assert repaired[0]["path"] == "@weights/TAFT/LSDFormer/w1/w1.pth"
    assert repaired[0]["source_path"] == str(tmp_path / "source.pth")
    assert weights_path.with_suffix(".json.bak_portable_paths").exists()


def test_register_weight_resolves_configured_path_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    weight_path = data_root / "LSDSAM_finetune" / "models_save" / "LS70k" / "weights.pth"
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"checkpoint")
    monkeypatch.setenv("LSD_APP_PATH_ALIASES", f"/portable/data={data_root}")

    entry = register_weight(
        registry_root=tmp_path / "outputs",
        model_id="lsdsam",
        stage="DAFT",
        version="alias",
        path="/portable/data/LSDSAM_finetune/models_save/LS70k/weights.pth",
        weight_id="alias-weight",
    )

    assert entry.source_path == weight_path.resolve()
    assert Path(entry.path).exists()


def test_register_dataset_handles_daft_images_only_and_taft_gt_persistence(tmp_path: Path) -> None:
    imagery = tmp_path / "source.tif"
    imagery.write_bytes(b"raster")

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 3,2,1",
        tile_schema="packaged_images",
        source="customer",
        notes="baseline for evaluation",
    )

    assert entry.dataset_id.startswith("daft-")
    rows = list_datasets(registry_root=tmp_path / "outputs")
    assert rows[0]["dataset_class"] == "DAFT"
    assert rows[0]["label_path"] is None
    assert rows[0]["truth_path"] is None
    assert rows[0]["has_gt"] is False
    assert rows[0]["RGB_bands_class"] == "RGB bands 3,2,1"
    assert rows[0]["class"] == "RGB bands 3,2,1"

    truth = tmp_path / "truth.gpkg"
    truth.write_bytes(b"vector")
    tsft = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT baseline",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=truth,
        has_gt=True,
        RGB_bands="RGB_bands: 3,2,1",
        tile_schema="tile_size=512;stride=256",
        source="customer",
        notes="baseline for evaluation",
    )
    assert tsft.dataset_class == "TSFT"
    assert tsft.has_gt is True

    taft = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TAFT imagery only",
        dataset_class="TAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 3,2,1",
    )
    assert taft.dataset_class == "TAFT"
    assert taft.has_gt is False

    with pytest.raises(RegistryError, match="Datasets marked has_gt require"):
        register_dataset(
            registry_root=tmp_path / "outputs",
            name="Missing truth",
            dataset_class="TSFT",
            imagery_path=imagery,
            label_path=None,
            has_gt=True,
            RGB_bands="RGB_bands: 3,2,1",
        )


def test_list_datasets_does_not_migrate_legacy_tsft_test_split(tmp_path: Path) -> None:
    registry_root = tmp_path / "outputs"
    registry = registry_root / "registry"
    tsft_root = registry / "dataset_store" / "TSFT" / "legacy-tsft"
    train_image = tsft_root / "train" / "images" / "train.tif"
    train_label = tsft_root / "train" / "labels" / "train.tif"
    test_image = tsft_root / "test" / "images" / "test.tif"
    test_label = tsft_root / "test" / "labels" / "test.tif"
    for path in (train_image, train_label, test_image, test_label):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    manifest = {
        "dataset_id": "tsft-legacy-tsft",
        "dataset_class": "TSFT",
        "version": "legacy-tsft",
        "root_dir": str(tsft_root),
        "has_gt": True,
        "RGB_bands": "RGB_bands: 1,2,3",
        "sample_shape_layout": "(H, W, C)",
        "sample_dtype": "float32",
        "subsets": [
            {
                "subset": "legacy-tsft",
                "sample_count": 2,
                "train_count": 1,
                "test_count": 1,
                "samples": [
                    {"sample_id": "train", "split": "train", "image_path": str(train_image), "label_path": str(train_label)},
                    {"sample_id": "test", "split": "test", "image_path": str(test_image), "label_path": str(test_label)},
                ],
            }
        ],
        "standardization": {"label_polarity": "foreground_white"},
    }
    manifest_path = tsft_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "datasets.json").write_text(
        json.dumps(
            [
                {
                    "dataset_id": "tsft-legacy-tsft",
                    "name": "legacy-tsft",
                    "dataset_class": "TSFT",
                    "imagery_path": str(tsft_root),
                    "label_path": str(tsft_root),
                    "has_gt": True,
                    "RGB_bands": "RGB_bands: 1,2,3",
                        "RGB_bands_class": "RGB bands 1,2,3",
                    "tile_schema": "packaged_images_and_labels",
                    "source": "user",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "notes": "legacy",
                    "description": "legacy",
                    "dataset_type": "TSFT",
                    "truth_path": str(tsft_root),
                    "version": "legacy-tsft",
                    "repository_path": str(tsft_root),
                    "manifest_path": str(manifest_path),
                    "sample_shape_layout": "(H, W, C)",
                    "sample_dtype": "float32",
                    "sample_count": 2,
                    "train_count": 1,
                    "test_count": 1,
                    "subset_count": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    first = list_datasets(registry_root=registry_root)
    second = list_datasets(registry_root=registry_root)
    eval_rows = [row for row in second if row.get("dataset_class") == "Evaluation"]

    assert eval_rows == []
    assert first == second
    assert len(second) == 1
    tsft_row = next(row for row in second if row.get("dataset_class") == "TSFT")
    assert tsft_row["test_count"] == 1
    tsft_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert tsft_manifest["subsets"][0]["test_count"] == 1
    assert [sample["split"] for sample in tsft_manifest["subsets"][0]["samples"]] == ["train", "test"]
