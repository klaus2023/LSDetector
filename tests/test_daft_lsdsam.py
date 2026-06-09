import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsd_app.registry import register_dataset
from lsd_app.models.lsdsam import _checkpoint_map_location
from lsd_app.training import daft_lsdsam
from lsd_app.training.daft_lsdsam import (
    DaftLSDSAMConfig,
    DEFAULT_DAFT_NUM_WORKERS,
    DaftTrainingError,
    build_daft_training_payload,
    _effective_daft_num_workers,
    _label_to_foreground_mask,
    _legacy_root_from_config,
    learning_rate_for_epoch,
    parse_halve_epochs,
    resolve_daft_dataset_materialization,
    _lsdsam_dataset_from_materialization,
    resolve_training_device,
)


def test_training_device_prefers_gpu_and_falls_back_to_cpu() -> None:
    cuda_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    cpu_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    assert resolve_training_device("auto", cuda_torch) == "cuda"
    assert resolve_training_device("auto", cpu_torch) == "cpu"
    assert resolve_training_device("cpu", cuda_torch) == "cuda"


def test_lsdsam_checkpoint_loads_stage_cuda_checkpoints_on_cpu() -> None:
    assert _checkpoint_map_location("cuda") == "cpu"
    assert _checkpoint_map_location("cuda:0") == "cpu"
    assert _checkpoint_map_location("cpu") == "cpu"


def test_daft_default_workers_are_low_memory_safe() -> None:
    payload = build_daft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        source_dataset_id="source-1",
        target_dataset_id="target-1",
        alignment_mode="IST",
        total_epochs=1,
        initial_lr=2e-4,
        halve_epochs="",
        output_dir="outputs/daft",
    )

    assert DEFAULT_DAFT_NUM_WORKERS == 0
    assert payload["num_workers"] == 0
    assert payload["save_optimizer_state"] is False


def test_daft_training_payload_can_request_exact_optimizer_resume() -> None:
    payload = build_daft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        source_dataset_id="source-1",
        target_dataset_id="target-1",
        alignment_mode="IST",
        total_epochs=1,
        initial_lr=2e-4,
        halve_epochs="",
        output_dir="outputs/daft",
        save_optimizer_state=True,
    )

    assert payload["save_optimizer_state"] is True


def test_daft_workers_downgrade_for_cuda_strict_overcommit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lsd_app.training.daft_lsdsam._strict_memory_overcommit_enabled", lambda: True)

    assert _effective_daft_num_workers(8, "cuda") == 0
    assert _effective_daft_num_workers(8, "cuda:0") == 0
    assert _effective_daft_num_workers(8, "cpu") == 8


def test_daft_default_legacy_root_uses_bundled_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_dir = tmp_path / "LSD__APP"
    legacy_root = tmp_path / "LSDSAM_finetune"
    app_dir.mkdir()
    legacy_root.mkdir()
    monkeypatch.chdir(app_dir)

    resolved = _legacy_root_from_config(
        DaftLSDSAMConfig(
            registry_root=tmp_path / "outputs",
            source_dataset_id="source",
            target_dataset_id="target",
            base_weight_id="weight",
            output_dir=tmp_path / "daft",
        )
    )

    assert resolved == (
        Path(daft_lsdsam.__file__).resolve().parents[2]
        / "lsd_app"
        / "models"
        / "model_repository"
        / "LSDSAM"
    ).resolve()


def test_daft_explicit_legacy_root_can_use_external_repository(tmp_path: Path) -> None:
    legacy_root = tmp_path / "LSDSAM_finetune"
    legacy_root.mkdir()

    resolved = _legacy_root_from_config(
        DaftLSDSAMConfig(
            registry_root=tmp_path / "outputs",
            source_dataset_id="source",
            target_dataset_id="target",
            base_weight_id="weight",
            output_dir=tmp_path / "daft",
            legacy_root=legacy_root,
        )
    )

    assert resolved == legacy_root.resolve()


def test_build_lsdsam_for_daft_uses_full_registered_peft_base_weight_without_default_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    weight = tmp_path / "taft_full_peft.pth"
    weight.write_bytes(b"fake")

    class FakeTensor:
        shape = (16, 4)

    state = {
        "base_model.model.image_encoder.blocks.0.attn.qkv.base_layer.weight": FakeTensor(),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_A.default.weight": FakeTensor(),
        "base_model.model.image_encoder.blocks.0.attn.qkv.lora_B.default.weight": FakeTensor(),
    }

    class FakeParam:
        def __init__(self) -> None:
            self.requires_grad = True

    class FakeImageEncoder:
        def resize_pos_embed(self, new_img_size: int) -> None:
            calls.append(("resize_pos_embed", new_img_size))

    class FakeModel:
        image_encoder = FakeImageEncoder()

        def parameters(self):
            return [FakeParam()]

        def named_modules(self):
            return [("image_encoder.blocks.0.attn.qkv", object())]

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("model_load_state_dict", state_dict, strict))

        def add_adapter(self, name, config):
            calls.append(("add_adapter", name, config.r, config.lora_alpha))

        def set_adapter(self, name):
            calls.append(("set_adapter", name))

    class FakeBuildModule:
        __file__ = str(tmp_path / "build_sam_diy.py")

        @staticmethod
        def build_sam_h(checkpoint, map_location="cpu"):
            calls.append(("build_sam_h", checkpoint, map_location))
            return FakeModel()

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=False):
            calls.append(("torch.load", str(path), map_location, weights_only))
            if str(path).endswith("sam_vit_h_4b8939.pth"):
                raise AssertionError("full registered base_weight should not load default SAM checkpoint")
            return {"model_state_dict": state}

    class FakeLoraConfig:
        def __init__(self, r, lora_alpha, target_modules, lora_dropout, bias):
            self.r = r
            self.lora_alpha = lora_alpha
            self.target_modules = target_modules
            self.lora_dropout = lora_dropout
            self.bias = bias

    class FakePeftModel:
        def __init__(self, model):
            self.model = model

        def load_state_dict(self, state_dict, strict=False):
            calls.append(("peft_load_state_dict", state_dict, strict))

        def merge_and_unload(self):
            calls.append("merge_and_unload")
            return self.model

    class FakePeft:
        LoraConfig = FakeLoraConfig

        @staticmethod
        def get_peft_model(model, config, adapter_name="default"):
            calls.append(("get_peft_model", adapter_name, config.r, config.lora_alpha))
            return FakePeftModel(model)

    import lsd_app.models.lsdsam as lsdsam_module

    monkeypatch.setattr(lsdsam_module, "_import_legacy_build_module", lambda _root: FakeBuildModule)

    model, base_checkpoint = daft_lsdsam._build_lsdsam_for_daft(
        config=DaftLSDSAMConfig(
            registry_root=tmp_path / "outputs",
            source_dataset_id="source",
            target_dataset_id="target",
            base_weight_id="LSDSAM_H_LS77K",
            output_dir=tmp_path / "daft",
            model_id="LSDSAM-H",
        ),
        weight_row={"path": str(weight), "_registry_root": str(tmp_path / "outputs")},
        torch=FakeTorch,
        peft=FakePeft,
        device="cpu",
    )

    assert model is not None
    assert base_checkpoint is None
    assert ("build_sam_h", False, "cpu") in calls
    assert ("get_peft_model", "default", 16, 64) in calls
    assert "merge_and_unload" in calls
    assert ("add_adapter", "Task-Specific", 16, 32) in calls


def test_halve_epochs_parse_and_lr_schedule() -> None:
    assert parse_halve_epochs("8, 6,6") == (6, 8)
    assert learning_rate_for_epoch(epoch_number=1, initial_lr=2e-4, halve_epochs=(6, 8)) == pytest.approx(2e-4)
    assert learning_rate_for_epoch(epoch_number=6, initial_lr=2e-4, halve_epochs=(6, 8)) == pytest.approx(1e-4)
    assert learning_rate_for_epoch(epoch_number=8, initial_lr=2e-4, halve_epochs=(6, 8)) == pytest.approx(5e-5)
    with pytest.raises(DaftTrainingError, match="positive"):
        parse_halve_epochs("0")


def test_daft_training_payload_is_json_ready_and_auto_device() -> None:
    payload = build_daft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        source_dataset_id="source-1",
        target_dataset_id="target-1",
        alignment_mode="coral",
        total_epochs=10,
        initial_lr=2e-4,
        halve_epochs="6,8",
        output_dir="outputs/daft",
    )

    assert payload["alignment_mode"] == "CoRAL"
    assert payload["halve_epochs"] == [6, 8]
    assert payload["device"] == "auto"


def test_evaluate_model_progress_without_epoch_context_has_no_epoch_placeholder() -> None:
    source = inspect.getsource(daft_lsdsam._evaluate_model)

    assert "Epoch {epoch_text}" not in source
    assert "Epoch ?" not in source
    assert "Evaluating {progress_split}" in source


def test_resolve_daft_dataset_materialization_taft_and_daft(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("rasterio")
    source_images = tmp_path / "source_images"
    source_labels = tmp_path / "source_labels"
    for index in range(4):
        _write_chip(source_images / f"src_{index}.tif", value=10 + index)
        _write_label(source_labels / f"src_{index}.tif", value=index % 2)
    target_images = tmp_path / "target_images"
    for index in range(2):
        _write_chip(target_images / f"tgt_{index}.tif", value=40 + index)

    source = register_dataset(
        registry_root=tmp_path / "outputs",
        name="Source",
        dataset_class="TAFT",
        imagery_path=source_images,
        label_path=source_labels,
        RGB_bands="RGB_bands: 3,2,1",
        version="source-v1",
        standardize=True,
    )
    target = register_dataset(
        registry_root=tmp_path / "outputs",
        name="Target",
        dataset_class="DAFT",
        imagery_path=target_images,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-v1",
        standardize=True,
    )

    source_materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id=source.dataset_id,
    )
    target_materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id=target.dataset_id,
    )

    assert len(source_materialized.supervised_train_samples()) == 3
    assert all(sample.label_path is not None for sample in source_materialized.supervised_train_samples())
    assert len(target_materialized.target_train_samples()) == 2
    assert target_materialized.target_eval_samples() == ()


def test_resolve_taft_tree_subset_manifest_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "outputs" / "registry" / "dataset_store" / "TAFT" / "tree-v1"
    subset = root / "subsets" / "group"
    image = subset / "train" / "images" / "a.tif"
    label = subset / "train" / "labels" / "a.tif"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_bytes(b"label")
    subset_manifest = subset / "subset_manifest.json"
    subset_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "a",
                        "split": "train",
                        "image_path": str(image),
                        "label_path": str(label),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "taft-tree-v1",
                "dataset_class": "TAFT",
                "version": "tree-v1",
                "root_dir": str(root),
                "has_gt": True,
                "RGB_bands": "RGB_bands: 3,2,1",
                "subsets": [{"subset_manifest_path": str(subset_manifest), "RGB_bands": "RGB_bands: 3,2,1"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lsd_app.training.daft_lsdsam.list_datasets",
        lambda registry_root: [
            {
                "dataset_id": "taft-tree-v1",
                "dataset_class": "TAFT",
                "repository_path": str(root),
                "manifest_path": str(manifest),
                "has_gt": True,
            }
        ],
    )

    materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id="taft-tree-v1",
    )

    assert [sample.sample_id for sample in materialized.supervised_train_samples()] == ["a"]


def test_resolve_taft_tree_falls_back_to_source_root_when_store_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "Landslide70K"
    image = source_root / "CAS" / "palu" / "train" / "images" / "a.tif"
    label = source_root / "CAS" / "palu" / "train" / "labels" / "a.tif"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_bytes(b"label")

    root = tmp_path / "outputs" / "registry" / "dataset_store" / "TAFT" / "tree-v1"
    manifest = root / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "taft-tree-v1",
                "dataset_class": "TAFT",
                "version": "tree-v1",
                "root_dir": str(root),
                "source_root": str(source_root),
                "has_gt": True,
                "RGB_bands": "mixed",
                "subsets": [
                    {
                        "subset": "CAS/palu",
                        "source_rel": "CAS/palu",
                        "subset_id": "CAS__palu",
                        "RGB_bands": "RGB_bands: 1,2,3",
                        "label_polarity": "foreground_black",
                        "subset_manifest_path": str(root / "subsets" / "CAS__palu" / "subset_manifest.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lsd_app.training.daft_lsdsam.list_datasets",
        lambda registry_root: [
            {
                "dataset_id": "taft-tree-v1",
                "dataset_class": "TAFT",
                "repository_path": str(root),
                "manifest_path": str(manifest),
                "has_gt": True,
            }
        ],
    )

    materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id="taft-tree-v1",
    )

    supervised = materialized.supervised_train_samples()
    assert len(supervised) == 1
    assert supervised[0].sample_id == "CAS/palu/train/a"
    assert supervised[0].image_path == image
    assert supervised[0].label_path == label
    assert supervised[0].RGB_bands == "RGB_bands: 1,2,3"
    assert supervised[0].label_polarity == "foreground_black"


def test_label_to_foreground_mask_respects_black_polarity() -> None:
    np = pytest.importorskip("numpy")
    label = np.array([[0, 255], [10, 128]], dtype=np.uint8)

    black = _label_to_foreground_mask(label, label_polarity="foreground_black", np=np)
    white = _label_to_foreground_mask(label, label_polarity="foreground_white", np=np)

    assert black.tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert white.tolist() == [[0.0, 1.0], [0.0, 1.0]]


def _write_chip(path: Path, *, value: int) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(
        [
            np.full((4, 5), value, dtype=np.uint16),
            np.full((4, 5), value + 1, dtype=np.uint16),
            np.full((4, 5), value + 2, dtype=np.uint16),
        ],
        axis=0,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=3,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data)


def _write_label(path: Path, *, value: int) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, 5), value * 255, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data, 1)




def _fake_torch_with_concat_dataset():
    class FakeConcatDataset:
        def __init__(self, datasets):
            self.datasets = tuple(datasets)

        def __len__(self):
            return sum(len(dataset) for dataset in self.datasets)

    return SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(ConcatDataset=FakeConcatDataset)))

def _write_simple_manifest_dataset(
    root: Path,
    *,
    dataset_id: str,
    dataset_class: str,
    count: int,
    labeled: bool,
    split: str = "train",
) -> Path:
    samples = []
    for index in range(count):
        image = root / "subsets" / "main" / split / "images" / f"{dataset_id}_{index}.tif"
        label = root / "subsets" / "main" / split / "labels" / f"{dataset_id}_{index}.tif"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label_path = None
        if labeled:
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_bytes(b"label")
            label_path = str(label)
        samples.append(
            {
                "sample_id": f"{dataset_id}-{index}",
                "split": split,
                "image_path": str(image),
                "label_path": label_path,
            }
        )
    subset_manifest = root / "subsets" / "main" / "subset_manifest.json"
    subset_manifest.parent.mkdir(parents=True, exist_ok=True)
    subset_manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_class": dataset_class,
                "version": dataset_id,
                "root_dir": str(root),
                "has_gt": labeled,
                "RGB_bands": "RGB_bands: 3,2,1",
                "subsets": [
                    {
                        "subset_id": "main",
                        "RGB_bands": "RGB_bands: 3,2,1",
                        "subset_manifest_path": str(subset_manifest),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _concat_registry_rows(tmp_path: Path, *, dataset_class: str, sources: list[tuple[str, Path, Path, int]]) -> list[dict[str, object]]:
    concat_root = tmp_path / "outputs" / "registry" / "dataset_store" / dataset_class / "concat"
    concat_manifest = concat_root / "manifest.json"
    concat_manifest.parent.mkdir(parents=True, exist_ok=True)
    concat_manifest.write_text(
        json.dumps(
            {
                "dataset_id": f"{dataset_class.lower()}-concat",
                "dataset_class": dataset_class,
                "version": "concat",
                "root_dir": str(concat_root),
                "is_concat": True,
                "concat_backend": "torch.utils.data.ConcatDataset",
                "concat_sources": [
                    {
                        "dataset_class": dataset_class,
                        "version": source_id,
                        "manifest_path": str(manifest),
                        "root_dir": str(root),
                        "sample_count": sample_count,
                    }
                    for source_id, manifest, root, sample_count in sources
                ],
            }
        ),
        encoding="utf-8",
    )
    return [
        {
            "dataset_id": f"{dataset_class.lower()}-concat",
            "dataset_class": dataset_class,
            "repository_path": str(concat_root),
            "manifest_path": str(concat_manifest),
            "has_gt": dataset_class != "DAFT",
        }
    ]


def test_resolve_daft_concat_manifest_from_source_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_root = tmp_path / "store" / "DAFT" / "first"
    second_root = tmp_path / "store" / "DAFT" / "second"
    first_manifest = _write_simple_manifest_dataset(first_root, dataset_id="daft-a", dataset_class="DAFT", count=2, labeled=False)
    second_manifest = _write_simple_manifest_dataset(second_root, dataset_id="daft-b", dataset_class="DAFT", count=3, labeled=False)
    rows = _concat_registry_rows(
        tmp_path,
        dataset_class="DAFT",
        sources=[("daft-a", first_manifest, first_root, 2), ("daft-b", second_manifest, second_root, 3)],
    )
    monkeypatch.setattr("lsd_app.training.daft_lsdsam.list_datasets", lambda registry_root: rows)

    materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id="daft-concat",
    )

    assert materialized.is_concat is True
    assert len(materialized.source_materializations) == 2
    assert len(materialized.target_train_samples()) == 5
    assert [source.version for source in materialized.source_materializations] == ["daft-a", "daft-b"]


def test_lsdsam_daft_concat_builds_torch_concat_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch_with_concat_dataset()
    first_root = tmp_path / "store" / "DAFT" / "first"
    second_root = tmp_path / "store" / "DAFT" / "second"
    first_manifest = _write_simple_manifest_dataset(first_root, dataset_id="daft-a", dataset_class="DAFT", count=1, labeled=False)
    second_manifest = _write_simple_manifest_dataset(second_root, dataset_id="daft-b", dataset_class="DAFT", count=2, labeled=False)
    rows = _concat_registry_rows(
        tmp_path,
        dataset_class="DAFT",
        sources=[("daft-a", first_manifest, first_root, 1), ("daft-b", second_manifest, second_root, 2)],
    )
    monkeypatch.setattr("lsd_app.training.daft_lsdsam.list_datasets", lambda registry_root: rows)
    materialized = resolve_daft_dataset_materialization(registry_root=tmp_path / "outputs", dataset_id="daft-concat")

    dataset = _lsdsam_dataset_from_materialization(
        torch,
        materialized,
        sample_selector="target_train_samples",
        dataset_cls=daft_lsdsam._LSDSAMImagesDataset,
        image_size=16,
        cv2=object(),
        np=object(),
    )

    assert isinstance(dataset, torch.utils.data.ConcatDataset)
    assert len(dataset) == 3


def test_concat_manifest_rejects_source_class_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "store" / "TSFT" / "source"
    source_manifest = _write_simple_manifest_dataset(source_root, dataset_id="tsft-a", dataset_class="TSFT", count=1, labeled=True)
    rows = _concat_registry_rows(
        tmp_path,
        dataset_class="DAFT",
        sources=[("tsft-a", source_manifest, source_root, 1)],
    )
    monkeypatch.setattr("lsd_app.training.daft_lsdsam.list_datasets", lambda registry_root: rows)

    with pytest.raises(DaftTrainingError, match="expected source class DAFT"):
        resolve_daft_dataset_materialization(registry_root=tmp_path / "outputs", dataset_id="daft-concat")

def test_evaluation_metrics_match_legacy_segmentation_metric_formula() -> None:
    torch = pytest.importorskip("torch")

    from lsd_app.training.daft_lsdsam import _legacy_metrics_from_predictions

    labels = torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.int64)
    predictions = torch.tensor([1, 0, 1, 1, 1, 0, 1], dtype=torch.int64)
    metrics = _legacy_metrics_from_predictions(predictions, labels, torch=torch, device="cpu")

    assert metrics["confusion_matrix"] == [[1.0, 2.0], [1.0, 3.0]]
    assert metrics["PA"] == pytest.approx(4 / 7)
    assert metrics["CPA"] == pytest.approx([1 / 3, 3 / 4])
    assert metrics["MPA"] == pytest.approx(((1 / 3) + (3 / 4)) / 2)
    assert metrics["IoU"] == pytest.approx([0.25, 0.5])
    assert metrics["mIoU"] == pytest.approx(0.375)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["F1"] == pytest.approx(2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4)))
    assert metrics["kappa"] == pytest.approx(2 / 23)


def test_daft_selection_summary_uses_last_checkpoint_without_evaluation() -> None:
    from lsd_app.training.daft_lsdsam import _training_selection_summary

    assert _training_selection_summary(best_score=None, best_epoch=10, metric_scope="train") == "selection=last_checkpoint;epoch=10"
    assert _training_selection_summary(best_score=0.7, best_epoch=3, metric_scope="eval") == "best_eval_mIoU=0.7000;best_epoch=3"


def test_daft_epoch_logic_does_not_evaluate_source_dataset() -> None:
    source = Path("lsd_app/training/daft_lsdsam.py").read_text(encoding="utf-8")

    assert "source_eval_dataset" not in source
    assert "train_metrics = _evaluate_model" not in source
    assert source.count("_evaluate_model(") == 2
    assert "progress_callback=emit" in source
    assert '"batch_losses": epoch_batch_losses' in source
    assert '"last_checkpoint"' in source
    assert "checkpoint_to_merge = best_checkpoint if evaluation_samples" in source



def test_daft_dispatches_lsdformer_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake_module = types.ModuleType("lsd_app.training.lsdformer")

    def fake_run(config, progress_callback=None):
        return {"model_id": config.model_id, "progress_callback": progress_callback}

    fake_module.run_lsdformer_daft = fake_run
    monkeypatch.setitem(sys.modules, "lsd_app.training.lsdformer", fake_module)

    result = daft_lsdsam.run_lsdsam_daft(
        DaftLSDSAMConfig(
            registry_root=tmp_path / "outputs",
            source_dataset_id="source",
            target_dataset_id="target",
            base_weight_id="weight",
            output_dir=tmp_path / "daft",
            model_id="lsdformer-fast",
        ),
        progress_callback=lambda payload: None,
    )

    assert result["model_id"] == "lsdformer-fast"
    assert callable(result["progress_callback"])



def test_lora_checkpoint_is_atomic_and_omits_optimizer_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.pth"
    path.write_bytes(b"old-checkpoint")

    class FakeModel:
        def state_dict(self):
            return {"lora.weight": 1}

    class FakeOptimizer:
        def state_dict(self):
            raise AssertionError("optimizer state should not be saved by default")

    class FakeTorch:
        saved_payload = None

        @staticmethod
        def save(payload, target_path):
            FakeTorch.saved_payload = payload
            Path(target_path).write_bytes(b"new-checkpoint")

    materialized = SimpleNamespace(dataset_id="dataset")
    config = DaftLSDSAMConfig(
        registry_root=tmp_path / "outputs",
        source_dataset_id="source",
        target_dataset_id="target",
        base_weight_id="weight",
        output_dir=tmp_path / "daft",
    )
    monkeypatch.setattr(daft_lsdsam, "_lora_checkpoint_state_dict", lambda model: {"adapter.weight": 1})

    daft_lsdsam._save_lora_checkpoint(
        FakeTorch,
        path,
        model=FakeModel(),
        config=config,
        epoch=1,
        score=None,
        source=materialized,
        target=materialized,
        optimizer=FakeOptimizer(),
    )

    assert path.read_bytes() == b"new-checkpoint"
    assert FakeTorch.saved_payload["model_state_dict"] == {"adapter.weight": 1}
    assert "optimizer_state_dict" not in FakeTorch.saved_payload


def test_lora_checkpoint_failed_temp_write_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "latest.pth"
    path.write_bytes(b"old-checkpoint")

    class FakeModel:
        def state_dict(self):
            return {"lora.weight": 1}

    class FailingTorch:
        @staticmethod
        def save(payload, target_path):
            Path(target_path).write_bytes(b"partial")
            raise RuntimeError("disk write failed")

    materialized = SimpleNamespace(dataset_id="dataset")
    config = DaftLSDSAMConfig(
        registry_root=tmp_path / "outputs",
        source_dataset_id="source",
        target_dataset_id="target",
        base_weight_id="weight",
        output_dir=tmp_path / "daft",
    )
    monkeypatch.setattr(daft_lsdsam, "_lora_checkpoint_state_dict", lambda model: {"adapter.weight": 1})

    with pytest.raises(RuntimeError, match="disk write failed"):
        daft_lsdsam._save_lora_checkpoint(
            FailingTorch,
            path,
            model=FakeModel(),
            config=config,
            epoch=1,
            score=None,
            source=materialized,
            target=materialized,
        )

    assert path.read_bytes() == b"old-checkpoint"
    assert not list(tmp_path.glob(".latest.pth.*.tmp"))
