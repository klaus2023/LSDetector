import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsd_app.training.tsft import (
    TsftLSDSAMConfig,
    TsftTrainingError,
    _cleanup_tsft_lora_checkpoints,
    _metrics_history_with_current,
    _rename_eval_metrics,
    _tsft_epoch_log_line,
    _tsft_metric_record,
    _tsft_metrics_summary,
    _tsft_training_progress_metrics,
    _update_tsft_online_confusion,
    build_tsft_training_payload,
    finalize_tsft_checkpoint,
    tsft_warmup_batch_lr,
    tsft_warmup_cosine_lr,
)
from lsd_app.training.daft_lsdsam import _EpochStats, _lsdsam_dataset_from_materialization, resolve_daft_dataset_materialization
from lsd_app.web import app


def test_tsft_training_payload_is_json_ready_and_auto_device() -> None:
    payload = build_tsft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        dataset_id="tsft-hunansat-test",
        max_epoch=10,
        max_lr=2e-4,
        min_lr=1e-6,
        max_decay_epoch=8,
        output_dir="outputs/tsft_training",
        output_weight_id="tsft-custom",
    )

    assert payload["device"] == "auto"
    assert payload["max_epoch"] == 10
    assert payload["max_lr"] == pytest.approx(2e-4)
    assert payload["min_lr"] == pytest.approx(1e-6)
    assert payload["max_decay_epoch"] == 8
    assert "halve_epochs" not in payload
    assert payload["dataset_id"] == "tsft-hunansat-test"
    assert payload["evaluation_dataset_id"] == ""
    assert payload["output_weight_id"] == "tsft-custom"
    assert payload["cleanup_lora_checkpoints_after_register"] is True
    assert payload["save_optimizer_state"] is False


def test_cleanup_tsft_lora_checkpoints_removes_each_file_once(tmp_path: Path) -> None:
    latest = tmp_path / "lsdsam_tsft_latest_lora.pth"
    best = tmp_path / "lsdsam_tsft_best_lora.pth"
    latest.write_bytes(b"latest")
    best.write_bytes(b"best")

    removed = _cleanup_tsft_lora_checkpoints(best, latest, best)

    assert removed == [str(best), str(latest)]
    assert not best.exists()
    assert not latest.exists()


def test_tsft_config_validation_rejects_negative_workers(tmp_path: Path) -> None:
    config = TsftLSDSAMConfig(
        registry_root=tmp_path / "outputs",
        dataset_id="tsft-v1",
        base_weight_id="weight-1",
        output_dir=tmp_path / "tsft",
        num_workers=-1,
    )

    with pytest.raises(TsftTrainingError, match="num_workers"):
        config.normalized()


def test_tsft_config_validation_rejects_invalid_lr_schedule(tmp_path: Path) -> None:
    base = {
        "registry_root": tmp_path / "outputs",
        "dataset_id": "tsft-v1",
        "base_weight_id": "weight-1",
        "output_dir": tmp_path / "tsft",
    }

    with pytest.raises(TsftTrainingError, match="max_lr"):
        TsftLSDSAMConfig(**base, max_lr=1e-6, min_lr=1e-6).normalized()
    with pytest.raises(TsftTrainingError, match="max_decay_epoch"):
        TsftLSDSAMConfig(**base, max_epoch=4, max_decay_epoch=5).normalized()


def test_finalize_tsft_checkpoint_accepts_canonical_lsdsam_model_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lsd_app.training import tsft as tsft_module

    def fake_require_module(name: str, _message: str):
        raise RuntimeError(f"dependency reached: {name}")

    monkeypatch.setattr(tsft_module, "_require_module", fake_require_module)

    with pytest.raises(RuntimeError, match="dependency reached: torch"):
        finalize_tsft_checkpoint(
            TsftLSDSAMConfig(
                registry_root=tmp_path / "outputs",
                dataset_id="tsft-v1",
                base_weight_id="weight-1",
                output_dir=tmp_path / "tsft",
                model_id="LSDSAM-H",
            )
        )


def test_tsft_warmup_cosine_schedule_matches_legacy_shape() -> None:
    min_lr = 1e-6
    max_lr = 2e-4

    assert tsft_warmup_batch_lr(batch_index=0, total_batches=4, max_lr=max_lr, min_lr=min_lr) == pytest.approx(min_lr)
    assert tsft_warmup_batch_lr(batch_index=3, total_batches=4, max_lr=max_lr, min_lr=min_lr) == pytest.approx(max_lr)
    assert tsft_warmup_cosine_lr(
        epoch_number=1, max_epoch=10, max_lr=max_lr, min_lr=min_lr, max_decay_epoch=8
    ) == pytest.approx(min_lr)
    assert tsft_warmup_cosine_lr(
        epoch_number=2, max_epoch=10, max_lr=max_lr, min_lr=min_lr, max_decay_epoch=8
    ) == pytest.approx(max_lr)
    middle_lr = tsft_warmup_cosine_lr(
        epoch_number=5, max_epoch=10, max_lr=max_lr, min_lr=min_lr, max_decay_epoch=8
    )
    assert min_lr < middle_lr < max_lr
    assert tsft_warmup_cosine_lr(
        epoch_number=8, max_epoch=10, max_lr=max_lr, min_lr=min_lr, max_decay_epoch=8
    ) == pytest.approx(min_lr)
    assert tsft_warmup_cosine_lr(
        epoch_number=10, max_epoch=10, max_lr=max_lr, min_lr=min_lr, max_decay_epoch=8
    ) == pytest.approx(min_lr)


def test_resolve_tsft_dataset_materialization_train_and_test_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "registry" / "dataset_store" / "TSFT" / "tsft-v1"
    train_image = root / "subsets" / "main" / "train" / "images" / "train_a.tif"
    train_label = root / "subsets" / "main" / "train" / "labels" / "train_a.tif"
    test_image = root / "subsets" / "main" / "test" / "images" / "test_a.tif"
    test_label = root / "subsets" / "main" / "test" / "labels" / "test_a.tif"
    for file_path in (train_image, train_label, test_image, test_label):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"placeholder")
    subset_manifest = root / "subsets" / "main" / "subset_manifest.json"
    subset_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "train-a",
                        "split": "train",
                        "image_path": str(train_image),
                        "label_path": str(train_label),
                    },
                    {
                        "sample_id": "test-a",
                        "split": "test",
                        "image_path": str(test_image),
                        "label_path": str(test_label),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "tsft-v1",
                "dataset_class": "TSFT",
                "version": "tsft-v1",
                "root_dir": str(root),
                "has_gt": True,
                "RGB_bands": "RGB_bands: 3,2,1",
                "subsets": [
                    {
                        "subset_id": "main",
                        "RGB_bands": "RGB_bands: 1,2,3",
                        "subset_manifest_path": str(subset_manifest),
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
                "dataset_id": "tsft-v1",
                "dataset_class": "TSFT",
                "repository_path": str(root),
                "manifest_path": str(manifest),
                "has_gt": True,
            }
        ],
    )

    materialized = resolve_daft_dataset_materialization(
        registry_root=tmp_path / "outputs",
        dataset_id="tsft-v1",
    )

    train_samples = materialized.supervised_train_samples()
    test_samples = materialized.target_eval_samples()
    assert materialized.dataset_class == "TSFT"
    assert [sample.sample_id for sample in train_samples] == ["train-a"]
    assert [sample.sample_id for sample in test_samples] == ["test-a"]
    assert train_samples[0].RGB_bands == "RGB_bands: 1,2,3"




def _fake_torch_with_concat_dataset():
    class FakeConcatDataset:
        def __init__(self, datasets):
            self.datasets = tuple(datasets)

        def __len__(self):
            return sum(len(dataset) for dataset in self.datasets)

    return SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(ConcatDataset=FakeConcatDataset)))

def _write_simple_tsft_manifest(root: Path, *, dataset_id: str, count: int) -> Path:
    samples = []
    for index in range(count):
        image = root / "subsets" / "main" / "train" / "images" / f"{dataset_id}_{index}.tif"
        label = root / "subsets" / "main" / "train" / "labels" / f"{dataset_id}_{index}.tif"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label.write_bytes(b"label")
        samples.append(
            {
                "sample_id": f"{dataset_id}-{index}",
                "split": "train",
                "image_path": str(image),
                "label_path": str(label),
            }
        )
    subset_manifest = root / "subsets" / "main" / "subset_manifest.json"
    subset_manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_class": "TSFT",
                "version": dataset_id,
                "root_dir": str(root),
                "has_gt": True,
                "RGB_bands": "RGB_bands: 3,2,1",
                "subsets": [{"subset_id": "main", "subset_manifest_path": str(subset_manifest)}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_resolve_tsft_concat_manifest_and_builds_concat_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch_with_concat_dataset()
    from lsd_app.training import daft_lsdsam

    first_root = tmp_path / "store" / "TSFT" / "first"
    second_root = tmp_path / "store" / "TSFT" / "second"
    first_manifest = _write_simple_tsft_manifest(first_root, dataset_id="tsft-a", count=2)
    second_manifest = _write_simple_tsft_manifest(second_root, dataset_id="tsft-b", count=4)
    concat_root = tmp_path / "outputs" / "registry" / "dataset_store" / "TSFT" / "concat"
    concat_manifest = concat_root / "manifest.json"
    concat_manifest.parent.mkdir(parents=True, exist_ok=True)
    concat_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "tsft-concat",
                "dataset_class": "TSFT",
                "version": "concat",
                "root_dir": str(concat_root),
                "is_concat": True,
                "concat_backend": "torch.utils.data.ConcatDataset",
                "concat_sources": [
                    {
                        "dataset_class": "TSFT",
                        "version": "tsft-a",
                        "manifest_path": str(first_manifest),
                        "root_dir": str(first_root),
                        "sample_count": 2,
                    },
                    {
                        "dataset_class": "TSFT",
                        "version": "tsft-b",
                        "manifest_path": str(second_manifest),
                        "root_dir": str(second_root),
                        "sample_count": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lsd_app.training.daft_lsdsam.list_datasets",
        lambda registry_root: [
            {
                "dataset_id": "tsft-concat",
                "dataset_class": "TSFT",
                "repository_path": str(concat_root),
                "manifest_path": str(concat_manifest),
                "has_gt": True,
            }
        ],
    )

    materialized = resolve_daft_dataset_materialization(registry_root=tmp_path / "outputs", dataset_id="tsft-concat")
    dataset = _lsdsam_dataset_from_materialization(
        torch,
        materialized,
        sample_selector="supervised_train_samples",
        dataset_cls=daft_lsdsam._LSDSAMSupervisedDataset,
        image_size=16,
        cv2=object(),
        np=object(),
    )

    assert materialized.dataset_class == "TSFT"
    assert len(materialized.supervised_train_samples()) == 6
    assert isinstance(dataset, torch.utils.data.ConcatDataset)
    assert len(dataset) == 6

def test_tsft_metric_helpers_preserve_train_and_test_curves() -> None:
    history = _metrics_history_with_current(
        [{"epoch": 1, "train": {"loss": 0.5}}],
        {"epoch": 1, "train": {"loss": 0.4, "mIoU": 0.7, "IoU": 0.6}},
    )
    history = _metrics_history_with_current(
        history,
        {"epoch": 2, "train": {"loss": 0.3}, "test": {"loss": 0.35, "mIoU": 0.65, "IoU": 0.55}},
    )

    assert history[0]["train"]["loss"] == 0.4
    assert history[1]["test"]["IoU"] == 0.55
    renamed = _rename_eval_metrics(
        {
            "test_loss": 0.2,
            "mIoU": 0.8,
            "IoU": [0.85, 0.75],
            "IoU_landslide": 0.75,
            "IoU_background": 0.85,
            "pixel_accuracy": 0.9,
            "precision": 0.7,
            "recall": 0.6,
            "F1": 0.646,
            "kappa": 0.52,
        }
    )
    assert renamed["loss"] == 0.2
    assert renamed["mIoU"] == 0.8
    assert renamed["IoU"] == [0.85, 0.75]
    assert renamed["IoU_foreground"] == 0.75
    assert renamed["IoU_background"] == 0.85
    assert renamed["pixel_accuracy"] == 0.9
    assert renamed["precision"] == 0.7
    assert renamed["kappa"] == 0.52


def test_tsft_no_eval_training_metrics_use_online_confusion() -> None:
    torch = pytest.importorskip("torch")
    confusion = torch.zeros((2, 2), dtype=torch.float64)
    stats = _EpochStats()
    stats.update(loss=0.4, dice=0.1, focal=0.3)
    outputs = torch.tensor(
        [
            [
                [[0.9, 0.8], [0.2, 0.1]],
                [[0.1, 0.2], [0.8, 0.9]],
            ]
        ]
    )
    targets = torch.tensor([[[0, 0], [1, 0]]])

    _update_tsft_online_confusion(confusion, outputs=outputs, targets=targets, torch=torch, device="cpu")
    metrics = _tsft_training_progress_metrics(epoch_stats=stats, train_confusion=confusion, torch=torch)

    assert metrics["loss"] == pytest.approx(0.4)
    assert metrics["mIoU"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert metrics["IoU_background"] == pytest.approx(2 / 3)
    assert metrics["IoU_foreground"] == pytest.approx(1 / 2)


def test_tsft_no_eval_branch_skips_train_re_evaluation() -> None:
    from lsd_app.training import tsft as tsft_module

    source = inspect.getsource(tsft_module.run_lsdsam_tsft)

    assert "eval_dataset=train_dataset" not in source
    assert '"evaluation_split": "train"' not in source
    assert "train_eval_batches = 0" in source
    assert "train_metrics = _tsft_training_progress_metrics" in source


def test_lsdformer_tsft_uses_online_train_metrics() -> None:
    from lsd_app.training import lsdformer as lsdformer_module

    source = inspect.getsource(lsdformer_module.run_lsdformer_tsft)

    assert "eval_dataset=train_dataset" not in source
    assert '"evaluation_split": "train"' not in source
    assert "train_confusion = torch.zeros" in source
    assert "_update_tsft_online_confusion(train_confusion" in source
    assert "train_metrics = _tsft_training_progress_metrics" in source


def test_tsft_progress_total_includes_eval_batches_per_epoch() -> None:
    from lsd_app.training import lsdformer as lsdformer_module
    from lsd_app.training import tsft as tsft_module

    for run_fn in (tsft_module.run_lsdsam_tsft, lsdformer_module.run_lsdformer_tsft):
        source = inspect.getsource(run_fn)
        assert "eval_batches_per_epoch = _ceil_batches(len(test_dataset), cfg.batch_size)" in source
        assert "epoch_work_total = total_batches + eval_batches_per_epoch" in source
        assert "global_total = max(1, cfg.max_epoch * epoch_work_total)" in source
        assert "processed = start_epoch_index * epoch_work_total" in source
        assert "eval_total = global_total + test_eval_batches" not in source


def test_tsft_metric_record_writes_flat_and_nested_fields() -> None:
    record = _tsft_metric_record(
        epoch=1,
        lr=2e-4,
        train={"loss": 0.4, "mIoU": 0.7, "IoU": [0.8, 0.6], "IoU_foreground": 0.6, "kappa": 0.44},
        test={"loss": 0.5, "mIoU": 0.65, "IoU": [0.75, 0.55], "IoU_foreground": 0.55, "kappa": 0.33},
    )

    assert record["epoch"] == 1
    assert record["lr"] == 2e-4
    assert record["train.loss"] == 0.4
    assert record["train.mIoU"] == 0.7
    assert record["train.IoU"] == [0.8, 0.6]
    assert record["train.IoU_foreground"] == 0.6
    assert record["train.kappa"] == 0.44
    assert record["test.loss"] == 0.5
    assert record["test.mIoU"] == 0.65
    assert record["test.IoU"] == [0.75, 0.55]
    assert record["test.IoU_foreground"] == 0.55
    assert record["test.kappa"] == 0.33
    assert record["train"]["IoU"] == [0.8, 0.6]
    assert record["test"]["IoU"] == [0.75, 0.55]

def test_tsft_epoch_log_uses_foreground_iou_from_legacy_array() -> None:
    line = _tsft_epoch_log_line(
        epoch_number=1,
        total_epochs=10,
        train={"loss": 0.1, "mIoU": 0.2, "IoU": [0.3, 0.4]},
        test={"loss": 0.2, "mIoU": 0.3, "IoU": [0.4, 0.5]},
        best_epoch=1,
        best_score=0.5,
    )

    assert "train_IoU=0.4000" in line
    assert "eval_IoU=0.5000" in line


def test_tsft_metrics_summary_records_selection_and_counts() -> None:
    assert _tsft_metrics_summary(best_score=0.61234, best_epoch=3, has_test=False, train_count=4, test_count=0) == (
        "best_train_mIoU=0.6123;best_epoch=3;train_samples=4;eval_samples=0"
    )
    assert _tsft_metrics_summary(best_score=0.81234, best_epoch=2, has_test=True, train_count=4, test_count=2) == (
        "best_eval_mIoU=0.8123;best_epoch=2;train_samples=4;eval_samples=2"
    )


def test_tsft_training_job_id_and_metric_rows() -> None:
    payload = build_tsft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        dataset_id="tsft-v1",
        max_epoch=1,
        max_lr=2e-4,
        min_lr=1e-6,
        max_decay_epoch=1,
        output_dir="outputs/tsft_training",
    )

    rows = app._tsft_metric_history_rows(
        {
            "metrics_history": [
                {"epoch": 2, "train": {"loss": 0.3, "mIoU": 0.6, "IoU": [0.7, 0.5], "IoU_foreground": 0.5}},
                {"epoch": 1, "train": {"loss": 0.4, "mIoU": 0.5, "IoU": [0.6, 0.4], "IoU_foreground": 0.4}, "eval": {"loss": 0.45, "mIoU": 0.48, "IoU": [0.58, 0.38], "IoU_foreground": 0.38}},
            ]
        }
    )

    assert app.tsft_training_job_id(payload).startswith("tsft-training-tsft-v1-")
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["train_loss"] == 0.4
    assert rows[0]["train_IoU"] == 0.4
    assert rows[0]["eval_mIoU"] == 0.48
    assert rows[0]["eval_IoU"] == 0.38


def test_tsft_training_panel_uses_warmup_cosine_fields() -> None:
    source = inspect.getsource(app._render_tsft_training_panel)

    assert "max epoch" in source
    assert "max lr" in source
    assert "min lr" in source
    assert "max decay epoch" in source
    assert "halve epochs" not in source


def test_tsft_training_progress_renderer_uses_train_and_test_metrics() -> None:
    calls: list[tuple[str, str]] = []

    class FakeStreamlit:
        def markdown(self, body, **_kwargs):
            calls.append(("markdown", body))

        def error(self, body):
            calls.append(("error", body))

    app._render_tsft_training_progress(
        FakeStreamlit(),
        {
            "description": "TSFT LSDSAM tsft-v1",
            "status": "running",
            "phase": "evaluation",
            "percent": 0.5,
            "current_epoch": 1,
            "total_epochs": 10,
            "current_batch": 5,
            "total_batches": 20,
            "lr": 0.0002,
            "train": {"loss": 0.4, "mIoU": 0.7, "IoU": [0.8, 0.6], "IoU_foreground": 0.6},
            "eval": {"loss": 0.5, "mIoU": 0.65, "IoU": [0.75, 0.55], "IoU_foreground": 0.55},
            "best_score": 0.65,
            "best_epoch": 1,
            "gpu_memory_reserved_mb": 768,
            "gpu_memory_peak_reserved_mb": 1536,
            "selection_metric": "eval_mIoU",
            "latest_log": "train_loss=0.4000 | eval_mIoU=0.6500",
        },
    )

    html = "\n".join(body for kind, body in calls if kind == "markdown")
    assert "daft-train-panel" in html
    assert "train mIoU" in html
    assert "eval IoU" in html
    assert "train_loss=0.4000" in html
    assert "2.00e-04" in html
    assert "GPU peak" in html
    assert "1.50 GB" in html



def test_tsft_dispatches_lsdformer_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from lsd_app.training import tsft as tsft_module

    fake_module = types.ModuleType("lsd_app.training.lsdformer")

    def fake_run(config, progress_callback=None):
        return {"model_id": config.model_id, "progress_callback": progress_callback}

    fake_module.run_lsdformer_tsft = fake_run
    monkeypatch.setitem(sys.modules, "lsd_app.training.lsdformer", fake_module)

    result = tsft_module.run_lsdsam_tsft(
        TsftLSDSAMConfig(
            registry_root=tmp_path / "outputs",
            dataset_id="dataset",
            base_weight_id="weight",
            output_dir=tmp_path / "tsft",
            model_id="lsdformer-fast",
        ),
        progress_callback=lambda payload: None,
    )

    assert result["model_id"] == "lsdformer-fast"
    assert callable(result["progress_callback"])
