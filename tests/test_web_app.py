import inspect
import json
from pathlib import Path

import pytest

from lsd_app.config import PlatformConfig
from lsd_app.core.mosaic import MosaicError
from lsd_app.web import app
from lsd_app.training.tsft import build_tsft_training_payload


def test_import_does_not_require_streamlit() -> None:
    assert app.STAGES == ("prepare", "infer", "mosaic", "vectorize", "evaluate", "visualize")
    assert app.DEPLOYMENT_STAGES == ("prepare", "infer", "mosaic", "vectorize")
    assert app.SECTION_OPTIONS[5] == "Inference"
    assert app.MODEL_CHOICES == ("LSDFormer", "LSDSAM-H", "LSDSAM-L", "LSDSAM-B")
    assert app.WEIGHT_STAGES == ("TAFT", "DAFT", "TSFT", "inference-ready")


def test_model_repository_is_display_only_for_weights() -> None:
    source = inspect.getsource(app.main)

    model_repo_branch = source.split('if selected_section == "Model Repository":', 1)[1].split(
        'elif selected_section == "Weight Repository":',
        1,
    )[0]
    assert "registry_root=registry_root" in model_repo_branch
    assert "_render_model_weight_manager" not in model_repo_branch


def test_build_raw_config_maps_fields_and_truth_path(tmp_path: Path) -> None:
    raw = app.build_raw_config(
        geotiff_path=str(tmp_path / "source.tif"),
        output_dir=str(tmp_path / "out"),
        model="mock",
        checkpoint_path="",
        tile_size=64,
        stride=32,
        batch_size=2,
        threshold=0.7,
        min_area_pixels=9,
        device="auto",
        truth_path=str(tmp_path / "truth.tif"),
        RGB_bands="RGB_bands: 1,2,3",
    )

    assert raw["input"] == {"raster_path": str(tmp_path / "source.tif"), "RGB_bands": "RGB_bands: 1,2,3"}
    assert raw["output"] == {"dir": str(tmp_path / "out"), "vector_formats": ["gpkg", "geojson", "shapefile"]}
    assert raw["model"] == {"name": "mock", "checkpoint_path": None}
    assert raw["tiling"] == {"tile_size": 64, "stride": 32}
    assert raw["inference"]["batch_size"] == 2
    assert raw["inference"]["threshold"] == 0.7
    assert raw["postprocess"]["min_area_pixels"] == 9
    assert raw["vectorize"] == {"mode": "streaming"}
    assert raw["runtime"]["device"] == "auto"
    assert raw["truth"] == {"vector_path": None, "mask_path": str(tmp_path / "truth.tif")}

    vector_raw = app.build_raw_config(
        geotiff_path="source.tif",
        output_dir="out",
        model="mock",
        truth_path="truth.geojson",
    )
    assert vector_raw["truth"] == {"vector_path": "truth.geojson", "mask_path": None}
    assert vector_raw["inference"]["overwrite"] is False


def test_build_evaluation_config_uses_dataset_gt_and_weight_fields(tmp_path: Path) -> None:
    dataset = {
        "imagery_path": str(tmp_path / "source.tif"),
        "label_path": str(tmp_path / "truth.gpkg"),
        "RGB_bands": "RGB_bands: 3,2,1",
    }

    raw = app.build_evaluation_config(
        dataset=dataset,
        output_dir=str(tmp_path / "eval"),
        model="mock",
        checkpoint_path=str(tmp_path / "weights.pth"),
    )

    assert raw["input"] == {"raster_path": str(tmp_path / "source.tif"), "RGB_bands": "RGB_bands: 3,2,1"}
    assert raw["truth"] == {"vector_path": str(tmp_path / "truth.gpkg"), "mask_path": None}
    assert raw["model"] == {"name": "mock", "checkpoint_path": str(tmp_path / "weights.pth")}
    assert raw["tiling"] == {"tile_size": 512, "stride": 256}


def test_build_evaluation_dataset_payload_is_sample_based() -> None:
    payload = app.build_evaluation_dataset_payload(
        dataset_id="evaluation-v1",
        weight_id="weight-1",
        model_id="lsdsam-fewshot",
        output_dir="outputs/eval",
        checkpoint_path="/tmp/weight.pth",
        batch_size=4,
    )

    assert payload["dataset_id"] == "evaluation-v1"
    assert payload["weight_id"] == "weight-1"
    assert payload["model_id"] == "lsdsam-fewshot"
    assert payload["batch_size"] == 4
    assert "tile_size" not in payload
    assert "stride" not in payload


def test_evaluation_panel_does_not_show_sliding_window_controls() -> None:
    source = inspect.getsource(app)

    assert "evaluation batch size" in source
    assert "Advanced inference parameters" not in source
    assert "inference tile size" not in source
    assert "inference stride" not in source
    assert "evaluation tile size" not in source
    assert "evaluation stride" not in source


def test_evaluation_history_uses_autorefresh_fragment() -> None:
    source = inspect.getsource(app)

    assert "_render_evaluation_history_panel_auto" in source
    assert "@fragment(run_every=1.5)" in inspect.getsource(app._render_evaluation_history_panel_auto)


def test_completed_evaluation_monitor_does_not_render_artifacts_table() -> None:
    source = inspect.getsource(app._render_evaluation_job_monitor)

    completed_branch = source.split('if status == "completed":', 1)[1].split('elif status == "failed":', 1)[0]
    assert "_render_artifacts" not in completed_branch
    assert "_render_statuses" not in completed_branch
    assert "Evaluation completed" in completed_branch


def test_evaluation_background_task_uses_dataset_payload_not_deployment_pipeline() -> None:
    source = inspect.getsource(app.run_evaluation_background_task)

    assert "run_evaluation_dataset_task" in source
    assert "run_all_resumable_with_progress" not in source
    assert "run_stage" not in source


def test_evaluation_stage_rows_use_sample_percent_over_stale_overall() -> None:
    rows = app._evaluation_stage_rows_from_job(
        {
            "payload": {"dataset_id": "evaluation-v1"},
            "status": "running",
            "phase": "evaluate",
            "percent": 0.98,
            "overall_percent": 0.02,
            "message": "Evaluating evaluation 997/997: loss=0.1154",
            "processed": 997,
            "total": 997,
        }
    )

    assert rows == [
        {
            "stage": "evaluate dataset",
            "status": "running",
            "percent": "98.00%",
            "message": "Evaluating evaluation 997/997: loss=0.1154",
            "processed": 997,
            "total": 997,
            "phase": "evaluate",
        }
    ]


def test_evaluation_history_helpers_roundtrip_search_export_delete(tmp_path: Path) -> None:
    payload = {
        "dataset_id": "evaluation-v1",
        "weight_id": "HNSAT_DAFT2",
        "model_id": "lsdsam-fewshot",
        "output_dir": str(tmp_path / "eval"),
    }
    result = {
        "scores": {"miou": 0.746, "iou": 0.5337, "precision": 0.8, "recall": 0.7, "f1": 0.74, "kappa": 0.5},
        "metrics": {"loss": 0.1154, "sample_count": 997},
        "evaluation_json_path": str(tmp_path / "eval" / "metrics" / "evaluation.json"),
        "evaluation_csv_path": str(tmp_path / "eval" / "metrics" / "evaluation.csv"),
    }

    app._record_evaluation_history(registry_root=str(tmp_path), job_id="evaluation-job-1", payload=payload, result=result)

    rows = app._read_evaluation_history(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["model_id"] == "lsdsam-fewshot"
    assert rows[0]["weight_version"] == "HNSAT_DAFT2"
    assert rows[0]["dataset_id"] == "evaluation-v1"
    assert rows[0]["mIoU"] == pytest.approx(0.746)
    assert app._filter_evaluation_history(rows, "hnsat")[0]["history_id"] == "evaluation-job-1"
    assert app._filter_evaluation_history(rows, "missing") == []
    csv_text = app._evaluation_history_csv(rows)
    assert "weight_version" in csv_text
    assert "HNSAT_DAFT2" in csv_text

    assert app._delete_evaluation_history_entry(str(tmp_path), "evaluation-job-1") is True
    assert app._read_evaluation_history(str(tmp_path)) == []


def test_build_training_dry_run_is_reserved() -> None:
    plan = app.build_training_dry_run(
        stage="TSFT",
        model_id="lsdsam",
        weight_id="weight-1",
        dataset_id="tsft-few-shot",
        output_dir="outputs/tsft",
    )

    assert plan["stage"] == "TSFT"
    assert plan["status"] == "reserved-dry-run-no-training-launched"


def test_training_weight_rows_follow_selected_model() -> None:
    rows = [
        {"weight_id": "LSDSAM_H_LS77K", "model_id": "LSDSAM-H", "exists": True},
        {"weight_id": "LSDFormer_LS77K", "model_id": "LSDFormer", "exists": True},
        {"weight_id": "missing_lsdsam", "model_id": "LSDSAM-H", "exists": False},
        {"weight_id": "legacy_lsdsam_b", "model_id": "lsdsam-fewshot", "exists": True},
    ]

    lsdsam_h = app._training_weight_rows_for_model(rows, "LSDSAM-H")
    lsdformer = app._training_weight_rows_for_model(rows, "LSDFormer")
    lsdsam_b = app._training_weight_rows_for_model(rows, "LSDSAM-B")

    assert app._weight_id_options(lsdsam_h) == ["LSDSAM_H_LS77K"]
    assert app._weight_id_options(lsdformer) == ["LSDFormer_LS77K"]
    assert app._weight_id_options(lsdsam_b) == ["legacy_lsdsam_b"]
    assert app._training_model_widget_key("LSDSAM-H") == "LSDSAM_H"


def test_training_dataset_rows_are_class_specific() -> None:
    rows = [
        {"dataset_id": "taft-a", "dataset_class": "TAFT", "has_gt": True},
        {"dataset_id": "taft-empty", "dataset_class": "TAFT", "has_gt": False},
        {"dataset_id": "daft-a", "dataset_class": "DAFT", "has_gt": False},
        {"dataset_id": "tsft-a", "dataset_class": "TSFT", "has_gt": True},
        {"dataset_id": "eval-a", "dataset_class": "Evaluation", "has_gt": True},
    ]

    daft_source = app._training_dataset_rows_for_class(rows, "TAFT", require_gt=True)
    daft_target = app._training_dataset_rows_for_class(rows, "DAFT")
    tsft = app._training_dataset_rows_for_class(rows, "TSFT", require_gt=True)

    assert [row["dataset_id"] for row in daft_source] == ["taft-a"]
    assert [row["dataset_id"] for row in daft_target] == ["daft-a"]
    assert [row["dataset_id"] for row in tsft] == ["tsft-a"]


def test_daft_training_job_payload_uses_auto_device() -> None:
    payload = app.build_daft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        source_dataset_id="taft-v1",
        target_dataset_id="daft-v1",
        alignment_mode="IST",
        total_epochs=10,
        initial_lr=2e-4,
        halve_epochs="6,8",
        device="auto",
        output_dir="outputs/daft",
        output_weight_id="daft-rlzx-uav-custom",
    )

    assert payload["device"] == "auto"
    assert payload["halve_epochs"] == [6, 8]
    assert payload["output_weight_id"] == "daft-rlzx-uav-custom"
    assert app.daft_training_job_id(payload).startswith("daft-training-daft-v1-")


def test_daft_loss_history_rows_are_epoch_sorted() -> None:
    rows = app._daft_loss_history_rows(
        {
            "loss_history": [
                {"epoch": 2, "total": 0.4, "seg": 0.2, "adv": 0.1, "align": 0.01, "dis": 0.3},
                {"epoch": 1, "total": 0.5, "seg": 0.3, "adv": 0.2, "align": 0.02, "dis": 0.4},
            ]
        }
    )

    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["total"] == 0.5


def test_weight_repository_table_is_compact() -> None:
    rows = app._weight_repository_table(
        [
            {
                "weight_id": "tsft-main",
                "model_id": "lsdsam-fewshot",
                "stage": "TSFT",
                "file_size_bytes": 1536,
                "parameter_count": 1_250_000,
                "path": "/store/weight.pth",
                "created_at": "2026-05-07T17:00:00Z",
                "adapter_profile": "hidden",
                "lora_task_r": 32,
            }
        ]
    )

    assert rows == [
        {
            "weight_id": "tsft-main",
            "stage": "TSFT",
            "registered_model": "lsdsam-fewshot",
            "weight_size": "1.50 KB",
            "parameter_count": "1.25M",
            "registered_at": "2026-05-07T17:00:00Z",
            "stored_path": "/store/weight.pth",
        }
    ]


def test_taft_import_progress_panel_renders_visible_status() -> None:
    html = app._taft_import_progress_html(
        {
            "status": "running",
            "processed": 7,
            "total": 10,
            "written": 5,
            "skipped": 1,
            "rewritten": 1,
            "percent": 0.7,
            "message": "GDCLD/train/sample_007",
        }
    )

    assert "taft-import-panel" in html
    assert "70.00%" in html
    assert "processed 7/10" in html
    assert "GDCLD/train/sample_007" in html

    skipped_html = app._taft_import_progress_html(
        {
            "status": "running",
            "processed": 1,
            "total": 10,
            "skipped": 1,
            "percent": 0.1,
            "message": "GDCLD/train/sample_000",
            "current_action": "skipped",
            "current_reason": "standardized copy already exists and passed integrity validation",
        }
    )
    assert "skipped: standardized copy already exists and passed integrity validation" in skipped_html


def test_taft_annotation_helpers_build_confirmed_overrides() -> None:
    discovery = {
        "subsets": [
            {
                "parent_group": "CAS",
                "source_rel": "CAS/palu",
                "train_count": 2,
                "test_count": 1,
                "RGB_bands": "RGB_bands: 1,2,3",
                "label_polarity": "foreground_black",
                "auto_RGB_bands": "RGB_bands: 1,2,3",
                "auto_label_polarity": "foreground_black",
            }
        ]
    }

    rows = app._taft_tree_annotation_rows(discovery)
    assert rows == [
        {
            "parent_group": "CAS",
            "subset": "CAS/palu",
            "train_count": 2,
            "test_count": 1,
            "RGB_bands": "RGB_bands: 1,2,3",
            "label_polarity": "foreground_black",
            "confirmed": False,
        }
    ]
    assert app._taft_annotations_complete(rows) is False
    rows[0]["confirmed"] = True
    rows[0]["RGB_bands"] = "RGB_bands: 3,2,1"
    assert app._taft_annotations_complete(rows) is True
    assert app._taft_subset_overrides_from_rows(rows) == {
        "CAS/palu": {
            "RGB_bands": "RGB_bands: 3,2,1",
            "label_polarity": "foreground_black",
            "confirmed": True,
        }
    }
    cleared = app._taft_set_rows_confirmed(rows, False)
    assert cleared[0]["confirmed"] is False
    confirmed = app._taft_set_rows_confirmed(cleared, True)
    assert confirmed[0]["confirmed"] is True
    assert confirmed[0]["RGB_bands"] == "RGB_bands: 3,2,1"


def test_registered_taft_versions_lists_existing_taft_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list_datasets(*, registry_root):
        return [
            {"dataset_class": "TAFT", "version": "landslide70k-v1"},
            {"dataset_class": "DAFT", "version": "target-v1"},
            {"dataset_class": "TAFT", "version": "custom-v2"},
        ]

    monkeypatch.setattr(app, "list_datasets", fake_list_datasets)

    assert app._registered_taft_versions("outputs") == ["custom-v2", "landslide70k-v1"]


def test_tsft_sample_folder_paths_resolve_images_and_labels() -> None:
    assert app._tsft_sample_folder_paths("/data/tsft/train", split="train", required=True) == (
        "/data/tsft/train/images",
        "/data/tsft/train/labels",
    )
    assert app._tsft_sample_folder_paths("/data/tsft/test/", split="test", required=False) == (
        "/data/tsft/test/images",
        "/data/tsft/test/labels",
    )
    assert app._tsft_sample_folder_paths("", split="test", required=False) == ("", "")
    with pytest.raises(app.RegistryError, match="train samples path is required"):
        app._tsft_sample_folder_paths("", split="train", required=True)


def test_set_rgb_bands_preset_state_syncs_tsft_form() -> None:
    class FakeStreamlit:
        session_state: dict[str, object] = {}

    fake = FakeStreamlit()
    app._set_rgb_bands_preset_state(fake, "tsft_RGB_bands", "RGB_bands: 6,4,2")
    assert fake.session_state["tsft_RGB_bands_preset"] == "RGB_bands: 6,4,2"
    assert "tsft_RGB_bands" not in fake.session_state

    app._set_rgb_bands_preset_state(fake, "tsft_RGB_bands", "RGB_bands: 8,5,3")
    assert fake.session_state["tsft_RGB_bands_preset"] == app.RGB_BANDS_CUSTOM_OPTION
    assert fake.session_state["tsft_RGB_bands"] == "RGB_bands: 8,5,3"


def test_background_job_helpers_persist_progress(tmp_path: Path) -> None:
    from lsd_app.web.jobs import read_job_state, stable_job_id, start_background_job

    payload = {"version": "daft-v1", "imagery_path": "images"}
    job_id = stable_job_id(kind="dataset-upload-daft", version="daft-v1", payload=payload)

    def target(progress):
        progress({"status": "running", "processed": 1, "total": 2, "percent": 0.5, "message": "half"})
        return {"ok": True}

    job = start_background_job(
        registry_root=tmp_path / "outputs",
        job_id=job_id,
        kind="dataset-upload",
        description="DAFT upload daft-v1",
        payload=payload,
        target=target,
    )

    import time

    deadline = time.time() + 5
    state = read_job_state(tmp_path / "outputs", job.job_id)
    while state.get("status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = read_job_state(tmp_path / "outputs", job.job_id)

    assert state["status"] == "completed"
    assert state["result"] == {"ok": True}
    assert state["payload"] == payload


def test_subprocess_python_env_prefers_active_prefix_lib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lsd_app.web import jobs

    prefix = tmp_path / "env"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib64").mkdir()
    monkeypatch.setattr(jobs.sys, "prefix", str(prefix))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib:/opt/lib")

    env = jobs._subprocess_python_env()

    ld_paths = env["LD_LIBRARY_PATH"].split(":")
    assert ld_paths[:2] == [str(prefix / "lib"), str(prefix / "lib64")]
    assert "/usr/lib" in ld_paths
    assert env["CONDA_PREFIX"] == str(prefix)
    assert env["PYTHONUNBUFFERED"] == "1"


def test_job_json_ready_uses_captured_builtins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lsd_app.web import jobs

    monkeypatch.setattr(jobs, "isinstance", "shadowed", raising=False)
    monkeypatch.setattr(jobs, "str", "shadowed", raising=False)
    monkeypatch.setattr(jobs, "list", "shadowed", raising=False)
    monkeypatch.setattr(jobs, "tuple", "shadowed", raising=False)

    source = tmp_path / "source.tif"
    payload = jobs._json_ready(
        {source: [tmp_path / "mask.tif", (tmp_path / "vector.gpkg",)], "nested": {"ok": True}}
    )

    assert payload == {
        str(source): [str(tmp_path / "mask.tif"), [str(tmp_path / "vector.gpkg")]],
        "nested": {"ok": True},
    }


def test_daft_training_progress_renderer_uses_training_metrics() -> None:
    calls: list[tuple[str, str]] = []

    class FakeStreamlit:
        def markdown(self, body, **_kwargs):
            calls.append(("markdown", body))

        def error(self, body):
            calls.append(("error", body))

    app._render_daft_training_progress(
        FakeStreamlit(),
        {
            "description": "DAFT LSDSAM daft-rlzx-uav-1",
            "status": "running",
            "phase": "train",
            "percent": 0.25,
            "current_epoch": 1,
            "total_epochs": 10,
            "current_batch": 20,
            "total_batches": 290,
            "losses": {"total": 0.4567, "seg": 0.1234, "adv": 0.1111, "align": 0.00023, "dis": 0.2222},
            "lr_seg": 0.0002,
            "lr_dis": 0.0005,
            "gpu_memory_reserved_mb": 1536,
            "gpu_memory_peak_reserved_mb": 2048,
            "latest_log": "[Epoch 1/10] Batch 20/290 | Total=0.4567",
        },
    )

    html = "\n".join(body for kind, body in calls if kind == "markdown")
    assert "daft-train-panel" in html
    assert "epoch 1/10" in html
    assert "Total=0.4567" in html
    assert "2.00e-04" in html
    assert "GPU peak" in html
    assert "2.00 GB" in html
    assert "written" not in html


def test_daft_training_progress_renderer_handles_evaluation_phase() -> None:
    calls: list[tuple[str, str]] = []

    class FakeStreamlit:
        def markdown(self, body, **_kwargs):
            calls.append(("markdown", body))

        def error(self, body):
            calls.append(("error", body))

        def expander(self, *_args, **_kwargs):
            class _Context:
                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    return False

            return _Context()

        def code(self, body, language=None):
            calls.append(("code", body))

    app._render_daft_training_progress(
        FakeStreamlit(),
        {
            "description": "DAFT LSDSAM eval",
            "status": "failed",
            "phase": "evaluation",
            "percent": 0.5,
            "current_epoch": 1,
            "total_epochs": 2,
            "evaluation_split": "eval",
            "eval_current_batch": 3,
            "eval_total_batches": 5,
            "device_name": "cuda:0 (GPU)",
            "losses": {"total": 0.4, "seg": 0.2, "adv": 0.1, "align": 0.01, "dis": 0.3},
            "evaluation": {"mIoU": 0.7},
            "error_type": "AcceleratorError",
            "error": "CUDA error: unknown error",
            "traceback": "Traceback line",
        },
    )

    html = "\n".join(body for kind, body in calls if kind == "markdown")
    assert "eval eval 3/5" in html
    assert "cuda:0 (GPU)" in html
    assert any(kind == "code" and "Traceback line" in body for kind, body in calls)


def test_background_job_listing_ignores_macos_metadata_files(tmp_path: Path) -> None:
    from lsd_app.web.jobs import job_root, list_job_states

    root = job_root(tmp_path / "outputs")
    (root / "dataset-upload-v1.json").write_text(
        json.dumps({"job_id": "dataset-upload-v1", "kind": "dataset-upload", "status": "completed"}),
        encoding="utf-8",
    )
    (root / "._dataset-upload-v1.json").write_bytes(b"not json")

    jobs = list_job_states(tmp_path / "outputs", kind="dataset-upload")

    assert [job["job_id"] for job in jobs] == ["dataset-upload-v1"]


def test_background_job_listing_marks_stale_running_jobs(tmp_path: Path) -> None:
    from lsd_app.web.jobs import list_job_states, read_job_state, state_path_for_job

    job_id = "deployment-v1"
    state_path = state_path_for_job(tmp_path / "outputs", job_id)
    state_path.write_text(
        json.dumps({"job_id": job_id, "kind": "deployment", "status": "running", "message": "old"}),
        encoding="utf-8",
    )

    state = read_job_state(tmp_path / "outputs", job_id)
    jobs = list_job_states(tmp_path / "outputs", kind="deployment")

    assert state["status"] == "interrupted"
    assert [job["status"] for job in jobs] == ["interrupted"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "interrupted"


def test_background_job_atomic_writes_support_concurrent_updates(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from lsd_app.web.jobs import _write_json_atomic

    path = tmp_path / "outputs" / "registry" / "jobs" / "deployment-test.json"

    def write(index: int) -> None:
        _write_json_atomic(path, {"index": index, "status": "running"})

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(write, range(30)))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert "index" in data


def test_background_job_cancel_request_is_persisted(tmp_path: Path) -> None:
    from lsd_app.web.jobs import read_job_state, request_job_cancel, stable_job_id, start_background_job
    import time

    payload = {"version": "deploy-v1"}
    job_id = stable_job_id(kind="deployment", version="deploy-v1", payload=payload)

    def target(progress):
        progress({"status": "running", "percent": 0.1, "message": "started"})
        deadline = time.time() + 5
        while time.time() < deadline:
            time.sleep(0.02)
            progress({"status": "running", "percent": 0.2, "message": "working"})
        return {"ok": True}

    start_background_job(
        registry_root=tmp_path / "outputs",
        job_id=job_id,
        kind="deployment",
        description="Deployment deploy-v1",
        payload=payload,
        target=target,
    )

    deadline = time.time() + 5
    state = read_job_state(tmp_path / "outputs", job_id)
    while state.get("status") != "running" and time.time() < deadline:
        time.sleep(0.02)
        state = read_job_state(tmp_path / "outputs", job_id)

    assert request_job_cancel(tmp_path / "outputs", job_id)

    deadline = time.time() + 5
    state = read_job_state(tmp_path / "outputs", job_id)
    while state.get("status") != "cancelled" and time.time() < deadline:
        time.sleep(0.02)
        state = read_job_state(tmp_path / "outputs", job_id)

    assert state["status"] == "cancelled"
    assert state["cancel_requested"] is True


def test_dataset_store_filter_helpers_are_hierarchical() -> None:
    browser = {
        "classes": {
            "TAFT": {
                "versions": [
                    {
                        "version": "landslide70k-v1",
                        "dataset_id": "taft-landslide70k",
                        "sample_count": 3,
                        "train_count": 2,
                        "test_count": 1,
                        "subset_count": 2,
                        "has_gt": True,
                        "subsets": [
                            {
                                "parent_group": "CAS",
                                "subset": "CAS/palu",
                                "subset_id": "CAS__palu",
                                "sample_count": 2,
                                "train_count": 1,
                                "test_count": 1,
                                "RGB_bands": "RGB_bands: 1,2,3",
                                "label_polarity": "foreground_black",
                                "has_auxiliary": False,
                            },
                            {
                                "parent_group": "GDCLD",
                                "subset": "GDCLD",
                                "subset_id": "GDCLD",
                                "sample_count": 1,
                                "train_count": 1,
                                "test_count": 0,
                                "RGB_bands": "RGB_bands: 1,2,3",
                                "label_polarity": "foreground_white",
                                "has_auxiliary": False,
                            },
                        ],
                    }
                ]
            },
            "DAFT": {"versions": [{"version": "target-v1", "dataset_id": "daft-target", "subsets": []}]},
            "TSFT": {
                "versions": [
                    {
                        "version": "wp_concat",
                        "dataset_id": "tsft-wp-concat",
                        "sample_count": 3,
                        "train_count": 3,
                        "test_count": 0,
                        "is_concat": True,
                        "concat_source_count": 2,
                        "subsets": [{"subset": "wp_pos_1"}, {"subset": "wp_neg_1"}],
                    }
                ]
            },
        }
    }

    rows = app._flatten_dataset_store_rows(browser)
    assert len(rows) == 4
    taft_rows = [row for row in rows if row["dataset_class"] == "TAFT"]
    assert app._unique_values(taft_rows, "version") == ["landslide70k-v1"]
    assert app._unique_values(taft_rows, "subset_id") == ["CAS__palu", "GDCLD"]
    concat_rows = [row for row in rows if row["dataset_class"] == "TSFT"]
    assert len(concat_rows) == 1
    assert concat_rows[0]["subset"] == "2 sources"
    assert concat_rows[0]["upload_method"] == "virtual_concat"
    table = app._dataset_store_browse_table(taft_rows)
    assert table[0]["subset_manifest_path"] == ""


def test_normalize_web_config_uses_platform_validation(tmp_path: Path) -> None:
    raw = app.build_raw_config(
        geotiff_path=str(tmp_path / "source.tif"),
        output_dir=str(tmp_path / "out"),
        model="mock",
        tile_size=128,
        stride=64,
        batch_size=3,
        threshold=0.25,
        min_area_pixels=4,
        device="mps",
    )

    config = app.normalize_config(raw, create_output_dirs=False)

    assert isinstance(config, PlatformConfig)
    assert config.input_raster_path == tmp_path / "source.tif"
    assert config.output_dir == tmp_path / "out"
    assert config.model_name == "mock"
    assert config.tile_size == 128
    assert config.stride == 64
    assert config.batch_size == 3
    assert config.threshold == 0.25
    assert config.min_area_pixels == 4
    assert config.vectorize_mode == "streaming"
    assert config.device == "mps"


def test_preflight_deployment_reports_missing_inputs(tmp_path: Path) -> None:
    raw = app.build_raw_config(
        geotiff_path=str(tmp_path / "missing.tif"),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
        checkpoint_path=str(tmp_path / "missing.pth"),
    )

    config, result = app.preflight_deployment_config(raw, create_output_dirs=False)

    assert config is not None
    assert not result.ok
    assert any("输入 GeoTIFF 不存在" in item for item in result.errors)
    assert any("权重文件不存在" in item for item in result.errors)


def test_preflight_deployment_rejects_model_weight_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    weight = tmp_path / "weights.pth"
    source.write_bytes(b"fake")
    weight.write_bytes(b"fake")
    raw = app.build_raw_config(
        geotiff_path=str(source),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
        checkpoint_path=str(weight),
    )

    _, result = app.preflight_deployment_config(
        raw,
        selected_weight={"weight_id": "w-lsdsam", "model_id": "lsdsam-fewshot", "exists": True},
        create_output_dirs=False,
    )

    assert not result.ok
    assert any("属于模型 lsdsam-fewshot" in item for item in result.errors)



def test_inference_section_does_not_expose_manual_checkpoint_override() -> None:
    source = inspect.getsource(app.main)

    assert "checkpoint path override" not in source
    assert "checkpoint_override" not in source
    assert "manual checkpoint path" not in source
    assert "checkpoint_path = checkpoint_default" in source

def test_default_RGB_bands_from_weight_metadata() -> None:
    assert app._default_RGB_bands_from_weight({"RGB_bands": "RGB_bands: 3,2,1"}, "RGB_bands: 1,2,3") == "RGB_bands: 3,2,1"
    assert app._default_RGB_bands_from_weight({"channel_rule": "RGB + wavelets"}, "RGB_bands: 3,2,1") == "RGB_bands: 3,2,1"
    assert app._default_RGB_bands_from_weight({}, "RGB_bands: 3,2,1") == "RGB_bands: 3,2,1"


def test_run_all_uses_stage_runners_in_order_and_collects_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def make_runner(stage: str):
        def runner(received: PlatformConfig) -> dict[str, str]:
            assert received is config
            calls.append(stage)
            if stage == "vectorize":
                stats_path = received.outputs.metrics / "vector_stats.json"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text(json.dumps({"polygon_count": 1}), encoding="utf-8")
            return {"stage": stage}

        return runner

    monkeypatch.setattr(app, "STAGE_RUNNERS", {stage: make_runner(stage) for stage in app.STAGES})

    statuses = app.run_all(config)

    assert calls == list(app.DEPLOYMENT_STAGES)
    assert [status.stage for status in statuses] == list(app.DEPLOYMENT_STAGES)
    assert all(status.ok for status in statuses)
    assert statuses[0].result == {"stage": "prepare"}
    assert statuses[-1].artifacts["vector_stats"] == {"polygon_count": 1}


def test_run_all_with_progress_emits_weighted_stage_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    progress_events: list[dict[str, object]] = []

    def make_runner(stage: str):
        def runner(received: PlatformConfig, *, progress_callback=None) -> dict[str, str]:
            assert received is config
            calls.append(stage)
            assert progress_callback is not None
            progress_callback(
                {
                    "stage": stage,
                    "status": "running",
                    "percent": 0.5,
                    "processed": 1,
                    "total": 2,
                    "message": f"half {stage}",
                }
            )
            return {"stage": stage}

        return runner

    monkeypatch.setattr(app, "STAGE_RUNNERS", {stage: make_runner(stage) for stage in app.STAGES})

    statuses = app.run_all_with_progress(config, progress_callback=lambda event: progress_events.append(dict(event)))

    assert calls == list(app.DEPLOYMENT_STAGES)
    assert [status.stage for status in statuses] == list(app.DEPLOYMENT_STAGES)
    assert all(status.ok for status in statuses)
    assert any(event["stage"] == "prepare" and event["message"] == "half prepare" for event in progress_events)
    assert progress_events[-1]["overall_percent"] == 1.0


def test_run_all_resumable_skips_completed_stage_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    (config.output_dir / "patch_index.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"properties": {"patch_path": "p1.tif"}}]}),
        encoding="utf-8",
    )
    calls: list[str] = []
    progress_events: list[dict[str, object]] = []

    def make_runner(stage: str):
        def runner(received: PlatformConfig, *, progress_callback=None) -> dict[str, str]:
            calls.append(stage)
            if progress_callback is not None:
                progress_callback({"stage": stage, "status": "running", "percent": 1.0})
            return {"stage": stage}

        return runner

    monkeypatch.setattr(app, "STAGE_RUNNERS", {stage: make_runner(stage) for stage in app.STAGES})

    statuses = app.run_all_resumable_with_progress(
        config,
        progress_callback=lambda event: progress_events.append(dict(event)),
    )

    assert statuses[0].stage == "prepare"
    assert statuses[0].message == "Reused existing outputs."
    assert "prepare" not in calls
    assert calls == ["infer", "mosaic", "vectorize"]
    assert all(status.ok for status in statuses)
    assert any(event["stage"] == "prepare" and event["status"] == "skipped" for event in progress_events)


def test_deployment_cache_status_detects_config_change(tmp_path: Path) -> None:
    raw = app.build_raw_config(
        geotiff_path=str(tmp_path / "source.tif"),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
        tile_size=64,
        stride=32,
    )
    changed = app.build_raw_config(
        geotiff_path=str(tmp_path / "source.tif"),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
        tile_size=128,
        stride=64,
    )
    fingerprint = app.deployment_config_fingerprint(raw)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "patches").mkdir()
    (output_dir / "deployment_cache.json").write_text(
        json.dumps({"config_fingerprint": fingerprint}),
        encoding="utf-8",
    )

    same = app.output_cache_status(output_dir, fingerprint)
    different = app.output_cache_status(output_dir, app.deployment_config_fingerprint(changed))

    assert same["matches"]
    assert not different["matches"]
    assert different["has_cache"]


def test_clear_deployment_cache_ignores_missing_rmtree_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True)

    def flaky_rmtree(path: Path, *, onerror=None) -> None:
        assert onerror is not None
        missing = FileNotFoundError("._UAV_1_r0_c9216.tif disappeared")
        onerror(Path.unlink, str(Path(path) / "._UAV_1_r0_c9216.tif"), (FileNotFoundError, missing, None))
        Path(path).rmdir()

    monkeypatch.setattr(app.shutil, "rmtree", flaky_rmtree)

    assert app.clear_deployment_cache(output_dir) == ["patches"]
    assert not patches_dir.exists()



def test_cleanup_completed_deployment_intermediates_keeps_final_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    for name in ("patches", "patch_predictions", "rasters", "vectors", "metrics"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        (output_dir / name / "artifact.tif").write_bytes(b"x")
    (output_dir / "patch_index.geojson").write_text("{}", encoding="utf-8")
    (output_dir / "infer_manifest.json").write_text("{}", encoding="utf-8")

    removed = app.cleanup_completed_deployment_intermediates(output_dir)

    assert removed == ["patches", "patch_predictions", "patch_index.geojson", "infer_manifest.json"]
    assert not (output_dir / "patches").exists()
    assert not (output_dir / "patch_predictions").exists()
    assert not (output_dir / "patch_index.geojson").exists()
    assert not (output_dir / "infer_manifest.json").exists()
    assert (output_dir / "rasters" / "artifact.tif").exists()
    assert (output_dir / "vectors" / "artifact.tif").exists()
    assert (output_dir / "metrics" / "artifact.tif").exists()


def test_deployment_stage_completion_uses_manifest_after_intermediate_cleanup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "deployment_cache.json").write_text(
        json.dumps(
            {
                "statuses": [
                    {"stage": "prepare", "ok": True},
                    {"stage": "infer", "ok": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert app._deployment_stage_is_complete("prepare", config)
    assert app._deployment_stage_is_complete("infer", config)


def test_batch_deployment_payload_and_job_id_are_stable(tmp_path: Path) -> None:
    first = tmp_path / "a.tif"
    second = tmp_path / "b.tif"
    raw = app.build_raw_config(
        geotiff_path=str(first),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
        RGB_bands="RGB_bands: 1,2,3",
    )

    payload = app.build_batch_deployment_payload(raw, source_path=str(tmp_path), selected_rasters=[str(first), str(second)])
    same_payload = app.build_batch_deployment_payload(raw, source_path=str(tmp_path), selected_rasters=[str(first), str(second)])
    changed_payload = app.build_batch_deployment_payload(raw, source_path=str(tmp_path), selected_rasters=[str(second), str(first)])

    assert payload["deployment_mode"] == "batch"
    assert payload["selected_rasters"] == [str(first.resolve()), str(second.resolve())]
    assert app.batch_deployment_config_fingerprint(payload) == app.batch_deployment_config_fingerprint(same_payload)
    assert app.batch_deployment_job_id(payload) == app.batch_deployment_job_id(same_payload)
    assert app.batch_deployment_job_id(payload) != app.batch_deployment_job_id(changed_payload)


def test_batch_item_output_dirs_avoid_same_stem_collisions(tmp_path: Path) -> None:
    raster_a = tmp_path / "north" / "scene.tif"
    raster_b = tmp_path / "south" / "scene.tif"
    raw = app.build_raw_config(
        geotiff_path=str(raster_a),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
    )

    items = app._batch_deployment_items(
        output_dir=tmp_path / "outputs",
        raw_config=raw,
        selected_rasters=[str(raster_a), str(raster_b)],
    )

    assert len(items) == 2
    assert items[0]["output_dir"] != items[1]["output_dir"]
    assert Path(items[0]["output_dir"]).name.startswith("scene-")
    assert Path(items[1]["output_dir"]).name.startswith("scene-")


def test_batch_progress_wraps_child_stage_fields_without_overwriting_batch_fields() -> None:
    events: list[dict[str, object]] = []

    app._emit_batch_deployment_progress(
        lambda event: events.append(dict(event)),
        stage="prepare",
        percent=0.9,
        overall_percent=0.1,
        current_index=1,
        total=2,
    )

    assert events == [
        {
            "stage": "batch-deployment",
            "percent": 0.1,
            "overall_percent": 0.1,
            "current_index": 1,
            "total": 2,
        }
    ]
    wrapped = app._batch_stage_progress_payload(
        {
            "stage": "prepare",
            "status": "running",
            "processed": 4,
            "total": 9,
            "overall_percent": 0.25,
            "current_index": 99,
            "message": "preparing",
        }
    )
    assert wrapped == {
        "item_stage": "prepare",
        "item_stage_status": "running",
        "item_stage_processed": 4,
        "item_stage_total": 9,
        "item_stage_overall_percent": 0.25,
        "item_stage_current_index": 99,
        "message": "preparing",
    }


def test_batch_deployment_resume_skips_completed_and_cleans_new_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raster_a = tmp_path / "a.tif"
    raster_b = tmp_path / "b.tif"
    raster_a.write_bytes(b"fake")
    raster_b.write_bytes(b"fake")
    raw = app.build_raw_config(
        geotiff_path=str(raster_a),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
    )
    payload = app.build_batch_deployment_payload(raw, source_path=str(tmp_path), selected_rasters=[str(raster_a), str(raster_b)])
    output_dir = Path(raw["output"]["dir"])
    items = app._batch_deployment_items(output_dir=output_dir, raw_config=raw, selected_rasters=payload["selected_rasters"])
    skipped_item = dict(items[0])
    skipped_vector = Path(skipped_item["output_dir"]) / "vectors" / "landslides.gpkg"
    skipped_vector.parent.mkdir(parents=True, exist_ok=True)
    skipped_vector.write_bytes(b"gpkg")
    skipped_item.update(
        {
            "status": "completed",
            "artifacts": {"paths": {"landslides_gpkg": str(skipped_vector)}, "vector_stats": {"polygon_count": 2}},
            "polygon_count": 2,
        }
    )
    app._write_batch_deployment_manifest(
        output_dir,
        {
            "deployment_mode": "batch",
            "config_fingerprint": app.batch_deployment_config_fingerprint(payload),
            "status": "interrupted",
            "items": [skipped_item, items[1]],
        },
    )
    calls: list[str] = []
    progress_events: list[dict[str, object]] = []

    def fake_run(config: PlatformConfig, *, progress_callback=None):
        calls.append(str(config.input_raster_path))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "prepare",
                    "status": "running",
                    "percent": 1.0,
                    "processed": 4,
                    "total": 9,
                    "overall_percent": 0.25,
                    "current_index": 99,
                }
            )
        for name in ("patches", "patch_predictions"):
            (config.output_dir / name).mkdir(parents=True, exist_ok=True)
        (config.output_dir / "patch_index.geojson").write_text("{}", encoding="utf-8")
        (config.output_dir / "infer_manifest.json").write_text("{}", encoding="utf-8")
        config.outputs.vectors.mkdir(parents=True, exist_ok=True)
        config.outputs.rasters.mkdir(parents=True, exist_ok=True)
        config.outputs.metrics.mkdir(parents=True, exist_ok=True)
        (config.outputs.vectors / "landslides.gpkg").write_bytes(b"gpkg")
        (config.outputs.rasters / "probability.tif").write_bytes(b"prob")
        (config.outputs.rasters / "mask.tif").write_bytes(b"mask")
        (config.outputs.metrics / "vector_stats.json").write_text(json.dumps({"polygon_count": 1}), encoding="utf-8")
        return [app.StageStatus(stage=stage, ok=True, message="ok") for stage in app.DEPLOYMENT_STAGES]

    monkeypatch.setattr(app, "run_all_resumable_with_progress", fake_run)
    monkeypatch.setattr(
        app,
        "merge_batch_deployment_vectors",
        lambda output_dir, items: {"paths": {"merged_geojson": str(Path(output_dir) / "merged" / "landslides_merged.geojson")}, "feature_count": 3},
    )

    result = app.run_batch_deployment_background_task(
        payload=payload,
        progress_callback=lambda event: progress_events.append(dict(event)),
    )

    assert calls == [str(raster_b.resolve())]
    new_item = next(item for item in result["items"] if item["raster_path"] == str(raster_b.resolve()))
    new_output = Path(new_item["output_dir"])
    assert new_item["status"] == "completed"
    assert not (new_output / "patches").exists()
    assert not (new_output / "patch_predictions").exists()
    assert not (new_output / "patch_index.geojson").exists()
    assert not (new_output / "infer_manifest.json").exists()
    assert (new_output / "vectors" / "landslides.gpkg").exists()
    assert (new_output / "rasters" / "probability.tif").exists()

    wrapped_prepare = next(event for event in progress_events if event.get("item_stage") == "prepare")
    assert wrapped_prepare["item_stage_total"] == 9
    assert wrapped_prepare["item_stage_processed"] == 4
    assert wrapped_prepare["item_stage_current_index"] == 99
    assert wrapped_prepare["item_status"] == "running"
    assert wrapped_prepare["total"] == 2


def test_merge_batch_deployment_vectors_adds_source_fields(tmp_path: Path) -> None:
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    raster_a = tmp_path / "a.tif"
    raster_b = tmp_path / "b.tif"
    vector_a = tmp_path / "items" / "a" / "vectors" / "landslides.gpkg"
    vector_b = tmp_path / "items" / "b" / "vectors" / "landslides.gpkg"
    vector_a.parent.mkdir(parents=True, exist_ok=True)
    vector_b.parent.mkdir(parents=True, exist_ok=True)
    geopandas.GeoDataFrame(
        {"id": [1], "area_px": [4], "geometry": [box(0, 0, 1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(vector_a, driver="GPKG", layer="landslides")
    geopandas.GeoDataFrame(
        {"id": [1], "area_px": [9], "geometry": [box(2, 2, 3, 3)]},
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(vector_b, driver="GPKG", layer="landslides")
    items = [
        {
            "raster_path": str(raster_a),
            "source_stem": "a",
            "source_index": 1,
            "artifacts": {"paths": {"landslides_gpkg": str(vector_a)}},
        },
        {
            "raster_path": str(raster_b),
            "source_stem": "b",
            "source_index": 2,
            "artifacts": {"paths": {"landslides_gpkg": str(vector_b)}},
        },
    ]

    merged = app.merge_batch_deployment_vectors(tmp_path / "outputs", items)
    frame = geopandas.read_file(merged["paths"]["merged_gpkg"])

    assert merged["feature_count"] == 2
    assert {"source_raster", "source_stem", "source_index", "source_vector_path"}.issubset(frame.columns)
    assert set(frame["source_stem"]) == {"a", "b"}


def test_run_all_stops_on_captured_stage_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def prepare(received: PlatformConfig) -> dict[str, str]:
        calls.append("prepare")
        return {"stage": "prepare"}

    def infer(received: PlatformConfig) -> dict[str, str]:
        calls.append("infer")
        raise MosaicError("missing predictions")

    def mosaic(received: PlatformConfig) -> dict[str, str]:
        calls.append("mosaic")
        return {"stage": "mosaic"}

    monkeypatch.setattr(
        app,
        "STAGE_RUNNERS",
        {
            "prepare": prepare,
            "infer": infer,
            "mosaic": mosaic,
            "vectorize": mosaic,
            "evaluate": mosaic,
            "visualize": mosaic,
        },
    )

    statuses = app.run_all(config)

    assert calls == ["prepare", "infer"]
    assert [status.stage for status in statuses] == ["prepare", "infer"]
    assert statuses[0].ok
    assert not statuses[1].ok
    assert statuses[1].error_type == "MosaicError"
    assert statuses[1].message == "missing predictions"


def test_collect_artifacts_returns_paths_and_json_payloads(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.output_dir / "patch_index.geojson").write_text("{}", encoding="utf-8")
    config.outputs.metrics.mkdir(parents=True, exist_ok=True)
    config.outputs.visualizations.mkdir(parents=True, exist_ok=True)
    (config.outputs.metrics / "evaluation.json").write_text(
        json.dumps({"iou": 0.5, "confusion_matrix": {"true_positive": 2}}),
        encoding="utf-8",
    )
    (config.outputs.metrics / "vector_stats.json").write_text(
        json.dumps({"polygon_count": 3}),
        encoding="utf-8",
    )
    (config.outputs.visualizations / "overlay.png").write_bytes(b"png")

    artifacts = app.collect_artifacts(config)

    assert artifacts["paths"]["patch_index"] == str(config.output_dir / "patch_index.geojson")
    assert artifacts["evaluation"]["iou"] == 0.5
    assert artifacts["vector_stats"] == {"polygon_count": 3}
    assert artifacts["overlay_png"] == str(config.outputs.visualizations / "overlay.png")
    assert artifacts["expected_paths"]["mask_raster"] == str(config.outputs.rasters / "mask.tif")



def test_tsft_completed_job_with_checkpoint_shows_resume_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "latest_lora.pth"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[tuple[str, str]] = []

    class FakeColumn:
        def button(self, label, **_kwargs):
            calls.append(("button", label))
            return False

    class FakeStreamlit:
        def success(self, body):
            calls.append(("success", body))

        def dataframe(self, *_args, **_kwargs):
            calls.append(("dataframe", ""))

        def info(self, body):
            calls.append(("info", body))

        def container(self):
            return FakeColumn()

    monkeypatch.setattr(
        app,
        "read_job_state",
        lambda registry_root, job_id: {
            "job_id": job_id,
            "status": "completed",
            "description": "TSFT LSDSAM tsft-v1",
            "latest_checkpoint_path": str(checkpoint),
            "payload": {"dataset_id": "tsft-v1"},
            "result": {"registered_weight_id": "TSFT1"},
        },
    )
    monkeypatch.setattr(app, "_render_tsft_training_progress", lambda *_args, **_kwargs: None)

    app._render_tsft_training_job_monitor(FakeStreamlit(), registry_root=str(tmp_path / "outputs"), job_id="job-1")

    assert ("button", "Resume TSFT training") in calls
    assert any(kind == "info" and "resume" in body for kind, body in calls)


def test_latest_module_job_prefers_active_and_hides_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    active = {
        "job_id": "job-active",
        "status": "running",
        "updated_at": "2026-05-19T21:09:39Z",
    }
    completed = {
        "job_id": "job-done",
        "status": "completed",
        "updated_at": "2026-05-19T21:10:39Z",
    }
    monkeypatch.setattr(app, "list_job_states", lambda registry_root, kind=None: [completed, active])

    assert app._latest_module_job("outputs", "tsft-training") == active

    monkeypatch.setattr(app, "list_job_states", lambda registry_root, kind=None: [completed])

    assert app._latest_module_job("outputs", "tsft-training") is None


def test_module_job_progress_renderer_is_compact() -> None:
    calls: list[tuple[str, object]] = []

    class FakeStreamlit:
        def markdown(self, body, **_kwargs):
            calls.append(("markdown", body))

        def caption(self, body):
            calls.append(("caption", body))

        def button(self, label, **_kwargs):
            calls.append(("button", label))
            return False

    app._render_module_job_progress(
        FakeStreamlit(),
        registry_root="outputs",
        title="Current dataset upload",
        job={
            "job_id": "job-active",
            "description": "TAFT upload Landslide77K",
            "status": "running",
            "percent": 0.42,
            "processed": 42,
            "total": 100,
            "message": "GDCLD/train/001.tif",
        },
    )

    html_calls = [body for kind, body in calls if kind == "markdown"]
    assert html_calls
    assert "module-job-card" in str(html_calls[0])
    assert "Current dataset upload" in str(html_calls[0])
    assert "Background Jobs" not in str(html_calls[0])


def test_training_active_monitors_use_auto_refresh_fragments() -> None:
    source = inspect.getsource(app)

    assert "def _render_daft_training_job_monitor_auto" in source
    assert "def _render_tsft_training_job_monitor_auto" in source
    assert "_render_daft_training_job_monitor_auto(st, registry_root=registry_root, job_id=active_job_id)" in source
    assert "_render_tsft_training_job_monitor_auto(st, registry_root=registry_root, job_id=active_job_id)" in source
    assert "@fragment(run_every=1.5)" in source


def test_tsft_resume_job_id_changes_with_resume_run_id() -> None:
    payload = build_tsft_training_payload(
        registry_root="outputs",
        model_id="lsdsam-fewshot",
        base_weight_id="weight-1",
        dataset_id="tsft-v1",
        max_epoch=10,
        max_lr=2e-4,
        min_lr=1e-6,
        max_decay_epoch=10,
        output_dir="outputs/tsft_training",
    )
    resume_payload = {
        **payload,
        "resume": True,
        "resume_checkpoint_path": "outputs/tsft_training/checkpoints/lsdsam_tsft_latest_lora.pth",
        "resume_run_id": "run2",
    }

    assert app.tsft_training_job_id(payload) != app.tsft_training_job_id(resume_payload)
    assert "resume-run2" in app.tsft_training_job_id(resume_payload)


def test_completed_dataset_upload_job_only_blocks_when_dataset_exists(tmp_path: Path) -> None:
    payload = {"dataset_class": "TSFT", "version": "missing-neg"}
    completed_job = {"status": "completed", "payload": payload}

    assert app._job_is_blocking_resubmission(
        completed_job,
        registry_root=str(tmp_path / "outputs"),
        payload=payload,
    ) is False


def _config(tmp_path: Path) -> PlatformConfig:
    raw = app.build_raw_config(
        geotiff_path=str(tmp_path / "source.tif"),
        output_dir=str(tmp_path / "outputs"),
        model="mock",
    )
    return app.normalize_config(raw)


def test_daft_no_evaluation_message_uses_last_checkpoint_policy() -> None:
    source = Path("lsd_app/web/app.py").read_text(encoding="utf-8")

    assert "No Evaluation dataset selected. DAFT will use the last/latest checkpoint for registration" in source
    assert "Best checkpoint will be selected from supervised train metrics" not in source



def test_job_heartbeat_stall_normalizes_to_stalled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.web import jobs

    job_id = "daft-training-stalled"
    path = jobs.state_path_for_job(tmp_path, job_id)
    state = {
        "job_id": job_id,
        "kind": "daft-training",
        "status": "running",
        "execution_mode": "subprocess",
        "process_pid": 12345,
        "heartbeat_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    jobs._write_json_atomic(path, state)
    monkeypatch.setattr(jobs, "is_job_running", lambda _job_id: False)
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 12345)

    normalized = jobs.read_job_state(tmp_path, job_id)

    assert normalized["status"] == "stalled"
    assert "heartbeat" in normalized["stall_reason"]


def test_live_worker_with_stale_business_progress_normalizes_to_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lsd_app.web import jobs

    job_id = "daft-training-progress-stalled"
    path = jobs.state_path_for_job(tmp_path, job_id)
    jobs._write_json_atomic(
        path,
        {
            "job_id": job_id,
            "kind": "daft-training",
            "status": "running",
            "execution_mode": "subprocess",
            "process_pid": 12346,
            "worker_pid": 12346,
            "heartbeat_at": jobs._utc_now(),
            "last_progress_at": "2026-01-01T00:00:00Z",
            "phase": "train",
            "batch_subphase": "forward",
            "current_epoch": 4,
            "current_batch": 1,
            "updated_at": jobs._utc_now(),
        },
    )
    monkeypatch.setattr(jobs, "is_job_running", lambda _job_id: True)
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 12346)
    monkeypatch.setattr(jobs, "_pid_state", lambda pid: "S")

    normalized = jobs.read_job_state(tmp_path, job_id)

    assert normalized["status"] == "stalled"
    assert "training/inference progress" in normalized["stall_reason"]
    assert "phase=train" in normalized["stall_reason"]


def test_cancelling_d_state_normalizes_to_uninterruptible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.web import jobs

    job_id = "daft-training-d-state"
    path = jobs.state_path_for_job(tmp_path, job_id)
    jobs._write_json_atomic(
        path,
        {
            "job_id": job_id,
            "kind": "daft-training",
            "status": "cancelling",
            "cancel_requested": True,
            "execution_mode": "subprocess",
            "process_pid": 222,
            "heartbeat_at": jobs._utc_now(),
            "updated_at": jobs._utc_now(),
        },
    )
    monkeypatch.setattr(jobs, "is_job_running", lambda _job_id: False)
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 222)
    monkeypatch.setattr(jobs, "_pid_state", lambda pid: "D")

    normalized = jobs.read_job_state(tmp_path, job_id)

    assert normalized["status"] == "stalled_uninterruptible"
    assert "restart WSL/CUDA" in normalized["message"]


def test_terminal_d_state_normalizes_to_uninterruptible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.web import jobs

    job_id = "daft-training-cancelled-but-d-state"
    path = jobs.state_path_for_job(tmp_path, job_id)
    jobs._write_json_atomic(
        path,
        {
            "job_id": job_id,
            "kind": "daft-training",
            "status": "cancelled",
            "cancel_requested": True,
            "execution_mode": "subprocess",
            "process_pid": 11834,
            "worker_pid": 11834,
            "heartbeat_at": jobs._utc_now(),
            "updated_at": jobs._utc_now(),
            "finished_at": jobs._utc_now(),
        },
    )
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 11834)
    monkeypatch.setattr(jobs, "_pid_state", lambda pid: "D")

    normalized = jobs.read_job_state(tmp_path, job_id)

    assert normalized["status"] == "stalled_uninterruptible"
    assert normalized["process_pid"] == 11834
    assert "restart WSL/CUDA" in normalized["message"]


def test_cuda_device_lock_blocks_second_long_job_and_recovers_stale_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lsd_app.web import jobs

    lock_path = jobs.device_lock_root(tmp_path) / "cuda-0.json"
    jobs._write_json_atomic(
        lock_path,
        {
            "device_lock_id": "cuda-0",
            "device": "cuda:0",
            "job_id": "old-job",
            "kind": "daft-training",
            "worker_pid": 333,
            "process_pid": 333,
            "heartbeat_at": jobs._utc_now(),
        },
    )
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 333)
    monkeypatch.setattr(jobs, "_pid_holds_gpu", lambda pid: False)

    blocked = jobs._startup_blocking_device_lock(
        tmp_path,
        job_id="new-job",
        kind="tsft-training",
        payload={"device": "cuda"},
    )
    assert blocked is not None
    assert blocked["job_id"] == "old-job"

    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: False)
    blocked = jobs._startup_blocking_device_lock(
        tmp_path,
        job_id="new-job",
        kind="tsft-training",
        payload={"device": "cuda"},
    )
    assert blocked is None
    assert not lock_path.exists()



def test_startup_blocking_device_lock_scans_legacy_job_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lsd_app.web import jobs

    old_job = jobs.state_path_for_job(tmp_path, "old-done-but-stuck")
    jobs._write_json_atomic(
        old_job,
        {
            "job_id": "old-done-but-stuck",
            "kind": "daft-training",
            "status": "cancelled",
            "payload": {"device": "cuda"},
            "device": "cuda",
            "process_pid": 444,
            "worker_pid": 444,
        },
    )
    monkeypatch.setattr(jobs, "_pid_is_alive", lambda pid: pid == 444)
    monkeypatch.setattr(jobs, "_pid_holds_gpu", lambda pid: False)
    monkeypatch.setattr(jobs, "_pid_state", lambda pid: "D")

    blocked = jobs._startup_blocking_device_lock(
        tmp_path,
        job_id="new-job",
        kind="evaluation",
        payload={"device": "auto"},
    )

    assert blocked is not None
    assert blocked["source"] == "job_state"
    assert blocked["job_id"] == "old-done-but-stuck"
    assert blocked["status"] == "stalled_uninterruptible"
    assert blocked["pid_state"] == "D"

    blocked_state = jobs._resource_blocked_state({"job_id": "new-job", "kind": "evaluation"}, blocked)
    assert blocked_state["status"] == "stalled_uninterruptible"
    assert "restart WSL/CUDA" in blocked_state["message"]
