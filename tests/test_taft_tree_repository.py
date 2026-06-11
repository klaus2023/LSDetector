import json
from pathlib import Path
import shutil

import pytest

from lsd_app.cli.main import main
from lsd_app.core.paths import resolve_user_path
from lsd_app.data import build_dataset_repository_summary
from lsd_app.registry import (
    RegistryError,
    browse_dataset_repository,
    delete_dataset_store_entry,
    discover_taft_tree,
    list_datasets,
    register_taft_tree as _register_taft_tree,
)
from lsd_app.web import app


def _confirmed_overrides(root: Path, *, preset: str = "landslide77k") -> dict[str, dict[str, object]]:
    discovery = discover_taft_tree(root=root, preset=preset)
    return {
        row["source_rel"]: {
            "RGB_bands": row["RGB_bands"],
            "label_polarity": row["label_polarity"],
            "confirmed": True,
        }
        for row in discovery["subsets"]
    }


def register_taft_tree(**kwargs):
    kwargs.setdefault("subset_overrides", _confirmed_overrides(Path(kwargs["root"]), preset=kwargs.get("preset", "landslide77k")))
    return _register_taft_tree(**kwargs)


def test_discover_landslide77k_taft_tree_presets(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=2, test=1, label_value=255)
    _make_leaf(root / "CAS" / "palu", train=1, test=1, label_value=0)
    _make_leaf(root / "LMHLD" / "tw_224", train=1, test=0, label_value=0)
    (root / "landslide4sense" / "train" / "DEM_derives").mkdir(parents=True)
    _make_leaf(root / "landslide4sense", train=1, test=1, label_value=0)
    _make_leaf(root / "ELNT", train=1, test=0, label_value=0)
    _make_leaf(root / "RLZX" / "SAT", train=1, test=0, label_value=0)
    _make_leaf(root / "RLZX" / "UAV", train=1, test=0, label_value=0)

    discovery = discover_taft_tree(root=root, preset="landslide77k")

    assert discovery["subset_count"] == 7
    assert discovery["train_count"] == 8
    assert discovery["test_count"] == 3
    by_rel = {row["source_rel"]: row for row in discovery["subsets"]}
    assert by_rel["LMHLD/tw_224"]["RGB_bands"] == "RGB_bands: 3,2,1"
    assert by_rel["LMHLD/tw_224"]["auto_RGB_bands"] == "RGB_bands: 3,2,1"
    assert by_rel["ELNT"]["RGB_bands"] == "RGB_bands: 3,2,1"
    assert by_rel["ELNT"]["label_polarity"] == "foreground_black"
    assert by_rel["RLZX/SAT"]["RGB_bands"] == "RGB_bands: 3,2,1"
    assert by_rel["RLZX/SAT"]["label_polarity"] == "foreground_black"
    assert by_rel["RLZX/UAV"]["RGB_bands"] == "RGB_bands: 1,2,3"
    assert by_rel["RLZX/UAV"]["label_polarity"] == "foreground_black"
    assert by_rel["CAS/palu"]["RGB_bands"] == "RGB_bands: 1,2,3"
    assert by_rel["CAS/palu"]["auto_RGB_bands"] == "RGB_bands: 1,2,3"
    assert by_rel["GDCLD"]["label_polarity"] == "foreground_white"
    assert by_rel["GDCLD"]["auto_label_polarity"] == "foreground_white"
    assert by_rel["CAS/palu"]["label_polarity"] == "foreground_black"
    assert by_rel["CAS/palu"]["auto_label_polarity"] == "foreground_black"
    assert by_rel["landslide4sense"]["has_auxiliary"] is True
    assert list(by_rel["landslide4sense"]["auxiliary_dirs"]) == ["train/DEM_derives"]
    assert list(by_rel["CAS/palu"]["errors"]) == []

    rows = app._taft_tree_discovery_table(discovery)
    assert rows[0]["parent_group"]
    assert {row["leaf_subset"] for row in rows} == set(by_rel)


def test_discover_taft_tree_resolves_wsl_volume_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    root = data_root / "LSDSAM_finetune" / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=0, label_value=255)
    monkeypatch.setenv("LSD_APP_PATH_ALIASES", f"/mnt/example-data={data_root}")

    discovery = discover_taft_tree(root="/mnt/example-data/LSDSAM_finetune/Landslide77K")

    assert discovery["subset_count"] == 1
    assert discovery["subsets"][0]["source_rel"] == "GDCLD"


def test_register_landslide77k_taft_tree_standardizes_and_persists(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    _make_leaf(root / "CAS" / "palu", train=1, test=1, label_value=0)

    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
    )

    assert entry.dataset_id == "TAFT-landslide70k-v1"
    assert entry.dataset_class == "TAFT"
    assert entry.version == "landslide70k-v1"
    assert entry.subset_count == 2
    assert entry.train_count == 2
    assert entry.test_count == 2
    assert entry.sample_count == 4
    assert entry.has_gt is True
    assert entry.RGB_bands == "mixed"
    assert entry.RGB_bands_class == "MIXED"

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "TAFT-landslide70k-v1"
    assert manifest["subset_count"] == 2
    assert manifest["standardization"]["label_foreground_value"] == 1
    by_subset = {row["source_rel"]: row for row in manifest["subsets"]}
    assert by_subset["CAS/palu"]["subset_id"] == "CAS__palu"
    assert by_subset["CAS/palu"]["label_polarity"] == "foreground_black"
    assert by_subset["CAS/palu"]["auto_label_polarity"] == "foreground_black"
    assert by_subset["CAS/palu"]["annotation_source"] == "user_confirmed"
    assert by_subset["CAS/palu"]["user_confirmed"] is True

    cas_label_path = (
        Path(entry.repository_path)
        / "subsets"
        / "CAS__palu"
        / "train"
        / "labels"
        / "sample_000.tif"
    )
    with rasterio.open(cas_label_path) as src:
        cas_label = np.moveaxis(src.read(), 0, -1)
    assert cas_label.dtype == np.uint8
    assert cas_label.shape == (4, 5, 1)
    assert cas_label.min() == 1
    assert cas_label.max() == 1

    image_path = (
        Path(entry.repository_path)
        / "subsets"
        / "GDCLD"
        / "train"
        / "images"
        / "sample_000.tif"
    )
    with rasterio.open(image_path) as src:
        image = np.moveaxis(src.read(), 0, -1)
    assert image.dtype == np.uint16
    assert image.shape == (4, 5, 4)
    assert image.min() == 0
    assert image.max() == 29

    datasets = list_datasets(registry_root=tmp_path / "outputs")
    assert datasets[0]["dataset_id"] == "TAFT-landslide70k-v1"
    assert datasets[0]["subset_count"] == 2
    assert datasets[0]["sample_count"] == 4

    summary = build_dataset_repository_summary(store_root=tmp_path / "outputs" / "registry")
    assert len(summary) == 2
    assert {row["subset"] for row in summary} == {"GDCLD", "CAS/palu"}
    assert {row["RGB_bands"] for row in summary} == {"RGB_bands: 1,2,3"}
    assert {row["label_polarity"] for row in summary} == {"foreground_white", "foreground_black"}


def test_register_taft_tree_requires_confirmed_subset_overrides(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=0, label_value=255)

    with pytest.raises(RegistryError, match="subset annotations are required"):
        _register_taft_tree(
            registry_root=tmp_path / "outputs",
            name="Landslide77K",
            version="landslide70k-v1",
            root=root,
            preset="landslide77k",
            standardize=False,
        )

    overrides = _confirmed_overrides(root)
    overrides["GDCLD"]["confirmed"] = False
    with pytest.raises(RegistryError, match="must be confirmed"):
        _register_taft_tree(
            registry_root=tmp_path / "outputs",
            name="Landslide77K",
            version="landslide70k-v1",
            root=root,
            preset="landslide77k",
            standardize=False,
            subset_overrides=overrides,
        )


def test_register_taft_tree_manifest_uses_user_subset_overrides(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "CAS" / "palu", train=1, test=0, label_value=0)
    overrides = _confirmed_overrides(root)
    overrides["CAS/palu"]["RGB_bands"] = "RGB_bands: 3,2,1"
    overrides["CAS/palu"]["label_polarity"] = "foreground_white"

    entry = _register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
        subset_overrides=overrides,
    )

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["source_rel"] == "CAS/palu"
    assert subset["RGB_bands"] == "RGB_bands: 3,2,1"
    assert subset["label_polarity"] == "foreground_white"
    assert subset["auto_RGB_bands"] == "RGB_bands: 1,2,3"
    assert subset["auto_label_polarity"] == "foreground_black"
    assert subset["annotation_source"] == "user_confirmed"
    assert subset["user_confirmed"] is True

    subset_manifest_path = resolve_user_path(subset["subset_manifest_path"], workspace_root=tmp_path / "outputs")
    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    assert subset_manifest["RGB_bands"] == "RGB_bands: 3,2,1"
    assert subset_manifest["label_polarity"] == "foreground_white"
    assert subset_manifest["auto_RGB_bands"] == "RGB_bands: 1,2,3"
    assert subset_manifest["auto_label_polarity"] == "foreground_black"


def test_register_taft_tree_appends_new_subsets_to_existing_version(tmp_path: Path) -> None:
    root = tmp_path / "Initial70K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    first = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
    )

    extension_root = tmp_path / "Extension70K"
    _make_leaf(extension_root / "CAS" / "palu", train=1, test=0, label_value=0)
    appended = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=extension_root,
        preset="landslide77k",
        standardize=True,
        append_to_existing=True,
    )

    assert appended.repository_path == first.repository_path
    assert appended.subset_count == 2
    assert appended.sample_count == 3
    assert appended.train_count == 2
    assert appended.test_count == 1
    manifest = json.loads(Path(appended.manifest_path).read_text(encoding="utf-8"))
    by_subset = {row["source_rel"]: row for row in manifest["subsets"]}
    assert set(by_subset) == {"GDCLD", "CAS/palu"}
    assert by_subset["CAS/palu"]["label_polarity"] == "foreground_black"
    assert manifest["append_to_existing"] is True
    assert manifest["extension_sources"][-1]["subsets"] == ["CAS/palu"]

    datasets = list_datasets(registry_root=tmp_path / "outputs")
    assert len(datasets) == 1
    assert datasets[0]["subset_count"] == 2
    assert datasets[0]["sample_count"] == 3


def test_register_taft_tree_append_rejects_existing_subset(tmp_path: Path) -> None:
    root = tmp_path / "Initial70K"
    _make_leaf(root / "GDCLD", train=1, test=0, label_value=255)
    register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )

    with pytest.raises(RegistryError, match="already exist"):
        register_taft_tree(
            registry_root=tmp_path / "outputs",
            name="Landslide77K",
            version="landslide70k-v1",
            root=root,
            preset="landslide77k",
            standardize=False,
            append_to_existing=True,
        )


def test_register_taft_tree_reports_progress_and_resumes_complete_files(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=2, test=1, label_value=255)
    progress_events: list[dict] = []

    first_entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
        progress_callback=progress_events.append,
    )

    assert progress_events
    assert progress_events[0]["status"] == "discovering"
    assert {event["status"] for event in progress_events} >= {"discovering", "preparing", "running", "completed"}
    assert progress_events[-1]["status"] == "completed"
    assert progress_events[-1]["processed"] == 3
    assert first_entry.written_samples == 3
    state_path = Path(first_entry.repository_path) / "import_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["percent"] == 1.0

    second_entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
        overwrite=True,
    )

    assert second_entry.skipped_samples == 3
    assert second_entry.written_samples == 0
    assert second_entry.rewritten_samples == 0
    events_path = Path(second_entry.repository_path) / "import_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    skipped_events = [event for event in events if event["action"] == "skipped"]
    assert len(skipped_events) == 3
    assert skipped_events[0]["reason"] == "standardized copy already exists and passed integrity validation"
    assert skipped_events[0]["subset"] == "GDCLD"
    assert skipped_events[0]["output_image_path"].endswith(".tif")


def test_register_taft_tree_resume_rewrites_partial_pairs(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=3, test=0, label_value=255)
    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
    )

    sample_root = Path(entry.repository_path) / "subsets" / "GDCLD" / "train"
    image_path = sample_root / "images" / "sample_000.tif"
    missing_label_path = sample_root / "labels" / "sample_001.tif"
    mismatched_label_path = sample_root / "labels" / "sample_002.tif"
    image_path.write_bytes(b"partial")
    missing_label_path.unlink()
    _write_label(mismatched_label_path, value=255, height=3, width=5)
    resumed = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=True,
        overwrite=True,
    )

    assert resumed.rewritten_samples == 3
    assert resumed.written_samples == 3
    events_path = Path(resumed.repository_path) / "import_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    rewritten_events = [event for event in events if event["action"] == "rewritten"]
    assert len(rewritten_events) == 3
    assert {event["sample_id"] for event in rewritten_events} == {"sample_000", "sample_001", "sample_002"}
    assert all(
        event["reason"] == "existing standardized copy was incomplete or invalid, so it was rebuilt"
        for event in rewritten_events
    )
    with rasterio.open(image_path) as src:
        assert src.count == 4
        assert src.dtypes[0] == "uint16"
    with rasterio.open(missing_label_path) as src:
        assert (src.height, src.width, src.count) == (4, 5, 1)
        assert src.dtypes[0] == "uint8"
    with rasterio.open(mismatched_label_path) as src:
        assert (src.height, src.width, src.count) == (4, 5, 1)
        assert src.dtypes[0] == "uint8"


def test_browse_dataset_repository_returns_class_version_subset_tree(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )

    browser = browse_dataset_repository(registry_root=tmp_path / "outputs")
    taft = browser["classes"]["TAFT"]
    assert taft["version_count"] == 1
    assert taft["sample_count"] == 2
    assert browser["classes"]["DAFT"]["version_count"] == 0
    version = taft["versions"][0]
    assert version["version"] == "landslide70k-v1"
    assert version["subset_count"] == 1
    assert version["subsets"][0]["subset"] == "GDCLD"


def test_delete_dataset_store_entry_removes_standardized_copy_and_registry_row(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )
    repository_path = Path(entry.repository_path)
    assert repository_path.exists()
    assert list_datasets(registry_root=tmp_path / "outputs")

    result = delete_dataset_store_entry(
        registry_root=tmp_path / "outputs",
        dataset_class="TAFT",
        version="landslide70k-v1",
    )

    assert result["removed_registry_entries"] == 1
    assert not repository_path.exists()
    assert list_datasets(registry_root=tmp_path / "outputs") == []
    assert root.exists()


def test_delete_dataset_store_entry_rejects_mismatched_repository_path(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )
    repository_path = Path(entry.repository_path)
    wrong_path = tmp_path / "outputs" / "registry" / "dataset_store" / "TSFT" / "round-1"

    with pytest.raises(RegistryError, match="selected path does not match"):
        delete_dataset_store_entry(
            registry_root=tmp_path / "outputs",
            dataset_class="TAFT",
            version="landslide70k-v1",
            repository_path=wrong_path,
        )

    assert repository_path.exists()
    assert len(list_datasets(registry_root=tmp_path / "outputs")) == 1


def test_delete_dataset_store_entry_tolerates_missing_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )
    repository_path = Path(entry.repository_path)
    missing = FileNotFoundError("missing partial")

    def flaky_rmtree(path: Path, *, onerror=None, onexc=None) -> None:
        assert onerror is not None or onexc is not None
        if onexc is not None:
            onexc(Path.unlink, str(repository_path / "05338_t.tif"), missing)
        else:
            onerror(Path.unlink, str(repository_path / "05338_t.tif"), (FileNotFoundError, missing, None))
        for child in sorted(repository_path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        repository_path.rmdir()

    import lsd_app.data.repository as repository

    monkeypatch.setattr(repository.shutil, "rmtree", flaky_rmtree)

    result = delete_dataset_store_entry(
        registry_root=tmp_path / "outputs",
        dataset_class="TAFT",
        version="landslide70k-v1",
    )

    assert result["removed_registry_entries"] == 1
    assert not repository_path.exists()
    assert list_datasets(registry_root=tmp_path / "outputs") == []


def test_delete_dataset_store_entry_cleans_registry_when_folder_already_missing(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)
    entry = register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=root,
        preset="landslide77k",
        standardize=False,
    )
    repository_path = Path(entry.repository_path)
    shutil.rmtree(repository_path)

    result = delete_dataset_store_entry(
        registry_root=tmp_path / "outputs",
        dataset_class="TAFT",
        version="landslide70k-v1",
    )

    assert result["already_missing"] is True
    assert result["removed_registry_entries"] == 1
    assert list_datasets(registry_root=tmp_path / "outputs") == []


def test_register_taft_tree_blocks_pairing_errors_and_duplicate_versions(tmp_path: Path) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "CAS" / "palu", train=1, test=1, label_value=0)
    (root / "CAS" / "palu" / "train" / "labels" / "sample_000.tif").unlink()

    discovery = discover_taft_tree(root=root, preset="landslide77k")
    by_rel = {row["source_rel"]: row for row in discovery["subsets"]}
    assert by_rel["CAS/palu"]["errors"]

    with pytest.raises(RegistryError, match="TAFT tree discovery failed"):
        register_taft_tree(
            registry_root=tmp_path / "outputs",
            name="Landslide77K",
            version="landslide70k-v1",
            root=root,
            preset="landslide77k",
            standardize=False,
        )

    fixed_root = tmp_path / "Fixed70K"
    _make_leaf(fixed_root / "GDCLD", train=1, test=1, label_value=255)
    register_taft_tree(
        registry_root=tmp_path / "outputs",
        name="Landslide77K",
        version="landslide70k-v1",
        root=fixed_root,
        preset="landslide77k",
        standardize=False,
    )
    with pytest.raises(RegistryError, match="already exists"):
        register_taft_tree(
            registry_root=tmp_path / "outputs",
            name="Landslide77K",
            version="landslide70k-v1",
            root=fixed_root,
            preset="landslide77k",
            standardize=False,
        )


def test_cli_import_taft_tree_dry_run_and_standardize(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "Landslide77K"
    _make_leaf(root / "GDCLD", train=1, test=1, label_value=255)

    assert (
        main(
            [
                "datasets",
                "import-taft-tree",
                "--registry-root",
                str(tmp_path / "outputs"),
                "--version",
                "landslide70k-v1",
                "--root",
                str(root),
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"subset_count": 1' in capsys.readouterr().out

    overrides_path = tmp_path / "taft_overrides.json"
    overrides_path.write_text(json.dumps(_confirmed_overrides(root)), encoding="utf-8")
    assert (
        main(
            [
                "datasets",
                "import-taft-tree",
                "--registry-root",
                str(tmp_path / "outputs"),
                "--version",
                "landslide70k-v1",
                "--root",
                str(root),
                "--standardize",
                "--subset-overrides-json",
                str(overrides_path),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert '"dataset_id": "TAFT-landslide70k-v1"' in stdout
    assert '"subset_count": 1' in stdout

    assert main(["datasets", "summary", "--registry-root", str(tmp_path / "outputs")]) == 0
    assert '"subset": "GDCLD"' in capsys.readouterr().out


def _make_leaf(path: Path, *, train: int, test: int, label_value: int) -> None:
    for split, count in (("train", train), ("test", test)):
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _write_chip(path / split / "images" / f"sample_{index:03d}.tif", value=10 + index)
            _write_label(path / split / "labels" / f"sample_{index:03d}.tif", value=label_value)


def _write_chip(path: Path, *, value: int) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(
        [
            np.full((4, 5), value, dtype=np.uint16),
            np.arange(20, dtype=np.uint16).reshape(4, 5) + value,
            np.full((4, 5), value * 2, dtype=np.uint16),
            np.arange(20, dtype=np.uint16).reshape(4, 5),
        ],
        axis=0,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=4,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data)


def _write_label(path: Path, *, value: int, height: int = 4, width: int = 5) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((height, width), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data, 1)
