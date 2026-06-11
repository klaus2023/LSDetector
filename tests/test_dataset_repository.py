import json
from pathlib import Path

import pytest

from lsd_app.data import build_dataset_repository_summary, inspect_daft_upload_target
from lsd_app.data.repository import DatasetStandardizationError
from lsd_app.core.paths import resolve_user_path
from lsd_app.registry import RegistryError, list_datasets, register_dataset


def test_standardized_taft_upload_creates_hwc_train_test_repository(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    imagery = tmp_path / "raw_images"
    labels = tmp_path / "raw_labels"
    for index in range(20):
        _write_chip(imagery / f"chip_{index:02d}.tif", value=index + 1)
        _write_label(labels / f"chip_{index:02d}.tif", value=index % 2)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TAFT v1",
        dataset_class="TAFT",
        imagery_path=imagery,
        label_path=labels,
        RGB_bands="RGB_bands: 3,2,1",
        version="v1",
        subset_name="20260505",
        standardize=True,
    )

    assert entry.dataset_id == "TAFT-v1"
    assert entry.sample_shape_layout == "(H, W, C)"
    assert entry.sample_dtype == "uint16"
    assert entry.sample_count == 20
    assert entry.train_count == 19
    assert entry.test_count == 1
    assert entry.has_gt is True

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["subset"] == "20260505"
    assert subset["sample_count"] == 20
    assert subset["train_count"] == 19
    assert subset["test_count"] == 1
    assert subset["has_label"] is True
    assert subset["sample_layout"] == "(H, W, C)"
    assert manifest["standardization"]["band_policy"] == "preserve all source bands; model adapters extract RGB_bands later"
    assert manifest["standardization"]["dtype"] == "uint16"
    assert manifest["standardization"]["label_dtype"] == "uint8"

    image_path = resolve_user_path(subset["samples"][0]["image_path"], workspace_root=tmp_path / "outputs")
    stored = _read_chip(image_path)
    assert stored.shape == (4, 5, 3)
    assert stored.dtype == np.uint16
    assert stored.min() == 1
    assert stored.max() == 20

    summary = build_dataset_repository_summary(store_root=tmp_path / "outputs" / "registry")
    assert summary[0]["dataset_class"] == "TAFT"
    assert summary[0]["sample_count"] == 20
    assert summary[0]["train_count"] == 19
    assert summary[0]["test_count"] == 1


def test_standardized_daft_upload_is_images_only(tmp_path: Path) -> None:
    imagery = tmp_path / "daft_images"
    for index in range(3):
        _write_chip(imagery / f"target_{index}.tif", value=100 + index)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-v1",
        standardize=True,
    )

    assert entry.has_gt is False
    assert entry.sample_count == 3
    assert entry.train_count == 3
    assert entry.test_count == 0
    assert (Path(entry.repository_path) / "images").is_dir()
    assert not (Path(entry.repository_path) / "labels").exists()


def test_standardized_daft_upload_preserves_float_image_dtype(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "float_images"
    _write_float_chip(imagery / "float_chip.tif")

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT float target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="float-target-v1",
        standardize=True,
    )

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    image_path = resolve_user_path(subset["samples"][0]["image_path"], workspace_root=tmp_path / "outputs")
    with rasterio.open(image_path) as src:
        stored = np.moveaxis(src.read(), 0, -1)
    assert entry.sample_dtype == "float32"
    assert stored.dtype == np.float32
    assert stored.shape == (4, 5, 3)
    assert float(stored.min()) == pytest.approx(0.0)
    assert float(stored.max()) == pytest.approx(1.0)
    assert float(stored[2, 2, 1]) == pytest.approx(0.5, abs=1e-6)


def test_standardized_evaluation_upload_creates_eval_repository(tmp_path: Path) -> None:
    imagery = tmp_path / "eval_images"
    labels = tmp_path / "eval_labels"
    for index in range(2):
        _write_chip(imagery / f"eval_{index}.tif", value=70 + index)
        _write_label(labels / f"eval_{index}.tif", value=1)

    with pytest.raises(RegistryError, match="Evaluation datasets require"):
        register_dataset(
            registry_root=tmp_path / "outputs",
            name="Eval missing label",
            dataset_class="Evaluation",
            imagery_path=imagery,
            label_path=None,
            RGB_bands="RGB_bands: 1,2,3",
            version="eval-missing",
            standardize=True,
        )

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="Eval target",
        dataset_class="Evaluation",
        imagery_path=imagery,
        label_path=labels,
        RGB_bands="RGB_bands: 1,2,3",
        version="eval-v1",
        standardize=True,
    )

    root = Path(entry.repository_path)
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    samples = manifest["subsets"][0]["samples"]
    assert entry.dataset_class == "Evaluation"
    assert entry.has_gt is True
    assert entry.sample_count == 2
    assert entry.train_count == 0
    assert entry.test_count == 2
    assert (root / "eval" / "images" / "eval_0.tif").exists()
    assert {sample["split"] for sample in samples} == {"eval"}


def test_standardized_evaluation_tiff_slicing_writes_eval_split(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "eval_rasters"
    _write_chip(imagery / "eval_scene.tif", value=90)
    polygons = tmp_path / "eval_landslides.geojson"
    _write_polygon_vector(polygons, crs="EPSG:3857")

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="Evaluation sliced",
        dataset_class="Evaluation",
        imagery_path=imagery,
        label_path=polygons,
        RGB_bands="RGB_bands: 1,2,3",
        label_polarity="foreground_white",
        version="eval-sliced",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "eval" / "images").glob("*.tif"))
    label_chips = sorted((root / "eval" / "labels").glob("*.tif"))
    assert entry.dataset_class == "Evaluation"
    assert entry.has_gt is True
    assert entry.sample_count > 0
    assert entry.train_count == 0
    assert entry.test_count == entry.sample_count
    assert len(image_chips) == len(label_chips) == entry.sample_count

    saw_positive = False
    for image_path in image_chips:
        label_path = root / "eval" / "labels" / image_path.name
        with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
            assert (label_src.height, label_src.width) == (image_src.height, image_src.width)
            assert label_src.count == 1
            assert label_src.dtypes[0] == "uint8"
            label = label_src.read(1)
        assert set(np.unique(label).tolist()) <= {0, 255}
        saw_positive = saw_positive or bool(np.any(label == 255))
    assert saw_positive is True

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert {sample["split"] for sample in subset["samples"]} == {"eval"}
    assert subset["upload_method"] == "tiff_slicing"
    assert subset["label_source"] == "polygon"
    assert subset["eval_count"] == entry.sample_count
    assert subset["positive_count"] >= 1
    assert manifest["standardization"]["upload_method"] == "tiff_slicing"
    assert manifest["standardization"]["no_feature_tile_policy"] == "all_valid"


def test_standardized_daft_packaged_upload_resumes_matching_partial_target(tmp_path: Path) -> None:
    imagery = tmp_path / "daft_images"
    for index in range(3):
        _write_chip(imagery / f"target_{index}.tif", value=100 + index)

    progress: list[dict[str, object]] = []
    first = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-resume-v1",
        standardize=True,
        progress_callback=lambda row: progress.append(dict(row)),
    )
    root = Path(first.repository_path)
    first_manifest = Path(first.manifest_path)
    first_manifest.unlink()
    (root / "images" / "target_2.tif").unlink()

    status = inspect_daft_upload_target(
        store_root=tmp_path / "outputs" / "registry",
        dataset_id=first.dataset_id,
        version=first.version,
        imagery_path=imagery,
        RGB_bands="RGB_bands: 1,2,3",
        upload_method="packaged_images",
    )
    assert status["can_resume"] is True
    assert status["config_matches"] is True

    resumed_progress: list[dict[str, object]] = []
    resumed = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-resume-v1",
        standardize=True,
        resume=True,
        progress_callback=lambda row: resumed_progress.append(dict(row)),
    )

    assert resumed.sample_count == 3
    assert len(list((root / "images").glob("*.tif"))) == 3
    assert Path(resumed.manifest_path).exists()
    assert any(row.get("current_action") == "skipped" for row in resumed_progress)
    assert any(row.get("current_action") == "written" for row in resumed_progress)


def test_standardized_daft_packaged_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    imagery = tmp_path / "daft_images"
    for index in range(2):
        _write_chip(imagery / f"target_{index}.tif", value=100 + index)

    register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-mismatch-v1",
        standardize=True,
    )
    (tmp_path / "outputs" / "registry" / "dataset_store" / "DAFT" / "target-mismatch-v1" / "manifest.json").unlink()

    with pytest.raises(DatasetStandardizationError, match="config does not match"):
        from lsd_app.data import standardize_dataset_upload

        standardize_dataset_upload(
            store_root=tmp_path / "outputs" / "registry",
            dataset_id="daft-target-mismatch-v1",
            dataset_class="DAFT",
            version="target-mismatch-v1",
            imagery_path=imagery,
            RGB_bands="RGB_bands: 3,2,1",
            upload_method="packaged_images",
            resume=True,
        )


def test_standardized_daft_tiff_slicing_cuts_rasters_to_images_only(tmp_path: Path) -> None:
    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="ignored-name",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 3,2,1",
        version="target-sliced-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=3,
        stride=2,
    )

    root = Path(entry.repository_path)
    chips = sorted((root / "images").glob("*.tif"))
    assert entry.dataset_id == "daft-target-sliced-v1"
    assert entry.has_gt is False
    assert entry.sample_count == 4
    assert entry.train_count == 4
    assert entry.test_count == 0
    assert len(chips) == 4
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert manifest["standardization"]["upload_method"] == "tiff_slicing"
    assert manifest["subsets"][0]["tile_schema"] == "tiff_slicing;tile_size=3;stride=2;max_nodata_ratio=0.20"


def test_standardized_daft_tiff_slicing_applies_strict_aoi(tmp_path: Path) -> None:
    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    aoi = tmp_path / "aoi.geojson"
    _write_polygon_vector(aoi, crs="EPSG:3857", bounds=(19, 19, 41, 41))

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT AOI target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 1,2,3",
        version="target-sliced-aoi-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        slicing_aoi_path=aoi,
    )

    root = Path(entry.repository_path)
    chips = sorted((root / "images").glob("*.tif"))
    assert entry.sample_count == 1
    assert len(chips) == 1
    assert chips[0].name == "target_scene_r000000_c000002.tif"
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert manifest["standardization"]["slicing_aoi_path"] == str(aoi.resolve())
    assert manifest["subsets"][0]["tile_schema"].endswith(";aoi=strict_inside")


def test_standardized_daft_tiff_slicing_collects_primary_tiffs_from_folder(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "planet_tiles"
    _write_multiband_chip(imagery / "tile_0001" / "tile_0001_mosaic.tif", band_count=8)
    _write_multiband_chip(imagery / "tile_0001" / "tile_0001_mosaic_date.tif", band_count=8)
    _write_multiband_chip(imagery / "tile_0001" / "tile_0001_mosaic_provenance.tif", band_count=8)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT primary TIFF target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 6,4,2",
        version="target-primary-tiffs-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
    )

    root = Path(entry.repository_path)
    chips = sorted((root / "images").glob("*.tif"))
    assert chips
    assert all("mosaic_date" not in path.name for path in chips)
    assert all("mosaic_provenance" not in path.name for path in chips)
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    source_rasters = [
        sample.get("source_raster", "")
        for sample in manifest["subsets"][0]["samples"]
        if sample.get("source_raster")
    ]
    assert source_rasters
    assert all("tile_0001_mosaic.tif" in source for source in source_rasters)
    with rasterio.open(chips[0]) as src:
        assert src.count == 8


def test_standardized_daft_tiff_slicing_accepts_confirmed_raster_list(tmp_path: Path) -> None:
    imagery = tmp_path / "planet_tiles"
    selected = imagery / "tile_0001" / "tile_0001_mosaic.tif"
    ignored = imagery / "tile_0002" / "tile_0002_mosaic.tif"
    _write_multiband_chip(selected, band_count=8)
    _write_multiband_chip(ignored, band_count=8)
    raster_list = tmp_path / "selected_rasters.txt"
    raster_list.write_text(str(selected) + "\n", encoding="utf-8")

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT selected TIFF target",
        dataset_class="DAFT",
        imagery_path=raster_list,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 6,4,2",
        version="target-selected-tiffs-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
    )

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    source_rasters = [
        sample.get("source_raster", "")
        for sample in manifest["subsets"][0]["samples"]
        if sample.get("source_raster")
    ]
    assert source_rasters
    assert all("tile_0001_mosaic.tif" in source for source in source_rasters)
    assert all("tile_0002_mosaic.tif" not in source for source in source_rasters)


def test_standardized_daft_tiff_slicing_resumes_matching_partial_target(tmp_path: Path) -> None:
    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)

    first = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 3,2,1",
        version="target-sliced-resume-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=3,
        stride=2,
    )
    root = Path(first.repository_path)
    Path(first.manifest_path).unlink()
    sorted((root / "images").glob("*.tif"))[-1].unlink()

    resumed_progress: list[dict[str, object]] = []
    resumed = register_dataset(
        registry_root=tmp_path / "outputs",
        name="DAFT target",
        dataset_class="DAFT",
        imagery_path=imagery,
        label_path=None,
        has_gt=False,
        RGB_bands="RGB_bands: 3,2,1",
        version="target-sliced-resume-v1",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=3,
        stride=2,
        resume=True,
        progress_callback=lambda row: resumed_progress.append(dict(row)),
    )

    assert resumed.sample_count == 4
    assert len(list((root / "images").glob("*.tif"))) == 4
    assert any(row.get("current_action") == "skipped" for row in resumed_progress)
    assert any(row.get("current_action") == "written" for row in resumed_progress)


def test_standardized_tsft_upload_writes_train_only_layout(tmp_path: Path) -> None:
    imagery = tmp_path / "tsft_images"
    labels = tmp_path / "tsft_labels"
    for index in range(2):
        _write_chip(imagery / f"shot_{index}.tif", value=50 + index)
        _write_label(labels / f"shot_{index}.tif", value=1)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT few-shot",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=labels,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-1",
        standardize=True,
    )

    root = Path(entry.repository_path)
    assert (root / "train" / "images").is_dir()
    assert (root / "train" / "labels").is_dir()
    assert not (root / "test").exists()
    assert not (root / "few-shot-001").exists()
    assert entry.sample_count == 2
    assert entry.train_count == 2
    assert entry.test_count == 0

    rows = list_datasets(registry_root=tmp_path / "outputs")
    assert rows[0]["repository_path"] == "@datasets/TSFT/round-1"
    assert rows[0]["manifest_path"] == "@datasets/TSFT/round-1/manifest.json"
    assert rows[0]["resolved_repository_path"] == str(root)
    assert rows[0]["resolved_manifest_path"] == str(entry.manifest_path)
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert manifest["subsets"][0]["label_polarity"] == "foreground_white"
    assert manifest["root_dir"] == "@datasets/TSFT/round-1"


def test_standardized_tsft_foreground_black_labels_are_standardized_to_one(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "tsft_black_images"
    labels = tmp_path / "tsft_black_labels"
    _write_chip(imagery / "shot_black.tif", value=50)
    _write_label(labels / "shot_black.tif", value=0)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT black foreground",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=labels,
        RGB_bands="RGB_bands: 1,2,3",
        label_polarity="foreground_black",
        version="round-black",
        standardize=True,
    )

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["label_polarity"] == "foreground_black"
    label_path = resolve_user_path(subset["samples"][0]["label_path"], workspace_root=tmp_path / "outputs")
    with rasterio.open(label_path) as src:
        label = np.moveaxis(src.read(), 0, -1)
    assert label.dtype == np.uint8
    assert label.shape == (4, 5, 1)
    assert label.min() == 1
    assert label.max() == 1


def test_standardized_tsft_upload_rejects_legacy_test_split(tmp_path: Path) -> None:
    train_images = tmp_path / "train_images"
    train_labels = tmp_path / "train_labels"
    test_images = tmp_path / "test_images"
    test_labels = tmp_path / "test_labels"
    _write_chip(train_images / "shot_train.tif", value=50)
    _write_label(train_labels / "shot_train.tif", value=1)
    _write_chip(test_images / "shot_test.tif", value=60)
    _write_label(test_labels / "shot_test.tif", value=1)

    with pytest.raises(RegistryError, match="Evaluation dataset"):
        register_dataset(
            registry_root=tmp_path / "outputs",
            name="TSFT few-shot",
            dataset_class="TSFT",
            imagery_path=train_images,
            label_path=train_labels,
            test_imagery_path=test_images,
            test_label_path=test_labels,
            RGB_bands="RGB_bands: 1,2,3",
            version="round-2",
            standardize=True,
        )


def test_standardized_tsft_tiff_slicing_writes_equal_sized_labels_and_balanced_negatives(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    polygons = tmp_path / "landslides.shp"
    _write_polygon_vector(polygons, crs="EPSG:3857")

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT sliced",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=polygons,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-sliced",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        tsft_no_feature_tile_policy="balanced",
        tsft_negative_ratio=1.0,
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    label_chips = sorted((root / "train" / "labels").glob("*.tif"))
    assert entry.sample_count == 2
    assert entry.train_count == 2
    assert entry.test_count == 0
    assert len(image_chips) == len(label_chips) == 2
    saw_positive = False
    saw_negative = False
    for image_path in image_chips:
        label_path = root / "train" / "labels" / image_path.name
        with rasterio.open(image_path) as image_src, rasterio.open(label_path) as label_src:
            assert (label_src.height, label_src.width) == (image_src.height, image_src.width)
            assert label_src.count == 1
            assert label_src.dtypes[0] == "uint8"
            label = label_src.read(1)
        values = set(np.unique(label).tolist())
        assert values <= {0, 255}
        saw_positive = saw_positive or bool(np.any(label == 255))
        saw_negative = saw_negative or not bool(np.any(label == 255))
    assert saw_positive is True
    assert saw_negative is True
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["upload_method"] == "tiff_slicing"
    assert subset["label_polarity"] == "foreground_white"
    assert subset["positive_count"] == 1
    assert subset["negative_count"] == 1
    assert manifest["standardization"]["no_feature_tile_policy"] == "balanced"


def test_standardized_tsft_tiff_slicing_skips_auxiliary_single_band_tifs(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "planet_tiles"
    _write_multiband_chip(imagery / "tile_0001" / "tile_0001_mosaic.tif", band_count=8)
    _write_single_band_chip(imagery / "tile_0001" / "tile_0001_mosaic_date.tif")
    polygons = tmp_path / "landslides.geojson"
    _write_polygon_vector(polygons, crs="EPSG:3857", bounds=(1, 21, 9, 39))

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT Planet sliced",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=polygons,
        RGB_bands="RGB_bands: 6,4,2",
        version="round-planet-sliced",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    assert image_chips
    assert all("mosaic_date" not in path.name for path in image_chips)
    with rasterio.open(image_chips[0]) as src:
        assert src.count == 8
        data = src.read()
    assert np.any(data[5] > 0)


def test_standardized_tsft_tiff_slicing_filters_rasters_by_aoi_before_windowing(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "planet_tiles"
    _write_multiband_chip(imagery / "tile_0001" / "tile_0001_mosaic.tif", band_count=8)
    _write_multiband_chip(
        imagery / "tile_far" / "tile_far_mosaic.tif",
        band_count=8,
        origin_x=1000,
        origin_y=1040,
    )
    _write_single_band_chip(imagery / "tile_0001" / "tile_0001_mosaic_date.tif")
    polygons = tmp_path / "landslides.geojson"
    _write_polygon_vector(polygons, crs="EPSG:3857", bounds=(21, 21, 29, 29))
    aoi = tmp_path / "aoi.geojson"
    _write_polygon_vector(aoi, crs="EPSG:3857", bounds=(19, 19, 41, 41))

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT AOI raster filtered",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=polygons,
        RGB_bands="RGB_bands: 6,4,2",
        version="round-planet-aoi-filtered",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        slicing_aoi_path=aoi,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    assert image_chips
    assert all("tile_far" not in path.name for path in image_chips)
    assert all("mosaic_date" not in path.name for path in image_chips)
    with rasterio.open(image_chips[0]) as src:
        assert src.count == 8
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    source_rasters = [
        sample["source_raster"]
        for sample in manifest["subsets"][0]["samples"]
        if sample.get("source_raster")
    ]
    assert source_rasters
    assert all("tile_0001_mosaic.tif" in source for source in source_rasters)
    assert manifest["subsets"][0]["total_windows"] == 1


def test_standardized_tsft_tiff_slicing_without_polygons_writes_negative_only_dataset(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT negative only",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=None,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-negative-only",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    label_chips = sorted((root / "train" / "labels").glob("*.tif"))
    assert entry.has_gt is True
    assert entry.sample_count > 0
    assert len(image_chips) == len(label_chips) == entry.sample_count
    for label_path in label_chips:
        with rasterio.open(label_path) as label_src:
            assert label_src.count == 1
            assert label_src.dtypes[0] == "uint8"
            label = label_src.read(1)
        assert set(np.unique(label).tolist()) == {0}

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert manifest["source_label_path"] is None
    assert subset["label_source"] == "none_negative_only"
    assert subset["positive_count"] == 0
    assert subset["negative_candidate_count"] == entry.sample_count
    assert subset["negative_count"] == entry.sample_count
    assert subset["negative_count"] > 0
    assert "no_feature_tile_policy=all_valid" in subset["tile_schema"]


def test_standardized_tsft_tiff_slicing_letterboxes_small_negative_aoi(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    aoi = tmp_path / "small_aoi.geojson"
    _write_polygon_vector(aoi, crs="EPSG:3857", bounds=(11, 11, 19, 19))

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT small AOI negative only",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=None,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-small-aoi-negative",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=3,
        stride=3,
        slicing_aoi_path=aoi,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    label_chips = sorted((root / "train" / "labels").glob("*.tif"))
    assert len(image_chips) == len(label_chips) == 1
    assert image_chips[0].name == "target_scene_aoi000001_letterbox.tif"
    with rasterio.open(image_chips[0]) as src:
        image = src.read()
        assert (src.height, src.width, src.count) == (3, 3, 3)
    assert np.count_nonzero(image) > 0
    with rasterio.open(label_chips[0]) as src:
        label = src.read(1)
        assert (src.height, src.width, src.count) == (3, 3, 1)
        assert src.dtypes[0] == "uint8"
    assert set(np.unique(label).tolist()) == {0}

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["label_source"] == "none_negative_only"
    assert subset["small_aoi_fallback_count"] == 1
    assert subset["skipped_small_aoi_empty"] == 0
    assert subset["small_aoi_transform_policy"] == "letterbox_resize_pad"
    assert subset["negative_candidate_count"] == 1


def test_standardized_tsft_tiff_slicing_mixes_fixed_and_letterbox_negative_aois(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    aoi = tmp_path / "mixed_aoi.geojson"
    _write_polygon_vector_many(
        aoi,
        crs="EPSG:3857",
        bounds_list=[
            (9, 9, 41, 41),
            (11, 11, 19, 19),
        ],
    )

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT mixed AOI negative only",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=None,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-mixed-aoi-negative",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        slicing_aoi_path=aoi,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    assert len(image_chips) == 3
    assert any(path.name == "target_scene_aoi000001_letterbox.tif" for path in image_chips)
    assert any(path.name == "target_scene_aoi000002_letterbox.tif" for path in image_chips)
    fixed_chips = [path for path in image_chips if "_letterbox" not in path.name]
    assert len(fixed_chips) == 1
    with rasterio.open(root / "train" / "images" / "target_scene_aoi000002_letterbox.tif") as src:
        assert (src.height, src.width) == (2, 2)

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["aoi_candidate_count"] == 2
    assert subset["aoi_letterbox_count"] == 2
    assert subset["aoi_letterbox_policy"] == "one_patch_per_aoi_polygon"
    assert subset["small_aoi_fallback_count"] == 2
    assert subset["negative_candidate_count"] == 3
    assert subset["negative_count"] == 3


def test_standardized_tsft_tiff_slicing_applies_strict_aoi(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")

    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    polygons = tmp_path / "landslides.geojson"
    _write_polygon_vector(polygons, crs="EPSG:3857", bounds=(21, 21, 29, 29))
    aoi = tmp_path / "aoi.geojson"
    _write_polygon_vector(aoi, crs="EPSG:3857", bounds=(19, 19, 41, 41))

    entry = register_dataset(
        registry_root=tmp_path / "outputs",
        name="TSFT AOI sliced",
        dataset_class="TSFT",
        imagery_path=imagery,
        label_path=polygons,
        RGB_bands="RGB_bands: 1,2,3",
        version="round-sliced-aoi",
        standardize=True,
        upload_method="tiff_slicing",
        tile_size=2,
        stride=2,
        slicing_aoi_path=aoi,
        tsft_no_feature_tile_policy="all_valid",
    )

    root = Path(entry.repository_path)
    image_chips = sorted((root / "train" / "images").glob("*.tif"))
    assert entry.sample_count == 1
    assert len(image_chips) == 1
    assert image_chips[0].name == "target_scene_r000000_c000002.tif"
    with rasterio.open(image_chips[0]) as src:
        assert (src.height, src.width) == (2, 2)
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    subset = manifest["subsets"][0]
    assert subset["positive_count"] == 1
    assert subset["skipped_outside_aoi"] == 5
    assert subset["tile_schema"].endswith(";aoi=strict_inside")


def test_standardized_tsft_tiff_slicing_rejects_vector_without_crs(tmp_path: Path) -> None:
    imagery = tmp_path / "target_rasters"
    _write_chip(imagery / "target_scene.tif", value=100)
    polygons = tmp_path / "landslides_no_crs.shp"
    _write_polygon_vector(polygons, crs=None)

    with pytest.raises(RegistryError, match="CRS is missing"):
        register_dataset(
            registry_root=tmp_path / "outputs",
            name="TSFT sliced",
            dataset_class="TSFT",
            imagery_path=imagery,
            label_path=polygons,
            RGB_bands="RGB_bands: 1,2,3",
            version="round-sliced-no-crs",
            standardize=True,
            upload_method="tiff_slicing",
            tile_size=2,
            stride=2,
        )



def test_register_concat_dataset_entry_writes_virtual_tsft_manifest_and_registry_row(tmp_path: Path) -> None:
    from lsd_app.registry import delete_dataset_store_entry, register_concat_dataset_entry

    registry_root = tmp_path / "outputs"
    registry = registry_root / "registry"
    source_a = _write_virtual_manifest(
        registry,
        dataset_class="TSFT",
        version="round-a",
        sample_count=3,
        train_count=3,
        positive_count=2,
        negative_count=1,
        negative_candidate_count=4,
        total_windows=10,
        skipped_nodata=1,
        skipped_empty_mask=2,
        skipped_zero_rgb=3,
        low_positive_windows=4,
        mean_positive_pixel_ratio=0.25,
    )
    source_b = _write_virtual_manifest(
        registry,
        dataset_class="TSFT",
        version="round-b",
        sample_count=5,
        train_count=5,
        positive_count=3,
        negative_count=2,
        negative_candidate_count=6,
        total_windows=20,
        skipped_nodata=2,
        skipped_empty_mask=3,
        skipped_zero_rgb=4,
        low_positive_windows=5,
        mean_positive_pixel_ratio=0.5,
    )
    source_c = _write_virtual_manifest(
        registry,
        dataset_class="TSFT",
        version="round-c",
        sample_count=1,
        train_count=1,
        positive_count=4,
        negative_count=0,
        negative_candidate_count=1,
        total_windows=5,
        skipped_nodata=0,
        skipped_empty_mask=0,
        skipped_zero_rgb=0,
        low_positive_windows=1,
        mean_positive_pixel_ratio=None,
    )

    entry = register_concat_dataset_entry(
        registry_root=registry_root,
        dataset_class="TSFT",
        version="round-abc",
        source_versions=["round-a", "round-b", "round-c"],
        notes="virtual concat",
    )

    assert entry.dataset_id == "TSFT-round-abc"
    assert entry.dataset_class == "TSFT"
    assert entry.version == "round-abc"
    assert entry.sample_count == 9
    assert entry.train_count == 9
    assert entry.subset_count == 3

    rows = list_datasets(registry_root=registry_root)
    row = next(row for row in rows if row["dataset_id"] == "TSFT-round-abc")
    assert row["repository_path"] == "@datasets/TSFT/round-abc"
    assert row["sample_count"] == 9

    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert manifest["is_concat"] is True
    assert manifest["concat_backend"] == "torch.utils.data.ConcatDataset"
    assert [row["version"] for row in manifest["concat_sources"]] == ["round-a", "round-b", "round-c"]
    assert manifest["sample_count"] == 9
    assert manifest["subset_count"] == 3
    assert manifest["positive_count"] == 9
    assert manifest["negative_count"] == 3
    assert manifest["negative_candidate_count"] == 11
    assert manifest["total_windows"] == 35
    assert manifest["skipped_nodata"] == 3
    assert manifest["skipped_empty_mask"] == 5
    assert manifest["skipped_zero_rgb"] == 7
    assert manifest["low_positive_windows"] == 10
    assert manifest["mean_positive_pixel_ratio"] == pytest.approx(((0.25 * 2) + (0.5 * 3)) / 5)
    assert any("round-c" in warning for warning in manifest["warnings"])
    assert not (Path(entry.repository_path) / "train").exists()

    result = delete_dataset_store_entry(registry_root=registry_root, dataset_class="TSFT", version="round-abc")
    assert result["removed_registry_entries"] == 1
    assert not Path(entry.repository_path).exists()
    assert source_a.exists()
    assert source_b.exists()
    assert source_c.exists()


def test_register_concat_dataset_entry_validates_sources_and_class(tmp_path: Path) -> None:
    from lsd_app.registry import register_concat_dataset_entry

    registry_root = tmp_path / "outputs"
    registry = registry_root / "registry"
    _write_virtual_manifest(registry, dataset_class="DAFT", version="daft-a", sample_count=1, train_count=1)
    _write_virtual_manifest(registry, dataset_class="DAFT", version="daft-b", sample_count=2, train_count=2)

    entry = register_concat_dataset_entry(
        registry_root=registry_root,
        dataset_class="DAFT",
        version="daft-ab",
        source_versions=["daft-a", "daft-b"],
    )
    manifest = json.loads(Path(entry.manifest_path).read_text(encoding="utf-8"))
    assert entry.sample_count == 3
    assert entry.has_gt is False
    assert manifest["label_path"] is None
    assert manifest["concat_sources"][0]["sample_count"] == 1

    with pytest.raises(RegistryError, match="at least two"):
        register_concat_dataset_entry(
            registry_root=registry_root,
            dataset_class="DAFT",
            version="bad-single",
            source_versions=["daft-a"],
        )
    with pytest.raises(RegistryError, match="Missing source manifest"):
        register_concat_dataset_entry(
            registry_root=registry_root,
            dataset_class="DAFT",
            version="bad-missing",
            source_versions=["daft-a", "missing"],
        )
    with pytest.raises(RegistryError, match="only supported"):
        register_concat_dataset_entry(
            registry_root=registry_root,
            dataset_class="TAFT",
            version="bad-class",
            source_versions=["daft-a", "daft-b"],
        )
    with pytest.raises(RegistryError, match="already exists"):
        register_concat_dataset_entry(
            registry_root=registry_root,
            dataset_class="DAFT",
            version="daft-ab",
            source_versions=["daft-a", "daft-b"],
        )


def _write_virtual_manifest(
    registry: Path,
    *,
    dataset_class: str,
    version: str,
    sample_count: int,
    train_count: int,
    test_count: int = 0,
    positive_count: int = 0,
    negative_count: int = 0,
    negative_candidate_count: int = 0,
    total_windows: int = 0,
    skipped_nodata: int = 0,
    skipped_empty_mask: int = 0,
    skipped_zero_rgb: int = 0,
    low_positive_windows: int = 0,
    mean_positive_pixel_ratio: float | None = None,
) -> Path:
    root = registry / "dataset_store" / dataset_class / version
    root.mkdir(parents=True, exist_ok=True)
    subset = {
        "subset": version,
        "sample_count": sample_count,
        "train_count": train_count,
        "test_count": test_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "negative_candidate_count": negative_candidate_count,
        "total_windows": total_windows,
        "skipped_nodata": skipped_nodata,
        "skipped_empty_mask": skipped_empty_mask,
        "skipped_zero_rgb": skipped_zero_rgb,
        "low_positive_windows": low_positive_windows,
    }
    if mean_positive_pixel_ratio is not None:
        subset["mean_positive_pixel_ratio"] = mean_positive_pixel_ratio
    manifest = {
        "dataset_id": f"{dataset_class}-{version}",
        "dataset_class": dataset_class,
        "version": version,
        "root_dir": str(root),
        "imagery_path": str(root),
        "label_path": str(root) if dataset_class == "TSFT" else None,
        "has_gt": dataset_class == "TSFT",
        "RGB_bands": "RGB_bands: 1,2,3",
        "sample_shape_layout": "(H, W, C)",
        "sample_dtype": "uint16",
        "sample_count": sample_count,
        "train_count": train_count,
        "test_count": test_count,
        "subset_count": 1,
        "subsets": [subset],
        "manifest_path": str(root / "manifest.json"),
        "created_at": "2026-01-01T00:00:00Z",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root

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


def _write_float_chip(path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    data = np.stack([base, np.full((4, 5), 0.5, dtype=np.float32), 1.0 - base], axis=0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=3,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data)


def _write_multiband_chip(
    path: Path,
    *,
    band_count: int,
    origin_x: float = 0,
    origin_y: float = 40,
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(
        [np.full((4, 5), band + 1, dtype=np.uint16) for band in range(band_count)],
        axis=0,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=band_count,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(origin_x, origin_y, 10, 10),
    ) as dst:
        dst.write(data)


def _write_single_band_chip(path: Path) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((4, 5), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=5,
        count=1,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data, 1)


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


def _write_polygon_vector(path: Path, *, crs, bounds: tuple[float, float, float, float] = (1, 31, 9, 39)) -> None:
    gpd = pytest.importorskip("geopandas")
    shapely_box = pytest.importorskip("shapely.geometry").box

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(
        {"class": [1]},
        geometry=[shapely_box(*bounds)],
        crs=crs,
    )
    frame.to_file(path)


def _write_polygon_vector_many(path: Path, *, crs, bounds_list: list[tuple[float, float, float, float]]) -> None:
    gpd = pytest.importorskip("geopandas")
    shapely_box = pytest.importorskip("shapely.geometry").box

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(
        {"class": [1 for _ in bounds_list]},
        geometry=[shapely_box(*bounds) for bounds in bounds_list],
        crs=crs,
    )
    frame.to_file(path)


def _read_chip(path: Path):
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")

    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1)
