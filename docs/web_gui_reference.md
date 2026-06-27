# LSDetector Web GUI Reference

This document explains the main LSDetector Web GUI modules, options, and recommended usage patterns. It is written for users who need parameter guidance, not for developers reading source code.

The Web GUI and CLI share the same local registry. Assets registered in the GUI are visible to commands such as `lsd weights list` and `lsd datasets list`; assets registered by the CLI are also visible in the GUI.

## Global Layout

### Asset Workspace

`Asset Workspace` is the root directory for the active LSDetector asset workspace. It controls which registry the GUI reads and writes. In the normal local setup, this is the LSDetector project directory.

The registry is stored under:

```text
registry/
```

Common registry contents:

| Path | Purpose |
| --- | --- |
| `models.json` | Metadata for built-in and custom models. |
| `weights.json` | Registered weight entries. |
| `datasets.json` | Registered dataset entries. |
| `weight_repository/` | Weight files copied into the internal repository by the GUI or registration scripts. |
| `dataset_store/` | Standardized datasets and manifests. |
| `jobs/` | Background job status for upload, training, inference, and evaluation tasks. |

Change `Asset Workspace` only when you intentionally want to use a separate asset registry.

### Workspace Section

The sidebar `Workspace Section` selector switches between modules:

| Section | Purpose |
| --- | --- |
| `Model Repository` | View supported model metadata. |
| `Weight Repository` | Register, search, and delete checkpoints. |
| `Dataset Repository` | Register and inspect TAFT, DAFT, TSFT, and Evaluation datasets. |
| `DAFT Training` | Run domain-adversarial fine-tuning. |
| `TSFT Training` | Run supervised task fine-tuning. |
| `Inference` | Generate raster masks and vector landslide outputs from GeoTIFF files. |
| `Evaluation` | Evaluate registered weights on labeled evaluation datasets. |

## Model Repository

`Model Repository` shows the supported model families. Current built-in model IDs include:

- `LSDFormer`
- `LSDSAM-B`
- `LSDSAM-L`
- `LSDSAM-H`

| Field | Meaning |
| --- | --- |
| `model_id` | Stable model identifier used by training, inference, evaluation, and weight registration. |
| `input_mode` | Model input convention. `LSDFormer` uses RGB input; LSDSAM variants use RGB plus wavelet edge features internally. |
| `channel_rule` | Channel consumption rule. The dataset keeps raw bands, and model adapters extract visible RGB according to `RGB_bands`. |
| `status` / `description` | Model availability and short description. |

When registering weights or running training, inference, or evaluation, the selected `model_id` must match the actual checkpoint architecture.

## Weight Repository

`Weight Repository` manages checkpoint files and links each checkpoint to a model and training stage.

### Search And Filters

| Control | Meaning | Usage |
| --- | --- | --- |
| `Search Weights` | Searches weight ID, model ID, stage, path, source path, and creation time. | Use keywords to find a specific checkpoint quickly. |
| `Model ID` | Filters weights by model. | Check whether a model has a usable base weight. |
| `Stage` | Filters weights by training stage. | Common stages are `TAFT`, `DAFT`, `TSFT`, and `inference-ready`. |
| `File Status` | Filters by whether the registered file exists. | `available` means the file is present; `missing` means the registry path is broken. |

### Add Weight

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Model ID` | Model that owns the weight. | Must match the checkpoint architecture. For example, do not register an `LSDFormer` checkpoint as `LSDSAM-H`. |
| `Weight Stage` | Source stage of the weight. | Use `TAFT` for release base weights, `DAFT` after domain adaptation, `TSFT` after supervised fine-tuning, and `inference-ready` for deployment checkpoints. |
| `Weight ID` | User-facing stable name. | Prefer short stable names such as `LSDFormer_LS77K` or `DAFT_test`. Training and inference selectors show this ID first. |
| `Weight Path` | Source `.pth` file path. | GUI registration copies the file into `registry/weight_repository`, so later use does not depend on the original path. |
| `Add Weight` | Runs registration. | The progress display shows the copy into the internal weight repository. |

### Delete Weight

| Control | Meaning |
| --- | --- |
| `Weight ID To Delete` | Selects the registered weight entry to delete. |
| `Delete this weight from the internal repository` | Confirmation checkbox. |
| `Delete Weight` | Deletes the registry entry and, when applicable, the internal copied weight file. |

Deleting a weight does not delete other files in training output directories.

## Dataset Repository

`Dataset Repository` manages four dataset classes.

| Dataset Class | Purpose | Labels Required |
| --- | --- | --- |
| `TAFT` | Pretraining or source-domain data, commonly used as the DAFT source dataset. | Yes. |
| `DAFT` | Target-domain images for domain adaptation. | Usually no. |
| `TSFT` | Few-shot or task-specific supervised fine-tuning data. | Yes, or generated by polygon slicing. |
| `Evaluation` | Independent benchmark or evaluation data. | Yes. |

The top of the Dataset Repository shows registered versions, sample counts, train/test counts, subset counts, and manifest paths. Before training, confirm here that the intended dataset was registered correctly.

### Filterable Dataset Browser

This area browses standardized or manifest-backed dataset versions under `registry/dataset_store`.

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Dataset Class` | Filters by `TAFT`, `DAFT`, `TSFT`, or `Evaluation`. | Select the class before searching for a version. |
| `version` | Filters by dataset version. | Use this to verify whether an upload or import succeeded. |
| `Parent Group` | Parent group for TAFT trees or grouped datasets. | Useful when inspecting multi-source TAFT subsets. |
| `Subset ID` | Filters by subset ID. | Use this to locate a specific leaf subset. |
| `RGB bands` | Filters by RGB band mapping. | Check whether subsets use the expected channel order. |
| `label polarity` | Filters by foreground convention. | Check whether labels use the expected foreground direction. |
| `Has Auxiliary` | Filters by whether auxiliary directories are present. | Useful for inspecting DEM or auxiliary data records. |

The `path` and `subset_manifest_path` columns are the most important fields for debugging. Training usually resolves samples through these manifests.

### Concat Standardized Dataset Versions

This area creates a virtual concat dataset by combining multiple existing standardized DAFT or TSFT versions.

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Dataset Class` | Dataset class of the concat target. | Only `DAFT` and `TSFT` are supported, and source versions must be the same class. |
| `source versions` | Existing versions to combine. | Select at least two versions. |
| `new concat version` | Name of the new virtual version. | It must not duplicate an existing version of the same class. |
| `concat notes` | Notes for the concat dataset. | Record source versions and the reason for the merge. |
| `confirm virtual concat creation` | Confirmation checkbox. | A concat dataset creates a virtual manifest only; it does not copy or modify source data files. |
| `Create Concat Dataset` | Creates the virtual concat dataset. | The new version appears in training selectors like a normal dataset. |

### Delete Standardized Dataset Version

This area deletes a standardized dataset version and the matching registry entries.

| Control | Meaning | Caution |
| --- | --- | --- |
| `dataset version to delete` | Selects the version to delete. | The selector shows `dataset_class/version`, sample count, and path. |
| `type dataset/version to confirm` | Manual confirmation phrase. | Enter the exact phrase, such as `TSFT/example-version`, to prevent accidental deletion. |
| `Delete Dataset Version` | Deletes the dataset version. | This deletes the internal `dataset_store` version directory and matching registry entries. It does not delete original data outside the project. |

Before deleting, confirm that `path` points to the internal standardized directory you intend to remove.

### Dataset Upload Class

This radio control decides which dataset class the current upload form will create: `TAFT`, `DAFT`, `TSFT`, or `Evaluation`. The visible fields change according to the selected class.

## TAFT Pretraining Tree Upload

TAFT tree upload imports hierarchical pretraining datasets such as Landslide77K.

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `TAFT import mode` | `Register new TAFT version` creates a new version; `Add subsets to existing version` appends new subsets to an existing version. | Use a new version for the first import. Append only when adding extra subsets. |
| `TAFT version` | TAFT dataset version name. | Example: `Landslide77K`. This becomes the registry version and directory name. |
| `TAFT tree root` | Root directory of the raw TAFT tree. | Leaf subsets usually contain `train/images`, `train/labels`, `test/images`, and `test/labels`. |
| `resume matching partial import` | Continues a matching partial import. | Recommended for large datasets. If disabled and the target exists, the version is rebuilt. |
| `TAFT notes` | Dataset notes. | Record source, download version, and preprocessing details. |
| `Discover TAFT Tree` | Scans the tree and validates image/label pairs. | This scans only; it does not import data. |
| `Confirm all` | Marks all discovered subsets as confirmed. | Use this when automatic recognition is correct. |
| `Clear confirmations` | Clears subset confirmations. | Useful before changing RGB bands or label polarity. |
| `RGB_bands` | RGB band mapping for each subset. | Use 1-based band indices, for example `RGB_bands: 1,2,3`. |
| `label_polarity` | Foreground convention for each subset. | `foreground_white` means landslides are white or non-zero; `foreground_black` means landslides are black or zero. |
| `confirmed` | Confirmation state for a subset. | Every subset must be confirmed before import. |
| `Import TAFT Tree` / `Add TAFT Subsets` | Starts the background import job. | The Web GUI standardizes and copies TAFT data into `dataset_store`. For lightweight metadata-only registration, use `tools/register_release_assets.py --no-standardize`. |

## DAFT Images-Only Dataset Upload

DAFT datasets are usually target-domain images used for domain-adaptation training.

### Upload Method

| Option | Meaning |
| --- | --- |
| `Packaged Images` | Images are already tiled or organized in a directory. |
| `Tiff Slicing` | One or more GeoTIFF files are sliced into DAFT image chips. |

### DAFT Fields

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `DAFT version` | DAFT dataset version name. | Example: `DAFT-target-v1`. |
| `image directory or GeoTIFF path` | Input image path. | Use an image directory in packaged mode; use a single GeoTIFF or discovered rasters in slicing mode. |
| `processing AOI path` | Optional AOI polygon. | Supports `.shp`, `.gpkg`, and `.geojson`; slicing keeps windows inside the AOI. |
| `RGB bands` | RGB band mapping for the input raster. | Common values include `1,2,3`, `3,2,1`, `4,3,2`, and `6,4,2`. |
| `window size` | Slicing window size. | Common values are 512 or 1024. |
| `stride` | Sliding-window step size. | Smaller stride creates more overlap, more samples, and slower processing. |
| `max nodata ratio` | Drops windows with too much nodata or all-zero content. | `0.20` drops windows with more than 20 percent invalid pixels. |
| `Overwrite Standardized Target` | Replaces an existing standardized target. | Enable only when intentionally replacing an old version. |
| `DAFT notes` | Notes. | Record region, sensor, acquisition time, AOI, and preprocessing details. |
| `Upload DAFT Dataset` | Starts the upload or standardization task. | A job monitor appears below the form. |

## TSFT Few-Shot Dataset Upload

TSFT datasets are used for supervised fine-tuning.

### Upload Method

| Option | Meaning |
| --- | --- |
| `Packaged Images + Labels` | Images and labels are already tiled, usually under `images/` and `labels/`. |
| `Tiff Slicing` | GeoTIFF imagery and polygon labels are sliced into supervised samples. |

### TSFT Fields

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `TSFT version` | TSFT dataset version name. | Example: `TSFT-round-1`. |
| `train samples` | Packaged sample directory. | It should contain paired `images/` and `labels/`. |
| `source GeoTIFF path` | Raster path for slicing mode. | May be a single `.tif/.tiff` file or a folder containing candidate rasters. |
| `landslide polygons` | Polygon label path. | Supports `.shp`, `.gpkg`, `.geojson`, and `.json`; if empty, negative-only TSFT chips can be generated. |
| `processing AOI path` | Optional AOI polygon. | Restricts slicing to the AOI. It can also define the valid region for negative-only uploads. |
| `RGB bands` | RGB band mapping. | Must match the input imagery band order. |
| `Label Foreground` | Foreground convention for packaged labels. | Polygon slicing writes foreground labels automatically. |
| `Window Size` | Slicing window size. | 1024 is common for supervised chips. |
| `stride` | Sliding-window step size. | Smaller stride creates more samples. |
| `max nodata ratio` | Drops windows with too many invalid pixels. | The default 0.20 is usually a reasonable start. |
| `min polygon overlap ratio` | Minimum overlap between polygon and window. | Increase this to require landslides to be more fully inside the chip. |
| `min positive pixel ratio` | Minimum foreground pixel ratio for positive chips. | Avoids positive samples with almost no foreground. |
| `no-feature tile policy` | How to keep windows without landslides. | `only_with_features` keeps positive samples only; `balanced` keeps positives plus sampled negatives; `all_valid` keeps all valid negatives. |
| `negative ratio` | Maximum negative-to-positive ratio in `balanced` mode. | Example: 1.0 keeps at most the same number of negatives as positives. |
| `resume matching partial target` | Continues a matching partial import. | Recommended for large tasks; disabling it rebuilds the version. |
| `TSFT notes` | Notes. | Record label source, sampling policy, and region information. |
| `Upload TSFT Dataset` | Starts the upload or standardization task. | A job monitor appears below the form. |

## Evaluation Dataset Upload

Evaluation datasets are used for independent validation and must include labels.

### Upload Method

| Option | Meaning |
| --- | --- |
| `Packaged Images + Labels` | Pre-tiled image/label samples. |
| `Tiff Slicing` | GeoTIFF imagery and polygon labels are sliced into evaluation samples. |

### Evaluation Fields

Evaluation slicing is similar to TSFT slicing. The key difference is that Evaluation data must have ground truth.

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Evaluation version` | Evaluation dataset version name. | Example: `evaluation-v1`. |
| `evaluation samples` | Packaged evaluation sample directory. | It should contain paired images and labels. |
| `source GeoTIFF path` | Input raster for slicing mode. | Discovery can be used to select rasters from a folder. |
| `landslide polygons` | Ground-truth polygon path. | Required for Evaluation slicing. |
| `processing AOI path` | Optional AOI polygon. | Restricts evaluation slicing to the AOI. |
| `RGB bands` | RGB band mapping. | Must match the input imagery band order. |
| `Label Foreground` | Foreground convention for packaged labels. | Controls how landslide pixels are read from labels. |
| `Window Size` / `stride` | Slicing size and step. | Same interpretation as TSFT. |
| `max nodata ratio` | Drops windows with too many invalid pixels. | Default is 0.20. |
| `min polygon overlap ratio` | Minimum overlap between polygon and window. | Controls positive evaluation chip generation. |
| `min positive pixel ratio` | Minimum foreground ratio for positive chips. | Avoids samples with very small foreground regions. |
| `no-feature tile policy` | How to keep windows without landslides. | Evaluation commonly uses `all_valid` for more complete negative coverage. |
| `negative ratio` | Negative sampling ratio in `balanced` mode. | Used only when `balanced` is selected. |
| `resume matching partial target` | Continues a matching partial slicing task. | Enable to resume; disable to rebuild. |
| `Evaluation notes` | Notes. | Record benchmark split, region, label source, and sensor. |
| `Upload Evaluation Dataset` | Starts the upload or standardization task. | The uploaded dataset becomes available in the Evaluation module. |

## Discover GeoTIFF Rasters

Several modules include a `Discover GeoTIFF rasters` expander.

| Control | Meaning |
| --- | --- |
| `source path` | Raster file or folder to scan. |
| `AOI path` | Optional polygon used to filter rasters that intersect the AOI. |
| Raster selection table | Selects the raster files to upload or process. |

When a folder contains many GeoTIFF files or auxiliary files, use discovery first to avoid selecting QA, mask, date, or other non-primary imagery files.

## DAFT Training

DAFT means domain-adversarial fine-tuning. It uses a labeled source dataset and a target-domain dataset for domain adaptation.

### Main Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `model` | Training model architecture. | Must match the selected base weight `model_id`. |
| `base weight` | Starting checkpoint. | Usually a TAFT weight, or a same-model DAFT/TSFT weight. |
| `source dataset` | Labeled source dataset. | Usually `taft-landslide77k` or another labeled TAFT/TSFT dataset. |
| `target dataset` | Target-domain dataset. | Usually a DAFT dataset. |
| `Constraint` | Domain alignment constraint. | Options are provided by the training backend, such as `IST`, `MMD`, `CORAL`, or `none`. |
| `Evaluation Dataset` | Optional validation/evaluation dataset. | If none is selected, the latest checkpoint is registered. If selected, evaluation mIoU is used to choose the best checkpoint. |
| `Output Dir` | Training output directory. | Stores checkpoints, logs, metrics, and registration results. |
| `DAFT Weight ID` | Output weight ID. | Leave empty for auto-generation, or set a stable name such as `DAFT_test`. |
| `Epochs` | Total number of epochs. | Larger values train longer and may overfit. |
| `Initial LR` | Initial learning rate. | Default is 0.0002; lower it if loss is unstable. |
| `Halve Epochs` | Epoch list where the learning rate is halved. | Use comma-separated values such as `6,8`. |
| `Launch DAFT` | Starts DAFT training. | The background job is tracked by the monitor. |

### Advanced Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Batch Size` | Number of samples per batch. | Lower this when GPU memory is insufficient. |
| `Num Workers` | Number of DataLoader workers. | Lower values are often more stable on WSL/CUDA setups. |
| `Global Seed` | Random seed. | Use for reproducibility. |
| `Adv Lambda` | Weight for adversarial/domain loss. | Larger values strengthen domain alignment but may hurt segmentation quality. |
| `Alignment Lambda` | Weight for the selected constraint loss. | Applies to constraints such as IST, MMD, or CORAL. |
| `Save Optimizer State For Exact Resume` | Saves optimizer state. | Disabled by default to reduce checkpoint size; enable when exact resume is required. |
| `Pin Memory` | Enables DataLoader pinned memory. | Disabled by default for WSL stability. |
| `LoRA task r` | LoRA rank. | Larger rank increases capacity and resource use. |
| `LoRA task alpha` | LoRA scaling. | Usually paired with rank, for example 16/32. |
| `LoRA dropout` | LoRA dropout. | Default is 0.05 for regularization. |
| `Deterministic CUDA` | Enables deterministic CUDA behavior. | May be slower or force conservative kernels; enable only when needed. |
| `Stop DAFT training` | Requests training stop. | Already written checkpoints and logs are kept. |

## TSFT Training

TSFT means supervised fine-tuning. It is used for labeled few-shot or task-specific datasets.

### Main Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `model` | Training model architecture. | Must match the selected base weight. |
| `base weight` | Starting checkpoint. | Usually a TAFT or DAFT weight. |
| `TSFT dataset` | Labeled training dataset. | Must contain supervised samples. |
| `Evaluation Dataset` | Optional validation/evaluation dataset. | If none is selected, training metrics are the main feedback. |
| `Output Dir` | Output directory. | Stores checkpoints, logs, metrics, and registration results. |
| `TSFT Weight ID` | Output weight ID. | Leave empty for auto-generation, or set a stable release name. |
| `Max Epoch` | Maximum training epoch. | Default is 10. |
| `Max LR` | Maximum learning rate. | Default is 0.0002. |
| `Min LR` | Minimum learning rate. | Default is 0.000001. |
| `Max Decay Epoch` | Learning-rate decay horizon. | Usually set to the same value as `Max Epoch`. |
| `Launch TSFT` | Starts TSFT training. | The background monitor refreshes job state. |

### Advanced Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Batch Size` | Number of samples per batch. | Lower this when GPU memory is insufficient. |
| `Num Workers` | Number of DataLoader workers. | Low values are recommended on WSL if stability matters. |
| `Save Optimizer State For Exact Resume` | Saves optimizer state. | Enable only when exact resume is required. |
| `LoRA task r` | LoRA rank. | Larger rank increases capacity and resource use. |
| `LoRA task alpha` | LoRA scaling. | Usually paired with rank. |
| `LoRA dropout` | LoRA dropout. | Default is 0.05. |
| `Resume From Checkpoint` | Resumes from a checkpoint. | When enabled, use the specified path or the latest checkpoint in the output directory. |
| `Resume Checkpoint Path` | Checkpoint path for resume. | If empty, the latest adapter checkpoint under `Output Dir/checkpoints` is used. |
| `Stop TSFT training` | Requests training stop. | Already written files are kept. |

## Inference / Deployment

Inference converts GeoTIFF imagery into probability rasters, masks, and landslide vector outputs.

### Main Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `registered weight` | Registered weight used for inference. | Select a weight that matches the target model. |
| `GeoTIFF path` | Single input raster. | Use `Discover GeoTIFF rasters` for folders or batch-style selection. |
| `Output Dir` | Output directory. | Stores intermediate rasters, final vectors, overlays, and cache metadata. Use a new directory for a new configuration. |
| `model` | Model architecture. | Usually inferred from the selected weight. |
| `checkpoint` caption | Current checkpoint filename. | Use this to confirm that the intended weight is selected. |
| `Generate shp` | Runs full inference. | Produces raster and vector outputs. |
| `Preview` | Runs visual preview only. | Useful for quickly checking configuration. |
| `Reset Inference state` | Clears current page selections. | Does not delete output files. |

### Advanced Parameters

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `RGB bands` | RGB band mapping for the input image. | Must match the input imagery band order. |
| `tile size` | Inference patch size. | 512 is common; larger values use more GPU memory. |
| `stride` | Sliding-window step. | Smaller values increase overlap and can smooth boundaries, but are slower. |
| `Batch Size` | Number of tiles processed per batch. | Increase if GPU memory allows. |
| `threshold` | Probability threshold for binary masks. | Higher threshold may reduce false positives but increase misses. |
| `min area pixels` | Removes connected regions smaller than this pixel count. | Helps suppress tiny noisy polygons. |
| `device` | Device policy used by the GUI. | GPU first, CPU fallback. |

### Output Cache Warning

If `Output Dir` already contains intermediate files from another configuration, the GUI asks for confirmation before clearing cache files. This prevents old tiles, masks, or vectors from being mixed with a new configuration.

## Evaluation

Evaluation computes metrics for a registered weight on a labeled Evaluation dataset.

| Control | Meaning | Recommendation |
| --- | --- | --- |
| `Evaluation dataset` | Labeled evaluation dataset. | Must have `dataset_class=Evaluation` and `has_gt=True`. |
| `weight` | Weight being evaluated. | Should match `Evaluation Model`. |
| `Evaluation Model` | Model architecture used for evaluation. | If it differs from the registered weight model, the GUI shows a warning. |
| `Evaluation Output Dir` | Evaluation output directory. | Example: `outputs/evaluation_run`. |
| `Evaluation Batch Size` | Evaluation batch size. | Lower this when GPU memory is insufficient. |
| `Run Evaluation` | Starts evaluation. | Outputs files such as `metrics/evaluation.json` and `metrics/evaluation.csv`. |
| `Stop evaluation` | Requests evaluation stop. | Partially written files may remain. |

## Evaluation History

Evaluation History records previous evaluation runs. You can search, export CSV, and delete history entries. Deleting a history entry removes the record only; it does not necessarily delete the actual output directory.

| Control | Meaning | Caution |
| --- | --- | --- |
| `Search evaluation history` | Searches history entries. | Search by weight, dataset, time, or other visible fields. |
| `Export evaluation history CSV` | Exports the currently filtered history table. | Useful for comparing experiments. |
| `evaluation history entry to delete` | Selects the history entry to delete. | Affects only the history record. |
| `Delete this history entry` | Confirmation checkbox. | Required before deletion. |
| `Delete Evaluation History Entry` | Deletes the selected history record. | This is not the same as deleting evaluation output files. |

## Artifacts Panel

`Artifacts` appears after inference, evaluation, and some preview tasks finish.

| Item | Meaning |
| --- | --- |
| `Artifacts` table | Lists output file types and paths, such as probability raster, mask, vector, and overlay. |
| `evaluation.json` | Shows evaluation metrics JSON, commonly produced by Evaluation jobs. |
| `vector_stats.json` | Shows vectorization statistics, such as polygon count and area summaries. |
| `overlay.png` | Shows an overlay preview for quick visual inspection. |

If Artifacts is empty, the task may still be running, the output path may differ, or the current stage may not produce that artifact type.

## Job Monitor And Status

Long tasks run as background jobs. Job state is stored in the registry, so the page can keep tracking a job after refresh or reopen.

| Status | Meaning |
| --- | --- |
| `pending` | Submitted but not started. |
| `running` | Currently running. |
| `completed` | Finished successfully. |
| `failed` | Failed; expand the traceback for details. |
| `cancelled` | Exited after a user stop request. |
| `interrupted` | The app or process stopped before normal completion. |

`Clear previous ... display` only clears the previous monitor panel on the current page. It does not delete job records, outputs, datasets, or weights.

## Common Workflows

### Register Release Weights Only

This is suitable for users who only want TSFT or Inference:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_data_weights_v1
```

### Register TAFT Data For DAFT

Actual copied/standardized registration writes the TAFT data into `registry/dataset_store`:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_data_weights_v1 \
  --skip-weights \
  --register-taft \
  --force-rebuild
```

Lightweight metadata-only registration does not copy the large image/label files. It records manifests and `source_root` metadata only:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_data_weights_v1 \
  --skip-weights \
  --register-taft \
  --no-standardize
```

In lightweight mode, the extracted `Landslide77K/` directory must stay at the registered path because training reads samples from `source_root`.

### Run DAFT

1. Register a TAFT source dataset.
2. Upload or register a DAFT target dataset.
3. Register the selected model base weight.
4. Open `DAFT Training`, then select model, base weight, source dataset, target dataset, and output directory.
5. Optionally select an Evaluation dataset for best-checkpoint selection.
6. Click `Launch DAFT` and monitor the job.

### Run TSFT

1. Upload or register a TSFT dataset.
2. Register a base weight.
3. Open `TSFT Training`, then select model, base weight, dataset, and output directory.
4. Optionally select an Evaluation dataset.
5. Click `Launch TSFT`.

### Run Inference

1. Register an inference-ready or compatible training weight.
2. Open `Inference`.
3. Select `registered weight` and `GeoTIFF path`, or select rasters through discovery.
4. Confirm `RGB bands`, `tile size`, `stride`, `threshold`, and `Output Dir`.
5. Click `Generate shp`.

### Run Evaluation

1. Upload or register a labeled Evaluation dataset.
2. Register the weight to evaluate.
3. Open `Evaluation`, then select dataset, weight, model, and output directory.
4. Click `Run Evaluation`.
