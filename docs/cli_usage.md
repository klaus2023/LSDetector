# LSDetector CLI Usage Guide

This guide explains how to use the LSDetector command-line interface for asset registration, dataset preparation, inference, evaluation, DAFT training, and TSFT training. It is written for open-source users and uses placeholder paths only.

The CLI and the Web GUI share the same asset registry. A weight or dataset registered from the CLI appears in the Web GUI, and assets registered in the Web GUI are available to CLI commands.

## 1. Command Entry Points

After editable installation, the main command is:

```bash
lsd --help
```

If the console script is not available, use the module entry point:

```bash
PYTHONPATH="$PWD" python -m lsd_app.cli.main --help
```

All examples below assume you are inside the LSDetector project directory:

```bash
cd /path/to/LSDetector
```

Use `--help` on any command or subcommand to inspect the exact available parameters:

```bash
lsd deploy --help
lsd datasets add --help
lsd daft train --help
```

## 2. Core Concepts

### Registry Root

Most commands accept `--registry-root`. This is the asset workspace root that contains or will contain:

```text
registry/
  models.json
  weights.json
  datasets.json
  dataset_store/
  weight_repository/
```

For a normal project-local workflow, use:

```bash
--registry-root "$PWD"
```

### Weight ID

`weight_id` is the stable user-facing name of a checkpoint. Training, inference, and evaluation select weights by this ID. Prefer short readable names such as:

```text
LSDFormer_LS77K
DAFT_target_region_v1
TSFT_customer_round1
```

### Dataset ID

`dataset_id` is the stable registry ID for a dataset. DAFT, TSFT, and Evaluation commands select datasets by this ID. You can inspect available IDs with:

```bash
lsd datasets list --registry-root "$PWD"
```

### Dataset Classes

LSDetector uses four dataset classes:

| Class | Purpose | Labels |
| --- | --- | --- |
| `TAFT` | Source or pretraining data, commonly used as DAFT source data. | Required. |
| `DAFT` | Target-domain image data for domain adaptation. | Usually not required. |
| `TSFT` | Supervised task fine-tuning data. | Required. |
| `Evaluation` | Independent benchmark or validation data. | Required. |

### RGB Bands

`--rgb-bands` defines which three 1-based source bands are consumed as model-visible RGB. Common values:

```text
RGB_bands: 1,2,3
RGB_bands: 3,2,1
RGB_bands: 4,3,2
RGB_bands: 6,4,2
```

The dataset repository preserves source bands; model adapters read the RGB channels according to this metadata.

### Dry Run

Many heavy commands support `--dry-run`. Use it before long jobs to validate paths, registry IDs, payloads, and configuration without running training or inference.

## 3. Inspect The Registry

List built-in model metadata:

```bash
lsd models list --project-root "$PWD"
```

List registered weights:

```bash
lsd weights list --registry-root "$PWD"
```

List registered datasets:

```bash
lsd datasets list --registry-root "$PWD"
```

Show standardized dataset versions, subsets, and sample counts:

```bash
lsd datasets summary --registry-root "$PWD"
```

## 4. Register Weights

Use `lsd weights add` to register a checkpoint with the local asset registry.

```bash
lsd weights add \
  --registry-root "$PWD" \
  --model-id LSDFormer \
  --stage TAFT \
  --weight-id LSDFormer_LS77K \
  --version LS77K \
  --path /path/to/checkpoints/LSDFormer_LS77K.pth \
  --metrics-summary "TAFT release checkpoint"
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--model-id` | Model architecture that the checkpoint belongs to, such as `LSDFormer`, `LSDSAM-B`, `LSDSAM-L`, or `LSDSAM-H`. |
| `--stage` | Weight stage: `TAFT`, `DAFT`, `TSFT`, or `inference-ready`. |
| `--weight-id` | Stable ID used later by inference, evaluation, and training. |
| `--version` | Optional version label for display and provenance. |
| `--path` | Source checkpoint file. By default, LSDetector copies it into the internal `registry/weight_repository`. |
| `--metrics-summary` | Short human-readable metrics or provenance note. |
| `--notes` | Optional longer notes. |
| `--no-store-copy` | Debug option. Keeps the registry bound to the original checkpoint path instead of copying it into the internal repository. Avoid this for portable release setups. |

For LSDSAM weights, include adapter metadata when needed:

```bash
lsd weights add \
  --registry-root "$PWD" \
  --model-id LSDSAM-H \
  --stage TAFT \
  --weight-id LSDSAM_H_LS77K \
  --version LS77K \
  --path /path/to/checkpoints/LSDSAM_H_LS77K.pth \
  --weight-type checkpoint \
  --adapter-name lsdsam-fewshot \
  --image-size 512 \
  --input-mode rgb_wavelets \
  --channel-rule "RGB + wavelets" \
  --lora-base-r 32 \
  --lora-base-alpha 64 \
  --lora-task-r 32 \
  --lora-task-alpha 64 \
  --lora-dropout 0.05 \
  --peft-adapter-name default
```

Delete a registered weight:

```bash
lsd weights delete \
  --registry-root "$PWD" \
  --weight-id TSFT_customer_round1
```

Use `--keep-file` if you want to remove only the registry entry and keep the internal stored file.

## 5. Register Release Assets With The Helper Script

If a release provides a prepared asset package, place it in a layout like this:

```text
LSDetector_release_assets/
  weights/
    LSDFormer_LS77K.pth
    LSDSAM_H_LS77K.pth
    LSDSAM_L_77K.pth
    LSDSAM_B_71K.pth
  Landslide77K/
```

Register release weights:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_release_assets
```

Register TAFT data only if you need DAFT training. Actual copied registration writes the dataset into `registry/dataset_store`:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_release_assets \
  --skip-weights \
  --register-taft \
  --force-rebuild
```

Lightweight TAFT registration writes manifests and `source_root` metadata without copying image and label files:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_release_assets \
  --skip-weights \
  --register-taft \
  --no-standardize
```

In lightweight mode, keep the extracted source dataset at the registered path because training reads samples from `source_root`.

Use `--dry-run` to validate inputs without writing registry files:

```bash
python tools/register_release_assets.py \
  --assets-root /path/to/LSDetector_release_assets \
  --dry-run
```

## 6. Register Datasets

### Register Metadata Only

Use metadata-only registration when the data is already organized and you do not want LSDetector to copy or standardize it.

```bash
lsd datasets add \
  --registry-root "$PWD" \
  --name customer-tsft-round1 \
  --class TSFT \
  --imagery /path/to/tsft/images \
  --label /path/to/tsft/labels \
  --has-gt \
  --rgb-bands "RGB_bands: 3,2,1" \
  --tile-schema "tile_size=512;stride=256" \
  --source customer
```

### Standardize A Dataset

Use `--standardize` when you want LSDetector to copy and normalize samples into `registry/dataset_store`.

```bash
lsd datasets add \
  --registry-root "$PWD" \
  --name customer-tsft-round1 \
  --class TSFT \
  --version customer-tsft-round1 \
  --subset train \
  --imagery /path/to/tsft/images \
  --label /path/to/tsft/labels \
  --has-gt \
  --rgb-bands "RGB_bands: 3,2,1" \
  --label-polarity foreground_white \
  --standardize
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--class` | Dataset class: `TAFT`, `DAFT`, `TSFT`, or `Evaluation`. |
| `--imagery` | Imagery path, image-chip directory, or raster path. |
| `--label` | Label or ground-truth path. Required for labeled datasets. |
| `--has-gt` | Marks the dataset as having ground-truth labels. |
| `--rgb-bands` | RGB source band rule. |
| `--label-polarity` | Raster label foreground convention, such as `foreground_white`. |
| `--version` | Standardized dataset version folder. |
| `--subset` | Optional subset folder name. |
| `--standardize` | Copies and normalizes data into the internal dataset repository. |
| `--overwrite` | Replaces an existing standardized target. |

### Import A Hierarchical TAFT Tree

Use this for a source dataset organized as a hierarchy of leaf subsets.

```bash
lsd datasets import-taft-tree \
  --registry-root "$PWD" \
  --version Landslide77K \
  --root /path/to/Landslide77K \
  --standardize \
  --resume
```

Useful parameters:

| Parameter | Meaning |
| --- | --- |
| `--root` | Root directory of the hierarchical TAFT source tree. |
| `--version` | Dataset version and public registry label. |
| `--standardize` | Copies and normalizes the full tree into `dataset_store`. |
| `--resume` | Continues a partial import and skips valid completed files. |
| `--force-rebuild` | Deletes and rebuilds the target TAFT version. Use with care. |
| `--append-to-existing` | Adds newly discovered subsets to an existing TAFT version. |
| `--dry-run` | Discovers subsets and validates pairing without writing. |
| `--subset-overrides-json` | Optional JSON file with confirmed per-subset `RGB_bands` and `label_polarity` overrides. |

## 7. Run Inference

`lsd deploy` is the CLI equivalent of the Web GUI single-image `Generate shp` workflow. It reads the registered weight metadata, builds a Web-equivalent deployment payload, and writes probability rasters, masks, vector files, and summary artifacts.

Dry run first:

```bash
lsd deploy \
  --registry-root "$PWD" \
  --weight-id LSDFormer_LS77K \
  --geotiff /path/to/input_scene.tif \
  --output-dir outputs/deployment_run \
  --dry-run
```

Run inference:

```bash
lsd deploy \
  --registry-root "$PWD" \
  --weight-id LSDFormer_LS77K \
  --geotiff /path/to/input_scene.tif \
  --output-dir outputs/deployment_run \
  --tile-size 512 \
  --stride 256 \
  --batch-size 1 \
  --threshold 0.5 \
  --min-area-pixels 0 \
  --device auto
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--weight-id` | Registered checkpoint used for inference. |
| `--geotiff` | Input GeoTIFF. |
| `--output-dir` | Directory for generated rasters, vectors, metrics, and cache metadata. |
| `--model` | Optional model override. Defaults to the selected weight metadata. |
| `--rgb-bands` | Optional RGB band override. Defaults to weight metadata or project default. |
| `--tile-size` | Patch size used for sliding-window inference. |
| `--stride` | Sliding-window step. Smaller stride gives more overlap and slower inference. |
| `--batch-size` | Number of patches processed at once. Increase only if GPU memory allows. |
| `--threshold` | Probability threshold for binary mask generation. |
| `--min-area-pixels` | Removes connected components smaller than this pixel count. |
| `--device` | Runtime device. Use `auto` for GPU-first behavior or `cpu` for CPU-only runs. |
| `--clear-cache` | Clears mismatched deployment cache in the output directory before running. |

Typical outputs include:

```text
probability.tif
mask.tif
landslides.gpkg
landslides.geojson
landslides.shp
vector_stats.json
```

## 8. Run Evaluation

`lsd evaluate-dataset` evaluates a registered weight against a registered `Evaluation` dataset.

Dry run:

```bash
lsd evaluate-dataset \
  --registry-root "$PWD" \
  --dataset-id evaluation-v1 \
  --weight-id LSDFormer_LS77K \
  --output-dir outputs/evaluation_run \
  --dry-run
```

Run evaluation:

```bash
lsd evaluate-dataset \
  --registry-root "$PWD" \
  --dataset-id evaluation-v1 \
  --weight-id LSDFormer_LS77K \
  --output-dir outputs/evaluation_run \
  --batch-size 1 \
  --device auto
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--dataset-id` | Registered Evaluation dataset ID. The dataset must have labels. |
| `--weight-id` | Registered weight to evaluate. |
| `--model` | Optional model override. Defaults to the selected weight metadata. |
| `--checkpoint-path` | Optional checkpoint path override. Normally not needed. |
| `--output-dir` | Evaluation output directory. |
| `--batch-size` | Evaluation batch size. |
| `--device` | Runtime device. |

Expected outputs include `metrics/evaluation.json` and `metrics/evaluation.csv`.

## 9. Run DAFT Training

DAFT uses a labeled source dataset and a target-domain dataset for domain adaptation. Register a base weight, a source dataset, and a target dataset before launching training.

Dry run:

```bash
lsd daft train \
  --registry-root "$PWD" \
  --model-id LSDFormer \
  --base-weight-id LSDFormer_LS77K \
  --source-dataset-id taft-landslide77k \
  --target-dataset-id target-domain-v1 \
  --output-dir outputs/daft_training \
  --dry-run
```

Run training:

```bash
lsd daft train \
  --registry-root "$PWD" \
  --model-id LSDFormer \
  --base-weight-id LSDFormer_LS77K \
  --source-dataset-id taft-landslide77k \
  --target-dataset-id target-domain-v1 \
  --evaluation-dataset-id evaluation-v1 \
  --alignment-mode IST \
  --epochs 5 \
  --initial-lr 0.0002 \
  --halve-epochs 2,4 \
  --batch-size 2 \
  --num-workers 0 \
  --device auto \
  --output-dir outputs/daft_training \
  --output-weight-id DAFT_target_v1
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--model-id` | Training model architecture. Must match the base weight. |
| `--base-weight-id` | Registered starting checkpoint. |
| `--source-dataset-id` | Registered labeled source dataset, usually TAFT or TSFT. |
| `--target-dataset-id` | Registered target-domain dataset, usually DAFT. |
| `--evaluation-dataset-id` | Optional Evaluation dataset for best-checkpoint selection. |
| `--alignment-mode` | Domain alignment constraint, such as `IST`. |
| `--epochs` | Total DAFT epochs. |
| `--initial-lr` | Initial learning rate. |
| `--halve-epochs` | Comma-separated epochs where learning rate is halved. |
| `--adv-lambda` | Adversarial/domain loss weight. |
| `--align-lambda` | Alignment loss weight. |
| `--lora-task-r` | Task LoRA rank. |
| `--lora-task-alpha` | Task LoRA alpha. |
| `--lora-dropout` | LoRA dropout. |
| `--global-seed` | Random seed. |
| `--resume` | Resume from `--resume-checkpoint-path`. |
| `--save-optimizer-state` | Save optimizer state for exact resume. |
| `--pin-memory` | Enable DataLoader pinned memory. |
| `--deterministic-cuda` | Use deterministic CUDA behavior when available. |
| `--output-weight-id` | Registered ID for the trained DAFT weight. |

## 10. Run TSFT Training

TSFT is supervised fine-tuning on a registered TSFT dataset.

Dry run:

```bash
lsd tsft train \
  --registry-root "$PWD" \
  --model-id LSDFormer \
  --base-weight-id DAFT_target_v1 \
  --dataset-id customer-tsft-round1 \
  --output-dir outputs/tsft_training \
  --dry-run
```

Run training:

```bash
lsd tsft train \
  --registry-root "$PWD" \
  --model-id LSDFormer \
  --base-weight-id DAFT_target_v1 \
  --dataset-id customer-tsft-round1 \
  --evaluation-dataset-id evaluation-v1 \
  --max-epoch 10 \
  --max-lr 0.0002 \
  --min-lr 0.000001 \
  --max-decay-epoch 10 \
  --batch-size 2 \
  --num-workers 0 \
  --device auto \
  --output-dir outputs/tsft_training \
  --output-weight-id TSFT_customer_round1
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `--model-id` | Training model architecture. Must match the base weight. |
| `--base-weight-id` | Registered starting checkpoint. |
| `--dataset-id` | Registered TSFT dataset ID. |
| `--evaluation-dataset-id` | Optional Evaluation dataset. |
| `--max-epoch` | Maximum TSFT epochs. |
| `--max-lr` | Maximum learning rate. |
| `--min-lr` | Minimum learning rate. |
| `--max-decay-epoch` | Epoch where cosine decay reaches minimum LR. Defaults to `--max-epoch`. |
| `--lora-task-r` | Task LoRA rank. |
| `--lora-task-alpha` | Task LoRA alpha. |
| `--lora-dropout` | LoRA dropout. |
| `--resume` | Resume from `--resume-checkpoint-path`. |
| `--keep-lora-checkpoints` | Keep intermediate LoRA checkpoints after registering the merged output weight. |
| `--save-optimizer-state` | Save optimizer state for exact resume. |
| `--output-weight-id` | Registered ID for the trained TSFT weight. |

## 11. Legacy Config-file Pipeline

The older config-file pipeline is still available for staged deployment workflows:

```bash
lsd config --config configs/example_lsdformer.yaml
lsd prepare --config configs/example_lsdformer.yaml --dry-run
lsd infer --config configs/example_lsdformer.yaml --dry-run
lsd mosaic --config configs/example_lsdformer.yaml --dry-run
lsd vectorize --config configs/example_lsdformer.yaml --dry-run
lsd run --config configs/example_lsdformer.yaml --dry-run
```

The full `run` pipeline executes:

```text
prepare -> infer -> mosaic -> vectorize
```

Use `lsd evaluate --config ...` and `lsd visualize --config ...` when your config includes the required truth data and output paths.

## 12. Path Repair

If registry files contain stale absolute paths after moving a workspace, run:

```bash
lsd repair-paths --registry-root "$PWD"
```

This rewrites known internal registry paths to portable workspace-relative tokens where possible.

## 13. Recommended End-to-end Workflow

1. Install LSDetector and activate the environment.
2. Register base weights with `lsd weights add` or the release asset helper.
3. Register or standardize datasets with `lsd datasets add` or `lsd datasets import-taft-tree`.
4. Confirm registry state with `lsd weights list`, `lsd datasets list`, and `lsd datasets summary`.
5. Run `--dry-run` for the target operation.
6. Run inference with `lsd deploy`, evaluation with `lsd evaluate-dataset`, DAFT with `lsd daft train`, or TSFT with `lsd tsft train`.
7. Inspect outputs in the selected `--output-dir`.
8. Use the same registry in the Web GUI if you prefer visual inspection or job monitoring.

## 14. Common Troubleshooting

### Registered Weight Not Found

Run:

```bash
lsd weights list --registry-root "$PWD"
```

Confirm the exact `weight_id` and check that the registered file still exists.

### Registered Dataset Not Found

Run:

```bash
lsd datasets list --registry-root "$PWD"
lsd datasets summary --registry-root "$PWD"
```

Confirm the exact `dataset_id` and dataset class.

### Model And Weight Mismatch

The selected model must match the registered weight metadata. For example, an `LSDFormer` weight should not be used as `LSDSAM-H`.

### Output Directory Cache Mismatch

`lsd deploy` protects output directories with cache metadata. If you intentionally want to reuse an output directory with a different configuration, pass:

```bash
lsd deploy ... --clear-cache
```

You can also choose a new `--output-dir`.

### GPU Memory Issues

Lower:

```text
--batch-size
--tile-size
```

For training, also consider lowering `--num-workers` and using `--device cpu` only for small smoke tests.

### Large Dataset Registration Is Slow

Standardized TAFT registration copies and normalizes data into `registry/dataset_store`. For large source datasets, use `--resume` for interrupted imports and make sure enough disk space is available.
