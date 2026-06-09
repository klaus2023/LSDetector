# LSDetector Usage Guide

This guide describes the intended LSDetector workflow for academic non-commercial use. The old WSL `LSD__APP` README is deprecated and should not be used as the public reference.

## 1. Conceptual Workflow

LSDetector organizes landslide mapping around seven modules:

1. **Model Repository**: select LSDFormer or LSDSAM-B/L/H.
2. **Weight Repository**: register TAFT, DAFT, TSFT, or inference-ready checkpoints.
3. **Dataset Repository**: register source-domain, target-domain, local few-shot, and evaluation datasets.
4. **DAFT Training**: adapt the model to target-domain imagery through domain alignment.
5. **TSFT Training**: refine the model with local labels or corrected predictions.
6. **Inference / Deployment**: convert GeoTIFF imagery into probability rasters, masks, and landslide polygons.
7. **Evaluation**: run independent evaluation only when ground-truth labels are available.

Deployment and evaluation should remain separate. Deployment produces candidate polygons for GIS review. Evaluation requires a labeled dataset.

## 2. Installation

The software package, trained model weights, and data artifacts should be distributed through permitted release channels. A typical local setup is:

```bash
git clone https://github.com/klaus2023/LSDetector.git
cd LSDetector
conda env create -f environment.yml
conda activate lsdector
bash tools/start_lsd_app.sh
```

Then open:

```text
http://127.0.0.1:8501
```

The CLI entry is:

```bash
python -m lsd_app.cli.main --help
```

## 3. Asset Registration

### Model Repository

Use the model repository to select the AI engine. Recommended public model IDs are:

| Model ID | Meaning |
|---|---|
| `LSDFormer` | Lightweight CNN-Transformer landslide detector |
| `LSDSAM-B` | Base-scale SAM-derived LSDSAM model |
| `LSDSAM-L` | Large-scale LSDSAM model |
| `LSDSAM-H` | Huge-scale LSDSAM model for highest-capacity deployment |

### Weight Repository

The weight repository stores metadata and local checkpoint paths. It should not blindly commit trained weights to GitHub.

Recommended weight stages:

| Stage | Meaning |
|---|---|
| `TAFT` | Task-adaptive checkpoint trained with landslide prior data |
| `DAFT` | Domain-adapted checkpoint for target imagery |
| `TSFT` | Locally calibrated checkpoint using few-shot labels |
| `inference-ready` | Deployment checkpoint selected for production mapping |

### Dataset Repository

Recommended dataset classes:

| Class | Required content | Purpose |
|---|---|---|
| `TAFT` | image-mask samples | source-domain landslide prior learning |
| `DAFT` | target-domain images, labels optional | domain adaptation and target imagery registration |
| `TSFT` | local image-mask samples | few-shot local calibration |
| `Evaluation` | image-mask samples | independent benchmark evaluation |

## 4. Inference Hyperparameters

| Parameter | Meaning | Typical value | Practical effect |
|---|---|---:|---|
| `GeoTIFF path` | Input remote-sensing raster | target image path | Source image for deployment |
| `Output Dir` | Output folder | `outputs/deployment_run` | Stores rasters, vectors, manifests, and previews |
| `model` | AI model family | `LSDSAM-H` or `LSDFormer` | Controls model capacity and speed |
| `weight` | Registered checkpoint | inference-ready weight | Determines learned mapping behavior |
| `RGB_bands` | 1-based band indexes used as RGB | `3,2,1`, `4,3,2`, or sensor-specific | Controls visible-band extraction from multi-band imagery |
| `tile_size` | Patch size for tiling | 512 for LSDSAM, 256 or 512 for LSDFormer | Larger tiles preserve context but need more memory |
| `stride` | Sliding-window step | 256 for 512 tiles | Smaller stride increases overlap and runtime |
| `batch_size` | Patches per forward pass | 1-4 | Higher values improve throughput but need more GPU memory |
| `threshold` | Probability threshold for binary mask | 0.45-0.50 | Lower values increase recall and false positives |
| `min_area_pixels` | Remove small connected regions | 0-64 | Suppresses tiny noisy polygons before vector export |

Expected deployment outputs:

```text
probability.tif
mask.tif
landslides.gpkg
landslides.shp
landslides.geojson
infer_manifest.json
```

## 5. DAFT Hyperparameters

DAFT stands for **domain-adversarial fine-tuning**. It is used when target-domain imagery is available and the image distribution differs from the source domain.

| Parameter | Meaning | Default / typical value |
|---|---|---:|
| `source_dataset_id` | Labeled source or TAFT dataset | registered TAFT dataset |
| `target_dataset_id` | Target-domain imagery dataset | registered DAFT dataset |
| `base_weight_id` | Starting checkpoint | TAFT or previous checkpoint |
| `alignment_mode` | Domain-alignment constraint | `IST`, `MMD`, or `CoRAL` |
| `total_epochs` | Number of DAFT epochs | 10 |
| `initial_lr` | Initial segmentation learning rate | `2e-4` |
| `halve_epochs` | Epochs where LR is halved | `6,8` |
| `batch_size` | Source/target batch size | 2 |
| `adv_lambda` | Adversarial loss weight | 0.05 |
| `align_lambda` | Alignment loss weight | 0.005 |
| `lora_task_r` | LoRA rank for task adapter | 16 |
| `lora_task_alpha` | LoRA scaling factor | 32 |
| `lora_dropout` | LoRA dropout rate | 0.05 |

DAFT is optional. For very large SAM backbones, domain alignment can sometimes cause negative transfer in narrow target domains; validate with a small evaluation dataset when possible.

## 6. TSFT Hyperparameters

TSFT stands for **target-specific fine-tuning**. It is the local calibration step using user-provided labels or corrected outputs.

| Parameter | Meaning | Default / typical value |
|---|---|---:|
| `dataset_id` | Registered few-shot dataset | TSFT dataset with labels |
| `base_weight_id` | Starting checkpoint | TAFT, DAFT, or inference-ready checkpoint |
| `max_epoch` | Total TSFT epochs | 10 |
| `max_lr` | Peak learning rate | `2e-4` |
| `min_lr` | Minimum learning rate | `1e-6` |
| `max_decay_epoch` | Cosine decay length | 10 |
| `batch_size` | Training batch size | 2 |
| `lora_task_r` | LoRA rank | 16 |
| `lora_task_alpha` | LoRA scaling factor | 32 |
| `lora_dropout` | LoRA dropout rate | 0.05 |
| `resume_checkpoint_path` | Optional checkpoint for resume | latest checkpoint if blank |

## 7. Recommended Practical Procedure

1. Start with a TAFT or inference-ready checkpoint.
2. Run zero-shot inference on a small target area.
3. Manually correct false positives, missed landslides, and boundary errors in GIS.
4. Register the corrected samples as a TSFT dataset.
5. Fine-tune with TSFT and rerun inference over a larger area.
6. Repeat the correction and TSFT cycle until the inventory is suitable for expert review.

## 8. GIS Review

The output polygons should be reviewed in ArcGIS, ArcGIS Pro, QGIS, or a comparable GIS platform. Candidate polygons from LSDetector are not final verified inventories by default.
