"""CLI for the LSDetector engineering skeleton."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsd_app.config import ConfigError, PlatformConfig, load_config, load_raw_config, normalize_config
from lsd_app.core.channels import DEFAULT_RGB_BANDS, RGB_BAND_PRESETS, normalize_rgb_bands
from lsd_app.core.config import load_project_config
from lsd_app.core.paths import default_workspace_root, resolve_user_path
from lsd_app.core.inference import InferenceError, run_patch_inference
from lsd_app.core.metrics import EvaluationError, run_evaluate
from lsd_app.core.mosaic import MosaicError, run_mosaic
from lsd_app.core.pipeline import STAGES, run_stage
from lsd_app.core.raster import RasterError
from lsd_app.core.tiling import prepare_raster
from lsd_app.core.vectorize import VectorizeError, run_vectorize
from lsd_app.core.visualization import VisualizationError, run_visualize
from lsd_app.data import DAFT_UPLOAD_METHODS, DAFT_UPLOAD_METHOD_PACKAGED, LABEL_POLARITIES, LABEL_POLARITY_FOREGROUND_WHITE
from lsd_app.models import MODEL_SPECS, ModelAdapterError, get_model_spec
from lsd_app.registry import (
    DATASET_CLASSES,
    DATASET_TYPES,
    RegistryError,
    WEIGHT_STAGES,
    delete_weight,
    discover_taft_tree,
    list_datasets,
    list_dataset_repository_summary,
    list_models,
    list_weights,
    repair_registry_paths,
    register_dataset,
    register_prediction_run,
    register_taft_tree,
    register_weight,
)
from lsd_app.training import build_training_plan, describe_training_jobs, load_training_config, normalize_training_config
from lsd_app.training.daft_lsdsam import (
    ALIGNMENT_MODES as DAFT_ALIGNMENT_MODES,
    DEFAULT_DAFT_GLOBAL_SEED,
    DEFAULT_DAFT_NUM_WORKERS,
    DEFAULT_HALVE_EPOCHS,
    DaftLSDSAMConfig,
    DaftTrainingError,
    build_daft_training_payload,
    run_lsdsam_daft,
)
from lsd_app.training.tsft import (
    TSFT_TRAINING_DEVICE_POLICY,
    TsftLSDSAMConfig,
    TsftTrainingError,
    build_tsft_training_payload,
    run_lsdsam_tsft,
)


DEPLOYMENT_STAGES = ("prepare", "infer", "mosaic", "vectorize")
PIPELINE_STAGES = DEPLOYMENT_STAGES
DEFAULT_REGISTRY_ROOT = str(default_workspace_root())


class PipelineRunError(RuntimeError):
    """Raised when `lsd run` fails at a specific pipeline stage."""

    def __init__(self, stage: str, reason: BaseException, check_paths: list[Path]) -> None:
        self.stage = stage
        self.reason = reason
        self.check_paths = check_paths
        super().__init__(str(reason))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser for the `lsd` command."""

    parser = argparse.ArgumentParser(
        prog="lsd",
        description="Landslide detection pipeline CLI.",
    )
    parser.add_argument("--version", dest="package_version", action="store_true", help="Print package version.")

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    config_parser = subparsers.add_parser("config", help="Load and print a normalized task config.")
    config_parser.add_argument("--config", required=True, help="Path to a .yaml, .yml, or .json config file.")
    config_parser.add_argument(
        "--no-create-dirs",
        action="store_true",
        help="Validate config without creating the standard output directories.",
    )

    for stage in STAGES:
        stage_parser = subparsers.add_parser(stage, help=f"Run the {stage} stage.")
        stage_parser.add_argument("--config", help="Path to a stage config file.")
        stage_parser.add_argument("--output-dir", help="Override the output directory.")
        stage_parser.add_argument(
            "--model",
            choices=sorted(MODEL_SPECS),
            help="Model family to use for this stage.",
        )
        stage_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate wiring without running heavy compute.",
        )

    run_parser = subparsers.add_parser("run", help="Run deployment: prepare -> infer -> mosaic -> vectorize.")
    run_parser.add_argument("--config", required=True, help="Path to a .yaml, .yml, or .json platform config file.")
    run_parser.add_argument("--output-dir", help="Override the output directory.")
    run_parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        help="Model family to use for this run.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pipeline without executing stages.",
    )

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Run Web GUI-equivalent one-click deployment from a registered weight.",
    )
    deploy_parser.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root.")
    deploy_parser.add_argument("--weight-id", required=True, help="Registered Weight ID to use for inference.")
    deploy_parser.add_argument("--geotiff", required=True, help="Input GeoTIFF path.")
    deploy_parser.add_argument("--output-dir", default="outputs/deployment_run", help="Deployment output directory.")
    deploy_parser.add_argument("--model", choices=sorted(MODEL_SPECS), help="Override model ID; defaults to the selected weight's model_id.")
    deploy_parser.add_argument("--rgb-bands", default="", help=f"RGB band rule; defaults to weight metadata or {DEFAULT_RGB_BANDS!r}.")
    deploy_parser.add_argument("--tile-size", type=int, default=512, help="Patch tile size.")
    deploy_parser.add_argument("--stride", type=int, default=256, help="Sliding-window stride.")
    deploy_parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    deploy_parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    deploy_parser.add_argument("--min-area-pixels", type=int, default=0, help="Remove connected regions smaller than this pixel count.")
    deploy_parser.add_argument("--device", default="auto", help="Runtime device; use auto for GPU-first behavior.")
    deploy_parser.add_argument("--clear-cache", action="store_true", help="Clear mismatched deployment cache in the output directory before running.")
    deploy_parser.add_argument("--dry-run", action="store_true", help="Validate and print the Web-equivalent deployment payload without running.")

    eval_dataset_parser = subparsers.add_parser(
        "evaluate-dataset",
        help="Evaluate a registered Evaluation dataset with a registered weight, matching the Web Evaluation module.",
    )
    eval_dataset_parser.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root.")
    eval_dataset_parser.add_argument("--dataset-id", required=True, help="Registered Evaluation dataset ID.")
    eval_dataset_parser.add_argument("--weight-id", required=True, help="Registered Weight ID.")
    eval_dataset_parser.add_argument("--model", choices=sorted(MODEL_SPECS), help="Override model ID; defaults to the selected weight's model_id.")
    eval_dataset_parser.add_argument("--checkpoint-path", default="", help="Override checkpoint path; defaults to the selected weight path.")
    eval_dataset_parser.add_argument("--output-dir", default="outputs/evaluation_run", help="Evaluation output directory.")
    eval_dataset_parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    eval_dataset_parser.add_argument("--device", default="auto", help="Runtime device; use auto for GPU-first behavior.")
    eval_dataset_parser.add_argument("--dry-run", action="store_true", help="Validate and print the Web-equivalent evaluation payload without running.")

    models_parser = subparsers.add_parser("models", help="Inspect the local model repository.")
    models_subparsers = models_parser.add_subparsers(dest="models_command", metavar="models-command")
    models_list = models_subparsers.add_parser("list", help="List registered model families.")
    models_list.add_argument("--project-root", default=".", help="Project root used to resolve built-in model metadata.")

    weights_parser = subparsers.add_parser("weights", help="Manage local model checkpoint/weight versions.")
    weights_subparsers = weights_parser.add_subparsers(dest="weights_command", metavar="weights-command")
    weights_list = weights_subparsers.add_parser("list", help="List local weight registry entries.")
    weights_list.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    weights_add = weights_subparsers.add_parser("add", help="Register a local model weight/checkpoint.")
    weights_add.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    weights_add.add_argument("--model-id", required=True, help="Model repository ID linked to this weight.")
    weights_add.add_argument("--stage", choices=WEIGHT_STAGES, required=True, help="Training/inference stage for this weight.")
    weights_add.add_argument("--weight-id", help="Stable weight ID used by training and inference to select this weight.")
    weights_add.add_argument(
        "--version",
        dest="weight_version",
        default="",
        help="Optional weight version; defaults to the source filename.",
    )
    weights_add.add_argument("--path", required=True, help="Local checkpoint path.")
    weights_add.add_argument("--metrics-summary", default="", help="Optional compact metric summary.")
    weights_add.add_argument("--notes", default="", help="Optional weight notes.")
    weights_add.add_argument("--weight-type", default="checkpoint", help="Weight kind, e.g. checkpoint, lora_task, sam_base.")
    weights_add.add_argument("--base-checkpoint-path", help="Optional base checkpoint required by adapter composition.")
    weights_add.add_argument("--adapter-profile", default="", help="Optional adapter profile, e.g. lsdsam-sam-b-lora-edge4-after512-cf-mh.")
    weights_add.add_argument("--adapter-name", default="", help="Optional concrete model adapter name.")
    weights_add.add_argument("--image-size", type=int, help="Optional model image size metadata.")
    weights_add.add_argument("--input-mode", default="", help="Optional input mode metadata.")
    weights_add.add_argument("--channel-rule", default="", help="Optional channel rule metadata.")
    weights_add.add_argument("--lora-base-r", type=int, help="Optional base LoRA rank.")
    weights_add.add_argument("--lora-base-alpha", type=int, help="Optional base LoRA alpha.")
    weights_add.add_argument("--lora-task-r", type=int, help="Optional task LoRA rank.")
    weights_add.add_argument("--lora-task-alpha", type=int, help="Optional task LoRA alpha.")
    weights_add.add_argument("--lora-dropout", type=float, help="Optional LoRA dropout.")
    weights_add.add_argument("--peft-adapter-name", default="", help="Optional PEFT adapter name.")
    weights_add.add_argument("--target-modules", default="", help="Optional compact target module description.")
    weights_add.add_argument(
        "--no-store-copy",
        action="store_true",
        help="Debug only: keep the registry bound to the original path instead of copying into the internal weight repository.",
    )
    weights_delete = weights_subparsers.add_parser("delete", help="Delete a registered weight by weight ID.")
    weights_delete.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    weights_delete.add_argument("--weight-id", required=True, help="Stable weight ID to delete.")
    weights_delete.add_argument(
        "--keep-file",
        action="store_true",
        help="Remove only the registry entry and keep the internal stored weight file.",
    )
    weights_merge = weights_subparsers.add_parser(
        "merge-lsdsam",
        help="Merge a legacy LSDSAM LoRA checkpoint into a no-LoRA inference checkpoint.",
    )
    weights_merge.add_argument("--lora-checkpoint", required=True, help="Legacy LSDSAM LoRA checkpoint path.")
    weights_merge.add_argument("--output", required=True, help="Output merged checkpoint path.")
    weights_merge.add_argument("--legacy-root", required=True, help="Path to LSDSAM_finetune.")
    weights_merge.add_argument("--sam-checkpoint-path", help="Optional SAM-B base checkpoint path.")
    weights_merge.add_argument("--image-size", type=int, default=512, help="LSDSAM image size.")
    weights_merge.add_argument("--device", default="cpu", help="Merge device, usually cpu or cuda.")
    weights_merge.add_argument("--lora-base-r", type=int, default=32)
    weights_merge.add_argument("--lora-base-alpha", type=int, default=64)
    weights_merge.add_argument("--lora-task-r", type=int, default=16)
    weights_merge.add_argument("--lora-task-alpha", type=int, default=32)
    weights_merge.add_argument("--lora-dropout", type=float, default=0.05)
    weights_merge.add_argument("--peft-adapter-name", default="", help="Optional source adapter name; auto-detected when omitted.")

    datasets_parser = subparsers.add_parser("datasets", help="Manage local TAFT/DAFT/TSFT/Evaluation datasets.")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command", metavar="datasets-command")
    datasets_list = datasets_subparsers.add_parser("list", help="List local dataset registry entries.")
    datasets_list.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    datasets_add = datasets_subparsers.add_parser("add", help="Register a local dataset.")
    datasets_add.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    datasets_add.add_argument("--name", required=True, help="Human-readable dataset name.")
    datasets_add.add_argument("--type", choices=DATASET_TYPES, help="Backward-compatible alias for --class.")
    datasets_add.add_argument("--class", dest="dataset_class", choices=DATASET_CLASSES, help="Dataset class: TAFT, DAFT, TSFT, or Evaluation.")
    datasets_add.add_argument("--imagery", required=True, help="Imagery path or image-chip directory.")
    datasets_add.add_argument("--label", help="Label or ground-truth raster/vector path.")
    datasets_add.add_argument("--truth", help="Backward-compatible alias for --label.")
    datasets_add.add_argument("--test-imagery", help="Deprecated. Register labeled test data as an Evaluation dataset.")
    datasets_add.add_argument("--test-label", help="Deprecated. Register labeled test data as an Evaluation dataset.")
    datasets_add.add_argument("--has-gt", action="store_true", help="Mark this dataset as having ground truth labels.")
    datasets_add.add_argument(
        "--rgb-bands",
        dest="RGB_bands",
        required=True,
        help="Model RGB source bands, e.g. 'RGB_bands: 1,2,3', '3,2,1', '4,3,2', or '6,4,2'.",
    )
    datasets_add.add_argument("--label-polarity", choices=LABEL_POLARITIES, default=LABEL_POLARITY_FOREGROUND_WHITE, help="Raster label foreground convention for TAFT/TSFT/Evaluation standardization.")
    datasets_add.add_argument("--tile-schema", default="unverified", help="Tile schema, e.g. tile_size=512;stride=256.")
    datasets_add.add_argument("--source", default="user", help="Dataset provenance/source.")
    datasets_add.add_argument("--description", default="", help="Optional dataset notes.")
    datasets_add.add_argument("--version", help="Standardized dataset version folder.")
    datasets_add.add_argument("--subset", help="TAFT/TSFT/Evaluation subset folder name, usually a time stamp.")
    datasets_add.add_argument("--upload-method", choices=DAFT_UPLOAD_METHODS, default=DAFT_UPLOAD_METHOD_PACKAGED)
    datasets_add.add_argument("--tile-size", type=int, help="DAFT Tiff Slicing window size.")
    datasets_add.add_argument("--stride", type=int, help="DAFT Tiff Slicing stride.")
    datasets_add.add_argument(
        "--standardize",
        action="store_true",
        help="Upload data into the standardized dataset repository before registering it.",
    )
    datasets_add.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing standardized dataset target.",
    )
    datasets_summary = datasets_subparsers.add_parser(
        "summary",
        help="List standardized dataset versions/subsets with sample counts.",
    )
    datasets_summary.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    datasets_import_taft_tree = datasets_subparsers.add_parser(
        "import-taft-tree",
        help="Import a hierarchical TAFT dataset tree as one version with leaf subsets.",
    )
    datasets_import_taft_tree.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")
    datasets_import_taft_tree.add_argument("--name", help="Deprecated alias; version is the dataset identifier.")
    datasets_import_taft_tree.add_argument("--version", required=True, help="TAFT version folder, e.g. Landslide77K.")
    datasets_import_taft_tree.add_argument("--root", required=True, help="Hierarchical source tree root.")
    datasets_import_taft_tree.add_argument("--preset", default="landslide77k", choices=("landslide77k",), help=argparse.SUPPRESS)
    datasets_import_taft_tree.add_argument("--standardize", action="store_true", help="Create a standardized float32 copy.")
    datasets_import_taft_tree.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing matching TAFT import; valid files are skipped and partial files are rebuilt.",
    )
    datasets_import_taft_tree.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete an existing TAFT version folder and rebuild it from scratch.",
    )
    datasets_import_taft_tree.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    datasets_import_taft_tree.add_argument("--append-to-existing", action="store_true", help="Add newly discovered leaf subsets to an existing TAFT version.")
    datasets_import_taft_tree.add_argument("--dry-run", action="store_true", help="Discover subsets and validate pairing without writing.")
    datasets_import_taft_tree.add_argument("--subset-overrides-json", help="JSON file containing confirmed per-subset RGB_bands/label_polarity overrides.")
    datasets_import_taft_tree.add_argument("--notes", default="", help="Optional dataset notes.")

    repair_parser = subparsers.add_parser("repair-paths", help="Rewrite internal registry paths to portable tokens.")
    repair_parser.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root; registered assets are stored under <root>/registry, separate from disposable outputs/.")

    finetune_parser = subparsers.add_parser("finetune", help="Plan TAFT/DAFT/TSFT fine-tuning workflows.")
    finetune_parser.add_argument("--config", required=True, help="Path to a finetune config file.")
    finetune_parser.add_argument("--output-dir", help="Override the output directory.")
    finetune_parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        help="Model family to fine-tune.",
    )
    finetune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wiring without launching training.",
    )

    daft_parser = subparsers.add_parser("daft", help="Run or inspect DAFT training workflows.")
    daft_subparsers = daft_parser.add_subparsers(dest="daft_command", metavar="daft-command")
    daft_train = daft_subparsers.add_parser("train", help="Run Web GUI-equivalent DAFT training and register the output weight.")
    daft_train.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root.")
    daft_train.add_argument("--model-id", choices=sorted(MODEL_SPECS), default="LSDFormer", help="Training model ID.")
    daft_train.add_argument("--base-weight-id", required=True, help="Registered base weight ID.")
    daft_train.add_argument("--source-dataset-id", required=True, help="Registered labeled TAFT/TSFT source dataset ID.")
    daft_train.add_argument("--target-dataset-id", required=True, help="Registered DAFT/TSFT target dataset ID.")
    daft_train.add_argument("--evaluation-dataset-id", default="", help="Optional registered Evaluation dataset ID for best-checkpoint selection.")
    daft_train.add_argument("--alignment-mode", choices=DAFT_ALIGNMENT_MODES, default="IST", help="Domain alignment constraint.")
    daft_train.add_argument("--epochs", dest="total_epochs", type=int, default=10, help="Total DAFT epochs.")
    daft_train.add_argument("--initial-lr", type=float, default=2e-4, help="Initial segmentation learning rate.")
    daft_train.add_argument("--halve-epochs", default=",".join(str(value) for value in DEFAULT_HALVE_EPOCHS), help="Comma-separated 1-based LR halving epochs.")
    daft_train.add_argument("--output-dir", default="outputs/daft_training", help="Training output directory.")
    daft_train.add_argument("--output-weight-id", default="", help="Registered output weight ID; defaults to a timestamped ID.")
    daft_train.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    daft_train.add_argument("--num-workers", type=int, default=DEFAULT_DAFT_NUM_WORKERS, help="DataLoader workers.")
    daft_train.add_argument("--device", default="auto", help="Training device; auto uses GPU first.")
    daft_train.add_argument("--adv-lambda", type=float, default=0.05, help="Adversarial loss weight.")
    daft_train.add_argument("--align-lambda", type=float, default=0.005, help="Alignment loss weight.")
    daft_train.add_argument("--lora-task-r", type=int, default=16, help="Task LoRA rank.")
    daft_train.add_argument("--lora-task-alpha", type=int, default=32, help="Task LoRA alpha.")
    daft_train.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    daft_train.add_argument("--global-seed", type=int, default=DEFAULT_DAFT_GLOBAL_SEED, help="Global random seed.")
    daft_train.add_argument("--resume", action="store_true", help="Resume from --resume-checkpoint-path.")
    daft_train.add_argument("--resume-checkpoint-path", default="", help="Checkpoint to resume from.")
    daft_train.add_argument("--save-optimizer-state", action="store_true", help="Save optimizer state in checkpoints.")
    daft_train.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned memory.")
    daft_train.add_argument("--deterministic-cuda", action="store_true", help="Use deterministic CUDA settings where available.")
    daft_train.add_argument("--dry-run", action="store_true", help="Validate and print the DAFT payload without launching training.")

    tsft_parser = subparsers.add_parser("tsft", help="Run or inspect TSFT training workflows.")
    tsft_subparsers = tsft_parser.add_subparsers(dest="tsft_command", metavar="tsft-command")
    tsft_train = tsft_subparsers.add_parser("train", help="Run Web GUI-equivalent TSFT training and register the output weight.")
    tsft_train.add_argument("--registry-root", default=DEFAULT_REGISTRY_ROOT, help="Asset workspace root.")
    tsft_train.add_argument("--model-id", choices=sorted(MODEL_SPECS), default="LSDFormer", help="Training model ID.")
    tsft_train.add_argument("--base-weight-id", required=True, help="Registered base weight ID.")
    tsft_train.add_argument("--dataset-id", required=True, help="Registered TSFT dataset ID.")
    tsft_train.add_argument("--evaluation-dataset-id", default="", help="Optional registered Evaluation dataset ID.")
    tsft_train.add_argument("--max-epoch", type=int, default=10, help="Maximum TSFT epochs.")
    tsft_train.add_argument("--max-lr", type=float, default=2e-4, help="Maximum learning rate.")
    tsft_train.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    tsft_train.add_argument("--max-decay-epoch", type=int, help="Epoch at which cosine decay reaches min LR; defaults to --max-epoch.")
    tsft_train.add_argument("--output-dir", default="outputs/tsft_training", help="Training output directory.")
    tsft_train.add_argument("--output-weight-id", default="", help="Registered output weight ID; defaults to a timestamped ID.")
    tsft_train.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    tsft_train.add_argument("--num-workers", type=int, default=DEFAULT_DAFT_NUM_WORKERS, help="DataLoader workers.")
    tsft_train.add_argument("--device", default="auto", help="Training device; auto uses GPU first.")
    tsft_train.add_argument("--lora-task-r", type=int, default=16, help="Task LoRA rank.")
    tsft_train.add_argument("--lora-task-alpha", type=int, default=32, help="Task LoRA alpha.")
    tsft_train.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    tsft_train.add_argument("--resume", action="store_true", help="Resume from --resume-checkpoint-path.")
    tsft_train.add_argument("--resume-checkpoint-path", default="", help="Checkpoint to resume from.")
    tsft_train.add_argument("--keep-lora-checkpoints", action="store_true", help="Keep intermediate LoRA checkpoints after registering the merged weight.")
    tsft_train.add_argument("--save-optimizer-state", action="store_true", help="Save optimizer state in checkpoints.")
    tsft_train.add_argument("--dry-run", action="store_true", help="Validate and print the TSFT payload without launching training.")

    return parser


def _load_subset_overrides_json(path: str | None) -> Mapping[str, Mapping[str, Any]] | None:
    if path in {None, ""}:
        return None
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Failed to read TAFT subset override JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RegistryError("TAFT subset override JSON must be an object keyed by subset path.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-style status code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.package_version:
        from lsd_app import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "config":
        try:
            config = load_config(args.config, create_output_dirs=not args.no_create_dirs)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "models":
        if args.models_command != "list":
            parser.error("models requires a subcommand: list")
        print(json.dumps(list_models(project_root=args.project_root), indent=2, ensure_ascii=False))
        return 0

    if args.command == "weights":
        if args.weights_command == "list":
            print(json.dumps(list_weights(registry_root=args.registry_root), indent=2, ensure_ascii=False))
            return 0
        if args.weights_command == "add":
            try:
                entry = register_weight(
                    registry_root=args.registry_root,
                    model_id=args.model_id,
                    stage=args.stage,
                    version=args.weight_version,
                    path=args.path,
                    weight_id=args.weight_id,
                    metrics_summary=args.metrics_summary,
                    notes=args.notes,
                    weight_type=args.weight_type,
                    base_checkpoint_path=args.base_checkpoint_path,
                    adapter_profile=args.adapter_profile,
                    adapter_name=args.adapter_name,
                    image_size=args.image_size,
                    input_mode=args.input_mode,
                    channel_rule=args.channel_rule,
                    lora_base_r=args.lora_base_r,
                    lora_base_alpha=args.lora_base_alpha,
                    lora_task_r=args.lora_task_r,
                    lora_task_alpha=args.lora_task_alpha,
                    lora_dropout=args.lora_dropout,
                    peft_adapter_name=args.peft_adapter_name,
                    target_modules=args.target_modules,
                    store_copy=not args.no_store_copy,
                )
            except RegistryError as exc:
                parser.error(f"Weight registry failed: {exc}")
            print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if args.weights_command == "delete":
            try:
                result = delete_weight(
                    registry_root=args.registry_root,
                    weight_id=args.weight_id,
                    delete_stored_file=not args.keep_file,
                )
            except RegistryError as exc:
                parser.error(f"Weight delete failed: {exc}")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.weights_command == "merge-lsdsam":
            from lsd_app.models.lsdsam_merge import merge_lsdsam_lora_checkpoint

            try:
                merged_path = merge_lsdsam_lora_checkpoint(
                    lora_checkpoint_path=args.lora_checkpoint,
                    output_path=args.output,
                    legacy_root=args.legacy_root,
                    sam_checkpoint_path=args.sam_checkpoint_path,
                    image_size=args.image_size,
                    device=args.device,
                    lora_base_r=args.lora_base_r,
                    lora_base_alpha=args.lora_base_alpha,
                    lora_task_r=args.lora_task_r,
                    lora_task_alpha=args.lora_task_alpha,
                    lora_dropout=args.lora_dropout,
                    peft_adapter_name=args.peft_adapter_name,
                )
            except ModelAdapterError as exc:
                parser.error(f"LSDSAM merge failed: {exc}")
            print(json.dumps({"merged_checkpoint": str(merged_path)}, indent=2, ensure_ascii=False))
            return 0
        parser.error("weights requires a subcommand: list, add, delete, or merge-lsdsam")

    if args.command == "datasets":
        if args.datasets_command == "list":
            print(json.dumps(list_datasets(registry_root=args.registry_root), indent=2, ensure_ascii=False))
            return 0
        if args.datasets_command == "summary":
            print(json.dumps(list_dataset_repository_summary(registry_root=args.registry_root), indent=2, ensure_ascii=False))
            return 0
        if args.datasets_command == "import-taft-tree":
            try:
                if args.dry_run:
                    discovery = discover_taft_tree(root=args.root, preset=args.preset)
                    print(json.dumps(discovery, indent=2, ensure_ascii=False))
                    return 0
                subset_overrides = _load_subset_overrides_json(args.subset_overrides_json)
                entry = register_taft_tree(
                    registry_root=args.registry_root,
                    name=args.name,
                    version=args.version,
                    root=args.root,
                    preset=args.preset,
                    standardize=args.standardize,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    force_rebuild=args.force_rebuild,
                    append_to_existing=args.append_to_existing,
                    notes=args.notes,
                    subset_overrides=subset_overrides,
                )
            except RegistryError as exc:
                parser.error(f"TAFT tree import failed: {exc}")
            print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if args.datasets_command == "add":
            try:
                entry = register_dataset(
                    registry_root=args.registry_root,
                    name=args.name,
                    dataset_class=args.dataset_class or args.type,
                    imagery_path=args.imagery,
                    label_path=args.label or args.truth,
                    test_imagery_path=args.test_imagery,
                    test_label_path=args.test_label,
                    has_gt=args.has_gt or bool(args.label or args.truth),
                    RGB_bands=args.RGB_bands,
                    label_polarity=args.label_polarity,
                    tile_schema=args.tile_schema,
                    source=args.source,
                    notes=args.description,
                    version=args.version,
                    subset_name=args.subset,
                    standardize=args.standardize,
                    overwrite=args.overwrite,
                    upload_method=args.upload_method,
                    tile_size=args.tile_size,
                    stride=args.stride,
                )
            except RegistryError as exc:
                parser.error(f"Dataset registry failed: {exc}")
            print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))
            return 0
        parser.error("datasets requires a subcommand: list, summary, add, or import-taft-tree")

    if args.command == "repair-paths":
        try:
            result = repair_registry_paths(registry_root=args.registry_root)
        except RegistryError as exc:
            parser.error(f"Path repair failed: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "deploy":
        try:
            result = _run_registered_deployment_cli(args)
        except (ConfigError, RegistryError, RuntimeError, ValueError) as exc:
            parser.error(f"Deployment failed: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "evaluate-dataset":
        try:
            result = _run_registered_evaluation_cli(args)
        except (EvaluationError, RegistryError, RuntimeError, ValueError) as exc:
            parser.error(f"Evaluation failed: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "daft":
        if args.daft_command != "train":
            parser.error("daft requires a subcommand: train")
        try:
            result = _run_daft_training_cli(args)
        except (DaftTrainingError, RegistryError, RuntimeError, ValueError) as exc:
            parser.error(f"DAFT training failed: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "tsft":
        if args.tsft_command != "train":
            parser.error("tsft requires a subcommand: train")
        try:
            result = _run_tsft_training_cli(args)
        except (TsftTrainingError, RegistryError, RuntimeError, ValueError) as exc:
            parser.error(f"TSFT training failed: {exc}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "run":
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        if args.dry_run:
            print(f"[dry-run] run: model={model_spec.name} output={config.output_dir}")
            print("pipeline: " + " -> ".join(PIPELINE_STAGES))
            print(f"input: {config.input_raster_path}")
            print("evaluation: optional; run `lsd evaluate --config ...` with truth/baseline data")
            return 0

        try:
            _run_pipeline(config, model_name=model_spec.name)
        except PipelineRunError as exc:
            parser.error(_format_pipeline_error(exc))
        return 0

    if args.command == "finetune":
        try:
            training_config = load_training_config(args.config, create_output_dirs=False)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")
        if args.output_dir is not None or args.model is not None:
            training_config = _load_training_config_with_overrides(args)
        try:
            plan = build_training_plan(training_config)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        mode = "dry-run" if args.dry_run else "placeholder"
        model_name = str(training_config.model.get("name") or args.model or "configured-model")
        print(f"[{mode}] finetune: model={model_name} output={training_config.output_dir}")
        print("training stage is configured for TAFT/DAFT/TSFT planning; no training has been launched.")
        print(f"config: {Path(args.config).expanduser().resolve()}")
        print("planned jobs: " + ", ".join(describe_training_jobs()))
        print(plan.dry_run_text())
        return 0

    if args.command == "prepare" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        if args.dry_run:
            print(f"[dry-run] prepare: model={model_spec.name} output={config.output_dir}")
            print(
                "prepare stage is configured for raster tiling: "
                f"input={config.input_raster_path} tile_size={config.tile_size} stride={config.stride}"
            )
            return 0

        try:
            result = prepare_raster(config)
        except RasterError as exc:
            parser.error(f"Prepare failed: {exc}")

        print(f"[prepare] model={model_spec.name} output={config.output_dir}")
        print(
            f"saved={result.saved_patches} skipped_nodata={result.skipped_nodata} "
            f"skipped_all_zero={result.skipped_all_zero} total_windows={result.total_windows}"
        )
        print(f"patches: {result.patch_dir}")
        print(f"index: {result.index_path}")
        return 0

    if args.command == "infer" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        if args.dry_run:
            print(f"[dry-run] infer: model={model_spec.name} adapter={model_spec.adapter_name} output={config.output_dir}")
            print(
                "infer stage is configured for patch prediction: "
                f"patches={config.outputs.patches} predictions={config.outputs.patch_predictions} "
                f"batch_size={config.batch_size} num_workers={config.num_workers} "
                f"device={config.device} overwrite={config.overwrite} threshold={config.threshold}"
            )
            return 0

        try:
            result = run_patch_inference(config)
        except (InferenceError, ModelAdapterError, NotImplementedError, RasterError) as exc:
            parser.error(f"Infer failed: {exc}")

        print(f"[infer] model={model_spec.name} adapter={result.adapter_name} output={config.output_dir}")
        print(
            f"patches={result.patch_count} predictions={result.prediction_count} "
            f"skipped={result.skipped_count}"
        )
        print(f"patch_predictions: {result.prediction_dir}")
        print(f"manifest: {result.manifest_path}")
        return 0

    if args.command == "mosaic" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        if args.dry_run:
            print(f"[dry-run] mosaic: model={model_spec.name} output={config.output_dir}")
            print(
                "mosaic stage is configured for patch blending: "
                f"index={config.output_dir / 'patch_index.geojson'} "
                f"patch_predictions={config.outputs.patch_predictions} rasters={config.outputs.rasters} "
                f"threshold={config.threshold}"
            )
            return 0

        try:
            result = run_mosaic(config)
        except (MosaicError, RasterError) as exc:
            parser.error(f"Mosaic failed: {exc}")

        print(f"[mosaic] model={model_spec.name} output={config.output_dir}")
        print(f"patches={result.patch_count} width={result.width} height={result.height}")
        print(f"probability: {result.probability_path}")
        print(f"mask: {result.mask_path}")
        return 0

    if args.command == "vectorize" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        if args.dry_run:
            print(f"[dry-run] vectorize: model={model_spec.name} output={config.output_dir}")
            print(
                "vectorize stage is configured for mask polygonization: "
                f"mask={config.outputs.rasters / 'mask.tif'} "
                f"probability={config.outputs.rasters / 'probability.tif'} "
                f"vectors={config.outputs.vectors} metrics={config.outputs.metrics} "
                f"min_area_pixels={config.min_area_pixels} "
                f"simplify_tolerance={config.simplify_tolerance} "
                f"connectivity={config.connectivity} "
                f"mode={config.vectorize_mode} "
                f"formats={','.join(config.vector_formats)}"
            )
            return 0

        try:
            result = run_vectorize(config)
        except (VectorizeError, RasterError) as exc:
            parser.error(f"Vectorize failed: {exc}")

        print(f"[vectorize] model={model_spec.name} output={config.output_dir}")
        print(
            f"polygons={result.polygon_count} total_area_m2={result.total_area_m2} "
            f"mean_area_m2={result.mean_area_m2} mean_confidence={result.mean_confidence}"
        )
        print(f"gpkg: {result.gpkg_path}")
        if result.geojson_path:
            print(f"geojson: {result.geojson_path}")
        if result.shapefile_path:
            print(f"shapefile: {result.shapefile_path}")
        print(f"stats_csv: {result.stats_csv_path}")
        print(f"stats_json: {result.stats_json_path}")
        return 0

    if args.command == "evaluate" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        truth = config.truth_mask_path or config.truth_vector_path
        if args.dry_run:
            print(f"[dry-run] evaluate: model={model_spec.name} output={config.output_dir}")
            print(
                "evaluate stage is configured for mask metrics: "
                f"prediction={config.outputs.rasters / 'mask.tif'} "
                f"truth={truth if truth is not None else 'none'} "
                f"metrics={config.outputs.metrics} vectors={config.outputs.vectors}"
            )
            return 0

        try:
            result = run_evaluate(config)
        except (EvaluationError, RasterError) as exc:
            parser.error(f"Evaluate failed: {exc}")

        print(f"[evaluate] model={model_spec.name} output={config.output_dir}")
        if result.skipped:
            print(result.message)
            return 0
        assert result.scores is not None
        print(
            f"iou={result.scores.iou} miou={result.scores.miou} precision={result.scores.precision} "
            f"recall={result.scores.recall} f1={result.scores.f1} kappa={result.scores.kappa}"
        )
        print(
            "confusion_matrix: "
            f"TP={result.scores.true_positive} FP={result.scores.false_positive} "
            f"FN={result.scores.false_negative} TN={result.scores.true_negative}"
        )
        print(f"evaluation_json: {result.evaluation_json_path}")
        print(f"evaluation_csv: {result.evaluation_csv_path}")
        print(f"tp: {result.tp_vector_path}")
        print(f"fp: {result.fp_vector_path}")
        print(f"fn: {result.fn_vector_path}")
        return 0

    if args.command == "visualize" and _should_use_platform_config(args.config):
        try:
            config = _load_platform_config(args, create_output_dirs=not args.dry_run)
        except ConfigError as exc:
            parser.error(f"Invalid config: {exc}")

        model_spec = get_model_spec(config.model_name)
        truth = config.truth_mask_path or config.truth_vector_path
        if args.dry_run:
            print(f"[dry-run] visualize: model={model_spec.name} output={config.output_dir}")
            print(
                "visualize stage is configured for downsampled overlays: "
                f"source={config.input_raster_path} "
                f"prediction={config.outputs.rasters / 'mask.tif'} "
                f"truth={truth if truth is not None else 'none'} "
                f"visualizations={config.outputs.visualizations}"
            )
            return 0

        try:
            result = run_visualize(config)
        except (VisualizationError, RasterError) as exc:
            parser.error(f"Visualize failed: {exc}")

        print(f"[visualize] model={model_spec.name} output={config.output_dir}")
        print(result.message)
        print(f"overlay: {result.overlay_path}")
        print(f"overview: width={result.width} height={result.height} scale={result.scale}")
        return 0

    config = load_project_config(
        args.config,
        output_dir=args.output_dir,
        model=args.model,
    )
    model_spec = get_model_spec(config.model)
    result = run_stage(args.command, config)

    mode = "dry-run" if args.dry_run else "skeleton"
    print(f"[{mode}] {result.stage}: model={model_spec.name} output={result.output_root}")
    print(result.message)
    return 0


def _should_use_platform_config(config_path: str | None) -> bool:
    if config_path is None:
        return False
    suffix = Path(config_path).suffix.lower()
    return suffix in {".yaml", ".yml", ".json"}


def _load_platform_config(args: argparse.Namespace, *, create_output_dirs: bool) -> PlatformConfig:
    raw = deepcopy(dict(load_raw_config(args.config)))
    if args.output_dir is not None:
        output = _ensure_mapping(raw, "output")
        output["dir"] = args.output_dir
    if args.model is not None:
        model = _ensure_mapping(raw, "model")
        model["name"] = args.model
    return normalize_config(
        raw,
        config_path=Path(args.config).expanduser().resolve(),
        create_output_dirs=create_output_dirs,
    )


def _load_training_config_with_overrides(args: argparse.Namespace) -> Any:
    raw = deepcopy(dict(load_raw_config(args.config)))
    training = _ensure_mapping(raw, "training")
    if args.output_dir is not None:
        training["output_dir"] = args.output_dir
    if args.model is not None:
        model = _ensure_mapping(training, "model")
        model["name"] = args.model
    return normalize_training_config(
        raw,
        config_path=Path(args.config).expanduser().resolve(),
        create_output_dirs=False,
    )


def _registered_weight(registry_root: str | Path, weight_id: str) -> dict[str, Any]:
    normalized = str(weight_id or "").strip()
    if not normalized:
        raise RegistryError("weight_id is required.")
    for row in list_weights(registry_root=registry_root):
        if str(row.get("weight_id") or "") == normalized:
            return dict(row)
    raise RegistryError(f"Registered weight not found: {normalized}")


def _registered_dataset(registry_root: str | Path, dataset_id: str) -> dict[str, Any]:
    normalized = str(dataset_id or "").strip()
    if not normalized:
        raise RegistryError("dataset_id is required.")
    for row in list_datasets(registry_root=registry_root):
        if str(row.get("dataset_id") or "") == normalized:
            return dict(row)
    raise RegistryError(f"Registered dataset not found: {normalized}")


def _weight_checkpoint_path(registry_root: str | Path, weight: Mapping[str, Any]) -> str:
    raw_path = str(weight.get("path") or "").strip()
    if not raw_path:
        raise RegistryError(f"Weight {weight.get('weight_id') or ''} does not define a checkpoint path.")
    return str(resolve_user_path(raw_path, workspace_root=registry_root))


def _weight_model_id(weight: Mapping[str, Any], override: str | None = None) -> str:
    value = str(override or weight.get("model_id") or "").strip()
    if not value:
        raise RegistryError(f"Weight {weight.get('weight_id') or ''} does not define a model_id.")
    return value


def _rgb_bands_from_weight(weight: Mapping[str, Any], override: str | None = None) -> str:
    value = str(override or weight.get("RGB_bands") or "").strip() or DEFAULT_RGB_BANDS
    return normalize_rgb_bands(value)


def _model_options_from_weight(weight: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    mapping = {
        "weight_type": "weight_type",
        "base_checkpoint_path": "sam_checkpoint_path",
        "image_size": "image_size",
        "lora_base_r": "lora_base_r",
        "lora_base_alpha": "lora_base_alpha",
        "lora_task_r": "lora_task_r",
        "lora_task_alpha": "lora_task_alpha",
        "lora_dropout": "lora_dropout",
        "peft_adapter_name": "peft_adapter_name",
    }
    for source_key, option_key in mapping.items():
        value = weight.get(source_key)
        if value not in {None, ""}:
            options[option_key] = value
    return options


def _deployment_raw_config_from_args(args: argparse.Namespace, weight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registry_root": str(args.registry_root),
        "input": {"raster_path": str(args.geotiff), "RGB_bands": _rgb_bands_from_weight(weight, args.rgb_bands)},
        "output": {"dir": str(args.output_dir), "vector_formats": ["gpkg", "geojson", "shapefile"]},
        "model": {
            "name": _weight_model_id(weight, args.model),
            "checkpoint_path": _weight_checkpoint_path(args.registry_root, weight),
            "options": _model_options_from_weight(weight),
        },
        "tiling": {"tile_size": int(args.tile_size), "stride": int(args.stride)},
        "inference": {"batch_size": int(args.batch_size), "threshold": float(args.threshold), "overwrite": False},
        "postprocess": {"min_area_pixels": int(args.min_area_pixels), "simplify_tolerance": 0.0, "connectivity": 8},
        "vectorize": {"mode": "streaming"},
        "runtime": {"device": str(args.device or "auto"), "num_workers": 0},
        "geospatial": {
            "crs": {"preserve_input": True, "target": None, "allow_missing": False},
            "nodata": {"input": None, "output": 0, "mask": True},
        },
        "truth": {"vector_path": None, "mask_path": None},
    }


def _run_registered_deployment_cli(args: argparse.Namespace) -> dict[str, Any]:
    from lsd_app.web.app import (
        clear_deployment_cache,
        deployment_config_fingerprint,
        output_cache_status,
        preflight_deployment_config,
        run_deployment_background_task,
    )

    weight = _registered_weight(args.registry_root, args.weight_id)
    raw_config = _deployment_raw_config_from_args(args, weight)
    config, preflight = preflight_deployment_config(raw_config, selected_weight=weight, create_output_dirs=False)
    if args.dry_run:
        return {
            "mode": "dry-run",
            "ok": preflight.ok,
            "errors": list(preflight.errors),
            "warnings": list(preflight.warnings),
            "raw_config": raw_config,
        }
    if config is None or not preflight.ok:
        raise ConfigError("; ".join(preflight.errors) or "Deployment preflight failed.")

    fingerprint = deployment_config_fingerprint(raw_config)
    cache_status = output_cache_status(config.output_dir, fingerprint)
    if cache_status.get("has_cache") and not cache_status.get("matches"):
        if args.clear_cache:
            removed = clear_deployment_cache(config.output_dir)
            print(f"[deploy] cleared stale cache: {', '.join(removed) if removed else 'nothing'}")
        else:
            raise RuntimeError(
                "Output directory contains a different deployment cache. "
                "Use --clear-cache or choose another --output-dir."
            )

    config, preflight = preflight_deployment_config(raw_config, selected_weight=weight, create_output_dirs=True)
    if config is None or not preflight.ok:
        raise ConfigError("; ".join(preflight.errors) or "Deployment preflight failed.")

    print(f"[deploy] weight={args.weight_id} model={config.model_name} output={config.output_dir}")
    return run_deployment_background_task(
        config,
        raw_config=raw_config,
        progress_callback=_print_progress("deploy"),
    )


def _run_registered_evaluation_cli(args: argparse.Namespace) -> dict[str, Any]:
    from lsd_app.web.app import build_evaluation_dataset_payload, run_evaluation_background_task

    dataset = _registered_dataset(args.registry_root, args.dataset_id)
    if str(dataset.get("dataset_class") or "") != "Evaluation":
        raise EvaluationError(f"Dataset {args.dataset_id} is {dataset.get('dataset_class')}, expected Evaluation.")
    weight = _registered_weight(args.registry_root, args.weight_id)
    checkpoint_path = str(args.checkpoint_path or "").strip() or _weight_checkpoint_path(args.registry_root, weight)
    payload = build_evaluation_dataset_payload(
        dataset_id=args.dataset_id,
        weight_id=args.weight_id,
        model_id=_weight_model_id(weight, args.model),
        output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
        batch_size=int(args.batch_size),
        device=str(args.device or "auto"),
    )
    payload = {**payload, "registry_root": str(args.registry_root)}
    if args.dry_run:
        return {"mode": "dry-run", "dataset": dataset, "weight": weight, "payload": payload}

    print(f"[evaluate-dataset] dataset={args.dataset_id} weight={args.weight_id} output={args.output_dir}")
    return run_evaluation_background_task(
        raw_config=payload,
        progress_callback=_print_progress("evaluate"),
    )


def _run_daft_training_cli(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_daft_training_payload(
        registry_root=str(args.registry_root),
        model_id=args.model_id,
        base_weight_id=args.base_weight_id,
        source_dataset_id=args.source_dataset_id,
        target_dataset_id=args.target_dataset_id,
        evaluation_dataset_id=args.evaluation_dataset_id,
        alignment_mode=args.alignment_mode,
        total_epochs=int(args.total_epochs),
        initial_lr=float(args.initial_lr),
        halve_epochs=args.halve_epochs,
        device=str(args.device or "auto"),
        output_dir=args.output_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        adv_lambda=float(args.adv_lambda),
        align_lambda=float(args.align_lambda),
        lora_task_r=int(args.lora_task_r),
        lora_task_alpha=int(args.lora_task_alpha),
        lora_dropout=float(args.lora_dropout),
        output_weight_id=args.output_weight_id,
        resume=bool(args.resume),
        resume_checkpoint_path=args.resume_checkpoint_path,
        save_optimizer_state=bool(args.save_optimizer_state),
        pin_memory=bool(args.pin_memory),
        deterministic_cuda=bool(args.deterministic_cuda),
        global_seed=int(args.global_seed),
    )
    if args.dry_run:
        return {"mode": "dry-run", "payload": payload}
    config = _daft_config_from_payload(payload)
    print(f"[daft] model={config.model_id} source={config.source_dataset_id} target={config.target_dataset_id} output={config.output_dir}")
    return run_lsdsam_daft(config, progress_callback=_print_progress("daft")).to_dict()


def _run_tsft_training_cli(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_tsft_training_payload(
        registry_root=str(args.registry_root),
        model_id=args.model_id,
        base_weight_id=args.base_weight_id,
        dataset_id=args.dataset_id,
        evaluation_dataset_id=args.evaluation_dataset_id,
        max_epoch=int(args.max_epoch),
        max_lr=float(args.max_lr),
        min_lr=float(args.min_lr),
        max_decay_epoch=int(args.max_decay_epoch or args.max_epoch),
        device=str(args.device or "auto"),
        output_dir=args.output_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lora_task_r=int(args.lora_task_r),
        lora_task_alpha=int(args.lora_task_alpha),
        lora_dropout=float(args.lora_dropout),
        output_weight_id=args.output_weight_id,
        resume=bool(args.resume),
        resume_checkpoint_path=args.resume_checkpoint_path,
        cleanup_lora_checkpoints_after_register=not bool(args.keep_lora_checkpoints),
        save_optimizer_state=bool(args.save_optimizer_state),
    )
    if args.dry_run:
        return {"mode": "dry-run", "device_policy": TSFT_TRAINING_DEVICE_POLICY, "payload": payload}
    config = _tsft_config_from_payload(payload)
    print(f"[tsft] model={config.model_id} dataset={config.dataset_id} output={config.output_dir}")
    return run_lsdsam_tsft(config, progress_callback=_print_progress("tsft")).to_dict()


def _daft_config_from_payload(payload: Mapping[str, Any]) -> DaftLSDSAMConfig:
    return DaftLSDSAMConfig(
        registry_root=str(payload.get("registry_root") or DEFAULT_REGISTRY_ROOT),
        model_id=str(payload.get("model_id") or "LSDFormer"),
        base_weight_id=str(payload.get("base_weight_id") or ""),
        source_dataset_id=str(payload.get("source_dataset_id") or ""),
        target_dataset_id=str(payload.get("target_dataset_id") or ""),
        alignment_mode=str(payload.get("alignment_mode") or "IST"),
        total_epochs=int(payload.get("total_epochs") or 10),
        initial_lr=float(payload.get("initial_lr") or 2e-4),
        halve_epochs=tuple(int(value) for value in payload.get("halve_epochs", [])),
        device=str(payload.get("device") or "auto"),
        output_dir=str(payload.get("output_dir") or "outputs/daft_training"),
        evaluation_dataset_id=str(payload.get("evaluation_dataset_id") or ""),
        batch_size=int(payload.get("batch_size") or 2),
        num_workers=int(payload.get("num_workers", DEFAULT_DAFT_NUM_WORKERS)),
        seed=int(payload.get("global_seed", payload.get("seed", DEFAULT_DAFT_GLOBAL_SEED))),
        adv_lambda=float(payload.get("adv_lambda") or 0.05),
        align_lambda=float(payload.get("align_lambda") or 0.005),
        lora_task_r=int(payload.get("lora_task_r") or 16),
        lora_task_alpha=int(payload.get("lora_task_alpha") or 32),
        lora_dropout=float(payload.get("lora_dropout") if payload.get("lora_dropout") is not None else 0.05),
        output_weight_id=str(payload.get("output_weight_id") or ""),
        resume=bool(payload.get("resume")),
        resume_checkpoint_path=str(payload.get("resume_checkpoint_path") or ""),
        save_optimizer_state=bool(payload.get("save_optimizer_state")),
        pin_memory=bool(payload.get("pin_memory")),
        deterministic_cuda=bool(payload.get("deterministic_cuda")),
    )


def _tsft_config_from_payload(payload: Mapping[str, Any]) -> TsftLSDSAMConfig:
    max_epoch = int(payload.get("max_epoch") or payload.get("total_epochs") or 10)
    max_lr = float(payload.get("max_lr") or payload.get("initial_lr") or 2e-4)
    min_lr = float(payload.get("min_lr") if payload.get("min_lr") is not None else 1e-6)
    return TsftLSDSAMConfig(
        registry_root=str(payload.get("registry_root") or DEFAULT_REGISTRY_ROOT),
        model_id=str(payload.get("model_id") or "LSDFormer"),
        base_weight_id=str(payload.get("base_weight_id") or ""),
        dataset_id=str(payload.get("dataset_id") or ""),
        evaluation_dataset_id=str(payload.get("evaluation_dataset_id") or ""),
        max_epoch=max_epoch,
        max_lr=max_lr,
        min_lr=min_lr,
        max_decay_epoch=int(payload.get("max_decay_epoch") or max_epoch),
        device=str(payload.get("device") or "auto"),
        output_dir=str(payload.get("output_dir") or "outputs/tsft_training"),
        batch_size=int(payload.get("batch_size") or 2),
        num_workers=int(payload.get("num_workers", DEFAULT_DAFT_NUM_WORKERS)),
        lora_task_r=int(payload.get("lora_task_r") or 16),
        lora_task_alpha=int(payload.get("lora_task_alpha") or 32),
        lora_dropout=float(payload.get("lora_dropout") if payload.get("lora_dropout") is not None else 0.05),
        output_weight_id=str(payload.get("output_weight_id") or ""),
        resume=bool(payload.get("resume")),
        resume_checkpoint_path=str(payload.get("resume_checkpoint_path") or ""),
        cleanup_lora_checkpoints_after_register=payload.get("cleanup_lora_checkpoints_after_register", True) is not False,
        save_optimizer_state=bool(payload.get("save_optimizer_state")),
    )


def _print_progress(prefix: str):
    def callback(progress: Mapping[str, Any]) -> None:
        percent = progress.get("overall_percent", progress.get("percent"))
        phase = progress.get("phase") or progress.get("stage") or progress.get("progress_kind") or "progress"
        message = progress.get("message") or progress.get("current_action") or ""
        if percent is None:
            print(f"[{prefix}] {phase}: {message}")
            return
        try:
            value = float(percent) * 100.0
        except (TypeError, ValueError):
            print(f"[{prefix}] {phase}: {message}")
            return
        print(f"[{prefix}] {phase} {value:.1f}% {message}")

    return callback


def _load_prepare_config(args: argparse.Namespace, *, create_output_dirs: bool) -> PlatformConfig:
    return _load_platform_config(args, create_output_dirs=create_output_dirs)


def _ensure_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if isinstance(value, Mapping):
        copied = dict(value)
        raw[key] = copied
        return copied
    copied: dict[str, Any] = {}
    raw[key] = copied
    return copied


def _model_name_from_raw_config(config_path: Path) -> str | None:
    if config_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        return None
    try:
        raw = load_raw_config(config_path)
    except ConfigError:
        return None
    value = raw.get("model") if isinstance(raw, Mapping) else None
    if isinstance(value, Mapping):
        name = value.get("name")
        return str(name).strip() if name else None
    if isinstance(value, str):
        return value.strip()
    return None


def _run_pipeline(config: PlatformConfig, *, model_name: str) -> None:
    print(f"[run] model={model_name} output={config.output_dir}")
    print("[run] pipeline: " + " -> ".join(PIPELINE_STAGES))
    for stage in PIPELINE_STAGES:
        print(f"[run] starting {stage}")
        try:
            summary = _execute_pipeline_stage(stage, config)
        except (
            RasterError,
            InferenceError,
            ModelAdapterError,
            MosaicError,
            VectorizeError,
            EvaluationError,
            VisualizationError,
            NotImplementedError,
        ) as exc:
            raise PipelineRunError(stage, exc, _stage_check_paths(stage, config)) from exc
        print(f"[run] completed {stage}: {summary}")
    print("[run] pipeline completed")
    polygons_path = config.outputs.vectors / "landslides.gpkg"
    if polygons_path.exists():
        run_id = config.output_dir.name or "prediction-run"
        register_prediction_run(
            registry_root=config.output_dir.parent,
            run_id=run_id,
            model_id=model_name,
            input_raster_path=config.input_raster_path,
            output_dir=config.output_dir,
            polygons_path=polygons_path,
        )
        print(f"[run] prediction registered: {run_id}")


def _execute_pipeline_stage(stage: str, config: PlatformConfig) -> str:
    if stage == "prepare":
        result = prepare_raster(config)
        return (
            f"saved={result.saved_patches} skipped_nodata={result.skipped_nodata} "
            f"skipped_all_zero={result.skipped_all_zero} index={result.index_path}"
        )
    if stage == "infer":
        result = run_patch_inference(config)
        return (
            f"patches={result.patch_count} predictions={result.prediction_count} "
            f"skipped={result.skipped_count} manifest={result.manifest_path}"
        )
    if stage == "mosaic":
        result = run_mosaic(config)
        return f"patches={result.patch_count} probability={result.probability_path} mask={result.mask_path}"
    if stage == "vectorize":
        result = run_vectorize(config)
        return (
            f"polygons={result.polygon_count} gpkg={result.gpkg_path} "
            f"stats={result.stats_json_path}"
        )
    if stage == "evaluate":
        result = run_evaluate(config)
        if result.skipped:
            return result.message
        assert result.scores is not None
        return (
            f"iou={result.scores.iou} f1={result.scores.f1} "
            f"metrics={result.evaluation_json_path}"
        )
    if stage == "visualize":
        result = run_visualize(config)
        return f"overlay={result.overlay_path} width={result.width} height={result.height}"
    raise ValueError(f"Unknown pipeline stage: {stage}")


def _stage_check_paths(stage: str, config: PlatformConfig) -> list[Path]:
    values: dict[str, list[Path]] = {
        "prepare": [config.input_raster_path, config.outputs.patches, config.output_dir / "patch_index.geojson"],
        "infer": [config.outputs.patches, config.output_dir / "patch_index.geojson", config.outputs.patch_predictions],
        "mosaic": [config.output_dir / "patch_index.geojson", config.outputs.patch_predictions, config.outputs.rasters],
        "vectorize": [config.outputs.rasters / "mask.tif", config.outputs.rasters / "probability.tif", config.outputs.vectors],
        "evaluate": [
            config.outputs.rasters / "mask.tif",
            config.truth_mask_path or config.truth_vector_path or config.output_dir,
            config.outputs.metrics,
        ],
        "visualize": [config.input_raster_path, config.outputs.rasters / "mask.tif", config.outputs.visualizations],
    }
    return values.get(stage, [config.output_dir])


def _format_pipeline_error(error: PipelineRunError) -> str:
    paths = ", ".join(str(path) for path in error.check_paths)
    return f"Run failed at stage '{error.stage}': {error.reason}. Check: {paths}"


if __name__ == "__main__":
    raise SystemExit(main())
