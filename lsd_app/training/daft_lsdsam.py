"""LSDSAM DAFT domain-adaptation training.

The module keeps heavyweight ML imports inside runtime helpers so the regular
CLI, web app, and dry-run tests can import ``lsd_app.training`` without loading
torch or legacy LSDSAM code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import uuid
from typing import Any, Callable, Mapping, Sequence

from lsd_app.core.channels import DEFAULT_RGB_BANDS, extract_rgb_bands, normalize_rgb_bands
from lsd_app.core.paths import resolve_user_path
from lsd_app.models.registry import get_model_spec, normalize_model_name
from lsd_app.registry import list_datasets, list_weights, register_weight


ALIGNMENT_MODES = ("IST", "MMD", "CoRAL")
DAFT_TRAINING_DEVICE_POLICY = "auto-gpu-then-cpu"
DEFAULT_HALVE_EPOCHS = (6, 8)
DEFAULT_DAFT_GLOBAL_SEED = 2025
DEFAULT_DAFT_NUM_WORKERS = 0
DEFAULT_LORA_TARGET_MODULES = "image_encoder.blocks.*.(attn.qkv|attn.proj|mlp.lin1|mlp.lin2)"
LSDSAM_MODEL_IDS = ("LSDSAM-H", "LSDSAM-L", "LSDSAM-B", "lsdsam-fewshot")
LSDFORMER_MODEL_IDS = ("LSDFormer", "lsdformer-fast")
LABEL_POLARITY_FOREGROUND_WHITE = "foreground_white"
LABEL_POLARITY_FOREGROUND_BLACK = "foreground_black"
TRAINING_CONFIG_SNAPSHOT_FILENAME = "training_config_latest.json"

ProgressCallback = Callable[[Mapping[str, Any]], None]


class DaftTrainingError(RuntimeError):
    """Raised when DAFT training cannot be configured or launched."""


@dataclass(frozen=True)
class DaftLSDSAMConfig:
    """Runtime configuration for LSDSAM DAFT training."""

    registry_root: str | Path
    source_dataset_id: str
    target_dataset_id: str
    base_weight_id: str
    output_dir: str | Path
    evaluation_dataset_id: str = ""
    model_id: str = "lsdsam-fewshot"
    alignment_mode: str = "IST"
    total_epochs: int = 10
    initial_lr: float = 2e-4
    halve_epochs: tuple[int, ...] = DEFAULT_HALVE_EPOCHS
    batch_size: int = 2
    num_workers: int = DEFAULT_DAFT_NUM_WORKERS
    device: str = "auto"
    seed: int = DEFAULT_DAFT_GLOBAL_SEED
    image_size: int = 512
    adv_lambda: float = 0.05
    align_lambda: float = 0.005
    discriminator_lr: float | None = None
    min_lr: float = 1e-6
    lora_task_r: int = 16
    lora_task_alpha: int = 32
    lora_dropout: float = 0.05
    peft_adapter_name: str = "Task-Specific"
    output_weight_id: str = ""
    legacy_root: str | Path | None = None
    sam_checkpoint_path: str | Path | None = None
    resume: bool = False
    resume_checkpoint_path: str | Path | None = None
    save_optimizer_state: bool = False
    pin_memory: bool = False
    deterministic_cuda: bool = False

    def normalized(self) -> "DaftLSDSAMConfig":
        return DaftLSDSAMConfig(
            **{
                **asdict(self),
                "registry_root": str(Path(self.registry_root).expanduser().resolve()),
                "output_dir": str(Path(self.output_dir).expanduser().resolve()),
                "alignment_mode": normalize_alignment_mode(self.alignment_mode),
                "evaluation_dataset_id": str(self.evaluation_dataset_id or "").strip(),
                "halve_epochs": parse_halve_epochs(self.halve_epochs),
                "total_epochs": _positive_int(self.total_epochs, "total_epochs"),
                "initial_lr": _positive_float(self.initial_lr, "initial_lr"),
                "batch_size": _positive_int(self.batch_size, "batch_size"),
                "num_workers": _non_negative_int(self.num_workers, "num_workers"),
                "seed": _non_negative_int(self.seed, "seed"),
                "image_size": _positive_int(self.image_size, "image_size"),
                "adv_lambda": _non_negative_float(self.adv_lambda, "adv_lambda"),
                "align_lambda": _non_negative_float(self.align_lambda, "align_lambda"),
                "min_lr": _non_negative_float(self.min_lr, "min_lr"),
                "lora_task_r": _positive_int(self.lora_task_r, "lora_task_r"),
                "lora_task_alpha": _positive_int(self.lora_task_alpha, "lora_task_alpha"),
                "lora_dropout": _non_negative_float(self.lora_dropout, "lora_dropout"),
                "output_weight_id": str(self.output_weight_id or "").strip(),
                "resume": bool(self.resume),
                "resume_checkpoint_path": (
                    str(Path(self.resume_checkpoint_path).expanduser().resolve())
                    if self.resume_checkpoint_path not in {None, ""}
                    else ""
                ),
                "save_optimizer_state": bool(self.save_optimizer_state),
                "pin_memory": bool(self.pin_memory),
                "deterministic_cuda": bool(self.deterministic_cuda),
            }
        )


@dataclass(frozen=True)
class DaftTrainingSample:
    """One standardized training sample resolved from the dataset repository."""

    sample_id: str
    image_path: Path
    label_path: Path | None
    split: str = "train"
    RGB_bands: str = DEFAULT_RGB_BANDS
    label_polarity: str = LABEL_POLARITY_FOREGROUND_WHITE

    @property
    def has_label(self) -> bool:
        return self.label_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": str(self.image_path),
            "label_path": str(self.label_path) if self.label_path is not None else None,
            "split": self.split,
            "RGB_bands": self.RGB_bands,
            "label_polarity": self.label_polarity,
        }


@dataclass(frozen=True)
class DaftDatasetMaterialization:
    """Resolved dataset repository samples for DAFT training."""

    dataset_id: str
    dataset_class: str
    version: str
    root_dir: Path
    manifest_path: Path
    RGB_bands: str
    has_gt: bool
    samples: tuple[DaftTrainingSample, ...] = field(default_factory=tuple)
    source_materializations: tuple["DaftDatasetMaterialization", ...] = field(default_factory=tuple)

    @property
    def is_concat(self) -> bool:
        return bool(self.source_materializations)

    def supervised_train_samples(self) -> tuple[DaftTrainingSample, ...]:
        samples = tuple(
            sample for sample in self.samples if sample.has_label and sample.split.lower() != "test"
        )
        if samples:
            return samples
        return tuple(sample for sample in self.samples if sample.has_label)

    def target_train_samples(self) -> tuple[DaftTrainingSample, ...]:
        samples = tuple(sample for sample in self.samples if sample.split.lower() != "test")
        return samples or self.samples

    def target_eval_samples(self) -> tuple[DaftTrainingSample, ...]:
        return tuple(sample for sample in self.samples if sample.has_label and sample.split.lower() == "test")

    def evaluation_samples(self) -> tuple[DaftTrainingSample, ...]:
        return tuple(sample for sample in self.samples if sample.has_label)


@dataclass(frozen=True)
class DaftLSDSAMResult:
    """Artifacts produced by one LSDSAM DAFT run."""

    output_dir: Path
    lora_checkpoint_path: Path
    merged_checkpoint_path: Path
    registered_weight_id: str
    registered_weight_path: Path
    metrics_path: Path
    best_epoch: int
    best_score: float | None
    selection_metric: str
    source_sample_count: int
    target_sample_count: int
    target_has_gt: bool
    device: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "output_dir",
            "lora_checkpoint_path",
            "merged_checkpoint_path",
            "registered_weight_path",
            "metrics_path",
        ):
            data[key] = str(data[key])
        return data


def normalize_alignment_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    aliases = {"ist": "IST", "mmd": "MMD", "coral": "CoRAL", "coralloss": "CoRAL"}
    if mode in aliases:
        return aliases[mode]
    allowed = ", ".join(ALIGNMENT_MODES)
    raise DaftTrainingError(f"Unsupported alignment mode {value!r}. Expected one of: {allowed}.")


def parse_halve_epochs(value: str | int | Sequence[int] | None) -> tuple[int, ...]:
    """Parse comma-separated or sequence epoch milestones for LR halving."""

    if value is None or value == "":
        return ()
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [int(item) for item in value]
    if any(epoch <= 0 for epoch in values):
        raise DaftTrainingError("halve_epochs must contain positive 1-based epoch numbers.")
    return tuple(sorted(set(values)))


def learning_rate_for_epoch(
    *,
    epoch_number: int,
    initial_lr: float,
    halve_epochs: Sequence[int],
    min_lr: float = 1e-6,
) -> float:
    """Return the 1-based epoch LR after applying configured halvings."""

    epoch = _positive_int(epoch_number, "epoch_number")
    factor = 0.5 ** sum(1 for milestone in parse_halve_epochs(halve_epochs) if epoch >= milestone)
    return max(float(initial_lr) * factor, float(min_lr))


def resolve_training_device(_requested: str, torch_module: Any) -> str:
    """Use GPU when available, otherwise fall back to CPU."""

    return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"


def resolve_daft_dataset_materialization(
    *,
    registry_root: str | Path,
    dataset_id: str,
    _seen_manifests: frozenset[str] | None = None,
) -> DaftDatasetMaterialization:
    """Resolve standardized repository samples from a registered dataset id."""

    dataset = _lookup_dataset(registry_root=registry_root, dataset_id=dataset_id)
    manifest_path = _dataset_manifest_path(dataset)
    return _resolve_manifest_dataset_materialization(
        registry_root=registry_root,
        dataset=dataset,
        manifest_path=manifest_path,
        requested_dataset_id=dataset_id,
        seen_manifests=_seen_manifests or frozenset(),
    )


def _resolve_manifest_dataset_materialization(
    *,
    registry_root: str | Path,
    dataset: Mapping[str, Any],
    manifest_path: Path,
    requested_dataset_id: str,
    seen_manifests: frozenset[str],
) -> DaftDatasetMaterialization:
    manifest_identity = str(manifest_path.resolve())
    if manifest_identity in seen_manifests:
        raise DaftTrainingError(f"Concat dataset manifest cycle detected at {manifest_path}.")
    seen_manifests = seen_manifests | {manifest_identity}

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise DaftTrainingError(f"Dataset manifest is not a JSON object: {manifest_path}")
    root_dir = _resolve_training_path(
        manifest.get("root_dir") or dataset.get("repository_path") or manifest_path.parent,
        registry_root=registry_root,
        base_dir=manifest_path.parent,
    )
    if bool(manifest.get("is_concat")):
        return _resolve_concat_daft_dataset_materialization(
            registry_root=registry_root,
            requested_dataset_id=requested_dataset_id,
            dataset=dataset,
            manifest=manifest,
            manifest_path=manifest_path,
            root_dir=root_dir,
            seen_manifests=seen_manifests,
        )

    RGB_bands = _require_rgb_bands(manifest, dataset=dataset, source=str(manifest_path))
    samples = tuple(_samples_from_manifest(manifest, manifest_path=manifest_path, default_RGB_bands=RGB_bands))
    if not samples:
        samples = tuple(_infer_samples_from_repository(root_dir, manifest=manifest, default_RGB_bands=RGB_bands))
    if not samples:
        samples = tuple(_infer_samples_from_source_tree(manifest, default_RGB_bands=RGB_bands))
    return DaftDatasetMaterialization(
        dataset_id=str(manifest.get("dataset_id") or dataset.get("dataset_id") or requested_dataset_id),
        dataset_class=str(manifest.get("dataset_class") or dataset.get("dataset_class") or ""),
        version=str(manifest.get("version") or dataset.get("version") or ""),
        root_dir=root_dir,
        manifest_path=manifest_path,
        RGB_bands=RGB_bands,
        has_gt=bool(manifest.get("has_gt", dataset.get("has_gt", False))),
        samples=samples,
    )


def _resolve_concat_daft_dataset_materialization(
    *,
    registry_root: str | Path,
    requested_dataset_id: str,
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    root_dir: Path,
    seen_manifests: frozenset[str],
) -> DaftDatasetMaterialization:
    backend = str(manifest.get("concat_backend") or "").strip()
    if backend and backend != "torch.utils.data.ConcatDataset":
        raise DaftTrainingError(
            f"Concat dataset {requested_dataset_id} uses unsupported concat_backend={backend!r}."
        )
    sources_payload = manifest.get("concat_sources")
    if not isinstance(sources_payload, Sequence) or isinstance(sources_payload, (str, bytes)) or not sources_payload:
        raise DaftTrainingError(f"Concat dataset {requested_dataset_id} has no concat_sources.")

    expected_class = str(manifest.get("dataset_class") or dataset.get("dataset_class") or "").strip()
    source_materializations: list[DaftDatasetMaterialization] = []
    for index, source in enumerate(sources_payload):
        materialized = _resolve_concat_source_materialization(
            registry_root=registry_root,
            source=source,
            index=index,
            concat_dataset_id=requested_dataset_id,
            concat_manifest_path=manifest_path,
            seen_manifests=seen_manifests,
        )
        if expected_class and materialized.dataset_class != expected_class:
            raise DaftTrainingError(
                f"Concat dataset {requested_dataset_id} expected source class {expected_class}, "
                f"but source {materialized.dataset_id} is {materialized.dataset_class}."
            )
        source_materializations.append(materialized)

    samples = tuple(sample for source in source_materializations for sample in source.samples)
    RGB_bands = _concat_rgb_bands(manifest, dataset=dataset, sources=source_materializations, source=str(manifest_path))
    return DaftDatasetMaterialization(
        dataset_id=str(manifest.get("dataset_id") or dataset.get("dataset_id") or requested_dataset_id),
        dataset_class=expected_class,
        version=str(manifest.get("version") or dataset.get("version") or ""),
        root_dir=root_dir,
        manifest_path=manifest_path,
        RGB_bands=RGB_bands,
        has_gt=bool(manifest.get("has_gt", dataset.get("has_gt", any(source.has_gt for source in source_materializations)))),
        samples=samples,
        source_materializations=tuple(source_materializations),
    )


def _concat_source_dataset_id(source: Any, *, index: int, concat_dataset_id: str) -> str:
    if isinstance(source, Mapping):
        value = source.get("dataset_id") or source.get("id")
    else:
        value = source
    source_id = str(value or "").strip()
    if not source_id:
        raise DaftTrainingError(f"Concat dataset {concat_dataset_id} source #{index + 1} is missing dataset_id.")
    return source_id


def _resolve_concat_source_materialization(
    *,
    registry_root: str | Path,
    source: Any,
    index: int,
    concat_dataset_id: str,
    concat_manifest_path: Path,
    seen_manifests: frozenset[str],
) -> DaftDatasetMaterialization:
    if isinstance(source, Mapping) and source.get("manifest_path"):
        return _resolve_manifest_dataset_materialization(
            registry_root=registry_root,
            dataset=_dataset_row_from_concat_source(
                source,
                index=index,
                concat_dataset_id=concat_dataset_id,
                registry_root=registry_root,
            ),
            manifest_path=_resolve_training_path(source.get("manifest_path"), registry_root=registry_root, base_dir=concat_manifest_path.parent),
            requested_dataset_id=str(source.get("dataset_id") or source.get("version") or f"{concat_dataset_id}#{index + 1}"),
            seen_manifests=seen_manifests,
        )
    source_id = _concat_source_dataset_id(source, index=index, concat_dataset_id=concat_dataset_id)
    return resolve_daft_dataset_materialization(
        registry_root=registry_root,
        dataset_id=source_id,
        _seen_manifests=seen_manifests,
    )


def _dataset_row_from_concat_source(
    source: Mapping[str, Any],
    *,
    index: int,
    concat_dataset_id: str,
    registry_root: str | Path = "",
) -> Mapping[str, Any]:
    dataset_id = str(source.get("dataset_id") or source.get("version") or f"{concat_dataset_id}#{index + 1}")
    return {
        "dataset_id": dataset_id,
        "dataset_class": source.get("dataset_class") or "",
        "version": source.get("version") or "",
        "repository_path": source.get("root_dir") or "",
        "manifest_path": source.get("manifest_path") or "",
        "sample_count": source.get("sample_count"),
        "has_gt": source.get("has_gt", False),
        "RGB_bands": source.get("RGB_bands") or "",
        "_registry_root": str(registry_root),
    }


def _concat_rgb_bands(
    manifest: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    sources: Sequence[DaftDatasetMaterialization],
    source: str,
) -> str:
    value = manifest.get("RGB_bands") or dataset.get("RGB_bands")
    if value not in {None, ""}:
        if str(value).strip().lower() == "mixed":
            return "mixed"
        return _normalize_sample_rgb_bands(value)
    source_bands = {materialized.RGB_bands for materialized in sources}
    if len(source_bands) == 1:
        return next(iter(source_bands))
    if source_bands:
        return "mixed"
    raise DaftTrainingError(f"Dataset manifest is missing required RGB_bands: {source}")


def build_daft_training_payload(
    *,
    registry_root: str,
    model_id: str,
    base_weight_id: str,
    source_dataset_id: str,
    target_dataset_id: str,
    evaluation_dataset_id: str = "",
    alignment_mode: str,
    total_epochs: int,
    initial_lr: float,
    halve_epochs: str | Sequence[int],
    device: str = "auto",
    output_dir: str,
    batch_size: int = 2,
    num_workers: int = DEFAULT_DAFT_NUM_WORKERS,
    adv_lambda: float = 0.05,
    align_lambda: float = 0.005,
    lora_task_r: int = 16,
    lora_task_alpha: int = 32,
    lora_dropout: float = 0.05,
    output_weight_id: str = "",
    resume: bool = False,
    resume_checkpoint_path: str = "",
    save_optimizer_state: bool = False,
    pin_memory: bool = False,
    deterministic_cuda: bool = False,
    global_seed: int = DEFAULT_DAFT_GLOBAL_SEED,
) -> dict[str, Any]:
    """Build a JSON-ready DAFT training payload for UI jobs and tests."""

    return {
        "registry_root": registry_root,
        "model_id": model_id,
        "base_weight_id": base_weight_id,
        "source_dataset_id": source_dataset_id,
        "target_dataset_id": target_dataset_id,
        "evaluation_dataset_id": str(evaluation_dataset_id or "").strip(),
        "alignment_mode": normalize_alignment_mode(alignment_mode),
        "total_epochs": int(total_epochs),
        "initial_lr": float(initial_lr),
        "halve_epochs": list(parse_halve_epochs(halve_epochs)),
        "device": str(device),
        "output_dir": output_dir,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "global_seed": int(global_seed),
        "seed": int(global_seed),
        "adv_lambda": float(adv_lambda),
        "align_lambda": float(align_lambda),
        "lora_task_r": int(lora_task_r),
        "lora_task_alpha": int(lora_task_alpha),
        "lora_dropout": float(lora_dropout),
        "output_weight_id": str(output_weight_id or "").strip(),
        "resume": bool(resume),
        "resume_checkpoint_path": str(resume_checkpoint_path or "").strip(),
        "save_optimizer_state": bool(save_optimizer_state),
        "pin_memory": bool(pin_memory),
        "deterministic_cuda": bool(deterministic_cuda),
    }


def _mark_lsdsam_daft_trainable(model: Any, *, adapter_name: str) -> None:
    """Freeze all parameters except the new LoRA adapter, decoder, and seg head."""

    for param in model.parameters():
        param.requires_grad = False
    adapter_token = f".{adapter_name}."
    for name, param in model.named_parameters():
        if "lora_" in name and (adapter_token in name or adapter_name in name):
            param.requires_grad = True

    base_model = getattr(getattr(model, "base_model", None), "model", model)
    for module_name in ("mask_decoder", "seg_head"):
        module = getattr(base_model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def run_lsdsam_daft(
    config: DaftLSDSAMConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DaftLSDSAMResult:
    """Run LSDSAM DAFT training and register the merged DAFT weight."""

    cfg = config.normalized()
    normalized_model_id = normalize_model_name(cfg.model_id)
    if normalized_model_id in LSDFORMER_MODEL_IDS:
        from lsd_app.training.lsdformer import run_lsdformer_daft

        return run_lsdformer_daft(config, progress_callback=progress_callback)
    if normalized_model_id not in LSDSAM_MODEL_IDS:
        raise DaftTrainingError("DAFT training supports model_id='LSDSAM-H', 'LSDSAM-L', 'LSDSAM-B', or 'LSDFormer'.")

    torch = _require_module("torch", "Install the training extra before launching LSDSAM DAFT.")
    np = _require_module("numpy", "Install numpy before launching LSDSAM DAFT.")
    cv2 = _require_module("cv2", "Install opencv-python before launching LSDSAM DAFT.")
    _require_module("pywt", "Install PyWavelets before launching LSDSAM DAFT.")
    peft = _require_module("peft", "Install peft before launching LSDSAM DAFT.")
    SummaryWriter = _optional_summary_writer()

    device = resolve_training_device(cfg.device, torch)
    device_name = _training_device_name(torch, device)
    num_workers = _effective_daft_num_workers(cfg.num_workers, device)
    worker_note = (
        f"Requested DataLoader workers={cfg.num_workers}, using workers={num_workers} to avoid WSL/CUDA fork memory pressure."
        if num_workers != cfg.num_workers
        else f"DataLoader workers={num_workers}."
    )
    _validate_training_device(torch, device)
    _set_global_seeds(torch, np, cfg.seed, deterministic_cuda=cfg.deterministic_cuda)

    source = resolve_daft_dataset_materialization(
        registry_root=cfg.registry_root,
        dataset_id=cfg.source_dataset_id,
    )
    target = resolve_daft_dataset_materialization(
        registry_root=cfg.registry_root,
        dataset_id=cfg.target_dataset_id,
    )
    evaluation = (
        resolve_daft_dataset_materialization(
            registry_root=cfg.registry_root,
            dataset_id=cfg.evaluation_dataset_id,
        )
        if cfg.evaluation_dataset_id
        else None
    )
    if evaluation is not None and evaluation.dataset_class != "Evaluation":
        raise DaftTrainingError(
            f"Evaluation dataset {evaluation.dataset_id} is {evaluation.dataset_class}, expected Evaluation."
        )
    source_samples = source.supervised_train_samples()
    target_samples = target.target_train_samples()
    evaluation_samples = evaluation.evaluation_samples() if evaluation is not None else ()
    if not source_samples:
        raise DaftTrainingError(f"Source dataset {source.dataset_id} has no supervised train samples.")
    if not target_samples:
        raise DaftTrainingError(f"Target dataset {target.dataset_id} has no image samples.")

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    for path in (checkpoints_dir, logs_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.jsonl"
    config_snapshot_path = output_dir / TRAINING_CONFIG_SNAPSHOT_FILENAME

    emit = _progress_emitter(progress_callback)
    _save_training_config_snapshot(
        config_snapshot_path,
        config=cfg,
        kind="daft-training",
        latest_checkpoint_path=checkpoints_dir / "lsdsam_daft_latest_lora.pth",
        best_checkpoint_path=checkpoints_dir / "lsdsam_daft_best_lora.pth",
        metrics_path=metrics_path,
        selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint",
    )
    emit(
        {
            "status": "running",
            "percent": 0.02,
            "phase": "dataset",
            "message": "Resolved source/target datasets.",
            "source_sample_count": len(source_samples),
            "target_sample_count": len(target_samples),
            "evaluation_sample_count": len(evaluation_samples),
            "selection_metric": "eval_mIoU" if evaluation_samples else "last_checkpoint",
            "training_config_path": str(config_snapshot_path),
            "device": device,
            "device_name": device_name,
            "num_workers": num_workers,
            "configured_num_workers": cfg.num_workers,
            "worker_note": worker_note,
        }
    )

    weight_row = _lookup_weight(registry_root=cfg.registry_root, weight_id=cfg.base_weight_id)
    emit(
        {
            "status": "running",
            "percent": 0.04,
            "phase": "model",
            "message": f"Loading LSDSAM base weight on CPU, then moving model to {device_name}.",
            "device": device,
            "device_name": device_name,
        }
    )
    model, base_checkpoint = _build_lsdsam_for_daft(
        config=cfg,
        weight_row=weight_row,
        torch=torch,
        peft=peft,
        device=device_name,
    )
    model.to(device)
    _mark_lsdsam_daft_trainable(model, adapter_name=cfg.peft_adapter_name)
    emit(
        {
            "status": "running",
            "percent": 0.06,
            "phase": "model",
            "message": f"LSDSAM model is ready on {device_name}. {worker_note}",
            "device": device,
            "device_name": device_name,
            "num_workers": num_workers,
            "configured_num_workers": cfg.num_workers,
            "worker_note": worker_note,
        }
    )

    source_dataset = _lsdsam_dataset_from_materialization(
        torch,
        source,
        sample_selector="supervised_train_samples",
        dataset_cls=_LSDSAMSupervisedDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    target_dataset = _lsdsam_dataset_from_materialization(
        torch,
        target,
        sample_selector="target_train_samples",
        dataset_cls=_LSDSAMImagesDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    eval_dataset = (
        _lsdsam_dataset_from_materialization(
            torch,
            evaluation,
            sample_selector="evaluation_samples",
            dataset_cls=_LSDSAMSupervisedDataset,
            image_size=cfg.image_size,
            cv2=cv2,
            np=np,
        )
        if evaluation is not None and evaluation_samples
        else None
    )

    generator = torch.Generator()
    criterion_seg = _FocalDiceLoss(torch)
    criterion_adv = torch.nn.CrossEntropyLoss()
    align_loss_fn = _alignment_loss_fn(cfg.alignment_mode, torch=torch, device=device)
    discriminator = _DomainDiscriminator(torch, in_channels=256).to(device)
    trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    if not trainable_params:
        raise DaftTrainingError("No trainable LSDSAM parameters were found for DAFT.")
    opt_seg = torch.optim.AdamW(trainable_params, lr=cfg.initial_lr)
    opt_dis = torch.optim.AdamW(discriminator.parameters(), lr=cfg.discriminator_lr or cfg.initial_lr * 2.5)
    writer = SummaryWriter(log_dir=str(logs_dir)) if SummaryWriter is not None else None

    best_score: float | None = None
    best_epoch = 0
    best_checkpoint = checkpoints_dir / "lsdsam_daft_best_lora.pth"
    latest_checkpoint = checkpoints_dir / "lsdsam_daft_latest_lora.pth"
    total_batches = max(1, min(_ceil_batches(len(source_dataset), cfg.batch_size), _ceil_batches(len(target_dataset), cfg.batch_size)))
    global_total = max(1, cfg.total_epochs * total_batches)
    processed = 0
    loss_history: list[dict[str, Any]] = _load_daft_loss_history(metrics_path) if cfg.resume else []
    start_epoch_index = 0
    if cfg.resume:
        resume_checkpoint = (
            Path(cfg.resume_checkpoint_path).expanduser().resolve()
            if cfg.resume_checkpoint_path
            else latest_checkpoint
        )
        if not resume_checkpoint.exists():
            raise DaftTrainingError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_payload = _load_daft_checkpoint(torch, resume_checkpoint, device)
        resume_state = _checkpoint_model_state(resume_payload)
        model.load_state_dict(resume_state, strict=False)
        opt_seg_state = resume_payload.get("optimizer_state_dict") or resume_payload.get("optimizer_seg_state_dict")
        if isinstance(opt_seg_state, Mapping):
            try:
                opt_seg.load_state_dict(opt_seg_state)
            except (TypeError, ValueError, RuntimeError):
                pass
        opt_dis_state = resume_payload.get("optimizer_dis_state_dict")
        if isinstance(opt_dis_state, Mapping):
            try:
                opt_dis.load_state_dict(opt_dis_state)
            except (TypeError, ValueError, RuntimeError):
                pass
        resumed_epoch = _checkpoint_epoch(resume_payload)
        start_epoch_index = min(max(resumed_epoch, 0), cfg.total_epochs)
        processed = start_epoch_index * total_batches
        best_epoch, best_score = _checkpoint_epoch_score(torch, best_checkpoint, device)
        if best_epoch <= 0:
            best_epoch, best_score = _best_daft_epoch_score_from_history(loss_history, has_eval=bool(evaluation_samples))
        emit(
            {
                "status": "running",
                "percent": min(0.95, processed / global_total),
                "phase": "resume",
                "processed": processed,
                "total": global_total,
                "current_epoch": start_epoch_index + 1 if start_epoch_index < cfg.total_epochs else cfg.total_epochs,
                "total_epochs": cfg.total_epochs,
                "current_batch": 0,
                "total_batches": total_batches,
                "lr_seg": opt_seg.param_groups[0]["lr"],
                "lr_dis": opt_dis.param_groups[0]["lr"],
                "device": device,
                "device_name": device_name,
                "loss_history": loss_history,
                "best_epoch": best_epoch or None,
                "best_score": best_score,
                "selection_metric": "eval_mIoU" if evaluation_samples else "last_checkpoint",
                "training_config_path": str(config_snapshot_path),
                "latest_checkpoint_path": str(latest_checkpoint),
                "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "",
                "latest_log": f"Resumed DAFT from {resume_checkpoint.name} at epoch {resumed_epoch}.",
                "message": f"Resumed DAFT from checkpoint epoch {resumed_epoch}.",
            }
        )

    try:
        for epoch_index in range(start_epoch_index, cfg.total_epochs):
            epoch_number = epoch_index + 1
            generator.manual_seed(cfg.seed + epoch_index)
            seg_lr = cfg.initial_lr if epoch_number == 1 else learning_rate_for_epoch(
                epoch_number=epoch_number,
                initial_lr=cfg.initial_lr,
                halve_epochs=cfg.halve_epochs,
                min_lr=cfg.min_lr,
            )
            dis_base_lr = cfg.discriminator_lr or cfg.initial_lr * 2.5
            dis_lr = dis_base_lr if epoch_number == 1 else learning_rate_for_epoch(
                epoch_number=epoch_number,
                initial_lr=dis_base_lr,
                halve_epochs=cfg.halve_epochs,
                min_lr=cfg.min_lr,
            )
            _set_optimizer_lr(opt_seg, seg_lr)
            _set_optimizer_lr(opt_dis, dis_lr)

            source_loader = _make_loader(
                torch,
                source_dataset,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                generator=generator,
                collate_fn=_collate_supervised(torch),
                pin_memory=cfg.pin_memory,
            )
            target_loader = _make_loader(
                torch,
                target_dataset,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                generator=generator,
                collate_fn=_collate_images(torch),
                pin_memory=cfg.pin_memory,
            )

            model.train()
            discriminator.train()
            epoch_stats = _EpochStats()
            epoch_batch_losses: list[dict[str, Any]] = []
            for batch_index, ((src_imgs, src_masks, _src_ids), (tgt_imgs, _tgt_ids)) in enumerate(zip(source_loader, target_loader)):
                batch_display = batch_index + 1
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="batch_loaded",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                if epoch_number == 1:
                    warmup_fraction = (batch_index + 1) / max(1, total_batches)
                    _set_optimizer_lr(opt_seg, max(cfg.min_lr, cfg.initial_lr * warmup_fraction))
                    _set_optimizer_lr(opt_dis, max(cfg.min_lr, dis_base_lr * warmup_fraction))

                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="to_device",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                src_imgs = src_imgs.to(device).float()
                src_masks = src_masks.to(device).float()
                tgt_imgs = tgt_imgs.to(device).float()
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="forward",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                src_logits, src_feats = model(src_imgs, return_feats=True)
                _tgt_logits, tgt_feats = model(tgt_imgs, return_feats=True)
                src_targets = src_masks.argmax(dim=1)
                loss_seg, _dice_loss, _focal_loss = criterion_seg(src_logits, src_targets)

                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="discriminator_step",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                opt_dis.zero_grad()
                logits_src = discriminator(src_feats.detach())
                logits_tgt = discriminator(tgt_feats.detach())
                label_src = torch.zeros(logits_src.shape[0], dtype=torch.long, device=device)
                label_tgt = torch.ones(logits_tgt.shape[0], dtype=torch.long, device=device)
                loss_dis = 0.5 * (criterion_adv(logits_src, label_src) + criterion_adv(logits_tgt, label_tgt))
                loss_dis.backward()
                opt_dis.step()

                for param in discriminator.parameters():
                    param.requires_grad = False
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="adapter_adv_forward",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                opt_seg.zero_grad()
                pred_src = discriminator(src_feats)
                pred_tgt = discriminator(tgt_feats)
                label_src_as_tgt = torch.ones(pred_src.shape[0], dtype=torch.long, device=device)
                label_tgt_as_src = torch.zeros(pred_tgt.shape[0], dtype=torch.long, device=device)
                loss_adv = 0.5 * (criterion_adv(pred_src, label_src_as_tgt) + criterion_adv(pred_tgt, label_tgt_as_src))
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="alignment_loss",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                loss_align = align_loss_fn(src_feats, tgt_feats)
                total_loss = loss_seg + cfg.adv_lambda * loss_adv + cfg.align_lambda * loss_align
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="adapter_backward",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                total_loss.backward()
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="adapter_optimizer_step",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )
                opt_seg.step()
                for param in discriminator.parameters():
                    param.requires_grad = True
                _emit_batch_subphase(
                    emit,
                    "DAFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.total_epochs,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="optimizer_done",
                    processed=processed + 1,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_src_ids,
                    target_ids=_tgt_ids,
                    lr_seg=opt_seg.param_groups[0]["lr"],
                    lr_dis=opt_dis.param_groups[0]["lr"],
                )

                batch_loss_values = {
                    "seg": float(loss_seg.detach().cpu().item()),
                    "adv": float(loss_adv.detach().cpu().item()),
                    "dis": float(loss_dis.detach().cpu().item()),
                    "align": float(loss_align.detach().cpu().item()),
                    "total": float(total_loss.detach().cpu().item()),
                }
                epoch_stats.update(**batch_loss_values)
                processed += 1
                epoch_batch_losses.append(
                    {
                        "batch": batch_index + 1,
                        "lr_seg": float(opt_seg.param_groups[0]["lr"]),
                        "lr_dis": float(opt_dis.param_groups[0]["lr"]),
                        **batch_loss_values,
                    }
                )
                if batch_index == 0 or (batch_index + 1) % 20 == 0 or (batch_index + 1) == total_batches:
                    losses = epoch_stats.to_dict()
                    emit(
                        {
                            "status": "running",
                            "percent": min(0.95, processed / global_total),
                            "phase": "train",
                            "processed": processed,
                            "total": global_total,
                            "current_epoch": epoch_number,
                            "total_epochs": cfg.total_epochs,
                            "current_batch": batch_index + 1,
                            "total_batches": total_batches,
                            "lr_seg": opt_seg.param_groups[0]["lr"],
                            "lr_dis": opt_dis.param_groups[0]["lr"],
                            "device": device,
                            "device_name": device_name,
                            "losses": losses,
                            "loss_history": _loss_history_with_current(loss_history, epoch_number, losses),
                            "latest_log": (
                                f"[Epoch {epoch_number}/{cfg.total_epochs}] "
                                f"Batch {batch_display}/{total_batches} | "
                                f"Total={losses.get('total', 0.0):.4f} | "
                                f"Seg={losses.get('seg', 0.0):.4f} "
                                f"Adv(G)={losses.get('adv', 0.0):.4f} "
                                f"Dis(D)={losses.get('dis', 0.0):.4f} "
                                f"{cfg.alignment_mode}={losses.get('align', 0.0):.6f}"
                            ),
                            "message": (
                                f"Epoch {epoch_number}/{cfg.total_epochs} "
                                f"batch {batch_display}/{total_batches}: "
                                f"seg={epoch_stats.avg('seg'):.4f}, "
                                f"adv={epoch_stats.avg('adv'):.4f}, "
                                f"{cfg.alignment_mode}={epoch_stats.avg('align'):.6f}"
                            ),
                        }
                    )

            epoch_losses = epoch_stats.to_dict()
            loss_history.append(_loss_history_row(epoch_number, epoch_losses))
            train_metrics: dict[str, Any] = {}
            eval_metrics = (
                _evaluate_model(
                    model,
                    eval_dataset=eval_dataset,
                    torch=torch,
                    device=device,
                    criterion_seg=criterion_seg,
                    batch_size=cfg.batch_size,
                    num_workers=num_workers,
                    pin_memory=cfg.pin_memory,
                    progress_callback=emit,
                    progress_context={
                        "phase": "evaluation",
                        "evaluation_split": "evaluation",
                        "progress_offset": processed,
                        "total": global_total,
                        "current_epoch": epoch_number,
                        "total_epochs": cfg.total_epochs,
                        "current_batch": total_batches,
                        "total_batches": total_batches,
                        "lr_seg": opt_seg.param_groups[0]["lr"],
                        "lr_dis": opt_dis.param_groups[0]["lr"],
                        "device": device,
                        "device_name": device_name,
                        "losses": epoch_losses,
                        "loss_history": loss_history,
                        "best_epoch": best_epoch or None,
                        "best_score": best_score,
                        "selection_metric": "eval_mIoU",
                        "latest_checkpoint_path": str(latest_checkpoint),
                        "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "",
                    },
                )
                if eval_dataset is not None
                else {}
            )
            if eval_metrics and "mIoU" in eval_metrics:
                score = float(eval_metrics["mIoU"])
                is_best = best_score is None or score > best_score
            else:
                score = None
                is_best = epoch_number == cfg.total_epochs
            if is_best:
                best_score = score
                best_epoch = epoch_number
                emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving DAFT best checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
                _save_lora_checkpoint(
                    torch,
                    best_checkpoint,
                    model=model,
                    config=cfg,
                    epoch=epoch_number,
                    score=score,
                    source=source,
                    target=target,
                    optimizer=opt_seg,
                    optimizer_dis=opt_dis,
                )
                _save_training_config_snapshot(
                    config_snapshot_path,
                    config=cfg,
                    kind="daft-training",
                    latest_checkpoint_path=latest_checkpoint,
                    best_checkpoint_path=best_checkpoint,
                    metrics_path=metrics_path,
                    selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint",
                    current_epoch=epoch_number,
                    best_epoch=best_epoch,
                    best_score=best_score,
                )
            emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving DAFT latest checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
            _save_lora_checkpoint(
                torch,
                latest_checkpoint,
                model=model,
                config=cfg,
                epoch=epoch_number,
                score=score,
                source=source,
                target=target,
                optimizer=opt_seg,
                optimizer_dis=opt_dis,
            )
            _save_training_config_snapshot(
                config_snapshot_path,
                config=cfg,
                kind="daft-training",
                latest_checkpoint_path=latest_checkpoint,
                best_checkpoint_path=best_checkpoint,
                metrics_path=metrics_path,
                selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint",
                current_epoch=epoch_number,
                best_epoch=best_epoch,
                best_score=best_score,
            )
            emit({"status": "running", "phase": "checkpoint_saved", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saved DAFT checkpoints for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "latest_checkpoint_path": str(latest_checkpoint), "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "", "device": device, "device_name": device_name})
            record = {
                "epoch": epoch_number,
                "lr_seg": opt_seg.param_groups[0]["lr"],
                "lr_dis": opt_dis.param_groups[0]["lr"],
                "losses": epoch_losses,
                "batch_losses": epoch_batch_losses,
                "train": train_metrics,
                "evaluation": eval_metrics,
            }
            _append_jsonl(metrics_path, record)
            emit(
                {
                    "status": "running",
                    "percent": min(0.98, processed / global_total),
                    "phase": "evaluation" if eval_metrics else "checkpoint",
                    "processed": processed,
                    "total": global_total,
                    "current_epoch": epoch_number,
                    "total_epochs": cfg.total_epochs,
                    "total_batches": total_batches,
                    "lr_seg": opt_seg.param_groups[0]["lr"],
                    "lr_dis": opt_dis.param_groups[0]["lr"],
                    "device": device,
                    "device_name": device_name,
                    "losses": epoch_losses,
                    "loss_history": loss_history,
                    "train": train_metrics,
                    "evaluation": eval_metrics,
                    "best_epoch": best_epoch or epoch_number,
                    "best_score": best_score,
                    "selection_metric": "eval_mIoU" if eval_metrics else "last_checkpoint",
                    "training_config_path": str(config_snapshot_path),
                    "latest_checkpoint_path": str(latest_checkpoint),
                    "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "",
                    "latest_log": _epoch_log_line(
                        epoch_number=epoch_number,
                        total_epochs=cfg.total_epochs,
                        losses=epoch_losses,
                        evaluation=eval_metrics,
                        best_epoch=best_epoch or epoch_number,
                        best_score=best_score,
                        alignment_mode=cfg.alignment_mode,
                    ),
                    "message": (
                        f"Epoch {epoch_number}/{cfg.total_epochs} completed: "
                        f"total={epoch_losses.get('total', 0.0):.4f}"
                        + (f", eval_mIoU={eval_metrics['mIoU']:.4f}" if "mIoU" in eval_metrics else ", checkpoint=latest")
                    ),
                }
            )
            if writer is not None:
                for name, value in epoch_stats.to_dict().items():
                    writer.add_scalar(f"Epoch/{name}", value, epoch_number)
                writer.add_scalar("Epoch/LR_Seg", opt_seg.param_groups[0]["lr"], epoch_number)
                writer.add_scalar("Epoch/LR_Dis", opt_dis.param_groups[0]["lr"], epoch_number)
                for name, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"Evaluation/{name}", value, epoch_number)
    finally:
        if writer is not None:
            writer.close()

    checkpoint_to_merge = best_checkpoint if evaluation_samples and best_checkpoint.exists() else latest_checkpoint
    merged_checkpoint = checkpoints_dir / "lsdsam_daft_merged.pth"
    emit({"status": "running", "phase": "merge", "progress_kind": "merge", "percent": 0.99, "message": f"Merging DAFT checkpoint {checkpoint_to_merge.name}.", "device": device, "device_name": device_name})
    from lsd_app.models.lsdsam_merge import merge_lsdsam_lora_checkpoint

    merge_lsdsam_lora_checkpoint(
        lora_checkpoint_path=checkpoint_to_merge,
        output_path=merged_checkpoint,
        legacy_root=_legacy_root_from_config(cfg),
        sam_checkpoint_path=cfg.sam_checkpoint_path or weight_row.get("base_checkpoint_path") or base_checkpoint,
        image_size=cfg.image_size,
        device=device,
        lora_base_r=int(weight_row.get("lora_base_r") or 32),
        lora_base_alpha=int(weight_row.get("lora_base_alpha") or 64),
        lora_task_r=cfg.lora_task_r,
        lora_task_alpha=cfg.lora_task_alpha,
        lora_dropout=cfg.lora_dropout,
        peft_adapter_name=cfg.peft_adapter_name,
        sam_variant=_lsdsam_sam_variant(cfg.model_id),
    )

    emit({"status": "running", "phase": "register", "progress_kind": "register", "percent": 0.995, "message": "Registering merged DAFT weight.", "device": device, "device_name": device_name})
    registered = register_weight(
        registry_root=cfg.registry_root,
        model_id=cfg.model_id,
        stage="DAFT",
        version=f"DAFT-{target.version or target.dataset_id}-{_utc_compact()}",
        path=merged_checkpoint,
        metrics_summary=_training_selection_summary(
            best_score=best_score,
            best_epoch=best_epoch,
            metric_scope="eval" if evaluation_samples else "last_checkpoint",
        ),
        weight_id=cfg.output_weight_id or None,
        notes=(
            f"LSDSAM DAFT from source={source.dataset_id}, target={target.dataset_id}, "
            f"alignment={cfg.alignment_mode}, epochs={cfg.total_epochs}, device={device_name}."
        ),
        weight_type="merged",
        base_checkpoint_path=cfg.sam_checkpoint_path or weight_row.get("base_checkpoint_path") or base_checkpoint,
        adapter_profile=str(weight_row.get("adapter_profile") or "lsdsam-sam-b-lora-edge4-after512-cf-mh"),
        adapter_name="lsdsam-fewshot",
        image_size=cfg.image_size,
        input_mode="rgb_wavelets",
        channel_rule="RGB + wavelets",
        lora_base_r=int(weight_row.get("lora_base_r") or 32),
        lora_base_alpha=int(weight_row.get("lora_base_alpha") or 64),
        lora_task_r=cfg.lora_task_r,
        lora_task_alpha=cfg.lora_task_alpha,
        lora_dropout=cfg.lora_dropout,
        peft_adapter_name=cfg.peft_adapter_name,
        target_modules=DEFAULT_LORA_TARGET_MODULES,
    )
    result = DaftLSDSAMResult(
        output_dir=output_dir,
        lora_checkpoint_path=checkpoint_to_merge,
        merged_checkpoint_path=merged_checkpoint,
        registered_weight_id=registered.weight_id,
        registered_weight_path=registered.path,
        metrics_path=metrics_path,
        best_epoch=best_epoch or cfg.total_epochs,
        best_score=best_score,
        selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint",
        source_sample_count=len(source_samples),
        target_sample_count=len(target_samples),
        target_has_gt=bool(evaluation_samples),
        device=device_name,
    )
    emit(
        {
            "status": "running",
            "percent": 1.0,
            "phase": "register",
            "best_epoch": result.best_epoch,
            "best_score": result.best_score,
            "selection_metric": result.selection_metric,
            "device": device,
            "device_name": device_name,
            "merged_checkpoint_path": str(result.merged_checkpoint_path),
            "registered_weight_id": result.registered_weight_id,
            "registered_weight_path": str(result.registered_weight_path),
            "metrics_path": str(result.metrics_path),
            "training_config_path": str(config_snapshot_path),
            "loss_history": loss_history,
            "latest_log": f"Registered DAFT weight {registered.weight_id}.",
            "message": f"Registered DAFT weight {registered.weight_id}.",
        }
    )
    return result


def _lookup_dataset(*, registry_root: str | Path, dataset_id: str) -> Mapping[str, Any]:
    normalized = str(dataset_id).strip()
    for row in list_datasets(registry_root=registry_root):
        if str(row.get("dataset_id") or "") == normalized:
            return {**row, "_registry_root": str(registry_root)}
    raise DaftTrainingError(f"Unknown dataset id: {dataset_id}")


def _lookup_weight(*, registry_root: str | Path, weight_id: str) -> Mapping[str, Any]:
    normalized = str(weight_id).strip()
    for row in list_weights(registry_root=registry_root):
        if str(row.get("weight_id") or "") == normalized:
            if not row.get("exists", True):
                raise DaftTrainingError(f"Weight registry entry exists but file is missing: {weight_id}")
            return {**row, "_registry_root": str(registry_root)}
    raise DaftTrainingError(f"Unknown weight id: {weight_id}")


def _dataset_manifest_path(dataset: Mapping[str, Any]) -> Path:
    manifest = dataset.get("manifest_path")
    if manifest:
        path = _resolve_training_path(manifest, registry_root=dataset.get("_registry_root"))
        if path.exists():
            return path
    repository = dataset.get("repository_path")
    if repository:
        path = _resolve_training_path(repository, registry_root=dataset.get("_registry_root")) / "manifest.json"
        if path.exists():
            return path
    raise DaftTrainingError(f"Dataset has no readable manifest: {dataset.get('dataset_id')}")


def _require_rgb_bands(manifest: Mapping[str, Any], *, dataset: Mapping[str, Any], source: str) -> str:
    value = manifest.get("RGB_bands")
    if value in {None, ""}:
        value = dataset.get("RGB_bands")
    if value in {None, ""}:
        raise DaftTrainingError(f"Dataset manifest is missing required RGB_bands: {source}")
    if str(value).strip().lower() == "mixed":
        return "mixed"
    return _normalize_sample_rgb_bands(value)


def _normalize_sample_rgb_bands(value: Any) -> str:
    try:
        return normalize_rgb_bands(value)
    except ValueError as exc:
        raise DaftTrainingError(str(exc)) from exc


def _samples_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    default_RGB_bands: str,
) -> list[DaftTrainingSample]:
    rows: list[DaftTrainingSample] = []
    subset_rows = manifest.get("subsets", [])
    if not isinstance(subset_rows, Sequence) or isinstance(subset_rows, (str, bytes)):
        return rows
    for subset in subset_rows:
        if not isinstance(subset, Mapping):
            continue
        RGB_bands = _normalize_sample_rgb_bands(subset.get("RGB_bands") or default_RGB_bands)
        samples = subset.get("samples")
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            rows.extend(_samples_from_rows(samples, RGB_bands=RGB_bands))
        subset_manifest = subset.get("subset_manifest_path")
        if subset_manifest:
            subset_path = _resolve_training_path(subset_manifest, base_dir=manifest_path.parent)
            if not subset_path.exists():
                subset_path = (manifest_path.parent / str(subset_manifest)).resolve()
            if not subset_path.exists():
                continue
            subset_payload = _read_json(subset_path)
            subset_samples = subset_payload.get("samples", []) if isinstance(subset_payload, Mapping) else []
            if isinstance(subset_samples, Sequence) and not isinstance(subset_samples, (str, bytes)):
                rows.extend(_samples_from_rows(subset_samples, RGB_bands=RGB_bands))
    return rows


def _samples_from_rows(rows: Sequence[Any], *, RGB_bands: str) -> list[DaftTrainingSample]:
    samples: list[DaftTrainingSample] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        image_path = row.get("image_path") or row.get("output_image_path")
        if not image_path:
            continue
        label_path = row.get("label_path") or row.get("output_label_path")
        image = _resolve_training_path(image_path)
        label = _resolve_training_path(label_path) if label_path else None
        samples.append(
            DaftTrainingSample(
                sample_id=str(row.get("sample_id") or image.stem),
                image_path=image,
                label_path=label,
                split=str(row.get("split") or "train"),
                RGB_bands=_normalize_sample_rgb_bands(row.get("RGB_bands") or RGB_bands),
                label_polarity=LABEL_POLARITY_FOREGROUND_WHITE,
            )
        )
    return samples


def _infer_samples_from_repository(
    root_dir: Path,
    *,
    manifest: Mapping[str, Any],
    default_RGB_bands: str,
) -> list[DaftTrainingSample]:
    dataset_class = str(manifest.get("dataset_class") or "").upper()
    if dataset_class == "DAFT":
        return [
            DaftTrainingSample(path.stem, path, None, "all", default_RGB_bands)
            for path in sorted((root_dir / "images").glob("*.tif"))
        ]
    rows: list[DaftTrainingSample] = []
    RGB_bands_by_subset = _subset_rgb_bands(manifest)
    for image_dir in sorted(
        path
        for pattern in ("*/*/images", "subsets/*/*/images")
        for path in root_dir.glob(pattern)
        if path.is_dir()
    ):
        if "auxiliary" in image_dir.parts:
            continue
        split = image_dir.parent.name
        label_dir = image_dir.parent / "labels"
        subset_id = image_dir.parent.parent.name
        RGB_bands = RGB_bands_by_subset.get(subset_id, default_RGB_bands)
        for image_path in sorted(image_dir.glob("*.tif")):
            label_path = label_dir / image_path.name
            rows.append(
                DaftTrainingSample(
                    sample_id=f"{subset_id}/{split}/{image_path.stem}" if subset_id else image_path.stem,
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                    split=split,
                    RGB_bands=RGB_bands,
                    label_polarity=LABEL_POLARITY_FOREGROUND_WHITE,
                )
            )
    return rows


def _subset_rgb_bands(manifest: Mapping[str, Any]) -> dict[str, str]:
    orders: dict[str, str] = {}
    subset_rows = manifest.get("subsets", [])
    if not isinstance(subset_rows, Sequence) or isinstance(subset_rows, (str, bytes)):
        return orders
    for subset in subset_rows:
        if not isinstance(subset, Mapping):
            continue
        subset_id = str(subset.get("subset_id") or "").strip()
        RGB_bands = str(subset.get("RGB_bands") or "").strip()
        if subset_id and RGB_bands and RGB_bands != "mixed":
            orders[subset_id] = _normalize_sample_rgb_bands(RGB_bands)
    return orders


def _infer_samples_from_source_tree(
    manifest: Mapping[str, Any],
    *,
    default_RGB_bands: str,
) -> list[DaftTrainingSample]:
    source_root_value = (
        manifest.get("source_root")
        or manifest.get("source_imagery_path")
        or manifest.get("source_label_path")
    )
    if not source_root_value:
        return []
    source_root = Path(str(source_root_value)).expanduser().resolve()
    if not source_root.is_dir():
        return []

    rows: list[DaftTrainingSample] = []
    allowed_source_rels = _source_rel_filter(manifest)
    RGB_bands_by_source = _source_rel_rgb_bands(manifest)
    label_polarities = _source_rel_label_polarities(manifest)
    for image_dir in sorted(
        path
        for pattern in ("*/*/images", "*/*/*/images")
        for path in source_root.glob(pattern)
        if path.is_dir()
    ):
        if "DEM_derives" in image_dir.parts or "auxiliary" in image_dir.parts:
            continue
        split = image_dir.parent.name
        label_dir = image_dir.parent / "labels"
        source_rel = image_dir.parent.parent.relative_to(source_root).as_posix()
        if allowed_source_rels is not None and source_rel not in allowed_source_rels:
            continue
        RGB_bands = RGB_bands_by_source.get(source_rel, default_RGB_bands)
        label_polarity = label_polarities.get(source_rel, LABEL_POLARITY_FOREGROUND_WHITE)
        for image_path in sorted(_raster_files(image_dir)):
            label_path = _matching_label(label_dir, image_path.stem)
            rows.append(
                DaftTrainingSample(
                    sample_id=f"{source_rel}/{split}/{image_path.stem}",
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    RGB_bands=RGB_bands,
                    label_polarity=label_polarity,
                )
            )
    return rows


def _source_rel_filter(manifest: Mapping[str, Any]) -> set[str] | None:
    subset_rows = manifest.get("subsets", [])
    if not isinstance(subset_rows, Sequence) or isinstance(subset_rows, (str, bytes)):
        return None
    source_rels = {
        str(subset.get("source_rel") or subset.get("subset") or "").strip()
        for subset in subset_rows
        if isinstance(subset, Mapping)
    }
    source_rels.discard("")
    return source_rels or None


def _source_rel_rgb_bands(manifest: Mapping[str, Any]) -> dict[str, str]:
    orders: dict[str, str] = {}
    subset_rows = manifest.get("subsets", [])
    if not isinstance(subset_rows, Sequence) or isinstance(subset_rows, (str, bytes)):
        return orders
    for subset in subset_rows:
        if not isinstance(subset, Mapping):
            continue
        source_rel = str(subset.get("source_rel") or subset.get("subset") or "").strip()
        RGB_bands = str(subset.get("RGB_bands") or "").strip()
        if source_rel and RGB_bands and RGB_bands != "mixed":
            orders[source_rel] = _normalize_sample_rgb_bands(RGB_bands)
    return orders


def _source_rel_label_polarities(manifest: Mapping[str, Any]) -> dict[str, str]:
    polarities: dict[str, str] = {}
    subset_rows = manifest.get("subsets", [])
    if not isinstance(subset_rows, Sequence) or isinstance(subset_rows, (str, bytes)):
        return polarities
    for subset in subset_rows:
        if not isinstance(subset, Mapping):
            continue
        source_rel = str(subset.get("source_rel") or subset.get("subset") or "").strip()
        label_polarity = str(subset.get("label_polarity") or "").strip()
        if source_rel and label_polarity in {LABEL_POLARITY_FOREGROUND_WHITE, LABEL_POLARITY_FOREGROUND_BLACK}:
            polarities[source_rel] = label_polarity
    return polarities


def _raster_files(directory: Path) -> list[Path]:
    suffixes = {".tif", ".tiff", ".png"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _matching_label(label_dir: Path, image_stem: str) -> Path | None:
    for suffix in (".tif", ".tiff", ".png"):
        candidate = label_dir / f"{image_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _build_lsdsam_for_daft(
    *,
    config: DaftLSDSAMConfig,
    weight_row: Mapping[str, Any],
    torch: Any,
    peft: Any,
    device: str,
) -> tuple[Any, Path | None]:
    from lsd_app.models.lsdsam import (
        _extract_checkpoint_state_dict,
        _import_legacy_build_module,
        _infer_lora_rank,
        _is_merged_weight_type,
        _load_base_sam_checkpoint,
        _resolve_base_checkpoint_for_state,
        _resolve_build_sam_fn,
        _resolve_lora_adapter_name,
        _torch_load,
    )
    from lsd_app.models.lsdsam_merge import _matched_lora_modules

    legacy_root = Path(_legacy_root_from_config(config)).expanduser().resolve()
    build_module = _import_legacy_build_module(legacy_root)
    model_spec = get_model_spec(config.model_id)
    model_options = dict(model_spec.options or {})
    build_sam = _resolve_build_sam_fn(build_module, model_options)
    if build_sam is None:
        raise DaftTrainingError("Legacy LSDSAM build module does not expose the requested build_sam_* function.")
    weight_path = _resolve_training_path(weight_row.get("path") or "", registry_root=weight_row.get("_registry_root"))
    if not weight_path.exists():
        raise DaftTrainingError(f"Base weight path does not exist: {weight_path}")
    checkpoint_payload = _torch_load(torch, weight_path, "cpu")
    checkpoint_state = _extract_checkpoint_state_dict(checkpoint_payload, weight_path)
    base_checkpoint_value = config.sam_checkpoint_path or weight_row.get("base_checkpoint_path")
    base_options = dict(model_options)
    if base_checkpoint_value:
        base_options["sam_checkpoint_path"] = str(
            _resolve_training_path(base_checkpoint_value, registry_root=weight_row.get("_registry_root"))
        )
    base_checkpoint = _resolve_base_checkpoint_for_state(
        build_module,
        legacy_root,
        base_options,
        checkpoint_state,
        source=weight_path,
    )
    model = build_sam(checkpoint=False, map_location="cpu")
    if base_checkpoint is not None:
        _load_base_sam_checkpoint(torch, model, base_checkpoint, "cpu")
    model.image_encoder.resize_pos_embed(new_img_size=int(config.image_size))

    weight_type = str(weight_row.get("weight_type") or "").strip().lower()
    if _is_merged_weight_type(weight_type, checkpoint_state):
        model.load_state_dict(checkpoint_state, strict=False)
    else:
        matched_modules = _matched_lora_modules(model)
        adapter_name = _resolve_lora_adapter_name(
            checkpoint_state,
            preferred=str(weight_row.get("peft_adapter_name") or ""),
        )
        base_config = peft.LoraConfig(
            r=_infer_lora_rank(
                checkpoint_state,
                adapter_name=adapter_name,
                fallback=int(weight_row.get("lora_base_r") or weight_row.get("lora_task_r") or 32),
            ),
            lora_alpha=int(weight_row.get("lora_base_alpha") or weight_row.get("lora_task_alpha") or 64),
            target_modules=matched_modules,
            lora_dropout=float(weight_row.get("lora_dropout") or 0.05),
            bias="none",
        )
        model = peft.get_peft_model(model, base_config, adapter_name=adapter_name)
        model.load_state_dict(checkpoint_state, strict=False)
        model = model.merge_and_unload()

    for param in model.parameters():
        param.requires_grad = False
    matched_modules = _matched_lora_modules(model)
    new_config = peft.LoraConfig(
        r=int(config.lora_task_r),
        lora_alpha=int(config.lora_task_alpha),
        target_modules=matched_modules,
        lora_dropout=float(config.lora_dropout),
        bias="none",
    )
    if hasattr(model, "add_adapter"):
        model.add_adapter(config.peft_adapter_name, new_config)
        model.set_adapter(config.peft_adapter_name)
    else:
        model = peft.get_peft_model(model, new_config, adapter_name=config.peft_adapter_name)
        if hasattr(model, "set_adapter"):
            model.set_adapter(config.peft_adapter_name)
    return model, base_checkpoint


def _lsdsam_sam_variant(model_id: str) -> str:
    spec = get_model_spec(model_id)
    options = dict(spec.options or {})
    return str(options.get("sam_variant") or "B").strip().upper() or "B"


def _lsdsam_dataset_from_materialization(
    torch: Any,
    materialization: DaftDatasetMaterialization,
    *,
    sample_selector: str,
    dataset_cls: type,
    image_size: int,
    cv2: Any,
    np: Any,
) -> Any:
    if materialization.is_concat:
        datasets = [
            _lsdsam_dataset_from_materialization(
                torch,
                source,
                sample_selector=sample_selector,
                dataset_cls=dataset_cls,
                image_size=image_size,
                cv2=cv2,
                np=np,
            )
            for source in materialization.source_materializations
        ]
        datasets = [dataset for dataset in datasets if len(dataset) > 0]
        if not datasets:
            return dataset_cls((), image_size=image_size, cv2=cv2, torch=torch, np=np)
        return torch.utils.data.ConcatDataset(datasets)
    samples = getattr(materialization, sample_selector)()
    return dataset_cls(samples, image_size=image_size, cv2=cv2, torch=torch, np=np)


class _LSDSAMSupervisedDataset:
    def __init__(self, samples: Sequence[DaftTrainingSample], *, image_size: int, cv2: Any, torch: Any, np: Any) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.cv2 = cv2
        self.torch = torch
        self.np = np

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        sample = self.samples[index]
        if sample.label_path is None:
            raise DaftTrainingError(f"Supervised source sample has no label: {sample.sample_id}")
        image = _read_raster_hwc(sample.image_path, np=self.np)
        mask = _read_label_mask(
            sample.label_path,
            image_shape=image.shape[:2],
            image_size=self.image_size,
            label_polarity=sample.label_polarity,
            cv2=self.cv2,
            np=self.np,
        )
        tensor = _preprocess_lsdsam_image(
            image,
            RGB_bands=sample.RGB_bands,
            image_size=self.image_size,
            cv2=self.cv2,
            torch=self.torch,
            np=self.np,
        )
        one_hot = self.np.stack([1 - mask, mask], axis=0).astype("float32")
        return tensor, self.torch.as_tensor(one_hot, dtype=self.torch.float32), sample.sample_id


class _LSDSAMImagesDataset:
    def __init__(self, samples: Sequence[DaftTrainingSample], *, image_size: int, cv2: Any, torch: Any, np: Any) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.cv2 = cv2
        self.torch = torch
        self.np = np

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        sample = self.samples[index]
        image = _read_raster_hwc(sample.image_path, np=self.np)
        tensor = _preprocess_lsdsam_image(
            image,
            RGB_bands=sample.RGB_bands,
            image_size=self.image_size,
            cv2=self.cv2,
            torch=self.torch,
            np=self.np,
        )
        return tensor, sample.sample_id


def _preprocess_lsdsam_image(
    image: Any,
    *,
    RGB_bands: str,
    image_size: int,
    cv2: Any,
    torch: Any,
    np: Any,
) -> Any:
    from lsd_app.models.lsdsam import _build_edge_features, _ensure_unit_range, _imagenet_normalize, _resize_and_pad

    values = np.asarray(image, dtype=np.float32)
    chw = np.moveaxis(values, -1, 0)
    rgb = extract_rgb_bands(chw, RGB_bands, layout="chw")
    rgb_hwc = np.moveaxis(rgb, 0, -1)
    if rgb_hwc.shape[2] < 3:
        rgb_hwc = np.repeat(rgb_hwc[:, :, :1], 3, axis=2)
    rgb_hwc = _ensure_unit_range(rgb_hwc[:, :, :3])
    resized = _resize_and_pad(rgb_hwc, image_size, cv2)
    edge = _build_edge_features(resized, "wavelets", cv2)
    resized[:, :, :3] = _imagenet_normalize(resized[:, :, :3])
    model_input = np.concatenate((resized[:, :, :3], edge), axis=-1)
    model_input = np.moveaxis(model_input, -1, 0).astype("float32", copy=False)
    return torch.as_tensor(model_input, dtype=torch.float32)


def _read_raster_hwc(path: Path, *, np: Any) -> Any:
    rasterio = _require_module("rasterio", "Install geospatial dependencies before launching LSDSAM DAFT.")
    with rasterio.open(path) as src:
        values = src.read()
    if values.ndim == 2:
        values = values[None, ...]
    return np.moveaxis(values, 0, -1).astype("float32", copy=False)


def _read_label_mask(
    path: Path,
    *,
    image_shape: tuple[int, int],
    image_size: int,
    label_polarity: str = LABEL_POLARITY_FOREGROUND_WHITE,
    cv2: Any,
    np: Any,
) -> Any:
    if path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        raise DaftTrainingError(f"DAFT source supervision requires raster labels, got: {path}")
    rasterio = _require_module("rasterio", "Install geospatial dependencies before launching LSDSAM DAFT.")
    with rasterio.open(path) as src:
        label = src.read(1)
    if label.shape != image_shape:
        label = cv2.resize(label.astype("float32"), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = _label_to_foreground_mask(label, label_polarity=label_polarity, np=np)
    return _resize_and_pad_mask(mask, image_size=image_size, cv2=cv2, np=np)


def _label_to_foreground_mask(label: Any, *, label_polarity: str, np: Any) -> Any:
    values = np.asarray(label, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype="float32")
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        if label_polarity == LABEL_POLARITY_FOREGROUND_BLACK:
            foreground = values <= 0
        else:
            foreground = values > 0
        return foreground.astype("float32")
    scaled = values if low >= 0.0 and high <= 1.0 else (values - low) / (high - low)
    if label_polarity == LABEL_POLARITY_FOREGROUND_BLACK:
        foreground = scaled <= 0.5
    else:
        foreground = scaled > 0.5
    return foreground.astype("float32")


def _resize_and_pad_mask(mask: Any, *, image_size: int, cv2: Any, np: Any) -> Any:
    height, width = mask.shape[:2]
    scale = image_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_height, new_width) != (height, width):
        mask = cv2.resize(mask.astype("float32"), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((image_size, image_size), dtype="float32")
    padded[:new_height, :new_width] = (mask > 0).astype("float32")
    return padded


class _DomainDiscriminator:
    def __new__(cls, torch: Any, *, in_channels: int) -> Any:
        nn = torch.nn
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )


class _FocalDiceLoss:
    def __init__(self, torch: Any, *, smooth: float = 1.0, gamma: float = 2.0) -> None:
        self.torch = torch
        self.smooth = float(smooth)
        self.gamma = float(gamma)
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=255)

    def __call__(self, output: Any, target: Any) -> tuple[Any, Any, Any]:
        torch = self.torch
        one_hot = torch.nn.functional.one_hot(target.long(), num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
        probs = torch.nn.functional.softmax(output, dim=1)
        intersection = (probs * one_hot).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (probs.sum() + one_hot.sum() + self.smooth)
        ce_loss = self.ce(output, target.long())
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return dice_loss + focal_loss, dice_loss, focal_loss


def _alignment_loss_fn(mode: str, *, torch: Any, device: str) -> Callable[[Any, Any], Any]:
    normalized = normalize_alignment_mode(mode)
    if normalized == "IST":
        try:
            from pytorch_adapt.layers import ISTLoss
        except ModuleNotFoundError as exc:
            raise DaftTrainingError("IST alignment requires pytorch-adapt. Install it before launching IST DAFT.") from exc
        criterion = ISTLoss().to(device)

        def ist(src_feats: Any, tgt_feats: Any) -> Any:
            src_flat = _flatten_feats(src_feats)
            tgt_flat = _flatten_feats(tgt_feats)
            x = torch.cat([src_flat, tgt_flat], dim=0)
            y = torch.cat(
                [
                    torch.zeros(src_flat.shape[0], dtype=torch.long, device=x.device),
                    torch.ones(tgt_flat.shape[0], dtype=torch.long, device=x.device),
                ],
                dim=0,
            )
            return criterion(x, y)

        return ist
    if normalized == "MMD":
        return lambda src, tgt: _mmd_loss(torch, _flatten_feats(src), _flatten_feats(tgt))
    if normalized == "CoRAL":
        return lambda src, tgt: _coral_loss(torch, _flatten_feats(src), _flatten_feats(tgt))
    raise DaftTrainingError(f"Unsupported alignment mode: {mode}")


def _flatten_feats(feats: Any) -> Any:
    return feats.flatten(2).permute(0, 2, 1).reshape(-1, feats.size(1))


def _mmd_loss(torch: Any, source: Any, target: Any, *, kernel_mul: float = 2.0, kernel_num: int = 5) -> Any:
    total = torch.cat([source, target], dim=0)
    n_samples = total.shape[0]
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.shape[1])
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.shape[1])
    distances = ((total0 - total1) ** 2).sum(2)
    bandwidth = (distances.sum() - distances.trace()) / max(1, n_samples * (n_samples - 1))
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = sum(torch.exp(-distances / (bandwidth * (kernel_mul ** idx) + 1e-8)) for idx in range(kernel_num))
    batch_size = source.shape[0]
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


def _coral_loss(torch: Any, source: Any, target: Any) -> Any:
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().mm(source_centered) / max(1, source.shape[0] - 1)
    target_cov = target_centered.t().mm(target_centered) / max(1, target.shape[0] - 1)
    return torch.mean((source_cov - target_cov) ** 2) / (4 * source.shape[1] * source.shape[1])


def _epoch_log_line(
    *,
    epoch_number: int,
    total_epochs: int,
    losses: Mapping[str, float],
    evaluation: Mapping[str, float],
    best_epoch: int,
    best_score: float | None,
    alignment_mode: str,
) -> str:
    parts = [
        f"[Epoch {epoch_number}/{total_epochs}]",
        f"Total={float(losses.get('total', 0.0)):.4f}",
        f"Seg={float(losses.get('seg', 0.0)):.4f}",
        f"Adv(G)={float(losses.get('adv', 0.0)):.4f}",
        f"Dis(D)={float(losses.get('dis', 0.0)):.4f}",
        f"{alignment_mode}={float(losses.get('align', 0.0)):.6f}",
    ]
    if "mIoU" in evaluation:
        parts.append(f"mIoU={float(evaluation.get('mIoU', 0.0)):.4f}")
    if best_score is not None:
        parts.append(f"best={float(best_score):.4f}@epoch{best_epoch}")
    else:
        parts.append(f"checkpoint=epoch{best_epoch}")
    return " | ".join(parts)


def _loss_history_row(epoch_number: int, losses: Mapping[str, float]) -> dict[str, Any]:
    return {
        "epoch": int(epoch_number),
        "total": float(losses.get("total", 0.0)),
        "seg": float(losses.get("seg", 0.0)),
        "adv": float(losses.get("adv", 0.0)),
        "align": float(losses.get("align", 0.0)),
        "dis": float(losses.get("dis", 0.0)),
    }


def _loss_history_with_current(
    history: Sequence[Mapping[str, Any]],
    epoch_number: int,
    losses: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in history]
    current = _loss_history_row(epoch_number, losses)
    if rows and int(rows[-1].get("epoch") or 0) == int(epoch_number):
        rows[-1] = current
    else:
        rows.append(current)
    return rows


def _load_daft_checkpoint(torch: Any, path: str | Path, map_location: str) -> Mapping[str, Any]:
    checkpoint_location = "cpu" if str(map_location or "").lower().startswith("cuda") else map_location
    try:
        payload = torch.load(Path(path), map_location=checkpoint_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=checkpoint_location)
    if not isinstance(payload, Mapping):
        raise DaftTrainingError(f"Unsupported checkpoint payload: {path}")
    return payload


def _checkpoint_model_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise DaftTrainingError("Resume checkpoint does not contain a model_state_dict.")
    return state


def _checkpoint_epoch(payload: Mapping[str, Any]) -> int:
    try:
        return int(payload.get("epoch") or 0)
    except (TypeError, ValueError):
        return 0


def _checkpoint_score(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("mIoU")
    if value is None:
        value = payload.get("score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _checkpoint_epoch_score(torch: Any, path: Path, device: str) -> tuple[int, float | None]:
    if not path.exists():
        return 0, None
    payload = _load_daft_checkpoint(torch, path, device)
    return _checkpoint_epoch(payload), _checkpoint_score(payload)


def _load_daft_loss_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if not isinstance(item, Mapping):
            continue
        epoch = _checkpoint_epoch(item)
        losses = item.get("losses") if isinstance(item.get("losses"), Mapping) else {}
        if epoch > 0 and losses:
            rows.append(_loss_history_row(epoch, losses))
    return rows


def _best_daft_epoch_score_from_history(history: Sequence[Mapping[str, Any]], *, has_eval: bool) -> tuple[int, float | None]:
    best_epoch = 0
    best_score: float | None = None
    for row in history:
        epoch = _checkpoint_epoch(row)
        if epoch <= 0:
            continue
        score = None
        if has_eval and isinstance(row.get("evaluation"), Mapping):
            score = row["evaluation"].get("mIoU")
        else:
            score = None
        if score is None:
            continue
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            continue
        if best_score is None or score_float > best_score:
            best_epoch = epoch
            best_score = score_float
    return best_epoch, best_score


@dataclass
class _EpochStats:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, **values: float) -> None:
        self.count += 1
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def avg(self, key: str) -> float:
        return self.sums.get(key, 0.0) / max(1, self.count)

    def to_dict(self) -> dict[str, float]:
        return {key: self.avg(key) for key in sorted(self.sums)}


def _evaluate_model(
    model: Any,
    *,
    eval_dataset: Any,
    torch: Any,
    device: str,
    criterion_seg: Any,
    batch_size: int,
    num_workers: int,
    progress_callback: ProgressCallback | None = None,
    progress_context: Mapping[str, Any] | None = None,
    progress_every: int = 20,
    pin_memory: bool = True,
) -> dict[str, Any]:
    if eval_dataset is None or len(eval_dataset) == 0:
        return {}
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_supervised(torch),
        pin_memory=bool(pin_memory),
    )
    model.eval()
    total_loss = 0.0
    seen = 0
    total_batches = _ceil_batches(len(eval_dataset), batch_size)
    progress_every = max(1, int(progress_every))
    progress_context_dict = dict(progress_context or {})
    progress_offset = int(progress_context_dict.pop("progress_offset", 0) or 0)
    progress_total = int(progress_context_dict.get("total") or total_batches)
    progress_phase = str(progress_context_dict.get("phase") or "evaluation")
    progress_split = str(progress_context_dict.get("evaluation_split") or "eval")
    current_epoch = progress_context_dict.get("current_epoch")
    total_epochs = progress_context_dict.get("total_epochs")
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch_index, (images, masks, _ids) in enumerate(loader):
            current_batch = batch_index + 1
            if progress_callback is not None:
                progress_callback(
                    {
                        **progress_context_dict,
                        "status": "running",
                        "phase": progress_phase,
                        "progress_kind": "batch_subphase",
                        "batch_subphase": "eval_batch_loaded",
                        "evaluation_split": progress_split,
                        "eval_current_batch": current_batch,
                        "eval_total_batches": total_batches,
                        "processed": min(progress_total, progress_offset + batch_index),
                        "total": progress_total,
                        "percent": min(0.98, (progress_offset + batch_index) / max(1, progress_total)),
                        "latest_log": f"Evaluating {progress_split} {current_batch}/{total_batches} batch_loaded",
                        "message": f"Evaluating {progress_split} {current_batch}/{total_batches}: batch_loaded",
                    }
                )
            images = images.to(device).float()
            masks = masks.to(device).float()
            targets = masks.argmax(dim=1)
            if progress_callback is not None:
                progress_callback(
                    {
                        **progress_context_dict,
                        "status": "running",
                        "phase": progress_phase,
                        "progress_kind": "batch_subphase",
                        "batch_subphase": "eval_forward",
                        "evaluation_split": progress_split,
                        "eval_current_batch": current_batch,
                        "eval_total_batches": total_batches,
                        "processed": min(progress_total, progress_offset + batch_index),
                        "total": progress_total,
                        "percent": min(0.98, (progress_offset + batch_index) / max(1, progress_total)),
                        "latest_log": f"Evaluating {progress_split} {current_batch}/{total_batches} forward",
                        "message": f"Evaluating {progress_split} {current_batch}/{total_batches}: forward",
                    }
                )
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss, _dice, _focal = criterion_seg(outputs, targets)
            batch_size_seen = int(images.shape[0])
            seen += batch_size_seen
            total_loss += float(loss.detach().cpu().item()) * batch_size_seen
            preds = outputs.argmax(dim=1).flatten().to(dtype=torch.int64)
            truth = targets.flatten().to(dtype=torch.int64)
            valid = (truth >= 0) & (truth < 2) & (preds >= 0) & (preds < 2)
            if bool(valid.any()):
                bins = torch.bincount(truth[valid] * 2 + preds[valid], minlength=4)
                confusion += bins.reshape(2, 2).to(device=device, dtype=confusion.dtype)
            if progress_callback is not None and (
                batch_index == 0 or current_batch % progress_every == 0 or current_batch == total_batches
            ):
                avg_loss = total_loss / max(1, seen)
                processed = min(progress_total, progress_offset + current_batch)
                if current_epoch and total_epochs:
                    latest_log = (
                        f"[Epoch {current_epoch}/{total_epochs}] Evaluating {progress_split} "
                        f"{current_batch}/{total_batches} | Loss={avg_loss:.4f}"
                    )
                    message = (
                        f"Epoch {current_epoch}/{total_epochs} evaluating {progress_split} "
                        f"{current_batch}/{total_batches}: loss={avg_loss:.4f}"
                    )
                else:
                    latest_log = f"Evaluating {progress_split} {current_batch}/{total_batches} | Loss={avg_loss:.4f}"
                    message = f"Evaluating {progress_split} {current_batch}/{total_batches}: loss={avg_loss:.4f}"
                progress_callback(
                    {
                        **progress_context_dict,
                        "status": "running",
                        "phase": progress_phase,
                        "evaluation_split": progress_split,
                        "eval_current_batch": current_batch,
                        "eval_total_batches": total_batches,
                        "eval_loss": avg_loss,
                        "processed": processed,
                        "total": progress_total,
                        "percent": min(0.98, processed / max(1, progress_total)),
                        "latest_log": latest_log,
                        "message": message,
                    }
                )
    metrics = _legacy_metrics_from_confusion(confusion, torch=torch)
    avg_loss = total_loss / max(1, len(eval_dataset))
    return {
        "loss": avg_loss,
        "test_loss": avg_loss,
        **metrics,
    }


def _legacy_metrics_from_predictions(predictions: Any, labels: Any, *, torch: Any, device: str) -> dict[str, Any]:
    preds = predictions.flatten().to(device=device, dtype=torch.int64)
    truth = labels.flatten().to(device=device, dtype=torch.int64)
    valid = (truth >= 0) & (truth < 2) & (preds >= 0) & (preds < 2)
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    if bool(valid.any()):
        bins = torch.bincount(truth[valid] * 2 + preds[valid], minlength=4)
        confusion += bins.reshape(2, 2).to(device=device, dtype=confusion.dtype)
    return _legacy_metrics_from_confusion(confusion, torch=torch)


def _legacy_metrics_from_confusion(confusion: Any, *, torch: Any) -> dict[str, Any]:
    row_sum = confusion.sum(dim=1)
    col_sum = confusion.sum(dim=0)
    intersection = confusion.diag()
    union = row_sum + col_sum - intersection
    iou_tensor = torch.where(union > 0, intersection / union, torch.full_like(union, float("nan")))
    class_acc_tensor = torch.where(row_sum > 0, intersection / row_sum, torch.full_like(row_sum, float("nan")))
    total = confusion.sum()
    pixel_accuracy = float((intersection.sum() / total).detach().cpu().item()) if total > 0 else 0.0
    ious = [float(value) for value in iou_tensor.detach().cpu().tolist()]
    class_acc = [float(value) for value in class_acc_tensor.detach().cpu().tolist()]
    m_iou = float(torch.nanmean(iou_tensor).detach().cpu().item())
    mean_pixel_accuracy = float(torch.nanmean(class_acc_tensor).detach().cpu().item())
    tp = confusion[1, 1]
    fn = confusion[1, 0]
    fp = confusion[0, 1]
    recall_denom = tp + fn
    precision_denom = tp + fp
    recall = float((tp / recall_denom).detach().cpu().item()) if recall_denom > 0 else 0.0
    precision = float((tp / precision_denom).detach().cpu().item()) if precision_denom > 0 else 0.0
    f1_denom = precision + recall
    f1_score = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0
    kappa = 0.0
    if total > 0:
        observed = intersection.sum() / total
        expected = (row_sum * col_sum).sum() / (total * total)
        kappa_denom = 1 - expected
        kappa_denom_value = float(kappa_denom.detach().cpu().item())
        if abs(kappa_denom_value) > 1e-12:
            kappa = float(((observed - expected) / kappa_denom).detach().cpu().item())
    return {
        "pixel_accuracy": pixel_accuracy,
        "PA": pixel_accuracy,
        "class_pixel_accuracy": class_acc,
        "CPA": class_acc,
        "mean_pixel_accuracy": mean_pixel_accuracy,
        "MPA": mean_pixel_accuracy,
        "mIoU": m_iou,
        "IoU": ious,
        "IoU_background": ious[0],
        "IoU_landslide": ious[1],
        "IoU_foreground": ious[1],
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
        "kappa": kappa,
        "Kappa": kappa,
        "confusion_matrix": [[float(value) for value in row] for row in confusion.detach().cpu().tolist()],
    }


def _make_loader(
    torch: Any,
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    generator: Any,
    collate_fn: Any,
    pin_memory: bool = True,
) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        generator=generator,
        worker_init_fn=_worker_init_fn,
        pin_memory=bool(pin_memory),
    )


def _collate_supervised(torch: Any) -> Callable[[Sequence[Any]], tuple[Any, Any, list[str]]]:
    def collate(batch: Sequence[Any]) -> tuple[Any, Any, list[str]]:
        images, masks, ids = zip(*batch)
        return torch.stack(list(images), dim=0), torch.stack(list(masks), dim=0), list(ids)

    return collate


def _collate_images(torch: Any) -> Callable[[Sequence[Any]], tuple[Any, list[str]]]:
    def collate(batch: Sequence[Any]) -> tuple[Any, list[str]]:
        images, ids = zip(*batch)
        return torch.stack(list(images), dim=0), list(ids)

    return collate


def _save_lora_checkpoint(
    torch: Any,
    path: Path,
    *,
    model: Any,
    config: DaftLSDSAMConfig,
    epoch: int,
    score: float | None,
    source: DaftDatasetMaterialization,
    target: DaftDatasetMaterialization,
    optimizer: Any | None = None,
    optimizer_dis: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "mIoU": float(score) if score is not None else None,
        "model_state_dict": _lora_checkpoint_state_dict(model),
        "daft_config": _json_ready(asdict(config)),
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target.dataset_id,
        "peft_adapter_name": config.peft_adapter_name,
    }
    if bool(getattr(config, "save_optimizer_state", False)):
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
            payload["optimizer_seg_state_dict"] = optimizer.state_dict()
        if optimizer_dis is not None:
            payload["optimizer_dis_state_dict"] = optimizer_dis.state_dict()
    _ensure_torch_serialization_tls(torch)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _lora_checkpoint_state_dict(model: Any) -> Mapping[str, Any]:
    try:
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(model)
    except Exception:
        return model.state_dict()


def _ensure_torch_serialization_tls(torch: Any) -> None:
    """Fill PyTorch serialization TLS defaults missing in some nightly builds."""

    serialization = getattr(torch, "serialization", None)
    tls = getattr(serialization, "_serialization_tls", None)
    if tls is None:
        return
    if not hasattr(tls, "check_serializing_named_tensor"):
        setattr(tls, "check_serializing_named_tensor", False)


def _set_global_seeds(torch: Any, np: Any, seed: int, *, deterministic_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic_cuda)
        torch.backends.cudnn.benchmark = not bool(deterministic_cuda)
    try:
        torch.use_deterministic_algorithms(bool(deterministic_cuda), warn_only=True)
    except Exception:
        pass


def _worker_init_fn(worker_id: int) -> None:
    try:
        import numpy as np
        import torch

        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except Exception:
        random.seed(worker_id)


def _training_device_name(torch: Any, device: str) -> str:
    if str(device).startswith("cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
            return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
        except Exception:
            return "cuda"
    return str(device)


def _effective_daft_num_workers(requested: int, device: str) -> int:
    workers = max(0, int(requested))
    if workers > 0 and str(device).startswith("cuda") and _strict_memory_overcommit_enabled():
        return 0
    return workers


def _strict_memory_overcommit_enabled() -> bool:
    try:
        return Path("/proc/sys/vm/overcommit_memory").read_text(encoding="utf-8").strip() == "2"
    except OSError:
        return False


def _validate_training_device(torch: Any, device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        probe = torch.empty((1,), device=device)
        probe.zero_()
        torch.cuda.synchronize()
        del probe
    except Exception as exc:
        raise DaftTrainingError(
            "CUDA device is visible but failed a startup smoke test. "
            "This usually means the WSL CUDA context or driver is unhealthy, or another process recently exhausted/reset the GPU. "
            "Restart the WSL Python/Streamlit process or free GPU resources, then launch DAFT again. "
            f"Original CUDA error: {exc}"
        ) from exc


def _reset_cuda_memory_peak(torch: Any, device: str) -> None:
    if not str(device).startswith("cuda") or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _cuda_memory_progress(torch: Any, device: str) -> dict[str, float]:
    if not str(device).startswith("cuda") or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return {}
    try:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        return {
            "gpu_memory_allocated_mb": round(float(torch.cuda.memory_allocated(index)) / (1024 * 1024), 1),
            "gpu_memory_reserved_mb": round(float(torch.cuda.memory_reserved(index)) / (1024 * 1024), 1),
            "gpu_memory_peak_allocated_mb": round(float(torch.cuda.max_memory_allocated(index)) / (1024 * 1024), 1),
            "gpu_memory_peak_reserved_mb": round(float(torch.cuda.max_memory_reserved(index)) / (1024 * 1024), 1),
        }
    except Exception:
        return {}


def _cuda_memory_progress_from_payload(progress: Mapping[str, Any], seen_devices: set[str]) -> dict[str, float]:
    device = str(progress.get("device") or "")
    if not device.startswith("cuda"):
        return {}
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return {}
    if device not in seen_devices:
        _reset_cuda_memory_peak(torch_module, device)
        seen_devices.add(device)
    return _cuda_memory_progress(torch_module, device)


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _ceil_batches(sample_count: int, batch_size: int) -> int:
    return max(1, (sample_count + batch_size - 1) // batch_size)


def _progress_emitter(progress_callback: ProgressCallback | None) -> ProgressCallback:
    seen_cuda_devices: set[str] = set()

    def emit(progress: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            payload = dict(progress)
            payload.update(_cuda_memory_progress_from_payload(payload, seen_cuda_devices))
            progress_callback(payload)

    return emit


def _emit_batch_subphase(
    emit: ProgressCallback,
    prefix: str,
    *,
    epoch_number: int,
    total_epochs: int,
    batch_index: int,
    total_batches: int,
    subphase: str,
    processed: int,
    total: int,
    device: str,
    device_name: str,
    source_ids: Sequence[str] | None = None,
    target_ids: Sequence[str] | None = None,
    lr_seg: float | None = None,
    lr_dis: float | None = None,
    lr: float | None = None,
    percent_cap: float | None = None,
) -> None:
    sample_ids = [str(item) for item in tuple(source_ids or ())[:2]]
    sample_ids.extend(str(item) for item in tuple(target_ids or ())[:2])
    cap = float(percent_cap if percent_cap is not None else (0.98 if "TSFT" in prefix else 0.95))
    payload: dict[str, Any] = {
        "status": "running",
        "phase": "train",
        "progress_kind": "batch_subphase",
        "batch_subphase": subphase,
        "processed": int(processed),
        "total": int(total),
        "percent": min(cap, int(processed) / max(1, int(total))),
        "current_epoch": int(epoch_number),
        "total_epochs": int(total_epochs),
        "current_batch": int(batch_index),
        "total_batches": int(total_batches),
        "device": device,
        "device_name": device_name,
        "current_sample_ids": sample_ids,
        "latest_log": f"[{prefix} {epoch_number}/{total_epochs}] batch {batch_index}/{total_batches} {subphase}",
        "message": f"{prefix} epoch {epoch_number}/{total_epochs} batch {batch_index}/{total_batches}: {subphase}",
    }
    if lr_seg is not None:
        payload["lr_seg"] = lr_seg
    if lr_dis is not None:
        payload["lr_dis"] = lr_dis
    if lr is not None:
        payload["lr"] = lr
    emit(payload)


def _legacy_root_from_config(config: DaftLSDSAMConfig) -> Path:
    if config.legacy_root:
        return _resolve_legacy_root_path(config.legacy_root)
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "lsd_app" / "models" / "model_repository" / "LSDSAM",
    ]
    for candidate in candidates:
        resolved = _resolve_legacy_root_path(candidate)
        if resolved.exists():
            return resolved
    return _resolve_legacy_root_path(candidates[-1])


def _resolve_legacy_root_path(value: str | Path) -> Path:
    return resolve_user_path(value)


def _resolve_training_path(
    value: Any,
    *,
    registry_root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    return resolve_user_path(value, workspace_root=registry_root, base_dir=base_dir)


def _metrics_summary(*, best_score: float | None, best_epoch: int, target_has_gt: bool) -> str:
    if target_has_gt and best_score is not None:
        return f"best_mIoU={best_score:.4f};best_epoch={best_epoch}"
    return f"selection=final_epoch;epoch={best_epoch}"


def _training_selection_summary(*, best_score: float | None, best_epoch: int, metric_scope: str) -> str:
    if metric_scope != "eval" or best_score is None:
        return f"selection=last_checkpoint;epoch={best_epoch}"
    return f"best_eval_mIoU={best_score:.4f};best_epoch={best_epoch}"


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_training_config_snapshot(
    path: Path,
    *,
    config: Any,
    kind: str,
    latest_checkpoint_path: str | Path,
    best_checkpoint_path: str | Path,
    metrics_path: str | Path,
    selection_metric: str,
    current_epoch: int = 0,
    best_epoch: int = 0,
    best_score: float | None = None,
) -> None:
    _write_json_atomic(
        path,
        {
            "kind": kind,
            "config": asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "metrics_path": str(metrics_path),
            "selection_metric": selection_metric,
            "current_epoch": int(current_epoch or 0),
            "best_epoch": int(best_epoch or 0),
            "best_score": float(best_score) if best_score is not None else None,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DaftTrainingError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DaftTrainingError(f"Invalid JSON file: {path}") from exc


def _require_module(module_name: str, install_hint: str) -> Any:
    try:
        __import__(module_name)
        return sys.modules[module_name]
    except ModuleNotFoundError as exc:
        raise DaftTrainingError(f"Missing dependency '{module_name}'. {install_hint}") from exc


def _optional_summary_writer() -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return None
    return SummaryWriter


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _positive_int(value: Any, field_name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise DaftTrainingError(f"{field_name} must be positive.")
    return integer


def _non_negative_int(value: Any, field_name: str) -> int:
    integer = int(value)
    if integer < 0:
        raise DaftTrainingError(f"{field_name} must be non-negative.")
    return integer


def _positive_float(value: Any, field_name: str) -> float:
    number = float(value)
    if number <= 0:
        raise DaftTrainingError(f"{field_name} must be positive.")
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    number = float(value)
    if number < 0:
        raise DaftTrainingError(f"{field_name} must be non-negative.")
    return number
