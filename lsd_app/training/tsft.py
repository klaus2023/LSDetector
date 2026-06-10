"""TSFT stage planning and LSDSAM supervised fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsd_app.models.registry import normalize_model_name
from lsd_app.registry import register_weight
from lsd_app.training.config import StageConfig, TrainingConfig
from lsd_app.training.daft_lsdsam import (
    DAFT_TRAINING_DEVICE_POLICY,
    DEFAULT_LORA_TARGET_MODULES,
    DaftLSDSAMConfig,
    DaftTrainingError,
    ProgressCallback,
    _EpochStats,
    _FocalDiceLoss,
    _LSDSAMSupervisedDataset,
    _append_jsonl,
    _build_lsdsam_for_daft,
    _ceil_batches,
    _collate_supervised,
    _evaluate_model,
    _emit_batch_subphase,
    _legacy_metrics_from_confusion,
    _legacy_root_from_config,
    _lsdsam_dataset_from_materialization,
    _lsdsam_sam_variant,
    _lookup_weight,
    _make_loader,
    _metrics_summary,
    _optional_summary_writer,
    _positive_float,
    _positive_int,
    _progress_emitter,
    _require_module,
    _save_training_config_snapshot,
    _save_lora_checkpoint,
    _set_global_seeds,
    _set_optimizer_lr,
    _utc_compact,
    TRAINING_CONFIG_SNAPSHOT_FILENAME,
    resolve_daft_dataset_materialization,
    resolve_training_device,
)

TSFT_TRAINING_DEVICE_POLICY = DAFT_TRAINING_DEVICE_POLICY


class TsftTrainingError(DaftTrainingError):
    """Raised when TSFT training cannot be configured or launched."""


@dataclass(frozen=True)
class TsftLSDSAMConfig:
    """Runtime configuration for LSDSAM TSFT supervised training."""

    registry_root: str | Path
    dataset_id: str
    base_weight_id: str
    output_dir: str | Path
    evaluation_dataset_id: str = ""
    model_id: str = "lsdsam-fewshot"
    max_epoch: int = 10
    max_lr: float = 2e-4
    min_lr: float = 1e-6
    max_decay_epoch: int = 10
    batch_size: int = 2
    num_workers: int = 8
    device: str = "auto"
    seed: int = 2025
    image_size: int = 512
    lora_task_r: int = 16
    lora_task_alpha: int = 32
    lora_dropout: float = 0.05
    peft_adapter_name: str = "Task-Specific"
    output_weight_id: str = ""
    legacy_root: str | Path | None = None
    sam_checkpoint_path: str | Path | None = None
    resume: bool = False
    resume_checkpoint_path: str | Path | None = None
    cleanup_lora_checkpoints_after_register: bool = True
    save_optimizer_state: bool = False

    def normalized(self) -> "TsftLSDSAMConfig":
        num_workers = int(self.num_workers)
        if num_workers < 0:
            raise TsftTrainingError("num_workers must be non-negative.")
        min_lr = float(self.min_lr)
        if min_lr < 0:
            raise TsftTrainingError("min_lr must be non-negative.")
        max_lr = _positive_float(self.max_lr, "max_lr")
        if max_lr <= min_lr:
            raise TsftTrainingError("max_lr must be greater than min_lr.")
        max_epoch = _positive_int(self.max_epoch, "max_epoch")
        max_decay_epoch = _positive_int(self.max_decay_epoch, "max_decay_epoch")
        if max_decay_epoch > max_epoch:
            raise TsftTrainingError("max_decay_epoch must be less than or equal to max_epoch.")
        lora_dropout = float(self.lora_dropout)
        if lora_dropout < 0:
            raise TsftTrainingError("lora_dropout must be non-negative.")
        resume_checkpoint_path = (
            str(Path(self.resume_checkpoint_path).expanduser().resolve())
            if self.resume_checkpoint_path not in {None, ""}
            else None
        )
        return TsftLSDSAMConfig(
            **{
                **asdict(self),
                "registry_root": str(Path(self.registry_root).expanduser().resolve()),
                "output_dir": str(Path(self.output_dir).expanduser().resolve()),
                "evaluation_dataset_id": str(self.evaluation_dataset_id or "").strip(),
                "max_epoch": max_epoch,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "max_decay_epoch": max_decay_epoch,
                "batch_size": _positive_int(self.batch_size, "batch_size"),
                "num_workers": num_workers,
                "image_size": _positive_int(self.image_size, "image_size"),
                "lora_task_r": _positive_int(self.lora_task_r, "lora_task_r"),
                "lora_task_alpha": _positive_int(self.lora_task_alpha, "lora_task_alpha"),
                "lora_dropout": lora_dropout,
                "output_weight_id": str(self.output_weight_id or "").strip(),
                "resume": bool(self.resume),
                "resume_checkpoint_path": resume_checkpoint_path,
                "cleanup_lora_checkpoints_after_register": bool(self.cleanup_lora_checkpoints_after_register),
                "save_optimizer_state": bool(self.save_optimizer_state),
            }
        )

    def as_daft_config(self) -> DaftLSDSAMConfig:
        """Adapt TSFT config for shared LSDSAM/LoRA DAFT helpers."""

        return DaftLSDSAMConfig(
            registry_root=self.registry_root,
            source_dataset_id=self.dataset_id,
            target_dataset_id=self.dataset_id,
            base_weight_id=self.base_weight_id,
            output_dir=self.output_dir,
            evaluation_dataset_id=self.evaluation_dataset_id,
            model_id=self.model_id,
            total_epochs=self.max_epoch,
            initial_lr=self.max_lr,
            halve_epochs=(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            seed=self.seed,
            image_size=self.image_size,
            min_lr=self.min_lr,
            lora_task_r=self.lora_task_r,
            lora_task_alpha=self.lora_task_alpha,
            lora_dropout=self.lora_dropout,
            peft_adapter_name=self.peft_adapter_name,
            output_weight_id=self.output_weight_id,
            legacy_root=self.legacy_root,
            sam_checkpoint_path=self.sam_checkpoint_path,
            resume=self.resume,
            resume_checkpoint_path=self.resume_checkpoint_path,
            save_optimizer_state=self.save_optimizer_state,
        )


@dataclass(frozen=True)
class TsftTrainingResult:
    """Artifacts produced by one LSDSAM TSFT run."""

    output_dir: Path
    lora_checkpoint_path: Path
    merged_checkpoint_path: Path
    registered_weight_id: str
    registered_weight_path: Path
    metrics_path: Path
    best_epoch: int
    best_score: float | None
    selection_metric: str
    train_sample_count: int
    test_sample_count: int
    device: str
    cleanup_removed_checkpoint_paths: list[str]

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


def build_stage_plan(config: TrainingConfig, stage: StageConfig, previous: object | None = None):
    """Build the TSFT stage plan without importing training frameworks."""

    from lsd_app.training.pipeline import (
        StagePlan,
        stage_augmentation_profile,
        stage_checkpoint_input,
        stage_checkpoint_output,
    )

    return StagePlan(
        name="tsft",
        enabled=stage.enabled,
        output_dir=config.outputs.checkpoints / "tsft",
        checkpoint_input=stage_checkpoint_input(stage, previous),
        checkpoint_output=stage_checkpoint_output(config, "tsft"),
        data=stage.data,
        training=stage.training,
        params=stage.params,
        alignment_mode=stage.alignment_mode,
        notes=("task supervised fine-tuning dry-run",),
        augmentation_profile=stage_augmentation_profile(stage),
    )


def build_tsft_training_payload(
    *,
    registry_root: str,
    model_id: str,
    base_weight_id: str,
    dataset_id: str,
    evaluation_dataset_id: str = "",
    max_epoch: int,
    max_lr: float,
    min_lr: float,
    max_decay_epoch: int,
    device: str = "auto",
    output_dir: str,
    batch_size: int = 2,
    num_workers: int = 8,
    lora_task_r: int = 16,
    lora_task_alpha: int = 32,
    lora_dropout: float = 0.05,
    output_weight_id: str = "",
    resume: bool = False,
    resume_checkpoint_path: str = "",
    cleanup_lora_checkpoints_after_register: bool = True,
    save_optimizer_state: bool = False,
) -> dict[str, Any]:
    """Build a JSON-ready TSFT training payload for UI jobs and tests."""

    return {
        "registry_root": registry_root,
        "model_id": model_id,
        "base_weight_id": base_weight_id,
        "dataset_id": dataset_id,
        "evaluation_dataset_id": str(evaluation_dataset_id or "").strip(),
        "max_epoch": int(max_epoch),
        "max_lr": float(max_lr),
        "min_lr": float(min_lr),
        "max_decay_epoch": int(max_decay_epoch),
        "device": str(device),
        "output_dir": output_dir,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "lora_task_r": int(lora_task_r),
        "lora_task_alpha": int(lora_task_alpha),
        "lora_dropout": float(lora_dropout),
        "output_weight_id": str(output_weight_id or "").strip(),
        "resume": bool(resume),
        "resume_checkpoint_path": str(resume_checkpoint_path or "").strip(),
        "cleanup_lora_checkpoints_after_register": bool(cleanup_lora_checkpoints_after_register),
        "save_optimizer_state": bool(save_optimizer_state),
    }


def run_lsdsam_tsft(
    config: TsftLSDSAMConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> TsftTrainingResult:
    """Run LSDSAM TSFT supervised fine-tuning and register a merged TSFT weight."""

    cfg = config.normalized()
    normalized_model_id = normalize_model_name(cfg.model_id)
    if normalized_model_id in {"LSDFormer", "lsdformer-fast"}:
        from lsd_app.training.lsdformer import run_lsdformer_tsft

        return run_lsdformer_tsft(config, progress_callback=progress_callback)
    if normalized_model_id not in {"LSDSAM-H", "LSDSAM-L", "LSDSAM-B", "lsdsam-fewshot"}:
        raise TsftTrainingError("TSFT training supports model_id='LSDSAM-H', 'LSDSAM-L', 'LSDSAM-B', or 'LSDFormer'.")

    torch = _require_module("torch", "Install the training extra before launching LSDSAM TSFT.")
    np = _require_module("numpy", "Install numpy before launching LSDSAM TSFT.")
    cv2 = _require_module("cv2", "Install opencv-python before launching LSDSAM TSFT.")
    _require_module("pywt", "Install PyWavelets before launching LSDSAM TSFT.")
    peft = _require_module("peft", "Install peft before launching LSDSAM TSFT.")
    SummaryWriter = _optional_summary_writer()

    device = resolve_training_device(cfg.device, torch)
    device_name = _training_device_name(torch, device)
    _set_global_seeds(torch, np, cfg.seed)

    dataset = resolve_daft_dataset_materialization(
        registry_root=cfg.registry_root,
        dataset_id=cfg.dataset_id,
    )
    if dataset.dataset_class.upper() != "TSFT":
        raise TsftTrainingError(f"Dataset {dataset.dataset_id} is {dataset.dataset_class}, expected TSFT.")
    evaluation = (
        resolve_daft_dataset_materialization(
            registry_root=cfg.registry_root,
            dataset_id=cfg.evaluation_dataset_id,
        )
        if cfg.evaluation_dataset_id
        else None
    )
    if evaluation is not None and evaluation.dataset_class != "Evaluation":
        raise TsftTrainingError(
            f"Evaluation dataset {evaluation.dataset_id} is {evaluation.dataset_class}, expected Evaluation."
        )
    train_samples = dataset.supervised_train_samples()
    test_samples = evaluation.evaluation_samples() if evaluation is not None else ()
    if evaluation is not None and not test_samples:
        raise TsftTrainingError(f"Evaluation dataset {evaluation.dataset_id} has no labeled samples.")
    if not train_samples:
        raise TsftTrainingError(f"TSFT dataset {dataset.dataset_id} has no supervised train samples.")

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    for dir_path in (checkpoints_dir, logs_dir, metrics_dir):
        dir_path.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.jsonl"
    config_snapshot_path = output_dir / TRAINING_CONFIG_SNAPSHOT_FILENAME

    emit = _progress_emitter(progress_callback)
    _save_training_config_snapshot(
        config_snapshot_path,
        config=cfg,
        kind="tsft-training",
        latest_checkpoint_path=checkpoints_dir / "lsdsam_tsft_latest_lora.pth",
        best_checkpoint_path=checkpoints_dir / "lsdsam_tsft_best_lora.pth",
        metrics_path=metrics_path,
        selection_metric="eval_mIoU" if test_samples else "train_mIoU",
    )
    emit(
        {
            "status": "running",
            "percent": 0.02,
            "phase": "dataset",
            "message": "Resolved TSFT dataset.",
            "device": device,
            "device_name": device_name,
            "train_sample_count": len(train_samples),
            "test_sample_count": len(test_samples),
            "evaluation_sample_count": len(test_samples),
            "selection_metric": "eval_mIoU" if test_samples else "train_mIoU",
            "training_config_path": str(config_snapshot_path),
        }
    )

    daft_cfg = cfg.as_daft_config()
    weight_row = _lookup_weight(registry_root=cfg.registry_root, weight_id=cfg.base_weight_id)
    model, base_checkpoint = _build_lsdsam_for_daft(
        config=daft_cfg,
        weight_row=weight_row,
        torch=torch,
        peft=peft,
        device=device,
    )
    model.to(device)
    from lsd_app.training.daft_lsdsam import _mark_lsdsam_daft_trainable

    _mark_lsdsam_daft_trainable(model, adapter_name=cfg.peft_adapter_name)

    train_dataset = _lsdsam_dataset_from_materialization(
        torch,
        dataset,
        sample_selector="supervised_train_samples",
        dataset_cls=_LSDSAMSupervisedDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    test_dataset = (
        _lsdsam_dataset_from_materialization(
            torch,
            evaluation,
            sample_selector="evaluation_samples",
            dataset_cls=_LSDSAMSupervisedDataset,
            image_size=cfg.image_size,
            cv2=cv2,
            np=np,
        )
        if evaluation is not None and test_samples
        else None
    )

    generator = torch.Generator()
    criterion_seg = _FocalDiceLoss(torch)
    trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    if not trainable_params:
        raise TsftTrainingError("No trainable LSDSAM parameters were found for TSFT.")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.min_lr)
    writer = SummaryWriter(log_dir=str(logs_dir)) if SummaryWriter is not None else None

    best_score: float | None = None
    best_epoch = 0
    best_checkpoint = checkpoints_dir / "lsdsam_tsft_best_lora.pth"
    latest_checkpoint = checkpoints_dir / "lsdsam_tsft_latest_lora.pth"
    total_batches = _ceil_batches(len(train_dataset), cfg.batch_size)
    eval_batches_per_epoch = _ceil_batches(len(test_dataset), cfg.batch_size) if test_dataset is not None else 0
    epoch_work_total = total_batches + eval_batches_per_epoch
    global_total = max(1, cfg.max_epoch * epoch_work_total)
    processed = 0
    metrics_history: list[dict[str, Any]] = _load_tsft_metrics_history(metrics_path) if cfg.resume else []
    start_epoch_index = 0
    if cfg.resume:
        resume_checkpoint = (
            Path(cfg.resume_checkpoint_path).expanduser().resolve()
            if cfg.resume_checkpoint_path
            else latest_checkpoint
        )
        if not resume_checkpoint.exists():
            raise TsftTrainingError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_payload = _load_torch_checkpoint(torch, resume_checkpoint, device)
        resume_state = _checkpoint_model_state(resume_payload)
        model.load_state_dict(resume_state, strict=False)
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if isinstance(optimizer_state, Mapping):
            try:
                optimizer.load_state_dict(optimizer_state)
            except (TypeError, ValueError, RuntimeError):
                pass
        resumed_epoch = _checkpoint_epoch(resume_payload)
        start_epoch_index = min(max(resumed_epoch, 0), cfg.max_epoch)
        processed = start_epoch_index * epoch_work_total
        try:
            best_epoch, best_score = _checkpoint_epoch_score(torch, best_checkpoint, device)
        except (RuntimeError, EOFError, OSError, ValueError, TypeError):
            best_epoch, best_score = 0, None
        if best_epoch <= 0:
            best_epoch, best_score = _best_epoch_score_from_history(metrics_history, has_test=bool(test_samples))
        emit(
            {
                "status": "running",
                "percent": min(0.98, processed / global_total),
                "phase": "resume",
                "processed": processed,
                "total": global_total,
                "current_epoch": start_epoch_index + 1 if start_epoch_index < cfg.max_epoch else cfg.max_epoch,
                "total_epochs": cfg.max_epoch,
                "current_batch": 0,
                "total_batches": total_batches,
                "lr": optimizer.param_groups[0]["lr"],
                "device": device,
                "device_name": device_name,
                "metrics_history": metrics_history,
                "best_epoch": best_epoch or None,
                "best_score": best_score,
                "selection_metric": "eval_mIoU" if test_samples else "train_mIoU",
                "training_config_path": str(config_snapshot_path),
                "latest_checkpoint_path": str(latest_checkpoint),
                "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "",
                "latest_log": f"Resumed TSFT from {resume_checkpoint.name} at epoch {resumed_epoch}.",
                "message": f"Resumed TSFT from checkpoint epoch {resumed_epoch}.",
            }
        )

    try:
        for epoch_index in range(start_epoch_index, cfg.max_epoch):
            epoch_number = epoch_index + 1
            generator.manual_seed(cfg.seed + epoch_index)
            lr = tsft_warmup_cosine_lr(
                epoch_number=epoch_number,
                max_epoch=cfg.max_epoch,
                max_lr=cfg.max_lr,
                min_lr=cfg.min_lr,
                max_decay_epoch=cfg.max_decay_epoch,
            )
            _set_optimizer_lr(optimizer, lr)
            train_loader = _make_loader(
                torch,
                train_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                generator=generator,
                collate_fn=_collate_supervised(torch),
            )

            model.train()
            epoch_stats = _EpochStats()
            train_confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
            for batch_index, (images, masks, _ids) in enumerate(train_loader):
                batch_display = batch_index + 1
                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="batch_loaded",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )
                if epoch_number == 1:
                    _set_optimizer_lr(
                        optimizer,
                        tsft_warmup_batch_lr(
                            batch_index=batch_index,
                            total_batches=total_batches,
                            max_lr=cfg.max_lr,
                            min_lr=cfg.min_lr,
                        ),
                    )
                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="to_device",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )
                images = images.to(device).float()
                masks = masks.to(device).float()
                targets = masks.argmax(dim=1)
                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="forward",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss, dice_loss, focal_loss = criterion_seg(outputs, targets)
                _update_tsft_online_confusion(train_confusion, outputs=outputs, targets=targets, torch=torch, device=device)

                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="backward",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )
                optimizer.zero_grad()
                loss.backward()
                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="optimizer_step",
                    processed=processed,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )
                optimizer.step()
                _emit_batch_subphase(
                    emit,
                    "TSFT",
                    epoch_number=epoch_number,
                    total_epochs=cfg.max_epoch,
                    batch_index=batch_display,
                    total_batches=total_batches,
                    subphase="optimizer_done",
                    processed=processed + 1,
                    total=global_total,
                    device=device,
                    device_name=device_name,
                    source_ids=_ids,
                    lr=optimizer.param_groups[0]["lr"],
                )

                epoch_stats.update(
                    loss=float(loss.detach().cpu().item()),
                    dice=float(dice_loss.detach().cpu().item()),
                    focal=float(focal_loss.detach().cpu().item()),
                )
                processed += 1
                if batch_index == 0 or (batch_index + 1) % 20 == 0 or (batch_index + 1) == total_batches:
                    current_train = _tsft_training_progress_metrics(
                        epoch_stats=epoch_stats,
                        train_confusion=train_confusion,
                        torch=torch,
                    )
                    emit(
                        {
                            "status": "running",
                            "percent": min(0.98, processed / global_total),
                            "phase": "train",
                            "processed": processed,
                            "total": global_total,
                            "current_epoch": epoch_number,
                            "total_epochs": cfg.max_epoch,
                            "current_batch": batch_display,
                            "total_batches": total_batches,
                            "lr": optimizer.param_groups[0]["lr"],
                            "device": device,
                            "device_name": device_name,
                            "train": current_train,
                            "metrics_history": _metrics_history_with_current(
                                metrics_history,
                                {
                                    "epoch": epoch_number,
                                    "lr": optimizer.param_groups[0]["lr"],
                                    "train": current_train,
                                },
                            ),
                            "latest_log": (
                                f"[Epoch {epoch_number}/{cfg.max_epoch}] "
                                f"Batch {batch_display}/{total_batches} | "
                                f"Loss={current_train['loss']:.4f}"
                            ),
                            "message": (
                                f"Epoch {epoch_number}/{cfg.max_epoch} "
                                f"batch {batch_display}/{total_batches}: "
                                f"loss={current_train['loss']:.4f}"
                            ),
                        }
                    )

            train_metrics = _tsft_training_progress_metrics(
                epoch_stats=epoch_stats,
                train_confusion=train_confusion,
                torch=torch,
            )
            train_eval_batches = 0
            test_eval_batches = eval_batches_per_epoch
            eval_total = global_total
            processed_after_train_eval = processed
            if test_dataset is not None:
                emit(
                    {
                        "status": "running",
                        "percent": min(0.98, processed_after_train_eval / max(1, eval_total)),
                        "phase": "evaluation",
                        "evaluation_split": "eval",
                        "processed": processed_after_train_eval,
                        "total": eval_total,
                        "current_epoch": epoch_number,
                        "total_epochs": cfg.max_epoch,
                        "current_batch": total_batches,
                        "total_batches": total_batches,
                        "eval_current_batch": 0,
                        "eval_total_batches": test_eval_batches,
                        "lr": optimizer.param_groups[0]["lr"],
                        "device": device,
                        "device_name": device_name,
                        "train": train_metrics,
                        "metrics_history": metrics_history,
                        "latest_log": f"[Epoch {epoch_number}/{cfg.max_epoch}] Evaluating eval 0/{test_eval_batches}",
                        "message": f"Epoch {epoch_number}/{cfg.max_epoch} evaluating evaluation set.",
                    }
                )
                test_metrics = _rename_eval_metrics(
                    _evaluate_model(
                        model,
                        eval_dataset=test_dataset,
                        torch=torch,
                        device=device,
                        criterion_seg=criterion_seg,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        progress_callback=emit,
                        progress_context={
                            "phase": "evaluation",
                            "evaluation_split": "eval",
                            "progress_offset": processed_after_train_eval,
                            "total": eval_total,
                            "current_epoch": epoch_number,
                            "total_epochs": cfg.max_epoch,
                            "current_batch": total_batches,
                            "total_batches": total_batches,
                            "lr": optimizer.param_groups[0]["lr"],
                            "device": device,
                            "device_name": device_name,
                            "train": train_metrics,
                            "metrics_history": metrics_history,
                        },
                    )
                )
            else:
                test_metrics = {}
            processed += test_eval_batches
            scoring_metrics = test_metrics if test_metrics else train_metrics
            score = float(scoring_metrics["mIoU"]) if "mIoU" in scoring_metrics else None
            is_best = score is not None and (best_score is None or score > best_score)
            if score is None:
                is_best = epoch_number == cfg.max_epoch
            if is_best:
                best_score = score
                best_epoch = epoch_number
                emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving TSFT best checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
                _save_lora_checkpoint(
                    torch,
                    best_checkpoint,
                    model=model,
                    config=daft_cfg,
                    epoch=epoch_number,
                    score=score,
                    source=dataset,
                    target=dataset,
                    optimizer=optimizer,
                )
                _save_training_config_snapshot(
                    config_snapshot_path,
                    config=cfg,
                    kind="tsft-training",
                    latest_checkpoint_path=latest_checkpoint,
                    best_checkpoint_path=best_checkpoint,
                    metrics_path=metrics_path,
                    selection_metric="eval_mIoU" if test_samples else "train_mIoU",
                    current_epoch=epoch_number,
                    best_epoch=best_epoch,
                    best_score=best_score,
                )
            emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving TSFT latest checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
            _save_lora_checkpoint(
                torch,
                latest_checkpoint,
                model=model,
                config=daft_cfg,
                epoch=epoch_number,
                score=score,
                source=dataset,
                target=dataset,
                optimizer=optimizer,
            )
            _save_training_config_snapshot(
                config_snapshot_path,
                config=cfg,
                kind="tsft-training",
                latest_checkpoint_path=latest_checkpoint,
                best_checkpoint_path=best_checkpoint,
                metrics_path=metrics_path,
                selection_metric="eval_mIoU" if test_samples else "train_mIoU",
                current_epoch=epoch_number,
                best_epoch=best_epoch,
                best_score=best_score,
            )
            emit({"status": "running", "phase": "checkpoint_saved", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saved TSFT checkpoints for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "latest_checkpoint_path": str(latest_checkpoint), "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "", "device": device, "device_name": device_name})
            record = _tsft_metric_record(
                epoch=epoch_number,
                lr=optimizer.param_groups[0]["lr"],
                train=train_metrics,
                test=test_metrics,
            )
            _append_jsonl(metrics_path, record)
            metrics_history.append(record)
            emit(
                {
                    "status": "running",
                    "percent": min(0.98, processed / max(1, eval_total)),
                    "phase": "evaluation" if test_metrics else "checkpoint",
                    "processed": processed,
                    "total": eval_total,
                    "current_epoch": epoch_number,
                    "total_epochs": cfg.max_epoch,
                    "total_batches": total_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                    "device": device,
                    "device_name": device_name,
                    "train": train_metrics,
                    "test": test_metrics,
                    "eval": test_metrics,
                    "metrics_history": metrics_history,
                    "best_epoch": best_epoch or epoch_number,
                    "best_score": best_score,
                    "selection_metric": "eval_mIoU" if test_samples else "train_mIoU",
                    "training_config_path": str(config_snapshot_path),
                    "latest_checkpoint_path": str(latest_checkpoint),
                    "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "",
                    "latest_log": _tsft_epoch_log_line(
                        epoch_number=epoch_number,
                        total_epochs=cfg.max_epoch,
                        train=train_metrics,
                        test=test_metrics,
                        best_epoch=best_epoch or epoch_number,
                        best_score=best_score,
                    ),
                    "message": (
                        f"Epoch {epoch_number}/{cfg.max_epoch} completed: "
                        f"train_loss={train_metrics.get('loss', 0.0):.4f}, "
                        f"train_mIoU={train_metrics.get('mIoU', 0.0):.4f}"
                        + (f", eval_mIoU={test_metrics['mIoU']:.4f}" if "mIoU" in test_metrics else "")
                    ),
                }
            )
            if writer is not None:
                for prefix, values in (("Train", train_metrics), ("Evaluation", test_metrics)):
                    for name, value in values.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"{prefix}/{name}", value, epoch_number)
                writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch_number)
    finally:
        if writer is not None:
            writer.close()

    checkpoint_to_merge = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
    merged_checkpoint = checkpoints_dir / "lsdsam_tsft_merged.pth"
    emit({"status": "running", "phase": "merge", "progress_kind": "merge", "percent": 0.99, "message": f"Merging TSFT checkpoint {checkpoint_to_merge.name}.", "device": device, "device_name": device_name})
    from lsd_app.models.lsdsam_merge import merge_lsdsam_lora_checkpoint

    merge_lsdsam_lora_checkpoint(
        lora_checkpoint_path=checkpoint_to_merge,
        output_path=merged_checkpoint,
        legacy_root=_legacy_root_from_config(daft_cfg),
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

    emit({"status": "running", "phase": "register", "progress_kind": "register", "percent": 0.995, "message": "Registering merged TSFT weight.", "device": device, "device_name": device_name})
    registered = register_weight(
        registry_root=cfg.registry_root,
        model_id=cfg.model_id,
        stage="TSFT",
        version=f"TSFT-{dataset.version or dataset.dataset_id}-{_utc_compact()}",
        path=merged_checkpoint,
        metrics_summary=_tsft_metrics_summary(
            best_score=best_score,
            best_epoch=best_epoch or cfg.max_epoch,
            has_test=bool(test_samples),
            train_count=len(train_samples),
            test_count=len(test_samples),
        ),
        weight_id=cfg.output_weight_id or None,
        notes=(
            f"LSDSAM TSFT supervised fine-tuning on dataset={dataset.dataset_id}, "
            f"epochs={cfg.max_epoch}, device={device_name}."
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
    cleanup_removed = (
        _cleanup_tsft_lora_checkpoints(best_checkpoint, latest_checkpoint)
        if cfg.cleanup_lora_checkpoints_after_register
        else []
    )
    result = TsftTrainingResult(
        output_dir=output_dir,
        lora_checkpoint_path=checkpoint_to_merge,
        merged_checkpoint_path=merged_checkpoint,
        registered_weight_id=registered.weight_id,
        registered_weight_path=registered.path,
        metrics_path=metrics_path,
        best_epoch=best_epoch or cfg.max_epoch,
        best_score=best_score,
        selection_metric="eval_mIoU" if test_samples else "train_mIoU",
        train_sample_count=len(train_samples),
        test_sample_count=len(test_samples),
        device=device_name,
        cleanup_removed_checkpoint_paths=cleanup_removed,
    )
    emit(
        {
            "status": "running",
            "percent": 1.0,
            "phase": "register",
            "best_epoch": result.best_epoch,
            "best_score": result.best_score,
            "selection_metric": result.selection_metric,
            "merged_checkpoint_path": str(result.merged_checkpoint_path),
            "registered_weight_id": result.registered_weight_id,
            "registered_weight_path": str(result.registered_weight_path),
            "metrics_path": str(result.metrics_path),
            "training_config_path": str(config_snapshot_path),
            "device": device,
            "device_name": device_name,
            "metrics_history": metrics_history,
            "latest_checkpoint_path": "" if str(latest_checkpoint) in cleanup_removed else str(latest_checkpoint),
            "best_checkpoint_path": "" if str(best_checkpoint) in cleanup_removed else str(best_checkpoint),
            "cleanup_removed_checkpoint_paths": cleanup_removed,
            "latest_log": _tsft_register_log(registered.weight_id, cleanup_removed),
            "message": _tsft_register_log(registered.weight_id, cleanup_removed),
        }
    )
    return result



def finalize_tsft_checkpoint(
    config: TsftLSDSAMConfig,
    *,
    checkpoint_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TsftTrainingResult:
    """Merge/register an existing TSFT LoRA checkpoint, even after interruption."""

    cfg = config.normalized()
    normalized_model_id = normalize_model_name(cfg.model_id)
    if normalized_model_id in {"LSDFormer", "lsdformer-fast"}:
        raise TsftTrainingError(
            "Interrupted LSDFormer TSFT checkpoint finalization is not available yet. Resume the job to register it."
        )
    if normalized_model_id not in {"LSDSAM-H", "LSDSAM-L", "LSDSAM-B", "lsdsam-fewshot"}:
        raise TsftTrainingError("TSFT finalization supports model_id='LSDSAM-H', 'LSDSAM-L', or 'LSDSAM-B'.")
    torch = _require_module("torch", "Install the training extra before finalizing LSDSAM TSFT.")
    _require_module("peft", "Install peft before finalizing LSDSAM TSFT.")
    device = resolve_training_device(cfg.device, torch)
    device_name = _training_device_name(torch, device)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics" / "metrics.jsonl"
    best_checkpoint = checkpoints_dir / "lsdsam_tsft_best_lora.pth"
    latest_checkpoint = checkpoints_dir / "lsdsam_tsft_latest_lora.pth"
    checkpoint_to_merge = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path not in {None, ""}
        else (best_checkpoint if best_checkpoint.exists() else latest_checkpoint)
    )
    if not checkpoint_to_merge.exists():
        raise TsftTrainingError(f"TSFT checkpoint does not exist: {checkpoint_to_merge}")

    dataset = resolve_daft_dataset_materialization(
        registry_root=cfg.registry_root,
        dataset_id=cfg.dataset_id,
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
        raise TsftTrainingError(
            f"Evaluation dataset {evaluation.dataset_id} is {evaluation.dataset_class}, expected Evaluation."
        )
    train_samples = dataset.supervised_train_samples()
    test_samples = evaluation.evaluation_samples() if evaluation is not None else ()
    weight_row = _lookup_weight(registry_root=cfg.registry_root, weight_id=cfg.base_weight_id)
    checkpoint_payload = _load_torch_checkpoint(torch, checkpoint_to_merge, "cpu")
    best_epoch = _checkpoint_epoch(checkpoint_payload)
    best_score = _checkpoint_score(checkpoint_payload)
    history = _load_tsft_metrics_history(metrics_path)
    if best_epoch <= 0 or best_score is None:
        history_epoch, history_score = _best_epoch_score_from_history(history, has_test=bool(test_samples))
        best_epoch = best_epoch or history_epoch
        best_score = best_score if best_score is not None else history_score
    if best_epoch <= 0:
        best_epoch = int(history[-1].get("epoch", 0)) if history else 0

    emit = _progress_emitter(progress_callback)
    emit(
        {
            "status": "running",
            "percent": 0.05,
            "phase": "merge",
            "message": f"Merging TSFT checkpoint {checkpoint_to_merge.name}.",
            "best_epoch": best_epoch or None,
            "best_score": best_score,
            "selection_metric": "eval_mIoU" if test_samples else "train_mIoU",
        }
    )
    daft_cfg = cfg.as_daft_config()
    merged_checkpoint = checkpoints_dir / "lsdsam_tsft_merged.pth"
    from lsd_app.models.lsdsam_merge import merge_lsdsam_lora_checkpoint

    merge_lsdsam_lora_checkpoint(
        lora_checkpoint_path=checkpoint_to_merge,
        output_path=merged_checkpoint,
        legacy_root=_legacy_root_from_config(daft_cfg),
        sam_checkpoint_path=cfg.sam_checkpoint_path or weight_row.get("base_checkpoint_path"),
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
    registered = register_weight(
        registry_root=cfg.registry_root,
        model_id=cfg.model_id,
        stage="TSFT",
        version=f"TSFT-{dataset.version or dataset.dataset_id}-finalized-{_utc_compact()}",
        path=merged_checkpoint,
        metrics_summary=_tsft_metrics_summary(
            best_score=best_score,
            best_epoch=best_epoch or cfg.max_epoch,
            has_test=bool(test_samples),
            train_count=len(train_samples),
            test_count=len(test_samples),
        ),
        weight_id=cfg.output_weight_id or None,
        notes=(
            f"LSDSAM TSFT checkpoint finalization on dataset={dataset.dataset_id}, "
            f"checkpoint={checkpoint_to_merge.name}, device={device_name}."
        ),
        weight_type="merged",
        base_checkpoint_path=cfg.sam_checkpoint_path or weight_row.get("base_checkpoint_path"),
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
    cleanup_removed = (
        _cleanup_tsft_lora_checkpoints(best_checkpoint, latest_checkpoint)
        if cfg.cleanup_lora_checkpoints_after_register
        else []
    )
    result = TsftTrainingResult(
        output_dir=output_dir,
        lora_checkpoint_path=checkpoint_to_merge,
        merged_checkpoint_path=merged_checkpoint,
        registered_weight_id=registered.weight_id,
        registered_weight_path=registered.path,
        metrics_path=metrics_path,
        best_epoch=best_epoch or cfg.max_epoch,
        best_score=best_score,
        selection_metric="eval_mIoU" if test_samples else "train_mIoU",
        train_sample_count=len(train_samples),
        test_sample_count=len(test_samples),
        device=device_name,
        cleanup_removed_checkpoint_paths=cleanup_removed,
    )
    emit(
        {
            "status": "running",
            "percent": 1.0,
            "phase": "register",
            "best_epoch": result.best_epoch,
            "best_score": result.best_score,
            "selection_metric": result.selection_metric,
            "merged_checkpoint_path": str(result.merged_checkpoint_path),
            "registered_weight_id": result.registered_weight_id,
            "registered_weight_path": str(result.registered_weight_path),
            "metrics_path": str(result.metrics_path),
            "latest_checkpoint_path": "" if str(latest_checkpoint) in cleanup_removed else str(latest_checkpoint),
            "best_checkpoint_path": "" if str(best_checkpoint) in cleanup_removed else str(best_checkpoint),
            "cleanup_removed_checkpoint_paths": cleanup_removed,
            "latest_log": _tsft_register_log(registered.weight_id, cleanup_removed),
            "message": _tsft_register_log(registered.weight_id, cleanup_removed),
        }
    )
    return result



def _cleanup_tsft_lora_checkpoints(*paths: str | Path) -> list[str]:
    """Remove temporary LoRA checkpoints after a merged TSFT weight is registered."""

    removed: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            continue
        path.unlink()
        removed.append(str(path))
    return removed


def _tsft_register_log(weight_id: str, cleanup_removed: Sequence[str]) -> str:
    suffix = f" Cleaned up {len(cleanup_removed)} temporary LoRA checkpoint(s)." if cleanup_removed else ""
    return f"Registered TSFT weight {weight_id}.{suffix}"


def _load_torch_checkpoint(torch: Any, path: str | Path, map_location: str) -> Mapping[str, Any]:
    checkpoint_location = "cpu" if str(map_location or "").lower().startswith("cuda") else map_location
    try:
        payload = torch.load(Path(path), map_location=checkpoint_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=checkpoint_location)
    if not isinstance(payload, Mapping):
        raise TsftTrainingError(f"Unsupported checkpoint payload: {path}")
    return payload


def _checkpoint_model_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise TsftTrainingError("Resume checkpoint does not contain a model_state_dict.")
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
    payload = _load_torch_checkpoint(torch, path, device)
    return _checkpoint_epoch(payload), _checkpoint_score(payload)


def _load_tsft_metrics_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = __import__("json").loads(line)
        except ValueError:
            continue
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _update_tsft_online_confusion(
    confusion: Any,
    *,
    outputs: Any,
    targets: Any,
    torch: Any,
    device: str,
) -> None:
    """Accumulate train-set segmentation metrics from the same forward pass used for training."""

    with torch.no_grad():
        preds = outputs.detach().argmax(dim=1).flatten().to(device=device, dtype=torch.int64)
        truth = targets.detach().flatten().to(device=device, dtype=torch.int64)
        valid = (truth >= 0) & (truth < 2) & (preds >= 0) & (preds < 2)
        if bool(valid.any()):
            bins = torch.bincount(truth[valid] * 2 + preds[valid], minlength=4)
            confusion += bins.reshape(2, 2).to(device=device, dtype=confusion.dtype)


def _tsft_training_progress_metrics(
    *,
    epoch_stats: _EpochStats,
    train_confusion: Any | None,
    torch: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "loss": epoch_stats.avg("loss"),
        "dice": epoch_stats.avg("dice"),
        "focal": epoch_stats.avg("focal"),
    }
    if train_confusion is not None:
        metrics.update(_legacy_metrics_from_confusion(train_confusion, torch=torch))
    return metrics


def _best_epoch_score_from_history(history: Sequence[Mapping[str, Any]], *, has_test: bool) -> tuple[int, float | None]:
    best_epoch = 0
    best_score: float | None = None
    for row in history:
        if has_test:
            metrics = row.get("eval") if isinstance(row.get("eval"), Mapping) else row.get("test")
            flat_value = row.get("eval.mIoU", row.get("test.mIoU"))
        else:
            metrics = row.get("train") if isinstance(row.get("train"), Mapping) else {}
            flat_value = row.get("train.mIoU")
        value = metrics.get("mIoU") if isinstance(metrics, Mapping) else flat_value
        if value is None:
            value = flat_value
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = int(row.get("epoch") or 0)
    if best_epoch:
        return best_epoch, best_score
    if history:
        row = history[-1]
        return int(row.get("epoch") or 0), None
    return 0, None



def _training_device_name(torch: Any, device: str) -> str:
    if str(device).startswith("cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
            return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
        except Exception:
            return "cuda"
    return str(device)


def tsft_warmup_cosine_lr(
    *,
    epoch_number: int,
    max_epoch: int,
    max_lr: float,
    min_lr: float,
    max_decay_epoch: int,
) -> float:
    """Return the TSFT LR for an epoch: first epoch warmup, then cosine-to-min."""

    epoch_number = _positive_int(epoch_number, "epoch_number")
    max_epoch = _positive_int(max_epoch, "max_epoch")
    max_decay_epoch = _positive_int(max_decay_epoch, "max_decay_epoch")
    max_lr = _positive_float(max_lr, "max_lr")
    min_lr = float(min_lr)
    if min_lr < 0:
        raise TsftTrainingError("min_lr must be non-negative.")
    if max_lr <= min_lr:
        raise TsftTrainingError("max_lr must be greater than min_lr.")
    if max_decay_epoch > max_epoch:
        raise TsftTrainingError("max_decay_epoch must be less than or equal to max_epoch.")
    if epoch_number <= 1 or epoch_number >= max_decay_epoch or max_decay_epoch <= 2:
        return min_lr
    progress = (epoch_number - 2) / max(1, max_decay_epoch - 2)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (max_lr - min_lr) * (1 + cos(pi * progress)) / 2


def tsft_warmup_batch_lr(
    *,
    batch_index: int,
    total_batches: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return first-epoch warmup LR for a zero-based batch index."""

    total_batches = _positive_int(total_batches, "total_batches")
    max_lr = _positive_float(max_lr, "max_lr")
    min_lr = float(min_lr)
    if min_lr < 0:
        raise TsftTrainingError("min_lr must be non-negative.")
    if max_lr <= min_lr:
        raise TsftTrainingError("max_lr must be greater than min_lr.")
    if total_batches <= 1:
        return max_lr
    clipped_index = min(max(int(batch_index), 0), total_batches - 1)
    progress = clipped_index / (total_batches - 1)
    return min_lr + (max_lr - min_lr) * progress


def _tsft_metric_record(
    *,
    epoch: int,
    lr: float,
    train: Mapping[str, Any],
    test: Mapping[str, Any],
) -> dict[str, Any]:
    train_iou = _legacy_iou_array(train)
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "lr": float(lr),
        "train": dict(train),
        "train.loss": float(train.get("loss", 0.0)),
        "train.mIoU": float(train.get("mIoU", 0.0)),
        "train.IoU": train_iou,
        "train.IoU_background": _background_iou_value(train),
        "train.IoU_foreground": _foreground_iou_value(train),
        "train.kappa": float(train.get("kappa", train.get("Kappa", 0.0))),
    }
    if test:
        test_iou = _legacy_iou_array(test)
        record["eval"] = dict(test)
        record["eval.loss"] = float(test.get("loss", 0.0))
        record["eval.mIoU"] = float(test.get("mIoU", 0.0))
        record["eval.IoU"] = test_iou
        record["eval.IoU_background"] = _background_iou_value(test)
        record["eval.IoU_foreground"] = _foreground_iou_value(test)
        record["eval.kappa"] = float(test.get("kappa", test.get("Kappa", 0.0)))
        record["test"] = dict(test)
        record["test.loss"] = record["eval.loss"]
        record["test.mIoU"] = record["eval.mIoU"]
        record["test.IoU"] = test_iou
        record["test.IoU_background"] = record["eval.IoU_background"]
        record["test.IoU_foreground"] = record["eval.IoU_foreground"]
        record["test.kappa"] = record["eval.kappa"]
    return record


def _rename_eval_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    iou = metrics.get("IoU")
    iou_array = [float(value) for value in iou] if isinstance(iou, Sequence) and not isinstance(iou, (str, bytes)) else [
        float(metrics.get("IoU_background", 0.0)),
        float(metrics.get("IoU_foreground", metrics.get("IoU_landslide", 0.0))),
    ]
    return {
        "loss": float(metrics.get("test_loss", 0.0)),
        "mIoU": float(metrics.get("mIoU", 0.0)),
        "IoU": iou_array,
        "IoU_background": float(metrics.get("IoU_background", iou_array[0] if iou_array else 0.0)),
        "IoU_foreground": float(metrics.get("IoU_foreground", metrics.get("IoU_landslide", iou_array[1] if len(iou_array) > 1 else 0.0))),
        "IoU_landslide": float(metrics.get("IoU_landslide", metrics.get("IoU_foreground", iou_array[1] if len(iou_array) > 1 else 0.0))),
        "pixel_accuracy": float(metrics.get("pixel_accuracy", metrics.get("PA", 0.0))),
        "PA": float(metrics.get("PA", metrics.get("pixel_accuracy", 0.0))),
        "class_pixel_accuracy": list(metrics.get("class_pixel_accuracy", metrics.get("CPA", [])) or []),
        "mean_pixel_accuracy": float(metrics.get("mean_pixel_accuracy", metrics.get("MPA", 0.0))),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "F1": float(metrics.get("F1", 0.0)),
        "kappa": float(metrics.get("kappa", metrics.get("Kappa", 0.0))),
        "Kappa": float(metrics.get("Kappa", metrics.get("kappa", 0.0))),
    }


def _legacy_iou_array(metrics: Mapping[str, Any]) -> list[float]:
    iou = metrics.get("IoU")
    if isinstance(iou, Sequence) and not isinstance(iou, (str, bytes)):
        return [float(value) for value in iou]
    return [
        float(metrics.get("IoU_background", 0.0)),
        float(metrics.get("IoU_foreground", metrics.get("IoU_landslide", iou or 0.0))),
    ]


def _foreground_iou_value(metrics: Mapping[str, Any]) -> float:
    for key in ("IoU_foreground", "IoU_landslide"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    iou = metrics.get("IoU")
    if isinstance(iou, Sequence) and not isinstance(iou, (str, bytes)):
        if len(iou) > 1:
            return float(iou[1])
        if len(iou) == 1:
            return float(iou[0])
        return 0.0
    return float(iou or 0.0)


def _background_iou_value(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("IoU_background")
    if value is not None:
        return float(value)
    iou = metrics.get("IoU")
    if isinstance(iou, Sequence) and not isinstance(iou, (str, bytes)):
        return float(iou[0]) if iou else 0.0
    return 0.0


def _metrics_history_with_current(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in history]
    epoch = int(current.get("epoch") or 0)
    if rows and int(rows[-1].get("epoch") or 0) == epoch:
        rows[-1] = dict(current)
    else:
        rows.append(dict(current))
    return rows


def _tsft_epoch_log_line(
    *,
    epoch_number: int,
    total_epochs: int,
    train: Mapping[str, float],
    test: Mapping[str, float],
    best_epoch: int,
    best_score: float | None,
) -> str:
    parts = [
        f"[Epoch {epoch_number}/{total_epochs}]",
        f"train_loss={float(train.get('loss', 0.0)):.4f}",
        f"train_mIoU={float(train.get('mIoU', 0.0)):.4f}",
        f"train_IoU={_foreground_iou_value(train):.4f}",
    ]
    if test:
        parts.extend(
            [
                f"eval_loss={float(test.get('loss', 0.0)):.4f}",
                f"eval_mIoU={float(test.get('mIoU', 0.0)):.4f}",
                f"eval_IoU={_foreground_iou_value(test):.4f}",
            ]
        )
    if best_score is not None:
        parts.append(f"best={float(best_score):.4f}@epoch{best_epoch}")
    else:
        parts.append(f"checkpoint=epoch{best_epoch}")
    return " | ".join(parts)


def _tsft_metrics_summary(
    *,
    best_score: float | None,
    best_epoch: int,
    has_test: bool,
    train_count: int,
    test_count: int,
) -> str:
    if best_score is None:
        summary = f"selection=final_epoch;epoch={best_epoch}"
    elif has_test:
        summary = f"best_eval_mIoU={best_score:.4f};best_epoch={best_epoch}"
    else:
        summary = f"best_train_mIoU={best_score:.4f};best_epoch={best_epoch}"
    return f"{summary};train_samples={train_count};eval_samples={test_count}"
