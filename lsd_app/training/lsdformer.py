"""LSDFormer DAFT/TSFT training backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsd_app.core.channels import extract_rgb_bands
from lsd_app.core.paths import resolve_user_path
from lsd_app.models.lsdformer import (
    DEFAULT_LSDFORMER_MODEL_ROOT,
    LSDFORMER_TARGET_MODULES,
    LSDFormerFeatureWrapper,
    apply_lsdformer_hybrid_adapters,
    build_lsdformer_model,
    infer_lsdformer_feature_channels,
    merge_lsdformer_adapters,
    set_optimizer_group_lrs,
    trainable_lsdformer_param_groups,
)
from lsd_app.models.registry import normalize_model_name
from lsd_app.registry import register_weight
from lsd_app.training.daft_lsdsam import (
    DaftDatasetMaterialization,
    DaftLSDSAMConfig,
    DaftLSDSAMResult,
    DaftTrainingError,
    DaftTrainingSample,
    _DomainDiscriminator,
    _EpochStats,
    _FocalDiceLoss,
    _alignment_loss_fn,
    _append_jsonl,
    _best_daft_epoch_score_from_history,
    _ceil_batches,
    _checkpoint_epoch,
    _checkpoint_epoch_score,
    _checkpoint_model_state,
    _collate_images,
    _collate_supervised,
    _effective_daft_num_workers,
    _evaluate_model,
    _emit_batch_subphase,
    _load_daft_checkpoint,
    _lookup_weight,
    _make_loader,
    _metrics_summary,
    _optional_summary_writer,
    _progress_emitter,
    _read_label_mask,
    _read_raster_hwc,
    _require_module,
    _set_global_seeds,
    _save_training_config_snapshot,
    TRAINING_CONFIG_SNAPSHOT_FILENAME,
    _training_device_name,
    _utc_compact,
    learning_rate_for_epoch,
    resolve_daft_dataset_materialization,
    resolve_training_device,
)


LSDFORMER_ADAPTER_PROFILE = "lsdformer-hybrid-linear-lora-mb3-conv-lora"


class LSDFormerTrainingError(DaftTrainingError):
    """Raised when LSDFormer training cannot be configured or launched."""


class _LSDFormerSupervisedDataset:
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
            raise LSDFormerTrainingError(f"Supervised LSDFormer sample has no label: {sample.sample_id}")
        image = _read_raster_hwc(sample.image_path, np=self.np)
        mask = _read_label_mask(
            sample.label_path,
            image_shape=image.shape[:2],
            image_size=self.image_size,
            label_polarity=sample.label_polarity,
            cv2=self.cv2,
            np=self.np,
        )
        tensor = _preprocess_lsdformer_image(
            image,
            RGB_bands=sample.RGB_bands,
            image_size=self.image_size,
            cv2=self.cv2,
            torch=self.torch,
            np=self.np,
        )
        one_hot = self.np.stack([1 - mask, mask], axis=0).astype("float32")
        return tensor, self.torch.as_tensor(one_hot, dtype=self.torch.float32), sample.sample_id


class _LSDFormerImagesDataset:
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
        tensor = _preprocess_lsdformer_image(
            image,
            RGB_bands=sample.RGB_bands,
            image_size=self.image_size,
            cv2=self.cv2,
            torch=self.torch,
            np=self.np,
        )
        return tensor, sample.sample_id


def _lsdformer_dataset_from_materialization(
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
            _lsdformer_dataset_from_materialization(
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


def _preprocess_lsdformer_image(
    image: Any,
    *,
    RGB_bands: str,
    image_size: int,
    cv2: Any,
    torch: Any,
    np: Any,
) -> Any:
    values = np.asarray(image, dtype=np.float32)
    chw = np.moveaxis(values, -1, 0)
    rgb = extract_rgb_bands(chw, RGB_bands, layout="chw")
    rgb_hwc = np.moveaxis(rgb, 0, -1)
    if rgb_hwc.shape[2] < 3:
        rgb_hwc = np.repeat(rgb_hwc[:, :, :1], 3, axis=2)
    rgb_hwc = _ensure_unit_range(rgb_hwc[:, :, :3], np=np)
    resized = _resize_and_pad(rgb_hwc, image_size, cv2=cv2, np=np)
    normalized = _imagenet_normalize(resized, np=np)
    return torch.as_tensor(np.moveaxis(normalized, -1, 0).astype("float32", copy=False), dtype=torch.float32)


def _ensure_unit_range(image: Any, *, np: Any) -> Any:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(finite.max()) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_and_pad(image: Any, image_size: int, *, cv2: Any, np: Any) -> Any:
    height, width = image.shape[:2]
    scale = image_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_height, new_width) != (height, width):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((image_size, image_size, image.shape[2]), dtype=np.float32)
    padded[:new_height, :new_width, :] = image
    return padded


def _imagenet_normalize(image: Any, *, np: Any) -> Any:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (np.asarray(image, dtype=np.float32) - mean) / std


def _build_lsdformer_for_training(
    *,
    config: DaftLSDSAMConfig,
    weight_row: Mapping[str, Any],
    torch: Any,
    device: str,
) -> tuple[Any, Path, dict[str, int]]:
    weight_path = resolve_user_path(weight_row.get("path") or "", workspace_root=weight_row.get("_registry_root"))
    if not weight_path.exists():
        raise LSDFormerTrainingError(f"Base LSDFormer weight path does not exist: {weight_path}")
    model_root_value = config.legacy_root or weight_row.get("model_root") or weight_row.get("legacy_root") or DEFAULT_LSDFORMER_MODEL_ROOT
    model = build_lsdformer_model(
        torch,
        weight_path,
        model_root=model_root_value,
        options={"image_size": config.image_size},
        map_location="cpu",
    )
    conv_r = max(1, int(config.lora_task_r) // 2)
    conv_alpha = max(1, int(config.lora_task_alpha) // 2)
    counts = apply_lsdformer_hybrid_adapters(
        torch,
        model,
        linear_r=int(config.lora_task_r),
        linear_alpha=int(config.lora_task_alpha),
        conv_r=conv_r,
        conv_alpha=conv_alpha,
        dropout=float(config.lora_dropout),
    )
    wrapper = LSDFormerFeatureWrapper(torch, model).to(device)
    return wrapper, weight_path, counts


def _save_lsdformer_adapter_checkpoint(
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
    adapter_counts: Mapping[str, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "mIoU": float(score) if score is not None else None,
        "model_state_dict": model.state_dict(),
        "daft_config": _json_ready(asdict(config)),
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target.dataset_id,
        "adapter_profile": LSDFORMER_ADAPTER_PROFILE,
        "adapter_counts": dict(adapter_counts or {}),
    }
    if bool(getattr(config, "save_optimizer_state", False)) and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["optimizer_seg_state_dict"] = optimizer.state_dict()
    if bool(getattr(config, "save_optimizer_state", False)) and optimizer_dis is not None:
        payload["optimizer_dis_state_dict"] = optimizer_dis.state_dict()
    _ensure_torch_serialization_tls(torch)
    tmp_path = path.parent / f".{path.name}.{__import__('uuid').uuid4().hex}.tmp"
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_lsdformer_merged_checkpoint(
    torch: Any,
    checkpoint_to_merge: Path,
    output_path: Path,
    *,
    model: Any,
    device: str,
    metadata: Mapping[str, Any],
) -> None:
    payload = _load_daft_checkpoint(torch, checkpoint_to_merge, device)
    model.load_state_dict(_checkpoint_model_state(payload), strict=False)
    base_model = getattr(model, "model", model)
    merge_lsdformer_adapters(base_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_torch_serialization_tls(torch)
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "model_id": "LSDFormer",
            "weight_type": "merged",
            "adapter_profile": LSDFORMER_ADAPTER_PROFILE,
            **dict(metadata),
        },
        output_path,
    )


def _ensure_torch_serialization_tls(torch: Any) -> None:
    """Fill PyTorch serialization TLS defaults missing in some nightly builds."""

    serialization = getattr(torch, "serialization", None)
    tls = getattr(serialization, "_serialization_tls", None)
    if tls is None:
        return
    if not hasattr(tls, "check_serializing_named_tensor"):
        setattr(tls, "check_serializing_named_tensor", False)


def _load_lsdformer_loss_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        losses = row.get("losses") if isinstance(row, Mapping) else None
        if isinstance(losses, Mapping):
            rows.append({"epoch": int(row.get("epoch") or len(rows) + 1), **{key: float(value) for key, value in losses.items() if isinstance(value, (int, float))}})
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def run_lsdformer_daft(
    config: DaftLSDSAMConfig,
    *,
    progress_callback: Any | None = None,
) -> DaftLSDSAMResult:
    """Run LSDFormer DAFT training and register a merged full checkpoint."""

    cfg = config.normalized()
    if normalize_model_name(cfg.model_id) not in {"LSDFormer", "lsdformer-fast"}:
        raise LSDFormerTrainingError("LSDFormer DAFT requires model_id='LSDFormer'.")
    torch = _require_module("torch", "Install torch before launching LSDFormer DAFT.")
    np = _require_module("numpy", "Install numpy before launching LSDFormer DAFT.")
    cv2 = _require_module("cv2", "Install opencv-python before launching LSDFormer DAFT.")
    SummaryWriter = _optional_summary_writer()

    device = resolve_training_device(cfg.device, torch)
    device_name = _training_device_name(torch, device)
    num_workers = _effective_daft_num_workers(cfg.num_workers, device)
    _set_global_seeds(torch, np, cfg.seed)

    source = resolve_daft_dataset_materialization(registry_root=cfg.registry_root, dataset_id=cfg.source_dataset_id)
    target = resolve_daft_dataset_materialization(registry_root=cfg.registry_root, dataset_id=cfg.target_dataset_id)
    evaluation = (
        resolve_daft_dataset_materialization(registry_root=cfg.registry_root, dataset_id=cfg.evaluation_dataset_id)
        if cfg.evaluation_dataset_id
        else None
    )
    if evaluation is not None and evaluation.dataset_class != "Evaluation":
        raise LSDFormerTrainingError(f"Evaluation dataset {evaluation.dataset_id} is {evaluation.dataset_class}, expected Evaluation.")
    source_samples = source.supervised_train_samples()
    target_samples = target.target_train_samples()
    evaluation_samples = evaluation.evaluation_samples() if evaluation is not None else ()
    if not source_samples:
        raise LSDFormerTrainingError(f"Source dataset {source.dataset_id} has no supervised train samples.")
    if not target_samples:
        raise LSDFormerTrainingError(f"Target dataset {target.dataset_id} has no image samples.")

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
        kind="daft-training",
        latest_checkpoint_path=checkpoints_dir / "lsdformer_daft_latest_adapter.pth",
        best_checkpoint_path=checkpoints_dir / "lsdformer_daft_best_adapter.pth",
        metrics_path=metrics_path,
        selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint",
    )
    emit({"status": "running", "percent": 0.02, "phase": "dataset", "message": "Resolved LSDFormer source/target datasets.", "source_sample_count": len(source_samples), "target_sample_count": len(target_samples), "evaluation_sample_count": len(evaluation_samples), "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})

    weight_row = _lookup_weight(registry_root=cfg.registry_root, weight_id=cfg.base_weight_id)
    model, base_checkpoint, adapter_counts = _build_lsdformer_for_training(config=cfg, weight_row=weight_row, torch=torch, device=device)
    feature_channels = infer_lsdformer_feature_channels(model)
    emit({"status": "running", "percent": 0.06, "phase": "model", "message": f"LSDFormer hybrid adapters are ready; feature channels={feature_channels}.", "adapter_counts": adapter_counts, "device": device, "device_name": device_name})

    source_dataset = _lsdformer_dataset_from_materialization(
        torch,
        source,
        sample_selector="supervised_train_samples",
        dataset_cls=_LSDFormerSupervisedDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    target_dataset = _lsdformer_dataset_from_materialization(
        torch,
        target,
        sample_selector="target_train_samples",
        dataset_cls=_LSDFormerImagesDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    eval_dataset = (
        _lsdformer_dataset_from_materialization(
            torch,
            evaluation,
            sample_selector="evaluation_samples",
            dataset_cls=_LSDFormerSupervisedDataset,
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
    discriminator = _DomainDiscriminator(torch, in_channels=feature_channels).to(device)
    param_groups = trainable_lsdformer_param_groups(model, base_lr=cfg.initial_lr, conv_lr_scale=0.5)
    if not param_groups:
        raise LSDFormerTrainingError("No trainable LSDFormer parameters were found for DAFT.")
    opt_seg = torch.optim.AdamW(param_groups, lr=cfg.initial_lr)
    opt_dis = torch.optim.AdamW(discriminator.parameters(), lr=cfg.discriminator_lr or cfg.initial_lr * 2.5)
    writer = SummaryWriter(log_dir=str(logs_dir)) if SummaryWriter is not None else None

    best_score: float | None = None
    best_epoch = 0
    best_checkpoint = checkpoints_dir / "lsdformer_daft_best_adapter.pth"
    latest_checkpoint = checkpoints_dir / "lsdformer_daft_latest_adapter.pth"
    total_batches = max(1, min(_ceil_batches(len(source_dataset), cfg.batch_size), _ceil_batches(len(target_dataset), cfg.batch_size)))
    global_total = max(1, cfg.total_epochs * total_batches)
    processed = 0
    loss_history = _load_lsdformer_loss_history(metrics_path) if cfg.resume else []
    start_epoch_index = 0
    if cfg.resume:
        resume_checkpoint = Path(cfg.resume_checkpoint_path).expanduser().resolve() if cfg.resume_checkpoint_path else latest_checkpoint
        if not resume_checkpoint.exists():
            raise LSDFormerTrainingError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_payload = _load_daft_checkpoint(torch, resume_checkpoint, device)
        model.load_state_dict(_checkpoint_model_state(resume_payload), strict=False)
        if isinstance(resume_payload.get("optimizer_state_dict"), Mapping):
            opt_seg.load_state_dict(resume_payload["optimizer_state_dict"])
        if isinstance(resume_payload.get("optimizer_dis_state_dict"), Mapping):
            opt_dis.load_state_dict(resume_payload["optimizer_dis_state_dict"])
        start_epoch_index = min(max(_checkpoint_epoch(resume_payload), 0), cfg.total_epochs)
        processed = start_epoch_index * total_batches
        best_epoch, best_score = _checkpoint_epoch_score(torch, best_checkpoint, device)
        if best_epoch <= 0:
            best_epoch, best_score = _best_daft_epoch_score_from_history(loss_history, has_eval=bool(evaluation_samples))

    try:
        for epoch_index in range(start_epoch_index, cfg.total_epochs):
            epoch_number = epoch_index + 1
            generator.manual_seed(cfg.seed + epoch_index)
            seg_lr = cfg.initial_lr if epoch_number == 1 else learning_rate_for_epoch(epoch_number=epoch_number, initial_lr=cfg.initial_lr, halve_epochs=cfg.halve_epochs, min_lr=cfg.min_lr)
            dis_base_lr = cfg.discriminator_lr or cfg.initial_lr * 2.5
            dis_lr = dis_base_lr if epoch_number == 1 else learning_rate_for_epoch(epoch_number=epoch_number, initial_lr=dis_base_lr, halve_epochs=cfg.halve_epochs, min_lr=cfg.min_lr)
            set_optimizer_group_lrs(opt_seg, seg_lr)
            for group in opt_dis.param_groups:
                group["lr"] = dis_lr

            source_loader = _make_loader(torch, source_dataset, batch_size=cfg.batch_size, num_workers=num_workers, generator=generator, collate_fn=_collate_supervised(torch))
            target_loader = _make_loader(torch, target_dataset, batch_size=cfg.batch_size, num_workers=num_workers, generator=generator, collate_fn=_collate_images(torch))
            model.train()
            discriminator.train()
            epoch_stats = _EpochStats()
            epoch_batch_losses: list[dict[str, Any]] = []
            for batch_index, ((src_imgs, src_masks, _src_ids), (tgt_imgs, _tgt_ids)) in enumerate(zip(source_loader, target_loader)):
                batch_display = batch_index + 1
                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="batch_loaded", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])
                if epoch_number == 1:
                    warmup_fraction = (batch_index + 1) / max(1, total_batches)
                    set_optimizer_group_lrs(opt_seg, max(cfg.min_lr, cfg.initial_lr * warmup_fraction))
                    for group in opt_dis.param_groups:
                        group["lr"] = max(cfg.min_lr, dis_base_lr * warmup_fraction)
                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="to_device", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])
                src_imgs = src_imgs.to(device).float()
                src_masks = src_masks.to(device).float()
                tgt_imgs = tgt_imgs.to(device).float()
                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="forward", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])
                src_logits, src_feats = model(src_imgs, return_feats=True)
                _tgt_logits, tgt_feats = model(tgt_imgs, return_feats=True)
                src_targets = src_masks.argmax(dim=1)
                loss_seg, _dice_loss, _focal_loss = criterion_seg(src_logits, src_targets)

                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="discriminator_step", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])
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
                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="adapter_step", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])
                opt_seg.zero_grad()
                pred_src = discriminator(src_feats)
                pred_tgt = discriminator(tgt_feats)
                loss_adv = 0.5 * (
                    criterion_adv(pred_src, torch.ones(pred_src.shape[0], dtype=torch.long, device=device))
                    + criterion_adv(pred_tgt, torch.zeros(pred_tgt.shape[0], dtype=torch.long, device=device))
                )
                loss_align = align_loss_fn(src_feats, tgt_feats)
                total_loss = loss_seg + cfg.adv_lambda * loss_adv + cfg.align_lambda * loss_align
                total_loss.backward()
                opt_seg.step()
                for param in discriminator.parameters():
                    param.requires_grad = True
                _emit_batch_subphase(emit, "LSDFormer DAFT", epoch_number=epoch_number, total_epochs=cfg.total_epochs, batch_index=batch_display, total_batches=total_batches, subphase="optimizer_done", processed=processed + 1, total=global_total, device=device, device_name=device_name, source_ids=_src_ids, target_ids=_tgt_ids, lr_seg=opt_seg.param_groups[0]["lr"], lr_dis=opt_dis.param_groups[0]["lr"])

                batch_losses = {"seg": float(loss_seg.detach().cpu().item()), "adv": float(loss_adv.detach().cpu().item()), "dis": float(loss_dis.detach().cpu().item()), "align": float(loss_align.detach().cpu().item()), "total": float(total_loss.detach().cpu().item())}
                epoch_stats.update(**batch_losses)
                epoch_batch_losses.append({"batch": batch_index + 1, "lr_seg": float(opt_seg.param_groups[0]["lr"]), "lr_dis": float(opt_dis.param_groups[0]["lr"]), **batch_losses})
                processed += 1
                if batch_index == 0 or (batch_index + 1) % 20 == 0 or (batch_index + 1) == total_batches:
                    losses = epoch_stats.to_dict()
                    emit({"status": "running", "percent": min(0.95, processed / global_total), "phase": "train", "processed": processed, "total": global_total, "current_epoch": epoch_number, "total_epochs": cfg.total_epochs, "current_batch": batch_display, "total_batches": total_batches, "lr_seg": opt_seg.param_groups[0]["lr"], "lr_dis": opt_dis.param_groups[0]["lr"], "device": device, "device_name": device_name, "losses": losses, "latest_log": f"[LSDFormer DAFT {epoch_number}/{cfg.total_epochs}] batch {batch_display}/{total_batches} total={losses.get('total', 0.0):.4f} seg={losses.get('seg', 0.0):.4f} adv={losses.get('adv', 0.0):.4f} {cfg.alignment_mode}={losses.get('align', 0.0):.6f}"})

            epoch_losses = epoch_stats.to_dict()
            loss_history.append({"epoch": epoch_number, **epoch_losses})
            eval_metrics = _evaluate_model(model, eval_dataset=eval_dataset, torch=torch, device=device, criterion_seg=criterion_seg, batch_size=cfg.batch_size, num_workers=num_workers, progress_callback=emit, progress_context={"phase": "evaluation", "evaluation_split": "evaluation", "progress_offset": processed, "total": global_total, "current_epoch": epoch_number, "total_epochs": cfg.total_epochs, "current_batch": total_batches, "total_batches": total_batches, "lr_seg": opt_seg.param_groups[0]["lr"], "lr_dis": opt_dis.param_groups[0]["lr"], "device": device, "device_name": device_name}) if eval_dataset is not None else {}
            if eval_metrics and "mIoU" in eval_metrics:
                score = float(eval_metrics["mIoU"])
                is_best = best_score is None or score > best_score
            else:
                score = None
                is_best = epoch_number == cfg.total_epochs
            if is_best:
                best_score = score
                best_epoch = epoch_number
                emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving LSDFormer DAFT best checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
                _save_lsdformer_adapter_checkpoint(torch, best_checkpoint, model=model, config=cfg, epoch=epoch_number, score=score, source=source, target=target, optimizer=opt_seg, optimizer_dis=opt_dis, adapter_counts=adapter_counts)
            emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving LSDFormer DAFT latest checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
            _save_lsdformer_adapter_checkpoint(torch, latest_checkpoint, model=model, config=cfg, epoch=epoch_number, score=score, source=source, target=target, optimizer=opt_seg, optimizer_dis=opt_dis, adapter_counts=adapter_counts)
            _save_training_config_snapshot(config_snapshot_path, config=cfg, kind="daft-training", latest_checkpoint_path=latest_checkpoint, best_checkpoint_path=best_checkpoint, metrics_path=metrics_path, selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint", current_epoch=epoch_number, best_epoch=best_epoch, best_score=best_score)
            record = {"epoch": epoch_number, "lr_seg": opt_seg.param_groups[0]["lr"], "lr_dis": opt_dis.param_groups[0]["lr"], "losses": epoch_losses, "batch_losses": epoch_batch_losses, "evaluation": eval_metrics}
            _append_jsonl(metrics_path, record)
            emit({"status": "running", "percent": min(0.98, processed / global_total), "phase": "evaluation" if eval_metrics else "checkpoint", "current_epoch": epoch_number, "total_epochs": cfg.total_epochs, "losses": epoch_losses, "evaluation": eval_metrics, "best_epoch": best_epoch or epoch_number, "best_score": best_score, "selection_metric": "eval_mIoU" if eval_metrics else "last_checkpoint", "training_config_path": str(config_snapshot_path), "latest_checkpoint_path": str(latest_checkpoint), "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "", "message": f"LSDFormer DAFT epoch {epoch_number}/{cfg.total_epochs} completed."})
            if writer is not None:
                for name, value in epoch_losses.items():
                    writer.add_scalar(f"Epoch/{name}", value, epoch_number)
    finally:
        if writer is not None:
            writer.close()

    checkpoint_to_merge = best_checkpoint if evaluation_samples and best_checkpoint.exists() else latest_checkpoint
    merged_checkpoint = checkpoints_dir / "lsdformer_daft_merged.pth"
    _save_lsdformer_merged_checkpoint(torch, checkpoint_to_merge, merged_checkpoint, model=model, device=device, metadata={"epoch": best_epoch or cfg.total_epochs, "best_score": best_score, "stage": "DAFT"})
    registered = register_weight(
        registry_root=cfg.registry_root,
        model_id="LSDFormer",
        stage="DAFT",
        version=f"DAFT-{target.version or target.dataset_id}-{_utc_compact()}",
        path=merged_checkpoint,
        metrics_summary=_metrics_summary(best_score=best_score, best_epoch=best_epoch or cfg.total_epochs, target_has_gt=bool(evaluation_samples)),
        weight_id=cfg.output_weight_id or None,
        notes=f"LSDFormer DAFT from source={source.dataset_id}, target={target.dataset_id}, alignment={cfg.alignment_mode}, epochs={cfg.total_epochs}, device={device_name}.",
        weight_type="merged",
        base_checkpoint_path=base_checkpoint,
        adapter_profile=LSDFORMER_ADAPTER_PROFILE,
        adapter_name="LSDFormer",
        image_size=cfg.image_size,
        input_mode="rgb",
        channel_rule="RGB",
        lora_task_r=cfg.lora_task_r,
        lora_task_alpha=cfg.lora_task_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LSDFORMER_TARGET_MODULES,
    )
    result = DaftLSDSAMResult(output_dir=output_dir, lora_checkpoint_path=checkpoint_to_merge, merged_checkpoint_path=merged_checkpoint, registered_weight_id=registered.weight_id, registered_weight_path=registered.path, metrics_path=metrics_path, best_epoch=best_epoch or cfg.total_epochs, best_score=best_score, selection_metric="eval_mIoU" if evaluation_samples else "last_checkpoint", source_sample_count=len(source_samples), target_sample_count=len(target_samples), target_has_gt=bool(evaluation_samples), device=device_name)
    emit({"status": "running", "percent": 1.0, "phase": "register", "registered_weight_id": result.registered_weight_id, "registered_weight_path": str(result.registered_weight_path), "merged_checkpoint_path": str(result.merged_checkpoint_path), "metrics_path": str(result.metrics_path), "training_config_path": str(config_snapshot_path), "message": f"Registered LSDFormer DAFT weight {registered.weight_id}."})
    return result


def run_lsdformer_tsft(config: Any, *, progress_callback: Any | None = None) -> Any:
    """Run LSDFormer TSFT supervised fine-tuning and register a merged full checkpoint."""

    from lsd_app.training.tsft import (
        TsftTrainingResult,
        TsftTrainingError,
        _best_epoch_score_from_history,
        _cleanup_tsft_lora_checkpoints,
        _load_tsft_metrics_history,
        _rename_eval_metrics,
        _tsft_epoch_log_line,
        _tsft_metric_record,
        _tsft_metrics_summary,
        tsft_warmup_batch_lr,
        tsft_warmup_cosine_lr,
    )

    cfg = config.normalized()
    if normalize_model_name(cfg.model_id) not in {"LSDFormer", "lsdformer-fast"}:
        raise TsftTrainingError("LSDFormer TSFT requires model_id='LSDFormer'.")
    torch = _require_module("torch", "Install torch before launching LSDFormer TSFT.")
    np = _require_module("numpy", "Install numpy before launching LSDFormer TSFT.")
    cv2 = _require_module("cv2", "Install opencv-python before launching LSDFormer TSFT.")
    SummaryWriter = _optional_summary_writer()
    device = resolve_training_device(cfg.device, torch)
    device_name = _training_device_name(torch, device)
    _set_global_seeds(torch, np, cfg.seed)

    dataset = resolve_daft_dataset_materialization(registry_root=cfg.registry_root, dataset_id=cfg.dataset_id)
    if dataset.dataset_class.upper() != "TSFT":
        raise TsftTrainingError(f"Dataset {dataset.dataset_id} is {dataset.dataset_class}, expected TSFT.")
    evaluation = resolve_daft_dataset_materialization(registry_root=cfg.registry_root, dataset_id=cfg.evaluation_dataset_id) if cfg.evaluation_dataset_id else None
    if evaluation is not None and evaluation.dataset_class != "Evaluation":
        raise TsftTrainingError(f"Evaluation dataset {evaluation.dataset_id} is {evaluation.dataset_class}, expected Evaluation.")
    train_samples = dataset.supervised_train_samples()
    test_samples = evaluation.evaluation_samples() if evaluation is not None else ()
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
        latest_checkpoint_path=checkpoints_dir / "lsdformer_tsft_latest_adapter.pth",
        best_checkpoint_path=checkpoints_dir / "lsdformer_tsft_best_adapter.pth",
        metrics_path=metrics_path,
        selection_metric="eval_mIoU" if test_samples else "train_mIoU",
    )
    emit({"status": "running", "percent": 0.02, "phase": "dataset", "message": "Resolved LSDFormer TSFT dataset.", "train_sample_count": len(train_samples), "evaluation_sample_count": len(test_samples), "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})

    daft_cfg = cfg.as_daft_config()
    weight_row = _lookup_weight(registry_root=cfg.registry_root, weight_id=cfg.base_weight_id)
    model, base_checkpoint, adapter_counts = _build_lsdformer_for_training(config=daft_cfg, weight_row=weight_row, torch=torch, device=device)
    train_dataset = _lsdformer_dataset_from_materialization(
        torch,
        dataset,
        sample_selector="supervised_train_samples",
        dataset_cls=_LSDFormerSupervisedDataset,
        image_size=cfg.image_size,
        cv2=cv2,
        np=np,
    )
    test_dataset = (
        _lsdformer_dataset_from_materialization(
            torch,
            evaluation,
            sample_selector="evaluation_samples",
            dataset_cls=_LSDFormerSupervisedDataset,
            image_size=cfg.image_size,
            cv2=cv2,
            np=np,
        )
        if evaluation is not None and test_samples
        else None
    )

    criterion_seg = _FocalDiceLoss(torch)
    param_groups = trainable_lsdformer_param_groups(model, base_lr=cfg.max_lr, conv_lr_scale=0.25)
    if not param_groups:
        raise TsftTrainingError("No trainable LSDFormer parameters were found for TSFT.")
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.min_lr)
    writer = SummaryWriter(log_dir=str(logs_dir)) if SummaryWriter is not None else None
    generator = torch.Generator()

    best_score: float | None = None
    best_epoch = 0
    best_checkpoint = checkpoints_dir / "lsdformer_tsft_best_adapter.pth"
    latest_checkpoint = checkpoints_dir / "lsdformer_tsft_latest_adapter.pth"
    total_batches = _ceil_batches(len(train_dataset), cfg.batch_size)
    global_total = max(1, cfg.max_epoch * total_batches)
    processed = 0
    metrics_history = _load_tsft_metrics_history(metrics_path) if cfg.resume else []
    start_epoch_index = 0
    if cfg.resume:
        resume_checkpoint = Path(cfg.resume_checkpoint_path).expanduser().resolve() if cfg.resume_checkpoint_path else latest_checkpoint
        if not resume_checkpoint.exists():
            raise TsftTrainingError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        resume_payload = _load_daft_checkpoint(torch, resume_checkpoint, device)
        model.load_state_dict(_checkpoint_model_state(resume_payload), strict=False)
        if isinstance(resume_payload.get("optimizer_state_dict"), Mapping):
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch_index = min(max(_checkpoint_epoch(resume_payload), 0), cfg.max_epoch)
        processed = start_epoch_index * total_batches
        best_epoch, best_score = _checkpoint_epoch_score(torch, best_checkpoint, device)
        if best_epoch <= 0:
            best_epoch, best_score = _best_epoch_score_from_history(metrics_history, has_test=bool(test_samples))

    try:
        for epoch_index in range(start_epoch_index, cfg.max_epoch):
            epoch_number = epoch_index + 1
            generator.manual_seed(cfg.seed + epoch_index)
            lr = tsft_warmup_cosine_lr(epoch_number=epoch_number, max_epoch=cfg.max_epoch, max_lr=cfg.max_lr, min_lr=cfg.min_lr, max_decay_epoch=cfg.max_decay_epoch)
            set_optimizer_group_lrs(optimizer, lr)
            train_loader = _make_loader(torch, train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, generator=generator, collate_fn=_collate_supervised(torch))
            model.train()
            epoch_stats = _EpochStats()
            for batch_index, (images, masks, _ids) in enumerate(train_loader):
                batch_display = batch_index + 1
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="batch_loaded", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                if epoch_number == 1:
                    warmup_lr = tsft_warmup_batch_lr(batch_index=batch_index, total_batches=total_batches, max_lr=cfg.max_lr, min_lr=cfg.min_lr)
                    set_optimizer_group_lrs(optimizer, warmup_lr)
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="to_device", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                images = images.to(device).float()
                masks = masks.to(device).float()
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="forward", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                outputs = model(images)
                loss, dice_loss, focal_loss = criterion_seg(outputs, masks.argmax(dim=1))
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="backward", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                optimizer.zero_grad()
                loss.backward()
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="optimizer_step", processed=processed, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                optimizer.step()
                _emit_batch_subphase(emit, "LSDFormer TSFT", epoch_number=epoch_number, total_epochs=cfg.max_epoch, batch_index=batch_display, total_batches=total_batches, subphase="optimizer_done", processed=processed + 1, total=global_total, device=device, device_name=device_name, source_ids=_ids, lr=optimizer.param_groups[0]["lr"])
                epoch_stats.update(loss=float(loss.detach().cpu().item()), dice=float(dice_loss.detach().cpu().item()), focal=float(focal_loss.detach().cpu().item()))
                processed += 1
                if batch_index == 0 or (batch_index + 1) % 20 == 0 or (batch_index + 1) == total_batches:
                    current_train = {"loss": epoch_stats.avg("loss"), "dice": epoch_stats.avg("dice"), "focal": epoch_stats.avg("focal")}
                    emit({"status": "running", "percent": min(0.95, processed / global_total), "phase": "train", "processed": processed, "total": global_total, "current_epoch": epoch_number, "total_epochs": cfg.max_epoch, "current_batch": batch_display, "total_batches": total_batches, "lr": optimizer.param_groups[0]["lr"], "device": device, "device_name": device_name, "train": current_train, "latest_log": f"[LSDFormer TSFT {epoch_number}/{cfg.max_epoch}] batch {batch_display}/{total_batches} loss={current_train['loss']:.4f}"})

            train_metrics = _rename_eval_metrics(_evaluate_model(model, eval_dataset=train_dataset, torch=torch, device=device, criterion_seg=criterion_seg, batch_size=cfg.batch_size, num_workers=cfg.num_workers, progress_callback=emit, progress_context={"phase": "evaluation", "evaluation_split": "train", "progress_offset": processed, "total": global_total, "current_epoch": epoch_number, "total_epochs": cfg.max_epoch, "current_batch": total_batches, "total_batches": total_batches, "lr": optimizer.param_groups[0]["lr"], "device": device, "device_name": device_name}))
            test_metrics = _rename_eval_metrics(_evaluate_model(model, eval_dataset=test_dataset, torch=torch, device=device, criterion_seg=criterion_seg, batch_size=cfg.batch_size, num_workers=cfg.num_workers, progress_callback=emit, progress_context={"phase": "evaluation", "evaluation_split": "eval", "progress_offset": processed, "total": global_total, "current_epoch": epoch_number, "total_epochs": cfg.max_epoch, "current_batch": total_batches, "total_batches": total_batches, "lr": optimizer.param_groups[0]["lr"], "device": device, "device_name": device_name})) if test_dataset is not None else {}
            scoring_metrics = test_metrics if test_metrics else train_metrics
            score = float(scoring_metrics["mIoU"]) if "mIoU" in scoring_metrics else None
            is_best = score is not None and (best_score is None or score > best_score)
            if score is None:
                is_best = epoch_number == cfg.max_epoch
            if is_best:
                best_score = score
                best_epoch = epoch_number
                emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving LSDFormer TSFT best checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
                _save_lsdformer_adapter_checkpoint(torch, best_checkpoint, model=model, config=daft_cfg, epoch=epoch_number, score=score, source=dataset, target=dataset, optimizer=optimizer, adapter_counts=adapter_counts)
            emit({"status": "running", "phase": "checkpoint_saving", "progress_kind": "checkpoint", "current_epoch": epoch_number, "message": f"Saving LSDFormer TSFT latest checkpoint for epoch {epoch_number}.", "training_config_path": str(config_snapshot_path), "device": device, "device_name": device_name})
            _save_lsdformer_adapter_checkpoint(torch, latest_checkpoint, model=model, config=daft_cfg, epoch=epoch_number, score=score, source=dataset, target=dataset, optimizer=optimizer, adapter_counts=adapter_counts)
            _save_training_config_snapshot(config_snapshot_path, config=cfg, kind="tsft-training", latest_checkpoint_path=latest_checkpoint, best_checkpoint_path=best_checkpoint, metrics_path=metrics_path, selection_metric="eval_mIoU" if test_samples else "train_mIoU", current_epoch=epoch_number, best_epoch=best_epoch, best_score=best_score)
            record = _tsft_metric_record(epoch=epoch_number, lr=optimizer.param_groups[0]["lr"], train=train_metrics, test=test_metrics)
            _append_jsonl(metrics_path, record)
            metrics_history.append(record)
            emit({"status": "running", "percent": min(0.98, processed / global_total), "phase": "evaluation" if test_metrics else "checkpoint", "current_epoch": epoch_number, "total_epochs": cfg.max_epoch, "train": train_metrics, "test": test_metrics, "best_epoch": best_epoch or epoch_number, "best_score": best_score, "selection_metric": "eval_mIoU" if test_samples else "train_mIoU", "training_config_path": str(config_snapshot_path), "latest_checkpoint_path": str(latest_checkpoint), "best_checkpoint_path": str(best_checkpoint) if best_checkpoint.exists() else "", "latest_log": _tsft_epoch_log_line(epoch_number=epoch_number, total_epochs=cfg.max_epoch, train=train_metrics, test=test_metrics, best_epoch=best_epoch or epoch_number, best_score=best_score), "message": f"LSDFormer TSFT epoch {epoch_number}/{cfg.max_epoch} completed."})
            if writer is not None:
                writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch_number)
    finally:
        if writer is not None:
            writer.close()

    checkpoint_to_merge = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
    merged_checkpoint = checkpoints_dir / "lsdformer_tsft_merged.pth"
    _save_lsdformer_merged_checkpoint(torch, checkpoint_to_merge, merged_checkpoint, model=model, device=device, metadata={"epoch": best_epoch or cfg.max_epoch, "best_score": best_score, "stage": "TSFT"})
    registered = register_weight(
        registry_root=cfg.registry_root,
        model_id="LSDFormer",
        stage="TSFT",
        version=f"TSFT-{dataset.version or dataset.dataset_id}-{_utc_compact()}",
        path=merged_checkpoint,
        metrics_summary=_tsft_metrics_summary(best_score=best_score, best_epoch=best_epoch or cfg.max_epoch, has_test=bool(test_samples), train_count=len(train_samples), test_count=len(test_samples)),
        weight_id=cfg.output_weight_id or None,
        notes=f"LSDFormer TSFT supervised fine-tuning on dataset={dataset.dataset_id}, epochs={cfg.max_epoch}, device={device_name}.",
        weight_type="merged",
        base_checkpoint_path=base_checkpoint,
        adapter_profile=LSDFORMER_ADAPTER_PROFILE,
        adapter_name="LSDFormer",
        image_size=cfg.image_size,
        input_mode="rgb",
        channel_rule="RGB",
        lora_task_r=cfg.lora_task_r,
        lora_task_alpha=cfg.lora_task_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LSDFORMER_TARGET_MODULES,
    )
    cleanup_removed = _cleanup_tsft_lora_checkpoints(best_checkpoint, latest_checkpoint) if cfg.cleanup_lora_checkpoints_after_register else []
    result = TsftTrainingResult(output_dir=output_dir, lora_checkpoint_path=checkpoint_to_merge, merged_checkpoint_path=merged_checkpoint, registered_weight_id=registered.weight_id, registered_weight_path=registered.path, metrics_path=metrics_path, best_epoch=best_epoch or cfg.max_epoch, best_score=best_score, selection_metric="eval_mIoU" if test_samples else "train_mIoU", train_sample_count=len(train_samples), test_sample_count=len(test_samples), device=device_name, cleanup_removed_checkpoint_paths=cleanup_removed)
    emit({"status": "running", "percent": 1.0, "phase": "register", "registered_weight_id": result.registered_weight_id, "registered_weight_path": str(result.registered_weight_path), "merged_checkpoint_path": str(result.merged_checkpoint_path), "training_config_path": str(config_snapshot_path), "cleanup_removed_checkpoint_paths": cleanup_removed, "message": f"Registered LSDFormer TSFT weight {registered.weight_id}."})
    return result
