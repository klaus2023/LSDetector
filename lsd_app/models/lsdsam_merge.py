"""Utilities for exporting merged LSDSAM inference checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lsd_app.models.base import ModelAdapterError
from lsd_app.models.lsdsam import (
    _extract_checkpoint_state_dict,
    _import_legacy_build_module,
    _is_merged_weight_type,
    _infer_lora_rank,
    _attach_lora_adapter,
    _load_base_sam_checkpoint,
    _lora_adapter_names,
    _resolve_lora_alpha,
    _resolve_base_checkpoint_for_state,
    _resolve_lora_adapter_name,
    _select_device,
    _torch_load,
)


def merge_lsdsam_lora_checkpoint(
    *,
    lora_checkpoint_path: str | Path,
    output_path: str | Path,
    legacy_root: str | Path,
    sam_checkpoint_path: str | Path | None = None,
    image_size: int = 512,
    device: str = "cpu",
    lora_base_r: int = 32,
    lora_base_alpha: int = 64,
    lora_task_r: int = 16,
    lora_task_alpha: int = 32,
    lora_dropout: float = 0.05,
    peft_adapter_name: str = "",
    sam_variant: str = "B",
) -> Path:
    """Merge a legacy LSDSAM LoRA checkpoint into a plain inference state_dict."""

    torch = _require_module("torch")
    peft = _require_module("peft")
    selected_device = _select_device(torch, str(device).lower())
    lora_checkpoint = Path(lora_checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    root = Path(legacy_root).expanduser().resolve()

    build_module = _import_legacy_build_module(root)
    variant = str(sam_variant or "B").strip().lower()
    if variant not in {"h", "l", "b"}:
        variant = "b"
    build_sam = getattr(build_module, f"build_sam_{variant}", None)
    if build_sam is None:
        raise ModelAdapterError(f"Legacy LSDSAM build module does not expose build_sam_{variant}.")

    checkpoint_payload = _torch_load(torch, lora_checkpoint, "cpu")
    checkpoint_state = _extract_checkpoint_state_dict(checkpoint_payload, lora_checkpoint)
    if _is_merged_weight_type("", checkpoint_state):
        output.parent.mkdir(parents=True, exist_ok=True)
        _ensure_torch_serialization_tls(torch)
        torch.save(_merged_payload(checkpoint_state, checkpoint_payload, source=lora_checkpoint), output)
        return output
    options = {"sam_checkpoint_path": str(sam_checkpoint_path)} if sam_checkpoint_path else {}
    base_checkpoint = _resolve_base_checkpoint_for_state(
        build_module,
        root,
        options,
        checkpoint_state,
        source=lora_checkpoint,
    )

    model = build_sam(checkpoint=False, map_location="cpu")
    if base_checkpoint is not None:
        _load_base_sam_checkpoint(torch, model, base_checkpoint, "cpu")
    model.image_encoder.resize_pos_embed(new_img_size=int(image_size))

    matched_modules = _matched_lora_modules(model)
    base_config = peft.LoraConfig(
        r=int(lora_base_r),
        lora_alpha=int(lora_base_alpha),
        target_modules=matched_modules,
        lora_dropout=float(lora_dropout),
        bias="none",
    )
    adapter_name = _resolve_lora_adapter_name(checkpoint_state, preferred=peft_adapter_name)
    task_r = _infer_lora_rank(checkpoint_state, adapter_name=adapter_name, fallback=int(lora_task_r))
    task_config = peft.LoraConfig(
        r=task_r,
        lora_alpha=_resolve_lora_alpha(
            {
                "lora_base_alpha": lora_base_alpha,
                "lora_task_alpha": lora_task_alpha,
            },
            adapter_name=adapter_name,
            fallback=int(lora_task_alpha),
        ),
        target_modules=matched_modules,
        lora_dropout=float(lora_dropout),
        bias="none",
    )
    if adapter_name == "default":
        model = peft.get_peft_model(model, task_config, adapter_name=adapter_name)
    else:
        model = peft.get_peft_model(model, base_config, adapter_name="default")
        model.merge_and_unload()
        model = _attach_lora_adapter(peft, model, adapter_name, task_config)
    model.load_state_dict(checkpoint_state)
    merged = model.merge_and_unload()
    merged = merged.to("cpu")
    merged.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_torch_serialization_tls(torch)
    torch.save(
        _merged_payload(
            merged.state_dict(),
            checkpoint_payload,
            source=lora_checkpoint,
            adapter_name=adapter_name,
            original_adapter_names=sorted(_lora_adapter_names(checkpoint_state)),
            image_size=int(image_size),
        ),
        output,
    )
    return output


def _ensure_torch_serialization_tls(torch: Any) -> None:
    """Fill PyTorch serialization TLS defaults missing in some nightly builds."""

    serialization = getattr(torch, "serialization", None)
    tls = getattr(serialization, "_serialization_tls", None)
    if tls is None:
        return
    if not hasattr(tls, "check_serializing_named_tensor"):
        setattr(tls, "check_serializing_named_tensor", False)


def _matched_lora_modules(model: Any) -> list[str]:
    return [
        name
        for name, _module in model.named_modules()
        if name.startswith("image_encoder.blocks.")
        and (
            name.endswith("attn.qkv")
            or name.endswith("attn.proj")
            or name.endswith("mlp.lin1")
            or name.endswith("mlp.lin2")
        )
    ]


def _merged_payload(
    state_dict: Mapping[str, Any],
    source_payload: Any,
    *,
    source: Path,
    adapter_name: str = "",
    original_adapter_names: list[str] | None = None,
    image_size: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": dict(state_dict),
        "weight_type": "merged",
        "source_checkpoint": str(source),
    }
    if isinstance(source_payload, Mapping):
        for key in ("epoch", "best_score", "metrics", "metrics_summary"):
            if key in source_payload:
                payload[key] = source_payload[key]
    if adapter_name:
        payload["merged_adapter_name"] = adapter_name
    if original_adapter_names is not None:
        payload["original_adapter_names"] = original_adapter_names
    if image_size is not None:
        payload["image_size"] = image_size
    return payload


def _require_module(module_name: str) -> Any:
    try:
        import importlib

        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModelAdapterError(f"Missing dependency required to merge LSDSAM weights: {module_name}") from exc
