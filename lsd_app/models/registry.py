"""Model adapter registry without importing heavyweight ML frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from lsd_app.models.base import ModelAdapter


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a registered model adapter or alias."""

    name: str
    adapter_name: str
    role: str
    status: str
    input_mode: str
    edge_mode: str | None = None
    code_path: str = ""
    options: dict[str, str] | None = None


_ADAPTER_IMPORTS = {
    "mock": "lsd_app.models.mock:MockAdapter",
    "lsdsam-fewshot": "lsd_app.models.lsdsam:LSDSAMFewShotAdapter",
    "lsdformer-fast": "lsd_app.models.lsdformer:LSDFormerFastAdapter",
}

_ALIASES = {
    "lsdformer": "LSDFormer",
    "lsdformer-fast": "lsdformer-fast",
    "mock": "mock",
    "lsdsam": "LSDSAM-B",
    "lsdsam-b": "LSDSAM-B",
    "lsdsam-fewshot": "lsdsam-fewshot",
    "lsdsam-h": "LSDSAM-H",
    "lsdsam-l": "LSDSAM-L",
    "sam_lora_legacy_adapter": "LSDSAM-B",
}

_CANONICAL_MODEL_NAMES = ("LSDFormer", "LSDSAM-H", "LSDSAM-L", "LSDSAM-B")
_MODEL_REPOSITORY_ROOT = Path(__file__).resolve().parent / "model_repository"
_LSDSAM_CODE_PATH = str((_MODEL_REPOSITORY_ROOT / "LSDSAM").resolve())
_LSDFORMER_ROOT = str((_MODEL_REPOSITORY_ROOT / "LSDFormer").resolve())
_LSDFORMER_CODE_PATH = str((_MODEL_REPOSITORY_ROOT / "LSDFormer" / "model").resolve())

MODEL_SPECS: dict[str, ModelSpec] = {
    "mock": ModelSpec(
        name="mock",
        adapter_name="mock",
        role="dependency-light placeholder segmentation adapter",
        status="hidden",
        input_mode="rgb",
    ),
    "lsdsam-fewshot": ModelSpec(
        name="lsdsam-fewshot",
        adapter_name="lsdsam-fewshot",
        role="legacy compatibility alias for LSDSAM-B",
        status="hidden",
        input_mode="rgb_wavelets",
        edge_mode="wavelets",
        code_path=_LSDSAM_CODE_PATH,
        options={"legacy_root": _LSDSAM_CODE_PATH, "sam_variant": "B"},
    ),
    "lsdformer-fast": ModelSpec(
        name="lsdformer-fast",
        adapter_name="lsdformer-fast",
        role="legacy compatibility alias for LSDFormer",
        status="hidden",
        input_mode="rgb",
        code_path=_LSDFORMER_CODE_PATH,
        options={"model_root": _LSDFORMER_ROOT},
    ),
    "LSDFormer": ModelSpec(
        name="LSDFormer",
        adapter_name="lsdformer-fast",
        role="LSDFormer encoder-decoder landslide segmentation model",
        status="registered",
        input_mode="rgb",
        code_path=_LSDFORMER_CODE_PATH,
        options={"model_root": _LSDFORMER_ROOT},
    ),
    "LSDSAM-H": ModelSpec(
        name="LSDSAM-H",
        adapter_name="lsdsam-fewshot",
        role="LSDSAM with SAM ViT-H backbone",
        status="registered",
        input_mode="rgb_wavelets",
        edge_mode="wavelets",
        code_path=_LSDSAM_CODE_PATH,
        options={"legacy_root": _LSDSAM_CODE_PATH, "sam_variant": "H"},
    ),
    "LSDSAM-L": ModelSpec(
        name="LSDSAM-L",
        adapter_name="lsdsam-fewshot",
        role="LSDSAM with SAM ViT-L backbone",
        status="registered",
        input_mode="rgb_wavelets",
        edge_mode="wavelets",
        code_path=_LSDSAM_CODE_PATH,
        options={"legacy_root": _LSDSAM_CODE_PATH, "sam_variant": "L"},
    ),
    "LSDSAM-B": ModelSpec(
        name="LSDSAM-B",
        adapter_name="lsdsam-fewshot",
        role="LSDSAM with SAM ViT-B backbone",
        status="registered",
        input_mode="rgb_wavelets",
        edge_mode="wavelets",
        code_path=_LSDSAM_CODE_PATH,
        options={"legacy_root": _LSDSAM_CODE_PATH, "sam_variant": "B"},
    ),
}


def available_model_names() -> tuple[str, ...]:
    """Return model names and aliases accepted by the registry."""

    return tuple(sorted({*MODEL_SPECS, *_ALIASES}))


def canonical_model_names() -> tuple[str, ...]:
    """Return the non-duplicated model names shown in repository views."""

    return _CANONICAL_MODEL_NAMES


def normalize_model_name(name: str) -> str:
    """Return the canonical adapter name for a model name or alias."""

    key = str(name or "").strip()
    if key in MODEL_SPECS:
        return key
    lower_key = key.lower()
    if lower_key in _ALIASES:
        return _ALIASES[lower_key]
    if lower_key in _ADAPTER_IMPORTS:
        return lower_key
    available = ", ".join(available_model_names())
    raise ValueError(f"Unknown model '{name}'. Available models: {available}")


def get_model_spec(name: str) -> ModelSpec:
    """Return a model spec by name or alias."""

    key = str(name or "").strip()
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    normalized = normalize_model_name(key)
    if normalized in MODEL_SPECS:
        return MODEL_SPECS[normalized]
    try:
        return MODEL_SPECS[next(model_id for model_id, spec in MODEL_SPECS.items() if spec.adapter_name == normalized)]
    except (KeyError, StopIteration) as exc:
        available = ", ".join(available_model_names())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc


def get_adapter(name: str) -> ModelAdapter:
    """Instantiate a model adapter by registry name or alias."""

    model_name = normalize_model_name(name)
    adapter_name = MODEL_SPECS[model_name].adapter_name
    import_path = _ADAPTER_IMPORTS[adapter_name]
    module_name, _, class_name = import_path.partition(":")
    module = import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()
