"""Model adapter registry for LSDFormer, LSDSAM, and test doubles."""

from lsd_app.models.base import ModelAdapter, ModelAdapterDependencyError, ModelAdapterError
from lsd_app.models.registry import (
    MODEL_SPECS,
    ModelSpec,
    available_model_names,
    canonical_model_names,
    get_adapter,
    get_model_spec,
    normalize_model_name,
)

__all__ = [
    "MODEL_SPECS",
    "ModelAdapter",
    "ModelAdapterDependencyError",
    "ModelAdapterError",
    "ModelSpec",
    "available_model_names",
    "canonical_model_names",
    "get_adapter",
    "get_model_spec",
    "normalize_model_name",
]
