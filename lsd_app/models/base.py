"""Common model adapter interface for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

if TYPE_CHECKING:
    from lsd_app.config import PlatformConfig


class ModelAdapterError(RuntimeError):
    """Raised when a model adapter cannot be loaded or used."""


class ModelAdapterDependencyError(ModelAdapterError):
    """Raised when an adapter-specific optional dependency is unavailable."""


class ModelAdapter(ABC):
    """Uniform interface shared by all model inference adapters."""

    name: ClassVar[str]

    @abstractmethod
    def load(self, config: PlatformConfig) -> "ModelAdapter":
        """Load model resources for a normalized task config."""

    @abstractmethod
    def preprocess_patch(self, array: Any) -> Any:
        """Convert one patch array into adapter input format."""

    @abstractmethod
    def predict_batch(self, batch: Sequence[Any]) -> Sequence[Any]:
        """Predict logits or scores for a batch of preprocessed patches."""

    @abstractmethod
    def postprocess_logits(self, logits: Any) -> Any:
        """Convert logits or scores into a writable prediction mask."""


def missing_dependency_message(adapter_name: str, dependency: str, install_hint: str) -> str:
    """Build a consistent optional-dependency error message."""

    return (
        f"Model adapter '{adapter_name}' requires optional dependency '{dependency}'. "
        f"{install_hint}"
    )
