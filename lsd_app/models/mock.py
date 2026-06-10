"""Dependency-light mock adapter for tests and wiring validation."""

from __future__ import annotations

from typing import Any, Sequence

from lsd_app.models.base import ModelAdapter


class MockAdapter(ModelAdapter):
    """Produce deterministic placeholder masks without loading model weights."""

    name = "mock"

    def __init__(self, *, mode: str = "threshold") -> None:
        self.mode = mode
        self.threshold = 0.5

    def load(self, config: Any) -> "MockAdapter":
        self.threshold = float(getattr(config, "threshold", 0.5))
        return self

    def preprocess_patch(self, array: Any) -> Any:
        import numpy as np

        return np.asarray(array, dtype=np.float32)

    def predict_batch(self, batch: Sequence[Any]) -> list[Any]:
        import numpy as np

        logits: list[Any] = []
        for patch in batch:
            if self.mode == "zeros":
                logits.append(np.zeros(patch.shape[-2:], dtype=np.float32))
                continue

            band_count = min(3, patch.shape[0])
            if band_count == 0:
                logits.append(np.zeros(patch.shape[-2:], dtype=np.float32))
            else:
                logits.append(np.mean(patch[:band_count], axis=0).astype(np.float32, copy=False))
        return logits

    def postprocess_logits(self, logits: Any) -> Any:
        import numpy as np

        mask = np.asarray(logits, dtype=np.float32) >= self.threshold
        return (mask.astype(np.uint8)) * 255
