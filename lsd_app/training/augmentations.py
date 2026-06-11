"""Training augmentation profiles mapped from the legacy LSDSAM dataloaders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from lsd_app.core.channels import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD


LEGACY_LSDSAM_AUGMENTATION_PROFILE: dict[str, Any] = {
    "profile_id": "legacy-lsdsam-train-v1",
    "source": "LSDSAM_finetune/dataloader_SAM.py::CustomDataset",
    "scope": "training-only",
    "train": [
        {"name": "LongestMaxSize", "max_size": 512},
        {"name": "PadIfNeeded", "min_height": 512, "min_width": 512, "border_mode": "constant", "value": 0},
        {"name": "HorizontalFlip", "p": 0.5},
        {"name": "VerticalFlip", "p": 0.5},
        {"name": "RandomResizedCrop", "width": 512, "height": 512, "scale": [0.5, 0.8], "ratio": [0.75, 1.333], "p": 0.3},
        {"name": "Rotate", "limit": 45, "p": 0.4, "border_mode": "constant", "value": 0},
        {"name": "GaussNoise", "var_limit": [0.0005, 0.005], "p": 0.2},
        {"name": "GaussianBlur", "blur_limit": [3, 7], "p": 0.3},
        {"name": "RandomBrightnessContrast", "brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.3},
    ],
    "eval": [
        {"name": "LongestMaxSize", "max_size": 512},
        {"name": "PadIfNeeded", "min_height": 512, "min_width": 512, "border_mode": "constant", "value": 0},
    ],
    "post_transform": [
        {"name": "wavelets", "applies_to": "LSDSAM edge channels"},
        {"name": "ImageNetNormalize", "mean": list(IMAGENET_RGB_MEAN), "std": list(IMAGENET_RGB_STD), "applies_to": "RGB first3"},
        {"name": "ToTensorV2", "transpose_mask": True},
    ],
}


def augmentation_profile_for_model(model: dict[str, Any]) -> dict[str, Any] | None:
    """Return a serializable augmentation profile for dry-run training plans."""

    model_name = str(model.get("name") or "lsdsam").strip().lower()
    if model_name in {"lsdsam", "lsdsam-fewshot", "sam_lora_legacy_adapter", ""}:
        return deepcopy(LEGACY_LSDSAM_AUGMENTATION_PROFILE)
    return None
