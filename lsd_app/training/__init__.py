"""Training dry-run configuration and planning."""

from lsd_app.training.config import (
    DEFAULT_STAGE_ORDER,
    STAGES,
    TRAINING_OUTPUT_SUBDIRS,
    StageConfig,
    TrainingConfig,
    TrainingOutputDirs,
    load_training_config,
    normalize_training_config,
)
from lsd_app.training.jobs import describe_training_jobs
from lsd_app.training.pipeline import StagePlan, TrainingPlan, build_training_plan
from lsd_app.training.augmentations import LEGACY_LSDSAM_AUGMENTATION_PROFILE, augmentation_profile_for_model

__all__ = [
    "DEFAULT_STAGE_ORDER",
    "STAGES",
    "TRAINING_OUTPUT_SUBDIRS",
    "StageConfig",
    "StagePlan",
    "TrainingConfig",
    "TrainingOutputDirs",
    "TrainingPlan",
    "LEGACY_LSDSAM_AUGMENTATION_PROFILE",
    "augmentation_profile_for_model",
    "build_training_plan",
    "describe_training_jobs",
    "load_training_config",
    "normalize_training_config",
]
