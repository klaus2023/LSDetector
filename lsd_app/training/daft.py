"""DAFT dry-run stage planning."""

from __future__ import annotations

from lsd_app.training.config import StageConfig, TrainingConfig


def build_stage_plan(config: TrainingConfig, stage: StageConfig, previous: object | None = None):
    """Build the DAFT stage plan without importing training frameworks."""

    from lsd_app.training.pipeline import (
        StagePlan,
        stage_augmentation_profile,
        stage_checkpoint_input,
        stage_checkpoint_output,
    )

    return StagePlan(
        name="daft",
        enabled=stage.enabled,
        output_dir=config.outputs.checkpoints / "daft",
        checkpoint_input=stage_checkpoint_input(stage, previous),
        checkpoint_output=stage_checkpoint_output(config, "daft"),
        data=stage.data,
        training=stage.training,
        params=stage.params,
        alignment_mode=stage.alignment_mode,
        notes=("domain adaptation dry-run",),
        augmentation_profile=stage_augmentation_profile(stage),
    )
