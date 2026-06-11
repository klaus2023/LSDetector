"""Dry-run training plan assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lsd_app.config import ConfigError
from lsd_app.training.config import STAGES, StageConfig, TrainingConfig, load_training_config
from lsd_app.training.augmentations import augmentation_profile_for_model
from lsd_app.training import daft, taft, tsft


@dataclass(frozen=True)
class StagePlan:
    """Dry-run plan for one training stage."""

    name: str
    enabled: bool
    output_dir: Path
    checkpoint_input: Path | None
    checkpoint_output: Path
    data: dict[str, Any]
    training: dict[str, Any]
    params: dict[str, Any]
    alignment_mode: str
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    augmentation_profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class TrainingPlan:
    """Dry-run plan for the configured training stages."""

    config: TrainingConfig
    stages: tuple[StagePlan, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def dry_run_text(self) -> str:
        names = " -> ".join(stage.name for stage in self.stages)
        lines = [
            "Training dry-run plan",
            f"Stage order: {names}",
            f"Output dir: {self.config.output_dir}",
        ]
        for stage in self.stages:
            status = "enabled" if stage.enabled else "disabled"
            checkpoint_input = stage.checkpoint_input if stage.checkpoint_input is not None else "initial model"
            lines.extend(
                [
                    "",
                    f"[{stage.name}] {status}",
                    f"  checkpoint in: {checkpoint_input}",
                    f"  checkpoint out: {stage.checkpoint_output}",
                    f"  data: {_summarize_mapping(stage.data)}",
                    f"  training: {_summarize_mapping(stage.training)}",
                    f"  params: {_summarize_mapping(stage.params)}",
                    f"  alignment: {stage.alignment_mode}",
                ]
            )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


def build_training_plan(config: str | Path | TrainingConfig, *, create_output_dirs: bool = False) -> TrainingPlan:
    """Build a dry-run TrainingPlan without importing training frameworks."""

    training_config = (
        load_training_config(config, create_output_dirs=create_output_dirs)
        if isinstance(config, (str, Path))
        else config
    )

    builders = {
        "taft": taft.build_stage_plan,
        "daft": daft.build_stage_plan,
        "tsft": tsft.build_stage_plan,
    }
    stage_plans: list[StagePlan] = []
    previous: StagePlan | None = None
    for stage_name in training_config.stage_order:
        if stage_name not in STAGES:
            raise ConfigError(f"Unknown training stage '{stage_name}'.")
        stage_plan = builders[stage_name](training_config, training_config.stages[stage_name], previous)
        stage_plans.append(stage_plan)
        previous = stage_plan

    warnings = tuple(training_config.warnings + tuple(warning for stage in stage_plans for warning in stage.warnings))
    return TrainingPlan(config=training_config, stages=tuple(stage_plans), warnings=warnings)


def stage_checkpoint_input(stage: StageConfig, previous: StagePlan | None) -> Path | None:
    """Resolve checkpoint input for a stage from explicit config or the previous stage."""

    configured = stage.model.get("checkpoint_path")
    if configured is not None:
        return Path(configured)
    if previous is not None:
        return previous.checkpoint_output
    configured = stage.model.get("base_checkpoint")
    if configured is not None:
        return Path(configured)
    return None


def stage_checkpoint_output(config: TrainingConfig, stage_name: str) -> Path:
    return config.outputs.checkpoints / stage_name / "checkpoint.dryrun"


def stage_augmentation_profile(stage: StageConfig) -> dict[str, Any] | None:
    return augmentation_profile_for_model(stage.model)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, dict):
        return {key: _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    return value


def _summarize_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    preferred_keys = (
        "epochs",
        "batch_size",
        "lr",
        "learning_rate",
        "loss",
        "train_images_dir",
        "train_masks_dir",
        "target_images_dir",
        "target_masks_dir",
    )
    pieces = []
    for key in preferred_keys:
        if key in values:
            pieces.append(f"{key}={values[key]}")
    if not pieces:
        pieces = [f"{key}={value}" for key, value in list(values.items())[:4]]
    if len(values) > len(pieces):
        pieces.append(f"+{len(values) - len(pieces)} more")
    return ", ".join(pieces)
