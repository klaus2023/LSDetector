"""Pipeline stage contracts for the MVP command chain."""

from __future__ import annotations

from dataclasses import dataclass

from lsd_app.core.config import ProjectConfig

STAGES = ("prepare", "infer", "mosaic", "vectorize", "evaluate", "visualize")


@dataclass(frozen=True)
class StageResult:
    """Result returned by a pipeline stage placeholder."""

    stage: str
    model: str
    output_root: str
    message: str


def run_stage(stage: str, config: ProjectConfig) -> StageResult:
    """Run a named stage.

    The first skeleton keeps stages side-effect-light: commands validate config
    and print the operation boundary. Heavy data transforms, model loading, and
    geospatial writes are added behind these stable names in later phases.
    """

    if stage not in STAGES:
        available = ", ".join(STAGES)
        raise ValueError(f"Unknown stage '{stage}'. Available stages: {available}")

    return StageResult(
        stage=stage,
        model=config.model,
        output_root=str(config.output_root),
        message=f"{stage} stage is wired for the MVP skeleton.",
    )
