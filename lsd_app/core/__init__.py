"""Core configuration and pipeline primitives.

The package avoids eager imports so small helpers such as
`lsd_app.core.channels` can be used by config parsing without pulling in the
full pipeline and creating circular imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ProjectConfig": "lsd_app.core.config",
    "load_project_config": "lsd_app.core.config",
    "InferenceError": "lsd_app.core.inference",
    "InferenceResult": "lsd_app.core.inference",
    "run_patch_inference": "lsd_app.core.inference",
    "EvaluationError": "lsd_app.core.metrics",
    "EvaluationResult": "lsd_app.core.metrics",
    "EvaluationScores": "lsd_app.core.metrics",
    "run_evaluate": "lsd_app.core.metrics",
    "MosaicError": "lsd_app.core.mosaic",
    "MosaicResult": "lsd_app.core.mosaic",
    "run_mosaic": "lsd_app.core.mosaic",
    "STAGES": "lsd_app.core.pipeline",
    "StageResult": "lsd_app.core.pipeline",
    "run_stage": "lsd_app.core.pipeline",
    "RasterError": "lsd_app.core.raster",
    "RasterMetadata": "lsd_app.core.raster",
    "read_raster_metadata": "lsd_app.core.raster",
    "VectorStats": "lsd_app.core.stats",
    "summarize_vector_features": "lsd_app.core.stats",
    "write_vector_stats": "lsd_app.core.stats",
    "PatchRecord": "lsd_app.core.tiling",
    "PrepareResult": "lsd_app.core.tiling",
    "iter_windows": "lsd_app.core.tiling",
    "prepare_raster": "lsd_app.core.tiling",
    "VectorFeature": "lsd_app.core.vectorize",
    "VectorizeError": "lsd_app.core.vectorize",
    "VectorizeResult": "lsd_app.core.vectorize",
    "run_vectorize": "lsd_app.core.vectorize",
    "VisualizationError": "lsd_app.core.visualization",
    "VisualizationResult": "lsd_app.core.visualization",
    "run_visualize": "lsd_app.core.visualization",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lsd_app.core' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "PatchRecord",
    "PrepareResult",
    "ProjectConfig",
    "InferenceError",
    "InferenceResult",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationScores",
    "MosaicError",
    "MosaicResult",
    "RasterError",
    "RasterMetadata",
    "STAGES",
    "StageResult",
    "VectorFeature",
    "VectorStats",
    "VectorizeError",
    "VectorizeResult",
    "VisualizationError",
    "VisualizationResult",
    "iter_windows",
    "load_project_config",
    "prepare_raster",
    "read_raster_metadata",
    "run_mosaic",
    "run_patch_inference",
    "run_stage",
    "run_evaluate",
    "run_visualize",
    "run_vectorize",
    "summarize_vector_features",
    "write_vector_stats",
]
