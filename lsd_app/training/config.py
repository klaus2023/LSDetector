"""Training configuration for dry-run planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsd_app.config import ConfigError, load_raw_config
from lsd_app.core.paths import resolve_user_path


STAGES = ("taft", "daft", "tsft")
DEFAULT_STAGE_ORDER = STAGES
TRAINING_OUTPUT_SUBDIRS = ("checkpoints", "logs", "metrics", "index_logs", "plans")
ALIGNMENT_MODES = ("none", "ist", "mmd", "coral")


@dataclass(frozen=True)
class TrainingOutputDirs:
    """Standard training output directories."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    index_logs: Path
    plans: Path

    @classmethod
    def build(cls, output_dir: Path, *, create: bool) -> "TrainingOutputDirs":
        values = {"root": output_dir}
        for name in TRAINING_OUTPUT_SUBDIRS:
            values[name] = output_dir / name

        dirs = cls(**values)
        if create:
            for directory in values.values():
                directory.mkdir(parents=True, exist_ok=True)
        return dirs


@dataclass(frozen=True)
class StageConfig:
    """Normalized configuration for one training stage."""

    name: str
    enabled: bool
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    alignment_mode: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class TrainingConfig:
    """Normalized configuration for TAFT, DAFT, and TSFT dry-run planning."""

    output_dir: Path
    stage_order: tuple[str, ...]
    data: dict[str, Any]
    model: dict[str, Any]
    runtime: dict[str, Any]
    stages: dict[str, StageConfig]
    config_path: Path | None
    outputs: TrainingOutputDirs
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def load_training_config(path: str | Path, *, create_output_dirs: bool = True) -> TrainingConfig:
    """Load and validate a YAML or JSON training config."""

    config_path = Path(path).expanduser().resolve()
    raw = load_raw_config(config_path)
    return normalize_training_config(raw, config_path=config_path, create_output_dirs=create_output_dirs)


def normalize_training_config(
    raw: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    create_output_dirs: bool = True,
) -> TrainingConfig:
    """Normalize raw training config data without importing training libraries."""

    root = _mapping(_value(raw, ("training",), default=raw), "training")
    path_base_dir = config_path.parent if config_path is not None else None
    output_dir = _path(
        _value(root, ("output_dir",), ("output", "dir"), default="outputs/training"),
        "training.output_dir",
        base_dir=path_base_dir,
    )
    stage_order = _stage_order(_value(root, ("stage_order",), default=DEFAULT_STAGE_ORDER))
    data = _expand_paths(_mapping(_value(root, ("data",), default={}), "training.data"), base_dir=path_base_dir)
    model = _expand_paths(_mapping(_value(root, ("model",), default={}), "training.model"), base_dir=path_base_dir)
    runtime = dict(_mapping(_value(root, ("runtime",), default={}), "training.runtime"))

    raw_stages = _mapping(_value(root, ("stages",), default={}), "training.stages")
    stages = {
        name: _stage_config(name, raw_stages.get(name, {}), data=data, model=model, base_dir=path_base_dir)
        for name in STAGES
    }

    outputs = TrainingOutputDirs.build(output_dir, create=create_output_dirs)
    warnings = tuple(_missing_path_warnings(data, "data") + _missing_path_warnings(model, "model"))
    for stage_name, stage in stages.items():
        warnings += tuple(_missing_path_warnings(stage.data, f"stages.{stage_name}.data"))
        warnings += tuple(_missing_path_warnings(stage.model, f"stages.{stage_name}.model"))

    return TrainingConfig(
        output_dir=output_dir,
        stage_order=stage_order,
        data=data,
        model=model,
        runtime=runtime,
        stages=stages,
        config_path=config_path,
        outputs=outputs,
        warnings=warnings,
    )


def _stage_config(
    name: str,
    raw: Any,
    *,
    data: dict[str, Any],
    model: dict[str, Any],
    base_dir: Path | None = None,
) -> StageConfig:
    stage_raw = _mapping(raw, f"training.stages.{name}") if raw is not None else {}
    enabled = _bool(_value(stage_raw, ("enabled",), default=True), f"training.stages.{name}.enabled")
    stage_data = _expand_paths(
        _merge(data, _mapping(_value(stage_raw, ("data",), default={}), f"training.stages.{name}.data")),
        base_dir=base_dir,
    )
    stage_model = _expand_paths(
        _merge(model, _mapping(_value(stage_raw, ("model",), default={}), f"training.stages.{name}.model")),
        base_dir=base_dir,
    )
    training = dict(_mapping(_value(stage_raw, ("training",), default={}), f"training.stages.{name}.training"))
    params = dict(_mapping(_value(stage_raw, ("params",), default={}), f"training.stages.{name}.params"))
    alignment_mode = _alignment_mode(_value(stage_raw, ("alignment_mode",), default="none"), f"training.stages.{name}.alignment_mode")
    return StageConfig(
        name=name,
        enabled=enabled,
        data=stage_data,
        model=stage_model,
        training=training,
        alignment_mode=alignment_mode,
        params=params,
    )


def _stage_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        order = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        order = [str(part).strip().lower() for part in value]
    else:
        raise ConfigError("Invalid training.stage_order: expected list or comma-separated string.")

    if not order:
        raise ConfigError("Invalid training.stage_order: expected at least one stage.")
    unknown = [stage for stage in order if stage not in STAGES]
    if unknown:
        allowed = ", ".join(STAGES)
        raise ConfigError(f"Unknown training stage '{unknown[0]}'. Expected one of: {allowed}.")
    return tuple(order)


def _alignment_mode(value: Any, field_name: str) -> str:
    mode = str(value).strip().lower()
    if mode not in ALIGNMENT_MODES:
        allowed = ", ".join(ALIGNMENT_MODES)
        raise ConfigError(f"Invalid {field_name}: expected one of {allowed}, got {value!r}.")
    return mode


def _missing_path_warnings(values: Mapping[str, Any], prefix: str) -> list[str]:
    warnings: list[str] = []
    for key, value in values.items():
        if isinstance(value, Mapping):
            warnings.extend(_missing_path_warnings(value, f"{prefix}.{key}"))
            continue
        if not _is_path_key(key) or _is_generated_path_key(key):
            continue
        for path in _path_values(value):
            if not path.exists():
                warnings.append(f"Missing path for {prefix}.{key}: {path}")
    return warnings


def _is_path_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.endswith("_path")
        or lowered.endswith("_paths")
        or lowered.endswith("_dir")
        or lowered.endswith("_dirs")
        or "checkpoint" in lowered
    )


def _is_generated_path_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.startswith("output_") or lowered in {"output_dir", "checkpoint_output", "output_checkpoint"}


def _path_values(value: Any) -> list[Path]:
    if value is None or value == "":
        return []
    if isinstance(value, (str, Path)):
        return [_path(value, "path")]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_path(item, "path") for item in value if item not in {None, ""}]
    return []


def _expand_paths(values: Mapping[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            expanded[key] = _expand_paths(value, base_dir=base_dir)
        elif _is_path_key(key) and isinstance(value, (str, Path)):
            expanded[key] = _path(value, key, base_dir=base_dir)
        elif _is_path_key(key) and isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            expanded[key] = [_path(item, key, base_dir=base_dir) if isinstance(item, (str, Path)) else item for item in value]
        else:
            expanded[key] = value
    return expanded


def _path(value: Any, field_name: str, *, base_dir: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"Invalid path for {field_name}: expected string, got {type(value).__name__}.")
    return resolve_user_path(value, base_dir=base_dir)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Invalid value for {field_name}: expected mapping/object.")
    return dict(value)


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ConfigError(f"Invalid {field_name}: expected boolean, got {value!r}.")


def _value(raw: Mapping[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        current: Any = raw
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    return default


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


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
