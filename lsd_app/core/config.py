"""Small configuration loader used by the CLI skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ProjectConfig:
    """Normalized project settings for the first open-source MVP."""

    raw: Mapping[str, Any]
    path: Path | None
    project_root: Path
    output_root: Path
    data_root: Path | None = None
    model: str = "lsdsam"


def load_project_config(
    path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    model: str | None = None,
) -> ProjectConfig:
    """Load a JSON/TOML config file and apply light normalization.

    YAML is intentionally left out of the required dependency set. Projects that
    need it can add PyYAML later without changing the public CLI contract.
    """

    config_path = Path(path).expanduser().resolve() if path else None
    raw: Mapping[str, Any] = {}
    if config_path:
        raw = _read_config(config_path)

    project_root = _as_path(raw.get("project_root"), default=Path.cwd())
    output_root = _as_path(
        output_dir if output_dir is not None else raw.get("output_root"),
        default=project_root / "outputs",
    )
    data_root_value = raw.get("data_root")
    data_root = _as_path(data_root_value, default=None) if data_root_value else None
    model_name = model or str(raw.get("model", "lsdsam"))

    return ProjectConfig(
        raw=raw,
        path=config_path,
        project_root=project_root,
        output_root=output_root,
        data_root=data_root,
        model=model_name,
    )


def _read_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML configs require Python 3.11+ or the tomli package.")
        with path.open("rb") as handle:
            return tomllib.load(handle)

    raise ValueError(f"Unsupported config format: {path.suffix}. Use .toml or .json.")


def _as_path(value: Any, *, default: Path | None) -> Path | None:
    if value is None:
        return default.resolve() if default is not None else None
    return Path(str(value)).expanduser().resolve()
