"""Platform configuration for LSDetector tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from lsd_app.core.channels import normalize_rgb_bands
from lsd_app.core.paths import resolve_user_path
from lsd_app.models.registry import available_model_names, get_model_spec


KNOWN_MODELS = available_model_names()
OUTPUT_SUBDIRS = (
    "patches",
    "patch_predictions",
    "rasters",
    "vectors",
    "metrics",
    "visualizations",
    "logs",
)


class ConfigError(ValueError):
    """Raised when a platform config is missing or invalid."""


@dataclass(frozen=True)
class CRSOptions:
    """Coordinate reference system handling options."""

    preserve_input: bool = True
    target: str | None = None
    allow_missing: bool = False


@dataclass(frozen=True)
class NoDataOptions:
    """NoData handling options."""

    input_value: int | float | str | None = None
    output_value: int | float | str | None = 0
    mask_nodata: bool = True


@dataclass(frozen=True)
class TaskOutputDirs:
    """Standardized output directories for every task."""

    root: Path
    patches: Path
    patch_predictions: Path
    rasters: Path
    vectors: Path
    metrics: Path
    visualizations: Path
    logs: Path

    @classmethod
    def build(cls, output_dir: Path, *, create: bool = True) -> "TaskOutputDirs":
        values = {"root": output_dir}
        for name in OUTPUT_SUBDIRS:
            values[name] = output_dir / name

        dirs = cls(**values)
        if create:
            for directory in values.values():
                directory.mkdir(parents=True, exist_ok=True)
        return dirs


@dataclass(frozen=True)
class PlatformConfig:
    """Normalized configuration for inference, post-processing, and evaluation."""

    input_raster_path: Path
    output_dir: Path
    model_name: str
    checkpoint_path: Path | None
    tile_size: int
    stride: int
    batch_size: int
    overwrite: bool
    threshold: float
    min_area_pixels: int
    simplify_tolerance: float
    connectivity: int
    vectorize_mode: str
    vector_formats: tuple[str, ...]
    device: str
    num_workers: int
    crs: CRSOptions
    nodata: NoDataOptions
    truth_vector_path: Path | None
    truth_mask_path: Path | None
    RGB_bands: str
    input_mode: str
    adapter: str | None
    adapter_options: dict[str, Any]
    raster_options: dict[str, Any]
    config_path: Path | None
    outputs: TaskOutputDirs

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return _json_ready(asdict(self))


def load_config(path: str | Path, *, create_output_dirs: bool = True) -> PlatformConfig:
    """Load and validate a YAML or JSON platform config."""

    config_path = Path(path).expanduser().resolve()
    raw = load_raw_config(config_path)
    return normalize_config(raw, config_path=config_path, create_output_dirs=create_output_dirs)


def load_raw_config(path: str | Path) -> Mapping[str, Any]:
    """Load a raw YAML or JSON mapping from disk."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON config: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        loaded = _load_yaml(config_path)
    else:
        raise ConfigError(f"Unsupported config format '{config_path.suffix}'. Use .yaml, .yml, or .json.")

    if not isinstance(loaded, Mapping):
        raise ConfigError("Config root must be a mapping/object.")
    return loaded


def normalize_config(
    raw: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    create_output_dirs: bool = True,
) -> PlatformConfig:
    """Normalize a raw mapping into a validated PlatformConfig."""

    path_base_dir = config_path.parent if config_path is not None else None
    workspace_root = _optional_workspace_root(raw)
    input_raster = _required_path(
        raw,
        "input.raster_path",
        ("input", "raster_path"),
        ("input_raster_path",),
        ("raster_path",),
        base_dir=path_base_dir,
        workspace_root=workspace_root,
    )
    output_dir = _optional_path(
        _value(raw, ("output", "dir"), ("output_dir",), default="outputs"),
        field_name="output.dir",
        base_dir=path_base_dir,
        workspace_root=workspace_root,
    )

    model_name_raw = _string(
        _value(raw, ("model", "name"), ("model_name",), ("model",), default="lsdsam"),
        "model.name",
    )
    model_name = model_name_raw if model_name_raw in KNOWN_MODELS else model_name_raw.lower()
    if model_name not in KNOWN_MODELS:
        allowed = ", ".join(KNOWN_MODELS)
        raise ConfigError(f"Unknown model name '{model_name}'. Expected one of: {allowed}.")
    model_spec = get_model_spec(model_name)

    checkpoint_path = _optional_path(
        _value(raw, ("model", "checkpoint_path"), ("checkpoint_path",), default=None),
        field_name="checkpoint_path",
        base_dir=path_base_dir,
        workspace_root=workspace_root,
    )
    adapter = _optional_string(_value(raw, ("model", "adapter"), ("adapter",), default=None), "model.adapter")
    adapter_options = _mapping(
        _value(raw, ("model", "options"), ("model", "adapter_options"), ("adapter_options",), default={}),
        "model.options",
    )
    if model_spec.options:
        adapter_options = {**dict(model_spec.options), **dict(adapter_options)}
    raster_options = _mapping(
        _value(raw, ("output", "raster_options"), ("raster_options",), default={}),
        "output.raster_options",
    )

    tile_size = _positive_int(_value(raw, ("tiling", "tile_size"), ("tile_size",), default=512), "tile_size")
    stride = _positive_int(_value(raw, ("tiling", "stride"), ("stride",), default=256), "stride")
    if stride > tile_size:
        raise ConfigError(f"Invalid tiling: stride ({stride}) must be <= tile_size ({tile_size}).")

    batch_size = _positive_int(_value(raw, ("inference", "batch_size"), ("batch_size",), default=1), "batch_size")
    overwrite = _bool(_value(raw, ("inference", "overwrite"), ("overwrite",), default=True), "inference.overwrite")
    threshold = _threshold(_value(raw, ("inference", "threshold"), ("threshold",), default=0.5))
    min_area_pixels = _non_negative_int(
        _value(raw, ("postprocess", "min_area_pixels"), ("min_area_pixels",), default=0),
        "min_area_pixels",
    )
    simplify_tolerance = _non_negative_float(
        _value(
            raw,
            ("postprocess", "simplify_tolerance"),
            ("vectorize", "simplify_tolerance"),
            ("simplify_tolerance",),
            default=0.0,
        ),
        "simplify_tolerance",
    )
    connectivity = _connectivity(
        _value(raw, ("postprocess", "connectivity"), ("vectorize", "connectivity"), ("connectivity",), default=8)
    )
    vectorize_mode = _vectorize_mode(_value(raw, ("vectorize", "mode"), ("vectorize_mode",), default="streaming"))
    vector_formats = _vector_formats(
        _value(raw, ("output", "vector_formats"), ("vectorize", "formats"), ("vector_formats",), default=("gpkg",))
    )
    device = _string(_value(raw, ("runtime", "device"), ("device",), default="cpu"), "device")
    num_workers = _non_negative_int(
        _value(raw, ("runtime", "num_workers"), ("num_workers",), default=0),
        "num_workers",
    )

    crs = CRSOptions(
        preserve_input=_bool(
            _value(raw, ("geospatial", "crs", "preserve_input"), ("crs_preserve_input",), default=True),
            "geospatial.crs.preserve_input",
        ),
        target=_optional_string(
            _value(raw, ("geospatial", "crs", "target"), ("target_crs",), default=None),
            "geospatial.crs.target",
        ),
        allow_missing=_bool(
            _value(raw, ("geospatial", "crs", "allow_missing"), ("allow_missing_crs",), default=False),
            "geospatial.crs.allow_missing",
        ),
    )
    nodata = NoDataOptions(
        input_value=_value(raw, ("geospatial", "nodata", "input"), ("input_nodata",), default=None),
        output_value=_value(raw, ("geospatial", "nodata", "output"), ("output_nodata",), default=0),
        mask_nodata=_bool(
            _value(raw, ("geospatial", "nodata", "mask"), ("mask_nodata",), default=True),
            "geospatial.nodata.mask",
        ),
    )

    truth_vector_path = _optional_path(
        _value(raw, ("truth", "vector_path"), ("truth_vector_path",), default=None),
        field_name="truth.vector_path",
        base_dir=path_base_dir,
        workspace_root=workspace_root,
    )
    truth_mask_path = _optional_path(
        _value(raw, ("truth", "mask_path"), ("truth_mask_path",), default=None),
        field_name="truth.mask_path",
        base_dir=path_base_dir,
        workspace_root=workspace_root,
    )
    if _value(raw, ("input", "channel_order"), ("data", "channel_order"), ("channel_order",), default=None) is not None:
        raise ConfigError("channel_order is no longer supported. Use input.RGB_bands, for example RGB_bands: 1,2,3.")
    RGB_bands = _RGB_bands(_value(raw, ("input", "RGB_bands"), ("data", "RGB_bands"), ("RGB_bands",), default=None))
    input_mode = str(_value(raw, ("model", "input_mode"), ("input_mode",), default=model_spec.input_mode)).strip().lower()
    if model_spec.input_mode == "rgb_wavelets":
        input_mode = "rgb_wavelets"
        adapter_options = dict(adapter_options)
        adapter_options["edge"] = "wavelets"
        adapter_options.pop("mode", None)
    elif input_mode != "rgb":
        input_mode = "rgb"

    outputs = TaskOutputDirs.build(output_dir, create=create_output_dirs)
    return PlatformConfig(
        input_raster_path=input_raster,
        output_dir=output_dir,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        overwrite=overwrite,
        threshold=threshold,
        min_area_pixels=min_area_pixels,
        simplify_tolerance=simplify_tolerance,
        connectivity=connectivity,
        vectorize_mode=vectorize_mode,
        vector_formats=vector_formats,
        device=device,
        num_workers=num_workers,
        crs=crs,
        nodata=nodata,
        truth_vector_path=truth_vector_path,
        truth_mask_path=truth_mask_path,
        RGB_bands=RGB_bands,
        input_mode=input_mode,
        adapter=adapter,
        adapter_options=adapter_options,
        raster_options=raster_options,
        config_path=config_path,
        outputs=outputs,
    )


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - PyYAML provides varied exception types
        raise ConfigError(f"Invalid YAML config: {exc}") from exc


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a small YAML mapping subset when PyYAML is unavailable."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]:
            raise ConfigError(f"Invalid YAML indentation at line {line_number}: tabs are not supported.")

        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        key, separator, value = stripped.partition(":")
        if not separator:
            raise ConfigError(f"Invalid YAML line {line_number}: expected 'key: value'.")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"Invalid YAML indentation at line {line_number}.")

        parent = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    lower = value.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


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


def _optional_workspace_root(raw: Mapping[str, Any]) -> str | Path | None:
    return _value(raw, ("registry_root",), ("workspace_root",), ("workspace",), default=None)


def _required_path(
    raw: Mapping[str, Any],
    field_name: str,
    *paths: tuple[str, ...],
    base_dir: Path | None = None,
    workspace_root: str | Path | None = None,
) -> Path:
    value = _value(raw, *paths, default=None)
    if value is None or str(value).strip() == "":
        raise ConfigError(f"Missing required config value: {field_name}.")
    return _path(value, field_name, base_dir=base_dir, workspace_root=workspace_root)


def _optional_path(
    value: Any,
    *,
    field_name: str,
    base_dir: Path | None = None,
    workspace_root: str | Path | None = None,
) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _path(value, field_name, base_dir=base_dir, workspace_root=workspace_root)


def _path(
    value: Any,
    field_name: str,
    *,
    base_dir: Path | None = None,
    workspace_root: str | Path | None = None,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"Invalid path for {field_name}: expected string, got {type(value).__name__}.")
    text = str(value).strip()
    token_workspace = workspace_root if text.startswith("@") else None
    return resolve_user_path(value, base_dir=base_dir, workspace_root=token_workspace)

def _string(value: Any, field_name: str) -> str:
    if value is None or str(value).strip() == "":
        raise ConfigError(f"Missing required config value: {field_name}.")
    return str(value).strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Invalid value for {field_name}: expected string or null.")
    return value.strip()


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Invalid value for {field_name}: expected mapping/object.")
    return dict(value)


def _positive_int(value: Any, field_name: str) -> int:
    number = _int(value, field_name)
    if number <= 0:
        raise ConfigError(f"Invalid {field_name}: expected a positive integer, got {number}.")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    number = _int(value, field_name)
    if number < 0:
        raise ConfigError(f"Invalid {field_name}: expected a non-negative integer, got {number}.")
    return number


def _non_negative_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {field_name}: expected a non-negative number, got {value!r}.") from exc
    if number < 0:
        raise ConfigError(f"Invalid {field_name}: expected a non-negative number, got {number}.")
    return number


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"Invalid {field_name}: expected integer, got boolean.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {field_name}: expected integer, got {value!r}.") from exc


def _threshold(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid threshold: expected a number between 0 and 1, got {value!r}.") from exc
    if not 0 <= number <= 1:
        raise ConfigError(f"Invalid threshold: expected a number between 0 and 1, got {number}.")
    return number


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


def _connectivity(value: Any) -> int:
    number = _int(value, "connectivity")
    if number not in {4, 8}:
        raise ConfigError(f"Invalid connectivity: expected 4 or 8, got {number}.")
    return number


def _vectorize_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"streaming", "legacy"}:
        raise ConfigError(f"Invalid vectorize mode '{value}'. Expected streaming or legacy.")
    return mode


def _RGB_bands(value: Any) -> str:
    try:
        return normalize_rgb_bands(value)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _vector_formats(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_formats = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_formats = list(value)
    else:
        raise ConfigError(f"Invalid vector formats: expected string or list, got {type(value).__name__}.")

    aliases = {"shp": "shapefile", "shape": "shapefile", "geopackage": "gpkg", "json": "geojson"}
    allowed = {"gpkg", "geojson", "shapefile"}
    formats: list[str] = []
    for raw_format in raw_formats:
        normalized = aliases.get(str(raw_format).strip().lower(), str(raw_format).strip().lower())
        if normalized not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ConfigError(f"Invalid vector format '{raw_format}'. Expected one of: {expected}.")
        if normalized not in formats:
            formats.append(normalized)
    if not formats:
        formats.append("gpkg")
    return tuple(formats)


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
