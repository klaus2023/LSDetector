"""Portable path helpers for LSD_APP workspaces and registries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


TOKEN_WORKSPACE = "@workspace"
TOKEN_REGISTRY = "@registry"
TOKEN_WEIGHTS = "@weights"
TOKEN_DATASETS = "@datasets"
TOKEN_MODELS = "@models"
TOKEN_PACKAGE = "@package"
TOKEN_APP = "@app"

PORTABLE_TOKENS = (
    TOKEN_WEIGHTS,
    TOKEN_DATASETS,
    TOKEN_REGISTRY,
    TOKEN_WORKSPACE,
    TOKEN_MODELS,
    TOKEN_PACKAGE,
    TOKEN_APP,
)

REGISTRY_DIRNAME = "registry"
WEIGHT_REPOSITORY_DIRNAME = "weight_repository"
DATASET_STORE_DIRNAME = "dataset_store"

ENV_WORKSPACE = "LSD_APP_WORKSPACE"
ENV_PATH_ALIASES = "LSD_APP_PATH_ALIASES"


def package_root() -> Path:
    """Return the installed ``lsd_app`` package directory."""

    return Path(__file__).resolve().parents[1]


def app_root() -> Path:
    """Return the source/project root containing the package."""

    return package_root().parent


def model_repository_root() -> Path:
    return package_root() / "models" / "model_repository"


def default_workspace_root() -> Path:
    """Return the durable workspace root for registered assets.

    Registered models, weights, datasets, jobs, and evaluation history live
    under ``<workspace>/registry``. Runtime outputs remain under ``outputs/``
    by convention and are intentionally separate from the durable registry.
    """

    configured = os.environ.get(ENV_WORKSPACE)
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()
    return app_root().resolve()


def workspace_root(value: str | Path | None = None) -> Path:
    if value in {None, ""}:
        return default_workspace_root()
    raw = os.path.expandvars(str(value).strip())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = app_root() / path
    return path.resolve(strict=False)


def registry_root_from_workspace(value: str | Path | None = None) -> Path:
    return workspace_root(value)


def registry_dir(workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / REGISTRY_DIRNAME


def weight_repository_dir(workspace: str | Path | None = None) -> Path:
    return registry_dir(workspace) / WEIGHT_REPOSITORY_DIRNAME


def dataset_repository_dir(workspace: str | Path | None = None) -> Path:
    return registry_dir(workspace) / DATASET_STORE_DIRNAME


def token_roots(workspace: str | Path | None = None) -> dict[str, Path]:
    ws = workspace_root(workspace)
    reg = ws / REGISTRY_DIRNAME
    return {
        TOKEN_WORKSPACE: ws,
        TOKEN_REGISTRY: reg,
        TOKEN_WEIGHTS: reg / WEIGHT_REPOSITORY_DIRNAME,
        TOKEN_DATASETS: reg / DATASET_STORE_DIRNAME,
        TOKEN_MODELS: model_repository_root(),
        TOKEN_PACKAGE: package_root(),
        TOKEN_APP: app_root(),
    }


def resolve_user_path(
    value: str | Path,
    *,
    base_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a path entered by a user or read from config/registry state.

    Absolute paths are kept absolute.  Relative paths are resolved against
    ``base_dir`` when supplied, otherwise against the workspace root.  Portable
    tokens such as ``@weights/...`` and ``@datasets/...`` resolve relative to
    the current workspace.
    """

    raw = str(value).strip()
    if not raw:
        raise ValueError("Path value is empty.")
    expanded = os.path.expandvars(raw)
    token_path = _resolve_token_path(expanded, workspace_root=workspace_root)
    if token_path is not None:
        path = token_path
    else:
        translated = _translate_path_alias(expanded)
        path = Path(translated).expanduser()
        if not path.is_absolute():
            anchor = Path(base_dir).expanduser() if base_dir not in {None, ""} else _workspace_anchor(workspace_root)
            path = anchor / path
    resolved = path.resolve(strict=False)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_optional_path(
    value: Any,
    *,
    base_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_user_path(value, base_dir=base_dir, workspace_root=workspace_root)


def portable_path(path: str | Path | None, *, workspace_root: str | Path | None = None) -> str | None:
    """Serialize internal package/workspace paths with stable tokens."""

    if path in {None, ""}:
        return None
    resolved = Path(path).expanduser().resolve(strict=False)
    roots = sorted(token_roots(workspace_root).items(), key=lambda item: len(str(item[1])), reverse=True)
    for token, root in roots:
        root_resolved = root.expanduser().resolve(strict=False)
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        return token if str(relative) == "." else f"{token}/{relative.as_posix()}"
    return str(resolved)


def portable_json(value: Any, *, workspace_root: str | Path | None = None) -> Any:
    """Convert Path objects in nested payloads into portable strings."""

    if isinstance(value, Path):
        return portable_path(value, workspace_root=workspace_root)
    if isinstance(value, Mapping):
        return {key: portable_json(item, workspace_root=workspace_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_json(item, workspace_root=workspace_root) for item in value]
    return value


def repair_portable_paths_in_payload(value: Any, *, workspace_root: str | Path | None = None) -> Any:
    """Rewrite absolute internal paths in an existing JSON-like payload."""

    if isinstance(value, Mapping):
        return {key: repair_portable_paths_in_payload(item, workspace_root=workspace_root) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_portable_paths_in_payload(item, workspace_root=workspace_root) for item in value]
    if isinstance(value, str):
        repaired = _portable_string_if_path(value, workspace_root=workspace_root)
        return repaired
    return value


def _portable_string_if_path(value: str, *, workspace_root: str | Path | None = None) -> str:
    text = value.strip()
    if not text or text.startswith("@"):
        return value
    if not _looks_like_path(text):
        return value
    try:
        path = Path(text).expanduser()
    except (OSError, ValueError):
        return value
    if not path.is_absolute():
        return value
    return portable_path(path, workspace_root=workspace_root) or value


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("~")
        or ":\\" in value
        or value.startswith("\\\\")
        or value.startswith("//")
    )


def _resolve_token_path(value: str, *, workspace_root: str | Path | None) -> Path | None:
    for token, root in token_roots(workspace_root).items():
        if value == token:
            return root
        prefix = f"{token}/"
        if value.startswith(prefix):
            return root / value[len(prefix):]
    return None


def _workspace_anchor(workspace: str | Path | None = None) -> Path:
    return Path(workspace).expanduser().resolve(strict=False) if workspace not in {None, ""} else default_workspace_root()


def _translate_path_alias(value: str) -> str:
    normalized = value.replace("\\", "/")
    for source_prefix, target_prefix in _configured_path_aliases():
        source = source_prefix.rstrip("/")
        target = target_prefix.rstrip("/")
        if normalized == source:
            return target
        if normalized.startswith(f"{source}/"):
            return f"{target}/{normalized[len(source) + 1:]}"
    return value


def _configured_path_aliases() -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    raw = os.environ.get(ENV_PATH_ALIASES, "")
    for item in raw.split(";"):
        if not item.strip() or "=" not in item:
            continue
        source, target = item.split("=", 1)
        source = source.strip().replace("\\", "/")
        target = target.strip().replace("\\", "/")
        if source and target:
            aliases.append((source, target))

    aliases.extend(
        (
            ("//wsl.localhost/Ubuntu-22.04", ""),
            ("/wsl.localhost/Ubuntu-22.04", ""),
            ("//wsl$/Ubuntu-22.04", ""),
            ("/wsl$/Ubuntu-22.04", ""),
        )
    )
    return tuple(aliases)
