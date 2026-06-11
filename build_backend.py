from __future__ import annotations

import base64
import hashlib
import shutil
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_NAME = "lsd_app"
VERSION = "0.1.1"
PREFIX = f"{DIST_NAME}-{VERSION}"
EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "lsd_app.egg-info",
    "outputs",
    "preprocess",
    "LSDSAM_finetune",
    "data",
    "datasets",
    "raw",
    "processed",
    "tiles",
    "mosaics",
    "vectors",
    "runs",
    "logs",
    "wandb",
    "tensorboard",
}
EXCLUDE_PREFIXES = ("._", ".__")
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
EXCLUDE_NAMES = {
    "prompt0.md",
    "prompt1.md",
    "prompt2.md",
    "prompt3.md",
    "prompt4.md",
    "prompt5.md",
    "prompt6.md",
    "prompt7.md",
    "prompt8.md",
    "prompt9.md",
    "prompt10.md",
    "Progress.md",
}
PACKAGE_DIRS = ["lsd_app", "tests", "configs", "examples", "docs"]
INCLUDE_FILES = ["README.md", "pyproject.toml", "MANIFEST.in"]


def get_requires_for_build_sdist(config_settings=None):
    return []


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_sdist(sdist_directory, config_settings=None):
    out_dir = Path(sdist_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{PREFIX}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="lsd-sdist-") as tmp:
        root = Path(tmp) / PREFIX
        _copy_tree(root)
        metadata = _metadata_text()
        (root / "PKG-INFO").write_text(metadata, encoding="utf-8")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(root, arcname=PREFIX)
    return tar_path.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    out_dir = Path(wheel_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{DIST_NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = out_dir / wheel_name
    dist_info = f"{DIST_NAME.replace('-', '_')}-{VERSION}.dist-info"
    record_rows: list[str] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path, data in _iter_wheel_files():
            zf.writestr(rel_path, data)
            record_rows.append(_record_row(rel_path, data))
        wheel_metadata = (
            "Wheel-Version: 1.0\n"
            "Generator: build_backend\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("utf-8")
        metadata = _metadata_text().encode("utf-8")
        zf.writestr(f"{dist_info}/WHEEL", wheel_metadata)
        zf.writestr(f"{dist_info}/METADATA", metadata)
        record_rows.append(_record_row(f"{dist_info}/WHEEL", wheel_metadata))
        record_rows.append(_record_row(f"{dist_info}/METADATA", metadata))
        record_rows.append(f"{dist_info}/RECORD,,")
        record = "\n".join(record_rows).encode("utf-8")
        zf.writestr(f"{dist_info}/RECORD", record)
    return wheel_name


def _copy_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in ROOT.iterdir():
        if _should_skip(item):
            continue
        dest = root / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest,
                ignore=shutil.ignore_patterns("._*", ".__*", "__pycache__", "*.pyc", "*.pyo"),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(item, dest)


def _iter_wheel_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("dist/") or rel.startswith("build/") or rel.startswith("lsd_app.egg-info/"):
            continue
        if rel.startswith("outputs/") or rel.startswith("preprocess/") or rel.startswith("LSDSAM_finetune/"):
            continue
        yield rel, path.read_bytes()


def _should_skip(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_DIRS or name in EXCLUDE_NAMES:
        return True
    if name.startswith(EXCLUDE_PREFIXES):
        return True
    if name.endswith(EXCLUDE_SUFFIXES):
        return True
    return False


def _record_row(rel_path: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{rel_path},sha256={encoded},{len(data)}"


def _metadata_text() -> str:
    readme = _readme_text()
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {DIST_NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Open-source engineering skeleton for landslide detection.\n"
        f"{_requires_dist_text()}"
        "Description-Content-Type: text/markdown\n"
        "\n"
        f"{readme}"
    )


def _readme_text() -> str:
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Open-source engineering skeleton for landslide detection."


def _requires_dist_text() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return ""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    lines: list[str] = []
    for requirement in project.get("dependencies", []) or []:
        lines.append(f"Requires-Dist: {requirement}\n")
    optional = project.get("optional-dependencies", {}) or {}
    for extra, requirements in optional.items():
        normalized_extra = str(extra).replace("_", "-")
        lines.append(f"Provides-Extra: {normalized_extra}\n")
        for requirement in requirements or []:
            lines.append(f"Requires-Dist: {requirement}; extra == \"{normalized_extra}\"\n")
    return "".join(lines)
