#!/usr/bin/env python3
"""Register LSDetector release assets downloaded from Baidu Netdisk.

Expected unpacked layout:

    LSDetector_data_weights_v1/
      weights/
        LSDFormer_LS77K.pth
        LSDSAM_H_LS77K.pth
        LSDSAM_L_77K.pth
        LSDSAM_B_71K.pth
      Landslide77K/                 # optional, from Landslide77K_raw.zip

The script registers the four TAFT weights by default. Add --register-taft
when you also want to register the optional Landslide77K/Landslide71K TAFT
repository for DAFT training.

TAFT registration modes:
  * Default actual registration: when --register-taft is used without
    --no-standardize, the full Landslide77K tree is standardized/copied into
    registry/dataset_store. This makes the registry self-contained but needs
    about 95 GB of free disk space and can take a long time.
  * Lightweight registration: add --no-standardize to write only manifests and
    source_root metadata. No image/label files are copied; the extracted
    Landslide77K folder must stay in place because training reads from it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsd_app.core.paths import portable_path  # noqa: E402
from lsd_app.data.repository import DatasetStandardizationError, discover_taft_tree  # noqa: E402
from lsd_app.registry import RegistryError, list_datasets, list_weights, register_taft_tree, register_weight  # noqa: E402

WEIGHT_FILENAMES = {
    "LSDFormer_LS77K": "LSDFormer_LS77K.pth",
    "LSDSAM_H_LS77K": "LSDSAM_H_LS77K.pth",
    "LSDSAM_L_77K": "LSDSAM_L_77K.pth",
    "LSDSAM_B_71K": "LSDSAM_B_71K.pth",
}

LANDSLIDE71K_EXCLUDED_SUBSETS = {"ELNT", "LRSTTC", "RLZX__SAT", "RLZX__UAV"}


@dataclass(frozen=True)
class WeightSpec:
    weight_id: str
    filename: str
    model_id: str
    version: str
    weight_type: str
    adapter_profile: str
    adapter_name: str
    image_size: int
    input_mode: str
    channel_rule: str
    metrics_summary: str
    notes: str
    lora_base_r: int | None = None
    lora_base_alpha: int | None = None
    lora_task_r: int | None = None
    lora_task_alpha: int | None = None
    lora_dropout: float | None = None
    peft_adapter_name: str = ""
    target_modules: str = ""


WEIGHT_SPECS = (
    WeightSpec(
        weight_id="LSDFormer_LS77K",
        filename="LSDFormer_LS77K.pth",
        model_id="LSDFormer",
        version="LS77K",
        weight_type="checkpoint",
        adapter_profile="lsdformer-taft-ls77k",
        adapter_name="lsdformer-fast",
        image_size=512,
        input_mode="rgb",
        channel_rule="RGB",
        metrics_summary="TAFT Landslide77K release checkpoint.",
        notes="Release TAFT weight downloaded from LSDetector_data_weights_v1/weights.",
    ),
    WeightSpec(
        weight_id="LSDSAM_H_LS77K",
        filename="LSDSAM_H_LS77K.pth",
        model_id="LSDSAM-H",
        version="LS77K",
        weight_type="checkpoint",
        adapter_profile="lsdsam-h-taft-ls77k-peft-lora-r32-edge4-after512",
        adapter_name="lsdsam-fewshot",
        image_size=512,
        input_mode="rgb_wavelets",
        channel_rule="RGB + wavelets",
        metrics_summary="TAFT Landslide77K release checkpoint.",
        notes="Release TAFT weight downloaded from LSDetector_data_weights_v1/weights.",
        lora_base_r=32,
        lora_base_alpha=64,
        lora_task_r=32,
        lora_task_alpha=64,
        lora_dropout=0.05,
        peft_adapter_name="default",
        target_modules="image_encoder.blocks.*.(attn.qkv|attn.proj|mlp.lin1|mlp.lin2)",
    ),
    WeightSpec(
        weight_id="LSDSAM_L_77K",
        filename="LSDSAM_L_77K.pth",
        model_id="LSDSAM-L",
        version="LS77K",
        weight_type="checkpoint",
        adapter_profile="lsdsam-l-taft-ls77k-peft-lora-r32-edge4-after512",
        adapter_name="lsdsam-fewshot",
        image_size=512,
        input_mode="rgb_wavelets",
        channel_rule="RGB + wavelets",
        metrics_summary="TAFT Landslide77K release checkpoint.",
        notes="Release TAFT weight downloaded from LSDetector_data_weights_v1/weights.",
        lora_base_r=32,
        lora_base_alpha=64,
        lora_task_r=32,
        lora_task_alpha=64,
        lora_dropout=0.05,
        peft_adapter_name="default",
        target_modules="image_encoder.blocks.*.(attn.qkv|attn.proj|mlp.lin1|mlp.lin2)",
    ),
    WeightSpec(
        weight_id="LSDSAM_B_71K",
        filename="LSDSAM_B_71K.pth",
        model_id="LSDSAM-B",
        version="LS71K",
        weight_type="checkpoint",
        adapter_profile="lsdsam-b-taft-ls71k-peft-lora-r32-edge4-after512-cf-mh",
        adapter_name="lsdsam-fewshot",
        image_size=512,
        input_mode="rgb_wavelets",
        channel_rule="RGB + wavelets",
        metrics_summary="TAFT Landslide71K release checkpoint.",
        notes="Release TAFT weight downloaded from LSDetector_data_weights_v1/weights.",
        lora_base_r=32,
        lora_base_alpha=64,
        lora_task_r=32,
        lora_task_alpha=64,
        lora_dropout=0.05,
        peft_adapter_name="default",
        target_modules="image_encoder.blocks.*.(attn.qkv|attn.proj|mlp.lin1|mlp.lin2)",
    ),
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _registry_dir(registry_root: Path) -> Path:
    return registry_root / "registry"


def _find_weights_dir(args: argparse.Namespace) -> Path:
    if args.weights_dir:
        return Path(args.weights_dir).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve() if args.assets_root else None
    if assets_root is None:
        raise SystemExit("Provide --assets-root or --weights-dir.")
    return (assets_root / "weights").resolve()


def _find_taft_root(args: argparse.Namespace) -> Path:
    if args.taft_root:
        return Path(args.taft_root).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve() if args.assets_root else None
    if assets_root is None:
        raise SystemExit("Provide --taft-root or --assets-root when using --register-taft.")
    return (assets_root / "Landslide77K").resolve()


def _validate_weights_dir(weights_dir: Path) -> list[tuple[WeightSpec, Path]]:
    if not weights_dir.exists() or not weights_dir.is_dir():
        raise SystemExit(f"Weights directory does not exist: {weights_dir}")
    pairs: list[tuple[WeightSpec, Path]] = []
    missing: list[str] = []
    for spec in WEIGHT_SPECS:
        source = weights_dir / spec.filename
        if not source.is_file():
            missing.append(str(source))
        else:
            pairs.append((spec, source))
    if missing:
        raise SystemExit("Missing release weight file(s):\n  " + "\n  ".join(missing))
    return pairs


def _validate_taft_root(taft_root: Path) -> None:
    if not taft_root.exists() or not taft_root.is_dir():
        raise SystemExit(f"TAFT root does not exist: {taft_root}")
    expected = ["CAS", "LMHLD", "RLZX"]
    missing = [name for name in expected if not (taft_root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"TAFT root does not look like extracted Landslide77K_raw.zip: {taft_root}\n"
            f"Missing top-level folder(s): {', '.join(missing)}"
        )


def _taft_subset_overrides(taft_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    discovery = discover_taft_tree(taft_root, preset="landslide77k")
    errors = [*discovery.errors]
    for subset in discovery.subsets:
        errors.extend(subset.errors)
    if errors:
        raise DatasetStandardizationError("TAFT discovery failed: " + "; ".join(errors[:8]))
    overrides = {
        subset.source_rel: {
            "confirmed": True,
            "RGB_bands": subset.RGB_bands,
            "label_polarity": subset.label_polarity,
        }
        for subset in discovery.subsets
    }
    summary = {
        "subset_count": discovery.subset_count,
        "sample_count": discovery.sample_count,
        "train_count": discovery.train_count,
        "test_count": discovery.test_count,
    }
    return overrides, summary


def _print_weight_plan(pairs: list[tuple[WeightSpec, Path]]) -> None:
    print("Weights to register:")
    for spec, source in pairs:
        size_gb = source.stat().st_size / (1024 ** 3)
        print(f"  - {spec.weight_id}: {source} ({size_gb:.2f} GiB) -> {spec.model_id} / TAFT")


def _register_weights(registry_root: Path, pairs: list[tuple[WeightSpec, Path]], *, dry_run: bool) -> None:
    _print_weight_plan(pairs)
    if dry_run:
        print("Dry run: no weight registry files were changed.")
        return
    for spec, source in pairs:
        print(f"Registering weight {spec.weight_id} from {source} ...")
        entry = register_weight(
            registry_root=registry_root,
            model_id=spec.model_id,
            stage="TAFT",
            version=spec.version,
            path=source,
            metrics_summary=spec.metrics_summary,
            notes=spec.notes,
            weight_id=spec.weight_id,
            weight_type=spec.weight_type,
            adapter_profile=spec.adapter_profile,
            adapter_name=spec.adapter_name,
            image_size=spec.image_size,
            input_mode=spec.input_mode,
            channel_rule=spec.channel_rule,
            lora_base_r=spec.lora_base_r,
            lora_base_alpha=spec.lora_base_alpha,
            lora_task_r=spec.lora_task_r,
            lora_task_alpha=spec.lora_task_alpha,
            lora_dropout=spec.lora_dropout,
            peft_adapter_name=spec.peft_adapter_name,
            target_modules=spec.target_modules,
            store_copy=True,
        )
        print(f"  stored: {entry.path}")
    registered = [row for row in list_weights(registry_root=registry_root) if row.get("weight_id") in WEIGHT_FILENAMES]
    print(f"Registered TAFT release weights: {len(registered)}/4")


def _progress(row: Mapping[str, Any]) -> None:
    status = str(row.get("status") or "")
    processed = int(row.get("processed") or 0)
    total = int(row.get("total") or 0)
    current = str(row.get("current_subset") or row.get("message") or "")
    if total:
        print(f"  [{status}] {processed}/{total} {current}".rstrip())
    else:
        print(f"  [{status}] {current}".rstrip())


def _register_taft(registry_root: Path, taft_root: Path, *, standardize: bool, force_rebuild: bool, dry_run: bool) -> None:
    _validate_taft_root(taft_root)
    overrides, summary = _taft_subset_overrides(taft_root)
    print(
        "TAFT source discovered: "
        f"{summary['subset_count']} subsets, {summary['sample_count']} samples "
        f"({summary['train_count']} train / {summary['test_count']} test)."
    )
    if dry_run:
        print("Dry run: no TAFT dataset registry files were changed.")
        return
    result = register_taft_tree(
        registry_root=registry_root,
        name="Landslide77K",
        version="Landslide77K",
        root=taft_root,
        preset="landslide77k",
        standardize=standardize,
        resume=True,
        force_rebuild=force_rebuild,
        notes=(
            "Optional TAFT source from Landslide77K_raw.zip. Required only for DAFT; "
            "TSFT and inference do not need this dataset."
        ),
        subset_overrides=overrides,
        progress_callback=_progress,
    )
    _normalize_landslide77k_dataset_id(registry_root)
    print(f"Registered taft-landslide77k: {result.sample_count} samples, standardized={standardize}.")
    _register_landslide71k_reference(registry_root)


def _normalize_landslide77k_dataset_id(registry_root: Path) -> None:
    """Keep the public dataset id compatible with README/UI examples."""

    workspace = registry_root.resolve()
    root = _registry_dir(workspace)
    manifest_path = root / "dataset_store" / "TAFT" / "Landslide77K" / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path, {})
        if isinstance(manifest, Mapping):
            manifest = dict(manifest)
            manifest["dataset_id"] = "taft-landslide77k"
            _write_json(manifest_path, manifest)
    subsets_root = root / "dataset_store" / "TAFT" / "Landslide77K" / "subsets"
    if subsets_root.exists():
        for subset_manifest_path in subsets_root.glob("*/subset_manifest.json"):
            subset_manifest = _read_json(subset_manifest_path, {})
            if isinstance(subset_manifest, Mapping):
                subset_manifest = dict(subset_manifest)
                subset_manifest["dataset_id"] = "taft-landslide77k"
                _write_json(subset_manifest_path, subset_manifest)
    datasets_path = root / "datasets.json"
    rows = _read_json(datasets_path, [])
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        updated = dict(row)
        dataset_id = str(updated.get("dataset_id") or "")
        if dataset_id.lower() == "taft-landslide77k" or updated.get("version") == "Landslide77K":
            updated["dataset_id"] = "taft-landslide77k"
        normalized.append(updated)
    _write_json(datasets_path, normalized)


def _register_landslide71k_reference(registry_root: Path) -> None:
    workspace = registry_root.resolve()
    root = _registry_dir(workspace)
    source_manifest_path = root / "dataset_store" / "TAFT" / "Landslide77K" / "manifest.json"
    if not source_manifest_path.exists():
        raise RegistryError(f"Cannot create Landslide71K reference; missing source manifest: {source_manifest_path}")
    source_manifest = _read_json(source_manifest_path, {})
    source_subsets = source_manifest.get("subsets") if isinstance(source_manifest, Mapping) else None
    if not isinstance(source_subsets, list):
        raise RegistryError(f"Landslide77K manifest has no subset list: {source_manifest_path}")
    kept_subsets = [
        dict(row)
        for row in source_subsets
        if isinstance(row, Mapping) and str(row.get("subset_id") or "") not in LANDSLIDE71K_EXCLUDED_SUBSETS
    ]
    target_root = root / "dataset_store" / "TAFT" / "Landslide71K"
    target_manifest_path = target_root / "manifest.json"
    sample_count = sum(int(row.get("sample_count") or 0) for row in kept_subsets)
    train_count = sum(int(row.get("train_count") or 0) for row in kept_subsets)
    test_count = sum(int(row.get("test_count") or 0) for row in kept_subsets)
    created_at = str(source_manifest.get("created_at") or source_manifest.get("updated_at") or "")
    manifest = {
        "dataset_id": "taft-landslide71k",
        "dataset_class": "TAFT",
        "dataset_type": "TAFT",
        "name": "Landslide71K",
        "version": "Landslide71K",
        "root_dir": portable_path(target_root, workspace_root=workspace),
        "manifest_path": portable_path(target_manifest_path, workspace_root=workspace),
        "source": "Derived reference view from Landslide77K release asset",
        "source_root": source_manifest.get("source_root", ""),
        "source_imagery_path": source_manifest.get("source_imagery_path", ""),
        "source_label_path": source_manifest.get("source_label_path", ""),
        "derived_from": "taft-landslide77k",
        "reference_only": True,
        "excluded_subsets": sorted(LANDSLIDE71K_EXCLUDED_SUBSETS),
        "standardized": bool(source_manifest.get("standardized", False)),
        "split_policy": source_manifest.get("split_policy", "preserve"),
        "has_gt": True,
        "RGB_bands": "mixed",
        "sample_shape_layout": source_manifest.get("sample_shape_layout", "(H, W, C)"),
        "sample_dtype": source_manifest.get("sample_dtype", "source_dtype"),
        "sample_count": sample_count,
        "train_count": train_count,
        "test_count": test_count,
        "subset_count": len(kept_subsets),
        "subsets": kept_subsets,
        "created_at": created_at,
        "updated_at": created_at,
        "notes": "Reference-only Landslide71K view derived by excluding four Landslide77K subsets.",
        "description": "Reference-only Landslide71K view derived by excluding four Landslide77K subsets.",
    }
    _write_json(target_manifest_path, manifest)

    datasets_path = root / "datasets.json"
    rows = _read_json(datasets_path, [])
    rows = [
        row
        for row in rows
        if not (
            isinstance(row, Mapping)
            and (str(row.get("dataset_id") or "").lower() == "taft-landslide71k" or row.get("version") == "Landslide71K")
        )
    ]
    rows.append(
        {
            "dataset_id": "taft-landslide71k",
            "name": "Landslide71K",
            "dataset_class": "TAFT",
            "imagery_path": portable_path(target_root, workspace_root=workspace),
            "label_path": portable_path(target_root, workspace_root=workspace),
            "has_gt": True,
            "RGB_bands": "mixed",
            "RGB_bands_class": "MIXED",
            "tile_schema": "standardized-tree:(H, W, C)",
            "source": "Derived reference view from Landslide77K release asset",
            "created_at": created_at,
            "notes": "Reference-only Landslide71K view derived by excluding four Landslide77K subsets.",
            "description": "Reference-only Landslide71K view derived by excluding four Landslide77K subsets.",
            "dataset_type": "TAFT",
            "truth_path": portable_path(target_root, workspace_root=workspace),
            "version": "Landslide71K",
            "repository_path": portable_path(target_root, workspace_root=workspace),
            "manifest_path": portable_path(target_manifest_path, workspace_root=workspace),
            "sample_shape_layout": manifest["sample_shape_layout"],
            "sample_dtype": manifest["sample_dtype"],
            "sample_count": sample_count,
            "train_count": train_count,
            "test_count": test_count,
            "subset_count": len(kept_subsets),
            "reference_only": True,
            "derived_from": "taft-landslide77k",
            "excluded_subsets": sorted(LANDSLIDE71K_EXCLUDED_SUBSETS),
        }
    )
    _write_json(datasets_path, rows)
    print(f"Registered reference dataset taft-landslide71k: {sample_count} samples, {len(kept_subsets)} subsets.")


def main(argv: list[str] | None = None) -> int:
    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Register LSDetector release weights and optional Landslide77K/Landslide71K TAFT datasets.",
        epilog="""TAFT registration modes:
  actual copy: use --register-taft without --no-standardize; the full TAFT tree is copied/standardized into registry/dataset_store.
  lightweight: add --no-standardize; only manifests and source_root metadata are written, so the extracted Landslide77K folder must remain available.

Examples:
  Register release weights only:
    python tools/register_release_assets.py --assets-root /path/to/LSDetector_data_weights_v1
  Register TAFT with actual copied storage:
    python tools/register_release_assets.py --assets-root /path/to/LSDetector_data_weights_v1 --skip-weights --register-taft --force-rebuild
  Register TAFT in lightweight metadata-only mode:
    python tools/register_release_assets.py --assets-root /path/to/LSDetector_data_weights_v1 --skip-weights --register-taft --no-standardize
""",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--registry-root", default=str(PROJECT_ROOT), help="LSDetector project/workspace root containing registry/.")
    parser.add_argument("--assets-root", help="Unpacked Baidu root, e.g. /path/LSDetector_data_weights_v1.")
    parser.add_argument("--weights-dir", help="Directory containing the four release .pth files. Overrides --assets-root/weights.")
    parser.add_argument("--taft-root", help="Extracted Landslide77K folder. Overrides --assets-root/Landslide77K.")
    parser.add_argument("--skip-weights", action="store_true", help="Do not register the four TAFT weights.")
    parser.add_argument("--register-taft", action="store_true", help="Also register Landslide77K and Landslide71K TAFT datasets. By default this performs actual copied/standardized storage unless --no-standardize is set.")
    parser.add_argument("--no-standardize", action="store_true", help="Lightweight TAFT registration: write manifests/source_root only and do not copy image/label files into registry/dataset_store. Keep the extracted Landslide77K folder available.")
    parser.add_argument("--force-rebuild", action="store_true", help="Delete and rebuild the target TAFT registry version before registration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned actions without writing registry files.")
    args = parser.parse_args(argv)

    registry_root = Path(args.registry_root).expanduser().resolve()
    if not registry_root.exists():
        raise SystemExit(f"Registry root does not exist: {registry_root}")

    if not args.skip_weights:
        pairs = _validate_weights_dir(_find_weights_dir(args))
        _register_weights(registry_root, pairs, dry_run=args.dry_run)

    if args.register_taft:
        taft_root = _find_taft_root(args)
        _register_taft(
            registry_root,
            taft_root,
            standardize=not args.no_standardize,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print("\nRegistry summary:")
        print(f"  weights: {len(list_weights(registry_root=registry_root))}")
        print(f"  datasets: {len(list_datasets(registry_root=registry_root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
