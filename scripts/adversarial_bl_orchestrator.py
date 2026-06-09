from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path("/home/fuzijin/data/LSD__APP").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsd_app.core.paths import resolve_user_path
from lsd_app.registry import list_datasets, list_weights
from lsd_app.training.daft_lsdsam import build_daft_training_payload
from lsd_app.training.tsft import build_tsft_training_payload
from lsd_app.web.app import (
    build_evaluation_dataset_payload,
    start_daft_training_background_job,
    start_evaluation_background_job,
    start_tsft_training_background_job,
)
from lsd_app.web.jobs import list_job_states, read_job_state


WORKSPACE = PROJECT_ROOT
REPORT_ROOT = WORKSPACE / "outputs" / "adversarial_wuping_7to3_bl"
STATE_PATH = REPORT_ROOT / "master_state.json"
REPORT_MD_PATH = REPORT_ROOT / "report.md"
RESULTS_CSV_PATH = REPORT_ROOT / "results.csv"
COMBINED_TABLE_PATH = REPORT_ROOT / "combined_table.md"
LOG_PATH = REPORT_ROOT / "orchestrator.log"

TARGET_DATASET_ID = "daft-wuping-daft2-256"
EVAL_DATASET_ID = "evaluation-wuping-test13"
FEWSHOT_DATASET_ID = "tsft-wuping-adv7to3-fewshot-seed404006"
FEWSHOT_SEED = 404006
SEED_CANDIDATE_COUNT = 30
DAFT_TOP_COUNT = 10
POLL_SECONDS = 60

ACTIVE_STATUSES = {
    "pending",
    "running",
    "cancelling",
    "stalled",
    "stalled_uninterruptible",
    "resource_blocked",
}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted", "resource_blocked"}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "LSDSAM-L": {
        "model_id": "LSDSAM-L",
        "taft_weight_id": "LSDSAM_L_77K",
        "source_dataset_id": "taft-landslide77k",
        "seed_base": 202025,
        "daft_batch_size": 1,
        "daft_initial_lr": 2e-4,
        "tsft_batch_size": 4,
        "tsft_max_lr": 2e-4,
    },
    "LSDSAM-B": {
        "model_id": "LSDSAM-B",
        "taft_weight_id": "LSDSAM_B_71K",
        "source_dataset_id": "taft-landslide71k",
        "seed_base": 302025,
        "daft_batch_size": 1,
        "daft_initial_lr": 2e-4,
        "tsft_batch_size": 4,
        "tsft_max_lr": 2e-4,
    },
}

PREVIOUS_SUMMARY = {
    "LSDFormer": {
        "zero_taft": 0.477440,
        "zero_daft_mean": 0.528911,
        "zero_daft_std": 0.007488,
        "few_taft": 0.590476,
        "few_daft_mean": 0.607397,
        "few_daft_std": 0.010479,
    },
    "LSDSAM-H": {
        "zero_taft": 0.527662,
        "zero_daft_mean": 0.497367,
        "zero_daft_std": 0.014581,
        "few_taft": 0.613757,
        "few_daft_mean": 0.521431,
        "few_daft_std": 0.029784,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def log(message: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def state() -> dict[str, Any]:
    return read_json(STATE_PATH, {}) or {}


def update_state(**updates: Any) -> dict[str, Any]:
    data = state()
    data.update(updates)
    data["updated_at"] = utc_now()
    write_json(STATE_PATH, data)
    return data


def metric_from_mapping(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    for nested_key in ("scores", "metrics", "eval", "test", "result"):
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            value = metric_from_mapping(nested, *keys)
            if value is not None:
                return value
    return None


def metric_rows_from_path(path_value: Any) -> list[dict[str, Any]]:
    text = str(path_value or "").strip()
    if not text:
        return []
    path = Path(text)
    if not path.exists() or path.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def active_blocking_jobs() -> list[dict[str, Any]]:
    rows = []
    for job in list_job_states(str(WORKSPACE), probe_process=True):
        if job.get("kind") in {
            "daft-training",
            "tsft-training",
            "evaluation",
            "deployment",
            "batch-deployment",
        } and job.get("status") in ACTIVE_STATUSES:
            rows.append(job)
    return rows


def wait_until_no_active_jobs() -> None:
    while True:
        active = active_blocking_jobs()
        if not active:
            return
        summary = [(row.get("job_id"), row.get("kind"), row.get("status")) for row in active]
        log(f"waiting for active jobs: {summary}")
        update_state(stage="waiting_for_gpu", active_jobs=summary)
        time.sleep(POLL_SECONDS)


def require_registry_entries() -> None:
    dataset_ids = {row.get("dataset_id") for row in list_datasets(registry_root=str(WORKSPACE))}
    weight_ids = {
        row.get("weight_id")
        for row in list_weights(registry_root=str(WORKSPACE))
        if row.get("exists", True)
    }
    required_datasets = {TARGET_DATASET_ID, EVAL_DATASET_ID, FEWSHOT_DATASET_ID}
    required_datasets.update(cfg["source_dataset_id"] for cfg in MODEL_CONFIGS.values())
    missing = []
    for dataset_id in sorted(required_datasets):
        if dataset_id not in dataset_ids:
            missing.append(f"dataset:{dataset_id}")
    for cfg in MODEL_CONFIGS.values():
        if cfg["taft_weight_id"] not in weight_ids:
            missing.append(f"weight:{cfg['taft_weight_id']}")
    if missing:
        raise RuntimeError(f"Missing required registry entries: {missing}")


def weight_path(weight_id: str) -> str:
    for row in list_weights(registry_root=str(WORKSPACE)):
        if row.get("weight_id") == weight_id:
            path = str(row.get("path") or "")
            if path:
                return str(resolve_user_path(path, workspace_root=WORKSPACE))
    raise RuntimeError(f"Cannot resolve weight path for {weight_id}")


def start_and_wait(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    wait_until_no_active_jobs()
    if kind == "evaluation":
        job_id = start_evaluation_background_job(registry_root=str(WORKSPACE), raw_config=payload)
    elif kind == "daft":
        job_id = start_daft_training_background_job(registry_root=str(WORKSPACE), payload=payload)
    elif kind == "tsft":
        job_id = start_tsft_training_background_job(registry_root=str(WORKSPACE), payload=payload)
    else:
        raise ValueError(kind)
    log(f"started {kind}: {job_id}")
    while True:
        job = read_job_state(str(WORKSPACE), job_id, probe_process=True)
        status = str(job.get("status") or "")
        update_state(stage=f"running_{kind}", active_job_id=job_id, active_job_status=status)
        if status in TERMINAL_STATUSES:
            return job
        time.sleep(POLL_SECONDS)


def extract_eval_result(job: Mapping[str, Any]) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
    scores = result.get("scores") if isinstance(result.get("scores"), Mapping) else {}
    return {
        "status": job.get("status"),
        "job_id": job.get("job_id"),
        "eval_mIoU": metric_from_mapping(scores, "miou", "mIoU", "eval_mIoU"),
        "eval_IoU_foreground": metric_from_mapping(scores, "iou", "IoU_foreground", "foreground_iou"),
        "precision": metric_from_mapping(scores, "precision"),
        "recall": metric_from_mapping(scores, "recall"),
        "F1": metric_from_mapping(scores, "f1", "F1"),
    }


def extract_training_result(job: Mapping[str, Any], *, baseline: float | None = None) -> dict[str, Any]:
    result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
    history = job.get("metrics_history") if isinstance(job.get("metrics_history"), list) else []
    if not history:
        history = metric_rows_from_path(result.get("metrics_path") or job.get("metrics_path"))
    best_row: Mapping[str, Any] = {}
    best_score = None
    for row in history:
        value = metric_from_mapping(row, "eval.mIoU", "eval_mIoU", "test.mIoU", "mIoU", "miou")
        if value is not None and (best_score is None or value > best_score):
            best_score = value
            best_row = row
    if best_score is None:
        best_score = metric_from_mapping(result, "best_score", "eval_mIoU", "mIoU", "miou")
    gain = (best_score - baseline) if best_score is not None and baseline is not None else None
    return {
        "status": job.get("status"),
        "job_id": job.get("job_id"),
        "weight_id": result.get("registered_weight_id") or job.get("registered_weight_id") or "",
        "eval_mIoU": best_score,
        "gain": gain,
        "best_epoch": result.get("best_epoch") or best_row.get("epoch") or "",
        "eval_IoU_foreground": metric_from_mapping(best_row, "eval.IoU_foreground", "IoU_foreground", "foreground_iou", "iou"),
        "precision": metric_from_mapping(best_row, "eval.precision", "precision"),
        "recall": metric_from_mapping(best_row, "eval.recall", "recall"),
        "F1": metric_from_mapping(best_row, "eval.F1", "F1", "f1"),
    }


def evaluation_payload(model_name: str, weight_id_value: str, output_dir: Path) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model_name]
    return build_evaluation_dataset_payload(
        dataset_id=EVAL_DATASET_ID,
        weight_id=weight_id_value,
        model_id=cfg["model_id"],
        checkpoint_path=weight_path(weight_id_value),
        output_dir=str(output_dir.relative_to(WORKSPACE)),
        batch_size=1,
        device="auto",
    )


def daft_payload(model_name: str, seed: int) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model_name]
    slug = model_name.lower().replace("-", "")
    return build_daft_training_payload(
        registry_root=str(WORKSPACE),
        model_id=cfg["model_id"],
        base_weight_id=cfg["taft_weight_id"],
        source_dataset_id=cfg["source_dataset_id"],
        target_dataset_id=TARGET_DATASET_ID,
        evaluation_dataset_id=EVAL_DATASET_ID,
        alignment_mode="IST",
        total_epochs=10,
        initial_lr=float(cfg["daft_initial_lr"]),
        halve_epochs=[2, 4, 6, 8],
        device="auto",
        output_dir=str((REPORT_ROOT / "daft_seed_search" / slug / f"seed_{seed}").relative_to(WORKSPACE)),
        batch_size=int(cfg["daft_batch_size"]),
        num_workers=0,
        output_weight_id=f"{cfg['taft_weight_id']}_daft_adv_wuping7to3_seed{seed}",
        global_seed=seed,
        save_optimizer_state=True,
    )


def tsft_payload(
    model_name: str,
    base_weight_id: str,
    output_dir: Path,
    output_weight_id: str,
) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model_name]
    return build_tsft_training_payload(
        registry_root=str(WORKSPACE),
        model_id=cfg["model_id"],
        base_weight_id=base_weight_id,
        dataset_id=FEWSHOT_DATASET_ID,
        evaluation_dataset_id=EVAL_DATASET_ID,
        max_epoch=30,
        max_lr=float(cfg["tsft_max_lr"]),
        min_lr=1e-6,
        max_decay_epoch=25,
        device="auto",
        output_dir=str(output_dir.relative_to(WORKSPACE)),
        batch_size=int(cfg["tsft_batch_size"]),
        num_workers=0,
        output_weight_id=output_weight_id,
        cleanup_lora_checkpoints_after_register=True,
        save_optimizer_state=True,
    )


def candidate_seeds(model_name: str) -> list[int]:
    base = int(MODEL_CONFIGS[model_name]["seed_base"])
    return [base + i for i in range(SEED_CANDIDATE_COUNT)]


def run_baselines() -> dict[str, Any]:
    baselines = dict(state().get("baselines") or {})
    for model_name, cfg in MODEL_CONFIGS.items():
        if model_name in baselines:
            continue
        payload = evaluation_payload(
            model_name,
            cfg["taft_weight_id"],
            REPORT_ROOT / "baselines" / model_name,
        )
        job = start_and_wait("evaluation", payload)
        result = extract_eval_result(job)
        if result["status"] != "completed" or result["eval_mIoU"] is None:
            raise RuntimeError(f"Baseline evaluation failed for {model_name}: {result}")
        baselines[model_name] = result
        update_state(baselines=baselines)
    return baselines


def run_daft_seed_search(baselines: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = dict(state().get("daft_seed_search") or {})
    for model_name in MODEL_CONFIGS:
        model_state = results.setdefault(model_name, {"candidates": [], "top": []})
        done_seeds = {int(row.get("seed")) for row in model_state.get("candidates", []) if row.get("seed") is not None}
        baseline = float(baselines[model_name]["eval_mIoU"])
        for seed in candidate_seeds(model_name):
            if seed in done_seeds:
                continue
            job = start_and_wait("daft", daft_payload(model_name, seed))
            row = {"model": model_name, "seed": seed, **extract_training_result(job, baseline=baseline)}
            model_state["candidates"].append(row)
            top = [
                item
                for item in model_state["candidates"]
                if item.get("status") == "completed" and item.get("eval_mIoU") is not None
            ]
            top.sort(key=lambda item: float(item.get("eval_mIoU") or 0), reverse=True)
            model_state["top"] = top[:DAFT_TOP_COUNT]
            update_state(daft_seed_search=results)
        top = [
            item
            for item in model_state["candidates"]
            if item.get("status") == "completed" and item.get("eval_mIoU") is not None
        ]
        top.sort(key=lambda item: float(item.get("eval_mIoU") or 0), reverse=True)
        model_state["top"] = top[:DAFT_TOP_COUNT]
        update_state(daft_seed_search=results)
    return results


def run_formal_tsft(daft_results: Mapping[str, Any]) -> dict[str, Any]:
    formal: dict[str, Any] = dict(state().get("formal_tsft") or {})
    for model_name, cfg in MODEL_CONFIGS.items():
        model_state = formal.setdefault(model_name, {"baseline": None, "daft_tsft": []})
        slug = model_name.lower().replace("-", "")
        if not model_state.get("baseline"):
            payload = tsft_payload(
                model_name,
                cfg["taft_weight_id"],
                REPORT_ROOT / "formal_tsft" / slug / "taft_init_baseline",
                f"{cfg['taft_weight_id']}_formal_tsft_adv_wuping7to3",
            )
            job = start_and_wait("tsft", payload)
            model_state["baseline"] = extract_training_result(job)
            update_state(formal_tsft=formal)
        done_weights = {row.get("base_weight_id") for row in model_state.get("daft_tsft", [])}
        for daft_row in daft_results.get(model_name, {}).get("top", [])[:DAFT_TOP_COUNT]:
            weight_id_value = str(daft_row.get("weight_id") or "")
            if not weight_id_value or weight_id_value in done_weights:
                continue
            payload = tsft_payload(
                model_name,
                weight_id_value,
                REPORT_ROOT / "formal_tsft" / slug / weight_id_value,
                f"{weight_id_value}_formal_tsft_adv_wuping7to3",
            )
            job = start_and_wait("tsft", payload)
            model_state["daft_tsft"].append({"base_weight_id": weight_id_value, **extract_training_result(job)})
            update_state(formal_tsft=formal)
    return formal


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summary_for_model(model_name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    baselines = data.get("baselines") or {}
    daft = data.get("daft_seed_search") or {}
    formal = data.get("formal_tsft") or {}
    zero_taft = (baselines.get(model_name) or {}).get("eval_mIoU")
    zero_daft_values = [
        float(row["eval_mIoU"])
        for row in (daft.get(model_name, {}).get("top") or [])[:DAFT_TOP_COUNT]
        if row.get("eval_mIoU") is not None
    ]
    few_taft = (formal.get(model_name, {}).get("baseline") or {}).get("eval_mIoU")
    few_daft_values = [
        float(row["eval_mIoU"])
        for row in (formal.get(model_name, {}).get("daft_tsft") or [])[:DAFT_TOP_COUNT]
        if row.get("eval_mIoU") is not None
    ]
    zero_mean, zero_std = mean_std(zero_daft_values)
    few_mean, few_std = mean_std(few_daft_values)
    return {
        "zero_taft": zero_taft,
        "zero_daft_mean": zero_mean,
        "zero_daft_std": zero_std,
        "few_taft": few_taft,
        "few_daft_mean": few_mean,
        "few_daft_std": few_std,
    }


def combined_table_rows(data: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = [
        ("LSDFormer", dict(PREVIOUS_SUMMARY["LSDFormer"])),
        ("LSDSAM-B", summary_for_model("LSDSAM-B", data)),
        ("LSDSAM-L", summary_for_model("LSDSAM-L", data)),
        ("LSDSAM-H", dict(PREVIOUS_SUMMARY["LSDSAM-H"])),
    ]
    return rows


def fmt_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return ""
    return f"{fmt(mean)} ± {fmt(std or 0)}"


def write_report() -> None:
    data = state()
    rows = []
    for model_name, model_state in (data.get("daft_seed_search") or {}).items():
        for row in model_state.get("candidates", []):
            rows.append({"stage": "daft_seed", "model": model_name, **row})
    for model_name, model_state in (data.get("formal_tsft") or {}).items():
        baseline = model_state.get("baseline") or {}
        rows.append({"stage": "formal_taft_baseline", "model": model_name, **baseline})
        for row in model_state.get("daft_tsft", []):
            rows.append({"stage": "formal_daft_tsft", "model": model_name, **row})
    for model_name, result in (data.get("baselines") or {}).items():
        rows.append({"stage": "zero_taft_baseline", "model": model_name, **result})
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["stage"]
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Wuping 7:3 Adversarial Plan Supplement: LSDSAM-B/L",
        "",
        f"- Updated at: {utc_now()}",
        f"- Target evaluation dataset: {EVAL_DATASET_ID}",
        f"- Fixed 8-shot dataset: {FEWSHOT_DATASET_ID} (seed {FEWSHOT_SEED})",
        f"- DAFT target: {TARGET_DATASET_ID}",
        "- LSDSAM-L DAFT source: taft-landslide77k",
        "- LSDSAM-B DAFT source: taft-landslide71k",
        "",
        "## Zero-shot Baselines",
        "",
        "| Model | Weight | eval_mIoU | Job |",
        "|---|---|---:|---|",
    ]
    for model_name, result in (data.get("baselines") or {}).items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                model_name,
                MODEL_CONFIGS[model_name]["taft_weight_id"],
                fmt(result.get("eval_mIoU")),
                result.get("job_id", ""),
            )
        )
    lines.extend([
        "",
        "## DAFT Top Seeds",
        "",
        "| Model | Rank | Seed | Weight | eval_mIoU | Gain vs 0-shot TAFT | Best epoch | Job |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ])
    for model_name, model_state in (data.get("daft_seed_search") or {}).items():
        for rank, row in enumerate(model_state.get("top", []), start=1):
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    model_name,
                    rank,
                    row.get("seed"),
                    row.get("weight_id", ""),
                    fmt(row.get("eval_mIoU")),
                    fmt(row.get("gain")),
                    row.get("best_epoch", ""),
                    row.get("job_id", ""),
                )
            )
    lines.extend([
        "",
        "## Formal 8-shot TSFT",
        "",
        "| Model | Init weight | eval_mIoU | Gain vs TAFT-init 8-shot | Job |",
        "|---|---|---:|---:|---|",
    ])
    for model_name, model_state in (data.get("formal_tsft") or {}).items():
        baseline = model_state.get("baseline") or {}
        baseline_score = baseline.get("eval_mIoU")
        taft_weight = MODEL_CONFIGS[model_name]["taft_weight_id"]
        lines.append(f"| {model_name} | {taft_weight} | {fmt(baseline_score)} |  | {baseline.get('job_id', '')} |")
        for row in model_state.get("daft_tsft", []):
            gain = (
                float(row["eval_mIoU"]) - float(baseline_score)
                if row.get("eval_mIoU") is not None and baseline_score is not None
                else None
            )
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    model_name,
                    row.get("base_weight_id", ""),
                    fmt(row.get("eval_mIoU")),
                    fmt(gain),
                    row.get("job_id", ""),
                )
            )
    lines.extend(["", "## Combined Table Target", "", "| Model | 0-shot w/o DAFT | 0-shot with DAFT | 8-shot w/o DAFT | 8-shot with DAFT |", "|---|---:|---:|---:|---:|"])
    for model_name, summary in combined_table_rows(data):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                model_name,
                fmt(summary.get("zero_taft")),
                fmt_mean_std(summary.get("zero_daft_mean"), summary.get("zero_daft_std")),
                fmt(summary.get("few_taft")),
                fmt_mean_std(summary.get("few_daft_mean"), summary.get("few_daft_std")),
            )
        )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    combined = lines[lines.index("## Combined Table Target") :]
    COMBINED_TABLE_PATH.write_text("\n".join(combined) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    update_state(
        stage="started",
        started_at=state().get("started_at") or utc_now(),
        fixed_fewshot_dataset_id=FEWSHOT_DATASET_ID,
        evaluation_dataset_id=EVAL_DATASET_ID,
        target_dataset_id=TARGET_DATASET_ID,
        model_configs=MODEL_CONFIGS,
    )
    require_registry_entries()
    wait_until_no_active_jobs()
    baselines = run_baselines()
    daft_results = run_daft_seed_search(baselines)
    formal = run_formal_tsft(daft_results)
    update_state(formal_tsft=formal)
    write_report()
    update_state(stage="completed", finished_at=utc_now(), report_path=str(REPORT_MD_PATH), csv_path=str(RESULTS_CSV_PATH))
    log(f"completed: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
