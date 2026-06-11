"""Fresh-interpreter worker for Streamlit background ML jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsd_app.web.jobs import (  # noqa: E402
    JobCancelled,
    JobResourceBlocked,
    _json_ready,
    _read_json,
    _utc_now,
    _with_job_duration,
    _write_json_atomic,
    record_job_progress,
    release_job_device_locks,
    start_job_heartbeat,
    state_path_for_job,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one LSDetector background ML job.")
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)

    registry_root = Path(args.registry_root).expanduser().resolve()
    state_path = state_path_for_job(registry_root, args.job_id)
    state = _read_json(state_path)
    if not state:
        raise SystemExit(f"Job state not found: {state_path}")

    job_id = str(state.get("job_id") or args.job_id)
    kind = str(state.get("kind") or "")
    description = str(state.get("description") or job_id)
    payload = state.get("payload") if isinstance(state.get("payload"), Mapping) else {}
    worker_pid = os.getpid()

    process_group_id = os.getpgrp()

    def progress_callback(progress: Mapping[str, Any]) -> None:
        record_job_progress(
            registry_root=registry_root,
            job_id=job_id,
            kind=kind,
            description=description,
            payload=payload,
            progress={**dict(progress), "execution_mode": "subprocess"},
            worker_pid=worker_pid,
            process_group_id=process_group_id,
        )

    heartbeat_stop = start_job_heartbeat(
        registry_root=registry_root,
        job_id=job_id,
        kind=kind,
        description=description,
        payload=payload,
        worker_pid=worker_pid,
        process_group_id=process_group_id,
    )
    try:
        progress_callback(
            {
                "status": "running",
                "phase": "startup",
                "message": "Running in a fresh Python subprocess.",
                "process_pid": worker_pid,
                "worker_pid": worker_pid,
                "process_group_id": process_group_id,
            }
        )
        result = _run_dispatch(
            kind=kind,
            registry_root=str(registry_root),
            payload=payload,
            progress_callback=progress_callback,
        )
    except JobCancelled as exc:
        heartbeat_stop.set()
        release_job_device_locks(registry_root, job_id)
        current = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **current,
                "status": "cancelled",
                "cancel_requested": True,
                "stall_reason": "",
                "message": str(exc) or "Cancelled by user.",
                "process_pid": worker_pid,
                "worker_pid": worker_pid,
                "process_group_id": process_group_id,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )
        return 0
    except JobResourceBlocked as exc:
        heartbeat_stop.set()
        current = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **current,
                **_json_ready(exc.details),
                "status": "resource_blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_pid": worker_pid,
                "worker_pid": worker_pid,
                "process_group_id": process_group_id,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )
        return 2
    except BaseException as exc:  # pragma: no cover - exercised by runtime failures.
        heartbeat_stop.set()
        release_job_device_locks(registry_root, job_id)
        current = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **current,
                "status": "failed",
                "stall_reason": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
                "process_pid": worker_pid,
                "worker_pid": worker_pid,
                "process_group_id": process_group_id,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )
        return 1

    heartbeat_stop.set()
    release_job_device_locks(registry_root, job_id)
    current = _read_json(state_path)
    _write_json_atomic(
        state_path,
        _with_job_duration({
            **current,
            "status": "completed",
            "percent": 1.0,
            "stall_reason": "",
            "result": _json_ready(result),
            "message": "Completed.",
            "process_pid": worker_pid,
            "worker_pid": worker_pid,
            "process_group_id": process_group_id,
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
        }),
    )
    return 0


def _run_dispatch(
    *,
    kind: str,
    registry_root: str,
    payload: Mapping[str, Any],
    progress_callback: Any,
) -> Any:
    if kind == "daft-training":
        from lsd_app.training.daft_lsdsam import run_lsdsam_daft
        from lsd_app.web.app import _daft_config_from_payload

        config = _daft_config_from_payload(registry_root=registry_root, payload=payload)
        result = run_lsdsam_daft(config, progress_callback=progress_callback)
        return result.to_dict() if hasattr(result, "to_dict") else result

    if kind == "tsft-training":
        from lsd_app.training.tsft import run_lsdsam_tsft
        from lsd_app.web.app import _tsft_config_from_payload

        config = _tsft_config_from_payload(registry_root=registry_root, payload=payload)
        result = run_lsdsam_tsft(config, progress_callback=progress_callback)
        return result.to_dict() if hasattr(result, "to_dict") else result

    if kind == "evaluation":
        from lsd_app.web.app import run_evaluation_background_task

        result = run_evaluation_background_task(
            raw_config={**dict(payload), "registry_root": registry_root},
            progress_callback=progress_callback,
        )
        return result.to_dict() if hasattr(result, "to_dict") else result

    if kind == "deployment":
        from lsd_app.web.app import normalize_web_config, run_deployment_background_task

        raw_config = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else payload
        config = normalize_web_config(raw_config, create_output_dirs=True)
        result = run_deployment_background_task(
            config,
            raw_config=raw_config,
            progress_callback=progress_callback,
        )
        return result.to_dict() if hasattr(result, "to_dict") else result

    if kind == "batch-deployment":
        from lsd_app.web.app import run_batch_deployment_background_task

        result = run_batch_deployment_background_task(
            payload=payload,
            progress_callback=progress_callback,
        )
        return result.to_dict() if hasattr(result, "to_dict") else result

    if kind == "dataset-upload":
        from lsd_app.web.app import run_dataset_upload_background_task

        result = run_dataset_upload_background_task(
            registry_root=registry_root,
            payload=payload,
            progress_callback=progress_callback,
        )
        return result.to_dict() if hasattr(result, "to_dict") else result

    raise RuntimeError(f"Unsupported subprocess job kind: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
