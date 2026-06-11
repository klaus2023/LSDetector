"""Small persistent background-job helpers for the Streamlit workbench."""

from __future__ import annotations

from collections.abc import Mapping as RuntimeMapping

from builtins import (
    isinstance as _builtin_isinstance,
    list as _builtin_list,
    str as _builtin_str,
    tuple as _builtin_tuple,
)
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid
import traceback
from typing import Any, Callable, Mapping

from lsd_app.core.paths import registry_dir

try:  # numpy is optional for unit tests that only exercise job plumbing.
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover
    _np = None


JobTarget = Callable[[Callable[[Mapping[str, Any]], None]], Any]


class JobCancelled(BaseException):
    """Raised inside a background job after the UI requests cancellation."""


class JobResourceBlocked(RuntimeError):
    """Raised when a CUDA job cannot start because another job owns the device."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class BackgroundJob:
    """Metadata for one locally running or completed workbench job."""

    job_id: str
    state_path: Path


_RUNNING_THREADS: dict[str, threading.Thread] = {}
_RUNNING_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}
_LOCK = threading.RLock()
ACTIVE_STATUSES = {"pending", "running", "cancelling", "stalled", "stalled_uninterruptible"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted", "resource_blocked"}
LONG_RUNNING_DEVICE_LOCK_KINDS = {"daft-training", "tsft-training", "deployment", "batch-deployment", "evaluation"}
STALE_THREAD_GRACE_SECONDS = 10.0
JOB_HEARTBEAT_INTERVAL_SECONDS = 5.0
JOB_STALL_SECONDS = 300.0
JOB_STALL_LONG_PHASE_SECONDS = 600.0
JOB_PROGRESS_STALL_SECONDS = 1200.0
JOB_PROGRESS_STALL_LONG_PHASE_SECONDS = 1800.0
LONG_STALL_PHASE_TOKENS = ("checkpoint", "merge", "register", "write", "output", "manifest")
PERSISTED_PROGRESS_LOG_SUFFIX = ".progress.jsonl"
PERSISTED_PROGRESS_LOG_KINDS = {"checkpoint", "merge", "register", "write_output", "dataset_build"}


def job_root(registry_root: str | Path) -> Path:
    root = registry_dir(registry_root) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def stable_job_id(*, kind: str, version: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{_safe_token(kind)}-{_safe_token(version)}-{digest}"


def state_path_for_job(registry_root: str | Path, job_id: str) -> Path:
    return job_root(registry_root) / f"{job_id}.json"


def device_lock_root(registry_root: str | Path) -> Path:
    root = job_root(registry_root) / "device_locks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_job_state(registry_root: str | Path, job_id: str, *, probe_process: bool = False) -> dict[str, Any]:
    path = state_path_for_job(registry_root, job_id)
    return _normalize_stale_job_state(_read_json(path), state_path=path, probe_process=probe_process)


def list_job_states(
    registry_root: str | Path,
    *,
    kind: str | None = None,
    probe_process: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = (path for path in job_root(registry_root).glob("*.json") if not path.name.startswith("._"))
    for path in sorted(paths):
        state = _normalize_stale_job_state(_read_json(path), state_path=path, probe_process=probe_process)
        if not state:
            continue
        if kind is not None and state.get("kind") != kind:
            continue
        state["state_path"] = str(path)
        rows.append(state)
    return rows


def start_process_background_job(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    target: JobTarget,
) -> BackgroundJob:
    """Start a persistent ML job in a fresh Python subprocess.

    CUDA cannot be safely initialized from a forked Streamlit process.  The
    callable target is kept for the older in-process API shape, but DAFT/TSFT
    jobs are dispatched by kind from ``lsd_app.web.job_worker`` inside a clean
    interpreter.
    """

    state_path = state_path_for_job(registry_root, job_id)
    registry_root_path = Path(registry_root).expanduser().resolve()
    existing = _normalize_stale_job_state(_read_json(state_path), state_path=state_path, probe_process=True)
    if existing.get("status") in ACTIVE_STATUSES:
        existing_pid = _int_pid(existing.get("process_pid"))
        if is_job_running(job_id) or _job_process_is_active(existing, existing_pid):
            return BackgroundJob(job_id=job_id, state_path=state_path)
    if existing.get("status") == "completed":
        return BackgroundJob(job_id=job_id, state_path=state_path)

    initial_state = _initial_job_state(
        existing=existing,
        job_id=job_id,
        kind=kind,
        description=description,
        payload=payload,
        execution_mode="subprocess",
    )
    initial_state["progress_log_path"] = str(_progress_log_path_for_state_path(state_path))
    blocking_lock = _startup_blocking_device_lock(
        registry_root_path,
        job_id=job_id,
        kind=kind,
        payload=payload,
    )
    if blocking_lock is not None:
        _write_json_atomic(state_path, _resource_blocked_state(initial_state, blocking_lock))
        return BackgroundJob(job_id=job_id, state_path=state_path)

    _write_json_atomic(state_path, initial_state)

    project_root = Path(__file__).resolve().parents[2]
    log_path = state_path.with_suffix(".log")
    env = _subprocess_python_env()
    cmd = [
        sys.executable,
        "-m",
        "lsd_app.web.job_worker",
        "--registry-root",
        str(registry_root_path),
        "--job-id",
        job_id,
    ]
    log_handle = log_path.open("ab")
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    log_handle.close()
    worker_identity = _worker_identity_fields(process.pid)
    update_job_state(
        registry_root,
        job_id,
        {
            **worker_identity,
            "process_pid": process.pid,
            "worker_pid": process.pid,
            "process_group_id": process.pid,
            "execution_mode": "subprocess",
            "log_path": str(log_path),
            "status": "running",
            "heartbeat_at": _utc_now(),
            "phase_started_at": _utc_now(),
            "message": "Running in a fresh Python subprocess.",
        },
    )

    def monitor_process() -> None:
        exit_code = process.wait()
        with _LOCK:
            _RUNNING_PROCESSES.pop(job_id, None)
            _RUNNING_THREADS.pop(job_id, None)
        release_job_device_locks(registry_root_path, job_id)
        state = _read_json(state_path)
        if not state or state.get("status") not in ACTIVE_STATUSES:
            return
        cancel_requested = bool(state.get("cancel_requested"))
        status = "cancelled" if cancel_requested else "failed"
        message = (
            "Cancelled by terminating the training process."
            if cancel_requested
            else f"Training process exited unexpectedly with code {exit_code}."
        )
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **state,
                "status": status,
                "stall_reason": "",
                "message": message,
                "error_type": "ProcessExit" if not cancel_requested else "",
                "error": "" if cancel_requested else message,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )

    thread = threading.Thread(target=monitor_process, name=f"lsd-app-monitor-{job_id}", daemon=True)
    with _LOCK:
        _RUNNING_PROCESSES[job_id] = process
        _RUNNING_THREADS[job_id] = thread
    thread.start()
    return BackgroundJob(job_id=job_id, state_path=state_path)


def _subprocess_python_env() -> dict[str, str]:
    """Build a clean worker env that prefers the active Python prefix libraries."""

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONNOUSERSITE", "1")
    prefix = Path(sys.prefix).expanduser().resolve()
    library_dirs = [prefix / "lib", prefix / "lib64"]
    existing = [item for item in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    preferred = [str(path) for path in library_dirs if path.exists()]
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*preferred, *[item for item in existing if item not in preferred]])
    env["CONDA_PREFIX"] = str(prefix)
    return env


def start_background_job(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    target: JobTarget,
) -> BackgroundJob:
    """Start a persistent background job unless it is already running or done."""

    state_path = state_path_for_job(registry_root, job_id)
    existing = _read_json(state_path)
    if existing.get("status") in {"running", "pending", "cancelling"} and is_job_running(job_id):
        return BackgroundJob(job_id=job_id, state_path=state_path)
    if existing.get("status") == "completed":
        return BackgroundJob(job_id=job_id, state_path=state_path)

    initial_state = _initial_job_state(
        existing=existing,
        job_id=job_id,
        kind=kind,
        description=description,
        payload=payload,
        execution_mode="thread",
    )
    initial_state["progress_log_path"] = str(_progress_log_path_for_state_path(state_path))
    _write_json_atomic(state_path, initial_state)

    def progress_callback(progress: Mapping[str, Any]) -> None:
        with _LOCK:
            state = _read_json(state_path)
            if not state:
                state = dict(initial_state)
            if state.get("cancel_requested"):
                raise JobCancelled("Cancellation requested by user.")
            now = _utc_now()
            progress_payload = _json_ready(dict(progress))
            old_phase = str(state.get("phase") or state.get("stage") or "")
            new_phase = str(progress_payload.get("phase") or progress_payload.get("stage") or old_phase or "running")
            updated = {
                **state,
                **progress_payload,
                "job_id": job_id,
                "kind": kind,
                "description": description,
                "payload": _json_ready(dict(payload)),
                "status": progress_payload.get("status", state.get("status", "running")),
                "heartbeat_at": now,
                "last_progress_at": now,
                "phase_started_at": state.get("phase_started_at") if new_phase == old_phase else now,
                "last_progress_kind": str(progress_payload.get("progress_kind") or "progress"),
                "progress_log_path": str(_progress_log_path_for_state_path(state_path)),
                "updated_at": now,
            }
            updated = _with_job_duration(updated, now=now)
            _write_json_atomic(state_path, updated)
            _append_job_progress_log_locked(state_path, updated, progress_payload)

    def run_job() -> None:
        progress_callback({"status": "running", "message": "Running."})
        try:
            result = target(progress_callback)
        except JobCancelled as exc:
            state = _read_json(state_path)
            _write_json_atomic(
                state_path,
                _with_job_duration({
                    **state,
                    "status": "cancelled",
                    "cancel_requested": True,
                    "stall_reason": "",
                    "message": str(exc) or "Cancelled by user.",
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                }),
            )
        except Exception as exc:  # pragma: no cover - exercised by UI/runtime failures.
            state = _read_json(state_path)
            _write_json_atomic(
                state_path,
                _with_job_duration({
                    **state,
                    "status": "failed",
                    "stall_reason": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                }),
            )
        else:
            state = _read_json(state_path)
            _write_json_atomic(
                state_path,
                _with_job_duration({
                    **state,
                    "status": "completed",
                    "percent": 1.0,
                    "stall_reason": "",
                    "result": _json_ready(result.to_dict() if hasattr(result, "to_dict") else result),
                    "message": "Completed.",
                    "updated_at": _utc_now(),
                    "finished_at": _utc_now(),
                }),
            )
        finally:
            with _LOCK:
                _RUNNING_THREADS.pop(job_id, None)

    thread = threading.Thread(target=run_job, name=f"lsd-app-{job_id}", daemon=True)
    with _LOCK:
        _RUNNING_THREADS[job_id] = thread
    thread.start()
    return BackgroundJob(job_id=job_id, state_path=state_path)


def start_legacy_thread_background_job(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    target: JobTarget,
) -> BackgroundJob:
    """Compatibility alias for rare jobs that must stay inside Streamlit."""

    return start_background_job(
        registry_root=registry_root,
        job_id=job_id,
        kind=kind,
        description=description,
        payload=payload,
        target=target,
    )


def is_job_running(job_id: str) -> bool:
    with _LOCK:
        thread = _RUNNING_THREADS.get(job_id)
        process = _RUNNING_PROCESSES.get(job_id)
    return bool((thread and thread.is_alive()) or _process_is_running(process))


def _normalize_stale_job_state(
    state: dict[str, Any],
    *,
    state_path: Path | None = None,
    probe_process: bool = False,
) -> dict[str, Any]:
    job_id = str(state.get("job_id") or "")
    status = str(state.get("status") or "")
    if not job_id:
        return state
    process_pid = _int_pid(state.get("process_pid") or state.get("worker_pid"))
    process_active: bool | None = None
    pid_state = ""
    if probe_process:
        process_active = _job_process_is_active(state, process_pid)
        pid_state = _pid_state(process_pid) if process_pid and process_active else ""
        if process_pid and process_active and pid_state == "D":
            updated = _stalled_uninterruptible_state(state, process_pid=process_pid)
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
    if status in TERMINAL_STATUSES:
        return state
    if not probe_process and status in {"stalled", "stalled_uninterruptible"}:
        return state
    if is_job_running(job_id) and state.get("execution_mode") != "subprocess":
        return state
    if state.get("execution_mode") == "thread" and _state_age_seconds(state) < STALE_THREAD_GRACE_SECONDS:
        return state

    if probe_process and process_active:
        if state.get("cancel_requested") and pid_state == "D":
            updated = _stalled_uninterruptible_state(state, process_pid=process_pid)
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
        heartbeat_age = _heartbeat_age_seconds(state)
        if status in ACTIVE_STATUSES and heartbeat_age is not None and heartbeat_age > _stall_timeout_seconds(state):
            updated = {
                **state,
                "status": "stalled",
                "stall_reason": f"No worker heartbeat for {heartbeat_age / 60:.1f} min.",
                "message": "The worker process is still alive but has stopped sending heartbeats. You can try stopping it, then resume from the latest checkpoint/cache.",
                "updated_at": _utc_now(),
            }
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
        progress_age = _progress_age_seconds(state)
        if status in ACTIVE_STATUSES and progress_age is not None and progress_age > _progress_stall_timeout_seconds(state):
            phase = str(state.get("phase") or state.get("stage") or "running")
            updated = {
                **state,
                "status": "stalled",
                "stall_reason": (
                    f"No training/inference progress for {progress_age / 60:.1f} min "
                    f"while the worker heartbeat is still alive (phase={phase})."
                ),
                "message": (
                    "The worker process is alive, but business progress has stopped. "
                    "Stop this job and resume from the latest checkpoint/cache; if the PID enters D state, restart WSL/CUDA."
                ),
                "updated_at": _utc_now(),
            }
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
        return state

    if not probe_process:
        heartbeat_age = _heartbeat_age_seconds(state)
        if status in ACTIVE_STATUSES and heartbeat_age is not None and heartbeat_age > _stall_timeout_seconds(state):
            phase = str(state.get("phase") or state.get("stage") or "running")
            updated = {
                **state,
                "status": "stalled",
                "stall_reason": f"No worker heartbeat for {heartbeat_age / 60:.1f} min (phase={phase}).",
                "message": (
                    "The worker stopped sending heartbeats. The UI did not probe PID/CUDA to avoid blocking; "
                    "try stopping this job, then resume from the latest checkpoint/cache."
                ),
                "updated_at": _utc_now(),
            }
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
        progress_age = _progress_age_seconds(state)
        if status in ACTIVE_STATUSES and progress_age is not None and progress_age > _progress_stall_timeout_seconds(state):
            phase = str(state.get("phase") or state.get("stage") or "running")
            updated = {
                **state,
                "status": "stalled",
                "stall_reason": f"No training/inference progress for {progress_age / 60:.1f} min (phase={phase}).",
                "message": (
                    "Business progress has stopped. The UI did not probe PID/CUDA to avoid blocking; "
                    "stop this job and resume from the latest checkpoint/cache."
                ),
                "updated_at": _utc_now(),
            }
            if state_path is not None:
                _write_json_atomic(state_path, updated)
            return updated
        if status in ACTIVE_STATUSES:
            if heartbeat_age is not None or process_pid:
                return state

    if status in {"stalled", "stalled_uninterruptible"}:
        state = {**state, "stall_reason": ""}
    cancelled = bool(state.get("cancel_requested"))
    now = _utc_now()
    stopped_at = state.get("finished_at") or state.get("updated_at") or now
    updated = {
        **state,
        "status": "cancelled" if cancelled else "interrupted",
        "cancel_requested": cancelled,
        "message": (
            "Cancellation completed after the worker stopped."
            if cancelled
            else "The app restarted before this job finished. Start it again to continue from existing outputs."
        ),
        "updated_at": state.get("updated_at") or now,
        "finished_at": stopped_at,
        "normalized_at": now,
    }
    if status in {"stalled", "stalled_uninterruptible"} and not cancelled:
        updated["message"] = "The previously stalled worker is no longer alive. Resume from the latest checkpoint/cache if needed."
    if state_path is not None:
        release_job_device_locks(state_path.parents[2], job_id)
        _write_json_atomic(state_path, updated)
    return updated


def _state_age_seconds(state: Mapping[str, Any]) -> float:
    return _timestamp_age_seconds(str(state.get("updated_at") or state.get("created_at") or "")) or float("inf")


def update_job_state(registry_root: str | Path, job_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge updates into one persisted job state."""

    state_path = state_path_for_job(registry_root, job_id)
    with _LOCK:
        state = _read_json(state_path)
        if not state:
            state = {"job_id": job_id}
        now = _utc_now()
        updated = {
            **state,
            **_json_ready(dict(updates)),
            "job_id": job_id,
            "progress_log_path": state.get("progress_log_path") or str(_progress_log_path_for_state_path(state_path)),
            "updated_at": now,
        }
        updated = _with_job_duration(updated, now=now)
        _write_json_atomic(state_path, updated)
    return updated


def request_job_cancel(registry_root: str | Path, job_id: str) -> bool:
    """Ask a running background job to stop, then escalate if the worker ignores it."""

    state_path = state_path_for_job(registry_root, job_id)
    with _LOCK:
        state = _read_json(state_path)
        if not state:
            return False
        if state.get("status") in TERMINAL_STATUSES:
            return False
        process_pid = _int_pid(state.get("process_pid") or state.get("worker_pid"))
        _write_json_atomic(
            state_path,
            {
                **state,
                "status": "cancelling",
                "cancel_requested": True,
                "message": (
                    "Cancellation requested. Terminating the isolated worker process."
                    if process_pid
                    else "Cancellation requested. Waiting for the current step to stop safely."
                ),
                "updated_at": _utc_now(),
            },
        )
    if not process_pid:
        return True

    outcome = _terminate_process(process_pid)
    if outcome.get("status") == "terminated":
        release_job_device_locks(registry_root, job_id)
        _write_json_atomic(
            state_path,
            {
                **_read_json(state_path),
                "status": "cancelled",
                "cancel_requested": True,
                "stall_reason": "",
                "message": "Cancelled by terminating the worker process.",
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            },
        )
    elif outcome.get("status") == "stalled_uninterruptible":
        _write_json_atomic(state_path, _stalled_uninterruptible_state(_read_json(state_path), process_pid=process_pid))
    elif outcome.get("status") == "stalled":
        _write_json_atomic(
            state_path,
            {
                **_read_json(state_path),
                "status": "stalled",
                "stall_reason": outcome.get("message") or "Worker ignored termination signals.",
                "message": outcome.get("message") or "The worker ignored termination signals. It may need manual cleanup.",
                "updated_at": _utc_now(),
            },
        )
    return True


def _initial_job_state(
    *,
    existing: Mapping[str, Any],
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "kind": kind,
        "description": description,
        "payload": _json_ready(dict(payload)),
        "status": "pending",
        "percent": 0.0,
        "processed": 0,
        "total": 0,
        "written": 0,
        "skipped": 0,
        "rewritten": 0,
        "cancel_requested": False,
        "message": "Queued.",
        "execution_mode": execution_mode,
        "process_pid": None,
        "worker_pid": None,
        "process_group_id": None,
        "device_lock_id": "",
        "heartbeat_at": "",
        "last_progress_at": "",
        "phase_started_at": _utc_now(),
        "last_progress_kind": "queued",
        "stall_reason": "",
        "progress_log_path": "",
        "created_at": existing.get("created_at") or _utc_now(),
        "started_at": existing.get("started_at") or _utc_now(),
        "duration_seconds": existing.get("duration_seconds") or 0.0,
        "updated_at": _utc_now(),
    }


def _run_process_job(
    state_path_text: str,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    target: JobTarget,
) -> None:
    """Legacy in-process worker kept only for reading old tracebacks/tests."""

    state_path = Path(state_path_text)
    worker_pid = os.getpid()

    def progress_callback(progress: Mapping[str, Any]) -> None:
        with _LOCK:
            state = _read_json(state_path)
            if state.get("cancel_requested"):
                raise JobCancelled("Cancellation requested by user.")
            updated = {
                **state,
                **_json_ready(dict(progress)),
                "job_id": job_id,
                "kind": kind,
                "description": description,
                "payload": _json_ready(dict(payload)),
                "status": progress.get("status", state.get("status", "running")),
                "execution_mode": "process",
                "process_pid": worker_pid,
                "updated_at": _utc_now(),
            }
            _write_json_atomic(state_path, updated)

    try:
        progress_callback({"status": "running", "message": "Running in isolated process.", "process_pid": worker_pid})
        result = target(progress_callback)
    except JobCancelled as exc:
        state = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **state,
                "status": "cancelled",
                "cancel_requested": True,
                "stall_reason": "",
                "message": str(exc) or "Cancelled by user.",
                "process_pid": worker_pid,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )
    except BaseException as exc:  # pragma: no cover - exercised by UI/runtime failures.
        state = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **state,
                "status": "failed",
                "stall_reason": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
                "process_pid": worker_pid,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )
    else:
        state = _read_json(state_path)
        _write_json_atomic(
            state_path,
            _with_job_duration({
                **state,
                "status": "completed",
                "percent": 1.0,
                "stall_reason": "",
                "result": _json_ready(result.to_dict() if hasattr(result, "to_dict") else result),
                "message": "Completed.",
                "process_pid": worker_pid,
                "updated_at": _utc_now(),
                "finished_at": _utc_now(),
            }),
        )


def record_job_progress(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    progress: Mapping[str, Any],
    worker_pid: int | None = None,
    process_group_id: int | None = None,
) -> dict[str, Any]:
    """Persist a business-progress update and refresh health metadata."""

    state_path = state_path_for_job(registry_root, job_id)
    with _LOCK:
        current = _read_json(state_path)
        if current.get("cancel_requested"):
            raise JobCancelled("Cancellation requested by user.")
        progress_payload = _json_ready(dict(progress))
        lock_fields = _ensure_device_lock_locked(
            registry_root,
            job_id=job_id,
            kind=kind,
            payload=payload,
            worker_pid=worker_pid,
            process_group_id=process_group_id,
            device=str(progress_payload.get("device") or ""),
        )
        now = _utc_now()
        old_phase = str(current.get("phase") or current.get("stage") or "")
        new_phase = str(progress_payload.get("phase") or progress_payload.get("stage") or old_phase or "running")
        identity_fields = _worker_identity_fields(worker_pid)
        updated = {
            **current,
            **progress_payload,
            **lock_fields,
            "job_id": job_id,
            "kind": kind,
            "description": description,
            "payload": _json_ready(dict(payload)),
            **identity_fields,
            "status": progress_payload.get("status", current.get("status", "running")),
            "stall_reason": "" if str(progress_payload.get("status", current.get("status", "running"))) in {"running", "pending"} else current.get("stall_reason", ""),
            "worker_pid": worker_pid or current.get("worker_pid") or current.get("process_pid"),
            "process_pid": worker_pid or current.get("process_pid"),
            "process_group_id": process_group_id or current.get("process_group_id"),
            "heartbeat_at": now,
            "last_progress_at": now,
            "phase_started_at": current.get("phase_started_at") if new_phase == old_phase else now,
            "last_progress_kind": str(progress_payload.get("progress_kind") or "progress"),
            "progress_log_path": str(_progress_log_path_for_state_path(state_path)),
            "updated_at": now,
        }
        updated = _with_job_duration(updated, now=now)
        _write_json_atomic(state_path, updated)
        _append_job_progress_log_locked(state_path, updated, progress_payload)
        return updated


def record_job_heartbeat(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    worker_pid: int | None = None,
    process_group_id: int | None = None,
) -> dict[str, Any]:
    """Refresh worker heartbeat without changing business progress."""

    state_path = state_path_for_job(registry_root, job_id)
    with _LOCK:
        current = _read_json(state_path)
        if not current or current.get("status") not in ACTIVE_STATUSES:
            return current
        now = _utc_now()
        lock_fields = _refresh_owned_device_lock_locked(
            registry_root,
            job_id=job_id,
            worker_pid=worker_pid,
            process_group_id=process_group_id,
        )
        identity_fields = _worker_identity_fields(worker_pid)
        updated = {
            **current,
            **lock_fields,
            "job_id": job_id,
            "kind": kind,
            "description": description,
            "payload": _json_ready(dict(payload)),
            **identity_fields,
            "worker_pid": worker_pid or current.get("worker_pid") or current.get("process_pid"),
            "process_pid": worker_pid or current.get("process_pid"),
            "process_group_id": process_group_id or current.get("process_group_id"),
            "heartbeat_at": now,
            "progress_log_path": current.get("progress_log_path") or str(_progress_log_path_for_state_path(state_path)),
            "updated_at": now,
        }
        updated = _with_job_duration(updated, now=now)
        _write_json_atomic(state_path, updated)
        return updated


def start_job_heartbeat(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    description: str,
    payload: Mapping[str, Any],
    worker_pid: int | None,
    process_group_id: int | None,
    interval_seconds: float = JOB_HEARTBEAT_INTERVAL_SECONDS,
) -> threading.Event:
    """Start a daemon heartbeat loop for one worker process."""

    stop = threading.Event()
    record_job_heartbeat(
        registry_root=registry_root,
        job_id=job_id,
        kind=kind,
        description=description,
        payload=payload,
        worker_pid=worker_pid,
        process_group_id=process_group_id,
    )

    def beat() -> None:
        while not stop.wait(interval_seconds):
            try:
                record_job_heartbeat(
                    registry_root=registry_root,
                    job_id=job_id,
                    kind=kind,
                    description=description,
                    payload=payload,
                    worker_pid=worker_pid,
                    process_group_id=process_group_id,
                )
            except Exception:
                break

    thread = threading.Thread(target=beat, name=f"lsd-app-heartbeat-{job_id}", daemon=True)
    thread.start()
    return stop


def reserve_job_device_lock(
    *,
    registry_root: str | Path,
    job_id: str,
    kind: str,
    payload: Mapping[str, Any],
    worker_pid: int | None,
    process_group_id: int | None,
    device: str | None = None,
) -> dict[str, Any]:
    with _LOCK:
        return _ensure_device_lock_locked(
            registry_root,
            job_id=job_id,
            kind=kind,
            payload=payload,
            worker_pid=worker_pid,
            process_group_id=process_group_id,
            device=device or "",
        )


def release_job_device_locks(registry_root: str | Path, job_id: str) -> None:
    root = device_lock_root(registry_root)
    for path in root.glob("*.json"):
        lock = _read_json(path)
        if str(lock.get("job_id") or "") == str(job_id):
            try:
                path.unlink()
            except OSError:
                pass


def clear_stale_job_blocker(registry_root: str | Path, job_id: str) -> dict[str, Any]:
    """Release a stale UI/device blocker after the operator has restarted WSL/CUDA.

    This intentionally does not try to kill a PID.  It only marks the persisted
    job record as interrupted and removes LSDetector-owned device locks so the UI
    can launch a fresh resume run.
    """

    state_path = state_path_for_job(registry_root, job_id)
    with _LOCK:
        state = _read_json(state_path)
        if not state:
            return {}
        status = str(state.get("status") or "")
        if status not in {"stalled", "stalled_uninterruptible", "resource_blocked", "cancelling"}:
            return state
        now = _utc_now()
        release_job_device_locks(registry_root, job_id)
        updated = {
            **state,
            "status": "interrupted",
            "cancel_requested": bool(state.get("cancel_requested")),
            "stall_reason": "",
            "message": (
                "Cleared stale LSDetector blocker after WSL/CUDA restart. "
                "Resume from the latest checkpoint/cache if needed."
            ),
            "cleared_at": now,
            "finished_at": state.get("finished_at") or now,
            "updated_at": now,
        }
        _write_json_atomic(state_path, updated)
        return updated


def _startup_blocking_device_lock(
    registry_root: str | Path,
    *,
    job_id: str,
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    lock_id = _requested_device_lock_id(kind=kind, payload=payload, device="")
    if not lock_id:
        return None
    with _LOCK:
        blocking = _find_blocking_device_lock_locked(registry_root, current_job_id=job_id, requested_lock_id=lock_id)
        if blocking is not None:
            return blocking
        return _find_blocking_historical_job_locked(registry_root, current_job_id=job_id, lock_id=lock_id)


def _resource_blocked_state(state: Mapping[str, Any], blocking_lock: Mapping[str, Any]) -> dict[str, Any]:
    owner = str(blocking_lock.get("job_id") or "unknown")
    pid = blocking_lock.get("worker_pid") or blocking_lock.get("process_pid") or "?"
    device = str(blocking_lock.get("device") or blocking_lock.get("device_lock_id") or "cuda")
    pid_int = _int_pid(pid)
    pid_state = _pid_state(pid_int) if pid_int else ""
    if pid_state == "D":
        message = (
            f"CUDA device {device} is blocked by job {owner} (PID {pid}) stuck in uninterruptible kernel sleep. "
            "Linux cannot kill this state from the app; restart WSL/CUDA before launching another CUDA job."
        )
        status = "stalled_uninterruptible"
    else:
        message = f"CUDA device {device} is already owned by job {owner} (PID {pid})."
        status = "resource_blocked"
    return {
        **dict(state),
        "status": status,
        "stall_reason": message,
        "message": message,
        "device": device,
        "device_lock_id": str(blocking_lock.get("device_lock_id") or ""),
        "device_lock_owner_job_id": owner,
        "device_lock_owner_pid": pid,
        "blocking_device_lock": _json_ready(dict(blocking_lock)),
        "updated_at": _utc_now(),
        "finished_at": _utc_now(),
    }


def _stalled_uninterruptible_state(state: Mapping[str, Any], *, process_pid: int) -> dict[str, Any]:
    message = (
        f"Worker PID {process_pid} is stuck in uninterruptible kernel sleep. "
        "Linux cannot kill this state from the app; restart WSL/CUDA before launching another CUDA job."
    )
    return {
        **dict(state),
        "status": "stalled_uninterruptible",
        "stall_reason": message,
        "message": message,
        "process_pid": process_pid,
        "worker_pid": state.get("worker_pid") or process_pid,
        "updated_at": _utc_now(),
    }


def _ensure_device_lock_locked(
    registry_root: str | Path,
    *,
    job_id: str,
    kind: str,
    payload: Mapping[str, Any],
    worker_pid: int | None,
    process_group_id: int | None,
    device: str | None,
) -> dict[str, Any]:
    lock_id = _requested_device_lock_id(kind=kind, payload=payload, device=device or "")
    if not lock_id:
        return {}
    blocking = _find_blocking_device_lock_locked(registry_root, current_job_id=job_id, requested_lock_id=lock_id)
    if blocking is not None:
        blocked_state = _resource_blocked_state({"job_id": job_id, "kind": kind, "payload": payload}, blocking)
        raise JobResourceBlocked(str(blocked_state["message"]), details=blocked_state)
    lock = _job_device_lock_payload(
        job_id=job_id,
        kind=kind,
        lock_id=lock_id,
        device=device or _requested_device_from_payload(payload),
        worker_pid=worker_pid,
        process_group_id=process_group_id,
    )
    _write_json_atomic(_device_lock_path(registry_root, lock_id), lock)
    return {
        "device_lock_id": lock_id,
        "device_lock": lock,
        "device_lock_owner_job_id": job_id,
        "device_lock_owner_pid": worker_pid,
    }


def _refresh_owned_device_lock_locked(
    registry_root: str | Path,
    *,
    job_id: str,
    worker_pid: int | None,
    process_group_id: int | None,
) -> dict[str, Any]:
    refreshed: dict[str, Any] = {}
    for path in device_lock_root(registry_root).glob("*.json"):
        lock = _read_json(path)
        if str(lock.get("job_id") or "") != str(job_id):
            continue
        now = _utc_now()
        identity_fields = _worker_identity_fields(worker_pid)
        lock = {
            **lock,
            "worker_pid": worker_pid or lock.get("worker_pid"),
            "process_pid": worker_pid or lock.get("process_pid"),
            "process_group_id": process_group_id or lock.get("process_group_id"),
            **identity_fields,
            "heartbeat_at": now,
            "updated_at": now,
        }
        _write_json_atomic(path, lock)
        refreshed = {
            "device_lock_id": str(lock.get("device_lock_id") or path.stem),
            "device_lock": lock,
            "device_lock_owner_job_id": job_id,
            "device_lock_owner_pid": lock.get("worker_pid") or lock.get("process_pid"),
        }
    return refreshed


def _job_device_lock_payload(
    *,
    job_id: str,
    kind: str,
    lock_id: str,
    device: str,
    worker_pid: int | None,
    process_group_id: int | None,
) -> dict[str, Any]:
    now = _utc_now()
    identity_fields = _worker_identity_fields(worker_pid)
    return {
        "device_lock_id": lock_id,
        "device": _device_from_lock_id(lock_id, fallback=device),
        "job_id": job_id,
        "kind": kind,
        "worker_pid": worker_pid,
        "process_pid": worker_pid,
        "process_group_id": process_group_id,
        **identity_fields,
        "created_at": now,
        "heartbeat_at": now,
        "updated_at": now,
    }


def _find_blocking_device_lock_locked(
    registry_root: str | Path,
    *,
    current_job_id: str,
    requested_lock_id: str | None = None,
) -> dict[str, Any] | None:
    for path in device_lock_root(registry_root).glob("*.json"):
        if requested_lock_id and path.stem != requested_lock_id:
            continue
        lock = _read_json(path)
        if not lock:
            continue
        if str(lock.get("job_id") or "") == str(current_job_id):
            continue
        owner_pid = _int_pid(lock.get("worker_pid") or lock.get("process_pid"))
        if _job_process_is_active(lock, owner_pid):
            return lock
        try:
            path.unlink()
        except OSError:
            pass
    return None


def _find_blocking_historical_job_locked(
    registry_root: str | Path,
    *,
    current_job_id: str,
    lock_id: str,
) -> dict[str, Any] | None:
    for path in job_root(registry_root).glob("*.json"):
        if path.name.startswith("._"):
            continue
        state = _read_json(path)
        owner_job_id = str(state.get("job_id") or "")
        if not owner_job_id or owner_job_id == current_job_id:
            continue
        if str(state.get("kind") or "") not in LONG_RUNNING_DEVICE_LOCK_KINDS:
            continue
        requested = _requested_device_lock_id(
            kind=str(state.get("kind") or ""),
            payload=state.get("payload") if isinstance(state.get("payload"), Mapping) else {},
            device=str(state.get("device") or ""),
        )
        if requested != lock_id:
            continue
        owner_pid = _int_pid(state.get("worker_pid") or state.get("process_pid"))
        if _job_process_is_active(state, owner_pid):
            pid_state = _pid_state(owner_pid) if owner_pid and _pid_is_alive(owner_pid) else ""
            return {
                "device_lock_id": lock_id,
                "device": _device_from_lock_id(lock_id),
                "job_id": owner_job_id,
                "kind": state.get("kind"),
                "worker_pid": owner_pid,
                "process_pid": owner_pid,
                "status": "stalled_uninterruptible" if pid_state == "D" else state.get("status"),
                "pid_state": pid_state,
                "source": "job_state",
            }
    return None


def _device_lock_path(registry_root: str | Path, lock_id: str) -> Path:
    return device_lock_root(registry_root) / f"{_safe_token(lock_id)}.json"



def _progress_log_path_for_state_path(state_path: Path) -> Path:
    return state_path.with_suffix(PERSISTED_PROGRESS_LOG_SUFFIX)


def _append_job_progress_log_locked(state_path: Path, state: Mapping[str, Any], progress: Mapping[str, Any]) -> None:
    if not _should_persist_progress_log(progress):
        return
    row = _compact_progress_log_row(state, progress)
    path = _progress_log_path_for_state_path(state_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
    except OSError:
        return


def _should_persist_progress_log(progress: Mapping[str, Any]) -> bool:
    status = str(progress.get("status") or "")
    progress_kind = str(progress.get("progress_kind") or "progress")
    phase = str(progress.get("phase") or progress.get("stage") or "")
    if progress_kind == "batch_subphase":
        return False
    if status == "completed" or status in TERMINAL_STATUSES:
        return True
    if any(key in progress for key in ("losses", "train", "test", "eval", "evaluation")):
        return True
    if any(key in progress for key in ("inference_duration_seconds", "patches_per_second", "inference_patches_per_second")):
        return True
    if progress_kind in PERSISTED_PROGRESS_LOG_KINDS:
        return True
    return phase in {"checkpoint", "checkpoint_saving", "checkpoint_saved", "evaluation", "merge", "register", "write_metrics", "write_outputs"}


def _compact_progress_log_row(state: Mapping[str, Any], progress: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "phase", "stage", "progress_kind", "percent", "overall_percent",
        "processed", "total", "written", "skipped", "rewritten", "completed",
        "current_epoch", "total_epochs", "current_batch", "total_batches",
        "eval_current_batch", "eval_total_batches", "evaluation_split",
        "lr", "lr_seg", "lr_dis", "device", "device_name", "current_action", "current_reason",
        "current_item_id", "current_raster", "item_status", "losses", "train", "test", "eval", "evaluation",
        "best_epoch", "best_score", "selection_metric", "latest_checkpoint_path", "best_checkpoint_path",
        "merged_checkpoint_path", "registered_weight_id", "registered_weight_path", "metrics_path",
        "training_config_path", "duration_seconds", "inference_duration_seconds", "patches_per_second",
        "predictions_per_second", "inference_patches_per_second", "inference_predictions_per_second",
        "inference_patch_count", "inference_prediction_count", "latest_log", "message",
    )
    row: dict[str, Any] = {
        "timestamp": state.get("updated_at") or _utc_now(),
        "job_id": state.get("job_id"),
        "kind": state.get("kind"),
    }
    for key in keys:
        if key in progress:
            row[key] = progress.get(key)
        elif key in state and key in {"duration_seconds"}:
            row[key] = state.get(key)
    return row


def _with_job_duration(state: Mapping[str, Any], *, now: str | None = None) -> dict[str, Any]:
    updated = dict(state)
    duration = _job_duration_seconds(updated, now=now)
    if duration is not None:
        updated["duration_seconds"] = duration
    return updated


def _job_duration_seconds(state: Mapping[str, Any], *, now: str | None = None) -> float | None:
    started_raw = str(state.get("started_at") or state.get("created_at") or "")
    if not started_raw:
        return None
    finished_raw = str(state.get("finished_at") or "") or now or _utc_now()
    try:
        started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (finished - started).total_seconds())

def _requested_device_lock_id(*, kind: str, payload: Mapping[str, Any], device: str) -> str:
    if kind not in LONG_RUNNING_DEVICE_LOCK_KINDS:
        return ""
    requested = str(device or _requested_device_from_payload(payload) or "").strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda"
    if requested == "cpu":
        return ""
    if requested.startswith("cuda"):
        suffix = requested.split(":", 1)[1] if ":" in requested else "0"
        return f"cuda-{suffix or '0'}"
    return ""


def _requested_device_from_payload(payload: Mapping[str, Any]) -> str:
    device = str(payload.get("device") or "").strip()
    if device:
        return device
    raw = payload.get("raw_config") if isinstance(payload.get("raw_config"), Mapping) else {}
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), Mapping) else {}
    return str(runtime.get("device") or raw.get("device") or "").strip()


def _device_from_lock_id(lock_id: str, *, fallback: str = "") -> str:
    if lock_id.startswith("cuda-"):
        return f"cuda:{lock_id.split('-', 1)[1]}"
    return fallback or lock_id


def _pid_holds_gpu(pid: int) -> bool:
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "")
    if not smi:
        return False
    try:
        result = subprocess.run(
            [smi, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return False
    return str(pid) in {line.strip().split(",", 1)[0] for line in result.stdout.splitlines() if line.strip()}


def _heartbeat_age_seconds(state: Mapping[str, Any]) -> float | None:
    raw = str(state.get("heartbeat_at") or "").strip()
    if not raw:
        return _timestamp_age_seconds(str(state.get("updated_at") or ""))
    return _timestamp_age_seconds(raw)


def _progress_age_seconds(state: Mapping[str, Any]) -> float | None:
    raw = str(state.get("last_progress_at") or "").strip()
    if not raw:
        return None
    return _timestamp_age_seconds(raw)


def _stall_timeout_seconds(state: Mapping[str, Any]) -> float:
    phase_text = " ".join(
        str(state.get(key) or "").lower()
        for key in ("phase", "stage", "current_action", "last_progress_kind")
    )
    if any(token in phase_text for token in LONG_STALL_PHASE_TOKENS):
        return JOB_STALL_LONG_PHASE_SECONDS
    return JOB_STALL_SECONDS


def _progress_stall_timeout_seconds(state: Mapping[str, Any]) -> float:
    phase_text = " ".join(
        str(state.get(key) or "").lower()
        for key in ("phase", "stage", "current_action", "last_progress_kind", "batch_subphase")
    )
    if any(token in phase_text for token in LONG_STALL_PHASE_TOKENS):
        return JOB_PROGRESS_STALL_LONG_PHASE_SECONDS
    return JOB_PROGRESS_STALL_SECONDS


def _timestamp_age_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _int_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_state(pid: int) -> str:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text(errors="ignore").splitlines():
            if line.startswith("State:"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else ""
    except OSError:
        return ""
    return ""


def _job_process_is_active(owner: Mapping[str, Any], pid: int | None) -> bool:
    if not pid:
        return False
    alive = _pid_is_alive(pid)
    holds_gpu = _pid_holds_gpu(pid)
    if not alive and not holds_gpu:
        return False
    return _pid_matches_recorded_owner(pid, owner)


def _pid_matches_recorded_owner(pid: int, owner: Mapping[str, Any]) -> bool:
    recorded_boot_id = str(owner.get("worker_boot_id") or owner.get("process_boot_id") or "").strip()
    current_boot_id = _current_boot_id()
    if recorded_boot_id and current_boot_id and recorded_boot_id != current_boot_id:
        return False

    recorded_start = str(owner.get("worker_start_ticks") or owner.get("process_start_ticks") or "").strip()
    current_start = _pid_start_ticks(pid)
    if recorded_start and current_start and recorded_start != current_start:
        return False

    recorded_cmdline = str(owner.get("worker_cmdline") or owner.get("process_cmdline") or "").strip()
    current_cmdline = _pid_cmdline(pid)
    job_id = str(owner.get("job_id") or "").strip()
    if recorded_cmdline and current_cmdline and recorded_cmdline != current_cmdline:
        if job_id and job_id in current_cmdline and "lsd_app.web.job_worker" in current_cmdline:
            return True
        return False
    return True


def _worker_identity_fields(pid: int | None) -> dict[str, Any]:
    if not pid:
        return {}
    fields: dict[str, Any] = {}
    boot_id = _current_boot_id()
    start_ticks = _pid_start_ticks(pid)
    cmdline = _pid_cmdline(pid)
    cwd = _pid_cwd(pid)
    if boot_id:
        fields["worker_boot_id"] = boot_id
    if start_ticks:
        fields["worker_start_ticks"] = start_ticks
    if cmdline:
        fields["worker_cmdline"] = cmdline
    if cwd:
        fields["worker_cwd"] = cwd
    return fields


def _current_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _pid_start_ticks(pid: int) -> str:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    close = text.rfind(")")
    if close < 0:
        return ""
    fields = text[close + 1 :].strip().split()
    return fields[19] if len(fields) > 19 else ""


def _pid_cmdline(pid: int) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part)


def _pid_cwd(pid: int) -> str:
    try:
        return str((Path("/proc") / str(pid) / "cwd").resolve())
    except OSError:
        return ""


def _terminate_process(pid: int) -> dict[str, Any]:
    _send_signal_to_process_or_group(pid, signal.SIGTERM)
    for _ in range(30):
        time.sleep(0.1)
        if not _pid_is_alive(pid):
            return {"status": "terminated"}
    if _pid_state(pid) == "D":
        return {
            "status": "stalled_uninterruptible",
            "message": "Worker is stuck in uninterruptible kernel sleep after SIGTERM; WSL/CUDA restart is required.",
        }
    _send_signal_to_process_or_group(pid, signal.SIGKILL)
    for _ in range(20):
        time.sleep(0.1)
        if not _pid_is_alive(pid):
            return {"status": "terminated"}
    if _pid_state(pid) == "D":
        return {
            "status": "stalled_uninterruptible",
            "message": "Worker is stuck in uninterruptible kernel sleep after SIGKILL; WSL/CUDA restart is required.",
        }
    return {"status": "stalled", "message": "Worker ignored SIGTERM and SIGKILL."}


def _send_signal_to_process_or_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, OSError):
        return


def _process_is_running(process: subprocess.Popen[bytes] | None) -> bool:
    return bool(process is not None and process.poll() is None)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(tmp_path, path)
    except FileNotFoundError:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _json_ready(value: Any) -> Any:
    import builtins as _builtins
    from collections.abc import Mapping as _RuntimeMapping

    if _builtins.isinstance(value, Path):
        return _builtins.str(value)
    if _np is not None:
        try:
            if _builtins.isinstance(value, _np.generic):
                return value.item()
        except Exception:
            pass
    if _builtins.isinstance(value, _RuntimeMapping):
        return {_builtins.str(key): _json_ready(item) for key, item in value.items()}
    if _builtins.isinstance(value, (_builtins.list, _builtins.tuple)):
        return [_json_ready(item) for item in value]
    try:
        json.dumps(value)
    except Exception:
        return _builtins.str(value)
    return value


def _safe_token(value: str) -> str:
    import re

    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-").lower()
    return text or "job"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
