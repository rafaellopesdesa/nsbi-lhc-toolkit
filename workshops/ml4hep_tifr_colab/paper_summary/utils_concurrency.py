"""Cross-runtime job claims for the shared SLCP paper-summary campaign.

The paper notebooks can be opened in several Colab runtimes that all mount the
same Google Drive artifact root. A claim is represented by an atomically
created directory, rather than by a check-then-create file or ``fcntl`` lock:
the latter does not coordinate reliably across distinct Drive mounts.

Claims are leases. A daemon heartbeat keeps an active lease fresh, and a claim
whose heartbeat has stopped for several hours can be reclaimed after an
interrupted Colab session. Scientific completion is still determined only by
the validated result/checkpoint artifact; the lease merely prevents duplicate
work while that artifact is being produced.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCK_SCHEMA = "slcp_paper_summary_job_claim_v1"
DEFAULT_HEARTBEAT_SECONDS = 60.0
DEFAULT_STALE_HOURS = 6.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else float(raw)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _safe_component(value: str, limit: int = 72) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return (clean or "job")[:limit]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class CampaignJobClaim:
    """A non-blocking or bounded-wait lease for one campaign job."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        campaign_signature: str,
        stage: str,
        job_key: str,
        wait: bool = False,
        timeout_seconds: float = 0.0,
        stale_hours: float | None = None,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.campaign_signature = str(campaign_signature)
        self.stage = str(stage)
        self.job_key = str(job_key)
        self.wait = bool(wait)
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.stale_seconds = 3600.0 * (
            _positive_environment_float(
                "PAPER_SUMMARY_LOCK_STALE_HOURS", DEFAULT_STALE_HOURS
            )
            if stale_hours is None
            else float(stale_hours)
        )
        self.heartbeat_seconds = (
            _positive_environment_float(
                "PAPER_SUMMARY_LOCK_HEARTBEAT_SECONDS",
                DEFAULT_HEARTBEAT_SECONDS,
            )
            if heartbeat_seconds is None
            else float(heartbeat_seconds)
        )
        digest = hashlib.sha256(self.job_key.encode("utf-8")).hexdigest()[:12]
        self.lock_directory = (
            self.artifact_root
            / ".paper_summary_locks"
            / _safe_component(self.campaign_signature, limit=48)
            / _safe_component(self.stage, limit=48)
            / f"{_safe_component(self.job_key)}--{digest}.lock"
        )
        self.token = uuid.uuid4().hex
        self.acquired = False
        self.reclaimed_stale = False
        self.busy_owner: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def owner_summary(self) -> str:
        owner = self.busy_owner or {}
        worker = owner.get("worker")
        started = owner.get("started_utc")
        if worker and started:
            return f"{worker}, since {started}"
        return str(worker or "another worker")

    def _owner_path(self) -> Path:
        return self.lock_directory / "owner.json"

    def _heartbeat_path(self) -> Path:
        return self.lock_directory / "heartbeat"

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            return json.loads(self._owner_path().read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def _lease_age_seconds(self) -> float:
        candidates = (
            self._heartbeat_path(),
            self._owner_path(),
            self.lock_directory,
        )
        for path in candidates:
            try:
                return max(0.0, time.time() - path.stat().st_mtime)
            except OSError:
                continue
        return 0.0

    def _reclaim_if_stale(self) -> bool:
        if self._lease_age_seconds() <= self.stale_seconds:
            return False
        quarantine = self.lock_directory.with_name(
            f"{self.lock_directory.name}.stale-{self.token}"
        )
        try:
            os.rename(self.lock_directory, quarantine)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        shutil.rmtree(quarantine, ignore_errors=True)
        self.reclaimed_stale = True
        return True

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                owner = self._read_owner()
                if owner is None or owner.get("token") != self.token:
                    return
                self._heartbeat_path().touch()
            except OSError:
                # A transient Drive error should not kill training. If it
                # persists beyond the generous stale interval, another worker
                # may safely reclaim the abandoned-looking lease.
                continue

    def _acquire_once(self) -> bool:
        self.lock_directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_directory.mkdir()
        except FileExistsError:
            self.busy_owner = self._read_owner()
            if self._reclaim_if_stale():
                return self._acquire_once()
            return False
        owner = {
            "schema": LOCK_SCHEMA,
            "campaign_signature": self.campaign_signature,
            "stage": self.stage,
            "job_key": self.job_key,
            "token": self.token,
            "worker": (
                os.environ.get("PAPER_SUMMARY_WORKER_ID", "").strip()
                or f"{platform.node() or 'runtime'}:{os.getpid()}"
            ),
            "pid": os.getpid(),
            "host": platform.node(),
            "started_utc": _utc_now(),
            "heartbeat_seconds": self.heartbeat_seconds,
            "stale_after_seconds": self.stale_seconds,
        }
        try:
            _atomic_write_json(self._owner_path(), owner)
            self._heartbeat_path().touch()
        except Exception:
            shutil.rmtree(self.lock_directory, ignore_errors=True)
            raise
        if (self._read_owner() or {}).get("token") != self.token:
            shutil.rmtree(self.lock_directory, ignore_errors=True)
            raise RuntimeError(
                f"Could not verify ownership of job claim {self.lock_directory}."
            )
        self.acquired = True
        self.busy_owner = None
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"paper-summary-heartbeat-{self.token[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return True

    def acquire(self) -> "CampaignJobClaim":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if self._acquire_once():
                return self
            if not self.wait or time.monotonic() >= deadline:
                return self
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))

    def release(self) -> None:
        if not self.acquired:
            return
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.heartbeat_seconds))
        owner = self._read_owner()
        if owner is not None and owner.get("token") == self.token:
            quarantine = self.lock_directory.with_name(
                f"{self.lock_directory.name}.released-{self.token}"
            )
            try:
                os.rename(self.lock_directory, quarantine)
            except (FileNotFoundError, FileExistsError, OSError):
                pass
            else:
                shutil.rmtree(quarantine, ignore_errors=True)
        self.acquired = False

    def __enter__(self) -> "CampaignJobClaim":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def campaign_job_claim(
    artifact_root: str | Path,
    *,
    campaign_signature: str,
    stage: str,
    job_key: str,
    wait: bool = False,
    timeout_seconds: float = 0.0,
) -> CampaignJobClaim:
    """Return a context-managed claim for one independently publishable job."""

    return CampaignJobClaim(
        artifact_root,
        campaign_signature=campaign_signature,
        stage=stage,
        job_key=job_key,
        wait=wait,
        timeout_seconds=timeout_seconds,
    )


__all__ = ["CampaignJobClaim", "campaign_job_claim"]

