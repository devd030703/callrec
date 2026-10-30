"""Explicitly queued jobs; no login service and no implicit restart after a crash."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from .common import CallrecError, atomic_json, lock, now, read_json, update_session

PENDING = ("queued", "transcribing", "auditing")
TERMINAL = ("complete", "complete_with_warnings", "transcription_failed", "interrupted")


def owned(owner):
    pid = owner.get("pid")
    token = owner.get("token")
    if not isinstance(pid, int) or pid <= 1 or not token:
        return False
    p = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "stat=", "-o", "command="], capture_output=True, text=True
    )
    return (
        p.returncode == 0
        and not p.stdout.lstrip().startswith("Z")
        and ("-m callrec.jobs" in p.stdout and f"--token {token}" in p.stdout)
    )


def age(value):
    if not value:
        return None
    return (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(value)).total_seconds()


def event(session, kind, **fields):
    with lock(session / ".events.lock", blocking=True):
        with (session / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": now(), "event": kind, **fields}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def sessions(root):
    return sorted((root / "sessions").glob("*/session.json"))


def enqueue(root, session):
    """Queue under one lock, including supervisor shutdown/startup handoff."""
    with lock(root / ".queue.lock", blocking=True):
        data = read_json(session / "session.json")
        if data["status"] in ("complete", "complete_with_warnings"):
            if (
                not (session / "transcript.json").exists()
                or not (session / "transcript.md").exists()
            ):
                raise CallrecError("Completed output is missing; investigate before overwriting.")
            return "already complete"
        owner = read_json(root / "processor.json", {})
        alive = owned(owner)
        if (
            data["status"] in PENDING
            and alive
            and data.get("processor_token") == owner.get("token")
        ):
            return "already queued or processing"
        if not alive:
            # A new explicit invocation must not resurrect other abandoned jobs.
            for path in sessions(root):
                old = read_json(path)
                if old["status"] in PENDING:
                    update_session(
                        path.parent,
                        status="interrupted",
                        transcription_error="Processing supervisor exited. Explicit retry required.",
                    )
                    event(path.parent, "supervisor_lost", retry=f"callrec retry {path.parent.name}")
            token = uuid.uuid4().hex
            with (root / "processor.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "callrec.jobs", "--root", str(root), "--token", token],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            owner = {"pid": process.pid, "token": token, "started_at": now(), "heartbeat_at": now()}
            atomic_json(root / "processor.json", owner)
            threading.Thread(target=process.wait, daemon=True).start()
        update_session(
            session,
            status="queued",
            queued_at=now(),
            processor_token=owner["token"],
            processor_pid=owner["pid"],
            transcription_error=None,
            current_window=None,
        )
        event(session, "queued", processor_token=owner["token"])
    return "queued"


def snapshot(root, session):
    from .common import owned_worker

    data = read_json(session / "session.json")
    state = data["status"]
    owner = read_json(root / "processor.json", {})
    alive = data.get("processor_token") == owner.get("token") and owned(owner)
    heartbeat_age = age(data.get("processing_heartbeat_at"))
    if state in PENDING:
        if not alive:
            health = "interrupted: supervisor absent; explicit retry required"
            effective = "interrupted"
        elif age(owner.get("heartbeat_at")) is not None and age(owner["heartbeat_at"]) > 20:
            health, effective = "stalled: supervisor alive, heartbeat stale", state
        elif state == "queued":
            health, effective = "queued", state
        elif heartbeat_age is None or heartbeat_age > 20:
            health, effective = "stalled: supervisor alive, session heartbeat stale", state
        else:
            window_age = age(data.get("window_started_at"))
            health = "processing"
            if window_age is not None and window_age > 180:
                health = "stalled: inference deadline/recovery pending"
            effective = state
    else:
        effective, health = state, "stopped"
    capture_age = age(data.get("heartbeat_at"))
    capture = "stopped; audio retained"
    if (
        not data.get("audio_seconds", 0)
        and not (session / "source.wav").exists()
        and not list((session / "chunks").glob("*.wav"))
    ):
        capture = "stopped; no audio captured"
        if data.get("empty_capture_closed_at"):
            capture = "failed session closed; no audio captured"
    if state in ("starting", "recording"):
        capture = (
            "receiving audio frames"
            if owned_worker(data)
            and capture_age is not None
            and capture_age < 4
            and data.get("seconds_since_frames", 999) < 3
            else "starting / stalled / interrupted; inspect capture.log"
        )
    quality = read_json(session / "quality.json", {})
    attempt_id = data.get("attempt_id")
    attempt = read_json(session / "attempts" / f"{attempt_id}.json", {}) if attempt_id else {}
    metrics = dict(data.get("metrics") or {})
    if "process_peak_rss_mib" in metrics:
        metrics["supervisor_peak_rss_mib"] = metrics.pop("process_peak_rss_mib")
    if attempt.get("model_process", {}).get("peak_rss_mib") is not None:
        metrics["model_process_peak_rss_mib"] = attempt["model_process"]["peak_rss_mib"]
    if attempt.get("active_processing_seconds") is not None:
        metrics["attempt_active_seconds"] = attempt["active_processing_seconds"]
    return {
        "session_id": session.name,
        "state": effective,
        "stored_state": state,
        "capture_health": capture,
        "transcription_health": health,
        "saved_audio_seconds": data.get("source_duration_seconds", data.get("audio_seconds", 0)),
        "audio_saved": (session / "audio.m4a").is_file(),
        "processed_windows": data.get("processed_windows", 0),
        "expected_windows": data.get("expected_windows"),
        "current_window": data.get("current_window"),
        "heartbeat_age_seconds": heartbeat_age,
        "warning_count": len(data.get("warnings", [])),
        "review_signal_count": len(quality.get("findings", [])),
        "last_error": data.get("transcription_error") or data.get("capture_error"),
        "retry_command": f"callrec retry {session.name}"
        if effective in ("interrupted", "transcription_failed")
        else None,
        "metrics": metrics or None,
        "legacy_metrics_incomplete": bool(data.get("metrics") and not data.get("attempts")),
        "paths": {
            name: str(session / name)
            for name in (
                "audio.m4a",
                "transcript.md",
                "transcript.json",
                "quality.md",
                "events.jsonl",
            )
        },
    }


def run(root, token):
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            with lock(root / ".queue.lock", blocking=True):
                owner = read_json(root / "processor.json", {})
                if owner.get("token") != token:
                    return
                atomic_json(root / "processor.json", {**owner, "heartbeat_at": now()})
            done.wait(5)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        _run_jobs(root, token)
    finally:
        done.set()
        thread.join(6)


def _run_jobs(root, token):
    from .transcription import transcribe_session

    # Only this supervisor can own the model. Audits acquire the same model lock.
    with lock(root / ".model.lock", blocking=True):
        while True:
            with lock(root / ".queue.lock", blocking=True):
                owner = read_json(root / "processor.json", {})
                if owner.get("token") != token:
                    return
                queued = [
                    p.parent
                    for p in sessions(root)
                    if (
                        (d := read_json(p))["status"] == "queued"
                        and d.get("processor_token") == token
                    )
                ]
                queued.sort(key=lambda p: (read_json(p / "session.json")["created_at"], p.name))
                if not queued:
                    (root / "processor.json").unlink(missing_ok=True)
                    return
                session = queued[0]
                update_session(session, status="transcribing", processing_heartbeat_at=now())
            done = threading.Event()

            def heartbeat():
                while not done.is_set():
                    update_session(session, processing_heartbeat_at=now())
                    done.wait(5)

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                with lock(session / ".process.lock"):
                    transcribe_session(session, root)
            except Exception as e:
                update_session(
                    session,
                    status="transcription_failed",
                    transcription_error=f"{type(e).__name__}: {e}",
                )
                event(session, "job_failed", error=str(e))
            finally:
                done.set()
                thread.join(6)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise SystemExit(f"Supervisor stopped by signal {signum}; retry required.")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    run(args.root, args.token)


if __name__ == "__main__":
    main()
