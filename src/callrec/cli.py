from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .common import (
    PROFILES,
    CallrecError,
    active_session,
    atomic_json,
    check_space,
    clear_active,
    create_session,
    duration,
    init_root,
    lock,
    model_path,
    now,
    owned_worker,
    read_json,
    resolve_session,
    root_path,
    sha256,
    tool,
    update_session,
)
from .jobs import enqueue, event, snapshot
from .setup import doctor, hark_path, setup
from .transcription import audit_session, playback_audio, prepare_audio


def load_context(path, engine):
    if not path:
        return None
    if engine not in ("qwen", "qwen-large"):
        raise CallrecError("Vocabulary context is currently supported only with the Qwen engine.")
    data = read_json(Path(path).expanduser())
    if not isinstance(data, dict) or set(data) != {"hotwords"}:
        raise CallrecError('Context must be a JSON object containing only a "hotwords" list.')
    words = data["hotwords"]
    if not isinstance(words, list) or not 1 <= len(words) <= 40:
        raise CallrecError("Context must contain between 1 and 40 short vocabulary phrases.")
    if any(
        not isinstance(w, str)
        or not w.strip()
        or len(w) > 80
        or len(w.split()) > 4
        or any(c in w for c in "\n\r.!?।")
        for w in words
    ):
        raise CallrecError("Context entries must be short phrases, not sentences.")
    return {"hotwords": [w.strip() for w in words]}


def start(root, profile, capture_only=False, engine="auto", context=None):
    engine = ("qwen" if profile == "hindi" else "whisper") if engine == "auto" else engine
    context = load_context(context, engine)
    init_root(root)
    hark_path(root)
    tool("ffmpeg")
    tool("ffprobe")
    if not capture_only:
        model_path(root, engine)
    check_space(root)
    with lock(root / ".control.lock"):
        existing = active_session(root)
        if existing:
            raise CallrecError(
                f"A session already exists: {existing.name}. Run status, then stop to finish/recover it."
            )
        session = create_session(root, profile, "capture", engine)
        if context:
            atomic_json(session / "context.json", context)
            update_session(session, context_sha256=sha256(session / "context.json"))
        atomic_json(root / "active.json", {"id": session.name})
        try:
            with (session / "worker.log").open("ab", buffering=0) as log:
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "callrec.worker",
                        "--root",
                        str(root),
                        "--session",
                        str(session),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            update_session(session, worker_pid=p.pid)
        except BaseException:
            (root / "active.json").unlink()
            raise
    print(
        f"Session: {session.name}\nProfile: {profile}\nWaiting for microphone + system audio frames…",
        flush=True,
    )
    for _ in range(60):
        data = read_json(session / "session.json")
        if data["status"] == "recording":
            print(
                f"Recording mic + system audio ({data.get('audio_seconds', 0):.1f}s received).\nFiles: {session}\nRun `callrec stop` after your call."
            )
            print(
                "Frame receipt does not verify that both sources are audible; use headphones and check your first recording."
            )
            return 0
        if data["status"] == "capture_failed" or p.poll() is not None:
            raise CallrecError(
                f"Capture failed: {data.get('capture_error') or 'worker exited'}. Log: {session / 'capture.log'}\nRun `callrec stop --audio-only` to recover/reset this session."
            )
        time.sleep(0.25)
    print(
        "STARTING, not yet verified: no audio frames confirmed. Run `callrec status`.\n"
        "If macOS shows an access request, approve it. Otherwise capture may be stalled; "
        "no permission denial has been established.\n"
        "Run `callrec stop --audio-only` to cancel. Startup times out after 90 seconds."
    )
    return 2


def stop_worker(session):
    data = read_json(session / "session.json")
    if owned_worker(data):
        os.kill(data["worker_pid"], signal.SIGUSR1)
        for _ in range(160):
            data = read_json(session / "session.json")
            if data.get("worker_pid") is None and data["status"] not in ("starting", "recording"):
                return
            if not owned_worker(data):
                break
            time.sleep(0.25)
        if owned_worker(data):
            raise CallrecError(
                "Capture is still shutting down; try status then stop again. Files retained."
            )
    # A killed worker may have left a Hark process behind. Never signal an
    # unrelated/reused PID: verify both executable and exact recording args.
    pid = data.get("capture_pid")
    if isinstance(pid, int) and pid > 1:
        p = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        )
        expected = " ".join(data.get("capture_command", []))
        if expected and p.stdout.strip() == expected:
            os.kill(pid, signal.SIGINT)
            for _ in range(60):
                probe = subprocess.run(
                    ["/bin/ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
                )
                if probe.returncode or probe.stdout.strip().startswith("Z"):
                    break
                time.sleep(0.25)
            else:
                raise CallrecError(
                    "Orphan capture is still alive; refusing another recording. Audio retained."
                )
    if data["status"] in ("starting", "recording"):
        update_session(
            session,
            status="capture_failed",
            worker_pid=None,
            capture_error="Capture worker exited without finalizing session metadata; recovered audio may be partial.",
            capture_stopped_at=now(),
        )


def finish(root, session, audio_only=False):
    with lock(session / ".process.lock"):
        stop_worker(session)
        data = read_json(session / "session.json")
        chunks = list((session / "chunks").glob("*.wav"))
        if (
            data["kind"] == "capture"
            and not chunks
            and not (session / "source.wav").exists()
            and not data.get("audio_seconds", 0)
        ):
            update_session(
                session,
                status="capture_failed",
                empty_capture_closed_at=now(),
                capture_error=data.get("capture_error")
                or "Capture stopped before any audio was received.",
            )
            event(session, "empty_capture_closed", audio_saved=False)
            clear_active(root, session)
            print(
                f"Session: {session.name}\nNo audio was captured. Failed session closed; logs retained.\n"
                "You can start a new recording. Audio capture is not yet verified."
            )
            return
        source = prepare_audio(session)
        playback_audio(session, source)
        update_session(session, status="captured", audio_only=audio_only)
        clear_active(root, session)
        if not audio_only:
            result = enqueue(root, session)
            print(f"Session: {session.name}\nTranscription: {result} (background)")
        print(f"Audio saved: {session / 'audio.m4a'}")
        print(
            f"{'Transcribe later: callrec retry' if audio_only else 'Follow: callrec wait'} {session.name}"
        )


def selected_sessions(root, value=None):
    if value:
        return [resolve_session(root, value)]
    paths = list((root / "sessions").glob("*/session.json"))
    selected = {
        p.parent
        for p in paths
        if read_json(p)["status"] in ("queued", "transcribing", "starting", "recording")
    }
    active = active_session(root)
    if active:
        selected.add(active)
    ended = [
        p
        for p in paths
        if read_json(p)["status"] not in ("queued", "transcribing", "starting", "recording")
    ]
    if ended:
        selected.add(max(ended, key=lambda p: read_json(p).get("updated_at", "")).parent)
    return sorted(selected)


def render_status(item):
    heartbeat = (
        f"{item['heartbeat_age_seconds']:.1f}s ago"
        if item["heartbeat_age_seconds"] is not None
        else "not available"
    )
    print(
        f"Session: {item['session_id']}\nState: {item['state']}\n"
        f"Audio saved/received: {item['saved_audio_seconds']:.1f}s\n"
        f"Capture: {item['capture_health']}\nTranscription: {item['transcription_health']}\n"
        f"Progress: {item['processed_windows']}/{item['expected_windows'] or '?'} windows\n"
        f"Current window: {item['current_window']}\nHeartbeat: {heartbeat}\n"
        f"Warnings: {item['warning_count']}; review signals: {item['review_signal_count']}\n"
        f"Audio file: {item['paths']['audio.m4a'] if item['audio_saved'] else 'not created'}"
    )
    if item["last_error"]:
        print(f"Last error: {item['last_error']}")
    if item["retry_command"]:
        print(f"Resume: {item['retry_command']}")
    if item["legacy_metrics_incomplete"]:
        print("Legacy timing metrics are incomplete; total processing history is unavailable.")
    if item["state"] in ("complete", "complete_with_warnings"):
        print(
            f"Transcript: {item['paths']['transcript.md']}\nReview: {item['paths']['quality.md']}"
        )


def status(root, value=None, as_json=False):
    items = [snapshot(root, p) for p in selected_sessions(root, value)]
    if as_json:
        print(json.dumps({"sessions": items}, ensure_ascii=False, indent=2))
    elif not items:
        print("No sessions. Start with `callrec start --profile french` (or hindi).")
    else:
        for item in items:
            render_status(item)
    return (
        2
        if any(
            i["state"] in ("interrupted", "transcription_failed", "capture_failed")
            or "stalled" in i["transcription_health"]
            for i in items
        )
        else 0
    )


def wait(root, value=None):
    if value:
        session = resolve_session(root, value)
    else:
        candidates = selected_sessions(root)
        if not candidates:
            raise CallrecError("No session to wait for.")
        pending = [
            p
            for p in candidates
            if read_json(p / "session.json")["status"] in ("queued", "transcribing")
        ]
        session = max(
            pending or candidates, key=lambda p: read_json(p / "session.json")["created_at"]
        )
    previous = None
    while True:
        item = snapshot(root, session)
        if item["stored_state"] in ("starting", "recording"):
            raise CallrecError("Session is still recording. Run `callrec stop` after your call.")
        progress = (item["state"], item["processed_windows"], item["transcription_health"])
        if progress != previous:
            render_status(item)
            previous = progress
        if item["state"] not in ("queued", "transcribing"):
            return 0 if item["state"] in ("complete", "complete_with_warnings", "captured") else 2
        time.sleep(1)


def import_audio(root, file, profile, engine="whisper", context=None, parent=None, expected=None):
    context = load_context(context, engine)
    file = Path(file).expanduser().resolve()
    if not file.is_file():
        raise CallrecError(f"Audio file not found: {file}")
    duration(file)
    session = create_session(root, profile, "import", engine)
    if context:
        atomic_json(session / "context.json", context)
        update_session(session, context_sha256=sha256(session / "context.json"))
    suffix = file.suffix if file.suffix.isascii() and file.suffix[1:].isalnum() else ".audio"
    target = session / f"input{suffix}"
    with file.open("rb") as src, target.open("xb") as out:
        shutil.copyfileobj(src, out)
    if expected and sha256(target) != expected:
        update_session(
            session, status="import_failed", import_error="Source changed while copying."
        )
        raise CallrecError(
            "Source changed while copying; the original and partial import are retained."
        )
    update_session(
        session, import_file=target.name, import_sha256=sha256(target), original_path=str(file)
    )
    if parent:
        update_session(session, parent_session_id=parent.name)
    print(f"Imported session: {session.name}", flush=True)
    print(f"Transcription: {enqueue(root, session)}\nFollow: callrec wait {session.name}")


def parser():
    p = argparse.ArgumentParser(
        description="Record Mac calls; transcribe original languages entirely offline."
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--data-dir", help="Override the local Application Support directory.")
    commands = p.add_subparsers(dest="command", required=True)
    s = commands.add_parser("setup", help="Download pinned Hark and model once (requires network).")
    s.add_argument("--engine", choices=("auto", "qwen", "qwen-large", "whisper"), default="auto")
    s.add_argument(
        "--capture-only", action="store_true", help="Install recorder without downloading model."
    )
    commands.add_parser(
        "doctor", help="Check dependencies; does not record or request permissions."
    )
    s = commands.add_parser("start", help="Start detached microphone + system recording.")
    s.add_argument("--profile", choices=PROFILES, default="general")
    s.add_argument("--engine", choices=("auto", "qwen", "qwen-large", "whisper"), default="auto")
    s.add_argument("--capture-only", action="store_true", help="Allow capture before model setup.")
    s.add_argument("--context", help="Private JSON vocabulary file (Qwen only).")
    s = commands.add_parser("status", help="Show capture, processing and latest result; read-only.")
    s.add_argument("session", nargs="?")
    s.add_argument("--json", action="store_true")
    s = commands.add_parser("wait", help="Follow background processing; Ctrl-C stops waiting only.")
    s.add_argument("session", nargs="?")
    s = commands.add_parser("audit", help="Add review findings without changing the transcript.")
    s.add_argument("session")
    s.add_argument(
        "--retry-suspect", action="store_true", help="Generate bounded local Qwen alternatives."
    )
    s = commands.add_parser(
        "stop", help="Stop capture and transcribe; recover interrupted recordings."
    )
    s.add_argument("--audio-only", action="store_true", help="Save audio without transcription.")
    s = commands.add_parser("transcribe", help="Copy and transcribe an existing audio/video file.")
    s.add_argument("file")
    s.add_argument("--profile", choices=PROFILES, default="general")
    s.add_argument("--engine", choices=("auto", "qwen", "qwen-large", "whisper"), default="auto")
    s.add_argument("--context", help="Private JSON vocabulary file (Qwen only).")
    s = commands.add_parser("retry", help="Resume an incomplete session from its checkpoints.")
    s.add_argument("session", help="Session ID or full session directory.")
    s = commands.add_parser("reprocess", help="Create a new session from saved lossless audio.")
    s.add_argument("session")
    s.add_argument("--model", choices=("qwen-0.6b", "qwen-1.7b", "whisper-large-v3"), required=True)
    s.add_argument("--context", help="Private JSON vocabulary file (Qwen only).")
    commands.add_parser("list", help="List saved session IDs and processing status.")
    return p


def main(argv=None):
    os.umask(0o077)
    args = parser().parse_args(argv)
    root = root_path(args.data_dir)
    try:
        if args.command == "setup":
            setup(root, args.capture_only, args.engine)
        elif args.command == "doctor":
            return doctor(root)
        elif args.command == "start":
            return start(root, args.profile, args.capture_only, args.engine, args.context)
        elif args.command == "status":
            return status(root, args.session, args.json)
        elif args.command == "wait":
            return wait(root, args.session)
        elif args.command == "audit":
            audit_session(resolve_session(root, args.session), root, args.retry_suspect)
        elif args.command == "stop":
            init_root(root)
            with lock(root / ".control.lock"):
                session = active_session(root)
            if not session:
                raise CallrecError(
                    "No active recording. Use `callrec retry SESSION` for earlier recordings."
                )
            finish(root, session, args.audio_only)
        elif args.command == "transcribe":
            engine = (
                ("qwen" if args.profile == "hindi" else "whisper")
                if args.engine == "auto"
                else args.engine
            )
            model_path(root, engine)
            import_audio(root, args.file, args.profile, engine, args.context)
        elif args.command == "retry":
            session = resolve_session(root, args.session)
            data = read_json(session / "session.json")
            if owned_worker(data) or (
                active_session(root) == session and data["status"] in ("starting", "recording")
            ):
                raise CallrecError("This session is recording. Run `callrec stop` first.")
            print(
                f"Session: {session.name}\nTranscription: {enqueue(root, session)}\nFollow: callrec wait {session.name}"
            )
        elif args.command == "reprocess":
            session = resolve_session(root, args.session)
            data = read_json(session / "session.json")
            if data["status"] in ("starting", "recording", "queued", "transcribing"):
                raise CallrecError("Wait for the source session to finish before reprocessing.")
            source = session / "source.wav"
            expected = data.get("source_sha256")
            if not source.is_file() or not expected or sha256(source) != expected:
                raise CallrecError(
                    "Verified lossless source is unavailable or changed; original retained."
                )
            engine = {
                "qwen-0.6b": "qwen",
                "qwen-1.7b": "qwen-large",
                "whisper-large-v3": "whisper",
            }[args.model]
            model_path(root, engine)
            import_audio(
                root,
                source,
                data["profile"],
                engine,
                args.context,
                parent=session,
                expected=expected,
            )
        elif args.command == "list":
            for path in sorted((root / "sessions").glob("*/session.json"), reverse=True):
                data = read_json(path)
                print(f"{data['id']}  {data['profile']:7s}  {data['status']}")
        return 0
    except KeyboardInterrupt:
        print(
            "Stopped waiting; background processing continues."
            if args.command == "wait"
            else "Interrupted. Saved audio/checkpoints are retained; use `callrec retry SESSION`.",
            file=sys.stderr,
        )
        return 130
    except (CallrecError, OSError, ValueError, RuntimeError) as e:
        print(f"callrec: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
