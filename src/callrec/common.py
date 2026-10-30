from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

HARK_VERSION = "0.4.3"
HARK_SHA256 = "5301df867b44f68f686a4b9230bfe7b9d9233f0091054fde1ac7560639e7fcae"
HARK_URL = f"https://github.com/PhantomYdn/hark/releases/download/v{HARK_VERSION}/hark-{HARK_VERSION}-macos-arm64.tar.gz"
MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
MODEL_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
MODEL_SHA256 = {
    "config.json": "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3",
    "weights.npz": "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d",
}
PROFILES = {
    "general": None,
    "hindi": "हिन्दी और English. आज हम हिन्दी सीख रहे हैं. Hindi and English conversation.",
    "french": "Français et English. Aujourd’hui, nous étudions le français. French and English conversation.",
}
RATE = 16000
CHANNELS = 2
FRAME_BYTES = CHANNELS * 2
CHUNK_SECONDS = 60


class CallrecError(Exception):
    pass


def root_path(value=None):
    return (
        Path(value).expanduser().resolve()
        if value
        else Path.home() / "Library/Application Support/Callrec"
    )


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def init_root(root):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("sessions", "models", "tools"):
        (root / name).mkdir(exist_ok=True, mode=0o700)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def atomic_text(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@contextlib.contextmanager
def lock(path, blocking=False):
    path = Path(path)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as e:
            raise CallrecError(
                "Another callrec operation is using this session; try status."
            ) from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update_session(session, **fields):
    with lock(session / ".metadata.lock", blocking=True):
        data = read_json(session / "session.json")
        data.update(fields, updated_at=now())
        atomic_json(session / "session.json", data)
    return data


def create_session(root, profile, kind, engine="whisper"):
    init_root(root)
    sid = dt.datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S") + "-" + uuid.uuid4().hex[:8]
    session = root / "sessions" / sid
    session.mkdir(mode=0o700)
    for folder in ("chunks", "checkpoints"):
        (session / folder).mkdir(mode=0o700)
    atomic_json(
        session / "session.json",
        {
            "schema_version": 1,
            "id": sid,
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="microseconds"),
            "profile": profile,
            "kind": kind,
            "engine": engine,
            "status": "starting" if kind == "capture" else "imported",
            "interruptions": [],
            "quality_verified": False,
            "warning": "Machine transcript: fidelity and mixed-script accuracy have not been human-verified.",
        },
    )
    return session


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tool(name):
    result = shutil.which(name)
    if result is None:
        raise CallrecError(f"Missing {name}. Install ffmpeg with `brew install ffmpeg`.")
    return result


def run(args, **kwargs):
    p = subprocess.run([str(x) for x in args], capture_output=True, **kwargs)
    if p.returncode:
        error = p.stderr.decode(errors="replace") if isinstance(p.stderr, bytes) else p.stderr
        raise CallrecError(f"{Path(str(args[0])).name} failed: {error[-1800:]}")
    return p


def duration(path):
    value = run(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).stdout.strip()
    try:
        result = float(value)
        if not 0 < result < 7 * 24 * 3600:
            raise ValueError(value)
        return result
    except ValueError as e:
        raise CallrecError(f"Invalid or empty audio: {path}") from e


def owned_worker(data):
    pid = data.get("worker_pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    p = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    )
    return p.returncode == 0 and "callrec.worker" in p.stdout and data["id"] in p.stdout


def active_session(root):
    active = read_json(root / "active.json")
    if not active:
        return None
    sid = active.get("id", "")
    if not sid or Path(sid).name != sid:
        raise CallrecError("Invalid active-session pointer.")
    session = root / "sessions" / sid
    if not (session / "session.json").is_file():
        raise CallrecError("Active-session metadata is missing; recordings were not deleted.")
    return session


def clear_active(root, session):
    with lock(root / ".control.lock", blocking=True):
        if active_session(root) == session:
            (root / "active.json").unlink()


def resolve_session(root, value):
    path = Path(value).expanduser()
    if not path.is_dir():
        path = root / "sessions" / value
    path = path.resolve()
    if path.parent != (root / "sessions").resolve() or not (path / "session.json").is_file():
        raise CallrecError("Choose a session ID or session directory shown by `callrec list`.")
    return path


def check_space(path, minimum=256 * 1024 * 1024):
    if shutil.disk_usage(path).free < minimum:
        raise CallrecError("Less than 256 MiB free. Recording stopped; existing audio is retained.")


def model_details(engine):
    if engine == "whisper":
        return {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "directory": "whisper-large-v3",
            "files": {
                n: {"sha256": h, "bytes": 269 if n == "config.json" else 3083520416}
                for n, h in MODEL_SHA256.items()
            },
        }
    if engine == "qwen":
        return read_json(Path(__file__).with_name("qwen-manifest.json"))
    raise CallrecError(f"Unknown transcription engine: {engine}")


def model_path(root, engine="whisper"):
    details = model_details(engine)
    path = root / "models" / details["directory"]
    manifest = read_json(path / "callrec-model.json", {})
    if manifest.get("revision") != details["revision"] or not all(
        (path / n).is_file() for n in details["files"]
    ):
        raise CallrecError(
            f"Local {engine} model is not ready. Run `callrec setup --engine {engine}` once while online."
        )
    return path
