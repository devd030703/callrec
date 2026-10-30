from __future__ import annotations

import os
import platform
import shutil
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import (
    HARK_SHA256,
    HARK_URL,
    HARK_VERSION,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    CallrecError,
    atomic_json,
    init_root,
    lock,
    model_details,
    model_path,
    read_json,
    run,
    sha256,
    tool,
)


def hark_path(root):
    path = root / "tools" / f"hark-{HARK_VERSION}" / "hark"
    manifest = read_json(path.parent / "callrec-release.json", {})
    if (
        not path.is_file()
        or manifest.get("archive_sha256") != HARK_SHA256
        or manifest.get("binary_sha256") != sha256(path)
    ):
        raise CallrecError("Pinned Hark is missing or changed. Run `callrec setup`.")
    return path


def install_hark(root):
    try:
        return hark_path(root)
    except CallrecError:
        pass
    target = root / "tools" / f"hark-{HARK_VERSION}"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "tools") as tmp:
        archive = Path(tmp) / "hark.tar.gz"
        print(f"Downloading Hark {HARK_VERSION} (checksum verified)…", flush=True)
        urllib.request.urlretrieve(HARK_URL, archive)
        if sha256(archive) != HARK_SHA256:
            raise CallrecError("Hark download checksum mismatch; refusing to install.")
        with tarfile.open(archive) as t:
            for name in ("hark", "LICENSE", "NOTICES", "README.md", "CHANGELOG.md"):
                member = t.getmember(f"hark-{HARK_VERSION}/{name}")
                if not member.isfile():
                    raise CallrecError("Unexpected Hark archive member.")
                source = t.extractfile(member)
                with (target / name).open("wb") as out:
                    shutil.copyfileobj(source, out)
        (target / "hark").chmod(0o755)
        version = run([target / "hark", "--version"], text=True).stdout.strip()
        if version != HARK_VERSION:
            raise CallrecError(f"Unexpected Hark version {version}")
        atomic_json(
            target / "callrec-release.json",
            {
                "version": version,
                "url": HARK_URL,
                "archive_sha256": HARK_SHA256,
                "binary_sha256": sha256(target / "hark"),
            },
        )
    return hark_path(root)


def download_model_file(target, name, size, *, repo=None, revision=None, expected=None):
    """Bounded HTTP ranges avoid one fragile multi-gigabyte transfer."""
    final = target / name
    expected = expected or MODEL_SHA256[name]
    repo = repo or MODEL_REPO
    revision = revision or MODEL_REVISION
    if final.is_file() and sha256(final) == expected:
        print(f"Verified existing {name}.", flush=True)
        return
    partial = target / f"{name}.download"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise CallrecError(f"Oversized partial download: {partial}. Retained for inspection.")

    def fetch_range(begin, end):
        for attempt in range(4):
            try:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true&offset={begin}&t={time.time_ns()}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    content_range = response.headers.get("Content-Range")
                    if response.status == 206:
                        if content_range != f"bytes {begin}-{end}/{size}":
                            raise CallrecError("Model server returned an unexpected byte range.")
                    elif not (response.status == 200 and begin == 0 and end == size - 1):
                        raise CallrecError(
                            "Model server did not honor the resumable download range."
                        )
                    payload = response.read(end - begin + 1)
                    if len(payload) != end - begin + 1:
                        raise OSError("Model response ended early; partial data retained.")
                    return payload
            except (OSError, TimeoutError) as e:
                if attempt == 3:
                    raise CallrecError(
                        f"Download interrupted; run setup again to resume {name}: {type(e).__name__}"
                    ) from e
                print(f"Retrying {name} at {begin / 1024**2:.0f} MiB…", flush=True)
        raise CallrecError("Download retry limit reached.")

    # Keep each response small and bounded. Eight concurrent one-MiB ranges
    # avoid the long-response stalls observed on this Mac's download path.
    # Append only in order, so any interruption leaves a resumable prefix.
    with ThreadPoolExecutor(max_workers=8) as pool:
        while offset < size:
            ranges = []
            begin = offset
            for _ in range(8):
                if begin >= size:
                    break
                end = min(size - 1, begin + 1024 * 1024 - 1)
                ranges.append((begin, end))
                begin = end + 1
            futures = [pool.submit(fetch_range, begin, end) for begin, end in ranges]
            for future in futures:
                payload = future.result()
                with partial.open("ab") as out:
                    out.write(payload)
                    out.flush()
                    os.fsync(out.fileno())
                offset += len(payload)
            print(
                f"{name}: {offset / size:.1%} ({offset / 1024**2:.0f}/{size / 1024**2:.0f} MiB)",
                flush=True,
            )
    if sha256(partial) != expected:
        raise CallrecError(f"Model checksum mismatch. Retained {partial}; refusing to load it.")
    os.replace(partial, final)


def setup(root, capture_only=False, engine="auto"):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise CallrecError("This version requires an Apple Silicon Mac.")
    init_root(root)
    with lock(root / ".setup.lock"):
        tool("ffmpeg")
        tool("ffprobe")
        install_hark(root)
        if not capture_only:
            for selected in ("whisper", "qwen") if engine == "auto" else (engine,):
                install_model(root, selected)
    print("Setup complete. Capture permissions are requested on first start.")
    print(
        "Microphone and System Audio Recording may need enabling in System Settings → Privacy & Security."
    )
    print("Permissions are NOT verified by setup; start waits for actual audio frames.")


def install_model(root, engine):
    details = model_details(engine)
    target = root / "models" / details["directory"]
    target.mkdir(exist_ok=True)
    print(f"Installing local {details['repo']} (no API/subscription)…", flush=True)
    for name, file in details["files"].items():
        download_model_file(
            target,
            name,
            file["bytes"],
            repo=details["repo"],
            revision=details["revision"],
            expected=file["sha256"],
        )
    actual = {n: sha256(target / n) for n in details["files"]}
    if actual != {n: f["sha256"] for n, f in details["files"].items()}:
        raise CallrecError("Model checksum mismatch; refusing to mark it ready.")
    atomic_json(
        target / "callrec-model.json",
        {"repo": details["repo"], "revision": details["revision"], "files": actual},
    )


def doctor(root):
    print(f"Data: {root}\nMac: {platform.mac_ver()[0]} / {platform.machine()}")
    checks = [
        ("ffmpeg", lambda: tool("ffmpeg")),
        ("ffprobe", lambda: tool("ffprobe")),
        ("Hark (version and binary checksum)", lambda: hark_path(root)),
        ("Hindi model (Qwen3-ASR)", lambda: model_path(root, "qwen")),
        ("French/general model (Whisper)", lambda: model_path(root, "whisper")),
    ]
    okay = True
    for name, check in checks:
        try:
            print(f"OK {name}: {check()}")
        except CallrecError as e:
            okay = False
            print(f"MISSING {name}: {e}")
    print(
        "Capture permissions: unverified until audio frames arrive. Both-source audibility needs a listening test."
    )
    return 0 if okay else 1
