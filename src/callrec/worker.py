"""Detached capture supervisor; no network or transcription work happens here."""

from __future__ import annotations

import argparse
import os
import selectors
import signal
import subprocess
import time
import wave
from pathlib import Path

from .common import (
    CHANNELS,
    CHUNK_SECONDS,
    FRAME_BYTES,
    RATE,
    CallrecError,
    check_space,
    lock,
    now,
    sha256,
    update_session,
)
from .setup import hark_path

WARMUP_DELAY = 3.0


def start_silent_output_probe(session, log):
    """Wake an idle system-output clock without playing an audible tone.

    This is a bounded diagnostic/workaround, not proof that either input is audible.
    """
    path = session / ".startup-silence.wav"
    with path.open("xb") as output:
        with wave.open(output, "wb") as audio:
            audio.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(48000 * 4 * 3))
    log.write(b"No startup PCM yet; trying a three-second silent system-output warmup.\n")
    update_session(session, capture_warmup_at=now(), capture_warmup_seconds=3)
    return subprocess.Popen(
        ["/usr/bin/afplay", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )


class ChunkWriter:
    """Incrementally patch WAV headers; a killed worker leaves playable PCM."""

    def __init__(self, folder, rate=RATE, channels=CHANNELS, seconds=CHUNK_SECONDS):
        self.folder = folder
        self.rate = rate
        self.channels = channels
        self.frame_bytes = channels * 2
        self.limit = rate * seconds * self.frame_bytes
        self.total = 0
        self.chunk_bytes = 0
        self.index = 0
        self.wave = None
        self.file = None

    def write(self, data):
        if len(data) % self.frame_bytes:
            raise CallrecError("Capture supplied a partial PCM frame.")
        while data:
            if self.wave is None:
                self.file = (self.folder / f"{self.index:06d}.wav").open("xb", buffering=0)
                self.wave = wave.open(self.file, "wb")
                self.wave.setparams((self.channels, 2, self.rate, 0, "NONE", "not compressed"))
                self.chunk_bytes = 0
            block, data = (
                data[: self.limit - self.chunk_bytes],
                data[self.limit - self.chunk_bytes :],
            )
            self.wave.writeframes(block)
            self.chunk_bytes += len(block)
            self.total += len(block)
            if self.chunk_bytes == self.limit:
                self.close_chunk()

    def sync(self):
        if self.file:
            os.fsync(self.file.fileno())

    def close_chunk(self):
        if self.wave:
            self.wave.close()
            os.fsync(self.file.fileno())
            self.file.close()
            self.wave = self.file = None
            self.index += 1

    def close(self):
        self.close_chunk()

    @property
    def duration(self):
        return self.total / self.frame_bytes / self.rate


def capture(session, root):
    stop = False
    requested = None

    def request(signum, frame):
        nonlocal stop, requested
        stop = True
        requested = signal.Signals(signum).name

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request)
    writer = ChunkWriter(session / "chunks")
    proc = None
    warmup = None
    warmup_attempted = False
    interruptions = []
    failure = None
    started = time.monotonic()
    last_frame = started
    last_update = 0
    stop_sent = None
    data_pending = b""
    first_frame_delay = None
    try:
        with lock(session / ".capture.lock"):
            update_session(session, worker_pid=os.getpid(), capture_started_at=now())
            check_space(session)
            args = [
                str(hark_path(root)),
                "--system",
                "--mix",
                "--capture-backend",
                "coreaudio",
                "--rate",
                str(RATE),
                "--bits",
                "16",
                "--channels",
                str(CHANNELS),
                "--keep-awake",
                "--no-translate",
                "--no-speakers",
                "--no-vad",
                "-a",
                "-",
                "--raw",
            ]
            env = {k: v for k, v in os.environ.items() if not k.startswith("HARK_")}
            with (session / "capture.log").open("ab", buffering=0) as log:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
                update_session(session, capture_pid=proc.pid, capture_command=args)
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                ended = False
                while not ended:
                    tick = time.monotonic()
                    if (
                        not stop
                        and not writer.total
                        and not warmup_attempted
                        and tick - started >= WARMUP_DELAY
                    ):
                        warmup_attempted = True
                        try:
                            warmup = start_silent_output_probe(session, log)
                        except OSError as e:
                            log.write(f"Silent output warmup unavailable: {e}\n".encode())
                    if stop and stop_sent is None:
                        if proc.poll() is None:
                            proc.send_signal(signal.SIGINT)
                        stop_sent = tick
                    if stop_sent is not None and tick - stop_sent > 12 and proc.poll() is None:
                        proc.kill()
                        failure = (
                            "Capture did not stop cleanly; completed WAV chunks were retained."
                        )
                    for key, _ in selector.select(timeout=0.25):
                        block = os.read(key.fd, 65536)
                        if not block:
                            ended = True
                            break
                        gap = tick - last_frame
                        if writer.total and gap > 2:
                            interruptions.append(
                                {
                                    "at_audio_seconds": round(writer.duration, 3),
                                    "wall_gap_seconds": round(gap, 3),
                                    "reason": "No PCM frames received",
                                }
                            )
                        last_frame = tick
                        data_pending += block
                        size = len(data_pending) // FRAME_BYTES * FRAME_BYTES
                        if size and first_frame_delay is None:
                            first_frame_delay = round(time.monotonic() - started, 3)
                            update_session(
                                session,
                                first_frame_at=now(),
                                startup_delay_seconds=first_frame_delay,
                            )
                        writer.write(data_pending[:size])
                        data_pending = data_pending[size:]
                    if tick - last_update >= 1:
                        writer.sync()
                        check_space(session)
                        elapsed = tick - started
                        update_session(
                            session,
                            status="recording" if writer.total else "starting",
                            audio_seconds=round(writer.duration, 3),
                            heartbeat_at=now(),
                            capture_wall_seconds=round(elapsed, 3),
                            seconds_since_frames=round(tick - last_frame, 3),
                            interruptions=interruptions,
                        )
                        last_update = tick
                    threshold = 90 if not writer.total else 20
                    if tick - last_frame > threshold:
                        raise CallrecError(
                            "Recorder produced no audio frames before the capture deadline. "
                            "Capture may be stalled or access unavailable; inspect capture.log. "
                            "This does not establish a permission denial."
                        )
                selector.close()
                code = proc.wait(timeout=15)
                if code != 0 or (not stop and code == 0):
                    failure = f"Capture exited unexpectedly (code {code}); inspect capture.log."
                if data_pending:
                    failure = (
                        "Capture ended with an incomplete PCM frame. Completed audio is retained."
                    )
    except BaseException as e:
        failure = f"{type(e).__name__}: {e}"
    finally:
        if warmup:
            if warmup.poll() is None:
                warmup.terminate()
            try:
                warmup.wait(timeout=3)
            except subprocess.TimeoutExpired:
                warmup.kill()
                warmup.wait()
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc and proc.stdout:
            proc.stdout.close()
        writer.close()
        if requested and requested != "SIGUSR1":
            failure = failure or f"Recording interrupted by {requested}."
        chunks = [
            {"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted((session / "chunks").glob("*.wav"))
        ]
        update_session(
            session,
            status="capture_failed" if failure or not chunks else "captured",
            capture_error=failure,
            capture_stopped_at=now(),
            worker_pid=None,
            audio_seconds=round(writer.duration, 3),
            capture_chunks=chunks,
            interruptions=interruptions,
            capture_wall_seconds=round(time.monotonic() - started, 3),
        )


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--session", required=True)
    args = p.parse_args()
    capture(Path(args.session), Path(args.root))


if __name__ == "__main__":
    main()
